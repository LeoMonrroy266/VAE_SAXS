# -*- coding: utf-8 -*-
"""
map2iq.py
------------------------------------------------------------------
Reads and parses SAXS data
Calculates SAXS profiles from voxel objects.
FFT-based I(q) calculation + scoring
"""


import numpy as np
from pathlib import Path
from dataclasses import dataclass
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy import optimize
from region_search import find_largest_connected_region

try:
    from scipy import ndimage  # optional (used only for gaussian_filter)
    _have_scipy = True
except ModuleNotFoundError:
    _have_scipy = False


@dataclass
class IqData:
    q: np.ndarray
    i: np.ndarray
    s: np.ndarray = None

def read_iq_ascii(fname: str | Path) -> IqData:
    """Read three-column ASCII: q  I(q)  sigma (σ optional)."""
    arr = np.loadtxt(fname)
    if arr.shape[1] == 2:
        q, i = arr.T
        s = np.ones_like(i)
    elif arr.shape[1] >= 3:
        q, i, s = arr[:, 0], arr[:, 1], arr[:, 2]
    else:
        raise ValueError("Expect 2 or 3 columns: q  I  [σ]")
    return IqData(q=q, i=i, s=s)


# ─────────────────────── FFT → radial average (q in Å⁻¹) ──────────────────────
def fft_intensity_scale_factor(voxel: np.ndarray, voxel_size: float = 3.33) -> tuple[np.ndarray, np.ndarray]:
    """
        Compute spherically averaged scattering intensity I(q) from a 3D real voxel grid.

        Parameters
        ----------
        voxel      : 3D ndarray
        voxel_size : float, size of each voxel in Å

        Returns
        -------
        q  : 1D array of q-values in Å⁻¹
        Iq : 1D array of intensity values
        """
    Fq = np.fft.fftn(voxel)

    I3d = np.abs(Fq) ** 2
    N = voxel.shape[0]

    freqs = np.fft.fftfreq(N, d=voxel_size)
    kx, ky, kz = np.meshgrid(freqs, freqs, freqs, indexing="ij")
    qgrid = 2 * np.pi * np.sqrt(kx ** 2 + ky ** 2 + kz ** 2)

    q_flat = qgrid.ravel()
    I_flat = I3d.ravel()

    dq = 0.001
    nbins = int(np.ceil(q_flat.max() / dq))
    bins = np.linspace(0, q_flat.max(), nbins + 1)

    idx = np.clip(np.digitize(q_flat, bins) - 1, 0, nbins - 1)
    I_sum = np.bincount(idx, weights=I_flat, minlength=nbins)
    N_sum = np.bincount(idx, minlength=nbins)

    valid = N_sum > 0
    I_rad = I_sum[valid] / N_sum[valid]

    q_mid = 0.5 * (bins[1:] + bins[:-1])
    q_rad = q_mid[valid]

    return q_rad, I_rad

def match_shorter_q(q1, I1, q2, I2):
    """
    Interpolate the curve with the larger q-range onto the one
    with the smaller q-range (based on q_max).

    Parameters
    ----------
    q1, I1 : np.ndarray
        q-values and intensities for the first curve.
    q2, I2 : np.ndarray
        q-values and intensities for the second curve.

    Returns
    -------
    q_common : np.ndarray
        q-grid of the curve with the smaller q_max.
    I1_interp : np.ndarray
        I1 values interpolated to q_common (if needed).
    I2_interp : np.ndarray
        I2 values interpolated to q_common (if needed).
    """
    # Sanity check and correction
    q1, I1 = np.array(q1), np.array(I1)
    q2, I2 = np.array(q2), np.array(I2)
    min_len1 = min(len(q1), len(I1))
    min_len2 = min(len(q2), len(I2))
    q1, I1 = q1[:min_len1], I1[:min_len1]
    q2, I2 = q2[:min_len2], I2[:min_len2]
    #print(f"q1 range: {q1.min():.4f}–{q1.max():.4f} (len={len(q1)})")
    #print(f"q2 range: {q2.min():.4f}–{q2.max():.4f} (len={len(q2)})")

    assert len(q1) == len(I1), f"Mismatch within q1/I1 ({len(q1)} vs {len(I1)})"
    assert len(q2) == len(I2), f"Mismatch within q2/I2 ({len(q2)} vs {len(I2)})"

    # Decide which q-grid is shorter in range
    if q1.max() <= q2.max():
        q_common = q1
        I1_interp = I1
        I2_interp = np.interp(q_common, q2, I2)
    else:
        q_common = q2
        I2_interp = I2
        I1_interp = np.interp(q_common, q1, I1)

    return q_common, I1_interp, I2_interp




def best_scaling_factor(I_model, target, err=None):
    #calc_factor = lambda x: 1 - r2_score(target, I_model * x)
    #alpha = optimize.minimize_scalar(calc_factor, bounds=(0, 1), options={'disp': False}, method='bounded').x
    alpha = np.dot(target, I_model) / np.dot(I_model, I_model)
    if alpha < 0:
        alpha = 1
    return alpha
# ───────────────────────── ED_map class ────────────────────────────
class ED_map:
    """
    • compute_saxs_profile(): FFT → radial average
    • target(): χ²-like deviation metric
    """
    def __init__(self,
                 iq_data: IqData,
                 ref_voxel,
                 test_scale,
                 ref_scale: float,
                 voxel_size: float = 3.33,
                 qmax: float = 0.6
                 ):

        self.exp  = iq_data
        self.voxel_size = voxel_size
        # keep only q ≤ qmax
        self.exp_q = iq_data.q
        sel = iq_data.q <= qmax
        self.exp_q = iq_data.q[sel]
        self.exp_I = iq_data.i[sel]
        self.exp_e = iq_data.e[sel]
        # Values from ground state structure
        self.dark = ref_voxel
        self.scale = ref_scale
        self.test_scale = test_scale


    # ------------- public API ------------------
    def compute_saxs_profile(self, voxel: np.ndarray,
                             filename: str = "diff_saxs_profile.png") -> np.ndarray:
        """
        Calculates the difference SAXS profile of generated voxel object and assumed starting structure
        (generated voxel - starting structure).

        Parameters
        ----------
        voxel : np.ndarray
        filename : str, optional

        Returns
        -------
        Difference SAXS curve interpolated to experimental q-range
        """
        # Corrected scattering for when transforming structure into voxel
        voxel_size_ground= 100 / (voxel.shape[0] * self.scale)
        voxel_size_active = 100 / (voxel.shape[0] * self.test_scale)




        dark_q, dark_I = fft_intensity_scale_factor(self.dark, voxel_size_ground)  # calc scatter for ground state
        q, Iq = fft_intensity_scale_factor(voxel, voxel_size_active)

        # Normalize
        dark_I = dark_I / dark_I[0]
        Iq = Iq / Iq[0]

        # Interpolate to get same q-spacing independent of voxelsize
        common_q, dark_I, Iq = match_shorter_q(dark_q, dark_I ,q ,Iq)



        # Calculate difference scattering
        delta_Iq = Iq - dark_I
        # Interpolate theoretical diff on experimental q scale
        delta_Iq_interpolate = np.interp(self.exp_q, common_q, delta_Iq)

        #scale = best_scaling_factor(delta_Iq_interpolate, self.exp_I)
        """
        plt.figure(figsize=(8, 5))
        plt.plot(self.exp_q, delta_Iq_interpolate * scale, label=r'Model $\Delta$I(q)')

        plt.plot(self.exp_q, self.exp_I , label=r'Exp. $\Delta$I(q)')
        plt.xlabel(r'q (Å$^{-1}$)')
        plt.ylabel(r'Intensity $\Delta$I(q)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        """
        return delta_Iq_interpolate


    def target(self, voxel) -> float:
        """
        Calculate the R² and Chi² between computed and experimental difference curves.
        Parameters
        ----------
        voxel : ndarray

        Returns
        -------
        r2 : float
        """
        calc = self.compute_saxs_profile(voxel)

        if np.all(calc == 0):
            calc = np.ones(calc.shape[0])

        scale = best_scaling_factor(calc, self.exp_I)

        # Apply least-squares scaling
        # Score depending on availability of exp_s
        if self.exp_e is None:
            # Use plain L2 distance
            chi = np.linalg.norm(self.exp_I-(calc*scale))
            nan_indices = np.where(np.isnan(self.exp_I))[0]
            print(f"NaNs found at indices: {nan_indices}")

            r2 = r2_score(self.exp_I,calc*scale)
        else:
            # Use chi-squared
            chi = np.linalg.norm(self.exp_I-(scale*calc))
            r2 = r2_score(self.exp_I,calc*scale)
        #print('R²:', r2, 'chi²:', chi)
        return r2,self.test_scale, scale
# ─────────────────── wrappers for GA code ────────────────────
def run(input_data,
        iq_file: IqData,
        ground_state_voxel,
        ground_state_scale: float,
        voxel_size: float = 3.33):

    # Last element is rmax, the rest is flattened voxel
    data = iq_file
    extrapolated_voxel,s_factor = input_data
    #test_scales = [s * s_factor for s in np.arange(.95, 1.1, .05)]
    #test_scales = [s * s_factor for s in np.arange(.75, 1.3, .05)]
    test_scales = [ground_state_scale] * 3
    current_d = -np.inf
    current_s = 1
    current_scale = 1
    for s_test in test_scales:
        d,s_factor,scale = ED_map(data, voxel_size=voxel_size, ref_voxel=ground_state_voxel, ref_scale=ground_state_scale, test_scale=s_test).target(extrapolated_voxel)

        if current_d < d:
            current_d = d
            current_s = s_factor
            current_scale = scale

    return [current_d, float(current_s), float(current_scale)]
    #return current_d

