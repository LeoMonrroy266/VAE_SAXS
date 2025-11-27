import gemmi
import matplotlib.pyplot as plt
import numpy as np
import os

def create_ccp4_map_(density_array, output_map_file, voxel_size, scaling_factor_file):
    """
    Create a CCP4 map in real Ångström space, using a stored scaling factor
    to reverse any previous normalization.
    """
    # Step 1: Load scaling factor
    scaling_factor = load_scaling_factor(scaling_factor_file)
    print(f"Loaded scaling factor")

    # Step 2: Get grid shape
    nx, ny, nz = density_array.shape
    print(f"Grid shape: {nx, ny, nz}")

    # Step 3: Compute scaled voxel size in Å
    voxel_size_real = voxel_size / scaling_factor
    print(f"Voxel size (Å): {voxel_size_real}")

    # Step 4: Compute real box dimensions
    box_x = nx * voxel_size_real
    box_y = ny * voxel_size_real
    box_z = nz * voxel_size_real
    print(f"Real box size (Å): {box_x, box_y, box_z}")

    
    # Step 5: Create grid
    grid = gemmi.FloatGrid(nx, ny, nz)
    grid.set_unit_cell(gemmi.UnitCell(box_x, box_y, box_z, 90, 90, 90))
    grid.spacegroup = gemmi.SpaceGroup("P 1")

 # Compute real-space origin shift so center is at (0,0,0)
    origin_shift = np.array([box_x/2, box_y/2, box_z/2])

    # Fill the grid
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # Position in real space
                x = i * voxel_size - origin_shift[0]
                y = j * voxel_size - origin_shift[1]
                z = k * voxel_size - origin_shift[2]
                # No need to convert to fractional coords for gemmi; set_value uses grid indices
                grid.set_value(i, j, k, float(density_array[i,j,k]))

    # Step 7: Write CCP4 map
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = grid
    ccp4.update_ccp4_header()
    ccp4.write_ccp4_map(output_map_file)

    print(f"✅ CCP4 map saved to: {output_map_file}")
    print(f"Unit cell (Å): {grid.unit_cell}")


def load_scaling_factor(scaling_factor_source):
    # If it's a string and points to a file, load from file
    if isinstance(scaling_factor_source, str) and os.path.isfile(scaling_factor_source):
        scaling_factor = float(np.load(scaling_factor_source))
    # If it's already a NumPy array
    elif isinstance(scaling_factor_source, np.float64):
        scaling_factor = float(scaling_factor_source)
    else:
        raise ValueError("scaling_factor_source must be a valid file path or a NumPy array.")

    print(f"Loaded scaling factor: {scaling_factor}")
    return scaling_factor


def normalize_minus1_1(arr):
    """Normalize an array to the range [-1, 1]."""
    arr_min = arr.min()
    arr_max = arr.max()
    if arr_max - arr_min == 0:
        return np.zeros_like(arr, dtype=np.float32)
    return 2 * (arr - arr_min) / (arr_max - arr_min) - 1


def save_2d_slices(diff_array, output_folder, name_prefix="diff"):
    """
    Save 6 2D slices from the middle ± quarter of the 3D difference array:
    XY, XZ, YZ planes at two positions along each axis.
    """
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    nx, ny, nz = diff_array.shape

    # Middle indices along each axis
    mid_x = nx // 2
    mid_y = ny // 2
    mid_z = nz // 2

    # Offsets for additional slices (middle ± quarter)
    offset_x = nx // 4
    offset_y = ny // 4
    offset_z = nz // 4

    slices = [
        ("XY_midZ", diff_array[:, :, mid_z]),
        ("XY_midZ_plus", diff_array[:, :, min(mid_z + offset_z, nz-1)]),
        ("XZ_midY", diff_array[:, mid_y, :]),
        ("XZ_midY_plus", diff_array[:, min(mid_y + offset_y, ny-1), :]),
        ("YZ_midX", diff_array[mid_x, :, :]),
        ("YZ_midX_plus", diff_array[min(mid_x + offset_x, nx-1), :, :])
    ]

    for suffix, slice_2d in slices:
        plt.figure()
        plt.imshow(slice_2d, cmap="viridis")
        plt.colorbar()
        plt.title(f"{name_prefix}_{suffix}")
        plt.savefig(Path(output_folder) / f"{name_prefix}_{suffix}.png")
        plt.close()



def plot_3d_heatmap(grid, threshold=0.0, colormap='viridis'):
    """
    Plot a 3D heatmap of a 3D grid using matplotlib.

    Parameters:
    - grid: 3D numpy array of values
    - threshold: minimum absolute value to plot (for visibility)
    - colormap: matplotlib colormap to use
    """
    nx, ny, nz = grid.shape
    x, y, z = np.indices(grid.shape)

    # Flatten arrays
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    values = grid.flatten()

    # Filter by threshold to reduce number of points plotted
    mask = np.abs(values) > threshold
    x = x[mask]
    y = y[mask]
    z = z[mask]
    values = values[mask]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot in 3D with color mapped to values
    p = ax.scatter(x, y, z, c=values, cmap=colormap, marker='o', s=20)
    fig.colorbar(p, ax=ax, label='Density')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Heatmap of Grid')
    plt.show()

def plot_2d_projections(grid, output_folder="projections", name_prefix="grid"):
    """
    Create 2D projection heatmaps of a 3D grid along X, Y, Z axes.
    
    Parameters:
    - grid: 3D numpy array
    - output_folder: folder to save images
    - name_prefix: prefix for saved image files
    """
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Project along axes (sum over the axis)
    proj_xy = np.sum(grid, axis=2)  # projection along Z, top-down view
    proj_xz = np.sum(grid, axis=1)  # projection along Y, front view
    proj_yz = np.sum(grid, axis=0)  # projection along X, side view
    
    projections = [
        ("XY_projection", proj_xy),
        ("XZ_projection", proj_xz),
        ("YZ_projection", proj_yz)
    ]
    
    for suffix, proj in projections:
        plt.figure(figsize=(6,5))
        plt.imshow(proj, cmap="seismic", origin="lower")
        plt.colorbar(label="Density")
        plt.title(f"{name_prefix}_{suffix}")
        plt.savefig(Path(output_folder) / f"{name_prefix}_{suffix}.png")
        plt.close()
        print(f"Saved {suffix} to {output_folder}")