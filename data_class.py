"""
Class that contains scattering data
"""
import numpy as np
import pandas as pd

class ScatterData:
    def __init__(self, file=None, sep=','):
        if file is None:
            self.q = np.zeros(1)
            self.i = np.zeros(1)
            self.e = []
        else:
            self.parse(file, sep)

    def __str__(self):
        return f'Scatter intensity with {self.i.shape[0]} points \n q-range varying from {np.nanmin(self.q)} to {self.q[-1]}'

    def parse(self, file, delim):
        """
        Parses scattering data into numpy arrays and
        stores within class object.

        Parameters
        ----------
        file : CSV file containing columns:
            q (scattering vector),
            I (scattering intensity),
            e (experimental error, optional)
        delim : delimiter
        """

        df = pd.read_csv(file, names=['q', 'I', 'e'], index_col=False, delimiter=delim)

        self.q = df['q'].to_numpy()
        self.i = df['I'].to_numpy()

        # If 'e' column exists and is not all NaN, use it; otherwise compute sqrt(I)
        if 'e' in df.columns and not df['e'].isnull().all():
            self.e = df['e'].to_numpy()
        else:
            self.e = np.ones(self.i.shape[0])

    def set_data(self, q, i, e=None):
        """
        Sets scattering data manually from arrays or
        likewise instead of file

        Parameters
        ----------
        q : np.array
        i : np.array
        e : np.array (optional)
        """
        if e is None:
            e = []
        self.q = q
        self.i = i
        self.e = e
        return

    def cut_q(self, q_min, q_max):
        """
        Adjust current q-range of scattering data
        to a new desired range
        Parameters
        ----------
        q_min : int
        q_max : int

        Returns
        -------
        A new ScatteringData object with new intensity, error and
        q-range adjusted to the newly set q-range.
        """
        indices = np.logical_and(q_min <= self.q, self.q <= q_max)
        q = self.q[indices]
        i = self.i[indices]
        if self.is_error():
            error = self.e[indices]
        else:
            error = []
        copy = ScatterData()
        copy.set_data(q, i, error)
        return copy

    def scale_intensity(self, k):
        """
        Scales the intensity of the data by a factor k
        Parameters
        ----------
        k : int

        Returns
        -------
        A new ScatterData object with scaled intensity

        """
        copy = self.copy()
        scaled_intensity = k * self.i
        copy.set_data(self.q, scaled_intensity, self.e)
        return copy

    def scale_q(self, k):
        """
        Scales the q-vector of the data by a factor k
        (Used to make data comparable between Å⁻¹ and nm⁻¹)
        Parameters
        ----------
        k : int

        Returns
        -------
         A new ScatterData object with scaled q
        """
        copy = self.copy()
        scaled_q = k * self.q
        copy.set_data(scaled_q, self.i, self.e)
        return copy

    def is_error(self):
        """
        Help function to see if errors are empty
        Returns

        -------
        Boolean value corresponding to state.
        """
        if type(self.e) == list:
            return False
        elif self.e.shape == self.i.shape:
            return True
        else:
            return False


    def copy(self):
        """
        Creates a new blank ScatterData objects
        (Used to return a new instance, to no overwrite current object)
        Returns
        -------
        A new ScatterData object
        """
        copy = ScatterData()
        return copy

    def remove_nan(self):
        """
        Removes NaN from intensity and removes those
        intensities and corresponding q-values.
        Returns
        -------
        A new ScatterData object without the NaN intensities
        """
        indices = np.where(self.i == self.i)
        i = self.i[indices]
        q = self.q[indices]
        if self.is_error():
            error = self.error[indices]
        else:
            error = []
        copy = self.copy()
        copy.set_data(q, i, error)
        return copy

    def normalize(self, q_range=False):
        copy = self.copy()
        copy.q = self.q
        if q_range:
            limits = np.logical_and(copy.q > q_range[0], copy.q < q_range[1])
            copy.i = self.i / np.nanmean(self.i[limits])
            copy.error = copy.i
        else:
            copy.i = self.i / np.nansum(self.i)
            copy.error = copy.i
        return copy

