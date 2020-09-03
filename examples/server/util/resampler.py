from scipy import interpolate
import numpy as np

class Resampler:
    """Generates interpolation on new 1d grid with extrapolation
    x_old - ndarray, (1,n)
    y_old - ndarray , (m,n)

    """
    def __init__(self, x_old, y_old,  extrapolate='extrapolate'):
        self.x_old = x_old
        self.y_old = y_old
        self.extrapolate = extrapolate
        self.f = interpolate.interp1d(x_old, y_old, fill_value=self.extrapolate)

    def resample(self, x_new):
        """ x_new - ndarray, (1,N_new)
        returns -  ndarray, (m,N_new), linear interpolated values """
        return self.f(x_new)
