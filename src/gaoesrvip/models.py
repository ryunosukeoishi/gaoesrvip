"""Area-normalized 2G profiles; velocities and widths are in km/s."""
import numpy as np
from scipy.special import erf
from scipy.optimize import brentq, minimize_scalar

C_KMS = 299792.458
SHAPE_NAMES = ("sigma1_kms", "fraction", "sigma2_kms", "delta_kms")
LOCAL_NAMES = ("bkg", "slope", "total_flux", "center")


def density(velocity, shape):
    """(1-f) N(v; 0, sigma1) + f N(v; delta, sigma2), in s/km."""
    sigma1, fraction, sigma2, delta = np.asarray(shape)
    v = np.asarray(velocity)
    return ((1-fraction)*np.exp(-0.5*(v/sigma1)**2)/(np.sqrt(2*np.pi)*sigma1)
            + fraction*np.exp(-0.5*((v-delta)/sigma2)**2)/(np.sqrt(2*np.pi)*sigma2))


def pixel_mass(velocity, shape, dv, *, tophat=True):
    """Fraction of total line flux per pixel; centered one-pixel top-hat."""
    sigma1, fraction, sigma2, delta = np.asarray(shape)
    v, dv = np.asarray(velocity), np.asarray(dv)
    if not tophat:
        return dv*density(v, shape)
    def component(mu, sigma):
        return 0.5*(erf((v+dv/2-mu)/(np.sqrt(2)*sigma))
                    - erf((v-dv/2-mu)/(np.sqrt(2)*sigma)))
    return (1-fraction)*component(0, sigma1)+fraction*component(delta, sigma2)


def prediction(x, dv, local, shape, *, tophat=True):
    bkg, slope, flux, center = np.asarray(local)
    offset = np.asarray(x)-center
    return bkg+slope*offset+flux*pixel_mass(offset*dv, shape, dv, tophat=tophat)


def profile_fwhm(shape, *, dv=None):
    """Continuous FWHM in km/s; optionally include a pixel top-hat."""
    sigma1, fraction, sigma2, delta = np.asarray(shape, dtype=float)
    extent = abs(delta)+12*max(sigma1, sigma2)+(dv or 0)
    grid = np.linspace(-extent, extent, 4001)
    profile = (lambda v: density(v, shape)) if dv is None else (
        lambda v: pixel_mass(v, shape, dv)/dv)
    values = profile(grid)
    peak = int(np.argmax(values))
    optimized = minimize_scalar(lambda v: -float(profile(v)),
        bounds=(grid[peak-1], grid[peak+1]), method="bounded")
    half = float(profile(optimized.x))/2
    above = np.flatnonzero(values > half)
    if len(above) == 0 or np.any(np.diff(above) != 1):
        return np.nan
    left, right = int(above[0]), int(above[-1])
    root = lambda v: float(profile(v))-half
    return brentq(root, grid[right], grid[right+1])-brentq(root, grid[left-1], grid[left])
