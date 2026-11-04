"""Area-normalized 2G profiles; velocities and widths are in km/s."""
import numpy as np
from scipy.optimize import brentq, minimize_scalar

C_KMS = 299792.458
SHAPE_NAMES = ("sigma1_kms", "fraction2", "sigma2_kms", "delta_kms", "mu1_kms")
LOCAL_NAMES = ("bkg", "slope", "total_flux")


def parameter_names(baseline_degree=1):
    if (not isinstance(baseline_degree, (int, np.integer))
            or isinstance(baseline_degree, (bool, np.bool_)) or baseline_degree not in (1, 2)):
        raise ValueError('baseline_degree must be 1 or 2')
    extra = ('quadratic',) if baseline_degree == 2 else ()
    return LOCAL_NAMES+extra, SHAPE_NAMES


def _unpack_shape(shape):
    """Also accepts JAX arrays without converting them to NumPy."""
    sigma1, fraction2, sigma2, delta, mu1 = shape
    return sigma1, fraction2, sigma2, delta, mu1


def density(velocity, shape):
    """Area-normalized 2G; both means are relative to the reference wavelength."""
    sigma1, fraction2, sigma2, delta, mu1 = _unpack_shape(np.asarray(shape))
    v = np.asarray(velocity)-mu1
    return ((1-fraction2)*np.exp(-0.5*(v/sigma1)**2)/(np.sqrt(2*np.pi)*sigma1)
            + fraction2*np.exp(-0.5*((v-delta)/sigma2)**2)/(np.sqrt(2*np.pi)*sigma2))


def pixel_mass(velocity, shape, dv):
    """Pixel-center profile density times the sampling width in km/s."""
    return np.asarray(dv)*density(velocity, shape)


def _local_offset(x, local, reference, baseline_degree):
    local_names, _ = parameter_names(baseline_degree)
    if np.shape(local) != (len(local_names),):
        raise ValueError('Local parameter dimensions do not match baseline_degree')
    if reference is None:
        raise ValueError('Prediction requires a table-pixel reference')
    return np.asarray(x)-reference


def _baseline(offset, bkg, slope, quadratic=0.):
    """Pixel-coordinate polynomial; also works on JAX arrays in the likelihood."""
    return bkg+slope*offset+quadratic*offset**2


def baseline(x, local, *, reference, baseline_degree=1):
    """Background in input flux units; quadratic is the coefficient of pixel offset squared."""
    offset = _local_offset(x, local, reference, baseline_degree)
    return _baseline(offset, local[0], local[1], local[-1] if baseline_degree == 2 else 0.)


def prediction(x, dv, local, shape, *, reference, baseline_degree=1):
    """Pixel flux relative to the fixed table_pixel reference."""
    offset = _local_offset(x, local, reference, baseline_degree)
    background = _baseline(offset, local[0], local[1], local[-1] if baseline_degree == 2 else 0.)
    return background+local[2]*pixel_mass(offset*dv, shape, dv)


def _model_peak(shape, dv):
    """Velocity of the continuous observation-space profile's global peak."""
    s1, f, s2, delta, mu1 = _unpack_shape(shape)
    core = np.array([s1, f, s2, delta, 0.])
    extent = abs(delta)+12*max(s1, s2)+dv
    v = np.linspace(-extent, extent, 4001)
    i = np.argmax(pixel_mass(v, core, dv))
    peak = minimize_scalar(lambda x: -float(pixel_mass(x, core, dv)/dv),
                           bounds=(v[i-1], v[i+1]), method='bounded').x
    return peak+mu1


def profile_fwhm(shape):
    """Intrinsic continuous FWHM in km/s."""
    sigma1, fraction2, sigma2, delta, _ = _unpack_shape(np.asarray(shape, dtype=float))
    # Translation cannot change FWHM; evaluate near zero even for nonzero mu1.
    shape = np.array([sigma1, fraction2, sigma2, delta, 0.])
    extent = abs(delta)+12*max(sigma1, sigma2)
    grid = np.linspace(-extent, extent, 4001)
    profile = lambda v: density(v, shape)
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
