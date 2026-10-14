"""Area-normalized 2G profiles; velocities and widths are in km/s."""
import numpy as np
from scipy.special import erf
from scipy.optimize import brentq, minimize_scalar

C_KMS = 299792.458
SHAPE_NAMES = ("sigma1_kms", "fraction2", "sigma2_kms", "delta_kms")
LOCAL_NAMES = ("bkg", "slope", "total_flux", "center")
LAB_SHAPE_NAMES = SHAPE_NAMES + ("mu1_kms",)
LAB_LOCAL_NAMES = LOCAL_NAMES[:3]


def parameter_names(model, baseline_degree=1):
    if (not isinstance(baseline_degree, (int, np.integer))
            or isinstance(baseline_degree, (bool, np.bool_)) or baseline_degree not in (1, 2)):
        raise ValueError('baseline_degree must be 1 or 2')
    extra = ('quadratic',) if baseline_degree == 2 else ()
    if model == 'free_centers':
        return LOCAL_NAMES+extra, SHAPE_NAMES
    if model == 'lab_centered':
        return LAB_LOCAL_NAMES+extra, LAB_SHAPE_NAMES
    raise ValueError("model must be free_centers/lab_centered")


def _unpack_shape(shape, model):
    """Also accepts JAX arrays without converting them to NumPy."""
    parameter_names(model)
    if model == 'lab_centered':
        sigma1, fraction2, sigma2, delta, mu1 = shape
    else:
        sigma1, fraction2, sigma2, delta = shape
        mu1 = 0.
    return sigma1, fraction2, sigma2, delta, mu1


def density(velocity, shape, *, model='free_centers'):
    """Area-normalized 2G; lab_centered appends mu1 to the four shape parameters."""
    sigma1, fraction2, sigma2, delta, mu1 = _unpack_shape(np.asarray(shape), model)
    v = np.asarray(velocity)-mu1
    return ((1-fraction2)*np.exp(-0.5*(v/sigma1)**2)/(np.sqrt(2*np.pi)*sigma1)
            + fraction2*np.exp(-0.5*((v-delta)/sigma2)**2)/(np.sqrt(2*np.pi)*sigma2))


def pixel_mass(velocity, shape, dv, *, tophat=True, model='free_centers'):
    """Fraction of total line flux per pixel; centered one-pixel top-hat."""
    sigma1, fraction2, sigma2, delta, mu1 = _unpack_shape(np.asarray(shape), model)
    v, dv = np.asarray(velocity)-mu1, np.asarray(dv)
    if not tophat:
        return dv*density(velocity, shape, model=model)
    def component(mu, sigma):
        return 0.5*(erf((v+dv/2-mu)/(np.sqrt(2)*sigma))
                    - erf((v-dv/2-mu)/(np.sqrt(2)*sigma)))
    return (1-fraction2)*component(0, sigma1)+fraction2*component(delta, sigma2)


def _local_offset(x, local, model, reference, baseline_degree):
    local_names, _ = parameter_names(model, baseline_degree)
    if np.shape(local) != (len(local_names),):
        raise ValueError('Local parameter dimensions do not match model and baseline_degree')
    if model == 'lab_centered':
        if reference is None:
            raise ValueError('lab_centered prediction requires a table-pixel reference')
        center = reference
    else:
        center = local[3]
    return np.asarray(x)-center


def _baseline(offset, bkg, slope, quadratic=0.):
    """Pixel-coordinate polynomial; also works on JAX arrays in the likelihood."""
    return bkg+slope*offset+quadratic*offset**2


def baseline(x, local, *, model='free_centers', reference=None, baseline_degree=1):
    """Background in input flux units; quadratic is the coefficient of pixel offset squared."""
    offset = _local_offset(x, local, model, reference, baseline_degree)
    return _baseline(offset, local[0], local[1], local[-1] if baseline_degree == 2 else 0.)


def prediction(x, dv, local, shape, *, tophat=True, model='free_centers', reference=None,
               baseline_degree=1):
    """Pixel flux; lab_centered requires the fixed table_pixel as reference."""
    offset = _local_offset(x, local, model, reference, baseline_degree)
    background = _baseline(offset, local[0], local[1], local[-1] if baseline_degree == 2 else 0.)
    return background+local[2]*pixel_mass(offset*dv, shape, dv, tophat=tophat, model=model)


def _model_peak(shape, dv, *, tophat, model='free_centers'):
    """Velocity of the continuous observation-space profile's global peak."""
    s1, f, s2, delta, mu1 = _unpack_shape(shape, model)
    core = np.array([s1, f, s2, delta])
    extent = abs(delta)+12*max(s1, s2)+dv
    v = np.linspace(-extent, extent, 4001)
    i = np.argmax(pixel_mass(v, core, dv, tophat=tophat))
    peak = minimize_scalar(lambda x: -float(pixel_mass(x, core, dv, tophat=tophat)/dv),
                           bounds=(v[i-1], v[i+1]), method='bounded').x
    return peak+mu1


def profile_fwhm(shape, *, dv=None, model='free_centers'):
    """Continuous FWHM in km/s; optionally include a pixel top-hat."""
    sigma1, fraction2, sigma2, delta, _ = _unpack_shape(np.asarray(shape, dtype=float), model)
    # Translation cannot change FWHM; evaluate near zero even for nonzero mu1.
    shape = np.array([sigma1, fraction2, sigma2, delta])
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
