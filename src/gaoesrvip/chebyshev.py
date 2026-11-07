"""Wavelength-dependent, lab-referenced 2G shapes and coefficient posteriors."""
from dataclasses import dataclass
import numpy as np
from numpy.polynomial.chebyshev import chebvander


COORDINATE_NAMES = ('mu1_kms', 'log_delta_kms', 'log_sigma1_kms',
                    'log_sigma2_kms', 'logit_fraction2')


def _degree(value):
    if (not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_))
            or value < 0):
        raise ValueError('Chebyshev degree must be a nonnegative integer')
    return int(value)


@dataclass(frozen=True)
class ChebyshevIP:
    """Polynomial shape model; degree 2 gives 15 shape coefficients per order.

    variation_scale multiplies the Normal prior SDs of nonconstant terms.
    All five coordinates use the same degree, with no independent line shifts.
    """
    degree: int = 2
    variation_scale: float = 1.
    wavelength_domain: tuple[float, float] | None = None
    reference_dv: float | None = None

    def __post_init__(self):
        _degree(self.degree)
        if not np.isfinite(self.variation_scale) or self.variation_scale <= 0:
            raise ValueError('variation_scale must be finite and positive')
        if self.wavelength_domain is not None:
            _basis(np.asarray(self.wavelength_domain), self.wavelength_domain, self.degree)
        if self.reference_dv is not None and (not np.isfinite(self.reference_dv) or self.reference_dv <= 0):
            raise ValueError('reference_dv must be finite and positive')

    def to_dict(self):
        result = dict(degree=int(self.degree), variation_scale=float(self.variation_scale))
        if self.wavelength_domain is not None:
            result['wavelength_domain'] = list(map(float, self.wavelength_domain))
        if self.reference_dv is not None:
            result['reference_dv'] = float(self.reference_dv)
        return result


def _basis(wavelength, domain, degree, *, extrapolate=False):
    wave = np.asarray(wavelength, dtype=float)
    domain = np.asarray(domain, dtype=float)
    if wave.ndim != 1 or not wave.size or not np.isfinite(wave).all():
        raise ValueError('wavelength must be a nonempty finite one-dimensional array')
    if domain.shape != (2,) or not np.isfinite(domain).all() or domain[1] < domain[0]:
        raise ValueError('wavelength domain must contain finite ordered endpoints')
    if not extrapolate and np.any((wave < domain[0]) | (wave > domain[1])):
        raise ValueError('Wavelength outside fitted domain; set extrapolate=True to allow extrapolation')
    half_span = (domain[1]-domain[0])/2
    if half_span == 0 and degree > 0:
        raise ValueError('Positive Chebyshev degree requires a nonzero wavelength span')
    coordinate = (wave-(domain[0]+domain[1])/2)/(half_span if half_span else 1.)
    return chebvander(coordinate, degree)


def _coordinates_to_shape(q, *, xp=np):
    """Final axis: mu1, log(delta), log(sigma1), log(sigma2), logit(f)."""
    # logaddexp avoids overflow in the sigmoid (also differentiable in JAX).
    fraction = xp.exp(-xp.logaddexp(0., -q[..., 4]))
    return xp.stack((xp.exp(q[..., 2]), fraction, xp.exp(q[..., 3]),
                     xp.exp(q[..., 1]), q[..., 0]), axis=-1)


def _shape_to_coordinates(shape):
    shape = np.asarray(shape)
    s1, f, s2, delta, mu1 = np.moveaxis(shape, -1, 0)
    return np.stack((mu1, np.log(delta), np.log(s1), np.log(s2),
                     np.log(f)-np.log1p(-f)), axis=-1)


def _coefficient_prior(config, dv):
    reference = float(np.median(dv)) if config.reference_dv is None else config.reference_dv
    mean = np.zeros((config.degree+1, 5))
    # At the mean coefficients, f=0.5 and delta=1.5*reference: centroid=0.
    mean[0] = [-.75*reference, np.log(1.5*reference), np.log(1.2*reference),
               np.log(1.2*reference), 0.]
    scale = np.empty_like(mean)
    scale[0] = [2*reference, 1., 1., 1., 2.]
    for k in range(1, config.degree+1):
        scale[k] = np.array([reference, .5, .5, .5, 1.])*config.variation_scale/k**2
    return mean, scale


def _prepare(data, config):
    wave = np.asarray(data.lab_wave, dtype=float)
    if wave.shape != (len(data.x0),) or not wave.size or not np.isfinite(wave).all():
        raise ValueError('Chebyshev IP requires finite lab_wave for every line')
    if len(np.unique(wave)) < config.degree+1:
        raise ValueError(f'Order {data.order}: degree {config.degree} requires at least '
                         f'{config.degree+1} distinct reference wavelengths')
    domain = np.array([wave.min(), wave.max()] if config.wavelength_domain is None
                      else config.wavelength_domain, dtype=float)
    basis = _basis(wave, domain, config.degree)
    if np.linalg.matrix_rank(basis) < config.degree+1:
        raise ValueError(f'Order {data.order}: Chebyshev design matrix is rank deficient')
    mean, scale = _coefficient_prior(config, data.dv)
    return dict(domain=domain, basis=basis, prior_mean=mean, prior_scale=scale)


def cumulants(shape):
    """Intrinsic 2G cumulants κ1–κ5, in (km/s)**k; final axis is the cumulant."""
    s1, f, s2, delta, mu1 = np.moveaxis(np.asarray(shape), -1, 0)
    a, b = -f*delta, (1-f)*delta
    k2 = (1-f)*(s1*s1+a*a)+f*(s2*s2+b*b)
    k3 = (1-f)*(a**3+3*a*s1*s1)+f*(b**3+3*b*s2*s2)
    m4 = (1-f)*(a**4+6*a*a*s1*s1+3*s1**4)+f*(b**4+6*b*b*s2*s2+3*s2**4)
    m5 = (1-f)*(a**5+10*a**3*s1*s1+15*a*s1**4)+f*(b**5+10*b**3*s2*s2+15*b*s2**4)
    return np.stack((mu1+f*delta, k2, k3, m4-3*k2*k2, m5-10*k2*k3), axis=-1)


@dataclass
class ChebyshevFit:
    """MAP estimate, optional genuine posterior draws, and independent LS fits.

    Empty posterior arrays denote MAP-only inference, never a zero-width CI.
    Individual fits are comparison points, not observations in the mixture.
    """
    coefficients: np.ndarray
    domain: np.ndarray
    prior_mean: np.ndarray
    prior_scale: np.ndarray
    observed_peak: np.ndarray
    map_local: np.ndarray
    map_inlier_probability: np.ndarray
    mixture_map: np.ndarray  # s_in, s_out, epsilon
    individual_local: np.ndarray
    individual_shape: np.ndarray
    individual_success: np.ndarray
    coefficient_samples: np.ndarray
    inlier_probability_samples: np.ndarray
    mixture_samples: np.ndarray
    diagnostics: dict

    array_fields = ('coefficients', 'domain', 'prior_mean', 'prior_scale', 'observed_peak',
        'map_local', 'map_inlier_probability', 'mixture_map', 'individual_local',
        'individual_shape', 'individual_success', 'coefficient_samples',
        'inlier_probability_samples', 'mixture_samples')

    def __post_init__(self):
        for name in self.array_fields:
            setattr(self, name, np.asarray(getattr(self, name)))
        n = len(self.observed_peak)
        if self.coefficients.ndim != 2 or self.coefficients.shape[1] != 5:
            raise ValueError('Chebyshev coefficients must have shape (degree + 1, 5)')
        if (self.prior_mean.shape != self.coefficients.shape
                or self.prior_scale.shape != self.coefficients.shape
                or self.individual_shape.shape != (n, 5)
                or self.individual_success.shape != (n,)
                or self.map_inlier_probability.shape != (n,)
                or self.coefficient_samples.shape[1:] != self.coefficients.shape
                or self.inlier_probability_samples.shape != (len(self.coefficient_samples), n)
                or self.mixture_samples.shape != (len(self.coefficient_samples), 3)):
            raise ValueError('Inconsistent Chebyshev result arrays')
        _basis(self.domain, self.domain, self.degree)

    @property
    def degree(self):
        return len(self.coefficients)-1

    @property
    def has_posterior(self):
        return len(self.coefficient_samples) > 0

    @property
    def inlier_probability(self):
        return (self.inlier_probability_samples.mean(axis=0) if self.has_posterior
                else self.map_inlier_probability)

    def predict(self, wavelength, *, samples=True, extrapolate=False):
        """Return (draw, position, 5); MAP-only returns one point estimate.

        samples=False evaluates one valid curve at posterior-median coefficients
        (or the MAP coefficients), preserving the deterministic wavelength model.
        """
        basis = _basis(wavelength, self.domain, self.degree, extrapolate=extrapolate)
        if samples and self.has_posterior:
            coefficients = self.coefficient_samples
        else:
            point = (np.median(self.coefficient_samples, axis=0) if self.has_posterior
                     else self.coefficients)
            coefficients = point[None]
        with np.errstate(over='ignore', under='ignore'):
            shape = _coordinates_to_shape(np.einsum('nk,dkq->dnq', basis, coefficients))
        if (not np.isfinite(shape).all() or np.any(shape[..., [0, 2, 3]] <= 0)
                or np.any((shape[..., 1] <= 0) | (shape[..., 1] >= 1))):
            raise ValueError('Predicted IP exceeds floating-point range')
        return shape


