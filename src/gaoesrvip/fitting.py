"""ThAr-only individual/common fits with reference-wavelength 2G profiles."""
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from .data import OrderData
from .models import pixel_mass, parameter_names, _unpack_shape, _baseline


def _bounds(data, baseline_degree=1):
    """Bounds and seeds; profile locations are km/s from the reference wavelength."""
    parameter_names(baseline_degree)
    n = len(data.x0)
    ymin = np.min(np.where(data.mask, data.y, np.inf), axis=1)
    ymax = np.max(np.where(data.mask, data.y, -np.inf), axis=1)
    span = np.maximum(ymax-ymin, 1.)
    edge = np.array([np.median(y[m]) for y, m in zip(data.y, data.mask)])
    positive = np.maximum(np.sum(np.where(data.mask, np.maximum(data.y-edge[:, None], 0), 0), axis=1), 1.)
    peak_offsets = np.array([x[m][np.argmax(y[m])]-x0
        for x, y, m, x0 in zip(data.x, data.y, data.mask, data.x0)])
    local = np.column_stack((edge, np.zeros(n), positive))
    shape = np.column_stack((1.2*data.dv, np.full(n, 0.25), 1.2*data.dv, 1.5*data.dv))
    lower = np.column_stack((ymin-2*span, -span, np.full(n, 1e-8)))
    upper = np.column_stack((ymax+2*span, span, np.maximum(10*positive, 1.)))
    reference = float(np.median(data.dv))
    if not np.isfinite(data.table_pixel).all():
        raise ValueError('Fitting requires finite table_pixel for every line')
    if not np.isfinite(data.dv).all() or np.any(data.dv <= 0):
        raise ValueError('Fitting requires finite positive dv for every line')
    peak_velocity = (peak_offsets+data.x0-data.table_pixel)*data.dv
    mu1 = peak_velocity-shape[:, 1]*shape[:, 3]
    shape = np.column_stack((shape, mu1))
    shape_lower = np.array([0.4*reference, 0., 0.4*reference, 0., -4*reference])
    shape_upper = np.array([6*reference, 1., 6*reference, 4*reference, 4*reference])
    if baseline_degree == 2:
        origin = data.table_pixel
        radius = np.maximum(np.max(np.where(data.mask, np.abs(data.x-origin[:, None]), 0.), axis=1), 1.)
        curvature_limit = span/radius**2
        local = np.column_stack((local, np.zeros(n)))
        lower = np.column_stack((lower, -curvature_limit))
        upper = np.column_stack((upper, curvature_limit))
    return lower, upper, shape_lower, shape_upper, local, shape


def _optimize_block(data, max_nfev, lo, hi, slo, shi, local, shape, sparse,
                    baseline_degree=1):
    n = len(data.x0)
    nlocal = local.shape[1]
    lower, upper = np.r_[lo.ravel(), slo], np.r_[hi.ravel(), shi]
    initial = np.clip((np.r_[local.ravel(), shape]-lower)/(upper-lower), 1e-5, 1-1e-5)
    origin = data.table_pixel
    x = np.where(data.mask, data.x-origin[:, None], 0.)
    y, noise = np.where(data.mask, data.y, 0.), np.where(data.mask, data.noise, 1.)

    def unpack(scaled):
        physical = lower+scaled*(upper-lower)
        return physical[:n*nlocal].reshape(n, nlocal), physical[n*nlocal:]

    def residual(scaled):
        params, profile = unpack(scaled)
        offset = x
        mass = pixel_mass(offset*data.dv[:, None], profile, data.dv[:, None])
        quadratic = params[:, -1, None] if baseline_degree == 2 else 0.
        predicted = _baseline(offset, params[:, 0, None], params[:, 1, None], quadratic)+params[:, 2, None]*mass
        return ((predicted-y)/noise)[data.mask]

    options = {}
    if sparse:
        jac = lil_matrix((int(data.mask.sum()), len(initial)), dtype=int)
        start = 0
        for i, count in enumerate(data.mask.sum(axis=1)):
            jac[start:start+count, i*nlocal:(i+1)*nlocal] = 1
            jac[start:start+count, n*nlocal:] = 1
            start += count
        options['jac_sparsity'] = jac.tocsr()
    trials = []
    for trial in (None, .5, .75):
        seed = initial.copy()
        _, shape_seed = unpack(seed)
        shape_seed = shape_seed.copy()
        if trial is not None:
            shape_seed[4] += (shape_seed[1]-trial)*shape_seed[3]
            shape_seed[1] = trial
        seed[n*nlocal:] = np.clip((shape_seed-slo)/(shi-slo), 1e-5, 1-1e-5)
        trials.append(least_squares(residual, seed, bounds=(1e-8, 1-1e-8),
            max_nfev=max_nfev, ftol=1e-8, xtol=1e-8, gtol=1e-8, **options))
    successful = [trial for trial in trials if trial.success and np.isfinite(trial.cost)]
    if not successful:
        raise RuntimeError(f'Order {data.order}: initialization did not converge')
    best = min(successful, key=lambda trial: trial.cost)
    return *unpack(best.x), best


def _optimize(data, mode, max_nfev, baseline_degree=1):
    lo, hi, slo, shi, local, shape = _bounds(data, baseline_degree)
    if mode == 'common':
        local, shape, _ = _optimize_block(data, max_nfev, lo, hi, slo, shi,
                                            local, np.median(shape, axis=0), True, baseline_degree)
    else:
        blocks = [_optimize_block(data.select([i]), max_nfev, lo[i:i+1], hi[i:i+1],
                  slo, shi, local[i:i+1], shape[i], False, baseline_degree)
                  for i in range(len(data.x0))]
        local, shape = np.vstack([b[0] for b in blocks]), np.array([b[1] for b in blocks])
    return local, shape, (lo, hi, slo, shi)


def _sample(data, mode, bounds, local, shape, *, warmup, samples, chains,
            seed, target_accept, max_tree_depth, progress, baseline_degree=1):
    import jax
    # Matches the original ThAr analysis's numerical precision.
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS, init_to_value
    lo, hi, slo, shi = (jnp.asarray(a) for a in bounds)
    origin = data.table_pixel
    x = jnp.asarray(np.where(data.mask, data.x-origin[:, None], 0))
    y = jnp.asarray(np.where(data.mask, data.y, 0))
    noise = jnp.asarray(np.where(data.mask, data.noise, 1))
    dv, mask = jnp.asarray(data.dv[:, None]), jnp.asarray(data.mask)

    def numpyro_model():
        local_scaled = numpyro.sample("local_scaled", dist.Uniform(jnp.zeros(lo.shape), jnp.ones(lo.shape)).to_event(2))
        shape_scaled = numpyro.sample("shape_scaled", dist.Uniform(jnp.zeros(shape.shape), jnp.ones(shape.shape)).to_event(len(shape.shape)))
        local_values = lo+local_scaled*(hi-lo)
        shape_values = slo+shape_scaled*(shi-slo)
        numpyro.deterministic("local", local_values)
        numpyro.deterministic("shape", shape_values)
        if mode == "individual":
            profile = [shape_values[:, i, None] for i in range(shape.shape[-1])]
        else:
            profile = shape_values
        s1, fraction2, s2, delta, mu1 = _unpack_shape(profile)
        offset = x
        v = offset*dv-mu1
        def component(mu, sigma):
            return dv*jnp.exp(-0.5*((v-mu)/sigma)**2)/(jnp.sqrt(2*jnp.pi)*sigma)
        mass = (1-fraction2)*component(0, s1)+fraction2*component(delta, s2)
        quadratic = local_values[:, -1, None] if baseline_degree == 2 else 0.
        prediction = (_baseline(offset, local_values[:, 0, None], local_values[:, 1, None], quadratic)
                      + local_values[:, 2, None]*mass)
        with numpyro.handlers.mask(mask=mask):
            numpyro.sample("obs", dist.Normal(prediction, noise), obs=y)

    init = {"local_scaled": np.clip((local-np.asarray(lo))/(np.asarray(hi-lo)), 1e-5, 1-1e-5),
            "shape_scaled": np.clip((shape-np.asarray(slo))/np.asarray(shi-slo), 1e-5, 1-1e-5)}
    # High-S/N lines constrain some scaled coordinates much more tightly than
    # the default mass-matrix variance floor. Learn their actual scales.
    kernel = NUTS(numpyro_model, init_strategy=init_to_value(values=init), target_accept_prob=target_accept,
                  max_tree_depth=max_tree_depth, regularize_mass_matrix=False)
    sampler = MCMC(kernel, num_warmup=warmup, num_samples=samples, num_chains=chains,
                   chain_method="sequential", progress_bar=progress)
    sampler.run(jax.random.PRNGKey(seed))
    draws = {k: np.asarray(v) for k, v in sampler.get_samples().items() if k in ("local", "shape")}
    if samples >= 4:
        sampler.print_summary()
    return draws['local'], draws['shape']


def fit(input_data, *, mode="common", baseline_degree=1,
        warmup=500, samples=500, chains=1, seed=0, target_accept=0.95,
        max_tree_depth=10, max_nfev=2000, progress=True, degree=2, nuts=None, maxiter=10000,
        degrees=None):
    """Fit pixel-center 2G profiles relative to the reference wavelength.

    common shares all five IP parameters; individual fits them per line.
    Baseline coefficients and flux remain local. baseline_degree=1 fits
    bkg+slope*u; 2 adds quadratic*u**2, with u in pixels from table_pixel.
    mode='chebyshev' fits a wavelength-dependent IP and a whole-line residual
    mixture, retaining independent LS comparison fits. It uses MAP only by
    default; nuts=True adds sampling initialized at the MAP. The degree is
    user-selected. Alternatively, degrees searches candidate degrees per order
    using BIC_MAP = -2*logL(MAP) + k*log(n_valid_pixels). The selected MAPs are
    reused, with optional NUTS only for those degrees. degree_scores records
    the comparison, including failed candidates. This is a BIC-like heuristic,
    not maximum-likelihood BIC. individual/common always use NUTS after initialization.
    """
    from .results import FitResult, OrderFit
    if mode not in ("individual", "common", "chebyshev"):
        raise ValueError("mode must be individual/common/chebyshev")
    if degrees is not None and mode != 'chebyshev':
        raise ValueError('degrees is only available for mode=chebyshev')
    parameter_names(baseline_degree)
    if mode == 'chebyshev':
        from .spectral_mixture import fit_chebyshev, fit_chebyshev_degrees
        fit_model = fit_chebyshev if degrees is None else fit_chebyshev_degrees
        degree_option = dict(degree=degree) if degrees is None else dict(degrees=degrees)
        return fit_model(input_data, **degree_option, nuts=False if nuts is None else nuts,
            baseline_degree=baseline_degree, warmup=warmup, samples=samples,
            chains=chains, seed=seed, target_accept=target_accept, max_tree_depth=max_tree_depth,
            max_nfev=max_nfev, maxiter=maxiter, progress=progress)
    if nuts is not None:
        raise ValueError('nuts is only configurable for mode=chebyshev; individual/common use NUTS')
    if isinstance(input_data, OrderData):
        data = [input_data]
    else:
        data = list(input_data)
    result_orders = []
    for index, d in enumerate(sorted(data, key=lambda d: d.order)):
        if progress:
            print(f"Order {d.order}: {len(d.x0)} lines, {mode}, MCMC, baseline_degree={baseline_degree}")
        local, shape, bounds = _optimize(d, mode, max_nfev, baseline_degree)
        local, shape = _sample(d, mode, bounds, local, shape,
            warmup=warmup, samples=samples, chains=chains, seed=(seed+index) % 2**32,
            target_accept=target_accept, max_tree_depth=max_tree_depth, progress=progress,
            baseline_degree=baseline_degree)
        result_orders.append(OrderFit(d, local, shape, baseline_degree=baseline_degree))
    settings = dict(mode=mode, baseline_degree=int(baseline_degree),
        warmup=int(warmup), samples=int(samples), chains=int(chains), seed=int(seed),
        target_accept=float(target_accept), max_tree_depth=int(max_tree_depth), max_nfev=int(max_nfev))
    return FitResult(result_orders, settings)
