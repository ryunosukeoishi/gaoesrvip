"""ThAr-only individual/common fits with the original bounded 2G model."""
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from .data import OrderData
from .models import pixel_mass


def _bounds(data):
    """Original common velocity-model bounds; centers use x-x0 internally."""
    n = len(data.x0)
    ymin = np.min(np.where(data.mask, data.y, np.inf), axis=1)
    ymax = np.max(np.where(data.mask, data.y, -np.inf), axis=1)
    span = np.maximum(ymax-ymin, 1.)
    edge = np.array([np.median(y[m]) for y, m in zip(data.y, data.mask)])
    positive = np.maximum(np.sum(np.where(data.mask, np.maximum(data.y-edge[:, None], 0), 0), axis=1), 1.)
    peak_offsets = np.array([x[m][np.argmax(y[m])]-x0
        for x, y, m, x0 in zip(data.x, data.y, data.mask, data.x0)])
    local = np.column_stack((edge, np.zeros(n), positive, np.clip(peak_offsets, -1.9, 1.9)))
    shape = np.column_stack((1.2*data.dv, np.full(n, 0.25), 1.2*data.dv, 1.5*data.dv))
    lower = np.column_stack((ymin-2*span, -span, np.full(n, 1e-8), np.full(n, -2.)))
    upper = np.column_stack((ymax+2*span, span, np.maximum(10*positive, 1.), np.full(n, 2.)))
    reference = float(np.median(data.dv))
    shape_lower = np.array([0.4*reference, 0.001, 0.4*reference, -4*reference])
    shape_upper = np.array([4*reference, 0.49, 6*reference, 4*reference])
    return lower, upper, shape_lower, shape_upper, local, shape


def _optimize_block(data, tophat, max_nfev, lo, hi, slo, shi, local, shape, reference, sparse):
    n = len(data.x0)
    lower, upper = np.r_[lo.ravel(), slo], np.r_[hi.ravel(), shi]
    initial = np.clip((np.r_[local.ravel(), shape]-lower)/(upper-lower), 1e-5, 1-1e-5)
    x = np.where(data.mask, data.x-data.x0[:, None], 0.)
    y, noise = np.where(data.mask, data.y, 0.), np.where(data.mask, data.noise, 1.)

    def unpack(scaled):
        physical = lower+scaled*(upper-lower)
        return physical[:n*4].reshape(n, 4), physical[n*4:]

    def residual(scaled):
        params, profile = unpack(scaled)
        offset = x-params[:, 3, None]
        mass = pixel_mass(offset*data.dv[:, None], profile, data.dv[:, None], tophat=tophat)
        model = params[:, 0, None]+params[:, 1, None]*offset+params[:, 2, None]*mass
        return ((model-y)/noise)[data.mask]

    options = {}
    if sparse:
        jac = lil_matrix((int(data.mask.sum()), len(initial)), dtype=int)
        start = 0
        for i, count in enumerate(data.mask.sum(axis=1)):
            jac[start:start+count, i*4:(i+1)*4] = 1
            jac[start:start+count, n*4:] = 1
            start += count
        options['jac_sparsity'] = jac.tocsr()
    trials = []
    for sign in (None, 1, -1):
        seed = initial.copy()
        _, shape_seed = unpack(seed)
        shape_seed = shape_seed.copy()
        if sign is not None:
            shape_seed[3] = sign*max(abs(shape_seed[3]), reference)
        seed[n*4:] = np.clip((shape_seed-slo)/(shi-slo), 1e-5, 1-1e-5)
        trials.append(least_squares(residual, seed, bounds=(1e-8, 1-1e-8),
            max_nfev=max_nfev, ftol=1e-8, xtol=1e-8, gtol=1e-8, **options))
    successful = [trial for trial in trials if trial.success and np.isfinite(trial.cost)]
    if not successful:
        raise RuntimeError(f'Order {data.order}: initialization did not converge')
    best = min(successful, key=lambda trial: trial.cost)
    return *unpack(best.x), best


def _optimize(data, mode, tophat, max_nfev):
    lo, hi, slo, shi, local, shape = _bounds(data)
    reference = float(np.median(data.dv))
    if mode == 'common':
        local, shape, _ = _optimize_block(data, tophat, max_nfev, lo, hi, slo, shi,
                                            local, np.median(shape, axis=0), reference, True)
    else:
        blocks = [_optimize_block(data.select([i]), tophat, max_nfev, lo[i:i+1], hi[i:i+1],
                  slo, shi, local[i:i+1], shape[i], reference, False) for i in range(len(data.x0))]
        local, shape = np.vstack([b[0] for b in blocks]), np.array([b[1] for b in blocks])
    return local, shape, (lo, hi, slo, shi)


def _sample(data, mode, tophat, bounds, local, shape, *, warmup, samples, chains,
            seed, target_accept, max_tree_depth, progress):
    import jax
    # Matches the original ThAr analysis's numerical precision.
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS, init_to_value
    lo, hi, slo, shi = (jnp.asarray(a) for a in bounds)
    x = jnp.asarray(np.where(data.mask, data.x-data.x0[:, None], 0))
    y = jnp.asarray(np.where(data.mask, data.y, 0))
    noise = jnp.asarray(np.where(data.mask, data.noise, 1))
    dv, mask = jnp.asarray(data.dv[:, None]), jnp.asarray(data.mask)

    def model():
        local_scaled = numpyro.sample("local_scaled", dist.Uniform(jnp.zeros(lo.shape), jnp.ones(lo.shape)).to_event(2))
        shape_scaled = numpyro.sample("shape_scaled", dist.Uniform(jnp.zeros(shape.shape), jnp.ones(shape.shape)).to_event(len(shape.shape)))
        local_values = lo+local_scaled*(hi-lo)
        shape_values = slo+shape_scaled*(shi-slo)
        numpyro.deterministic("local", local_values)
        numpyro.deterministic("shape", shape_values)
        if mode == "individual":
            s1, fraction, s2, delta = (shape_values[:, i, None] for i in range(4))
        else:
            s1, fraction, s2, delta = shape_values
        offset = x-local_values[:, 3, None]
        v = offset*dv
        def component(mu, sigma):
            if tophat:
                return 0.5*(jax.scipy.special.erf((v+dv/2-mu)/(jnp.sqrt(2)*sigma))
                            - jax.scipy.special.erf((v-dv/2-mu)/(jnp.sqrt(2)*sigma)))
            return dv*jnp.exp(-0.5*((v-mu)/sigma)**2)/(jnp.sqrt(2*jnp.pi)*sigma)
        mass = (1-fraction)*component(0, s1)+fraction*component(delta, s2)
        prediction = local_values[:, 0, None]+local_values[:, 1, None]*offset+local_values[:, 2, None]*mass
        with numpyro.handlers.mask(mask=mask):
            numpyro.sample("obs", dist.Normal(prediction, noise), obs=y)

    init = {"local_scaled": np.clip((local-np.asarray(lo))/(np.asarray(hi-lo)), 1e-5, 1-1e-5),
            "shape_scaled": np.clip((shape-np.asarray(slo))/np.asarray(shi-slo), 1e-5, 1-1e-5)}
    # High-S/N lines constrain some scaled coordinates much more tightly than
    # the default mass-matrix variance floor. Learn their actual scales.
    kernel = NUTS(model, init_strategy=init_to_value(values=init), target_accept_prob=target_accept,
                  max_tree_depth=max_tree_depth, regularize_mass_matrix=False)
    sampler = MCMC(kernel, num_warmup=warmup, num_samples=samples, num_chains=chains,
                   chain_method="sequential", progress_bar=progress)
    sampler.run(jax.random.PRNGKey(seed))
    draws = {k: np.asarray(v) for k, v in sampler.get_samples().items() if k in ("local", "shape")}
    if samples >= 4:
        sampler.print_summary()
    return draws['local'], draws['shape']


def fit(input_data, *, mode="common", tophat=True,
        warmup=500, samples=500, chains=1, seed=0, target_accept=0.95,
        max_tree_depth=10, max_nfev=2000, progress=True):
    """Fit selected lines with NumPyro NUTS and print the MCMC summary."""
    from .results import FitResult, OrderFit
    if mode not in ("individual", "common"):
        raise ValueError("mode must be individual/common")
    if isinstance(input_data, OrderData):
        data = [input_data]
    else:
        data = list(input_data)
    result_orders = []
    for index, d in enumerate(sorted(data, key=lambda d: d.order)):
        if progress:
            print(f"Order {d.order}: {len(d.x0)} lines, {mode}, MCMC, tophat={tophat}")
        local, shape, bounds = _optimize(d, mode, tophat, max_nfev)
        local, shape = _sample(d, mode, tophat, bounds, local, shape,
            warmup=warmup, samples=samples, chains=chains, seed=(seed+index) % 2**32,
            target_accept=target_accept, max_tree_depth=max_tree_depth, progress=progress)
        local = local.copy()
        local[:, :, 3] += d.x0
        result_orders.append(OrderFit(d, local, shape))
    settings = dict(mode=mode, tophat=bool(tophat),
        warmup=int(warmup), samples=int(samples), chains=int(chains), seed=int(seed),
        target_accept=float(target_accept), max_tree_depth=int(max_tree_depth), max_nfev=int(max_nfev))
    return FitResult(result_orders, settings)
