"""Direct Chebyshev spectral MAP fit with one residual-mixture label per line.

No independently estimated IP coordinates or covariance matrices enter this
fit. Both mixture components use the curve IP; the broad component downweights
lines that the curve cannot describe within the narrow residual tolerance.
"""
from time import perf_counter
import numpy as np
from scipy.optimize import minimize, least_squares, lsq_linear
from scipy.special import erf, logsumexp

from .chebyshev import (ChebyshevIP, _prepare, _basis, _coordinates_to_shape,
                        _shape_to_coordinates, ChebyshevFit)
from .fitting import _bounds
from .data import OrderData


def _peak_from_data(data, i):
    y = data.y[i, data.mask[i]]
    k = max(1, min(3, len(y)//4))
    return float(np.max(y)-np.median(np.r_[y[:k], y[-k:]]))


def _mass_and_gradient(velocity, dv, q, *, tophat):
    """Pixel mass and derivatives in (mu1, log delta, log s1, log s2, logit f)."""
    shape = _coordinates_to_shape(q)
    s1, fraction, s2, delta, mu = (shape[:, j, None] for j in range(5))
    normalizer = np.sqrt(2*np.pi)

    def component(center, sigma):
        v = velocity-center
        if tophat:
            a, b = (v-dv/2)/sigma, (v+dv/2)/sigma
            pa, pb = np.exp(-.5*a*a)/normalizer, np.exp(-.5*b*b)/normalizer
            mass = .5*(erf(b/np.sqrt(2))-erf(a/np.sqrt(2)))
            return mass, (pa-pb)/sigma, a*pa-b*pb
        z = v/sigma
        mass = dv*np.exp(-.5*z*z)/(normalizer*sigma)
        return mass, mass*z/sigma, mass*(z*z-1)

    first, dmu_first, dlog_first = component(mu, s1)
    second, dmu_second, dlog_second = component(mu+delta, s2)
    mass = (1-fraction)*first+fraction*second
    derivative = np.stack(((1-fraction)*dmu_first+fraction*dmu_second,
        fraction*dmu_second*delta, (1-fraction)*dlog_first,
        fraction*dlog_second, fraction*(1-fraction)*(second-first)), axis=-1)
    return mass, derivative


def _line_mixture(residual, noise, mask, inlier_scale, extra_scale, epsilon):
    """Normalized-flux likelihood: sum pixel log densities BEFORE mixing lines."""
    variance = np.stack((noise**2+inlier_scale**2,
                        noise**2+inlier_scale**2+extra_scale**2), axis=-1)
    log_density = -.5*np.sum(np.where(mask[..., None],
        residual[..., None]**2/variance+np.log(2*np.pi*variance), 0.), axis=1)
    log_components = log_density+np.log([1-epsilon, epsilon])
    log_likelihood = logsumexp(log_components, axis=1)
    responsibility = np.exp(log_components-log_likelihood[:, None])
    weight = np.sum(responsibility[:, None, :]/variance, axis=-1)*mask
    variance_gradient = .5*np.sum(np.where(mask[..., None], responsibility[:, None, :]*
        (1/variance-residual[..., None]**2/variance**2), 0.), axis=(0, 1))
    return log_likelihood, responsibility, weight, variance_gradient


class _SpectralObjective:
    def __init__(self, data, context, *, tophat, baseline_degree,
                 inlier_scale_prior, outlier_extra_prior, log_scale_sd, outlier_beta,
                 prior_weight=1.):
        self.data, self.context = data, context
        self.tophat, self.baseline_degree = tophat, baseline_degree
        self.inlier_base, self.extra_base = inlier_scale_prior, outlier_extra_prior
        self.log_scale_sd, self.beta = np.asarray(log_scale_sd), np.asarray(outlier_beta)
        if not np.isfinite(prior_weight) or prior_weight < 0:
            raise ValueError('prior_weight must be finite and nonnegative')
        self.prior_weight = float(prior_weight)
        self.mask = data.mask
        self.u = np.where(data.mask, data.x-data.table_pixel[:, None], 0.)
        self.velocity, self.dv = self.u*data.dv[:, None], data.dv[:, None]
        # Fixed observed peak: normalization cannot be manipulated by fitted flux.
        self.peak = np.array([_peak_from_data(data, i) for i in range(len(data.x0))])
        if np.any(~np.isfinite(self.peak) | (self.peak <= 0)):
            raise ValueError('Every line requires a positive finite observed peak')
        self.y = np.where(data.mask, data.y/self.peak[:, None], 0.)
        self.noise = np.where(data.mask, data.noise/self.peak[:, None], 1.)
        lo, hi, _, _, local, shape = _bounds(data, 'lab_centered', baseline_degree)
        radius = np.maximum(1., np.max(np.abs(self.u), axis=1))
        self.area = np.maximum(1., local[:, 2]/self.peak)
        self.local_scale = np.column_stack((self.peak, self.peak/radius, self.peak*self.area))
        if baseline_degree == 2:
            self.local_scale = np.column_stack((self.local_scale, self.peak/radius**2))
        self.local_lower, self.local_upper = lo/self.local_scale, hi/self.local_scale
        self.local_shape = local.shape
        self.nlocal = local.size
        self.ncoef = context['prior_mean'].size
        self.initial_local, self.initial_shape = local, shape
        self.baseline_design = [np.ones_like(self.u), self.u/radius[:, None]]
        if baseline_degree == 2:
            self.baseline_design.append((self.u/radius[:, None])**2)

    def evaluate(self, theta):
        local = theta[:self.nlocal].reshape(self.local_shape)
        z = theta[self.nlocal:self.nlocal+self.ncoef].reshape(self.context['prior_mean'].shape)
        epsilon = theta[-3]
        log_scale = theta[-2:]
        inside, extra = np.array([self.inlier_base, self.extra_base])*np.exp(log_scale)
        coefficients = self.context['prior_mean']+self.context['prior_scale']*z
        q = self.context['basis']@coefficients
        mass, derivative = _mass_and_gradient(self.velocity, self.dv, q, tophat=self.tophat)
        design = np.stack((self.baseline_design[0], self.baseline_design[1],
                           self.area[:, None]*mass, *self.baseline_design[2:]), axis=-1)
        predicted = np.einsum('npj,nj->np', design, local)
        residual = predicted-self.y
        loglik, probability, weight, gvariance = _line_mixture(
            residual, self.noise, self.mask, inside, extra, epsilon)
        alpha, beta = self.beta
        penalty = (.5*np.sum(z*z)+.5*np.sum((log_scale/self.log_scale_sd)**2)
                   -(alpha-1)*np.log(epsilon)-(beta-1)*np.log1p(-epsilon))
        value = -np.sum(loglik)+self.prior_weight*penalty
        gprediction = residual*weight
        glocal = np.einsum('np,npj->nj', gprediction, design)
        gq = np.einsum('np,npq->nq', gprediction*(local[:, 2]*self.area)[:, None], derivative)
        gz = (self.context['basis'].T@gq)*self.context['prior_scale']+self.prior_weight*z
        gepsilon = np.sum(probability[:, 0]/(1-epsilon)-probability[:, 1]/epsilon)
        gepsilon += self.prior_weight*(-(alpha-1)/epsilon+(beta-1)/(1-epsilon))
        gscale = np.array([2*inside**2*np.sum(gvariance), 2*extra**2*gvariance[1]])
        gscale += self.prior_weight*log_scale/self.log_scale_sd**2
        gradient = np.r_[glocal.ravel(), gz.ravel(), gepsilon, gscale]
        details = dict(coefficients=coefficients, local=local*self.local_scale,
            prediction=predicted*self.peak[:, None], shape=_coordinates_to_shape(q),
            inlier_probability=probability[:, 0], log_likelihood=loglik,
            inlier_scale=inside, outlier_scale=np.hypot(inside, extra),
            epsilon=epsilon, weight=weight, mass=mass, derivative=derivative,
            design=design, residual=residual)
        return float(value), gradient, details

    def __call__(self, theta):
        value, gradient, _ = self.evaluate(theta)
        return value, gradient

    def initial(self, coefficients, epsilon=.15, log_scale=None):
        mass, _ = _mass_and_gradient(self.velocity, self.dv,
                                    self.context['basis']@coefficients, tophat=self.tophat)
        design = np.stack((self.baseline_design[0], self.baseline_design[1],
                           self.area[:, None]*mass, *self.baseline_design[2:]), axis=-1)
        local = []
        for i in range(len(self.peak)):
            m = self.mask[i]
            sigma = np.sqrt(self.noise[i, m]**2+self.inlier_base**2)
            solution = lsq_linear(design[i, m]/sigma[:, None], self.y[i, m]/sigma,
                bounds=(self.local_lower[i], self.local_upper[i]), tol=1e-10)
            if not solution.success:
                raise RuntimeError(f'Initial nuisance fit failed for line {self.data.line_ids[i]}')
            local.append(solution.x)
        z = (coefficients-self.context['prior_mean'])/self.context['prior_scale']
        return np.r_[np.array(local).ravel(), z.ravel(), epsilon,
                     np.zeros(2) if log_scale is None else log_scale]


def _initial_residual_jacobian(theta, objective, scatter):
    """Same 2% spectral initializer as the NUTS notebook, with an analytic Jacobian."""
    o = objective
    n, nlocal = o.local_shape
    local = theta[:o.nlocal].reshape(o.local_shape)
    z = theta[o.nlocal:].reshape(o.context['prior_mean'].shape)
    coefficients = o.context['prior_mean']+o.context['prior_scale']*z
    mass, derivative = _mass_and_gradient(o.velocity, o.dv,
        o.context['basis']@coefficients, tophat=o.tophat)
    design = np.stack((o.baseline_design[0], o.baseline_design[1],
                       o.area[:, None]*mass, *o.baseline_design[2:]), axis=-1)
    predicted = np.einsum('npj,nj->np', design, local)
    sigma = np.sqrt(o.noise**2+scatter**2)
    residual = np.r_[((predicted-o.y)/sigma)[o.mask], z.ravel()]
    jacobian = np.zeros((*o.y.shape, o.nlocal+o.ncoef))
    for i in range(n):
        jacobian[i, :, i*nlocal:(i+1)*nlocal] = design[i]/sigma[i, :, None]
    jq = derivative*(o.area*local[:, 2])[:, None, None]/sigma[..., None]
    jacobian[:, :, o.nlocal:] = (jq[:, :, None, :]*o.context['basis'][:, None, :, None]
        *o.context['prior_scale'][None, None]).reshape(n, -1, o.ncoef)
    prior_jacobian = np.zeros((o.ncoef, o.nlocal+o.ncoef))
    prior_jacobian[:, o.nlocal:] = np.eye(o.ncoef)
    return residual, np.vstack((jacobian[o.mask], prior_jacobian))


def _initialize(objective, scatter, max_nfev, coefficient_guard=12.):
    o = objective
    shape = np.median(o.initial_shape, axis=0)
    shape[4] += (shape[1]-.5)*shape[3]
    shape[1] = .5
    coefficients = np.zeros_like(o.context['prior_mean'])
    coefficients[0] = _shape_to_coordinates(shape)
    initial = np.r_[(o.initial_local/o.local_scale).ravel(),
                    ((coefficients-o.context['prior_mean'])/o.context['prior_scale']).ravel()]
    lower = np.r_[o.local_lower.ravel(), np.full(o.ncoef, -coefficient_guard)]
    upper = np.r_[o.local_upper.ravel(), np.full(o.ncoef, coefficient_guard)]
    result = least_squares(lambda x: _initial_residual_jacobian(x, o, scatter)[0],
        np.clip(initial, lower+1e-8, upper-1e-8),
        jac=lambda x: _initial_residual_jacobian(x, o, scatter)[1],
        bounds=(lower, upper), x_scale='jac', max_nfev=max_nfev)
    if not result.success or not np.isfinite(result.fun).all():
        raise RuntimeError(f'Least-squares initialization failed: {result.message}')
    return np.r_[result.x, .2, np.log(scatter/o.inlier_base), 0.]


def _independent_fits(data, tophat, baseline_degree, max_nfev):
    """Keep every line in the joint fit even if its comparison fit fails."""
    from .fitting import _optimize_block
    lo, hi, slo, shi, seed_local, seed_shape = _bounds(data, 'lab_centered', baseline_degree)
    local, shape = np.full_like(seed_local, np.nan), np.full_like(seed_shape, np.nan)
    success, reports = np.zeros(len(data.x0), bool), []
    started = perf_counter()
    for i in range(len(data.x0)):
        try:
            li, si, solution = _optimize_block(data.select([i]), tophat, max_nfev,
                lo[i:i+1], hi[i:i+1], slo, shi, seed_local[i:i+1], seed_shape[i],
                float(np.median(data.dv)), False, 'lab_centered', baseline_degree)
            local[i], shape[i], success[i] = li[0], si, True
            reports.append(dict(line_id=str(data.line_ids[i]), success=True,
                                cost=float(solution.cost), message=str(solution.message)))
        except (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            reports.append(dict(line_id=str(data.line_ids[i]), success=False, message=str(exc)))
    return local, shape, success, dict(seconds=perf_counter()-started, lines=reports)


def _map_fit(objective, maxiter, max_nfev):
    o = objective
    started = perf_counter()
    initial = _initialize(o, .02, max_nfev)
    bounds = (list(zip(o.local_lower.ravel(), o.local_upper.ravel()))
              +[(-12., 12.)]*o.ncoef+[(1e-6, 1-1e-6), (-8., 5.), (-8., 5.)])
    solution = minimize(o, initial, jac=True, method='L-BFGS-B', bounds=bounds,
        options=dict(maxiter=maxiter, maxfun=5*maxiter, ftol=1e-15, gtol=1e-6,
                     maxls=40, maxcor=40))
    value, gradient, details = o.evaluate(solution.x)
    projected, guard_hits = gradient.copy(), 0
    for j, (lower, upper) in enumerate(bounds):
        at_lower, at_upper = solution.x[j] <= lower+1e-6, solution.x[j] >= upper-1e-6
        if at_lower:
            projected[j] = min(0., projected[j])
        if at_upper:
            projected[j] = max(0., projected[j])
        # epsilon can have a legitimate MAP at zero (Beta(1,4) prior).
        if (at_lower or at_upper) and j >= o.nlocal and j != len(solution.x)-3:
            guard_hits += 1
    norm = float(np.max(np.abs(projected)))
    report = dict(success=bool(solution.success), objective=float(value),
        log_likelihood=float(details['log_likelihood'].sum()), projected_gradient=norm,
        numerical_bound_hits=guard_hits, iterations=int(solution.nit),
        message=str(solution.message), seconds=perf_counter()-started)
    if not solution.success or not np.isfinite(value) or norm > .01 or guard_hits:
        raise RuntimeError(f'Chebyshev MAP did not converge: {report}')
    return solution.x, details, report


def _numpyro_model(o):
    """Same density as the MAP objective, up to parameter-independent constants."""
    import jax.numpy as jnp
    import jax.scipy as jsp
    import numpyro
    import numpyro.distributions as dist

    z = numpyro.sample('coefficient_z',
        dist.Normal(jnp.zeros(o.context['prior_mean'].shape), 1.).to_event(2))
    unit = numpyro.sample('local_unit', dist.Uniform(jnp.zeros(o.local_shape), 1.).to_event(2))
    epsilon = numpyro.sample('epsilon', dist.Beta(*o.beta))
    log_scale = numpyro.sample('log_scale', dist.Normal(jnp.zeros(2), jnp.asarray(o.log_scale_sd)).to_event(1))
    coefficients = jnp.asarray(o.context['prior_mean'])+jnp.asarray(o.context['prior_scale'])*z
    shape = _coordinates_to_shape(jnp.asarray(o.context['basis'])@coefficients, xp=jnp)
    local_scaled = jnp.asarray(o.local_lower)+unit*jnp.asarray(o.local_upper-o.local_lower)
    local = local_scaled*jnp.asarray(o.local_scale)
    s1, fraction, s2, delta, mu = (shape[:, j, None] for j in range(5))
    velocity, dv = jnp.asarray(o.velocity), jnp.asarray(o.dv)

    def component(center, sigma):
        v = velocity-center
        if o.tophat:
            return .5*(jsp.special.erf((v+dv/2)/(jnp.sqrt(2)*sigma))
                       -jsp.special.erf((v-dv/2)/(jnp.sqrt(2)*sigma)))
        return dv*jnp.exp(-.5*(v/sigma)**2)/(jnp.sqrt(2*jnp.pi)*sigma)

    mass = (1-fraction)*component(mu, s1)+fraction*component(mu+delta, s2)
    u = jnp.asarray(o.u)
    prediction = local[:, 0, None]+local[:, 1, None]*u+local[:, 2, None]*mass
    if o.baseline_degree == 2:
        prediction = prediction+local[:, 3, None]*u*u
    residual = prediction/jnp.asarray(o.peak[:, None])-jnp.asarray(o.y)
    inside, extra = jnp.asarray([o.inlier_base, o.extra_base])*jnp.exp(log_scale)
    variance = jnp.asarray(o.noise[..., None])**2+jnp.stack((inside**2, inside**2+extra**2))
    log_pixel = -.5*(residual[..., None]**2/variance+jnp.log(2*jnp.pi*variance))
    log_line = jnp.where(jnp.asarray(o.mask[..., None]), log_pixel, 0.).sum(axis=1)
    log_in, log_out = jnp.log1p(-epsilon)+log_line[:, 0], jnp.log(epsilon)+log_line[:, 1]
    loglik = jnp.logaddexp(log_in, log_out)
    numpyro.factor('spectral_likelihood', loglik.sum())
    numpyro.deterministic('coefficients', coefficients)
    numpyro.deterministic('local', local)
    numpyro.deterministic('shape', shape)
    numpyro.deterministic('inlier_probability', jnp.exp(log_in-loglik))
    numpyro.deterministic('mixture', jnp.stack((inside, jnp.hypot(inside, extra), epsilon)))


def _sample_mixture(o, theta, *, warmup, samples, chains, seed, target_accept,
                    max_tree_depth, progress):
    import jax
    jax.config.update('jax_enable_x64', True)
    from numpyro.infer import MCMC, NUTS, init_to_value
    from numpyro.diagnostics import summary
    local = theta[:o.nlocal].reshape(o.local_shape)
    initial = dict(coefficient_z=theta[o.nlocal:o.nlocal+o.ncoef].reshape(o.context['prior_mean'].shape),
        local_unit=np.clip((local-o.local_lower)/(o.local_upper-o.local_lower), 1e-5, 1-1e-5),
        epsilon=np.clip(theta[-3], 1e-4, 1-1e-4), log_scale=theta[-2:])
    kernel = NUTS(lambda: _numpyro_model(o), init_strategy=init_to_value(values=initial),
                  dense_mass=True, target_accept_prob=target_accept, max_tree_depth=max_tree_depth)
    sampler = MCMC(kernel, num_warmup=warmup, num_samples=samples, num_chains=chains,
                   chain_method='sequential', progress_bar=progress)
    started = perf_counter()
    sampler.run(jax.random.PRNGKey(seed), extra_fields=('diverging', 'num_steps', 'accept_prob'))
    extra = sampler.get_extra_fields()
    diagnostics = dict(seconds=perf_counter()-started, divergences=int(np.sum(extra['diverging'])),
        max_tree_depth_hits=int(np.sum(np.asarray(extra['num_steps']) >= 2**max_tree_depth-1)),
        mean_accept_prob=float(np.mean(extra['accept_prob'])), max_rhat=None, min_ess=None)
    if samples >= 4:
        grouped = sampler.get_samples(group_by_chain=True)
        stats = summary({key: grouped[key] for key in initial})
        for field, name, reduce in [('r_hat', 'max_rhat', np.max), ('n_eff', 'min_ess', np.min)]:
            values = np.concatenate([np.ravel(s[field]) for s in stats.values()])
            diagnostics[name] = float(reduce(values)) if np.isfinite(values).all() else None
    draws = {key: np.asarray(value) for key, value in sampler.get_samples().items()
             if key in ('coefficients', 'local', 'shape', 'inlier_probability', 'mixture')}
    return draws, diagnostics


def fit_chebyshev(input_data, *, degree, nuts, tophat, baseline_degree, warmup,
                  samples, chains, seed, target_accept, max_tree_depth, max_nfev,
                  maxiter, progress):
    """Independent LS comparison, joint spectral MAP, then optional NUTS."""
    from .results import FitResult, OrderFit
    data = [input_data] if isinstance(input_data, OrderData) else list(input_data)
    if not data:
        raise ValueError('No orders to fit')
    config = ChebyshevIP(degree=degree)
    if not isinstance(nuts, (bool, np.bool_)):
        raise ValueError('nuts must be a boolean')
    for name, value in [('maxiter', maxiter), ('max_nfev', max_nfev),
                        *([('warmup', warmup), ('samples', samples), ('chains', chains),
                           ('max_tree_depth', max_tree_depth)] if nuts else [])]:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if nuts and not 0 < target_accept < 1:
        raise ValueError('target_accept must be between 0 and 1')
    if len({d.order for d in data}) != len(data):
        raise ValueError('Order identifiers must be unique')
    objectives = []
    for d in sorted(data, key=lambda d: d.order):
        if (np.any(d.mask.sum(axis=1) < 1) or np.any(~np.isfinite(d.x[d.mask]))
                or np.any(~np.isfinite(d.y[d.mask])) or np.any(~np.isfinite(d.noise[d.mask]))
                or np.any(d.noise[d.mask] <= 0) or np.any(~np.isfinite(d.dv)) or np.any(d.dv <= 0)):
            raise ValueError('Need valid observations, positive noise, and positive sampling widths')
        objectives.append(_SpectralObjective(d, _prepare(d, config), tophat=tophat,
            baseline_degree=baseline_degree, inlier_scale_prior=.01, outlier_extra_prior=.05,
            log_scale_sd=(.7, .7), outlier_beta=(1., 4.)))
    orders = []
    for index, o in enumerate(objectives):
        d = o.data
        if progress:
            print(f'Order {d.order}: {len(d.x0)} lines; independent LS then degree {degree} spectral MAP'
                  + (' + NUTS' if nuts else ''))
        ilocal, ishape, success, individual_report = _independent_fits(d, tophat, baseline_degree, max_nfev)
        theta, point, map_report = _map_fit(o, maxiter, max_nfev)
        draws, nuts_report = {}, None
        if nuts:
            draws, nuts_report = _sample_mixture(o, theta, warmup=warmup, samples=samples,
                chains=chains, seed=(seed+index) % 2**32, target_accept=target_accept,
                max_tree_depth=max_tree_depth, progress=progress)
        curve = ChebyshevFit(coefficients=point['coefficients'], domain=o.context['domain'],
            prior_mean=o.context['prior_mean'], prior_scale=o.context['prior_scale'],
            observed_peak=o.peak, map_local=point['local'],
            map_inlier_probability=point['inlier_probability'],
            mixture_map=np.array([point['inlier_scale'], point['outlier_scale'], point['epsilon']]),
            individual_local=ilocal, individual_shape=ishape, individual_success=success,
            coefficient_samples=draws.get('coefficients', np.empty((0, degree+1, 5))),
            inlier_probability_samples=draws.get('inlier_probability', np.empty((0, len(d.x0)))),
            mixture_samples=draws.get('mixture', np.empty((0, 3))),
            diagnostics=dict(individual=individual_report, optimization=map_report, nuts=nuts_report))
        orders.append(OrderFit(d, draws.get('local', point['local'][None]),
            draws.get('shape', point['shape'][None]), model='lab_centered',
            baseline_degree=baseline_degree, chebyshev=curve))
        if progress:
            print(f"  individual {individual_report['seconds']:.2f}s; MAP {map_report['seconds']:.2f}s; "
                  f"{np.sum(curve.inlier_probability > .5)}/{len(d.x0)} lines with P(inlier) > 0.5")
            if nuts:
                print('  NUTS:', nuts_report)
    settings = dict(mode='chebyshev', model='lab_centered', tophat=bool(tophat),
        baseline_degree=int(baseline_degree), degree=int(degree), nuts=bool(nuts),
        inference='nuts' if nuts else 'map', seed=int(seed), max_nfev=int(max_nfev), maxiter=int(maxiter),
        individual_inference='multistart_least_squares', initial_scatter=.02,
        mixture=dict(inlier_scale_prior=.01, outlier_extra_prior=.05,
                     log_scale_sd=[.7, .7], outlier_beta=[1., 4.]),
        coefficient_prior=config.to_dict())
    if nuts:
        settings.update(warmup=int(warmup), samples=int(samples), chains=int(chains),
                        target_accept=float(target_accept), max_tree_depth=int(max_tree_depth))
    return FitResult(orders, settings)


