"""Spectral probability, optimizer/sampler agreement, portable results and plots."""
from copy import deepcopy
from dataclasses import replace
import numpy as np
import pytest
import matplotlib.pyplot as plt
from scipy.optimize._numdiff import approx_derivative
from gaoesrvip import OrderData, fit, load_result, save_analysis_figures
from gaoesrvip.models import prediction
from gaoesrvip.chebyshev import ChebyshevIP, _prepare, _coordinates_to_shape, cumulants
from gaoesrvip.spectral_mixture import _SpectralObjective, _numpyro_model, _initial_residual_jacobian


def synthetic(n=7, *, baseline_degree=1, contaminated=True):
    rng = np.random.default_rng(12)
    wave = np.linspace(5000., 5100., n)
    x = np.tile(np.arange(-9., 10.), (n, 1))
    dv = np.full(n, 1.2)
    q = np.tile([-.4, np.log(1.6), np.log(1.3), np.log(1.7), -.5], (n, 1))
    q[:, 2] += np.linspace(-.04, .04, n)
    shape = _coordinates_to_shape(q)
    local = np.tile([25., .1, 4000.] + ([.015] if baseline_degree == 2 else []), (n, 1))
    y = np.array([prediction(xi, step, li, si, reference=0., baseline_degree=baseline_degree)
                  for xi, step, li, si in zip(x, dv, local, shape)])
    y += rng.normal(0., 2., y.shape)
    if contaminated:
        y[n//2] += 180*np.exp(-.5*((x[n//2]*dv[n//2]-5.5)/.7)**2)
    return OrderData(104, x, y, np.full_like(y, 2.), np.ones_like(y, bool),
                     np.zeros(n), dv, np.arange(n).astype(str), wave, wave, np.zeros(n))


def objective(data, *, degree=1, baseline_degree=1):
    return _SpectralObjective(data, _prepare(data, ChebyshevIP(degree)), baseline_degree=baseline_degree, inlier_scale_prior=.01, outlier_extra_prior=.05, log_scale_sd=(.7, .7), outlier_beta=(1., 4.))


@pytest.fixture(scope='module')
def fitted():
    return fit(synthetic(), mode='chebyshev', degree=1, progress=False)


@pytest.mark.parametrize('baseline_degree', [1, 2])
def test_gradient_and_numpyro_density_match(baseline_degree):
    import jax
    jax.config.update('jax_enable_x64', True)
    from numpyro.infer.util import log_density
    d = synthetic(4, baseline_degree=baseline_degree)
    d.mask[0, -2:] = False
    d.y[~d.mask] = np.nan
    o = objective(d, baseline_degree=baseline_degree)
    theta = o.initial(o.context['prior_mean'], epsilon=.2, log_scale=np.array([.1, -.1]))
    value, grad = o(theta)
    numeric = approx_derivative(lambda t: o(t)[0], theta, method='3-point').ravel()
    np.testing.assert_allclose(grad, numeric, rtol=2e-4, atol=2e-4)
    residual, jac = _initial_residual_jacobian(theta[:-3], o, .02)
    np.testing.assert_allclose(jac, approx_derivative(lambda t: _initial_residual_jacobian(t, o, .02)[0],
        theta[:-3]), rtol=2e-4, atol=2e-4)

    def numpyro_value(t):
        local = t[:o.nlocal].reshape(o.local_shape)
        params = dict(coefficient_z=t[o.nlocal:o.nlocal+o.ncoef].reshape(o.context['prior_mean'].shape),
            local_unit=(local-o.local_lower)/(o.local_upper-o.local_lower),
            epsilon=t[-3], log_scale=t[-2:])
        lp, trace = log_density(lambda: _numpyro_model(o), (), {}, params)
        return float(lp), trace
    lp, trace = numpyro_value(theta)
    modified = theta.copy()
    modified[o.nlocal+2] += .03
    modified[-3] += .02
    lp2, _ = numpyro_value(modified)
    assert lp2-lp == pytest.approx(value-o(modified)[0], abs=1e-8)
    np.testing.assert_allclose(trace['inlier_probability']['value'],
                               o.evaluate(theta)[2]['inlier_probability'], atol=1e-12)


def test_direct_fit_membership_and_curves(fitted):
    o = fitted.orders[0]
    c = o.chebyshev
    assert fitted.settings['inference'] == 'map'
    assert not c.has_posterior
    assert c.individual_success.all()
    assert c.inlier_probability[3] < .01
    assert np.median(np.delete(c.inlier_probability, 3)) > .9
    assert c.mixture_map[1] > c.mixture_map[0] > 0
    np.testing.assert_allclose(fitted.predict_ip(o.data.lab_wave, order=104), o.shape_samples)
    assert fitted.predict_ip(np.linspace(5000, 5100, 51), order=104).shape == (1, 51, 5)
    assert fitted.parameters[['p16', 'p84']].isna().all().all()
    with pytest.raises(ValueError, match='outside fitted domain'):
        fitted.predict_ip([4999], order=104)


def test_failed_individual_does_not_remove_line(monkeypatch):
    from gaoesrvip import fitting
    original = fitting._optimize_block
    def fail_one(data, *args, **kwargs):
        if data.line_ids[0] == '1':
            raise RuntimeError('deliberate local fit failure')
        return original(data, *args, **kwargs)
    monkeypatch.setattr(fitting, '_optimize_block', fail_one)
    r = fit(synthetic(4), mode='chebyshev', degree=1, progress=False)
    assert len(r.line_membership) == 4
    assert not r.orders[0].chebyshev.individual_success[1]
    assert np.isnan(r.orders[0].chebyshev.individual_shape[1]).all()
    assert np.isfinite(r.orders[0].shape_samples).all()


def test_cumulants_gaussian_and_translation():
    shape = np.array([1.4, .3, 1.4, 0., 2.])
    np.testing.assert_allclose(cumulants(shape), [2, 1.4**2, 0, 0, 0], atol=1e-12)
    shape = np.array([1.1, .4, 1.8, 2., -.3])
    shifted = shape.copy()
    shifted[4] += 7
    np.testing.assert_allclose(cumulants(shifted)-cumulants(shape), [7, 0, 0, 0, 0], atol=1e-11)


def test_portable_map_results_and_figures(fitted, tmp_path):
    r = deepcopy(fitted)
    # Two orders exercise per-order prediction, figure layout and save/load.
    r.orders.append(replace(deepcopy(r.orders[0]), data=replace(deepcopy(r.orders[0].data), order=105)))
    output = r.save(tmp_path/'result', figures=False)
    restored = load_result(output)
    assert restored.settings == r.settings
    assert restored.diagnostics == r.diagnostics
    for original, loaded in zip(r.orders, restored.orders):
        for name in original.chebyshev.array_fields:
            np.testing.assert_equal(getattr(loaded.chebyshev, name), getattr(original.chebyshev, name))
    assert len(restored.individual_parameters) == 14
    before = set(plt.get_fignums())
    save_analysis_figures(restored, tmp_path/'figures', dpi=30)
    assert set(plt.get_fignums()) == before
    for order in (104, 105):
        assert (tmp_path/f'figures/cumulants_order_{order}.png').exists()
        assert (tmp_path/f'figures/overlay_inliers_order_{order}.png').exists()
    for name in ('residual_summary', 'rms_vs_peak', 'cumulants_all_orders'):
        assert (tmp_path/f'figures/{name}.png').exists()
    assert len(list((tmp_path/'figures').glob('lines_order_*.png'))) == 2
    fig = restored.overlay(min_inlier_probability=.5, wavelength_range=(5000, 5050))
    assert '3 lines' in fig.axes[0].get_title()
    plt.close(fig)
    with pytest.raises(ValueError, match='No lines'):
        restored.overlay(min_inlier_probability=1.)
    assert len(restored.line_membership) == 14  # Plot filtering never changes fit membership.
    fig = restored.plot_rms_vs_peak()
    metrics = restored.residual_summary()
    np.testing.assert_allclose(fig.axes[0].collections[0].get_offsets(),
                              metrics[['observed_peak', 'rms_observed_peak_percent']])
    np.testing.assert_allclose(fig.axes[0].collections[0].get_array(), metrics.inlier_probability)
    assert fig.axes[0].get_xscale() == fig.axes[0].get_yscale() == 'log'
    plt.close(fig)


def test_short_nuts_and_credible_bands(tmp_path):
    # Smoke test of the real sampler, not a convergence/accuracy claim.
    result = fit(synthetic(3, contaminated=False), mode='chebyshev', degree=0,
                 nuts=True, warmup=10, samples=6, chains=1, max_tree_depth=3, progress=False)
    c = result.orders[0].chebyshev
    assert c.coefficient_samples.shape == (6, 1, 5)
    assert c.has_posterior
    np.testing.assert_allclose(result.line_membership.inlier_probability,
                               c.inlier_probability_samples.mean(axis=0))
    loaded = load_result(result.save(tmp_path/'nuts', figures=False))
    np.testing.assert_equal(loaded.predict_ip([5050], order=104), result.predict_ip([5050], order=104))
    fig = loaded.plot_cumulants(n_positions=20, zoom_outliers=False)
    bands = [item for item in fig.axes[0].collections if type(item).__name__ == 'FillBetweenPolyCollection']
    assert len(bands) == 1
    plt.close(fig)


@pytest.mark.parametrize('kwargs', [dict(degree=-1), dict(degree=10), dict(degree=True),
    dict(nuts='yes'), dict(maxiter=0), dict(nuts=True, samples=0)])
def test_invalid_settings_fail_before_fit(kwargs, monkeypatch):
    import gaoesrvip.spectral_mixture as sm
    def forbidden(*args, **kwargs):
        raise AssertionError('individual fits ran before validation')
    monkeypatch.setattr(sm, '_independent_fits', forbidden)
    with pytest.raises(ValueError):
        fit(synthetic(3), mode='chebyshev', progress=False, **kwargs)


def test_analysis_chebyshev_without_corner(tmp_path, monkeypatch, fitted):
    import gaoesrvip.analysis as analysis
    monkeypatch.setattr(analysis, 'select_lines', lambda *args, **kw: [fitted.orders[0].data])
    monkeypatch.setattr(analysis, 'fit', lambda *args, **kw: deepcopy(fitted))
    monkeypatch.setattr(analysis, '_require_corner', lambda: pytest.fail('Chebyshev must not require corner'))
    a, b = tmp_path/'thar', tmp_path/'ec'
    a.touch(); b.touch()
    r = analysis.analyze_thar_ip_2g(a, b, tmp_path/'analysis', orders=[104], gain=2.,
                                   read_noise=4., mode='chebyshev', degree=1)
    assert r.settings['mode'] == 'chebyshev'
    assert load_result(tmp_path/'analysis').orders[0].chebyshev is not None
