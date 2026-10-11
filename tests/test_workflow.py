"""Numerical and notebook workflow checks; no observational data required."""
import numpy as np
import pytest
from scipy.special import erf
from gaoesrvip import OrderData, fit, load_result
from gaoesrvip.models import density, pixel_mass, prediction, profile_fwhm


def synthetic(*, tophat=True, origin=0., shapes=None):
    x = np.tile(np.arange(-8, 9, dtype=float)+origin, (3, 1))
    dv = np.array([1.2, 1.3, 1.4])
    local = np.array([[12., .1, 1400., -.25+origin], [20., -.15, 1000., .2+origin], [8., .05, 1800., .05+origin]])
    if shapes is None:
        shapes = np.tile([1.45, .35, 1.2, 2.2], (3, 1))
    y = np.array([prediction(row, scale, p, s, tophat=tophat)
                  for row, scale, p, s in zip(x, dv, local, shapes)])
    return OrderData(104, x, y, np.full_like(x, 2.), np.ones_like(x, bool),
                     np.full(3, origin), dv, ['a', 'b', 'c'], table_pixel=np.full(3, origin))


def initialized_result(data, *, mode='common', tophat=True):
    """Inspect the internal initializer without running MCMC in plotting tests."""
    from gaoesrvip.fitting import _optimize
    from gaoesrvip.results import FitResult, OrderFit
    orders = [data] if isinstance(data, OrderData) else list(data)
    results = []
    for d in orders:
        local, shape, _ = _optimize(d, mode, tophat, 2000)
        local = local.copy()
        local[:, 3] += d.x0
        results.append(OrderFit(d, local[None], shape[None]))
    return FitResult(results, dict(mode=mode, tophat=tophat))


def test_pixel_integration_and_area():
    shape = np.array([1.5, .3, .9, 2.])
    v = np.arange(-40, 41)*1.2
    mass = pixel_mass(v, shape, 1.2)
    assert mass.sum() == pytest.approx(1., abs=1e-12)
    # Independent pixel-space expression from the original ThAr model.
    x = v/1.2
    component = lambda mu, sigma: .5*(erf((x+.5-mu)/(np.sqrt(2)*sigma))
                                      - erf((x-.5-mu)/(np.sqrt(2)*sigma)))
    expected = .7*component(0, 1.5/1.2)+.3*component(2/1.2, .9/1.2)
    np.testing.assert_allclose(mass, expected, atol=1e-14)
    np.testing.assert_allclose(pixel_mass(v, shape, 1.2, tophat=False), 1.2*density(v, shape))


def test_fwhm_definitions():
    shape = [1.5, .3, 1.5, 0.]
    assert profile_fwhm(shape) == pytest.approx(2*np.sqrt(2*np.log(2))*1.5, rel=1e-8)
    assert profile_fwhm(shape, dv=1.5) > profile_fwhm(shape)
    assert np.isnan(profile_fwhm([.3, .45, .3, 4.]))


@pytest.mark.parametrize('tophat', [True, False])
@pytest.mark.parametrize('mode', ['individual', 'common'])
def test_recovers_noise_free_profiles(mode, tophat):
    shapes = None if mode == 'common' else np.array([[1.45, .35, 1.2, 2.2],
                                                  [1.7, .25, 1.1, -2.],
                                                  [1.3, .4, 1.6, 2.5]])
    result = initialized_result(synthetic(tophat=tophat, origin=3700, shapes=shapes),
                                mode=mode, tophat=tophat)
    residuals = result.orders[0].residual_lines(tophat=tophat)
    assert max(np.sqrt(np.mean(r**2)) for _, r in residuals) < 1e-5
    common_shapes = result.orders[0].shape_samples
    assert common_shapes.shape == ((1, 4) if mode == 'common' else (1, 3, 4))


@pytest.mark.parametrize('mode', ['individual', 'common'])
def test_roundtrip_and_padded_pixels(tmp_path, mode):
    data = synthetic()
    arrays = {name: np.pad(getattr(data, name), ((0, 0), (0, 2)),
                          constant_values=False if name == 'mask' else np.nan)
              for name in ('x', 'y', 'noise', 'mask')}
    data = OrderData(data.order, **arrays, x0=data.x0, dv=data.dv, line_ids=data.line_ids,
                     table_pixel=data.table_pixel)
    result = initialized_result(data, mode=mode)
    output = result.save(tmp_path/'result')
    reread = load_result(output)
    reread.save(tmp_path/'resaved', figures=False)
    np.testing.assert_allclose(reread.orders[0].local_samples, result.orders[0].local_samples)
    np.testing.assert_allclose(reread.orders[0].data.table_pixel, data.table_pixel)
    np.testing.assert_allclose(reread.parameters['median'], result.parameters['median'])
    name = 'overlay.png' if mode == 'common' else 'fits_001.png'
    assert (output/name).exists()
    assert not list(output.glob('*.pdf'))
    if mode == 'common':
        assert not list(output.glob('fits_*.png'))
        with pytest.raises(ValueError, match='overlay'):
            result.plot_lines()
    with pytest.raises(FileExistsError):
        result.save(output)


def test_input_validation():
    data = synthetic()
    with pytest.raises(ValueError):
        fit(data, mode='joint')


@pytest.mark.parametrize('mode', ['common', 'individual'])
def test_mcmc_smoke(mode, capsys):
    # Tiny chains verify sampling/serialization shapes only, not convergence.
    result = fit(synthetic(), mode=mode, warmup=20, samples=10, chains=1,
                 seed=11, max_tree_depth=4, progress=False)
    sampled = result.orders[0]
    assert sampled.local_samples.shape == (10, 3, 4)
    assert sampled.shape_samples.shape == ((10, 4) if mode == 'common' else (10, 3, 4))
    assert np.isfinite(sampled.local_samples).all()
    assert np.isfinite(sampled.shape_samples).all()
    output = capsys.readouterr().out
    assert 'local_scaled[0,0]' in output and 'shape_scaled' in output
    assert 'n_eff' in output and 'r_hat' in output
    assert 'Number of divergences:' in output


def test_nointegration_multiple_chains_roundtrip(tmp_path):
    result = fit(synthetic(tophat=False), tophat=False, warmup=20,
                 samples=8, chains=2, seed=12, max_tree_depth=4, progress=False)
    assert result.orders[0].local_samples.shape == (16, 3, 4)
    result.save(tmp_path/'mcmc', figures=False)
    restored = load_result(tmp_path/'mcmc')
    np.testing.assert_array_equal(restored.orders[0].shape_samples, result.orders[0].shape_samples)
    np.testing.assert_allclose(restored.parameters['median'], result.parameters['median'])


@pytest.mark.parametrize('tophat', [True, False])
def test_overlay_matches_pixel_model(tophat):
    """The display components sum to the observation-space prediction."""
    import matplotlib.pyplot as plt
    data = synthetic(tophat=tophat)
    result = initialized_result(data, tophat=tophat)
    fig = result.overlay()
    top, bottom, zoom = fig.axes
    n = len(data.x0)
    total, first, second = top.lines[n:n+3]
    shape = np.median(result.orders[0].shape_samples, axis=0)
    dv = np.median(data.dv)
    from gaoesrvip.plotting import _model_peak
    peak = _model_peak(shape, dv, tophat=tophat)
    expected = pixel_mass(total.get_xdata()+peak, shape, dv, tophat=tophat)/dv
    np.testing.assert_allclose(total.get_ydata(), expected, atol=1e-14)
    np.testing.assert_allclose(first.get_ydata()+second.get_ydata(), expected, atol=1e-14)
    assert abs(total.get_xdata()[np.argmax(total.get_ydata())]) <= .01
    for i, (v, residual) in enumerate(result.orders[0].residual_lines(tophat=tophat)):
        curve = top.lines[i]
        np.testing.assert_allclose(curve.get_xdata(), v-peak)
        predicted = pixel_mass(v, shape, data.dv[i], tophat=tophat)/data.dv[i]
        np.testing.assert_allclose(curve.get_ydata()-predicted, residual, atol=1e-14)
    assert bottom.get_shared_x_axes().joined(top, bottom)
    assert zoom.get_shared_x_axes().joined(top, zoom)
    np.testing.assert_allclose(zoom.lines[0].get_ydata(), bottom.lines[n].get_ydata())
    assert not zoom.get_shared_y_axes().joined(bottom, zoom)
    plt.close(fig)


@pytest.mark.parametrize('tophat', [True, False])
def test_individual_line_plots_match_observations(tophat):
    import matplotlib.pyplot as plt
    data = synthetic(tophat=tophat)
    result = initialized_result(data, mode='individual', tophat=tophat)
    figures = result.plot_lines()
    assert len(figures) == 1
    for i, (ax, (v, residual), dv, shape) in enumerate(zip(figures[0].axes,
            result.orders[0].residual_lines(tophat=tophat), data.dv,
            result.orders[0].shape_samples[0])):
        observed = ax.lines[0]
        center = np.median(result.orders[0].local_samples[:, i, 3])
        reference = data.table_pixel[i]
        np.testing.assert_allclose(observed.get_xdata(), v+(center-reference)*dv)
        predicted = pixel_mass(v, shape, dv, tophat=tophat)/dv
        np.testing.assert_allclose(observed.get_ydata()-predicted, residual, atol=1e-14)
    plt.close(figures[0])


@pytest.mark.parametrize('tophat', [True, False])
def test_fit_components_use_median_parameters(tophat):
    """Use parameter medians consistently for model and normalization."""
    import matplotlib.pyplot as plt
    from gaoesrvip.results import FitResult, OrderFit
    d = synthetic(tophat=tophat).select([0])
    local = np.array([[[12., .1, 1400., -.25]], [[25., -.12, 1700., .5]],
                      [[40., .2, 2400., 1.2]]])
    shape = np.array([[[1.2, .25, 1.8, 1.]], [[2., .45, .8, -1.5]],
                      [[2.8, .4, 1.4, 3.]]])
    r = FitResult([OrderFit(d, local, shape)],
                  dict(mode='individual', tophat=tophat))
    fig = r.plot_lines()[0]
    observed, total, first, second = fig.axes[0].lines[:4]
    params, profile = np.median(local[:, 0], axis=0), np.median(shape[:, 0], axis=0)
    x = d.table_pixel[0]+total.get_xdata()/d.dv[0]
    continuum = params[0]+params[1]*(x-params[3])
    predicted = prediction(x, d.dv[0], params, profile, tophat=tophat)
    np.testing.assert_allclose(total.get_ydata(), (predicted-continuum)/params[2]/d.dv[0], atol=1e-14)
    np.testing.assert_allclose(first.get_ydata()+second.get_ydata(), total.get_ydata(), atol=1e-14)
    xdata = d.x[0, d.mask[0]]
    baseline = params[0]+params[1]*(xdata-params[3])
    np.testing.assert_allclose(observed.get_ydata(), (d.y[0]-baseline)/params[2]/d.dv[0], atol=1e-14)
    plt.close(fig)


@pytest.mark.parametrize('mode', ['individual', 'common'])
def test_parameter_table_columns(mode):
    from gaoesrvip.results import FitResult, OrderFit
    d = synthetic()
    shape = np.ones((4, 3, 4) if mode == 'individual' else (4, 4))
    order = OrderFit(d, np.ones((4, 3, 4)), shape)
    order.intrinsic_fwhm_samples = np.full(shape.shape[:-1], 4.)
    r = FitResult([order], dict(mode=mode))
    assert ('line_id' in r.parameters.columns) == (mode == 'individual')
    assert 'undefined_draws' not in r.parameters.columns
