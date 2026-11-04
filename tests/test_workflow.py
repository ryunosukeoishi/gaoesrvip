"""Numerical and notebook workflow checks; no observational data required."""
import numpy as np
import pytest
from gaoesrvip import OrderData, fit, load_result
from gaoesrvip.models import density, pixel_mass, prediction, profile_fwhm


def synthetic(*, origin=0.0, shapes=None):
    x = np.tile(np.arange(-8, 9, dtype=float)+origin, (3, 1))
    dv = np.array([1.2, 1.3, 1.4])
    local = np.array([[12., .1, 1400.], [20., -.15, 1000.], [8., .05, 1800.]])
    if shapes is None:
        shapes = np.tile([1.45, .35, 1.2, 2.2, -.3], (3, 1))
    y = np.array([prediction(row, scale, p, s, reference=origin)
                  for row, scale, p, s in zip(x, dv, local, shapes)])
    return OrderData(104, x, y, np.full_like(x, 2.), np.ones_like(x, bool),
                     np.full(3, origin), dv, ['a', 'b', 'c'], table_pixel=np.full(3, origin))


def initialized_result(data, *, mode='common'):
    """Inspect the internal initializer without running MCMC in plotting tests."""
    from gaoesrvip.fitting import _optimize
    from gaoesrvip.results import FitResult, OrderFit
    orders = [data] if isinstance(data, OrderData) else list(data)
    results = []
    for d in orders:
        local, shape, _ = _optimize(d, mode, 2000)
        results.append(OrderFit(d, local[None], shape[None]))
    return FitResult(results, dict(mode=mode))


def test_pixel_center_density_and_area():
    from scipy.integrate import quad
    shape = np.array([1.5, .3, .9, 2., -.4])
    v = np.arange(-40, 41)*1.2
    expected = 1.2*(.7*np.exp(-.5*((v+.4)/1.5)**2)/(np.sqrt(2*np.pi)*1.5)
                    +.3*np.exp(-.5*((v-1.6)/.9)**2)/(np.sqrt(2*np.pi)*.9))
    np.testing.assert_allclose(pixel_mass(v, shape, 1.2), expected, atol=1e-14)
    assert quad(lambda v: density(v, shape), -40, 40)[0] == pytest.approx(1.)


def test_fwhm_definitions():
    shape = [1.5, .3, 1.5, 0., .2]
    assert profile_fwhm(shape) == pytest.approx(2*np.sqrt(2*np.log(2))*1.5, rel=1e-8)
    assert np.isnan(profile_fwhm([.3, .45, .3, 4., 0.]))


@pytest.mark.parametrize('mode', ['individual', 'common'])
def test_recovers_noise_free_profiles(mode):
    shapes = None if mode == 'common' else np.array([[1.45, .35, 1.2, 2.2, -.3],
                                                  [1.7, .25, 1.1, 2., .2],
                                                  [1.3, .4, 1.6, 2.5, -.1]])
    result = initialized_result(synthetic(origin=3700, shapes=shapes), mode=mode)
    residuals = result.orders[0].residual_lines()
    assert max(np.sqrt(np.mean(r**2)) for _, r in residuals) < 1e-5
    common_shapes = result.orders[0].shape_samples
    assert common_shapes.shape == ((1, 5) if mode == 'common' else (1, 3, 5))


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
    assert sampled.local_samples.shape == (10, 3, 3)
    assert sampled.shape_samples.shape == ((10, 5) if mode == 'common' else (10, 3, 5))
    assert np.isfinite(sampled.local_samples).all()
    assert np.isfinite(sampled.shape_samples).all()
    output = capsys.readouterr().out
    assert 'local_scaled[0,0]' in output and 'shape_scaled' in output
    assert 'n_eff' in output and 'r_hat' in output
    assert 'Number of divergences:' in output


def test_nointegration_multiple_chains_roundtrip(tmp_path):
    result = fit(synthetic(), warmup=20, samples=8, chains=2, seed=12, max_tree_depth=4, progress=False)
    assert result.orders[0].local_samples.shape == (16, 3, 3)
    result.save(tmp_path/'mcmc', figures=False)
    restored = load_result(tmp_path/'mcmc')
    np.testing.assert_array_equal(restored.orders[0].shape_samples, result.orders[0].shape_samples)
    np.testing.assert_allclose(restored.parameters['median'], result.parameters['median'])


def test_overlay_matches_pixel_model():
    """The display components sum to the observation-space prediction."""
    import matplotlib.pyplot as plt
    data = synthetic()
    result = initialized_result(data)
    fig = result.overlay()
    top, bottom, zoom = fig.axes
    n = len(data.x0)
    total, first, second = top.lines[n:n+3]
    shape = np.median(result.orders[0].shape_samples, axis=0)
    dv = np.median(data.dv)
    peak = 0.
    expected = pixel_mass(total.get_xdata()+peak, shape, dv)/dv
    np.testing.assert_allclose(total.get_ydata(), expected, atol=1e-14)
    np.testing.assert_allclose(first.get_ydata()+second.get_ydata(), expected, atol=1e-14)
    for i, (v, residual) in enumerate(result.orders[0].residual_lines()):
        curve = top.lines[i]
        np.testing.assert_allclose(curve.get_xdata(), v-peak)
        predicted = pixel_mass(v, shape, data.dv[i])/data.dv[i]
        np.testing.assert_allclose(curve.get_ydata()-predicted, residual, atol=1e-14)
    assert bottom.get_shared_x_axes().joined(top, bottom)
    assert zoom.get_shared_x_axes().joined(top, zoom)
    np.testing.assert_allclose(zoom.lines[0].get_ydata(), bottom.lines[n].get_ydata())
    assert not zoom.get_shared_y_axes().joined(bottom, zoom)
    plt.close(fig)


@pytest.mark.parametrize('wavelength_range, selected', [((5000, 5100), [1, 2]), ((5000, 5000), [1])])
def test_overlay_wavelength_selection(wavelength_range, selected):
    """Select displayed lines and residual statistics while retaining the fitted IP."""
    from dataclasses import replace
    import matplotlib.pyplot as plt
    data = synthetic()
    data.lab_wave = np.array([4900., 5000., 5100.])
    data.fit_wave = np.array([5000., 4900., 4900.])
    other = replace(data, order=105, lab_wave=np.array([6000., np.nan, 6200.]))
    result = initialized_result([other, data])
    order = result.orders[1]
    # Give the lines distinct nonzero residuals after fitting.
    data.y += np.array([100., 2., -3.])[:, None]
    local_before, shape_before = order.local_samples.copy(), order.shape_samples.copy()
    fig = result.overlay(wavelength_range=wavelength_range)
    assert len(fig.axes) == 3  # The first order has no matching lines.
    top, bottom, zoom = fig.axes
    n = len(selected)
    assert top.get_title() == f'Order 104 ({n} lines)'
    assert len(top.lines) == n+4
    assert len(bottom.lines) == n+2
    shape = np.median(shape_before, axis=0)
    dv = np.median(data.dv[selected])
    peak = 0.
    total, first, second = top.lines[n:n+3]
    expected = pixel_mass(total.get_xdata()+peak, shape, dv)/dv
    np.testing.assert_allclose(total.get_ydata(), expected, atol=1e-14)
    np.testing.assert_allclose(first.get_ydata()+second.get_ydata(), expected, atol=1e-14)
    residuals = order.residual_lines()
    grid = zoom.lines[0].get_xdata()
    interpolated = []
    for panel, i in enumerate(selected):
        v, observed = order.normalized_line(i)
        np.testing.assert_allclose(top.lines[panel].get_xdata(), v-peak)
        np.testing.assert_allclose(top.lines[panel].get_ydata(), observed)
        v, residual = residuals[i]
        np.testing.assert_allclose(bottom.lines[panel].get_xdata(), v-peak)
        np.testing.assert_allclose(bottom.lines[panel].get_ydata(), residual)
        interpolated.append(np.interp(grid, v-peak, residual, left=np.nan, right=np.nan))
    expected_median = np.nanmedian(interpolated, axis=0)
    np.testing.assert_allclose(zoom.lines[0].get_ydata(), expected_median)
    np.testing.assert_allclose(bottom.lines[n].get_ydata(), expected_median)
    assert len(result.orders) == 2
    np.testing.assert_array_equal(order.data.lab_wave, [4900., 5000., 5100.])
    np.testing.assert_array_equal(order.local_samples, local_before)
    np.testing.assert_array_equal(order.shape_samples, shape_before)
    plt.close(fig)


@pytest.mark.parametrize('bounds', [(5100, 5000), (np.nan, 5100), (5000, np.inf),
                                   (5000,), (5000, 5100, 5200), 5000])
def test_overlay_rejects_invalid_wavelength_range(bounds):
    from gaoesrvip.results import FitResult
    with pytest.raises(ValueError, match='wavelength_range must be two finite bounds'):
        FitResult([], dict(mode='common')).overlay(wavelength_range=bounds)


def test_overlay_no_matching_wavelengths():
    import matplotlib.pyplot as plt
    result = initialized_result(synthetic())
    result.orders[0].data.lab_wave = np.array([4900., np.nan, 5200.])
    figures_before = plt.get_fignums()
    with pytest.raises(ValueError, match='No lines within wavelength_range'):
        result.overlay(wavelength_range=(5000, 5100))
    assert plt.get_fignums() == figures_before


@pytest.mark.parametrize('mode', ['individual', 'common'])
def test_line_plots_match_observations(mode):
    import matplotlib.pyplot as plt
    data = synthetic()
    result = initialized_result(data, mode=mode)
    figures = result.plot_lines()
    assert len(figures) == 1
    assert figures[0]._suptitle.get_text() == f'Order {data.order}: {mode} 2G'
    order = result.orders[0]
    for i, (ax, (v, residual), dv) in enumerate(zip(figures[0].axes,
            order.residual_lines(), data.dv)):
        shape = order.shape_for_line(i)[0]
        observed = ax.lines[0]
        center = data.table_pixel[i]
        reference = data.table_pixel[i]
        np.testing.assert_allclose(observed.get_xdata(), v+(center-reference)*dv)
        predicted = pixel_mass(v, shape, dv)/dv
        np.testing.assert_allclose(observed.get_ydata()-predicted, residual, atol=1e-14)
        total = ax.lines[1]
        expected = pixel_mass(total.get_xdata()+(reference-center)*dv, shape, dv)/dv
        np.testing.assert_allclose(total.get_ydata(), expected, atol=1e-14)
    plt.close(figures[0])


@pytest.mark.parametrize('mode', ['individual', 'common'])
def test_fit_components_use_median_parameters(mode):
    """Use parameter medians consistently for model and normalization."""
    import matplotlib.pyplot as plt
    from gaoesrvip.results import FitResult, OrderFit
    d = synthetic().select([0])
    local = np.array([[[12., .1, 1400.]], [[25., -.12, 1700.]],
                      [[40., .2, 2400.]]])
    shape = np.array([[[1.2, .25, 1.8, 1., -.25]], [[2., .45, .8, 1.5, .5]],
                      [[2.8, .4, 1.4, 3., 1.2]]])
    samples = shape[:, 0] if mode == 'common' else shape
    r = FitResult([OrderFit(d, local, samples)], dict(mode=mode))
    fig = r.plot_lines()[0]
    observed, total, first, second = fig.axes[0].lines[:4]
    params, profile = np.median(local[:, 0], axis=0), np.median(shape[:, 0], axis=0)
    x = d.table_pixel[0]+total.get_xdata()/d.dv[0]
    continuum = params[0]+params[1]*(x-d.table_pixel[0])
    predicted = prediction(x, d.dv[0], params, profile, reference=d.table_pixel[0])
    np.testing.assert_allclose(total.get_ydata(), (predicted-continuum)/params[2]/d.dv[0], atol=1e-14)
    np.testing.assert_allclose(first.get_ydata()+second.get_ydata(), total.get_ydata(), atol=1e-14)
    xdata = d.x[0, d.mask[0]]
    baseline = params[0]+params[1]*(xdata-d.table_pixel[0])
    np.testing.assert_allclose(observed.get_ydata(), (d.y[0]-baseline)/params[2]/d.dv[0], atol=1e-14)
    posterior = fig.axes[0].lines[4:-1]
    assert len(posterior) == len(local)
    for draw_local, draw_shape in zip(local[:, 0], shape[:, 0]):
        predicted = prediction(x, d.dv[0], draw_local, draw_shape, reference=d.table_pixel[0])
        expected = (predicted-continuum)/params[2]/d.dv[0]
        assert sum(np.allclose(curve.get_ydata(), expected, atol=1e-14)
                   for curve in posterior) == 1
    plt.close(fig)


@pytest.mark.parametrize('mode', ['individual', 'common'])
def test_parameter_table_columns(mode):
    from gaoesrvip.results import FitResult, OrderFit
    d = synthetic()
    shape = np.ones((4, 3, 5) if mode == 'individual' else (4, 5))
    order = OrderFit(d, np.ones((4, 3, 3)), shape)
    order.intrinsic_fwhm_samples = np.full(shape.shape[:-1], 4.)
    r = FitResult([order], dict(mode=mode))
    assert ('line_id' in r.parameters.columns) == (mode == 'individual')
    assert 'undefined_draws' not in r.parameters.columns
