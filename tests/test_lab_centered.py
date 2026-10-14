"""Lab-referenced locations, ordered components, and compatible result handling."""
import json
import numpy as np
import pytest
from scipy.special import erf
from gaoesrvip import OrderData, fit, load_result
from gaoesrvip.fitting import _bounds, _optimize, _sample
from gaoesrvip.models import density, pixel_mass, prediction, profile_fwhm
from gaoesrvip.results import FitResult, OrderFit
from test_workflow import synthetic, initialized_result


def expected_mass(v, shape, dv, tophat):
    s1, f, s2, delta, mu1 = shape
    def gaussian(mu, sigma):
        if tophat:
            return .5*(erf((v+dv/2-mu)/(np.sqrt(2)*sigma))
                       - erf((v-dv/2-mu)/(np.sqrt(2)*sigma)))
        return dv*np.exp(-.5*((v-mu)/sigma)**2)/(np.sqrt(2*np.pi)*sigma)
    return (1-f)*gaussian(mu1, s1)+f*gaussian(mu1+delta, s2)


def lab_data(tophat=True, individual=False):
    table = np.array([3700.2, 3800.6, 3900.1])
    x = np.floor(table[:, None])+np.arange(-9., 11.)
    dv = np.array([1.1, 1.25, 1.4])
    shape = np.tile([1.35, .7, 1.65, 2.6, -.9], (3, 1))
    if individual:
        shape[1] = [1.6, .3, 1.1, 2.1, .25]
        shape[2] = [1.1, .6, 1.45, 2.9, -1.5]
    local = np.array([[12., .1, 1400.], [20., -.15, 1000.], [8., .05, 1800.]])
    y = np.array([p[0]+p[1]*(row-ref)+p[2]*expected_mass((row-ref)*scale, s, scale, tophat)
                  for row, ref, scale, p, s in zip(x, table, dv, local, shape)])
    data = OrderData(104, x, y, np.full_like(x, 2.), np.ones_like(x, bool),
                     table+np.array([.15, -.2, .3]), dv, ['a', 'b', 'c'],
                     lab_wave=[4900., 5000., 5100.], table_pixel=table)
    return data, local, shape


def known_result(mode='common', tophat=True, draws=1):
    data, local, shape = lab_data(tophat, mode == 'individual')
    local = np.repeat(local[None], draws, axis=0)
    shape = np.repeat((shape[0] if mode == 'common' else shape)[None], draws, axis=0)
    order = OrderFit(data, local, shape, model='lab_centered')
    return FitResult([order], dict(mode=mode, tophat=tophat, model='lab_centered'))


@pytest.mark.parametrize('tophat', [True, False])
def test_translated_profile_and_legacy_equivalence(tophat):
    shape = np.array([1.3, .72, 1.8, 2.5, -.8])
    dv, reference = 1.2, 3700.3
    v = np.linspace(-10, 10, 131)
    actual = pixel_mass(v, shape, dv, tophat=tophat, model='lab_centered')
    np.testing.assert_allclose(actual, expected_mass(v, shape, dv, tophat), atol=1e-15)
    np.testing.assert_allclose(density(v, shape, model='lab_centered'),
                               expected_mass(v, shape, 1., False), atol=1e-15)
    assert pixel_mass(np.arange(-40, 41)*dv, shape, dv, model='lab_centered').sum() == pytest.approx(1.)
    local = [15., .2, 2000.]
    x = reference+v/dv
    predicted = prediction(x, dv, local, shape, model='lab_centered', reference=reference, tophat=tophat)
    np.testing.assert_allclose(predicted, local[0]+local[1]*(x-reference)+local[2]*actual, atol=1e-9)
    # Exchange labels to express the stronger right-hand component as legacy G1.
    center = reference+(shape[4]+shape[3])/dv
    legacy_local = [local[0]+local[1]*(center-reference), local[1], local[2], center]
    legacy_shape = [shape[2], 1-shape[1], shape[0], -shape[3]]
    np.testing.assert_allclose(predicted, prediction(x, dv, legacy_local, legacy_shape, tophat=tophat), atol=1e-9)
    assert profile_fwhm(shape, model='lab_centered') == pytest.approx(profile_fwhm(legacy_shape))
    translated = shape.copy()
    translated[4] += 100.
    assert profile_fwhm(translated, dv=dv, model='lab_centered') == pytest.approx(
        profile_fwhm(shape, dv=dv, model='lab_centered'))


@pytest.mark.parametrize('mode', ['common', 'individual'])
@pytest.mark.parametrize('tophat', [True, False])
def test_initializer_recovers_ordered_profile(mode, tophat):
    data, expected_local, expected_shape = lab_data(tophat, mode == 'individual')
    local, shape, _ = _optimize(data, mode, tophat, 2000, model='lab_centered')
    if mode == 'common':
        expected_shape = expected_shape[0]
    np.testing.assert_allclose(local, expected_local, rtol=1e-4, atol=2e-3)
    np.testing.assert_allclose(shape, expected_shape, rtol=1e-4, atol=2e-3)
    assert np.all(shape[..., 3] > 0)
    assert np.max(shape[..., 1]) > .5


def test_common_has_no_independent_line_shifts():
    data, _, shapes = lab_data()
    shapes[:, 4] += [-.7, 0., .7]
    for i, (s, dv) in enumerate(zip(shapes, data.dv)):
        data.y[i] = 10.+1500.*expected_mass((data.x[i]-data.table_pixel[i])*dv, s, dv, True)
    residual_rms = {}
    for mode in ('individual', 'common'):
        local, shape, _ = _optimize(data, mode, True, 2000, model='lab_centered')
        order = OrderFit(data, local[None], shape[None], model='lab_centered')
        residual_rms[mode] = max(np.sqrt(np.mean(r**2)) for _, r in order.residual_lines(tophat=True))
    assert residual_rms['individual'] < 1e-5
    assert residual_rms['common'] > 1e-3


@pytest.mark.parametrize('mode', ['common', 'individual'])
@pytest.mark.parametrize('tophat', [True, False])
def test_numpyro_likelihood_uses_table_origin(mode, tophat, monkeypatch):
    """Check the actual sampling likelihood independently of the NumPy evaluator."""
    from numpyro import handlers
    data, local, shape = lab_data(tophat, mode == 'individual')
    if mode == 'common':
        shape = shape[0]
    lo, hi, slo, shi, _, _ = _bounds(data, 'lab_centered')
    # Exercise masked padding too: NaN pixels must not enter the likelihood.
    data.mask[:, -1] = False
    data.x[:, -1] = np.nan
    data.y[:, -1] = np.nan
    class CheckSampler:
        def __init__(self, kernel, **kwargs):
            self.model = kernel.model

        def run(self, key):
            substitute = handlers.substitute(self.model, data={
                'local_scaled': (local-lo)/(hi-lo), 'shape_scaled': (shape-slo)/(shi-slo)})
            trace = handlers.trace(handlers.seed(substitute, rng_seed=0)).get_trace()
            likelihood = trace['obs']['fn']
            np.testing.assert_allclose(np.asarray(likelihood.base_dist.loc)[data.mask],
                                       data.y[data.mask], rtol=1e-12, atol=1e-10)
            assert np.isfinite(np.asarray(likelihood.log_prob(trace['obs']['value']))).all()
            self.draws = {name: np.asarray(trace[name]['value'])[None] for name in ('local', 'shape')}

        def get_samples(self):
            return self.draws

    monkeypatch.setattr('numpyro.infer.MCMC', CheckSampler)
    _sample(data, mode, tophat, (lo, hi, slo, shi), local, shape, warmup=1, samples=1,
            chains=1, seed=0, target_accept=.95, max_tree_depth=4, progress=False, model='lab_centered')


@pytest.mark.parametrize('mode', ['common', 'individual'])
@pytest.mark.parametrize('tophat', [True, False])
def test_lab_mcmc_smoke(mode, tophat):
    data, _, _ = lab_data(tophat, mode == 'individual')
    # Short chains check wiring and dimensions, not posterior convergence.
    result = fit(data, mode=mode, model='lab_centered', tophat=tophat,
                 warmup=20, samples=8, chains=2, seed=23, max_tree_depth=4, progress=False)
    order = result.orders[0]
    assert order.local_samples.shape == (16, 3, 3)
    assert order.shape_samples.shape == ((16, 5) if mode == 'common' else (16, 3, 5))
    assert np.isfinite(order.local_samples).all() and np.isfinite(order.shape_samples).all()
    assert np.all(order.shape_samples[..., 3] > 0)
    assert np.all((order.shape_samples[..., 1] > 0) & (order.shape_samples[..., 1] < 1))
    assert result.settings['model'] == order.model == 'lab_centered'


@pytest.mark.parametrize('tophat', [True, False])
@pytest.mark.parametrize('wavelength_range, selected', [(None, [0, 1, 2]), ((5000, 5100), [1, 2]), ((5000, 5000), [1])])
def test_overlay_and_line_plots_share_table_origin(tophat, wavelength_range, selected):
    import matplotlib.pyplot as plt
    result = known_result(tophat=tophat)
    order = result.orders[0]
    d = order.data
    # Distinct residuals verify selection and normalization, not just zero residuals.
    d.y += np.array([15., 2., -3.])[:, None]
    before = order.local_samples.copy()
    lines = result.plot_lines(n_draws=0)[0]
    fig = result.overlay(wavelength_range=wavelength_range)
    top, bottom, zoom = fig.axes
    shape = order.shape_samples[0]
    total, first, second = top.lines[len(selected):len(selected)+3]
    dv = np.median(d.dv[selected])
    np.testing.assert_allclose(total.get_ydata(), expected_mass(total.get_xdata(), shape, dv, tophat)/dv)
    np.testing.assert_allclose(first.get_ydata()+second.get_ydata(), total.get_ydata())
    # The model peak is offset from the table wavelength and must stay offset.
    assert abs(total.get_xdata()[np.argmax(total.get_ydata())]) > .3
    residuals = []
    grid = zoom.lines[0].get_xdata()
    for j, i in enumerate(selected):
        observed = top.lines[j]
        np.testing.assert_array_equal(observed.get_xdata(), lines.axes[i].lines[0].get_xdata())
        np.testing.assert_array_equal(observed.get_ydata(), lines.axes[i].lines[0].get_ydata())
        expected_v = (d.x[i]-d.table_pixel[i])*d.dv[i]
        np.testing.assert_allclose(observed.get_xdata(), expected_v)
        residual = observed.get_ydata()-expected_mass(expected_v, shape, d.dv[i], tophat)/d.dv[i]
        np.testing.assert_allclose(bottom.lines[j].get_ydata(), residual, atol=1e-14)
        residuals.append(np.interp(grid, expected_v, residual, left=np.nan, right=np.nan))
    np.testing.assert_allclose(zoom.lines[0].get_ydata(), np.nanmedian(residuals, axis=0), atol=1e-14)
    assert zoom.get_xlabel() == 'Velocity from table wavelength [km/s]'
    np.testing.assert_array_equal(order.local_samples, before)
    plt.close(lines)
    plt.close(fig)


@pytest.mark.parametrize('mode', ['common', 'individual'])
@pytest.mark.parametrize('tophat', [True, False])
def test_posterior_curves_and_derived_locations(mode, tophat):
    import matplotlib.pyplot as plt
    result = known_result(mode, tophat, draws=3)
    order = result.orders[0]
    order.local_samples[:, 0, 0] = [10., 12., 16.]
    order.local_samples[:, 0, 2] = [1300., 1400., 1700.]
    profile = order.shape_for_line(0)
    profile[:, 4] = [-1.2, -.9, -.6]
    profile[:, 1] = [.5, .7, .6]
    profile[:, 3] = [2.4, 2.6, 3.]
    fig = result.plot_lines()[0]
    ax = fig.axes[0]
    v = ax.lines[1].get_xdata()
    b, slope, flux = np.median(order.local_samples[:, 0], axis=0)
    dv = order.data.dv[0]
    for k, params in enumerate(order.local_samples[:, 0]):
        expected = (params[0]-b+(params[1]-slope)*v/dv
                    + params[2]*expected_mass(v, profile[k], dv, tophat))/flux/dv
        assert sum(np.allclose(curve.get_ydata(), expected, atol=1e-12) for curve in ax.lines[4:-1]) == 1
    table = result.parameters
    if mode == 'individual':
        table = table.loc[table.line_id == 'a']
    for name, values in [('mu2_kms', profile[:, 4]+profile[:, 3]),
                         ('centroid_kms', profile[:, 4]+profile[:, 1]*profile[:, 3])]:
        row = table.loc[table.parameter == name, ['p16', 'median', 'p84']].iloc[0]
        np.testing.assert_allclose(row, np.percentile(values, [16, 50, 84]))
    assert 'mu1_kms' in order.shape_names and 'center' not in order.local_names
    assert set(result.line_parameters.parameter) == {'bkg', 'slope', 'total_flux'}
    plt.close(fig)


@pytest.mark.parametrize('mode', ['common', 'individual'])
def test_lab_roundtrip_with_figures(tmp_path, mode):
    result = known_result(mode)
    output = result.save(tmp_path/'new')
    restored = load_result(output)
    assert restored.settings['model'] == 'lab_centered'
    assert restored.orders[0].shape_names == result.orders[0].shape_names
    np.testing.assert_array_equal(restored.orders[0].local_samples, result.orders[0].local_samples)
    np.testing.assert_array_equal(restored.orders[0].shape_samples, result.orders[0].shape_samples)
    np.testing.assert_array_equal(restored.parameters, result.parameters)
    np.testing.assert_array_equal(restored.line_parameters, result.line_parameters)
    assert (output/('overlay.png' if mode == 'common' else 'fits_001.png')).exists()
    if mode == 'individual':
        with pytest.raises(ValueError, match='mode=common'):
            restored.overlay(wavelength_range=(4900, 5100))


def test_load_original_schema(tmp_path):
    result = initialized_result(synthetic())
    output = result.save(tmp_path/'old', figures=False)
    metadata_path = output/'result.json'
    metadata = json.loads(metadata_path.read_text())
    metadata['schema_version'] = 1
    del metadata['settings']['model']
    metadata_path.write_text(json.dumps(metadata))
    restored = load_result(output)
    assert restored.settings['model'] == restored.orders[0].model == 'free_centers'
    np.testing.assert_array_equal(restored.parameters, result.parameters)
    restored.save(tmp_path/'resaved', figures=False)


def test_model_validation_and_priors():
    data, _, _ = lab_data()
    with pytest.raises(ValueError, match='model must be'):
        fit(data, model='unknown')
    _, _, lo, hi, _, _ = _bounds(data, 'lab_centered')
    np.testing.assert_array_equal(lo[[0, 2]], [lo[0], lo[0]])
    np.testing.assert_array_equal(hi[[0, 2]], [hi[0], hi[0]])
    assert lo[1] == 0 and hi[1] == 1 and lo[3] == 0
    _, _, _, legacy_hi, _, _ = _bounds(data)
    assert legacy_hi[1] == .49
    data.table_pixel[1] = np.nan
    with pytest.raises(ValueError, match='finite table_pixel'):
        fit(data, model='lab_centered')
    data.table_pixel[1] = 3800.6
    data.dv[0] = -1.
    with pytest.raises(ValueError, match='positive dv'):
        fit(data, model='lab_centered')
