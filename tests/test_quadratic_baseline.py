"""Recover curved continua and keep likelihoods, diagnostics, and saved results consistent."""
import json
import numpy as np
import pytest
from gaoesrvip import fit, load_result
from gaoesrvip.fitting import _bounds, _optimize, _sample
from gaoesrvip.models import prediction
from gaoesrvip.results import FitResult, OrderFit
from test_lab_centered import lab_data
from test_residuals import residual_case


def quadratic_data(mode):
    q = np.array([.12, -.09, .18])
    d, local, shapes = lab_data(mode == 'individual')
    origin = d.table_pixel
    d.y += q[:, None]*(d.x-origin[:, None])**2
    local = np.column_stack((local, q))
    return d, local, shapes[0] if mode == 'common' else shapes


@pytest.mark.parametrize('mode', ['common', 'individual'])
def test_recover_quadratic_background_and_profile(mode):
    d, expected_local, expected_shape = quadratic_data(mode)
    local, shape, _ = _optimize(d, mode, 2000, baseline_degree=2)

    np.testing.assert_allclose(local, expected_local, rtol=2e-4, atol=2e-3)
    np.testing.assert_allclose(shape, expected_shape, rtol=2e-4, atol=2e-3)
    for i in range(len(d.x0)):
        predicted = prediction(d.x[i], d.dv[i], local[i], shape if mode == 'common' else shape[i], reference=d.table_pixel[i], baseline_degree=2)
        np.testing.assert_allclose(predicted, d.y[i], atol=2e-3)


@pytest.mark.parametrize('mode', ['common', 'individual'])
def test_quadratic_numpyro_likelihood(mode, monkeypatch):
    from numpyro import handlers
    d, local, shape = quadratic_data(mode)

    # The prior and likelihood must ignore padded/nonfinite pixels.
    d.mask[:, -1] = False
    d.x[:, -1] = d.y[:, -1] = np.nan
    lo, hi, slo, shi, _, _ = _bounds(d, baseline_degree=2)
    assert np.isfinite(lo).all() and np.isfinite(hi).all()
    assert np.all((local > lo) & (local < hi))

    class CheckSampler:
        def __init__(self, kernel, **kwargs):
            self.model = kernel.model

        def run(self, key):
            substitute = handlers.substitute(self.model, data={
                'local_scaled': (local-lo)/(hi-lo), 'shape_scaled': (shape-slo)/(shi-slo)})
            trace = handlers.trace(handlers.seed(substitute, rng_seed=0)).get_trace()
            likelihood = trace['obs']['fn']
            np.testing.assert_allclose(np.asarray(likelihood.base_dist.loc)[d.mask],
                                       d.y[d.mask], rtol=1e-11, atol=1e-9)
            assert np.isfinite(np.asarray(likelihood.log_prob(trace['obs']['value']))).all()
            self.draws = {name: np.asarray(trace[name]['value'])[None] for name in ('local', 'shape')}

        def get_samples(self):
            return self.draws

    monkeypatch.setattr('numpyro.infer.MCMC', CheckSampler)
    _sample(d, mode, (lo, hi, slo, shi), local, shape, warmup=1, samples=1, chains=1, seed=0, target_accept=.95, max_tree_depth=4, progress=False, baseline_degree=2)


@pytest.mark.parametrize('mode', ['individual', 'common'])
def test_quadratic_mcmc_smoke(mode):
    d, _, _ = quadratic_data(mode)
    result = fit(d, mode=mode, baseline_degree=2, warmup=20, samples=8, max_tree_depth=4, progress=False)
    order = result.orders[0]
    assert order.baseline_degree == result.settings['baseline_degree'] == 2
    assert order.local_samples.shape == (8, 3, 4)
    assert np.isfinite(order.local_samples).all()
    assert np.isfinite(result.residual_summary().rms_sigma).all()


@pytest.mark.parametrize('mode', ['common', 'individual'])
def test_quadratic_roundtrip_and_selected_overlay(tmp_path, mode):
    import matplotlib.pyplot as plt
    result, residual, _ = residual_case(mode, baseline_degree=2)
    output = result.save(tmp_path/'quadratic', figures=False)
    restored = load_result(output)
    assert restored.orders[0].local_names[-1] == 'quadratic'
    np.testing.assert_array_equal(restored.line_parameters, result.line_parameters)
    np.testing.assert_array_equal(restored.residual_summary(), result.residual_summary())
    np.testing.assert_array_equal(restored.orders[0].local_samples, result.orders[0].local_samples)
    if mode == 'common':
        fig = restored.overlay(wavelength_range=(5100, 5100))
        order = restored.orders[0]
        normalization = order.local_samples[0, 1, 2]*order.data.dv[1]
        np.testing.assert_allclose(fig.axes[1].lines[0].get_ydata(), residual[1]/normalization, atol=1e-14)
        plt.close(fig)


@pytest.mark.parametrize('schema', [2, 3])
def test_old_results_default_to_linear(tmp_path, schema):
    result, _, _ = residual_case('common')
    output = result.save(tmp_path/'linear', figures=False)
    path = output/'result.json'
    metadata = json.loads(path.read_text())
    metadata['schema_version'] = schema
    metadata['settings'].update(model='lab_centered', tophat=False)
    del metadata['settings']['baseline_degree']
    path.write_text(json.dumps(metadata))
    restored = load_result(output)
    assert restored.settings['baseline_degree'] == restored.orders[0].baseline_degree == 1
    np.testing.assert_array_equal(restored.residual_summary(), result.residual_summary())


@pytest.mark.parametrize('degree', [0, 3, 1.5, '2', True])
def test_invalid_degree(degree):
    with pytest.raises(ValueError, match='baseline_degree must be 1 or 2'):
        fit([], baseline_degree=degree)


def test_baseline_dimensions_and_metadata_must_agree():
    d, local, shape = quadratic_data('common')
    with pytest.raises(ValueError, match='dimensions'):
        OrderFit(d, local[None], shape[None])
    order = OrderFit(d, local[None], shape[None], baseline_degree=2)
    with pytest.raises(ValueError, match='baseline degrees'):
        FitResult([order], dict(mode='common'))
