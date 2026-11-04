"""Degree selection, sampling stored MAPs, and portable coefficient definitions."""
from copy import deepcopy
from dataclasses import replace
import json
import numpy as np
import pandas as pd
import pytest
from gaoesrvip import fit, load_result, FitResult
from gaoesrvip.spectral_mixture import _objective_at_map
from test_chebyshev_mixture import synthetic


@pytest.fixture(scope='module')
def candidates():
    d = synthetic(4, contaminated=False)
    return {degree: fit(d, mode='chebyshev', degree=degree, progress=False)
            for degree in (0, 1)}


def check_export(result, definition):
    """Evaluate only the documented export fields, without library transforms."""
    assert definition['basis']['coefficient_axes'] == ['degree', 'coordinate']
    assert definition['wavelength_unit'] == 'Angstrom'
    for order in result.orders:
        spec = definition['orders'][str(order.data.order)]
        wave = np.linspace(*spec['wavelength_domain'], 13)
        x = (wave-spec['wavelength_center'])/spec['wavelength_scale']
        q = np.polynomial.chebyshev.chebval(x, spec['coefficients']).T
        shape = np.column_stack((np.exp(q[:, 2]), 1/(1+np.exp(-q[:, 4])),
                                 np.exp(q[:, 3]), np.exp(q[:, 1]), q[:, 0]))
        np.testing.assert_allclose(shape, result.predict_ip(wave, order=order.data.order, samples=False)[0])


def test_search_matches_manual_degree_loop(candidates, tmp_path):
    data = candidates[0].orders[0].data
    result = fit(data, mode='chebyshev', degrees=[1, 0, 1], progress=False)
    scores = result.degree_scores.set_index('degree')
    assert list(scores.index) == [0, 1]
    expected = {}
    for degree, candidate in candidates.items():
        loglik = candidate.diagnostics[104]['optimization']['log_likelihood']
        k = 5*(degree+1)+3*len(data.x0)+3
        expected[degree] = -2*loglik+k*np.log(data.mask.sum())
        assert scores.loc[degree, 'BIC_MAP'] == pytest.approx(expected[degree], abs=1e-9)
        assert scores.loc[degree, 'k'] == k
    winner = min(expected, key=expected.get)
    assert result.orders[0].chebyshev.degree == winner
    np.testing.assert_array_equal(result.orders[0].shape_samples, candidates[winner].orders[0].shape_samples)
    assert scores.loc[winner, 'selected'] and scores.loc[winner, 'delta_BIC_MAP'] == 0.
    assert result.settings['degree_by_order'] == {'104': winner}
    assert 'degree' not in result.settings and 'coefficient_prior' not in result.settings
    output = result.save(tmp_path/'selected', figures=False)
    loaded = load_result(output)
    pd.testing.assert_frame_equal(loaded.degree_scores, result.degree_scores)
    pd.testing.assert_frame_equal(pd.read_csv(output/'degree_scores.csv').fillna(''),
                                  result.degree_scores.fillna(''), check_dtype=False)
    check_export(loaded, json.loads((output/'ip_model.json').read_text()))
    assert not list(output.glob('prediction*'))


def test_search_records_failed_and_unsupported_candidates(candidates, monkeypatch, tmp_path):
    import gaoesrvip.spectral_mixture as sm
    def candidate(data, *, degree, **kwargs):
        if degree == 1:
            raise RuntimeError('deliberate optimization failure')
        return candidates[degree]
    monkeypatch.setattr(sm, 'fit_chebyshev', candidate)
    data = candidates[0].orders[0].data
    result = fit(data, mode='chebyshev', degrees=[0, 1, 9], progress=False)
    scores = result.degree_scores
    assert scores.success.tolist() == [True, False, False]
    assert scores.selected.tolist() == [True, False, False]
    assert scores.loc[1:, 'BIC_MAP'].isna().all()
    assert 'deliberate' in scores.loc[1, 'error']
    loaded = load_result(result.save(tmp_path/'failures', figures=False))
    pd.testing.assert_frame_equal(loaded.degree_scores, scores)
    with pytest.raises(RuntimeError, match='no candidate degree converged'):
        fit(data, mode='chebyshev', degrees=[1, 9], progress=False)


def test_ties_choose_smaller_degree_per_order(candidates, monkeypatch):
    import gaoesrvip.spectral_mixture as sm
    d = candidates[0].orders[0].data
    def candidate(data, *, degree, **kwargs):
        result = deepcopy(candidates[degree])
        result.orders[0].data = data
        curve = result.orders[0].chebyshev
        k = 5*(degree+1)+3*len(data.x0)+3
        # Equal scores for 104; degree 1 is decisively better for 105.
        target = 0. if data.order == 104 else -100.*degree
        curve.diagnostics['optimization']['log_likelihood'] = (k*np.log(data.mask.sum())-target)/2
        return result
    monkeypatch.setattr(sm, 'fit_chebyshev', candidate)
    result = fit([replace(d, order=105), d], mode='chebyshev', degrees=[1, 0], progress=False)
    assert result.settings['degree_by_order'] == {'104': 0, '105': 1}
    assert [o.chebyshev.degree for o in result.orders] == [0, 1]


@pytest.mark.parametrize('degrees', [[], 2, [-1], [True], [1.2]])
def test_invalid_search_degrees(degrees, candidates):
    with pytest.raises(ValueError):
        fit(candidates[0].orders[0].data, mode='chebyshev', degrees=degrees, progress=False)


def test_search_is_chebyshev_only():
    with pytest.raises(ValueError, match='mode=chebyshev'):
        fit([], mode='common', degrees=[0, 1])


def test_stored_map_coordinates_with_quadratic_baseline():
    result = fit(synthetic(4, baseline_degree=2), mode='chebyshev', degree=1,
                 baseline_degree=2, progress=False)
    order = result.orders[0]
    objective, theta = _objective_at_map(order, result.settings)
    value, _, point = objective.evaluate(theta)
    report = result.diagnostics[104]['optimization']
    assert value == pytest.approx(report['objective'], abs=1e-8)
    assert point['log_likelihood'].sum() == pytest.approx(report['log_likelihood'], abs=1e-8)
    np.testing.assert_allclose(point['local'], order.chebyshev.map_local)
    np.testing.assert_allclose(point['shape'], order.shape_samples[0])


def test_sample_loaded_mixed_degrees_without_refitting(candidates, monkeypatch, tmp_path):
    import gaoesrvip.spectral_mixture as sm
    orders = [deepcopy(candidates[0].orders[0]), deepcopy(candidates[1].orders[0])]
    orders[1].data = replace(orders[1].data, order=105)
    settings = dict(candidates[0].settings, degree_by_order={'104': 0, '105': 1})
    settings.pop('degree')
    settings.pop('coefficient_prior')
    result = FitResult(orders, settings)
    output = result.save(tmp_path/'map', figures=False)
    loaded = load_result(output)
    before = loaded.parameters.copy()  # Sampling must not reuse cached MAP summaries.
    check_export(loaded, loaded.ip_model)
    for order in loaded.orders:
        objective, theta = sm._objective_at_map(order, loaded.settings)
        _, _, point = objective.evaluate(theta)
        report = loaded.diagnostics[order.data.order]['optimization']
        assert point['log_likelihood'].sum() == pytest.approx(report['log_likelihood'], abs=1e-8)

    def forbidden(*args, **kwargs):
        raise AssertionError('Sampling repeated an optimization')
    monkeypatch.setattr(sm, '_map_fit', forbidden)
    monkeypatch.setattr(sm, '_independent_fits', forbidden)
    sampled = loaded.sample_nuts(warmup=8, samples=6, chains=1, seed=17,
                                 max_tree_depth=3, progress=False)
    assert sampled.settings['inference'] == 'nuts'
    assert sampled.settings['degree_by_order'] == {'104': 0, '105': 1}
    assert [o.chebyshev.coefficient_samples.shape for o in sampled.orders] == [(6, 1, 5), (6, 2, 5)]
    for original, new in zip(loaded.orders, sampled.orders):
        assert not original.chebyshev.has_posterior
        np.testing.assert_array_equal(original.chebyshev.coefficients, new.chebyshev.coefficients)
        np.testing.assert_array_equal(original.chebyshev.individual_shape, new.chebyshev.individual_shape)
    pd.testing.assert_frame_equal(loaded.parameters, before)
    assert sampled.parameters.p16.notna().any()
    saved = sampled.save(tmp_path/'nuts', figures=False)
    restored = load_result(saved)
    check_export(restored, json.loads((saved/'ip_model.json').read_text()))
    assert restored.ip_model['orders']['105']['coefficient_estimate'] == 'posterior_median'


def test_search_nuts_runs_only_selected_maps(candidates, monkeypatch):
    import gaoesrvip.spectral_mixture as sm
    sampled = []
    def candidate(data, *, degree, nuts, **kwargs):
        assert nuts is False
        return candidates[degree]
    def sample(result, **kwargs):
        sampled.append(result)
        return result
    monkeypatch.setattr(sm, 'fit_chebyshev', candidate)
    monkeypatch.setattr(sm, 'sample_chebyshev_result', sample)
    result = fit(candidates[0].orders[0].data, mode='chebyshev', degrees=[0, 1],
                 nuts=True, warmup=5, samples=5, progress=False)
    assert sampled == [result]
    assert result.degree_scores.selected.sum() == 1


def test_residual_plot_keeps_observed_peak_for_negligible_fitted_flux(candidates):
    import matplotlib.pyplot as plt
    result = deepcopy(candidates[0])
    result.orders[0].local_samples[:, 0, 2] = 1e-8
    summary = result.residual_summary()
    assert summary.rms_peak_percent.iloc[0] > 1e8
    assert summary.rms_observed_peak_percent.iloc[0] < 100
    fig = result.plot_residual_summary()
    np.testing.assert_allclose(fig.axes[0].lines[0].get_ydata(), summary.rms_observed_peak_percent)
    assert fig.axes[0].get_ylabel() == 'Residual RMS / observed peak [%]'
    np.testing.assert_allclose(fig.axes[1].lines[0].get_ydata(), summary.rms_sigma)
    plt.close(fig)
