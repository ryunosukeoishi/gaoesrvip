"""Complete notebook exports, reload/replot, and checks before expensive fitting."""
from dataclasses import replace
import sys

import matplotlib.pyplot as plt
import numpy as np
import pytest

from gaoesrvip import analyze_thar_ip_2g, load_result, save_analysis_figures
from gaoesrvip.results import FitResult
from test_lab_centered import known_result


def sampled_result(counts=(3, 1)):
    result = known_result(draws=128)
    original = result.orders[0]
    rng = np.random.default_rng(7)
    orders = []
    for number, count in enumerate(counts, 104):
        indices = np.arange(count) % len(original.data.x0)
        data = replace(original.data.select(indices), order=number,
                       line_ids=np.arange(count).astype(str))
        shapes = original.shape_samples+rng.normal(0, .02, original.shape_samples.shape)
        orders.append(replace(original, data=data, shape_samples=shapes,
                              local_samples=original.local_samples[:, indices]))
    return FitResult(orders, result.settings)


def test_all_figures_after_reload_including_every_order_and_page(tmp_path):
    pytest.importorskip('corner')
    # 41 lines require two pages; a second order must get its own plots too.
    result = sampled_result((41, 1))
    result.save(tmp_path/'result', figures=False)
    restored = load_result(tmp_path/'result')
    existing = plt.figure()
    before = set(plt.get_fignums())
    try:
        with plt.ion():
            directory = save_analysis_figures(restored, tmp_path/'figures', dpi=40)
            assert plt.isinteractive()
        assert set(plt.get_fignums()) == before
        assert {p.name for p in directory.iterdir()} == {
            'lines_order_104_001.png', 'lines_order_104_002.png',
            'lines_order_105_001.png', 'overlay.png',
            'corner_order_104.png', 'corner_order_105.png', 'residual_summary.png',
        }
        for path in directory.iterdir():
            pixels = plt.imread(path)
            assert pixels.shape[0] > 50 and pixels.shape[1] > 50
            assert pixels[..., :3].std() > .01
    finally:
        plt.close(existing)


def test_analysis_saves_result_and_passes_notebook_defaults_and_options(tmp_path, monkeypatch):
    pytest.importorskip('corner')
    result = sampled_result()
    thar, wavelength = tmp_path/'thar.fits', tmp_path/'wavelength.db'
    thar.touch()
    wavelength.touch()
    summed = tmp_path/'sum.fits'
    summed.touch()
    selected = [o.data for o in result.orders]
    calls = {}

    def select(path, wavelength_file, **kwargs):
        assert path == thar and wavelength_file == wavelength
        calls['selection'] = kwargs
        return selected

    def fit(data, **kwargs):
        assert data is selected
        calls['fit'] = kwargs
        return FitResult(result.orders, kwargs)

    monkeypatch.setattr('gaoesrvip.analysis.select_lines', select)
    monkeypatch.setattr('gaoesrvip.analysis.fit', fit)
    output = tmp_path/'run'
    actual = analyze_thar_ip_2g(thar, wavelength, output, orders=[104, 105],
                                gain=1.935, read_noise=4., half_window=9,
                                warmup=10, samples=128, chains=2, seed=13, progress=False,
                                fit_spectrum=summed)
    assert calls['selection']['fit_spectrum'] == summed
    assert calls['selection']['saturation'] == 5000
    assert calls['selection']['half_window'] == 9
    assert calls['selection']['gain'] == 1.935
    assert calls['selection']['read_noise'] == 4.
    assert calls['selection']['orders'] == [104, 105]
    assert calls['fit']['mode'] == 'common'
    assert 'model' not in calls['fit']
    assert 'tophat' not in calls['fit']
    assert calls['fit']['baseline_degree'] == 1
    assert calls['fit']['warmup'] == 10
    assert calls['fit']['samples'] == 128
    assert calls['fit']['chains'] == 2
    assert calls['fit']['seed'] == 13
    assert calls['fit']['progress'] is False
    restored = load_result(output)
    assert restored.settings == actual.settings
    for source, saved in zip(actual.orders, restored.orders):
        np.testing.assert_array_equal(source.shape_samples, saved.shape_samples)
    assert {p.name for p in output.iterdir()} == {
        'result.json', 'parameters.csv', 'line_parameters.csv', 'residual_summary.csv',
        'posterior_order_104.npz', 'posterior_order_105.npz', 'figures',
    }
    assert len(list((output/'figures').glob('*.png'))) == 6


@pytest.mark.parametrize('problem', ['nonempty', 'output_file', 'missing_input', 'missing_corner'])
def test_preflight_before_selection_or_fit(tmp_path, monkeypatch, problem):
    def unexpected(*args, **kwargs):
        pytest.fail('Selection or MCMC ran despite a preflight error')

    monkeypatch.setattr('gaoesrvip.analysis.select_lines', unexpected)
    monkeypatch.setattr('gaoesrvip.analysis.fit', unexpected)
    thar, wavelength = tmp_path/'thar.fits', tmp_path/'wavelength.db'
    thar.touch()
    wavelength.touch()
    output = tmp_path/'output'
    if problem == 'nonempty':
        output.mkdir()
        (output/'keep.txt').write_text('original')
        error = FileExistsError
    elif problem == 'output_file':
        output.write_text('original')
        error = NotADirectoryError
    elif problem == 'missing_input':
        wavelength.unlink()
        error = FileNotFoundError
    else:
        monkeypatch.setitem(sys.modules, 'corner', None)
        error = ImportError
    with pytest.raises(error):
        analyze_thar_ip_2g(thar, wavelength, output, orders=[104], gain=1.935, read_noise=4.)
    if problem == 'nonempty':
        assert (output/'keep.txt').read_text() == 'original'
    elif problem == 'output_file':
        assert output.read_text() == 'original'
    else:
        assert not output.exists()


def test_plot_failure_keeps_saved_samples_and_closes_only_new_figures(tmp_path, monkeypatch):
    pytest.importorskip('corner')
    result = sampled_result((1,))
    thar, wavelength = tmp_path/'thar.fits', tmp_path/'wavelength.db'
    thar.touch()
    wavelength.touch()
    monkeypatch.setattr('gaoesrvip.analysis.select_lines', lambda *args, **kwargs: [result.orders[0].data])
    monkeypatch.setattr('gaoesrvip.analysis.fit', lambda *args, **kwargs: result)

    def failed_plot(*args, **kwargs):
        plt.figure()
        raise RuntimeError('plot failed')

    monkeypatch.setattr('corner.corner', failed_plot)
    existing = plt.figure()
    before = set(plt.get_fignums())
    try:
        with pytest.raises(RuntimeError, match='plot failed'):
            analyze_thar_ip_2g(thar, wavelength, tmp_path/'run', orders=[104], gain=1.935, read_noise=4.)
        assert set(plt.get_fignums()) == before
        restored = load_result(tmp_path/'run')
        np.testing.assert_array_equal(restored.orders[0].shape_samples, result.orders[0].shape_samples)
    finally:
        plt.close(existing)


def test_figure_export_checks_mode_and_protects_existing_directory(tmp_path):
    with pytest.raises(ValueError, match='mode=common'):
        save_analysis_figures(known_result(mode='individual'), tmp_path/'individual')
    result = sampled_result()
    output = tmp_path/'figures'
    output.mkdir()
    (output/'keep.txt').write_text('original')
    with pytest.raises(FileExistsError):
        save_analysis_figures(result, output)
    assert (output/'keep.txt').read_text() == 'original'


@pytest.mark.parametrize('options, expected', [
    ({'max_degree': 0}, {'degrees': [0]}),
    ({'max_degree': np.int64(3)}, {'degrees': [0, 1, 2, 3]}),
    ({'degree': 0}, {'degree': 0}),
    ({}, {'degree': 2}),
])
@pytest.mark.parametrize('nuts', [False, True])
def test_chebyshev_degree_options_and_plots_are_forwarded(tmp_path, monkeypatch, options, expected, nuts):
    import gaoesrvip.analysis as analysis
    calls = {}
    class Result:
        def save(self, output, *, figures):
            calls['save'] = output, figures
    result = Result()
    def selected(*args, **kwargs):
        calls['selection'] = kwargs
        return ['selected windows']
    def fitted(data, **kwargs):
        assert data == ['selected windows']
        calls['fit'] = kwargs
        return result
    def figures(actual, output, **kwargs):
        assert actual is result
        calls['plots'] = output, kwargs
    monkeypatch.setattr(analysis, 'select_lines', selected)
    monkeypatch.setattr(analysis, 'fit', fitted)
    monkeypatch.setattr(analysis, 'save_analysis_figures', figures)
    monkeypatch.setattr(analysis, '_require_corner', lambda: pytest.fail('Chebyshev requires no corner'))
    a, b = tmp_path/'thar', tmp_path/'ec'
    a.touch(); b.touch()
    actual = analyze_thar_ip_2g(a, b, tmp_path/'result', orders=[104], gain=1., read_noise=4.,
        mode='chebyshev', nuts=nuts, warmup=8, samples=6, seed=31,
        min_inlier_probability=.7, ci=.9, plot_n_positions=81, zoom_outliers=False, **options)
    assert actual is result
    for key, value in expected.items():
        assert (list(calls['fit'][key]) if key == 'degrees' else calls['fit'][key]) == value
    assert ('degree' in calls['fit']) != ('degrees' in calls['fit'])
    assert calls['fit']['nuts'] is nuts
    assert (calls['fit']['warmup'], calls['fit']['samples'], calls['fit']['seed']) == (8, 6, 31)
    assert calls['save'] == (tmp_path/'result', False)
    assert calls['plots'] == (tmp_path/'result/figures', dict(min_inlier_probability=.7,
        ci=.9, plot_n_positions=81, zoom_outliers=False))


@pytest.mark.parametrize('options, message', [
    ({'max_degree': -1}, 'max_degree'),
    ({'max_degree': True}, 'max_degree'),
    ({'max_degree': 1.5}, 'max_degree'),
    ({'max_degree': '4'}, 'max_degree'),
    ({'max_degree': 0, 'degree': 0}, 'not both'),
    ({'degree': -1}, 'degree'),
    ({'mode': 'common', 'max_degree': 2}, 'mode=chebyshev'),
])
def test_analysis_rejects_invalid_degree_options_before_work(tmp_path, monkeypatch, options, message):
    def unexpected(*args, **kwargs):
        pytest.fail('Selection or fitting ran for invalid degree options')
    monkeypatch.setattr('gaoesrvip.analysis.select_lines', unexpected)
    monkeypatch.setattr('gaoesrvip.analysis.fit', unexpected)
    kwargs = dict(mode='chebyshev')
    kwargs.update(options)
    with pytest.raises(ValueError, match=message):
        analyze_thar_ip_2g(tmp_path/'thar', tmp_path/'ec', tmp_path/'output',
                          orders=[104], gain=1., read_noise=4., **kwargs)
    assert not (tmp_path/'output').exists()


def test_full_chebyshev_degree_search_saves_selected_result(tmp_path, monkeypatch):
    from gaoesrvip import fit
    from test_chebyshev_mixture import synthetic
    data = synthetic(4, contaminated=False)
    monkeypatch.setattr('gaoesrvip.analysis.select_lines', lambda *args, **kwargs: [data])
    monkeypatch.setattr('gaoesrvip.analysis._require_corner', lambda: pytest.fail('No corner needed'))
    a, b = tmp_path/'thar', tmp_path/'ec'
    a.touch(); b.touch()
    actual = analyze_thar_ip_2g(a, b, tmp_path/'analysis', orders=[104], gain=1., read_noise=4.,
        mode='chebyshev', max_degree=1, nuts=False, progress=False,
        min_inlier_probability=1., plot_n_positions=30)
    expected = fit(data, mode='chebyshev', degrees=[0, 1], nuts=False, progress=False)
    assert actual.ip_model == expected.ip_model
    np.testing.assert_allclose(actual.degree_scores.BIC_MAP, expected.degree_scores.BIC_MAP)
    restored = load_result(tmp_path/'analysis')
    assert restored.ip_model == actual.ip_model
    assert restored.settings['degree_search']['degrees'] == [0, 1]
    assert {p.name for p in (tmp_path/'analysis/figures').iterdir()} == {
        'lines_order_104.png', 'cumulants_order_104.png', 'cumulants_all_orders.png',
        'residual_summary.png', 'rms_vs_peak.png',
    }  # No line passes the strict P(in) > 1 overlay threshold.
    assert (tmp_path/'analysis/ip_model.json').is_file()
    assert (tmp_path/'analysis/degree_scores.csv').is_file()
    assert not list((tmp_path/'analysis').glob('prediction*'))
