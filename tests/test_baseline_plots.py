"""Full-flux line plots preserve the fitted baseline and joint posterior draws."""
import numpy as np
import pytest
from gaoesrvip.models import prediction
from gaoesrvip.results import FitResult, OrderFit
from test_workflow import synthetic


@pytest.mark.parametrize('model', ['free_centers', 'lab_centered'])
@pytest.mark.parametrize('mode', ['common', 'individual'])
@pytest.mark.parametrize('tophat', [True, False])
@pytest.mark.parametrize('baseline_degree', [1, 2])
def test_baseline_plot_matches_full_prediction(model, mode, tophat, baseline_degree):
    import matplotlib.pyplot as plt
    d = synthetic(tophat=tophat, origin=3700.).select([0])
    d.table_pixel[0] += .3
    d.mask[0, -1] = False
    d.y[0, -1] = np.nan
    local = np.array([[[12., .1, 1400., 3699.75]], [[25., -.12, 1700., 3700.5]],
                      [[40., .2, 2400., 3701.2]]])
    shape = np.array([[[1.2, .25, 1.8, 1.]], [[2., .45, .8, 2.5]],
                      [[2.8, .4, 1.4, 3.]]])
    if model == 'lab_centered':
        local = local[:, :, :3]
        shape = np.concatenate((shape, np.array([-.8, .1, .7])[:, None, None]), axis=-1)
    if baseline_degree == 2:
        local = np.concatenate((local, np.array([.08, -.12, .2])[:, None, None]), axis=-1)
    samples = shape[:, 0] if mode == 'common' else shape
    result = FitResult([OrderFit(d, local, samples, model=model, baseline_degree=baseline_degree)],
                       dict(mode=mode, model=model, tophat=tophat, baseline_degree=baseline_degree))
    local_before, shape_before, data_before = local.copy(), samples.copy(), d.y.copy()
    fig = result.plot_lines(include_baseline=True)[0]
    ax = fig.axes[0]
    observed, total, first, second, baseline = ax.lines[:5]
    dv, reference = d.dv[0], d.table_pixel[0]
    v = total.get_xdata()
    x = reference+v/dv
    params, profile = np.median(local[:, 0], axis=0), np.median(shape[:, 0], axis=0)
    origin = reference if model == 'lab_centered' else params[3]
    expected_baseline = params[0]+params[1]*(x-origin)
    if baseline_degree == 2:
        expected_baseline += params[-1]*(x-origin)**2
    expected = prediction(x, dv, params, profile, model=model, reference=reference,
                          tophat=tophat, baseline_degree=baseline_degree)
    np.testing.assert_array_equal(observed.get_ydata(), d.y[0, d.mask[0]])
    np.testing.assert_allclose(observed.get_xdata(), (d.x[0, d.mask[0]]-reference)*dv)
    np.testing.assert_allclose(total.get_ydata(), expected)
    np.testing.assert_allclose(baseline.get_ydata(), expected_baseline)
    np.testing.assert_allclose(first.get_ydata()+second.get_ydata()-baseline.get_ydata(), expected)
    assert baseline.get_linestyle() == '--'
    assert fig._supylabel.get_text() == 'Flux [input units]'
    posterior = ax.lines[5:]
    assert len(posterior) == 3
    for draw_local, draw_shape in zip(local[:, 0], shape[:, 0]):
        expected = prediction(x, dv, draw_local, draw_shape, tophat=tophat,
                              model=model, reference=reference, baseline_degree=baseline_degree)
        assert sum(np.allclose(curve.get_ydata(), expected) for curve in posterior) == 1
    normalized = result.plot_lines(n_draws=0)[0]
    np.testing.assert_allclose(normalized.axes[0].lines[1].get_ydata(),
                               (total.get_ydata()-expected_baseline)/params[2]/dv, atol=1e-14)
    data_offset = d.x[0, d.mask[0]]-origin
    background = params[0]+params[1]*data_offset
    if baseline_degree == 2:
        background += params[-1]*data_offset**2
    np.testing.assert_allclose(normalized.axes[0].lines[0].get_ydata(),
                               (d.y[0, d.mask[0]]-background)/params[2]/dv)
    normalized_draws = result.plot_lines()[0]
    for curve in posterior:
        expected = (curve.get_ydata()-expected_baseline)/params[2]/dv
        assert sum(np.allclose(draw.get_ydata(), expected)
                   for draw in normalized_draws.axes[0].lines[4:-1]) == 1
    no_draws = result.plot_lines(include_baseline=True, n_draws=0)[0]
    assert len(no_draws.axes[0].lines) == 5
    for actual, before in [(local, local_before), (samples, shape_before), (d.y, data_before)]:
        np.testing.assert_array_equal(actual, before)
    for figure in (fig, normalized, no_draws, normalized_draws):
        plt.close(figure)
