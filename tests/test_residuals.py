"""Known residuals verify per-line percentages, noise units, and pixel selection."""
import numpy as np
import pandas as pd
import pytest
from scipy.special import erf
from gaoesrvip import OrderData, load_result
from gaoesrvip.results import FitResult, OrderFit


def residual_case(model, mode, tophat, baseline_degree=1):
    table = np.array([100.2, 201.6])
    x = np.floor(table[:, None])+np.arange(-7., 8.)
    dv = np.array([1.1, 1.4])
    width = [1.4, 1.8] if mode == 'individual' else [1.4, 1.4]
    mu = [-.8, .6] if mode == 'individual' else [.37, .37]
    local = np.array([[15., .2, 1000.], [30., -.3, 1500.]])
    shapes = np.array([[s, .5, s, 1., m] for s, m in zip(width, mu)])
    center = table
    if model == 'free_centers':
        center = table+np.array([.25, -.3])
        local = np.column_stack((local, center))
        shapes = np.array([[s, .3, s, 0.] for s in width])
    if baseline_degree == 2:
        local = np.column_stack((local, [.12, -.08]))
    residual = np.array([2.+np.sin(np.arange(15)), np.linspace(-3., 6., 15)])
    noise = np.tile(np.linspace(1., 4., 15), (2, 1))
    y, peaks = [], []
    for i, (s, scale) in enumerate(zip(width, dv)):
        def gaussian(v, mu):
            if tophat:
                return .5*(erf((v+scale/2-mu)/(np.sqrt(2)*s))
                           - erf((v-scale/2-mu)/(np.sqrt(2)*s)))
            return scale*np.exp(-.5*((v-mu)/s)**2)/(np.sqrt(2*np.pi)*s)
        if model == 'lab_centered':
            mass = lambda v: .5*gaussian(v, mu[i])+.5*gaussian(v, mu[i]+1.)
            peak_velocity = mu[i]+.5  # Unique symmetry-axis maximum since delta < 2*sigma.
        else:
            mass = lambda v: gaussian(v, 0.)
            peak_velocity = 0.
        offset = x[i]-center[i]
        peak = local[i, 2]*mass(peak_velocity)
        assert local[i, 2]*mass(offset*scale).max() < peak  # Subpixel phase must not set the denominator.
        background = local[i, 0]+local[i, 1]*offset
        if baseline_degree == 2:
            background += local[i, -1]*offset**2
        y.append(background+local[i, 2]*mass(offset*scale)+residual[i])
        peaks.append(peak)
    d = OrderData(104, x, y, noise, np.ones_like(x, bool), table, dv, ['a', 'b'],
                  lab_wave=[5000., 5100.], table_pixel=table)
    shape = shapes[0] if mode == 'common' else shapes
    result = FitResult([OrderFit(d, local[None], shape[None], model=model, baseline_degree=baseline_degree)],
                       dict(mode=mode, model=model, tophat=tophat, baseline_degree=baseline_degree))
    return result, residual, np.array(peaks)


@pytest.mark.parametrize('model', ['free_centers', 'lab_centered'])
@pytest.mark.parametrize('mode', ['common', 'individual'])
@pytest.mark.parametrize('tophat', [True, False])
@pytest.mark.parametrize('baseline_degree', [1, 2])
def test_residual_units_and_continuous_peak(model, mode, tophat, baseline_degree):
    result, residual, peaks = residual_case(model, mode, tophat, baseline_degree)
    order = result.orders[0]
    before = order.local_samples.copy(), order.shape_samples.copy(), order.data.y.copy()
    summary = result.residual_summary()
    assert list(summary.line_id) == ['a', 'b']
    np.testing.assert_array_equal(summary.lab_wave, [5000., 5100.])
    np.testing.assert_array_equal(summary.n_pixels, [15, 15])
    np.testing.assert_allclose(summary.rms_peak_percent, 100*np.sqrt(np.mean(residual**2, axis=1))/peaks, rtol=1e-10)
    np.testing.assert_allclose(summary.rms_sigma, np.sqrt(np.mean((residual/order.data.noise)**2, axis=1)), rtol=1e-12)
    v = (order.data.x-order.data.table_pixel[:, None])*order.data.dv[:, None]
    np.testing.assert_allclose(summary.vmin_kms, v.min(axis=1))
    np.testing.assert_allclose(summary.vmax_kms, v.max(axis=1))
    for actual, expected in zip((order.local_samples, order.shape_samples, order.data.y), before):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('model', ['free_centers', 'lab_centered'])
def test_masked_invalid_pixels_and_inclusive_table_window(model):
    result, residual, peaks = residual_case(model, 'individual', True)
    d = result.orders[0].data
    d.mask[0, 0] = False
    d.y[0, 0] = 1e12
    d.x[0, 1] = np.nan
    d.noise[0, 2] = 0.
    d.noise[0, 3] = -1.
    d.noise[1, 4] = np.nan
    d.y[1, 5] = np.nan
    valid = np.ones_like(d.mask)
    valid[0, :4] = False
    valid[1, 4:6] = False
    v = (d.x-d.table_pixel[:, None])*d.dv[:, None]
    bounds = (v[0, 5], v[0, 9])
    for window in (None, bounds):
        summary = result.residual_summary(velocity_range=window)
        for i, row in summary.iterrows():
            selected = valid[i].copy()
            if window is not None:
                selected &= (v[i] >= window[0]) & (v[i] <= window[1])
            assert row.n_pixels == selected.sum()
            assert row.rms_peak_percent == pytest.approx(100*np.sqrt(np.mean(residual[i, selected]**2))/peaks[i])
            assert row.rms_sigma == pytest.approx(np.sqrt(np.mean((residual[i, selected]/d.noise[i, selected])**2)))
        if window is not None:
            assert summary.iloc[0].n_pixels == 5
            assert summary.iloc[0].vmin_kms == bounds[0]
            assert summary.iloc[0].vmax_kms == bounds[1]


def test_summary_uses_parameter_medians():
    result, residual, peaks = residual_case('lab_centered', 'common', False)
    order = result.orders[0]
    order.local_samples = np.repeat(order.local_samples, 3, axis=0)
    order.shape_samples = np.repeat(order.shape_samples, 3, axis=0)
    order.local_samples[0, :, 0] -= 3.
    order.local_samples[2, :, 0] += 5.
    order.local_samples[0, :, 2] *= .8
    order.local_samples[2, :, 2] *= 1.4
    order.shape_samples[:, 4] += [-.4, 0., .7]
    summary = result.residual_summary()
    np.testing.assert_allclose(summary.rms_peak_percent, 100*np.sqrt(np.mean(residual**2, axis=1))/peaks)


def test_no_pixels_and_missing_legacy_table_reference():
    result, _, _ = residual_case('free_centers', 'common', False)
    empty = result.residual_summary(velocity_range=(100, 101))
    assert len(empty) == 2 and (empty.n_pixels == 0).all()
    assert empty[['rms_peak_percent', 'rms_sigma', 'vmin_kms', 'vmax_kms']].isna().all().all()
    result.orders[0].data.table_pixel[:] = np.nan
    summary = result.residual_summary()
    assert np.isfinite(summary[['rms_peak_percent', 'rms_sigma']]).all().all()
    assert summary[['vmin_kms', 'vmax_kms']].isna().all().all()
    with pytest.raises(ValueError, match='finite table_pixel'):
        result.residual_summary(velocity_range=(-5, 5))


@pytest.mark.parametrize('window', [(1, -1), (0,), (0, 1, 2), (0, np.nan), (0, np.inf), 5])
def test_invalid_velocity_range(window):
    result = FitResult([], dict(mode='common', tophat=True))
    with pytest.raises(ValueError, match='velocity_range must be two finite bounds'):
        result.residual_summary(velocity_range=window)


@pytest.mark.parametrize('model', ['free_centers', 'lab_centered'])
@pytest.mark.parametrize('mode', ['common', 'individual'])
def test_summary_csv_and_roundtrip(tmp_path, model, mode):
    result, _, _ = residual_case(model, mode, True)
    output = result.save(tmp_path/'result', figures=False)
    summary = result.residual_summary()
    pd.testing.assert_frame_equal(pd.read_csv(output/'residual_summary.csv'), summary)
    restored = load_result(output)
    pd.testing.assert_frame_equal(restored.residual_summary(), summary)


def test_summary_plot_matches_table():
    import matplotlib.pyplot as plt
    result, _, _ = residual_case('lab_centered', 'individual', False)
    window = (-5, 5)
    summary = result.residual_summary(velocity_range=window)
    fig = result.plot_residual_summary(velocity_range=window)
    for ax, column in zip(fig.axes, ['rms_peak_percent', 'rms_sigma']):
        np.testing.assert_array_equal(ax.lines[0].get_xdata(), summary.lab_wave)
        np.testing.assert_allclose(ax.lines[0].get_ydata(), summary[column])
    assert fig.axes[0].get_shared_x_axes().joined(*fig.axes)
    np.testing.assert_array_equal(fig.axes[1].lines[1].get_ydata(), [1., 1.])
    assert 'Table velocity: -5 to 5 km/s' in fig._suptitle.get_text()
    plt.close(fig)
    figures = plt.get_fignums()
    with pytest.raises(ValueError, match='No lines with finite wavelengths'):
        result.plot_residual_summary(velocity_range=(100, 101))
    assert plt.get_fignums() == figures
