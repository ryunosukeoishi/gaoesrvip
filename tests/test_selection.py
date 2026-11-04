"""Pipeline adapters and coherent posterior plotting regressions."""
import numpy as np
import pytest
from astropy.io import fits
from gaoesrvip import read_ecidentify, select_lines
from gaoesrvip.models import prediction
from gaoesrvip.plotting import _posterior_lines
from gaoesrvip.results import OrderFit
from test_workflow import synthetic


def database(features, constant=500000.):
    coefficients = [1, 2, 2, 1, 1, 120, 100, 110, constant, 200, 30, 4]
    return ('begin\tecidentify thar\n\tfeatures\t'+str(len(features))+'\n'
            + ''.join('\t\t'+' '.join(map(str, row))+'\n' for row in features)
            + '\tcoefficients\t12\n'+''.join('\t\t'+str(v)+'\n' for v in coefficients))


def test_ecidentify_last_solution_and_catalog(tmp_path):
    p = tmp_path/'ecThAr'
    p.write_text(database([[4, 105, 25, 5000, 5000, 1, 1, 1]])
                 + database([[4, 105, 50, 5000, 5001, 1, 1, 1]], 510000))
    solution = read_ecidentify(p)
    assert list(solution.catalog) == [5000, 5001]
    assert list(solution.features.pixel) == [50]
    xn = (2*np.array([1, 30, 120])-121)/119
    yn = (2*103-210)/10
    expected = (510000+200*xn+30*yn+4*xn*yn)/103
    np.testing.assert_allclose(solution.wavelength(np.array([1, 30, 120]), 103), expected)
    assert solution.dispersion(30, 103) == pytest.approx((200+4*yn)*2/119/103)
    p.write_text(database([[4, 105, 25, 5000, 5000, 1, 1, 1]]).replace('\t\t1\n','\t\t2\n',1))
    with pytest.raises(NotImplementedError):
        read_ecidentify(p)


def test_selection_reasons_and_cutout_extraction(tmp_path, monkeypatch):
    # Unused catalog neighbors still matter for blend screening.
    features = [[4, 105, x, wave, wave, 1, 1, used] for x, wave, used in
                [(25, 5000, 1), (50, 4999, 1), (53, 4999.01, 0),
                 (75, 5002, 1), (100, 5001, 1), (2, 5006, 1)]]
    p = tmp_path/'ecthar'
    p.write_text(database(features, constant=525000))
    image = np.full((4, 120), 5.)
    image[3, 24], image[3, 74], image[3, 99] = 10, 1000, 30
    thar = tmp_path/'thar.fits'
    fits.writeto(thar, image)
    data = select_lines(thar, p, orders=[104], gain=2, read_noise=4, saturation=500)
    d = data[0]
    assert d.order == 104 and list(d.line_ids) == ['0', '1']
    assert d.table_pixel[0] == pytest.approx(60.5)
    np.testing.assert_array_equal(d.y[0], image[3, 17:32])
    np.testing.assert_allclose(d.noise[0]**2, d.y[0]/2+4)
    default_data = select_lines(thar, p, orders=[104], gain=2, read_noise=4,
                                half_window=5, saturation=500)
    assert isinstance(default_data, list)
    assert len(default_data[0].x0) == 2  # Both isolated, unsaturated windows survive without a fit-quality cut.
    np.testing.assert_array_equal(default_data[0].x[0], np.arange(20, 31))
    from gaoesrvip import fit
    from gaoesrvip.fitting import _bounds
    # This fixture tests selection plumbing; its DB and artificial peaks are unrelated.
    def initialize(d, mode, max_nfev, baseline_degree):
        lo, hi, slo, shi, local, shape = _bounds(d, baseline_degree)
        return local, shape, (lo, hi, slo, shi)
    monkeypatch.setattr('gaoesrvip.fitting._optimize', initialize)
    def sample(d, mode, bounds, local, shape, **kwargs):
        return np.repeat(local[None], 4, axis=0), np.repeat(shape[None], 4, axis=0)
    monkeypatch.setattr('gaoesrvip.fitting._sample', sample)
    result = fit(default_data, mode='individual', progress=False)
    assert result.orders[0].data is default_data[0]
    # The wavelength database location/name is explicitly supplied.
    relocated = tmp_path/'calibration'/'wavelength.db'
    relocated.parent.mkdir()
    p.rename(relocated)
    pipeline_data = select_lines(thar, relocated, orders=[104], gain=2, read_noise=4,
                                 half_window=5, saturation=500)
    np.testing.assert_array_equal(pipeline_data[0].y, default_data[0].y)
    with pytest.raises(FileNotFoundError):
        select_lines(thar, p, orders=[104], gain=2, read_noise=4)


def test_posterior_curves_keep_joint_draws_and_fixed_data_frame():
    d = synthetic()
    local = np.tile([[12., .1, 1400.], [20., -.15, 1000.],
                     [8., .05, 1800.]], (3, 1, 1))
    shape = np.array([np.tile([1.5, .3, 1.2, 2., 0.], (3, 1)) for _ in range(3)])
    local[:, 0, 0] = [8, 12, 20]
    local[:, 0, 2] = [1000, 1400, 2000]
    shape[:, 0, 4] = [-.8, -.25, .5]
    shape[:, 0, 3] = [1, 2, 3]
    order = OrderFit(d, local, shape)
    v = np.linspace(-8, 8, 41)
    curves = _posterior_lines(order, 0, v, [2, 0])
    b, slope, flux = np.median(local[:, 0], axis=0)
    center = d.table_pixel[0]
    for curve, index in zip(curves, [2, 0]):
        expected = (prediction(center+v/d.dv[0], d.dv[0], local[index, 0], shape[index, 0], reference=center)-b-slope*v/d.dv[0])/flux/d.dv[0]
        np.testing.assert_allclose(curve, expected)


@pytest.mark.parametrize('kwargs', [dict(screen_fit=True), dict(screen_fit=False),
                                    dict(max_center_shift=1.), dict(max_width_error=.1)])
def test_removed_screen_options_raise(kwargs):
    with pytest.raises(TypeError, match='unexpected keyword argument'):
        select_lines('unused.fits', 'unused.db', orders=[104], gain=2., read_noise=4., **kwargs)
