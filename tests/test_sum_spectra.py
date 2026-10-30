"""Central selection, multispec coverage, and redward inference on sums."""
import numpy as np
import pytest
from astropy.io import fits
from gaoesrvip import select_lines, fit, load_result
from gaoesrvip.selection import read_ecidentify
from gaoesrvip.models import pixel_mass, C_KMS
from gaoesrvip.spectra import read_sum
from test_selection import database


def write_sum(path, wave, flux, total=70, kind=0, aperture=4):
    header = fits.Header()
    header['NCOMBINE'] = total
    text = f'wtype=multispec spec1 = "{aperture} 999 {kind} {wave[0]:.15g} {wave[1]-wave[0]:.15g} {len(wave)} 0 0 1"'
    for i, start in enumerate(range(0, len(text), 68), 1):
        header[f'WAT2_{i:03d}'] = text[start:start+68]
    # Finite padding is not part of the valid spectrum.
    fits.writeto(path, np.r_[flux, [1.e9]*10][None], header, overwrite=True)


@pytest.mark.parametrize('reverse_native', [False, True])
@pytest.mark.parametrize('reverse_sum', [False, True])
def test_sum_uses_central_selection_and_redward_samples(tmp_path, monkeypatch, reverse_native, reverse_sum):
    db = tmp_path/'ec'
    text = database([[4, 105, 60.5, 5000, 5000, 1, 1, 1]], constant=525000)
    if reverse_native:
        # Reverse the dispersion term, preserving the table location.
        text = text.replace('\t\t200\n', '\t\t-200\n').replace('\t\t4\n', '\t\t-4\n')
    db.write_text(text)
    sol = read_ecidentify(db)
    wave = sol.wavelength(np.arange(1., 121.), 105)
    scale = abs(sol.dispersion(60.5, 105))*C_KMS/5000
    v = C_KMS*(wave-5000)/5000
    shape = [1.3, .3, 1.6, 1.2, .4]
    image = np.ones((4, 120))*10
    image[3] += 300*pixel_mass(v, shape, scale, tophat=False, model='lab_centered')
    center = tmp_path/'center.fits'
    fits.writeto(center, image)
    calls = []
    def screen(x, y, *args):
        calls.append(y.copy())
        return 4., 0., .02
    monkeypatch.setattr('gaoesrvip.selection._screen_fit', screen)
    kw = dict(orders=[104], gain=1.935, read_noise=4., saturation=5000)
    native = select_lines(center, db, **kw)[0]
    assert native.metadata['coordinate']['velocity_positive'] == 'red'
    np.testing.assert_allclose((native.x[0]-native.table_pixel[0])*native.dv[0], np.sort(v[52:68]), atol=1e-8)
    swave = np.linspace(4998, 5002, 201)
    sdv = C_KMS*(swave[1]-swave[0])/5000
    sy = 7000+20000*pixel_mass(C_KMS*(swave-5000)/5000, shape, sdv, tophat=False, model='lab_centered')
    path = tmp_path/'sum.fits'
    write_sum(path, swave[::-1] if reverse_sum else swave, sy[::-1] if reverse_sum else sy)
    data = select_lines(center, db, fit_spectrum=path, **kw)
    d = data[0]
    assert len(calls) == 2  # Exactly one central screening per call; no sum screening.
    assert d.lab_wave.tolist() == native.lab_wave.tolist()
    np.testing.assert_allclose(d.y[0], sy[d.x[0].astype(int)-1], atol=1e-6)
    np.testing.assert_allclose(d.noise**2, d.y/1.935+70*(4/1.935)**2)
    assert d.y.max() > 5000
    assert d.metadata['noise']['ncombine'] == 70
    assert d.metadata['input_files']['selection_spectrum'] == str(center)
    assert d.x.max() <= len(swave)
    # Known physical parameters isolate coordinate correctness from optimizer behavior.
    from gaoesrvip.fitting import _bounds
    def initialize(d, mode, tophat, max_nfev, model, baseline_degree):
        lo, hi, slo, shi, _, _ = _bounds(d, model, baseline_degree)
        local = np.array([[7000., 0., 20000.]])
        profile = np.array(shape) if mode == 'common' else np.array([shape])
        return local, profile, (lo, hi, slo, shi)
    monkeypatch.setattr('gaoesrvip.fitting._optimize', initialize)
    for mode in ['individual', 'common']:
        def sample(d, mode, tophat, bounds, local, shape, **kwargs):
            return np.repeat(local[None], 4, axis=0), np.repeat(shape[None], 4, axis=0)
        monkeypatch.setattr('gaoesrvip.fitting._sample', sample)
        result = fit(data, mode=mode, model='lab_centered', tophat=False, progress=False)
        o = result.orders[0]
        np.testing.assert_allclose(np.median(o.shape_samples, axis=0).reshape(-1, 5)[0], shape, atol=.02)
        dest = result.save(tmp_path/mode, figures=False)
        restored = load_result(dest)
        assert restored.metadata == result.metadata
        np.testing.assert_array_equal(restored.orders[0].data.y, d.y)
        assert restored.residual_summary().rms_sigma.max() < 1e-6


def test_sum_header_validation_and_padding(tmp_path):
    path = tmp_path/'sum.fits'
    wave = np.arange(5000., 5020.)
    write_sum(path, wave, np.ones(20), kind=2)
    with pytest.raises(ValueError, match='linear'):
        read_sum(path)
    write_sum(path, wave, np.ones(20))
    with fits.open(path, mode='update') as hdus:
        del hdus[0].header['NCOMBINE']
    with pytest.raises(ValueError, match='NCOMBINE'):
        read_sum(path)
    write_sum(path, wave, np.ones(20))
    spectra, total = read_sum(path)
    assert total == 70 and len(spectra[4][0]) == 20


def test_sum_keeps_finite_nonpositive_flux_and_reports_missing_windows(tmp_path):
    from test_lab_centered import lab_data
    from gaoesrvip.spectra import cut_sum
    data, _, _ = lab_data(tophat=False)
    data = data.select([1])
    data.metadata = dict(input_files=dict(spectrum='center.fits'))
    # Shift the analysis grid so the line lies on this test spectrum.
    data.x -= 3750
    data.x0 -= 3750
    data.table_pixel -= 3750
    wave = 5000+(np.arange(1., 101.)-data.table_pixel[0])*5000*data.dv[0]/C_KMS
    flux = np.full(100, 100.)
    flux[45:50] = [0., -1., np.nan, 0., -2.]
    summed = ({4: (wave, flux, 1, 0)}, 107)
    d = cut_sum([data], summed, tmp_path/'sum.fits', 100, 1.935, 4.)[0]
    assert np.any((d.y <= 0) & d.mask)
    assert np.any(~d.mask)
    # Global count, even near edges; no requirement that all strips contribute.
    assert d.metadata['noise']['ncombine'] == 107
    with pytest.warns(UserWarning, match='insufficient coverage'):
        with pytest.raises(ValueError, match='No selected lines'):
            cut_sum([data], ({4: (wave[:10], flux[:10], 1, 0)}, 107), tmp_path/'sum.fits', 100, 1.935, 4.)
    with pytest.raises(ValueError, match='Missing aperture'):
        cut_sum([data], ({5: (wave, flux, 1, 0)}, 107), tmp_path/'sum.fits', 100, 1.935, 4.)
