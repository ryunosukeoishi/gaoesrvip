"""Input provenance survives selection, fitting and portable result serialization."""
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import pytest
from astropy.io import fits
from gaoesrvip import fit, load_result, select_lines
from gaoesrvip.fitting import _bounds
from test_selection import database
from test_residuals import residual_case
from test_workflow import synthetic


def test_selection_metadata_through_fit_and_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    thar, wavelength = Path('thar.fits'), Path('wavelength.db')
    image = np.full((5, 120), 5.)
    image[3, 49], image[4, 69] = 100., 200.
    fits.writeto(thar, image)
    wavelength.write_text(database([[4, 105, 50, 5000, 5000, 1, 1, 1],
                                    [5, 106, 70, 4953, 4953, 1, 1, 1]], constant=525000))
    monkeypatch.setattr('gaoesrvip.selection._screen_fit', lambda *args: (4., 0., .02))
    data = select_lines(thar, wavelength, orders=np.array([104, 105, 106]),
                        gain=np.float64(1.935), read_noise=np.float32(4),
                        order_offset=np.int64(100), half_window=np.int64(9), saturation=None,
                        blend_guard=np.float32(1.5), resolving_power=np.int64(60000),
                        max_center_shift=np.float64(.5), max_width_error=np.float64(.05))
    expected = dict(
        input_files=dict(spectrum=str(thar.resolve()), wavelength_solution=str(wavelength.resolve())),
        selection=dict(orders=[104, 105, 106], gain=1.935, read_noise=4., order_offset=100,
                       half_window=9, saturation=None, blend_guard=1.5, resolving_power=60000,
                       max_center_shift=.5, max_width_error=.05))
    expected['coordinate'] = dict(velocity_positive='red', kind='native',
                                  original_pixel_scale=1, original_pixel_offset=0)
    expected['noise'] = dict(model='diagonal Poisson plus read noise', ncombine=1)
    assert [d.order for d in data] == [104, 105]
    assert all(d.metadata == expected for d in data)
    assert json.loads(json.dumps(data[0].metadata)) == expected
    data[1].metadata['selection']['half_window'] = 99
    assert data[0].metadata == expected  # No shared nested dictionaries across orders.

    # Combine orders selected with different settings, including default arguments.
    other = select_lines(thar, wavelength, orders=[105], gain=2., read_noise=3.)[0]
    assert other.metadata['selection'] == dict(orders=[105], gain=2., read_noise=3.,
        order_offset=100, half_window=7, saturation=51000, blend_guard=2.,
        resolving_power=65000, max_center_shift=1., max_width_error=.1)

    # The test checks metadata transport, without numerical fitting or sampling.
    def initialize(d, mode, tophat, max_nfev, model, baseline_degree):
        lo, hi, slo, shi, local, shape = _bounds(d, model, baseline_degree)
        return local, np.median(shape, axis=0), (lo, hi, slo, shi)

    def sample(d, mode, tophat, bounds, local, shape, **kwargs):
        return np.repeat(local[None], 4, axis=0), np.repeat(shape[None], 4, axis=0)

    monkeypatch.setattr('gaoesrvip.fitting._optimize', initialize)
    monkeypatch.setattr('gaoesrvip.fitting._sample', sample)
    result = fit([other, data[0]], model='lab_centered', baseline_degree=2, progress=False)
    assert result.metadata == {104: expected, 105: other.metadata}
    result.metadata[104]['selection']['half_window'] = 999
    assert result.metadata[104] == expected  # Inspection returns a detached copy.
    output = result.save(tmp_path/'result', figures=False)
    saved = json.loads((output/'result.json').read_text())
    assert saved['data_metadata'] == {'104': expected, '105': other.metadata}
    assert saved['settings']['baseline_degree'] == 2
    # Loading never requires the input files or the original working directory.
    thar.unlink()
    wavelength.unlink()
    monkeypatch.chdir(tmp_path.parent)
    restored = load_result(output)
    assert restored.metadata == result.metadata
    for original, loaded in zip(result.orders, restored.orders):
        np.testing.assert_array_equal(loaded.data.lab_wave, original.data.lab_wave)
        np.testing.assert_array_equal(loaded.local_samples, original.local_samples)
    again = restored.save(tmp_path/'resaved', figures=False)
    assert load_result(again).metadata == result.metadata


def test_manual_line_filter_preserves_original_metadata_and_final_lines(tmp_path):
    result, _, _ = residual_case('lab_centered', 'common', False)
    order = result.orders[0]
    order.data.metadata = dict(selection=dict(half_window=10, orders=[104]))
    selected = order.data.select([1])
    assert selected.metadata == order.data.metadata
    selected.metadata['selection']['orders'].append(105)
    assert order.data.metadata['selection']['orders'] == [104]
    selected.metadata['selection']['orders'] = [104]
    filtered = replace(result, orders=[replace(order, data=selected,
                                               local_samples=order.local_samples[:, [1]])])
    restored = load_result(filtered.save(tmp_path/'subset', figures=False))
    assert restored.metadata == result.metadata
    np.testing.assert_array_equal(restored.orders[0].data.line_ids, ['b'])
    np.testing.assert_array_equal(restored.orders[0].data.lab_wave, [5100.])


@pytest.mark.parametrize('schema', [1, 2, 3])
def test_old_results_without_metadata_load_as_unknown(tmp_path, schema):
    model = 'free_centers' if schema == 1 else 'lab_centered'
    result, _, _ = residual_case(model, 'common', False)
    output = result.save(tmp_path/'old', figures=False)
    path = output/'result.json'
    saved = json.loads(path.read_text())
    del saved['data_metadata']
    saved['schema_version'] = schema
    if schema < 3:
        del saved['settings']['baseline_degree']
    if schema == 1:
        del saved['settings']['model']
    path.write_text(json.dumps(saved))
    restored = load_result(output)
    assert restored.metadata == {104: {}}
    np.testing.assert_array_equal(restored.orders[0].local_samples, result.orders[0].local_samples)


def test_manually_created_data_have_independent_empty_metadata():
    first, second = synthetic(), synthetic()
    assert first.metadata == second.metadata == {}
    first.metadata['note'] = 'test'
    assert second.metadata == {}
