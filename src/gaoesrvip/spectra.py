"""Read wavelength-calibrated FITS/pickle sums and prepare redward coordinates."""
from copy import deepcopy
from pathlib import Path
import re
import warnings
import numpy as np
from astropy.io import fits
from .data import OrderData
from .models import C_KMS


def _read_pickle_sum(path):
    """Read a trusted local ThAr pickle containing a DataFrame and FITS header."""
    import pandas as pd
    payload = pd.read_pickle(path)
    if not isinstance(payload, dict) or not {'data', 'header'} <= payload.keys():
        raise ValueError('ThAr pickle must contain data and header')
    table, header = payload['data'], payload['header']
    if (not isinstance(table, pd.DataFrame) or not table.columns.is_unique
            or not {'order', 'lam', 'flux'} <= set(table.columns)):
        raise ValueError('ThAr pickle data must be a DataFrame with order, lam and flux columns')
    if not isinstance(header, (dict, fits.Header)):
        raise ValueError('ThAr pickle header must be a FITS Header or dictionary')
    if (table.empty or not pd.api.types.is_integer_dtype(table['order'])
            or table['order'].isna().any()):
        raise ValueError('ThAr pickle order must contain zero-based integer row indices')
    groups = list(table.groupby('order', sort=True))
    if [index for index, _ in groups] != list(range(len(groups))):
        raise ValueError('ThAr pickle order must contain consecutive zero-based row indices')
    wave = [group['lam'].to_numpy(dtype=float) for _, group in groups]
    flux = [group['flux'].to_numpy(dtype=float) for _, group in groups]
    return flux, wave, header


def read_sum(path):
    """Read a FITS or ThAr .pkl/.pickle sum and its global NCOMBINE count.

    Pickles use data.order as the zero-based multispec row, data.lam as the
    wavelength, and data.flux (not fflux). The header supplies the aperture and
    valid length; padded tails are discarded before wavelength validation.
    """
    path = Path(path).expanduser()
    wavelength = None
    if path.suffix.lower() in ('.pkl', '.pickle'):
        flux, wavelength, header = _read_pickle_sum(path)
    else:
        flux, header = fits.getdata(path, header=True)
        if flux.ndim != 2:
            raise ValueError('Summed spectrum must be a 2D IRAF multispec image')
    total = header.get('NCOMBINE')
    if (not isinstance(total, (int, float)) or not np.isfinite(total)
            or total < 1 or total != int(total)):
        raise ValueError('Summed spectrum requires a positive integer NCOMBINE')
    wat = ''.join(str(header[k]).ljust(68) for k in sorted(header) if k.startswith('WAT2_'))
    records = re.findall(r'spec(\d+)\s*=\s*"([^"]+)"', wat)
    if len(records) != len(flux):
        raise ValueError('Incomplete IRAF multispec wavelength header')
    spectra, rows = {}, set()
    for index, record in records:
        values = record.split()
        if len(values) < 9:
            raise ValueError('Incomplete IRAF multispec spectrum record')
        ap, _, kind = map(int, values[:3])
        if kind != 0 or float(values[6]) != 0:
            raise ValueError('Only unshifted linear multispec wavelengths are supported')
        row, size = int(index)-1, int(values[5])
        start, step = float(values[3]), float(values[4])
        if (row in rows or ap in spectra or not 0 <= row < len(flux)
                or not 1 < size <= len(flux[row]) or not np.isfinite([start, step]).all()
                or step == 0):
            raise ValueError('Invalid or duplicate multispec spectrum record')
        wave = start+np.arange(size)*step
        if wavelength is not None:
            supplied = wavelength[row][:size]
            if not np.allclose(supplied, wave, rtol=0, atol=1e-8):
                raise ValueError(f'ThAr pickle order {row}: lam does not match the unshifted linear WCS')
            wave = supplied
        if np.any(wave <= 0):
            raise ValueError('Wavelengths must be positive')
        reverse = step < 0
        spectra[ap] = (wave[::-1] if reverse else wave,
                       np.asarray(flux[row][:size], float)[::-1] if reverse else
                       np.asarray(flux[row][:size], float),
                       -1 if reverse else 1, size+1 if reverse else 0)
        rows.add(row)
    return spectra, int(total)


def cut_sum(selected, summed, path, order_offset, gain, read_noise):
    """Map central selections to sum samples without interpolation or screening fits."""
    spectra, total = summed
    result, skipped = [], []
    for original in selected:
        aperture = original.order-order_offset
        if aperture not in spectra:
            raise ValueError(f'Missing aperture {aperture} in summed spectrum')
        wave, flux, sign, offset = spectra[aperture]
        step = wave[1]-wave[0]
        cuts, locations, tables, scales, retained = [], [], [], [], []
        for i, lab in enumerate(original.lab_wave):
            table = 1+(lab-wave[0])/step
            dv = C_KMS*step/lab
            v = (original.x[i, original.mask[i]]-original.table_pixel[i])*original.dv[i]
            lo, hi = int(np.floor(table+v.min()/dv)), int(np.ceil(table+v.max()/dv))
            if lo < 1 or hi > len(wave):
                skipped.append((original.order, str(original.line_ids[i])))
                continue
            xp = np.arange(lo, hi+1, dtype=float)
            y = flux[lo-1:hi]
            noise = np.sqrt(np.maximum(y, 0)/gain+total*(read_noise/gain)**2)
            mask = np.isfinite(y) & np.isfinite(noise) & (noise > 0)
            if mask.sum() < 9:
                skipped.append((original.order, str(original.line_ids[i])))
                continue
            cuts.append((xp, y, noise, mask))
            locations.append(table+(original.x0[i]-original.table_pixel[i])*original.dv[i]/dv)
            tables.append(table)
            scales.append(dv)
            retained.append(i)
        if not cuts:
            continue
        count, width = len(cuts), max(len(c[0]) for c in cuts)
        x, y, noise = np.zeros((count, width)), np.zeros((count, width)), np.ones((count, width))
        mask = np.zeros((count, width), bool)
        for i, cut in enumerate(cuts):
            for dest, values in zip((x, y, noise, mask), cut):
                dest[i, :len(values)] = values
        metadata = deepcopy(original.metadata)
        metadata['input_files']['selection_spectrum'] = metadata['input_files']['spectrum']
        metadata['input_files']['spectrum'] = str(Path(path).expanduser().resolve())
        metadata['coordinate'] = dict(velocity_positive='red', kind='resampled',
                                      original_pixel_scale=sign, original_pixel_offset=offset)
        metadata['noise'] = dict(model='diagonal Poisson plus NCOMBINE read noise',
                                 ncombine=total, resampling_covariance=False,
                                 wavelength_dependent_contributions=False)
        result.append(OrderData(original.order, x, y, noise, mask, np.array(locations),
                                np.array(scales), original.line_ids[retained],
                                wave[0]+(np.array(locations)-1)*step,
                                original.lab_wave[retained], table_pixel=np.array(tables),
                                metadata=metadata))
    if skipped:
        warnings.warn(f'Summed spectrum has insufficient coverage or valid samples for lines: {skipped}',
                      stacklevel=2)
    if not result:
        raise ValueError('No selected lines have usable summed-spectrum windows')
    return result
