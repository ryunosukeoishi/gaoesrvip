"""Select isolated ThAr lines from extracted FITS and IRAF ecidentify output."""
from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from .models import C_KMS
from .data import OrderData


@dataclass
class WavelengthSolution:
    features: pd.DataFrame
    coefficients: np.ndarray
    catalog: np.ndarray

    def wavelength(self, pixel, db_order):
        """Evaluate the IRAF wavelength solution (Angstrom); pixels are 1-based."""
        function, nx, ny, cross, xmin, xmax, ymin, ymax = self.coefficients[:8]
        coef = self.coefficients[8:].reshape((int(nx), int(ny)), order='F')
        xn = (2*np.asarray(pixel)-xmin-xmax)/(xmax-xmin)
        yn = (2*db_order-ymin-ymax)/(ymax-ymin)
        return np.polynomial.chebyshev.chebval2d(xn, np.full_like(xn, yn), coef)/db_order

    def dispersion(self, pixel, db_order):
        return (self.wavelength(pixel+.01, db_order)-self.wavelength(pixel-.01, db_order))/.02


def read_ecidentify(path):
    """Read the last full-cross-term Chebyshev solution and the line catalog."""
    records = []
    for block in re.split(r'(?m)^begin\s+ecidentify\s+', Path(path).read_text())[1:]:
        lines = block.splitlines()
        def values(key):
            index = next(i for i, line in enumerate(lines) if re.match(r'\s*'+key+r'\s+', line))
            count = int(lines[index].split()[1])
            return lines[index+1:index+1+count]
        features = pd.DataFrame([[float(v) for v in line.split()[:8]] for line in values('features')],
            columns=['aperture', 'db_order', 'pixel', 'fit_wave_db', 'lab_wave', 'weight', 'flag', 'used_db'])
        coef = np.array([float(line.strip()) for line in values('coefficients')])
        if int(coef[0]) != 1 or int(coef[3]) != 1:
            raise NotImplementedError('Only full-cross-term IRAF Chebyshev surfaces are supported')
        records.append((features, coef))
    catalog = np.unique(np.concatenate([f.lab_wave.to_numpy() for f, _ in records]))
    catalog = catalog[np.isfinite(catalog) & (catalog > 0)]
    return WavelengthSolution(*records[-1], catalog)


def select_lines(path, wavelength_file, *, orders, gain, read_noise, order_offset=100,
                 half_window=7, saturation=51000, blend_guard=2., resolving_power=65000,
                 fit_spectrum=None):
    """Select isolated, unsaturated lines without a preliminary fit.

    Selection always uses path and its ecidentify database. Optional fit_spectrum
    supplies a wavelength-calibrated IRAF linear multispec sum for fitting,
    either FITS or a ThAr pickle with data/header (using lam and flux).
    Selection uses catalog, blend, saturation and usable-cutout checks only.
    Returned coordinates increase redward; dv is a positive sampling width.
    """
    from astropy.io import fits
    from .spectra import read_sum, cut_sum
    summed = read_sum(fit_spectrum) if fit_spectrum is not None else None
    if not np.isfinite(gain) or gain <= 0 or not np.isfinite(read_noise) or read_noise < 0:
        raise ValueError("gain must be positive and read_noise nonnegative")
    orders = list(orders)
    selection = dict(orders=np.asarray(orders).tolist(), gain=gain, read_noise=read_noise,
                     order_offset=order_offset, half_window=half_window, saturation=saturation,
                     blend_guard=blend_guard, resolving_power=resolving_power)
    # NumPy scalar arguments are common in notebooks; store JSON-native values.
    selection = {key: value.item() if isinstance(value, np.generic) else value
                 for key, value in selection.items()}
    metadata = dict(input_files=dict(spectrum=str(Path(path).resolve()),
                                    wavelength_solution=str(Path(wavelength_file).resolve())),
                    selection=selection)
    solution = read_ecidentify(wavelength_file)
    flux = np.asarray(fits.getdata(path), float)
    table = solution.features.copy()
    table['order'] = table.aperture.astype(int)+order_offset
    candidates = table[table.order.isin(orders) & (table.used_db == 1)]
    rows = []
    for row in candidates.itertuples(index=False):
        ap, x0 = int(row.aperture), float(row.pixel)
        lo, hi = int(np.floor(x0-half_window)), int(np.ceil(x0+half_window))
        if lo < 1 or hi > flux.shape[1]:
            continue
        x = np.arange(lo, hi+1, dtype=float)
        y = flux[ap-1, lo-1:hi]
        distances = np.abs(C_KMS*np.log(solution.catalog/row.lab_wave))
        nearest = distances[distances > 1e-5].min(initial=np.inf)
        if (nearest <= blend_guard*C_KMS/resolving_power or not np.isfinite(y).all()
                or len(x) < 9 or (saturation is not None and np.any(y >= saturation))):
            continue
        wave = float(solution.wavelength(x0, row.db_order))
        dispersion = float(solution.dispersion(x0, row.db_order))
        target = lambda pixel: float(solution.wavelength(pixel, row.db_order))-row.lab_wave
        table_pixel = brentq(target, 1., float(flux.shape[1]), xtol=1e-9)
        rows.append(dict(order=int(row.order), aperture=ap, pixel=x0, fit_wave=wave,
                         lab_wave=row.lab_wave, dispersion=dispersion, table_pixel=table_pixel))
    if not rows:
        raise ValueError('No lines passed selection')
    accepted = pd.DataFrame(rows)
    data = []
    for order, group in accepted.groupby('order', sort=True):
        pixels = [np.arange(int(np.floor(p-half_window)), int(np.ceil(p+half_window))+1,
                            dtype=float) for p in group.pixel]
        n, width = len(group), max(map(len, pixels))
        x, y, noise = np.zeros((n, width)), np.zeros((n, width)), np.ones((n, width))
        mask = np.zeros((n, width), bool)
        for i, (row, xp) in enumerate(zip(group.itertuples(index=False), pixels)):
            values = flux[int(row.aperture)-1, xp.astype(int)-1]
            count = len(xp)
            x[i, :count], y[i, :count] = xp, values
            noise[i, :count] = np.sqrt(np.maximum(values, 0)/gain+(read_noise/gain)**2)
            mask[i, :count] = noise[i, :count] > 0
        direction = np.sign(group.dispersion.to_numpy())
        if not np.all(direction == direction[0]) or direction[0] == 0:
            raise ValueError(f"Order {order}: dispersion must have a consistent nonzero sign")
        reverse = bool(direction[0] < 0)
        origin = flux.shape[1]+1 if reverse else 0
        sign = -1 if reverse else 1
        if reverse:
            x = origin-x[:, ::-1]
            y, noise, mask = y[:, ::-1], noise[:, ::-1], mask[:, ::-1]
        order_metadata = dict(metadata, coordinate=dict(velocity_positive="red",
            kind="native", original_pixel_scale=sign, original_pixel_offset=origin),
            noise=dict(model="diagonal Poisson plus read noise", ncombine=1))
        data.append(OrderData(order, x, y, noise, mask, origin+sign*group.pixel.to_numpy(),
            C_KMS*np.abs(group.dispersion.to_numpy())/group.fit_wave.to_numpy(),
            np.arange(n).astype(str), group.fit_wave.to_numpy(), group.lab_wave.to_numpy(),
            table_pixel=origin+sign*group.table_pixel.to_numpy(), metadata=order_metadata))
    if summed is not None:
        data = cut_sum(data, summed, fit_spectrum, order_offset, gain, read_noise)
    return data
