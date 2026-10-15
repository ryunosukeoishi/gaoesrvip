"""Select isolated ThAr lines from extracted FITS and IRAF ecidentify output."""
from dataclasses import dataclass
from pathlib import Path
import re
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, brentq
from .models import C_KMS, pixel_mass
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


def _model(x, bkg, slope, flux, center, s1, fraction2, s2, delta):
    return bkg+slope*(x-center)+flux*pixel_mass(x-center, [s1, fraction2, s2, delta], 1.)


def _width(shape):
    # Fast effective width and peak for screening/covariance propagation.
    s1, f, s2, delta = shape
    extent = max(15., abs(delta)+12*max(s1, s2))
    grid = np.linspace(-extent, extent, 801)
    profile = pixel_mass(grid, shape, 1.)
    peak = int(np.argmax(profile))
    half = .5*profile[peak]
    left_candidates = np.flatnonzero(profile[:peak] <= half)
    right_candidates = np.flatnonzero(profile[peak+1:] <= half)
    if not len(left_candidates) or not len(right_candidates):
        return np.nan, np.nan
    left, right = left_candidates[-1], peak+1+right_candidates[0]
    lo = np.interp(half, profile[left:left+2], grid[left:left+2])
    hi = np.interp(half, profile[right-1:right+1][::-1], grid[right-1:right+1][::-1])
    return hi-lo, grid[peak]


def _screen_fit(x, y, x0, seed):
    bkg = float(np.median(np.r_[y[:3], y[-3:]]))
    amp = max(float(y.max()-bkg), 1.)
    def gaussian(x, b, slope, a, center, sigma):
        return b+slope*(x-center)+a*np.exp(-.5*((x-center)/sigma)**2)
    try:
        p, _ = curve_fit(gaussian, x, y, p0=[bkg, 0, amp, x[np.argmax(y)], 1.7],
            bounds=([-np.inf, -np.inf, 0, x0-2, .5], [np.inf, np.inf, np.inf, x0+2, 5]), maxfev=20000)
    except (ValueError, RuntimeError):
        p = [bkg, 0, amp, x0, 1.7]
    b, slope, a, center, sigma = p
    lower = np.array([-np.inf, -np.inf, 0, x0-2, .4, .001, .4, -4])
    upper = np.array([np.inf, np.inf, np.inf, x0+2, 4, .49, 6, 4])
    initial = [b, slope, a*sigma*np.sqrt(2*np.pi), center,
               np.clip(.85*sigma, .41, 3.9), .18, np.clip(1.35*sigma, .41, 5.9), .8]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        best, cov = curve_fit(_model, x, y, p0=initial, bounds=(lower, upper), maxfev=50000)
    width, peak = _width(best[4:])
    relative_error = np.nan
    covariance = cov[4:, 4:]
    if np.isfinite(covariance).all():
        eigenvalues, vectors = np.linalg.eigh((covariance+covariance.T)/2)
        transform = vectors@np.diag(np.sqrt(np.clip(eigenvalues, 0, None)))
        rng = np.random.default_rng(seed)
        draws = best[4:]+rng.standard_normal((20000, 4))@transform.T
        shapes = draws[np.all((draws > lower[4:]) & (draws < upper[4:]), axis=1)][:200]
        widths = np.array([_width(shape)[0] for shape in shapes])
        widths = widths[np.isfinite(widths) & (widths > 0)]
        if len(widths) >= 50:
            p16, p50, p84 = np.percentile(widths, [16, 50, 84])
            relative_error = (p84-p16)/(2*p50)
    return width, best[3]+peak, relative_error


def select_lines(path, wavelength_file, *, orders, gain, read_noise, order_offset=100,
                 half_window=7, saturation=51000, blend_guard=2., resolving_power=65000,
                 max_center_shift=1., max_width_error=.1):
    """Select isolated, unsaturated lines with reliable centers and widths.

    Each order's metadata records input paths and all selection arguments.
    """
    from astropy.io import fits
    orders = list(orders)
    selection = dict(orders=np.asarray(orders).tolist(), gain=gain, read_noise=read_noise,
                     order_offset=order_offset, half_window=half_window, saturation=saturation,
                     blend_guard=blend_guard, resolving_power=resolving_power,
                     max_center_shift=max_center_shift, max_width_error=max_width_error)
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
        try:
            width, peak, error = _screen_fit(x-x0, y, 0., int(ap*10000000+round(x0*1000)))
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError):
            continue
        if not np.isfinite([width, peak, error]).all() or abs(peak) > max_center_shift or error > max_width_error:
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
        data.append(OrderData(order, x, y, noise, mask, group.pixel.to_numpy(),
            C_KMS*np.abs(group.dispersion.to_numpy())/group.fit_wave.to_numpy(),
            np.arange(n).astype(str), group.fit_wave.to_numpy(), group.lab_wave.to_numpy(),
            table_pixel=group.table_pixel.to_numpy(), metadata=metadata))
    return data
