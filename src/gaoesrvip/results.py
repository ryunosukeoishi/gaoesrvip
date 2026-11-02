"""Inspectable results, portable serialization, and parameter tables."""
from copy import deepcopy
from dataclasses import dataclass, replace
from functools import cached_property
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .data import DATA_FIELDS, OrderData
from .models import C_KMS, parameter_names, baseline, pixel_mass, prediction, profile_fwhm, _model_peak
from .plotting import _plot_lines, _plot_overlay, _plot_resolving_power, _plot_residual_summary
from .chebyshev import ChebyshevFit, cumulants


@dataclass
class OrderFit:
    data: OrderData
    local_samples: np.ndarray  # draw, line, parameter; MAP-only has one estimate, not posterior draws
    shape_samples: np.ndarray  # draw, [line,] shape_parameter
    model: str = 'free_centers'
    baseline_degree: int = 1
    chebyshev: ChebyshevFit | None = None

    def __post_init__(self):
        local_names, shape_names = parameter_names(self.model, self.baseline_degree)
        self.local_samples = np.asarray(self.local_samples)
        self.shape_samples = np.asarray(self.shape_samples)
        nlines = len(self.data.x0)
        if (self.local_samples.ndim != 3 or self.local_samples.shape[1:] != (nlines, len(local_names))
                or self.shape_samples.ndim not in (2, 3)
                or self.shape_samples.shape[-1] != len(shape_names)
                or self.shape_samples.shape[0] != self.local_samples.shape[0]
                or (self.shape_samples.ndim == 3 and self.shape_samples.shape[1] != nlines)):
            raise ValueError(f'Posterior array dimensions do not match {self.model}')
        if self.chebyshev is not None and len(self.chebyshev.observed_peak) != nlines:
            raise ValueError('Chebyshev result does not match the selected lines')

    @property
    def local_names(self):
        return parameter_names(self.model, self.baseline_degree)[0]

    @property
    def shape_names(self):
        return parameter_names(self.model)[1]

    def reference_pixel(self, i):
        """Origin used for the continuum and normalized velocity of this line."""
        if self.model == 'lab_centered':
            return self.data.table_pixel[i]
        return np.median(self.local_samples[:, i, 3])

    def shape_for_line(self, index):
        return self.shape_samples if self.shape_samples.ndim == 2 else self.shape_samples[:, index]

    def point_shape(self, index):
        if self.chebyshev is not None:
            return self.chebyshev.predict(self.data.lab_wave[index:index+1], samples=False)[0, 0]
        return np.median(self.shape_for_line(index), axis=0)

    def median_baseline(self, i, x):
        return baseline(x, np.median(self.local_samples[:, i], axis=0), model=self.model,
                        reference=self.data.table_pixel[i], baseline_degree=self.baseline_degree)

    @cached_property
    def intrinsic_fwhm_samples(self):
        shape = self.shape_samples
        values = np.array([profile_fwhm(s, model=self.model) for s in shape.reshape(-1, len(self.shape_names))])
        return values.reshape(shape.shape[:-1])

    def normalized_line(self, i):
        d = self.data
        valid = d.mask[i]
        flux = np.median(self.local_samples[:, i, 2])
        center = self.reference_pixel(i)
        offset = d.x[i, valid]-center
        return offset*d.dv[i], (d.y[i, valid]-self.median_baseline(i, d.x[i, valid]))/flux/d.dv[i]

    def residual_lines(self, *, tophat):
        lines = []
        for i in range(len(self.data.x0)):
            velocity, observed = self.normalized_line(i)
            dv = self.data.dv[i]
            shape = self.point_shape(i)
            lines.append((velocity, observed-pixel_mass(velocity, shape, dv, tophat=tophat, model=self.model)/dv))
        return lines


@dataclass
class FitResult:
    orders: list[OrderFit]
    settings: dict

    def __post_init__(self):
        self.settings = dict(self.settings)
        model = self.settings.setdefault('model', 'free_centers')
        baseline_degree = self.settings.setdefault('baseline_degree', 1)
        parameter_names(model, baseline_degree)
        if any(o.model != model for o in self.orders):
            raise ValueError('Order models do not match FitResult settings')
        if any(o.baseline_degree != baseline_degree for o in self.orders):
            raise ValueError('Order baseline degrees do not match FitResult settings')
        if self.settings.get('mode') == 'chebyshev' and any(o.chebyshev is None for o in self.orders):
            raise ValueError('Chebyshev mode requires curve results for every order')

    def _interval(self, values):
        finite = values[np.isfinite(values)]
        if not len(finite):
            return dict(p16=np.nan, median=np.nan, p84=np.nan)
        if self.settings.get('inference') == 'map':
            return dict(p16=np.nan, median=float(np.median(finite)), p84=np.nan)
        return dict(zip(('p16', 'median', 'p84'), np.percentile(finite, [16, 50, 84])))

    @property
    def metadata(self):
        """Input and selection metadata by integer order number, returned as a copy."""
        return {o.data.order: deepcopy(o.data.metadata) for o in self.orders}

    @property
    def diagnostics(self):
        return {o.data.order: deepcopy(o.chebyshev.diagnostics) for o in self.orders
                if o.chebyshev is not None}

    @property
    def line_membership(self):
        """MAP conditional membership, or posterior-mean membership for NUTS."""
        rows = []
        for o in self.orders:
            if o.chebyshev is None:
                continue
            c, d = o.chebyshev, o.data
            for i in range(len(d.x0)):
                rows.append(dict(order=d.order, line_id=d.line_ids[i], lab_wave=d.lab_wave[i],
                    observed_peak=c.observed_peak[i], inlier_probability=c.inlier_probability[i],
                    outlier_probability=1-c.inlier_probability[i],
                    individual_success=bool(c.individual_success[i])))
        return pd.DataFrame(rows)

    @property
    def individual_parameters(self):
        """Independent optimization points; no posterior intervals are implied."""
        rows = []
        for o in self.orders:
            if o.chebyshev is None:
                continue
            c = o.chebyshev
            kappa = cumulants(c.individual_shape)
            for i, line_id in enumerate(o.data.line_ids):
                row = dict(order=o.data.order, line_id=line_id, lab_wave=o.data.lab_wave[i],
                           success=bool(c.individual_success[i]))
                row.update(zip(o.shape_names, c.individual_shape[i]))
                row.update(zip(o.local_names, c.individual_local[i]))
                row.update({f'kappa{k+1}': value for k, value in enumerate(kappa[i])})
                rows.append(row)
        return pd.DataFrame(rows)

    def predict_ip(self, wavelength, *, order, samples=True, extrapolate=False):
        """Evaluate the curve: (draw, wavelength, 5), one estimate for MAP-only."""
        match = next((o for o in self.orders if o.data.order == order), None)
        if match is None or match.chebyshev is None:
            raise ValueError('Requested order has no Chebyshev fit')
        return match.chebyshev.predict(np.atleast_1d(wavelength), samples=samples, extrapolate=extrapolate)

    def plot_cumulants(self, *, ci=.68, n_positions=500, zoom_outliers=True):
        """κ1–κ5 curve and independent points colored by spectral membership.

        CI bands are pointwise posterior intervals and only appear for NUTS.
        zoom_outliers changes display limits only; clipped points use triangles.
        """
        from .chebyshev_plotting import plot_cumulants
        return plot_cumulants(self, ci=ci, n_positions=n_positions, zoom_outliers=zoom_outliers)

    @cached_property
    def parameters(self):
        """Shape and intrinsic R; lab_centered locations refer to the table wavelength."""
        rows = []
        for o in self.orders:
            names = o.shape_names
            if o.model == 'lab_centered':
                names += ('mu2_kms',)
            names += ('centroid_kms', 'intrinsic_fwhm_kms', 'R')
            indices = [None] if self.settings['mode'] == 'common' else range(len(o.data.x0))
            for i in indices:
                draws = o.shape_for_line(i)
                fwhm = o.intrinsic_fwhm_samples if i is None else o.intrinsic_fwhm_samples[:, i]
                identity = dict(order=o.data.order)
                if i is not None:
                    identity['line_id'] = o.data.line_ids[i]
                centroid = draws[:, 1]*draws[:, 3]
                columns = [draws]
                if o.model == 'lab_centered':
                    centroid = centroid+draws[:, 4]
                    columns.append(draws[:, 4]+draws[:, 3])
                values = np.column_stack((*columns, centroid, fwhm, C_KMS/fwhm))
                rows.extend(dict(**identity, parameter=name, **self._interval(column))
                            for name, column in zip(names, values.T))
        return pd.DataFrame(rows)

    @cached_property
    def line_parameters(self):
        rows = []
        for result in self.orders:
            for i, line_id in enumerate(result.data.line_ids):
                for k, name in enumerate(result.local_names):
                    rows.append(dict(order=result.data.order, line_id=line_id, parameter=name,
                                     **self._interval(result.local_samples[:, i, k])))
        return pd.DataFrame(rows)

    def residual_summary(self, *, velocity_range=None):
        """Per-line residual RMS at posterior-median parameters, without refitting.

        rms_peak_percent = 100 * RMS(data - model) / continuum-subtracted model peak.
        rms_sigma = sqrt(mean(((data - model) / noise)**2)), with no degrees-of-freedom
        correction. Both metrics use the same valid pixels and the fitted tophat setting.

        velocity_range optionally sets inclusive km/s bounds relative to table_pixel
        for either model. None uses all valid fitted pixels. vmin_kms/vmax_kms report
        the actual sampled extent; they are NaN if table_pixel is unavailable.
        Lines with no selected pixels remain as rows with n_pixels=0 and NaN metrics.
        """
        bounds = None
        if velocity_range is not None:
            bounds = np.asarray(velocity_range, dtype=float)
            if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] > bounds[1]:
                raise ValueError('velocity_range must be two finite bounds (min, max) in km/s with min <= max')
        tophat = self.settings['tophat']
        rows = []
        for order in self.orders:
            d = order.data
            for i, line_id in enumerate(d.line_ids):
                if bounds is not None and not np.isfinite(d.table_pixel[i]):
                    raise ValueError(f'Order {d.order}, line {line_id}: velocity_range requires finite table_pixel')
                v = (d.x[i]-d.table_pixel[i])*d.dv[i]
                selected = (d.mask[i] & np.isfinite(d.x[i]) & np.isfinite(d.y[i])
                            & np.isfinite(d.noise[i]) & (d.noise[i] > 0))
                if bounds is not None:
                    selected &= (v >= bounds[0]) & (v <= bounds[1])
                local = np.median(order.local_samples[:, i], axis=0)
                shape = order.point_shape(i)
                predicted = prediction(d.x[i, selected], d.dv[i], local, shape, tophat=tophat,
                                       model=order.model, reference=d.table_pixel[i],
                                       baseline_degree=order.baseline_degree)
                finite = np.isfinite(predicted)
                residual = d.y[i, selected][finite]-predicted[finite]
                noise, velocity = d.noise[i, selected][finite], v[selected][finite]
                row = dict(order=d.order, line_id=line_id, lab_wave=d.lab_wave[i],
                           n_pixels=len(residual), vmin_kms=np.nan, vmax_kms=np.nan,
                           rms_peak_percent=np.nan, rms_sigma=np.nan)
                if len(residual):
                    peak_v = _model_peak(shape, d.dv[i], tophat=tophat, model=order.model)
                    peak = local[2]*float(pixel_mass(peak_v, shape, d.dv[i], tophat=tophat, model=order.model))
                    if np.isfinite(peak) and peak > 0:
                        row['rms_peak_percent'] = 100*np.sqrt(np.mean(residual**2))/peak
                    row['rms_sigma'] = np.sqrt(np.mean((residual/noise)**2))
                    if np.isfinite(velocity).all():
                        row.update(vmin_kms=velocity.min(), vmax_kms=velocity.max())
                if order.chebyshev is not None:
                    peak = order.chebyshev.observed_peak[i]
                    row.update(observed_peak=peak,
                        rms_observed_peak_percent=100*np.sqrt(np.mean(residual**2))/peak if len(residual) else np.nan,
                        inlier_probability=order.chebyshev.inlier_probability[i])
                rows.append(row)
        columns = ['order', 'line_id', 'lab_wave', 'n_pixels',
                   'vmin_kms', 'vmax_kms', 'rms_peak_percent', 'rms_sigma']
        if self.settings['mode'] == 'chebyshev':
            columns += ['observed_peak', 'rms_observed_peak_percent', 'inlier_probability']
        return pd.DataFrame(rows, columns=columns)

    def plot_residual_summary(self, *, velocity_range=None):
        """Plot per-line peak-relative and noise-relative RMS against lab wavelength."""
        return _plot_residual_summary(self, velocity_range=velocity_range)

    def plot_lines(self, *, n_draws=30, seed=0, include_baseline=False):
        """Plot normalized profiles, or full flux with a dashed baseline.

        include_baseline=True retains the input flux units. Dotted components
        are each offset by the median baseline; faint draws include their own
        backgrounds, slopes and fluxes.
        """
        if self.settings['mode'] == 'chebyshev':
            from .chebyshev_plotting import plot_lines
            return plot_lines(self, n_draws=n_draws, seed=seed, include_baseline=include_baseline)
        return _plot_lines(self, n_draws=n_draws, seed=seed, include_baseline=include_baseline)

    def resolving_power(self):
        return _plot_resolving_power(self)

    def overlay(self, *, wavelength_range=None, min_inlier_probability=None):
        """Overlay profiles, optionally selecting an inclusive lab-wave range in Å.

        Selection affects the displayed lines and residuals without refitting.
        Orders without matching lines are omitted; no matches raises ValueError.
        Chebyshev fits can require P(inlier) > min_inlier_probability. Their
        model curves are evaluated at each line's wavelength.
        """
        if self.settings['mode'] == 'chebyshev':
            from .chebyshev_plotting import plot_overlay
            return plot_overlay(self, wavelength_range=wavelength_range,
                                min_inlier_probability=min_inlier_probability)
        if min_inlier_probability is not None:
            raise ValueError('Membership filtering requires mode=chebyshev')
        if self.settings['mode'] != 'common':
            raise ValueError('overlay requires mode=common')
        if wavelength_range is None:
            return _plot_overlay(self)
        bounds = np.asarray(wavelength_range, dtype=float)
        if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] > bounds[1]:
            raise ValueError('wavelength_range must be two finite bounds (min, max) in Angstrom with min <= max')
        orders = []
        for order in self.orders:
            wave = order.data.lab_wave
            selected = (wave >= bounds[0]) & (wave <= bounds[1])
            if selected.any():
                orders.append(replace(order, data=order.data.select(selected),
                                      local_samples=order.local_samples[:, selected]))
        if not orders:
            raise ValueError('No lines within wavelength_range')
        return _plot_overlay(replace(self, orders=orders))

    def save(self, output_dir, *, figures=True, overwrite=False):
        """Save parameter CSVs, posterior arrays, input metadata and optional figures."""
        output = Path(output_dir)
        if output.exists() and any(output.iterdir()) and not overwrite:
            raise FileExistsError(f'Output is not empty: {output}; choose a new directory')
        tables = dict(parameters=self.parameters, line_parameters=self.line_parameters,
                      residual_summary=self.residual_summary())
        if self.settings['mode'] == 'chebyshev':
            tables.update(line_membership=self.line_membership, individual_parameters=self.individual_parameters)
        output.mkdir(parents=True, exist_ok=True)
        for name, table in tables.items():
            table.to_csv(output/f'{name}.csv', index=False)
        metadata = dict(schema_version=4 if self.settings['mode'] == 'chebyshev' else 3, settings=self.settings,
                        orders=[r.data.order for r in self.orders], data_metadata=self.metadata)
        if self.settings['mode'] == 'chebyshev':
            metadata['diagnostics'] = self.diagnostics
            from importlib.metadata import version
            import platform
            metadata['software'] = dict(python=platform.python_version(),
                **{name: version(name) for name in ('numpy', 'scipy', 'jax', 'numpyro')})
        (output/'result.json').write_text(json.dumps(metadata, indent=2)+'\n')
        for result in self.orders:
            d = result.data
            arrays = {name: getattr(d, name) for name in DATA_FIELDS}
            arrays.update(local_samples=result.local_samples, shape_samples=result.shape_samples,
                          intrinsic_fwhm_samples=result.intrinsic_fwhm_samples)
            if result.chebyshev is not None:
                arrays.update({f'chebyshev_{name}': getattr(result.chebyshev, name)
                               for name in ChebyshevFit.array_fields})
            prefix = 'fit' if result.chebyshev is not None else 'posterior'
            np.savez_compressed(output/f'{prefix}_order_{d.order}.npz', **arrays)
        if figures:
            if self.settings['mode'] == 'chebyshev':
                from .analysis import save_analysis_figures
                save_analysis_figures(self, output/'figures')
                return output
            import matplotlib.pyplot as plt
            plots = ([('overlay', self.overlay())] if self.settings['mode'] == 'common' else
                     [(f'fits_{i+1:03d}', fig) for i, fig in enumerate(self.plot_lines())])
            for name, fig in plots:
                fig.savefig(output/f'{name}.png', dpi=180, bbox_inches='tight')
                plt.close(fig)
        return output


def load_result(output_dir):
    """Reload arrays and settings for inspection or replotting without refitting."""
    output = Path(output_dir)
    metadata = json.loads((output/'result.json').read_text())
    if metadata.get('schema_version', 1) not in (1, 2, 3, 4):
        raise ValueError('Unsupported result schema_version')
    model = metadata['settings'].get('model', 'free_centers')
    baseline_degree = metadata['settings'].get('baseline_degree', 1)
    parameter_names(model, baseline_degree)
    results = []
    for order in metadata['orders']:
        is_chebyshev = metadata['settings']['mode'] == 'chebyshev'
        prefix = 'fit' if is_chebyshev else 'posterior'
        with np.load(output/f'{prefix}_order_{order}.npz', allow_pickle=False) as saved:
            d = OrderData(**{name: saved[name] for name in DATA_FIELDS},
                          metadata=metadata.get('data_metadata', {}).get(str(order), {}))
            curve = None
            if is_chebyshev:
                curve = ChebyshevFit(**{name: saved[f'chebyshev_{name}'] for name in ChebyshevFit.array_fields},
                                    diagnostics=metadata['diagnostics'][str(order)])
            result = OrderFit(d, saved['local_samples'], saved['shape_samples'],
                              model=model, baseline_degree=baseline_degree, chebyshev=curve)
            result.intrinsic_fwhm_samples = saved['intrinsic_fwhm_samples']
            results.append(result)
    return FitResult(results, metadata['settings'])
