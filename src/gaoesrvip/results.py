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
from .chebyshev import ChebyshevFit, COORDINATE_NAMES, cumulants


@dataclass
class OrderFit:
    data: OrderData
    local_samples: np.ndarray  # draw, line, parameter; MAP-only has one estimate, not posterior draws
    shape_samples: np.ndarray  # draw, [line,] shape_parameter
    baseline_degree: int = 1
    chebyshev: ChebyshevFit | None = None

    def __post_init__(self):
        local_names, shape_names = parameter_names(self.baseline_degree)
        self.local_samples = np.asarray(self.local_samples)
        self.shape_samples = np.asarray(self.shape_samples)
        nlines = len(self.data.x0)
        if (self.local_samples.ndim != 3 or self.local_samples.shape[1:] != (nlines, len(local_names))
                or self.shape_samples.ndim not in (2, 3)
                or self.shape_samples.shape[-1] != len(shape_names)
                or self.shape_samples.shape[0] != self.local_samples.shape[0]
                or (self.shape_samples.ndim == 3 and self.shape_samples.shape[1] != nlines)):
            raise ValueError('Posterior array dimensions do not match the 2G parameters')
        if self.chebyshev is not None and len(self.chebyshev.observed_peak) != nlines:
            raise ValueError('Chebyshev result does not match the selected lines')

    @property
    def local_names(self):
        return parameter_names(self.baseline_degree)[0]

    @property
    def shape_names(self):
        return parameter_names()[1]

    def reference_pixel(self, i):
        """Fixed origin for the continuum and normalized velocity of this line."""
        return self.data.table_pixel[i]

    def shape_for_line(self, index):
        return self.shape_samples if self.shape_samples.ndim == 2 else self.shape_samples[:, index]

    def point_shape(self, index):
        if self.chebyshev is not None:
            return self.chebyshev.predict(self.data.lab_wave[index:index+1], samples=False)[0, 0]
        return np.median(self.shape_for_line(index), axis=0)

    def median_baseline(self, i, x):
        return baseline(x, np.median(self.local_samples[:, i], axis=0),
                        reference=self.data.table_pixel[i], baseline_degree=self.baseline_degree)

    @cached_property
    def intrinsic_fwhm_samples(self):
        shape = self.shape_samples
        values = np.array([profile_fwhm(s) for s in shape.reshape(-1, len(self.shape_names))])
        return values.reshape(shape.shape[:-1])

    def normalized_line(self, i):
        d = self.data
        valid = d.mask[i]
        flux = np.median(self.local_samples[:, i, 2])
        center = self.reference_pixel(i)
        offset = d.x[i, valid]-center
        return offset*d.dv[i], (d.y[i, valid]-self.median_baseline(i, d.x[i, valid]))/flux/d.dv[i]

    def residual_lines(self):
        lines = []
        for i in range(len(self.data.x0)):
            velocity, observed = self.normalized_line(i)
            dv = self.data.dv[i]
            shape = self.point_shape(i)
            lines.append((velocity, observed-pixel_mass(velocity, shape, dv)/dv))
        return lines


@dataclass
class FitResult:
    orders: list[OrderFit]
    settings: dict

    def __post_init__(self):
        self.settings = dict(self.settings)
        if 'model' in self.settings or 'tophat' in self.settings:
            raise ValueError('model and tophat are no longer result settings; omit them')
        baseline_degree = self.settings.setdefault('baseline_degree', 1)
        parameter_names(baseline_degree)
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
    def degree_scores(self):
        """Per-order BIC_MAP comparison; empty if a single degree was specified."""
        return pd.DataFrame(deepcopy(self.settings.get('degree_search', {}).get('scores', [])))

    @property
    def ip_model(self):
        """Portable Chebyshev coefficients and their complete evaluation convention.

        Uses the same representative coefficients as plotted curves: MAP, or
        posterior medians when NUTS draws exist. Full draws remain in the NPZ
        result files; this compact definition contains no wavelength grid.
        """
        if self.settings.get('mode') != 'chebyshev':
            raise ValueError('ip_model requires mode=chebyshev')
        orders = {}
        for order in self.orders:
            c = order.chebyshev
            lo, hi = c.domain
            coefficients = np.median(c.coefficient_samples, axis=0) if c.has_posterior else c.coefficients
            orders[str(order.data.order)] = dict(degree=c.degree, wavelength_domain=c.domain.tolist(),
                wavelength_center=float((lo+hi)/2), wavelength_scale=float((hi-lo)/2) if hi > lo else 1.,
                coefficient_estimate='posterior_median' if c.has_posterior else 'map',
                coefficients=coefficients.tolist())
        return dict(schema_version=1, wavelength_unit='Angstrom', velocity_unit='km/s',
            velocity_origin='reference wavelength',
            basis=dict(name='Chebyshev polynomials of the first kind',
                recurrence='T_0(x)=1; T_1(x)=x; T_(k+1)(x)=2*x*T_k(x)-T_(k-1)(x)',
                normalization='x=(wavelength-wavelength_center)/wavelength_scale',
                expansion='q[j]=sum_k coefficients[k][j]*T_k(x)',
                coefficient_axes=['degree', 'coordinate'], coordinates=list(COORDINATE_NAMES)),
            shape_parameter_order=['sigma1_kms', 'fraction2', 'sigma2_kms', 'delta_kms', 'mu1_kms'],
            transforms=dict(sigma1_kms='exp(q[2])', fraction2='1/(1+exp(-q[4]))',
                sigma2_kms='exp(q[3])', delta_kms='exp(q[1])', mu1_kms='q[0]'),
            profile='(1-fraction2)*Normal(v; mu1_kms, sigma1_kms) '
                    '+ fraction2*Normal(v; mu1_kms+delta_kms, sigma2_kms)',
            profile_normalization='unit integrated area in velocity', orders=orders)

    def sample_nuts(self, *, warmup=500, samples=500, chains=1, seed=0,
                    target_accept=.95, max_tree_depth=10, progress=True):
        """Sample the stored Chebyshev MAP without repeating any optimization.

        Returns a new FitResult, preserving this result and its MAP comparison.
        Each order keeps its chosen degree; intervals are conditional on that
        degree. Also accepts saved results loaded with load_result().
        """
        from .spectral_mixture import sample_chebyshev_result
        return sample_chebyshev_result(self, warmup=warmup, samples=samples,
            chains=chains, seed=seed, target_accept=target_accept,
            max_tree_depth=max_tree_depth, progress=progress)

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

    def plot_cumulants(self, *, ci=.68, n_positions=500, zoom_outliers=True, overlay_orders=False):
        """κ1–κ5 curve and independent points colored by spectral membership.

        CI bands are pointwise posterior intervals and only appear for NUTS.
        zoom_outliers changes display limits only; clipped points use triangles.
        overlay_orders=True overlays all orders on five shared wavelength axes;
        curves, bands and points share each order's color. Small dots have
        P(inlier) > 0.5; faint crosses have P(inlier) <= 0.5.
        """
        from .chebyshev_plotting import plot_cumulants
        return plot_cumulants(self, ci=ci, n_positions=n_positions,
                              zoom_outliers=zoom_outliers, overlay_orders=overlay_orders)

    @cached_property
    def parameters(self):
        """Shape and intrinsic R; locations refer to the reference wavelength."""
        rows = []
        for o in self.orders:
            names = o.shape_names+('mu2_kms',)
            names += ('centroid_kms', 'intrinsic_fwhm_kms', 'R')
            indices = [None] if self.settings['mode'] == 'common' else range(len(o.data.x0))
            for i in indices:
                draws = o.shape_for_line(i)
                fwhm = o.intrinsic_fwhm_samples if i is None else o.intrinsic_fwhm_samples[:, i]
                identity = dict(order=o.data.order)
                if i is not None:
                    identity['line_id'] = o.data.line_ids[i]
                centroid = draws[:, 4]+draws[:, 1]*draws[:, 3]
                columns = [draws, draws[:, 4]+draws[:, 3]]
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
        correction. Both metrics use the same valid pixels and pixel-center predictions.

        velocity_range optionally sets inclusive km/s bounds relative to table_pixel
        None uses all valid fitted pixels. vmin_kms/vmax_kms report
        the actual sampled extent; they are NaN if table_pixel is unavailable.
        Lines with no selected pixels remain as rows with n_pixels=0 and NaN metrics.
        """
        bounds = None
        if velocity_range is not None:
            bounds = np.asarray(velocity_range, dtype=float)
            if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] > bounds[1]:
                raise ValueError('velocity_range must be two finite bounds (min, max) in km/s with min <= max')
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
                predicted = prediction(d.x[i, selected], d.dv[i], local, shape,
                                       reference=d.table_pixel[i],
                                       baseline_degree=order.baseline_degree)
                finite = np.isfinite(predicted)
                residual = d.y[i, selected][finite]-predicted[finite]
                noise, velocity = d.noise[i, selected][finite], v[selected][finite]
                row = dict(order=d.order, line_id=line_id, lab_wave=d.lab_wave[i],
                           n_pixels=len(residual), vmin_kms=np.nan, vmax_kms=np.nan,
                           rms_peak_percent=np.nan, rms_sigma=np.nan)
                if len(residual):
                    peak_v = _model_peak(shape, d.dv[i])
                    peak = local[2]*float(pixel_mass(peak_v, shape, d.dv[i]))
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
        """Plot peak-relative and noise-relative RMS against reference wavelength.

        Chebyshev plots use the fixed observed peak, so outliers with negligible
        fitted flux keep a meaningful denominator, and logarithmic RMS axes.
        Other modes use the model peak and linear RMS axes.
        """
        return _plot_residual_summary(self, velocity_range=velocity_range)

    def plot_rms_vs_peak(self):
        """All Chebyshev orders on one log-log axis, colored by P(inlier)."""
        if self.settings['mode'] != 'chebyshev' or not self.orders:
            raise ValueError('plot_rms_vs_peak requires a nonempty Chebyshev result')
        from .chebyshev_plotting import plot_rms_vs_peak
        return plot_rms_vs_peak(self)

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
            if 'degree_search' in self.settings:
                tables['degree_scores'] = self.degree_scores
        output.mkdir(parents=True, exist_ok=True)
        for name, table in tables.items():
            table.to_csv(output/f'{name}.csv', index=False)
        metadata = dict(schema_version=5, settings=self.settings,
                        orders=[r.data.order for r in self.orders], data_metadata=self.metadata)
        if self.settings['mode'] == 'chebyshev':
            metadata['diagnostics'] = self.diagnostics
            from importlib.metadata import version
            import platform
            metadata['software'] = dict(python=platform.python_version(),
                **{name: version(name) for name in ('numpy', 'scipy', 'jax', 'numpyro')})
        (output/'result.json').write_text(json.dumps(metadata, indent=2)+'\n')
        if self.settings['mode'] == 'chebyshev':
            (output/'ip_model.json').write_text(json.dumps(self.ip_model, indent=2)+'\n')
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


def _validate_legacy_profile_settings(settings):
    # Only old files need model flags; schema 5 fixes the profile definition.
    if settings.get('model') != 'lab_centered' or settings.get('tophat') is not False:
        raise ValueError('Unsupported saved profile: requires a fixed reference wavelength '
                         'and pixel-center evaluation; use the earlier code to inspect other results')


def load_result(output_dir):
    """Reload arrays and settings for inspection or replotting without refitting."""
    output = Path(output_dir)
    metadata = json.loads((output/'result.json').read_text())
    schema = metadata.get('schema_version', 1)
    if schema not in (1, 2, 3, 4, 5):
        raise ValueError('Unsupported result schema_version')
    if schema < 5:
        _validate_legacy_profile_settings(metadata['settings'])
        metadata['settings'].pop('model')
        metadata['settings'].pop('tophat')
    baseline_degree = metadata['settings'].get('baseline_degree', 1)
    parameter_names(baseline_degree)
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
                              baseline_degree=baseline_degree, chebyshev=curve)
            result.intrinsic_fwhm_samples = saved['intrinsic_fwhm_samples']
            results.append(result)
    return FitResult(results, metadata['settings'])
