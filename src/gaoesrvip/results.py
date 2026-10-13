"""Inspectable results, portable serialization, and parameter tables."""
from dataclasses import dataclass, replace
from functools import cached_property
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .data import DATA_FIELDS, OrderData
from .models import C_KMS, LOCAL_NAMES, SHAPE_NAMES, pixel_mass, profile_fwhm
from .plotting import _plot_lines, _plot_overlay, _plot_resolving_power


@dataclass
class OrderFit:
    data: OrderData
    local_samples: np.ndarray  # draw, line, 4; center is absolute pixel
    shape_samples: np.ndarray  # draw, 4 (common) or draw, line, 4

    def shape_for_line(self, index):
        return self.shape_samples if self.shape_samples.ndim == 2 else self.shape_samples[:, index]

    @cached_property
    def intrinsic_fwhm_samples(self):
        shape = self.shape_samples
        values = np.array([profile_fwhm(s) for s in shape.reshape(-1, 4)])
        return values.reshape(shape.shape[:-1])

    def normalized_line(self, i):
        d = self.data
        valid = d.mask[i]
        bkg, slope, flux, center = np.median(self.local_samples[:, i], axis=0)
        offset = d.x[i, valid]-center
        return offset*d.dv[i], (d.y[i, valid]-bkg-slope*offset)/flux/d.dv[i]

    def residual_lines(self, *, tophat):
        lines = []
        for i in range(len(self.data.x0)):
            velocity, observed = self.normalized_line(i)
            dv = self.data.dv[i]
            shape = np.median(self.shape_for_line(i), axis=0)
            lines.append((velocity, observed-pixel_mass(velocity, shape, dv, tophat=tophat)/dv))
        return lines


@dataclass
class FitResult:
    orders: list[OrderFit]
    settings: dict

    def _interval(self, values):
        finite = values[np.isfinite(values)]
        if not len(finite):
            return dict(p16=np.nan, median=np.nan, p84=np.nan)
        return dict(zip(('p16', 'median', 'p84'), np.percentile(finite, [16, 50, 84])))

    @cached_property
    def parameters(self):
        """Shape and R = c/FWHM, excluding the pixel top-hat."""
        rows = []
        names = SHAPE_NAMES + ('centroid_kms', 'intrinsic_fwhm_kms', 'R')
        for o in self.orders:
            indices = [None] if self.settings['mode'] == 'common' else range(len(o.data.x0))
            for i in indices:
                draws = o.shape_for_line(i)
                fwhm = o.intrinsic_fwhm_samples if i is None else o.intrinsic_fwhm_samples[:, i]
                identity = dict(order=o.data.order)
                if i is not None:
                    identity['line_id'] = o.data.line_ids[i]
                values = np.column_stack((draws, draws[:, 1]*draws[:, 3], fwhm, C_KMS/fwhm))
                rows.extend(dict(**identity, parameter=name, **self._interval(column))
                            for name, column in zip(names, values.T))
        return pd.DataFrame(rows)

    @cached_property
    def line_parameters(self):
        rows = []
        for result in self.orders:
            for i, line_id in enumerate(result.data.line_ids):
                for k, name in enumerate(LOCAL_NAMES):
                    rows.append(dict(order=result.data.order, line_id=line_id, parameter=name,
                                     **self._interval(result.local_samples[:, i, k])))
        return pd.DataFrame(rows)

    def plot_lines(self, *, n_draws=30, seed=0):
        return _plot_lines(self, n_draws=n_draws, seed=seed)

    def resolving_power(self):
        return _plot_resolving_power(self)

    def overlay(self, *, wavelength_range=None):
        """Overlay common fits, optionally selecting an inclusive lab-wave range in Å.

        Selection affects the displayed lines and residuals without refitting.
        Orders without matching lines are omitted; no matches raises ValueError.
        """
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
        """Save parameter CSVs, posterior arrays and optional figures."""
        output = Path(output_dir)
        if output.exists() and any(output.iterdir()) and not overwrite:
            raise FileExistsError(f'Output is not empty: {output}; choose a new directory')
        tables = dict(parameters=self.parameters, line_parameters=self.line_parameters)
        output.mkdir(parents=True, exist_ok=True)
        for name, table in tables.items():
            table.to_csv(output/f'{name}.csv', index=False)
        metadata = dict(schema_version=1, settings=self.settings,
                        orders=[r.data.order for r in self.orders])
        (output/'result.json').write_text(json.dumps(metadata, indent=2)+'\n')
        for result in self.orders:
            d = result.data
            arrays = {name: getattr(d, name) for name in DATA_FIELDS}
            arrays.update(local_samples=result.local_samples, shape_samples=result.shape_samples,
                          intrinsic_fwhm_samples=result.intrinsic_fwhm_samples)
            np.savez_compressed(output/f'posterior_order_{d.order}.npz', **arrays)
        if figures:
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
    results = []
    for order in metadata['orders']:
        with np.load(output/f'posterior_order_{order}.npz', allow_pickle=False) as saved:
            d = OrderData(**{name: saved[name] for name in DATA_FIELDS})
            result = OrderFit(d, saved['local_samples'], saved['shape_samples'])
            result.intrinsic_fwhm_samples = saved['intrinsic_fwhm_samples']
            results.append(result)
    return FitResult(results, metadata['settings'])
