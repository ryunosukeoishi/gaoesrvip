"""Spectral-mixture diagnostics; membership cuts affect display only."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from .chebyshev import cumulants
from .models import prediction
from .plotting import BLUE, RESIDUAL_ORANGE, _components, _median_residual, _posterior_lines


def plot_lines(result, *, n_draws, seed, include_baseline):
    figures, rng = [], np.random.default_rng(seed)
    velocity = np.linspace(-10., 10., 801)
    tophat = result.settings['tophat']
    for order in result.orders:
        d, c = order.data, order.chebyshev
        draws = (rng.choice(len(order.local_samples), min(n_draws, len(order.local_samples)), replace=False)
                 if c.has_posterior else [])
        for start in range(0, len(d.x0), 12):
            page = list(range(start, min(start+12, len(d.x0))))
            cols, rows = min(4, len(page)), (len(page)+3)//4
            fig = plt.figure(figsize=(3.5*cols, 3.0*rows+.5), layout='constrained')
            grid = fig.add_gridspec(rows, cols)
            for panel, i in enumerate(page):
                pair = grid[panel//cols, panel%cols].subgridspec(2, 1, height_ratios=[3, 1], hspace=.03)
                top = fig.add_subplot(pair[0])
                bottom = fig.add_subplot(pair[1], sharex=top)
                local = np.median(order.local_samples[:, i], axis=0)
                shape = order.point_shape(i)
                valid = d.mask[i]
                v = (d.x[i, valid]-d.table_pixel[i])*d.dv[i]
                x = d.table_pixel[i]+velocity/d.dv[i]
                baseline = order.median_baseline(i, x)
                first, second = _components(velocity, shape, d.dv[i], tophat=tophat, model=order.model)
                scale = local[2]*d.dv[i]
                observed = d.y[i, valid]
                fitted = prediction(d.x[i, valid], d.dv[i], local, shape, tophat=tophat,
                                    model=order.model, reference=d.table_pixel[i],
                                    baseline_degree=order.baseline_degree)
                residual = 100*(observed-fitted)/c.observed_peak[i]
                if include_baseline:
                    first, second, total = baseline+scale*first, baseline+scale*second, baseline+scale*(first+second)
                    error = d.noise[i, valid]
                    top.plot(velocity, baseline, '--', color='.5', lw=.7, label='baseline')
                else:
                    total = first+second
                    observed = (observed-order.median_baseline(i, d.x[i, valid]))/scale
                    error = d.noise[i, valid]/scale
                top.errorbar(v, observed, yerr=error, fmt='.', color='k', markersize=4, lw=.6, label='data')
                top.plot(velocity, total, color=BLUE, lw=1.4, label='Chebyshev 2G')
                top.plot(velocity, first, velocity, second, color=BLUE, ls=':', lw=.7)
                for curve in _posterior_lines(order, i, velocity, draws, tophat=tophat,
                        include_baseline=include_baseline, reference=d.table_pixel[i]):
                    top.plot(velocity, curve, color=BLUE, alpha=.12, lw=.6)
                bottom.errorbar(v, residual, yerr=100*d.noise[i, valid]/c.observed_peak[i],
                                fmt='.-', color=RESIDUAL_ORANGE, lw=.6, markersize=3)
                bottom.axhline(0, color='.5', lw=.6)
                rms = np.sqrt(np.mean(residual**2))
                top.set_title(f'{d.lab_wave[i]:.4f} Å  P(out)={1-c.inlier_probability[i]:.3f}\n'
                              f'RMS / observed peak = {rms:.2f}%', fontsize=9)
                top.tick_params(labelbottom=False)
                bottom.set(xlim=(-10, 10), xticks=[-5, 0, 5])
                for ax in (top, bottom):
                    ax.tick_params(labelsize=8)
                if panel % cols == 0:
                    top.set_ylabel('Flux' if include_baseline else 'Density [s/km]', fontsize=9)
                    bottom.set_ylabel('Res. [%]', fontsize=9)
                if panel == 0:
                    top.legend(fontsize=7)
            fig.supxlabel('Velocity from laboratory wavelength [km/s]')
            fig.suptitle(f"Order {d.order}: degree {c.degree} Chebyshev mixture ({result.settings['inference'].upper()})")
            figures.append(fig)
    return figures


def plot_overlay(result, *, wavelength_range, min_inlier_probability):
    if min_inlier_probability is not None and (not np.isfinite(min_inlier_probability)
                                               or not 0 <= min_inlier_probability <= 1):
        raise ValueError('min_inlier_probability must be between 0 and 1')
    bounds = None if wavelength_range is None else np.asarray(wavelength_range, float)
    if bounds is not None and (bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] > bounds[1]):
        raise ValueError('wavelength_range must be finite ordered bounds')
    panels = []
    for o in result.orders:
        selected = np.ones(len(o.data.x0), bool)
        if bounds is not None:
            selected &= (o.data.lab_wave >= bounds[0]) & (o.data.lab_wave <= bounds[1])
        if min_inlier_probability is not None:
            selected &= o.chebyshev.inlier_probability > min_inlier_probability
        if selected.any():
            panels.append((o, np.flatnonzero(selected)))
    if not panels:
        raise ValueError('No lines satisfy the overlay selection')
    fig, axes = plt.subplots(3, len(panels), squeeze=False, figsize=(6*len(panels), 8),
        layout='constrained', gridspec_kw={'height_ratios': [3, 1, 1]})
    velocity, grid = np.linspace(-10, 10, 801), np.linspace(-9.5, 9.5, 381)
    for col, (o, indices) in enumerate(panels):
        top, bottom, zoom = axes[:, col]
        residuals = []
        for i in indices:
            v, observed = o.normalized_line(i)
            first, second = _components(velocity, o.point_shape(i), o.data.dv[i],
                                        tophat=result.settings['tophat'], model=o.model)
            top.plot(v, observed, '.-', color='k', alpha=.25, lw=.6, markersize=3)
            top.plot(velocity, first+second, color=BLUE, alpha=.3, lw=.8)
            a, b = _components(v, o.point_shape(i), o.data.dv[i],
                               tophat=result.settings['tophat'], model=o.model)
            residuals.append((v, observed-a-b))
        for v, residual in residuals:
            bottom.plot(v, residual, '.-', color='k', alpha=.25, lw=.6, markersize=2)
        vmed, median = _median_residual(residuals, grid)
        for ax in (bottom, zoom):
            ax.plot(vmed, median, color=RESIDUAL_ORANGE, lw=1.5)
            ax.axhline(0, color='.5', lw=.6)
        top.plot([], [], color='k', label='data')
        top.plot([], [], color=BLUE, label='Chebyshev IP at each line')
        top.legend(fontsize=8)
        cut = 'all lines' if min_inlier_probability is None else f'P(in) > {min_inlier_probability:g}'
        top.set_title(f'Order {o.data.order}: {len(indices)} lines, {cut}')
        for ax, label in zip((top, bottom, zoom), ('Density [s/km]', 'Residual [s/km]', 'Median residual [s/km]')):
            ax.set(xlim=(-10, 10), ylabel=label)
        top.tick_params(labelbottom=False)
        bottom.tick_params(labelbottom=False)
        zoom.set_xlabel('Velocity from laboratory wavelength [km/s]')
    return fig


def plot_cumulants(result, *, ci, n_positions, zoom_outliers):
    if not result.orders or any(o.chebyshev is None for o in result.orders):
        raise ValueError('plot_cumulants requires Chebyshev results')
    if ci is not None and (not np.isfinite(ci) or not 0 < ci < 1):
        raise ValueError('ci must be between 0 and 1, or None')
    if isinstance(n_positions, bool) or not isinstance(n_positions, (int, np.integer)) or n_positions < 2:
        raise ValueError('n_positions must be an integer >= 2')
    fig, axes = plt.subplots(5, len(result.orders), squeeze=False,
        figsize=(8*len(result.orders), 11), layout='constrained', sharex='col')
    for col, o in enumerate(result.orders):
        c, d = o.chebyshev, o.data
        wave = np.linspace(*c.domain, n_positions)
        central = cumulants(c.predict(wave, samples=False)[0])
        points = cumulants(c.individual_shape)
        low = high = None
        if c.has_posterior and ci is not None:
            low, high = np.quantile(cumulants(c.predict(wave)), [(1-ci)/2, (1+ci)/2], axis=0)
        for j, ax in enumerate(axes[:, col]):
            ax.plot(wave, central[:, j], color='k', lw=1.5, label='Chebyshev curve')
            extent = central[:, j]
            if low is not None:
                ax.fill_between(wave, low[:, j], high[:, j], color='.5', alpha=.25,
                                label=f'{100*ci:g}% pointwise credible interval')
                extent = np.r_[extent, low[:, j], high[:, j]]
            finite = c.individual_success & np.isfinite(points[:, j])
            scale_points = finite & (c.inlier_probability > .5) if zoom_outliers else finite.copy()
            if zoom_outliers and scale_points.sum() >= 4:
                q1, q3 = np.percentile(points[scale_points, j], [25, 75])
                if q3 > q1:
                    scale_points &= (points[:, j] >= q1-3*(q3-q1)) & (points[:, j] <= q3+3*(q3-q1))
            extent = np.r_[extent, points[scale_points, j]]
            lower, upper = float(np.min(extent)), float(np.max(extent))
            padding = .08*max(upper-lower, .01)
            lower, upper = lower-padding, upper+padding
            visible = finite & (points[:, j] >= lower) & (points[:, j] <= upper)
            ax.scatter(d.lab_wave[visible], points[visible, j], c=c.inlier_probability[visible],
                       cmap='coolwarm_r', vmin=0, vmax=1, s=30, edgecolors='.3', linewidths=.4,
                       label='Individual LS')
            for outside, height, marker in [(finite & (points[:, j] < lower), .025, 'v'),
                                            (finite & (points[:, j] > upper), .975, '^')]:
                ax.scatter(d.lab_wave[outside], np.full(outside.sum(), height),
                    c=c.inlier_probability[outside], cmap='coolwarm_r', vmin=0, vmax=1,
                    marker=marker, transform=ax.get_xaxis_transform(), s=35)
            ax.set(ylabel=rf'$\kappa_{{{j+1}}}$ [(km/s)$^{{{j+1}}}$]', ylim=(lower, upper))
            ax.grid(alpha=.2)
        axes[0, col].set_title(f'Order {d.order}: degree {c.degree} ({result.settings["inference"].upper()})')
        axes[0, col].legend(fontsize=8)
        axes[-1, col].set_xlabel('Laboratory wavelength [Å]')
    fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap='coolwarm_r'), ax=axes.ravel().tolist(),
                 label='P(inlier | spectra)', fraction=.025)
    if zoom_outliers:
        fig.suptitle('Intrinsic 2G cumulants; triangles mark points outside the displayed range')
    return fig
