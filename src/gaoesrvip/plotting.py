"""Fit figures in input flux units or a fixed posterior-median normalization."""
import numpy as np
import matplotlib.pyplot as plt
from .models import pixel_mass, prediction, _unpack_shape, _model_peak

BLUE, RESIDUAL_ORANGE = '#2a62d5', '#d55e00'


def _components(v, shape, dv, *, tophat, model='free_centers'):
    s1, f, s2, delta, mu1 = _unpack_shape(shape, model)
    return ((1-f)*pixel_mass(v-mu1, [s1, 0, s1, 0], dv, tophat=tophat)/dv,
            f*pixel_mass(v-mu1-delta, [s2, 0, s2, 0], dv, tophat=tophat)/dv)


def _median_residual(lines, grid):
    stack = np.full((len(lines), len(grid)), np.nan)
    for i, (v, residual) in enumerate(lines):
        inside = (grid >= v.min()) & (grid <= v.max())
        stack[i, inside] = np.interp(grid[inside], v, residual)
    support = np.isfinite(stack).sum(axis=0) >= max(1, (len(lines)+1)//2)
    return grid[support], np.nanmedian(stack[:, support], axis=0)


def _plot_overlay(result):
    if not result.orders:
        raise ValueError('No orders to overlay')
    cols = min(3, len(result.orders))
    rows = (len(result.orders)+cols-1)//cols
    fig, axes = plt.subplots(3*rows, cols, squeeze=False, figsize=(5.2*cols, 7.25*rows),
        layout='constrained', gridspec_kw={'height_ratios': [3, 1, 1]*rows})
    velocity, grid = np.linspace(-10, 10, 2001), np.linspace(-9.5, 9.5, 381)
    residual_rows, median_rows = [[] for _ in range(rows)], [[] for _ in range(rows)]
    tophat = result.settings['tophat']
    for panel, order in enumerate(result.orders):
        row, col = divmod(panel, cols)
        top, bottom, zoom = axes[3*row:3*row+3, col]
        for j, ax in enumerate((top, bottom, zoom)):
            if j:
                ax.sharex(top)
            if col:
                ax.sharey(axes[3*row+j, 0])
        d, shape = order.data, np.median(order.shape_samples, axis=0)
        dv = np.median(d.dv)
        peak = 0. if order.model == 'lab_centered' else _model_peak(shape, dv, tophat=tophat)
        for i in range(len(d.x0)):
            v, observed = order.normalized_line(i)
            top.plot(v-peak, observed, '.-', color='black', alpha=.25, lw=.7, markersize=2.5)
        first, second = _components(velocity+peak, shape, dv, tophat=tophat, model=order.model)
        model, = top.plot(velocity, first+second, color=BLUE, lw=2)
        top.plot(velocity, first, velocity, second, color=BLUE, ls=':', lw=1)
        top.axhline(0, color='.75', lw=.6)
        title = f'Order {d.order} ({len(d.x0)} lines)'
        if order.model == 'lab_centered':
            title += ': lab_centered'
        if order.baseline_degree == 2:
            title += ' (quadratic baseline)'
        top.set(title=title, xlim=(-10, 10), xticks=[-10, -5, 0, 5, 10])
        if panel == 0:
            top.legend([model, top.lines[0]], ['common 2G', 'data'], fontsize=10)
        residuals = [(v-peak, r) for v, r in order.residual_lines(tophat=tophat)]
        for v, residual in residuals:
            bottom.plot(v, residual, '.-', color='black', alpha=.2, lw=.7, markersize=2)
            residual_rows[row].extend(residual[np.abs(v) <= 8])
        vmed, median = _median_residual(residuals, grid)
        for ax in (bottom, zoom):
            ax.plot(vmed, median, color=RESIDUAL_ORANGE, lw=2, label='median residual')
            ax.axhline(0, color='.65', lw=.7)
        if panel == 0:
            bottom.legend(fontsize=9)
        median_rows[row].extend(median[np.abs(vmed) <= 8])
        for ax in (top, bottom):
            ax.tick_params(labelbottom=False)
        if col == 0:
            for ax, label in zip((top, bottom, zoom), ('Normalized density', 'Residual', 'Median residual')):
                ax.set_ylabel(label+' [s/km]')
        origin = 'table wavelength' if order.model == 'lab_centered' else 'model peak'
        zoom.set_xlabel(f'Velocity from {origin} [km/s]')
    for row in range(rows):
        for j, values, floor, percentile in [(1, residual_rows[row], .005, 98), (2, median_rows[row], .0005, 99)]:
            extent = max(floor, 1.15*np.nanpercentile(np.abs(values), percentile))
            axes[3*row+j, 0].set_ylim(-extent, extent)
    for panel in range(len(result.orders), rows*cols):
        row, col = divmod(panel, cols)
        for ax in axes[3*row:3*row+3, col]:
            ax.set_visible(False)
    return fig


def _plot_residual_summary(result, *, velocity_range):
    summary = result.residual_summary(velocity_range=velocity_range)
    columns = ['lab_wave', 'rms_peak_percent', 'rms_sigma']
    valid = np.isfinite(summary[columns].to_numpy()).all(axis=1)
    summary = summary.loc[valid]
    if summary.empty:
        raise ValueError('No lines with finite wavelengths and residual metrics to plot')
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6.4, 6.1), layout='constrained')
    colors = [BLUE, *plt.get_cmap('tab10').colors[1:]]
    for i, (order, rows) in enumerate(summary.groupby('order', sort=False)):
        for ax, column in zip(axes, columns[1:]):
            ax.plot(rows.lab_wave, rows[column], 'o', color=colors[i % len(colors)],
                    markersize=4.5, label=f'Order {order}')
    axes[0].set_ylabel('Residual RMS / model peak [%]')
    axes[1].set_ylabel('RMS(residual / noise)')
    axes[1].axhline(1., color='.5', ls='--', lw=.8)
    axes[1].set_xlabel('Wavelength [Å]')
    for ax in axes:
        ax.set_ylim(bottom=0)
    axes[0].legend(fontsize=9)
    window = ('All valid fitted pixels' if velocity_range is None else
              f'Table velocity: {velocity_range[0]:g} to {velocity_range[1]:g} km/s')
    title = f"{result.settings['mode']} 2G ({result.settings['model']})"
    if result.settings['baseline_degree'] == 2:
        title += ', quadratic baseline'
    fig.suptitle(f'{title}\n{window}')
    return fig


def _posterior_lines(order, i, velocity, indices, *, tophat, reference=None, include_baseline=False):
    d = order.data
    flux = np.median(order.local_samples[:, i, 2])
    center = order.reference_pixel(i)
    reference = center if reference is None else reference
    x = reference+velocity/d.dv[i]
    if not len(indices):
        return np.empty((0, len(velocity)))
    curves = np.array([prediction(x, d.dv[i], order.local_samples[k, i],
        order.shape_for_line(i)[k], tophat=tophat, model=order.model,
        reference=d.table_pixel[i], baseline_degree=order.baseline_degree) for k in indices])
    if include_baseline:
        return curves
    return (curves-order.median_baseline(i, x))/flux/d.dv[i]


def _plot_lines(result, *, n_draws=30, seed=0, include_baseline=False):
    figures, rng = [], np.random.default_rng(seed)
    velocity = np.linspace(-10, 10, 2001)
    tophat = result.settings['tophat']
    for order in result.orders:
        d = order.data
        indices = rng.choice(len(order.local_samples), min(n_draws, len(order.local_samples)), replace=False)
        for start in range(0, len(d.x0), 40):
            page = range(start, min(len(d.x0), start+40))
            cols = min(5, len(page))
            rows = (len(page)+cols-1)//cols
            fig, axes = plt.subplots(rows, cols, squeeze=False, sharex=True,
                figsize=(2.8*cols, 1.8*rows+.6), layout='constrained')
            title = f"Order {d.order}: {result.settings['mode']} 2G"
            if order.model == 'lab_centered':
                title += ' (lab_centered)'
            if order.baseline_degree == 2:
                title += ' [quadratic baseline]'
            if include_baseline:
                title += ' + baseline'
            fig.suptitle(title, fontsize=13)
            for panel, (ax, i) in enumerate(zip(axes.flat, page)):
                center = order.reference_pixel(i)
                shape = np.median(order.shape_for_line(i), axis=0)
                reference = d.table_pixel[i]
                first, second = _components(velocity+(reference-center)*d.dv[i], shape, d.dv[i],
                                            tophat=tophat, model=order.model)
                total = first+second
                if include_baseline:
                    flux = np.median(order.local_samples[:, i, 2])
                    x = reference+velocity/d.dv[i]
                    baseline = order.median_baseline(i, x)
                    scale = flux*d.dv[i]
                    total, first, second = [baseline+scale*curve for curve in (total, first, second)]
                    valid = d.mask[i]
                    observed = d.y[i, valid]
                    data_velocity = (d.x[i, valid]-reference)*d.dv[i]
                else:
                    v, observed = order.normalized_line(i)
                    data_velocity = v+(center-reference)*d.dv[i]
                data, = ax.plot(data_velocity, observed, 'o', color='black', markersize=4.5, alpha=.8, zorder=4)
                model, = ax.plot(velocity, total, color=BLUE, lw=1.2, zorder=3)
                ax.plot(velocity, first, velocity, second, color=BLUE, ls=':', lw=.7, zorder=3)
                if include_baseline:
                    baseline_line, = ax.plot(velocity, baseline, color='.4', ls='--', lw=.9, zorder=3)
                for curve in _posterior_lines(order, i, velocity, indices, tophat=tophat,
                                              reference=reference, include_baseline=include_baseline):
                    ax.plot(velocity, curve, color=BLUE, alpha=.15, lw=.6, zorder=2)
                ax.set_title(f'{d.lab_wave[i]:.2f} Å', fontsize=9)
                if not include_baseline:
                    ax.axhline(0, color='.8', lw=.5)
                ax.set(xlim=(-10, 10), xticks=[-5, 0, 5])
                ax.tick_params(labelsize=8)
                if panel == 0:
                    if include_baseline:
                        ax.legend([model, data, baseline_line], ['2G + baseline', 'data', 'baseline'], fontsize=7)
                    else:
                        ax.legend([model, data], ['2G', 'data'], fontsize=7)
            for ax in axes.flat[len(page):]:
                ax.set_visible(False)
            fig.supxlabel('Velocity from table wavelength [km/s]', fontsize=11)
            ylabel = 'Flux [input units]' if include_baseline else 'Normalized density [s/km]'
            fig.supylabel(ylabel, fontsize=11)
            figures.append(fig)
    return figures


def _plot_resolving_power(result):
    r = result.parameters.query("parameter == 'R'")
    fig, ax = plt.subplots(figsize=(6.4, 4.4), layout='constrained')
    colors = [BLUE, *plt.get_cmap('tab10').colors[1:]]
    for i, order in enumerate(result.orders):
        values = r.loc[r.order == order.data.order]
        wave = order.data.lab_wave
        valid = np.isfinite(wave) & np.isfinite(values['median'].to_numpy())
        wave, values = wave[valid], values.loc[valid]
        color = colors[i % len(colors)]
        median = np.median(values['median'])
        ax.errorbar(wave, values['median'],
                    yerr=[values['median']-values.p16, values.p84-values['median']],
                    fmt='o', color=color, markersize=4.5, elinewidth=.8, capsize=2)
        ax.axhline(median, color=color, ls='--', lw=1,
                   label=f'Order {order.data.order}: median={median:.0f}')
    ax.set(xlabel='Wavelength [Å]', ylabel=r'$R = c/\mathrm{FWHM}$', title='Individual 2G')
    ax.legend(fontsize=9)
    return fig
