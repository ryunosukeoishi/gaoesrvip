"""Fit figures in a fixed posterior-median normalization."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from .models import pixel_mass, prediction

BLUE, MEDIAN_BLUE = '#2a62d5', '#164a8b'


def _components(v, shape, dv, *, tophat):
    s1, f, s2, delta = shape
    return ((1-f)*pixel_mass(v, [s1, 0, s1, 0], dv, tophat=tophat)/dv,
            f*pixel_mass(v-delta, [s2, 0, s2, 0], dv, tophat=tophat)/dv)


def _model_peak(shape, dv, *, tophat):
    extent = abs(shape[3])+12*max(shape[0], shape[2])+dv
    v = np.linspace(-extent, extent, 4001)
    i = np.argmax(pixel_mass(v, shape, dv, tophat=tophat))
    return minimize_scalar(lambda x: -float(pixel_mass(x, shape, dv, tophat=tophat)/dv),
                           bounds=(v[i-1], v[i+1]), method='bounded').x


def _median_residual(lines, grid):
    stack = np.full((len(lines), len(grid)), np.nan)
    for i, (v, residual) in enumerate(lines):
        inside = (grid >= v.min()) & (grid <= v.max())
        stack[i, inside] = np.interp(grid[inside], v, residual)
    support = np.isfinite(stack).sum(axis=0) >= max(1, (len(lines)+1)//2)
    return grid[support], np.nanmedian(stack[:, support], axis=0)


def _plot_overlay(result):
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
        peak = _model_peak(shape, dv, tophat=tophat)
        for i in range(len(d.x0)):
            v, observed = order.normalized_line(i)
            top.plot(v-peak, observed, '.-', color='black', alpha=.25, lw=.7, markersize=2.5)
        first, second = _components(velocity+peak, shape, dv, tophat=tophat)
        model, = top.plot(velocity, first+second, color=BLUE, lw=2)
        top.plot(velocity, first, velocity, second, color=BLUE, ls=':', lw=1)
        top.axhline(0, color='.75', lw=.6)
        top.set(title=f'Order {d.order} ({len(d.x0)} lines)', xlim=(-10, 10), xticks=[-10, -5, 0, 5, 10])
        if panel == 0:
            top.legend([model, top.lines[0]], ['common 2G', 'data'], fontsize=10)
        residuals = [(v-peak, r) for v, r in order.residual_lines(tophat=tophat)]
        for v, residual in residuals:
            bottom.plot(v, residual, '.-', color='black', alpha=.2, lw=.7, markersize=2)
            residual_rows[row].extend(residual[np.abs(v) <= 8])
        vmed, median = _median_residual(residuals, grid)
        for ax in (bottom, zoom):
            ax.plot(vmed, median, color=MEDIAN_BLUE, lw=2)
            ax.axhline(0, color='.65', lw=.7)
        median_rows[row].extend(median[np.abs(vmed) <= 8])
        for ax in (top, bottom):
            ax.tick_params(labelbottom=False)
        if col == 0:
            for ax, label in zip((top, bottom, zoom), ('Normalized density', 'Residual', 'Median residual')):
                ax.set_ylabel(label+' [s/km]')
        zoom.set_xlabel('Velocity from model peak [km/s]')
    for row in range(rows):
        for j, values, floor, percentile in [(1, residual_rows[row], .005, 98), (2, median_rows[row], .0005, 99)]:
            extent = max(floor, 1.15*np.nanpercentile(np.abs(values), percentile))
            axes[3*row+j, 0].set_ylim(-extent, extent)
    for panel in range(len(result.orders), rows*cols):
        row, col = divmod(panel, cols)
        for ax in axes[3*row:3*row+3, col]:
            ax.set_visible(False)
    return fig


def _posterior_lines(order, i, velocity, indices, *, tophat, reference=None):
    d = order.data
    bkg, slope, flux, center = np.median(order.local_samples[:, i], axis=0)
    reference = center if reference is None else reference
    x = reference+velocity/d.dv[i]
    return np.array([(prediction(x, d.dv[i], order.local_samples[k, i],
        order.shape_for_line(i)[k], tophat=tophat)-bkg-slope*(x-center))/flux/d.dv[i] for k in indices])


def _plot_lines(result, *, n_draws=30, seed=0):
    if result.settings['mode'] != 'individual':
        raise ValueError('Use overlay() for common fits')
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
            fig.suptitle(f'Order {d.order}: individual 2G', fontsize=13)
            for panel, (ax, i) in enumerate(zip(axes.flat, page)):
                center = np.median(order.local_samples[:, i, 3])
                shape = np.median(order.shape_for_line(i), axis=0)
                reference = d.table_pixel[i]
                v, observed = order.normalized_line(i)
                first, second = _components(velocity+(reference-center)*d.dv[i], shape, d.dv[i], tophat=tophat)
                data, = ax.plot(v+(center-reference)*d.dv[i], observed, 'o', color='black', markersize=4.5, alpha=.8, zorder=4)
                model, = ax.plot(velocity, first+second, color=BLUE, lw=1.2, zorder=3)
                ax.plot(velocity, first, velocity, second, color=BLUE, ls=':', lw=.7, zorder=3)
                for curve in _posterior_lines(order, i, velocity, indices, tophat=tophat, reference=reference):
                    ax.plot(velocity, curve, color=BLUE, alpha=.15, lw=.6, zorder=2)
                ax.set_title(f'{d.lab_wave[i]:.2f} Å', fontsize=9)
                ax.axhline(0, color='.8', lw=.5)
                ax.set(xlim=(-10, 10), xticks=[-5, 0, 5])
                ax.tick_params(labelsize=8)
                if panel == 0:
                    ax.legend([model, data], ['2G', 'data'], fontsize=7)
            for ax in axes.flat[len(page):]:
                ax.set_visible(False)
            fig.supxlabel('Velocity from table wavelength [km/s]', fontsize=11)
            fig.supylabel('Normalized density [s/km]', fontsize=11)
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
