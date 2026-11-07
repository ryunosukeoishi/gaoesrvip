"""Run common or Chebyshev IP workflows and save their diagnostic figures."""
from dataclasses import replace
from pathlib import Path

from .fitting import fit
from .selection import select_lines
from .chebyshev import _degree


def _empty_output_directory(path):
    output = Path(path).expanduser()
    if output.exists():
        if not output.is_dir():
            raise NotADirectoryError(f'Output is not a directory: {output}')
        if any(output.iterdir()):
            raise FileExistsError(f'Output is not empty: {output}; choose a new directory')
    return output


def _require_corner():
    try:
        import corner
    except ImportError as exc:
        raise ImportError(
            "Saving all analysis figures requires corner; install gaoesrvip[analysis] "
            "(from this repository: pip install -e '.[analysis]')."
        ) from exc
    return corner


def save_analysis_figures(result, output_dir, *, dpi=180, min_inlier_probability=.5,
                          ci=.68, plot_n_positions=500, zoom_outliers=True):
    """Save common or Chebyshev diagnostic figures, without fitting again.

    output_dir is the figure directory itself and must be empty or new.
    Saves all pages of line fits in input flux units (including the baseline),
    an overlay, one corner plot per order, and the per-line residual summary.
    All figures are PNGs. Only figures created here are closed; the caller's
    existing figures and interactive plotting state are preserved.
    Common fits require the optional ``analysis`` dependencies. Chebyshev fits
    save memberships in line-panel titles, one filtered overlay and cumulant
    figure per order (with credible intervals for NUTS), full-window RMS against
    wavelength and observed peak, and a final all-order cumulant overlay.
    Chebyshev plot options match the notebook and do not require corner.
    Returns the output Path.
    """
    chebyshev = result.settings['mode'] == 'chebyshev'
    if result.settings['mode'] not in ('common', 'chebyshev'):
        raise ValueError('save_analysis_figures requires mode=common or chebyshev')
    if not result.orders:
        raise ValueError('No orders to plot')
    output = _empty_output_directory(output_dir)
    corner = None if chebyshev else _require_corner()
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    existing_figures = set(plt.get_fignums())

    def save(fig, name):
        fig.savefig(output/f'{name}.png', dpi=dpi, bbox_inches='tight')
        plt.close(fig)

    with plt.ioff():
        try:
            for order in result.orders:
                single_order = replace(result, orders=[order])
                for page, fig in enumerate(single_order.plot_lines(include_baseline=True), 1):
                    name = f'lines_order_{order.data.order}'
                    save(fig, name if chebyshev else f'{name}_{page:03d}')
                if chebyshev:
                    if (min_inlier_probability is None or
                            (order.chebyshev.inlier_probability > min_inlier_probability).any()):
                        save(single_order.overlay(min_inlier_probability=min_inlier_probability),
                             f'overlay_inliers_order_{order.data.order}')
                    save(single_order.plot_cumulants(ci=ci, n_positions=plot_n_positions,
                        zoom_outliers=zoom_outliers), f'cumulants_order_{order.data.order}')
            if not chebyshev:
                save(result.overlay(), 'overlay')
                for order in result.orders:
                    fig = corner.corner(order.shape_samples, labels=order.shape_names,
                                        quantiles=[.16, .5, .84], show_titles=True)
                    fig.suptitle(f'Order {order.data.order}', y=1.02)
                    save(fig, f'corner_order_{order.data.order}')
            save(result.plot_residual_summary(), 'residual_summary')
            if chebyshev:
                save(result.plot_rms_vs_peak(), 'rms_vs_peak')
                save(result.plot_cumulants(overlay_orders=True, ci=ci,
                    n_positions=plot_n_positions, zoom_outliers=zoom_outliers), 'cumulants_all_orders')
        finally:
            # Also close figures created by a plotting function that raised.
            for number in set(plt.get_fignums())-existing_figures:
                plt.close(number)
    return output


def analyze_thar_ip_2g(thar, wavelength_file, output_dir, *, orders, read_noise,
                       saturation=5000, order_offset=100, half_window=7, blend_guard=2.,
                       fit_spectrum=None,
                       resolving_power=65000, baseline_degree=1,
                       warmup=500, samples=500, chains=1, seed=0, target_accept=.95,
                       max_tree_depth=10, max_nfev=2000, progress=True,
                       mode='common', degree=None, max_degree=None, nuts=None, maxiter=10000,
                       min_inlier_probability=.5, ci=.68, plot_n_positions=500,
                       zoom_outliers=True):
    """Select ThAr lines, fit common or Chebyshev 2G IPs, and save results/figures.

    Defaults match ``thar_ip_analysis_common_for_jaxstar.ipynb``: a fixed reference
    wavelength, pixel-center evaluation, a linear baseline, and saturation=5000. Supply
    orders and read_noise for the input spectrum. Input flux, read_noise and
    saturation are in electrons. Selection and sampling
    keywords have the same meaning as in select_lines() and fit().
    fit_spectrum optionally supplies a wavelength-calibrated summed ThAr,
    as an IRAF multispec FITS or a ThAr .pkl/.pickle containing data/header.

    output_dir must be empty or new. Input paths, the output directory and
    corner availability are checked before selection or MCMC. Tables, settings,
    input metadata and posterior arrays use the load_result() format. All
    notebook figures are saved under output_dir/figures. Numerical results are
    saved before plotting so they remain available if figure generation fails.
    Requires the optional ``analysis`` dependencies. Returns the FitResult.
    mode='chebyshev' includes independent optimization points and a spectral
    mixture MAP fit; nuts=True optionally adds a posterior and credible bands.
    max_degree searches all degrees from zero through that value independently
    for each order, selects the smallest BIC_MAP, and samples only those selected
    MAPs when nuts=True. Alternatively, degree fixes the degree (default 2 when
    neither is supplied). degree and max_degree are mutually exclusive and only
    available in Chebyshev mode. The low-level fit() accepts arbitrary degrees.
    Chebyshev plot options are passed to save_analysis_figures().
    No preliminary fit or fit-quality cuts are applied during selection.
    Chebyshev figures do not require corner.
    """
    output = _empty_output_directory(output_dir)
    if mode not in ('common', 'chebyshev'):
        raise ValueError('analyze_thar_ip_2g mode must be common or chebyshev')
    if degree is not None and max_degree is not None:
        raise ValueError('Specify either degree or max_degree, not both')
    if mode != 'chebyshev' and (degree is not None or max_degree is not None):
        raise ValueError('degree and max_degree require mode=chebyshev')
    mode_options = {}
    if mode == 'chebyshev':
        mode_options = dict(nuts=nuts, maxiter=maxiter)
        if max_degree is None:
            mode_options['degree'] = _degree(2 if degree is None else degree)
        else:
            try:
                maximum = _degree(max_degree)
            except ValueError as exc:
                raise ValueError('max_degree must be a nonnegative integer') from exc
            mode_options['degrees'] = range(maximum+1)
    thar, wavelength_file = Path(thar).expanduser(), Path(wavelength_file).expanduser()
    inputs = [thar, wavelength_file]
    if fit_spectrum is not None:
        inputs.append(Path(fit_spectrum).expanduser())
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(f'Input file does not exist or is not a file: {path}')
    if mode == 'common':
        _require_corner()
    output.mkdir(parents=True, exist_ok=True)
    data = select_lines(
        thar, wavelength_file, orders=orders, read_noise=read_noise,
        saturation=saturation, order_offset=order_offset, half_window=half_window,
        blend_guard=blend_guard, resolving_power=resolving_power,
        fit_spectrum=fit_spectrum,
    )
    result = fit(
        data, mode=mode, baseline_degree=baseline_degree,
        warmup=warmup, samples=samples, chains=chains, seed=seed,
        target_accept=target_accept, max_tree_depth=max_tree_depth,
        max_nfev=max_nfev, progress=progress,
        **mode_options,
    )
    result.save(output, figures=False)
    save_analysis_figures(result, output/'figures', min_inlier_probability=min_inlier_probability,
                          ci=ci, plot_n_positions=plot_n_positions, zoom_outliers=zoom_outliers)
    return result
