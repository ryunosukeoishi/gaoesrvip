"""Run the common-IP notebook workflow and save all its diagnostic figures."""
from dataclasses import replace
from pathlib import Path

from .fitting import fit
from .selection import select_lines


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


def save_analysis_figures(result, output_dir, *, dpi=180):
    """Save every diagnostic figure from a common fit, without fitting again.

    output_dir is the figure directory itself and must be empty or new.
    Saves all pages of line fits in input flux units (including the baseline),
    an overlay, one corner plot per order, and the per-line residual summary.
    All figures are PNGs. Only figures created here are closed; the caller's
    existing figures and interactive plotting state are preserved.
    Requires the optional ``analysis`` dependencies. Returns the output Path.
    """
    if result.settings['mode'] != 'common':
        raise ValueError('save_analysis_figures requires mode=common')
    if not result.orders:
        raise ValueError('No orders to plot')
    output = _empty_output_directory(output_dir)
    corner = _require_corner()
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
                    save(fig, f'lines_order_{order.data.order}_{page:03d}')
            save(result.overlay(), 'overlay')
            for order in result.orders:
                fig = corner.corner(order.shape_samples, labels=order.shape_names,
                                    quantiles=[.16, .5, .84], show_titles=True)
                fig.suptitle(f'Order {order.data.order}', y=1.02)
                save(fig, f'corner_order_{order.data.order}')
            save(result.plot_residual_summary(), 'residual_summary')
        finally:
            # Also close figures created by a plotting function that raised.
            for number in set(plt.get_fignums())-existing_figures:
                plt.close(number)
    return output


def analyze_thar_ip_2g(thar, wavelength_file, output_dir, *, orders, gain, read_noise,
                       saturation=5000, order_offset=100, half_window=7, blend_guard=2.,
                       resolving_power=65000, max_center_shift=1., max_width_error=.1,
                       model='lab_centered', tophat=False, baseline_degree=1,
                       warmup=500, samples=500, chains=1, seed=0, target_accept=.95,
                       max_tree_depth=10, max_nfev=2000, progress=True):
    """Select ThAr lines, fit a common two-Gaussian IP, and save results and figures.

    Defaults match ``thar_ip_analysis_for_jaxstar.ipynb``: lab_centered,
    no pixel integration, a linear baseline, and saturation=5000. Supply
    orders, gain and read_noise for the input spectrum. Selection and sampling
    keywords have the same meaning as in select_lines() and fit().

    output_dir must be empty or new. Input paths, the output directory and
    corner availability are checked before selection or MCMC. Tables, settings,
    input metadata and posterior arrays use the load_result() format. All
    notebook figures are saved under output_dir/figures. Numerical results are
    saved before plotting so they remain available if figure generation fails.
    Requires the optional ``analysis`` dependencies. Returns the FitResult.
    """
    output = _empty_output_directory(output_dir)
    thar, wavelength_file = Path(thar).expanduser(), Path(wavelength_file).expanduser()
    for path in (thar, wavelength_file):
        if not path.is_file():
            raise FileNotFoundError(f'Input file does not exist or is not a file: {path}')
    _require_corner()
    output.mkdir(parents=True, exist_ok=True)
    data = select_lines(
        thar, wavelength_file, orders=orders, gain=gain, read_noise=read_noise,
        saturation=saturation, order_offset=order_offset, half_window=half_window,
        blend_guard=blend_guard, resolving_power=resolving_power,
        max_center_shift=max_center_shift, max_width_error=max_width_error,
    )
    result = fit(
        data, mode='common', model=model, tophat=tophat, baseline_degree=baseline_degree,
        warmup=warmup, samples=samples, chains=chains, seed=seed,
        target_accept=target_accept, max_tree_depth=max_tree_depth,
        max_nfev=max_nfev, progress=progress,
    )
    result.save(output, figures=False)
    save_analysis_figures(result, output/'figures')
    return result
