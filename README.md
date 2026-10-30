# gaoesrvip

Infer two-Gaussian instrumental profiles (IP) from native or wavelength-calibrated, summed GAOES-RV ThAr spectra using NumPyro NUTS.

## Installation

Python ≥ 3.10.

```bash
pip install -e .
```

## Usage

Provide a pipeline-extracted ThAr FITS file and its IRAF wavelength calibration file to
`select_lines()`. See [examples/thar_linefit.ipynb](examples/thar_linefit.ipynb) for a real-data example.

```python
from gaoesrvip import select_lines, fit

data = select_lines(
    "data/ThAr20251107_2.center.fits",
    "data/ecThAr20251107_2.center",
    orders=[104, 105], gain=1.935, read_noise=4.0,
    order_offset=100, half_window=7,
)

individual = fit(data, mode="individual", tophat=True)
individual.parameters
individual.plot_lines()
individual.resolving_power()

common = fit(data, mode="common", tophat=True)
common.parameters
common.plot_lines()
common.overlay()
```

To fit a ThAr spectrum processed with `dispcor` and `scombine`, add
`fit_spectrum`. Line selection, including the saturation threshold and preliminary
fits, still uses the central spectrum and its database:

```python
data = select_lines(
    "data/ThAr20251107_2.center.fits",
    "data/ecThAr20251107_2.center",
    fit_spectrum="data/ThAr20251107_2_ecfw.fits",
    orders=[104, 105], gain=1.935, read_noise=4.0, saturation=5000,
)
individual = fit(data, mode="individual", model="lab_centered", tophat=False)
common = fit(data, mode="common", model="lab_centered", tophat=False)
```

The bundled `data/ThAr20251107_2_ecfw.fits` is the same lamp exposure as the
central example, extracted over offsets -54 to +53 (107 one-pixel strips) and
combined with the archived position-dependent wavelength solutions using
`dispcor(log=no)` and `scombine(combine="sum", reject="none")`, without per-strip
flat division.

The summed input must be a 2D IRAF multispec FITS with unshifted linear wavelength
coordinates and a positive integer `NCOMBINE`. Apertures are matched using
`order_offset`, not the FITS beam numbers. The selected central windows are mapped
to the sum's velocity sampling, rounding outward to whole output pixels. No extra
interpolation or screening fit is applied. Uncovered windows or windows with fewer
than nine usable samples are skipped with a warning; finite zero and negative flux
samples are retained. FITS padding outside each spectrum's WCS length is excluded.

For sums, the diagonal variance in ADU squared is approximated as
`max(flux, 0)/gain + NCOMBINE*(read_noise/gain)**2`. This uses the global count even
near edges with fewer contributors and ignores interpolation weights and pixel
correlations. Native spectra use the same expression with a count of one.
`gain` is in electrons/ADU and `read_noise` in electrons. `saturation` is an ADU
selection limit on the central input, not a test of other strips or the summed flux.

Both inputs produce red-positive analysis coordinates. Decreasing-wavelength inputs
are reversed without interpolation; `x`, `x0`, and `table_pixel` then refer to the
redward analysis grid. Metadata records the mapping back to original input pixels:
`original_pixel = original_pixel_offset + original_pixel_scale * x`.
For a sum this maps to its FITS sampling, not to the raw CCD.

- `individual`: infer a separate two-Gaussian profile for each line.
- `common`: share the profile within each order. With the default model, fit background, slope, flux, and center for each line.
- `tophat=True`: integrate over a one-pixel top-hat. `False`: evaluate at pixel centers.

`select_lines()` excludes blends, saturated lines, and incomplete cutouts, then performs
one two-Gaussian fit per candidate to check the center shift and width uncertainty.
Physical orders are `aperture + order_offset`.
Set the gain, read noise, and `order_offset` for your input data.
`fit()` defaults to `warmup=500, samples=500, chains=1` and prints the MCMC summary after sampling.

## Run and save the complete analysis

`analyze_thar_ip_2g()` runs the workflow in
[thar_ip_analysis_for_jaxstar.ipynb](examples/thar_ip_analysis_for_jaxstar.ipynb)
and saves the numerical results and **all** its diagnostic figures. Install the
corner-plot dependency first (also included in the `notebook` extra):

```bash
pip install -e '.[analysis]'
```

```python
from gaoesrvip import analyze_thar_ip_2g

common = analyze_thar_ip_2g(
    thar="data/ThAr20251107_2.center.fits",
    wavelength_file="data/ecThAr20251107_2.center",
    output_dir="results/ThAr20251107_2",
    orders=[104, 105],
    gain=1.935, read_noise=4.0, saturation=5000,
)
common.parameters
```

This convenience function always fits `mode="common"`. Its defaults match the
notebook: `model="lab_centered"`, `tophat=False`, `baseline_degree=1`, and
`saturation=5000`. The lower-level `fit()` and `select_lines()` have different
defaults for model, pixel integration, and saturation. The optional `fit_spectrum` argument is also accepted. Selection options such as
`half_window` and `max_width_error`, and fit options such as `baseline_degree`,
`warmup`, `samples`, `chains`, and `seed`, can be supplied directly as keywords.

Choose an empty or new output directory; existing results are never overwritten.
Input files, the output directory, and the corner dependency are checked before
selection or MCMC. The returned object is the usual `FitResult`.

The output contains `result.json`, `parameters.csv`, `line_parameters.csv`,
`residual_summary.csv`, and `posterior_order_<order>.npz`, using the existing
`load_result()` format. The `figures/` subdirectory contains 180 dpi PNGs:

- `lines_order_<order>_<page>.png`: every line-fit page, with the baseline included.
- `overlay.png`: normalized profiles and residuals for all fitted orders.
- `corner_order_<order>.png`: the shared IP posterior for every fitted order.
- `residual_summary.png`: per-line residual RMS versus wavelength.

Numerical results are saved before plotting. If plotting fails, or you want to
regenerate figures at another resolution, reload the result and choose a new
figure directory; no selection or MCMC is repeated:

```python
from gaoesrvip import load_result, save_analysis_figures

common = load_result("results/ThAr20251107_2")
save_analysis_figures(common, "results/ThAr20251107_2/figures_300dpi", dpi=300)
```

`save_analysis_figures()` also accepts a common `FitResult` already in memory.
Its output path is the figure directory itself. It closes the figures it creates
and preserves figures already open in a notebook.

## Model choice

`fit(..., model="free_centers")` is the default and preserves the original model and
priors. Gaussian 1 has a free pixel center for each line, `delta_kms` can have either
sign, and `fraction2` lies between 0.001 and 0.49. In common mode the four shape
parameters are shared, while line centers remain independent.

Use `model="lab_centered"` to keep the existing wavelength solution fixed and measure
both Gaussian centers relative to each line's tabulated wavelength:

```python
common_lab = fit(data, mode="common", model="lab_centered", tophat=True)
common_lab.parameters
common_lab.plot_lines()
common_lab.overlay(wavelength_range=(5000, 5100))

individual_lab = fit(data, mode="individual", model="lab_centered", tophat=True)
```

The profile is `(1-f) N(v; mu1, sigma1) + f N(v; mu1+delta, sigma2)`.
Here `f = fraction2` is Gaussian 2's fraction of the integrated line flux;
Gaussian 1's fraction is `1 - fraction2`.
`delta > 0` orders the components from left to right, and `0 < f < 1` allows
either component to have the larger area. Common fits share all five parameters
within each order, including `mu1`; individual fits infer all five for each line.
Background, slope, and total flux are always fitted separately for each line.
There is no additional free line center in this model. The background is defined
at the table wavelength and the slope remains in flux per pixel.

The fixed coordinate is `v = (pixel - table_pixel) * dv`, using the existing local
velocity scale in km/s per pixel. `table_pixel` comes from inverting the supplied
wavelength solution at the tabulated wavelength. The analysis grid increases with
wavelength, so positive velocity denotes a redshift. `dv` remains the positive
sampling width in km/s. No wavelength-solution coefficients are fitted.

The model uses independent uniform priors, with `d = median(dv)` within the order:

| Parameter | Range |
| --- | --- |
| `sigma1_kms`, `sigma2_kms` | `0.4*d` to `6*d`, identical for both components |
| `fraction2` | 0 to 1 |
| `delta_kms` | 0 to `4*d` |
| `mu1_kms` | `-4*d` to `4*d` |

Individual `lab_centered` profiles can represent the same two-Gaussian shapes as
the original parameterization, but the changed prior ranges mean their posteriors
need not be identical. Common `lab_centered` fits additionally constrain the line
locations: offsets that differ between lines cannot be absorbed by independent centers.

### Baseline degree

Both models and fit modes accept `baseline_degree=1` (the default) or `2`:

```python
common_quadratic = fit(data, mode="common", model="lab_centered",
                       tophat=False, baseline_degree=2)
common_quadratic.plot_lines(include_baseline=True)
common_quadratic.overlay()
```

The baseline is `bkg + slope*u` for degree 1 and
`bkg + slope*u + quadratic*u**2` for degree 2. Here `u` is the pixel offset from
the table wavelength for `lab_centered`, or from the fitted Gaussian-1 center for
`free_centers`. Each line has its own coefficients, including in common mode.
`quadratic` is in input flux units per pixel squared and is appended to
`line_parameters` and `order.local_names`.

Its prior is uniform between `-span/L**2` and `span/L**2`, where
`span = max(max(y)-min(y), 1)` over valid pixels in that line and `L` is the largest
valid pixel distance from `table_pixel` (`lab_centered`) or `x0` (`free_centers`),
with a minimum of one pixel. The other priors retain their existing bounds.
The pixel top-hat applies to the Gaussian profile; the baseline is evaluated at
pixel centers for both settings.

The selected degree is used throughout fitting, baseline subtraction, plots,
residual diagnostics, and save/load. Reuse the same `data` to compare degrees on
identical lines and windows. This option does not change `select_lines()` or its
initial line screening, which continues to use a linear baseline.

## Results

`parameters` contains p16, median, and p84 for the profile parameters and derived quantities.
`line_parameters` contains the background, slope, and flux for each line, plus
`center` for `free_centers` results and `quadratic` when `baseline_degree=2`.
`R = c/FWHM` uses the two-Gaussian width excluding the pixel top-hat.
In `free_centers`, the profile is `(1-f) N(v; 0, sigma1) + f N(v; delta, sigma2)`.
The fitted `center` is Gaussian 1's position on the analysis grid, and
`centroid_kms = f*delta` is relative to that center.
In `lab_centered`, `parameters` also includes `mu1_kms` and the derived
`mu2_kms = mu1_kms + delta_kms`; `centroid_kms = mu1_kms + f*delta_kms` is relative
to the table wavelength. Derived quantities are computed for each posterior draw
before taking quantiles.
Representative curves use the median of each posterior parameter.

- `individual.plot_lines()` / `common.plot_lines()`: line data, the two-Gaussian model and its components, and curves drawn from the posterior.
  Common fits use the order's shared profile with each line's own local parameters.
  Zero velocity corresponds to the tabulated wavelength.
  By default, the median baseline is subtracted and the flux is normalized.
  Use `common.plot_lines(include_baseline=True)` (also available for individual fits)
  to show the original flux and the full model, with the median baseline drawn as
  a gray dashed line. Dotted curves show each Gaussian plus that baseline, and
  faint posterior curves include each draw's own baseline and flux. The baseline
  is counted once in the total model. This option only changes the display.
- `individual.resolving_power()`: R versus wavelength with p16–p84 error bars.
  Each order has its own color and a matching horizontal median line with its value in the legend.
- `common.overlay()`: three panels showing data and model, residuals, and median residuals.
  For `free_centers`, zero velocity corresponds to the peak of the displayed model;
  data are aligned using the separately fitted line centers.
  For `lab_centered`, zero is the table wavelength, exactly as in `plot_lines()`.
  Data and model retain their fitted offsets; no peak alignment is applied.
  Use `common.overlay(wavelength_range=(5000, 5100))` to display only lines whose
  tabulated wavelengths (`lab_wave`, in Å) fall within the inclusive range.
  This reuses the fitted common IP and line parameters; residual medians and the
  representative pixel width are computed from the selected lines.
  Orders with no matching lines are omitted; if no lines match, a `ValueError` is raised.

### Per-line residual diagnostics

Both models and both fit modes provide a table and a wavelength comparison plot:

```python
common_lab.residual_summary()
common_lab.residual_summary(velocity_range=(-5, 5))
common_lab.plot_residual_summary(velocity_range=(-5, 5))
```

Each row identifies the order, line ID, and `lab_wave`, and reports:

- `rms_peak_percent`: `100 * sqrt(mean(residual**2)) / model_peak`. A value of 1 means
  1% of the fitted line peak above the continuum. The peak is the maximum of the smooth
  observation-space model, using that line's pixel width and the fitted `tophat`
  setting; it is independent of where the sampled pixels fall relative to the peak.
- `rms_sigma`: `sqrt(mean((residual / noise)**2))`. This uses the noise supplied to
  the fit, with no degrees-of-freedom correction. The plot marks 1 as a noise-scale
  reference, not an automatic acceptance threshold.
- `n_pixels`, `vmin_kms`, `vmax_kms`: the number and actual table-relative velocity
  extent of the pixels used. Both metrics use the same selection.

Residuals use the median of each fitted parameter, as in the line plots. The
default uses all valid fitted pixels. `velocity_range=(min, max)` selects pixel
centers within inclusive bounds in km/s relative to the **table wavelength** for
either model; it does not recenter individual lines. Use the same range when
comparing lines to avoid diluting the RMS with different amounts of continuum.
Masked or nonfinite pixels and nonpositive/nonfinite noise are excluded. Lines
with no selected pixels remain in the table with `n_pixels=0` and NaN metrics.
For older data without `table_pixel`, the full-cutout metrics remain available,
but velocity bounds require that reference and the reported velocity extent is NaN.

`save()` also writes `residual_summary.csv` for all valid fitted pixels.
For a restricted range, save the returned table with its `to_csv()` method.

Posterior arrays are available per order. `order.shape_names` and `order.local_names`
give their last-axis labels for the selected model:

```python
order = common_lab.orders[0]
order.shape_names  # sigma1_kms, fraction2, sigma2_kms, delta_kms, mu1_kms
order.shape_samples  # (draw, 5); individual mode: (draw, line, 5)
order.local_samples  # (draw, line, 3); baseline_degree=2 adds a fourth coefficient
# With the optional corner package:
# corner.corner(order.shape_samples, labels=order.shape_names)
```

`result.save(directory)` stores the model choice, settings, posterior arrays,
parameter tables, and figures. `load_result(directory)` restores either model;
original schema-1 results without a model identifier load as `free_centers`.
Results without a baseline degree load with the original linear baseline.
The parameter label `fraction2` replaces the former `fraction` label in both models;
existing posterior arrays remain readable and use the new label when loaded.

### Input and selection metadata

`select_lines()` records the absolute paths of the spectrum and wavelength-solution
files, plus all selection arguments (including defaults): `orders`, `gain`,
`read_noise`, `order_offset`, `half_window`, `saturation`, `blend_guard`,
`resolving_power`, `max_center_shift`, and `max_width_error`.

```python
data[0].metadata
common.metadata[104]["input_files"]
common.metadata[104]["selection"]
common.save("results/common", figures=False)

from gaoesrvip import load_result
restored = load_result("results/common")
restored.metadata[104]
```

For a summed fit, input paths also identify the central selection spectrum.
Coordinate metadata specifies the redward grid and its original-pixel mapping;
noise metadata records the count and diagonal approximation.

Metadata is kept per order, so orders from different inputs or selection runs retain
their own provenance. `common.metadata` returns a copy keyed by integer order number.
`OrderData.select()` preserves the original selection metadata; the saved arrays
contain the final retained line IDs, wavelengths, and pixels after any manual filtering.
Metadata is written to `result.json` under `data_metadata` and restored without
accessing the original input files. Existing results and manually constructed data
without metadata use an empty dictionary per order; past selection conditions are
not inferred. To capture conditions automatically, create new data with the updated
`select_lines()` before fitting.
