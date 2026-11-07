# gaoesrvip

Infer two-Gaussian instrumental profiles (IP) from native or wavelength-calibrated, summed GAOES-RV ThAr spectra. Common/individual fits use NumPyro NUTS; Chebyshev spectral-mixture fits use optimization with optional NUTS.

## Installation

Python ≥ 3.10.

```bash
pip install -e .
```

## Usage

Provide a pipeline-extracted ThAr FITS file and its IRAF wavelength calibration file to
`select_lines()`. See [the common-IP notebook](examples/thar_ip_analysis_common_for_jaxstar.ipynb) for a real-data example.

```python
from gaoesrvip import select_lines, fit

data = select_lines(
    "data/ThAr20251107_2.center.fits",
    "data/ecThAr20251107_2.center",
    orders=[104, 105], read_noise=4.0,
    order_offset=100, half_window=7,
)

individual = fit(data, mode="individual")
individual.parameters
individual.plot_lines()
individual.resolving_power()

common = fit(data, mode="common")
common.parameters
common.plot_lines()
common.overlay()
```

To fit a ThAr spectrum processed with `dispcor` and `scombine`, add
`fit_spectrum`. Line selection, including the saturation threshold, uses the central spectrum
and its database:

```python
data = select_lines(
    "data/ThAr20251107_2.center.fits",
    "data/ecThAr20251107_2.center",
    fit_spectrum="data/ThAr20251107_2_ecfw.fits",
    orders=[104, 105], read_noise=4.0, saturation=5000,
)
individual = fit(data, mode="individual")
common = fit(data, mode="common")
```

The bundled `data/ThAr20251107_2_ecfw.fits` is the same lamp exposure as the
central example, extracted over offsets -54 to +53 (107 one-pixel strips) and
combined with the archived position-dependent wavelength solutions using
`dispcor(log=no)` and `scombine(combine="sum", reject="none")`, without per-strip
flat division.

The summed input can be a 2D IRAF multispec FITS or a ThAr `.pkl`/`.pickle`:

```python
# Accepted by select_lines() and analyze_thar_ip_2g():
fit_spectrum = "data/ThAr20251107_1_ip.pkl"
```

A trusted local ThAr pickle must contain `data` (a DataFrame with `order`, `lam`,
and `flux`) and `header` (its FITS header). `order` is the zero-based multispec
row index; the header supplies the aperture mapping and each row's valid length.
The reader uses `flux`, not `fflux`, and discards padded tails before checking
`lam` against the linear WCS. It does not resample or apply flat/barycentric
corrections. The central selection spectrum and ecidentify database are still
required. The fitting model and noise calculation are the same as for FITS.

Both formats require unshifted linear wavelength coordinates and a positive
integer `NCOMBINE`. Apertures are matched using
`order_offset`, not the FITS beam numbers. The selected central windows are mapped
to the sum's velocity sampling, rounding outward to whole output pixels. No extra
interpolation is applied. Uncovered windows or windows with fewer
than nine usable samples are skipped with a warning; finite zero and negative flux
samples are retained. FITS padding outside each spectrum's WCS length is excluded.

Both input spectra must already be in electrons. The GAOES-RV overscan processing
applies the amplifier gains; its FITS headers can retain the original `BUNIT=ADU`.
This package does not infer units from that header or apply a gain conversion.
The `gain` argument has been removed; omit it in existing calls.

For sums, the diagonal variance in electrons squared is approximated as
`max(flux, 0) + NCOMBINE*read_noise**2`. This uses the global count even
near edges with fewer contributors and ignores interpolation weights and pixel
correlations. Native spectra use the same expression with a count of one.
`read_noise` and `saturation` are in electrons. `saturation` is a
selection limit on the central input, not a test of other strips or the summed flux.
The selection default is `51000 * 1.92 = 97920` electrons, converting the
[51,000 ADU instrument specification](https://www.o.kwasan.kyoto-u.ac.jp/inst/gaoes-rv/spec.html)
with the smaller amplifier gain. The common-analysis default remains 5000 electrons.

Both inputs produce red-positive analysis coordinates. Decreasing-wavelength inputs
are reversed without interpolation; `x`, `x0`, and `table_pixel` then refer to the
redward analysis grid. Metadata records the mapping back to original input pixels:
`original_pixel = original_pixel_offset + original_pixel_scale * x`.
For a sum this maps to its FITS sampling, not to the raw CCD.

- `individual`: infer a separate two-Gaussian profile for each line.
- `common`: share the profile within each order. Fit background, slope and flux separately for each line.
- `chebyshev`: fit a smooth wavelength-dependent IP with a whole-line spectral mixture; retain independent least-squares comparison points.

All modes use a 2G profile relative to the reference wavelength, evaluated at pixel centers.
`select_lines()` applies database, blend, saturation, finite-data and usable-window
checks, without a preliminary fitting step. `screen_fit` and its quality thresholds,
`tophat`, and the `model` selector have been removed from the API. Omit these
arguments in existing code. The profile definition is fixed; no model or
pixel-integration flags are stored in new results.
Physical orders are `aperture + order_offset`.
Set the read noise and `order_offset` for your input data.
Individual/common `fit()` defaults to `warmup=500, samples=500, chains=1` and prints the MCMC summary after sampling.

## Chebyshev spectral mixture

See [thar_ip_analysis_chebyshev_for_jaxstar.ipynb](examples/thar_ip_analysis_chebyshev_for_jaxstar.ipynb)
for a summed-spectrum example with selectable orders and FITS or pickle input.
Set `orders` in the settings cell;
diagnostics, per-order line figures and saved results follow that list.
The notebook searches degrees 0 through `max_degree` for each selected order, adopts the
smallest BIC_MAP, and optionally samples from those MAPs without refitting.
Line fits, overlays and cumulants are displayed separately for each order.
Full-window residual RMS is shown against wavelength (legend outside) and
observed peak (all orders together). These are the main diagnostics; the all-order
cumulant overlay is the final reference figure.
Comparison tables are saved rather than displayed. No prediction grid is saved.

```python
chebyshev = fit(data, mode="chebyshev", degree=4, nuts=False)
# Default baseline_degree=1.
# Each line is first optimized independently (three starts, no NUTS).
# The joint fit uses spectra directly; individual points are comparison data.

chebyshev.plot_lines(include_baseline=True)  # data, components, residuals, P(out)
chebyshev.overlay(min_inlier_probability=0.5)  # strict P(in) > 0.5, display only
chebyshev.plot_cumulants(ci=0.68, zoom_outliers=True)
chebyshev.plot_cumulants(overlay_orders=True)  # all orders on the same κ1–κ5 axes
chebyshev.line_membership
chebyshev.individual_parameters
chebyshev.residual_summary()
chebyshev.diagnostics

chebyshev.save("results/chebyshev", figures=True)
restored = load_result("results/chebyshev")  # from gaoesrvip import load_result
restored.plot_cumulants()  # no refitting or sampling
```

To choose degrees per order, supply `degrees` instead of a fixed `degree`:

```python
chebyshev_map = fit(data, mode="chebyshev", degrees=range(0, 6), nuts=False)
chebyshev_map.degree_scores  # logL, k, n_pixels, BIC_MAP, delta, selection and failure status

# Optional: run only NUTS from the stored MAPs, including after load_result().
chebyshev = chebyshev_map.sample_nuts(warmup=500, samples=500, chains=1)
```

`degrees` takes precedence over `degree` in Chebyshev mode. The search uses the
same windows for every candidate and evaluates `BIC_MAP = -2*logL(MAP) + k*log(n)`.
Here `n` is the valid pixel count and `k` counts the `5*(degree+1)` coefficients,
all line-local parameters and three mixture parameters. This is a BIC-like
heuristic at a regularized MAP, not maximum-likelihood BIC. Ties select the smaller
degree. Failed or unsupported candidates are recorded and excluded; an order
with no converged candidate raises an error. Individual comparison fits do not
contribute parameters to this likelihood. `fit(..., degrees=..., nuts=True)` also
works: NUTS runs only on the selected MAPs. The chosen degree stays fixed during
sampling, so posterior bands are conditional on that choice.

The selected result is the usual `FitResult`, even when degrees differ by order.
`settings["degree_by_order"]` records them; `degree_scores.csv` and `result.json`
preserve the complete comparison. Sampling returns a new result, leaving the
MAP result and its degree comparison intact.

For transfer to another code, `save()` writes **`ip_model.json`** for any Chebyshev
result. `result.ip_model` returns the same JSON-compatible dictionary. It includes:

- Each order's degree, coefficient matrix, wavelength domain, center and scale.
- Chebyshev first-kind recurrence and normalization `x=(wavelength-center)/scale`.
- The expansion `q[j] = sum_k coefficients[k][j] * T_k(x)`, including the full constant term.
- Coordinate order, transforms to physical 2G parameters, units and velocity origin.

These coefficients reproduce the displayed curve (MAP or posterior-median
coefficients). They do not require a saved wavelength grid or the fitting package
to evaluate. Full posterior draws remain in `fit_order_<order>.npz`; the compact
JSON contains the representative curve only. Keep the wavelength convention of
the reference database when using the coefficients externally.

[thar_ip_model_from_json.ipynb](examples/thar_ip_model_from_json.ipynb) reads this
JSON directly and plots one order's 2G parameters and cumulants without fitting.

Set `nuts=True, warmup=500, samples=500, chains=2` in the same `fit` call to
sample the joint model initialized at its MAP. Individual comparison points remain
least-squares estimates. Inspect `diagnostics` for divergences, tree-depth hits,
R-hat and effective sample size before interpreting a short NUTS run.
With NUTS, the smooth central curve uses posterior-median coefficients, and
pointwise credible bands are computed by transforming every joint posterior draw.
MAP-only results have no posterior bands; interval-table entries are NaN.
The saved format stores the MAP, actual posterior draws when present, independent
fit points/status, membership, priors, observations, and diagnostics together.

All five coordinates `(mu1, log(delta), log(sigma1), log(sigma2), logit(fraction2))`
are degree-`d` Chebyshev polynomials on each order's reference-wavelength range,
mapped to [-1, 1]. Each line's IP equals the polynomial prediction exactly.
Background, slope and total flux (and quadratic background when requested) are
local parameters. A failed individual optimization is reported, not a selection cut.

The model mixes **whole spectral windows** after summing their pixel log densities.
Both components use the same Chebyshev IP. Pixel variances are the supplied noise
variance plus `(observed_peak * s_in)**2` or `(observed_peak * s_out)**2`.
The observed peak is fixed from the data after subtracting a median edge baseline;
fitted flux cannot change this normalization. The additional scales describe
model discrepancy, not detector noise alone. `P(inlier)` is conditional on the MAP
or averaged over posterior draws for NUTS. It is not a fixed residual-tolerance cut
for comparing polynomial degrees.

Optimization and NUTS use the same priors. For `r = median(dv)`, the coefficient
intercept means are `(-0.75r, log(1.5r), log(1.2r), log(1.2r), 0)` with SDs
`(2r, 1, 1, 1, 2)`; term `k >= 1` has zero mean and SDs
`(r, .5, .5, .5, 1) / k**2`. At these mean coefficients, `f2=0.5` and
`delta=1.5r`, so the profile centroid `mu1+f2*delta` is zero. This centers the
profile at the prior-mean coefficients; it does not fix the fitted centroid.
The outlier fraction has a Beta(1,4) prior.
`s_in = .01 exp(a)`, `s_out = hypot(s_in, .05 exp(b))`, with independent
`a,b ~ Normal(0,.7)`. Local parameter bounds are the existing line-fit bounds.
An initial regularized least-squares fit uses a 2%-of-peak residual floor only to
initialize MAP; it is not a fixed likelihood scale. Its IP starts at the prior-mean
coefficients (zero centroid and zero nonconstant terms), independently of the
observed peak positions. Local background/flux seeds still use the data, and all
parameters are free to move during optimization. This is a MAP workflow, rather
than the unregularized maximum-likelihood fits in the experimental degree-search notebook.

`plot_cumulants()` shows intrinsic κ1–κ5, excluding pixel integration, with the
individual points colored by spectral `P(inlier)`. Display zoom uses the curve
and non-extreme inlier points; triangles identify clipped points. Set
`zoom_outliers=False` to show the full range. Independent points have no error bars.
With `overlay_orders=True`, curves, credible bands and points share the order's
color. Small dots indicate `P(inlier) > 0.5`; faint crosses indicate the remaining
points. This binary distinction affects display only. Clipped points use filled
triangles for inliers and faint open triangles otherwise.
`residual_summary()` additionally exposes `observed_peak`, `rms_observed_peak_percent`
and membership; the existing `rms_peak_percent` column retains its fitted-model-peak
normalization. The Chebyshev RMS plot uses the fixed observed peak to keep
outliers with negligible fitted flux readable. Both RMS metrics are unweighted diagnostics.

```python
import numpy as np
from gaoesrvip import cumulants
wave = np.linspace(*chebyshev.orders[0].chebyshev.domain, 1000)
shapes = chebyshev.predict_ip(wave, order=104)  # (draw, wavelength, 5); MAP: one estimate
kappa = cumulants(shapes)                     # (draw, wavelength, 5), units (km/s)**k
```

`analyze_thar_ip_2g(..., mode="chebyshev", max_degree=4, nuts=False)` runs the
complete selection, degree search, fit and save workflow. It tries every degree
from 0 through 4, independently selects the smallest BIC_MAP for each order,
and reuses those MAPs. Set `nuts=True` to sample only the selected models.
Alternatively, `degree=2` fixes the degree; specifying both `degree` and
`max_degree` is an error. If neither is supplied, Chebyshev mode uses degree 2.
Arbitrary candidate sets remain available through the low-level `fit(degrees=...)`.

```python
chebyshev = analyze_thar_ip_2g(
    thar="data/ThAr20251107_2.center.fits",
    wavelength_file="data/ecThAr20251107_2.center",
    fit_spectrum="data/ThAr20251107_2_ecfw.fits",
    output_dir="examples/results_chebyshev/my_run",
    orders=[104, 105], read_noise=4.0, saturation=51000 * 1.92,
    mode="chebyshev", max_degree=4, nuts=False,
)
```

The paired notebooks are
[common IP](examples/thar_ip_analysis_common_for_jaxstar.ipynb) and
[Chebyshev mixture](examples/thar_ip_analysis_chebyshev_for_jaxstar.ipynb).
They save results under `examples/results_common/` and
`examples/results_chebyshev/`, respectively.
The Chebyshev notebook ends with the one-call alternative, using the same input
and plotting settings as its step-by-step workflow.

Chebyshev exports include `degree_scores.csv`, `ip_model.json` (coefficients and
basis definition), individual fits, memberships, MAPs and optional posterior
draws. Its figures are `lines_order_<order>.png`,
`overlay_inliers_order_<order>.png`, `cumulants_order_<order>.png`,
`residual_summary.png`, `rms_vs_peak.png` (all orders on one axis), and the final
`cumulants_all_orders.png`. Orders with no lines passing the overlay cut have
no overlay figure. Use `min_inlier_probability`, `ci`, `plot_n_positions` and
`zoom_outliers` to configure these plots; the same options are accepted by
`save_analysis_figures`. `result.plot_rms_vs_peak()` provides the combined RMS
plot interactively. Chebyshev analysis does not require `corner`.

## Run and save the complete analysis

With its default `mode="common"`, `analyze_thar_ip_2g()` runs the workflow in
[thar_ip_analysis_common_for_jaxstar.ipynb](examples/thar_ip_analysis_common_for_jaxstar.ipynb)
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
    read_noise=4.0, saturation=5000,
)
common.parameters
```

This convenience function defaults to `mode="common"`, `baseline_degree=1`, and
`saturation=5000`; lower-level `select_lines()` defaults to `saturation=51000 * 1.92`.
The optional `fit_spectrum` argument is also accepted. Selection options such as
`half_window` and `blend_guard`, and fit options such as `baseline_degree`,
`warmup`, `samples`, `chains`, and `seed`, can be supplied directly as keywords.

Choose an empty or new output directory; existing results are never overwritten.
Input files, the output directory, and the corner dependency are checked before
selection or MCMC. The returned object is the usual `FitResult`.

For common mode, the output contains `result.json`, `parameters.csv`, `line_parameters.csv`,
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

## Reference-wavelength 2G profile

The wavelength solution is fixed. Both Gaussian centers are measured relative to
each line's tabulated wavelength, and the profile is evaluated at pixel centers:

```python
common_lab = fit(data, mode="common")
common_lab.parameters
common_lab.plot_lines()
common_lab.overlay(wavelength_range=(5000, 5100))

individual_lab = fit(data, mode="individual")
```

The profile is `(1-f) N(v; mu1, sigma1) + f N(v; mu1+delta, sigma2)`.
Here `f = fraction2` is Gaussian 2's fraction of the integrated line flux;
Gaussian 1's fraction is `1 - fraction2`.
`delta > 0` orders the components from left to right, and `0 < f < 1` allows
either component to have the larger area. Common fits share all five parameters
within each order, including `mu1`; individual fits infer all five for each line.
Background, slope, and total flux are always fitted separately for each line.
There is no additional free line center in this model. The background is defined
at the reference wavelength and the slope remains in flux per pixel.

The fixed coordinate is `v = (pixel - table_pixel) * dv`, using the existing local
velocity scale in km/s per pixel. `table_pixel` comes from inverting the supplied
wavelength solution at the tabulated wavelength. The analysis grid increases with
wavelength, so positive velocity denotes a redshift. `dv` remains the positive
sampling width in km/s. No wavelength-solution coefficients are fitted.

Individual/common fits use independent uniform priors, with `d = median(dv)`
within the order (Chebyshev coefficient priors are described above):

| Parameter | Range |
| --- | --- |
| `sigma1_kms`, `sigma2_kms` | `0.4*d` to `6*d`, identical for both components |
| `fraction2` | 0 to 1 |
| `delta_kms` | 0 to `4*d` |
| `mu1_kms` | `-4*d` to `4*d` |

### Baseline degree

All fit modes accept `baseline_degree=1` (the default) or `2`:

```python
common_quadratic = fit(data, mode="common", baseline_degree=2)
common_quadratic.plot_lines(include_baseline=True)
common_quadratic.overlay()
```

The baseline is `bkg + slope*u` for degree 1 and
`bkg + slope*u + quadratic*u**2` for degree 2. Here `u` is the pixel offset from
the reference wavelength. Each line has its own coefficients, including in common mode.
`quadratic` is in input flux units per pixel squared and is appended to
`line_parameters` and `order.local_names`.

Its prior is uniform between `-span/L**2` and `span/L**2`, where
`span = max(max(y)-min(y), 1)` over valid pixels in that line and `L` is the largest
valid pixel distance from `table_pixel`,
with a minimum of one pixel. The other priors retain their existing bounds.
The profile and baseline are both evaluated at pixel centers.

The selected degree is used throughout fitting, baseline subtraction, plots,
residual diagnostics, and save/load. Reuse the same `data` to compare degrees on
identical lines and windows. This option does not change `select_lines()`.

## Results

`parameters` contains p16, median, and p84 for the profile parameters and derived quantities.
`line_parameters` contains the background, slope and flux for each line, plus
`quadratic` when `baseline_degree=2`. `R = c/FWHM` uses the intrinsic two-Gaussian
width. `parameters` includes `mu1_kms` and the derived
`mu2_kms = mu1_kms + delta_kms`; `centroid_kms = mu1_kms + f*delta_kms` is relative
to the reference wavelength. Derived quantities are computed for each posterior draw
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
  Zero velocity is the reference wavelength, exactly as in `plot_lines()`.
  Data and model retain their fitted offsets; no peak alignment is applied.
  Use `common.overlay(wavelength_range=(5000, 5100))` to display only lines whose
  tabulated wavelengths (`lab_wave`, in Å) fall within the inclusive range.
  This reuses the fitted common IP and line parameters; residual medians and the
  representative pixel width are computed from the selected lines.
  Orders with no matching lines are omitted; if no lines match, a `ValueError` is raised.

### Per-line residual diagnostics

All fit modes provide a table and a wavelength comparison plot:

```python
common_lab.residual_summary()
common_lab.residual_summary(velocity_range=(-5, 5))
common_lab.plot_residual_summary(velocity_range=(-5, 5))
```

Each row identifies the order, line ID, and `lab_wave`, and reports:

- `rms_peak_percent`: `100 * sqrt(mean(residual**2)) / model_peak`. A value of 1 means
  1% of the fitted line peak above the continuum. The peak is the maximum of the smooth
  pixel-center model, using that line's pixel width; it is independent of where the sampled pixels fall relative to the peak.
- `rms_sigma`: `sqrt(mean((residual / noise)**2))`. This uses the noise supplied to
  the fit, with no degrees-of-freedom correction. The plot marks 1 as a noise-scale
  reference, not an automatic acceptance threshold.
- `n_pixels`, `vmin_kms`, `vmax_kms`: the number and actual table-relative velocity
  extent of the pixels used. Both metrics use the same selection.

Residuals use the median of each fitted parameter, as in the line plots. The
default uses all valid fitted pixels. `velocity_range=(min, max)` selects pixel
centers within inclusive bounds in km/s relative to the **reference wavelength** for
all modes; it does not recenter individual lines. Use the same range when
comparing lines to avoid diluting the RMS with different amounts of continuum.
Masked or nonfinite pixels and nonpositive/nonfinite noise are excluded. Lines
with no selected pixels remain in the table with `n_pixels=0` and NaN metrics.

`save()` also writes `residual_summary.csv` for all valid fitted pixels.
For a restricted range, save the returned table with its `to_csv()` method.

Posterior arrays are available per order. `order.shape_names` and `order.local_names`
give their last-axis labels:

```python
order = common_lab.orders[0]
order.shape_names  # sigma1_kms, fraction2, sigma2_kms, delta_kms, mu1_kms
order.shape_samples  # (draw, 5); individual mode: (draw, line, 5)
order.local_samples  # (draw, line, 3); baseline_degree=2 adds a fourth coefficient
# With the optional corner package:
# corner.corner(order.shape_samples, labels=order.shape_names)
```

`result.save(directory)` stores settings, posterior arrays, parameter tables and
figures in schema 5, which fixes the reference-wavelength, pixel-center profile
definition without model-selection flags. Compatible older results remain
readable; obsolete flags are removed on load. Results using independent line
centers or pixel integration are rejected rather than reinterpreted; use the
earlier code to inspect those files.
Results without a baseline degree use the original linear baseline.

### Input and selection metadata

`select_lines()` records the absolute paths of the spectrum and wavelength-solution
files, plus all selection arguments (including defaults): `orders`,
`read_noise`, `order_offset`, `half_window`, `saturation`, `blend_guard`,
and `resolving_power`.

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
