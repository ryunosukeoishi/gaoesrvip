# gaoesrvip

Infer two-Gaussian instrumental profiles (IP) from GAOES-RV ThAr spectra using NumPyro NUTS.

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
common.overlay()
```

- `individual`: infer a separate two-Gaussian profile for each line.
- `common`: share the profile within each order; fit background, slope, flux, and center for each line.
- `tophat=True`: integrate over a one-pixel top-hat. `False`: evaluate at pixel centers.

`select_lines()` excludes blends, saturated lines, and incomplete cutouts, then performs
one two-Gaussian fit per candidate to check the center shift and width uncertainty.
Physical orders are `aperture + order_offset`.
Set the gain, read noise, and `order_offset` for your input data.
`fit()` defaults to `warmup=500, samples=500, chains=1` and prints the MCMC summary after sampling.

## Results

`parameters` contains p16, median, and p84 for the profile parameters and derived quantities.
`line_parameters` contains the background, slope, flux, and center for each line.
`R = c/FWHM` uses the two-Gaussian width excluding the pixel top-hat.
The profile is `(1-f) N(v; 0, sigma1) + f N(v; delta, sigma2)`, with `f < 0.5`.
The fitted `center` is the center of Gaussian 1.
Representative curves use the median of each posterior parameter.

- `individual.plot_lines()`: line data, the two-Gaussian model and its components, and curves drawn from the posterior.
  Zero velocity corresponds to the tabulated wavelength.
- `individual.resolving_power()`: R versus wavelength with p16–p84 error bars.
  Each order has its own color and a matching horizontal median line with its value in the legend.
- `common.overlay()`: three panels showing data and model, residuals, and median residuals.
  Zero velocity corresponds to the peak of the displayed model.
