"""ThAr-only two-Gaussian IP inference for Python and notebooks."""
from .data import OrderData
from .selection import read_ecidentify, select_lines
from .fitting import fit
from .results import FitResult, load_result
from .analysis import analyze_thar_ip_2g, save_analysis_figures
from .models import density, pixel_mass, profile_fwhm

__version__ = "0.1.0"
__all__ = ["read_ecidentify", "select_lines", "OrderData", "fit",
           "FitResult", "load_result", "density", "pixel_mass", "profile_fwhm",
           "analyze_thar_ip_2g", "save_analysis_figures"]
