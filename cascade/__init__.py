"""CASCADE — BCARS hyperspectral spectral fitting package."""

from .dataset_utils import load_h5_file, save_h5_file, voigt_peak, RamanDataset
from .tidytorch_utils import (
    estimate_fit_characteristics,
    process_conv_deriv_fit,
    denoise_spectrum,
    init_sweep_context,
    pseudo_voigt,
)

__all__ = [
    "load_h5_file",
    "save_h5_file",
    "voigt_peak",
    "RamanDataset",
    "estimate_fit_characteristics",
    "process_conv_deriv_fit",
    "denoise_spectrum",
    "init_sweep_context",
    "pseudo_voigt",
]
