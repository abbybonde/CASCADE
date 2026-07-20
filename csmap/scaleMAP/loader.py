"""
CASCADE HDF5 data loader for ScaleMAP.

Provides a single public function, :func:`load_scalemap_h5`, that reads the
standard CASCADE output HDF5 file and returns the arrays needed for UMAP
analysis.  No ``pickle`` is used anywhere in this module.
"""

from __future__ import annotations

import logging
from typing import Optional

import h5py
import numpy as np

__all__ = ["load_scalemap_h5"]

logger = logging.getLogger(__name__)


def load_scalemap_h5(
    path: str,
    model_dataset: str = "preprocessed_images/model",
    peak_params_dataset: str = "preprocessed_images/peak_params",
    x_axis_dataset: str = "preprocessed_images/x_axis",
) -> dict[str, np.ndarray | int | None]:
    """Load a CASCADE HDF5 output file and return the core arrays.

    Opens *path* in read-only mode using :func:`h5py.File` and reads the
    model cube, peak-parameter cube, and (optionally) the spectral axis.
    ``FileNotFoundError`` and ``OSError`` propagate unchanged if the file
    cannot be opened.

    Parameters
    ----------
    path:
        Absolute or relative path to the CASCADE HDF5 file.
    model_dataset:
        HDF5 dataset path for the reconstructed spectral model cube.
        Shape on disk must be ``(x, y, a)`` where ``a`` is the number of
        spectral channels.  Default: ``"preprocessed_images/model"``.
    peak_params_dataset:
        HDF5 dataset path for the fitted peak-parameter cube.
        Shape on disk must be ``(x, y, b)`` where ``b = max_peaks * 4``
        and the four parameters per peak are
        ``(amplitude, center, sigma, gamma)``.
        Default: ``"preprocessed_images/peak_params"``.
    x_axis_dataset:
        HDF5 dataset path for the Raman shift (wavenumber) axis.
        Shape on disk must be ``(a,)`` matching ``model_cube.shape[2]``.
        If this dataset is absent the key ``"x_axis"`` is set to ``None``
        and a ``WARNING``-level log message is emitted.
        Default: ``"preprocessed_images/x_axis"``.

    Returns
    -------
    dict with the following keys:

    ``"model_cube"`` : numpy.ndarray, shape ``(x, y, a)``, dtype ``float32``
        Reconstructed spectral model cube.
    ``"peak_params_cube"`` : numpy.ndarray, shape ``(x, y, b)``, dtype ``float32``
        Fitted peak-parameter cube.  ``b = max_peaks * 4``.
    ``"x_axis"`` : numpy.ndarray of shape ``(a,)``, dtype ``float32``, or ``None``
        Wavenumber axis.  ``None`` when the dataset is absent from the file.
    ``"max_peaks"`` : int
        Number of peak slots inferred as ``peak_params_cube.shape[2] // 4``.
        Not hardcoded; derived from the loaded data at runtime.

    Raises
    ------
    FileNotFoundError
        Propagated unchanged when *path* does not exist.
    OSError
        Propagated unchanged when the file exists but cannot be opened by
        ``h5py`` (e.g., not a valid HDF5 file, permission error).
    KeyError
        Raised (with a descriptive message naming both the missing dataset
        path and the file path) when *model_dataset* or
        *peak_params_dataset* is absent from the HDF5 file.

    Notes
    -----
    **CRIkit coordinate system.**  The ``(x, y)`` spatial dimensions of the
    returned arrays reflect CASCADE's internal reshape convention, which may
    differ from the CRIkit coordinate ordering (rows, cols).  The caller
    **must** verify axis ordering against the CRIkit coordinate system before
    use and **must not** assume that dimension 0 corresponds to rows and
    dimension 1 to columns without explicit validation.
    """
    # FileNotFoundError and OSError propagate unchanged (requirement 14.6).
    with h5py.File(path, "r") as f:
        # --- required datasets -------------------------------------------
        if model_dataset not in f:
            raise KeyError(
                f"Required dataset '{model_dataset}' not found in '{path}'."
            )
        if peak_params_dataset not in f:
            raise KeyError(
                f"Required dataset '{peak_params_dataset}' not found in '{path}'."
            )

        model_cube: np.ndarray = np.asarray(f[model_dataset], dtype=np.float32)
        peak_params_cube: np.ndarray = np.asarray(
            f[peak_params_dataset], dtype=np.float32
        )

        # --- optional spectral axis --------------------------------------
        x_axis: Optional[np.ndarray]
        if x_axis_dataset in f:
            x_axis = np.asarray(f[x_axis_dataset], dtype=np.float32)
        else:
            x_axis = None
            logger.warning(
                "Optional dataset '%s' not found in '%s'. "
                "Returning x_axis=None.",
                x_axis_dataset,
                path,
            )

    # Infer max_peaks from the loaded data — never hardcoded (req. 14.8).
    max_peaks: int = int(peak_params_cube.shape[2]) // 4

    return {
        "model_cube": model_cube,          # shape (x, y, a), float32
        "peak_params_cube": peak_params_cube,  # shape (x, y, b), float32
        "x_axis": x_axis,                  # shape (a,) or None
        "max_peaks": max_peaks,            # int
    }
