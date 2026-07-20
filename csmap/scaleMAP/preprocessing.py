"""
Spectral preprocessing functions for ScaleMAP.

Provides three pure, stateless functions that transform raw loaded data into
UMAP-ready feature matrices:

- :func:`spectral_concat` — extract and concatenate wavenumber sub-ranges
  from a 3-D spectral image cube, with optional per-range scale factors.
- :func:`zscore_normalize` — column-wise z-score normalization of a 2-D
  feature matrix with an epsilon guard against division by zero.
- :func:`apply_background_mask` — subset rows of a feature matrix using a
  boolean mask, selecting either background-excluded or foreground-included
  pixels.
- :func:`build_peak_descriptor` — render each pixel's fitted peak set onto a
  fixed canonical wavenumber grid as a sum of pseudo-Voigt profiles, producing
  a permutation-invariant, image-independent per-pixel descriptor.
- :func:`normalize_peak_descriptor` — cross-image normalization of a peak
  descriptor matrix (per-pixel L1, reference-stats replay, or passthrough).

None of these functions mutate their inputs or carry global state.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "spectral_concat",
    "zscore_normalize",
    "apply_background_mask",
    "build_peak_descriptor",
    "normalize_peak_descriptor",
]


def spectral_concat(
    model_cube: np.ndarray,
    wavenumber_ranges: list[tuple[float, float]],
    scale_factors: list[float] | None = None,
    x_axis: np.ndarray | None = None,
) -> np.ndarray:
    """Extract and concatenate spectral sub-ranges from a 3-D image cube.

    Iterates over *wavenumber_ranges*; for each ``(start, stop)`` pair
    extracts a slice along axis 2.  If *scale_factors* is provided each
    slice is multiplied by the corresponding factor before concatenation.
    All slices are then joined along axis 2 and the concatenated array is
    returned.

    Parameters
    ----------
    model_cube:
        3-D spectral image cube of shape ``(x, y, W)`` where ``W`` is
        the total number of wavenumber channels.
    wavenumber_ranges:
        Sequence of ``(start, stop)`` pairs identifying the sub-ranges to
        extract.  Interpretation depends on *x_axis*:

        * ``x_axis is None`` — the pairs are **channel indices** (half-open,
          Python slice semantics).  They must be integers; float values
          raise a :exc:`ValueError` telling the caller to pass *x_axis*.
        * ``x_axis`` provided — the pairs are **wavenumber values** in the
          units of *x_axis* (e.g. cm⁻¹) and may be floats.  Each pair is
          converted to channel indices with :func:`numpy.searchsorted`, so
          the extracted slice covers all channels with
          ``start <= x_axis[i] < stop``.
    scale_factors:
        Optional per-range multiplicative scale factors.  When provided,
        ``len(scale_factors)`` **must** equal ``len(wavenumber_ranges)``; a
        :exc:`ValueError` is raised otherwise.  Each slice is multiplied by
        the corresponding factor *before* concatenation, allowing e.g. the
        CH-stretch region to be attenuated relative to the fingerprint region.
        Pass ``None`` (the default) to skip scaling.
    x_axis:
        Optional wavenumber axis of shape ``(W,)`` matching axis 2 of
        *model_cube*.  Must be monotonically increasing.  When provided,
        *wavenumber_ranges* are treated as physical wavenumber values.

    Returns
    -------
    numpy.ndarray
        Concatenated array of shape ``(x, y, k)`` where ``k`` is the total
        number of selected channels.

    Raises
    ------
    ValueError
        When *scale_factors* length mismatches, when float ranges are given
        without *x_axis*, when *x_axis* is not increasing or does not match
        ``model_cube.shape[2]``, or when a range selects zero channels.

    Examples
    --------
    >>> import numpy as np
    >>> cube = np.ones((10, 10, 200), dtype=np.float32)
    >>> out = spectral_concat(cube, [(0, 100), (150, 175)])
    >>> out.shape
    (10, 10, 125)

    >>> wn = np.linspace(500.0, 3500.0, 200)
    >>> out_wn = spectral_concat(cube, [(600.0, 1750.0)], x_axis=wn)
    >>> out_wn.shape[2] > 0
    True
    """
    if scale_factors is not None and len(scale_factors) != len(wavenumber_ranges):
        raise ValueError(
            f"len(scale_factors) == {len(scale_factors)} does not match "
            f"len(wavenumber_ranges) == {len(wavenumber_ranges)}."
        )

    if x_axis is not None:
        x_axis = np.asarray(x_axis).ravel()
        if x_axis.shape[0] != model_cube.shape[2]:
            raise ValueError(
                f"x_axis has {x_axis.shape[0]} entries but model_cube has "
                f"{model_cube.shape[2]} spectral channels."
            )
        if np.any(np.diff(x_axis) <= 0):
            raise ValueError(
                "x_axis must be monotonically increasing to map wavenumber "
                "ranges onto channel indices."
            )

    slices: list[np.ndarray] = []
    for i, (start, stop) in enumerate(wavenumber_ranges):
        if x_axis is not None:
            # Wavenumber values -> channel indices (half-open in wavenumber).
            start_idx = int(np.searchsorted(x_axis, start, side="left"))
            stop_idx = int(np.searchsorted(x_axis, stop, side="left"))
        else:
            if float(start) != int(start) or float(stop) != int(stop):
                raise ValueError(
                    f"wavenumber_ranges[{i}] = ({start}, {stop}) has non-integer "
                    "values but no x_axis was provided. Pass x_axis to select "
                    "by wavenumber, or use integer channel indices."
                )
            start_idx, stop_idx = int(start), int(stop)

        if stop_idx <= start_idx:
            raise ValueError(
                f"wavenumber_ranges[{i}] = ({start}, {stop}) selects zero "
                "channels."
            )

        chunk = model_cube[..., start_idx:stop_idx]
        if scale_factors is not None:
            chunk = chunk * scale_factors[i]
        slices.append(chunk)

    return np.concatenate(slices, axis=2)


def zscore_normalize(
    X: np.ndarray,
    epsilon: float = 1e-8,
    clip_percentiles: tuple[float, float] | None = None,
) -> np.ndarray:
    """Apply column-wise z-score normalization to a 2-D feature matrix.

    For each feature column ``j``:

    .. math::

        X_{\\text{out}}[:, j] = \\frac{X[:, j] - \\mu_j}{\\sigma_j + \\varepsilon}

    where :math:`\\mu_j` and :math:`\\sigma_j` are the column mean and
    standard deviation respectively.  The *epsilon* guard prevents
    division-by-zero on constant columns (e.g. a spectral band with zero
    variance across all pixels).

    Parameters
    ----------
    X:
        2-D feature matrix of shape ``(n_pixels, n_features)``.
    epsilon:
        Small positive constant added to each column standard deviation
        before division.  Default: ``1e-8``.
    clip_percentiles:
        Optional ``(low, high)`` percentiles.  When provided, each column is
        clipped to its own ``[low, high]`` percentile range **before** the
        z-score, so a handful of extreme "weird" pixels (cosmic rays, fit
        blow-ups) cannot dominate the column statistics or downstream
        clustering.  ``None`` (default) preserves the historical behaviour.

    Returns
    -------
    numpy.ndarray
        Normalized array of the same shape as *X*.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((100, 10))
    >>> X_norm = zscore_normalize(X)
    >>> X_norm.shape
    (100, 10)
    >>> abs(X_norm.mean(axis=0)).max() < 1e-5
    True
    """
    if clip_percentiles is not None:
        low, high = clip_percentiles
        if not (0.0 <= low < high <= 100.0):
            raise ValueError(
                f"clip_percentiles must satisfy 0 <= low < high <= 100; "
                f"got ({low}, {high})."
            )
        lo_vals = np.percentile(X, low, axis=0)
        hi_vals = np.percentile(X, high, axis=0)
        X = np.clip(X, lo_vals, hi_vals)

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return (X - mean) / (std + epsilon)


def apply_background_mask(
    X: np.ndarray,
    mask_flat: np.ndarray,
    mode: str,
) -> np.ndarray:
    """Subset rows of a feature matrix using a boolean pixel mask.

    Uses *mask_flat* to select a subset of rows from *X* according to
    *mode*:

    - ``'exclude_background'``: return ``X[~mask_flat]`` — retains pixels
      **not** marked by the mask (i.e. foreground pixels after lasso-marking
      the background).
    - ``'include_foreground'``: return ``X[mask_flat]`` — retains pixels
      **in** the mask (i.e. foreground pixels after lasso-marking the
      foreground directly).

    Parameters
    ----------
    X:
        2-D feature matrix of shape ``(n_pixels, n_features)``.
    mask_flat:
        1-D boolean array of shape ``(n_pixels,)`` where ``True`` indicates
        pixels belonging to the drawn lasso region.
    mode:
        Masking strategy.  Must be one of ``'exclude_background'`` or
        ``'include_foreground'``.

    Returns
    -------
    numpy.ndarray
        Row-subsetted array.  Shape is ``(n_kept, n_features)`` where
        ``n_kept = np.sum(~mask_flat)`` for ``'exclude_background'`` and
        ``n_kept = np.sum(mask_flat)`` for ``'include_foreground'``.  An
        empty result (``n_kept == 0``) is returned without error — the
        caller is responsible for checking whether the result is empty.

    Raises
    ------
    ValueError
        When *mode* is not one of the two valid strings.  The error message
        names the invalid value and lists the valid options.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.arange(20, dtype=float).reshape(10, 2)
    >>> mask = np.array([True, False, True, False, True,
    ...                  False, True, False, True, False])
    >>> apply_background_mask(X, mask, 'exclude_background').shape
    (5, 2)
    >>> apply_background_mask(X, mask, 'include_foreground').shape
    (5, 2)
    """
    if mode == "exclude_background":
        return X[~mask_flat]
    elif mode == "include_foreground":
        return X[mask_flat]
    else:
        raise ValueError(
            f"Invalid mode '{mode}'. "
            "Must be 'exclude_background' or 'include_foreground'."
        )


def _pseudo_voigt_profile(
    grid: np.ndarray,
    amplitude: np.ndarray,
    center: np.ndarray,
    sigma: np.ndarray,
    gamma: np.ndarray,
) -> np.ndarray:
    """Evaluate unit-height pseudo-Voigt profiles on *grid* (broadcasting).

    Uses the Thompson–Cox–Hastings (1987) single-profile approximation to the
    Voigt function: the Gaussian FWHM ``fG = 2*sigma*sqrt(2*ln 2)`` and
    Lorentzian FWHM ``fL = 2*gamma`` are combined into an effective FWHM ``f``
    and mixing fraction ``eta``, and the profile is

        pV(x) = amplitude * [eta * L(x; f) + (1 - eta) * G(x; f)]

    where ``L`` and ``G`` are unit-height Lorentzian and Gaussian profiles of
    FWHM ``f`` centred on *center*.  ``amplitude`` is therefore the peak
    height, matching CASCADE's height-normalized amplitude convention.

    Parameters are arrays of shape ``(n, 1)`` (peaks) and *grid* has shape
    ``(len(grid),)``; the result has shape ``(n, len(grid))``.  Peaks whose
    effective FWHM is not strictly positive (degenerate zero-width fits)
    contribute zero.
    """
    fg = 2.0 * np.sqrt(2.0 * np.log(2.0)) * np.abs(sigma)
    fl = 2.0 * np.abs(gamma)
    f = (
        fg**5
        + 2.69269 * fg**4 * fl
        + 2.42843 * fg**3 * fl**2
        + 4.47163 * fg**2 * fl**3
        + 0.07842 * fg * fl**4
        + fl**5
    ) ** 0.2

    valid = f > 0.0
    f_safe = np.where(valid, f, 1.0)

    ratio = fl / f_safe
    eta = 1.36603 * ratio - 0.47719 * ratio**2 + 0.11116 * ratio**3

    dx = grid[np.newaxis, :] - center
    lorentz = 1.0 / (1.0 + (2.0 * dx / f_safe) ** 2)
    gauss = np.exp(-4.0 * np.log(2.0) * (dx / f_safe) ** 2)

    profile = amplitude * (eta * lorentz + (1.0 - eta) * gauss)
    return np.where(valid, profile, 0.0)


def build_peak_descriptor(
    peak_params_cube: np.ndarray,
    x_axis: np.ndarray | None,
    *,
    amp_threshold: float,
    grid: np.ndarray,
    sigma_render: str | float = "fitted",
    chunk_size: int = 4096,
) -> np.ndarray:
    """Render each pixel's fitted peak set onto a fixed canonical grid.

    For every pixel, each **real** peak — a 4-tuple ``(amplitude, center,
    sigma, gamma)`` whose ``amplitude > amp_threshold`` (zero-padded slots are
    excluded) — is rendered as a pseudo-Voigt profile onto the shared
    wavenumber *grid* and the profiles are summed.  The result is a dense
    per-pixel descriptor in which channel *i* means the same wavenumber for
    every pixel of every dataset, so descriptors from different images (and
    different ``max_peaks`` values) are directly comparable.

    Properties (by construction):

    * **Permutation-invariant** — summation over peaks is commutative, so
      reordering a pixel's peak slots leaves its descriptor row unchanged.
    * **Fixed width, image-independent** — the output width is
      ``len(grid)`` regardless of ``max_peaks``, so one dataset's pixels can
      be embedded into another dataset's reference frame.

    Parameters
    ----------
    peak_params_cube:
        Peak-parameter array of shape ``(x, y, max_peaks * 4)`` (a CASCADE
        cube) or ``(n_pixels, max_peaks * 4)`` (already flattened).  Each
        4-block is ``(amplitude, center, sigma, gamma)``; slot order within a
        pixel is arbitrary.
    x_axis:
        Measured wavenumber axis of the source image, shape ``(a,)``, or
        ``None``.  Used only as a sanity check: when provided, *grid* values
        outside ``[x_axis.min(), x_axis.max()]`` indicate a grid/data mismatch
        and a :exc:`ValueError` is raised if the two ranges do not overlap at
        all.  It never affects the rendered values — the canonical *grid* is
        deliberately independent of any one image's axis.
    amp_threshold:
        Minimum amplitude for a peak slot to count as a real peak (CASCADE
        production default ``1e-3``).  Slots at or below the threshold are
        excluded from rendering.
    grid:
        1-D array of canonical wavenumber values, **shared by all pixels and
        all images** that are to be compared or co-embedded.  Comes from
        configuration (e.g. ``DESCRIPTOR_GRID``) — never derive it from a
        particular image.
    sigma_render:
        ``"fitted"`` (default) renders each peak with its own fitted
        ``sigma``/``gamma`` widths.  A positive float renders every peak as a
        pure Gaussian of that sigma (in wavenumber units), ignoring the
        fitted widths — useful to decouple embedding behaviour from width
        estimation noise.
    chunk_size:
        Number of pixels rendered per vectorized block (memory/speed
        trade-off only; the result is independent of this value).

    Returns
    -------
    numpy.ndarray
        Descriptor matrix of shape ``(n_pixels, len(grid))``, dtype
        ``float32``, where ``n_pixels = x * y`` for cube input.  Pixels with
        no real peaks yield an all-zero row.

    Raises
    ------
    ValueError
        If the trailing dimension of *peak_params_cube* is not a multiple of
        4, if *grid* is not 1-D or empty, if *sigma_render* is invalid, or if
        *grid* and *x_axis* do not overlap at all.
    """
    grid = np.asarray(grid, dtype=np.float64)
    if grid.ndim != 1 or grid.size == 0:
        raise ValueError("grid must be a non-empty 1-D array of wavenumbers.")

    if peak_params_cube.shape[-1] % 4 != 0:
        raise ValueError(
            f"Trailing dimension {peak_params_cube.shape[-1]} of "
            "peak_params_cube is not a multiple of 4 "
            "(expected max_peaks * 4 with (amplitude, center, sigma, gamma) blocks)."
        )

    if isinstance(sigma_render, str):
        if sigma_render != "fitted":
            raise ValueError(
                f"Invalid sigma_render '{sigma_render}'. "
                "Must be 'fitted' or a positive float."
            )
    elif not (float(sigma_render) > 0.0):
        raise ValueError("sigma_render as a float must be positive.")

    if x_axis is not None:
        x_axis = np.asarray(x_axis)
        if grid.min() > x_axis.max() or grid.max() < x_axis.min():
            raise ValueError(
                f"grid range [{grid.min():g}, {grid.max():g}] does not overlap "
                f"the measured x_axis range [{x_axis.min():g}, {x_axis.max():g}]."
            )

    flat = peak_params_cube.reshape(-1, peak_params_cube.shape[-1])
    n_pixels = flat.shape[0]
    max_peaks = flat.shape[1] // 4
    peaks = flat.reshape(n_pixels, max_peaks, 4).astype(np.float64)

    descriptor = np.zeros((n_pixels, grid.size), dtype=np.float32)

    for start in range(0, n_pixels, chunk_size):
        stop = min(start + chunk_size, n_pixels)
        block = peaks[start:stop]                      # (m, max_peaks, 4)
        amp = block[:, :, 0]
        real = amp > amp_threshold                     # (m, max_peaks)

        acc = np.zeros((stop - start, grid.size), dtype=np.float64)
        for s in range(max_peaks):
            sel = real[:, s]
            if not sel.any():
                continue
            a = amp[sel, s][:, np.newaxis]
            c = block[sel, s, 1][:, np.newaxis]
            if sigma_render == "fitted":
                sg = block[sel, s, 2][:, np.newaxis]
                gm = block[sel, s, 3][:, np.newaxis]
            else:
                sg = np.full_like(a, float(sigma_render))
                gm = np.zeros_like(a)
            acc[sel] += _pseudo_voigt_profile(grid, a, c, sg, gm)

        descriptor[start:stop] = acc.astype(np.float32)

    return descriptor


def normalize_peak_descriptor(
    D: np.ndarray,
    *,
    mode: str,
    stats: dict | None = None,
) -> tuple[np.ndarray, dict]:
    """Normalize a peak-descriptor matrix for cross-image comparability.

    Parameters
    ----------
    D:
        Descriptor matrix of shape ``(n_pixels, n_channels)`` from
        :func:`build_peak_descriptor`.
    mode:
        * ``'l1_per_pixel'`` — scale each pixel's descriptor row to unit L1
          norm.  This removes absolute-intensity/gain differences between
          acquisitions so that peak *composition*, not magnitude, is the
          signal.  **Default choice for cross-image comparability.**
          All-zero rows (no real peaks) are left as zeros.
        * ``'reference'`` — replay the normalization recorded in *stats* from
          a designated **reference** dataset (not the current image).  The
          *stats* dict must be one previously returned by this function;
          raises :exc:`ValueError` when ``stats is None``.
        * ``'none'`` — passthrough (a float copy of *D* is still returned so
          callers can mutate safely).
    stats:
        Only used with ``mode='reference'``: the stats dict returned by the
        call that normalized the reference dataset.

    Returns
    -------
    tuple[numpy.ndarray, dict]
        ``(D_normalized, stats_used)``.  ``stats_used`` is a self-describing
        dict — at minimum ``{'mode': <effective mode>}`` — that can be
        persisted (e.g. inside a reference bundle) and passed back later with
        ``mode='reference'`` so that every subsequent image is normalized
        exactly like the reference.

    Raises
    ------
    ValueError
        Unknown *mode*; ``mode='reference'`` with ``stats=None`` or with a
        stats dict whose recorded mode is unknown.
    """
    if mode == "reference":
        if stats is None:
            raise ValueError(
                "mode='reference' requires the stats dict returned when the "
                "reference dataset was normalized (got stats=None)."
            )
        effective_mode = stats.get("mode")
        if effective_mode not in ("l1_per_pixel", "none"):
            raise ValueError(
                f"Reference stats record unknown normalization mode "
                f"'{effective_mode}'. Expected 'l1_per_pixel' or 'none'."
            )
    elif mode in ("l1_per_pixel", "none"):
        effective_mode = mode
    else:
        raise ValueError(
            f"Invalid mode '{mode}'. "
            "Must be 'l1_per_pixel', 'reference', or 'none'."
        )

    if effective_mode == "l1_per_pixel":
        # Per-pixel operation: no dataset-level statistics are needed, so the
        # reference replay is the identical operation on the new image.
        norms = np.abs(D).sum(axis=1, keepdims=True)
        safe = np.where(norms > 0.0, norms, 1.0)
        D_out = (D / safe).astype(np.float32, copy=False)
    else:  # 'none'
        D_out = np.array(D, dtype=np.float32, copy=True)

    return D_out, {"mode": effective_mode}
