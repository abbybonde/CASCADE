"""
visualizer/colormaps.py
=======================
Canonical color-mapping utilities for hyperspectral UMAP visualizations.

Public API
----------
recolor_image_with_umap  -- single canonical RGBA colorizer (red_blue or lab)
simulate_rg_colorblind   -- red-green colorblind simulation helper
_hv_color_grid01         -- HSV inscribed-circle color grid helper (private)
"""

from __future__ import annotations

import numpy as np
from skimage import color as skcolor


# ---------------------------------------------------------------------------
# Public: recolor_image_with_umap
# ---------------------------------------------------------------------------

def recolor_image_with_umap(
    np_hyperspectral_img: np.ndarray,
    embedding: np.ndarray,
    color_mapping: str = 'red_blue',
    mask: np.ndarray | None = None,
    clip_percentiles: tuple[float, float] | None = None,
) -> np.ndarray:
    """Recolor a hyperspectral image using UMAP embedding coordinates.

    Parameters
    ----------
    np_hyperspectral_img : np.ndarray
        Hyperspectral image cube of shape ``(rows, cols, W)``.
    embedding : np.ndarray
        UMAP 2-D coordinates of shape ``(n_pixels, 2)`` where
        ``n_pixels == rows * cols`` when no mask is supplied, or
        ``n_pixels == mask.sum()`` when a mask is supplied.
    color_mapping : {'red_blue', 'lab'}
        Colorization scheme.
    clip_percentiles : (float, float) or None
        Optional contrast control.  When supplied, each embedding axis is
        clipped to its ``[low, high]`` percentile range (computed over the
        supplied pixels) *before* the min–max normalization below, so
        embedding outliers stop compressing the color range of the bulk of
        the image.  ``None`` (default) preserves the historical behaviour
        (no clipping).

        * ``'red_blue'`` — UMAP axis 0 → red channel, axis 1 → blue channel.
          Returns dtype ``float32``, shape ``(rows, cols, 4)``.
        * ``'lab'`` — UMAP axes mapped to CIE L*a*b* *a* and *b* channels,
          converted to RGB via :func:`skimage.color.lab2rgb`.
          Returns dtype ``float64``, shape ``(rows, cols, 4)``.

        All returned values are in ``[0.0, 1.0]``.
    mask : np.ndarray or None
        Optional boolean mask of shape ``(rows, cols)``.  When supplied,
        ``embedding[i]`` corresponds to the *i*-th ``True`` pixel in raster
        order.  Unmasked pixels receive ``RGBA = (0, 0, 0, 0)``.  When
        ``None``, all pixels are considered foreground.

    Returns
    -------
    np.ndarray
        RGBA image of shape ``(rows, cols, 4)``.
        dtype is ``float32`` for ``'red_blue'``, ``float64`` for ``'lab'``.
        All values lie in ``[0.0, 1.0]``.
    """
    rows, cols = np_hyperspectral_img.shape[0], np_hyperspectral_img.shape[1]

    if mask is None:
        mask = np.full((rows, cols), True, dtype=bool)

    if clip_percentiles is not None:
        low, high = clip_percentiles
        if not (0.0 <= low < high <= 100.0):
            raise ValueError(
                f"clip_percentiles must satisfy 0 <= low < high <= 100; "
                f"got ({low}, {high})."
            )
        lo_vals = np.percentile(embedding, low, axis=0)
        hi_vals = np.percentile(embedding, high, axis=0)
        embedding = np.clip(embedding, lo_vals, hi_vals)

    if color_mapping == 'red_blue':
        final_image = np.zeros((rows, cols, 4), dtype=np.float32)

        # Assign embedding axes to R and B channels for masked pixels
        final_image[mask, 0] = embedding[:, 0]   # Red channel
        final_image[mask, 2] = embedding[:, 1]   # Blue channel

        # Normalize each channel to [0, 1] independently
        for ch in (0, 2):
            ch_min = final_image[:, :, ch].min()
            ch_max = final_image[:, :, ch].max()
            if ch_max > ch_min:
                final_image[:, :, ch] = (final_image[:, :, ch] - ch_min) / (ch_max - ch_min)
            else:
                final_image[:, :, ch] = 0.0

        # Clamp to [0, 1] to guard against floating-point edge cases
        np.clip(final_image[:, :, 0], 0.0, 1.0, out=final_image[:, :, 0])
        np.clip(final_image[:, :, 2], 0.0, 1.0, out=final_image[:, :, 2])

        # Alpha = 1 for foreground pixels
        final_image[mask, 3] = 1.0

        return final_image  # dtype float32

    elif color_mapping == 'lab':
        # Build a CIE L*a*b* image (float64)
        lab_image = np.zeros((rows, cols, 3), dtype=np.float64)
        lab_image[..., 0] = 50.0          # constant lightness

        lab_image[mask, 1] = embedding[:, 0]   # a* channel
        lab_image[mask, 2] = embedding[:, 1]   # b* channel

        # Normalize a* and b* to the standard L*a*b* range [-128, 127]
        for i in (1, 2):
            ch_min = lab_image[:, :, i].min()
            ch_max = lab_image[:, :, i].max()
            if ch_max > ch_min:
                lab_image[:, :, i] = (lab_image[:, :, i] - ch_min) / (ch_max - ch_min)
            else:
                lab_image[:, :, i] = 0.0
            lab_image[:, :, i] = lab_image[:, :, i] * 255.0 - 128.0

        # Convert L*a*b* → sRGB (output is float64 in [0, 1])
        rgb_image = skcolor.lab2rgb(lab_image)

        # Clamp to [0, 1] (lab2rgb may produce tiny out-of-gamut values)
        np.clip(rgb_image, 0.0, 1.0, out=rgb_image)

        # Build alpha channel
        alpha = np.zeros((rows, cols), dtype=np.float64)
        alpha[mask] = 1.0

        final_image = np.dstack([rgb_image, alpha])   # dtype float64

        return final_image  # dtype float64

    else:
        raise ValueError(
            f"Unknown color_mapping '{color_mapping}'. "
            "Must be 'red_blue' or 'lab'."
        )


# ---------------------------------------------------------------------------
# Helper: _hv_color_grid01
# ---------------------------------------------------------------------------

def _hv_color_grid01(
    embedding: np.ndarray,
    saturation: float = 0.9,
    value: float = 0.9,
) -> np.ndarray:
    """Map a 2-D UMAP embedding to HSV colors via an inscribed-circle grid.

    Each point is placed on an HSV color wheel where:

    * **Hue** is determined by the angular position (``arctan2``) of the point
      relative to the embedding centroid.
    * **Saturation** is proportional to the radial distance from the centroid,
      scaled so that the farthest point sits on the inscribed circle of the
      unit square (radius = 0.5), and clamped to ``[0, saturation]``.
    * **Value** is constant.

    Parameters
    ----------
    embedding : np.ndarray
        Array of shape ``(n, 2)`` with UMAP coordinates.
    saturation : float
        Maximum saturation value in ``[0, 1]``.  Default 0.9.
    value : float
        HSV value (brightness) in ``[0, 1]``.  Default 0.9.

    Returns
    -------
    np.ndarray
        Float32 array of shape ``(n, 3)`` with HSV values in ``[0, 1]``.
    """
    # Centre and scale the embedding so it fits in the unit square
    pts = embedding.astype(np.float64)
    pts -= pts.mean(axis=0)

    # Scale so the maximum absolute coordinate is 0.5 (inscribed-circle radius)
    scale = np.abs(pts).max()
    if scale > 0:
        pts /= (2.0 * scale)   # now in [-0.5, 0.5]

    # Hue from angle, normalised to [0, 1]
    hue = (np.arctan2(pts[:, 1], pts[:, 0]) / (2.0 * np.pi)) % 1.0

    # Saturation from radius, clamped to [0, saturation]
    radius = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
    # inscribed-circle radius is 0.5; normalise
    sat = np.clip(radius / 0.5, 0.0, 1.0) * saturation

    val = np.full(len(pts), value, dtype=np.float64)

    hsv = np.stack([hue, sat, val], axis=1).astype(np.float32)
    return hsv


# ---------------------------------------------------------------------------
# Helper: simulate_rg_colorblind
# ---------------------------------------------------------------------------

def simulate_rg_colorblind(
    image: np.ndarray,
    severity: float = 1.0,
    mode: str = 'deuteranopia',
) -> np.ndarray:
    """Simulate red-green colour blindness on an RGBA or RGB image.

    Uses a linear RGB transform that approximates the Brettel (1997) model for
    deuteranopia (green-cone absence) or protanopia (red-cone absence).

    Parameters
    ----------
    image : np.ndarray
        Input image of shape ``(H, W, 3)`` or ``(H, W, 4)``, dtype float in
        ``[0, 1]`` or uint8 in ``[0, 255]``.
    severity : float
        Interpolation factor between original (0.0) and fully simulated (1.0)
        colour blindness.  Default 1.0.
    mode : {'deuteranopia', 'protanopia'}
        Type of red-green colour blindness to simulate.

    Returns
    -------
    np.ndarray
        Simulated image with the same shape and dtype as *image*, values in
        the same range as the input.
    """
    if mode not in ('deuteranopia', 'protanopia'):
        raise ValueError(
            f"Unknown mode '{mode}'. Must be 'deuteranopia' or 'protanopia'."
        )

    # Work in float [0, 1]
    uint8_input = image.dtype == np.uint8
    img = image.astype(np.float64)
    if uint8_input:
        img /= 255.0

    has_alpha = img.shape[2] == 4
    rgb = img[..., :3]

    # Colour-blindness simulation matrices (linear sRGB)
    # Deuteranopia (Machado et al. 2009, severity=1.0)
    deuteranopia = np.array([
        [0.625, 0.375, 0.000],
        [0.700, 0.300, 0.000],
        [0.000, 0.300, 0.700],
    ], dtype=np.float64)

    # Protanopia
    protanopia = np.array([
        [0.567, 0.433, 0.000],
        [0.558, 0.442, 0.000],
        [0.000, 0.242, 0.758],
    ], dtype=np.float64)

    matrix = deuteranopia if mode == 'deuteranopia' else protanopia

    # Apply transform: (H, W, 3) @ (3, 3).T
    simulated_rgb = rgb @ matrix.T

    # Blend with original based on severity
    blended_rgb = (1.0 - severity) * rgb + severity * simulated_rgb
    np.clip(blended_rgb, 0.0, 1.0, out=blended_rgb)

    if has_alpha:
        result = np.concatenate([blended_rgb, img[..., 3:4]], axis=2)
    else:
        result = blended_rgb

    if uint8_input:
        result = (result * 255.0).round().astype(np.uint8)
    else:
        result = result.astype(image.dtype)

    return result
