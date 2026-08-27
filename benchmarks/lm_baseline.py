"""
lm_baseline.py
===============
NumPy / SciPy reference implementations of a "conventional" Levenberg-Marquardt
(LM) peak-fitting pipeline, used to quantitatively benchmark CASCADE
(CWT + physics-constrained Adam) against the class of nonlinear least-squares
methods discussed in the Introduction (scipy.optimize.least_squares,
method='lm', which wraps the MINPACK lmdif routine used by lmfit and similar
Raman-fitting packages cited in the paper).

Two initialization strategies are provided:
  - lm_fit_from_p0    : LM given the SAME CWT-derived initial guess CASCADE
                        uses (isolates the optimizer: Adam vs. LM).
  - lm_fit_naive_init : LM given a conventional single-scale
                        scipy.signal.find_peaks initialization on the raw
                        (noisy) spectrum, with no wavelet decomposition
                        (represents typical unassisted LM practice).

The pseudo-Voigt model matches tidytorch_utils.pseudo_voigt exactly (same
Thompson-Cox-Hastings approximation), so any performance difference between
CASCADE and the LM baselines here is attributable to initialization and
optimizer choice, not to a different peak model.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks

FWHM_FROM_SIGMA = 2.35482


def pseudo_voigt_np(x: np.ndarray, sigma: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """Thompson-Cox-Hastings pseudo-Voigt, matching tidytorch_utils.pseudo_voigt."""
    sigma = np.abs(sigma) + 1e-6
    gamma = np.abs(gamma) + 1e-6

    fwhm_g = FWHM_FROM_SIGMA * sigma
    fwhm_l = 2.0 * gamma

    fwhm = (fwhm_g ** 5 + 2.69269 * fwhm_g ** 4 * fwhm_l +
            2.42843 * fwhm_g ** 3 * fwhm_l ** 2 + 4.47163 * fwhm_g ** 2 * fwhm_l ** 3 +
            0.07842 * fwhm_g * fwhm_l ** 4 + fwhm_l ** 5) ** 0.2
    fwhm = np.maximum(fwhm, 1e-6)

    ratio = fwhm_l / fwhm
    eta = 1.36603 * ratio - 0.47719 * ratio ** 2 + 0.11116 * ratio ** 3
    eta = np.clip(eta, 0.0, 1.0)

    z = np.clip(x / fwhm, -50.0, 50.0)
    gaussian = np.exp(-4.0 * np.log(2.0) * z ** 2)
    lorentzian = 1.0 / (1.0 + 4.0 * z ** 2)
    return eta * lorentzian + (1.0 - eta) * gaussian


def compute_model_np(p_flat: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Vectorised sum-of-pseudo-Voigt model. p_flat: (n_peaks*4,) [amp,ctr,sig,gam,...]."""
    p = p_flat.reshape(-1, 4)
    amps, ctrs, sigs, gams = p[:, 0:1], p[:, 1:2], p[:, 2:3], p[:, 3:4]
    xs = x[None, :] - ctrs
    pv = pseudo_voigt_np(xs, sigs, gams)
    return (amps * pv).sum(axis=0)


def _residual(p_flat, x, y):
    return compute_model_np(p_flat, x) - y


def trim_zero_peaks(p0_flat: np.ndarray, max_peaks: int = 60) -> np.ndarray:
    """Drop zero-amplitude padding rows from a CASCADE-style flat p0 vector.

    Also caps the candidate count at the ``max_peaks`` largest-amplitude
    entries. CASCADE's own CWT stage is deliberately over-inclusive (a
    generous ``max_peaks`` catches weak/overlapping peaks that later get
    pruned post-fit); handing every one of those candidates to MINPACK LM
    as a free parameter is not how LM is used in practice and makes the
    per-spectrum parameter count (and thus the O(n^2)-O(n^3) Jacobian/
    normal-equation cost) balloon well past any realistic dataset. Capping
    keeps the comparison representative of standard LM usage.
    """
    p = p0_flat.reshape(-1, 4)
    p = p[p[:, 0] > 0]
    if p.shape[0] > max_peaks:
        top = np.argsort(p[:, 0])[::-1][:max_peaks]
        p = p[np.sort(top)]
    return p.reshape(-1)


def lm_fit_from_p0(spectrum: np.ndarray, x: np.ndarray, p0_flat: np.ndarray,
                    max_nfev: int = 20000):
    """Classic (unbounded) Levenberg-Marquardt fit, seeded with CASCADE's own
    CWT-derived initial guess. Isolates the optimizer (Adam vs. LM) as the
    only variable relative to CASCADE.
    """
    p0 = trim_zero_peaks(p0_flat)
    if p0.size == 0:
        return np.zeros(0, dtype=np.float32), 0.0, False, 0

    res = least_squares(
        _residual, p0, args=(x, spectrum), method="lm", max_nfev=max_nfev,
    )
    return res.x.astype(np.float32), float(res.cost * 2), bool(res.success), int(res.nfev)


def naive_initial_guess(spectrum: np.ndarray, x: np.ndarray,
                         height_frac: float = 0.05, distance_pts: int = 4,
                         default_fwhm: float = 15.0, default_gamma: float = 5.0,
                         max_peaks: int = 60) -> np.ndarray:
    """Conventional single-scale peak-picking initialization (no CWT):
    scipy.signal.find_peaks on the raw, noisy spectrum. Represents the
    standard practice this paper's Introduction critiques (Pezzotti2022,
    Lemoine2019, etc.) — no multiscale denoising, no wavelet-informed width
    estimate. Candidates are capped at the ``max_peaks`` tallest detections,
    matching how a practitioner would bound problem size for MINPACK LM.
    """
    height = height_frac * float(np.max(spectrum))
    idx, _ = find_peaks(spectrum, height=height, distance=distance_pts)
    if idx.size == 0:
        return np.zeros(0, dtype=np.float32)

    if idx.size > max_peaks:
        top = np.argsort(spectrum[idx])[::-1][:max_peaks]
        idx = np.sort(idx[top])

    sigma0 = default_fwhm / FWHM_FROM_SIGMA
    amps = spectrum[idx]
    ctrs = x[idx]
    sigs = np.full_like(amps, sigma0)
    gams = np.full_like(amps, default_gamma)
    p0 = np.stack([amps, ctrs, sigs, gams], axis=1).astype(np.float32)
    return p0.reshape(-1)


def lm_fit_naive_init(spectrum: np.ndarray, x: np.ndarray, max_nfev: int = 20000, **kwargs):
    """Classic LM fit with conventional (non-CWT) peak-picking initialization."""
    p0 = naive_initial_guess(spectrum, x, **kwargs)
    if p0.size == 0:
        return np.zeros(0, dtype=np.float32), 0.0, False, 0

    res = least_squares(
        _residual, p0, args=(x, spectrum), method="lm", max_nfev=max_nfev,
    )
    return res.x.astype(np.float32), float(res.cost * 2), bool(res.success), int(res.nfev)
