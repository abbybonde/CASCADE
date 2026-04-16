## ── Config ───────────────────────────────────────────────────────────────────
NPZ_PATH  = "bcars_batched_fit_2026-04-16_13-40-54_YOUR_FILE_NAME_HERE_tolerance=1e-05.npz"  # path to .npz output from Full_ROI_BCARSFitting.py
N_SAMPLE  = 9                                   # number of random pixels to plot
SEED      = 42

# Future: uncomment to override random sampling
# SELECT_MODE   = "best_worst"          # "random" | "manual" | "best_worst"
# PIXEL_COORDS  = [(x0, y0), (x1, y1)] # used when SELECT_MODE == "manual"

## ── Imports ──────────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

## ── Load data ────────────────────────────────────────────────────────────────
_d    = np.load(NPZ_PATH)
orig  = _d["unprocessed_spectrum"]
model = _d["processed_spectrum"].astype(np.float32)     # (y, x, n_wn)  real — fitted to imag(denoised(orig))
wn    = _d["wn_axis"].astype(np.float32)                # (n_wn,)  cm⁻¹, cropped to 400–1800 cm⁻¹

# The fitter extracts np.imag() from complex BCARS data before fitting (Full_ROI_BCARSFitting.py:314).
# unprocessed_spectrum is saved as raw (possibly complex), so we match that transform here.
if np.iscomplexobj(orig):
    print(f"  unprocessed_spectrum is complex — extracting imaginary part to match fitter input")
    orig = np.imag(orig)
orig = orig.astype(np.float32)

ny, nx, n_wn = orig.shape
print(f"Loaded: orig {orig.shape}, model {model.shape}, wn [{wn[0]:.1f}–{wn[-1]:.1f}] cm⁻¹ ({n_wn} pts)")

## ── Quick sanity check: single wavenumber slice ──────────────────────────────
WN_SANITY = wn[n_wn // 2]                        # pick middle wavenumber; change as needed
wn_idx    = int(np.argmin(np.abs(wn - WN_SANITY)))
print(f"Sanity plot at wn={wn[wn_idx]:.1f} cm⁻¹ (index {wn_idx})")

sl_orig  = orig[:, :, wn_idx]
sl_model = model[:, :, wn_idx]
vmin = min(sl_orig.min(), sl_model.min())
vmax = max(sl_orig.max(), sl_model.max())

fig0, (ax0a, ax0b, ax0c) = plt.subplots(1, 3, figsize=(13, 4))
im0a = ax0a.imshow(sl_orig,          cmap="inferno", vmin=vmin, vmax=vmax, origin="upper")
im0b = ax0b.imshow(sl_model,         cmap="inferno", vmin=vmin, vmax=vmax, origin="upper")
im0c = ax0c.imshow(sl_orig - sl_model, cmap="bwr",   origin="upper")
for ax, title in zip([ax0a, ax0b, ax0c], ["Original", "Fitted model", "Residual (orig − model)"]):
    ax.set_title(title)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
fig0.colorbar(im0a, ax=ax0a, fraction=0.046, pad=0.04)
fig0.colorbar(im0b, ax=ax0b, fraction=0.046, pad=0.04)
fig0.colorbar(im0c, ax=ax0c, fraction=0.046, pad=0.04)
fig0.suptitle(f"Sanity check — spatial slice at {wn[wn_idx]:.1f} cm⁻¹", fontsize=13)
fig0.tight_layout()
fig0.savefig("validate_sanity_slice.png", dpi=150)
print("Saved validate_sanity_slice.png")
plt.pause(0.1)

## ── Per-pixel metrics (vectorised over spectral axis) ────────────────────────
residual  = orig - model                                                  # (y, x, n_wn)
ss_res    = (residual ** 2).sum(axis=2)                                   # (y, x)
ss_tot    = ((orig - orig.mean(axis=2, keepdims=True)) ** 2).sum(axis=2)  # (y, x)
r2_map    = 1.0 - ss_res / (ss_tot + 1e-12)                              # (y, x)
rmse_map  = np.sqrt((residual ** 2).mean(axis=2))                         # (y, x)
sig_range = orig.max(axis=2) - orig.min(axis=2)                           # (y, x)
nrmse_map = rmse_map / (sig_range + 1e-12)                                # (y, x)

for name, m in [("R²", r2_map), ("RMSE", rmse_map), ("NRMSE", nrmse_map)]:
    print(f"  {name:6s}  median={np.median(m):.4f}  p5={np.percentile(m,5):.4f}  p95={np.percentile(m,95):.4f}")

## ── Figure 1: spatial metric maps ───────────────────────────────────────────
fig1, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, data_map, title, cmap in zip(
    axes,
    [r2_map, rmse_map, nrmse_map],
    ["R²", "RMSE", "NRMSE"],
    ["viridis", "hot_r", "hot_r"],
):
    im = ax.imshow(data_map, cmap=cmap, origin="upper")
    ax.set_title(title)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    fig1.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig1.suptitle("Per-pixel fit quality", fontsize=13)
fig1.tight_layout()
fig1.savefig("validate_metric_maps.png", dpi=150)
print("Saved validate_metric_maps.png")

## ── Pixel selection ──────────────────────────────────────────────────────────
rng      = np.random.default_rng(SEED)
flat_idx = rng.choice(ny * nx, size=min(N_SAMPLE, ny * nx), replace=False)
yx_pairs = [(i // nx, i % nx) for i in flat_idx]

# Future hook — best/worst:
# flat_r2  = r2_map.ravel()
# order    = np.argsort(flat_r2)
# yx_pairs = [(i // nx, i % nx) for i in np.concatenate([order[:N_SAMPLE//2], order[-(N_SAMPLE//2):]])]

# Future hook — manual:
# yx_pairs = [(y0, x0) for x0, y0 in PIXEL_COORDS]

## ── Figure 2: side-by-side spectrum plots ────────────────────────────────────
ncols  = 3
nrows  = int(np.ceil(len(yx_pairs) / ncols))
fig2   = plt.figure(figsize=(5 * ncols, 4 * nrows))
gs_outer = gridspec.GridSpec(nrows, ncols, figure=fig2, hspace=0.45, wspace=0.35)

for k, (yi, xi) in enumerate(yx_pairs):
    gs_inner = gs_outer[k // ncols, k % ncols].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.08)
    ax_spec  = fig2.add_subplot(gs_inner[0])
    ax_res   = fig2.add_subplot(gs_inner[1], sharex=ax_spec)

    y_orig  = orig[yi, xi]
    y_model = model[yi, xi]
    y_res   = y_orig - y_model
    res_std = y_res.std()

    ax_spec.plot(wn, y_orig,  color="black",  lw=1.2, label="original")
    ax_spec.plot(wn, y_model, color="crimson", lw=1.2, ls="--", label="fitted")
    ax_spec.set_ylabel("Intensity")
    ax_spec.legend(fontsize=7, loc="upper right")
    ax_spec.set_title(
        f"({xi}, {yi})   R²={r2_map[yi,xi]:.3f}  RMSE={rmse_map[yi,xi]:.3g}",
        fontsize=8,
    )
    plt.setp(ax_spec.get_xticklabels(), visible=False)

    ax_res.plot(wn, y_res, color="steelblue", lw=0.9)
    ax_res.axhline(0,          color="gray",  lw=0.7, ls="--")
    ax_res.axhline( res_std,   color="gray",  lw=0.5, ls=":")
    ax_res.axhline(-res_std,   color="gray",  lw=0.5, ls=":")
    ax_res.set_ylabel("Residual", fontsize=7)
    ax_res.set_xlabel("Wavenumber (cm⁻¹)", fontsize=8)

fig2.suptitle(f"Random sample of {len(yx_pairs)} pixels — original vs fitted", fontsize=12)
fig2.savefig("validate_spectra_sample.png", dpi=150)
print("Saved validate_spectra_sample.png")

## ── Optional: SSIM / PSNR per wavenumber slice ───────────────────────────────
try:
    from skimage.metrics import structural_similarity, peak_signal_noise_ratio

    data_min = orig.min()
    data_max = orig.max()
    data_range = float(data_max - data_min)

    ssim_vs_wn = np.zeros(n_wn)
    psnr_vs_wn = np.zeros(n_wn)
    for i in range(n_wn):
        sl_orig  = orig[:, :, i]
        sl_model = model[:, :, i]
        ssim_vs_wn[i] = structural_similarity(sl_orig, sl_model, data_range=data_range)
        psnr_vs_wn[i] = peak_signal_noise_ratio(sl_orig, sl_model, data_range=data_range)

    fig3, (ax_s, ax_p) = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    ax_s.plot(wn, ssim_vs_wn, lw=1.1)
    ax_s.set_ylabel("SSIM")
    ax_s.axhline(1.0, color="gray", lw=0.7, ls="--")
    ax_p.plot(wn, psnr_vs_wn, lw=1.1, color="darkorange")
    ax_p.set_ylabel("PSNR (dB)")
    ax_p.set_xlabel("Wavenumber (cm⁻¹)")
    fig3.suptitle("SSIM / PSNR per wavenumber slice (spatial 2-D maps)", fontsize=12)
    fig3.tight_layout()
    fig3.savefig("validate_ssim_psnr.png", dpi=150)
    print("Saved validate_ssim_psnr.png")
    print(f"  SSIM  median={np.median(ssim_vs_wn):.4f}  p5={np.percentile(ssim_vs_wn,5):.4f}")
    print(f"  PSNR  median={np.median(psnr_vs_wn):.2f} dB")

except ImportError:
    print("scikit-image not found — skipping SSIM/PSNR plots")

plt.show()
