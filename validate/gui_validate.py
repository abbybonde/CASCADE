#!/usr/bin/env python3
"""
gui_validate.py — Interactive BCARS fit validation GUI

Usage:
    python gui_validate.py [path_to_npz]

Controls:
    Slider   — scrub through wavenumber axis, images update live
    Click    — click any pixel on the Original or Fitted image to plot its spectrum
    Load .npz — opens a file-picker dialog (falls back to argv if tkinter unavailable)
"""
import os, sys
import numpy as np

# Set an interactive backend before importing pyplot.
# TkAgg works on most HPC + VS Code environments; Qt5Agg is the fallback.
import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    try:
        matplotlib.use("Qt5Agg")
    except Exception:
        pass

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button

try:
    from tkinter import Tk
    from tkinter.filedialog import askopenfilename
    _HAS_TK = True
except ImportError:
    _HAS_TK = False

plt.style.use("dark_background")

# ── Application state ─────────────────────────────────────────────────────────
class _S:
    orig   = None   # (ny, nx, n_wn) float32 — imag part of raw BCARS
    model  = None   # (ny, nx, n_wn) float32 — fitted model
    wn     = None   # (n_wn,)        float32 — wavenumber axis cm⁻¹
    wn_idx = 0
    px     = None   # last clicked pixel (xi, yi)

# ── Persistent artist handles (filled once data loads) ───────────────────────
_im      = [None, None]          # [im_orig, im_fit]
_cross   = [[None, None],        # crosshair lines on ax_orig [h, v]
            [None, None]]        # crosshair lines on ax_fit  [h, v]
_ln_spec = [None, None]          # spectrum axes: [raw, fit]
_ln_diff = [None]                # residual axis: [diff]

# ── Layout ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 9))
fig.suptitle("BCARS Fit Validator", fontsize=13, fontweight="bold")

gs = gridspec.GridSpec(
    3, 2,
    height_ratios=[5, 4, 0.7],
    hspace=0.42, wspace=0.28,
    left=0.06, right=0.97, top=0.93, bottom=0.10,
)
ax_orig = fig.add_subplot(gs[0, 0])
ax_fit  = fig.add_subplot(gs[0, 1])
ax_spec = fig.add_subplot(gs[1, 0])
ax_diff = fig.add_subplot(gs[1, 1], sharex=ax_spec)

# Slider and button sit in explicit axes below the gridspec
ax_slide = fig.add_axes([0.12, 0.035, 0.63, 0.022])
ax_btn   = fig.add_axes([0.80, 0.022, 0.16, 0.048])

# Placeholder text while no data is loaded
for ax, label in [(ax_orig, "Original"), (ax_fit, "Fitted model")]:
    ax.text(0.5, 0.5, f"{label}\n\nLoad a .npz file to begin",
            ha="center", va="center", transform=ax.transAxes,
            color="#888", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])

for ax, label in [(ax_spec, "Spectrum  (click a pixel on the images above)"),
                  (ax_diff, "Residual  (click a pixel on the images above)")]:
    ax.text(0.5, 0.5, label, ha="center", va="center",
            transform=ax.transAxes, color="#888", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

# Slider (range updated after load)
slider_wn = Slider(ax_slide, "wn (cm⁻¹)", 0, 1, valinit=0.5, color="#4477cc")
slider_wn.label.set_color("white")
slider_wn.valtext.set_color("white")

# Load button
btn_load = Button(ax_btn, "Load .npz", color="#444", hovercolor="#666")
btn_load.label.set_color("white")
btn_load.label.set_fontsize(10)


# ── Data loading ──────────────────────────────────────────────────────────────
def _load(path: str):
    print(f"Loading {path} …")
    d = np.load(path)
    orig = d["unprocessed_spectrum"]
    if np.iscomplexobj(orig):
        print("  Complex data detected — extracting imaginary part")
        orig = np.imag(orig)
    _S.orig   = orig.astype(np.float32)
    _S.model  = d["processed_spectrum"].astype(np.float32)
    _S.wn     = d["wn_axis"].astype(np.float32)
    _S.wn_idx = len(_S.wn) // 2
    _S.px     = None
    ny, nx, n_wn = _S.orig.shape
    print(f"  Shape: {ny}×{nx} pixels, {n_wn} spectral pts, "
          f"wn=[{_S.wn[0]:.1f}–{_S.wn[-1]:.1f}] cm⁻¹")
    _init_artists()


def _init_artists():
    wn    = _S.wn
    n_wn  = len(wn)
    sl    = _S.wn_idx
    ny, nx = _S.orig.shape[:2]

    # ── Update slider range ───────────────────────────────────────────────────
    # Set valmin/valmax first so set_val clamps correctly.
    # set_val fires on_changed → _update_images, which guards on _im[0] is None.
    slider_wn.valmin = float(wn[0])
    slider_wn.valmax = float(wn[-1])
    ax_slide.set_xlim(wn[0], wn[-1])
    slider_wn.set_val(float(wn[sl]))

    # ── Image axes ────────────────────────────────────────────────────────────
    sl_orig  = _S.orig[:, :, sl]
    sl_model = _S.model[:, :, sl]
    vmin = min(sl_orig.min(), sl_model.min())
    vmax = max(sl_orig.max(), sl_model.max())

    ax_orig.cla(); ax_fit.cla()
    _im[0] = ax_orig.imshow(sl_orig,  cmap="inferno", vmin=vmin, vmax=vmax,
                             origin="upper", aspect="auto")
    _im[1] = ax_fit.imshow(sl_model,  cmap="inferno", vmin=vmin, vmax=vmax,
                            origin="upper", aspect="auto")

    for ax, title in [(ax_orig, "Original (imag)"), (ax_fit, "Fitted model")]:
        ax.set_title(f"{title}  ·  {wn[sl]:.1f} cm⁻¹", fontsize=10)
        ax.set_xlabel("x  (px)"); ax.set_ylabel("y  (px)")

    # Crosshairs — hidden until first click
    _cross[0][0], = ax_orig.plot([], [], color="cyan", lw=0.8, alpha=0.7)
    _cross[0][1], = ax_orig.plot([], [], color="cyan", lw=0.8, alpha=0.7)
    _cross[1][0], = ax_fit.plot([], [], color="cyan", lw=0.8, alpha=0.7)
    _cross[1][1], = ax_fit.plot([], [], color="cyan", lw=0.8, alpha=0.7)

    # ── Spectrum axes ─────────────────────────────────────────────────────────
    ax_spec.cla(); ax_diff.cla()
    zeros = np.zeros(n_wn)

    _ln_spec[0], = ax_spec.plot(wn, zeros, color="white",   lw=1.1, label="Original")
    _ln_spec[1], = ax_spec.plot(wn, zeros, color="crimson", lw=1.1, ls="--", label="Fitted")
    ax_spec.axhline(0, color="#555", lw=0.6)
    ax_spec.set_ylabel("Intensity"); ax_spec.set_xlabel("Wavenumber (cm⁻¹)")
    ax_spec.set_title("Spectrum  (click a pixel above)", fontsize=9)
    ax_spec.legend(fontsize=8, framealpha=0.3)

    _ln_diff[0], = ax_diff.plot(wn, zeros, color="steelblue", lw=1.0)
    ax_diff.axhline(0, color="#555", lw=0.6)
    ax_diff.set_ylabel("Residual  (orig − fit)"); ax_diff.set_xlabel("Wavenumber (cm⁻¹)")
    ax_diff.set_title("Residual  (click a pixel above)", fontsize=9)

    fig.canvas.draw_idle()


# ── Update helpers ────────────────────────────────────────────────────────────
def _update_images(wn_val: float):
    if _S.orig is None or _im[0] is None:
        return
    idx = int(np.argmin(np.abs(_S.wn - wn_val)))
    _S.wn_idx = idx
    sl_o = _S.orig[:, :, idx]
    sl_m = _S.model[:, :, idx]
    vmin = min(sl_o.min(), sl_m.min())
    vmax = max(sl_o.max(), sl_m.max())
    _im[0].set_data(sl_o); _im[0].set_clim(vmin, vmax)
    _im[1].set_data(sl_m); _im[1].set_clim(vmin, vmax)
    ax_orig.set_title(f"Original (imag)  ·  {_S.wn[idx]:.1f} cm⁻¹", fontsize=10)
    ax_fit.set_title( f"Fitted model  ·  {_S.wn[idx]:.1f} cm⁻¹",    fontsize=10)
    fig.canvas.draw_idle()


def _update_spectrum(xi: int, yi: int):
    if _S.orig is None or _ln_spec[0] is None:
        return
    ny, nx = _S.orig.shape[:2]
    xi = int(np.clip(xi, 0, nx - 1))
    yi = int(np.clip(yi, 0, ny - 1))
    _S.px = (xi, yi)

    sp_o = _S.orig[yi, xi]
    sp_m = _S.model[yi, xi]
    sp_d = sp_o - sp_m

    _ln_spec[0].set_ydata(sp_o)
    _ln_spec[1].set_ydata(sp_m)
    _ln_diff[0].set_ydata(sp_d)

    for ax in [ax_spec, ax_diff]:
        ax.relim(); ax.autoscale_view()
    ax_spec.set_title(f"Spectrum at pixel  ({xi}, {yi})", fontsize=9)
    ax_diff.set_title(f"Residual at pixel  ({xi}, {yi})  "
                      f"  RMSE={np.sqrt((sp_d**2).mean()):.3g}", fontsize=9)

    # Draw crosshairs on both image axes
    for i, ax in enumerate([ax_orig, ax_fit]):
        _cross[i][0].set_data([0, nx - 1], [yi, yi])   # horizontal
        _cross[i][1].set_data([xi, xi], [0, ny - 1])   # vertical

    fig.canvas.draw_idle()


# ── Event callbacks ───────────────────────────────────────────────────────────
def _on_load(_event):
    if _HAS_TK:
        root = Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = askopenfilename(
            title="Open BCARS batched-fit .npz",
            filetypes=[("NumPy archive", "*.npz"), ("All files", "*.*")],
        )
        root.destroy()
        if path:
            _load(path)
    else:
        print("tkinter unavailable — pass the .npz path as a command-line argument")


def _on_slider(val: float):
    _update_images(val)


def _on_click(event):
    if event.inaxes not in (ax_orig, ax_fit):
        return
    if event.xdata is None or event.ydata is None:
        return
    _update_spectrum(int(round(event.xdata)), int(round(event.ydata)))


btn_load.on_clicked(_on_load)
slider_wn.on_changed(_on_slider)
fig.canvas.mpl_connect("button_press_event", _on_click)

# ── Auto-load from command-line argument ──────────────────────────────────────
if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
    _load(sys.argv[1])

plt.show()
