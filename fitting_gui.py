"""
fitting_gui.py — CASCADE Interactive Fitting GUI
=================================================
Two-panel layout that mirrors the BCARSFitting notebook workflow:

  Left  — Unfit data:   spatial image (top) + raw pixel spectrum (bottom)
  Right — Fit results:  fit map (top) + decomposed spectrum & residual (bottom)

Clicking either spatial image selects the same pixel on both sides.
The parameter bar at the bottom matches the knobs in BCARSFitting.ipynb;
"Fit pixel" runs the fitting for the selected pixel, "Fit batch" processes
a specified spatial ROI with a progress bar.

Usage
-----
    python fitting_gui.py
"""

import os
import sys
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.cm as mpl_cm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec


# ── File / array helpers ──────────────────────────────────────────────────────

def _orient(data: np.ndarray, wn_dim: int) -> np.ndarray:
    """Move spectral axis to -1 so result is (rows, cols, n_wn)."""
    if data.ndim == 2:
        return data[np.newaxis, :, :]
    return np.moveaxis(data, wn_dim, -1)


def _to_real(arr: np.ndarray, take_imag: bool) -> np.ndarray:
    if np.iscomplexobj(arr):
        return (np.imag(arr) if take_imag else np.real(arr)).astype(np.float32)
    return arr.astype(np.float32)


def _try_load_wn(path: str) -> "np.ndarray | None":
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in (".h5", ".hdf5"):
            import h5py
            with h5py.File(path, "r") as f:
                for k in ("preprocessed_images/x_axis", "x_axis",
                          "wn_axis", "wavenumbers"):
                    if k in f and len(f[k].shape) == 1:
                        return np.array(f[k], dtype=np.float32)
        elif ext == ".npz":
            d = np.load(path, allow_pickle=False)
            for k in ("wn_axis", "x_axis", "wavenumbers"):
                if k in d and d[k].ndim == 1:
                    return d[k].astype(np.float32)
    except Exception:
        pass
    return None


def _load_h5_array(path: str, dset: str) -> np.ndarray:
    import h5py
    with h5py.File(path, "r") as f:
        if dset and dset in f:
            return np.array(f[dset])
        candidates: list[tuple[int, str]] = []
        f.visititems(lambda n, o: candidates.append((int(np.prod(o.shape)), n))
                     if hasattr(o, "shape") and len(o.shape) >= 2 else None)
        if not candidates:
            raise KeyError(f"No 2-D+ dataset found in {path}")
        chosen = sorted(candidates, reverse=True)[0][1]
        messagebox.showinfo("Auto-detected",
                            f"Loaded: {chosen}\nshape: {tuple(f[chosen].shape)}")
        return np.array(f[chosen])


def _load_npz_array(path: str, key: str) -> np.ndarray:
    d = np.load(path, allow_pickle=False)
    if key and key in d:
        return d[key]
    for k in ("unprocessed_spectrum", "data", "spectrum", "raw"):
        if k in d and np.asarray(d[k]).ndim >= 2:
            return d[k]
    keys = list(d.keys())
    if keys:
        return d[keys[0]]
    raise KeyError(f"No arrays found in {path}")


def _voigt_np(x: np.ndarray, amp: float, ctr: float,
              sig: float, gam: float) -> np.ndarray:
    """Thompson-Cox-Hastings pseudo-Voigt (numpy, matches tidytorch formula)."""
    sig = max(float(sig), 1e-9); gam = max(float(gam), 1e-9)
    fg  = 2.35482 * sig; fl = 2.0 * gam
    fv  = (fg**5 + 2.69269*fg**4*fl + 2.42843*fg**3*fl**2
           + 4.47163*fg**2*fl**3 + 0.07842*fg*fl**4 + fl**5) ** 0.2
    fv  = max(fv, 1e-9)
    z   = (np.asarray(x, dtype=np.float64) - ctr) / fv
    r   = fl / fv
    eta = float(np.clip(1.36603*r - 0.47719*r**2 + 0.11116*r**3, 0.0, 1.0))
    return float(amp) * (eta / (1.0 + 4.0*z**2)
                         + (1.0 - eta) * np.exp(-4.0 * np.log(2.0) * z**2))


# ── Main GUI class ────────────────────────────────────────────────────────────

class CascadeFitGUI(tk.Tk):
    _PAD = 4

    # Fitting parameter defaults — mirrors BCARSFitting.ipynb
    _PARAM_DEFS = [
        ("adj",                       "Offset adj",        "0.0",    8),
        ("denoise_sigma",             "Denoise σ",         "3.0",    6),
        ("response_threshold",        "Resp. threshold",   "0.0001", 8),
        ("amp_threshold",             "Amp threshold",     "1e-6",   8),
        ("min_scale_votes",           "Min votes",         "3",      5),
        ("min_spacing_in",            "Min spacing in",    "5.0",    6),
        ("min_spacing_post",          "Min spacing post",  "7.0",    6),
        ("max_iter",                  "Max iter",          "4000",   6),
        ("tol",                       "Tolerance",         "1e-5",   8),
        ("scale_preference_fraction", "Scale pref frac",   "0.9",    6),
    ]

    def __init__(self):
        super().__init__()
        self.title("CASCADE — Fitting GUI")
        self.geometry("1460x930")
        self.minsize(1000, 700)

        # ── Mutable state ─────────────────────────────────────────────────
        self._raw_data: "np.ndarray | None" = None
        self._data:     "np.ndarray | None" = None      # (rows, cols, n_wn)
        self._wn:       "np.ndarray | None" = None      # (n_wn,)
        self._pixel_params: "dict[tuple, np.ndarray]" = {}
        self._fit_map:     "np.ndarray | None" = None   # (rows, cols), NaN=unfitted
        self._npeaks_map:  "np.ndarray | None" = None   # (rows, cols)
        self._sel_row = self._sel_col = 0
        self._fit_ctx_ready = False
        self._stop_flag     = False
        self._busy          = False
        self._fit_queue: "queue.Queue[dict]" = queue.Queue()
        self._integrate = False

        self._build_file_bar()
        self._build_main_panels()
        self._build_param_bar()
        self._build_status_bar()
        self._poll_queue()

    # ═════════════════════════════ UI builders ════════════════════════════════

    def _build_file_bar(self):
        bar = ttk.LabelFrame(self, text="Data", padding=self._PAD)
        bar.pack(side=tk.TOP, fill=tk.X, padx=self._PAD, pady=(self._PAD, 0))

        r0 = ttk.Frame(bar); r0.pack(fill=tk.X)
        ttk.Label(r0, text="File:", width=5).pack(side=tk.LEFT)
        self._data_path = tk.StringVar()
        ttk.Entry(r0, textvariable=self._data_path, width=55).pack(side=tk.LEFT, padx=2)
        ttk.Button(r0, text="Browse…", command=self._browse).pack(side=tk.LEFT, padx=2)
        ttk.Label(r0, text="  Dataset:").pack(side=tk.LEFT)
        self._data_dset = tk.StringVar()
        ttk.Entry(r0, textvariable=self._data_dset, width=40).pack(side=tk.LEFT, padx=2)
        ttk.Button(r0, text="Load", command=self._load_data).pack(side=tk.LEFT, padx=6)

        r1 = ttk.Frame(bar); r1.pack(fill=tk.X, pady=(2, 0))
        ttk.Label(r1, text="WN axis:").pack(side=tk.LEFT)
        self._wn_src = tk.StringVar(value="auto")
        ttk.Radiobutton(r1, text="From file", variable=self._wn_src,
                        value="auto").pack(side=tk.LEFT, padx=(4, 0))
        ttk.Radiobutton(r1, text="Manual", variable=self._wn_src,
                        value="manual").pack(side=tk.LEFT, padx=(2, 0))
        for lbl, attr, w in [(" start:", "_wn_start", 8),
                              (" end:",   "_wn_end",   8),
                              (" n:",     "_wn_n",     6)]:
            ttk.Label(r1, text=lbl).pack(side=tk.LEFT)
            v = tk.StringVar()
            setattr(self, attr, v)
            ttk.Entry(r1, textvariable=v, width=w).pack(side=tk.LEFT)
        ttk.Button(r1, text=" Apply WN ", command=self._apply_wn).pack(side=tk.LEFT, padx=6)
        ttk.Separator(r1, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Label(r1, text="WN dim:").pack(side=tk.LEFT)
        self._wn_dim = tk.IntVar(value=2)
        for val, lbl in [(0, "0"), (1, "1"), (2, "2 ✓")]:
            ttk.Radiobutton(r1, text=lbl, variable=self._wn_dim, value=val,
                            command=self._on_wn_dim_change).pack(side=tk.LEFT, padx=2)
        ttk.Separator(r1, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        self._take_imag = tk.BooleanVar(value=True)
        ttk.Checkbutton(r1, text="Imag (BCARS)", variable=self._take_imag,
                        command=self._on_imag_toggle).pack(side=tk.LEFT)

    def _build_main_panels(self):
        pane = tk.PanedWindow(self, orient=tk.HORIZONTAL,
                              sashrelief=tk.RAISED, sashwidth=5)
        pane.pack(fill=tk.BOTH, expand=True, padx=self._PAD, pady=self._PAD)

        left = ttk.LabelFrame(pane, text="Unfit data")
        pane.add(left, minsize=420)
        self._build_left_panel(left)

        right = ttk.LabelFrame(pane, text="Fit results")
        pane.add(right, minsize=420)
        self._build_right_panel(right)

    def _build_left_panel(self, parent):
        fig = Figure(figsize=(5, 5.8), dpi=92)
        gs  = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[3, 2], hspace=0.18)
        self._limg_ax = fig.add_subplot(gs[0])
        self._lsp_ax  = fig.add_subplot(gs[1])
        self._limg_ax.set_title("No data — click to select pixel", fontsize=8)
        self._lsp_ax.set_xlabel("Wavenumber", fontsize=8)
        self._lsp_ax.set_ylabel("Intensity", fontsize=8)
        for ax in (self._limg_ax, self._lsp_ax):
            ax.tick_params(labelsize=7)
        fig.tight_layout(pad=0.9)

        self._left_canvas = FigureCanvasTkAgg(fig, master=parent)
        self._left_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._left_canvas.mpl_connect("button_press_event", self._on_left_click)
        self._left_fig = fig

        ctrl = ttk.Frame(parent); ctrl.pack(fill=tk.X, pady=(2, 0))
        ttk.Label(ctrl, text="Channel:").pack(side=tk.LEFT, padx=(4, 0))
        self._wn_slice = tk.IntVar(value=0)
        self._wn_slider = tk.Scale(ctrl, from_=0, to=1, orient=tk.HORIZONTAL,
                                   variable=self._wn_slice, showvalue=True, resolution=1,
                                   command=lambda _: self._refresh_left_image(), length=160)
        self._wn_slider.pack(side=tk.LEFT, padx=4)
        self._int_btn = ttk.Button(ctrl, text="∫ Integrate",
                                   command=self._toggle_integrate)
        self._int_btn.pack(side=tk.LEFT)
        self._pixel_lbl = ttk.Label(ctrl, text="Pixel: —",
                                    font=("TkDefaultFont", 9, "bold"))
        self._pixel_lbl.pack(side=tk.LEFT, padx=12)

    def _build_right_panel(self, parent):
        fig = Figure(figsize=(5, 5.8), dpi=92)
        gs  = gridspec.GridSpec(3, 1, figure=fig,
                                height_ratios=[3, 2, 1], hspace=0.12)
        self._rimg_ax = fig.add_subplot(gs[0])
        self._rsp_ax  = fig.add_subplot(gs[1])
        self._rres_ax = fig.add_subplot(gs[2], sharex=self._rsp_ax)
        self._rimg_ax.set_title("No fits yet", fontsize=8)
        self._rsp_ax.set_ylabel("Intensity", fontsize=8)
        self._rsp_ax.tick_params(labelsize=7, labelbottom=False)
        self._rres_ax.set_xlabel("Wavenumber", fontsize=8)
        self._rres_ax.set_ylabel("Residual", fontsize=7)
        for ax in (self._rimg_ax, self._rres_ax):
            ax.tick_params(labelsize=7)
        fig.tight_layout(pad=0.9)

        self._right_canvas = FigureCanvasTkAgg(fig, master=parent)
        self._right_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._right_canvas.mpl_connect("button_press_event", self._on_right_click)
        self._right_fig = fig

        ctrl = ttk.Frame(parent); ctrl.pack(fill=tk.X, pady=(2, 0))
        ttk.Label(ctrl, text="Map:").pack(side=tk.LEFT, padx=(4, 0))
        self._map_mode = tk.StringVar(value="model")
        ttk.Radiobutton(ctrl, text="Integrated model", variable=self._map_mode,
                        value="model",
                        command=self._refresh_right_image).pack(side=tk.LEFT, padx=4)
        ttk.Radiobutton(ctrl, text="N peaks", variable=self._map_mode,
                        value="npeaks",
                        command=self._refresh_right_image).pack(side=tk.LEFT)

    def _build_param_bar(self):
        bar = ttk.LabelFrame(self, text="Fitting parameters", padding=self._PAD)
        bar.pack(side=tk.BOTTOM, fill=tk.X, padx=self._PAD, pady=(0, 2))

        grid = ttk.Frame(bar); grid.pack(fill=tk.X)
        self._pv: dict[str, tk.StringVar] = {}
        for i, (key, label, default, width) in enumerate(self._PARAM_DEFS):
            col = (i % 5) * 2
            row = i // 5
            ttk.Label(grid, text=label + ":",
                      font=("TkDefaultFont", 8),
                      foreground="#444").grid(row=row, column=col,
                                             sticky=tk.E, padx=(8, 2), pady=1)
            v = tk.StringVar(value=default)
            ttk.Entry(grid, textvariable=v,
                      width=width).grid(row=row, column=col + 1, sticky=tk.W)
            self._pv[key] = v

        actions = ttk.Frame(bar); actions.pack(fill=tk.X, pady=(6, 0))
        self._fit_px_btn = ttk.Button(actions, text="▶  Fit pixel",
                                      command=self._fit_pixel)
        self._fit_px_btn.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Separator(actions, orient=tk.VERTICAL).pack(side=tk.LEFT,
                                                         fill=tk.Y, padx=6)
        ttk.Label(actions, text="Batch ROI  rows:").pack(side=tk.LEFT)
        self._roi_r0 = tk.StringVar(value="0")
        self._roi_r1 = tk.StringVar(value="")
        ttk.Entry(actions, textvariable=self._roi_r0, width=5).pack(side=tk.LEFT, padx=1)
        ttk.Label(actions, text="–").pack(side=tk.LEFT)
        ttk.Entry(actions, textvariable=self._roi_r1, width=5).pack(side=tk.LEFT, padx=(1, 6))
        ttk.Label(actions, text="cols:").pack(side=tk.LEFT)
        self._roi_c0 = tk.StringVar(value="0")
        self._roi_c1 = tk.StringVar(value="")
        ttk.Entry(actions, textvariable=self._roi_c0, width=5).pack(side=tk.LEFT, padx=1)
        ttk.Label(actions, text="–").pack(side=tk.LEFT)
        ttk.Entry(actions, textvariable=self._roi_c1, width=5).pack(side=tk.LEFT, padx=(1, 8))
        self._fit_batch_btn = ttk.Button(actions, text="⚙  Fit batch",
                                         command=self._fit_batch)
        self._fit_batch_btn.pack(side=tk.LEFT, padx=(0, 4))
        self._stop_btn = ttk.Button(actions, text="■  Stop",
                                    command=self._stop_fitting, state=tk.DISABLED)
        self._stop_btn.pack(side=tk.LEFT, padx=4)

        self._progress = ttk.Progressbar(actions, mode="determinate", length=200)
        self._progress.pack(side=tk.LEFT, padx=10)
        self._prog_lbl = ttk.Label(actions, text="",
                                   font=("TkDefaultFont", 8))
        self._prog_lbl.pack(side=tk.LEFT)

        ttk.Separator(actions, orient=tk.VERTICAL).pack(side=tk.LEFT,
                                                         fill=tk.Y, padx=8)
        ttk.Button(actions, text="Save results…",
                   command=self._save_results).pack(side=tk.LEFT, padx=4)

    def _build_status_bar(self):
        self._status = tk.StringVar(value="Ready — load a data file to begin.")
        ttk.Label(self, textvariable=self._status, relief=tk.SUNKEN, anchor=tk.W,
                  font=("TkDefaultFont", 8)).pack(side=tk.BOTTOM, fill=tk.X)

    # ═══════════════════════════════ File I/O ═════════════════════════════════

    def _browse(self):
        path = filedialog.askopenfilename(
            filetypes=[("HDF5", "*.h5 *.hdf5"),
                       ("NumPy archive", "*.npz"),
                       ("All files", "*.*")],
        )
        if path:
            self._data_path.set(path)
            self._suggest_dset(path)

    def _suggest_dset(self, path: str):
        ext = os.path.splitext(path)[1].lower()
        if ext in (".h5", ".hdf5"):
            import h5py
            best = None; best_sz = 0
            with h5py.File(path, "r") as f:
                cands: list[str] = []
                f.visititems(lambda n, o: cands.append(n)
                             if hasattr(o, "shape") and len(o.shape) >= 3 else None)
                for d in cands:
                    sz = int(np.prod(f[d].shape))
                    if sz > best_sz:
                        best_sz = sz; best = d
            if best and not self._data_dset.get():
                self._data_dset.set(best)
        elif ext == ".npz":
            d = np.load(path, allow_pickle=False)
            for k in ("unprocessed_spectrum", "data", "spectrum"):
                if k in d and d[k].ndim >= 3:
                    self._data_dset.set(k); break
        wn = _try_load_wn(path)
        if wn is not None:
            self._set_wn(wn)

    def _set_wn(self, wn: np.ndarray):
        self._wn = wn
        self._wn_src.set("auto")
        self._wn_start.set(f"{wn[0]:.3f}")
        self._wn_end.set(f"{wn[-1]:.3f}")
        self._wn_n.set(str(len(wn)))
        self._fit_ctx_ready = False

    def _load_data(self):
        path = self._data_path.get().strip()
        if not path:
            messagebox.showerror("No file", "Specify a data file first.")
            return
        dset = self._data_dset.get().strip()
        try:
            ext = os.path.splitext(path)[1].lower()
            arr = (_load_h5_array(path, dset) if ext in (".h5", ".hdf5")
                   else _load_npz_array(path, dset))
        except Exception as exc:
            messagebox.showerror("Load error", str(exc)); return

        self._raw_data = arr
        self._apply_data(arr)
        if self._wn is None:
            wn = _try_load_wn(path)
            if wn is not None:
                self._set_wn(wn)
        self._status.set(
            f"Loaded {arr.shape}  dtype={arr.dtype}  "
            f"→ oriented {self._data.shape}"
        )

    def _apply_data(self, arr: np.ndarray):
        oriented = _orient(arr, self._wn_dim.get())
        self._data = _to_real(oriented, self._take_imag.get())
        rows, cols, n_wn = self._data.shape
        self._wn_slider.configure(to=n_wn - 1)
        self._wn_slice.set(n_wn // 2)
        if self._wn is None:
            self._wn = np.arange(n_wn, dtype=np.float32)
            self._wn_start.set("0")
            self._wn_end.set(str(n_wn - 1))
            self._wn_n.set(str(n_wn))
        self._fit_map    = np.full((rows, cols), np.nan, dtype=np.float32)
        self._npeaks_map = np.full((rows, cols), np.nan, dtype=np.float32)
        self._pixel_params.clear()
        self._sel_row = rows // 2; self._sel_col = cols // 2
        self._roi_r1.set(str(rows)); self._roi_c1.set(str(cols))
        self._fit_ctx_ready = False
        self._pixel_lbl.config(
            text=f"Pixel: row={self._sel_row}, col={self._sel_col}")
        self._refresh_left_image()
        self._refresh_left_spectrum()
        self._refresh_right_image()

    def _apply_wn(self):
        try:
            start = float(self._wn_start.get())
            end   = float(self._wn_end.get())
            n     = int(self._wn_n.get())
        except ValueError:
            messagebox.showerror("Invalid", "Enter numeric values."); return
        self._wn = np.linspace(start, end, n, dtype=np.float32)
        self._wn_src.set("manual")
        self._fit_ctx_ready = False
        self._status.set(
            f"WN axis: {start:.2f} → {end:.2f}, {n} pts "
            f"(wavelet bank will be recomputed on first fit)"
        )
        self._refresh_left_spectrum()
        self._refresh_right_spectrum()

    # ═════════════════════════════ Control callbacks ══════════════════════════

    def _on_wn_dim_change(self):
        if self._raw_data is not None:
            self._apply_data(self._raw_data)

    def _on_imag_toggle(self):
        if self._raw_data is not None:
            self._data = _to_real(
                _orient(self._raw_data, self._wn_dim.get()),
                self._take_imag.get(),
            )
        self._refresh_left_image()
        self._refresh_left_spectrum()

    def _toggle_integrate(self):
        self._integrate = not self._integrate
        self._int_btn.configure(
            text="Per channel" if self._integrate else "∫ Integrate")
        self._refresh_left_image()

    def _on_left_click(self, event):
        if event.inaxes is self._limg_ax and self._data is not None:
            self._select_pixel(
                int(np.clip(round(event.ydata), 0, self._data.shape[0] - 1)),
                int(np.clip(round(event.xdata), 0, self._data.shape[1] - 1)),
            )

    def _on_right_click(self, event):
        if event.inaxes is self._rimg_ax and self._data is not None:
            self._select_pixel(
                int(np.clip(round(event.ydata), 0, self._data.shape[0] - 1)),
                int(np.clip(round(event.xdata), 0, self._data.shape[1] - 1)),
            )

    def _select_pixel(self, row: int, col: int):
        self._sel_row, self._sel_col = row, col
        self._pixel_lbl.config(text=f"Pixel: row={row}, col={col}")
        self._refresh_left_image()
        self._refresh_left_spectrum()
        self._refresh_right_image()
        self._refresh_right_spectrum()

    # ═══════════════════════════════ Rendering ════════════════════════════════

    def _get_left_img(self) -> "np.ndarray | None":
        if self._data is None:
            return None
        if self._integrate:
            return self._data.mean(axis=2)
        idx = int(np.clip(self._wn_slice.get(), 0, self._data.shape[2] - 1))
        return self._data[:, :, idx]

    def _draw_crosshair(self, ax):
        ax.plot(self._sel_col, self._sel_row, "r+",
                markersize=16, markeredgewidth=2.5, zorder=5, clip_on=True)

    def _refresh_left_image(self):
        ax = self._limg_ax; ax.cla()
        img = self._get_left_img()
        if img is not None:
            vmin, vmax = np.nanpercentile(img, [2, 98])
            ax.imshow(img, aspect="auto", cmap="viridis",
                      interpolation="nearest", vmin=vmin, vmax=vmax)
            if self._integrate:
                label = "Integrated intensity"
            else:
                idx = int(np.clip(self._wn_slice.get(), 0, self._data.shape[2] - 1))
                label = f"Channel {idx}"
                if self._wn is not None:
                    label += f"  ({self._wn[idx]:.1f} cm⁻¹)"
            ax.set_title(label + "\n(click to select pixel)", fontsize=8)
            self._draw_crosshair(ax)
        else:
            ax.set_title("No data — click to select pixel", fontsize=8)
        ax.set_xlabel("col", fontsize=7); ax.set_ylabel("row", fontsize=7)
        ax.tick_params(labelsize=7)
        self._left_fig.tight_layout(pad=0.7)
        self._left_canvas.draw_idle()

    def _refresh_left_spectrum(self):
        ax = self._lsp_ax; ax.cla()
        if self._data is None:
            self._left_fig.tight_layout(pad=0.7)
            self._left_canvas.draw_idle(); return
        wn = self._wn if self._wn is not None else np.arange(self._data.shape[2])
        r, c = self._sel_row, self._sel_col
        ax.plot(wn, self._data[r, c, :], color="black", lw=1.2, alpha=0.8)
        ax.set_title(f"Raw spectrum  (row={r}, col={c})", fontsize=8)
        ax.set_xlabel("Wavenumber", fontsize=8)
        ax.set_ylabel("Intensity", fontsize=8)
        ax.tick_params(labelsize=7)
        self._left_fig.tight_layout(pad=0.7)
        self._left_canvas.draw_idle()

    def _refresh_right_image(self):
        ax = self._rimg_ax; ax.cla()
        arr = (self._npeaks_map if self._map_mode.get() == "npeaks"
               else self._fit_map)
        if arr is not None and not np.all(np.isnan(arr)):
            cmap = mpl_cm.get_cmap("viridis").copy()
            cmap.set_bad(color="#d0d0d0")
            vmin, vmax = np.nanpercentile(arr, [2, 98])
            ax.imshow(arr, aspect="auto", cmap=cmap,
                      interpolation="nearest", vmin=vmin, vmax=vmax)
            n_fit = int(np.sum(~np.isnan(arr)))
            n_tot = int(arr.size)
            label = "N peaks" if self._map_mode.get() == "npeaks" else "Integrated model"
            ax.set_title(f"{label}  ({n_fit}/{n_tot} fitted)", fontsize=8)
        else:
            ax.set_facecolor("#eeeeee")
            ax.set_title("Fit map — no fits yet", fontsize=8)
        if self._data is not None:
            self._draw_crosshair(ax)
        ax.set_xlabel("col", fontsize=7); ax.set_ylabel("row", fontsize=7)
        ax.tick_params(labelsize=7)
        self._right_fig.tight_layout(pad=0.7)
        self._right_canvas.draw_idle()

    def _refresh_right_spectrum(self):
        sp_ax  = self._rsp_ax;  sp_ax.cla()
        res_ax = self._rres_ax; res_ax.cla()
        if self._data is None:
            self._right_fig.tight_layout(pad=0.7)
            self._right_canvas.draw_idle(); return

        r, c = self._sel_row, self._sel_col
        wn = self._wn if self._wn is not None else np.arange(self._data.shape[2])
        spectrum = self._data[r, c, :]
        sp_ax.plot(wn, spectrum, color="black", lw=1.2, alpha=0.75, label="Raw")

        params_flat = self._pixel_params.get((r, c))
        if params_flat is not None:
            p = np.asarray(params_flat).reshape(-1, 4)
            valid = p[p[:, 0] > 1e-4]
            valid = valid[np.argsort(valid[:, 1])]
            n_v   = len(valid)
            rainbow = mpl_cm.get_cmap("rainbow")
            model = np.zeros(len(wn), dtype=np.float64)
            for k, (amp, ctr, sig, gam) in enumerate(valid):
                pk     = _voigt_np(wn, amp, ctr, sig, gam)
                model += pk
                sp_ax.plot(wn, pk,
                           color=rainbow(k / max(n_v - 1, 1)),
                           lw=0.9, alpha=0.65, zorder=2)
            sp_ax.plot(wn, model, color="#e74c3c", lw=1.8, ls="--",
                       label=f"Fit  (n={n_v})", zorder=4)
            resid    = spectrum - model
            sigma_n  = float(np.std(resid))
            res_ax.plot(wn, resid, color="gray", lw=0.9)
            res_ax.axhline(0, color="black", ls="--", lw=0.8, alpha=0.5)
            res_ax.fill_between(wn, resid, alpha=0.25, color="gray")
            for s in (+sigma_n, -sigma_n):
                res_ax.axhline(s, color="#2980b9", ls=":", lw=1.1, alpha=0.85)
            res_ax.set_title(f"σ = {sigma_n:.4f}", fontsize=7, loc="right")
        else:
            res_ax.text(0.5, 0.5, "Not fitted yet",
                        transform=res_ax.transAxes,
                        ha="center", va="center", fontsize=9, color="#888")

        sp_ax.set_title(f"Fitted spectrum  (row={r}, col={c})", fontsize=8)
        sp_ax.set_ylabel("Intensity", fontsize=8)
        sp_ax.tick_params(labelsize=7, labelbottom=False)
        sp_ax.legend(fontsize=7, loc="upper right")
        res_ax.set_xlabel("Wavenumber", fontsize=8)
        res_ax.set_ylabel("Residual", fontsize=7)
        res_ax.tick_params(labelsize=7)
        self._right_fig.tight_layout(pad=0.7)
        self._right_canvas.draw_idle()

    # ═════════════════════════════ Fitting ════════════════════════════════════

    def _get_fit_params(self) -> dict:
        def _f(k): return float(self._pv[k].get())
        def _i(k): return int(self._pv[k].get())
        return {
            "adj":                       _f("adj"),
            "denoise_sigma":             _f("denoise_sigma"),
            "response_threshold":        _f("response_threshold"),
            "amp_threshold":             _f("amp_threshold"),
            "min_scale_votes":           _i("min_scale_votes"),
            "min_spacing_in":            _f("min_spacing_in"),
            "min_spacing_post":          _f("min_spacing_post"),
            "max_iter":                  _i("max_iter"),
            "tol":                       _f("tol"),
            "scale_preference_fraction": _f("scale_preference_fraction"),
        }

    def _precompute_fit_ctx(self):
        """Build wavelet bank from the current WN axis. Runs in the fit thread."""
        import torch
        import tidytorch_utils as ttu

        x      = self._wn
        dx     = float((x[-1] - x[0]) / max(len(x) - 1, 1))
        widths = np.linspace(1, 10, 100) * dx

        x_t    = torch.as_tensor(x,      dtype=torch.float32)
        sigs   = torch.as_tensor(widths, dtype=torch.float32)
        gams   = torch.tensor([1.0],     dtype=torch.float32)

        x_c    = x_t - x_t.mean()
        prof   = ttu.pseudo_voigt(x_c.view(1, 1, -1),
                                  sigs.view(-1, 1, 1),
                                  gams.view(1, -1, 1))
        wav    = prof / (prof.amax(dim=-1, keepdim=True) + 1e-12)

        self._x_t   = x_t
        self._sigs  = sigs
        self._gams  = gams
        self._wav   = wav                  # (n_widths, 1, n_pts) on CPU
        self._fit_ctx_ready = True
        self._fit_queue.put({"type": "status",
                              "msg": f"Wavelet bank ready  ({len(widths)} widths)"})

    def _run_one_pixel(self, row: int, col: int, fp: dict) -> dict:
        """Fit one pixel and return a result dict. Runs in a worker thread."""
        import torch
        import tidytorch_utils as ttu

        if not self._fit_ctx_ready:
            self._precompute_fit_ctx()

        spectrum = self._data[row, col, :]
        spec_t   = torch.as_tensor(spectrum, dtype=torch.float32)
        spec_d   = ttu.denoise_spectrum(spec_t, self._x_t,
                                        sigma=fp["denoise_sigma"], gamma=0.0)
        spec_in  = spec_d.numpy() + fp["adj"]

        params, converged, n_iter, *_ = ttu.process_conv_deriv_fit(
            spec_in,   self._wn,
            sigmas=self._sigs, gammas=self._gams,
            wavelet_peaks=self._wav,
            response_threshold=fp["response_threshold"],
            amp_threshold=fp["amp_threshold"],
            min_scale_votes=fp["min_scale_votes"],
            min_spacing_in=fp["min_spacing_in"],
            min_spacing_post=fp["min_spacing_post"],
            max_iter=fp["max_iter"],
            tol=fp["tol"],
            scale_preference_fraction=fp["scale_preference_fraction"],
        )
        return {
            "type": "pixel",
            "row": row, "col": col,
            "params": params.detach().cpu().numpy().astype(np.float32),
            "converged": bool(converged),
            "n_iter": int(n_iter),
        }

    def _fit_pixel(self):
        if self._data is None or self._wn is None:
            messagebox.showwarning("No data", "Load data and set WN axis first.")
            return
        if self._busy:
            messagebox.showwarning("Busy", "A fit is already running."); return
        try:
            fp = self._get_fit_params()
        except ValueError as exc:
            messagebox.showerror("Parameter error", str(exc)); return
        row, col = self._sel_row, self._sel_col
        self._set_busy(True, f"Fitting pixel ({row}, {col})…")
        threading.Thread(target=self._pixel_worker,
                         args=(row, col, fp), daemon=True).start()

    def _pixel_worker(self, row: int, col: int, fp: dict):
        try:
            msg = self._run_one_pixel(row, col, fp)
            self._fit_queue.put(msg)
        except Exception as exc:
            self._fit_queue.put({"type": "error", "msg": str(exc)})

    def _fit_batch(self):
        if self._data is None or self._wn is None:
            messagebox.showwarning("No data", "Load data and set WN axis first.")
            return
        if self._busy:
            messagebox.showwarning("Busy", "A fit is already running."); return
        try:
            fp    = self._get_fit_params()
            rows, cols = self._data.shape[:2]
            r0 = int(self._roi_r0.get())
            r1 = int(self._roi_r1.get() or rows)
            c0 = int(self._roi_c0.get())
            c1 = int(self._roi_c1.get() or cols)
        except ValueError as exc:
            messagebox.showerror("Parameter error", str(exc)); return
        total = max((r1 - r0) * (c1 - c0), 0)
        if total == 0:
            messagebox.showerror("ROI error", "ROI is empty — check bounds.")
            return
        self._stop_flag = False
        self._progress.configure(maximum=total, value=0)
        self._set_busy(True, f"Batch fitting {total} pixels…")
        threading.Thread(target=self._batch_worker,
                         args=(r0, r1, c0, c1, fp), daemon=True).start()

    def _batch_worker(self, r0: int, r1: int, c0: int, c1: int, fp: dict):
        try:
            done = 0
            total = (r1 - r0) * (c1 - c0)
            for row in range(r0, r1):
                for col in range(c0, c1):
                    if self._stop_flag:
                        self._fit_queue.put({"type": "stopped"})
                        return
                    try:
                        msg = self._run_one_pixel(row, col, fp)
                    except Exception as exc:
                        msg = {"type": "pixel_error", "row": row, "col": col,
                               "msg": str(exc)}
                    done += 1
                    msg["done"]  = done
                    msg["total"] = total
                    self._fit_queue.put(msg)
            self._fit_queue.put({"type": "batch_done", "n": done})
        except Exception as exc:
            self._fit_queue.put({"type": "error", "msg": str(exc)})

    def _stop_fitting(self):
        self._stop_flag = True
        self._status.set("Stopping after current pixel…")

    # ═════════════════════════════ Queue / update ═════════════════════════════

    def _poll_queue(self):
        try:
            while True:
                self._handle_msg(self._fit_queue.get_nowait())
        except queue.Empty:
            pass
        self.after(80, self._poll_queue)

    def _handle_msg(self, msg: dict):
        t = msg["type"]
        if t == "pixel":
            r, c, fp_arr = msg["row"], msg["col"], msg["params"]
            self._pixel_params[(r, c)] = fp_arr
            self._update_maps(r, c, fp_arr)
            done, total = msg.get("done"), msg.get("total")
            if done is not None:
                self._progress["value"] = done
                self._prog_lbl.config(text=f"{done}/{total}")
            if (r, c) == (self._sel_row, self._sel_col):
                self._refresh_right_spectrum()
            if done is None or done % max(total // 50 if total else 1, 1) == 0:
                self._refresh_right_image()
            if done is None:  # single pixel — done
                n_peaks = int((fp_arr.reshape(-1, 4)[:, 0] > 1e-4).sum())
                self._set_busy(False)
                self._status.set(
                    f"Fitted ({r}, {c}) — {n_peaks} peaks  "
                    f"converged={msg.get('converged', '?')}  "
                    f"iters={msg.get('n_iter', '?')}"
                )
        elif t == "pixel_error":
            r, c = msg["row"], msg["col"]
            self._status.set(f"Pixel ({r},{c}) error: {msg['msg']}")
        elif t == "batch_done":
            self._set_busy(False)
            self._refresh_right_image()
            self._status.set(f"Batch complete — {msg['n']} pixels fitted.")
        elif t == "stopped":
            self._set_busy(False)
            self._refresh_right_image()
            self._status.set("Batch stopped by user.")
        elif t == "error":
            self._set_busy(False)
            messagebox.showerror("Fit error", msg["msg"])
        elif t == "status":
            self._status.set(msg["msg"])

    def _update_maps(self, row: int, col: int, params_flat: np.ndarray):
        if self._fit_map is None:
            return
        wn  = self._wn if self._wn is not None else np.arange(self._data.shape[2])
        p   = np.asarray(params_flat).reshape(-1, 4)
        valid = p[p[:, 0] > 1e-4]
        model = sum((_voigt_np(wn, *pk) for pk in valid),
                    np.zeros(len(wn), dtype=np.float64))
        dx_wn = float(wn[1] - wn[0]) if len(wn) > 1 else 1.0
        self._fit_map[row, col]    = float(np.sum(model) * dx_wn)
        self._npeaks_map[row, col] = float(len(valid))

    def _set_busy(self, state: bool, msg: str = ""):
        self._busy = state
        s = tk.DISABLED if state else tk.NORMAL
        self._fit_px_btn.configure(state=s)
        self._fit_batch_btn.configure(state=s)
        self._stop_btn.configure(state=tk.NORMAL if state else tk.DISABLED)
        if msg:
            self._status.set(msg)

    # ═════════════════════════════ Save ═══════════════════════════════════════

    def _save_results(self):
        if not self._pixel_params:
            messagebox.showwarning("Nothing to save", "Run the fitting first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".npz",
            filetypes=[("NumPy archive", "*.npz"), ("HDF5", "*.h5 *.hdf5")],
        )
        if not path:
            return
        rows, cols, n_wn = self._data.shape
        max_len = max((v.shape[0] for v in self._pixel_params.values()), default=4)
        cube = np.full((rows, cols, max_len), np.nan, dtype=np.float32)
        for (r, c), v in self._pixel_params.items():
            n = min(v.shape[0], max_len)
            cube[r, c, :n] = v[:n]
        wn_arr = self._wn if self._wn is not None else np.arange(n_wn, dtype=np.float32)
        try:
            ext = os.path.splitext(path)[1].lower()
            if ext == ".npz":
                np.savez_compressed(path,
                                    raw_data=self._data,
                                    peak_params=cube,
                                    x_axis=wn_arr,
                                    fit_map=self._fit_map,
                                    n_peaks_map=self._npeaks_map)
            else:
                import h5py
                with h5py.File(path, "w") as f:
                    f.create_dataset("raw_data",    data=self._data)
                    f.create_dataset("peak_params", data=cube)
                    f.create_dataset("x_axis",      data=wn_arr)
                    f.create_dataset("fit_map",     data=self._fit_map)
                    f.create_dataset("n_peaks_map", data=self._npeaks_map)
            self._status.set(f"Saved → {os.path.basename(path)}")
        except Exception as exc:
            messagebox.showerror("Save error", str(exc))


# ── Entry point ───────────────────────────────────────────────────────────────

def launch():
    app = CascadeFitGUI()
    app.mainloop()


if __name__ == "__main__":
    launch()
