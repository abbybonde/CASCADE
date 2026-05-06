"""
metrics_gui.py — CASCADE Fit-Characteristics Inspector
=======================================================
Interactive pixel picker that shows the noise and peak-separation metrics
returned by ``estimate_fit_characteristics()`` for any pixel in a fitted
hyperspectral dataset.

Usage
-----
    python metrics_gui.py

Layout
------
  Top bar   — two-row file loader (data cube + params cube, H5 or NPZ)
  Left pane — clickable 2-D spatial image with:
               • WN-axis dimension selector (0 / 1 / 2)
               • Imag / Real toggle (for complex BCARS data)
               • Per-channel slider or integrated-intensity view
               • Manual wavenumber-axis entry (start / end / n pts)
  Right pane — spectrum viewer (raw spectrum, fitted model, component
               peaks, residual with ±σ bands) + scalar metrics table
               and per-peak table (centre, amplitude, FWHM, separability)
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.cm as cm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec


# ── CASCADE import (lazy, so the file can be opened without a GPU) ────────────

_ESTIMATE_FN = None


def _get_estimate_fn():
    global _ESTIMATE_FN
    if _ESTIMATE_FN is None:
        try:
            from tidytorch_utils import estimate_fit_characteristics  # noqa: PLC0415
            _ESTIMATE_FN = estimate_fit_characteristics
        except ImportError as exc:
            messagebox.showerror(
                "Import error",
                f"Cannot import estimate_fit_characteristics:\n{exc}\n\n"
                "Run metrics_gui.py from the CASCADE project directory.",
            )
            sys.exit(1)
    return _ESTIMATE_FN


# ── Small file-I/O helpers ────────────────────────────────────────────────────

def _h5_datasets(path: str) -> list[str]:
    import h5py
    out: list[str] = []
    with h5py.File(path, "r") as f:
        f.visititems(lambda n, o: out.append(n) if hasattr(o, "shape") else None)
    return out


def _load_h5_array(path: str, dset: str) -> np.ndarray:
    import h5py
    with h5py.File(path, "r") as f:
        if dset and dset in f:
            return np.array(f[dset])
        # Auto-detect: largest dataset with ≥ 2 axes
        candidates: list[tuple[int, str]] = []
        def _visit(name, obj):
            if hasattr(obj, "shape") and len(obj.shape) >= 2:
                candidates.append((int(np.prod(obj.shape)), name))
        f.visititems(_visit)
        if not candidates:
            raise KeyError(f"No 2-D+ dataset found in {path}")
        candidates.sort(reverse=True)
        chosen = candidates[0][1]
        messagebox.showinfo(
            "Dataset auto-detected",
            f"No dataset path given — loaded:\n  {chosen}\n"
            f"  shape: {tuple(f[chosen].shape)}\n\n"
            "Paste this path into the 'Dataset' field to suppress.",
        )
        return np.array(f[chosen])


def _load_npz_array(path: str, key: str) -> np.ndarray:
    d = np.load(path, allow_pickle=False)
    if key and key in d:
        return d[key]
    for k in ("peak_params", "params", "fitted_params",
              "unprocessed_spectrum", "data", "spectrum"):
        if k in d:
            messagebox.showinfo("Key auto-detected",
                                f"Loaded key '{k}' from {os.path.basename(path)}")
            return d[k]
    keys = list(d.keys())
    if keys:
        messagebox.showinfo("Key auto-detected",
                            f"Loaded key '{keys[0]}' from {os.path.basename(path)}")
        return d[keys[0]]
    raise KeyError(f"No arrays found in {path}")


def _try_load_wn(path: str) -> np.ndarray | None:
    """Return a 1-D wavenumber axis from an H5 or NPZ file, or None."""
    ext = os.path.splitext(path)[1].lower()
    wn_h5_paths = [
        "preprocessed_images/x_axis", "x_axis", "wn_axis",
        "wavenumbers", "spectral_axis",
    ]
    try:
        if ext in (".h5", ".hdf5"):
            import h5py
            with h5py.File(path, "r") as f:
                for p in wn_h5_paths:
                    if p in f and len(f[p].shape) == 1:
                        return np.array(f[p], dtype=np.float32)
        elif ext == ".npz":
            d = np.load(path, allow_pickle=False)
            for k in ("wn_axis", "x_axis", "wavenumbers", "spectral_axis"):
                if k in d and d[k].ndim == 1:
                    return d[k].astype(np.float32)
    except Exception:
        pass
    return None


def _orient(data: np.ndarray, wn_dim: int) -> np.ndarray:
    """Reorder so the result is always (rows, cols, n_wn)."""
    if data.ndim == 2:
        # Treat as a single-row image: (1, n_cols, n_wn)
        return data[np.newaxis, :, :]
    if data.ndim != 3:
        raise ValueError(f"Expected 2-D or 3-D array, got shape {data.shape}")
    return np.moveaxis(data, wn_dim, -1)


def _to_real(arr: np.ndarray, take_imag: bool) -> np.ndarray:
    if np.iscomplexobj(arr):
        return (np.imag(arr) if take_imag else np.real(arr)).astype(np.float32)
    return arr.astype(np.float32)


# ── Pure-numpy pseudo-Voigt for drawing individual peaks ─────────────────────

def _voigt_np(x: np.ndarray, amp: float, ctr: float,
              sig: float, gam: float) -> np.ndarray:
    """Thompson-Cox-Hastings pseudo-Voigt, matching pseudo_voigt() in tidytorch."""
    sig = max(sig, 1e-9)
    gam = max(gam, 1e-9)
    fg  = 2.35482 * sig
    fl  = 2.0     * gam
    fv  = (fg**5 + 2.69269*fg**4*fl + 2.42843*fg**3*fl**2
           + 4.47163*fg**2*fl**3 + 0.07842*fg*fl**4 + fl**5) ** 0.2
    fv  = max(fv, 1e-9)
    z   = (x - ctr) / fv
    r   = fl / fv
    eta = float(np.clip(1.36603*r - 0.47719*r**2 + 0.11116*r**3, 0.0, 1.0))
    return amp * (eta / (1.0 + 4.0*z**2) + (1.0 - eta)*np.exp(-4.0*np.log(2.0)*z**2))


# ── Main application ──────────────────────────────────────────────────────────

class CascadeInspector(tk.Tk):
    _PAD = 4

    def __init__(self):
        super().__init__()
        self.title("CASCADE — Fit Characteristics Inspector")
        self.geometry("1380x840")
        self.minsize(960, 640)

        # ── Mutable state ─────────────────────────────────────────────────
        self._raw_data: np.ndarray | None = None  # original array, un-oriented
        self._data:     np.ndarray | None = None  # (rows, cols, n_wn) float32
        self._params:   np.ndarray | None = None  # (rows, cols, n_params)
        self._wn:       np.ndarray | None = None  # (n_wn,)
        self._sel_row: int = 0
        self._sel_col: int = 0
        self._integrate: bool = False

        self._build_ui()

    # ═══════════════════════════════════════ UI construction ═══════════════════

    def _build_ui(self):
        self._build_file_bar()
        self._build_main_pane()
        self._build_status_bar()

    # ── File-loading bar ──────────────────────────────────────────────────────

    def _build_file_bar(self):
        bar = ttk.LabelFrame(self, text="Data & Parameters", padding=self._PAD)
        bar.pack(side=tk.TOP, fill=tk.X, padx=self._PAD, pady=(self._PAD, 0))

        # Row 0 — raw data cube
        r0 = ttk.Frame(bar)
        r0.pack(fill=tk.X)
        ttk.Label(r0, text="Data file:", width=11).pack(side=tk.LEFT)
        self._data_path = tk.StringVar()
        ttk.Entry(r0, textvariable=self._data_path, width=50).pack(side=tk.LEFT, padx=2)
        ttk.Button(r0, text="Browse…", command=self._browse_data).pack(side=tk.LEFT, padx=2)
        ttk.Label(r0, text="  Dataset:").pack(side=tk.LEFT)
        self._data_dset = tk.StringVar()
        ttk.Entry(r0, textvariable=self._data_dset, width=38).pack(side=tk.LEFT, padx=2)
        ttk.Button(r0, text="Load Data", command=self._load_data,
                   style="Accent.TButton").pack(side=tk.LEFT, padx=6)

        # Row 1 — fitted params cube
        r1 = ttk.Frame(bar)
        r1.pack(fill=tk.X, pady=(2, 0))
        ttk.Label(r1, text="Params file:", width=11).pack(side=tk.LEFT)
        self._params_path = tk.StringVar()
        ttk.Entry(r1, textvariable=self._params_path, width=50).pack(side=tk.LEFT, padx=2)
        ttk.Button(r1, text="Browse…", command=self._browse_params).pack(side=tk.LEFT, padx=2)
        ttk.Label(r1, text="  Dataset:").pack(side=tk.LEFT)
        self._params_dset = tk.StringVar()
        ttk.Entry(r1, textvariable=self._params_dset, width=38).pack(side=tk.LEFT, padx=2)
        ttk.Button(r1, text="Load Params", command=self._load_params,
                   style="Accent.TButton").pack(side=tk.LEFT, padx=6)

    # ── Main pane (left image | right spectrum+metrics) ───────────────────────

    def _build_main_pane(self):
        pane = tk.PanedWindow(self, orient=tk.HORIZONTAL,
                              sashrelief=tk.RAISED, sashwidth=5)
        pane.pack(fill=tk.BOTH, expand=True,
                  padx=self._PAD, pady=self._PAD)

        left = ttk.Frame(pane)
        pane.add(left, minsize=320)
        self._build_image_panel(left)

        right = ttk.Frame(pane)
        pane.add(right, minsize=560)
        self._build_spectrum_panel(right)
        self._build_metrics_panel(right)

    # ── Left: image + controls ────────────────────────────────────────────────

    def _build_image_panel(self, parent):
        # Matplotlib figure
        self._img_fig = Figure(figsize=(4, 3.6), dpi=95)
        self._img_ax  = self._img_fig.add_subplot(111)
        self._img_ax.set_title("No data loaded — click to select pixel", fontsize=8)
        self._img_fig.tight_layout(pad=1.0)

        self._img_canvas = FigureCanvasTkAgg(self._img_fig, master=parent)
        self._img_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._img_canvas.mpl_connect("button_press_event", self._on_image_click)

        # ── Controls below the image ──────────────────────────────────────
        ctrl = ttk.LabelFrame(parent, text="Display & axis", padding=self._PAD)
        ctrl.pack(fill=tk.X, pady=(2, 0))

        # WN dim selector + imag toggle
        r1 = ttk.Frame(ctrl)
        r1.pack(fill=tk.X)
        ttk.Label(r1, text="WN axis dim:").pack(side=tk.LEFT)
        self._wn_dim = tk.IntVar(value=2)
        for val, label in [(0, "0"), (1, "1"), (2, "2 ✓")]:
            ttk.Radiobutton(r1, text=label, variable=self._wn_dim, value=val,
                            command=self._on_wn_dim_change).pack(side=tk.LEFT, padx=3)

        ttk.Separator(r1, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        self._take_imag = tk.BooleanVar(value=True)
        ttk.Checkbutton(r1, text="Imag (BCARS)",
                        variable=self._take_imag,
                        command=self._on_imag_toggle).pack(side=tk.LEFT, padx=4)

        # WN-slice slider for image channel
        r2 = ttk.Frame(ctrl)
        r2.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(r2, text="Image channel:").pack(side=tk.LEFT)
        self._wn_slice = tk.IntVar(value=0)
        self._wn_slider = tk.Scale(
            r2, from_=0, to=1, orient=tk.HORIZONTAL,
            variable=self._wn_slice, showvalue=True,
            command=lambda _: self._refresh_image(),
            length=160, resolution=1,
        )
        self._wn_slider.pack(side=tk.LEFT, padx=4)
        self._integrate_btn = ttk.Button(r2, text="∫ Integrate",
                                         command=self._toggle_integrate)
        self._integrate_btn.pack(side=tk.LEFT, padx=4)

        # Wavenumber axis — from file or manual
        r3 = ttk.LabelFrame(ctrl, text="Wavenumber axis", padding=2)
        r3.pack(fill=tk.X, pady=(4, 0))
        self._wn_src = tk.StringVar(value="auto")
        ttk.Radiobutton(r3, text="From file",
                        variable=self._wn_src, value="auto").pack(side=tk.LEFT)
        ttk.Radiobutton(r3, text="Manual",
                        variable=self._wn_src, value="manual").pack(side=tk.LEFT, padx=6)

        r4 = ttk.Frame(r3)
        r4.pack(fill=tk.X, pady=(2, 0))
        for label, attr, w in [("start:", "_wn_start", 7),
                                ("end:",   "_wn_end",   7),
                                ("n pts:", "_wn_n",     6)]:
            ttk.Label(r4, text=label).pack(side=tk.LEFT, padx=(6, 1))
            var = tk.StringVar()
            setattr(self, attr, var)
            ttk.Entry(r4, textvariable=var, width=w).pack(side=tk.LEFT)
        ttk.Button(r4, text="Apply", command=self._apply_wn_axis).pack(side=tk.LEFT, padx=6)

        # Selected pixel readout
        r5 = ttk.Frame(ctrl)
        r5.pack(fill=tk.X, pady=(6, 2))
        self._pixel_lbl = ttk.Label(r5, text="Selected pixel: —",
                                    font=("TkDefaultFont", 9, "bold"))
        self._pixel_lbl.pack(side=tk.LEFT)

    # ── Right-top: spectrum viewer ─────────────────────────────────────────────

    def _build_spectrum_panel(self, parent):
        sp_frame = ttk.LabelFrame(parent, text="Spectrum", padding=0)
        sp_frame.pack(fill=tk.BOTH, expand=True)

        self._sp_fig = Figure(figsize=(7, 3.8), dpi=95)
        gs = gridspec.GridSpec(2, 1, figure=self._sp_fig,
                               height_ratios=[3, 1], hspace=0.05)
        self._sp_ax  = self._sp_fig.add_subplot(gs[0])
        self._res_ax = self._sp_fig.add_subplot(gs[1], sharex=self._sp_ax)

        self._sp_ax.set_ylabel("Intensity", fontsize=9)
        self._sp_ax.tick_params(labelsize=7, labelbottom=False)
        self._res_ax.set_ylabel("Residual", fontsize=8)
        self._res_ax.set_xlabel("Wavenumber", fontsize=9)
        self._res_ax.tick_params(labelsize=7)
        self._sp_fig.tight_layout(pad=0.8)

        self._sp_canvas = FigureCanvasTkAgg(self._sp_fig, master=sp_frame)
        self._sp_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ── Right-bottom: metrics ─────────────────────────────────────────────────

    def _build_metrics_panel(self, parent):
        mf = ttk.LabelFrame(parent, text="Fit characteristics", padding=self._PAD)
        mf.pack(fill=tk.X, pady=(3, 0))

        # Scalar metrics in a compact grid
        grid = ttk.Frame(mf)
        grid.pack(fill=tk.X)

        _METRIC_DEFS = [
            ("n_peaks",             "Fitted peaks"),
            ("noise_std",           "Noise σ"),
            ("noise_to_peak_ratio", "Noise / peak"),
            ("min_separability",    "Min sep"),
            ("median_separability", "Median sep"),
            ("mean_separability",   "Mean sep"),
        ]
        self._m_vars: dict[str, tk.StringVar] = {}
        for col, (key, label) in enumerate(_METRIC_DEFS):
            ttk.Label(grid, text=label + ":", font=("TkDefaultFont", 8),
                      foreground="#555").grid(row=0, column=col*2, sticky=tk.W, padx=(8, 1))
            var = tk.StringVar(value="—")
            ttk.Label(grid, textvariable=var, width=9,
                      font=("TkFixedFont", 9, "bold"),
                      foreground="#1a5276").grid(row=0, column=col*2+1, sticky=tk.W)
            self._m_vars[key] = var

        # Per-peak table
        tbl_frame = ttk.Frame(mf)
        tbl_frame.pack(fill=tk.X, pady=(6, 0))
        _COLS = ("center", "amplitude", "fwhm", "separability")
        self._peak_table = ttk.Treeview(
            tbl_frame, columns=_COLS, show="headings",
            height=5, selectmode="none",
        )
        for col, (key, head, width) in enumerate([
            ("center",       "Centre (wn)",   100),
            ("amplitude",    "Amplitude",       90),
            ("fwhm",         "FWHM",            80),
            ("separability", "Separability",   100),
        ]):
            self._peak_table.heading(key, text=head)
            self._peak_table.column(key, width=width, anchor=tk.CENTER)

        vsb = ttk.Scrollbar(tbl_frame, orient=tk.VERTICAL,
                            command=self._peak_table.yview)
        self._peak_table.configure(yscrollcommand=vsb.set)
        self._peak_table.pack(side=tk.LEFT, fill=tk.X, expand=True)
        vsb.pack(side=tk.LEFT, fill=tk.Y)

    # ── Status bar ────────────────────────────────────────────────────────────

    def _build_status_bar(self):
        self._status = tk.StringVar(value="Ready — load data to begin.")
        ttk.Label(self, textvariable=self._status,
                  relief=tk.SUNKEN, anchor=tk.W,
                  font=("TkDefaultFont", 8)).pack(side=tk.BOTTOM, fill=tk.X)

    # ═══════════════════════════════════════ File I/O ══════════════════════════

    def _browse_data(self):
        path = filedialog.askopenfilename(
            title="Select data file",
            filetypes=[("HDF5", "*.h5 *.hdf5"),
                       ("NumPy archive", "*.npz"),
                       ("All files", "*.*")],
        )
        if path:
            self._data_path.set(path)
            self._suggest_data_dset(path)

    def _browse_params(self):
        path = filedialog.askopenfilename(
            title="Select parameters file (can be the same as data)",
            filetypes=[("HDF5", "*.h5 *.hdf5"),
                       ("NumPy archive", "*.npz"),
                       ("All files", "*.*")],
        )
        if path:
            self._params_path.set(path)
            self._suggest_params_dset(path)

    def _suggest_data_dset(self, path: str):
        """Pre-fill the dataset field and auto-load WN axis when browsing."""
        ext = os.path.splitext(path)[1].lower()
        if ext in (".h5", ".hdf5"):
            import h5py
            dsets = _h5_datasets(path)
            best = None; best_sz = 0
            with h5py.File(path, "r") as f:
                for d in dsets:
                    if d in f and len(f[d].shape) >= 3:
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
        # Try to harvest a WN axis
        wn = _try_load_wn(path)
        if wn is not None:
            self._set_wn(wn)
            self._status.set(f"WN axis found in data file ({len(wn)} pts)")

    def _suggest_params_dset(self, path: str):
        ext = os.path.splitext(path)[1].lower()
        if ext in (".h5", ".hdf5"):
            if not self._params_dset.get():
                self._params_dset.set("preprocessed_images/peak_params")
        elif ext == ".npz":
            d = np.load(path, allow_pickle=False)
            for k in ("peak_params", "params", "fitted_params"):
                if k in d:
                    self._params_dset.set(k); break
        if self._wn is None:
            wn = _try_load_wn(path)
            if wn is not None:
                self._set_wn(wn)
                self._status.set(f"WN axis found in params file ({len(wn)} pts)")

    def _set_wn(self, wn: np.ndarray):
        self._wn = wn
        self._wn_src.set("auto")
        self._wn_start.set(f"{wn[0]:.3f}")
        self._wn_end.set(f"{wn[-1]:.3f}")
        self._wn_n.set(str(len(wn)))

    def _load_data(self):
        path = self._data_path.get().strip()
        if not path:
            messagebox.showerror("No file", "Specify a data file first.")
            return
        dset = self._data_dset.get().strip()
        try:
            ext = os.path.splitext(path)[1].lower()
            if ext in (".h5", ".hdf5"):
                arr = _load_h5_array(path, dset)
            elif ext in (".npz", ".npy"):
                arr = _load_npz_array(path, dset)
            else:
                messagebox.showerror("Unsupported", f"Cannot load '{ext}' files.")
                return
        except Exception as exc:
            messagebox.showerror("Load error", str(exc))
            return

        self._raw_data = arr
        self._apply_data(arr)

        if self._wn is None:
            wn = _try_load_wn(path)
            if wn is not None:
                self._set_wn(wn)

        self._status.set(
            f"Data loaded: {arr.shape}  dtype={arr.dtype}  "
            f"(WN dim={self._wn_dim.get()}, "
            f"{'imag' if self._take_imag.get() else 'real'})"
        )

    def _apply_data(self, arr: np.ndarray):
        """Orient + cast the raw array and refresh the image."""
        oriented = _orient(arr, self._wn_dim.get())
        self._data = _to_real(oriented, self._take_imag.get())

        n_wn = self._data.shape[2]
        self._wn_slider.configure(to=n_wn - 1)
        mid = n_wn // 2
        self._wn_slice.set(mid)

        if self._wn is None:
            self._wn = np.arange(n_wn, dtype=np.float32)
            self._wn_start.set("0")
            self._wn_end.set(str(n_wn - 1))
            self._wn_n.set(str(n_wn))

        rows, cols = self._data.shape[:2]
        self._sel_row = rows // 2
        self._sel_col = cols // 2
        self._refresh_image()

    def _load_params(self):
        # Allow pointing to the same file as data
        path = self._params_path.get().strip() or self._data_path.get().strip()
        if not path:
            messagebox.showerror("No file", "Specify a params file first.")
            return
        dset = self._params_dset.get().strip()
        try:
            ext = os.path.splitext(path)[1].lower()
            if ext in (".h5", ".hdf5"):
                arr = _load_h5_array(path, dset)
            elif ext == ".npz":
                arr = _load_npz_array(path, dset)
            else:
                messagebox.showerror("Unsupported", f"Cannot load '{ext}' files.")
                return
        except Exception as exc:
            messagebox.showerror("Load error", str(exc))
            return

        # Reshape flat (B, n_params) → (rows, cols, n_params) if needed
        if arr.ndim == 2 and self._data is not None:
            rows, cols = self._data.shape[:2]
            try:
                arr = arr.reshape(rows, cols, -1)
            except ValueError:
                messagebox.showwarning(
                    "Shape mismatch",
                    f"Cannot reshape params {arr.shape} to ({rows}, {cols}, -1).\n"
                    "Ensure the params match the loaded data's spatial dimensions.",
                )
                return
        elif arr.ndim == 2 and self._data is None:
            messagebox.showwarning("No data", "Load the data cube first.")
            return

        self._params = arr.astype(np.float32)

        if self._wn is None:
            wn = _try_load_wn(path)
            if wn is not None:
                self._set_wn(wn)

        self._status.set(f"Params loaded: {arr.shape}")
        self._update_spectrum_and_metrics()

    # ═══════════════════════════════════════ Control callbacks ═════════════════

    def _on_wn_dim_change(self):
        if self._raw_data is not None:
            self._apply_data(self._raw_data)

    def _on_imag_toggle(self):
        if self._raw_data is not None:
            oriented = _orient(self._raw_data, self._wn_dim.get())
            self._data = _to_real(oriented, self._take_imag.get())
        self._refresh_image()
        self._update_spectrum_and_metrics()

    def _toggle_integrate(self):
        self._integrate = not self._integrate
        lbl = "Channel" if self._integrate else "∫ Integrate"
        self._integrate_btn.configure(text="∫ Integrate" if not self._integrate else "Per channel")
        self._refresh_image()

    def _apply_wn_axis(self):
        try:
            start = float(self._wn_start.get())
            end   = float(self._wn_end.get())
            n     = int(self._wn_n.get())
        except ValueError:
            messagebox.showerror("Invalid", "Enter numeric values for start, end, n pts.")
            return
        self._wn = np.linspace(start, end, n, dtype=np.float32)
        self._wn_src.set("manual")
        self._status.set(f"WN axis set manually: {start:.2f} → {end:.2f}, {n} pts")
        self._update_spectrum_and_metrics()

    def _on_image_click(self, event):
        if event.inaxes is not self._img_ax or self._data is None:
            return
        col = int(np.clip(round(event.xdata), 0, self._data.shape[1] - 1))
        row = int(np.clip(round(event.ydata), 0, self._data.shape[0] - 1))
        self._sel_row = row
        self._sel_col = col
        self._pixel_lbl.config(text=f"Selected pixel: row={row}, col={col}")
        self._refresh_image()
        self._update_spectrum_and_metrics()

    # ═══════════════════════════════════════ Rendering ═════════════════════════

    def _get_image_array(self) -> np.ndarray | None:
        if self._data is None:
            return None
        if self._integrate:
            return self._data.mean(axis=2)
        idx = int(np.clip(self._wn_slice.get(), 0, self._data.shape[2] - 1))
        return self._data[:, :, idx]

    def _refresh_image(self):
        ax = self._img_ax
        ax.cla()

        img = self._get_image_array()
        if img is not None:
            vmin, vmax = np.nanpercentile(img, [2, 98])
            ax.imshow(img, aspect="auto", cmap="viridis",
                      interpolation="nearest",
                      vmin=vmin, vmax=vmax)
            label = ("Integrated intensity"
                     if self._integrate
                     else f"Channel {self._wn_slice.get()}")
            wn_val = ""
            if self._wn is not None and not self._integrate:
                idx = int(np.clip(self._wn_slice.get(), 0, len(self._wn) - 1))
                wn_val = f"  ({self._wn[idx]:.1f} cm⁻¹)"
            ax.set_title(f"{label}{wn_val}\n(click to select pixel)", fontsize=8)
            ax.set_xlabel("col", fontsize=8)
            ax.set_ylabel("row", fontsize=8)
            # Crosshair at selected pixel
            ax.plot(self._sel_col, self._sel_row, "r+",
                    markersize=16, markeredgewidth=2.5, zorder=5)
        else:
            ax.set_title("No data loaded — click to select pixel", fontsize=8)

        ax.tick_params(labelsize=7)
        self._img_fig.tight_layout(pad=0.6)
        self._img_canvas.draw_idle()

    def _update_spectrum_and_metrics(self):
        if self._data is None:
            return

        row, col = self._sel_row, self._sel_col
        spectrum = self._data[row, col, :]
        wn = (self._wn if self._wn is not None
              else np.arange(len(spectrum), dtype=np.float32))

        params_flat: np.ndarray | None = None
        if (self._params is not None
                and self._params.shape[:2] == self._data.shape[:2]):
            params_flat = self._params[row, col, :]

        result = None
        if params_flat is not None:
            try:
                result = _get_estimate_fn()(spectrum, params_flat, wn)
            except Exception as exc:
                self._status.set(f"estimate_fit_characteristics error: {exc}")

        self._draw_spectrum(wn, spectrum, params_flat, result)
        self._update_metrics(result)

    def _draw_spectrum(self, wn, spectrum, params_flat, result):
        sp_ax  = self._sp_ax
        res_ax = self._res_ax
        sp_ax.cla()
        res_ax.cla()

        sp_ax.plot(wn, spectrum, color="black", lw=1.2,
                   alpha=0.75, label="Raw spectrum", zorder=3)

        if result is not None and result["n_peaks"] > 0:
            n_peaks = result["n_peaks"]
            model   = result["fitted_model"]
            resid   = result["residuals"]
            noise   = result["noise_std"]

            # Re-extract valid peaks in centre-sorted order
            p = params_flat.reshape(-1, 4)
            valid = p[:, 0] > 1e-2
            p_v   = p[valid]
            p_v   = p_v[np.argsort(p_v[:, 1])]

            rainbow = cm.get_cmap("rainbow")
            n_v = len(p_v)
            for k, (amp, ctr, sig, gam) in enumerate(p_v):
                peak_y = _voigt_np(wn, amp, ctr, sig, gam)
                sp_ax.plot(wn, peak_y,
                           color=rainbow(k / max(n_v - 1, 1)),
                           lw=0.9, alpha=0.65, zorder=2)

            sp_ax.plot(wn, model, color="#e74c3c", lw=1.8,
                       ls="--", label=f"Fit (n={n_peaks})", zorder=4)

            # Residual panel
            res_ax.plot(wn, resid, color="gray", lw=0.9, label="Residual")
            res_ax.axhline(0, color="black", ls="--", lw=0.8, alpha=0.5)
            res_ax.fill_between(wn, resid, alpha=0.25, color="gray")
            for sign in (+1, -1):
                res_ax.axhline(sign * noise, color="#2980b9",
                               ls=":", lw=1.2, alpha=0.85)
            res_ax.axhline(noise, color="#2980b9", ls=":", lw=1.2,
                           alpha=0.85, label=f"±σ = {noise:.4f}")
            res_ax.legend(fontsize=7, loc="upper right")

        row, col = self._sel_row, self._sel_col
        sp_ax.set_title(f"Pixel (row={row}, col={col})", fontsize=9)
        sp_ax.set_ylabel("Intensity", fontsize=9)
        sp_ax.tick_params(labelsize=7, labelbottom=False)
        sp_ax.legend(fontsize=8, loc="upper right")

        res_ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=9)
        res_ax.set_ylabel("Residual", fontsize=8)
        res_ax.tick_params(labelsize=7)

        self._sp_fig.tight_layout(pad=0.8)
        self._sp_canvas.draw_idle()

    def _update_metrics(self, result):
        def _fmt(v) -> str:
            if v is None:
                return "—"
            if isinstance(v, float) and np.isnan(v):
                return "NaN"
            if isinstance(v, float) and np.isinf(v):
                return "∞"
            if isinstance(v, (int, np.integer)):
                return str(int(v))
            return f"{float(v):.4g}"

        if result is None:
            for var in self._m_vars.values():
                var.set("—")
            for item in self._peak_table.get_children():
                self._peak_table.delete(item)
            return

        self._m_vars["n_peaks"].set(_fmt(result["n_peaks"]))
        self._m_vars["noise_std"].set(_fmt(result["noise_std"]))
        self._m_vars["noise_to_peak_ratio"].set(_fmt(result["noise_to_peak_ratio"]))
        self._m_vars["min_separability"].set(_fmt(result["min_separability"]))
        self._m_vars["median_separability"].set(_fmt(result["median_separability"]))
        self._m_vars["mean_separability"].set(_fmt(result["mean_separability"]))

        # Refresh per-peak table
        for item in self._peak_table.get_children():
            self._peak_table.delete(item)
        ctrs = result["peak_centers"]
        amps = result["peak_amplitudes"]
        fwhm = result["peak_fwhm"]
        seps = result["peak_separability"]
        for k in range(result["n_peaks"]):
            sep_str = "∞" if np.isinf(seps[k]) else f"{seps[k]:.3f}"
            self._peak_table.insert("", tk.END, values=(
                f"{ctrs[k]:.2f}",
                f"{amps[k]:.5f}",
                f"{fwhm[k]:.2f}",
                sep_str,
            ))

        self._status.set(
            f"Pixel ({self._sel_row}, {self._sel_col})  —  "
            f"{result['n_peaks']} peaks  |  "
            f"noise σ = {result['noise_std']:.4f}  |  "
            f"noise/peak = {result['noise_to_peak_ratio']:.4f}  |  "
            f"min sep = {_fmt(result['min_separability'])}"
        )


# ── Entry point ───────────────────────────────────────────────────────────────

def launch():
    app = CascadeInspector()
    app.mainloop()


if __name__ == "__main__":
    launch()
