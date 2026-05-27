# fitting_gui.spec — PyInstaller build spec for CASCADE Fitting GUI
#
# Build with:
#   pyinstaller fitting_gui.spec
#
# Output is placed in dist/cascade_fitting_gui/
#
# Prerequisites
# -------------
#   pip install pyinstaller
#   pip install -r requirements.txt   (plus PyTorch for your platform)
#
# Expected output size
# --------------------
#   CPU-only PyTorch : ~700 MB – 1 GB
#   CUDA PyTorch     : 2 – 4 GB  (CUDA DLLs are large)
#   Apple MPS        : ~800 MB
#
# The large size is dominated by PyTorch.  If you need a smaller bundle,
# build with a CPU-only wheel even on a CUDA machine:
#   pip install torch --index-url https://download.pytorch.org/whl/cpu
#
# Notes
# -----
# * torch.compile / Dynamo / Triton are excluded.  fitting_gui.py patches
#   torch.compile to a no-op at runtime when frozen, so all fitting runs in
#   standard eager mode.  Performance is indistinguishable for single-pixel
#   and small-batch fits typical in an interactive GUI.
# * The three CASCADE source files (tidytorch_utils, dataset_utils, plot_utils)
#   are copied into the bundle root so that `import tidytorch_utils` works.
# * lazy5 is an indirect dependency pulled in by dataset_utils at import time.

from PyInstaller.utils.hooks import collect_all, collect_submodules, collect_data_files

block_cipher = None

# ── Collect packages that have native extensions or data files ────────────────
# collect_all returns (datas, binaries, hiddenimports) for each package.

_torch_d,   _torch_b,   _torch_h   = collect_all("torch")
_numpy_d,   _numpy_b,   _numpy_h   = collect_all("numpy")
_mpl_d,     _mpl_b,     _mpl_h     = collect_all("matplotlib")
_h5py_d,    _h5py_b,    _h5py_h    = collect_all("h5py")
_scipy_d,   _scipy_b,   _scipy_h   = collect_all("scipy")
_lazy5_d,   _lazy5_b,   _lazy5_h   = collect_all("lazy5")

a = Analysis(
    ["cascade/fitting_gui.py"],
    pathex=["."],

    binaries=(
        _torch_b + _numpy_b + _mpl_b + _h5py_b + _scipy_b + _lazy5_b
    ),

    datas=(
        _torch_d + _numpy_d + _mpl_d + _h5py_d + _scipy_d + _lazy5_d
        # CASCADE package — copy the whole cascade/ directory into the bundle.
        + [("cascade", "cascade")]
    ),

    hiddenimports=(
        _torch_h + _numpy_h + _mpl_h + _h5py_h + _scipy_h + _lazy5_h
        + [
            # CASCADE package modules (lazy-imported inside functions)
            "cascade",
            "cascade.tidytorch_utils",
            "cascade.dataset_utils",
            "cascade.plot_utils",
            "cascade.fitting_gui",
            "cascade.metrics_gui",

            # lazy5 submodules used by dataset_utils
            "lazy5.inspect",
            "lazy5.create",
            "lazy5.alter",

            # scipy submodules used in CASCADE
            "scipy.optimize",
            "scipy.optimize._lsap",          # linear_sum_assignment C extension
            "scipy.special",
            "scipy.special.wofz",
            "scipy.interpolate",
            "scipy.signal",

            # tkinter (may be missed on some platforms)
            "tkinter",
            "tkinter.ttk",
            "tkinter.filedialog",
            "tkinter.messagebox",
            "_tkinter",

            # matplotlib TkAgg backend
            "matplotlib.backends.backend_tkagg",
            "matplotlib.backends._backend_tk",

            # torch internals that are sometimes missed
            "torch.nn.modules.activation",
            "torch.nn.functional",

            # pkg_resources is used by several packages
            "pkg_resources",
            "pkg_resources.extern",
        ]
    ),

    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],

    excludes=[
        # Triton / Dynamo / Inductor — the JIT compilation stack.
        # fitting_gui.py patches torch.compile to a no-op when frozen,
        # so these are never called and can be stripped to save space.
        "triton",
        "torch._inductor",
        "torch._dynamo",
        "torch.fx",

        # Optional torch extras not used by CASCADE
        "torchvision",
        "torchaudio",
        "torch.distributed",
        "torch.testing",
        "caffe2",

        # Jupyter / IPython (pulled in by some torch versions)
        "IPython",
        "ipykernel",
        "notebook",
        "jupyterlab",
    ],

    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="cascade_fitting_gui",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    # console=True keeps a terminal window alongside the GUI.
    # Useful during testing so you can see tracebacks; set to False for
    # a clean production build (errors will be silently lost on Windows).
    console=True,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="cascade_fitting_gui",
)

# ── macOS app bundle (only active on macOS) ───────────────────────────────────
# Uncomment to also produce a .app:
#
# app = BUNDLE(
#     coll,
#     name="CASCADE Fitting GUI.app",
#     icon=None,
#     bundle_identifier="com.cascade.fitting_gui",
#     info_plist={
#         "NSHighResolutionCapable": True,
#         "CFBundleShortVersionString": "1.0.0",
#     },
# )
