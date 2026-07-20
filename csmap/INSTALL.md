# ScaleMAP — Installation & Environment Setup

ScaleMAP runs UMAP-based dimensionality reduction on hyperspectral Raman image data.
This guide gets you from a fresh machine to a working JupyterLab session in about
10 minutes.

---

## Prerequisites

| Tool | Minimum version | Notes |
|------|----------------|-------|
| [conda / miniforge](https://github.com/conda-forge/miniforge) | any recent | miniforge recommended (conda-forge default) |
| git | any | to clone the repo |
| macOS / Linux | — | Windows untested |

---

## 1 — Clone the repository

```bash
git clone <repo-url> csMAP
cd csMAP
```

---

## 2 — Create the conda environment

```bash
conda env create -f environment.yml
```

This creates an environment named **`scalemap`** and installs all dependencies
including the `scaleMAP` and `visualizer` packages in editable mode
(`pip install -e .`), so any source edits are picked up immediately without
reinstalling.

> **Numba / Python 3.14 note**
> numba ≥ 0.65 ships Python 3.14 wheels on conda-forge.
> If the solver cannot find a compatible wheel at time of install, pin a
> pre-release build in `environment.yml`:
> ```yaml
> - numba>=0.65.0rc1
> ```
> or build numba from source — see the comment block in `environment.yml`.

---

## 3 — Activate and verify

```bash
conda activate scalemap

# smoke test — should print nothing (no import errors)
python -c "
import holoviews, panel, bokeh, datashader
import umap, numba, numpy, scipy, sklearn, matplotlib, skimage
from scaleMAP import UMAP
from scaleMAP.loader import load_scalemap_h5
from scaleMAP.preprocessing import spectral_concat, zscore_normalize, apply_background_mask
from scaleMAP.pipeline import run_scalemap_pipeline
from visualizer.visualizer import generate_freehand_overlay_spectrum_2, flatten_image
from visualizer.colormaps import recolor_image_with_umap
from visualizer.clustering import kmeanscluster, plot_kmeans_clusters
print('All imports OK')
"
```

---

## 4 — Launch JupyterLab

```bash
jupyter lab scalemap_analysis.ipynb
```

Open `scalemap_analysis.ipynb` and run cells top-to-bottom.
Set `H5_FILE_PATH` in Cell 1 to point at your CASCADE HDF5 output file.

---

## 5 — Run the test suite (optional)

```bash
pytest tests/ -v
```

All property-based tests use [Hypothesis](https://hypothesis.readthedocs.io).
A full run takes roughly 2–5 minutes depending on hardware.

---

## Directory structure

```
csMAP/
├── environment.yml            ← conda env spec (primary install path)
├── pyproject.toml             ← editable pip install config
├── INSTALL.md                 ← this file
├── scalemap_analysis.ipynb    ← 13-cell analysis notebook (canonical entry point)
├── scaleMAP/
│   ├── __init__.py            ← importlib.metadata version detection
│   ├── distances.py           ← numba distance functions + chamfer_peak_distance
│   ├── layouts.py             ← SGD layout kernels (static @njit decoration)
│   ├── loader.py              ← CASCADE HDF5 loader
│   ├── preprocessing.py       ← spectral_concat, zscore_normalize, apply_background_mask
│   ├── pipeline.py            ← run_scalemap_pipeline (dill cache)
│   └── umap_.py               ← UMAP core
├── visualizer/
│   ├── __init__.py
│   ├── visualizer.py          ← generate_freehand_overlay_spectrum_2, DispersionStream
│   ├── colormaps.py           ← recolor_image_with_umap (canonical)
│   ├── clustering.py          ← kmeanscluster, plot_kmeans_clusters
│   ├── metrics.py             ← calculate_r2
│   ├── ot_utils.py            ← experimental OT utilities
│   └── visualizer_environment.yml  ← legacy env file (kept for compatibility)
└── tests/
    ├── test_visualizer_pure.py   ← Properties 1–10
    ├── test_loader.py            ← Properties 11–12 + error paths
    ├── test_chamfer.py           ← Properties 13–15 + edge cases
    ├── test_preprocessing.py     ← Properties 16–18 + edge cases
    ├── test_pipeline.py          ← Property 19 + cache behavior
    ├── test_visualizer_peaks.py  ← Property 20 + peak stem overlay
    └── test_smoke.py             ← import smoke tests + API contracts
```

---

## Updating the environment

To add a package without recreating from scratch:

```bash
conda activate scalemap
pip install <package>          # for pip-only packages
conda install -c conda-forge <package>  # for conda packages
```

To fully recreate after editing `environment.yml`:

```bash
conda env remove -n scalemap
conda env create -f environment.yml
```

---

## Analysis modes

| `ANALYSIS_MODE` | Input to UMAP | Distance metric |
|---|---|---|
| `'spectral'` | Concatenated & z-scored spectral windows | Euclidean |
| `'peak_params'` | Fitted peak-parameter cube (amp, center, σ, γ) | `chamfer_peak_distance` |

Set `ANALYSIS_MODE` in Cell 1 of `scalemap_analysis.ipynb` before running.
