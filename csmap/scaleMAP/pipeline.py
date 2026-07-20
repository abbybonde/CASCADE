"""
scaleMAP/pipeline.py — UMAP pipeline with dill-based file cache and
cross-image reference bundles.

Running a UMAP fit on a large hyperspectral image can take several minutes.
This module wraps the fit with a simple file cache so that notebook re-execution
reloads a previously computed result instead of recomputing it.

``dill`` is used instead of ``pickle`` because the UMAP model object may contain
lambda closures and numba-compiled metric functions that the standard ``pickle``
module cannot serialize.

Cross-image embedding (reference bundles)
-----------------------------------------
:func:`fit_reference` fits the first ("reference") image and persists a
self-describing bundle — fitted model/descriptors + normalization stats +
canonical grid + ``amp_threshold`` + ``random_state`` + UMAP hyperparameters.
:func:`transform_into_reference` loads a bundle and places a **new** image's
normalized descriptors into that fitted frame without refitting the frame.

Two transform backends are supported (``transform_backend``):

* ``'parametric'`` — :class:`ParametricUMAP` learns an explicit neural-network
  encoder during the reference fit, so **arbitrary out-of-sample pixels** can
  later be pushed through the fixed encoder.  Best when many future images
  must be projected into one frozen map.  Requires TensorFlow.
* ``'aligned'`` — :class:`AlignedUMAP` (+ :func:`procrustes_align`) jointly
  optimizes the reference and the new image with anchor relations, then
  rigidly aligns the joint frame onto the persisted reference embedding.
  Best for jointly aligning a known set of images; no TensorFlow needed, but
  each transform re-runs an (aligned) optimization.
"""

from __future__ import annotations

import os

import dill
import numpy as np

from .umap_ import UMAP

#: Bump when the bundle schema changes; mismatched bundles refuse to load.
BUNDLE_FORMAT_VERSION = 1

_BUNDLE_ROLES = ("reference",)
_TRANSFORM_BACKENDS = ("parametric", "aligned")


def run_scalemap_pipeline(
    X: np.ndarray,
    savepath: str,
    umap_kwargs: dict,
) -> tuple[np.ndarray, object]:
    """Run a UMAP fit with a dill-based file cache.

    On the **first** call for a given *savepath* (cache miss), this function
    fits a :class:`UMAP` model to *X*, serialises the result to *savepath*
    using :func:`dill.dump`, and returns the result.

    On subsequent calls where *savepath* already exists (cache hit), the
    previously serialised ``(embedding_, fitted_model)`` tuple is loaded from
    disk with :func:`dill.load` and returned immediately — no fitting occurs.

    Parameters
    ----------
    X:
        Feature matrix of shape ``(n_pixels, n_features)``.  Must be a 2-D
        NumPy array compatible with the UMAP implementation in this package.
    savepath:
        Filesystem path where the serialised result is written on a cache miss
        and read on a cache hit.  Typically derived from ``SAVEFILE_BASENAME``
        and ``ANALYSIS_MODE`` in the analysis notebook, e.g.
        ``f"{SAVEFILE_BASENAME}_{ANALYSIS_MODE}_full.dill"``.
    umap_kwargs:
        Keyword arguments forwarded verbatim to :class:`UMAP`.  Common keys
        include ``n_components``, ``n_neighbors``, ``min_dist``,
        ``n_epochs``, ``densmap``, ``dens_frac``, ``metric``, and
        ``metric_kwds``.

    Returns
    -------
    tuple[numpy.ndarray, object]
        A two-element tuple ``(embedding_, fitted_model)`` where:

        * ``embedding_`` — NumPy array of shape ``(n_pixels, n_components)``
          containing the low-dimensional embedding coordinates.
        * ``fitted_model`` — the fitted :class:`UMAP` instance, which can be
          used later for :meth:`transform` calls on new data.

    Notes
    -----
    * ``dill`` is used exclusively — never ``pickle``.  This is required
      because the UMAP model may hold references to numba-compiled functions
      or lambda closures that ``pickle`` cannot serialise.
    * No global state is modified; all results are returned explicitly.
    * The cache is keyed solely by *savepath*.  The caller is responsible for
      ensuring that different parameter combinations are mapped to different
      paths (e.g. by including ``ANALYSIS_MODE`` in the filename).
    * If the process is interrupted during the write step, a partial file may
      be left at *savepath*.  Delete the file manually before re-running to
      force a fresh fit.
    """
    if os.path.exists(savepath):
        # Cache hit — reload from disk, but only trust it if it was fit on
        # the same pixels.  The cache is keyed by path alone, so re-running
        # with a different background mask used to hand back a stale
        # embedding with the wrong number of rows (e.g. a 189185-row
        # embedding for a 29465-row X_masked).
        with open(savepath, "rb") as f:
            embedding_, fitted_model = dill.load(f)
        if embedding_.shape[0] == X.shape[0]:
            return embedding_, fitted_model
        print(
            f"Cache '{savepath}' holds an embedding for "
            f"{embedding_.shape[0]} pixels but the current input has "
            f"{X.shape[0]} — the mask changed. Refitting and overwriting "
            "the cache."
        )

    # Cache miss (or stale cache) — fit the model and persist the result.
    fitted_model = UMAP(**umap_kwargs).fit(X)
    embedding_: np.ndarray = fitted_model.embedding_

    with open(savepath, "wb") as f:
        dill.dump((embedding_, fitted_model), f)

    return embedding_, fitted_model


def make_cache_path(
    basename: str,
    analysis_mode: str,
    role: str,
    suffix: str = "full",
) -> str:
    """Build a cache path that encodes analysis mode **and** embedding role.

    Encoding the role (``'reference'`` vs ``'transformed'``, or
    ``'standalone'`` for the spectral path) guarantees that a reference fit
    and a transform of the same image never collide in the cache.

    Example: ``make_cache_path('run1', 'peak_params', 'reference')`` →
    ``'run1_peak_params_reference_full.dill'``.
    """
    return f"{basename}_{analysis_mode}_{role}_{suffix}.dill"


def _require_random_state(umap_kwargs: dict) -> int:
    """Return the seed from *umap_kwargs*, raising if it is missing.

    Every UMAP construction in the reference/transform pipeline must be
    seeded so that repeated fits on identical input are identical.
    """
    random_state = umap_kwargs.get("random_state")
    if random_state is None:
        raise ValueError(
            "umap_kwargs must include an explicit integer 'random_state' "
            "(config RANDOM_STATE) so embeddings are reproducible."
        )
    return int(random_state)


def fit_reference(
    X: np.ndarray,
    bundle_path: str,
    umap_kwargs: dict,
    *,
    grid: np.ndarray,
    amp_threshold: float,
    norm_stats: dict,
    transform_backend: str = "aligned",
    embedding_cache_path: str | None = None,
) -> tuple[np.ndarray, dict]:
    """Fit the reference image and persist a self-describing reference bundle.

    Parameters
    ----------
    X:
        Normalized peak-descriptor matrix of the reference image, shape
        ``(n_pixels, len(grid))`` (from :func:`build_peak_descriptor` +
        :func:`normalize_peak_descriptor`).
    bundle_path:
        Where the dill reference bundle is written.  If the file already
        exists it is loaded, validated against *grid*/*amp_threshold*, and
        returned without refitting.
    umap_kwargs:
        UMAP hyperparameters.  **Must** contain an explicit integer
        ``random_state``.
    grid, amp_threshold, norm_stats:
        The canonical descriptor grid, amplitude threshold, and the
        normalization stats returned by :func:`normalize_peak_descriptor` for
        this reference — persisted so every later image can be processed
        identically and mismatches can be detected at load time.
    transform_backend:
        ``'parametric'`` or ``'aligned'`` (see module docstring for when to
        use each).
    embedding_cache_path:
        Optional dill cache for the reference embedding itself (aligned
        backend only), following :func:`run_scalemap_pipeline` semantics.

    Returns
    -------
    tuple[numpy.ndarray, dict]
        ``(embedding, bundle)`` — the reference embedding of shape
        ``(n_pixels, n_components)`` and the persisted bundle dict.
    """
    if transform_backend not in _TRANSFORM_BACKENDS:
        raise ValueError(
            f"Invalid transform_backend '{transform_backend}'. "
            f"Must be one of {_TRANSFORM_BACKENDS}."
        )
    _require_random_state(umap_kwargs)
    grid = np.asarray(grid, dtype=np.float64)

    if os.path.exists(bundle_path):
        bundle = load_reference_bundle(
            bundle_path, expected_grid=grid, expected_amp_threshold=amp_threshold
        )
        if bundle["transform_backend"] != transform_backend:
            raise ValueError(
                f"Reference bundle '{bundle_path}' was fitted with "
                f"transform_backend='{bundle['transform_backend']}' but the "
                f"current config requests '{transform_backend}'. Delete the "
                "bundle or change TRANSFORM_BACKEND."
            )
        if bundle["embedding"].shape[0] != X.shape[0]:
            raise ValueError(
                f"Reference bundle '{bundle_path}' was fitted on "
                f"{bundle['embedding'].shape[0]} pixels but the current input "
                f"has {X.shape[0]} (the mask or foreground selection "
                "changed). Delete the bundle to refit, or point "
                "REFERENCE_BUNDLE_PATH at the correct reference."
            )
        return bundle["embedding"], bundle

    bundle: dict = {
        "format_version": BUNDLE_FORMAT_VERSION,
        "role": "reference",
        "transform_backend": transform_backend,
        "grid": grid,
        "amp_threshold": float(amp_threshold),
        "norm_stats": dict(norm_stats),
        "random_state": _require_random_state(umap_kwargs),
        "umap_kwargs": dict(umap_kwargs),
        "parametric_model_dir": None,
        "descriptors": None,
    }

    if transform_backend == "parametric":
        # Lazy import: scaleMAP.parametric_umap requires TensorFlow, which is
        # only needed for this backend.
        from .parametric_umap import ParametricUMAP

        model = ParametricUMAP(**umap_kwargs)
        model.fit(X)
        embedding = np.asarray(model.embedding_)
        # Keras encoders cannot round-trip through dill; use the class's own
        # save format in a sibling directory referenced from the bundle.
        model_dir = bundle_path + ".parametric_model"
        model.save(model_dir)
        bundle["parametric_model_dir"] = model_dir
    else:
        # Aligned backend: the reference embedding is a plain (seeded) UMAP
        # fit; the descriptors are persisted because every later transform
        # jointly re-optimizes them alongside the new image.
        if embedding_cache_path is not None:
            embedding, _model = run_scalemap_pipeline(
                X, embedding_cache_path, umap_kwargs
            )
        else:
            embedding = UMAP(**umap_kwargs).fit(X).embedding_
        bundle["descriptors"] = np.asarray(X, dtype=np.float32)

    bundle["embedding"] = np.asarray(embedding)

    with open(bundle_path, "wb") as f:
        dill.dump(bundle, f)

    return bundle["embedding"], bundle


def load_reference_bundle(
    bundle_path: str,
    *,
    expected_grid: np.ndarray,
    expected_amp_threshold: float,
) -> dict:
    """Load a reference bundle, refusing anything that disagrees with config.

    A bundle whose ``format_version``, ``grid``, or ``amp_threshold`` does
    not match the caller's current configuration **raises**
    :exc:`ValueError` — silently proceeding would put two images that were
    described differently into the same map.
    """
    with open(bundle_path, "rb") as f:
        bundle = dill.load(f)

    if not isinstance(bundle, dict) or "format_version" not in bundle:
        raise ValueError(
            f"'{bundle_path}' is not a ScaleMAP reference bundle "
            "(missing format_version)."
        )
    if bundle["format_version"] != BUNDLE_FORMAT_VERSION:
        raise ValueError(
            f"Reference bundle '{bundle_path}' has format_version "
            f"{bundle['format_version']}; this code expects "
            f"{BUNDLE_FORMAT_VERSION}."
        )
    if bundle.get("role") not in _BUNDLE_ROLES:
        raise ValueError(
            f"'{bundle_path}' has role '{bundle.get('role')}'; expected a "
            "reference bundle."
        )

    expected_grid = np.asarray(expected_grid, dtype=np.float64)
    grid = np.asarray(bundle["grid"], dtype=np.float64)
    if grid.shape != expected_grid.shape or not np.allclose(grid, expected_grid):
        raise ValueError(
            f"Reference bundle '{bundle_path}' was built on a different "
            f"descriptor grid ({grid.size} channels, "
            f"[{grid.min():g}, {grid.max():g}]) than the current config "
            f"({expected_grid.size} channels, "
            f"[{expected_grid.min():g}, {expected_grid.max():g}]). "
            "Descriptors are only comparable on an identical grid."
        )
    if float(bundle["amp_threshold"]) != float(expected_amp_threshold):
        raise ValueError(
            f"Reference bundle '{bundle_path}' used amp_threshold="
            f"{bundle['amp_threshold']:g} but the current config uses "
            f"{expected_amp_threshold:g}."
        )

    return bundle


def _mutual_nn_relations(
    X_ref: np.ndarray,
    X_new: np.ndarray,
    metric: str,
) -> dict[int, int]:
    """Anchor relations for AlignedUMAP: mutual nearest neighbors.

    Maps reference-pixel index → new-pixel index for every pair that are
    each other's nearest neighbor in descriptor space.  Mutuality keeps
    one-sided (unreliable) matches from anchoring the alignment.
    """
    from sklearn.neighbors import NearestNeighbors

    nn_new = NearestNeighbors(n_neighbors=1, metric=metric).fit(X_new)
    ref_to_new = nn_new.kneighbors(X_ref, return_distance=False)[:, 0]
    nn_ref = NearestNeighbors(n_neighbors=1, metric=metric).fit(X_ref)
    new_to_ref = nn_ref.kneighbors(X_new, return_distance=False)[:, 0]

    return {
        int(i): int(j)
        for i, j in enumerate(ref_to_new)
        if new_to_ref[j] == i
    }


# AlignedUMAP.__init__ accepts a narrower set of kwargs than UMAP; anything
# else (densmap, verbose flags, etc.) is dropped when building the joint fit.
_ALIGNED_UMAP_KWARGS = (
    "n_neighbors", "n_components", "metric", "metric_kwds", "n_epochs",
    "learning_rate", "init", "min_dist", "spread", "low_memory",
    "set_op_mix_ratio", "local_connectivity", "repulsion_strength",
    "negative_sample_rate", "transform_queue_size", "a", "b", "random_state",
    "angular_rp_forest", "transform_seed", "verbose", "unique",
)


def transform_into_reference(
    X_new: np.ndarray,
    bundle_path: str,
    *,
    grid: np.ndarray,
    amp_threshold: float,
    cache_path: str | None = None,
) -> np.ndarray:
    """Place a new image's normalized descriptors into a reference frame.

    Loads the reference bundle at *bundle_path* (validating it against the
    caller's *grid* and *amp_threshold* — mismatches raise) and transforms
    *X_new* into the persisted reference embedding's coordinate system
    **without refitting the reference frame**.

    Backend behaviour (recorded in the bundle at fit time):

    * ``'parametric'`` — the persisted encoder network maps *X_new* directly;
      the reference frame is exactly frozen.
    * ``'aligned'`` — a joint :class:`AlignedUMAP` optimization of
      ``[reference descriptors, X_new]`` with mutual-nearest-neighbor anchor
      relations, after which the joint frame is rigidly rotated (via
      :func:`procrustes_align`, with centering and uniform scale matching)
      onto the persisted reference embedding.

    Parameters
    ----------
    X_new:
        Normalized descriptor matrix of the new image, shape
        ``(n_pixels_new, len(grid))``.  Must be normalized with the
        reference's persisted ``norm_stats``
        (``normalize_peak_descriptor(..., mode='reference', stats=...)``).
    bundle_path:
        Path of a bundle written by :func:`fit_reference`.
    grid, amp_threshold:
        Current config values; must match the bundle or a
        :exc:`ValueError` is raised.
    cache_path:
        Optional dill cache for the transformed embedding.  Use a path that
        encodes the ``transformed`` role (see :func:`make_cache_path`) so it
        can never collide with a reference fit.

    Returns
    -------
    numpy.ndarray
        Embedding of *X_new* in the reference frame, shape
        ``(n_pixels_new, n_components)``.
    """
    if cache_path is not None and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cached = dill.load(f)
        if cached.shape[0] == np.asarray(X_new).shape[0]:
            return cached
        print(
            f"Cache '{cache_path}' holds a transform for {cached.shape[0]} "
            f"pixels but the current input has {np.asarray(X_new).shape[0]} "
            "— recomputing and overwriting the cache."
        )

    bundle = load_reference_bundle(
        bundle_path, expected_grid=grid, expected_amp_threshold=amp_threshold
    )
    X_new = np.asarray(X_new, dtype=np.float32)
    if X_new.ndim != 2 or X_new.shape[1] != np.asarray(bundle["grid"]).size:
        raise ValueError(
            f"X_new has shape {X_new.shape}; expected "
            f"(n_pixels, {np.asarray(bundle['grid']).size}) to match the "
            "reference descriptor grid."
        )

    if bundle["transform_backend"] == "parametric":
        from .parametric_umap import load_ParametricUMAP

        model = load_ParametricUMAP(bundle["parametric_model_dir"])
        embedding_new = np.asarray(model.transform(X_new))
    else:
        embedding_new = _aligned_transform(X_new, bundle)

    if cache_path is not None:
        with open(cache_path, "wb") as f:
            dill.dump(embedding_new, f)

    return embedding_new


def _aligned_transform(X_new: np.ndarray, bundle: dict) -> np.ndarray:
    """AlignedUMAP joint fit + Procrustes onto the persisted reference frame."""
    from .aligned_umap import AlignedUMAP, procrustes_align

    X_ref = np.asarray(bundle["descriptors"], dtype=np.float32)
    E_ref = np.asarray(bundle["embedding"], dtype=np.float64)
    umap_kwargs = bundle["umap_kwargs"]

    metric = umap_kwargs.get("metric", "euclidean")
    relations = _mutual_nn_relations(X_ref, X_new, metric)
    if len(relations) < 3:
        raise ValueError(
            f"Only {len(relations)} mutual-nearest-neighbor anchors found "
            "between the new image and the reference; the images may not "
            "share chemistry, or normalization is inconsistent."
        )

    aligned_kwargs = {
        k: v for k, v in umap_kwargs.items() if k in _ALIGNED_UMAP_KWARGS
    }
    mapper = AlignedUMAP(**aligned_kwargs)
    mapper.fit([X_ref, X_new], relations=[relations])
    joint_ref = np.asarray(mapper.embeddings_[0], dtype=np.float64)
    joint_new = np.asarray(mapper.embeddings_[1], dtype=np.float64)

    # Rigidly carry the joint frame onto the persisted reference frame:
    # center both, match overall scale, then rotate with procrustes_align
    # using the exact refit-reference ↔ stored-reference correspondence.
    finite_ref = np.isfinite(joint_ref).all(axis=1) & np.isfinite(E_ref).all(axis=1)
    anchors_idx = np.flatnonzero(finite_ref)
    mu_stored = E_ref[anchors_idx].mean(axis=0)
    mu_joint = joint_ref[anchors_idx].mean(axis=0)

    stored_c = E_ref - mu_stored
    joint_ref_c = joint_ref - mu_joint
    joint_new_c = joint_new - mu_joint

    scale_stored = np.linalg.norm(stored_c[anchors_idx])
    scale_joint = np.linalg.norm(joint_ref_c[anchors_idx])
    if scale_joint > 0.0:
        s = scale_stored / scale_joint
        joint_ref_c = joint_ref_c * s
        joint_new_c = joint_new_c * s

    n_ref = joint_ref_c.shape[0]
    stacked = np.ascontiguousarray(
        np.vstack([joint_ref_c, joint_new_c]).astype(np.float64)
    )
    anchors = np.vstack([anchors_idx, anchors_idx])
    rotated = procrustes_align(
        np.ascontiguousarray(stored_c), stacked, anchors
    )

    return rotated[n_ref:] + mu_stored
