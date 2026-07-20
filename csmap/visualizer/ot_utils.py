"""
visualizer/ot_utils.py — Experimental optimal-transport utilities.

These functions are NOT called by the production analysis notebook
(``scalemap_analysis.ipynb``). They are provided for exploratory use only.

Functions
---------
sinkhorn
barycentric_map_from_plan
sample_points_in_path
infer_square_projector
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from matplotlib import path as mpl_path
from scipy.interpolate import RBFInterpolator


def sample_points_in_path(
    polygon_path: mpl_path.Path,
    n: int = 1000,
    seed: int = 0,
) -> np.ndarray:
    """EXPERIMENTAL — not called by the production analysis notebook.

    Uniform-ish rejection sampling in the polygon's bounding box using
    ``Path.contains_points``.

    Parameters
    ----------
    polygon_path : matplotlib.path.Path
        Closed polygon to sample within.
    n : int, optional
        Number of interior points to return. Default ``1000``.
    seed : int, optional
        Random seed for reproducibility. Default ``0``.

    Returns
    -------
    np.ndarray
        Array of shape ``(n, 2)`` containing 2-D points inside the polygon.

    Raises
    ------
    ValueError
        If no points can be sampled inside the polygon (empty or degenerate path).
    """
    rng = np.random.default_rng(seed)
    verts = polygon_path.vertices
    xmin, ymin = verts.min(axis=0)
    xmax, ymax = verts.max(axis=0)

    pts = []
    tries = 0
    while len(pts) < n and tries < n * 200:
        tries += 1
        batch = rng.uniform([xmin, ymin], [xmax, ymax], size=(max(1024, n // 2), 2))
        mask = polygon_path.contains_points(batch)
        if np.any(mask):
            pts.append(batch[mask])
        if sum(x.shape[0] for x in pts) >= n:
            break
    if not pts:
        raise ValueError(
            "Could not sample any points inside polygon (is it valid / non-empty?)."
        )
    return np.vstack(pts)[:n]


def sinkhorn(
    a: np.ndarray,
    b: np.ndarray,
    C: np.ndarray,
    eps: float = 0.05,
    n_iter: int = 800,
    tol: float = 1e-9,
    max_exp: float = 50.0,
) -> np.ndarray:
    """EXPERIMENTAL — not called by the production analysis notebook.

    Sinkhorn–Knopp algorithm for entropic regularized optimal transport.

    Computes the regularized transport plan ``P`` between source distribution
    ``a`` and target distribution ``b`` with cost matrix ``C``, using
    entropic regularization parameter ``eps``.

    Parameters
    ----------
    a : np.ndarray
        Source histogram of shape ``(n,)``. Must sum to 1.
    b : np.ndarray
        Target histogram of shape ``(m,)``. Must sum to 1.
    C : np.ndarray
        Cost matrix of shape ``(n, m)``.
    eps : float, optional
        Entropic regularization strength. Smaller values → sharper plan,
        higher risk of numerical instability. Default ``0.05``.
    n_iter : int, optional
        Maximum number of Sinkhorn iterations. Default ``800``.
    tol : float, optional
        Convergence tolerance on the scaling vector ``u``. Default ``1e-9``.
    max_exp : float, optional
        Exponent clipping bound to prevent underflow: ``exp`` values are
        clipped to ``[-max_exp, 0]``. Default ``50.0``.

    Returns
    -------
    np.ndarray
        Transport plan matrix of shape ``(n, m)``.
    """
    # K = exp(-C/eps) but clip exponent to avoid underflow to 0
    E = -C / eps
    E = np.clip(E, -max_exp, 0.0)
    K = np.exp(E)

    u = np.ones_like(a)
    v = np.ones_like(b)
    for it in range(n_iter):
        u_prev = u
        u = a / (K @ v + 1e-300)
        v = b / (K.T @ u + 1e-300)
        if it % 50 == 0 and np.max(np.abs(u - u_prev)) < tol:
            break
    return (u[:, None] * K) * v[None, :]


def barycentric_map_from_plan(P: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """EXPERIMENTAL — not called by the production analysis notebook.

    Compute the barycentric projection of a transport plan onto target points.

    For each source point ``i``, the barycentric projection is the weighted
    average of the target points ``Y`` with weights given by row ``i`` of the
    transport plan ``P``.

    Parameters
    ----------
    P : np.ndarray
        Transport plan of shape ``(n, m)`` as returned by :func:`sinkhorn`.
    Y : np.ndarray
        Target point cloud of shape ``(m, d)``.

    Returns
    -------
    np.ndarray
        Mapped source points of shape ``(n, d)``, each being the weighted
        barycentre of the target points under the transport plan row.
    """
    mass = P.sum(axis=1, keepdims=True) + 1e-300
    return (P @ Y) / mass


def infer_square_projector(
    polygon_path: mpl_path.Path,
    *,
    training_points: Optional[np.ndarray] = None,
    N: int = 400,
    seed_train: int = 1,
    seed_target: int = 2,
    sinkhorn_eps: float = 0.05,
    sinkhorn_iters: int = 800,
    sinkhorn_tol: float = 1e-9,
    rbf_kernel: str = "thin_plate_spline",
    rbf_smoothing: float = 1e-6,
):
    """EXPERIMENTAL — not called by the production analysis notebook.

    Build a callable ``f(points) -> points_in_[0,1]^2`` that approximately
    equalizes density and fills the square using entropic OT (Sinkhorn) +
    barycentric projection + smooth RBF.

    Usage::

        sq_projector = infer_square_projector(polygon_path)
        embedding_sq_projection = sq_projector(embedding)

    Notes
    -----
    - Not guaranteed bijective.
    - ``training_points``, if provided, must be inside ``polygon_path``.

    Parameters
    ----------
    polygon_path : matplotlib.path.Path
        Closed polygon defining the region of interest (e.g., the UMAP
        embedding boundary).
    training_points : np.ndarray or None, optional
        Pre-sampled points of shape ``(N, 2)`` inside ``polygon_path``.
        If ``None``, points are sampled by rejection sampling. Default ``None``.
    N : int, optional
        Number of training / target points for OT. Default ``400``.
    seed_train : int, optional
        Random seed for training-point sampling. Default ``1``.
    seed_target : int, optional
        Random seed for target jittered grid. Default ``2``.
    sinkhorn_eps : float, optional
        Entropic regularization for Sinkhorn. Default ``0.05``.
    sinkhorn_iters : int, optional
        Maximum Sinkhorn iterations. Default ``800``.
    sinkhorn_tol : float, optional
        Convergence tolerance for Sinkhorn. Default ``1e-9``.
    rbf_kernel : str, optional
        RBF kernel type passed to ``scipy.interpolate.RBFInterpolator``.
        Default ``'thin_plate_spline'``.
    rbf_smoothing : float, optional
        Smoothing factor for the RBF interpolator. Default ``1e-6``.

    Returns
    -------
    callable
        A function ``projector(points: np.ndarray) -> np.ndarray`` that maps
        an array of shape ``(n, 2)`` (or ``(2,)``) to ``[0, 1]^2``.
    """
    # --- normalize coordinates to stabilize Sinkhorn / RBF ---
    verts = polygon_path.vertices
    vmin = verts.min(axis=0)
    vmax = verts.max(axis=0)
    scale = vmax - vmin
    scale = np.where(scale == 0, 1.0, scale)  # avoid divide-by-zero

    def norm_xy(P: np.ndarray) -> np.ndarray:
        return (P - vmin) / scale

    if training_points is None:
        X = sample_points_in_path(polygon_path, n=N, seed=seed_train)
    else:
        X = np.asarray(training_points, dtype=float)
        if X.ndim != 2 or X.shape[1] != 2:
            raise ValueError("training_points must be an array of shape (N, 2).")
        N = X.shape[0]
        inside = polygon_path.contains_points(X)
        if not bool(np.all(inside)):
            raise ValueError("Some training_points are outside polygon_path.")

    Xn = norm_xy(X)  # normalized training coords in ~[0,1]^2

    # target points in square: jittered grid (ensure >= N points)
    m = int(np.ceil(np.sqrt(N)))
    rng = np.random.default_rng(seed_target)
    gx, gy = np.meshgrid((np.arange(m) + 0.5) / m, (np.arange(m) + 0.5) / m)
    Y = np.c_[gx.ravel(), gy.ravel()][:N]
    Y += rng.normal(scale=0.20 / m, size=Y.shape)  # jitter
    Y = np.clip(Y, 0, 1)

    # OT weights (uniform)
    a = np.ones(N) / N
    b = np.ones(N) / N

    # cost matrix (squared euclidean) in normalized space
    C = ((Xn[:, None, :] - Y[None, :, :]) ** 2).sum(axis=2)

    # solve entropic OT, then barycentric map
    P = sinkhorn(a, b, C, eps=sinkhorn_eps, n_iter=sinkhorn_iters, tol=sinkhorn_tol)
    T = barycentric_map_from_plan(P, Y)

    # Smooth extension to a map defined everywhere in the polygon (in normalized space)
    f_rbf = RBFInterpolator(Xn, T, kernel=rbf_kernel, smoothing=rbf_smoothing)

    def projector(points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=float)
        if pts.ndim == 1:
            pts = pts[None, :]
        if pts.shape[1] != 2:
            raise ValueError("points must be shape (n,2) or (2,).")
        ptsn = norm_xy(pts)
        out = f_rbf(ptsn)
        out = np.clip(out, 0, 1)
        return out

    return projector
