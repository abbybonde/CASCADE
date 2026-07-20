"""
visualizer/clustering.py — KMeans clustering utilities for ScaleMAP visualizer.

Provides scikit-learn KMeans wrapping and a matplotlib figure factory for
displaying cluster labels as a categorical image.
"""

from __future__ import annotations

import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans


def kmeanscluster(
    X: np.ndarray,
    n_clusters: int,
    random_state: int = 0,
    n_init: int | str = 10,
) -> np.ndarray:
    """Wrap sklearn KMeans. Returns 1D integer label array of shape (n_pixels,).

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape ``(n_pixels, n_features)``.
    n_clusters : int
        Number of clusters to form.
    random_state : int, optional
        Random seed for reproducibility. Default ``0``.
    n_init : int or 'auto', optional
        Number of KMeans restarts; the best inertia wins.  Default ``10``.

    Returns
    -------
    np.ndarray
        Integer label array of shape ``(n_pixels,)`` with values in
        ``[0, n_clusters - 1]``.
    """
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    labels: np.ndarray = km.fit_predict(X)
    return labels.astype(int)


def cluster_colors(n_clusters: int) -> list:
    """Consistent per-cluster RGBA colors (tab10/tab20) used by the cluster
    image and any per-cluster spectra plots."""
    base = plt.get_cmap("tab10" if n_clusters <= 10 else "tab20")
    return [base(i % base.N) for i in range(n_clusters)]


def plot_kmeans_clusters(
    labels: np.ndarray,
    rows: int,
    cols: int,
    background_label: int = -1,
    background_color: str = "0.85",
    **plot_kwargs,
) -> matplotlib.figure.Figure:
    """Reshape labels to (rows, cols), display as categorical image. Returns Figure.

    Exactly one color per cluster (from :func:`cluster_colors`), with a
    discrete colorbar — so an ``n_clusters`` run can never *display* more
    than ``n_clusters`` colors.  Pixels carrying *background_label* (e.g.
    ``-1`` for background excluded from the analysis) are rendered in
    *background_color* and omitted from the colorbar.

    Parameters
    ----------
    labels : np.ndarray
        1D integer label array of shape ``(n_pixels,)`` as returned by
        :func:`kmeanscluster`, optionally containing *background_label*.
    rows, cols : int
        Spatial image grid shape; ``len(labels)`` must equal ``rows * cols``.
    background_label : int
        Label value treated as background. Default ``-1``.
    background_color : matplotlib color
        Fill for background pixels. Default light gray.
    **plot_kwargs
        Additional keyword arguments forwarded to :func:`matplotlib.pyplot.imshow`.

    Raises
    ------
    ValueError
        If ``len(labels) != rows * cols``.
    """
    n_pixels = rows * cols
    if len(labels) != n_pixels:
        raise ValueError(
            f"len(labels)={len(labels)} does not match rows*cols={n_pixels}."
        )

    labels = np.asarray(labels).reshape(rows, cols)
    foreground = labels != background_label
    cluster_ids = np.unique(labels[foreground]) if foreground.any() else np.array([0])
    n_clusters = int(cluster_ids.max()) + 1

    cmap = matplotlib.colors.ListedColormap(cluster_colors(n_clusters))
    cmap.set_bad(background_color)
    norm = matplotlib.colors.BoundaryNorm(
        np.arange(-0.5, n_clusters + 0.5), cmap.N
    )

    label_image = np.ma.masked_where(~foreground, labels)

    imshow_kwargs: dict = {"interpolation": "nearest"}
    imshow_kwargs.update(plot_kwargs)

    fig, ax = plt.subplots()
    im = ax.imshow(label_image, cmap=cmap, norm=norm, **imshow_kwargs)
    cbar = fig.colorbar(im, ax=ax, label="Cluster", ticks=np.arange(n_clusters))
    cbar.ax.set_yticklabels([str(i) for i in range(n_clusters)])
    ax.set_title("KMeans Clusters")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    return fig

def generate_cluster_explorer(
    feature_sources: dict,
    model_cube: np.ndarray,
    x_axis: np.ndarray | None,
    pixel_indices: np.ndarray,
    n_clusters: int = 5,
    random_state: int = 0,
    width: int = 500,
    height: int = 420,
):
    """Interactive KMeans explorer: pick features, pick k, re-run live.

    Widgets: a feature-source selector (e.g. peak ``descriptor`` vs. z-scored
    ``model_cube`` spectra vs. 2-D ``embedding``), a cluster-count slider,
    and a **Run clustering** button.  Each run shows

    * the spatial cluster map (one discrete color per cluster, excluded
      pixels transparent), and
    * a stacked ("waterfall") plot of each cluster's mean ``model_cube``
      spectrum in its cluster color, vertically offset so the spectra sit
      on top of each other for comparison.

    Uses :class:`~sklearn.cluster.MiniBatchKMeans` so re-runs are fast
    enough to iterate on interactively (the static notebook cell keeps full
    KMeans for the publication run).

    Parameters
    ----------
    feature_sources:
        Mapping ``name -> (n_retained, n_features)`` feature matrix; rows
        must align with *pixel_indices*.
    model_cube:
        Full spectral cube ``(rows, cols, W)`` used for the mean-spectrum
        display (regardless of which features drive the clustering).
    x_axis:
        Wavenumber axis for the spectrum plot, or ``None`` for channels.
    pixel_indices:
        Flat raster-order indices of the retained (foreground, masked)
        pixels that the feature rows describe.
    n_clusters:
        Initial cluster-count slider value.
    """
    import holoviews as hv
    import panel as pn

    rows, cols = model_cube.shape[:2]
    spec_x = np.asarray(x_axis) if x_axis is not None else np.arange(model_cube.shape[2])
    flat_cube = model_cube.reshape(-1, model_cube.shape[2])

    feature_select = pn.widgets.Select(
        name='Cluster on', options=list(feature_sources), width=180)
    k_slider = pn.widgets.IntSlider(
        name='N clusters', start=2, end=15, value=int(n_clusters), width=220)
    run_button = pn.widgets.Button(name='Run clustering', button_type='primary',
                                   width=140)
    status = pn.widgets.StaticText(value='')
    plot_pane = pn.pane.HoloViews(sizing_mode='fixed')

    def _cluster_layout(feature_name, k):
        X = feature_sources[feature_name]
        km = MiniBatchKMeans(n_clusters=k, random_state=random_state,
                             n_init=5, batch_size=4096)
        labels = km.fit_predict(X)

        colors = [matplotlib.colors.to_hex(c) for c in cluster_colors(k)]

        label_img = np.full(rows * cols, np.nan)
        label_img[pixel_indices] = labels
        cluster_map = hv.Image(
            label_img.reshape(rows, cols), bounds=(0, 0, cols, rows),
            kdims=['x', 'y'], vdims=['cluster'],
        ).opts(
            width=width, height=height, cmap=colors, colorbar=True,
            clim=(-0.5, k - 0.5), color_levels=k,
            title=f'{k} clusters on {feature_name}',
        )

        means = []
        for j in range(k):
            sel = pixel_indices[labels == j]
            means.append(flat_cube[sel].mean(axis=0) if len(sel)
                         else np.zeros(model_cube.shape[2]))
        offset = 1.05 * max(float(np.ptp(m)) for m in means) or 1.0
        curves = [
            hv.Curve((spec_x, means[j] + j * offset),
                     'wavenumber', 'offset intensity',
                     label=f'cluster {j}').opts(color=colors[j], line_width=1.5)
            for j in range(k)
        ]
        waterfall = hv.Overlay(curves).opts(
            width=width, height=height, show_legend=True, legend_position='right',
            title='Cluster mean model_cube spectra (stacked)',
        )
        return (cluster_map + waterfall).opts(shared_axes=False)

    def _run(event=None):
        status.value = 'clustering…'
        try:
            plot_pane.object = _cluster_layout(feature_select.value, k_slider.value)
            status.value = ''
        except Exception as exc:  # surface errors in the widget, not silently
            status.value = f'error: {type(exc).__name__}: {exc}'

    run_button.on_click(_run)
    _run()

    return pn.Column(
        pn.Row(feature_select, k_slider, run_button, status),
        plot_pane,
    )
