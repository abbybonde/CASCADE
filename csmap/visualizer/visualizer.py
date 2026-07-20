import logging
import numpy as np
import panel as pn
import holoviews as hv
import param
from holoviews import opts, streams
from holoviews.operation.datashader import datashade, rasterize, dynspread, spread
import matplotlib.path as mpl_path
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from skimage import color
import pandas as pd

from visualizer.colormaps import recolor_image_with_umap  # noqa: F401  canonical implementation

# One FreehandDraw stream per analysis role, with a distinguishable tool
# description (shown as the toolbar tooltip).  Bokeh 3.x supports any number
# of draw tools per figure; only one *active drag* tool at a time, switched
# from the toolbar.
#
# Pen scoping (unchanged from the original scaleMAP template): pens 1–2 work
# on the EMBEDDING panel, pens 3–4 work on the IMAGE panel.  Each pen's
# stroke color matches its spectrum curve in the analysis window below
# (pen 1 = blue, pen 2 = orange on either panel).
TOOL_EMBEDDING_1 = 'Embedding pen 1 — blue (background / spectrum 1)'
TOOL_EMBEDDING_2 = 'Embedding pen 2 — orange (spectrum 2)'
TOOL_IMAGE_1 = 'Image pen 1 — blue (background / spectrum 1)'
TOOL_IMAGE_2 = 'Image pen 2 — orange (spectrum 2)'

PEN_1_COLOR = '#1f77b4'   # blue   (strokes + spectrum curves, pen 1)
PEN_2_COLOR = '#ff7f0e'   # orange (strokes + spectrum curves, pen 2)

# Cross-panel highlight colors. Embedding-pen selections paint the image;
# image-pen selections paint points onto the embedding. Both pens of each
# panel cross-highlight, in distinguishable colors.
HIGHLIGHT_IMG_1 = (0.0, 0.9, 1.0, 1.0)    # electric blue — embedding pen 1 → image
HIGHLIGHT_IMG_2 = (1.0, 0.09, 0.27, 1.0)  # hot red       — embedding pen 2 → image
HIGHLIGHT_EMB_1 = '#7ec8ff'               # light blue    — image pen 1 → embedding
HIGHLIGHT_EMB_2 = '#e65100'               # deep orange   — image pen 2 → embedding


class DispersionStream(hv.streams.Stream):
    """Stream that carries the currently selected dispersion mode.

    Parameters
    ----------
    dispersion_mode : str
        One of 'None', 'Std', 'MinMax'. Default 'None'.
    """
    dispersion_mode = param.String(default='None')


class ContrastStream(hv.streams.Stream):
    """Stream carrying percentile-clip bounds for the false-color image."""
    clip_low = param.Number(default=1.0)
    clip_high = param.Number(default=99.0)


def _last_stroke(data):
    """Return ``(xs, ys)`` of the most recent committed stroke, or ``None``.

    *data* is a FreehandDraw stream data dict: ``{'xs': [[...], ...],
    'ys': [[...], ...]}`` with one list per stroke.  A stroke needs at least
    three vertices to enclose anything.
    """
    if not data or not data.get('xs') or len(data['xs']) == 0:
        return None
    xs, ys = data['xs'][-1], data['ys'][-1]
    if xs is None or len(xs) < 3:
        return None
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def image_lasso_to_mask(data, rows, cols):
    """Convert an image-panel lasso into a flat raster-order pixel mask.

    The image panel is rendered with ``hv.RGB(..., bounds=(0, 0, cols,
    rows))``, whose data coordinates put array row 0 at the *top* (data-y
    near ``rows``).  The ``rows - y`` flip is applied here — and only in
    image-panel handlers — so a pixel ``(r, c)`` is tested at its center
    ``(c + 0.5, r + 0.5)`` in flipped coordinates.  A drawn rectangle
    therefore selects exactly the pixel block whose centers it covers.

    Parameters
    ----------
    data : dict
        FreehandDraw stream data from an image-panel stream.
    rows, cols : int
        Image grid shape.

    Returns
    -------
    np.ndarray
        Boolean mask of shape ``(rows * cols,)`` in raster (row-major)
        order, matching ``flatten_image`` / feature-matrix row order.
        All-False when no stroke has been committed.
    """
    stroke = _last_stroke(data)
    if stroke is None:
        return np.zeros(rows * cols, dtype=bool)
    xs, ys = stroke
    verts = np.column_stack([xs, rows - ys])  # y-flip: image panel only
    cc, rr = np.meshgrid(np.arange(cols) + 0.5, np.arange(rows) + 0.5)
    centers = np.column_stack([cc.ravel(), rr.ravel()])
    return mpl_path.Path(verts).contains_points(centers)


def embedding_lasso_to_mask(data, embedding):
    """Convert an embedding-panel lasso into a point-selection mask.

    Selection is evaluated directly in UMAP coordinates with
    ``Path(verts).contains_points(embedding)`` — no flip, no transpose.

    Returns a boolean mask of shape ``(len(embedding),)``; all-False when no
    stroke has been committed.
    """
    stroke = _last_stroke(data)
    if stroke is None:
        return np.zeros(len(embedding), dtype=bool)
    xs, ys = stroke
    verts = np.column_stack([xs, ys])
    return mpl_path.Path(verts).contains_points(np.asarray(embedding)[:, :2])


def extract_lasso_mask(dashboard, panel='image', slot=1):
    """Read a committed lasso from a dashboard and return a flat pixel mask.

    Parameters
    ----------
    dashboard :
        The ``pn.Column`` returned by
        :func:`generate_freehand_overlay_spectrum_2`.
    panel : {'image', 'embedding'}
        Which figure's lasso to read.
    slot : int
        Which selection tool on that figure (1 or 2).

    Returns
    -------
    np.ndarray
        Boolean mask of shape ``(rows * cols,)`` in raster order —
        ``True`` for pixels inside the drawn region.  Embedding-panel
        selections are mapped back to image pixels through the dashboard's
        ``pixel_indices``.  All-False when nothing has been drawn.
    """
    ctx = dashboard._lasso_context
    stream = dashboard._lasso_streams[f'{panel}_{slot}']
    rows, cols = ctx['rows'], ctx['cols']
    if panel == 'image':
        return image_lasso_to_mask(stream.data, rows, cols)
    elif panel == 'embedding':
        sel = embedding_lasso_to_mask(stream.data, ctx['embedding'])
        flat = np.zeros(rows * cols, dtype=bool)
        flat[ctx['pixel_indices'][sel]] = True
        return flat
    raise ValueError(f"Invalid panel '{panel}'. Must be 'image' or 'embedding'.")


def _draw_tools_hook(default_description):
    """Bokeh plot hook making multiple draw tools per figure selectable.

    Disables toolbar grouping of identically-typed tools (so every
    FreehandDraw tool appears as its own toolbar entry, distinguishable by
    its description/tooltip) and arms the tool whose description matches
    *default_description* as the default drag tool.
    """
    def hook(plot, element):
        from bokeh.models import FreehandDrawTool
        fig = plot.state
        try:
            fig.toolbar.group = False
        except (AttributeError, ValueError):
            pass  # older bokeh: no toolbar grouping to disable
        draw_tools = [t for t in fig.tools if isinstance(t, FreehandDrawTool)]
        for tool in draw_tools:
            if tool.description == default_description:
                fig.toolbar.active_drag = tool
    return hook


def generate_freehand_overlay_spectrum_2(
    hyperspectral_image, embedding, spectrum_x_axis=None, spread_px=2, cnorm='linear',
    width=400, height=400, clim=None, color_mapping='red_blue',
    peak_params_cube=None, analysis_mode='spectral', amp_threshold=1e-3,
    pixel_indices=None, contrast_percentiles=(1.0, 99.0),
    embedding_cmap='inferno', selection_size=3,
):
    """Interactive lasso dashboard: embedding + false-color image + spectra.

    Parameters (beyond the historical ones)
    ---------------------------------------
    peak_params_cube, analysis_mode, amp_threshold :
        Accepted for backward compatibility but no longer used — the
        mean-peak Spikes overlay was removed (superseded by
        ``generate_peak_explorer``).
    pixel_indices : np.ndarray or None
        Flat raster-order indices of the image pixels that ``embedding``
        rows correspond to.  ``None`` (default) means the full image was
        embedded (``embedding`` has ``rows * cols`` rows).  Pass
        ``np.flatnonzero(keep_mask)`` for a background-masked embedding so
        cross-panel highlighting and spectra stay pixel-accurate.
    contrast_percentiles : (float, float)
        Initial percentile-clip bounds for the false-color image contrast
        slider (see ``recolor_image_with_umap``).
    embedding_cmap : str
        Colormap for the rasterized embedding density.  Default 'inferno'.
    selection_size :
        Glyph size of the image-lasso → embedding highlight points.
        Rendered as actual point glyphs (not a rasterized block overlay) so
        they scale with the underlying scatter and never blanket the
        neighborhood.  Pen 1 highlights light blue, pen 2 deep orange; on
        the image panel, embedding-pen selections paint electric blue
        (pen 1) and hot red (pen 2).

    Returns
    -------
    pn.Column
        The dashboard.  The committed lasso selections are exposed for
        downstream cells via ``extract_lasso_mask(dashboard, panel, slot)``.
    """
    embedding = np.asarray(embedding)
    im_y, im_x = hyperspectral_image.shape[0:2]  # rows, cols

    if pixel_indices is None:
        if embedding.shape[0] != im_y * im_x:
            raise ValueError(
                f"embedding has {embedding.shape[0]} rows but the image has "
                f"{im_y * im_x} pixels; pass pixel_indices for a masked "
                "embedding."
            )
        pixel_indices = np.arange(im_y * im_x)
    else:
        pixel_indices = np.asarray(pixel_indices)
        if pixel_indices.shape[0] != embedding.shape[0]:
            raise ValueError(
                f"pixel_indices has {pixel_indices.shape[0]} entries but "
                f"embedding has {embedding.shape[0]} rows."
            )

    if spectrum_x_axis is None:
        spectrum_x_axis = np.arange(hyperspectral_image.shape[2])

    # Convert embedding to Points and rasterize
    hv_embedding = hv.Points(embedding)
    if clim is None:
        embedding_hist2d = rasterize(hv_embedding).opts(
            colorbar=True, width=500, height=400, cnorm=cnorm, cmap=embedding_cmap)
    else:
        embedding_hist2d = rasterize(hv_embedding).opts(
            colorbar=True, width=500, height=400, cnorm=cnorm, clim=clim,
            cmap=embedding_cmap)

    # Store embedding range for overlay layer
    embedding_xrange = (embedding[:, 0].min(), embedding[:, 0].max())
    embedding_yrange = (embedding[:, 1].min(), embedding[:, 1].max())

    # Store ylim for hsi
    spectral_yrange = (hyperspectral_image.min(), hyperspectral_image.max())

    # Draw sources — one Polygons element per analysis role. Stroke colors
    # match the corresponding spectrum curves in the analysis window
    # (pen 1 = blue, pen 2 = orange).
    path_on_points = hv.Polygons([]).opts(
        line_color=PEN_1_COLOR, line_width=2, fill_alpha=0.15, fill_color=PEN_1_COLOR)
    path_on_points_2 = hv.Polygons([]).opts(
        line_color=PEN_2_COLOR, line_width=2, fill_alpha=0.15, fill_color=PEN_2_COLOR)
    path_on_img = hv.Polygons([]).opts(
        line_color=PEN_1_COLOR, line_width=2, fill_alpha=0.15, fill_color=PEN_1_COLOR)
    path_on_img_2 = hv.Polygons([]).opts(
        line_color=PEN_2_COLOR, line_width=2, fill_alpha=0.15, fill_color=PEN_2_COLOR)

    # One stream per role; distinct tooltips become distinct, separately
    # selectable toolbar entries (see _draw_tools_hook).
    path_on_points_stream = streams.FreehandDraw(
        source=path_on_points, num_objects=1, tooltip=TOOL_EMBEDDING_1)
    path_on_points_stream_2 = streams.FreehandDraw(
        source=path_on_points_2, num_objects=1, tooltip=TOOL_EMBEDDING_2)
    path_on_img_stream = streams.FreehandDraw(
        source=path_on_img, num_objects=1, tooltip=TOOL_IMAGE_1)
    path_on_img_stream_2 = streams.FreehandDraw(
        source=path_on_img_2, num_objects=1, tooltip=TOOL_IMAGE_2)

    out_img_empty = np.zeros((im_y, im_x, 4), dtype=np.float32)

    def _image_mask2d_from_image_lasso(data):
        """Image-panel lasso → 2-D pixel mask (flip handled in helper)."""
        return image_lasso_to_mask(data, im_y, im_x).reshape(im_y, im_x)

    def _image_mask2d_from_embedding_lasso(data):
        """Embedding-panel lasso → 2-D pixel mask via pixel_indices."""
        sel = embedding_lasso_to_mask(data, embedding)
        flat = np.zeros(im_y * im_x, dtype=bool)
        flat[pixel_indices[sel]] = True
        return flat.reshape(im_y, im_x)

    def _make_embedding_highlight(color):
        """Image-panel lasso → highlight matching points in the embedding.

        Renders the selected pixels as small point glyphs (same scale as
        the underlying scatter) instead of a rasterized+spread block, which
        drew oversized squares that blanketed the neighborhood.
        """
        opts_kwargs = dict(color=color, size=selection_size, alpha=0.8)
        state = {'last': hv.Points([]).opts(**opts_kwargs)}

        def callback(data):
            try:
                flat = image_lasso_to_mask(data, im_y, im_x)
                sel = flat[pixel_indices]
                selected = embedding[sel] if sel.any() else []
                result = hv.Points(selected).opts(**opts_kwargs)
                state['last'] = result
                return result
            except Exception as exc:
                logging.warning("embedding highlight callback failed: %s: %s",
                                type(exc).__name__, exc)
                return state['last']

        return callback

    def _make_image_highlight(rgba):
        """Embedding-panel lasso → highlight matching pixels in the image."""
        state = {'last': hv.RGB(out_img_empty, bounds=(0, 0, im_x, im_y))}

        def callback(data):
            try:
                mask2d = _image_mask2d_from_embedding_lasso(data)
                if mask2d.any():
                    out_img = out_img_empty.copy()
                    out_img[mask2d] = list(rgba)
                else:
                    out_img = out_img_empty
                result = hv.RGB(out_img, bounds=(0, 0, im_x, im_y))
                state['last'] = result
                return result
            except Exception as exc:
                logging.warning("image highlight callback failed: %s: %s",
                                type(exc).__name__, exc)
                return state['last']

        return callback

    update_points   = _make_embedding_highlight(HIGHLIGHT_EMB_1)
    update_points_2 = _make_embedding_highlight(HIGHLIGHT_EMB_2)
    update_img      = _make_image_highlight(HIGHLIGHT_IMG_1)
    update_img_2    = _make_image_highlight(HIGHLIGHT_IMG_2)

    def _spectrum_overlay(mask2d, dispersion_mode, pen_color):
        """Mean spectrum (+ dispersion band) for a pixel mask.

        Spectral extraction always comes from the model cube, never the
        descriptor.  (The mean-peak ``hv.Spikes`` overlay that used to
        accompany peak mode was removed: averaging peak parameters over a
        large selection produces meaningless spikes, and the peak explorer
        answers the per-peak question properly.)

        The returned overlay has the SAME structure on every call
        (Curve * Area, empty elements when nothing is selected).  A
        DynamicMap whose callback changes structure between calls cannot be
        updated in place by the bokeh backend — that was why the analysis
        window never updated.  The curve is drawn in *pen_color* so each
        spectrum is attributable to the pen that produced it.
        """
        if mask2d.any():
            mean = np.mean(hyperspectral_image[mask2d], axis=0)
        else:
            mean = np.zeros(hyperspectral_image.shape[2])

        curve = hv.Curve((spectrum_x_axis, mean), vdims=['mean']).opts(
            width=475, height=400, line_width=1.5, ylim=spectral_yrange,
            color=pen_color,
        )

        if dispersion_mode == 'Std' and mask2d.any():
            std = np.std(hyperspectral_image[mask2d], axis=0)
            area = hv.Area((spectrum_x_axis, mean - std / 2, mean + std / 2), vdims=['low', 'high'])
        elif dispersion_mode == 'MinMax' and mask2d.any():
            min_ = np.min(hyperspectral_image[mask2d], axis=0)
            max_ = np.max(hyperspectral_image[mask2d], axis=0)
            area = hv.Area((spectrum_x_axis, min_, max_), vdims=['low', 'high'])
        else:
            area = hv.Area((spectrum_x_axis, mean, mean), vdims=['low', 'high'])

        return curve * area.opts(line_alpha=0.1, fill_alpha=0.4, color=pen_color)

    _no_selection = np.zeros((im_y, im_x), dtype=bool)

    def _make_spectrum_callback(mask_fn, pen_color):
        """Build a stream callback bound to a mask source and a pen color."""
        state = {'last': _spectrum_overlay(_no_selection, 'None', pen_color)}

        def callback(data, dispersion_mode):
            try:
                if dispersion_mode not in ('Std', 'MinMax'):
                    dispersion_mode = 'None'
                output = _spectrum_overlay(mask_fn(data), dispersion_mode, pen_color)
                state['last'] = output
                return output
            except Exception as exc:
                logging.warning("spectrum callback failed: %s: %s", type(exc).__name__, exc)
                return state['last']

        return callback

    update_point_spectrum   = _make_spectrum_callback(_image_mask2d_from_embedding_lasso, PEN_1_COLOR)
    update_point_spectrum_2 = _make_spectrum_callback(_image_mask2d_from_embedding_lasso, PEN_2_COLOR)
    update_img_spectrum     = _make_spectrum_callback(_image_mask2d_from_image_lasso, PEN_1_COLOR)
    update_img_spectrum_2   = _make_spectrum_callback(_image_mask2d_from_image_lasso, PEN_2_COLOR)

    def b1_update(event):
        b1_stream.event(dispersion_mode=dispersion.value)

    def b2_update(event):
        b2_stream.event(dispersion_mode=dispersion.value)

    def b3_update(event):
        b3_stream.event(dispersion_mode=dispersion.value)

    def b4_update(event):
        b4_stream.event(dispersion_mode=dispersion.value)

    def copy_x(event):
        pd.Series(spectrum_x_axis, name='x_axis').to_clipboard(index=False)

    def copy_c1(event):
        dmap_dict = point_spectrum_dmap.data[()]
        if copy_dispersion.value:
            pd.concat([dmap_dict.data[('Curve', 'I')].data['mean'],
                       dmap_dict.data[('Area', 'I')].data['low'],
                       dmap_dict.data[('Area', 'I')].data['high']], axis=1).to_clipboard(index=False)
        else:
            dmap_dict.data[('Curve', 'I')].data['mean'].to_clipboard(index=False)

    def copy_c2(event):
        dmap_dict = point_spectrum_dmap_2.data[()]
        if copy_dispersion.value:
            pd.concat([dmap_dict.data[('Curve', 'I')].data['mean'],
                       dmap_dict.data[('Area', 'I')].data['low'],
                       dmap_dict.data[('Area', 'I')].data['high']], axis=1).to_clipboard(index=False)
        else:
            dmap_dict.data[('Curve', 'I')].data['mean'].to_clipboard(index=False)

    def copy_c3(event):
        dmap_dict = img_spectrum_dmap.data[()]
        if copy_dispersion.value:
            pd.concat([dmap_dict.data[('Curve', 'I')].data['mean'],
                       dmap_dict.data[('Area', 'I')].data['low'],
                       dmap_dict.data[('Area', 'I')].data['high']], axis=1).to_clipboard(index=False)
        else:
            dmap_dict.data[('Curve', 'I')].data['mean'].to_clipboard(index=False)

    def copy_c4(event):
        dmap_dict = img_spectrum_dmap_2.data[()]
        if copy_dispersion.value:
            pd.concat([dmap_dict.data[('Curve', 'I')].data['mean'],
                       dmap_dict.data[('Area', 'I')].data['low'],
                       dmap_dict.data[('Area', 'I')].data['high']], axis=1).to_clipboard(index=False)
        else:
            dmap_dict.data[('Curve', 'I')].data['mean'].to_clipboard(index=False)

    b1_stream = DispersionStream()
    b2_stream = DispersionStream()
    b3_stream = DispersionStream()
    b4_stream = DispersionStream()

    dispersion = pn.widgets.RadioBoxGroup(name='dispersion', options=['None', 'Std', 'MinMax'], inline=True, width=175, styles={'align-self': 'center'})
    update_1   = pn.widgets.Button(name='Points 1', width=50, styles={'align-self': 'center'})
    update_2   = pn.widgets.Button(name='Points 2', width=50, styles={'align-self': 'center'})
    update_3   = pn.widgets.Button(name='Image 1', width=50, styles={'align-self': 'center'})
    update_4   = pn.widgets.Button(name='Image 2', width=50, styles={'align-self': 'center'})

    update_1.on_click(b1_update)
    update_2.on_click(b2_update)
    update_3.on_click(b3_update)
    update_4.on_click(b4_update)

    dispersion_info = pn.widgets.StaticText(name='Choose dispersion mode', value='', styles={'align-self': 'center'})
    update_text     = pn.widgets.StaticText(name='Update', value='', styles={'align-self': 'center'})

    selection_parameters = pn.Row(dispersion_info, dispersion, update_text, update_1, update_2, update_3, update_4)

    # Clipboard tools
    clipboard_info  = pn.widgets.StaticText(name='Copy', value='', styles={'align-self': 'center'})
    copy_dispersion = pn.widgets.Toggle(name='Include dispersion (±)', button_type='primary', width=150, styles={'align-self': 'center'})
    copy_x_axis     = pn.widgets.Button(name='X_axis', width=50, styles={'align-self': 'center'})
    copy_1          = pn.widgets.Button(name='Points 1', width=50, styles={'align-self': 'center'})
    copy_2          = pn.widgets.Button(name='Points 2', width=50, styles={'align-self': 'center'})
    copy_3          = pn.widgets.Button(name='Image 1', width=50, styles={'align-self': 'center'})
    copy_4          = pn.widgets.Button(name='Image 2', width=50, styles={'align-self': 'center'})

    copy_x_axis.on_click(copy_x)
    copy_1.on_click(copy_c1)
    copy_2.on_click(copy_c2)
    copy_3.on_click(copy_c3)
    copy_4.on_click(copy_c4)

    clipboard_tools = pn.Row(clipboard_info, copy_dispersion, copy_x_axis, copy_1, copy_2, copy_3, copy_4)

    # --- Image contrast control (percentile clip of embedding channels) ----
    contrast_stream = ContrastStream(
        clip_low=float(contrast_percentiles[0]),
        clip_high=float(contrast_percentiles[1]),
    )

    def update_rgb(clip_low, clip_high):
        img = recolor_image_with_umap(
            hyperspectral_image, embedding, color_mapping=color_mapping,
            mask=None if len(pixel_indices) == im_y * im_x
                 else np.isin(np.arange(im_y * im_x), pixel_indices).reshape(im_y, im_x),
            clip_percentiles=(clip_low, clip_high),
        )
        return hv.RGB(img, bounds=(0, 0, im_x, im_y))

    rgb_dmap = hv.DynamicMap(update_rgb, streams=[contrast_stream])

    contrast_slider = pn.widgets.RangeSlider(
        name='Image contrast (percentile clip)', start=0.0, end=100.0,
        value=(float(contrast_percentiles[0]), float(contrast_percentiles[1])),
        step=0.5, width=300, styles={'align-self': 'center'})

    def _contrast_changed(event):
        low, high = event.new
        if high - low >= 0.5:
            contrast_stream.event(clip_low=float(low), clip_high=float(high))

    contrast_slider.param.watch(_contrast_changed, 'value')

    img_dmap      = hv.DynamicMap(update_img,      streams=[path_on_points_stream])
    img_dmap_2    = hv.DynamicMap(update_img_2,    streams=[path_on_points_stream_2])
    hist2d_dmap   = hv.DynamicMap(update_points,   streams=[path_on_img_stream])
    hist2d_dmap_2 = hv.DynamicMap(update_points_2, streams=[path_on_img_stream_2])

    point_spectrum_dmap   = hv.DynamicMap(update_point_spectrum,   streams=[path_on_points_stream,   b1_stream])
    point_spectrum_dmap_2 = hv.DynamicMap(update_point_spectrum_2, streams=[path_on_points_stream_2, b2_stream])
    img_spectrum_dmap     = hv.DynamicMap(update_img_spectrum,     streams=[path_on_img_stream,      b3_stream])
    img_spectrum_dmap_2   = hv.DynamicMap(update_img_spectrum_2,   streams=[path_on_img_stream_2,    b4_stream])

    embedding_panel = (
        embedding_hist2d * hist2d_dmap * hist2d_dmap_2
        * path_on_points * path_on_points_2
    ).opts(hooks=[_draw_tools_hook(TOOL_EMBEDDING_1)])
    image_panel = (
        rgb_dmap * img_dmap * img_dmap_2 * path_on_img * path_on_img_2
    ).opts(hooks=[_draw_tools_hook(TOOL_IMAGE_1)])

    overlay = embedding_panel + image_panel

    overlay.opts(shared_axes=False)
    overlay.opts(
        opts.RGB(width=width, height=height),
    )

    dashboard = pn.Column(overlay,
                          pn.Row(selection_parameters, contrast_slider),
                          (point_spectrum_dmap * point_spectrum_dmap_2 + img_spectrum_dmap * img_spectrum_dmap_2),
                          clipboard_tools)

    # Expose committed lassos to downstream notebook cells
    # (see extract_lasso_mask).
    dashboard._lasso_streams = {
        'embedding_1': path_on_points_stream,
        'embedding_2': path_on_points_stream_2,
        'image_1': path_on_img_stream,
        'image_2': path_on_img_stream_2,
    }
    dashboard._lasso_context = {
        'rows': im_y,
        'cols': im_x,
        'embedding': embedding,
        'pixel_indices': pixel_indices,
    }
    return dashboard

# To use the function:
# overlay_result = generate_overlay(image, embedding)


def generate_peak_explorer(
    peak_params_cube,
    x_axis=None,
    amp_threshold=1e-3,
    tolerance=10.0,
    n_bins=200,
    width=500,
    height=400,
):
    """Interactive explorer: which pixels contain a peak at a given center?

    Layout:

    * **Left** — histogram of every real peak center in the image (one entry
      per fitted peak above *amp_threshold*).  **Tap a bar** to jump the
      wavenumber slider to that peak.
    * **Right** — spatial map of the pixels that contain at least one real
      peak whose center lies within ``wavenumber ± tolerance``, colored by
      that peak's amplitude (0 elsewhere).
    * Sliders — scroll ``wavenumber`` through the spectrum and adjust the
      matching ``tolerance``.

    Parameters
    ----------
    peak_params_cube : np.ndarray
        CASCADE peak cube of shape ``(rows, cols, max_peaks * 4)`` with
        ``(amplitude, center, sigma, gamma)`` blocks.
    x_axis : np.ndarray or None
        Optional wavenumber axis; only used to bound the slider range.
    amp_threshold : float
        Minimum amplitude for a slot to count as a real peak.
    tolerance : float
        Initial half-width (in wavenumber units) of the center-matching
        window.
    n_bins : int
        Number of histogram bins over the observed center range.

    Returns
    -------
    pn.Column
        The explorer dashboard.
    """
    rows, cols = peak_params_cube.shape[:2]
    max_peaks = peak_params_cube.shape[2] // 4
    blocks = peak_params_cube.reshape(rows, cols, max_peaks, 4)
    amps = blocks[..., 0]
    centers = blocks[..., 1]
    real = amps > amp_threshold

    real_centers = centers[real]
    real_amps = amps[real]
    if real_centers.size == 0:
        return pn.Column(pn.pane.Markdown(
            f'**No peaks above amp_threshold={amp_threshold:g} in this image.**'))

    if x_axis is not None:
        w_min, w_max = float(np.min(x_axis)), float(np.max(x_axis))
    else:
        w_min, w_max = float(real_centers.min()), float(real_centers.max())

    counts, edges = np.histogram(real_centers, bins=n_bins, range=(w_min, w_max))
    histogram = hv.Histogram((edges, counts), kdims=['center'],
                             vdims=['peak count']).opts(
        width=width, height=height, tools=['tap', 'hover'],
        fill_color=PEN_1_COLOR, line_color='white',
        title='All fitted peak centers (tap to select)')

    w_slider = pn.widgets.FloatSlider(
        name='Peak center wavenumber', start=w_min, end=w_max,
        value=float(edges[np.argmax(counts)]), step=(w_max - w_min) / 1000.0,
        width=380)
    tol_slider = pn.widgets.FloatSlider(
        name='Center tolerance (±)', start=0.5, end=100.0,
        value=float(tolerance), step=0.5, width=280)
    gain_slider = pn.widgets.FloatSlider(
        name='Color gain (×)', start=1.0, end=50.0, value=1.0, step=0.5,
        width=280)
    cnorm_select = pn.widgets.Select(
        name='Color scale', options=['linear', 'log', 'eq_hist'],
        value='eq_hist', width=120)

    def _peak_map(wavenumber, tol, gain, cnorm):
        match = real & (np.abs(centers - wavenumber) <= tol)
        amp_map = np.where(match, amps, 0.0).max(axis=2)
        n_matching = int(match.any(axis=2).sum())
        # Contrast scales to the CURRENT selection (not the global max) and
        # the gain saturates the top end, so sparse/weak peaks stay visible.
        top = float(amp_map.max())
        clim_hi = (top / gain) if top > 0 else 1.0
        return hv.Image(
            amp_map, bounds=(0, 0, cols, rows), kdims=['x', 'y'],
            vdims=['amplitude'],
        ).opts(
            width=width, height=height, cmap='inferno', colorbar=True,
            cnorm=cnorm,
            clim=(1e-6 if cnorm == 'log' else 0.0, clim_hi),
            clipping_colors={'max': '#ffff99'},
            title=(f'Pixels with a peak at {wavenumber:.1f} ± {tol:.1f} '
                   f'({n_matching} px)'),
        )

    peak_map_dmap = hv.DynamicMap(pn.bind(_peak_map, wavenumber=w_slider,
                                          tol=tol_slider, gain=gain_slider,
                                          cnorm=cnorm_select))

    tap = streams.SingleTap(source=histogram)

    def _on_tap(x, y):
        if x is not None:
            w_slider.value = float(np.clip(x, w_min, w_max))

    tap.add_subscriber(_on_tap)

    marker_dmap = hv.DynamicMap(pn.bind(
        lambda wavenumber: hv.VLine(wavenumber).opts(color=PEN_2_COLOR,
                                                     line_width=2),
        wavenumber=w_slider))

    return pn.Column(
        pn.Row(w_slider, tol_slider),
        pn.Row(gain_slider, cnorm_select),
        (histogram * marker_dmap + peak_map_dmap).opts(shared_axes=False),
    )


def flatten_image(hyperspectral_image):
    """
    Flattens a hyperspectral 2D image while keeping the last dimension intact.

    Parameters:
    - hyperspectral_image: NumPy array representing the hyperspectral image.

    Returns:
    - Flattened hyperspectral image as a 2D NumPy array.
    """
    return hyperspectral_image.reshape(-1, hyperspectral_image.shape[-1])
