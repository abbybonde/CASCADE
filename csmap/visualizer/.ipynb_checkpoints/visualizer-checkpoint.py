import numpy as np
import panel as pn
import holoviews as hv
from holoviews import opts, streams
from holoviews.operation.datashader import datashade, rasterize, dynspread, spread
import matplotlib.path as mpl_path
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from skimage import color
import pandas as pd

def recolor_image_with_umap(np_hyperspectral_img, embedding, color_mapping='red_blue', mask=None):
    """
    Recolors a hyperspectral image using either red/blue color mapping or LAB color mapping.
    
    Parameters:
        np_hyperspectral_img (numpy.ndarray): The hyperspectral image to be recolored.
        embedding (numpy.ndarray): UMAP coordinates for color mapping.
        color_mapping (str): 'red_blue' for red/blue color mapping or 'lab' for LAB color mapping.
        mask (numpy.ndarray): Optional mask for selective color mapping.
    
    Returns:
        numpy.ndarray: The recolored image in RGBA format.
    """
    if mask is None:
        mask = np.full_like(np_hyperspectral_img[:, :, 0], True, dtype=bool)
    
    if color_mapping == 'red_blue':
        # Red/Blue color mapping
        final_image = np.zeros((np_hyperspectral_img.shape[0], np_hyperspectral_img.shape[1], 4), dtype=np.float32)
        final_image[mask, 0] = embedding[:, 0]  # Red channel
        final_image[mask, 2] = embedding[:, 1]  # Blue channel
        final_image[:, :, 0] -= final_image[:, :, 0].min()
        final_image[:, :, 0] /= final_image[:, :, 0].max()
        final_image[:, :, 2] -= final_image[:, :, 2].min()
        final_image[:, :, 2] /= final_image[:, :, 2].max()
        final_image[mask, 3] = 1  # Alpha channel

    elif color_mapping == 'lab':
        # LAB color mapping
        lab_image = np.zeros((np_hyperspectral_img.shape[0], np_hyperspectral_img.shape[1], 3))
        lab_image[..., 0] = 50
        lab_image[mask, 1] = embedding[:, 0]  # a channel
        lab_image[mask, 2] = embedding[:, 1]  # b channel
        for i in range(1, 3):
            lab_image[:, :, i] -= lab_image[:, :, i].min()
            lab_image[:, :, i] /= lab_image[:, :, i].max()
            lab_image[:, :, i] = lab_image[:, :, i] * 255 - 128
        rgb_image = color.lab2rgb(lab_image)
        alpha = np.zeros((np_hyperspectral_img.shape[0], np_hyperspectral_img.shape[1]))
        alpha[mask] = 1
        final_image = np.dstack([rgb_image, alpha])
    
    return final_image

# Usage example with mask for red/blue color mapping:
# masked_red_blue_image = recolor_image(np_hyperspectral_img, embedding, color_mapping='red_blue', mask)

# Usage example without mask for LAB color mapping:
# lab_image = recolor_image(np_hyperspectral_img, embedding, color_mapping='lab')

global mask, selected_points, point_spectrum_dmap

def generate_freehand_overlay_spectrum_2(hyperspectral_image, embedding, spectrum_x_axis=None, spread_px=2, cnorm='linear', width=400, height=400, clim=None, color_mapping='red_blue'):
    image = recolor_image_with_umap(hyperspectral_image, embedding, color_mapping=color_mapping)    
    
    if spectrum_x_axis is None:
        spectrum_x_axis = np.arange(hyperspectral_image.shape[2])
    
    global mask, selected_points, point_spectrum_dmap
    # Convert embedding to Points and rasterize
    hv_embedding = hv.Points(embedding)
    if clim is None:
        embedding_hist2d = rasterize(hv_embedding).opts(
            colorbar=True, width=500, height=400, cnorm=cnorm)
    else:
        embedding_hist2d = rasterize(hv_embedding).opts(
            colorbar=True, width=500, height=400, cnorm=cnorm, clim=clim)
        
    # Store embedding range for overlay layer
    embedding_xrange = (embedding[:,0].min(), embedding[:,0].max())
    embedding_yrange = (embedding[:,1].min(), embedding[:,1].max())
    
    # Store ylim for hsi
    spectral_yrange = (hyperspectral_image.min(), hyperspectral_image.max())

    # Make empty polys
    path_on_points   = hv.Polygons([])
    path_on_points_2 = hv.Polygons([])
    path_on_img      = hv.Polygons([])
    path_on_img_2    = hv.Polygons([])

    # Attach stream to this
    path_on_points_stream   = streams.FreehandDraw(source=path_on_points, num_objects=1)
    path_on_points_stream_2 = streams.FreehandDraw(source=path_on_points_2, num_objects=1)
    path_on_img_stream = streams.FreehandDraw(source=path_on_img, num_objects=1)
    path_on_img_stream_2 = streams.FreehandDraw(source=path_on_img_2, num_objects=1)

    # Identify image size
    im_y, im_x = image.shape[0:2]

    # Create x,y array to check if within polygon 
    x, y = np.meshgrid(np.arange(im_x), np.arange(im_y))
    _mask = np.full_like(image[:,:,0], True, dtype=bool)
    x, y = x[_mask], y[_mask]
    img_points = np.vstack((x, y)).T
    out_img_empty = np.multiply.outer(np.full_like(image[:,:,0],True),np.array([0,0,0,0]))

    def update_points(data):
        if data and 'xs' in data and 'ys' in data and len(data['xs']) > 0:
            coords = list(zip(data['xs'], data['ys']))
            mpl_coords = np.array(coords)
            mpl_coords[0].T[:,1] = im_y - mpl_coords[0].T[:,1]
            polygon_path = mpl_path.Path(mpl_coords[0].T)
            mask = polygon_path.contains_points(img_points)
            selected_points = hv_embedding.iloc[mask]
        else:
            selected_points = hv.Points([])
        return spread(
                rasterize(selected_points, dynamic=False, 
                          y_range = embedding_yrange,
                          x_range = embedding_xrange                          
                         ).opts(cmap='kr_r'),
                    px=spread_px)

    def update_img(data):
        global mask
        if data and 'xs' in data and 'ys' in data and len(data['xs']) > 0:
            coords = list(zip(data['xs'], data['ys']))
            polygon_path = mpl_path.Path(np.array(coords)[0].T)
            mask = polygon_path.contains_points(embedding)
            mask = mask.reshape(image.shape[0:2])
            out_img = out_img_empty.copy()
            out_img[mask] = [0,255,0,255]
        else:
            out_img = out_img_empty
        return hv.RGB(out_img, bounds=(0, 0, im_x, im_y))
    
    def update_point_spectrum(data, dispersion_mode):
        global selected_points, mask
        if data and 'xs' in data and 'ys' in data and len(data['xs']) > 0:
            coords = list(zip(data['xs'], data['ys']))
            polygon_path = mpl_path.Path(np.array(coords)[0].T)
            mask = polygon_path.contains_points(embedding)
            mask = mask.reshape(image.shape[0:2])
            mean = np.mean(hyperspectral_image[mask],axis=0)
        else:
            mean = np.zeros(hyperspectral_image.shape[2])

        if (dispersion.value == 'Std'):
            std  = np.std(hyperspectral_image[mask],axis=0)
            area = hv.Area((spectrum_x_axis, mean - std/2, mean + std/2), vdims=['low','high'])
        elif (dispersion.value == 'MinMax'):
            min_  = np.min(hyperspectral_image[mask],axis=0)
            max_  = np.max(hyperspectral_image[mask],axis=0)
            area = hv.Area((spectrum_x_axis, min_, max_), vdims=['low','high'])
        else:
            area = hv.Area((spectrum_x_axis, mean, mean), vdims=['low','high'])
            
        output = hv.Curve((spectrum_x_axis, mean), vdims=['mean']).opts(width=475,height=400,line_width=1, ylim=spectral_yrange) * \
                 area.opts(line_alpha=0.1, fill_alpha=0.4)
                 
        return output
    
    def update_img_spectrum(data, dispersion_mode):
        global selected_points, mask
        if data and 'xs' in data and 'ys' in data and len(data['xs']) > 0:
            coords = list(zip(data['xs'], data['ys']))
            mpl_coords = np.array(coords)
            mpl_coords[0].T[:,1] = im_y - mpl_coords[0].T[:,1]
            polygon_path = mpl_path.Path(mpl_coords[0].T)
            mask = polygon_path.contains_points(img_points)
            mask_reshape = mask.reshape(hyperspectral_image.shape[0:2])
            mean = np.mean(hyperspectral_image[mask_reshape],axis=0)
        else:
            mean = np.zeros(hyperspectral_image.shape[2])
        
        if (dispersion.value == 'Std'):
            std  = np.std(hyperspectral_image[mask_reshape],axis=0)
            area = hv.Area((spectrum_x_axis, mean - std, mean + std), vdims=['low','high'])
        elif (dispersion.value == 'MinMax'):
            min_  = np.min(hyperspectral_image[mask_reshape],axis=0)
            max_  = np.max(hyperspectral_image[mask_reshape],axis=0)
            area = hv.Area((spectrum_x_axis, min_, max_), vdims=['low','high'])
        else:
            area = hv.Area((spectrum_x_axis, mean, mean), vdims=['low','high'])
            
        output = hv.Curve((spectrum_x_axis, mean), vdims=['mean']).opts(width=475,height=400,line_width=1, ylim=spectral_yrange) * \
                 area.opts(line_alpha=0.1, fill_alpha=0.4)
            
        return output
    
    def b1_update(event):
        point_spectrum_dmap.event(dispersion_mode = dispersion.value)
        
    def b2_update(event):
        point_spectrum_dmap_2.event(dispersion_mode = dispersion.value)
        
    def b3_update(event):
        img_spectrum_dmap.event(dispersion_mode = dispersion.value)
        
    def b4_update(event):
        img_spectrum_dmap_2.event(dispersion_mode = dispersion.value)
        
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

    dispersion = pn.widgets.RadioBoxGroup(name='dispersion', options=['None','Std', 'MinMax'], inline=True, width = 175, align='center')
    update_1   = pn.widgets.Button(name = 'Points 1', width = 50, align='center')
    update_2   = pn.widgets.Button(name = 'Points 2', width = 50, align='center')
    update_3   = pn.widgets.Button(name = 'Image 1', width = 50, align='center')
    update_4   = pn.widgets.Button(name = 'Image 2', width = 50, align='center')
    
    update_1.on_click(b1_update)
    update_2.on_click(b2_update)
    update_3.on_click(b3_update)
    update_4.on_click(b4_update)
    
    dispersion_info = pn.widgets.StaticText(name='Choose dispersion mode', value = '', align='center')
    update_text     = pn.widgets.StaticText(name='Update', value = '', align='center')
    
    selection_parameters = pn.Row(dispersion_info, dispersion, update_text, update_1, update_2, update_3, update_4)
   
    b1_stream = hv.streams.Stream.define('dispersion_mode', dispersion_mode = 'None')
    b2_stream = hv.streams.Stream.define('dispersion_mode', dispersion_mode = 'None')
    b3_stream = hv.streams.Stream.define('dispersion_mode', dispersion_mode = 'None')
    b4_stream = hv.streams.Stream.define('dispersion_mode', dispersion_mode = 'None')
    
    # Clipboard tools
    clipboard_info  = pn.widgets.StaticText(name='Copy', value = '', align='center')
    copy_dispersion = pn.widgets.Toggle(name='Include dispersion (±)', button_type='primary', width = 150, align='center')
    copy_x_axis     = pn.widgets.Button(name = 'X_axis', width = 50, align='center')
    copy_1          = pn.widgets.Button(name = 'Points 1', width = 50, align='center')
    copy_2          = pn.widgets.Button(name = 'Points 2', width = 50, align='center')
    copy_3          = pn.widgets.Button(name = 'Image 1', width = 50, align='center')
    copy_4          = pn.widgets.Button(name = 'Image 2', width = 50, align='center')
    
    copy_x_axis.on_click(copy_x)
    copy_1.on_click(copy_c1)
    copy_2.on_click(copy_c2)
    copy_3.on_click(copy_c3)
    copy_4.on_click(copy_c4)
    
    clipboard_tools = pn.Row(clipboard_info, copy_dispersion, copy_x_axis, copy_1, copy_2, copy_3, copy_4)
    
    img_dmap    = hv.DynamicMap(update_img, streams=[path_on_points_stream])    
    hist2d_dmap = hv.DynamicMap(update_points, streams=[path_on_img_stream])
    
    point_spectrum_dmap   = hv.DynamicMap(update_point_spectrum, streams=[path_on_points_stream,   b1_stream()])    
    point_spectrum_dmap_2 = hv.DynamicMap(update_point_spectrum, streams=[path_on_points_stream_2, b2_stream()])    
    img_spectrum_dmap     = hv.DynamicMap(update_img_spectrum, streams=[path_on_img_stream,        b3_stream()])      
    img_spectrum_dmap_2   = hv.DynamicMap(update_img_spectrum, streams=[path_on_img_stream_2,      b4_stream()])     

    overlay = (embedding_hist2d * hist2d_dmap * path_on_points * path_on_points_2
                + (hv.RGB(image, bounds=(0, 0, im_x, im_y)) * img_dmap * path_on_img * path_on_img_2))
    
    overlay.opts(shared_axes=False)
    overlay.opts(
        opts.RGB(width=width, height=height),
        opts.Polygons(active_tools=['freehand_draw'], fill_alpha=0.3)
    )
    
    return pn.Column(overlay,
                     selection_parameters,
                     (point_spectrum_dmap * point_spectrum_dmap_2 + img_spectrum_dmap * img_spectrum_dmap_2),
                     clipboard_tools) #,
                     # (point_spectrum_dmap * img_spectrum_dmap).opts(width=950))

# To use the function:
# overlay_result = generate_overlay(image, embedding)

def flatten_image(hyperspectral_image):
    """
    Flattens a hyperspectral 2D image while keeping the last dimension intact.

    Parameters:
    - hyperspectral_image: NumPy array representing the hyperspectral image.

    Returns:
    - Flattened hyperspectral image as a 2D NumPy array.
    """
    return hyperspectral_image.reshape(-1, hyperspectral_image.shape[-1])
