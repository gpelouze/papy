#!/usr/share/bin python

import numpy as np
import scipy.interpolate as si

import matplotlib as mpl

def get_imshowsave_ax(shape, fig, clearfig=True):
    ''' Get an axis to save the result of an imshow as a picture with
    the same resolution as an array of given shape.

    Parameters
    ==========
    shape : tuple
        2-tuple of ints containing the shape of the array to build the axis
        for.
    fig : matplotlib.figure.Figure
        The figure to which the axis is to be attached.
    clearfig : bool (default: True)
        Clear the figure before appending the axis (recommended).

    Example
    =======

    >>> img = get_img()
    >>> img.shape
    (527, 326)
    >>> ax = get_imshowsave_ax(img.shape, plt.figure(1, clear=True))
    >>> ax.imshow(img)
    >>> ax.imshow(img > 10, alpha=0.5)
    >>> ax.figure.savefig('img.png', dpi=ax.figure.dpi)

    Note that specifying the dpi when saving the image is still required.
    '''

    if clearfig:
        fig.clear()

    fig.set_frameon(False)

    # Set figure size to 1 in, and figure DPI to array size along axis 0.
    dpi = shape[0]
    size_inches = (np.array(shape) / dpi)[::-1]
    fig.set_size_inches(*size_inches)
    fig.set_dpi(dpi)

    # add axis that fills the figure
    ax = mpl.axes.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)

    return ax

def map_extent(img, coordinates):
    ''' Compute the extent to use in ax.imshow to plot an image with
    coordinates

    Parameters
    ==========
    img : np.ndarray
        A 2D array.
    coordinates : tuple
        Either a list of bounding coordinates [xmin, xmax, ymin, ymax], pretty
        much like the extent keyword of ax.imshow, or a tuple containing
        two 1D arrays of evenly-spaced x and y values.

    The computed boundaries are the centers of the corresponding pixel. This
    differs from the behaviour of ax.imshow with the extent keyword, where the
    boundaries are the left, right, top, or bottom of the bounding pixels.
    '''
    ny, nx = img.shape
    try:
        xmin, xmax, ymin, ymax = coordinates
    except ValueError:
        x, y = coordinates
        xmin = x[0];  xmax = x[-1]
        ymin = y[0];  ymax = y[-1]
        # swap values if values were in decreasing order
        if xmin > xmax:
            xmin, xmax = xmax, xmin
        if ymin > ymax:
            ymin, ymax = ymax, ymin

    dx = (xmax - xmin) / (nx - 1)
    dy = (ymax - ymin) / (ny - 1)
    xmin -= dx / 2;  xmax += dx / 2
    ymin -= dy / 2;  ymax += dy / 2
    return xmin, xmax, ymin, ymax

def plot_map(ax, img, coordinates=None, xlog=None, ylog=None, **kwargs):
    ''' Plot an image with coordinates

    Parameters
    ==========
    ax : matplotlib.axes.Axes
    img : np.ndarray
        A 2D array.
    coordinates : tuple or None (default: None)
        Either a list of bounding coordinates [xmin, xmax, ymin, ymax], pretty
        much like the extent keyword of ax.imshow, or a tuple containing
        two 1D arrays of evenly-spaced x and y values. If None, this function
        is equivalent to ax.imshow().
    xlog, ylog: tuple, array, or None (default: None)
        If not None, display log ticks on the corresponding axis. In this case,
        this is either a 2-tuple containing the start and stop exponents
        for the log values, or an 1D array of the same length as the size of
        the image along the corresponding axis, containing the log values.
    **kwargs :
        Passed to ax.imshow.

    This function relies on map_extent(), see its docstring.
    '''

    if coordinates:
        extent = map_extent(img, coordinates)
    else:
        try:
            extent = kwargs.pop('extent')
        except KeyError:
            extent = None

    im = ax.imshow(
        img,
        extent=extent,
        **kwargs)

    if xlog is not None:
        add_log_labels_to_map(ax, img, coordinates, extent, xlog, 'x')
    if ylog is not None:
        add_log_labels_to_map(ax, img, coordinates, extent, ylog, 'y')

    return im

def log_labels(axis, ref_locations, log_locations, minor=False):
    ''' Put log labels on a linear axis

    Parameters
    ==========
    axis : matplotlib.axis.Axis
        The axis on which to put the labels.
        Such object is found at eg. `plt.gca().xaxis`.
    ref_locations : array
        Reference locations on the axis.
    log_locations : array
        Log values for the ref_locations.
    minor : bool (default: False)
        Wheter to print minor or major ticks.
    '''
    if minor:
        subs = 'all'
    else:
        subs = (1.0,)
    locator= mpl.ticker.LogLocator(subs=subs)
    locator.create_dummy_axis()
    locator.set_view_interval(log_locations.min(), log_locations.max())
    ticks_values = locator()
    ticks_locations = np.interp(
        ticks_values, log_locations, ref_locations,
        left=np.nan, right=np.nan)
    ticks_to_keep = ~np.isnan(ticks_locations)
    ticks_locations = ticks_locations[ticks_to_keep]
    ticks_values = ticks_values[ticks_to_keep]
    axis.set_ticks(ticks_locations, minor=minor)

    if not minor:
        formatter = mpl.ticker.LogFormatter()
        formatter.create_dummy_axis()
        axis.set_ticklabels([formatter(v) for v in ticks_values])

def add_log_labels_to_map(ax, img, coordinates, extent, log_descr, axis_name):
    axis = {'x': ax.xaxis, 'y': ax.yaxis}[axis_name]
    axis_id = {'x': 0, 'y': 1}[axis_name]
    axis_size = img.shape[::-1][axis_id]
    vmin, vmax = extent[2*axis_id:2*axis_id+2]

    if len(coordinates) == 2:
        lin_coord = coordinates[axis_id]
    else:
        lin_coord = np.linspace(vmin, vmax, axis_size)
    if len(log_descr) == axis_size:
        log_coord = log_descr
    else:
        log_min, log_max = log_descr
        log_coord = np.logspace(log_min, log_max, axis_size)

    log_labels(axis, lin_coord, log_coord)
    log_labels(axis, lin_coord, log_coord, minor=True)

def cube_hist(ax, cube, bins=100, **kwargs):
    ''' Plot a 2D image, where lines are the histograms of each frames of an
    input cube.  '''

    # compute histograms
    cube = np.ma.array(cube, mask=np.isnan(cube))
    hist_range = (cube.min(), cube.max())
    full_hist, bin_edges = np.histogram(
        cube.compressed(), range=hist_range, bins=bins)
    frames_hist = np.array([
        np.histogram(frame.compressed(), range=hist_range, bins=bins)[0]
        for frame in cube])

    # prepare x and y coordinates
    x = (bin_edges[1:] + bin_edges[:-1]) / 2
    y = np.arange(frames_hist.shape[0])

    # plot histograms for each frame
    # frames_hist = np.ma.array(frames_hist, mask=np.isinf(frames_hist)).filled(0)
    im = plot_map(
        ax,
        frames_hist,
        coordinates=(x, y),
        aspect=(x.max() - x.min()) / (y.max() - y.min()),
        **kwargs)
    cb = ax.figure.colorbar(im)
    ax.set_xlabel('value')
    ax.set_ylabel('frame')
    cb.set_label('count')
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())

    return im

def pixel_boundary(ax, x0, y0, pos, *args, w=1, h=1, **kwargs):
    ''' Plot the boundaries of a pixel as lines.

    Parameters
    ==========
    ax : matplotlib axis
    x0, y0 : int
        The central position of the pixel
    pos : str or list
        The boundaries to plot, chosen from 'left', 'right', 'top', and
        'bottom'.
    w, h : float (default 1.)
        The total width and height of a pixel
    Additional *args and **kwargs are passed to `ax.plot` when plotting
    the lines.
    '''
    w /= 2.
    h /= 2.
    xy = {
        'left': ((x0-w, x0-w), (y0-w, y0+w)),
        'right': ((x0+w, x0+w), (y0-w, y0+w)),
        'top': ((x0-w, x0+w), (y0+w, y0+w)),
        'bottom': ((x0-w, x0+w), (y0-w, y0-w)),
        }
    if isinstance(pos, str):
        pos = [pos]
    for p in pos:
        x, y = xy[p]
        ax.plot(x, y, *args, **kwargs)

def _in_mask(mask, x, y):
    ''' Return `True` if `mask[x, y] == True`, `False` in **all** other case. 
    In particular, if x or y are out of bounds (including x < 0 or y < 0),
    return False.
    '''
    if x < 0 or y < 0:
        return False
    try:
        return mask[x, y]
    except IndexError:
        return False

def plot_pixel_contour(ax, mask, *args, coordinates=None, **kwargs):
    ''' Plot the pixel boundaries of connex regions in a mask.

    Parameters
    ==========
    ax : matplotlib axis
    mask : np.ndarray
        A np.ndarray of booleans.
    coordinates : tuple or None (default: None)
        Either a list of bounding coordinates [xmin, xmax, ymin, ymax], pretty
        much like the extent keyword of ax.imshow, or a tuple containing
        two 1D arrays of evenly-spaced x and y values. If None, this function
        is equivalent to ax.imshow().
    Additional *args and **kwargs are passed to `ax.plot` when plotting
    the lines. If the
    '''

    # Fallback to color from prop_cycler if it isn't defined by user:
    try:
        color_in_args = False
        for arg in args:
            if arg[0] in 'bgrcmykw':
                color_in_args = True
    except IndexError:
        pass
    color_in_kwargs = ('color' in kwargs.keys())
    if (not color_in_args) and (not color_in_kwargs):
        default_props = next(ax._get_lines.prop_cycler)
        kwargs['color'] = default_props['color']

    if coordinates:
        xmin, xmax, ymin, ymax = map_extent(mask, coordinates)
        ny, nx = mask.shape
        dx = (xmax - xmin) / nx
        dy = (ymax - ymin) / ny
        xmin += dx / 2
        ymin += dy / 2
    else:
        xmin, ymin = 0, 0
        dx, dy = 1, 1

    mask = mask.T
    w, h = 1, 1
    for x, y in zip(*np.where(mask)):
        positions = []
        if not _in_mask(mask, x-w, y):
            positions.append('left')
        if not _in_mask(mask, x+w, y):
            positions.append('right')
        if not _in_mask(mask, x, y+w):
            positions.append('top')
        if not _in_mask(mask, x, y-w):
            positions.append('bottom')
        x = xmin + dx * x
        y = ymin + dy * y
        pixel_boundary(ax, x, y, positions, *args, w=dx, h=dy, **kwargs)
