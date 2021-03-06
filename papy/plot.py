''' Matplotlib tools. '''

from astropy.wcs import WCS
import numpy as np

import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Plot ------------------------------------------------------------------------

def plot_segments(ax, x, y, gap, **kwargs):
    ''' Plot data x, y to ax, while allowing for some gaps in the data.

    Let’s assume that our data contain groups of points separated by some big
    gaps in x. This functions allows to draw lines between consecutive points
    that are close to each other, while leaving a gap between points that are
    further apart.

    Parameters
    ==========
    ax : matplotlib.axes.Axes
    x : np.ndarray(ndim=1)
    y : np.ndarray(ndim=1)
    gap : float
        the threshold x gap between two consecutive points, above which they
        are not joined with a line
    **kwargs :
        Passed to ax.plot

    Returns
    =======
    lines : list
        a list of matplotlib.lines.Line2d returned by ax.plot

    Example
    =======
    >>> x = np.hstack((
    ...     np.arange(0, 1, 0.1),
    ...     np.arange(3, 4, 0.1),
    ...     ))
    >>> y = np.sin(x)
    >>> plot_segments(ax, x, y, 0.5, marker='.')
    '''

    assert x.shape == y.shape, 'x and y must have the same shape'
    assert len(x.shape), 'x and y must be 1D arrays'

    gap_positions = (x[1:] - x[:-1]) > gap
    gap_positions = np.where(gap_positions)[0]
    gap_positions = np.array([-1] + list(gap_positions) + [len(x)])
    gap_start_stop = zip(gap_positions[:-1] + 1, gap_positions[1:] + 1)
    segments = [(x[start:stop], y[start:stop])
                for start, stop in gap_start_stop]

    if 'color' not in kwargs:
        kwargs['color'] = next(ax._get_lines.prop_cycler)['color']

    # Make sure that isolated points are represented
    user_marker = kwargs.pop('marker', None)
    bundled_points_marker = user_marker
    if user_marker in ['', ' ', 'None', None]:
        single_point_marker = '.'
    else:
        single_point_marker = user_marker

    lines = []
    for segment in segments:
        if len(segment[0]) == 1:
            marker = single_point_marker
        else:
            marker = bundled_points_marker
        line = ax.plot(*segment, marker=marker, **kwargs)
        lines.append(line)

    return lines

# Labels ----------------------------------------------------------------------

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
    locator = mpl.ticker.LogLocator(subs=subs)
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
    ''' Plot log axis labels on an image map '''
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

# Imshow ----------------------------------------------------------------------

def clear_image(ax):
    ''' Clear all images and associated colorbars from axes. '''
    for img in ax.images:
        if img.colorbar:
            img.colorbar.remove()
        img.remove()

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

# Spatial limits --------------------------------------------------------------

def aggressive_autoscale(axes, axis, margin=0.1):
    ''' Autoscale an axis taking into account the limit along the other axis.

    Example use case: set the x-limits, then autoscale the y-axis using only
    the y-data within the x-limits. (Matplotlib's behaviour would be to use the
    full y-data.)

    Parameters
    ==========
    axes : matplotlib axes object
    axis : 'x' or 'y'
        The axis to autoscale
    margin : float
        The margin to add around the data limit, as a fraction of the data
        amplitude.

    Adapted from https://stackoverflow.com/a/35094823/4352108
    '''
    if axis not in ('x', 'y'):
        raise ValueError('invalid axis: '.format(axis))

    # determine axes data limits
    data_limits = []
    for line in axes.get_lines():
        # determine line data limits
        xdata = line.get_xdata()
        ydata = line.get_ydata()
        xmin, xmax = line.axes.get_xlim()
        ymin, ymax = line.axes.get_ylim()
        xdata_displayed = xdata[(ymin < ydata) & (ydata < ymax)]
        ydata_displayed = ydata[(xmin < xdata) & (xdata < xmax)]
        xmin = np.nanmin(xdata_displayed)
        xmax = np.nanmax(xdata_displayed)
        ymin = np.nanmin(ydata_displayed)
        ymax = np.nanmax(ydata_displayed)
        line_limits = (xmin, xmax), (ymin, ymax)
        data_limits.append(line_limits)
    data_limits = np.array(data_limits)
    xmin, ymin = np.min(data_limits, axis=(0, 2))
    xmax, ymax = np.max(data_limits, axis=(0, 2))

    # apply margin
    x_margin = (xmax - xmin) * margin / 2
    y_margin = (ymax - ymin) * margin / 2
    xmin -= x_margin
    xmax += x_margin
    ymin -= y_margin
    ymax += y_margin

    # scale axes
    if axis == 'x':
        axes.set_xlim(xmin, xmax)
    elif axis == 'y':
        axes.set_ylim(ymin, ymax)

def auto_lim(img, x=None, y=None):
    ''' Determine the best x and y lim for displaying non-missing pixels of an
    image.

    Parameters
    ==========
    img : 2D array of shape (ny, nx)
        Image data. If values are numbers, choose limits to display non-NaN
        values. If values are booleans, display pixels containing True values.
    x, y : 1D arrays of resp. shapes (nx, ) and (ny, )
        Coordinates of the pixels.
        If None, use range(nx) and range(ny).

    Returns
    =======
    xlim, ylim : 2-tuples of floats
        The optimal x and y limits for displaying the image.
    '''
    if np.issubdtype(img.dtype, np.bool_):
        displayed_pixels = img
    else:
        displayed_pixels = ~np.isnan(img)
    ny, nx = img.shape
    if x is None:
        x = range(nx)
    if y is None:
        y = range(ny)
    displayed_px_pos = np.where(displayed_pixels)
    displayed_x = x[displayed_px_pos[1]]
    displayed_y = y[displayed_px_pos[0]]
    xlim = displayed_x.min(), displayed_x.max()
    ylim = displayed_y.min(), displayed_y.max()
    return xlim, ylim

def auto_lim_cube(data, x=None, y=None, threshold=0.9):
    ''' Determine the best x and y lim for displaying frames of a cube that
    contain missing NaN data.

    Parameters
    ==========
    data : 3D array of shape (nt, ny, nx)
        Series of image frames.
    x, y : 1D arrays of resp. shapes (nx, ) and (ny, )
        Coordinates of the pixels.
        If None, use range(nx) and range(ny).
    threshold : float between 0 and 1 (default: 0.9)
        Coverage threshold above which a pixel is guaranteed to be within
        the chosen x and y limits. Coverage is defined for each pixel, as
        the fraction of frames (ie along axis 0) in which the pixel has
        non-NaN values.

    Returns
    =======
    xlim, ylim : 2-tuples of floats
        The optimal x and y limits for displaying the frames of the cube.
    '''
    nt = len(data)
    coverage = 1 - np.sum(np.isnan(data), axis=0) / nt
    displayed_pixels = (coverage > (1 - threshold))
    return auto_lim(displayed_pixels, x, y)

# Normalize and colormap ------------------------------------------------------

class SymmetricNormalize(mpl.colors.Normalize):
    """
    An extension of the matplotlib.colors.Normalize class which guarantees that
    the data interval is centered around 0.
    """
    def __init__(self, vlim=None, clip=False):
        """
        If *vlim* is not given, it is initialized from the
        minimum and maximum value of the first input
        processed.  That is, *__call__(A)* calls *autoscale_None(A)*.
        If *clip* is *True* and the given value falls outside the range,
        the returned value will be 0 or 1, whichever is closer.
        Returns 0 if::

            vlim==0

        Works with scalars or arrays, including masked arrays.  If
        *clip* is *True*, masked values are set to 1; otherwise they
        remain masked.  Clipping silently defeats the purpose of setting
        the over, under, and masked colors in the colormap, so it is
        likely to lead to surprises; therefore the default is
        *clip* = *False*.
        """
        if vlim is None:
            vmin = vmax = vlim
        else:
            vmin = -abs(vlim)
            vmax = +abs(vlim)
        super().__init__(vmin=vmin, vmax=vmax, clip=clip)

    def autoscale(self, A):
        """
        Set *vmin* and *vmax* such that *vmax >= 0*, *vmin = -vmax*, and all
        values of *A* fall between *vmin* and *vmax*.
        """
        A = np.asanyarray(A)
        vlim = np.max(np.abs([A.min(), A.max()]))
        self.vmin = -vlim
        self.vmax = +vlim

    def autoscale_None(self, A):
        """Same as autoscale to ensure that *vmin = -vmax*."""
        A = np.asanyarray(A)
        if self.vmin is None and self.vmax is None and A.size:
            self.autoscale(A)

# Colorbar --------------------------------------------------------------------

def get_extend(data, **kwargs):
    ''' Get the extent to pass to plt.colorbar()

    If any vmin or vmax were passed to the plot command, pass them in kwargs
    '''
    vmin = kwargs.get('vmin', None)
    vmax = kwargs.get('vmax', None)
    if vmin is None:
        vmin = np.nan
    if vmax is None:
        vmax = np.nan
    extend_min = np.nanmin(data) < vmin
    extend_max = np.nanmax(data) > vmax
    extend = 'neither'
    if extend_min:
        extend = 'min'
    if extend_max:
        extend = 'max'
    if extend_min and extend_max:
        extend = 'both'
    return extend

def colorbar(ax, mappable, position, size='10%', pad=0.5,
                       aspect=1/30, **kwargs):
    ''' Add a colorbar to axes, ensuring that both are the same size

    The cbar axis is divided with mpl_toolkits.axes_grid1.make_axes_locatable
    to create new axes (cax) where the colorbar is put.

    Parameters
    ==========
    ax : matplotlib axes
    mappable : matplotlib mappable passed to colorbar
    position : str
        The position of the colorbar relatively to ax.
        'left', 'right', 'top', or 'bottom'.
    size : str
        The size of the color bar axes, in percent of the input ax.
    pad : float
        The padding between the ax and cax.
    aspect : float
        The aspect ratio of the cax. Should always be smaller than 1.
    **kwargs :
        Passed to cax.colorbar()

    Returns
    =======
    cb : matplotlib.colorbar.Colorbar
        The newly created colorbar
    '''
    axes_divider = make_axes_locatable(ax)
    cax = axes_divider.append_axes(position, size=size, pad=pad)
    default_orientations = dict(left='vertical', right='vertical',
                                top='horizontal', bottom='horizontal')
    orientation = kwargs.pop('orientation', default_orientations[position])
    cb = ax.figure.colorbar(cax=cax, mappable=mappable,
                            orientation=orientation, **kwargs)
    if orientation == 'vertical':
        cax.set_aspect(1 / aspect)
    else:
        cax.set_aspect(aspect)
    return cb

# Map -------------------------------------------------------------------------

def map_extent(img, coordinates, regularity_threshold=0.01):
    ''' Compute the extent to use in ax.imshow to plot an image with
    coordinates

    Parameters
    ==========
    img : np.ndarray
        An array of shape (n, m) or (n, m, 3) or (n, m, 4).
    coordinates : tuple
        Either a list of bounding coordinates [xmin, xmax, ymin, ymax], pretty
        much like the extent keyword of ax.imshow, or a tuple containing
        two 1D arrays of evenly-spaced x and y values.
    regularity_threshold : float or None (default: 0.01)
        If coordinates is a tuple of x and y values, check that they are evenly
        spaced by ensuring that :
            std(x) / mean(x) < regularity_threshold   and
            std(y) / mean(y) < regularity_threshold
        Raise ValueError('unevenly-spaced array') if this is not verified.
        If regularity_threshold=None, the check is not performed.

    The computed boundaries are the centers of the corresponding pixel. This
    differs from the behaviour of ax.imshow with the extent keyword, where the
    boundaries are the left, right, top, or bottom of the bounding pixels.
    '''
    if img.ndim == 2:
        ny, nx = img.shape
    elif img.ndim == 3:
        ny, nx, ncolor = img.shape
        if not ncolor in (3, 4):
            raise ValueError('invalid array shape')
    else:
        raise ValueError('invalid number of dimensions')

    if len(coordinates) == 4:
        xmin, xmax, ymin, ymax = coordinates
    elif len(coordinates) == 2:
        x, y = coordinates
        if regularity_threshold is not None:
            for s in (x, y):
                ds = s[1:] - s[:-1]
                if np.abs(np.std(ds) / np.mean(ds)) > regularity_threshold:
                    raise ValueError('unevenly-spaced array')
        xmin = x[0]
        xmax = x[-1]
        ymin = y[0]
        ymax = y[-1]
    else:
        raise ValueError('invalid coordinates shape')

    x_step = (xmax - xmin) / (nx - 1)
    y_step = (ymax - ymin) / (ny - 1)
    xmin -= x_step / 2
    xmax += x_step / 2
    ymin -= y_step / 2
    ymax += y_step / 2
    return xmin, xmax, ymin, ymax

def plot_map(ax, arr, coordinates=None, xlog=None, ylog=None,
             regularity_threshold=0.01, **kwargs):
    ''' Plot an image with coordinates

    Parameters
    ==========
    ax : matplotlib.axes.Axes
    arr : np.ndarray
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
    regularity_threshold : float or None (default: 0.01)
        passed to map_extent() (see doc there).
    **kwargs :
        Passed to ax.imshow.

    This function relies on map_extent(), see its docstring.
    '''

    if coordinates:
        extent = map_extent(arr, coordinates,
            regularity_threshold=regularity_threshold)
    else:
        try:
            extent = kwargs.pop('extent')
        except KeyError:
            extent = None

    if 'origin' in kwargs:
        origin = kwargs.pop('origin')
        if origin not in ('lower', 'upper'):
            raise ValueError(f'unsupported origin: {origin}')
        if origin == 'upper':
            arr = arr[::-1]

    img = ax.imshow(
        arr,
        extent=extent,
        origin='lower',
        **kwargs)

    if xlog is not None:
        add_log_labels_to_map(ax, arr, coordinates, extent, xlog, 'x')
    if ylog is not None:
        add_log_labels_to_map(ax, arr, coordinates, extent, ylog, 'y')

    return img

class MapMovie():
    def __init__(self, fig, arr, coordinates=None, interval=50, **kwargs):
        ''' Display or save a movie of maps

        Parameters
        ==========
        fig : matplotlib.figure.Figure
            Matplotlib figure to which the movie is played.
        arr : 3D array
            Movie data, with axes as (time, y, x)
        coordinates : None or 2-tuple
            Passed to papy.plot.plot_map
        interval : number or None (default: 50)
            Delay between frames in milliseconds. (Passed to
            mpl.animation.FuncAnimation.)
        **kwargs :
            Passed to fig.gca().imshow through papy.plot_map.

        Methods
        =======
        play() :
            Display the movie in a matplotlib window
        save(path) :
            Save a mp4 movie.
        '''

        self.fig = fig
        self.ax = None

        self.arr = arr
        self.coordinates = coordinates

        # Args passed to matplotlib imshow
        self.imshow_kwargs = kwargs

    def init_plot(self):
        self.fig.clear()
        self.ax = self.fig.gca()
        self.im = plot_map(
            self.ax,
            self.arr[0],
            coordinates=self.coordinates,
            animated=True,
            **self.imshow_kwargs)
        self.cbar = self.fig.colorbar(self.im)

    def update(self, i):
        self.im.set_data(self.arr[i])
        self.ax.set_title('Step {:03d}'.format(i))

        # Set vmin and vmax from current frame data if no norm, vlim, or vmax
        # are passed as kwargs.
        if 'norm' not in self.imshow_kwargs:
            try:
                vmin = self.imshow_kwargs['vmin']
            except KeyError:
                vmin = np.nanmin(self.arr[i])
            try:
                vmax = self.imshow_kwargs['vmax']
            except KeyError:
                vmax = np.nanmax(self.arr[i])
            self.im.set_clim(vmin, vmax)

        return self.im,

    def play(self):
        ''' Play the movie in a matplotlib window. '''
        self.init_plot()
        self.anim = mpl.animation.FuncAnimation(
            self.fig,
            self.update,
            frames=len(self.arr),
            interval=self.interval,
            )

    def save(self, filename, fps=15, bitrate=1800, **kwargs):
        ''' Save a mp4 movie to `filename`. kwargs are passed to the matplotlp
        ffmpeg writer '''
        try:
            self.anim
        except AttributeError:
            self.play()
        Writer = mpl.animation.writers['ffmpeg']
        writer = Writer(fps=fps, bitrate=bitrate)
        self.anim.save(filename, writer=writer)

# FITS ------------------------------------------------------------------------

def wcs_transform(wcs):
    ''' Get the matplotlib affine transform from a WCS object

    Parameters
    ==========
    wcs : astropy.wcs.WCS
        The WCS object from which to determine the transform.

    Returns
    =======
    transform : matplotlib.transforms.Affine2D
        The affine transformation from wcs.
    '''
    if wcs.has_distortion:
        raise ValueError('distortion not supported')
    matrix_linear = wcs.pixel_scale_matrix * 3600
    tr_x, tr_y = wcs.all_world2pix(0, 0, 0)
    matrix_affine = np.zeros((3, 3))
    matrix_affine[:2, :2] = matrix_linear
    matrix_affine[2, 2] = 1
    matrix_affine = np.linalg.inv(matrix_affine)
    matrix_affine[0, 2] = tr_x
    matrix_affine[1, 2] = tr_y
    matrix_affine = np.linalg.inv(matrix_affine)
    return mpl.transforms.Affine2D(matrix_affine)

def set_wcs_transform(artist, wcs):
    ''' Transform an artist with a WCS object

    Parameters
    ==========
    artist : matplotlib artist
        The object to transform (can be image, lines, etc.).
    wcs : astropy.wcs.WCS
        The WCS object used to transform the artist.
        See also `wcs_transform()`
    '''
    trans_data = wcs_transform(wcs) + artist.axes.transData
    artist.set_transform(trans_data)

def plot_image_hdu(ax, hdu, **kwargs):
    ''' Plot the image from a FITS HDU, taking into account affine WCS
    transforms.

    Parameters
    ==========
    ax : matplotlib axis
    hdu : an astropy HDU object
        The HDU containing a 2D image data and a header with WCS keywords.
    **kwargs :
        Passed to ax.imshow.
    '''
    im = ax.imshow(hdu.data, **kwargs)
    set_wcs_transform(im, WCS(hdu.header))
    return im

# Pixel contours --------------------------------------------------------------

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

    inspired from https://stackoverflow.com/a/24540564/4352108
    '''

    if not np.any(mask):
        return

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
        x_step = (xmax - xmin) / nx
        y_step = (ymax - ymin) / ny
        xmin += x_step / 2
        ymin += y_step / 2
    else:
        xmin, ymin = 0, 0
        x_step, y_step = 1, 1

    # shape = np.array(mask.shape)
    # larger_mask = np.zeros(shape + (2, 2), dtype=bool)
    # larger_mask[1:-1, 1:-1] = mask
    # mask = larger_mask

    horizontal_segments = np.array(np.where((mask[:, 1:] != mask[:, :-1]))).T
    vertical_segments = np.array(np.where(mask[1:, :] != mask[:-1, :])).T

    segments = []
    for x, y in horizontal_segments:
        segments.append((x, y+1))
        segments.append((x+1, y+1))
        segments.append((np.nan, np.nan))
    for x, y in vertical_segments:
        segments.append((x+1, y))
        segments.append((x+1, y+1))
        segments.append((np.nan, np.nan))
    seg_y, seg_x = np.array(segments).T

    seg_x = xmin + x_step * (seg_x - .5)
    seg_y = ymin + y_step * (seg_y - .5)

    return ax.plot(seg_x, seg_y, *args, **kwargs)


# 2D histograms ---------------------------------------------------------------

def cube_hist(ax, cube, bins=100, **kwargs):
    ''' Plot a 2D image, where lines are the histograms of each frames of an
    input cube.  '''

    # compute histograms
    cube = np.ma.array(cube, mask=np.isnan(cube))
    hist_range = (cube.min(), cube.max())
    _, bin_edges = np.histogram(
        cube.compressed(), range=hist_range, bins=bins)
    frames_hist = np.array([
        np.histogram(frame.compressed(), range=hist_range, bins=bins)[0]
        for frame in cube])

    # prepare x and y coordinates
    x = (bin_edges[1:] + bin_edges[:-1]) / 2
    y = np.arange(frames_hist.shape[0])

    # plot histograms for each frame
    # frames_hist = np.ma.array(frames_hist, mask=np.isinf(frames_hist)).filled(0)
    img = plot_map(
        ax,
        frames_hist,
        coordinates=(x, y),
        aspect=(x.max() - x.min()) / (y.max() - y.min()),
        **kwargs)
    cbar = ax.figure.colorbar(img)
    ax.set_xlabel('value')
    ax.set_ylabel('frame')
    cbar.set_label('count')
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())

    return img
