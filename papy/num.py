''' Numerical tools, and probably the messiest file. '''

import itertools
import multiprocessing as mp
import warnings

import numpy as np
import scipy.interpolate as si

# Manipulate array dimensions -------------------------------------------------

def almost_identical(arr, threshold, **kwargs):
    ''' Reduce an array of almost identical values to a single one.

    Parameters
    ==========
    arr : np.ndarray
        An array of almost identical values.
    threshold : float
        The maximum standard deviation that is tolerated for the values in arr.
    **kwargs :
        Passed to np.std and np.average. Can be used to reduce arr across a
        choosen dimension.

    Raises
    ======
    ValueError if the standard deviation of the values in arr exceedes the
    specified threshold value

    Returns
    =======
    average : float or np.ndarray
        The average value of arr.
    '''

    irregularity = np.std(arr, **kwargs)
    if np.any(irregularity > threshold):
        msg = 'Uneven array:\n'
        irr_stats = [
            ('irregularity:', irregularity),
            ('irr. mean:', np.mean(irregularity)),
            ('irr. std:', np.std(irregularity)),
            ('irr. min:', np.min(irregularity)),
            ('irr. max:', np.max(irregularity)),
            ]
        for title, value in irr_stats:
            msg += '{} {}\n'.format(title, value)
        msg += 'array percentiles:\n'
        percentiles = [0, 1, 25, 50, 75, 99, 100]
        for percentile in percentiles:
            m = '{: 5d}: {:.2f}\n'
            m = m.format(percentile, np.percentile(arr, percentile))
            msg += m
        raise ValueError(msg)

    return np.average(arr, **kwargs)

def chunks(array, n):
    ''' Split array chunks of size n.

    http://stackoverflow.com/a/1751478/4352108
    '''
    n = max(1, n)
    return (array[i:i+n] for i in range(0, len(array), n))

def rebin(arr, binning, cut_to_bin=False, method=np.sum):
    ''' Rebin an array by summing its pixels values.

    Parameters
    ==========
    arr : np.ndarray
        The numpy array to bin. The array dimensions must be divisible by the
        requested binning.
    binning : tuple
        A tuple of size arr.ndim containing the binning. This could be (2, 3)
        to perform binning 2x3.
    cut_to_bin : bool (default: False)
        If set to true, and the dimensions of `arr` are not multiples of
        `binning`, clip `arr`, and still bin it.
    method : function (default: np.sum)
        Method to use when gathering pixels values. This value must take a
        np.ndarray as first argument, and accept kwarg `axis`.

    Partly copied from <https://gist.github.com/derricw/95eab740e1b08b78c03f>.
    '''
    new_shape = np.array(arr.shape) // np.array(binning)
    new_shape_residual = np.array(arr.shape) % np.array(binning)
    if np.any(new_shape_residual):
        m = 'Bad binning {} for array with dimension {}.'
        m = m.format(binning, arr.shape)
        if cut_to_bin:
            m += ' Clipping array to {}.'
            m = m.format(tuple(np.array(arr.shape) - new_shape_residual))
            print(m)
            new_slice = [slice(None, -i) if i else slice(None)
                         for i in new_shape_residual]
            arr = arr[new_slice]
        else:
            raise ValueError(m)

    compression_pairs = [
        (d, c//d) for d, c in zip(new_shape, arr.shape)]
    flattened = [l for p in compression_pairs for l in p]
    arr = arr.reshape(flattened)
    axis_to_sum = (2*i + 1 for i in range(len(new_shape)))
    arr = method(arr, axis=tuple(axis_to_sum))

    assert np.all(arr.shape == new_shape)

    return arr

def roll_nd(arr, shifts=None):
    ''' Roll a n-D array along its axes.

    (A wrapper around np.roll, for n-D array.)

    Parameters
    ==========
    arr : array_like
        Input array.
    shifts : tuple of ints or None (default: None)
        Tuple containing, for each axis, the number of places by which elements
        are shifted along this axes. If a value of this tuple is None, elements
        of the corresponding axis are shifted by half the axis length.

        If None, shift all axes by half their respective lengths.

    Returns
    =======
    output : ndarray
        Array with the same shape as the input array.

    Example
    =======
    >>> a = np.arange(9).reshape((3,3))
    >>> a
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])
    >>> roll_2d(a, (1, 1))
    array([[8, 6, 7],
           [2, 0, 1],
           [5, 3, 4]])

    '''
    if shifts is None:
        shifts = np.array(arr.shape) // 2
    for i, shift in enumerate(shifts):
        if shift is None:
            shift = arr.shape[i] // 2
        arr = np.roll(arr, shift, axis=i)
    return arr

# NaN in arrays ---------------------------------------------------------------

def add_nan_border(arr, size=1):
    ''' Create a larger array containing the original array surrounded by nan

    Parameters
    ==========
    array : ndarray
    size : int (default: 1)
        The size of the nan border. This border will be added at both ends of
        each axis.

    Returns
    =======
    new_array : ndarray
        A new array of shape `array.shape + 2*size`
    '''
    shape = np.array(arr.shape)
    new_arr = np.ones(shape + 2*size) * np.nan
    slice_ = [slice(+size, -size)] * arr.ndim
    new_arr[slice_] = arr
    return new_arr

def stack_arrays(arr_list, fill_value=np.nan):
    ''' Stack arrays of different shapes in a larger array, where smaller
    arrays are padded with a fill_value.

    Parameters
    ==========
    arr_list : list of array-likes
        A list of n-dimension arrays, that can have different shapes.
    fill_value : float (default: np.nan)
        Value with which smaller arrays are padded.

    Returns
    =======
    new_arr : ndarray
        A (n+1)-dimension array that has as many items as `arr_list` along its
        first dimension, and the largest size found across all elements of
        `arr_list` for all subsequent dimensions.

    Example
    =======
    >>> a = np.full((2, 5), 1)
    array([[1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1]])
    >>> b = np.full((3, 2), 2)
    array([[2, 2],
           [2, 2],
           [2, 2]])
    >>> num.stack_arrays((a, b), fill_value=0)
    array([[[1, 1, 1, 1, 1],
	    [1, 1, 1, 1, 1],
	    [0, 0, 0, 0, 0]],

	   [[2, 2, 0, 0, 0],
	    [2, 2, 0, 0, 0],
	    [2, 2, 0, 0, 0]]])
    '''

    ndims = [arr.ndim for arr in arr_list]
    if len(set(ndims)) != 1:
        raise ValueError('Arrays must have the same number of dimensions')
    shapes = np.array([arr.shape for arr in arr_list])
    nitems = len(arr_list)
    new_shape = [nitems, *np.max(shapes, axis=0)]

    new_arr = np.full(new_shape, fill_value)

    for i, arr in enumerate(arr_list):
        slice_ = [i] + [slice(None, s) for s in arr.shape]
        new_arr[tuple(slice_)] = arr

    return new_arr

# Affine transforms -----------------------------------------------------------

def affine_transform(x, y, transform_matrix, center=(0, 0)):
    ''' Apply an affine transform to an array of coordinates.

    Parameters
    ==========
    x, y : arrays with the same shape
        x and y coordinates to be transformed
    transform_matrix : array_like with shape (2, 3)
        The matrix of the affine transform, [[A, B, C], [D, E, F]]. The new
        coordinates (x', y') are computed from the input coordinates (x, y)
        as follows:

            x' = A*x + B*y + C
            y' = D*x + E*y + F

    center : 2-tulpe of floats (default: (0, 0))
        The center of the transformation. In particular, this is useful for
        rotating arrays around their central value and not the origin.

    Returns
    =======
    transformed_x, transformed_y : arrays
        Arrays with the same shape as the input x and y, and with their values
        transformed by `transform_matrix`.
    '''

    # build array of coordinates, where the 1st axis contains (x, y, ones)
    # values
    ones = np.ones_like(x)
    coordinates = np.array((x, y, ones))

    # transform transform_matrix from (2, 3) to (3, 3)
    transform_matrix = np.vstack((transform_matrix, [0, 0, 1]))

    # add translation to and from the transform center to
    # transformation_matrix
    x_cen, y_cen = center
    translation_to_center = np.array([
        [1, 0, - x_cen],
        [0, 1, - y_cen],
        [0, 0, 1]])
    translation_from_center = np.array([
        [1, 0, x_cen],
        [0, 1, y_cen],
        [0, 0, 1]])
    transform_matrix = np.matmul(transform_matrix, translation_to_center)
    transform_matrix = np.matmul(translation_from_center, transform_matrix)

    # apply transform
    # start with coordinates of shape : (3, d1, ..., dn)
    coordinates = coordinates.reshape(3, -1) # (3, N) N = product(d1, ..., dn)
    coordinates = np.moveaxis(coordinates, 0, -1) # (N, 3)
    coordinates = coordinates.reshape(-1, 3, 1) # (N, 3, 1)
    new_coordinates = np.matmul(transform_matrix, coordinates) # (N, 3, 1)
    new_coordinates = new_coordinates.reshape(-1, 3) # (N, 3)
    new_coordinates = np.moveaxis(new_coordinates, -1, 0) # (3, N)
    new_coordinates = new_coordinates.reshape(3, *ones.shape)
    transformed_x, transformed_y, _ = new_coordinates

    return transformed_x, transformed_y

# Running stuff ---------------------------------------------------------------

def running_average(arr, n):
    ''' Compute a non-weighted running average on arr, with a window of width
    2n+1 centered on each value:

    ret_i = \sum_{j=-n}^n arr_{i+j} / (2n+1)
    '''
    cumsum = np.cumsum(arr)
    ret = arr.copy()
    ret[n:len(arr)-n] = (cumsum[2*n:] - cumsum[:len(arr)-2*n]) / (2*n)
    return ret

def running_median(arr, mask):
    ''' Compute a running median, after applying mask shifted to align its
    central value with each points.

    arr : 1D array
    mask : 1D array, smaller than arr and with 2n+1 elements
    '''

    n = len(arr)
    m = len(mask)
    assert m % 2 == 1, 'mask must have an odd number of events'

    full_mask = np.empty_like(arr)
    full_mask[:] = np.nan
    full_mask[:m] = mask

    ret = np.empty_like(arr)
    for x in range(n):
        m = np.roll(full_mask, x - m // 2)
        if x < m // 2:
            m[- m // 2 + x + 1:] = np.nan
        if n - x <= m // 2:
            m[:m // 2 - n + x + 1] = np.nan
        ret[x] = np.nanmedian(arr * m)

    return ret

def weighted_running_average(arr, weight_func, x=None):
    ''' Compute a running average on arr, weighting the contribution of each
    term with weight-function.

    Parameters
    ==========
    arr : np.ndarray(ndim=1)
        Array of values on which to compute the running average.
    weight_func : function
        A function which takes a distance as input, and returns a weight.
        Weights don't have to be normalised.
    x : np.ndarray or None (default: None)
        If x is an array, use it to compute the distances before they are
        passed to weight_func. This allows to compute a running average on
        non-regularly sampled data.

    Returns
    =======
    ret : np.ndarray
        An array of the same shape as the input arr, equivalent to:
        - when x is None:
            $ret_i = \sum_{j=-n}^n arr_{i+j} × w(j) / (2n+1)$
        - when x is specified:
            $ret_i = \sum_{j=-n}^n arr_{i+j} × w(|x_i - x_{i+j}|) / (2n+1)$.
    '''
    return weighted_running_function(arr, weight_func, np.mean, x=x)

def weighted_running_function(arr, weight_func, function, x=None):
    ''' Apply a running function on arr, weighting the contribution of each
    term with weight-function. This is used to compute running averages, stds,
    medians, etc.

    Parameters
    ==========
    arr : np.ndarray(ndim=1)
        Array of values on which to compute the running average.
    weight_func : function
        A function which takes a distance as input, and returns a weight.
        Weights don't have to be normalised.
    function : np.ufunc
        A function used to reduce the dimension of a 2D array down to 1D. It
        must accept an array and an `axis` kwargs as input, and output an
        array.
    x : np.ndarray or None (default: None)
        If x is an array, use it to compute the distances before they are
        passed to weight_func. This allows to compute a running average on
        non-regularly sampled data.

    Returns
    =======
    ret : np.array
        An array of the same shape as the input arr, to which `function` has
        been applied after weighting the terms with weight_func.
    '''
    weight_func = np.vectorize(weight_func)

    n = len(arr)
    if x is None:
        x = np.arange(n)

    distances = x.repeat(n).reshape(-1, n).T
    distances = np.abs(np.array([d - d[i] for i, d in enumerate(distances)]))
    weights = weight_func(distances)
    norm = n / weights.sum(axis=1).repeat(n).reshape(-1, n)
    weights *= norm

    ret = arr.copy()
    ret = arr.repeat(n).reshape(-1, n).T
    ret *= weights
    ret = function(ret, axis=1)

    return ret

# Interpolation ---------------------------------------------------------------

def get_griddata_points(grid):
    ''' Retrieve points in mesh grid of coordinates, that are shaped for use
    with scipy.interpolate.griddata.

    Parameters
    ==========
    grid : np.ndarray
        An array of shape (2, x_dim, y_dim) containing (x, y) coordinates.
        (This should work with more than 2D coordinates.)
    '''
    if isinstance(grid, (list, tuple)):
        grid = np.array(grid)
    points = np.array([
        grid[i].flatten() for i in range(grid.shape[0])])
    return points

def friendly_griddata(points, values, new_points, **kwargs):
    ''' A friendly wrapper around scipy.interpolate.griddata.

    Parameters
    ==========
    points : tuple
        Data point coordinates. This is a tuple of ndim arrays, each having the
        same shape as `values`, and each containing data point coordinates
        along a given axis.
    values : array
        Data values. This is an array of dimension ndim.
    new_points : tuple
        Points at which to interpolate data. This has the same structure as
        `points`, but not necessarily the same shape.
    kwargs :
        passed to scipy.interpolate.griddata
    '''
    new_shape = new_points[0].shape
    # make values griddata-friendly
    points = get_griddata_points(points)
    values = values.flatten()
    new_points = get_griddata_points(new_points)
    # projection
    new_values = si.griddata(
        points.T, values, new_points.T,
        **kwargs)
    # make values user-friendly
    new_values = new_values.reshape(*new_shape)
    return new_values

def _interpolate_cube_worker(values, coord, new_coord, method):
    ''' Worker to paralellize the projection of a cube slice. '''
    points = np.stack(coord).reshape(2, -1).T
    values = values.flatten()
    new_points = np.stack(new_coord).reshape(2, -1).T
    new_values = si.griddata(points, values, new_points, method=method)
    new_values = new_values.reshape(*new_coord[0].shape)
    return new_values

def interpolate_cube(cube, coord, new_coord, method='linear', cores=None):
    ''' Transform time series of 2D data one coordinate system to another.

    The cube is cut in slices along its axis 0, and the interpolation is
    performed within each slice. Hence, there is no inter-slice interpolation.

    The projection is performed by scipy.interpolate.griddata.

    Parameters
    ==========
    cube : np.ndarray
        A cube containing data on a (T, Y, X) grid.
    coord, new_coord: 2-tuples of np.ndarray
        3D arrays containing the coordinates values. Shape (nt, ny, nx).
        - coord (x, y) correspond to the coordinates in the input cube.
        - new_coord (new_x, new_y) correspond to the expected coordinates for
          the output cube.
        - there are no restrictions on the regularity of these arrays.
          Eg. the coordinates could be either helioprojective or heliographic
          coordinates.
    method : str (default: linear)
        The method to use for the projection, passed to
        scipy.interpolate.griddata.
    cores : float or None (default: None)
        If not None, use multiprocessing.Pool to parallelize the projections

    Returns
    =======
    new_cube : np.ndarray
        A new cube, containing values interpolated at new_x, new_y positions.
    '''

    x, y = coord
    new_x, new_y = new_coord

    if cube.ndim == 2:
        new_cube = _interpolate_cube_worker(cube, coord, new_coord, method)

    else:
        if cores is not None:
            pool = mp.Pool(cores)
            try:
                new_cube = pool.starmap(
                    _interpolate_cube_worker,
                    zip(cube, x, y, new_x, new_y, itertools.repeat(method)),
                    chunksize=1)
            finally:
                pool.terminate()
        else:
            new_cube = list(itertools.starmap(
                _interpolate_cube_worker,
                zip(cube, x, y, new_x, new_y, itertools.repeat(method))))

    return np.array(new_cube)

def frame_to_cube(frame, n):
    ''' Repeat a frame n times to get a cube of shape (n, *arr.shape). '''
    cube = np.repeat(frame, n)
    nx, ny = frame.shape
    cube = cube.reshape(nx, ny, n)
    cube = np.moveaxis(cube, -1, 0)
    return cube

def replace_missing_values(arr, missing, inplace=False, deg=1):
    ''' Interpolate missing elements in a 1D array using a polynomial
    interpolation from the non-missing values.

    Parameters
    ==========
    arr : np.ndarray
        The 1D array in which to replace the element.
    missing : np.ndarray
        A boolean array where the missing elements are marked as True.
    inplace : bool (default: False)
        If True, perform operations in place. If False, copy the array before
        replacing the element.
    deg : int (default: 1)
        The degree of the polynome used for the interpolation.

    Returns
    =======
    arr : np.ndarray
        Updated array.
    '''

    assert arr.ndim == 1, 'arr must be 1D'
    assert arr.shape == missing.shape, \
        'arr and missing must have the same shape'
    assert not np.all(missing), \
        'at least one element must not be missing'

    if not inplace:
        arr = arr.copy()

    x = np.arange(len(arr))
    poly_params = np.polyfit(x[~missing], arr[~missing], deg)
    poly = np.poly1d(poly_params)
    arr[missing] = poly(x[missing])

    return arr

def exterpolate_nans(arr, deg):
    ''' Exterminate nans in an array by replacing them with values extrapolated
    using a polynomial fit.

    Parameters
    ==========
    arr : 1D ndarray
        Array containing the nans to remove.
    deg : int
        Degree of the polynome used to fit the data in array.
    '''
    msg = 'expected 1 dimension for arr, got {}'
    assert arr.ndim == 1, msg.format(arr.ndim)
    nan_mask = np.isnan(arr)
    x = np.arange(len(arr))
    poly_params = np.polyfit(x[~nan_mask], arr[~nan_mask], deg)
    poly = np.poly1d(poly_params)
    arr = arr.copy()
    arr[nan_mask] = poly(x[nan_mask])
    return arr

def exterpolate_nans_in_rows(arr, deg):
    ''' Apply exterpolate_nans() to each row of arr '''
    arr = arr.copy()
    for i, row in enumerate(arr):
        arr[i] = exterpolate_nans(row, deg)
    return arr

# Misc ------------------------------------------------------------------------

def recarray_to_dict(recarray, lower=False):
    ''' Transform a 1-row recarray to a dictionnary.

    if lower, lower all keys
    '''
    while recarray.dtype is np.dtype('O'):
        recarray = recarray[0]
    assert len(recarray) == 1, 'structure contains more than one row!'
    array = dict(zip(recarray.dtype.names, recarray[0]))
    if lower:
        array = {k.lower(): v for k, v in array.items()}
    return array

def ma_delete(arr, obj, axis=None):
    ''' Wrapper around np.delete to handle masked arrays. '''
    if isinstance(arr, np.ma.MaskedArray):
        return np.ma.array(
            np.delete(arr.data, obj, axis=axis),
            mask=np.delete(arr.mask, obj, axis=axis),
            fill_value=arr.fill_value,
            )
    return np.delete(arr, obj, axis=axis)

def get_max_location(arr, sub_px=True):
    ''' Get the location of the max of an array.

    Parameters
    ==========
    arr : ndarray
    sub_px : bool (default: True)
        whether to perform a parabolic interpolation about the maximum to find
        the maximum with a sub-pixel resolution.

    Returns
    =======
    max_loc : 1D array
        Coordinates of the maximum of the input array.
    '''
    maxcc = np.nanmax(arr)
    if np.isnan(maxcc):
        return np.array([np.nan] * arr.ndim)
    max_px = np.where(arr == maxcc)
    if not np.all([len(m) == 1 for m in max_px]):
        warnings.warn('could not find a unique maximum', RuntimeWarning)
    max_px = np.array([m[0] for m in max_px])
    max_loc = max_px.copy()

    if sub_px:
        max_loc = max_loc.astype(float)
        for dim in range(arr.ndim):
            arr_slice = list(max_px)
            dim_max = max_px[dim]
            if dim_max in (0, arr.shape[dim] - 1):
                m = 'maximum is on the edge of axis {}'.format(dim)
                warnings.warn(m, RuntimeWarning)
                max_loc[dim] = dim_max
            else:
                arr_slice[dim] = [dim_max-1, dim_max, (dim_max+1)]
                interp_points = arr[tuple(arr_slice)]
                a, b, c = interp_points**2
                d = a - 2*b + c
                if d != 0 and not np.isnan(d):
                    max_loc[dim] = dim_max - (c-b)/d + 0.5

    return max_loc

def clip_to_nan(arr, t_low=None, t_high=None):
    ''' Replace the values in `arr` that are below `t_low` or above `t_high`
    with np.nan.

    Input
    =====
    arr : np.ndarray
        The array to clip.
    t_low : float
        The lower threshold. If None, use no lower limit.
    t_high : float
        The higher threshold. If None, use no higher limit.

    Returns
    =======
    A np.ndarray of the same dimension as `arr`, where values below `t_low`
    or above `t_high` are set to `np.nan`.

    Note
    ====
    Will raise `ValueError` if `arr` contains integers, since `np.nan` cannot be
    converted to `int`.

    '''
    # create mask
    mask = np.zeros_like(arr, dtype=bool)
    if t_low:
        mask |= arr < t_low
    if t_high:
        mask |= arr > t_high
    # mask array
    arr = np.ma.array(arr, mask=mask)
    # fill with np.nan
    arr = arr.filled(fill_value=np.nan)
    return arr
