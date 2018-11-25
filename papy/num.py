#!/usr/bin/env python3

import datetime
import dateutil.parser
import itertools
import multiprocessing as mp
import warnings

import numpy as np
import scipy.interpolate as si
import scipy.signal

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

    compression_pairs = [(d, c//d)
        for d, c in zip(new_shape, arr.shape)]
    flattened = [l for p in compression_pairs for l in p]
    arr = arr.reshape(flattened)
    axis_to_sum = (2*i + 1 for i in range(len(new_shape)))
    arr = method(arr, axis=tuple(axis_to_sum))

    assert np.all(arr.shape == new_shape)

    return arr

def chunks(l, n):
    ''' Split list l in chunks of size n.

    http://stackoverflow.com/a/1751478/4352108
    '''
    n = max(1, n)
    return (l[i:i+n] for i in range(0, len(l), n))

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

def running_average(arr, n):
    ''' Compute a non-weighted running average on arr, with a window of width
    2n+1 centered on each value:

    ret_i = \sum_{j=-n}^n arr_{i+j} / (2n+1)
    '''
    N = len(arr)
    cs = np.cumsum(arr)
    ret = arr.copy()
    ret[n:N-n] = (cs[2*n:] - cs[:N-2*n]) / (2*n)
    return ret

def running_median(arr, mask):
    ''' Compute a running median, after applying mask shifted to align its
    central value with each points.

    arr : 1D array
    mask : 1D array, smaller than arr and with 2n+1 elements
    '''

    N = len(arr)
    M = len(mask)
    assert M % 2 == 1, 'mask must have an odd number of events'

    full_mask = np.empty_like(arr)
    full_mask[:] = np.nan
    full_mask[:M] = mask

    ret = np.empty_like(arr)
    for x in range(N):
        m = np.roll(full_mask, x - M // 2)
        if x < M // 2:
            m[- M // 2 + x + 1:] = np.nan
        if N - x <= M // 2:
            m[:M // 2 - N + x + 1] = np.nan
        ret[x] = np.nanmedian(arr * m)

    return ret

def weighted_running_average(arr, wf, x=None):
    ''' Compute a running average on arr, weighting the contribution of each
    term with weight-function wf.

    Parameters
    ==========
    arr : np.ndarray(ndim=1)
        Array of values on which to compute the running average.
    wf : function
        A function which takes a distance as input, and returns a weight.
        Weights don't have to be normalised.
    x : np.ndarray or None (default: None)
        If x is an array, use it to compute the distances before they are
        passed to wf. This allows to compute a running average on non-regularly
        sampled data.

    Returns
    =======
    ret : np.ndarray
        An array of the same shape as the input arr, equivalent to:
        - when x is None:
            $ret_i = \sum_{j=-n}^n arr_{i+j} × w(j) / (2n+1)$
        - when x is specified:
            $ret_i = \sum_{j=-n}^n arr_{i+j} × w(|x_i - x_{i+j}|) / (2n+1)$.
    '''
    return weighted_running_function(arr, wf, np.mean, x=x)

def weighted_running_function(arr, wf, function, x=None):
    ''' Apply a running function on arr, weighting the contribution of each
    term with weight-function wf. This is used to compute running averages,
    stds, medians, etc.

    Parameters
    ==========
    arr : np.ndarray(ndim=1)
        Array of values on which to compute the running average.
    wf : function
        A function which takes a distance as input, and returns a weight.
        Weights don't have to be normalised.
    function : np.ufunc
        A function used to reduce the dimension of a 2D array down to 1D. It
        must accept an array and an `axis` kwargs as input, and output an
        array.
    x : np.ndarray or None (default: None)
        If x is an array, use it to compute the distances before they are
        passed to wf. This allows to compute a running average on non-regularly
        sampled data.

    Returns
    =======
    ret : np.array
        An array of the same shape as the input arr, to which `function` has
        been applied after weighting the terms with wf.
    '''
    wf = np.vectorize(wf)

    N = len(arr)
    if x is None:
        x = np.arange(N)

    distances = x.repeat(N).reshape(-1, N).T
    distances = np.abs(np.array([d - d[i] for i, d in enumerate(distances)]))
    weights = wf(distances)
    norm = N / weights.sum(axis=1).repeat(N).reshape(-1, N)
    weights *= norm

    ret = arr.copy()
    ret = arr.repeat(N).reshape(-1, N).T
    ret *= weights
    ret = function(ret, axis=1)

    return ret

def datetime_average(a, b):
    ''' Average function that is friendly with datetime formats that only
    support substraction. '''
    return a + (b - a) / 2

def ma_delete(arr, obj, axis=None):
    ''' Wrapper around np.delete to handle masked arrays. '''
    if isinstance(arr, np.ma.MaskedArray):
        return np.ma.array(
            np.delete(arr.data, obj, axis=axis),
            mask=np.delete(arr.mask, obj, axis=axis),
            fill_value=arr.fill_value,
            )
    else:
        return np.delete(arr, obj, axis=axis)

def mask_nans(arr):
    ''' Create a masked version of arr, where the nans are hidden
    '''
    return np.ma.array(arr, mask=np.isnan(arr))

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
    sl = [slice(+size, -size)] * arr.ndim
    new_arr[sl] = arr
    return new_arr

def rms(arr, axis=None):
    ''' Compute root mean square of arr: sqrt(average(arr²)). '''
    return np.sqrt(np.average(arr**2, axis=axis))

def normalized_periodogram(x, y, f):
    ''' Returns a normalized Lomb-Scargle periodogram.

    Wrapper around scipy.signal.lombscargle.

    This normalization is meaningful if len(x) is large enough.
    '''
    s = scipy.signal.lombscargle(x, y, f)
    return np.sqrt(4 * s / len(x))

def get_griddata_points(grid):
    ''' Retrieve points in mesh grid of coordinates, that are shaped for use
    with scipy.interpolate.griddata.

    Parameters
    ==========
    grid : np.ndarray
        An array of shape (2, x_dim, y_dim) containing (x, y) coordinates.
        (This should work with more than 2D coordinates.)
    '''
    if type(grid) in [list, tuple]:
        grid = np.array(grid)
    points = np.array([grid[i].flatten()
        for i in range(grid.shape[0])])
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

def _interpolate_cube_worker(values, x, y, new_x, new_y, method):
    ''' Worker to paralellize the projection of a cube slice. '''
    points = np.stack((x, y)).reshape(2, -1).T
    values = values.flatten()
    new_points = np.stack((new_x, new_y)).reshape(2, -1).T
    new_values = si.griddata(points, values, new_points, method=method)
    new_values = new_values.reshape(*new_x.shape)
    return new_values

def interpolate_cube(cube, x, y, new_x, new_y, method='linear', cores=None):
    ''' Transform time series of 2D data one coordinate system to another.

    The cube is cut in slices along its axis 0, and the interpolation is
    performed within each slice. Hence, there is no inter-slice interpolation.

    The projection is performed by scipy.interpolate.griddata.

    Parameters
    ==========
    cube : np.ndarray
        A cube containing data on a (T, Y, X) grid.
    x, y, new_x, new_y : np.ndarray
        3D arrays containing the coordinates values. Shape (nt, ny, nx).
        - x, y correspond to the coordinates in the input cube.
        - new_x, new_y correspond to the expected coordinates for the output
          cube.
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

    if cube.ndim == 2:
        new_cube = _interpolate_cube_worker(cube, x, y, new_x, new_y, method)

    else:
        if cores is not None:
            p = mp.Pool(cores)
            try:
                new_cube = p.starmap(
                    _interpolate_cube_worker,
                    zip(cube, x, y, new_x, new_y, itertools.repeat(method)),
                    chunksize=1)
            finally:
                p.terminate()
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
    npx = len(arr)

    if not inplace:
        arr = arr.copy()

    x = np.arange(len(arr))
    c = np.polyfit(x[~missing], arr[~missing], deg)
    p = np.poly1d(c)
    arr[missing] = p(x[missing])

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
    c = np.polyfit(x[~nan_mask], arr[~nan_mask], deg)
    p = np.poly1d(c)
    arr = arr.copy()
    arr[nan_mask] = p(x[nan_mask])
    return arr

def exterpolate_nans_in_rows(arr, deg):
    ''' Apply exterpolate_nans() to each row of arr '''
    arr = arr.copy()
    for i, row in enumerate(arr):
        arr[i] = exterpolate_nans(row, deg)
    return arr

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
        for p in percentiles:
            msg += '{: 5d}: {:.2f}\n'.format(p, np.percentile(arr, p))
        raise ValueError(msg)

    return np.average(arr, **kwargs)

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
            if dim_max == 0 or dim_max == arr.shape[dim] - 1:
                m = 'maximum is on the edge of axis {}'.format(dim)
                warnings.warn(m, RuntimeWarning)
                max_loc[dim] = dim_max
            else:
                arr_slice[dim] = [dim_max-1, dim_max, (dim_max+1)]
                interp_points = arr[arr_slice]
                a, b, c = interp_points**2
                d = a - 2*b + c
                if d != 0 and not np.isnan(d):
                    max_loc[dim] = dim_max - (c-b)/d + 0.5

    return max_loc

@np.vectorize
def total_seconds(timedelta):
    return timedelta.total_seconds()

@np.vectorize
def parse_date(date):
    return dateutil.parser.parse(date)

def seconds_to_timedelta(arr):
    ''' Parse an array of seconds and convert it to timedelta.
    '''
    to_timedelta = np.vectorize(lambda s: datetime.timedelta(seconds=s))
    mask = ~np.isnan(arr)
    td = arr.astype(object)
    td[mask] = to_timedelta(td[mask])
    return td

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
