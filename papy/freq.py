''' Frequency analysis tools. '''

import itertools
import multiprocessing as mp
import warnings

import numpy as np
import scipy.signal
import scipy.special

try:
    from tqdm import tqdm
except ImportError:
    warnings.warn('Could not import tqdm.')
    tqdm = lambda x: x

def normalized_periodogram(x, y, angular_freqs):
    ''' Returns a normalized Lomb-Scargle periodogram.

    Wrapper around scipy.signal.lombscargle.

    This normalization is meaningful if len(x) is large enough.
    '''
    pgram = scipy.signal.lombscargle(x, y, angular_freqs)
    return np.sqrt(4 * pgram / len(x))

# cube periodogram ------------------------------------------------------------

def periodogram_freqs(times, oversampling=4, f_max=4):
    ''' Get the optimal frequencies for a periodogram
    '''
    n_points = len(times)

    if f_max < 1:
        warnings.warn(
            'freq_max < freq_nyquist. Expect crappy confidence levels.')
    n_independent_freqs = f_max * n_points

    average_dt = (np.nanmax(times) - np.nanmin(times)) / n_points
    freq_nyquist = 1 / (2 * average_dt)
    freq_max = f_max * freq_nyquist
    dfreq = 1 / (oversampling * n_points * average_dt)
    n_frequencies = int(oversampling * f_max * n_points / 2)
    freqs = np.linspace(dfreq, freq_max, num=n_frequencies)

    return freqs, n_independent_freqs

def _pgram_worker(t, arr, non_nan, afreqs):
    if np.any(non_nan):
        pgram = scipy.signal.lombscargle(
            t[non_nan], arr[non_nan], afreqs, normalize=True)
    else:
        pgram = np.full_like(afreqs, np.nan)
    return pgram

def periodogram(arrays, times, oversampling=4, f_max=2,
                cores=None):
    ''' Make a Lomb-Scargle periodogram from a ndarray.

    Parameters
    ==========
    arrays : np.ndarray
        An array of shape (N, ...) containing the data.
    times : np.ndarray
        An array of shape (N, ...) containing the times associated with
        arr. These do not need to be evenly-spaced, but they will be assumed
        to be sorted and in ascending order.
    oversampling : float (default: 4)
        The factor by which to multiply the standard frequency resolution when
        computing the periodogram.
        The frequency interval will be 1 / (oversampling * N * dt)
    f_max : float (default: 2)
        Max frequency in the periodogram, expressed as a multiple of the
        Nyquist frequency fc.
        The maximum frequency in the periodogram will be f_max / (2 * dt).
        (By default, the maximum frequency is double the Nyquist frequency.)
    cores : int or None (default: None)
        The number of cores to use for multiprocessing, if any. This only makes
        when arrays and times have more than one dimension.

    Notes
    =====
    - When input `arrays` and `times` have more than one dimension, independent
      periodograms are computed for all time series of shape (N, ).
    - Value are assumed not to be closely clumped in time.
    - dt is estimated using dt = (max(t) - min(t)) / N.
    - The number of points in the periodogram is oversampling * f_max * N / 2.
    - See Press+2007, §13.8.

    Returns
    =======
    pgram : np.ndarray
        An array of size (oversampling * f_max * N / 2, ...) containing the
        periodograms.
    freqs : np.ndarray
        A 1D array of size (oversampling * f_max * N / 2) containing the
        frequencies associated with pgram.
    n_independent_freqs : int
        An estimation of the number of independent frequencies, assuming that
        points are not closely clumped in time.
    '''

    if arrays.ndim == 1:
        arrays = arrays.reshape(-1, 1)
        times = times.reshape(-1, 1)

    n_points = arrays.shape[0]
    otherdim = arrays.shape[1:]

    freqs, n_independent_freqs = periodogram_freqs(
        times, oversampling=oversampling, f_max=f_max)

    # cast to float64 as required by scipy.signal.lombscargle()
    freqs = freqs.astype(np.float64)
    times = times.reshape(n_points, -1).astype(np.float64)
    arrays = arrays.reshape(n_points, -1).astype(np.float64)
    arrays = (arrays - np.nanmean(arrays, axis=0)) / np.nanstd(arrays, axis=0)**2
    non_nans = ~np.isnan(times * arrays)

    pgram_iter = zip(
        times.T, arrays.T, non_nans.T, itertools.repeat(2*np.pi*freqs))

    if cores is not None:
        pool = mp.Pool(cores)
        try:
            pgrams = pool.starmap(_pgram_worker, pgram_iter)
        finally:
            pool.terminate()
    else:
        pgram_iter = tqdm(pgram_iter, total=times.shape[1])
        pgrams = list(itertools.starmap(_pgram_worker, pgram_iter))

    pgrams = np.array(pgrams).T
    pgrams = pgrams.reshape(len(freqs), *otherdim)

    # normalize
    norm = np.nanmean(pgrams.reshape(len(freqs), -1), axis=0)
    pgrams /= norm.reshape([-1] + [1] * len(otherdim))

    if pgrams.shape == (len(freqs), 1):
        pgrams = pgrams[:, 0]

    return pgrams, freqs, n_independent_freqs

# confidence levels -----------------------------------------------------------

def confidence_level(n_independent_freqs, sigma=None, p_val=None):
    ''' Get the confidence level for a periodogram from either the sigma or p
    values.

    Parameters
    ==========
    n_independent_freqs : int
        number of independent frequencies in the periodogram
    sigma : iterable of float, or None (default: None)
    p_val : iterable of float, or None (default: None)
        The number of sigmas, or the p-values at which to plot confidence level
        lines. Exactly one of these two parameters must be set.

    Returns
    =======
    confidence_level : float
        The value for the confidence level.
    '''
    if not (sigma is None) != (p_val is None):
        raise ValueError('Exactly one of p_val or sigma must be specified.')
    if p_val is None:
        p_val = 1 - scipy.special.erf(sigma / np.sqrt(2))
    return -np.log(1 - (1 - p_val)**(1/n_independent_freqs))

def plot_confidence_levels(ax, n_independent_freqs,
                           sigma=None, p_val=None,
                           show_labels=True,
                           **kwargs):
    ''' Plot confidence level lines on a plot.

    Parameters
    ==========
    ax : matplotlib axes
    n_independent_freqs : int
        number of independent frequencies in the periodogram
    sigma : iterable of float, or None (default: None)
    p_val : iterable of float, or None (default: None)
        The number of sigmas, or the p-values at which to plot confidence level
        lines. Exactly one of these two parameters must be set.
    show_labels: bool (default: True)
        If True, show threshold values at the right of the plot.
    **kwargs :
        passed to ax.axhline
    '''
    if not (sigma is None) != (p_val is None):
        raise ValueError('Exactly one of p_val or sigma must be specified.')
    if sigma is not None:
        spec = 'sigma'
        vformat = '{}σ'
        values = sigma
    if p_val is not None:
        spec = 'p_val'
        vformat = '{}'
        values = p_val
    style = {
        'linestyle': '--',
        'color': '#bababa',
        'linewidth': 1,
        }
    style.update(kwargs)
    for val in values:
        clevel = confidence_level(n_independent_freqs, **{spec: val})
        ax.axhline(clevel, **style)
        if show_labels:
            ax.text(
                ax.get_xlim()[1] * 1.1, clevel,
                vformat.format(val),
                horizontalalignment='left',
                verticalalignment='center',
                )
