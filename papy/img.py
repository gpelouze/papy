''' Image analysis tools. '''

from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion

def detect_peaks(image):
    ''' Detect peaks in an image using local maximum filter.

    Parameters
    ==========
    image : array-like (ndim = 2)
        The image where to detect the peaks.

    Returns
    =======
    res : np.ndarray (dtype = bool)
        A boolean mask of the same dimension as `image`, where pixels that are
        local maxima contain True, and other pixels contain False.

    Source: <http://stackoverflow.com/a/3689710/4352108>.

    FIXME: check that the returned array contains bools and not zeros and ones.
    '''

    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(2, 2)

    # apply the local maximum filter; all pixel of maximal value
    # in their neighborhood are set to 1
    local_max = maximum_filter(image, footprint=neighborhood) == image
    # local_max is a mask that contains the peaks we are
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.

    # we create the mask of the background
    background = (image == 0)

    # a little technicality: we must erode the background in order to
    # successfully subtract it form local_max, otherwise a line will
    # appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    # we obtain the final mask, containing only peaks,
    # by removing the background from the local_max mask
    detected_peaks = local_max - eroded_background

    return detected_peaks
