#!/usr/bin/env python2
# vim: fileencoding=utf-8

''' Tools for finding, downloading and storing AIA data.
'''

import datetime as dt
import os
import shutil

from astropy import time
import dateutil.parser
import numpy as np
import requests
try:
    import sitools2.clients.sdo_client_medoc as md
except:
    print('Cannot import sitools2.')
import sunpy.map

verb = False

def filename_from_medoc_result(res, path_prefix=''):
    ''' Determine the path an name of an AIA fits from the object returned by
    the sitools2 medoc client.
    '''

    # Extract metadata and store them to a dict
    metadata = {}
    # date:
    date = ('year', 'month', 'day', 'hour', 'minute', 'second', 'microsecond')
    metadata.update({
            k: res.date_obs.__getattribute__(k) for k in date
            })
    # data_level
    try:
        metadata['level'] = res.series_name.strip('aia.lev')
    except AttributeError:
        # pySitools 1.0.1 doesn't provide res.series_name. Default to 1
        metadata['level'] = '1'
    # wavelength
    metadata['wave'] = res.wave
    metadata['isodate'] = res.date_obs.strftime('%Y-%m-%dT%H-%M-%S')

    # define file path and name templates
    # eg path: "sdo/aia/level1/2012/06/05/"
    path = os.path.join(
        path_prefix, 'sdo', 'aia', 'level{level}',
        '{year:04d}', '{month:02d}', '{day:02d}')
    # eg format: "aia.lev1.171A_2012-06-05T11-15-36.image_lev1.fits"
    filename = 'aia.lev{level}.{wave}A_{isodate}.image_lev{level}.fits'

    path = path.format(**metadata)
    filename = filename.format(**metadata)

    return path, filename

def download_fits(medoc_result, dirname, filename):
    ''' Download the FITS for a given medoc request result, and store it to
    dirname/filename.
    '''
    if verb:
        print('AIA: downloading to {}'.format(os.path.join(dirname, filename)))
    medoc_result.get_file(target_dir=dirname, filename=filename)

def get_fits(aia_res, ias_location_prefix='/'):
    ''' Get the path to the fits file for a md.Sdo_data object object.

    If the file exists in ias_location_prefix, return the path to this file.
    Else, download it and return the path to the downloaded file.

    The files are downloaded under the path specified in environment variable
    $SOL_OBSERVATION_DATA if is set, and in the current directory if not.
    '''

    # build aia path
    path_prefix = os.environ.get('SOL_OBSERVATION_DATA', None)
    aia_dir, aia_file = filename_from_medoc_result(
        aia_res, path_prefix=path_prefix)
    if path_prefix is None:
        aia_dir = '.'

    # build path to fits in ias_location
    ias_location = aia_res.ias_location.strip('/')
    ias_location_fits = os.path.join(
        ias_location_prefix, ias_location, 'S00000', 'image_lev1.fits')
    ias_location_fits = os.path.expanduser(ias_location_fits)

    # check for fits in ias_location
    if os.path.exists(ias_location_fits):
        if verb:
            print('AIA: using {}'.format(ias_location_fits))
        fits_path = ias_location_fits

    # if no fits in ias_location, get it over HTTP
    else:
        if not os.path.exists(aia_dir):
            os.makedirs(aia_dir)
        if not os.path.exists(os.path.join(aia_dir, aia_file)):
            download_fits(aia_res, aia_dir, aia_file)
        if verb:
            print('AIA: using {}'.format(os.path.join(aia_dir, aia_file)))
        fits_path = os.path.join(aia_dir, aia_file)

    return fits_path

def get_map(obs_date, wavelength, ias_location_prefix='/',
        search_window=1, discard_bad_quality=False):
    ''' Get a map containing the closest AIA image found in Medoc for the given
    date and wavelength.

    Parameters
    ==========
    obs_date : str
        The date for which to search AIA data, in ISO format.
    wavelength : int or str
        The wavelength of the desired AIA channel.
    ias_location_prefix : str (default: '/')
        The local path from which to retrieve the data.
        eg. to use '~/sshfs' to retrieve data from ~/sshfs/SUM02/foo
        instead the default /SUM02/foo.
    search_window : float (default: 1)
        The window (in hours), within which to search for the data.
    discard_bad_quality : bool (default: False)
        If set to True, discard data for which quality is not 0.

    Raises
    ======
    - ValueError when no data matching the requested parameters were found.
    '''

    # AIA data: get closest AIA corresponding to EIS
    # dates
    d_eis = dateutil.parser.parse(obs_date)
    d1 = d_eis - dt.timedelta(hours=search_window / 2)
    d2 = d_eis + dt.timedelta(hours=search_window / 2)

    # get data from Medoc
    aia_res = md.media_search(
        dates=[d1, d2],
        waves=[str(wavelength)],
        cadence=['1m'],
        )

    # discard images with bad quality
    try:
        if discard_bad_quality:
            quality = md.media_metadata_search(
                media_data_list=aia_res,
                keywords=['quality'],
                )
            quality = np.array([q['quality'] for q in quality])
            aia_res = list(np.array(aia_res)[quality == 0])
        if len(aia_res) == 0:
            raise ValueError
    except ValueError:
        no_data_msg = 'Could not find data matching the requested parameters.'
        raise ValueError(no_data_msg)

    # get closest image in date
    aia_dates_obs = np.array([res.date_obs for res in aia_res])
    aia_closest = np.argmin(np.abs(d_eis - aia_dates_obs))
    aia_res = aia_res[aia_closest]

    fits_location = get_fits(
        aia_res,
        ias_location_prefix=ias_location_prefix,
        )
    aia_map = sunpy.map.Map(fits_location)

    aia_map = aia_map.rotate(rmatrix=aia_map.rotation_matrix)

    return aia_map

def temperature_to_channel(logT):
    ''' Get the most appropriate AIA channel for a given temperature.

    Parameters
    ==========
    temp : float
        log(T in K)

    Returns
    =======
    channel : str
        The string describing the AIA channel
    '''

    temperatures = np.array([3.7, 4.7, 5.6, 5.8,
        6.2, 6.3, 6.4, 6.8, 7, 7.3])
    channels = np.array(['1700', '304', '131', '171', '193',
        '211', '335', '94', '131', '193'])
    return channels[np.argmin(np.abs(logT - temperatures))]

def sdotime_to_utc(sdo_time):
    ''' Convert SDO time as in the MEDOC database (seconds since
    1977-01-01T00:00:00TAI) to UTC datetime. '''
    t_ref = time.Time('1977-01-01T00:00:00', scale='tai')
    t_tai = t_ref + time.TimeDelta(sdo_time, format='sec', scale='tai')
    return t_tai.utc.datetime

def utc_to_sdotime(utc):
    ''' Convert UTC datetime to SDO time as in the MEDOC database (seconds
    since 1977-01-01T00:00:00TAI). '''
    t_ref = time.Time('1977-01-01T00:00:00', scale='tai')
    sdo_time = time.Time(utc, scale='utc') - t_ref
    return sdo_time.sec
