#!/usr/share/bin python

from datetime import timedelta
import warnings

from astropy.io import fits
from astropy.time import Time
import numpy as np
from numpy import sin, cos, tan, sqrt, arcsin, arctan

from papy.misc import cached_property

# Numerical tools =============================================================

def arg(y, x):
    ''' Element-wise arc tangent of x/y.

    Arg function as used by Thompson2006, using np.arctan2(x, y).
    '''
    return np.arctan2(x, y)

class ang:
    ''' Tools to convert angles. '''

    def rad2deg(ang):
        return ang / np.pi * 180

    def deg2rad(ang):
        return ang / 180 * np.pi

    def arcsec2rad(ang):
        return ang / 3600 / 180 * np.pi

    def rad2arcsec(ang):
        return ang / np.pi * 180 * 3600

    def rectangular_to_polar(x, y):
        ''' Convert 2D cartesian coordinates to polar coordinates.

        Parameters
        ==========
        x, y : float or array
            The cartesian coordinates.

        Returns
        =======
        r, theta : float or array
            The polar radius and angle. theta is in radians.
        '''
        r = sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        return r, theta

@np.vectorize
def total_seconds(timedelta):
    return timedelta.total_seconds()


# Observer location ===========================================================

def get_sun_geocentric(dateobs, print_list=False, standard_units=False):
    ''' Provide geocentric physical ephemeris of the Sun. (Adapted from Solar
    Soft's get_sun function.)

    Parameters
    ==========
    dateobs :
        Reference time for ephemeris data, passed to astropy.time.Time().
        If no time scale is specified, it is assumed to be UTC.
    print_list : bool (default: False)
        Print the ephemeris before returning it.
    standard_units : bool (default: False)
        If True, convert all returned angle values to radians, and all lengths
        to meters.

    Returns
    =======
    data : vector of solar ephemeris data:
        - Distance (AU)
        - Semi-diameter of disk (arcsec)
        - True longitude (deg)
        - True latitude (0 always)
        - Apparent longitude (deg)
        - Apparent latitude (0 always)
        - True RA (hour angle)
        - True Dec (deg)
        - Apparent RA (hour angle)
        - Apparent Dec (deg)
        - Longitude at center of disk (deg)
        - Latitude at center of disk (deg)
        - Position angle of rotation axis (deg)
        - decimal Carrington rotation number

        If standard_units=True, all angles are converted to radians (be they in
        arcsec, degrees, or hour angles), and all lengths are converted to
        meters.

    Notes
    =====
    Based on the book Astronomical Formulae for Calculators, by Jean Meeus.

    Original Copyright notice
    =========================
    Copyright (C) 1991, Johns Hopkins University/Applied Physics Laboratory
    This software may be used, copied, or redistributed as long as it is not
    sold and this copyright notice is reproduced on each copy made. This
    routine is provided as is without any express or implied warranties
    whatsoever. Other limitations apply as described in the file
    disclaimer.txt.
    '''
    radeg = 180 / np.pi

    # convert input time to internal format
    dateobs = Time(dateobs)

    # Julian date:
    jd = dateobs.jd

    # Julian Centuries from 1900.0:
    t = (jd - 2415020) / 36525

    # Carrington Rotation Number:
    carr = (1 / 27.2753) * (jd - 2398167) + 1

    # Geometric Mean Longitude (deg):
    mnl = 279.69668 + 36000.76892 * t + 0.0003025 * t**2
    mnl = mnl % 360

    # Mean anomaly (deg):
    mna = 358.47583 + 35999.04975 * t - 0.000150 * t**2 - 0.0000033 * t**3
    mna = mna % 360

    # Eccentricity of orbit:
    e = 0.01675104 - 0.0000418 * t - 0.000000126 * t**2

    # Sun's equation of center (deg):
    c = (1.919460 - 0.004789 * t - 0.000014 * t**2) * sin(mna / radeg) + \
        (0.020094 - 0.000100 * t) * sin(2 * mna / radeg) + \
        0.000293 * sin(3 * mna / radeg)

    # Sun's true geometric longitude (deg)
    #   (Refered to the mean equinox of date.  Question: Should the higher
    #    accuracy terms from which app_long is derived be added to true_long?)
    true_long = (mnl + c) % 360

    # Sun's true anomaly (deg):
    ta = (mna + c) % 360

    # Sun's radius vector (AU).  There are a set of higher accuracy
    #   terms not included here.  The values calculated here agree with
    #   the example in the book:
    dist = 1.0000002 * (1 - e **2) / (1 + e * cos(ta / radeg))

    # Semidiameter (arc sec):
    sd = 959.63 / dist

    # Apparent longitude (deg) from true longitude:
    omega = 259.18 - 1934.142 * t # deg
    app_long = true_long - 0.00569 - 0.00479 * sin(omega / radeg)

    # Latitudes (deg) for completeness.  Never more than 1.2 arc sec from 0,
    # always set to 0 here:
    if np.array(dist).ndim == 0:
        true_lat = 0
        app_lat = 0
    else:
        true_lat = np.zeros_like(dist)
        app_lat = np.zeros_like(dist)

    # True Obliquity of the ecliptic (deg):
    ob1 = 23.452294 - 0.0130125 * t - 0.00000164 * t**2 + 0.000000503 * t**3

    # True RA, Dec (is this correct?):
    y = cos(ob1 / radeg) * sin(true_long / radeg)
    x = cos(true_long / radeg)
    _, true_ra = ang.rectangular_to_polar(x, y)
    true_ra *= radeg
    true_ra = true_ra % 360
    if np.any(true_ra < 0):
        true_ra[true_ra < 0] += 360
    true_ra = true_ra / 15
    true_dec = arcsin(sin(ob1 / radeg) * sin(true_long / radeg)) * radeg

    # Apparent  Obliquity of the ecliptic:
    ob2 = ob1 + 0.00256 * cos(omega / radeg) # Correction.

    # Apparent  RA, Dec (agrees with example in book):
    y = cos(ob2 / radeg) * sin(app_long / radeg)
    x = cos(app_long / radeg)
    _, app_ra = ang.rectangular_to_polar(x, y)
    app_ra *= radeg
    app_ra = app_ra % 360
    if np.any(app_ra < 0):
        app_ra[app_ra < 0] += 360
    app_ra = app_ra / 15
    app_dec = arcsin(sin(ob2 / radeg) * sin(app_long / radeg)) * radeg

    # Heliographic coordinates:
    theta = (jd - 2398220) * 360 / 25.38 # deg
    i = 7.25 # deg
    k = 74.3646 + 1.395833 * t # deg
    lamda = true_long - 0.00569
    lamda2 = lamda - 0.00479 * sin(omega / radeg)
    diff = (lamda - k) / radeg
    x = arctan( -cos(lamda2 / radeg) * tan(ob1 / radeg)) * radeg
    y = arctan( -cos(diff) * tan(i / radeg)) * radeg

    # Position of north pole (deg):
    pa = x + y

    # Latitude at center of disk (deg):
    he_lat = arcsin(sin(diff) * sin(i / radeg)) * radeg

    # Longitude at center of disk (deg):
    y = -sin(diff) * cos(i / radeg)
    x = -cos(diff)
    _, eta = ang.rectangular_to_polar(x, y)
    eta *= radeg
    he_lon = (eta - theta) % 360
    if np.any(he_lon < 0):
        he_lon[he_lon < 0] += 360

    if print_list:
        msg = (
            'Solar Ephemeris for {dateobs}',
            '',
            'Distance = {dist:.5f} au',
            'Semidiameter = {sd:.3f} arcsec',
            'True (long, lat) = ({true_long:.3f}°, {true_lat:.1f}°)',
            'Apparent (long, lat) = ({app_long:.3f}°, {app_lat:.1f}°)',
            'True (RA, Dec) = ({true_ra:.3f} h, {true_dec:.3f}°)',
            'Apparent (RA, Dec) = ({app_ra:.3f} h, {app_dec:.3f}°)',
            'Heliographic long. and lat. of disk center = ({he_lon:.3f}°, {he_lat:.3f}°)',
            'Position angle of north pole = {pa:.3f}°',
            'Carrington Rotation Number = {carr:.6f}',
            )
        msg = '\n'.join(msg)
        msg = msg.format(**locals())
        print(msg)

    if standard_units:
        dist *= 149597870700
        # convert all angles to radians
        degra = np.pi / 180
        arcsra = degra / 3600
        hra = np.pi / 12
        sd *= degra
        true_long *= degra
        true_lat *= degra
        app_long *= degra
        app_lat *= degra
        true_ra *= hra
        true_dec *= degra
        app_ra *= hra
        app_dec *= degra
        he_lon *= degra
        he_lat *= degra
        pa *= degra

    data = np.stack((dist,sd,true_long,true_lat,app_long,app_lat,
        true_ra,true_dec,app_ra,app_dec,he_lon,he_lat, pa,carr))

    return data

class Observer():
    ''' Prototype for a class that should provide position of an observer
    relatively to the Sun, as well as L0. '''
    def __init__(self, date_obs):
        '''
        Parameters
        ==========
        date_obs :
            Reference time for ephemeris data, passed to astropy.time.Time().
            If no time scale is specified, it is assumed to be UTC.
        '''
        self.date_obs = Time(date_obs)
        ''' Carrington longitude of the central meridian as seen from Earth '''
        self.L0 = get_sun_geocentric(self.date_obs, standard_units=True)[10]
        ''' Stonyhurst longitude of the observer '''
        self.Phi0 = None
        ''' Stonyhurst latitude of the observer '''
        self.B0 = None
        ''' Distance between the Sun center and the observer '''
        self.D0 = None

class ObserverEarth(Observer):
    def __init__(self, date_obs):
        super().__init__(date_obs)
        self.date_obs = Time(date_obs)
        self.ephemeris = get_sun_geocentric(self.date_obs, standard_units=True)
        self.L0 = self.ephemeris[10]
        self.Phi0 = 0 # by definition of the Stonyhurst coordinates
        self.B0 = self.ephemeris[11]
        self.D0 = self.ephemeris[0]

class ObserverFITS(Observer):
    ''' Observer object loaded from FITS header data. '''
    def __init__(self, fits_path, hdu=0,
                 keywords=dict(date_obs='DATE-OBS', hglt_obs='HGLT_OBS',
                               hgln_obs='HGLN_OBS', dsun_obs='DSUN_OBS')):
        '''
        Parameters
        ==========
        fits_path :
            The path to a FITS file which header contains:
            - date_obs : the observation date (in any format understood by
              astropy.time.Time()),
            - hglt_obs, hgln_obs : the Stonyhurst coordinates of the observer
              (in degrees),
            - dsun_obs : the distance from the observer to the Sun (in meters).
            (Also see 'keywords' kwarg.)
        hdu : int (default: 0)
            The index of the HDU from which to take the header.
        keywords : dict
            The FITS header keywords for the four above parameters.
            Default values correspond to those used in the AIA level 1 FITS
            headers.
        '''
        header = fits.open(fits_path)[hdu].header
        super().__init__(header[keywords['date_obs']])
        self.B0 = ang.deg2rad(header[keywords['hglt_obs']])
        self.Phi0 = ang.deg2rad(header[keywords['hgln_obs']])
        self.D0 = header[keywords['dsun_obs']]

# Misc. =======================================================================

def diff_rot(lat, wvl='default'):
    ''' Return the angular velocity difference between differential and
    Carrington rotation.

    Parameters
    ==========
    lat : float
        The latitude, in radians
    wvl : str (default: 'default'
        The wavelength, or the band to return the rotation from.

    Returns
    =======
    corr : float
        The difference in angular velocities between the differential and
        Carrington rotations, in radians per second:
            Δω(θ) = ω_Car - ω_diff(θ)
            with ω_Car = 360° / (25.38 days)
            and  ω_diff(θ) = A + B sin² θ + C sin⁴ θ
    '''
    p = {
        # ° day⁻¹; Hortin (2003):
        'EIT 171': (14.56, -2.65, 0.96),
        'EIT 195': (14.50, -2.14, 0.66),
        'EIT 284': (14.60, -0.71, -1.18),
        'EIT 304': (14.51, -3.12, 0.34),
        }
    p['default'] = p['EIT 195']
    A, B, C = p[wvl]
    A_car = 360 / 25.38 # ° day⁻¹
    corr = A - A_car + B * sin(lat)**2 + C * sin(lat)**4 # ° day⁻¹
    corr = ang.deg2rad(corr / 86400) # rad s⁻¹
    return corr

def R0_to_m(R0):
    R_sun = 695700000 # m
    return R_sun * R0

def default_r(r, R0):
    if r is None:
        r = R0_to_m(R0)
    return r


# Coordinate systems conversion ===============================================

# Base functions --------------------------------------------------------------

# Units assumptions:
# lengths in meters
# angles in radians
#
# helioprojective d and heliographic (Stonyhurst or Carrington) r may be
# undefined, because these coordinates system often come projected.
# Functions that convert from these systems handle None values for d or r.
# In that case, all points are assumed to be located on the sphere centered on
# the Sun, and with the radius the keyword argument of the function R0,
# expressed in units of R_sun = 695508000 m.

def heeq_to_stonyhurst(Xheeq, Yheeq, Zheeq):
    ''' Convert heliocentric earth equatorial (HEEQ) to heliographic Stonyhurst
    coordinates.  Thompson2006 Eq. (1)

    Parameters
    ==========
    Xheeq, Yheeq, Zheeq : float
        The heliocentric earth equatorial coordinates, in meters.

    Returns
    =======
    lon_hg, lat : float
        The heliographic Stonyhurst longitude and latitude, in radians.
    r : float
        The radius coordinate, in meters.
    '''
    r = np.sqrt(Xheeq**2 + Yheeq**2 + Zheeq**2)
    lat = arctan(Zheeq / np.sqrt(Xheeq**2 + Yheeq**2))
    lon_hg = arg(Xheeq, Yheeq)
    return lon_hg, lat, r

def stonyhurst_to_heeq(lon_hg, lat, r, R0=1):
    ''' Convert heliographic Stonyhurst to heliocentric earth equatorial (HEEQ)
    coordinates.  Thompson2006 Eq. (2)

    Parameters
    ==========
    lon_hg, lat : float
        The heliographic Stonyhurst longitude and latitude, in degrees.
    r : float or None
        The radius coordinate, in meters.
    R0 : float (default: 1)
        Radius of the projection sphere which is used if r is None, in solar
        radii.

    Returns
    =======
    Xheeq, Yheeq, Zheeq : float
        The heliocentric earth equatorial coordinates, in meters.
    '''
    r = default_r(r, R0)
    Xheeq = r * cos(lat) * cos(lon_hg)
    Yheeq = r * cos(lat) * sin(lon_hg)
    Zheeq = r * sin(lat)
    return Xheeq, Yheeq, Zheeq


def carrington_to_stonyhurst(lon_car, lat, r, observer):
    ''' Convert heliographic Stonyhurst to Carrington coordinates.
    Thompson2006 Eq. (3)

    Parameters
    ==========
    lon_car, lat : float
        The heliographic Carrington longitude and latitude, in radians.
    r : float
        The radius coordinate, in meters.
    observer : Observer
        Object that gives the position of the observer relative to the Sun.

    Returns
    =======
    lon_hg, lat : float
        The heliographic Stonyhurst longitude and latitude, in radians.
    r : float
        The radius coordinate, in meters.

    **Note that only lon_hg is changed.**
    '''
    lon_hg = lon_car - observer.L0
    return lon_hg, lat, r

def stonyhurst_to_carington(lon_hg, lat, r, observer):
    ''' Convert heliographic Stonyhurst to Carrington coordinates.
    Thompson2006 Eq. (3)

    Parameters
    ==========
    lon_hg, lat : float
        The heliographic Stonyhurst longitude and latitude, in radians.
    r : float
        The radius coordinate, in meters.
    observer : Observer
        Object that gives the position of the observer relative to the Sun.

    Returns
    =======
    lon_car, lat : float
        The heliographic Carrington longitude and latitude, in radians.
    r : float
        The radius coordinate, in meters.

    **Note that only lon_car is changed.**
    '''
    lon_car = lon_hg + observer.L0
    return lon_car, lat, r


def stonyhurst_to_heliocentric(lon_hg, lat, r, observer, R0=1):
    ''' Convert heliocentric cartesian to heliographic Stonyhurst coordinates.
    Thompson2006 Eq. (11)

    Parameters
    ==========
    lon_hg, lat : float
        The heliographic Stonyhurst longitude and latitude, in radians.
    r : float or None
        The radius coordinate, in meters.
    observer : Observer
        Object that gives the position of the observer relative to the Sun.
    R0 : float (default: 1)
        Radius of the projection sphere which is used if r is None, in solar
        radii.

    Returns
    =======
    x, y, z : float
        The heliocentric cartesian coordinates, in meters.
    '''
    r = default_r(r, R0)
    x = r * cos(lat) * sin(lon_hg - observer.Phi0)
    y = r * (sin(lat) * cos(observer.B0) - cos(lat) * cos(lon_hg - observer.Phi0) * sin(observer.B0))
    z = r * (sin(lat) * sin(observer.B0) + cos(lat) * cos(lon_hg - observer.Phi0) * cos(observer.B0))
    return x, y, z

def heliocentric_to_stonyhurst(x, y, z, observer):
    ''' Convert heliocentric cartesian to heliographic Stonyhurst coordinates.
    Thompson2006 Eq. (12)

    Parameters
    ==========
    x, y, z : float
        The heliocentric cartesian coordinates, in meters.
    observer : Observer
        Object that gives the position of the observer relative to the Sun.

    Returns
    =======
    lon_hg, lat : float
        The heliographic Stonyhurst longitude and latitude, in radians.
    r : float
        The radius coordinate, in meters.
    '''
    r = np.sqrt(x**2 + y**2 + z**2)
    lat = arcsin((y * cos(observer.B0) + z * sin(observer.B0)) / r)
    lon_hg = observer.Phi0 + arg(z * cos(observer.B0) - y * sin(observer.B0), x)
    return lon_hg, lat, r


def helioprojective_to_heliocentric(Tx, Ty, d, observer, R0=1):
    ''' Convert helioprojective to heliocentric cartesian coordinates.
    Thompson2006 Eq. (15)

    Parameters
    ==========
    Tx, Ty : float
        The helioprojective cartesian coordinates, in arcsec.
    d : float or None
        The distance to the point from the observer, in meters.
    observer : Observer
        Object that gives the position of the observer relative to the Sun.
    R0 : float (default: 1)
        Radius of the projection sphere which is used if d is None, in solar
        radii.

    Returns
    =======
    x, y, z : float
        The heliocentric cartesian coordinates, in meters.
    '''
    if d is None:
        d = observer.D0
        default_d = True
    else:
        default_d = False
    x = d * cos(Ty) * sin(Tx)
    y = d * sin(Ty)
    if default_d:
        z = sqrt(R0_to_m(R0)**2 - x**2 - y**2)
    else:
        z = observer.D0 - d * cos(Ty) * cos(Tx)
    return x, y, z

def heliocentric_to_helioprojective(x, y, z, observer):
    ''' Convert heliocentric to helioprojective cartesian coordinates.
    Thompson2006 Eq. (16)

    Parameters
    ==========
    x, y, z : float
        The heliocentric cartesian coordinates, in meters.
    observer : Observer
        Object that gives the position of the observer relative to the Sun.

    Returns
    =======
    Tx, Ty : float
        The helioprojective cartesian coordinates, in arcsec.
    d : float
        The distance to the point from the observer, in meters.
    '''
    d = np.sqrt(x**2 + y**2 + (observer.D0 - z)**2)
    Tx = arg(observer.D0 - z, x)
    Ty = arcsin(y / d)
    return Tx, Ty, d

# composed functions ----------------------------------------------------------

def carrington_to_helioprojective(lon_car, lat, r, observer, R0=1,
        diff_rot_ref=None, occultation=False):
    ''' Convert heliographic Carrington to helioprojective cartesian
    coordinates.

    Parameters
    ==========
    lon_car, lat : float
        The heliographic Carrington longitude and latitude, in radians.
    r : float or None
        The radius coordinate, in meters.
    observer : Observer
        Object that gives the position of the observer relative to the Sun.
    R0 : float (default: 1)
        Radius of the projection sphere which is used if r is None, in solar
        radii.
    diff_rot_ref : datetime.datetime or None (default: None)
        If not None, correct for differential rotation, using this reference
        date.
    occultation : bool (default: False)
        If True, set all points that would be hidden behind the disk to nan.

    Returns
    =======
    Tx, Ty : float
        The helioprojective cartesian coordinates, in arcsec.
    d : float
        The distance to the point from the observer, in meters.
    '''

    r = default_r(r, R0)

    if diff_rot_ref is not None:
        lon_car += diff_rot(lat) * total_seconds(observer.date_obs - diff_rot_ref)

    lon_hg, lat, r = carrington_to_stonyhurst(lon_car, lat, r, observer)
    x, y, z = stonyhurst_to_heliocentric(lon_hg, lat, r, observer)

    if occultation:
        x[z < 0] = np.nan
        y[z < 0] = np.nan

    Tx, Ty, d = heliocentric_to_helioprojective(x, y, z, observer)

    return Tx, Ty, d

def helioprojective_to_carrington(Tx, Ty, d, observer, R0=1, diff_rot_ref=None):
    ''' Convert helioprojective cartesian to heliographic Carrington
    coordinates.

    Parameters
    ==========
    Tx, Ty : float
        The helioprojective cartesian coordinates, in arcsec.
    d : float or None
        The distance to the point from the observer, in meters.
    observer : Observer
        Object that gives the position of the observer relative to the Sun.
    R0 : float (default: 1)
        Radius of the projection sphere which is used if d is None, in solar
        radii.
    diff_rot_ref : datetime.datetime or None (default: None)
        If not None, correct for differential rotation, using this reference
        date.

    Returns
    =======
    lon_car, lat : float
        The heliographic Carrington longitude and latitude, in radians.
    r : float
        The radius coordinate, in meters.
    '''
    x, y, z = helioprojective_to_heliocentric(Tx, Ty, d, observer)
    lon_hg, lat, r = heliocentric_to_stonyhurst(x, y, z, observer)
    lon_car, lat, r = stonyhurst_to_carington(lon_hg, lat, r, observer)

    if diff_rot_ref is not None:
        lon_car -= diff_rot(lat) * total_seconds(observer.date_obs - diff_rot_ref)

    return lon_car, lat, r


def heeq_to_helioprojective(Xheeq, Yheeq, Zheeq, observer):
    ''' Convert heliocentric earth equatorial (HEEQ) to helioprojective

    Parameters
    ==========
    Xheeq, Yheeq, Zheeq : float
        The heliocentric earth equatorial coordinates, in meters.
    observer : Observer
        Object that gives the position of the observer relative to the Sun.

    Returns
    =======
    Tx, Ty : float
        The helioprojective cartesian coordinates, in arcsec.
    d : float
        The distance to the point from the observer, in meters.
    '''
    lon_hg, lat, r = heeq_to_stonyhurst(Xheeq, Yheeq, Zheeq)
    x, y, z = stonyhurst_to_heliocentric(lon_hg, lat, r, observer)
    Tx, Ty, d = heliocentric_to_helioprojective(x, y, z, observer)
    return Tx, Ty, d

def helioprojective_to_heeq(Tx, Ty, d, observer):
    ''' Convert helioprojective to heliocentric earth equatorial (HEEQ)

    Parameters
    ==========
    Tx, Ty : float
        The helioprojective cartesian coordinates, in arcsec.
    d : float
        The distance to the point from the observer, in meters.
    observer : Observer
        Object that gives the position of the observer relative to the Sun.

    Returns
    =======
    Xheeq, Yheeq, Zheeq : float
        The heliocentric earth equatorial coordinates, in meters.
    '''
    x, y, z = helioprojective_to_heliocentric(Tx, Ty, d, observer)
    lon_hg, lat, r = heliocentric_to_stonyhurst(x, y, z, observer)
    Xheeq, Yheeq, Zheeq = stonyhurst_to_heeq(lon_hg, lat, r)
    return Xheeq, Yheeq, Zheeq

# Compatibility ===============================================================

def car_to_hp_earth(lon, lat, dateobs, R0=1.,
        occultation=False, diff_rot_ref=None):
    ''' Convert Heliographic Carrington coordinates to Helioprojective
    coordinates for an observer on Earth.

    DEPRECATED FUNCTION KEPT FOR COMPATIBILITY.
    Use car_to_hp instead.

    Parameters
    ==========
    lon, lat : float
        The heliographic Carrington coordinates, in degrees.
    dateobs : datetime.datetime
        The date of observation.
    R : float (default: 1.)
        The size, in solar radius, of the projection sphere.
    occultation : bool (default: False
        If True set all points not visible from the observer to nan.
    diff_rot_ref : datetime.datetime or None (default: None)
        If not None, correct for differential rotation, using this reference
        date.

    neturns
    =======
    Tx, Ty : float
        The helioprojective cartesian coordinates, in arcsec.
    dateobs : datetime.datetime
    '''
    warnings.warn('deprecated, use car_to_hp instead', DeprecationWarning)
    lon = ang.deg2rad(lon)
    lat = ang.deg2rad(lat)
    observer = ObserverEarth(dateobs)
    Tx, Ty, _ = carrington_to_helioprojective(
        lon, lat, None,
        ObserverEarth(dateobs),
        R0=R0,
        diff_rot_ref=diff_rot_ref)
    Tx = ang.rad2arcsec(Tx)
    Ty = ang.rad2arcsec(Ty)
    return Tx, Ty, dateobs

def hp_to_car_earth(Tx, Ty, dateobs, R0=1., diff_rot_ref=None):
    ''' Convert Helioprojective cartesian to Heliographic Carrington
    coordinates for an observer on Earth.

    DEPRECATED FUNCTION KEPT FOR COMPATIBILITY.
    Use hp_to_car instead.

    Parameters
    ==========
    Tx, Ty: float
        The helioprojective coordinates in arcsec.
    dateobs : datetime.datetime
        The date of observation
    R : float (default: 1.)
        The size, in solar radius, of the projection sphere.
    diff_rot_ref : datetime.datetime or None (default: None)
        If not None, correct for differential rotation, using this reference
        date.

    Returns
    =======
    lon, lat: float
        The heliographic Carrington coordinates, in degrees.
    dateobs : datetime.datetime
    '''
    warnings.warn('deprecated, use hp_to_car instead', DeprecationWarning)
    Tx = ang.arcsec2rad(Tx)
    Ty = ang.arcsec2rad(Ty)
    lon, lat, _ = helioprojective_to_carrington(
        Tx, Ty, None,
        ObserverEarth(dateobs),
        R0=R0,
        diff_rot_ref=diff_rot_ref)
    lon = ang.rad2deg(lon)
    lat = ang.rad2deg(lat)
    return lon, lat, dateobs


if __name__ == '__main__':

    import datetime

    dateobs = datetime.datetime(2010, 3, 18, 7, 13, 3)
    tx = -123
    ty = 540
    random_test = False
    if random_test:
        dateobs = datetime.datetime.fromtimestamp(
            np.random.randint(2000000000))
        tx = np.random.rand() * 1800 - 900
        ty = np.random.rand() * 1800 - 900
    ln, lt, _ = hp_to_car_earth(tx, ty, dateobs)
    Tx, Ty, _ = car_to_hp_earth(ln, lt, dateobs)
    Ln, Lt, _ = hp_to_car_earth(Tx, Ty, dateobs)
    print(dateobs)
    print(tx, ty)
    print(Tx, Ty)
    print(ln, lt)
    print(Ln, Lt)
