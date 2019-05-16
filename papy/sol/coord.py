#!/usr/share/bin python

from datetime import timedelta

from astropy.time import Time
import numpy as np
from numpy import sin, cos, tan, sqrt, arcsin, arctan

def arg(y, x):
    ''' Element-wise arc tangent of x/y.

    Arg function as used by Thompson2006, using np.arctan2(x, y).
    '''
    return np.arctan2(x, y)

class Angles:
    ''' Tools to convert angles. '''

    def rad_to_deg(ang):
        return ang / np.pi * 180

    def deg_to_rad(ang):
        return ang / 180 * np.pi

    def arcsec_to_rad(ang):
        return ang / 3600 / 180 * np.pi

    def rad_to_arcsec(ang):
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

def get_sun(dateobs, print_list=False):
    ''' Provide geocentric physical ephemeris of the Sun. (Adapted from Solar
    Soft's get_sun function.)

    Parameters
    ==========
    dateobs :
        Reference time for ephemeris data, passed to astropy.time.Time().
        If no time scale is specified, it is assumed to be UTC.
    print_list : bool (default: False)
        Print the ephemeris before returning it.

    Returns
    =======
    data : vector of solar ephemeris data:
        - Distance (AU)
        - Semidiameter of disk (sec)
        - True longitude (deg)
        - True latitude (0 always)
        - Apparent longitude (deg)
        - Apparent latitude (0 always)
        - True RA (hours)
        - True Dec (deg)
        - Apparent RA (hours)
        - Apparent Dec (deg)
        - Longitude at center of disk (deg)
        - Latitude at center of disk (deg)
        - Position angle of rotation axis (deg)
        - decimal carrington rotation number

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
    _, true_ra = Angles.rectangular_to_polar(x, y)
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
    _, app_ra = Angles.rectangular_to_polar(x, y)
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
    _, eta = Angles.rectangular_to_polar(x, y)
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

    data = np.stack((dist,sd,true_long,true_lat,app_long,app_lat,
        true_ra,true_dec,app_ra,app_dec,he_lon,he_lat, pa,carr))

    return data

def get_earth_L0_B0_D0(dateobs):
    ''' Get the Earth L0 [°], B0 [°], and D0 [m] coordinates.

    Based on get_sun() copied from SSW.
    '''
    d = get_sun(dateobs)
    L0 = d[10]
    B0 = d[11]
    D0 = d[0] * 149597870700
    return L0, B0, D0

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
    corr = Angles.deg_to_rad(corr / 86400) # rad s⁻¹
    return corr

def car_to_hp_earth(lon, lat, dateobs, R=1.,
        occultation=False, diff_rot_ref=None):
    ''' Convert Heliographic Carrington coordinates to Helioprojective
    coordinates for an observer on Earth.

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

    Returns
    =======
    Tx, Ty : float
        The helioprojective cartesian coordinates, in arcsec.
    dateobs : datetime.datetime
    '''
    L0, B0, D0 = get_earth_L0_B0_D0(dateobs)
    R0 = 695508000 # m

    lon = Angles.deg_to_rad(lon)
    lat = Angles.deg_to_rad(lat)
    L0 = Angles.deg_to_rad(L0)
    B0 = Angles.deg_to_rad(B0)

    if diff_rot_ref is not None:
        lon += diff_rot(lat) * total_seconds(dateobs - diff_rot_ref)

    r = R * R0
    x = r * cos(lat) * sin(lon - L0)
    y = r * (sin(lat) * cos(B0) - cos(lat) * cos(lon - L0) * sin(B0))
    z = r * (sin(lat) * sin(B0) + cos(lat) * cos(lon - L0) * cos(B0))

    if occultation:
        x[z < 0] = np.nan
        y[z < 0] = np.nan

    d = sqrt(x**2 + y**2 + (D0 - z)**2)
    Tx = arg(D0 - z, x)
    Ty = arcsin(y / d)

    Tx = Angles.rad_to_arcsec(Tx)
    Ty = Angles.rad_to_arcsec(Ty)

    return Tx, Ty, dateobs

def hp_to_stonyhurst_earth(Tx, Ty, dateobs, R=1., diff_rot_ref=None):
    ''' Convert Helioprojective cartesian to Heliographic Stonyhurst
    coordinates for an observer on Earth.

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
        The heliographic Stonyhurst coordinates, in degrees.
    dateobs : datetime.datetime
    '''
    L0, B0, D0 = get_earth_L0_B0_D0(dateobs)
    R0 = 695508000 # m

    Tx = Angles.arcsec_to_rad(Tx)
    Ty = Angles.arcsec_to_rad(Ty)
    L0 = Angles.deg_to_rad(L0)
    B0 = Angles.deg_to_rad(B0)

    d = D0
    r = R * R0
    x = d * cos(Ty) * sin(Tx)
    y = d * sin(Ty)
    z = sqrt(r**2 - x**2 - y**2)

    r = R * R0
    lon = arg(z * cos(B0) - y * sin(B0), x)
    lat = arcsin((y * cos(B0) + z * sin(B0)) / r)

    if diff_rot_ref is not None:
        lon += diff_rot(lat) * total_seconds(dateobs - diff_rot_ref)

    lon = Angles.rad_to_deg(lon)
    lat = Angles.rad_to_deg(lat)

    return lon, lat, dateobs

def hp_to_car_earth(Tx, Ty, dateobs, R=1., diff_rot_ref=None):
    ''' Convert Helioprojective cartesian to Heliographic Carrington
    coordinates for an observer on Earth.

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
    L0, B0, D0 = get_earth_L0_B0_D0(dateobs)
    R0 = 695508000 # m

    Tx = Angles.arcsec_to_rad(Tx)
    Ty = Angles.arcsec_to_rad(Ty)
    L0 = Angles.deg_to_rad(L0)
    B0 = Angles.deg_to_rad(B0)

    d = D0
    r = R * R0
    x = d * cos(Ty) * sin(Tx)
    y = d * sin(Ty)
    z = sqrt(r**2 - x**2 - y**2)

    r = R * R0
    lon = L0 + arg(z * cos(B0) - y * sin(B0), x)
    lat = arcsin((y * cos(B0) + z * sin(B0)) / r)

    if diff_rot_ref is not None:
        lon -= diff_rot(lat) * total_seconds(dateobs - diff_rot_ref)

    lon = Angles.rad_to_deg(lon)
    lat = Angles.rad_to_deg(lat)

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
