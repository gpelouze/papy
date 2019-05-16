#!/usr/share/bin python

import astropy.units as u
import matplotlib as mpl
import numpy as np

from . import coord as sc

def draw_grid(ax, dateobs=None, l0=None, b0=None, dsun=None,
        grid_spacing=15*u.deg, unit=u.arcsec, **kwargs):
    ''' Draw a grid over the surface of the Sun.

    Parameters
    ==========
    x : matplotlib axis
    dateobs :
        The observation date. Setting this keyword will lead to assume that the
        observer is on Earth, and ignore l0, b0 and dsun.
    l0 : astropy.units.Quantity
        The heliographic longitude of the observer.
    b0 : astropy.units.Quantity
        The heliographic latitude of the observer.
    dsun : astropy.units.Quantity
        The distance of the observer to the Sun.
    grid_spacing : astropy.units.Quantity
        Spacing for longitude and latitude grid.
    unit : astropy.units.Unit
        The unit to use on the plot.
    **kwargs :
        Passed to ax.plot.

    Totally copied from sunpy.map.mapbase.GenericMap.draw_grid.
    '''

    if dateobs:
        l0, b0, dsun = sc.get_earth_L0_B0_D0(dateobs)
        l0 = u.Quantity(l0, 'deg')
        b0 = u.Quantity(b0, 'deg')
        dsun = u.Quantity(dsun, 'm').to('au')
    elif not (l0 and b0 and dsun):
        msg = 'You must specify either dateobs, or l0, b0 and dsun.'
        raise ValueError(msg)

    l0 = l0.to(u.deg).value
    b0 = b0.to(u.deg).value
    dsun = dsun.to(u.m).value

    # Prep the plot kwargs
    plot_kw = {'color': 'black',
               'linestyle': 'dotted',
               'zorder': 100,
               }
    plot_kw.update(kwargs)

    lines = []

    # Do not automatically rescale axes when plotting the overlay
    ax.set_autoscale_on(False)

    hg_longitude_deg = np.linspace(-180, 180, num=361) + l0
    hg_latitude_deg = np.arange(-90, 90, grid_spacing.to(u.deg).value)

    # draw the latitude lines
    for lat in hg_latitude_deg:
        x, y, _ = sc.car_to_hp_earth(
            hg_longitude_deg, lat * np.ones(361),
            dateobs=dateobs, occultation=True)
        valid = np.logical_and(np.isfinite(x), np.isfinite(y))
        x = x[valid]
        y = y[valid]
        lines += ax.plot(x, y, **plot_kw)

    hg_longitude_deg = np.arange(-180, 180, grid_spacing.to(u.deg).value) + l0
    hg_latitude_deg = np.linspace(-90, 90, num=181)

    # draw the longitude lines
    for lon in hg_longitude_deg:
        x, y, _ = sc.car_to_hp_earth(
            lon * np.ones(181), hg_latitude_deg,
            dateobs=dateobs, occultation=True)
        valid = np.logical_and(np.isfinite(x), np.isfinite(y))
        x = x[valid]
        y = y[valid]
        lines += ax.plot(x, y, **plot_kw)

    # Turn autoscaling back on.
    ax.set_autoscale_on(True)

    return lines

def draw_limb(ax, dateobs=None, rsun=None, unit=u.arcsec, **kwargs):
    ''' Draw a circle representing the solar limb.

    Parameters
    ==========
    x : matplotlib axis
    dateobs :
        The observation date. Setting this keyword will lead to assume that the
        observer is on Earth, and ignore rsun.
    rsun : astropy.units.Quantity
        The angular size of the Sun.
    unit : astropy.units.Unit
        The unit to use on the plot.
    **kwargs :
        Passed to matplotlib.patches.

    Totally copied from sunpy.map.mapbase.GenericMap.draw_limb.
    '''

    if dateobs:
        rsun = sc.get_sun(dateobs)[1]
        rsun = u.Quantity(rsun, 'arcsec')
    elif not rsun:
        raise ValueError('You must specify either dateobs or rsun.')
    rsun = rsun.to(unit).value

    c_kw = {'radius': rsun,
            'fill': False,
            'color': 'black',
            'zorder': 100,
            }
    c_kw.update(kwargs)

    circ = mpl.patches.Circle([0, 0], **c_kw)
    ax.add_artist(circ)

    return [circ]

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    plt.ion()

    y = 4 * np.array(range(10)) - 10
    x = 4 * np.array(range(15)) - 10

    y_grid, x_grid = np.meshgrid(y, x, indexing='ij')
    arr = y_grid * x_grid
    contour = (-100 < arr) & (arr < 100)

    # test draw_grid and draw_limb
    plt.clf()
    ax = plt.gca()
    ax.set_aspect(1)
    dateobs = '2012-09-01T00:00:00'
    draw_grid(ax, dateobs=dateobs)
    draw_limb(ax, dateobs=dateobs)
    ax.autoscale()
