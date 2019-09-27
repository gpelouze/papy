#!/usr/share/bin python

from datetime import timedelta
from dateutil.parser import parse
import os
import re
import shutil
import sys
import pickle
import warnings

from astropy.io import fits
import astropy.units as u
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import requests

# Util to silence md.media_search
class nostdout(object):
    class DummyFile(object):
        def write(self, x): pass
        def flush(self, x): pass
    def __enter__(self):
        sys.stdout = nostdout.DummyFile()
    def __exit__(self, type, value, traceback):
        sys.stdout = sys.__stdout__

class ReFiles():

    ''' Regex patterns to be combined and compiled at instanciation '''
    patterns = {
        'mssl data': 'http://solar.ads.rl.ac.uk/MSSL-data/',
        'path/name': ('eis/level(?P<lev>\d)/(?P<year>\d{4})/(?P<month>\d{2})/(?P<day>\d{2})/'
                      'eis_(?P<lev_str>[le][0-9e])_(?P=year)(?P=month)(?P=day)_(?P<time>\d{6})(\.fits\.gz)?'),
        'name': 'eis_(?P<lev_str>[le][0-9r])_(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})_(?P<time>\d{6})(\.fits\.gz)?',
        }

    def __init__(self):
        ''' Regex for matching the URL of the FITS returned after a SQL query. '''
        mssl_fits_url = '{mssl data}{path/name}'.format(**self.patterns)
        self.mssl_fits_url = re.compile(mssl_fits_url)
        ''' Regex for matching the path/filename of a FITS '''
        fits_path = '{path/name}'.format(**self.patterns)
        self.fits_path = re.compile(fits_path)
        ''' Regex for matching the filename of a FITS '''
        fits_name = '{name}'.format(**self.patterns)
        self.fits_name = re.compile(fits_name)

# compile regexes at import
re_files = ReFiles()

def fits_path(eis_file, absolute=False, url=False, gz=False):
    ''' Determine the path of the FITS file (in the Hinode data directory) from
    a `prop_dict`.

    Parameters
    ==========
    eis_file : str or dict.
        The EIS filename (eg 'eis_l0_20110803_113520'), path, URL in the MSSL
        archives, or 'prop dict'.
    absolute : bool
        If true, return the absolute path, reading environment variable
        $HINODE_DATA, with fallback to $SOL_OBSERVATION_DATA/hinode.
    url : bool
        If true, return the URL to the MSSL archives instead of the path.
    gz : bool (default: False)
        If true, use the .fits.gz extension. Else, simply return .gz.
    '''
    p = ('eis/level{lev}/{year}/{month}/{day}/'
         'eis_{lev_str}_{year}{month}{day}_{time}.fits')
    if gz:
        p += '.gz'
    if url:
        p = ReFiles.patterns['mssl data'] + p
    if absolute:
        try:
            hinode_data = os.environ['HINODE_DATA']
        except KeyError:
            try:
                hinode_data = os.path.join(os.environ['SOL_OBSERVATION_DATA'], 'hinode')
            except KeyError:
                raise ValueError('Could not find $HINODE_DATA nor $SOL_OBSERVATION_DATA')
        p = os.path.join(hinode_data, p)
    prop_dict = prop_from_filename(eis_file)
    return p.format(**prop_dict)

def prop_from_filename(filename):
    ''' Parse an EIS file name, path or URL to get a 'prop_dict'.

    If passed a dict, return it if it's a 'prop_dict', raise ValueError if not.
    '''

    # check if filename could be a prop_dict
    if type(filename) is dict:
        d = filename
        prop_dict_keys = {'lev', 'lev_str', 'year', 'month', 'day', 'time'}
        if prop_dict_keys.issubset(set(d.keys())):
            # d contains at least all prop_dict keys
            return d
        else:
            raise ValueError('Got a dict that does not look like a prop_dict.')

    # handle filename as... a filename!
    regex_to_try = (
        re_files.mssl_fits_url,
        re_files.fits_path,
        re_files.fits_name,
        )
    for r in regex_to_try:
        m = r.match(filename)
        if m:
            prop_dict = m.groupdict()
            if 'lev' not in prop_dict:
                if prop_dict['lev_str'] == 'er':
                    prop_dict['lev'] = '1'
                else: # assume lev_str is 'l\d'
                    prop_dict['lev'] = prop_dict['lev_str'][-1]
            return prop_dict
    msg = "Not a valid EIS file name/path/url format: {}".format(filename)
    raise ValueError(msg)

def mssl_query_sql(query):
    ''' Submit a SQL query to the MSSL SDC and return the resulting prop_dicts.
    These dicts may be converted to a path / url list using fits_path().
    '''
    r = requests.post(
        'http://solarb.mssl.ucl.ac.uk/SolarB/SQLConn.jsp',
        data={'comments': query},
        )

    # with open('tutu.html', 'w') as f:
        # f.write(r.text)

    res_list = []

    b = BeautifulSoup(r.text, 'lxml')
    sqlList = b.find('div', attrs={'class': 'sqlList'})
    if sqlList:
        fits_table = list(sqlList.children)[1]
        fits_list = b.find_all('a', attrs={'href': re.compile('.*\.fits\.gz')})
        for a in fits_list:
            r = re_files.mssl_fits_url.match(a.attrs['href'])
            res_list.append(r.groupdict())

    return res_list

def get_fits(eis_file, custom_dest=None, force_download=False, silent=True):
    ''' Get a given EIS FITS. If not found locally, download it from the MSSL
    SDC.

    Parameters
    ==========
    eis_file : str or dict.
        The EIS filename (eg 'eis_l0_20110803_113520'), path, URL in the MSSL
        archives, or 'prop dict'.
    custom_dest : str (default: None)
        If set, the location where to save the FITS.
        If custom_dest is not set, the FITS is saved to
        $SOL_OBSERVATION_DATA/hinode/eis/ if $SOL_OBSERVATION_DATA exists,
        or to $HOME/hinode/eis/ if it doesn't.
    force_download : bool (default: False)
        If True, always download the FITS, overwriting any local version.

    Notes
    =====
    Some regular FITS in the MSSL archives have the wrong extension, .fits.gz.
    This function tests if it is the case, and renames the file if needed.

    **Warning:** when using `force_download=True` to retrieve, eg.
    `foo.fits.gz`, any existing `foo.fits` might be overwritten.
    '''

    # determine fits url and save path
    eis_properties = prop_from_filename(eis_file)
    fits_url = fits_path(eis_properties, url=True, gz=True)
    if custom_dest:
        fits_save_path = custom_dest
    else:
        fits_save_path = os.path.join(
            os.environ.get('SOL_OBSERVATION_DATA', os.environ.get('HOME')),
            'hinode',
            fits_path(eis_properties, gz=True))
    # determine if .fits.gz or .fits
    fits_base, fits_ext = os.path.splitext(fits_save_path)
    if fits_ext == '.gz':
        fits_save_path_unzip = fits_base
    else:
        fits_save_path_unzip = fits_save_path

    # download path
    if not (os.path.exists(fits_save_path) or
            os.path.exists(fits_save_path_unzip)) or force_download:
        if not silent:
            print('Downloading {} to {}'.format(fits_url, fits_save_path))
        response = requests.get(fits_url, stream=True)
        if not response.ok:
            m = 'Could not get {}'.format(fits_url)
            raise ValueError(m)
            # TODO find better exception
        fits_dir, fits_filename = os.path.split(fits_save_path)
        if not os.path.exists(fits_dir):
            os.makedirs(fits_dir)
        with open(fits_save_path, 'wb') as f:
            for block in response.iter_content(1024):
                f.write(block)
    try:
        # print('Trying {}'.format(fits_save_path))
        f = fits.open(fits_save_path)
    except IOError as e:
        if fits_ext == '.gz':
            # then the error is most likely due to the fact that the file is a
            # regular FITS with a .fits.gz extension, or has already been
            # deflated.
            if not os.path.exists(fits_save_path_unzip) or force_download:
                shutil.move(fits_save_path, fits_save_path_unzip)
            # print('Opening {}'.format(fits_save_path_unzip))
            f = fits.open(fits_save_path_unzip)
        else:
            raise e
    if not silent:
        print('Opened {}'.format(f.filename()))
    return f


class HTTPCache(object):
    def __init__(self, filename=None):
        self.filename = filename
        self.cache = {}
        self._load_data()

    def __contains__(self, url):
        return self.cache.__contains__(url)

    def _download(self, url):
        with requests.get(url) as response:
            response.raise_for_status()
        return response.content

    def get(self, url, force_update=False):
        if force_update or url not in self.cache:
            self.update(url, self._download(url))
        return self.cache[url]

    def update(self, url, value):
        self.cache[url] = value
        self._dump_data()

    def remove(self, url):
        del self.cache[url]
        self._dump_data()

    def _dump_data(self):
        if self.filename is not None:
            with open(self.filename, 'wb') as f:
                pickle.dump(self.cache, f, pickle.HIGHEST_PROTOCOL)

    def _load_data(self):
        if self.filename is not None:
            if os.path.isfile(self.filename):
                with open(self.filename, 'rb') as f:
                    self.cache = pickle.load(f)
            else:
                warnings.warn('Cache file does not exist, it will be created.')

class HTMLCache(HTTPCache):
    def get_bs(self, key, features='lxml'):
        return BeautifulSoup(self.get(key), features=features)

class HTMLTable(list):
    def __init__(self, table, n_cols=None, discard_empty_cells=False):
        ''' Parse a HTML table as nested lists

        Parameters
        ==========
        table : bs4.element.Tag
            <table> HTML tag
        n_cols : int or None
            If set, discard rows with a different number of columns
        discard_empty_cells : bool
            If True, empty cells are discarded.
        '''
        parsed_table = []
        for row in table.find_all('tr'):
            cells = [cell.text for cell in row.find_all('td')]
            if discard_empty_cells:
                try:
                    cells.remove('')
                except ValueError:
                    pass
            if n_cols and len(cells) != n_cols:
                continue
            parsed_table.append(cells)
        super().__init__(parsed_table)

    def process_cells(self, func):
        ''' Apply func to each cell of the table, and replace them with the
        output.  '''
        for i, row in enumerate(self):
            for j, cell in enumerate(row):
                self[i][j] = func(cell)


class EISFile(object):
    def __init__(self, name, populate=True, cache=None):
        self.name = name
        self.props = None
        self.cache = cache
        if self.cache is None:
            self.cache = HTMLCache()
        if populate:
            self.populate_props()

    def __repr__(self):
        return '<{}>'.format(self.name)

    def populate_props(self):
        url = 'http://sdc.uio.no/search/show_details?FILE=' + self.name
        b = self.cache.get_bs(url)

        table = b.find('table', attrs={'class': 'show_details'})
        table = HTMLTable(table, n_cols=3, discard_empty_cells=True)
        self.props = EISPropsDict.from_table(table)

    def get_props(self):
        if not self.props:
            self.populate_props()
        return self.props

    def get_prop(self, key):
        return self.get_props()[key]

class EISPropsDict(dict):
    def __init__(self, props):
        d = {}
        for p in props:
            d[p.name] = p
        super().__init__(d)

    def from_table(table):
        props = [EISProp(*cell) for cell in table]
        return EISPropsDict(props)

    def add(self, prop):
        if prop.name in self.keys():
            raise ValueError('Property already exists: ' + prop.name)
        self[prop.name] = prop

class EISProp(object):
    def __init__(self, name, value, comment):
        self.name = name
        self.value = value
        self.comment = comment

    def __repr__(self):
        return self.value.__repr__()

    def __str__(self):
        return self.value.__repr__()


class Study(object):
    def __init__(self, study_id, populate=True, cache=None):
        self.id = int(study_id)
        self.cache = cache
        if self.cache is None:
            self.cache = HTMLCache()
        self.details = None
        self.n_rasters = None
        self.url = 'http://solarb.mssl.ucl.ac.uk:8080/SolarB/ShowEisStudy.jsp'
        self.url += '?study={}'.format(self.id)
        if populate:
            self.populate_props()

    def populate_props(self):
        b = self.cache.get_bs(self.url)

        tables = b.find(id='content').find_all('table')
        for i, table in enumerate(tables):
            table = HTMLTable(table, n_cols=2)
            table.process_cells(lambda s: s.strip('\r'))
            table = table[1:]
            tables[i] = table

        # This only works for studies that have the same number of rasters and
        # line lists. But are there other cases?
        self.n_rasters = len(tables) // 2
        if len(tables) != (2 * self.n_rasters + 1):
            raise ValueError('unexpected number of tables found in study page')
        details = tables[0]
        rasters, line_lists = [], []
        for i in range(self.n_rasters):
            rasters.append(tables[2*i+1])
            line_lists.append(tables[2*i+2])

        self.details = StudyDetails(details)
        self.rasters = [Raster(r) for r in rasters]
        self.line_lists = line_lists

    @property
    def raster(self):
        if not self.rasters:
            return
        elif len(self.rasters) == 1:
            return self.rasters[0]
        else:
            raise ValueError('study contains multiple rasters')

    @property
    def line_list(self):
        if not self.line_lists:
            return
        elif len(self.line_lists) == 1:
            return self.line_lists[0]
        else:
            raise ValueError('study contains multiple line lists')

    def __repr__(self):
        return '<EIS Study {}>'.format(self.id)

class StudyComponent(dict):
    def __init__(self, table):
        d = {}
        for k, v in table:
            d[k] = v
        super().__init__(d)

    def __getattr__(self, attr):
        return self[self.attrs[attr]]

class StudyDetails(StudyComponent):
    attrs = {
        'id': 'ID',
        'acronym': 'ACRONYM',
        'title': 'TITLE',
        'target': 'TARGET',
        'n_rasters': 'NO. OF RASTERS',
        }

class Raster(StudyComponent):
    attrs = {
        'id': 'RASTER ID',
        'acronym': 'ACRONYM',
        'll_id': 'LL_ID',
        'type': 'RASTER TYPE',
        'n_pointing_pos': 'NO. OF POINTING POSITIONS',
        'scan_step': 'SCAN STEP SIZE (arcsec)',
        'n_windows': 'NO. OF WINDOWS',
        'window_widths': 'WINDOW WIDTHS (pixels)',
        'window_height': 'WINDOW HEIGHT (pixels)',
        'slit': 'SLIT/SLOT',
        'exp_times': 'EXPOSURE TIMES (ms)',
        'exp_delay': 'EXPOSURE DELAY (ms)',
        }
    slit_conversion = {
        '0': '1"',
        '2': '2"',
        }
    def __init__(self, d):
        super().__init__(d)
        slit = self['SLIT/SLOT']
        self['SLIT/SLOT'] = self.slit_conversion.get(slit, slit)


def view_fov(
        eis_file,
        full_disk_instr='aia', context_wvl='171', fig=None,
        ias_location_prefix='/',
        print_metadata=False,
        silent=True):
    ''' Display EIS FOV determined from FITS metadata, over a full disk image.

    Parameters
    ==========
    eis_file : str or dict.
        The EIS filename (eg 'eis_l0_20110803_113520'), path, URL in the MSSL
        archives, or 'prop dict'.
    full_disk_instr : string (default: 'aia')
        For now, AIA is the only instrument supported.
    context_wvl : string (default: 'aia')
        The wavelength for the context image
    fig : matplotlib.figure.Figure (default: None)
        If set, plot to this figure. If not, create a new figure.
    ias_location_prefix : str
        Passed to eis.get_map

    TODO
    ====
    - Allow eis_file to be an astropy.io.fits or a path to a fits.

    '''

    # get EIS and AIA data

    try:
        # when executing this file
        import aia
    except ImportError:
        # when importing this file from sol.data
        from . import aia

    eis_fits = get_fits(eis_file, silent=silent)
    h = eis_fits[0].header
    with nostdout():
        aia_map = aia.get_map(
            h['date_obs'], context_wvl,
            ias_location_prefix=ias_location_prefix,
            search_window=10,
            discard_bad_quality=True,
            )

    # determine EIS FOV limits
    xycen = u.Quantity((h['xcen'], h['ycen']), unit=u.arcsec)
    fovxy = u.Quantity((h['fovx'], h['fovy']), unit=u.arcsec)
    bottom_left = xycen - fovxy / 2

    # Plot figures
    if not fig:
        fig = plt.figure()
    fig.clear()

    ax_full = fig.add_subplot(1, 2, 1)
    ax_zoom = fig.add_subplot(1, 2, 2)

    # full disk + FOV rectangle
    aia_map.plot(axes=ax_full)
    aia_map.draw_rectangle(bottom_left, *fovxy, color='w')

    # zoom to FOV
    aia_map.plot(axes=ax_zoom)
    xymin = bottom_left.value
    xymax = (bottom_left + fovxy).value
    ax_zoom.set_xlim(xymin[0], xymax[0])
    ax_zoom.set_ylim(xymin[1], xymax[1])

    # suptitle with metadata
    metadata = eis_fits[0].header
    _, eis_filename = os.path.split(eis_fits.filename())
    eis_filename, _ = os.path.splitext(eis_filename)
    metadata['EIS_FILENAME'] = eis_filename

    date_start = parse(metadata['date_obs'])
    date_end = parse(metadata['date_end'])
    metadata['DURATION'] = str(date_end - date_start)

    suptitle = (
        '{EIS_FILENAME}',
        'duration: {DURATION}',
        'nexp: {NEXP}',
        'slit ID: {SLIT_ID}',
        )
    suptitle = '\n'.join(suptitle).format(**metadata)
    fig.suptitle(suptitle)

    return aia_map


if __name__ == '__main__':

    plt.ion()

    # # test mssl queries
    # sql_query = ('select FitsLocation from eis_level0 where '
        # 'DATE_OBS between "2007-03-15" AND "2007-03-17"')
    # # sql_query = ('select * from eis_level0 where '
        # # 'FitsLocation="http://solar.ads.rl.ac.uk/MSSL-data/'
        # # 'eis/level0/2007/03/15/eis_l0_20070315_112713.fits.gz"')
    # res_list = mssl_query_sql(sql_query)

    # f = get_fits('eis_l0_20110803_113520')
    fig = plt.figure(1)
    # view_fov('eis_l0_20110803_113520', fig=fig)
    view_fov('eis_l0_20110804_003143', fig=fig)
