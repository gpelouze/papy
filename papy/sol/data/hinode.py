#!/usr/share/bin python

import os
import warnings

import numpy as np
import requests


class HinodeQuery:
    ''' A class for browsing the Hinode SDC (http://sdc.uio.no).

    Example
    =======

    >>> search_params = {
    ...     's': ',FILE',
    ...     'INSTRUME': 'EIS',
    ...     'EPOCH_START': '2012-01-01',
    ...     'EPOCH_END': '2012-01-02',
    ...     }
    >>> return_fields = [
    ...     'FILE', 'DATE_OBS', 'E__SLIT_NR',
    ...     'FOVX', 'FOVY', 'XCEN', 'YCEN']
    >>> q = HinodeQuery(search_params, mode='html')
    >>> res = q.get_results(return_fields=return_fields)
    >>> print(res[:4].__repr__())
    rec.array([
     ('eis_l0_20120101_231443', '2012-01-01 23:14:43', 33021, 179.712, 152.0, -433.88, -378.936),
     ('eis_l0_20120101_134553', '2012-01-01 13:45:53', 33022, 179.712, 152.0, -500.699, -381.717),
     ('eis_l0_20120101_230923', '2012-01-01 23:09:23', 33021, 179.712, 152.0, -434.576, -379.168),
     ('eis_l0_20120101_213832', '2012-01-01 21:38:32', 33021, 179.712, 152.0, -445.39, -379.554)],
              dtype=[(('FILE', 'file'), 'O'), (('DATE_OBS', 'date_obs'), 'O'), (('E__SLIT_NR', 'e__slit_nr'), '<i8'), (('FOVX', 'fovx'), '<f8'), (('FOVY', 'fovy'), '<f8'), (('XCEN', 'xcen'), '<

    Notes
    =====
    - To access some fits keywords specific to EIS, prefix them with 'E__'. eg:
      'E__SLIT_NR' for 'SLIT_NR'.
    '''

    default_search_params = {
        'display': {
            'el': ',,192.39,195.12',  # eis lines to display in thumbnails
            'en': '3',  # max eis lines
            'ep': 'IV',  # thumbs to display: I, V, W, IV, VW, or IVW. (Int, Vel, Wid)
            's': ',FILE,INSTRUME,DATE_OBS',  # columns displayed in the table
            'c_s': 'y',  # Auto-include search fields (y/n)
            'th': 'y',  # Show thumbnails (y/n)
            'O': 'DATE',  # Sort order
            'o': 'D',  # Sort order direction (Ascending/Descending)
            'Gx': 'NONE',  # Expand result to include whole group (same values as G)
            },
        'html': {
            'P': '1',  # page
            'L': '10',  # rows per page
            'G': 'IUMODE1',  # grouping (IUMODE[0-3]: very fine, fine, medium, coarse)
            # WEEK, DAY, HOUR, SCI_OBJ_PROG, OBSTITLE_PROG_VER, STUDY_SEQ, NONE)
            },
        'text': {
            'j': 'y',  # ?
            },
        }

    search_url_patterns = {
        'html': 'http://sdc.uio.no/search/result?{query_string}',
        'text': 'http://sdc.uio.no/search/plainserve?{query_string}',
        }

    def __init__(self, search_params, mode='text'):
        ''' Create a new HinodeQuery instance.

        Parameters
        ==========
        search_params : dict
            A dictionnary containing search params.
        mode : str or None (default: None)
            Set whether to build search params for text or html results.
        '''

        self.default_mode = mode
        # search params inputed by the user, not intended to be overwritten.
        self.search_params = search_params

    def _get_mode(self, mode=None):
        ''' Get the the appropriate mode (html or text).

        If mode is None, return self.default_mode. Else, return mode.

        mode : str or None (default: None)
            Set whether we're building search params for text or html results.
            If not set, use self.default_mode.
        '''
        if mode is None:
            return self.default_mode
        else:
            return mode

    def _filled_search_params(self, search_params, mode=None):
        ''' Fill empty search params with default values.

        Parameters
        ==========
        search_params : dict
            A dictionnary containing search params.
        mode : str or None (default: None)
            Set whether we're building search params for text or html results.
            If not set, use self.default_mode.

        If a required key does not exist in `search_params`, fall back to the
        value in HinodeQuery.default_search_params. If a required key exists
        but is set to `None`, remove it. This allows to totally skip some
        search params, falling back to the server's default behaviour.

        '''
        # fallback to default mode
        mode = self._get_mode(mode=mode)
        # build default params
        default_params = self.default_search_params['display'].copy()
        default_params.update(self.default_search_params[mode])
        # use default params for keys that do not exist
        default_params.update(search_params)
        params = default_params
        # delete keys set to None
        params = {k: v for k, v in params.items() if v is not None}
        return params

    def _get_query_string(self, mode=None):
        ''' Get a query string computed from the search params.

        Parameters
        ==========
        mode : str or None (default: None)
            Set whether we're building search params for text or html results.
            If not set, use self.default_mode.

        Returns
        =======
        query_string : str
            A string representation of the search params.

        '''
        # fill with default
        params = self._filled_search_params(
            self.search_params,
            mode=mode)
        # url-encode values
        params = {k: requests.utils.requote_uri(v)
                  for k, v in params.items() if v is not None}
        # join (k1=v1;k2=v2;k3=v3)
        params = ['='.join(p) for p in params.items()]
        query_string = ';'.join(params)
        return query_string

    def search_url(self, mode=None):
        ''' Get the search URL for the given search params.

        Parameters
        ==========
        mode : str or None (default: None)
            Set whether we're building search params for text or html results.
            If not set, use self.default_mode.

        '''
        # fallback to default mode
        if not mode:
            mode = self.default_mode
        # convert to url
        query_string = self._get_query_string(mode=mode)
        url = self.search_url_patterns[mode].format(query_string=query_string)
        return url

    def _parse_text_result(self, text):
        ''' Parse an arbitrary text result from the SDC and convert it to a
        Python list.

        Parameters
        ==========
        text : str
            A result page in text format, downloaded from the archives.
        '''
        # split text into lines
        # split lines into rows (rows are separated by \t)
        # strip leading/trailing quotes and spaces from each cell
        res = [[cell.strip("' ") for cell in row.split('\t')]
               for row in text.splitlines()]
        # first line is the number of results found; remove it
        res = res[1:]
        return res

    def _result_list_to_recarray(self, res):
        ''' Convert a list of results (obtained using _parse_text_result) to a
        np.recarray.

        Note
        ====
        dtypes are determined from header informations, but the parsing may not
        be perfect.
        '''

        # first row contains a header where tags are as 'title:dtype'
        header = res[0]
        res = res[1:]

        # conversion rules for dtypes in header
        dtype_convert = {
            'FALLBACK': str,
            'text': str,
            'double': float,
            'int': int,
            'L': int,
            }

        # determine the appropriate dtype for each column
        labels = []
        dtypes = []
        for h in header:
            label, dtype = h.split(':')
            try:
                dtype = dtype_convert[dtype]
            except KeyError:
                w = 'Could not determine dtype for {}.'.format(h)
                warnings.warn(w, Warning)
                dtype = dtype_convert['FALLBACK']
            labels.append((str(label), str(label.lower())))
            # (explicit str conversion for py 2.7
            dtypes.append(dtype)

        # convert elements to the appropriate dtype
        def parse_row(row):
            new_row = []
            for t, v in zip(dtypes, row):
                new_row.append(t(v))
            return new_row
        res = list(map(parse_row, res))

        # Store strings as object to allow arbitrary length. Thus, replace
        # occurrences of str with object in dtypes.
        dtypes = [object if dt is str else dt for dt in dtypes]

        # zip labels and dtype
        dtype = list(zip(labels, dtypes))
        # convert to list(tuple) so that np doesn't complain
        res = list(map(tuple, res))
        # convert to recarray
        res = np.array(res, dtype=dtype)
        res = res.view(np.recarray)

        return res

    def get_results(self, return_fields=['FILE']):
        ''' Query the EIS database, and get metadata values for the fields
        listed in `return_fields`.

        Parameters
        ==========
        return_fields : list (default: ['FILE'])
            A list containing the metadata fields to include in the results.

        Returns
        =======
        result : np.recarray
            A numpy record array containing the results returned from the
            archives. Columns correspond to the `return_fields`.
        None if the server returned no result.

        Note
        ====
        The 's' field in the search parameters will be overwritten to use the
        values in `return_fields`.

        See also
        ========
        `HinodeQuery._filled_search_params`
        '''

        # force text mode in order to parse results
        mode = 'text'
        # get params with defaults
        params = self._filled_search_params(
            self.search_params,
            mode=mode)
        # insert return_fields into params
        params['s'] = ',' + ','.join(return_fields)
        # get query URL, using a text-mode params
        qr_text = HinodeQuery(params, mode=mode)
        url = qr_text.search_url(mode=mode)

        # download data
        r = requests.get(url)
        r.raise_for_status()
        data = r.text
        data = self._parse_text_result(data)
        try:
            data = self._result_list_to_recarray(data)
        except IndexError as e:
            # IndexError occurs when data is empty, but might be caused by
            # something else.
            if len(data) == 0:
                # data is empty
                data = None
            else:
                # data is not empty and another error occured
                raise e
        return data


class HinodeData:
    ''' A class for downloading data from the Hinode SDC (http://sdc.uio.no).
    '''

    def __init__(self, data_dir='.'):
        ''' Create new HinodeData instance.

        Parameters
        ==========
        data_dir : str or '.'
            Directory where downloaded data is stored.
        '''
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def _add_ext(self, file, gz=True):
        ''' Add the .fits or .fits.gz extension to a file. '''
        if not file.endswith('.fits'):
            file = file + '.fits'
        if gz and not file.endswith('.gz'):
            file = file + '.gz'
        return file

    def _download_dest(self, file, gz=True):
        ''' Get the download destination for a file. '''
        return os.path.join(
            self.data_dir,
            self._add_ext(file, gz=gz))

    def _download_file(self, url, dst):
        ''' Download the file at url and save it at dst. '''
        response = requests.get(url, stream=True)
        if not response.ok:
            raise ValueError('Could not download {}'.format(url))
        with open(dst, 'wb') as f:
            for block in response.iter_content(1024):
                f.write(block)

    def file_url(self, file, gz=True):
        ''' Get the download URL of a file.

        Parameters
        ==========
        file : str
            Filename (e.g. 'eis_l0_20091013_182041.fits').
        gz : bool (default: True)
            If True, get the url of the gzipped file.

        Returns
        =======
        url : str
            Download URL (eg
            'http://sdc.uio.no/search/file/eis_l0_20091013_182041.fits').
        '''
        return 'http://sdc.uio.no/search/file/' + self._add_ext(file, gz=gz)

    def download(self, file, gz=True, force_download=False, silent=False):
        ''' Download a file

        Parameters
        ==========
        file : str
            Filename (e.g. 'eis_l0_20091013_182041.fits').
        gz : bool (default: True)
            If True, download a gzipped file.
        force_download : bool (default: False)
            Download the file, even if a local copy already exists.
        silent : bool (default: False)
            Suppress output.

        Returns
        =======
        downloaded_file : str
            Path of the downloaded file.
        '''
        url = self.file_url(file)
        downloaded_file = self._download_dest(file, gz=gz)
        if force_download or not os.path.isfile(downloaded_file):
            if not silent:
                print(f'Downloading {url} to {downloaded_file}')
            self._download_file(url, downloaded_file)
        else:
            if not silent:
                print(f'Found file {downloaded_file}, skipping download')
        return downloaded_file


if __name__ == '__main__':

    search_params = {
        's': ',FILE',
        'INSTRUME': 'EIS',
        'EPOCH_START': '2012-01-01',
        'EPOCH_END': '2012-01-02',
        }

    q = HinodeQuery(search_params, mode='html')

    return_fields = [
        'FILE', 'DATE_OBS', 'E__SLIT_NR',
        'FOVX', 'FOVY', 'XCEN', 'YCEN']

    res = q.get_results(return_fields=return_fields)

    print(res[:4].__repr__())
