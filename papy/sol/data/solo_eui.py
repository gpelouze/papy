#!/usr/share/bin python

import getpass

import bs4
import keyring
import pandas as pd
import requests


class CredentialsManager():
    service_id = 'EUI_SELEKTOR_CLIENT'
    username_key = 'EUI_SELEKTOR_CLIENT_USERNAME'

    def _get_credentials_from_user(self):
        ''' Get credentials from user '''
        username = input('Username: ')
        password = getpass.getpass()
        return username, password

    def _store_credentials_to_keyring(self, credentials):
        ''' Store credentials to the keyring '''
        username, password = credentials
        keyring.set_password(self.service_id, self.username_key, username)
        keyring.set_password(self.service_id, username, password)

    def _retrieve_credentials_from_keyring(self):
        ''' Retrieve credentials from the keyring '''
        username = keyring.get_password(self.service_id, self.username_key)
        if username is None:
            return None
        password = keyring.get_password(self.service_id, username)
        if password is None:
            return None
        return username, password

    def get_credentials(self, force_keyring_update=False):
        ''' Get credentials, either from user input or keyring

        Parameters
        ==========
        force_keyring_update : bool (default: False)
            If True, update the credentials that are already stored in the
            keyring. This means that credentials will be requested to the user.

        Returns
        =======
        username : str
        password : str
        '''
        credentials = self._retrieve_credentials_from_keyring()
        if (credentials is None) or force_keyring_update:
            credentials = self._get_credentials_from_user()
            self._store_credentials_to_keyring(credentials)
        return credentials


class SearchParams:
    class _Base():
        def __init__(self, name, comment=None):
            self.name = name
            self.comment = comment

        @property
        def comment_repr(self):
            if self.comment is not None:
                return f'  # {self.comment}'
            else:
                return ''

    class Number(_Base):
        def __init__(self, name, min, max, comment=None):
            super().__init__(name, comment=comment)
            self.min = min
            self.max = max

        def __repr__(self):
            s = f'{self.name}: number ({self.min}-{self.max})'
            s += self.comment_repr
            return s

    class Date(_Base):
        def __init__(self, name, date, comment=None):
            super().__init__(name, comment=comment)
            self.date = date

        def __repr__(self):
            s = f"{self.name}: date ('YYYY-MM-DD')"
            s += self.comment_repr
            return s

    class _Choice(_Base):
        def __init__(self, name, options, comment=None):
            super().__init__(name, comment=comment)
            self.options = options

        @property
        def options_repr(self):
            options_repr = [repr(opt) for opt in self.options]
            return ', '.join(options_repr)

    class SingleChoice(_Choice):
        def __repr__(self):
            s = f'{self.name}: list (choices: {self.options_repr})'
            s += self.comment_repr
            return s

    class MultipleChoice(_Choice):
        def __repr__(self):
            s = f'{self.name}: string (choices: {self.options_repr})'
            s += self.comment_repr
            return s


class EUISelektorClient():
    ''' EUI selektor client (https://wwwbis.sidc.be/EUI/data_internal/selektor)

    Example
    =======
    >>> client = EUISelektorClient()
    >>> client.show_search_params()
        EUI SELEKTOR search parameters
        ------------------------------
        ..: ...
        detector[]: string (choices: 'FSI', 'HRI_EUV', 'HRI_LYA')  # DETECTOR
        wavelnth[]: string (choices: '174', '304', '1216')  # WAVELNTH
        ..: ...
        date_begin_start: date ('YYYY-MM-DD')  # DATE-BEG
        date_begin_start_hour: number (0-23)  # DATE-BEG
        date_begin_start_minute: number (0-59)  # DATE-BEG
        date_begin_end: date ('YYYY-MM-DD')  # DATE-BEG
        date_begin_end_hour: number (0-23)  # DATE-BEG
        date_begin_end_minute: number (0-59)  # DATE-BEG
    >>> res = client.search({
    ...   'detector[]': 'FSI',
    ...   'wavelnth[]': '304',
    ...   'imgtype[]:': 'solar image',
    ...   'date_begin_start': '2021-12-01',
    ...   'date_begin_end': '2021-12-31',
    ...   'limit[]': 500,
    ...   })
    >>> print(res)
               #                 date-beg detector doorint  ...      crval1      crval2  crpix1  crpix2
        0      0  2021-12-30T23:45:15.335      FSI    open  ...  121.233211  102.484419  1536.5  1536.5
        1      1  2021-12-30T23:30:15.547      FSI    open  ...  121.109280  102.692080  1536.5  1536.5
        2      2  2021-12-30T23:15:15.269      FSI    open  ...  121.120932  102.401791  1536.5  1536.5
        3      3  2021-12-30T23:00:15.501      FSI    open  ...  121.221168  102.394845  1536.5  1536.5
        4      4  2021-12-30T22:45:15.239      FSI    open  ...  121.335848  102.337948  1536.5  1536.5
        ..   ...                      ...      ...     ...  ...         ...         ...     ...     ...
        495  495  2021-12-26T15:38:15.297      FSI    open  ...  121.346022  102.821920  1536.5  1536.5
        496  496  2021-12-26T15:36:15.297      FSI    open  ...  121.463879  102.388034  1536.5  1536.5
        497  497  2021-12-26T15:34:15.297      FSI    open  ...  121.447314  102.156946  1536.5  1536.5
        498  498  2021-12-26T15:32:15.297      FSI    open  ...  121.285647  102.694615  1536.5  1536.5
        499  499  2021-12-26T15:30:15.296      FSI    open  ...  121.402533  102.723120  1536.5  1536.5
        [500 rows x 118 columns]

    '''

    base_url = 'https://wwwbis.sidc.be/EUI/data_internal/selektor/index.php'

    default_search_params = {
        'level[]': 'L1',
        'orderby[]': 'date-beg',
        'order[]': 'DESC',
        'limit[]': 100,
        }

    def __init__(self, base_url=None):
        ''' Initialize client

        Parameters
        ==========
        base_url : str or None (default: None)
            Selektor base url (defaults to `EUISelektorClient.base_url`).
        '''
        if base_url is not None:
            self.base_url = base_url
        self.__http_auth = self._get_http_auth()

    def _get_http_auth(self):
        ''' Get a HTTPBasicAuth object populated using CredentialsManager() '''
        cred = CredentialsManager().get_credentials()
        return requests.auth.HTTPBasicAuth(*cred)

    def update_credentials(self):
        ''' Update the username and password stored in the keyring
        (users will be asked to type them in).
        '''
        CredentialsManager().get_credentials(force_keyring_update=True)
        self.__http_auth = self._get_http_auth()

    def _query(self, params=None):
        ''' Send HTTP query to selektor and return response. '''
        auth = self.__http_auth
        with requests.get(self.base_url, params=params, auth=auth) as r:
            r.raise_for_status()
        return r

    def _parse_detreg_form_table(self, tab):
        ''' Parse the form table containing detreg values (whatever it is) '''
        rows = list(tab.children)
        rows = rows[1::2]  # drop header rows, names are already in input tags
        search_params = []
        for row in rows:
            for input_widget in row.find_all('input'):
                sp = SearchParams.Number(
                    input_widget['name'],
                    input_widget['min'],
                    input_widget['max'],
                    )
                search_params.append(sp)
        return search_params

    def _parse_main_form_table(self, tab):
        ''' Parse the main form table '''
        rows = list(tab.children)
        search_params = []
        for row in rows:
            comment = [td.text for td in row.find_all('td')
                       if td.find('input') is None]
            comment = ' '.join(comment)
            inputs = row.find_all('input')
            if inputs[0]['name'].endswith('[]'):
                ''' Row consists of a single multiple choice field '''
                name = inputs[0]['name']
                options = [inp['value'] for inp in row.find_all('input')]
                input_type = inputs[0]['type']
                if input_type == 'checkbox':
                    sp = SearchParams.MultipleChoice(
                        name, options, comment=comment)
                elif input_type == 'radio':
                    sp = SearchParams.SingleChoice(
                        name, options, comment=comment)
                else:
                    msg = 'could not parse form: unknown input type'
                    raise ValueError(msg)
                search_params.append(sp)
            else:
                ''' Row consists of several (single-choice) fields '''
                for input_widget in inputs:
                    if input_widget['type'] == 'number':
                        sp = SearchParams.Number(
                            input_widget['name'],
                            input_widget['min'],
                            input_widget['max'],
                            comment=comment,
                            )
                    elif input_widget['type'] == 'date':
                        sp = SearchParams.Date(
                            input_widget['name'],
                            input_widget['value'],
                            comment=comment,
                            )
                    else:
                        msg = 'could not parse form: unknown input type'
                        raise ValueError(msg)
                    search_params.append(sp)
        return search_params

    def get_search_form(self):
        ''' Query the form from the selektor web page

        Returns
        =======
        search_params : list of SearchParams
            Search parameters, with description and allowed values
        '''
        r = self._query()

        b = bs4.BeautifulSoup(r.content, features='html.parser')
        tables = b.find_all('table')
        search_params = self._parse_detreg_form_table(tables[0])
        search_params += self._parse_main_form_table(tables[1])

        return search_params

    def show_search_params(self):
        ''' Query the form from the selektor web page and display it '''
        print('EUI SELEKTOR search parameters')
        print('------------------------------')
        for param in self.get_search_form():
            print(param)

    def _fill_default_search_params(self, search_params):
        default_params = self.default_search_params.copy()
        # use default params for keys that do not exist
        default_params.update(search_params)
        params = default_params
        # delete keys set to None
        params = {k: v for k, v in params.items() if v is not None}
        return params

    def search(self, search_params):
        ''' Send search query and parse results

        Parameters
        ==========
        search_params : dict
            Dict of search parameters (call `.show_search_params()` to display
            available search keywords).

        Returns
        =======
        retults : pd.DataFrame
            Search results.
        '''
        search_params = self._fill_default_search_params(search_params)
        r = self._query(params=search_params)
        dfs = pd.read_html(r.content)
        try:
            return dfs[2]
        except IndexError:
            return None


if __name__ == '__main__':

    client = EUISelektorClient()
    client.show_search_params()
    res = client.search({
        'detector[]': 'FSI',
        'wavelnth[]': '304',
        'imgtype[]:': 'solar image',
        'date_begin_start': '2021-12-01',
        'date_begin_end': '2021-12-31',
        'limit[]': 500,
        })

    # # To update username and password:
    # client.update_credentials()
