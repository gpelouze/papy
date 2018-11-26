''' Tools to manipulate time arrays. '''

import datetime

import dateutil.parser
import numpy as np

def datetime_average(arr_1, arr_2):
    ''' Average function that is friendly with datetime formats that only
    support subtraction. '''
    return arr_1 + (arr_2 - arr_1) / 2

@np.vectorize
def total_seconds(timedelta):
    ''' Vectorised version of timedelta.total_seconds() '''
    return timedelta.total_seconds()

@np.vectorize
def parse(date):
    ''' Vectorised dateutil.parser.parse '''
    return dateutil.parser.parse(date)

def seconds_to_timedelta(arr):
    ''' Parse an array of seconds and convert it to timedelta. '''
    to_timedelta = np.vectorize(lambda s: datetime.timedelta(seconds=s))
    mask = ~np.isnan(arr)
    timedelta = arr.astype(object)
    timedelta[mask] = to_timedelta(timedelta[mask])
    return timedelta
