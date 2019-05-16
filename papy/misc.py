#!/usr/bin/env python3

import functools
import time

class Chrono():
    def __init__(self, msg=None):
        if msg:
            print(msg)
        self.t0 = time.time()
        self.t = self.t0
    def lap(self, name=None):
        now = time.time()
        if name:
            print(name, end=': ')
        msg = '{:.2g} s (total: {:.2g} s)'
        msg = msg.format(now - self.t, now - self.t0)
        print(msg)
        self.t = now

def cached_property(func):
    ''' Decorator composition of @property with @functools.lru_cache() '''
    return property(functools.lru_cache()(func))
