
import os
import time, timeit
import sys
import logging

def clock(func):

    def clocked(*args, **kwargs):

        t0 = timeit.default_timer()
        result = func(*args, **kwargs)
        elapsed = timeit.default_timer() - t0
        name = func.__name__
        print('[%d:%s] costs [%0.8fs]' % (os.getpid(), name, elapsed))
    return clocked

class timer():
    def __init__(self, name, log_id = ''):
        self.t0 = timeit.default_timer()
        self.name = name
        self.log_id = log_id

    def __del__(self):
        elapsed = (timeit.default_timer() - self.t0) * 1000
        logging.info("%d %s %s fun_cost %0.4f"
                     % (os.getpid(), self.name, self.log_id, elapsed))

