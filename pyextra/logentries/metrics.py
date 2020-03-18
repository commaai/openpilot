from logentries import LogentriesHandler
from threading import Lock
from functools import wraps
import logging
import time
import sys
import psutil

glob_time = 0
glob_name = 0

log = logging.getLogger('logentries')
log.setLevel(logging.INFO)

class Metric(object):

    def __init__(self, token):
        self._count = 0.0
        self._sum = 0.0
        self._lock = Lock()
        self.token = token
        handler = LogentriesHandler(token)
        log.addHandler(handler)

    def observe(self, amount):
        with self._lock:
            self._count += 1
            self._sum += amount

    def metric(self):
        '''Mesaure function execution time in seconds
           and forward it to Logentries'''

        class Timer(object):

            def __init__(self, summary):
                self._summary = summary

            def __enter__(self):
                self._start = time.time()

            def __exit__(self, typ, value, traceback):
                global glob_time
                self._summary.observe(max(time.time() - self._start, 0))
                glob_time = time.time()- self._start
                log.info("function_name=" + glob_name + " " + "execution_time=" + str(glob_time) + " " + "cpu=" + str(psutil.cpu_percent(interval=None)) + " " + "cpu_count=" + str(psutil.cpu_count())+ " " + "memory=" + str(psutil.virtual_memory()) )

            def __call__(self, f):
                @wraps(f)
                def wrapped(*args, **kwargs):
                    with self:
                        global glob_name
                        glob_name = f.__name__

                        return f(*args, **kwargs)
                return wrapped
        return Timer(self)
