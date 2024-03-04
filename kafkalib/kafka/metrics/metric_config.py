from __future__ import absolute_import

import sys


class MetricConfig(object):
    """Configuration values for metrics"""
    def __init__(self, quota=None, samples=2, event_window=sys.maxsize,
                 time_window_ms=30 * 1000, tags=None):
        """
        Arguments:
            quota (Quota, optional): Upper or lower bound of a value.
            samples (int, optional): Max number of samples kept per metric.
            event_window (int, optional): Max number of values per sample.
            time_window_ms (int, optional): Max age of an individual sample.
            tags (dict of {str: str}, optional): Tags for each metric.
        """
        self.quota = quota
        self._samples = samples
        self.event_window = event_window
        self.time_window_ms = time_window_ms
        # tags should be OrderedDict (not supported in py26)
        self.tags = tags if tags else {}

    @property
    def samples(self):
        return self._samples

    @samples.setter
    def samples(self, value):
        if value < 1:
            raise ValueError('The number of samples must be at least 1.')
        self._samples = value
