from __future__ import absolute_import


class Percentile(object):
    def __init__(self, metric_name, percentile):
        self._metric_name = metric_name
        self._percentile = float(percentile)

    @property
    def name(self):
        return self._metric_name

    @property
    def percentile(self):
        return self._percentile
