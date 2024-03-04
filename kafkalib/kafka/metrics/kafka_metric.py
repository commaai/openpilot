from __future__ import absolute_import

import time


class KafkaMetric(object):
    # NOTE java constructor takes a lock instance
    def __init__(self, metric_name, measurable, config):
        if not metric_name:
            raise ValueError('metric_name must be non-empty')
        if not measurable:
            raise ValueError('measurable must be non-empty')
        self._metric_name = metric_name
        self._measurable = measurable
        self._config = config

    @property
    def metric_name(self):
        return self._metric_name

    @property
    def measurable(self):
        return self._measurable

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config):
        self._config = config

    def value(self, time_ms=None):
        if time_ms is None:
            time_ms = time.time() * 1000
        return self.measurable.measure(self.config, time_ms)
