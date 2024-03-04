from __future__ import absolute_import

import logging
import threading

from kafka.metrics.metrics_reporter import AbstractMetricsReporter

logger = logging.getLogger(__name__)


class DictReporter(AbstractMetricsReporter):
    """A basic dictionary based metrics reporter.

    Store all metrics in a two level dictionary of category > name > metric.
    """
    def __init__(self, prefix=''):
        self._lock = threading.Lock()
        self._prefix = prefix if prefix else ''  # never allow None
        self._store = {}

    def snapshot(self):
        """
        Return a nested dictionary snapshot of all metrics and their
        values at this time. Example:
        {
            'category': {
                'metric1_name': 42.0,
                'metric2_name': 'foo'
            }
        }
        """
        return dict((category, dict((name, metric.value())
                                    for name, metric in list(metrics.items())))
                    for category, metrics in
                    list(self._store.items()))

    def init(self, metrics):
        for metric in metrics:
            self.metric_change(metric)

    def metric_change(self, metric):
        with self._lock:
            category = self.get_category(metric)
            if category not in self._store:
                self._store[category] = {}
            self._store[category][metric.metric_name.name] = metric

    def metric_removal(self, metric):
        with self._lock:
            category = self.get_category(metric)
            metrics = self._store.get(category, {})
            removed = metrics.pop(metric.metric_name.name, None)
            if not metrics:
                self._store.pop(category, None)
            return removed

    def get_category(self, metric):
        """
        Return a string category for the metric.

        The category is made up of this reporter's prefix and the
        metric's group and tags.

        Examples:
            prefix = 'foo', group = 'bar', tags = {'a': 1, 'b': 2}
            returns: 'foo.bar.a=1,b=2'

            prefix = 'foo', group = 'bar', tags = None
            returns: 'foo.bar'

            prefix = None, group = 'bar', tags = None
            returns: 'bar'
        """
        tags = ','.join('%s=%s' % (k, v) for k, v in
                        sorted(metric.metric_name.tags.items()))
        return '.'.join(x for x in
                        [self._prefix, metric.metric_name.group, tags] if x)

    def configure(self, configs):
        pass

    def close(self):
        pass
