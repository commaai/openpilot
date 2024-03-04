from __future__ import absolute_import

import threading
import time

from kafka.errors import QuotaViolationError
from kafka.metrics import KafkaMetric


class Sensor(object):
    """
    A sensor applies a continuous sequence of numerical values
    to a set of associated metrics. For example a sensor on
    message size would record a sequence of message sizes using
    the `record(double)` api and would maintain a set
    of metrics about request sizes such as the average or max.
    """
    def __init__(self, registry, name, parents, config,
                 inactive_sensor_expiration_time_seconds):
        if not name:
            raise ValueError('name must be non-empty')
        self._lock = threading.RLock()
        self._registry = registry
        self._name = name
        self._parents = parents or []
        self._metrics = []
        self._stats = []
        self._config = config
        self._inactive_sensor_expiration_time_ms = (
            inactive_sensor_expiration_time_seconds * 1000)
        self._last_record_time = time.time() * 1000
        self._check_forest(set())

    def _check_forest(self, sensors):
        """Validate that this sensor doesn't end up referencing itself."""
        if self in sensors:
            raise ValueError('Circular dependency in sensors: %s is its own'
                             'parent.' % (self.name,))
        sensors.add(self)
        for parent in self._parents:
            parent._check_forest(sensors)

    @property
    def name(self):
        """
        The name this sensor is registered with.
        This name will be unique among all registered sensors.
        """
        return self._name

    @property
    def metrics(self):
        return tuple(self._metrics)

    def record(self, value=1.0, time_ms=None):
        """
        Record a value at a known time.
        Arguments:
            value (double): The value we are recording
            time_ms (int): A POSIX timestamp in milliseconds.
                Default: The time when record() is evaluated (now)

        Raises:
            QuotaViolationException: if recording this value moves a
                metric beyond its configured maximum or minimum bound
        """
        if time_ms is None:
            time_ms = time.time() * 1000
        self._last_record_time = time_ms
        with self._lock:  # XXX high volume, might be performance issue
            # increment all the stats
            for stat in self._stats:
                stat.record(self._config, value, time_ms)
            self._check_quotas(time_ms)
        for parent in self._parents:
            parent.record(value, time_ms)

    def _check_quotas(self, time_ms):
        """
        Check if we have violated our quota for any metric that
        has a configured quota
        """
        for metric in self._metrics:
            if metric.config and metric.config.quota:
                value = metric.value(time_ms)
                if not metric.config.quota.is_acceptable(value):
                    raise QuotaViolationError("'%s' violated quota. Actual: "
                                              "%d, Threshold: %d" %
                                              (metric.metric_name,
                                               value,
                                               metric.config.quota.bound))

    def add_compound(self, compound_stat, config=None):
        """
        Register a compound statistic with this sensor which
        yields multiple measurable quantities (like a histogram)

        Arguments:
            stat (AbstractCompoundStat): The stat to register
            config (MetricConfig): The configuration for this stat.
                If None then the stat will use the default configuration
                for this sensor.
        """
        if not compound_stat:
            raise ValueError('compound stat must be non-empty')
        self._stats.append(compound_stat)
        for named_measurable in compound_stat.stats():
            metric = KafkaMetric(named_measurable.name, named_measurable.stat,
                                 config or self._config)
            self._registry.register_metric(metric)
            self._metrics.append(metric)

    def add(self, metric_name, stat, config=None):
        """
        Register a metric with this sensor

        Arguments:
            metric_name (MetricName): The name of the metric
            stat (AbstractMeasurableStat): The statistic to keep
            config (MetricConfig): A special configuration for this metric.
                If None use the sensor default configuration.
        """
        with self._lock:
            metric = KafkaMetric(metric_name, stat, config or self._config)
            self._registry.register_metric(metric)
            self._metrics.append(metric)
            self._stats.append(stat)

    def has_expired(self):
        """
        Return True if the Sensor is eligible for removal due to inactivity.
        """
        return ((time.time() * 1000 - self._last_record_time) >
                self._inactive_sensor_expiration_time_ms)
