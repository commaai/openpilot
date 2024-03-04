from __future__ import absolute_import

import logging
import sys
import time
import threading

from kafka.metrics import AnonMeasurable, KafkaMetric, MetricConfig, MetricName
from kafka.metrics.stats import Sensor

logger = logging.getLogger(__name__)


class Metrics(object):
    """
    A registry of sensors and metrics.

    A metric is a named, numerical measurement. A sensor is a handle to
    record numerical measurements as they occur. Each Sensor has zero or
    more associated metrics. For example a Sensor might represent message
    sizes and we might associate with this sensor a metric for the average,
    maximum, or other statistics computed off the sequence of message sizes
    that are recorded by the sensor.

    Usage looks something like this:
        # set up metrics:
        metrics = Metrics() # the global repository of metrics and sensors
        sensor = metrics.sensor('message-sizes')
        metric_name = MetricName('message-size-avg', 'producer-metrics')
        sensor.add(metric_name, Avg())
        metric_name = MetricName('message-size-max', 'producer-metrics')
        sensor.add(metric_name, Max())

        # as messages are sent we record the sizes
        sensor.record(message_size);
    """
    def __init__(self, default_config=None, reporters=None,
                 enable_expiration=False):
        """
        Create a metrics repository with a default config, given metric
        reporters and the ability to expire eligible sensors

        Arguments:
            default_config (MetricConfig, optional): The default config
            reporters (list of AbstractMetricsReporter, optional):
                The metrics reporters
            enable_expiration (bool, optional): true if the metrics instance
                can garbage collect inactive sensors, false otherwise
        """
        self._lock = threading.RLock()
        self._config = default_config or MetricConfig()
        self._sensors = {}
        self._metrics = {}
        self._children_sensors = {}
        self._reporters = reporters or []
        for reporter in self._reporters:
            reporter.init([])

        if enable_expiration:
            def expire_loop():
                while True:
                    # delay 30 seconds
                    time.sleep(30)
                    self.ExpireSensorTask.run(self)
            metrics_scheduler = threading.Thread(target=expire_loop)
            # Creating a daemon thread to not block shutdown
            metrics_scheduler.daemon = True
            metrics_scheduler.start()

        self.add_metric(self.metric_name('count', 'kafka-metrics-count',
                                         'total number of registered metrics'),
                        AnonMeasurable(lambda config, now: len(self._metrics)))

    @property
    def config(self):
        return self._config

    @property
    def metrics(self):
        """
        Get all the metrics currently maintained and indexed by metricName
        """
        return self._metrics

    def metric_name(self, name, group, description='', tags=None):
        """
        Create a MetricName with the given name, group, description and tags,
        plus default tags specified in the metric configuration.
        Tag in tags takes precedence if the same tag key is specified in
        the default metric configuration.

        Arguments:
            name (str): The name of the metric
            group (str): logical group name of the metrics to which this
                metric belongs
            description (str, optional): A human-readable description to
                include in the metric
            tags (dict, optionals): additional key/value attributes of
                the metric
        """
        combined_tags = dict(self.config.tags)
        combined_tags.update(tags or {})
        return MetricName(name, group, description, combined_tags)

    def get_sensor(self, name):
        """
        Get the sensor with the given name if it exists

        Arguments:
            name (str): The name of the sensor

        Returns:
            Sensor: The sensor or None if no such sensor exists
        """
        if not name:
            raise ValueError('name must be non-empty')
        return self._sensors.get(name, None)

    def sensor(self, name, config=None,
               inactive_sensor_expiration_time_seconds=sys.maxsize,
               parents=None):
        """
        Get or create a sensor with the given unique name and zero or
        more parent sensors. All parent sensors will receive every value
        recorded with this sensor.

        Arguments:
            name (str): The name of the sensor
            config (MetricConfig, optional): A default configuration to use
                for this sensor for metrics that don't have their own config
            inactive_sensor_expiration_time_seconds (int, optional):
                If no value if recorded on the Sensor for this duration of
                time, it is eligible for removal
            parents (list of Sensor): The parent sensors

        Returns:
            Sensor: The sensor that is created
        """
        sensor = self.get_sensor(name)
        if sensor:
            return sensor

        with self._lock:
            sensor = self.get_sensor(name)
            if not sensor:
                sensor = Sensor(self, name, parents, config or self.config,
                                inactive_sensor_expiration_time_seconds)
                self._sensors[name] = sensor
                if parents:
                    for parent in parents:
                        children = self._children_sensors.get(parent)
                        if not children:
                            children = []
                            self._children_sensors[parent] = children
                        children.append(sensor)
                logger.debug('Added sensor with name %s', name)
            return sensor

    def remove_sensor(self, name):
        """
        Remove a sensor (if it exists), associated metrics and its children.

        Arguments:
            name (str): The name of the sensor to be removed
        """
        sensor = self._sensors.get(name)
        if sensor:
            child_sensors = None
            with sensor._lock:
                with self._lock:
                    val = self._sensors.pop(name, None)
                    if val and val == sensor:
                        for metric in sensor.metrics:
                            self.remove_metric(metric.metric_name)
                        logger.debug('Removed sensor with name %s', name)
                        child_sensors = self._children_sensors.pop(sensor, None)
            if child_sensors:
                for child_sensor in child_sensors:
                    self.remove_sensor(child_sensor.name)

    def add_metric(self, metric_name, measurable, config=None):
        """
        Add a metric to monitor an object that implements measurable.
        This metric won't be associated with any sensor.
        This is a way to expose existing values as metrics.

        Arguments:
            metricName (MetricName): The name of the metric
            measurable (AbstractMeasurable): The measurable that will be
                measured by this metric
            config (MetricConfig, optional): The configuration to use when
                measuring this measurable
        """
        # NOTE there was a lock here, but i don't think it's needed
        metric = KafkaMetric(metric_name, measurable, config or self.config)
        self.register_metric(metric)

    def remove_metric(self, metric_name):
        """
        Remove a metric if it exists and return it. Return None otherwise.
        If a metric is removed, `metric_removal` will be invoked
        for each reporter.

        Arguments:
            metric_name (MetricName): The name of the metric

        Returns:
            KafkaMetric: the removed `KafkaMetric` or None if no such
                metric exists
        """
        with self._lock:
            metric = self._metrics.pop(metric_name, None)
            if metric:
                for reporter in self._reporters:
                    reporter.metric_removal(metric)
            return metric

    def add_reporter(self, reporter):
        """Add a MetricReporter"""
        with self._lock:
            reporter.init(list(self.metrics.values()))
            self._reporters.append(reporter)

    def register_metric(self, metric):
        with self._lock:
            if metric.metric_name in self.metrics:
                raise ValueError('A metric named "%s" already exists, cannot'
                                 ' register another one.' % (metric.metric_name,))
            self.metrics[metric.metric_name] = metric
            for reporter in self._reporters:
                reporter.metric_change(metric)

    class ExpireSensorTask(object):
        """
        This iterates over every Sensor and triggers a remove_sensor
        if it has expired. Package private for testing
        """
        @staticmethod
        def run(metrics):
            items = list(metrics._sensors.items())
            for name, sensor in items:
                # remove_sensor also locks the sensor object. This is fine
                # because synchronized is reentrant. There is however a minor
                # race condition here. Assume we have a parent sensor P and
                # child sensor C. Calling record on C would cause a record on
                # P as well. So expiration time for P == expiration time for C.
                # If the record on P happens via C just after P is removed,
                # that will cause C to also get removed. Since the expiration
                # time is typically high it is not expected to be a significant
                # concern and thus not necessary to optimize
                with sensor._lock:
                    if sensor.has_expired():
                        logger.debug('Removing expired sensor %s', name)
                        metrics.remove_sensor(name)

    def close(self):
        """Close this metrics repository."""
        for reporter in self._reporters:
            reporter.close()

        self._metrics.clear()
