from __future__ import absolute_import

import copy


class MetricName(object):
    """
    This class encapsulates a metric's name, logical group and its
    related attributes (tags).

    group, tags parameters can be used to create unique metric names.
    e.g. domainName:type=group,key1=val1,key2=val2

    Usage looks something like this:

        # set up metrics:
        metric_tags = {'client-id': 'producer-1', 'topic': 'topic'}
        metric_config = MetricConfig(tags=metric_tags)

        # metrics is the global repository of metrics and sensors
        metrics = Metrics(metric_config)

        sensor = metrics.sensor('message-sizes')
        metric_name = metrics.metric_name('message-size-avg',
                                          'producer-metrics',
                                          'average message size')
        sensor.add(metric_name, Avg())

        metric_name = metrics.metric_name('message-size-max',
        sensor.add(metric_name, Max())

        tags = {'client-id': 'my-client', 'topic': 'my-topic'}
        metric_name = metrics.metric_name('message-size-min',
                                          'producer-metrics',
                                          'message minimum size', tags)
        sensor.add(metric_name, Min())

        # as messages are sent we record the sizes
        sensor.record(message_size)
    """

    def __init__(self, name, group, description=None, tags=None):
        """
        Arguments:
            name (str): The name of the metric.
            group (str): The logical group name of the metrics to which this
                metric belongs.
            description (str, optional): A human-readable description to
                include in the metric.
            tags (dict, optional): Additional key/val attributes of the metric.
        """
        if not (name and group):
            raise ValueError('name and group must be non-empty.')
        if tags is not None and not isinstance(tags, dict):
            raise ValueError('tags must be a dict if present.')

        self._name = name
        self._group = group
        self._description = description
        self._tags = copy.copy(tags)
        self._hash = 0

    @property
    def name(self):
        return self._name

    @property
    def group(self):
        return self._group

    @property
    def description(self):
        return self._description

    @property
    def tags(self):
        return copy.copy(self._tags)

    def __hash__(self):
        if self._hash != 0:
            return self._hash
        prime = 31
        result = 1
        result = prime * result + hash(self.group)
        result = prime * result + hash(self.name)
        tags_hash = hash(frozenset(self.tags.items())) if self.tags else 0
        result = prime * result + tags_hash
        self._hash = result
        return result

    def __eq__(self, other):
        if self is other:
            return True
        if other is None:
            return False
        return (type(self) == type(other) and
                self.group == other.group and
                self.name == other.name and
                self.tags == other.tags)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return 'MetricName(name=%s, group=%s, description=%s, tags=%s)' % (
            self.name, self.group, self.description, self.tags)
