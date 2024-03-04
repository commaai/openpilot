from __future__ import absolute_import

from itertools import cycle
import logging
import random

from kafka.vendor.six.moves import xrange # pylint: disable=import-error

from .base import Producer


log = logging.getLogger(__name__)


class SimpleProducer(Producer):
    """A simple, round-robin producer.

    See Producer class for Base Arguments

    Additional Arguments:
        random_start (bool, optional): randomize the initial partition which
            the first message block will be published to, otherwise
            if false, the first message block will always publish
            to partition 0 before cycling through each partition,
            defaults to True.
    """
    def __init__(self, *args, **kwargs):
        self.partition_cycles = {}
        self.random_start = kwargs.pop('random_start', True)
        super(SimpleProducer, self).__init__(*args, **kwargs)

    def _next_partition(self, topic):
        if topic not in self.partition_cycles:
            if not self.client.has_metadata_for_topic(topic):
                self.client.ensure_topic_exists(topic)

            self.partition_cycles[topic] = cycle(self.client.get_partition_ids_for_topic(topic))

            # Randomize the initial partition that is returned
            if self.random_start:
                num_partitions = len(self.client.get_partition_ids_for_topic(topic))
                for _ in xrange(random.randint(0, num_partitions-1)):
                    next(self.partition_cycles[topic])

        return next(self.partition_cycles[topic])

    def send_messages(self, topic, *msg):
        partition = self._next_partition(topic)
        return super(SimpleProducer, self).send_messages(
            topic, partition, *msg
        )

    def __repr__(self):
        return '<SimpleProducer batch=%s>' % self.async
