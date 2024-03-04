from __future__ import absolute_import

import logging
import warnings

from .base import Producer
from ..partitioner import HashedPartitioner


log = logging.getLogger(__name__)


class KeyedProducer(Producer):
    """
    A producer which distributes messages to partitions based on the key

    See Producer class for Arguments

    Additional Arguments:
        partitioner: A partitioner class that will be used to get the partition
            to send the message to. Must be derived from Partitioner.
            Defaults to HashedPartitioner.
    """
    def __init__(self, *args, **kwargs):
        self.partitioner_class = kwargs.pop('partitioner', HashedPartitioner)
        self.partitioners = {}
        super(KeyedProducer, self).__init__(*args, **kwargs)

    def _next_partition(self, topic, key):
        if topic not in self.partitioners:
            if not self.client.has_metadata_for_topic(topic):
                self.client.load_metadata_for_topics(topic, ignore_leadernotavailable=True)

            self.partitioners[topic] = self.partitioner_class(self.client.get_partition_ids_for_topic(topic))

        partitioner = self.partitioners[topic]
        return partitioner.partition(key)

    def send_messages(self, topic, key, *msg):
        partition = self._next_partition(topic, key)
        return self._send_messages(topic, partition, *msg, key=key)

    # DEPRECATED
    def send(self, topic, key, msg):
        warnings.warn("KeyedProducer.send is deprecated in favor of send_messages", DeprecationWarning)
        return self.send_messages(topic, key, msg)

    def __repr__(self):
        return '<KeyedProducer batch=%s>' % self.async
