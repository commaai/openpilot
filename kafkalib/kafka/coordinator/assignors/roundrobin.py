from __future__ import absolute_import

import collections
import itertools
import logging

from kafka.vendor import six

from kafka.coordinator.assignors.abstract import AbstractPartitionAssignor
from kafka.coordinator.protocol import ConsumerProtocolMemberMetadata, ConsumerProtocolMemberAssignment
from kafka.structs import TopicPartition

log = logging.getLogger(__name__)


class RoundRobinPartitionAssignor(AbstractPartitionAssignor):
    """
    The roundrobin assignor lays out all the available partitions and all the
    available consumers. It then proceeds to do a roundrobin assignment from
    partition to consumer. If the subscriptions of all consumer instances are
    identical, then the partitions will be uniformly distributed. (i.e., the
    partition ownership counts will be within a delta of exactly one across all
    consumers.)

    For example, suppose there are two consumers C0 and C1, two topics t0 and
    t1, and each topic has 3 partitions, resulting in partitions t0p0, t0p1,
    t0p2, t1p0, t1p1, and t1p2.

    The assignment will be:
        C0: [t0p0, t0p2, t1p1]
        C1: [t0p1, t1p0, t1p2]

    When subscriptions differ across consumer instances, the assignment process
    still considers each consumer instance in round robin fashion but skips
    over an instance if it is not subscribed to the topic. Unlike the case when
    subscriptions are identical, this can result in imbalanced assignments.

    For example, suppose we have three consumers C0, C1, C2, and three topics
    t0, t1, t2, with unbalanced partitions t0p0, t1p0, t1p1, t2p0, t2p1, t2p2,
    where C0 is subscribed to t0; C1 is subscribed to t0, t1; and C2 is
    subscribed to t0, t1, t2.

    The assignment will be:
        C0: [t0p0]
        C1: [t1p0]
        C2: [t1p1, t2p0, t2p1, t2p2]
    """
    name = 'roundrobin'
    version = 0

    @classmethod
    def assign(cls, cluster, member_metadata):
        all_topics = set()
        for metadata in six.itervalues(member_metadata):
            all_topics.update(metadata.subscription)

        all_topic_partitions = []
        for topic in all_topics:
            partitions = cluster.partitions_for_topic(topic)
            if partitions is None:
                log.warning('No partition metadata for topic %s', topic)
                continue
            for partition in partitions:
                all_topic_partitions.append(TopicPartition(topic, partition))
        all_topic_partitions.sort()

        # construct {member_id: {topic: [partition, ...]}}
        assignment = collections.defaultdict(lambda: collections.defaultdict(list))

        member_iter = itertools.cycle(sorted(member_metadata.keys()))
        for partition in all_topic_partitions:
            member_id = next(member_iter)

            # Because we constructed all_topic_partitions from the set of
            # member subscribed topics, we should be safe assuming that
            # each topic in all_topic_partitions is in at least one member
            # subscription; otherwise this could yield an infinite loop
            while partition.topic not in member_metadata[member_id].subscription:
                member_id = next(member_iter)
            assignment[member_id][partition.topic].append(partition.partition)

        protocol_assignment = {}
        for member_id in member_metadata:
            protocol_assignment[member_id] = ConsumerProtocolMemberAssignment(
                cls.version,
                sorted(assignment[member_id].items()),
                b'')
        return protocol_assignment

    @classmethod
    def metadata(cls, topics):
        return ConsumerProtocolMemberMetadata(cls.version, list(topics), b'')

    @classmethod
    def on_assignment(cls, assignment):
        pass
