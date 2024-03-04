from __future__ import absolute_import

from kafka.protocol.struct import Struct
from kafka.protocol.types import Array, Bytes, Int16, Int32, Schema, String
from kafka.structs import TopicPartition


class ConsumerProtocolMemberMetadata(Struct):
    SCHEMA = Schema(
        ('version', Int16),
        ('subscription', Array(String('utf-8'))),
        ('user_data', Bytes))


class ConsumerProtocolMemberAssignment(Struct):
    SCHEMA = Schema(
        ('version', Int16),
        ('assignment', Array(
            ('topic', String('utf-8')),
            ('partitions', Array(Int32)))),
        ('user_data', Bytes))

    def partitions(self):
        return [TopicPartition(topic, partition)
                for topic, partitions in self.assignment # pylint: disable-msg=no-member
                for partition in partitions]


class ConsumerProtocol(object):
    PROTOCOL_TYPE = 'consumer'
    ASSIGNMENT_STRATEGIES = ('range', 'roundrobin')
    METADATA = ConsumerProtocolMemberMetadata
    ASSIGNMENT = ConsumerProtocolMemberAssignment
