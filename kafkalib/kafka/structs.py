""" Other useful structs """
from __future__ import absolute_import

from collections import namedtuple


"""A topic and partition tuple

Keyword Arguments:
    topic (str): A topic name
    partition (int): A partition id
"""
TopicPartition = namedtuple("TopicPartition",
    ["topic", "partition"])


"""A Kafka broker metadata used by admin tools.

Keyword Arguments:
    nodeID (int): The Kafka broker id.
    host (str): The Kafka broker hostname.
    port (int): The Kafka broker port.
    rack (str): The rack of the broker, which is used to in rack aware
                partition assignment for fault tolerance.
    Examples: `RACK1`, `us-east-1d`. Default: None
"""
BrokerMetadata = namedtuple("BrokerMetadata",
    ["nodeId", "host", "port", "rack"])


"""A topic partition metadata describing the state in the MetadataResponse.

Keyword Arguments:
    topic (str): The topic name of the partition this metadata relates to.
    partition (int): The id of the partition this metadata relates to.
    leader (int): The id of the broker that is the leader for the partition.
    replicas (List[int]): The ids of all brokers that contain replicas of the
                          partition.
    isr (List[int]): The ids of all brokers that contain in-sync replicas of
                     the partition.
    error (KafkaError): A KafkaError object associated with the request for
                        this partition metadata.
"""
PartitionMetadata = namedtuple("PartitionMetadata",
    ["topic", "partition", "leader", "replicas", "isr", "error"])


"""The Kafka offset commit API

The Kafka offset commit API allows users to provide additional metadata
(in the form of a string) when an offset is committed. This can be useful
(for example) to store information about which node made the commit,
what time the commit was made, etc.

Keyword Arguments:
    offset (int): The offset to be committed
    metadata (str): Non-null metadata
"""
OffsetAndMetadata = namedtuple("OffsetAndMetadata",
    # TODO add leaderEpoch: OffsetAndMetadata(offset, leaderEpoch, metadata)
    ["offset", "metadata"])


"""An offset and timestamp tuple

Keyword Arguments:
    offset (int): An offset
    timestamp (int): The timestamp associated to the offset
"""
OffsetAndTimestamp = namedtuple("OffsetAndTimestamp",
    ["offset", "timestamp"])

MemberInformation = namedtuple("MemberInformation",
    ["member_id", "client_id", "client_host", "member_metadata", "member_assignment"])

GroupInformation = namedtuple("GroupInformation",
    ["error_code", "group", "state", "protocol_type", "protocol", "members", "authorized_operations"])

"""Define retry policy for async producer

Keyword Arguments:
    Limit (int): Number of retries. limit >= 0, 0 means no retries
    backoff_ms (int): Milliseconds to backoff.
    retry_on_timeouts:
"""
RetryOptions = namedtuple("RetryOptions",
    ["limit", "backoff_ms", "retry_on_timeouts"])
