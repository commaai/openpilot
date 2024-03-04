from __future__ import absolute_import

import logging
import struct

from kafka.vendor import six  # pylint: disable=import-error

import kafka.protocol.commit
import kafka.protocol.fetch
import kafka.protocol.message
import kafka.protocol.metadata
import kafka.protocol.offset
import kafka.protocol.produce
import kafka.structs

from kafka.codec import gzip_encode, snappy_encode
from kafka.errors import ProtocolError, UnsupportedCodecError
from kafka.structs import ConsumerMetadataResponse
from kafka.util import (
    crc32, read_short_string, relative_unpack,
    write_int_string, group_by_topic_and_partition)


log = logging.getLogger(__name__)

ATTRIBUTE_CODEC_MASK = 0x03
CODEC_NONE = 0x00
CODEC_GZIP = 0x01
CODEC_SNAPPY = 0x02
ALL_CODECS = (CODEC_NONE, CODEC_GZIP, CODEC_SNAPPY)


class KafkaProtocol(object):
    """
    Class to encapsulate all of the protocol encoding/decoding.
    This class does not have any state associated with it, it is purely
    for organization.
    """
    PRODUCE_KEY = 0
    FETCH_KEY = 1
    OFFSET_KEY = 2
    METADATA_KEY = 3
    OFFSET_COMMIT_KEY = 8
    OFFSET_FETCH_KEY = 9
    CONSUMER_METADATA_KEY = 10

    ###################
    #   Private API   #
    ###################

    @classmethod
    def _encode_message_header(cls, client_id, correlation_id, request_key,
                               version=0):
        """
        Encode the common request envelope
        """
        return struct.pack('>hhih%ds' % len(client_id),
                           request_key,          # ApiKey
                           version,              # ApiVersion
                           correlation_id,       # CorrelationId
                           len(client_id),       # ClientId size
                           client_id)            # ClientId

    @classmethod
    def _encode_message_set(cls, messages):
        """
        Encode a MessageSet. Unlike other arrays in the protocol,
        MessageSets are not length-prefixed

        Format
        ======
        MessageSet => [Offset MessageSize Message]
          Offset => int64
          MessageSize => int32
        """
        message_set = []
        for message in messages:
            encoded_message = KafkaProtocol._encode_message(message)
            message_set.append(struct.pack('>qi%ds' % len(encoded_message), 0,
                                           len(encoded_message),
                                           encoded_message))
        return b''.join(message_set)

    @classmethod
    def _encode_message(cls, message):
        """
        Encode a single message.

        The magic number of a message is a format version number.
        The only supported magic number right now is zero

        Format
        ======
        Message => Crc MagicByte Attributes Key Value
          Crc => int32
          MagicByte => int8
          Attributes => int8
          Key => bytes
          Value => bytes
        """
        if message.magic == 0:
            msg = b''.join([
                struct.pack('>BB', message.magic, message.attributes),
                write_int_string(message.key),
                write_int_string(message.value)
            ])
            crc = crc32(msg)
            msg = struct.pack('>i%ds' % len(msg), crc, msg)
        else:
            raise ProtocolError("Unexpected magic number: %d" % message.magic)
        return msg

    ##################
    #   Public API   #
    ##################

    @classmethod
    def encode_produce_request(cls, payloads=(), acks=1, timeout=1000):
        """
        Encode a ProduceRequest struct

        Arguments:
            payloads: list of ProduceRequestPayload
            acks: How "acky" you want the request to be
                1: written to disk by the leader
                0: immediate response
                -1: waits for all replicas to be in sync
            timeout: Maximum time (in ms) the server will wait for replica acks.
                This is _not_ a socket timeout

        Returns: ProduceRequest
        """
        if acks not in (1, 0, -1):
            raise ValueError('ProduceRequest acks (%s) must be 1, 0, -1' % acks)

        topics = []
        for topic, topic_payloads in group_by_topic_and_partition(payloads).items():
            topic_msgs = []
            for partition, payload in topic_payloads.items():
                partition_msgs = []
                for msg in payload.messages:
                    m = kafka.protocol.message.Message(
                          msg.value, key=msg.key,
                          magic=msg.magic, attributes=msg.attributes
                    )
                    partition_msgs.append((0, m.encode()))
                topic_msgs.append((partition, partition_msgs))
            topics.append((topic, topic_msgs))


        return kafka.protocol.produce.ProduceRequest[0](
            required_acks=acks,
            timeout=timeout,
            topics=topics
        )

    @classmethod
    def decode_produce_response(cls, response):
        """
        Decode ProduceResponse to ProduceResponsePayload

        Arguments:
            response: ProduceResponse

        Return: list of ProduceResponsePayload
        """
        return [
            kafka.structs.ProduceResponsePayload(topic, partition, error, offset)
            for topic, partitions in response.topics
            for partition, error, offset in partitions
        ]

    @classmethod
    def encode_fetch_request(cls, payloads=(), max_wait_time=100, min_bytes=4096):
        """
        Encodes a FetchRequest struct

        Arguments:
            payloads: list of FetchRequestPayload
            max_wait_time (int, optional): ms to block waiting for min_bytes
                data. Defaults to 100.
            min_bytes (int, optional): minimum bytes required to return before
                max_wait_time. Defaults to 4096.

        Return: FetchRequest
        """
        return kafka.protocol.fetch.FetchRequest[0](
            replica_id=-1,
            max_wait_time=max_wait_time,
            min_bytes=min_bytes,
            topics=[(
                topic,
                [(
                    partition,
                    payload.offset,
                    payload.max_bytes)
                for partition, payload in topic_payloads.items()])
            for topic, topic_payloads in group_by_topic_and_partition(payloads).items()])

    @classmethod
    def decode_fetch_response(cls, response):
        """
        Decode FetchResponse struct to FetchResponsePayloads

        Arguments:
            response: FetchResponse
        """
        return [
            kafka.structs.FetchResponsePayload(
                topic, partition, error, highwater_offset, [
                    offset_and_msg
                    for offset_and_msg in cls.decode_message_set(messages)])
            for topic, partitions in response.topics
                for partition, error, highwater_offset, messages in partitions
        ]

    @classmethod
    def decode_message_set(cls, messages):
        for offset, _, message in messages:
            if isinstance(message, kafka.protocol.message.Message) and message.is_compressed():
                inner_messages = message.decompress()
                for (inner_offset, _msg_size, inner_msg) in inner_messages:
                    yield kafka.structs.OffsetAndMessage(inner_offset, inner_msg)
            else:
                yield kafka.structs.OffsetAndMessage(offset, message)

    @classmethod
    def encode_offset_request(cls, payloads=()):
        return kafka.protocol.offset.OffsetRequest[0](
            replica_id=-1,
            topics=[(
                topic,
                [(
                    partition,
                    payload.time,
                    payload.max_offsets)
                for partition, payload in six.iteritems(topic_payloads)])
            for topic, topic_payloads in six.iteritems(group_by_topic_and_partition(payloads))])

    @classmethod
    def decode_offset_response(cls, response):
        """
        Decode OffsetResponse into OffsetResponsePayloads

        Arguments:
            response: OffsetResponse

        Returns: list of OffsetResponsePayloads
        """
        return [
            kafka.structs.OffsetResponsePayload(topic, partition, error, tuple(offsets))
            for topic, partitions in response.topics
            for partition, error, offsets in partitions
        ]

    @classmethod
    def encode_list_offset_request(cls, payloads=()):
        return kafka.protocol.offset.OffsetRequest[1](
            replica_id=-1,
            topics=[(
                topic,
                [(
                    partition,
                    payload.time)
                for partition, payload in six.iteritems(topic_payloads)])
            for topic, topic_payloads in six.iteritems(group_by_topic_and_partition(payloads))])

    @classmethod
    def decode_list_offset_response(cls, response):
        """
        Decode OffsetResponse_v2 into ListOffsetResponsePayloads

        Arguments:
            response: OffsetResponse_v2

        Returns: list of ListOffsetResponsePayloads
        """
        return [
            kafka.structs.ListOffsetResponsePayload(topic, partition, error, timestamp, offset)
            for topic, partitions in response.topics
            for partition, error, timestamp, offset in partitions
        ]


    @classmethod
    def encode_metadata_request(cls, topics=(), payloads=None):
        """
        Encode a MetadataRequest

        Arguments:
            topics: list of strings
        """
        if payloads is not None:
            topics = payloads

        return kafka.protocol.metadata.MetadataRequest[0](topics)

    @classmethod
    def decode_metadata_response(cls, response):
        return response

    @classmethod
    def encode_consumer_metadata_request(cls, client_id, correlation_id, payloads):
        """
        Encode a ConsumerMetadataRequest

        Arguments:
            client_id: string
            correlation_id: int
            payloads: string (consumer group)
        """
        message = []
        message.append(cls._encode_message_header(client_id, correlation_id,
                                                  KafkaProtocol.CONSUMER_METADATA_KEY))
        message.append(struct.pack('>h%ds' % len(payloads), len(payloads), payloads))

        msg = b''.join(message)
        return write_int_string(msg)

    @classmethod
    def decode_consumer_metadata_response(cls, data):
        """
        Decode bytes to a ConsumerMetadataResponse

        Arguments:
            data: bytes to decode
        """
        ((correlation_id, error, nodeId), cur) = relative_unpack('>ihi', data, 0)
        (host, cur) = read_short_string(data, cur)
        ((port,), cur) = relative_unpack('>i', data, cur)

        return ConsumerMetadataResponse(error, nodeId, host, port)

    @classmethod
    def encode_offset_commit_request(cls, group, payloads):
        """
        Encode an OffsetCommitRequest struct

        Arguments:
            group: string, the consumer group you are committing offsets for
            payloads: list of OffsetCommitRequestPayload
        """
        return kafka.protocol.commit.OffsetCommitRequest[0](
            consumer_group=group,
            topics=[(
                topic,
                [(
                    partition,
                    payload.offset,
                    payload.metadata)
                for partition, payload in six.iteritems(topic_payloads)])
            for topic, topic_payloads in six.iteritems(group_by_topic_and_partition(payloads))])

    @classmethod
    def decode_offset_commit_response(cls, response):
        """
        Decode OffsetCommitResponse to an OffsetCommitResponsePayload

        Arguments:
            response: OffsetCommitResponse
        """
        return [
            kafka.structs.OffsetCommitResponsePayload(topic, partition, error)
            for topic, partitions in response.topics
            for partition, error in partitions
        ]

    @classmethod
    def encode_offset_fetch_request(cls, group, payloads, from_kafka=False):
        """
        Encode an OffsetFetchRequest struct. The request is encoded using
        version 0 if from_kafka is false, indicating a request for Zookeeper
        offsets. It is encoded using version 1 otherwise, indicating a request
        for Kafka offsets.

        Arguments:
            group: string, the consumer group you are fetching offsets for
            payloads: list of OffsetFetchRequestPayload
            from_kafka: bool, default False, set True for Kafka-committed offsets
        """
        version = 1 if from_kafka else 0
        return kafka.protocol.commit.OffsetFetchRequest[version](
            consumer_group=group,
            topics=[(
                topic,
                list(topic_payloads.keys()))
            for topic, topic_payloads in six.iteritems(group_by_topic_and_partition(payloads))])

    @classmethod
    def decode_offset_fetch_response(cls, response):
        """
        Decode OffsetFetchResponse to OffsetFetchResponsePayloads

        Arguments:
            response: OffsetFetchResponse
        """
        return [
            kafka.structs.OffsetFetchResponsePayload(
                topic, partition, offset, metadata, error
            )
            for topic, partitions in response.topics
            for partition, offset, metadata, error in partitions
        ]


def create_message(payload, key=None):
    """
    Construct a Message

    Arguments:
        payload: bytes, the payload to send to Kafka
        key: bytes, a key used for partition routing (optional)

    """
    return kafka.structs.Message(0, 0, key, payload)


def create_gzip_message(payloads, key=None, compresslevel=None):
    """
    Construct a Gzipped Message containing multiple Messages

    The given payloads will be encoded, compressed, and sent as a single atomic
    message to Kafka.

    Arguments:
        payloads: list(bytes), a list of payload to send be sent to Kafka
        key: bytes, a key used for partition routing (optional)

    """
    message_set = KafkaProtocol._encode_message_set(
        [create_message(payload, pl_key) for payload, pl_key in payloads])

    gzipped = gzip_encode(message_set, compresslevel=compresslevel)
    codec = ATTRIBUTE_CODEC_MASK & CODEC_GZIP

    return kafka.structs.Message(0, 0x00 | codec, key, gzipped)


def create_snappy_message(payloads, key=None):
    """
    Construct a Snappy Message containing multiple Messages

    The given payloads will be encoded, compressed, and sent as a single atomic
    message to Kafka.

    Arguments:
        payloads: list(bytes), a list of payload to send be sent to Kafka
        key: bytes, a key used for partition routing (optional)

    """
    message_set = KafkaProtocol._encode_message_set(
        [create_message(payload, pl_key) for payload, pl_key in payloads])

    snapped = snappy_encode(message_set)
    codec = ATTRIBUTE_CODEC_MASK & CODEC_SNAPPY

    return kafka.structs.Message(0, 0x00 | codec, key, snapped)


def create_message_set(messages, codec=CODEC_NONE, key=None, compresslevel=None):
    """Create a message set using the given codec.

    If codec is CODEC_NONE, return a list of raw Kafka messages. Otherwise,
    return a list containing a single codec-encoded message.
    """
    if codec == CODEC_NONE:
        return [create_message(m, k) for m, k in messages]
    elif codec == CODEC_GZIP:
        return [create_gzip_message(messages, key, compresslevel)]
    elif codec == CODEC_SNAPPY:
        return [create_snappy_message(messages, key)]
    else:
        raise UnsupportedCodecError("Codec 0x%02x unsupported" % codec)
