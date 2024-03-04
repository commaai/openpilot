from __future__ import absolute_import

import io
import time

from kafka.codec import (has_gzip, has_snappy, has_lz4, has_zstd,
                     gzip_decode, snappy_decode, zstd_decode,
                     lz4_decode, lz4_decode_old_kafka)
from kafka.protocol.frame import KafkaBytes
from kafka.protocol.struct import Struct
from kafka.protocol.types import (
    Int8, Int32, Int64, Bytes, Schema, AbstractType
)
from kafka.util import crc32, WeakMethod


class Message(Struct):
    SCHEMAS = [
        Schema(
            ('crc', Int32),
            ('magic', Int8),
            ('attributes', Int8),
            ('key', Bytes),
            ('value', Bytes)),
        Schema(
            ('crc', Int32),
            ('magic', Int8),
            ('attributes', Int8),
            ('timestamp', Int64),
            ('key', Bytes),
            ('value', Bytes)),
    ]
    SCHEMA = SCHEMAS[1]
    CODEC_MASK = 0x07
    CODEC_GZIP = 0x01
    CODEC_SNAPPY = 0x02
    CODEC_LZ4 = 0x03
    CODEC_ZSTD = 0x04
    TIMESTAMP_TYPE_MASK = 0x08
    HEADER_SIZE = 22  # crc(4), magic(1), attributes(1), timestamp(8), key+value size(4*2)

    def __init__(self, value, key=None, magic=0, attributes=0, crc=0,
                 timestamp=None):
        assert value is None or isinstance(value, bytes), 'value must be bytes'
        assert key is None or isinstance(key, bytes), 'key must be bytes'
        assert magic > 0 or timestamp is None, 'timestamp not supported in v0'

        # Default timestamp to now for v1 messages
        if magic > 0 and timestamp is None:
            timestamp = int(time.time() * 1000)
        self.timestamp = timestamp
        self.crc = crc
        self._validated_crc = None
        self.magic = magic
        self.attributes = attributes
        self.key = key
        self.value = value
        self.encode = WeakMethod(self._encode_self)

    @property
    def timestamp_type(self):
        """0 for CreateTime; 1 for LogAppendTime; None if unsupported.

        Value is determined by broker; produced messages should always set to 0
        Requires Kafka >= 0.10 / message version >= 1
        """
        if self.magic == 0:
            return None
        elif self.attributes & self.TIMESTAMP_TYPE_MASK:
            return 1
        else:
            return 0

    def _encode_self(self, recalc_crc=True):
        version = self.magic
        if version == 1:
            fields = (self.crc, self.magic, self.attributes, self.timestamp, self.key, self.value)
        elif version == 0:
            fields = (self.crc, self.magic, self.attributes, self.key, self.value)
        else:
            raise ValueError('Unrecognized message version: %s' % (version,))
        message = Message.SCHEMAS[version].encode(fields)
        if not recalc_crc:
            return message
        self.crc = crc32(message[4:])
        crc_field = self.SCHEMAS[version].fields[0]
        return crc_field.encode(self.crc) + message[4:]

    @classmethod
    def decode(cls, data):
        _validated_crc = None
        if isinstance(data, bytes):
            _validated_crc = crc32(data[4:])
            data = io.BytesIO(data)
        # Partial decode required to determine message version
        base_fields = cls.SCHEMAS[0].fields[0:3]
        crc, magic, attributes = [field.decode(data) for field in base_fields]
        remaining = cls.SCHEMAS[magic].fields[3:]
        fields = [field.decode(data) for field in remaining]
        if magic == 1:
            timestamp = fields[0]
        else:
            timestamp = None
        msg = cls(fields[-1], key=fields[-2],
                  magic=magic, attributes=attributes, crc=crc,
                  timestamp=timestamp)
        msg._validated_crc = _validated_crc
        return msg

    def validate_crc(self):
        if self._validated_crc is None:
            raw_msg = self._encode_self(recalc_crc=False)
            self._validated_crc = crc32(raw_msg[4:])
        if self.crc == self._validated_crc:
            return True
        return False

    def is_compressed(self):
        return self.attributes & self.CODEC_MASK != 0

    def decompress(self):
        codec = self.attributes & self.CODEC_MASK
        assert codec in (self.CODEC_GZIP, self.CODEC_SNAPPY, self.CODEC_LZ4, self.CODEC_ZSTD)
        if codec == self.CODEC_GZIP:
            assert has_gzip(), 'Gzip decompression unsupported'
            raw_bytes = gzip_decode(self.value)
        elif codec == self.CODEC_SNAPPY:
            assert has_snappy(), 'Snappy decompression unsupported'
            raw_bytes = snappy_decode(self.value)
        elif codec == self.CODEC_LZ4:
            assert has_lz4(), 'LZ4 decompression unsupported'
            if self.magic == 0:
                raw_bytes = lz4_decode_old_kafka(self.value)
            else:
                raw_bytes = lz4_decode(self.value)
        elif codec == self.CODEC_ZSTD:
            assert has_zstd(), "ZSTD decompression unsupported"
            raw_bytes = zstd_decode(self.value)
        else:
            raise Exception('This should be impossible')

        return MessageSet.decode(raw_bytes, bytes_to_read=len(raw_bytes))

    def __hash__(self):
        return hash(self._encode_self(recalc_crc=False))


class PartialMessage(bytes):
    def __repr__(self):
        return 'PartialMessage(%s)' % (self,)


class MessageSet(AbstractType):
    ITEM = Schema(
        ('offset', Int64),
        ('message', Bytes)
    )
    HEADER_SIZE = 12  # offset + message_size

    @classmethod
    def encode(cls, items, prepend_size=True):
        # RecordAccumulator encodes messagesets internally
        if isinstance(items, (io.BytesIO, KafkaBytes)):
            size = Int32.decode(items)
            if prepend_size:
                # rewind and return all the bytes
                items.seek(items.tell() - 4)
                size += 4
            return items.read(size)

        encoded_values = []
        for (offset, message) in items:
            encoded_values.append(Int64.encode(offset))
            encoded_values.append(Bytes.encode(message))
        encoded = b''.join(encoded_values)
        if prepend_size:
            return Bytes.encode(encoded)
        else:
            return encoded

    @classmethod
    def decode(cls, data, bytes_to_read=None):
        """Compressed messages should pass in bytes_to_read (via message size)
        otherwise, we decode from data as Int32
        """
        if isinstance(data, bytes):
            data = io.BytesIO(data)
        if bytes_to_read is None:
            bytes_to_read = Int32.decode(data)

        # if FetchRequest max_bytes is smaller than the available message set
        # the server returns partial data for the final message
        # So create an internal buffer to avoid over-reading
        raw = io.BytesIO(data.read(bytes_to_read))

        items = []
        while bytes_to_read:
            try:
                offset = Int64.decode(raw)
                msg_bytes = Bytes.decode(raw)
                bytes_to_read -= 8 + 4 + len(msg_bytes)
                items.append((offset, len(msg_bytes), Message.decode(msg_bytes)))
            except ValueError:
                # PartialMessage to signal that max_bytes may be too small
                items.append((None, None, PartialMessage()))
                break
        return items

    @classmethod
    def repr(cls, messages):
        if isinstance(messages, (KafkaBytes, io.BytesIO)):
            offset = messages.tell()
            decoded = cls.decode(messages)
            messages.seek(offset)
            messages = decoded
        return str([cls.ITEM.repr(m) for m in messages])
