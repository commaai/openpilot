# See:
# https://github.com/apache/kafka/blob/trunk/clients/src/main/java/org/\
#    apache/kafka/common/record/LegacyRecord.java

# Builder and reader implementation for V0 and V1 record versions. As of Kafka
# 0.11.0.0 those were replaced with V2, thus the Legacy naming.

# The schema is given below (see
# https://kafka.apache.org/protocol#protocol_message_sets for more details):

# MessageSet => [Offset MessageSize Message]
#   Offset => int64
#   MessageSize => int32

# v0
# Message => Crc MagicByte Attributes Key Value
#   Crc => int32
#   MagicByte => int8
#   Attributes => int8
#   Key => bytes
#   Value => bytes

# v1 (supported since 0.10.0)
# Message => Crc MagicByte Attributes Key Value
#   Crc => int32
#   MagicByte => int8
#   Attributes => int8
#   Timestamp => int64
#   Key => bytes
#   Value => bytes

# The message attribute bits are given below:
#   * Unused (4-7)
#   * Timestamp Type (3) (added in V1)
#   * Compression Type (0-2)

# Note that when compression is enabled (see attributes above), the whole
# array of MessageSet's is compressed and places into a message as the `value`.
# Only the parent message is marked with `compression` bits in attributes.

# The CRC covers the data from the Magic byte to the end of the message.


import struct
import time

from kafka.record.abc import ABCRecord, ABCRecordBatch, ABCRecordBatchBuilder
from kafka.record.util import calc_crc32

from kafka.codec import (
    gzip_encode, snappy_encode, lz4_encode, lz4_encode_old_kafka,
    gzip_decode, snappy_decode, lz4_decode, lz4_decode_old_kafka,
)
import kafka.codec as codecs
from kafka.errors import CorruptRecordException, UnsupportedCodecError


class LegacyRecordBase(object):

    __slots__ = ()

    HEADER_STRUCT_V0 = struct.Struct(
        ">q"  # BaseOffset => Int64
        "i"  # Length => Int32
        "I"  # CRC => Int32
        "b"  # Magic => Int8
        "b"  # Attributes => Int8
    )
    HEADER_STRUCT_V1 = struct.Struct(
        ">q"  # BaseOffset => Int64
        "i"  # Length => Int32
        "I"  # CRC => Int32
        "b"  # Magic => Int8
        "b"  # Attributes => Int8
        "q"  # timestamp => Int64
    )

    LOG_OVERHEAD = CRC_OFFSET = struct.calcsize(
        ">q"  # Offset
        "i"   # Size
    )
    MAGIC_OFFSET = LOG_OVERHEAD + struct.calcsize(
        ">I"  # CRC
    )
    # Those are used for fast size calculations
    RECORD_OVERHEAD_V0 = struct.calcsize(
        ">I"  # CRC
        "b"   # magic
        "b"   # attributes
        "i"   # Key length
        "i"   # Value length
    )
    RECORD_OVERHEAD_V1 = struct.calcsize(
        ">I"  # CRC
        "b"   # magic
        "b"   # attributes
        "q"   # timestamp
        "i"   # Key length
        "i"   # Value length
    )

    KEY_OFFSET_V0 = HEADER_STRUCT_V0.size
    KEY_OFFSET_V1 = HEADER_STRUCT_V1.size
    KEY_LENGTH = VALUE_LENGTH = struct.calcsize(">i")  # Bytes length is Int32

    CODEC_MASK = 0x07
    CODEC_NONE = 0x00
    CODEC_GZIP = 0x01
    CODEC_SNAPPY = 0x02
    CODEC_LZ4 = 0x03
    TIMESTAMP_TYPE_MASK = 0x08

    LOG_APPEND_TIME = 1
    CREATE_TIME = 0

    NO_TIMESTAMP = -1

    def _assert_has_codec(self, compression_type):
        if compression_type == self.CODEC_GZIP:
            checker, name = codecs.has_gzip, "gzip"
        elif compression_type == self.CODEC_SNAPPY:
            checker, name = codecs.has_snappy, "snappy"
        elif compression_type == self.CODEC_LZ4:
            checker, name = codecs.has_lz4, "lz4"
        if not checker():
            raise UnsupportedCodecError(
                "Libraries for {} compression codec not found".format(name))


class LegacyRecordBatch(ABCRecordBatch, LegacyRecordBase):

    __slots__ = ("_buffer", "_magic", "_offset", "_crc", "_timestamp",
                 "_attributes", "_decompressed")

    def __init__(self, buffer, magic):
        self._buffer = memoryview(buffer)
        self._magic = magic

        offset, length, crc, magic_, attrs, timestamp = self._read_header(0)
        assert length == len(buffer) - self.LOG_OVERHEAD
        assert magic == magic_

        self._offset = offset
        self._crc = crc
        self._timestamp = timestamp
        self._attributes = attrs
        self._decompressed = False

    @property
    def timestamp_type(self):
        """0 for CreateTime; 1 for LogAppendTime; None if unsupported.

        Value is determined by broker; produced messages should always set to 0
        Requires Kafka >= 0.10 / message version >= 1
        """
        if self._magic == 0:
            return None
        elif self._attributes & self.TIMESTAMP_TYPE_MASK:
            return 1
        else:
            return 0

    @property
    def compression_type(self):
        return self._attributes & self.CODEC_MASK

    def validate_crc(self):
        crc = calc_crc32(self._buffer[self.MAGIC_OFFSET:])
        return self._crc == crc

    def _decompress(self, key_offset):
        # Copy of `_read_key_value`, but uses memoryview
        pos = key_offset
        key_size = struct.unpack_from(">i", self._buffer, pos)[0]
        pos += self.KEY_LENGTH
        if key_size != -1:
            pos += key_size
        value_size = struct.unpack_from(">i", self._buffer, pos)[0]
        pos += self.VALUE_LENGTH
        if value_size == -1:
            raise CorruptRecordException("Value of compressed message is None")
        else:
            data = self._buffer[pos:pos + value_size]

        compression_type = self.compression_type
        self._assert_has_codec(compression_type)
        if compression_type == self.CODEC_GZIP:
            uncompressed = gzip_decode(data)
        elif compression_type == self.CODEC_SNAPPY:
            uncompressed = snappy_decode(data.tobytes())
        elif compression_type == self.CODEC_LZ4:
            if self._magic == 0:
                uncompressed = lz4_decode_old_kafka(data.tobytes())
            else:
                uncompressed = lz4_decode(data.tobytes())
        return uncompressed

    def _read_header(self, pos):
        if self._magic == 0:
            offset, length, crc, magic_read, attrs = \
                self.HEADER_STRUCT_V0.unpack_from(self._buffer, pos)
            timestamp = None
        else:
            offset, length, crc, magic_read, attrs, timestamp = \
                self.HEADER_STRUCT_V1.unpack_from(self._buffer, pos)
        return offset, length, crc, magic_read, attrs, timestamp

    def _read_all_headers(self):
        pos = 0
        msgs = []
        buffer_len = len(self._buffer)
        while pos < buffer_len:
            header = self._read_header(pos)
            msgs.append((header, pos))
            pos += self.LOG_OVERHEAD + header[1]  # length
        return msgs

    def _read_key_value(self, pos):
        key_size = struct.unpack_from(">i", self._buffer, pos)[0]
        pos += self.KEY_LENGTH
        if key_size == -1:
            key = None
        else:
            key = self._buffer[pos:pos + key_size].tobytes()
            pos += key_size

        value_size = struct.unpack_from(">i", self._buffer, pos)[0]
        pos += self.VALUE_LENGTH
        if value_size == -1:
            value = None
        else:
            value = self._buffer[pos:pos + value_size].tobytes()
        return key, value

    def __iter__(self):
        if self._magic == 1:
            key_offset = self.KEY_OFFSET_V1
        else:
            key_offset = self.KEY_OFFSET_V0
        timestamp_type = self.timestamp_type

        if self.compression_type:
            # In case we will call iter again
            if not self._decompressed:
                self._buffer = memoryview(self._decompress(key_offset))
                self._decompressed = True

            # If relative offset is used, we need to decompress the entire
            # message first to compute the absolute offset.
            headers = self._read_all_headers()
            if self._magic > 0:
                msg_header, _ = headers[-1]
                absolute_base_offset = self._offset - msg_header[0]
            else:
                absolute_base_offset = -1

            for header, msg_pos in headers:
                offset, _, crc, _, attrs, timestamp = header
                # There should only ever be a single layer of compression
                assert not attrs & self.CODEC_MASK, (
                    'MessageSet at offset %d appears double-compressed. This '
                    'should not happen -- check your producers!' % (offset,))

                # When magic value is greater than 0, the timestamp
                # of a compressed message depends on the
                # typestamp type of the wrapper message:
                if timestamp_type == self.LOG_APPEND_TIME:
                    timestamp = self._timestamp

                if absolute_base_offset >= 0:
                    offset += absolute_base_offset

                key, value = self._read_key_value(msg_pos + key_offset)
                yield LegacyRecord(
                    offset, timestamp, timestamp_type,
                    key, value, crc)
        else:
            key, value = self._read_key_value(key_offset)
            yield LegacyRecord(
                self._offset, self._timestamp, timestamp_type,
                key, value, self._crc)


class LegacyRecord(ABCRecord):

    __slots__ = ("_offset", "_timestamp", "_timestamp_type", "_key", "_value",
                 "_crc")

    def __init__(self, offset, timestamp, timestamp_type, key, value, crc):
        self._offset = offset
        self._timestamp = timestamp
        self._timestamp_type = timestamp_type
        self._key = key
        self._value = value
        self._crc = crc

    @property
    def offset(self):
        return self._offset

    @property
    def timestamp(self):
        """ Epoch milliseconds
        """
        return self._timestamp

    @property
    def timestamp_type(self):
        """ CREATE_TIME(0) or APPEND_TIME(1)
        """
        return self._timestamp_type

    @property
    def key(self):
        """ Bytes key or None
        """
        return self._key

    @property
    def value(self):
        """ Bytes value or None
        """
        return self._value

    @property
    def headers(self):
        return []

    @property
    def checksum(self):
        return self._crc

    def __repr__(self):
        return (
            "LegacyRecord(offset={!r}, timestamp={!r}, timestamp_type={!r},"
            " key={!r}, value={!r}, crc={!r})".format(
                self._offset, self._timestamp, self._timestamp_type,
                self._key, self._value, self._crc)
        )


class LegacyRecordBatchBuilder(ABCRecordBatchBuilder, LegacyRecordBase):

    __slots__ = ("_magic", "_compression_type", "_batch_size", "_buffer")

    def __init__(self, magic, compression_type, batch_size):
        self._magic = magic
        self._compression_type = compression_type
        self._batch_size = batch_size
        self._buffer = bytearray()

    def append(self, offset, timestamp, key, value, headers=None):
        """ Append message to batch.
        """
        assert not headers, "Headers not supported in v0/v1"
        # Check types
        if type(offset) != int:
            raise TypeError(offset)
        if self._magic == 0:
            timestamp = self.NO_TIMESTAMP
        elif timestamp is None:
            timestamp = int(time.time() * 1000)
        elif type(timestamp) != int:
            raise TypeError(
                "`timestamp` should be int, but {} provided".format(
                    type(timestamp)))
        if not (key is None or
                isinstance(key, (bytes, bytearray, memoryview))):
            raise TypeError(
                "Not supported type for key: {}".format(type(key)))
        if not (value is None or
                isinstance(value, (bytes, bytearray, memoryview))):
            raise TypeError(
                "Not supported type for value: {}".format(type(value)))

        # Check if we have room for another message
        pos = len(self._buffer)
        size = self.size_in_bytes(offset, timestamp, key, value)
        # We always allow at least one record to be appended
        if offset != 0 and pos + size >= self._batch_size:
            return None

        # Allocate proper buffer length
        self._buffer.extend(bytearray(size))

        # Encode message
        crc = self._encode_msg(pos, offset, timestamp, key, value)

        return LegacyRecordMetadata(offset, crc, size, timestamp)

    def _encode_msg(self, start_pos, offset, timestamp, key, value,
                    attributes=0):
        """ Encode msg data into the `msg_buffer`, which should be allocated
            to at least the size of this message.
        """
        magic = self._magic
        buf = self._buffer
        pos = start_pos

        # Write key and value
        pos += self.KEY_OFFSET_V0 if magic == 0 else self.KEY_OFFSET_V1

        if key is None:
            struct.pack_into(">i", buf, pos, -1)
            pos += self.KEY_LENGTH
        else:
            key_size = len(key)
            struct.pack_into(">i", buf, pos, key_size)
            pos += self.KEY_LENGTH
            buf[pos: pos + key_size] = key
            pos += key_size

        if value is None:
            struct.pack_into(">i", buf, pos, -1)
            pos += self.VALUE_LENGTH
        else:
            value_size = len(value)
            struct.pack_into(">i", buf, pos, value_size)
            pos += self.VALUE_LENGTH
            buf[pos: pos + value_size] = value
            pos += value_size
        length = (pos - start_pos) - self.LOG_OVERHEAD

        # Write msg header. Note, that Crc will be updated later
        if magic == 0:
            self.HEADER_STRUCT_V0.pack_into(
                buf, start_pos,
                offset, length, 0, magic, attributes)
        else:
            self.HEADER_STRUCT_V1.pack_into(
                buf, start_pos,
                offset, length, 0, magic, attributes, timestamp)

        # Calculate CRC for msg
        crc_data = memoryview(buf)[start_pos + self.MAGIC_OFFSET:]
        crc = calc_crc32(crc_data)
        struct.pack_into(">I", buf, start_pos + self.CRC_OFFSET, crc)
        return crc

    def _maybe_compress(self):
        if self._compression_type:
            self._assert_has_codec(self._compression_type)
            data = bytes(self._buffer)
            if self._compression_type == self.CODEC_GZIP:
                compressed = gzip_encode(data)
            elif self._compression_type == self.CODEC_SNAPPY:
                compressed = snappy_encode(data)
            elif self._compression_type == self.CODEC_LZ4:
                if self._magic == 0:
                    compressed = lz4_encode_old_kafka(data)
                else:
                    compressed = lz4_encode(data)
            size = self.size_in_bytes(
                0, timestamp=0, key=None, value=compressed)
            # We will try to reuse the same buffer if we have enough space
            if size > len(self._buffer):
                self._buffer = bytearray(size)
            else:
                del self._buffer[size:]
            self._encode_msg(
                start_pos=0,
                offset=0, timestamp=0, key=None, value=compressed,
                attributes=self._compression_type)
            return True
        return False

    def build(self):
        """Compress batch to be ready for send"""
        self._maybe_compress()
        return self._buffer

    def size(self):
        """ Return current size of data written to buffer
        """
        return len(self._buffer)

    # Size calculations. Just copied Java's implementation

    def size_in_bytes(self, offset, timestamp, key, value, headers=None):
        """ Actual size of message to add
        """
        assert not headers, "Headers not supported in v0/v1"
        magic = self._magic
        return self.LOG_OVERHEAD + self.record_size(magic, key, value)

    @classmethod
    def record_size(cls, magic, key, value):
        message_size = cls.record_overhead(magic)
        if key is not None:
            message_size += len(key)
        if value is not None:
            message_size += len(value)
        return message_size

    @classmethod
    def record_overhead(cls, magic):
        assert magic in [0, 1], "Not supported magic"
        if magic == 0:
            return cls.RECORD_OVERHEAD_V0
        else:
            return cls.RECORD_OVERHEAD_V1

    @classmethod
    def estimate_size_in_bytes(cls, magic, compression_type, key, value):
        """ Upper bound estimate of record size.
        """
        assert magic in [0, 1], "Not supported magic"
        # In case of compression we may need another overhead for inner msg
        if compression_type:
            return (
                cls.LOG_OVERHEAD + cls.record_overhead(magic) +
                cls.record_size(magic, key, value)
            )
        return cls.LOG_OVERHEAD + cls.record_size(magic, key, value)


class LegacyRecordMetadata(object):

    __slots__ = ("_crc", "_size", "_timestamp", "_offset")

    def __init__(self, offset, crc, size, timestamp):
        self._offset = offset
        self._crc = crc
        self._size = size
        self._timestamp = timestamp

    @property
    def offset(self):
        return self._offset

    @property
    def crc(self):
        return self._crc

    @property
    def size(self):
        return self._size

    @property
    def timestamp(self):
        return self._timestamp

    def __repr__(self):
        return (
            "LegacyRecordMetadata(offset={!r}, crc={!r}, size={!r},"
            " timestamp={!r})".format(
                self._offset, self._crc, self._size, self._timestamp)
        )
