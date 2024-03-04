# See:
# https://github.com/apache/kafka/blob/trunk/clients/src/main/java/org/\
#    apache/kafka/common/record/DefaultRecordBatch.java
# https://github.com/apache/kafka/blob/trunk/clients/src/main/java/org/\
#    apache/kafka/common/record/DefaultRecord.java

# RecordBatch and Record implementation for magic 2 and above.
# The schema is given below:

# RecordBatch =>
#  BaseOffset => Int64
#  Length => Int32
#  PartitionLeaderEpoch => Int32
#  Magic => Int8
#  CRC => Uint32
#  Attributes => Int16
#  LastOffsetDelta => Int32 // also serves as LastSequenceDelta
#  FirstTimestamp => Int64
#  MaxTimestamp => Int64
#  ProducerId => Int64
#  ProducerEpoch => Int16
#  BaseSequence => Int32
#  Records => [Record]

# Record =>
#   Length => Varint
#   Attributes => Int8
#   TimestampDelta => Varlong
#   OffsetDelta => Varint
#   Key => Bytes
#   Value => Bytes
#   Headers => [HeaderKey HeaderValue]
#     HeaderKey => String
#     HeaderValue => Bytes

# Note that when compression is enabled (see attributes below), the compressed
# record data is serialized directly following the count of the number of
# records. (ie Records => [Record], but without length bytes)

# The CRC covers the data from the attributes to the end of the batch (i.e. all
# the bytes that follow the CRC). It is located after the magic byte, which
# means that clients must parse the magic byte before deciding how to interpret
# the bytes between the batch length and the magic byte. The partition leader
# epoch field is not included in the CRC computation to avoid the need to
# recompute the CRC when this field is assigned for every batch that is
# received by the broker. The CRC-32C (Castagnoli) polynomial is used for the
# computation.

# The current RecordBatch attributes are given below:
#
# * Unused (6-15)
# * Control (5)
# * Transactional (4)
# * Timestamp Type (3)
# * Compression Type (0-2)

import struct
import time
from kafka.record.abc import ABCRecord, ABCRecordBatch, ABCRecordBatchBuilder
from kafka.record.util import (
    decode_varint, encode_varint, calc_crc32c, size_of_varint
)
from kafka.errors import CorruptRecordException, UnsupportedCodecError
from kafka.codec import (
    gzip_encode, snappy_encode, lz4_encode, zstd_encode,
    gzip_decode, snappy_decode, lz4_decode, zstd_decode
)
import kafka.codec as codecs


class DefaultRecordBase(object):

    __slots__ = ()

    HEADER_STRUCT = struct.Struct(
        ">q"  # BaseOffset => Int64
        "i"  # Length => Int32
        "i"  # PartitionLeaderEpoch => Int32
        "b"  # Magic => Int8
        "I"  # CRC => Uint32
        "h"  # Attributes => Int16
        "i"  # LastOffsetDelta => Int32 // also serves as LastSequenceDelta
        "q"  # FirstTimestamp => Int64
        "q"  # MaxTimestamp => Int64
        "q"  # ProducerId => Int64
        "h"  # ProducerEpoch => Int16
        "i"  # BaseSequence => Int32
        "i"  # Records count => Int32
    )
    # Byte offset in HEADER_STRUCT of attributes field. Used to calculate CRC
    ATTRIBUTES_OFFSET = struct.calcsize(">qiibI")
    CRC_OFFSET = struct.calcsize(">qiib")
    AFTER_LEN_OFFSET = struct.calcsize(">qi")

    CODEC_MASK = 0x07
    CODEC_NONE = 0x00
    CODEC_GZIP = 0x01
    CODEC_SNAPPY = 0x02
    CODEC_LZ4 = 0x03
    CODEC_ZSTD = 0x04
    TIMESTAMP_TYPE_MASK = 0x08
    TRANSACTIONAL_MASK = 0x10
    CONTROL_MASK = 0x20

    LOG_APPEND_TIME = 1
    CREATE_TIME = 0

    def _assert_has_codec(self, compression_type):
        if compression_type == self.CODEC_GZIP:
            checker, name = codecs.has_gzip, "gzip"
        elif compression_type == self.CODEC_SNAPPY:
            checker, name = codecs.has_snappy, "snappy"
        elif compression_type == self.CODEC_LZ4:
            checker, name = codecs.has_lz4, "lz4"
        elif compression_type == self.CODEC_ZSTD:
            checker, name = codecs.has_zstd, "zstd"
        if not checker():
            raise UnsupportedCodecError(
                "Libraries for {} compression codec not found".format(name))


class DefaultRecordBatch(DefaultRecordBase, ABCRecordBatch):

    __slots__ = ("_buffer", "_header_data", "_pos", "_num_records",
                 "_next_record_index", "_decompressed")

    def __init__(self, buffer):
        self._buffer = bytearray(buffer)
        self._header_data = self.HEADER_STRUCT.unpack_from(self._buffer)
        self._pos = self.HEADER_STRUCT.size
        self._num_records = self._header_data[12]
        self._next_record_index = 0
        self._decompressed = False

    @property
    def base_offset(self):
        return self._header_data[0]

    @property
    def magic(self):
        return self._header_data[3]

    @property
    def crc(self):
        return self._header_data[4]

    @property
    def attributes(self):
        return self._header_data[5]

    @property
    def last_offset_delta(self):
        return self._header_data[6]

    @property
    def compression_type(self):
        return self.attributes & self.CODEC_MASK

    @property
    def timestamp_type(self):
        return int(bool(self.attributes & self.TIMESTAMP_TYPE_MASK))

    @property
    def is_transactional(self):
        return bool(self.attributes & self.TRANSACTIONAL_MASK)

    @property
    def is_control_batch(self):
        return bool(self.attributes & self.CONTROL_MASK)

    @property
    def first_timestamp(self):
        return self._header_data[7]

    @property
    def max_timestamp(self):
        return self._header_data[8]

    def _maybe_uncompress(self):
        if not self._decompressed:
            compression_type = self.compression_type
            if compression_type != self.CODEC_NONE:
                self._assert_has_codec(compression_type)
                data = memoryview(self._buffer)[self._pos:]
                if compression_type == self.CODEC_GZIP:
                    uncompressed = gzip_decode(data)
                if compression_type == self.CODEC_SNAPPY:
                    uncompressed = snappy_decode(data.tobytes())
                if compression_type == self.CODEC_LZ4:
                    uncompressed = lz4_decode(data.tobytes())
                if compression_type == self.CODEC_ZSTD:
                    uncompressed = zstd_decode(data.tobytes())
                self._buffer = bytearray(uncompressed)
                self._pos = 0
        self._decompressed = True

    def _read_msg(
            self,
            decode_varint=decode_varint):
        # Record =>
        #   Length => Varint
        #   Attributes => Int8
        #   TimestampDelta => Varlong
        #   OffsetDelta => Varint
        #   Key => Bytes
        #   Value => Bytes
        #   Headers => [HeaderKey HeaderValue]
        #     HeaderKey => String
        #     HeaderValue => Bytes

        buffer = self._buffer
        pos = self._pos
        length, pos = decode_varint(buffer, pos)
        start_pos = pos
        _, pos = decode_varint(buffer, pos)  # attrs can be skipped for now

        ts_delta, pos = decode_varint(buffer, pos)
        if self.timestamp_type == self.LOG_APPEND_TIME:
            timestamp = self.max_timestamp
        else:
            timestamp = self.first_timestamp + ts_delta

        offset_delta, pos = decode_varint(buffer, pos)
        offset = self.base_offset + offset_delta

        key_len, pos = decode_varint(buffer, pos)
        if key_len >= 0:
            key = bytes(buffer[pos: pos + key_len])
            pos += key_len
        else:
            key = None

        value_len, pos = decode_varint(buffer, pos)
        if value_len >= 0:
            value = bytes(buffer[pos: pos + value_len])
            pos += value_len
        else:
            value = None

        header_count, pos = decode_varint(buffer, pos)
        if header_count < 0:
            raise CorruptRecordException("Found invalid number of record "
                                         "headers {}".format(header_count))
        headers = []
        while header_count:
            # Header key is of type String, that can't be None
            h_key_len, pos = decode_varint(buffer, pos)
            if h_key_len < 0:
                raise CorruptRecordException(
                    "Invalid negative header key size {}".format(h_key_len))
            h_key = buffer[pos: pos + h_key_len].decode("utf-8")
            pos += h_key_len

            # Value is of type NULLABLE_BYTES, so it can be None
            h_value_len, pos = decode_varint(buffer, pos)
            if h_value_len >= 0:
                h_value = bytes(buffer[pos: pos + h_value_len])
                pos += h_value_len
            else:
                h_value = None

            headers.append((h_key, h_value))
            header_count -= 1

        # validate whether we have read all header bytes in the current record
        if pos - start_pos != length:
            raise CorruptRecordException(
                "Invalid record size: expected to read {} bytes in record "
                "payload, but instead read {}".format(length, pos - start_pos))
        self._pos = pos

        return DefaultRecord(
            offset, timestamp, self.timestamp_type, key, value, headers)

    def __iter__(self):
        self._maybe_uncompress()
        return self

    def __next__(self):
        if self._next_record_index >= self._num_records:
            if self._pos != len(self._buffer):
                raise CorruptRecordException(
                    "{} unconsumed bytes after all records consumed".format(
                        len(self._buffer) - self._pos))
            raise StopIteration
        try:
            msg = self._read_msg()
        except (ValueError, IndexError) as err:
            raise CorruptRecordException(
                "Found invalid record structure: {!r}".format(err))
        else:
            self._next_record_index += 1
        return msg

    next = __next__

    def validate_crc(self):
        assert self._decompressed is False, \
            "Validate should be called before iteration"

        crc = self.crc
        data_view = memoryview(self._buffer)[self.ATTRIBUTES_OFFSET:]
        verify_crc = calc_crc32c(data_view.tobytes())
        return crc == verify_crc


class DefaultRecord(ABCRecord):

    __slots__ = ("_offset", "_timestamp", "_timestamp_type", "_key", "_value",
                 "_headers")

    def __init__(self, offset, timestamp, timestamp_type, key, value, headers):
        self._offset = offset
        self._timestamp = timestamp
        self._timestamp_type = timestamp_type
        self._key = key
        self._value = value
        self._headers = headers

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
        return self._headers

    @property
    def checksum(self):
        return None

    def __repr__(self):
        return (
            "DefaultRecord(offset={!r}, timestamp={!r}, timestamp_type={!r},"
            " key={!r}, value={!r}, headers={!r})".format(
                self._offset, self._timestamp, self._timestamp_type,
                self._key, self._value, self._headers)
        )


class DefaultRecordBatchBuilder(DefaultRecordBase, ABCRecordBatchBuilder):

    # excluding key, value and headers:
    # 5 bytes length + 10 bytes timestamp + 5 bytes offset + 1 byte attributes
    MAX_RECORD_OVERHEAD = 21

    __slots__ = ("_magic", "_compression_type", "_batch_size", "_is_transactional",
                 "_producer_id", "_producer_epoch", "_base_sequence",
                 "_first_timestamp", "_max_timestamp", "_last_offset", "_num_records",
                 "_buffer")

    def __init__(
            self, magic, compression_type, is_transactional,
            producer_id, producer_epoch, base_sequence, batch_size):
        assert magic >= 2
        self._magic = magic
        self._compression_type = compression_type & self.CODEC_MASK
        self._batch_size = batch_size
        self._is_transactional = bool(is_transactional)
        # KIP-98 fields for EOS
        self._producer_id = producer_id
        self._producer_epoch = producer_epoch
        self._base_sequence = base_sequence

        self._first_timestamp = None
        self._max_timestamp = None
        self._last_offset = 0
        self._num_records = 0

        self._buffer = bytearray(self.HEADER_STRUCT.size)

    def _get_attributes(self, include_compression_type=True):
        attrs = 0
        if include_compression_type:
            attrs |= self._compression_type
        # Timestamp Type is set by Broker
        if self._is_transactional:
            attrs |= self.TRANSACTIONAL_MASK
        # Control batches are only created by Broker
        return attrs

    def append(self, offset, timestamp, key, value, headers,
               # Cache for LOAD_FAST opcodes
               encode_varint=encode_varint, size_of_varint=size_of_varint,
               get_type=type, type_int=int, time_time=time.time,
               byte_like=(bytes, bytearray, memoryview),
               bytearray_type=bytearray, len_func=len, zero_len_varint=1
               ):
        """ Write message to messageset buffer with MsgVersion 2
        """
        # Check types
        if get_type(offset) != type_int:
            raise TypeError(offset)
        if timestamp is None:
            timestamp = type_int(time_time() * 1000)
        elif get_type(timestamp) != type_int:
            raise TypeError(timestamp)
        if not (key is None or get_type(key) in byte_like):
            raise TypeError(
                "Not supported type for key: {}".format(type(key)))
        if not (value is None or get_type(value) in byte_like):
            raise TypeError(
                "Not supported type for value: {}".format(type(value)))

        # We will always add the first message, so those will be set
        if self._first_timestamp is None:
            self._first_timestamp = timestamp
            self._max_timestamp = timestamp
            timestamp_delta = 0
            first_message = 1
        else:
            timestamp_delta = timestamp - self._first_timestamp
            first_message = 0

        # We can't write record right away to out buffer, we need to
        # precompute the length as first value...
        message_buffer = bytearray_type(b"\x00")  # Attributes
        write_byte = message_buffer.append
        write = message_buffer.extend

        encode_varint(timestamp_delta, write_byte)
        # Base offset is always 0 on Produce
        encode_varint(offset, write_byte)

        if key is not None:
            encode_varint(len_func(key), write_byte)
            write(key)
        else:
            write_byte(zero_len_varint)

        if value is not None:
            encode_varint(len_func(value), write_byte)
            write(value)
        else:
            write_byte(zero_len_varint)

        encode_varint(len_func(headers), write_byte)

        for h_key, h_value in headers:
            h_key = h_key.encode("utf-8")
            encode_varint(len_func(h_key), write_byte)
            write(h_key)
            if h_value is not None:
                encode_varint(len_func(h_value), write_byte)
                write(h_value)
            else:
                write_byte(zero_len_varint)

        message_len = len_func(message_buffer)
        main_buffer = self._buffer

        required_size = message_len + size_of_varint(message_len)
        # Check if we can write this message
        if (required_size + len_func(main_buffer) > self._batch_size and
                not first_message):
            return None

        # Those should be updated after the length check
        if self._max_timestamp < timestamp:
            self._max_timestamp = timestamp
        self._num_records += 1
        self._last_offset = offset

        encode_varint(message_len, main_buffer.append)
        main_buffer.extend(message_buffer)

        return DefaultRecordMetadata(offset, required_size, timestamp)

    def write_header(self, use_compression_type=True):
        batch_len = len(self._buffer)
        self.HEADER_STRUCT.pack_into(
            self._buffer, 0,
            0,  # BaseOffset, set by broker
            batch_len - self.AFTER_LEN_OFFSET,  # Size from here to end
            0,  # PartitionLeaderEpoch, set by broker
            self._magic,
            0,  # CRC will be set below, as we need a filled buffer for it
            self._get_attributes(use_compression_type),
            self._last_offset,
            self._first_timestamp,
            self._max_timestamp,
            self._producer_id,
            self._producer_epoch,
            self._base_sequence,
            self._num_records
        )
        crc = calc_crc32c(self._buffer[self.ATTRIBUTES_OFFSET:])
        struct.pack_into(">I", self._buffer, self.CRC_OFFSET, crc)

    def _maybe_compress(self):
        if self._compression_type != self.CODEC_NONE:
            self._assert_has_codec(self._compression_type)
            header_size = self.HEADER_STRUCT.size
            data = bytes(self._buffer[header_size:])
            if self._compression_type == self.CODEC_GZIP:
                compressed = gzip_encode(data)
            elif self._compression_type == self.CODEC_SNAPPY:
                compressed = snappy_encode(data)
            elif self._compression_type == self.CODEC_LZ4:
                compressed = lz4_encode(data)
            elif self._compression_type == self.CODEC_ZSTD:
                compressed = zstd_encode(data)
            compressed_size = len(compressed)
            if len(data) <= compressed_size:
                # We did not get any benefit from compression, lets send
                # uncompressed
                return False
            else:
                # Trim bytearray to the required size
                needed_size = header_size + compressed_size
                del self._buffer[needed_size:]
                self._buffer[header_size:needed_size] = compressed
                return True
        return False

    def build(self):
        send_compressed = self._maybe_compress()
        self.write_header(send_compressed)
        return self._buffer

    def size(self):
        """ Return current size of data written to buffer
        """
        return len(self._buffer)

    def size_in_bytes(self, offset, timestamp, key, value, headers):
        if self._first_timestamp is not None:
            timestamp_delta = timestamp - self._first_timestamp
        else:
            timestamp_delta = 0
        size_of_body = (
            1 +  # Attrs
            size_of_varint(offset) +
            size_of_varint(timestamp_delta) +
            self.size_of(key, value, headers)
        )
        return size_of_body + size_of_varint(size_of_body)

    @classmethod
    def size_of(cls, key, value, headers):
        size = 0
        # Key size
        if key is None:
            size += 1
        else:
            key_len = len(key)
            size += size_of_varint(key_len) + key_len
        # Value size
        if value is None:
            size += 1
        else:
            value_len = len(value)
            size += size_of_varint(value_len) + value_len
        # Header size
        size += size_of_varint(len(headers))
        for h_key, h_value in headers:
            h_key_len = len(h_key.encode("utf-8"))
            size += size_of_varint(h_key_len) + h_key_len

            if h_value is None:
                size += 1
            else:
                h_value_len = len(h_value)
                size += size_of_varint(h_value_len) + h_value_len
        return size

    @classmethod
    def estimate_size_in_bytes(cls, key, value, headers):
        """ Get the upper bound estimate on the size of record
        """
        return (
            cls.HEADER_STRUCT.size + cls.MAX_RECORD_OVERHEAD +
            cls.size_of(key, value, headers)
        )


class DefaultRecordMetadata(object):

    __slots__ = ("_size", "_timestamp", "_offset")

    def __init__(self, offset, size, timestamp):
        self._offset = offset
        self._size = size
        self._timestamp = timestamp

    @property
    def offset(self):
        return self._offset

    @property
    def crc(self):
        return None

    @property
    def size(self):
        return self._size

    @property
    def timestamp(self):
        return self._timestamp

    def __repr__(self):
        return (
            "DefaultRecordMetadata(offset={!r}, size={!r}, timestamp={!r})"
            .format(self._offset, self._size, self._timestamp)
        )
