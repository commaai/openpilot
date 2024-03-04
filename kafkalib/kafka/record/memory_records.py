# This class takes advantage of the fact that all formats v0, v1 and v2 of
# messages storage has the same byte offsets for Length and Magic fields.
# Lets look closely at what leading bytes all versions have:
#
# V0 and V1 (Offset is MessageSet part, other bytes are Message ones):
#  Offset => Int64
#  BytesLength => Int32
#  CRC => Int32
#  Magic => Int8
#  ...
#
# V2:
#  BaseOffset => Int64
#  Length => Int32
#  PartitionLeaderEpoch => Int32
#  Magic => Int8
#  ...
#
# So we can iterate over batches just by knowing offsets of Length. Magic is
# used to construct the correct class for Batch itself.
from __future__ import division

import struct

from kafka.errors import CorruptRecordException
from kafka.record.abc import ABCRecords
from kafka.record.legacy_records import LegacyRecordBatch, LegacyRecordBatchBuilder
from kafka.record.default_records import DefaultRecordBatch, DefaultRecordBatchBuilder


class MemoryRecords(ABCRecords):

    LENGTH_OFFSET = struct.calcsize(">q")
    LOG_OVERHEAD = struct.calcsize(">qi")
    MAGIC_OFFSET = struct.calcsize(">qii")

    # Minimum space requirements for Record V0
    MIN_SLICE = LOG_OVERHEAD + LegacyRecordBatch.RECORD_OVERHEAD_V0

    __slots__ = ("_buffer", "_pos", "_next_slice", "_remaining_bytes")

    def __init__(self, bytes_data):
        self._buffer = bytes_data
        self._pos = 0
        # We keep one slice ahead so `has_next` will return very fast
        self._next_slice = None
        self._remaining_bytes = None
        self._cache_next()

    def size_in_bytes(self):
        return len(self._buffer)

    def valid_bytes(self):
        # We need to read the whole buffer to get the valid_bytes.
        # NOTE: in Fetcher we do the call after iteration, so should be fast
        if self._remaining_bytes is None:
            next_slice = self._next_slice
            pos = self._pos
            while self._remaining_bytes is None:
                self._cache_next()
            # Reset previous iterator position
            self._next_slice = next_slice
            self._pos = pos
        return len(self._buffer) - self._remaining_bytes

    # NOTE: we cache offsets here as kwargs for a bit more speed, as cPython
    # will use LOAD_FAST opcode in this case
    def _cache_next(self, len_offset=LENGTH_OFFSET, log_overhead=LOG_OVERHEAD):
        buffer = self._buffer
        buffer_len = len(buffer)
        pos = self._pos
        remaining = buffer_len - pos
        if remaining < log_overhead:
            # Will be re-checked in Fetcher for remaining bytes.
            self._remaining_bytes = remaining
            self._next_slice = None
            return

        length, = struct.unpack_from(
            ">i", buffer, pos + len_offset)

        slice_end = pos + log_overhead + length
        if slice_end > buffer_len:
            # Will be re-checked in Fetcher for remaining bytes
            self._remaining_bytes = remaining
            self._next_slice = None
            return

        self._next_slice = memoryview(buffer)[pos: slice_end]
        self._pos = slice_end

    def has_next(self):
        return self._next_slice is not None

    # NOTE: same cache for LOAD_FAST as above
    def next_batch(self, _min_slice=MIN_SLICE,
                   _magic_offset=MAGIC_OFFSET):
        next_slice = self._next_slice
        if next_slice is None:
            return None
        if len(next_slice) < _min_slice:
            raise CorruptRecordException(
                "Record size is less than the minimum record overhead "
                "({})".format(_min_slice - self.LOG_OVERHEAD))
        self._cache_next()
        magic, = struct.unpack_from(">b", next_slice, _magic_offset)
        if magic <= 1:
            return LegacyRecordBatch(next_slice, magic)
        else:
            return DefaultRecordBatch(next_slice)


class MemoryRecordsBuilder(object):

    __slots__ = ("_builder", "_batch_size", "_buffer", "_next_offset", "_closed",
                 "_bytes_written")

    def __init__(self, magic, compression_type, batch_size):
        assert magic in [0, 1, 2], "Not supported magic"
        assert compression_type in [0, 1, 2, 3, 4], "Not valid compression type"
        if magic >= 2:
            self._builder = DefaultRecordBatchBuilder(
                magic=magic, compression_type=compression_type,
                is_transactional=False, producer_id=-1, producer_epoch=-1,
                base_sequence=-1, batch_size=batch_size)
        else:
            self._builder = LegacyRecordBatchBuilder(
                magic=magic, compression_type=compression_type,
                batch_size=batch_size)
        self._batch_size = batch_size
        self._buffer = None

        self._next_offset = 0
        self._closed = False
        self._bytes_written = 0

    def append(self, timestamp, key, value, headers=[]):
        """ Append a message to the buffer.

        Returns: RecordMetadata or None if unable to append
        """
        if self._closed:
            return None

        offset = self._next_offset
        metadata = self._builder.append(offset, timestamp, key, value, headers)
        # Return of None means there's no space to add a new message
        if metadata is None:
            return None

        self._next_offset += 1
        return metadata

    def close(self):
        # This method may be called multiple times on the same batch
        # i.e., on retries
        # we need to make sure we only close it out once
        # otherwise compressed messages may be double-compressed
        # see Issue 718
        if not self._closed:
            self._bytes_written = self._builder.size()
            self._buffer = bytes(self._builder.build())
            self._builder = None
        self._closed = True

    def size_in_bytes(self):
        if not self._closed:
            return self._builder.size()
        else:
            return len(self._buffer)

    def compression_rate(self):
        assert self._closed
        return self.size_in_bytes() / self._bytes_written

    def is_full(self):
        if self._closed:
            return True
        else:
            return self._builder.size() >= self._batch_size

    def next_offset(self):
        return self._next_offset

    def buffer(self):
        assert self._closed
        return self._buffer
