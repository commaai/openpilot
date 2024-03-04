from __future__ import absolute_import

import collections
import threading

from kafka import errors as Errors
from kafka.future import Future


class FutureProduceResult(Future):
    def __init__(self, topic_partition):
        super(FutureProduceResult, self).__init__()
        self.topic_partition = topic_partition
        self._latch = threading.Event()

    def success(self, value):
        ret = super(FutureProduceResult, self).success(value)
        self._latch.set()
        return ret

    def failure(self, error):
        ret = super(FutureProduceResult, self).failure(error)
        self._latch.set()
        return ret

    def wait(self, timeout=None):
        # wait() on python2.6 returns None instead of the flag value
        return self._latch.wait(timeout) or self._latch.is_set()


class FutureRecordMetadata(Future):
    def __init__(self, produce_future, relative_offset, timestamp_ms, checksum, serialized_key_size, serialized_value_size, serialized_header_size):
        super(FutureRecordMetadata, self).__init__()
        self._produce_future = produce_future
        # packing args as a tuple is a minor speed optimization
        self.args = (relative_offset, timestamp_ms, checksum, serialized_key_size, serialized_value_size, serialized_header_size)
        produce_future.add_callback(self._produce_success)
        produce_future.add_errback(self.failure)

    def _produce_success(self, offset_and_timestamp):
        offset, produce_timestamp_ms, log_start_offset = offset_and_timestamp

        # Unpacking from args tuple is minor speed optimization
        (relative_offset, timestamp_ms, checksum,
         serialized_key_size, serialized_value_size, serialized_header_size) = self.args

        # None is when Broker does not support the API (<0.10) and
        # -1 is when the broker is configured for CREATE_TIME timestamps
        if produce_timestamp_ms is not None and produce_timestamp_ms != -1:
            timestamp_ms = produce_timestamp_ms
        if offset != -1 and relative_offset is not None:
            offset += relative_offset
        tp = self._produce_future.topic_partition
        metadata = RecordMetadata(tp[0], tp[1], tp, offset, timestamp_ms, log_start_offset,
                                  checksum, serialized_key_size,
                                  serialized_value_size, serialized_header_size)
        self.success(metadata)

    def get(self, timeout=None):
        if not self.is_done and not self._produce_future.wait(timeout):
            raise Errors.KafkaTimeoutError(
                "Timeout after waiting for %s secs." % (timeout,))
        assert self.is_done
        if self.failed():
            raise self.exception # pylint: disable-msg=raising-bad-type
        return self.value


RecordMetadata = collections.namedtuple(
    'RecordMetadata', ['topic', 'partition', 'topic_partition', 'offset', 'timestamp', 'log_start_offset',
                       'checksum', 'serialized_key_size', 'serialized_value_size', 'serialized_header_size'])
