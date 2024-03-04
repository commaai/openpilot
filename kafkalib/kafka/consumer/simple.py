from __future__ import absolute_import

try:
    from itertools import zip_longest as izip_longest, repeat  # pylint: disable=E0611
except ImportError:
    from itertools import izip_longest as izip_longest, repeat  # pylint: disable=E0611
import logging
import sys
import time
import warnings

from kafka.vendor import six
from kafka.vendor.six.moves import queue # pylint: disable=import-error

from .base import (
    Consumer,
    FETCH_DEFAULT_BLOCK_TIMEOUT,
    AUTO_COMMIT_MSG_COUNT,
    AUTO_COMMIT_INTERVAL,
    FETCH_MIN_BYTES,
    FETCH_BUFFER_SIZE_BYTES,
    MAX_FETCH_BUFFER_SIZE_BYTES,
    FETCH_MAX_WAIT_TIME,
    ITER_TIMEOUT_SECONDS,
    NO_MESSAGES_WAIT_TIME_SECONDS
)
from ..common import (
    FetchRequestPayload, KafkaError, OffsetRequestPayload,
    ConsumerFetchSizeTooSmall,
    UnknownTopicOrPartitionError, NotLeaderForPartitionError,
    OffsetOutOfRangeError, FailedPayloadsError, check_error
)
from kafka.protocol.message import PartialMessage


log = logging.getLogger(__name__)


class FetchContext(object):
    """
    Class for managing the state of a consumer during fetch
    """
    def __init__(self, consumer, block, timeout):
        warnings.warn('deprecated - this class will be removed in a future'
                      ' release', DeprecationWarning)
        self.consumer = consumer
        self.block = block

        if block:
            if not timeout:
                timeout = FETCH_DEFAULT_BLOCK_TIMEOUT
            self.timeout = timeout * 1000

    def __enter__(self):
        """Set fetch values based on blocking status"""
        self.orig_fetch_max_wait_time = self.consumer.fetch_max_wait_time
        self.orig_fetch_min_bytes = self.consumer.fetch_min_bytes
        if self.block:
            self.consumer.fetch_max_wait_time = self.timeout
            self.consumer.fetch_min_bytes = 1
        else:
            self.consumer.fetch_min_bytes = 0

    def __exit__(self, type, value, traceback):
        """Reset values"""
        self.consumer.fetch_max_wait_time = self.orig_fetch_max_wait_time
        self.consumer.fetch_min_bytes = self.orig_fetch_min_bytes


class SimpleConsumer(Consumer):
    """
    A simple consumer implementation that consumes all/specified partitions
    for a topic

    Arguments:
        client: a connected SimpleClient
        group: a name for this consumer, used for offset storage and must be unique
            If you are connecting to a server that does not support offset
            commit/fetch (any prior to 0.8.1.1), then you *must* set this to None
        topic: the topic to consume

    Keyword Arguments:
        partitions: An optional list of partitions to consume the data from

        auto_commit: default True. Whether or not to auto commit the offsets

        auto_commit_every_n: default 100. How many messages to consume
             before a commit

        auto_commit_every_t: default 5000. How much time (in milliseconds) to
             wait before commit
        fetch_size_bytes: number of bytes to request in a FetchRequest

        buffer_size: default 4K. Initial number of bytes to tell kafka we
             have available. This will double as needed.

        max_buffer_size: default 16K. Max number of bytes to tell kafka we have
             available. None means no limit.

        iter_timeout: default None. How much time (in seconds) to wait for a
             message in the iterator before exiting. None means no
             timeout, so it will wait forever.

        auto_offset_reset: default largest. Reset partition offsets upon
             OffsetOutOfRangeError. Valid values are largest and smallest.
             Otherwise, do not reset the offsets and raise OffsetOutOfRangeError.

    Auto commit details:
    If both auto_commit_every_n and auto_commit_every_t are set, they will
    reset one another when one is triggered. These triggers simply call the
    commit method on this class. A manual call to commit will also reset
    these triggers
    """
    def __init__(self, client, group, topic, auto_commit=True, partitions=None,
                 auto_commit_every_n=AUTO_COMMIT_MSG_COUNT,
                 auto_commit_every_t=AUTO_COMMIT_INTERVAL,
                 fetch_size_bytes=FETCH_MIN_BYTES,
                 buffer_size=FETCH_BUFFER_SIZE_BYTES,
                 max_buffer_size=MAX_FETCH_BUFFER_SIZE_BYTES,
                 iter_timeout=None,
                 auto_offset_reset='largest'):
        warnings.warn('deprecated - this class will be removed in a future'
                      ' release. Use KafkaConsumer instead.',
                      DeprecationWarning)
        super(SimpleConsumer, self).__init__(
            client, group, topic,
            partitions=partitions,
            auto_commit=auto_commit,
            auto_commit_every_n=auto_commit_every_n,
            auto_commit_every_t=auto_commit_every_t)

        if max_buffer_size is not None and buffer_size > max_buffer_size:
            raise ValueError('buffer_size (%d) is greater than '
                             'max_buffer_size (%d)' %
                             (buffer_size, max_buffer_size))
        self.buffer_size = buffer_size
        self.max_buffer_size = max_buffer_size
        self.fetch_max_wait_time = FETCH_MAX_WAIT_TIME
        self.fetch_min_bytes = fetch_size_bytes
        self.fetch_offsets = self.offsets.copy()
        self.iter_timeout = iter_timeout
        self.auto_offset_reset = auto_offset_reset
        self.queue = queue.Queue()

    def __repr__(self):
        return '<SimpleConsumer group=%s, topic=%s, partitions=%s>' % \
            (self.group, self.topic, str(self.offsets.keys()))

    def reset_partition_offset(self, partition):
        """Update offsets using auto_offset_reset policy (smallest|largest)

        Arguments:
            partition (int): the partition for which offsets should be updated

        Returns: Updated offset on success, None on failure
        """
        LATEST = -1
        EARLIEST = -2
        if self.auto_offset_reset == 'largest':
            reqs = [OffsetRequestPayload(self.topic, partition, LATEST, 1)]
        elif self.auto_offset_reset == 'smallest':
            reqs = [OffsetRequestPayload(self.topic, partition, EARLIEST, 1)]
        else:
            # Let's raise an reasonable exception type if user calls
            # outside of an exception context
            if sys.exc_info() == (None, None, None):
                raise OffsetOutOfRangeError('Cannot reset partition offsets without a '
                                            'valid auto_offset_reset setting '
                                            '(largest|smallest)')
            # Otherwise we should re-raise the upstream exception
            # b/c it typically includes additional data about
            # the request that triggered it, and we do not want to drop that
            raise # pylint: disable=E0704

        # send_offset_request
        log.info('Resetting topic-partition offset to %s for %s:%d',
                 self.auto_offset_reset, self.topic, partition)
        try:
            (resp, ) = self.client.send_offset_request(reqs)
        except KafkaError as e:
            log.error('%s sending offset request for %s:%d',
                      e.__class__.__name__, self.topic, partition)
        else:
            self.offsets[partition] = resp.offsets[0]
            self.fetch_offsets[partition] = resp.offsets[0]
            return resp.offsets[0]

    def seek(self, offset, whence=None, partition=None):
        """
        Alter the current offset in the consumer, similar to fseek

        Arguments:
            offset: how much to modify the offset
            whence: where to modify it from, default is None

                * None is an absolute offset
                * 0    is relative to the earliest available offset (head)
                * 1    is relative to the current offset
                * 2    is relative to the latest known offset (tail)

            partition: modify which partition, default is None.
                If partition is None, would modify all partitions.
        """

        if whence is None: # set an absolute offset
            if partition is None:
                for tmp_partition in self.offsets:
                    self.offsets[tmp_partition] = offset
            else:
                self.offsets[partition] = offset
        elif whence == 1:  # relative to current position
            if partition is None:
                for tmp_partition, _offset in self.offsets.items():
                    self.offsets[tmp_partition] = _offset + offset
            else:
                self.offsets[partition] += offset
        elif whence in (0, 2):  # relative to beginning or end
            reqs = []
            deltas = {}
            if partition is None:
                # divide the request offset by number of partitions,
                # distribute the remained evenly
                (delta, rem) = divmod(offset, len(self.offsets))
                for tmp_partition, r in izip_longest(self.offsets.keys(),
                                                     repeat(1, rem),
                                                     fillvalue=0):
                    deltas[tmp_partition] = delta + r

                for tmp_partition in self.offsets.keys():
                    if whence == 0:
                        reqs.append(OffsetRequestPayload(self.topic, tmp_partition, -2, 1))
                    elif whence == 2:
                        reqs.append(OffsetRequestPayload(self.topic, tmp_partition, -1, 1))
                    else:
                        pass
            else:
                deltas[partition] = offset
                if whence == 0:
                    reqs.append(OffsetRequestPayload(self.topic, partition, -2, 1))
                elif whence == 2:
                    reqs.append(OffsetRequestPayload(self.topic, partition, -1, 1))
                else:
                    pass

            resps = self.client.send_offset_request(reqs)
            for resp in resps:
                self.offsets[resp.partition] = \
                    resp.offsets[0] + deltas[resp.partition]
        else:
            raise ValueError('Unexpected value for `whence`, %d' % whence)

        # Reset queue and fetch offsets since they are invalid
        self.fetch_offsets = self.offsets.copy()
        self.count_since_commit += 1
        if self.auto_commit:
            self.commit()

        self.queue = queue.Queue()

    def get_messages(self, count=1, block=True, timeout=0.1):
        """
        Fetch the specified number of messages

        Keyword Arguments:
            count: Indicates the maximum number of messages to be fetched
            block: If True, the API will block till all messages are fetched.
                If block is a positive integer the API will block until that
                many messages are fetched.
            timeout: When blocking is requested the function will block for
                the specified time (in seconds) until count messages is
                fetched. If None, it will block forever.
        """
        messages = []
        if timeout is not None:
            timeout += time.time()

        new_offsets = {}
        log.debug('getting %d messages', count)
        while len(messages) < count:
            block_time = timeout - time.time()
            log.debug('calling _get_message block=%s timeout=%s', block, block_time)
            block_next_call = block is True or block > len(messages)
            result = self._get_message(block_next_call, block_time,
                                       get_partition_info=True,
                                       update_offset=False)
            log.debug('got %s from _get_messages', result)
            if not result:
                if block_next_call and (timeout is None or time.time() <= timeout):
                    continue
                break

            partition, message = result
            _msg = (partition, message) if self.partition_info else message
            messages.append(_msg)
            new_offsets[partition] = message.offset + 1

        # Update and commit offsets if necessary
        self.offsets.update(new_offsets)
        self.count_since_commit += len(messages)
        self._auto_commit()
        log.debug('got %d messages: %s', len(messages), messages)
        return messages

    def get_message(self, block=True, timeout=0.1, get_partition_info=None):
        return self._get_message(block, timeout, get_partition_info)

    def _get_message(self, block=True, timeout=0.1, get_partition_info=None,
                     update_offset=True):
        """
        If no messages can be fetched, returns None.
        If get_partition_info is None, it defaults to self.partition_info
        If get_partition_info is True, returns (partition, message)
        If get_partition_info is False, returns message
        """
        start_at = time.time()
        while self.queue.empty():
            # We're out of messages, go grab some more.
            log.debug('internal queue empty, fetching more messages')
            with FetchContext(self, block, timeout):
                self._fetch()

            if not block or time.time() > (start_at + timeout):
                break

        try:
            partition, message = self.queue.get_nowait()

            if update_offset:
                # Update partition offset
                self.offsets[partition] = message.offset + 1

                # Count, check and commit messages if necessary
                self.count_since_commit += 1
                self._auto_commit()

            if get_partition_info is None:
                get_partition_info = self.partition_info
            if get_partition_info:
                return partition, message
            else:
                return message
        except queue.Empty:
            log.debug('internal queue empty after fetch - returning None')
            return None

    def __iter__(self):
        if self.iter_timeout is None:
            timeout = ITER_TIMEOUT_SECONDS
        else:
            timeout = self.iter_timeout

        while True:
            message = self.get_message(True, timeout)
            if message:
                yield message
            elif self.iter_timeout is None:
                # We did not receive any message yet but we don't have a
                # timeout, so give up the CPU for a while before trying again
                time.sleep(NO_MESSAGES_WAIT_TIME_SECONDS)
            else:
                # Timed out waiting for a message
                break

    def _fetch(self):
        # Create fetch request payloads for all the partitions
        partitions = dict((p, self.buffer_size)
                      for p in self.fetch_offsets.keys())
        while partitions:
            requests = []
            for partition, buffer_size in six.iteritems(partitions):
                requests.append(FetchRequestPayload(self.topic, partition,
                                                    self.fetch_offsets[partition],
                                                    buffer_size))
            # Send request
            responses = self.client.send_fetch_request(
                requests,
                max_wait_time=int(self.fetch_max_wait_time),
                min_bytes=self.fetch_min_bytes,
                fail_on_error=False
            )

            retry_partitions = {}
            for resp in responses:

                try:
                    check_error(resp)
                except UnknownTopicOrPartitionError:
                    log.error('UnknownTopicOrPartitionError for %s:%d',
                              resp.topic, resp.partition)
                    self.client.reset_topic_metadata(resp.topic)
                    raise
                except NotLeaderForPartitionError:
                    log.error('NotLeaderForPartitionError for %s:%d',
                              resp.topic, resp.partition)
                    self.client.reset_topic_metadata(resp.topic)
                    continue
                except OffsetOutOfRangeError:
                    log.warning('OffsetOutOfRangeError for %s:%d. '
                                'Resetting partition offset...',
                                resp.topic, resp.partition)
                    self.reset_partition_offset(resp.partition)
                    # Retry this partition
                    retry_partitions[resp.partition] = partitions[resp.partition]
                    continue
                except FailedPayloadsError as e:
                    log.warning('FailedPayloadsError for %s:%d',
                                e.payload.topic, e.payload.partition)
                    # Retry this partition
                    retry_partitions[e.payload.partition] = partitions[e.payload.partition]
                    continue

                partition = resp.partition
                buffer_size = partitions[partition]

                # Check for partial message
                if resp.messages and isinstance(resp.messages[-1].message, PartialMessage):

                    # If buffer is at max and all we got was a partial message
                    # raise ConsumerFetchSizeTooSmall
                    if (self.max_buffer_size is not None and
                        buffer_size == self.max_buffer_size and
                        len(resp.messages) == 1):

                        log.error('Max fetch size %d too small', self.max_buffer_size)
                        raise ConsumerFetchSizeTooSmall()

                    if self.max_buffer_size is None:
                        buffer_size *= 2
                    else:
                        buffer_size = min(buffer_size * 2, self.max_buffer_size)
                    log.warning('Fetch size too small, increase to %d (2x) '
                                'and retry', buffer_size)
                    retry_partitions[partition] = buffer_size
                    resp.messages.pop()

                for message in resp.messages:
                    if message.offset < self.fetch_offsets[partition]:
                        log.debug('Skipping message %s because its offset is less than the consumer offset',
                                  message)
                        continue
                    # Put the message in our queue
                    self.queue.put((partition, message))
                    self.fetch_offsets[partition] = message.offset + 1
            partitions = retry_partitions
