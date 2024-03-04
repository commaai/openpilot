from __future__ import absolute_import, division

import collections
import io
import threading
import time

from kafka.metrics.stats import Rate

import kafka.errors as Errors


class SimpleBufferPool(object):
    """A simple pool of BytesIO objects with a weak memory ceiling."""
    def __init__(self, memory, poolable_size, metrics=None, metric_group_prefix='producer-metrics'):
        """Create a new buffer pool.

        Arguments:
            memory (int): maximum memory that this buffer pool can allocate
            poolable_size (int): memory size per buffer to cache in the free
                list rather than deallocating
        """
        self._poolable_size = poolable_size
        self._lock = threading.RLock()

        buffers = int(memory / poolable_size) if poolable_size else 0
        self._free = collections.deque([io.BytesIO() for _ in range(buffers)])

        self._waiters = collections.deque()
        self.wait_time = None
        if metrics:
            self.wait_time = metrics.sensor('bufferpool-wait-time')
            self.wait_time.add(metrics.metric_name(
                'bufferpool-wait-ratio', metric_group_prefix,
                'The fraction of time an appender waits for space allocation.'),
                Rate())

    def allocate(self, size, max_time_to_block_ms):
        """
        Allocate a buffer of the given size. This method blocks if there is not
        enough memory and the buffer pool is configured with blocking mode.

        Arguments:
            size (int): The buffer size to allocate in bytes [ignored]
            max_time_to_block_ms (int): The maximum time in milliseconds to
                block for buffer memory to be available

        Returns:
            io.BytesIO
        """
        with self._lock:
            # check if we have a free buffer of the right size pooled
            if self._free:
                return self._free.popleft()

            elif self._poolable_size == 0:
                return io.BytesIO()

            else:
                # we are out of buffers and will have to block
                buf = None
                more_memory = threading.Condition(self._lock)
                self._waiters.append(more_memory)
                # loop over and over until we have a buffer or have reserved
                # enough memory to allocate one
                while buf is None:
                    start_wait = time.time()
                    more_memory.wait(max_time_to_block_ms / 1000.0)
                    end_wait = time.time()
                    if self.wait_time:
                        self.wait_time.record(end_wait - start_wait)

                    if self._free:
                        buf = self._free.popleft()
                    else:
                        self._waiters.remove(more_memory)
                        raise Errors.KafkaTimeoutError(
                            "Failed to allocate memory within the configured"
                            " max blocking time")

                # remove the condition for this thread to let the next thread
                # in line start getting memory
                removed = self._waiters.popleft()
                assert removed is more_memory, 'Wrong condition'

                # signal any additional waiters if there is more memory left
                # over for them
                if self._free and self._waiters:
                    self._waiters[0].notify()

                # unlock and return the buffer
                return buf

    def deallocate(self, buf):
        """
        Return buffers to the pool. If they are of the poolable size add them
        to the free list, otherwise just mark the memory as free.

        Arguments:
            buffer_ (io.BytesIO): The buffer to return
        """
        with self._lock:
            # BytesIO.truncate here makes the pool somewhat pointless
            # but we stick with the BufferPool API until migrating to
            # bytesarray / memoryview. The buffer we return must not
            # expose any prior data on read().
            buf.truncate(0)
            self._free.append(buf)
            if self._waiters:
                self._waiters[0].notify()

    def queued(self):
        """The number of threads blocked waiting on memory."""
        with self._lock:
            return len(self._waiters)
