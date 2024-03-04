from __future__ import absolute_import

from collections import namedtuple
import logging
from multiprocessing import Process, Manager as MPManager
import time
import warnings

from kafka.vendor.six.moves import queue # pylint: disable=import-error

from ..common import KafkaError
from .base import (
    Consumer,
    AUTO_COMMIT_MSG_COUNT, AUTO_COMMIT_INTERVAL,
    NO_MESSAGES_WAIT_TIME_SECONDS,
    FULL_QUEUE_WAIT_TIME_SECONDS,
    MAX_BACKOFF_SECONDS,
)
from .simple import SimpleConsumer


log = logging.getLogger(__name__)

Events = namedtuple("Events", ["start", "pause", "exit"])


def _mp_consume(client, group, topic, message_queue, size, events, **consumer_options):
    """
    A child process worker which consumes messages based on the
    notifications given by the controller process

    NOTE: Ideally, this should have been a method inside the Consumer
    class. However, multiprocessing module has issues in windows. The
    functionality breaks unless this function is kept outside of a class
    """

    # Initial interval for retries in seconds.
    interval = 1
    while not events.exit.is_set():
        try:
            # Make the child processes open separate socket connections
            client.reinit()

            # We will start consumers without auto-commit. Auto-commit will be
            # done by the master controller process.
            consumer = SimpleConsumer(client, group, topic,
                                      auto_commit=False,
                                      auto_commit_every_n=None,
                                      auto_commit_every_t=None,
                                      **consumer_options)

            # Ensure that the consumer provides the partition information
            consumer.provide_partition_info()

            while True:
                # Wait till the controller indicates us to start consumption
                events.start.wait()

                # If we are asked to quit, do so
                if events.exit.is_set():
                    break

                # Consume messages and add them to the queue. If the controller
                # indicates a specific number of messages, follow that advice
                count = 0

                message = consumer.get_message()
                if message:
                    while True:
                        try:
                            message_queue.put(message, timeout=FULL_QUEUE_WAIT_TIME_SECONDS)
                            break
                        except queue.Full:
                            if events.exit.is_set(): break

                    count += 1

                    # We have reached the required size. The controller might have
                    # more than what he needs. Wait for a while.
                    # Without this logic, it is possible that we run into a big
                    # loop consuming all available messages before the controller
                    # can reset the 'start' event
                    if count == size.value:
                        events.pause.wait()

                else:
                    # In case we did not receive any message, give up the CPU for
                    # a while before we try again
                    time.sleep(NO_MESSAGES_WAIT_TIME_SECONDS)

            consumer.stop()

        except KafkaError as e:
            # Retry with exponential backoff
            log.error("Problem communicating with Kafka (%s), retrying in %d seconds..." % (e, interval))
            time.sleep(interval)
            interval = interval*2 if interval*2 < MAX_BACKOFF_SECONDS else MAX_BACKOFF_SECONDS


class MultiProcessConsumer(Consumer):
    """
    A consumer implementation that consumes partitions for a topic in
    parallel using multiple processes

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
        num_procs: Number of processes to start for consuming messages.
            The available partitions will be divided among these processes
        partitions_per_proc: Number of partitions to be allocated per process
            (overrides num_procs)

    Auto commit details:
    If both auto_commit_every_n and auto_commit_every_t are set, they will
    reset one another when one is triggered. These triggers simply call the
    commit method on this class. A manual call to commit will also reset
    these triggers
    """
    def __init__(self, client, group, topic,
                 partitions=None,
                 auto_commit=True,
                 auto_commit_every_n=AUTO_COMMIT_MSG_COUNT,
                 auto_commit_every_t=AUTO_COMMIT_INTERVAL,
                 num_procs=1,
                 partitions_per_proc=0,
                 **simple_consumer_options):

        warnings.warn('This class has been deprecated and will be removed in a'
                      ' future release. Use KafkaConsumer instead',
                      DeprecationWarning)

        # Initiate the base consumer class
        super(MultiProcessConsumer, self).__init__(
            client, group, topic,
            partitions=partitions,
            auto_commit=auto_commit,
            auto_commit_every_n=auto_commit_every_n,
            auto_commit_every_t=auto_commit_every_t)

        # Variables for managing and controlling the data flow from
        # consumer child process to master
        manager = MPManager()
        self.queue = manager.Queue(1024)  # Child consumers dump messages into this
        self.events = Events(
            start = manager.Event(),        # Indicates the consumers to start fetch
            exit  = manager.Event(),        # Requests the consumers to shutdown
            pause = manager.Event())        # Requests the consumers to pause fetch
        self.size = manager.Value('i', 0)   # Indicator of number of messages to fetch

        # dict.keys() returns a view in py3 + it's not a thread-safe operation
        # http://blog.labix.org/2008/06/27/watch-out-for-listdictkeys-in-python-3
        # It's safer to copy dict as it only runs during the init.
        partitions = list(self.offsets.copy().keys())

        # By default, start one consumer process for all partitions
        # The logic below ensures that
        # * we do not cross the num_procs limit
        # * we have an even distribution of partitions among processes

        if partitions_per_proc:
            num_procs = len(partitions) / partitions_per_proc
            if num_procs * partitions_per_proc < len(partitions):
                num_procs += 1

        # The final set of chunks
        chunks = [partitions[proc::num_procs] for proc in range(num_procs)]

        self.procs = []
        for chunk in chunks:
            options = {'partitions': list(chunk)}
            if simple_consumer_options:
                simple_consumer_options.pop('partitions', None)
                options.update(simple_consumer_options)

            args = (client.copy(), self.group, self.topic, self.queue,
                    self.size, self.events)
            proc = Process(target=_mp_consume, args=args, kwargs=options)
            proc.daemon = True
            proc.start()
            self.procs.append(proc)

    def __repr__(self):
        return '<MultiProcessConsumer group=%s, topic=%s, consumers=%d>' % \
            (self.group, self.topic, len(self.procs))

    def stop(self):
        # Set exit and start off all waiting consumers
        self.events.exit.set()
        self.events.pause.set()
        self.events.start.set()

        for proc in self.procs:
            proc.join()
            proc.terminate()

        super(MultiProcessConsumer, self).stop()

    def __iter__(self):
        """
        Iterator to consume the messages available on this consumer
        """
        # Trigger the consumer procs to start off.
        # We will iterate till there are no more messages available
        self.size.value = 0
        self.events.pause.set()

        while True:
            self.events.start.set()
            try:
                # We will block for a small while so that the consumers get
                # a chance to run and put some messages in the queue
                # TODO: This is a hack and will make the consumer block for
                # at least one second. Need to find a better way of doing this
                partition, message = self.queue.get(block=True, timeout=1)
            except queue.Empty:
                break

            # Count, check and commit messages if necessary
            self.offsets[partition] = message.offset + 1
            self.events.start.clear()
            self.count_since_commit += 1
            self._auto_commit()
            yield message

        self.events.start.clear()

    def get_messages(self, count=1, block=True, timeout=10):
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

        # Give a size hint to the consumers. Each consumer process will fetch
        # a maximum of "count" messages. This will fetch more messages than
        # necessary, but these will not be committed to kafka. Also, the extra
        # messages can be provided in subsequent runs
        self.size.value = count
        self.events.pause.clear()

        if timeout is not None:
            max_time = time.time() + timeout

        new_offsets = {}
        while count > 0 and (timeout is None or timeout > 0):
            # Trigger consumption only if the queue is empty
            # By doing this, we will ensure that consumers do not
            # go into overdrive and keep consuming thousands of
            # messages when the user might need only a few
            if self.queue.empty():
                self.events.start.set()

            block_next_call = block is True or block > len(messages)
            try:
                partition, message = self.queue.get(block_next_call,
                                                    timeout)
            except queue.Empty:
                break

            _msg = (partition, message) if self.partition_info else message
            messages.append(_msg)
            new_offsets[partition] = message.offset + 1
            count -= 1
            if timeout is not None:
                timeout = max_time - time.time()

        self.size.value = 0
        self.events.start.clear()
        self.events.pause.set()

        # Update and commit offsets if necessary
        self.offsets.update(new_offsets)
        self.count_since_commit += len(messages)
        self._auto_commit()

        return messages
