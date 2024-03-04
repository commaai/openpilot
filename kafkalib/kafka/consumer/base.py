from __future__ import absolute_import

import atexit
import logging
import numbers
from threading import Lock
import warnings

from kafka.errors import (
    UnknownTopicOrPartitionError, check_error, KafkaError)
from kafka.structs import (
    OffsetRequestPayload, OffsetCommitRequestPayload, OffsetFetchRequestPayload)
from kafka.util import ReentrantTimer


log = logging.getLogger('kafka.consumer')

AUTO_COMMIT_MSG_COUNT = 100
AUTO_COMMIT_INTERVAL = 5000

FETCH_DEFAULT_BLOCK_TIMEOUT = 1
FETCH_MAX_WAIT_TIME = 100
FETCH_MIN_BYTES = 4096
FETCH_BUFFER_SIZE_BYTES = 4096
MAX_FETCH_BUFFER_SIZE_BYTES = FETCH_BUFFER_SIZE_BYTES * 8

ITER_TIMEOUT_SECONDS = 60
NO_MESSAGES_WAIT_TIME_SECONDS = 0.1
FULL_QUEUE_WAIT_TIME_SECONDS = 0.1

MAX_BACKOFF_SECONDS = 60

class Consumer(object):
    """
    Base class to be used by other consumers. Not to be used directly

    This base class provides logic for

    * initialization and fetching metadata of partitions
    * Auto-commit logic
    * APIs for fetching pending message count

    """
    def __init__(self, client, group, topic, partitions=None, auto_commit=True,
                 auto_commit_every_n=AUTO_COMMIT_MSG_COUNT,
                 auto_commit_every_t=AUTO_COMMIT_INTERVAL):

        warnings.warn('deprecated -- this class will be removed in a future'
                      ' release. Use KafkaConsumer instead.',
                      DeprecationWarning)
        self.client = client
        self.topic = topic
        self.group = group
        self.client.load_metadata_for_topics(topic, ignore_leadernotavailable=True)
        self.offsets = {}

        if partitions is None:
            partitions = self.client.get_partition_ids_for_topic(topic)
        else:
            assert all(isinstance(x, numbers.Integral) for x in partitions)

        # Variables for handling offset commits
        self.commit_lock = Lock()
        self.commit_timer = None
        self.count_since_commit = 0
        self.auto_commit = auto_commit
        self.auto_commit_every_n = auto_commit_every_n
        self.auto_commit_every_t = auto_commit_every_t

        # Set up the auto-commit timer
        if auto_commit is True and auto_commit_every_t is not None:
            self.commit_timer = ReentrantTimer(auto_commit_every_t,
                                               self.commit)
            self.commit_timer.start()

        # Set initial offsets
        if self.group is not None:
            self.fetch_last_known_offsets(partitions)
        else:
            for partition in partitions:
                self.offsets[partition] = 0

        # Register a cleanup handler
        def cleanup(obj):
            obj.stop()
        self._cleanup_func = cleanup
        atexit.register(cleanup, self)

        self.partition_info = False     # Do not return partition info in msgs

    def provide_partition_info(self):
        """
        Indicates that partition info must be returned by the consumer
        """
        self.partition_info = True

    def fetch_last_known_offsets(self, partitions=None):
        if self.group is None:
            raise ValueError('SimpleClient.group must not be None')

        if partitions is None:
            partitions = self.client.get_partition_ids_for_topic(self.topic)

        responses = self.client.send_offset_fetch_request(
            self.group,
            [OffsetFetchRequestPayload(self.topic, p) for p in partitions],
            fail_on_error=False
        )

        for resp in responses:
            try:
                check_error(resp)
            # API spec says server won't set an error here
            # but 0.8.1.1 does actually...
            except UnknownTopicOrPartitionError:
                pass

            # -1 offset signals no commit is currently stored
            if resp.offset == -1:
                self.offsets[resp.partition] = 0

            # Otherwise we committed the stored offset
            # and need to fetch the next one
            else:
                self.offsets[resp.partition] = resp.offset

    def commit(self, partitions=None):
        """Commit stored offsets to Kafka via OffsetCommitRequest (v0)

        Keyword Arguments:
            partitions (list): list of partitions to commit, default is to commit
                all of them

        Returns: True on success, False on failure
        """

        # short circuit if nothing happened. This check is kept outside
        # to prevent un-necessarily acquiring a lock for checking the state
        if self.count_since_commit == 0:
            return

        with self.commit_lock:
            # Do this check again, just in case the state has changed
            # during the lock acquiring timeout
            if self.count_since_commit == 0:
                return

            reqs = []
            if partitions is None:  # commit all partitions
                partitions = list(self.offsets.keys())

            log.debug('Committing new offsets for %s, partitions %s',
                     self.topic, partitions)
            for partition in partitions:
                offset = self.offsets[partition]
                log.debug('Commit offset %d in SimpleConsumer: '
                          'group=%s, topic=%s, partition=%s',
                          offset, self.group, self.topic, partition)

                reqs.append(OffsetCommitRequestPayload(self.topic, partition,
                                                offset, None))

            try:
                self.client.send_offset_commit_request(self.group, reqs)
            except KafkaError as e:
                log.error('%s saving offsets: %s', e.__class__.__name__, e)
                return False
            else:
                self.count_since_commit = 0
                return True

    def _auto_commit(self):
        """
        Check if we have to commit based on number of messages and commit
        """

        # Check if we are supposed to do an auto-commit
        if not self.auto_commit or self.auto_commit_every_n is None:
            return

        if self.count_since_commit >= self.auto_commit_every_n:
            self.commit()

    def stop(self):
        if self.commit_timer is not None:
            self.commit_timer.stop()
            self.commit()

        if hasattr(self, '_cleanup_func'):
            # Remove cleanup handler now that we've stopped

            # py3 supports unregistering
            if hasattr(atexit, 'unregister'):
                atexit.unregister(self._cleanup_func) # pylint: disable=no-member

            # py2 requires removing from private attribute...
            else:

                # ValueError on list.remove() if the exithandler no longer
                # exists is fine here
                try:
                    atexit._exithandlers.remove(  # pylint: disable=no-member
                        (self._cleanup_func, (self,), {}))
                except ValueError:
                    pass

            del self._cleanup_func

    def pending(self, partitions=None):
        """
        Gets the pending message count

        Keyword Arguments:
            partitions (list): list of partitions to check for, default is to check all
        """
        if partitions is None:
            partitions = self.offsets.keys()

        total = 0
        reqs = []

        for partition in partitions:
            reqs.append(OffsetRequestPayload(self.topic, partition, -1, 1))

        resps = self.client.send_offset_request(reqs)
        for resp in resps:
            partition = resp.partition
            pending = resp.offsets[0]
            offset = self.offsets[partition]
            total += pending - offset

        return total
