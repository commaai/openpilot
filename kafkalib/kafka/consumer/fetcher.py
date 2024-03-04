from __future__ import absolute_import

import collections
import copy
import logging
import random
import sys
import time

from kafka.vendor import six

import kafka.errors as Errors
from kafka.future import Future
from kafka.metrics.stats import Avg, Count, Max, Rate
from kafka.protocol.fetch import FetchRequest
from kafka.protocol.offset import (
    OffsetRequest, OffsetResetStrategy, UNKNOWN_OFFSET
)
from kafka.record import MemoryRecords
from kafka.serializer import Deserializer
from kafka.structs import TopicPartition, OffsetAndTimestamp

log = logging.getLogger(__name__)


# Isolation levels
READ_UNCOMMITTED = 0
READ_COMMITTED = 1

ConsumerRecord = collections.namedtuple("ConsumerRecord",
    ["topic", "partition", "offset", "timestamp", "timestamp_type",
     "key", "value", "headers", "checksum", "serialized_key_size", "serialized_value_size", "serialized_header_size"])


CompletedFetch = collections.namedtuple("CompletedFetch",
    ["topic_partition", "fetched_offset", "response_version",
     "partition_data", "metric_aggregator"])


class NoOffsetForPartitionError(Errors.KafkaError):
    pass


class RecordTooLargeError(Errors.KafkaError):
    pass


class Fetcher(six.Iterator):
    DEFAULT_CONFIG = {
        'key_deserializer': None,
        'value_deserializer': None,
        'fetch_min_bytes': 1,
        'fetch_max_wait_ms': 500,
        'fetch_max_bytes': 52428800,
        'max_partition_fetch_bytes': 1048576,
        'max_poll_records': sys.maxsize,
        'check_crcs': True,
        'iterator_refetch_records': 1,  # undocumented -- interface may change
        'metric_group_prefix': 'consumer',
        'api_version': (0, 8, 0),
        'retry_backoff_ms': 100
    }

    def __init__(self, client, subscriptions, metrics, **configs):
        """Initialize a Kafka Message Fetcher.

        Keyword Arguments:
            key_deserializer (callable): Any callable that takes a
                raw message key and returns a deserialized key.
            value_deserializer (callable, optional): Any callable that takes a
                raw message value and returns a deserialized value.
            fetch_min_bytes (int): Minimum amount of data the server should
                return for a fetch request, otherwise wait up to
                fetch_max_wait_ms for more data to accumulate. Default: 1.
            fetch_max_wait_ms (int): The maximum amount of time in milliseconds
                the server will block before answering the fetch request if
                there isn't sufficient data to immediately satisfy the
                requirement given by fetch_min_bytes. Default: 500.
            fetch_max_bytes (int): The maximum amount of data the server should
                return for a fetch request. This is not an absolute maximum, if
                the first message in the first non-empty partition of the fetch
                is larger than this value, the message will still be returned
                to ensure that the consumer can make progress. NOTE: consumer
                performs fetches to multiple brokers in parallel so memory
                usage will depend on the number of brokers containing
                partitions for the topic.
                Supported Kafka version >= 0.10.1.0. Default: 52428800 (50 MB).
            max_partition_fetch_bytes (int): The maximum amount of data
                per-partition the server will return. The maximum total memory
                used for a request = #partitions * max_partition_fetch_bytes.
                This size must be at least as large as the maximum message size
                the server allows or else it is possible for the producer to
                send messages larger than the consumer can fetch. If that
                happens, the consumer can get stuck trying to fetch a large
                message on a certain partition. Default: 1048576.
            check_crcs (bool): Automatically check the CRC32 of the records
                consumed. This ensures no on-the-wire or on-disk corruption to
                the messages occurred. This check adds some overhead, so it may
                be disabled in cases seeking extreme performance. Default: True
        """
        self.config = copy.copy(self.DEFAULT_CONFIG)
        for key in self.config:
            if key in configs:
                self.config[key] = configs[key]

        self._client = client
        self._subscriptions = subscriptions
        self._completed_fetches = collections.deque()  # Unparsed responses
        self._next_partition_records = None  # Holds a single PartitionRecords until fully consumed
        self._iterator = None
        self._fetch_futures = collections.deque()
        self._sensors = FetchManagerMetrics(metrics, self.config['metric_group_prefix'])
        self._isolation_level = READ_UNCOMMITTED

    def send_fetches(self):
        """Send FetchRequests for all assigned partitions that do not already have
        an in-flight fetch or pending fetch data.

        Returns:
            List of Futures: each future resolves to a FetchResponse
        """
        futures = []
        for node_id, request in six.iteritems(self._create_fetch_requests()):
            if self._client.ready(node_id):
                log.debug("Sending FetchRequest to node %s", node_id)
                future = self._client.send(node_id, request, wakeup=False)
                future.add_callback(self._handle_fetch_response, request, time.time())
                future.add_errback(log.error, 'Fetch to node %s failed: %s', node_id)
                futures.append(future)
        self._fetch_futures.extend(futures)
        self._clean_done_fetch_futures()
        return futures

    def reset_offsets_if_needed(self, partitions):
        """Lookup and set offsets for any partitions which are awaiting an
        explicit reset.

        Arguments:
            partitions (set of TopicPartitions): the partitions to reset
        """
        for tp in partitions:
            # TODO: If there are several offsets to reset, we could submit offset requests in parallel
            if self._subscriptions.is_assigned(tp) and self._subscriptions.is_offset_reset_needed(tp):
                self._reset_offset(tp)

    def _clean_done_fetch_futures(self):
        while True:
            if not self._fetch_futures:
                break
            if not self._fetch_futures[0].is_done:
                break
            self._fetch_futures.popleft()

    def in_flight_fetches(self):
        """Return True if there are any unprocessed FetchRequests in flight."""
        self._clean_done_fetch_futures()
        return bool(self._fetch_futures)

    def update_fetch_positions(self, partitions):
        """Update the fetch positions for the provided partitions.

        Arguments:
            partitions (list of TopicPartitions): partitions to update

        Raises:
            NoOffsetForPartitionError: if no offset is stored for a given
                partition and no reset policy is available
        """
        # reset the fetch position to the committed position
        for tp in partitions:
            if not self._subscriptions.is_assigned(tp):
                log.warning("partition %s is not assigned - skipping offset"
                            " update", tp)
                continue
            elif self._subscriptions.is_fetchable(tp):
                log.warning("partition %s is still fetchable -- skipping offset"
                            " update", tp)
                continue

            if self._subscriptions.is_offset_reset_needed(tp):
                self._reset_offset(tp)
            elif self._subscriptions.assignment[tp].committed is None:
                # there's no committed position, so we need to reset with the
                # default strategy
                self._subscriptions.need_offset_reset(tp)
                self._reset_offset(tp)
            else:
                committed = self._subscriptions.assignment[tp].committed.offset
                log.debug("Resetting offset for partition %s to the committed"
                          " offset %s", tp, committed)
                self._subscriptions.seek(tp, committed)

    def get_offsets_by_times(self, timestamps, timeout_ms):
        offsets = self._retrieve_offsets(timestamps, timeout_ms)
        for tp in timestamps:
            if tp not in offsets:
                offsets[tp] = None
            else:
                offset, timestamp = offsets[tp]
                offsets[tp] = OffsetAndTimestamp(offset, timestamp)
        return offsets

    def beginning_offsets(self, partitions, timeout_ms):
        return self.beginning_or_end_offset(
            partitions, OffsetResetStrategy.EARLIEST, timeout_ms)

    def end_offsets(self, partitions, timeout_ms):
        return self.beginning_or_end_offset(
            partitions, OffsetResetStrategy.LATEST, timeout_ms)

    def beginning_or_end_offset(self, partitions, timestamp, timeout_ms):
        timestamps = dict([(tp, timestamp) for tp in partitions])
        offsets = self._retrieve_offsets(timestamps, timeout_ms)
        for tp in timestamps:
            offsets[tp] = offsets[tp][0]
        return offsets

    def _reset_offset(self, partition):
        """Reset offsets for the given partition using the offset reset strategy.

        Arguments:
            partition (TopicPartition): the partition that needs reset offset

        Raises:
            NoOffsetForPartitionError: if no offset reset strategy is defined
        """
        timestamp = self._subscriptions.assignment[partition].reset_strategy
        if timestamp is OffsetResetStrategy.EARLIEST:
            strategy = 'earliest'
        elif timestamp is OffsetResetStrategy.LATEST:
            strategy = 'latest'
        else:
            raise NoOffsetForPartitionError(partition)

        log.debug("Resetting offset for partition %s to %s offset.",
                  partition, strategy)
        offsets = self._retrieve_offsets({partition: timestamp})

        if partition in offsets:
            offset = offsets[partition][0]

            # we might lose the assignment while fetching the offset,
            # so check it is still active
            if self._subscriptions.is_assigned(partition):
                self._subscriptions.seek(partition, offset)
        else:
            log.debug("Could not find offset for partition %s since it is probably deleted" % (partition,))

    def _retrieve_offsets(self, timestamps, timeout_ms=float("inf")):
        """Fetch offset for each partition passed in ``timestamps`` map.

        Blocks until offsets are obtained, a non-retriable exception is raised
        or ``timeout_ms`` passed.

        Arguments:
            timestamps: {TopicPartition: int} dict with timestamps to fetch
                offsets by. -1 for the latest available, -2 for the earliest
                available. Otherwise timestamp is treated as epoch milliseconds.

        Returns:
            {TopicPartition: (int, int)}: Mapping of partition to
                retrieved offset and timestamp. If offset does not exist for
                the provided timestamp, that partition will be missing from
                this mapping.
        """
        if not timestamps:
            return {}

        start_time = time.time()
        remaining_ms = timeout_ms
        timestamps = copy.copy(timestamps)
        while remaining_ms > 0:
            if not timestamps:
                return {}

            future = self._send_offset_requests(timestamps)
            self._client.poll(future=future, timeout_ms=remaining_ms)

            if future.succeeded():
                return future.value
            if not future.retriable():
                raise future.exception  # pylint: disable-msg=raising-bad-type

            elapsed_ms = (time.time() - start_time) * 1000
            remaining_ms = timeout_ms - elapsed_ms
            if remaining_ms < 0:
                break

            if future.exception.invalid_metadata:
                refresh_future = self._client.cluster.request_update()
                self._client.poll(future=refresh_future, timeout_ms=remaining_ms)

                # Issue #1780
                # Recheck partition existence after after a successful metadata refresh
                if refresh_future.succeeded() and isinstance(future.exception, Errors.StaleMetadata):
                    log.debug("Stale metadata was raised, and we now have an updated metadata. Rechecking partition existence")
                    unknown_partition = future.exception.args[0]  # TopicPartition from StaleMetadata
                    if self._client.cluster.leader_for_partition(unknown_partition) is None:
                        log.debug("Removed partition %s from offsets retrieval" % (unknown_partition, ))
                        timestamps.pop(unknown_partition)
            else:
                time.sleep(self.config['retry_backoff_ms'] / 1000.0)

            elapsed_ms = (time.time() - start_time) * 1000
            remaining_ms = timeout_ms - elapsed_ms

        raise Errors.KafkaTimeoutError(
            "Failed to get offsets by timestamps in %s ms" % (timeout_ms,))

    def fetched_records(self, max_records=None, update_offsets=True):
        """Returns previously fetched records and updates consumed offsets.

        Arguments:
            max_records (int): Maximum number of records returned. Defaults
                to max_poll_records configuration.

        Raises:
            OffsetOutOfRangeError: if no subscription offset_reset_strategy
            CorruptRecordException: if message crc validation fails (check_crcs
                must be set to True)
            RecordTooLargeError: if a message is larger than the currently
                configured max_partition_fetch_bytes
            TopicAuthorizationError: if consumer is not authorized to fetch
                messages from the topic

        Returns: (records (dict), partial (bool))
            records: {TopicPartition: [messages]}
            partial: True if records returned did not fully drain any pending
                partition requests. This may be useful for choosing when to
                pipeline additional fetch requests.
        """
        if max_records is None:
            max_records = self.config['max_poll_records']
        assert max_records > 0

        drained = collections.defaultdict(list)
        records_remaining = max_records

        while records_remaining > 0:
            if not self._next_partition_records:
                if not self._completed_fetches:
                    break
                completion = self._completed_fetches.popleft()
                self._next_partition_records = self._parse_fetched_data(completion)
            else:
                records_remaining -= self._append(drained,
                                                  self._next_partition_records,
                                                  records_remaining,
                                                  update_offsets)
        return dict(drained), bool(self._completed_fetches)

    def _append(self, drained, part, max_records, update_offsets):
        if not part:
            return 0

        tp = part.topic_partition
        fetch_offset = part.fetch_offset
        if not self._subscriptions.is_assigned(tp):
            # this can happen when a rebalance happened before
            # fetched records are returned to the consumer's poll call
            log.debug("Not returning fetched records for partition %s"
                      " since it is no longer assigned", tp)
        else:
            # note that the position should always be available
            # as long as the partition is still assigned
            position = self._subscriptions.assignment[tp].position
            if not self._subscriptions.is_fetchable(tp):
                # this can happen when a partition is paused before
                # fetched records are returned to the consumer's poll call
                log.debug("Not returning fetched records for assigned partition"
                          " %s since it is no longer fetchable", tp)

            elif fetch_offset == position:
                # we are ensured to have at least one record since we already checked for emptiness
                part_records = part.take(max_records)
                next_offset = part_records[-1].offset + 1

                log.log(0, "Returning fetched records at offset %d for assigned"
                           " partition %s and update position to %s", position,
                           tp, next_offset)

                for record in part_records:
                    drained[tp].append(record)

                if update_offsets:
                    self._subscriptions.assignment[tp].position = next_offset
                return len(part_records)

            else:
                # these records aren't next in line based on the last consumed
                # position, ignore them they must be from an obsolete request
                log.debug("Ignoring fetched records for %s at offset %s since"
                          " the current position is %d", tp, part.fetch_offset,
                          position)

        part.discard()
        return 0

    def _message_generator(self):
        """Iterate over fetched_records"""
        while self._next_partition_records or self._completed_fetches:

            if not self._next_partition_records:
                completion = self._completed_fetches.popleft()
                self._next_partition_records = self._parse_fetched_data(completion)
                continue

            # Send additional FetchRequests when the internal queue is low
            # this should enable moderate pipelining
            if len(self._completed_fetches) <= self.config['iterator_refetch_records']:
                self.send_fetches()

            tp = self._next_partition_records.topic_partition

            # We can ignore any prior signal to drop pending message sets
            # because we are starting from a fresh one where fetch_offset == position
            # i.e., the user seek()'d to this position
            self._subscriptions.assignment[tp].drop_pending_message_set = False

            for msg in self._next_partition_records.take():

                # Because we are in a generator, it is possible for
                # subscription state to change between yield calls
                # so we need to re-check on each loop
                # this should catch assignment changes, pauses
                # and resets via seek_to_beginning / seek_to_end
                if not self._subscriptions.is_fetchable(tp):
                    log.debug("Not returning fetched records for partition %s"
                              " since it is no longer fetchable", tp)
                    self._next_partition_records = None
                    break

                # If there is a seek during message iteration,
                # we should stop unpacking this message set and
                # wait for a new fetch response that aligns with the
                # new seek position
                elif self._subscriptions.assignment[tp].drop_pending_message_set:
                    log.debug("Skipping remainder of message set for partition %s", tp)
                    self._subscriptions.assignment[tp].drop_pending_message_set = False
                    self._next_partition_records = None
                    break

                # Compressed messagesets may include earlier messages
                elif msg.offset < self._subscriptions.assignment[tp].position:
                    log.debug("Skipping message offset: %s (expecting %s)",
                              msg.offset,
                              self._subscriptions.assignment[tp].position)
                    continue

                self._subscriptions.assignment[tp].position = msg.offset + 1
                yield msg

            self._next_partition_records = None

    def _unpack_message_set(self, tp, records):
        try:
            batch = records.next_batch()
            while batch is not None:

                # LegacyRecordBatch cannot access either base_offset or last_offset_delta
                try:
                    self._subscriptions.assignment[tp].last_offset_from_message_batch = batch.base_offset + \
                                                                                        batch.last_offset_delta
                except AttributeError:
                    pass

                for record in batch:
                    key_size = len(record.key) if record.key is not None else -1
                    value_size = len(record.value) if record.value is not None else -1
                    key = self._deserialize(
                        self.config['key_deserializer'],
                        tp.topic, record.key)
                    value = self._deserialize(
                        self.config['value_deserializer'],
                        tp.topic, record.value)
                    headers = record.headers
                    header_size = sum(
                        len(h_key.encode("utf-8")) + (len(h_val) if h_val is not None else 0) for h_key, h_val in
                        headers) if headers else -1
                    yield ConsumerRecord(
                        tp.topic, tp.partition, record.offset, record.timestamp,
                        record.timestamp_type, key, value, headers, record.checksum,
                        key_size, value_size, header_size)

                batch = records.next_batch()

        # If unpacking raises StopIteration, it is erroneously
        # caught by the generator. We want all exceptions to be raised
        # back to the user. See Issue 545
        except StopIteration as e:
            log.exception('StopIteration raised unpacking messageset')
            raise RuntimeError('StopIteration raised unpacking messageset')

    def __iter__(self):  # pylint: disable=non-iterator-returned
        return self

    def __next__(self):
        if not self._iterator:
            self._iterator = self._message_generator()
        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator = None
            raise

    def _deserialize(self, f, topic, bytes_):
        if not f:
            return bytes_
        if isinstance(f, Deserializer):
            return f.deserialize(topic, bytes_)
        return f(bytes_)

    def _send_offset_requests(self, timestamps):
        """Fetch offsets for each partition in timestamps dict. This may send
        request to multiple nodes, based on who is Leader for partition.

        Arguments:
            timestamps (dict): {TopicPartition: int} mapping of fetching
                timestamps.

        Returns:
            Future: resolves to a mapping of retrieved offsets
        """
        timestamps_by_node = collections.defaultdict(dict)
        for partition, timestamp in six.iteritems(timestamps):
            node_id = self._client.cluster.leader_for_partition(partition)
            if node_id is None:
                self._client.add_topic(partition.topic)
                log.debug("Partition %s is unknown for fetching offset,"
                          " wait for metadata refresh", partition)
                return Future().failure(Errors.StaleMetadata(partition))
            elif node_id == -1:
                log.debug("Leader for partition %s unavailable for fetching "
                          "offset, wait for metadata refresh", partition)
                return Future().failure(
                    Errors.LeaderNotAvailableError(partition))
            else:
                timestamps_by_node[node_id][partition] = timestamp

        # Aggregate results until we have all
        list_offsets_future = Future()
        responses = []
        node_count = len(timestamps_by_node)

        def on_success(value):
            responses.append(value)
            if len(responses) == node_count:
                offsets = {}
                for r in responses:
                    offsets.update(r)
                list_offsets_future.success(offsets)

        def on_fail(err):
            if not list_offsets_future.is_done:
                list_offsets_future.failure(err)

        for node_id, timestamps in six.iteritems(timestamps_by_node):
            _f = self._send_offset_request(node_id, timestamps)
            _f.add_callback(on_success)
            _f.add_errback(on_fail)
        return list_offsets_future

    def _send_offset_request(self, node_id, timestamps):
        by_topic = collections.defaultdict(list)
        for tp, timestamp in six.iteritems(timestamps):
            if self.config['api_version'] >= (0, 10, 1):
                data = (tp.partition, timestamp)
            else:
                data = (tp.partition, timestamp, 1)
            by_topic[tp.topic].append(data)

        if self.config['api_version'] >= (0, 10, 1):
            request = OffsetRequest[1](-1, list(six.iteritems(by_topic)))
        else:
            request = OffsetRequest[0](-1, list(six.iteritems(by_topic)))

        # Client returns a future that only fails on network issues
        # so create a separate future and attach a callback to update it
        # based on response error codes
        future = Future()

        _f = self._client.send(node_id, request)
        _f.add_callback(self._handle_offset_response, future)
        _f.add_errback(lambda e: future.failure(e))
        return future

    def _handle_offset_response(self, future, response):
        """Callback for the response of the list offset call above.

        Arguments:
            future (Future): the future to update based on response
            response (OffsetResponse): response from the server

        Raises:
            AssertionError: if response does not match partition
        """
        timestamp_offset_map = {}
        for topic, part_data in response.topics:
            for partition_info in part_data:
                partition, error_code = partition_info[:2]
                partition = TopicPartition(topic, partition)
                error_type = Errors.for_code(error_code)
                if error_type is Errors.NoError:
                    if response.API_VERSION == 0:
                        offsets = partition_info[2]
                        assert len(offsets) <= 1, 'Expected OffsetResponse with one offset'
                        if not offsets:
                            offset = UNKNOWN_OFFSET
                        else:
                            offset = offsets[0]
                        log.debug("Handling v0 ListOffsetResponse response for %s. "
                                  "Fetched offset %s", partition, offset)
                        if offset != UNKNOWN_OFFSET:
                            timestamp_offset_map[partition] = (offset, None)
                    else:
                        timestamp, offset = partition_info[2:]
                        log.debug("Handling ListOffsetResponse response for %s. "
                                  "Fetched offset %s, timestamp %s",
                                  partition, offset, timestamp)
                        if offset != UNKNOWN_OFFSET:
                            timestamp_offset_map[partition] = (offset, timestamp)
                elif error_type is Errors.UnsupportedForMessageFormatError:
                    # The message format on the broker side is before 0.10.0,
                    # we simply put None in the response.
                    log.debug("Cannot search by timestamp for partition %s because the"
                              " message format version is before 0.10.0", partition)
                elif error_type is Errors.NotLeaderForPartitionError:
                    log.debug("Attempt to fetch offsets for partition %s failed due"
                              " to obsolete leadership information, retrying.",
                              partition)
                    future.failure(error_type(partition))
                    return
                elif error_type is Errors.UnknownTopicOrPartitionError:
                    log.warning("Received unknown topic or partition error in ListOffset "
                             "request for partition %s. The topic/partition " +
                             "may not exist or the user may not have Describe access "
                             "to it.", partition)
                    future.failure(error_type(partition))
                    return
                else:
                    log.warning("Attempt to fetch offsets for partition %s failed due to:"
                                " %s", partition, error_type)
                    future.failure(error_type(partition))
                    return
        if not future.is_done:
            future.success(timestamp_offset_map)

    def _fetchable_partitions(self):
        fetchable = self._subscriptions.fetchable_partitions()
        # do not fetch a partition if we have a pending fetch response to process
        current = self._next_partition_records
        pending = copy.copy(self._completed_fetches)
        if current:
            fetchable.discard(current.topic_partition)
        for fetch in pending:
            fetchable.discard(fetch.topic_partition)
        return fetchable

    def _create_fetch_requests(self):
        """Create fetch requests for all assigned partitions, grouped by node.

        FetchRequests skipped if no leader, or node has requests in flight

        Returns:
            dict: {node_id: FetchRequest, ...} (version depends on api_version)
        """
        # create the fetch info as a dict of lists of partition info tuples
        # which can be passed to FetchRequest() via .items()
        fetchable = collections.defaultdict(lambda: collections.defaultdict(list))

        for partition in self._fetchable_partitions():
            node_id = self._client.cluster.leader_for_partition(partition)

            # advance position for any deleted compacted messages if required
            if self._subscriptions.assignment[partition].last_offset_from_message_batch:
                next_offset_from_batch_header = self._subscriptions.assignment[partition].last_offset_from_message_batch + 1
                if next_offset_from_batch_header > self._subscriptions.assignment[partition].position:
                    log.debug(
                        "Advance position for partition %s from %s to %s (last message batch location plus one)"
                        " to correct for deleted compacted messages",
                        partition, self._subscriptions.assignment[partition].position, next_offset_from_batch_header)
                    self._subscriptions.assignment[partition].position = next_offset_from_batch_header

            position = self._subscriptions.assignment[partition].position

            # fetch if there is a leader and no in-flight requests
            if node_id is None or node_id == -1:
                log.debug("No leader found for partition %s."
                          " Requesting metadata update", partition)
                self._client.cluster.request_update()

            elif self._client.in_flight_request_count(node_id) == 0:
                partition_info = (
                    partition.partition,
                    position,
                    self.config['max_partition_fetch_bytes']
                )
                fetchable[node_id][partition.topic].append(partition_info)
                log.debug("Adding fetch request for partition %s at offset %d",
                          partition, position)
            else:
                log.log(0, "Skipping fetch for partition %s because there is an inflight request to node %s",
                        partition, node_id)

        if self.config['api_version'] >= (0, 11, 0):
            version = 4
        elif self.config['api_version'] >= (0, 10, 1):
            version = 3
        elif self.config['api_version'] >= (0, 10):
            version = 2
        elif self.config['api_version'] == (0, 9):
            version = 1
        else:
            version = 0
        requests = {}
        for node_id, partition_data in six.iteritems(fetchable):
            if version < 3:
                requests[node_id] = FetchRequest[version](
                    -1,  # replica_id
                    self.config['fetch_max_wait_ms'],
                    self.config['fetch_min_bytes'],
                    partition_data.items())
            else:
                # As of version == 3 partitions will be returned in order as
                # they are requested, so to avoid starvation with
                # `fetch_max_bytes` option we need this shuffle
                # NOTE: we do have partition_data in random order due to usage
                #       of unordered structures like dicts, but that does not
                #       guarantee equal distribution, and starting in Python3.6
                #       dicts retain insert order.
                partition_data = list(partition_data.items())
                random.shuffle(partition_data)
                if version == 3:
                    requests[node_id] = FetchRequest[version](
                        -1,  # replica_id
                        self.config['fetch_max_wait_ms'],
                        self.config['fetch_min_bytes'],
                        self.config['fetch_max_bytes'],
                        partition_data)
                else:
                    requests[node_id] = FetchRequest[version](
                        -1,  # replica_id
                        self.config['fetch_max_wait_ms'],
                        self.config['fetch_min_bytes'],
                        self.config['fetch_max_bytes'],
                        self._isolation_level,
                        partition_data)
        return requests

    def _handle_fetch_response(self, request, send_time, response):
        """The callback for fetch completion"""
        fetch_offsets = {}
        for topic, partitions in request.topics:
            for partition_data in partitions:
                partition, offset = partition_data[:2]
                fetch_offsets[TopicPartition(topic, partition)] = offset

        partitions = set([TopicPartition(topic, partition_data[0])
                          for topic, partitions in response.topics
                          for partition_data in partitions])
        metric_aggregator = FetchResponseMetricAggregator(self._sensors, partitions)

        # randomized ordering should improve balance for short-lived consumers
        random.shuffle(response.topics)
        for topic, partitions in response.topics:
            random.shuffle(partitions)
            for partition_data in partitions:
                tp = TopicPartition(topic, partition_data[0])
                completed_fetch = CompletedFetch(
                    tp, fetch_offsets[tp],
                    response.API_VERSION,
                    partition_data[1:],
                    metric_aggregator
                )
                self._completed_fetches.append(completed_fetch)

        if response.API_VERSION >= 1:
            self._sensors.fetch_throttle_time_sensor.record(response.throttle_time_ms)
        self._sensors.fetch_latency.record((time.time() - send_time) * 1000)

    def _parse_fetched_data(self, completed_fetch):
        tp = completed_fetch.topic_partition
        fetch_offset = completed_fetch.fetched_offset
        num_bytes = 0
        records_count = 0
        parsed_records = None

        error_code, highwater = completed_fetch.partition_data[:2]
        error_type = Errors.for_code(error_code)

        try:
            if not self._subscriptions.is_fetchable(tp):
                # this can happen when a rebalance happened or a partition
                # consumption paused while fetch is still in-flight
                log.debug("Ignoring fetched records for partition %s"
                          " since it is no longer fetchable", tp)

            elif error_type is Errors.NoError:
                self._subscriptions.assignment[tp].highwater = highwater

                # we are interested in this fetch only if the beginning
                # offset (of the *request*) matches the current consumed position
                # Note that the *response* may return a messageset that starts
                # earlier (e.g., compressed messages) or later (e.g., compacted topic)
                position = self._subscriptions.assignment[tp].position
                if position is None or position != fetch_offset:
                    log.debug("Discarding fetch response for partition %s"
                              " since its offset %d does not match the"
                              " expected offset %d", tp, fetch_offset,
                              position)
                    return None

                records = MemoryRecords(completed_fetch.partition_data[-1])
                if records.has_next():
                    log.debug("Adding fetched record for partition %s with"
                              " offset %d to buffered record list", tp,
                              position)
                    unpacked = list(self._unpack_message_set(tp, records))
                    parsed_records = self.PartitionRecords(fetch_offset, tp, unpacked)
                    last_offset = unpacked[-1].offset
                    self._sensors.records_fetch_lag.record(highwater - last_offset)
                    num_bytes = records.valid_bytes()
                    records_count = len(unpacked)
                elif records.size_in_bytes() > 0:
                    # we did not read a single message from a non-empty
                    # buffer because that message's size is larger than
                    # fetch size, in this case record this exception
                    record_too_large_partitions = {tp: fetch_offset}
                    raise RecordTooLargeError(
                        "There are some messages at [Partition=Offset]: %s "
                        " whose size is larger than the fetch size %s"
                        " and hence cannot be ever returned."
                        " Increase the fetch size, or decrease the maximum message"
                        " size the broker will allow." % (
                            record_too_large_partitions,
                            self.config['max_partition_fetch_bytes']),
                        record_too_large_partitions)
                self._sensors.record_topic_fetch_metrics(tp.topic, num_bytes, records_count)

            elif error_type in (Errors.NotLeaderForPartitionError,
                                Errors.UnknownTopicOrPartitionError):
                self._client.cluster.request_update()
            elif error_type is Errors.OffsetOutOfRangeError:
                position = self._subscriptions.assignment[tp].position
                if position is None or position != fetch_offset:
                    log.debug("Discarding stale fetch response for partition %s"
                              " since the fetched offset %d does not match the"
                              " current offset %d", tp, fetch_offset, position)
                elif self._subscriptions.has_default_offset_reset_policy():
                    log.info("Fetch offset %s is out of range for topic-partition %s", fetch_offset, tp)
                    self._subscriptions.need_offset_reset(tp)
                else:
                    raise Errors.OffsetOutOfRangeError({tp: fetch_offset})

            elif error_type is Errors.TopicAuthorizationFailedError:
                log.warning("Not authorized to read from topic %s.", tp.topic)
                raise Errors.TopicAuthorizationFailedError(set(tp.topic))
            elif error_type is Errors.UnknownError:
                log.warning("Unknown error fetching data for topic-partition %s", tp)
            else:
                raise error_type('Unexpected error while fetching data')

        finally:
            completed_fetch.metric_aggregator.record(tp, num_bytes, records_count)

        return parsed_records

    class PartitionRecords(object):
        def __init__(self, fetch_offset, tp, messages):
            self.fetch_offset = fetch_offset
            self.topic_partition = tp
            self.messages = messages
            # When fetching an offset that is in the middle of a
            # compressed batch, we will get all messages in the batch.
            # But we want to start 'take' at the fetch_offset
            # (or the next highest offset in case the message was compacted)
            for i, msg in enumerate(messages):
                if msg.offset < fetch_offset:
                    log.debug("Skipping message offset: %s (expecting %s)",
                              msg.offset, fetch_offset)
                else:
                    self.message_idx = i
                    break

            else:
                self.message_idx = 0
                self.messages = None

        # For truthiness evaluation we need to define __len__ or __nonzero__
        def __len__(self):
            if self.messages is None or self.message_idx >= len(self.messages):
                return 0
            return len(self.messages) - self.message_idx

        def discard(self):
            self.messages = None

        def take(self, n=None):
            if not len(self):
                return []
            if n is None or n > len(self):
                n = len(self)
            next_idx = self.message_idx + n
            res = self.messages[self.message_idx:next_idx]
            self.message_idx = next_idx
            # fetch_offset should be incremented by 1 to parallel the
            # subscription position (also incremented by 1)
            self.fetch_offset = max(self.fetch_offset, res[-1].offset + 1)
            return res


class FetchResponseMetricAggregator(object):
    """
    Since we parse the message data for each partition from each fetch
    response lazily, fetch-level metrics need to be aggregated as the messages
    from each partition are parsed. This class is used to facilitate this
    incremental aggregation.
    """
    def __init__(self, sensors, partitions):
        self.sensors = sensors
        self.unrecorded_partitions = partitions
        self.total_bytes = 0
        self.total_records = 0

    def record(self, partition, num_bytes, num_records):
        """
        After each partition is parsed, we update the current metric totals
        with the total bytes and number of records parsed. After all partitions
        have reported, we write the metric.
        """
        self.unrecorded_partitions.remove(partition)
        self.total_bytes += num_bytes
        self.total_records += num_records

        # once all expected partitions from the fetch have reported in, record the metrics
        if not self.unrecorded_partitions:
            self.sensors.bytes_fetched.record(self.total_bytes)
            self.sensors.records_fetched.record(self.total_records)


class FetchManagerMetrics(object):
    def __init__(self, metrics, prefix):
        self.metrics = metrics
        self.group_name = '%s-fetch-manager-metrics' % (prefix,)

        self.bytes_fetched = metrics.sensor('bytes-fetched')
        self.bytes_fetched.add(metrics.metric_name('fetch-size-avg', self.group_name,
            'The average number of bytes fetched per request'), Avg())
        self.bytes_fetched.add(metrics.metric_name('fetch-size-max', self.group_name,
            'The maximum number of bytes fetched per request'), Max())
        self.bytes_fetched.add(metrics.metric_name('bytes-consumed-rate', self.group_name,
            'The average number of bytes consumed per second'), Rate())

        self.records_fetched = self.metrics.sensor('records-fetched')
        self.records_fetched.add(metrics.metric_name('records-per-request-avg', self.group_name,
            'The average number of records in each request'), Avg())
        self.records_fetched.add(metrics.metric_name('records-consumed-rate', self.group_name,
            'The average number of records consumed per second'), Rate())

        self.fetch_latency = metrics.sensor('fetch-latency')
        self.fetch_latency.add(metrics.metric_name('fetch-latency-avg', self.group_name,
            'The average time taken for a fetch request.'), Avg())
        self.fetch_latency.add(metrics.metric_name('fetch-latency-max', self.group_name,
            'The max time taken for any fetch request.'), Max())
        self.fetch_latency.add(metrics.metric_name('fetch-rate', self.group_name,
            'The number of fetch requests per second.'), Rate(sampled_stat=Count()))

        self.records_fetch_lag = metrics.sensor('records-lag')
        self.records_fetch_lag.add(metrics.metric_name('records-lag-max', self.group_name,
            'The maximum lag in terms of number of records for any partition in self window'), Max())

        self.fetch_throttle_time_sensor = metrics.sensor('fetch-throttle-time')
        self.fetch_throttle_time_sensor.add(metrics.metric_name('fetch-throttle-time-avg', self.group_name,
            'The average throttle time in ms'), Avg())
        self.fetch_throttle_time_sensor.add(metrics.metric_name('fetch-throttle-time-max', self.group_name,
            'The maximum throttle time in ms'), Max())

    def record_topic_fetch_metrics(self, topic, num_bytes, num_records):
        # record bytes fetched
        name = '.'.join(['topic', topic, 'bytes-fetched'])
        bytes_fetched = self.metrics.get_sensor(name)
        if not bytes_fetched:
            metric_tags = {'topic': topic.replace('.', '_')}

            bytes_fetched = self.metrics.sensor(name)
            bytes_fetched.add(self.metrics.metric_name('fetch-size-avg',
                    self.group_name,
                    'The average number of bytes fetched per request for topic %s' % (topic,),
                    metric_tags), Avg())
            bytes_fetched.add(self.metrics.metric_name('fetch-size-max',
                    self.group_name,
                    'The maximum number of bytes fetched per request for topic %s' % (topic,),
                    metric_tags), Max())
            bytes_fetched.add(self.metrics.metric_name('bytes-consumed-rate',
                    self.group_name,
                    'The average number of bytes consumed per second for topic %s' % (topic,),
                    metric_tags), Rate())
        bytes_fetched.record(num_bytes)

        # record records fetched
        name = '.'.join(['topic', topic, 'records-fetched'])
        records_fetched = self.metrics.get_sensor(name)
        if not records_fetched:
            metric_tags = {'topic': topic.replace('.', '_')}

            records_fetched = self.metrics.sensor(name)
            records_fetched.add(self.metrics.metric_name('records-per-request-avg',
                    self.group_name,
                    'The average number of records in each request for topic %s' % (topic,),
                    metric_tags), Avg())
            records_fetched.add(self.metrics.metric_name('records-consumed-rate',
                    self.group_name,
                    'The average number of records consumed per second for topic %s' % (topic,),
                    metric_tags), Rate())
        records_fetched.record(num_records)
