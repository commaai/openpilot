from __future__ import absolute_import, division

import collections
import copy
import logging
import threading
import time

from kafka.vendor import six

from kafka import errors as Errors
from kafka.metrics.measurable import AnonMeasurable
from kafka.metrics.stats import Avg, Max, Rate
from kafka.protocol.produce import ProduceRequest
from kafka.structs import TopicPartition
from kafka.version import __version__

log = logging.getLogger(__name__)


class Sender(threading.Thread):
    """
    The background thread that handles the sending of produce requests to the
    Kafka cluster. This thread makes metadata requests to renew its view of the
    cluster and then sends produce requests to the appropriate nodes.
    """
    DEFAULT_CONFIG = {
        'max_request_size': 1048576,
        'acks': 1,
        'retries': 0,
        'request_timeout_ms': 30000,
        'guarantee_message_order': False,
        'client_id': 'kafka-python-' + __version__,
        'api_version': (0, 8, 0),
    }

    def __init__(self, client, metadata, accumulator, metrics, **configs):
        super(Sender, self).__init__()
        self.config = copy.copy(self.DEFAULT_CONFIG)
        for key in self.config:
            if key in configs:
                self.config[key] = configs.pop(key)

        self.name = self.config['client_id'] + '-network-thread'
        self._client = client
        self._accumulator = accumulator
        self._metadata = client.cluster
        self._running = True
        self._force_close = False
        self._topics_to_add = set()
        self._sensors = SenderMetrics(metrics, self._client, self._metadata)

    def run(self):
        """The main run loop for the sender thread."""
        log.debug("Starting Kafka producer I/O thread.")

        # main loop, runs until close is called
        while self._running:
            try:
                self.run_once()
            except Exception:
                log.exception("Uncaught error in kafka producer I/O thread")

        log.debug("Beginning shutdown of Kafka producer I/O thread, sending"
                  " remaining records.")

        # okay we stopped accepting requests but there may still be
        # requests in the accumulator or waiting for acknowledgment,
        # wait until these are completed.
        while (not self._force_close
               and (self._accumulator.has_unsent()
                    or self._client.in_flight_request_count() > 0)):
            try:
                self.run_once()
            except Exception:
                log.exception("Uncaught error in kafka producer I/O thread")

        if self._force_close:
            # We need to fail all the incomplete batches and wake up the
            # threads waiting on the futures.
            self._accumulator.abort_incomplete_batches()

        try:
            self._client.close()
        except Exception:
            log.exception("Failed to close network client")

        log.debug("Shutdown of Kafka producer I/O thread has completed.")

    def run_once(self):
        """Run a single iteration of sending."""
        while self._topics_to_add:
            self._client.add_topic(self._topics_to_add.pop())

        # get the list of partitions with data ready to send
        result = self._accumulator.ready(self._metadata)
        ready_nodes, next_ready_check_delay, unknown_leaders_exist = result

        # if there are any partitions whose leaders are not known yet, force
        # metadata update
        if unknown_leaders_exist:
            log.debug('Unknown leaders exist, requesting metadata update')
            self._metadata.request_update()

        # remove any nodes we aren't ready to send to
        not_ready_timeout = float('inf')
        for node in list(ready_nodes):
            if not self._client.is_ready(node):
                log.debug('Node %s not ready; delaying produce of accumulated batch', node)
                self._client.maybe_connect(node, wakeup=False)
                ready_nodes.remove(node)
                not_ready_timeout = min(not_ready_timeout,
                                        self._client.connection_delay(node))

        # create produce requests
        batches_by_node = self._accumulator.drain(
            self._metadata, ready_nodes, self.config['max_request_size'])

        if self.config['guarantee_message_order']:
            # Mute all the partitions drained
            for batch_list in six.itervalues(batches_by_node):
                for batch in batch_list:
                    self._accumulator.muted.add(batch.topic_partition)

        expired_batches = self._accumulator.abort_expired_batches(
            self.config['request_timeout_ms'], self._metadata)
        for expired_batch in expired_batches:
            self._sensors.record_errors(expired_batch.topic_partition.topic, expired_batch.record_count)

        self._sensors.update_produce_request_metrics(batches_by_node)
        requests = self._create_produce_requests(batches_by_node)
        # If we have any nodes that are ready to send + have sendable data,
        # poll with 0 timeout so this can immediately loop and try sending more
        # data. Otherwise, the timeout is determined by nodes that have
        # partitions with data that isn't yet sendable (e.g. lingering, backing
        # off). Note that this specifically does not include nodes with
        # sendable data that aren't ready to send since they would cause busy
        # looping.
        poll_timeout_ms = min(next_ready_check_delay * 1000, not_ready_timeout)
        if ready_nodes:
            log.debug("Nodes with data ready to send: %s", ready_nodes) # trace
            log.debug("Created %d produce requests: %s", len(requests), requests) # trace
            poll_timeout_ms = 0

        for node_id, request in six.iteritems(requests):
            batches = batches_by_node[node_id]
            log.debug('Sending Produce Request: %r', request)
            (self._client.send(node_id, request, wakeup=False)
                 .add_callback(
                     self._handle_produce_response, node_id, time.time(), batches)
                 .add_errback(
                     self._failed_produce, batches, node_id))

        # if some partitions are already ready to be sent, the select time
        # would be 0; otherwise if some partition already has some data
        # accumulated but not ready yet, the select time will be the time
        # difference between now and its linger expiry time; otherwise the
        # select time will be the time difference between now and the
        # metadata expiry time
        self._client.poll(timeout_ms=poll_timeout_ms)

    def initiate_close(self):
        """Start closing the sender (won't complete until all data is sent)."""
        self._running = False
        self._accumulator.close()
        self.wakeup()

    def force_close(self):
        """Closes the sender without sending out any pending messages."""
        self._force_close = True
        self.initiate_close()

    def add_topic(self, topic):
        # This is generally called from a separate thread
        # so this needs to be a thread-safe operation
        # we assume that checking set membership across threads
        # is ok where self._client._topics should never
        # remove topics for a producer instance, only add them.
        if topic not in self._client._topics:
            self._topics_to_add.add(topic)
            self.wakeup()

    def _failed_produce(self, batches, node_id, error):
        log.debug("Error sending produce request to node %d: %s", node_id, error) # trace
        for batch in batches:
            self._complete_batch(batch, error, -1, None)

    def _handle_produce_response(self, node_id, send_time, batches, response):
        """Handle a produce response."""
        # if we have a response, parse it
        log.debug('Parsing produce response: %r', response)
        if response:
            batches_by_partition = dict([(batch.topic_partition, batch)
                                         for batch in batches])

            for topic, partitions in response.topics:
                for partition_info in partitions:
                    global_error = None
                    log_start_offset = None
                    if response.API_VERSION < 2:
                        partition, error_code, offset = partition_info
                        ts = None
                    elif 2 <= response.API_VERSION <= 4:
                        partition, error_code, offset, ts = partition_info
                    elif 5 <= response.API_VERSION <= 7:
                        partition, error_code, offset, ts, log_start_offset = partition_info
                    else:
                        # the ignored parameter is record_error of type list[(batch_index: int, error_message: str)]
                        partition, error_code, offset, ts, log_start_offset, _, global_error = partition_info
                    tp = TopicPartition(topic, partition)
                    error = Errors.for_code(error_code)
                    batch = batches_by_partition[tp]
                    self._complete_batch(batch, error, offset, ts, log_start_offset, global_error)

            if response.API_VERSION > 0:
                self._sensors.record_throttle_time(response.throttle_time_ms, node=node_id)

        else:
            # this is the acks = 0 case, just complete all requests
            for batch in batches:
                self._complete_batch(batch, None, -1, None)

    def _complete_batch(self, batch, error, base_offset, timestamp_ms=None, log_start_offset=None, global_error=None):
        """Complete or retry the given batch of records.

        Arguments:
            batch (RecordBatch): The record batch
            error (Exception): The error (or None if none)
            base_offset (int): The base offset assigned to the records if successful
            timestamp_ms (int, optional): The timestamp returned by the broker for this batch
            log_start_offset (int): The start offset of the log at the time this produce response was created
            global_error (str): The summarising error message
        """
        # Standardize no-error to None
        if error is Errors.NoError:
            error = None

        if error is not None and self._can_retry(batch, error):
            # retry
            log.warning("Got error produce response on topic-partition %s,"
                        " retrying (%d attempts left). Error: %s",
                        batch.topic_partition,
                        self.config['retries'] - batch.attempts - 1,
                        global_error or error)
            self._accumulator.reenqueue(batch)
            self._sensors.record_retries(batch.topic_partition.topic, batch.record_count)
        else:
            if error is Errors.TopicAuthorizationFailedError:
                error = error(batch.topic_partition.topic)

            # tell the user the result of their request
            batch.done(base_offset, timestamp_ms, error, log_start_offset, global_error)
            self._accumulator.deallocate(batch)
            if error is not None:
                self._sensors.record_errors(batch.topic_partition.topic, batch.record_count)

        if getattr(error, 'invalid_metadata', False):
            self._metadata.request_update()

        # Unmute the completed partition.
        if self.config['guarantee_message_order']:
            self._accumulator.muted.remove(batch.topic_partition)

    def _can_retry(self, batch, error):
        """
        We can retry a send if the error is transient and the number of
        attempts taken is fewer than the maximum allowed
        """
        return (batch.attempts < self.config['retries']
                and getattr(error, 'retriable', False))

    def _create_produce_requests(self, collated):
        """
        Transfer the record batches into a list of produce requests on a
        per-node basis.

        Arguments:
            collated: {node_id: [RecordBatch]}

        Returns:
            dict: {node_id: ProduceRequest} (version depends on api_version)
        """
        requests = {}
        for node_id, batches in six.iteritems(collated):
            requests[node_id] = self._produce_request(
                node_id, self.config['acks'],
                self.config['request_timeout_ms'], batches)
        return requests

    def _produce_request(self, node_id, acks, timeout, batches):
        """Create a produce request from the given record batches.

        Returns:
            ProduceRequest (version depends on api_version)
        """
        produce_records_by_partition = collections.defaultdict(dict)
        for batch in batches:
            topic = batch.topic_partition.topic
            partition = batch.topic_partition.partition

            buf = batch.records.buffer()
            produce_records_by_partition[topic][partition] = buf

        kwargs = {}
        if self.config['api_version'] >= (2, 1):
            version = 7
        elif self.config['api_version'] >= (2, 0):
            version = 6
        elif self.config['api_version'] >= (1, 1):
            version = 5
        elif self.config['api_version'] >= (1, 0):
            version = 4
        elif self.config['api_version'] >= (0, 11):
            version = 3
            kwargs = dict(transactional_id=None)
        elif self.config['api_version'] >= (0, 10):
            version = 2
        elif self.config['api_version'] == (0, 9):
            version = 1
        else:
            version = 0
        return ProduceRequest[version](
            required_acks=acks,
            timeout=timeout,
            topics=[(topic, list(partition_info.items()))
                    for topic, partition_info
                    in six.iteritems(produce_records_by_partition)],
            **kwargs
        )

    def wakeup(self):
        """Wake up the selector associated with this send thread."""
        self._client.wakeup()

    def bootstrap_connected(self):
        return self._client.bootstrap_connected()


class SenderMetrics(object):

    def __init__(self, metrics, client, metadata):
        self.metrics = metrics
        self._client = client
        self._metadata = metadata

        sensor_name = 'batch-size'
        self.batch_size_sensor = self.metrics.sensor(sensor_name)
        self.add_metric('batch-size-avg', Avg(),
                        sensor_name=sensor_name,
                        description='The average number of bytes sent per partition per-request.')
        self.add_metric('batch-size-max', Max(),
                        sensor_name=sensor_name,
                        description='The max number of bytes sent per partition per-request.')

        sensor_name = 'compression-rate'
        self.compression_rate_sensor = self.metrics.sensor(sensor_name)
        self.add_metric('compression-rate-avg', Avg(),
                        sensor_name=sensor_name,
                        description='The average compression rate of record batches.')

        sensor_name = 'queue-time'
        self.queue_time_sensor = self.metrics.sensor(sensor_name)
        self.add_metric('record-queue-time-avg', Avg(),
                        sensor_name=sensor_name,
                        description='The average time in ms record batches spent in the record accumulator.')
        self.add_metric('record-queue-time-max', Max(),
                        sensor_name=sensor_name,
                        description='The maximum time in ms record batches spent in the record accumulator.')

        sensor_name = 'produce-throttle-time'
        self.produce_throttle_time_sensor = self.metrics.sensor(sensor_name)
        self.add_metric('produce-throttle-time-avg', Avg(),
                        sensor_name=sensor_name,
                        description='The average throttle time in ms')
        self.add_metric('produce-throttle-time-max', Max(),
                        sensor_name=sensor_name,
                        description='The maximum throttle time in ms')

        sensor_name = 'records-per-request'
        self.records_per_request_sensor = self.metrics.sensor(sensor_name)
        self.add_metric('record-send-rate', Rate(),
                        sensor_name=sensor_name,
                        description='The average number of records sent per second.')
        self.add_metric('records-per-request-avg', Avg(),
                        sensor_name=sensor_name,
                        description='The average number of records per request.')

        sensor_name = 'bytes'
        self.byte_rate_sensor = self.metrics.sensor(sensor_name)
        self.add_metric('byte-rate', Rate(),
                        sensor_name=sensor_name,
                        description='The average number of bytes sent per second.')

        sensor_name = 'record-retries'
        self.retry_sensor = self.metrics.sensor(sensor_name)
        self.add_metric('record-retry-rate', Rate(),
                        sensor_name=sensor_name,
                        description='The average per-second number of retried record sends')

        sensor_name = 'errors'
        self.error_sensor = self.metrics.sensor(sensor_name)
        self.add_metric('record-error-rate', Rate(),
                        sensor_name=sensor_name,
                        description='The average per-second number of record sends that resulted in errors')

        sensor_name = 'record-size-max'
        self.max_record_size_sensor = self.metrics.sensor(sensor_name)
        self.add_metric('record-size-max', Max(),
                        sensor_name=sensor_name,
                        description='The maximum record size across all batches')
        self.add_metric('record-size-avg', Avg(),
                        sensor_name=sensor_name,
                        description='The average maximum record size per batch')

        self.add_metric('requests-in-flight',
                        AnonMeasurable(lambda *_: self._client.in_flight_request_count()),
                        description='The current number of in-flight requests awaiting a response.')

        self.add_metric('metadata-age',
                        AnonMeasurable(lambda _, now: (now - self._metadata._last_successful_refresh_ms) / 1000),
                        description='The age in seconds of the current producer metadata being used.')

    def add_metric(self, metric_name, measurable, group_name='producer-metrics',
                   description=None, tags=None,
                   sensor_name=None):
        m = self.metrics
        metric = m.metric_name(metric_name, group_name, description, tags)
        if sensor_name:
            sensor = m.sensor(sensor_name)
            sensor.add(metric, measurable)
        else:
            m.add_metric(metric, measurable)

    def maybe_register_topic_metrics(self, topic):

        def sensor_name(name):
            return 'topic.{0}.{1}'.format(topic, name)

        # if one sensor of the metrics has been registered for the topic,
        # then all other sensors should have been registered; and vice versa
        if not self.metrics.get_sensor(sensor_name('records-per-batch')):

            self.add_metric('record-send-rate', Rate(),
                            sensor_name=sensor_name('records-per-batch'),
                            group_name='producer-topic-metrics.' + topic,
                            description= 'Records sent per second for topic ' + topic)

            self.add_metric('byte-rate', Rate(),
                            sensor_name=sensor_name('bytes'),
                            group_name='producer-topic-metrics.' + topic,
                            description='Bytes per second for topic ' + topic)

            self.add_metric('compression-rate', Avg(),
                            sensor_name=sensor_name('compression-rate'),
                            group_name='producer-topic-metrics.' + topic,
                            description='Average Compression ratio for topic ' + topic)

            self.add_metric('record-retry-rate', Rate(),
                            sensor_name=sensor_name('record-retries'),
                            group_name='producer-topic-metrics.' + topic,
                            description='Record retries per second for topic ' + topic)

            self.add_metric('record-error-rate', Rate(),
                            sensor_name=sensor_name('record-errors'),
                            group_name='producer-topic-metrics.' + topic,
                            description='Record errors per second for topic ' + topic)

    def update_produce_request_metrics(self, batches_map):
        for node_batch in batches_map.values():
            records = 0
            total_bytes = 0
            for batch in node_batch:
                # register all per-topic metrics at once
                topic = batch.topic_partition.topic
                self.maybe_register_topic_metrics(topic)

                # per-topic record send rate
                topic_records_count = self.metrics.get_sensor(
                    'topic.' + topic + '.records-per-batch')
                topic_records_count.record(batch.record_count)

                # per-topic bytes send rate
                topic_byte_rate = self.metrics.get_sensor(
                    'topic.' + topic + '.bytes')
                topic_byte_rate.record(batch.records.size_in_bytes())

                # per-topic compression rate
                topic_compression_rate = self.metrics.get_sensor(
                    'topic.' + topic + '.compression-rate')
                topic_compression_rate.record(batch.records.compression_rate())

                # global metrics
                self.batch_size_sensor.record(batch.records.size_in_bytes())
                if batch.drained:
                    self.queue_time_sensor.record(batch.drained - batch.created)
                self.compression_rate_sensor.record(batch.records.compression_rate())
                self.max_record_size_sensor.record(batch.max_record_size)
                records += batch.record_count
                total_bytes += batch.records.size_in_bytes()

            self.records_per_request_sensor.record(records)
            self.byte_rate_sensor.record(total_bytes)

    def record_retries(self, topic, count):
        self.retry_sensor.record(count)
        sensor = self.metrics.get_sensor('topic.' + topic + '.record-retries')
        if sensor:
            sensor.record(count)

    def record_errors(self, topic, count):
        self.error_sensor.record(count)
        sensor = self.metrics.get_sensor('topic.' + topic + '.record-errors')
        if sensor:
            sensor.record(count)

    def record_throttle_time(self, throttle_time_ms, node=None):
        self.produce_throttle_time_sensor.record(throttle_time_ms)
