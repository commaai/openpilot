from __future__ import absolute_import

import atexit
import copy
import logging
import socket
import threading
import time
import weakref

from kafka.vendor import six

import kafka.errors as Errors
from kafka.client_async import KafkaClient, selectors
from kafka.codec import has_gzip, has_snappy, has_lz4, has_zstd
from kafka.metrics import MetricConfig, Metrics
from kafka.partitioner.default import DefaultPartitioner
from kafka.producer.future import FutureRecordMetadata, FutureProduceResult
from kafka.producer.record_accumulator import AtomicInteger, RecordAccumulator
from kafka.producer.sender import Sender
from kafka.record.default_records import DefaultRecordBatchBuilder
from kafka.record.legacy_records import LegacyRecordBatchBuilder
from kafka.serializer import Serializer
from kafka.structs import TopicPartition


log = logging.getLogger(__name__)
PRODUCER_CLIENT_ID_SEQUENCE = AtomicInteger()


class KafkaProducer(object):
    """A Kafka client that publishes records to the Kafka cluster.

    The producer is thread safe and sharing a single producer instance across
    threads will generally be faster than having multiple instances.

    The producer consists of a pool of buffer space that holds records that
    haven't yet been transmitted to the server as well as a background I/O
    thread that is responsible for turning these records into requests and
    transmitting them to the cluster.

    :meth:`~kafka.KafkaProducer.send` is asynchronous. When called it adds the
    record to a buffer of pending record sends and immediately returns. This
    allows the producer to batch together individual records for efficiency.

    The 'acks' config controls the criteria under which requests are considered
    complete. The "all" setting will result in blocking on the full commit of
    the record, the slowest but most durable setting.

    If the request fails, the producer can automatically retry, unless
    'retries' is configured to 0. Enabling retries also opens up the
    possibility of duplicates (see the documentation on message
    delivery semantics for details:
    https://kafka.apache.org/documentation.html#semantics
    ).

    The producer maintains buffers of unsent records for each partition. These
    buffers are of a size specified by the 'batch_size' config. Making this
    larger can result in more batching, but requires more memory (since we will
    generally have one of these buffers for each active partition).

    By default a buffer is available to send immediately even if there is
    additional unused space in the buffer. However if you want to reduce the
    number of requests you can set 'linger_ms' to something greater than 0.
    This will instruct the producer to wait up to that number of milliseconds
    before sending a request in hope that more records will arrive to fill up
    the same batch. This is analogous to Nagle's algorithm in TCP. Note that
    records that arrive close together in time will generally batch together
    even with linger_ms=0 so under heavy load batching will occur regardless of
    the linger configuration; however setting this to something larger than 0
    can lead to fewer, more efficient requests when not under maximal load at
    the cost of a small amount of latency.

    The buffer_memory controls the total amount of memory available to the
    producer for buffering. If records are sent faster than they can be
    transmitted to the server then this buffer space will be exhausted. When
    the buffer space is exhausted additional send calls will block.

    The key_serializer and value_serializer instruct how to turn the key and
    value objects the user provides into bytes.

    Keyword Arguments:
        bootstrap_servers: 'host[:port]' string (or list of 'host[:port]'
            strings) that the producer should contact to bootstrap initial
            cluster metadata. This does not have to be the full node list.
            It just needs to have at least one broker that will respond to a
            Metadata API Request. Default port is 9092. If no servers are
            specified, will default to localhost:9092.
        client_id (str): a name for this client. This string is passed in
            each request to servers and can be used to identify specific
            server-side log entries that correspond to this client.
            Default: 'kafka-python-producer-#' (appended with a unique number
            per instance)
        key_serializer (callable): used to convert user-supplied keys to bytes
            If not None, called as f(key), should return bytes. Default: None.
        value_serializer (callable): used to convert user-supplied message
            values to bytes. If not None, called as f(value), should return
            bytes. Default: None.
        acks (0, 1, 'all'): The number of acknowledgments the producer requires
            the leader to have received before considering a request complete.
            This controls the durability of records that are sent. The
            following settings are common:

            0: Producer will not wait for any acknowledgment from the server.
                The message will immediately be added to the socket
                buffer and considered sent. No guarantee can be made that the
                server has received the record in this case, and the retries
                configuration will not take effect (as the client won't
                generally know of any failures). The offset given back for each
                record will always be set to -1.
            1: Wait for leader to write the record to its local log only.
                Broker will respond without awaiting full acknowledgement from
                all followers. In this case should the leader fail immediately
                after acknowledging the record but before the followers have
                replicated it then the record will be lost.
            all: Wait for the full set of in-sync replicas to write the record.
                This guarantees that the record will not be lost as long as at
                least one in-sync replica remains alive. This is the strongest
                available guarantee.
            If unset, defaults to acks=1.
        compression_type (str): The compression type for all data generated by
            the producer. Valid values are 'gzip', 'snappy', 'lz4', 'zstd' or None.
            Compression is of full batches of data, so the efficacy of batching
            will also impact the compression ratio (more batching means better
            compression). Default: None.
        retries (int): Setting a value greater than zero will cause the client
            to resend any record whose send fails with a potentially transient
            error. Note that this retry is no different than if the client
            resent the record upon receiving the error. Allowing retries
            without setting max_in_flight_requests_per_connection to 1 will
            potentially change the ordering of records because if two batches
            are sent to a single partition, and the first fails and is retried
            but the second succeeds, then the records in the second batch may
            appear first.
            Default: 0.
        batch_size (int): Requests sent to brokers will contain multiple
            batches, one for each partition with data available to be sent.
            A small batch size will make batching less common and may reduce
            throughput (a batch size of zero will disable batching entirely).
            Default: 16384
        linger_ms (int): The producer groups together any records that arrive
            in between request transmissions into a single batched request.
            Normally this occurs only under load when records arrive faster
            than they can be sent out. However in some circumstances the client
            may want to reduce the number of requests even under moderate load.
            This setting accomplishes this by adding a small amount of
            artificial delay; that is, rather than immediately sending out a
            record the producer will wait for up to the given delay to allow
            other records to be sent so that the sends can be batched together.
            This can be thought of as analogous to Nagle's algorithm in TCP.
            This setting gives the upper bound on the delay for batching: once
            we get batch_size worth of records for a partition it will be sent
            immediately regardless of this setting, however if we have fewer
            than this many bytes accumulated for this partition we will
            'linger' for the specified time waiting for more records to show
            up. This setting defaults to 0 (i.e. no delay). Setting linger_ms=5
            would have the effect of reducing the number of requests sent but
            would add up to 5ms of latency to records sent in the absence of
            load. Default: 0.
        partitioner (callable): Callable used to determine which partition
            each message is assigned to. Called (after key serialization):
            partitioner(key_bytes, all_partitions, available_partitions).
            The default partitioner implementation hashes each non-None key
            using the same murmur2 algorithm as the java client so that
            messages with the same key are assigned to the same partition.
            When a key is None, the message is delivered to a random partition
            (filtered to partitions with available leaders only, if possible).
        buffer_memory (int): The total bytes of memory the producer should use
            to buffer records waiting to be sent to the server. If records are
            sent faster than they can be delivered to the server the producer
            will block up to max_block_ms, raising an exception on timeout.
            In the current implementation, this setting is an approximation.
            Default: 33554432 (32MB)
        connections_max_idle_ms: Close idle connections after the number of
            milliseconds specified by this config. The broker closes idle
            connections after connections.max.idle.ms, so this avoids hitting
            unexpected socket disconnected errors on the client.
            Default: 540000
        max_block_ms (int): Number of milliseconds to block during
            :meth:`~kafka.KafkaProducer.send` and
            :meth:`~kafka.KafkaProducer.partitions_for`. These methods can be
            blocked either because the buffer is full or metadata unavailable.
            Blocking in the user-supplied serializers or partitioner will not be
            counted against this timeout. Default: 60000.
        max_request_size (int): The maximum size of a request. This is also
            effectively a cap on the maximum record size. Note that the server
            has its own cap on record size which may be different from this.
            This setting will limit the number of record batches the producer
            will send in a single request to avoid sending huge requests.
            Default: 1048576.
        metadata_max_age_ms (int): The period of time in milliseconds after
            which we force a refresh of metadata even if we haven't seen any
            partition leadership changes to proactively discover any new
            brokers or partitions. Default: 300000
        retry_backoff_ms (int): Milliseconds to backoff when retrying on
            errors. Default: 100.
        request_timeout_ms (int): Client request timeout in milliseconds.
            Default: 30000.
        receive_buffer_bytes (int): The size of the TCP receive buffer
            (SO_RCVBUF) to use when reading data. Default: None (relies on
            system defaults). Java client defaults to 32768.
        send_buffer_bytes (int): The size of the TCP send buffer
            (SO_SNDBUF) to use when sending data. Default: None (relies on
            system defaults). Java client defaults to 131072.
        socket_options (list): List of tuple-arguments to socket.setsockopt
            to apply to broker connection sockets. Default:
            [(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)]
        reconnect_backoff_ms (int): The amount of time in milliseconds to
            wait before attempting to reconnect to a given host.
            Default: 50.
        reconnect_backoff_max_ms (int): The maximum amount of time in
            milliseconds to backoff/wait when reconnecting to a broker that has
            repeatedly failed to connect. If provided, the backoff per host
            will increase exponentially for each consecutive connection
            failure, up to this maximum. Once the maximum is reached,
            reconnection attempts will continue periodically with this fixed
            rate. To avoid connection storms, a randomization factor of 0.2
            will be applied to the backoff resulting in a random range between
            20% below and 20% above the computed value. Default: 1000.
        max_in_flight_requests_per_connection (int): Requests are pipelined
            to kafka brokers up to this number of maximum requests per
            broker connection. Note that if this setting is set to be greater
            than 1 and there are failed sends, there is a risk of message
            re-ordering due to retries (i.e., if retries are enabled).
            Default: 5.
        security_protocol (str): Protocol used to communicate with brokers.
            Valid values are: PLAINTEXT, SSL, SASL_PLAINTEXT, SASL_SSL.
            Default: PLAINTEXT.
        ssl_context (ssl.SSLContext): pre-configured SSLContext for wrapping
            socket connections. If provided, all other ssl_* configurations
            will be ignored. Default: None.
        ssl_check_hostname (bool): flag to configure whether ssl handshake
            should verify that the certificate matches the brokers hostname.
            default: true.
        ssl_cafile (str): optional filename of ca file to use in certificate
            veriication. default: none.
        ssl_certfile (str): optional filename of file in pem format containing
            the client certificate, as well as any ca certificates needed to
            establish the certificate's authenticity. default: none.
        ssl_keyfile (str): optional filename containing the client private key.
            default: none.
        ssl_password (str): optional password to be used when loading the
            certificate chain. default: none.
        ssl_crlfile (str): optional filename containing the CRL to check for
            certificate expiration. By default, no CRL check is done. When
            providing a file, only the leaf certificate will be checked against
            this CRL. The CRL can only be checked with Python 3.4+ or 2.7.9+.
            default: none.
        ssl_ciphers (str): optionally set the available ciphers for ssl
            connections. It should be a string in the OpenSSL cipher list
            format. If no cipher can be selected (because compile-time options
            or other configuration forbids use of all the specified ciphers),
            an ssl.SSLError will be raised. See ssl.SSLContext.set_ciphers
        api_version (tuple): Specify which Kafka API version to use. If set to
            None, the client will attempt to infer the broker version by probing
            various APIs. Example: (0, 10, 2). Default: None
        api_version_auto_timeout_ms (int): number of milliseconds to throw a
            timeout exception from the constructor when checking the broker
            api version. Only applies if api_version set to None.
        metric_reporters (list): A list of classes to use as metrics reporters.
            Implementing the AbstractMetricsReporter interface allows plugging
            in classes that will be notified of new metric creation. Default: []
        metrics_num_samples (int): The number of samples maintained to compute
            metrics. Default: 2
        metrics_sample_window_ms (int): The maximum age in milliseconds of
            samples used to compute metrics. Default: 30000
        selector (selectors.BaseSelector): Provide a specific selector
            implementation to use for I/O multiplexing.
            Default: selectors.DefaultSelector
        sasl_mechanism (str): Authentication mechanism when security_protocol
            is configured for SASL_PLAINTEXT or SASL_SSL. Valid values are:
            PLAIN, GSSAPI, OAUTHBEARER, SCRAM-SHA-256, SCRAM-SHA-512.
        sasl_plain_username (str): username for sasl PLAIN and SCRAM authentication.
            Required if sasl_mechanism is PLAIN or one of the SCRAM mechanisms.
        sasl_plain_password (str): password for sasl PLAIN and SCRAM authentication.
            Required if sasl_mechanism is PLAIN or one of the SCRAM mechanisms.
        sasl_kerberos_service_name (str): Service name to include in GSSAPI
            sasl mechanism handshake. Default: 'kafka'
        sasl_kerberos_domain_name (str): kerberos domain name to use in GSSAPI
            sasl mechanism handshake. Default: one of bootstrap servers
        sasl_oauth_token_provider (AbstractTokenProvider): OAuthBearer token provider
            instance. (See kafka.oauth.abstract). Default: None

    Note:
        Configuration parameters are described in more detail at
        https://kafka.apache.org/0100/configuration.html#producerconfigs
    """
    DEFAULT_CONFIG = {
        'bootstrap_servers': 'localhost',
        'client_id': None,
        'key_serializer': None,
        'value_serializer': None,
        'acks': 1,
        'bootstrap_topics_filter': set(),
        'compression_type': None,
        'retries': 0,
        'batch_size': 16384,
        'linger_ms': 0,
        'partitioner': DefaultPartitioner(),
        'buffer_memory': 33554432,
        'connections_max_idle_ms': 9 * 60 * 1000,
        'max_block_ms': 60000,
        'max_request_size': 1048576,
        'metadata_max_age_ms': 300000,
        'retry_backoff_ms': 100,
        'request_timeout_ms': 30000,
        'receive_buffer_bytes': None,
        'send_buffer_bytes': None,
        'socket_options': [(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)],
        'sock_chunk_bytes': 4096,  # undocumented experimental option
        'sock_chunk_buffer_count': 1000,  # undocumented experimental option
        'reconnect_backoff_ms': 50,
        'reconnect_backoff_max_ms': 1000,
        'max_in_flight_requests_per_connection': 5,
        'security_protocol': 'PLAINTEXT',
        'ssl_context': None,
        'ssl_check_hostname': True,
        'ssl_cafile': None,
        'ssl_certfile': None,
        'ssl_keyfile': None,
        'ssl_crlfile': None,
        'ssl_password': None,
        'ssl_ciphers': None,
        'api_version': None,
        'api_version_auto_timeout_ms': 2000,
        'metric_reporters': [],
        'metrics_num_samples': 2,
        'metrics_sample_window_ms': 30000,
        'selector': selectors.DefaultSelector,
        'sasl_mechanism': None,
        'sasl_plain_username': None,
        'sasl_plain_password': None,
        'sasl_kerberos_service_name': 'kafka',
        'sasl_kerberos_domain_name': None,
        'sasl_oauth_token_provider': None
    }

    _COMPRESSORS = {
        'gzip': (has_gzip, LegacyRecordBatchBuilder.CODEC_GZIP),
        'snappy': (has_snappy, LegacyRecordBatchBuilder.CODEC_SNAPPY),
        'lz4': (has_lz4, LegacyRecordBatchBuilder.CODEC_LZ4),
        'zstd': (has_zstd, DefaultRecordBatchBuilder.CODEC_ZSTD),
        None: (lambda: True, LegacyRecordBatchBuilder.CODEC_NONE),
    }

    def __init__(self, **configs):
        log.debug("Starting the Kafka producer")  # trace
        self.config = copy.copy(self.DEFAULT_CONFIG)
        for key in self.config:
            if key in configs:
                self.config[key] = configs.pop(key)

        # Only check for extra config keys in top-level class
        assert not configs, 'Unrecognized configs: %s' % (configs,)

        if self.config['client_id'] is None:
            self.config['client_id'] = 'kafka-python-producer-%s' % \
                                       (PRODUCER_CLIENT_ID_SEQUENCE.increment(),)

        if self.config['acks'] == 'all':
            self.config['acks'] = -1

        # api_version was previously a str. accept old format for now
        if isinstance(self.config['api_version'], str):
            deprecated = self.config['api_version']
            if deprecated == 'auto':
                self.config['api_version'] = None
            else:
                self.config['api_version'] = tuple(map(int, deprecated.split('.')))
            log.warning('use api_version=%s [tuple] -- "%s" as str is deprecated',
                        str(self.config['api_version']), deprecated)

        # Configure metrics
        metrics_tags = {'client-id': self.config['client_id']}
        metric_config = MetricConfig(samples=self.config['metrics_num_samples'],
                                     time_window_ms=self.config['metrics_sample_window_ms'],
                                     tags=metrics_tags)
        reporters = [reporter() for reporter in self.config['metric_reporters']]
        self._metrics = Metrics(metric_config, reporters)

        client = KafkaClient(metrics=self._metrics, metric_group_prefix='producer',
                             wakeup_timeout_ms=self.config['max_block_ms'],
                             **self.config)

        # Get auto-discovered version from client if necessary
        if self.config['api_version'] is None:
            self.config['api_version'] = client.config['api_version']

        if self.config['compression_type'] == 'lz4':
            assert self.config['api_version'] >= (0, 8, 2), 'LZ4 Requires >= Kafka 0.8.2 Brokers'

        if self.config['compression_type'] == 'zstd':
            assert self.config['api_version'] >= (2, 1, 0), 'Zstd Requires >= Kafka 2.1.0 Brokers'

        # Check compression_type for library support
        ct = self.config['compression_type']
        if ct not in self._COMPRESSORS:
            raise ValueError("Not supported codec: {}".format(ct))
        else:
            checker, compression_attrs = self._COMPRESSORS[ct]
            assert checker(), "Libraries for {} compression codec not found".format(ct)
            self.config['compression_attrs'] = compression_attrs

        message_version = self._max_usable_produce_magic()
        self._accumulator = RecordAccumulator(message_version=message_version, metrics=self._metrics, **self.config)
        self._metadata = client.cluster
        guarantee_message_order = bool(self.config['max_in_flight_requests_per_connection'] == 1)
        self._sender = Sender(client, self._metadata,
                              self._accumulator, self._metrics,
                              guarantee_message_order=guarantee_message_order,
                              **self.config)
        self._sender.daemon = True
        self._sender.start()
        self._closed = False

        self._cleanup = self._cleanup_factory()
        atexit.register(self._cleanup)
        log.debug("Kafka producer started")

    def bootstrap_connected(self):
        """Return True if the bootstrap is connected."""
        return self._sender.bootstrap_connected()

    def _cleanup_factory(self):
        """Build a cleanup clojure that doesn't increase our ref count"""
        _self = weakref.proxy(self)
        def wrapper():
            try:
                _self.close(timeout=0)
            except (ReferenceError, AttributeError):
                pass
        return wrapper

    def _unregister_cleanup(self):
        if getattr(self, '_cleanup', None):
            if hasattr(atexit, 'unregister'):
                atexit.unregister(self._cleanup)  # pylint: disable=no-member

            # py2 requires removing from private attribute...
            else:

                # ValueError on list.remove() if the exithandler no longer exists
                # but that is fine here
                try:
                    atexit._exithandlers.remove(  # pylint: disable=no-member
                        (self._cleanup, (), {}))
                except ValueError:
                    pass
        self._cleanup = None

    def __del__(self):
        # Disable logger during destruction to avoid touching dangling references
        class NullLogger(object):
            def __getattr__(self, name):
                return lambda *args: None

        global log
        log = NullLogger()

        self.close()

    def close(self, timeout=None):
        """Close this producer.

        Arguments:
            timeout (float, optional): timeout in seconds to wait for completion.
        """

        # drop our atexit handler now to avoid leaks
        self._unregister_cleanup()

        if not hasattr(self, '_closed') or self._closed:
            log.info('Kafka producer closed')
            return
        if timeout is None:
            # threading.TIMEOUT_MAX is available in Python3.3+
            timeout = getattr(threading, 'TIMEOUT_MAX', float('inf'))
        if getattr(threading, 'TIMEOUT_MAX', False):
            assert 0 <= timeout <= getattr(threading, 'TIMEOUT_MAX')
        else:
            assert timeout >= 0

        log.info("Closing the Kafka producer with %s secs timeout.", timeout)
        invoked_from_callback = bool(threading.current_thread() is self._sender)
        if timeout > 0:
            if invoked_from_callback:
                log.warning("Overriding close timeout %s secs to 0 in order to"
                            " prevent useless blocking due to self-join. This"
                            " means you have incorrectly invoked close with a"
                            " non-zero timeout from the producer call-back.",
                            timeout)
            else:
                # Try to close gracefully.
                if self._sender is not None:
                    self._sender.initiate_close()
                    self._sender.join(timeout)

        if self._sender is not None and self._sender.is_alive():
            log.info("Proceeding to force close the producer since pending"
                     " requests could not be completed within timeout %s.",
                     timeout)
            self._sender.force_close()

        self._metrics.close()
        try:
            self.config['key_serializer'].close()
        except AttributeError:
            pass
        try:
            self.config['value_serializer'].close()
        except AttributeError:
            pass
        self._closed = True
        log.debug("The Kafka producer has closed.")

    def partitions_for(self, topic):
        """Returns set of all known partitions for the topic."""
        max_wait = self.config['max_block_ms'] / 1000.0
        return self._wait_on_metadata(topic, max_wait)

    def _max_usable_produce_magic(self):
        if self.config['api_version'] >= (0, 11):
            return 2
        elif self.config['api_version'] >= (0, 10):
            return 1
        else:
            return 0

    def _estimate_size_in_bytes(self, key, value, headers=[]):
        magic = self._max_usable_produce_magic()
        if magic == 2:
            return DefaultRecordBatchBuilder.estimate_size_in_bytes(
                key, value, headers)
        else:
            return LegacyRecordBatchBuilder.estimate_size_in_bytes(
                magic, self.config['compression_type'], key, value)

    def send(self, topic, value=None, key=None, headers=None, partition=None, timestamp_ms=None):
        """Publish a message to a topic.

        Arguments:
            topic (str): topic where the message will be published
            value (optional): message value. Must be type bytes, or be
                serializable to bytes via configured value_serializer. If value
                is None, key is required and message acts as a 'delete'.
                See kafka compaction documentation for more details:
                https://kafka.apache.org/documentation.html#compaction
                (compaction requires kafka >= 0.8.1)
            partition (int, optional): optionally specify a partition. If not
                set, the partition will be selected using the configured
                'partitioner'.
            key (optional): a key to associate with the message. Can be used to
                determine which partition to send the message to. If partition
                is None (and producer's partitioner config is left as default),
                then messages with the same key will be delivered to the same
                partition (but if key is None, partition is chosen randomly).
                Must be type bytes, or be serializable to bytes via configured
                key_serializer.
            headers (optional): a list of header key value pairs. List items
                are tuples of str key and bytes value.
            timestamp_ms (int, optional): epoch milliseconds (from Jan 1 1970 UTC)
                to use as the message timestamp. Defaults to current time.

        Returns:
            FutureRecordMetadata: resolves to RecordMetadata

        Raises:
            KafkaTimeoutError: if unable to fetch topic metadata, or unable
                to obtain memory buffer prior to configured max_block_ms
        """
        assert value is not None or self.config['api_version'] >= (0, 8, 1), (
            'Null messages require kafka >= 0.8.1')
        assert not (value is None and key is None), 'Need at least one: key or value'
        key_bytes = value_bytes = None
        try:
            self._wait_on_metadata(topic, self.config['max_block_ms'] / 1000.0)

            key_bytes = self._serialize(
                self.config['key_serializer'],
                topic, key)
            value_bytes = self._serialize(
                self.config['value_serializer'],
                topic, value)
            assert type(key_bytes) in (bytes, bytearray, memoryview, type(None))
            assert type(value_bytes) in (bytes, bytearray, memoryview, type(None))

            partition = self._partition(topic, partition, key, value,
                                        key_bytes, value_bytes)

            if headers is None:
                headers = []
            assert type(headers) == list
            assert all(type(item) == tuple and len(item) == 2 and type(item[0]) == str and type(item[1]) == bytes for item in headers)

            message_size = self._estimate_size_in_bytes(key_bytes, value_bytes, headers)
            self._ensure_valid_record_size(message_size)

            tp = TopicPartition(topic, partition)
            log.debug("Sending (key=%r value=%r headers=%r) to %s", key, value, headers, tp)
            result = self._accumulator.append(tp, timestamp_ms,
                                              key_bytes, value_bytes, headers,
                                              self.config['max_block_ms'],
                                              estimated_size=message_size)
            future, batch_is_full, new_batch_created = result
            if batch_is_full or new_batch_created:
                log.debug("Waking up the sender since %s is either full or"
                          " getting a new batch", tp)
                self._sender.wakeup()

            return future
            # handling exceptions and record the errors;
            # for API exceptions return them in the future,
            # for other exceptions raise directly
        except Errors.BrokerResponseError as e:
            log.debug("Exception occurred during message send: %s", e)
            return FutureRecordMetadata(
                FutureProduceResult(TopicPartition(topic, partition)),
                -1, None, None,
                len(key_bytes) if key_bytes is not None else -1,
                len(value_bytes) if value_bytes is not None else -1,
                sum(len(h_key.encode("utf-8")) + len(h_value) for h_key, h_value in headers) if headers else -1,
            ).failure(e)

    def flush(self, timeout=None):
        """
        Invoking this method makes all buffered records immediately available
        to send (even if linger_ms is greater than 0) and blocks on the
        completion of the requests associated with these records. The
        post-condition of :meth:`~kafka.KafkaProducer.flush` is that any
        previously sent record will have completed
        (e.g. Future.is_done() == True). A request is considered completed when
        either it is successfully acknowledged according to the 'acks'
        configuration for the producer, or it results in an error.

        Other threads can continue sending messages while one thread is blocked
        waiting for a flush call to complete; however, no guarantee is made
        about the completion of messages sent after the flush call begins.

        Arguments:
            timeout (float, optional): timeout in seconds to wait for completion.

        Raises:
            KafkaTimeoutError: failure to flush buffered records within the
                provided timeout
        """
        log.debug("Flushing accumulated records in producer.")  # trace
        self._accumulator.begin_flush()
        self._sender.wakeup()
        self._accumulator.await_flush_completion(timeout=timeout)

    def _ensure_valid_record_size(self, size):
        """Validate that the record size isn't too large."""
        if size > self.config['max_request_size']:
            raise Errors.MessageSizeTooLargeError(
                "The message is %d bytes when serialized which is larger than"
                " the maximum request size you have configured with the"
                " max_request_size configuration" % (size,))
        if size > self.config['buffer_memory']:
            raise Errors.MessageSizeTooLargeError(
                "The message is %d bytes when serialized which is larger than"
                " the total memory buffer you have configured with the"
                " buffer_memory configuration." % (size,))

    def _wait_on_metadata(self, topic, max_wait):
        """
        Wait for cluster metadata including partitions for the given topic to
        be available.

        Arguments:
            topic (str): topic we want metadata for
            max_wait (float): maximum time in secs for waiting on the metadata

        Returns:
            set: partition ids for the topic

        Raises:
            KafkaTimeoutError: if partitions for topic were not obtained before
                specified max_wait timeout
        """
        # add topic to metadata topic list if it is not there already.
        self._sender.add_topic(topic)
        begin = time.time()
        elapsed = 0.0
        metadata_event = None
        while True:
            partitions = self._metadata.partitions_for_topic(topic)
            if partitions is not None:
                return partitions

            if not metadata_event:
                metadata_event = threading.Event()

            log.debug("Requesting metadata update for topic %s", topic)

            metadata_event.clear()
            future = self._metadata.request_update()
            future.add_both(lambda e, *args: e.set(), metadata_event)
            self._sender.wakeup()
            metadata_event.wait(max_wait - elapsed)
            elapsed = time.time() - begin
            if not metadata_event.is_set():
                raise Errors.KafkaTimeoutError(
                    "Failed to update metadata after %.1f secs." % (max_wait,))
            elif topic in self._metadata.unauthorized_topics:
                raise Errors.TopicAuthorizationFailedError(topic)
            else:
                log.debug("_wait_on_metadata woke after %s secs.", elapsed)

    def _serialize(self, f, topic, data):
        if not f:
            return data
        if isinstance(f, Serializer):
            return f.serialize(topic, data)
        return f(data)

    def _partition(self, topic, partition, key, value,
                   serialized_key, serialized_value):
        if partition is not None:
            assert partition >= 0
            assert partition in self._metadata.partitions_for_topic(topic), 'Unrecognized partition'
            return partition

        all_partitions = sorted(self._metadata.partitions_for_topic(topic))
        available = list(self._metadata.available_partitions_for_topic(topic))
        return self.config['partitioner'](serialized_key,
                                          all_partitions,
                                          available)

    def metrics(self, raw=False):
        """Get metrics on producer performance.

        This is ported from the Java Producer, for details see:
        https://kafka.apache.org/documentation/#producer_monitoring

        Warning:
            This is an unstable interface. It may change in future
            releases without warning.
        """
        if raw:
            return self._metrics.metrics.copy()

        metrics = {}
        for k, v in six.iteritems(self._metrics.metrics.copy()):
            if k.group not in metrics:
                metrics[k.group] = {}
            if k.name not in metrics[k.group]:
                metrics[k.group][k.name] = {}
            metrics[k.group][k.name] = v.value()
        return metrics
