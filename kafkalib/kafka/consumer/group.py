from __future__ import absolute_import, division

import copy
import logging
import socket
import time

from kafka.errors import KafkaConfigurationError, UnsupportedVersionError

from kafka.vendor import six

from kafka.client_async import KafkaClient, selectors
from kafka.consumer.fetcher import Fetcher
from kafka.consumer.subscription_state import SubscriptionState
from kafka.coordinator.consumer import ConsumerCoordinator
from kafka.coordinator.assignors.range import RangePartitionAssignor
from kafka.coordinator.assignors.roundrobin import RoundRobinPartitionAssignor
from kafka.metrics import MetricConfig, Metrics
from kafka.protocol.offset import OffsetResetStrategy
from kafka.structs import TopicPartition
from kafka.version import __version__

log = logging.getLogger(__name__)


class KafkaConsumer(six.Iterator):
    """Consume records from a Kafka cluster.

    The consumer will transparently handle the failure of servers in the Kafka
    cluster, and adapt as topic-partitions are created or migrate between
    brokers. It also interacts with the assigned kafka Group Coordinator node
    to allow multiple consumers to load balance consumption of topics (requires
    kafka >= 0.9.0.0).

    The consumer is not thread safe and should not be shared across threads.

    Arguments:
        *topics (str): optional list of topics to subscribe to. If not set,
            call :meth:`~kafka.KafkaConsumer.subscribe` or
            :meth:`~kafka.KafkaConsumer.assign` before consuming records.

    Keyword Arguments:
        bootstrap_servers: 'host[:port]' string (or list of 'host[:port]'
            strings) that the consumer should contact to bootstrap initial
            cluster metadata. This does not have to be the full node list.
            It just needs to have at least one broker that will respond to a
            Metadata API Request. Default port is 9092. If no servers are
            specified, will default to localhost:9092.
        client_id (str): A name for this client. This string is passed in
            each request to servers and can be used to identify specific
            server-side log entries that correspond to this client. Also
            submitted to GroupCoordinator for logging with respect to
            consumer group administration. Default: 'kafka-python-{version}'
        group_id (str or None): The name of the consumer group to join for dynamic
            partition assignment (if enabled), and to use for fetching and
            committing offsets. If None, auto-partition assignment (via
            group coordinator) and offset commits are disabled.
            Default: None
        key_deserializer (callable): Any callable that takes a
            raw message key and returns a deserialized key.
        value_deserializer (callable): Any callable that takes a
            raw message value and returns a deserialized value.
        fetch_min_bytes (int): Minimum amount of data the server should
            return for a fetch request, otherwise wait up to
            fetch_max_wait_ms for more data to accumulate. Default: 1.
        fetch_max_wait_ms (int): The maximum amount of time in milliseconds
            the server will block before answering the fetch request if
            there isn't sufficient data to immediately satisfy the
            requirement given by fetch_min_bytes. Default: 500.
        fetch_max_bytes (int): The maximum amount of data the server should
            return for a fetch request. This is not an absolute maximum, if the
            first message in the first non-empty partition of the fetch is
            larger than this value, the message will still be returned to
            ensure that the consumer can make progress. NOTE: consumer performs
            fetches to multiple brokers in parallel so memory usage will depend
            on the number of brokers containing partitions for the topic.
            Supported Kafka version >= 0.10.1.0. Default: 52428800 (50 MB).
        max_partition_fetch_bytes (int): The maximum amount of data
            per-partition the server will return. The maximum total memory
            used for a request = #partitions * max_partition_fetch_bytes.
            This size must be at least as large as the maximum message size
            the server allows or else it is possible for the producer to
            send messages larger than the consumer can fetch. If that
            happens, the consumer can get stuck trying to fetch a large
            message on a certain partition. Default: 1048576.
        request_timeout_ms (int): Client request timeout in milliseconds.
            Default: 305000.
        retry_backoff_ms (int): Milliseconds to backoff when retrying on
            errors. Default: 100.
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
            broker connection. Default: 5.
        auto_offset_reset (str): A policy for resetting offsets on
            OffsetOutOfRange errors: 'earliest' will move to the oldest
            available message, 'latest' will move to the most recent. Any
            other value will raise the exception. Default: 'latest'.
        enable_auto_commit (bool): If True , the consumer's offset will be
            periodically committed in the background. Default: True.
        auto_commit_interval_ms (int): Number of milliseconds between automatic
            offset commits, if enable_auto_commit is True. Default: 5000.
        default_offset_commit_callback (callable): Called as
            callback(offsets, response) response will be either an Exception
            or an OffsetCommitResponse struct. This callback can be used to
            trigger custom actions when a commit request completes.
        check_crcs (bool): Automatically check the CRC32 of the records
            consumed. This ensures no on-the-wire or on-disk corruption to
            the messages occurred. This check adds some overhead, so it may
            be disabled in cases seeking extreme performance. Default: True
        metadata_max_age_ms (int): The period of time in milliseconds after
            which we force a refresh of metadata, even if we haven't seen any
            partition leadership changes to proactively discover any new
            brokers or partitions. Default: 300000
        partition_assignment_strategy (list): List of objects to use to
            distribute partition ownership amongst consumer instances when
            group management is used.
            Default: [RangePartitionAssignor, RoundRobinPartitionAssignor]
        max_poll_records (int): The maximum number of records returned in a
            single call to :meth:`~kafka.KafkaConsumer.poll`. Default: 500
        max_poll_interval_ms (int): The maximum delay between invocations of
            :meth:`~kafka.KafkaConsumer.poll` when using consumer group
            management. This places an upper bound on the amount of time that
            the consumer can be idle before fetching more records. If
            :meth:`~kafka.KafkaConsumer.poll` is not called before expiration
            of this timeout, then the consumer is considered failed and the
            group will rebalance in order to reassign the partitions to another
            member. Default 300000
        session_timeout_ms (int): The timeout used to detect failures when
            using Kafka's group management facilities. The consumer sends
            periodic heartbeats to indicate its liveness to the broker. If
            no heartbeats are received by the broker before the expiration of
            this session timeout, then the broker will remove this consumer
            from the group and initiate a rebalance. Note that the value must
            be in the allowable range as configured in the broker configuration
            by group.min.session.timeout.ms and group.max.session.timeout.ms.
            Default: 10000
        heartbeat_interval_ms (int): The expected time in milliseconds
            between heartbeats to the consumer coordinator when using
            Kafka's group management facilities. Heartbeats are used to ensure
            that the consumer's session stays active and to facilitate
            rebalancing when new consumers join or leave the group. The
            value must be set lower than session_timeout_ms, but typically
            should be set no higher than 1/3 of that value. It can be
            adjusted even lower to control the expected time for normal
            rebalances. Default: 3000
        receive_buffer_bytes (int): The size of the TCP receive buffer
            (SO_RCVBUF) to use when reading data. Default: None (relies on
            system defaults). The java client defaults to 32768.
        send_buffer_bytes (int): The size of the TCP send buffer
            (SO_SNDBUF) to use when sending data. Default: None (relies on
            system defaults). The java client defaults to 131072.
        socket_options (list): List of tuple-arguments to socket.setsockopt
            to apply to broker connection sockets. Default:
            [(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)]
        consumer_timeout_ms (int): number of milliseconds to block during
            message iteration before raising StopIteration (i.e., ending the
            iterator). Default block forever [float('inf')].
        security_protocol (str): Protocol used to communicate with brokers.
            Valid values are: PLAINTEXT, SSL, SASL_PLAINTEXT, SASL_SSL.
            Default: PLAINTEXT.
        ssl_context (ssl.SSLContext): Pre-configured SSLContext for wrapping
            socket connections. If provided, all other ssl_* configurations
            will be ignored. Default: None.
        ssl_check_hostname (bool): Flag to configure whether ssl handshake
            should verify that the certificate matches the brokers hostname.
            Default: True.
        ssl_cafile (str): Optional filename of ca file to use in certificate
            verification. Default: None.
        ssl_certfile (str): Optional filename of file in pem format containing
            the client certificate, as well as any ca certificates needed to
            establish the certificate's authenticity. Default: None.
        ssl_keyfile (str): Optional filename containing the client private key.
            Default: None.
        ssl_password (str): Optional password to be used when loading the
            certificate chain. Default: None.
        ssl_crlfile (str): Optional filename containing the CRL to check for
            certificate expiration. By default, no CRL check is done. When
            providing a file, only the leaf certificate will be checked against
            this CRL. The CRL can only be checked with Python 3.4+ or 2.7.9+.
            Default: None.
        ssl_ciphers (str): optionally set the available ciphers for ssl
            connections. It should be a string in the OpenSSL cipher list
            format. If no cipher can be selected (because compile-time options
            or other configuration forbids use of all the specified ciphers),
            an ssl.SSLError will be raised. See ssl.SSLContext.set_ciphers
        api_version (tuple): Specify which Kafka API version to use. If set to
            None, the client will attempt to infer the broker version by probing
            various APIs. Different versions enable different functionality.

            Examples:
                (0, 9) enables full group coordination features with automatic
                    partition assignment and rebalancing,
                (0, 8, 2) enables kafka-storage offset commits with manual
                    partition assignment only,
                (0, 8, 1) enables zookeeper-storage offset commits with manual
                    partition assignment only,
                (0, 8, 0) enables basic functionality but requires manual
                    partition assignment and offset management.

            Default: None
        api_version_auto_timeout_ms (int): number of milliseconds to throw a
            timeout exception from the constructor when checking the broker
            api version. Only applies if api_version set to None.
        connections_max_idle_ms: Close idle connections after the number of
            milliseconds specified by this config. The broker closes idle
            connections after connections.max.idle.ms, so this avoids hitting
            unexpected socket disconnected errors on the client.
            Default: 540000
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
        exclude_internal_topics (bool): Whether records from internal topics
            (such as offsets) should be exposed to the consumer. If set to True
            the only way to receive records from an internal topic is
            subscribing to it. Requires 0.10+ Default: True
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
        https://kafka.apache.org/documentation/#consumerconfigs
    """
    DEFAULT_CONFIG = {
        'bootstrap_servers': 'localhost',
        'client_id': 'kafka-python-' + __version__,
        'group_id': None,
        'key_deserializer': None,
        'value_deserializer': None,
        'fetch_max_wait_ms': 500,
        'fetch_min_bytes': 1,
        'fetch_max_bytes': 52428800,
        'max_partition_fetch_bytes': 1 * 1024 * 1024,
        'request_timeout_ms': 305000, # chosen to be higher than the default of max_poll_interval_ms
        'retry_backoff_ms': 100,
        'reconnect_backoff_ms': 50,
        'reconnect_backoff_max_ms': 1000,
        'max_in_flight_requests_per_connection': 5,
        'auto_offset_reset': 'latest',
        'enable_auto_commit': True,
        'auto_commit_interval_ms': 5000,
        'default_offset_commit_callback': lambda offsets, response: True,
        'check_crcs': True,
        'metadata_max_age_ms': 5 * 60 * 1000,
        'partition_assignment_strategy': (RangePartitionAssignor, RoundRobinPartitionAssignor),
        'max_poll_records': 500,
        'max_poll_interval_ms': 300000,
        'session_timeout_ms': 10000,
        'heartbeat_interval_ms': 3000,
        'receive_buffer_bytes': None,
        'send_buffer_bytes': None,
        'socket_options': [(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)],
        'sock_chunk_bytes': 4096,  # undocumented experimental option
        'sock_chunk_buffer_count': 1000,  # undocumented experimental option
        'consumer_timeout_ms': float('inf'),
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
        'connections_max_idle_ms': 9 * 60 * 1000,
        'metric_reporters': [],
        'metrics_num_samples': 2,
        'metrics_sample_window_ms': 30000,
        'metric_group_prefix': 'consumer',
        'selector': selectors.DefaultSelector,
        'exclude_internal_topics': True,
        'sasl_mechanism': None,
        'sasl_plain_username': None,
        'sasl_plain_password': None,
        'sasl_kerberos_service_name': 'kafka',
        'sasl_kerberos_domain_name': None,
        'sasl_oauth_token_provider': None,
        'legacy_iterator': False, # enable to revert to < 1.4.7 iterator
    }
    DEFAULT_SESSION_TIMEOUT_MS_0_9 = 30000

    def __init__(self, *topics, **configs):
        # Only check for extra config keys in top-level class
        extra_configs = set(configs).difference(self.DEFAULT_CONFIG)
        if extra_configs:
            raise KafkaConfigurationError("Unrecognized configs: %s" % (extra_configs,))

        self.config = copy.copy(self.DEFAULT_CONFIG)
        self.config.update(configs)

        deprecated = {'smallest': 'earliest', 'largest': 'latest'}
        if self.config['auto_offset_reset'] in deprecated:
            new_config = deprecated[self.config['auto_offset_reset']]
            log.warning('use auto_offset_reset=%s (%s is deprecated)',
                        new_config, self.config['auto_offset_reset'])
            self.config['auto_offset_reset'] = new_config

        connections_max_idle_ms = self.config['connections_max_idle_ms']
        request_timeout_ms = self.config['request_timeout_ms']
        fetch_max_wait_ms = self.config['fetch_max_wait_ms']
        if not (fetch_max_wait_ms < request_timeout_ms < connections_max_idle_ms):
            raise KafkaConfigurationError(
                "connections_max_idle_ms ({}) must be larger than "
                "request_timeout_ms ({}) which must be larger than "
                "fetch_max_wait_ms ({})."
                .format(connections_max_idle_ms, request_timeout_ms, fetch_max_wait_ms))

        metrics_tags = {'client-id': self.config['client_id']}
        metric_config = MetricConfig(samples=self.config['metrics_num_samples'],
                                     time_window_ms=self.config['metrics_sample_window_ms'],
                                     tags=metrics_tags)
        reporters = [reporter() for reporter in self.config['metric_reporters']]
        self._metrics = Metrics(metric_config, reporters)
        # TODO _metrics likely needs to be passed to KafkaClient, etc.

        # api_version was previously a str. Accept old format for now
        if isinstance(self.config['api_version'], str):
            str_version = self.config['api_version']
            if str_version == 'auto':
                self.config['api_version'] = None
            else:
                self.config['api_version'] = tuple(map(int, str_version.split('.')))
            log.warning('use api_version=%s [tuple] -- "%s" as str is deprecated',
                        str(self.config['api_version']), str_version)

        self._client = KafkaClient(metrics=self._metrics, **self.config)

        # Get auto-discovered version from client if necessary
        if self.config['api_version'] is None:
            self.config['api_version'] = self._client.config['api_version']

        # Coordinator configurations are different for older brokers
        # max_poll_interval_ms is not supported directly -- it must the be
        # the same as session_timeout_ms. If the user provides one of them,
        # use it for both. Otherwise use the old default of 30secs
        if self.config['api_version'] < (0, 10, 1):
            if 'session_timeout_ms' not in configs:
                if 'max_poll_interval_ms' in configs:
                    self.config['session_timeout_ms'] = configs['max_poll_interval_ms']
                else:
                    self.config['session_timeout_ms'] = self.DEFAULT_SESSION_TIMEOUT_MS_0_9
            if 'max_poll_interval_ms' not in configs:
                self.config['max_poll_interval_ms'] = self.config['session_timeout_ms']

        if self.config['group_id'] is not None:
            if self.config['request_timeout_ms'] <= self.config['session_timeout_ms']:
                raise KafkaConfigurationError(
                    "Request timeout (%s) must be larger than session timeout (%s)" %
                    (self.config['request_timeout_ms'], self.config['session_timeout_ms']))

        self._subscription = SubscriptionState(self.config['auto_offset_reset'])
        self._fetcher = Fetcher(
            self._client, self._subscription, self._metrics, **self.config)
        self._coordinator = ConsumerCoordinator(
            self._client, self._subscription, self._metrics,
            assignors=self.config['partition_assignment_strategy'],
            **self.config)
        self._closed = False
        self._iterator = None
        self._consumer_timeout = float('inf')

        if topics:
            self._subscription.subscribe(topics=topics)
            self._client.set_topics(topics)

    def bootstrap_connected(self):
        """Return True if the bootstrap is connected."""
        return self._client.bootstrap_connected()

    def assign(self, partitions):
        """Manually assign a list of TopicPartitions to this consumer.

        Arguments:
            partitions (list of TopicPartition): Assignment for this instance.

        Raises:
            IllegalStateError: If consumer has already called
            :meth:`~kafka.KafkaConsumer.subscribe`.

        Warning:
            It is not possible to use both manual partition assignment with
            :meth:`~kafka.KafkaConsumer.assign` and group assignment with
            :meth:`~kafka.KafkaConsumer.subscribe`.

        Note:
            This interface does not support incremental assignment and will
            replace the previous assignment (if there was one).

        Note:
            Manual topic assignment through this method does not use the
            consumer's group management functionality. As such, there will be
            no rebalance operation triggered when group membership or cluster
            and topic metadata change.
        """
        self._subscription.assign_from_user(partitions)
        self._client.set_topics([tp.topic for tp in partitions])

    def assignment(self):
        """Get the TopicPartitions currently assigned to this consumer.

        If partitions were directly assigned using
        :meth:`~kafka.KafkaConsumer.assign`, then this will simply return the
        same partitions that were previously assigned.  If topics were
        subscribed using :meth:`~kafka.KafkaConsumer.subscribe`, then this will
        give the set of topic partitions currently assigned to the consumer
        (which may be None if the assignment hasn't happened yet, or if the
        partitions are in the process of being reassigned).

        Returns:
            set: {TopicPartition, ...}
        """
        return self._subscription.assigned_partitions()

    def close(self, autocommit=True):
        """Close the consumer, waiting indefinitely for any needed cleanup.

        Keyword Arguments:
            autocommit (bool): If auto-commit is configured for this consumer,
                this optional flag causes the consumer to attempt to commit any
                pending consumed offsets prior to close. Default: True
        """
        if self._closed:
            return
        log.debug("Closing the KafkaConsumer.")
        self._closed = True
        self._coordinator.close(autocommit=autocommit)
        self._metrics.close()
        self._client.close()
        try:
            self.config['key_deserializer'].close()
        except AttributeError:
            pass
        try:
            self.config['value_deserializer'].close()
        except AttributeError:
            pass
        log.debug("The KafkaConsumer has closed.")

    def commit_async(self, offsets=None, callback=None):
        """Commit offsets to kafka asynchronously, optionally firing callback.

        This commits offsets only to Kafka. The offsets committed using this API
        will be used on the first fetch after every rebalance and also on
        startup. As such, if you need to store offsets in anything other than
        Kafka, this API should not be used. To avoid re-processing the last
        message read if a consumer is restarted, the committed offset should be
        the next message your application should consume, i.e.: last_offset + 1.

        This is an asynchronous call and will not block. Any errors encountered
        are either passed to the callback (if provided) or discarded.

        Arguments:
            offsets (dict, optional): {TopicPartition: OffsetAndMetadata} dict
                to commit with the configured group_id. Defaults to currently
                consumed offsets for all subscribed partitions.
            callback (callable, optional): Called as callback(offsets, response)
                with response as either an Exception or an OffsetCommitResponse
                struct. This callback can be used to trigger custom actions when
                a commit request completes.

        Returns:
            kafka.future.Future
        """
        assert self.config['api_version'] >= (0, 8, 1), 'Requires >= Kafka 0.8.1'
        assert self.config['group_id'] is not None, 'Requires group_id'
        if offsets is None:
            offsets = self._subscription.all_consumed_offsets()
        log.debug("Committing offsets: %s", offsets)
        future = self._coordinator.commit_offsets_async(
            offsets, callback=callback)
        return future

    def commit(self, offsets=None):
        """Commit offsets to kafka, blocking until success or error.

        This commits offsets only to Kafka. The offsets committed using this API
        will be used on the first fetch after every rebalance and also on
        startup. As such, if you need to store offsets in anything other than
        Kafka, this API should not be used. To avoid re-processing the last
        message read if a consumer is restarted, the committed offset should be
        the next message your application should consume, i.e.: last_offset + 1.

        Blocks until either the commit succeeds or an unrecoverable error is
        encountered (in which case it is thrown to the caller).

        Currently only supports kafka-topic offset storage (not zookeeper).

        Arguments:
            offsets (dict, optional): {TopicPartition: OffsetAndMetadata} dict
                to commit with the configured group_id. Defaults to currently
                consumed offsets for all subscribed partitions.
        """
        assert self.config['api_version'] >= (0, 8, 1), 'Requires >= Kafka 0.8.1'
        assert self.config['group_id'] is not None, 'Requires group_id'
        if offsets is None:
            offsets = self._subscription.all_consumed_offsets()
        self._coordinator.commit_offsets_sync(offsets)

    def committed(self, partition, metadata=False):
        """Get the last committed offset for the given partition.

        This offset will be used as the position for the consumer
        in the event of a failure.

        This call may block to do a remote call if the partition in question
        isn't assigned to this consumer or if the consumer hasn't yet
        initialized its cache of committed offsets.

        Arguments:
            partition (TopicPartition): The partition to check.
            metadata (bool, optional): If True, return OffsetAndMetadata struct
                instead of offset int. Default: False.

        Returns:
            The last committed offset (int or OffsetAndMetadata), or None if there was no prior commit.
        """
        assert self.config['api_version'] >= (0, 8, 1), 'Requires >= Kafka 0.8.1'
        assert self.config['group_id'] is not None, 'Requires group_id'
        if not isinstance(partition, TopicPartition):
            raise TypeError('partition must be a TopicPartition namedtuple')
        if self._subscription.is_assigned(partition):
            committed = self._subscription.assignment[partition].committed
            if committed is None:
                self._coordinator.refresh_committed_offsets_if_needed()
                committed = self._subscription.assignment[partition].committed
        else:
            commit_map = self._coordinator.fetch_committed_offsets([partition])
            if partition in commit_map:
                committed = commit_map[partition]
            else:
                committed = None

        if committed is not None:
            if metadata:
                return committed
            else:
                return committed.offset

    def _fetch_all_topic_metadata(self):
        """A blocking call that fetches topic metadata for all topics in the
        cluster that the user is authorized to view.
        """
        cluster = self._client.cluster
        if self._client._metadata_refresh_in_progress and self._client._topics:
            future = cluster.request_update()
            self._client.poll(future=future)
        stash = cluster.need_all_topic_metadata
        cluster.need_all_topic_metadata = True
        future = cluster.request_update()
        self._client.poll(future=future)
        cluster.need_all_topic_metadata = stash

    def topics(self):
        """Get all topics the user is authorized to view.
        This will always issue a remote call to the cluster to fetch the latest
        information.

        Returns:
            set: topics
        """
        self._fetch_all_topic_metadata()
        return self._client.cluster.topics()

    def partitions_for_topic(self, topic):
        """This method first checks the local metadata cache for information
        about the topic. If the topic is not found (either because the topic
        does not exist, the user is not authorized to view the topic, or the
        metadata cache is not populated), then it will issue a metadata update
        call to the cluster.

        Arguments:
            topic (str): Topic to check.

        Returns:
            set: Partition ids
        """
        cluster = self._client.cluster
        partitions = cluster.partitions_for_topic(topic)
        if partitions is None:
            self._fetch_all_topic_metadata()
            partitions = cluster.partitions_for_topic(topic)
        return partitions

    def poll(self, timeout_ms=0, max_records=None, update_offsets=True):
        """Fetch data from assigned topics / partitions.

        Records are fetched and returned in batches by topic-partition.
        On each poll, consumer will try to use the last consumed offset as the
        starting offset and fetch sequentially. The last consumed offset can be
        manually set through :meth:`~kafka.KafkaConsumer.seek` or automatically
        set as the last committed offset for the subscribed list of partitions.

        Incompatible with iterator interface -- use one or the other, not both.

        Arguments:
            timeout_ms (int, optional): Milliseconds spent waiting in poll if
                data is not available in the buffer. If 0, returns immediately
                with any records that are available currently in the buffer,
                else returns empty. Must not be negative. Default: 0
            max_records (int, optional): The maximum number of records returned
                in a single call to :meth:`~kafka.KafkaConsumer.poll`.
                Default: Inherit value from max_poll_records.

        Returns:
            dict: Topic to list of records since the last fetch for the
                subscribed list of topics and partitions.
        """
        # Note: update_offsets is an internal-use only argument. It is used to
        # support the python iterator interface, and which wraps consumer.poll()
        # and requires that the partition offsets tracked by the fetcher are not
        # updated until the iterator returns each record to the user. As such,
        # the argument is not documented and should not be relied on by library
        # users to not break in the future.
        assert timeout_ms >= 0, 'Timeout must not be negative'
        if max_records is None:
            max_records = self.config['max_poll_records']
        assert isinstance(max_records, int), 'max_records must be an integer'
        assert max_records > 0, 'max_records must be positive'
        assert not self._closed, 'KafkaConsumer is closed'

        # Poll for new data until the timeout expires
        start = time.time()
        remaining = timeout_ms
        while True:
            records = self._poll_once(remaining, max_records, update_offsets=update_offsets)
            if records:
                return records

            elapsed_ms = (time.time() - start) * 1000
            remaining = timeout_ms - elapsed_ms

            if remaining <= 0:
                return {}

    def _poll_once(self, timeout_ms, max_records, update_offsets=True):
        """Do one round of polling. In addition to checking for new data, this does
        any needed heart-beating, auto-commits, and offset updates.

        Arguments:
            timeout_ms (int): The maximum time in milliseconds to block.

        Returns:
            dict: Map of topic to list of records (may be empty).
        """
        self._coordinator.poll()

        # Fetch positions if we have partitions we're subscribed to that we
        # don't know the offset for
        if not self._subscription.has_all_fetch_positions():
            self._update_fetch_positions(self._subscription.missing_fetch_positions())

        # If data is available already, e.g. from a previous network client
        # poll() call to commit, then just return it immediately
        records, partial = self._fetcher.fetched_records(max_records, update_offsets=update_offsets)
        if records:
            # Before returning the fetched records, we can send off the
            # next round of fetches and avoid block waiting for their
            # responses to enable pipelining while the user is handling the
            # fetched records.
            if not partial:
                futures = self._fetcher.send_fetches()
                if len(futures):
                    self._client.poll(timeout_ms=0)
            return records

        # Send any new fetches (won't resend pending fetches)
        futures = self._fetcher.send_fetches()
        if len(futures):
            self._client.poll(timeout_ms=0)

        timeout_ms = min(timeout_ms, self._coordinator.time_to_next_poll() * 1000)
        self._client.poll(timeout_ms=timeout_ms)
        # after the long poll, we should check whether the group needs to rebalance
        # prior to returning data so that the group can stabilize faster
        if self._coordinator.need_rejoin():
            return {}

        records, _ = self._fetcher.fetched_records(max_records, update_offsets=update_offsets)
        return records

    def position(self, partition):
        """Get the offset of the next record that will be fetched

        Arguments:
            partition (TopicPartition): Partition to check

        Returns:
            int: Offset
        """
        if not isinstance(partition, TopicPartition):
            raise TypeError('partition must be a TopicPartition namedtuple')
        assert self._subscription.is_assigned(partition), 'Partition is not assigned'
        offset = self._subscription.assignment[partition].position
        if offset is None:
            self._update_fetch_positions([partition])
            offset = self._subscription.assignment[partition].position
        return offset

    def highwater(self, partition):
        """Last known highwater offset for a partition.

        A highwater offset is the offset that will be assigned to the next
        message that is produced. It may be useful for calculating lag, by
        comparing with the reported position. Note that both position and
        highwater refer to the *next* offset -- i.e., highwater offset is
        one greater than the newest available message.

        Highwater offsets are returned in FetchResponse messages, so will
        not be available if no FetchRequests have been sent for this partition
        yet.

        Arguments:
            partition (TopicPartition): Partition to check

        Returns:
            int or None: Offset if available
        """
        if not isinstance(partition, TopicPartition):
            raise TypeError('partition must be a TopicPartition namedtuple')
        assert self._subscription.is_assigned(partition), 'Partition is not assigned'
        return self._subscription.assignment[partition].highwater

    def pause(self, *partitions):
        """Suspend fetching from the requested partitions.

        Future calls to :meth:`~kafka.KafkaConsumer.poll` will not return any
        records from these partitions until they have been resumed using
        :meth:`~kafka.KafkaConsumer.resume`.

        Note: This method does not affect partition subscription. In particular,
        it does not cause a group rebalance when automatic assignment is used.

        Arguments:
            *partitions (TopicPartition): Partitions to pause.
        """
        if not all([isinstance(p, TopicPartition) for p in partitions]):
            raise TypeError('partitions must be TopicPartition namedtuples')
        for partition in partitions:
            log.debug("Pausing partition %s", partition)
            self._subscription.pause(partition)
        # Because the iterator checks is_fetchable() on each iteration
        # we expect pauses to get handled automatically and therefore
        # we do not need to reset the full iterator (forcing a full refetch)

    def paused(self):
        """Get the partitions that were previously paused using
        :meth:`~kafka.KafkaConsumer.pause`.

        Returns:
            set: {partition (TopicPartition), ...}
        """
        return self._subscription.paused_partitions()

    def resume(self, *partitions):
        """Resume fetching from the specified (paused) partitions.

        Arguments:
            *partitions (TopicPartition): Partitions to resume.
        """
        if not all([isinstance(p, TopicPartition) for p in partitions]):
            raise TypeError('partitions must be TopicPartition namedtuples')
        for partition in partitions:
            log.debug("Resuming partition %s", partition)
            self._subscription.resume(partition)

    def seek(self, partition, offset):
        """Manually specify the fetch offset for a TopicPartition.

        Overrides the fetch offsets that the consumer will use on the next
        :meth:`~kafka.KafkaConsumer.poll`. If this API is invoked for the same
        partition more than once, the latest offset will be used on the next
        :meth:`~kafka.KafkaConsumer.poll`.

        Note: You may lose data if this API is arbitrarily used in the middle of
        consumption to reset the fetch offsets.

        Arguments:
            partition (TopicPartition): Partition for seek operation
            offset (int): Message offset in partition

        Raises:
            AssertionError: If offset is not an int >= 0; or if partition is not
                currently assigned.
        """
        if not isinstance(partition, TopicPartition):
            raise TypeError('partition must be a TopicPartition namedtuple')
        assert isinstance(offset, int) and offset >= 0, 'Offset must be >= 0'
        assert partition in self._subscription.assigned_partitions(), 'Unassigned partition'
        log.debug("Seeking to offset %s for partition %s", offset, partition)
        self._subscription.assignment[partition].seek(offset)
        if not self.config['legacy_iterator']:
            self._iterator = None

    def seek_to_beginning(self, *partitions):
        """Seek to the oldest available offset for partitions.

        Arguments:
            *partitions: Optionally provide specific TopicPartitions, otherwise
                default to all assigned partitions.

        Raises:
            AssertionError: If any partition is not currently assigned, or if
                no partitions are assigned.
        """
        if not all([isinstance(p, TopicPartition) for p in partitions]):
            raise TypeError('partitions must be TopicPartition namedtuples')
        if not partitions:
            partitions = self._subscription.assigned_partitions()
            assert partitions, 'No partitions are currently assigned'
        else:
            for p in partitions:
                assert p in self._subscription.assigned_partitions(), 'Unassigned partition'

        for tp in partitions:
            log.debug("Seeking to beginning of partition %s", tp)
            self._subscription.need_offset_reset(tp, OffsetResetStrategy.EARLIEST)
        if not self.config['legacy_iterator']:
            self._iterator = None

    def seek_to_end(self, *partitions):
        """Seek to the most recent available offset for partitions.

        Arguments:
            *partitions: Optionally provide specific TopicPartitions, otherwise
                default to all assigned partitions.

        Raises:
            AssertionError: If any partition is not currently assigned, or if
                no partitions are assigned.
        """
        if not all([isinstance(p, TopicPartition) for p in partitions]):
            raise TypeError('partitions must be TopicPartition namedtuples')
        if not partitions:
            partitions = self._subscription.assigned_partitions()
            assert partitions, 'No partitions are currently assigned'
        else:
            for p in partitions:
                assert p in self._subscription.assigned_partitions(), 'Unassigned partition'

        for tp in partitions:
            log.debug("Seeking to end of partition %s", tp)
            self._subscription.need_offset_reset(tp, OffsetResetStrategy.LATEST)
        if not self.config['legacy_iterator']:
            self._iterator = None

    def subscribe(self, topics=(), pattern=None, listener=None):
        """Subscribe to a list of topics, or a topic regex pattern.

        Partitions will be dynamically assigned via a group coordinator.
        Topic subscriptions are not incremental: this list will replace the
        current assignment (if there is one).

        This method is incompatible with :meth:`~kafka.KafkaConsumer.assign`.

        Arguments:
            topics (list): List of topics for subscription.
            pattern (str): Pattern to match available topics. You must provide
                either topics or pattern, but not both.
            listener (ConsumerRebalanceListener): Optionally include listener
                callback, which will be called before and after each rebalance
                operation.

                As part of group management, the consumer will keep track of the
                list of consumers that belong to a particular group and will
                trigger a rebalance operation if one of the following events
                trigger:

                * Number of partitions change for any of the subscribed topics
                * Topic is created or deleted
                * An existing member of the consumer group dies
                * A new member is added to the consumer group

                When any of these events are triggered, the provided listener
                will be invoked first to indicate that the consumer's assignment
                has been revoked, and then again when the new assignment has
                been received. Note that this listener will immediately override
                any listener set in a previous call to subscribe. It is
                guaranteed, however, that the partitions revoked/assigned
                through this interface are from topics subscribed in this call.

        Raises:
            IllegalStateError: If called after previously calling
                :meth:`~kafka.KafkaConsumer.assign`.
            AssertionError: If neither topics or pattern is provided.
            TypeError: If listener is not a ConsumerRebalanceListener.
        """
        # SubscriptionState handles error checking
        self._subscription.subscribe(topics=topics,
                                     pattern=pattern,
                                     listener=listener)

        # Regex will need all topic metadata
        if pattern is not None:
            self._client.cluster.need_all_topic_metadata = True
            self._client.set_topics([])
            self._client.cluster.request_update()
            log.debug("Subscribed to topic pattern: %s", pattern)
        else:
            self._client.cluster.need_all_topic_metadata = False
            self._client.set_topics(self._subscription.group_subscription())
            log.debug("Subscribed to topic(s): %s", topics)

    def subscription(self):
        """Get the current topic subscription.

        Returns:
            set: {topic, ...}
        """
        if self._subscription.subscription is None:
            return None
        return self._subscription.subscription.copy()

    def unsubscribe(self):
        """Unsubscribe from all topics and clear all assigned partitions."""
        self._subscription.unsubscribe()
        self._coordinator.close()
        self._client.cluster.need_all_topic_metadata = False
        self._client.set_topics([])
        log.debug("Unsubscribed all topics or patterns and assigned partitions")
        if not self.config['legacy_iterator']:
            self._iterator = None

    def metrics(self, raw=False):
        """Get metrics on consumer performance.

        This is ported from the Java Consumer, for details see:
        https://kafka.apache.org/documentation/#consumer_monitoring

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

    def offsets_for_times(self, timestamps):
        """Look up the offsets for the given partitions by timestamp. The
        returned offset for each partition is the earliest offset whose
        timestamp is greater than or equal to the given timestamp in the
        corresponding partition.

        This is a blocking call. The consumer does not have to be assigned the
        partitions.

        If the message format version in a partition is before 0.10.0, i.e.
        the messages do not have timestamps, ``None`` will be returned for that
        partition. ``None`` will also be returned for the partition if there
        are no messages in it.

        Note:
            This method may block indefinitely if the partition does not exist.

        Arguments:
            timestamps (dict): ``{TopicPartition: int}`` mapping from partition
                to the timestamp to look up. Unit should be milliseconds since
                beginning of the epoch (midnight Jan 1, 1970 (UTC))

        Returns:
            ``{TopicPartition: OffsetAndTimestamp}``: mapping from partition
            to the timestamp and offset of the first message with timestamp
            greater than or equal to the target timestamp.

        Raises:
            ValueError: If the target timestamp is negative
            UnsupportedVersionError: If the broker does not support looking
                up the offsets by timestamp.
            KafkaTimeoutError: If fetch failed in request_timeout_ms
        """
        if self.config['api_version'] <= (0, 10, 0):
            raise UnsupportedVersionError(
                "offsets_for_times API not supported for cluster version {}"
                .format(self.config['api_version']))
        for tp, ts in six.iteritems(timestamps):
            timestamps[tp] = int(ts)
            if ts < 0:
                raise ValueError(
                    "The target time for partition {} is {}. The target time "
                    "cannot be negative.".format(tp, ts))
        return self._fetcher.get_offsets_by_times(
            timestamps, self.config['request_timeout_ms'])

    def beginning_offsets(self, partitions):
        """Get the first offset for the given partitions.

        This method does not change the current consumer position of the
        partitions.

        Note:
            This method may block indefinitely if the partition does not exist.

        Arguments:
            partitions (list): List of TopicPartition instances to fetch
                offsets for.

        Returns:
            ``{TopicPartition: int}``: The earliest available offsets for the
            given partitions.

        Raises:
            UnsupportedVersionError: If the broker does not support looking
                up the offsets by timestamp.
            KafkaTimeoutError: If fetch failed in request_timeout_ms.
        """
        offsets = self._fetcher.beginning_offsets(
            partitions, self.config['request_timeout_ms'])
        return offsets

    def end_offsets(self, partitions):
        """Get the last offset for the given partitions. The last offset of a
        partition is the offset of the upcoming message, i.e. the offset of the
        last available message + 1.

        This method does not change the current consumer position of the
        partitions.

        Note:
            This method may block indefinitely if the partition does not exist.

        Arguments:
            partitions (list): List of TopicPartition instances to fetch
                offsets for.

        Returns:
            ``{TopicPartition: int}``: The end offsets for the given partitions.

        Raises:
            UnsupportedVersionError: If the broker does not support looking
                up the offsets by timestamp.
            KafkaTimeoutError: If fetch failed in request_timeout_ms
        """
        offsets = self._fetcher.end_offsets(
            partitions, self.config['request_timeout_ms'])
        return offsets

    def _use_consumer_group(self):
        """Return True iff this consumer can/should join a broker-coordinated group."""
        if self.config['api_version'] < (0, 9):
            return False
        elif self.config['group_id'] is None:
            return False
        elif not self._subscription.partitions_auto_assigned():
            return False
        return True

    def _update_fetch_positions(self, partitions):
        """Set the fetch position to the committed position (if there is one)
        or reset it using the offset reset policy the user has configured.

        Arguments:
            partitions (List[TopicPartition]): The partitions that need
                updating fetch positions.

        Raises:
            NoOffsetForPartitionError: If no offset is stored for a given
                partition and no offset reset policy is defined.
        """
        # Lookup any positions for partitions which are awaiting reset (which may be the
        # case if the user called :meth:`seek_to_beginning` or :meth:`seek_to_end`. We do
        # this check first to avoid an unnecessary lookup of committed offsets (which
        # typically occurs when the user is manually assigning partitions and managing
        # their own offsets).
        self._fetcher.reset_offsets_if_needed(partitions)

        if not self._subscription.has_all_fetch_positions():
            # if we still don't have offsets for all partitions, then we should either seek
            # to the last committed position or reset using the auto reset policy
            if (self.config['api_version'] >= (0, 8, 1) and
                self.config['group_id'] is not None):
                # first refresh commits for all assigned partitions
                self._coordinator.refresh_committed_offsets_if_needed()

            # Then, do any offset lookups in case some positions are not known
            self._fetcher.update_fetch_positions(partitions)

    def _message_generator_v2(self):
        timeout_ms = 1000 * (self._consumer_timeout - time.time())
        record_map = self.poll(timeout_ms=timeout_ms, update_offsets=False)
        for tp, records in six.iteritems(record_map):
            # Generators are stateful, and it is possible that the tp / records
            # here may become stale during iteration -- i.e., we seek to a
            # different offset, pause consumption, or lose assignment.
            for record in records:
                # is_fetchable(tp) should handle assignment changes and offset
                # resets; for all other changes (e.g., seeks) we'll rely on the
                # outer function destroying the existing iterator/generator
                # via self._iterator = None
                if not self._subscription.is_fetchable(tp):
                    log.debug("Not returning fetched records for partition %s"
                              " since it is no longer fetchable", tp)
                    break
                self._subscription.assignment[tp].position = record.offset + 1
                yield record

    def _message_generator(self):
        assert self.assignment() or self.subscription() is not None, 'No topic subscription or manual partition assignment'
        while time.time() < self._consumer_timeout:

            self._coordinator.poll()

            # Fetch offsets for any subscribed partitions that we arent tracking yet
            if not self._subscription.has_all_fetch_positions():
                partitions = self._subscription.missing_fetch_positions()
                self._update_fetch_positions(partitions)

            poll_ms = min((1000 * (self._consumer_timeout - time.time())), self.config['retry_backoff_ms'])
            self._client.poll(timeout_ms=poll_ms)

            # after the long poll, we should check whether the group needs to rebalance
            # prior to returning data so that the group can stabilize faster
            if self._coordinator.need_rejoin():
                continue

            # We need to make sure we at least keep up with scheduled tasks,
            # like heartbeats, auto-commits, and metadata refreshes
            timeout_at = self._next_timeout()

            # Short-circuit the fetch iterator if we are already timed out
            # to avoid any unintentional interaction with fetcher setup
            if time.time() > timeout_at:
                continue

            for msg in self._fetcher:
                yield msg
                if time.time() > timeout_at:
                    log.debug("internal iterator timeout - breaking for poll")
                    break
                self._client.poll(timeout_ms=0)

            # An else block on a for loop only executes if there was no break
            # so this should only be called on a StopIteration from the fetcher
            # We assume that it is safe to init_fetches when fetcher is done
            # i.e., there are no more records stored internally
            else:
                self._fetcher.send_fetches()

    def _next_timeout(self):
        timeout = min(self._consumer_timeout,
                      self._client.cluster.ttl() / 1000.0 + time.time(),
                      self._coordinator.time_to_next_poll() + time.time())
        return timeout

    def __iter__(self):  # pylint: disable=non-iterator-returned
        return self

    def __next__(self):
        if self._closed:
            raise StopIteration('KafkaConsumer closed')
        # Now that the heartbeat thread runs in the background
        # there should be no reason to maintain a separate iterator
        # but we'll keep it available for a few releases just in case
        if self.config['legacy_iterator']:
            return self.next_v1()
        else:
            return self.next_v2()

    def next_v2(self):
        self._set_consumer_timeout()
        while time.time() < self._consumer_timeout:
            if not self._iterator:
                self._iterator = self._message_generator_v2()
            try:
                return next(self._iterator)
            except StopIteration:
                self._iterator = None
        raise StopIteration()

    def next_v1(self):
        if not self._iterator:
            self._iterator = self._message_generator()

        self._set_consumer_timeout()
        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator = None
            raise

    def _set_consumer_timeout(self):
        # consumer_timeout_ms can be used to stop iteration early
        if self.config['consumer_timeout_ms'] >= 0:
            self._consumer_timeout = time.time() + (
                self.config['consumer_timeout_ms'] / 1000.0)
