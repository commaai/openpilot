from __future__ import absolute_import

import collections
import copy
import logging
import threading
import time

from kafka.vendor import six

from kafka import errors as Errors
from kafka.conn import collect_hosts
from kafka.future import Future
from kafka.structs import BrokerMetadata, PartitionMetadata, TopicPartition

log = logging.getLogger(__name__)


class ClusterMetadata(object):
    """
    A class to manage kafka cluster metadata.

    This class does not perform any IO. It simply updates internal state
    given API responses (MetadataResponse, GroupCoordinatorResponse).

    Keyword Arguments:
        retry_backoff_ms (int): Milliseconds to backoff when retrying on
            errors. Default: 100.
        metadata_max_age_ms (int): The period of time in milliseconds after
            which we force a refresh of metadata even if we haven't seen any
            partition leadership changes to proactively discover any new
            brokers or partitions. Default: 300000
        bootstrap_servers: 'host[:port]' string (or list of 'host[:port]'
            strings) that the client should contact to bootstrap initial
            cluster metadata. This does not have to be the full node list.
            It just needs to have at least one broker that will respond to a
            Metadata API Request. Default port is 9092. If no servers are
            specified, will default to localhost:9092.
    """
    DEFAULT_CONFIG = {
        'retry_backoff_ms': 100,
        'metadata_max_age_ms': 300000,
        'bootstrap_servers': [],
    }

    def __init__(self, **configs):
        self._brokers = {}  # node_id -> BrokerMetadata
        self._partitions = {}  # topic -> partition -> PartitionMetadata
        self._broker_partitions = collections.defaultdict(set)  # node_id -> {TopicPartition...}
        self._groups = {}  # group_name -> node_id
        self._last_refresh_ms = 0
        self._last_successful_refresh_ms = 0
        self._need_update = True
        self._future = None
        self._listeners = set()
        self._lock = threading.Lock()
        self.need_all_topic_metadata = False
        self.unauthorized_topics = set()
        self.internal_topics = set()
        self.controller = None

        self.config = copy.copy(self.DEFAULT_CONFIG)
        for key in self.config:
            if key in configs:
                self.config[key] = configs[key]

        self._bootstrap_brokers = self._generate_bootstrap_brokers()
        self._coordinator_brokers = {}

    def _generate_bootstrap_brokers(self):
        # collect_hosts does not perform DNS, so we should be fine to re-use
        bootstrap_hosts = collect_hosts(self.config['bootstrap_servers'])

        brokers = {}
        for i, (host, port, _) in enumerate(bootstrap_hosts):
            node_id = 'bootstrap-%s' % i
            brokers[node_id] = BrokerMetadata(node_id, host, port, None)
        return brokers

    def is_bootstrap(self, node_id):
        return node_id in self._bootstrap_brokers

    def brokers(self):
        """Get all BrokerMetadata

        Returns:
            set: {BrokerMetadata, ...}
        """
        return set(self._brokers.values()) or set(self._bootstrap_brokers.values())

    def broker_metadata(self, broker_id):
        """Get BrokerMetadata

        Arguments:
            broker_id (int): node_id for a broker to check

        Returns:
            BrokerMetadata or None if not found
        """
        return (
            self._brokers.get(broker_id) or
            self._bootstrap_brokers.get(broker_id) or
            self._coordinator_brokers.get(broker_id)
        )

    def partitions_for_topic(self, topic):
        """Return set of all partitions for topic (whether available or not)

        Arguments:
            topic (str): topic to check for partitions

        Returns:
            set: {partition (int), ...}
        """
        if topic not in self._partitions:
            return None
        return set(self._partitions[topic].keys())

    def available_partitions_for_topic(self, topic):
        """Return set of partitions with known leaders

        Arguments:
            topic (str): topic to check for partitions

        Returns:
            set: {partition (int), ...}
            None if topic not found.
        """
        if topic not in self._partitions:
            return None
        return set([partition for partition, metadata
                              in six.iteritems(self._partitions[topic])
                              if metadata.leader != -1])

    def leader_for_partition(self, partition):
        """Return node_id of leader, -1 unavailable, None if unknown."""
        if partition.topic not in self._partitions:
            return None
        elif partition.partition not in self._partitions[partition.topic]:
            return None
        return self._partitions[partition.topic][partition.partition].leader

    def partitions_for_broker(self, broker_id):
        """Return TopicPartitions for which the broker is a leader.

        Arguments:
            broker_id (int): node id for a broker

        Returns:
            set: {TopicPartition, ...}
            None if the broker either has no partitions or does not exist.
        """
        return self._broker_partitions.get(broker_id)

    def coordinator_for_group(self, group):
        """Return node_id of group coordinator.

        Arguments:
            group (str): name of consumer group

        Returns:
            int: node_id for group coordinator
            None if the group does not exist.
        """
        return self._groups.get(group)

    def ttl(self):
        """Milliseconds until metadata should be refreshed"""
        now = time.time() * 1000
        if self._need_update:
            ttl = 0
        else:
            metadata_age = now - self._last_successful_refresh_ms
            ttl = self.config['metadata_max_age_ms'] - metadata_age

        retry_age = now - self._last_refresh_ms
        next_retry = self.config['retry_backoff_ms'] - retry_age

        return max(ttl, next_retry, 0)

    def refresh_backoff(self):
        """Return milliseconds to wait before attempting to retry after failure"""
        return self.config['retry_backoff_ms']

    def request_update(self):
        """Flags metadata for update, return Future()

        Actual update must be handled separately. This method will only
        change the reported ttl()

        Returns:
            kafka.future.Future (value will be the cluster object after update)
        """
        with self._lock:
            self._need_update = True
            if not self._future or self._future.is_done:
                self._future = Future()
            return self._future

    def topics(self, exclude_internal_topics=True):
        """Get set of known topics.

        Arguments:
            exclude_internal_topics (bool): Whether records from internal topics
                (such as offsets) should be exposed to the consumer. If set to
                True the only way to receive records from an internal topic is
                subscribing to it. Default True

        Returns:
            set: {topic (str), ...}
        """
        topics = set(self._partitions.keys())
        if exclude_internal_topics:
            return topics - self.internal_topics
        else:
            return topics

    def failed_update(self, exception):
        """Update cluster state given a failed MetadataRequest."""
        f = None
        with self._lock:
            if self._future:
                f = self._future
                self._future = None
        if f:
            f.failure(exception)
        self._last_refresh_ms = time.time() * 1000

    def update_metadata(self, metadata):
        """Update cluster state given a MetadataResponse.

        Arguments:
            metadata (MetadataResponse): broker response to a metadata request

        Returns: None
        """
        # In the common case where we ask for a single topic and get back an
        # error, we should fail the future
        if len(metadata.topics) == 1 and metadata.topics[0][0] != 0:
            error_code, topic = metadata.topics[0][:2]
            error = Errors.for_code(error_code)(topic)
            return self.failed_update(error)

        if not metadata.brokers:
            log.warning("No broker metadata found in MetadataResponse -- ignoring.")
            return self.failed_update(Errors.MetadataEmptyBrokerList(metadata))

        _new_brokers = {}
        for broker in metadata.brokers:
            if metadata.API_VERSION == 0:
                node_id, host, port = broker
                rack = None
            else:
                node_id, host, port, rack = broker
            _new_brokers.update({
                node_id: BrokerMetadata(node_id, host, port, rack)
            })

        if metadata.API_VERSION == 0:
            _new_controller = None
        else:
            _new_controller = _new_brokers.get(metadata.controller_id)

        _new_partitions = {}
        _new_broker_partitions = collections.defaultdict(set)
        _new_unauthorized_topics = set()
        _new_internal_topics = set()

        for topic_data in metadata.topics:
            if metadata.API_VERSION == 0:
                error_code, topic, partitions = topic_data
                is_internal = False
            else:
                error_code, topic, is_internal, partitions = topic_data
            if is_internal:
                _new_internal_topics.add(topic)
            error_type = Errors.for_code(error_code)
            if error_type is Errors.NoError:
                _new_partitions[topic] = {}
                for p_error, partition, leader, replicas, isr in partitions:
                    _new_partitions[topic][partition] = PartitionMetadata(
                        topic=topic, partition=partition, leader=leader,
                        replicas=replicas, isr=isr, error=p_error)
                    if leader != -1:
                        _new_broker_partitions[leader].add(
                            TopicPartition(topic, partition))

            # Specific topic errors can be ignored if this is a full metadata fetch
            elif self.need_all_topic_metadata:
                continue

            elif error_type is Errors.LeaderNotAvailableError:
                log.warning("Topic %s is not available during auto-create"
                            " initialization", topic)
            elif error_type is Errors.UnknownTopicOrPartitionError:
                log.error("Topic %s not found in cluster metadata", topic)
            elif error_type is Errors.TopicAuthorizationFailedError:
                log.error("Topic %s is not authorized for this client", topic)
                _new_unauthorized_topics.add(topic)
            elif error_type is Errors.InvalidTopicError:
                log.error("'%s' is not a valid topic name", topic)
            else:
                log.error("Error fetching metadata for topic %s: %s",
                          topic, error_type)

        with self._lock:
            self._brokers = _new_brokers
            self.controller = _new_controller
            self._partitions = _new_partitions
            self._broker_partitions = _new_broker_partitions
            self.unauthorized_topics = _new_unauthorized_topics
            self.internal_topics = _new_internal_topics
            f = None
            if self._future:
                f = self._future
            self._future = None
            self._need_update = False

        now = time.time() * 1000
        self._last_refresh_ms = now
        self._last_successful_refresh_ms = now

        if f:
            f.success(self)
        log.debug("Updated cluster metadata to %s", self)

        for listener in self._listeners:
            listener(self)

        if self.need_all_topic_metadata:
            # the listener may change the interested topics,
            # which could cause another metadata refresh.
            # If we have already fetched all topics, however,
            # another fetch should be unnecessary.
            self._need_update = False

    def add_listener(self, listener):
        """Add a callback function to be called on each metadata update"""
        self._listeners.add(listener)

    def remove_listener(self, listener):
        """Remove a previously added listener callback"""
        self._listeners.remove(listener)

    def add_group_coordinator(self, group, response):
        """Update with metadata for a group coordinator

        Arguments:
            group (str): name of group from GroupCoordinatorRequest
            response (GroupCoordinatorResponse): broker response

        Returns:
            string: coordinator node_id if metadata is updated, None on error
        """
        log.debug("Updating coordinator for %s: %s", group, response)
        error_type = Errors.for_code(response.error_code)
        if error_type is not Errors.NoError:
            log.error("GroupCoordinatorResponse error: %s", error_type)
            self._groups[group] = -1
            return

        # Use a coordinator-specific node id so that group requests
        # get a dedicated connection
        node_id = 'coordinator-{}'.format(response.coordinator_id)
        coordinator = BrokerMetadata(
            node_id,
            response.host,
            response.port,
            None)

        log.info("Group coordinator for %s is %s", group, coordinator)
        self._coordinator_brokers[node_id] = coordinator
        self._groups[group] = node_id
        return node_id

    def with_partitions(self, partitions_to_add):
        """Returns a copy of cluster metadata with partitions added"""
        new_metadata = ClusterMetadata(**self.config)
        new_metadata._brokers = copy.deepcopy(self._brokers)
        new_metadata._partitions = copy.deepcopy(self._partitions)
        new_metadata._broker_partitions = copy.deepcopy(self._broker_partitions)
        new_metadata._groups = copy.deepcopy(self._groups)
        new_metadata.internal_topics = copy.deepcopy(self.internal_topics)
        new_metadata.unauthorized_topics = copy.deepcopy(self.unauthorized_topics)

        for partition in partitions_to_add:
            new_metadata._partitions[partition.topic][partition.partition] = partition

            if partition.leader is not None and partition.leader != -1:
                new_metadata._broker_partitions[partition.leader].add(
                    TopicPartition(partition.topic, partition.partition))

        return new_metadata

    def __str__(self):
        return 'ClusterMetadata(brokers: %d, topics: %d, groups: %d)' % \
               (len(self._brokers), len(self._partitions), len(self._groups))
