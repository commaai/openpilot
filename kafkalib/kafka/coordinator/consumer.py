from __future__ import absolute_import, division

import collections
import copy
import functools
import logging
import time

from kafka.vendor import six

from kafka.coordinator.base import BaseCoordinator, Generation
from kafka.coordinator.assignors.range import RangePartitionAssignor
from kafka.coordinator.assignors.roundrobin import RoundRobinPartitionAssignor
from kafka.coordinator.assignors.sticky.sticky_assignor import StickyPartitionAssignor
from kafka.coordinator.protocol import ConsumerProtocol
import kafka.errors as Errors
from kafka.future import Future
from kafka.metrics import AnonMeasurable
from kafka.metrics.stats import Avg, Count, Max, Rate
from kafka.protocol.commit import OffsetCommitRequest, OffsetFetchRequest
from kafka.structs import OffsetAndMetadata, TopicPartition
from kafka.util import WeakMethod


log = logging.getLogger(__name__)


class ConsumerCoordinator(BaseCoordinator):
    """This class manages the coordination process with the consumer coordinator."""
    DEFAULT_CONFIG = {
        'group_id': 'kafka-python-default-group',
        'enable_auto_commit': True,
        'auto_commit_interval_ms': 5000,
        'default_offset_commit_callback': None,
        'assignors': (RangePartitionAssignor, RoundRobinPartitionAssignor, StickyPartitionAssignor),
        'session_timeout_ms': 10000,
        'heartbeat_interval_ms': 3000,
        'max_poll_interval_ms': 300000,
        'retry_backoff_ms': 100,
        'api_version': (0, 10, 1),
        'exclude_internal_topics': True,
        'metric_group_prefix': 'consumer'
    }

    def __init__(self, client, subscription, metrics, **configs):
        """Initialize the coordination manager.

        Keyword Arguments:
            group_id (str): name of the consumer group to join for dynamic
                partition assignment (if enabled), and to use for fetching and
                committing offsets. Default: 'kafka-python-default-group'
            enable_auto_commit (bool): If true the consumer's offset will be
                periodically committed in the background. Default: True.
            auto_commit_interval_ms (int): milliseconds between automatic
                offset commits, if enable_auto_commit is True. Default: 5000.
            default_offset_commit_callback (callable): called as
                callback(offsets, exception) response will be either an Exception
                or None. This callback can be used to trigger custom actions when
                a commit request completes.
            assignors (list): List of objects to use to distribute partition
                ownership amongst consumer instances when group management is
                used. Default: [RangePartitionAssignor, RoundRobinPartitionAssignor]
            heartbeat_interval_ms (int): The expected time in milliseconds
                between heartbeats to the consumer coordinator when using
                Kafka's group management feature. Heartbeats are used to ensure
                that the consumer's session stays active and to facilitate
                rebalancing when new consumers join or leave the group. The
                value must be set lower than session_timeout_ms, but typically
                should be set no higher than 1/3 of that value. It can be
                adjusted even lower to control the expected time for normal
                rebalances. Default: 3000
            session_timeout_ms (int): The timeout used to detect failures when
                using Kafka's group management facilities. Default: 30000
            retry_backoff_ms (int): Milliseconds to backoff when retrying on
                errors. Default: 100.
            exclude_internal_topics (bool): Whether records from internal topics
                (such as offsets) should be exposed to the consumer. If set to
                True the only way to receive records from an internal topic is
                subscribing to it. Requires 0.10+. Default: True
        """
        super(ConsumerCoordinator, self).__init__(client, metrics, **configs)

        self.config = copy.copy(self.DEFAULT_CONFIG)
        for key in self.config:
            if key in configs:
                self.config[key] = configs[key]

        self._subscription = subscription
        self._is_leader = False
        self._joined_subscription = set()
        self._metadata_snapshot = self._build_metadata_snapshot(subscription, client.cluster)
        self._assignment_snapshot = None
        self._cluster = client.cluster
        self.auto_commit_interval = self.config['auto_commit_interval_ms'] / 1000
        self.next_auto_commit_deadline = None
        self.completed_offset_commits = collections.deque()

        if self.config['default_offset_commit_callback'] is None:
            self.config['default_offset_commit_callback'] = self._default_offset_commit_callback

        if self.config['group_id'] is not None:
            if self.config['api_version'] >= (0, 9):
                if not self.config['assignors']:
                    raise Errors.KafkaConfigurationError('Coordinator requires assignors')
            if self.config['api_version'] < (0, 10, 1):
                if self.config['max_poll_interval_ms'] != self.config['session_timeout_ms']:
                    raise Errors.KafkaConfigurationError("Broker version %s does not support "
                                                         "different values for max_poll_interval_ms "
                                                         "and session_timeout_ms")

        if self.config['enable_auto_commit']:
            if self.config['api_version'] < (0, 8, 1):
                log.warning('Broker version (%s) does not support offset'
                            ' commits; disabling auto-commit.',
                            self.config['api_version'])
                self.config['enable_auto_commit'] = False
            elif self.config['group_id'] is None:
                log.warning('group_id is None: disabling auto-commit.')
                self.config['enable_auto_commit'] = False
            else:
                self.next_auto_commit_deadline = time.time() + self.auto_commit_interval

        self.consumer_sensors = ConsumerCoordinatorMetrics(
            metrics, self.config['metric_group_prefix'], self._subscription)

        self._cluster.request_update()
        self._cluster.add_listener(WeakMethod(self._handle_metadata_update))

    def __del__(self):
        if hasattr(self, '_cluster') and self._cluster:
            self._cluster.remove_listener(WeakMethod(self._handle_metadata_update))
        super(ConsumerCoordinator, self).__del__()

    def protocol_type(self):
        return ConsumerProtocol.PROTOCOL_TYPE

    def group_protocols(self):
        """Returns list of preferred (protocols, metadata)"""
        if self._subscription.subscription is None:
            raise Errors.IllegalStateError('Consumer has not subscribed to topics')
        # dpkp note: I really dislike this.
        # why? because we are using this strange method group_protocols,
        # which is seemingly innocuous, to set internal state (_joined_subscription)
        # that is later used to check whether metadata has changed since we joined a group
        # but there is no guarantee that this method, group_protocols, will get called
        # in the correct sequence or that it will only be called when we want it to be.
        # So this really should be moved elsewhere, but I don't have the energy to
        # work that out right now. If you read this at some later date after the mutable
        # state has bitten you... I'm sorry! It mimics the java client, and that's the
        # best I've got for now.
        self._joined_subscription = set(self._subscription.subscription)
        metadata_list = []
        for assignor in self.config['assignors']:
            metadata = assignor.metadata(self._joined_subscription)
            group_protocol = (assignor.name, metadata)
            metadata_list.append(group_protocol)
        return metadata_list

    def _handle_metadata_update(self, cluster):
        # if we encounter any unauthorized topics, raise an exception
        if cluster.unauthorized_topics:
            raise Errors.TopicAuthorizationFailedError(cluster.unauthorized_topics)

        if self._subscription.subscribed_pattern:
            topics = []
            for topic in cluster.topics(self.config['exclude_internal_topics']):
                if self._subscription.subscribed_pattern.match(topic):
                    topics.append(topic)

            if set(topics) != self._subscription.subscription:
                self._subscription.change_subscription(topics)
                self._client.set_topics(self._subscription.group_subscription())

        # check if there are any changes to the metadata which should trigger
        # a rebalance
        if self._subscription.partitions_auto_assigned():
            metadata_snapshot = self._build_metadata_snapshot(self._subscription, cluster)
            if self._metadata_snapshot != metadata_snapshot:
                self._metadata_snapshot = metadata_snapshot

                # If we haven't got group coordinator support,
                # just assign all partitions locally
                if self._auto_assign_all_partitions():
                    self._subscription.assign_from_subscribed([
                        TopicPartition(topic, partition)
                        for topic in self._subscription.subscription
                        for partition in self._metadata_snapshot[topic]
                    ])

    def _auto_assign_all_partitions(self):
        # For users that use "subscribe" without group support,
        # we will simply assign all partitions to this consumer
        if self.config['api_version'] < (0, 9):
            return True
        elif self.config['group_id'] is None:
            return True
        else:
            return False

    def _build_metadata_snapshot(self, subscription, cluster):
        metadata_snapshot = {}
        for topic in subscription.group_subscription():
            partitions = cluster.partitions_for_topic(topic) or []
            metadata_snapshot[topic] = set(partitions)
        return metadata_snapshot

    def _lookup_assignor(self, name):
        for assignor in self.config['assignors']:
            if assignor.name == name:
                return assignor
        return None

    def _on_join_complete(self, generation, member_id, protocol,
                          member_assignment_bytes):
        # only the leader is responsible for monitoring for metadata changes
        # (i.e. partition changes)
        if not self._is_leader:
            self._assignment_snapshot = None

        assignor = self._lookup_assignor(protocol)
        assert assignor, 'Coordinator selected invalid assignment protocol: %s' % (protocol,)

        assignment = ConsumerProtocol.ASSIGNMENT.decode(member_assignment_bytes)

        # set the flag to refresh last committed offsets
        self._subscription.needs_fetch_committed_offsets = True

        # update partition assignment
        try:
            self._subscription.assign_from_subscribed(assignment.partitions())
        except ValueError as e:
            log.warning("%s. Probably due to a deleted topic. Requesting Re-join" % e)
            self.request_rejoin()

        # give the assignor a chance to update internal state
        # based on the received assignment
        assignor.on_assignment(assignment)
        if assignor.name == 'sticky':
            assignor.on_generation_assignment(generation)

        # reschedule the auto commit starting from now
        self.next_auto_commit_deadline = time.time() + self.auto_commit_interval

        assigned = set(self._subscription.assigned_partitions())
        log.info("Setting newly assigned partitions %s for group %s",
                 assigned, self.group_id)

        # execute the user's callback after rebalance
        if self._subscription.listener:
            try:
                self._subscription.listener.on_partitions_assigned(assigned)
            except Exception:
                log.exception("User provided listener %s for group %s"
                              " failed on partition assignment: %s",
                              self._subscription.listener, self.group_id,
                              assigned)

    def poll(self):
        """
        Poll for coordinator events. Only applicable if group_id is set, and
        broker version supports GroupCoordinators. This ensures that the
        coordinator is known, and if using automatic partition assignment,
        ensures that the consumer has joined the group. This also handles
        periodic offset commits if they are enabled.
        """
        if self.group_id is None:
            return

        self._invoke_completed_offset_commit_callbacks()
        self.ensure_coordinator_ready()

        if self.config['api_version'] >= (0, 9) and self._subscription.partitions_auto_assigned():
            if self.need_rejoin():
                # due to a race condition between the initial metadata fetch and the
                # initial rebalance, we need to ensure that the metadata is fresh
                # before joining initially, and then request the metadata update. If
                # metadata update arrives while the rebalance is still pending (for
                # example, when the join group is still inflight), then we will lose
                # track of the fact that we need to rebalance again to reflect the
                # change to the topic subscription. Without ensuring that the
                # metadata is fresh, any metadata update that changes the topic
                # subscriptions and arrives while a rebalance is in progress will
                # essentially be ignored. See KAFKA-3949 for the complete
                # description of the problem.
                if self._subscription.subscribed_pattern:
                    metadata_update = self._client.cluster.request_update()
                    self._client.poll(future=metadata_update)

                self.ensure_active_group()

            self.poll_heartbeat()

        self._maybe_auto_commit_offsets_async()

    def time_to_next_poll(self):
        """Return seconds (float) remaining until :meth:`.poll` should be called again"""
        if not self.config['enable_auto_commit']:
            return self.time_to_next_heartbeat()

        if time.time() > self.next_auto_commit_deadline:
            return 0

        return min(self.next_auto_commit_deadline - time.time(),
                   self.time_to_next_heartbeat())

    def _perform_assignment(self, leader_id, assignment_strategy, members):
        assignor = self._lookup_assignor(assignment_strategy)
        assert assignor, 'Invalid assignment protocol: %s' % (assignment_strategy,)
        member_metadata = {}
        all_subscribed_topics = set()
        for member_id, metadata_bytes in members:
            metadata = ConsumerProtocol.METADATA.decode(metadata_bytes)
            member_metadata[member_id] = metadata
            all_subscribed_topics.update(metadata.subscription) # pylint: disable-msg=no-member

        # the leader will begin watching for changes to any of the topics
        # the group is interested in, which ensures that all metadata changes
        # will eventually be seen
        # Because assignment typically happens within response callbacks,
        # we cannot block on metadata updates here (no recursion into poll())
        self._subscription.group_subscribe(all_subscribed_topics)
        self._client.set_topics(self._subscription.group_subscription())

        # keep track of the metadata used for assignment so that we can check
        # after rebalance completion whether anything has changed
        self._cluster.request_update()
        self._is_leader = True
        self._assignment_snapshot = self._metadata_snapshot

        log.debug("Performing assignment for group %s using strategy %s"
                  " with subscriptions %s", self.group_id, assignor.name,
                  member_metadata)

        assignments = assignor.assign(self._cluster, member_metadata)

        log.debug("Finished assignment for group %s: %s", self.group_id, assignments)

        group_assignment = {}
        for member_id, assignment in six.iteritems(assignments):
            group_assignment[member_id] = assignment
        return group_assignment

    def _on_join_prepare(self, generation, member_id):
        # commit offsets prior to rebalance if auto-commit enabled
        self._maybe_auto_commit_offsets_sync()

        # execute the user's callback before rebalance
        log.info("Revoking previously assigned partitions %s for group %s",
                 self._subscription.assigned_partitions(), self.group_id)
        if self._subscription.listener:
            try:
                revoked = set(self._subscription.assigned_partitions())
                self._subscription.listener.on_partitions_revoked(revoked)
            except Exception:
                log.exception("User provided subscription listener %s"
                              " for group %s failed on_partitions_revoked",
                              self._subscription.listener, self.group_id)

        self._is_leader = False
        self._subscription.reset_group_subscription()

    def need_rejoin(self):
        """Check whether the group should be rejoined

        Returns:
            bool: True if consumer should rejoin group, False otherwise
        """
        if not self._subscription.partitions_auto_assigned():
            return False

        if self._auto_assign_all_partitions():
            return False

        # we need to rejoin if we performed the assignment and metadata has changed
        if (self._assignment_snapshot is not None
            and self._assignment_snapshot != self._metadata_snapshot):
            return True

        # we need to join if our subscription has changed since the last join
        if (self._joined_subscription is not None
            and self._joined_subscription != self._subscription.subscription):
            return True

        return super(ConsumerCoordinator, self).need_rejoin()

    def refresh_committed_offsets_if_needed(self):
        """Fetch committed offsets for assigned partitions."""
        if self._subscription.needs_fetch_committed_offsets:
            offsets = self.fetch_committed_offsets(self._subscription.assigned_partitions())
            for partition, offset in six.iteritems(offsets):
                # verify assignment is still active
                if self._subscription.is_assigned(partition):
                    self._subscription.assignment[partition].committed = offset
            self._subscription.needs_fetch_committed_offsets = False

    def fetch_committed_offsets(self, partitions):
        """Fetch the current committed offsets for specified partitions

        Arguments:
            partitions (list of TopicPartition): partitions to fetch

        Returns:
            dict: {TopicPartition: OffsetAndMetadata}
        """
        if not partitions:
            return {}

        while True:
            self.ensure_coordinator_ready()

            # contact coordinator to fetch committed offsets
            future = self._send_offset_fetch_request(partitions)
            self._client.poll(future=future)

            if future.succeeded():
                return future.value

            if not future.retriable():
                raise future.exception # pylint: disable-msg=raising-bad-type

            time.sleep(self.config['retry_backoff_ms'] / 1000)

    def close(self, autocommit=True):
        """Close the coordinator, leave the current group,
        and reset local generation / member_id.

        Keyword Arguments:
            autocommit (bool): If auto-commit is configured for this consumer,
                this optional flag causes the consumer to attempt to commit any
                pending consumed offsets prior to close. Default: True
        """
        try:
            if autocommit:
                self._maybe_auto_commit_offsets_sync()
        finally:
            super(ConsumerCoordinator, self).close()

    def _invoke_completed_offset_commit_callbacks(self):
        while self.completed_offset_commits:
            callback, offsets, exception = self.completed_offset_commits.popleft()
            callback(offsets, exception)

    def commit_offsets_async(self, offsets, callback=None):
        """Commit specific offsets asynchronously.

        Arguments:
            offsets (dict {TopicPartition: OffsetAndMetadata}): what to commit
            callback (callable, optional): called as callback(offsets, response)
                response will be either an Exception or a OffsetCommitResponse
                struct. This callback can be used to trigger custom actions when
                a commit request completes.

        Returns:
            kafka.future.Future
        """
        self._invoke_completed_offset_commit_callbacks()
        if not self.coordinator_unknown():
            future = self._do_commit_offsets_async(offsets, callback)
        else:
            # we don't know the current coordinator, so try to find it and then
            # send the commit or fail (we don't want recursive retries which can
            # cause offset commits to arrive out of order). Note that there may
            # be multiple offset commits chained to the same coordinator lookup
            # request. This is fine because the listeners will be invoked in the
            # same order that they were added. Note also that BaseCoordinator
            # prevents multiple concurrent coordinator lookup requests.
            future = self.lookup_coordinator()
            future.add_callback(lambda r: functools.partial(self._do_commit_offsets_async, offsets, callback)())
            if callback:
                future.add_errback(lambda e: self.completed_offset_commits.appendleft((callback, offsets, e)))

        # ensure the commit has a chance to be transmitted (without blocking on
        # its completion). Note that commits are treated as heartbeats by the
        # coordinator, so there is no need to explicitly allow heartbeats
        # through delayed task execution.
        self._client.poll(timeout_ms=0) # no wakeup if we add that feature

        return future

    def _do_commit_offsets_async(self, offsets, callback=None):
        assert self.config['api_version'] >= (0, 8, 1), 'Unsupported Broker API'
        assert all(map(lambda k: isinstance(k, TopicPartition), offsets))
        assert all(map(lambda v: isinstance(v, OffsetAndMetadata),
                       offsets.values()))
        if callback is None:
            callback = self.config['default_offset_commit_callback']
        self._subscription.needs_fetch_committed_offsets = True
        future = self._send_offset_commit_request(offsets)
        future.add_both(lambda res: self.completed_offset_commits.appendleft((callback, offsets, res)))
        return future

    def commit_offsets_sync(self, offsets):
        """Commit specific offsets synchronously.

        This method will retry until the commit completes successfully or an
        unrecoverable error is encountered.

        Arguments:
            offsets (dict {TopicPartition: OffsetAndMetadata}): what to commit

        Raises error on failure
        """
        assert self.config['api_version'] >= (0, 8, 1), 'Unsupported Broker API'
        assert all(map(lambda k: isinstance(k, TopicPartition), offsets))
        assert all(map(lambda v: isinstance(v, OffsetAndMetadata),
                       offsets.values()))
        self._invoke_completed_offset_commit_callbacks()
        if not offsets:
            return

        while True:
            self.ensure_coordinator_ready()

            future = self._send_offset_commit_request(offsets)
            self._client.poll(future=future)

            if future.succeeded():
                return future.value

            if not future.retriable():
                raise future.exception # pylint: disable-msg=raising-bad-type

            time.sleep(self.config['retry_backoff_ms'] / 1000)

    def _maybe_auto_commit_offsets_sync(self):
        if self.config['enable_auto_commit']:
            try:
                self.commit_offsets_sync(self._subscription.all_consumed_offsets())

            # The three main group membership errors are known and should not
            # require a stacktrace -- just a warning
            except (Errors.UnknownMemberIdError,
                    Errors.IllegalGenerationError,
                    Errors.RebalanceInProgressError):
                log.warning("Offset commit failed: group membership out of date"
                            " This is likely to cause duplicate message"
                            " delivery.")
            except Exception:
                log.exception("Offset commit failed: This is likely to cause"
                              " duplicate message delivery")

    def _send_offset_commit_request(self, offsets):
        """Commit offsets for the specified list of topics and partitions.

        This is a non-blocking call which returns a request future that can be
        polled in the case of a synchronous commit or ignored in the
        asynchronous case.

        Arguments:
            offsets (dict of {TopicPartition: OffsetAndMetadata}): what should
                be committed

        Returns:
            Future: indicating whether the commit was successful or not
        """
        assert self.config['api_version'] >= (0, 8, 1), 'Unsupported Broker API'
        assert all(map(lambda k: isinstance(k, TopicPartition), offsets))
        assert all(map(lambda v: isinstance(v, OffsetAndMetadata),
                       offsets.values()))
        if not offsets:
            log.debug('No offsets to commit')
            return Future().success(None)

        node_id = self.coordinator()
        if node_id is None:
            return Future().failure(Errors.GroupCoordinatorNotAvailableError)


        # create the offset commit request
        offset_data = collections.defaultdict(dict)
        for tp, offset in six.iteritems(offsets):
            offset_data[tp.topic][tp.partition] = offset

        if self._subscription.partitions_auto_assigned():
            generation = self.generation()
        else:
            generation = Generation.NO_GENERATION

        # if the generation is None, we are not part of an active group
        # (and we expect to be). The only thing we can do is fail the commit
        # and let the user rejoin the group in poll()
        if self.config['api_version'] >= (0, 9) and generation is None:
            return Future().failure(Errors.CommitFailedError())

        if self.config['api_version'] >= (0, 9):
            request = OffsetCommitRequest[2](
                self.group_id,
                generation.generation_id,
                generation.member_id,
                OffsetCommitRequest[2].DEFAULT_RETENTION_TIME,
                [(
                    topic, [(
                        partition,
                        offset.offset,
                        offset.metadata
                    ) for partition, offset in six.iteritems(partitions)]
                ) for topic, partitions in six.iteritems(offset_data)]
            )
        elif self.config['api_version'] >= (0, 8, 2):
            request = OffsetCommitRequest[1](
                self.group_id, -1, '',
                [(
                    topic, [(
                        partition,
                        offset.offset,
                        -1,
                        offset.metadata
                    ) for partition, offset in six.iteritems(partitions)]
                ) for topic, partitions in six.iteritems(offset_data)]
            )
        elif self.config['api_version'] >= (0, 8, 1):
            request = OffsetCommitRequest[0](
                self.group_id,
                [(
                    topic, [(
                        partition,
                        offset.offset,
                        offset.metadata
                    ) for partition, offset in six.iteritems(partitions)]
                ) for topic, partitions in six.iteritems(offset_data)]
            )

        log.debug("Sending offset-commit request with %s for group %s to %s",
                  offsets, self.group_id, node_id)

        future = Future()
        _f = self._client.send(node_id, request)
        _f.add_callback(self._handle_offset_commit_response, offsets, future, time.time())
        _f.add_errback(self._failed_request, node_id, request, future)
        return future

    def _handle_offset_commit_response(self, offsets, future, send_time, response):
        # TODO look at adding request_latency_ms to response (like java kafka)
        self.consumer_sensors.commit_latency.record((time.time() - send_time) * 1000)
        unauthorized_topics = set()

        for topic, partitions in response.topics:
            for partition, error_code in partitions:
                tp = TopicPartition(topic, partition)
                offset = offsets[tp]

                error_type = Errors.for_code(error_code)
                if error_type is Errors.NoError:
                    log.debug("Group %s committed offset %s for partition %s",
                              self.group_id, offset, tp)
                    if self._subscription.is_assigned(tp):
                        self._subscription.assignment[tp].committed = offset
                elif error_type is Errors.GroupAuthorizationFailedError:
                    log.error("Not authorized to commit offsets for group %s",
                              self.group_id)
                    future.failure(error_type(self.group_id))
                    return
                elif error_type is Errors.TopicAuthorizationFailedError:
                    unauthorized_topics.add(topic)
                elif error_type in (Errors.OffsetMetadataTooLargeError,
                                    Errors.InvalidCommitOffsetSizeError):
                    # raise the error to the user
                    log.debug("OffsetCommit for group %s failed on partition %s"
                              " %s", self.group_id, tp, error_type.__name__)
                    future.failure(error_type())
                    return
                elif error_type is Errors.GroupLoadInProgressError:
                    # just retry
                    log.debug("OffsetCommit for group %s failed: %s",
                              self.group_id, error_type.__name__)
                    future.failure(error_type(self.group_id))
                    return
                elif error_type in (Errors.GroupCoordinatorNotAvailableError,
                                    Errors.NotCoordinatorForGroupError,
                                    Errors.RequestTimedOutError):
                    log.debug("OffsetCommit for group %s failed: %s",
                              self.group_id, error_type.__name__)
                    self.coordinator_dead(error_type())
                    future.failure(error_type(self.group_id))
                    return
                elif error_type in (Errors.UnknownMemberIdError,
                                    Errors.IllegalGenerationError,
                                    Errors.RebalanceInProgressError):
                    # need to re-join group
                    error = error_type(self.group_id)
                    log.debug("OffsetCommit for group %s failed: %s",
                              self.group_id, error)
                    self.reset_generation()
                    future.failure(Errors.CommitFailedError())
                    return
                else:
                    log.error("Group %s failed to commit partition %s at offset"
                              " %s: %s", self.group_id, tp, offset,
                              error_type.__name__)
                    future.failure(error_type())
                    return

        if unauthorized_topics:
            log.error("Not authorized to commit to topics %s for group %s",
                      unauthorized_topics, self.group_id)
            future.failure(Errors.TopicAuthorizationFailedError(unauthorized_topics))
        else:
            future.success(None)

    def _send_offset_fetch_request(self, partitions):
        """Fetch the committed offsets for a set of partitions.

        This is a non-blocking call. The returned future can be polled to get
        the actual offsets returned from the broker.

        Arguments:
            partitions (list of TopicPartition): the partitions to fetch

        Returns:
            Future: resolves to dict of offsets: {TopicPartition: OffsetAndMetadata}
        """
        assert self.config['api_version'] >= (0, 8, 1), 'Unsupported Broker API'
        assert all(map(lambda k: isinstance(k, TopicPartition), partitions))
        if not partitions:
            return Future().success({})

        node_id = self.coordinator()
        if node_id is None:
            return Future().failure(Errors.GroupCoordinatorNotAvailableError)

        # Verify node is ready
        if not self._client.ready(node_id):
            log.debug("Node %s not ready -- failing offset fetch request",
                      node_id)
            return Future().failure(Errors.NodeNotReadyError)

        log.debug("Group %s fetching committed offsets for partitions: %s",
                  self.group_id, partitions)
        # construct the request
        topic_partitions = collections.defaultdict(set)
        for tp in partitions:
            topic_partitions[tp.topic].add(tp.partition)

        if self.config['api_version'] >= (0, 8, 2):
            request = OffsetFetchRequest[1](
                self.group_id,
                list(topic_partitions.items())
            )
        else:
            request = OffsetFetchRequest[0](
                self.group_id,
                list(topic_partitions.items())
            )

        # send the request with a callback
        future = Future()
        _f = self._client.send(node_id, request)
        _f.add_callback(self._handle_offset_fetch_response, future)
        _f.add_errback(self._failed_request, node_id, request, future)
        return future

    def _handle_offset_fetch_response(self, future, response):
        offsets = {}
        for topic, partitions in response.topics:
            for partition, offset, metadata, error_code in partitions:
                tp = TopicPartition(topic, partition)
                error_type = Errors.for_code(error_code)
                if error_type is not Errors.NoError:
                    error = error_type()
                    log.debug("Group %s failed to fetch offset for partition"
                              " %s: %s", self.group_id, tp, error)
                    if error_type is Errors.GroupLoadInProgressError:
                        # just retry
                        future.failure(error)
                    elif error_type is Errors.NotCoordinatorForGroupError:
                        # re-discover the coordinator and retry
                        self.coordinator_dead(error_type())
                        future.failure(error)
                    elif error_type is Errors.UnknownTopicOrPartitionError:
                        log.warning("OffsetFetchRequest -- unknown topic %s"
                                    " (have you committed any offsets yet?)",
                                    topic)
                        continue
                    else:
                        log.error("Unknown error fetching offsets for %s: %s",
                                  tp, error)
                        future.failure(error)
                    return
                elif offset >= 0:
                    # record the position with the offset
                    # (-1 indicates no committed offset to fetch)
                    offsets[tp] = OffsetAndMetadata(offset, metadata)
                else:
                    log.debug("Group %s has no committed offset for partition"
                              " %s", self.group_id, tp)
        future.success(offsets)

    def _default_offset_commit_callback(self, offsets, exception):
        if exception is not None:
            log.error("Offset commit failed: %s", exception)

    def _commit_offsets_async_on_complete(self, offsets, exception):
        if exception is not None:
            log.warning("Auto offset commit failed for group %s: %s",
                        self.group_id, exception)
            if getattr(exception, 'retriable', False):
                self.next_auto_commit_deadline = min(time.time() + self.config['retry_backoff_ms'] / 1000, self.next_auto_commit_deadline)
        else:
            log.debug("Completed autocommit of offsets %s for group %s",
                      offsets, self.group_id)

    def _maybe_auto_commit_offsets_async(self):
        if self.config['enable_auto_commit']:
            if self.coordinator_unknown():
                self.next_auto_commit_deadline = time.time() + self.config['retry_backoff_ms'] / 1000
            elif time.time() > self.next_auto_commit_deadline:
                self.next_auto_commit_deadline = time.time() + self.auto_commit_interval
                self.commit_offsets_async(self._subscription.all_consumed_offsets(),
                                          self._commit_offsets_async_on_complete)


class ConsumerCoordinatorMetrics(object):
    def __init__(self, metrics, metric_group_prefix, subscription):
        self.metrics = metrics
        self.metric_group_name = '%s-coordinator-metrics' % (metric_group_prefix,)

        self.commit_latency = metrics.sensor('commit-latency')
        self.commit_latency.add(metrics.metric_name(
            'commit-latency-avg', self.metric_group_name,
            'The average time taken for a commit request'), Avg())
        self.commit_latency.add(metrics.metric_name(
            'commit-latency-max', self.metric_group_name,
            'The max time taken for a commit request'), Max())
        self.commit_latency.add(metrics.metric_name(
            'commit-rate', self.metric_group_name,
            'The number of commit calls per second'), Rate(sampled_stat=Count()))

        num_parts = AnonMeasurable(lambda config, now:
                                   len(subscription.assigned_partitions()))
        metrics.add_metric(metrics.metric_name(
            'assigned-partitions', self.metric_group_name,
            'The number of partitions currently assigned to this consumer'),
            num_parts)
