from __future__ import absolute_import, division

import abc
import copy
import logging
import threading
import time
import weakref

from kafka.vendor import six

from kafka.coordinator.heartbeat import Heartbeat
from kafka import errors as Errors
from kafka.future import Future
from kafka.metrics import AnonMeasurable
from kafka.metrics.stats import Avg, Count, Max, Rate
from kafka.protocol.commit import GroupCoordinatorRequest, OffsetCommitRequest
from kafka.protocol.group import (HeartbeatRequest, JoinGroupRequest,
                            LeaveGroupRequest, SyncGroupRequest)

log = logging.getLogger('kafka.coordinator')


class MemberState(object):
    UNJOINED = '<unjoined>'  # the client is not part of a group
    REBALANCING = '<rebalancing>'  # the client has begun rebalancing
    STABLE = '<stable>'  # the client has joined and is sending heartbeats


class Generation(object):
    def __init__(self, generation_id, member_id, protocol):
        self.generation_id = generation_id
        self.member_id = member_id
        self.protocol = protocol

Generation.NO_GENERATION = Generation(
    OffsetCommitRequest[2].DEFAULT_GENERATION_ID,
    JoinGroupRequest[0].UNKNOWN_MEMBER_ID,
    None)


class UnjoinedGroupException(Errors.KafkaError):
    retriable = True


class BaseCoordinator(object):
    """
    BaseCoordinator implements group management for a single group member
    by interacting with a designated Kafka broker (the coordinator). Group
    semantics are provided by extending this class.  See ConsumerCoordinator
    for example usage.

    From a high level, Kafka's group management protocol consists of the
    following sequence of actions:

    1. Group Registration: Group members register with the coordinator providing
       their own metadata (such as the set of topics they are interested in).

    2. Group/Leader Selection: The coordinator select the members of the group
       and chooses one member as the leader.

    3. State Assignment: The leader collects the metadata from all the members
       of the group and assigns state.

    4. Group Stabilization: Each member receives the state assigned by the
       leader and begins processing.

    To leverage this protocol, an implementation must define the format of
    metadata provided by each member for group registration in
    :meth:`.group_protocols` and the format of the state assignment provided by
    the leader in :meth:`._perform_assignment` and which becomes available to
    members in :meth:`._on_join_complete`.

    Note on locking: this class shares state between the caller and a background
    thread which is used for sending heartbeats after the client has joined the
    group. All mutable state as well as state transitions are protected with the
    class's monitor. Generally this means acquiring the lock before reading or
    writing the state of the group (e.g. generation, member_id) and holding the
    lock when sending a request that affects the state of the group
    (e.g. JoinGroup, LeaveGroup).
    """

    DEFAULT_CONFIG = {
        'group_id': 'kafka-python-default-group',
        'session_timeout_ms': 10000,
        'heartbeat_interval_ms': 3000,
        'max_poll_interval_ms': 300000,
        'retry_backoff_ms': 100,
        'api_version': (0, 10, 1),
        'metric_group_prefix': '',
    }

    def __init__(self, client, metrics, **configs):
        """
        Keyword Arguments:
            group_id (str): name of the consumer group to join for dynamic
                partition assignment (if enabled), and to use for fetching and
                committing offsets. Default: 'kafka-python-default-group'
            session_timeout_ms (int): The timeout used to detect failures when
                using Kafka's group management facilities. Default: 30000
            heartbeat_interval_ms (int): The expected time in milliseconds
                between heartbeats to the consumer coordinator when using
                Kafka's group management feature. Heartbeats are used to ensure
                that the consumer's session stays active and to facilitate
                rebalancing when new consumers join or leave the group. The
                value must be set lower than session_timeout_ms, but typically
                should be set no higher than 1/3 of that value. It can be
                adjusted even lower to control the expected time for normal
                rebalances. Default: 3000
            retry_backoff_ms (int): Milliseconds to backoff when retrying on
                errors. Default: 100.
        """
        self.config = copy.copy(self.DEFAULT_CONFIG)
        for key in self.config:
            if key in configs:
                self.config[key] = configs[key]

        if self.config['api_version'] < (0, 10, 1):
            if self.config['max_poll_interval_ms'] != self.config['session_timeout_ms']:
                raise Errors.KafkaConfigurationError("Broker version %s does not support "
                                                     "different values for max_poll_interval_ms "
                                                     "and session_timeout_ms")

        self._client = client
        self.group_id = self.config['group_id']
        self.heartbeat = Heartbeat(**self.config)
        self._heartbeat_thread = None
        self._lock = threading.Condition()
        self.rejoin_needed = True
        self.rejoining = False  # renamed / complement of java needsJoinPrepare
        self.state = MemberState.UNJOINED
        self.join_future = None
        self.coordinator_id = None
        self._find_coordinator_future = None
        self._generation = Generation.NO_GENERATION
        self.sensors = GroupCoordinatorMetrics(self.heartbeat, metrics,
                                               self.config['metric_group_prefix'])

    @abc.abstractmethod
    def protocol_type(self):
        """
        Unique identifier for the class of supported protocols
        (e.g. "consumer" or "connect").

        Returns:
            str: protocol type name
        """
        pass

    @abc.abstractmethod
    def group_protocols(self):
        """Return the list of supported group protocols and metadata.

        This list is submitted by each group member via a JoinGroupRequest.
        The order of the protocols in the list indicates the preference of the
        protocol (the first entry is the most preferred). The coordinator takes
        this preference into account when selecting the generation protocol
        (generally more preferred protocols will be selected as long as all
        members support them and there is no disagreement on the preference).

        Note: metadata must be type bytes or support an encode() method

        Returns:
            list: [(protocol, metadata), ...]
        """
        pass

    @abc.abstractmethod
    def _on_join_prepare(self, generation, member_id):
        """Invoked prior to each group join or rejoin.

        This is typically used to perform any cleanup from the previous
        generation (such as committing offsets for the consumer)

        Arguments:
            generation (int): The previous generation or -1 if there was none
            member_id (str): The identifier of this member in the previous group
                or '' if there was none
        """
        pass

    @abc.abstractmethod
    def _perform_assignment(self, leader_id, protocol, members):
        """Perform assignment for the group.

        This is used by the leader to push state to all the members of the group
        (e.g. to push partition assignments in the case of the new consumer)

        Arguments:
            leader_id (str): The id of the leader (which is this member)
            protocol (str): the chosen group protocol (assignment strategy)
            members (list): [(member_id, metadata_bytes)] from
                JoinGroupResponse. metadata_bytes are associated with the chosen
                group protocol, and the Coordinator subclass is responsible for
                decoding metadata_bytes based on that protocol.

        Returns:
            dict: {member_id: assignment}; assignment must either be bytes
                or have an encode() method to convert to bytes
        """
        pass

    @abc.abstractmethod
    def _on_join_complete(self, generation, member_id, protocol,
                          member_assignment_bytes):
        """Invoked when a group member has successfully joined a group.

        Arguments:
            generation (int): the generation that was joined
            member_id (str): the identifier for the local member in the group
            protocol (str): the protocol selected by the coordinator
            member_assignment_bytes (bytes): the protocol-encoded assignment
                propagated from the group leader. The Coordinator instance is
                responsible for decoding based on the chosen protocol.
        """
        pass

    def coordinator_unknown(self):
        """Check if we know who the coordinator is and have an active connection

        Side-effect: reset coordinator_id to None if connection failed

        Returns:
            bool: True if the coordinator is unknown
        """
        return self.coordinator() is None

    def coordinator(self):
        """Get the current coordinator

        Returns: the current coordinator id or None if it is unknown
        """
        if self.coordinator_id is None:
            return None
        elif self._client.is_disconnected(self.coordinator_id):
            self.coordinator_dead('Node Disconnected')
            return None
        else:
            return self.coordinator_id

    def ensure_coordinator_ready(self):
        """Block until the coordinator for this group is known
        (and we have an active connection -- java client uses unsent queue).
        """
        with self._client._lock, self._lock:
            while self.coordinator_unknown():

                # Prior to 0.8.2 there was no group coordinator
                # so we will just pick a node at random and treat
                # it as the "coordinator"
                if self.config['api_version'] < (0, 8, 2):
                    self.coordinator_id = self._client.least_loaded_node()
                    if self.coordinator_id is not None:
                        self._client.maybe_connect(self.coordinator_id)
                    continue

                future = self.lookup_coordinator()
                self._client.poll(future=future)

                if future.failed():
                    if future.retriable():
                        if getattr(future.exception, 'invalid_metadata', False):
                            log.debug('Requesting metadata for group coordinator request: %s', future.exception)
                            metadata_update = self._client.cluster.request_update()
                            self._client.poll(future=metadata_update)
                        else:
                            time.sleep(self.config['retry_backoff_ms'] / 1000)
                    else:
                        raise future.exception  # pylint: disable-msg=raising-bad-type

    def _reset_find_coordinator_future(self, result):
        self._find_coordinator_future = None

    def lookup_coordinator(self):
        with self._lock:
            if self._find_coordinator_future is not None:
                return self._find_coordinator_future

            # If there is an error sending the group coordinator request
            # then _reset_find_coordinator_future will immediately fire and
            # set _find_coordinator_future = None
            # To avoid returning None, we capture the future in a local variable
            future = self._send_group_coordinator_request()
            self._find_coordinator_future = future
            self._find_coordinator_future.add_both(self._reset_find_coordinator_future)
            return future

    def need_rejoin(self):
        """Check whether the group should be rejoined (e.g. if metadata changes)

        Returns:
            bool: True if it should, False otherwise
        """
        return self.rejoin_needed

    def poll_heartbeat(self):
        """
        Check the status of the heartbeat thread (if it is active) and indicate
        the liveness of the client. This must be called periodically after
        joining with :meth:`.ensure_active_group` to ensure that the member stays
        in the group. If an interval of time longer than the provided rebalance
        timeout (max_poll_interval_ms) expires without calling this method, then
        the client will proactively leave the group.

        Raises: RuntimeError for unexpected errors raised from the heartbeat thread
        """
        with self._lock:
            if self._heartbeat_thread is not None:
                if self._heartbeat_thread.failed:
                    # set the heartbeat thread to None and raise an exception.
                    # If the user catches it, the next call to ensure_active_group()
                    # will spawn a new heartbeat thread.
                    cause = self._heartbeat_thread.failed
                    self._heartbeat_thread = None
                    raise cause  # pylint: disable-msg=raising-bad-type

                # Awake the heartbeat thread if needed
                if self.heartbeat.should_heartbeat():
                    self._lock.notify()
                self.heartbeat.poll()

    def time_to_next_heartbeat(self):
        """Returns seconds (float) remaining before next heartbeat should be sent

        Note: Returns infinite if group is not joined
        """
        with self._lock:
            # if we have not joined the group, we don't need to send heartbeats
            if self.state is MemberState.UNJOINED:
                return float('inf')
            return self.heartbeat.time_to_next_heartbeat()

    def _handle_join_success(self, member_assignment_bytes):
        with self._lock:
            log.info("Successfully joined group %s with generation %s",
                     self.group_id, self._generation.generation_id)
            self.state = MemberState.STABLE
            self.rejoin_needed = False
            if self._heartbeat_thread:
                self._heartbeat_thread.enable()

    def _handle_join_failure(self, _):
        with self._lock:
            self.state = MemberState.UNJOINED

    def ensure_active_group(self):
        """Ensure that the group is active (i.e. joined and synced)"""
        with self._client._lock, self._lock:
            if self._heartbeat_thread is None:
                self._start_heartbeat_thread()

            while self.need_rejoin() or self._rejoin_incomplete():
                self.ensure_coordinator_ready()

                # call on_join_prepare if needed. We set a flag
                # to make sure that we do not call it a second
                # time if the client is woken up before a pending
                # rebalance completes. This must be called on each
                # iteration of the loop because an event requiring
                # a rebalance (such as a metadata refresh which
                # changes the matched subscription set) can occur
                # while another rebalance is still in progress.
                if not self.rejoining:
                    self._on_join_prepare(self._generation.generation_id,
                                          self._generation.member_id)
                    self.rejoining = True

                # ensure that there are no pending requests to the coordinator.
                # This is important in particular to avoid resending a pending
                # JoinGroup request.
                while not self.coordinator_unknown():
                    if not self._client.in_flight_request_count(self.coordinator_id):
                        break
                    self._client.poll()
                else:
                    continue

                # we store the join future in case we are woken up by the user
                # after beginning the rebalance in the call to poll below.
                # This ensures that we do not mistakenly attempt to rejoin
                # before the pending rebalance has completed.
                if self.join_future is None:
                    # Fence off the heartbeat thread explicitly so that it cannot
                    # interfere with the join group. Note that this must come after
                    # the call to _on_join_prepare since we must be able to continue
                    # sending heartbeats if that callback takes some time.
                    self._heartbeat_thread.disable()

                    self.state = MemberState.REBALANCING
                    future = self._send_join_group_request()

                    self.join_future = future  # this should happen before adding callbacks

                    # handle join completion in the callback so that the
                    # callback will be invoked even if the consumer is woken up
                    # before finishing the rebalance
                    future.add_callback(self._handle_join_success)

                    # we handle failures below after the request finishes.
                    # If the join completes after having been woken up, the
                    # exception is ignored and we will rejoin
                    future.add_errback(self._handle_join_failure)

                else:
                    future = self.join_future

                self._client.poll(future=future)

                if future.succeeded():
                    self._on_join_complete(self._generation.generation_id,
                                           self._generation.member_id,
                                           self._generation.protocol,
                                           future.value)
                    self.join_future = None
                    self.rejoining = False

                else:
                    self.join_future = None
                    exception = future.exception
                    if isinstance(exception, (Errors.UnknownMemberIdError,
                                              Errors.RebalanceInProgressError,
                                              Errors.IllegalGenerationError)):
                        continue
                    elif not future.retriable():
                        raise exception  # pylint: disable-msg=raising-bad-type
                    time.sleep(self.config['retry_backoff_ms'] / 1000)

    def _rejoin_incomplete(self):
        return self.join_future is not None

    def _send_join_group_request(self):
        """Join the group and return the assignment for the next generation.

        This function handles both JoinGroup and SyncGroup, delegating to
        :meth:`._perform_assignment` if elected leader by the coordinator.

        Returns:
            Future: resolves to the encoded-bytes assignment returned from the
                group leader
        """
        if self.coordinator_unknown():
            e = Errors.GroupCoordinatorNotAvailableError(self.coordinator_id)
            return Future().failure(e)

        elif not self._client.ready(self.coordinator_id, metadata_priority=False):
            e = Errors.NodeNotReadyError(self.coordinator_id)
            return Future().failure(e)

        # send a join group request to the coordinator
        log.info("(Re-)joining group %s", self.group_id)
        member_metadata = [
            (protocol, metadata if isinstance(metadata, bytes) else metadata.encode())
            for protocol, metadata in self.group_protocols()
        ]
        if self.config['api_version'] < (0, 9):
            raise Errors.KafkaError('JoinGroupRequest api requires 0.9+ brokers')
        elif (0, 9) <= self.config['api_version'] < (0, 10, 1):
            request = JoinGroupRequest[0](
                self.group_id,
                self.config['session_timeout_ms'],
                self._generation.member_id,
                self.protocol_type(),
                member_metadata)
        elif (0, 10, 1) <= self.config['api_version'] < (0, 11, 0):
            request = JoinGroupRequest[1](
                self.group_id,
                self.config['session_timeout_ms'],
                self.config['max_poll_interval_ms'],
                self._generation.member_id,
                self.protocol_type(),
                member_metadata)
        else:
            request = JoinGroupRequest[2](
                self.group_id,
                self.config['session_timeout_ms'],
                self.config['max_poll_interval_ms'],
                self._generation.member_id,
                self.protocol_type(),
                member_metadata)

        # create the request for the coordinator
        log.debug("Sending JoinGroup (%s) to coordinator %s", request, self.coordinator_id)
        future = Future()
        _f = self._client.send(self.coordinator_id, request)
        _f.add_callback(self._handle_join_group_response, future, time.time())
        _f.add_errback(self._failed_request, self.coordinator_id,
                       request, future)
        return future

    def _failed_request(self, node_id, request, future, error):
        # Marking coordinator dead
        # unless the error is caused by internal client pipelining
        if not isinstance(error, (Errors.NodeNotReadyError,
                                  Errors.TooManyInFlightRequests)):
            log.error('Error sending %s to node %s [%s]',
                      request.__class__.__name__, node_id, error)
            self.coordinator_dead(error)
        else:
            log.debug('Error sending %s to node %s [%s]',
                      request.__class__.__name__, node_id, error)
        future.failure(error)

    def _handle_join_group_response(self, future, send_time, response):
        error_type = Errors.for_code(response.error_code)
        if error_type is Errors.NoError:
            log.debug("Received successful JoinGroup response for group %s: %s",
                      self.group_id, response)
            self.sensors.join_latency.record((time.time() - send_time) * 1000)
            with self._lock:
                if self.state is not MemberState.REBALANCING:
                    # if the consumer was woken up before a rebalance completes,
                    # we may have already left the group. In this case, we do
                    # not want to continue with the sync group.
                    future.failure(UnjoinedGroupException())
                else:
                    self._generation = Generation(response.generation_id,
                                                  response.member_id,
                                                  response.group_protocol)

                if response.leader_id == response.member_id:
                    log.info("Elected group leader -- performing partition"
                             " assignments using %s", self._generation.protocol)
                    self._on_join_leader(response).chain(future)
                else:
                    self._on_join_follower().chain(future)

        elif error_type is Errors.GroupLoadInProgressError:
            log.debug("Attempt to join group %s rejected since coordinator %s"
                      " is loading the group.", self.group_id, self.coordinator_id)
            # backoff and retry
            future.failure(error_type(response))
        elif error_type is Errors.UnknownMemberIdError:
            # reset the member id and retry immediately
            error = error_type(self._generation.member_id)
            self.reset_generation()
            log.debug("Attempt to join group %s failed due to unknown member id",
                      self.group_id)
            future.failure(error)
        elif error_type in (Errors.GroupCoordinatorNotAvailableError,
                            Errors.NotCoordinatorForGroupError):
            # re-discover the coordinator and retry with backoff
            self.coordinator_dead(error_type())
            log.debug("Attempt to join group %s failed due to obsolete "
                      "coordinator information: %s", self.group_id,
                      error_type.__name__)
            future.failure(error_type())
        elif error_type in (Errors.InconsistentGroupProtocolError,
                            Errors.InvalidSessionTimeoutError,
                            Errors.InvalidGroupIdError):
            # log the error and re-throw the exception
            error = error_type(response)
            log.error("Attempt to join group %s failed due to fatal error: %s",
                      self.group_id, error)
            future.failure(error)
        elif error_type is Errors.GroupAuthorizationFailedError:
            future.failure(error_type(self.group_id))
        else:
            # unexpected error, throw the exception
            error = error_type()
            log.error("Unexpected error in join group response: %s", error)
            future.failure(error)

    def _on_join_follower(self):
        # send follower's sync group with an empty assignment
        version = 0 if self.config['api_version'] < (0, 11, 0) else 1
        request = SyncGroupRequest[version](
            self.group_id,
            self._generation.generation_id,
            self._generation.member_id,
            {})
        log.debug("Sending follower SyncGroup for group %s to coordinator %s: %s",
                  self.group_id, self.coordinator_id, request)
        return self._send_sync_group_request(request)

    def _on_join_leader(self, response):
        """
        Perform leader synchronization and send back the assignment
        for the group via SyncGroupRequest

        Arguments:
            response (JoinResponse): broker response to parse

        Returns:
            Future: resolves to member assignment encoded-bytes
        """
        try:
            group_assignment = self._perform_assignment(response.leader_id,
                                                        response.group_protocol,
                                                        response.members)
        except Exception as e:
            return Future().failure(e)

        version = 0 if self.config['api_version'] < (0, 11, 0) else 1
        request = SyncGroupRequest[version](
            self.group_id,
            self._generation.generation_id,
            self._generation.member_id,
            [(member_id,
              assignment if isinstance(assignment, bytes) else assignment.encode())
             for member_id, assignment in six.iteritems(group_assignment)])

        log.debug("Sending leader SyncGroup for group %s to coordinator %s: %s",
                  self.group_id, self.coordinator_id, request)
        return self._send_sync_group_request(request)

    def _send_sync_group_request(self, request):
        if self.coordinator_unknown():
            e = Errors.GroupCoordinatorNotAvailableError(self.coordinator_id)
            return Future().failure(e)

        # We assume that coordinator is ready if we're sending SyncGroup
        # as it typically follows a successful JoinGroup
        # Also note that if client.ready() enforces a metadata priority policy,
        # we can get into an infinite loop if the leader assignment process
        # itself requests a metadata update

        future = Future()
        _f = self._client.send(self.coordinator_id, request)
        _f.add_callback(self._handle_sync_group_response, future, time.time())
        _f.add_errback(self._failed_request, self.coordinator_id,
                       request, future)
        return future

    def _handle_sync_group_response(self, future, send_time, response):
        error_type = Errors.for_code(response.error_code)
        if error_type is Errors.NoError:
            self.sensors.sync_latency.record((time.time() - send_time) * 1000)
            future.success(response.member_assignment)
            return

        # Always rejoin on error
        self.request_rejoin()
        if error_type is Errors.GroupAuthorizationFailedError:
            future.failure(error_type(self.group_id))
        elif error_type is Errors.RebalanceInProgressError:
            log.debug("SyncGroup for group %s failed due to coordinator"
                      " rebalance", self.group_id)
            future.failure(error_type(self.group_id))
        elif error_type in (Errors.UnknownMemberIdError,
                            Errors.IllegalGenerationError):
            error = error_type()
            log.debug("SyncGroup for group %s failed due to %s", self.group_id, error)
            self.reset_generation()
            future.failure(error)
        elif error_type in (Errors.GroupCoordinatorNotAvailableError,
                            Errors.NotCoordinatorForGroupError):
            error = error_type()
            log.debug("SyncGroup for group %s failed due to %s", self.group_id, error)
            self.coordinator_dead(error)
            future.failure(error)
        else:
            error = error_type()
            log.error("Unexpected error from SyncGroup: %s", error)
            future.failure(error)

    def _send_group_coordinator_request(self):
        """Discover the current coordinator for the group.

        Returns:
            Future: resolves to the node id of the coordinator
        """
        node_id = self._client.least_loaded_node()
        if node_id is None:
            return Future().failure(Errors.NoBrokersAvailable())

        elif not self._client.ready(node_id, metadata_priority=False):
            e = Errors.NodeNotReadyError(node_id)
            return Future().failure(e)

        log.debug("Sending group coordinator request for group %s to broker %s",
                  self.group_id, node_id)
        request = GroupCoordinatorRequest[0](self.group_id)
        future = Future()
        _f = self._client.send(node_id, request)
        _f.add_callback(self._handle_group_coordinator_response, future)
        _f.add_errback(self._failed_request, node_id, request, future)
        return future

    def _handle_group_coordinator_response(self, future, response):
        log.debug("Received group coordinator response %s", response)

        error_type = Errors.for_code(response.error_code)
        if error_type is Errors.NoError:
            with self._lock:
                coordinator_id = self._client.cluster.add_group_coordinator(self.group_id, response)
                if not coordinator_id:
                    # This could happen if coordinator metadata is different
                    # than broker metadata
                    future.failure(Errors.IllegalStateError())
                    return

                self.coordinator_id = coordinator_id
                log.info("Discovered coordinator %s for group %s",
                         self.coordinator_id, self.group_id)
                self._client.maybe_connect(self.coordinator_id)
                self.heartbeat.reset_timeouts()
            future.success(self.coordinator_id)

        elif error_type is Errors.GroupCoordinatorNotAvailableError:
            log.debug("Group Coordinator Not Available; retry")
            future.failure(error_type())
        elif error_type is Errors.GroupAuthorizationFailedError:
            error = error_type(self.group_id)
            log.error("Group Coordinator Request failed: %s", error)
            future.failure(error)
        else:
            error = error_type()
            log.error("Group coordinator lookup for group %s failed: %s",
                      self.group_id, error)
            future.failure(error)

    def coordinator_dead(self, error):
        """Mark the current coordinator as dead."""
        if self.coordinator_id is not None:
            log.warning("Marking the coordinator dead (node %s) for group %s: %s.",
                        self.coordinator_id, self.group_id, error)
            self.coordinator_id = None

    def generation(self):
        """Get the current generation state if the group is stable.

        Returns: the current generation or None if the group is unjoined/rebalancing
        """
        with self._lock:
            if self.state is not MemberState.STABLE:
                return None
            return self._generation

    def reset_generation(self):
        """Reset the generation and memberId because we have fallen out of the group."""
        with self._lock:
            self._generation = Generation.NO_GENERATION
            self.rejoin_needed = True
            self.state = MemberState.UNJOINED

    def request_rejoin(self):
        self.rejoin_needed = True

    def _start_heartbeat_thread(self):
        if self._heartbeat_thread is None:
            log.info('Starting new heartbeat thread')
            self._heartbeat_thread = HeartbeatThread(weakref.proxy(self))
            self._heartbeat_thread.daemon = True
            self._heartbeat_thread.start()

    def _close_heartbeat_thread(self):
        if self._heartbeat_thread is not None:
            log.info('Stopping heartbeat thread')
            try:
                self._heartbeat_thread.close()
            except ReferenceError:
                pass
            self._heartbeat_thread = None

    def __del__(self):
        self._close_heartbeat_thread()

    def close(self):
        """Close the coordinator, leave the current group,
        and reset local generation / member_id"""
        self._close_heartbeat_thread()
        self.maybe_leave_group()

    def maybe_leave_group(self):
        """Leave the current group and reset local generation/memberId."""
        with self._client._lock, self._lock:
            if (not self.coordinator_unknown()
                and self.state is not MemberState.UNJOINED
                and self._generation is not Generation.NO_GENERATION):

                # this is a minimal effort attempt to leave the group. we do not
                # attempt any resending if the request fails or times out.
                log.info('Leaving consumer group (%s).', self.group_id)
                version = 0 if self.config['api_version'] < (0, 11, 0) else 1
                request = LeaveGroupRequest[version](self.group_id, self._generation.member_id)
                future = self._client.send(self.coordinator_id, request)
                future.add_callback(self._handle_leave_group_response)
                future.add_errback(log.error, "LeaveGroup request failed: %s")
                self._client.poll(future=future)

            self.reset_generation()

    def _handle_leave_group_response(self, response):
        error_type = Errors.for_code(response.error_code)
        if error_type is Errors.NoError:
            log.debug("LeaveGroup request for group %s returned successfully",
                      self.group_id)
        else:
            log.error("LeaveGroup request for group %s failed with error: %s",
                      self.group_id, error_type())

    def _send_heartbeat_request(self):
        """Send a heartbeat request"""
        if self.coordinator_unknown():
            e = Errors.GroupCoordinatorNotAvailableError(self.coordinator_id)
            return Future().failure(e)

        elif not self._client.ready(self.coordinator_id, metadata_priority=False):
            e = Errors.NodeNotReadyError(self.coordinator_id)
            return Future().failure(e)

        version = 0 if self.config['api_version'] < (0, 11, 0) else 1
        request = HeartbeatRequest[version](self.group_id,
                                            self._generation.generation_id,
                                            self._generation.member_id)
        log.debug("Heartbeat: %s[%s] %s", request.group, request.generation_id, request.member_id)  # pylint: disable-msg=no-member
        future = Future()
        _f = self._client.send(self.coordinator_id, request)
        _f.add_callback(self._handle_heartbeat_response, future, time.time())
        _f.add_errback(self._failed_request, self.coordinator_id,
                       request, future)
        return future

    def _handle_heartbeat_response(self, future, send_time, response):
        self.sensors.heartbeat_latency.record((time.time() - send_time) * 1000)
        error_type = Errors.for_code(response.error_code)
        if error_type is Errors.NoError:
            log.debug("Received successful heartbeat response for group %s",
                      self.group_id)
            future.success(None)
        elif error_type in (Errors.GroupCoordinatorNotAvailableError,
                            Errors.NotCoordinatorForGroupError):
            log.warning("Heartbeat failed for group %s: coordinator (node %s)"
                        " is either not started or not valid", self.group_id,
                        self.coordinator())
            self.coordinator_dead(error_type())
            future.failure(error_type())
        elif error_type is Errors.RebalanceInProgressError:
            log.warning("Heartbeat failed for group %s because it is"
                        " rebalancing", self.group_id)
            self.request_rejoin()
            future.failure(error_type())
        elif error_type is Errors.IllegalGenerationError:
            log.warning("Heartbeat failed for group %s: generation id is not "
                        " current.", self.group_id)
            self.reset_generation()
            future.failure(error_type())
        elif error_type is Errors.UnknownMemberIdError:
            log.warning("Heartbeat: local member_id was not recognized;"
                        " this consumer needs to re-join")
            self.reset_generation()
            future.failure(error_type)
        elif error_type is Errors.GroupAuthorizationFailedError:
            error = error_type(self.group_id)
            log.error("Heartbeat failed: authorization error: %s", error)
            future.failure(error)
        else:
            error = error_type()
            log.error("Heartbeat failed: Unhandled error: %s", error)
            future.failure(error)


class GroupCoordinatorMetrics(object):
    def __init__(self, heartbeat, metrics, prefix, tags=None):
        self.heartbeat = heartbeat
        self.metrics = metrics
        self.metric_group_name = prefix + "-coordinator-metrics"

        self.heartbeat_latency = metrics.sensor('heartbeat-latency')
        self.heartbeat_latency.add(metrics.metric_name(
            'heartbeat-response-time-max', self.metric_group_name,
            'The max time taken to receive a response to a heartbeat request',
            tags), Max())
        self.heartbeat_latency.add(metrics.metric_name(
            'heartbeat-rate', self.metric_group_name,
            'The average number of heartbeats per second',
            tags), Rate(sampled_stat=Count()))

        self.join_latency = metrics.sensor('join-latency')
        self.join_latency.add(metrics.metric_name(
            'join-time-avg', self.metric_group_name,
            'The average time taken for a group rejoin',
            tags), Avg())
        self.join_latency.add(metrics.metric_name(
            'join-time-max', self.metric_group_name,
            'The max time taken for a group rejoin',
            tags), Max())
        self.join_latency.add(metrics.metric_name(
            'join-rate', self.metric_group_name,
            'The number of group joins per second',
            tags), Rate(sampled_stat=Count()))

        self.sync_latency = metrics.sensor('sync-latency')
        self.sync_latency.add(metrics.metric_name(
            'sync-time-avg', self.metric_group_name,
            'The average time taken for a group sync',
            tags), Avg())
        self.sync_latency.add(metrics.metric_name(
            'sync-time-max', self.metric_group_name,
            'The max time taken for a group sync',
            tags), Max())
        self.sync_latency.add(metrics.metric_name(
            'sync-rate', self.metric_group_name,
            'The number of group syncs per second',
            tags), Rate(sampled_stat=Count()))

        metrics.add_metric(metrics.metric_name(
            'last-heartbeat-seconds-ago', self.metric_group_name,
            'The number of seconds since the last controller heartbeat was sent',
            tags), AnonMeasurable(
                lambda _, now: (now / 1000) - self.heartbeat.last_send))


class HeartbeatThread(threading.Thread):
    def __init__(self, coordinator):
        super(HeartbeatThread, self).__init__()
        self.name = coordinator.group_id + '-heartbeat'
        self.coordinator = coordinator
        self.enabled = False
        self.closed = False
        self.failed = None

    def enable(self):
        with self.coordinator._lock:
            self.enabled = True
            self.coordinator.heartbeat.reset_timeouts()
            self.coordinator._lock.notify()

    def disable(self):
        self.enabled = False

    def close(self):
        self.closed = True
        with self.coordinator._lock:
            self.coordinator._lock.notify()
        if self.is_alive():
            self.join(self.coordinator.config['heartbeat_interval_ms'] / 1000)
        if self.is_alive():
            log.warning("Heartbeat thread did not fully terminate during close")

    def run(self):
        try:
            log.debug('Heartbeat thread started')
            while not self.closed:
                self._run_once()

        except ReferenceError:
            log.debug('Heartbeat thread closed due to coordinator gc')

        except RuntimeError as e:
            log.error("Heartbeat thread for group %s failed due to unexpected error: %s",
                      self.coordinator.group_id, e)
            self.failed = e

        finally:
            log.debug('Heartbeat thread closed')

    def _run_once(self):
        with self.coordinator._client._lock, self.coordinator._lock:
            if self.enabled and self.coordinator.state is MemberState.STABLE:
                # TODO: When consumer.wakeup() is implemented, we need to
                # disable here to prevent propagating an exception to this
                # heartbeat thread
                # must get client._lock, or maybe deadlock at heartbeat 
                # failure callbak in consumer poll
                self.coordinator._client.poll(timeout_ms=0)

        with self.coordinator._lock:
            if not self.enabled:
                log.debug('Heartbeat disabled. Waiting')
                self.coordinator._lock.wait()
                log.debug('Heartbeat re-enabled.')
                return

            if self.coordinator.state is not MemberState.STABLE:
                # the group is not stable (perhaps because we left the
                # group or because the coordinator kicked us out), so
                # disable heartbeats and wait for the main thread to rejoin.
                log.debug('Group state is not stable, disabling heartbeats')
                self.disable()
                return

            if self.coordinator.coordinator_unknown():
                future = self.coordinator.lookup_coordinator()
                if not future.is_done or future.failed():
                    # the immediate future check ensures that we backoff
                    # properly in the case that no brokers are available
                    # to connect to (and the future is automatically failed).
                    self.coordinator._lock.wait(self.coordinator.config['retry_backoff_ms'] / 1000)

            elif self.coordinator.heartbeat.session_timeout_expired():
                # the session timeout has expired without seeing a
                # successful heartbeat, so we should probably make sure
                # the coordinator is still healthy.
                log.warning('Heartbeat session expired, marking coordinator dead')
                self.coordinator.coordinator_dead('Heartbeat session expired')

            elif self.coordinator.heartbeat.poll_timeout_expired():
                # the poll timeout has expired, which means that the
                # foreground thread has stalled in between calls to
                # poll(), so we explicitly leave the group.
                log.warning('Heartbeat poll expired, leaving group')
                self.coordinator.maybe_leave_group()

            elif not self.coordinator.heartbeat.should_heartbeat():
                # poll again after waiting for the retry backoff in case
                # the heartbeat failed or the coordinator disconnected
                log.log(0, 'Not ready to heartbeat, waiting')
                self.coordinator._lock.wait(self.coordinator.config['retry_backoff_ms'] / 1000)

            else:
                self.coordinator.heartbeat.sent_heartbeat()
                future = self.coordinator._send_heartbeat_request()
                future.add_callback(self._handle_heartbeat_success)
                future.add_errback(self._handle_heartbeat_failure)

    def _handle_heartbeat_success(self, result):
        with self.coordinator._lock:
            self.coordinator.heartbeat.received_heartbeat()

    def _handle_heartbeat_failure(self, exception):
        with self.coordinator._lock:
            if isinstance(exception, Errors.RebalanceInProgressError):
                # it is valid to continue heartbeating while the group is
                # rebalancing. This ensures that the coordinator keeps the
                # member in the group for as long as the duration of the
                # rebalance timeout. If we stop sending heartbeats, however,
                # then the session timeout may expire before we can rejoin.
                self.coordinator.heartbeat.received_heartbeat()
            else:
                self.coordinator.heartbeat.fail_heartbeat()
                # wake up the thread if it's sleeping to reschedule the heartbeat
                self.coordinator._lock.notify()
