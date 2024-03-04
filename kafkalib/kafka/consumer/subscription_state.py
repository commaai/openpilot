from __future__ import absolute_import

import abc
import logging
import re

from kafka.vendor import six

from kafka.errors import IllegalStateError
from kafka.protocol.offset import OffsetResetStrategy
from kafka.structs import OffsetAndMetadata

log = logging.getLogger(__name__)


class SubscriptionState(object):
    """
    A class for tracking the topics, partitions, and offsets for the consumer.
    A partition is "assigned" either directly with assign_from_user() (manual
    assignment) or with assign_from_subscribed() (automatic assignment from
    subscription).

    Once assigned, the partition is not considered "fetchable" until its initial
    position has been set with seek(). Fetchable partitions track a fetch
    position which is used to set the offset of the next fetch, and a consumed
    position which is the last offset that has been returned to the user. You
    can suspend fetching from a partition through pause() without affecting the
    fetched/consumed offsets. The partition will remain unfetchable until the
    resume() is used. You can also query the pause state independently with
    is_paused().

    Note that pause state as well as fetch/consumed positions are not preserved
    when partition assignment is changed whether directly by the user or
    through a group rebalance.

    This class also maintains a cache of the latest commit position for each of
    the assigned partitions. This is updated through committed() and can be used
    to set the initial fetch position (e.g. Fetcher._reset_offset() ).
    """
    _SUBSCRIPTION_EXCEPTION_MESSAGE = (
        "You must choose only one way to configure your consumer:"
        " (1) subscribe to specific topics by name,"
        " (2) subscribe to topics matching a regex pattern,"
        " (3) assign itself specific topic-partitions.")

    # Taken from: https://github.com/apache/kafka/blob/39eb31feaeebfb184d98cc5d94da9148c2319d81/clients/src/main/java/org/apache/kafka/common/internals/Topic.java#L29
    _MAX_NAME_LENGTH = 249
    _TOPIC_LEGAL_CHARS = re.compile('^[a-zA-Z0-9._-]+$')

    def __init__(self, offset_reset_strategy='earliest'):
        """Initialize a SubscriptionState instance

        Keyword Arguments:
            offset_reset_strategy: 'earliest' or 'latest', otherwise
                exception will be raised when fetching an offset that is no
                longer available. Default: 'earliest'
        """
        try:
            offset_reset_strategy = getattr(OffsetResetStrategy,
                                            offset_reset_strategy.upper())
        except AttributeError:
            log.warning('Unrecognized offset_reset_strategy, using NONE')
            offset_reset_strategy = OffsetResetStrategy.NONE
        self._default_offset_reset_strategy = offset_reset_strategy

        self.subscription = None # set() or None
        self.subscribed_pattern = None # regex str or None
        self._group_subscription = set()
        self._user_assignment = set()
        self.assignment = dict()
        self.listener = None

        # initialize to true for the consumers to fetch offset upon starting up
        self.needs_fetch_committed_offsets = True

    def subscribe(self, topics=(), pattern=None, listener=None):
        """Subscribe to a list of topics, or a topic regex pattern.

        Partitions will be dynamically assigned via a group coordinator.
        Topic subscriptions are not incremental: this list will replace the
        current assignment (if there is one).

        This method is incompatible with assign_from_user()

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
        """
        if self._user_assignment or (topics and pattern):
            raise IllegalStateError(self._SUBSCRIPTION_EXCEPTION_MESSAGE)
        assert topics or pattern, 'Must provide topics or pattern'

        if pattern:
            log.info('Subscribing to pattern: /%s/', pattern)
            self.subscription = set()
            self.subscribed_pattern = re.compile(pattern)
        else:
            self.change_subscription(topics)

        if listener and not isinstance(listener, ConsumerRebalanceListener):
            raise TypeError('listener must be a ConsumerRebalanceListener')
        self.listener = listener

    def _ensure_valid_topic_name(self, topic):
        """ Ensures that the topic name is valid according to the kafka source. """

        # See Kafka Source:
        # https://github.com/apache/kafka/blob/39eb31feaeebfb184d98cc5d94da9148c2319d81/clients/src/main/java/org/apache/kafka/common/internals/Topic.java
        if topic is None:
            raise TypeError('All topics must not be None')
        if not isinstance(topic, six.string_types):
            raise TypeError('All topics must be strings')
        if len(topic) == 0:
            raise ValueError('All topics must be non-empty strings')
        if topic == '.' or topic == '..':
            raise ValueError('Topic name cannot be "." or ".."')
        if len(topic) > self._MAX_NAME_LENGTH:
            raise ValueError('Topic name is illegal, it can\'t be longer than {0} characters, topic: "{1}"'.format(self._MAX_NAME_LENGTH, topic))
        if not self._TOPIC_LEGAL_CHARS.match(topic):
            raise ValueError('Topic name "{0}" is illegal, it contains a character other than ASCII alphanumerics, ".", "_" and "-"'.format(topic))

    def change_subscription(self, topics):
        """Change the topic subscription.

        Arguments:
            topics (list of str): topics for subscription

        Raises:
            IllegalStateError: if assign_from_user has been used already
            TypeError: if a topic is None or a non-str
            ValueError: if a topic is an empty string or
                        - a topic name is '.' or '..' or
                        - a topic name does not consist of ASCII-characters/'-'/'_'/'.'
        """
        if self._user_assignment:
            raise IllegalStateError(self._SUBSCRIPTION_EXCEPTION_MESSAGE)

        if isinstance(topics, six.string_types):
            topics = [topics]

        if self.subscription == set(topics):
            log.warning("subscription unchanged by change_subscription(%s)",
                        topics)
            return

        for t in topics:
            self._ensure_valid_topic_name(t)

        log.info('Updating subscribed topics to: %s', topics)
        self.subscription = set(topics)
        self._group_subscription.update(topics)

        # Remove any assigned partitions which are no longer subscribed to
        for tp in set(self.assignment.keys()):
            if tp.topic not in self.subscription:
                del self.assignment[tp]

    def group_subscribe(self, topics):
        """Add topics to the current group subscription.

        This is used by the group leader to ensure that it receives metadata
        updates for all topics that any member of the group is subscribed to.

        Arguments:
            topics (list of str): topics to add to the group subscription
        """
        if self._user_assignment:
            raise IllegalStateError(self._SUBSCRIPTION_EXCEPTION_MESSAGE)
        self._group_subscription.update(topics)

    def reset_group_subscription(self):
        """Reset the group's subscription to only contain topics subscribed by this consumer."""
        if self._user_assignment:
            raise IllegalStateError(self._SUBSCRIPTION_EXCEPTION_MESSAGE)
        assert self.subscription is not None, 'Subscription required'
        self._group_subscription.intersection_update(self.subscription)

    def assign_from_user(self, partitions):
        """Manually assign a list of TopicPartitions to this consumer.

        This interface does not allow for incremental assignment and will
        replace the previous assignment (if there was one).

        Manual topic assignment through this method does not use the consumer's
        group management functionality. As such, there will be no rebalance
        operation triggered when group membership or cluster and topic metadata
        change. Note that it is not possible to use both manual partition
        assignment with assign() and group assignment with subscribe().

        Arguments:
            partitions (list of TopicPartition): assignment for this instance.

        Raises:
            IllegalStateError: if consumer has already called subscribe()
        """
        if self.subscription is not None:
            raise IllegalStateError(self._SUBSCRIPTION_EXCEPTION_MESSAGE)

        if self._user_assignment != set(partitions):
            self._user_assignment = set(partitions)

            for partition in partitions:
                if partition not in self.assignment:
                    self._add_assigned_partition(partition)

            for tp in set(self.assignment.keys()) - self._user_assignment:
                del self.assignment[tp]

            self.needs_fetch_committed_offsets = True

    def assign_from_subscribed(self, assignments):
        """Update the assignment to the specified partitions

        This method is called by the coordinator to dynamically assign
        partitions based on the consumer's topic subscription. This is different
        from assign_from_user() which directly sets the assignment from a
        user-supplied TopicPartition list.

        Arguments:
            assignments (list of TopicPartition): partitions to assign to this
                consumer instance.
        """
        if not self.partitions_auto_assigned():
            raise IllegalStateError(self._SUBSCRIPTION_EXCEPTION_MESSAGE)

        for tp in assignments:
            if tp.topic not in self.subscription:
                raise ValueError("Assigned partition %s for non-subscribed topic." % (tp,))

        # after rebalancing, we always reinitialize the assignment state
        self.assignment.clear()
        for tp in assignments:
            self._add_assigned_partition(tp)
        self.needs_fetch_committed_offsets = True
        log.info("Updated partition assignment: %s", assignments)

    def unsubscribe(self):
        """Clear all topic subscriptions and partition assignments"""
        self.subscription = None
        self._user_assignment.clear()
        self.assignment.clear()
        self.subscribed_pattern = None

    def group_subscription(self):
        """Get the topic subscription for the group.

        For the leader, this will include the union of all member subscriptions.
        For followers, it is the member's subscription only.

        This is used when querying topic metadata to detect metadata changes
        that would require rebalancing (the leader fetches metadata for all
        topics in the group so that it can do partition assignment).

        Returns:
            set: topics
        """
        return self._group_subscription

    def seek(self, partition, offset):
        """Manually specify the fetch offset for a TopicPartition.

        Overrides the fetch offsets that the consumer will use on the next
        poll(). If this API is invoked for the same partition more than once,
        the latest offset will be used on the next poll(). Note that you may
        lose data if this API is arbitrarily used in the middle of consumption,
        to reset the fetch offsets.

        Arguments:
            partition (TopicPartition): partition for seek operation
            offset (int): message offset in partition
        """
        self.assignment[partition].seek(offset)

    def assigned_partitions(self):
        """Return set of TopicPartitions in current assignment."""
        return set(self.assignment.keys())

    def paused_partitions(self):
        """Return current set of paused TopicPartitions."""
        return set(partition for partition in self.assignment
                   if self.is_paused(partition))

    def fetchable_partitions(self):
        """Return set of TopicPartitions that should be Fetched."""
        fetchable = set()
        for partition, state in six.iteritems(self.assignment):
            if state.is_fetchable():
                fetchable.add(partition)
        return fetchable

    def partitions_auto_assigned(self):
        """Return True unless user supplied partitions manually."""
        return self.subscription is not None

    def all_consumed_offsets(self):
        """Returns consumed offsets as {TopicPartition: OffsetAndMetadata}"""
        all_consumed = {}
        for partition, state in six.iteritems(self.assignment):
            if state.has_valid_position:
                all_consumed[partition] = OffsetAndMetadata(state.position, '')
        return all_consumed

    def need_offset_reset(self, partition, offset_reset_strategy=None):
        """Mark partition for offset reset using specified or default strategy.

        Arguments:
            partition (TopicPartition): partition to mark
            offset_reset_strategy (OffsetResetStrategy, optional)
        """
        if offset_reset_strategy is None:
            offset_reset_strategy = self._default_offset_reset_strategy
        self.assignment[partition].await_reset(offset_reset_strategy)

    def has_default_offset_reset_policy(self):
        """Return True if default offset reset policy is Earliest or Latest"""
        return self._default_offset_reset_strategy != OffsetResetStrategy.NONE

    def is_offset_reset_needed(self, partition):
        return self.assignment[partition].awaiting_reset

    def has_all_fetch_positions(self):
        for state in self.assignment.values():
            if not state.has_valid_position:
                return False
        return True

    def missing_fetch_positions(self):
        missing = set()
        for partition, state in six.iteritems(self.assignment):
            if not state.has_valid_position:
                missing.add(partition)
        return missing

    def is_assigned(self, partition):
        return partition in self.assignment

    def is_paused(self, partition):
        return partition in self.assignment and self.assignment[partition].paused

    def is_fetchable(self, partition):
        return partition in self.assignment and self.assignment[partition].is_fetchable()

    def pause(self, partition):
        self.assignment[partition].pause()

    def resume(self, partition):
        self.assignment[partition].resume()

    def _add_assigned_partition(self, partition):
        self.assignment[partition] = TopicPartitionState()


class TopicPartitionState(object):
    def __init__(self):
        self.committed = None # last committed OffsetAndMetadata
        self.has_valid_position = False # whether we have valid position
        self.paused = False # whether this partition has been paused by the user
        self.awaiting_reset = False # whether we are awaiting reset
        self.reset_strategy = None # the reset strategy if awaitingReset is set
        self._position = None # offset exposed to the user
        self.highwater = None
        self.drop_pending_message_set = False
        # The last message offset hint available from a message batch with
        # magic=2 which includes deleted compacted messages
        self.last_offset_from_message_batch = None

    def _set_position(self, offset):
        assert self.has_valid_position, 'Valid position required'
        self._position = offset

    def _get_position(self):
        return self._position

    position = property(_get_position, _set_position, None, "last position")

    def await_reset(self, strategy):
        self.awaiting_reset = True
        self.reset_strategy = strategy
        self._position = None
        self.last_offset_from_message_batch = None
        self.has_valid_position = False

    def seek(self, offset):
        self._position = offset
        self.awaiting_reset = False
        self.reset_strategy = None
        self.has_valid_position = True
        self.drop_pending_message_set = True
        self.last_offset_from_message_batch = None

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def is_fetchable(self):
        return not self.paused and self.has_valid_position


class ConsumerRebalanceListener(object):
    """
    A callback interface that the user can implement to trigger custom actions
    when the set of partitions assigned to the consumer changes.

    This is applicable when the consumer is having Kafka auto-manage group
    membership. If the consumer's directly assign partitions, those
    partitions will never be reassigned and this callback is not applicable.

    When Kafka is managing the group membership, a partition re-assignment will
    be triggered any time the members of the group changes or the subscription
    of the members changes. This can occur when processes die, new process
    instances are added or old instances come back to life after failure.
    Rebalances can also be triggered by changes affecting the subscribed
    topics (e.g. when then number of partitions is administratively adjusted).

    There are many uses for this functionality. One common use is saving offsets
    in a custom store. By saving offsets in the on_partitions_revoked(), call we
    can ensure that any time partition assignment changes the offset gets saved.

    Another use is flushing out any kind of cache of intermediate results the
    consumer may be keeping. For example, consider a case where the consumer is
    subscribed to a topic containing user page views, and the goal is to count
    the number of page views per users for each five minute window.  Let's say
    the topic is partitioned by the user id so that all events for a particular
    user will go to a single consumer instance. The consumer can keep in memory
    a running tally of actions per user and only flush these out to a remote
    data store when its cache gets too big. However if a partition is reassigned
    it may want to automatically trigger a flush of this cache, before the new
    owner takes over consumption.

    This callback will execute in the user thread as part of the Consumer.poll()
    whenever partition assignment changes.

    It is guaranteed that all consumer processes will invoke
    on_partitions_revoked() prior to any process invoking
    on_partitions_assigned(). So if offsets or other state is saved in the
    on_partitions_revoked() call, it should be saved by the time the process
    taking over that partition has their on_partitions_assigned() callback
    called to load the state.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def on_partitions_revoked(self, revoked):
        """
        A callback method the user can implement to provide handling of offset
        commits to a customized store on the start of a rebalance operation.
        This method will be called before a rebalance operation starts and
        after the consumer stops fetching data. It is recommended that offsets
        should be committed in this callback to either Kafka or a custom offset
        store to prevent duplicate data.

        NOTE: This method is only called before rebalances. It is not called
        prior to KafkaConsumer.close()

        Arguments:
            revoked (list of TopicPartition): the partitions that were assigned
                to the consumer on the last rebalance
        """
        pass

    @abc.abstractmethod
    def on_partitions_assigned(self, assigned):
        """
        A callback method the user can implement to provide handling of
        customized offsets on completion of a successful partition
        re-assignment. This method will be called after an offset re-assignment
        completes and before the consumer starts fetching data.

        It is guaranteed that all the processes in a consumer group will execute
        their on_partitions_revoked() callback before any instance executes its
        on_partitions_assigned() callback.

        Arguments:
            assigned (list of TopicPartition): the partitions assigned to the
                consumer (may include partitions that were previously assigned)
        """
        pass
