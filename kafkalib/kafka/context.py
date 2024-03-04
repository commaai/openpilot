"""
Context manager to commit/rollback consumer offsets.
"""
from __future__ import absolute_import

from logging import getLogger

from kafka.errors import check_error, OffsetOutOfRangeError
from kafka.structs import OffsetCommitRequestPayload


class OffsetCommitContext(object):
    """
    Provides commit/rollback semantics around a `SimpleConsumer`.

    Usage assumes that `auto_commit` is disabled, that messages are consumed in
    batches, and that the consuming process will record its own successful
    processing of each message. Both the commit and rollback operations respect
    a "high-water mark" to ensure that last unsuccessfully processed message
    will be retried.

    Example:

    .. code:: python

        consumer = SimpleConsumer(client, group, topic, auto_commit=False)
        consumer.provide_partition_info()
        consumer.fetch_last_known_offsets()

        while some_condition:
            with OffsetCommitContext(consumer) as context:
                messages = consumer.get_messages(count, block=False)

                for partition, message in messages:
                    if can_process(message):
                        context.mark(partition, message.offset)
                    else:
                        break

                if not context:
                    sleep(delay)


    These semantics allow for deferred message processing (e.g. if `can_process`
    compares message time to clock time) and for repeated processing of the last
    unsuccessful message (until some external error is resolved).
    """

    def __init__(self, consumer):
        """
        :param consumer: an instance of `SimpleConsumer`
        """
        self.consumer = consumer
        self.initial_offsets = None
        self.high_water_mark = None
        self.logger = getLogger("kafka.context")

    def mark(self, partition, offset):
        """
        Set the high-water mark in the current context.

        In order to know the current partition, it is helpful to initialize
        the consumer to provide partition info via:

        .. code:: python

            consumer.provide_partition_info()

        """
        max_offset = max(offset + 1, self.high_water_mark.get(partition, 0))

        self.logger.debug("Setting high-water mark to: %s",
                          {partition: max_offset})

        self.high_water_mark[partition] = max_offset

    def __nonzero__(self):
        """
        Return whether any operations were marked in the context.
        """
        return bool(self.high_water_mark)

    def __enter__(self):
        """
        Start a new context:

         -  Record the initial offsets for rollback
         -  Reset the high-water mark
        """
        self.initial_offsets = dict(self.consumer.offsets)
        self.high_water_mark = dict()

        self.logger.debug("Starting context at: %s", self.initial_offsets)

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        End a context.

         -  If there was no exception, commit up to the current high-water mark.
         -  If there was an offset of range error, attempt to find the correct
            initial offset.
         -  If there was any other error, roll back to the initial offsets.
        """
        if exc_type is None:
            self.commit()
        elif isinstance(exc_value, OffsetOutOfRangeError):
            self.handle_out_of_range()
            return True
        else:
            self.rollback()

    def commit(self):
        """
        Commit this context's offsets:

         -  If the high-water mark has moved, commit up to and position the
            consumer at the high-water mark.
         -  Otherwise, reset to the consumer to the initial offsets.
        """
        if self.high_water_mark:
            self.logger.info("Committing offsets: %s", self.high_water_mark)
            self.commit_partition_offsets(self.high_water_mark)
            self.update_consumer_offsets(self.high_water_mark)
        else:
            self.update_consumer_offsets(self.initial_offsets)

    def rollback(self):
        """
        Rollback this context:

         -  Position the consumer at the initial offsets.
        """
        self.logger.info("Rolling back context: %s", self.initial_offsets)
        self.update_consumer_offsets(self.initial_offsets)

    def commit_partition_offsets(self, partition_offsets):
        """
        Commit explicit partition/offset pairs.
        """
        self.logger.debug("Committing partition offsets: %s", partition_offsets)

        commit_requests = [
            OffsetCommitRequestPayload(self.consumer.topic, partition, offset, None)
            for partition, offset in partition_offsets.items()
        ]
        commit_responses = self.consumer.client.send_offset_commit_request(
            self.consumer.group,
            commit_requests,
        )
        for commit_response in commit_responses:
            check_error(commit_response)

    def update_consumer_offsets(self, partition_offsets):
        """
        Update consumer offsets to explicit positions.
        """
        self.logger.debug("Updating consumer offsets to: %s", partition_offsets)

        for partition, offset in partition_offsets.items():
            self.consumer.offsets[partition] = offset

        # consumer keeps other offset states beyond its `offsets` dictionary,
        # a relative seek with zero delta forces the consumer to reset to the
        # current value of the `offsets` dictionary
        self.consumer.seek(0, 1)

    def handle_out_of_range(self):
        """
        Handle out of range condition by seeking to the beginning of valid
        ranges.

        This assumes that an out of range doesn't happen by seeking past the end
        of valid ranges -- which is far less likely.
        """
        self.logger.info("Seeking beginning of partition on out of range error")
        self.consumer.seek(0, 0)
