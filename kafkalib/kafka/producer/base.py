from __future__ import absolute_import

import atexit
import logging
import time

try:
    from queue import Empty, Full, Queue  # pylint: disable=import-error
except ImportError:
    from Queue import Empty, Full, Queue  # pylint: disable=import-error
from collections import defaultdict

from threading import Thread, Event

from kafka.vendor import six

from kafka.structs import (
    ProduceRequestPayload, ProduceResponsePayload, TopicPartition, RetryOptions)
from kafka.errors import (
    kafka_errors, UnsupportedCodecError, FailedPayloadsError,
    RequestTimedOutError, AsyncProducerQueueFull, UnknownError,
    RETRY_ERROR_TYPES, RETRY_BACKOFF_ERROR_TYPES, RETRY_REFRESH_ERROR_TYPES)
from kafka.protocol import CODEC_NONE, ALL_CODECS, create_message_set

log = logging.getLogger('kafka.producer')

BATCH_SEND_DEFAULT_INTERVAL = 20
BATCH_SEND_MSG_COUNT = 20

# unlimited
ASYNC_QUEUE_MAXSIZE = 0
ASYNC_QUEUE_PUT_TIMEOUT = 0
# unlimited retries by default
ASYNC_RETRY_LIMIT = None
ASYNC_RETRY_BACKOFF_MS = 100
ASYNC_RETRY_ON_TIMEOUTS = True
ASYNC_LOG_MESSAGES_ON_ERROR = True

STOP_ASYNC_PRODUCER = -1
ASYNC_STOP_TIMEOUT_SECS = 30

SYNC_FAIL_ON_ERROR_DEFAULT = True


def _send_upstream(queue, client, codec, batch_time, batch_size,
                   req_acks, ack_timeout, retry_options, stop_event,
                   log_messages_on_error=ASYNC_LOG_MESSAGES_ON_ERROR,
                   stop_timeout=ASYNC_STOP_TIMEOUT_SECS,
                   codec_compresslevel=None):
    """Private method to manage producing messages asynchronously

    Listens on the queue for a specified number of messages or until
    a specified timeout and then sends messages to the brokers in grouped
    requests (one per broker).

    Messages placed on the queue should be tuples that conform to this format:
        ((topic, partition), message, key)

    Currently does not mark messages with task_done. Do not attempt to
    :meth:`join`!

    Arguments:
        queue (threading.Queue): the queue from which to get messages
        client (kafka.SimpleClient): instance to use for communicating
            with brokers
        codec (kafka.protocol.ALL_CODECS): compression codec to use
        batch_time (int): interval in seconds to send message batches
        batch_size (int): count of messages that will trigger an immediate send
        req_acks: required acks to use with ProduceRequests. see server protocol
        ack_timeout: timeout to wait for required acks. see server protocol
        retry_options (RetryOptions): settings for retry limits, backoff etc
        stop_event (threading.Event): event to monitor for shutdown signal.
            when this event is 'set', the producer will stop sending messages.
        log_messages_on_error (bool, optional): log stringified message-contents
            on any produce error, otherwise only log a hash() of the contents,
            defaults to True.
        stop_timeout (int or float, optional): number of seconds to continue
            retrying messages after stop_event is set, defaults to 30.
    """
    request_tries = {}

    while not stop_event.is_set():
        try:
            client.reinit()
        except Exception as e:
            log.warn('Async producer failed to connect to brokers; backoff for %s(ms) before retrying', retry_options.backoff_ms)
            time.sleep(float(retry_options.backoff_ms) / 1000)
        else:
            break

    stop_at = None
    while not (stop_event.is_set() and queue.empty() and not request_tries):

        # Handle stop_timeout
        if stop_event.is_set():
            if not stop_at:
                stop_at = stop_timeout + time.time()
            if time.time() > stop_at:
                log.debug('Async producer stopping due to stop_timeout')
                break

        timeout = batch_time
        count = batch_size
        send_at = time.time() + timeout
        msgset = defaultdict(list)

        # Merging messages will require a bit more work to manage correctly
        # for now, don't look for new batches if we have old ones to retry
        if request_tries:
            count = 0
            log.debug('Skipping new batch collection to handle retries')
        else:
            log.debug('Batching size: %s, timeout: %s', count, timeout)

        # Keep fetching till we gather enough messages or a
        # timeout is reached
        while count > 0 and timeout >= 0:
            try:
                topic_partition, msg, key = queue.get(timeout=timeout)
            except Empty:
                break

            # Check if the controller has requested us to stop
            if topic_partition == STOP_ASYNC_PRODUCER:
                stop_event.set()
                break

            # Adjust the timeout to match the remaining period
            count -= 1
            timeout = send_at - time.time()
            msgset[topic_partition].append((msg, key))

        # Send collected requests upstream
        for topic_partition, msg in msgset.items():
            messages = create_message_set(msg, codec, key, codec_compresslevel)
            req = ProduceRequestPayload(
                topic_partition.topic,
                topic_partition.partition,
                tuple(messages))
            request_tries[req] = 0

        if not request_tries:
            continue

        reqs_to_retry, error_cls = [], None
        retry_state = {
            'do_backoff': False,
            'do_refresh': False
        }

        def _handle_error(error_cls, request):
            if issubclass(error_cls, RETRY_ERROR_TYPES) or (retry_options.retry_on_timeouts and issubclass(error_cls, RequestTimedOutError)):
                reqs_to_retry.append(request)
            if issubclass(error_cls, RETRY_BACKOFF_ERROR_TYPES):
                retry_state['do_backoff'] |= True
            if issubclass(error_cls, RETRY_REFRESH_ERROR_TYPES):
                retry_state['do_refresh'] |= True

        requests = list(request_tries.keys())
        log.debug('Sending: %s', requests)
        responses = client.send_produce_request(requests,
                                                acks=req_acks,
                                                timeout=ack_timeout,
                                                fail_on_error=False)

        log.debug('Received: %s', responses)
        for i, response in enumerate(responses):
            error_cls = None
            if isinstance(response, FailedPayloadsError):
                error_cls = response.__class__
                orig_req = response.payload

            elif isinstance(response, ProduceResponsePayload) and response.error:
                error_cls = kafka_errors.get(response.error, UnknownError)
                orig_req = requests[i]

            if error_cls:
                _handle_error(error_cls, orig_req)
                log.error('%s sending ProduceRequestPayload (#%d of %d) '
                          'to %s:%d with msgs %s',
                          error_cls.__name__, (i + 1), len(requests),
                          orig_req.topic, orig_req.partition,
                          orig_req.messages if log_messages_on_error
                                            else hash(orig_req.messages))

        if not reqs_to_retry:
            request_tries = {}
            continue

        # doing backoff before next retry
        if retry_state['do_backoff'] and retry_options.backoff_ms:
            log.warn('Async producer backoff for %s(ms) before retrying', retry_options.backoff_ms)
            time.sleep(float(retry_options.backoff_ms) / 1000)

        # refresh topic metadata before next retry
        if retry_state['do_refresh']:
            log.warn('Async producer forcing metadata refresh metadata before retrying')
            try:
                client.load_metadata_for_topics()
            except Exception:
                log.exception("Async producer couldn't reload topic metadata.")

        # Apply retry limit, dropping messages that are over
        request_tries = dict(
            (key, count + 1)
            for (key, count) in request_tries.items()
                if key in reqs_to_retry
                    and (retry_options.limit is None
                    or (count < retry_options.limit))
        )

        # Log messages we are going to retry
        for orig_req in request_tries.keys():
            log.info('Retrying ProduceRequestPayload to %s:%d with msgs %s',
                     orig_req.topic, orig_req.partition,
                     orig_req.messages if log_messages_on_error
                                       else hash(orig_req.messages))

    if request_tries or not queue.empty():
        log.error('Stopped producer with %d unsent messages', len(request_tries) + queue.qsize())


class Producer(object):
    """
    Base class to be used by producers

    Arguments:
        client (kafka.SimpleClient): instance to use for broker
            communications. If async=True, the background thread will use
            :meth:`client.copy`, which is expected to return a thread-safe
            object.
        codec (kafka.protocol.ALL_CODECS): compression codec to use.
        req_acks (int, optional): A value indicating the acknowledgements that
            the server must receive before responding to the request,
            defaults to 1 (local ack).
        ack_timeout (int, optional): millisecond timeout to wait for the
            configured req_acks, defaults to 1000.
        sync_fail_on_error (bool, optional): whether sync producer should
            raise exceptions (True), or just return errors (False),
            defaults to True.
        async (bool, optional): send message using a background thread,
            defaults to False.
        batch_send_every_n (int, optional): If async is True, messages are
            sent in batches of this size, defaults to 20.
        batch_send_every_t (int or float, optional): If async is True,
            messages are sent immediately after this timeout in seconds, even
            if there are fewer than batch_send_every_n, defaults to 20.
        async_retry_limit (int, optional): number of retries for failed messages
            or None for unlimited, defaults to None / unlimited.
        async_retry_backoff_ms (int, optional): milliseconds to backoff on
            failed messages, defaults to 100.
        async_retry_on_timeouts (bool, optional): whether to retry on
            RequestTimedOutError, defaults to True.
        async_queue_maxsize (int, optional): limit to the size of the
            internal message queue in number of messages (not size), defaults
            to 0 (no limit).
        async_queue_put_timeout (int or float, optional): timeout seconds
            for queue.put in send_messages for async producers -- will only
            apply if async_queue_maxsize > 0 and the queue is Full,
            defaults to 0 (fail immediately on full queue).
        async_log_messages_on_error (bool, optional): set to False and the
            async producer will only log hash() contents on failed produce
            requests, defaults to True (log full messages). Hash logging
            will not allow you to identify the specific message that failed,
            but it will allow you to match failures with retries.
        async_stop_timeout (int or float, optional): seconds to continue
            attempting to send queued messages after :meth:`producer.stop`,
            defaults to 30.

    Deprecated Arguments:
        batch_send (bool, optional): If True, messages are sent by a background
            thread in batches, defaults to False. Deprecated, use 'async'
    """
    ACK_NOT_REQUIRED = 0            # No ack is required
    ACK_AFTER_LOCAL_WRITE = 1       # Send response after it is written to log
    ACK_AFTER_CLUSTER_COMMIT = -1   # Send response after data is committed
    DEFAULT_ACK_TIMEOUT = 1000

    def __init__(self, client,
                 req_acks=ACK_AFTER_LOCAL_WRITE,
                 ack_timeout=DEFAULT_ACK_TIMEOUT,
                 codec=None,
                 codec_compresslevel=None,
                 sync_fail_on_error=SYNC_FAIL_ON_ERROR_DEFAULT,
                 async=False,
                 batch_send=False,  # deprecated, use async
                 batch_send_every_n=BATCH_SEND_MSG_COUNT,
                 batch_send_every_t=BATCH_SEND_DEFAULT_INTERVAL,
                 async_retry_limit=ASYNC_RETRY_LIMIT,
                 async_retry_backoff_ms=ASYNC_RETRY_BACKOFF_MS,
                 async_retry_on_timeouts=ASYNC_RETRY_ON_TIMEOUTS,
                 async_queue_maxsize=ASYNC_QUEUE_MAXSIZE,
                 async_queue_put_timeout=ASYNC_QUEUE_PUT_TIMEOUT,
                 async_log_messages_on_error=ASYNC_LOG_MESSAGES_ON_ERROR,
                 async_stop_timeout=ASYNC_STOP_TIMEOUT_SECS):

        if async:
            assert batch_send_every_n > 0
            assert batch_send_every_t > 0
            assert async_queue_maxsize >= 0

        self.client = client
        self.async = async
        self.req_acks = req_acks
        self.ack_timeout = ack_timeout
        self.stopped = False

        if codec is None:
            codec = CODEC_NONE
        elif codec not in ALL_CODECS:
            raise UnsupportedCodecError("Codec 0x%02x unsupported" % codec)

        self.codec = codec
        self.codec_compresslevel = codec_compresslevel

        if self.async:
            # Messages are sent through this queue
            self.queue = Queue(async_queue_maxsize)
            self.async_queue_put_timeout = async_queue_put_timeout
            async_retry_options = RetryOptions(
                limit=async_retry_limit,
                backoff_ms=async_retry_backoff_ms,
                retry_on_timeouts=async_retry_on_timeouts)
            self.thread_stop_event = Event()
            self.thread = Thread(
                target=_send_upstream,
                args=(self.queue, self.client.copy(), self.codec,
                      batch_send_every_t, batch_send_every_n,
                      self.req_acks, self.ack_timeout,
                      async_retry_options, self.thread_stop_event),
                kwargs={'log_messages_on_error': async_log_messages_on_error,
                        'stop_timeout': async_stop_timeout,
                        'codec_compresslevel': self.codec_compresslevel}
            )

            # Thread will die if main thread exits
            self.thread.daemon = True
            self.thread.start()

            def cleanup(obj):
                if not obj.stopped:
                    obj.stop()
            self._cleanup_func = cleanup
            atexit.register(cleanup, self)
        else:
            self.sync_fail_on_error = sync_fail_on_error

    def send_messages(self, topic, partition, *msg):
        """Helper method to send produce requests.

        Note that msg type *must* be encoded to bytes by user. Passing unicode
        message will not work, for example you should encode before calling
        send_messages via something like `unicode_message.encode('utf-8')`
        All messages will set the message 'key' to None.

        Arguments:
            topic (str): name of topic for produce request
            partition (int): partition number for produce request
            *msg (bytes): one or more message payloads

        Returns:
            ResponseRequest returned by server

        Raises:
            FailedPayloadsError: low-level connection error, can be caused by
                networking failures, or a malformed request.
            ConnectionError:
            KafkaUnavailableError: all known brokers are down when attempting
                to refresh metadata.
            LeaderNotAvailableError: topic or partition is initializing or
                a broker failed and leadership election is in progress.
            NotLeaderForPartitionError: metadata is out of sync; the broker
                that the request was sent to is not the leader for the topic
                or partition.
            UnknownTopicOrPartitionError: the topic or partition has not
                been created yet and auto-creation is not available.
            AsyncProducerQueueFull: in async mode, if too many messages are
                unsent and remain in the internal queue.
        """
        return self._send_messages(topic, partition, *msg)

    def _send_messages(self, topic, partition, *msg, **kwargs):
        key = kwargs.pop('key', None)

        # Guarantee that msg is actually a list or tuple (should always be true)
        if not isinstance(msg, (list, tuple)):
            raise TypeError("msg is not a list or tuple!")

        for m in msg:
            # The protocol allows to have key & payload with null values both,
            # (https://goo.gl/o694yN) but having (null,null) pair doesn't make sense.
            if m is None:
                if key is None:
                    raise TypeError("key and payload can't be null in one")
            # Raise TypeError if any non-null message is not encoded as bytes
            elif not isinstance(m, six.binary_type):
                raise TypeError("all produce message payloads must be null or type bytes")

        # Raise TypeError if the key is not encoded as bytes
        if key is not None and not isinstance(key, six.binary_type):
            raise TypeError("the key must be type bytes")

        if self.async:
            for idx, m in enumerate(msg):
                try:
                    item = (TopicPartition(topic, partition), m, key)
                    if self.async_queue_put_timeout == 0:
                        self.queue.put_nowait(item)
                    else:
                        self.queue.put(item, True, self.async_queue_put_timeout)
                except Full:
                    raise AsyncProducerQueueFull(
                        msg[idx:],
                        'Producer async queue overfilled. '
                        'Current queue size %d.' % self.queue.qsize())
            resp = []
        else:
            messages = create_message_set([(m, key) for m in msg], self.codec, key, self.codec_compresslevel)
            req = ProduceRequestPayload(topic, partition, messages)
            try:
                resp = self.client.send_produce_request(
                    [req], acks=self.req_acks, timeout=self.ack_timeout,
                    fail_on_error=self.sync_fail_on_error
                )
            except Exception:
                log.exception("Unable to send messages")
                raise
        return resp

    def stop(self, timeout=None):
        """
        Stop the producer (async mode). Blocks until async thread completes.
        """
        if timeout is not None:
            log.warning('timeout argument to stop() is deprecated - '
                        'it will be removed in future release')

        if not self.async:
            log.warning('producer.stop() called, but producer is not async')
            return

        if self.stopped:
            log.warning('producer.stop() called, but producer is already stopped')
            return

        if self.async:
            self.queue.put((STOP_ASYNC_PRODUCER, None, None))
            self.thread_stop_event.set()
            self.thread.join()

        if hasattr(self, '_cleanup_func'):
            # Remove cleanup handler now that we've stopped

            # py3 supports unregistering
            if hasattr(atexit, 'unregister'):
                atexit.unregister(self._cleanup_func)  # pylint: disable=no-member

            # py2 requires removing from private attribute...
            else:

                # ValueError on list.remove() if the exithandler no longer exists
                # but that is fine here
                try:
                    atexit._exithandlers.remove(  # pylint: disable=no-member
                        (self._cleanup_func, (self,), {}))
                except ValueError:
                    pass

            del self._cleanup_func

        self.stopped = True

    def __del__(self):
        if self.async and not self.stopped:
            self.stop()
