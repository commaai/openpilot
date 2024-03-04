from __future__ import absolute_import

import collections
import logging

import kafka.errors as Errors
from kafka.protocol.api import RequestHeader
from kafka.protocol.commit import GroupCoordinatorResponse
from kafka.protocol.frame import KafkaBytes
from kafka.protocol.types import Int32
from kafka.version import __version__

log = logging.getLogger(__name__)


class KafkaProtocol(object):
    """Manage the kafka network protocol

    Use an instance of KafkaProtocol to manage bytes send/recv'd
    from a network socket to a broker.

    Arguments:
        client_id (str): identifier string to be included in each request
        api_version (tuple): Optional tuple to specify api_version to use.
            Currently only used to check for 0.8.2 protocol quirks, but
            may be used for more in the future.
    """
    def __init__(self, client_id=None, api_version=None):
        if client_id is None:
            client_id = self._gen_client_id()
        self._client_id = client_id
        self._api_version = api_version
        self._correlation_id = 0
        self._header = KafkaBytes(4)
        self._rbuffer = None
        self._receiving = False
        self.in_flight_requests = collections.deque()
        self.bytes_to_send = []

    def _next_correlation_id(self):
        self._correlation_id = (self._correlation_id + 1) % 2**31
        return self._correlation_id

    def _gen_client_id(self):
        return 'kafka-python' + __version__

    def send_request(self, request, correlation_id=None):
        """Encode and queue a kafka api request for sending.

        Arguments:
            request (object): An un-encoded kafka request.
            correlation_id (int, optional): Optionally specify an ID to
                correlate requests with responses. If not provided, an ID will
                be generated automatically.

        Returns:
            correlation_id
        """
        log.debug('Sending request %s', request)
        if correlation_id is None:
            correlation_id = self._next_correlation_id()
        header = RequestHeader(request,
                               correlation_id=correlation_id,
                               client_id=self._client_id)
        message = b''.join([header.encode(), request.encode()])
        size = Int32.encode(len(message))
        data = size + message
        self.bytes_to_send.append(data)
        if request.expect_response():
            ifr = (correlation_id, request)
            self.in_flight_requests.append(ifr)
        return correlation_id

    def send_bytes(self):
        """Retrieve all pending bytes to send on the network"""
        data = b''.join(self.bytes_to_send)
        self.bytes_to_send = []
        return data

    def receive_bytes(self, data):
        """Process bytes received from the network.

        Arguments:
            data (bytes): any length bytes received from a network connection
                to a kafka broker.

        Returns:
            responses (list of (correlation_id, response)): any/all completed
                responses, decoded from bytes to python objects.

        Raises:
             KafkaProtocolError: if the bytes received could not be decoded.
             CorrelationIdError: if the response does not match the request
                 correlation id.
        """
        i = 0
        n = len(data)
        responses = []
        while i < n:

            # Not receiving is the state of reading the payload header
            if not self._receiving:
                bytes_to_read = min(4 - self._header.tell(), n - i)
                self._header.write(data[i:i+bytes_to_read])
                i += bytes_to_read

                if self._header.tell() == 4:
                    self._header.seek(0)
                    nbytes = Int32.decode(self._header)
                    # reset buffer and switch state to receiving payload bytes
                    self._rbuffer = KafkaBytes(nbytes)
                    self._receiving = True
                elif self._header.tell() > 4:
                    raise Errors.KafkaError('this should not happen - are you threading?')

            if self._receiving:
                total_bytes = len(self._rbuffer)
                staged_bytes = self._rbuffer.tell()
                bytes_to_read = min(total_bytes - staged_bytes, n - i)
                self._rbuffer.write(data[i:i+bytes_to_read])
                i += bytes_to_read

                staged_bytes = self._rbuffer.tell()
                if staged_bytes > total_bytes:
                    raise Errors.KafkaError('Receive buffer has more bytes than expected?')

                if staged_bytes != total_bytes:
                    break

                self._receiving = False
                self._rbuffer.seek(0)
                resp = self._process_response(self._rbuffer)
                responses.append(resp)
                self._reset_buffer()
        return responses

    def _process_response(self, read_buffer):
        recv_correlation_id = Int32.decode(read_buffer)
        log.debug('Received correlation id: %d', recv_correlation_id)

        if not self.in_flight_requests:
            raise Errors.CorrelationIdError(
                'No in-flight-request found for server response'
                ' with correlation ID %d'
                % (recv_correlation_id,))

        (correlation_id, request) = self.in_flight_requests.popleft()

        # 0.8.2 quirk
        if (recv_correlation_id == 0 and
            correlation_id != 0 and
            request.RESPONSE_TYPE is GroupCoordinatorResponse[0] and
            (self._api_version == (0, 8, 2) or self._api_version is None)):
            log.warning('Kafka 0.8.2 quirk -- GroupCoordinatorResponse'
                        ' Correlation ID does not match request. This'
                        ' should go away once at least one topic has been'
                        ' initialized on the broker.')

        elif correlation_id != recv_correlation_id:
            # return or raise?
            raise Errors.CorrelationIdError(
                'Correlation IDs do not match: sent %d, recv %d'
                % (correlation_id, recv_correlation_id))

        # decode response
        log.debug('Processing response %s', request.RESPONSE_TYPE.__name__)
        try:
            response = request.RESPONSE_TYPE.decode(read_buffer)
        except ValueError:
            read_buffer.seek(0)
            buf = read_buffer.read()
            log.error('Response %d [ResponseType: %s Request: %s]:'
                      ' Unable to decode %d-byte buffer: %r',
                      correlation_id, request.RESPONSE_TYPE,
                      request, len(buf), buf)
            raise Errors.KafkaProtocolError('Unable to decode response')

        return (correlation_id, response)

    def _reset_buffer(self):
        self._receiving = False
        self._header.seek(0)
        self._rbuffer = None
