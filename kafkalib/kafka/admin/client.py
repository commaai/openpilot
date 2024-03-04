from __future__ import absolute_import

from collections import defaultdict
import copy
import logging
import socket

from . import ConfigResourceType
from kafka.vendor import six

from kafka.admin.acl_resource import ACLOperation, ACLPermissionType, ACLFilter, ACL, ResourcePattern, ResourceType, \
    ACLResourcePatternType
from kafka.client_async import KafkaClient, selectors
from kafka.coordinator.protocol import ConsumerProtocolMemberMetadata, ConsumerProtocolMemberAssignment, ConsumerProtocol
import kafka.errors as Errors
from kafka.errors import (
    IncompatibleBrokerVersion, KafkaConfigurationError, NotControllerError,
    UnrecognizedBrokerVersion, IllegalArgumentError)
from kafka.metrics import MetricConfig, Metrics
from kafka.protocol.admin import (
    CreateTopicsRequest, DeleteTopicsRequest, DescribeConfigsRequest, AlterConfigsRequest, CreatePartitionsRequest,
    ListGroupsRequest, DescribeGroupsRequest, DescribeAclsRequest, CreateAclsRequest, DeleteAclsRequest,
    DeleteGroupsRequest
)
from kafka.protocol.commit import GroupCoordinatorRequest, OffsetFetchRequest
from kafka.protocol.metadata import MetadataRequest
from kafka.protocol.types import Array
from kafka.structs import TopicPartition, OffsetAndMetadata, MemberInformation, GroupInformation
from kafka.version import __version__


log = logging.getLogger(__name__)


class KafkaAdminClient(object):
    """A class for administering the Kafka cluster.

    Warning:
        This is an unstable interface that was recently added and is subject to
        change without warning. In particular, many methods currently return
        raw protocol tuples. In future releases, we plan to make these into
        nicer, more pythonic objects. Unfortunately, this will likely break
        those interfaces.

    The KafkaAdminClient class will negotiate for the latest version of each message
    protocol format supported by both the kafka-python client library and the
    Kafka broker. Usage of optional fields from protocol versions that are not
    supported by the broker will result in IncompatibleBrokerVersion exceptions.

    Use of this class requires a minimum broker version >= 0.10.0.0.

    Keyword Arguments:
        bootstrap_servers: 'host[:port]' string (or list of 'host[:port]'
            strings) that the consumer should contact to bootstrap initial
            cluster metadata. This does not have to be the full node list.
            It just needs to have at least one broker that will respond to a
            Metadata API Request. Default port is 9092. If no servers are
            specified, will default to localhost:9092.
        client_id (str): a name for this client. This string is passed in
            each request to servers and can be used to identify specific
            server-side log entries that correspond to this client. Also
            submitted to GroupCoordinator for logging with respect to
            consumer group administration. Default: 'kafka-python-{version}'
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
        request_timeout_ms (int): Client request timeout in milliseconds.
            Default: 30000.
        connections_max_idle_ms: Close idle connections after the number of
            milliseconds specified by this config. The broker closes idle
            connections after connections.max.idle.ms, so this avoids hitting
            unexpected socket disconnected errors on the client.
            Default: 540000
        retry_backoff_ms (int): Milliseconds to backoff when retrying on
            errors. Default: 100.
        max_in_flight_requests_per_connection (int): Requests are pipelined
            to kafka brokers up to this number of maximum requests per
            broker connection. Default: 5.
        receive_buffer_bytes (int): The size of the TCP receive buffer
            (SO_RCVBUF) to use when reading data. Default: None (relies on
            system defaults). Java client defaults to 32768.
        send_buffer_bytes (int): The size of the TCP send buffer
            (SO_SNDBUF) to use when sending data. Default: None (relies on
            system defaults). Java client defaults to 131072.
        socket_options (list): List of tuple-arguments to socket.setsockopt
            to apply to broker connection sockets. Default:
            [(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)]
        metadata_max_age_ms (int): The period of time in milliseconds after
            which we force a refresh of metadata even if we haven't seen any
            partition leadership changes to proactively discover any new
            brokers or partitions. Default: 300000
        security_protocol (str): Protocol used to communicate with brokers.
            Valid values are: PLAINTEXT, SSL, SASL_PLAINTEXT, SASL_SSL.
            Default: PLAINTEXT.
        ssl_context (ssl.SSLContext): Pre-configured SSLContext for wrapping
            socket connections. If provided, all other ssl_* configurations
            will be ignored. Default: None.
        ssl_check_hostname (bool): Flag to configure whether SSL handshake
            should verify that the certificate matches the broker's hostname.
            Default: True.
        ssl_cafile (str): Optional filename of CA file to use in certificate
            verification. Default: None.
        ssl_certfile (str): Optional filename of file in PEM format containing
            the client certificate, as well as any CA certificates needed to
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
        api_version (tuple): Specify which Kafka API version to use. If set
            to None, KafkaClient will attempt to infer the broker version by
            probing various APIs. Example: (0, 10, 2). Default: None
        api_version_auto_timeout_ms (int): number of milliseconds to throw a
            timeout exception from the constructor when checking the broker
            api version. Only applies if api_version is None
        selector (selectors.BaseSelector): Provide a specific selector
            implementation to use for I/O multiplexing.
            Default: selectors.DefaultSelector
        metrics (kafka.metrics.Metrics): Optionally provide a metrics
            instance for capturing network IO stats. Default: None.
        metric_group_prefix (str): Prefix for metric names. Default: ''
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

    """
    DEFAULT_CONFIG = {
        # client configs
        'bootstrap_servers': 'localhost',
        'client_id': 'kafka-python-' + __version__,
        'request_timeout_ms': 30000,
        'connections_max_idle_ms': 9 * 60 * 1000,
        'reconnect_backoff_ms': 50,
        'reconnect_backoff_max_ms': 1000,
        'max_in_flight_requests_per_connection': 5,
        'receive_buffer_bytes': None,
        'send_buffer_bytes': None,
        'socket_options': [(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)],
        'sock_chunk_bytes': 4096,  # undocumented experimental option
        'sock_chunk_buffer_count': 1000,  # undocumented experimental option
        'retry_backoff_ms': 100,
        'metadata_max_age_ms': 300000,
        'security_protocol': 'PLAINTEXT',
        'ssl_context': None,
        'ssl_check_hostname': True,
        'ssl_cafile': None,
        'ssl_certfile': None,
        'ssl_keyfile': None,
        'ssl_password': None,
        'ssl_crlfile': None,
        'api_version': None,
        'api_version_auto_timeout_ms': 2000,
        'selector': selectors.DefaultSelector,
        'sasl_mechanism': None,
        'sasl_plain_username': None,
        'sasl_plain_password': None,
        'sasl_kerberos_service_name': 'kafka',
        'sasl_kerberos_domain_name': None,
        'sasl_oauth_token_provider': None,

        # metrics configs
        'metric_reporters': [],
        'metrics_num_samples': 2,
        'metrics_sample_window_ms': 30000,
    }

    def __init__(self, **configs):
        log.debug("Starting KafkaAdminClient with configuration: %s", configs)
        extra_configs = set(configs).difference(self.DEFAULT_CONFIG)
        if extra_configs:
            raise KafkaConfigurationError("Unrecognized configs: {}".format(extra_configs))

        self.config = copy.copy(self.DEFAULT_CONFIG)
        self.config.update(configs)

        # Configure metrics
        metrics_tags = {'client-id': self.config['client_id']}
        metric_config = MetricConfig(samples=self.config['metrics_num_samples'],
                                     time_window_ms=self.config['metrics_sample_window_ms'],
                                     tags=metrics_tags)
        reporters = [reporter() for reporter in self.config['metric_reporters']]
        self._metrics = Metrics(metric_config, reporters)

        self._client = KafkaClient(metrics=self._metrics,
                                   metric_group_prefix='admin',
                                   **self.config)
        self._client.check_version(timeout=(self.config['api_version_auto_timeout_ms'] / 1000))

        # Get auto-discovered version from client if necessary
        if self.config['api_version'] is None:
            self.config['api_version'] = self._client.config['api_version']

        self._closed = False
        self._refresh_controller_id()
        log.debug("KafkaAdminClient started.")

    def close(self):
        """Close the KafkaAdminClient connection to the Kafka broker."""
        if not hasattr(self, '_closed') or self._closed:
            log.info("KafkaAdminClient already closed.")
            return

        self._metrics.close()
        self._client.close()
        self._closed = True
        log.debug("KafkaAdminClient is now closed.")

    def _matching_api_version(self, operation):
        """Find the latest version of the protocol operation supported by both
        this library and the broker.

        This resolves to the lesser of either the latest api version this
        library supports, or the max version supported by the broker.

        :param operation: A list of protocol operation versions from kafka.protocol.
        :return: The max matching version number between client and broker.
        """
        broker_api_versions = self._client.get_api_versions()
        api_key = operation[0].API_KEY
        if broker_api_versions is None or api_key not in broker_api_versions:
            raise IncompatibleBrokerVersion(
                "Kafka broker does not support the '{}' Kafka protocol."
                .format(operation[0].__name__))
        min_version, max_version = broker_api_versions[api_key]
        version = min(len(operation) - 1, max_version)
        if version < min_version:
            # max library version is less than min broker version. Currently,
            # no Kafka versions specify a min msg version. Maybe in the future?
            raise IncompatibleBrokerVersion(
                "No version of the '{}' Kafka protocol is supported by both the client and broker."
                .format(operation[0].__name__))
        return version

    def _validate_timeout(self, timeout_ms):
        """Validate the timeout is set or use the configuration default.

        :param timeout_ms: The timeout provided by api call, in milliseconds.
        :return: The timeout to use for the operation.
        """
        return timeout_ms or self.config['request_timeout_ms']

    def _refresh_controller_id(self):
        """Determine the Kafka cluster controller."""
        version = self._matching_api_version(MetadataRequest)
        if 1 <= version <= 6:
            request = MetadataRequest[version]()
            future = self._send_request_to_node(self._client.least_loaded_node(), request)

            self._wait_for_futures([future])

            response = future.value
            controller_id = response.controller_id
            # verify the controller is new enough to support our requests
            controller_version = self._client.check_version(controller_id, timeout=(self.config['api_version_auto_timeout_ms'] / 1000))
            if controller_version < (0, 10, 0):
                raise IncompatibleBrokerVersion(
                    "The controller appears to be running Kafka {}. KafkaAdminClient requires brokers >= 0.10.0.0."
                    .format(controller_version))
            self._controller_id = controller_id
        else:
            raise UnrecognizedBrokerVersion(
                "Kafka Admin interface cannot determine the controller using MetadataRequest_v{}."
                .format(version))

    def _find_coordinator_id_send_request(self, group_id):
        """Send a FindCoordinatorRequest to a broker.

        :param group_id: The consumer group ID. This is typically the group
            name as a string.
        :return: A message future
        """
        # TODO add support for dynamically picking version of
        # GroupCoordinatorRequest which was renamed to FindCoordinatorRequest.
        # When I experimented with this, the coordinator value returned in
        # GroupCoordinatorResponse_v1 didn't match the value returned by
        # GroupCoordinatorResponse_v0 and I couldn't figure out why.
        version = 0
        # version = self._matching_api_version(GroupCoordinatorRequest)
        if version <= 0:
            request = GroupCoordinatorRequest[version](group_id)
        else:
            raise NotImplementedError(
                "Support for GroupCoordinatorRequest_v{} has not yet been added to KafkaAdminClient."
                .format(version))
        return self._send_request_to_node(self._client.least_loaded_node(), request)

    def _find_coordinator_id_process_response(self, response):
        """Process a FindCoordinatorResponse.

        :param response: a FindCoordinatorResponse.
        :return: The node_id of the broker that is the coordinator.
        """
        if response.API_VERSION <= 0:
            error_type = Errors.for_code(response.error_code)
            if error_type is not Errors.NoError:
                # Note: When error_type.retriable, Java will retry... see
                # KafkaAdminClient's handleFindCoordinatorError method
                raise error_type(
                    "FindCoordinatorRequest failed with response '{}'."
                    .format(response))
        else:
            raise NotImplementedError(
                "Support for FindCoordinatorRequest_v{} has not yet been added to KafkaAdminClient."
                .format(response.API_VERSION))
        return response.coordinator_id

    def _find_coordinator_ids(self, group_ids):
        """Find the broker node_ids of the coordinators of the given groups.

        Sends a FindCoordinatorRequest message to the cluster for each group_id.
        Will block until the FindCoordinatorResponse is received for all groups.
        Any errors are immediately raised.

        :param group_ids: A list of consumer group IDs. This is typically the group
            name as a string.
        :return: A dict of {group_id: node_id} where node_id is the id of the
            broker that is the coordinator for the corresponding group.
        """
        groups_futures = {
            group_id: self._find_coordinator_id_send_request(group_id)
            for group_id in group_ids
        }
        self._wait_for_futures(groups_futures.values())
        groups_coordinators = {
            group_id: self._find_coordinator_id_process_response(future.value)
            for group_id, future in groups_futures.items()
        }
        return groups_coordinators

    def _send_request_to_node(self, node_id, request):
        """Send a Kafka protocol message to a specific broker.

        Returns a future that may be polled for status and results.

        :param node_id: The broker id to which to send the message.
        :param request: The message to send.
        :return: A future object that may be polled for status and results.
        :exception: The exception if the message could not be sent.
        """
        while not self._client.ready(node_id):
            # poll until the connection to broker is ready, otherwise send()
            # will fail with NodeNotReadyError
            self._client.poll()
        return self._client.send(node_id, request)

    def _send_request_to_controller(self, request):
        """Send a Kafka protocol message to the cluster controller.

        Will block until the message result is received.

        :param request: The message to send.
        :return: The Kafka protocol response for the message.
        """
        tries = 2  # in case our cached self._controller_id is outdated
        while tries:
            tries -= 1
            future = self._send_request_to_node(self._controller_id, request)

            self._wait_for_futures([future])

            response = future.value
            # In Java, the error field name is inconsistent:
            #  - CreateTopicsResponse / CreatePartitionsResponse uses topic_errors
            #  - DeleteTopicsResponse uses topic_error_codes
            # So this is a little brittle in that it assumes all responses have
            # one of these attributes and that they always unpack into
            # (topic, error_code) tuples.
            topic_error_tuples = (response.topic_errors if hasattr(response, 'topic_errors')
                    else response.topic_error_codes)
            # Also small py2/py3 compatibility -- py3 can ignore extra values
            # during unpack via: for x, y, *rest in list_of_values. py2 cannot.
            # So for now we have to map across the list and explicitly drop any
            # extra values (usually the error_message)
            for topic, error_code in map(lambda e: e[:2], topic_error_tuples):
                error_type = Errors.for_code(error_code)
                if tries and error_type is NotControllerError:
                    # No need to inspect the rest of the errors for
                    # non-retriable errors because NotControllerError should
                    # either be thrown for all errors or no errors.
                    self._refresh_controller_id()
                    break
                elif error_type is not Errors.NoError:
                    raise error_type(
                        "Request '{}' failed with response '{}'."
                        .format(request, response))
            else:
                return response
        raise RuntimeError("This should never happen, please file a bug with full stacktrace if encountered")

    @staticmethod
    def _convert_new_topic_request(new_topic):
        return (
            new_topic.name,
            new_topic.num_partitions,
            new_topic.replication_factor,
            [
                (partition_id, replicas) for partition_id, replicas in new_topic.replica_assignments.items()
            ],
            [
                (config_key, config_value) for config_key, config_value in new_topic.topic_configs.items()
            ]
        )

    def create_topics(self, new_topics, timeout_ms=None, validate_only=False):
        """Create new topics in the cluster.

        :param new_topics: A list of NewTopic objects.
        :param timeout_ms: Milliseconds to wait for new topics to be created
            before the broker returns.
        :param validate_only: If True, don't actually create new topics.
            Not supported by all versions. Default: False
        :return: Appropriate version of CreateTopicResponse class.
        """
        version = self._matching_api_version(CreateTopicsRequest)
        timeout_ms = self._validate_timeout(timeout_ms)
        if version == 0:
            if validate_only:
                raise IncompatibleBrokerVersion(
                    "validate_only requires CreateTopicsRequest >= v1, which is not supported by Kafka {}."
                    .format(self.config['api_version']))
            request = CreateTopicsRequest[version](
                create_topic_requests=[self._convert_new_topic_request(new_topic) for new_topic in new_topics],
                timeout=timeout_ms
            )
        elif version <= 3:
            request = CreateTopicsRequest[version](
                create_topic_requests=[self._convert_new_topic_request(new_topic) for new_topic in new_topics],
                timeout=timeout_ms,
                validate_only=validate_only
            )
        else:
            raise NotImplementedError(
                "Support for CreateTopics v{} has not yet been added to KafkaAdminClient."
                .format(version))
        # TODO convert structs to a more pythonic interface
        # TODO raise exceptions if errors
        return self._send_request_to_controller(request)

    def delete_topics(self, topics, timeout_ms=None):
        """Delete topics from the cluster.

        :param topics: A list of topic name strings.
        :param timeout_ms: Milliseconds to wait for topics to be deleted
            before the broker returns.
        :return: Appropriate version of DeleteTopicsResponse class.
        """
        version = self._matching_api_version(DeleteTopicsRequest)
        timeout_ms = self._validate_timeout(timeout_ms)
        if version <= 3:
            request = DeleteTopicsRequest[version](
                topics=topics,
                timeout=timeout_ms
            )
            response = self._send_request_to_controller(request)
        else:
            raise NotImplementedError(
                "Support for DeleteTopics v{} has not yet been added to KafkaAdminClient."
                .format(version))
        return response


    def _get_cluster_metadata(self, topics=None, auto_topic_creation=False):
        """
        topics == None means "get all topics"
        """
        version = self._matching_api_version(MetadataRequest)
        if version <= 3:
            if auto_topic_creation:
                raise IncompatibleBrokerVersion(
                    "auto_topic_creation requires MetadataRequest >= v4, which"
                    " is not supported by Kafka {}"
                    .format(self.config['api_version']))

            request = MetadataRequest[version](topics=topics)
        elif version <= 5:
            request = MetadataRequest[version](
                topics=topics,
                allow_auto_topic_creation=auto_topic_creation
            )

        future = self._send_request_to_node(
            self._client.least_loaded_node(),
            request
        )
        self._wait_for_futures([future])
        return future.value

    def list_topics(self):
        metadata = self._get_cluster_metadata(topics=None)
        obj = metadata.to_object()
        return [t['topic'] for t in obj['topics']]

    def describe_topics(self, topics=None):
        metadata = self._get_cluster_metadata(topics=topics)
        obj = metadata.to_object()
        return obj['topics']

    def describe_cluster(self):
        metadata = self._get_cluster_metadata()
        obj = metadata.to_object()
        obj.pop('topics')  # We have 'describe_topics' for this
        return obj

    @staticmethod
    def _convert_describe_acls_response_to_acls(describe_response):
        version = describe_response.API_VERSION

        error = Errors.for_code(describe_response.error_code)
        acl_list = []
        for resources in describe_response.resources:
            if version == 0:
                resource_type, resource_name, acls = resources
                resource_pattern_type = ACLResourcePatternType.LITERAL.value
            elif version <= 1:
                resource_type, resource_name, resource_pattern_type, acls = resources
            else:
                raise NotImplementedError(
                    "Support for DescribeAcls Response v{} has not yet been added to KafkaAdmin."
                        .format(version)
                )
            for acl in acls:
                principal, host, operation, permission_type = acl
                conv_acl = ACL(
                    principal=principal,
                    host=host,
                    operation=ACLOperation(operation),
                    permission_type=ACLPermissionType(permission_type),
                    resource_pattern=ResourcePattern(
                        ResourceType(resource_type),
                        resource_name,
                        ACLResourcePatternType(resource_pattern_type)
                    )
                )
                acl_list.append(conv_acl)

        return (acl_list, error,)

    def describe_acls(self, acl_filter):
        """Describe a set of ACLs

        Used to return a set of ACLs matching the supplied ACLFilter.
        The cluster must be configured with an authorizer for this to work, or
        you will get a SecurityDisabledError

        :param acl_filter: an ACLFilter object
        :return: tuple of a list of matching ACL objects and a KafkaError (NoError if successful)
        """

        version = self._matching_api_version(DescribeAclsRequest)
        if version == 0:
            request = DescribeAclsRequest[version](
                resource_type=acl_filter.resource_pattern.resource_type,
                resource_name=acl_filter.resource_pattern.resource_name,
                principal=acl_filter.principal,
                host=acl_filter.host,
                operation=acl_filter.operation,
                permission_type=acl_filter.permission_type
            )
        elif version <= 1:
            request = DescribeAclsRequest[version](
                resource_type=acl_filter.resource_pattern.resource_type,
                resource_name=acl_filter.resource_pattern.resource_name,
                resource_pattern_type_filter=acl_filter.resource_pattern.pattern_type,
                principal=acl_filter.principal,
                host=acl_filter.host,
                operation=acl_filter.operation,
                permission_type=acl_filter.permission_type

            )
        else:
            raise NotImplementedError(
                "Support for DescribeAcls v{} has not yet been added to KafkaAdmin."
                    .format(version)
            )

        future = self._send_request_to_node(self._client.least_loaded_node(), request)
        self._wait_for_futures([future])
        response = future.value

        error_type = Errors.for_code(response.error_code)
        if error_type is not Errors.NoError:
            # optionally we could retry if error_type.retriable
            raise error_type(
                "Request '{}' failed with response '{}'."
                    .format(request, response))

        return self._convert_describe_acls_response_to_acls(response)

    @staticmethod
    def _convert_create_acls_resource_request_v0(acl):

        return (
            acl.resource_pattern.resource_type,
            acl.resource_pattern.resource_name,
            acl.principal,
            acl.host,
            acl.operation,
            acl.permission_type
        )

    @staticmethod
    def _convert_create_acls_resource_request_v1(acl):

        return (
            acl.resource_pattern.resource_type,
            acl.resource_pattern.resource_name,
            acl.resource_pattern.pattern_type,
            acl.principal,
            acl.host,
            acl.operation,
            acl.permission_type
        )

    @staticmethod
    def _convert_create_acls_response_to_acls(acls, create_response):
        version = create_response.API_VERSION

        creations_error = []
        creations_success = []
        for i, creations in enumerate(create_response.creation_responses):
            if version <= 1:
                error_code, error_message = creations
                acl = acls[i]
                error = Errors.for_code(error_code)
            else:
                raise NotImplementedError(
                    "Support for DescribeAcls Response v{} has not yet been added to KafkaAdmin."
                        .format(version)
                )

            if error is Errors.NoError:
                creations_success.append(acl)
            else:
                creations_error.append((acl, error,))

        return {"succeeded": creations_success, "failed": creations_error}

    def create_acls(self, acls):
        """Create a list of ACLs

        This endpoint only accepts a list of concrete ACL objects, no ACLFilters.
        Throws TopicAlreadyExistsError if topic is already present.

        :param acls: a list of ACL objects
        :return: dict of successes and failures
        """

        for acl in acls:
            if not isinstance(acl, ACL):
                raise IllegalArgumentError("acls must contain ACL objects")

        version = self._matching_api_version(CreateAclsRequest)
        if version == 0:
            request = CreateAclsRequest[version](
                creations=[self._convert_create_acls_resource_request_v0(acl) for acl in acls]
            )
        elif version <= 1:
            request = CreateAclsRequest[version](
                creations=[self._convert_create_acls_resource_request_v1(acl) for acl in acls]
            )
        else:
            raise NotImplementedError(
                "Support for CreateAcls v{} has not yet been added to KafkaAdmin."
                    .format(version)
            )

        future = self._send_request_to_node(self._client.least_loaded_node(), request)
        self._wait_for_futures([future])
        response = future.value

        return self._convert_create_acls_response_to_acls(acls, response)

    @staticmethod
    def _convert_delete_acls_resource_request_v0(acl):
        return (
            acl.resource_pattern.resource_type,
            acl.resource_pattern.resource_name,
            acl.principal,
            acl.host,
            acl.operation,
            acl.permission_type
        )

    @staticmethod
    def _convert_delete_acls_resource_request_v1(acl):
        return (
            acl.resource_pattern.resource_type,
            acl.resource_pattern.resource_name,
            acl.resource_pattern.pattern_type,
            acl.principal,
            acl.host,
            acl.operation,
            acl.permission_type
        )

    @staticmethod
    def _convert_delete_acls_response_to_matching_acls(acl_filters, delete_response):
        version = delete_response.API_VERSION
        filter_result_list = []
        for i, filter_responses in enumerate(delete_response.filter_responses):
            filter_error_code, filter_error_message, matching_acls = filter_responses
            filter_error = Errors.for_code(filter_error_code)
            acl_result_list = []
            for acl in matching_acls:
                if version == 0:
                    error_code, error_message, resource_type, resource_name, principal, host, operation, permission_type = acl
                    resource_pattern_type = ACLResourcePatternType.LITERAL.value
                elif version == 1:
                    error_code, error_message, resource_type, resource_name, resource_pattern_type, principal, host, operation, permission_type = acl
                else:
                    raise NotImplementedError(
                        "Support for DescribeAcls Response v{} has not yet been added to KafkaAdmin."
                            .format(version)
                    )
                acl_error = Errors.for_code(error_code)
                conv_acl = ACL(
                    principal=principal,
                    host=host,
                    operation=ACLOperation(operation),
                    permission_type=ACLPermissionType(permission_type),
                    resource_pattern=ResourcePattern(
                        ResourceType(resource_type),
                        resource_name,
                        ACLResourcePatternType(resource_pattern_type)
                    )
                )
                acl_result_list.append((conv_acl, acl_error,))
            filter_result_list.append((acl_filters[i], acl_result_list, filter_error,))
        return filter_result_list

    def delete_acls(self, acl_filters):
        """Delete a set of ACLs

        Deletes all ACLs matching the list of input ACLFilter

        :param acl_filters: a list of ACLFilter
        :return: a list of 3-tuples corresponding to the list of input filters.
                 The tuples hold (the input ACLFilter, list of affected ACLs, KafkaError instance)
        """

        for acl in acl_filters:
            if not isinstance(acl, ACLFilter):
                raise IllegalArgumentError("acl_filters must contain ACLFilter type objects")

        version = self._matching_api_version(DeleteAclsRequest)

        if version == 0:
            request = DeleteAclsRequest[version](
                filters=[self._convert_delete_acls_resource_request_v0(acl) for acl in acl_filters]
            )
        elif version <= 1:
            request = DeleteAclsRequest[version](
                filters=[self._convert_delete_acls_resource_request_v1(acl) for acl in acl_filters]
            )
        else:
            raise NotImplementedError(
                "Support for DeleteAcls v{} has not yet been added to KafkaAdmin."
                    .format(version)
            )

        future = self._send_request_to_node(self._client.least_loaded_node(), request)
        self._wait_for_futures([future])
        response = future.value

        return self._convert_delete_acls_response_to_matching_acls(acl_filters, response)

    @staticmethod
    def _convert_describe_config_resource_request(config_resource):
        return (
            config_resource.resource_type,
            config_resource.name,
            [
                config_key for config_key, config_value in config_resource.configs.items()
            ] if config_resource.configs else None
        )

    def describe_configs(self, config_resources, include_synonyms=False):
        """Fetch configuration parameters for one or more Kafka resources.

        :param config_resources: An list of ConfigResource objects.
            Any keys in ConfigResource.configs dict will be used to filter the
            result. Setting the configs dict to None will get all values. An
            empty dict will get zero values (as per Kafka protocol).
        :param include_synonyms: If True, return synonyms in response. Not
            supported by all versions. Default: False.
        :return: Appropriate version of DescribeConfigsResponse class.
        """

        # Break up requests by type - a broker config request must be sent to the specific broker.
        # All other (currently just topic resources) can be sent to any broker.
        broker_resources = []
        topic_resources = []

        for config_resource in config_resources:
            if config_resource.resource_type == ConfigResourceType.BROKER:
                broker_resources.append(self._convert_describe_config_resource_request(config_resource))
            else:
                topic_resources.append(self._convert_describe_config_resource_request(config_resource))

        futures = []
        version = self._matching_api_version(DescribeConfigsRequest)
        if version == 0:
            if include_synonyms:
                raise IncompatibleBrokerVersion(
                    "include_synonyms requires DescribeConfigsRequest >= v1, which is not supported by Kafka {}."
                        .format(self.config['api_version']))

            if len(broker_resources) > 0:
                for broker_resource in broker_resources:
                    try:
                        broker_id = int(broker_resource[1])
                    except ValueError:
                        raise ValueError("Broker resource names must be an integer or a string represented integer")

                    futures.append(self._send_request_to_node(
                        broker_id,
                        DescribeConfigsRequest[version](resources=[broker_resource])
                    ))

            if len(topic_resources) > 0:
                futures.append(self._send_request_to_node(
                    self._client.least_loaded_node(),
                    DescribeConfigsRequest[version](resources=topic_resources)
                ))

        elif version <= 2:
            if len(broker_resources) > 0:
                for broker_resource in broker_resources:
                    try:
                        broker_id = int(broker_resource[1])
                    except ValueError:
                        raise ValueError("Broker resource names must be an integer or a string represented integer")

                    futures.append(self._send_request_to_node(
                        broker_id,
                        DescribeConfigsRequest[version](
                            resources=[broker_resource],
                            include_synonyms=include_synonyms)
                    ))

            if len(topic_resources) > 0:
                futures.append(self._send_request_to_node(
                    self._client.least_loaded_node(),
                    DescribeConfigsRequest[version](resources=topic_resources, include_synonyms=include_synonyms)
                ))
        else:
            raise NotImplementedError(
                "Support for DescribeConfigs v{} has not yet been added to KafkaAdminClient.".format(version))

        self._wait_for_futures(futures)
        return [f.value for f in futures]

    @staticmethod
    def _convert_alter_config_resource_request(config_resource):
        return (
            config_resource.resource_type,
            config_resource.name,
            [
                (config_key, config_value) for config_key, config_value in config_resource.configs.items()
            ]
        )

    def alter_configs(self, config_resources):
        """Alter configuration parameters of one or more Kafka resources.

        Warning:
            This is currently broken for BROKER resources because those must be
            sent to that specific broker, versus this always picks the
            least-loaded node. See the comment in the source code for details.
            We would happily accept a PR fixing this.

        :param config_resources: A list of ConfigResource objects.
        :return: Appropriate version of AlterConfigsResponse class.
        """
        version = self._matching_api_version(AlterConfigsRequest)
        if version <= 1:
            request = AlterConfigsRequest[version](
                resources=[self._convert_alter_config_resource_request(config_resource) for config_resource in config_resources]
            )
        else:
            raise NotImplementedError(
                "Support for AlterConfigs v{} has not yet been added to KafkaAdminClient."
                .format(version))
        # TODO the Java client has the note:
        # // We must make a separate AlterConfigs request for every BROKER resource we want to alter
        # // and send the request to that specific broker. Other resources are grouped together into
        # // a single request that may be sent to any broker.
        #
        # So this is currently broken as it always sends to the least_loaded_node()
        future = self._send_request_to_node(self._client.least_loaded_node(), request)

        self._wait_for_futures([future])
        response = future.value
        return response

    # alter replica logs dir protocol not yet implemented
    # Note: have to lookup the broker with the replica assignment and send the request to that broker

    # describe log dirs protocol not yet implemented
    # Note: have to lookup the broker with the replica assignment and send the request to that broker

    @staticmethod
    def _convert_create_partitions_request(topic_name, new_partitions):
        return (
            topic_name,
            (
                new_partitions.total_count,
                new_partitions.new_assignments
            )
        )

    def create_partitions(self, topic_partitions, timeout_ms=None, validate_only=False):
        """Create additional partitions for an existing topic.

        :param topic_partitions: A map of topic name strings to NewPartition objects.
        :param timeout_ms: Milliseconds to wait for new partitions to be
            created before the broker returns.
        :param validate_only: If True, don't actually create new partitions.
            Default: False
        :return: Appropriate version of CreatePartitionsResponse class.
        """
        version = self._matching_api_version(CreatePartitionsRequest)
        timeout_ms = self._validate_timeout(timeout_ms)
        if version <= 1:
            request = CreatePartitionsRequest[version](
                topic_partitions=[self._convert_create_partitions_request(topic_name, new_partitions) for topic_name, new_partitions in topic_partitions.items()],
                timeout=timeout_ms,
                validate_only=validate_only
            )
        else:
            raise NotImplementedError(
                "Support for CreatePartitions v{} has not yet been added to KafkaAdminClient."
                .format(version))
        return self._send_request_to_controller(request)

    # delete records protocol not yet implemented
    # Note: send the request to the partition leaders

    # create delegation token protocol not yet implemented
    # Note: send the request to the least_loaded_node()

    # renew delegation token protocol not yet implemented
    # Note: send the request to the least_loaded_node()

    # expire delegation_token protocol not yet implemented
    # Note: send the request to the least_loaded_node()

    # describe delegation_token protocol not yet implemented
    # Note: send the request to the least_loaded_node()

    def _describe_consumer_groups_send_request(self, group_id, group_coordinator_id, include_authorized_operations=False):
        """Send a DescribeGroupsRequest to the group's coordinator.

        :param group_id: The group name as a string
        :param group_coordinator_id: The node_id of the groups' coordinator
            broker.
        :return: A message future.
        """
        version = self._matching_api_version(DescribeGroupsRequest)
        if version <= 2:
            if include_authorized_operations:
                raise IncompatibleBrokerVersion(
                    "include_authorized_operations requests "
                    "DescribeGroupsRequest >= v3, which is not "
                    "supported by Kafka {}".format(version)
                )
            # Note: KAFKA-6788 A potential optimization is to group the
            # request per coordinator and send one request with a list of
            # all consumer groups. Java still hasn't implemented this
            # because the error checking is hard to get right when some
            # groups error and others don't.
            request = DescribeGroupsRequest[version](groups=(group_id,))
        elif version <= 3:
            request = DescribeGroupsRequest[version](
                groups=(group_id,),
                include_authorized_operations=include_authorized_operations
            )
        else:
            raise NotImplementedError(
                "Support for DescribeGroupsRequest_v{} has not yet been added to KafkaAdminClient."
                .format(version))
        return self._send_request_to_node(group_coordinator_id, request)

    def _describe_consumer_groups_process_response(self, response):
        """Process a DescribeGroupsResponse into a group description."""
        if response.API_VERSION <= 3:
            assert len(response.groups) == 1
            for response_field, response_name in zip(response.SCHEMA.fields, response.SCHEMA.names):
                if isinstance(response_field, Array):
                    described_groups_field_schema = response_field.array_of
                    described_group = response.__dict__[response_name][0]
                    described_group_information_list = []
                    protocol_type_is_consumer = False
                    for (described_group_information, group_information_name, group_information_field) in zip(described_group, described_groups_field_schema.names, described_groups_field_schema.fields):
                        if group_information_name == 'protocol_type':
                            protocol_type = described_group_information
                            protocol_type_is_consumer = (protocol_type == ConsumerProtocol.PROTOCOL_TYPE or not protocol_type)
                        if isinstance(group_information_field, Array):
                            member_information_list = []
                            member_schema = group_information_field.array_of
                            for members in described_group_information:
                                member_information = []
                                for (member, member_field, member_name)  in zip(members, member_schema.fields, member_schema.names):
                                    if protocol_type_is_consumer:
                                        if member_name == 'member_metadata' and member:
                                            member_information.append(ConsumerProtocolMemberMetadata.decode(member))
                                        elif member_name == 'member_assignment' and member:
                                            member_information.append(ConsumerProtocolMemberAssignment.decode(member))
                                        else:
                                            member_information.append(member)
                                member_info_tuple = MemberInformation._make(member_information)
                                member_information_list.append(member_info_tuple)
                            described_group_information_list.append(member_information_list)
                        else:
                            described_group_information_list.append(described_group_information)
                    # Version 3 of the DescribeGroups API introduced the "authorized_operations" field.
                    # This will cause the namedtuple to fail.
                    # Therefore, appending a placeholder of None in it.
                    if response.API_VERSION <=2:
                        described_group_information_list.append(None)
                    group_description = GroupInformation._make(described_group_information_list)
            error_code = group_description.error_code
            error_type = Errors.for_code(error_code)
            # Java has the note: KAFKA-6789, we can retry based on the error code
            if error_type is not Errors.NoError:
                raise error_type(
                    "DescribeGroupsResponse failed with response '{}'."
                    .format(response))
        else:
            raise NotImplementedError(
                "Support for DescribeGroupsResponse_v{} has not yet been added to KafkaAdminClient."
                .format(response.API_VERSION))
        return group_description

    def describe_consumer_groups(self, group_ids, group_coordinator_id=None, include_authorized_operations=False):
        """Describe a set of consumer groups.

        Any errors are immediately raised.

        :param group_ids: A list of consumer group IDs. These are typically the
            group names as strings.
        :param group_coordinator_id: The node_id of the groups' coordinator
            broker. If set to None, it will query the cluster for each group to
            find that group's coordinator. Explicitly specifying this can be
            useful for avoiding extra network round trips if you already know
            the group coordinator. This is only useful when all the group_ids
            have the same coordinator, otherwise it will error. Default: None.
        :param include_authorized_operations: Whether or not to include
            information about the operations a group is allowed to perform.
            Only supported on API version >= v3. Default: False.
        :return: A list of group descriptions. For now the group descriptions
            are the raw results from the DescribeGroupsResponse. Long-term, we
            plan to change this to return namedtuples as well as decoding the
            partition assignments.
        """
        group_descriptions = []

        if group_coordinator_id is not None:
            groups_coordinators = {group_id: group_coordinator_id for group_id in group_ids}
        else:
            groups_coordinators = self._find_coordinator_ids(group_ids)

        futures = [
            self._describe_consumer_groups_send_request(
                group_id,
                coordinator_id,
                include_authorized_operations)
            for group_id, coordinator_id in groups_coordinators.items()
        ]
        self._wait_for_futures(futures)

        for future in futures:
            response = future.value
            group_description = self._describe_consumer_groups_process_response(response)
            group_descriptions.append(group_description)

        return group_descriptions

    def _list_consumer_groups_send_request(self, broker_id):
        """Send a ListGroupsRequest to a broker.

        :param broker_id: The broker's node_id.
        :return: A message future
        """
        version = self._matching_api_version(ListGroupsRequest)
        if version <= 2:
            request = ListGroupsRequest[version]()
        else:
            raise NotImplementedError(
                "Support for ListGroupsRequest_v{} has not yet been added to KafkaAdminClient."
                .format(version))
        return self._send_request_to_node(broker_id, request)

    def _list_consumer_groups_process_response(self, response):
        """Process a ListGroupsResponse into a list of groups."""
        if response.API_VERSION <= 2:
            error_type = Errors.for_code(response.error_code)
            if error_type is not Errors.NoError:
                raise error_type(
                    "ListGroupsRequest failed with response '{}'."
                    .format(response))
        else:
            raise NotImplementedError(
                "Support for ListGroupsResponse_v{} has not yet been added to KafkaAdminClient."
                .format(response.API_VERSION))
        return response.groups

    def list_consumer_groups(self, broker_ids=None):
        """List all consumer groups known to the cluster.

        This returns a list of Consumer Group tuples. The tuples are
        composed of the consumer group name and the consumer group protocol
        type.

        Only consumer groups that store their offsets in Kafka are returned.
        The protocol type will be an empty string for groups created using
        Kafka < 0.9 APIs because, although they store their offsets in Kafka,
        they don't use Kafka for group coordination. For groups created using
        Kafka >= 0.9, the protocol type will typically be "consumer".

        As soon as any error is encountered, it is immediately raised.

        :param broker_ids: A list of broker node_ids to query for consumer
            groups. If set to None, will query all brokers in the cluster.
            Explicitly specifying broker(s) can be useful for determining which
            consumer groups are coordinated by those broker(s). Default: None
        :return list: List of tuples of Consumer Groups.
        :exception GroupCoordinatorNotAvailableError: The coordinator is not
            available, so cannot process requests.
        :exception GroupLoadInProgressError: The coordinator is loading and
            hence can't process requests.
        """
        # While we return a list, internally use a set to prevent duplicates
        # because if a group coordinator fails after being queried, and its
        # consumer groups move to new brokers that haven't yet been queried,
        # then the same group could be returned by multiple brokers.
        consumer_groups = set()
        if broker_ids is None:
            broker_ids = [broker.nodeId for broker in self._client.cluster.brokers()]
        futures = [self._list_consumer_groups_send_request(b) for b in broker_ids]
        self._wait_for_futures(futures)
        for f in futures:
            response = f.value
            consumer_groups.update(self._list_consumer_groups_process_response(response))
        return list(consumer_groups)

    def _list_consumer_group_offsets_send_request(self, group_id,
                group_coordinator_id, partitions=None):
        """Send an OffsetFetchRequest to a broker.

        :param group_id: The consumer group id name for which to fetch offsets.
        :param group_coordinator_id: The node_id of the group's coordinator
            broker.
        :return: A message future
        """
        version = self._matching_api_version(OffsetFetchRequest)
        if version <= 3:
            if partitions is None:
                if version <= 1:
                    raise ValueError(
                        """OffsetFetchRequest_v{} requires specifying the
                        partitions for which to fetch offsets. Omitting the
                        partitions is only supported on brokers >= 0.10.2.
                        For details, see KIP-88.""".format(version))
                topics_partitions = None
            else:
                # transform from [TopicPartition("t1", 1), TopicPartition("t1", 2)] to [("t1", [1, 2])]
                topics_partitions_dict = defaultdict(set)
                for topic, partition in partitions:
                    topics_partitions_dict[topic].add(partition)
                topics_partitions = list(six.iteritems(topics_partitions_dict))
            request = OffsetFetchRequest[version](group_id, topics_partitions)
        else:
            raise NotImplementedError(
                "Support for OffsetFetchRequest_v{} has not yet been added to KafkaAdminClient."
                .format(version))
        return self._send_request_to_node(group_coordinator_id, request)

    def _list_consumer_group_offsets_process_response(self, response):
        """Process an OffsetFetchResponse.

        :param response: an OffsetFetchResponse.
        :return: A dictionary composed of TopicPartition keys and
            OffsetAndMetada values.
        """
        if response.API_VERSION <= 3:

            # OffsetFetchResponse_v1 lacks a top-level error_code
            if response.API_VERSION > 1:
                error_type = Errors.for_code(response.error_code)
                if error_type is not Errors.NoError:
                    # optionally we could retry if error_type.retriable
                    raise error_type(
                        "OffsetFetchResponse failed with response '{}'."
                        .format(response))

            # transform response into a dictionary with TopicPartition keys and
            # OffsetAndMetada values--this is what the Java AdminClient returns
            offsets = {}
            for topic, partitions in response.topics:
                for partition, offset, metadata, error_code in partitions:
                    error_type = Errors.for_code(error_code)
                    if error_type is not Errors.NoError:
                        raise error_type(
                            "Unable to fetch consumer group offsets for topic {}, partition {}"
                            .format(topic, partition))
                    offsets[TopicPartition(topic, partition)] = OffsetAndMetadata(offset, metadata)
        else:
            raise NotImplementedError(
                "Support for OffsetFetchResponse_v{} has not yet been added to KafkaAdminClient."
                .format(response.API_VERSION))
        return offsets

    def list_consumer_group_offsets(self, group_id, group_coordinator_id=None,
                                    partitions=None):
        """Fetch Consumer Offsets for a single consumer group.

        Note:
        This does not verify that the group_id or partitions actually exist
        in the cluster.

        As soon as any error is encountered, it is immediately raised.

        :param group_id: The consumer group id name for which to fetch offsets.
        :param group_coordinator_id: The node_id of the group's coordinator
            broker. If set to None, will query the cluster to find the group
            coordinator. Explicitly specifying this can be useful to prevent
            that extra network round trip if you already know the group
            coordinator. Default: None.
        :param partitions: A list of TopicPartitions for which to fetch
            offsets. On brokers >= 0.10.2, this can be set to None to fetch all
            known offsets for the consumer group. Default: None.
        :return dictionary: A dictionary with TopicPartition keys and
            OffsetAndMetada values. Partitions that are not specified and for
            which the group_id does not have a recorded offset are omitted. An
            offset value of `-1` indicates the group_id has no offset for that
            TopicPartition. A `-1` can only happen for partitions that are
            explicitly specified.
        """
        if group_coordinator_id is None:
            group_coordinator_id = self._find_coordinator_ids([group_id])[group_id]
        future = self._list_consumer_group_offsets_send_request(
                                    group_id, group_coordinator_id, partitions)
        self._wait_for_futures([future])
        response = future.value
        return self._list_consumer_group_offsets_process_response(response)

    def delete_consumer_groups(self, group_ids, group_coordinator_id=None):
        """Delete Consumer Group Offsets for given consumer groups.

        Note:
        This does not verify that the group ids actually exist and
        group_coordinator_id is the correct coordinator for all these groups.

        The result needs checking for potential errors.

        :param group_ids: The consumer group ids of the groups which are to be deleted.
        :param group_coordinator_id: The node_id of the broker which is the coordinator for
            all the groups. Use only if all groups are coordinated by the same broker.
            If set to None, will query the cluster to find the coordinator for every single group.
            Explicitly specifying this can be useful to prevent
            that extra network round trips if you already know the group
            coordinator. Default: None.
        :return: A list of tuples (group_id, KafkaError)
        """
        if group_coordinator_id is not None:
            futures = [self._delete_consumer_groups_send_request(group_ids, group_coordinator_id)]
        else:
            coordinators_groups = defaultdict(list)
            for group_id, coordinator_id in self._find_coordinator_ids(group_ids).items():
                coordinators_groups[coordinator_id].append(group_id)
            futures = [
                self._delete_consumer_groups_send_request(group_ids, coordinator_id)
                for coordinator_id, group_ids in coordinators_groups.items()
            ]

        self._wait_for_futures(futures)

        results = []
        for f in futures:
            results.extend(self._convert_delete_groups_response(f.value))
        return results

    def _convert_delete_groups_response(self, response):
        if response.API_VERSION <= 1:
            results = []
            for group_id, error_code in response.results:
                results.append((group_id, Errors.for_code(error_code)))
            return results
        else:
            raise NotImplementedError(
                "Support for DeleteGroupsResponse_v{} has not yet been added to KafkaAdminClient."
                    .format(response.API_VERSION))

    def _delete_consumer_groups_send_request(self, group_ids, group_coordinator_id):
        """Send a DeleteGroups request to a broker.

        :param group_ids: The consumer group ids of the groups which are to be deleted.
        :param group_coordinator_id: The node_id of the broker which is the coordinator for
            all the groups.
        :return: A message future
        """
        version = self._matching_api_version(DeleteGroupsRequest)
        if version <= 1:
            request = DeleteGroupsRequest[version](group_ids)
        else:
            raise NotImplementedError(
                "Support for DeleteGroupsRequest_v{} has not yet been added to KafkaAdminClient."
                    .format(version))
        return self._send_request_to_node(group_coordinator_id, request)

    def _wait_for_futures(self, futures):
        while not all(future.succeeded() for future in futures):
            for future in futures:
                self._client.poll(future=future)

                if future.failed():
                    raise future.exception  # pylint: disable-msg=raising-bad-type
