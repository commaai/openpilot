from __future__ import absolute_import
from kafka.errors import IllegalArgumentError

# enum in stdlib as of py3.4
try:
    from enum import IntEnum  # pylint: disable=import-error
except ImportError:
    # vendored backport module
    from kafka.vendor.enum34 import IntEnum


class ResourceType(IntEnum):
    """Type of kafka resource to set ACL for

    The ANY value is only valid in a filter context
    """

    UNKNOWN = 0,
    ANY = 1,
    CLUSTER = 4,
    DELEGATION_TOKEN = 6,
    GROUP = 3,
    TOPIC = 2,
    TRANSACTIONAL_ID = 5


class ACLOperation(IntEnum):
    """Type of operation

    The ANY value is only valid in a filter context
    """

    ANY = 1,
    ALL = 2,
    READ = 3,
    WRITE = 4,
    CREATE = 5,
    DELETE = 6,
    ALTER = 7,
    DESCRIBE = 8,
    CLUSTER_ACTION = 9,
    DESCRIBE_CONFIGS = 10,
    ALTER_CONFIGS = 11,
    IDEMPOTENT_WRITE = 12


class ACLPermissionType(IntEnum):
    """An enumerated type of permissions

    The ANY value is only valid in a filter context
    """

    ANY = 1,
    DENY = 2,
    ALLOW = 3


class ACLResourcePatternType(IntEnum):
    """An enumerated type of resource patterns

    More details on the pattern types and how they work
    can be found in KIP-290 (Support for prefixed ACLs)
    https://cwiki.apache.org/confluence/display/KAFKA/KIP-290%3A+Support+for+Prefixed+ACLs
    """

    ANY = 1,
    MATCH = 2,
    LITERAL = 3,
    PREFIXED = 4


class ACLFilter(object):
    """Represents a filter to use with describing and deleting ACLs

    The difference between this class and the ACL class is mainly that
    we allow using ANY with the operation, permission, and resource type objects
    to fetch ALCs matching any of the properties.

    To make a filter matching any principal, set principal to None
    """

    def __init__(
        self,
        principal,
        host,
        operation,
        permission_type,
        resource_pattern
    ):
        self.principal = principal
        self.host = host
        self.operation = operation
        self.permission_type = permission_type
        self.resource_pattern = resource_pattern

        self.validate()

    def validate(self):
        if not isinstance(self.operation, ACLOperation):
            raise IllegalArgumentError("operation must be an ACLOperation object, and cannot be ANY")
        if not isinstance(self.permission_type, ACLPermissionType):
            raise IllegalArgumentError("permission_type must be an ACLPermissionType object, and cannot be ANY")
        if not isinstance(self.resource_pattern, ResourcePatternFilter):
            raise IllegalArgumentError("resource_pattern must be a ResourcePatternFilter object")

    def __repr__(self):
        return "<ACL principal={principal}, resource={resource}, operation={operation}, type={type}, host={host}>".format(
            principal=self.principal,
            host=self.host,
            operation=self.operation.name,
            type=self.permission_type.name,
            resource=self.resource_pattern
        )

    def __eq__(self, other):
        return all((
            self.principal == other.principal,
            self.host == other.host,
            self.operation == other.operation,
            self.permission_type == other.permission_type,
            self.resource_pattern == other.resource_pattern
        ))

    def __hash__(self):
        return hash((
            self.principal,
            self.host,
            self.operation,
            self.permission_type,
            self.resource_pattern,
        ))


class ACL(ACLFilter):
    """Represents a concrete ACL for a specific ResourcePattern

    In kafka an ACL is a 4-tuple of (principal, host, operation, permission_type)
    that limits who can do what on a specific resource (or since KIP-290 a resource pattern)

    Terminology:
    Principal -> This is the identifier for the user. Depending on the authorization method used (SSL, SASL etc)
        the principal will look different. See http://kafka.apache.org/documentation/#security_authz for details.
        The principal must be on the format "User:<name>" or kafka will treat it as invalid. It's possible to use
        other principal types than "User" if using a custom authorizer for the cluster.
    Host -> This must currently be an IP address. It cannot be a range, and it cannot be a domain name.
        It can be set to "*", which is special cased in kafka to mean "any host"
    Operation -> Which client operation this ACL refers to. Has different meaning depending
        on the resource type the ACL refers to. See https://docs.confluent.io/current/kafka/authorization.html#acl-format
        for a list of which combinations of resource/operation that unlocks which kafka APIs
    Permission Type: Whether this ACL is allowing or denying access
    Resource Pattern -> This is a representation of the resource or resource pattern that the ACL
        refers to. See the ResourcePattern class for details.

    """

    def __init__(
            self,
            principal,
            host,
            operation,
            permission_type,
            resource_pattern
    ):
        super(ACL, self).__init__(principal, host, operation, permission_type, resource_pattern)
        self.validate()

    def validate(self):
        if self.operation == ACLOperation.ANY:
            raise IllegalArgumentError("operation cannot be ANY")
        if self.permission_type == ACLPermissionType.ANY:
            raise IllegalArgumentError("permission_type cannot be ANY")
        if not isinstance(self.resource_pattern, ResourcePattern):
            raise IllegalArgumentError("resource_pattern must be a ResourcePattern object")


class ResourcePatternFilter(object):
    def __init__(
            self,
            resource_type,
            resource_name,
            pattern_type
    ):
        self.resource_type = resource_type
        self.resource_name = resource_name
        self.pattern_type = pattern_type

        self.validate()

    def validate(self):
        if not isinstance(self.resource_type, ResourceType):
            raise IllegalArgumentError("resource_type must be a ResourceType object")
        if not isinstance(self.pattern_type, ACLResourcePatternType):
            raise IllegalArgumentError("pattern_type must be an ACLResourcePatternType object")

    def __repr__(self):
        return "<ResourcePattern type={}, name={}, pattern={}>".format(
            self.resource_type.name,
            self.resource_name,
            self.pattern_type.name
        )

    def __eq__(self, other):
        return all((
            self.resource_type == other.resource_type,
            self.resource_name == other.resource_name,
            self.pattern_type == other.pattern_type,
        ))

    def __hash__(self):
        return hash((
            self.resource_type,
            self.resource_name,
            self.pattern_type
        ))


class ResourcePattern(ResourcePatternFilter):
    """A resource pattern to apply the ACL to

    Resource patterns are used to be able to specify which resources an ACL
    describes in a more flexible way than just pointing to a literal topic name for example.
    Since KIP-290 (kafka 2.0) it's possible to set an ACL for a prefixed resource name, which
    can cut down considerably on the number of ACLs needed when the number of topics and
    consumer groups start to grow.
    The default pattern_type is LITERAL, and it describes a specific resource. This is also how
    ACLs worked before the introduction of prefixed ACLs
    """

    def __init__(
            self,
            resource_type,
            resource_name,
            pattern_type=ACLResourcePatternType.LITERAL
    ):
        super(ResourcePattern, self).__init__(resource_type, resource_name, pattern_type)
        self.validate()

    def validate(self):
        if self.resource_type == ResourceType.ANY:
            raise IllegalArgumentError("resource_type cannot be ANY")
        if self.pattern_type in [ACLResourcePatternType.ANY, ACLResourcePatternType.MATCH]:
            raise IllegalArgumentError(
                "pattern_type cannot be {} on a concrete ResourcePattern".format(self.pattern_type.name)
            )
