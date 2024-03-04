from __future__ import absolute_import

from kafka.protocol.api import Request, Response
from kafka.protocol.types import Array, Int8, Int16, Int32, Int64, Schema, String

UNKNOWN_OFFSET = -1


class OffsetResetStrategy(object):
    LATEST = -1
    EARLIEST = -2
    NONE = 0


class OffsetResponse_v0(Response):
    API_KEY = 2
    API_VERSION = 0
    SCHEMA = Schema(
        ('topics', Array(
            ('topic', String('utf-8')),
            ('partitions', Array(
                ('partition', Int32),
                ('error_code', Int16),
                ('offsets', Array(Int64))))))
    )

class OffsetResponse_v1(Response):
    API_KEY = 2
    API_VERSION = 1
    SCHEMA = Schema(
        ('topics', Array(
            ('topic', String('utf-8')),
            ('partitions', Array(
                ('partition', Int32),
                ('error_code', Int16),
                ('timestamp', Int64),
                ('offset', Int64)))))
    )


class OffsetResponse_v2(Response):
    API_KEY = 2
    API_VERSION = 2
    SCHEMA = Schema(
        ('throttle_time_ms', Int32),
        ('topics', Array(
            ('topic', String('utf-8')),
            ('partitions', Array(
                ('partition', Int32),
                ('error_code', Int16),
                ('timestamp', Int64),
                ('offset', Int64)))))
    )


class OffsetResponse_v3(Response):
    """
    on quota violation, brokers send out responses before throttling
    """
    API_KEY = 2
    API_VERSION = 3
    SCHEMA = OffsetResponse_v2.SCHEMA


class OffsetResponse_v4(Response):
    """
    Add leader_epoch to response
    """
    API_KEY = 2
    API_VERSION = 4
    SCHEMA = Schema(
        ('throttle_time_ms', Int32),
        ('topics', Array(
            ('topic', String('utf-8')),
            ('partitions', Array(
                ('partition', Int32),
                ('error_code', Int16),
                ('timestamp', Int64),
                ('offset', Int64),
                ('leader_epoch', Int32)))))
    )


class OffsetResponse_v5(Response):
    """
    adds a new error code, OFFSET_NOT_AVAILABLE
    """
    API_KEY = 2
    API_VERSION = 5
    SCHEMA = OffsetResponse_v4.SCHEMA


class OffsetRequest_v0(Request):
    API_KEY = 2
    API_VERSION = 0
    RESPONSE_TYPE = OffsetResponse_v0
    SCHEMA = Schema(
        ('replica_id', Int32),
        ('topics', Array(
            ('topic', String('utf-8')),
            ('partitions', Array(
                ('partition', Int32),
                ('timestamp', Int64),
                ('max_offsets', Int32)))))
    )
    DEFAULTS = {
        'replica_id': -1
    }

class OffsetRequest_v1(Request):
    API_KEY = 2
    API_VERSION = 1
    RESPONSE_TYPE = OffsetResponse_v1
    SCHEMA = Schema(
        ('replica_id', Int32),
        ('topics', Array(
            ('topic', String('utf-8')),
            ('partitions', Array(
                ('partition', Int32),
                ('timestamp', Int64)))))
    )
    DEFAULTS = {
        'replica_id': -1
    }


class OffsetRequest_v2(Request):
    API_KEY = 2
    API_VERSION = 2
    RESPONSE_TYPE = OffsetResponse_v2
    SCHEMA = Schema(
        ('replica_id', Int32),
        ('isolation_level', Int8),  # <- added isolation_level
        ('topics', Array(
            ('topic', String('utf-8')),
            ('partitions', Array(
                ('partition', Int32),
                ('timestamp', Int64)))))
    )
    DEFAULTS = {
        'replica_id': -1
    }


class OffsetRequest_v3(Request):
    API_KEY = 2
    API_VERSION = 3
    RESPONSE_TYPE = OffsetResponse_v3
    SCHEMA = OffsetRequest_v2.SCHEMA
    DEFAULTS = {
        'replica_id': -1
    }


class OffsetRequest_v4(Request):
    """
    Add current_leader_epoch to request
    """
    API_KEY = 2
    API_VERSION = 4
    RESPONSE_TYPE = OffsetResponse_v4
    SCHEMA = Schema(
        ('replica_id', Int32),
        ('isolation_level', Int8),  # <- added isolation_level
        ('topics', Array(
            ('topic', String('utf-8')),
            ('partitions', Array(
                ('partition', Int32),
                ('current_leader_epoch', Int64),
                ('timestamp', Int64)))))
    )
    DEFAULTS = {
        'replica_id': -1
    }


class OffsetRequest_v5(Request):
    API_KEY = 2
    API_VERSION = 5
    RESPONSE_TYPE = OffsetResponse_v5
    SCHEMA = OffsetRequest_v4.SCHEMA
    DEFAULTS = {
        'replica_id': -1
    }


OffsetRequest = [
    OffsetRequest_v0, OffsetRequest_v1, OffsetRequest_v2,
    OffsetRequest_v3, OffsetRequest_v4, OffsetRequest_v5,
]
OffsetResponse = [
    OffsetResponse_v0, OffsetResponse_v1, OffsetResponse_v2,
    OffsetResponse_v3, OffsetResponse_v4, OffsetResponse_v5,
]
