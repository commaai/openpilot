# mypy: ignore-errors
# -*- coding: utf-8 -*-
#
# TARGET arch is: []
# WORD_SIZE is: 8
# POINTER_SIZE is: 8
# LONGDOUBLE_SIZE is: 16
#
import ctypes, ctypes.util


class AsDictMixin:
    @classmethod
    def as_dict(cls, self):
        result = {}
        if not isinstance(self, AsDictMixin):
            # not a structure, assume it's already a python object
            return self
        if not hasattr(cls, "_fields_"):
            return result
        # sys.version_info >= (3, 5)
        # for (field, *_) in cls._fields_:  # noqa
        for field_tuple in cls._fields_:  # noqa
            field = field_tuple[0]
            if field.startswith('PADDING_'):
                continue
            value = getattr(self, field)
            type_ = type(value)
            if hasattr(value, "_length_") and hasattr(value, "_type_"):
                # array
                if not hasattr(type_, "as_dict"):
                    value = [v for v in value]
                else:
                    type_ = type_._type_
                    value = [type_.as_dict(v) for v in value]
            elif hasattr(value, "contents") and hasattr(value, "_type_"):
                # pointer
                try:
                    if not hasattr(type_, "as_dict"):
                        value = value.contents
                    else:
                        type_ = type_._type_
                        value = type_.as_dict(value.contents)
                except ValueError:
                    # nullptr
                    value = None
            elif isinstance(value, AsDictMixin):
                # other structure
                value = type_.as_dict(value)
            result[field] = value
        return result


class Structure(ctypes.Structure, AsDictMixin):

    def __init__(self, *args, **kwds):
        # We don't want to use positional arguments fill PADDING_* fields

        args = dict(zip(self.__class__._field_names_(), args))
        args.update(kwds)
        super(Structure, self).__init__(**args)

    @classmethod
    def _field_names_(cls):
        if hasattr(cls, '_fields_'):
            return (f[0] for f in cls._fields_ if not f[0].startswith('PADDING'))
        else:
            return ()

    @classmethod
    def get_type(cls, field):
        for f in cls._fields_:
            if f[0] == field:
                return f[1]
        return None

    @classmethod
    def bind(cls, bound_fields):
        fields = {}
        for name, type_ in cls._fields_:
            if hasattr(type_, "restype"):
                if name in bound_fields:
                    if bound_fields[name] is None:
                        fields[name] = type_()
                    else:
                        # use a closure to capture the callback from the loop scope
                        fields[name] = (
                            type_((lambda callback: lambda *args: callback(*args))(
                                bound_fields[name]))
                        )
                    del bound_fields[name]
                else:
                    # default callback implementation (does nothing)
                    try:
                        default_ = type_(0).restype().value
                    except TypeError:
                        default_ = None
                    fields[name] = type_((
                        lambda default_: lambda *args: default_)(default_))
            else:
                # not a callback function, use default initialization
                if name in bound_fields:
                    fields[name] = bound_fields[name]
                    del bound_fields[name]
                else:
                    fields[name] = type_()
        if len(bound_fields) != 0:
            raise ValueError(
                "Cannot bind the following unknown callback(s) {}.{}".format(
                    cls.__name__, bound_fields.keys()
            ))
        return cls(**fields)


class Union(ctypes.Union, AsDictMixin):
    pass



c_int128 = ctypes.c_ubyte*16
c_uint128 = c_int128
void = None
if ctypes.sizeof(ctypes.c_longdouble) == 16:
    c_long_double_t = ctypes.c_longdouble
else:
    c_long_double_t = ctypes.c_ubyte*16

def string_cast(char_pointer, encoding='utf-8', errors='strict'):
    value = ctypes.cast(char_pointer, ctypes.c_char_p).value
    if value is not None and encoding is not None:
        value = value.decode(encoding, errors=errors)
    return value


def char_pointer_cast(string, encoding='utf-8'):
    if encoding is not None:
        try:
            string = string.encode(encoding)
        except AttributeError:
            # In Python3, bytes has no encode attribute
            pass
    string = ctypes.c_char_p(string)
    return ctypes.cast(string, ctypes.POINTER(ctypes.c_char))



_libraries = {}
_libraries['libcuda.so'] = ctypes.CDLL(ctypes.util.find_library('cuda'))


cuuint32_t = ctypes.c_uint32
cuuint64_t = ctypes.c_uint64
CUdeviceptr_v2 = ctypes.c_uint64
CUdeviceptr = ctypes.c_uint64
CUdevice_v1 = ctypes.c_int32
CUdevice = ctypes.c_int32
class struct_CUctx_st(Structure):
    pass

CUcontext = ctypes.POINTER(struct_CUctx_st)
class struct_CUmod_st(Structure):
    pass

CUmodule = ctypes.POINTER(struct_CUmod_st)
class struct_CUfunc_st(Structure):
    pass

CUfunction = ctypes.POINTER(struct_CUfunc_st)
class struct_CUarray_st(Structure):
    pass

CUarray = ctypes.POINTER(struct_CUarray_st)
class struct_CUmipmappedArray_st(Structure):
    pass

CUmipmappedArray = ctypes.POINTER(struct_CUmipmappedArray_st)
class struct_CUtexref_st(Structure):
    pass

CUtexref = ctypes.POINTER(struct_CUtexref_st)
class struct_CUsurfref_st(Structure):
    pass

CUsurfref = ctypes.POINTER(struct_CUsurfref_st)
class struct_CUevent_st(Structure):
    pass

CUevent = ctypes.POINTER(struct_CUevent_st)
class struct_CUstream_st(Structure):
    pass

CUstream = ctypes.POINTER(struct_CUstream_st)
class struct_CUgraphicsResource_st(Structure):
    pass

CUgraphicsResource = ctypes.POINTER(struct_CUgraphicsResource_st)
CUtexObject_v1 = ctypes.c_uint64
CUtexObject = ctypes.c_uint64
CUsurfObject_v1 = ctypes.c_uint64
CUsurfObject = ctypes.c_uint64
class struct_CUextMemory_st(Structure):
    pass

CUexternalMemory = ctypes.POINTER(struct_CUextMemory_st)
class struct_CUextSemaphore_st(Structure):
    pass

CUexternalSemaphore = ctypes.POINTER(struct_CUextSemaphore_st)
class struct_CUgraph_st(Structure):
    pass

CUgraph = ctypes.POINTER(struct_CUgraph_st)
class struct_CUgraphNode_st(Structure):
    pass

CUgraphNode = ctypes.POINTER(struct_CUgraphNode_st)
class struct_CUgraphExec_st(Structure):
    pass

CUgraphExec = ctypes.POINTER(struct_CUgraphExec_st)
class struct_CUmemPoolHandle_st(Structure):
    pass

CUmemoryPool = ctypes.POINTER(struct_CUmemPoolHandle_st)
class struct_CUuserObject_st(Structure):
    pass

CUuserObject = ctypes.POINTER(struct_CUuserObject_st)
class struct_CUuuid_st(Structure):
    pass

struct_CUuuid_st._pack_ = 1 # source:False
struct_CUuuid_st._fields_ = [
    ('bytes', ctypes.c_char * 16),
]

CUuuid = struct_CUuuid_st
class struct_CUipcEventHandle_st(Structure):
    pass

struct_CUipcEventHandle_st._pack_ = 1 # source:False
struct_CUipcEventHandle_st._fields_ = [
    ('reserved', ctypes.c_char * 64),
]

CUipcEventHandle_v1 = struct_CUipcEventHandle_st
CUipcEventHandle = struct_CUipcEventHandle_st
class struct_CUipcMemHandle_st(Structure):
    pass

struct_CUipcMemHandle_st._pack_ = 1 # source:False
struct_CUipcMemHandle_st._fields_ = [
    ('reserved', ctypes.c_char * 64),
]

CUipcMemHandle_v1 = struct_CUipcMemHandle_st
CUipcMemHandle = struct_CUipcMemHandle_st

# values for enumeration 'CUipcMem_flags_enum'
CUipcMem_flags_enum__enumvalues = {
    1: 'CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS',
}
CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 1
CUipcMem_flags_enum = ctypes.c_uint32 # enum
CUipcMem_flags = CUipcMem_flags_enum
CUipcMem_flags__enumvalues = CUipcMem_flags_enum__enumvalues

# values for enumeration 'CUmemAttach_flags_enum'
CUmemAttach_flags_enum__enumvalues = {
    1: 'CU_MEM_ATTACH_GLOBAL',
    2: 'CU_MEM_ATTACH_HOST',
    4: 'CU_MEM_ATTACH_SINGLE',
}
CU_MEM_ATTACH_GLOBAL = 1
CU_MEM_ATTACH_HOST = 2
CU_MEM_ATTACH_SINGLE = 4
CUmemAttach_flags_enum = ctypes.c_uint32 # enum
CUmemAttach_flags = CUmemAttach_flags_enum
CUmemAttach_flags__enumvalues = CUmemAttach_flags_enum__enumvalues

# values for enumeration 'CUctx_flags_enum'
CUctx_flags_enum__enumvalues = {
    0: 'CU_CTX_SCHED_AUTO',
    1: 'CU_CTX_SCHED_SPIN',
    2: 'CU_CTX_SCHED_YIELD',
    4: 'CU_CTX_SCHED_BLOCKING_SYNC',
    4: 'CU_CTX_BLOCKING_SYNC',
    7: 'CU_CTX_SCHED_MASK',
    8: 'CU_CTX_MAP_HOST',
    16: 'CU_CTX_LMEM_RESIZE_TO_MAX',
    31: 'CU_CTX_FLAGS_MASK',
}
CU_CTX_SCHED_AUTO = 0
CU_CTX_SCHED_SPIN = 1
CU_CTX_SCHED_YIELD = 2
CU_CTX_SCHED_BLOCKING_SYNC = 4
CU_CTX_BLOCKING_SYNC = 4
CU_CTX_SCHED_MASK = 7
CU_CTX_MAP_HOST = 8
CU_CTX_LMEM_RESIZE_TO_MAX = 16
CU_CTX_FLAGS_MASK = 31
CUctx_flags_enum = ctypes.c_uint32 # enum
CUctx_flags = CUctx_flags_enum
CUctx_flags__enumvalues = CUctx_flags_enum__enumvalues

# values for enumeration 'CUstream_flags_enum'
CUstream_flags_enum__enumvalues = {
    0: 'CU_STREAM_DEFAULT',
    1: 'CU_STREAM_NON_BLOCKING',
}
CU_STREAM_DEFAULT = 0
CU_STREAM_NON_BLOCKING = 1
CUstream_flags_enum = ctypes.c_uint32 # enum
CUstream_flags = CUstream_flags_enum
CUstream_flags__enumvalues = CUstream_flags_enum__enumvalues

# values for enumeration 'CUevent_flags_enum'
CUevent_flags_enum__enumvalues = {
    0: 'CU_EVENT_DEFAULT',
    1: 'CU_EVENT_BLOCKING_SYNC',
    2: 'CU_EVENT_DISABLE_TIMING',
    4: 'CU_EVENT_INTERPROCESS',
}
CU_EVENT_DEFAULT = 0
CU_EVENT_BLOCKING_SYNC = 1
CU_EVENT_DISABLE_TIMING = 2
CU_EVENT_INTERPROCESS = 4
CUevent_flags_enum = ctypes.c_uint32 # enum
CUevent_flags = CUevent_flags_enum
CUevent_flags__enumvalues = CUevent_flags_enum__enumvalues

# values for enumeration 'CUevent_record_flags_enum'
CUevent_record_flags_enum__enumvalues = {
    0: 'CU_EVENT_RECORD_DEFAULT',
    1: 'CU_EVENT_RECORD_EXTERNAL',
}
CU_EVENT_RECORD_DEFAULT = 0
CU_EVENT_RECORD_EXTERNAL = 1
CUevent_record_flags_enum = ctypes.c_uint32 # enum
CUevent_record_flags = CUevent_record_flags_enum
CUevent_record_flags__enumvalues = CUevent_record_flags_enum__enumvalues

# values for enumeration 'CUevent_wait_flags_enum'
CUevent_wait_flags_enum__enumvalues = {
    0: 'CU_EVENT_WAIT_DEFAULT',
    1: 'CU_EVENT_WAIT_EXTERNAL',
}
CU_EVENT_WAIT_DEFAULT = 0
CU_EVENT_WAIT_EXTERNAL = 1
CUevent_wait_flags_enum = ctypes.c_uint32 # enum
CUevent_wait_flags = CUevent_wait_flags_enum
CUevent_wait_flags__enumvalues = CUevent_wait_flags_enum__enumvalues

# values for enumeration 'CUstreamWaitValue_flags_enum'
CUstreamWaitValue_flags_enum__enumvalues = {
    0: 'CU_STREAM_WAIT_VALUE_GEQ',
    1: 'CU_STREAM_WAIT_VALUE_EQ',
    2: 'CU_STREAM_WAIT_VALUE_AND',
    3: 'CU_STREAM_WAIT_VALUE_NOR',
    1073741824: 'CU_STREAM_WAIT_VALUE_FLUSH',
}
CU_STREAM_WAIT_VALUE_GEQ = 0
CU_STREAM_WAIT_VALUE_EQ = 1
CU_STREAM_WAIT_VALUE_AND = 2
CU_STREAM_WAIT_VALUE_NOR = 3
CU_STREAM_WAIT_VALUE_FLUSH = 1073741824
CUstreamWaitValue_flags_enum = ctypes.c_uint32 # enum
CUstreamWaitValue_flags = CUstreamWaitValue_flags_enum
CUstreamWaitValue_flags__enumvalues = CUstreamWaitValue_flags_enum__enumvalues

# values for enumeration 'CUstreamWriteValue_flags_enum'
CUstreamWriteValue_flags_enum__enumvalues = {
    0: 'CU_STREAM_WRITE_VALUE_DEFAULT',
    1: 'CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER',
}
CU_STREAM_WRITE_VALUE_DEFAULT = 0
CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER = 1
CUstreamWriteValue_flags_enum = ctypes.c_uint32 # enum
CUstreamWriteValue_flags = CUstreamWriteValue_flags_enum
CUstreamWriteValue_flags__enumvalues = CUstreamWriteValue_flags_enum__enumvalues

# values for enumeration 'CUstreamBatchMemOpType_enum'
CUstreamBatchMemOpType_enum__enumvalues = {
    1: 'CU_STREAM_MEM_OP_WAIT_VALUE_32',
    2: 'CU_STREAM_MEM_OP_WRITE_VALUE_32',
    4: 'CU_STREAM_MEM_OP_WAIT_VALUE_64',
    5: 'CU_STREAM_MEM_OP_WRITE_VALUE_64',
    3: 'CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES',
}
CU_STREAM_MEM_OP_WAIT_VALUE_32 = 1
CU_STREAM_MEM_OP_WRITE_VALUE_32 = 2
CU_STREAM_MEM_OP_WAIT_VALUE_64 = 4
CU_STREAM_MEM_OP_WRITE_VALUE_64 = 5
CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES = 3
CUstreamBatchMemOpType_enum = ctypes.c_uint32 # enum
CUstreamBatchMemOpType = CUstreamBatchMemOpType_enum
CUstreamBatchMemOpType__enumvalues = CUstreamBatchMemOpType_enum__enumvalues
class union_CUstreamBatchMemOpParams_union(Union):
    pass

class struct_CUstreamMemOpWaitValueParams_st(Structure):
    pass

class union_CUstreamMemOpWaitValueParams_st_0(Union):
    pass

union_CUstreamMemOpWaitValueParams_st_0._pack_ = 1 # source:False
union_CUstreamMemOpWaitValueParams_st_0._fields_ = [
    ('value', ctypes.c_uint32),
    ('value64', ctypes.c_uint64),
]

struct_CUstreamMemOpWaitValueParams_st._pack_ = 1 # source:False
struct_CUstreamMemOpWaitValueParams_st._anonymous_ = ('_0',)
struct_CUstreamMemOpWaitValueParams_st._fields_ = [
    ('operation', CUstreamBatchMemOpType),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('address', ctypes.c_uint64),
    ('_0', union_CUstreamMemOpWaitValueParams_st_0),
    ('flags', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('alias', ctypes.c_uint64),
]

class struct_CUstreamMemOpWriteValueParams_st(Structure):
    pass

class union_CUstreamMemOpWriteValueParams_st_0(Union):
    pass

union_CUstreamMemOpWriteValueParams_st_0._pack_ = 1 # source:False
union_CUstreamMemOpWriteValueParams_st_0._fields_ = [
    ('value', ctypes.c_uint32),
    ('value64', ctypes.c_uint64),
]

struct_CUstreamMemOpWriteValueParams_st._pack_ = 1 # source:False
struct_CUstreamMemOpWriteValueParams_st._anonymous_ = ('_0',)
struct_CUstreamMemOpWriteValueParams_st._fields_ = [
    ('operation', CUstreamBatchMemOpType),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('address', ctypes.c_uint64),
    ('_0', union_CUstreamMemOpWriteValueParams_st_0),
    ('flags', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('alias', ctypes.c_uint64),
]

class struct_CUstreamMemOpFlushRemoteWritesParams_st(Structure):
    pass

struct_CUstreamMemOpFlushRemoteWritesParams_st._pack_ = 1 # source:False
struct_CUstreamMemOpFlushRemoteWritesParams_st._fields_ = [
    ('operation', CUstreamBatchMemOpType),
    ('flags', ctypes.c_uint32),
]

union_CUstreamBatchMemOpParams_union._pack_ = 1 # source:False
union_CUstreamBatchMemOpParams_union._fields_ = [
    ('operation', CUstreamBatchMemOpType),
    ('waitValue', struct_CUstreamMemOpWaitValueParams_st),
    ('writeValue', struct_CUstreamMemOpWriteValueParams_st),
    ('flushRemoteWrites', struct_CUstreamMemOpFlushRemoteWritesParams_st),
    ('pad', ctypes.c_uint64 * 6),
]

CUstreamBatchMemOpParams_v1 = union_CUstreamBatchMemOpParams_union
CUstreamBatchMemOpParams = union_CUstreamBatchMemOpParams_union

# values for enumeration 'CUoccupancy_flags_enum'
CUoccupancy_flags_enum__enumvalues = {
    0: 'CU_OCCUPANCY_DEFAULT',
    1: 'CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE',
}
CU_OCCUPANCY_DEFAULT = 0
CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE = 1
CUoccupancy_flags_enum = ctypes.c_uint32 # enum
CUoccupancy_flags = CUoccupancy_flags_enum
CUoccupancy_flags__enumvalues = CUoccupancy_flags_enum__enumvalues

# values for enumeration 'CUstreamUpdateCaptureDependencies_flags_enum'
CUstreamUpdateCaptureDependencies_flags_enum__enumvalues = {
    0: 'CU_STREAM_ADD_CAPTURE_DEPENDENCIES',
    1: 'CU_STREAM_SET_CAPTURE_DEPENDENCIES',
}
CU_STREAM_ADD_CAPTURE_DEPENDENCIES = 0
CU_STREAM_SET_CAPTURE_DEPENDENCIES = 1
CUstreamUpdateCaptureDependencies_flags_enum = ctypes.c_uint32 # enum
CUstreamUpdateCaptureDependencies_flags = CUstreamUpdateCaptureDependencies_flags_enum
CUstreamUpdateCaptureDependencies_flags__enumvalues = CUstreamUpdateCaptureDependencies_flags_enum__enumvalues

# values for enumeration 'CUarray_format_enum'
CUarray_format_enum__enumvalues = {
    1: 'CU_AD_FORMAT_UNSIGNED_INT8',
    2: 'CU_AD_FORMAT_UNSIGNED_INT16',
    3: 'CU_AD_FORMAT_UNSIGNED_INT32',
    8: 'CU_AD_FORMAT_SIGNED_INT8',
    9: 'CU_AD_FORMAT_SIGNED_INT16',
    10: 'CU_AD_FORMAT_SIGNED_INT32',
    16: 'CU_AD_FORMAT_HALF',
    32: 'CU_AD_FORMAT_FLOAT',
    176: 'CU_AD_FORMAT_NV12',
    192: 'CU_AD_FORMAT_UNORM_INT8X1',
    193: 'CU_AD_FORMAT_UNORM_INT8X2',
    194: 'CU_AD_FORMAT_UNORM_INT8X4',
    195: 'CU_AD_FORMAT_UNORM_INT16X1',
    196: 'CU_AD_FORMAT_UNORM_INT16X2',
    197: 'CU_AD_FORMAT_UNORM_INT16X4',
    198: 'CU_AD_FORMAT_SNORM_INT8X1',
    199: 'CU_AD_FORMAT_SNORM_INT8X2',
    200: 'CU_AD_FORMAT_SNORM_INT8X4',
    201: 'CU_AD_FORMAT_SNORM_INT16X1',
    202: 'CU_AD_FORMAT_SNORM_INT16X2',
    203: 'CU_AD_FORMAT_SNORM_INT16X4',
    145: 'CU_AD_FORMAT_BC1_UNORM',
    146: 'CU_AD_FORMAT_BC1_UNORM_SRGB',
    147: 'CU_AD_FORMAT_BC2_UNORM',
    148: 'CU_AD_FORMAT_BC2_UNORM_SRGB',
    149: 'CU_AD_FORMAT_BC3_UNORM',
    150: 'CU_AD_FORMAT_BC3_UNORM_SRGB',
    151: 'CU_AD_FORMAT_BC4_UNORM',
    152: 'CU_AD_FORMAT_BC4_SNORM',
    153: 'CU_AD_FORMAT_BC5_UNORM',
    154: 'CU_AD_FORMAT_BC5_SNORM',
    155: 'CU_AD_FORMAT_BC6H_UF16',
    156: 'CU_AD_FORMAT_BC6H_SF16',
    157: 'CU_AD_FORMAT_BC7_UNORM',
    158: 'CU_AD_FORMAT_BC7_UNORM_SRGB',
}
CU_AD_FORMAT_UNSIGNED_INT8 = 1
CU_AD_FORMAT_UNSIGNED_INT16 = 2
CU_AD_FORMAT_UNSIGNED_INT32 = 3
CU_AD_FORMAT_SIGNED_INT8 = 8
CU_AD_FORMAT_SIGNED_INT16 = 9
CU_AD_FORMAT_SIGNED_INT32 = 10
CU_AD_FORMAT_HALF = 16
CU_AD_FORMAT_FLOAT = 32
CU_AD_FORMAT_NV12 = 176
CU_AD_FORMAT_UNORM_INT8X1 = 192
CU_AD_FORMAT_UNORM_INT8X2 = 193
CU_AD_FORMAT_UNORM_INT8X4 = 194
CU_AD_FORMAT_UNORM_INT16X1 = 195
CU_AD_FORMAT_UNORM_INT16X2 = 196
CU_AD_FORMAT_UNORM_INT16X4 = 197
CU_AD_FORMAT_SNORM_INT8X1 = 198
CU_AD_FORMAT_SNORM_INT8X2 = 199
CU_AD_FORMAT_SNORM_INT8X4 = 200
CU_AD_FORMAT_SNORM_INT16X1 = 201
CU_AD_FORMAT_SNORM_INT16X2 = 202
CU_AD_FORMAT_SNORM_INT16X4 = 203
CU_AD_FORMAT_BC1_UNORM = 145
CU_AD_FORMAT_BC1_UNORM_SRGB = 146
CU_AD_FORMAT_BC2_UNORM = 147
CU_AD_FORMAT_BC2_UNORM_SRGB = 148
CU_AD_FORMAT_BC3_UNORM = 149
CU_AD_FORMAT_BC3_UNORM_SRGB = 150
CU_AD_FORMAT_BC4_UNORM = 151
CU_AD_FORMAT_BC4_SNORM = 152
CU_AD_FORMAT_BC5_UNORM = 153
CU_AD_FORMAT_BC5_SNORM = 154
CU_AD_FORMAT_BC6H_UF16 = 155
CU_AD_FORMAT_BC6H_SF16 = 156
CU_AD_FORMAT_BC7_UNORM = 157
CU_AD_FORMAT_BC7_UNORM_SRGB = 158
CUarray_format_enum = ctypes.c_uint32 # enum
CUarray_format = CUarray_format_enum
CUarray_format__enumvalues = CUarray_format_enum__enumvalues

# values for enumeration 'CUaddress_mode_enum'
CUaddress_mode_enum__enumvalues = {
    0: 'CU_TR_ADDRESS_MODE_WRAP',
    1: 'CU_TR_ADDRESS_MODE_CLAMP',
    2: 'CU_TR_ADDRESS_MODE_MIRROR',
    3: 'CU_TR_ADDRESS_MODE_BORDER',
}
CU_TR_ADDRESS_MODE_WRAP = 0
CU_TR_ADDRESS_MODE_CLAMP = 1
CU_TR_ADDRESS_MODE_MIRROR = 2
CU_TR_ADDRESS_MODE_BORDER = 3
CUaddress_mode_enum = ctypes.c_uint32 # enum
CUaddress_mode = CUaddress_mode_enum
CUaddress_mode__enumvalues = CUaddress_mode_enum__enumvalues

# values for enumeration 'CUfilter_mode_enum'
CUfilter_mode_enum__enumvalues = {
    0: 'CU_TR_FILTER_MODE_POINT',
    1: 'CU_TR_FILTER_MODE_LINEAR',
}
CU_TR_FILTER_MODE_POINT = 0
CU_TR_FILTER_MODE_LINEAR = 1
CUfilter_mode_enum = ctypes.c_uint32 # enum
CUfilter_mode = CUfilter_mode_enum
CUfilter_mode__enumvalues = CUfilter_mode_enum__enumvalues

# values for enumeration 'CUdevice_attribute_enum'
CUdevice_attribute_enum__enumvalues = {
    1: 'CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK',
    2: 'CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X',
    3: 'CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y',
    4: 'CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z',
    5: 'CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X',
    6: 'CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y',
    7: 'CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z',
    8: 'CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK',
    8: 'CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK',
    9: 'CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY',
    10: 'CU_DEVICE_ATTRIBUTE_WARP_SIZE',
    11: 'CU_DEVICE_ATTRIBUTE_MAX_PITCH',
    12: 'CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK',
    12: 'CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK',
    13: 'CU_DEVICE_ATTRIBUTE_CLOCK_RATE',
    14: 'CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT',
    15: 'CU_DEVICE_ATTRIBUTE_GPU_OVERLAP',
    16: 'CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT',
    17: 'CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT',
    18: 'CU_DEVICE_ATTRIBUTE_INTEGRATED',
    19: 'CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY',
    20: 'CU_DEVICE_ATTRIBUTE_COMPUTE_MODE',
    21: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH',
    22: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH',
    23: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT',
    24: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH',
    25: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT',
    26: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH',
    27: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH',
    28: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT',
    29: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS',
    27: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH',
    28: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT',
    29: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES',
    30: 'CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT',
    31: 'CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS',
    32: 'CU_DEVICE_ATTRIBUTE_ECC_ENABLED',
    33: 'CU_DEVICE_ATTRIBUTE_PCI_BUS_ID',
    34: 'CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID',
    35: 'CU_DEVICE_ATTRIBUTE_TCC_DRIVER',
    36: 'CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE',
    37: 'CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH',
    38: 'CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE',
    39: 'CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR',
    40: 'CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT',
    41: 'CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING',
    42: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH',
    43: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS',
    44: 'CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER',
    45: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH',
    46: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT',
    47: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE',
    48: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE',
    49: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE',
    50: 'CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID',
    51: 'CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT',
    52: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH',
    53: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH',
    54: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS',
    55: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH',
    56: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH',
    57: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT',
    58: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH',
    59: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT',
    60: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH',
    61: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH',
    62: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS',
    63: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH',
    64: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT',
    65: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS',
    66: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH',
    67: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH',
    68: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS',
    69: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH',
    70: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH',
    71: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT',
    72: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH',
    73: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH',
    74: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT',
    75: 'CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR',
    76: 'CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR',
    77: 'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH',
    78: 'CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED',
    79: 'CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED',
    80: 'CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED',
    81: 'CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR',
    82: 'CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR',
    83: 'CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY',
    84: 'CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD',
    85: 'CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID',
    86: 'CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED',
    87: 'CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO',
    88: 'CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS',
    89: 'CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS',
    90: 'CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED',
    91: 'CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM',
    92: 'CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS',
    93: 'CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS',
    94: 'CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR',
    95: 'CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH',
    96: 'CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH',
    97: 'CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN',
    98: 'CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES',
    99: 'CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED',
    100: 'CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES',
    101: 'CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST',
    102: 'CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED',
    102: 'CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED',
    103: 'CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED',
    104: 'CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED',
    105: 'CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED',
    106: 'CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR',
    107: 'CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED',
    108: 'CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE',
    109: 'CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE',
    110: 'CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED',
    111: 'CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK',
    112: 'CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED',
    113: 'CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED',
    114: 'CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED',
    115: 'CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED',
    116: 'CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED',
    117: 'CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS',
    118: 'CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING',
    119: 'CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES',
    120: 'CU_DEVICE_ATTRIBUTE_MAX',
}
CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1
CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2
CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3
CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4
CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5
CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6
CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7
CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8
CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK = 8
CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9
CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10
CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11
CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12
CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK = 12
CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13
CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14
CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15
CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17
CU_DEVICE_ATTRIBUTE_INTEGRATED = 18
CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19
CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH = 21
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH = 22
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT = 23
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH = 24
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT = 25
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH = 26
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH = 27
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = 28
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS = 29
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH = 27
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT = 28
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES = 29
CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT = 30
CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 31
CU_DEVICE_ATTRIBUTE_ECC_ENABLED = 32
CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33
CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34
CU_DEVICE_ATTRIBUTE_TCC_DRIVER = 35
CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36
CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37
CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38
CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39
CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40
CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH = 42
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS = 43
CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER = 44
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH = 45
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT = 46
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = 47
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = 48
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = 49
CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50
CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT = 51
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH = 52
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = 53
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = 54
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH = 55
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH = 56
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT = 57
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH = 58
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT = 59
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH = 60
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH = 61
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS = 62
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH = 63
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT = 64
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS = 65
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH = 66
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = 67
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = 68
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH = 69
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH = 70
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = 71
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH = 72
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = 73
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = 74
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = 77
CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = 78
CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = 79
CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = 80
CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81
CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 82
CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = 83
CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD = 84
CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID = 85
CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED = 86
CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = 87
CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = 88
CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = 89
CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = 90
CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91
CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS = 92
CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS = 93
CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR = 94
CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = 95
CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH = 96
CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97
CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES = 98
CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED = 99
CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = 100
CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST = 101
CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED = 102
CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED = 102
CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED = 103
CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED = 104
CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED = 105
CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR = 106
CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED = 107
CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE = 108
CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE = 109
CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED = 110
CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK = 111
CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED = 112
CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED = 113
CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED = 114
CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED = 115
CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED = 116
CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS = 117
CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING = 118
CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES = 119
CU_DEVICE_ATTRIBUTE_MAX = 120
CUdevice_attribute_enum = ctypes.c_uint32 # enum
CUdevice_attribute = CUdevice_attribute_enum
CUdevice_attribute__enumvalues = CUdevice_attribute_enum__enumvalues
class struct_CUdevprop_st(Structure):
    pass

struct_CUdevprop_st._pack_ = 1 # source:False
struct_CUdevprop_st._fields_ = [
    ('maxThreadsPerBlock', ctypes.c_int32),
    ('maxThreadsDim', ctypes.c_int32 * 3),
    ('maxGridSize', ctypes.c_int32 * 3),
    ('sharedMemPerBlock', ctypes.c_int32),
    ('totalConstantMemory', ctypes.c_int32),
    ('SIMDWidth', ctypes.c_int32),
    ('memPitch', ctypes.c_int32),
    ('regsPerBlock', ctypes.c_int32),
    ('clockRate', ctypes.c_int32),
    ('textureAlign', ctypes.c_int32),
]

CUdevprop_v1 = struct_CUdevprop_st
CUdevprop = struct_CUdevprop_st

# values for enumeration 'CUpointer_attribute_enum'
CUpointer_attribute_enum__enumvalues = {
    1: 'CU_POINTER_ATTRIBUTE_CONTEXT',
    2: 'CU_POINTER_ATTRIBUTE_MEMORY_TYPE',
    3: 'CU_POINTER_ATTRIBUTE_DEVICE_POINTER',
    4: 'CU_POINTER_ATTRIBUTE_HOST_POINTER',
    5: 'CU_POINTER_ATTRIBUTE_P2P_TOKENS',
    6: 'CU_POINTER_ATTRIBUTE_SYNC_MEMOPS',
    7: 'CU_POINTER_ATTRIBUTE_BUFFER_ID',
    8: 'CU_POINTER_ATTRIBUTE_IS_MANAGED',
    9: 'CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL',
    10: 'CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE',
    11: 'CU_POINTER_ATTRIBUTE_RANGE_START_ADDR',
    12: 'CU_POINTER_ATTRIBUTE_RANGE_SIZE',
    13: 'CU_POINTER_ATTRIBUTE_MAPPED',
    14: 'CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES',
    15: 'CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE',
    16: 'CU_POINTER_ATTRIBUTE_ACCESS_FLAGS',
    17: 'CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE',
}
CU_POINTER_ATTRIBUTE_CONTEXT = 1
CU_POINTER_ATTRIBUTE_MEMORY_TYPE = 2
CU_POINTER_ATTRIBUTE_DEVICE_POINTER = 3
CU_POINTER_ATTRIBUTE_HOST_POINTER = 4
CU_POINTER_ATTRIBUTE_P2P_TOKENS = 5
CU_POINTER_ATTRIBUTE_SYNC_MEMOPS = 6
CU_POINTER_ATTRIBUTE_BUFFER_ID = 7
CU_POINTER_ATTRIBUTE_IS_MANAGED = 8
CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL = 9
CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE = 10
CU_POINTER_ATTRIBUTE_RANGE_START_ADDR = 11
CU_POINTER_ATTRIBUTE_RANGE_SIZE = 12
CU_POINTER_ATTRIBUTE_MAPPED = 13
CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES = 14
CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE = 15
CU_POINTER_ATTRIBUTE_ACCESS_FLAGS = 16
CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE = 17
CUpointer_attribute_enum = ctypes.c_uint32 # enum
CUpointer_attribute = CUpointer_attribute_enum
CUpointer_attribute__enumvalues = CUpointer_attribute_enum__enumvalues

# values for enumeration 'CUfunction_attribute_enum'
CUfunction_attribute_enum__enumvalues = {
    0: 'CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK',
    1: 'CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES',
    2: 'CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES',
    3: 'CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES',
    4: 'CU_FUNC_ATTRIBUTE_NUM_REGS',
    5: 'CU_FUNC_ATTRIBUTE_PTX_VERSION',
    6: 'CU_FUNC_ATTRIBUTE_BINARY_VERSION',
    7: 'CU_FUNC_ATTRIBUTE_CACHE_MODE_CA',
    8: 'CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES',
    9: 'CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT',
    10: 'CU_FUNC_ATTRIBUTE_MAX',
}
CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0
CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1
CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES = 2
CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3
CU_FUNC_ATTRIBUTE_NUM_REGS = 4
CU_FUNC_ATTRIBUTE_PTX_VERSION = 5
CU_FUNC_ATTRIBUTE_BINARY_VERSION = 6
CU_FUNC_ATTRIBUTE_CACHE_MODE_CA = 7
CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8
CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 9
CU_FUNC_ATTRIBUTE_MAX = 10
CUfunction_attribute_enum = ctypes.c_uint32 # enum
CUfunction_attribute = CUfunction_attribute_enum
CUfunction_attribute__enumvalues = CUfunction_attribute_enum__enumvalues

# values for enumeration 'CUfunc_cache_enum'
CUfunc_cache_enum__enumvalues = {
    0: 'CU_FUNC_CACHE_PREFER_NONE',
    1: 'CU_FUNC_CACHE_PREFER_SHARED',
    2: 'CU_FUNC_CACHE_PREFER_L1',
    3: 'CU_FUNC_CACHE_PREFER_EQUAL',
}
CU_FUNC_CACHE_PREFER_NONE = 0
CU_FUNC_CACHE_PREFER_SHARED = 1
CU_FUNC_CACHE_PREFER_L1 = 2
CU_FUNC_CACHE_PREFER_EQUAL = 3
CUfunc_cache_enum = ctypes.c_uint32 # enum
CUfunc_cache = CUfunc_cache_enum
CUfunc_cache__enumvalues = CUfunc_cache_enum__enumvalues

# values for enumeration 'CUsharedconfig_enum'
CUsharedconfig_enum__enumvalues = {
    0: 'CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE',
    1: 'CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE',
    2: 'CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE',
}
CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE = 0
CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE = 1
CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE = 2
CUsharedconfig_enum = ctypes.c_uint32 # enum
CUsharedconfig = CUsharedconfig_enum
CUsharedconfig__enumvalues = CUsharedconfig_enum__enumvalues

# values for enumeration 'CUshared_carveout_enum'
CUshared_carveout_enum__enumvalues = {
    -1: 'CU_SHAREDMEM_CARVEOUT_DEFAULT',
    100: 'CU_SHAREDMEM_CARVEOUT_MAX_SHARED',
    0: 'CU_SHAREDMEM_CARVEOUT_MAX_L1',
}
CU_SHAREDMEM_CARVEOUT_DEFAULT = -1
CU_SHAREDMEM_CARVEOUT_MAX_SHARED = 100
CU_SHAREDMEM_CARVEOUT_MAX_L1 = 0
CUshared_carveout_enum = ctypes.c_int32 # enum
CUshared_carveout = CUshared_carveout_enum
CUshared_carveout__enumvalues = CUshared_carveout_enum__enumvalues

# values for enumeration 'CUmemorytype_enum'
CUmemorytype_enum__enumvalues = {
    1: 'CU_MEMORYTYPE_HOST',
    2: 'CU_MEMORYTYPE_DEVICE',
    3: 'CU_MEMORYTYPE_ARRAY',
    4: 'CU_MEMORYTYPE_UNIFIED',
}
CU_MEMORYTYPE_HOST = 1
CU_MEMORYTYPE_DEVICE = 2
CU_MEMORYTYPE_ARRAY = 3
CU_MEMORYTYPE_UNIFIED = 4
CUmemorytype_enum = ctypes.c_uint32 # enum
CUmemorytype = CUmemorytype_enum
CUmemorytype__enumvalues = CUmemorytype_enum__enumvalues

# values for enumeration 'CUcomputemode_enum'
CUcomputemode_enum__enumvalues = {
    0: 'CU_COMPUTEMODE_DEFAULT',
    2: 'CU_COMPUTEMODE_PROHIBITED',
    3: 'CU_COMPUTEMODE_EXCLUSIVE_PROCESS',
}
CU_COMPUTEMODE_DEFAULT = 0
CU_COMPUTEMODE_PROHIBITED = 2
CU_COMPUTEMODE_EXCLUSIVE_PROCESS = 3
CUcomputemode_enum = ctypes.c_uint32 # enum
CUcomputemode = CUcomputemode_enum
CUcomputemode__enumvalues = CUcomputemode_enum__enumvalues

# values for enumeration 'CUmem_advise_enum'
CUmem_advise_enum__enumvalues = {
    1: 'CU_MEM_ADVISE_SET_READ_MOSTLY',
    2: 'CU_MEM_ADVISE_UNSET_READ_MOSTLY',
    3: 'CU_MEM_ADVISE_SET_PREFERRED_LOCATION',
    4: 'CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION',
    5: 'CU_MEM_ADVISE_SET_ACCESSED_BY',
    6: 'CU_MEM_ADVISE_UNSET_ACCESSED_BY',
}
CU_MEM_ADVISE_SET_READ_MOSTLY = 1
CU_MEM_ADVISE_UNSET_READ_MOSTLY = 2
CU_MEM_ADVISE_SET_PREFERRED_LOCATION = 3
CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION = 4
CU_MEM_ADVISE_SET_ACCESSED_BY = 5
CU_MEM_ADVISE_UNSET_ACCESSED_BY = 6
CUmem_advise_enum = ctypes.c_uint32 # enum
CUmem_advise = CUmem_advise_enum
CUmem_advise__enumvalues = CUmem_advise_enum__enumvalues

# values for enumeration 'CUmem_range_attribute_enum'
CUmem_range_attribute_enum__enumvalues = {
    1: 'CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY',
    2: 'CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION',
    3: 'CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY',
    4: 'CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION',
}
CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY = 1
CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION = 2
CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY = 3
CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION = 4
CUmem_range_attribute_enum = ctypes.c_uint32 # enum
CUmem_range_attribute = CUmem_range_attribute_enum
CUmem_range_attribute__enumvalues = CUmem_range_attribute_enum__enumvalues

# values for enumeration 'CUjit_option_enum'
CUjit_option_enum__enumvalues = {
    0: 'CU_JIT_MAX_REGISTERS',
    1: 'CU_JIT_THREADS_PER_BLOCK',
    2: 'CU_JIT_WALL_TIME',
    3: 'CU_JIT_INFO_LOG_BUFFER',
    4: 'CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES',
    5: 'CU_JIT_ERROR_LOG_BUFFER',
    6: 'CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES',
    7: 'CU_JIT_OPTIMIZATION_LEVEL',
    8: 'CU_JIT_TARGET_FROM_CUCONTEXT',
    9: 'CU_JIT_TARGET',
    10: 'CU_JIT_FALLBACK_STRATEGY',
    11: 'CU_JIT_GENERATE_DEBUG_INFO',
    12: 'CU_JIT_LOG_VERBOSE',
    13: 'CU_JIT_GENERATE_LINE_INFO',
    14: 'CU_JIT_CACHE_MODE',
    15: 'CU_JIT_NEW_SM3X_OPT',
    16: 'CU_JIT_FAST_COMPILE',
    17: 'CU_JIT_GLOBAL_SYMBOL_NAMES',
    18: 'CU_JIT_GLOBAL_SYMBOL_ADDRESSES',
    19: 'CU_JIT_GLOBAL_SYMBOL_COUNT',
    20: 'CU_JIT_LTO',
    21: 'CU_JIT_FTZ',
    22: 'CU_JIT_PREC_DIV',
    23: 'CU_JIT_PREC_SQRT',
    24: 'CU_JIT_FMA',
    25: 'CU_JIT_NUM_OPTIONS',
}
CU_JIT_MAX_REGISTERS = 0
CU_JIT_THREADS_PER_BLOCK = 1
CU_JIT_WALL_TIME = 2
CU_JIT_INFO_LOG_BUFFER = 3
CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4
CU_JIT_ERROR_LOG_BUFFER = 5
CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6
CU_JIT_OPTIMIZATION_LEVEL = 7
CU_JIT_TARGET_FROM_CUCONTEXT = 8
CU_JIT_TARGET = 9
CU_JIT_FALLBACK_STRATEGY = 10
CU_JIT_GENERATE_DEBUG_INFO = 11
CU_JIT_LOG_VERBOSE = 12
CU_JIT_GENERATE_LINE_INFO = 13
CU_JIT_CACHE_MODE = 14
CU_JIT_NEW_SM3X_OPT = 15
CU_JIT_FAST_COMPILE = 16
CU_JIT_GLOBAL_SYMBOL_NAMES = 17
CU_JIT_GLOBAL_SYMBOL_ADDRESSES = 18
CU_JIT_GLOBAL_SYMBOL_COUNT = 19
CU_JIT_LTO = 20
CU_JIT_FTZ = 21
CU_JIT_PREC_DIV = 22
CU_JIT_PREC_SQRT = 23
CU_JIT_FMA = 24
CU_JIT_NUM_OPTIONS = 25
CUjit_option_enum = ctypes.c_uint32 # enum
CUjit_option = CUjit_option_enum
CUjit_option__enumvalues = CUjit_option_enum__enumvalues

# values for enumeration 'CUjit_target_enum'
CUjit_target_enum__enumvalues = {
    20: 'CU_TARGET_COMPUTE_20',
    21: 'CU_TARGET_COMPUTE_21',
    30: 'CU_TARGET_COMPUTE_30',
    32: 'CU_TARGET_COMPUTE_32',
    35: 'CU_TARGET_COMPUTE_35',
    37: 'CU_TARGET_COMPUTE_37',
    50: 'CU_TARGET_COMPUTE_50',
    52: 'CU_TARGET_COMPUTE_52',
    53: 'CU_TARGET_COMPUTE_53',
    60: 'CU_TARGET_COMPUTE_60',
    61: 'CU_TARGET_COMPUTE_61',
    62: 'CU_TARGET_COMPUTE_62',
    70: 'CU_TARGET_COMPUTE_70',
    72: 'CU_TARGET_COMPUTE_72',
    75: 'CU_TARGET_COMPUTE_75',
    80: 'CU_TARGET_COMPUTE_80',
    86: 'CU_TARGET_COMPUTE_86',
}
CU_TARGET_COMPUTE_20 = 20
CU_TARGET_COMPUTE_21 = 21
CU_TARGET_COMPUTE_30 = 30
CU_TARGET_COMPUTE_32 = 32
CU_TARGET_COMPUTE_35 = 35
CU_TARGET_COMPUTE_37 = 37
CU_TARGET_COMPUTE_50 = 50
CU_TARGET_COMPUTE_52 = 52
CU_TARGET_COMPUTE_53 = 53
CU_TARGET_COMPUTE_60 = 60
CU_TARGET_COMPUTE_61 = 61
CU_TARGET_COMPUTE_62 = 62
CU_TARGET_COMPUTE_70 = 70
CU_TARGET_COMPUTE_72 = 72
CU_TARGET_COMPUTE_75 = 75
CU_TARGET_COMPUTE_80 = 80
CU_TARGET_COMPUTE_86 = 86
CUjit_target_enum = ctypes.c_uint32 # enum
CUjit_target = CUjit_target_enum
CUjit_target__enumvalues = CUjit_target_enum__enumvalues

# values for enumeration 'CUjit_fallback_enum'
CUjit_fallback_enum__enumvalues = {
    0: 'CU_PREFER_PTX',
    1: 'CU_PREFER_BINARY',
}
CU_PREFER_PTX = 0
CU_PREFER_BINARY = 1
CUjit_fallback_enum = ctypes.c_uint32 # enum
CUjit_fallback = CUjit_fallback_enum
CUjit_fallback__enumvalues = CUjit_fallback_enum__enumvalues

# values for enumeration 'CUjit_cacheMode_enum'
CUjit_cacheMode_enum__enumvalues = {
    0: 'CU_JIT_CACHE_OPTION_NONE',
    1: 'CU_JIT_CACHE_OPTION_CG',
    2: 'CU_JIT_CACHE_OPTION_CA',
}
CU_JIT_CACHE_OPTION_NONE = 0
CU_JIT_CACHE_OPTION_CG = 1
CU_JIT_CACHE_OPTION_CA = 2
CUjit_cacheMode_enum = ctypes.c_uint32 # enum
CUjit_cacheMode = CUjit_cacheMode_enum
CUjit_cacheMode__enumvalues = CUjit_cacheMode_enum__enumvalues

# values for enumeration 'CUjitInputType_enum'
CUjitInputType_enum__enumvalues = {
    0: 'CU_JIT_INPUT_CUBIN',
    1: 'CU_JIT_INPUT_PTX',
    2: 'CU_JIT_INPUT_FATBINARY',
    3: 'CU_JIT_INPUT_OBJECT',
    4: 'CU_JIT_INPUT_LIBRARY',
    5: 'CU_JIT_INPUT_NVVM',
    6: 'CU_JIT_NUM_INPUT_TYPES',
}
CU_JIT_INPUT_CUBIN = 0
CU_JIT_INPUT_PTX = 1
CU_JIT_INPUT_FATBINARY = 2
CU_JIT_INPUT_OBJECT = 3
CU_JIT_INPUT_LIBRARY = 4
CU_JIT_INPUT_NVVM = 5
CU_JIT_NUM_INPUT_TYPES = 6
CUjitInputType_enum = ctypes.c_uint32 # enum
CUjitInputType = CUjitInputType_enum
CUjitInputType__enumvalues = CUjitInputType_enum__enumvalues
class struct_CUlinkState_st(Structure):
    pass

CUlinkState = ctypes.POINTER(struct_CUlinkState_st)

# values for enumeration 'CUgraphicsRegisterFlags_enum'
CUgraphicsRegisterFlags_enum__enumvalues = {
    0: 'CU_GRAPHICS_REGISTER_FLAGS_NONE',
    1: 'CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY',
    2: 'CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD',
    4: 'CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST',
    8: 'CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER',
}
CU_GRAPHICS_REGISTER_FLAGS_NONE = 0
CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY = 1
CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD = 2
CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST = 4
CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER = 8
CUgraphicsRegisterFlags_enum = ctypes.c_uint32 # enum
CUgraphicsRegisterFlags = CUgraphicsRegisterFlags_enum
CUgraphicsRegisterFlags__enumvalues = CUgraphicsRegisterFlags_enum__enumvalues

# values for enumeration 'CUgraphicsMapResourceFlags_enum'
CUgraphicsMapResourceFlags_enum__enumvalues = {
    0: 'CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE',
    1: 'CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY',
    2: 'CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD',
}
CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE = 0
CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY = 1
CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD = 2
CUgraphicsMapResourceFlags_enum = ctypes.c_uint32 # enum
CUgraphicsMapResourceFlags = CUgraphicsMapResourceFlags_enum
CUgraphicsMapResourceFlags__enumvalues = CUgraphicsMapResourceFlags_enum__enumvalues

# values for enumeration 'CUarray_cubemap_face_enum'
CUarray_cubemap_face_enum__enumvalues = {
    0: 'CU_CUBEMAP_FACE_POSITIVE_X',
    1: 'CU_CUBEMAP_FACE_NEGATIVE_X',
    2: 'CU_CUBEMAP_FACE_POSITIVE_Y',
    3: 'CU_CUBEMAP_FACE_NEGATIVE_Y',
    4: 'CU_CUBEMAP_FACE_POSITIVE_Z',
    5: 'CU_CUBEMAP_FACE_NEGATIVE_Z',
}
CU_CUBEMAP_FACE_POSITIVE_X = 0
CU_CUBEMAP_FACE_NEGATIVE_X = 1
CU_CUBEMAP_FACE_POSITIVE_Y = 2
CU_CUBEMAP_FACE_NEGATIVE_Y = 3
CU_CUBEMAP_FACE_POSITIVE_Z = 4
CU_CUBEMAP_FACE_NEGATIVE_Z = 5
CUarray_cubemap_face_enum = ctypes.c_uint32 # enum
CUarray_cubemap_face = CUarray_cubemap_face_enum
CUarray_cubemap_face__enumvalues = CUarray_cubemap_face_enum__enumvalues

# values for enumeration 'CUlimit_enum'
CUlimit_enum__enumvalues = {
    0: 'CU_LIMIT_STACK_SIZE',
    1: 'CU_LIMIT_PRINTF_FIFO_SIZE',
    2: 'CU_LIMIT_MALLOC_HEAP_SIZE',
    3: 'CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH',
    4: 'CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT',
    5: 'CU_LIMIT_MAX_L2_FETCH_GRANULARITY',
    6: 'CU_LIMIT_PERSISTING_L2_CACHE_SIZE',
    7: 'CU_LIMIT_MAX',
}
CU_LIMIT_STACK_SIZE = 0
CU_LIMIT_PRINTF_FIFO_SIZE = 1
CU_LIMIT_MALLOC_HEAP_SIZE = 2
CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH = 3
CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT = 4
CU_LIMIT_MAX_L2_FETCH_GRANULARITY = 5
CU_LIMIT_PERSISTING_L2_CACHE_SIZE = 6
CU_LIMIT_MAX = 7
CUlimit_enum = ctypes.c_uint32 # enum
CUlimit = CUlimit_enum
CUlimit__enumvalues = CUlimit_enum__enumvalues

# values for enumeration 'CUresourcetype_enum'
CUresourcetype_enum__enumvalues = {
    0: 'CU_RESOURCE_TYPE_ARRAY',
    1: 'CU_RESOURCE_TYPE_MIPMAPPED_ARRAY',
    2: 'CU_RESOURCE_TYPE_LINEAR',
    3: 'CU_RESOURCE_TYPE_PITCH2D',
}
CU_RESOURCE_TYPE_ARRAY = 0
CU_RESOURCE_TYPE_MIPMAPPED_ARRAY = 1
CU_RESOURCE_TYPE_LINEAR = 2
CU_RESOURCE_TYPE_PITCH2D = 3
CUresourcetype_enum = ctypes.c_uint32 # enum
CUresourcetype = CUresourcetype_enum
CUresourcetype__enumvalues = CUresourcetype_enum__enumvalues
CUhostFn = ctypes.CFUNCTYPE(None, ctypes.POINTER(None))

# values for enumeration 'CUaccessProperty_enum'
CUaccessProperty_enum__enumvalues = {
    0: 'CU_ACCESS_PROPERTY_NORMAL',
    1: 'CU_ACCESS_PROPERTY_STREAMING',
    2: 'CU_ACCESS_PROPERTY_PERSISTING',
}
CU_ACCESS_PROPERTY_NORMAL = 0
CU_ACCESS_PROPERTY_STREAMING = 1
CU_ACCESS_PROPERTY_PERSISTING = 2
CUaccessProperty_enum = ctypes.c_uint32 # enum
CUaccessProperty = CUaccessProperty_enum
CUaccessProperty__enumvalues = CUaccessProperty_enum__enumvalues
class struct_CUaccessPolicyWindow_st(Structure):
    pass

struct_CUaccessPolicyWindow_st._pack_ = 1 # source:False
struct_CUaccessPolicyWindow_st._fields_ = [
    ('base_ptr', ctypes.POINTER(None)),
    ('num_bytes', ctypes.c_uint64),
    ('hitRatio', ctypes.c_float),
    ('hitProp', CUaccessProperty),
    ('missProp', CUaccessProperty),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

CUaccessPolicyWindow_v1 = struct_CUaccessPolicyWindow_st
CUaccessPolicyWindow = struct_CUaccessPolicyWindow_st
class struct_CUDA_KERNEL_NODE_PARAMS_st(Structure):
    pass

struct_CUDA_KERNEL_NODE_PARAMS_st._pack_ = 1 # source:False
struct_CUDA_KERNEL_NODE_PARAMS_st._fields_ = [
    ('func', ctypes.POINTER(struct_CUfunc_st)),
    ('gridDimX', ctypes.c_uint32),
    ('gridDimY', ctypes.c_uint32),
    ('gridDimZ', ctypes.c_uint32),
    ('blockDimX', ctypes.c_uint32),
    ('blockDimY', ctypes.c_uint32),
    ('blockDimZ', ctypes.c_uint32),
    ('sharedMemBytes', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('kernelParams', ctypes.POINTER(ctypes.POINTER(None))),
    ('extra', ctypes.POINTER(ctypes.POINTER(None))),
]

CUDA_KERNEL_NODE_PARAMS_v1 = struct_CUDA_KERNEL_NODE_PARAMS_st
CUDA_KERNEL_NODE_PARAMS = struct_CUDA_KERNEL_NODE_PARAMS_st
class struct_CUDA_MEMSET_NODE_PARAMS_st(Structure):
    pass

struct_CUDA_MEMSET_NODE_PARAMS_st._pack_ = 1 # source:False
struct_CUDA_MEMSET_NODE_PARAMS_st._fields_ = [
    ('dst', ctypes.c_uint64),
    ('pitch', ctypes.c_uint64),
    ('value', ctypes.c_uint32),
    ('elementSize', ctypes.c_uint32),
    ('width', ctypes.c_uint64),
    ('height', ctypes.c_uint64),
]

CUDA_MEMSET_NODE_PARAMS_v1 = struct_CUDA_MEMSET_NODE_PARAMS_st
CUDA_MEMSET_NODE_PARAMS = struct_CUDA_MEMSET_NODE_PARAMS_st
class struct_CUDA_HOST_NODE_PARAMS_st(Structure):
    pass

struct_CUDA_HOST_NODE_PARAMS_st._pack_ = 1 # source:False
struct_CUDA_HOST_NODE_PARAMS_st._fields_ = [
    ('fn', ctypes.CFUNCTYPE(None, ctypes.POINTER(None))),
    ('userData', ctypes.POINTER(None)),
]

CUDA_HOST_NODE_PARAMS_v1 = struct_CUDA_HOST_NODE_PARAMS_st
CUDA_HOST_NODE_PARAMS = struct_CUDA_HOST_NODE_PARAMS_st

# values for enumeration 'CUgraphNodeType_enum'
CUgraphNodeType_enum__enumvalues = {
    0: 'CU_GRAPH_NODE_TYPE_KERNEL',
    1: 'CU_GRAPH_NODE_TYPE_MEMCPY',
    2: 'CU_GRAPH_NODE_TYPE_MEMSET',
    3: 'CU_GRAPH_NODE_TYPE_HOST',
    4: 'CU_GRAPH_NODE_TYPE_GRAPH',
    5: 'CU_GRAPH_NODE_TYPE_EMPTY',
    6: 'CU_GRAPH_NODE_TYPE_WAIT_EVENT',
    7: 'CU_GRAPH_NODE_TYPE_EVENT_RECORD',
    8: 'CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL',
    9: 'CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT',
    10: 'CU_GRAPH_NODE_TYPE_MEM_ALLOC',
    11: 'CU_GRAPH_NODE_TYPE_MEM_FREE',
}
CU_GRAPH_NODE_TYPE_KERNEL = 0
CU_GRAPH_NODE_TYPE_MEMCPY = 1
CU_GRAPH_NODE_TYPE_MEMSET = 2
CU_GRAPH_NODE_TYPE_HOST = 3
CU_GRAPH_NODE_TYPE_GRAPH = 4
CU_GRAPH_NODE_TYPE_EMPTY = 5
CU_GRAPH_NODE_TYPE_WAIT_EVENT = 6
CU_GRAPH_NODE_TYPE_EVENT_RECORD = 7
CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL = 8
CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT = 9
CU_GRAPH_NODE_TYPE_MEM_ALLOC = 10
CU_GRAPH_NODE_TYPE_MEM_FREE = 11
CUgraphNodeType_enum = ctypes.c_uint32 # enum
CUgraphNodeType = CUgraphNodeType_enum
CUgraphNodeType__enumvalues = CUgraphNodeType_enum__enumvalues

# values for enumeration 'CUsynchronizationPolicy_enum'
CUsynchronizationPolicy_enum__enumvalues = {
    1: 'CU_SYNC_POLICY_AUTO',
    2: 'CU_SYNC_POLICY_SPIN',
    3: 'CU_SYNC_POLICY_YIELD',
    4: 'CU_SYNC_POLICY_BLOCKING_SYNC',
}
CU_SYNC_POLICY_AUTO = 1
CU_SYNC_POLICY_SPIN = 2
CU_SYNC_POLICY_YIELD = 3
CU_SYNC_POLICY_BLOCKING_SYNC = 4
CUsynchronizationPolicy_enum = ctypes.c_uint32 # enum
CUsynchronizationPolicy = CUsynchronizationPolicy_enum
CUsynchronizationPolicy__enumvalues = CUsynchronizationPolicy_enum__enumvalues

# values for enumeration 'CUkernelNodeAttrID_enum'
CUkernelNodeAttrID_enum__enumvalues = {
    1: 'CU_KERNEL_NODE_ATTRIBUTE_ACCESS_POLICY_WINDOW',
    2: 'CU_KERNEL_NODE_ATTRIBUTE_COOPERATIVE',
}
CU_KERNEL_NODE_ATTRIBUTE_ACCESS_POLICY_WINDOW = 1
CU_KERNEL_NODE_ATTRIBUTE_COOPERATIVE = 2
CUkernelNodeAttrID_enum = ctypes.c_uint32 # enum
CUkernelNodeAttrID = CUkernelNodeAttrID_enum
CUkernelNodeAttrID__enumvalues = CUkernelNodeAttrID_enum__enumvalues
class union_CUkernelNodeAttrValue_union(Union):
    pass

union_CUkernelNodeAttrValue_union._pack_ = 1 # source:False
union_CUkernelNodeAttrValue_union._fields_ = [
    ('accessPolicyWindow', CUaccessPolicyWindow),
    ('cooperative', ctypes.c_int32),
    ('PADDING_0', ctypes.c_ubyte * 28),
]

CUkernelNodeAttrValue_v1 = union_CUkernelNodeAttrValue_union
CUkernelNodeAttrValue = union_CUkernelNodeAttrValue_union

# values for enumeration 'CUstreamCaptureStatus_enum'
CUstreamCaptureStatus_enum__enumvalues = {
    0: 'CU_STREAM_CAPTURE_STATUS_NONE',
    1: 'CU_STREAM_CAPTURE_STATUS_ACTIVE',
    2: 'CU_STREAM_CAPTURE_STATUS_INVALIDATED',
}
CU_STREAM_CAPTURE_STATUS_NONE = 0
CU_STREAM_CAPTURE_STATUS_ACTIVE = 1
CU_STREAM_CAPTURE_STATUS_INVALIDATED = 2
CUstreamCaptureStatus_enum = ctypes.c_uint32 # enum
CUstreamCaptureStatus = CUstreamCaptureStatus_enum
CUstreamCaptureStatus__enumvalues = CUstreamCaptureStatus_enum__enumvalues

# values for enumeration 'CUstreamCaptureMode_enum'
CUstreamCaptureMode_enum__enumvalues = {
    0: 'CU_STREAM_CAPTURE_MODE_GLOBAL',
    1: 'CU_STREAM_CAPTURE_MODE_THREAD_LOCAL',
    2: 'CU_STREAM_CAPTURE_MODE_RELAXED',
}
CU_STREAM_CAPTURE_MODE_GLOBAL = 0
CU_STREAM_CAPTURE_MODE_THREAD_LOCAL = 1
CU_STREAM_CAPTURE_MODE_RELAXED = 2
CUstreamCaptureMode_enum = ctypes.c_uint32 # enum
CUstreamCaptureMode = CUstreamCaptureMode_enum
CUstreamCaptureMode__enumvalues = CUstreamCaptureMode_enum__enumvalues

# values for enumeration 'CUstreamAttrID_enum'
CUstreamAttrID_enum__enumvalues = {
    1: 'CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW',
    3: 'CU_STREAM_ATTRIBUTE_SYNCHRONIZATION_POLICY',
}
CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW = 1
CU_STREAM_ATTRIBUTE_SYNCHRONIZATION_POLICY = 3
CUstreamAttrID_enum = ctypes.c_uint32 # enum
CUstreamAttrID = CUstreamAttrID_enum
CUstreamAttrID__enumvalues = CUstreamAttrID_enum__enumvalues
class union_CUstreamAttrValue_union(Union):
    pass

union_CUstreamAttrValue_union._pack_ = 1 # source:False
union_CUstreamAttrValue_union._fields_ = [
    ('accessPolicyWindow', CUaccessPolicyWindow),
    ('syncPolicy', CUsynchronizationPolicy),
    ('PADDING_0', ctypes.c_ubyte * 28),
]

CUstreamAttrValue_v1 = union_CUstreamAttrValue_union
CUstreamAttrValue = union_CUstreamAttrValue_union

# values for enumeration 'CUdriverProcAddress_flags_enum'
CUdriverProcAddress_flags_enum__enumvalues = {
    0: 'CU_GET_PROC_ADDRESS_DEFAULT',
    1: 'CU_GET_PROC_ADDRESS_LEGACY_STREAM',
    2: 'CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM',
}
CU_GET_PROC_ADDRESS_DEFAULT = 0
CU_GET_PROC_ADDRESS_LEGACY_STREAM = 1
CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM = 2
CUdriverProcAddress_flags_enum = ctypes.c_uint32 # enum
CUdriverProcAddress_flags = CUdriverProcAddress_flags_enum
CUdriverProcAddress_flags__enumvalues = CUdriverProcAddress_flags_enum__enumvalues

# values for enumeration 'CUexecAffinityType_enum'
CUexecAffinityType_enum__enumvalues = {
    0: 'CU_EXEC_AFFINITY_TYPE_SM_COUNT',
    1: 'CU_EXEC_AFFINITY_TYPE_MAX',
}
CU_EXEC_AFFINITY_TYPE_SM_COUNT = 0
CU_EXEC_AFFINITY_TYPE_MAX = 1
CUexecAffinityType_enum = ctypes.c_uint32 # enum
CUexecAffinityType = CUexecAffinityType_enum
CUexecAffinityType__enumvalues = CUexecAffinityType_enum__enumvalues
class struct_CUexecAffinitySmCount_st(Structure):
    pass

struct_CUexecAffinitySmCount_st._pack_ = 1 # source:False
struct_CUexecAffinitySmCount_st._fields_ = [
    ('val', ctypes.c_uint32),
]

CUexecAffinitySmCount_v1 = struct_CUexecAffinitySmCount_st
CUexecAffinitySmCount = struct_CUexecAffinitySmCount_st
class struct_CUexecAffinityParam_st(Structure):
    pass

class union_CUexecAffinityParam_st_param(Union):
    _pack_ = 1 # source:False
    _fields_ = [
    ('smCount', CUexecAffinitySmCount),
     ]

struct_CUexecAffinityParam_st._pack_ = 1 # source:False
struct_CUexecAffinityParam_st._fields_ = [
    ('type', CUexecAffinityType),
    ('param', union_CUexecAffinityParam_st_param),
]

CUexecAffinityParam_v1 = struct_CUexecAffinityParam_st
CUexecAffinityParam = struct_CUexecAffinityParam_st

# values for enumeration 'cudaError_enum'
cudaError_enum__enumvalues = {
    0: 'CUDA_SUCCESS',
    1: 'CUDA_ERROR_INVALID_VALUE',
    2: 'CUDA_ERROR_OUT_OF_MEMORY',
    3: 'CUDA_ERROR_NOT_INITIALIZED',
    4: 'CUDA_ERROR_DEINITIALIZED',
    5: 'CUDA_ERROR_PROFILER_DISABLED',
    6: 'CUDA_ERROR_PROFILER_NOT_INITIALIZED',
    7: 'CUDA_ERROR_PROFILER_ALREADY_STARTED',
    8: 'CUDA_ERROR_PROFILER_ALREADY_STOPPED',
    34: 'CUDA_ERROR_STUB_LIBRARY',
    100: 'CUDA_ERROR_NO_DEVICE',
    101: 'CUDA_ERROR_INVALID_DEVICE',
    102: 'CUDA_ERROR_DEVICE_NOT_LICENSED',
    200: 'CUDA_ERROR_INVALID_IMAGE',
    201: 'CUDA_ERROR_INVALID_CONTEXT',
    202: 'CUDA_ERROR_CONTEXT_ALREADY_CURRENT',
    205: 'CUDA_ERROR_MAP_FAILED',
    206: 'CUDA_ERROR_UNMAP_FAILED',
    207: 'CUDA_ERROR_ARRAY_IS_MAPPED',
    208: 'CUDA_ERROR_ALREADY_MAPPED',
    209: 'CUDA_ERROR_NO_BINARY_FOR_GPU',
    210: 'CUDA_ERROR_ALREADY_ACQUIRED',
    211: 'CUDA_ERROR_NOT_MAPPED',
    212: 'CUDA_ERROR_NOT_MAPPED_AS_ARRAY',
    213: 'CUDA_ERROR_NOT_MAPPED_AS_POINTER',
    214: 'CUDA_ERROR_ECC_UNCORRECTABLE',
    215: 'CUDA_ERROR_UNSUPPORTED_LIMIT',
    216: 'CUDA_ERROR_CONTEXT_ALREADY_IN_USE',
    217: 'CUDA_ERROR_PEER_ACCESS_UNSUPPORTED',
    218: 'CUDA_ERROR_INVALID_PTX',
    219: 'CUDA_ERROR_INVALID_GRAPHICS_CONTEXT',
    220: 'CUDA_ERROR_NVLINK_UNCORRECTABLE',
    221: 'CUDA_ERROR_JIT_COMPILER_NOT_FOUND',
    222: 'CUDA_ERROR_UNSUPPORTED_PTX_VERSION',
    223: 'CUDA_ERROR_JIT_COMPILATION_DISABLED',
    224: 'CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY',
    300: 'CUDA_ERROR_INVALID_SOURCE',
    301: 'CUDA_ERROR_FILE_NOT_FOUND',
    302: 'CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND',
    303: 'CUDA_ERROR_SHARED_OBJECT_INIT_FAILED',
    304: 'CUDA_ERROR_OPERATING_SYSTEM',
    400: 'CUDA_ERROR_INVALID_HANDLE',
    401: 'CUDA_ERROR_ILLEGAL_STATE',
    500: 'CUDA_ERROR_NOT_FOUND',
    600: 'CUDA_ERROR_NOT_READY',
    700: 'CUDA_ERROR_ILLEGAL_ADDRESS',
    701: 'CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES',
    702: 'CUDA_ERROR_LAUNCH_TIMEOUT',
    703: 'CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING',
    704: 'CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED',
    705: 'CUDA_ERROR_PEER_ACCESS_NOT_ENABLED',
    708: 'CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE',
    709: 'CUDA_ERROR_CONTEXT_IS_DESTROYED',
    710: 'CUDA_ERROR_ASSERT',
    711: 'CUDA_ERROR_TOO_MANY_PEERS',
    712: 'CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED',
    713: 'CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED',
    714: 'CUDA_ERROR_HARDWARE_STACK_ERROR',
    715: 'CUDA_ERROR_ILLEGAL_INSTRUCTION',
    716: 'CUDA_ERROR_MISALIGNED_ADDRESS',
    717: 'CUDA_ERROR_INVALID_ADDRESS_SPACE',
    718: 'CUDA_ERROR_INVALID_PC',
    719: 'CUDA_ERROR_LAUNCH_FAILED',
    720: 'CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE',
    800: 'CUDA_ERROR_NOT_PERMITTED',
    801: 'CUDA_ERROR_NOT_SUPPORTED',
    802: 'CUDA_ERROR_SYSTEM_NOT_READY',
    803: 'CUDA_ERROR_SYSTEM_DRIVER_MISMATCH',
    804: 'CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE',
    805: 'CUDA_ERROR_MPS_CONNECTION_FAILED',
    806: 'CUDA_ERROR_MPS_RPC_FAILURE',
    807: 'CUDA_ERROR_MPS_SERVER_NOT_READY',
    808: 'CUDA_ERROR_MPS_MAX_CLIENTS_REACHED',
    809: 'CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED',
    900: 'CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED',
    901: 'CUDA_ERROR_STREAM_CAPTURE_INVALIDATED',
    902: 'CUDA_ERROR_STREAM_CAPTURE_MERGE',
    903: 'CUDA_ERROR_STREAM_CAPTURE_UNMATCHED',
    904: 'CUDA_ERROR_STREAM_CAPTURE_UNJOINED',
    905: 'CUDA_ERROR_STREAM_CAPTURE_ISOLATION',
    906: 'CUDA_ERROR_STREAM_CAPTURE_IMPLICIT',
    907: 'CUDA_ERROR_CAPTURED_EVENT',
    908: 'CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD',
    909: 'CUDA_ERROR_TIMEOUT',
    910: 'CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE',
    911: 'CUDA_ERROR_EXTERNAL_DEVICE',
    999: 'CUDA_ERROR_UNKNOWN',
}
CUDA_SUCCESS = 0
CUDA_ERROR_INVALID_VALUE = 1
CUDA_ERROR_OUT_OF_MEMORY = 2
CUDA_ERROR_NOT_INITIALIZED = 3
CUDA_ERROR_DEINITIALIZED = 4
CUDA_ERROR_PROFILER_DISABLED = 5
CUDA_ERROR_PROFILER_NOT_INITIALIZED = 6
CUDA_ERROR_PROFILER_ALREADY_STARTED = 7
CUDA_ERROR_PROFILER_ALREADY_STOPPED = 8
CUDA_ERROR_STUB_LIBRARY = 34
CUDA_ERROR_NO_DEVICE = 100
CUDA_ERROR_INVALID_DEVICE = 101
CUDA_ERROR_DEVICE_NOT_LICENSED = 102
CUDA_ERROR_INVALID_IMAGE = 200
CUDA_ERROR_INVALID_CONTEXT = 201
CUDA_ERROR_CONTEXT_ALREADY_CURRENT = 202
CUDA_ERROR_MAP_FAILED = 205
CUDA_ERROR_UNMAP_FAILED = 206
CUDA_ERROR_ARRAY_IS_MAPPED = 207
CUDA_ERROR_ALREADY_MAPPED = 208
CUDA_ERROR_NO_BINARY_FOR_GPU = 209
CUDA_ERROR_ALREADY_ACQUIRED = 210
CUDA_ERROR_NOT_MAPPED = 211
CUDA_ERROR_NOT_MAPPED_AS_ARRAY = 212
CUDA_ERROR_NOT_MAPPED_AS_POINTER = 213
CUDA_ERROR_ECC_UNCORRECTABLE = 214
CUDA_ERROR_UNSUPPORTED_LIMIT = 215
CUDA_ERROR_CONTEXT_ALREADY_IN_USE = 216
CUDA_ERROR_PEER_ACCESS_UNSUPPORTED = 217
CUDA_ERROR_INVALID_PTX = 218
CUDA_ERROR_INVALID_GRAPHICS_CONTEXT = 219
CUDA_ERROR_NVLINK_UNCORRECTABLE = 220
CUDA_ERROR_JIT_COMPILER_NOT_FOUND = 221
CUDA_ERROR_UNSUPPORTED_PTX_VERSION = 222
CUDA_ERROR_JIT_COMPILATION_DISABLED = 223
CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY = 224
CUDA_ERROR_INVALID_SOURCE = 300
CUDA_ERROR_FILE_NOT_FOUND = 301
CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302
CUDA_ERROR_SHARED_OBJECT_INIT_FAILED = 303
CUDA_ERROR_OPERATING_SYSTEM = 304
CUDA_ERROR_INVALID_HANDLE = 400
CUDA_ERROR_ILLEGAL_STATE = 401
CUDA_ERROR_NOT_FOUND = 500
CUDA_ERROR_NOT_READY = 600
CUDA_ERROR_ILLEGAL_ADDRESS = 700
CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701
CUDA_ERROR_LAUNCH_TIMEOUT = 702
CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING = 703
CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704
CUDA_ERROR_PEER_ACCESS_NOT_ENABLED = 705
CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE = 708
CUDA_ERROR_CONTEXT_IS_DESTROYED = 709
CUDA_ERROR_ASSERT = 710
CUDA_ERROR_TOO_MANY_PEERS = 711
CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712
CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED = 713
CUDA_ERROR_HARDWARE_STACK_ERROR = 714
CUDA_ERROR_ILLEGAL_INSTRUCTION = 715
CUDA_ERROR_MISALIGNED_ADDRESS = 716
CUDA_ERROR_INVALID_ADDRESS_SPACE = 717
CUDA_ERROR_INVALID_PC = 718
CUDA_ERROR_LAUNCH_FAILED = 719
CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE = 720
CUDA_ERROR_NOT_PERMITTED = 800
CUDA_ERROR_NOT_SUPPORTED = 801
CUDA_ERROR_SYSTEM_NOT_READY = 802
CUDA_ERROR_SYSTEM_DRIVER_MISMATCH = 803
CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE = 804
CUDA_ERROR_MPS_CONNECTION_FAILED = 805
CUDA_ERROR_MPS_RPC_FAILURE = 806
CUDA_ERROR_MPS_SERVER_NOT_READY = 807
CUDA_ERROR_MPS_MAX_CLIENTS_REACHED = 808
CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED = 809
CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED = 900
CUDA_ERROR_STREAM_CAPTURE_INVALIDATED = 901
CUDA_ERROR_STREAM_CAPTURE_MERGE = 902
CUDA_ERROR_STREAM_CAPTURE_UNMATCHED = 903
CUDA_ERROR_STREAM_CAPTURE_UNJOINED = 904
CUDA_ERROR_STREAM_CAPTURE_ISOLATION = 905
CUDA_ERROR_STREAM_CAPTURE_IMPLICIT = 906
CUDA_ERROR_CAPTURED_EVENT = 907
CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD = 908
CUDA_ERROR_TIMEOUT = 909
CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE = 910
CUDA_ERROR_EXTERNAL_DEVICE = 911
CUDA_ERROR_UNKNOWN = 999
cudaError_enum = ctypes.c_uint32 # enum
CUresult = cudaError_enum
CUresult__enumvalues = cudaError_enum__enumvalues

# values for enumeration 'CUdevice_P2PAttribute_enum'
CUdevice_P2PAttribute_enum__enumvalues = {
    1: 'CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK',
    2: 'CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED',
    3: 'CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED',
    4: 'CU_DEVICE_P2P_ATTRIBUTE_ACCESS_ACCESS_SUPPORTED',
    4: 'CU_DEVICE_P2P_ATTRIBUTE_CUDA_ARRAY_ACCESS_SUPPORTED',
}
CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK = 1
CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED = 2
CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED = 3
CU_DEVICE_P2P_ATTRIBUTE_ACCESS_ACCESS_SUPPORTED = 4
CU_DEVICE_P2P_ATTRIBUTE_CUDA_ARRAY_ACCESS_SUPPORTED = 4
CUdevice_P2PAttribute_enum = ctypes.c_uint32 # enum
CUdevice_P2PAttribute = CUdevice_P2PAttribute_enum
CUdevice_P2PAttribute__enumvalues = CUdevice_P2PAttribute_enum__enumvalues
CUstreamCallback = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_CUstream_st), cudaError_enum, ctypes.POINTER(None))
CUoccupancyB2DSize = ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.c_int32)
class struct_CUDA_MEMCPY2D_st(Structure):
    pass

struct_CUDA_MEMCPY2D_st._pack_ = 1 # source:False
struct_CUDA_MEMCPY2D_st._fields_ = [
    ('srcXInBytes', ctypes.c_uint64),
    ('srcY', ctypes.c_uint64),
    ('srcMemoryType', CUmemorytype),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('srcHost', ctypes.POINTER(None)),
    ('srcDevice', ctypes.c_uint64),
    ('srcArray', ctypes.POINTER(struct_CUarray_st)),
    ('srcPitch', ctypes.c_uint64),
    ('dstXInBytes', ctypes.c_uint64),
    ('dstY', ctypes.c_uint64),
    ('dstMemoryType', CUmemorytype),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('dstHost', ctypes.POINTER(None)),
    ('dstDevice', ctypes.c_uint64),
    ('dstArray', ctypes.POINTER(struct_CUarray_st)),
    ('dstPitch', ctypes.c_uint64),
    ('WidthInBytes', ctypes.c_uint64),
    ('Height', ctypes.c_uint64),
]

CUDA_MEMCPY2D_v2 = struct_CUDA_MEMCPY2D_st
CUDA_MEMCPY2D = struct_CUDA_MEMCPY2D_st
class struct_CUDA_MEMCPY3D_st(Structure):
    pass

struct_CUDA_MEMCPY3D_st._pack_ = 1 # source:False
struct_CUDA_MEMCPY3D_st._fields_ = [
    ('srcXInBytes', ctypes.c_uint64),
    ('srcY', ctypes.c_uint64),
    ('srcZ', ctypes.c_uint64),
    ('srcLOD', ctypes.c_uint64),
    ('srcMemoryType', CUmemorytype),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('srcHost', ctypes.POINTER(None)),
    ('srcDevice', ctypes.c_uint64),
    ('srcArray', ctypes.POINTER(struct_CUarray_st)),
    ('reserved0', ctypes.POINTER(None)),
    ('srcPitch', ctypes.c_uint64),
    ('srcHeight', ctypes.c_uint64),
    ('dstXInBytes', ctypes.c_uint64),
    ('dstY', ctypes.c_uint64),
    ('dstZ', ctypes.c_uint64),
    ('dstLOD', ctypes.c_uint64),
    ('dstMemoryType', CUmemorytype),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('dstHost', ctypes.POINTER(None)),
    ('dstDevice', ctypes.c_uint64),
    ('dstArray', ctypes.POINTER(struct_CUarray_st)),
    ('reserved1', ctypes.POINTER(None)),
    ('dstPitch', ctypes.c_uint64),
    ('dstHeight', ctypes.c_uint64),
    ('WidthInBytes', ctypes.c_uint64),
    ('Height', ctypes.c_uint64),
    ('Depth', ctypes.c_uint64),
]

CUDA_MEMCPY3D_v2 = struct_CUDA_MEMCPY3D_st
CUDA_MEMCPY3D = struct_CUDA_MEMCPY3D_st
class struct_CUDA_MEMCPY3D_PEER_st(Structure):
    pass

struct_CUDA_MEMCPY3D_PEER_st._pack_ = 1 # source:False
struct_CUDA_MEMCPY3D_PEER_st._fields_ = [
    ('srcXInBytes', ctypes.c_uint64),
    ('srcY', ctypes.c_uint64),
    ('srcZ', ctypes.c_uint64),
    ('srcLOD', ctypes.c_uint64),
    ('srcMemoryType', CUmemorytype),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('srcHost', ctypes.POINTER(None)),
    ('srcDevice', ctypes.c_uint64),
    ('srcArray', ctypes.POINTER(struct_CUarray_st)),
    ('srcContext', ctypes.POINTER(struct_CUctx_st)),
    ('srcPitch', ctypes.c_uint64),
    ('srcHeight', ctypes.c_uint64),
    ('dstXInBytes', ctypes.c_uint64),
    ('dstY', ctypes.c_uint64),
    ('dstZ', ctypes.c_uint64),
    ('dstLOD', ctypes.c_uint64),
    ('dstMemoryType', CUmemorytype),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('dstHost', ctypes.POINTER(None)),
    ('dstDevice', ctypes.c_uint64),
    ('dstArray', ctypes.POINTER(struct_CUarray_st)),
    ('dstContext', ctypes.POINTER(struct_CUctx_st)),
    ('dstPitch', ctypes.c_uint64),
    ('dstHeight', ctypes.c_uint64),
    ('WidthInBytes', ctypes.c_uint64),
    ('Height', ctypes.c_uint64),
    ('Depth', ctypes.c_uint64),
]

CUDA_MEMCPY3D_PEER_v1 = struct_CUDA_MEMCPY3D_PEER_st
CUDA_MEMCPY3D_PEER = struct_CUDA_MEMCPY3D_PEER_st
class struct_CUDA_ARRAY_DESCRIPTOR_st(Structure):
    pass

struct_CUDA_ARRAY_DESCRIPTOR_st._pack_ = 1 # source:False
struct_CUDA_ARRAY_DESCRIPTOR_st._fields_ = [
    ('Width', ctypes.c_uint64),
    ('Height', ctypes.c_uint64),
    ('Format', CUarray_format),
    ('NumChannels', ctypes.c_uint32),
]

CUDA_ARRAY_DESCRIPTOR_v2 = struct_CUDA_ARRAY_DESCRIPTOR_st
CUDA_ARRAY_DESCRIPTOR = struct_CUDA_ARRAY_DESCRIPTOR_st
class struct_CUDA_ARRAY3D_DESCRIPTOR_st(Structure):
    pass

struct_CUDA_ARRAY3D_DESCRIPTOR_st._pack_ = 1 # source:False
struct_CUDA_ARRAY3D_DESCRIPTOR_st._fields_ = [
    ('Width', ctypes.c_uint64),
    ('Height', ctypes.c_uint64),
    ('Depth', ctypes.c_uint64),
    ('Format', CUarray_format),
    ('NumChannels', ctypes.c_uint32),
    ('Flags', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

CUDA_ARRAY3D_DESCRIPTOR_v2 = struct_CUDA_ARRAY3D_DESCRIPTOR_st
CUDA_ARRAY3D_DESCRIPTOR = struct_CUDA_ARRAY3D_DESCRIPTOR_st
class struct_CUDA_ARRAY_SPARSE_PROPERTIES_st(Structure):
    pass

class struct_CUDA_ARRAY_SPARSE_PROPERTIES_st_tileExtent(Structure):
    pass

struct_CUDA_ARRAY_SPARSE_PROPERTIES_st_tileExtent._pack_ = 1 # source:False
struct_CUDA_ARRAY_SPARSE_PROPERTIES_st_tileExtent._fields_ = [
    ('width', ctypes.c_uint32),
    ('height', ctypes.c_uint32),
    ('depth', ctypes.c_uint32),
]

struct_CUDA_ARRAY_SPARSE_PROPERTIES_st._pack_ = 1 # source:False
struct_CUDA_ARRAY_SPARSE_PROPERTIES_st._fields_ = [
    ('tileExtent', struct_CUDA_ARRAY_SPARSE_PROPERTIES_st_tileExtent),
    ('miptailFirstLevel', ctypes.c_uint32),
    ('miptailSize', ctypes.c_uint64),
    ('flags', ctypes.c_uint32),
    ('reserved', ctypes.c_uint32 * 4),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

CUDA_ARRAY_SPARSE_PROPERTIES_v1 = struct_CUDA_ARRAY_SPARSE_PROPERTIES_st
CUDA_ARRAY_SPARSE_PROPERTIES = struct_CUDA_ARRAY_SPARSE_PROPERTIES_st
class struct_CUDA_RESOURCE_DESC_st(Structure):
    pass

class union_CUDA_RESOURCE_DESC_st_res(Union):
    pass

class struct_CUDA_RESOURCE_DESC_st_0_array(Structure):
    pass

struct_CUDA_RESOURCE_DESC_st_0_array._pack_ = 1 # source:False
struct_CUDA_RESOURCE_DESC_st_0_array._fields_ = [
    ('hArray', ctypes.POINTER(struct_CUarray_st)),
]

class struct_CUDA_RESOURCE_DESC_st_0_mipmap(Structure):
    pass

struct_CUDA_RESOURCE_DESC_st_0_mipmap._pack_ = 1 # source:False
struct_CUDA_RESOURCE_DESC_st_0_mipmap._fields_ = [
    ('hMipmappedArray', ctypes.POINTER(struct_CUmipmappedArray_st)),
]

class struct_CUDA_RESOURCE_DESC_st_0_linear(Structure):
    pass

struct_CUDA_RESOURCE_DESC_st_0_linear._pack_ = 1 # source:False
struct_CUDA_RESOURCE_DESC_st_0_linear._fields_ = [
    ('devPtr', ctypes.c_uint64),
    ('format', CUarray_format),
    ('numChannels', ctypes.c_uint32),
    ('sizeInBytes', ctypes.c_uint64),
]

class struct_CUDA_RESOURCE_DESC_st_0_pitch2D(Structure):
    pass

struct_CUDA_RESOURCE_DESC_st_0_pitch2D._pack_ = 1 # source:False
struct_CUDA_RESOURCE_DESC_st_0_pitch2D._fields_ = [
    ('devPtr', ctypes.c_uint64),
    ('format', CUarray_format),
    ('numChannels', ctypes.c_uint32),
    ('width', ctypes.c_uint64),
    ('height', ctypes.c_uint64),
    ('pitchInBytes', ctypes.c_uint64),
]

class struct_CUDA_RESOURCE_DESC_st_0_reserved(Structure):
    pass

struct_CUDA_RESOURCE_DESC_st_0_reserved._pack_ = 1 # source:False
struct_CUDA_RESOURCE_DESC_st_0_reserved._fields_ = [
    ('reserved', ctypes.c_int32 * 32),
]

union_CUDA_RESOURCE_DESC_st_res._pack_ = 1 # source:False
union_CUDA_RESOURCE_DESC_st_res._fields_ = [
    ('array', struct_CUDA_RESOURCE_DESC_st_0_array),
    ('mipmap', struct_CUDA_RESOURCE_DESC_st_0_mipmap),
    ('linear', struct_CUDA_RESOURCE_DESC_st_0_linear),
    ('pitch2D', struct_CUDA_RESOURCE_DESC_st_0_pitch2D),
    ('reserved', struct_CUDA_RESOURCE_DESC_st_0_reserved),
]

struct_CUDA_RESOURCE_DESC_st._pack_ = 1 # source:False
struct_CUDA_RESOURCE_DESC_st._fields_ = [
    ('resType', CUresourcetype),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('res', union_CUDA_RESOURCE_DESC_st_res),
    ('flags', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

CUDA_RESOURCE_DESC_v1 = struct_CUDA_RESOURCE_DESC_st
CUDA_RESOURCE_DESC = struct_CUDA_RESOURCE_DESC_st
class struct_CUDA_TEXTURE_DESC_st(Structure):
    pass

struct_CUDA_TEXTURE_DESC_st._pack_ = 1 # source:False
struct_CUDA_TEXTURE_DESC_st._fields_ = [
    ('addressMode', CUaddress_mode_enum * 3),
    ('filterMode', CUfilter_mode),
    ('flags', ctypes.c_uint32),
    ('maxAnisotropy', ctypes.c_uint32),
    ('mipmapFilterMode', CUfilter_mode),
    ('mipmapLevelBias', ctypes.c_float),
    ('minMipmapLevelClamp', ctypes.c_float),
    ('maxMipmapLevelClamp', ctypes.c_float),
    ('borderColor', ctypes.c_float * 4),
    ('reserved', ctypes.c_int32 * 12),
]

CUDA_TEXTURE_DESC_v1 = struct_CUDA_TEXTURE_DESC_st
CUDA_TEXTURE_DESC = struct_CUDA_TEXTURE_DESC_st

# values for enumeration 'CUresourceViewFormat_enum'
CUresourceViewFormat_enum__enumvalues = {
    0: 'CU_RES_VIEW_FORMAT_NONE',
    1: 'CU_RES_VIEW_FORMAT_UINT_1X8',
    2: 'CU_RES_VIEW_FORMAT_UINT_2X8',
    3: 'CU_RES_VIEW_FORMAT_UINT_4X8',
    4: 'CU_RES_VIEW_FORMAT_SINT_1X8',
    5: 'CU_RES_VIEW_FORMAT_SINT_2X8',
    6: 'CU_RES_VIEW_FORMAT_SINT_4X8',
    7: 'CU_RES_VIEW_FORMAT_UINT_1X16',
    8: 'CU_RES_VIEW_FORMAT_UINT_2X16',
    9: 'CU_RES_VIEW_FORMAT_UINT_4X16',
    10: 'CU_RES_VIEW_FORMAT_SINT_1X16',
    11: 'CU_RES_VIEW_FORMAT_SINT_2X16',
    12: 'CU_RES_VIEW_FORMAT_SINT_4X16',
    13: 'CU_RES_VIEW_FORMAT_UINT_1X32',
    14: 'CU_RES_VIEW_FORMAT_UINT_2X32',
    15: 'CU_RES_VIEW_FORMAT_UINT_4X32',
    16: 'CU_RES_VIEW_FORMAT_SINT_1X32',
    17: 'CU_RES_VIEW_FORMAT_SINT_2X32',
    18: 'CU_RES_VIEW_FORMAT_SINT_4X32',
    19: 'CU_RES_VIEW_FORMAT_FLOAT_1X16',
    20: 'CU_RES_VIEW_FORMAT_FLOAT_2X16',
    21: 'CU_RES_VIEW_FORMAT_FLOAT_4X16',
    22: 'CU_RES_VIEW_FORMAT_FLOAT_1X32',
    23: 'CU_RES_VIEW_FORMAT_FLOAT_2X32',
    24: 'CU_RES_VIEW_FORMAT_FLOAT_4X32',
    25: 'CU_RES_VIEW_FORMAT_UNSIGNED_BC1',
    26: 'CU_RES_VIEW_FORMAT_UNSIGNED_BC2',
    27: 'CU_RES_VIEW_FORMAT_UNSIGNED_BC3',
    28: 'CU_RES_VIEW_FORMAT_UNSIGNED_BC4',
    29: 'CU_RES_VIEW_FORMAT_SIGNED_BC4',
    30: 'CU_RES_VIEW_FORMAT_UNSIGNED_BC5',
    31: 'CU_RES_VIEW_FORMAT_SIGNED_BC5',
    32: 'CU_RES_VIEW_FORMAT_UNSIGNED_BC6H',
    33: 'CU_RES_VIEW_FORMAT_SIGNED_BC6H',
    34: 'CU_RES_VIEW_FORMAT_UNSIGNED_BC7',
}
CU_RES_VIEW_FORMAT_NONE = 0
CU_RES_VIEW_FORMAT_UINT_1X8 = 1
CU_RES_VIEW_FORMAT_UINT_2X8 = 2
CU_RES_VIEW_FORMAT_UINT_4X8 = 3
CU_RES_VIEW_FORMAT_SINT_1X8 = 4
CU_RES_VIEW_FORMAT_SINT_2X8 = 5
CU_RES_VIEW_FORMAT_SINT_4X8 = 6
CU_RES_VIEW_FORMAT_UINT_1X16 = 7
CU_RES_VIEW_FORMAT_UINT_2X16 = 8
CU_RES_VIEW_FORMAT_UINT_4X16 = 9
CU_RES_VIEW_FORMAT_SINT_1X16 = 10
CU_RES_VIEW_FORMAT_SINT_2X16 = 11
CU_RES_VIEW_FORMAT_SINT_4X16 = 12
CU_RES_VIEW_FORMAT_UINT_1X32 = 13
CU_RES_VIEW_FORMAT_UINT_2X32 = 14
CU_RES_VIEW_FORMAT_UINT_4X32 = 15
CU_RES_VIEW_FORMAT_SINT_1X32 = 16
CU_RES_VIEW_FORMAT_SINT_2X32 = 17
CU_RES_VIEW_FORMAT_SINT_4X32 = 18
CU_RES_VIEW_FORMAT_FLOAT_1X16 = 19
CU_RES_VIEW_FORMAT_FLOAT_2X16 = 20
CU_RES_VIEW_FORMAT_FLOAT_4X16 = 21
CU_RES_VIEW_FORMAT_FLOAT_1X32 = 22
CU_RES_VIEW_FORMAT_FLOAT_2X32 = 23
CU_RES_VIEW_FORMAT_FLOAT_4X32 = 24
CU_RES_VIEW_FORMAT_UNSIGNED_BC1 = 25
CU_RES_VIEW_FORMAT_UNSIGNED_BC2 = 26
CU_RES_VIEW_FORMAT_UNSIGNED_BC3 = 27
CU_RES_VIEW_FORMAT_UNSIGNED_BC4 = 28
CU_RES_VIEW_FORMAT_SIGNED_BC4 = 29
CU_RES_VIEW_FORMAT_UNSIGNED_BC5 = 30
CU_RES_VIEW_FORMAT_SIGNED_BC5 = 31
CU_RES_VIEW_FORMAT_UNSIGNED_BC6H = 32
CU_RES_VIEW_FORMAT_SIGNED_BC6H = 33
CU_RES_VIEW_FORMAT_UNSIGNED_BC7 = 34
CUresourceViewFormat_enum = ctypes.c_uint32 # enum
CUresourceViewFormat = CUresourceViewFormat_enum
CUresourceViewFormat__enumvalues = CUresourceViewFormat_enum__enumvalues
class struct_CUDA_RESOURCE_VIEW_DESC_st(Structure):
    pass

struct_CUDA_RESOURCE_VIEW_DESC_st._pack_ = 1 # source:False
struct_CUDA_RESOURCE_VIEW_DESC_st._fields_ = [
    ('format', CUresourceViewFormat),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('width', ctypes.c_uint64),
    ('height', ctypes.c_uint64),
    ('depth', ctypes.c_uint64),
    ('firstMipmapLevel', ctypes.c_uint32),
    ('lastMipmapLevel', ctypes.c_uint32),
    ('firstLayer', ctypes.c_uint32),
    ('lastLayer', ctypes.c_uint32),
    ('reserved', ctypes.c_uint32 * 16),
]

CUDA_RESOURCE_VIEW_DESC_v1 = struct_CUDA_RESOURCE_VIEW_DESC_st
CUDA_RESOURCE_VIEW_DESC = struct_CUDA_RESOURCE_VIEW_DESC_st
class struct_CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st(Structure):
    pass

struct_CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st._pack_ = 1 # source:False
struct_CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st._fields_ = [
    ('p2pToken', ctypes.c_uint64),
    ('vaSpaceToken', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_v1 = struct_CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st
CUDA_POINTER_ATTRIBUTE_P2P_TOKENS = struct_CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st

# values for enumeration 'CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS_enum'
CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS_enum__enumvalues = {
    0: 'CU_POINTER_ATTRIBUTE_ACCESS_FLAG_NONE',
    1: 'CU_POINTER_ATTRIBUTE_ACCESS_FLAG_READ',
    3: 'CU_POINTER_ATTRIBUTE_ACCESS_FLAG_READWRITE',
}
CU_POINTER_ATTRIBUTE_ACCESS_FLAG_NONE = 0
CU_POINTER_ATTRIBUTE_ACCESS_FLAG_READ = 1
CU_POINTER_ATTRIBUTE_ACCESS_FLAG_READWRITE = 3
CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS_enum = ctypes.c_uint32 # enum
CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS = CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS_enum
CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS__enumvalues = CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS_enum__enumvalues
class struct_CUDA_LAUNCH_PARAMS_st(Structure):
    pass

struct_CUDA_LAUNCH_PARAMS_st._pack_ = 1 # source:False
struct_CUDA_LAUNCH_PARAMS_st._fields_ = [
    ('function', ctypes.POINTER(struct_CUfunc_st)),
    ('gridDimX', ctypes.c_uint32),
    ('gridDimY', ctypes.c_uint32),
    ('gridDimZ', ctypes.c_uint32),
    ('blockDimX', ctypes.c_uint32),
    ('blockDimY', ctypes.c_uint32),
    ('blockDimZ', ctypes.c_uint32),
    ('sharedMemBytes', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('hStream', ctypes.POINTER(struct_CUstream_st)),
    ('kernelParams', ctypes.POINTER(ctypes.POINTER(None))),
]

CUDA_LAUNCH_PARAMS_v1 = struct_CUDA_LAUNCH_PARAMS_st
CUDA_LAUNCH_PARAMS = struct_CUDA_LAUNCH_PARAMS_st

# values for enumeration 'CUexternalMemoryHandleType_enum'
CUexternalMemoryHandleType_enum__enumvalues = {
    1: 'CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD',
    2: 'CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32',
    3: 'CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT',
    4: 'CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP',
    5: 'CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE',
    6: 'CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE',
    7: 'CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE_KMT',
    8: 'CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF',
}
CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD = 1
CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32 = 2
CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT = 3
CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP = 4
CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE = 5
CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE = 6
CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE_KMT = 7
CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF = 8
CUexternalMemoryHandleType_enum = ctypes.c_uint32 # enum
CUexternalMemoryHandleType = CUexternalMemoryHandleType_enum
CUexternalMemoryHandleType__enumvalues = CUexternalMemoryHandleType_enum__enumvalues
class struct_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st(Structure):
    pass

class union_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st_handle(Union):
    pass

class struct_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st_0_win32(Structure):
    pass

struct_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st_0_win32._pack_ = 1 # source:False
struct_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st_0_win32._fields_ = [
    ('handle', ctypes.POINTER(None)),
    ('name', ctypes.POINTER(None)),
]

union_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st_handle._pack_ = 1 # source:False
union_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st_handle._fields_ = [
    ('fd', ctypes.c_int32),
    ('win32', struct_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st_0_win32),
    ('nvSciBufObject', ctypes.POINTER(None)),
    ('PADDING_0', ctypes.c_ubyte * 8),
]

struct_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st._pack_ = 1 # source:False
struct_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st._fields_ = [
    ('type', CUexternalMemoryHandleType),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('handle', union_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st_handle),
    ('size', ctypes.c_uint64),
    ('flags', ctypes.c_uint32),
    ('reserved', ctypes.c_uint32 * 16),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1 = struct_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st
CUDA_EXTERNAL_MEMORY_HANDLE_DESC = struct_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st
class struct_CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st(Structure):
    pass

struct_CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st._pack_ = 1 # source:False
struct_CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st._fields_ = [
    ('offset', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
    ('flags', ctypes.c_uint32),
    ('reserved', ctypes.c_uint32 * 16),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

CUDA_EXTERNAL_MEMORY_BUFFER_DESC_v1 = struct_CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st
CUDA_EXTERNAL_MEMORY_BUFFER_DESC = struct_CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st
class struct_CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st(Structure):
    pass

struct_CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st._pack_ = 1 # source:False
struct_CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st._fields_ = [
    ('offset', ctypes.c_uint64),
    ('arrayDesc', CUDA_ARRAY3D_DESCRIPTOR),
    ('numLevels', ctypes.c_uint32),
    ('reserved', ctypes.c_uint32 * 16),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_v1 = struct_CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st
CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC = struct_CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st

# values for enumeration 'CUexternalSemaphoreHandleType_enum'
CUexternalSemaphoreHandleType_enum__enumvalues = {
    1: 'CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD',
    2: 'CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32',
    3: 'CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT',
    4: 'CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE',
    5: 'CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_FENCE',
    6: 'CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC',
    7: 'CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX',
    8: 'CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX_KMT',
    9: 'CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD',
    10: 'CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_WIN32',
}
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD = 1
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32 = 2
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT = 3
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE = 4
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_FENCE = 5
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC = 6
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX = 7
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX_KMT = 8
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD = 9
CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_WIN32 = 10
CUexternalSemaphoreHandleType_enum = ctypes.c_uint32 # enum
CUexternalSemaphoreHandleType = CUexternalSemaphoreHandleType_enum
CUexternalSemaphoreHandleType__enumvalues = CUexternalSemaphoreHandleType_enum__enumvalues
class struct_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st(Structure):
    pass

class union_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st_handle(Union):
    pass

class struct_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st_0_win32(Structure):
    pass

struct_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st_0_win32._pack_ = 1 # source:False
struct_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st_0_win32._fields_ = [
    ('handle', ctypes.POINTER(None)),
    ('name', ctypes.POINTER(None)),
]

union_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st_handle._pack_ = 1 # source:False
union_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st_handle._fields_ = [
    ('fd', ctypes.c_int32),
    ('win32', struct_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st_0_win32),
    ('nvSciSyncObj', ctypes.POINTER(None)),
    ('PADDING_0', ctypes.c_ubyte * 8),
]

struct_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st._pack_ = 1 # source:False
struct_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st._fields_ = [
    ('type', CUexternalSemaphoreHandleType),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('handle', union_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st_handle),
    ('flags', ctypes.c_uint32),
    ('reserved', ctypes.c_uint32 * 16),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1 = struct_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st
CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC = struct_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st
class struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st(Structure):
    pass

class struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_params(Structure):
    pass

class struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_0_fence(Structure):
    pass

struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_0_fence._pack_ = 1 # source:False
struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_0_fence._fields_ = [
    ('value', ctypes.c_uint64),
]

class union_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_0_nvSciSync(Union):
    pass

union_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_0_nvSciSync._pack_ = 1 # source:False
union_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_0_nvSciSync._fields_ = [
    ('fence', ctypes.POINTER(None)),
    ('reserved', ctypes.c_uint64),
]

class struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_0_keyedMutex(Structure):
    pass

struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_0_keyedMutex._pack_ = 1 # source:False
struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_0_keyedMutex._fields_ = [
    ('key', ctypes.c_uint64),
]

struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_params._pack_ = 1 # source:False
struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_params._fields_ = [
    ('fence', struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_0_fence),
    ('nvSciSync', union_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_0_nvSciSync),
    ('keyedMutex', struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_0_keyedMutex),
    ('reserved', ctypes.c_uint32 * 12),
]

struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st._pack_ = 1 # source:False
struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st._fields_ = [
    ('params', struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_params),
    ('flags', ctypes.c_uint32),
    ('reserved', ctypes.c_uint32 * 16),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1 = struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st
CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS = struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st
class struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st(Structure):
    pass

class struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_params(Structure):
    pass

class struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_0_fence(Structure):
    pass

struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_0_fence._pack_ = 1 # source:False
struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_0_fence._fields_ = [
    ('value', ctypes.c_uint64),
]

class union_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_0_nvSciSync(Union):
    pass

union_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_0_nvSciSync._pack_ = 1 # source:False
union_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_0_nvSciSync._fields_ = [
    ('fence', ctypes.POINTER(None)),
    ('reserved', ctypes.c_uint64),
]

class struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_0_keyedMutex(Structure):
    pass

struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_0_keyedMutex._pack_ = 1 # source:False
struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_0_keyedMutex._fields_ = [
    ('key', ctypes.c_uint64),
    ('timeoutMs', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_params._pack_ = 1 # source:False
struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_params._fields_ = [
    ('fence', struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_0_fence),
    ('nvSciSync', union_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_0_nvSciSync),
    ('keyedMutex', struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_0_keyedMutex),
    ('reserved', ctypes.c_uint32 * 10),
]

struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st._pack_ = 1 # source:False
struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st._fields_ = [
    ('params', struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_params),
    ('flags', ctypes.c_uint32),
    ('reserved', ctypes.c_uint32 * 16),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1 = struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st
CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS = struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st
class struct_CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_st(Structure):
    pass

struct_CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_st._pack_ = 1 # source:False
struct_CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_st._fields_ = [
    ('extSemArray', ctypes.POINTER(ctypes.POINTER(struct_CUextSemaphore_st))),
    ('paramsArray', ctypes.POINTER(struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st)),
    ('numExtSems', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v1 = struct_CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_st
CUDA_EXT_SEM_SIGNAL_NODE_PARAMS = struct_CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_st
class struct_CUDA_EXT_SEM_WAIT_NODE_PARAMS_st(Structure):
    pass

struct_CUDA_EXT_SEM_WAIT_NODE_PARAMS_st._pack_ = 1 # source:False
struct_CUDA_EXT_SEM_WAIT_NODE_PARAMS_st._fields_ = [
    ('extSemArray', ctypes.POINTER(ctypes.POINTER(struct_CUextSemaphore_st))),
    ('paramsArray', ctypes.POINTER(struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st)),
    ('numExtSems', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

CUDA_EXT_SEM_WAIT_NODE_PARAMS_v1 = struct_CUDA_EXT_SEM_WAIT_NODE_PARAMS_st
CUDA_EXT_SEM_WAIT_NODE_PARAMS = struct_CUDA_EXT_SEM_WAIT_NODE_PARAMS_st
CUmemGenericAllocationHandle_v1 = ctypes.c_uint64
CUmemGenericAllocationHandle = ctypes.c_uint64

# values for enumeration 'CUmemAllocationHandleType_enum'
CUmemAllocationHandleType_enum__enumvalues = {
    0: 'CU_MEM_HANDLE_TYPE_NONE',
    1: 'CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR',
    2: 'CU_MEM_HANDLE_TYPE_WIN32',
    4: 'CU_MEM_HANDLE_TYPE_WIN32_KMT',
    2147483647: 'CU_MEM_HANDLE_TYPE_MAX',
}
CU_MEM_HANDLE_TYPE_NONE = 0
CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 1
CU_MEM_HANDLE_TYPE_WIN32 = 2
CU_MEM_HANDLE_TYPE_WIN32_KMT = 4
CU_MEM_HANDLE_TYPE_MAX = 2147483647
CUmemAllocationHandleType_enum = ctypes.c_uint32 # enum
CUmemAllocationHandleType = CUmemAllocationHandleType_enum
CUmemAllocationHandleType__enumvalues = CUmemAllocationHandleType_enum__enumvalues

# values for enumeration 'CUmemAccess_flags_enum'
CUmemAccess_flags_enum__enumvalues = {
    0: 'CU_MEM_ACCESS_FLAGS_PROT_NONE',
    1: 'CU_MEM_ACCESS_FLAGS_PROT_READ',
    3: 'CU_MEM_ACCESS_FLAGS_PROT_READWRITE',
    2147483647: 'CU_MEM_ACCESS_FLAGS_PROT_MAX',
}
CU_MEM_ACCESS_FLAGS_PROT_NONE = 0
CU_MEM_ACCESS_FLAGS_PROT_READ = 1
CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3
CU_MEM_ACCESS_FLAGS_PROT_MAX = 2147483647
CUmemAccess_flags_enum = ctypes.c_uint32 # enum
CUmemAccess_flags = CUmemAccess_flags_enum
CUmemAccess_flags__enumvalues = CUmemAccess_flags_enum__enumvalues

# values for enumeration 'CUmemLocationType_enum'
CUmemLocationType_enum__enumvalues = {
    0: 'CU_MEM_LOCATION_TYPE_INVALID',
    1: 'CU_MEM_LOCATION_TYPE_DEVICE',
    2147483647: 'CU_MEM_LOCATION_TYPE_MAX',
}
CU_MEM_LOCATION_TYPE_INVALID = 0
CU_MEM_LOCATION_TYPE_DEVICE = 1
CU_MEM_LOCATION_TYPE_MAX = 2147483647
CUmemLocationType_enum = ctypes.c_uint32 # enum
CUmemLocationType = CUmemLocationType_enum
CUmemLocationType__enumvalues = CUmemLocationType_enum__enumvalues

# values for enumeration 'CUmemAllocationType_enum'
CUmemAllocationType_enum__enumvalues = {
    0: 'CU_MEM_ALLOCATION_TYPE_INVALID',
    1: 'CU_MEM_ALLOCATION_TYPE_PINNED',
    2147483647: 'CU_MEM_ALLOCATION_TYPE_MAX',
}
CU_MEM_ALLOCATION_TYPE_INVALID = 0
CU_MEM_ALLOCATION_TYPE_PINNED = 1
CU_MEM_ALLOCATION_TYPE_MAX = 2147483647
CUmemAllocationType_enum = ctypes.c_uint32 # enum
CUmemAllocationType = CUmemAllocationType_enum
CUmemAllocationType__enumvalues = CUmemAllocationType_enum__enumvalues

# values for enumeration 'CUmemAllocationGranularity_flags_enum'
CUmemAllocationGranularity_flags_enum__enumvalues = {
    0: 'CU_MEM_ALLOC_GRANULARITY_MINIMUM',
    1: 'CU_MEM_ALLOC_GRANULARITY_RECOMMENDED',
}
CU_MEM_ALLOC_GRANULARITY_MINIMUM = 0
CU_MEM_ALLOC_GRANULARITY_RECOMMENDED = 1
CUmemAllocationGranularity_flags_enum = ctypes.c_uint32 # enum
CUmemAllocationGranularity_flags = CUmemAllocationGranularity_flags_enum
CUmemAllocationGranularity_flags__enumvalues = CUmemAllocationGranularity_flags_enum__enumvalues

# values for enumeration 'CUarraySparseSubresourceType_enum'
CUarraySparseSubresourceType_enum__enumvalues = {
    0: 'CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_SPARSE_LEVEL',
    1: 'CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_MIPTAIL',
}
CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_SPARSE_LEVEL = 0
CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_MIPTAIL = 1
CUarraySparseSubresourceType_enum = ctypes.c_uint32 # enum
CUarraySparseSubresourceType = CUarraySparseSubresourceType_enum
CUarraySparseSubresourceType__enumvalues = CUarraySparseSubresourceType_enum__enumvalues

# values for enumeration 'CUmemOperationType_enum'
CUmemOperationType_enum__enumvalues = {
    1: 'CU_MEM_OPERATION_TYPE_MAP',
    2: 'CU_MEM_OPERATION_TYPE_UNMAP',
}
CU_MEM_OPERATION_TYPE_MAP = 1
CU_MEM_OPERATION_TYPE_UNMAP = 2
CUmemOperationType_enum = ctypes.c_uint32 # enum
CUmemOperationType = CUmemOperationType_enum
CUmemOperationType__enumvalues = CUmemOperationType_enum__enumvalues

# values for enumeration 'CUmemHandleType_enum'
CUmemHandleType_enum__enumvalues = {
    0: 'CU_MEM_HANDLE_TYPE_GENERIC',
}
CU_MEM_HANDLE_TYPE_GENERIC = 0
CUmemHandleType_enum = ctypes.c_uint32 # enum
CUmemHandleType = CUmemHandleType_enum
CUmemHandleType__enumvalues = CUmemHandleType_enum__enumvalues
class struct_CUarrayMapInfo_st(Structure):
    pass

class union_CUarrayMapInfo_st_resource(Union):
    pass

union_CUarrayMapInfo_st_resource._pack_ = 1 # source:False
union_CUarrayMapInfo_st_resource._fields_ = [
    ('mipmap', ctypes.POINTER(struct_CUmipmappedArray_st)),
    ('array', ctypes.POINTER(struct_CUarray_st)),
]

class union_CUarrayMapInfo_st_subresource(Union):
    pass

class struct_CUarrayMapInfo_st_1_sparseLevel(Structure):
    pass

struct_CUarrayMapInfo_st_1_sparseLevel._pack_ = 1 # source:False
struct_CUarrayMapInfo_st_1_sparseLevel._fields_ = [
    ('level', ctypes.c_uint32),
    ('layer', ctypes.c_uint32),
    ('offsetX', ctypes.c_uint32),
    ('offsetY', ctypes.c_uint32),
    ('offsetZ', ctypes.c_uint32),
    ('extentWidth', ctypes.c_uint32),
    ('extentHeight', ctypes.c_uint32),
    ('extentDepth', ctypes.c_uint32),
]

class struct_CUarrayMapInfo_st_1_miptail(Structure):
    pass

struct_CUarrayMapInfo_st_1_miptail._pack_ = 1 # source:False
struct_CUarrayMapInfo_st_1_miptail._fields_ = [
    ('layer', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('offset', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
]

union_CUarrayMapInfo_st_subresource._pack_ = 1 # source:False
union_CUarrayMapInfo_st_subresource._fields_ = [
    ('sparseLevel', struct_CUarrayMapInfo_st_1_sparseLevel),
    ('miptail', struct_CUarrayMapInfo_st_1_miptail),
    ('PADDING_0', ctypes.c_ubyte * 8),
]

class union_CUarrayMapInfo_st_memHandle(Union):
    pass

union_CUarrayMapInfo_st_memHandle._pack_ = 1 # source:False
union_CUarrayMapInfo_st_memHandle._fields_ = [
    ('memHandle', ctypes.c_uint64),
]

struct_CUarrayMapInfo_st._pack_ = 1 # source:False
struct_CUarrayMapInfo_st._fields_ = [
    ('resourceType', CUresourcetype),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('resource', union_CUarrayMapInfo_st_resource),
    ('subresourceType', CUarraySparseSubresourceType),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('subresource', union_CUarrayMapInfo_st_subresource),
    ('memOperationType', CUmemOperationType),
    ('memHandleType', CUmemHandleType),
    ('memHandle', union_CUarrayMapInfo_st_memHandle),
    ('offset', ctypes.c_uint64),
    ('deviceBitMask', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('reserved', ctypes.c_uint32 * 2),
]

CUarrayMapInfo_v1 = struct_CUarrayMapInfo_st
CUarrayMapInfo = struct_CUarrayMapInfo_st
class struct_CUmemLocation_st(Structure):
    pass

struct_CUmemLocation_st._pack_ = 1 # source:False
struct_CUmemLocation_st._fields_ = [
    ('type', CUmemLocationType),
    ('id', ctypes.c_int32),
]

CUmemLocation_v1 = struct_CUmemLocation_st
CUmemLocation = struct_CUmemLocation_st

# values for enumeration 'CUmemAllocationCompType_enum'
CUmemAllocationCompType_enum__enumvalues = {
    0: 'CU_MEM_ALLOCATION_COMP_NONE',
    1: 'CU_MEM_ALLOCATION_COMP_GENERIC',
}
CU_MEM_ALLOCATION_COMP_NONE = 0
CU_MEM_ALLOCATION_COMP_GENERIC = 1
CUmemAllocationCompType_enum = ctypes.c_uint32 # enum
CUmemAllocationCompType = CUmemAllocationCompType_enum
CUmemAllocationCompType__enumvalues = CUmemAllocationCompType_enum__enumvalues
class struct_CUmemAllocationProp_st(Structure):
    pass

class struct_CUmemAllocationProp_st_allocFlags(Structure):
    pass

struct_CUmemAllocationProp_st_allocFlags._pack_ = 1 # source:False
struct_CUmemAllocationProp_st_allocFlags._fields_ = [
    ('compressionType', ctypes.c_ubyte),
    ('gpuDirectRDMACapable', ctypes.c_ubyte),
    ('usage', ctypes.c_uint16),
    ('reserved', ctypes.c_ubyte * 4),
]

struct_CUmemAllocationProp_st._pack_ = 1 # source:False
struct_CUmemAllocationProp_st._fields_ = [
    ('type', CUmemAllocationType),
    ('requestedHandleTypes', CUmemAllocationHandleType),
    ('location', CUmemLocation),
    ('win32HandleMetaData', ctypes.POINTER(None)),
    ('allocFlags', struct_CUmemAllocationProp_st_allocFlags),
]

CUmemAllocationProp_v1 = struct_CUmemAllocationProp_st
CUmemAllocationProp = struct_CUmemAllocationProp_st
class struct_CUmemAccessDesc_st(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('location', CUmemLocation),
    ('flags', CUmemAccess_flags),
     ]

CUmemAccessDesc_v1 = struct_CUmemAccessDesc_st
CUmemAccessDesc = struct_CUmemAccessDesc_st

# values for enumeration 'CUgraphExecUpdateResult_enum'
CUgraphExecUpdateResult_enum__enumvalues = {
    0: 'CU_GRAPH_EXEC_UPDATE_SUCCESS',
    1: 'CU_GRAPH_EXEC_UPDATE_ERROR',
    2: 'CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED',
    3: 'CU_GRAPH_EXEC_UPDATE_ERROR_NODE_TYPE_CHANGED',
    4: 'CU_GRAPH_EXEC_UPDATE_ERROR_FUNCTION_CHANGED',
    5: 'CU_GRAPH_EXEC_UPDATE_ERROR_PARAMETERS_CHANGED',
    6: 'CU_GRAPH_EXEC_UPDATE_ERROR_NOT_SUPPORTED',
    7: 'CU_GRAPH_EXEC_UPDATE_ERROR_UNSUPPORTED_FUNCTION_CHANGE',
}
CU_GRAPH_EXEC_UPDATE_SUCCESS = 0
CU_GRAPH_EXEC_UPDATE_ERROR = 1
CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED = 2
CU_GRAPH_EXEC_UPDATE_ERROR_NODE_TYPE_CHANGED = 3
CU_GRAPH_EXEC_UPDATE_ERROR_FUNCTION_CHANGED = 4
CU_GRAPH_EXEC_UPDATE_ERROR_PARAMETERS_CHANGED = 5
CU_GRAPH_EXEC_UPDATE_ERROR_NOT_SUPPORTED = 6
CU_GRAPH_EXEC_UPDATE_ERROR_UNSUPPORTED_FUNCTION_CHANGE = 7
CUgraphExecUpdateResult_enum = ctypes.c_uint32 # enum
CUgraphExecUpdateResult = CUgraphExecUpdateResult_enum
CUgraphExecUpdateResult__enumvalues = CUgraphExecUpdateResult_enum__enumvalues

# values for enumeration 'CUmemPool_attribute_enum'
CUmemPool_attribute_enum__enumvalues = {
    1: 'CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES',
    2: 'CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC',
    3: 'CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES',
    4: 'CU_MEMPOOL_ATTR_RELEASE_THRESHOLD',
    5: 'CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT',
    6: 'CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH',
    7: 'CU_MEMPOOL_ATTR_USED_MEM_CURRENT',
    8: 'CU_MEMPOOL_ATTR_USED_MEM_HIGH',
}
CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES = 1
CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC = 2
CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES = 3
CU_MEMPOOL_ATTR_RELEASE_THRESHOLD = 4
CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT = 5
CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH = 6
CU_MEMPOOL_ATTR_USED_MEM_CURRENT = 7
CU_MEMPOOL_ATTR_USED_MEM_HIGH = 8
CUmemPool_attribute_enum = ctypes.c_uint32 # enum
CUmemPool_attribute = CUmemPool_attribute_enum
CUmemPool_attribute__enumvalues = CUmemPool_attribute_enum__enumvalues
class struct_CUmemPoolProps_st(Structure):
    pass

struct_CUmemPoolProps_st._pack_ = 1 # source:False
struct_CUmemPoolProps_st._fields_ = [
    ('allocType', CUmemAllocationType),
    ('handleTypes', CUmemAllocationHandleType),
    ('location', CUmemLocation),
    ('win32SecurityAttributes', ctypes.POINTER(None)),
    ('reserved', ctypes.c_ubyte * 64),
]

CUmemPoolProps_v1 = struct_CUmemPoolProps_st
CUmemPoolProps = struct_CUmemPoolProps_st
class struct_CUmemPoolPtrExportData_st(Structure):
    pass

struct_CUmemPoolPtrExportData_st._pack_ = 1 # source:False
struct_CUmemPoolPtrExportData_st._fields_ = [
    ('reserved', ctypes.c_ubyte * 64),
]

CUmemPoolPtrExportData_v1 = struct_CUmemPoolPtrExportData_st
CUmemPoolPtrExportData = struct_CUmemPoolPtrExportData_st
class struct_CUDA_MEM_ALLOC_NODE_PARAMS_st(Structure):
    pass

struct_CUDA_MEM_ALLOC_NODE_PARAMS_st._pack_ = 1 # source:False
struct_CUDA_MEM_ALLOC_NODE_PARAMS_st._fields_ = [
    ('poolProps', CUmemPoolProps),
    ('accessDescs', ctypes.POINTER(struct_CUmemAccessDesc_st)),
    ('accessDescCount', ctypes.c_uint64),
    ('bytesize', ctypes.c_uint64),
    ('dptr', ctypes.c_uint64),
]

CUDA_MEM_ALLOC_NODE_PARAMS = struct_CUDA_MEM_ALLOC_NODE_PARAMS_st

# values for enumeration 'CUgraphMem_attribute_enum'
CUgraphMem_attribute_enum__enumvalues = {
    0: 'CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT',
    1: 'CU_GRAPH_MEM_ATTR_USED_MEM_HIGH',
    2: 'CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT',
    3: 'CU_GRAPH_MEM_ATTR_RESERVED_MEM_HIGH',
}
CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT = 0
CU_GRAPH_MEM_ATTR_USED_MEM_HIGH = 1
CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT = 2
CU_GRAPH_MEM_ATTR_RESERVED_MEM_HIGH = 3
CUgraphMem_attribute_enum = ctypes.c_uint32 # enum
CUgraphMem_attribute = CUgraphMem_attribute_enum
CUgraphMem_attribute__enumvalues = CUgraphMem_attribute_enum__enumvalues

# values for enumeration 'CUflushGPUDirectRDMAWritesOptions_enum'
CUflushGPUDirectRDMAWritesOptions_enum__enumvalues = {
    1: 'CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_HOST',
    2: 'CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_MEMOPS',
}
CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_HOST = 1
CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_MEMOPS = 2
CUflushGPUDirectRDMAWritesOptions_enum = ctypes.c_uint32 # enum
CUflushGPUDirectRDMAWritesOptions = CUflushGPUDirectRDMAWritesOptions_enum
CUflushGPUDirectRDMAWritesOptions__enumvalues = CUflushGPUDirectRDMAWritesOptions_enum__enumvalues

# values for enumeration 'CUGPUDirectRDMAWritesOrdering_enum'
CUGPUDirectRDMAWritesOrdering_enum__enumvalues = {
    0: 'CU_GPU_DIRECT_RDMA_WRITES_ORDERING_NONE',
    100: 'CU_GPU_DIRECT_RDMA_WRITES_ORDERING_OWNER',
    200: 'CU_GPU_DIRECT_RDMA_WRITES_ORDERING_ALL_DEVICES',
}
CU_GPU_DIRECT_RDMA_WRITES_ORDERING_NONE = 0
CU_GPU_DIRECT_RDMA_WRITES_ORDERING_OWNER = 100
CU_GPU_DIRECT_RDMA_WRITES_ORDERING_ALL_DEVICES = 200
CUGPUDirectRDMAWritesOrdering_enum = ctypes.c_uint32 # enum
CUGPUDirectRDMAWritesOrdering = CUGPUDirectRDMAWritesOrdering_enum
CUGPUDirectRDMAWritesOrdering__enumvalues = CUGPUDirectRDMAWritesOrdering_enum__enumvalues

# values for enumeration 'CUflushGPUDirectRDMAWritesScope_enum'
CUflushGPUDirectRDMAWritesScope_enum__enumvalues = {
    100: 'CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER',
    200: 'CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_ALL_DEVICES',
}
CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER = 100
CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_ALL_DEVICES = 200
CUflushGPUDirectRDMAWritesScope_enum = ctypes.c_uint32 # enum
CUflushGPUDirectRDMAWritesScope = CUflushGPUDirectRDMAWritesScope_enum
CUflushGPUDirectRDMAWritesScope__enumvalues = CUflushGPUDirectRDMAWritesScope_enum__enumvalues

# values for enumeration 'CUflushGPUDirectRDMAWritesTarget_enum'
CUflushGPUDirectRDMAWritesTarget_enum__enumvalues = {
    0: 'CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TARGET_CURRENT_CTX',
}
CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TARGET_CURRENT_CTX = 0
CUflushGPUDirectRDMAWritesTarget_enum = ctypes.c_uint32 # enum
CUflushGPUDirectRDMAWritesTarget = CUflushGPUDirectRDMAWritesTarget_enum
CUflushGPUDirectRDMAWritesTarget__enumvalues = CUflushGPUDirectRDMAWritesTarget_enum__enumvalues

# values for enumeration 'CUgraphDebugDot_flags_enum'
CUgraphDebugDot_flags_enum__enumvalues = {
    1: 'CU_GRAPH_DEBUG_DOT_FLAGS_VERBOSE',
    2: 'CU_GRAPH_DEBUG_DOT_FLAGS_RUNTIME_TYPES',
    4: 'CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_PARAMS',
    8: 'CU_GRAPH_DEBUG_DOT_FLAGS_MEMCPY_NODE_PARAMS',
    16: 'CU_GRAPH_DEBUG_DOT_FLAGS_MEMSET_NODE_PARAMS',
    32: 'CU_GRAPH_DEBUG_DOT_FLAGS_HOST_NODE_PARAMS',
    64: 'CU_GRAPH_DEBUG_DOT_FLAGS_EVENT_NODE_PARAMS',
    128: 'CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_SIGNAL_NODE_PARAMS',
    256: 'CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_WAIT_NODE_PARAMS',
    512: 'CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_ATTRIBUTES',
    1024: 'CU_GRAPH_DEBUG_DOT_FLAGS_HANDLES',
    2048: 'CU_GRAPH_DEBUG_DOT_FLAGS_MEM_ALLOC_NODE_PARAMS',
    4096: 'CU_GRAPH_DEBUG_DOT_FLAGS_MEM_FREE_NODE_PARAMS',
}
CU_GRAPH_DEBUG_DOT_FLAGS_VERBOSE = 1
CU_GRAPH_DEBUG_DOT_FLAGS_RUNTIME_TYPES = 2
CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_PARAMS = 4
CU_GRAPH_DEBUG_DOT_FLAGS_MEMCPY_NODE_PARAMS = 8
CU_GRAPH_DEBUG_DOT_FLAGS_MEMSET_NODE_PARAMS = 16
CU_GRAPH_DEBUG_DOT_FLAGS_HOST_NODE_PARAMS = 32
CU_GRAPH_DEBUG_DOT_FLAGS_EVENT_NODE_PARAMS = 64
CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_SIGNAL_NODE_PARAMS = 128
CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_WAIT_NODE_PARAMS = 256
CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_ATTRIBUTES = 512
CU_GRAPH_DEBUG_DOT_FLAGS_HANDLES = 1024
CU_GRAPH_DEBUG_DOT_FLAGS_MEM_ALLOC_NODE_PARAMS = 2048
CU_GRAPH_DEBUG_DOT_FLAGS_MEM_FREE_NODE_PARAMS = 4096
CUgraphDebugDot_flags_enum = ctypes.c_uint32 # enum
CUgraphDebugDot_flags = CUgraphDebugDot_flags_enum
CUgraphDebugDot_flags__enumvalues = CUgraphDebugDot_flags_enum__enumvalues

# values for enumeration 'CUuserObject_flags_enum'
CUuserObject_flags_enum__enumvalues = {
    1: 'CU_USER_OBJECT_NO_DESTRUCTOR_SYNC',
}
CU_USER_OBJECT_NO_DESTRUCTOR_SYNC = 1
CUuserObject_flags_enum = ctypes.c_uint32 # enum
CUuserObject_flags = CUuserObject_flags_enum
CUuserObject_flags__enumvalues = CUuserObject_flags_enum__enumvalues

# values for enumeration 'CUuserObjectRetain_flags_enum'
CUuserObjectRetain_flags_enum__enumvalues = {
    1: 'CU_GRAPH_USER_OBJECT_MOVE',
}
CU_GRAPH_USER_OBJECT_MOVE = 1
CUuserObjectRetain_flags_enum = ctypes.c_uint32 # enum
CUuserObjectRetain_flags = CUuserObjectRetain_flags_enum
CUuserObjectRetain_flags__enumvalues = CUuserObjectRetain_flags_enum__enumvalues

# values for enumeration 'CUgraphInstantiate_flags_enum'
CUgraphInstantiate_flags_enum__enumvalues = {
    1: 'CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH',
}
CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH = 1
CUgraphInstantiate_flags_enum = ctypes.c_uint32 # enum
CUgraphInstantiate_flags = CUgraphInstantiate_flags_enum
CUgraphInstantiate_flags__enumvalues = CUgraphInstantiate_flags_enum__enumvalues
try:
    cuGetErrorString = _libraries['libcuda.so'].cuGetErrorString
    cuGetErrorString.restype = CUresult
    cuGetErrorString.argtypes = [CUresult, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError:
    pass
try:
    cuGetErrorName = _libraries['libcuda.so'].cuGetErrorName
    cuGetErrorName.restype = CUresult
    cuGetErrorName.argtypes = [CUresult, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError:
    pass
try:
    cuInit = _libraries['libcuda.so'].cuInit
    cuInit.restype = CUresult
    cuInit.argtypes = [ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuDriverGetVersion = _libraries['libcuda.so'].cuDriverGetVersion
    cuDriverGetVersion.restype = CUresult
    cuDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    cuDeviceGet = _libraries['libcuda.so'].cuDeviceGet
    cuDeviceGet.restype = CUresult
    cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int32]
except AttributeError:
    pass
try:
    cuDeviceGetCount = _libraries['libcuda.so'].cuDeviceGetCount
    cuDeviceGetCount.restype = CUresult
    cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    cuDeviceGetName = _libraries['libcuda.so'].cuDeviceGetName
    cuDeviceGetName.restype = CUresult
    cuDeviceGetName.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.c_int32, CUdevice]
except AttributeError:
    pass
try:
    cuDeviceGetUuid = _libraries['libcuda.so'].cuDeviceGetUuid
    cuDeviceGetUuid.restype = CUresult
    cuDeviceGetUuid.argtypes = [ctypes.POINTER(struct_CUuuid_st), CUdevice]
except AttributeError:
    pass
try:
    cuDeviceGetUuid_v2 = _libraries['libcuda.so'].cuDeviceGetUuid_v2
    cuDeviceGetUuid_v2.restype = CUresult
    cuDeviceGetUuid_v2.argtypes = [ctypes.POINTER(struct_CUuuid_st), CUdevice]
except AttributeError:
    pass
try:
    cuDeviceGetLuid = _libraries['libcuda.so'].cuDeviceGetLuid
    cuDeviceGetLuid.restype = CUresult
    cuDeviceGetLuid.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_uint32), CUdevice]
except AttributeError:
    pass
try:
    cuDeviceTotalMem_v2 = _libraries['libcuda.so'].cuDeviceTotalMem_v2
    cuDeviceTotalMem_v2.restype = CUresult
    cuDeviceTotalMem_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), CUdevice]
except AttributeError:
    pass
try:
    cuDeviceGetTexture1DLinearMaxWidth = _libraries['libcuda.so'].cuDeviceGetTexture1DLinearMaxWidth
    cuDeviceGetTexture1DLinearMaxWidth.restype = CUresult
    cuDeviceGetTexture1DLinearMaxWidth.argtypes = [ctypes.POINTER(ctypes.c_uint64), CUarray_format, ctypes.c_uint32, CUdevice]
except AttributeError:
    pass
try:
    cuDeviceGetAttribute = _libraries['libcuda.so'].cuDeviceGetAttribute
    cuDeviceGetAttribute.restype = CUresult
    cuDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int32), CUdevice_attribute, CUdevice]
except AttributeError:
    pass
try:
    cuDeviceGetNvSciSyncAttributes = _libraries['libcuda.so'].cuDeviceGetNvSciSyncAttributes
    cuDeviceGetNvSciSyncAttributes.restype = CUresult
    cuDeviceGetNvSciSyncAttributes.argtypes = [ctypes.POINTER(None), CUdevice, ctypes.c_int32]
except AttributeError:
    pass
try:
    cuDeviceSetMemPool = _libraries['libcuda.so'].cuDeviceSetMemPool
    cuDeviceSetMemPool.restype = CUresult
    cuDeviceSetMemPool.argtypes = [CUdevice, CUmemoryPool]
except AttributeError:
    pass
try:
    cuDeviceGetMemPool = _libraries['libcuda.so'].cuDeviceGetMemPool
    cuDeviceGetMemPool.restype = CUresult
    cuDeviceGetMemPool.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUmemPoolHandle_st)), CUdevice]
except AttributeError:
    pass
try:
    cuDeviceGetDefaultMemPool = _libraries['libcuda.so'].cuDeviceGetDefaultMemPool
    cuDeviceGetDefaultMemPool.restype = CUresult
    cuDeviceGetDefaultMemPool.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUmemPoolHandle_st)), CUdevice]
except AttributeError:
    pass
try:
    cuFlushGPUDirectRDMAWrites = _libraries['libcuda.so'].cuFlushGPUDirectRDMAWrites
    cuFlushGPUDirectRDMAWrites.restype = CUresult
    cuFlushGPUDirectRDMAWrites.argtypes = [CUflushGPUDirectRDMAWritesTarget, CUflushGPUDirectRDMAWritesScope]
except AttributeError:
    pass
try:
    cuDeviceGetProperties = _libraries['libcuda.so'].cuDeviceGetProperties
    cuDeviceGetProperties.restype = CUresult
    cuDeviceGetProperties.argtypes = [ctypes.POINTER(struct_CUdevprop_st), CUdevice]
except AttributeError:
    pass
try:
    cuDeviceComputeCapability = _libraries['libcuda.so'].cuDeviceComputeCapability
    cuDeviceComputeCapability.restype = CUresult
    cuDeviceComputeCapability.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), CUdevice]
except AttributeError:
    pass
try:
    cuDevicePrimaryCtxRetain = _libraries['libcuda.so'].cuDevicePrimaryCtxRetain
    cuDevicePrimaryCtxRetain.restype = CUresult
    cuDevicePrimaryCtxRetain.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUctx_st)), CUdevice]
except AttributeError:
    pass
try:
    cuDevicePrimaryCtxRelease_v2 = _libraries['libcuda.so'].cuDevicePrimaryCtxRelease_v2
    cuDevicePrimaryCtxRelease_v2.restype = CUresult
    cuDevicePrimaryCtxRelease_v2.argtypes = [CUdevice]
except AttributeError:
    pass
try:
    cuDevicePrimaryCtxSetFlags_v2 = _libraries['libcuda.so'].cuDevicePrimaryCtxSetFlags_v2
    cuDevicePrimaryCtxSetFlags_v2.restype = CUresult
    cuDevicePrimaryCtxSetFlags_v2.argtypes = [CUdevice, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuDevicePrimaryCtxGetState = _libraries['libcuda.so'].cuDevicePrimaryCtxGetState
    cuDevicePrimaryCtxGetState.restype = CUresult
    cuDevicePrimaryCtxGetState.argtypes = [CUdevice, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    cuDevicePrimaryCtxReset_v2 = _libraries['libcuda.so'].cuDevicePrimaryCtxReset_v2
    cuDevicePrimaryCtxReset_v2.restype = CUresult
    cuDevicePrimaryCtxReset_v2.argtypes = [CUdevice]
except AttributeError:
    pass
try:
    cuDeviceGetExecAffinitySupport = _libraries['libcuda.so'].cuDeviceGetExecAffinitySupport
    cuDeviceGetExecAffinitySupport.restype = CUresult
    cuDeviceGetExecAffinitySupport.argtypes = [ctypes.POINTER(ctypes.c_int32), CUexecAffinityType, CUdevice]
except AttributeError:
    pass
try:
    cuCtxCreate_v2 = _libraries['libcuda.so'].cuCtxCreate_v2
    cuCtxCreate_v2.restype = CUresult
    cuCtxCreate_v2.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUctx_st)), ctypes.c_uint32, CUdevice]
except AttributeError:
    pass
try:
    cuCtxCreate_v3 = _libraries['libcuda.so'].cuCtxCreate_v3
    cuCtxCreate_v3.restype = CUresult
    cuCtxCreate_v3.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUctx_st)), ctypes.POINTER(struct_CUexecAffinityParam_st), ctypes.c_int32, ctypes.c_uint32, CUdevice]
except AttributeError:
    pass
try:
    cuCtxDestroy_v2 = _libraries['libcuda.so'].cuCtxDestroy_v2
    cuCtxDestroy_v2.restype = CUresult
    cuCtxDestroy_v2.argtypes = [CUcontext]
except AttributeError:
    pass
try:
    cuCtxPushCurrent_v2 = _libraries['libcuda.so'].cuCtxPushCurrent_v2
    cuCtxPushCurrent_v2.restype = CUresult
    cuCtxPushCurrent_v2.argtypes = [CUcontext]
except AttributeError:
    pass
try:
    cuCtxPopCurrent_v2 = _libraries['libcuda.so'].cuCtxPopCurrent_v2
    cuCtxPopCurrent_v2.restype = CUresult
    cuCtxPopCurrent_v2.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUctx_st))]
except AttributeError:
    pass
try:
    cuCtxSetCurrent = _libraries['libcuda.so'].cuCtxSetCurrent
    cuCtxSetCurrent.restype = CUresult
    cuCtxSetCurrent.argtypes = [CUcontext]
except AttributeError:
    pass
try:
    cuCtxGetCurrent = _libraries['libcuda.so'].cuCtxGetCurrent
    cuCtxGetCurrent.restype = CUresult
    cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUctx_st))]
except AttributeError:
    pass
try:
    cuCtxGetDevice = _libraries['libcuda.so'].cuCtxGetDevice
    cuCtxGetDevice.restype = CUresult
    cuCtxGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    cuCtxGetFlags = _libraries['libcuda.so'].cuCtxGetFlags
    cuCtxGetFlags.restype = CUresult
    cuCtxGetFlags.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    cuCtxSynchronize = _libraries['libcuda.so'].cuCtxSynchronize
    cuCtxSynchronize.restype = CUresult
    cuCtxSynchronize.argtypes = []
except AttributeError:
    pass
size_t = ctypes.c_uint64
try:
    cuCtxSetLimit = _libraries['libcuda.so'].cuCtxSetLimit
    cuCtxSetLimit.restype = CUresult
    cuCtxSetLimit.argtypes = [CUlimit, size_t]
except AttributeError:
    pass
try:
    cuCtxGetLimit = _libraries['libcuda.so'].cuCtxGetLimit
    cuCtxGetLimit.restype = CUresult
    cuCtxGetLimit.argtypes = [ctypes.POINTER(ctypes.c_uint64), CUlimit]
except AttributeError:
    pass
try:
    cuCtxGetCacheConfig = _libraries['libcuda.so'].cuCtxGetCacheConfig
    cuCtxGetCacheConfig.restype = CUresult
    cuCtxGetCacheConfig.argtypes = [ctypes.POINTER(CUfunc_cache_enum)]
except AttributeError:
    pass
try:
    cuCtxSetCacheConfig = _libraries['libcuda.so'].cuCtxSetCacheConfig
    cuCtxSetCacheConfig.restype = CUresult
    cuCtxSetCacheConfig.argtypes = [CUfunc_cache]
except AttributeError:
    pass
try:
    cuCtxGetSharedMemConfig = _libraries['libcuda.so'].cuCtxGetSharedMemConfig
    cuCtxGetSharedMemConfig.restype = CUresult
    cuCtxGetSharedMemConfig.argtypes = [ctypes.POINTER(CUsharedconfig_enum)]
except AttributeError:
    pass
try:
    cuCtxSetSharedMemConfig = _libraries['libcuda.so'].cuCtxSetSharedMemConfig
    cuCtxSetSharedMemConfig.restype = CUresult
    cuCtxSetSharedMemConfig.argtypes = [CUsharedconfig]
except AttributeError:
    pass
try:
    cuCtxGetApiVersion = _libraries['libcuda.so'].cuCtxGetApiVersion
    cuCtxGetApiVersion.restype = CUresult
    cuCtxGetApiVersion.argtypes = [CUcontext, ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    cuCtxGetStreamPriorityRange = _libraries['libcuda.so'].cuCtxGetStreamPriorityRange
    cuCtxGetStreamPriorityRange.restype = CUresult
    cuCtxGetStreamPriorityRange.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    cuCtxResetPersistingL2Cache = _libraries['libcuda.so'].cuCtxResetPersistingL2Cache
    cuCtxResetPersistingL2Cache.restype = CUresult
    cuCtxResetPersistingL2Cache.argtypes = []
except AttributeError:
    pass
try:
    cuCtxGetExecAffinity = _libraries['libcuda.so'].cuCtxGetExecAffinity
    cuCtxGetExecAffinity.restype = CUresult
    cuCtxGetExecAffinity.argtypes = [ctypes.POINTER(struct_CUexecAffinityParam_st), CUexecAffinityType]
except AttributeError:
    pass
try:
    cuCtxAttach = _libraries['libcuda.so'].cuCtxAttach
    cuCtxAttach.restype = CUresult
    cuCtxAttach.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUctx_st)), ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuCtxDetach = _libraries['libcuda.so'].cuCtxDetach
    cuCtxDetach.restype = CUresult
    cuCtxDetach.argtypes = [CUcontext]
except AttributeError:
    pass
try:
    cuModuleLoad = _libraries['libcuda.so'].cuModuleLoad
    cuModuleLoad.restype = CUresult
    cuModuleLoad.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUmod_st)), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    cuModuleLoadData = _libraries['libcuda.so'].cuModuleLoadData
    cuModuleLoadData.restype = CUresult
    cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUmod_st)), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    cuModuleLoadDataEx = _libraries['libcuda.so'].cuModuleLoadDataEx
    cuModuleLoadDataEx.restype = CUresult
    cuModuleLoadDataEx.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUmod_st)), ctypes.POINTER(None), ctypes.c_uint32, ctypes.POINTER(CUjit_option_enum), ctypes.POINTER(ctypes.POINTER(None))]
except AttributeError:
    pass
try:
    cuModuleLoadFatBinary = _libraries['libcuda.so'].cuModuleLoadFatBinary
    cuModuleLoadFatBinary.restype = CUresult
    cuModuleLoadFatBinary.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUmod_st)), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    cuModuleUnload = _libraries['libcuda.so'].cuModuleUnload
    cuModuleUnload.restype = CUresult
    cuModuleUnload.argtypes = [CUmodule]
except AttributeError:
    pass
try:
    cuModuleGetFunction = _libraries['libcuda.so'].cuModuleGetFunction
    cuModuleGetFunction.restype = CUresult
    cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUfunc_st)), CUmodule, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    cuModuleGetGlobal_v2 = _libraries['libcuda.so'].cuModuleGetGlobal_v2
    cuModuleGetGlobal_v2.restype = CUresult
    cuModuleGetGlobal_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), CUmodule, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    cuModuleGetTexRef = _libraries['libcuda.so'].cuModuleGetTexRef
    cuModuleGetTexRef.restype = CUresult
    cuModuleGetTexRef.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUtexref_st)), CUmodule, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    cuModuleGetSurfRef = _libraries['libcuda.so'].cuModuleGetSurfRef
    cuModuleGetSurfRef.restype = CUresult
    cuModuleGetSurfRef.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUsurfref_st)), CUmodule, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    cuLinkCreate_v2 = _libraries['libcuda.so'].cuLinkCreate_v2
    cuLinkCreate_v2.restype = CUresult
    cuLinkCreate_v2.argtypes = [ctypes.c_uint32, ctypes.POINTER(CUjit_option_enum), ctypes.POINTER(ctypes.POINTER(None)), ctypes.POINTER(ctypes.POINTER(struct_CUlinkState_st))]
except AttributeError:
    pass
try:
    cuLinkAddData_v2 = _libraries['libcuda.so'].cuLinkAddData_v2
    cuLinkAddData_v2.restype = CUresult
    cuLinkAddData_v2.argtypes = [CUlinkState, CUjitInputType, ctypes.POINTER(None), size_t, ctypes.POINTER(ctypes.c_char), ctypes.c_uint32, ctypes.POINTER(CUjit_option_enum), ctypes.POINTER(ctypes.POINTER(None))]
except AttributeError:
    pass
try:
    cuLinkAddFile_v2 = _libraries['libcuda.so'].cuLinkAddFile_v2
    cuLinkAddFile_v2.restype = CUresult
    cuLinkAddFile_v2.argtypes = [CUlinkState, CUjitInputType, ctypes.POINTER(ctypes.c_char), ctypes.c_uint32, ctypes.POINTER(CUjit_option_enum), ctypes.POINTER(ctypes.POINTER(None))]
except AttributeError:
    pass
try:
    cuLinkComplete = _libraries['libcuda.so'].cuLinkComplete
    cuLinkComplete.restype = CUresult
    cuLinkComplete.argtypes = [CUlinkState, ctypes.POINTER(ctypes.POINTER(None)), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    cuLinkDestroy = _libraries['libcuda.so'].cuLinkDestroy
    cuLinkDestroy.restype = CUresult
    cuLinkDestroy.argtypes = [CUlinkState]
except AttributeError:
    pass
try:
    cuMemGetInfo_v2 = _libraries['libcuda.so'].cuMemGetInfo_v2
    cuMemGetInfo_v2.restype = CUresult
    cuMemGetInfo_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    cuMemAlloc_v2 = _libraries['libcuda.so'].cuMemAlloc_v2
    cuMemAlloc_v2.restype = CUresult
    cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), size_t]
except AttributeError:
    pass
try:
    cuMemAllocPitch_v2 = _libraries['libcuda.so'].cuMemAllocPitch_v2
    cuMemAllocPitch_v2.restype = CUresult
    cuMemAllocPitch_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), size_t, size_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuMemFree_v2 = _libraries['libcuda.so'].cuMemFree_v2
    cuMemFree_v2.restype = CUresult
    cuMemFree_v2.argtypes = [CUdeviceptr]
except AttributeError:
    pass
try:
    cuMemGetAddressRange_v2 = _libraries['libcuda.so'].cuMemGetAddressRange_v2
    cuMemGetAddressRange_v2.restype = CUresult
    cuMemGetAddressRange_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), CUdeviceptr]
except AttributeError:
    pass
try:
    cuMemAllocHost_v2 = _libraries['libcuda.so'].cuMemAllocHost_v2
    cuMemAllocHost_v2.restype = CUresult
    cuMemAllocHost_v2.argtypes = [ctypes.POINTER(ctypes.POINTER(None)), size_t]
except AttributeError:
    pass
try:
    cuMemFreeHost = _libraries['libcuda.so'].cuMemFreeHost
    cuMemFreeHost.restype = CUresult
    cuMemFreeHost.argtypes = [ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    cuMemHostAlloc = _libraries['libcuda.so'].cuMemHostAlloc
    cuMemHostAlloc.restype = CUresult
    cuMemHostAlloc.argtypes = [ctypes.POINTER(ctypes.POINTER(None)), size_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuMemHostGetDevicePointer_v2 = _libraries['libcuda.so'].cuMemHostGetDevicePointer_v2
    cuMemHostGetDevicePointer_v2.restype = CUresult
    cuMemHostGetDevicePointer_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(None), ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuMemHostGetFlags = _libraries['libcuda.so'].cuMemHostGetFlags
    cuMemHostGetFlags.restype = CUresult
    cuMemHostGetFlags.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    cuMemAllocManaged = _libraries['libcuda.so'].cuMemAllocManaged
    cuMemAllocManaged.restype = CUresult
    cuMemAllocManaged.argtypes = [ctypes.POINTER(ctypes.c_uint64), size_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuDeviceGetByPCIBusId = _libraries['libcuda.so'].cuDeviceGetByPCIBusId
    cuDeviceGetByPCIBusId.restype = CUresult
    cuDeviceGetByPCIBusId.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    cuDeviceGetPCIBusId = _libraries['libcuda.so'].cuDeviceGetPCIBusId
    cuDeviceGetPCIBusId.restype = CUresult
    cuDeviceGetPCIBusId.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.c_int32, CUdevice]
except AttributeError:
    pass
try:
    cuIpcGetEventHandle = _libraries['libcuda.so'].cuIpcGetEventHandle
    cuIpcGetEventHandle.restype = CUresult
    cuIpcGetEventHandle.argtypes = [ctypes.POINTER(struct_CUipcEventHandle_st), CUevent]
except AttributeError:
    pass
try:
    cuIpcOpenEventHandle = _libraries['libcuda.so'].cuIpcOpenEventHandle
    cuIpcOpenEventHandle.restype = CUresult
    cuIpcOpenEventHandle.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUevent_st)), CUipcEventHandle]
except AttributeError:
    pass
try:
    cuIpcGetMemHandle = _libraries['libcuda.so'].cuIpcGetMemHandle
    cuIpcGetMemHandle.restype = CUresult
    cuIpcGetMemHandle.argtypes = [ctypes.POINTER(struct_CUipcMemHandle_st), CUdeviceptr]
except AttributeError:
    pass
try:
    cuIpcOpenMemHandle_v2 = _libraries['libcuda.so'].cuIpcOpenMemHandle_v2
    cuIpcOpenMemHandle_v2.restype = CUresult
    cuIpcOpenMemHandle_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), CUipcMemHandle, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuIpcCloseMemHandle = _libraries['libcuda.so'].cuIpcCloseMemHandle
    cuIpcCloseMemHandle.restype = CUresult
    cuIpcCloseMemHandle.argtypes = [CUdeviceptr]
except AttributeError:
    pass
try:
    cuMemHostRegister_v2 = _libraries['libcuda.so'].cuMemHostRegister_v2
    cuMemHostRegister_v2.restype = CUresult
    cuMemHostRegister_v2.argtypes = [ctypes.POINTER(None), size_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuMemHostUnregister = _libraries['libcuda.so'].cuMemHostUnregister
    cuMemHostUnregister.restype = CUresult
    cuMemHostUnregister.argtypes = [ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    cuMemcpy = _libraries['libcuda.so'].cuMemcpy
    cuMemcpy.restype = CUresult
    cuMemcpy.argtypes = [CUdeviceptr, CUdeviceptr, size_t]
except AttributeError:
    pass
try:
    cuMemcpyPeer = _libraries['libcuda.so'].cuMemcpyPeer
    cuMemcpyPeer.restype = CUresult
    cuMemcpyPeer.argtypes = [CUdeviceptr, CUcontext, CUdeviceptr, CUcontext, size_t]
except AttributeError:
    pass
try:
    cuMemcpyHtoD_v2 = _libraries['libcuda.so'].cuMemcpyHtoD_v2
    cuMemcpyHtoD_v2.restype = CUresult
    cuMemcpyHtoD_v2.argtypes = [CUdeviceptr, ctypes.POINTER(None), size_t]
except AttributeError:
    pass
try:
    cuMemcpyDtoH_v2 = _libraries['libcuda.so'].cuMemcpyDtoH_v2
    cuMemcpyDtoH_v2.restype = CUresult
    cuMemcpyDtoH_v2.argtypes = [ctypes.POINTER(None), CUdeviceptr, size_t]
except AttributeError:
    pass
try:
    cuMemcpyDtoD_v2 = _libraries['libcuda.so'].cuMemcpyDtoD_v2
    cuMemcpyDtoD_v2.restype = CUresult
    cuMemcpyDtoD_v2.argtypes = [CUdeviceptr, CUdeviceptr, size_t]
except AttributeError:
    pass
try:
    cuMemcpyDtoA_v2 = _libraries['libcuda.so'].cuMemcpyDtoA_v2
    cuMemcpyDtoA_v2.restype = CUresult
    cuMemcpyDtoA_v2.argtypes = [CUarray, size_t, CUdeviceptr, size_t]
except AttributeError:
    pass
try:
    cuMemcpyAtoD_v2 = _libraries['libcuda.so'].cuMemcpyAtoD_v2
    cuMemcpyAtoD_v2.restype = CUresult
    cuMemcpyAtoD_v2.argtypes = [CUdeviceptr, CUarray, size_t, size_t]
except AttributeError:
    pass
try:
    cuMemcpyHtoA_v2 = _libraries['libcuda.so'].cuMemcpyHtoA_v2
    cuMemcpyHtoA_v2.restype = CUresult
    cuMemcpyHtoA_v2.argtypes = [CUarray, size_t, ctypes.POINTER(None), size_t]
except AttributeError:
    pass
try:
    cuMemcpyAtoH_v2 = _libraries['libcuda.so'].cuMemcpyAtoH_v2
    cuMemcpyAtoH_v2.restype = CUresult
    cuMemcpyAtoH_v2.argtypes = [ctypes.POINTER(None), CUarray, size_t, size_t]
except AttributeError:
    pass
try:
    cuMemcpyAtoA_v2 = _libraries['libcuda.so'].cuMemcpyAtoA_v2
    cuMemcpyAtoA_v2.restype = CUresult
    cuMemcpyAtoA_v2.argtypes = [CUarray, size_t, CUarray, size_t, size_t]
except AttributeError:
    pass
try:
    cuMemcpy2D_v2 = _libraries['libcuda.so'].cuMemcpy2D_v2
    cuMemcpy2D_v2.restype = CUresult
    cuMemcpy2D_v2.argtypes = [ctypes.POINTER(struct_CUDA_MEMCPY2D_st)]
except AttributeError:
    pass
try:
    cuMemcpy2DUnaligned_v2 = _libraries['libcuda.so'].cuMemcpy2DUnaligned_v2
    cuMemcpy2DUnaligned_v2.restype = CUresult
    cuMemcpy2DUnaligned_v2.argtypes = [ctypes.POINTER(struct_CUDA_MEMCPY2D_st)]
except AttributeError:
    pass
try:
    cuMemcpy3D_v2 = _libraries['libcuda.so'].cuMemcpy3D_v2
    cuMemcpy3D_v2.restype = CUresult
    cuMemcpy3D_v2.argtypes = [ctypes.POINTER(struct_CUDA_MEMCPY3D_st)]
except AttributeError:
    pass
try:
    cuMemcpy3DPeer = _libraries['libcuda.so'].cuMemcpy3DPeer
    cuMemcpy3DPeer.restype = CUresult
    cuMemcpy3DPeer.argtypes = [ctypes.POINTER(struct_CUDA_MEMCPY3D_PEER_st)]
except AttributeError:
    pass
try:
    cuMemcpyAsync = _libraries['libcuda.so'].cuMemcpyAsync
    cuMemcpyAsync.restype = CUresult
    cuMemcpyAsync.argtypes = [CUdeviceptr, CUdeviceptr, size_t, CUstream]
except AttributeError:
    pass
try:
    cuMemcpyPeerAsync = _libraries['libcuda.so'].cuMemcpyPeerAsync
    cuMemcpyPeerAsync.restype = CUresult
    cuMemcpyPeerAsync.argtypes = [CUdeviceptr, CUcontext, CUdeviceptr, CUcontext, size_t, CUstream]
except AttributeError:
    pass
try:
    cuMemcpyHtoDAsync_v2 = _libraries['libcuda.so'].cuMemcpyHtoDAsync_v2
    cuMemcpyHtoDAsync_v2.restype = CUresult
    cuMemcpyHtoDAsync_v2.argtypes = [CUdeviceptr, ctypes.POINTER(None), size_t, CUstream]
except AttributeError:
    pass
try:
    cuMemcpyDtoHAsync_v2 = _libraries['libcuda.so'].cuMemcpyDtoHAsync_v2
    cuMemcpyDtoHAsync_v2.restype = CUresult
    cuMemcpyDtoHAsync_v2.argtypes = [ctypes.POINTER(None), CUdeviceptr, size_t, CUstream]
except AttributeError:
    pass
try:
    cuMemcpyDtoDAsync_v2 = _libraries['libcuda.so'].cuMemcpyDtoDAsync_v2
    cuMemcpyDtoDAsync_v2.restype = CUresult
    cuMemcpyDtoDAsync_v2.argtypes = [CUdeviceptr, CUdeviceptr, size_t, CUstream]
except AttributeError:
    pass
try:
    cuMemcpyHtoAAsync_v2 = _libraries['libcuda.so'].cuMemcpyHtoAAsync_v2
    cuMemcpyHtoAAsync_v2.restype = CUresult
    cuMemcpyHtoAAsync_v2.argtypes = [CUarray, size_t, ctypes.POINTER(None), size_t, CUstream]
except AttributeError:
    pass
try:
    cuMemcpyAtoHAsync_v2 = _libraries['libcuda.so'].cuMemcpyAtoHAsync_v2
    cuMemcpyAtoHAsync_v2.restype = CUresult
    cuMemcpyAtoHAsync_v2.argtypes = [ctypes.POINTER(None), CUarray, size_t, size_t, CUstream]
except AttributeError:
    pass
try:
    cuMemcpy2DAsync_v2 = _libraries['libcuda.so'].cuMemcpy2DAsync_v2
    cuMemcpy2DAsync_v2.restype = CUresult
    cuMemcpy2DAsync_v2.argtypes = [ctypes.POINTER(struct_CUDA_MEMCPY2D_st), CUstream]
except AttributeError:
    pass
try:
    cuMemcpy3DAsync_v2 = _libraries['libcuda.so'].cuMemcpy3DAsync_v2
    cuMemcpy3DAsync_v2.restype = CUresult
    cuMemcpy3DAsync_v2.argtypes = [ctypes.POINTER(struct_CUDA_MEMCPY3D_st), CUstream]
except AttributeError:
    pass
try:
    cuMemcpy3DPeerAsync = _libraries['libcuda.so'].cuMemcpy3DPeerAsync
    cuMemcpy3DPeerAsync.restype = CUresult
    cuMemcpy3DPeerAsync.argtypes = [ctypes.POINTER(struct_CUDA_MEMCPY3D_PEER_st), CUstream]
except AttributeError:
    pass
try:
    cuMemsetD8_v2 = _libraries['libcuda.so'].cuMemsetD8_v2
    cuMemsetD8_v2.restype = CUresult
    cuMemsetD8_v2.argtypes = [CUdeviceptr, ctypes.c_ubyte, size_t]
except AttributeError:
    pass
try:
    cuMemsetD16_v2 = _libraries['libcuda.so'].cuMemsetD16_v2
    cuMemsetD16_v2.restype = CUresult
    cuMemsetD16_v2.argtypes = [CUdeviceptr, ctypes.c_uint16, size_t]
except AttributeError:
    pass
try:
    cuMemsetD32_v2 = _libraries['libcuda.so'].cuMemsetD32_v2
    cuMemsetD32_v2.restype = CUresult
    cuMemsetD32_v2.argtypes = [CUdeviceptr, ctypes.c_uint32, size_t]
except AttributeError:
    pass
try:
    cuMemsetD2D8_v2 = _libraries['libcuda.so'].cuMemsetD2D8_v2
    cuMemsetD2D8_v2.restype = CUresult
    cuMemsetD2D8_v2.argtypes = [CUdeviceptr, size_t, ctypes.c_ubyte, size_t, size_t]
except AttributeError:
    pass
try:
    cuMemsetD2D16_v2 = _libraries['libcuda.so'].cuMemsetD2D16_v2
    cuMemsetD2D16_v2.restype = CUresult
    cuMemsetD2D16_v2.argtypes = [CUdeviceptr, size_t, ctypes.c_uint16, size_t, size_t]
except AttributeError:
    pass
try:
    cuMemsetD2D32_v2 = _libraries['libcuda.so'].cuMemsetD2D32_v2
    cuMemsetD2D32_v2.restype = CUresult
    cuMemsetD2D32_v2.argtypes = [CUdeviceptr, size_t, ctypes.c_uint32, size_t, size_t]
except AttributeError:
    pass
try:
    cuMemsetD8Async = _libraries['libcuda.so'].cuMemsetD8Async
    cuMemsetD8Async.restype = CUresult
    cuMemsetD8Async.argtypes = [CUdeviceptr, ctypes.c_ubyte, size_t, CUstream]
except AttributeError:
    pass
try:
    cuMemsetD16Async = _libraries['libcuda.so'].cuMemsetD16Async
    cuMemsetD16Async.restype = CUresult
    cuMemsetD16Async.argtypes = [CUdeviceptr, ctypes.c_uint16, size_t, CUstream]
except AttributeError:
    pass
try:
    cuMemsetD32Async = _libraries['libcuda.so'].cuMemsetD32Async
    cuMemsetD32Async.restype = CUresult
    cuMemsetD32Async.argtypes = [CUdeviceptr, ctypes.c_uint32, size_t, CUstream]
except AttributeError:
    pass
try:
    cuMemsetD2D8Async = _libraries['libcuda.so'].cuMemsetD2D8Async
    cuMemsetD2D8Async.restype = CUresult
    cuMemsetD2D8Async.argtypes = [CUdeviceptr, size_t, ctypes.c_ubyte, size_t, size_t, CUstream]
except AttributeError:
    pass
try:
    cuMemsetD2D16Async = _libraries['libcuda.so'].cuMemsetD2D16Async
    cuMemsetD2D16Async.restype = CUresult
    cuMemsetD2D16Async.argtypes = [CUdeviceptr, size_t, ctypes.c_uint16, size_t, size_t, CUstream]
except AttributeError:
    pass
try:
    cuMemsetD2D32Async = _libraries['libcuda.so'].cuMemsetD2D32Async
    cuMemsetD2D32Async.restype = CUresult
    cuMemsetD2D32Async.argtypes = [CUdeviceptr, size_t, ctypes.c_uint32, size_t, size_t, CUstream]
except AttributeError:
    pass
try:
    cuArrayCreate_v2 = _libraries['libcuda.so'].cuArrayCreate_v2
    cuArrayCreate_v2.restype = CUresult
    cuArrayCreate_v2.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUarray_st)), ctypes.POINTER(struct_CUDA_ARRAY_DESCRIPTOR_st)]
except AttributeError:
    pass
try:
    cuArrayGetDescriptor_v2 = _libraries['libcuda.so'].cuArrayGetDescriptor_v2
    cuArrayGetDescriptor_v2.restype = CUresult
    cuArrayGetDescriptor_v2.argtypes = [ctypes.POINTER(struct_CUDA_ARRAY_DESCRIPTOR_st), CUarray]
except AttributeError:
    pass
try:
    cuArrayGetSparseProperties = _libraries['libcuda.so'].cuArrayGetSparseProperties
    cuArrayGetSparseProperties.restype = CUresult
    cuArrayGetSparseProperties.argtypes = [ctypes.POINTER(struct_CUDA_ARRAY_SPARSE_PROPERTIES_st), CUarray]
except AttributeError:
    pass
try:
    cuMipmappedArrayGetSparseProperties = _libraries['libcuda.so'].cuMipmappedArrayGetSparseProperties
    cuMipmappedArrayGetSparseProperties.restype = CUresult
    cuMipmappedArrayGetSparseProperties.argtypes = [ctypes.POINTER(struct_CUDA_ARRAY_SPARSE_PROPERTIES_st), CUmipmappedArray]
except AttributeError:
    pass
try:
    cuArrayGetPlane = _libraries['libcuda.so'].cuArrayGetPlane
    cuArrayGetPlane.restype = CUresult
    cuArrayGetPlane.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUarray_st)), CUarray, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuArrayDestroy = _libraries['libcuda.so'].cuArrayDestroy
    cuArrayDestroy.restype = CUresult
    cuArrayDestroy.argtypes = [CUarray]
except AttributeError:
    pass
try:
    cuArray3DCreate_v2 = _libraries['libcuda.so'].cuArray3DCreate_v2
    cuArray3DCreate_v2.restype = CUresult
    cuArray3DCreate_v2.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUarray_st)), ctypes.POINTER(struct_CUDA_ARRAY3D_DESCRIPTOR_st)]
except AttributeError:
    pass
try:
    cuArray3DGetDescriptor_v2 = _libraries['libcuda.so'].cuArray3DGetDescriptor_v2
    cuArray3DGetDescriptor_v2.restype = CUresult
    cuArray3DGetDescriptor_v2.argtypes = [ctypes.POINTER(struct_CUDA_ARRAY3D_DESCRIPTOR_st), CUarray]
except AttributeError:
    pass
try:
    cuMipmappedArrayCreate = _libraries['libcuda.so'].cuMipmappedArrayCreate
    cuMipmappedArrayCreate.restype = CUresult
    cuMipmappedArrayCreate.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUmipmappedArray_st)), ctypes.POINTER(struct_CUDA_ARRAY3D_DESCRIPTOR_st), ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuMipmappedArrayGetLevel = _libraries['libcuda.so'].cuMipmappedArrayGetLevel
    cuMipmappedArrayGetLevel.restype = CUresult
    cuMipmappedArrayGetLevel.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUarray_st)), CUmipmappedArray, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuMipmappedArrayDestroy = _libraries['libcuda.so'].cuMipmappedArrayDestroy
    cuMipmappedArrayDestroy.restype = CUresult
    cuMipmappedArrayDestroy.argtypes = [CUmipmappedArray]
except AttributeError:
    pass
try:
    cuMemAddressReserve = _libraries['libcuda.so'].cuMemAddressReserve
    cuMemAddressReserve.restype = CUresult
    cuMemAddressReserve.argtypes = [ctypes.POINTER(ctypes.c_uint64), size_t, size_t, CUdeviceptr, ctypes.c_uint64]
except AttributeError:
    pass
try:
    cuMemAddressFree = _libraries['libcuda.so'].cuMemAddressFree
    cuMemAddressFree.restype = CUresult
    cuMemAddressFree.argtypes = [CUdeviceptr, size_t]
except AttributeError:
    pass
try:
    cuMemCreate = _libraries['libcuda.so'].cuMemCreate
    cuMemCreate.restype = CUresult
    cuMemCreate.argtypes = [ctypes.POINTER(ctypes.c_uint64), size_t, ctypes.POINTER(struct_CUmemAllocationProp_st), ctypes.c_uint64]
except AttributeError:
    pass
try:
    cuMemRelease = _libraries['libcuda.so'].cuMemRelease
    cuMemRelease.restype = CUresult
    cuMemRelease.argtypes = [CUmemGenericAllocationHandle]
except AttributeError:
    pass
try:
    cuMemMap = _libraries['libcuda.so'].cuMemMap
    cuMemMap.restype = CUresult
    cuMemMap.argtypes = [CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle, ctypes.c_uint64]
except AttributeError:
    pass
try:
    cuMemMapArrayAsync = _libraries['libcuda.so'].cuMemMapArrayAsync
    cuMemMapArrayAsync.restype = CUresult
    cuMemMapArrayAsync.argtypes = [ctypes.POINTER(struct_CUarrayMapInfo_st), ctypes.c_uint32, CUstream]
except AttributeError:
    pass
try:
    cuMemUnmap = _libraries['libcuda.so'].cuMemUnmap
    cuMemUnmap.restype = CUresult
    cuMemUnmap.argtypes = [CUdeviceptr, size_t]
except AttributeError:
    pass
try:
    cuMemSetAccess = _libraries['libcuda.so'].cuMemSetAccess
    cuMemSetAccess.restype = CUresult
    cuMemSetAccess.argtypes = [CUdeviceptr, size_t, ctypes.POINTER(struct_CUmemAccessDesc_st), size_t]
except AttributeError:
    pass
try:
    cuMemGetAccess = _libraries['libcuda.so'].cuMemGetAccess
    cuMemGetAccess.restype = CUresult
    cuMemGetAccess.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(struct_CUmemLocation_st), CUdeviceptr]
except AttributeError:
    pass
try:
    cuMemExportToShareableHandle = _libraries['libcuda.so'].cuMemExportToShareableHandle
    cuMemExportToShareableHandle.restype = CUresult
    cuMemExportToShareableHandle.argtypes = [ctypes.POINTER(None), CUmemGenericAllocationHandle, CUmemAllocationHandleType, ctypes.c_uint64]
except AttributeError:
    pass
try:
    cuMemImportFromShareableHandle = _libraries['libcuda.so'].cuMemImportFromShareableHandle
    cuMemImportFromShareableHandle.restype = CUresult
    cuMemImportFromShareableHandle.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(None), CUmemAllocationHandleType]
except AttributeError:
    pass
try:
    cuMemGetAllocationGranularity = _libraries['libcuda.so'].cuMemGetAllocationGranularity
    cuMemGetAllocationGranularity.restype = CUresult
    cuMemGetAllocationGranularity.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(struct_CUmemAllocationProp_st), CUmemAllocationGranularity_flags]
except AttributeError:
    pass
try:
    cuMemGetAllocationPropertiesFromHandle = _libraries['libcuda.so'].cuMemGetAllocationPropertiesFromHandle
    cuMemGetAllocationPropertiesFromHandle.restype = CUresult
    cuMemGetAllocationPropertiesFromHandle.argtypes = [ctypes.POINTER(struct_CUmemAllocationProp_st), CUmemGenericAllocationHandle]
except AttributeError:
    pass
try:
    cuMemRetainAllocationHandle = _libraries['libcuda.so'].cuMemRetainAllocationHandle
    cuMemRetainAllocationHandle.restype = CUresult
    cuMemRetainAllocationHandle.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    cuMemFreeAsync = _libraries['libcuda.so'].cuMemFreeAsync
    cuMemFreeAsync.restype = CUresult
    cuMemFreeAsync.argtypes = [CUdeviceptr, CUstream]
except AttributeError:
    pass
try:
    cuMemAllocAsync = _libraries['libcuda.so'].cuMemAllocAsync
    cuMemAllocAsync.restype = CUresult
    cuMemAllocAsync.argtypes = [ctypes.POINTER(ctypes.c_uint64), size_t, CUstream]
except AttributeError:
    pass
try:
    cuMemPoolTrimTo = _libraries['libcuda.so'].cuMemPoolTrimTo
    cuMemPoolTrimTo.restype = CUresult
    cuMemPoolTrimTo.argtypes = [CUmemoryPool, size_t]
except AttributeError:
    pass
try:
    cuMemPoolSetAttribute = _libraries['libcuda.so'].cuMemPoolSetAttribute
    cuMemPoolSetAttribute.restype = CUresult
    cuMemPoolSetAttribute.argtypes = [CUmemoryPool, CUmemPool_attribute, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    cuMemPoolGetAttribute = _libraries['libcuda.so'].cuMemPoolGetAttribute
    cuMemPoolGetAttribute.restype = CUresult
    cuMemPoolGetAttribute.argtypes = [CUmemoryPool, CUmemPool_attribute, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    cuMemPoolSetAccess = _libraries['libcuda.so'].cuMemPoolSetAccess
    cuMemPoolSetAccess.restype = CUresult
    cuMemPoolSetAccess.argtypes = [CUmemoryPool, ctypes.POINTER(struct_CUmemAccessDesc_st), size_t]
except AttributeError:
    pass
try:
    cuMemPoolGetAccess = _libraries['libcuda.so'].cuMemPoolGetAccess
    cuMemPoolGetAccess.restype = CUresult
    cuMemPoolGetAccess.argtypes = [ctypes.POINTER(CUmemAccess_flags_enum), CUmemoryPool, ctypes.POINTER(struct_CUmemLocation_st)]
except AttributeError:
    pass
try:
    cuMemPoolCreate = _libraries['libcuda.so'].cuMemPoolCreate
    cuMemPoolCreate.restype = CUresult
    cuMemPoolCreate.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUmemPoolHandle_st)), ctypes.POINTER(struct_CUmemPoolProps_st)]
except AttributeError:
    pass
try:
    cuMemPoolDestroy = _libraries['libcuda.so'].cuMemPoolDestroy
    cuMemPoolDestroy.restype = CUresult
    cuMemPoolDestroy.argtypes = [CUmemoryPool]
except AttributeError:
    pass
try:
    cuMemAllocFromPoolAsync = _libraries['libcuda.so'].cuMemAllocFromPoolAsync
    cuMemAllocFromPoolAsync.restype = CUresult
    cuMemAllocFromPoolAsync.argtypes = [ctypes.POINTER(ctypes.c_uint64), size_t, CUmemoryPool, CUstream]
except AttributeError:
    pass
try:
    cuMemPoolExportToShareableHandle = _libraries['libcuda.so'].cuMemPoolExportToShareableHandle
    cuMemPoolExportToShareableHandle.restype = CUresult
    cuMemPoolExportToShareableHandle.argtypes = [ctypes.POINTER(None), CUmemoryPool, CUmemAllocationHandleType, ctypes.c_uint64]
except AttributeError:
    pass
try:
    cuMemPoolImportFromShareableHandle = _libraries['libcuda.so'].cuMemPoolImportFromShareableHandle
    cuMemPoolImportFromShareableHandle.restype = CUresult
    cuMemPoolImportFromShareableHandle.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUmemPoolHandle_st)), ctypes.POINTER(None), CUmemAllocationHandleType, ctypes.c_uint64]
except AttributeError:
    pass
try:
    cuMemPoolExportPointer = _libraries['libcuda.so'].cuMemPoolExportPointer
    cuMemPoolExportPointer.restype = CUresult
    cuMemPoolExportPointer.argtypes = [ctypes.POINTER(struct_CUmemPoolPtrExportData_st), CUdeviceptr]
except AttributeError:
    pass
try:
    cuMemPoolImportPointer = _libraries['libcuda.so'].cuMemPoolImportPointer
    cuMemPoolImportPointer.restype = CUresult
    cuMemPoolImportPointer.argtypes = [ctypes.POINTER(ctypes.c_uint64), CUmemoryPool, ctypes.POINTER(struct_CUmemPoolPtrExportData_st)]
except AttributeError:
    pass
try:
    cuPointerGetAttribute = _libraries['libcuda.so'].cuPointerGetAttribute
    cuPointerGetAttribute.restype = CUresult
    cuPointerGetAttribute.argtypes = [ctypes.POINTER(None), CUpointer_attribute, CUdeviceptr]
except AttributeError:
    pass
try:
    cuMemPrefetchAsync = _libraries['libcuda.so'].cuMemPrefetchAsync
    cuMemPrefetchAsync.restype = CUresult
    cuMemPrefetchAsync.argtypes = [CUdeviceptr, size_t, CUdevice, CUstream]
except AttributeError:
    pass
try:
    cuMemAdvise = _libraries['libcuda.so'].cuMemAdvise
    cuMemAdvise.restype = CUresult
    cuMemAdvise.argtypes = [CUdeviceptr, size_t, CUmem_advise, CUdevice]
except AttributeError:
    pass
try:
    cuMemRangeGetAttribute = _libraries['libcuda.so'].cuMemRangeGetAttribute
    cuMemRangeGetAttribute.restype = CUresult
    cuMemRangeGetAttribute.argtypes = [ctypes.POINTER(None), size_t, CUmem_range_attribute, CUdeviceptr, size_t]
except AttributeError:
    pass
try:
    cuMemRangeGetAttributes = _libraries['libcuda.so'].cuMemRangeGetAttributes
    cuMemRangeGetAttributes.restype = CUresult
    cuMemRangeGetAttributes.argtypes = [ctypes.POINTER(ctypes.POINTER(None)), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(CUmem_range_attribute_enum), size_t, CUdeviceptr, size_t]
except AttributeError:
    pass
try:
    cuPointerSetAttribute = _libraries['libcuda.so'].cuPointerSetAttribute
    cuPointerSetAttribute.restype = CUresult
    cuPointerSetAttribute.argtypes = [ctypes.POINTER(None), CUpointer_attribute, CUdeviceptr]
except AttributeError:
    pass
try:
    cuPointerGetAttributes = _libraries['libcuda.so'].cuPointerGetAttributes
    cuPointerGetAttributes.restype = CUresult
    cuPointerGetAttributes.argtypes = [ctypes.c_uint32, ctypes.POINTER(CUpointer_attribute_enum), ctypes.POINTER(ctypes.POINTER(None)), CUdeviceptr]
except AttributeError:
    pass
try:
    cuStreamCreate = _libraries['libcuda.so'].cuStreamCreate
    cuStreamCreate.restype = CUresult
    cuStreamCreate.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUstream_st)), ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuStreamCreateWithPriority = _libraries['libcuda.so'].cuStreamCreateWithPriority
    cuStreamCreateWithPriority.restype = CUresult
    cuStreamCreateWithPriority.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUstream_st)), ctypes.c_uint32, ctypes.c_int32]
except AttributeError:
    pass
try:
    cuStreamGetPriority = _libraries['libcuda.so'].cuStreamGetPriority
    cuStreamGetPriority.restype = CUresult
    cuStreamGetPriority.argtypes = [CUstream, ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    cuStreamGetFlags = _libraries['libcuda.so'].cuStreamGetFlags
    cuStreamGetFlags.restype = CUresult
    cuStreamGetFlags.argtypes = [CUstream, ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    cuStreamGetCtx = _libraries['libcuda.so'].cuStreamGetCtx
    cuStreamGetCtx.restype = CUresult
    cuStreamGetCtx.argtypes = [CUstream, ctypes.POINTER(ctypes.POINTER(struct_CUctx_st))]
except AttributeError:
    pass
try:
    cuStreamWaitEvent = _libraries['libcuda.so'].cuStreamWaitEvent
    cuStreamWaitEvent.restype = CUresult
    cuStreamWaitEvent.argtypes = [CUstream, CUevent, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuStreamAddCallback = _libraries['libcuda.so'].cuStreamAddCallback
    cuStreamAddCallback.restype = CUresult
    cuStreamAddCallback.argtypes = [CUstream, CUstreamCallback, ctypes.POINTER(None), ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuStreamBeginCapture_v2 = _libraries['libcuda.so'].cuStreamBeginCapture_v2
    cuStreamBeginCapture_v2.restype = CUresult
    cuStreamBeginCapture_v2.argtypes = [CUstream, CUstreamCaptureMode]
except AttributeError:
    pass
try:
    cuThreadExchangeStreamCaptureMode = _libraries['libcuda.so'].cuThreadExchangeStreamCaptureMode
    cuThreadExchangeStreamCaptureMode.restype = CUresult
    cuThreadExchangeStreamCaptureMode.argtypes = [ctypes.POINTER(CUstreamCaptureMode_enum)]
except AttributeError:
    pass
try:
    cuStreamEndCapture = _libraries['libcuda.so'].cuStreamEndCapture
    cuStreamEndCapture.restype = CUresult
    cuStreamEndCapture.argtypes = [CUstream, ctypes.POINTER(ctypes.POINTER(struct_CUgraph_st))]
except AttributeError:
    pass
try:
    cuStreamIsCapturing = _libraries['libcuda.so'].cuStreamIsCapturing
    cuStreamIsCapturing.restype = CUresult
    cuStreamIsCapturing.argtypes = [CUstream, ctypes.POINTER(CUstreamCaptureStatus_enum)]
except AttributeError:
    pass
try:
    cuStreamGetCaptureInfo = _libraries['libcuda.so'].cuStreamGetCaptureInfo
    cuStreamGetCaptureInfo.restype = CUresult
    cuStreamGetCaptureInfo.argtypes = [CUstream, ctypes.POINTER(CUstreamCaptureStatus_enum), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    cuStreamGetCaptureInfo_v2 = _libraries['libcuda.so'].cuStreamGetCaptureInfo_v2
    cuStreamGetCaptureInfo_v2.restype = CUresult
    cuStreamGetCaptureInfo_v2.argtypes = [CUstream, ctypes.POINTER(CUstreamCaptureStatus_enum), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.POINTER(struct_CUgraph_st)), ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st))), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    cuStreamUpdateCaptureDependencies = _libraries['libcuda.so'].cuStreamUpdateCaptureDependencies
    cuStreamUpdateCaptureDependencies.restype = CUresult
    cuStreamUpdateCaptureDependencies.argtypes = [CUstream, ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), size_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuStreamAttachMemAsync = _libraries['libcuda.so'].cuStreamAttachMemAsync
    cuStreamAttachMemAsync.restype = CUresult
    cuStreamAttachMemAsync.argtypes = [CUstream, CUdeviceptr, size_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuStreamQuery = _libraries['libcuda.so'].cuStreamQuery
    cuStreamQuery.restype = CUresult
    cuStreamQuery.argtypes = [CUstream]
except AttributeError:
    pass
try:
    cuStreamSynchronize = _libraries['libcuda.so'].cuStreamSynchronize
    cuStreamSynchronize.restype = CUresult
    cuStreamSynchronize.argtypes = [CUstream]
except AttributeError:
    pass
try:
    cuStreamDestroy_v2 = _libraries['libcuda.so'].cuStreamDestroy_v2
    cuStreamDestroy_v2.restype = CUresult
    cuStreamDestroy_v2.argtypes = [CUstream]
except AttributeError:
    pass
try:
    cuStreamCopyAttributes = _libraries['libcuda.so'].cuStreamCopyAttributes
    cuStreamCopyAttributes.restype = CUresult
    cuStreamCopyAttributes.argtypes = [CUstream, CUstream]
except AttributeError:
    pass
try:
    cuStreamGetAttribute = _libraries['libcuda.so'].cuStreamGetAttribute
    cuStreamGetAttribute.restype = CUresult
    cuStreamGetAttribute.argtypes = [CUstream, CUstreamAttrID, ctypes.POINTER(union_CUstreamAttrValue_union)]
except AttributeError:
    pass
try:
    cuStreamSetAttribute = _libraries['libcuda.so'].cuStreamSetAttribute
    cuStreamSetAttribute.restype = CUresult
    cuStreamSetAttribute.argtypes = [CUstream, CUstreamAttrID, ctypes.POINTER(union_CUstreamAttrValue_union)]
except AttributeError:
    pass
try:
    cuEventCreate = _libraries['libcuda.so'].cuEventCreate
    cuEventCreate.restype = CUresult
    cuEventCreate.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUevent_st)), ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuEventRecord = _libraries['libcuda.so'].cuEventRecord
    cuEventRecord.restype = CUresult
    cuEventRecord.argtypes = [CUevent, CUstream]
except AttributeError:
    pass
try:
    cuEventRecordWithFlags = _libraries['libcuda.so'].cuEventRecordWithFlags
    cuEventRecordWithFlags.restype = CUresult
    cuEventRecordWithFlags.argtypes = [CUevent, CUstream, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuEventQuery = _libraries['libcuda.so'].cuEventQuery
    cuEventQuery.restype = CUresult
    cuEventQuery.argtypes = [CUevent]
except AttributeError:
    pass
try:
    cuEventSynchronize = _libraries['libcuda.so'].cuEventSynchronize
    cuEventSynchronize.restype = CUresult
    cuEventSynchronize.argtypes = [CUevent]
except AttributeError:
    pass
try:
    cuEventDestroy_v2 = _libraries['libcuda.so'].cuEventDestroy_v2
    cuEventDestroy_v2.restype = CUresult
    cuEventDestroy_v2.argtypes = [CUevent]
except AttributeError:
    pass
try:
    cuEventElapsedTime = _libraries['libcuda.so'].cuEventElapsedTime
    cuEventElapsedTime.restype = CUresult
    cuEventElapsedTime.argtypes = [ctypes.POINTER(ctypes.c_float), CUevent, CUevent]
except AttributeError:
    pass
try:
    cuImportExternalMemory = _libraries['libcuda.so'].cuImportExternalMemory
    cuImportExternalMemory.restype = CUresult
    cuImportExternalMemory.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUextMemory_st)), ctypes.POINTER(struct_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st)]
except AttributeError:
    pass
try:
    cuExternalMemoryGetMappedBuffer = _libraries['libcuda.so'].cuExternalMemoryGetMappedBuffer
    cuExternalMemoryGetMappedBuffer.restype = CUresult
    cuExternalMemoryGetMappedBuffer.argtypes = [ctypes.POINTER(ctypes.c_uint64), CUexternalMemory, ctypes.POINTER(struct_CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st)]
except AttributeError:
    pass
try:
    cuExternalMemoryGetMappedMipmappedArray = _libraries['libcuda.so'].cuExternalMemoryGetMappedMipmappedArray
    cuExternalMemoryGetMappedMipmappedArray.restype = CUresult
    cuExternalMemoryGetMappedMipmappedArray.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUmipmappedArray_st)), CUexternalMemory, ctypes.POINTER(struct_CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st)]
except AttributeError:
    pass
try:
    cuDestroyExternalMemory = _libraries['libcuda.so'].cuDestroyExternalMemory
    cuDestroyExternalMemory.restype = CUresult
    cuDestroyExternalMemory.argtypes = [CUexternalMemory]
except AttributeError:
    pass
try:
    cuImportExternalSemaphore = _libraries['libcuda.so'].cuImportExternalSemaphore
    cuImportExternalSemaphore.restype = CUresult
    cuImportExternalSemaphore.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUextSemaphore_st)), ctypes.POINTER(struct_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st)]
except AttributeError:
    pass
try:
    cuSignalExternalSemaphoresAsync = _libraries['libcuda.so'].cuSignalExternalSemaphoresAsync
    cuSignalExternalSemaphoresAsync.restype = CUresult
    cuSignalExternalSemaphoresAsync.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUextSemaphore_st)), ctypes.POINTER(struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st), ctypes.c_uint32, CUstream]
except AttributeError:
    pass
try:
    cuWaitExternalSemaphoresAsync = _libraries['libcuda.so'].cuWaitExternalSemaphoresAsync
    cuWaitExternalSemaphoresAsync.restype = CUresult
    cuWaitExternalSemaphoresAsync.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUextSemaphore_st)), ctypes.POINTER(struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st), ctypes.c_uint32, CUstream]
except AttributeError:
    pass
try:
    cuDestroyExternalSemaphore = _libraries['libcuda.so'].cuDestroyExternalSemaphore
    cuDestroyExternalSemaphore.restype = CUresult
    cuDestroyExternalSemaphore.argtypes = [CUexternalSemaphore]
except AttributeError:
    pass
try:
    cuStreamWaitValue32 = _libraries['libcuda.so'].cuStreamWaitValue32
    cuStreamWaitValue32.restype = CUresult
    cuStreamWaitValue32.argtypes = [CUstream, CUdeviceptr, cuuint32_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuStreamWaitValue64 = _libraries['libcuda.so'].cuStreamWaitValue64
    cuStreamWaitValue64.restype = CUresult
    cuStreamWaitValue64.argtypes = [CUstream, CUdeviceptr, cuuint64_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuStreamWriteValue32 = _libraries['libcuda.so'].cuStreamWriteValue32
    cuStreamWriteValue32.restype = CUresult
    cuStreamWriteValue32.argtypes = [CUstream, CUdeviceptr, cuuint32_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuStreamWriteValue64 = _libraries['libcuda.so'].cuStreamWriteValue64
    cuStreamWriteValue64.restype = CUresult
    cuStreamWriteValue64.argtypes = [CUstream, CUdeviceptr, cuuint64_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuStreamBatchMemOp = _libraries['libcuda.so'].cuStreamBatchMemOp
    cuStreamBatchMemOp.restype = CUresult
    cuStreamBatchMemOp.argtypes = [CUstream, ctypes.c_uint32, ctypes.POINTER(union_CUstreamBatchMemOpParams_union), ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuFuncGetAttribute = _libraries['libcuda.so'].cuFuncGetAttribute
    cuFuncGetAttribute.restype = CUresult
    cuFuncGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int32), CUfunction_attribute, CUfunction]
except AttributeError:
    pass
try:
    cuFuncSetAttribute = _libraries['libcuda.so'].cuFuncSetAttribute
    cuFuncSetAttribute.restype = CUresult
    cuFuncSetAttribute.argtypes = [CUfunction, CUfunction_attribute, ctypes.c_int32]
except AttributeError:
    pass
try:
    cuFuncSetCacheConfig = _libraries['libcuda.so'].cuFuncSetCacheConfig
    cuFuncSetCacheConfig.restype = CUresult
    cuFuncSetCacheConfig.argtypes = [CUfunction, CUfunc_cache]
except AttributeError:
    pass
try:
    cuFuncSetSharedMemConfig = _libraries['libcuda.so'].cuFuncSetSharedMemConfig
    cuFuncSetSharedMemConfig.restype = CUresult
    cuFuncSetSharedMemConfig.argtypes = [CUfunction, CUsharedconfig]
except AttributeError:
    pass
try:
    cuFuncGetModule = _libraries['libcuda.so'].cuFuncGetModule
    cuFuncGetModule.restype = CUresult
    cuFuncGetModule.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUmod_st)), CUfunction]
except AttributeError:
    pass
try:
    cuLaunchKernel = _libraries['libcuda.so'].cuLaunchKernel
    cuLaunchKernel.restype = CUresult
    cuLaunchKernel.argtypes = [CUfunction, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, CUstream, ctypes.POINTER(ctypes.POINTER(None)), ctypes.POINTER(ctypes.POINTER(None))]
except AttributeError:
    pass
try:
    cuLaunchCooperativeKernel = _libraries['libcuda.so'].cuLaunchCooperativeKernel
    cuLaunchCooperativeKernel.restype = CUresult
    cuLaunchCooperativeKernel.argtypes = [CUfunction, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, CUstream, ctypes.POINTER(ctypes.POINTER(None))]
except AttributeError:
    pass
try:
    cuLaunchCooperativeKernelMultiDevice = _libraries['libcuda.so'].cuLaunchCooperativeKernelMultiDevice
    cuLaunchCooperativeKernelMultiDevice.restype = CUresult
    cuLaunchCooperativeKernelMultiDevice.argtypes = [ctypes.POINTER(struct_CUDA_LAUNCH_PARAMS_st), ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuLaunchHostFunc = _libraries['libcuda.so'].cuLaunchHostFunc
    cuLaunchHostFunc.restype = CUresult
    cuLaunchHostFunc.argtypes = [CUstream, CUhostFn, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    cuFuncSetBlockShape = _libraries['libcuda.so'].cuFuncSetBlockShape
    cuFuncSetBlockShape.restype = CUresult
    cuFuncSetBlockShape.argtypes = [CUfunction, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
except AttributeError:
    pass
try:
    cuFuncSetSharedSize = _libraries['libcuda.so'].cuFuncSetSharedSize
    cuFuncSetSharedSize.restype = CUresult
    cuFuncSetSharedSize.argtypes = [CUfunction, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuParamSetSize = _libraries['libcuda.so'].cuParamSetSize
    cuParamSetSize.restype = CUresult
    cuParamSetSize.argtypes = [CUfunction, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuParamSeti = _libraries['libcuda.so'].cuParamSeti
    cuParamSeti.restype = CUresult
    cuParamSeti.argtypes = [CUfunction, ctypes.c_int32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuParamSetf = _libraries['libcuda.so'].cuParamSetf
    cuParamSetf.restype = CUresult
    cuParamSetf.argtypes = [CUfunction, ctypes.c_int32, ctypes.c_float]
except AttributeError:
    pass
try:
    cuParamSetv = _libraries['libcuda.so'].cuParamSetv
    cuParamSetv.restype = CUresult
    cuParamSetv.argtypes = [CUfunction, ctypes.c_int32, ctypes.POINTER(None), ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuLaunch = _libraries['libcuda.so'].cuLaunch
    cuLaunch.restype = CUresult
    cuLaunch.argtypes = [CUfunction]
except AttributeError:
    pass
try:
    cuLaunchGrid = _libraries['libcuda.so'].cuLaunchGrid
    cuLaunchGrid.restype = CUresult
    cuLaunchGrid.argtypes = [CUfunction, ctypes.c_int32, ctypes.c_int32]
except AttributeError:
    pass
try:
    cuLaunchGridAsync = _libraries['libcuda.so'].cuLaunchGridAsync
    cuLaunchGridAsync.restype = CUresult
    cuLaunchGridAsync.argtypes = [CUfunction, ctypes.c_int32, ctypes.c_int32, CUstream]
except AttributeError:
    pass
try:
    cuParamSetTexRef = _libraries['libcuda.so'].cuParamSetTexRef
    cuParamSetTexRef.restype = CUresult
    cuParamSetTexRef.argtypes = [CUfunction, ctypes.c_int32, CUtexref]
except AttributeError:
    pass
try:
    cuGraphCreate = _libraries['libcuda.so'].cuGraphCreate
    cuGraphCreate.restype = CUresult
    cuGraphCreate.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUgraph_st)), ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuGraphAddKernelNode = _libraries['libcuda.so'].cuGraphAddKernelNode
    cuGraphAddKernelNode.restype = CUresult
    cuGraphAddKernelNode.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), CUgraph, ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), size_t, ctypes.POINTER(struct_CUDA_KERNEL_NODE_PARAMS_st)]
except AttributeError:
    pass
try:
    cuGraphKernelNodeGetParams = _libraries['libcuda.so'].cuGraphKernelNodeGetParams
    cuGraphKernelNodeGetParams.restype = CUresult
    cuGraphKernelNodeGetParams.argtypes = [CUgraphNode, ctypes.POINTER(struct_CUDA_KERNEL_NODE_PARAMS_st)]
except AttributeError:
    pass
try:
    cuGraphKernelNodeSetParams = _libraries['libcuda.so'].cuGraphKernelNodeSetParams
    cuGraphKernelNodeSetParams.restype = CUresult
    cuGraphKernelNodeSetParams.argtypes = [CUgraphNode, ctypes.POINTER(struct_CUDA_KERNEL_NODE_PARAMS_st)]
except AttributeError:
    pass
try:
    cuGraphAddMemcpyNode = _libraries['libcuda.so'].cuGraphAddMemcpyNode
    cuGraphAddMemcpyNode.restype = CUresult
    cuGraphAddMemcpyNode.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), CUgraph, ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), size_t, ctypes.POINTER(struct_CUDA_MEMCPY3D_st), CUcontext]
except AttributeError:
    pass
try:
    cuGraphMemcpyNodeGetParams = _libraries['libcuda.so'].cuGraphMemcpyNodeGetParams
    cuGraphMemcpyNodeGetParams.restype = CUresult
    cuGraphMemcpyNodeGetParams.argtypes = [CUgraphNode, ctypes.POINTER(struct_CUDA_MEMCPY3D_st)]
except AttributeError:
    pass
try:
    cuGraphMemcpyNodeSetParams = _libraries['libcuda.so'].cuGraphMemcpyNodeSetParams
    cuGraphMemcpyNodeSetParams.restype = CUresult
    cuGraphMemcpyNodeSetParams.argtypes = [CUgraphNode, ctypes.POINTER(struct_CUDA_MEMCPY3D_st)]
except AttributeError:
    pass
try:
    cuGraphAddMemsetNode = _libraries['libcuda.so'].cuGraphAddMemsetNode
    cuGraphAddMemsetNode.restype = CUresult
    cuGraphAddMemsetNode.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), CUgraph, ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), size_t, ctypes.POINTER(struct_CUDA_MEMSET_NODE_PARAMS_st), CUcontext]
except AttributeError:
    pass
try:
    cuGraphMemsetNodeGetParams = _libraries['libcuda.so'].cuGraphMemsetNodeGetParams
    cuGraphMemsetNodeGetParams.restype = CUresult
    cuGraphMemsetNodeGetParams.argtypes = [CUgraphNode, ctypes.POINTER(struct_CUDA_MEMSET_NODE_PARAMS_st)]
except AttributeError:
    pass
try:
    cuGraphMemsetNodeSetParams = _libraries['libcuda.so'].cuGraphMemsetNodeSetParams
    cuGraphMemsetNodeSetParams.restype = CUresult
    cuGraphMemsetNodeSetParams.argtypes = [CUgraphNode, ctypes.POINTER(struct_CUDA_MEMSET_NODE_PARAMS_st)]
except AttributeError:
    pass
try:
    cuGraphAddHostNode = _libraries['libcuda.so'].cuGraphAddHostNode
    cuGraphAddHostNode.restype = CUresult
    cuGraphAddHostNode.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), CUgraph, ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), size_t, ctypes.POINTER(struct_CUDA_HOST_NODE_PARAMS_st)]
except AttributeError:
    pass
try:
    cuGraphHostNodeGetParams = _libraries['libcuda.so'].cuGraphHostNodeGetParams
    cuGraphHostNodeGetParams.restype = CUresult
    cuGraphHostNodeGetParams.argtypes = [CUgraphNode, ctypes.POINTER(struct_CUDA_HOST_NODE_PARAMS_st)]
except AttributeError:
    pass
try:
    cuGraphHostNodeSetParams = _libraries['libcuda.so'].cuGraphHostNodeSetParams
    cuGraphHostNodeSetParams.restype = CUresult
    cuGraphHostNodeSetParams.argtypes = [CUgraphNode, ctypes.POINTER(struct_CUDA_HOST_NODE_PARAMS_st)]
except AttributeError:
    pass
try:
    cuGraphAddChildGraphNode = _libraries['libcuda.so'].cuGraphAddChildGraphNode
    cuGraphAddChildGraphNode.restype = CUresult
    cuGraphAddChildGraphNode.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), CUgraph, ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), size_t, CUgraph]
except AttributeError:
    pass
try:
    cuGraphChildGraphNodeGetGraph = _libraries['libcuda.so'].cuGraphChildGraphNodeGetGraph
    cuGraphChildGraphNodeGetGraph.restype = CUresult
    cuGraphChildGraphNodeGetGraph.argtypes = [CUgraphNode, ctypes.POINTER(ctypes.POINTER(struct_CUgraph_st))]
except AttributeError:
    pass
try:
    cuGraphAddEmptyNode = _libraries['libcuda.so'].cuGraphAddEmptyNode
    cuGraphAddEmptyNode.restype = CUresult
    cuGraphAddEmptyNode.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), CUgraph, ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), size_t]
except AttributeError:
    pass
try:
    cuGraphAddEventRecordNode = _libraries['libcuda.so'].cuGraphAddEventRecordNode
    cuGraphAddEventRecordNode.restype = CUresult
    cuGraphAddEventRecordNode.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), CUgraph, ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), size_t, CUevent]
except AttributeError:
    pass
try:
    cuGraphEventRecordNodeGetEvent = _libraries['libcuda.so'].cuGraphEventRecordNodeGetEvent
    cuGraphEventRecordNodeGetEvent.restype = CUresult
    cuGraphEventRecordNodeGetEvent.argtypes = [CUgraphNode, ctypes.POINTER(ctypes.POINTER(struct_CUevent_st))]
except AttributeError:
    pass
try:
    cuGraphEventRecordNodeSetEvent = _libraries['libcuda.so'].cuGraphEventRecordNodeSetEvent
    cuGraphEventRecordNodeSetEvent.restype = CUresult
    cuGraphEventRecordNodeSetEvent.argtypes = [CUgraphNode, CUevent]
except AttributeError:
    pass
try:
    cuGraphAddEventWaitNode = _libraries['libcuda.so'].cuGraphAddEventWaitNode
    cuGraphAddEventWaitNode.restype = CUresult
    cuGraphAddEventWaitNode.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), CUgraph, ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), size_t, CUevent]
except AttributeError:
    pass
try:
    cuGraphEventWaitNodeGetEvent = _libraries['libcuda.so'].cuGraphEventWaitNodeGetEvent
    cuGraphEventWaitNodeGetEvent.restype = CUresult
    cuGraphEventWaitNodeGetEvent.argtypes = [CUgraphNode, ctypes.POINTER(ctypes.POINTER(struct_CUevent_st))]
except AttributeError:
    pass
try:
    cuGraphEventWaitNodeSetEvent = _libraries['libcuda.so'].cuGraphEventWaitNodeSetEvent
    cuGraphEventWaitNodeSetEvent.restype = CUresult
    cuGraphEventWaitNodeSetEvent.argtypes = [CUgraphNode, CUevent]
except AttributeError:
    pass
try:
    cuGraphAddExternalSemaphoresSignalNode = _libraries['libcuda.so'].cuGraphAddExternalSemaphoresSignalNode
    cuGraphAddExternalSemaphoresSignalNode.restype = CUresult
    cuGraphAddExternalSemaphoresSignalNode.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), CUgraph, ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), size_t, ctypes.POINTER(struct_CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_st)]
except AttributeError:
    pass
try:
    cuGraphExternalSemaphoresSignalNodeGetParams = _libraries['libcuda.so'].cuGraphExternalSemaphoresSignalNodeGetParams
    cuGraphExternalSemaphoresSignalNodeGetParams.restype = CUresult
    cuGraphExternalSemaphoresSignalNodeGetParams.argtypes = [CUgraphNode, ctypes.POINTER(struct_CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_st)]
except AttributeError:
    pass
try:
    cuGraphExternalSemaphoresSignalNodeSetParams = _libraries['libcuda.so'].cuGraphExternalSemaphoresSignalNodeSetParams
    cuGraphExternalSemaphoresSignalNodeSetParams.restype = CUresult
    cuGraphExternalSemaphoresSignalNodeSetParams.argtypes = [CUgraphNode, ctypes.POINTER(struct_CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_st)]
except AttributeError:
    pass
try:
    cuGraphAddExternalSemaphoresWaitNode = _libraries['libcuda.so'].cuGraphAddExternalSemaphoresWaitNode
    cuGraphAddExternalSemaphoresWaitNode.restype = CUresult
    cuGraphAddExternalSemaphoresWaitNode.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), CUgraph, ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), size_t, ctypes.POINTER(struct_CUDA_EXT_SEM_WAIT_NODE_PARAMS_st)]
except AttributeError:
    pass
try:
    cuGraphExternalSemaphoresWaitNodeGetParams = _libraries['libcuda.so'].cuGraphExternalSemaphoresWaitNodeGetParams
    cuGraphExternalSemaphoresWaitNodeGetParams.restype = CUresult
    cuGraphExternalSemaphoresWaitNodeGetParams.argtypes = [CUgraphNode, ctypes.POINTER(struct_CUDA_EXT_SEM_WAIT_NODE_PARAMS_st)]
except AttributeError:
    pass
try:
    cuGraphExternalSemaphoresWaitNodeSetParams = _libraries['libcuda.so'].cuGraphExternalSemaphoresWaitNodeSetParams
    cuGraphExternalSemaphoresWaitNodeSetParams.restype = CUresult
    cuGraphExternalSemaphoresWaitNodeSetParams.argtypes = [CUgraphNode, ctypes.POINTER(struct_CUDA_EXT_SEM_WAIT_NODE_PARAMS_st)]
except AttributeError:
    pass
try:
    cuGraphAddMemAllocNode = _libraries['libcuda.so'].cuGraphAddMemAllocNode
    cuGraphAddMemAllocNode.restype = CUresult
    cuGraphAddMemAllocNode.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), CUgraph, ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), size_t, ctypes.POINTER(struct_CUDA_MEM_ALLOC_NODE_PARAMS_st)]
except AttributeError:
    pass
try:
    cuGraphMemAllocNodeGetParams = _libraries['libcuda.so'].cuGraphMemAllocNodeGetParams
    cuGraphMemAllocNodeGetParams.restype = CUresult
    cuGraphMemAllocNodeGetParams.argtypes = [CUgraphNode, ctypes.POINTER(struct_CUDA_MEM_ALLOC_NODE_PARAMS_st)]
except AttributeError:
    pass
try:
    cuGraphAddMemFreeNode = _libraries['libcuda.so'].cuGraphAddMemFreeNode
    cuGraphAddMemFreeNode.restype = CUresult
    cuGraphAddMemFreeNode.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), CUgraph, ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), size_t, CUdeviceptr]
except AttributeError:
    pass
try:
    cuGraphMemFreeNodeGetParams = _libraries['libcuda.so'].cuGraphMemFreeNodeGetParams
    cuGraphMemFreeNodeGetParams.restype = CUresult
    cuGraphMemFreeNodeGetParams.argtypes = [CUgraphNode, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    cuDeviceGraphMemTrim = _libraries['libcuda.so'].cuDeviceGraphMemTrim
    cuDeviceGraphMemTrim.restype = CUresult
    cuDeviceGraphMemTrim.argtypes = [CUdevice]
except AttributeError:
    pass
try:
    cuDeviceGetGraphMemAttribute = _libraries['libcuda.so'].cuDeviceGetGraphMemAttribute
    cuDeviceGetGraphMemAttribute.restype = CUresult
    cuDeviceGetGraphMemAttribute.argtypes = [CUdevice, CUgraphMem_attribute, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    cuDeviceSetGraphMemAttribute = _libraries['libcuda.so'].cuDeviceSetGraphMemAttribute
    cuDeviceSetGraphMemAttribute.restype = CUresult
    cuDeviceSetGraphMemAttribute.argtypes = [CUdevice, CUgraphMem_attribute, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    cuGraphClone = _libraries['libcuda.so'].cuGraphClone
    cuGraphClone.restype = CUresult
    cuGraphClone.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUgraph_st)), CUgraph]
except AttributeError:
    pass
try:
    cuGraphNodeFindInClone = _libraries['libcuda.so'].cuGraphNodeFindInClone
    cuGraphNodeFindInClone.restype = CUresult
    cuGraphNodeFindInClone.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), CUgraphNode, CUgraph]
except AttributeError:
    pass
try:
    cuGraphNodeGetType = _libraries['libcuda.so'].cuGraphNodeGetType
    cuGraphNodeGetType.restype = CUresult
    cuGraphNodeGetType.argtypes = [CUgraphNode, ctypes.POINTER(CUgraphNodeType_enum)]
except AttributeError:
    pass
try:
    cuGraphGetNodes = _libraries['libcuda.so'].cuGraphGetNodes
    cuGraphGetNodes.restype = CUresult
    cuGraphGetNodes.argtypes = [CUgraph, ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    cuGraphGetRootNodes = _libraries['libcuda.so'].cuGraphGetRootNodes
    cuGraphGetRootNodes.restype = CUresult
    cuGraphGetRootNodes.argtypes = [CUgraph, ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    cuGraphGetEdges = _libraries['libcuda.so'].cuGraphGetEdges
    cuGraphGetEdges.restype = CUresult
    cuGraphGetEdges.argtypes = [CUgraph, ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    cuGraphNodeGetDependencies = _libraries['libcuda.so'].cuGraphNodeGetDependencies
    cuGraphNodeGetDependencies.restype = CUresult
    cuGraphNodeGetDependencies.argtypes = [CUgraphNode, ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    cuGraphNodeGetDependentNodes = _libraries['libcuda.so'].cuGraphNodeGetDependentNodes
    cuGraphNodeGetDependentNodes.restype = CUresult
    cuGraphNodeGetDependentNodes.argtypes = [CUgraphNode, ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    cuGraphAddDependencies = _libraries['libcuda.so'].cuGraphAddDependencies
    cuGraphAddDependencies.restype = CUresult
    cuGraphAddDependencies.argtypes = [CUgraph, ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), size_t]
except AttributeError:
    pass
try:
    cuGraphRemoveDependencies = _libraries['libcuda.so'].cuGraphRemoveDependencies
    cuGraphRemoveDependencies.restype = CUresult
    cuGraphRemoveDependencies.argtypes = [CUgraph, ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), size_t]
except AttributeError:
    pass
try:
    cuGraphDestroyNode = _libraries['libcuda.so'].cuGraphDestroyNode
    cuGraphDestroyNode.restype = CUresult
    cuGraphDestroyNode.argtypes = [CUgraphNode]
except AttributeError:
    pass
try:
    cuGraphInstantiate_v2 = _libraries['libcuda.so'].cuGraphInstantiate_v2
    cuGraphInstantiate_v2.restype = CUresult
    cuGraphInstantiate_v2.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUgraphExec_st)), CUgraph, ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
try:
    cuGraphInstantiateWithFlags = _libraries['libcuda.so'].cuGraphInstantiateWithFlags
    cuGraphInstantiateWithFlags.restype = CUresult
    cuGraphInstantiateWithFlags.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUgraphExec_st)), CUgraph, ctypes.c_uint64]
except AttributeError:
    pass
try:
    cuGraphExecKernelNodeSetParams = _libraries['libcuda.so'].cuGraphExecKernelNodeSetParams
    cuGraphExecKernelNodeSetParams.restype = CUresult
    cuGraphExecKernelNodeSetParams.argtypes = [CUgraphExec, CUgraphNode, ctypes.POINTER(struct_CUDA_KERNEL_NODE_PARAMS_st)]
except AttributeError:
    pass
try:
    cuGraphExecMemcpyNodeSetParams = _libraries['libcuda.so'].cuGraphExecMemcpyNodeSetParams
    cuGraphExecMemcpyNodeSetParams.restype = CUresult
    cuGraphExecMemcpyNodeSetParams.argtypes = [CUgraphExec, CUgraphNode, ctypes.POINTER(struct_CUDA_MEMCPY3D_st), CUcontext]
except AttributeError:
    pass
try:
    cuGraphExecMemsetNodeSetParams = _libraries['libcuda.so'].cuGraphExecMemsetNodeSetParams
    cuGraphExecMemsetNodeSetParams.restype = CUresult
    cuGraphExecMemsetNodeSetParams.argtypes = [CUgraphExec, CUgraphNode, ctypes.POINTER(struct_CUDA_MEMSET_NODE_PARAMS_st), CUcontext]
except AttributeError:
    pass
try:
    cuGraphExecHostNodeSetParams = _libraries['libcuda.so'].cuGraphExecHostNodeSetParams
    cuGraphExecHostNodeSetParams.restype = CUresult
    cuGraphExecHostNodeSetParams.argtypes = [CUgraphExec, CUgraphNode, ctypes.POINTER(struct_CUDA_HOST_NODE_PARAMS_st)]
except AttributeError:
    pass
try:
    cuGraphExecChildGraphNodeSetParams = _libraries['libcuda.so'].cuGraphExecChildGraphNodeSetParams
    cuGraphExecChildGraphNodeSetParams.restype = CUresult
    cuGraphExecChildGraphNodeSetParams.argtypes = [CUgraphExec, CUgraphNode, CUgraph]
except AttributeError:
    pass
try:
    cuGraphExecEventRecordNodeSetEvent = _libraries['libcuda.so'].cuGraphExecEventRecordNodeSetEvent
    cuGraphExecEventRecordNodeSetEvent.restype = CUresult
    cuGraphExecEventRecordNodeSetEvent.argtypes = [CUgraphExec, CUgraphNode, CUevent]
except AttributeError:
    pass
try:
    cuGraphExecEventWaitNodeSetEvent = _libraries['libcuda.so'].cuGraphExecEventWaitNodeSetEvent
    cuGraphExecEventWaitNodeSetEvent.restype = CUresult
    cuGraphExecEventWaitNodeSetEvent.argtypes = [CUgraphExec, CUgraphNode, CUevent]
except AttributeError:
    pass
try:
    cuGraphExecExternalSemaphoresSignalNodeSetParams = _libraries['libcuda.so'].cuGraphExecExternalSemaphoresSignalNodeSetParams
    cuGraphExecExternalSemaphoresSignalNodeSetParams.restype = CUresult
    cuGraphExecExternalSemaphoresSignalNodeSetParams.argtypes = [CUgraphExec, CUgraphNode, ctypes.POINTER(struct_CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_st)]
except AttributeError:
    pass
try:
    cuGraphExecExternalSemaphoresWaitNodeSetParams = _libraries['libcuda.so'].cuGraphExecExternalSemaphoresWaitNodeSetParams
    cuGraphExecExternalSemaphoresWaitNodeSetParams.restype = CUresult
    cuGraphExecExternalSemaphoresWaitNodeSetParams.argtypes = [CUgraphExec, CUgraphNode, ctypes.POINTER(struct_CUDA_EXT_SEM_WAIT_NODE_PARAMS_st)]
except AttributeError:
    pass
try:
    cuGraphUpload = _libraries['libcuda.so'].cuGraphUpload
    cuGraphUpload.restype = CUresult
    cuGraphUpload.argtypes = [CUgraphExec, CUstream]
except AttributeError:
    pass
try:
    cuGraphLaunch = _libraries['libcuda.so'].cuGraphLaunch
    cuGraphLaunch.restype = CUresult
    cuGraphLaunch.argtypes = [CUgraphExec, CUstream]
except AttributeError:
    pass
try:
    cuGraphExecDestroy = _libraries['libcuda.so'].cuGraphExecDestroy
    cuGraphExecDestroy.restype = CUresult
    cuGraphExecDestroy.argtypes = [CUgraphExec]
except AttributeError:
    pass
try:
    cuGraphDestroy = _libraries['libcuda.so'].cuGraphDestroy
    cuGraphDestroy.restype = CUresult
    cuGraphDestroy.argtypes = [CUgraph]
except AttributeError:
    pass
try:
    cuGraphExecUpdate = _libraries['libcuda.so'].cuGraphExecUpdate
    cuGraphExecUpdate.restype = CUresult
    cuGraphExecUpdate.argtypes = [CUgraphExec, CUgraph, ctypes.POINTER(ctypes.POINTER(struct_CUgraphNode_st)), ctypes.POINTER(CUgraphExecUpdateResult_enum)]
except AttributeError:
    pass
try:
    cuGraphKernelNodeCopyAttributes = _libraries['libcuda.so'].cuGraphKernelNodeCopyAttributes
    cuGraphKernelNodeCopyAttributes.restype = CUresult
    cuGraphKernelNodeCopyAttributes.argtypes = [CUgraphNode, CUgraphNode]
except AttributeError:
    pass
try:
    cuGraphKernelNodeGetAttribute = _libraries['libcuda.so'].cuGraphKernelNodeGetAttribute
    cuGraphKernelNodeGetAttribute.restype = CUresult
    cuGraphKernelNodeGetAttribute.argtypes = [CUgraphNode, CUkernelNodeAttrID, ctypes.POINTER(union_CUkernelNodeAttrValue_union)]
except AttributeError:
    pass
try:
    cuGraphKernelNodeSetAttribute = _libraries['libcuda.so'].cuGraphKernelNodeSetAttribute
    cuGraphKernelNodeSetAttribute.restype = CUresult
    cuGraphKernelNodeSetAttribute.argtypes = [CUgraphNode, CUkernelNodeAttrID, ctypes.POINTER(union_CUkernelNodeAttrValue_union)]
except AttributeError:
    pass
try:
    cuGraphDebugDotPrint = _libraries['libcuda.so'].cuGraphDebugDotPrint
    cuGraphDebugDotPrint.restype = CUresult
    cuGraphDebugDotPrint.argtypes = [CUgraph, ctypes.POINTER(ctypes.c_char), ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuUserObjectCreate = _libraries['libcuda.so'].cuUserObjectCreate
    cuUserObjectCreate.restype = CUresult
    cuUserObjectCreate.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUuserObject_st)), ctypes.POINTER(None), CUhostFn, ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuUserObjectRetain = _libraries['libcuda.so'].cuUserObjectRetain
    cuUserObjectRetain.restype = CUresult
    cuUserObjectRetain.argtypes = [CUuserObject, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuUserObjectRelease = _libraries['libcuda.so'].cuUserObjectRelease
    cuUserObjectRelease.restype = CUresult
    cuUserObjectRelease.argtypes = [CUuserObject, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuGraphRetainUserObject = _libraries['libcuda.so'].cuGraphRetainUserObject
    cuGraphRetainUserObject.restype = CUresult
    cuGraphRetainUserObject.argtypes = [CUgraph, CUuserObject, ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuGraphReleaseUserObject = _libraries['libcuda.so'].cuGraphReleaseUserObject
    cuGraphReleaseUserObject.restype = CUresult
    cuGraphReleaseUserObject.argtypes = [CUgraph, CUuserObject, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuOccupancyMaxActiveBlocksPerMultiprocessor = _libraries['libcuda.so'].cuOccupancyMaxActiveBlocksPerMultiprocessor
    cuOccupancyMaxActiveBlocksPerMultiprocessor.restype = CUresult
    cuOccupancyMaxActiveBlocksPerMultiprocessor.argtypes = [ctypes.POINTER(ctypes.c_int32), CUfunction, ctypes.c_int32, size_t]
except AttributeError:
    pass
try:
    cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags = _libraries['libcuda.so'].cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
    cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.restype = CUresult
    cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.argtypes = [ctypes.POINTER(ctypes.c_int32), CUfunction, ctypes.c_int32, size_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuOccupancyMaxPotentialBlockSize = _libraries['libcuda.so'].cuOccupancyMaxPotentialBlockSize
    cuOccupancyMaxPotentialBlockSize.restype = CUresult
    cuOccupancyMaxPotentialBlockSize.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), CUfunction, CUoccupancyB2DSize, size_t, ctypes.c_int32]
except AttributeError:
    pass
try:
    cuOccupancyMaxPotentialBlockSizeWithFlags = _libraries['libcuda.so'].cuOccupancyMaxPotentialBlockSizeWithFlags
    cuOccupancyMaxPotentialBlockSizeWithFlags.restype = CUresult
    cuOccupancyMaxPotentialBlockSizeWithFlags.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), CUfunction, CUoccupancyB2DSize, size_t, ctypes.c_int32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuOccupancyAvailableDynamicSMemPerBlock = _libraries['libcuda.so'].cuOccupancyAvailableDynamicSMemPerBlock
    cuOccupancyAvailableDynamicSMemPerBlock.restype = CUresult
    cuOccupancyAvailableDynamicSMemPerBlock.argtypes = [ctypes.POINTER(ctypes.c_uint64), CUfunction, ctypes.c_int32, ctypes.c_int32]
except AttributeError:
    pass
try:
    cuTexRefSetArray = _libraries['libcuda.so'].cuTexRefSetArray
    cuTexRefSetArray.restype = CUresult
    cuTexRefSetArray.argtypes = [CUtexref, CUarray, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuTexRefSetMipmappedArray = _libraries['libcuda.so'].cuTexRefSetMipmappedArray
    cuTexRefSetMipmappedArray.restype = CUresult
    cuTexRefSetMipmappedArray.argtypes = [CUtexref, CUmipmappedArray, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuTexRefSetAddress_v2 = _libraries['libcuda.so'].cuTexRefSetAddress_v2
    cuTexRefSetAddress_v2.restype = CUresult
    cuTexRefSetAddress_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), CUtexref, CUdeviceptr, size_t]
except AttributeError:
    pass
try:
    cuTexRefSetAddress2D_v3 = _libraries['libcuda.so'].cuTexRefSetAddress2D_v3
    cuTexRefSetAddress2D_v3.restype = CUresult
    cuTexRefSetAddress2D_v3.argtypes = [CUtexref, ctypes.POINTER(struct_CUDA_ARRAY_DESCRIPTOR_st), CUdeviceptr, size_t]
except AttributeError:
    pass
try:
    cuTexRefSetFormat = _libraries['libcuda.so'].cuTexRefSetFormat
    cuTexRefSetFormat.restype = CUresult
    cuTexRefSetFormat.argtypes = [CUtexref, CUarray_format, ctypes.c_int32]
except AttributeError:
    pass
try:
    cuTexRefSetAddressMode = _libraries['libcuda.so'].cuTexRefSetAddressMode
    cuTexRefSetAddressMode.restype = CUresult
    cuTexRefSetAddressMode.argtypes = [CUtexref, ctypes.c_int32, CUaddress_mode]
except AttributeError:
    pass
try:
    cuTexRefSetFilterMode = _libraries['libcuda.so'].cuTexRefSetFilterMode
    cuTexRefSetFilterMode.restype = CUresult
    cuTexRefSetFilterMode.argtypes = [CUtexref, CUfilter_mode]
except AttributeError:
    pass
try:
    cuTexRefSetMipmapFilterMode = _libraries['libcuda.so'].cuTexRefSetMipmapFilterMode
    cuTexRefSetMipmapFilterMode.restype = CUresult
    cuTexRefSetMipmapFilterMode.argtypes = [CUtexref, CUfilter_mode]
except AttributeError:
    pass
try:
    cuTexRefSetMipmapLevelBias = _libraries['libcuda.so'].cuTexRefSetMipmapLevelBias
    cuTexRefSetMipmapLevelBias.restype = CUresult
    cuTexRefSetMipmapLevelBias.argtypes = [CUtexref, ctypes.c_float]
except AttributeError:
    pass
try:
    cuTexRefSetMipmapLevelClamp = _libraries['libcuda.so'].cuTexRefSetMipmapLevelClamp
    cuTexRefSetMipmapLevelClamp.restype = CUresult
    cuTexRefSetMipmapLevelClamp.argtypes = [CUtexref, ctypes.c_float, ctypes.c_float]
except AttributeError:
    pass
try:
    cuTexRefSetMaxAnisotropy = _libraries['libcuda.so'].cuTexRefSetMaxAnisotropy
    cuTexRefSetMaxAnisotropy.restype = CUresult
    cuTexRefSetMaxAnisotropy.argtypes = [CUtexref, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuTexRefSetBorderColor = _libraries['libcuda.so'].cuTexRefSetBorderColor
    cuTexRefSetBorderColor.restype = CUresult
    cuTexRefSetBorderColor.argtypes = [CUtexref, ctypes.POINTER(ctypes.c_float)]
except AttributeError:
    pass
try:
    cuTexRefSetFlags = _libraries['libcuda.so'].cuTexRefSetFlags
    cuTexRefSetFlags.restype = CUresult
    cuTexRefSetFlags.argtypes = [CUtexref, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuTexRefGetAddress_v2 = _libraries['libcuda.so'].cuTexRefGetAddress_v2
    cuTexRefGetAddress_v2.restype = CUresult
    cuTexRefGetAddress_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), CUtexref]
except AttributeError:
    pass
try:
    cuTexRefGetArray = _libraries['libcuda.so'].cuTexRefGetArray
    cuTexRefGetArray.restype = CUresult
    cuTexRefGetArray.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUarray_st)), CUtexref]
except AttributeError:
    pass
try:
    cuTexRefGetMipmappedArray = _libraries['libcuda.so'].cuTexRefGetMipmappedArray
    cuTexRefGetMipmappedArray.restype = CUresult
    cuTexRefGetMipmappedArray.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUmipmappedArray_st)), CUtexref]
except AttributeError:
    pass
try:
    cuTexRefGetAddressMode = _libraries['libcuda.so'].cuTexRefGetAddressMode
    cuTexRefGetAddressMode.restype = CUresult
    cuTexRefGetAddressMode.argtypes = [ctypes.POINTER(CUaddress_mode_enum), CUtexref, ctypes.c_int32]
except AttributeError:
    pass
try:
    cuTexRefGetFilterMode = _libraries['libcuda.so'].cuTexRefGetFilterMode
    cuTexRefGetFilterMode.restype = CUresult
    cuTexRefGetFilterMode.argtypes = [ctypes.POINTER(CUfilter_mode_enum), CUtexref]
except AttributeError:
    pass
try:
    cuTexRefGetFormat = _libraries['libcuda.so'].cuTexRefGetFormat
    cuTexRefGetFormat.restype = CUresult
    cuTexRefGetFormat.argtypes = [ctypes.POINTER(CUarray_format_enum), ctypes.POINTER(ctypes.c_int32), CUtexref]
except AttributeError:
    pass
try:
    cuTexRefGetMipmapFilterMode = _libraries['libcuda.so'].cuTexRefGetMipmapFilterMode
    cuTexRefGetMipmapFilterMode.restype = CUresult
    cuTexRefGetMipmapFilterMode.argtypes = [ctypes.POINTER(CUfilter_mode_enum), CUtexref]
except AttributeError:
    pass
try:
    cuTexRefGetMipmapLevelBias = _libraries['libcuda.so'].cuTexRefGetMipmapLevelBias
    cuTexRefGetMipmapLevelBias.restype = CUresult
    cuTexRefGetMipmapLevelBias.argtypes = [ctypes.POINTER(ctypes.c_float), CUtexref]
except AttributeError:
    pass
try:
    cuTexRefGetMipmapLevelClamp = _libraries['libcuda.so'].cuTexRefGetMipmapLevelClamp
    cuTexRefGetMipmapLevelClamp.restype = CUresult
    cuTexRefGetMipmapLevelClamp.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), CUtexref]
except AttributeError:
    pass
try:
    cuTexRefGetMaxAnisotropy = _libraries['libcuda.so'].cuTexRefGetMaxAnisotropy
    cuTexRefGetMaxAnisotropy.restype = CUresult
    cuTexRefGetMaxAnisotropy.argtypes = [ctypes.POINTER(ctypes.c_int32), CUtexref]
except AttributeError:
    pass
try:
    cuTexRefGetBorderColor = _libraries['libcuda.so'].cuTexRefGetBorderColor
    cuTexRefGetBorderColor.restype = CUresult
    cuTexRefGetBorderColor.argtypes = [ctypes.POINTER(ctypes.c_float), CUtexref]
except AttributeError:
    pass
try:
    cuTexRefGetFlags = _libraries['libcuda.so'].cuTexRefGetFlags
    cuTexRefGetFlags.restype = CUresult
    cuTexRefGetFlags.argtypes = [ctypes.POINTER(ctypes.c_uint32), CUtexref]
except AttributeError:
    pass
try:
    cuTexRefCreate = _libraries['libcuda.so'].cuTexRefCreate
    cuTexRefCreate.restype = CUresult
    cuTexRefCreate.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUtexref_st))]
except AttributeError:
    pass
try:
    cuTexRefDestroy = _libraries['libcuda.so'].cuTexRefDestroy
    cuTexRefDestroy.restype = CUresult
    cuTexRefDestroy.argtypes = [CUtexref]
except AttributeError:
    pass
try:
    cuSurfRefSetArray = _libraries['libcuda.so'].cuSurfRefSetArray
    cuSurfRefSetArray.restype = CUresult
    cuSurfRefSetArray.argtypes = [CUsurfref, CUarray, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuSurfRefGetArray = _libraries['libcuda.so'].cuSurfRefGetArray
    cuSurfRefGetArray.restype = CUresult
    cuSurfRefGetArray.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUarray_st)), CUsurfref]
except AttributeError:
    pass
try:
    cuTexObjectCreate = _libraries['libcuda.so'].cuTexObjectCreate
    cuTexObjectCreate.restype = CUresult
    cuTexObjectCreate.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(struct_CUDA_RESOURCE_DESC_st), ctypes.POINTER(struct_CUDA_TEXTURE_DESC_st), ctypes.POINTER(struct_CUDA_RESOURCE_VIEW_DESC_st)]
except AttributeError:
    pass
try:
    cuTexObjectDestroy = _libraries['libcuda.so'].cuTexObjectDestroy
    cuTexObjectDestroy.restype = CUresult
    cuTexObjectDestroy.argtypes = [CUtexObject]
except AttributeError:
    pass
try:
    cuTexObjectGetResourceDesc = _libraries['libcuda.so'].cuTexObjectGetResourceDesc
    cuTexObjectGetResourceDesc.restype = CUresult
    cuTexObjectGetResourceDesc.argtypes = [ctypes.POINTER(struct_CUDA_RESOURCE_DESC_st), CUtexObject]
except AttributeError:
    pass
try:
    cuTexObjectGetTextureDesc = _libraries['libcuda.so'].cuTexObjectGetTextureDesc
    cuTexObjectGetTextureDesc.restype = CUresult
    cuTexObjectGetTextureDesc.argtypes = [ctypes.POINTER(struct_CUDA_TEXTURE_DESC_st), CUtexObject]
except AttributeError:
    pass
try:
    cuTexObjectGetResourceViewDesc = _libraries['libcuda.so'].cuTexObjectGetResourceViewDesc
    cuTexObjectGetResourceViewDesc.restype = CUresult
    cuTexObjectGetResourceViewDesc.argtypes = [ctypes.POINTER(struct_CUDA_RESOURCE_VIEW_DESC_st), CUtexObject]
except AttributeError:
    pass
try:
    cuSurfObjectCreate = _libraries['libcuda.so'].cuSurfObjectCreate
    cuSurfObjectCreate.restype = CUresult
    cuSurfObjectCreate.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(struct_CUDA_RESOURCE_DESC_st)]
except AttributeError:
    pass
try:
    cuSurfObjectDestroy = _libraries['libcuda.so'].cuSurfObjectDestroy
    cuSurfObjectDestroy.restype = CUresult
    cuSurfObjectDestroy.argtypes = [CUsurfObject]
except AttributeError:
    pass
try:
    cuSurfObjectGetResourceDesc = _libraries['libcuda.so'].cuSurfObjectGetResourceDesc
    cuSurfObjectGetResourceDesc.restype = CUresult
    cuSurfObjectGetResourceDesc.argtypes = [ctypes.POINTER(struct_CUDA_RESOURCE_DESC_st), CUsurfObject]
except AttributeError:
    pass
try:
    cuDeviceCanAccessPeer = _libraries['libcuda.so'].cuDeviceCanAccessPeer
    cuDeviceCanAccessPeer.restype = CUresult
    cuDeviceCanAccessPeer.argtypes = [ctypes.POINTER(ctypes.c_int32), CUdevice, CUdevice]
except AttributeError:
    pass
try:
    cuCtxEnablePeerAccess = _libraries['libcuda.so'].cuCtxEnablePeerAccess
    cuCtxEnablePeerAccess.restype = CUresult
    cuCtxEnablePeerAccess.argtypes = [CUcontext, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuCtxDisablePeerAccess = _libraries['libcuda.so'].cuCtxDisablePeerAccess
    cuCtxDisablePeerAccess.restype = CUresult
    cuCtxDisablePeerAccess.argtypes = [CUcontext]
except AttributeError:
    pass
try:
    cuDeviceGetP2PAttribute = _libraries['libcuda.so'].cuDeviceGetP2PAttribute
    cuDeviceGetP2PAttribute.restype = CUresult
    cuDeviceGetP2PAttribute.argtypes = [ctypes.POINTER(ctypes.c_int32), CUdevice_P2PAttribute, CUdevice, CUdevice]
except AttributeError:
    pass
try:
    cuGraphicsUnregisterResource = _libraries['libcuda.so'].cuGraphicsUnregisterResource
    cuGraphicsUnregisterResource.restype = CUresult
    cuGraphicsUnregisterResource.argtypes = [CUgraphicsResource]
except AttributeError:
    pass
try:
    cuGraphicsSubResourceGetMappedArray = _libraries['libcuda.so'].cuGraphicsSubResourceGetMappedArray
    cuGraphicsSubResourceGetMappedArray.restype = CUresult
    cuGraphicsSubResourceGetMappedArray.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUarray_st)), CUgraphicsResource, ctypes.c_uint32, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuGraphicsResourceGetMappedMipmappedArray = _libraries['libcuda.so'].cuGraphicsResourceGetMappedMipmappedArray
    cuGraphicsResourceGetMappedMipmappedArray.restype = CUresult
    cuGraphicsResourceGetMappedMipmappedArray.argtypes = [ctypes.POINTER(ctypes.POINTER(struct_CUmipmappedArray_st)), CUgraphicsResource]
except AttributeError:
    pass
try:
    cuGraphicsResourceGetMappedPointer_v2 = _libraries['libcuda.so'].cuGraphicsResourceGetMappedPointer_v2
    cuGraphicsResourceGetMappedPointer_v2.restype = CUresult
    cuGraphicsResourceGetMappedPointer_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), CUgraphicsResource]
except AttributeError:
    pass
try:
    cuGraphicsResourceSetMapFlags_v2 = _libraries['libcuda.so'].cuGraphicsResourceSetMapFlags_v2
    cuGraphicsResourceSetMapFlags_v2.restype = CUresult
    cuGraphicsResourceSetMapFlags_v2.argtypes = [CUgraphicsResource, ctypes.c_uint32]
except AttributeError:
    pass
try:
    cuGraphicsMapResources = _libraries['libcuda.so'].cuGraphicsMapResources
    cuGraphicsMapResources.restype = CUresult
    cuGraphicsMapResources.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.POINTER(struct_CUgraphicsResource_st)), CUstream]
except AttributeError:
    pass
try:
    cuGraphicsUnmapResources = _libraries['libcuda.so'].cuGraphicsUnmapResources
    cuGraphicsUnmapResources.restype = CUresult
    cuGraphicsUnmapResources.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.POINTER(struct_CUgraphicsResource_st)), CUstream]
except AttributeError:
    pass
try:
    cuGetProcAddress = _libraries['libcuda.so'].cuGetProcAddress
    cuGetProcAddress.restype = CUresult
    cuGetProcAddress.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(None)), ctypes.c_int32, cuuint64_t]
except AttributeError:
    pass
try:
    cuGetExportTable = _libraries['libcuda.so'].cuGetExportTable
    cuGetExportTable.restype = CUresult
    cuGetExportTable.argtypes = [ctypes.POINTER(ctypes.POINTER(None)), ctypes.POINTER(struct_CUuuid_st)]
except AttributeError:
    pass
__all__ = \
    ['CUDA_ARRAY3D_DESCRIPTOR', 'CUDA_ARRAY3D_DESCRIPTOR_v2',
    'CUDA_ARRAY_DESCRIPTOR', 'CUDA_ARRAY_DESCRIPTOR_v2',
    'CUDA_ARRAY_SPARSE_PROPERTIES', 'CUDA_ARRAY_SPARSE_PROPERTIES_v1',
    'CUDA_ERROR_ALREADY_ACQUIRED', 'CUDA_ERROR_ALREADY_MAPPED',
    'CUDA_ERROR_ARRAY_IS_MAPPED', 'CUDA_ERROR_ASSERT',
    'CUDA_ERROR_CAPTURED_EVENT',
    'CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE',
    'CUDA_ERROR_CONTEXT_ALREADY_CURRENT',
    'CUDA_ERROR_CONTEXT_ALREADY_IN_USE',
    'CUDA_ERROR_CONTEXT_IS_DESTROYED',
    'CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE',
    'CUDA_ERROR_DEINITIALIZED', 'CUDA_ERROR_DEVICE_NOT_LICENSED',
    'CUDA_ERROR_ECC_UNCORRECTABLE', 'CUDA_ERROR_EXTERNAL_DEVICE',
    'CUDA_ERROR_FILE_NOT_FOUND',
    'CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE',
    'CUDA_ERROR_HARDWARE_STACK_ERROR',
    'CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED',
    'CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED',
    'CUDA_ERROR_ILLEGAL_ADDRESS', 'CUDA_ERROR_ILLEGAL_INSTRUCTION',
    'CUDA_ERROR_ILLEGAL_STATE', 'CUDA_ERROR_INVALID_ADDRESS_SPACE',
    'CUDA_ERROR_INVALID_CONTEXT', 'CUDA_ERROR_INVALID_DEVICE',
    'CUDA_ERROR_INVALID_GRAPHICS_CONTEXT',
    'CUDA_ERROR_INVALID_HANDLE', 'CUDA_ERROR_INVALID_IMAGE',
    'CUDA_ERROR_INVALID_PC', 'CUDA_ERROR_INVALID_PTX',
    'CUDA_ERROR_INVALID_SOURCE', 'CUDA_ERROR_INVALID_VALUE',
    'CUDA_ERROR_JIT_COMPILATION_DISABLED',
    'CUDA_ERROR_JIT_COMPILER_NOT_FOUND', 'CUDA_ERROR_LAUNCH_FAILED',
    'CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING',
    'CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES', 'CUDA_ERROR_LAUNCH_TIMEOUT',
    'CUDA_ERROR_MAP_FAILED', 'CUDA_ERROR_MISALIGNED_ADDRESS',
    'CUDA_ERROR_MPS_CONNECTION_FAILED',
    'CUDA_ERROR_MPS_MAX_CLIENTS_REACHED',
    'CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED',
    'CUDA_ERROR_MPS_RPC_FAILURE', 'CUDA_ERROR_MPS_SERVER_NOT_READY',
    'CUDA_ERROR_NOT_FOUND', 'CUDA_ERROR_NOT_INITIALIZED',
    'CUDA_ERROR_NOT_MAPPED', 'CUDA_ERROR_NOT_MAPPED_AS_ARRAY',
    'CUDA_ERROR_NOT_MAPPED_AS_POINTER', 'CUDA_ERROR_NOT_PERMITTED',
    'CUDA_ERROR_NOT_READY', 'CUDA_ERROR_NOT_SUPPORTED',
    'CUDA_ERROR_NO_BINARY_FOR_GPU', 'CUDA_ERROR_NO_DEVICE',
    'CUDA_ERROR_NVLINK_UNCORRECTABLE', 'CUDA_ERROR_OPERATING_SYSTEM',
    'CUDA_ERROR_OUT_OF_MEMORY',
    'CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED',
    'CUDA_ERROR_PEER_ACCESS_NOT_ENABLED',
    'CUDA_ERROR_PEER_ACCESS_UNSUPPORTED',
    'CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE',
    'CUDA_ERROR_PROFILER_ALREADY_STARTED',
    'CUDA_ERROR_PROFILER_ALREADY_STOPPED',
    'CUDA_ERROR_PROFILER_DISABLED',
    'CUDA_ERROR_PROFILER_NOT_INITIALIZED',
    'CUDA_ERROR_SHARED_OBJECT_INIT_FAILED',
    'CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND',
    'CUDA_ERROR_STREAM_CAPTURE_IMPLICIT',
    'CUDA_ERROR_STREAM_CAPTURE_INVALIDATED',
    'CUDA_ERROR_STREAM_CAPTURE_ISOLATION',
    'CUDA_ERROR_STREAM_CAPTURE_MERGE',
    'CUDA_ERROR_STREAM_CAPTURE_UNJOINED',
    'CUDA_ERROR_STREAM_CAPTURE_UNMATCHED',
    'CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED',
    'CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD',
    'CUDA_ERROR_STUB_LIBRARY', 'CUDA_ERROR_SYSTEM_DRIVER_MISMATCH',
    'CUDA_ERROR_SYSTEM_NOT_READY', 'CUDA_ERROR_TIMEOUT',
    'CUDA_ERROR_TOO_MANY_PEERS', 'CUDA_ERROR_UNKNOWN',
    'CUDA_ERROR_UNMAP_FAILED', 'CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY',
    'CUDA_ERROR_UNSUPPORTED_LIMIT',
    'CUDA_ERROR_UNSUPPORTED_PTX_VERSION',
    'CUDA_EXTERNAL_MEMORY_BUFFER_DESC',
    'CUDA_EXTERNAL_MEMORY_BUFFER_DESC_v1',
    'CUDA_EXTERNAL_MEMORY_HANDLE_DESC',
    'CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1',
    'CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC',
    'CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_v1',
    'CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC',
    'CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1',
    'CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS',
    'CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1',
    'CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS',
    'CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1',
    'CUDA_EXT_SEM_SIGNAL_NODE_PARAMS',
    'CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v1',
    'CUDA_EXT_SEM_WAIT_NODE_PARAMS',
    'CUDA_EXT_SEM_WAIT_NODE_PARAMS_v1',
    'CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH',
    'CUDA_HOST_NODE_PARAMS', 'CUDA_HOST_NODE_PARAMS_v1',
    'CUDA_KERNEL_NODE_PARAMS', 'CUDA_KERNEL_NODE_PARAMS_v1',
    'CUDA_LAUNCH_PARAMS', 'CUDA_LAUNCH_PARAMS_v1', 'CUDA_MEMCPY2D',
    'CUDA_MEMCPY2D_v2', 'CUDA_MEMCPY3D', 'CUDA_MEMCPY3D_PEER',
    'CUDA_MEMCPY3D_PEER_v1', 'CUDA_MEMCPY3D_v2',
    'CUDA_MEMSET_NODE_PARAMS', 'CUDA_MEMSET_NODE_PARAMS_v1',
    'CUDA_MEM_ALLOC_NODE_PARAMS',
    'CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS',
    'CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS__enumvalues',
    'CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS_enum',
    'CUDA_POINTER_ATTRIBUTE_P2P_TOKENS',
    'CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_v1', 'CUDA_RESOURCE_DESC',
    'CUDA_RESOURCE_DESC_v1', 'CUDA_RESOURCE_VIEW_DESC',
    'CUDA_RESOURCE_VIEW_DESC_v1', 'CUDA_SUCCESS', 'CUDA_TEXTURE_DESC',
    'CUDA_TEXTURE_DESC_v1', 'CUGPUDirectRDMAWritesOrdering',
    'CUGPUDirectRDMAWritesOrdering__enumvalues',
    'CUGPUDirectRDMAWritesOrdering_enum', 'CU_ACCESS_PROPERTY_NORMAL',
    'CU_ACCESS_PROPERTY_PERSISTING', 'CU_ACCESS_PROPERTY_STREAMING',
    'CU_AD_FORMAT_BC1_UNORM', 'CU_AD_FORMAT_BC1_UNORM_SRGB',
    'CU_AD_FORMAT_BC2_UNORM', 'CU_AD_FORMAT_BC2_UNORM_SRGB',
    'CU_AD_FORMAT_BC3_UNORM', 'CU_AD_FORMAT_BC3_UNORM_SRGB',
    'CU_AD_FORMAT_BC4_SNORM', 'CU_AD_FORMAT_BC4_UNORM',
    'CU_AD_FORMAT_BC5_SNORM', 'CU_AD_FORMAT_BC5_UNORM',
    'CU_AD_FORMAT_BC6H_SF16', 'CU_AD_FORMAT_BC6H_UF16',
    'CU_AD_FORMAT_BC7_UNORM', 'CU_AD_FORMAT_BC7_UNORM_SRGB',
    'CU_AD_FORMAT_FLOAT', 'CU_AD_FORMAT_HALF', 'CU_AD_FORMAT_NV12',
    'CU_AD_FORMAT_SIGNED_INT16', 'CU_AD_FORMAT_SIGNED_INT32',
    'CU_AD_FORMAT_SIGNED_INT8', 'CU_AD_FORMAT_SNORM_INT16X1',
    'CU_AD_FORMAT_SNORM_INT16X2', 'CU_AD_FORMAT_SNORM_INT16X4',
    'CU_AD_FORMAT_SNORM_INT8X1', 'CU_AD_FORMAT_SNORM_INT8X2',
    'CU_AD_FORMAT_SNORM_INT8X4', 'CU_AD_FORMAT_UNORM_INT16X1',
    'CU_AD_FORMAT_UNORM_INT16X2', 'CU_AD_FORMAT_UNORM_INT16X4',
    'CU_AD_FORMAT_UNORM_INT8X1', 'CU_AD_FORMAT_UNORM_INT8X2',
    'CU_AD_FORMAT_UNORM_INT8X4', 'CU_AD_FORMAT_UNSIGNED_INT16',
    'CU_AD_FORMAT_UNSIGNED_INT32', 'CU_AD_FORMAT_UNSIGNED_INT8',
    'CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_MIPTAIL',
    'CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_SPARSE_LEVEL',
    'CU_COMPUTEMODE_DEFAULT', 'CU_COMPUTEMODE_EXCLUSIVE_PROCESS',
    'CU_COMPUTEMODE_PROHIBITED', 'CU_CTX_BLOCKING_SYNC',
    'CU_CTX_FLAGS_MASK', 'CU_CTX_LMEM_RESIZE_TO_MAX',
    'CU_CTX_MAP_HOST', 'CU_CTX_SCHED_AUTO',
    'CU_CTX_SCHED_BLOCKING_SYNC', 'CU_CTX_SCHED_MASK',
    'CU_CTX_SCHED_SPIN', 'CU_CTX_SCHED_YIELD',
    'CU_CUBEMAP_FACE_NEGATIVE_X', 'CU_CUBEMAP_FACE_NEGATIVE_Y',
    'CU_CUBEMAP_FACE_NEGATIVE_Z', 'CU_CUBEMAP_FACE_POSITIVE_X',
    'CU_CUBEMAP_FACE_POSITIVE_Y', 'CU_CUBEMAP_FACE_POSITIVE_Z',
    'CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT',
    'CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES',
    'CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY',
    'CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER',
    'CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS',
    'CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM',
    'CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS',
    'CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR',
    'CU_DEVICE_ATTRIBUTE_CLOCK_RATE',
    'CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR',
    'CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR',
    'CU_DEVICE_ATTRIBUTE_COMPUTE_MODE',
    'CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED',
    'CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS',
    'CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS',
    'CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH',
    'CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH',
    'CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST',
    'CU_DEVICE_ATTRIBUTE_ECC_ENABLED',
    'CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED',
    'CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED',
    'CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH',
    'CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS',
    'CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED',
    'CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED',
    'CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING',
    'CU_DEVICE_ATTRIBUTE_GPU_OVERLAP',
    'CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED',
    'CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED',
    'CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED',
    'CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED',
    'CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED',
    'CU_DEVICE_ATTRIBUTE_INTEGRATED',
    'CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT',
    'CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE',
    'CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED',
    'CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY', 'CU_DEVICE_ATTRIBUTE_MAX',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH',
    'CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH',
    'CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE',
    'CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR',
    'CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X',
    'CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y',
    'CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z',
    'CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X',
    'CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y',
    'CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z',
    'CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE',
    'CU_DEVICE_ATTRIBUTE_MAX_PITCH',
    'CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK',
    'CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR',
    'CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK',
    'CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN',
    'CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR',
    'CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK',
    'CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR',
    'CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE',
    'CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED',
    'CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES',
    'CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT',
    'CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD',
    'CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID',
    'CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS',
    'CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES',
    'CU_DEVICE_ATTRIBUTE_PCI_BUS_ID',
    'CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID',
    'CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID',
    'CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED',
    'CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK',
    'CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK',
    'CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK',
    'CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO',
    'CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED',
    'CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED',
    'CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT',
    'CU_DEVICE_ATTRIBUTE_TCC_DRIVER',
    'CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT',
    'CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT',
    'CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED',
    'CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY',
    'CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING',
    'CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED',
    'CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED',
    'CU_DEVICE_ATTRIBUTE_WARP_SIZE',
    'CU_DEVICE_P2P_ATTRIBUTE_ACCESS_ACCESS_SUPPORTED',
    'CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED',
    'CU_DEVICE_P2P_ATTRIBUTE_CUDA_ARRAY_ACCESS_SUPPORTED',
    'CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED',
    'CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK',
    'CU_EVENT_BLOCKING_SYNC', 'CU_EVENT_DEFAULT',
    'CU_EVENT_DISABLE_TIMING', 'CU_EVENT_INTERPROCESS',
    'CU_EVENT_RECORD_DEFAULT', 'CU_EVENT_RECORD_EXTERNAL',
    'CU_EVENT_WAIT_DEFAULT', 'CU_EVENT_WAIT_EXTERNAL',
    'CU_EXEC_AFFINITY_TYPE_MAX', 'CU_EXEC_AFFINITY_TYPE_SM_COUNT',
    'CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE',
    'CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE_KMT',
    'CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP',
    'CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE',
    'CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF',
    'CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD',
    'CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32',
    'CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT',
    'CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_FENCE',
    'CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX',
    'CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX_KMT',
    'CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE',
    'CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC',
    'CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD',
    'CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32',
    'CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT',
    'CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD',
    'CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_WIN32',
    'CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_HOST',
    'CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_MEMOPS',
    'CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TARGET_CURRENT_CTX',
    'CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_ALL_DEVICES',
    'CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER',
    'CU_FUNC_ATTRIBUTE_BINARY_VERSION',
    'CU_FUNC_ATTRIBUTE_CACHE_MODE_CA',
    'CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES',
    'CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES', 'CU_FUNC_ATTRIBUTE_MAX',
    'CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES',
    'CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK',
    'CU_FUNC_ATTRIBUTE_NUM_REGS',
    'CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT',
    'CU_FUNC_ATTRIBUTE_PTX_VERSION',
    'CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES',
    'CU_FUNC_CACHE_PREFER_EQUAL', 'CU_FUNC_CACHE_PREFER_L1',
    'CU_FUNC_CACHE_PREFER_NONE', 'CU_FUNC_CACHE_PREFER_SHARED',
    'CU_GET_PROC_ADDRESS_DEFAULT',
    'CU_GET_PROC_ADDRESS_LEGACY_STREAM',
    'CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM',
    'CU_GPU_DIRECT_RDMA_WRITES_ORDERING_ALL_DEVICES',
    'CU_GPU_DIRECT_RDMA_WRITES_ORDERING_NONE',
    'CU_GPU_DIRECT_RDMA_WRITES_ORDERING_OWNER',
    'CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE',
    'CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY',
    'CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD',
    'CU_GRAPHICS_REGISTER_FLAGS_NONE',
    'CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY',
    'CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST',
    'CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER',
    'CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD',
    'CU_GRAPH_DEBUG_DOT_FLAGS_EVENT_NODE_PARAMS',
    'CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_SIGNAL_NODE_PARAMS',
    'CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_WAIT_NODE_PARAMS',
    'CU_GRAPH_DEBUG_DOT_FLAGS_HANDLES',
    'CU_GRAPH_DEBUG_DOT_FLAGS_HOST_NODE_PARAMS',
    'CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_ATTRIBUTES',
    'CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_PARAMS',
    'CU_GRAPH_DEBUG_DOT_FLAGS_MEMCPY_NODE_PARAMS',
    'CU_GRAPH_DEBUG_DOT_FLAGS_MEMSET_NODE_PARAMS',
    'CU_GRAPH_DEBUG_DOT_FLAGS_MEM_ALLOC_NODE_PARAMS',
    'CU_GRAPH_DEBUG_DOT_FLAGS_MEM_FREE_NODE_PARAMS',
    'CU_GRAPH_DEBUG_DOT_FLAGS_RUNTIME_TYPES',
    'CU_GRAPH_DEBUG_DOT_FLAGS_VERBOSE', 'CU_GRAPH_EXEC_UPDATE_ERROR',
    'CU_GRAPH_EXEC_UPDATE_ERROR_FUNCTION_CHANGED',
    'CU_GRAPH_EXEC_UPDATE_ERROR_NODE_TYPE_CHANGED',
    'CU_GRAPH_EXEC_UPDATE_ERROR_NOT_SUPPORTED',
    'CU_GRAPH_EXEC_UPDATE_ERROR_PARAMETERS_CHANGED',
    'CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED',
    'CU_GRAPH_EXEC_UPDATE_ERROR_UNSUPPORTED_FUNCTION_CHANGE',
    'CU_GRAPH_EXEC_UPDATE_SUCCESS',
    'CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT',
    'CU_GRAPH_MEM_ATTR_RESERVED_MEM_HIGH',
    'CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT',
    'CU_GRAPH_MEM_ATTR_USED_MEM_HIGH', 'CU_GRAPH_NODE_TYPE_EMPTY',
    'CU_GRAPH_NODE_TYPE_EVENT_RECORD',
    'CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL',
    'CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT', 'CU_GRAPH_NODE_TYPE_GRAPH',
    'CU_GRAPH_NODE_TYPE_HOST', 'CU_GRAPH_NODE_TYPE_KERNEL',
    'CU_GRAPH_NODE_TYPE_MEMCPY', 'CU_GRAPH_NODE_TYPE_MEMSET',
    'CU_GRAPH_NODE_TYPE_MEM_ALLOC', 'CU_GRAPH_NODE_TYPE_MEM_FREE',
    'CU_GRAPH_NODE_TYPE_WAIT_EVENT', 'CU_GRAPH_USER_OBJECT_MOVE',
    'CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS', 'CU_JIT_CACHE_MODE',
    'CU_JIT_CACHE_OPTION_CA', 'CU_JIT_CACHE_OPTION_CG',
    'CU_JIT_CACHE_OPTION_NONE', 'CU_JIT_ERROR_LOG_BUFFER',
    'CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES', 'CU_JIT_FALLBACK_STRATEGY',
    'CU_JIT_FAST_COMPILE', 'CU_JIT_FMA', 'CU_JIT_FTZ',
    'CU_JIT_GENERATE_DEBUG_INFO', 'CU_JIT_GENERATE_LINE_INFO',
    'CU_JIT_GLOBAL_SYMBOL_ADDRESSES', 'CU_JIT_GLOBAL_SYMBOL_COUNT',
    'CU_JIT_GLOBAL_SYMBOL_NAMES', 'CU_JIT_INFO_LOG_BUFFER',
    'CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES', 'CU_JIT_INPUT_CUBIN',
    'CU_JIT_INPUT_FATBINARY', 'CU_JIT_INPUT_LIBRARY',
    'CU_JIT_INPUT_NVVM', 'CU_JIT_INPUT_OBJECT', 'CU_JIT_INPUT_PTX',
    'CU_JIT_LOG_VERBOSE', 'CU_JIT_LTO', 'CU_JIT_MAX_REGISTERS',
    'CU_JIT_NEW_SM3X_OPT', 'CU_JIT_NUM_INPUT_TYPES',
    'CU_JIT_NUM_OPTIONS', 'CU_JIT_OPTIMIZATION_LEVEL',
    'CU_JIT_PREC_DIV', 'CU_JIT_PREC_SQRT', 'CU_JIT_TARGET',
    'CU_JIT_TARGET_FROM_CUCONTEXT', 'CU_JIT_THREADS_PER_BLOCK',
    'CU_JIT_WALL_TIME',
    'CU_KERNEL_NODE_ATTRIBUTE_ACCESS_POLICY_WINDOW',
    'CU_KERNEL_NODE_ATTRIBUTE_COOPERATIVE',
    'CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT',
    'CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH', 'CU_LIMIT_MALLOC_HEAP_SIZE',
    'CU_LIMIT_MAX', 'CU_LIMIT_MAX_L2_FETCH_GRANULARITY',
    'CU_LIMIT_PERSISTING_L2_CACHE_SIZE', 'CU_LIMIT_PRINTF_FIFO_SIZE',
    'CU_LIMIT_STACK_SIZE', 'CU_MEMORYTYPE_ARRAY',
    'CU_MEMORYTYPE_DEVICE', 'CU_MEMORYTYPE_HOST',
    'CU_MEMORYTYPE_UNIFIED', 'CU_MEMPOOL_ATTR_RELEASE_THRESHOLD',
    'CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT',
    'CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH',
    'CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES',
    'CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC',
    'CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES',
    'CU_MEMPOOL_ATTR_USED_MEM_CURRENT',
    'CU_MEMPOOL_ATTR_USED_MEM_HIGH', 'CU_MEM_ACCESS_FLAGS_PROT_MAX',
    'CU_MEM_ACCESS_FLAGS_PROT_NONE', 'CU_MEM_ACCESS_FLAGS_PROT_READ',
    'CU_MEM_ACCESS_FLAGS_PROT_READWRITE',
    'CU_MEM_ADVISE_SET_ACCESSED_BY',
    'CU_MEM_ADVISE_SET_PREFERRED_LOCATION',
    'CU_MEM_ADVISE_SET_READ_MOSTLY',
    'CU_MEM_ADVISE_UNSET_ACCESSED_BY',
    'CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION',
    'CU_MEM_ADVISE_UNSET_READ_MOSTLY',
    'CU_MEM_ALLOCATION_COMP_GENERIC', 'CU_MEM_ALLOCATION_COMP_NONE',
    'CU_MEM_ALLOCATION_TYPE_INVALID', 'CU_MEM_ALLOCATION_TYPE_MAX',
    'CU_MEM_ALLOCATION_TYPE_PINNED',
    'CU_MEM_ALLOC_GRANULARITY_MINIMUM',
    'CU_MEM_ALLOC_GRANULARITY_RECOMMENDED', 'CU_MEM_ATTACH_GLOBAL',
    'CU_MEM_ATTACH_HOST', 'CU_MEM_ATTACH_SINGLE',
    'CU_MEM_HANDLE_TYPE_GENERIC', 'CU_MEM_HANDLE_TYPE_MAX',
    'CU_MEM_HANDLE_TYPE_NONE',
    'CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR',
    'CU_MEM_HANDLE_TYPE_WIN32', 'CU_MEM_HANDLE_TYPE_WIN32_KMT',
    'CU_MEM_LOCATION_TYPE_DEVICE', 'CU_MEM_LOCATION_TYPE_INVALID',
    'CU_MEM_LOCATION_TYPE_MAX', 'CU_MEM_OPERATION_TYPE_MAP',
    'CU_MEM_OPERATION_TYPE_UNMAP',
    'CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY',
    'CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION',
    'CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION',
    'CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY', 'CU_OCCUPANCY_DEFAULT',
    'CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE',
    'CU_POINTER_ATTRIBUTE_ACCESS_FLAGS',
    'CU_POINTER_ATTRIBUTE_ACCESS_FLAG_NONE',
    'CU_POINTER_ATTRIBUTE_ACCESS_FLAG_READ',
    'CU_POINTER_ATTRIBUTE_ACCESS_FLAG_READWRITE',
    'CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES',
    'CU_POINTER_ATTRIBUTE_BUFFER_ID', 'CU_POINTER_ATTRIBUTE_CONTEXT',
    'CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL',
    'CU_POINTER_ATTRIBUTE_DEVICE_POINTER',
    'CU_POINTER_ATTRIBUTE_HOST_POINTER',
    'CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE',
    'CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE',
    'CU_POINTER_ATTRIBUTE_IS_MANAGED', 'CU_POINTER_ATTRIBUTE_MAPPED',
    'CU_POINTER_ATTRIBUTE_MEMORY_TYPE',
    'CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE',
    'CU_POINTER_ATTRIBUTE_P2P_TOKENS',
    'CU_POINTER_ATTRIBUTE_RANGE_SIZE',
    'CU_POINTER_ATTRIBUTE_RANGE_START_ADDR',
    'CU_POINTER_ATTRIBUTE_SYNC_MEMOPS', 'CU_PREFER_BINARY',
    'CU_PREFER_PTX', 'CU_RESOURCE_TYPE_ARRAY',
    'CU_RESOURCE_TYPE_LINEAR', 'CU_RESOURCE_TYPE_MIPMAPPED_ARRAY',
    'CU_RESOURCE_TYPE_PITCH2D', 'CU_RES_VIEW_FORMAT_FLOAT_1X16',
    'CU_RES_VIEW_FORMAT_FLOAT_1X32', 'CU_RES_VIEW_FORMAT_FLOAT_2X16',
    'CU_RES_VIEW_FORMAT_FLOAT_2X32', 'CU_RES_VIEW_FORMAT_FLOAT_4X16',
    'CU_RES_VIEW_FORMAT_FLOAT_4X32', 'CU_RES_VIEW_FORMAT_NONE',
    'CU_RES_VIEW_FORMAT_SIGNED_BC4', 'CU_RES_VIEW_FORMAT_SIGNED_BC5',
    'CU_RES_VIEW_FORMAT_SIGNED_BC6H', 'CU_RES_VIEW_FORMAT_SINT_1X16',
    'CU_RES_VIEW_FORMAT_SINT_1X32', 'CU_RES_VIEW_FORMAT_SINT_1X8',
    'CU_RES_VIEW_FORMAT_SINT_2X16', 'CU_RES_VIEW_FORMAT_SINT_2X32',
    'CU_RES_VIEW_FORMAT_SINT_2X8', 'CU_RES_VIEW_FORMAT_SINT_4X16',
    'CU_RES_VIEW_FORMAT_SINT_4X32', 'CU_RES_VIEW_FORMAT_SINT_4X8',
    'CU_RES_VIEW_FORMAT_UINT_1X16', 'CU_RES_VIEW_FORMAT_UINT_1X32',
    'CU_RES_VIEW_FORMAT_UINT_1X8', 'CU_RES_VIEW_FORMAT_UINT_2X16',
    'CU_RES_VIEW_FORMAT_UINT_2X32', 'CU_RES_VIEW_FORMAT_UINT_2X8',
    'CU_RES_VIEW_FORMAT_UINT_4X16', 'CU_RES_VIEW_FORMAT_UINT_4X32',
    'CU_RES_VIEW_FORMAT_UINT_4X8', 'CU_RES_VIEW_FORMAT_UNSIGNED_BC1',
    'CU_RES_VIEW_FORMAT_UNSIGNED_BC2',
    'CU_RES_VIEW_FORMAT_UNSIGNED_BC3',
    'CU_RES_VIEW_FORMAT_UNSIGNED_BC4',
    'CU_RES_VIEW_FORMAT_UNSIGNED_BC5',
    'CU_RES_VIEW_FORMAT_UNSIGNED_BC6H',
    'CU_RES_VIEW_FORMAT_UNSIGNED_BC7',
    'CU_SHAREDMEM_CARVEOUT_DEFAULT', 'CU_SHAREDMEM_CARVEOUT_MAX_L1',
    'CU_SHAREDMEM_CARVEOUT_MAX_SHARED',
    'CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE',
    'CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE',
    'CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE',
    'CU_STREAM_ADD_CAPTURE_DEPENDENCIES',
    'CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW',
    'CU_STREAM_ATTRIBUTE_SYNCHRONIZATION_POLICY',
    'CU_STREAM_CAPTURE_MODE_GLOBAL', 'CU_STREAM_CAPTURE_MODE_RELAXED',
    'CU_STREAM_CAPTURE_MODE_THREAD_LOCAL',
    'CU_STREAM_CAPTURE_STATUS_ACTIVE',
    'CU_STREAM_CAPTURE_STATUS_INVALIDATED',
    'CU_STREAM_CAPTURE_STATUS_NONE', 'CU_STREAM_DEFAULT',
    'CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES',
    'CU_STREAM_MEM_OP_WAIT_VALUE_32',
    'CU_STREAM_MEM_OP_WAIT_VALUE_64',
    'CU_STREAM_MEM_OP_WRITE_VALUE_32',
    'CU_STREAM_MEM_OP_WRITE_VALUE_64', 'CU_STREAM_NON_BLOCKING',
    'CU_STREAM_SET_CAPTURE_DEPENDENCIES', 'CU_STREAM_WAIT_VALUE_AND',
    'CU_STREAM_WAIT_VALUE_EQ', 'CU_STREAM_WAIT_VALUE_FLUSH',
    'CU_STREAM_WAIT_VALUE_GEQ', 'CU_STREAM_WAIT_VALUE_NOR',
    'CU_STREAM_WRITE_VALUE_DEFAULT',
    'CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER', 'CU_SYNC_POLICY_AUTO',
    'CU_SYNC_POLICY_BLOCKING_SYNC', 'CU_SYNC_POLICY_SPIN',
    'CU_SYNC_POLICY_YIELD', 'CU_TARGET_COMPUTE_20',
    'CU_TARGET_COMPUTE_21', 'CU_TARGET_COMPUTE_30',
    'CU_TARGET_COMPUTE_32', 'CU_TARGET_COMPUTE_35',
    'CU_TARGET_COMPUTE_37', 'CU_TARGET_COMPUTE_50',
    'CU_TARGET_COMPUTE_52', 'CU_TARGET_COMPUTE_53',
    'CU_TARGET_COMPUTE_60', 'CU_TARGET_COMPUTE_61',
    'CU_TARGET_COMPUTE_62', 'CU_TARGET_COMPUTE_70',
    'CU_TARGET_COMPUTE_72', 'CU_TARGET_COMPUTE_75',
    'CU_TARGET_COMPUTE_80', 'CU_TARGET_COMPUTE_86',
    'CU_TR_ADDRESS_MODE_BORDER', 'CU_TR_ADDRESS_MODE_CLAMP',
    'CU_TR_ADDRESS_MODE_MIRROR', 'CU_TR_ADDRESS_MODE_WRAP',
    'CU_TR_FILTER_MODE_LINEAR', 'CU_TR_FILTER_MODE_POINT',
    'CU_USER_OBJECT_NO_DESTRUCTOR_SYNC', 'CUaccessPolicyWindow',
    'CUaccessPolicyWindow_v1', 'CUaccessProperty',
    'CUaccessProperty__enumvalues', 'CUaccessProperty_enum',
    'CUaddress_mode', 'CUaddress_mode__enumvalues',
    'CUaddress_mode_enum', 'CUarray', 'CUarrayMapInfo',
    'CUarrayMapInfo_v1', 'CUarraySparseSubresourceType',
    'CUarraySparseSubresourceType__enumvalues',
    'CUarraySparseSubresourceType_enum', 'CUarray_cubemap_face',
    'CUarray_cubemap_face__enumvalues', 'CUarray_cubemap_face_enum',
    'CUarray_format', 'CUarray_format__enumvalues',
    'CUarray_format_enum', 'CUcomputemode',
    'CUcomputemode__enumvalues', 'CUcomputemode_enum', 'CUcontext',
    'CUctx_flags', 'CUctx_flags__enumvalues', 'CUctx_flags_enum',
    'CUdevice', 'CUdevice_P2PAttribute',
    'CUdevice_P2PAttribute__enumvalues', 'CUdevice_P2PAttribute_enum',
    'CUdevice_attribute', 'CUdevice_attribute__enumvalues',
    'CUdevice_attribute_enum', 'CUdevice_v1', 'CUdeviceptr',
    'CUdeviceptr_v2', 'CUdevprop', 'CUdevprop_v1',
    'CUdriverProcAddress_flags',
    'CUdriverProcAddress_flags__enumvalues',
    'CUdriverProcAddress_flags_enum', 'CUevent', 'CUevent_flags',
    'CUevent_flags__enumvalues', 'CUevent_flags_enum',
    'CUevent_record_flags', 'CUevent_record_flags__enumvalues',
    'CUevent_record_flags_enum', 'CUevent_wait_flags',
    'CUevent_wait_flags__enumvalues', 'CUevent_wait_flags_enum',
    'CUexecAffinityParam', 'CUexecAffinityParam_v1',
    'CUexecAffinitySmCount', 'CUexecAffinitySmCount_v1',
    'CUexecAffinityType', 'CUexecAffinityType__enumvalues',
    'CUexecAffinityType_enum', 'CUexternalMemory',
    'CUexternalMemoryHandleType',
    'CUexternalMemoryHandleType__enumvalues',
    'CUexternalMemoryHandleType_enum', 'CUexternalSemaphore',
    'CUexternalSemaphoreHandleType',
    'CUexternalSemaphoreHandleType__enumvalues',
    'CUexternalSemaphoreHandleType_enum', 'CUfilter_mode',
    'CUfilter_mode__enumvalues', 'CUfilter_mode_enum',
    'CUflushGPUDirectRDMAWritesOptions',
    'CUflushGPUDirectRDMAWritesOptions__enumvalues',
    'CUflushGPUDirectRDMAWritesOptions_enum',
    'CUflushGPUDirectRDMAWritesScope',
    'CUflushGPUDirectRDMAWritesScope__enumvalues',
    'CUflushGPUDirectRDMAWritesScope_enum',
    'CUflushGPUDirectRDMAWritesTarget',
    'CUflushGPUDirectRDMAWritesTarget__enumvalues',
    'CUflushGPUDirectRDMAWritesTarget_enum', 'CUfunc_cache',
    'CUfunc_cache__enumvalues', 'CUfunc_cache_enum', 'CUfunction',
    'CUfunction_attribute', 'CUfunction_attribute__enumvalues',
    'CUfunction_attribute_enum', 'CUgraph', 'CUgraphDebugDot_flags',
    'CUgraphDebugDot_flags__enumvalues', 'CUgraphDebugDot_flags_enum',
    'CUgraphExec', 'CUgraphExecUpdateResult',
    'CUgraphExecUpdateResult__enumvalues',
    'CUgraphExecUpdateResult_enum', 'CUgraphInstantiate_flags',
    'CUgraphInstantiate_flags__enumvalues',
    'CUgraphInstantiate_flags_enum', 'CUgraphMem_attribute',
    'CUgraphMem_attribute__enumvalues', 'CUgraphMem_attribute_enum',
    'CUgraphNode', 'CUgraphNodeType', 'CUgraphNodeType__enumvalues',
    'CUgraphNodeType_enum', 'CUgraphicsMapResourceFlags',
    'CUgraphicsMapResourceFlags__enumvalues',
    'CUgraphicsMapResourceFlags_enum', 'CUgraphicsRegisterFlags',
    'CUgraphicsRegisterFlags__enumvalues',
    'CUgraphicsRegisterFlags_enum', 'CUgraphicsResource', 'CUhostFn',
    'CUipcEventHandle', 'CUipcEventHandle_v1', 'CUipcMemHandle',
    'CUipcMemHandle_v1', 'CUipcMem_flags',
    'CUipcMem_flags__enumvalues', 'CUipcMem_flags_enum',
    'CUjitInputType', 'CUjitInputType__enumvalues',
    'CUjitInputType_enum', 'CUjit_cacheMode',
    'CUjit_cacheMode__enumvalues', 'CUjit_cacheMode_enum',
    'CUjit_fallback', 'CUjit_fallback__enumvalues',
    'CUjit_fallback_enum', 'CUjit_option', 'CUjit_option__enumvalues',
    'CUjit_option_enum', 'CUjit_target', 'CUjit_target__enumvalues',
    'CUjit_target_enum', 'CUkernelNodeAttrID',
    'CUkernelNodeAttrID__enumvalues', 'CUkernelNodeAttrID_enum',
    'CUkernelNodeAttrValue', 'CUkernelNodeAttrValue_v1', 'CUlimit',
    'CUlimit__enumvalues', 'CUlimit_enum', 'CUlinkState',
    'CUmemAccessDesc', 'CUmemAccessDesc_v1', 'CUmemAccess_flags',
    'CUmemAccess_flags__enumvalues', 'CUmemAccess_flags_enum',
    'CUmemAllocationCompType', 'CUmemAllocationCompType__enumvalues',
    'CUmemAllocationCompType_enum',
    'CUmemAllocationGranularity_flags',
    'CUmemAllocationGranularity_flags__enumvalues',
    'CUmemAllocationGranularity_flags_enum',
    'CUmemAllocationHandleType',
    'CUmemAllocationHandleType__enumvalues',
    'CUmemAllocationHandleType_enum', 'CUmemAllocationProp',
    'CUmemAllocationProp_v1', 'CUmemAllocationType',
    'CUmemAllocationType__enumvalues', 'CUmemAllocationType_enum',
    'CUmemAttach_flags', 'CUmemAttach_flags__enumvalues',
    'CUmemAttach_flags_enum', 'CUmemGenericAllocationHandle',
    'CUmemGenericAllocationHandle_v1', 'CUmemHandleType',
    'CUmemHandleType__enumvalues', 'CUmemHandleType_enum',
    'CUmemLocation', 'CUmemLocationType',
    'CUmemLocationType__enumvalues', 'CUmemLocationType_enum',
    'CUmemLocation_v1', 'CUmemOperationType',
    'CUmemOperationType__enumvalues', 'CUmemOperationType_enum',
    'CUmemPoolProps', 'CUmemPoolProps_v1', 'CUmemPoolPtrExportData',
    'CUmemPoolPtrExportData_v1', 'CUmemPool_attribute',
    'CUmemPool_attribute__enumvalues', 'CUmemPool_attribute_enum',
    'CUmem_advise', 'CUmem_advise__enumvalues', 'CUmem_advise_enum',
    'CUmem_range_attribute', 'CUmem_range_attribute__enumvalues',
    'CUmem_range_attribute_enum', 'CUmemoryPool', 'CUmemorytype',
    'CUmemorytype__enumvalues', 'CUmemorytype_enum',
    'CUmipmappedArray', 'CUmodule', 'CUoccupancyB2DSize',
    'CUoccupancy_flags', 'CUoccupancy_flags__enumvalues',
    'CUoccupancy_flags_enum', 'CUpointer_attribute',
    'CUpointer_attribute__enumvalues', 'CUpointer_attribute_enum',
    'CUresourceViewFormat', 'CUresourceViewFormat__enumvalues',
    'CUresourceViewFormat_enum', 'CUresourcetype',
    'CUresourcetype__enumvalues', 'CUresourcetype_enum', 'CUresult',
    'CUresult__enumvalues', 'CUshared_carveout',
    'CUshared_carveout__enumvalues', 'CUshared_carveout_enum',
    'CUsharedconfig', 'CUsharedconfig__enumvalues',
    'CUsharedconfig_enum', 'CUstream', 'CUstreamAttrID',
    'CUstreamAttrID__enumvalues', 'CUstreamAttrID_enum',
    'CUstreamAttrValue', 'CUstreamAttrValue_v1',
    'CUstreamBatchMemOpParams', 'CUstreamBatchMemOpParams_v1',
    'CUstreamBatchMemOpType', 'CUstreamBatchMemOpType__enumvalues',
    'CUstreamBatchMemOpType_enum', 'CUstreamCallback',
    'CUstreamCaptureMode', 'CUstreamCaptureMode__enumvalues',
    'CUstreamCaptureMode_enum', 'CUstreamCaptureStatus',
    'CUstreamCaptureStatus__enumvalues', 'CUstreamCaptureStatus_enum',
    'CUstreamUpdateCaptureDependencies_flags',
    'CUstreamUpdateCaptureDependencies_flags__enumvalues',
    'CUstreamUpdateCaptureDependencies_flags_enum',
    'CUstreamWaitValue_flags', 'CUstreamWaitValue_flags__enumvalues',
    'CUstreamWaitValue_flags_enum', 'CUstreamWriteValue_flags',
    'CUstreamWriteValue_flags__enumvalues',
    'CUstreamWriteValue_flags_enum', 'CUstream_flags',
    'CUstream_flags__enumvalues', 'CUstream_flags_enum',
    'CUsurfObject', 'CUsurfObject_v1', 'CUsurfref',
    'CUsynchronizationPolicy', 'CUsynchronizationPolicy__enumvalues',
    'CUsynchronizationPolicy_enum', 'CUtexObject', 'CUtexObject_v1',
    'CUtexref', 'CUuserObject', 'CUuserObjectRetain_flags',
    'CUuserObjectRetain_flags__enumvalues',
    'CUuserObjectRetain_flags_enum', 'CUuserObject_flags',
    'CUuserObject_flags__enumvalues', 'CUuserObject_flags_enum',
    'CUuuid', 'cuArray3DCreate_v2', 'cuArray3DGetDescriptor_v2',
    'cuArrayCreate_v2', 'cuArrayDestroy', 'cuArrayGetDescriptor_v2',
    'cuArrayGetPlane', 'cuArrayGetSparseProperties', 'cuCtxAttach',
    'cuCtxCreate_v2', 'cuCtxCreate_v3', 'cuCtxDestroy_v2',
    'cuCtxDetach', 'cuCtxDisablePeerAccess', 'cuCtxEnablePeerAccess',
    'cuCtxGetApiVersion', 'cuCtxGetCacheConfig', 'cuCtxGetCurrent',
    'cuCtxGetDevice', 'cuCtxGetExecAffinity', 'cuCtxGetFlags',
    'cuCtxGetLimit', 'cuCtxGetSharedMemConfig',
    'cuCtxGetStreamPriorityRange', 'cuCtxPopCurrent_v2',
    'cuCtxPushCurrent_v2', 'cuCtxResetPersistingL2Cache',
    'cuCtxSetCacheConfig', 'cuCtxSetCurrent', 'cuCtxSetLimit',
    'cuCtxSetSharedMemConfig', 'cuCtxSynchronize',
    'cuDestroyExternalMemory', 'cuDestroyExternalSemaphore',
    'cuDeviceCanAccessPeer', 'cuDeviceComputeCapability',
    'cuDeviceGet', 'cuDeviceGetAttribute', 'cuDeviceGetByPCIBusId',
    'cuDeviceGetCount', 'cuDeviceGetDefaultMemPool',
    'cuDeviceGetExecAffinitySupport', 'cuDeviceGetGraphMemAttribute',
    'cuDeviceGetLuid', 'cuDeviceGetMemPool', 'cuDeviceGetName',
    'cuDeviceGetNvSciSyncAttributes', 'cuDeviceGetP2PAttribute',
    'cuDeviceGetPCIBusId', 'cuDeviceGetProperties',
    'cuDeviceGetTexture1DLinearMaxWidth', 'cuDeviceGetUuid',
    'cuDeviceGetUuid_v2', 'cuDeviceGraphMemTrim',
    'cuDevicePrimaryCtxGetState', 'cuDevicePrimaryCtxRelease_v2',
    'cuDevicePrimaryCtxReset_v2', 'cuDevicePrimaryCtxRetain',
    'cuDevicePrimaryCtxSetFlags_v2', 'cuDeviceSetGraphMemAttribute',
    'cuDeviceSetMemPool', 'cuDeviceTotalMem_v2', 'cuDriverGetVersion',
    'cuEventCreate', 'cuEventDestroy_v2', 'cuEventElapsedTime',
    'cuEventQuery', 'cuEventRecord', 'cuEventRecordWithFlags',
    'cuEventSynchronize', 'cuExternalMemoryGetMappedBuffer',
    'cuExternalMemoryGetMappedMipmappedArray',
    'cuFlushGPUDirectRDMAWrites', 'cuFuncGetAttribute',
    'cuFuncGetModule', 'cuFuncSetAttribute', 'cuFuncSetBlockShape',
    'cuFuncSetCacheConfig', 'cuFuncSetSharedMemConfig',
    'cuFuncSetSharedSize', 'cuGetErrorName', 'cuGetErrorString',
    'cuGetExportTable', 'cuGetProcAddress',
    'cuGraphAddChildGraphNode', 'cuGraphAddDependencies',
    'cuGraphAddEmptyNode', 'cuGraphAddEventRecordNode',
    'cuGraphAddEventWaitNode',
    'cuGraphAddExternalSemaphoresSignalNode',
    'cuGraphAddExternalSemaphoresWaitNode', 'cuGraphAddHostNode',
    'cuGraphAddKernelNode', 'cuGraphAddMemAllocNode',
    'cuGraphAddMemFreeNode', 'cuGraphAddMemcpyNode',
    'cuGraphAddMemsetNode', 'cuGraphChildGraphNodeGetGraph',
    'cuGraphClone', 'cuGraphCreate', 'cuGraphDebugDotPrint',
    'cuGraphDestroy', 'cuGraphDestroyNode',
    'cuGraphEventRecordNodeGetEvent',
    'cuGraphEventRecordNodeSetEvent', 'cuGraphEventWaitNodeGetEvent',
    'cuGraphEventWaitNodeSetEvent',
    'cuGraphExecChildGraphNodeSetParams', 'cuGraphExecDestroy',
    'cuGraphExecEventRecordNodeSetEvent',
    'cuGraphExecEventWaitNodeSetEvent',
    'cuGraphExecExternalSemaphoresSignalNodeSetParams',
    'cuGraphExecExternalSemaphoresWaitNodeSetParams',
    'cuGraphExecHostNodeSetParams', 'cuGraphExecKernelNodeSetParams',
    'cuGraphExecMemcpyNodeSetParams',
    'cuGraphExecMemsetNodeSetParams', 'cuGraphExecUpdate',
    'cuGraphExternalSemaphoresSignalNodeGetParams',
    'cuGraphExternalSemaphoresSignalNodeSetParams',
    'cuGraphExternalSemaphoresWaitNodeGetParams',
    'cuGraphExternalSemaphoresWaitNodeSetParams', 'cuGraphGetEdges',
    'cuGraphGetNodes', 'cuGraphGetRootNodes',
    'cuGraphHostNodeGetParams', 'cuGraphHostNodeSetParams',
    'cuGraphInstantiateWithFlags', 'cuGraphInstantiate_v2',
    'cuGraphKernelNodeCopyAttributes',
    'cuGraphKernelNodeGetAttribute', 'cuGraphKernelNodeGetParams',
    'cuGraphKernelNodeSetAttribute', 'cuGraphKernelNodeSetParams',
    'cuGraphLaunch', 'cuGraphMemAllocNodeGetParams',
    'cuGraphMemFreeNodeGetParams', 'cuGraphMemcpyNodeGetParams',
    'cuGraphMemcpyNodeSetParams', 'cuGraphMemsetNodeGetParams',
    'cuGraphMemsetNodeSetParams', 'cuGraphNodeFindInClone',
    'cuGraphNodeGetDependencies', 'cuGraphNodeGetDependentNodes',
    'cuGraphNodeGetType', 'cuGraphReleaseUserObject',
    'cuGraphRemoveDependencies', 'cuGraphRetainUserObject',
    'cuGraphUpload', 'cuGraphicsMapResources',
    'cuGraphicsResourceGetMappedMipmappedArray',
    'cuGraphicsResourceGetMappedPointer_v2',
    'cuGraphicsResourceSetMapFlags_v2',
    'cuGraphicsSubResourceGetMappedArray', 'cuGraphicsUnmapResources',
    'cuGraphicsUnregisterResource', 'cuImportExternalMemory',
    'cuImportExternalSemaphore', 'cuInit', 'cuIpcCloseMemHandle',
    'cuIpcGetEventHandle', 'cuIpcGetMemHandle',
    'cuIpcOpenEventHandle', 'cuIpcOpenMemHandle_v2', 'cuLaunch',
    'cuLaunchCooperativeKernel',
    'cuLaunchCooperativeKernelMultiDevice', 'cuLaunchGrid',
    'cuLaunchGridAsync', 'cuLaunchHostFunc', 'cuLaunchKernel',
    'cuLinkAddData_v2', 'cuLinkAddFile_v2', 'cuLinkComplete',
    'cuLinkCreate_v2', 'cuLinkDestroy', 'cuMemAddressFree',
    'cuMemAddressReserve', 'cuMemAdvise', 'cuMemAllocAsync',
    'cuMemAllocFromPoolAsync', 'cuMemAllocHost_v2',
    'cuMemAllocManaged', 'cuMemAllocPitch_v2', 'cuMemAlloc_v2',
    'cuMemCreate', 'cuMemExportToShareableHandle', 'cuMemFreeAsync',
    'cuMemFreeHost', 'cuMemFree_v2', 'cuMemGetAccess',
    'cuMemGetAddressRange_v2', 'cuMemGetAllocationGranularity',
    'cuMemGetAllocationPropertiesFromHandle', 'cuMemGetInfo_v2',
    'cuMemHostAlloc', 'cuMemHostGetDevicePointer_v2',
    'cuMemHostGetFlags', 'cuMemHostRegister_v2',
    'cuMemHostUnregister', 'cuMemImportFromShareableHandle',
    'cuMemMap', 'cuMemMapArrayAsync', 'cuMemPoolCreate',
    'cuMemPoolDestroy', 'cuMemPoolExportPointer',
    'cuMemPoolExportToShareableHandle', 'cuMemPoolGetAccess',
    'cuMemPoolGetAttribute', 'cuMemPoolImportFromShareableHandle',
    'cuMemPoolImportPointer', 'cuMemPoolSetAccess',
    'cuMemPoolSetAttribute', 'cuMemPoolTrimTo', 'cuMemPrefetchAsync',
    'cuMemRangeGetAttribute', 'cuMemRangeGetAttributes',
    'cuMemRelease', 'cuMemRetainAllocationHandle', 'cuMemSetAccess',
    'cuMemUnmap', 'cuMemcpy', 'cuMemcpy2DAsync_v2',
    'cuMemcpy2DUnaligned_v2', 'cuMemcpy2D_v2', 'cuMemcpy3DAsync_v2',
    'cuMemcpy3DPeer', 'cuMemcpy3DPeerAsync', 'cuMemcpy3D_v2',
    'cuMemcpyAsync', 'cuMemcpyAtoA_v2', 'cuMemcpyAtoD_v2',
    'cuMemcpyAtoHAsync_v2', 'cuMemcpyAtoH_v2', 'cuMemcpyDtoA_v2',
    'cuMemcpyDtoDAsync_v2', 'cuMemcpyDtoD_v2', 'cuMemcpyDtoHAsync_v2',
    'cuMemcpyDtoH_v2', 'cuMemcpyHtoAAsync_v2', 'cuMemcpyHtoA_v2',
    'cuMemcpyHtoDAsync_v2', 'cuMemcpyHtoD_v2', 'cuMemcpyPeer',
    'cuMemcpyPeerAsync', 'cuMemsetD16Async', 'cuMemsetD16_v2',
    'cuMemsetD2D16Async', 'cuMemsetD2D16_v2', 'cuMemsetD2D32Async',
    'cuMemsetD2D32_v2', 'cuMemsetD2D8Async', 'cuMemsetD2D8_v2',
    'cuMemsetD32Async', 'cuMemsetD32_v2', 'cuMemsetD8Async',
    'cuMemsetD8_v2', 'cuMipmappedArrayCreate',
    'cuMipmappedArrayDestroy', 'cuMipmappedArrayGetLevel',
    'cuMipmappedArrayGetSparseProperties', 'cuModuleGetFunction',
    'cuModuleGetGlobal_v2', 'cuModuleGetSurfRef', 'cuModuleGetTexRef',
    'cuModuleLoad', 'cuModuleLoadData', 'cuModuleLoadDataEx',
    'cuModuleLoadFatBinary', 'cuModuleUnload',
    'cuOccupancyAvailableDynamicSMemPerBlock',
    'cuOccupancyMaxActiveBlocksPerMultiprocessor',
    'cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags',
    'cuOccupancyMaxPotentialBlockSize',
    'cuOccupancyMaxPotentialBlockSizeWithFlags', 'cuParamSetSize',
    'cuParamSetTexRef', 'cuParamSetf', 'cuParamSeti', 'cuParamSetv',
    'cuPointerGetAttribute', 'cuPointerGetAttributes',
    'cuPointerSetAttribute', 'cuSignalExternalSemaphoresAsync',
    'cuStreamAddCallback', 'cuStreamAttachMemAsync',
    'cuStreamBatchMemOp', 'cuStreamBeginCapture_v2',
    'cuStreamCopyAttributes', 'cuStreamCreate',
    'cuStreamCreateWithPriority', 'cuStreamDestroy_v2',
    'cuStreamEndCapture', 'cuStreamGetAttribute',
    'cuStreamGetCaptureInfo', 'cuStreamGetCaptureInfo_v2',
    'cuStreamGetCtx', 'cuStreamGetFlags', 'cuStreamGetPriority',
    'cuStreamIsCapturing', 'cuStreamQuery', 'cuStreamSetAttribute',
    'cuStreamSynchronize', 'cuStreamUpdateCaptureDependencies',
    'cuStreamWaitEvent', 'cuStreamWaitValue32', 'cuStreamWaitValue64',
    'cuStreamWriteValue32', 'cuStreamWriteValue64',
    'cuSurfObjectCreate', 'cuSurfObjectDestroy',
    'cuSurfObjectGetResourceDesc', 'cuSurfRefGetArray',
    'cuSurfRefSetArray', 'cuTexObjectCreate', 'cuTexObjectDestroy',
    'cuTexObjectGetResourceDesc', 'cuTexObjectGetResourceViewDesc',
    'cuTexObjectGetTextureDesc', 'cuTexRefCreate', 'cuTexRefDestroy',
    'cuTexRefGetAddressMode', 'cuTexRefGetAddress_v2',
    'cuTexRefGetArray', 'cuTexRefGetBorderColor',
    'cuTexRefGetFilterMode', 'cuTexRefGetFlags', 'cuTexRefGetFormat',
    'cuTexRefGetMaxAnisotropy', 'cuTexRefGetMipmapFilterMode',
    'cuTexRefGetMipmapLevelBias', 'cuTexRefGetMipmapLevelClamp',
    'cuTexRefGetMipmappedArray', 'cuTexRefSetAddress2D_v3',
    'cuTexRefSetAddressMode', 'cuTexRefSetAddress_v2',
    'cuTexRefSetArray', 'cuTexRefSetBorderColor',
    'cuTexRefSetFilterMode', 'cuTexRefSetFlags', 'cuTexRefSetFormat',
    'cuTexRefSetMaxAnisotropy', 'cuTexRefSetMipmapFilterMode',
    'cuTexRefSetMipmapLevelBias', 'cuTexRefSetMipmapLevelClamp',
    'cuTexRefSetMipmappedArray', 'cuThreadExchangeStreamCaptureMode',
    'cuUserObjectCreate', 'cuUserObjectRelease', 'cuUserObjectRetain',
    'cuWaitExternalSemaphoresAsync', 'cudaError_enum', 'cuuint32_t',
    'cuuint64_t', 'size_t', 'struct_CUDA_ARRAY3D_DESCRIPTOR_st',
    'struct_CUDA_ARRAY_DESCRIPTOR_st',
    'struct_CUDA_ARRAY_SPARSE_PROPERTIES_st',
    'struct_CUDA_ARRAY_SPARSE_PROPERTIES_st_tileExtent',
    'struct_CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st',
    'struct_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st',
    'struct_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st_0_win32',
    'struct_CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st',
    'struct_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st',
    'struct_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st_0_win32',
    'struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st',
    'struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_0_fence',
    'struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_0_keyedMutex',
    'struct_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_params',
    'struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st',
    'struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_0_fence',
    'struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_0_keyedMutex',
    'struct_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_params',
    'struct_CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_st',
    'struct_CUDA_EXT_SEM_WAIT_NODE_PARAMS_st',
    'struct_CUDA_HOST_NODE_PARAMS_st',
    'struct_CUDA_KERNEL_NODE_PARAMS_st',
    'struct_CUDA_LAUNCH_PARAMS_st', 'struct_CUDA_MEMCPY2D_st',
    'struct_CUDA_MEMCPY3D_PEER_st', 'struct_CUDA_MEMCPY3D_st',
    'struct_CUDA_MEMSET_NODE_PARAMS_st',
    'struct_CUDA_MEM_ALLOC_NODE_PARAMS_st',
    'struct_CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st',
    'struct_CUDA_RESOURCE_DESC_st',
    'struct_CUDA_RESOURCE_DESC_st_0_array',
    'struct_CUDA_RESOURCE_DESC_st_0_linear',
    'struct_CUDA_RESOURCE_DESC_st_0_mipmap',
    'struct_CUDA_RESOURCE_DESC_st_0_pitch2D',
    'struct_CUDA_RESOURCE_DESC_st_0_reserved',
    'struct_CUDA_RESOURCE_VIEW_DESC_st',
    'struct_CUDA_TEXTURE_DESC_st', 'struct_CUaccessPolicyWindow_st',
    'struct_CUarrayMapInfo_st', 'struct_CUarrayMapInfo_st_1_miptail',
    'struct_CUarrayMapInfo_st_1_sparseLevel', 'struct_CUarray_st',
    'struct_CUctx_st', 'struct_CUdevprop_st', 'struct_CUevent_st',
    'struct_CUexecAffinityParam_st',
    'struct_CUexecAffinitySmCount_st', 'struct_CUextMemory_st',
    'struct_CUextSemaphore_st', 'struct_CUfunc_st',
    'struct_CUgraphExec_st', 'struct_CUgraphNode_st',
    'struct_CUgraph_st', 'struct_CUgraphicsResource_st',
    'struct_CUipcEventHandle_st', 'struct_CUipcMemHandle_st',
    'struct_CUlinkState_st', 'struct_CUmemAccessDesc_st',
    'struct_CUmemAllocationProp_st',
    'struct_CUmemAllocationProp_st_allocFlags',
    'struct_CUmemLocation_st', 'struct_CUmemPoolHandle_st',
    'struct_CUmemPoolProps_st', 'struct_CUmemPoolPtrExportData_st',
    'struct_CUmipmappedArray_st', 'struct_CUmod_st',
    'struct_CUstreamMemOpFlushRemoteWritesParams_st',
    'struct_CUstreamMemOpWaitValueParams_st',
    'struct_CUstreamMemOpWriteValueParams_st', 'struct_CUstream_st',
    'struct_CUsurfref_st', 'struct_CUtexref_st',
    'struct_CUuserObject_st', 'struct_CUuuid_st',
    'union_CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st_handle',
    'union_CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st_handle',
    'union_CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st_0_nvSciSync',
    'union_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st_0_nvSciSync',
    'union_CUDA_RESOURCE_DESC_st_res',
    'union_CUarrayMapInfo_st_memHandle',
    'union_CUarrayMapInfo_st_resource',
    'union_CUarrayMapInfo_st_subresource',
    'union_CUexecAffinityParam_st_param',
    'union_CUkernelNodeAttrValue_union',
    'union_CUstreamAttrValue_union',
    'union_CUstreamBatchMemOpParams_union',
    'union_CUstreamMemOpWaitValueParams_st_0',
    'union_CUstreamMemOpWriteValueParams_st_0']
