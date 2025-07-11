# type: ignore
# -*- coding: utf-8 -*-
#
# TARGET arch is: []
# WORD_SIZE is: 8
# POINTER_SIZE is: 8
# LONGDOUBLE_SIZE is: 16
#
import ctypes


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




# values for enumeration 'kgsl_user_mem_type'
kgsl_user_mem_type__enumvalues = {
    0: 'KGSL_USER_MEM_TYPE_PMEM',
    1: 'KGSL_USER_MEM_TYPE_ASHMEM',
    2: 'KGSL_USER_MEM_TYPE_ADDR',
    3: 'KGSL_USER_MEM_TYPE_ION',
    3: 'KGSL_USER_MEM_TYPE_DMABUF',
    7: 'KGSL_USER_MEM_TYPE_MAX',
}
KGSL_USER_MEM_TYPE_PMEM = 0
KGSL_USER_MEM_TYPE_ASHMEM = 1
KGSL_USER_MEM_TYPE_ADDR = 2
KGSL_USER_MEM_TYPE_ION = 3
KGSL_USER_MEM_TYPE_DMABUF = 3
KGSL_USER_MEM_TYPE_MAX = 7
kgsl_user_mem_type = ctypes.c_uint32 # enum

# values for enumeration 'kgsl_ctx_reset_stat'
kgsl_ctx_reset_stat__enumvalues = {
    0: 'KGSL_CTX_STAT_NO_ERROR',
    1: 'KGSL_CTX_STAT_GUILTY_CONTEXT_RESET_EXT',
    2: 'KGSL_CTX_STAT_INNOCENT_CONTEXT_RESET_EXT',
    3: 'KGSL_CTX_STAT_UNKNOWN_CONTEXT_RESET_EXT',
}
KGSL_CTX_STAT_NO_ERROR = 0
KGSL_CTX_STAT_GUILTY_CONTEXT_RESET_EXT = 1
KGSL_CTX_STAT_INNOCENT_CONTEXT_RESET_EXT = 2
KGSL_CTX_STAT_UNKNOWN_CONTEXT_RESET_EXT = 3
kgsl_ctx_reset_stat = ctypes.c_uint32 # enum

# values for enumeration 'kgsl_deviceid'
kgsl_deviceid__enumvalues = {
    0: 'KGSL_DEVICE_3D0',
    1: 'KGSL_DEVICE_MAX',
}
KGSL_DEVICE_3D0 = 0
KGSL_DEVICE_MAX = 1
kgsl_deviceid = ctypes.c_uint32 # enum
class struct_kgsl_devinfo(Structure):
    pass

struct_kgsl_devinfo._pack_ = 1 # source:False
struct_kgsl_devinfo._fields_ = [
    ('device_id', ctypes.c_uint32),
    ('chip_id', ctypes.c_uint32),
    ('mmu_enabled', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('gmem_gpubaseaddr', ctypes.c_uint64),
    ('gpu_id', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('gmem_sizebytes', ctypes.c_uint64),
]

class struct_kgsl_devmemstore(Structure):
    pass

struct_kgsl_devmemstore._pack_ = 1 # source:False
struct_kgsl_devmemstore._fields_ = [
    ('soptimestamp', ctypes.c_uint32),
    ('sbz', ctypes.c_uint32),
    ('eoptimestamp', ctypes.c_uint32),
    ('sbz2', ctypes.c_uint32),
    ('preempted', ctypes.c_uint32),
    ('sbz3', ctypes.c_uint32),
    ('ref_wait_ts', ctypes.c_uint32),
    ('sbz4', ctypes.c_uint32),
    ('current_context', ctypes.c_uint32),
    ('sbz5', ctypes.c_uint32),
]


# values for enumeration 'kgsl_timestamp_type'
kgsl_timestamp_type__enumvalues = {
    1: 'KGSL_TIMESTAMP_CONSUMED',
    2: 'KGSL_TIMESTAMP_RETIRED',
    3: 'KGSL_TIMESTAMP_QUEUED',
}
KGSL_TIMESTAMP_CONSUMED = 1
KGSL_TIMESTAMP_RETIRED = 2
KGSL_TIMESTAMP_QUEUED = 3
kgsl_timestamp_type = ctypes.c_uint32 # enum
class struct_kgsl_shadowprop(Structure):
    pass

struct_kgsl_shadowprop._pack_ = 1 # source:False
struct_kgsl_shadowprop._fields_ = [
    ('gpuaddr', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
    ('flags', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

class struct_kgsl_version(Structure):
    pass

struct_kgsl_version._pack_ = 1 # source:False
struct_kgsl_version._fields_ = [
    ('drv_major', ctypes.c_uint32),
    ('drv_minor', ctypes.c_uint32),
    ('dev_major', ctypes.c_uint32),
    ('dev_minor', ctypes.c_uint32),
]

class struct_kgsl_sp_generic_mem(Structure):
    pass

struct_kgsl_sp_generic_mem._pack_ = 1 # source:False
struct_kgsl_sp_generic_mem._fields_ = [
    ('local', ctypes.c_uint64),
    ('pvt', ctypes.c_uint64),
]

class struct_kgsl_ucode_version(Structure):
    pass

struct_kgsl_ucode_version._pack_ = 1 # source:False
struct_kgsl_ucode_version._fields_ = [
    ('pfp', ctypes.c_uint32),
    ('pm4', ctypes.c_uint32),
]

class struct_kgsl_gpmu_version(Structure):
    pass

struct_kgsl_gpmu_version._pack_ = 1 # source:False
struct_kgsl_gpmu_version._fields_ = [
    ('major', ctypes.c_uint32),
    ('minor', ctypes.c_uint32),
    ('features', ctypes.c_uint32),
]

class struct_kgsl_ibdesc(Structure):
    pass

struct_kgsl_ibdesc._pack_ = 1 # source:False
struct_kgsl_ibdesc._fields_ = [
    ('gpuaddr', ctypes.c_uint64),
    ('__pad', ctypes.c_uint64),
    ('sizedwords', ctypes.c_uint64),
    ('ctrl', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

class struct_kgsl_cmdbatch_profiling_buffer(Structure):
    pass

struct_kgsl_cmdbatch_profiling_buffer._pack_ = 1 # source:False
struct_kgsl_cmdbatch_profiling_buffer._fields_ = [
    ('wall_clock_s', ctypes.c_uint64),
    ('wall_clock_ns', ctypes.c_uint64),
    ('gpu_ticks_queued', ctypes.c_uint64),
    ('gpu_ticks_submitted', ctypes.c_uint64),
    ('gpu_ticks_retired', ctypes.c_uint64),
]

class struct_kgsl_device_getproperty(Structure):
    pass

struct_kgsl_device_getproperty._pack_ = 1 # source:False
struct_kgsl_device_getproperty._fields_ = [
    ('type', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('value', ctypes.POINTER(None)),
    ('sizebytes', ctypes.c_uint64),
]

class struct_kgsl_device_waittimestamp(Structure):
    pass

struct_kgsl_device_waittimestamp._pack_ = 1 # source:False
struct_kgsl_device_waittimestamp._fields_ = [
    ('timestamp', ctypes.c_uint32),
    ('timeout', ctypes.c_uint32),
]

class struct_kgsl_device_waittimestamp_ctxtid(Structure):
    pass

struct_kgsl_device_waittimestamp_ctxtid._pack_ = 1 # source:False
struct_kgsl_device_waittimestamp_ctxtid._fields_ = [
    ('context_id', ctypes.c_uint32),
    ('timestamp', ctypes.c_uint32),
    ('timeout', ctypes.c_uint32),
]

class struct_kgsl_ringbuffer_issueibcmds(Structure):
    pass

struct_kgsl_ringbuffer_issueibcmds._pack_ = 1 # source:False
struct_kgsl_ringbuffer_issueibcmds._fields_ = [
    ('drawctxt_id', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('ibdesc_addr', ctypes.c_uint64),
    ('numibs', ctypes.c_uint32),
    ('timestamp', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

class struct_kgsl_cmdstream_readtimestamp(Structure):
    pass

struct_kgsl_cmdstream_readtimestamp._pack_ = 1 # source:False
struct_kgsl_cmdstream_readtimestamp._fields_ = [
    ('type', ctypes.c_uint32),
    ('timestamp', ctypes.c_uint32),
]

class struct_kgsl_cmdstream_freememontimestamp(Structure):
    pass

struct_kgsl_cmdstream_freememontimestamp._pack_ = 1 # source:False
struct_kgsl_cmdstream_freememontimestamp._fields_ = [
    ('gpuaddr', ctypes.c_uint64),
    ('type', ctypes.c_uint32),
    ('timestamp', ctypes.c_uint32),
]

class struct_kgsl_drawctxt_create(Structure):
    pass

struct_kgsl_drawctxt_create._pack_ = 1 # source:False
struct_kgsl_drawctxt_create._fields_ = [
    ('flags', ctypes.c_uint32),
    ('drawctxt_id', ctypes.c_uint32),
]

class struct_kgsl_drawctxt_destroy(Structure):
    pass

struct_kgsl_drawctxt_destroy._pack_ = 1 # source:False
struct_kgsl_drawctxt_destroy._fields_ = [
    ('drawctxt_id', ctypes.c_uint32),
]

class struct_kgsl_map_user_mem(Structure):
    pass

struct_kgsl_map_user_mem._pack_ = 1 # source:False
struct_kgsl_map_user_mem._fields_ = [
    ('fd', ctypes.c_int32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('gpuaddr', ctypes.c_uint64),
    ('len', ctypes.c_uint64),
    ('offset', ctypes.c_uint64),
    ('hostptr', ctypes.c_uint64),
    ('memtype', kgsl_user_mem_type),
    ('flags', ctypes.c_uint32),
]

class struct_kgsl_cmdstream_readtimestamp_ctxtid(Structure):
    pass

struct_kgsl_cmdstream_readtimestamp_ctxtid._pack_ = 1 # source:False
struct_kgsl_cmdstream_readtimestamp_ctxtid._fields_ = [
    ('context_id', ctypes.c_uint32),
    ('type', ctypes.c_uint32),
    ('timestamp', ctypes.c_uint32),
]

class struct_kgsl_cmdstream_freememontimestamp_ctxtid(Structure):
    pass

struct_kgsl_cmdstream_freememontimestamp_ctxtid._pack_ = 1 # source:False
struct_kgsl_cmdstream_freememontimestamp_ctxtid._fields_ = [
    ('context_id', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('gpuaddr', ctypes.c_uint64),
    ('type', ctypes.c_uint32),
    ('timestamp', ctypes.c_uint32),
]

class struct_kgsl_sharedmem_from_pmem(Structure):
    pass

struct_kgsl_sharedmem_from_pmem._pack_ = 1 # source:False
struct_kgsl_sharedmem_from_pmem._fields_ = [
    ('pmem_fd', ctypes.c_int32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('gpuaddr', ctypes.c_uint64),
    ('len', ctypes.c_uint32),
    ('offset', ctypes.c_uint32),
]

class struct_kgsl_sharedmem_free(Structure):
    pass

struct_kgsl_sharedmem_free._pack_ = 1 # source:False
struct_kgsl_sharedmem_free._fields_ = [
    ('gpuaddr', ctypes.c_uint64),
]

class struct_kgsl_cff_user_event(Structure):
    pass

struct_kgsl_cff_user_event._pack_ = 1 # source:False
struct_kgsl_cff_user_event._fields_ = [
    ('cff_opcode', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 3),
    ('op1', ctypes.c_uint32),
    ('op2', ctypes.c_uint32),
    ('op3', ctypes.c_uint32),
    ('op4', ctypes.c_uint32),
    ('op5', ctypes.c_uint32),
    ('__pad', ctypes.c_uint32 * 2),
]

class struct_kgsl_gmem_desc(Structure):
    pass

struct_kgsl_gmem_desc._pack_ = 1 # source:False
struct_kgsl_gmem_desc._fields_ = [
    ('x', ctypes.c_uint32),
    ('y', ctypes.c_uint32),
    ('width', ctypes.c_uint32),
    ('height', ctypes.c_uint32),
    ('pitch', ctypes.c_uint32),
]

class struct_kgsl_buffer_desc(Structure):
    pass

struct_kgsl_buffer_desc._pack_ = 1 # source:False
struct_kgsl_buffer_desc._fields_ = [
    ('hostptr', ctypes.POINTER(None)),
    ('gpuaddr', ctypes.c_uint64),
    ('size', ctypes.c_int32),
    ('format', ctypes.c_uint32),
    ('pitch', ctypes.c_uint32),
    ('enabled', ctypes.c_uint32),
]

class struct_kgsl_bind_gmem_shadow(Structure):
    pass

struct_kgsl_bind_gmem_shadow._pack_ = 1 # source:False
struct_kgsl_bind_gmem_shadow._fields_ = [
    ('drawctxt_id', ctypes.c_uint32),
    ('gmem_desc', struct_kgsl_gmem_desc),
    ('shadow_x', ctypes.c_uint32),
    ('shadow_y', ctypes.c_uint32),
    ('shadow_buffer', struct_kgsl_buffer_desc),
    ('buffer_id', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

class struct_kgsl_sharedmem_from_vmalloc(Structure):
    pass

struct_kgsl_sharedmem_from_vmalloc._pack_ = 1 # source:False
struct_kgsl_sharedmem_from_vmalloc._fields_ = [
    ('gpuaddr', ctypes.c_uint64),
    ('hostptr', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
]

class struct_kgsl_drawctxt_set_bin_base_offset(Structure):
    pass

struct_kgsl_drawctxt_set_bin_base_offset._pack_ = 1 # source:False
struct_kgsl_drawctxt_set_bin_base_offset._fields_ = [
    ('drawctxt_id', ctypes.c_uint32),
    ('offset', ctypes.c_uint32),
]


# values for enumeration 'kgsl_cmdwindow_type'
kgsl_cmdwindow_type__enumvalues = {
    0: 'KGSL_CMDWINDOW_MIN',
    0: 'KGSL_CMDWINDOW_2D',
    1: 'KGSL_CMDWINDOW_3D',
    2: 'KGSL_CMDWINDOW_MMU',
    255: 'KGSL_CMDWINDOW_ARBITER',
    255: 'KGSL_CMDWINDOW_MAX',
}
KGSL_CMDWINDOW_MIN = 0
KGSL_CMDWINDOW_2D = 0
KGSL_CMDWINDOW_3D = 1
KGSL_CMDWINDOW_MMU = 2
KGSL_CMDWINDOW_ARBITER = 255
KGSL_CMDWINDOW_MAX = 255
kgsl_cmdwindow_type = ctypes.c_uint32 # enum
class struct_kgsl_cmdwindow_write(Structure):
    pass

struct_kgsl_cmdwindow_write._pack_ = 1 # source:False
struct_kgsl_cmdwindow_write._fields_ = [
    ('target', kgsl_cmdwindow_type),
    ('addr', ctypes.c_uint32),
    ('data', ctypes.c_uint32),
]

class struct_kgsl_gpumem_alloc(Structure):
    pass

struct_kgsl_gpumem_alloc._pack_ = 1 # source:False
struct_kgsl_gpumem_alloc._fields_ = [
    ('gpuaddr', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
    ('flags', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

class struct_kgsl_cff_syncmem(Structure):
    pass

struct_kgsl_cff_syncmem._pack_ = 1 # source:False
struct_kgsl_cff_syncmem._fields_ = [
    ('gpuaddr', ctypes.c_uint64),
    ('len', ctypes.c_uint64),
    ('__pad', ctypes.c_uint32 * 2),
]

class struct_kgsl_timestamp_event(Structure):
    pass

struct_kgsl_timestamp_event._pack_ = 1 # source:False
struct_kgsl_timestamp_event._fields_ = [
    ('type', ctypes.c_int32),
    ('timestamp', ctypes.c_uint32),
    ('context_id', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('priv', ctypes.POINTER(None)),
    ('len', ctypes.c_uint64),
]

class struct_kgsl_timestamp_event_genlock(Structure):
    pass

struct_kgsl_timestamp_event_genlock._pack_ = 1 # source:False
struct_kgsl_timestamp_event_genlock._fields_ = [
    ('handle', ctypes.c_int32),
]

class struct_kgsl_timestamp_event_fence(Structure):
    pass

struct_kgsl_timestamp_event_fence._pack_ = 1 # source:False
struct_kgsl_timestamp_event_fence._fields_ = [
    ('fence_fd', ctypes.c_int32),
]

class struct_kgsl_gpumem_alloc_id(Structure):
    pass

struct_kgsl_gpumem_alloc_id._pack_ = 1 # source:False
struct_kgsl_gpumem_alloc_id._fields_ = [
    ('id', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('size', ctypes.c_uint64),
    ('mmapsize', ctypes.c_uint64),
    ('gpuaddr', ctypes.c_uint64),
    ('__pad', ctypes.c_uint64 * 2),
]

class struct_kgsl_gpumem_free_id(Structure):
    pass

struct_kgsl_gpumem_free_id._pack_ = 1 # source:False
struct_kgsl_gpumem_free_id._fields_ = [
    ('id', ctypes.c_uint32),
    ('__pad', ctypes.c_uint32),
]

class struct_kgsl_gpumem_get_info(Structure):
    pass

struct_kgsl_gpumem_get_info._pack_ = 1 # source:False
struct_kgsl_gpumem_get_info._fields_ = [
    ('gpuaddr', ctypes.c_uint64),
    ('id', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('size', ctypes.c_uint64),
    ('mmapsize', ctypes.c_uint64),
    ('useraddr', ctypes.c_uint64),
    ('__pad', ctypes.c_uint64 * 4),
]

class struct_kgsl_gpumem_sync_cache(Structure):
    pass

struct_kgsl_gpumem_sync_cache._pack_ = 1 # source:False
struct_kgsl_gpumem_sync_cache._fields_ = [
    ('gpuaddr', ctypes.c_uint64),
    ('id', ctypes.c_uint32),
    ('op', ctypes.c_uint32),
    ('offset', ctypes.c_uint64),
    ('length', ctypes.c_uint64),
]

class struct_kgsl_perfcounter_get(Structure):
    pass

struct_kgsl_perfcounter_get._pack_ = 1 # source:False
struct_kgsl_perfcounter_get._fields_ = [
    ('groupid', ctypes.c_uint32),
    ('countable', ctypes.c_uint32),
    ('offset', ctypes.c_uint32),
    ('offset_hi', ctypes.c_uint32),
    ('__pad', ctypes.c_uint32),
]

class struct_kgsl_perfcounter_put(Structure):
    pass

struct_kgsl_perfcounter_put._pack_ = 1 # source:False
struct_kgsl_perfcounter_put._fields_ = [
    ('groupid', ctypes.c_uint32),
    ('countable', ctypes.c_uint32),
    ('__pad', ctypes.c_uint32 * 2),
]

class struct_kgsl_perfcounter_query(Structure):
    pass

struct_kgsl_perfcounter_query._pack_ = 1 # source:False
struct_kgsl_perfcounter_query._fields_ = [
    ('groupid', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('countables', ctypes.POINTER(ctypes.c_uint32)),
    ('count', ctypes.c_uint32),
    ('max_counters', ctypes.c_uint32),
    ('__pad', ctypes.c_uint32 * 2),
]

class struct_kgsl_perfcounter_read_group(Structure):
    pass

struct_kgsl_perfcounter_read_group._pack_ = 1 # source:False
struct_kgsl_perfcounter_read_group._fields_ = [
    ('groupid', ctypes.c_uint32),
    ('countable', ctypes.c_uint32),
    ('value', ctypes.c_uint64),
]

class struct_kgsl_perfcounter_read(Structure):
    pass

struct_kgsl_perfcounter_read._pack_ = 1 # source:False
struct_kgsl_perfcounter_read._fields_ = [
    ('reads', ctypes.POINTER(struct_kgsl_perfcounter_read_group)),
    ('count', ctypes.c_uint32),
    ('__pad', ctypes.c_uint32 * 2),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

class struct_kgsl_gpumem_sync_cache_bulk(Structure):
    pass

struct_kgsl_gpumem_sync_cache_bulk._pack_ = 1 # source:False
struct_kgsl_gpumem_sync_cache_bulk._fields_ = [
    ('id_list', ctypes.POINTER(ctypes.c_uint32)),
    ('count', ctypes.c_uint32),
    ('op', ctypes.c_uint32),
    ('__pad', ctypes.c_uint32 * 2),
]

class struct_kgsl_cmd_syncpoint_timestamp(Structure):
    pass

struct_kgsl_cmd_syncpoint_timestamp._pack_ = 1 # source:False
struct_kgsl_cmd_syncpoint_timestamp._fields_ = [
    ('context_id', ctypes.c_uint32),
    ('timestamp', ctypes.c_uint32),
]

class struct_kgsl_cmd_syncpoint_fence(Structure):
    pass

struct_kgsl_cmd_syncpoint_fence._pack_ = 1 # source:False
struct_kgsl_cmd_syncpoint_fence._fields_ = [
    ('fd', ctypes.c_int32),
]

class struct_kgsl_cmd_syncpoint(Structure):
    pass

struct_kgsl_cmd_syncpoint._pack_ = 1 # source:False
struct_kgsl_cmd_syncpoint._fields_ = [
    ('type', ctypes.c_int32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('priv', ctypes.POINTER(None)),
    ('size', ctypes.c_uint64),
]

class struct_kgsl_submit_commands(Structure):
    pass

struct_kgsl_submit_commands._pack_ = 1 # source:False
struct_kgsl_submit_commands._fields_ = [
    ('context_id', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('cmdlist', ctypes.POINTER(struct_kgsl_ibdesc)),
    ('numcmds', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('synclist', ctypes.POINTER(struct_kgsl_cmd_syncpoint)),
    ('numsyncs', ctypes.c_uint32),
    ('timestamp', ctypes.c_uint32),
    ('__pad', ctypes.c_uint32 * 4),
]

class struct_kgsl_device_constraint(Structure):
    pass

struct_kgsl_device_constraint._pack_ = 1 # source:False
struct_kgsl_device_constraint._fields_ = [
    ('type', ctypes.c_uint32),
    ('context_id', ctypes.c_uint32),
    ('data', ctypes.POINTER(None)),
    ('size', ctypes.c_uint64),
]

class struct_kgsl_device_constraint_pwrlevel(Structure):
    pass

struct_kgsl_device_constraint_pwrlevel._pack_ = 1 # source:False
struct_kgsl_device_constraint_pwrlevel._fields_ = [
    ('level', ctypes.c_uint32),
]

class struct_kgsl_syncsource_create(Structure):
    pass

struct_kgsl_syncsource_create._pack_ = 1 # source:False
struct_kgsl_syncsource_create._fields_ = [
    ('id', ctypes.c_uint32),
    ('__pad', ctypes.c_uint32 * 3),
]

class struct_kgsl_syncsource_destroy(Structure):
    pass

struct_kgsl_syncsource_destroy._pack_ = 1 # source:False
struct_kgsl_syncsource_destroy._fields_ = [
    ('id', ctypes.c_uint32),
    ('__pad', ctypes.c_uint32 * 3),
]

class struct_kgsl_syncsource_create_fence(Structure):
    pass

struct_kgsl_syncsource_create_fence._pack_ = 1 # source:False
struct_kgsl_syncsource_create_fence._fields_ = [
    ('id', ctypes.c_uint32),
    ('fence_fd', ctypes.c_int32),
    ('__pad', ctypes.c_uint32 * 4),
]

class struct_kgsl_syncsource_signal_fence(Structure):
    pass

struct_kgsl_syncsource_signal_fence._pack_ = 1 # source:False
struct_kgsl_syncsource_signal_fence._fields_ = [
    ('id', ctypes.c_uint32),
    ('fence_fd', ctypes.c_int32),
    ('__pad', ctypes.c_uint32 * 4),
]

class struct_kgsl_cff_sync_gpuobj(Structure):
    pass

struct_kgsl_cff_sync_gpuobj._pack_ = 1 # source:False
struct_kgsl_cff_sync_gpuobj._fields_ = [
    ('offset', ctypes.c_uint64),
    ('length', ctypes.c_uint64),
    ('id', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

class struct_kgsl_gpuobj_alloc(Structure):
    pass

struct_kgsl_gpuobj_alloc._pack_ = 1 # source:False
struct_kgsl_gpuobj_alloc._fields_ = [
    ('size', ctypes.c_uint64),
    ('flags', ctypes.c_uint64),
    ('va_len', ctypes.c_uint64),
    ('mmapsize', ctypes.c_uint64),
    ('id', ctypes.c_uint32),
    ('metadata_len', ctypes.c_uint32),
    ('metadata', ctypes.c_uint64),
]

class struct_kgsl_gpuobj_free(Structure):
    pass

struct_kgsl_gpuobj_free._pack_ = 1 # source:False
struct_kgsl_gpuobj_free._fields_ = [
    ('flags', ctypes.c_uint64),
    ('priv', ctypes.c_uint64),
    ('id', ctypes.c_uint32),
    ('type', ctypes.c_uint32),
    ('len', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

class struct_kgsl_gpu_event_timestamp(Structure):
    pass

struct_kgsl_gpu_event_timestamp._pack_ = 1 # source:False
struct_kgsl_gpu_event_timestamp._fields_ = [
    ('context_id', ctypes.c_uint32),
    ('timestamp', ctypes.c_uint32),
]

class struct_kgsl_gpu_event_fence(Structure):
    pass

struct_kgsl_gpu_event_fence._pack_ = 1 # source:False
struct_kgsl_gpu_event_fence._fields_ = [
    ('fd', ctypes.c_int32),
]

class struct_kgsl_gpuobj_info(Structure):
    pass

struct_kgsl_gpuobj_info._pack_ = 1 # source:False
struct_kgsl_gpuobj_info._fields_ = [
    ('gpuaddr', ctypes.c_uint64),
    ('flags', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
    ('va_len', ctypes.c_uint64),
    ('va_addr', ctypes.c_uint64),
    ('id', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

class struct_kgsl_gpuobj_import(Structure):
    pass

struct_kgsl_gpuobj_import._pack_ = 1 # source:False
struct_kgsl_gpuobj_import._fields_ = [
    ('priv', ctypes.c_uint64),
    ('priv_len', ctypes.c_uint64),
    ('flags', ctypes.c_uint64),
    ('type', ctypes.c_uint32),
    ('id', ctypes.c_uint32),
]

class struct_kgsl_gpuobj_import_dma_buf(Structure):
    pass

struct_kgsl_gpuobj_import_dma_buf._pack_ = 1 # source:False
struct_kgsl_gpuobj_import_dma_buf._fields_ = [
    ('fd', ctypes.c_int32),
]

class struct_kgsl_gpuobj_import_useraddr(Structure):
    pass

struct_kgsl_gpuobj_import_useraddr._pack_ = 1 # source:False
struct_kgsl_gpuobj_import_useraddr._fields_ = [
    ('virtaddr', ctypes.c_uint64),
]

class struct_kgsl_gpuobj_sync_obj(Structure):
    pass

struct_kgsl_gpuobj_sync_obj._pack_ = 1 # source:False
struct_kgsl_gpuobj_sync_obj._fields_ = [
    ('offset', ctypes.c_uint64),
    ('length', ctypes.c_uint64),
    ('id', ctypes.c_uint32),
    ('op', ctypes.c_uint32),
]

class struct_kgsl_gpuobj_sync(Structure):
    pass

struct_kgsl_gpuobj_sync._pack_ = 1 # source:False
struct_kgsl_gpuobj_sync._fields_ = [
    ('objs', ctypes.c_uint64),
    ('obj_len', ctypes.c_uint32),
    ('count', ctypes.c_uint32),
]

class struct_kgsl_command_object(Structure):
    pass

struct_kgsl_command_object._pack_ = 1 # source:False
struct_kgsl_command_object._fields_ = [
    ('offset', ctypes.c_uint64),
    ('gpuaddr', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
    ('flags', ctypes.c_uint32),
    ('id', ctypes.c_uint32),
]

class struct_kgsl_command_syncpoint(Structure):
    pass

struct_kgsl_command_syncpoint._pack_ = 1 # source:False
struct_kgsl_command_syncpoint._fields_ = [
    ('priv', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
    ('type', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

class struct_kgsl_gpu_command(Structure):
    pass

struct_kgsl_gpu_command._pack_ = 1 # source:False
struct_kgsl_gpu_command._fields_ = [
    ('flags', ctypes.c_uint64),
    ('cmdlist', ctypes.c_uint64),
    ('cmdsize', ctypes.c_uint32),
    ('numcmds', ctypes.c_uint32),
    ('objlist', ctypes.c_uint64),
    ('objsize', ctypes.c_uint32),
    ('numobjs', ctypes.c_uint32),
    ('synclist', ctypes.c_uint64),
    ('syncsize', ctypes.c_uint32),
    ('numsyncs', ctypes.c_uint32),
    ('context_id', ctypes.c_uint32),
    ('timestamp', ctypes.c_uint32),
]

class struct_kgsl_preemption_counters_query(Structure):
    pass

struct_kgsl_preemption_counters_query._pack_ = 1 # source:False
struct_kgsl_preemption_counters_query._fields_ = [
    ('counters', ctypes.c_uint64),
    ('size_user', ctypes.c_uint32),
    ('size_priority_level', ctypes.c_uint32),
    ('max_priority_level', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

class struct_kgsl_gpuobj_set_info(Structure):
    pass

struct_kgsl_gpuobj_set_info._pack_ = 1 # source:False
struct_kgsl_gpuobj_set_info._fields_ = [
    ('flags', ctypes.c_uint64),
    ('metadata', ctypes.c_uint64),
    ('id', ctypes.c_uint32),
    ('metadata_len', ctypes.c_uint32),
    ('type', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

__all__ = \
    ['KGSL_CMDWINDOW_2D', 'KGSL_CMDWINDOW_3D',
    'KGSL_CMDWINDOW_ARBITER', 'KGSL_CMDWINDOW_MAX',
    'KGSL_CMDWINDOW_MIN', 'KGSL_CMDWINDOW_MMU',
    'KGSL_CTX_STAT_GUILTY_CONTEXT_RESET_EXT',
    'KGSL_CTX_STAT_INNOCENT_CONTEXT_RESET_EXT',
    'KGSL_CTX_STAT_NO_ERROR',
    'KGSL_CTX_STAT_UNKNOWN_CONTEXT_RESET_EXT', 'KGSL_DEVICE_3D0',
    'KGSL_DEVICE_MAX', 'KGSL_TIMESTAMP_CONSUMED',
    'KGSL_TIMESTAMP_QUEUED', 'KGSL_TIMESTAMP_RETIRED',
    'KGSL_USER_MEM_TYPE_ADDR', 'KGSL_USER_MEM_TYPE_ASHMEM',
    'KGSL_USER_MEM_TYPE_DMABUF', 'KGSL_USER_MEM_TYPE_ION',
    'KGSL_USER_MEM_TYPE_MAX', 'KGSL_USER_MEM_TYPE_PMEM',
    'kgsl_cmdwindow_type', 'kgsl_ctx_reset_stat', 'kgsl_deviceid',
    'kgsl_timestamp_type', 'kgsl_user_mem_type',
    'struct_kgsl_bind_gmem_shadow', 'struct_kgsl_buffer_desc',
    'struct_kgsl_cff_sync_gpuobj', 'struct_kgsl_cff_syncmem',
    'struct_kgsl_cff_user_event', 'struct_kgsl_cmd_syncpoint',
    'struct_kgsl_cmd_syncpoint_fence',
    'struct_kgsl_cmd_syncpoint_timestamp',
    'struct_kgsl_cmdbatch_profiling_buffer',
    'struct_kgsl_cmdstream_freememontimestamp',
    'struct_kgsl_cmdstream_freememontimestamp_ctxtid',
    'struct_kgsl_cmdstream_readtimestamp',
    'struct_kgsl_cmdstream_readtimestamp_ctxtid',
    'struct_kgsl_cmdwindow_write', 'struct_kgsl_command_object',
    'struct_kgsl_command_syncpoint', 'struct_kgsl_device_constraint',
    'struct_kgsl_device_constraint_pwrlevel',
    'struct_kgsl_device_getproperty',
    'struct_kgsl_device_waittimestamp',
    'struct_kgsl_device_waittimestamp_ctxtid', 'struct_kgsl_devinfo',
    'struct_kgsl_devmemstore', 'struct_kgsl_drawctxt_create',
    'struct_kgsl_drawctxt_destroy',
    'struct_kgsl_drawctxt_set_bin_base_offset',
    'struct_kgsl_gmem_desc', 'struct_kgsl_gpmu_version',
    'struct_kgsl_gpu_command', 'struct_kgsl_gpu_event_fence',
    'struct_kgsl_gpu_event_timestamp', 'struct_kgsl_gpumem_alloc',
    'struct_kgsl_gpumem_alloc_id', 'struct_kgsl_gpumem_free_id',
    'struct_kgsl_gpumem_get_info', 'struct_kgsl_gpumem_sync_cache',
    'struct_kgsl_gpumem_sync_cache_bulk', 'struct_kgsl_gpuobj_alloc',
    'struct_kgsl_gpuobj_free', 'struct_kgsl_gpuobj_import',
    'struct_kgsl_gpuobj_import_dma_buf',
    'struct_kgsl_gpuobj_import_useraddr', 'struct_kgsl_gpuobj_info',
    'struct_kgsl_gpuobj_set_info', 'struct_kgsl_gpuobj_sync',
    'struct_kgsl_gpuobj_sync_obj', 'struct_kgsl_ibdesc',
    'struct_kgsl_map_user_mem', 'struct_kgsl_perfcounter_get',
    'struct_kgsl_perfcounter_put', 'struct_kgsl_perfcounter_query',
    'struct_kgsl_perfcounter_read',
    'struct_kgsl_perfcounter_read_group',
    'struct_kgsl_preemption_counters_query',
    'struct_kgsl_ringbuffer_issueibcmds', 'struct_kgsl_shadowprop',
    'struct_kgsl_sharedmem_free', 'struct_kgsl_sharedmem_from_pmem',
    'struct_kgsl_sharedmem_from_vmalloc',
    'struct_kgsl_sp_generic_mem', 'struct_kgsl_submit_commands',
    'struct_kgsl_syncsource_create',
    'struct_kgsl_syncsource_create_fence',
    'struct_kgsl_syncsource_destroy',
    'struct_kgsl_syncsource_signal_fence',
    'struct_kgsl_timestamp_event',
    'struct_kgsl_timestamp_event_fence',
    'struct_kgsl_timestamp_event_genlock',
    'struct_kgsl_ucode_version', 'struct_kgsl_version']
