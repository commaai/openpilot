# mypy: ignore-errors
import ctypes
from tinygrad.runtime.support.c import DLL, Struct, CEnum, _IO, _IOW, _IOR, _IOWR
enum_kgsl_user_mem_type = CEnum(ctypes.c_uint32)
KGSL_USER_MEM_TYPE_PMEM = enum_kgsl_user_mem_type.define('KGSL_USER_MEM_TYPE_PMEM', 0)
KGSL_USER_MEM_TYPE_ASHMEM = enum_kgsl_user_mem_type.define('KGSL_USER_MEM_TYPE_ASHMEM', 1)
KGSL_USER_MEM_TYPE_ADDR = enum_kgsl_user_mem_type.define('KGSL_USER_MEM_TYPE_ADDR', 2)
KGSL_USER_MEM_TYPE_ION = enum_kgsl_user_mem_type.define('KGSL_USER_MEM_TYPE_ION', 3)
KGSL_USER_MEM_TYPE_DMABUF = enum_kgsl_user_mem_type.define('KGSL_USER_MEM_TYPE_DMABUF', 3)
KGSL_USER_MEM_TYPE_MAX = enum_kgsl_user_mem_type.define('KGSL_USER_MEM_TYPE_MAX', 7)

enum_kgsl_ctx_reset_stat = CEnum(ctypes.c_uint32)
KGSL_CTX_STAT_NO_ERROR = enum_kgsl_ctx_reset_stat.define('KGSL_CTX_STAT_NO_ERROR', 0)
KGSL_CTX_STAT_GUILTY_CONTEXT_RESET_EXT = enum_kgsl_ctx_reset_stat.define('KGSL_CTX_STAT_GUILTY_CONTEXT_RESET_EXT', 1)
KGSL_CTX_STAT_INNOCENT_CONTEXT_RESET_EXT = enum_kgsl_ctx_reset_stat.define('KGSL_CTX_STAT_INNOCENT_CONTEXT_RESET_EXT', 2)
KGSL_CTX_STAT_UNKNOWN_CONTEXT_RESET_EXT = enum_kgsl_ctx_reset_stat.define('KGSL_CTX_STAT_UNKNOWN_CONTEXT_RESET_EXT', 3)

enum_kgsl_deviceid = CEnum(ctypes.c_uint32)
KGSL_DEVICE_3D0 = enum_kgsl_deviceid.define('KGSL_DEVICE_3D0', 0)
KGSL_DEVICE_MAX = enum_kgsl_deviceid.define('KGSL_DEVICE_MAX', 1)

class struct_kgsl_devinfo(Struct): pass
struct_kgsl_devinfo._fields_ = [
  ('device_id', ctypes.c_uint32),
  ('chip_id', ctypes.c_uint32),
  ('mmu_enabled', ctypes.c_uint32),
  ('gmem_gpubaseaddr', ctypes.c_uint64),
  ('gpu_id', ctypes.c_uint32),
  ('gmem_sizebytes', ctypes.c_uint64),
]
class struct_kgsl_devmemstore(Struct): pass
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
enum_kgsl_timestamp_type = CEnum(ctypes.c_uint32)
KGSL_TIMESTAMP_CONSUMED = enum_kgsl_timestamp_type.define('KGSL_TIMESTAMP_CONSUMED', 1)
KGSL_TIMESTAMP_RETIRED = enum_kgsl_timestamp_type.define('KGSL_TIMESTAMP_RETIRED', 2)
KGSL_TIMESTAMP_QUEUED = enum_kgsl_timestamp_type.define('KGSL_TIMESTAMP_QUEUED', 3)

class struct_kgsl_shadowprop(Struct): pass
struct_kgsl_shadowprop._fields_ = [
  ('gpuaddr', ctypes.c_uint64),
  ('size', ctypes.c_uint64),
  ('flags', ctypes.c_uint32),
]
class struct_kgsl_version(Struct): pass
struct_kgsl_version._fields_ = [
  ('drv_major', ctypes.c_uint32),
  ('drv_minor', ctypes.c_uint32),
  ('dev_major', ctypes.c_uint32),
  ('dev_minor', ctypes.c_uint32),
]
class struct_kgsl_sp_generic_mem(Struct): pass
struct_kgsl_sp_generic_mem._fields_ = [
  ('local', ctypes.c_uint64),
  ('pvt', ctypes.c_uint64),
]
class struct_kgsl_ucode_version(Struct): pass
struct_kgsl_ucode_version._fields_ = [
  ('pfp', ctypes.c_uint32),
  ('pm4', ctypes.c_uint32),
]
class struct_kgsl_gpmu_version(Struct): pass
struct_kgsl_gpmu_version._fields_ = [
  ('major', ctypes.c_uint32),
  ('minor', ctypes.c_uint32),
  ('features', ctypes.c_uint32),
]
class struct_kgsl_ibdesc(Struct): pass
struct_kgsl_ibdesc._fields_ = [
  ('gpuaddr', ctypes.c_uint64),
  ('__pad', ctypes.c_uint64),
  ('sizedwords', ctypes.c_uint64),
  ('ctrl', ctypes.c_uint32),
]
class struct_kgsl_cmdbatch_profiling_buffer(Struct): pass
struct_kgsl_cmdbatch_profiling_buffer._fields_ = [
  ('wall_clock_s', ctypes.c_uint64),
  ('wall_clock_ns', ctypes.c_uint64),
  ('gpu_ticks_queued', ctypes.c_uint64),
  ('gpu_ticks_submitted', ctypes.c_uint64),
  ('gpu_ticks_retired', ctypes.c_uint64),
]
class struct_kgsl_device_getproperty(Struct): pass
struct_kgsl_device_getproperty._fields_ = [
  ('type', ctypes.c_uint32),
  ('value', ctypes.c_void_p),
  ('sizebytes', ctypes.c_uint64),
]
class struct_kgsl_device_waittimestamp(Struct): pass
struct_kgsl_device_waittimestamp._fields_ = [
  ('timestamp', ctypes.c_uint32),
  ('timeout', ctypes.c_uint32),
]
class struct_kgsl_device_waittimestamp_ctxtid(Struct): pass
struct_kgsl_device_waittimestamp_ctxtid._fields_ = [
  ('context_id', ctypes.c_uint32),
  ('timestamp', ctypes.c_uint32),
  ('timeout', ctypes.c_uint32),
]
class struct_kgsl_ringbuffer_issueibcmds(Struct): pass
struct_kgsl_ringbuffer_issueibcmds._fields_ = [
  ('drawctxt_id', ctypes.c_uint32),
  ('ibdesc_addr', ctypes.c_uint64),
  ('numibs', ctypes.c_uint32),
  ('timestamp', ctypes.c_uint32),
  ('flags', ctypes.c_uint32),
]
class struct_kgsl_cmdstream_readtimestamp(Struct): pass
struct_kgsl_cmdstream_readtimestamp._fields_ = [
  ('type', ctypes.c_uint32),
  ('timestamp', ctypes.c_uint32),
]
class struct_kgsl_cmdstream_freememontimestamp(Struct): pass
struct_kgsl_cmdstream_freememontimestamp._fields_ = [
  ('gpuaddr', ctypes.c_uint64),
  ('type', ctypes.c_uint32),
  ('timestamp', ctypes.c_uint32),
]
class struct_kgsl_drawctxt_create(Struct): pass
struct_kgsl_drawctxt_create._fields_ = [
  ('flags', ctypes.c_uint32),
  ('drawctxt_id', ctypes.c_uint32),
]
class struct_kgsl_drawctxt_destroy(Struct): pass
struct_kgsl_drawctxt_destroy._fields_ = [
  ('drawctxt_id', ctypes.c_uint32),
]
class struct_kgsl_map_user_mem(Struct): pass
struct_kgsl_map_user_mem._fields_ = [
  ('fd', ctypes.c_int32),
  ('gpuaddr', ctypes.c_uint64),
  ('len', ctypes.c_uint64),
  ('offset', ctypes.c_uint64),
  ('hostptr', ctypes.c_uint64),
  ('memtype', enum_kgsl_user_mem_type),
  ('flags', ctypes.c_uint32),
]
class struct_kgsl_cmdstream_readtimestamp_ctxtid(Struct): pass
struct_kgsl_cmdstream_readtimestamp_ctxtid._fields_ = [
  ('context_id', ctypes.c_uint32),
  ('type', ctypes.c_uint32),
  ('timestamp', ctypes.c_uint32),
]
class struct_kgsl_cmdstream_freememontimestamp_ctxtid(Struct): pass
struct_kgsl_cmdstream_freememontimestamp_ctxtid._fields_ = [
  ('context_id', ctypes.c_uint32),
  ('gpuaddr', ctypes.c_uint64),
  ('type', ctypes.c_uint32),
  ('timestamp', ctypes.c_uint32),
]
class struct_kgsl_sharedmem_from_pmem(Struct): pass
struct_kgsl_sharedmem_from_pmem._fields_ = [
  ('pmem_fd', ctypes.c_int32),
  ('gpuaddr', ctypes.c_uint64),
  ('len', ctypes.c_uint32),
  ('offset', ctypes.c_uint32),
]
class struct_kgsl_sharedmem_free(Struct): pass
struct_kgsl_sharedmem_free._fields_ = [
  ('gpuaddr', ctypes.c_uint64),
]
class struct_kgsl_cff_user_event(Struct): pass
struct_kgsl_cff_user_event._fields_ = [
  ('cff_opcode', ctypes.c_ubyte),
  ('op1', ctypes.c_uint32),
  ('op2', ctypes.c_uint32),
  ('op3', ctypes.c_uint32),
  ('op4', ctypes.c_uint32),
  ('op5', ctypes.c_uint32),
  ('__pad', (ctypes.c_uint32 * 2)),
]
class struct_kgsl_gmem_desc(Struct): pass
struct_kgsl_gmem_desc._fields_ = [
  ('x', ctypes.c_uint32),
  ('y', ctypes.c_uint32),
  ('width', ctypes.c_uint32),
  ('height', ctypes.c_uint32),
  ('pitch', ctypes.c_uint32),
]
class struct_kgsl_buffer_desc(Struct): pass
struct_kgsl_buffer_desc._fields_ = [
  ('hostptr', ctypes.c_void_p),
  ('gpuaddr', ctypes.c_uint64),
  ('size', ctypes.c_int32),
  ('format', ctypes.c_uint32),
  ('pitch', ctypes.c_uint32),
  ('enabled', ctypes.c_uint32),
]
class struct_kgsl_bind_gmem_shadow(Struct): pass
struct_kgsl_bind_gmem_shadow._fields_ = [
  ('drawctxt_id', ctypes.c_uint32),
  ('gmem_desc', struct_kgsl_gmem_desc),
  ('shadow_x', ctypes.c_uint32),
  ('shadow_y', ctypes.c_uint32),
  ('shadow_buffer', struct_kgsl_buffer_desc),
  ('buffer_id', ctypes.c_uint32),
]
class struct_kgsl_sharedmem_from_vmalloc(Struct): pass
struct_kgsl_sharedmem_from_vmalloc._fields_ = [
  ('gpuaddr', ctypes.c_uint64),
  ('hostptr', ctypes.c_uint32),
  ('flags', ctypes.c_uint32),
]
class struct_kgsl_drawctxt_set_bin_base_offset(Struct): pass
struct_kgsl_drawctxt_set_bin_base_offset._fields_ = [
  ('drawctxt_id', ctypes.c_uint32),
  ('offset', ctypes.c_uint32),
]
enum_kgsl_cmdwindow_type = CEnum(ctypes.c_uint32)
KGSL_CMDWINDOW_MIN = enum_kgsl_cmdwindow_type.define('KGSL_CMDWINDOW_MIN', 0)
KGSL_CMDWINDOW_2D = enum_kgsl_cmdwindow_type.define('KGSL_CMDWINDOW_2D', 0)
KGSL_CMDWINDOW_3D = enum_kgsl_cmdwindow_type.define('KGSL_CMDWINDOW_3D', 1)
KGSL_CMDWINDOW_MMU = enum_kgsl_cmdwindow_type.define('KGSL_CMDWINDOW_MMU', 2)
KGSL_CMDWINDOW_ARBITER = enum_kgsl_cmdwindow_type.define('KGSL_CMDWINDOW_ARBITER', 255)
KGSL_CMDWINDOW_MAX = enum_kgsl_cmdwindow_type.define('KGSL_CMDWINDOW_MAX', 255)

class struct_kgsl_cmdwindow_write(Struct): pass
struct_kgsl_cmdwindow_write._fields_ = [
  ('target', enum_kgsl_cmdwindow_type),
  ('addr', ctypes.c_uint32),
  ('data', ctypes.c_uint32),
]
class struct_kgsl_gpumem_alloc(Struct): pass
struct_kgsl_gpumem_alloc._fields_ = [
  ('gpuaddr', ctypes.c_uint64),
  ('size', ctypes.c_uint64),
  ('flags', ctypes.c_uint32),
]
class struct_kgsl_cff_syncmem(Struct): pass
struct_kgsl_cff_syncmem._fields_ = [
  ('gpuaddr', ctypes.c_uint64),
  ('len', ctypes.c_uint64),
  ('__pad', (ctypes.c_uint32 * 2)),
]
class struct_kgsl_timestamp_event(Struct): pass
struct_kgsl_timestamp_event._fields_ = [
  ('type', ctypes.c_int32),
  ('timestamp', ctypes.c_uint32),
  ('context_id', ctypes.c_uint32),
  ('priv', ctypes.c_void_p),
  ('len', ctypes.c_uint64),
]
class struct_kgsl_timestamp_event_genlock(Struct): pass
struct_kgsl_timestamp_event_genlock._fields_ = [
  ('handle', ctypes.c_int32),
]
class struct_kgsl_timestamp_event_fence(Struct): pass
struct_kgsl_timestamp_event_fence._fields_ = [
  ('fence_fd', ctypes.c_int32),
]
class struct_kgsl_gpumem_alloc_id(Struct): pass
struct_kgsl_gpumem_alloc_id._fields_ = [
  ('id', ctypes.c_uint32),
  ('flags', ctypes.c_uint32),
  ('size', ctypes.c_uint64),
  ('mmapsize', ctypes.c_uint64),
  ('gpuaddr', ctypes.c_uint64),
  ('__pad', (ctypes.c_uint64 * 2)),
]
class struct_kgsl_gpumem_free_id(Struct): pass
struct_kgsl_gpumem_free_id._fields_ = [
  ('id', ctypes.c_uint32),
  ('__pad', ctypes.c_uint32),
]
class struct_kgsl_gpumem_get_info(Struct): pass
struct_kgsl_gpumem_get_info._fields_ = [
  ('gpuaddr', ctypes.c_uint64),
  ('id', ctypes.c_uint32),
  ('flags', ctypes.c_uint32),
  ('size', ctypes.c_uint64),
  ('mmapsize', ctypes.c_uint64),
  ('useraddr', ctypes.c_uint64),
  ('__pad', (ctypes.c_uint64 * 4)),
]
class struct_kgsl_gpumem_sync_cache(Struct): pass
struct_kgsl_gpumem_sync_cache._fields_ = [
  ('gpuaddr', ctypes.c_uint64),
  ('id', ctypes.c_uint32),
  ('op', ctypes.c_uint32),
  ('offset', ctypes.c_uint64),
  ('length', ctypes.c_uint64),
]
class struct_kgsl_perfcounter_get(Struct): pass
struct_kgsl_perfcounter_get._fields_ = [
  ('groupid', ctypes.c_uint32),
  ('countable', ctypes.c_uint32),
  ('offset', ctypes.c_uint32),
  ('offset_hi', ctypes.c_uint32),
  ('__pad', ctypes.c_uint32),
]
class struct_kgsl_perfcounter_put(Struct): pass
struct_kgsl_perfcounter_put._fields_ = [
  ('groupid', ctypes.c_uint32),
  ('countable', ctypes.c_uint32),
  ('__pad', (ctypes.c_uint32 * 2)),
]
class struct_kgsl_perfcounter_query(Struct): pass
struct_kgsl_perfcounter_query._fields_ = [
  ('groupid', ctypes.c_uint32),
  ('countables', ctypes.POINTER(ctypes.c_uint32)),
  ('count', ctypes.c_uint32),
  ('max_counters', ctypes.c_uint32),
  ('__pad', (ctypes.c_uint32 * 2)),
]
class struct_kgsl_perfcounter_read_group(Struct): pass
struct_kgsl_perfcounter_read_group._fields_ = [
  ('groupid', ctypes.c_uint32),
  ('countable', ctypes.c_uint32),
  ('value', ctypes.c_uint64),
]
class struct_kgsl_perfcounter_read(Struct): pass
struct_kgsl_perfcounter_read._fields_ = [
  ('reads', ctypes.POINTER(struct_kgsl_perfcounter_read_group)),
  ('count', ctypes.c_uint32),
  ('__pad', (ctypes.c_uint32 * 2)),
]
class struct_kgsl_gpumem_sync_cache_bulk(Struct): pass
struct_kgsl_gpumem_sync_cache_bulk._fields_ = [
  ('id_list', ctypes.POINTER(ctypes.c_uint32)),
  ('count', ctypes.c_uint32),
  ('op', ctypes.c_uint32),
  ('__pad', (ctypes.c_uint32 * 2)),
]
class struct_kgsl_cmd_syncpoint_timestamp(Struct): pass
struct_kgsl_cmd_syncpoint_timestamp._fields_ = [
  ('context_id', ctypes.c_uint32),
  ('timestamp', ctypes.c_uint32),
]
class struct_kgsl_cmd_syncpoint_fence(Struct): pass
struct_kgsl_cmd_syncpoint_fence._fields_ = [
  ('fd', ctypes.c_int32),
]
class struct_kgsl_cmd_syncpoint(Struct): pass
struct_kgsl_cmd_syncpoint._fields_ = [
  ('type', ctypes.c_int32),
  ('priv', ctypes.c_void_p),
  ('size', ctypes.c_uint64),
]
class struct_kgsl_submit_commands(Struct): pass
struct_kgsl_submit_commands._fields_ = [
  ('context_id', ctypes.c_uint32),
  ('flags', ctypes.c_uint32),
  ('cmdlist', ctypes.POINTER(struct_kgsl_ibdesc)),
  ('numcmds', ctypes.c_uint32),
  ('synclist', ctypes.POINTER(struct_kgsl_cmd_syncpoint)),
  ('numsyncs', ctypes.c_uint32),
  ('timestamp', ctypes.c_uint32),
  ('__pad', (ctypes.c_uint32 * 4)),
]
class struct_kgsl_device_constraint(Struct): pass
struct_kgsl_device_constraint._fields_ = [
  ('type', ctypes.c_uint32),
  ('context_id', ctypes.c_uint32),
  ('data', ctypes.c_void_p),
  ('size', ctypes.c_uint64),
]
class struct_kgsl_device_constraint_pwrlevel(Struct): pass
struct_kgsl_device_constraint_pwrlevel._fields_ = [
  ('level', ctypes.c_uint32),
]
class struct_kgsl_syncsource_create(Struct): pass
struct_kgsl_syncsource_create._fields_ = [
  ('id', ctypes.c_uint32),
  ('__pad', (ctypes.c_uint32 * 3)),
]
class struct_kgsl_syncsource_destroy(Struct): pass
struct_kgsl_syncsource_destroy._fields_ = [
  ('id', ctypes.c_uint32),
  ('__pad', (ctypes.c_uint32 * 3)),
]
class struct_kgsl_syncsource_create_fence(Struct): pass
struct_kgsl_syncsource_create_fence._fields_ = [
  ('id', ctypes.c_uint32),
  ('fence_fd', ctypes.c_int32),
  ('__pad', (ctypes.c_uint32 * 4)),
]
class struct_kgsl_syncsource_signal_fence(Struct): pass
struct_kgsl_syncsource_signal_fence._fields_ = [
  ('id', ctypes.c_uint32),
  ('fence_fd', ctypes.c_int32),
  ('__pad', (ctypes.c_uint32 * 4)),
]
class struct_kgsl_cff_sync_gpuobj(Struct): pass
struct_kgsl_cff_sync_gpuobj._fields_ = [
  ('offset', ctypes.c_uint64),
  ('length', ctypes.c_uint64),
  ('id', ctypes.c_uint32),
]
class struct_kgsl_gpuobj_alloc(Struct): pass
struct_kgsl_gpuobj_alloc._fields_ = [
  ('size', ctypes.c_uint64),
  ('flags', ctypes.c_uint64),
  ('va_len', ctypes.c_uint64),
  ('mmapsize', ctypes.c_uint64),
  ('id', ctypes.c_uint32),
  ('metadata_len', ctypes.c_uint32),
  ('metadata', ctypes.c_uint64),
]
class struct_kgsl_gpuobj_free(Struct): pass
struct_kgsl_gpuobj_free._fields_ = [
  ('flags', ctypes.c_uint64),
  ('priv', ctypes.c_uint64),
  ('id', ctypes.c_uint32),
  ('type', ctypes.c_uint32),
  ('len', ctypes.c_uint32),
]
class struct_kgsl_gpu_event_timestamp(Struct): pass
struct_kgsl_gpu_event_timestamp._fields_ = [
  ('context_id', ctypes.c_uint32),
  ('timestamp', ctypes.c_uint32),
]
class struct_kgsl_gpu_event_fence(Struct): pass
struct_kgsl_gpu_event_fence._fields_ = [
  ('fd', ctypes.c_int32),
]
class struct_kgsl_gpuobj_info(Struct): pass
struct_kgsl_gpuobj_info._fields_ = [
  ('gpuaddr', ctypes.c_uint64),
  ('flags', ctypes.c_uint64),
  ('size', ctypes.c_uint64),
  ('va_len', ctypes.c_uint64),
  ('va_addr', ctypes.c_uint64),
  ('id', ctypes.c_uint32),
]
class struct_kgsl_gpuobj_import(Struct): pass
struct_kgsl_gpuobj_import._fields_ = [
  ('priv', ctypes.c_uint64),
  ('priv_len', ctypes.c_uint64),
  ('flags', ctypes.c_uint64),
  ('type', ctypes.c_uint32),
  ('id', ctypes.c_uint32),
]
class struct_kgsl_gpuobj_import_dma_buf(Struct): pass
struct_kgsl_gpuobj_import_dma_buf._fields_ = [
  ('fd', ctypes.c_int32),
]
class struct_kgsl_gpuobj_import_useraddr(Struct): pass
struct_kgsl_gpuobj_import_useraddr._fields_ = [
  ('virtaddr', ctypes.c_uint64),
]
class struct_kgsl_gpuobj_sync_obj(Struct): pass
struct_kgsl_gpuobj_sync_obj._fields_ = [
  ('offset', ctypes.c_uint64),
  ('length', ctypes.c_uint64),
  ('id', ctypes.c_uint32),
  ('op', ctypes.c_uint32),
]
class struct_kgsl_gpuobj_sync(Struct): pass
struct_kgsl_gpuobj_sync._fields_ = [
  ('objs', ctypes.c_uint64),
  ('obj_len', ctypes.c_uint32),
  ('count', ctypes.c_uint32),
]
class struct_kgsl_command_object(Struct): pass
struct_kgsl_command_object._fields_ = [
  ('offset', ctypes.c_uint64),
  ('gpuaddr', ctypes.c_uint64),
  ('size', ctypes.c_uint64),
  ('flags', ctypes.c_uint32),
  ('id', ctypes.c_uint32),
]
class struct_kgsl_command_syncpoint(Struct): pass
struct_kgsl_command_syncpoint._fields_ = [
  ('priv', ctypes.c_uint64),
  ('size', ctypes.c_uint64),
  ('type', ctypes.c_uint32),
]
class struct_kgsl_gpu_command(Struct): pass
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
class struct_kgsl_preemption_counters_query(Struct): pass
struct_kgsl_preemption_counters_query._fields_ = [
  ('counters', ctypes.c_uint64),
  ('size_user', ctypes.c_uint32),
  ('size_priority_level', ctypes.c_uint32),
  ('max_priority_level', ctypes.c_uint32),
]
class struct_kgsl_gpuobj_set_info(Struct): pass
struct_kgsl_gpuobj_set_info._fields_ = [
  ('flags', ctypes.c_uint64),
  ('metadata', ctypes.c_uint64),
  ('id', ctypes.c_uint32),
  ('metadata_len', ctypes.c_uint32),
  ('type', ctypes.c_uint32),
]
KGSL_VERSION_MAJOR = 3
KGSL_VERSION_MINOR = 14
KGSL_CONTEXT_SAVE_GMEM = 0x00000001
KGSL_CONTEXT_NO_GMEM_ALLOC = 0x00000002
KGSL_CONTEXT_SUBMIT_IB_LIST = 0x00000004
KGSL_CONTEXT_CTX_SWITCH = 0x00000008
KGSL_CONTEXT_PREAMBLE = 0x00000010
KGSL_CONTEXT_TRASH_STATE = 0x00000020
KGSL_CONTEXT_PER_CONTEXT_TS = 0x00000040
KGSL_CONTEXT_USER_GENERATED_TS = 0x00000080
KGSL_CONTEXT_END_OF_FRAME = 0x00000100
KGSL_CONTEXT_NO_FAULT_TOLERANCE = 0x00000200
KGSL_CONTEXT_SYNC = 0x00000400
KGSL_CONTEXT_PWR_CONSTRAINT = 0x00000800
KGSL_CONTEXT_PRIORITY_MASK = 0x0000F000
KGSL_CONTEXT_PRIORITY_SHIFT = 12
KGSL_CONTEXT_PRIORITY_UNDEF = 0
KGSL_CONTEXT_IFH_NOP = 0x00010000
KGSL_CONTEXT_SECURE = 0x00020000
KGSL_CONTEXT_PREEMPT_STYLE_MASK = 0x0E000000
KGSL_CONTEXT_PREEMPT_STYLE_SHIFT = 25
KGSL_CONTEXT_PREEMPT_STYLE_DEFAULT = 0x0
KGSL_CONTEXT_PREEMPT_STYLE_RINGBUFFER = 0x1
KGSL_CONTEXT_PREEMPT_STYLE_FINEGRAIN = 0x2
KGSL_CONTEXT_TYPE_MASK = 0x01F00000
KGSL_CONTEXT_TYPE_SHIFT = 20
KGSL_CONTEXT_TYPE_ANY = 0
KGSL_CONTEXT_TYPE_GL = 1
KGSL_CONTEXT_TYPE_CL = 2
KGSL_CONTEXT_TYPE_C2D = 3
KGSL_CONTEXT_TYPE_RS = 4
KGSL_CONTEXT_TYPE_UNKNOWN = 0x1E
KGSL_CONTEXT_INVALID = 0xffffffff
KGSL_CMDBATCH_MEMLIST = 0x00000001
KGSL_CMDBATCH_MARKER = 0x00000002
KGSL_CMDBATCH_SUBMIT_IB_LIST = KGSL_CONTEXT_SUBMIT_IB_LIST
KGSL_CMDBATCH_CTX_SWITCH = KGSL_CONTEXT_CTX_SWITCH
KGSL_CMDBATCH_PROFILING = 0x00000010
KGSL_CMDBATCH_PROFILING_KTIME = 0x00000020
KGSL_CMDBATCH_END_OF_FRAME = KGSL_CONTEXT_END_OF_FRAME
KGSL_CMDBATCH_SYNC = KGSL_CONTEXT_SYNC
KGSL_CMDBATCH_PWR_CONSTRAINT = KGSL_CONTEXT_PWR_CONSTRAINT
KGSL_CMDLIST_IB = 0x00000001
KGSL_CMDLIST_CTXTSWITCH_PREAMBLE = 0x00000002
KGSL_CMDLIST_IB_PREAMBLE = 0x00000004
KGSL_OBJLIST_MEMOBJ = 0x00000008
KGSL_OBJLIST_PROFILE = 0x00000010
KGSL_CMD_SYNCPOINT_TYPE_TIMESTAMP = 0
KGSL_CMD_SYNCPOINT_TYPE_FENCE = 1
KGSL_MEMFLAGS_SECURE = 0x00000008
KGSL_MEMFLAGS_GPUREADONLY = 0x01000000
KGSL_MEMFLAGS_GPUWRITEONLY = 0x02000000
KGSL_MEMFLAGS_FORCE_32BIT = 0x100000000
KGSL_CACHEMODE_MASK = 0x0C000000
KGSL_CACHEMODE_SHIFT = 26
KGSL_CACHEMODE_WRITECOMBINE = 0
KGSL_CACHEMODE_UNCACHED = 1
KGSL_CACHEMODE_WRITETHROUGH = 2
KGSL_CACHEMODE_WRITEBACK = 3
KGSL_MEMFLAGS_USE_CPU_MAP = 0x10000000
KGSL_MEMTYPE_MASK = 0x0000FF00
KGSL_MEMTYPE_SHIFT = 8
KGSL_MEMTYPE_OBJECTANY = 0
KGSL_MEMTYPE_FRAMEBUFFER = 1
KGSL_MEMTYPE_RENDERBUFFER = 2
KGSL_MEMTYPE_ARRAYBUFFER = 3
KGSL_MEMTYPE_ELEMENTARRAYBUFFER = 4
KGSL_MEMTYPE_VERTEXARRAYBUFFER = 5
KGSL_MEMTYPE_TEXTURE = 6
KGSL_MEMTYPE_SURFACE = 7
KGSL_MEMTYPE_EGL_SURFACE = 8
KGSL_MEMTYPE_GL = 9
KGSL_MEMTYPE_CL = 10
KGSL_MEMTYPE_CL_BUFFER_MAP = 11
KGSL_MEMTYPE_CL_BUFFER_NOMAP = 12
KGSL_MEMTYPE_CL_IMAGE_MAP = 13
KGSL_MEMTYPE_CL_IMAGE_NOMAP = 14
KGSL_MEMTYPE_CL_KERNEL_STACK = 15
KGSL_MEMTYPE_COMMAND = 16
KGSL_MEMTYPE_2D = 17
KGSL_MEMTYPE_EGL_IMAGE = 18
KGSL_MEMTYPE_EGL_SHADOW = 19
KGSL_MEMTYPE_MULTISAMPLE = 20
KGSL_MEMTYPE_KERNEL = 255
KGSL_MEMALIGN_MASK = 0x00FF0000
KGSL_MEMALIGN_SHIFT = 16
KGSL_MEMFLAGS_USERMEM_MASK = 0x000000e0
KGSL_MEMFLAGS_USERMEM_SHIFT = 5
KGSL_USERMEM_FLAG = lambda x: (((x) + 1) << KGSL_MEMFLAGS_USERMEM_SHIFT)
KGSL_MEMFLAGS_NOT_USERMEM = 0
KGSL_MEMFLAGS_USERMEM_PMEM = KGSL_USERMEM_FLAG(KGSL_USER_MEM_TYPE_PMEM)
KGSL_MEMFLAGS_USERMEM_ASHMEM = KGSL_USERMEM_FLAG(KGSL_USER_MEM_TYPE_ASHMEM)
KGSL_MEMFLAGS_USERMEM_ADDR = KGSL_USERMEM_FLAG(KGSL_USER_MEM_TYPE_ADDR)
KGSL_MEMFLAGS_USERMEM_ION = KGSL_USERMEM_FLAG(KGSL_USER_MEM_TYPE_ION)
KGSL_FLAGS_NORMALMODE = 0x00000000
KGSL_FLAGS_SAFEMODE = 0x00000001
KGSL_FLAGS_INITIALIZED0 = 0x00000002
KGSL_FLAGS_INITIALIZED = 0x00000004
KGSL_FLAGS_STARTED = 0x00000008
KGSL_FLAGS_ACTIVE = 0x00000010
KGSL_FLAGS_RESERVED0 = 0x00000020
KGSL_FLAGS_RESERVED1 = 0x00000040
KGSL_FLAGS_RESERVED2 = 0x00000080
KGSL_FLAGS_SOFT_RESET = 0x00000100
KGSL_FLAGS_PER_CONTEXT_TIMESTAMPS = 0x00000200
KGSL_SYNCOBJ_SERVER_TIMEOUT = 2000
KGSL_CONVERT_TO_MBPS = lambda val: (val*1000*1000)
KGSL_MEMSTORE_OFFSET = lambda ctxt_id,field: ((ctxt_id)*sizeof(struct_kgsl_devmemstore) + offsetof(struct_kgsl_devmemstore, field))
KGSL_PROP_DEVICE_INFO = 0x1
KGSL_PROP_DEVICE_SHADOW = 0x2
KGSL_PROP_DEVICE_POWER = 0x3
KGSL_PROP_SHMEM = 0x4
KGSL_PROP_SHMEM_APERTURES = 0x5
KGSL_PROP_MMU_ENABLE = 0x6
KGSL_PROP_INTERRUPT_WAITS = 0x7
KGSL_PROP_VERSION = 0x8
KGSL_PROP_GPU_RESET_STAT = 0x9
KGSL_PROP_PWRCTRL = 0xE
KGSL_PROP_PWR_CONSTRAINT = 0x12
KGSL_PROP_UCHE_GMEM_VADDR = 0x13
KGSL_PROP_SP_GENERIC_MEM = 0x14
KGSL_PROP_UCODE_VERSION = 0x15
KGSL_PROP_GPMU_VERSION = 0x16
KGSL_PROP_DEVICE_BITNESS = 0x18
KGSL_PERFCOUNTER_GROUP_CP = 0x0
KGSL_PERFCOUNTER_GROUP_RBBM = 0x1
KGSL_PERFCOUNTER_GROUP_PC = 0x2
KGSL_PERFCOUNTER_GROUP_VFD = 0x3
KGSL_PERFCOUNTER_GROUP_HLSQ = 0x4
KGSL_PERFCOUNTER_GROUP_VPC = 0x5
KGSL_PERFCOUNTER_GROUP_TSE = 0x6
KGSL_PERFCOUNTER_GROUP_RAS = 0x7
KGSL_PERFCOUNTER_GROUP_UCHE = 0x8
KGSL_PERFCOUNTER_GROUP_TP = 0x9
KGSL_PERFCOUNTER_GROUP_SP = 0xA
KGSL_PERFCOUNTER_GROUP_RB = 0xB
KGSL_PERFCOUNTER_GROUP_PWR = 0xC
KGSL_PERFCOUNTER_GROUP_VBIF = 0xD
KGSL_PERFCOUNTER_GROUP_VBIF_PWR = 0xE
KGSL_PERFCOUNTER_GROUP_MH = 0xF
KGSL_PERFCOUNTER_GROUP_PA_SU = 0x10
KGSL_PERFCOUNTER_GROUP_SQ = 0x11
KGSL_PERFCOUNTER_GROUP_SX = 0x12
KGSL_PERFCOUNTER_GROUP_TCF = 0x13
KGSL_PERFCOUNTER_GROUP_TCM = 0x14
KGSL_PERFCOUNTER_GROUP_TCR = 0x15
KGSL_PERFCOUNTER_GROUP_L2 = 0x16
KGSL_PERFCOUNTER_GROUP_VSC = 0x17
KGSL_PERFCOUNTER_GROUP_CCU = 0x18
KGSL_PERFCOUNTER_GROUP_LRZ = 0x19
KGSL_PERFCOUNTER_GROUP_CMP = 0x1A
KGSL_PERFCOUNTER_GROUP_ALWAYSON = 0x1B
KGSL_PERFCOUNTER_GROUP_SP_PWR = 0x1C
KGSL_PERFCOUNTER_GROUP_TP_PWR = 0x1D
KGSL_PERFCOUNTER_GROUP_RB_PWR = 0x1E
KGSL_PERFCOUNTER_GROUP_CCU_PWR = 0x1F
KGSL_PERFCOUNTER_GROUP_UCHE_PWR = 0x20
KGSL_PERFCOUNTER_GROUP_CP_PWR = 0x21
KGSL_PERFCOUNTER_GROUP_GPMU_PWR = 0x22
KGSL_PERFCOUNTER_GROUP_ALWAYSON_PWR = 0x23
KGSL_PERFCOUNTER_GROUP_MAX = 0x24
KGSL_PERFCOUNTER_NOT_USED = 0xFFFFFFFF
KGSL_PERFCOUNTER_BROKEN = 0xFFFFFFFE
KGSL_IOC_TYPE = 0x09
IOCTL_KGSL_DEVICE_GETPROPERTY = _IOWR(KGSL_IOC_TYPE, 0x2, struct_kgsl_device_getproperty)
IOCTL_KGSL_DEVICE_WAITTIMESTAMP = _IOW(KGSL_IOC_TYPE, 0x6, struct_kgsl_device_waittimestamp)
IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID = _IOW(KGSL_IOC_TYPE, 0x7, struct_kgsl_device_waittimestamp_ctxtid)
IOCTL_KGSL_RINGBUFFER_ISSUEIBCMDS = _IOWR(KGSL_IOC_TYPE, 0x10, struct_kgsl_ringbuffer_issueibcmds)
IOCTL_KGSL_CMDSTREAM_READTIMESTAMP_OLD = _IOR(KGSL_IOC_TYPE, 0x11, struct_kgsl_cmdstream_readtimestamp)
IOCTL_KGSL_CMDSTREAM_READTIMESTAMP = _IOWR(KGSL_IOC_TYPE, 0x11, struct_kgsl_cmdstream_readtimestamp)
IOCTL_KGSL_CMDSTREAM_FREEMEMONTIMESTAMP = _IOW(KGSL_IOC_TYPE, 0x12, struct_kgsl_cmdstream_freememontimestamp)
IOCTL_KGSL_CMDSTREAM_FREEMEMONTIMESTAMP_OLD = _IOR(KGSL_IOC_TYPE, 0x12, struct_kgsl_cmdstream_freememontimestamp)
IOCTL_KGSL_DRAWCTXT_CREATE = _IOWR(KGSL_IOC_TYPE, 0x13, struct_kgsl_drawctxt_create)
IOCTL_KGSL_DRAWCTXT_DESTROY = _IOW(KGSL_IOC_TYPE, 0x14, struct_kgsl_drawctxt_destroy)
IOCTL_KGSL_MAP_USER_MEM = _IOWR(KGSL_IOC_TYPE, 0x15, struct_kgsl_map_user_mem)
IOCTL_KGSL_CMDSTREAM_READTIMESTAMP_CTXTID = _IOWR(KGSL_IOC_TYPE, 0x16, struct_kgsl_cmdstream_readtimestamp_ctxtid)
IOCTL_KGSL_CMDSTREAM_FREEMEMONTIMESTAMP_CTXTID = _IOW(KGSL_IOC_TYPE, 0x17, struct_kgsl_cmdstream_freememontimestamp_ctxtid)
IOCTL_KGSL_SHAREDMEM_FROM_PMEM = _IOWR(KGSL_IOC_TYPE, 0x20, struct_kgsl_sharedmem_from_pmem)
IOCTL_KGSL_SHAREDMEM_FREE = _IOW(KGSL_IOC_TYPE, 0x21, struct_kgsl_sharedmem_free)
IOCTL_KGSL_CFF_USER_EVENT = _IOW(KGSL_IOC_TYPE, 0x31, struct_kgsl_cff_user_event)
IOCTL_KGSL_DRAWCTXT_BIND_GMEM_SHADOW = _IOW(KGSL_IOC_TYPE, 0x22, struct_kgsl_bind_gmem_shadow)
IOCTL_KGSL_SHAREDMEM_FROM_VMALLOC = _IOWR(KGSL_IOC_TYPE, 0x23, struct_kgsl_sharedmem_from_vmalloc)
IOCTL_KGSL_SHAREDMEM_FLUSH_CACHE = _IOW(KGSL_IOC_TYPE, 0x24, struct_kgsl_sharedmem_free)
IOCTL_KGSL_DRAWCTXT_SET_BIN_BASE_OFFSET = _IOW(KGSL_IOC_TYPE, 0x25, struct_kgsl_drawctxt_set_bin_base_offset)
IOCTL_KGSL_CMDWINDOW_WRITE = _IOW(KGSL_IOC_TYPE, 0x2e, struct_kgsl_cmdwindow_write)
IOCTL_KGSL_GPUMEM_ALLOC = _IOWR(KGSL_IOC_TYPE, 0x2f, struct_kgsl_gpumem_alloc)
IOCTL_KGSL_CFF_SYNCMEM = _IOW(KGSL_IOC_TYPE, 0x30, struct_kgsl_cff_syncmem)
IOCTL_KGSL_TIMESTAMP_EVENT_OLD = _IOW(KGSL_IOC_TYPE, 0x31, struct_kgsl_timestamp_event)
KGSL_TIMESTAMP_EVENT_GENLOCK = 1
KGSL_TIMESTAMP_EVENT_FENCE = 2
IOCTL_KGSL_SETPROPERTY = _IOW(KGSL_IOC_TYPE, 0x32, struct_kgsl_device_getproperty)
IOCTL_KGSL_TIMESTAMP_EVENT = _IOWR(KGSL_IOC_TYPE, 0x33, struct_kgsl_timestamp_event)
IOCTL_KGSL_GPUMEM_ALLOC_ID = _IOWR(KGSL_IOC_TYPE, 0x34, struct_kgsl_gpumem_alloc_id)
IOCTL_KGSL_GPUMEM_FREE_ID = _IOWR(KGSL_IOC_TYPE, 0x35, struct_kgsl_gpumem_free_id)
IOCTL_KGSL_GPUMEM_GET_INFO = _IOWR(KGSL_IOC_TYPE, 0x36, struct_kgsl_gpumem_get_info)
KGSL_GPUMEM_CACHE_CLEAN = (1 << 0)
KGSL_GPUMEM_CACHE_TO_GPU = KGSL_GPUMEM_CACHE_CLEAN
KGSL_GPUMEM_CACHE_INV = (1 << 1)
KGSL_GPUMEM_CACHE_FROM_GPU = KGSL_GPUMEM_CACHE_INV
KGSL_GPUMEM_CACHE_FLUSH = (KGSL_GPUMEM_CACHE_CLEAN | KGSL_GPUMEM_CACHE_INV)
KGSL_GPUMEM_CACHE_RANGE = (1 << 31)
IOCTL_KGSL_GPUMEM_SYNC_CACHE = _IOW(KGSL_IOC_TYPE, 0x37, struct_kgsl_gpumem_sync_cache)
IOCTL_KGSL_PERFCOUNTER_GET = _IOWR(KGSL_IOC_TYPE, 0x38, struct_kgsl_perfcounter_get)
IOCTL_KGSL_PERFCOUNTER_PUT = _IOW(KGSL_IOC_TYPE, 0x39, struct_kgsl_perfcounter_put)
IOCTL_KGSL_PERFCOUNTER_QUERY = _IOWR(KGSL_IOC_TYPE, 0x3A, struct_kgsl_perfcounter_query)
IOCTL_KGSL_PERFCOUNTER_READ = _IOWR(KGSL_IOC_TYPE, 0x3B, struct_kgsl_perfcounter_read)
IOCTL_KGSL_GPUMEM_SYNC_CACHE_BULK = _IOWR(KGSL_IOC_TYPE, 0x3C, struct_kgsl_gpumem_sync_cache_bulk)
KGSL_IBDESC_MEMLIST = 0x1
KGSL_IBDESC_PROFILING_BUFFER = 0x2
IOCTL_KGSL_SUBMIT_COMMANDS = _IOWR(KGSL_IOC_TYPE, 0x3D, struct_kgsl_submit_commands)
KGSL_CONSTRAINT_NONE = 0
KGSL_CONSTRAINT_PWRLEVEL = 1
KGSL_CONSTRAINT_PWR_MIN = 0
KGSL_CONSTRAINT_PWR_MAX = 1
IOCTL_KGSL_SYNCSOURCE_CREATE = _IOWR(KGSL_IOC_TYPE, 0x40, struct_kgsl_syncsource_create)
IOCTL_KGSL_SYNCSOURCE_DESTROY = _IOWR(KGSL_IOC_TYPE, 0x41, struct_kgsl_syncsource_destroy)
IOCTL_KGSL_SYNCSOURCE_CREATE_FENCE = _IOWR(KGSL_IOC_TYPE, 0x42, struct_kgsl_syncsource_create_fence)
IOCTL_KGSL_SYNCSOURCE_SIGNAL_FENCE = _IOWR(KGSL_IOC_TYPE, 0x43, struct_kgsl_syncsource_signal_fence)
IOCTL_KGSL_CFF_SYNC_GPUOBJ = _IOW(KGSL_IOC_TYPE, 0x44, struct_kgsl_cff_sync_gpuobj)
KGSL_GPUOBJ_ALLOC_METADATA_MAX = 64
IOCTL_KGSL_GPUOBJ_ALLOC = _IOWR(KGSL_IOC_TYPE, 0x45, struct_kgsl_gpuobj_alloc)
KGSL_GPUOBJ_FREE_ON_EVENT = 1
KGSL_GPU_EVENT_TIMESTAMP = 1
KGSL_GPU_EVENT_FENCE = 2
IOCTL_KGSL_GPUOBJ_FREE = _IOW(KGSL_IOC_TYPE, 0x46, struct_kgsl_gpuobj_free)
IOCTL_KGSL_GPUOBJ_INFO = _IOWR(KGSL_IOC_TYPE, 0x47, struct_kgsl_gpuobj_info)
IOCTL_KGSL_GPUOBJ_IMPORT = _IOWR(KGSL_IOC_TYPE, 0x48, struct_kgsl_gpuobj_import)
IOCTL_KGSL_GPUOBJ_SYNC = _IOW(KGSL_IOC_TYPE, 0x49, struct_kgsl_gpuobj_sync)
IOCTL_KGSL_GPU_COMMAND = _IOWR(KGSL_IOC_TYPE, 0x4A, struct_kgsl_gpu_command)
IOCTL_KGSL_PREEMPTIONCOUNTER_QUERY = _IOWR(KGSL_IOC_TYPE, 0x4B, struct_kgsl_preemption_counters_query)
KGSL_GPUOBJ_SET_INFO_METADATA = (1 << 0)
KGSL_GPUOBJ_SET_INFO_TYPE = (1 << 1)
IOCTL_KGSL_GPUOBJ_SET_INFO = _IOW(KGSL_IOC_TYPE, 0x4C, struct_kgsl_gpuobj_set_info)