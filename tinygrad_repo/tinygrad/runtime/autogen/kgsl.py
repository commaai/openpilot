# mypy: disable-error-code="empty-body"
from __future__ import annotations
import ctypes
from typing import Literal, TypeAlias
from tinygrad.runtime.support.c import _IO, _IOW, _IOR, _IOWR
from tinygrad.runtime.support import c
enum_kgsl_user_mem_type: dict[int, str] = {(KGSL_USER_MEM_TYPE_PMEM:=0): 'KGSL_USER_MEM_TYPE_PMEM', (KGSL_USER_MEM_TYPE_ASHMEM:=1): 'KGSL_USER_MEM_TYPE_ASHMEM', (KGSL_USER_MEM_TYPE_ADDR:=2): 'KGSL_USER_MEM_TYPE_ADDR', (KGSL_USER_MEM_TYPE_ION:=3): 'KGSL_USER_MEM_TYPE_ION', (KGSL_USER_MEM_TYPE_DMABUF:=3): 'KGSL_USER_MEM_TYPE_DMABUF', (KGSL_USER_MEM_TYPE_MAX:=7): 'KGSL_USER_MEM_TYPE_MAX'}
enum_kgsl_ctx_reset_stat: dict[int, str] = {(KGSL_CTX_STAT_NO_ERROR:=0): 'KGSL_CTX_STAT_NO_ERROR', (KGSL_CTX_STAT_GUILTY_CONTEXT_RESET_EXT:=1): 'KGSL_CTX_STAT_GUILTY_CONTEXT_RESET_EXT', (KGSL_CTX_STAT_INNOCENT_CONTEXT_RESET_EXT:=2): 'KGSL_CTX_STAT_INNOCENT_CONTEXT_RESET_EXT', (KGSL_CTX_STAT_UNKNOWN_CONTEXT_RESET_EXT:=3): 'KGSL_CTX_STAT_UNKNOWN_CONTEXT_RESET_EXT'}
enum_kgsl_deviceid: dict[int, str] = {(KGSL_DEVICE_3D0:=0): 'KGSL_DEVICE_3D0', (KGSL_DEVICE_MAX:=1): 'KGSL_DEVICE_MAX'}
@c.record
class struct_kgsl_devinfo(c.Struct):
  SIZE = 40
  device_id: int
  chip_id: int
  mmu_enabled: int
  gmem_gpubaseaddr: int
  gpu_id: int
  gmem_sizebytes: int
struct_kgsl_devinfo.register_fields([('device_id', ctypes.c_uint32, 0), ('chip_id', ctypes.c_uint32, 4), ('mmu_enabled', ctypes.c_uint32, 8), ('gmem_gpubaseaddr', ctypes.c_uint64, 16), ('gpu_id', ctypes.c_uint32, 24), ('gmem_sizebytes', ctypes.c_uint64, 32)])
@c.record
class struct_kgsl_devmemstore(c.Struct):
  SIZE = 40
  soptimestamp: int
  sbz: int
  eoptimestamp: int
  sbz2: int
  preempted: int
  sbz3: int
  ref_wait_ts: int
  sbz4: int
  current_context: int
  sbz5: int
struct_kgsl_devmemstore.register_fields([('soptimestamp', ctypes.c_uint32, 0), ('sbz', ctypes.c_uint32, 4), ('eoptimestamp', ctypes.c_uint32, 8), ('sbz2', ctypes.c_uint32, 12), ('preempted', ctypes.c_uint32, 16), ('sbz3', ctypes.c_uint32, 20), ('ref_wait_ts', ctypes.c_uint32, 24), ('sbz4', ctypes.c_uint32, 28), ('current_context', ctypes.c_uint32, 32), ('sbz5', ctypes.c_uint32, 36)])
enum_kgsl_timestamp_type: dict[int, str] = {(KGSL_TIMESTAMP_CONSUMED:=1): 'KGSL_TIMESTAMP_CONSUMED', (KGSL_TIMESTAMP_RETIRED:=2): 'KGSL_TIMESTAMP_RETIRED', (KGSL_TIMESTAMP_QUEUED:=3): 'KGSL_TIMESTAMP_QUEUED'}
@c.record
class struct_kgsl_shadowprop(c.Struct):
  SIZE = 24
  gpuaddr: int
  size: int
  flags: int
struct_kgsl_shadowprop.register_fields([('gpuaddr', ctypes.c_uint64, 0), ('size', ctypes.c_uint64, 8), ('flags', ctypes.c_uint32, 16)])
@c.record
class struct_kgsl_version(c.Struct):
  SIZE = 16
  drv_major: int
  drv_minor: int
  dev_major: int
  dev_minor: int
struct_kgsl_version.register_fields([('drv_major', ctypes.c_uint32, 0), ('drv_minor', ctypes.c_uint32, 4), ('dev_major', ctypes.c_uint32, 8), ('dev_minor', ctypes.c_uint32, 12)])
@c.record
class struct_kgsl_sp_generic_mem(c.Struct):
  SIZE = 16
  local: int
  pvt: int
struct_kgsl_sp_generic_mem.register_fields([('local', ctypes.c_uint64, 0), ('pvt', ctypes.c_uint64, 8)])
@c.record
class struct_kgsl_ucode_version(c.Struct):
  SIZE = 8
  pfp: int
  pm4: int
struct_kgsl_ucode_version.register_fields([('pfp', ctypes.c_uint32, 0), ('pm4', ctypes.c_uint32, 4)])
@c.record
class struct_kgsl_gpmu_version(c.Struct):
  SIZE = 12
  major: int
  minor: int
  features: int
struct_kgsl_gpmu_version.register_fields([('major', ctypes.c_uint32, 0), ('minor', ctypes.c_uint32, 4), ('features', ctypes.c_uint32, 8)])
@c.record
class struct_kgsl_ibdesc(c.Struct):
  SIZE = 32
  gpuaddr: int
  __pad: int
  sizedwords: int
  ctrl: int
struct_kgsl_ibdesc.register_fields([('gpuaddr', ctypes.c_uint64, 0), ('__pad', ctypes.c_uint64, 8), ('sizedwords', ctypes.c_uint64, 16), ('ctrl', ctypes.c_uint32, 24)])
@c.record
class struct_kgsl_cmdbatch_profiling_buffer(c.Struct):
  SIZE = 40
  wall_clock_s: int
  wall_clock_ns: int
  gpu_ticks_queued: int
  gpu_ticks_submitted: int
  gpu_ticks_retired: int
struct_kgsl_cmdbatch_profiling_buffer.register_fields([('wall_clock_s', ctypes.c_uint64, 0), ('wall_clock_ns', ctypes.c_uint64, 8), ('gpu_ticks_queued', ctypes.c_uint64, 16), ('gpu_ticks_submitted', ctypes.c_uint64, 24), ('gpu_ticks_retired', ctypes.c_uint64, 32)])
@c.record
class struct_kgsl_device_getproperty(c.Struct):
  SIZE = 24
  type: int
  value: ctypes.c_void_p
  sizebytes: int
struct_kgsl_device_getproperty.register_fields([('type', ctypes.c_uint32, 0), ('value', ctypes.c_void_p, 8), ('sizebytes', ctypes.c_uint64, 16)])
@c.record
class struct_kgsl_device_waittimestamp(c.Struct):
  SIZE = 8
  timestamp: int
  timeout: int
struct_kgsl_device_waittimestamp.register_fields([('timestamp', ctypes.c_uint32, 0), ('timeout', ctypes.c_uint32, 4)])
@c.record
class struct_kgsl_device_waittimestamp_ctxtid(c.Struct):
  SIZE = 12
  context_id: int
  timestamp: int
  timeout: int
struct_kgsl_device_waittimestamp_ctxtid.register_fields([('context_id', ctypes.c_uint32, 0), ('timestamp', ctypes.c_uint32, 4), ('timeout', ctypes.c_uint32, 8)])
@c.record
class struct_kgsl_ringbuffer_issueibcmds(c.Struct):
  SIZE = 32
  drawctxt_id: int
  ibdesc_addr: int
  numibs: int
  timestamp: int
  flags: int
struct_kgsl_ringbuffer_issueibcmds.register_fields([('drawctxt_id', ctypes.c_uint32, 0), ('ibdesc_addr', ctypes.c_uint64, 8), ('numibs', ctypes.c_uint32, 16), ('timestamp', ctypes.c_uint32, 20), ('flags', ctypes.c_uint32, 24)])
@c.record
class struct_kgsl_cmdstream_readtimestamp(c.Struct):
  SIZE = 8
  type: int
  timestamp: int
struct_kgsl_cmdstream_readtimestamp.register_fields([('type', ctypes.c_uint32, 0), ('timestamp', ctypes.c_uint32, 4)])
@c.record
class struct_kgsl_cmdstream_freememontimestamp(c.Struct):
  SIZE = 16
  gpuaddr: int
  type: int
  timestamp: int
struct_kgsl_cmdstream_freememontimestamp.register_fields([('gpuaddr', ctypes.c_uint64, 0), ('type', ctypes.c_uint32, 8), ('timestamp', ctypes.c_uint32, 12)])
@c.record
class struct_kgsl_drawctxt_create(c.Struct):
  SIZE = 8
  flags: int
  drawctxt_id: int
struct_kgsl_drawctxt_create.register_fields([('flags', ctypes.c_uint32, 0), ('drawctxt_id', ctypes.c_uint32, 4)])
@c.record
class struct_kgsl_drawctxt_destroy(c.Struct):
  SIZE = 4
  drawctxt_id: int
struct_kgsl_drawctxt_destroy.register_fields([('drawctxt_id', ctypes.c_uint32, 0)])
@c.record
class struct_kgsl_map_user_mem(c.Struct):
  SIZE = 48
  fd: int
  gpuaddr: int
  len: int
  offset: int
  hostptr: int
  memtype: int
  flags: int
struct_kgsl_map_user_mem.register_fields([('fd', ctypes.c_int32, 0), ('gpuaddr', ctypes.c_uint64, 8), ('len', ctypes.c_uint64, 16), ('offset', ctypes.c_uint64, 24), ('hostptr', ctypes.c_uint64, 32), ('memtype', ctypes.c_uint32, 40), ('flags', ctypes.c_uint32, 44)])
@c.record
class struct_kgsl_cmdstream_readtimestamp_ctxtid(c.Struct):
  SIZE = 12
  context_id: int
  type: int
  timestamp: int
struct_kgsl_cmdstream_readtimestamp_ctxtid.register_fields([('context_id', ctypes.c_uint32, 0), ('type', ctypes.c_uint32, 4), ('timestamp', ctypes.c_uint32, 8)])
@c.record
class struct_kgsl_cmdstream_freememontimestamp_ctxtid(c.Struct):
  SIZE = 24
  context_id: int
  gpuaddr: int
  type: int
  timestamp: int
struct_kgsl_cmdstream_freememontimestamp_ctxtid.register_fields([('context_id', ctypes.c_uint32, 0), ('gpuaddr', ctypes.c_uint64, 8), ('type', ctypes.c_uint32, 16), ('timestamp', ctypes.c_uint32, 20)])
@c.record
class struct_kgsl_sharedmem_from_pmem(c.Struct):
  SIZE = 24
  pmem_fd: int
  gpuaddr: int
  len: int
  offset: int
struct_kgsl_sharedmem_from_pmem.register_fields([('pmem_fd', ctypes.c_int32, 0), ('gpuaddr', ctypes.c_uint64, 8), ('len', ctypes.c_uint32, 16), ('offset', ctypes.c_uint32, 20)])
@c.record
class struct_kgsl_sharedmem_free(c.Struct):
  SIZE = 8
  gpuaddr: int
struct_kgsl_sharedmem_free.register_fields([('gpuaddr', ctypes.c_uint64, 0)])
@c.record
class struct_kgsl_cff_user_event(c.Struct):
  SIZE = 32
  cff_opcode: int
  op1: int
  op2: int
  op3: int
  op4: int
  op5: int
  __pad: c.Array[ctypes.c_uint32, Literal[2]]
struct_kgsl_cff_user_event.register_fields([('cff_opcode', ctypes.c_ubyte, 0), ('op1', ctypes.c_uint32, 4), ('op2', ctypes.c_uint32, 8), ('op3', ctypes.c_uint32, 12), ('op4', ctypes.c_uint32, 16), ('op5', ctypes.c_uint32, 20), ('__pad', c.Array[ctypes.c_uint32, Literal[2]], 24)])
@c.record
class struct_kgsl_gmem_desc(c.Struct):
  SIZE = 20
  x: int
  y: int
  width: int
  height: int
  pitch: int
struct_kgsl_gmem_desc.register_fields([('x', ctypes.c_uint32, 0), ('y', ctypes.c_uint32, 4), ('width', ctypes.c_uint32, 8), ('height', ctypes.c_uint32, 12), ('pitch', ctypes.c_uint32, 16)])
@c.record
class struct_kgsl_buffer_desc(c.Struct):
  SIZE = 32
  hostptr: ctypes.c_void_p
  gpuaddr: int
  size: int
  format: int
  pitch: int
  enabled: int
struct_kgsl_buffer_desc.register_fields([('hostptr', ctypes.c_void_p, 0), ('gpuaddr', ctypes.c_uint64, 8), ('size', ctypes.c_int32, 16), ('format', ctypes.c_uint32, 20), ('pitch', ctypes.c_uint32, 24), ('enabled', ctypes.c_uint32, 28)])
@c.record
class struct_kgsl_bind_gmem_shadow(c.Struct):
  SIZE = 72
  drawctxt_id: int
  gmem_desc: struct_kgsl_gmem_desc
  shadow_x: int
  shadow_y: int
  shadow_buffer: struct_kgsl_buffer_desc
  buffer_id: int
struct_kgsl_bind_gmem_shadow.register_fields([('drawctxt_id', ctypes.c_uint32, 0), ('gmem_desc', struct_kgsl_gmem_desc, 4), ('shadow_x', ctypes.c_uint32, 24), ('shadow_y', ctypes.c_uint32, 28), ('shadow_buffer', struct_kgsl_buffer_desc, 32), ('buffer_id', ctypes.c_uint32, 64)])
@c.record
class struct_kgsl_sharedmem_from_vmalloc(c.Struct):
  SIZE = 16
  gpuaddr: int
  hostptr: int
  flags: int
struct_kgsl_sharedmem_from_vmalloc.register_fields([('gpuaddr', ctypes.c_uint64, 0), ('hostptr', ctypes.c_uint32, 8), ('flags', ctypes.c_uint32, 12)])
@c.record
class struct_kgsl_drawctxt_set_bin_base_offset(c.Struct):
  SIZE = 8
  drawctxt_id: int
  offset: int
struct_kgsl_drawctxt_set_bin_base_offset.register_fields([('drawctxt_id', ctypes.c_uint32, 0), ('offset', ctypes.c_uint32, 4)])
enum_kgsl_cmdwindow_type: dict[int, str] = {(KGSL_CMDWINDOW_MIN:=0): 'KGSL_CMDWINDOW_MIN', (KGSL_CMDWINDOW_2D:=0): 'KGSL_CMDWINDOW_2D', (KGSL_CMDWINDOW_3D:=1): 'KGSL_CMDWINDOW_3D', (KGSL_CMDWINDOW_MMU:=2): 'KGSL_CMDWINDOW_MMU', (KGSL_CMDWINDOW_ARBITER:=255): 'KGSL_CMDWINDOW_ARBITER', (KGSL_CMDWINDOW_MAX:=255): 'KGSL_CMDWINDOW_MAX'}
@c.record
class struct_kgsl_cmdwindow_write(c.Struct):
  SIZE = 12
  target: int
  addr: int
  data: int
struct_kgsl_cmdwindow_write.register_fields([('target', ctypes.c_uint32, 0), ('addr', ctypes.c_uint32, 4), ('data', ctypes.c_uint32, 8)])
@c.record
class struct_kgsl_gpumem_alloc(c.Struct):
  SIZE = 24
  gpuaddr: int
  size: int
  flags: int
struct_kgsl_gpumem_alloc.register_fields([('gpuaddr', ctypes.c_uint64, 0), ('size', ctypes.c_uint64, 8), ('flags', ctypes.c_uint32, 16)])
@c.record
class struct_kgsl_cff_syncmem(c.Struct):
  SIZE = 24
  gpuaddr: int
  len: int
  __pad: c.Array[ctypes.c_uint32, Literal[2]]
struct_kgsl_cff_syncmem.register_fields([('gpuaddr', ctypes.c_uint64, 0), ('len', ctypes.c_uint64, 8), ('__pad', c.Array[ctypes.c_uint32, Literal[2]], 16)])
@c.record
class struct_kgsl_timestamp_event(c.Struct):
  SIZE = 32
  type: int
  timestamp: int
  context_id: int
  priv: ctypes.c_void_p
  len: int
struct_kgsl_timestamp_event.register_fields([('type', ctypes.c_int32, 0), ('timestamp', ctypes.c_uint32, 4), ('context_id', ctypes.c_uint32, 8), ('priv', ctypes.c_void_p, 16), ('len', ctypes.c_uint64, 24)])
@c.record
class struct_kgsl_timestamp_event_genlock(c.Struct):
  SIZE = 4
  handle: int
struct_kgsl_timestamp_event_genlock.register_fields([('handle', ctypes.c_int32, 0)])
@c.record
class struct_kgsl_timestamp_event_fence(c.Struct):
  SIZE = 4
  fence_fd: int
struct_kgsl_timestamp_event_fence.register_fields([('fence_fd', ctypes.c_int32, 0)])
@c.record
class struct_kgsl_gpumem_alloc_id(c.Struct):
  SIZE = 48
  id: int
  flags: int
  size: int
  mmapsize: int
  gpuaddr: int
  __pad: c.Array[ctypes.c_uint64, Literal[2]]
struct_kgsl_gpumem_alloc_id.register_fields([('id', ctypes.c_uint32, 0), ('flags', ctypes.c_uint32, 4), ('size', ctypes.c_uint64, 8), ('mmapsize', ctypes.c_uint64, 16), ('gpuaddr', ctypes.c_uint64, 24), ('__pad', c.Array[ctypes.c_uint64, Literal[2]], 32)])
@c.record
class struct_kgsl_gpumem_free_id(c.Struct):
  SIZE = 8
  id: int
  __pad: int
struct_kgsl_gpumem_free_id.register_fields([('id', ctypes.c_uint32, 0), ('__pad', ctypes.c_uint32, 4)])
@c.record
class struct_kgsl_gpumem_get_info(c.Struct):
  SIZE = 72
  gpuaddr: int
  id: int
  flags: int
  size: int
  mmapsize: int
  useraddr: int
  __pad: c.Array[ctypes.c_uint64, Literal[4]]
struct_kgsl_gpumem_get_info.register_fields([('gpuaddr', ctypes.c_uint64, 0), ('id', ctypes.c_uint32, 8), ('flags', ctypes.c_uint32, 12), ('size', ctypes.c_uint64, 16), ('mmapsize', ctypes.c_uint64, 24), ('useraddr', ctypes.c_uint64, 32), ('__pad', c.Array[ctypes.c_uint64, Literal[4]], 40)])
@c.record
class struct_kgsl_gpumem_sync_cache(c.Struct):
  SIZE = 32
  gpuaddr: int
  id: int
  op: int
  offset: int
  length: int
struct_kgsl_gpumem_sync_cache.register_fields([('gpuaddr', ctypes.c_uint64, 0), ('id', ctypes.c_uint32, 8), ('op', ctypes.c_uint32, 12), ('offset', ctypes.c_uint64, 16), ('length', ctypes.c_uint64, 24)])
@c.record
class struct_kgsl_perfcounter_get(c.Struct):
  SIZE = 20
  groupid: int
  countable: int
  offset: int
  offset_hi: int
  __pad: int
struct_kgsl_perfcounter_get.register_fields([('groupid', ctypes.c_uint32, 0), ('countable', ctypes.c_uint32, 4), ('offset', ctypes.c_uint32, 8), ('offset_hi', ctypes.c_uint32, 12), ('__pad', ctypes.c_uint32, 16)])
@c.record
class struct_kgsl_perfcounter_put(c.Struct):
  SIZE = 16
  groupid: int
  countable: int
  __pad: c.Array[ctypes.c_uint32, Literal[2]]
struct_kgsl_perfcounter_put.register_fields([('groupid', ctypes.c_uint32, 0), ('countable', ctypes.c_uint32, 4), ('__pad', c.Array[ctypes.c_uint32, Literal[2]], 8)])
@c.record
class struct_kgsl_perfcounter_query(c.Struct):
  SIZE = 32
  groupid: int
  countables: c.POINTER[ctypes.c_uint32]
  count: int
  max_counters: int
  __pad: c.Array[ctypes.c_uint32, Literal[2]]
struct_kgsl_perfcounter_query.register_fields([('groupid', ctypes.c_uint32, 0), ('countables', c.POINTER[ctypes.c_uint32], 8), ('count', ctypes.c_uint32, 16), ('max_counters', ctypes.c_uint32, 20), ('__pad', c.Array[ctypes.c_uint32, Literal[2]], 24)])
@c.record
class struct_kgsl_perfcounter_read_group(c.Struct):
  SIZE = 16
  groupid: int
  countable: int
  value: int
struct_kgsl_perfcounter_read_group.register_fields([('groupid', ctypes.c_uint32, 0), ('countable', ctypes.c_uint32, 4), ('value', ctypes.c_uint64, 8)])
@c.record
class struct_kgsl_perfcounter_read(c.Struct):
  SIZE = 24
  reads: c.POINTER[struct_kgsl_perfcounter_read_group]
  count: int
  __pad: c.Array[ctypes.c_uint32, Literal[2]]
struct_kgsl_perfcounter_read.register_fields([('reads', c.POINTER[struct_kgsl_perfcounter_read_group], 0), ('count', ctypes.c_uint32, 8), ('__pad', c.Array[ctypes.c_uint32, Literal[2]], 12)])
@c.record
class struct_kgsl_gpumem_sync_cache_bulk(c.Struct):
  SIZE = 24
  id_list: c.POINTER[ctypes.c_uint32]
  count: int
  op: int
  __pad: c.Array[ctypes.c_uint32, Literal[2]]
struct_kgsl_gpumem_sync_cache_bulk.register_fields([('id_list', c.POINTER[ctypes.c_uint32], 0), ('count', ctypes.c_uint32, 8), ('op', ctypes.c_uint32, 12), ('__pad', c.Array[ctypes.c_uint32, Literal[2]], 16)])
@c.record
class struct_kgsl_cmd_syncpoint_timestamp(c.Struct):
  SIZE = 8
  context_id: int
  timestamp: int
struct_kgsl_cmd_syncpoint_timestamp.register_fields([('context_id', ctypes.c_uint32, 0), ('timestamp', ctypes.c_uint32, 4)])
@c.record
class struct_kgsl_cmd_syncpoint_fence(c.Struct):
  SIZE = 4
  fd: int
struct_kgsl_cmd_syncpoint_fence.register_fields([('fd', ctypes.c_int32, 0)])
@c.record
class struct_kgsl_cmd_syncpoint(c.Struct):
  SIZE = 24
  type: int
  priv: ctypes.c_void_p
  size: int
struct_kgsl_cmd_syncpoint.register_fields([('type', ctypes.c_int32, 0), ('priv', ctypes.c_void_p, 8), ('size', ctypes.c_uint64, 16)])
@c.record
class struct_kgsl_submit_commands(c.Struct):
  SIZE = 56
  context_id: int
  flags: int
  cmdlist: c.POINTER[struct_kgsl_ibdesc]
  numcmds: int
  synclist: c.POINTER[struct_kgsl_cmd_syncpoint]
  numsyncs: int
  timestamp: int
  __pad: c.Array[ctypes.c_uint32, Literal[4]]
struct_kgsl_submit_commands.register_fields([('context_id', ctypes.c_uint32, 0), ('flags', ctypes.c_uint32, 4), ('cmdlist', c.POINTER[struct_kgsl_ibdesc], 8), ('numcmds', ctypes.c_uint32, 16), ('synclist', c.POINTER[struct_kgsl_cmd_syncpoint], 24), ('numsyncs', ctypes.c_uint32, 32), ('timestamp', ctypes.c_uint32, 36), ('__pad', c.Array[ctypes.c_uint32, Literal[4]], 40)])
@c.record
class struct_kgsl_device_constraint(c.Struct):
  SIZE = 24
  type: int
  context_id: int
  data: ctypes.c_void_p
  size: int
struct_kgsl_device_constraint.register_fields([('type', ctypes.c_uint32, 0), ('context_id', ctypes.c_uint32, 4), ('data', ctypes.c_void_p, 8), ('size', ctypes.c_uint64, 16)])
@c.record
class struct_kgsl_device_constraint_pwrlevel(c.Struct):
  SIZE = 4
  level: int
struct_kgsl_device_constraint_pwrlevel.register_fields([('level', ctypes.c_uint32, 0)])
@c.record
class struct_kgsl_syncsource_create(c.Struct):
  SIZE = 16
  id: int
  __pad: c.Array[ctypes.c_uint32, Literal[3]]
struct_kgsl_syncsource_create.register_fields([('id', ctypes.c_uint32, 0), ('__pad', c.Array[ctypes.c_uint32, Literal[3]], 4)])
@c.record
class struct_kgsl_syncsource_destroy(c.Struct):
  SIZE = 16
  id: int
  __pad: c.Array[ctypes.c_uint32, Literal[3]]
struct_kgsl_syncsource_destroy.register_fields([('id', ctypes.c_uint32, 0), ('__pad', c.Array[ctypes.c_uint32, Literal[3]], 4)])
@c.record
class struct_kgsl_syncsource_create_fence(c.Struct):
  SIZE = 24
  id: int
  fence_fd: int
  __pad: c.Array[ctypes.c_uint32, Literal[4]]
struct_kgsl_syncsource_create_fence.register_fields([('id', ctypes.c_uint32, 0), ('fence_fd', ctypes.c_int32, 4), ('__pad', c.Array[ctypes.c_uint32, Literal[4]], 8)])
@c.record
class struct_kgsl_syncsource_signal_fence(c.Struct):
  SIZE = 24
  id: int
  fence_fd: int
  __pad: c.Array[ctypes.c_uint32, Literal[4]]
struct_kgsl_syncsource_signal_fence.register_fields([('id', ctypes.c_uint32, 0), ('fence_fd', ctypes.c_int32, 4), ('__pad', c.Array[ctypes.c_uint32, Literal[4]], 8)])
@c.record
class struct_kgsl_cff_sync_gpuobj(c.Struct):
  SIZE = 24
  offset: int
  length: int
  id: int
struct_kgsl_cff_sync_gpuobj.register_fields([('offset', ctypes.c_uint64, 0), ('length', ctypes.c_uint64, 8), ('id', ctypes.c_uint32, 16)])
@c.record
class struct_kgsl_gpuobj_alloc(c.Struct):
  SIZE = 48
  size: int
  flags: int
  va_len: int
  mmapsize: int
  id: int
  metadata_len: int
  metadata: int
struct_kgsl_gpuobj_alloc.register_fields([('size', ctypes.c_uint64, 0), ('flags', ctypes.c_uint64, 8), ('va_len', ctypes.c_uint64, 16), ('mmapsize', ctypes.c_uint64, 24), ('id', ctypes.c_uint32, 32), ('metadata_len', ctypes.c_uint32, 36), ('metadata', ctypes.c_uint64, 40)])
@c.record
class struct_kgsl_gpuobj_free(c.Struct):
  SIZE = 32
  flags: int
  priv: int
  id: int
  type: int
  len: int
struct_kgsl_gpuobj_free.register_fields([('flags', ctypes.c_uint64, 0), ('priv', ctypes.c_uint64, 8), ('id', ctypes.c_uint32, 16), ('type', ctypes.c_uint32, 20), ('len', ctypes.c_uint32, 24)])
@c.record
class struct_kgsl_gpu_event_timestamp(c.Struct):
  SIZE = 8
  context_id: int
  timestamp: int
struct_kgsl_gpu_event_timestamp.register_fields([('context_id', ctypes.c_uint32, 0), ('timestamp', ctypes.c_uint32, 4)])
@c.record
class struct_kgsl_gpu_event_fence(c.Struct):
  SIZE = 4
  fd: int
struct_kgsl_gpu_event_fence.register_fields([('fd', ctypes.c_int32, 0)])
@c.record
class struct_kgsl_gpuobj_info(c.Struct):
  SIZE = 48
  gpuaddr: int
  flags: int
  size: int
  va_len: int
  va_addr: int
  id: int
struct_kgsl_gpuobj_info.register_fields([('gpuaddr', ctypes.c_uint64, 0), ('flags', ctypes.c_uint64, 8), ('size', ctypes.c_uint64, 16), ('va_len', ctypes.c_uint64, 24), ('va_addr', ctypes.c_uint64, 32), ('id', ctypes.c_uint32, 40)])
@c.record
class struct_kgsl_gpuobj_import(c.Struct):
  SIZE = 32
  priv: int
  priv_len: int
  flags: int
  type: int
  id: int
struct_kgsl_gpuobj_import.register_fields([('priv', ctypes.c_uint64, 0), ('priv_len', ctypes.c_uint64, 8), ('flags', ctypes.c_uint64, 16), ('type', ctypes.c_uint32, 24), ('id', ctypes.c_uint32, 28)])
@c.record
class struct_kgsl_gpuobj_import_dma_buf(c.Struct):
  SIZE = 4
  fd: int
struct_kgsl_gpuobj_import_dma_buf.register_fields([('fd', ctypes.c_int32, 0)])
@c.record
class struct_kgsl_gpuobj_import_useraddr(c.Struct):
  SIZE = 8
  virtaddr: int
struct_kgsl_gpuobj_import_useraddr.register_fields([('virtaddr', ctypes.c_uint64, 0)])
@c.record
class struct_kgsl_gpuobj_sync_obj(c.Struct):
  SIZE = 24
  offset: int
  length: int
  id: int
  op: int
struct_kgsl_gpuobj_sync_obj.register_fields([('offset', ctypes.c_uint64, 0), ('length', ctypes.c_uint64, 8), ('id', ctypes.c_uint32, 16), ('op', ctypes.c_uint32, 20)])
@c.record
class struct_kgsl_gpuobj_sync(c.Struct):
  SIZE = 16
  objs: int
  obj_len: int
  count: int
struct_kgsl_gpuobj_sync.register_fields([('objs', ctypes.c_uint64, 0), ('obj_len', ctypes.c_uint32, 8), ('count', ctypes.c_uint32, 12)])
@c.record
class struct_kgsl_command_object(c.Struct):
  SIZE = 32
  offset: int
  gpuaddr: int
  size: int
  flags: int
  id: int
struct_kgsl_command_object.register_fields([('offset', ctypes.c_uint64, 0), ('gpuaddr', ctypes.c_uint64, 8), ('size', ctypes.c_uint64, 16), ('flags', ctypes.c_uint32, 24), ('id', ctypes.c_uint32, 28)])
@c.record
class struct_kgsl_command_syncpoint(c.Struct):
  SIZE = 24
  priv: int
  size: int
  type: int
struct_kgsl_command_syncpoint.register_fields([('priv', ctypes.c_uint64, 0), ('size', ctypes.c_uint64, 8), ('type', ctypes.c_uint32, 16)])
@c.record
class struct_kgsl_gpu_command(c.Struct):
  SIZE = 64
  flags: int
  cmdlist: int
  cmdsize: int
  numcmds: int
  objlist: int
  objsize: int
  numobjs: int
  synclist: int
  syncsize: int
  numsyncs: int
  context_id: int
  timestamp: int
struct_kgsl_gpu_command.register_fields([('flags', ctypes.c_uint64, 0), ('cmdlist', ctypes.c_uint64, 8), ('cmdsize', ctypes.c_uint32, 16), ('numcmds', ctypes.c_uint32, 20), ('objlist', ctypes.c_uint64, 24), ('objsize', ctypes.c_uint32, 32), ('numobjs', ctypes.c_uint32, 36), ('synclist', ctypes.c_uint64, 40), ('syncsize', ctypes.c_uint32, 48), ('numsyncs', ctypes.c_uint32, 52), ('context_id', ctypes.c_uint32, 56), ('timestamp', ctypes.c_uint32, 60)])
@c.record
class struct_kgsl_preemption_counters_query(c.Struct):
  SIZE = 24
  counters: int
  size_user: int
  size_priority_level: int
  max_priority_level: int
struct_kgsl_preemption_counters_query.register_fields([('counters', ctypes.c_uint64, 0), ('size_user', ctypes.c_uint32, 8), ('size_priority_level', ctypes.c_uint32, 12), ('max_priority_level', ctypes.c_uint32, 16)])
@c.record
class struct_kgsl_gpuobj_set_info(c.Struct):
  SIZE = 32
  flags: int
  metadata: int
  id: int
  metadata_len: int
  type: int
struct_kgsl_gpuobj_set_info.register_fields([('flags', ctypes.c_uint64, 0), ('metadata', ctypes.c_uint64, 8), ('id', ctypes.c_uint32, 16), ('metadata_len', ctypes.c_uint32, 20), ('type', ctypes.c_uint32, 24)])
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
KGSL_USERMEM_FLAG = lambda x: (((x) + 1) << KGSL_MEMFLAGS_USERMEM_SHIFT) # type: ignore
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
KGSL_CONVERT_TO_MBPS = lambda val: (val*1000*1000) # type: ignore
KGSL_MEMSTORE_OFFSET = lambda ctxt_id,field: ((ctxt_id)*sizeof(struct_kgsl_devmemstore) + offsetof(struct_kgsl_devmemstore, field)) # type: ignore
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