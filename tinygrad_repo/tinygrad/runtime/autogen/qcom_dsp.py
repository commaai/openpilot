# mypy: disable-error-code="empty-body"
from __future__ import annotations
import ctypes
from typing import Annotated, Literal, TypeAlias
from tinygrad.runtime.support.c import _IO, _IOW, _IOR, _IOWR
from tinygrad.runtime.support import c
ion_user_handle_t: TypeAlias = Annotated[int, ctypes.c_int32]
class enum_ion_heap_type(Annotated[int, ctypes.c_uint32], c.Enum): pass
ION_HEAP_TYPE_SYSTEM = enum_ion_heap_type.define('ION_HEAP_TYPE_SYSTEM', 0)
ION_HEAP_TYPE_SYSTEM_CONTIG = enum_ion_heap_type.define('ION_HEAP_TYPE_SYSTEM_CONTIG', 1)
ION_HEAP_TYPE_CARVEOUT = enum_ion_heap_type.define('ION_HEAP_TYPE_CARVEOUT', 2)
ION_HEAP_TYPE_CHUNK = enum_ion_heap_type.define('ION_HEAP_TYPE_CHUNK', 3)
ION_HEAP_TYPE_DMA = enum_ion_heap_type.define('ION_HEAP_TYPE_DMA', 4)
ION_HEAP_TYPE_CUSTOM = enum_ion_heap_type.define('ION_HEAP_TYPE_CUSTOM', 5)
ION_NUM_HEAPS = enum_ion_heap_type.define('ION_NUM_HEAPS', 16)

@c.record
class struct_ion_allocation_data(c.Struct):
  SIZE = 32
  len: Annotated[size_t, 0]
  align: Annotated[size_t, 8]
  heap_id_mask: Annotated[Annotated[int, ctypes.c_uint32], 16]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 20]
  handle: Annotated[ion_user_handle_t, 24]
size_t: TypeAlias = Annotated[int, ctypes.c_uint64]
@c.record
class struct_ion_fd_data(c.Struct):
  SIZE = 8
  handle: Annotated[ion_user_handle_t, 0]
  fd: Annotated[Annotated[int, ctypes.c_int32], 4]
@c.record
class struct_ion_handle_data(c.Struct):
  SIZE = 4
  handle: Annotated[ion_user_handle_t, 0]
@c.record
class struct_ion_custom_data(c.Struct):
  SIZE = 16
  cmd: Annotated[Annotated[int, ctypes.c_uint32], 0]
  arg: Annotated[Annotated[int, ctypes.c_uint64], 8]
class enum_msm_ion_heap_types(Annotated[int, ctypes.c_uint32], c.Enum): pass
ION_HEAP_TYPE_MSM_START = enum_msm_ion_heap_types.define('ION_HEAP_TYPE_MSM_START', 6)
ION_HEAP_TYPE_SECURE_DMA = enum_msm_ion_heap_types.define('ION_HEAP_TYPE_SECURE_DMA', 6)
ION_HEAP_TYPE_SYSTEM_SECURE = enum_msm_ion_heap_types.define('ION_HEAP_TYPE_SYSTEM_SECURE', 7)
ION_HEAP_TYPE_HYP_CMA = enum_msm_ion_heap_types.define('ION_HEAP_TYPE_HYP_CMA', 8)

class enum_ion_heap_ids(Annotated[int, ctypes.c_int32], c.Enum): pass
INVALID_HEAP_ID = enum_ion_heap_ids.define('INVALID_HEAP_ID', -1)
ION_CP_MM_HEAP_ID = enum_ion_heap_ids.define('ION_CP_MM_HEAP_ID', 8)
ION_SECURE_HEAP_ID = enum_ion_heap_ids.define('ION_SECURE_HEAP_ID', 9)
ION_SECURE_DISPLAY_HEAP_ID = enum_ion_heap_ids.define('ION_SECURE_DISPLAY_HEAP_ID', 10)
ION_CP_MFC_HEAP_ID = enum_ion_heap_ids.define('ION_CP_MFC_HEAP_ID', 12)
ION_CP_WB_HEAP_ID = enum_ion_heap_ids.define('ION_CP_WB_HEAP_ID', 16)
ION_CAMERA_HEAP_ID = enum_ion_heap_ids.define('ION_CAMERA_HEAP_ID', 20)
ION_SYSTEM_CONTIG_HEAP_ID = enum_ion_heap_ids.define('ION_SYSTEM_CONTIG_HEAP_ID', 21)
ION_ADSP_HEAP_ID = enum_ion_heap_ids.define('ION_ADSP_HEAP_ID', 22)
ION_PIL1_HEAP_ID = enum_ion_heap_ids.define('ION_PIL1_HEAP_ID', 23)
ION_SF_HEAP_ID = enum_ion_heap_ids.define('ION_SF_HEAP_ID', 24)
ION_SYSTEM_HEAP_ID = enum_ion_heap_ids.define('ION_SYSTEM_HEAP_ID', 25)
ION_PIL2_HEAP_ID = enum_ion_heap_ids.define('ION_PIL2_HEAP_ID', 26)
ION_QSECOM_HEAP_ID = enum_ion_heap_ids.define('ION_QSECOM_HEAP_ID', 27)
ION_AUDIO_HEAP_ID = enum_ion_heap_ids.define('ION_AUDIO_HEAP_ID', 28)
ION_MM_FIRMWARE_HEAP_ID = enum_ion_heap_ids.define('ION_MM_FIRMWARE_HEAP_ID', 29)
ION_HEAP_ID_RESERVED = enum_ion_heap_ids.define('ION_HEAP_ID_RESERVED', 31)

class enum_ion_fixed_position(Annotated[int, ctypes.c_uint32], c.Enum): pass
NOT_FIXED = enum_ion_fixed_position.define('NOT_FIXED', 0)
FIXED_LOW = enum_ion_fixed_position.define('FIXED_LOW', 1)
FIXED_MIDDLE = enum_ion_fixed_position.define('FIXED_MIDDLE', 2)
FIXED_HIGH = enum_ion_fixed_position.define('FIXED_HIGH', 3)

class enum_cp_mem_usage(Annotated[int, ctypes.c_uint32], c.Enum): pass
VIDEO_BITSTREAM = enum_cp_mem_usage.define('VIDEO_BITSTREAM', 1)
VIDEO_PIXEL = enum_cp_mem_usage.define('VIDEO_PIXEL', 2)
VIDEO_NONPIXEL = enum_cp_mem_usage.define('VIDEO_NONPIXEL', 3)
DISPLAY_SECURE_CP_USAGE = enum_cp_mem_usage.define('DISPLAY_SECURE_CP_USAGE', 4)
CAMERA_SECURE_CP_USAGE = enum_cp_mem_usage.define('CAMERA_SECURE_CP_USAGE', 5)
MAX_USAGE = enum_cp_mem_usage.define('MAX_USAGE', 6)
UNKNOWN = enum_cp_mem_usage.define('UNKNOWN', 2147483647)

@c.record
class struct_ion_flush_data(c.Struct):
  SIZE = 24
  handle: Annotated[ion_user_handle_t, 0]
  fd: Annotated[Annotated[int, ctypes.c_int32], 4]
  vaddr: Annotated[ctypes.c_void_p, 8]
  offset: Annotated[Annotated[int, ctypes.c_uint32], 16]
  length: Annotated[Annotated[int, ctypes.c_uint32], 20]
@c.record
class struct_ion_prefetch_regions(c.Struct):
  SIZE = 24
  vmid: Annotated[Annotated[int, ctypes.c_uint32], 0]
  sizes: Annotated[c.POINTER[size_t], 8]
  nr_sizes: Annotated[Annotated[int, ctypes.c_uint32], 16]
@c.record
class struct_ion_prefetch_data(c.Struct):
  SIZE = 32
  heap_id: Annotated[Annotated[int, ctypes.c_int32], 0]
  len: Annotated[Annotated[int, ctypes.c_uint64], 8]
  regions: Annotated[c.POINTER[struct_ion_prefetch_regions], 16]
  nr_regions: Annotated[Annotated[int, ctypes.c_uint32], 24]
@c.record
class struct_remote_buf64(c.Struct):
  SIZE = 16
  pv: Annotated[uint64_t, 0]
  len: Annotated[uint64_t, 8]
uint64_t: TypeAlias = Annotated[int, ctypes.c_uint64]
@c.record
class struct_remote_dma_handle64(c.Struct):
  SIZE = 12
  fd: Annotated[Annotated[int, ctypes.c_int32], 0]
  offset: Annotated[uint32_t, 4]
  len: Annotated[uint32_t, 8]
uint32_t: TypeAlias = Annotated[int, ctypes.c_uint32]
@c.record
class union_remote_arg64(c.Struct):
  SIZE = 16
  buf: Annotated[struct_remote_buf64, 0]
  dma: Annotated[struct_remote_dma_handle64, 0]
  h: Annotated[uint32_t, 0]
@c.record
class struct_remote_buf(c.Struct):
  SIZE = 16
  pv: Annotated[ctypes.c_void_p, 0]
  len: Annotated[size_t, 8]
@c.record
class struct_remote_dma_handle(c.Struct):
  SIZE = 8
  fd: Annotated[Annotated[int, ctypes.c_int32], 0]
  offset: Annotated[uint32_t, 4]
@c.record
class union_remote_arg(c.Struct):
  SIZE = 16
  buf: Annotated[struct_remote_buf, 0]
  dma: Annotated[struct_remote_dma_handle, 0]
  h: Annotated[uint32_t, 0]
@c.record
class struct_fastrpc_ioctl_invoke(c.Struct):
  SIZE = 16
  handle: Annotated[uint32_t, 0]
  sc: Annotated[uint32_t, 4]
  pra: Annotated[c.POINTER[union_remote_arg], 8]
@c.record
class struct_fastrpc_ioctl_invoke_fd(c.Struct):
  SIZE = 24
  inv: Annotated[struct_fastrpc_ioctl_invoke, 0]
  fds: Annotated[c.POINTER[Annotated[int, ctypes.c_int32]], 16]
@c.record
class struct_fastrpc_ioctl_invoke_attrs(c.Struct):
  SIZE = 32
  inv: Annotated[struct_fastrpc_ioctl_invoke, 0]
  fds: Annotated[c.POINTER[Annotated[int, ctypes.c_int32]], 16]
  attrs: Annotated[c.POINTER[Annotated[int, ctypes.c_uint32]], 24]
@c.record
class struct_fastrpc_ioctl_invoke_crc(c.Struct):
  SIZE = 40
  inv: Annotated[struct_fastrpc_ioctl_invoke, 0]
  fds: Annotated[c.POINTER[Annotated[int, ctypes.c_int32]], 16]
  attrs: Annotated[c.POINTER[Annotated[int, ctypes.c_uint32]], 24]
  crc: Annotated[c.POINTER[Annotated[int, ctypes.c_uint32]], 32]
@c.record
class struct_fastrpc_ioctl_init(c.Struct):
  SIZE = 40
  flags: Annotated[uint32_t, 0]
  file: Annotated[uintptr_t, 8]
  filelen: Annotated[uint32_t, 16]
  filefd: Annotated[int32_t, 20]
  mem: Annotated[uintptr_t, 24]
  memlen: Annotated[uint32_t, 32]
  memfd: Annotated[int32_t, 36]
uintptr_t: TypeAlias = Annotated[int, ctypes.c_uint64]
int32_t: TypeAlias = Annotated[int, ctypes.c_int32]
@c.record
class struct_fastrpc_ioctl_init_attrs(c.Struct):
  SIZE = 48
  init: Annotated[struct_fastrpc_ioctl_init, 0]
  attrs: Annotated[Annotated[int, ctypes.c_int32], 40]
  siglen: Annotated[Annotated[int, ctypes.c_uint32], 44]
@c.record
class struct_fastrpc_ioctl_munmap(c.Struct):
  SIZE = 16
  vaddrout: Annotated[uintptr_t, 0]
  size: Annotated[size_t, 8]
@c.record
class struct_fastrpc_ioctl_munmap_64(c.Struct):
  SIZE = 16
  vaddrout: Annotated[uint64_t, 0]
  size: Annotated[size_t, 8]
@c.record
class struct_fastrpc_ioctl_mmap(c.Struct):
  SIZE = 32
  fd: Annotated[Annotated[int, ctypes.c_int32], 0]
  flags: Annotated[uint32_t, 4]
  vaddrin: Annotated[uintptr_t, 8]
  size: Annotated[size_t, 16]
  vaddrout: Annotated[uintptr_t, 24]
@c.record
class struct_fastrpc_ioctl_mmap_64(c.Struct):
  SIZE = 32
  fd: Annotated[Annotated[int, ctypes.c_int32], 0]
  flags: Annotated[uint32_t, 4]
  vaddrin: Annotated[uint64_t, 8]
  size: Annotated[size_t, 16]
  vaddrout: Annotated[uint64_t, 24]
@c.record
class struct_fastrpc_ioctl_munmap_fd(c.Struct):
  SIZE = 24
  fd: Annotated[Annotated[int, ctypes.c_int32], 0]
  flags: Annotated[uint32_t, 4]
  va: Annotated[uintptr_t, 8]
  len: Annotated[ssize_t, 16]
ssize_t: TypeAlias = Annotated[int, ctypes.c_int64]
@c.record
class struct_fastrpc_ioctl_perf(c.Struct):
  SIZE = 24
  data: Annotated[uintptr_t, 0]
  numkeys: Annotated[uint32_t, 8]
  keys: Annotated[uintptr_t, 16]
@c.record
class struct_fastrpc_ctrl_latency(c.Struct):
  SIZE = 8
  enable: Annotated[uint32_t, 0]
  level: Annotated[uint32_t, 4]
@c.record
class struct_fastrpc_ctrl_smmu(c.Struct):
  SIZE = 4
  sharedcb: Annotated[uint32_t, 0]
@c.record
class struct_fastrpc_ctrl_kalloc(c.Struct):
  SIZE = 4
  kalloc_support: Annotated[uint32_t, 0]
@c.record
class struct_fastrpc_ioctl_control(c.Struct):
  SIZE = 12
  req: Annotated[uint32_t, 0]
  lp: Annotated[struct_fastrpc_ctrl_latency, 4]
  smmu: Annotated[struct_fastrpc_ctrl_smmu, 4]
  kalloc: Annotated[struct_fastrpc_ctrl_kalloc, 4]
@c.record
class struct_smq_null_invoke(c.Struct):
  SIZE = 16
  ctx: Annotated[uint64_t, 0]
  handle: Annotated[uint32_t, 8]
  sc: Annotated[uint32_t, 12]
@c.record
class struct_smq_phy_page(c.Struct):
  SIZE = 16
  addr: Annotated[uint64_t, 0]
  size: Annotated[uint64_t, 8]
@c.record
class struct_smq_invoke_buf(c.Struct):
  SIZE = 8
  num: Annotated[Annotated[int, ctypes.c_int32], 0]
  pgidx: Annotated[Annotated[int, ctypes.c_int32], 4]
@c.record
class struct_smq_invoke(c.Struct):
  SIZE = 32
  header: Annotated[struct_smq_null_invoke, 0]
  page: Annotated[struct_smq_phy_page, 16]
@c.record
class struct_smq_msg(c.Struct):
  SIZE = 40
  pid: Annotated[uint32_t, 0]
  tid: Annotated[uint32_t, 4]
  invoke: Annotated[struct_smq_invoke, 8]
@c.record
class struct_smq_invoke_rsp(c.Struct):
  SIZE = 16
  ctx: Annotated[uint64_t, 0]
  retval: Annotated[Annotated[int, ctypes.c_int32], 8]
remote_handle: TypeAlias = Annotated[int, ctypes.c_uint32]
remote_handle64: TypeAlias = Annotated[int, ctypes.c_uint64]
fastrpc_async_jobid: TypeAlias = Annotated[int, ctypes.c_uint64]
@c.record
class remote_buf(c.Struct):
  SIZE = 16
  pv: Annotated[ctypes.c_void_p, 0]
  nLen: Annotated[size_t, 8]
@c.record
class remote_dma_handle(c.Struct):
  SIZE = 8
  fd: Annotated[int32_t, 0]
  offset: Annotated[uint32_t, 4]
@c.record
class remote_arg(c.Struct):
  SIZE = 16
  buf: Annotated[remote_buf, 0]
  h: Annotated[remote_handle, 0]
  h64: Annotated[remote_handle64, 0]
  dma: Annotated[remote_dma_handle, 0]
class enum_fastrpc_async_notify_type(Annotated[int, ctypes.c_uint32], c.Enum): pass
FASTRPC_ASYNC_NO_SYNC = enum_fastrpc_async_notify_type.define('FASTRPC_ASYNC_NO_SYNC', 0)
FASTRPC_ASYNC_CALLBACK = enum_fastrpc_async_notify_type.define('FASTRPC_ASYNC_CALLBACK', 1)
FASTRPC_ASYNC_POLL = enum_fastrpc_async_notify_type.define('FASTRPC_ASYNC_POLL', 2)
FASTRPC_ASYNC_TYPE_MAX = enum_fastrpc_async_notify_type.define('FASTRPC_ASYNC_TYPE_MAX', 3)

@c.record
class struct_fastrpc_async_callback(c.Struct):
  SIZE = 16
  fn: Annotated[c.CFUNCTYPE[None, [fastrpc_async_jobid, ctypes.c_void_p, Annotated[int, ctypes.c_int32]]], 0]
  context: Annotated[ctypes.c_void_p, 8]
fastrpc_async_callback_t: TypeAlias = struct_fastrpc_async_callback
@c.record
class struct_fastrpc_async_descriptor(c.Struct):
  SIZE = 32
  type: Annotated[enum_fastrpc_async_notify_type, 0]
  jobid: Annotated[fastrpc_async_jobid, 8]
  cb: Annotated[fastrpc_async_callback_t, 16]
fastrpc_async_descriptor_t: TypeAlias = struct_fastrpc_async_descriptor
class enum_fastrpc_process_type(Annotated[int, ctypes.c_uint32], c.Enum): pass
PROCESS_TYPE_SIGNED = enum_fastrpc_process_type.define('PROCESS_TYPE_SIGNED', 0)
PROCESS_TYPE_UNSIGNED = enum_fastrpc_process_type.define('PROCESS_TYPE_UNSIGNED', 1)

class enum_handle_control_req_id(Annotated[int, ctypes.c_uint32], c.Enum): pass
DSPRPC_CONTROL_LATENCY = enum_handle_control_req_id.define('DSPRPC_CONTROL_LATENCY', 1)
DSPRPC_GET_DSP_INFO = enum_handle_control_req_id.define('DSPRPC_GET_DSP_INFO', 2)
DSPRPC_CONTROL_WAKELOCK = enum_handle_control_req_id.define('DSPRPC_CONTROL_WAKELOCK', 3)
DSPRPC_GET_DOMAIN = enum_handle_control_req_id.define('DSPRPC_GET_DOMAIN', 4)

class enum_remote_rpc_latency_flags(Annotated[int, ctypes.c_uint32], c.Enum): pass
RPC_DISABLE_QOS = enum_remote_rpc_latency_flags.define('RPC_DISABLE_QOS', 0)
RPC_PM_QOS = enum_remote_rpc_latency_flags.define('RPC_PM_QOS', 1)
RPC_ADAPTIVE_QOS = enum_remote_rpc_latency_flags.define('RPC_ADAPTIVE_QOS', 2)
RPC_POLL_QOS = enum_remote_rpc_latency_flags.define('RPC_POLL_QOS', 3)

remote_rpc_control_latency_t: TypeAlias = enum_remote_rpc_latency_flags
@c.record
class struct_remote_rpc_control_latency(c.Struct):
  SIZE = 8
  enable: Annotated[remote_rpc_control_latency_t, 0]
  latency: Annotated[uint32_t, 4]
class enum_remote_dsp_attributes(Annotated[int, ctypes.c_uint32], c.Enum): pass
DOMAIN_SUPPORT = enum_remote_dsp_attributes.define('DOMAIN_SUPPORT', 0)
UNSIGNED_PD_SUPPORT = enum_remote_dsp_attributes.define('UNSIGNED_PD_SUPPORT', 1)
HVX_SUPPORT_64B = enum_remote_dsp_attributes.define('HVX_SUPPORT_64B', 2)
HVX_SUPPORT_128B = enum_remote_dsp_attributes.define('HVX_SUPPORT_128B', 3)
VTCM_PAGE = enum_remote_dsp_attributes.define('VTCM_PAGE', 4)
VTCM_COUNT = enum_remote_dsp_attributes.define('VTCM_COUNT', 5)
ARCH_VER = enum_remote_dsp_attributes.define('ARCH_VER', 6)
HMX_SUPPORT_DEPTH = enum_remote_dsp_attributes.define('HMX_SUPPORT_DEPTH', 7)
HMX_SUPPORT_SPATIAL = enum_remote_dsp_attributes.define('HMX_SUPPORT_SPATIAL', 8)
ASYNC_FASTRPC_SUPPORT = enum_remote_dsp_attributes.define('ASYNC_FASTRPC_SUPPORT', 9)
STATUS_NOTIFICATION_SUPPORT = enum_remote_dsp_attributes.define('STATUS_NOTIFICATION_SUPPORT', 10)
FASTRPC_MAX_DSP_ATTRIBUTES = enum_remote_dsp_attributes.define('FASTRPC_MAX_DSP_ATTRIBUTES', 11)

@c.record
class struct_remote_dsp_capability(c.Struct):
  SIZE = 12
  domain: Annotated[uint32_t, 0]
  attribute_ID: Annotated[uint32_t, 4]
  capability: Annotated[uint32_t, 8]
fastrpc_capability: TypeAlias = struct_remote_dsp_capability
@c.record
class struct_remote_rpc_control_wakelock(c.Struct):
  SIZE = 4
  enable: Annotated[uint32_t, 0]
@c.record
class struct_remote_rpc_get_domain(c.Struct):
  SIZE = 4
  domain: Annotated[Annotated[int, ctypes.c_int32], 0]
remote_rpc_get_domain_t: TypeAlias = struct_remote_rpc_get_domain
class enum_session_control_req_id(Annotated[int, ctypes.c_uint32], c.Enum): pass
FASTRPC_THREAD_PARAMS = enum_session_control_req_id.define('FASTRPC_THREAD_PARAMS', 1)
DSPRPC_CONTROL_UNSIGNED_MODULE = enum_session_control_req_id.define('DSPRPC_CONTROL_UNSIGNED_MODULE', 2)
FASTRPC_RELATIVE_THREAD_PRIORITY = enum_session_control_req_id.define('FASTRPC_RELATIVE_THREAD_PRIORITY', 4)
FASTRPC_REMOTE_PROCESS_KILL = enum_session_control_req_id.define('FASTRPC_REMOTE_PROCESS_KILL', 6)
FASTRPC_SESSION_CLOSE = enum_session_control_req_id.define('FASTRPC_SESSION_CLOSE', 7)
FASTRPC_CONTROL_PD_DUMP = enum_session_control_req_id.define('FASTRPC_CONTROL_PD_DUMP', 8)
FASTRPC_REMOTE_PROCESS_EXCEPTION = enum_session_control_req_id.define('FASTRPC_REMOTE_PROCESS_EXCEPTION', 9)
FASTRPC_REMOTE_PROCESS_TYPE = enum_session_control_req_id.define('FASTRPC_REMOTE_PROCESS_TYPE', 10)
FASTRPC_REGISTER_STATUS_NOTIFICATIONS = enum_session_control_req_id.define('FASTRPC_REGISTER_STATUS_NOTIFICATIONS', 11)

@c.record
class struct_remote_rpc_thread_params(c.Struct):
  SIZE = 12
  domain: Annotated[Annotated[int, ctypes.c_int32], 0]
  prio: Annotated[Annotated[int, ctypes.c_int32], 4]
  stack_size: Annotated[Annotated[int, ctypes.c_int32], 8]
@c.record
class struct_remote_rpc_control_unsigned_module(c.Struct):
  SIZE = 8
  domain: Annotated[Annotated[int, ctypes.c_int32], 0]
  enable: Annotated[Annotated[int, ctypes.c_int32], 4]
@c.record
class struct_remote_rpc_relative_thread_priority(c.Struct):
  SIZE = 8
  domain: Annotated[Annotated[int, ctypes.c_int32], 0]
  relative_thread_priority: Annotated[Annotated[int, ctypes.c_int32], 4]
@c.record
class struct_remote_rpc_process_clean_params(c.Struct):
  SIZE = 4
  domain: Annotated[Annotated[int, ctypes.c_int32], 0]
@c.record
class struct_remote_rpc_session_close(c.Struct):
  SIZE = 4
  domain: Annotated[Annotated[int, ctypes.c_int32], 0]
@c.record
class struct_remote_rpc_control_pd_dump(c.Struct):
  SIZE = 8
  domain: Annotated[Annotated[int, ctypes.c_int32], 0]
  enable: Annotated[Annotated[int, ctypes.c_int32], 4]
@c.record
class struct_remote_process_type(c.Struct):
  SIZE = 8
  domain: Annotated[Annotated[int, ctypes.c_int32], 0]
  process_type: Annotated[Annotated[int, ctypes.c_int32], 4]
remote_rpc_process_exception: TypeAlias = struct_remote_rpc_process_clean_params
class enum_remote_rpc_status_flags(Annotated[int, ctypes.c_uint32], c.Enum): pass
FASTRPC_USER_PD_UP = enum_remote_rpc_status_flags.define('FASTRPC_USER_PD_UP', 0)
FASTRPC_USER_PD_EXIT = enum_remote_rpc_status_flags.define('FASTRPC_USER_PD_EXIT', 1)
FASTRPC_USER_PD_FORCE_KILL = enum_remote_rpc_status_flags.define('FASTRPC_USER_PD_FORCE_KILL', 2)
FASTRPC_USER_PD_EXCEPTION = enum_remote_rpc_status_flags.define('FASTRPC_USER_PD_EXCEPTION', 3)
FASTRPC_DSP_SSR = enum_remote_rpc_status_flags.define('FASTRPC_DSP_SSR', 4)

remote_rpc_status_flags_t: TypeAlias = enum_remote_rpc_status_flags
fastrpc_notif_fn_t: TypeAlias = c.CFUNCTYPE[Annotated[int, ctypes.c_int32], [ctypes.c_void_p, Annotated[int, ctypes.c_int32], Annotated[int, ctypes.c_int32], enum_remote_rpc_status_flags]]
@c.record
class struct_remote_rpc_notif_register(c.Struct):
  SIZE = 24
  context: Annotated[ctypes.c_void_p, 0]
  domain: Annotated[Annotated[int, ctypes.c_int32], 8]
  notifier_fn: Annotated[fastrpc_notif_fn_t, 16]
remote_rpc_notif_register_t: TypeAlias = struct_remote_rpc_notif_register
class enum_remote_mem_map_flags(Annotated[int, ctypes.c_uint32], c.Enum): pass
REMOTE_MAP_MEM_STATIC = enum_remote_mem_map_flags.define('REMOTE_MAP_MEM_STATIC', 0)
REMOTE_MAP_MAX_FLAG = enum_remote_mem_map_flags.define('REMOTE_MAP_MAX_FLAG', 1)

class enum_remote_buf_attributes(Annotated[int, ctypes.c_uint32], c.Enum): pass
FASTRPC_ATTR_NON_COHERENT = enum_remote_buf_attributes.define('FASTRPC_ATTR_NON_COHERENT', 2)
FASTRPC_ATTR_COHERENT = enum_remote_buf_attributes.define('FASTRPC_ATTR_COHERENT', 4)
FASTRPC_ATTR_KEEP_MAP = enum_remote_buf_attributes.define('FASTRPC_ATTR_KEEP_MAP', 8)
FASTRPC_ATTR_NOMAP = enum_remote_buf_attributes.define('FASTRPC_ATTR_NOMAP', 16)
FASTRPC_ATTR_FORCE_NOFLUSH = enum_remote_buf_attributes.define('FASTRPC_ATTR_FORCE_NOFLUSH', 32)
FASTRPC_ATTR_FORCE_NOINVALIDATE = enum_remote_buf_attributes.define('FASTRPC_ATTR_FORCE_NOINVALIDATE', 64)
FASTRPC_ATTR_TRY_MAP_STATIC = enum_remote_buf_attributes.define('FASTRPC_ATTR_TRY_MAP_STATIC', 128)

class enum_fastrpc_map_flags(Annotated[int, ctypes.c_uint32], c.Enum): pass
FASTRPC_MAP_STATIC = enum_fastrpc_map_flags.define('FASTRPC_MAP_STATIC', 0)
FASTRPC_MAP_RESERVED = enum_fastrpc_map_flags.define('FASTRPC_MAP_RESERVED', 1)
FASTRPC_MAP_FD = enum_fastrpc_map_flags.define('FASTRPC_MAP_FD', 2)
FASTRPC_MAP_FD_DELAYED = enum_fastrpc_map_flags.define('FASTRPC_MAP_FD_DELAYED', 3)
FASTRPC_MAP_MAX = enum_fastrpc_map_flags.define('FASTRPC_MAP_MAX', 4)

@c.record
class struct__cstring1_s(c.Struct):
  SIZE = 16
  data: Annotated[c.POINTER[Annotated[bytes, ctypes.c_char]], 0]
  dataLen: Annotated[Annotated[int, ctypes.c_int32], 8]
_cstring1_t: TypeAlias = struct__cstring1_s
apps_std_FILE: TypeAlias = Annotated[int, ctypes.c_int32]
class enum_apps_std_SEEK(Annotated[int, ctypes.c_uint32], c.Enum): pass
APPS_STD_SEEK_SET = enum_apps_std_SEEK.define('APPS_STD_SEEK_SET', 0)
APPS_STD_SEEK_CUR = enum_apps_std_SEEK.define('APPS_STD_SEEK_CUR', 1)
APPS_STD_SEEK_END = enum_apps_std_SEEK.define('APPS_STD_SEEK_END', 2)
_32BIT_PLACEHOLDER_apps_std_SEEK = enum_apps_std_SEEK.define('_32BIT_PLACEHOLDER_apps_std_SEEK', 2147483647)

apps_std_SEEK: TypeAlias = enum_apps_std_SEEK
@c.record
class struct_apps_std_DIR(c.Struct):
  SIZE = 8
  handle: Annotated[uint64, 0]
uint64: TypeAlias = Annotated[int, ctypes.c_uint64]
apps_std_DIR: TypeAlias = struct_apps_std_DIR
@c.record
class struct_apps_std_DIRENT(c.Struct):
  SIZE = 260
  ino: Annotated[Annotated[int, ctypes.c_int32], 0]
  name: Annotated[c.Array[Annotated[bytes, ctypes.c_char], Literal[255]], 4]
apps_std_DIRENT: TypeAlias = struct_apps_std_DIRENT
@c.record
class struct_apps_std_STAT(c.Struct):
  SIZE = 96
  tsz: Annotated[uint64, 0]
  dev: Annotated[uint64, 8]
  ino: Annotated[uint64, 16]
  mode: Annotated[uint32, 24]
  nlink: Annotated[uint32, 28]
  rdev: Annotated[uint64, 32]
  size: Annotated[uint64, 40]
  atime: Annotated[int64, 48]
  atimensec: Annotated[int64, 56]
  mtime: Annotated[int64, 64]
  mtimensec: Annotated[int64, 72]
  ctime: Annotated[int64, 80]
  ctimensec: Annotated[int64, 88]
uint32: TypeAlias = Annotated[int, ctypes.c_uint32]
int64: TypeAlias = Annotated[int, ctypes.c_int64]
apps_std_STAT: TypeAlias = struct_apps_std_STAT
c.init_records()
ION_HEAP_SYSTEM_MASK = ((1 << ION_HEAP_TYPE_SYSTEM)) # type: ignore
ION_HEAP_SYSTEM_CONTIG_MASK = ((1 << ION_HEAP_TYPE_SYSTEM_CONTIG)) # type: ignore
ION_HEAP_CARVEOUT_MASK = ((1 << ION_HEAP_TYPE_CARVEOUT)) # type: ignore
ION_HEAP_TYPE_DMA_MASK = ((1 << ION_HEAP_TYPE_DMA)) # type: ignore
ION_FLAG_CACHED = 1 # type: ignore
ION_FLAG_CACHED_NEEDS_SYNC = 2 # type: ignore
ION_IOC_MAGIC = 'I' # type: ignore
ION_IOC_ALLOC = _IOWR(ION_IOC_MAGIC, 0, struct_ion_allocation_data) # type: ignore
ION_IOC_FREE = _IOWR(ION_IOC_MAGIC, 1, struct_ion_handle_data) # type: ignore
ION_IOC_MAP = _IOWR(ION_IOC_MAGIC, 2, struct_ion_fd_data) # type: ignore
ION_IOC_SHARE = _IOWR(ION_IOC_MAGIC, 4, struct_ion_fd_data) # type: ignore
ION_IOC_IMPORT = _IOWR(ION_IOC_MAGIC, 5, struct_ion_fd_data) # type: ignore
ION_IOC_SYNC = _IOWR(ION_IOC_MAGIC, 7, struct_ion_fd_data) # type: ignore
ION_IOC_CUSTOM = _IOWR(ION_IOC_MAGIC, 6, struct_ion_custom_data) # type: ignore
ION_IOMMU_HEAP_ID = ION_SYSTEM_HEAP_ID # type: ignore
ION_HEAP_TYPE_IOMMU = ION_HEAP_TYPE_SYSTEM # type: ignore
ION_FLAG_CP_TOUCH = (1 << 17) # type: ignore
ION_FLAG_CP_BITSTREAM = (1 << 18) # type: ignore
ION_FLAG_CP_PIXEL = (1 << 19) # type: ignore
ION_FLAG_CP_NON_PIXEL = (1 << 20) # type: ignore
ION_FLAG_CP_CAMERA = (1 << 21) # type: ignore
ION_FLAG_CP_HLOS = (1 << 22) # type: ignore
ION_FLAG_CP_HLOS_FREE = (1 << 23) # type: ignore
ION_FLAG_CP_SEC_DISPLAY = (1 << 25) # type: ignore
ION_FLAG_CP_APP = (1 << 26) # type: ignore
ION_FLAG_ALLOW_NON_CONTIG = (1 << 24) # type: ignore
ION_FLAG_SECURE = (1 << ION_HEAP_ID_RESERVED) # type: ignore
ION_FLAG_FORCE_CONTIGUOUS = (1 << 30) # type: ignore
ION_FLAG_POOL_FORCE_ALLOC = (1 << 16) # type: ignore
ION_FLAG_POOL_PREFETCH = (1 << 27) # type: ignore
ION_SECURE = ION_FLAG_SECURE # type: ignore
ION_FORCE_CONTIGUOUS = ION_FLAG_FORCE_CONTIGUOUS # type: ignore
ION_HEAP = lambda bit: (1 << (bit)) # type: ignore
ION_ADSP_HEAP_NAME = "adsp" # type: ignore
ION_SYSTEM_HEAP_NAME = "system" # type: ignore
ION_VMALLOC_HEAP_NAME = ION_SYSTEM_HEAP_NAME # type: ignore
ION_KMALLOC_HEAP_NAME = "kmalloc" # type: ignore
ION_AUDIO_HEAP_NAME = "audio" # type: ignore
ION_SF_HEAP_NAME = "sf" # type: ignore
ION_MM_HEAP_NAME = "mm" # type: ignore
ION_CAMERA_HEAP_NAME = "camera_preview" # type: ignore
ION_IOMMU_HEAP_NAME = "iommu" # type: ignore
ION_MFC_HEAP_NAME = "mfc" # type: ignore
ION_WB_HEAP_NAME = "wb" # type: ignore
ION_MM_FIRMWARE_HEAP_NAME = "mm_fw" # type: ignore
ION_PIL1_HEAP_NAME = "pil_1" # type: ignore
ION_PIL2_HEAP_NAME = "pil_2" # type: ignore
ION_QSECOM_HEAP_NAME = "qsecom" # type: ignore
ION_SECURE_HEAP_NAME = "secure_heap" # type: ignore
ION_SECURE_DISPLAY_HEAP_NAME = "secure_display" # type: ignore
ION_SET_CACHED = lambda __cache: (__cache | ION_FLAG_CACHED) # type: ignore
ION_SET_UNCACHED = lambda __cache: (__cache & ~ION_FLAG_CACHED) # type: ignore
ION_IS_CACHED = lambda __flags: ((__flags) & ION_FLAG_CACHED) # type: ignore
ION_IOC_MSM_MAGIC = 'M' # type: ignore
ION_IOC_CLEAN_CACHES = _IOWR(ION_IOC_MSM_MAGIC, 0, struct_ion_flush_data) # type: ignore
ION_IOC_INV_CACHES = _IOWR(ION_IOC_MSM_MAGIC, 1, struct_ion_flush_data) # type: ignore
ION_IOC_CLEAN_INV_CACHES = _IOWR(ION_IOC_MSM_MAGIC, 2, struct_ion_flush_data) # type: ignore
ION_IOC_PREFETCH = _IOWR(ION_IOC_MSM_MAGIC, 3, struct_ion_prefetch_data) # type: ignore
ION_IOC_DRAIN = _IOWR(ION_IOC_MSM_MAGIC, 4, struct_ion_prefetch_data) # type: ignore
FASTRPC_IOCTL_INVOKE = _IOWR('R', 1, struct_fastrpc_ioctl_invoke) # type: ignore
FASTRPC_IOCTL_MMAP = _IOWR('R', 2, struct_fastrpc_ioctl_mmap) # type: ignore
FASTRPC_IOCTL_MUNMAP = _IOWR('R', 3, struct_fastrpc_ioctl_munmap) # type: ignore
FASTRPC_IOCTL_MMAP_64 = _IOWR('R', 14, struct_fastrpc_ioctl_mmap_64) # type: ignore
FASTRPC_IOCTL_MUNMAP_64 = _IOWR('R', 15, struct_fastrpc_ioctl_munmap_64) # type: ignore
FASTRPC_IOCTL_INVOKE_FD = _IOWR('R', 4, struct_fastrpc_ioctl_invoke_fd) # type: ignore
FASTRPC_IOCTL_SETMODE = _IOWR('R', 5, uint32_t) # type: ignore
FASTRPC_IOCTL_INIT = _IOWR('R', 6, struct_fastrpc_ioctl_init) # type: ignore
FASTRPC_IOCTL_INVOKE_ATTRS = _IOWR('R', 7, struct_fastrpc_ioctl_invoke_attrs) # type: ignore
FASTRPC_IOCTL_GETINFO = _IOWR('R', 8, uint32_t) # type: ignore
FASTRPC_IOCTL_GETPERF = _IOWR('R', 9, struct_fastrpc_ioctl_perf) # type: ignore
FASTRPC_IOCTL_INIT_ATTRS = _IOWR('R', 10, struct_fastrpc_ioctl_init_attrs) # type: ignore
FASTRPC_IOCTL_INVOKE_CRC = _IOWR('R', 11, struct_fastrpc_ioctl_invoke_crc) # type: ignore
FASTRPC_IOCTL_CONTROL = _IOWR('R', 12, struct_fastrpc_ioctl_control) # type: ignore
FASTRPC_IOCTL_MUNMAP_FD = _IOWR('R', 13, struct_fastrpc_ioctl_munmap_fd) # type: ignore
FASTRPC_GLINK_GUID = "fastrpcglink-apps-dsp" # type: ignore
FASTRPC_SMD_GUID = "fastrpcsmd-apps-dsp" # type: ignore
DEVICE_NAME = "adsprpc-smd" # type: ignore
FASTRPC_ATTR_NOVA = 0x1 # type: ignore
FASTRPC_ATTR_NON_COHERENT = 0x2 # type: ignore
FASTRPC_ATTR_COHERENT = 0x4 # type: ignore
FASTRPC_ATTR_KEEP_MAP = 0x8 # type: ignore
FASTRPC_ATTR_NOMAP = (16) # type: ignore
FASTRPC_MODE_PARALLEL = 0 # type: ignore
FASTRPC_MODE_SERIAL = 1 # type: ignore
FASTRPC_MODE_PROFILE = 2 # type: ignore
FASTRPC_MODE_SESSION = 4 # type: ignore
FASTRPC_INIT_ATTACH = 0 # type: ignore
FASTRPC_INIT_CREATE = 1 # type: ignore
FASTRPC_INIT_CREATE_STATIC = 2 # type: ignore
FASTRPC_INIT_ATTACH_SENSORS = 3 # type: ignore
REMOTE_SCALARS_INBUFS = lambda sc: (((sc) >> 16) & 0x0ff) # type: ignore
REMOTE_SCALARS_OUTBUFS = lambda sc: (((sc) >> 8) & 0x0ff) # type: ignore
REMOTE_SCALARS_INHANDLES = lambda sc: (((sc) >> 4) & 0x0f) # type: ignore
REMOTE_SCALARS_OUTHANDLES = lambda sc: ((sc) & 0x0f) # type: ignore
REMOTE_SCALARS_LENGTH = lambda sc: (REMOTE_SCALARS_INBUFS(sc) + REMOTE_SCALARS_OUTBUFS(sc) + REMOTE_SCALARS_INHANDLES(sc) + REMOTE_SCALARS_OUTHANDLES(sc)) # type: ignore
__TOSTR__ = lambda x: __STR__(x) # type: ignore
remote_arg64_t = union_remote_arg64 # type: ignore
remote_arg_t = union_remote_arg # type: ignore
FASTRPC_CONTROL_LATENCY = (1) # type: ignore
FASTRPC_CONTROL_SMMU = (2) # type: ignore
FASTRPC_CONTROL_KALLOC = (3) # type: ignore
REMOTE_SCALARS_METHOD_ATTR = lambda dwScalars: (((dwScalars) >> 29) & 0x7) # type: ignore
REMOTE_SCALARS_METHOD = lambda dwScalars: (((dwScalars) >> 24) & 0x1f) # type: ignore
REMOTE_SCALARS_INBUFS = lambda dwScalars: (((dwScalars) >> 16) & 0x0ff) # type: ignore
REMOTE_SCALARS_OUTBUFS = lambda dwScalars: (((dwScalars) >> 8) & 0x0ff) # type: ignore
REMOTE_SCALARS_INHANDLES = lambda dwScalars: (((dwScalars) >> 4) & 0x0f) # type: ignore
REMOTE_SCALARS_OUTHANDLES = lambda dwScalars: ((dwScalars) & 0x0f) # type: ignore
REMOTE_SCALARS_MAKEX = lambda nAttr,nMethod,nIn,nOut,noIn,noOut: ((((uint32_t)   (nAttr) &  0x7) << 29) | (((uint32_t) (nMethod) & 0x1f) << 24) | (((uint32_t)     (nIn) & 0xff) << 16) | (((uint32_t)    (nOut) & 0xff) <<  8) | (((uint32_t)    (noIn) & 0x0f) <<  4) | ((uint32_t)   (noOut) & 0x0f)) # type: ignore
REMOTE_SCALARS_MAKE = lambda nMethod,nIn,nOut: REMOTE_SCALARS_MAKEX(0,nMethod,nIn,nOut,0,0) # type: ignore
REMOTE_SCALARS_LENGTH = lambda sc: (REMOTE_SCALARS_INBUFS(sc) + REMOTE_SCALARS_OUTBUFS(sc) + REMOTE_SCALARS_INHANDLES(sc) + REMOTE_SCALARS_OUTHANDLES(sc)) # type: ignore
__QAIC_REMOTE = lambda ff: ff # type: ignore
NUM_DOMAINS = 4 # type: ignore
NUM_SESSIONS = 2 # type: ignore
DOMAIN_ID_MASK = 3 # type: ignore
DEFAULT_DOMAIN_ID = 0 # type: ignore
ADSP_DOMAIN_ID = 0 # type: ignore
MDSP_DOMAIN_ID = 1 # type: ignore
SDSP_DOMAIN_ID = 2 # type: ignore
CDSP_DOMAIN_ID = 3 # type: ignore
ADSP_DOMAIN = "&_dom=adsp" # type: ignore
MDSP_DOMAIN = "&_dom=mdsp" # type: ignore
SDSP_DOMAIN = "&_dom=sdsp" # type: ignore
CDSP_DOMAIN = "&_dom=cdsp" # type: ignore
FASTRPC_WAKELOCK_CONTROL_SUPPORTED = 1 # type: ignore
REMOTE_MODE_PARALLEL = 0 # type: ignore
REMOTE_MODE_SERIAL = 1 # type: ignore
ITRANSPORT_PREFIX = "'\":;./\\" # type: ignore
__QAIC_HEADER = lambda ff: ff # type: ignore
__QAIC_IMPL = lambda ff: ff # type: ignore