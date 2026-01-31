# mypy: disable-error-code="empty-body"
from __future__ import annotations
import ctypes
from typing import Annotated, Literal, TypeAlias
from tinygrad.runtime.support.c import _IO, _IOW, _IOR, _IOWR
from tinygrad.runtime.support import c
drm_handle_t: TypeAlias = Annotated[int, ctypes.c_uint32]
drm_context_t: TypeAlias = Annotated[int, ctypes.c_uint32]
drm_drawable_t: TypeAlias = Annotated[int, ctypes.c_uint32]
drm_magic_t: TypeAlias = Annotated[int, ctypes.c_uint32]
@c.record
class struct_drm_clip_rect(c.Struct):
  SIZE = 8
  x1: Annotated[Annotated[int, ctypes.c_uint16], 0]
  y1: Annotated[Annotated[int, ctypes.c_uint16], 2]
  x2: Annotated[Annotated[int, ctypes.c_uint16], 4]
  y2: Annotated[Annotated[int, ctypes.c_uint16], 6]
@c.record
class struct_drm_drawable_info(c.Struct):
  SIZE = 16
  num_rects: Annotated[Annotated[int, ctypes.c_uint32], 0]
  rects: Annotated[c.POINTER[struct_drm_clip_rect], 8]
@c.record
class struct_drm_tex_region(c.Struct):
  SIZE = 8
  next: Annotated[Annotated[int, ctypes.c_ubyte], 0]
  prev: Annotated[Annotated[int, ctypes.c_ubyte], 1]
  in_use: Annotated[Annotated[int, ctypes.c_ubyte], 2]
  padding: Annotated[Annotated[int, ctypes.c_ubyte], 3]
  age: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_drm_hw_lock(c.Struct):
  SIZE = 64
  lock: Annotated[Annotated[int, ctypes.c_uint32], 0]
  padding: Annotated[c.Array[Annotated[bytes, ctypes.c_char], Literal[60]], 4]
@c.record
class struct_drm_version(c.Struct):
  SIZE = 64
  version_major: Annotated[Annotated[int, ctypes.c_int32], 0]
  version_minor: Annotated[Annotated[int, ctypes.c_int32], 4]
  version_patchlevel: Annotated[Annotated[int, ctypes.c_int32], 8]
  name_len: Annotated[Annotated[int, ctypes.c_uint64], 16]
  name: Annotated[c.POINTER[Annotated[bytes, ctypes.c_char]], 24]
  date_len: Annotated[Annotated[int, ctypes.c_uint64], 32]
  date: Annotated[c.POINTER[Annotated[bytes, ctypes.c_char]], 40]
  desc_len: Annotated[Annotated[int, ctypes.c_uint64], 48]
  desc: Annotated[c.POINTER[Annotated[bytes, ctypes.c_char]], 56]
__kernel_size_t: TypeAlias = Annotated[int, ctypes.c_uint64]
@c.record
class struct_drm_unique(c.Struct):
  SIZE = 16
  unique_len: Annotated[Annotated[int, ctypes.c_uint64], 0]
  unique: Annotated[c.POINTER[Annotated[bytes, ctypes.c_char]], 8]
@c.record
class struct_drm_list(c.Struct):
  SIZE = 16
  count: Annotated[Annotated[int, ctypes.c_int32], 0]
  version: Annotated[c.POINTER[struct_drm_version], 8]
@c.record
class struct_drm_block(c.Struct):
  SIZE = 4
  unused: Annotated[Annotated[int, ctypes.c_int32], 0]
@c.record
class struct_drm_control(c.Struct):
  SIZE = 8
  func: Annotated[struct_drm_control_func, 0]
  irq: Annotated[Annotated[int, ctypes.c_int32], 4]
class struct_drm_control_func(Annotated[int, ctypes.c_uint32], c.Enum): pass
DRM_ADD_COMMAND = struct_drm_control_func.define('DRM_ADD_COMMAND', 0)
DRM_RM_COMMAND = struct_drm_control_func.define('DRM_RM_COMMAND', 1)
DRM_INST_HANDLER = struct_drm_control_func.define('DRM_INST_HANDLER', 2)
DRM_UNINST_HANDLER = struct_drm_control_func.define('DRM_UNINST_HANDLER', 3)

class enum_drm_map_type(Annotated[int, ctypes.c_uint32], c.Enum): pass
_DRM_FRAME_BUFFER = enum_drm_map_type.define('_DRM_FRAME_BUFFER', 0)
_DRM_REGISTERS = enum_drm_map_type.define('_DRM_REGISTERS', 1)
_DRM_SHM = enum_drm_map_type.define('_DRM_SHM', 2)
_DRM_AGP = enum_drm_map_type.define('_DRM_AGP', 3)
_DRM_SCATTER_GATHER = enum_drm_map_type.define('_DRM_SCATTER_GATHER', 4)
_DRM_CONSISTENT = enum_drm_map_type.define('_DRM_CONSISTENT', 5)

class enum_drm_map_flags(Annotated[int, ctypes.c_uint32], c.Enum): pass
_DRM_RESTRICTED = enum_drm_map_flags.define('_DRM_RESTRICTED', 1)
_DRM_READ_ONLY = enum_drm_map_flags.define('_DRM_READ_ONLY', 2)
_DRM_LOCKED = enum_drm_map_flags.define('_DRM_LOCKED', 4)
_DRM_KERNEL = enum_drm_map_flags.define('_DRM_KERNEL', 8)
_DRM_WRITE_COMBINING = enum_drm_map_flags.define('_DRM_WRITE_COMBINING', 16)
_DRM_CONTAINS_LOCK = enum_drm_map_flags.define('_DRM_CONTAINS_LOCK', 32)
_DRM_REMOVABLE = enum_drm_map_flags.define('_DRM_REMOVABLE', 64)
_DRM_DRIVER = enum_drm_map_flags.define('_DRM_DRIVER', 128)

@c.record
class struct_drm_ctx_priv_map(c.Struct):
  SIZE = 16
  ctx_id: Annotated[Annotated[int, ctypes.c_uint32], 0]
  handle: Annotated[ctypes.c_void_p, 8]
@c.record
class struct_drm_map(c.Struct):
  SIZE = 40
  offset: Annotated[Annotated[int, ctypes.c_uint64], 0]
  size: Annotated[Annotated[int, ctypes.c_uint64], 8]
  type: Annotated[enum_drm_map_type, 16]
  flags: Annotated[enum_drm_map_flags, 20]
  handle: Annotated[ctypes.c_void_p, 24]
  mtrr: Annotated[Annotated[int, ctypes.c_int32], 32]
@c.record
class struct_drm_client(c.Struct):
  SIZE = 40
  idx: Annotated[Annotated[int, ctypes.c_int32], 0]
  auth: Annotated[Annotated[int, ctypes.c_int32], 4]
  pid: Annotated[Annotated[int, ctypes.c_uint64], 8]
  uid: Annotated[Annotated[int, ctypes.c_uint64], 16]
  magic: Annotated[Annotated[int, ctypes.c_uint64], 24]
  iocs: Annotated[Annotated[int, ctypes.c_uint64], 32]
class enum_drm_stat_type(Annotated[int, ctypes.c_uint32], c.Enum): pass
_DRM_STAT_LOCK = enum_drm_stat_type.define('_DRM_STAT_LOCK', 0)
_DRM_STAT_OPENS = enum_drm_stat_type.define('_DRM_STAT_OPENS', 1)
_DRM_STAT_CLOSES = enum_drm_stat_type.define('_DRM_STAT_CLOSES', 2)
_DRM_STAT_IOCTLS = enum_drm_stat_type.define('_DRM_STAT_IOCTLS', 3)
_DRM_STAT_LOCKS = enum_drm_stat_type.define('_DRM_STAT_LOCKS', 4)
_DRM_STAT_UNLOCKS = enum_drm_stat_type.define('_DRM_STAT_UNLOCKS', 5)
_DRM_STAT_VALUE = enum_drm_stat_type.define('_DRM_STAT_VALUE', 6)
_DRM_STAT_BYTE = enum_drm_stat_type.define('_DRM_STAT_BYTE', 7)
_DRM_STAT_COUNT = enum_drm_stat_type.define('_DRM_STAT_COUNT', 8)
_DRM_STAT_IRQ = enum_drm_stat_type.define('_DRM_STAT_IRQ', 9)
_DRM_STAT_PRIMARY = enum_drm_stat_type.define('_DRM_STAT_PRIMARY', 10)
_DRM_STAT_SECONDARY = enum_drm_stat_type.define('_DRM_STAT_SECONDARY', 11)
_DRM_STAT_DMA = enum_drm_stat_type.define('_DRM_STAT_DMA', 12)
_DRM_STAT_SPECIAL = enum_drm_stat_type.define('_DRM_STAT_SPECIAL', 13)
_DRM_STAT_MISSED = enum_drm_stat_type.define('_DRM_STAT_MISSED', 14)

@c.record
class struct_drm_stats(c.Struct):
  SIZE = 248
  count: Annotated[Annotated[int, ctypes.c_uint64], 0]
  data: Annotated[c.Array[struct_drm_stats_data, Literal[15]], 8]
@c.record
class struct_drm_stats_data(c.Struct):
  SIZE = 16
  value: Annotated[Annotated[int, ctypes.c_uint64], 0]
  type: Annotated[enum_drm_stat_type, 8]
class enum_drm_lock_flags(Annotated[int, ctypes.c_uint32], c.Enum): pass
_DRM_LOCK_READY = enum_drm_lock_flags.define('_DRM_LOCK_READY', 1)
_DRM_LOCK_QUIESCENT = enum_drm_lock_flags.define('_DRM_LOCK_QUIESCENT', 2)
_DRM_LOCK_FLUSH = enum_drm_lock_flags.define('_DRM_LOCK_FLUSH', 4)
_DRM_LOCK_FLUSH_ALL = enum_drm_lock_flags.define('_DRM_LOCK_FLUSH_ALL', 8)
_DRM_HALT_ALL_QUEUES = enum_drm_lock_flags.define('_DRM_HALT_ALL_QUEUES', 16)
_DRM_HALT_CUR_QUEUES = enum_drm_lock_flags.define('_DRM_HALT_CUR_QUEUES', 32)

@c.record
class struct_drm_lock(c.Struct):
  SIZE = 8
  context: Annotated[Annotated[int, ctypes.c_int32], 0]
  flags: Annotated[enum_drm_lock_flags, 4]
class enum_drm_dma_flags(Annotated[int, ctypes.c_uint32], c.Enum): pass
_DRM_DMA_BLOCK = enum_drm_dma_flags.define('_DRM_DMA_BLOCK', 1)
_DRM_DMA_WHILE_LOCKED = enum_drm_dma_flags.define('_DRM_DMA_WHILE_LOCKED', 2)
_DRM_DMA_PRIORITY = enum_drm_dma_flags.define('_DRM_DMA_PRIORITY', 4)
_DRM_DMA_WAIT = enum_drm_dma_flags.define('_DRM_DMA_WAIT', 16)
_DRM_DMA_SMALLER_OK = enum_drm_dma_flags.define('_DRM_DMA_SMALLER_OK', 32)
_DRM_DMA_LARGER_OK = enum_drm_dma_flags.define('_DRM_DMA_LARGER_OK', 64)

@c.record
class struct_drm_buf_desc(c.Struct):
  SIZE = 32
  count: Annotated[Annotated[int, ctypes.c_int32], 0]
  size: Annotated[Annotated[int, ctypes.c_int32], 4]
  low_mark: Annotated[Annotated[int, ctypes.c_int32], 8]
  high_mark: Annotated[Annotated[int, ctypes.c_int32], 12]
  flags: Annotated[struct_drm_buf_desc_flags, 16]
  agp_start: Annotated[Annotated[int, ctypes.c_uint64], 24]
class struct_drm_buf_desc_flags(Annotated[int, ctypes.c_uint32], c.Enum): pass
_DRM_PAGE_ALIGN = struct_drm_buf_desc_flags.define('_DRM_PAGE_ALIGN', 1)
_DRM_AGP_BUFFER = struct_drm_buf_desc_flags.define('_DRM_AGP_BUFFER', 2)
_DRM_SG_BUFFER = struct_drm_buf_desc_flags.define('_DRM_SG_BUFFER', 4)
_DRM_FB_BUFFER = struct_drm_buf_desc_flags.define('_DRM_FB_BUFFER', 8)
_DRM_PCI_BUFFER_RO = struct_drm_buf_desc_flags.define('_DRM_PCI_BUFFER_RO', 16)

@c.record
class struct_drm_buf_info(c.Struct):
  SIZE = 16
  count: Annotated[Annotated[int, ctypes.c_int32], 0]
  list: Annotated[c.POINTER[struct_drm_buf_desc], 8]
@c.record
class struct_drm_buf_free(c.Struct):
  SIZE = 16
  count: Annotated[Annotated[int, ctypes.c_int32], 0]
  list: Annotated[c.POINTER[Annotated[int, ctypes.c_int32]], 8]
@c.record
class struct_drm_buf_pub(c.Struct):
  SIZE = 24
  idx: Annotated[Annotated[int, ctypes.c_int32], 0]
  total: Annotated[Annotated[int, ctypes.c_int32], 4]
  used: Annotated[Annotated[int, ctypes.c_int32], 8]
  address: Annotated[ctypes.c_void_p, 16]
@c.record
class struct_drm_buf_map(c.Struct):
  SIZE = 24
  count: Annotated[Annotated[int, ctypes.c_int32], 0]
  virtual: Annotated[ctypes.c_void_p, 8]
  list: Annotated[c.POINTER[struct_drm_buf_pub], 16]
@c.record
class struct_drm_dma(c.Struct):
  SIZE = 64
  context: Annotated[Annotated[int, ctypes.c_int32], 0]
  send_count: Annotated[Annotated[int, ctypes.c_int32], 4]
  send_indices: Annotated[c.POINTER[Annotated[int, ctypes.c_int32]], 8]
  send_sizes: Annotated[c.POINTER[Annotated[int, ctypes.c_int32]], 16]
  flags: Annotated[enum_drm_dma_flags, 24]
  request_count: Annotated[Annotated[int, ctypes.c_int32], 28]
  request_size: Annotated[Annotated[int, ctypes.c_int32], 32]
  request_indices: Annotated[c.POINTER[Annotated[int, ctypes.c_int32]], 40]
  request_sizes: Annotated[c.POINTER[Annotated[int, ctypes.c_int32]], 48]
  granted_count: Annotated[Annotated[int, ctypes.c_int32], 56]
class enum_drm_ctx_flags(Annotated[int, ctypes.c_uint32], c.Enum): pass
_DRM_CONTEXT_PRESERVED = enum_drm_ctx_flags.define('_DRM_CONTEXT_PRESERVED', 1)
_DRM_CONTEXT_2DONLY = enum_drm_ctx_flags.define('_DRM_CONTEXT_2DONLY', 2)

@c.record
class struct_drm_ctx(c.Struct):
  SIZE = 8
  handle: Annotated[drm_context_t, 0]
  flags: Annotated[enum_drm_ctx_flags, 4]
@c.record
class struct_drm_ctx_res(c.Struct):
  SIZE = 16
  count: Annotated[Annotated[int, ctypes.c_int32], 0]
  contexts: Annotated[c.POINTER[struct_drm_ctx], 8]
@c.record
class struct_drm_draw(c.Struct):
  SIZE = 4
  handle: Annotated[drm_drawable_t, 0]
class drm_drawable_info_type_t(Annotated[int, ctypes.c_uint32], c.Enum): pass
DRM_DRAWABLE_CLIPRECTS = drm_drawable_info_type_t.define('DRM_DRAWABLE_CLIPRECTS', 0)

@c.record
class struct_drm_update_draw(c.Struct):
  SIZE = 24
  handle: Annotated[drm_drawable_t, 0]
  type: Annotated[Annotated[int, ctypes.c_uint32], 4]
  num: Annotated[Annotated[int, ctypes.c_uint32], 8]
  data: Annotated[Annotated[int, ctypes.c_uint64], 16]
@c.record
class struct_drm_auth(c.Struct):
  SIZE = 4
  magic: Annotated[drm_magic_t, 0]
@c.record
class struct_drm_irq_busid(c.Struct):
  SIZE = 16
  irq: Annotated[Annotated[int, ctypes.c_int32], 0]
  busnum: Annotated[Annotated[int, ctypes.c_int32], 4]
  devnum: Annotated[Annotated[int, ctypes.c_int32], 8]
  funcnum: Annotated[Annotated[int, ctypes.c_int32], 12]
class enum_drm_vblank_seq_type(Annotated[int, ctypes.c_uint32], c.Enum): pass
_DRM_VBLANK_ABSOLUTE = enum_drm_vblank_seq_type.define('_DRM_VBLANK_ABSOLUTE', 0)
_DRM_VBLANK_RELATIVE = enum_drm_vblank_seq_type.define('_DRM_VBLANK_RELATIVE', 1)
_DRM_VBLANK_HIGH_CRTC_MASK = enum_drm_vblank_seq_type.define('_DRM_VBLANK_HIGH_CRTC_MASK', 62)
_DRM_VBLANK_EVENT = enum_drm_vblank_seq_type.define('_DRM_VBLANK_EVENT', 67108864)
_DRM_VBLANK_FLIP = enum_drm_vblank_seq_type.define('_DRM_VBLANK_FLIP', 134217728)
_DRM_VBLANK_NEXTONMISS = enum_drm_vblank_seq_type.define('_DRM_VBLANK_NEXTONMISS', 268435456)
_DRM_VBLANK_SECONDARY = enum_drm_vblank_seq_type.define('_DRM_VBLANK_SECONDARY', 536870912)
_DRM_VBLANK_SIGNAL = enum_drm_vblank_seq_type.define('_DRM_VBLANK_SIGNAL', 1073741824)

@c.record
class struct_drm_wait_vblank_request(c.Struct):
  SIZE = 16
  type: Annotated[enum_drm_vblank_seq_type, 0]
  sequence: Annotated[Annotated[int, ctypes.c_uint32], 4]
  signal: Annotated[Annotated[int, ctypes.c_uint64], 8]
@c.record
class struct_drm_wait_vblank_reply(c.Struct):
  SIZE = 24
  type: Annotated[enum_drm_vblank_seq_type, 0]
  sequence: Annotated[Annotated[int, ctypes.c_uint32], 4]
  tval_sec: Annotated[Annotated[int, ctypes.c_int64], 8]
  tval_usec: Annotated[Annotated[int, ctypes.c_int64], 16]
@c.record
class union_drm_wait_vblank(c.Struct):
  SIZE = 24
  request: Annotated[struct_drm_wait_vblank_request, 0]
  reply: Annotated[struct_drm_wait_vblank_reply, 0]
@c.record
class struct_drm_modeset_ctl(c.Struct):
  SIZE = 8
  crtc: Annotated[Annotated[int, ctypes.c_uint32], 0]
  cmd: Annotated[Annotated[int, ctypes.c_uint32], 4]
__u32: TypeAlias = Annotated[int, ctypes.c_uint32]
@c.record
class struct_drm_agp_mode(c.Struct):
  SIZE = 8
  mode: Annotated[Annotated[int, ctypes.c_uint64], 0]
@c.record
class struct_drm_agp_buffer(c.Struct):
  SIZE = 32
  size: Annotated[Annotated[int, ctypes.c_uint64], 0]
  handle: Annotated[Annotated[int, ctypes.c_uint64], 8]
  type: Annotated[Annotated[int, ctypes.c_uint64], 16]
  physical: Annotated[Annotated[int, ctypes.c_uint64], 24]
@c.record
class struct_drm_agp_binding(c.Struct):
  SIZE = 16
  handle: Annotated[Annotated[int, ctypes.c_uint64], 0]
  offset: Annotated[Annotated[int, ctypes.c_uint64], 8]
@c.record
class struct_drm_agp_info(c.Struct):
  SIZE = 56
  agp_version_major: Annotated[Annotated[int, ctypes.c_int32], 0]
  agp_version_minor: Annotated[Annotated[int, ctypes.c_int32], 4]
  mode: Annotated[Annotated[int, ctypes.c_uint64], 8]
  aperture_base: Annotated[Annotated[int, ctypes.c_uint64], 16]
  aperture_size: Annotated[Annotated[int, ctypes.c_uint64], 24]
  memory_allowed: Annotated[Annotated[int, ctypes.c_uint64], 32]
  memory_used: Annotated[Annotated[int, ctypes.c_uint64], 40]
  id_vendor: Annotated[Annotated[int, ctypes.c_uint16], 48]
  id_device: Annotated[Annotated[int, ctypes.c_uint16], 50]
@c.record
class struct_drm_scatter_gather(c.Struct):
  SIZE = 16
  size: Annotated[Annotated[int, ctypes.c_uint64], 0]
  handle: Annotated[Annotated[int, ctypes.c_uint64], 8]
@c.record
class struct_drm_set_version(c.Struct):
  SIZE = 16
  drm_di_major: Annotated[Annotated[int, ctypes.c_int32], 0]
  drm_di_minor: Annotated[Annotated[int, ctypes.c_int32], 4]
  drm_dd_major: Annotated[Annotated[int, ctypes.c_int32], 8]
  drm_dd_minor: Annotated[Annotated[int, ctypes.c_int32], 12]
@c.record
class struct_drm_gem_close(c.Struct):
  SIZE = 8
  handle: Annotated[Annotated[int, ctypes.c_uint32], 0]
  pad: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_drm_gem_flink(c.Struct):
  SIZE = 8
  handle: Annotated[Annotated[int, ctypes.c_uint32], 0]
  name: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_drm_gem_open(c.Struct):
  SIZE = 16
  name: Annotated[Annotated[int, ctypes.c_uint32], 0]
  handle: Annotated[Annotated[int, ctypes.c_uint32], 4]
  size: Annotated[Annotated[int, ctypes.c_uint64], 8]
__u64: TypeAlias = Annotated[int, ctypes.c_uint64]
@c.record
class struct_drm_get_cap(c.Struct):
  SIZE = 16
  capability: Annotated[Annotated[int, ctypes.c_uint64], 0]
  value: Annotated[Annotated[int, ctypes.c_uint64], 8]
@c.record
class struct_drm_set_client_cap(c.Struct):
  SIZE = 16
  capability: Annotated[Annotated[int, ctypes.c_uint64], 0]
  value: Annotated[Annotated[int, ctypes.c_uint64], 8]
@c.record
class struct_drm_prime_handle(c.Struct):
  SIZE = 12
  handle: Annotated[Annotated[int, ctypes.c_uint32], 0]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 4]
  fd: Annotated[Annotated[int, ctypes.c_int32], 8]
__s32: TypeAlias = Annotated[int, ctypes.c_int32]
@c.record
class struct_drm_syncobj_create(c.Struct):
  SIZE = 8
  handle: Annotated[Annotated[int, ctypes.c_uint32], 0]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_drm_syncobj_destroy(c.Struct):
  SIZE = 8
  handle: Annotated[Annotated[int, ctypes.c_uint32], 0]
  pad: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_drm_syncobj_handle(c.Struct):
  SIZE = 16
  handle: Annotated[Annotated[int, ctypes.c_uint32], 0]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 4]
  fd: Annotated[Annotated[int, ctypes.c_int32], 8]
  pad: Annotated[Annotated[int, ctypes.c_uint32], 12]
@c.record
class struct_drm_syncobj_transfer(c.Struct):
  SIZE = 32
  src_handle: Annotated[Annotated[int, ctypes.c_uint32], 0]
  dst_handle: Annotated[Annotated[int, ctypes.c_uint32], 4]
  src_point: Annotated[Annotated[int, ctypes.c_uint64], 8]
  dst_point: Annotated[Annotated[int, ctypes.c_uint64], 16]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 24]
  pad: Annotated[Annotated[int, ctypes.c_uint32], 28]
@c.record
class struct_drm_syncobj_wait(c.Struct):
  SIZE = 40
  handles: Annotated[Annotated[int, ctypes.c_uint64], 0]
  timeout_nsec: Annotated[Annotated[int, ctypes.c_int64], 8]
  count_handles: Annotated[Annotated[int, ctypes.c_uint32], 16]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 20]
  first_signaled: Annotated[Annotated[int, ctypes.c_uint32], 24]
  pad: Annotated[Annotated[int, ctypes.c_uint32], 28]
  deadline_nsec: Annotated[Annotated[int, ctypes.c_uint64], 32]
__s64: TypeAlias = Annotated[int, ctypes.c_int64]
@c.record
class struct_drm_syncobj_timeline_wait(c.Struct):
  SIZE = 48
  handles: Annotated[Annotated[int, ctypes.c_uint64], 0]
  points: Annotated[Annotated[int, ctypes.c_uint64], 8]
  timeout_nsec: Annotated[Annotated[int, ctypes.c_int64], 16]
  count_handles: Annotated[Annotated[int, ctypes.c_uint32], 24]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 28]
  first_signaled: Annotated[Annotated[int, ctypes.c_uint32], 32]
  pad: Annotated[Annotated[int, ctypes.c_uint32], 36]
  deadline_nsec: Annotated[Annotated[int, ctypes.c_uint64], 40]
@c.record
class struct_drm_syncobj_eventfd(c.Struct):
  SIZE = 24
  handle: Annotated[Annotated[int, ctypes.c_uint32], 0]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 4]
  point: Annotated[Annotated[int, ctypes.c_uint64], 8]
  fd: Annotated[Annotated[int, ctypes.c_int32], 16]
  pad: Annotated[Annotated[int, ctypes.c_uint32], 20]
@c.record
class struct_drm_syncobj_array(c.Struct):
  SIZE = 16
  handles: Annotated[Annotated[int, ctypes.c_uint64], 0]
  count_handles: Annotated[Annotated[int, ctypes.c_uint32], 8]
  pad: Annotated[Annotated[int, ctypes.c_uint32], 12]
@c.record
class struct_drm_syncobj_timeline_array(c.Struct):
  SIZE = 24
  handles: Annotated[Annotated[int, ctypes.c_uint64], 0]
  points: Annotated[Annotated[int, ctypes.c_uint64], 8]
  count_handles: Annotated[Annotated[int, ctypes.c_uint32], 16]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 20]
@c.record
class struct_drm_crtc_get_sequence(c.Struct):
  SIZE = 24
  crtc_id: Annotated[Annotated[int, ctypes.c_uint32], 0]
  active: Annotated[Annotated[int, ctypes.c_uint32], 4]
  sequence: Annotated[Annotated[int, ctypes.c_uint64], 8]
  sequence_ns: Annotated[Annotated[int, ctypes.c_int64], 16]
@c.record
class struct_drm_crtc_queue_sequence(c.Struct):
  SIZE = 24
  crtc_id: Annotated[Annotated[int, ctypes.c_uint32], 0]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 4]
  sequence: Annotated[Annotated[int, ctypes.c_uint64], 8]
  user_data: Annotated[Annotated[int, ctypes.c_uint64], 16]
@c.record
class struct_drm_event(c.Struct):
  SIZE = 8
  type: Annotated[Annotated[int, ctypes.c_uint32], 0]
  length: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_drm_event_vblank(c.Struct):
  SIZE = 32
  base: Annotated[struct_drm_event, 0]
  user_data: Annotated[Annotated[int, ctypes.c_uint64], 8]
  tv_sec: Annotated[Annotated[int, ctypes.c_uint32], 16]
  tv_usec: Annotated[Annotated[int, ctypes.c_uint32], 20]
  sequence: Annotated[Annotated[int, ctypes.c_uint32], 24]
  crtc_id: Annotated[Annotated[int, ctypes.c_uint32], 28]
@c.record
class struct_drm_event_crtc_sequence(c.Struct):
  SIZE = 32
  base: Annotated[struct_drm_event, 0]
  user_data: Annotated[Annotated[int, ctypes.c_uint64], 8]
  time_ns: Annotated[Annotated[int, ctypes.c_int64], 16]
  sequence: Annotated[Annotated[int, ctypes.c_uint64], 24]
drm_clip_rect_t: TypeAlias = struct_drm_clip_rect
drm_drawable_info_t: TypeAlias = struct_drm_drawable_info
drm_tex_region_t: TypeAlias = struct_drm_tex_region
drm_hw_lock_t: TypeAlias = struct_drm_hw_lock
drm_version_t: TypeAlias = struct_drm_version
drm_unique_t: TypeAlias = struct_drm_unique
drm_list_t: TypeAlias = struct_drm_list
drm_block_t: TypeAlias = struct_drm_block
drm_control_t: TypeAlias = struct_drm_control
drm_map_type_t: TypeAlias = enum_drm_map_type
drm_map_flags_t: TypeAlias = enum_drm_map_flags
drm_ctx_priv_map_t: TypeAlias = struct_drm_ctx_priv_map
drm_map_t: TypeAlias = struct_drm_map
drm_client_t: TypeAlias = struct_drm_client
drm_stat_type_t: TypeAlias = enum_drm_stat_type
drm_stats_t: TypeAlias = struct_drm_stats
drm_lock_flags_t: TypeAlias = enum_drm_lock_flags
drm_lock_t: TypeAlias = struct_drm_lock
drm_dma_flags_t: TypeAlias = enum_drm_dma_flags
drm_buf_desc_t: TypeAlias = struct_drm_buf_desc
drm_buf_info_t: TypeAlias = struct_drm_buf_info
drm_buf_free_t: TypeAlias = struct_drm_buf_free
drm_buf_pub_t: TypeAlias = struct_drm_buf_pub
drm_buf_map_t: TypeAlias = struct_drm_buf_map
drm_dma_t: TypeAlias = struct_drm_dma
drm_wait_vblank_t: TypeAlias = union_drm_wait_vblank
drm_agp_mode_t: TypeAlias = struct_drm_agp_mode
drm_ctx_flags_t: TypeAlias = enum_drm_ctx_flags
drm_ctx_t: TypeAlias = struct_drm_ctx
drm_ctx_res_t: TypeAlias = struct_drm_ctx_res
drm_draw_t: TypeAlias = struct_drm_draw
drm_update_draw_t: TypeAlias = struct_drm_update_draw
drm_auth_t: TypeAlias = struct_drm_auth
drm_irq_busid_t: TypeAlias = struct_drm_irq_busid
drm_vblank_seq_type_t: TypeAlias = enum_drm_vblank_seq_type
drm_agp_buffer_t: TypeAlias = struct_drm_agp_buffer
drm_agp_binding_t: TypeAlias = struct_drm_agp_binding
drm_agp_info_t: TypeAlias = struct_drm_agp_info
drm_scatter_gather_t: TypeAlias = struct_drm_scatter_gather
drm_set_version_t: TypeAlias = struct_drm_set_version
@c.record
class struct_drm_amdgpu_gem_create_in(c.Struct):
  SIZE = 32
  bo_size: Annotated[Annotated[int, ctypes.c_uint64], 0]
  alignment: Annotated[Annotated[int, ctypes.c_uint64], 8]
  domains: Annotated[Annotated[int, ctypes.c_uint64], 16]
  domain_flags: Annotated[Annotated[int, ctypes.c_uint64], 24]
@c.record
class struct_drm_amdgpu_gem_create_out(c.Struct):
  SIZE = 8
  handle: Annotated[Annotated[int, ctypes.c_uint32], 0]
  _pad: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class union_drm_amdgpu_gem_create(c.Struct):
  SIZE = 32
  _in: Annotated[struct_drm_amdgpu_gem_create_in, 0]
  out: Annotated[struct_drm_amdgpu_gem_create_out, 0]
@c.record
class struct_drm_amdgpu_bo_list_in(c.Struct):
  SIZE = 24
  operation: Annotated[Annotated[int, ctypes.c_uint32], 0]
  list_handle: Annotated[Annotated[int, ctypes.c_uint32], 4]
  bo_number: Annotated[Annotated[int, ctypes.c_uint32], 8]
  bo_info_size: Annotated[Annotated[int, ctypes.c_uint32], 12]
  bo_info_ptr: Annotated[Annotated[int, ctypes.c_uint64], 16]
@c.record
class struct_drm_amdgpu_bo_list_entry(c.Struct):
  SIZE = 8
  bo_handle: Annotated[Annotated[int, ctypes.c_uint32], 0]
  bo_priority: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_drm_amdgpu_bo_list_out(c.Struct):
  SIZE = 8
  list_handle: Annotated[Annotated[int, ctypes.c_uint32], 0]
  _pad: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class union_drm_amdgpu_bo_list(c.Struct):
  SIZE = 24
  _in: Annotated[struct_drm_amdgpu_bo_list_in, 0]
  out: Annotated[struct_drm_amdgpu_bo_list_out, 0]
@c.record
class struct_drm_amdgpu_ctx_in(c.Struct):
  SIZE = 16
  op: Annotated[Annotated[int, ctypes.c_uint32], 0]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 4]
  ctx_id: Annotated[Annotated[int, ctypes.c_uint32], 8]
  priority: Annotated[Annotated[int, ctypes.c_int32], 12]
@c.record
class union_drm_amdgpu_ctx_out(c.Struct):
  SIZE = 16
  alloc: Annotated[union_drm_amdgpu_ctx_out_alloc, 0]
  state: Annotated[union_drm_amdgpu_ctx_out_state, 0]
  pstate: Annotated[union_drm_amdgpu_ctx_out_pstate, 0]
@c.record
class union_drm_amdgpu_ctx_out_alloc(c.Struct):
  SIZE = 8
  ctx_id: Annotated[Annotated[int, ctypes.c_uint32], 0]
  _pad: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class union_drm_amdgpu_ctx_out_state(c.Struct):
  SIZE = 16
  flags: Annotated[Annotated[int, ctypes.c_uint64], 0]
  hangs: Annotated[Annotated[int, ctypes.c_uint32], 8]
  reset_status: Annotated[Annotated[int, ctypes.c_uint32], 12]
@c.record
class union_drm_amdgpu_ctx_out_pstate(c.Struct):
  SIZE = 8
  flags: Annotated[Annotated[int, ctypes.c_uint32], 0]
  _pad: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class union_drm_amdgpu_ctx(c.Struct):
  SIZE = 16
  _in: Annotated[struct_drm_amdgpu_ctx_in, 0]
  out: Annotated[union_drm_amdgpu_ctx_out, 0]
@c.record
class struct_drm_amdgpu_userq_in(c.Struct):
  SIZE = 72
  op: Annotated[Annotated[int, ctypes.c_uint32], 0]
  queue_id: Annotated[Annotated[int, ctypes.c_uint32], 4]
  ip_type: Annotated[Annotated[int, ctypes.c_uint32], 8]
  doorbell_handle: Annotated[Annotated[int, ctypes.c_uint32], 12]
  doorbell_offset: Annotated[Annotated[int, ctypes.c_uint32], 16]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 20]
  queue_va: Annotated[Annotated[int, ctypes.c_uint64], 24]
  queue_size: Annotated[Annotated[int, ctypes.c_uint64], 32]
  rptr_va: Annotated[Annotated[int, ctypes.c_uint64], 40]
  wptr_va: Annotated[Annotated[int, ctypes.c_uint64], 48]
  mqd: Annotated[Annotated[int, ctypes.c_uint64], 56]
  mqd_size: Annotated[Annotated[int, ctypes.c_uint64], 64]
@c.record
class struct_drm_amdgpu_userq_out(c.Struct):
  SIZE = 8
  queue_id: Annotated[Annotated[int, ctypes.c_uint32], 0]
  _pad: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class union_drm_amdgpu_userq(c.Struct):
  SIZE = 72
  _in: Annotated[struct_drm_amdgpu_userq_in, 0]
  out: Annotated[struct_drm_amdgpu_userq_out, 0]
@c.record
class struct_drm_amdgpu_userq_mqd_gfx11(c.Struct):
  SIZE = 16
  shadow_va: Annotated[Annotated[int, ctypes.c_uint64], 0]
  csa_va: Annotated[Annotated[int, ctypes.c_uint64], 8]
@c.record
class struct_drm_amdgpu_userq_mqd_sdma_gfx11(c.Struct):
  SIZE = 8
  csa_va: Annotated[Annotated[int, ctypes.c_uint64], 0]
@c.record
class struct_drm_amdgpu_userq_mqd_compute_gfx11(c.Struct):
  SIZE = 8
  eop_va: Annotated[Annotated[int, ctypes.c_uint64], 0]
@c.record
class struct_drm_amdgpu_userq_signal(c.Struct):
  SIZE = 48
  queue_id: Annotated[Annotated[int, ctypes.c_uint32], 0]
  pad: Annotated[Annotated[int, ctypes.c_uint32], 4]
  syncobj_handles: Annotated[Annotated[int, ctypes.c_uint64], 8]
  num_syncobj_handles: Annotated[Annotated[int, ctypes.c_uint64], 16]
  bo_read_handles: Annotated[Annotated[int, ctypes.c_uint64], 24]
  bo_write_handles: Annotated[Annotated[int, ctypes.c_uint64], 32]
  num_bo_read_handles: Annotated[Annotated[int, ctypes.c_uint32], 40]
  num_bo_write_handles: Annotated[Annotated[int, ctypes.c_uint32], 44]
@c.record
class struct_drm_amdgpu_userq_fence_info(c.Struct):
  SIZE = 16
  va: Annotated[Annotated[int, ctypes.c_uint64], 0]
  value: Annotated[Annotated[int, ctypes.c_uint64], 8]
@c.record
class struct_drm_amdgpu_userq_wait(c.Struct):
  SIZE = 72
  waitq_id: Annotated[Annotated[int, ctypes.c_uint32], 0]
  pad: Annotated[Annotated[int, ctypes.c_uint32], 4]
  syncobj_handles: Annotated[Annotated[int, ctypes.c_uint64], 8]
  syncobj_timeline_handles: Annotated[Annotated[int, ctypes.c_uint64], 16]
  syncobj_timeline_points: Annotated[Annotated[int, ctypes.c_uint64], 24]
  bo_read_handles: Annotated[Annotated[int, ctypes.c_uint64], 32]
  bo_write_handles: Annotated[Annotated[int, ctypes.c_uint64], 40]
  num_syncobj_timeline_handles: Annotated[Annotated[int, ctypes.c_uint16], 48]
  num_fences: Annotated[Annotated[int, ctypes.c_uint16], 50]
  num_syncobj_handles: Annotated[Annotated[int, ctypes.c_uint32], 52]
  num_bo_read_handles: Annotated[Annotated[int, ctypes.c_uint32], 56]
  num_bo_write_handles: Annotated[Annotated[int, ctypes.c_uint32], 60]
  out_fences: Annotated[Annotated[int, ctypes.c_uint64], 64]
__u16: TypeAlias = Annotated[int, ctypes.c_uint16]
class struct_drm_amdgpu_sem_in(ctypes.Structure): pass
class union_drm_amdgpu_sem_out(ctypes.Union): pass
class union_drm_amdgpu_sem(ctypes.Union): pass
@c.record
class struct_drm_amdgpu_vm_in(c.Struct):
  SIZE = 8
  op: Annotated[Annotated[int, ctypes.c_uint32], 0]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_drm_amdgpu_vm_out(c.Struct):
  SIZE = 8
  flags: Annotated[Annotated[int, ctypes.c_uint64], 0]
@c.record
class union_drm_amdgpu_vm(c.Struct):
  SIZE = 8
  _in: Annotated[struct_drm_amdgpu_vm_in, 0]
  out: Annotated[struct_drm_amdgpu_vm_out, 0]
@c.record
class struct_drm_amdgpu_sched_in(c.Struct):
  SIZE = 16
  op: Annotated[Annotated[int, ctypes.c_uint32], 0]
  fd: Annotated[Annotated[int, ctypes.c_uint32], 4]
  priority: Annotated[Annotated[int, ctypes.c_int32], 8]
  ctx_id: Annotated[Annotated[int, ctypes.c_uint32], 12]
@c.record
class union_drm_amdgpu_sched(c.Struct):
  SIZE = 16
  _in: Annotated[struct_drm_amdgpu_sched_in, 0]
@c.record
class struct_drm_amdgpu_gem_userptr(c.Struct):
  SIZE = 24
  addr: Annotated[Annotated[int, ctypes.c_uint64], 0]
  size: Annotated[Annotated[int, ctypes.c_uint64], 8]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 16]
  handle: Annotated[Annotated[int, ctypes.c_uint32], 20]
@c.record
class struct_drm_amdgpu_gem_dgma(c.Struct):
  SIZE = 24
  addr: Annotated[Annotated[int, ctypes.c_uint64], 0]
  size: Annotated[Annotated[int, ctypes.c_uint64], 8]
  op: Annotated[Annotated[int, ctypes.c_uint32], 16]
  handle: Annotated[Annotated[int, ctypes.c_uint32], 20]
@c.record
class struct_drm_amdgpu_gem_metadata(c.Struct):
  SIZE = 288
  handle: Annotated[Annotated[int, ctypes.c_uint32], 0]
  op: Annotated[Annotated[int, ctypes.c_uint32], 4]
  data: Annotated[struct_drm_amdgpu_gem_metadata_data, 8]
@c.record
class struct_drm_amdgpu_gem_metadata_data(c.Struct):
  SIZE = 280
  flags: Annotated[Annotated[int, ctypes.c_uint64], 0]
  tiling_info: Annotated[Annotated[int, ctypes.c_uint64], 8]
  data_size_bytes: Annotated[Annotated[int, ctypes.c_uint32], 16]
  data: Annotated[c.Array[Annotated[int, ctypes.c_uint32], Literal[64]], 20]
@c.record
class struct_drm_amdgpu_gem_mmap_in(c.Struct):
  SIZE = 8
  handle: Annotated[Annotated[int, ctypes.c_uint32], 0]
  _pad: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_drm_amdgpu_gem_mmap_out(c.Struct):
  SIZE = 8
  addr_ptr: Annotated[Annotated[int, ctypes.c_uint64], 0]
@c.record
class union_drm_amdgpu_gem_mmap(c.Struct):
  SIZE = 8
  _in: Annotated[struct_drm_amdgpu_gem_mmap_in, 0]
  out: Annotated[struct_drm_amdgpu_gem_mmap_out, 0]
@c.record
class struct_drm_amdgpu_gem_wait_idle_in(c.Struct):
  SIZE = 16
  handle: Annotated[Annotated[int, ctypes.c_uint32], 0]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 4]
  timeout: Annotated[Annotated[int, ctypes.c_uint64], 8]
@c.record
class struct_drm_amdgpu_gem_wait_idle_out(c.Struct):
  SIZE = 8
  status: Annotated[Annotated[int, ctypes.c_uint32], 0]
  domain: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class union_drm_amdgpu_gem_wait_idle(c.Struct):
  SIZE = 16
  _in: Annotated[struct_drm_amdgpu_gem_wait_idle_in, 0]
  out: Annotated[struct_drm_amdgpu_gem_wait_idle_out, 0]
@c.record
class struct_drm_amdgpu_wait_cs_in(c.Struct):
  SIZE = 32
  handle: Annotated[Annotated[int, ctypes.c_uint64], 0]
  timeout: Annotated[Annotated[int, ctypes.c_uint64], 8]
  ip_type: Annotated[Annotated[int, ctypes.c_uint32], 16]
  ip_instance: Annotated[Annotated[int, ctypes.c_uint32], 20]
  ring: Annotated[Annotated[int, ctypes.c_uint32], 24]
  ctx_id: Annotated[Annotated[int, ctypes.c_uint32], 28]
@c.record
class struct_drm_amdgpu_wait_cs_out(c.Struct):
  SIZE = 8
  status: Annotated[Annotated[int, ctypes.c_uint64], 0]
@c.record
class union_drm_amdgpu_wait_cs(c.Struct):
  SIZE = 32
  _in: Annotated[struct_drm_amdgpu_wait_cs_in, 0]
  out: Annotated[struct_drm_amdgpu_wait_cs_out, 0]
@c.record
class struct_drm_amdgpu_fence(c.Struct):
  SIZE = 24
  ctx_id: Annotated[Annotated[int, ctypes.c_uint32], 0]
  ip_type: Annotated[Annotated[int, ctypes.c_uint32], 4]
  ip_instance: Annotated[Annotated[int, ctypes.c_uint32], 8]
  ring: Annotated[Annotated[int, ctypes.c_uint32], 12]
  seq_no: Annotated[Annotated[int, ctypes.c_uint64], 16]
@c.record
class struct_drm_amdgpu_wait_fences_in(c.Struct):
  SIZE = 24
  fences: Annotated[Annotated[int, ctypes.c_uint64], 0]
  fence_count: Annotated[Annotated[int, ctypes.c_uint32], 8]
  wait_all: Annotated[Annotated[int, ctypes.c_uint32], 12]
  timeout_ns: Annotated[Annotated[int, ctypes.c_uint64], 16]
@c.record
class struct_drm_amdgpu_wait_fences_out(c.Struct):
  SIZE = 8
  status: Annotated[Annotated[int, ctypes.c_uint32], 0]
  first_signaled: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class union_drm_amdgpu_wait_fences(c.Struct):
  SIZE = 24
  _in: Annotated[struct_drm_amdgpu_wait_fences_in, 0]
  out: Annotated[struct_drm_amdgpu_wait_fences_out, 0]
@c.record
class struct_drm_amdgpu_gem_op(c.Struct):
  SIZE = 16
  handle: Annotated[Annotated[int, ctypes.c_uint32], 0]
  op: Annotated[Annotated[int, ctypes.c_uint32], 4]
  value: Annotated[Annotated[int, ctypes.c_uint64], 8]
@c.record
class struct_drm_amdgpu_gem_va(c.Struct):
  SIZE = 64
  handle: Annotated[Annotated[int, ctypes.c_uint32], 0]
  _pad: Annotated[Annotated[int, ctypes.c_uint32], 4]
  operation: Annotated[Annotated[int, ctypes.c_uint32], 8]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 12]
  va_address: Annotated[Annotated[int, ctypes.c_uint64], 16]
  offset_in_bo: Annotated[Annotated[int, ctypes.c_uint64], 24]
  map_size: Annotated[Annotated[int, ctypes.c_uint64], 32]
  vm_timeline_point: Annotated[Annotated[int, ctypes.c_uint64], 40]
  vm_timeline_syncobj_out: Annotated[Annotated[int, ctypes.c_uint32], 48]
  num_syncobj_handles: Annotated[Annotated[int, ctypes.c_uint32], 52]
  input_fence_syncobj_handles: Annotated[Annotated[int, ctypes.c_uint64], 56]
@c.record
class struct_drm_amdgpu_cs_chunk(c.Struct):
  SIZE = 16
  chunk_id: Annotated[Annotated[int, ctypes.c_uint32], 0]
  length_dw: Annotated[Annotated[int, ctypes.c_uint32], 4]
  chunk_data: Annotated[Annotated[int, ctypes.c_uint64], 8]
@c.record
class struct_drm_amdgpu_cs_in(c.Struct):
  SIZE = 24
  ctx_id: Annotated[Annotated[int, ctypes.c_uint32], 0]
  bo_list_handle: Annotated[Annotated[int, ctypes.c_uint32], 4]
  num_chunks: Annotated[Annotated[int, ctypes.c_uint32], 8]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 12]
  chunks: Annotated[Annotated[int, ctypes.c_uint64], 16]
@c.record
class struct_drm_amdgpu_cs_out(c.Struct):
  SIZE = 8
  handle: Annotated[Annotated[int, ctypes.c_uint64], 0]
@c.record
class union_drm_amdgpu_cs(c.Struct):
  SIZE = 24
  _in: Annotated[struct_drm_amdgpu_cs_in, 0]
  out: Annotated[struct_drm_amdgpu_cs_out, 0]
@c.record
class struct_drm_amdgpu_cs_chunk_ib(c.Struct):
  SIZE = 32
  _pad: Annotated[Annotated[int, ctypes.c_uint32], 0]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 4]
  va_start: Annotated[Annotated[int, ctypes.c_uint64], 8]
  ib_bytes: Annotated[Annotated[int, ctypes.c_uint32], 16]
  ip_type: Annotated[Annotated[int, ctypes.c_uint32], 20]
  ip_instance: Annotated[Annotated[int, ctypes.c_uint32], 24]
  ring: Annotated[Annotated[int, ctypes.c_uint32], 28]
@c.record
class struct_drm_amdgpu_cs_chunk_dep(c.Struct):
  SIZE = 24
  ip_type: Annotated[Annotated[int, ctypes.c_uint32], 0]
  ip_instance: Annotated[Annotated[int, ctypes.c_uint32], 4]
  ring: Annotated[Annotated[int, ctypes.c_uint32], 8]
  ctx_id: Annotated[Annotated[int, ctypes.c_uint32], 12]
  handle: Annotated[Annotated[int, ctypes.c_uint64], 16]
@c.record
class struct_drm_amdgpu_cs_chunk_fence(c.Struct):
  SIZE = 8
  handle: Annotated[Annotated[int, ctypes.c_uint32], 0]
  offset: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_drm_amdgpu_cs_chunk_sem(c.Struct):
  SIZE = 4
  handle: Annotated[Annotated[int, ctypes.c_uint32], 0]
@c.record
class struct_drm_amdgpu_cs_chunk_syncobj(c.Struct):
  SIZE = 16
  handle: Annotated[Annotated[int, ctypes.c_uint32], 0]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 4]
  point: Annotated[Annotated[int, ctypes.c_uint64], 8]
@c.record
class union_drm_amdgpu_fence_to_handle(c.Struct):
  SIZE = 32
  _in: Annotated[union_drm_amdgpu_fence_to_handle_in, 0]
  out: Annotated[union_drm_amdgpu_fence_to_handle_out, 0]
@c.record
class union_drm_amdgpu_fence_to_handle_in(c.Struct):
  SIZE = 32
  fence: Annotated[struct_drm_amdgpu_fence, 0]
  what: Annotated[Annotated[int, ctypes.c_uint32], 24]
  pad: Annotated[Annotated[int, ctypes.c_uint32], 28]
@c.record
class union_drm_amdgpu_fence_to_handle_out(c.Struct):
  SIZE = 4
  handle: Annotated[Annotated[int, ctypes.c_uint32], 0]
@c.record
class struct_drm_amdgpu_cs_chunk_data(c.Struct):
  SIZE = 32
  ib_data: Annotated[struct_drm_amdgpu_cs_chunk_ib, 0]
  fence_data: Annotated[struct_drm_amdgpu_cs_chunk_fence, 0]
@c.record
class struct_drm_amdgpu_cs_chunk_cp_gfx_shadow(c.Struct):
  SIZE = 32
  shadow_va: Annotated[Annotated[int, ctypes.c_uint64], 0]
  csa_va: Annotated[Annotated[int, ctypes.c_uint64], 8]
  gds_va: Annotated[Annotated[int, ctypes.c_uint64], 16]
  flags: Annotated[Annotated[int, ctypes.c_uint64], 24]
@c.record
class struct_drm_amdgpu_query_fw(c.Struct):
  SIZE = 16
  fw_type: Annotated[Annotated[int, ctypes.c_uint32], 0]
  ip_instance: Annotated[Annotated[int, ctypes.c_uint32], 4]
  index: Annotated[Annotated[int, ctypes.c_uint32], 8]
  _pad: Annotated[Annotated[int, ctypes.c_uint32], 12]
@c.record
class struct_drm_amdgpu_info(c.Struct):
  SIZE = 16
  return_pointer: Annotated[Annotated[int, ctypes.c_uint64], 0]
  return_size: Annotated[Annotated[int, ctypes.c_uint32], 8]
  query: Annotated[Annotated[int, ctypes.c_uint32], 12]
@c.record
class struct_drm_amdgpu_info_gds(c.Struct):
  SIZE = 32
  gds_gfx_partition_size: Annotated[Annotated[int, ctypes.c_uint32], 0]
  compute_partition_size: Annotated[Annotated[int, ctypes.c_uint32], 4]
  gds_total_size: Annotated[Annotated[int, ctypes.c_uint32], 8]
  gws_per_gfx_partition: Annotated[Annotated[int, ctypes.c_uint32], 12]
  gws_per_compute_partition: Annotated[Annotated[int, ctypes.c_uint32], 16]
  oa_per_gfx_partition: Annotated[Annotated[int, ctypes.c_uint32], 20]
  oa_per_compute_partition: Annotated[Annotated[int, ctypes.c_uint32], 24]
  _pad: Annotated[Annotated[int, ctypes.c_uint32], 28]
@c.record
class struct_drm_amdgpu_info_vram_gtt(c.Struct):
  SIZE = 24
  vram_size: Annotated[Annotated[int, ctypes.c_uint64], 0]
  vram_cpu_accessible_size: Annotated[Annotated[int, ctypes.c_uint64], 8]
  gtt_size: Annotated[Annotated[int, ctypes.c_uint64], 16]
@c.record
class struct_drm_amdgpu_heap_info(c.Struct):
  SIZE = 32
  total_heap_size: Annotated[Annotated[int, ctypes.c_uint64], 0]
  usable_heap_size: Annotated[Annotated[int, ctypes.c_uint64], 8]
  heap_usage: Annotated[Annotated[int, ctypes.c_uint64], 16]
  max_allocation: Annotated[Annotated[int, ctypes.c_uint64], 24]
@c.record
class struct_drm_amdgpu_memory_info(c.Struct):
  SIZE = 96
  vram: Annotated[struct_drm_amdgpu_heap_info, 0]
  cpu_accessible_vram: Annotated[struct_drm_amdgpu_heap_info, 32]
  gtt: Annotated[struct_drm_amdgpu_heap_info, 64]
@c.record
class struct_drm_amdgpu_info_firmware(c.Struct):
  SIZE = 8
  ver: Annotated[Annotated[int, ctypes.c_uint32], 0]
  feature: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_drm_amdgpu_info_vbios(c.Struct):
  SIZE = 200
  name: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 0]
  vbios_pn: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[64]], 64]
  version: Annotated[Annotated[int, ctypes.c_uint32], 128]
  pad: Annotated[Annotated[int, ctypes.c_uint32], 132]
  vbios_ver_str: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 136]
  date: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[32]], 168]
__u8: TypeAlias = Annotated[int, ctypes.c_ubyte]
@c.record
class struct_drm_amdgpu_info_device(c.Struct):
  SIZE = 448
  device_id: Annotated[Annotated[int, ctypes.c_uint32], 0]
  chip_rev: Annotated[Annotated[int, ctypes.c_uint32], 4]
  external_rev: Annotated[Annotated[int, ctypes.c_uint32], 8]
  pci_rev: Annotated[Annotated[int, ctypes.c_uint32], 12]
  family: Annotated[Annotated[int, ctypes.c_uint32], 16]
  num_shader_engines: Annotated[Annotated[int, ctypes.c_uint32], 20]
  num_shader_arrays_per_engine: Annotated[Annotated[int, ctypes.c_uint32], 24]
  gpu_counter_freq: Annotated[Annotated[int, ctypes.c_uint32], 28]
  max_engine_clock: Annotated[Annotated[int, ctypes.c_uint64], 32]
  max_memory_clock: Annotated[Annotated[int, ctypes.c_uint64], 40]
  cu_active_number: Annotated[Annotated[int, ctypes.c_uint32], 48]
  cu_ao_mask: Annotated[Annotated[int, ctypes.c_uint32], 52]
  cu_bitmap: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_uint32], Literal[4]], Literal[4]], 56]
  enabled_rb_pipes_mask: Annotated[Annotated[int, ctypes.c_uint32], 120]
  num_rb_pipes: Annotated[Annotated[int, ctypes.c_uint32], 124]
  num_hw_gfx_contexts: Annotated[Annotated[int, ctypes.c_uint32], 128]
  pcie_gen: Annotated[Annotated[int, ctypes.c_uint32], 132]
  ids_flags: Annotated[Annotated[int, ctypes.c_uint64], 136]
  virtual_address_offset: Annotated[Annotated[int, ctypes.c_uint64], 144]
  virtual_address_max: Annotated[Annotated[int, ctypes.c_uint64], 152]
  virtual_address_alignment: Annotated[Annotated[int, ctypes.c_uint32], 160]
  pte_fragment_size: Annotated[Annotated[int, ctypes.c_uint32], 164]
  gart_page_size: Annotated[Annotated[int, ctypes.c_uint32], 168]
  ce_ram_size: Annotated[Annotated[int, ctypes.c_uint32], 172]
  vram_type: Annotated[Annotated[int, ctypes.c_uint32], 176]
  vram_bit_width: Annotated[Annotated[int, ctypes.c_uint32], 180]
  vce_harvest_config: Annotated[Annotated[int, ctypes.c_uint32], 184]
  gc_double_offchip_lds_buf: Annotated[Annotated[int, ctypes.c_uint32], 188]
  prim_buf_gpu_addr: Annotated[Annotated[int, ctypes.c_uint64], 192]
  pos_buf_gpu_addr: Annotated[Annotated[int, ctypes.c_uint64], 200]
  cntl_sb_buf_gpu_addr: Annotated[Annotated[int, ctypes.c_uint64], 208]
  param_buf_gpu_addr: Annotated[Annotated[int, ctypes.c_uint64], 216]
  prim_buf_size: Annotated[Annotated[int, ctypes.c_uint32], 224]
  pos_buf_size: Annotated[Annotated[int, ctypes.c_uint32], 228]
  cntl_sb_buf_size: Annotated[Annotated[int, ctypes.c_uint32], 232]
  param_buf_size: Annotated[Annotated[int, ctypes.c_uint32], 236]
  wave_front_size: Annotated[Annotated[int, ctypes.c_uint32], 240]
  num_shader_visible_vgprs: Annotated[Annotated[int, ctypes.c_uint32], 244]
  num_cu_per_sh: Annotated[Annotated[int, ctypes.c_uint32], 248]
  num_tcc_blocks: Annotated[Annotated[int, ctypes.c_uint32], 252]
  gs_vgt_table_depth: Annotated[Annotated[int, ctypes.c_uint32], 256]
  gs_prim_buffer_depth: Annotated[Annotated[int, ctypes.c_uint32], 260]
  max_gs_waves_per_vgt: Annotated[Annotated[int, ctypes.c_uint32], 264]
  pcie_num_lanes: Annotated[Annotated[int, ctypes.c_uint32], 268]
  cu_ao_bitmap: Annotated[c.Array[c.Array[Annotated[int, ctypes.c_uint32], Literal[4]], Literal[4]], 272]
  high_va_offset: Annotated[Annotated[int, ctypes.c_uint64], 336]
  high_va_max: Annotated[Annotated[int, ctypes.c_uint64], 344]
  pa_sc_tile_steering_override: Annotated[Annotated[int, ctypes.c_uint32], 352]
  tcc_disabled_mask: Annotated[Annotated[int, ctypes.c_uint64], 360]
  min_engine_clock: Annotated[Annotated[int, ctypes.c_uint64], 368]
  min_memory_clock: Annotated[Annotated[int, ctypes.c_uint64], 376]
  tcp_cache_size: Annotated[Annotated[int, ctypes.c_uint32], 384]
  num_sqc_per_wgp: Annotated[Annotated[int, ctypes.c_uint32], 388]
  sqc_data_cache_size: Annotated[Annotated[int, ctypes.c_uint32], 392]
  sqc_inst_cache_size: Annotated[Annotated[int, ctypes.c_uint32], 396]
  gl1c_cache_size: Annotated[Annotated[int, ctypes.c_uint32], 400]
  gl2c_cache_size: Annotated[Annotated[int, ctypes.c_uint32], 404]
  mall_size: Annotated[Annotated[int, ctypes.c_uint64], 408]
  enabled_rb_pipes_mask_hi: Annotated[Annotated[int, ctypes.c_uint32], 416]
  shadow_size: Annotated[Annotated[int, ctypes.c_uint32], 420]
  shadow_alignment: Annotated[Annotated[int, ctypes.c_uint32], 424]
  csa_size: Annotated[Annotated[int, ctypes.c_uint32], 428]
  csa_alignment: Annotated[Annotated[int, ctypes.c_uint32], 432]
  userq_ip_mask: Annotated[Annotated[int, ctypes.c_uint32], 436]
  pad: Annotated[Annotated[int, ctypes.c_uint32], 440]
@c.record
class struct_drm_amdgpu_info_hw_ip(c.Struct):
  SIZE = 32
  hw_ip_version_major: Annotated[Annotated[int, ctypes.c_uint32], 0]
  hw_ip_version_minor: Annotated[Annotated[int, ctypes.c_uint32], 4]
  capabilities_flags: Annotated[Annotated[int, ctypes.c_uint64], 8]
  ib_start_alignment: Annotated[Annotated[int, ctypes.c_uint32], 16]
  ib_size_alignment: Annotated[Annotated[int, ctypes.c_uint32], 20]
  available_rings: Annotated[Annotated[int, ctypes.c_uint32], 24]
  ip_discovery_version: Annotated[Annotated[int, ctypes.c_uint32], 28]
@c.record
class struct_drm_amdgpu_info_uq_fw_areas_gfx(c.Struct):
  SIZE = 16
  shadow_size: Annotated[Annotated[int, ctypes.c_uint32], 0]
  shadow_alignment: Annotated[Annotated[int, ctypes.c_uint32], 4]
  csa_size: Annotated[Annotated[int, ctypes.c_uint32], 8]
  csa_alignment: Annotated[Annotated[int, ctypes.c_uint32], 12]
@c.record
class struct_drm_amdgpu_info_uq_fw_areas(c.Struct):
  SIZE = 16
  gfx: Annotated[struct_drm_amdgpu_info_uq_fw_areas_gfx, 0]
@c.record
class struct_drm_amdgpu_info_num_handles(c.Struct):
  SIZE = 8
  uvd_max_handles: Annotated[Annotated[int, ctypes.c_uint32], 0]
  uvd_used_handles: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_drm_amdgpu_info_vce_clock_table_entry(c.Struct):
  SIZE = 16
  sclk: Annotated[Annotated[int, ctypes.c_uint32], 0]
  mclk: Annotated[Annotated[int, ctypes.c_uint32], 4]
  eclk: Annotated[Annotated[int, ctypes.c_uint32], 8]
  pad: Annotated[Annotated[int, ctypes.c_uint32], 12]
@c.record
class struct_drm_amdgpu_info_vce_clock_table(c.Struct):
  SIZE = 104
  entries: Annotated[c.Array[struct_drm_amdgpu_info_vce_clock_table_entry, Literal[6]], 0]
  num_valid_entries: Annotated[Annotated[int, ctypes.c_uint32], 96]
  pad: Annotated[Annotated[int, ctypes.c_uint32], 100]
@c.record
class struct_drm_amdgpu_info_video_codec_info(c.Struct):
  SIZE = 24
  valid: Annotated[Annotated[int, ctypes.c_uint32], 0]
  max_width: Annotated[Annotated[int, ctypes.c_uint32], 4]
  max_height: Annotated[Annotated[int, ctypes.c_uint32], 8]
  max_pixels_per_frame: Annotated[Annotated[int, ctypes.c_uint32], 12]
  max_level: Annotated[Annotated[int, ctypes.c_uint32], 16]
  pad: Annotated[Annotated[int, ctypes.c_uint32], 20]
@c.record
class struct_drm_amdgpu_info_video_caps(c.Struct):
  SIZE = 192
  codec_info: Annotated[c.Array[struct_drm_amdgpu_info_video_codec_info, Literal[8]], 0]
@c.record
class struct_drm_amdgpu_info_gpuvm_fault(c.Struct):
  SIZE = 16
  addr: Annotated[Annotated[int, ctypes.c_uint64], 0]
  status: Annotated[Annotated[int, ctypes.c_uint32], 8]
  vmhub: Annotated[Annotated[int, ctypes.c_uint32], 12]
@c.record
class struct_drm_amdgpu_info_uq_metadata_gfx(c.Struct):
  SIZE = 16
  shadow_size: Annotated[Annotated[int, ctypes.c_uint32], 0]
  shadow_alignment: Annotated[Annotated[int, ctypes.c_uint32], 4]
  csa_size: Annotated[Annotated[int, ctypes.c_uint32], 8]
  csa_alignment: Annotated[Annotated[int, ctypes.c_uint32], 12]
@c.record
class struct_drm_amdgpu_info_uq_metadata(c.Struct):
  SIZE = 16
  gfx: Annotated[struct_drm_amdgpu_info_uq_metadata_gfx, 0]
class _anonstruct0(ctypes.Structure): pass
class struct_drm_amdgpu_virtual_range(ctypes.Structure): pass
@c.record
class struct_drm_amdgpu_capability(c.Struct):
  SIZE = 8
  flag: Annotated[Annotated[int, ctypes.c_uint32], 0]
  direct_gma_size: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_drm_amdgpu_freesync(c.Struct):
  SIZE = 32
  op: Annotated[Annotated[int, ctypes.c_uint32], 0]
  spare: Annotated[c.Array[Annotated[int, ctypes.c_uint32], Literal[7]], 4]
c.init_records()
DRM_NAME = "drm" # type: ignore
DRM_MIN_ORDER = 5 # type: ignore
DRM_MAX_ORDER = 22 # type: ignore
DRM_RAM_PERCENT = 10 # type: ignore
_DRM_LOCK_HELD = 0x80000000 # type: ignore
_DRM_LOCK_CONT = 0x40000000 # type: ignore
_DRM_LOCK_IS_HELD = lambda lock: ((lock) & _DRM_LOCK_HELD) # type: ignore
_DRM_LOCK_IS_CONT = lambda lock: ((lock) & _DRM_LOCK_CONT) # type: ignore
_DRM_LOCKING_CONTEXT = lambda lock: ((lock) & ~(_DRM_LOCK_HELD|_DRM_LOCK_CONT)) # type: ignore
_DRM_VBLANK_HIGH_CRTC_SHIFT = 1 # type: ignore
_DRM_VBLANK_TYPES_MASK = (_DRM_VBLANK_ABSOLUTE | _DRM_VBLANK_RELATIVE) # type: ignore
_DRM_VBLANK_FLAGS_MASK = (_DRM_VBLANK_EVENT | _DRM_VBLANK_SIGNAL | _DRM_VBLANK_SECONDARY | _DRM_VBLANK_NEXTONMISS) # type: ignore
_DRM_PRE_MODESET = 1 # type: ignore
_DRM_POST_MODESET = 2 # type: ignore
DRM_CAP_DUMB_BUFFER = 0x1 # type: ignore
DRM_CAP_VBLANK_HIGH_CRTC = 0x2 # type: ignore
DRM_CAP_DUMB_PREFERRED_DEPTH = 0x3 # type: ignore
DRM_CAP_DUMB_PREFER_SHADOW = 0x4 # type: ignore
DRM_CAP_PRIME = 0x5 # type: ignore
DRM_PRIME_CAP_IMPORT = 0x1 # type: ignore
DRM_PRIME_CAP_EXPORT = 0x2 # type: ignore
DRM_CAP_TIMESTAMP_MONOTONIC = 0x6 # type: ignore
DRM_CAP_ASYNC_PAGE_FLIP = 0x7 # type: ignore
DRM_CAP_CURSOR_WIDTH = 0x8 # type: ignore
DRM_CAP_CURSOR_HEIGHT = 0x9 # type: ignore
DRM_CAP_ADDFB2_MODIFIERS = 0x10 # type: ignore
DRM_CAP_PAGE_FLIP_TARGET = 0x11 # type: ignore
DRM_CAP_CRTC_IN_VBLANK_EVENT = 0x12 # type: ignore
DRM_CAP_SYNCOBJ = 0x13 # type: ignore
DRM_CAP_SYNCOBJ_TIMELINE = 0x14 # type: ignore
DRM_CAP_ATOMIC_ASYNC_PAGE_FLIP = 0x15 # type: ignore
DRM_CLIENT_CAP_STEREO_3D = 1 # type: ignore
DRM_CLIENT_CAP_UNIVERSAL_PLANES = 2 # type: ignore
DRM_CLIENT_CAP_ATOMIC = 3 # type: ignore
DRM_CLIENT_CAP_ASPECT_RATIO = 4 # type: ignore
DRM_CLIENT_CAP_WRITEBACK_CONNECTORS = 5 # type: ignore
DRM_CLIENT_CAP_CURSOR_PLANE_HOTSPOT = 6 # type: ignore
DRM_SYNCOBJ_CREATE_SIGNALED = (1 << 0) # type: ignore
DRM_SYNCOBJ_FD_TO_HANDLE_FLAGS_IMPORT_SYNC_FILE = (1 << 0) # type: ignore
DRM_SYNCOBJ_HANDLE_TO_FD_FLAGS_EXPORT_SYNC_FILE = (1 << 0) # type: ignore
DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL = (1 << 0) # type: ignore
DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT = (1 << 1) # type: ignore
DRM_SYNCOBJ_WAIT_FLAGS_WAIT_AVAILABLE = (1 << 2) # type: ignore
DRM_SYNCOBJ_WAIT_FLAGS_WAIT_DEADLINE = (1 << 3) # type: ignore
DRM_SYNCOBJ_QUERY_FLAGS_LAST_SUBMITTED = (1 << 0) # type: ignore
DRM_CRTC_SEQUENCE_RELATIVE = 0x00000001 # type: ignore
DRM_CRTC_SEQUENCE_NEXT_ON_MISS = 0x00000002 # type: ignore
DRM_IOCTL_BASE = 'd' # type: ignore
DRM_IO = lambda nr: _IO(DRM_IOCTL_BASE,nr) # type: ignore
DRM_IOR = lambda nr,type: _IOR(DRM_IOCTL_BASE,nr,type) # type: ignore
DRM_IOW = lambda nr,type: _IOW(DRM_IOCTL_BASE,nr,type) # type: ignore
DRM_IOWR = lambda nr,type: _IOWR(DRM_IOCTL_BASE,nr,type) # type: ignore
DRM_IOCTL_VERSION = DRM_IOWR(0x00, struct_drm_version) # type: ignore
DRM_IOCTL_GET_UNIQUE = DRM_IOWR(0x01, struct_drm_unique) # type: ignore
DRM_IOCTL_GET_MAGIC = DRM_IOR( 0x02, struct_drm_auth) # type: ignore
DRM_IOCTL_IRQ_BUSID = DRM_IOWR(0x03, struct_drm_irq_busid) # type: ignore
DRM_IOCTL_GET_MAP = DRM_IOWR(0x04, struct_drm_map) # type: ignore
DRM_IOCTL_GET_CLIENT = DRM_IOWR(0x05, struct_drm_client) # type: ignore
DRM_IOCTL_GET_STATS = DRM_IOR( 0x06, struct_drm_stats) # type: ignore
DRM_IOCTL_SET_VERSION = DRM_IOWR(0x07, struct_drm_set_version) # type: ignore
DRM_IOCTL_MODESET_CTL = DRM_IOW(0x08, struct_drm_modeset_ctl) # type: ignore
DRM_IOCTL_GEM_CLOSE = DRM_IOW (0x09, struct_drm_gem_close) # type: ignore
DRM_IOCTL_GEM_FLINK = DRM_IOWR(0x0a, struct_drm_gem_flink) # type: ignore
DRM_IOCTL_GEM_OPEN = DRM_IOWR(0x0b, struct_drm_gem_open) # type: ignore
DRM_IOCTL_GET_CAP = DRM_IOWR(0x0c, struct_drm_get_cap) # type: ignore
DRM_IOCTL_SET_CLIENT_CAP = DRM_IOW( 0x0d, struct_drm_set_client_cap) # type: ignore
DRM_IOCTL_SET_UNIQUE = DRM_IOW( 0x10, struct_drm_unique) # type: ignore
DRM_IOCTL_AUTH_MAGIC = DRM_IOW( 0x11, struct_drm_auth) # type: ignore
DRM_IOCTL_BLOCK = DRM_IOWR(0x12, struct_drm_block) # type: ignore
DRM_IOCTL_UNBLOCK = DRM_IOWR(0x13, struct_drm_block) # type: ignore
DRM_IOCTL_CONTROL = DRM_IOW( 0x14, struct_drm_control) # type: ignore
DRM_IOCTL_ADD_MAP = DRM_IOWR(0x15, struct_drm_map) # type: ignore
DRM_IOCTL_ADD_BUFS = DRM_IOWR(0x16, struct_drm_buf_desc) # type: ignore
DRM_IOCTL_MARK_BUFS = DRM_IOW( 0x17, struct_drm_buf_desc) # type: ignore
DRM_IOCTL_INFO_BUFS = DRM_IOWR(0x18, struct_drm_buf_info) # type: ignore
DRM_IOCTL_MAP_BUFS = DRM_IOWR(0x19, struct_drm_buf_map) # type: ignore
DRM_IOCTL_FREE_BUFS = DRM_IOW( 0x1a, struct_drm_buf_free) # type: ignore
DRM_IOCTL_RM_MAP = DRM_IOW( 0x1b, struct_drm_map) # type: ignore
DRM_IOCTL_SET_SAREA_CTX = DRM_IOW( 0x1c, struct_drm_ctx_priv_map) # type: ignore
DRM_IOCTL_GET_SAREA_CTX = DRM_IOWR(0x1d, struct_drm_ctx_priv_map) # type: ignore
DRM_IOCTL_SET_MASTER = DRM_IO(0x1e) # type: ignore
DRM_IOCTL_DROP_MASTER = DRM_IO(0x1f) # type: ignore
DRM_IOCTL_ADD_CTX = DRM_IOWR(0x20, struct_drm_ctx) # type: ignore
DRM_IOCTL_RM_CTX = DRM_IOWR(0x21, struct_drm_ctx) # type: ignore
DRM_IOCTL_MOD_CTX = DRM_IOW( 0x22, struct_drm_ctx) # type: ignore
DRM_IOCTL_GET_CTX = DRM_IOWR(0x23, struct_drm_ctx) # type: ignore
DRM_IOCTL_SWITCH_CTX = DRM_IOW( 0x24, struct_drm_ctx) # type: ignore
DRM_IOCTL_NEW_CTX = DRM_IOW( 0x25, struct_drm_ctx) # type: ignore
DRM_IOCTL_RES_CTX = DRM_IOWR(0x26, struct_drm_ctx_res) # type: ignore
DRM_IOCTL_ADD_DRAW = DRM_IOWR(0x27, struct_drm_draw) # type: ignore
DRM_IOCTL_RM_DRAW = DRM_IOWR(0x28, struct_drm_draw) # type: ignore
DRM_IOCTL_DMA = DRM_IOWR(0x29, struct_drm_dma) # type: ignore
DRM_IOCTL_LOCK = DRM_IOW( 0x2a, struct_drm_lock) # type: ignore
DRM_IOCTL_UNLOCK = DRM_IOW( 0x2b, struct_drm_lock) # type: ignore
DRM_IOCTL_FINISH = DRM_IOW( 0x2c, struct_drm_lock) # type: ignore
DRM_IOCTL_PRIME_HANDLE_TO_FD = DRM_IOWR(0x2d, struct_drm_prime_handle) # type: ignore
DRM_IOCTL_PRIME_FD_TO_HANDLE = DRM_IOWR(0x2e, struct_drm_prime_handle) # type: ignore
DRM_IOCTL_AGP_ACQUIRE = DRM_IO(  0x30) # type: ignore
DRM_IOCTL_AGP_RELEASE = DRM_IO(  0x31) # type: ignore
DRM_IOCTL_AGP_ENABLE = DRM_IOW( 0x32, struct_drm_agp_mode) # type: ignore
DRM_IOCTL_AGP_INFO = DRM_IOR( 0x33, struct_drm_agp_info) # type: ignore
DRM_IOCTL_AGP_ALLOC = DRM_IOWR(0x34, struct_drm_agp_buffer) # type: ignore
DRM_IOCTL_AGP_FREE = DRM_IOW( 0x35, struct_drm_agp_buffer) # type: ignore
DRM_IOCTL_AGP_BIND = DRM_IOW( 0x36, struct_drm_agp_binding) # type: ignore
DRM_IOCTL_AGP_UNBIND = DRM_IOW( 0x37, struct_drm_agp_binding) # type: ignore
DRM_IOCTL_SG_ALLOC = DRM_IOWR(0x38, struct_drm_scatter_gather) # type: ignore
DRM_IOCTL_SG_FREE = DRM_IOW( 0x39, struct_drm_scatter_gather) # type: ignore
DRM_IOCTL_WAIT_VBLANK = DRM_IOWR(0x3a, union_drm_wait_vblank) # type: ignore
DRM_IOCTL_CRTC_GET_SEQUENCE = DRM_IOWR(0x3b, struct_drm_crtc_get_sequence) # type: ignore
DRM_IOCTL_CRTC_QUEUE_SEQUENCE = DRM_IOWR(0x3c, struct_drm_crtc_queue_sequence) # type: ignore
DRM_IOCTL_UPDATE_DRAW = DRM_IOW(0x3f, struct_drm_update_draw) # type: ignore
DRM_IOCTL_SYNCOBJ_CREATE = DRM_IOWR(0xBF, struct_drm_syncobj_create) # type: ignore
DRM_IOCTL_SYNCOBJ_DESTROY = DRM_IOWR(0xC0, struct_drm_syncobj_destroy) # type: ignore
DRM_IOCTL_SYNCOBJ_HANDLE_TO_FD = DRM_IOWR(0xC1, struct_drm_syncobj_handle) # type: ignore
DRM_IOCTL_SYNCOBJ_FD_TO_HANDLE = DRM_IOWR(0xC2, struct_drm_syncobj_handle) # type: ignore
DRM_IOCTL_SYNCOBJ_WAIT = DRM_IOWR(0xC3, struct_drm_syncobj_wait) # type: ignore
DRM_IOCTL_SYNCOBJ_RESET = DRM_IOWR(0xC4, struct_drm_syncobj_array) # type: ignore
DRM_IOCTL_SYNCOBJ_SIGNAL = DRM_IOWR(0xC5, struct_drm_syncobj_array) # type: ignore
DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT = DRM_IOWR(0xCA, struct_drm_syncobj_timeline_wait) # type: ignore
DRM_IOCTL_SYNCOBJ_QUERY = DRM_IOWR(0xCB, struct_drm_syncobj_timeline_array) # type: ignore
DRM_IOCTL_SYNCOBJ_TRANSFER = DRM_IOWR(0xCC, struct_drm_syncobj_transfer) # type: ignore
DRM_IOCTL_SYNCOBJ_TIMELINE_SIGNAL = DRM_IOWR(0xCD, struct_drm_syncobj_timeline_array) # type: ignore
DRM_IOCTL_SYNCOBJ_EVENTFD = DRM_IOWR(0xCF, struct_drm_syncobj_eventfd) # type: ignore
DRM_COMMAND_BASE = 0x40 # type: ignore
DRM_COMMAND_END = 0xA0 # type: ignore
DRM_EVENT_VBLANK = 0x01 # type: ignore
DRM_EVENT_FLIP_COMPLETE = 0x02 # type: ignore
DRM_EVENT_CRTC_SEQUENCE = 0x03 # type: ignore
DRM_AMDGPU_GEM_CREATE = 0x00 # type: ignore
DRM_AMDGPU_GEM_MMAP = 0x01 # type: ignore
DRM_AMDGPU_CTX = 0x02 # type: ignore
DRM_AMDGPU_BO_LIST = 0x03 # type: ignore
DRM_AMDGPU_CS = 0x04 # type: ignore
DRM_AMDGPU_INFO = 0x05 # type: ignore
DRM_AMDGPU_GEM_METADATA = 0x06 # type: ignore
DRM_AMDGPU_GEM_WAIT_IDLE = 0x07 # type: ignore
DRM_AMDGPU_GEM_VA = 0x08 # type: ignore
DRM_AMDGPU_WAIT_CS = 0x09 # type: ignore
DRM_AMDGPU_GEM_OP = 0x10 # type: ignore
DRM_AMDGPU_GEM_USERPTR = 0x11 # type: ignore
DRM_AMDGPU_WAIT_FENCES = 0x12 # type: ignore
DRM_AMDGPU_VM = 0x13 # type: ignore
DRM_AMDGPU_FENCE_TO_HANDLE = 0x14 # type: ignore
DRM_AMDGPU_SCHED = 0x15 # type: ignore
DRM_AMDGPU_USERQ = 0x16 # type: ignore
DRM_AMDGPU_USERQ_SIGNAL = 0x17 # type: ignore
DRM_AMDGPU_USERQ_WAIT = 0x18 # type: ignore
DRM_AMDGPU_GEM_DGMA = 0x5c # type: ignore
DRM_AMDGPU_SEM = 0x5b # type: ignore
DRM_IOCTL_AMDGPU_GEM_CREATE = DRM_IOWR(DRM_COMMAND_BASE + DRM_AMDGPU_GEM_CREATE, union_drm_amdgpu_gem_create) # type: ignore
DRM_IOCTL_AMDGPU_GEM_MMAP = DRM_IOWR(DRM_COMMAND_BASE + DRM_AMDGPU_GEM_MMAP, union_drm_amdgpu_gem_mmap) # type: ignore
DRM_IOCTL_AMDGPU_CTX = DRM_IOWR(DRM_COMMAND_BASE + DRM_AMDGPU_CTX, union_drm_amdgpu_ctx) # type: ignore
DRM_IOCTL_AMDGPU_BO_LIST = DRM_IOWR(DRM_COMMAND_BASE + DRM_AMDGPU_BO_LIST, union_drm_amdgpu_bo_list) # type: ignore
DRM_IOCTL_AMDGPU_CS = DRM_IOWR(DRM_COMMAND_BASE + DRM_AMDGPU_CS, union_drm_amdgpu_cs) # type: ignore
DRM_IOCTL_AMDGPU_INFO = DRM_IOW(DRM_COMMAND_BASE + DRM_AMDGPU_INFO, struct_drm_amdgpu_info) # type: ignore
DRM_IOCTL_AMDGPU_GEM_METADATA = DRM_IOWR(DRM_COMMAND_BASE + DRM_AMDGPU_GEM_METADATA, struct_drm_amdgpu_gem_metadata) # type: ignore
DRM_IOCTL_AMDGPU_GEM_WAIT_IDLE = DRM_IOWR(DRM_COMMAND_BASE + DRM_AMDGPU_GEM_WAIT_IDLE, union_drm_amdgpu_gem_wait_idle) # type: ignore
DRM_IOCTL_AMDGPU_GEM_VA = DRM_IOW(DRM_COMMAND_BASE + DRM_AMDGPU_GEM_VA, struct_drm_amdgpu_gem_va) # type: ignore
DRM_IOCTL_AMDGPU_WAIT_CS = DRM_IOWR(DRM_COMMAND_BASE + DRM_AMDGPU_WAIT_CS, union_drm_amdgpu_wait_cs) # type: ignore
DRM_IOCTL_AMDGPU_GEM_OP = DRM_IOWR(DRM_COMMAND_BASE + DRM_AMDGPU_GEM_OP, struct_drm_amdgpu_gem_op) # type: ignore
DRM_IOCTL_AMDGPU_GEM_USERPTR = DRM_IOWR(DRM_COMMAND_BASE + DRM_AMDGPU_GEM_USERPTR, struct_drm_amdgpu_gem_userptr) # type: ignore
DRM_IOCTL_AMDGPU_WAIT_FENCES = DRM_IOWR(DRM_COMMAND_BASE + DRM_AMDGPU_WAIT_FENCES, union_drm_amdgpu_wait_fences) # type: ignore
DRM_IOCTL_AMDGPU_VM = DRM_IOWR(DRM_COMMAND_BASE + DRM_AMDGPU_VM, union_drm_amdgpu_vm) # type: ignore
DRM_IOCTL_AMDGPU_FENCE_TO_HANDLE = DRM_IOWR(DRM_COMMAND_BASE + DRM_AMDGPU_FENCE_TO_HANDLE, union_drm_amdgpu_fence_to_handle) # type: ignore
DRM_IOCTL_AMDGPU_SCHED = DRM_IOW(DRM_COMMAND_BASE + DRM_AMDGPU_SCHED, union_drm_amdgpu_sched) # type: ignore
DRM_IOCTL_AMDGPU_USERQ = DRM_IOWR(DRM_COMMAND_BASE + DRM_AMDGPU_USERQ, union_drm_amdgpu_userq) # type: ignore
DRM_IOCTL_AMDGPU_USERQ_SIGNAL = DRM_IOWR(DRM_COMMAND_BASE + DRM_AMDGPU_USERQ_SIGNAL, struct_drm_amdgpu_userq_signal) # type: ignore
DRM_IOCTL_AMDGPU_USERQ_WAIT = DRM_IOWR(DRM_COMMAND_BASE + DRM_AMDGPU_USERQ_WAIT, struct_drm_amdgpu_userq_wait) # type: ignore
DRM_IOCTL_AMDGPU_GEM_DGMA = DRM_IOWR(DRM_COMMAND_BASE + DRM_AMDGPU_GEM_DGMA, struct_drm_amdgpu_gem_dgma) # type: ignore
DRM_IOCTL_AMDGPU_SEM = DRM_IOWR(DRM_COMMAND_BASE + DRM_AMDGPU_SEM, union_drm_amdgpu_sem) # type: ignore
AMDGPU_GEM_DOMAIN_CPU = 0x1 # type: ignore
AMDGPU_GEM_DOMAIN_GTT = 0x2 # type: ignore
AMDGPU_GEM_DOMAIN_VRAM = 0x4 # type: ignore
AMDGPU_GEM_DOMAIN_GDS = 0x8 # type: ignore
AMDGPU_GEM_DOMAIN_GWS = 0x10 # type: ignore
AMDGPU_GEM_DOMAIN_OA = 0x20 # type: ignore
AMDGPU_GEM_DOMAIN_DOORBELL = 0x40 # type: ignore
AMDGPU_GEM_DOMAIN_DGMA = 0x400 # type: ignore
AMDGPU_GEM_DOMAIN_DGMA_IMPORT = 0x800 # type: ignore
AMDGPU_GEM_DOMAIN_MASK = (AMDGPU_GEM_DOMAIN_CPU | AMDGPU_GEM_DOMAIN_GTT | AMDGPU_GEM_DOMAIN_VRAM | AMDGPU_GEM_DOMAIN_GDS | AMDGPU_GEM_DOMAIN_GWS | AMDGPU_GEM_DOMAIN_OA | AMDGPU_GEM_DOMAIN_DOORBELL | AMDGPU_GEM_DOMAIN_DGMA | AMDGPU_GEM_DOMAIN_DGMA_IMPORT) # type: ignore
AMDGPU_GEM_CREATE_CPU_ACCESS_REQUIRED = (1 << 0) # type: ignore
AMDGPU_GEM_CREATE_NO_CPU_ACCESS = (1 << 1) # type: ignore
AMDGPU_GEM_CREATE_CPU_GTT_USWC = (1 << 2) # type: ignore
AMDGPU_GEM_CREATE_VRAM_CLEARED = (1 << 3) # type: ignore
AMDGPU_GEM_CREATE_VRAM_CONTIGUOUS = (1 << 5) # type: ignore
AMDGPU_GEM_CREATE_VM_ALWAYS_VALID = (1 << 6) # type: ignore
AMDGPU_GEM_CREATE_EXPLICIT_SYNC = (1 << 7) # type: ignore
AMDGPU_GEM_CREATE_CP_MQD_GFX9 = (1 << 8) # type: ignore
AMDGPU_GEM_CREATE_VRAM_WIPE_ON_RELEASE = (1 << 9) # type: ignore
AMDGPU_GEM_CREATE_ENCRYPTED = (1 << 10) # type: ignore
AMDGPU_GEM_CREATE_PREEMPTIBLE = (1 << 11) # type: ignore
AMDGPU_GEM_CREATE_DISCARDABLE = (1 << 12) # type: ignore
AMDGPU_GEM_CREATE_COHERENT = (1 << 13) # type: ignore
AMDGPU_GEM_CREATE_UNCACHED = (1 << 14) # type: ignore
AMDGPU_GEM_CREATE_EXT_COHERENT = (1 << 15) # type: ignore
AMDGPU_GEM_CREATE_GFX12_DCC = (1 << 16) # type: ignore
AMDGPU_GEM_CREATE_SPARSE = (1 << 29) # type: ignore
AMDGPU_GEM_CREATE_TOP_DOWN = (1 << 30) # type: ignore
AMDGPU_GEM_CREATE_NO_EVICT = (1 << 31) # type: ignore
AMDGPU_BO_LIST_OP_CREATE = 0 # type: ignore
AMDGPU_BO_LIST_OP_DESTROY = 1 # type: ignore
AMDGPU_BO_LIST_OP_UPDATE = 2 # type: ignore
AMDGPU_CTX_OP_ALLOC_CTX = 1 # type: ignore
AMDGPU_CTX_OP_FREE_CTX = 2 # type: ignore
AMDGPU_CTX_OP_QUERY_STATE = 3 # type: ignore
AMDGPU_CTX_OP_QUERY_STATE2 = 4 # type: ignore
AMDGPU_CTX_OP_GET_STABLE_PSTATE = 5 # type: ignore
AMDGPU_CTX_OP_SET_STABLE_PSTATE = 6 # type: ignore
AMDGPU_CTX_NO_RESET = 0 # type: ignore
AMDGPU_CTX_GUILTY_RESET = 1 # type: ignore
AMDGPU_CTX_INNOCENT_RESET = 2 # type: ignore
AMDGPU_CTX_UNKNOWN_RESET = 3 # type: ignore
AMDGPU_CTX_QUERY2_FLAGS_RESET = (1<<0) # type: ignore
AMDGPU_CTX_QUERY2_FLAGS_VRAMLOST = (1<<1) # type: ignore
AMDGPU_CTX_QUERY2_FLAGS_GUILTY = (1<<2) # type: ignore
AMDGPU_CTX_QUERY2_FLAGS_RAS_CE = (1<<3) # type: ignore
AMDGPU_CTX_QUERY2_FLAGS_RAS_UE = (1<<4) # type: ignore
AMDGPU_CTX_QUERY2_FLAGS_RESET_IN_PROGRESS = (1<<5) # type: ignore
AMDGPU_CTX_PRIORITY_UNSET = -2048 # type: ignore
AMDGPU_CTX_PRIORITY_VERY_LOW = -1023 # type: ignore
AMDGPU_CTX_PRIORITY_LOW = -512 # type: ignore
AMDGPU_CTX_PRIORITY_NORMAL = 0 # type: ignore
AMDGPU_CTX_PRIORITY_HIGH = 512 # type: ignore
AMDGPU_CTX_PRIORITY_VERY_HIGH = 1023 # type: ignore
AMDGPU_CTX_STABLE_PSTATE_FLAGS_MASK = 0xf # type: ignore
AMDGPU_CTX_STABLE_PSTATE_NONE = 0 # type: ignore
AMDGPU_CTX_STABLE_PSTATE_STANDARD = 1 # type: ignore
AMDGPU_CTX_STABLE_PSTATE_MIN_SCLK = 2 # type: ignore
AMDGPU_CTX_STABLE_PSTATE_MIN_MCLK = 3 # type: ignore
AMDGPU_CTX_STABLE_PSTATE_PEAK = 4 # type: ignore
AMDGPU_USERQ_OP_CREATE = 1 # type: ignore
AMDGPU_USERQ_OP_FREE = 2 # type: ignore
AMDGPU_USERQ_CREATE_FLAGS_QUEUE_PRIORITY_MASK = 0x3 # type: ignore
AMDGPU_USERQ_CREATE_FLAGS_QUEUE_PRIORITY_SHIFT = 0 # type: ignore
AMDGPU_USERQ_CREATE_FLAGS_QUEUE_PRIORITY_NORMAL_LOW = 0 # type: ignore
AMDGPU_USERQ_CREATE_FLAGS_QUEUE_PRIORITY_LOW = 1 # type: ignore
AMDGPU_USERQ_CREATE_FLAGS_QUEUE_PRIORITY_NORMAL_HIGH = 2 # type: ignore
AMDGPU_USERQ_CREATE_FLAGS_QUEUE_PRIORITY_HIGH = 3 # type: ignore
AMDGPU_USERQ_CREATE_FLAGS_QUEUE_SECURE = (1 << 2) # type: ignore
AMDGPU_SEM_OP_CREATE_SEM = 1 # type: ignore
AMDGPU_SEM_OP_WAIT_SEM = 2 # type: ignore
AMDGPU_SEM_OP_SIGNAL_SEM = 3 # type: ignore
AMDGPU_SEM_OP_DESTROY_SEM = 4 # type: ignore
AMDGPU_SEM_OP_IMPORT_SEM = 5 # type: ignore
AMDGPU_SEM_OP_EXPORT_SEM = 6 # type: ignore
AMDGPU_VM_OP_RESERVE_VMID = 1 # type: ignore
AMDGPU_VM_OP_UNRESERVE_VMID = 2 # type: ignore
AMDGPU_SCHED_OP_PROCESS_PRIORITY_OVERRIDE = 1 # type: ignore
AMDGPU_SCHED_OP_CONTEXT_PRIORITY_OVERRIDE = 2 # type: ignore
AMDGPU_GEM_USERPTR_READONLY = (1 << 0) # type: ignore
AMDGPU_GEM_USERPTR_ANONONLY = (1 << 1) # type: ignore
AMDGPU_GEM_USERPTR_VALIDATE = (1 << 2) # type: ignore
AMDGPU_GEM_USERPTR_REGISTER = (1 << 3) # type: ignore
AMDGPU_GEM_DGMA_IMPORT = 0 # type: ignore
AMDGPU_GEM_DGMA_QUERY_PHYS_ADDR = 1 # type: ignore
AMDGPU_TILING_ARRAY_MODE_SHIFT = 0 # type: ignore
AMDGPU_TILING_ARRAY_MODE_MASK = 0xf # type: ignore
AMDGPU_TILING_PIPE_CONFIG_SHIFT = 4 # type: ignore
AMDGPU_TILING_PIPE_CONFIG_MASK = 0x1f # type: ignore
AMDGPU_TILING_TILE_SPLIT_SHIFT = 9 # type: ignore
AMDGPU_TILING_TILE_SPLIT_MASK = 0x7 # type: ignore
AMDGPU_TILING_MICRO_TILE_MODE_SHIFT = 12 # type: ignore
AMDGPU_TILING_MICRO_TILE_MODE_MASK = 0x7 # type: ignore
AMDGPU_TILING_BANK_WIDTH_SHIFT = 15 # type: ignore
AMDGPU_TILING_BANK_WIDTH_MASK = 0x3 # type: ignore
AMDGPU_TILING_BANK_HEIGHT_SHIFT = 17 # type: ignore
AMDGPU_TILING_BANK_HEIGHT_MASK = 0x3 # type: ignore
AMDGPU_TILING_MACRO_TILE_ASPECT_SHIFT = 19 # type: ignore
AMDGPU_TILING_MACRO_TILE_ASPECT_MASK = 0x3 # type: ignore
AMDGPU_TILING_NUM_BANKS_SHIFT = 21 # type: ignore
AMDGPU_TILING_NUM_BANKS_MASK = 0x3 # type: ignore
AMDGPU_TILING_SWIZZLE_MODE_SHIFT = 0 # type: ignore
AMDGPU_TILING_SWIZZLE_MODE_MASK = 0x1f # type: ignore
AMDGPU_TILING_DCC_OFFSET_256B_SHIFT = 5 # type: ignore
AMDGPU_TILING_DCC_OFFSET_256B_MASK = 0xFFFFFF # type: ignore
AMDGPU_TILING_DCC_PITCH_MAX_SHIFT = 29 # type: ignore
AMDGPU_TILING_DCC_PITCH_MAX_MASK = 0x3FFF # type: ignore
AMDGPU_TILING_DCC_INDEPENDENT_64B_SHIFT = 43 # type: ignore
AMDGPU_TILING_DCC_INDEPENDENT_64B_MASK = 0x1 # type: ignore
AMDGPU_TILING_DCC_INDEPENDENT_128B_SHIFT = 44 # type: ignore
AMDGPU_TILING_DCC_INDEPENDENT_128B_MASK = 0x1 # type: ignore
AMDGPU_TILING_SCANOUT_SHIFT = 63 # type: ignore
AMDGPU_TILING_SCANOUT_MASK = 0x1 # type: ignore
AMDGPU_TILING_GFX12_SWIZZLE_MODE_SHIFT = 0 # type: ignore
AMDGPU_TILING_GFX12_SWIZZLE_MODE_MASK = 0x7 # type: ignore
AMDGPU_TILING_GFX12_DCC_MAX_COMPRESSED_BLOCK_SHIFT = 3 # type: ignore
AMDGPU_TILING_GFX12_DCC_MAX_COMPRESSED_BLOCK_MASK = 0x3 # type: ignore
AMDGPU_TILING_GFX12_DCC_NUMBER_TYPE_SHIFT = 5 # type: ignore
AMDGPU_TILING_GFX12_DCC_NUMBER_TYPE_MASK = 0x7 # type: ignore
AMDGPU_TILING_GFX12_DCC_DATA_FORMAT_SHIFT = 8 # type: ignore
AMDGPU_TILING_GFX12_DCC_DATA_FORMAT_MASK = 0x3f # type: ignore
AMDGPU_TILING_GFX12_DCC_WRITE_COMPRESS_DISABLE_SHIFT = 14 # type: ignore
AMDGPU_TILING_GFX12_DCC_WRITE_COMPRESS_DISABLE_MASK = 0x1 # type: ignore
AMDGPU_TILING_GFX12_SCANOUT_SHIFT = 63 # type: ignore
AMDGPU_TILING_GFX12_SCANOUT_MASK = 0x1 # type: ignore
AMDGPU_GEM_METADATA_OP_SET_METADATA = 1 # type: ignore
AMDGPU_GEM_METADATA_OP_GET_METADATA = 2 # type: ignore
AMDGPU_GEM_OP_GET_GEM_CREATE_INFO = 0 # type: ignore
AMDGPU_GEM_OP_SET_PLACEMENT = 1 # type: ignore
AMDGPU_VA_OP_MAP = 1 # type: ignore
AMDGPU_VA_OP_UNMAP = 2 # type: ignore
AMDGPU_VA_OP_CLEAR = 3 # type: ignore
AMDGPU_VA_OP_REPLACE = 4 # type: ignore
AMDGPU_VM_DELAY_UPDATE = (1 << 0) # type: ignore
AMDGPU_VM_PAGE_READABLE = (1 << 1) # type: ignore
AMDGPU_VM_PAGE_WRITEABLE = (1 << 2) # type: ignore
AMDGPU_VM_PAGE_EXECUTABLE = (1 << 3) # type: ignore
AMDGPU_VM_PAGE_PRT = (1 << 4) # type: ignore
AMDGPU_VM_MTYPE_MASK = (0xf << 5) # type: ignore
AMDGPU_VM_MTYPE_DEFAULT = (0 << 5) # type: ignore
AMDGPU_VM_MTYPE_NC = (1 << 5) # type: ignore
AMDGPU_VM_MTYPE_WC = (2 << 5) # type: ignore
AMDGPU_VM_MTYPE_CC = (3 << 5) # type: ignore
AMDGPU_VM_MTYPE_UC = (4 << 5) # type: ignore
AMDGPU_VM_MTYPE_RW = (5 << 5) # type: ignore
AMDGPU_VM_PAGE_NOALLOC = (1 << 9) # type: ignore
AMDGPU_HW_IP_GFX = 0 # type: ignore
AMDGPU_HW_IP_COMPUTE = 1 # type: ignore
AMDGPU_HW_IP_DMA = 2 # type: ignore
AMDGPU_HW_IP_UVD = 3 # type: ignore
AMDGPU_HW_IP_VCE = 4 # type: ignore
AMDGPU_HW_IP_UVD_ENC = 5 # type: ignore
AMDGPU_HW_IP_VCN_DEC = 6 # type: ignore
AMDGPU_HW_IP_VCN_ENC = 7 # type: ignore
AMDGPU_HW_IP_VCN_JPEG = 8 # type: ignore
AMDGPU_HW_IP_VPE = 9 # type: ignore
AMDGPU_HW_IP_NUM = 10 # type: ignore
AMDGPU_HW_IP_INSTANCE_MAX_COUNT = 1 # type: ignore
AMDGPU_CHUNK_ID_IB = 0x01 # type: ignore
AMDGPU_CHUNK_ID_FENCE = 0x02 # type: ignore
AMDGPU_CHUNK_ID_DEPENDENCIES = 0x03 # type: ignore
AMDGPU_CHUNK_ID_SYNCOBJ_IN = 0x04 # type: ignore
AMDGPU_CHUNK_ID_SYNCOBJ_OUT = 0x05 # type: ignore
AMDGPU_CHUNK_ID_BO_HANDLES = 0x06 # type: ignore
AMDGPU_CHUNK_ID_SCHEDULED_DEPENDENCIES = 0x07 # type: ignore
AMDGPU_CHUNK_ID_SYNCOBJ_TIMELINE_WAIT = 0x08 # type: ignore
AMDGPU_CHUNK_ID_SYNCOBJ_TIMELINE_SIGNAL = 0x09 # type: ignore
AMDGPU_CHUNK_ID_CP_GFX_SHADOW = 0x0a # type: ignore
AMDGPU_IB_FLAG_CE = (1<<0) # type: ignore
AMDGPU_IB_FLAG_PREAMBLE = (1<<1) # type: ignore
AMDGPU_IB_FLAG_PREEMPT = (1<<2) # type: ignore
AMDGPU_IB_FLAG_TC_WB_NOT_INVALIDATE = (1 << 3) # type: ignore
AMDGPU_IB_FLAG_RESET_GDS_MAX_WAVE_ID = (1 << 4) # type: ignore
AMDGPU_IB_FLAGS_SECURE = (1 << 5) # type: ignore
AMDGPU_IB_FLAG_EMIT_MEM_SYNC = (1 << 6) # type: ignore
AMDGPU_FENCE_TO_HANDLE_GET_SYNCOBJ = 0 # type: ignore
AMDGPU_FENCE_TO_HANDLE_GET_SYNCOBJ_FD = 1 # type: ignore
AMDGPU_FENCE_TO_HANDLE_GET_SYNC_FILE_FD = 2 # type: ignore
AMDGPU_CS_CHUNK_CP_GFX_SHADOW_FLAGS_INIT_SHADOW = 0x1 # type: ignore
AMDGPU_IDS_FLAGS_FUSION = 0x1 # type: ignore
AMDGPU_IDS_FLAGS_PREEMPTION = 0x2 # type: ignore
AMDGPU_IDS_FLAGS_TMZ = 0x4 # type: ignore
AMDGPU_IDS_FLAGS_CONFORMANT_TRUNC_COORD = 0x8 # type: ignore
AMDGPU_IDS_FLAGS_MODE_MASK = 0x300 # type: ignore
AMDGPU_IDS_FLAGS_MODE_SHIFT = 0x8 # type: ignore
AMDGPU_IDS_FLAGS_MODE_PF = 0x0 # type: ignore
AMDGPU_IDS_FLAGS_MODE_VF = 0x1 # type: ignore
AMDGPU_IDS_FLAGS_MODE_PT = 0x2 # type: ignore
AMDGPU_INFO_ACCEL_WORKING = 0x00 # type: ignore
AMDGPU_INFO_CRTC_FROM_ID = 0x01 # type: ignore
AMDGPU_INFO_HW_IP_INFO = 0x02 # type: ignore
AMDGPU_INFO_HW_IP_COUNT = 0x03 # type: ignore
AMDGPU_INFO_TIMESTAMP = 0x05 # type: ignore
AMDGPU_INFO_FW_VERSION = 0x0e # type: ignore
AMDGPU_INFO_FW_VCE = 0x1 # type: ignore
AMDGPU_INFO_FW_UVD = 0x2 # type: ignore
AMDGPU_INFO_FW_GMC = 0x03 # type: ignore
AMDGPU_INFO_FW_GFX_ME = 0x04 # type: ignore
AMDGPU_INFO_FW_GFX_PFP = 0x05 # type: ignore
AMDGPU_INFO_FW_GFX_CE = 0x06 # type: ignore
AMDGPU_INFO_FW_GFX_RLC = 0x07 # type: ignore
AMDGPU_INFO_FW_GFX_MEC = 0x08 # type: ignore
AMDGPU_INFO_FW_SMC = 0x0a # type: ignore
AMDGPU_INFO_FW_SDMA = 0x0b # type: ignore
AMDGPU_INFO_FW_SOS = 0x0c # type: ignore
AMDGPU_INFO_FW_ASD = 0x0d # type: ignore
AMDGPU_INFO_FW_VCN = 0x0e # type: ignore
AMDGPU_INFO_FW_GFX_RLC_RESTORE_LIST_CNTL = 0x0f # type: ignore
AMDGPU_INFO_FW_GFX_RLC_RESTORE_LIST_GPM_MEM = 0x10 # type: ignore
AMDGPU_INFO_FW_GFX_RLC_RESTORE_LIST_SRM_MEM = 0x11 # type: ignore
AMDGPU_INFO_FW_DMCU = 0x12 # type: ignore
AMDGPU_INFO_FW_TA = 0x13 # type: ignore
AMDGPU_INFO_FW_DMCUB = 0x14 # type: ignore
AMDGPU_INFO_FW_TOC = 0x15 # type: ignore
AMDGPU_INFO_FW_CAP = 0x16 # type: ignore
AMDGPU_INFO_FW_GFX_RLCP = 0x17 # type: ignore
AMDGPU_INFO_FW_GFX_RLCV = 0x18 # type: ignore
AMDGPU_INFO_FW_MES_KIQ = 0x19 # type: ignore
AMDGPU_INFO_FW_MES = 0x1a # type: ignore
AMDGPU_INFO_FW_IMU = 0x1b # type: ignore
AMDGPU_INFO_FW_VPE = 0x1c # type: ignore
AMDGPU_INFO_NUM_BYTES_MOVED = 0x0f # type: ignore
AMDGPU_INFO_VRAM_USAGE = 0x10 # type: ignore
AMDGPU_INFO_GTT_USAGE = 0x11 # type: ignore
AMDGPU_INFO_GDS_CONFIG = 0x13 # type: ignore
AMDGPU_INFO_VRAM_GTT = 0x14 # type: ignore
AMDGPU_INFO_READ_MMR_REG = 0x15 # type: ignore
AMDGPU_INFO_DEV_INFO = 0x16 # type: ignore
AMDGPU_INFO_VIS_VRAM_USAGE = 0x17 # type: ignore
AMDGPU_INFO_NUM_EVICTIONS = 0x18 # type: ignore
AMDGPU_INFO_MEMORY = 0x19 # type: ignore
AMDGPU_INFO_VCE_CLOCK_TABLE = 0x1A # type: ignore
AMDGPU_INFO_VBIOS = 0x1B # type: ignore
AMDGPU_INFO_VBIOS_SIZE = 0x1 # type: ignore
AMDGPU_INFO_VBIOS_IMAGE = 0x2 # type: ignore
AMDGPU_INFO_VBIOS_INFO = 0x3 # type: ignore
AMDGPU_INFO_NUM_HANDLES = 0x1C # type: ignore
AMDGPU_INFO_SENSOR = 0x1D # type: ignore
AMDGPU_INFO_SENSOR_GFX_SCLK = 0x1 # type: ignore
AMDGPU_INFO_SENSOR_GFX_MCLK = 0x2 # type: ignore
AMDGPU_INFO_SENSOR_GPU_TEMP = 0x3 # type: ignore
AMDGPU_INFO_SENSOR_GPU_LOAD = 0x4 # type: ignore
AMDGPU_INFO_SENSOR_GPU_AVG_POWER = 0x5 # type: ignore
AMDGPU_INFO_SENSOR_VDDNB = 0x6 # type: ignore
AMDGPU_INFO_SENSOR_VDDGFX = 0x7 # type: ignore
AMDGPU_INFO_SENSOR_STABLE_PSTATE_GFX_SCLK = 0x8 # type: ignore
AMDGPU_INFO_SENSOR_STABLE_PSTATE_GFX_MCLK = 0x9 # type: ignore
AMDGPU_INFO_SENSOR_PEAK_PSTATE_GFX_SCLK = 0xa # type: ignore
AMDGPU_INFO_SENSOR_PEAK_PSTATE_GFX_MCLK = 0xb # type: ignore
AMDGPU_INFO_SENSOR_GPU_INPUT_POWER = 0xc # type: ignore
AMDGPU_INFO_NUM_VRAM_CPU_PAGE_FAULTS = 0x1E # type: ignore
AMDGPU_INFO_VRAM_LOST_COUNTER = 0x1F # type: ignore
AMDGPU_INFO_RAS_ENABLED_FEATURES = 0x20 # type: ignore
AMDGPU_INFO_RAS_ENABLED_UMC = (1 << 0) # type: ignore
AMDGPU_INFO_RAS_ENABLED_SDMA = (1 << 1) # type: ignore
AMDGPU_INFO_RAS_ENABLED_GFX = (1 << 2) # type: ignore
AMDGPU_INFO_RAS_ENABLED_MMHUB = (1 << 3) # type: ignore
AMDGPU_INFO_RAS_ENABLED_ATHUB = (1 << 4) # type: ignore
AMDGPU_INFO_RAS_ENABLED_PCIE = (1 << 5) # type: ignore
AMDGPU_INFO_RAS_ENABLED_HDP = (1 << 6) # type: ignore
AMDGPU_INFO_RAS_ENABLED_XGMI = (1 << 7) # type: ignore
AMDGPU_INFO_RAS_ENABLED_DF = (1 << 8) # type: ignore
AMDGPU_INFO_RAS_ENABLED_SMN = (1 << 9) # type: ignore
AMDGPU_INFO_RAS_ENABLED_SEM = (1 << 10) # type: ignore
AMDGPU_INFO_RAS_ENABLED_MP0 = (1 << 11) # type: ignore
AMDGPU_INFO_RAS_ENABLED_MP1 = (1 << 12) # type: ignore
AMDGPU_INFO_RAS_ENABLED_FUSE = (1 << 13) # type: ignore
AMDGPU_INFO_VIDEO_CAPS = 0x21 # type: ignore
AMDGPU_INFO_VIDEO_CAPS_DECODE = 0 # type: ignore
AMDGPU_INFO_VIDEO_CAPS_ENCODE = 1 # type: ignore
AMDGPU_INFO_MAX_IBS = 0x22 # type: ignore
AMDGPU_INFO_GPUVM_FAULT = 0x23 # type: ignore
AMDGPU_INFO_UQ_FW_AREAS = 0x24 # type: ignore
AMDGPU_INFO_CAPABILITY = 0x50 # type: ignore
AMDGPU_INFO_VIRTUAL_RANGE = 0x51 # type: ignore
AMDGPU_CAPABILITY_PIN_MEM_FLAG = (1 << 0) # type: ignore
AMDGPU_CAPABILITY_DIRECT_GMA_FLAG = (1 << 1) # type: ignore
AMDGPU_INFO_MMR_SE_INDEX_SHIFT = 0 # type: ignore
AMDGPU_INFO_MMR_SE_INDEX_MASK = 0xff # type: ignore
AMDGPU_INFO_MMR_SH_INDEX_SHIFT = 8 # type: ignore
AMDGPU_INFO_MMR_SH_INDEX_MASK = 0xff # type: ignore
AMDGPU_VRAM_TYPE_UNKNOWN = 0 # type: ignore
AMDGPU_VRAM_TYPE_GDDR1 = 1 # type: ignore
AMDGPU_VRAM_TYPE_DDR2 = 2 # type: ignore
AMDGPU_VRAM_TYPE_GDDR3 = 3 # type: ignore
AMDGPU_VRAM_TYPE_GDDR4 = 4 # type: ignore
AMDGPU_VRAM_TYPE_GDDR5 = 5 # type: ignore
AMDGPU_VRAM_TYPE_HBM = 6 # type: ignore
AMDGPU_VRAM_TYPE_DDR3 = 7 # type: ignore
AMDGPU_VRAM_TYPE_DDR4 = 8 # type: ignore
AMDGPU_VRAM_TYPE_GDDR6 = 9 # type: ignore
AMDGPU_VRAM_TYPE_DDR5 = 10 # type: ignore
AMDGPU_VRAM_TYPE_LPDDR4 = 11 # type: ignore
AMDGPU_VRAM_TYPE_LPDDR5 = 12 # type: ignore
AMDGPU_VRAM_TYPE_HBM3E = 13 # type: ignore
AMDGPU_VRAM_TYPE_HBM_WIDTH = 4096 # type: ignore
AMDGPU_VCE_CLOCK_TABLE_ENTRIES = 6 # type: ignore
AMDGPU_INFO_VIDEO_CAPS_CODEC_IDX_MPEG2 = 0 # type: ignore
AMDGPU_INFO_VIDEO_CAPS_CODEC_IDX_MPEG4 = 1 # type: ignore
AMDGPU_INFO_VIDEO_CAPS_CODEC_IDX_VC1 = 2 # type: ignore
AMDGPU_INFO_VIDEO_CAPS_CODEC_IDX_MPEG4_AVC = 3 # type: ignore
AMDGPU_INFO_VIDEO_CAPS_CODEC_IDX_HEVC = 4 # type: ignore
AMDGPU_INFO_VIDEO_CAPS_CODEC_IDX_JPEG = 5 # type: ignore
AMDGPU_INFO_VIDEO_CAPS_CODEC_IDX_VP9 = 6 # type: ignore
AMDGPU_INFO_VIDEO_CAPS_CODEC_IDX_AV1 = 7 # type: ignore
AMDGPU_INFO_VIDEO_CAPS_CODEC_IDX_COUNT = 8 # type: ignore
AMDGPU_VMHUB_TYPE_MASK = 0xff # type: ignore
AMDGPU_VMHUB_TYPE_SHIFT = 0 # type: ignore
AMDGPU_VMHUB_TYPE_GFX = 0 # type: ignore
AMDGPU_VMHUB_TYPE_MM0 = 1 # type: ignore
AMDGPU_VMHUB_TYPE_MM1 = 2 # type: ignore
AMDGPU_VMHUB_IDX_MASK = 0xff00 # type: ignore
AMDGPU_VMHUB_IDX_SHIFT = 8 # type: ignore
AMDGPU_FAMILY_UNKNOWN = 0 # type: ignore
AMDGPU_FAMILY_SI = 110 # type: ignore
AMDGPU_FAMILY_CI = 120 # type: ignore
AMDGPU_FAMILY_KV = 125 # type: ignore
AMDGPU_FAMILY_VI = 130 # type: ignore
AMDGPU_FAMILY_CZ = 135 # type: ignore
AMDGPU_FAMILY_AI = 141 # type: ignore
AMDGPU_FAMILY_RV = 142 # type: ignore
AMDGPU_FAMILY_NV = 143 # type: ignore
AMDGPU_FAMILY_VGH = 144 # type: ignore
AMDGPU_FAMILY_GC_11_0_0 = 145 # type: ignore
AMDGPU_FAMILY_YC = 146 # type: ignore
AMDGPU_FAMILY_GC_11_0_1 = 148 # type: ignore
AMDGPU_FAMILY_GC_10_3_6 = 149 # type: ignore
AMDGPU_FAMILY_GC_10_3_7 = 151 # type: ignore
AMDGPU_FAMILY_GC_11_5_0 = 150 # type: ignore
AMDGPU_FAMILY_GC_12_0_0 = 152 # type: ignore
AMDGPU_SUA_APERTURE_PRIVATE = 1 # type: ignore
AMDGPU_SUA_APERTURE_SHARED = 2 # type: ignore
AMDGPU_FREESYNC_FULLSCREEN_ENTER = 1 # type: ignore
AMDGPU_FREESYNC_FULLSCREEN_EXIT = 2 # type: ignore