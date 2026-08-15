# mypy: disable-error-code="empty-body"
from __future__ import annotations
import ctypes
from typing import Literal, TypeAlias
from tinygrad.runtime.support.c import _IO, _IOW, _IOR, _IOWR
from tinygrad.runtime.support import c
drm_handle_t: TypeAlias = ctypes.c_uint32
drm_context_t: TypeAlias = ctypes.c_uint32
drm_drawable_t: TypeAlias = ctypes.c_uint32
drm_magic_t: TypeAlias = ctypes.c_uint32
@c.record
class struct_drm_clip_rect(c.Struct):
  SIZE = 8
  x1: int
  y1: int
  x2: int
  y2: int
struct_drm_clip_rect.register_fields([('x1', ctypes.c_uint16, 0), ('y1', ctypes.c_uint16, 2), ('x2', ctypes.c_uint16, 4), ('y2', ctypes.c_uint16, 6)])
@c.record
class struct_drm_drawable_info(c.Struct):
  SIZE = 16
  num_rects: int
  rects: c.POINTER[struct_drm_clip_rect]
struct_drm_drawable_info.register_fields([('num_rects', ctypes.c_uint32, 0), ('rects', c.POINTER[struct_drm_clip_rect], 8)])
@c.record
class struct_drm_tex_region(c.Struct):
  SIZE = 8
  next: int
  prev: int
  in_use: int
  padding: int
  age: int
struct_drm_tex_region.register_fields([('next', ctypes.c_ubyte, 0), ('prev', ctypes.c_ubyte, 1), ('in_use', ctypes.c_ubyte, 2), ('padding', ctypes.c_ubyte, 3), ('age', ctypes.c_uint32, 4)])
@c.record
class struct_drm_hw_lock(c.Struct):
  SIZE = 64
  lock: int
  padding: c.Array[ctypes.c_char, Literal[60]]
struct_drm_hw_lock.register_fields([('lock', ctypes.c_uint32, 0), ('padding', c.Array[ctypes.c_char, Literal[60]], 4)])
@c.record
class struct_drm_version(c.Struct):
  SIZE = 64
  version_major: int
  version_minor: int
  version_patchlevel: int
  name_len: int
  name: c.POINTER[ctypes.c_char]
  date_len: int
  date: c.POINTER[ctypes.c_char]
  desc_len: int
  desc: c.POINTER[ctypes.c_char]
__kernel_size_t: TypeAlias = ctypes.c_uint64
struct_drm_version.register_fields([('version_major', ctypes.c_int32, 0), ('version_minor', ctypes.c_int32, 4), ('version_patchlevel', ctypes.c_int32, 8), ('name_len', ctypes.c_uint64, 16), ('name', c.POINTER[ctypes.c_char], 24), ('date_len', ctypes.c_uint64, 32), ('date', c.POINTER[ctypes.c_char], 40), ('desc_len', ctypes.c_uint64, 48), ('desc', c.POINTER[ctypes.c_char], 56)])
@c.record
class struct_drm_unique(c.Struct):
  SIZE = 16
  unique_len: int
  unique: c.POINTER[ctypes.c_char]
struct_drm_unique.register_fields([('unique_len', ctypes.c_uint64, 0), ('unique', c.POINTER[ctypes.c_char], 8)])
@c.record
class struct_drm_list(c.Struct):
  SIZE = 16
  count: int
  version: c.POINTER[struct_drm_version]
struct_drm_list.register_fields([('count', ctypes.c_int32, 0), ('version', c.POINTER[struct_drm_version], 8)])
@c.record
class struct_drm_block(c.Struct):
  SIZE = 4
  unused: int
struct_drm_block.register_fields([('unused', ctypes.c_int32, 0)])
@c.record
class struct_drm_control(c.Struct):
  SIZE = 8
  func: int
  irq: int
struct_drm_control_func: dict[int, str] = {(DRM_ADD_COMMAND:=0): 'DRM_ADD_COMMAND', (DRM_RM_COMMAND:=1): 'DRM_RM_COMMAND', (DRM_INST_HANDLER:=2): 'DRM_INST_HANDLER', (DRM_UNINST_HANDLER:=3): 'DRM_UNINST_HANDLER'}
struct_drm_control.register_fields([('func', ctypes.c_uint32, 0), ('irq', ctypes.c_int32, 4)])
enum_drm_map_type: dict[int, str] = {(_DRM_FRAME_BUFFER:=0): '_DRM_FRAME_BUFFER', (_DRM_REGISTERS:=1): '_DRM_REGISTERS', (_DRM_SHM:=2): '_DRM_SHM', (_DRM_AGP:=3): '_DRM_AGP', (_DRM_SCATTER_GATHER:=4): '_DRM_SCATTER_GATHER', (_DRM_CONSISTENT:=5): '_DRM_CONSISTENT'}
enum_drm_map_flags: dict[int, str] = {(_DRM_RESTRICTED:=1): '_DRM_RESTRICTED', (_DRM_READ_ONLY:=2): '_DRM_READ_ONLY', (_DRM_LOCKED:=4): '_DRM_LOCKED', (_DRM_KERNEL:=8): '_DRM_KERNEL', (_DRM_WRITE_COMBINING:=16): '_DRM_WRITE_COMBINING', (_DRM_CONTAINS_LOCK:=32): '_DRM_CONTAINS_LOCK', (_DRM_REMOVABLE:=64): '_DRM_REMOVABLE', (_DRM_DRIVER:=128): '_DRM_DRIVER'}
@c.record
class struct_drm_ctx_priv_map(c.Struct):
  SIZE = 16
  ctx_id: int
  handle: ctypes.c_void_p
struct_drm_ctx_priv_map.register_fields([('ctx_id', ctypes.c_uint32, 0), ('handle', ctypes.c_void_p, 8)])
@c.record
class struct_drm_map(c.Struct):
  SIZE = 40
  offset: int
  size: int
  type: int
  flags: int
  handle: ctypes.c_void_p
  mtrr: int
struct_drm_map.register_fields([('offset', ctypes.c_uint64, 0), ('size', ctypes.c_uint64, 8), ('type', ctypes.c_uint32, 16), ('flags', ctypes.c_uint32, 20), ('handle', ctypes.c_void_p, 24), ('mtrr', ctypes.c_int32, 32)])
@c.record
class struct_drm_client(c.Struct):
  SIZE = 40
  idx: int
  auth: int
  pid: int
  uid: int
  magic: int
  iocs: int
struct_drm_client.register_fields([('idx', ctypes.c_int32, 0), ('auth', ctypes.c_int32, 4), ('pid', ctypes.c_uint64, 8), ('uid', ctypes.c_uint64, 16), ('magic', ctypes.c_uint64, 24), ('iocs', ctypes.c_uint64, 32)])
enum_drm_stat_type: dict[int, str] = {(_DRM_STAT_LOCK:=0): '_DRM_STAT_LOCK', (_DRM_STAT_OPENS:=1): '_DRM_STAT_OPENS', (_DRM_STAT_CLOSES:=2): '_DRM_STAT_CLOSES', (_DRM_STAT_IOCTLS:=3): '_DRM_STAT_IOCTLS', (_DRM_STAT_LOCKS:=4): '_DRM_STAT_LOCKS', (_DRM_STAT_UNLOCKS:=5): '_DRM_STAT_UNLOCKS', (_DRM_STAT_VALUE:=6): '_DRM_STAT_VALUE', (_DRM_STAT_BYTE:=7): '_DRM_STAT_BYTE', (_DRM_STAT_COUNT:=8): '_DRM_STAT_COUNT', (_DRM_STAT_IRQ:=9): '_DRM_STAT_IRQ', (_DRM_STAT_PRIMARY:=10): '_DRM_STAT_PRIMARY', (_DRM_STAT_SECONDARY:=11): '_DRM_STAT_SECONDARY', (_DRM_STAT_DMA:=12): '_DRM_STAT_DMA', (_DRM_STAT_SPECIAL:=13): '_DRM_STAT_SPECIAL', (_DRM_STAT_MISSED:=14): '_DRM_STAT_MISSED'}
@c.record
class struct_drm_stats(c.Struct):
  SIZE = 248
  count: int
  data: c.Array[struct_drm_stats_data, Literal[15]]
@c.record
class struct_drm_stats_data(c.Struct):
  SIZE = 16
  value: int
  type: int
struct_drm_stats_data.register_fields([('value', ctypes.c_uint64, 0), ('type', ctypes.c_uint32, 8)])
struct_drm_stats.register_fields([('count', ctypes.c_uint64, 0), ('data', c.Array[struct_drm_stats_data, Literal[15]], 8)])
enum_drm_lock_flags: dict[int, str] = {(_DRM_LOCK_READY:=1): '_DRM_LOCK_READY', (_DRM_LOCK_QUIESCENT:=2): '_DRM_LOCK_QUIESCENT', (_DRM_LOCK_FLUSH:=4): '_DRM_LOCK_FLUSH', (_DRM_LOCK_FLUSH_ALL:=8): '_DRM_LOCK_FLUSH_ALL', (_DRM_HALT_ALL_QUEUES:=16): '_DRM_HALT_ALL_QUEUES', (_DRM_HALT_CUR_QUEUES:=32): '_DRM_HALT_CUR_QUEUES'}
@c.record
class struct_drm_lock(c.Struct):
  SIZE = 8
  context: int
  flags: int
struct_drm_lock.register_fields([('context', ctypes.c_int32, 0), ('flags', ctypes.c_uint32, 4)])
enum_drm_dma_flags: dict[int, str] = {(_DRM_DMA_BLOCK:=1): '_DRM_DMA_BLOCK', (_DRM_DMA_WHILE_LOCKED:=2): '_DRM_DMA_WHILE_LOCKED', (_DRM_DMA_PRIORITY:=4): '_DRM_DMA_PRIORITY', (_DRM_DMA_WAIT:=16): '_DRM_DMA_WAIT', (_DRM_DMA_SMALLER_OK:=32): '_DRM_DMA_SMALLER_OK', (_DRM_DMA_LARGER_OK:=64): '_DRM_DMA_LARGER_OK'}
@c.record
class struct_drm_buf_desc(c.Struct):
  SIZE = 32
  count: int
  size: int
  low_mark: int
  high_mark: int
  flags: int
  agp_start: int
struct_drm_buf_desc_flags: dict[int, str] = {(_DRM_PAGE_ALIGN:=1): '_DRM_PAGE_ALIGN', (_DRM_AGP_BUFFER:=2): '_DRM_AGP_BUFFER', (_DRM_SG_BUFFER:=4): '_DRM_SG_BUFFER', (_DRM_FB_BUFFER:=8): '_DRM_FB_BUFFER', (_DRM_PCI_BUFFER_RO:=16): '_DRM_PCI_BUFFER_RO'}
struct_drm_buf_desc.register_fields([('count', ctypes.c_int32, 0), ('size', ctypes.c_int32, 4), ('low_mark', ctypes.c_int32, 8), ('high_mark', ctypes.c_int32, 12), ('flags', ctypes.c_uint32, 16), ('agp_start', ctypes.c_uint64, 24)])
@c.record
class struct_drm_buf_info(c.Struct):
  SIZE = 16
  count: int
  list: c.POINTER[struct_drm_buf_desc]
struct_drm_buf_info.register_fields([('count', ctypes.c_int32, 0), ('list', c.POINTER[struct_drm_buf_desc], 8)])
@c.record
class struct_drm_buf_free(c.Struct):
  SIZE = 16
  count: int
  list: c.POINTER[ctypes.c_int32]
struct_drm_buf_free.register_fields([('count', ctypes.c_int32, 0), ('list', c.POINTER[ctypes.c_int32], 8)])
@c.record
class struct_drm_buf_pub(c.Struct):
  SIZE = 24
  idx: int
  total: int
  used: int
  address: ctypes.c_void_p
struct_drm_buf_pub.register_fields([('idx', ctypes.c_int32, 0), ('total', ctypes.c_int32, 4), ('used', ctypes.c_int32, 8), ('address', ctypes.c_void_p, 16)])
@c.record
class struct_drm_buf_map(c.Struct):
  SIZE = 24
  count: int
  virtual: ctypes.c_void_p
  list: c.POINTER[struct_drm_buf_pub]
struct_drm_buf_map.register_fields([('count', ctypes.c_int32, 0), ('virtual', ctypes.c_void_p, 8), ('list', c.POINTER[struct_drm_buf_pub], 16)])
@c.record
class struct_drm_dma(c.Struct):
  SIZE = 64
  context: int
  send_count: int
  send_indices: c.POINTER[ctypes.c_int32]
  send_sizes: c.POINTER[ctypes.c_int32]
  flags: int
  request_count: int
  request_size: int
  request_indices: c.POINTER[ctypes.c_int32]
  request_sizes: c.POINTER[ctypes.c_int32]
  granted_count: int
struct_drm_dma.register_fields([('context', ctypes.c_int32, 0), ('send_count', ctypes.c_int32, 4), ('send_indices', c.POINTER[ctypes.c_int32], 8), ('send_sizes', c.POINTER[ctypes.c_int32], 16), ('flags', ctypes.c_uint32, 24), ('request_count', ctypes.c_int32, 28), ('request_size', ctypes.c_int32, 32), ('request_indices', c.POINTER[ctypes.c_int32], 40), ('request_sizes', c.POINTER[ctypes.c_int32], 48), ('granted_count', ctypes.c_int32, 56)])
enum_drm_ctx_flags: dict[int, str] = {(_DRM_CONTEXT_PRESERVED:=1): '_DRM_CONTEXT_PRESERVED', (_DRM_CONTEXT_2DONLY:=2): '_DRM_CONTEXT_2DONLY'}
@c.record
class struct_drm_ctx(c.Struct):
  SIZE = 8
  handle: int
  flags: int
struct_drm_ctx.register_fields([('handle', drm_context_t, 0), ('flags', ctypes.c_uint32, 4)])
@c.record
class struct_drm_ctx_res(c.Struct):
  SIZE = 16
  count: int
  contexts: c.POINTER[struct_drm_ctx]
struct_drm_ctx_res.register_fields([('count', ctypes.c_int32, 0), ('contexts', c.POINTER[struct_drm_ctx], 8)])
@c.record
class struct_drm_draw(c.Struct):
  SIZE = 4
  handle: int
struct_drm_draw.register_fields([('handle', drm_drawable_t, 0)])
drm_drawable_info_type_t: dict[int, str] = {(DRM_DRAWABLE_CLIPRECTS:=0): 'DRM_DRAWABLE_CLIPRECTS'}
@c.record
class struct_drm_update_draw(c.Struct):
  SIZE = 24
  handle: int
  type: int
  num: int
  data: int
struct_drm_update_draw.register_fields([('handle', drm_drawable_t, 0), ('type', ctypes.c_uint32, 4), ('num', ctypes.c_uint32, 8), ('data', ctypes.c_uint64, 16)])
@c.record
class struct_drm_auth(c.Struct):
  SIZE = 4
  magic: int
struct_drm_auth.register_fields([('magic', drm_magic_t, 0)])
@c.record
class struct_drm_irq_busid(c.Struct):
  SIZE = 16
  irq: int
  busnum: int
  devnum: int
  funcnum: int
struct_drm_irq_busid.register_fields([('irq', ctypes.c_int32, 0), ('busnum', ctypes.c_int32, 4), ('devnum', ctypes.c_int32, 8), ('funcnum', ctypes.c_int32, 12)])
enum_drm_vblank_seq_type: dict[int, str] = {(_DRM_VBLANK_ABSOLUTE:=0): '_DRM_VBLANK_ABSOLUTE', (_DRM_VBLANK_RELATIVE:=1): '_DRM_VBLANK_RELATIVE', (_DRM_VBLANK_HIGH_CRTC_MASK:=62): '_DRM_VBLANK_HIGH_CRTC_MASK', (_DRM_VBLANK_EVENT:=67108864): '_DRM_VBLANK_EVENT', (_DRM_VBLANK_FLIP:=134217728): '_DRM_VBLANK_FLIP', (_DRM_VBLANK_NEXTONMISS:=268435456): '_DRM_VBLANK_NEXTONMISS', (_DRM_VBLANK_SECONDARY:=536870912): '_DRM_VBLANK_SECONDARY', (_DRM_VBLANK_SIGNAL:=1073741824): '_DRM_VBLANK_SIGNAL'}
@c.record
class struct_drm_wait_vblank_request(c.Struct):
  SIZE = 16
  type: int
  sequence: int
  signal: int
struct_drm_wait_vblank_request.register_fields([('type', ctypes.c_uint32, 0), ('sequence', ctypes.c_uint32, 4), ('signal', ctypes.c_uint64, 8)])
@c.record
class struct_drm_wait_vblank_reply(c.Struct):
  SIZE = 24
  type: int
  sequence: int
  tval_sec: int
  tval_usec: int
struct_drm_wait_vblank_reply.register_fields([('type', ctypes.c_uint32, 0), ('sequence', ctypes.c_uint32, 4), ('tval_sec', ctypes.c_int64, 8), ('tval_usec', ctypes.c_int64, 16)])
@c.record
class union_drm_wait_vblank(c.Struct):
  SIZE = 24
  request: struct_drm_wait_vblank_request
  reply: struct_drm_wait_vblank_reply
union_drm_wait_vblank.register_fields([('request', struct_drm_wait_vblank_request, 0), ('reply', struct_drm_wait_vblank_reply, 0)])
@c.record
class struct_drm_modeset_ctl(c.Struct):
  SIZE = 8
  crtc: int
  cmd: int
__u32: TypeAlias = ctypes.c_uint32
struct_drm_modeset_ctl.register_fields([('crtc', ctypes.c_uint32, 0), ('cmd', ctypes.c_uint32, 4)])
@c.record
class struct_drm_agp_mode(c.Struct):
  SIZE = 8
  mode: int
struct_drm_agp_mode.register_fields([('mode', ctypes.c_uint64, 0)])
@c.record
class struct_drm_agp_buffer(c.Struct):
  SIZE = 32
  size: int
  handle: int
  type: int
  physical: int
struct_drm_agp_buffer.register_fields([('size', ctypes.c_uint64, 0), ('handle', ctypes.c_uint64, 8), ('type', ctypes.c_uint64, 16), ('physical', ctypes.c_uint64, 24)])
@c.record
class struct_drm_agp_binding(c.Struct):
  SIZE = 16
  handle: int
  offset: int
struct_drm_agp_binding.register_fields([('handle', ctypes.c_uint64, 0), ('offset', ctypes.c_uint64, 8)])
@c.record
class struct_drm_agp_info(c.Struct):
  SIZE = 56
  agp_version_major: int
  agp_version_minor: int
  mode: int
  aperture_base: int
  aperture_size: int
  memory_allowed: int
  memory_used: int
  id_vendor: int
  id_device: int
struct_drm_agp_info.register_fields([('agp_version_major', ctypes.c_int32, 0), ('agp_version_minor', ctypes.c_int32, 4), ('mode', ctypes.c_uint64, 8), ('aperture_base', ctypes.c_uint64, 16), ('aperture_size', ctypes.c_uint64, 24), ('memory_allowed', ctypes.c_uint64, 32), ('memory_used', ctypes.c_uint64, 40), ('id_vendor', ctypes.c_uint16, 48), ('id_device', ctypes.c_uint16, 50)])
@c.record
class struct_drm_scatter_gather(c.Struct):
  SIZE = 16
  size: int
  handle: int
struct_drm_scatter_gather.register_fields([('size', ctypes.c_uint64, 0), ('handle', ctypes.c_uint64, 8)])
@c.record
class struct_drm_set_version(c.Struct):
  SIZE = 16
  drm_di_major: int
  drm_di_minor: int
  drm_dd_major: int
  drm_dd_minor: int
struct_drm_set_version.register_fields([('drm_di_major', ctypes.c_int32, 0), ('drm_di_minor', ctypes.c_int32, 4), ('drm_dd_major', ctypes.c_int32, 8), ('drm_dd_minor', ctypes.c_int32, 12)])
@c.record
class struct_drm_gem_close(c.Struct):
  SIZE = 8
  handle: int
  pad: int
struct_drm_gem_close.register_fields([('handle', ctypes.c_uint32, 0), ('pad', ctypes.c_uint32, 4)])
@c.record
class struct_drm_gem_flink(c.Struct):
  SIZE = 8
  handle: int
  name: int
struct_drm_gem_flink.register_fields([('handle', ctypes.c_uint32, 0), ('name', ctypes.c_uint32, 4)])
@c.record
class struct_drm_gem_open(c.Struct):
  SIZE = 16
  name: int
  handle: int
  size: int
__u64: TypeAlias = ctypes.c_uint64
struct_drm_gem_open.register_fields([('name', ctypes.c_uint32, 0), ('handle', ctypes.c_uint32, 4), ('size', ctypes.c_uint64, 8)])
@c.record
class struct_drm_get_cap(c.Struct):
  SIZE = 16
  capability: int
  value: int
struct_drm_get_cap.register_fields([('capability', ctypes.c_uint64, 0), ('value', ctypes.c_uint64, 8)])
@c.record
class struct_drm_set_client_cap(c.Struct):
  SIZE = 16
  capability: int
  value: int
struct_drm_set_client_cap.register_fields([('capability', ctypes.c_uint64, 0), ('value', ctypes.c_uint64, 8)])
@c.record
class struct_drm_prime_handle(c.Struct):
  SIZE = 12
  handle: int
  flags: int
  fd: int
__s32: TypeAlias = ctypes.c_int32
struct_drm_prime_handle.register_fields([('handle', ctypes.c_uint32, 0), ('flags', ctypes.c_uint32, 4), ('fd', ctypes.c_int32, 8)])
@c.record
class struct_drm_syncobj_create(c.Struct):
  SIZE = 8
  handle: int
  flags: int
struct_drm_syncobj_create.register_fields([('handle', ctypes.c_uint32, 0), ('flags', ctypes.c_uint32, 4)])
@c.record
class struct_drm_syncobj_destroy(c.Struct):
  SIZE = 8
  handle: int
  pad: int
struct_drm_syncobj_destroy.register_fields([('handle', ctypes.c_uint32, 0), ('pad', ctypes.c_uint32, 4)])
@c.record
class struct_drm_syncobj_handle(c.Struct):
  SIZE = 16
  handle: int
  flags: int
  fd: int
  pad: int
struct_drm_syncobj_handle.register_fields([('handle', ctypes.c_uint32, 0), ('flags', ctypes.c_uint32, 4), ('fd', ctypes.c_int32, 8), ('pad', ctypes.c_uint32, 12)])
@c.record
class struct_drm_syncobj_transfer(c.Struct):
  SIZE = 32
  src_handle: int
  dst_handle: int
  src_point: int
  dst_point: int
  flags: int
  pad: int
struct_drm_syncobj_transfer.register_fields([('src_handle', ctypes.c_uint32, 0), ('dst_handle', ctypes.c_uint32, 4), ('src_point', ctypes.c_uint64, 8), ('dst_point', ctypes.c_uint64, 16), ('flags', ctypes.c_uint32, 24), ('pad', ctypes.c_uint32, 28)])
@c.record
class struct_drm_syncobj_wait(c.Struct):
  SIZE = 40
  handles: int
  timeout_nsec: int
  count_handles: int
  flags: int
  first_signaled: int
  pad: int
  deadline_nsec: int
__s64: TypeAlias = ctypes.c_int64
struct_drm_syncobj_wait.register_fields([('handles', ctypes.c_uint64, 0), ('timeout_nsec', ctypes.c_int64, 8), ('count_handles', ctypes.c_uint32, 16), ('flags', ctypes.c_uint32, 20), ('first_signaled', ctypes.c_uint32, 24), ('pad', ctypes.c_uint32, 28), ('deadline_nsec', ctypes.c_uint64, 32)])
@c.record
class struct_drm_syncobj_timeline_wait(c.Struct):
  SIZE = 48
  handles: int
  points: int
  timeout_nsec: int
  count_handles: int
  flags: int
  first_signaled: int
  pad: int
  deadline_nsec: int
struct_drm_syncobj_timeline_wait.register_fields([('handles', ctypes.c_uint64, 0), ('points', ctypes.c_uint64, 8), ('timeout_nsec', ctypes.c_int64, 16), ('count_handles', ctypes.c_uint32, 24), ('flags', ctypes.c_uint32, 28), ('first_signaled', ctypes.c_uint32, 32), ('pad', ctypes.c_uint32, 36), ('deadline_nsec', ctypes.c_uint64, 40)])
@c.record
class struct_drm_syncobj_eventfd(c.Struct):
  SIZE = 24
  handle: int
  flags: int
  point: int
  fd: int
  pad: int
struct_drm_syncobj_eventfd.register_fields([('handle', ctypes.c_uint32, 0), ('flags', ctypes.c_uint32, 4), ('point', ctypes.c_uint64, 8), ('fd', ctypes.c_int32, 16), ('pad', ctypes.c_uint32, 20)])
@c.record
class struct_drm_syncobj_array(c.Struct):
  SIZE = 16
  handles: int
  count_handles: int
  pad: int
struct_drm_syncobj_array.register_fields([('handles', ctypes.c_uint64, 0), ('count_handles', ctypes.c_uint32, 8), ('pad', ctypes.c_uint32, 12)])
@c.record
class struct_drm_syncobj_timeline_array(c.Struct):
  SIZE = 24
  handles: int
  points: int
  count_handles: int
  flags: int
struct_drm_syncobj_timeline_array.register_fields([('handles', ctypes.c_uint64, 0), ('points', ctypes.c_uint64, 8), ('count_handles', ctypes.c_uint32, 16), ('flags', ctypes.c_uint32, 20)])
@c.record
class struct_drm_crtc_get_sequence(c.Struct):
  SIZE = 24
  crtc_id: int
  active: int
  sequence: int
  sequence_ns: int
struct_drm_crtc_get_sequence.register_fields([('crtc_id', ctypes.c_uint32, 0), ('active', ctypes.c_uint32, 4), ('sequence', ctypes.c_uint64, 8), ('sequence_ns', ctypes.c_int64, 16)])
@c.record
class struct_drm_crtc_queue_sequence(c.Struct):
  SIZE = 24
  crtc_id: int
  flags: int
  sequence: int
  user_data: int
struct_drm_crtc_queue_sequence.register_fields([('crtc_id', ctypes.c_uint32, 0), ('flags', ctypes.c_uint32, 4), ('sequence', ctypes.c_uint64, 8), ('user_data', ctypes.c_uint64, 16)])
@c.record
class struct_drm_event(c.Struct):
  SIZE = 8
  type: int
  length: int
struct_drm_event.register_fields([('type', ctypes.c_uint32, 0), ('length', ctypes.c_uint32, 4)])
@c.record
class struct_drm_event_vblank(c.Struct):
  SIZE = 32
  base: struct_drm_event
  user_data: int
  tv_sec: int
  tv_usec: int
  sequence: int
  crtc_id: int
struct_drm_event_vblank.register_fields([('base', struct_drm_event, 0), ('user_data', ctypes.c_uint64, 8), ('tv_sec', ctypes.c_uint32, 16), ('tv_usec', ctypes.c_uint32, 20), ('sequence', ctypes.c_uint32, 24), ('crtc_id', ctypes.c_uint32, 28)])
@c.record
class struct_drm_event_crtc_sequence(c.Struct):
  SIZE = 32
  base: struct_drm_event
  user_data: int
  time_ns: int
  sequence: int
struct_drm_event_crtc_sequence.register_fields([('base', struct_drm_event, 0), ('user_data', ctypes.c_uint64, 8), ('time_ns', ctypes.c_int64, 16), ('sequence', ctypes.c_uint64, 24)])
drm_clip_rect_t: TypeAlias = struct_drm_clip_rect
drm_drawable_info_t: TypeAlias = struct_drm_drawable_info
drm_tex_region_t: TypeAlias = struct_drm_tex_region
drm_hw_lock_t: TypeAlias = struct_drm_hw_lock
drm_version_t: TypeAlias = struct_drm_version
drm_unique_t: TypeAlias = struct_drm_unique
drm_list_t: TypeAlias = struct_drm_list
drm_block_t: TypeAlias = struct_drm_block
drm_control_t: TypeAlias = struct_drm_control
drm_map_type_t: TypeAlias = ctypes.c_uint32
drm_map_flags_t: TypeAlias = ctypes.c_uint32
drm_ctx_priv_map_t: TypeAlias = struct_drm_ctx_priv_map
drm_map_t: TypeAlias = struct_drm_map
drm_client_t: TypeAlias = struct_drm_client
drm_stat_type_t: TypeAlias = ctypes.c_uint32
drm_stats_t: TypeAlias = struct_drm_stats
drm_lock_flags_t: TypeAlias = ctypes.c_uint32
drm_lock_t: TypeAlias = struct_drm_lock
drm_dma_flags_t: TypeAlias = ctypes.c_uint32
drm_buf_desc_t: TypeAlias = struct_drm_buf_desc
drm_buf_info_t: TypeAlias = struct_drm_buf_info
drm_buf_free_t: TypeAlias = struct_drm_buf_free
drm_buf_pub_t: TypeAlias = struct_drm_buf_pub
drm_buf_map_t: TypeAlias = struct_drm_buf_map
drm_dma_t: TypeAlias = struct_drm_dma
drm_wait_vblank_t: TypeAlias = union_drm_wait_vblank
drm_agp_mode_t: TypeAlias = struct_drm_agp_mode
drm_ctx_flags_t: TypeAlias = ctypes.c_uint32
drm_ctx_t: TypeAlias = struct_drm_ctx
drm_ctx_res_t: TypeAlias = struct_drm_ctx_res
drm_draw_t: TypeAlias = struct_drm_draw
drm_update_draw_t: TypeAlias = struct_drm_update_draw
drm_auth_t: TypeAlias = struct_drm_auth
drm_irq_busid_t: TypeAlias = struct_drm_irq_busid
drm_vblank_seq_type_t: TypeAlias = ctypes.c_uint32
drm_agp_buffer_t: TypeAlias = struct_drm_agp_buffer
drm_agp_binding_t: TypeAlias = struct_drm_agp_binding
drm_agp_info_t: TypeAlias = struct_drm_agp_info
drm_scatter_gather_t: TypeAlias = struct_drm_scatter_gather
drm_set_version_t: TypeAlias = struct_drm_set_version
@c.record
class struct_drm_amdgpu_gem_create_in(c.Struct):
  SIZE = 32
  bo_size: int
  alignment: int
  domains: int
  domain_flags: int
struct_drm_amdgpu_gem_create_in.register_fields([('bo_size', ctypes.c_uint64, 0), ('alignment', ctypes.c_uint64, 8), ('domains', ctypes.c_uint64, 16), ('domain_flags', ctypes.c_uint64, 24)])
@c.record
class struct_drm_amdgpu_gem_create_out(c.Struct):
  SIZE = 8
  handle: int
  _pad: int
struct_drm_amdgpu_gem_create_out.register_fields([('handle', ctypes.c_uint32, 0), ('_pad', ctypes.c_uint32, 4)])
@c.record
class union_drm_amdgpu_gem_create(c.Struct):
  SIZE = 32
  _in: struct_drm_amdgpu_gem_create_in
  out: struct_drm_amdgpu_gem_create_out
union_drm_amdgpu_gem_create.register_fields([('_in', struct_drm_amdgpu_gem_create_in, 0), ('out', struct_drm_amdgpu_gem_create_out, 0)])
@c.record
class struct_drm_amdgpu_bo_list_in(c.Struct):
  SIZE = 24
  operation: int
  list_handle: int
  bo_number: int
  bo_info_size: int
  bo_info_ptr: int
struct_drm_amdgpu_bo_list_in.register_fields([('operation', ctypes.c_uint32, 0), ('list_handle', ctypes.c_uint32, 4), ('bo_number', ctypes.c_uint32, 8), ('bo_info_size', ctypes.c_uint32, 12), ('bo_info_ptr', ctypes.c_uint64, 16)])
@c.record
class struct_drm_amdgpu_bo_list_entry(c.Struct):
  SIZE = 8
  bo_handle: int
  bo_priority: int
struct_drm_amdgpu_bo_list_entry.register_fields([('bo_handle', ctypes.c_uint32, 0), ('bo_priority', ctypes.c_uint32, 4)])
@c.record
class struct_drm_amdgpu_bo_list_out(c.Struct):
  SIZE = 8
  list_handle: int
  _pad: int
struct_drm_amdgpu_bo_list_out.register_fields([('list_handle', ctypes.c_uint32, 0), ('_pad', ctypes.c_uint32, 4)])
@c.record
class union_drm_amdgpu_bo_list(c.Struct):
  SIZE = 24
  _in: struct_drm_amdgpu_bo_list_in
  out: struct_drm_amdgpu_bo_list_out
union_drm_amdgpu_bo_list.register_fields([('_in', struct_drm_amdgpu_bo_list_in, 0), ('out', struct_drm_amdgpu_bo_list_out, 0)])
@c.record
class struct_drm_amdgpu_ctx_in(c.Struct):
  SIZE = 16
  op: int
  flags: int
  ctx_id: int
  priority: int
struct_drm_amdgpu_ctx_in.register_fields([('op', ctypes.c_uint32, 0), ('flags', ctypes.c_uint32, 4), ('ctx_id', ctypes.c_uint32, 8), ('priority', ctypes.c_int32, 12)])
@c.record
class union_drm_amdgpu_ctx_out(c.Struct):
  SIZE = 16
  alloc: union_drm_amdgpu_ctx_out_alloc
  state: union_drm_amdgpu_ctx_out_state
  pstate: union_drm_amdgpu_ctx_out_pstate
@c.record
class union_drm_amdgpu_ctx_out_alloc(c.Struct):
  SIZE = 8
  ctx_id: int
  _pad: int
union_drm_amdgpu_ctx_out_alloc.register_fields([('ctx_id', ctypes.c_uint32, 0), ('_pad', ctypes.c_uint32, 4)])
@c.record
class union_drm_amdgpu_ctx_out_state(c.Struct):
  SIZE = 16
  flags: int
  hangs: int
  reset_status: int
union_drm_amdgpu_ctx_out_state.register_fields([('flags', ctypes.c_uint64, 0), ('hangs', ctypes.c_uint32, 8), ('reset_status', ctypes.c_uint32, 12)])
@c.record
class union_drm_amdgpu_ctx_out_pstate(c.Struct):
  SIZE = 8
  flags: int
  _pad: int
union_drm_amdgpu_ctx_out_pstate.register_fields([('flags', ctypes.c_uint32, 0), ('_pad', ctypes.c_uint32, 4)])
union_drm_amdgpu_ctx_out.register_fields([('alloc', union_drm_amdgpu_ctx_out_alloc, 0), ('state', union_drm_amdgpu_ctx_out_state, 0), ('pstate', union_drm_amdgpu_ctx_out_pstate, 0)])
@c.record
class union_drm_amdgpu_ctx(c.Struct):
  SIZE = 16
  _in: struct_drm_amdgpu_ctx_in
  out: union_drm_amdgpu_ctx_out
union_drm_amdgpu_ctx.register_fields([('_in', struct_drm_amdgpu_ctx_in, 0), ('out', union_drm_amdgpu_ctx_out, 0)])
@c.record
class struct_drm_amdgpu_userq_in(c.Struct):
  SIZE = 72
  op: int
  queue_id: int
  ip_type: int
  doorbell_handle: int
  doorbell_offset: int
  flags: int
  queue_va: int
  queue_size: int
  rptr_va: int
  wptr_va: int
  mqd: int
  mqd_size: int
struct_drm_amdgpu_userq_in.register_fields([('op', ctypes.c_uint32, 0), ('queue_id', ctypes.c_uint32, 4), ('ip_type', ctypes.c_uint32, 8), ('doorbell_handle', ctypes.c_uint32, 12), ('doorbell_offset', ctypes.c_uint32, 16), ('flags', ctypes.c_uint32, 20), ('queue_va', ctypes.c_uint64, 24), ('queue_size', ctypes.c_uint64, 32), ('rptr_va', ctypes.c_uint64, 40), ('wptr_va', ctypes.c_uint64, 48), ('mqd', ctypes.c_uint64, 56), ('mqd_size', ctypes.c_uint64, 64)])
@c.record
class struct_drm_amdgpu_userq_out(c.Struct):
  SIZE = 8
  queue_id: int
  _pad: int
struct_drm_amdgpu_userq_out.register_fields([('queue_id', ctypes.c_uint32, 0), ('_pad', ctypes.c_uint32, 4)])
@c.record
class union_drm_amdgpu_userq(c.Struct):
  SIZE = 72
  _in: struct_drm_amdgpu_userq_in
  out: struct_drm_amdgpu_userq_out
union_drm_amdgpu_userq.register_fields([('_in', struct_drm_amdgpu_userq_in, 0), ('out', struct_drm_amdgpu_userq_out, 0)])
@c.record
class struct_drm_amdgpu_userq_mqd_gfx11(c.Struct):
  SIZE = 16
  shadow_va: int
  csa_va: int
struct_drm_amdgpu_userq_mqd_gfx11.register_fields([('shadow_va', ctypes.c_uint64, 0), ('csa_va', ctypes.c_uint64, 8)])
@c.record
class struct_drm_amdgpu_userq_mqd_sdma_gfx11(c.Struct):
  SIZE = 8
  csa_va: int
struct_drm_amdgpu_userq_mqd_sdma_gfx11.register_fields([('csa_va', ctypes.c_uint64, 0)])
@c.record
class struct_drm_amdgpu_userq_mqd_compute_gfx11(c.Struct):
  SIZE = 8
  eop_va: int
struct_drm_amdgpu_userq_mqd_compute_gfx11.register_fields([('eop_va', ctypes.c_uint64, 0)])
@c.record
class struct_drm_amdgpu_userq_signal(c.Struct):
  SIZE = 48
  queue_id: int
  pad: int
  syncobj_handles: int
  num_syncobj_handles: int
  bo_read_handles: int
  bo_write_handles: int
  num_bo_read_handles: int
  num_bo_write_handles: int
struct_drm_amdgpu_userq_signal.register_fields([('queue_id', ctypes.c_uint32, 0), ('pad', ctypes.c_uint32, 4), ('syncobj_handles', ctypes.c_uint64, 8), ('num_syncobj_handles', ctypes.c_uint64, 16), ('bo_read_handles', ctypes.c_uint64, 24), ('bo_write_handles', ctypes.c_uint64, 32), ('num_bo_read_handles', ctypes.c_uint32, 40), ('num_bo_write_handles', ctypes.c_uint32, 44)])
@c.record
class struct_drm_amdgpu_userq_fence_info(c.Struct):
  SIZE = 16
  va: int
  value: int
struct_drm_amdgpu_userq_fence_info.register_fields([('va', ctypes.c_uint64, 0), ('value', ctypes.c_uint64, 8)])
@c.record
class struct_drm_amdgpu_userq_wait(c.Struct):
  SIZE = 72
  waitq_id: int
  pad: int
  syncobj_handles: int
  syncobj_timeline_handles: int
  syncobj_timeline_points: int
  bo_read_handles: int
  bo_write_handles: int
  num_syncobj_timeline_handles: int
  num_fences: int
  num_syncobj_handles: int
  num_bo_read_handles: int
  num_bo_write_handles: int
  out_fences: int
__u16: TypeAlias = ctypes.c_uint16
struct_drm_amdgpu_userq_wait.register_fields([('waitq_id', ctypes.c_uint32, 0), ('pad', ctypes.c_uint32, 4), ('syncobj_handles', ctypes.c_uint64, 8), ('syncobj_timeline_handles', ctypes.c_uint64, 16), ('syncobj_timeline_points', ctypes.c_uint64, 24), ('bo_read_handles', ctypes.c_uint64, 32), ('bo_write_handles', ctypes.c_uint64, 40), ('num_syncobj_timeline_handles', ctypes.c_uint16, 48), ('num_fences', ctypes.c_uint16, 50), ('num_syncobj_handles', ctypes.c_uint32, 52), ('num_bo_read_handles', ctypes.c_uint32, 56), ('num_bo_write_handles', ctypes.c_uint32, 60), ('out_fences', ctypes.c_uint64, 64)])
class struct_drm_amdgpu_sem_in(c.Struct): pass
class union_drm_amdgpu_sem_out(c.Struct): pass
class union_drm_amdgpu_sem(c.Struct): pass
@c.record
class struct_drm_amdgpu_vm_in(c.Struct):
  SIZE = 8
  op: int
  flags: int
struct_drm_amdgpu_vm_in.register_fields([('op', ctypes.c_uint32, 0), ('flags', ctypes.c_uint32, 4)])
@c.record
class struct_drm_amdgpu_vm_out(c.Struct):
  SIZE = 8
  flags: int
struct_drm_amdgpu_vm_out.register_fields([('flags', ctypes.c_uint64, 0)])
@c.record
class union_drm_amdgpu_vm(c.Struct):
  SIZE = 8
  _in: struct_drm_amdgpu_vm_in
  out: struct_drm_amdgpu_vm_out
union_drm_amdgpu_vm.register_fields([('_in', struct_drm_amdgpu_vm_in, 0), ('out', struct_drm_amdgpu_vm_out, 0)])
@c.record
class struct_drm_amdgpu_sched_in(c.Struct):
  SIZE = 16
  op: int
  fd: int
  priority: int
  ctx_id: int
struct_drm_amdgpu_sched_in.register_fields([('op', ctypes.c_uint32, 0), ('fd', ctypes.c_uint32, 4), ('priority', ctypes.c_int32, 8), ('ctx_id', ctypes.c_uint32, 12)])
@c.record
class union_drm_amdgpu_sched(c.Struct):
  SIZE = 16
  _in: struct_drm_amdgpu_sched_in
union_drm_amdgpu_sched.register_fields([('_in', struct_drm_amdgpu_sched_in, 0)])
@c.record
class struct_drm_amdgpu_gem_userptr(c.Struct):
  SIZE = 24
  addr: int
  size: int
  flags: int
  handle: int
struct_drm_amdgpu_gem_userptr.register_fields([('addr', ctypes.c_uint64, 0), ('size', ctypes.c_uint64, 8), ('flags', ctypes.c_uint32, 16), ('handle', ctypes.c_uint32, 20)])
@c.record
class struct_drm_amdgpu_gem_dgma(c.Struct):
  SIZE = 24
  addr: int
  size: int
  op: int
  handle: int
struct_drm_amdgpu_gem_dgma.register_fields([('addr', ctypes.c_uint64, 0), ('size', ctypes.c_uint64, 8), ('op', ctypes.c_uint32, 16), ('handle', ctypes.c_uint32, 20)])
@c.record
class struct_drm_amdgpu_gem_metadata(c.Struct):
  SIZE = 288
  handle: int
  op: int
  data: struct_drm_amdgpu_gem_metadata_data
@c.record
class struct_drm_amdgpu_gem_metadata_data(c.Struct):
  SIZE = 280
  flags: int
  tiling_info: int
  data_size_bytes: int
  data: c.Array[ctypes.c_uint32, Literal[64]]
struct_drm_amdgpu_gem_metadata_data.register_fields([('flags', ctypes.c_uint64, 0), ('tiling_info', ctypes.c_uint64, 8), ('data_size_bytes', ctypes.c_uint32, 16), ('data', c.Array[ctypes.c_uint32, Literal[64]], 20)])
struct_drm_amdgpu_gem_metadata.register_fields([('handle', ctypes.c_uint32, 0), ('op', ctypes.c_uint32, 4), ('data', struct_drm_amdgpu_gem_metadata_data, 8)])
@c.record
class struct_drm_amdgpu_gem_mmap_in(c.Struct):
  SIZE = 8
  handle: int
  _pad: int
struct_drm_amdgpu_gem_mmap_in.register_fields([('handle', ctypes.c_uint32, 0), ('_pad', ctypes.c_uint32, 4)])
@c.record
class struct_drm_amdgpu_gem_mmap_out(c.Struct):
  SIZE = 8
  addr_ptr: int
struct_drm_amdgpu_gem_mmap_out.register_fields([('addr_ptr', ctypes.c_uint64, 0)])
@c.record
class union_drm_amdgpu_gem_mmap(c.Struct):
  SIZE = 8
  _in: struct_drm_amdgpu_gem_mmap_in
  out: struct_drm_amdgpu_gem_mmap_out
union_drm_amdgpu_gem_mmap.register_fields([('_in', struct_drm_amdgpu_gem_mmap_in, 0), ('out', struct_drm_amdgpu_gem_mmap_out, 0)])
@c.record
class struct_drm_amdgpu_gem_wait_idle_in(c.Struct):
  SIZE = 16
  handle: int
  flags: int
  timeout: int
struct_drm_amdgpu_gem_wait_idle_in.register_fields([('handle', ctypes.c_uint32, 0), ('flags', ctypes.c_uint32, 4), ('timeout', ctypes.c_uint64, 8)])
@c.record
class struct_drm_amdgpu_gem_wait_idle_out(c.Struct):
  SIZE = 8
  status: int
  domain: int
struct_drm_amdgpu_gem_wait_idle_out.register_fields([('status', ctypes.c_uint32, 0), ('domain', ctypes.c_uint32, 4)])
@c.record
class union_drm_amdgpu_gem_wait_idle(c.Struct):
  SIZE = 16
  _in: struct_drm_amdgpu_gem_wait_idle_in
  out: struct_drm_amdgpu_gem_wait_idle_out
union_drm_amdgpu_gem_wait_idle.register_fields([('_in', struct_drm_amdgpu_gem_wait_idle_in, 0), ('out', struct_drm_amdgpu_gem_wait_idle_out, 0)])
@c.record
class struct_drm_amdgpu_wait_cs_in(c.Struct):
  SIZE = 32
  handle: int
  timeout: int
  ip_type: int
  ip_instance: int
  ring: int
  ctx_id: int
struct_drm_amdgpu_wait_cs_in.register_fields([('handle', ctypes.c_uint64, 0), ('timeout', ctypes.c_uint64, 8), ('ip_type', ctypes.c_uint32, 16), ('ip_instance', ctypes.c_uint32, 20), ('ring', ctypes.c_uint32, 24), ('ctx_id', ctypes.c_uint32, 28)])
@c.record
class struct_drm_amdgpu_wait_cs_out(c.Struct):
  SIZE = 8
  status: int
struct_drm_amdgpu_wait_cs_out.register_fields([('status', ctypes.c_uint64, 0)])
@c.record
class union_drm_amdgpu_wait_cs(c.Struct):
  SIZE = 32
  _in: struct_drm_amdgpu_wait_cs_in
  out: struct_drm_amdgpu_wait_cs_out
union_drm_amdgpu_wait_cs.register_fields([('_in', struct_drm_amdgpu_wait_cs_in, 0), ('out', struct_drm_amdgpu_wait_cs_out, 0)])
@c.record
class struct_drm_amdgpu_fence(c.Struct):
  SIZE = 24
  ctx_id: int
  ip_type: int
  ip_instance: int
  ring: int
  seq_no: int
struct_drm_amdgpu_fence.register_fields([('ctx_id', ctypes.c_uint32, 0), ('ip_type', ctypes.c_uint32, 4), ('ip_instance', ctypes.c_uint32, 8), ('ring', ctypes.c_uint32, 12), ('seq_no', ctypes.c_uint64, 16)])
@c.record
class struct_drm_amdgpu_wait_fences_in(c.Struct):
  SIZE = 24
  fences: int
  fence_count: int
  wait_all: int
  timeout_ns: int
struct_drm_amdgpu_wait_fences_in.register_fields([('fences', ctypes.c_uint64, 0), ('fence_count', ctypes.c_uint32, 8), ('wait_all', ctypes.c_uint32, 12), ('timeout_ns', ctypes.c_uint64, 16)])
@c.record
class struct_drm_amdgpu_wait_fences_out(c.Struct):
  SIZE = 8
  status: int
  first_signaled: int
struct_drm_amdgpu_wait_fences_out.register_fields([('status', ctypes.c_uint32, 0), ('first_signaled', ctypes.c_uint32, 4)])
@c.record
class union_drm_amdgpu_wait_fences(c.Struct):
  SIZE = 24
  _in: struct_drm_amdgpu_wait_fences_in
  out: struct_drm_amdgpu_wait_fences_out
union_drm_amdgpu_wait_fences.register_fields([('_in', struct_drm_amdgpu_wait_fences_in, 0), ('out', struct_drm_amdgpu_wait_fences_out, 0)])
@c.record
class struct_drm_amdgpu_gem_op(c.Struct):
  SIZE = 16
  handle: int
  op: int
  value: int
struct_drm_amdgpu_gem_op.register_fields([('handle', ctypes.c_uint32, 0), ('op', ctypes.c_uint32, 4), ('value', ctypes.c_uint64, 8)])
@c.record
class struct_drm_amdgpu_gem_va(c.Struct):
  SIZE = 64
  handle: int
  _pad: int
  operation: int
  flags: int
  va_address: int
  offset_in_bo: int
  map_size: int
  vm_timeline_point: int
  vm_timeline_syncobj_out: int
  num_syncobj_handles: int
  input_fence_syncobj_handles: int
struct_drm_amdgpu_gem_va.register_fields([('handle', ctypes.c_uint32, 0), ('_pad', ctypes.c_uint32, 4), ('operation', ctypes.c_uint32, 8), ('flags', ctypes.c_uint32, 12), ('va_address', ctypes.c_uint64, 16), ('offset_in_bo', ctypes.c_uint64, 24), ('map_size', ctypes.c_uint64, 32), ('vm_timeline_point', ctypes.c_uint64, 40), ('vm_timeline_syncobj_out', ctypes.c_uint32, 48), ('num_syncobj_handles', ctypes.c_uint32, 52), ('input_fence_syncobj_handles', ctypes.c_uint64, 56)])
@c.record
class struct_drm_amdgpu_cs_chunk(c.Struct):
  SIZE = 16
  chunk_id: int
  length_dw: int
  chunk_data: int
struct_drm_amdgpu_cs_chunk.register_fields([('chunk_id', ctypes.c_uint32, 0), ('length_dw', ctypes.c_uint32, 4), ('chunk_data', ctypes.c_uint64, 8)])
@c.record
class struct_drm_amdgpu_cs_in(c.Struct):
  SIZE = 24
  ctx_id: int
  bo_list_handle: int
  num_chunks: int
  flags: int
  chunks: int
struct_drm_amdgpu_cs_in.register_fields([('ctx_id', ctypes.c_uint32, 0), ('bo_list_handle', ctypes.c_uint32, 4), ('num_chunks', ctypes.c_uint32, 8), ('flags', ctypes.c_uint32, 12), ('chunks', ctypes.c_uint64, 16)])
@c.record
class struct_drm_amdgpu_cs_out(c.Struct):
  SIZE = 8
  handle: int
struct_drm_amdgpu_cs_out.register_fields([('handle', ctypes.c_uint64, 0)])
@c.record
class union_drm_amdgpu_cs(c.Struct):
  SIZE = 24
  _in: struct_drm_amdgpu_cs_in
  out: struct_drm_amdgpu_cs_out
union_drm_amdgpu_cs.register_fields([('_in', struct_drm_amdgpu_cs_in, 0), ('out', struct_drm_amdgpu_cs_out, 0)])
@c.record
class struct_drm_amdgpu_cs_chunk_ib(c.Struct):
  SIZE = 32
  _pad: int
  flags: int
  va_start: int
  ib_bytes: int
  ip_type: int
  ip_instance: int
  ring: int
struct_drm_amdgpu_cs_chunk_ib.register_fields([('_pad', ctypes.c_uint32, 0), ('flags', ctypes.c_uint32, 4), ('va_start', ctypes.c_uint64, 8), ('ib_bytes', ctypes.c_uint32, 16), ('ip_type', ctypes.c_uint32, 20), ('ip_instance', ctypes.c_uint32, 24), ('ring', ctypes.c_uint32, 28)])
@c.record
class struct_drm_amdgpu_cs_chunk_dep(c.Struct):
  SIZE = 24
  ip_type: int
  ip_instance: int
  ring: int
  ctx_id: int
  handle: int
struct_drm_amdgpu_cs_chunk_dep.register_fields([('ip_type', ctypes.c_uint32, 0), ('ip_instance', ctypes.c_uint32, 4), ('ring', ctypes.c_uint32, 8), ('ctx_id', ctypes.c_uint32, 12), ('handle', ctypes.c_uint64, 16)])
@c.record
class struct_drm_amdgpu_cs_chunk_fence(c.Struct):
  SIZE = 8
  handle: int
  offset: int
struct_drm_amdgpu_cs_chunk_fence.register_fields([('handle', ctypes.c_uint32, 0), ('offset', ctypes.c_uint32, 4)])
@c.record
class struct_drm_amdgpu_cs_chunk_sem(c.Struct):
  SIZE = 4
  handle: int
struct_drm_amdgpu_cs_chunk_sem.register_fields([('handle', ctypes.c_uint32, 0)])
@c.record
class struct_drm_amdgpu_cs_chunk_syncobj(c.Struct):
  SIZE = 16
  handle: int
  flags: int
  point: int
struct_drm_amdgpu_cs_chunk_syncobj.register_fields([('handle', ctypes.c_uint32, 0), ('flags', ctypes.c_uint32, 4), ('point', ctypes.c_uint64, 8)])
@c.record
class union_drm_amdgpu_fence_to_handle(c.Struct):
  SIZE = 32
  _in: union_drm_amdgpu_fence_to_handle_in
  out: union_drm_amdgpu_fence_to_handle_out
@c.record
class union_drm_amdgpu_fence_to_handle_in(c.Struct):
  SIZE = 32
  fence: struct_drm_amdgpu_fence
  what: int
  pad: int
union_drm_amdgpu_fence_to_handle_in.register_fields([('fence', struct_drm_amdgpu_fence, 0), ('what', ctypes.c_uint32, 24), ('pad', ctypes.c_uint32, 28)])
@c.record
class union_drm_amdgpu_fence_to_handle_out(c.Struct):
  SIZE = 4
  handle: int
union_drm_amdgpu_fence_to_handle_out.register_fields([('handle', ctypes.c_uint32, 0)])
union_drm_amdgpu_fence_to_handle.register_fields([('_in', union_drm_amdgpu_fence_to_handle_in, 0), ('out', union_drm_amdgpu_fence_to_handle_out, 0)])
@c.record
class struct_drm_amdgpu_cs_chunk_data(c.Struct):
  SIZE = 32
  ib_data: struct_drm_amdgpu_cs_chunk_ib
  fence_data: struct_drm_amdgpu_cs_chunk_fence
struct_drm_amdgpu_cs_chunk_data.register_fields([('ib_data', struct_drm_amdgpu_cs_chunk_ib, 0), ('fence_data', struct_drm_amdgpu_cs_chunk_fence, 0)])
@c.record
class struct_drm_amdgpu_cs_chunk_cp_gfx_shadow(c.Struct):
  SIZE = 32
  shadow_va: int
  csa_va: int
  gds_va: int
  flags: int
struct_drm_amdgpu_cs_chunk_cp_gfx_shadow.register_fields([('shadow_va', ctypes.c_uint64, 0), ('csa_va', ctypes.c_uint64, 8), ('gds_va', ctypes.c_uint64, 16), ('flags', ctypes.c_uint64, 24)])
@c.record
class struct_drm_amdgpu_query_fw(c.Struct):
  SIZE = 16
  fw_type: int
  ip_instance: int
  index: int
  _pad: int
struct_drm_amdgpu_query_fw.register_fields([('fw_type', ctypes.c_uint32, 0), ('ip_instance', ctypes.c_uint32, 4), ('index', ctypes.c_uint32, 8), ('_pad', ctypes.c_uint32, 12)])
@c.record
class struct_drm_amdgpu_info(c.Struct):
  SIZE = 16
  return_pointer: int
  return_size: int
  query: int
struct_drm_amdgpu_info.register_fields([('return_pointer', ctypes.c_uint64, 0), ('return_size', ctypes.c_uint32, 8), ('query', ctypes.c_uint32, 12)])
@c.record
class struct_drm_amdgpu_info_gds(c.Struct):
  SIZE = 32
  gds_gfx_partition_size: int
  compute_partition_size: int
  gds_total_size: int
  gws_per_gfx_partition: int
  gws_per_compute_partition: int
  oa_per_gfx_partition: int
  oa_per_compute_partition: int
  _pad: int
struct_drm_amdgpu_info_gds.register_fields([('gds_gfx_partition_size', ctypes.c_uint32, 0), ('compute_partition_size', ctypes.c_uint32, 4), ('gds_total_size', ctypes.c_uint32, 8), ('gws_per_gfx_partition', ctypes.c_uint32, 12), ('gws_per_compute_partition', ctypes.c_uint32, 16), ('oa_per_gfx_partition', ctypes.c_uint32, 20), ('oa_per_compute_partition', ctypes.c_uint32, 24), ('_pad', ctypes.c_uint32, 28)])
@c.record
class struct_drm_amdgpu_info_vram_gtt(c.Struct):
  SIZE = 24
  vram_size: int
  vram_cpu_accessible_size: int
  gtt_size: int
struct_drm_amdgpu_info_vram_gtt.register_fields([('vram_size', ctypes.c_uint64, 0), ('vram_cpu_accessible_size', ctypes.c_uint64, 8), ('gtt_size', ctypes.c_uint64, 16)])
@c.record
class struct_drm_amdgpu_heap_info(c.Struct):
  SIZE = 32
  total_heap_size: int
  usable_heap_size: int
  heap_usage: int
  max_allocation: int
struct_drm_amdgpu_heap_info.register_fields([('total_heap_size', ctypes.c_uint64, 0), ('usable_heap_size', ctypes.c_uint64, 8), ('heap_usage', ctypes.c_uint64, 16), ('max_allocation', ctypes.c_uint64, 24)])
@c.record
class struct_drm_amdgpu_memory_info(c.Struct):
  SIZE = 96
  vram: struct_drm_amdgpu_heap_info
  cpu_accessible_vram: struct_drm_amdgpu_heap_info
  gtt: struct_drm_amdgpu_heap_info
struct_drm_amdgpu_memory_info.register_fields([('vram', struct_drm_amdgpu_heap_info, 0), ('cpu_accessible_vram', struct_drm_amdgpu_heap_info, 32), ('gtt', struct_drm_amdgpu_heap_info, 64)])
@c.record
class struct_drm_amdgpu_info_firmware(c.Struct):
  SIZE = 8
  ver: int
  feature: int
struct_drm_amdgpu_info_firmware.register_fields([('ver', ctypes.c_uint32, 0), ('feature', ctypes.c_uint32, 4)])
@c.record
class struct_drm_amdgpu_info_vbios(c.Struct):
  SIZE = 200
  name: c.Array[ctypes.c_ubyte, Literal[64]]
  vbios_pn: c.Array[ctypes.c_ubyte, Literal[64]]
  version: int
  pad: int
  vbios_ver_str: c.Array[ctypes.c_ubyte, Literal[32]]
  date: c.Array[ctypes.c_ubyte, Literal[32]]
__u8: TypeAlias = ctypes.c_ubyte
struct_drm_amdgpu_info_vbios.register_fields([('name', c.Array[ctypes.c_ubyte, Literal[64]], 0), ('vbios_pn', c.Array[ctypes.c_ubyte, Literal[64]], 64), ('version', ctypes.c_uint32, 128), ('pad', ctypes.c_uint32, 132), ('vbios_ver_str', c.Array[ctypes.c_ubyte, Literal[32]], 136), ('date', c.Array[ctypes.c_ubyte, Literal[32]], 168)])
@c.record
class struct_drm_amdgpu_info_device(c.Struct):
  SIZE = 448
  device_id: int
  chip_rev: int
  external_rev: int
  pci_rev: int
  family: int
  num_shader_engines: int
  num_shader_arrays_per_engine: int
  gpu_counter_freq: int
  max_engine_clock: int
  max_memory_clock: int
  cu_active_number: int
  cu_ao_mask: int
  cu_bitmap: c.Array[c.Array[ctypes.c_uint32, Literal[4]], Literal[4]]
  enabled_rb_pipes_mask: int
  num_rb_pipes: int
  num_hw_gfx_contexts: int
  pcie_gen: int
  ids_flags: int
  virtual_address_offset: int
  virtual_address_max: int
  virtual_address_alignment: int
  pte_fragment_size: int
  gart_page_size: int
  ce_ram_size: int
  vram_type: int
  vram_bit_width: int
  vce_harvest_config: int
  gc_double_offchip_lds_buf: int
  prim_buf_gpu_addr: int
  pos_buf_gpu_addr: int
  cntl_sb_buf_gpu_addr: int
  param_buf_gpu_addr: int
  prim_buf_size: int
  pos_buf_size: int
  cntl_sb_buf_size: int
  param_buf_size: int
  wave_front_size: int
  num_shader_visible_vgprs: int
  num_cu_per_sh: int
  num_tcc_blocks: int
  gs_vgt_table_depth: int
  gs_prim_buffer_depth: int
  max_gs_waves_per_vgt: int
  pcie_num_lanes: int
  cu_ao_bitmap: c.Array[c.Array[ctypes.c_uint32, Literal[4]], Literal[4]]
  high_va_offset: int
  high_va_max: int
  pa_sc_tile_steering_override: int
  tcc_disabled_mask: int
  min_engine_clock: int
  min_memory_clock: int
  tcp_cache_size: int
  num_sqc_per_wgp: int
  sqc_data_cache_size: int
  sqc_inst_cache_size: int
  gl1c_cache_size: int
  gl2c_cache_size: int
  mall_size: int
  enabled_rb_pipes_mask_hi: int
  shadow_size: int
  shadow_alignment: int
  csa_size: int
  csa_alignment: int
  userq_ip_mask: int
  pad: int
struct_drm_amdgpu_info_device.register_fields([('device_id', ctypes.c_uint32, 0), ('chip_rev', ctypes.c_uint32, 4), ('external_rev', ctypes.c_uint32, 8), ('pci_rev', ctypes.c_uint32, 12), ('family', ctypes.c_uint32, 16), ('num_shader_engines', ctypes.c_uint32, 20), ('num_shader_arrays_per_engine', ctypes.c_uint32, 24), ('gpu_counter_freq', ctypes.c_uint32, 28), ('max_engine_clock', ctypes.c_uint64, 32), ('max_memory_clock', ctypes.c_uint64, 40), ('cu_active_number', ctypes.c_uint32, 48), ('cu_ao_mask', ctypes.c_uint32, 52), ('cu_bitmap', c.Array[c.Array[ctypes.c_uint32, Literal[4]], Literal[4]], 56), ('enabled_rb_pipes_mask', ctypes.c_uint32, 120), ('num_rb_pipes', ctypes.c_uint32, 124), ('num_hw_gfx_contexts', ctypes.c_uint32, 128), ('pcie_gen', ctypes.c_uint32, 132), ('ids_flags', ctypes.c_uint64, 136), ('virtual_address_offset', ctypes.c_uint64, 144), ('virtual_address_max', ctypes.c_uint64, 152), ('virtual_address_alignment', ctypes.c_uint32, 160), ('pte_fragment_size', ctypes.c_uint32, 164), ('gart_page_size', ctypes.c_uint32, 168), ('ce_ram_size', ctypes.c_uint32, 172), ('vram_type', ctypes.c_uint32, 176), ('vram_bit_width', ctypes.c_uint32, 180), ('vce_harvest_config', ctypes.c_uint32, 184), ('gc_double_offchip_lds_buf', ctypes.c_uint32, 188), ('prim_buf_gpu_addr', ctypes.c_uint64, 192), ('pos_buf_gpu_addr', ctypes.c_uint64, 200), ('cntl_sb_buf_gpu_addr', ctypes.c_uint64, 208), ('param_buf_gpu_addr', ctypes.c_uint64, 216), ('prim_buf_size', ctypes.c_uint32, 224), ('pos_buf_size', ctypes.c_uint32, 228), ('cntl_sb_buf_size', ctypes.c_uint32, 232), ('param_buf_size', ctypes.c_uint32, 236), ('wave_front_size', ctypes.c_uint32, 240), ('num_shader_visible_vgprs', ctypes.c_uint32, 244), ('num_cu_per_sh', ctypes.c_uint32, 248), ('num_tcc_blocks', ctypes.c_uint32, 252), ('gs_vgt_table_depth', ctypes.c_uint32, 256), ('gs_prim_buffer_depth', ctypes.c_uint32, 260), ('max_gs_waves_per_vgt', ctypes.c_uint32, 264), ('pcie_num_lanes', ctypes.c_uint32, 268), ('cu_ao_bitmap', c.Array[c.Array[ctypes.c_uint32, Literal[4]], Literal[4]], 272), ('high_va_offset', ctypes.c_uint64, 336), ('high_va_max', ctypes.c_uint64, 344), ('pa_sc_tile_steering_override', ctypes.c_uint32, 352), ('tcc_disabled_mask', ctypes.c_uint64, 360), ('min_engine_clock', ctypes.c_uint64, 368), ('min_memory_clock', ctypes.c_uint64, 376), ('tcp_cache_size', ctypes.c_uint32, 384), ('num_sqc_per_wgp', ctypes.c_uint32, 388), ('sqc_data_cache_size', ctypes.c_uint32, 392), ('sqc_inst_cache_size', ctypes.c_uint32, 396), ('gl1c_cache_size', ctypes.c_uint32, 400), ('gl2c_cache_size', ctypes.c_uint32, 404), ('mall_size', ctypes.c_uint64, 408), ('enabled_rb_pipes_mask_hi', ctypes.c_uint32, 416), ('shadow_size', ctypes.c_uint32, 420), ('shadow_alignment', ctypes.c_uint32, 424), ('csa_size', ctypes.c_uint32, 428), ('csa_alignment', ctypes.c_uint32, 432), ('userq_ip_mask', ctypes.c_uint32, 436), ('pad', ctypes.c_uint32, 440)])
@c.record
class struct_drm_amdgpu_info_hw_ip(c.Struct):
  SIZE = 32
  hw_ip_version_major: int
  hw_ip_version_minor: int
  capabilities_flags: int
  ib_start_alignment: int
  ib_size_alignment: int
  available_rings: int
  ip_discovery_version: int
struct_drm_amdgpu_info_hw_ip.register_fields([('hw_ip_version_major', ctypes.c_uint32, 0), ('hw_ip_version_minor', ctypes.c_uint32, 4), ('capabilities_flags', ctypes.c_uint64, 8), ('ib_start_alignment', ctypes.c_uint32, 16), ('ib_size_alignment', ctypes.c_uint32, 20), ('available_rings', ctypes.c_uint32, 24), ('ip_discovery_version', ctypes.c_uint32, 28)])
@c.record
class struct_drm_amdgpu_info_uq_fw_areas_gfx(c.Struct):
  SIZE = 16
  shadow_size: int
  shadow_alignment: int
  csa_size: int
  csa_alignment: int
struct_drm_amdgpu_info_uq_fw_areas_gfx.register_fields([('shadow_size', ctypes.c_uint32, 0), ('shadow_alignment', ctypes.c_uint32, 4), ('csa_size', ctypes.c_uint32, 8), ('csa_alignment', ctypes.c_uint32, 12)])
@c.record
class struct_drm_amdgpu_info_uq_fw_areas(c.Struct):
  SIZE = 16
  gfx: struct_drm_amdgpu_info_uq_fw_areas_gfx
struct_drm_amdgpu_info_uq_fw_areas.register_fields([('gfx', struct_drm_amdgpu_info_uq_fw_areas_gfx, 0)])
@c.record
class struct_drm_amdgpu_info_num_handles(c.Struct):
  SIZE = 8
  uvd_max_handles: int
  uvd_used_handles: int
struct_drm_amdgpu_info_num_handles.register_fields([('uvd_max_handles', ctypes.c_uint32, 0), ('uvd_used_handles', ctypes.c_uint32, 4)])
@c.record
class struct_drm_amdgpu_info_vce_clock_table_entry(c.Struct):
  SIZE = 16
  sclk: int
  mclk: int
  eclk: int
  pad: int
struct_drm_amdgpu_info_vce_clock_table_entry.register_fields([('sclk', ctypes.c_uint32, 0), ('mclk', ctypes.c_uint32, 4), ('eclk', ctypes.c_uint32, 8), ('pad', ctypes.c_uint32, 12)])
@c.record
class struct_drm_amdgpu_info_vce_clock_table(c.Struct):
  SIZE = 104
  entries: c.Array[struct_drm_amdgpu_info_vce_clock_table_entry, Literal[6]]
  num_valid_entries: int
  pad: int
struct_drm_amdgpu_info_vce_clock_table.register_fields([('entries', c.Array[struct_drm_amdgpu_info_vce_clock_table_entry, Literal[6]], 0), ('num_valid_entries', ctypes.c_uint32, 96), ('pad', ctypes.c_uint32, 100)])
@c.record
class struct_drm_amdgpu_info_video_codec_info(c.Struct):
  SIZE = 24
  valid: int
  max_width: int
  max_height: int
  max_pixels_per_frame: int
  max_level: int
  pad: int
struct_drm_amdgpu_info_video_codec_info.register_fields([('valid', ctypes.c_uint32, 0), ('max_width', ctypes.c_uint32, 4), ('max_height', ctypes.c_uint32, 8), ('max_pixels_per_frame', ctypes.c_uint32, 12), ('max_level', ctypes.c_uint32, 16), ('pad', ctypes.c_uint32, 20)])
@c.record
class struct_drm_amdgpu_info_video_caps(c.Struct):
  SIZE = 192
  codec_info: c.Array[struct_drm_amdgpu_info_video_codec_info, Literal[8]]
struct_drm_amdgpu_info_video_caps.register_fields([('codec_info', c.Array[struct_drm_amdgpu_info_video_codec_info, Literal[8]], 0)])
@c.record
class struct_drm_amdgpu_info_gpuvm_fault(c.Struct):
  SIZE = 16
  addr: int
  status: int
  vmhub: int
struct_drm_amdgpu_info_gpuvm_fault.register_fields([('addr', ctypes.c_uint64, 0), ('status', ctypes.c_uint32, 8), ('vmhub', ctypes.c_uint32, 12)])
@c.record
class struct_drm_amdgpu_info_uq_metadata_gfx(c.Struct):
  SIZE = 16
  shadow_size: int
  shadow_alignment: int
  csa_size: int
  csa_alignment: int
struct_drm_amdgpu_info_uq_metadata_gfx.register_fields([('shadow_size', ctypes.c_uint32, 0), ('shadow_alignment', ctypes.c_uint32, 4), ('csa_size', ctypes.c_uint32, 8), ('csa_alignment', ctypes.c_uint32, 12)])
@c.record
class struct_drm_amdgpu_info_uq_metadata(c.Struct):
  SIZE = 16
  gfx: struct_drm_amdgpu_info_uq_metadata_gfx
struct_drm_amdgpu_info_uq_metadata.register_fields([('gfx', struct_drm_amdgpu_info_uq_metadata_gfx, 0)])
class _anonstruct0(c.Struct): pass
class struct_drm_amdgpu_virtual_range(c.Struct): pass
@c.record
class struct_drm_amdgpu_capability(c.Struct):
  SIZE = 8
  flag: int
  direct_gma_size: int
struct_drm_amdgpu_capability.register_fields([('flag', ctypes.c_uint32, 0), ('direct_gma_size', ctypes.c_uint32, 4)])
@c.record
class struct_drm_amdgpu_freesync(c.Struct):
  SIZE = 32
  op: int
  spare: c.Array[ctypes.c_uint32, Literal[7]]
struct_drm_amdgpu_freesync.register_fields([('op', ctypes.c_uint32, 0), ('spare', c.Array[ctypes.c_uint32, Literal[7]], 4)])
DRM_NAME = "drm"
DRM_MIN_ORDER = 5
DRM_MAX_ORDER = 22
DRM_RAM_PERCENT = 10
_DRM_LOCK_HELD = 0x80000000
_DRM_LOCK_CONT = 0x40000000
_DRM_LOCK_IS_HELD = lambda lock: ((lock) & _DRM_LOCK_HELD) # type: ignore
_DRM_LOCK_IS_CONT = lambda lock: ((lock) & _DRM_LOCK_CONT) # type: ignore
_DRM_LOCKING_CONTEXT = lambda lock: ((lock) & ~(_DRM_LOCK_HELD|_DRM_LOCK_CONT)) # type: ignore
_DRM_VBLANK_HIGH_CRTC_SHIFT = 1
_DRM_VBLANK_TYPES_MASK = (_DRM_VBLANK_ABSOLUTE | _DRM_VBLANK_RELATIVE)
_DRM_VBLANK_FLAGS_MASK = (_DRM_VBLANK_EVENT | _DRM_VBLANK_SIGNAL | _DRM_VBLANK_SECONDARY | _DRM_VBLANK_NEXTONMISS)
_DRM_PRE_MODESET = 1
_DRM_POST_MODESET = 2
DRM_CAP_DUMB_BUFFER = 0x1
DRM_CAP_VBLANK_HIGH_CRTC = 0x2
DRM_CAP_DUMB_PREFERRED_DEPTH = 0x3
DRM_CAP_DUMB_PREFER_SHADOW = 0x4
DRM_CAP_PRIME = 0x5
DRM_PRIME_CAP_IMPORT = 0x1
DRM_PRIME_CAP_EXPORT = 0x2
DRM_CAP_TIMESTAMP_MONOTONIC = 0x6
DRM_CAP_ASYNC_PAGE_FLIP = 0x7
DRM_CAP_CURSOR_WIDTH = 0x8
DRM_CAP_CURSOR_HEIGHT = 0x9
DRM_CAP_ADDFB2_MODIFIERS = 0x10
DRM_CAP_PAGE_FLIP_TARGET = 0x11
DRM_CAP_CRTC_IN_VBLANK_EVENT = 0x12
DRM_CAP_SYNCOBJ = 0x13
DRM_CAP_SYNCOBJ_TIMELINE = 0x14
DRM_CAP_ATOMIC_ASYNC_PAGE_FLIP = 0x15
DRM_CLIENT_CAP_STEREO_3D = 1
DRM_CLIENT_CAP_UNIVERSAL_PLANES = 2
DRM_CLIENT_CAP_ATOMIC = 3
DRM_CLIENT_CAP_ASPECT_RATIO = 4
DRM_CLIENT_CAP_WRITEBACK_CONNECTORS = 5
DRM_CLIENT_CAP_CURSOR_PLANE_HOTSPOT = 6
DRM_SYNCOBJ_CREATE_SIGNALED = (1 << 0)
DRM_SYNCOBJ_FD_TO_HANDLE_FLAGS_IMPORT_SYNC_FILE = (1 << 0)
DRM_SYNCOBJ_HANDLE_TO_FD_FLAGS_EXPORT_SYNC_FILE = (1 << 0)
DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL = (1 << 0)
DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT = (1 << 1)
DRM_SYNCOBJ_WAIT_FLAGS_WAIT_AVAILABLE = (1 << 2)
DRM_SYNCOBJ_WAIT_FLAGS_WAIT_DEADLINE = (1 << 3)
DRM_SYNCOBJ_QUERY_FLAGS_LAST_SUBMITTED = (1 << 0)
DRM_CRTC_SEQUENCE_RELATIVE = 0x00000001
DRM_CRTC_SEQUENCE_NEXT_ON_MISS = 0x00000002
DRM_IOCTL_BASE = 'd'
DRM_IO = lambda nr: _IO(DRM_IOCTL_BASE,nr) # type: ignore
DRM_IOR = lambda nr,type: _IOR(DRM_IOCTL_BASE,nr,type) # type: ignore
DRM_IOW = lambda nr,type: _IOW(DRM_IOCTL_BASE,nr,type) # type: ignore
DRM_IOWR = lambda nr,type: _IOWR(DRM_IOCTL_BASE,nr,type) # type: ignore
DRM_IOCTL_VERSION = DRM_IOWR(0x00, struct_drm_version)
DRM_IOCTL_GET_UNIQUE = DRM_IOWR(0x01, struct_drm_unique)
DRM_IOCTL_GET_MAGIC = DRM_IOR( 0x02, struct_drm_auth)
DRM_IOCTL_IRQ_BUSID = DRM_IOWR(0x03, struct_drm_irq_busid)
DRM_IOCTL_GET_MAP = DRM_IOWR(0x04, struct_drm_map)
DRM_IOCTL_GET_CLIENT = DRM_IOWR(0x05, struct_drm_client)
DRM_IOCTL_GET_STATS = DRM_IOR( 0x06, struct_drm_stats)
DRM_IOCTL_SET_VERSION = DRM_IOWR(0x07, struct_drm_set_version)
DRM_IOCTL_MODESET_CTL = DRM_IOW(0x08, struct_drm_modeset_ctl)
DRM_IOCTL_GEM_CLOSE = DRM_IOW (0x09, struct_drm_gem_close)
DRM_IOCTL_GEM_FLINK = DRM_IOWR(0x0a, struct_drm_gem_flink)
DRM_IOCTL_GEM_OPEN = DRM_IOWR(0x0b, struct_drm_gem_open)
DRM_IOCTL_GET_CAP = DRM_IOWR(0x0c, struct_drm_get_cap)
DRM_IOCTL_SET_CLIENT_CAP = DRM_IOW( 0x0d, struct_drm_set_client_cap)
DRM_IOCTL_SET_UNIQUE = DRM_IOW( 0x10, struct_drm_unique)
DRM_IOCTL_AUTH_MAGIC = DRM_IOW( 0x11, struct_drm_auth)
DRM_IOCTL_BLOCK = DRM_IOWR(0x12, struct_drm_block)
DRM_IOCTL_UNBLOCK = DRM_IOWR(0x13, struct_drm_block)
DRM_IOCTL_CONTROL = DRM_IOW( 0x14, struct_drm_control)
DRM_IOCTL_ADD_MAP = DRM_IOWR(0x15, struct_drm_map)
DRM_IOCTL_ADD_BUFS = DRM_IOWR(0x16, struct_drm_buf_desc)
DRM_IOCTL_MARK_BUFS = DRM_IOW( 0x17, struct_drm_buf_desc)
DRM_IOCTL_INFO_BUFS = DRM_IOWR(0x18, struct_drm_buf_info)
DRM_IOCTL_MAP_BUFS = DRM_IOWR(0x19, struct_drm_buf_map)
DRM_IOCTL_FREE_BUFS = DRM_IOW( 0x1a, struct_drm_buf_free)
DRM_IOCTL_RM_MAP = DRM_IOW( 0x1b, struct_drm_map)
DRM_IOCTL_SET_SAREA_CTX = DRM_IOW( 0x1c, struct_drm_ctx_priv_map)
DRM_IOCTL_GET_SAREA_CTX = DRM_IOWR(0x1d, struct_drm_ctx_priv_map)
DRM_IOCTL_SET_MASTER = DRM_IO(0x1e)
DRM_IOCTL_DROP_MASTER = DRM_IO(0x1f)
DRM_IOCTL_ADD_CTX = DRM_IOWR(0x20, struct_drm_ctx)
DRM_IOCTL_RM_CTX = DRM_IOWR(0x21, struct_drm_ctx)
DRM_IOCTL_MOD_CTX = DRM_IOW( 0x22, struct_drm_ctx)
DRM_IOCTL_GET_CTX = DRM_IOWR(0x23, struct_drm_ctx)
DRM_IOCTL_SWITCH_CTX = DRM_IOW( 0x24, struct_drm_ctx)
DRM_IOCTL_NEW_CTX = DRM_IOW( 0x25, struct_drm_ctx)
DRM_IOCTL_RES_CTX = DRM_IOWR(0x26, struct_drm_ctx_res)
DRM_IOCTL_ADD_DRAW = DRM_IOWR(0x27, struct_drm_draw)
DRM_IOCTL_RM_DRAW = DRM_IOWR(0x28, struct_drm_draw)
DRM_IOCTL_DMA = DRM_IOWR(0x29, struct_drm_dma)
DRM_IOCTL_LOCK = DRM_IOW( 0x2a, struct_drm_lock)
DRM_IOCTL_UNLOCK = DRM_IOW( 0x2b, struct_drm_lock)
DRM_IOCTL_FINISH = DRM_IOW( 0x2c, struct_drm_lock)
DRM_IOCTL_PRIME_HANDLE_TO_FD = DRM_IOWR(0x2d, struct_drm_prime_handle)
DRM_IOCTL_PRIME_FD_TO_HANDLE = DRM_IOWR(0x2e, struct_drm_prime_handle)
DRM_IOCTL_AGP_ACQUIRE = DRM_IO(  0x30)
DRM_IOCTL_AGP_RELEASE = DRM_IO(  0x31)
DRM_IOCTL_AGP_ENABLE = DRM_IOW( 0x32, struct_drm_agp_mode)
DRM_IOCTL_AGP_INFO = DRM_IOR( 0x33, struct_drm_agp_info)
DRM_IOCTL_AGP_ALLOC = DRM_IOWR(0x34, struct_drm_agp_buffer)
DRM_IOCTL_AGP_FREE = DRM_IOW( 0x35, struct_drm_agp_buffer)
DRM_IOCTL_AGP_BIND = DRM_IOW( 0x36, struct_drm_agp_binding)
DRM_IOCTL_AGP_UNBIND = DRM_IOW( 0x37, struct_drm_agp_binding)
DRM_IOCTL_SG_ALLOC = DRM_IOWR(0x38, struct_drm_scatter_gather)
DRM_IOCTL_SG_FREE = DRM_IOW( 0x39, struct_drm_scatter_gather)
DRM_IOCTL_WAIT_VBLANK = DRM_IOWR(0x3a, union_drm_wait_vblank)
DRM_IOCTL_CRTC_GET_SEQUENCE = DRM_IOWR(0x3b, struct_drm_crtc_get_sequence)
DRM_IOCTL_CRTC_QUEUE_SEQUENCE = DRM_IOWR(0x3c, struct_drm_crtc_queue_sequence)
DRM_IOCTL_UPDATE_DRAW = DRM_IOW(0x3f, struct_drm_update_draw)
DRM_IOCTL_SYNCOBJ_CREATE = DRM_IOWR(0xBF, struct_drm_syncobj_create)
DRM_IOCTL_SYNCOBJ_DESTROY = DRM_IOWR(0xC0, struct_drm_syncobj_destroy)
DRM_IOCTL_SYNCOBJ_HANDLE_TO_FD = DRM_IOWR(0xC1, struct_drm_syncobj_handle)
DRM_IOCTL_SYNCOBJ_FD_TO_HANDLE = DRM_IOWR(0xC2, struct_drm_syncobj_handle)
DRM_IOCTL_SYNCOBJ_WAIT = DRM_IOWR(0xC3, struct_drm_syncobj_wait)
DRM_IOCTL_SYNCOBJ_RESET = DRM_IOWR(0xC4, struct_drm_syncobj_array)
DRM_IOCTL_SYNCOBJ_SIGNAL = DRM_IOWR(0xC5, struct_drm_syncobj_array)
DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT = DRM_IOWR(0xCA, struct_drm_syncobj_timeline_wait)
DRM_IOCTL_SYNCOBJ_QUERY = DRM_IOWR(0xCB, struct_drm_syncobj_timeline_array)
DRM_IOCTL_SYNCOBJ_TRANSFER = DRM_IOWR(0xCC, struct_drm_syncobj_transfer)
DRM_IOCTL_SYNCOBJ_TIMELINE_SIGNAL = DRM_IOWR(0xCD, struct_drm_syncobj_timeline_array)
DRM_IOCTL_SYNCOBJ_EVENTFD = DRM_IOWR(0xCF, struct_drm_syncobj_eventfd)
DRM_COMMAND_BASE = 0x40
DRM_COMMAND_END = 0xA0
DRM_EVENT_VBLANK = 0x01
DRM_EVENT_FLIP_COMPLETE = 0x02
DRM_EVENT_CRTC_SEQUENCE = 0x03
DRM_AMDGPU_GEM_CREATE = 0x00
DRM_AMDGPU_GEM_MMAP = 0x01
DRM_AMDGPU_CTX = 0x02
DRM_AMDGPU_BO_LIST = 0x03
DRM_AMDGPU_CS = 0x04
DRM_AMDGPU_INFO = 0x05
DRM_AMDGPU_GEM_METADATA = 0x06
DRM_AMDGPU_GEM_WAIT_IDLE = 0x07
DRM_AMDGPU_GEM_VA = 0x08
DRM_AMDGPU_WAIT_CS = 0x09
DRM_AMDGPU_GEM_OP = 0x10
DRM_AMDGPU_GEM_USERPTR = 0x11
DRM_AMDGPU_WAIT_FENCES = 0x12
DRM_AMDGPU_VM = 0x13
DRM_AMDGPU_FENCE_TO_HANDLE = 0x14
DRM_AMDGPU_SCHED = 0x15
DRM_AMDGPU_USERQ = 0x16
DRM_AMDGPU_USERQ_SIGNAL = 0x17
DRM_AMDGPU_USERQ_WAIT = 0x18
DRM_AMDGPU_GEM_DGMA = 0x5c
DRM_AMDGPU_SEM = 0x5b
DRM_IOCTL_AMDGPU_GEM_CREATE = DRM_IOWR(DRM_COMMAND_BASE + DRM_AMDGPU_GEM_CREATE, union_drm_amdgpu_gem_create)
DRM_IOCTL_AMDGPU_GEM_MMAP = DRM_IOWR(DRM_COMMAND_BASE + DRM_AMDGPU_GEM_MMAP, union_drm_amdgpu_gem_mmap)
DRM_IOCTL_AMDGPU_CTX = DRM_IOWR(DRM_COMMAND_BASE + DRM_AMDGPU_CTX, union_drm_amdgpu_ctx)
DRM_IOCTL_AMDGPU_BO_LIST = DRM_IOWR(DRM_COMMAND_BASE + DRM_AMDGPU_BO_LIST, union_drm_amdgpu_bo_list)
DRM_IOCTL_AMDGPU_CS = DRM_IOWR(DRM_COMMAND_BASE + DRM_AMDGPU_CS, union_drm_amdgpu_cs)
DRM_IOCTL_AMDGPU_INFO = DRM_IOW(DRM_COMMAND_BASE + DRM_AMDGPU_INFO, struct_drm_amdgpu_info)
DRM_IOCTL_AMDGPU_GEM_METADATA = DRM_IOWR(DRM_COMMAND_BASE + DRM_AMDGPU_GEM_METADATA, struct_drm_amdgpu_gem_metadata)
DRM_IOCTL_AMDGPU_GEM_WAIT_IDLE = DRM_IOWR(DRM_COMMAND_BASE + DRM_AMDGPU_GEM_WAIT_IDLE, union_drm_amdgpu_gem_wait_idle)
DRM_IOCTL_AMDGPU_GEM_VA = DRM_IOW(DRM_COMMAND_BASE + DRM_AMDGPU_GEM_VA, struct_drm_amdgpu_gem_va)
DRM_IOCTL_AMDGPU_WAIT_CS = DRM_IOWR(DRM_COMMAND_BASE + DRM_AMDGPU_WAIT_CS, union_drm_amdgpu_wait_cs)
DRM_IOCTL_AMDGPU_GEM_OP = DRM_IOWR(DRM_COMMAND_BASE + DRM_AMDGPU_GEM_OP, struct_drm_amdgpu_gem_op)
DRM_IOCTL_AMDGPU_GEM_USERPTR = DRM_IOWR(DRM_COMMAND_BASE + DRM_AMDGPU_GEM_USERPTR, struct_drm_amdgpu_gem_userptr)
DRM_IOCTL_AMDGPU_WAIT_FENCES = DRM_IOWR(DRM_COMMAND_BASE + DRM_AMDGPU_WAIT_FENCES, union_drm_amdgpu_wait_fences)
DRM_IOCTL_AMDGPU_VM = DRM_IOWR(DRM_COMMAND_BASE + DRM_AMDGPU_VM, union_drm_amdgpu_vm)
DRM_IOCTL_AMDGPU_FENCE_TO_HANDLE = DRM_IOWR(DRM_COMMAND_BASE + DRM_AMDGPU_FENCE_TO_HANDLE, union_drm_amdgpu_fence_to_handle)
DRM_IOCTL_AMDGPU_SCHED = DRM_IOW(DRM_COMMAND_BASE + DRM_AMDGPU_SCHED, union_drm_amdgpu_sched)
DRM_IOCTL_AMDGPU_USERQ = DRM_IOWR(DRM_COMMAND_BASE + DRM_AMDGPU_USERQ, union_drm_amdgpu_userq)
DRM_IOCTL_AMDGPU_USERQ_SIGNAL = DRM_IOWR(DRM_COMMAND_BASE + DRM_AMDGPU_USERQ_SIGNAL, struct_drm_amdgpu_userq_signal)
DRM_IOCTL_AMDGPU_USERQ_WAIT = DRM_IOWR(DRM_COMMAND_BASE + DRM_AMDGPU_USERQ_WAIT, struct_drm_amdgpu_userq_wait)
DRM_IOCTL_AMDGPU_GEM_DGMA = DRM_IOWR(DRM_COMMAND_BASE + DRM_AMDGPU_GEM_DGMA, struct_drm_amdgpu_gem_dgma)
DRM_IOCTL_AMDGPU_SEM = DRM_IOWR(DRM_COMMAND_BASE + DRM_AMDGPU_SEM, union_drm_amdgpu_sem)
AMDGPU_GEM_DOMAIN_CPU = 0x1
AMDGPU_GEM_DOMAIN_GTT = 0x2
AMDGPU_GEM_DOMAIN_VRAM = 0x4
AMDGPU_GEM_DOMAIN_GDS = 0x8
AMDGPU_GEM_DOMAIN_GWS = 0x10
AMDGPU_GEM_DOMAIN_OA = 0x20
AMDGPU_GEM_DOMAIN_DOORBELL = 0x40
AMDGPU_GEM_DOMAIN_DGMA = 0x400
AMDGPU_GEM_DOMAIN_DGMA_IMPORT = 0x800
AMDGPU_GEM_DOMAIN_MASK = (AMDGPU_GEM_DOMAIN_CPU | AMDGPU_GEM_DOMAIN_GTT | AMDGPU_GEM_DOMAIN_VRAM | AMDGPU_GEM_DOMAIN_GDS | AMDGPU_GEM_DOMAIN_GWS | AMDGPU_GEM_DOMAIN_OA | AMDGPU_GEM_DOMAIN_DOORBELL | AMDGPU_GEM_DOMAIN_DGMA | AMDGPU_GEM_DOMAIN_DGMA_IMPORT)
AMDGPU_GEM_CREATE_CPU_ACCESS_REQUIRED = (1 << 0)
AMDGPU_GEM_CREATE_NO_CPU_ACCESS = (1 << 1)
AMDGPU_GEM_CREATE_CPU_GTT_USWC = (1 << 2)
AMDGPU_GEM_CREATE_VRAM_CLEARED = (1 << 3)
AMDGPU_GEM_CREATE_VRAM_CONTIGUOUS = (1 << 5)
AMDGPU_GEM_CREATE_VM_ALWAYS_VALID = (1 << 6)
AMDGPU_GEM_CREATE_EXPLICIT_SYNC = (1 << 7)
AMDGPU_GEM_CREATE_CP_MQD_GFX9 = (1 << 8)
AMDGPU_GEM_CREATE_VRAM_WIPE_ON_RELEASE = (1 << 9)
AMDGPU_GEM_CREATE_ENCRYPTED = (1 << 10)
AMDGPU_GEM_CREATE_PREEMPTIBLE = (1 << 11)
AMDGPU_GEM_CREATE_DISCARDABLE = (1 << 12)
AMDGPU_GEM_CREATE_COHERENT = (1 << 13)
AMDGPU_GEM_CREATE_UNCACHED = (1 << 14)
AMDGPU_GEM_CREATE_EXT_COHERENT = (1 << 15)
AMDGPU_GEM_CREATE_GFX12_DCC = (1 << 16)
AMDGPU_GEM_CREATE_SPARSE = (1 << 29)
AMDGPU_GEM_CREATE_TOP_DOWN = (1 << 30)
AMDGPU_GEM_CREATE_NO_EVICT = (1 << 31)
AMDGPU_BO_LIST_OP_CREATE = 0
AMDGPU_BO_LIST_OP_DESTROY = 1
AMDGPU_BO_LIST_OP_UPDATE = 2
AMDGPU_CTX_OP_ALLOC_CTX = 1
AMDGPU_CTX_OP_FREE_CTX = 2
AMDGPU_CTX_OP_QUERY_STATE = 3
AMDGPU_CTX_OP_QUERY_STATE2 = 4
AMDGPU_CTX_OP_GET_STABLE_PSTATE = 5
AMDGPU_CTX_OP_SET_STABLE_PSTATE = 6
AMDGPU_CTX_NO_RESET = 0
AMDGPU_CTX_GUILTY_RESET = 1
AMDGPU_CTX_INNOCENT_RESET = 2
AMDGPU_CTX_UNKNOWN_RESET = 3
AMDGPU_CTX_QUERY2_FLAGS_RESET = (1<<0)
AMDGPU_CTX_QUERY2_FLAGS_VRAMLOST = (1<<1)
AMDGPU_CTX_QUERY2_FLAGS_GUILTY = (1<<2)
AMDGPU_CTX_QUERY2_FLAGS_RAS_CE = (1<<3)
AMDGPU_CTX_QUERY2_FLAGS_RAS_UE = (1<<4)
AMDGPU_CTX_QUERY2_FLAGS_RESET_IN_PROGRESS = (1<<5)
AMDGPU_CTX_PRIORITY_UNSET = -2048
AMDGPU_CTX_PRIORITY_VERY_LOW = -1023
AMDGPU_CTX_PRIORITY_LOW = -512
AMDGPU_CTX_PRIORITY_NORMAL = 0
AMDGPU_CTX_PRIORITY_HIGH = 512
AMDGPU_CTX_PRIORITY_VERY_HIGH = 1023
AMDGPU_CTX_STABLE_PSTATE_FLAGS_MASK = 0xf
AMDGPU_CTX_STABLE_PSTATE_NONE = 0
AMDGPU_CTX_STABLE_PSTATE_STANDARD = 1
AMDGPU_CTX_STABLE_PSTATE_MIN_SCLK = 2
AMDGPU_CTX_STABLE_PSTATE_MIN_MCLK = 3
AMDGPU_CTX_STABLE_PSTATE_PEAK = 4
AMDGPU_USERQ_OP_CREATE = 1
AMDGPU_USERQ_OP_FREE = 2
AMDGPU_USERQ_CREATE_FLAGS_QUEUE_PRIORITY_MASK = 0x3
AMDGPU_USERQ_CREATE_FLAGS_QUEUE_PRIORITY_SHIFT = 0
AMDGPU_USERQ_CREATE_FLAGS_QUEUE_PRIORITY_NORMAL_LOW = 0
AMDGPU_USERQ_CREATE_FLAGS_QUEUE_PRIORITY_LOW = 1
AMDGPU_USERQ_CREATE_FLAGS_QUEUE_PRIORITY_NORMAL_HIGH = 2
AMDGPU_USERQ_CREATE_FLAGS_QUEUE_PRIORITY_HIGH = 3
AMDGPU_USERQ_CREATE_FLAGS_QUEUE_SECURE = (1 << 2)
AMDGPU_SEM_OP_CREATE_SEM = 1
AMDGPU_SEM_OP_WAIT_SEM = 2
AMDGPU_SEM_OP_SIGNAL_SEM = 3
AMDGPU_SEM_OP_DESTROY_SEM = 4
AMDGPU_SEM_OP_IMPORT_SEM = 5
AMDGPU_SEM_OP_EXPORT_SEM = 6
AMDGPU_VM_OP_RESERVE_VMID = 1
AMDGPU_VM_OP_UNRESERVE_VMID = 2
AMDGPU_SCHED_OP_PROCESS_PRIORITY_OVERRIDE = 1
AMDGPU_SCHED_OP_CONTEXT_PRIORITY_OVERRIDE = 2
AMDGPU_GEM_USERPTR_READONLY = (1 << 0)
AMDGPU_GEM_USERPTR_ANONONLY = (1 << 1)
AMDGPU_GEM_USERPTR_VALIDATE = (1 << 2)
AMDGPU_GEM_USERPTR_REGISTER = (1 << 3)
AMDGPU_GEM_DGMA_IMPORT = 0
AMDGPU_GEM_DGMA_QUERY_PHYS_ADDR = 1
AMDGPU_TILING_ARRAY_MODE_SHIFT = 0
AMDGPU_TILING_ARRAY_MODE_MASK = 0xf
AMDGPU_TILING_PIPE_CONFIG_SHIFT = 4
AMDGPU_TILING_PIPE_CONFIG_MASK = 0x1f
AMDGPU_TILING_TILE_SPLIT_SHIFT = 9
AMDGPU_TILING_TILE_SPLIT_MASK = 0x7
AMDGPU_TILING_MICRO_TILE_MODE_SHIFT = 12
AMDGPU_TILING_MICRO_TILE_MODE_MASK = 0x7
AMDGPU_TILING_BANK_WIDTH_SHIFT = 15
AMDGPU_TILING_BANK_WIDTH_MASK = 0x3
AMDGPU_TILING_BANK_HEIGHT_SHIFT = 17
AMDGPU_TILING_BANK_HEIGHT_MASK = 0x3
AMDGPU_TILING_MACRO_TILE_ASPECT_SHIFT = 19
AMDGPU_TILING_MACRO_TILE_ASPECT_MASK = 0x3
AMDGPU_TILING_NUM_BANKS_SHIFT = 21
AMDGPU_TILING_NUM_BANKS_MASK = 0x3
AMDGPU_TILING_SWIZZLE_MODE_SHIFT = 0
AMDGPU_TILING_SWIZZLE_MODE_MASK = 0x1f
AMDGPU_TILING_DCC_OFFSET_256B_SHIFT = 5
AMDGPU_TILING_DCC_OFFSET_256B_MASK = 0xFFFFFF
AMDGPU_TILING_DCC_PITCH_MAX_SHIFT = 29
AMDGPU_TILING_DCC_PITCH_MAX_MASK = 0x3FFF
AMDGPU_TILING_DCC_INDEPENDENT_64B_SHIFT = 43
AMDGPU_TILING_DCC_INDEPENDENT_64B_MASK = 0x1
AMDGPU_TILING_DCC_INDEPENDENT_128B_SHIFT = 44
AMDGPU_TILING_DCC_INDEPENDENT_128B_MASK = 0x1
AMDGPU_TILING_SCANOUT_SHIFT = 63
AMDGPU_TILING_SCANOUT_MASK = 0x1
AMDGPU_TILING_GFX12_SWIZZLE_MODE_SHIFT = 0
AMDGPU_TILING_GFX12_SWIZZLE_MODE_MASK = 0x7
AMDGPU_TILING_GFX12_DCC_MAX_COMPRESSED_BLOCK_SHIFT = 3
AMDGPU_TILING_GFX12_DCC_MAX_COMPRESSED_BLOCK_MASK = 0x3
AMDGPU_TILING_GFX12_DCC_NUMBER_TYPE_SHIFT = 5
AMDGPU_TILING_GFX12_DCC_NUMBER_TYPE_MASK = 0x7
AMDGPU_TILING_GFX12_DCC_DATA_FORMAT_SHIFT = 8
AMDGPU_TILING_GFX12_DCC_DATA_FORMAT_MASK = 0x3f
AMDGPU_TILING_GFX12_DCC_WRITE_COMPRESS_DISABLE_SHIFT = 14
AMDGPU_TILING_GFX12_DCC_WRITE_COMPRESS_DISABLE_MASK = 0x1
AMDGPU_TILING_GFX12_SCANOUT_SHIFT = 63
AMDGPU_TILING_GFX12_SCANOUT_MASK = 0x1
AMDGPU_GEM_METADATA_OP_SET_METADATA = 1
AMDGPU_GEM_METADATA_OP_GET_METADATA = 2
AMDGPU_GEM_OP_GET_GEM_CREATE_INFO = 0
AMDGPU_GEM_OP_SET_PLACEMENT = 1
AMDGPU_VA_OP_MAP = 1
AMDGPU_VA_OP_UNMAP = 2
AMDGPU_VA_OP_CLEAR = 3
AMDGPU_VA_OP_REPLACE = 4
AMDGPU_VM_DELAY_UPDATE = (1 << 0)
AMDGPU_VM_PAGE_READABLE = (1 << 1)
AMDGPU_VM_PAGE_WRITEABLE = (1 << 2)
AMDGPU_VM_PAGE_EXECUTABLE = (1 << 3)
AMDGPU_VM_PAGE_PRT = (1 << 4)
AMDGPU_VM_MTYPE_MASK = (0xf << 5)
AMDGPU_VM_MTYPE_DEFAULT = (0 << 5)
AMDGPU_VM_MTYPE_NC = (1 << 5)
AMDGPU_VM_MTYPE_WC = (2 << 5)
AMDGPU_VM_MTYPE_CC = (3 << 5)
AMDGPU_VM_MTYPE_UC = (4 << 5)
AMDGPU_VM_MTYPE_RW = (5 << 5)
AMDGPU_VM_PAGE_NOALLOC = (1 << 9)
AMDGPU_HW_IP_GFX = 0
AMDGPU_HW_IP_COMPUTE = 1
AMDGPU_HW_IP_DMA = 2
AMDGPU_HW_IP_UVD = 3
AMDGPU_HW_IP_VCE = 4
AMDGPU_HW_IP_UVD_ENC = 5
AMDGPU_HW_IP_VCN_DEC = 6
AMDGPU_HW_IP_VCN_ENC = 7
AMDGPU_HW_IP_VCN_JPEG = 8
AMDGPU_HW_IP_VPE = 9
AMDGPU_HW_IP_NUM = 10
AMDGPU_HW_IP_INSTANCE_MAX_COUNT = 1
AMDGPU_CHUNK_ID_IB = 0x01
AMDGPU_CHUNK_ID_FENCE = 0x02
AMDGPU_CHUNK_ID_DEPENDENCIES = 0x03
AMDGPU_CHUNK_ID_SYNCOBJ_IN = 0x04
AMDGPU_CHUNK_ID_SYNCOBJ_OUT = 0x05
AMDGPU_CHUNK_ID_BO_HANDLES = 0x06
AMDGPU_CHUNK_ID_SCHEDULED_DEPENDENCIES = 0x07
AMDGPU_CHUNK_ID_SYNCOBJ_TIMELINE_WAIT = 0x08
AMDGPU_CHUNK_ID_SYNCOBJ_TIMELINE_SIGNAL = 0x09
AMDGPU_CHUNK_ID_CP_GFX_SHADOW = 0x0a
AMDGPU_IB_FLAG_CE = (1<<0)
AMDGPU_IB_FLAG_PREAMBLE = (1<<1)
AMDGPU_IB_FLAG_PREEMPT = (1<<2)
AMDGPU_IB_FLAG_TC_WB_NOT_INVALIDATE = (1 << 3)
AMDGPU_IB_FLAG_RESET_GDS_MAX_WAVE_ID = (1 << 4)
AMDGPU_IB_FLAGS_SECURE = (1 << 5)
AMDGPU_IB_FLAG_EMIT_MEM_SYNC = (1 << 6)
AMDGPU_FENCE_TO_HANDLE_GET_SYNCOBJ = 0
AMDGPU_FENCE_TO_HANDLE_GET_SYNCOBJ_FD = 1
AMDGPU_FENCE_TO_HANDLE_GET_SYNC_FILE_FD = 2
AMDGPU_CS_CHUNK_CP_GFX_SHADOW_FLAGS_INIT_SHADOW = 0x1
AMDGPU_IDS_FLAGS_FUSION = 0x1
AMDGPU_IDS_FLAGS_PREEMPTION = 0x2
AMDGPU_IDS_FLAGS_TMZ = 0x4
AMDGPU_IDS_FLAGS_CONFORMANT_TRUNC_COORD = 0x8
AMDGPU_IDS_FLAGS_MODE_MASK = 0x300
AMDGPU_IDS_FLAGS_MODE_SHIFT = 0x8
AMDGPU_IDS_FLAGS_MODE_PF = 0x0
AMDGPU_IDS_FLAGS_MODE_VF = 0x1
AMDGPU_IDS_FLAGS_MODE_PT = 0x2
AMDGPU_INFO_ACCEL_WORKING = 0x00
AMDGPU_INFO_CRTC_FROM_ID = 0x01
AMDGPU_INFO_HW_IP_INFO = 0x02
AMDGPU_INFO_HW_IP_COUNT = 0x03
AMDGPU_INFO_TIMESTAMP = 0x05
AMDGPU_INFO_FW_VERSION = 0x0e
AMDGPU_INFO_FW_VCE = 0x1
AMDGPU_INFO_FW_UVD = 0x2
AMDGPU_INFO_FW_GMC = 0x03
AMDGPU_INFO_FW_GFX_ME = 0x04
AMDGPU_INFO_FW_GFX_PFP = 0x05
AMDGPU_INFO_FW_GFX_CE = 0x06
AMDGPU_INFO_FW_GFX_RLC = 0x07
AMDGPU_INFO_FW_GFX_MEC = 0x08
AMDGPU_INFO_FW_SMC = 0x0a
AMDGPU_INFO_FW_SDMA = 0x0b
AMDGPU_INFO_FW_SOS = 0x0c
AMDGPU_INFO_FW_ASD = 0x0d
AMDGPU_INFO_FW_VCN = 0x0e
AMDGPU_INFO_FW_GFX_RLC_RESTORE_LIST_CNTL = 0x0f
AMDGPU_INFO_FW_GFX_RLC_RESTORE_LIST_GPM_MEM = 0x10
AMDGPU_INFO_FW_GFX_RLC_RESTORE_LIST_SRM_MEM = 0x11
AMDGPU_INFO_FW_DMCU = 0x12
AMDGPU_INFO_FW_TA = 0x13
AMDGPU_INFO_FW_DMCUB = 0x14
AMDGPU_INFO_FW_TOC = 0x15
AMDGPU_INFO_FW_CAP = 0x16
AMDGPU_INFO_FW_GFX_RLCP = 0x17
AMDGPU_INFO_FW_GFX_RLCV = 0x18
AMDGPU_INFO_FW_MES_KIQ = 0x19
AMDGPU_INFO_FW_MES = 0x1a
AMDGPU_INFO_FW_IMU = 0x1b
AMDGPU_INFO_FW_VPE = 0x1c
AMDGPU_INFO_NUM_BYTES_MOVED = 0x0f
AMDGPU_INFO_VRAM_USAGE = 0x10
AMDGPU_INFO_GTT_USAGE = 0x11
AMDGPU_INFO_GDS_CONFIG = 0x13
AMDGPU_INFO_VRAM_GTT = 0x14
AMDGPU_INFO_READ_MMR_REG = 0x15
AMDGPU_INFO_DEV_INFO = 0x16
AMDGPU_INFO_VIS_VRAM_USAGE = 0x17
AMDGPU_INFO_NUM_EVICTIONS = 0x18
AMDGPU_INFO_MEMORY = 0x19
AMDGPU_INFO_VCE_CLOCK_TABLE = 0x1A
AMDGPU_INFO_VBIOS = 0x1B
AMDGPU_INFO_VBIOS_SIZE = 0x1
AMDGPU_INFO_VBIOS_IMAGE = 0x2
AMDGPU_INFO_VBIOS_INFO = 0x3
AMDGPU_INFO_NUM_HANDLES = 0x1C
AMDGPU_INFO_SENSOR = 0x1D
AMDGPU_INFO_SENSOR_GFX_SCLK = 0x1
AMDGPU_INFO_SENSOR_GFX_MCLK = 0x2
AMDGPU_INFO_SENSOR_GPU_TEMP = 0x3
AMDGPU_INFO_SENSOR_GPU_LOAD = 0x4
AMDGPU_INFO_SENSOR_GPU_AVG_POWER = 0x5
AMDGPU_INFO_SENSOR_VDDNB = 0x6
AMDGPU_INFO_SENSOR_VDDGFX = 0x7
AMDGPU_INFO_SENSOR_STABLE_PSTATE_GFX_SCLK = 0x8
AMDGPU_INFO_SENSOR_STABLE_PSTATE_GFX_MCLK = 0x9
AMDGPU_INFO_SENSOR_PEAK_PSTATE_GFX_SCLK = 0xa
AMDGPU_INFO_SENSOR_PEAK_PSTATE_GFX_MCLK = 0xb
AMDGPU_INFO_SENSOR_GPU_INPUT_POWER = 0xc
AMDGPU_INFO_NUM_VRAM_CPU_PAGE_FAULTS = 0x1E
AMDGPU_INFO_VRAM_LOST_COUNTER = 0x1F
AMDGPU_INFO_RAS_ENABLED_FEATURES = 0x20
AMDGPU_INFO_RAS_ENABLED_UMC = (1 << 0)
AMDGPU_INFO_RAS_ENABLED_SDMA = (1 << 1)
AMDGPU_INFO_RAS_ENABLED_GFX = (1 << 2)
AMDGPU_INFO_RAS_ENABLED_MMHUB = (1 << 3)
AMDGPU_INFO_RAS_ENABLED_ATHUB = (1 << 4)
AMDGPU_INFO_RAS_ENABLED_PCIE = (1 << 5)
AMDGPU_INFO_RAS_ENABLED_HDP = (1 << 6)
AMDGPU_INFO_RAS_ENABLED_XGMI = (1 << 7)
AMDGPU_INFO_RAS_ENABLED_DF = (1 << 8)
AMDGPU_INFO_RAS_ENABLED_SMN = (1 << 9)
AMDGPU_INFO_RAS_ENABLED_SEM = (1 << 10)
AMDGPU_INFO_RAS_ENABLED_MP0 = (1 << 11)
AMDGPU_INFO_RAS_ENABLED_MP1 = (1 << 12)
AMDGPU_INFO_RAS_ENABLED_FUSE = (1 << 13)
AMDGPU_INFO_VIDEO_CAPS = 0x21
AMDGPU_INFO_VIDEO_CAPS_DECODE = 0
AMDGPU_INFO_VIDEO_CAPS_ENCODE = 1
AMDGPU_INFO_MAX_IBS = 0x22
AMDGPU_INFO_GPUVM_FAULT = 0x23
AMDGPU_INFO_UQ_FW_AREAS = 0x24
AMDGPU_INFO_CAPABILITY = 0x50
AMDGPU_INFO_VIRTUAL_RANGE = 0x51
AMDGPU_CAPABILITY_PIN_MEM_FLAG = (1 << 0)
AMDGPU_CAPABILITY_DIRECT_GMA_FLAG = (1 << 1)
AMDGPU_INFO_MMR_SE_INDEX_SHIFT = 0
AMDGPU_INFO_MMR_SE_INDEX_MASK = 0xff
AMDGPU_INFO_MMR_SH_INDEX_SHIFT = 8
AMDGPU_INFO_MMR_SH_INDEX_MASK = 0xff
AMDGPU_VRAM_TYPE_UNKNOWN = 0
AMDGPU_VRAM_TYPE_GDDR1 = 1
AMDGPU_VRAM_TYPE_DDR2 = 2
AMDGPU_VRAM_TYPE_GDDR3 = 3
AMDGPU_VRAM_TYPE_GDDR4 = 4
AMDGPU_VRAM_TYPE_GDDR5 = 5
AMDGPU_VRAM_TYPE_HBM = 6
AMDGPU_VRAM_TYPE_DDR3 = 7
AMDGPU_VRAM_TYPE_DDR4 = 8
AMDGPU_VRAM_TYPE_GDDR6 = 9
AMDGPU_VRAM_TYPE_DDR5 = 10
AMDGPU_VRAM_TYPE_LPDDR4 = 11
AMDGPU_VRAM_TYPE_LPDDR5 = 12
AMDGPU_VRAM_TYPE_HBM3E = 13
AMDGPU_VRAM_TYPE_HBM_WIDTH = 4096
AMDGPU_VCE_CLOCK_TABLE_ENTRIES = 6
AMDGPU_INFO_VIDEO_CAPS_CODEC_IDX_MPEG2 = 0
AMDGPU_INFO_VIDEO_CAPS_CODEC_IDX_MPEG4 = 1
AMDGPU_INFO_VIDEO_CAPS_CODEC_IDX_VC1 = 2
AMDGPU_INFO_VIDEO_CAPS_CODEC_IDX_MPEG4_AVC = 3
AMDGPU_INFO_VIDEO_CAPS_CODEC_IDX_HEVC = 4
AMDGPU_INFO_VIDEO_CAPS_CODEC_IDX_JPEG = 5
AMDGPU_INFO_VIDEO_CAPS_CODEC_IDX_VP9 = 6
AMDGPU_INFO_VIDEO_CAPS_CODEC_IDX_AV1 = 7
AMDGPU_INFO_VIDEO_CAPS_CODEC_IDX_COUNT = 8
AMDGPU_VMHUB_TYPE_MASK = 0xff
AMDGPU_VMHUB_TYPE_SHIFT = 0
AMDGPU_VMHUB_TYPE_GFX = 0
AMDGPU_VMHUB_TYPE_MM0 = 1
AMDGPU_VMHUB_TYPE_MM1 = 2
AMDGPU_VMHUB_IDX_MASK = 0xff00
AMDGPU_VMHUB_IDX_SHIFT = 8
AMDGPU_FAMILY_UNKNOWN = 0
AMDGPU_FAMILY_SI = 110
AMDGPU_FAMILY_CI = 120
AMDGPU_FAMILY_KV = 125
AMDGPU_FAMILY_VI = 130
AMDGPU_FAMILY_CZ = 135
AMDGPU_FAMILY_AI = 141
AMDGPU_FAMILY_RV = 142
AMDGPU_FAMILY_NV = 143
AMDGPU_FAMILY_VGH = 144
AMDGPU_FAMILY_GC_11_0_0 = 145
AMDGPU_FAMILY_YC = 146
AMDGPU_FAMILY_GC_11_0_1 = 148
AMDGPU_FAMILY_GC_10_3_6 = 149
AMDGPU_FAMILY_GC_10_3_7 = 151
AMDGPU_FAMILY_GC_11_5_0 = 150
AMDGPU_FAMILY_GC_12_0_0 = 152
AMDGPU_SUA_APERTURE_PRIVATE = 1
AMDGPU_SUA_APERTURE_SHARED = 2
AMDGPU_FREESYNC_FULLSCREEN_ENTER = 1
AMDGPU_FREESYNC_FULLSCREEN_EXIT = 2