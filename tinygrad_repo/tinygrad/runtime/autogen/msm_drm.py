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
class struct_drm_gem_change_handle(c.Struct):
  SIZE = 8
  handle: int
  new_handle: int
struct_drm_gem_change_handle.register_fields([('handle', ctypes.c_uint32, 0), ('new_handle', ctypes.c_uint32, 4)])
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
  SIZE = 24
  handle: int
  flags: int
  fd: int
  pad: int
  point: int
struct_drm_syncobj_handle.register_fields([('handle', ctypes.c_uint32, 0), ('flags', ctypes.c_uint32, 4), ('fd', ctypes.c_int32, 8), ('pad', ctypes.c_uint32, 12), ('point', ctypes.c_uint64, 16)])
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
class struct_drm_set_client_name(c.Struct):
  SIZE = 16
  name_len: int
  name: int
struct_drm_set_client_name.register_fields([('name_len', ctypes.c_uint64, 0), ('name', ctypes.c_uint64, 8)])
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
class struct_drm_msm_timespec(c.Struct):
  SIZE = 16
  tv_sec: int
  tv_nsec: int
struct_drm_msm_timespec.register_fields([('tv_sec', ctypes.c_int64, 0), ('tv_nsec', ctypes.c_int64, 8)])
@c.record
class struct_drm_msm_param(c.Struct):
  SIZE = 24
  pipe: int
  param: int
  value: int
  len: int
  pad: int
struct_drm_msm_param.register_fields([('pipe', ctypes.c_uint32, 0), ('param', ctypes.c_uint32, 4), ('value', ctypes.c_uint64, 8), ('len', ctypes.c_uint32, 16), ('pad', ctypes.c_uint32, 20)])
@c.record
class struct_drm_msm_gem_new(c.Struct):
  SIZE = 16
  size: int
  flags: int
  handle: int
struct_drm_msm_gem_new.register_fields([('size', ctypes.c_uint64, 0), ('flags', ctypes.c_uint32, 8), ('handle', ctypes.c_uint32, 12)])
@c.record
class struct_drm_msm_gem_info(c.Struct):
  SIZE = 24
  handle: int
  info: int
  value: int
  len: int
  pad: int
struct_drm_msm_gem_info.register_fields([('handle', ctypes.c_uint32, 0), ('info', ctypes.c_uint32, 4), ('value', ctypes.c_uint64, 8), ('len', ctypes.c_uint32, 16), ('pad', ctypes.c_uint32, 20)])
@c.record
class struct_drm_msm_gem_cpu_prep(c.Struct):
  SIZE = 24
  handle: int
  op: int
  timeout: struct_drm_msm_timespec
struct_drm_msm_gem_cpu_prep.register_fields([('handle', ctypes.c_uint32, 0), ('op', ctypes.c_uint32, 4), ('timeout', struct_drm_msm_timespec, 8)])
@c.record
class struct_drm_msm_gem_cpu_fini(c.Struct):
  SIZE = 4
  handle: int
struct_drm_msm_gem_cpu_fini.register_fields([('handle', ctypes.c_uint32, 0)])
@c.record
class struct_drm_msm_syncobj(c.Struct):
  SIZE = 16
  handle: int
  flags: int
  point: int
struct_drm_msm_syncobj.register_fields([('handle', ctypes.c_uint32, 0), ('flags', ctypes.c_uint32, 4), ('point', ctypes.c_uint64, 8)])
@c.record
class struct_drm_msm_gem_submit_reloc(c.Struct):
  SIZE = 24
  submit_offset: int
  _or: int
  shift: int
  reloc_idx: int
  reloc_offset: int
struct_drm_msm_gem_submit_reloc.register_fields([('submit_offset', ctypes.c_uint32, 0), ('_or', ctypes.c_uint32, 4), ('shift', ctypes.c_int32, 8), ('reloc_idx', ctypes.c_uint32, 12), ('reloc_offset', ctypes.c_uint64, 16)])
@c.record
class struct_drm_msm_gem_submit_cmd(c.Struct):
  SIZE = 32
  type: int
  submit_idx: int
  submit_offset: int
  size: int
  pad: int
  nr_relocs: int
  relocs: int
  iova: int
struct_drm_msm_gem_submit_cmd.register_fields([('type', ctypes.c_uint32, 0), ('submit_idx', ctypes.c_uint32, 4), ('submit_offset', ctypes.c_uint32, 8), ('size', ctypes.c_uint32, 12), ('pad', ctypes.c_uint32, 16), ('nr_relocs', ctypes.c_uint32, 20), ('relocs', ctypes.c_uint64, 24), ('iova', ctypes.c_uint64, 24)])
@c.record
class struct_drm_msm_gem_submit_bo(c.Struct):
  SIZE = 16
  flags: int
  handle: int
  presumed: int
struct_drm_msm_gem_submit_bo.register_fields([('flags', ctypes.c_uint32, 0), ('handle', ctypes.c_uint32, 4), ('presumed', ctypes.c_uint64, 8)])
@c.record
class struct_drm_msm_gem_submit(c.Struct):
  SIZE = 72
  flags: int
  fence: int
  nr_bos: int
  nr_cmds: int
  bos: int
  cmds: int
  fence_fd: int
  queueid: int
  in_syncobjs: int
  out_syncobjs: int
  nr_in_syncobjs: int
  nr_out_syncobjs: int
  syncobj_stride: int
  pad: int
struct_drm_msm_gem_submit.register_fields([('flags', ctypes.c_uint32, 0), ('fence', ctypes.c_uint32, 4), ('nr_bos', ctypes.c_uint32, 8), ('nr_cmds', ctypes.c_uint32, 12), ('bos', ctypes.c_uint64, 16), ('cmds', ctypes.c_uint64, 24), ('fence_fd', ctypes.c_int32, 32), ('queueid', ctypes.c_uint32, 36), ('in_syncobjs', ctypes.c_uint64, 40), ('out_syncobjs', ctypes.c_uint64, 48), ('nr_in_syncobjs', ctypes.c_uint32, 56), ('nr_out_syncobjs', ctypes.c_uint32, 60), ('syncobj_stride', ctypes.c_uint32, 64), ('pad', ctypes.c_uint32, 68)])
@c.record
class struct_drm_msm_vm_bind_op(c.Struct):
  SIZE = 40
  op: int
  handle: int
  obj_offset: int
  iova: int
  range: int
  flags: int
  pad: int
struct_drm_msm_vm_bind_op.register_fields([('op', ctypes.c_uint32, 0), ('handle', ctypes.c_uint32, 4), ('obj_offset', ctypes.c_uint64, 8), ('iova', ctypes.c_uint64, 16), ('range', ctypes.c_uint64, 24), ('flags', ctypes.c_uint32, 32), ('pad', ctypes.c_uint32, 36)])
@c.record
class struct_drm_msm_vm_bind(c.Struct):
  SIZE = 88
  flags: int
  nr_ops: int
  fence_fd: int
  queue_id: int
  in_syncobjs: int
  out_syncobjs: int
  nr_in_syncobjs: int
  nr_out_syncobjs: int
  syncobj_stride: int
  op_stride: int
  op: struct_drm_msm_vm_bind_op
  ops: int
struct_drm_msm_vm_bind.register_fields([('flags', ctypes.c_uint32, 0), ('nr_ops', ctypes.c_uint32, 4), ('fence_fd', ctypes.c_int32, 8), ('queue_id', ctypes.c_uint32, 12), ('in_syncobjs', ctypes.c_uint64, 16), ('out_syncobjs', ctypes.c_uint64, 24), ('nr_in_syncobjs', ctypes.c_uint32, 32), ('nr_out_syncobjs', ctypes.c_uint32, 36), ('syncobj_stride', ctypes.c_uint32, 40), ('op_stride', ctypes.c_uint32, 44), ('op', struct_drm_msm_vm_bind_op, 48), ('ops', ctypes.c_uint64, 48)])
@c.record
class struct_drm_msm_wait_fence(c.Struct):
  SIZE = 32
  fence: int
  flags: int
  timeout: struct_drm_msm_timespec
  queueid: int
struct_drm_msm_wait_fence.register_fields([('fence', ctypes.c_uint32, 0), ('flags', ctypes.c_uint32, 4), ('timeout', struct_drm_msm_timespec, 8), ('queueid', ctypes.c_uint32, 24)])
@c.record
class struct_drm_msm_gem_madvise(c.Struct):
  SIZE = 12
  handle: int
  madv: int
  retained: int
struct_drm_msm_gem_madvise.register_fields([('handle', ctypes.c_uint32, 0), ('madv', ctypes.c_uint32, 4), ('retained', ctypes.c_uint32, 8)])
@c.record
class struct_drm_msm_submitqueue(c.Struct):
  SIZE = 12
  flags: int
  prio: int
  id: int
struct_drm_msm_submitqueue.register_fields([('flags', ctypes.c_uint32, 0), ('prio', ctypes.c_uint32, 4), ('id', ctypes.c_uint32, 8)])
@c.record
class struct_drm_msm_submitqueue_query(c.Struct):
  SIZE = 24
  data: int
  id: int
  param: int
  len: int
  pad: int
struct_drm_msm_submitqueue_query.register_fields([('data', ctypes.c_uint64, 0), ('id', ctypes.c_uint32, 8), ('param', ctypes.c_uint32, 12), ('len', ctypes.c_uint32, 16), ('pad', ctypes.c_uint32, 20)])
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
DRM_SYNCOBJ_FD_TO_HANDLE_FLAGS_TIMELINE = (1 << 1)
DRM_SYNCOBJ_HANDLE_TO_FD_FLAGS_EXPORT_SYNC_FILE = (1 << 0)
DRM_SYNCOBJ_HANDLE_TO_FD_FLAGS_TIMELINE = (1 << 1)
DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL = (1 << 0)
DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT = (1 << 1)
DRM_SYNCOBJ_WAIT_FLAGS_WAIT_AVAILABLE = (1 << 2)
DRM_SYNCOBJ_WAIT_FLAGS_WAIT_DEADLINE = (1 << 3)
DRM_SYNCOBJ_QUERY_FLAGS_LAST_SUBMITTED = (1 << 0)
DRM_CRTC_SEQUENCE_RELATIVE = 0x00000001
DRM_CRTC_SEQUENCE_NEXT_ON_MISS = 0x00000002
DRM_CLIENT_NAME_MAX_LEN = 64
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
DRM_IOCTL_SET_CLIENT_NAME = DRM_IOWR(0xD1, struct_drm_set_client_name)
DRM_IOCTL_GEM_CHANGE_HANDLE = DRM_IOWR(0xD2, struct_drm_gem_change_handle)
DRM_COMMAND_BASE = 0x40
DRM_COMMAND_END = 0xA0
DRM_EVENT_VBLANK = 0x01
DRM_EVENT_FLIP_COMPLETE = 0x02
DRM_EVENT_CRTC_SEQUENCE = 0x03
MSM_PIPE_NONE = 0x00
MSM_PIPE_2D0 = 0x01
MSM_PIPE_2D1 = 0x02
MSM_PIPE_3D0 = 0x10
MSM_PIPE_ID_MASK = 0xffff
MSM_PIPE_ID = lambda x: ((x) & MSM_PIPE_ID_MASK) # type: ignore
MSM_PIPE_FLAGS = lambda x: ((x) & ~MSM_PIPE_ID_MASK) # type: ignore
MSM_PARAM_GPU_ID = 0x01
MSM_PARAM_GMEM_SIZE = 0x02
MSM_PARAM_CHIP_ID = 0x03
MSM_PARAM_MAX_FREQ = 0x04
MSM_PARAM_TIMESTAMP = 0x05
MSM_PARAM_GMEM_BASE = 0x06
MSM_PARAM_PRIORITIES = 0x07
MSM_PARAM_PP_PGTABLE = 0x08
MSM_PARAM_FAULTS = 0x09
MSM_PARAM_SUSPENDS = 0x0a
MSM_PARAM_SYSPROF = 0x0b
MSM_PARAM_COMM = 0x0c
MSM_PARAM_CMDLINE = 0x0d
MSM_PARAM_VA_START = 0x0e
MSM_PARAM_VA_SIZE = 0x0f
MSM_PARAM_HIGHEST_BANK_BIT = 0x10
MSM_PARAM_RAYTRACING = 0x11
MSM_PARAM_UBWC_SWIZZLE = 0x12
MSM_PARAM_MACROTILE_MODE = 0x13
MSM_PARAM_UCHE_TRAP_BASE = 0x14
MSM_PARAM_HAS_PRR = 0x15
MSM_PARAM_EN_VM_BIND = 0x16
MSM_PARAM_NR_RINGS = MSM_PARAM_PRIORITIES
MSM_BO_SCANOUT = 0x00000001
MSM_BO_GPU_READONLY = 0x00000002
MSM_BO_NO_SHARE = 0x00000004
MSM_BO_CACHE_MASK = 0x000f0000
MSM_BO_CACHED = 0x00010000
MSM_BO_WC = 0x00020000
MSM_BO_UNCACHED = 0x00040000
MSM_BO_CACHED_COHERENT = 0x080000
MSM_BO_FLAGS = (MSM_BO_SCANOUT | MSM_BO_GPU_READONLY | MSM_BO_NO_SHARE | MSM_BO_CACHE_MASK)
MSM_INFO_GET_OFFSET = 0x00
MSM_INFO_GET_IOVA = 0x01
MSM_INFO_SET_NAME = 0x02
MSM_INFO_GET_NAME = 0x03
MSM_INFO_SET_IOVA = 0x04
MSM_INFO_GET_FLAGS = 0x05
MSM_INFO_SET_METADATA = 0x06
MSM_INFO_GET_METADATA = 0x07
MSM_PREP_READ = 0x01
MSM_PREP_WRITE = 0x02
MSM_PREP_NOSYNC = 0x04
MSM_PREP_BOOST = 0x08
MSM_PREP_FLAGS = (MSM_PREP_READ | MSM_PREP_WRITE | MSM_PREP_NOSYNC | MSM_PREP_BOOST | 0)
MSM_SYNCOBJ_RESET = 0x00000001
MSM_SYNCOBJ_FLAGS = ( MSM_SYNCOBJ_RESET | 0)
MSM_SUBMIT_CMD_BUF = 0x0001
MSM_SUBMIT_CMD_IB_TARGET_BUF = 0x0002
MSM_SUBMIT_CMD_CTX_RESTORE_BUF = 0x0003
MSM_SUBMIT_BO_READ = 0x0001
MSM_SUBMIT_BO_WRITE = 0x0002
MSM_SUBMIT_BO_DUMP = 0x0004
MSM_SUBMIT_BO_NO_IMPLICIT = 0x0008
MSM_SUBMIT_BO_FLAGS = (MSM_SUBMIT_BO_READ | MSM_SUBMIT_BO_WRITE | MSM_SUBMIT_BO_DUMP | MSM_SUBMIT_BO_NO_IMPLICIT)
MSM_SUBMIT_NO_IMPLICIT = 0x80000000
MSM_SUBMIT_FENCE_FD_IN = 0x40000000
MSM_SUBMIT_FENCE_FD_OUT = 0x20000000
MSM_SUBMIT_SUDO = 0x10000000
MSM_SUBMIT_SYNCOBJ_IN = 0x08000000
MSM_SUBMIT_SYNCOBJ_OUT = 0x04000000
MSM_SUBMIT_FENCE_SN_IN = 0x02000000
MSM_SUBMIT_FLAGS = ( MSM_SUBMIT_NO_IMPLICIT   | MSM_SUBMIT_FENCE_FD_IN   | MSM_SUBMIT_FENCE_FD_OUT  | MSM_SUBMIT_SUDO          | MSM_SUBMIT_SYNCOBJ_IN    | MSM_SUBMIT_SYNCOBJ_OUT   | MSM_SUBMIT_FENCE_SN_IN   | 0)
MSM_VM_BIND_OP_UNMAP = 0
MSM_VM_BIND_OP_MAP = 1
MSM_VM_BIND_OP_MAP_NULL = 2
MSM_VM_BIND_OP_DUMP = 1
MSM_VM_BIND_OP_FLAGS = ( MSM_VM_BIND_OP_DUMP | 0)
MSM_VM_BIND_FENCE_FD_IN = 0x00000001
MSM_VM_BIND_FENCE_FD_OUT = 0x00000002
MSM_VM_BIND_FLAGS = ( MSM_VM_BIND_FENCE_FD_IN | MSM_VM_BIND_FENCE_FD_OUT | 0)
MSM_WAIT_FENCE_BOOST = 0x00000001
MSM_WAIT_FENCE_FLAGS = ( MSM_WAIT_FENCE_BOOST | 0)
MSM_MADV_WILLNEED = 0
MSM_MADV_DONTNEED = 1
__MSM_MADV_PURGED = 2
MSM_SUBMITQUEUE_ALLOW_PREEMPT = 0x00000001
MSM_SUBMITQUEUE_VM_BIND = 0x00000002
MSM_SUBMITQUEUE_FLAGS = ( MSM_SUBMITQUEUE_ALLOW_PREEMPT | MSM_SUBMITQUEUE_VM_BIND | 0)
MSM_SUBMITQUEUE_PARAM_FAULTS = 0
DRM_MSM_GET_PARAM = 0x00
DRM_MSM_SET_PARAM = 0x01
DRM_MSM_GEM_NEW = 0x02
DRM_MSM_GEM_INFO = 0x03
DRM_MSM_GEM_CPU_PREP = 0x04
DRM_MSM_GEM_CPU_FINI = 0x05
DRM_MSM_GEM_SUBMIT = 0x06
DRM_MSM_WAIT_FENCE = 0x07
DRM_MSM_GEM_MADVISE = 0x08
DRM_MSM_SUBMITQUEUE_NEW = 0x0A
DRM_MSM_SUBMITQUEUE_CLOSE = 0x0B
DRM_MSM_SUBMITQUEUE_QUERY = 0x0C
DRM_MSM_VM_BIND = 0x0D
DRM_IOCTL_MSM_GET_PARAM = DRM_IOWR(DRM_COMMAND_BASE + DRM_MSM_GET_PARAM, struct_drm_msm_param)
DRM_IOCTL_MSM_SET_PARAM = DRM_IOW (DRM_COMMAND_BASE + DRM_MSM_SET_PARAM, struct_drm_msm_param)
DRM_IOCTL_MSM_GEM_NEW = DRM_IOWR(DRM_COMMAND_BASE + DRM_MSM_GEM_NEW, struct_drm_msm_gem_new)
DRM_IOCTL_MSM_GEM_INFO = DRM_IOWR(DRM_COMMAND_BASE + DRM_MSM_GEM_INFO, struct_drm_msm_gem_info)
DRM_IOCTL_MSM_GEM_CPU_PREP = DRM_IOW (DRM_COMMAND_BASE + DRM_MSM_GEM_CPU_PREP, struct_drm_msm_gem_cpu_prep)
DRM_IOCTL_MSM_GEM_CPU_FINI = DRM_IOW (DRM_COMMAND_BASE + DRM_MSM_GEM_CPU_FINI, struct_drm_msm_gem_cpu_fini)
DRM_IOCTL_MSM_GEM_SUBMIT = DRM_IOWR(DRM_COMMAND_BASE + DRM_MSM_GEM_SUBMIT, struct_drm_msm_gem_submit)
DRM_IOCTL_MSM_WAIT_FENCE = DRM_IOW (DRM_COMMAND_BASE + DRM_MSM_WAIT_FENCE, struct_drm_msm_wait_fence)
DRM_IOCTL_MSM_GEM_MADVISE = DRM_IOWR(DRM_COMMAND_BASE + DRM_MSM_GEM_MADVISE, struct_drm_msm_gem_madvise)
DRM_IOCTL_MSM_SUBMITQUEUE_NEW = DRM_IOWR(DRM_COMMAND_BASE + DRM_MSM_SUBMITQUEUE_NEW, struct_drm_msm_submitqueue)
DRM_IOCTL_MSM_SUBMITQUEUE_CLOSE = DRM_IOW (DRM_COMMAND_BASE + DRM_MSM_SUBMITQUEUE_CLOSE, __u32)
DRM_IOCTL_MSM_SUBMITQUEUE_QUERY = DRM_IOW (DRM_COMMAND_BASE + DRM_MSM_SUBMITQUEUE_QUERY, struct_drm_msm_submitqueue_query)
DRM_IOCTL_MSM_VM_BIND = DRM_IOWR(DRM_COMMAND_BASE + DRM_MSM_VM_BIND, struct_drm_msm_vm_bind)