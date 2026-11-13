# mypy: disable-error-code="empty-body"
from __future__ import annotations
import ctypes
from typing import Literal, TypeAlias
from tinygrad.runtime.support.c import _IO, _IOW, _IOR, _IOWR
from tinygrad.runtime.support import c
@c.record
class struct_io_uring_sq(c.Struct):
  SIZE = 104
  khead: c.POINTER[ctypes.c_uint32]
  ktail: c.POINTER[ctypes.c_uint32]
  kring_mask: c.POINTER[ctypes.c_uint32]
  kring_entries: c.POINTER[ctypes.c_uint32]
  kflags: c.POINTER[ctypes.c_uint32]
  kdropped: c.POINTER[ctypes.c_uint32]
  array: c.POINTER[ctypes.c_uint32]
  sqes: c.POINTER[struct_io_uring_sqe]
  sqe_head: int
  sqe_tail: int
  ring_sz: int
  ring_ptr: ctypes.c_void_p
  ring_mask: int
  ring_entries: int
  sqes_sz: int
  pad: int
@c.record
class struct_io_uring_sqe(c.Struct):
  SIZE = 64
  opcode: int
  flags: int
  ioprio: int
  fd: int
  off: int
  addr2: int
  cmd_op: int
  __pad1: int
  addr: int
  splice_off_in: int
  len: int
  rw_flags: int
  fsync_flags: int
  poll_events: int
  poll32_events: int
  sync_range_flags: int
  msg_flags: int
  timeout_flags: int
  accept_flags: int
  cancel_flags: int
  open_flags: int
  statx_flags: int
  fadvise_advice: int
  splice_flags: int
  rename_flags: int
  unlink_flags: int
  hardlink_flags: int
  xattr_flags: int
  msg_ring_flags: int
  uring_cmd_flags: int
  user_data: int
  buf_index: int
  buf_group: int
  personality: int
  splice_fd_in: int
  file_index: int
  addr_len: int
  __pad3: c.Array[ctypes.c_uint16, Literal[1]]
  addr3: int
  __pad2: c.Array[ctypes.c_uint64, Literal[1]]
  cmd: c.Array[ctypes.c_ubyte, Literal[0]]
__u8: TypeAlias = ctypes.c_ubyte
__u16: TypeAlias = ctypes.c_uint16
__s32: TypeAlias = ctypes.c_int32
__u64: TypeAlias = ctypes.c_uint64
__u32: TypeAlias = ctypes.c_uint32
__kernel_rwf_t: TypeAlias = ctypes.c_int32
struct_io_uring_sqe.register_fields([('opcode', ctypes.c_ubyte, 0), ('flags', ctypes.c_ubyte, 1), ('ioprio', ctypes.c_uint16, 2), ('fd', ctypes.c_int32, 4), ('off', ctypes.c_uint64, 8), ('addr2', ctypes.c_uint64, 8), ('cmd_op', ctypes.c_uint32, 8), ('__pad1', ctypes.c_uint32, 12), ('addr', ctypes.c_uint64, 16), ('splice_off_in', ctypes.c_uint64, 16), ('len', ctypes.c_uint32, 24), ('rw_flags', ctypes.c_int32, 28), ('fsync_flags', ctypes.c_uint32, 28), ('poll_events', ctypes.c_uint16, 28), ('poll32_events', ctypes.c_uint32, 28), ('sync_range_flags', ctypes.c_uint32, 28), ('msg_flags', ctypes.c_uint32, 28), ('timeout_flags', ctypes.c_uint32, 28), ('accept_flags', ctypes.c_uint32, 28), ('cancel_flags', ctypes.c_uint32, 28), ('open_flags', ctypes.c_uint32, 28), ('statx_flags', ctypes.c_uint32, 28), ('fadvise_advice', ctypes.c_uint32, 28), ('splice_flags', ctypes.c_uint32, 28), ('rename_flags', ctypes.c_uint32, 28), ('unlink_flags', ctypes.c_uint32, 28), ('hardlink_flags', ctypes.c_uint32, 28), ('xattr_flags', ctypes.c_uint32, 28), ('msg_ring_flags', ctypes.c_uint32, 28), ('uring_cmd_flags', ctypes.c_uint32, 28), ('user_data', ctypes.c_uint64, 32), ('buf_index', ctypes.c_uint16, 40), ('buf_group', ctypes.c_uint16, 40), ('personality', ctypes.c_uint16, 42), ('splice_fd_in', ctypes.c_int32, 44), ('file_index', ctypes.c_uint32, 44), ('addr_len', ctypes.c_uint16, 44), ('__pad3', c.Array[ctypes.c_uint16, Literal[1]], 46), ('addr3', ctypes.c_uint64, 48), ('__pad2', c.Array[ctypes.c_uint64, Literal[1]], 56), ('cmd', c.Array[ctypes.c_ubyte, Literal[0]], 48)])
size_t: TypeAlias = ctypes.c_uint64
struct_io_uring_sq.register_fields([('khead', c.POINTER[ctypes.c_uint32], 0), ('ktail', c.POINTER[ctypes.c_uint32], 8), ('kring_mask', c.POINTER[ctypes.c_uint32], 16), ('kring_entries', c.POINTER[ctypes.c_uint32], 24), ('kflags', c.POINTER[ctypes.c_uint32], 32), ('kdropped', c.POINTER[ctypes.c_uint32], 40), ('array', c.POINTER[ctypes.c_uint32], 48), ('sqes', c.POINTER[struct_io_uring_sqe], 56), ('sqe_head', ctypes.c_uint32, 64), ('sqe_tail', ctypes.c_uint32, 68), ('ring_sz', size_t, 72), ('ring_ptr', ctypes.c_void_p, 80), ('ring_mask', ctypes.c_uint32, 88), ('ring_entries', ctypes.c_uint32, 92), ('sqes_sz', ctypes.c_uint32, 96), ('pad', ctypes.c_uint32, 100)])
@c.record
class struct_io_uring_cq(c.Struct):
  SIZE = 88
  khead: c.POINTER[ctypes.c_uint32]
  ktail: c.POINTER[ctypes.c_uint32]
  kring_mask: c.POINTER[ctypes.c_uint32]
  kring_entries: c.POINTER[ctypes.c_uint32]
  kflags: c.POINTER[ctypes.c_uint32]
  koverflow: c.POINTER[ctypes.c_uint32]
  cqes: c.POINTER[struct_io_uring_cqe]
  ring_sz: int
  ring_ptr: ctypes.c_void_p
  ring_mask: int
  ring_entries: int
  pad: c.Array[ctypes.c_uint32, Literal[2]]
@c.record
class struct_io_uring_cqe(c.Struct):
  SIZE = 16
  user_data: int
  res: int
  flags: int
  big_cqe: c.Array[ctypes.c_uint64, Literal[0]]
struct_io_uring_cqe.register_fields([('user_data', ctypes.c_uint64, 0), ('res', ctypes.c_int32, 8), ('flags', ctypes.c_uint32, 12), ('big_cqe', c.Array[ctypes.c_uint64, Literal[0]], 16)])
struct_io_uring_cq.register_fields([('khead', c.POINTER[ctypes.c_uint32], 0), ('ktail', c.POINTER[ctypes.c_uint32], 8), ('kring_mask', c.POINTER[ctypes.c_uint32], 16), ('kring_entries', c.POINTER[ctypes.c_uint32], 24), ('kflags', c.POINTER[ctypes.c_uint32], 32), ('koverflow', c.POINTER[ctypes.c_uint32], 40), ('cqes', c.POINTER[struct_io_uring_cqe], 48), ('ring_sz', size_t, 56), ('ring_ptr', ctypes.c_void_p, 64), ('ring_mask', ctypes.c_uint32, 72), ('ring_entries', ctypes.c_uint32, 76), ('pad', c.Array[ctypes.c_uint32, Literal[2]], 80)])
@c.record
class struct_io_uring(c.Struct):
  SIZE = 216
  sq: struct_io_uring_sq
  cq: struct_io_uring_cq
  flags: int
  ring_fd: int
  features: int
  enter_ring_fd: int
  int_flags: int
  pad: c.Array[ctypes.c_ubyte, Literal[3]]
  pad2: int
struct_io_uring.register_fields([('sq', struct_io_uring_sq, 0), ('cq', struct_io_uring_cq, 104), ('flags', ctypes.c_uint32, 192), ('ring_fd', ctypes.c_int32, 196), ('features', ctypes.c_uint32, 200), ('enter_ring_fd', ctypes.c_int32, 204), ('int_flags', ctypes.c_ubyte, 208), ('pad', c.Array[ctypes.c_ubyte, Literal[3]], 209), ('pad2', ctypes.c_uint32, 212)])
@c.record
class struct_io_uring_zcrx_rq(c.Struct):
  SIZE = 40
  khead: c.POINTER[ctypes.c_uint32]
  ktail: c.POINTER[ctypes.c_uint32]
  rq_tail: int
  ring_entries: int
  rqes: c.POINTER[struct_io_uring_zcrx_rqe]
  ring_ptr: ctypes.c_void_p
@c.record
class struct_io_uring_zcrx_rqe(c.Struct):
  SIZE = 16
  off: int
  len: int
  __pad: int
struct_io_uring_zcrx_rq.register_fields([('khead', c.POINTER[ctypes.c_uint32], 0), ('ktail', c.POINTER[ctypes.c_uint32], 8), ('rq_tail', ctypes.c_uint32, 16), ('ring_entries', ctypes.c_uint32, 20), ('rqes', c.POINTER[struct_io_uring_zcrx_rqe], 24), ('ring_ptr', ctypes.c_void_p, 32)])
@c.record
class struct_io_uring_cqe_iter(c.Struct):
  SIZE = 24
  cqes: c.POINTER[struct_io_uring_cqe]
  mask: int
  shift: int
  head: int
  tail: int
struct_io_uring_cqe_iter.register_fields([('cqes', c.POINTER[struct_io_uring_cqe], 0), ('mask', ctypes.c_uint32, 8), ('shift', ctypes.c_uint32, 12), ('head', ctypes.c_uint32, 16), ('tail', ctypes.c_uint32, 20)])
class struct_epoll_event(c.Struct): pass
class struct_statx(c.Struct): pass
class struct_futex_waitv(c.Struct): pass
@c.record
class struct_io_uring_attr_pi(c.Struct):
  SIZE = 32
  flags: int
  app_tag: int
  len: int
  addr: int
  seed: int
  rsvd: int
struct_io_uring_attr_pi.register_fields([('flags', ctypes.c_uint16, 0), ('app_tag', ctypes.c_uint16, 2), ('len', ctypes.c_uint32, 4), ('addr', ctypes.c_uint64, 8), ('seed', ctypes.c_uint64, 16), ('rsvd', ctypes.c_uint64, 24)])
enum_io_uring_sqe_flags_bit: dict[int, str] = {(IOSQE_FIXED_FILE_BIT:=0): 'IOSQE_FIXED_FILE_BIT', (IOSQE_IO_DRAIN_BIT:=1): 'IOSQE_IO_DRAIN_BIT', (IOSQE_IO_LINK_BIT:=2): 'IOSQE_IO_LINK_BIT', (IOSQE_IO_HARDLINK_BIT:=3): 'IOSQE_IO_HARDLINK_BIT', (IOSQE_ASYNC_BIT:=4): 'IOSQE_ASYNC_BIT', (IOSQE_BUFFER_SELECT_BIT:=5): 'IOSQE_BUFFER_SELECT_BIT', (IOSQE_CQE_SKIP_SUCCESS_BIT:=6): 'IOSQE_CQE_SKIP_SUCCESS_BIT'}
enum_io_uring_op: dict[int, str] = {(IORING_OP_NOP:=0): 'IORING_OP_NOP', (IORING_OP_READV:=1): 'IORING_OP_READV', (IORING_OP_WRITEV:=2): 'IORING_OP_WRITEV', (IORING_OP_FSYNC:=3): 'IORING_OP_FSYNC', (IORING_OP_READ_FIXED:=4): 'IORING_OP_READ_FIXED', (IORING_OP_WRITE_FIXED:=5): 'IORING_OP_WRITE_FIXED', (IORING_OP_POLL_ADD:=6): 'IORING_OP_POLL_ADD', (IORING_OP_POLL_REMOVE:=7): 'IORING_OP_POLL_REMOVE', (IORING_OP_SYNC_FILE_RANGE:=8): 'IORING_OP_SYNC_FILE_RANGE', (IORING_OP_SENDMSG:=9): 'IORING_OP_SENDMSG', (IORING_OP_RECVMSG:=10): 'IORING_OP_RECVMSG', (IORING_OP_TIMEOUT:=11): 'IORING_OP_TIMEOUT', (IORING_OP_TIMEOUT_REMOVE:=12): 'IORING_OP_TIMEOUT_REMOVE', (IORING_OP_ACCEPT:=13): 'IORING_OP_ACCEPT', (IORING_OP_ASYNC_CANCEL:=14): 'IORING_OP_ASYNC_CANCEL', (IORING_OP_LINK_TIMEOUT:=15): 'IORING_OP_LINK_TIMEOUT', (IORING_OP_CONNECT:=16): 'IORING_OP_CONNECT', (IORING_OP_FALLOCATE:=17): 'IORING_OP_FALLOCATE', (IORING_OP_OPENAT:=18): 'IORING_OP_OPENAT', (IORING_OP_CLOSE:=19): 'IORING_OP_CLOSE', (IORING_OP_FILES_UPDATE:=20): 'IORING_OP_FILES_UPDATE', (IORING_OP_STATX:=21): 'IORING_OP_STATX', (IORING_OP_READ:=22): 'IORING_OP_READ', (IORING_OP_WRITE:=23): 'IORING_OP_WRITE', (IORING_OP_FADVISE:=24): 'IORING_OP_FADVISE', (IORING_OP_MADVISE:=25): 'IORING_OP_MADVISE', (IORING_OP_SEND:=26): 'IORING_OP_SEND', (IORING_OP_RECV:=27): 'IORING_OP_RECV', (IORING_OP_OPENAT2:=28): 'IORING_OP_OPENAT2', (IORING_OP_EPOLL_CTL:=29): 'IORING_OP_EPOLL_CTL', (IORING_OP_SPLICE:=30): 'IORING_OP_SPLICE', (IORING_OP_PROVIDE_BUFFERS:=31): 'IORING_OP_PROVIDE_BUFFERS', (IORING_OP_REMOVE_BUFFERS:=32): 'IORING_OP_REMOVE_BUFFERS', (IORING_OP_TEE:=33): 'IORING_OP_TEE', (IORING_OP_SHUTDOWN:=34): 'IORING_OP_SHUTDOWN', (IORING_OP_RENAMEAT:=35): 'IORING_OP_RENAMEAT', (IORING_OP_UNLINKAT:=36): 'IORING_OP_UNLINKAT', (IORING_OP_MKDIRAT:=37): 'IORING_OP_MKDIRAT', (IORING_OP_SYMLINKAT:=38): 'IORING_OP_SYMLINKAT', (IORING_OP_LINKAT:=39): 'IORING_OP_LINKAT', (IORING_OP_MSG_RING:=40): 'IORING_OP_MSG_RING', (IORING_OP_FSETXATTR:=41): 'IORING_OP_FSETXATTR', (IORING_OP_SETXATTR:=42): 'IORING_OP_SETXATTR', (IORING_OP_FGETXATTR:=43): 'IORING_OP_FGETXATTR', (IORING_OP_GETXATTR:=44): 'IORING_OP_GETXATTR', (IORING_OP_SOCKET:=45): 'IORING_OP_SOCKET', (IORING_OP_URING_CMD:=46): 'IORING_OP_URING_CMD', (IORING_OP_SEND_ZC:=47): 'IORING_OP_SEND_ZC', (IORING_OP_SENDMSG_ZC:=48): 'IORING_OP_SENDMSG_ZC', (IORING_OP_READ_MULTISHOT:=49): 'IORING_OP_READ_MULTISHOT', (IORING_OP_WAITID:=50): 'IORING_OP_WAITID', (IORING_OP_FUTEX_WAIT:=51): 'IORING_OP_FUTEX_WAIT', (IORING_OP_FUTEX_WAKE:=52): 'IORING_OP_FUTEX_WAKE', (IORING_OP_FUTEX_WAITV:=53): 'IORING_OP_FUTEX_WAITV', (IORING_OP_FIXED_FD_INSTALL:=54): 'IORING_OP_FIXED_FD_INSTALL', (IORING_OP_FTRUNCATE:=55): 'IORING_OP_FTRUNCATE', (IORING_OP_BIND:=56): 'IORING_OP_BIND', (IORING_OP_LISTEN:=57): 'IORING_OP_LISTEN', (IORING_OP_RECV_ZC:=58): 'IORING_OP_RECV_ZC', (IORING_OP_EPOLL_WAIT:=59): 'IORING_OP_EPOLL_WAIT', (IORING_OP_READV_FIXED:=60): 'IORING_OP_READV_FIXED', (IORING_OP_WRITEV_FIXED:=61): 'IORING_OP_WRITEV_FIXED', (IORING_OP_PIPE:=62): 'IORING_OP_PIPE', (IORING_OP_LAST:=63): 'IORING_OP_LAST'}
enum_io_uring_msg_ring_flags: dict[int, str] = {(IORING_MSG_DATA:=0): 'IORING_MSG_DATA', (IORING_MSG_SEND_FD:=1): 'IORING_MSG_SEND_FD'}
@c.record
class struct_io_sqring_offsets(c.Struct):
  SIZE = 40
  head: int
  tail: int
  ring_mask: int
  ring_entries: int
  flags: int
  dropped: int
  array: int
  resv1: int
  user_addr: int
struct_io_sqring_offsets.register_fields([('head', ctypes.c_uint32, 0), ('tail', ctypes.c_uint32, 4), ('ring_mask', ctypes.c_uint32, 8), ('ring_entries', ctypes.c_uint32, 12), ('flags', ctypes.c_uint32, 16), ('dropped', ctypes.c_uint32, 20), ('array', ctypes.c_uint32, 24), ('resv1', ctypes.c_uint32, 28), ('user_addr', ctypes.c_uint64, 32)])
@c.record
class struct_io_cqring_offsets(c.Struct):
  SIZE = 40
  head: int
  tail: int
  ring_mask: int
  ring_entries: int
  overflow: int
  cqes: int
  flags: int
  resv1: int
  user_addr: int
struct_io_cqring_offsets.register_fields([('head', ctypes.c_uint32, 0), ('tail', ctypes.c_uint32, 4), ('ring_mask', ctypes.c_uint32, 8), ('ring_entries', ctypes.c_uint32, 12), ('overflow', ctypes.c_uint32, 16), ('cqes', ctypes.c_uint32, 20), ('flags', ctypes.c_uint32, 24), ('resv1', ctypes.c_uint32, 28), ('user_addr', ctypes.c_uint64, 32)])
@c.record
class struct_io_uring_params(c.Struct):
  SIZE = 120
  sq_entries: int
  cq_entries: int
  flags: int
  sq_thread_cpu: int
  sq_thread_idle: int
  features: int
  wq_fd: int
  resv: c.Array[ctypes.c_uint32, Literal[3]]
  sq_off: struct_io_sqring_offsets
  cq_off: struct_io_cqring_offsets
struct_io_uring_params.register_fields([('sq_entries', ctypes.c_uint32, 0), ('cq_entries', ctypes.c_uint32, 4), ('flags', ctypes.c_uint32, 8), ('sq_thread_cpu', ctypes.c_uint32, 12), ('sq_thread_idle', ctypes.c_uint32, 16), ('features', ctypes.c_uint32, 20), ('wq_fd', ctypes.c_uint32, 24), ('resv', c.Array[ctypes.c_uint32, Literal[3]], 28), ('sq_off', struct_io_sqring_offsets, 40), ('cq_off', struct_io_cqring_offsets, 80)])
enum_io_uring_register_op: dict[int, str] = {(IORING_REGISTER_BUFFERS:=0): 'IORING_REGISTER_BUFFERS', (IORING_UNREGISTER_BUFFERS:=1): 'IORING_UNREGISTER_BUFFERS', (IORING_REGISTER_FILES:=2): 'IORING_REGISTER_FILES', (IORING_UNREGISTER_FILES:=3): 'IORING_UNREGISTER_FILES', (IORING_REGISTER_EVENTFD:=4): 'IORING_REGISTER_EVENTFD', (IORING_UNREGISTER_EVENTFD:=5): 'IORING_UNREGISTER_EVENTFD', (IORING_REGISTER_FILES_UPDATE:=6): 'IORING_REGISTER_FILES_UPDATE', (IORING_REGISTER_EVENTFD_ASYNC:=7): 'IORING_REGISTER_EVENTFD_ASYNC', (IORING_REGISTER_PROBE:=8): 'IORING_REGISTER_PROBE', (IORING_REGISTER_PERSONALITY:=9): 'IORING_REGISTER_PERSONALITY', (IORING_UNREGISTER_PERSONALITY:=10): 'IORING_UNREGISTER_PERSONALITY', (IORING_REGISTER_RESTRICTIONS:=11): 'IORING_REGISTER_RESTRICTIONS', (IORING_REGISTER_ENABLE_RINGS:=12): 'IORING_REGISTER_ENABLE_RINGS', (IORING_REGISTER_FILES2:=13): 'IORING_REGISTER_FILES2', (IORING_REGISTER_FILES_UPDATE2:=14): 'IORING_REGISTER_FILES_UPDATE2', (IORING_REGISTER_BUFFERS2:=15): 'IORING_REGISTER_BUFFERS2', (IORING_REGISTER_BUFFERS_UPDATE:=16): 'IORING_REGISTER_BUFFERS_UPDATE', (IORING_REGISTER_IOWQ_AFF:=17): 'IORING_REGISTER_IOWQ_AFF', (IORING_UNREGISTER_IOWQ_AFF:=18): 'IORING_UNREGISTER_IOWQ_AFF', (IORING_REGISTER_IOWQ_MAX_WORKERS:=19): 'IORING_REGISTER_IOWQ_MAX_WORKERS', (IORING_REGISTER_RING_FDS:=20): 'IORING_REGISTER_RING_FDS', (IORING_UNREGISTER_RING_FDS:=21): 'IORING_UNREGISTER_RING_FDS', (IORING_REGISTER_PBUF_RING:=22): 'IORING_REGISTER_PBUF_RING', (IORING_UNREGISTER_PBUF_RING:=23): 'IORING_UNREGISTER_PBUF_RING', (IORING_REGISTER_SYNC_CANCEL:=24): 'IORING_REGISTER_SYNC_CANCEL', (IORING_REGISTER_FILE_ALLOC_RANGE:=25): 'IORING_REGISTER_FILE_ALLOC_RANGE', (IORING_REGISTER_PBUF_STATUS:=26): 'IORING_REGISTER_PBUF_STATUS', (IORING_REGISTER_NAPI:=27): 'IORING_REGISTER_NAPI', (IORING_UNREGISTER_NAPI:=28): 'IORING_UNREGISTER_NAPI', (IORING_REGISTER_CLOCK:=29): 'IORING_REGISTER_CLOCK', (IORING_REGISTER_CLONE_BUFFERS:=30): 'IORING_REGISTER_CLONE_BUFFERS', (IORING_REGISTER_SEND_MSG_RING:=31): 'IORING_REGISTER_SEND_MSG_RING', (IORING_REGISTER_ZCRX_IFQ:=32): 'IORING_REGISTER_ZCRX_IFQ', (IORING_REGISTER_RESIZE_RINGS:=33): 'IORING_REGISTER_RESIZE_RINGS', (IORING_REGISTER_MEM_REGION:=34): 'IORING_REGISTER_MEM_REGION', (IORING_REGISTER_QUERY:=35): 'IORING_REGISTER_QUERY', (IORING_REGISTER_LAST:=36): 'IORING_REGISTER_LAST', (IORING_REGISTER_USE_REGISTERED_RING:=2147483648): 'IORING_REGISTER_USE_REGISTERED_RING'}
enum_io_wq_type: dict[int, str] = {(IO_WQ_BOUND:=0): 'IO_WQ_BOUND', (IO_WQ_UNBOUND:=1): 'IO_WQ_UNBOUND'}
@c.record
class struct_io_uring_files_update(c.Struct):
  SIZE = 16
  offset: int
  resv: int
  fds: int
struct_io_uring_files_update.register_fields([('offset', ctypes.c_uint32, 0), ('resv', ctypes.c_uint32, 4), ('fds', ctypes.c_uint64, 8)])
_anonenum0: dict[int, str] = {(IORING_MEM_REGION_TYPE_USER:=1): 'IORING_MEM_REGION_TYPE_USER'}
@c.record
class struct_io_uring_region_desc(c.Struct):
  SIZE = 64
  user_addr: int
  size: int
  flags: int
  id: int
  mmap_offset: int
  __resv: c.Array[ctypes.c_uint64, Literal[4]]
struct_io_uring_region_desc.register_fields([('user_addr', ctypes.c_uint64, 0), ('size', ctypes.c_uint64, 8), ('flags', ctypes.c_uint32, 16), ('id', ctypes.c_uint32, 20), ('mmap_offset', ctypes.c_uint64, 24), ('__resv', c.Array[ctypes.c_uint64, Literal[4]], 32)])
_anonenum1: dict[int, str] = {(IORING_MEM_REGION_REG_WAIT_ARG:=1): 'IORING_MEM_REGION_REG_WAIT_ARG'}
@c.record
class struct_io_uring_mem_region_reg(c.Struct):
  SIZE = 32
  region_uptr: int
  flags: int
  __resv: c.Array[ctypes.c_uint64, Literal[2]]
struct_io_uring_mem_region_reg.register_fields([('region_uptr', ctypes.c_uint64, 0), ('flags', ctypes.c_uint64, 8), ('__resv', c.Array[ctypes.c_uint64, Literal[2]], 16)])
@c.record
class struct_io_uring_rsrc_register(c.Struct):
  SIZE = 32
  nr: int
  flags: int
  resv2: int
  data: int
  tags: int
struct_io_uring_rsrc_register.register_fields([('nr', ctypes.c_uint32, 0), ('flags', ctypes.c_uint32, 4), ('resv2', ctypes.c_uint64, 8), ('data', ctypes.c_uint64, 16), ('tags', ctypes.c_uint64, 24)])
@c.record
class struct_io_uring_rsrc_update(c.Struct):
  SIZE = 16
  offset: int
  resv: int
  data: int
struct_io_uring_rsrc_update.register_fields([('offset', ctypes.c_uint32, 0), ('resv', ctypes.c_uint32, 4), ('data', ctypes.c_uint64, 8)])
@c.record
class struct_io_uring_rsrc_update2(c.Struct):
  SIZE = 32
  offset: int
  resv: int
  data: int
  tags: int
  nr: int
  resv2: int
struct_io_uring_rsrc_update2.register_fields([('offset', ctypes.c_uint32, 0), ('resv', ctypes.c_uint32, 4), ('data', ctypes.c_uint64, 8), ('tags', ctypes.c_uint64, 16), ('nr', ctypes.c_uint32, 24), ('resv2', ctypes.c_uint32, 28)])
@c.record
class struct_io_uring_probe_op(c.Struct):
  SIZE = 8
  op: int
  resv: int
  flags: int
  resv2: int
struct_io_uring_probe_op.register_fields([('op', ctypes.c_ubyte, 0), ('resv', ctypes.c_ubyte, 1), ('flags', ctypes.c_uint16, 2), ('resv2', ctypes.c_uint32, 4)])
@c.record
class struct_io_uring_probe(c.Struct):
  SIZE = 16
  last_op: int
  ops_len: int
  resv: int
  resv2: c.Array[ctypes.c_uint32, Literal[3]]
  ops: c.Array[struct_io_uring_probe_op, Literal[0]]
struct_io_uring_probe.register_fields([('last_op', ctypes.c_ubyte, 0), ('ops_len', ctypes.c_ubyte, 1), ('resv', ctypes.c_uint16, 2), ('resv2', c.Array[ctypes.c_uint32, Literal[3]], 4), ('ops', c.Array[struct_io_uring_probe_op, Literal[0]], 16)])
@c.record
class struct_io_uring_restriction(c.Struct):
  SIZE = 16
  opcode: int
  register_op: int
  sqe_op: int
  sqe_flags: int
  resv: int
  resv2: c.Array[ctypes.c_uint32, Literal[3]]
struct_io_uring_restriction.register_fields([('opcode', ctypes.c_uint16, 0), ('register_op', ctypes.c_ubyte, 2), ('sqe_op', ctypes.c_ubyte, 2), ('sqe_flags', ctypes.c_ubyte, 2), ('resv', ctypes.c_ubyte, 3), ('resv2', c.Array[ctypes.c_uint32, Literal[3]], 4)])
@c.record
class struct_io_uring_clock_register(c.Struct):
  SIZE = 16
  clockid: int
  __resv: c.Array[ctypes.c_uint32, Literal[3]]
struct_io_uring_clock_register.register_fields([('clockid', ctypes.c_uint32, 0), ('__resv', c.Array[ctypes.c_uint32, Literal[3]], 4)])
_anonenum2: dict[int, str] = {(IORING_REGISTER_SRC_REGISTERED:=1): 'IORING_REGISTER_SRC_REGISTERED', (IORING_REGISTER_DST_REPLACE:=2): 'IORING_REGISTER_DST_REPLACE'}
@c.record
class struct_io_uring_clone_buffers(c.Struct):
  SIZE = 32
  src_fd: int
  flags: int
  src_off: int
  dst_off: int
  nr: int
  pad: c.Array[ctypes.c_uint32, Literal[3]]
struct_io_uring_clone_buffers.register_fields([('src_fd', ctypes.c_uint32, 0), ('flags', ctypes.c_uint32, 4), ('src_off', ctypes.c_uint32, 8), ('dst_off', ctypes.c_uint32, 12), ('nr', ctypes.c_uint32, 16), ('pad', c.Array[ctypes.c_uint32, Literal[3]], 20)])
@c.record
class struct_io_uring_buf(c.Struct):
  SIZE = 16
  addr: int
  len: int
  bid: int
  resv: int
struct_io_uring_buf.register_fields([('addr', ctypes.c_uint64, 0), ('len', ctypes.c_uint32, 8), ('bid', ctypes.c_uint16, 12), ('resv', ctypes.c_uint16, 14)])
@c.record
class struct_io_uring_buf_ring(c.Struct):
  SIZE = 16
  resv1: int
  resv2: int
  resv3: int
  tail: int
  __empty_bufs: struct_io_uring_buf_ring___empty_bufs
  bufs: c.Array[struct_io_uring_buf, Literal[0]]
class struct_io_uring_buf_ring___empty_bufs(c.Struct): pass
struct_io_uring_buf_ring.register_fields([('resv1', ctypes.c_uint64, 0), ('resv2', ctypes.c_uint32, 8), ('resv3', ctypes.c_uint16, 12), ('tail', ctypes.c_uint16, 14), ('__empty_bufs', struct_io_uring_buf_ring___empty_bufs, 0), ('bufs', c.Array[struct_io_uring_buf, Literal[0]], 0)])
enum_io_uring_register_pbuf_ring_flags: dict[int, str] = {(IOU_PBUF_RING_MMAP:=1): 'IOU_PBUF_RING_MMAP', (IOU_PBUF_RING_INC:=2): 'IOU_PBUF_RING_INC'}
@c.record
class struct_io_uring_buf_reg(c.Struct):
  SIZE = 40
  ring_addr: int
  ring_entries: int
  bgid: int
  flags: int
  resv: c.Array[ctypes.c_uint64, Literal[3]]
struct_io_uring_buf_reg.register_fields([('ring_addr', ctypes.c_uint64, 0), ('ring_entries', ctypes.c_uint32, 8), ('bgid', ctypes.c_uint16, 12), ('flags', ctypes.c_uint16, 14), ('resv', c.Array[ctypes.c_uint64, Literal[3]], 16)])
@c.record
class struct_io_uring_buf_status(c.Struct):
  SIZE = 40
  buf_group: int
  head: int
  resv: c.Array[ctypes.c_uint32, Literal[8]]
struct_io_uring_buf_status.register_fields([('buf_group', ctypes.c_uint32, 0), ('head', ctypes.c_uint32, 4), ('resv', c.Array[ctypes.c_uint32, Literal[8]], 8)])
enum_io_uring_napi_op: dict[int, str] = {(IO_URING_NAPI_REGISTER_OP:=0): 'IO_URING_NAPI_REGISTER_OP', (IO_URING_NAPI_STATIC_ADD_ID:=1): 'IO_URING_NAPI_STATIC_ADD_ID', (IO_URING_NAPI_STATIC_DEL_ID:=2): 'IO_URING_NAPI_STATIC_DEL_ID'}
enum_io_uring_napi_tracking_strategy: dict[int, str] = {(IO_URING_NAPI_TRACKING_DYNAMIC:=0): 'IO_URING_NAPI_TRACKING_DYNAMIC', (IO_URING_NAPI_TRACKING_STATIC:=1): 'IO_URING_NAPI_TRACKING_STATIC', (IO_URING_NAPI_TRACKING_INACTIVE:=255): 'IO_URING_NAPI_TRACKING_INACTIVE'}
@c.record
class struct_io_uring_napi(c.Struct):
  SIZE = 16
  busy_poll_to: int
  prefer_busy_poll: int
  opcode: int
  pad: c.Array[ctypes.c_ubyte, Literal[2]]
  op_param: int
  resv: int
struct_io_uring_napi.register_fields([('busy_poll_to', ctypes.c_uint32, 0), ('prefer_busy_poll', ctypes.c_ubyte, 4), ('opcode', ctypes.c_ubyte, 5), ('pad', c.Array[ctypes.c_ubyte, Literal[2]], 6), ('op_param', ctypes.c_uint32, 8), ('resv', ctypes.c_uint32, 12)])
enum_io_uring_register_restriction_op: dict[int, str] = {(IORING_RESTRICTION_REGISTER_OP:=0): 'IORING_RESTRICTION_REGISTER_OP', (IORING_RESTRICTION_SQE_OP:=1): 'IORING_RESTRICTION_SQE_OP', (IORING_RESTRICTION_SQE_FLAGS_ALLOWED:=2): 'IORING_RESTRICTION_SQE_FLAGS_ALLOWED', (IORING_RESTRICTION_SQE_FLAGS_REQUIRED:=3): 'IORING_RESTRICTION_SQE_FLAGS_REQUIRED', (IORING_RESTRICTION_LAST:=4): 'IORING_RESTRICTION_LAST'}
_anonenum3: dict[int, str] = {(IORING_REG_WAIT_TS:=1): 'IORING_REG_WAIT_TS'}
@c.record
class struct_io_uring_reg_wait(c.Struct):
  SIZE = 64
  ts: struct___kernel_timespec
  min_wait_usec: int
  flags: int
  sigmask: int
  sigmask_sz: int
  pad: c.Array[ctypes.c_uint32, Literal[3]]
  pad2: c.Array[ctypes.c_uint64, Literal[2]]
@c.record
class struct___kernel_timespec(c.Struct):
  SIZE = 16
  tv_sec: int
  tv_nsec: int
__kernel_time64_t: TypeAlias = ctypes.c_int64
struct___kernel_timespec.register_fields([('tv_sec', ctypes.c_int64, 0), ('tv_nsec', ctypes.c_int64, 8)])
struct_io_uring_reg_wait.register_fields([('ts', struct___kernel_timespec, 0), ('min_wait_usec', ctypes.c_uint32, 16), ('flags', ctypes.c_uint32, 20), ('sigmask', ctypes.c_uint64, 24), ('sigmask_sz', ctypes.c_uint32, 32), ('pad', c.Array[ctypes.c_uint32, Literal[3]], 36), ('pad2', c.Array[ctypes.c_uint64, Literal[2]], 48)])
@c.record
class struct_io_uring_getevents_arg(c.Struct):
  SIZE = 24
  sigmask: int
  sigmask_sz: int
  min_wait_usec: int
  ts: int
struct_io_uring_getevents_arg.register_fields([('sigmask', ctypes.c_uint64, 0), ('sigmask_sz', ctypes.c_uint32, 8), ('min_wait_usec', ctypes.c_uint32, 12), ('ts', ctypes.c_uint64, 16)])
@c.record
class struct_io_uring_sync_cancel_reg(c.Struct):
  SIZE = 64
  addr: int
  fd: int
  flags: int
  timeout: struct___kernel_timespec
  opcode: int
  pad: c.Array[ctypes.c_ubyte, Literal[7]]
  pad2: c.Array[ctypes.c_uint64, Literal[3]]
struct_io_uring_sync_cancel_reg.register_fields([('addr', ctypes.c_uint64, 0), ('fd', ctypes.c_int32, 8), ('flags', ctypes.c_uint32, 12), ('timeout', struct___kernel_timespec, 16), ('opcode', ctypes.c_ubyte, 32), ('pad', c.Array[ctypes.c_ubyte, Literal[7]], 33), ('pad2', c.Array[ctypes.c_uint64, Literal[3]], 40)])
@c.record
class struct_io_uring_file_index_range(c.Struct):
  SIZE = 16
  off: int
  len: int
  resv: int
struct_io_uring_file_index_range.register_fields([('off', ctypes.c_uint32, 0), ('len', ctypes.c_uint32, 4), ('resv', ctypes.c_uint64, 8)])
@c.record
class struct_io_uring_recvmsg_out(c.Struct):
  SIZE = 16
  namelen: int
  controllen: int
  payloadlen: int
  flags: int
struct_io_uring_recvmsg_out.register_fields([('namelen', ctypes.c_uint32, 0), ('controllen', ctypes.c_uint32, 4), ('payloadlen', ctypes.c_uint32, 8), ('flags', ctypes.c_uint32, 12)])
enum_io_uring_socket_op: dict[int, str] = {(SOCKET_URING_OP_SIOCINQ:=0): 'SOCKET_URING_OP_SIOCINQ', (SOCKET_URING_OP_SIOCOUTQ:=1): 'SOCKET_URING_OP_SIOCOUTQ', (SOCKET_URING_OP_GETSOCKOPT:=2): 'SOCKET_URING_OP_GETSOCKOPT', (SOCKET_URING_OP_SETSOCKOPT:=3): 'SOCKET_URING_OP_SETSOCKOPT', (SOCKET_URING_OP_TX_TIMESTAMP:=4): 'SOCKET_URING_OP_TX_TIMESTAMP'}
@c.record
class struct_io_timespec(c.Struct):
  SIZE = 16
  tv_sec: int
  tv_nsec: int
struct_io_timespec.register_fields([('tv_sec', ctypes.c_uint64, 0), ('tv_nsec', ctypes.c_uint64, 8)])
struct_io_uring_zcrx_rqe.register_fields([('off', ctypes.c_uint64, 0), ('len', ctypes.c_uint32, 8), ('__pad', ctypes.c_uint32, 12)])
@c.record
class struct_io_uring_zcrx_cqe(c.Struct):
  SIZE = 16
  off: int
  __pad: int
struct_io_uring_zcrx_cqe.register_fields([('off', ctypes.c_uint64, 0), ('__pad', ctypes.c_uint64, 8)])
@c.record
class struct_io_uring_zcrx_offsets(c.Struct):
  SIZE = 32
  head: int
  tail: int
  rqes: int
  __resv2: int
  __resv: c.Array[ctypes.c_uint64, Literal[2]]
struct_io_uring_zcrx_offsets.register_fields([('head', ctypes.c_uint32, 0), ('tail', ctypes.c_uint32, 4), ('rqes', ctypes.c_uint32, 8), ('__resv2', ctypes.c_uint32, 12), ('__resv', c.Array[ctypes.c_uint64, Literal[2]], 16)])
enum_io_uring_zcrx_area_flags: dict[int, str] = {(IORING_ZCRX_AREA_DMABUF:=1): 'IORING_ZCRX_AREA_DMABUF'}
@c.record
class struct_io_uring_zcrx_area_reg(c.Struct):
  SIZE = 48
  addr: int
  len: int
  rq_area_token: int
  flags: int
  dmabuf_fd: int
  __resv2: c.Array[ctypes.c_uint64, Literal[2]]
struct_io_uring_zcrx_area_reg.register_fields([('addr', ctypes.c_uint64, 0), ('len', ctypes.c_uint64, 8), ('rq_area_token', ctypes.c_uint64, 16), ('flags', ctypes.c_uint32, 24), ('dmabuf_fd', ctypes.c_uint32, 28), ('__resv2', c.Array[ctypes.c_uint64, Literal[2]], 32)])
@c.record
class struct_io_uring_zcrx_ifq_reg(c.Struct):
  SIZE = 96
  if_idx: int
  if_rxq: int
  rq_entries: int
  flags: int
  area_ptr: int
  region_ptr: int
  offsets: struct_io_uring_zcrx_offsets
  zcrx_id: int
  __resv2: int
  __resv: c.Array[ctypes.c_uint64, Literal[3]]
struct_io_uring_zcrx_ifq_reg.register_fields([('if_idx', ctypes.c_uint32, 0), ('if_rxq', ctypes.c_uint32, 4), ('rq_entries', ctypes.c_uint32, 8), ('flags', ctypes.c_uint32, 12), ('area_ptr', ctypes.c_uint64, 16), ('region_ptr', ctypes.c_uint64, 24), ('offsets', struct_io_uring_zcrx_offsets, 32), ('zcrx_id', ctypes.c_uint32, 64), ('__resv2', ctypes.c_uint32, 68), ('__resv', c.Array[ctypes.c_uint64, Literal[3]], 72)])
uring_unlikely = lambda cond: __builtin_expect( not  not (cond), 0) # type: ignore
uring_likely = lambda cond: __builtin_expect( not  not (cond), 1) # type: ignore
NR_io_uring_setup = 425
NR_io_uring_enter = 426
NR_io_uring_register = 427
IO_URING_CHECK_VERSION = lambda major,minor: (major > IO_URING_VERSION_MAJOR or (major == IO_URING_VERSION_MAJOR and minor > IO_URING_VERSION_MINOR)) # type: ignore
IORING_RW_ATTR_FLAG_PI = (1 << 0)
IORING_FILE_INDEX_ALLOC = (~0)
IOSQE_FIXED_FILE = (1 << IOSQE_FIXED_FILE_BIT)
IOSQE_IO_DRAIN = (1 << IOSQE_IO_DRAIN_BIT)
IOSQE_IO_LINK = (1 << IOSQE_IO_LINK_BIT)
IOSQE_IO_HARDLINK = (1 << IOSQE_IO_HARDLINK_BIT)
IOSQE_ASYNC = (1 << IOSQE_ASYNC_BIT)
IOSQE_BUFFER_SELECT = (1 << IOSQE_BUFFER_SELECT_BIT)
IOSQE_CQE_SKIP_SUCCESS = (1 << IOSQE_CQE_SKIP_SUCCESS_BIT)
IORING_SETUP_IOPOLL = (1 << 0)
IORING_SETUP_SQPOLL = (1 << 1)
IORING_SETUP_SQ_AFF = (1 << 2)
IORING_SETUP_CQSIZE = (1 << 3)
IORING_SETUP_CLAMP = (1 << 4)
IORING_SETUP_ATTACH_WQ = (1 << 5)
IORING_SETUP_R_DISABLED = (1 << 6)
IORING_SETUP_SUBMIT_ALL = (1 << 7)
IORING_SETUP_COOP_TASKRUN = (1 << 8)
IORING_SETUP_TASKRUN_FLAG = (1 << 9)
IORING_SETUP_SQE128 = (1 << 10)
IORING_SETUP_CQE32 = (1 << 11)
IORING_SETUP_SINGLE_ISSUER = (1 << 12)
IORING_SETUP_DEFER_TASKRUN = (1 << 13)
IORING_SETUP_NO_MMAP = (1 << 14)
IORING_SETUP_REGISTERED_FD_ONLY = (1 << 15)
IORING_SETUP_NO_SQARRAY = (1 << 16)
IORING_SETUP_HYBRID_IOPOLL = (1 << 17)
IORING_SETUP_CQE_MIXED = (1 << 18)
IORING_URING_CMD_FIXED = (1 << 0)
IORING_URING_CMD_MULTISHOT = (1 << 1)
IORING_URING_CMD_MASK = (IORING_URING_CMD_FIXED | IORING_URING_CMD_MULTISHOT)
IORING_FSYNC_DATASYNC = (1 << 0)
IORING_TIMEOUT_ABS = (1 << 0)
IORING_TIMEOUT_UPDATE = (1 << 1)
IORING_TIMEOUT_BOOTTIME = (1 << 2)
IORING_TIMEOUT_REALTIME = (1 << 3)
IORING_LINK_TIMEOUT_UPDATE = (1 << 4)
IORING_TIMEOUT_ETIME_SUCCESS = (1 << 5)
IORING_TIMEOUT_MULTISHOT = (1 << 6)
IORING_TIMEOUT_CLOCK_MASK = (IORING_TIMEOUT_BOOTTIME | IORING_TIMEOUT_REALTIME)
IORING_TIMEOUT_UPDATE_MASK = (IORING_TIMEOUT_UPDATE | IORING_LINK_TIMEOUT_UPDATE)
SPLICE_F_FD_IN_FIXED = (1 << 31)
IORING_POLL_ADD_MULTI = (1 << 0)
IORING_POLL_UPDATE_EVENTS = (1 << 1)
IORING_POLL_UPDATE_USER_DATA = (1 << 2)
IORING_POLL_ADD_LEVEL = (1 << 3)
IORING_ASYNC_CANCEL_ALL = (1 << 0)
IORING_ASYNC_CANCEL_FD = (1 << 1)
IORING_ASYNC_CANCEL_ANY = (1 << 2)
IORING_ASYNC_CANCEL_FD_FIXED = (1 << 3)
IORING_ASYNC_CANCEL_USERDATA = (1 << 4)
IORING_ASYNC_CANCEL_OP = (1 << 5)
IORING_RECVSEND_POLL_FIRST = (1 << 0)
IORING_RECV_MULTISHOT = (1 << 1)
IORING_RECVSEND_FIXED_BUF = (1 << 2)
IORING_SEND_ZC_REPORT_USAGE = (1 << 3)
IORING_RECVSEND_BUNDLE = (1 << 4)
IORING_SEND_VECTORIZED = (1 << 5)
IORING_NOTIF_USAGE_ZC_COPIED = (1 << 31)
IORING_ACCEPT_MULTISHOT = (1 << 0)
IORING_ACCEPT_DONTWAIT = (1 << 1)
IORING_ACCEPT_POLL_FIRST = (1 << 2)
IORING_MSG_RING_CQE_SKIP = (1 << 0)
IORING_MSG_RING_FLAGS_PASS = (1 << 1)
IORING_FIXED_FD_NO_CLOEXEC = (1 << 0)
IORING_NOP_INJECT_RESULT = (1 << 0)
IORING_NOP_FILE = (1 << 1)
IORING_NOP_FIXED_FILE = (1 << 2)
IORING_NOP_FIXED_BUFFER = (1 << 3)
IORING_NOP_TW = (1 << 4)
IORING_NOP_CQE32 = (1 << 5)
IORING_CQE_F_BUFFER = (1 << 0)
IORING_CQE_F_MORE = (1 << 1)
IORING_CQE_F_SOCK_NONEMPTY = (1 << 2)
IORING_CQE_F_NOTIF = (1 << 3)
IORING_CQE_F_BUF_MORE = (1 << 4)
IORING_CQE_F_SKIP = (1 << 5)
IORING_CQE_F_32 = (1 << 15)
IORING_CQE_BUFFER_SHIFT = 16
IORING_OFF_SQ_RING = 0
IORING_OFF_CQ_RING = 0x8000000
IORING_OFF_SQES = 0x10000000
IORING_OFF_PBUF_RING = 0x80000000
IORING_OFF_PBUF_SHIFT = 16
IORING_OFF_MMAP_MASK = 0xf8000000
IORING_SQ_NEED_WAKEUP = (1 << 0)
IORING_SQ_CQ_OVERFLOW = (1 << 1)
IORING_SQ_TASKRUN = (1 << 2)
IORING_CQ_EVENTFD_DISABLED = (1 << 0)
IORING_ENTER_GETEVENTS = (1 << 0)
IORING_ENTER_SQ_WAKEUP = (1 << 1)
IORING_ENTER_SQ_WAIT = (1 << 2)
IORING_ENTER_EXT_ARG = (1 << 3)
IORING_ENTER_REGISTERED_RING = (1 << 4)
IORING_ENTER_ABS_TIMER = (1 << 5)
IORING_ENTER_EXT_ARG_REG = (1 << 6)
IORING_ENTER_NO_IOWAIT = (1 << 7)
IORING_FEAT_SINGLE_MMAP = (1 << 0)
IORING_FEAT_NODROP = (1 << 1)
IORING_FEAT_SUBMIT_STABLE = (1 << 2)
IORING_FEAT_RW_CUR_POS = (1 << 3)
IORING_FEAT_CUR_PERSONALITY = (1 << 4)
IORING_FEAT_FAST_POLL = (1 << 5)
IORING_FEAT_POLL_32BITS = (1 << 6)
IORING_FEAT_SQPOLL_NONFIXED = (1 << 7)
IORING_FEAT_EXT_ARG = (1 << 8)
IORING_FEAT_NATIVE_WORKERS = (1 << 9)
IORING_FEAT_RSRC_TAGS = (1 << 10)
IORING_FEAT_CQE_SKIP = (1 << 11)
IORING_FEAT_LINKED_FILE = (1 << 12)
IORING_FEAT_REG_REG_RING = (1 << 13)
IORING_FEAT_RECVSEND_BUNDLE = (1 << 14)
IORING_FEAT_MIN_TIMEOUT = (1 << 15)
IORING_FEAT_RW_ATTR = (1 << 16)
IORING_FEAT_NO_IOWAIT = (1 << 17)
IORING_RSRC_REGISTER_SPARSE = (1 << 0)
IORING_REGISTER_FILES_SKIP = (-2)
IO_URING_OP_SUPPORTED = (1 << 0)
IORING_TIMESTAMP_HW_SHIFT = 16
IORING_TIMESTAMP_TYPE_SHIFT = (IORING_TIMESTAMP_HW_SHIFT + 1)
IORING_ZCRX_AREA_SHIFT = 48
__SC_3264 = lambda _nr,_32,_64: __SYSCALL(_nr, _64) # type: ignore
__SC_COMP = lambda _nr,_sys,_comp: __SYSCALL(_nr, _sys) # type: ignore
__SC_COMP_3264 = lambda _nr,_32,_64,_comp: __SC_3264(_nr, _32, _64) # type: ignore
NR_io_setup = 0
NR_io_destroy = 1
NR_io_submit = 2
NR_io_cancel = 3
NR_io_getevents = 4
NR_setxattr = 5
NR_lsetxattr = 6
NR_fsetxattr = 7
NR_getxattr = 8
NR_lgetxattr = 9
NR_fgetxattr = 10
NR_listxattr = 11
NR_llistxattr = 12
NR_flistxattr = 13
NR_removexattr = 14
NR_lremovexattr = 15
NR_fremovexattr = 16
NR_getcwd = 17
NR_lookup_dcookie = 18
NR_eventfd2 = 19
NR_epoll_create1 = 20
NR_epoll_ctl = 21
NR_epoll_pwait = 22
NR_dup = 23
NR_dup3 = 24
NR3264_fcntl = 25
NR_inotify_init1 = 26
NR_inotify_add_watch = 27
NR_inotify_rm_watch = 28
NR_ioctl = 29
NR_ioprio_set = 30
NR_ioprio_get = 31
NR_flock = 32
NR_mknodat = 33
NR_mkdirat = 34
NR_unlinkat = 35
NR_symlinkat = 36
NR_linkat = 37
NR_umount2 = 39
NR_mount = 40
NR_pivot_root = 41
NR_nfsservctl = 42
NR3264_statfs = 43
NR3264_fstatfs = 44
NR3264_truncate = 45
NR3264_ftruncate = 46
NR_fallocate = 47
NR_faccessat = 48
NR_chdir = 49
NR_fchdir = 50
NR_chroot = 51
NR_fchmod = 52
NR_fchmodat = 53
NR_fchownat = 54
NR_fchown = 55
NR_openat = 56
NR_close = 57
NR_vhangup = 58
NR_pipe2 = 59
NR_quotactl = 60
NR_getdents64 = 61
NR3264_lseek = 62
NR_read = 63
NR_write = 64
NR_readv = 65
NR_writev = 66
NR_pread64 = 67
NR_pwrite64 = 68
NR_preadv = 69
NR_pwritev = 70
NR3264_sendfile = 71
NR_pselect6 = 72
NR_ppoll = 73
NR_signalfd4 = 74
NR_vmsplice = 75
NR_splice = 76
NR_tee = 77
NR_readlinkat = 78
NR_sync = 81
NR_fsync = 82
NR_fdatasync = 83
NR_sync_file_range = 84
NR_timerfd_create = 85
NR_timerfd_settime = 86
NR_timerfd_gettime = 87
NR_utimensat = 88
NR_acct = 89
NR_capget = 90
NR_capset = 91
NR_personality = 92
NR_exit = 93
NR_exit_group = 94
NR_waitid = 95
NR_set_tid_address = 96
NR_unshare = 97
NR_futex = 98
NR_set_robust_list = 99
NR_get_robust_list = 100
NR_nanosleep = 101
NR_getitimer = 102
NR_setitimer = 103
NR_kexec_load = 104
NR_init_module = 105
NR_delete_module = 106
NR_timer_create = 107
NR_timer_gettime = 108
NR_timer_getoverrun = 109
NR_timer_settime = 110
NR_timer_delete = 111
NR_clock_settime = 112
NR_clock_gettime = 113
NR_clock_getres = 114
NR_clock_nanosleep = 115
NR_syslog = 116
NR_ptrace = 117
NR_sched_setparam = 118
NR_sched_setscheduler = 119
NR_sched_getscheduler = 120
NR_sched_getparam = 121
NR_sched_setaffinity = 122
NR_sched_getaffinity = 123
NR_sched_yield = 124
NR_sched_get_priority_max = 125
NR_sched_get_priority_min = 126
NR_sched_rr_get_interval = 127
NR_restart_syscall = 128
NR_kill = 129
NR_tkill = 130
NR_tgkill = 131
NR_sigaltstack = 132
NR_rt_sigsuspend = 133
NR_rt_sigaction = 134
NR_rt_sigprocmask = 135
NR_rt_sigpending = 136
NR_rt_sigtimedwait = 137
NR_rt_sigqueueinfo = 138
NR_rt_sigreturn = 139
NR_setpriority = 140
NR_getpriority = 141
NR_reboot = 142
NR_setregid = 143
NR_setgid = 144
NR_setreuid = 145
NR_setuid = 146
NR_setresuid = 147
NR_getresuid = 148
NR_setresgid = 149
NR_getresgid = 150
NR_setfsuid = 151
NR_setfsgid = 152
NR_times = 153
NR_setpgid = 154
NR_getpgid = 155
NR_getsid = 156
NR_setsid = 157
NR_getgroups = 158
NR_setgroups = 159
NR_uname = 160
NR_sethostname = 161
NR_setdomainname = 162
NR_getrusage = 165
NR_umask = 166
NR_prctl = 167
NR_getcpu = 168
NR_gettimeofday = 169
NR_settimeofday = 170
NR_adjtimex = 171
NR_getpid = 172
NR_getppid = 173
NR_getuid = 174
NR_geteuid = 175
NR_getgid = 176
NR_getegid = 177
NR_gettid = 178
NR_sysinfo = 179
NR_mq_open = 180
NR_mq_unlink = 181
NR_mq_timedsend = 182
NR_mq_timedreceive = 183
NR_mq_notify = 184
NR_mq_getsetattr = 185
NR_msgget = 186
NR_msgctl = 187
NR_msgrcv = 188
NR_msgsnd = 189
NR_semget = 190
NR_semctl = 191
NR_semtimedop = 192
NR_semop = 193
NR_shmget = 194
NR_shmctl = 195
NR_shmat = 196
NR_shmdt = 197
NR_socket = 198
NR_socketpair = 199
NR_bind = 200
NR_listen = 201
NR_accept = 202
NR_connect = 203
NR_getsockname = 204
NR_getpeername = 205
NR_sendto = 206
NR_recvfrom = 207
NR_setsockopt = 208
NR_getsockopt = 209
NR_shutdown = 210
NR_sendmsg = 211
NR_recvmsg = 212
NR_readahead = 213
NR_brk = 214
NR_munmap = 215
NR_mremap = 216
NR_add_key = 217
NR_request_key = 218
NR_keyctl = 219
NR_clone = 220
NR_execve = 221
NR3264_mmap = 222
NR3264_fadvise64 = 223
NR_swapon = 224
NR_swapoff = 225
NR_mprotect = 226
NR_msync = 227
NR_mlock = 228
NR_munlock = 229
NR_mlockall = 230
NR_munlockall = 231
NR_mincore = 232
NR_madvise = 233
NR_remap_file_pages = 234
NR_mbind = 235
NR_get_mempolicy = 236
NR_set_mempolicy = 237
NR_migrate_pages = 238
NR_move_pages = 239
NR_rt_tgsigqueueinfo = 240
NR_perf_event_open = 241
NR_accept4 = 242
NR_recvmmsg = 243
NR_arch_specific_syscall = 244
NR_wait4 = 260
NR_prlimit64 = 261
NR_fanotify_init = 262
NR_fanotify_mark = 263
NR_name_to_handle_at = 264
NR_open_by_handle_at = 265
NR_clock_adjtime = 266
NR_syncfs = 267
NR_setns = 268
NR_sendmmsg = 269
NR_process_vm_readv = 270
NR_process_vm_writev = 271
NR_kcmp = 272
NR_finit_module = 273
NR_sched_setattr = 274
NR_sched_getattr = 275
NR_renameat2 = 276
NR_seccomp = 277
NR_getrandom = 278
NR_memfd_create = 279
NR_bpf = 280
NR_execveat = 281
NR_userfaultfd = 282
NR_membarrier = 283
NR_mlock2 = 284
NR_copy_file_range = 285
NR_preadv2 = 286
NR_pwritev2 = 287
NR_pkey_mprotect = 288
NR_pkey_alloc = 289
NR_pkey_free = 290
NR_statx = 291
NR_io_pgetevents = 292
NR_rseq = 293
NR_kexec_file_load = 294
NR_pidfd_send_signal = 424
NR_io_uring_setup = 425
NR_io_uring_enter = 426
NR_io_uring_register = 427
NR_open_tree = 428
NR_move_mount = 429
NR_fsopen = 430
NR_fsconfig = 431
NR_fsmount = 432
NR_fspick = 433
NR_pidfd_open = 434
NR_clone3 = 435
NR_close_range = 436
NR_openat2 = 437
NR_pidfd_getfd = 438
NR_faccessat2 = 439
NR_process_madvise = 440
NR_epoll_pwait2 = 441
NR_mount_setattr = 442
NR_quotactl_fd = 443
NR_landlock_create_ruleset = 444
NR_landlock_add_rule = 445
NR_landlock_restrict_self = 446
NR_process_mrelease = 448
NR_futex_waitv = 449
NR_set_mempolicy_home_node = 450
NR_cachestat = 451
NR_fchmodat2 = 452
NR_map_shadow_stack = 453
NR_futex_wake = 454
NR_futex_wait = 455
NR_futex_requeue = 456
NR_statmount = 457
NR_listmount = 458
NR_lsm_get_self_attr = 459
NR_lsm_set_self_attr = 460
NR_lsm_list_modules = 461
NR_mseal = 462
NR_setxattrat = 463
NR_getxattrat = 464
NR_listxattrat = 465
NR_removexattrat = 466
NR_open_tree_attr = 467
NR_file_getattr = 468
NR_file_setattr = 469
NR_syscalls = 470
NR_fcntl = NR3264_fcntl
NR_statfs = NR3264_statfs
NR_fstatfs = NR3264_fstatfs
NR_truncate = NR3264_truncate
NR_ftruncate = NR3264_ftruncate
NR_lseek = NR3264_lseek
NR_sendfile = NR3264_sendfile
NR_mmap = NR3264_mmap
NR_fadvise64 = NR3264_fadvise64