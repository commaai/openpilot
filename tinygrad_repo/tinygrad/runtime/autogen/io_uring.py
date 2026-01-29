# mypy: disable-error-code="empty-body"
from __future__ import annotations
import ctypes
from typing import Annotated, Literal, TypeAlias
from tinygrad.runtime.support.c import _IO, _IOW, _IOR, _IOWR
from tinygrad.runtime.support import c
@c.record
class struct_io_uring_sq(c.Struct):
  SIZE = 104
  khead: Annotated[c.POINTER[Annotated[int, ctypes.c_uint32]], 0]
  ktail: Annotated[c.POINTER[Annotated[int, ctypes.c_uint32]], 8]
  kring_mask: Annotated[c.POINTER[Annotated[int, ctypes.c_uint32]], 16]
  kring_entries: Annotated[c.POINTER[Annotated[int, ctypes.c_uint32]], 24]
  kflags: Annotated[c.POINTER[Annotated[int, ctypes.c_uint32]], 32]
  kdropped: Annotated[c.POINTER[Annotated[int, ctypes.c_uint32]], 40]
  array: Annotated[c.POINTER[Annotated[int, ctypes.c_uint32]], 48]
  sqes: Annotated[c.POINTER[struct_io_uring_sqe], 56]
  sqe_head: Annotated[Annotated[int, ctypes.c_uint32], 64]
  sqe_tail: Annotated[Annotated[int, ctypes.c_uint32], 68]
  ring_sz: Annotated[size_t, 72]
  ring_ptr: Annotated[ctypes.c_void_p, 80]
  ring_mask: Annotated[Annotated[int, ctypes.c_uint32], 88]
  ring_entries: Annotated[Annotated[int, ctypes.c_uint32], 92]
  pad: Annotated[c.Array[Annotated[int, ctypes.c_uint32], Literal[2]], 96]
@c.record
class struct_io_uring_sqe(c.Struct):
  SIZE = 64
  opcode: Annotated[Annotated[int, ctypes.c_ubyte], 0]
  flags: Annotated[Annotated[int, ctypes.c_ubyte], 1]
  ioprio: Annotated[Annotated[int, ctypes.c_uint16], 2]
  fd: Annotated[Annotated[int, ctypes.c_int32], 4]
  off: Annotated[Annotated[int, ctypes.c_uint64], 8]
  addr2: Annotated[Annotated[int, ctypes.c_uint64], 8]
  cmd_op: Annotated[Annotated[int, ctypes.c_uint32], 8]
  __pad1: Annotated[Annotated[int, ctypes.c_uint32], 12]
  addr: Annotated[Annotated[int, ctypes.c_uint64], 16]
  splice_off_in: Annotated[Annotated[int, ctypes.c_uint64], 16]
  len: Annotated[Annotated[int, ctypes.c_uint32], 24]
  rw_flags: Annotated[Annotated[int, ctypes.c_int32], 28]
  fsync_flags: Annotated[Annotated[int, ctypes.c_uint32], 28]
  poll_events: Annotated[Annotated[int, ctypes.c_uint16], 28]
  poll32_events: Annotated[Annotated[int, ctypes.c_uint32], 28]
  sync_range_flags: Annotated[Annotated[int, ctypes.c_uint32], 28]
  msg_flags: Annotated[Annotated[int, ctypes.c_uint32], 28]
  timeout_flags: Annotated[Annotated[int, ctypes.c_uint32], 28]
  accept_flags: Annotated[Annotated[int, ctypes.c_uint32], 28]
  cancel_flags: Annotated[Annotated[int, ctypes.c_uint32], 28]
  open_flags: Annotated[Annotated[int, ctypes.c_uint32], 28]
  statx_flags: Annotated[Annotated[int, ctypes.c_uint32], 28]
  fadvise_advice: Annotated[Annotated[int, ctypes.c_uint32], 28]
  splice_flags: Annotated[Annotated[int, ctypes.c_uint32], 28]
  rename_flags: Annotated[Annotated[int, ctypes.c_uint32], 28]
  unlink_flags: Annotated[Annotated[int, ctypes.c_uint32], 28]
  hardlink_flags: Annotated[Annotated[int, ctypes.c_uint32], 28]
  xattr_flags: Annotated[Annotated[int, ctypes.c_uint32], 28]
  msg_ring_flags: Annotated[Annotated[int, ctypes.c_uint32], 28]
  uring_cmd_flags: Annotated[Annotated[int, ctypes.c_uint32], 28]
  user_data: Annotated[Annotated[int, ctypes.c_uint64], 32]
  buf_index: Annotated[Annotated[int, ctypes.c_uint16], 40]
  buf_group: Annotated[Annotated[int, ctypes.c_uint16], 40]
  personality: Annotated[Annotated[int, ctypes.c_uint16], 42]
  splice_fd_in: Annotated[Annotated[int, ctypes.c_int32], 44]
  file_index: Annotated[Annotated[int, ctypes.c_uint32], 44]
  addr_len: Annotated[Annotated[int, ctypes.c_uint16], 44]
  __pad3: Annotated[c.Array[Annotated[int, ctypes.c_uint16], Literal[1]], 46]
  addr3: Annotated[Annotated[int, ctypes.c_uint64], 48]
  __pad2: Annotated[c.Array[Annotated[int, ctypes.c_uint64], Literal[1]], 56]
  cmd: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[0]], 48]
__u8: TypeAlias = Annotated[int, ctypes.c_ubyte]
__u16: TypeAlias = Annotated[int, ctypes.c_uint16]
__s32: TypeAlias = Annotated[int, ctypes.c_int32]
__u64: TypeAlias = Annotated[int, ctypes.c_uint64]
__u32: TypeAlias = Annotated[int, ctypes.c_uint32]
__kernel_rwf_t: TypeAlias = Annotated[int, ctypes.c_int32]
size_t: TypeAlias = Annotated[int, ctypes.c_uint64]
@c.record
class struct_io_uring_cq(c.Struct):
  SIZE = 88
  khead: Annotated[c.POINTER[Annotated[int, ctypes.c_uint32]], 0]
  ktail: Annotated[c.POINTER[Annotated[int, ctypes.c_uint32]], 8]
  kring_mask: Annotated[c.POINTER[Annotated[int, ctypes.c_uint32]], 16]
  kring_entries: Annotated[c.POINTER[Annotated[int, ctypes.c_uint32]], 24]
  kflags: Annotated[c.POINTER[Annotated[int, ctypes.c_uint32]], 32]
  koverflow: Annotated[c.POINTER[Annotated[int, ctypes.c_uint32]], 40]
  cqes: Annotated[c.POINTER[struct_io_uring_cqe], 48]
  ring_sz: Annotated[size_t, 56]
  ring_ptr: Annotated[ctypes.c_void_p, 64]
  ring_mask: Annotated[Annotated[int, ctypes.c_uint32], 72]
  ring_entries: Annotated[Annotated[int, ctypes.c_uint32], 76]
  pad: Annotated[c.Array[Annotated[int, ctypes.c_uint32], Literal[2]], 80]
@c.record
class struct_io_uring_cqe(c.Struct):
  SIZE = 16
  user_data: Annotated[Annotated[int, ctypes.c_uint64], 0]
  res: Annotated[Annotated[int, ctypes.c_int32], 8]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 12]
  big_cqe: Annotated[c.Array[Annotated[int, ctypes.c_uint64], Literal[0]], 16]
@c.record
class struct_io_uring(c.Struct):
  SIZE = 216
  sq: Annotated[struct_io_uring_sq, 0]
  cq: Annotated[struct_io_uring_cq, 104]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 192]
  ring_fd: Annotated[Annotated[int, ctypes.c_int32], 196]
  features: Annotated[Annotated[int, ctypes.c_uint32], 200]
  enter_ring_fd: Annotated[Annotated[int, ctypes.c_int32], 204]
  int_flags: Annotated[Annotated[int, ctypes.c_ubyte], 208]
  pad: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 209]
  pad2: Annotated[Annotated[int, ctypes.c_uint32], 212]
@c.record
class struct_statx(c.Struct):
  SIZE = 256
  stx_mask: Annotated[Annotated[int, ctypes.c_uint32], 0]
  stx_blksize: Annotated[Annotated[int, ctypes.c_uint32], 4]
  stx_attributes: Annotated[Annotated[int, ctypes.c_uint64], 8]
  stx_nlink: Annotated[Annotated[int, ctypes.c_uint32], 16]
  stx_uid: Annotated[Annotated[int, ctypes.c_uint32], 20]
  stx_gid: Annotated[Annotated[int, ctypes.c_uint32], 24]
  stx_mode: Annotated[Annotated[int, ctypes.c_uint16], 28]
  __spare0: Annotated[c.Array[Annotated[int, ctypes.c_uint16], Literal[1]], 30]
  stx_ino: Annotated[Annotated[int, ctypes.c_uint64], 32]
  stx_size: Annotated[Annotated[int, ctypes.c_uint64], 40]
  stx_blocks: Annotated[Annotated[int, ctypes.c_uint64], 48]
  stx_attributes_mask: Annotated[Annotated[int, ctypes.c_uint64], 56]
  stx_atime: Annotated[struct_statx_timestamp, 64]
  stx_btime: Annotated[struct_statx_timestamp, 80]
  stx_ctime: Annotated[struct_statx_timestamp, 96]
  stx_mtime: Annotated[struct_statx_timestamp, 112]
  stx_rdev_major: Annotated[Annotated[int, ctypes.c_uint32], 128]
  stx_rdev_minor: Annotated[Annotated[int, ctypes.c_uint32], 132]
  stx_dev_major: Annotated[Annotated[int, ctypes.c_uint32], 136]
  stx_dev_minor: Annotated[Annotated[int, ctypes.c_uint32], 140]
  stx_mnt_id: Annotated[Annotated[int, ctypes.c_uint64], 144]
  stx_dio_mem_align: Annotated[Annotated[int, ctypes.c_uint32], 152]
  stx_dio_offset_align: Annotated[Annotated[int, ctypes.c_uint32], 156]
  __spare3: Annotated[c.Array[Annotated[int, ctypes.c_uint64], Literal[12]], 160]
@c.record
class struct_statx_timestamp(c.Struct):
  SIZE = 16
  tv_sec: Annotated[Annotated[int, ctypes.c_int64], 0]
  tv_nsec: Annotated[Annotated[int, ctypes.c_uint32], 8]
  __reserved: Annotated[Annotated[int, ctypes.c_int32], 12]
__s64: TypeAlias = Annotated[int, ctypes.c_int64]
class struct_epoll_event(ctypes.Structure): pass
class _anonenum0(Annotated[int, ctypes.c_uint32], c.Enum): pass
IOSQE_FIXED_FILE_BIT = _anonenum0.define('IOSQE_FIXED_FILE_BIT', 0)
IOSQE_IO_DRAIN_BIT = _anonenum0.define('IOSQE_IO_DRAIN_BIT', 1)
IOSQE_IO_LINK_BIT = _anonenum0.define('IOSQE_IO_LINK_BIT', 2)
IOSQE_IO_HARDLINK_BIT = _anonenum0.define('IOSQE_IO_HARDLINK_BIT', 3)
IOSQE_ASYNC_BIT = _anonenum0.define('IOSQE_ASYNC_BIT', 4)
IOSQE_BUFFER_SELECT_BIT = _anonenum0.define('IOSQE_BUFFER_SELECT_BIT', 5)
IOSQE_CQE_SKIP_SUCCESS_BIT = _anonenum0.define('IOSQE_CQE_SKIP_SUCCESS_BIT', 6)

class enum_io_uring_op(Annotated[int, ctypes.c_uint32], c.Enum): pass
IORING_OP_NOP = enum_io_uring_op.define('IORING_OP_NOP', 0)
IORING_OP_READV = enum_io_uring_op.define('IORING_OP_READV', 1)
IORING_OP_WRITEV = enum_io_uring_op.define('IORING_OP_WRITEV', 2)
IORING_OP_FSYNC = enum_io_uring_op.define('IORING_OP_FSYNC', 3)
IORING_OP_READ_FIXED = enum_io_uring_op.define('IORING_OP_READ_FIXED', 4)
IORING_OP_WRITE_FIXED = enum_io_uring_op.define('IORING_OP_WRITE_FIXED', 5)
IORING_OP_POLL_ADD = enum_io_uring_op.define('IORING_OP_POLL_ADD', 6)
IORING_OP_POLL_REMOVE = enum_io_uring_op.define('IORING_OP_POLL_REMOVE', 7)
IORING_OP_SYNC_FILE_RANGE = enum_io_uring_op.define('IORING_OP_SYNC_FILE_RANGE', 8)
IORING_OP_SENDMSG = enum_io_uring_op.define('IORING_OP_SENDMSG', 9)
IORING_OP_RECVMSG = enum_io_uring_op.define('IORING_OP_RECVMSG', 10)
IORING_OP_TIMEOUT = enum_io_uring_op.define('IORING_OP_TIMEOUT', 11)
IORING_OP_TIMEOUT_REMOVE = enum_io_uring_op.define('IORING_OP_TIMEOUT_REMOVE', 12)
IORING_OP_ACCEPT = enum_io_uring_op.define('IORING_OP_ACCEPT', 13)
IORING_OP_ASYNC_CANCEL = enum_io_uring_op.define('IORING_OP_ASYNC_CANCEL', 14)
IORING_OP_LINK_TIMEOUT = enum_io_uring_op.define('IORING_OP_LINK_TIMEOUT', 15)
IORING_OP_CONNECT = enum_io_uring_op.define('IORING_OP_CONNECT', 16)
IORING_OP_FALLOCATE = enum_io_uring_op.define('IORING_OP_FALLOCATE', 17)
IORING_OP_OPENAT = enum_io_uring_op.define('IORING_OP_OPENAT', 18)
IORING_OP_CLOSE = enum_io_uring_op.define('IORING_OP_CLOSE', 19)
IORING_OP_FILES_UPDATE = enum_io_uring_op.define('IORING_OP_FILES_UPDATE', 20)
IORING_OP_STATX = enum_io_uring_op.define('IORING_OP_STATX', 21)
IORING_OP_READ = enum_io_uring_op.define('IORING_OP_READ', 22)
IORING_OP_WRITE = enum_io_uring_op.define('IORING_OP_WRITE', 23)
IORING_OP_FADVISE = enum_io_uring_op.define('IORING_OP_FADVISE', 24)
IORING_OP_MADVISE = enum_io_uring_op.define('IORING_OP_MADVISE', 25)
IORING_OP_SEND = enum_io_uring_op.define('IORING_OP_SEND', 26)
IORING_OP_RECV = enum_io_uring_op.define('IORING_OP_RECV', 27)
IORING_OP_OPENAT2 = enum_io_uring_op.define('IORING_OP_OPENAT2', 28)
IORING_OP_EPOLL_CTL = enum_io_uring_op.define('IORING_OP_EPOLL_CTL', 29)
IORING_OP_SPLICE = enum_io_uring_op.define('IORING_OP_SPLICE', 30)
IORING_OP_PROVIDE_BUFFERS = enum_io_uring_op.define('IORING_OP_PROVIDE_BUFFERS', 31)
IORING_OP_REMOVE_BUFFERS = enum_io_uring_op.define('IORING_OP_REMOVE_BUFFERS', 32)
IORING_OP_TEE = enum_io_uring_op.define('IORING_OP_TEE', 33)
IORING_OP_SHUTDOWN = enum_io_uring_op.define('IORING_OP_SHUTDOWN', 34)
IORING_OP_RENAMEAT = enum_io_uring_op.define('IORING_OP_RENAMEAT', 35)
IORING_OP_UNLINKAT = enum_io_uring_op.define('IORING_OP_UNLINKAT', 36)
IORING_OP_MKDIRAT = enum_io_uring_op.define('IORING_OP_MKDIRAT', 37)
IORING_OP_SYMLINKAT = enum_io_uring_op.define('IORING_OP_SYMLINKAT', 38)
IORING_OP_LINKAT = enum_io_uring_op.define('IORING_OP_LINKAT', 39)
IORING_OP_MSG_RING = enum_io_uring_op.define('IORING_OP_MSG_RING', 40)
IORING_OP_FSETXATTR = enum_io_uring_op.define('IORING_OP_FSETXATTR', 41)
IORING_OP_SETXATTR = enum_io_uring_op.define('IORING_OP_SETXATTR', 42)
IORING_OP_FGETXATTR = enum_io_uring_op.define('IORING_OP_FGETXATTR', 43)
IORING_OP_GETXATTR = enum_io_uring_op.define('IORING_OP_GETXATTR', 44)
IORING_OP_SOCKET = enum_io_uring_op.define('IORING_OP_SOCKET', 45)
IORING_OP_URING_CMD = enum_io_uring_op.define('IORING_OP_URING_CMD', 46)
IORING_OP_SEND_ZC = enum_io_uring_op.define('IORING_OP_SEND_ZC', 47)
IORING_OP_SENDMSG_ZC = enum_io_uring_op.define('IORING_OP_SENDMSG_ZC', 48)
IORING_OP_READ_MULTISHOT = enum_io_uring_op.define('IORING_OP_READ_MULTISHOT', 49)
IORING_OP_WAITID = enum_io_uring_op.define('IORING_OP_WAITID', 50)
IORING_OP_FUTEX_WAIT = enum_io_uring_op.define('IORING_OP_FUTEX_WAIT', 51)
IORING_OP_FUTEX_WAKE = enum_io_uring_op.define('IORING_OP_FUTEX_WAKE', 52)
IORING_OP_FUTEX_WAITV = enum_io_uring_op.define('IORING_OP_FUTEX_WAITV', 53)
IORING_OP_FIXED_FD_INSTALL = enum_io_uring_op.define('IORING_OP_FIXED_FD_INSTALL', 54)
IORING_OP_LAST = enum_io_uring_op.define('IORING_OP_LAST', 55)

class _anonenum1(Annotated[int, ctypes.c_uint32], c.Enum): pass
IORING_MSG_DATA = _anonenum1.define('IORING_MSG_DATA', 0)
IORING_MSG_SEND_FD = _anonenum1.define('IORING_MSG_SEND_FD', 1)

class _anonenum2(Annotated[int, ctypes.c_uint32], c.Enum): pass
IORING_CQE_BUFFER_SHIFT = _anonenum2.define('IORING_CQE_BUFFER_SHIFT', 16)

@c.record
class struct_io_sqring_offsets(c.Struct):
  SIZE = 40
  head: Annotated[Annotated[int, ctypes.c_uint32], 0]
  tail: Annotated[Annotated[int, ctypes.c_uint32], 4]
  ring_mask: Annotated[Annotated[int, ctypes.c_uint32], 8]
  ring_entries: Annotated[Annotated[int, ctypes.c_uint32], 12]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 16]
  dropped: Annotated[Annotated[int, ctypes.c_uint32], 20]
  array: Annotated[Annotated[int, ctypes.c_uint32], 24]
  resv1: Annotated[Annotated[int, ctypes.c_uint32], 28]
  user_addr: Annotated[Annotated[int, ctypes.c_uint64], 32]
@c.record
class struct_io_cqring_offsets(c.Struct):
  SIZE = 40
  head: Annotated[Annotated[int, ctypes.c_uint32], 0]
  tail: Annotated[Annotated[int, ctypes.c_uint32], 4]
  ring_mask: Annotated[Annotated[int, ctypes.c_uint32], 8]
  ring_entries: Annotated[Annotated[int, ctypes.c_uint32], 12]
  overflow: Annotated[Annotated[int, ctypes.c_uint32], 16]
  cqes: Annotated[Annotated[int, ctypes.c_uint32], 20]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 24]
  resv1: Annotated[Annotated[int, ctypes.c_uint32], 28]
  user_addr: Annotated[Annotated[int, ctypes.c_uint64], 32]
@c.record
class struct_io_uring_params(c.Struct):
  SIZE = 120
  sq_entries: Annotated[Annotated[int, ctypes.c_uint32], 0]
  cq_entries: Annotated[Annotated[int, ctypes.c_uint32], 4]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 8]
  sq_thread_cpu: Annotated[Annotated[int, ctypes.c_uint32], 12]
  sq_thread_idle: Annotated[Annotated[int, ctypes.c_uint32], 16]
  features: Annotated[Annotated[int, ctypes.c_uint32], 20]
  wq_fd: Annotated[Annotated[int, ctypes.c_uint32], 24]
  resv: Annotated[c.Array[Annotated[int, ctypes.c_uint32], Literal[3]], 28]
  sq_off: Annotated[struct_io_sqring_offsets, 40]
  cq_off: Annotated[struct_io_cqring_offsets, 80]
class _anonenum3(Annotated[int, ctypes.c_uint32], c.Enum): pass
IORING_REGISTER_BUFFERS = _anonenum3.define('IORING_REGISTER_BUFFERS', 0)
IORING_UNREGISTER_BUFFERS = _anonenum3.define('IORING_UNREGISTER_BUFFERS', 1)
IORING_REGISTER_FILES = _anonenum3.define('IORING_REGISTER_FILES', 2)
IORING_UNREGISTER_FILES = _anonenum3.define('IORING_UNREGISTER_FILES', 3)
IORING_REGISTER_EVENTFD = _anonenum3.define('IORING_REGISTER_EVENTFD', 4)
IORING_UNREGISTER_EVENTFD = _anonenum3.define('IORING_UNREGISTER_EVENTFD', 5)
IORING_REGISTER_FILES_UPDATE = _anonenum3.define('IORING_REGISTER_FILES_UPDATE', 6)
IORING_REGISTER_EVENTFD_ASYNC = _anonenum3.define('IORING_REGISTER_EVENTFD_ASYNC', 7)
IORING_REGISTER_PROBE = _anonenum3.define('IORING_REGISTER_PROBE', 8)
IORING_REGISTER_PERSONALITY = _anonenum3.define('IORING_REGISTER_PERSONALITY', 9)
IORING_UNREGISTER_PERSONALITY = _anonenum3.define('IORING_UNREGISTER_PERSONALITY', 10)
IORING_REGISTER_RESTRICTIONS = _anonenum3.define('IORING_REGISTER_RESTRICTIONS', 11)
IORING_REGISTER_ENABLE_RINGS = _anonenum3.define('IORING_REGISTER_ENABLE_RINGS', 12)
IORING_REGISTER_FILES2 = _anonenum3.define('IORING_REGISTER_FILES2', 13)
IORING_REGISTER_FILES_UPDATE2 = _anonenum3.define('IORING_REGISTER_FILES_UPDATE2', 14)
IORING_REGISTER_BUFFERS2 = _anonenum3.define('IORING_REGISTER_BUFFERS2', 15)
IORING_REGISTER_BUFFERS_UPDATE = _anonenum3.define('IORING_REGISTER_BUFFERS_UPDATE', 16)
IORING_REGISTER_IOWQ_AFF = _anonenum3.define('IORING_REGISTER_IOWQ_AFF', 17)
IORING_UNREGISTER_IOWQ_AFF = _anonenum3.define('IORING_UNREGISTER_IOWQ_AFF', 18)
IORING_REGISTER_IOWQ_MAX_WORKERS = _anonenum3.define('IORING_REGISTER_IOWQ_MAX_WORKERS', 19)
IORING_REGISTER_RING_FDS = _anonenum3.define('IORING_REGISTER_RING_FDS', 20)
IORING_UNREGISTER_RING_FDS = _anonenum3.define('IORING_UNREGISTER_RING_FDS', 21)
IORING_REGISTER_PBUF_RING = _anonenum3.define('IORING_REGISTER_PBUF_RING', 22)
IORING_UNREGISTER_PBUF_RING = _anonenum3.define('IORING_UNREGISTER_PBUF_RING', 23)
IORING_REGISTER_SYNC_CANCEL = _anonenum3.define('IORING_REGISTER_SYNC_CANCEL', 24)
IORING_REGISTER_FILE_ALLOC_RANGE = _anonenum3.define('IORING_REGISTER_FILE_ALLOC_RANGE', 25)
IORING_REGISTER_PBUF_STATUS = _anonenum3.define('IORING_REGISTER_PBUF_STATUS', 26)
IORING_REGISTER_LAST = _anonenum3.define('IORING_REGISTER_LAST', 27)
IORING_REGISTER_USE_REGISTERED_RING = _anonenum3.define('IORING_REGISTER_USE_REGISTERED_RING', 2147483648)

class _anonenum4(Annotated[int, ctypes.c_uint32], c.Enum): pass
IO_WQ_BOUND = _anonenum4.define('IO_WQ_BOUND', 0)
IO_WQ_UNBOUND = _anonenum4.define('IO_WQ_UNBOUND', 1)

@c.record
class struct_io_uring_files_update(c.Struct):
  SIZE = 16
  offset: Annotated[Annotated[int, ctypes.c_uint32], 0]
  resv: Annotated[Annotated[int, ctypes.c_uint32], 4]
  fds: Annotated[Annotated[int, ctypes.c_uint64], 8]
@c.record
class struct_io_uring_rsrc_register(c.Struct):
  SIZE = 32
  nr: Annotated[Annotated[int, ctypes.c_uint32], 0]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 4]
  resv2: Annotated[Annotated[int, ctypes.c_uint64], 8]
  data: Annotated[Annotated[int, ctypes.c_uint64], 16]
  tags: Annotated[Annotated[int, ctypes.c_uint64], 24]
@c.record
class struct_io_uring_rsrc_update(c.Struct):
  SIZE = 16
  offset: Annotated[Annotated[int, ctypes.c_uint32], 0]
  resv: Annotated[Annotated[int, ctypes.c_uint32], 4]
  data: Annotated[Annotated[int, ctypes.c_uint64], 8]
@c.record
class struct_io_uring_rsrc_update2(c.Struct):
  SIZE = 32
  offset: Annotated[Annotated[int, ctypes.c_uint32], 0]
  resv: Annotated[Annotated[int, ctypes.c_uint32], 4]
  data: Annotated[Annotated[int, ctypes.c_uint64], 8]
  tags: Annotated[Annotated[int, ctypes.c_uint64], 16]
  nr: Annotated[Annotated[int, ctypes.c_uint32], 24]
  resv2: Annotated[Annotated[int, ctypes.c_uint32], 28]
@c.record
class struct_io_uring_probe_op(c.Struct):
  SIZE = 8
  op: Annotated[Annotated[int, ctypes.c_ubyte], 0]
  resv: Annotated[Annotated[int, ctypes.c_ubyte], 1]
  flags: Annotated[Annotated[int, ctypes.c_uint16], 2]
  resv2: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_io_uring_probe(c.Struct):
  SIZE = 16
  last_op: Annotated[Annotated[int, ctypes.c_ubyte], 0]
  ops_len: Annotated[Annotated[int, ctypes.c_ubyte], 1]
  resv: Annotated[Annotated[int, ctypes.c_uint16], 2]
  resv2: Annotated[c.Array[Annotated[int, ctypes.c_uint32], Literal[3]], 4]
  ops: Annotated[c.Array[struct_io_uring_probe_op, Literal[0]], 16]
@c.record
class struct_io_uring_restriction(c.Struct):
  SIZE = 16
  opcode: Annotated[Annotated[int, ctypes.c_uint16], 0]
  register_op: Annotated[Annotated[int, ctypes.c_ubyte], 2]
  sqe_op: Annotated[Annotated[int, ctypes.c_ubyte], 2]
  sqe_flags: Annotated[Annotated[int, ctypes.c_ubyte], 2]
  resv: Annotated[Annotated[int, ctypes.c_ubyte], 3]
  resv2: Annotated[c.Array[Annotated[int, ctypes.c_uint32], Literal[3]], 4]
@c.record
class struct_io_uring_buf(c.Struct):
  SIZE = 16
  addr: Annotated[Annotated[int, ctypes.c_uint64], 0]
  len: Annotated[Annotated[int, ctypes.c_uint32], 8]
  bid: Annotated[Annotated[int, ctypes.c_uint16], 12]
  resv: Annotated[Annotated[int, ctypes.c_uint16], 14]
@c.record
class struct_io_uring_buf_ring(c.Struct):
  SIZE = 16
  resv1: Annotated[Annotated[int, ctypes.c_uint64], 0]
  resv2: Annotated[Annotated[int, ctypes.c_uint32], 8]
  resv3: Annotated[Annotated[int, ctypes.c_uint16], 12]
  tail: Annotated[Annotated[int, ctypes.c_uint16], 14]
  __empty_bufs: Annotated[struct_io_uring_buf_ring___empty_bufs, 0]
  bufs: Annotated[c.Array[struct_io_uring_buf, Literal[0]], 0]
class struct_io_uring_buf_ring___empty_bufs(ctypes.Structure): pass
class _anonenum5(Annotated[int, ctypes.c_uint32], c.Enum): pass
IOU_PBUF_RING_MMAP = _anonenum5.define('IOU_PBUF_RING_MMAP', 1)

@c.record
class struct_io_uring_buf_reg(c.Struct):
  SIZE = 40
  ring_addr: Annotated[Annotated[int, ctypes.c_uint64], 0]
  ring_entries: Annotated[Annotated[int, ctypes.c_uint32], 8]
  bgid: Annotated[Annotated[int, ctypes.c_uint16], 12]
  flags: Annotated[Annotated[int, ctypes.c_uint16], 14]
  resv: Annotated[c.Array[Annotated[int, ctypes.c_uint64], Literal[3]], 16]
@c.record
class struct_io_uring_buf_status(c.Struct):
  SIZE = 40
  buf_group: Annotated[Annotated[int, ctypes.c_uint32], 0]
  head: Annotated[Annotated[int, ctypes.c_uint32], 4]
  resv: Annotated[c.Array[Annotated[int, ctypes.c_uint32], Literal[8]], 8]
class _anonenum6(Annotated[int, ctypes.c_uint32], c.Enum): pass
IORING_RESTRICTION_REGISTER_OP = _anonenum6.define('IORING_RESTRICTION_REGISTER_OP', 0)
IORING_RESTRICTION_SQE_OP = _anonenum6.define('IORING_RESTRICTION_SQE_OP', 1)
IORING_RESTRICTION_SQE_FLAGS_ALLOWED = _anonenum6.define('IORING_RESTRICTION_SQE_FLAGS_ALLOWED', 2)
IORING_RESTRICTION_SQE_FLAGS_REQUIRED = _anonenum6.define('IORING_RESTRICTION_SQE_FLAGS_REQUIRED', 3)
IORING_RESTRICTION_LAST = _anonenum6.define('IORING_RESTRICTION_LAST', 4)

@c.record
class struct_io_uring_getevents_arg(c.Struct):
  SIZE = 24
  sigmask: Annotated[Annotated[int, ctypes.c_uint64], 0]
  sigmask_sz: Annotated[Annotated[int, ctypes.c_uint32], 8]
  pad: Annotated[Annotated[int, ctypes.c_uint32], 12]
  ts: Annotated[Annotated[int, ctypes.c_uint64], 16]
@c.record
class struct_io_uring_sync_cancel_reg(c.Struct):
  SIZE = 64
  addr: Annotated[Annotated[int, ctypes.c_uint64], 0]
  fd: Annotated[Annotated[int, ctypes.c_int32], 8]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 12]
  timeout: Annotated[struct___kernel_timespec, 16]
  opcode: Annotated[Annotated[int, ctypes.c_ubyte], 32]
  pad: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[7]], 33]
  pad2: Annotated[c.Array[Annotated[int, ctypes.c_uint64], Literal[3]], 40]
@c.record
class struct___kernel_timespec(c.Struct):
  SIZE = 16
  tv_sec: Annotated[Annotated[int, ctypes.c_int64], 0]
  tv_nsec: Annotated[Annotated[int, ctypes.c_int64], 8]
__kernel_time64_t: TypeAlias = Annotated[int, ctypes.c_int64]
@c.record
class struct_io_uring_file_index_range(c.Struct):
  SIZE = 16
  off: Annotated[Annotated[int, ctypes.c_uint32], 0]
  len: Annotated[Annotated[int, ctypes.c_uint32], 4]
  resv: Annotated[Annotated[int, ctypes.c_uint64], 8]
@c.record
class struct_io_uring_recvmsg_out(c.Struct):
  SIZE = 16
  namelen: Annotated[Annotated[int, ctypes.c_uint32], 0]
  controllen: Annotated[Annotated[int, ctypes.c_uint32], 4]
  payloadlen: Annotated[Annotated[int, ctypes.c_uint32], 8]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 12]
class _anonenum7(Annotated[int, ctypes.c_uint32], c.Enum): pass
SOCKET_URING_OP_SIOCINQ = _anonenum7.define('SOCKET_URING_OP_SIOCINQ', 0)
SOCKET_URING_OP_SIOCOUTQ = _anonenum7.define('SOCKET_URING_OP_SIOCOUTQ', 1)
SOCKET_URING_OP_GETSOCKOPT = _anonenum7.define('SOCKET_URING_OP_GETSOCKOPT', 2)
SOCKET_URING_OP_SETSOCKOPT = _anonenum7.define('SOCKET_URING_OP_SETSOCKOPT', 3)

c.init_records()
_XOPEN_SOURCE = 500 # type: ignore
uring_unlikely = lambda cond: __builtin_expect( not  not (cond), 0) # type: ignore
uring_likely = lambda cond: __builtin_expect( not  not (cond), 1) # type: ignore
NR_io_uring_setup = 425 # type: ignore
NR_io_uring_enter = 426 # type: ignore
NR_io_uring_register = 427 # type: ignore
io_uring_cqe_index = lambda ring,ptr,mask: (((ptr) & (mask)) << io_uring_cqe_shift(ring)) # type: ignore
UNUSED = lambda x: (void)(x) # type: ignore
IO_URING_CHECK_VERSION = lambda major,minor: (major > IO_URING_VERSION_MAJOR or (major == IO_URING_VERSION_MAJOR and minor >= IO_URING_VERSION_MINOR)) # type: ignore
IORING_FILE_INDEX_ALLOC = (~0) # type: ignore
IOSQE_FIXED_FILE = (1 << IOSQE_FIXED_FILE_BIT) # type: ignore
IOSQE_IO_DRAIN = (1 << IOSQE_IO_DRAIN_BIT) # type: ignore
IOSQE_IO_LINK = (1 << IOSQE_IO_LINK_BIT) # type: ignore
IOSQE_IO_HARDLINK = (1 << IOSQE_IO_HARDLINK_BIT) # type: ignore
IOSQE_ASYNC = (1 << IOSQE_ASYNC_BIT) # type: ignore
IOSQE_BUFFER_SELECT = (1 << IOSQE_BUFFER_SELECT_BIT) # type: ignore
IOSQE_CQE_SKIP_SUCCESS = (1 << IOSQE_CQE_SKIP_SUCCESS_BIT) # type: ignore
IORING_SETUP_IOPOLL = (1 << 0) # type: ignore
IORING_SETUP_SQPOLL = (1 << 1) # type: ignore
IORING_SETUP_SQ_AFF = (1 << 2) # type: ignore
IORING_SETUP_CQSIZE = (1 << 3) # type: ignore
IORING_SETUP_CLAMP = (1 << 4) # type: ignore
IORING_SETUP_ATTACH_WQ = (1 << 5) # type: ignore
IORING_SETUP_R_DISABLED = (1 << 6) # type: ignore
IORING_SETUP_SUBMIT_ALL = (1 << 7) # type: ignore
IORING_SETUP_COOP_TASKRUN = (1 << 8) # type: ignore
IORING_SETUP_TASKRUN_FLAG = (1 << 9) # type: ignore
IORING_SETUP_SQE128 = (1 << 10) # type: ignore
IORING_SETUP_CQE32 = (1 << 11) # type: ignore
IORING_SETUP_SINGLE_ISSUER = (1 << 12) # type: ignore
IORING_SETUP_DEFER_TASKRUN = (1 << 13) # type: ignore
IORING_SETUP_NO_MMAP = (1 << 14) # type: ignore
IORING_SETUP_REGISTERED_FD_ONLY = (1 << 15) # type: ignore
IORING_SETUP_NO_SQARRAY = (1 << 16) # type: ignore
IORING_URING_CMD_FIXED = (1 << 0) # type: ignore
IORING_URING_CMD_MASK = IORING_URING_CMD_FIXED # type: ignore
IORING_FSYNC_DATASYNC = (1 << 0) # type: ignore
IORING_TIMEOUT_ABS = (1 << 0) # type: ignore
IORING_TIMEOUT_UPDATE = (1 << 1) # type: ignore
IORING_TIMEOUT_BOOTTIME = (1 << 2) # type: ignore
IORING_TIMEOUT_REALTIME = (1 << 3) # type: ignore
IORING_LINK_TIMEOUT_UPDATE = (1 << 4) # type: ignore
IORING_TIMEOUT_ETIME_SUCCESS = (1 << 5) # type: ignore
IORING_TIMEOUT_MULTISHOT = (1 << 6) # type: ignore
IORING_TIMEOUT_CLOCK_MASK = (IORING_TIMEOUT_BOOTTIME | IORING_TIMEOUT_REALTIME) # type: ignore
IORING_TIMEOUT_UPDATE_MASK = (IORING_TIMEOUT_UPDATE | IORING_LINK_TIMEOUT_UPDATE) # type: ignore
SPLICE_F_FD_IN_FIXED = (1 << 31) # type: ignore
IORING_POLL_ADD_MULTI = (1 << 0) # type: ignore
IORING_POLL_UPDATE_EVENTS = (1 << 1) # type: ignore
IORING_POLL_UPDATE_USER_DATA = (1 << 2) # type: ignore
IORING_POLL_ADD_LEVEL = (1 << 3) # type: ignore
IORING_ASYNC_CANCEL_ALL = (1 << 0) # type: ignore
IORING_ASYNC_CANCEL_FD = (1 << 1) # type: ignore
IORING_ASYNC_CANCEL_ANY = (1 << 2) # type: ignore
IORING_ASYNC_CANCEL_FD_FIXED = (1 << 3) # type: ignore
IORING_ASYNC_CANCEL_USERDATA = (1 << 4) # type: ignore
IORING_ASYNC_CANCEL_OP = (1 << 5) # type: ignore
IORING_RECVSEND_POLL_FIRST = (1 << 0) # type: ignore
IORING_RECV_MULTISHOT = (1 << 1) # type: ignore
IORING_RECVSEND_FIXED_BUF = (1 << 2) # type: ignore
IORING_SEND_ZC_REPORT_USAGE = (1 << 3) # type: ignore
IORING_NOTIF_USAGE_ZC_COPIED = (1 << 31) # type: ignore
IORING_ACCEPT_MULTISHOT = (1 << 0) # type: ignore
IORING_MSG_RING_CQE_SKIP = (1 << 0) # type: ignore
IORING_MSG_RING_FLAGS_PASS = (1 << 1) # type: ignore
IORING_FIXED_FD_NO_CLOEXEC = (1 << 0) # type: ignore
IORING_CQE_F_BUFFER = (1 << 0) # type: ignore
IORING_CQE_F_MORE = (1 << 1) # type: ignore
IORING_CQE_F_SOCK_NONEMPTY = (1 << 2) # type: ignore
IORING_CQE_F_NOTIF = (1 << 3) # type: ignore
IORING_OFF_SQ_RING = 0 # type: ignore
IORING_OFF_CQ_RING = 0x8000000 # type: ignore
IORING_OFF_SQES = 0x10000000 # type: ignore
IORING_OFF_PBUF_RING = 0x80000000 # type: ignore
IORING_OFF_PBUF_SHIFT = 16 # type: ignore
IORING_OFF_MMAP_MASK = 0xf8000000 # type: ignore
IORING_SQ_NEED_WAKEUP = (1 << 0) # type: ignore
IORING_SQ_CQ_OVERFLOW = (1 << 1) # type: ignore
IORING_SQ_TASKRUN = (1 << 2) # type: ignore
IORING_CQ_EVENTFD_DISABLED = (1 << 0) # type: ignore
IORING_ENTER_GETEVENTS = (1 << 0) # type: ignore
IORING_ENTER_SQ_WAKEUP = (1 << 1) # type: ignore
IORING_ENTER_SQ_WAIT = (1 << 2) # type: ignore
IORING_ENTER_EXT_ARG = (1 << 3) # type: ignore
IORING_ENTER_REGISTERED_RING = (1 << 4) # type: ignore
IORING_FEAT_SINGLE_MMAP = (1 << 0) # type: ignore
IORING_FEAT_NODROP = (1 << 1) # type: ignore
IORING_FEAT_SUBMIT_STABLE = (1 << 2) # type: ignore
IORING_FEAT_RW_CUR_POS = (1 << 3) # type: ignore
IORING_FEAT_CUR_PERSONALITY = (1 << 4) # type: ignore
IORING_FEAT_FAST_POLL = (1 << 5) # type: ignore
IORING_FEAT_POLL_32BITS = (1 << 6) # type: ignore
IORING_FEAT_SQPOLL_NONFIXED = (1 << 7) # type: ignore
IORING_FEAT_EXT_ARG = (1 << 8) # type: ignore
IORING_FEAT_NATIVE_WORKERS = (1 << 9) # type: ignore
IORING_FEAT_RSRC_TAGS = (1 << 10) # type: ignore
IORING_FEAT_CQE_SKIP = (1 << 11) # type: ignore
IORING_FEAT_LINKED_FILE = (1 << 12) # type: ignore
IORING_FEAT_REG_REG_RING = (1 << 13) # type: ignore
IORING_RSRC_REGISTER_SPARSE = (1 << 0) # type: ignore
IORING_REGISTER_FILES_SKIP = (-2) # type: ignore
IO_URING_OP_SUPPORTED = (1 << 0) # type: ignore
__SC_3264 = lambda _nr,_32,_64: __SYSCALL(_nr, _64) # type: ignore
__SC_COMP = lambda _nr,_sys,_comp: __SYSCALL(_nr, _sys) # type: ignore
__SC_COMP_3264 = lambda _nr,_32,_64,_comp: __SC_3264(_nr, _32, _64) # type: ignore
NR_io_setup = 0 # type: ignore
NR_io_destroy = 1 # type: ignore
NR_io_submit = 2 # type: ignore
NR_io_cancel = 3 # type: ignore
NR_io_getevents = 4 # type: ignore
NR_setxattr = 5 # type: ignore
NR_lsetxattr = 6 # type: ignore
NR_fsetxattr = 7 # type: ignore
NR_getxattr = 8 # type: ignore
NR_lgetxattr = 9 # type: ignore
NR_fgetxattr = 10 # type: ignore
NR_listxattr = 11 # type: ignore
NR_llistxattr = 12 # type: ignore
NR_flistxattr = 13 # type: ignore
NR_removexattr = 14 # type: ignore
NR_lremovexattr = 15 # type: ignore
NR_fremovexattr = 16 # type: ignore
NR_getcwd = 17 # type: ignore
NR_lookup_dcookie = 18 # type: ignore
NR_eventfd2 = 19 # type: ignore
NR_epoll_create1 = 20 # type: ignore
NR_epoll_ctl = 21 # type: ignore
NR_epoll_pwait = 22 # type: ignore
NR_dup = 23 # type: ignore
NR_dup3 = 24 # type: ignore
NR3264_fcntl = 25 # type: ignore
NR_inotify_init1 = 26 # type: ignore
NR_inotify_add_watch = 27 # type: ignore
NR_inotify_rm_watch = 28 # type: ignore
NR_ioctl = 29 # type: ignore
NR_ioprio_set = 30 # type: ignore
NR_ioprio_get = 31 # type: ignore
NR_flock = 32 # type: ignore
NR_mknodat = 33 # type: ignore
NR_mkdirat = 34 # type: ignore
NR_unlinkat = 35 # type: ignore
NR_symlinkat = 36 # type: ignore
NR_linkat = 37 # type: ignore
NR_umount2 = 39 # type: ignore
NR_mount = 40 # type: ignore
NR_pivot_root = 41 # type: ignore
NR_nfsservctl = 42 # type: ignore
NR3264_statfs = 43 # type: ignore
NR3264_fstatfs = 44 # type: ignore
NR3264_truncate = 45 # type: ignore
NR3264_ftruncate = 46 # type: ignore
NR_fallocate = 47 # type: ignore
NR_faccessat = 48 # type: ignore
NR_chdir = 49 # type: ignore
NR_fchdir = 50 # type: ignore
NR_chroot = 51 # type: ignore
NR_fchmod = 52 # type: ignore
NR_fchmodat = 53 # type: ignore
NR_fchownat = 54 # type: ignore
NR_fchown = 55 # type: ignore
NR_openat = 56 # type: ignore
NR_close = 57 # type: ignore
NR_vhangup = 58 # type: ignore
NR_pipe2 = 59 # type: ignore
NR_quotactl = 60 # type: ignore
NR_getdents64 = 61 # type: ignore
NR3264_lseek = 62 # type: ignore
NR_read = 63 # type: ignore
NR_write = 64 # type: ignore
NR_readv = 65 # type: ignore
NR_writev = 66 # type: ignore
NR_pread64 = 67 # type: ignore
NR_pwrite64 = 68 # type: ignore
NR_preadv = 69 # type: ignore
NR_pwritev = 70 # type: ignore
NR3264_sendfile = 71 # type: ignore
NR_pselect6 = 72 # type: ignore
NR_ppoll = 73 # type: ignore
NR_signalfd4 = 74 # type: ignore
NR_vmsplice = 75 # type: ignore
NR_splice = 76 # type: ignore
NR_tee = 77 # type: ignore
NR_readlinkat = 78 # type: ignore
NR_sync = 81 # type: ignore
NR_fsync = 82 # type: ignore
NR_fdatasync = 83 # type: ignore
NR_sync_file_range = 84 # type: ignore
NR_timerfd_create = 85 # type: ignore
NR_timerfd_settime = 86 # type: ignore
NR_timerfd_gettime = 87 # type: ignore
NR_utimensat = 88 # type: ignore
NR_acct = 89 # type: ignore
NR_capget = 90 # type: ignore
NR_capset = 91 # type: ignore
NR_personality = 92 # type: ignore
NR_exit = 93 # type: ignore
NR_exit_group = 94 # type: ignore
NR_waitid = 95 # type: ignore
NR_set_tid_address = 96 # type: ignore
NR_unshare = 97 # type: ignore
NR_futex = 98 # type: ignore
NR_set_robust_list = 99 # type: ignore
NR_get_robust_list = 100 # type: ignore
NR_nanosleep = 101 # type: ignore
NR_getitimer = 102 # type: ignore
NR_setitimer = 103 # type: ignore
NR_kexec_load = 104 # type: ignore
NR_init_module = 105 # type: ignore
NR_delete_module = 106 # type: ignore
NR_timer_create = 107 # type: ignore
NR_timer_gettime = 108 # type: ignore
NR_timer_getoverrun = 109 # type: ignore
NR_timer_settime = 110 # type: ignore
NR_timer_delete = 111 # type: ignore
NR_clock_settime = 112 # type: ignore
NR_clock_gettime = 113 # type: ignore
NR_clock_getres = 114 # type: ignore
NR_clock_nanosleep = 115 # type: ignore
NR_syslog = 116 # type: ignore
NR_ptrace = 117 # type: ignore
NR_sched_setparam = 118 # type: ignore
NR_sched_setscheduler = 119 # type: ignore
NR_sched_getscheduler = 120 # type: ignore
NR_sched_getparam = 121 # type: ignore
NR_sched_setaffinity = 122 # type: ignore
NR_sched_getaffinity = 123 # type: ignore
NR_sched_yield = 124 # type: ignore
NR_sched_get_priority_max = 125 # type: ignore
NR_sched_get_priority_min = 126 # type: ignore
NR_sched_rr_get_interval = 127 # type: ignore
NR_restart_syscall = 128 # type: ignore
NR_kill = 129 # type: ignore
NR_tkill = 130 # type: ignore
NR_tgkill = 131 # type: ignore
NR_sigaltstack = 132 # type: ignore
NR_rt_sigsuspend = 133 # type: ignore
NR_rt_sigaction = 134 # type: ignore
NR_rt_sigprocmask = 135 # type: ignore
NR_rt_sigpending = 136 # type: ignore
NR_rt_sigtimedwait = 137 # type: ignore
NR_rt_sigqueueinfo = 138 # type: ignore
NR_rt_sigreturn = 139 # type: ignore
NR_setpriority = 140 # type: ignore
NR_getpriority = 141 # type: ignore
NR_reboot = 142 # type: ignore
NR_setregid = 143 # type: ignore
NR_setgid = 144 # type: ignore
NR_setreuid = 145 # type: ignore
NR_setuid = 146 # type: ignore
NR_setresuid = 147 # type: ignore
NR_getresuid = 148 # type: ignore
NR_setresgid = 149 # type: ignore
NR_getresgid = 150 # type: ignore
NR_setfsuid = 151 # type: ignore
NR_setfsgid = 152 # type: ignore
NR_times = 153 # type: ignore
NR_setpgid = 154 # type: ignore
NR_getpgid = 155 # type: ignore
NR_getsid = 156 # type: ignore
NR_setsid = 157 # type: ignore
NR_getgroups = 158 # type: ignore
NR_setgroups = 159 # type: ignore
NR_uname = 160 # type: ignore
NR_sethostname = 161 # type: ignore
NR_setdomainname = 162 # type: ignore
NR_getrusage = 165 # type: ignore
NR_umask = 166 # type: ignore
NR_prctl = 167 # type: ignore
NR_getcpu = 168 # type: ignore
NR_gettimeofday = 169 # type: ignore
NR_settimeofday = 170 # type: ignore
NR_adjtimex = 171 # type: ignore
NR_getpid = 172 # type: ignore
NR_getppid = 173 # type: ignore
NR_getuid = 174 # type: ignore
NR_geteuid = 175 # type: ignore
NR_getgid = 176 # type: ignore
NR_getegid = 177 # type: ignore
NR_gettid = 178 # type: ignore
NR_sysinfo = 179 # type: ignore
NR_mq_open = 180 # type: ignore
NR_mq_unlink = 181 # type: ignore
NR_mq_timedsend = 182 # type: ignore
NR_mq_timedreceive = 183 # type: ignore
NR_mq_notify = 184 # type: ignore
NR_mq_getsetattr = 185 # type: ignore
NR_msgget = 186 # type: ignore
NR_msgctl = 187 # type: ignore
NR_msgrcv = 188 # type: ignore
NR_msgsnd = 189 # type: ignore
NR_semget = 190 # type: ignore
NR_semctl = 191 # type: ignore
NR_semtimedop = 192 # type: ignore
NR_semop = 193 # type: ignore
NR_shmget = 194 # type: ignore
NR_shmctl = 195 # type: ignore
NR_shmat = 196 # type: ignore
NR_shmdt = 197 # type: ignore
NR_socket = 198 # type: ignore
NR_socketpair = 199 # type: ignore
NR_bind = 200 # type: ignore
NR_listen = 201 # type: ignore
NR_accept = 202 # type: ignore
NR_connect = 203 # type: ignore
NR_getsockname = 204 # type: ignore
NR_getpeername = 205 # type: ignore
NR_sendto = 206 # type: ignore
NR_recvfrom = 207 # type: ignore
NR_setsockopt = 208 # type: ignore
NR_getsockopt = 209 # type: ignore
NR_shutdown = 210 # type: ignore
NR_sendmsg = 211 # type: ignore
NR_recvmsg = 212 # type: ignore
NR_readahead = 213 # type: ignore
NR_brk = 214 # type: ignore
NR_munmap = 215 # type: ignore
NR_mremap = 216 # type: ignore
NR_add_key = 217 # type: ignore
NR_request_key = 218 # type: ignore
NR_keyctl = 219 # type: ignore
NR_clone = 220 # type: ignore
NR_execve = 221 # type: ignore
NR3264_mmap = 222 # type: ignore
NR3264_fadvise64 = 223 # type: ignore
NR_swapon = 224 # type: ignore
NR_swapoff = 225 # type: ignore
NR_mprotect = 226 # type: ignore
NR_msync = 227 # type: ignore
NR_mlock = 228 # type: ignore
NR_munlock = 229 # type: ignore
NR_mlockall = 230 # type: ignore
NR_munlockall = 231 # type: ignore
NR_mincore = 232 # type: ignore
NR_madvise = 233 # type: ignore
NR_remap_file_pages = 234 # type: ignore
NR_mbind = 235 # type: ignore
NR_get_mempolicy = 236 # type: ignore
NR_set_mempolicy = 237 # type: ignore
NR_migrate_pages = 238 # type: ignore
NR_move_pages = 239 # type: ignore
NR_rt_tgsigqueueinfo = 240 # type: ignore
NR_perf_event_open = 241 # type: ignore
NR_accept4 = 242 # type: ignore
NR_recvmmsg = 243 # type: ignore
NR_arch_specific_syscall = 244 # type: ignore
NR_wait4 = 260 # type: ignore
NR_prlimit64 = 261 # type: ignore
NR_fanotify_init = 262 # type: ignore
NR_fanotify_mark = 263 # type: ignore
NR_name_to_handle_at = 264 # type: ignore
NR_open_by_handle_at = 265 # type: ignore
NR_clock_adjtime = 266 # type: ignore
NR_syncfs = 267 # type: ignore
NR_setns = 268 # type: ignore
NR_sendmmsg = 269 # type: ignore
NR_process_vm_readv = 270 # type: ignore
NR_process_vm_writev = 271 # type: ignore
NR_kcmp = 272 # type: ignore
NR_finit_module = 273 # type: ignore
NR_sched_setattr = 274 # type: ignore
NR_sched_getattr = 275 # type: ignore
NR_renameat2 = 276 # type: ignore
NR_seccomp = 277 # type: ignore
NR_getrandom = 278 # type: ignore
NR_memfd_create = 279 # type: ignore
NR_bpf = 280 # type: ignore
NR_execveat = 281 # type: ignore
NR_userfaultfd = 282 # type: ignore
NR_membarrier = 283 # type: ignore
NR_mlock2 = 284 # type: ignore
NR_copy_file_range = 285 # type: ignore
NR_preadv2 = 286 # type: ignore
NR_pwritev2 = 287 # type: ignore
NR_pkey_mprotect = 288 # type: ignore
NR_pkey_alloc = 289 # type: ignore
NR_pkey_free = 290 # type: ignore
NR_statx = 291 # type: ignore
NR_io_pgetevents = 292 # type: ignore
NR_rseq = 293 # type: ignore
NR_kexec_file_load = 294 # type: ignore
NR_pidfd_send_signal = 424 # type: ignore
NR_io_uring_setup = 425 # type: ignore
NR_io_uring_enter = 426 # type: ignore
NR_io_uring_register = 427 # type: ignore
NR_open_tree = 428 # type: ignore
NR_move_mount = 429 # type: ignore
NR_fsopen = 430 # type: ignore
NR_fsconfig = 431 # type: ignore
NR_fsmount = 432 # type: ignore
NR_fspick = 433 # type: ignore
NR_pidfd_open = 434 # type: ignore
NR_close_range = 436 # type: ignore
NR_openat2 = 437 # type: ignore
NR_pidfd_getfd = 438 # type: ignore
NR_faccessat2 = 439 # type: ignore
NR_process_madvise = 440 # type: ignore
NR_epoll_pwait2 = 441 # type: ignore
NR_mount_setattr = 442 # type: ignore
NR_quotactl_fd = 443 # type: ignore
NR_landlock_create_ruleset = 444 # type: ignore
NR_landlock_add_rule = 445 # type: ignore
NR_landlock_restrict_self = 446 # type: ignore
NR_process_mrelease = 448 # type: ignore
NR_futex_waitv = 449 # type: ignore
NR_set_mempolicy_home_node = 450 # type: ignore
NR_cachestat = 451 # type: ignore
NR_fchmodat2 = 452 # type: ignore
NR_map_shadow_stack = 453 # type: ignore
NR_futex_wake = 454 # type: ignore
NR_futex_wait = 455 # type: ignore
NR_futex_requeue = 456 # type: ignore
NR_statmount = 457 # type: ignore
NR_listmount = 458 # type: ignore
NR_lsm_get_self_attr = 459 # type: ignore
NR_lsm_set_self_attr = 460 # type: ignore
NR_lsm_list_modules = 461 # type: ignore
NR_syscalls = 462 # type: ignore
NR_fcntl = NR3264_fcntl # type: ignore
NR_statfs = NR3264_statfs # type: ignore
NR_fstatfs = NR3264_fstatfs # type: ignore
NR_truncate = NR3264_truncate # type: ignore
NR_ftruncate = NR3264_ftruncate # type: ignore
NR_lseek = NR3264_lseek # type: ignore
NR_sendfile = NR3264_sendfile # type: ignore
NR_mmap = NR3264_mmap # type: ignore
NR_fadvise64 = NR3264_fadvise64 # type: ignore