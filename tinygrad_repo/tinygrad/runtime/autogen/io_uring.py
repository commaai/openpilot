# mypy: ignore-errors
import ctypes
from tinygrad.runtime.support.c import DLL, Struct, CEnum, _IO, _IOW, _IOR, _IOWR
class struct_io_uring_sq(Struct): pass
class struct_io_uring_sqe(Struct): pass
__u8 = ctypes.c_ubyte
__u16 = ctypes.c_uint16
__s32 = ctypes.c_int32
class struct_io_uring_sqe_0(ctypes.Union): pass
__u64 = ctypes.c_uint64
class struct_io_uring_sqe_0_0(Struct): pass
__u32 = ctypes.c_uint32
struct_io_uring_sqe_0_0._fields_ = [
  ('cmd_op', ctypes.c_uint32),
  ('__pad1', ctypes.c_uint32),
]
struct_io_uring_sqe_0._anonymous_ = ['_0']
struct_io_uring_sqe_0._fields_ = [
  ('off', ctypes.c_uint64),
  ('addr2', ctypes.c_uint64),
  ('_0', struct_io_uring_sqe_0_0),
]
class struct_io_uring_sqe_1(ctypes.Union): pass
struct_io_uring_sqe_1._fields_ = [
  ('addr', ctypes.c_uint64),
  ('splice_off_in', ctypes.c_uint64),
]
class struct_io_uring_sqe_2(ctypes.Union): pass
__kernel_rwf_t = ctypes.c_int32
struct_io_uring_sqe_2._fields_ = [
  ('rw_flags', ctypes.c_int32),
  ('fsync_flags', ctypes.c_uint32),
  ('poll_events', ctypes.c_uint16),
  ('poll32_events', ctypes.c_uint32),
  ('sync_range_flags', ctypes.c_uint32),
  ('msg_flags', ctypes.c_uint32),
  ('timeout_flags', ctypes.c_uint32),
  ('accept_flags', ctypes.c_uint32),
  ('cancel_flags', ctypes.c_uint32),
  ('open_flags', ctypes.c_uint32),
  ('statx_flags', ctypes.c_uint32),
  ('fadvise_advice', ctypes.c_uint32),
  ('splice_flags', ctypes.c_uint32),
  ('rename_flags', ctypes.c_uint32),
  ('unlink_flags', ctypes.c_uint32),
  ('hardlink_flags', ctypes.c_uint32),
  ('xattr_flags', ctypes.c_uint32),
  ('msg_ring_flags', ctypes.c_uint32),
  ('uring_cmd_flags', ctypes.c_uint32),
]
class struct_io_uring_sqe_3(ctypes.Union): pass
struct_io_uring_sqe_3._packed_ = True
struct_io_uring_sqe_3._fields_ = [
  ('buf_index', ctypes.c_uint16),
  ('buf_group', ctypes.c_uint16),
]
class struct_io_uring_sqe_4(ctypes.Union): pass
class struct_io_uring_sqe_4_0(Struct): pass
struct_io_uring_sqe_4_0._fields_ = [
  ('addr_len', ctypes.c_uint16),
  ('__pad3', (ctypes.c_uint16 * 1)),
]
struct_io_uring_sqe_4._anonymous_ = ['_0']
struct_io_uring_sqe_4._fields_ = [
  ('splice_fd_in', ctypes.c_int32),
  ('file_index', ctypes.c_uint32),
  ('_0', struct_io_uring_sqe_4_0),
]
class struct_io_uring_sqe_5(ctypes.Union): pass
class struct_io_uring_sqe_5_0(Struct): pass
struct_io_uring_sqe_5_0._fields_ = [
  ('addr3', ctypes.c_uint64),
  ('__pad2', (ctypes.c_uint64 * 1)),
]
struct_io_uring_sqe_5._anonymous_ = ['_0']
struct_io_uring_sqe_5._fields_ = [
  ('_0', struct_io_uring_sqe_5_0),
  ('cmd', (ctypes.c_ubyte * 0)),
]
struct_io_uring_sqe._anonymous_ = ['_0', '_1', '_2', '_3', '_4', '_5']
struct_io_uring_sqe._fields_ = [
  ('opcode', ctypes.c_ubyte),
  ('flags', ctypes.c_ubyte),
  ('ioprio', ctypes.c_uint16),
  ('fd', ctypes.c_int32),
  ('_0', struct_io_uring_sqe_0),
  ('_1', struct_io_uring_sqe_1),
  ('len', ctypes.c_uint32),
  ('_2', struct_io_uring_sqe_2),
  ('user_data', ctypes.c_uint64),
  ('_3', struct_io_uring_sqe_3),
  ('personality', ctypes.c_uint16),
  ('_4', struct_io_uring_sqe_4),
  ('_5', struct_io_uring_sqe_5),
]
size_t = ctypes.c_uint64
struct_io_uring_sq._fields_ = [
  ('khead', ctypes.POINTER(ctypes.c_uint32)),
  ('ktail', ctypes.POINTER(ctypes.c_uint32)),
  ('kring_mask', ctypes.POINTER(ctypes.c_uint32)),
  ('kring_entries', ctypes.POINTER(ctypes.c_uint32)),
  ('kflags', ctypes.POINTER(ctypes.c_uint32)),
  ('kdropped', ctypes.POINTER(ctypes.c_uint32)),
  ('array', ctypes.POINTER(ctypes.c_uint32)),
  ('sqes', ctypes.POINTER(struct_io_uring_sqe)),
  ('sqe_head', ctypes.c_uint32),
  ('sqe_tail', ctypes.c_uint32),
  ('ring_sz', size_t),
  ('ring_ptr', ctypes.c_void_p),
  ('ring_mask', ctypes.c_uint32),
  ('ring_entries', ctypes.c_uint32),
  ('pad', (ctypes.c_uint32 * 2)),
]
class struct_io_uring_cq(Struct): pass
class struct_io_uring_cqe(Struct): pass
struct_io_uring_cqe._fields_ = [
  ('user_data', ctypes.c_uint64),
  ('res', ctypes.c_int32),
  ('flags', ctypes.c_uint32),
  ('big_cqe', (ctypes.c_uint64 * 0)),
]
struct_io_uring_cq._fields_ = [
  ('khead', ctypes.POINTER(ctypes.c_uint32)),
  ('ktail', ctypes.POINTER(ctypes.c_uint32)),
  ('kring_mask', ctypes.POINTER(ctypes.c_uint32)),
  ('kring_entries', ctypes.POINTER(ctypes.c_uint32)),
  ('kflags', ctypes.POINTER(ctypes.c_uint32)),
  ('koverflow', ctypes.POINTER(ctypes.c_uint32)),
  ('cqes', ctypes.POINTER(struct_io_uring_cqe)),
  ('ring_sz', size_t),
  ('ring_ptr', ctypes.c_void_p),
  ('ring_mask', ctypes.c_uint32),
  ('ring_entries', ctypes.c_uint32),
  ('pad', (ctypes.c_uint32 * 2)),
]
class struct_io_uring(Struct): pass
struct_io_uring._fields_ = [
  ('sq', struct_io_uring_sq),
  ('cq', struct_io_uring_cq),
  ('flags', ctypes.c_uint32),
  ('ring_fd', ctypes.c_int32),
  ('features', ctypes.c_uint32),
  ('enter_ring_fd', ctypes.c_int32),
  ('int_flags', ctypes.c_ubyte),
  ('pad', (ctypes.c_ubyte * 3)),
  ('pad2', ctypes.c_uint32),
]
class struct_statx(Struct): pass
class struct_statx_timestamp(Struct): pass
__s64 = ctypes.c_int64
struct_statx_timestamp._fields_ = [
  ('tv_sec', ctypes.c_int64),
  ('tv_nsec', ctypes.c_uint32),
  ('__reserved', ctypes.c_int32),
]
struct_statx._fields_ = [
  ('stx_mask', ctypes.c_uint32),
  ('stx_blksize', ctypes.c_uint32),
  ('stx_attributes', ctypes.c_uint64),
  ('stx_nlink', ctypes.c_uint32),
  ('stx_uid', ctypes.c_uint32),
  ('stx_gid', ctypes.c_uint32),
  ('stx_mode', ctypes.c_uint16),
  ('__spare0', (ctypes.c_uint16 * 1)),
  ('stx_ino', ctypes.c_uint64),
  ('stx_size', ctypes.c_uint64),
  ('stx_blocks', ctypes.c_uint64),
  ('stx_attributes_mask', ctypes.c_uint64),
  ('stx_atime', struct_statx_timestamp),
  ('stx_btime', struct_statx_timestamp),
  ('stx_ctime', struct_statx_timestamp),
  ('stx_mtime', struct_statx_timestamp),
  ('stx_rdev_major', ctypes.c_uint32),
  ('stx_rdev_minor', ctypes.c_uint32),
  ('stx_dev_major', ctypes.c_uint32),
  ('stx_dev_minor', ctypes.c_uint32),
  ('stx_mnt_id', ctypes.c_uint64),
  ('stx_dio_mem_align', ctypes.c_uint32),
  ('stx_dio_offset_align', ctypes.c_uint32),
  ('__spare3', (ctypes.c_uint64 * 12)),
]
class struct_epoll_event(Struct): pass
_anonenum0 = CEnum(ctypes.c_uint32)
IOSQE_FIXED_FILE_BIT = _anonenum0.define('IOSQE_FIXED_FILE_BIT', 0)
IOSQE_IO_DRAIN_BIT = _anonenum0.define('IOSQE_IO_DRAIN_BIT', 1)
IOSQE_IO_LINK_BIT = _anonenum0.define('IOSQE_IO_LINK_BIT', 2)
IOSQE_IO_HARDLINK_BIT = _anonenum0.define('IOSQE_IO_HARDLINK_BIT', 3)
IOSQE_ASYNC_BIT = _anonenum0.define('IOSQE_ASYNC_BIT', 4)
IOSQE_BUFFER_SELECT_BIT = _anonenum0.define('IOSQE_BUFFER_SELECT_BIT', 5)
IOSQE_CQE_SKIP_SUCCESS_BIT = _anonenum0.define('IOSQE_CQE_SKIP_SUCCESS_BIT', 6)

enum_io_uring_op = CEnum(ctypes.c_uint32)
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

_anonenum1 = CEnum(ctypes.c_uint32)
IORING_MSG_DATA = _anonenum1.define('IORING_MSG_DATA', 0)
IORING_MSG_SEND_FD = _anonenum1.define('IORING_MSG_SEND_FD', 1)

_anonenum2 = CEnum(ctypes.c_uint32)
IORING_CQE_BUFFER_SHIFT = _anonenum2.define('IORING_CQE_BUFFER_SHIFT', 16)

class struct_io_sqring_offsets(Struct): pass
struct_io_sqring_offsets._fields_ = [
  ('head', ctypes.c_uint32),
  ('tail', ctypes.c_uint32),
  ('ring_mask', ctypes.c_uint32),
  ('ring_entries', ctypes.c_uint32),
  ('flags', ctypes.c_uint32),
  ('dropped', ctypes.c_uint32),
  ('array', ctypes.c_uint32),
  ('resv1', ctypes.c_uint32),
  ('user_addr', ctypes.c_uint64),
]
class struct_io_cqring_offsets(Struct): pass
struct_io_cqring_offsets._fields_ = [
  ('head', ctypes.c_uint32),
  ('tail', ctypes.c_uint32),
  ('ring_mask', ctypes.c_uint32),
  ('ring_entries', ctypes.c_uint32),
  ('overflow', ctypes.c_uint32),
  ('cqes', ctypes.c_uint32),
  ('flags', ctypes.c_uint32),
  ('resv1', ctypes.c_uint32),
  ('user_addr', ctypes.c_uint64),
]
class struct_io_uring_params(Struct): pass
struct_io_uring_params._fields_ = [
  ('sq_entries', ctypes.c_uint32),
  ('cq_entries', ctypes.c_uint32),
  ('flags', ctypes.c_uint32),
  ('sq_thread_cpu', ctypes.c_uint32),
  ('sq_thread_idle', ctypes.c_uint32),
  ('features', ctypes.c_uint32),
  ('wq_fd', ctypes.c_uint32),
  ('resv', (ctypes.c_uint32 * 3)),
  ('sq_off', struct_io_sqring_offsets),
  ('cq_off', struct_io_cqring_offsets),
]
_anonenum3 = CEnum(ctypes.c_uint32)
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

_anonenum4 = CEnum(ctypes.c_uint32)
IO_WQ_BOUND = _anonenum4.define('IO_WQ_BOUND', 0)
IO_WQ_UNBOUND = _anonenum4.define('IO_WQ_UNBOUND', 1)

class struct_io_uring_files_update(Struct): pass
struct_io_uring_files_update._fields_ = [
  ('offset', ctypes.c_uint32),
  ('resv', ctypes.c_uint32),
  ('fds', ctypes.c_uint64),
]
class struct_io_uring_rsrc_register(Struct): pass
struct_io_uring_rsrc_register._fields_ = [
  ('nr', ctypes.c_uint32),
  ('flags', ctypes.c_uint32),
  ('resv2', ctypes.c_uint64),
  ('data', ctypes.c_uint64),
  ('tags', ctypes.c_uint64),
]
class struct_io_uring_rsrc_update(Struct): pass
struct_io_uring_rsrc_update._fields_ = [
  ('offset', ctypes.c_uint32),
  ('resv', ctypes.c_uint32),
  ('data', ctypes.c_uint64),
]
class struct_io_uring_rsrc_update2(Struct): pass
struct_io_uring_rsrc_update2._fields_ = [
  ('offset', ctypes.c_uint32),
  ('resv', ctypes.c_uint32),
  ('data', ctypes.c_uint64),
  ('tags', ctypes.c_uint64),
  ('nr', ctypes.c_uint32),
  ('resv2', ctypes.c_uint32),
]
class struct_io_uring_probe_op(Struct): pass
struct_io_uring_probe_op._fields_ = [
  ('op', ctypes.c_ubyte),
  ('resv', ctypes.c_ubyte),
  ('flags', ctypes.c_uint16),
  ('resv2', ctypes.c_uint32),
]
class struct_io_uring_probe(Struct): pass
struct_io_uring_probe._fields_ = [
  ('last_op', ctypes.c_ubyte),
  ('ops_len', ctypes.c_ubyte),
  ('resv', ctypes.c_uint16),
  ('resv2', (ctypes.c_uint32 * 3)),
  ('ops', (struct_io_uring_probe_op * 0)),
]
class struct_io_uring_restriction(Struct): pass
class struct_io_uring_restriction_0(ctypes.Union): pass
struct_io_uring_restriction_0._fields_ = [
  ('register_op', ctypes.c_ubyte),
  ('sqe_op', ctypes.c_ubyte),
  ('sqe_flags', ctypes.c_ubyte),
]
struct_io_uring_restriction._anonymous_ = ['_0']
struct_io_uring_restriction._fields_ = [
  ('opcode', ctypes.c_uint16),
  ('_0', struct_io_uring_restriction_0),
  ('resv', ctypes.c_ubyte),
  ('resv2', (ctypes.c_uint32 * 3)),
]
class struct_io_uring_buf(Struct): pass
struct_io_uring_buf._fields_ = [
  ('addr', ctypes.c_uint64),
  ('len', ctypes.c_uint32),
  ('bid', ctypes.c_uint16),
  ('resv', ctypes.c_uint16),
]
class struct_io_uring_buf_ring(Struct): pass
class struct_io_uring_buf_ring_0(ctypes.Union): pass
class struct_io_uring_buf_ring_0_0(Struct): pass
struct_io_uring_buf_ring_0_0._fields_ = [
  ('resv1', ctypes.c_uint64),
  ('resv2', ctypes.c_uint32),
  ('resv3', ctypes.c_uint16),
  ('tail', ctypes.c_uint16),
]
class struct_io_uring_buf_ring_0_1(Struct): pass
class struct_io_uring_buf_ring_0_1___empty_bufs(Struct): pass
struct_io_uring_buf_ring_0_1._fields_ = [
  ('__empty_bufs', struct_io_uring_buf_ring_0_1___empty_bufs),
  ('bufs', (struct_io_uring_buf * 0)),
]
struct_io_uring_buf_ring_0._anonymous_ = ['_0', '_1']
struct_io_uring_buf_ring_0._fields_ = [
  ('_0', struct_io_uring_buf_ring_0_0),
  ('_1', struct_io_uring_buf_ring_0_1),
]
struct_io_uring_buf_ring._anonymous_ = ['_0']
struct_io_uring_buf_ring._fields_ = [
  ('_0', struct_io_uring_buf_ring_0),
]
_anonenum5 = CEnum(ctypes.c_uint32)
IOU_PBUF_RING_MMAP = _anonenum5.define('IOU_PBUF_RING_MMAP', 1)

class struct_io_uring_buf_reg(Struct): pass
struct_io_uring_buf_reg._fields_ = [
  ('ring_addr', ctypes.c_uint64),
  ('ring_entries', ctypes.c_uint32),
  ('bgid', ctypes.c_uint16),
  ('flags', ctypes.c_uint16),
  ('resv', (ctypes.c_uint64 * 3)),
]
class struct_io_uring_buf_status(Struct): pass
struct_io_uring_buf_status._fields_ = [
  ('buf_group', ctypes.c_uint32),
  ('head', ctypes.c_uint32),
  ('resv', (ctypes.c_uint32 * 8)),
]
_anonenum6 = CEnum(ctypes.c_uint32)
IORING_RESTRICTION_REGISTER_OP = _anonenum6.define('IORING_RESTRICTION_REGISTER_OP', 0)
IORING_RESTRICTION_SQE_OP = _anonenum6.define('IORING_RESTRICTION_SQE_OP', 1)
IORING_RESTRICTION_SQE_FLAGS_ALLOWED = _anonenum6.define('IORING_RESTRICTION_SQE_FLAGS_ALLOWED', 2)
IORING_RESTRICTION_SQE_FLAGS_REQUIRED = _anonenum6.define('IORING_RESTRICTION_SQE_FLAGS_REQUIRED', 3)
IORING_RESTRICTION_LAST = _anonenum6.define('IORING_RESTRICTION_LAST', 4)

class struct_io_uring_getevents_arg(Struct): pass
struct_io_uring_getevents_arg._fields_ = [
  ('sigmask', ctypes.c_uint64),
  ('sigmask_sz', ctypes.c_uint32),
  ('pad', ctypes.c_uint32),
  ('ts', ctypes.c_uint64),
]
class struct_io_uring_sync_cancel_reg(Struct): pass
class struct___kernel_timespec(Struct): pass
__kernel_time64_t = ctypes.c_int64
struct___kernel_timespec._fields_ = [
  ('tv_sec', ctypes.c_int64),
  ('tv_nsec', ctypes.c_int64),
]
struct_io_uring_sync_cancel_reg._fields_ = [
  ('addr', ctypes.c_uint64),
  ('fd', ctypes.c_int32),
  ('flags', ctypes.c_uint32),
  ('timeout', struct___kernel_timespec),
  ('opcode', ctypes.c_ubyte),
  ('pad', (ctypes.c_ubyte * 7)),
  ('pad2', (ctypes.c_uint64 * 3)),
]
class struct_io_uring_file_index_range(Struct): pass
struct_io_uring_file_index_range._fields_ = [
  ('off', ctypes.c_uint32),
  ('len', ctypes.c_uint32),
  ('resv', ctypes.c_uint64),
]
class struct_io_uring_recvmsg_out(Struct): pass
struct_io_uring_recvmsg_out._fields_ = [
  ('namelen', ctypes.c_uint32),
  ('controllen', ctypes.c_uint32),
  ('payloadlen', ctypes.c_uint32),
  ('flags', ctypes.c_uint32),
]
_anonenum7 = CEnum(ctypes.c_uint32)
SOCKET_URING_OP_SIOCINQ = _anonenum7.define('SOCKET_URING_OP_SIOCINQ', 0)
SOCKET_URING_OP_SIOCOUTQ = _anonenum7.define('SOCKET_URING_OP_SIOCOUTQ', 1)
SOCKET_URING_OP_GETSOCKOPT = _anonenum7.define('SOCKET_URING_OP_GETSOCKOPT', 2)
SOCKET_URING_OP_SETSOCKOPT = _anonenum7.define('SOCKET_URING_OP_SETSOCKOPT', 3)

_XOPEN_SOURCE = 500
uring_unlikely = lambda cond: __builtin_expect( not  not (cond), 0)
uring_likely = lambda cond: __builtin_expect( not  not (cond), 1)
NR_io_uring_setup = 425
NR_io_uring_enter = 426
NR_io_uring_register = 427
io_uring_cqe_index = lambda ring,ptr,mask: (((ptr) & (mask)) << io_uring_cqe_shift(ring))
UNUSED = lambda x: (void)(x)
IO_URING_CHECK_VERSION = lambda major,minor: (major > IO_URING_VERSION_MAJOR or (major == IO_URING_VERSION_MAJOR and minor >= IO_URING_VERSION_MINOR))
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
IORING_URING_CMD_FIXED = (1 << 0)
IORING_URING_CMD_MASK = IORING_URING_CMD_FIXED
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
IORING_NOTIF_USAGE_ZC_COPIED = (1 << 31)
IORING_ACCEPT_MULTISHOT = (1 << 0)
IORING_MSG_RING_CQE_SKIP = (1 << 0)
IORING_MSG_RING_FLAGS_PASS = (1 << 1)
IORING_FIXED_FD_NO_CLOEXEC = (1 << 0)
IORING_CQE_F_BUFFER = (1 << 0)
IORING_CQE_F_MORE = (1 << 1)
IORING_CQE_F_SOCK_NONEMPTY = (1 << 2)
IORING_CQE_F_NOTIF = (1 << 3)
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
IORING_RSRC_REGISTER_SPARSE = (1 << 0)
IORING_REGISTER_FILES_SKIP = (-2)
IO_URING_OP_SUPPORTED = (1 << 0)
__SC_3264 = lambda _nr,_32,_64: __SYSCALL(_nr, _64)
__SC_COMP = lambda _nr,_sys,_comp: __SYSCALL(_nr, _sys)
__SC_COMP_3264 = lambda _nr,_32,_64,_comp: __SC_3264(_nr, _32, _64)
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
NR_syscalls = 462
NR_fcntl = NR3264_fcntl
NR_statfs = NR3264_statfs
NR_fstatfs = NR3264_fstatfs
NR_truncate = NR3264_truncate
NR_ftruncate = NR3264_ftruncate
NR_lseek = NR3264_lseek
NR_sendfile = NR3264_sendfile
NR_mmap = NR3264_mmap
NR_fadvise64 = NR3264_fadvise64