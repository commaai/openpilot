# mypy: disable-error-code="empty-body"
from __future__ import annotations
import ctypes
from typing import Annotated, Literal, TypeAlias
from tinygrad.runtime.support.c import _IO, _IOW, _IOR, _IOWR
from tinygrad.runtime.support import c
dll = c.DLL('ib', 'ibverbs', use_errno=True)
@c.record
class union_ibv_gid(c.Struct):
  SIZE = 16
  raw: Annotated[c.Array[uint8_t, Literal[16]], 0]
  _global: Annotated[union_ibv_gid_global, 0]
uint8_t: TypeAlias = Annotated[int, ctypes.c_ubyte]
@c.record
class union_ibv_gid_global(c.Struct):
  SIZE = 16
  subnet_prefix: Annotated[Annotated[int, ctypes.c_uint64], 0]
  interface_id: Annotated[Annotated[int, ctypes.c_uint64], 8]
__be64: TypeAlias = Annotated[int, ctypes.c_uint64]
class enum_ibv_gid_type(Annotated[int, ctypes.c_uint32], c.Enum): pass
IBV_GID_TYPE_IB = enum_ibv_gid_type.define('IBV_GID_TYPE_IB', 0)
IBV_GID_TYPE_ROCE_V1 = enum_ibv_gid_type.define('IBV_GID_TYPE_ROCE_V1', 1)
IBV_GID_TYPE_ROCE_V2 = enum_ibv_gid_type.define('IBV_GID_TYPE_ROCE_V2', 2)

@c.record
class struct_ibv_gid_entry(c.Struct):
  SIZE = 32
  gid: Annotated[union_ibv_gid, 0]
  gid_index: Annotated[uint32_t, 16]
  port_num: Annotated[uint32_t, 20]
  gid_type: Annotated[uint32_t, 24]
  ndev_ifindex: Annotated[uint32_t, 28]
uint32_t: TypeAlias = Annotated[int, ctypes.c_uint32]
class enum_ibv_node_type(Annotated[int, ctypes.c_int32], c.Enum): pass
IBV_NODE_UNKNOWN = enum_ibv_node_type.define('IBV_NODE_UNKNOWN', -1)
IBV_NODE_CA = enum_ibv_node_type.define('IBV_NODE_CA', 1)
IBV_NODE_SWITCH = enum_ibv_node_type.define('IBV_NODE_SWITCH', 2)
IBV_NODE_ROUTER = enum_ibv_node_type.define('IBV_NODE_ROUTER', 3)
IBV_NODE_RNIC = enum_ibv_node_type.define('IBV_NODE_RNIC', 4)
IBV_NODE_USNIC = enum_ibv_node_type.define('IBV_NODE_USNIC', 5)
IBV_NODE_USNIC_UDP = enum_ibv_node_type.define('IBV_NODE_USNIC_UDP', 6)
IBV_NODE_UNSPECIFIED = enum_ibv_node_type.define('IBV_NODE_UNSPECIFIED', 7)

class enum_ibv_transport_type(Annotated[int, ctypes.c_int32], c.Enum): pass
IBV_TRANSPORT_UNKNOWN = enum_ibv_transport_type.define('IBV_TRANSPORT_UNKNOWN', -1)
IBV_TRANSPORT_IB = enum_ibv_transport_type.define('IBV_TRANSPORT_IB', 0)
IBV_TRANSPORT_IWARP = enum_ibv_transport_type.define('IBV_TRANSPORT_IWARP', 1)
IBV_TRANSPORT_USNIC = enum_ibv_transport_type.define('IBV_TRANSPORT_USNIC', 2)
IBV_TRANSPORT_USNIC_UDP = enum_ibv_transport_type.define('IBV_TRANSPORT_USNIC_UDP', 3)
IBV_TRANSPORT_UNSPECIFIED = enum_ibv_transport_type.define('IBV_TRANSPORT_UNSPECIFIED', 4)

class enum_ibv_device_cap_flags(Annotated[int, ctypes.c_uint32], c.Enum): pass
IBV_DEVICE_RESIZE_MAX_WR = enum_ibv_device_cap_flags.define('IBV_DEVICE_RESIZE_MAX_WR', 1)
IBV_DEVICE_BAD_PKEY_CNTR = enum_ibv_device_cap_flags.define('IBV_DEVICE_BAD_PKEY_CNTR', 2)
IBV_DEVICE_BAD_QKEY_CNTR = enum_ibv_device_cap_flags.define('IBV_DEVICE_BAD_QKEY_CNTR', 4)
IBV_DEVICE_RAW_MULTI = enum_ibv_device_cap_flags.define('IBV_DEVICE_RAW_MULTI', 8)
IBV_DEVICE_AUTO_PATH_MIG = enum_ibv_device_cap_flags.define('IBV_DEVICE_AUTO_PATH_MIG', 16)
IBV_DEVICE_CHANGE_PHY_PORT = enum_ibv_device_cap_flags.define('IBV_DEVICE_CHANGE_PHY_PORT', 32)
IBV_DEVICE_UD_AV_PORT_ENFORCE = enum_ibv_device_cap_flags.define('IBV_DEVICE_UD_AV_PORT_ENFORCE', 64)
IBV_DEVICE_CURR_QP_STATE_MOD = enum_ibv_device_cap_flags.define('IBV_DEVICE_CURR_QP_STATE_MOD', 128)
IBV_DEVICE_SHUTDOWN_PORT = enum_ibv_device_cap_flags.define('IBV_DEVICE_SHUTDOWN_PORT', 256)
IBV_DEVICE_INIT_TYPE = enum_ibv_device_cap_flags.define('IBV_DEVICE_INIT_TYPE', 512)
IBV_DEVICE_PORT_ACTIVE_EVENT = enum_ibv_device_cap_flags.define('IBV_DEVICE_PORT_ACTIVE_EVENT', 1024)
IBV_DEVICE_SYS_IMAGE_GUID = enum_ibv_device_cap_flags.define('IBV_DEVICE_SYS_IMAGE_GUID', 2048)
IBV_DEVICE_RC_RNR_NAK_GEN = enum_ibv_device_cap_flags.define('IBV_DEVICE_RC_RNR_NAK_GEN', 4096)
IBV_DEVICE_SRQ_RESIZE = enum_ibv_device_cap_flags.define('IBV_DEVICE_SRQ_RESIZE', 8192)
IBV_DEVICE_N_NOTIFY_CQ = enum_ibv_device_cap_flags.define('IBV_DEVICE_N_NOTIFY_CQ', 16384)
IBV_DEVICE_MEM_WINDOW = enum_ibv_device_cap_flags.define('IBV_DEVICE_MEM_WINDOW', 131072)
IBV_DEVICE_UD_IP_CSUM = enum_ibv_device_cap_flags.define('IBV_DEVICE_UD_IP_CSUM', 262144)
IBV_DEVICE_XRC = enum_ibv_device_cap_flags.define('IBV_DEVICE_XRC', 1048576)
IBV_DEVICE_MEM_MGT_EXTENSIONS = enum_ibv_device_cap_flags.define('IBV_DEVICE_MEM_MGT_EXTENSIONS', 2097152)
IBV_DEVICE_MEM_WINDOW_TYPE_2A = enum_ibv_device_cap_flags.define('IBV_DEVICE_MEM_WINDOW_TYPE_2A', 8388608)
IBV_DEVICE_MEM_WINDOW_TYPE_2B = enum_ibv_device_cap_flags.define('IBV_DEVICE_MEM_WINDOW_TYPE_2B', 16777216)
IBV_DEVICE_RC_IP_CSUM = enum_ibv_device_cap_flags.define('IBV_DEVICE_RC_IP_CSUM', 33554432)
IBV_DEVICE_RAW_IP_CSUM = enum_ibv_device_cap_flags.define('IBV_DEVICE_RAW_IP_CSUM', 67108864)
IBV_DEVICE_MANAGED_FLOW_STEERING = enum_ibv_device_cap_flags.define('IBV_DEVICE_MANAGED_FLOW_STEERING', 536870912)

class enum_ibv_fork_status(Annotated[int, ctypes.c_uint32], c.Enum): pass
IBV_FORK_DISABLED = enum_ibv_fork_status.define('IBV_FORK_DISABLED', 0)
IBV_FORK_ENABLED = enum_ibv_fork_status.define('IBV_FORK_ENABLED', 1)
IBV_FORK_UNNEEDED = enum_ibv_fork_status.define('IBV_FORK_UNNEEDED', 2)

class enum_ibv_atomic_cap(Annotated[int, ctypes.c_uint32], c.Enum): pass
IBV_ATOMIC_NONE = enum_ibv_atomic_cap.define('IBV_ATOMIC_NONE', 0)
IBV_ATOMIC_HCA = enum_ibv_atomic_cap.define('IBV_ATOMIC_HCA', 1)
IBV_ATOMIC_GLOB = enum_ibv_atomic_cap.define('IBV_ATOMIC_GLOB', 2)

@c.record
class struct_ibv_alloc_dm_attr(c.Struct):
  SIZE = 16
  length: Annotated[size_t, 0]
  log_align_req: Annotated[uint32_t, 8]
  comp_mask: Annotated[uint32_t, 12]
size_t: TypeAlias = Annotated[int, ctypes.c_uint64]
class enum_ibv_dm_mask(Annotated[int, ctypes.c_uint32], c.Enum): pass
IBV_DM_MASK_HANDLE = enum_ibv_dm_mask.define('IBV_DM_MASK_HANDLE', 1)

@c.record
class struct_ibv_dm(c.Struct):
  SIZE = 32
  context: Annotated[c.POINTER[struct_ibv_context], 0]
  memcpy_to_dm: Annotated[c.CFUNCTYPE[Annotated[int, ctypes.c_int32], [c.POINTER[struct_ibv_dm], uint64_t, ctypes.c_void_p, size_t]], 8]
  memcpy_from_dm: Annotated[c.CFUNCTYPE[Annotated[int, ctypes.c_int32], [ctypes.c_void_p, c.POINTER[struct_ibv_dm], uint64_t, size_t]], 16]
  comp_mask: Annotated[uint32_t, 24]
  handle: Annotated[uint32_t, 28]
@c.record
class struct_ibv_context(c.Struct):
  SIZE = 328
  device: Annotated[c.POINTER[struct_ibv_device], 0]
  ops: Annotated[struct_ibv_context_ops, 8]
  cmd_fd: Annotated[Annotated[int, ctypes.c_int32], 264]
  async_fd: Annotated[Annotated[int, ctypes.c_int32], 268]
  num_comp_vectors: Annotated[Annotated[int, ctypes.c_int32], 272]
  mutex: Annotated[pthread_mutex_t, 280]
  abi_compat: Annotated[ctypes.c_void_p, 320]
@c.record
class struct_ibv_device(c.Struct):
  SIZE = 664
  _ops: Annotated[struct__ibv_device_ops, 0]
  node_type: Annotated[enum_ibv_node_type, 16]
  transport_type: Annotated[enum_ibv_transport_type, 20]
  name: Annotated[c.Array[Annotated[bytes, ctypes.c_char], Literal[64]], 24]
  dev_name: Annotated[c.Array[Annotated[bytes, ctypes.c_char], Literal[64]], 88]
  dev_path: Annotated[c.Array[Annotated[bytes, ctypes.c_char], Literal[256]], 152]
  ibdev_path: Annotated[c.Array[Annotated[bytes, ctypes.c_char], Literal[256]], 408]
@c.record
class struct__ibv_device_ops(c.Struct):
  SIZE = 16
  _dummy1: Annotated[c.CFUNCTYPE[c.POINTER[struct_ibv_context], [c.POINTER[struct_ibv_device], Annotated[int, ctypes.c_int32]]], 0]
  _dummy2: Annotated[c.CFUNCTYPE[None, [c.POINTER[struct_ibv_context]]], 8]
@c.record
class struct_ibv_context_ops(c.Struct):
  SIZE = 256
  _compat_query_device: Annotated[c.CFUNCTYPE[Annotated[int, ctypes.c_int32], [c.POINTER[struct_ibv_context], c.POINTER[struct_ibv_device_attr]]], 0]
  _compat_query_port: Annotated[c.CFUNCTYPE[Annotated[int, ctypes.c_int32], [c.POINTER[struct_ibv_context], uint8_t, c.POINTER[struct__compat_ibv_port_attr]]], 8]
  _compat_alloc_pd: Annotated[c.CFUNCTYPE[ctypes.c_void_p, []], 16]
  _compat_dealloc_pd: Annotated[c.CFUNCTYPE[ctypes.c_void_p, []], 24]
  _compat_reg_mr: Annotated[c.CFUNCTYPE[ctypes.c_void_p, []], 32]
  _compat_rereg_mr: Annotated[c.CFUNCTYPE[ctypes.c_void_p, []], 40]
  _compat_dereg_mr: Annotated[c.CFUNCTYPE[ctypes.c_void_p, []], 48]
  alloc_mw: Annotated[c.CFUNCTYPE[c.POINTER[struct_ibv_mw], [c.POINTER[struct_ibv_pd], enum_ibv_mw_type]], 56]
  bind_mw: Annotated[c.CFUNCTYPE[Annotated[int, ctypes.c_int32], [c.POINTER[struct_ibv_qp], c.POINTER[struct_ibv_mw], c.POINTER[struct_ibv_mw_bind]]], 64]
  dealloc_mw: Annotated[c.CFUNCTYPE[Annotated[int, ctypes.c_int32], [c.POINTER[struct_ibv_mw]]], 72]
  _compat_create_cq: Annotated[c.CFUNCTYPE[ctypes.c_void_p, []], 80]
  poll_cq: Annotated[c.CFUNCTYPE[Annotated[int, ctypes.c_int32], [c.POINTER[struct_ibv_cq], Annotated[int, ctypes.c_int32], c.POINTER[struct_ibv_wc]]], 88]
  req_notify_cq: Annotated[c.CFUNCTYPE[Annotated[int, ctypes.c_int32], [c.POINTER[struct_ibv_cq], Annotated[int, ctypes.c_int32]]], 96]
  _compat_cq_event: Annotated[c.CFUNCTYPE[ctypes.c_void_p, []], 104]
  _compat_resize_cq: Annotated[c.CFUNCTYPE[ctypes.c_void_p, []], 112]
  _compat_destroy_cq: Annotated[c.CFUNCTYPE[ctypes.c_void_p, []], 120]
  _compat_create_srq: Annotated[c.CFUNCTYPE[ctypes.c_void_p, []], 128]
  _compat_modify_srq: Annotated[c.CFUNCTYPE[ctypes.c_void_p, []], 136]
  _compat_query_srq: Annotated[c.CFUNCTYPE[ctypes.c_void_p, []], 144]
  _compat_destroy_srq: Annotated[c.CFUNCTYPE[ctypes.c_void_p, []], 152]
  post_srq_recv: Annotated[c.CFUNCTYPE[Annotated[int, ctypes.c_int32], [c.POINTER[struct_ibv_srq], c.POINTER[struct_ibv_recv_wr], c.POINTER[c.POINTER[struct_ibv_recv_wr]]]], 160]
  _compat_create_qp: Annotated[c.CFUNCTYPE[ctypes.c_void_p, []], 168]
  _compat_query_qp: Annotated[c.CFUNCTYPE[ctypes.c_void_p, []], 176]
  _compat_modify_qp: Annotated[c.CFUNCTYPE[ctypes.c_void_p, []], 184]
  _compat_destroy_qp: Annotated[c.CFUNCTYPE[ctypes.c_void_p, []], 192]
  post_send: Annotated[c.CFUNCTYPE[Annotated[int, ctypes.c_int32], [c.POINTER[struct_ibv_qp], c.POINTER[struct_ibv_send_wr], c.POINTER[c.POINTER[struct_ibv_send_wr]]]], 200]
  post_recv: Annotated[c.CFUNCTYPE[Annotated[int, ctypes.c_int32], [c.POINTER[struct_ibv_qp], c.POINTER[struct_ibv_recv_wr], c.POINTER[c.POINTER[struct_ibv_recv_wr]]]], 208]
  _compat_create_ah: Annotated[c.CFUNCTYPE[ctypes.c_void_p, []], 216]
  _compat_destroy_ah: Annotated[c.CFUNCTYPE[ctypes.c_void_p, []], 224]
  _compat_attach_mcast: Annotated[c.CFUNCTYPE[ctypes.c_void_p, []], 232]
  _compat_detach_mcast: Annotated[c.CFUNCTYPE[ctypes.c_void_p, []], 240]
  _compat_async_event: Annotated[c.CFUNCTYPE[ctypes.c_void_p, []], 248]
@c.record
class struct_ibv_device_attr(c.Struct):
  SIZE = 232
  fw_ver: Annotated[c.Array[Annotated[bytes, ctypes.c_char], Literal[64]], 0]
  node_guid: Annotated[Annotated[int, ctypes.c_uint64], 64]
  sys_image_guid: Annotated[Annotated[int, ctypes.c_uint64], 72]
  max_mr_size: Annotated[uint64_t, 80]
  page_size_cap: Annotated[uint64_t, 88]
  vendor_id: Annotated[uint32_t, 96]
  vendor_part_id: Annotated[uint32_t, 100]
  hw_ver: Annotated[uint32_t, 104]
  max_qp: Annotated[Annotated[int, ctypes.c_int32], 108]
  max_qp_wr: Annotated[Annotated[int, ctypes.c_int32], 112]
  device_cap_flags: Annotated[Annotated[int, ctypes.c_uint32], 116]
  max_sge: Annotated[Annotated[int, ctypes.c_int32], 120]
  max_sge_rd: Annotated[Annotated[int, ctypes.c_int32], 124]
  max_cq: Annotated[Annotated[int, ctypes.c_int32], 128]
  max_cqe: Annotated[Annotated[int, ctypes.c_int32], 132]
  max_mr: Annotated[Annotated[int, ctypes.c_int32], 136]
  max_pd: Annotated[Annotated[int, ctypes.c_int32], 140]
  max_qp_rd_atom: Annotated[Annotated[int, ctypes.c_int32], 144]
  max_ee_rd_atom: Annotated[Annotated[int, ctypes.c_int32], 148]
  max_res_rd_atom: Annotated[Annotated[int, ctypes.c_int32], 152]
  max_qp_init_rd_atom: Annotated[Annotated[int, ctypes.c_int32], 156]
  max_ee_init_rd_atom: Annotated[Annotated[int, ctypes.c_int32], 160]
  atomic_cap: Annotated[enum_ibv_atomic_cap, 164]
  max_ee: Annotated[Annotated[int, ctypes.c_int32], 168]
  max_rdd: Annotated[Annotated[int, ctypes.c_int32], 172]
  max_mw: Annotated[Annotated[int, ctypes.c_int32], 176]
  max_raw_ipv6_qp: Annotated[Annotated[int, ctypes.c_int32], 180]
  max_raw_ethy_qp: Annotated[Annotated[int, ctypes.c_int32], 184]
  max_mcast_grp: Annotated[Annotated[int, ctypes.c_int32], 188]
  max_mcast_qp_attach: Annotated[Annotated[int, ctypes.c_int32], 192]
  max_total_mcast_qp_attach: Annotated[Annotated[int, ctypes.c_int32], 196]
  max_ah: Annotated[Annotated[int, ctypes.c_int32], 200]
  max_fmr: Annotated[Annotated[int, ctypes.c_int32], 204]
  max_map_per_fmr: Annotated[Annotated[int, ctypes.c_int32], 208]
  max_srq: Annotated[Annotated[int, ctypes.c_int32], 212]
  max_srq_wr: Annotated[Annotated[int, ctypes.c_int32], 216]
  max_srq_sge: Annotated[Annotated[int, ctypes.c_int32], 220]
  max_pkeys: Annotated[uint16_t, 224]
  local_ca_ack_delay: Annotated[uint8_t, 226]
  phys_port_cnt: Annotated[uint8_t, 227]
uint64_t: TypeAlias = Annotated[int, ctypes.c_uint64]
uint16_t: TypeAlias = Annotated[int, ctypes.c_uint16]
class struct__compat_ibv_port_attr(ctypes.Structure): pass
@c.record
class struct_ibv_mw(c.Struct):
  SIZE = 32
  context: Annotated[c.POINTER[struct_ibv_context], 0]
  pd: Annotated[c.POINTER[struct_ibv_pd], 8]
  rkey: Annotated[uint32_t, 16]
  handle: Annotated[uint32_t, 20]
  type: Annotated[enum_ibv_mw_type, 24]
@c.record
class struct_ibv_pd(c.Struct):
  SIZE = 16
  context: Annotated[c.POINTER[struct_ibv_context], 0]
  handle: Annotated[uint32_t, 8]
class enum_ibv_mw_type(Annotated[int, ctypes.c_uint32], c.Enum): pass
IBV_MW_TYPE_1 = enum_ibv_mw_type.define('IBV_MW_TYPE_1', 1)
IBV_MW_TYPE_2 = enum_ibv_mw_type.define('IBV_MW_TYPE_2', 2)

@c.record
class struct_ibv_qp(c.Struct):
  SIZE = 160
  context: Annotated[c.POINTER[struct_ibv_context], 0]
  qp_context: Annotated[ctypes.c_void_p, 8]
  pd: Annotated[c.POINTER[struct_ibv_pd], 16]
  send_cq: Annotated[c.POINTER[struct_ibv_cq], 24]
  recv_cq: Annotated[c.POINTER[struct_ibv_cq], 32]
  srq: Annotated[c.POINTER[struct_ibv_srq], 40]
  handle: Annotated[uint32_t, 48]
  qp_num: Annotated[uint32_t, 52]
  state: Annotated[enum_ibv_qp_state, 56]
  qp_type: Annotated[enum_ibv_qp_type, 60]
  mutex: Annotated[pthread_mutex_t, 64]
  cond: Annotated[pthread_cond_t, 104]
  events_completed: Annotated[uint32_t, 152]
@c.record
class struct_ibv_cq(c.Struct):
  SIZE = 128
  context: Annotated[c.POINTER[struct_ibv_context], 0]
  channel: Annotated[c.POINTER[struct_ibv_comp_channel], 8]
  cq_context: Annotated[ctypes.c_void_p, 16]
  handle: Annotated[uint32_t, 24]
  cqe: Annotated[Annotated[int, ctypes.c_int32], 28]
  mutex: Annotated[pthread_mutex_t, 32]
  cond: Annotated[pthread_cond_t, 72]
  comp_events_completed: Annotated[uint32_t, 120]
  async_events_completed: Annotated[uint32_t, 124]
@c.record
class struct_ibv_comp_channel(c.Struct):
  SIZE = 16
  context: Annotated[c.POINTER[struct_ibv_context], 0]
  fd: Annotated[Annotated[int, ctypes.c_int32], 8]
  refcnt: Annotated[Annotated[int, ctypes.c_int32], 12]
@c.record
class pthread_mutex_t(c.Struct):
  SIZE = 40
  __data: Annotated[struct___pthread_mutex_s, 0]
  __size: Annotated[c.Array[Annotated[bytes, ctypes.c_char], Literal[40]], 0]
  __align: Annotated[Annotated[int, ctypes.c_int64], 0]
@c.record
class struct___pthread_mutex_s(c.Struct):
  SIZE = 40
  __lock: Annotated[Annotated[int, ctypes.c_int32], 0]
  __count: Annotated[Annotated[int, ctypes.c_uint32], 4]
  __owner: Annotated[Annotated[int, ctypes.c_int32], 8]
  __nusers: Annotated[Annotated[int, ctypes.c_uint32], 12]
  __kind: Annotated[Annotated[int, ctypes.c_int32], 16]
  __spins: Annotated[Annotated[int, ctypes.c_int16], 20]
  __elision: Annotated[Annotated[int, ctypes.c_int16], 22]
  __list: Annotated[struct___pthread_internal_list, 24]
@c.record
class struct___pthread_internal_list(c.Struct):
  SIZE = 16
  __prev: Annotated[c.POINTER[struct___pthread_internal_list], 0]
  __next: Annotated[c.POINTER[struct___pthread_internal_list], 8]
__pthread_list_t: TypeAlias = struct___pthread_internal_list
@c.record
class pthread_cond_t(c.Struct):
  SIZE = 48
  __data: Annotated[struct___pthread_cond_s, 0]
  __size: Annotated[c.Array[Annotated[bytes, ctypes.c_char], Literal[48]], 0]
  __align: Annotated[Annotated[int, ctypes.c_int64], 0]
@c.record
class struct___pthread_cond_s(c.Struct):
  SIZE = 48
  __wseq: Annotated[__atomic_wide_counter, 0]
  __g1_start: Annotated[__atomic_wide_counter, 8]
  __g_refs: Annotated[c.Array[Annotated[int, ctypes.c_uint32], Literal[2]], 16]
  __g_size: Annotated[c.Array[Annotated[int, ctypes.c_uint32], Literal[2]], 24]
  __g1_orig_size: Annotated[Annotated[int, ctypes.c_uint32], 32]
  __wrefs: Annotated[Annotated[int, ctypes.c_uint32], 36]
  __g_signals: Annotated[c.Array[Annotated[int, ctypes.c_uint32], Literal[2]], 40]
@c.record
class __atomic_wide_counter(c.Struct):
  SIZE = 8
  __value64: Annotated[Annotated[int, ctypes.c_uint64], 0]
  __value32: Annotated[__atomic_wide_counter___value32, 0]
@c.record
class __atomic_wide_counter___value32(c.Struct):
  SIZE = 8
  __low: Annotated[Annotated[int, ctypes.c_uint32], 0]
  __high: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_ibv_srq(c.Struct):
  SIZE = 128
  context: Annotated[c.POINTER[struct_ibv_context], 0]
  srq_context: Annotated[ctypes.c_void_p, 8]
  pd: Annotated[c.POINTER[struct_ibv_pd], 16]
  handle: Annotated[uint32_t, 24]
  mutex: Annotated[pthread_mutex_t, 32]
  cond: Annotated[pthread_cond_t, 72]
  events_completed: Annotated[uint32_t, 120]
class enum_ibv_qp_state(Annotated[int, ctypes.c_uint32], c.Enum): pass
IBV_QPS_RESET = enum_ibv_qp_state.define('IBV_QPS_RESET', 0)
IBV_QPS_INIT = enum_ibv_qp_state.define('IBV_QPS_INIT', 1)
IBV_QPS_RTR = enum_ibv_qp_state.define('IBV_QPS_RTR', 2)
IBV_QPS_RTS = enum_ibv_qp_state.define('IBV_QPS_RTS', 3)
IBV_QPS_SQD = enum_ibv_qp_state.define('IBV_QPS_SQD', 4)
IBV_QPS_SQE = enum_ibv_qp_state.define('IBV_QPS_SQE', 5)
IBV_QPS_ERR = enum_ibv_qp_state.define('IBV_QPS_ERR', 6)
IBV_QPS_UNKNOWN = enum_ibv_qp_state.define('IBV_QPS_UNKNOWN', 7)

class enum_ibv_qp_type(Annotated[int, ctypes.c_uint32], c.Enum): pass
IBV_QPT_RC = enum_ibv_qp_type.define('IBV_QPT_RC', 2)
IBV_QPT_UC = enum_ibv_qp_type.define('IBV_QPT_UC', 3)
IBV_QPT_UD = enum_ibv_qp_type.define('IBV_QPT_UD', 4)
IBV_QPT_RAW_PACKET = enum_ibv_qp_type.define('IBV_QPT_RAW_PACKET', 8)
IBV_QPT_XRC_SEND = enum_ibv_qp_type.define('IBV_QPT_XRC_SEND', 9)
IBV_QPT_XRC_RECV = enum_ibv_qp_type.define('IBV_QPT_XRC_RECV', 10)
IBV_QPT_DRIVER = enum_ibv_qp_type.define('IBV_QPT_DRIVER', 255)

@c.record
class struct_ibv_mw_bind(c.Struct):
  SIZE = 48
  wr_id: Annotated[uint64_t, 0]
  send_flags: Annotated[Annotated[int, ctypes.c_uint32], 8]
  bind_info: Annotated[struct_ibv_mw_bind_info, 16]
@c.record
class struct_ibv_mw_bind_info(c.Struct):
  SIZE = 32
  mr: Annotated[c.POINTER[struct_ibv_mr], 0]
  addr: Annotated[uint64_t, 8]
  length: Annotated[uint64_t, 16]
  mw_access_flags: Annotated[Annotated[int, ctypes.c_uint32], 24]
@c.record
class struct_ibv_mr(c.Struct):
  SIZE = 48
  context: Annotated[c.POINTER[struct_ibv_context], 0]
  pd: Annotated[c.POINTER[struct_ibv_pd], 8]
  addr: Annotated[ctypes.c_void_p, 16]
  length: Annotated[size_t, 24]
  handle: Annotated[uint32_t, 32]
  lkey: Annotated[uint32_t, 36]
  rkey: Annotated[uint32_t, 40]
@c.record
class struct_ibv_wc(c.Struct):
  SIZE = 48
  wr_id: Annotated[uint64_t, 0]
  status: Annotated[enum_ibv_wc_status, 8]
  opcode: Annotated[enum_ibv_wc_opcode, 12]
  vendor_err: Annotated[uint32_t, 16]
  byte_len: Annotated[uint32_t, 20]
  imm_data: Annotated[Annotated[int, ctypes.c_uint32], 24]
  invalidated_rkey: Annotated[uint32_t, 24]
  qp_num: Annotated[uint32_t, 28]
  src_qp: Annotated[uint32_t, 32]
  wc_flags: Annotated[Annotated[int, ctypes.c_uint32], 36]
  pkey_index: Annotated[uint16_t, 40]
  slid: Annotated[uint16_t, 42]
  sl: Annotated[uint8_t, 44]
  dlid_path_bits: Annotated[uint8_t, 45]
class enum_ibv_wc_status(Annotated[int, ctypes.c_uint32], c.Enum): pass
IBV_WC_SUCCESS = enum_ibv_wc_status.define('IBV_WC_SUCCESS', 0)
IBV_WC_LOC_LEN_ERR = enum_ibv_wc_status.define('IBV_WC_LOC_LEN_ERR', 1)
IBV_WC_LOC_QP_OP_ERR = enum_ibv_wc_status.define('IBV_WC_LOC_QP_OP_ERR', 2)
IBV_WC_LOC_EEC_OP_ERR = enum_ibv_wc_status.define('IBV_WC_LOC_EEC_OP_ERR', 3)
IBV_WC_LOC_PROT_ERR = enum_ibv_wc_status.define('IBV_WC_LOC_PROT_ERR', 4)
IBV_WC_WR_FLUSH_ERR = enum_ibv_wc_status.define('IBV_WC_WR_FLUSH_ERR', 5)
IBV_WC_MW_BIND_ERR = enum_ibv_wc_status.define('IBV_WC_MW_BIND_ERR', 6)
IBV_WC_BAD_RESP_ERR = enum_ibv_wc_status.define('IBV_WC_BAD_RESP_ERR', 7)
IBV_WC_LOC_ACCESS_ERR = enum_ibv_wc_status.define('IBV_WC_LOC_ACCESS_ERR', 8)
IBV_WC_REM_INV_REQ_ERR = enum_ibv_wc_status.define('IBV_WC_REM_INV_REQ_ERR', 9)
IBV_WC_REM_ACCESS_ERR = enum_ibv_wc_status.define('IBV_WC_REM_ACCESS_ERR', 10)
IBV_WC_REM_OP_ERR = enum_ibv_wc_status.define('IBV_WC_REM_OP_ERR', 11)
IBV_WC_RETRY_EXC_ERR = enum_ibv_wc_status.define('IBV_WC_RETRY_EXC_ERR', 12)
IBV_WC_RNR_RETRY_EXC_ERR = enum_ibv_wc_status.define('IBV_WC_RNR_RETRY_EXC_ERR', 13)
IBV_WC_LOC_RDD_VIOL_ERR = enum_ibv_wc_status.define('IBV_WC_LOC_RDD_VIOL_ERR', 14)
IBV_WC_REM_INV_RD_REQ_ERR = enum_ibv_wc_status.define('IBV_WC_REM_INV_RD_REQ_ERR', 15)
IBV_WC_REM_ABORT_ERR = enum_ibv_wc_status.define('IBV_WC_REM_ABORT_ERR', 16)
IBV_WC_INV_EECN_ERR = enum_ibv_wc_status.define('IBV_WC_INV_EECN_ERR', 17)
IBV_WC_INV_EEC_STATE_ERR = enum_ibv_wc_status.define('IBV_WC_INV_EEC_STATE_ERR', 18)
IBV_WC_FATAL_ERR = enum_ibv_wc_status.define('IBV_WC_FATAL_ERR', 19)
IBV_WC_RESP_TIMEOUT_ERR = enum_ibv_wc_status.define('IBV_WC_RESP_TIMEOUT_ERR', 20)
IBV_WC_GENERAL_ERR = enum_ibv_wc_status.define('IBV_WC_GENERAL_ERR', 21)
IBV_WC_TM_ERR = enum_ibv_wc_status.define('IBV_WC_TM_ERR', 22)
IBV_WC_TM_RNDV_INCOMPLETE = enum_ibv_wc_status.define('IBV_WC_TM_RNDV_INCOMPLETE', 23)

class enum_ibv_wc_opcode(Annotated[int, ctypes.c_uint32], c.Enum): pass
IBV_WC_SEND = enum_ibv_wc_opcode.define('IBV_WC_SEND', 0)
IBV_WC_RDMA_WRITE = enum_ibv_wc_opcode.define('IBV_WC_RDMA_WRITE', 1)
IBV_WC_RDMA_READ = enum_ibv_wc_opcode.define('IBV_WC_RDMA_READ', 2)
IBV_WC_COMP_SWAP = enum_ibv_wc_opcode.define('IBV_WC_COMP_SWAP', 3)
IBV_WC_FETCH_ADD = enum_ibv_wc_opcode.define('IBV_WC_FETCH_ADD', 4)
IBV_WC_BIND_MW = enum_ibv_wc_opcode.define('IBV_WC_BIND_MW', 5)
IBV_WC_LOCAL_INV = enum_ibv_wc_opcode.define('IBV_WC_LOCAL_INV', 6)
IBV_WC_TSO = enum_ibv_wc_opcode.define('IBV_WC_TSO', 7)
IBV_WC_FLUSH = enum_ibv_wc_opcode.define('IBV_WC_FLUSH', 8)
IBV_WC_ATOMIC_WRITE = enum_ibv_wc_opcode.define('IBV_WC_ATOMIC_WRITE', 9)
IBV_WC_RECV = enum_ibv_wc_opcode.define('IBV_WC_RECV', 128)
IBV_WC_RECV_RDMA_WITH_IMM = enum_ibv_wc_opcode.define('IBV_WC_RECV_RDMA_WITH_IMM', 129)
IBV_WC_TM_ADD = enum_ibv_wc_opcode.define('IBV_WC_TM_ADD', 130)
IBV_WC_TM_DEL = enum_ibv_wc_opcode.define('IBV_WC_TM_DEL', 131)
IBV_WC_TM_SYNC = enum_ibv_wc_opcode.define('IBV_WC_TM_SYNC', 132)
IBV_WC_TM_RECV = enum_ibv_wc_opcode.define('IBV_WC_TM_RECV', 133)
IBV_WC_TM_NO_TAG = enum_ibv_wc_opcode.define('IBV_WC_TM_NO_TAG', 134)
IBV_WC_DRIVER1 = enum_ibv_wc_opcode.define('IBV_WC_DRIVER1', 135)
IBV_WC_DRIVER2 = enum_ibv_wc_opcode.define('IBV_WC_DRIVER2', 136)
IBV_WC_DRIVER3 = enum_ibv_wc_opcode.define('IBV_WC_DRIVER3', 137)

__be32: TypeAlias = Annotated[int, ctypes.c_uint32]
@c.record
class struct_ibv_recv_wr(c.Struct):
  SIZE = 32
  wr_id: Annotated[uint64_t, 0]
  next: Annotated[c.POINTER[struct_ibv_recv_wr], 8]
  sg_list: Annotated[c.POINTER[struct_ibv_sge], 16]
  num_sge: Annotated[Annotated[int, ctypes.c_int32], 24]
@c.record
class struct_ibv_sge(c.Struct):
  SIZE = 16
  addr: Annotated[uint64_t, 0]
  length: Annotated[uint32_t, 8]
  lkey: Annotated[uint32_t, 12]
@c.record
class struct_ibv_send_wr(c.Struct):
  SIZE = 128
  wr_id: Annotated[uint64_t, 0]
  next: Annotated[c.POINTER[struct_ibv_send_wr], 8]
  sg_list: Annotated[c.POINTER[struct_ibv_sge], 16]
  num_sge: Annotated[Annotated[int, ctypes.c_int32], 24]
  opcode: Annotated[enum_ibv_wr_opcode, 28]
  send_flags: Annotated[Annotated[int, ctypes.c_uint32], 32]
  imm_data: Annotated[Annotated[int, ctypes.c_uint32], 36]
  invalidate_rkey: Annotated[uint32_t, 36]
  wr: Annotated[struct_ibv_send_wr_wr, 40]
  qp_type: Annotated[struct_ibv_send_wr_qp_type, 72]
  bind_mw: Annotated[struct_ibv_send_wr_bind_mw, 80]
  tso: Annotated[struct_ibv_send_wr_tso, 80]
class enum_ibv_wr_opcode(Annotated[int, ctypes.c_uint32], c.Enum): pass
IBV_WR_RDMA_WRITE = enum_ibv_wr_opcode.define('IBV_WR_RDMA_WRITE', 0)
IBV_WR_RDMA_WRITE_WITH_IMM = enum_ibv_wr_opcode.define('IBV_WR_RDMA_WRITE_WITH_IMM', 1)
IBV_WR_SEND = enum_ibv_wr_opcode.define('IBV_WR_SEND', 2)
IBV_WR_SEND_WITH_IMM = enum_ibv_wr_opcode.define('IBV_WR_SEND_WITH_IMM', 3)
IBV_WR_RDMA_READ = enum_ibv_wr_opcode.define('IBV_WR_RDMA_READ', 4)
IBV_WR_ATOMIC_CMP_AND_SWP = enum_ibv_wr_opcode.define('IBV_WR_ATOMIC_CMP_AND_SWP', 5)
IBV_WR_ATOMIC_FETCH_AND_ADD = enum_ibv_wr_opcode.define('IBV_WR_ATOMIC_FETCH_AND_ADD', 6)
IBV_WR_LOCAL_INV = enum_ibv_wr_opcode.define('IBV_WR_LOCAL_INV', 7)
IBV_WR_BIND_MW = enum_ibv_wr_opcode.define('IBV_WR_BIND_MW', 8)
IBV_WR_SEND_WITH_INV = enum_ibv_wr_opcode.define('IBV_WR_SEND_WITH_INV', 9)
IBV_WR_TSO = enum_ibv_wr_opcode.define('IBV_WR_TSO', 10)
IBV_WR_DRIVER1 = enum_ibv_wr_opcode.define('IBV_WR_DRIVER1', 11)
IBV_WR_FLUSH = enum_ibv_wr_opcode.define('IBV_WR_FLUSH', 14)
IBV_WR_ATOMIC_WRITE = enum_ibv_wr_opcode.define('IBV_WR_ATOMIC_WRITE', 15)

@c.record
class struct_ibv_send_wr_wr(c.Struct):
  SIZE = 32
  rdma: Annotated[struct_ibv_send_wr_wr_rdma, 0]
  atomic: Annotated[struct_ibv_send_wr_wr_atomic, 0]
  ud: Annotated[struct_ibv_send_wr_wr_ud, 0]
@c.record
class struct_ibv_send_wr_wr_rdma(c.Struct):
  SIZE = 16
  remote_addr: Annotated[uint64_t, 0]
  rkey: Annotated[uint32_t, 8]
@c.record
class struct_ibv_send_wr_wr_atomic(c.Struct):
  SIZE = 32
  remote_addr: Annotated[uint64_t, 0]
  compare_add: Annotated[uint64_t, 8]
  swap: Annotated[uint64_t, 16]
  rkey: Annotated[uint32_t, 24]
@c.record
class struct_ibv_send_wr_wr_ud(c.Struct):
  SIZE = 16
  ah: Annotated[c.POINTER[struct_ibv_ah], 0]
  remote_qpn: Annotated[uint32_t, 8]
  remote_qkey: Annotated[uint32_t, 12]
@c.record
class struct_ibv_ah(c.Struct):
  SIZE = 24
  context: Annotated[c.POINTER[struct_ibv_context], 0]
  pd: Annotated[c.POINTER[struct_ibv_pd], 8]
  handle: Annotated[uint32_t, 16]
@c.record
class struct_ibv_send_wr_qp_type(c.Struct):
  SIZE = 4
  xrc: Annotated[struct_ibv_send_wr_qp_type_xrc, 0]
@c.record
class struct_ibv_send_wr_qp_type_xrc(c.Struct):
  SIZE = 4
  remote_srqn: Annotated[uint32_t, 0]
@c.record
class struct_ibv_send_wr_bind_mw(c.Struct):
  SIZE = 48
  mw: Annotated[c.POINTER[struct_ibv_mw], 0]
  rkey: Annotated[uint32_t, 8]
  bind_info: Annotated[struct_ibv_mw_bind_info, 16]
@c.record
class struct_ibv_send_wr_tso(c.Struct):
  SIZE = 16
  hdr: Annotated[ctypes.c_void_p, 0]
  hdr_sz: Annotated[uint16_t, 8]
  mss: Annotated[uint16_t, 10]
@c.record
class struct_ibv_query_device_ex_input(c.Struct):
  SIZE = 4
  comp_mask: Annotated[uint32_t, 0]
class enum_ibv_odp_transport_cap_bits(Annotated[int, ctypes.c_uint32], c.Enum): pass
IBV_ODP_SUPPORT_SEND = enum_ibv_odp_transport_cap_bits.define('IBV_ODP_SUPPORT_SEND', 1)
IBV_ODP_SUPPORT_RECV = enum_ibv_odp_transport_cap_bits.define('IBV_ODP_SUPPORT_RECV', 2)
IBV_ODP_SUPPORT_WRITE = enum_ibv_odp_transport_cap_bits.define('IBV_ODP_SUPPORT_WRITE', 4)
IBV_ODP_SUPPORT_READ = enum_ibv_odp_transport_cap_bits.define('IBV_ODP_SUPPORT_READ', 8)
IBV_ODP_SUPPORT_ATOMIC = enum_ibv_odp_transport_cap_bits.define('IBV_ODP_SUPPORT_ATOMIC', 16)
IBV_ODP_SUPPORT_SRQ_RECV = enum_ibv_odp_transport_cap_bits.define('IBV_ODP_SUPPORT_SRQ_RECV', 32)

@c.record
class struct_ibv_odp_caps(c.Struct):
  SIZE = 24
  general_caps: Annotated[uint64_t, 0]
  per_transport_caps: Annotated[struct_ibv_odp_caps_per_transport_caps, 8]
@c.record
class struct_ibv_odp_caps_per_transport_caps(c.Struct):
  SIZE = 12
  rc_odp_caps: Annotated[uint32_t, 0]
  uc_odp_caps: Annotated[uint32_t, 4]
  ud_odp_caps: Annotated[uint32_t, 8]
class enum_ibv_odp_general_caps(Annotated[int, ctypes.c_uint32], c.Enum): pass
IBV_ODP_SUPPORT = enum_ibv_odp_general_caps.define('IBV_ODP_SUPPORT', 1)
IBV_ODP_SUPPORT_IMPLICIT = enum_ibv_odp_general_caps.define('IBV_ODP_SUPPORT_IMPLICIT', 2)

@c.record
class struct_ibv_tso_caps(c.Struct):
  SIZE = 8
  max_tso: Annotated[uint32_t, 0]
  supported_qpts: Annotated[uint32_t, 4]
class enum_ibv_rx_hash_function_flags(Annotated[int, ctypes.c_uint32], c.Enum): pass
IBV_RX_HASH_FUNC_TOEPLITZ = enum_ibv_rx_hash_function_flags.define('IBV_RX_HASH_FUNC_TOEPLITZ', 1)

class enum_ibv_rx_hash_fields(Annotated[int, ctypes.c_uint32], c.Enum): pass
IBV_RX_HASH_SRC_IPV4 = enum_ibv_rx_hash_fields.define('IBV_RX_HASH_SRC_IPV4', 1)
IBV_RX_HASH_DST_IPV4 = enum_ibv_rx_hash_fields.define('IBV_RX_HASH_DST_IPV4', 2)
IBV_RX_HASH_SRC_IPV6 = enum_ibv_rx_hash_fields.define('IBV_RX_HASH_SRC_IPV6', 4)
IBV_RX_HASH_DST_IPV6 = enum_ibv_rx_hash_fields.define('IBV_RX_HASH_DST_IPV6', 8)
IBV_RX_HASH_SRC_PORT_TCP = enum_ibv_rx_hash_fields.define('IBV_RX_HASH_SRC_PORT_TCP', 16)
IBV_RX_HASH_DST_PORT_TCP = enum_ibv_rx_hash_fields.define('IBV_RX_HASH_DST_PORT_TCP', 32)
IBV_RX_HASH_SRC_PORT_UDP = enum_ibv_rx_hash_fields.define('IBV_RX_HASH_SRC_PORT_UDP', 64)
IBV_RX_HASH_DST_PORT_UDP = enum_ibv_rx_hash_fields.define('IBV_RX_HASH_DST_PORT_UDP', 128)
IBV_RX_HASH_IPSEC_SPI = enum_ibv_rx_hash_fields.define('IBV_RX_HASH_IPSEC_SPI', 256)
IBV_RX_HASH_INNER = enum_ibv_rx_hash_fields.define('IBV_RX_HASH_INNER', 2147483648)

@c.record
class struct_ibv_rss_caps(c.Struct):
  SIZE = 32
  supported_qpts: Annotated[uint32_t, 0]
  max_rwq_indirection_tables: Annotated[uint32_t, 4]
  max_rwq_indirection_table_size: Annotated[uint32_t, 8]
  rx_hash_fields_mask: Annotated[uint64_t, 16]
  rx_hash_function: Annotated[uint8_t, 24]
@c.record
class struct_ibv_packet_pacing_caps(c.Struct):
  SIZE = 12
  qp_rate_limit_min: Annotated[uint32_t, 0]
  qp_rate_limit_max: Annotated[uint32_t, 4]
  supported_qpts: Annotated[uint32_t, 8]
class enum_ibv_raw_packet_caps(Annotated[int, ctypes.c_uint32], c.Enum): pass
IBV_RAW_PACKET_CAP_CVLAN_STRIPPING = enum_ibv_raw_packet_caps.define('IBV_RAW_PACKET_CAP_CVLAN_STRIPPING', 1)
IBV_RAW_PACKET_CAP_SCATTER_FCS = enum_ibv_raw_packet_caps.define('IBV_RAW_PACKET_CAP_SCATTER_FCS', 2)
IBV_RAW_PACKET_CAP_IP_CSUM = enum_ibv_raw_packet_caps.define('IBV_RAW_PACKET_CAP_IP_CSUM', 4)
IBV_RAW_PACKET_CAP_DELAY_DROP = enum_ibv_raw_packet_caps.define('IBV_RAW_PACKET_CAP_DELAY_DROP', 8)

class enum_ibv_tm_cap_flags(Annotated[int, ctypes.c_uint32], c.Enum): pass
IBV_TM_CAP_RC = enum_ibv_tm_cap_flags.define('IBV_TM_CAP_RC', 1)

@c.record
class struct_ibv_tm_caps(c.Struct):
  SIZE = 20
  max_rndv_hdr_size: Annotated[uint32_t, 0]
  max_num_tags: Annotated[uint32_t, 4]
  flags: Annotated[uint32_t, 8]
  max_ops: Annotated[uint32_t, 12]
  max_sge: Annotated[uint32_t, 16]
@c.record
class struct_ibv_cq_moderation_caps(c.Struct):
  SIZE = 4
  max_cq_count: Annotated[uint16_t, 0]
  max_cq_period: Annotated[uint16_t, 2]
class enum_ibv_pci_atomic_op_size(Annotated[int, ctypes.c_uint32], c.Enum): pass
IBV_PCI_ATOMIC_OPERATION_4_BYTE_SIZE_SUP = enum_ibv_pci_atomic_op_size.define('IBV_PCI_ATOMIC_OPERATION_4_BYTE_SIZE_SUP', 1)
IBV_PCI_ATOMIC_OPERATION_8_BYTE_SIZE_SUP = enum_ibv_pci_atomic_op_size.define('IBV_PCI_ATOMIC_OPERATION_8_BYTE_SIZE_SUP', 2)
IBV_PCI_ATOMIC_OPERATION_16_BYTE_SIZE_SUP = enum_ibv_pci_atomic_op_size.define('IBV_PCI_ATOMIC_OPERATION_16_BYTE_SIZE_SUP', 4)

@c.record
class struct_ibv_pci_atomic_caps(c.Struct):
  SIZE = 6
  fetch_add: Annotated[uint16_t, 0]
  swap: Annotated[uint16_t, 2]
  compare_swap: Annotated[uint16_t, 4]
@c.record
class struct_ibv_device_attr_ex(c.Struct):
  SIZE = 400
  orig_attr: Annotated[struct_ibv_device_attr, 0]
  comp_mask: Annotated[uint32_t, 232]
  odp_caps: Annotated[struct_ibv_odp_caps, 240]
  completion_timestamp_mask: Annotated[uint64_t, 264]
  hca_core_clock: Annotated[uint64_t, 272]
  device_cap_flags_ex: Annotated[uint64_t, 280]
  tso_caps: Annotated[struct_ibv_tso_caps, 288]
  rss_caps: Annotated[struct_ibv_rss_caps, 296]
  max_wq_type_rq: Annotated[uint32_t, 328]
  packet_pacing_caps: Annotated[struct_ibv_packet_pacing_caps, 332]
  raw_packet_caps: Annotated[uint32_t, 344]
  tm_caps: Annotated[struct_ibv_tm_caps, 348]
  cq_mod_caps: Annotated[struct_ibv_cq_moderation_caps, 368]
  max_dm_size: Annotated[uint64_t, 376]
  pci_atomic_caps: Annotated[struct_ibv_pci_atomic_caps, 384]
  xrc_odp_caps: Annotated[uint32_t, 392]
  phys_port_cnt_ex: Annotated[uint32_t, 396]
class enum_ibv_mtu(Annotated[int, ctypes.c_uint32], c.Enum): pass
IBV_MTU_256 = enum_ibv_mtu.define('IBV_MTU_256', 1)
IBV_MTU_512 = enum_ibv_mtu.define('IBV_MTU_512', 2)
IBV_MTU_1024 = enum_ibv_mtu.define('IBV_MTU_1024', 3)
IBV_MTU_2048 = enum_ibv_mtu.define('IBV_MTU_2048', 4)
IBV_MTU_4096 = enum_ibv_mtu.define('IBV_MTU_4096', 5)

class enum_ibv_port_state(Annotated[int, ctypes.c_uint32], c.Enum): pass
IBV_PORT_NOP = enum_ibv_port_state.define('IBV_PORT_NOP', 0)
IBV_PORT_DOWN = enum_ibv_port_state.define('IBV_PORT_DOWN', 1)
IBV_PORT_INIT = enum_ibv_port_state.define('IBV_PORT_INIT', 2)
IBV_PORT_ARMED = enum_ibv_port_state.define('IBV_PORT_ARMED', 3)
IBV_PORT_ACTIVE = enum_ibv_port_state.define('IBV_PORT_ACTIVE', 4)
IBV_PORT_ACTIVE_DEFER = enum_ibv_port_state.define('IBV_PORT_ACTIVE_DEFER', 5)

class _anonenum0(Annotated[int, ctypes.c_uint32], c.Enum): pass
IBV_LINK_LAYER_UNSPECIFIED = _anonenum0.define('IBV_LINK_LAYER_UNSPECIFIED', 0)
IBV_LINK_LAYER_INFINIBAND = _anonenum0.define('IBV_LINK_LAYER_INFINIBAND', 1)
IBV_LINK_LAYER_ETHERNET = _anonenum0.define('IBV_LINK_LAYER_ETHERNET', 2)

class enum_ibv_port_cap_flags(Annotated[int, ctypes.c_uint32], c.Enum): pass
IBV_PORT_SM = enum_ibv_port_cap_flags.define('IBV_PORT_SM', 2)
IBV_PORT_NOTICE_SUP = enum_ibv_port_cap_flags.define('IBV_PORT_NOTICE_SUP', 4)
IBV_PORT_TRAP_SUP = enum_ibv_port_cap_flags.define('IBV_PORT_TRAP_SUP', 8)
IBV_PORT_OPT_IPD_SUP = enum_ibv_port_cap_flags.define('IBV_PORT_OPT_IPD_SUP', 16)
IBV_PORT_AUTO_MIGR_SUP = enum_ibv_port_cap_flags.define('IBV_PORT_AUTO_MIGR_SUP', 32)
IBV_PORT_SL_MAP_SUP = enum_ibv_port_cap_flags.define('IBV_PORT_SL_MAP_SUP', 64)
IBV_PORT_MKEY_NVRAM = enum_ibv_port_cap_flags.define('IBV_PORT_MKEY_NVRAM', 128)
IBV_PORT_PKEY_NVRAM = enum_ibv_port_cap_flags.define('IBV_PORT_PKEY_NVRAM', 256)
IBV_PORT_LED_INFO_SUP = enum_ibv_port_cap_flags.define('IBV_PORT_LED_INFO_SUP', 512)
IBV_PORT_SYS_IMAGE_GUID_SUP = enum_ibv_port_cap_flags.define('IBV_PORT_SYS_IMAGE_GUID_SUP', 2048)
IBV_PORT_PKEY_SW_EXT_PORT_TRAP_SUP = enum_ibv_port_cap_flags.define('IBV_PORT_PKEY_SW_EXT_PORT_TRAP_SUP', 4096)
IBV_PORT_EXTENDED_SPEEDS_SUP = enum_ibv_port_cap_flags.define('IBV_PORT_EXTENDED_SPEEDS_SUP', 16384)
IBV_PORT_CAP_MASK2_SUP = enum_ibv_port_cap_flags.define('IBV_PORT_CAP_MASK2_SUP', 32768)
IBV_PORT_CM_SUP = enum_ibv_port_cap_flags.define('IBV_PORT_CM_SUP', 65536)
IBV_PORT_SNMP_TUNNEL_SUP = enum_ibv_port_cap_flags.define('IBV_PORT_SNMP_TUNNEL_SUP', 131072)
IBV_PORT_REINIT_SUP = enum_ibv_port_cap_flags.define('IBV_PORT_REINIT_SUP', 262144)
IBV_PORT_DEVICE_MGMT_SUP = enum_ibv_port_cap_flags.define('IBV_PORT_DEVICE_MGMT_SUP', 524288)
IBV_PORT_VENDOR_CLASS_SUP = enum_ibv_port_cap_flags.define('IBV_PORT_VENDOR_CLASS_SUP', 1048576)
IBV_PORT_DR_NOTICE_SUP = enum_ibv_port_cap_flags.define('IBV_PORT_DR_NOTICE_SUP', 2097152)
IBV_PORT_CAP_MASK_NOTICE_SUP = enum_ibv_port_cap_flags.define('IBV_PORT_CAP_MASK_NOTICE_SUP', 4194304)
IBV_PORT_BOOT_MGMT_SUP = enum_ibv_port_cap_flags.define('IBV_PORT_BOOT_MGMT_SUP', 8388608)
IBV_PORT_LINK_LATENCY_SUP = enum_ibv_port_cap_flags.define('IBV_PORT_LINK_LATENCY_SUP', 16777216)
IBV_PORT_CLIENT_REG_SUP = enum_ibv_port_cap_flags.define('IBV_PORT_CLIENT_REG_SUP', 33554432)
IBV_PORT_IP_BASED_GIDS = enum_ibv_port_cap_flags.define('IBV_PORT_IP_BASED_GIDS', 67108864)

class enum_ibv_port_cap_flags2(Annotated[int, ctypes.c_uint32], c.Enum): pass
IBV_PORT_SET_NODE_DESC_SUP = enum_ibv_port_cap_flags2.define('IBV_PORT_SET_NODE_DESC_SUP', 1)
IBV_PORT_INFO_EXT_SUP = enum_ibv_port_cap_flags2.define('IBV_PORT_INFO_EXT_SUP', 2)
IBV_PORT_VIRT_SUP = enum_ibv_port_cap_flags2.define('IBV_PORT_VIRT_SUP', 4)
IBV_PORT_SWITCH_PORT_STATE_TABLE_SUP = enum_ibv_port_cap_flags2.define('IBV_PORT_SWITCH_PORT_STATE_TABLE_SUP', 8)
IBV_PORT_LINK_WIDTH_2X_SUP = enum_ibv_port_cap_flags2.define('IBV_PORT_LINK_WIDTH_2X_SUP', 16)
IBV_PORT_LINK_SPEED_HDR_SUP = enum_ibv_port_cap_flags2.define('IBV_PORT_LINK_SPEED_HDR_SUP', 32)
IBV_PORT_LINK_SPEED_NDR_SUP = enum_ibv_port_cap_flags2.define('IBV_PORT_LINK_SPEED_NDR_SUP', 1024)
IBV_PORT_LINK_SPEED_XDR_SUP = enum_ibv_port_cap_flags2.define('IBV_PORT_LINK_SPEED_XDR_SUP', 4096)

@c.record
class struct_ibv_port_attr(c.Struct):
  SIZE = 56
  state: Annotated[enum_ibv_port_state, 0]
  max_mtu: Annotated[enum_ibv_mtu, 4]
  active_mtu: Annotated[enum_ibv_mtu, 8]
  gid_tbl_len: Annotated[Annotated[int, ctypes.c_int32], 12]
  port_cap_flags: Annotated[uint32_t, 16]
  max_msg_sz: Annotated[uint32_t, 20]
  bad_pkey_cntr: Annotated[uint32_t, 24]
  qkey_viol_cntr: Annotated[uint32_t, 28]
  pkey_tbl_len: Annotated[uint16_t, 32]
  lid: Annotated[uint16_t, 34]
  sm_lid: Annotated[uint16_t, 36]
  lmc: Annotated[uint8_t, 38]
  max_vl_num: Annotated[uint8_t, 39]
  sm_sl: Annotated[uint8_t, 40]
  subnet_timeout: Annotated[uint8_t, 41]
  init_type_reply: Annotated[uint8_t, 42]
  active_width: Annotated[uint8_t, 43]
  active_speed: Annotated[uint8_t, 44]
  phys_state: Annotated[uint8_t, 45]
  link_layer: Annotated[uint8_t, 46]
  flags: Annotated[uint8_t, 47]
  port_cap_flags2: Annotated[uint16_t, 48]
  active_speed_ex: Annotated[uint32_t, 52]
class enum_ibv_event_type(Annotated[int, ctypes.c_uint32], c.Enum): pass
IBV_EVENT_CQ_ERR = enum_ibv_event_type.define('IBV_EVENT_CQ_ERR', 0)
IBV_EVENT_QP_FATAL = enum_ibv_event_type.define('IBV_EVENT_QP_FATAL', 1)
IBV_EVENT_QP_REQ_ERR = enum_ibv_event_type.define('IBV_EVENT_QP_REQ_ERR', 2)
IBV_EVENT_QP_ACCESS_ERR = enum_ibv_event_type.define('IBV_EVENT_QP_ACCESS_ERR', 3)
IBV_EVENT_COMM_EST = enum_ibv_event_type.define('IBV_EVENT_COMM_EST', 4)
IBV_EVENT_SQ_DRAINED = enum_ibv_event_type.define('IBV_EVENT_SQ_DRAINED', 5)
IBV_EVENT_PATH_MIG = enum_ibv_event_type.define('IBV_EVENT_PATH_MIG', 6)
IBV_EVENT_PATH_MIG_ERR = enum_ibv_event_type.define('IBV_EVENT_PATH_MIG_ERR', 7)
IBV_EVENT_DEVICE_FATAL = enum_ibv_event_type.define('IBV_EVENT_DEVICE_FATAL', 8)
IBV_EVENT_PORT_ACTIVE = enum_ibv_event_type.define('IBV_EVENT_PORT_ACTIVE', 9)
IBV_EVENT_PORT_ERR = enum_ibv_event_type.define('IBV_EVENT_PORT_ERR', 10)
IBV_EVENT_LID_CHANGE = enum_ibv_event_type.define('IBV_EVENT_LID_CHANGE', 11)
IBV_EVENT_PKEY_CHANGE = enum_ibv_event_type.define('IBV_EVENT_PKEY_CHANGE', 12)
IBV_EVENT_SM_CHANGE = enum_ibv_event_type.define('IBV_EVENT_SM_CHANGE', 13)
IBV_EVENT_SRQ_ERR = enum_ibv_event_type.define('IBV_EVENT_SRQ_ERR', 14)
IBV_EVENT_SRQ_LIMIT_REACHED = enum_ibv_event_type.define('IBV_EVENT_SRQ_LIMIT_REACHED', 15)
IBV_EVENT_QP_LAST_WQE_REACHED = enum_ibv_event_type.define('IBV_EVENT_QP_LAST_WQE_REACHED', 16)
IBV_EVENT_CLIENT_REREGISTER = enum_ibv_event_type.define('IBV_EVENT_CLIENT_REREGISTER', 17)
IBV_EVENT_GID_CHANGE = enum_ibv_event_type.define('IBV_EVENT_GID_CHANGE', 18)
IBV_EVENT_WQ_FATAL = enum_ibv_event_type.define('IBV_EVENT_WQ_FATAL', 19)

@c.record
class struct_ibv_async_event(c.Struct):
  SIZE = 16
  element: Annotated[struct_ibv_async_event_element, 0]
  event_type: Annotated[enum_ibv_event_type, 8]
@c.record
class struct_ibv_async_event_element(c.Struct):
  SIZE = 8
  cq: Annotated[c.POINTER[struct_ibv_cq], 0]
  qp: Annotated[c.POINTER[struct_ibv_qp], 0]
  srq: Annotated[c.POINTER[struct_ibv_srq], 0]
  wq: Annotated[c.POINTER[struct_ibv_wq], 0]
  port_num: Annotated[Annotated[int, ctypes.c_int32], 0]
@c.record
class struct_ibv_wq(c.Struct):
  SIZE = 152
  context: Annotated[c.POINTER[struct_ibv_context], 0]
  wq_context: Annotated[ctypes.c_void_p, 8]
  pd: Annotated[c.POINTER[struct_ibv_pd], 16]
  cq: Annotated[c.POINTER[struct_ibv_cq], 24]
  wq_num: Annotated[uint32_t, 32]
  handle: Annotated[uint32_t, 36]
  state: Annotated[enum_ibv_wq_state, 40]
  wq_type: Annotated[enum_ibv_wq_type, 44]
  post_recv: Annotated[c.CFUNCTYPE[Annotated[int, ctypes.c_int32], [c.POINTER[struct_ibv_wq], c.POINTER[struct_ibv_recv_wr], c.POINTER[c.POINTER[struct_ibv_recv_wr]]]], 48]
  mutex: Annotated[pthread_mutex_t, 56]
  cond: Annotated[pthread_cond_t, 96]
  events_completed: Annotated[uint32_t, 144]
  comp_mask: Annotated[uint32_t, 148]
class enum_ibv_wq_state(Annotated[int, ctypes.c_uint32], c.Enum): pass
IBV_WQS_RESET = enum_ibv_wq_state.define('IBV_WQS_RESET', 0)
IBV_WQS_RDY = enum_ibv_wq_state.define('IBV_WQS_RDY', 1)
IBV_WQS_ERR = enum_ibv_wq_state.define('IBV_WQS_ERR', 2)
IBV_WQS_UNKNOWN = enum_ibv_wq_state.define('IBV_WQS_UNKNOWN', 3)

class enum_ibv_wq_type(Annotated[int, ctypes.c_uint32], c.Enum): pass
IBV_WQT_RQ = enum_ibv_wq_type.define('IBV_WQT_RQ', 0)

@dll.bind
def ibv_wc_status_str(status:enum_ibv_wc_status) -> c.POINTER[Annotated[bytes, ctypes.c_char]]: ...
class _anonenum1(Annotated[int, ctypes.c_uint32], c.Enum): pass
IBV_WC_IP_CSUM_OK_SHIFT = _anonenum1.define('IBV_WC_IP_CSUM_OK_SHIFT', 2)

class enum_ibv_create_cq_wc_flags(Annotated[int, ctypes.c_uint32], c.Enum): pass
IBV_WC_EX_WITH_BYTE_LEN = enum_ibv_create_cq_wc_flags.define('IBV_WC_EX_WITH_BYTE_LEN', 1)
IBV_WC_EX_WITH_IMM = enum_ibv_create_cq_wc_flags.define('IBV_WC_EX_WITH_IMM', 2)
IBV_WC_EX_WITH_QP_NUM = enum_ibv_create_cq_wc_flags.define('IBV_WC_EX_WITH_QP_NUM', 4)
IBV_WC_EX_WITH_SRC_QP = enum_ibv_create_cq_wc_flags.define('IBV_WC_EX_WITH_SRC_QP', 8)
IBV_WC_EX_WITH_SLID = enum_ibv_create_cq_wc_flags.define('IBV_WC_EX_WITH_SLID', 16)
IBV_WC_EX_WITH_SL = enum_ibv_create_cq_wc_flags.define('IBV_WC_EX_WITH_SL', 32)
IBV_WC_EX_WITH_DLID_PATH_BITS = enum_ibv_create_cq_wc_flags.define('IBV_WC_EX_WITH_DLID_PATH_BITS', 64)
IBV_WC_EX_WITH_COMPLETION_TIMESTAMP = enum_ibv_create_cq_wc_flags.define('IBV_WC_EX_WITH_COMPLETION_TIMESTAMP', 128)
IBV_WC_EX_WITH_CVLAN = enum_ibv_create_cq_wc_flags.define('IBV_WC_EX_WITH_CVLAN', 256)
IBV_WC_EX_WITH_FLOW_TAG = enum_ibv_create_cq_wc_flags.define('IBV_WC_EX_WITH_FLOW_TAG', 512)
IBV_WC_EX_WITH_TM_INFO = enum_ibv_create_cq_wc_flags.define('IBV_WC_EX_WITH_TM_INFO', 1024)
IBV_WC_EX_WITH_COMPLETION_TIMESTAMP_WALLCLOCK = enum_ibv_create_cq_wc_flags.define('IBV_WC_EX_WITH_COMPLETION_TIMESTAMP_WALLCLOCK', 2048)

class _anonenum2(Annotated[int, ctypes.c_uint32], c.Enum): pass
IBV_WC_STANDARD_FLAGS = _anonenum2.define('IBV_WC_STANDARD_FLAGS', 127)

class _anonenum3(Annotated[int, ctypes.c_uint32], c.Enum): pass
IBV_CREATE_CQ_SUP_WC_FLAGS = _anonenum3.define('IBV_CREATE_CQ_SUP_WC_FLAGS', 4095)

class enum_ibv_wc_flags(Annotated[int, ctypes.c_uint32], c.Enum): pass
IBV_WC_GRH = enum_ibv_wc_flags.define('IBV_WC_GRH', 1)
IBV_WC_WITH_IMM = enum_ibv_wc_flags.define('IBV_WC_WITH_IMM', 2)
IBV_WC_IP_CSUM_OK = enum_ibv_wc_flags.define('IBV_WC_IP_CSUM_OK', 4)
IBV_WC_WITH_INV = enum_ibv_wc_flags.define('IBV_WC_WITH_INV', 8)
IBV_WC_TM_SYNC_REQ = enum_ibv_wc_flags.define('IBV_WC_TM_SYNC_REQ', 16)
IBV_WC_TM_MATCH = enum_ibv_wc_flags.define('IBV_WC_TM_MATCH', 32)
IBV_WC_TM_DATA_VALID = enum_ibv_wc_flags.define('IBV_WC_TM_DATA_VALID', 64)

class enum_ibv_access_flags(Annotated[int, ctypes.c_uint32], c.Enum): pass
IBV_ACCESS_LOCAL_WRITE = enum_ibv_access_flags.define('IBV_ACCESS_LOCAL_WRITE', 1)
IBV_ACCESS_REMOTE_WRITE = enum_ibv_access_flags.define('IBV_ACCESS_REMOTE_WRITE', 2)
IBV_ACCESS_REMOTE_READ = enum_ibv_access_flags.define('IBV_ACCESS_REMOTE_READ', 4)
IBV_ACCESS_REMOTE_ATOMIC = enum_ibv_access_flags.define('IBV_ACCESS_REMOTE_ATOMIC', 8)
IBV_ACCESS_MW_BIND = enum_ibv_access_flags.define('IBV_ACCESS_MW_BIND', 16)
IBV_ACCESS_ZERO_BASED = enum_ibv_access_flags.define('IBV_ACCESS_ZERO_BASED', 32)
IBV_ACCESS_ON_DEMAND = enum_ibv_access_flags.define('IBV_ACCESS_ON_DEMAND', 64)
IBV_ACCESS_HUGETLB = enum_ibv_access_flags.define('IBV_ACCESS_HUGETLB', 128)
IBV_ACCESS_FLUSH_GLOBAL = enum_ibv_access_flags.define('IBV_ACCESS_FLUSH_GLOBAL', 256)
IBV_ACCESS_FLUSH_PERSISTENT = enum_ibv_access_flags.define('IBV_ACCESS_FLUSH_PERSISTENT', 512)
IBV_ACCESS_RELAXED_ORDERING = enum_ibv_access_flags.define('IBV_ACCESS_RELAXED_ORDERING', 1048576)

@c.record
class struct_ibv_td_init_attr(c.Struct):
  SIZE = 4
  comp_mask: Annotated[uint32_t, 0]
@c.record
class struct_ibv_td(c.Struct):
  SIZE = 8
  context: Annotated[c.POINTER[struct_ibv_context], 0]
class enum_ibv_xrcd_init_attr_mask(Annotated[int, ctypes.c_uint32], c.Enum): pass
IBV_XRCD_INIT_ATTR_FD = enum_ibv_xrcd_init_attr_mask.define('IBV_XRCD_INIT_ATTR_FD', 1)
IBV_XRCD_INIT_ATTR_OFLAGS = enum_ibv_xrcd_init_attr_mask.define('IBV_XRCD_INIT_ATTR_OFLAGS', 2)
IBV_XRCD_INIT_ATTR_RESERVED = enum_ibv_xrcd_init_attr_mask.define('IBV_XRCD_INIT_ATTR_RESERVED', 4)

@c.record
class struct_ibv_xrcd_init_attr(c.Struct):
  SIZE = 12
  comp_mask: Annotated[uint32_t, 0]
  fd: Annotated[Annotated[int, ctypes.c_int32], 4]
  oflags: Annotated[Annotated[int, ctypes.c_int32], 8]
@c.record
class struct_ibv_xrcd(c.Struct):
  SIZE = 8
  context: Annotated[c.POINTER[struct_ibv_context], 0]
class enum_ibv_rereg_mr_flags(Annotated[int, ctypes.c_uint32], c.Enum): pass
IBV_REREG_MR_CHANGE_TRANSLATION = enum_ibv_rereg_mr_flags.define('IBV_REREG_MR_CHANGE_TRANSLATION', 1)
IBV_REREG_MR_CHANGE_PD = enum_ibv_rereg_mr_flags.define('IBV_REREG_MR_CHANGE_PD', 2)
IBV_REREG_MR_CHANGE_ACCESS = enum_ibv_rereg_mr_flags.define('IBV_REREG_MR_CHANGE_ACCESS', 4)
IBV_REREG_MR_FLAGS_SUPPORTED = enum_ibv_rereg_mr_flags.define('IBV_REREG_MR_FLAGS_SUPPORTED', 7)

@c.record
class struct_ibv_global_route(c.Struct):
  SIZE = 24
  dgid: Annotated[union_ibv_gid, 0]
  flow_label: Annotated[uint32_t, 16]
  sgid_index: Annotated[uint8_t, 20]
  hop_limit: Annotated[uint8_t, 21]
  traffic_class: Annotated[uint8_t, 22]
@c.record
class struct_ibv_grh(c.Struct):
  SIZE = 40
  version_tclass_flow: Annotated[Annotated[int, ctypes.c_uint32], 0]
  paylen: Annotated[Annotated[int, ctypes.c_uint16], 4]
  next_hdr: Annotated[uint8_t, 6]
  hop_limit: Annotated[uint8_t, 7]
  sgid: Annotated[union_ibv_gid, 8]
  dgid: Annotated[union_ibv_gid, 24]
__be16: TypeAlias = Annotated[int, ctypes.c_uint16]
class enum_ibv_rate(Annotated[int, ctypes.c_uint32], c.Enum): pass
IBV_RATE_MAX = enum_ibv_rate.define('IBV_RATE_MAX', 0)
IBV_RATE_2_5_GBPS = enum_ibv_rate.define('IBV_RATE_2_5_GBPS', 2)
IBV_RATE_5_GBPS = enum_ibv_rate.define('IBV_RATE_5_GBPS', 5)
IBV_RATE_10_GBPS = enum_ibv_rate.define('IBV_RATE_10_GBPS', 3)
IBV_RATE_20_GBPS = enum_ibv_rate.define('IBV_RATE_20_GBPS', 6)
IBV_RATE_30_GBPS = enum_ibv_rate.define('IBV_RATE_30_GBPS', 4)
IBV_RATE_40_GBPS = enum_ibv_rate.define('IBV_RATE_40_GBPS', 7)
IBV_RATE_60_GBPS = enum_ibv_rate.define('IBV_RATE_60_GBPS', 8)
IBV_RATE_80_GBPS = enum_ibv_rate.define('IBV_RATE_80_GBPS', 9)
IBV_RATE_120_GBPS = enum_ibv_rate.define('IBV_RATE_120_GBPS', 10)
IBV_RATE_14_GBPS = enum_ibv_rate.define('IBV_RATE_14_GBPS', 11)
IBV_RATE_56_GBPS = enum_ibv_rate.define('IBV_RATE_56_GBPS', 12)
IBV_RATE_112_GBPS = enum_ibv_rate.define('IBV_RATE_112_GBPS', 13)
IBV_RATE_168_GBPS = enum_ibv_rate.define('IBV_RATE_168_GBPS', 14)
IBV_RATE_25_GBPS = enum_ibv_rate.define('IBV_RATE_25_GBPS', 15)
IBV_RATE_100_GBPS = enum_ibv_rate.define('IBV_RATE_100_GBPS', 16)
IBV_RATE_200_GBPS = enum_ibv_rate.define('IBV_RATE_200_GBPS', 17)
IBV_RATE_300_GBPS = enum_ibv_rate.define('IBV_RATE_300_GBPS', 18)
IBV_RATE_28_GBPS = enum_ibv_rate.define('IBV_RATE_28_GBPS', 19)
IBV_RATE_50_GBPS = enum_ibv_rate.define('IBV_RATE_50_GBPS', 20)
IBV_RATE_400_GBPS = enum_ibv_rate.define('IBV_RATE_400_GBPS', 21)
IBV_RATE_600_GBPS = enum_ibv_rate.define('IBV_RATE_600_GBPS', 22)
IBV_RATE_800_GBPS = enum_ibv_rate.define('IBV_RATE_800_GBPS', 23)
IBV_RATE_1200_GBPS = enum_ibv_rate.define('IBV_RATE_1200_GBPS', 24)

@dll.bind
def ibv_rate_to_mult(rate:enum_ibv_rate) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def mult_to_ibv_rate(mult:Annotated[int, ctypes.c_int32]) -> enum_ibv_rate: ...
@dll.bind
def ibv_rate_to_mbps(rate:enum_ibv_rate) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def mbps_to_ibv_rate(mbps:Annotated[int, ctypes.c_int32]) -> enum_ibv_rate: ...
@c.record
class struct_ibv_ah_attr(c.Struct):
  SIZE = 32
  grh: Annotated[struct_ibv_global_route, 0]
  dlid: Annotated[uint16_t, 24]
  sl: Annotated[uint8_t, 26]
  src_path_bits: Annotated[uint8_t, 27]
  static_rate: Annotated[uint8_t, 28]
  is_global: Annotated[uint8_t, 29]
  port_num: Annotated[uint8_t, 30]
class enum_ibv_srq_attr_mask(Annotated[int, ctypes.c_uint32], c.Enum): pass
IBV_SRQ_MAX_WR = enum_ibv_srq_attr_mask.define('IBV_SRQ_MAX_WR', 1)
IBV_SRQ_LIMIT = enum_ibv_srq_attr_mask.define('IBV_SRQ_LIMIT', 2)

@c.record
class struct_ibv_srq_attr(c.Struct):
  SIZE = 12
  max_wr: Annotated[uint32_t, 0]
  max_sge: Annotated[uint32_t, 4]
  srq_limit: Annotated[uint32_t, 8]
@c.record
class struct_ibv_srq_init_attr(c.Struct):
  SIZE = 24
  srq_context: Annotated[ctypes.c_void_p, 0]
  attr: Annotated[struct_ibv_srq_attr, 8]
class enum_ibv_srq_type(Annotated[int, ctypes.c_uint32], c.Enum): pass
IBV_SRQT_BASIC = enum_ibv_srq_type.define('IBV_SRQT_BASIC', 0)
IBV_SRQT_XRC = enum_ibv_srq_type.define('IBV_SRQT_XRC', 1)
IBV_SRQT_TM = enum_ibv_srq_type.define('IBV_SRQT_TM', 2)

class enum_ibv_srq_init_attr_mask(Annotated[int, ctypes.c_uint32], c.Enum): pass
IBV_SRQ_INIT_ATTR_TYPE = enum_ibv_srq_init_attr_mask.define('IBV_SRQ_INIT_ATTR_TYPE', 1)
IBV_SRQ_INIT_ATTR_PD = enum_ibv_srq_init_attr_mask.define('IBV_SRQ_INIT_ATTR_PD', 2)
IBV_SRQ_INIT_ATTR_XRCD = enum_ibv_srq_init_attr_mask.define('IBV_SRQ_INIT_ATTR_XRCD', 4)
IBV_SRQ_INIT_ATTR_CQ = enum_ibv_srq_init_attr_mask.define('IBV_SRQ_INIT_ATTR_CQ', 8)
IBV_SRQ_INIT_ATTR_TM = enum_ibv_srq_init_attr_mask.define('IBV_SRQ_INIT_ATTR_TM', 16)
IBV_SRQ_INIT_ATTR_RESERVED = enum_ibv_srq_init_attr_mask.define('IBV_SRQ_INIT_ATTR_RESERVED', 32)

@c.record
class struct_ibv_tm_cap(c.Struct):
  SIZE = 8
  max_num_tags: Annotated[uint32_t, 0]
  max_ops: Annotated[uint32_t, 4]
@c.record
class struct_ibv_srq_init_attr_ex(c.Struct):
  SIZE = 64
  srq_context: Annotated[ctypes.c_void_p, 0]
  attr: Annotated[struct_ibv_srq_attr, 8]
  comp_mask: Annotated[uint32_t, 20]
  srq_type: Annotated[enum_ibv_srq_type, 24]
  pd: Annotated[c.POINTER[struct_ibv_pd], 32]
  xrcd: Annotated[c.POINTER[struct_ibv_xrcd], 40]
  cq: Annotated[c.POINTER[struct_ibv_cq], 48]
  tm_cap: Annotated[struct_ibv_tm_cap, 56]
class enum_ibv_wq_init_attr_mask(Annotated[int, ctypes.c_uint32], c.Enum): pass
IBV_WQ_INIT_ATTR_FLAGS = enum_ibv_wq_init_attr_mask.define('IBV_WQ_INIT_ATTR_FLAGS', 1)
IBV_WQ_INIT_ATTR_RESERVED = enum_ibv_wq_init_attr_mask.define('IBV_WQ_INIT_ATTR_RESERVED', 2)

class enum_ibv_wq_flags(Annotated[int, ctypes.c_uint32], c.Enum): pass
IBV_WQ_FLAGS_CVLAN_STRIPPING = enum_ibv_wq_flags.define('IBV_WQ_FLAGS_CVLAN_STRIPPING', 1)
IBV_WQ_FLAGS_SCATTER_FCS = enum_ibv_wq_flags.define('IBV_WQ_FLAGS_SCATTER_FCS', 2)
IBV_WQ_FLAGS_DELAY_DROP = enum_ibv_wq_flags.define('IBV_WQ_FLAGS_DELAY_DROP', 4)
IBV_WQ_FLAGS_PCI_WRITE_END_PADDING = enum_ibv_wq_flags.define('IBV_WQ_FLAGS_PCI_WRITE_END_PADDING', 8)
IBV_WQ_FLAGS_RESERVED = enum_ibv_wq_flags.define('IBV_WQ_FLAGS_RESERVED', 16)

@c.record
class struct_ibv_wq_init_attr(c.Struct):
  SIZE = 48
  wq_context: Annotated[ctypes.c_void_p, 0]
  wq_type: Annotated[enum_ibv_wq_type, 8]
  max_wr: Annotated[uint32_t, 12]
  max_sge: Annotated[uint32_t, 16]
  pd: Annotated[c.POINTER[struct_ibv_pd], 24]
  cq: Annotated[c.POINTER[struct_ibv_cq], 32]
  comp_mask: Annotated[uint32_t, 40]
  create_flags: Annotated[uint32_t, 44]
class enum_ibv_wq_attr_mask(Annotated[int, ctypes.c_uint32], c.Enum): pass
IBV_WQ_ATTR_STATE = enum_ibv_wq_attr_mask.define('IBV_WQ_ATTR_STATE', 1)
IBV_WQ_ATTR_CURR_STATE = enum_ibv_wq_attr_mask.define('IBV_WQ_ATTR_CURR_STATE', 2)
IBV_WQ_ATTR_FLAGS = enum_ibv_wq_attr_mask.define('IBV_WQ_ATTR_FLAGS', 4)
IBV_WQ_ATTR_RESERVED = enum_ibv_wq_attr_mask.define('IBV_WQ_ATTR_RESERVED', 8)

@c.record
class struct_ibv_wq_attr(c.Struct):
  SIZE = 20
  attr_mask: Annotated[uint32_t, 0]
  wq_state: Annotated[enum_ibv_wq_state, 4]
  curr_wq_state: Annotated[enum_ibv_wq_state, 8]
  flags: Annotated[uint32_t, 12]
  flags_mask: Annotated[uint32_t, 16]
@c.record
class struct_ibv_rwq_ind_table(c.Struct):
  SIZE = 24
  context: Annotated[c.POINTER[struct_ibv_context], 0]
  ind_tbl_handle: Annotated[Annotated[int, ctypes.c_int32], 8]
  ind_tbl_num: Annotated[Annotated[int, ctypes.c_int32], 12]
  comp_mask: Annotated[uint32_t, 16]
class enum_ibv_ind_table_init_attr_mask(Annotated[int, ctypes.c_uint32], c.Enum): pass
IBV_CREATE_IND_TABLE_RESERVED = enum_ibv_ind_table_init_attr_mask.define('IBV_CREATE_IND_TABLE_RESERVED', 1)

@c.record
class struct_ibv_rwq_ind_table_init_attr(c.Struct):
  SIZE = 24
  log_ind_tbl_size: Annotated[uint32_t, 0]
  ind_tbl: Annotated[c.POINTER[c.POINTER[struct_ibv_wq]], 8]
  comp_mask: Annotated[uint32_t, 16]
@c.record
class struct_ibv_qp_cap(c.Struct):
  SIZE = 20
  max_send_wr: Annotated[uint32_t, 0]
  max_recv_wr: Annotated[uint32_t, 4]
  max_send_sge: Annotated[uint32_t, 8]
  max_recv_sge: Annotated[uint32_t, 12]
  max_inline_data: Annotated[uint32_t, 16]
@c.record
class struct_ibv_qp_init_attr(c.Struct):
  SIZE = 64
  qp_context: Annotated[ctypes.c_void_p, 0]
  send_cq: Annotated[c.POINTER[struct_ibv_cq], 8]
  recv_cq: Annotated[c.POINTER[struct_ibv_cq], 16]
  srq: Annotated[c.POINTER[struct_ibv_srq], 24]
  cap: Annotated[struct_ibv_qp_cap, 32]
  qp_type: Annotated[enum_ibv_qp_type, 52]
  sq_sig_all: Annotated[Annotated[int, ctypes.c_int32], 56]
class enum_ibv_qp_init_attr_mask(Annotated[int, ctypes.c_uint32], c.Enum): pass
IBV_QP_INIT_ATTR_PD = enum_ibv_qp_init_attr_mask.define('IBV_QP_INIT_ATTR_PD', 1)
IBV_QP_INIT_ATTR_XRCD = enum_ibv_qp_init_attr_mask.define('IBV_QP_INIT_ATTR_XRCD', 2)
IBV_QP_INIT_ATTR_CREATE_FLAGS = enum_ibv_qp_init_attr_mask.define('IBV_QP_INIT_ATTR_CREATE_FLAGS', 4)
IBV_QP_INIT_ATTR_MAX_TSO_HEADER = enum_ibv_qp_init_attr_mask.define('IBV_QP_INIT_ATTR_MAX_TSO_HEADER', 8)
IBV_QP_INIT_ATTR_IND_TABLE = enum_ibv_qp_init_attr_mask.define('IBV_QP_INIT_ATTR_IND_TABLE', 16)
IBV_QP_INIT_ATTR_RX_HASH = enum_ibv_qp_init_attr_mask.define('IBV_QP_INIT_ATTR_RX_HASH', 32)
IBV_QP_INIT_ATTR_SEND_OPS_FLAGS = enum_ibv_qp_init_attr_mask.define('IBV_QP_INIT_ATTR_SEND_OPS_FLAGS', 64)

class enum_ibv_qp_create_flags(Annotated[int, ctypes.c_uint32], c.Enum): pass
IBV_QP_CREATE_BLOCK_SELF_MCAST_LB = enum_ibv_qp_create_flags.define('IBV_QP_CREATE_BLOCK_SELF_MCAST_LB', 2)
IBV_QP_CREATE_SCATTER_FCS = enum_ibv_qp_create_flags.define('IBV_QP_CREATE_SCATTER_FCS', 256)
IBV_QP_CREATE_CVLAN_STRIPPING = enum_ibv_qp_create_flags.define('IBV_QP_CREATE_CVLAN_STRIPPING', 512)
IBV_QP_CREATE_SOURCE_QPN = enum_ibv_qp_create_flags.define('IBV_QP_CREATE_SOURCE_QPN', 1024)
IBV_QP_CREATE_PCI_WRITE_END_PADDING = enum_ibv_qp_create_flags.define('IBV_QP_CREATE_PCI_WRITE_END_PADDING', 2048)

class enum_ibv_qp_create_send_ops_flags(Annotated[int, ctypes.c_uint32], c.Enum): pass
IBV_QP_EX_WITH_RDMA_WRITE = enum_ibv_qp_create_send_ops_flags.define('IBV_QP_EX_WITH_RDMA_WRITE', 1)
IBV_QP_EX_WITH_RDMA_WRITE_WITH_IMM = enum_ibv_qp_create_send_ops_flags.define('IBV_QP_EX_WITH_RDMA_WRITE_WITH_IMM', 2)
IBV_QP_EX_WITH_SEND = enum_ibv_qp_create_send_ops_flags.define('IBV_QP_EX_WITH_SEND', 4)
IBV_QP_EX_WITH_SEND_WITH_IMM = enum_ibv_qp_create_send_ops_flags.define('IBV_QP_EX_WITH_SEND_WITH_IMM', 8)
IBV_QP_EX_WITH_RDMA_READ = enum_ibv_qp_create_send_ops_flags.define('IBV_QP_EX_WITH_RDMA_READ', 16)
IBV_QP_EX_WITH_ATOMIC_CMP_AND_SWP = enum_ibv_qp_create_send_ops_flags.define('IBV_QP_EX_WITH_ATOMIC_CMP_AND_SWP', 32)
IBV_QP_EX_WITH_ATOMIC_FETCH_AND_ADD = enum_ibv_qp_create_send_ops_flags.define('IBV_QP_EX_WITH_ATOMIC_FETCH_AND_ADD', 64)
IBV_QP_EX_WITH_LOCAL_INV = enum_ibv_qp_create_send_ops_flags.define('IBV_QP_EX_WITH_LOCAL_INV', 128)
IBV_QP_EX_WITH_BIND_MW = enum_ibv_qp_create_send_ops_flags.define('IBV_QP_EX_WITH_BIND_MW', 256)
IBV_QP_EX_WITH_SEND_WITH_INV = enum_ibv_qp_create_send_ops_flags.define('IBV_QP_EX_WITH_SEND_WITH_INV', 512)
IBV_QP_EX_WITH_TSO = enum_ibv_qp_create_send_ops_flags.define('IBV_QP_EX_WITH_TSO', 1024)
IBV_QP_EX_WITH_FLUSH = enum_ibv_qp_create_send_ops_flags.define('IBV_QP_EX_WITH_FLUSH', 2048)
IBV_QP_EX_WITH_ATOMIC_WRITE = enum_ibv_qp_create_send_ops_flags.define('IBV_QP_EX_WITH_ATOMIC_WRITE', 4096)

@c.record
class struct_ibv_rx_hash_conf(c.Struct):
  SIZE = 24
  rx_hash_function: Annotated[uint8_t, 0]
  rx_hash_key_len: Annotated[uint8_t, 1]
  rx_hash_key: Annotated[c.POINTER[uint8_t], 8]
  rx_hash_fields_mask: Annotated[uint64_t, 16]
@c.record
class struct_ibv_qp_init_attr_ex(c.Struct):
  SIZE = 136
  qp_context: Annotated[ctypes.c_void_p, 0]
  send_cq: Annotated[c.POINTER[struct_ibv_cq], 8]
  recv_cq: Annotated[c.POINTER[struct_ibv_cq], 16]
  srq: Annotated[c.POINTER[struct_ibv_srq], 24]
  cap: Annotated[struct_ibv_qp_cap, 32]
  qp_type: Annotated[enum_ibv_qp_type, 52]
  sq_sig_all: Annotated[Annotated[int, ctypes.c_int32], 56]
  comp_mask: Annotated[uint32_t, 60]
  pd: Annotated[c.POINTER[struct_ibv_pd], 64]
  xrcd: Annotated[c.POINTER[struct_ibv_xrcd], 72]
  create_flags: Annotated[uint32_t, 80]
  max_tso_header: Annotated[uint16_t, 84]
  rwq_ind_tbl: Annotated[c.POINTER[struct_ibv_rwq_ind_table], 88]
  rx_hash_conf: Annotated[struct_ibv_rx_hash_conf, 96]
  source_qpn: Annotated[uint32_t, 120]
  send_ops_flags: Annotated[uint64_t, 128]
class enum_ibv_qp_open_attr_mask(Annotated[int, ctypes.c_uint32], c.Enum): pass
IBV_QP_OPEN_ATTR_NUM = enum_ibv_qp_open_attr_mask.define('IBV_QP_OPEN_ATTR_NUM', 1)
IBV_QP_OPEN_ATTR_XRCD = enum_ibv_qp_open_attr_mask.define('IBV_QP_OPEN_ATTR_XRCD', 2)
IBV_QP_OPEN_ATTR_CONTEXT = enum_ibv_qp_open_attr_mask.define('IBV_QP_OPEN_ATTR_CONTEXT', 4)
IBV_QP_OPEN_ATTR_TYPE = enum_ibv_qp_open_attr_mask.define('IBV_QP_OPEN_ATTR_TYPE', 8)
IBV_QP_OPEN_ATTR_RESERVED = enum_ibv_qp_open_attr_mask.define('IBV_QP_OPEN_ATTR_RESERVED', 16)

@c.record
class struct_ibv_qp_open_attr(c.Struct):
  SIZE = 32
  comp_mask: Annotated[uint32_t, 0]
  qp_num: Annotated[uint32_t, 4]
  xrcd: Annotated[c.POINTER[struct_ibv_xrcd], 8]
  qp_context: Annotated[ctypes.c_void_p, 16]
  qp_type: Annotated[enum_ibv_qp_type, 24]
class enum_ibv_qp_attr_mask(Annotated[int, ctypes.c_uint32], c.Enum): pass
IBV_QP_STATE = enum_ibv_qp_attr_mask.define('IBV_QP_STATE', 1)
IBV_QP_CUR_STATE = enum_ibv_qp_attr_mask.define('IBV_QP_CUR_STATE', 2)
IBV_QP_EN_SQD_ASYNC_NOTIFY = enum_ibv_qp_attr_mask.define('IBV_QP_EN_SQD_ASYNC_NOTIFY', 4)
IBV_QP_ACCESS_FLAGS = enum_ibv_qp_attr_mask.define('IBV_QP_ACCESS_FLAGS', 8)
IBV_QP_PKEY_INDEX = enum_ibv_qp_attr_mask.define('IBV_QP_PKEY_INDEX', 16)
IBV_QP_PORT = enum_ibv_qp_attr_mask.define('IBV_QP_PORT', 32)
IBV_QP_QKEY = enum_ibv_qp_attr_mask.define('IBV_QP_QKEY', 64)
IBV_QP_AV = enum_ibv_qp_attr_mask.define('IBV_QP_AV', 128)
IBV_QP_PATH_MTU = enum_ibv_qp_attr_mask.define('IBV_QP_PATH_MTU', 256)
IBV_QP_TIMEOUT = enum_ibv_qp_attr_mask.define('IBV_QP_TIMEOUT', 512)
IBV_QP_RETRY_CNT = enum_ibv_qp_attr_mask.define('IBV_QP_RETRY_CNT', 1024)
IBV_QP_RNR_RETRY = enum_ibv_qp_attr_mask.define('IBV_QP_RNR_RETRY', 2048)
IBV_QP_RQ_PSN = enum_ibv_qp_attr_mask.define('IBV_QP_RQ_PSN', 4096)
IBV_QP_MAX_QP_RD_ATOMIC = enum_ibv_qp_attr_mask.define('IBV_QP_MAX_QP_RD_ATOMIC', 8192)
IBV_QP_ALT_PATH = enum_ibv_qp_attr_mask.define('IBV_QP_ALT_PATH', 16384)
IBV_QP_MIN_RNR_TIMER = enum_ibv_qp_attr_mask.define('IBV_QP_MIN_RNR_TIMER', 32768)
IBV_QP_SQ_PSN = enum_ibv_qp_attr_mask.define('IBV_QP_SQ_PSN', 65536)
IBV_QP_MAX_DEST_RD_ATOMIC = enum_ibv_qp_attr_mask.define('IBV_QP_MAX_DEST_RD_ATOMIC', 131072)
IBV_QP_PATH_MIG_STATE = enum_ibv_qp_attr_mask.define('IBV_QP_PATH_MIG_STATE', 262144)
IBV_QP_CAP = enum_ibv_qp_attr_mask.define('IBV_QP_CAP', 524288)
IBV_QP_DEST_QPN = enum_ibv_qp_attr_mask.define('IBV_QP_DEST_QPN', 1048576)
IBV_QP_RATE_LIMIT = enum_ibv_qp_attr_mask.define('IBV_QP_RATE_LIMIT', 33554432)

class enum_ibv_query_qp_data_in_order_flags(Annotated[int, ctypes.c_uint32], c.Enum): pass
IBV_QUERY_QP_DATA_IN_ORDER_RETURN_CAPS = enum_ibv_query_qp_data_in_order_flags.define('IBV_QUERY_QP_DATA_IN_ORDER_RETURN_CAPS', 1)

class enum_ibv_query_qp_data_in_order_caps(Annotated[int, ctypes.c_uint32], c.Enum): pass
IBV_QUERY_QP_DATA_IN_ORDER_WHOLE_MSG = enum_ibv_query_qp_data_in_order_caps.define('IBV_QUERY_QP_DATA_IN_ORDER_WHOLE_MSG', 1)
IBV_QUERY_QP_DATA_IN_ORDER_ALIGNED_128_BYTES = enum_ibv_query_qp_data_in_order_caps.define('IBV_QUERY_QP_DATA_IN_ORDER_ALIGNED_128_BYTES', 2)

class enum_ibv_mig_state(Annotated[int, ctypes.c_uint32], c.Enum): pass
IBV_MIG_MIGRATED = enum_ibv_mig_state.define('IBV_MIG_MIGRATED', 0)
IBV_MIG_REARM = enum_ibv_mig_state.define('IBV_MIG_REARM', 1)
IBV_MIG_ARMED = enum_ibv_mig_state.define('IBV_MIG_ARMED', 2)

@c.record
class struct_ibv_qp_attr(c.Struct):
  SIZE = 144
  qp_state: Annotated[enum_ibv_qp_state, 0]
  cur_qp_state: Annotated[enum_ibv_qp_state, 4]
  path_mtu: Annotated[enum_ibv_mtu, 8]
  path_mig_state: Annotated[enum_ibv_mig_state, 12]
  qkey: Annotated[uint32_t, 16]
  rq_psn: Annotated[uint32_t, 20]
  sq_psn: Annotated[uint32_t, 24]
  dest_qp_num: Annotated[uint32_t, 28]
  qp_access_flags: Annotated[Annotated[int, ctypes.c_uint32], 32]
  cap: Annotated[struct_ibv_qp_cap, 36]
  ah_attr: Annotated[struct_ibv_ah_attr, 56]
  alt_ah_attr: Annotated[struct_ibv_ah_attr, 88]
  pkey_index: Annotated[uint16_t, 120]
  alt_pkey_index: Annotated[uint16_t, 122]
  en_sqd_async_notify: Annotated[uint8_t, 124]
  sq_draining: Annotated[uint8_t, 125]
  max_rd_atomic: Annotated[uint8_t, 126]
  max_dest_rd_atomic: Annotated[uint8_t, 127]
  min_rnr_timer: Annotated[uint8_t, 128]
  port_num: Annotated[uint8_t, 129]
  timeout: Annotated[uint8_t, 130]
  retry_cnt: Annotated[uint8_t, 131]
  rnr_retry: Annotated[uint8_t, 132]
  alt_port_num: Annotated[uint8_t, 133]
  alt_timeout: Annotated[uint8_t, 134]
  rate_limit: Annotated[uint32_t, 136]
@c.record
class struct_ibv_qp_rate_limit_attr(c.Struct):
  SIZE = 16
  rate_limit: Annotated[uint32_t, 0]
  max_burst_sz: Annotated[uint32_t, 4]
  typical_pkt_sz: Annotated[uint16_t, 8]
  comp_mask: Annotated[uint32_t, 12]
@dll.bind
def ibv_wr_opcode_str(opcode:enum_ibv_wr_opcode) -> c.POINTER[Annotated[bytes, ctypes.c_char]]: ...
class enum_ibv_send_flags(Annotated[int, ctypes.c_uint32], c.Enum): pass
IBV_SEND_FENCE = enum_ibv_send_flags.define('IBV_SEND_FENCE', 1)
IBV_SEND_SIGNALED = enum_ibv_send_flags.define('IBV_SEND_SIGNALED', 2)
IBV_SEND_SOLICITED = enum_ibv_send_flags.define('IBV_SEND_SOLICITED', 4)
IBV_SEND_INLINE = enum_ibv_send_flags.define('IBV_SEND_INLINE', 8)
IBV_SEND_IP_CSUM = enum_ibv_send_flags.define('IBV_SEND_IP_CSUM', 16)

class enum_ibv_placement_type(Annotated[int, ctypes.c_uint32], c.Enum): pass
IBV_FLUSH_GLOBAL = enum_ibv_placement_type.define('IBV_FLUSH_GLOBAL', 1)
IBV_FLUSH_PERSISTENT = enum_ibv_placement_type.define('IBV_FLUSH_PERSISTENT', 2)

class enum_ibv_selectivity_level(Annotated[int, ctypes.c_uint32], c.Enum): pass
IBV_FLUSH_RANGE = enum_ibv_selectivity_level.define('IBV_FLUSH_RANGE', 0)
IBV_FLUSH_MR = enum_ibv_selectivity_level.define('IBV_FLUSH_MR', 1)

@c.record
class struct_ibv_data_buf(c.Struct):
  SIZE = 16
  addr: Annotated[ctypes.c_void_p, 0]
  length: Annotated[size_t, 8]
class enum_ibv_ops_wr_opcode(Annotated[int, ctypes.c_uint32], c.Enum): pass
IBV_WR_TAG_ADD = enum_ibv_ops_wr_opcode.define('IBV_WR_TAG_ADD', 0)
IBV_WR_TAG_DEL = enum_ibv_ops_wr_opcode.define('IBV_WR_TAG_DEL', 1)
IBV_WR_TAG_SYNC = enum_ibv_ops_wr_opcode.define('IBV_WR_TAG_SYNC', 2)

class enum_ibv_ops_flags(Annotated[int, ctypes.c_uint32], c.Enum): pass
IBV_OPS_SIGNALED = enum_ibv_ops_flags.define('IBV_OPS_SIGNALED', 1)
IBV_OPS_TM_SYNC = enum_ibv_ops_flags.define('IBV_OPS_TM_SYNC', 2)

@c.record
class struct_ibv_ops_wr(c.Struct):
  SIZE = 72
  wr_id: Annotated[uint64_t, 0]
  next: Annotated[c.POINTER[struct_ibv_ops_wr], 8]
  opcode: Annotated[enum_ibv_ops_wr_opcode, 16]
  flags: Annotated[Annotated[int, ctypes.c_int32], 20]
  tm: Annotated[struct_ibv_ops_wr_tm, 24]
@c.record
class struct_ibv_ops_wr_tm(c.Struct):
  SIZE = 48
  unexpected_cnt: Annotated[uint32_t, 0]
  handle: Annotated[uint32_t, 4]
  add: Annotated[struct_ibv_ops_wr_tm_add, 8]
@c.record
class struct_ibv_ops_wr_tm_add(c.Struct):
  SIZE = 40
  recv_wr_id: Annotated[uint64_t, 0]
  sg_list: Annotated[c.POINTER[struct_ibv_sge], 8]
  num_sge: Annotated[Annotated[int, ctypes.c_int32], 16]
  tag: Annotated[uint64_t, 24]
  mask: Annotated[uint64_t, 32]
@c.record
class struct_ibv_qp_ex(c.Struct):
  SIZE = 360
  qp_base: Annotated[struct_ibv_qp, 0]
  comp_mask: Annotated[uint64_t, 160]
  wr_id: Annotated[uint64_t, 168]
  wr_flags: Annotated[Annotated[int, ctypes.c_uint32], 176]
  wr_atomic_cmp_swp: Annotated[c.CFUNCTYPE[None, [c.POINTER[struct_ibv_qp_ex], uint32_t, uint64_t, uint64_t, uint64_t]], 184]
  wr_atomic_fetch_add: Annotated[c.CFUNCTYPE[None, [c.POINTER[struct_ibv_qp_ex], uint32_t, uint64_t, uint64_t]], 192]
  wr_bind_mw: Annotated[c.CFUNCTYPE[None, [c.POINTER[struct_ibv_qp_ex], c.POINTER[struct_ibv_mw], uint32_t, c.POINTER[struct_ibv_mw_bind_info]]], 200]
  wr_local_inv: Annotated[c.CFUNCTYPE[None, [c.POINTER[struct_ibv_qp_ex], uint32_t]], 208]
  wr_rdma_read: Annotated[c.CFUNCTYPE[None, [c.POINTER[struct_ibv_qp_ex], uint32_t, uint64_t]], 216]
  wr_rdma_write: Annotated[c.CFUNCTYPE[None, [c.POINTER[struct_ibv_qp_ex], uint32_t, uint64_t]], 224]
  wr_rdma_write_imm: Annotated[c.CFUNCTYPE[None, [c.POINTER[struct_ibv_qp_ex], uint32_t, uint64_t, Annotated[int, ctypes.c_uint32]]], 232]
  wr_send: Annotated[c.CFUNCTYPE[None, [c.POINTER[struct_ibv_qp_ex]]], 240]
  wr_send_imm: Annotated[c.CFUNCTYPE[None, [c.POINTER[struct_ibv_qp_ex], Annotated[int, ctypes.c_uint32]]], 248]
  wr_send_inv: Annotated[c.CFUNCTYPE[None, [c.POINTER[struct_ibv_qp_ex], uint32_t]], 256]
  wr_send_tso: Annotated[c.CFUNCTYPE[None, [c.POINTER[struct_ibv_qp_ex], ctypes.c_void_p, uint16_t, uint16_t]], 264]
  wr_set_ud_addr: Annotated[c.CFUNCTYPE[None, [c.POINTER[struct_ibv_qp_ex], c.POINTER[struct_ibv_ah], uint32_t, uint32_t]], 272]
  wr_set_xrc_srqn: Annotated[c.CFUNCTYPE[None, [c.POINTER[struct_ibv_qp_ex], uint32_t]], 280]
  wr_set_inline_data: Annotated[c.CFUNCTYPE[None, [c.POINTER[struct_ibv_qp_ex], ctypes.c_void_p, size_t]], 288]
  wr_set_inline_data_list: Annotated[c.CFUNCTYPE[None, [c.POINTER[struct_ibv_qp_ex], size_t, c.POINTER[struct_ibv_data_buf]]], 296]
  wr_set_sge: Annotated[c.CFUNCTYPE[None, [c.POINTER[struct_ibv_qp_ex], uint32_t, uint64_t, uint32_t]], 304]
  wr_set_sge_list: Annotated[c.CFUNCTYPE[None, [c.POINTER[struct_ibv_qp_ex], size_t, c.POINTER[struct_ibv_sge]]], 312]
  wr_start: Annotated[c.CFUNCTYPE[None, [c.POINTER[struct_ibv_qp_ex]]], 320]
  wr_complete: Annotated[c.CFUNCTYPE[Annotated[int, ctypes.c_int32], [c.POINTER[struct_ibv_qp_ex]]], 328]
  wr_abort: Annotated[c.CFUNCTYPE[None, [c.POINTER[struct_ibv_qp_ex]]], 336]
  wr_atomic_write: Annotated[c.CFUNCTYPE[None, [c.POINTER[struct_ibv_qp_ex], uint32_t, uint64_t, ctypes.c_void_p]], 344]
  wr_flush: Annotated[c.CFUNCTYPE[None, [c.POINTER[struct_ibv_qp_ex], uint32_t, uint64_t, size_t, uint8_t, uint8_t]], 352]
@dll.bind
def ibv_qp_to_qp_ex(qp:c.POINTER[struct_ibv_qp]) -> c.POINTER[struct_ibv_qp_ex]: ...
@c.record
class struct_ibv_ece(c.Struct):
  SIZE = 12
  vendor_id: Annotated[uint32_t, 0]
  options: Annotated[uint32_t, 4]
  comp_mask: Annotated[uint32_t, 8]
@c.record
class struct_ibv_poll_cq_attr(c.Struct):
  SIZE = 4
  comp_mask: Annotated[uint32_t, 0]
@c.record
class struct_ibv_wc_tm_info(c.Struct):
  SIZE = 16
  tag: Annotated[uint64_t, 0]
  priv: Annotated[uint32_t, 8]
@c.record
class struct_ibv_cq_ex(c.Struct):
  SIZE = 288
  context: Annotated[c.POINTER[struct_ibv_context], 0]
  channel: Annotated[c.POINTER[struct_ibv_comp_channel], 8]
  cq_context: Annotated[ctypes.c_void_p, 16]
  handle: Annotated[uint32_t, 24]
  cqe: Annotated[Annotated[int, ctypes.c_int32], 28]
  mutex: Annotated[pthread_mutex_t, 32]
  cond: Annotated[pthread_cond_t, 72]
  comp_events_completed: Annotated[uint32_t, 120]
  async_events_completed: Annotated[uint32_t, 124]
  comp_mask: Annotated[uint32_t, 128]
  status: Annotated[enum_ibv_wc_status, 132]
  wr_id: Annotated[uint64_t, 136]
  start_poll: Annotated[c.CFUNCTYPE[Annotated[int, ctypes.c_int32], [c.POINTER[struct_ibv_cq_ex], c.POINTER[struct_ibv_poll_cq_attr]]], 144]
  next_poll: Annotated[c.CFUNCTYPE[Annotated[int, ctypes.c_int32], [c.POINTER[struct_ibv_cq_ex]]], 152]
  end_poll: Annotated[c.CFUNCTYPE[None, [c.POINTER[struct_ibv_cq_ex]]], 160]
  read_opcode: Annotated[c.CFUNCTYPE[enum_ibv_wc_opcode, [c.POINTER[struct_ibv_cq_ex]]], 168]
  read_vendor_err: Annotated[c.CFUNCTYPE[uint32_t, [c.POINTER[struct_ibv_cq_ex]]], 176]
  read_byte_len: Annotated[c.CFUNCTYPE[uint32_t, [c.POINTER[struct_ibv_cq_ex]]], 184]
  read_imm_data: Annotated[c.CFUNCTYPE[Annotated[int, ctypes.c_uint32], [c.POINTER[struct_ibv_cq_ex]]], 192]
  read_qp_num: Annotated[c.CFUNCTYPE[uint32_t, [c.POINTER[struct_ibv_cq_ex]]], 200]
  read_src_qp: Annotated[c.CFUNCTYPE[uint32_t, [c.POINTER[struct_ibv_cq_ex]]], 208]
  read_wc_flags: Annotated[c.CFUNCTYPE[Annotated[int, ctypes.c_uint32], [c.POINTER[struct_ibv_cq_ex]]], 216]
  read_slid: Annotated[c.CFUNCTYPE[uint32_t, [c.POINTER[struct_ibv_cq_ex]]], 224]
  read_sl: Annotated[c.CFUNCTYPE[uint8_t, [c.POINTER[struct_ibv_cq_ex]]], 232]
  read_dlid_path_bits: Annotated[c.CFUNCTYPE[uint8_t, [c.POINTER[struct_ibv_cq_ex]]], 240]
  read_completion_ts: Annotated[c.CFUNCTYPE[uint64_t, [c.POINTER[struct_ibv_cq_ex]]], 248]
  read_cvlan: Annotated[c.CFUNCTYPE[uint16_t, [c.POINTER[struct_ibv_cq_ex]]], 256]
  read_flow_tag: Annotated[c.CFUNCTYPE[uint32_t, [c.POINTER[struct_ibv_cq_ex]]], 264]
  read_tm_info: Annotated[c.CFUNCTYPE[None, [c.POINTER[struct_ibv_cq_ex], c.POINTER[struct_ibv_wc_tm_info]]], 272]
  read_completion_wallclock_ns: Annotated[c.CFUNCTYPE[uint64_t, [c.POINTER[struct_ibv_cq_ex]]], 280]
class enum_ibv_cq_attr_mask(Annotated[int, ctypes.c_uint32], c.Enum): pass
IBV_CQ_ATTR_MODERATE = enum_ibv_cq_attr_mask.define('IBV_CQ_ATTR_MODERATE', 1)
IBV_CQ_ATTR_RESERVED = enum_ibv_cq_attr_mask.define('IBV_CQ_ATTR_RESERVED', 2)

@c.record
class struct_ibv_moderate_cq(c.Struct):
  SIZE = 4
  cq_count: Annotated[uint16_t, 0]
  cq_period: Annotated[uint16_t, 2]
@c.record
class struct_ibv_modify_cq_attr(c.Struct):
  SIZE = 8
  attr_mask: Annotated[uint32_t, 0]
  moderate: Annotated[struct_ibv_moderate_cq, 4]
class enum_ibv_flow_flags(Annotated[int, ctypes.c_uint32], c.Enum): pass
IBV_FLOW_ATTR_FLAGS_DONT_TRAP = enum_ibv_flow_flags.define('IBV_FLOW_ATTR_FLAGS_DONT_TRAP', 2)
IBV_FLOW_ATTR_FLAGS_EGRESS = enum_ibv_flow_flags.define('IBV_FLOW_ATTR_FLAGS_EGRESS', 4)

class enum_ibv_flow_attr_type(Annotated[int, ctypes.c_uint32], c.Enum): pass
IBV_FLOW_ATTR_NORMAL = enum_ibv_flow_attr_type.define('IBV_FLOW_ATTR_NORMAL', 0)
IBV_FLOW_ATTR_ALL_DEFAULT = enum_ibv_flow_attr_type.define('IBV_FLOW_ATTR_ALL_DEFAULT', 1)
IBV_FLOW_ATTR_MC_DEFAULT = enum_ibv_flow_attr_type.define('IBV_FLOW_ATTR_MC_DEFAULT', 2)
IBV_FLOW_ATTR_SNIFFER = enum_ibv_flow_attr_type.define('IBV_FLOW_ATTR_SNIFFER', 3)

class enum_ibv_flow_spec_type(Annotated[int, ctypes.c_uint32], c.Enum): pass
IBV_FLOW_SPEC_ETH = enum_ibv_flow_spec_type.define('IBV_FLOW_SPEC_ETH', 32)
IBV_FLOW_SPEC_IPV4 = enum_ibv_flow_spec_type.define('IBV_FLOW_SPEC_IPV4', 48)
IBV_FLOW_SPEC_IPV6 = enum_ibv_flow_spec_type.define('IBV_FLOW_SPEC_IPV6', 49)
IBV_FLOW_SPEC_IPV4_EXT = enum_ibv_flow_spec_type.define('IBV_FLOW_SPEC_IPV4_EXT', 50)
IBV_FLOW_SPEC_ESP = enum_ibv_flow_spec_type.define('IBV_FLOW_SPEC_ESP', 52)
IBV_FLOW_SPEC_TCP = enum_ibv_flow_spec_type.define('IBV_FLOW_SPEC_TCP', 64)
IBV_FLOW_SPEC_UDP = enum_ibv_flow_spec_type.define('IBV_FLOW_SPEC_UDP', 65)
IBV_FLOW_SPEC_VXLAN_TUNNEL = enum_ibv_flow_spec_type.define('IBV_FLOW_SPEC_VXLAN_TUNNEL', 80)
IBV_FLOW_SPEC_GRE = enum_ibv_flow_spec_type.define('IBV_FLOW_SPEC_GRE', 81)
IBV_FLOW_SPEC_MPLS = enum_ibv_flow_spec_type.define('IBV_FLOW_SPEC_MPLS', 96)
IBV_FLOW_SPEC_INNER = enum_ibv_flow_spec_type.define('IBV_FLOW_SPEC_INNER', 256)
IBV_FLOW_SPEC_ACTION_TAG = enum_ibv_flow_spec_type.define('IBV_FLOW_SPEC_ACTION_TAG', 4096)
IBV_FLOW_SPEC_ACTION_DROP = enum_ibv_flow_spec_type.define('IBV_FLOW_SPEC_ACTION_DROP', 4097)
IBV_FLOW_SPEC_ACTION_HANDLE = enum_ibv_flow_spec_type.define('IBV_FLOW_SPEC_ACTION_HANDLE', 4098)
IBV_FLOW_SPEC_ACTION_COUNT = enum_ibv_flow_spec_type.define('IBV_FLOW_SPEC_ACTION_COUNT', 4099)

@c.record
class struct_ibv_flow_eth_filter(c.Struct):
  SIZE = 16
  dst_mac: Annotated[c.Array[uint8_t, Literal[6]], 0]
  src_mac: Annotated[c.Array[uint8_t, Literal[6]], 6]
  ether_type: Annotated[uint16_t, 12]
  vlan_tag: Annotated[uint16_t, 14]
@c.record
class struct_ibv_flow_spec_eth(c.Struct):
  SIZE = 40
  type: Annotated[enum_ibv_flow_spec_type, 0]
  size: Annotated[uint16_t, 4]
  val: Annotated[struct_ibv_flow_eth_filter, 6]
  mask: Annotated[struct_ibv_flow_eth_filter, 22]
@c.record
class struct_ibv_flow_ipv4_filter(c.Struct):
  SIZE = 8
  src_ip: Annotated[uint32_t, 0]
  dst_ip: Annotated[uint32_t, 4]
@c.record
class struct_ibv_flow_spec_ipv4(c.Struct):
  SIZE = 24
  type: Annotated[enum_ibv_flow_spec_type, 0]
  size: Annotated[uint16_t, 4]
  val: Annotated[struct_ibv_flow_ipv4_filter, 8]
  mask: Annotated[struct_ibv_flow_ipv4_filter, 16]
@c.record
class struct_ibv_flow_ipv4_ext_filter(c.Struct):
  SIZE = 12
  src_ip: Annotated[uint32_t, 0]
  dst_ip: Annotated[uint32_t, 4]
  proto: Annotated[uint8_t, 8]
  tos: Annotated[uint8_t, 9]
  ttl: Annotated[uint8_t, 10]
  flags: Annotated[uint8_t, 11]
@c.record
class struct_ibv_flow_spec_ipv4_ext(c.Struct):
  SIZE = 32
  type: Annotated[enum_ibv_flow_spec_type, 0]
  size: Annotated[uint16_t, 4]
  val: Annotated[struct_ibv_flow_ipv4_ext_filter, 8]
  mask: Annotated[struct_ibv_flow_ipv4_ext_filter, 20]
@c.record
class struct_ibv_flow_ipv6_filter(c.Struct):
  SIZE = 40
  src_ip: Annotated[c.Array[uint8_t, Literal[16]], 0]
  dst_ip: Annotated[c.Array[uint8_t, Literal[16]], 16]
  flow_label: Annotated[uint32_t, 32]
  next_hdr: Annotated[uint8_t, 36]
  traffic_class: Annotated[uint8_t, 37]
  hop_limit: Annotated[uint8_t, 38]
@c.record
class struct_ibv_flow_spec_ipv6(c.Struct):
  SIZE = 88
  type: Annotated[enum_ibv_flow_spec_type, 0]
  size: Annotated[uint16_t, 4]
  val: Annotated[struct_ibv_flow_ipv6_filter, 8]
  mask: Annotated[struct_ibv_flow_ipv6_filter, 48]
@c.record
class struct_ibv_flow_esp_filter(c.Struct):
  SIZE = 8
  spi: Annotated[uint32_t, 0]
  seq: Annotated[uint32_t, 4]
@c.record
class struct_ibv_flow_spec_esp(c.Struct):
  SIZE = 24
  type: Annotated[enum_ibv_flow_spec_type, 0]
  size: Annotated[uint16_t, 4]
  val: Annotated[struct_ibv_flow_esp_filter, 8]
  mask: Annotated[struct_ibv_flow_esp_filter, 16]
@c.record
class struct_ibv_flow_tcp_udp_filter(c.Struct):
  SIZE = 4
  dst_port: Annotated[uint16_t, 0]
  src_port: Annotated[uint16_t, 2]
@c.record
class struct_ibv_flow_spec_tcp_udp(c.Struct):
  SIZE = 16
  type: Annotated[enum_ibv_flow_spec_type, 0]
  size: Annotated[uint16_t, 4]
  val: Annotated[struct_ibv_flow_tcp_udp_filter, 6]
  mask: Annotated[struct_ibv_flow_tcp_udp_filter, 10]
@c.record
class struct_ibv_flow_gre_filter(c.Struct):
  SIZE = 8
  c_ks_res0_ver: Annotated[uint16_t, 0]
  protocol: Annotated[uint16_t, 2]
  key: Annotated[uint32_t, 4]
@c.record
class struct_ibv_flow_spec_gre(c.Struct):
  SIZE = 24
  type: Annotated[enum_ibv_flow_spec_type, 0]
  size: Annotated[uint16_t, 4]
  val: Annotated[struct_ibv_flow_gre_filter, 8]
  mask: Annotated[struct_ibv_flow_gre_filter, 16]
@c.record
class struct_ibv_flow_mpls_filter(c.Struct):
  SIZE = 4
  label: Annotated[uint32_t, 0]
@c.record
class struct_ibv_flow_spec_mpls(c.Struct):
  SIZE = 16
  type: Annotated[enum_ibv_flow_spec_type, 0]
  size: Annotated[uint16_t, 4]
  val: Annotated[struct_ibv_flow_mpls_filter, 8]
  mask: Annotated[struct_ibv_flow_mpls_filter, 12]
@c.record
class struct_ibv_flow_tunnel_filter(c.Struct):
  SIZE = 4
  tunnel_id: Annotated[uint32_t, 0]
@c.record
class struct_ibv_flow_spec_tunnel(c.Struct):
  SIZE = 16
  type: Annotated[enum_ibv_flow_spec_type, 0]
  size: Annotated[uint16_t, 4]
  val: Annotated[struct_ibv_flow_tunnel_filter, 8]
  mask: Annotated[struct_ibv_flow_tunnel_filter, 12]
@c.record
class struct_ibv_flow_spec_action_tag(c.Struct):
  SIZE = 12
  type: Annotated[enum_ibv_flow_spec_type, 0]
  size: Annotated[uint16_t, 4]
  tag_id: Annotated[uint32_t, 8]
@c.record
class struct_ibv_flow_spec_action_drop(c.Struct):
  SIZE = 8
  type: Annotated[enum_ibv_flow_spec_type, 0]
  size: Annotated[uint16_t, 4]
@c.record
class struct_ibv_flow_spec_action_handle(c.Struct):
  SIZE = 16
  type: Annotated[enum_ibv_flow_spec_type, 0]
  size: Annotated[uint16_t, 4]
  action: Annotated[c.POINTER[struct_ibv_flow_action], 8]
@c.record
class struct_ibv_flow_action(c.Struct):
  SIZE = 8
  context: Annotated[c.POINTER[struct_ibv_context], 0]
@c.record
class struct_ibv_flow_spec_counter_action(c.Struct):
  SIZE = 16
  type: Annotated[enum_ibv_flow_spec_type, 0]
  size: Annotated[uint16_t, 4]
  counters: Annotated[c.POINTER[struct_ibv_counters], 8]
@c.record
class struct_ibv_counters(c.Struct):
  SIZE = 8
  context: Annotated[c.POINTER[struct_ibv_context], 0]
@c.record
class struct_ibv_flow_spec(c.Struct):
  SIZE = 88
  hdr: Annotated[struct_ibv_flow_spec_hdr, 0]
  eth: Annotated[struct_ibv_flow_spec_eth, 0]
  ipv4: Annotated[struct_ibv_flow_spec_ipv4, 0]
  tcp_udp: Annotated[struct_ibv_flow_spec_tcp_udp, 0]
  ipv4_ext: Annotated[struct_ibv_flow_spec_ipv4_ext, 0]
  ipv6: Annotated[struct_ibv_flow_spec_ipv6, 0]
  esp: Annotated[struct_ibv_flow_spec_esp, 0]
  tunnel: Annotated[struct_ibv_flow_spec_tunnel, 0]
  gre: Annotated[struct_ibv_flow_spec_gre, 0]
  mpls: Annotated[struct_ibv_flow_spec_mpls, 0]
  flow_tag: Annotated[struct_ibv_flow_spec_action_tag, 0]
  drop: Annotated[struct_ibv_flow_spec_action_drop, 0]
  handle: Annotated[struct_ibv_flow_spec_action_handle, 0]
  flow_count: Annotated[struct_ibv_flow_spec_counter_action, 0]
@c.record
class struct_ibv_flow_spec_hdr(c.Struct):
  SIZE = 8
  type: Annotated[enum_ibv_flow_spec_type, 0]
  size: Annotated[uint16_t, 4]
@c.record
class struct_ibv_flow_attr(c.Struct):
  SIZE = 20
  comp_mask: Annotated[uint32_t, 0]
  type: Annotated[enum_ibv_flow_attr_type, 4]
  size: Annotated[uint16_t, 8]
  priority: Annotated[uint16_t, 10]
  num_of_specs: Annotated[uint8_t, 12]
  port: Annotated[uint8_t, 13]
  flags: Annotated[uint32_t, 16]
@c.record
class struct_ibv_flow(c.Struct):
  SIZE = 24
  comp_mask: Annotated[uint32_t, 0]
  context: Annotated[c.POINTER[struct_ibv_context], 8]
  handle: Annotated[uint32_t, 16]
class enum_ibv_flow_action_esp_mask(Annotated[int, ctypes.c_uint32], c.Enum): pass
IBV_FLOW_ACTION_ESP_MASK_ESN = enum_ibv_flow_action_esp_mask.define('IBV_FLOW_ACTION_ESP_MASK_ESN', 1)

@c.record
class struct_ibv_flow_action_esp_attr(c.Struct):
  SIZE = 56
  esp_attr: Annotated[c.POINTER[struct_ib_uverbs_flow_action_esp], 0]
  keymat_proto: Annotated[enum_ib_uverbs_flow_action_esp_keymat, 8]
  keymat_len: Annotated[uint16_t, 12]
  keymat_ptr: Annotated[ctypes.c_void_p, 16]
  replay_proto: Annotated[enum_ib_uverbs_flow_action_esp_replay, 24]
  replay_len: Annotated[uint16_t, 28]
  replay_ptr: Annotated[ctypes.c_void_p, 32]
  esp_encap: Annotated[c.POINTER[struct_ib_uverbs_flow_action_esp_encap], 40]
  comp_mask: Annotated[uint32_t, 48]
  esn: Annotated[uint32_t, 52]
@c.record
class struct_ib_uverbs_flow_action_esp(c.Struct):
  SIZE = 24
  spi: Annotated[Annotated[int, ctypes.c_uint32], 0]
  seq: Annotated[Annotated[int, ctypes.c_uint32], 4]
  tfc_pad: Annotated[Annotated[int, ctypes.c_uint32], 8]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 12]
  hard_limit_pkts: Annotated[Annotated[int, ctypes.c_uint64], 16]
__u32: TypeAlias = Annotated[int, ctypes.c_uint32]
__u64: TypeAlias = Annotated[int, ctypes.c_uint64]
class enum_ib_uverbs_flow_action_esp_keymat(Annotated[int, ctypes.c_uint32], c.Enum): pass
IB_UVERBS_FLOW_ACTION_ESP_KEYMAT_AES_GCM = enum_ib_uverbs_flow_action_esp_keymat.define('IB_UVERBS_FLOW_ACTION_ESP_KEYMAT_AES_GCM', 0)

class enum_ib_uverbs_flow_action_esp_replay(Annotated[int, ctypes.c_uint32], c.Enum): pass
IB_UVERBS_FLOW_ACTION_ESP_REPLAY_NONE = enum_ib_uverbs_flow_action_esp_replay.define('IB_UVERBS_FLOW_ACTION_ESP_REPLAY_NONE', 0)
IB_UVERBS_FLOW_ACTION_ESP_REPLAY_BMP = enum_ib_uverbs_flow_action_esp_replay.define('IB_UVERBS_FLOW_ACTION_ESP_REPLAY_BMP', 1)

@c.record
class struct_ib_uverbs_flow_action_esp_encap(c.Struct):
  SIZE = 24
  val_ptr: Annotated[ctypes.c_void_p, 0]
  val_ptr_data_u64: Annotated[Annotated[int, ctypes.c_uint64], 0]
  next_ptr: Annotated[c.POINTER[struct_ib_uverbs_flow_action_esp_encap], 8]
  next_ptr_data_u64: Annotated[Annotated[int, ctypes.c_uint64], 8]
  len: Annotated[Annotated[int, ctypes.c_uint16], 16]
  type: Annotated[Annotated[int, ctypes.c_uint16], 18]
__u16: TypeAlias = Annotated[int, ctypes.c_uint16]
class _anonenum4(Annotated[int, ctypes.c_uint32], c.Enum): pass
IBV_SYSFS_NAME_MAX = _anonenum4.define('IBV_SYSFS_NAME_MAX', 64)
IBV_SYSFS_PATH_MAX = _anonenum4.define('IBV_SYSFS_PATH_MAX', 256)

class enum_ibv_cq_init_attr_mask(Annotated[int, ctypes.c_uint32], c.Enum): pass
IBV_CQ_INIT_ATTR_MASK_FLAGS = enum_ibv_cq_init_attr_mask.define('IBV_CQ_INIT_ATTR_MASK_FLAGS', 1)
IBV_CQ_INIT_ATTR_MASK_PD = enum_ibv_cq_init_attr_mask.define('IBV_CQ_INIT_ATTR_MASK_PD', 2)

class enum_ibv_create_cq_attr_flags(Annotated[int, ctypes.c_uint32], c.Enum): pass
IBV_CREATE_CQ_ATTR_SINGLE_THREADED = enum_ibv_create_cq_attr_flags.define('IBV_CREATE_CQ_ATTR_SINGLE_THREADED', 1)
IBV_CREATE_CQ_ATTR_IGNORE_OVERRUN = enum_ibv_create_cq_attr_flags.define('IBV_CREATE_CQ_ATTR_IGNORE_OVERRUN', 2)

@c.record
class struct_ibv_cq_init_attr_ex(c.Struct):
  SIZE = 56
  cqe: Annotated[uint32_t, 0]
  cq_context: Annotated[ctypes.c_void_p, 8]
  channel: Annotated[c.POINTER[struct_ibv_comp_channel], 16]
  comp_vector: Annotated[uint32_t, 24]
  wc_flags: Annotated[uint64_t, 32]
  comp_mask: Annotated[uint32_t, 40]
  flags: Annotated[uint32_t, 44]
  parent_domain: Annotated[c.POINTER[struct_ibv_pd], 48]
class enum_ibv_parent_domain_init_attr_mask(Annotated[int, ctypes.c_uint32], c.Enum): pass
IBV_PARENT_DOMAIN_INIT_ATTR_ALLOCATORS = enum_ibv_parent_domain_init_attr_mask.define('IBV_PARENT_DOMAIN_INIT_ATTR_ALLOCATORS', 1)
IBV_PARENT_DOMAIN_INIT_ATTR_PD_CONTEXT = enum_ibv_parent_domain_init_attr_mask.define('IBV_PARENT_DOMAIN_INIT_ATTR_PD_CONTEXT', 2)

@c.record
class struct_ibv_parent_domain_init_attr(c.Struct):
  SIZE = 48
  pd: Annotated[c.POINTER[struct_ibv_pd], 0]
  td: Annotated[c.POINTER[struct_ibv_td], 8]
  comp_mask: Annotated[uint32_t, 16]
  alloc: Annotated[c.CFUNCTYPE[ctypes.c_void_p, [c.POINTER[struct_ibv_pd], ctypes.c_void_p, size_t, size_t, uint64_t]], 24]
  free: Annotated[c.CFUNCTYPE[None, [c.POINTER[struct_ibv_pd], ctypes.c_void_p, ctypes.c_void_p, uint64_t]], 32]
  pd_context: Annotated[ctypes.c_void_p, 40]
@c.record
class struct_ibv_counters_init_attr(c.Struct):
  SIZE = 4
  comp_mask: Annotated[uint32_t, 0]
class enum_ibv_counter_description(Annotated[int, ctypes.c_uint32], c.Enum): pass
IBV_COUNTER_PACKETS = enum_ibv_counter_description.define('IBV_COUNTER_PACKETS', 0)
IBV_COUNTER_BYTES = enum_ibv_counter_description.define('IBV_COUNTER_BYTES', 1)

@c.record
class struct_ibv_counter_attach_attr(c.Struct):
  SIZE = 12
  counter_desc: Annotated[enum_ibv_counter_description, 0]
  index: Annotated[uint32_t, 4]
  comp_mask: Annotated[uint32_t, 8]
class enum_ibv_read_counters_flags(Annotated[int, ctypes.c_uint32], c.Enum): pass
IBV_READ_COUNTERS_ATTR_PREFER_CACHED = enum_ibv_read_counters_flags.define('IBV_READ_COUNTERS_ATTR_PREFER_CACHED', 1)

class enum_ibv_values_mask(Annotated[int, ctypes.c_uint32], c.Enum): pass
IBV_VALUES_MASK_RAW_CLOCK = enum_ibv_values_mask.define('IBV_VALUES_MASK_RAW_CLOCK', 1)
IBV_VALUES_MASK_RESERVED = enum_ibv_values_mask.define('IBV_VALUES_MASK_RESERVED', 2)

@c.record
class struct_ibv_values_ex(c.Struct):
  SIZE = 24
  comp_mask: Annotated[uint32_t, 0]
  raw_clock: Annotated[struct_timespec, 8]
@c.record
class struct_timespec(c.Struct):
  SIZE = 16
  tv_sec: Annotated[Annotated[int, ctypes.c_int64], 0]
  tv_nsec: Annotated[Annotated[int, ctypes.c_int64], 8]
__time_t: TypeAlias = Annotated[int, ctypes.c_int64]
__syscall_slong_t: TypeAlias = Annotated[int, ctypes.c_int64]
@c.record
class struct_verbs_context(c.Struct):
  SIZE = 648
  query_port: Annotated[c.CFUNCTYPE[Annotated[int, ctypes.c_int32], [c.POINTER[struct_ibv_context], uint8_t, c.POINTER[struct_ibv_port_attr], size_t]], 0]
  advise_mr: Annotated[c.CFUNCTYPE[Annotated[int, ctypes.c_int32], [c.POINTER[struct_ibv_pd], enum_ib_uverbs_advise_mr_advice, uint32_t, c.POINTER[struct_ibv_sge], uint32_t]], 8]
  alloc_null_mr: Annotated[c.CFUNCTYPE[c.POINTER[struct_ibv_mr], [c.POINTER[struct_ibv_pd]]], 16]
  read_counters: Annotated[c.CFUNCTYPE[Annotated[int, ctypes.c_int32], [c.POINTER[struct_ibv_counters], c.POINTER[uint64_t], uint32_t, uint32_t]], 24]
  attach_counters_point_flow: Annotated[c.CFUNCTYPE[Annotated[int, ctypes.c_int32], [c.POINTER[struct_ibv_counters], c.POINTER[struct_ibv_counter_attach_attr], c.POINTER[struct_ibv_flow]]], 32]
  create_counters: Annotated[c.CFUNCTYPE[c.POINTER[struct_ibv_counters], [c.POINTER[struct_ibv_context], c.POINTER[struct_ibv_counters_init_attr]]], 40]
  destroy_counters: Annotated[c.CFUNCTYPE[Annotated[int, ctypes.c_int32], [c.POINTER[struct_ibv_counters]]], 48]
  reg_dm_mr: Annotated[c.CFUNCTYPE[c.POINTER[struct_ibv_mr], [c.POINTER[struct_ibv_pd], c.POINTER[struct_ibv_dm], uint64_t, size_t, Annotated[int, ctypes.c_uint32]]], 56]
  alloc_dm: Annotated[c.CFUNCTYPE[c.POINTER[struct_ibv_dm], [c.POINTER[struct_ibv_context], c.POINTER[struct_ibv_alloc_dm_attr]]], 64]
  free_dm: Annotated[c.CFUNCTYPE[Annotated[int, ctypes.c_int32], [c.POINTER[struct_ibv_dm]]], 72]
  modify_flow_action_esp: Annotated[c.CFUNCTYPE[Annotated[int, ctypes.c_int32], [c.POINTER[struct_ibv_flow_action], c.POINTER[struct_ibv_flow_action_esp_attr]]], 80]
  destroy_flow_action: Annotated[c.CFUNCTYPE[Annotated[int, ctypes.c_int32], [c.POINTER[struct_ibv_flow_action]]], 88]
  create_flow_action_esp: Annotated[c.CFUNCTYPE[c.POINTER[struct_ibv_flow_action], [c.POINTER[struct_ibv_context], c.POINTER[struct_ibv_flow_action_esp_attr]]], 96]
  modify_qp_rate_limit: Annotated[c.CFUNCTYPE[Annotated[int, ctypes.c_int32], [c.POINTER[struct_ibv_qp], c.POINTER[struct_ibv_qp_rate_limit_attr]]], 104]
  alloc_parent_domain: Annotated[c.CFUNCTYPE[c.POINTER[struct_ibv_pd], [c.POINTER[struct_ibv_context], c.POINTER[struct_ibv_parent_domain_init_attr]]], 112]
  dealloc_td: Annotated[c.CFUNCTYPE[Annotated[int, ctypes.c_int32], [c.POINTER[struct_ibv_td]]], 120]
  alloc_td: Annotated[c.CFUNCTYPE[c.POINTER[struct_ibv_td], [c.POINTER[struct_ibv_context], c.POINTER[struct_ibv_td_init_attr]]], 128]
  modify_cq: Annotated[c.CFUNCTYPE[Annotated[int, ctypes.c_int32], [c.POINTER[struct_ibv_cq], c.POINTER[struct_ibv_modify_cq_attr]]], 136]
  post_srq_ops: Annotated[c.CFUNCTYPE[Annotated[int, ctypes.c_int32], [c.POINTER[struct_ibv_srq], c.POINTER[struct_ibv_ops_wr], c.POINTER[c.POINTER[struct_ibv_ops_wr]]]], 144]
  destroy_rwq_ind_table: Annotated[c.CFUNCTYPE[Annotated[int, ctypes.c_int32], [c.POINTER[struct_ibv_rwq_ind_table]]], 152]
  create_rwq_ind_table: Annotated[c.CFUNCTYPE[c.POINTER[struct_ibv_rwq_ind_table], [c.POINTER[struct_ibv_context], c.POINTER[struct_ibv_rwq_ind_table_init_attr]]], 160]
  destroy_wq: Annotated[c.CFUNCTYPE[Annotated[int, ctypes.c_int32], [c.POINTER[struct_ibv_wq]]], 168]
  modify_wq: Annotated[c.CFUNCTYPE[Annotated[int, ctypes.c_int32], [c.POINTER[struct_ibv_wq], c.POINTER[struct_ibv_wq_attr]]], 176]
  create_wq: Annotated[c.CFUNCTYPE[c.POINTER[struct_ibv_wq], [c.POINTER[struct_ibv_context], c.POINTER[struct_ibv_wq_init_attr]]], 184]
  query_rt_values: Annotated[c.CFUNCTYPE[Annotated[int, ctypes.c_int32], [c.POINTER[struct_ibv_context], c.POINTER[struct_ibv_values_ex]]], 192]
  create_cq_ex: Annotated[c.CFUNCTYPE[c.POINTER[struct_ibv_cq_ex], [c.POINTER[struct_ibv_context], c.POINTER[struct_ibv_cq_init_attr_ex]]], 200]
  priv: Annotated[c.POINTER[struct_verbs_ex_private], 208]
  query_device_ex: Annotated[c.CFUNCTYPE[Annotated[int, ctypes.c_int32], [c.POINTER[struct_ibv_context], c.POINTER[struct_ibv_query_device_ex_input], c.POINTER[struct_ibv_device_attr_ex], size_t]], 216]
  ibv_destroy_flow: Annotated[c.CFUNCTYPE[Annotated[int, ctypes.c_int32], [c.POINTER[struct_ibv_flow]]], 224]
  ABI_placeholder2: Annotated[c.CFUNCTYPE[None, []], 232]
  ibv_create_flow: Annotated[c.CFUNCTYPE[c.POINTER[struct_ibv_flow], [c.POINTER[struct_ibv_qp], c.POINTER[struct_ibv_flow_attr]]], 240]
  ABI_placeholder1: Annotated[c.CFUNCTYPE[None, []], 248]
  open_qp: Annotated[c.CFUNCTYPE[c.POINTER[struct_ibv_qp], [c.POINTER[struct_ibv_context], c.POINTER[struct_ibv_qp_open_attr]]], 256]
  create_qp_ex: Annotated[c.CFUNCTYPE[c.POINTER[struct_ibv_qp], [c.POINTER[struct_ibv_context], c.POINTER[struct_ibv_qp_init_attr_ex]]], 264]
  get_srq_num: Annotated[c.CFUNCTYPE[Annotated[int, ctypes.c_int32], [c.POINTER[struct_ibv_srq], c.POINTER[uint32_t]]], 272]
  create_srq_ex: Annotated[c.CFUNCTYPE[c.POINTER[struct_ibv_srq], [c.POINTER[struct_ibv_context], c.POINTER[struct_ibv_srq_init_attr_ex]]], 280]
  open_xrcd: Annotated[c.CFUNCTYPE[c.POINTER[struct_ibv_xrcd], [c.POINTER[struct_ibv_context], c.POINTER[struct_ibv_xrcd_init_attr]]], 288]
  close_xrcd: Annotated[c.CFUNCTYPE[Annotated[int, ctypes.c_int32], [c.POINTER[struct_ibv_xrcd]]], 296]
  _ABI_placeholder3: Annotated[uint64_t, 304]
  sz: Annotated[size_t, 312]
  context: Annotated[struct_ibv_context, 320]
class enum_ib_uverbs_advise_mr_advice(Annotated[int, ctypes.c_uint32], c.Enum): pass
IB_UVERBS_ADVISE_MR_ADVICE_PREFETCH = enum_ib_uverbs_advise_mr_advice.define('IB_UVERBS_ADVISE_MR_ADVICE_PREFETCH', 0)
IB_UVERBS_ADVISE_MR_ADVICE_PREFETCH_WRITE = enum_ib_uverbs_advise_mr_advice.define('IB_UVERBS_ADVISE_MR_ADVICE_PREFETCH_WRITE', 1)
IB_UVERBS_ADVISE_MR_ADVICE_PREFETCH_NO_FAULT = enum_ib_uverbs_advise_mr_advice.define('IB_UVERBS_ADVISE_MR_ADVICE_PREFETCH_NO_FAULT', 2)

class struct_verbs_ex_private(ctypes.Structure): pass
@dll.bind
def ibv_get_device_list(num_devices:c.POINTER[Annotated[int, ctypes.c_int32]]) -> c.POINTER[c.POINTER[struct_ibv_device]]: ...
@dll.bind
def ibv_free_device_list(list:c.POINTER[c.POINTER[struct_ibv_device]]) -> None: ...
@dll.bind
def ibv_get_device_name(device:c.POINTER[struct_ibv_device]) -> c.POINTER[Annotated[bytes, ctypes.c_char]]: ...
@dll.bind
def ibv_get_device_index(device:c.POINTER[struct_ibv_device]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def ibv_get_device_guid(device:c.POINTER[struct_ibv_device]) -> Annotated[int, ctypes.c_uint64]: ...
@dll.bind
def ibv_open_device(device:c.POINTER[struct_ibv_device]) -> c.POINTER[struct_ibv_context]: ...
@dll.bind
def ibv_close_device(context:c.POINTER[struct_ibv_context]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def ibv_import_device(cmd_fd:Annotated[int, ctypes.c_int32]) -> c.POINTER[struct_ibv_context]: ...
@dll.bind
def ibv_import_pd(context:c.POINTER[struct_ibv_context], pd_handle:uint32_t) -> c.POINTER[struct_ibv_pd]: ...
@dll.bind
def ibv_unimport_pd(pd:c.POINTER[struct_ibv_pd]) -> None: ...
@dll.bind
def ibv_import_mr(pd:c.POINTER[struct_ibv_pd], mr_handle:uint32_t) -> c.POINTER[struct_ibv_mr]: ...
@dll.bind
def ibv_unimport_mr(mr:c.POINTER[struct_ibv_mr]) -> None: ...
@dll.bind
def ibv_import_dm(context:c.POINTER[struct_ibv_context], dm_handle:uint32_t) -> c.POINTER[struct_ibv_dm]: ...
@dll.bind
def ibv_unimport_dm(dm:c.POINTER[struct_ibv_dm]) -> None: ...
@dll.bind
def ibv_get_async_event(context:c.POINTER[struct_ibv_context], event:c.POINTER[struct_ibv_async_event]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def ibv_ack_async_event(event:c.POINTER[struct_ibv_async_event]) -> None: ...
@dll.bind
def ibv_query_device(context:c.POINTER[struct_ibv_context], device_attr:c.POINTER[struct_ibv_device_attr]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def ibv_query_port(context:c.POINTER[struct_ibv_context], port_num:uint8_t, port_attr:c.POINTER[struct__compat_ibv_port_attr]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def ibv_query_gid(context:c.POINTER[struct_ibv_context], port_num:uint8_t, index:Annotated[int, ctypes.c_int32], gid:c.POINTER[union_ibv_gid]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def _ibv_query_gid_ex(context:c.POINTER[struct_ibv_context], port_num:uint32_t, gid_index:uint32_t, entry:c.POINTER[struct_ibv_gid_entry], flags:uint32_t, entry_size:size_t) -> Annotated[int, ctypes.c_int32]: ...
ssize_t: TypeAlias = Annotated[int, ctypes.c_int64]
@dll.bind
def _ibv_query_gid_table(context:c.POINTER[struct_ibv_context], entries:c.POINTER[struct_ibv_gid_entry], max_entries:size_t, flags:uint32_t, entry_size:size_t) -> ssize_t: ...
@dll.bind
def ibv_query_pkey(context:c.POINTER[struct_ibv_context], port_num:uint8_t, index:Annotated[int, ctypes.c_int32], pkey:c.POINTER[Annotated[int, ctypes.c_uint16]]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def ibv_get_pkey_index(context:c.POINTER[struct_ibv_context], port_num:uint8_t, pkey:Annotated[int, ctypes.c_uint16]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def ibv_alloc_pd(context:c.POINTER[struct_ibv_context]) -> c.POINTER[struct_ibv_pd]: ...
@dll.bind
def ibv_dealloc_pd(pd:c.POINTER[struct_ibv_pd]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def ibv_reg_mr_iova2(pd:c.POINTER[struct_ibv_pd], addr:ctypes.c_void_p, length:size_t, iova:uint64_t, access:Annotated[int, ctypes.c_uint32]) -> c.POINTER[struct_ibv_mr]: ...
@dll.bind
def ibv_reg_mr(pd:c.POINTER[struct_ibv_pd], addr:ctypes.c_void_p, length:size_t, access:Annotated[int, ctypes.c_int32]) -> c.POINTER[struct_ibv_mr]: ...
@dll.bind
def ibv_reg_mr_iova(pd:c.POINTER[struct_ibv_pd], addr:ctypes.c_void_p, length:size_t, iova:uint64_t, access:Annotated[int, ctypes.c_int32]) -> c.POINTER[struct_ibv_mr]: ...
@dll.bind
def ibv_reg_dmabuf_mr(pd:c.POINTER[struct_ibv_pd], offset:uint64_t, length:size_t, iova:uint64_t, fd:Annotated[int, ctypes.c_int32], access:Annotated[int, ctypes.c_int32]) -> c.POINTER[struct_ibv_mr]: ...
class enum_ibv_rereg_mr_err_code(Annotated[int, ctypes.c_int32], c.Enum): pass
IBV_REREG_MR_ERR_INPUT = enum_ibv_rereg_mr_err_code.define('IBV_REREG_MR_ERR_INPUT', -1)
IBV_REREG_MR_ERR_DONT_FORK_NEW = enum_ibv_rereg_mr_err_code.define('IBV_REREG_MR_ERR_DONT_FORK_NEW', -2)
IBV_REREG_MR_ERR_DO_FORK_OLD = enum_ibv_rereg_mr_err_code.define('IBV_REREG_MR_ERR_DO_FORK_OLD', -3)
IBV_REREG_MR_ERR_CMD = enum_ibv_rereg_mr_err_code.define('IBV_REREG_MR_ERR_CMD', -4)
IBV_REREG_MR_ERR_CMD_AND_DO_FORK_NEW = enum_ibv_rereg_mr_err_code.define('IBV_REREG_MR_ERR_CMD_AND_DO_FORK_NEW', -5)

@dll.bind
def ibv_rereg_mr(mr:c.POINTER[struct_ibv_mr], flags:Annotated[int, ctypes.c_int32], pd:c.POINTER[struct_ibv_pd], addr:ctypes.c_void_p, length:size_t, access:Annotated[int, ctypes.c_int32]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def ibv_dereg_mr(mr:c.POINTER[struct_ibv_mr]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def ibv_create_comp_channel(context:c.POINTER[struct_ibv_context]) -> c.POINTER[struct_ibv_comp_channel]: ...
@dll.bind
def ibv_destroy_comp_channel(channel:c.POINTER[struct_ibv_comp_channel]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def ibv_create_cq(context:c.POINTER[struct_ibv_context], cqe:Annotated[int, ctypes.c_int32], cq_context:ctypes.c_void_p, channel:c.POINTER[struct_ibv_comp_channel], comp_vector:Annotated[int, ctypes.c_int32]) -> c.POINTER[struct_ibv_cq]: ...
@dll.bind
def ibv_resize_cq(cq:c.POINTER[struct_ibv_cq], cqe:Annotated[int, ctypes.c_int32]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def ibv_destroy_cq(cq:c.POINTER[struct_ibv_cq]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def ibv_get_cq_event(channel:c.POINTER[struct_ibv_comp_channel], cq:c.POINTER[c.POINTER[struct_ibv_cq]], cq_context:c.POINTER[ctypes.c_void_p]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def ibv_ack_cq_events(cq:c.POINTER[struct_ibv_cq], nevents:Annotated[int, ctypes.c_uint32]) -> None: ...
@dll.bind
def ibv_create_srq(pd:c.POINTER[struct_ibv_pd], srq_init_attr:c.POINTER[struct_ibv_srq_init_attr]) -> c.POINTER[struct_ibv_srq]: ...
@dll.bind
def ibv_modify_srq(srq:c.POINTER[struct_ibv_srq], srq_attr:c.POINTER[struct_ibv_srq_attr], srq_attr_mask:Annotated[int, ctypes.c_int32]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def ibv_query_srq(srq:c.POINTER[struct_ibv_srq], srq_attr:c.POINTER[struct_ibv_srq_attr]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def ibv_destroy_srq(srq:c.POINTER[struct_ibv_srq]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def ibv_create_qp(pd:c.POINTER[struct_ibv_pd], qp_init_attr:c.POINTER[struct_ibv_qp_init_attr]) -> c.POINTER[struct_ibv_qp]: ...
@dll.bind
def ibv_modify_qp(qp:c.POINTER[struct_ibv_qp], attr:c.POINTER[struct_ibv_qp_attr], attr_mask:Annotated[int, ctypes.c_int32]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def ibv_query_qp_data_in_order(qp:c.POINTER[struct_ibv_qp], op:enum_ibv_wr_opcode, flags:uint32_t) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def ibv_query_qp(qp:c.POINTER[struct_ibv_qp], attr:c.POINTER[struct_ibv_qp_attr], attr_mask:Annotated[int, ctypes.c_int32], init_attr:c.POINTER[struct_ibv_qp_init_attr]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def ibv_destroy_qp(qp:c.POINTER[struct_ibv_qp]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def ibv_create_ah(pd:c.POINTER[struct_ibv_pd], attr:c.POINTER[struct_ibv_ah_attr]) -> c.POINTER[struct_ibv_ah]: ...
@dll.bind
def ibv_init_ah_from_wc(context:c.POINTER[struct_ibv_context], port_num:uint8_t, wc:c.POINTER[struct_ibv_wc], grh:c.POINTER[struct_ibv_grh], ah_attr:c.POINTER[struct_ibv_ah_attr]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def ibv_create_ah_from_wc(pd:c.POINTER[struct_ibv_pd], wc:c.POINTER[struct_ibv_wc], grh:c.POINTER[struct_ibv_grh], port_num:uint8_t) -> c.POINTER[struct_ibv_ah]: ...
@dll.bind
def ibv_destroy_ah(ah:c.POINTER[struct_ibv_ah]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def ibv_attach_mcast(qp:c.POINTER[struct_ibv_qp], gid:c.POINTER[union_ibv_gid], lid:uint16_t) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def ibv_detach_mcast(qp:c.POINTER[struct_ibv_qp], gid:c.POINTER[union_ibv_gid], lid:uint16_t) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def ibv_fork_init() -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def ibv_is_fork_initialized() -> enum_ibv_fork_status: ...
@dll.bind
def ibv_node_type_str(node_type:enum_ibv_node_type) -> c.POINTER[Annotated[bytes, ctypes.c_char]]: ...
@dll.bind
def ibv_port_state_str(port_state:enum_ibv_port_state) -> c.POINTER[Annotated[bytes, ctypes.c_char]]: ...
@dll.bind
def ibv_event_type_str(event:enum_ibv_event_type) -> c.POINTER[Annotated[bytes, ctypes.c_char]]: ...
@dll.bind
def ibv_resolve_eth_l2_from_gid(context:c.POINTER[struct_ibv_context], attr:c.POINTER[struct_ibv_ah_attr], eth_mac:c.Array[uint8_t, Literal[6]], vid:c.POINTER[uint16_t]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def ibv_set_ece(qp:c.POINTER[struct_ibv_qp], ece:c.POINTER[struct_ibv_ece]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def ibv_query_ece(qp:c.POINTER[struct_ibv_qp], ece:c.POINTER[struct_ibv_ece]) -> Annotated[int, ctypes.c_int32]: ...
class enum_ib_uverbs_core_support(Annotated[int, ctypes.c_uint32], c.Enum): pass
IB_UVERBS_CORE_SUPPORT_OPTIONAL_MR_ACCESS = enum_ib_uverbs_core_support.define('IB_UVERBS_CORE_SUPPORT_OPTIONAL_MR_ACCESS', 1)

class enum_ib_uverbs_access_flags(Annotated[int, ctypes.c_uint32], c.Enum): pass
IB_UVERBS_ACCESS_LOCAL_WRITE = enum_ib_uverbs_access_flags.define('IB_UVERBS_ACCESS_LOCAL_WRITE', 1)
IB_UVERBS_ACCESS_REMOTE_WRITE = enum_ib_uverbs_access_flags.define('IB_UVERBS_ACCESS_REMOTE_WRITE', 2)
IB_UVERBS_ACCESS_REMOTE_READ = enum_ib_uverbs_access_flags.define('IB_UVERBS_ACCESS_REMOTE_READ', 4)
IB_UVERBS_ACCESS_REMOTE_ATOMIC = enum_ib_uverbs_access_flags.define('IB_UVERBS_ACCESS_REMOTE_ATOMIC', 8)
IB_UVERBS_ACCESS_MW_BIND = enum_ib_uverbs_access_flags.define('IB_UVERBS_ACCESS_MW_BIND', 16)
IB_UVERBS_ACCESS_ZERO_BASED = enum_ib_uverbs_access_flags.define('IB_UVERBS_ACCESS_ZERO_BASED', 32)
IB_UVERBS_ACCESS_ON_DEMAND = enum_ib_uverbs_access_flags.define('IB_UVERBS_ACCESS_ON_DEMAND', 64)
IB_UVERBS_ACCESS_HUGETLB = enum_ib_uverbs_access_flags.define('IB_UVERBS_ACCESS_HUGETLB', 128)
IB_UVERBS_ACCESS_FLUSH_GLOBAL = enum_ib_uverbs_access_flags.define('IB_UVERBS_ACCESS_FLUSH_GLOBAL', 256)
IB_UVERBS_ACCESS_FLUSH_PERSISTENT = enum_ib_uverbs_access_flags.define('IB_UVERBS_ACCESS_FLUSH_PERSISTENT', 512)
IB_UVERBS_ACCESS_RELAXED_ORDERING = enum_ib_uverbs_access_flags.define('IB_UVERBS_ACCESS_RELAXED_ORDERING', 1048576)
IB_UVERBS_ACCESS_OPTIONAL_RANGE = enum_ib_uverbs_access_flags.define('IB_UVERBS_ACCESS_OPTIONAL_RANGE', 1072693248)

class enum_ib_uverbs_srq_type(Annotated[int, ctypes.c_uint32], c.Enum): pass
IB_UVERBS_SRQT_BASIC = enum_ib_uverbs_srq_type.define('IB_UVERBS_SRQT_BASIC', 0)
IB_UVERBS_SRQT_XRC = enum_ib_uverbs_srq_type.define('IB_UVERBS_SRQT_XRC', 1)
IB_UVERBS_SRQT_TM = enum_ib_uverbs_srq_type.define('IB_UVERBS_SRQT_TM', 2)

class enum_ib_uverbs_wq_type(Annotated[int, ctypes.c_uint32], c.Enum): pass
IB_UVERBS_WQT_RQ = enum_ib_uverbs_wq_type.define('IB_UVERBS_WQT_RQ', 0)

class enum_ib_uverbs_wq_flags(Annotated[int, ctypes.c_uint32], c.Enum): pass
IB_UVERBS_WQ_FLAGS_CVLAN_STRIPPING = enum_ib_uverbs_wq_flags.define('IB_UVERBS_WQ_FLAGS_CVLAN_STRIPPING', 1)
IB_UVERBS_WQ_FLAGS_SCATTER_FCS = enum_ib_uverbs_wq_flags.define('IB_UVERBS_WQ_FLAGS_SCATTER_FCS', 2)
IB_UVERBS_WQ_FLAGS_DELAY_DROP = enum_ib_uverbs_wq_flags.define('IB_UVERBS_WQ_FLAGS_DELAY_DROP', 4)
IB_UVERBS_WQ_FLAGS_PCI_WRITE_END_PADDING = enum_ib_uverbs_wq_flags.define('IB_UVERBS_WQ_FLAGS_PCI_WRITE_END_PADDING', 8)

class enum_ib_uverbs_qp_type(Annotated[int, ctypes.c_uint32], c.Enum): pass
IB_UVERBS_QPT_RC = enum_ib_uverbs_qp_type.define('IB_UVERBS_QPT_RC', 2)
IB_UVERBS_QPT_UC = enum_ib_uverbs_qp_type.define('IB_UVERBS_QPT_UC', 3)
IB_UVERBS_QPT_UD = enum_ib_uverbs_qp_type.define('IB_UVERBS_QPT_UD', 4)
IB_UVERBS_QPT_RAW_PACKET = enum_ib_uverbs_qp_type.define('IB_UVERBS_QPT_RAW_PACKET', 8)
IB_UVERBS_QPT_XRC_INI = enum_ib_uverbs_qp_type.define('IB_UVERBS_QPT_XRC_INI', 9)
IB_UVERBS_QPT_XRC_TGT = enum_ib_uverbs_qp_type.define('IB_UVERBS_QPT_XRC_TGT', 10)
IB_UVERBS_QPT_DRIVER = enum_ib_uverbs_qp_type.define('IB_UVERBS_QPT_DRIVER', 255)

class enum_ib_uverbs_qp_create_flags(Annotated[int, ctypes.c_uint32], c.Enum): pass
IB_UVERBS_QP_CREATE_BLOCK_MULTICAST_LOOPBACK = enum_ib_uverbs_qp_create_flags.define('IB_UVERBS_QP_CREATE_BLOCK_MULTICAST_LOOPBACK', 2)
IB_UVERBS_QP_CREATE_SCATTER_FCS = enum_ib_uverbs_qp_create_flags.define('IB_UVERBS_QP_CREATE_SCATTER_FCS', 256)
IB_UVERBS_QP_CREATE_CVLAN_STRIPPING = enum_ib_uverbs_qp_create_flags.define('IB_UVERBS_QP_CREATE_CVLAN_STRIPPING', 512)
IB_UVERBS_QP_CREATE_PCI_WRITE_END_PADDING = enum_ib_uverbs_qp_create_flags.define('IB_UVERBS_QP_CREATE_PCI_WRITE_END_PADDING', 2048)
IB_UVERBS_QP_CREATE_SQ_SIG_ALL = enum_ib_uverbs_qp_create_flags.define('IB_UVERBS_QP_CREATE_SQ_SIG_ALL', 4096)

class enum_ib_uverbs_query_port_cap_flags(Annotated[int, ctypes.c_uint32], c.Enum): pass
IB_UVERBS_PCF_SM = enum_ib_uverbs_query_port_cap_flags.define('IB_UVERBS_PCF_SM', 2)
IB_UVERBS_PCF_NOTICE_SUP = enum_ib_uverbs_query_port_cap_flags.define('IB_UVERBS_PCF_NOTICE_SUP', 4)
IB_UVERBS_PCF_TRAP_SUP = enum_ib_uverbs_query_port_cap_flags.define('IB_UVERBS_PCF_TRAP_SUP', 8)
IB_UVERBS_PCF_OPT_IPD_SUP = enum_ib_uverbs_query_port_cap_flags.define('IB_UVERBS_PCF_OPT_IPD_SUP', 16)
IB_UVERBS_PCF_AUTO_MIGR_SUP = enum_ib_uverbs_query_port_cap_flags.define('IB_UVERBS_PCF_AUTO_MIGR_SUP', 32)
IB_UVERBS_PCF_SL_MAP_SUP = enum_ib_uverbs_query_port_cap_flags.define('IB_UVERBS_PCF_SL_MAP_SUP', 64)
IB_UVERBS_PCF_MKEY_NVRAM = enum_ib_uverbs_query_port_cap_flags.define('IB_UVERBS_PCF_MKEY_NVRAM', 128)
IB_UVERBS_PCF_PKEY_NVRAM = enum_ib_uverbs_query_port_cap_flags.define('IB_UVERBS_PCF_PKEY_NVRAM', 256)
IB_UVERBS_PCF_LED_INFO_SUP = enum_ib_uverbs_query_port_cap_flags.define('IB_UVERBS_PCF_LED_INFO_SUP', 512)
IB_UVERBS_PCF_SM_DISABLED = enum_ib_uverbs_query_port_cap_flags.define('IB_UVERBS_PCF_SM_DISABLED', 1024)
IB_UVERBS_PCF_SYS_IMAGE_GUID_SUP = enum_ib_uverbs_query_port_cap_flags.define('IB_UVERBS_PCF_SYS_IMAGE_GUID_SUP', 2048)
IB_UVERBS_PCF_PKEY_SW_EXT_PORT_TRAP_SUP = enum_ib_uverbs_query_port_cap_flags.define('IB_UVERBS_PCF_PKEY_SW_EXT_PORT_TRAP_SUP', 4096)
IB_UVERBS_PCF_EXTENDED_SPEEDS_SUP = enum_ib_uverbs_query_port_cap_flags.define('IB_UVERBS_PCF_EXTENDED_SPEEDS_SUP', 16384)
IB_UVERBS_PCF_CM_SUP = enum_ib_uverbs_query_port_cap_flags.define('IB_UVERBS_PCF_CM_SUP', 65536)
IB_UVERBS_PCF_SNMP_TUNNEL_SUP = enum_ib_uverbs_query_port_cap_flags.define('IB_UVERBS_PCF_SNMP_TUNNEL_SUP', 131072)
IB_UVERBS_PCF_REINIT_SUP = enum_ib_uverbs_query_port_cap_flags.define('IB_UVERBS_PCF_REINIT_SUP', 262144)
IB_UVERBS_PCF_DEVICE_MGMT_SUP = enum_ib_uverbs_query_port_cap_flags.define('IB_UVERBS_PCF_DEVICE_MGMT_SUP', 524288)
IB_UVERBS_PCF_VENDOR_CLASS_SUP = enum_ib_uverbs_query_port_cap_flags.define('IB_UVERBS_PCF_VENDOR_CLASS_SUP', 1048576)
IB_UVERBS_PCF_DR_NOTICE_SUP = enum_ib_uverbs_query_port_cap_flags.define('IB_UVERBS_PCF_DR_NOTICE_SUP', 2097152)
IB_UVERBS_PCF_CAP_MASK_NOTICE_SUP = enum_ib_uverbs_query_port_cap_flags.define('IB_UVERBS_PCF_CAP_MASK_NOTICE_SUP', 4194304)
IB_UVERBS_PCF_BOOT_MGMT_SUP = enum_ib_uverbs_query_port_cap_flags.define('IB_UVERBS_PCF_BOOT_MGMT_SUP', 8388608)
IB_UVERBS_PCF_LINK_LATENCY_SUP = enum_ib_uverbs_query_port_cap_flags.define('IB_UVERBS_PCF_LINK_LATENCY_SUP', 16777216)
IB_UVERBS_PCF_CLIENT_REG_SUP = enum_ib_uverbs_query_port_cap_flags.define('IB_UVERBS_PCF_CLIENT_REG_SUP', 33554432)
IB_UVERBS_PCF_LINK_SPEED_WIDTH_TABLE_SUP = enum_ib_uverbs_query_port_cap_flags.define('IB_UVERBS_PCF_LINK_SPEED_WIDTH_TABLE_SUP', 134217728)
IB_UVERBS_PCF_VENDOR_SPECIFIC_MADS_TABLE_SUP = enum_ib_uverbs_query_port_cap_flags.define('IB_UVERBS_PCF_VENDOR_SPECIFIC_MADS_TABLE_SUP', 268435456)
IB_UVERBS_PCF_MCAST_PKEY_TRAP_SUPPRESSION_SUP = enum_ib_uverbs_query_port_cap_flags.define('IB_UVERBS_PCF_MCAST_PKEY_TRAP_SUPPRESSION_SUP', 536870912)
IB_UVERBS_PCF_MCAST_FDB_TOP_SUP = enum_ib_uverbs_query_port_cap_flags.define('IB_UVERBS_PCF_MCAST_FDB_TOP_SUP', 1073741824)
IB_UVERBS_PCF_HIERARCHY_INFO_SUP = enum_ib_uverbs_query_port_cap_flags.define('IB_UVERBS_PCF_HIERARCHY_INFO_SUP', 2147483648)
IB_UVERBS_PCF_IP_BASED_GIDS = enum_ib_uverbs_query_port_cap_flags.define('IB_UVERBS_PCF_IP_BASED_GIDS', 67108864)

class enum_ib_uverbs_query_port_flags(Annotated[int, ctypes.c_uint32], c.Enum): pass
IB_UVERBS_QPF_GRH_REQUIRED = enum_ib_uverbs_query_port_flags.define('IB_UVERBS_QPF_GRH_REQUIRED', 1)

class enum_ib_uverbs_flow_action_esp_keymat_aes_gcm_iv_algo(Annotated[int, ctypes.c_uint32], c.Enum): pass
IB_UVERBS_FLOW_ACTION_IV_ALGO_SEQ = enum_ib_uverbs_flow_action_esp_keymat_aes_gcm_iv_algo.define('IB_UVERBS_FLOW_ACTION_IV_ALGO_SEQ', 0)

@c.record
class struct_ib_uverbs_flow_action_esp_keymat_aes_gcm(c.Struct):
  SIZE = 56
  iv: Annotated[Annotated[int, ctypes.c_uint64], 0]
  iv_algo: Annotated[Annotated[int, ctypes.c_uint32], 8]
  salt: Annotated[Annotated[int, ctypes.c_uint32], 12]
  icv_len: Annotated[Annotated[int, ctypes.c_uint32], 16]
  key_len: Annotated[Annotated[int, ctypes.c_uint32], 20]
  aes_key: Annotated[c.Array[Annotated[int, ctypes.c_uint32], Literal[8]], 24]
@c.record
class struct_ib_uverbs_flow_action_esp_replay_bmp(c.Struct):
  SIZE = 4
  size: Annotated[Annotated[int, ctypes.c_uint32], 0]
class enum_ib_uverbs_flow_action_esp_flags(Annotated[int, ctypes.c_uint32], c.Enum): pass
IB_UVERBS_FLOW_ACTION_ESP_FLAGS_INLINE_CRYPTO = enum_ib_uverbs_flow_action_esp_flags.define('IB_UVERBS_FLOW_ACTION_ESP_FLAGS_INLINE_CRYPTO', 0)
IB_UVERBS_FLOW_ACTION_ESP_FLAGS_FULL_OFFLOAD = enum_ib_uverbs_flow_action_esp_flags.define('IB_UVERBS_FLOW_ACTION_ESP_FLAGS_FULL_OFFLOAD', 1)
IB_UVERBS_FLOW_ACTION_ESP_FLAGS_TUNNEL = enum_ib_uverbs_flow_action_esp_flags.define('IB_UVERBS_FLOW_ACTION_ESP_FLAGS_TUNNEL', 0)
IB_UVERBS_FLOW_ACTION_ESP_FLAGS_TRANSPORT = enum_ib_uverbs_flow_action_esp_flags.define('IB_UVERBS_FLOW_ACTION_ESP_FLAGS_TRANSPORT', 2)
IB_UVERBS_FLOW_ACTION_ESP_FLAGS_DECRYPT = enum_ib_uverbs_flow_action_esp_flags.define('IB_UVERBS_FLOW_ACTION_ESP_FLAGS_DECRYPT', 0)
IB_UVERBS_FLOW_ACTION_ESP_FLAGS_ENCRYPT = enum_ib_uverbs_flow_action_esp_flags.define('IB_UVERBS_FLOW_ACTION_ESP_FLAGS_ENCRYPT', 4)
IB_UVERBS_FLOW_ACTION_ESP_FLAGS_ESN_NEW_WINDOW = enum_ib_uverbs_flow_action_esp_flags.define('IB_UVERBS_FLOW_ACTION_ESP_FLAGS_ESN_NEW_WINDOW', 8)

class enum_ib_uverbs_read_counters_flags(Annotated[int, ctypes.c_uint32], c.Enum): pass
IB_UVERBS_READ_COUNTERS_PREFER_CACHED = enum_ib_uverbs_read_counters_flags.define('IB_UVERBS_READ_COUNTERS_PREFER_CACHED', 1)

class enum_ib_uverbs_advise_mr_flag(Annotated[int, ctypes.c_uint32], c.Enum): pass
IB_UVERBS_ADVISE_MR_FLAG_FLUSH = enum_ib_uverbs_advise_mr_flag.define('IB_UVERBS_ADVISE_MR_FLAG_FLUSH', 1)

@c.record
class struct_ib_uverbs_query_port_resp_ex(c.Struct):
  SIZE = 48
  legacy_resp: Annotated[struct_ib_uverbs_query_port_resp, 0]
  port_cap_flags2: Annotated[Annotated[int, ctypes.c_uint16], 40]
  reserved: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 42]
  active_speed_ex: Annotated[Annotated[int, ctypes.c_uint32], 44]
@c.record
class struct_ib_uverbs_query_port_resp(c.Struct):
  SIZE = 40
  port_cap_flags: Annotated[Annotated[int, ctypes.c_uint32], 0]
  max_msg_sz: Annotated[Annotated[int, ctypes.c_uint32], 4]
  bad_pkey_cntr: Annotated[Annotated[int, ctypes.c_uint32], 8]
  qkey_viol_cntr: Annotated[Annotated[int, ctypes.c_uint32], 12]
  gid_tbl_len: Annotated[Annotated[int, ctypes.c_uint32], 16]
  pkey_tbl_len: Annotated[Annotated[int, ctypes.c_uint16], 20]
  lid: Annotated[Annotated[int, ctypes.c_uint16], 22]
  sm_lid: Annotated[Annotated[int, ctypes.c_uint16], 24]
  state: Annotated[Annotated[int, ctypes.c_ubyte], 26]
  max_mtu: Annotated[Annotated[int, ctypes.c_ubyte], 27]
  active_mtu: Annotated[Annotated[int, ctypes.c_ubyte], 28]
  lmc: Annotated[Annotated[int, ctypes.c_ubyte], 29]
  max_vl_num: Annotated[Annotated[int, ctypes.c_ubyte], 30]
  sm_sl: Annotated[Annotated[int, ctypes.c_ubyte], 31]
  subnet_timeout: Annotated[Annotated[int, ctypes.c_ubyte], 32]
  init_type_reply: Annotated[Annotated[int, ctypes.c_ubyte], 33]
  active_width: Annotated[Annotated[int, ctypes.c_ubyte], 34]
  active_speed: Annotated[Annotated[int, ctypes.c_ubyte], 35]
  phys_state: Annotated[Annotated[int, ctypes.c_ubyte], 36]
  link_layer: Annotated[Annotated[int, ctypes.c_ubyte], 37]
  flags: Annotated[Annotated[int, ctypes.c_ubyte], 38]
  reserved: Annotated[Annotated[int, ctypes.c_ubyte], 39]
__u8: TypeAlias = Annotated[int, ctypes.c_ubyte]
@c.record
class struct_ib_uverbs_qp_cap(c.Struct):
  SIZE = 20
  max_send_wr: Annotated[Annotated[int, ctypes.c_uint32], 0]
  max_recv_wr: Annotated[Annotated[int, ctypes.c_uint32], 4]
  max_send_sge: Annotated[Annotated[int, ctypes.c_uint32], 8]
  max_recv_sge: Annotated[Annotated[int, ctypes.c_uint32], 12]
  max_inline_data: Annotated[Annotated[int, ctypes.c_uint32], 16]
class enum_rdma_driver_id(Annotated[int, ctypes.c_uint32], c.Enum): pass
RDMA_DRIVER_UNKNOWN = enum_rdma_driver_id.define('RDMA_DRIVER_UNKNOWN', 0)
RDMA_DRIVER_MLX5 = enum_rdma_driver_id.define('RDMA_DRIVER_MLX5', 1)
RDMA_DRIVER_MLX4 = enum_rdma_driver_id.define('RDMA_DRIVER_MLX4', 2)
RDMA_DRIVER_CXGB3 = enum_rdma_driver_id.define('RDMA_DRIVER_CXGB3', 3)
RDMA_DRIVER_CXGB4 = enum_rdma_driver_id.define('RDMA_DRIVER_CXGB4', 4)
RDMA_DRIVER_MTHCA = enum_rdma_driver_id.define('RDMA_DRIVER_MTHCA', 5)
RDMA_DRIVER_BNXT_RE = enum_rdma_driver_id.define('RDMA_DRIVER_BNXT_RE', 6)
RDMA_DRIVER_OCRDMA = enum_rdma_driver_id.define('RDMA_DRIVER_OCRDMA', 7)
RDMA_DRIVER_NES = enum_rdma_driver_id.define('RDMA_DRIVER_NES', 8)
RDMA_DRIVER_I40IW = enum_rdma_driver_id.define('RDMA_DRIVER_I40IW', 9)
RDMA_DRIVER_IRDMA = enum_rdma_driver_id.define('RDMA_DRIVER_IRDMA', 9)
RDMA_DRIVER_VMW_PVRDMA = enum_rdma_driver_id.define('RDMA_DRIVER_VMW_PVRDMA', 10)
RDMA_DRIVER_QEDR = enum_rdma_driver_id.define('RDMA_DRIVER_QEDR', 11)
RDMA_DRIVER_HNS = enum_rdma_driver_id.define('RDMA_DRIVER_HNS', 12)
RDMA_DRIVER_USNIC = enum_rdma_driver_id.define('RDMA_DRIVER_USNIC', 13)
RDMA_DRIVER_RXE = enum_rdma_driver_id.define('RDMA_DRIVER_RXE', 14)
RDMA_DRIVER_HFI1 = enum_rdma_driver_id.define('RDMA_DRIVER_HFI1', 15)
RDMA_DRIVER_QIB = enum_rdma_driver_id.define('RDMA_DRIVER_QIB', 16)
RDMA_DRIVER_EFA = enum_rdma_driver_id.define('RDMA_DRIVER_EFA', 17)
RDMA_DRIVER_SIW = enum_rdma_driver_id.define('RDMA_DRIVER_SIW', 18)
RDMA_DRIVER_ERDMA = enum_rdma_driver_id.define('RDMA_DRIVER_ERDMA', 19)
RDMA_DRIVER_MANA = enum_rdma_driver_id.define('RDMA_DRIVER_MANA', 20)

class enum_ib_uverbs_gid_type(Annotated[int, ctypes.c_uint32], c.Enum): pass
IB_UVERBS_GID_TYPE_IB = enum_ib_uverbs_gid_type.define('IB_UVERBS_GID_TYPE_IB', 0)
IB_UVERBS_GID_TYPE_ROCE_V1 = enum_ib_uverbs_gid_type.define('IB_UVERBS_GID_TYPE_ROCE_V1', 1)
IB_UVERBS_GID_TYPE_ROCE_V2 = enum_ib_uverbs_gid_type.define('IB_UVERBS_GID_TYPE_ROCE_V2', 2)

@c.record
class struct_ib_uverbs_gid_entry(c.Struct):
  SIZE = 32
  gid: Annotated[c.Array[Annotated[int, ctypes.c_uint64], Literal[2]], 0]
  gid_index: Annotated[Annotated[int, ctypes.c_uint32], 16]
  port_num: Annotated[Annotated[int, ctypes.c_uint32], 20]
  gid_type: Annotated[Annotated[int, ctypes.c_uint32], 24]
  netdev_ifindex: Annotated[Annotated[int, ctypes.c_uint32], 28]
class enum_ib_uverbs_write_cmds(Annotated[int, ctypes.c_uint32], c.Enum): pass
IB_USER_VERBS_CMD_GET_CONTEXT = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_GET_CONTEXT', 0)
IB_USER_VERBS_CMD_QUERY_DEVICE = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_QUERY_DEVICE', 1)
IB_USER_VERBS_CMD_QUERY_PORT = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_QUERY_PORT', 2)
IB_USER_VERBS_CMD_ALLOC_PD = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_ALLOC_PD', 3)
IB_USER_VERBS_CMD_DEALLOC_PD = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_DEALLOC_PD', 4)
IB_USER_VERBS_CMD_CREATE_AH = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_CREATE_AH', 5)
IB_USER_VERBS_CMD_MODIFY_AH = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_MODIFY_AH', 6)
IB_USER_VERBS_CMD_QUERY_AH = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_QUERY_AH', 7)
IB_USER_VERBS_CMD_DESTROY_AH = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_DESTROY_AH', 8)
IB_USER_VERBS_CMD_REG_MR = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_REG_MR', 9)
IB_USER_VERBS_CMD_REG_SMR = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_REG_SMR', 10)
IB_USER_VERBS_CMD_REREG_MR = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_REREG_MR', 11)
IB_USER_VERBS_CMD_QUERY_MR = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_QUERY_MR', 12)
IB_USER_VERBS_CMD_DEREG_MR = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_DEREG_MR', 13)
IB_USER_VERBS_CMD_ALLOC_MW = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_ALLOC_MW', 14)
IB_USER_VERBS_CMD_BIND_MW = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_BIND_MW', 15)
IB_USER_VERBS_CMD_DEALLOC_MW = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_DEALLOC_MW', 16)
IB_USER_VERBS_CMD_CREATE_COMP_CHANNEL = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_CREATE_COMP_CHANNEL', 17)
IB_USER_VERBS_CMD_CREATE_CQ = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_CREATE_CQ', 18)
IB_USER_VERBS_CMD_RESIZE_CQ = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_RESIZE_CQ', 19)
IB_USER_VERBS_CMD_DESTROY_CQ = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_DESTROY_CQ', 20)
IB_USER_VERBS_CMD_POLL_CQ = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_POLL_CQ', 21)
IB_USER_VERBS_CMD_PEEK_CQ = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_PEEK_CQ', 22)
IB_USER_VERBS_CMD_REQ_NOTIFY_CQ = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_REQ_NOTIFY_CQ', 23)
IB_USER_VERBS_CMD_CREATE_QP = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_CREATE_QP', 24)
IB_USER_VERBS_CMD_QUERY_QP = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_QUERY_QP', 25)
IB_USER_VERBS_CMD_MODIFY_QP = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_MODIFY_QP', 26)
IB_USER_VERBS_CMD_DESTROY_QP = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_DESTROY_QP', 27)
IB_USER_VERBS_CMD_POST_SEND = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_POST_SEND', 28)
IB_USER_VERBS_CMD_POST_RECV = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_POST_RECV', 29)
IB_USER_VERBS_CMD_ATTACH_MCAST = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_ATTACH_MCAST', 30)
IB_USER_VERBS_CMD_DETACH_MCAST = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_DETACH_MCAST', 31)
IB_USER_VERBS_CMD_CREATE_SRQ = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_CREATE_SRQ', 32)
IB_USER_VERBS_CMD_MODIFY_SRQ = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_MODIFY_SRQ', 33)
IB_USER_VERBS_CMD_QUERY_SRQ = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_QUERY_SRQ', 34)
IB_USER_VERBS_CMD_DESTROY_SRQ = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_DESTROY_SRQ', 35)
IB_USER_VERBS_CMD_POST_SRQ_RECV = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_POST_SRQ_RECV', 36)
IB_USER_VERBS_CMD_OPEN_XRCD = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_OPEN_XRCD', 37)
IB_USER_VERBS_CMD_CLOSE_XRCD = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_CLOSE_XRCD', 38)
IB_USER_VERBS_CMD_CREATE_XSRQ = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_CREATE_XSRQ', 39)
IB_USER_VERBS_CMD_OPEN_QP = enum_ib_uverbs_write_cmds.define('IB_USER_VERBS_CMD_OPEN_QP', 40)

class _anonenum5(Annotated[int, ctypes.c_uint32], c.Enum): pass
IB_USER_VERBS_EX_CMD_QUERY_DEVICE = _anonenum5.define('IB_USER_VERBS_EX_CMD_QUERY_DEVICE', 1)
IB_USER_VERBS_EX_CMD_CREATE_CQ = _anonenum5.define('IB_USER_VERBS_EX_CMD_CREATE_CQ', 18)
IB_USER_VERBS_EX_CMD_CREATE_QP = _anonenum5.define('IB_USER_VERBS_EX_CMD_CREATE_QP', 24)
IB_USER_VERBS_EX_CMD_MODIFY_QP = _anonenum5.define('IB_USER_VERBS_EX_CMD_MODIFY_QP', 26)
IB_USER_VERBS_EX_CMD_CREATE_FLOW = _anonenum5.define('IB_USER_VERBS_EX_CMD_CREATE_FLOW', 50)
IB_USER_VERBS_EX_CMD_DESTROY_FLOW = _anonenum5.define('IB_USER_VERBS_EX_CMD_DESTROY_FLOW', 51)
IB_USER_VERBS_EX_CMD_CREATE_WQ = _anonenum5.define('IB_USER_VERBS_EX_CMD_CREATE_WQ', 52)
IB_USER_VERBS_EX_CMD_MODIFY_WQ = _anonenum5.define('IB_USER_VERBS_EX_CMD_MODIFY_WQ', 53)
IB_USER_VERBS_EX_CMD_DESTROY_WQ = _anonenum5.define('IB_USER_VERBS_EX_CMD_DESTROY_WQ', 54)
IB_USER_VERBS_EX_CMD_CREATE_RWQ_IND_TBL = _anonenum5.define('IB_USER_VERBS_EX_CMD_CREATE_RWQ_IND_TBL', 55)
IB_USER_VERBS_EX_CMD_DESTROY_RWQ_IND_TBL = _anonenum5.define('IB_USER_VERBS_EX_CMD_DESTROY_RWQ_IND_TBL', 56)
IB_USER_VERBS_EX_CMD_MODIFY_CQ = _anonenum5.define('IB_USER_VERBS_EX_CMD_MODIFY_CQ', 57)

class enum_ib_placement_type(Annotated[int, ctypes.c_uint32], c.Enum): pass
IB_FLUSH_GLOBAL = enum_ib_placement_type.define('IB_FLUSH_GLOBAL', 1)
IB_FLUSH_PERSISTENT = enum_ib_placement_type.define('IB_FLUSH_PERSISTENT', 2)

class enum_ib_selectivity_level(Annotated[int, ctypes.c_uint32], c.Enum): pass
IB_FLUSH_RANGE = enum_ib_selectivity_level.define('IB_FLUSH_RANGE', 0)
IB_FLUSH_MR = enum_ib_selectivity_level.define('IB_FLUSH_MR', 1)

@c.record
class struct_ib_uverbs_async_event_desc(c.Struct):
  SIZE = 16
  element: Annotated[Annotated[int, ctypes.c_uint64], 0]
  event_type: Annotated[Annotated[int, ctypes.c_uint32], 8]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 12]
@c.record
class struct_ib_uverbs_comp_event_desc(c.Struct):
  SIZE = 8
  cq_handle: Annotated[Annotated[int, ctypes.c_uint64], 0]
@c.record
class struct_ib_uverbs_cq_moderation_caps(c.Struct):
  SIZE = 8
  max_cq_moderation_count: Annotated[Annotated[int, ctypes.c_uint16], 0]
  max_cq_moderation_period: Annotated[Annotated[int, ctypes.c_uint16], 2]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_ib_uverbs_cmd_hdr(c.Struct):
  SIZE = 8
  command: Annotated[Annotated[int, ctypes.c_uint32], 0]
  in_words: Annotated[Annotated[int, ctypes.c_uint16], 4]
  out_words: Annotated[Annotated[int, ctypes.c_uint16], 6]
@c.record
class struct_ib_uverbs_ex_cmd_hdr(c.Struct):
  SIZE = 16
  response: Annotated[Annotated[int, ctypes.c_uint64], 0]
  provider_in_words: Annotated[Annotated[int, ctypes.c_uint16], 8]
  provider_out_words: Annotated[Annotated[int, ctypes.c_uint16], 10]
  cmd_hdr_reserved: Annotated[Annotated[int, ctypes.c_uint32], 12]
@c.record
class struct_ib_uverbs_get_context(c.Struct):
  SIZE = 8
  response: Annotated[Annotated[int, ctypes.c_uint64], 0]
  driver_data: Annotated[c.Array[Annotated[int, ctypes.c_uint64], Literal[0]], 8]
@c.record
class struct_ib_uverbs_get_context_resp(c.Struct):
  SIZE = 8
  async_fd: Annotated[Annotated[int, ctypes.c_uint32], 0]
  num_comp_vectors: Annotated[Annotated[int, ctypes.c_uint32], 4]
  driver_data: Annotated[c.Array[Annotated[int, ctypes.c_uint64], Literal[0]], 8]
@c.record
class struct_ib_uverbs_query_device(c.Struct):
  SIZE = 8
  response: Annotated[Annotated[int, ctypes.c_uint64], 0]
  driver_data: Annotated[c.Array[Annotated[int, ctypes.c_uint64], Literal[0]], 8]
@c.record
class struct_ib_uverbs_query_device_resp(c.Struct):
  SIZE = 176
  fw_ver: Annotated[Annotated[int, ctypes.c_uint64], 0]
  node_guid: Annotated[Annotated[int, ctypes.c_uint64], 8]
  sys_image_guid: Annotated[Annotated[int, ctypes.c_uint64], 16]
  max_mr_size: Annotated[Annotated[int, ctypes.c_uint64], 24]
  page_size_cap: Annotated[Annotated[int, ctypes.c_uint64], 32]
  vendor_id: Annotated[Annotated[int, ctypes.c_uint32], 40]
  vendor_part_id: Annotated[Annotated[int, ctypes.c_uint32], 44]
  hw_ver: Annotated[Annotated[int, ctypes.c_uint32], 48]
  max_qp: Annotated[Annotated[int, ctypes.c_uint32], 52]
  max_qp_wr: Annotated[Annotated[int, ctypes.c_uint32], 56]
  device_cap_flags: Annotated[Annotated[int, ctypes.c_uint32], 60]
  max_sge: Annotated[Annotated[int, ctypes.c_uint32], 64]
  max_sge_rd: Annotated[Annotated[int, ctypes.c_uint32], 68]
  max_cq: Annotated[Annotated[int, ctypes.c_uint32], 72]
  max_cqe: Annotated[Annotated[int, ctypes.c_uint32], 76]
  max_mr: Annotated[Annotated[int, ctypes.c_uint32], 80]
  max_pd: Annotated[Annotated[int, ctypes.c_uint32], 84]
  max_qp_rd_atom: Annotated[Annotated[int, ctypes.c_uint32], 88]
  max_ee_rd_atom: Annotated[Annotated[int, ctypes.c_uint32], 92]
  max_res_rd_atom: Annotated[Annotated[int, ctypes.c_uint32], 96]
  max_qp_init_rd_atom: Annotated[Annotated[int, ctypes.c_uint32], 100]
  max_ee_init_rd_atom: Annotated[Annotated[int, ctypes.c_uint32], 104]
  atomic_cap: Annotated[Annotated[int, ctypes.c_uint32], 108]
  max_ee: Annotated[Annotated[int, ctypes.c_uint32], 112]
  max_rdd: Annotated[Annotated[int, ctypes.c_uint32], 116]
  max_mw: Annotated[Annotated[int, ctypes.c_uint32], 120]
  max_raw_ipv6_qp: Annotated[Annotated[int, ctypes.c_uint32], 124]
  max_raw_ethy_qp: Annotated[Annotated[int, ctypes.c_uint32], 128]
  max_mcast_grp: Annotated[Annotated[int, ctypes.c_uint32], 132]
  max_mcast_qp_attach: Annotated[Annotated[int, ctypes.c_uint32], 136]
  max_total_mcast_qp_attach: Annotated[Annotated[int, ctypes.c_uint32], 140]
  max_ah: Annotated[Annotated[int, ctypes.c_uint32], 144]
  max_fmr: Annotated[Annotated[int, ctypes.c_uint32], 148]
  max_map_per_fmr: Annotated[Annotated[int, ctypes.c_uint32], 152]
  max_srq: Annotated[Annotated[int, ctypes.c_uint32], 156]
  max_srq_wr: Annotated[Annotated[int, ctypes.c_uint32], 160]
  max_srq_sge: Annotated[Annotated[int, ctypes.c_uint32], 164]
  max_pkeys: Annotated[Annotated[int, ctypes.c_uint16], 168]
  local_ca_ack_delay: Annotated[Annotated[int, ctypes.c_ubyte], 170]
  phys_port_cnt: Annotated[Annotated[int, ctypes.c_ubyte], 171]
  reserved: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[4]], 172]
@c.record
class struct_ib_uverbs_ex_query_device(c.Struct):
  SIZE = 8
  comp_mask: Annotated[Annotated[int, ctypes.c_uint32], 0]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_ib_uverbs_odp_caps(c.Struct):
  SIZE = 24
  general_caps: Annotated[Annotated[int, ctypes.c_uint64], 0]
  per_transport_caps: Annotated[struct_ib_uverbs_odp_caps_per_transport_caps, 8]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 20]
@c.record
class struct_ib_uverbs_odp_caps_per_transport_caps(c.Struct):
  SIZE = 12
  rc_odp_caps: Annotated[Annotated[int, ctypes.c_uint32], 0]
  uc_odp_caps: Annotated[Annotated[int, ctypes.c_uint32], 4]
  ud_odp_caps: Annotated[Annotated[int, ctypes.c_uint32], 8]
@c.record
class struct_ib_uverbs_rss_caps(c.Struct):
  SIZE = 16
  supported_qpts: Annotated[Annotated[int, ctypes.c_uint32], 0]
  max_rwq_indirection_tables: Annotated[Annotated[int, ctypes.c_uint32], 4]
  max_rwq_indirection_table_size: Annotated[Annotated[int, ctypes.c_uint32], 8]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 12]
@c.record
class struct_ib_uverbs_tm_caps(c.Struct):
  SIZE = 24
  max_rndv_hdr_size: Annotated[Annotated[int, ctypes.c_uint32], 0]
  max_num_tags: Annotated[Annotated[int, ctypes.c_uint32], 4]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 8]
  max_ops: Annotated[Annotated[int, ctypes.c_uint32], 12]
  max_sge: Annotated[Annotated[int, ctypes.c_uint32], 16]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 20]
@c.record
class struct_ib_uverbs_ex_query_device_resp(c.Struct):
  SIZE = 304
  base: Annotated[struct_ib_uverbs_query_device_resp, 0]
  comp_mask: Annotated[Annotated[int, ctypes.c_uint32], 176]
  response_length: Annotated[Annotated[int, ctypes.c_uint32], 180]
  odp_caps: Annotated[struct_ib_uverbs_odp_caps, 184]
  timestamp_mask: Annotated[Annotated[int, ctypes.c_uint64], 208]
  hca_core_clock: Annotated[Annotated[int, ctypes.c_uint64], 216]
  device_cap_flags_ex: Annotated[Annotated[int, ctypes.c_uint64], 224]
  rss_caps: Annotated[struct_ib_uverbs_rss_caps, 232]
  max_wq_type_rq: Annotated[Annotated[int, ctypes.c_uint32], 248]
  raw_packet_caps: Annotated[Annotated[int, ctypes.c_uint32], 252]
  tm_caps: Annotated[struct_ib_uverbs_tm_caps, 256]
  cq_moderation_caps: Annotated[struct_ib_uverbs_cq_moderation_caps, 280]
  max_dm_size: Annotated[Annotated[int, ctypes.c_uint64], 288]
  xrc_odp_caps: Annotated[Annotated[int, ctypes.c_uint32], 296]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 300]
@c.record
class struct_ib_uverbs_query_port(c.Struct):
  SIZE = 16
  response: Annotated[Annotated[int, ctypes.c_uint64], 0]
  port_num: Annotated[Annotated[int, ctypes.c_ubyte], 8]
  reserved: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[7]], 9]
  driver_data: Annotated[c.Array[Annotated[int, ctypes.c_uint64], Literal[0]], 16]
@c.record
class struct_ib_uverbs_alloc_pd(c.Struct):
  SIZE = 8
  response: Annotated[Annotated[int, ctypes.c_uint64], 0]
  driver_data: Annotated[c.Array[Annotated[int, ctypes.c_uint64], Literal[0]], 8]
@c.record
class struct_ib_uverbs_alloc_pd_resp(c.Struct):
  SIZE = 4
  pd_handle: Annotated[Annotated[int, ctypes.c_uint32], 0]
  driver_data: Annotated[c.Array[Annotated[int, ctypes.c_uint32], Literal[0]], 4]
@c.record
class struct_ib_uverbs_dealloc_pd(c.Struct):
  SIZE = 4
  pd_handle: Annotated[Annotated[int, ctypes.c_uint32], 0]
@c.record
class struct_ib_uverbs_open_xrcd(c.Struct):
  SIZE = 16
  response: Annotated[Annotated[int, ctypes.c_uint64], 0]
  fd: Annotated[Annotated[int, ctypes.c_uint32], 8]
  oflags: Annotated[Annotated[int, ctypes.c_uint32], 12]
  driver_data: Annotated[c.Array[Annotated[int, ctypes.c_uint64], Literal[0]], 16]
@c.record
class struct_ib_uverbs_open_xrcd_resp(c.Struct):
  SIZE = 4
  xrcd_handle: Annotated[Annotated[int, ctypes.c_uint32], 0]
  driver_data: Annotated[c.Array[Annotated[int, ctypes.c_uint32], Literal[0]], 4]
@c.record
class struct_ib_uverbs_close_xrcd(c.Struct):
  SIZE = 4
  xrcd_handle: Annotated[Annotated[int, ctypes.c_uint32], 0]
@c.record
class struct_ib_uverbs_reg_mr(c.Struct):
  SIZE = 40
  response: Annotated[Annotated[int, ctypes.c_uint64], 0]
  start: Annotated[Annotated[int, ctypes.c_uint64], 8]
  length: Annotated[Annotated[int, ctypes.c_uint64], 16]
  hca_va: Annotated[Annotated[int, ctypes.c_uint64], 24]
  pd_handle: Annotated[Annotated[int, ctypes.c_uint32], 32]
  access_flags: Annotated[Annotated[int, ctypes.c_uint32], 36]
  driver_data: Annotated[c.Array[Annotated[int, ctypes.c_uint64], Literal[0]], 40]
@c.record
class struct_ib_uverbs_reg_mr_resp(c.Struct):
  SIZE = 12
  mr_handle: Annotated[Annotated[int, ctypes.c_uint32], 0]
  lkey: Annotated[Annotated[int, ctypes.c_uint32], 4]
  rkey: Annotated[Annotated[int, ctypes.c_uint32], 8]
  driver_data: Annotated[c.Array[Annotated[int, ctypes.c_uint32], Literal[0]], 12]
@c.record
class struct_ib_uverbs_rereg_mr(c.Struct):
  SIZE = 48
  response: Annotated[Annotated[int, ctypes.c_uint64], 0]
  mr_handle: Annotated[Annotated[int, ctypes.c_uint32], 8]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 12]
  start: Annotated[Annotated[int, ctypes.c_uint64], 16]
  length: Annotated[Annotated[int, ctypes.c_uint64], 24]
  hca_va: Annotated[Annotated[int, ctypes.c_uint64], 32]
  pd_handle: Annotated[Annotated[int, ctypes.c_uint32], 40]
  access_flags: Annotated[Annotated[int, ctypes.c_uint32], 44]
  driver_data: Annotated[c.Array[Annotated[int, ctypes.c_uint64], Literal[0]], 48]
@c.record
class struct_ib_uverbs_rereg_mr_resp(c.Struct):
  SIZE = 8
  lkey: Annotated[Annotated[int, ctypes.c_uint32], 0]
  rkey: Annotated[Annotated[int, ctypes.c_uint32], 4]
  driver_data: Annotated[c.Array[Annotated[int, ctypes.c_uint64], Literal[0]], 8]
@c.record
class struct_ib_uverbs_dereg_mr(c.Struct):
  SIZE = 4
  mr_handle: Annotated[Annotated[int, ctypes.c_uint32], 0]
@c.record
class struct_ib_uverbs_alloc_mw(c.Struct):
  SIZE = 16
  response: Annotated[Annotated[int, ctypes.c_uint64], 0]
  pd_handle: Annotated[Annotated[int, ctypes.c_uint32], 8]
  mw_type: Annotated[Annotated[int, ctypes.c_ubyte], 12]
  reserved: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[3]], 13]
  driver_data: Annotated[c.Array[Annotated[int, ctypes.c_uint64], Literal[0]], 16]
@c.record
class struct_ib_uverbs_alloc_mw_resp(c.Struct):
  SIZE = 8
  mw_handle: Annotated[Annotated[int, ctypes.c_uint32], 0]
  rkey: Annotated[Annotated[int, ctypes.c_uint32], 4]
  driver_data: Annotated[c.Array[Annotated[int, ctypes.c_uint64], Literal[0]], 8]
@c.record
class struct_ib_uverbs_dealloc_mw(c.Struct):
  SIZE = 4
  mw_handle: Annotated[Annotated[int, ctypes.c_uint32], 0]
@c.record
class struct_ib_uverbs_create_comp_channel(c.Struct):
  SIZE = 8
  response: Annotated[Annotated[int, ctypes.c_uint64], 0]
@c.record
class struct_ib_uverbs_create_comp_channel_resp(c.Struct):
  SIZE = 4
  fd: Annotated[Annotated[int, ctypes.c_uint32], 0]
@c.record
class struct_ib_uverbs_create_cq(c.Struct):
  SIZE = 32
  response: Annotated[Annotated[int, ctypes.c_uint64], 0]
  user_handle: Annotated[Annotated[int, ctypes.c_uint64], 8]
  cqe: Annotated[Annotated[int, ctypes.c_uint32], 16]
  comp_vector: Annotated[Annotated[int, ctypes.c_uint32], 20]
  comp_channel: Annotated[Annotated[int, ctypes.c_int32], 24]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 28]
  driver_data: Annotated[c.Array[Annotated[int, ctypes.c_uint64], Literal[0]], 32]
__s32: TypeAlias = Annotated[int, ctypes.c_int32]
class enum_ib_uverbs_ex_create_cq_flags(Annotated[int, ctypes.c_uint32], c.Enum): pass
IB_UVERBS_CQ_FLAGS_TIMESTAMP_COMPLETION = enum_ib_uverbs_ex_create_cq_flags.define('IB_UVERBS_CQ_FLAGS_TIMESTAMP_COMPLETION', 1)
IB_UVERBS_CQ_FLAGS_IGNORE_OVERRUN = enum_ib_uverbs_ex_create_cq_flags.define('IB_UVERBS_CQ_FLAGS_IGNORE_OVERRUN', 2)

@c.record
class struct_ib_uverbs_ex_create_cq(c.Struct):
  SIZE = 32
  user_handle: Annotated[Annotated[int, ctypes.c_uint64], 0]
  cqe: Annotated[Annotated[int, ctypes.c_uint32], 8]
  comp_vector: Annotated[Annotated[int, ctypes.c_uint32], 12]
  comp_channel: Annotated[Annotated[int, ctypes.c_int32], 16]
  comp_mask: Annotated[Annotated[int, ctypes.c_uint32], 20]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 24]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 28]
@c.record
class struct_ib_uverbs_create_cq_resp(c.Struct):
  SIZE = 8
  cq_handle: Annotated[Annotated[int, ctypes.c_uint32], 0]
  cqe: Annotated[Annotated[int, ctypes.c_uint32], 4]
  driver_data: Annotated[c.Array[Annotated[int, ctypes.c_uint64], Literal[0]], 8]
@c.record
class struct_ib_uverbs_ex_create_cq_resp(c.Struct):
  SIZE = 16
  base: Annotated[struct_ib_uverbs_create_cq_resp, 0]
  comp_mask: Annotated[Annotated[int, ctypes.c_uint32], 8]
  response_length: Annotated[Annotated[int, ctypes.c_uint32], 12]
@c.record
class struct_ib_uverbs_resize_cq(c.Struct):
  SIZE = 16
  response: Annotated[Annotated[int, ctypes.c_uint64], 0]
  cq_handle: Annotated[Annotated[int, ctypes.c_uint32], 8]
  cqe: Annotated[Annotated[int, ctypes.c_uint32], 12]
  driver_data: Annotated[c.Array[Annotated[int, ctypes.c_uint64], Literal[0]], 16]
@c.record
class struct_ib_uverbs_resize_cq_resp(c.Struct):
  SIZE = 8
  cqe: Annotated[Annotated[int, ctypes.c_uint32], 0]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 4]
  driver_data: Annotated[c.Array[Annotated[int, ctypes.c_uint64], Literal[0]], 8]
@c.record
class struct_ib_uverbs_poll_cq(c.Struct):
  SIZE = 16
  response: Annotated[Annotated[int, ctypes.c_uint64], 0]
  cq_handle: Annotated[Annotated[int, ctypes.c_uint32], 8]
  ne: Annotated[Annotated[int, ctypes.c_uint32], 12]
class enum_ib_uverbs_wc_opcode(Annotated[int, ctypes.c_uint32], c.Enum): pass
IB_UVERBS_WC_SEND = enum_ib_uverbs_wc_opcode.define('IB_UVERBS_WC_SEND', 0)
IB_UVERBS_WC_RDMA_WRITE = enum_ib_uverbs_wc_opcode.define('IB_UVERBS_WC_RDMA_WRITE', 1)
IB_UVERBS_WC_RDMA_READ = enum_ib_uverbs_wc_opcode.define('IB_UVERBS_WC_RDMA_READ', 2)
IB_UVERBS_WC_COMP_SWAP = enum_ib_uverbs_wc_opcode.define('IB_UVERBS_WC_COMP_SWAP', 3)
IB_UVERBS_WC_FETCH_ADD = enum_ib_uverbs_wc_opcode.define('IB_UVERBS_WC_FETCH_ADD', 4)
IB_UVERBS_WC_BIND_MW = enum_ib_uverbs_wc_opcode.define('IB_UVERBS_WC_BIND_MW', 5)
IB_UVERBS_WC_LOCAL_INV = enum_ib_uverbs_wc_opcode.define('IB_UVERBS_WC_LOCAL_INV', 6)
IB_UVERBS_WC_TSO = enum_ib_uverbs_wc_opcode.define('IB_UVERBS_WC_TSO', 7)
IB_UVERBS_WC_FLUSH = enum_ib_uverbs_wc_opcode.define('IB_UVERBS_WC_FLUSH', 8)
IB_UVERBS_WC_ATOMIC_WRITE = enum_ib_uverbs_wc_opcode.define('IB_UVERBS_WC_ATOMIC_WRITE', 9)

@c.record
class struct_ib_uverbs_wc(c.Struct):
  SIZE = 48
  wr_id: Annotated[Annotated[int, ctypes.c_uint64], 0]
  status: Annotated[Annotated[int, ctypes.c_uint32], 8]
  opcode: Annotated[Annotated[int, ctypes.c_uint32], 12]
  vendor_err: Annotated[Annotated[int, ctypes.c_uint32], 16]
  byte_len: Annotated[Annotated[int, ctypes.c_uint32], 20]
  ex: Annotated[struct_ib_uverbs_wc_ex, 24]
  qp_num: Annotated[Annotated[int, ctypes.c_uint32], 28]
  src_qp: Annotated[Annotated[int, ctypes.c_uint32], 32]
  wc_flags: Annotated[Annotated[int, ctypes.c_uint32], 36]
  pkey_index: Annotated[Annotated[int, ctypes.c_uint16], 40]
  slid: Annotated[Annotated[int, ctypes.c_uint16], 42]
  sl: Annotated[Annotated[int, ctypes.c_ubyte], 44]
  dlid_path_bits: Annotated[Annotated[int, ctypes.c_ubyte], 45]
  port_num: Annotated[Annotated[int, ctypes.c_ubyte], 46]
  reserved: Annotated[Annotated[int, ctypes.c_ubyte], 47]
@c.record
class struct_ib_uverbs_wc_ex(c.Struct):
  SIZE = 4
  imm_data: Annotated[Annotated[int, ctypes.c_uint32], 0]
  invalidate_rkey: Annotated[Annotated[int, ctypes.c_uint32], 0]
@c.record
class struct_ib_uverbs_poll_cq_resp(c.Struct):
  SIZE = 8
  count: Annotated[Annotated[int, ctypes.c_uint32], 0]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 4]
  wc: Annotated[c.Array[struct_ib_uverbs_wc, Literal[0]], 8]
@c.record
class struct_ib_uverbs_req_notify_cq(c.Struct):
  SIZE = 8
  cq_handle: Annotated[Annotated[int, ctypes.c_uint32], 0]
  solicited_only: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_ib_uverbs_destroy_cq(c.Struct):
  SIZE = 16
  response: Annotated[Annotated[int, ctypes.c_uint64], 0]
  cq_handle: Annotated[Annotated[int, ctypes.c_uint32], 8]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 12]
@c.record
class struct_ib_uverbs_destroy_cq_resp(c.Struct):
  SIZE = 8
  comp_events_reported: Annotated[Annotated[int, ctypes.c_uint32], 0]
  async_events_reported: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_ib_uverbs_global_route(c.Struct):
  SIZE = 24
  dgid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  flow_label: Annotated[Annotated[int, ctypes.c_uint32], 16]
  sgid_index: Annotated[Annotated[int, ctypes.c_ubyte], 20]
  hop_limit: Annotated[Annotated[int, ctypes.c_ubyte], 21]
  traffic_class: Annotated[Annotated[int, ctypes.c_ubyte], 22]
  reserved: Annotated[Annotated[int, ctypes.c_ubyte], 23]
@c.record
class struct_ib_uverbs_ah_attr(c.Struct):
  SIZE = 32
  grh: Annotated[struct_ib_uverbs_global_route, 0]
  dlid: Annotated[Annotated[int, ctypes.c_uint16], 24]
  sl: Annotated[Annotated[int, ctypes.c_ubyte], 26]
  src_path_bits: Annotated[Annotated[int, ctypes.c_ubyte], 27]
  static_rate: Annotated[Annotated[int, ctypes.c_ubyte], 28]
  is_global: Annotated[Annotated[int, ctypes.c_ubyte], 29]
  port_num: Annotated[Annotated[int, ctypes.c_ubyte], 30]
  reserved: Annotated[Annotated[int, ctypes.c_ubyte], 31]
@c.record
class struct_ib_uverbs_qp_attr(c.Struct):
  SIZE = 144
  qp_attr_mask: Annotated[Annotated[int, ctypes.c_uint32], 0]
  qp_state: Annotated[Annotated[int, ctypes.c_uint32], 4]
  cur_qp_state: Annotated[Annotated[int, ctypes.c_uint32], 8]
  path_mtu: Annotated[Annotated[int, ctypes.c_uint32], 12]
  path_mig_state: Annotated[Annotated[int, ctypes.c_uint32], 16]
  qkey: Annotated[Annotated[int, ctypes.c_uint32], 20]
  rq_psn: Annotated[Annotated[int, ctypes.c_uint32], 24]
  sq_psn: Annotated[Annotated[int, ctypes.c_uint32], 28]
  dest_qp_num: Annotated[Annotated[int, ctypes.c_uint32], 32]
  qp_access_flags: Annotated[Annotated[int, ctypes.c_uint32], 36]
  ah_attr: Annotated[struct_ib_uverbs_ah_attr, 40]
  alt_ah_attr: Annotated[struct_ib_uverbs_ah_attr, 72]
  max_send_wr: Annotated[Annotated[int, ctypes.c_uint32], 104]
  max_recv_wr: Annotated[Annotated[int, ctypes.c_uint32], 108]
  max_send_sge: Annotated[Annotated[int, ctypes.c_uint32], 112]
  max_recv_sge: Annotated[Annotated[int, ctypes.c_uint32], 116]
  max_inline_data: Annotated[Annotated[int, ctypes.c_uint32], 120]
  pkey_index: Annotated[Annotated[int, ctypes.c_uint16], 124]
  alt_pkey_index: Annotated[Annotated[int, ctypes.c_uint16], 126]
  en_sqd_async_notify: Annotated[Annotated[int, ctypes.c_ubyte], 128]
  sq_draining: Annotated[Annotated[int, ctypes.c_ubyte], 129]
  max_rd_atomic: Annotated[Annotated[int, ctypes.c_ubyte], 130]
  max_dest_rd_atomic: Annotated[Annotated[int, ctypes.c_ubyte], 131]
  min_rnr_timer: Annotated[Annotated[int, ctypes.c_ubyte], 132]
  port_num: Annotated[Annotated[int, ctypes.c_ubyte], 133]
  timeout: Annotated[Annotated[int, ctypes.c_ubyte], 134]
  retry_cnt: Annotated[Annotated[int, ctypes.c_ubyte], 135]
  rnr_retry: Annotated[Annotated[int, ctypes.c_ubyte], 136]
  alt_port_num: Annotated[Annotated[int, ctypes.c_ubyte], 137]
  alt_timeout: Annotated[Annotated[int, ctypes.c_ubyte], 138]
  reserved: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 139]
@c.record
class struct_ib_uverbs_create_qp(c.Struct):
  SIZE = 56
  response: Annotated[Annotated[int, ctypes.c_uint64], 0]
  user_handle: Annotated[Annotated[int, ctypes.c_uint64], 8]
  pd_handle: Annotated[Annotated[int, ctypes.c_uint32], 16]
  send_cq_handle: Annotated[Annotated[int, ctypes.c_uint32], 20]
  recv_cq_handle: Annotated[Annotated[int, ctypes.c_uint32], 24]
  srq_handle: Annotated[Annotated[int, ctypes.c_uint32], 28]
  max_send_wr: Annotated[Annotated[int, ctypes.c_uint32], 32]
  max_recv_wr: Annotated[Annotated[int, ctypes.c_uint32], 36]
  max_send_sge: Annotated[Annotated[int, ctypes.c_uint32], 40]
  max_recv_sge: Annotated[Annotated[int, ctypes.c_uint32], 44]
  max_inline_data: Annotated[Annotated[int, ctypes.c_uint32], 48]
  sq_sig_all: Annotated[Annotated[int, ctypes.c_ubyte], 52]
  qp_type: Annotated[Annotated[int, ctypes.c_ubyte], 53]
  is_srq: Annotated[Annotated[int, ctypes.c_ubyte], 54]
  reserved: Annotated[Annotated[int, ctypes.c_ubyte], 55]
  driver_data: Annotated[c.Array[Annotated[int, ctypes.c_uint64], Literal[0]], 56]
class enum_ib_uverbs_create_qp_mask(Annotated[int, ctypes.c_uint32], c.Enum): pass
IB_UVERBS_CREATE_QP_MASK_IND_TABLE = enum_ib_uverbs_create_qp_mask.define('IB_UVERBS_CREATE_QP_MASK_IND_TABLE', 1)

class _anonenum6(Annotated[int, ctypes.c_uint32], c.Enum): pass
IB_UVERBS_CREATE_QP_SUP_COMP_MASK = _anonenum6.define('IB_UVERBS_CREATE_QP_SUP_COMP_MASK', 1)

@c.record
class struct_ib_uverbs_ex_create_qp(c.Struct):
  SIZE = 64
  user_handle: Annotated[Annotated[int, ctypes.c_uint64], 0]
  pd_handle: Annotated[Annotated[int, ctypes.c_uint32], 8]
  send_cq_handle: Annotated[Annotated[int, ctypes.c_uint32], 12]
  recv_cq_handle: Annotated[Annotated[int, ctypes.c_uint32], 16]
  srq_handle: Annotated[Annotated[int, ctypes.c_uint32], 20]
  max_send_wr: Annotated[Annotated[int, ctypes.c_uint32], 24]
  max_recv_wr: Annotated[Annotated[int, ctypes.c_uint32], 28]
  max_send_sge: Annotated[Annotated[int, ctypes.c_uint32], 32]
  max_recv_sge: Annotated[Annotated[int, ctypes.c_uint32], 36]
  max_inline_data: Annotated[Annotated[int, ctypes.c_uint32], 40]
  sq_sig_all: Annotated[Annotated[int, ctypes.c_ubyte], 44]
  qp_type: Annotated[Annotated[int, ctypes.c_ubyte], 45]
  is_srq: Annotated[Annotated[int, ctypes.c_ubyte], 46]
  reserved: Annotated[Annotated[int, ctypes.c_ubyte], 47]
  comp_mask: Annotated[Annotated[int, ctypes.c_uint32], 48]
  create_flags: Annotated[Annotated[int, ctypes.c_uint32], 52]
  rwq_ind_tbl_handle: Annotated[Annotated[int, ctypes.c_uint32], 56]
  source_qpn: Annotated[Annotated[int, ctypes.c_uint32], 60]
@c.record
class struct_ib_uverbs_open_qp(c.Struct):
  SIZE = 32
  response: Annotated[Annotated[int, ctypes.c_uint64], 0]
  user_handle: Annotated[Annotated[int, ctypes.c_uint64], 8]
  pd_handle: Annotated[Annotated[int, ctypes.c_uint32], 16]
  qpn: Annotated[Annotated[int, ctypes.c_uint32], 20]
  qp_type: Annotated[Annotated[int, ctypes.c_ubyte], 24]
  reserved: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[7]], 25]
  driver_data: Annotated[c.Array[Annotated[int, ctypes.c_uint64], Literal[0]], 32]
@c.record
class struct_ib_uverbs_create_qp_resp(c.Struct):
  SIZE = 32
  qp_handle: Annotated[Annotated[int, ctypes.c_uint32], 0]
  qpn: Annotated[Annotated[int, ctypes.c_uint32], 4]
  max_send_wr: Annotated[Annotated[int, ctypes.c_uint32], 8]
  max_recv_wr: Annotated[Annotated[int, ctypes.c_uint32], 12]
  max_send_sge: Annotated[Annotated[int, ctypes.c_uint32], 16]
  max_recv_sge: Annotated[Annotated[int, ctypes.c_uint32], 20]
  max_inline_data: Annotated[Annotated[int, ctypes.c_uint32], 24]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 28]
  driver_data: Annotated[c.Array[Annotated[int, ctypes.c_uint32], Literal[0]], 32]
@c.record
class struct_ib_uverbs_ex_create_qp_resp(c.Struct):
  SIZE = 40
  base: Annotated[struct_ib_uverbs_create_qp_resp, 0]
  comp_mask: Annotated[Annotated[int, ctypes.c_uint32], 32]
  response_length: Annotated[Annotated[int, ctypes.c_uint32], 36]
@c.record
class struct_ib_uverbs_qp_dest(c.Struct):
  SIZE = 32
  dgid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  flow_label: Annotated[Annotated[int, ctypes.c_uint32], 16]
  dlid: Annotated[Annotated[int, ctypes.c_uint16], 20]
  reserved: Annotated[Annotated[int, ctypes.c_uint16], 22]
  sgid_index: Annotated[Annotated[int, ctypes.c_ubyte], 24]
  hop_limit: Annotated[Annotated[int, ctypes.c_ubyte], 25]
  traffic_class: Annotated[Annotated[int, ctypes.c_ubyte], 26]
  sl: Annotated[Annotated[int, ctypes.c_ubyte], 27]
  src_path_bits: Annotated[Annotated[int, ctypes.c_ubyte], 28]
  static_rate: Annotated[Annotated[int, ctypes.c_ubyte], 29]
  is_global: Annotated[Annotated[int, ctypes.c_ubyte], 30]
  port_num: Annotated[Annotated[int, ctypes.c_ubyte], 31]
@c.record
class struct_ib_uverbs_query_qp(c.Struct):
  SIZE = 16
  response: Annotated[Annotated[int, ctypes.c_uint64], 0]
  qp_handle: Annotated[Annotated[int, ctypes.c_uint32], 8]
  attr_mask: Annotated[Annotated[int, ctypes.c_uint32], 12]
  driver_data: Annotated[c.Array[Annotated[int, ctypes.c_uint64], Literal[0]], 16]
@c.record
class struct_ib_uverbs_query_qp_resp(c.Struct):
  SIZE = 128
  dest: Annotated[struct_ib_uverbs_qp_dest, 0]
  alt_dest: Annotated[struct_ib_uverbs_qp_dest, 32]
  max_send_wr: Annotated[Annotated[int, ctypes.c_uint32], 64]
  max_recv_wr: Annotated[Annotated[int, ctypes.c_uint32], 68]
  max_send_sge: Annotated[Annotated[int, ctypes.c_uint32], 72]
  max_recv_sge: Annotated[Annotated[int, ctypes.c_uint32], 76]
  max_inline_data: Annotated[Annotated[int, ctypes.c_uint32], 80]
  qkey: Annotated[Annotated[int, ctypes.c_uint32], 84]
  rq_psn: Annotated[Annotated[int, ctypes.c_uint32], 88]
  sq_psn: Annotated[Annotated[int, ctypes.c_uint32], 92]
  dest_qp_num: Annotated[Annotated[int, ctypes.c_uint32], 96]
  qp_access_flags: Annotated[Annotated[int, ctypes.c_uint32], 100]
  pkey_index: Annotated[Annotated[int, ctypes.c_uint16], 104]
  alt_pkey_index: Annotated[Annotated[int, ctypes.c_uint16], 106]
  qp_state: Annotated[Annotated[int, ctypes.c_ubyte], 108]
  cur_qp_state: Annotated[Annotated[int, ctypes.c_ubyte], 109]
  path_mtu: Annotated[Annotated[int, ctypes.c_ubyte], 110]
  path_mig_state: Annotated[Annotated[int, ctypes.c_ubyte], 111]
  sq_draining: Annotated[Annotated[int, ctypes.c_ubyte], 112]
  max_rd_atomic: Annotated[Annotated[int, ctypes.c_ubyte], 113]
  max_dest_rd_atomic: Annotated[Annotated[int, ctypes.c_ubyte], 114]
  min_rnr_timer: Annotated[Annotated[int, ctypes.c_ubyte], 115]
  port_num: Annotated[Annotated[int, ctypes.c_ubyte], 116]
  timeout: Annotated[Annotated[int, ctypes.c_ubyte], 117]
  retry_cnt: Annotated[Annotated[int, ctypes.c_ubyte], 118]
  rnr_retry: Annotated[Annotated[int, ctypes.c_ubyte], 119]
  alt_port_num: Annotated[Annotated[int, ctypes.c_ubyte], 120]
  alt_timeout: Annotated[Annotated[int, ctypes.c_ubyte], 121]
  sq_sig_all: Annotated[Annotated[int, ctypes.c_ubyte], 122]
  reserved: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[5]], 123]
  driver_data: Annotated[c.Array[Annotated[int, ctypes.c_uint64], Literal[0]], 128]
@c.record
class struct_ib_uverbs_modify_qp(c.Struct):
  SIZE = 112
  dest: Annotated[struct_ib_uverbs_qp_dest, 0]
  alt_dest: Annotated[struct_ib_uverbs_qp_dest, 32]
  qp_handle: Annotated[Annotated[int, ctypes.c_uint32], 64]
  attr_mask: Annotated[Annotated[int, ctypes.c_uint32], 68]
  qkey: Annotated[Annotated[int, ctypes.c_uint32], 72]
  rq_psn: Annotated[Annotated[int, ctypes.c_uint32], 76]
  sq_psn: Annotated[Annotated[int, ctypes.c_uint32], 80]
  dest_qp_num: Annotated[Annotated[int, ctypes.c_uint32], 84]
  qp_access_flags: Annotated[Annotated[int, ctypes.c_uint32], 88]
  pkey_index: Annotated[Annotated[int, ctypes.c_uint16], 92]
  alt_pkey_index: Annotated[Annotated[int, ctypes.c_uint16], 94]
  qp_state: Annotated[Annotated[int, ctypes.c_ubyte], 96]
  cur_qp_state: Annotated[Annotated[int, ctypes.c_ubyte], 97]
  path_mtu: Annotated[Annotated[int, ctypes.c_ubyte], 98]
  path_mig_state: Annotated[Annotated[int, ctypes.c_ubyte], 99]
  en_sqd_async_notify: Annotated[Annotated[int, ctypes.c_ubyte], 100]
  max_rd_atomic: Annotated[Annotated[int, ctypes.c_ubyte], 101]
  max_dest_rd_atomic: Annotated[Annotated[int, ctypes.c_ubyte], 102]
  min_rnr_timer: Annotated[Annotated[int, ctypes.c_ubyte], 103]
  port_num: Annotated[Annotated[int, ctypes.c_ubyte], 104]
  timeout: Annotated[Annotated[int, ctypes.c_ubyte], 105]
  retry_cnt: Annotated[Annotated[int, ctypes.c_ubyte], 106]
  rnr_retry: Annotated[Annotated[int, ctypes.c_ubyte], 107]
  alt_port_num: Annotated[Annotated[int, ctypes.c_ubyte], 108]
  alt_timeout: Annotated[Annotated[int, ctypes.c_ubyte], 109]
  reserved: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 110]
  driver_data: Annotated[c.Array[Annotated[int, ctypes.c_uint64], Literal[0]], 112]
@c.record
class struct_ib_uverbs_ex_modify_qp(c.Struct):
  SIZE = 120
  base: Annotated[struct_ib_uverbs_modify_qp, 0]
  rate_limit: Annotated[Annotated[int, ctypes.c_uint32], 112]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 116]
@c.record
class struct_ib_uverbs_ex_modify_qp_resp(c.Struct):
  SIZE = 8
  comp_mask: Annotated[Annotated[int, ctypes.c_uint32], 0]
  response_length: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_ib_uverbs_destroy_qp(c.Struct):
  SIZE = 16
  response: Annotated[Annotated[int, ctypes.c_uint64], 0]
  qp_handle: Annotated[Annotated[int, ctypes.c_uint32], 8]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 12]
@c.record
class struct_ib_uverbs_destroy_qp_resp(c.Struct):
  SIZE = 4
  events_reported: Annotated[Annotated[int, ctypes.c_uint32], 0]
@c.record
class struct_ib_uverbs_sge(c.Struct):
  SIZE = 16
  addr: Annotated[Annotated[int, ctypes.c_uint64], 0]
  length: Annotated[Annotated[int, ctypes.c_uint32], 8]
  lkey: Annotated[Annotated[int, ctypes.c_uint32], 12]
class enum_ib_uverbs_wr_opcode(Annotated[int, ctypes.c_uint32], c.Enum): pass
IB_UVERBS_WR_RDMA_WRITE = enum_ib_uverbs_wr_opcode.define('IB_UVERBS_WR_RDMA_WRITE', 0)
IB_UVERBS_WR_RDMA_WRITE_WITH_IMM = enum_ib_uverbs_wr_opcode.define('IB_UVERBS_WR_RDMA_WRITE_WITH_IMM', 1)
IB_UVERBS_WR_SEND = enum_ib_uverbs_wr_opcode.define('IB_UVERBS_WR_SEND', 2)
IB_UVERBS_WR_SEND_WITH_IMM = enum_ib_uverbs_wr_opcode.define('IB_UVERBS_WR_SEND_WITH_IMM', 3)
IB_UVERBS_WR_RDMA_READ = enum_ib_uverbs_wr_opcode.define('IB_UVERBS_WR_RDMA_READ', 4)
IB_UVERBS_WR_ATOMIC_CMP_AND_SWP = enum_ib_uverbs_wr_opcode.define('IB_UVERBS_WR_ATOMIC_CMP_AND_SWP', 5)
IB_UVERBS_WR_ATOMIC_FETCH_AND_ADD = enum_ib_uverbs_wr_opcode.define('IB_UVERBS_WR_ATOMIC_FETCH_AND_ADD', 6)
IB_UVERBS_WR_LOCAL_INV = enum_ib_uverbs_wr_opcode.define('IB_UVERBS_WR_LOCAL_INV', 7)
IB_UVERBS_WR_BIND_MW = enum_ib_uverbs_wr_opcode.define('IB_UVERBS_WR_BIND_MW', 8)
IB_UVERBS_WR_SEND_WITH_INV = enum_ib_uverbs_wr_opcode.define('IB_UVERBS_WR_SEND_WITH_INV', 9)
IB_UVERBS_WR_TSO = enum_ib_uverbs_wr_opcode.define('IB_UVERBS_WR_TSO', 10)
IB_UVERBS_WR_RDMA_READ_WITH_INV = enum_ib_uverbs_wr_opcode.define('IB_UVERBS_WR_RDMA_READ_WITH_INV', 11)
IB_UVERBS_WR_MASKED_ATOMIC_CMP_AND_SWP = enum_ib_uverbs_wr_opcode.define('IB_UVERBS_WR_MASKED_ATOMIC_CMP_AND_SWP', 12)
IB_UVERBS_WR_MASKED_ATOMIC_FETCH_AND_ADD = enum_ib_uverbs_wr_opcode.define('IB_UVERBS_WR_MASKED_ATOMIC_FETCH_AND_ADD', 13)
IB_UVERBS_WR_FLUSH = enum_ib_uverbs_wr_opcode.define('IB_UVERBS_WR_FLUSH', 14)
IB_UVERBS_WR_ATOMIC_WRITE = enum_ib_uverbs_wr_opcode.define('IB_UVERBS_WR_ATOMIC_WRITE', 15)

@c.record
class struct_ib_uverbs_send_wr(c.Struct):
  SIZE = 56
  wr_id: Annotated[Annotated[int, ctypes.c_uint64], 0]
  num_sge: Annotated[Annotated[int, ctypes.c_uint32], 8]
  opcode: Annotated[Annotated[int, ctypes.c_uint32], 12]
  send_flags: Annotated[Annotated[int, ctypes.c_uint32], 16]
  ex: Annotated[struct_ib_uverbs_send_wr_ex, 20]
  wr: Annotated[struct_ib_uverbs_send_wr_wr, 24]
@c.record
class struct_ib_uverbs_send_wr_ex(c.Struct):
  SIZE = 4
  imm_data: Annotated[Annotated[int, ctypes.c_uint32], 0]
  invalidate_rkey: Annotated[Annotated[int, ctypes.c_uint32], 0]
@c.record
class struct_ib_uverbs_send_wr_wr(c.Struct):
  SIZE = 32
  rdma: Annotated[struct_ib_uverbs_send_wr_wr_rdma, 0]
  atomic: Annotated[struct_ib_uverbs_send_wr_wr_atomic, 0]
  ud: Annotated[struct_ib_uverbs_send_wr_wr_ud, 0]
@c.record
class struct_ib_uverbs_send_wr_wr_rdma(c.Struct):
  SIZE = 16
  remote_addr: Annotated[Annotated[int, ctypes.c_uint64], 0]
  rkey: Annotated[Annotated[int, ctypes.c_uint32], 8]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 12]
@c.record
class struct_ib_uverbs_send_wr_wr_atomic(c.Struct):
  SIZE = 32
  remote_addr: Annotated[Annotated[int, ctypes.c_uint64], 0]
  compare_add: Annotated[Annotated[int, ctypes.c_uint64], 8]
  swap: Annotated[Annotated[int, ctypes.c_uint64], 16]
  rkey: Annotated[Annotated[int, ctypes.c_uint32], 24]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 28]
@c.record
class struct_ib_uverbs_send_wr_wr_ud(c.Struct):
  SIZE = 16
  ah: Annotated[Annotated[int, ctypes.c_uint32], 0]
  remote_qpn: Annotated[Annotated[int, ctypes.c_uint32], 4]
  remote_qkey: Annotated[Annotated[int, ctypes.c_uint32], 8]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 12]
@c.record
class struct_ib_uverbs_post_send(c.Struct):
  SIZE = 24
  response: Annotated[Annotated[int, ctypes.c_uint64], 0]
  qp_handle: Annotated[Annotated[int, ctypes.c_uint32], 8]
  wr_count: Annotated[Annotated[int, ctypes.c_uint32], 12]
  sge_count: Annotated[Annotated[int, ctypes.c_uint32], 16]
  wqe_size: Annotated[Annotated[int, ctypes.c_uint32], 20]
  send_wr: Annotated[c.Array[struct_ib_uverbs_send_wr, Literal[0]], 24]
@c.record
class struct_ib_uverbs_post_send_resp(c.Struct):
  SIZE = 4
  bad_wr: Annotated[Annotated[int, ctypes.c_uint32], 0]
@c.record
class struct_ib_uverbs_recv_wr(c.Struct):
  SIZE = 16
  wr_id: Annotated[Annotated[int, ctypes.c_uint64], 0]
  num_sge: Annotated[Annotated[int, ctypes.c_uint32], 8]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 12]
@c.record
class struct_ib_uverbs_post_recv(c.Struct):
  SIZE = 24
  response: Annotated[Annotated[int, ctypes.c_uint64], 0]
  qp_handle: Annotated[Annotated[int, ctypes.c_uint32], 8]
  wr_count: Annotated[Annotated[int, ctypes.c_uint32], 12]
  sge_count: Annotated[Annotated[int, ctypes.c_uint32], 16]
  wqe_size: Annotated[Annotated[int, ctypes.c_uint32], 20]
  recv_wr: Annotated[c.Array[struct_ib_uverbs_recv_wr, Literal[0]], 24]
@c.record
class struct_ib_uverbs_post_recv_resp(c.Struct):
  SIZE = 4
  bad_wr: Annotated[Annotated[int, ctypes.c_uint32], 0]
@c.record
class struct_ib_uverbs_post_srq_recv(c.Struct):
  SIZE = 24
  response: Annotated[Annotated[int, ctypes.c_uint64], 0]
  srq_handle: Annotated[Annotated[int, ctypes.c_uint32], 8]
  wr_count: Annotated[Annotated[int, ctypes.c_uint32], 12]
  sge_count: Annotated[Annotated[int, ctypes.c_uint32], 16]
  wqe_size: Annotated[Annotated[int, ctypes.c_uint32], 20]
  recv: Annotated[c.Array[struct_ib_uverbs_recv_wr, Literal[0]], 24]
@c.record
class struct_ib_uverbs_post_srq_recv_resp(c.Struct):
  SIZE = 4
  bad_wr: Annotated[Annotated[int, ctypes.c_uint32], 0]
@c.record
class struct_ib_uverbs_create_ah(c.Struct):
  SIZE = 56
  response: Annotated[Annotated[int, ctypes.c_uint64], 0]
  user_handle: Annotated[Annotated[int, ctypes.c_uint64], 8]
  pd_handle: Annotated[Annotated[int, ctypes.c_uint32], 16]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 20]
  attr: Annotated[struct_ib_uverbs_ah_attr, 24]
  driver_data: Annotated[c.Array[Annotated[int, ctypes.c_uint64], Literal[0]], 56]
@c.record
class struct_ib_uverbs_create_ah_resp(c.Struct):
  SIZE = 4
  ah_handle: Annotated[Annotated[int, ctypes.c_uint32], 0]
  driver_data: Annotated[c.Array[Annotated[int, ctypes.c_uint32], Literal[0]], 4]
@c.record
class struct_ib_uverbs_destroy_ah(c.Struct):
  SIZE = 4
  ah_handle: Annotated[Annotated[int, ctypes.c_uint32], 0]
@c.record
class struct_ib_uverbs_attach_mcast(c.Struct):
  SIZE = 24
  gid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  qp_handle: Annotated[Annotated[int, ctypes.c_uint32], 16]
  mlid: Annotated[Annotated[int, ctypes.c_uint16], 20]
  reserved: Annotated[Annotated[int, ctypes.c_uint16], 22]
  driver_data: Annotated[c.Array[Annotated[int, ctypes.c_uint64], Literal[0]], 24]
@c.record
class struct_ib_uverbs_detach_mcast(c.Struct):
  SIZE = 24
  gid: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  qp_handle: Annotated[Annotated[int, ctypes.c_uint32], 16]
  mlid: Annotated[Annotated[int, ctypes.c_uint16], 20]
  reserved: Annotated[Annotated[int, ctypes.c_uint16], 22]
  driver_data: Annotated[c.Array[Annotated[int, ctypes.c_uint64], Literal[0]], 24]
@c.record
class struct_ib_uverbs_flow_spec_hdr(c.Struct):
  SIZE = 8
  type: Annotated[Annotated[int, ctypes.c_uint32], 0]
  size: Annotated[Annotated[int, ctypes.c_uint16], 4]
  reserved: Annotated[Annotated[int, ctypes.c_uint16], 6]
  flow_spec_data: Annotated[c.Array[Annotated[int, ctypes.c_uint64], Literal[0]], 8]
@c.record
class struct_ib_uverbs_flow_eth_filter(c.Struct):
  SIZE = 16
  dst_mac: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[6]], 0]
  src_mac: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[6]], 6]
  ether_type: Annotated[Annotated[int, ctypes.c_uint16], 12]
  vlan_tag: Annotated[Annotated[int, ctypes.c_uint16], 14]
@c.record
class struct_ib_uverbs_flow_spec_eth(c.Struct):
  SIZE = 40
  hdr: Annotated[struct_ib_uverbs_flow_spec_hdr, 0]
  type: Annotated[Annotated[int, ctypes.c_uint32], 0]
  size: Annotated[Annotated[int, ctypes.c_uint16], 4]
  reserved: Annotated[Annotated[int, ctypes.c_uint16], 6]
  val: Annotated[struct_ib_uverbs_flow_eth_filter, 8]
  mask: Annotated[struct_ib_uverbs_flow_eth_filter, 24]
@c.record
class struct_ib_uverbs_flow_ipv4_filter(c.Struct):
  SIZE = 12
  src_ip: Annotated[Annotated[int, ctypes.c_uint32], 0]
  dst_ip: Annotated[Annotated[int, ctypes.c_uint32], 4]
  proto: Annotated[Annotated[int, ctypes.c_ubyte], 8]
  tos: Annotated[Annotated[int, ctypes.c_ubyte], 9]
  ttl: Annotated[Annotated[int, ctypes.c_ubyte], 10]
  flags: Annotated[Annotated[int, ctypes.c_ubyte], 11]
@c.record
class struct_ib_uverbs_flow_spec_ipv4(c.Struct):
  SIZE = 32
  hdr: Annotated[struct_ib_uverbs_flow_spec_hdr, 0]
  type: Annotated[Annotated[int, ctypes.c_uint32], 0]
  size: Annotated[Annotated[int, ctypes.c_uint16], 4]
  reserved: Annotated[Annotated[int, ctypes.c_uint16], 6]
  val: Annotated[struct_ib_uverbs_flow_ipv4_filter, 8]
  mask: Annotated[struct_ib_uverbs_flow_ipv4_filter, 20]
@c.record
class struct_ib_uverbs_flow_tcp_udp_filter(c.Struct):
  SIZE = 4
  dst_port: Annotated[Annotated[int, ctypes.c_uint16], 0]
  src_port: Annotated[Annotated[int, ctypes.c_uint16], 2]
@c.record
class struct_ib_uverbs_flow_spec_tcp_udp(c.Struct):
  SIZE = 16
  hdr: Annotated[struct_ib_uverbs_flow_spec_hdr, 0]
  type: Annotated[Annotated[int, ctypes.c_uint32], 0]
  size: Annotated[Annotated[int, ctypes.c_uint16], 4]
  reserved: Annotated[Annotated[int, ctypes.c_uint16], 6]
  val: Annotated[struct_ib_uverbs_flow_tcp_udp_filter, 8]
  mask: Annotated[struct_ib_uverbs_flow_tcp_udp_filter, 12]
@c.record
class struct_ib_uverbs_flow_ipv6_filter(c.Struct):
  SIZE = 40
  src_ip: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 0]
  dst_ip: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[16]], 16]
  flow_label: Annotated[Annotated[int, ctypes.c_uint32], 32]
  next_hdr: Annotated[Annotated[int, ctypes.c_ubyte], 36]
  traffic_class: Annotated[Annotated[int, ctypes.c_ubyte], 37]
  hop_limit: Annotated[Annotated[int, ctypes.c_ubyte], 38]
  reserved: Annotated[Annotated[int, ctypes.c_ubyte], 39]
@c.record
class struct_ib_uverbs_flow_spec_ipv6(c.Struct):
  SIZE = 88
  hdr: Annotated[struct_ib_uverbs_flow_spec_hdr, 0]
  type: Annotated[Annotated[int, ctypes.c_uint32], 0]
  size: Annotated[Annotated[int, ctypes.c_uint16], 4]
  reserved: Annotated[Annotated[int, ctypes.c_uint16], 6]
  val: Annotated[struct_ib_uverbs_flow_ipv6_filter, 8]
  mask: Annotated[struct_ib_uverbs_flow_ipv6_filter, 48]
@c.record
class struct_ib_uverbs_flow_spec_action_tag(c.Struct):
  SIZE = 16
  hdr: Annotated[struct_ib_uverbs_flow_spec_hdr, 0]
  type: Annotated[Annotated[int, ctypes.c_uint32], 0]
  size: Annotated[Annotated[int, ctypes.c_uint16], 4]
  reserved: Annotated[Annotated[int, ctypes.c_uint16], 6]
  tag_id: Annotated[Annotated[int, ctypes.c_uint32], 8]
  reserved1: Annotated[Annotated[int, ctypes.c_uint32], 12]
@c.record
class struct_ib_uverbs_flow_spec_action_drop(c.Struct):
  SIZE = 8
  hdr: Annotated[struct_ib_uverbs_flow_spec_hdr, 0]
  type: Annotated[Annotated[int, ctypes.c_uint32], 0]
  size: Annotated[Annotated[int, ctypes.c_uint16], 4]
  reserved: Annotated[Annotated[int, ctypes.c_uint16], 6]
@c.record
class struct_ib_uverbs_flow_spec_action_handle(c.Struct):
  SIZE = 16
  hdr: Annotated[struct_ib_uverbs_flow_spec_hdr, 0]
  type: Annotated[Annotated[int, ctypes.c_uint32], 0]
  size: Annotated[Annotated[int, ctypes.c_uint16], 4]
  reserved: Annotated[Annotated[int, ctypes.c_uint16], 6]
  handle: Annotated[Annotated[int, ctypes.c_uint32], 8]
  reserved1: Annotated[Annotated[int, ctypes.c_uint32], 12]
@c.record
class struct_ib_uverbs_flow_spec_action_count(c.Struct):
  SIZE = 16
  hdr: Annotated[struct_ib_uverbs_flow_spec_hdr, 0]
  type: Annotated[Annotated[int, ctypes.c_uint32], 0]
  size: Annotated[Annotated[int, ctypes.c_uint16], 4]
  reserved: Annotated[Annotated[int, ctypes.c_uint16], 6]
  handle: Annotated[Annotated[int, ctypes.c_uint32], 8]
  reserved1: Annotated[Annotated[int, ctypes.c_uint32], 12]
@c.record
class struct_ib_uverbs_flow_tunnel_filter(c.Struct):
  SIZE = 4
  tunnel_id: Annotated[Annotated[int, ctypes.c_uint32], 0]
@c.record
class struct_ib_uverbs_flow_spec_tunnel(c.Struct):
  SIZE = 16
  hdr: Annotated[struct_ib_uverbs_flow_spec_hdr, 0]
  type: Annotated[Annotated[int, ctypes.c_uint32], 0]
  size: Annotated[Annotated[int, ctypes.c_uint16], 4]
  reserved: Annotated[Annotated[int, ctypes.c_uint16], 6]
  val: Annotated[struct_ib_uverbs_flow_tunnel_filter, 8]
  mask: Annotated[struct_ib_uverbs_flow_tunnel_filter, 12]
@c.record
class struct_ib_uverbs_flow_spec_esp_filter(c.Struct):
  SIZE = 8
  spi: Annotated[Annotated[int, ctypes.c_uint32], 0]
  seq: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_ib_uverbs_flow_spec_esp(c.Struct):
  SIZE = 24
  hdr: Annotated[struct_ib_uverbs_flow_spec_hdr, 0]
  type: Annotated[Annotated[int, ctypes.c_uint32], 0]
  size: Annotated[Annotated[int, ctypes.c_uint16], 4]
  reserved: Annotated[Annotated[int, ctypes.c_uint16], 6]
  val: Annotated[struct_ib_uverbs_flow_spec_esp_filter, 8]
  mask: Annotated[struct_ib_uverbs_flow_spec_esp_filter, 16]
@c.record
class struct_ib_uverbs_flow_gre_filter(c.Struct):
  SIZE = 8
  c_ks_res0_ver: Annotated[Annotated[int, ctypes.c_uint16], 0]
  protocol: Annotated[Annotated[int, ctypes.c_uint16], 2]
  key: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_ib_uverbs_flow_spec_gre(c.Struct):
  SIZE = 24
  hdr: Annotated[struct_ib_uverbs_flow_spec_hdr, 0]
  type: Annotated[Annotated[int, ctypes.c_uint32], 0]
  size: Annotated[Annotated[int, ctypes.c_uint16], 4]
  reserved: Annotated[Annotated[int, ctypes.c_uint16], 6]
  val: Annotated[struct_ib_uverbs_flow_gre_filter, 8]
  mask: Annotated[struct_ib_uverbs_flow_gre_filter, 16]
@c.record
class struct_ib_uverbs_flow_mpls_filter(c.Struct):
  SIZE = 4
  label: Annotated[Annotated[int, ctypes.c_uint32], 0]
@c.record
class struct_ib_uverbs_flow_spec_mpls(c.Struct):
  SIZE = 16
  hdr: Annotated[struct_ib_uverbs_flow_spec_hdr, 0]
  type: Annotated[Annotated[int, ctypes.c_uint32], 0]
  size: Annotated[Annotated[int, ctypes.c_uint16], 4]
  reserved: Annotated[Annotated[int, ctypes.c_uint16], 6]
  val: Annotated[struct_ib_uverbs_flow_mpls_filter, 8]
  mask: Annotated[struct_ib_uverbs_flow_mpls_filter, 12]
@c.record
class struct_ib_uverbs_flow_attr(c.Struct):
  SIZE = 16
  type: Annotated[Annotated[int, ctypes.c_uint32], 0]
  size: Annotated[Annotated[int, ctypes.c_uint16], 4]
  priority: Annotated[Annotated[int, ctypes.c_uint16], 6]
  num_of_specs: Annotated[Annotated[int, ctypes.c_ubyte], 8]
  reserved: Annotated[c.Array[Annotated[int, ctypes.c_ubyte], Literal[2]], 9]
  port: Annotated[Annotated[int, ctypes.c_ubyte], 11]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 12]
  flow_specs: Annotated[c.Array[struct_ib_uverbs_flow_spec_hdr, Literal[0]], 16]
@c.record
class struct_ib_uverbs_create_flow(c.Struct):
  SIZE = 24
  comp_mask: Annotated[Annotated[int, ctypes.c_uint32], 0]
  qp_handle: Annotated[Annotated[int, ctypes.c_uint32], 4]
  flow_attr: Annotated[struct_ib_uverbs_flow_attr, 8]
@c.record
class struct_ib_uverbs_create_flow_resp(c.Struct):
  SIZE = 8
  comp_mask: Annotated[Annotated[int, ctypes.c_uint32], 0]
  flow_handle: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_ib_uverbs_destroy_flow(c.Struct):
  SIZE = 8
  comp_mask: Annotated[Annotated[int, ctypes.c_uint32], 0]
  flow_handle: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_ib_uverbs_create_srq(c.Struct):
  SIZE = 32
  response: Annotated[Annotated[int, ctypes.c_uint64], 0]
  user_handle: Annotated[Annotated[int, ctypes.c_uint64], 8]
  pd_handle: Annotated[Annotated[int, ctypes.c_uint32], 16]
  max_wr: Annotated[Annotated[int, ctypes.c_uint32], 20]
  max_sge: Annotated[Annotated[int, ctypes.c_uint32], 24]
  srq_limit: Annotated[Annotated[int, ctypes.c_uint32], 28]
  driver_data: Annotated[c.Array[Annotated[int, ctypes.c_uint64], Literal[0]], 32]
@c.record
class struct_ib_uverbs_create_xsrq(c.Struct):
  SIZE = 48
  response: Annotated[Annotated[int, ctypes.c_uint64], 0]
  user_handle: Annotated[Annotated[int, ctypes.c_uint64], 8]
  srq_type: Annotated[Annotated[int, ctypes.c_uint32], 16]
  pd_handle: Annotated[Annotated[int, ctypes.c_uint32], 20]
  max_wr: Annotated[Annotated[int, ctypes.c_uint32], 24]
  max_sge: Annotated[Annotated[int, ctypes.c_uint32], 28]
  srq_limit: Annotated[Annotated[int, ctypes.c_uint32], 32]
  max_num_tags: Annotated[Annotated[int, ctypes.c_uint32], 36]
  xrcd_handle: Annotated[Annotated[int, ctypes.c_uint32], 40]
  cq_handle: Annotated[Annotated[int, ctypes.c_uint32], 44]
  driver_data: Annotated[c.Array[Annotated[int, ctypes.c_uint64], Literal[0]], 48]
@c.record
class struct_ib_uverbs_create_srq_resp(c.Struct):
  SIZE = 16
  srq_handle: Annotated[Annotated[int, ctypes.c_uint32], 0]
  max_wr: Annotated[Annotated[int, ctypes.c_uint32], 4]
  max_sge: Annotated[Annotated[int, ctypes.c_uint32], 8]
  srqn: Annotated[Annotated[int, ctypes.c_uint32], 12]
  driver_data: Annotated[c.Array[Annotated[int, ctypes.c_uint32], Literal[0]], 16]
@c.record
class struct_ib_uverbs_modify_srq(c.Struct):
  SIZE = 16
  srq_handle: Annotated[Annotated[int, ctypes.c_uint32], 0]
  attr_mask: Annotated[Annotated[int, ctypes.c_uint32], 4]
  max_wr: Annotated[Annotated[int, ctypes.c_uint32], 8]
  srq_limit: Annotated[Annotated[int, ctypes.c_uint32], 12]
  driver_data: Annotated[c.Array[Annotated[int, ctypes.c_uint64], Literal[0]], 16]
@c.record
class struct_ib_uverbs_query_srq(c.Struct):
  SIZE = 16
  response: Annotated[Annotated[int, ctypes.c_uint64], 0]
  srq_handle: Annotated[Annotated[int, ctypes.c_uint32], 8]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 12]
  driver_data: Annotated[c.Array[Annotated[int, ctypes.c_uint64], Literal[0]], 16]
@c.record
class struct_ib_uverbs_query_srq_resp(c.Struct):
  SIZE = 16
  max_wr: Annotated[Annotated[int, ctypes.c_uint32], 0]
  max_sge: Annotated[Annotated[int, ctypes.c_uint32], 4]
  srq_limit: Annotated[Annotated[int, ctypes.c_uint32], 8]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 12]
@c.record
class struct_ib_uverbs_destroy_srq(c.Struct):
  SIZE = 16
  response: Annotated[Annotated[int, ctypes.c_uint64], 0]
  srq_handle: Annotated[Annotated[int, ctypes.c_uint32], 8]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 12]
@c.record
class struct_ib_uverbs_destroy_srq_resp(c.Struct):
  SIZE = 4
  events_reported: Annotated[Annotated[int, ctypes.c_uint32], 0]
@c.record
class struct_ib_uverbs_ex_create_wq(c.Struct):
  SIZE = 40
  comp_mask: Annotated[Annotated[int, ctypes.c_uint32], 0]
  wq_type: Annotated[Annotated[int, ctypes.c_uint32], 4]
  user_handle: Annotated[Annotated[int, ctypes.c_uint64], 8]
  pd_handle: Annotated[Annotated[int, ctypes.c_uint32], 16]
  cq_handle: Annotated[Annotated[int, ctypes.c_uint32], 20]
  max_wr: Annotated[Annotated[int, ctypes.c_uint32], 24]
  max_sge: Annotated[Annotated[int, ctypes.c_uint32], 28]
  create_flags: Annotated[Annotated[int, ctypes.c_uint32], 32]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 36]
@c.record
class struct_ib_uverbs_ex_create_wq_resp(c.Struct):
  SIZE = 24
  comp_mask: Annotated[Annotated[int, ctypes.c_uint32], 0]
  response_length: Annotated[Annotated[int, ctypes.c_uint32], 4]
  wq_handle: Annotated[Annotated[int, ctypes.c_uint32], 8]
  max_wr: Annotated[Annotated[int, ctypes.c_uint32], 12]
  max_sge: Annotated[Annotated[int, ctypes.c_uint32], 16]
  wqn: Annotated[Annotated[int, ctypes.c_uint32], 20]
@c.record
class struct_ib_uverbs_ex_destroy_wq(c.Struct):
  SIZE = 8
  comp_mask: Annotated[Annotated[int, ctypes.c_uint32], 0]
  wq_handle: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_ib_uverbs_ex_destroy_wq_resp(c.Struct):
  SIZE = 16
  comp_mask: Annotated[Annotated[int, ctypes.c_uint32], 0]
  response_length: Annotated[Annotated[int, ctypes.c_uint32], 4]
  events_reported: Annotated[Annotated[int, ctypes.c_uint32], 8]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 12]
@c.record
class struct_ib_uverbs_ex_modify_wq(c.Struct):
  SIZE = 24
  attr_mask: Annotated[Annotated[int, ctypes.c_uint32], 0]
  wq_handle: Annotated[Annotated[int, ctypes.c_uint32], 4]
  wq_state: Annotated[Annotated[int, ctypes.c_uint32], 8]
  curr_wq_state: Annotated[Annotated[int, ctypes.c_uint32], 12]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 16]
  flags_mask: Annotated[Annotated[int, ctypes.c_uint32], 20]
@c.record
class struct_ib_uverbs_ex_create_rwq_ind_table(c.Struct):
  SIZE = 8
  comp_mask: Annotated[Annotated[int, ctypes.c_uint32], 0]
  log_ind_tbl_size: Annotated[Annotated[int, ctypes.c_uint32], 4]
  wq_handles: Annotated[c.Array[Annotated[int, ctypes.c_uint32], Literal[0]], 8]
@c.record
class struct_ib_uverbs_ex_create_rwq_ind_table_resp(c.Struct):
  SIZE = 16
  comp_mask: Annotated[Annotated[int, ctypes.c_uint32], 0]
  response_length: Annotated[Annotated[int, ctypes.c_uint32], 4]
  ind_tbl_handle: Annotated[Annotated[int, ctypes.c_uint32], 8]
  ind_tbl_num: Annotated[Annotated[int, ctypes.c_uint32], 12]
@c.record
class struct_ib_uverbs_ex_destroy_rwq_ind_table(c.Struct):
  SIZE = 8
  comp_mask: Annotated[Annotated[int, ctypes.c_uint32], 0]
  ind_tbl_handle: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_ib_uverbs_cq_moderation(c.Struct):
  SIZE = 4
  cq_count: Annotated[Annotated[int, ctypes.c_uint16], 0]
  cq_period: Annotated[Annotated[int, ctypes.c_uint16], 2]
@c.record
class struct_ib_uverbs_ex_modify_cq(c.Struct):
  SIZE = 16
  cq_handle: Annotated[Annotated[int, ctypes.c_uint32], 0]
  attr_mask: Annotated[Annotated[int, ctypes.c_uint32], 4]
  attr: Annotated[struct_ib_uverbs_cq_moderation, 8]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 12]
class enum_ib_uverbs_device_cap_flags(Annotated[int, ctypes.c_uint64], c.Enum): pass
IB_UVERBS_DEVICE_RESIZE_MAX_WR = enum_ib_uverbs_device_cap_flags.define('IB_UVERBS_DEVICE_RESIZE_MAX_WR', 1)
IB_UVERBS_DEVICE_BAD_PKEY_CNTR = enum_ib_uverbs_device_cap_flags.define('IB_UVERBS_DEVICE_BAD_PKEY_CNTR', 2)
IB_UVERBS_DEVICE_BAD_QKEY_CNTR = enum_ib_uverbs_device_cap_flags.define('IB_UVERBS_DEVICE_BAD_QKEY_CNTR', 4)
IB_UVERBS_DEVICE_RAW_MULTI = enum_ib_uverbs_device_cap_flags.define('IB_UVERBS_DEVICE_RAW_MULTI', 8)
IB_UVERBS_DEVICE_AUTO_PATH_MIG = enum_ib_uverbs_device_cap_flags.define('IB_UVERBS_DEVICE_AUTO_PATH_MIG', 16)
IB_UVERBS_DEVICE_CHANGE_PHY_PORT = enum_ib_uverbs_device_cap_flags.define('IB_UVERBS_DEVICE_CHANGE_PHY_PORT', 32)
IB_UVERBS_DEVICE_UD_AV_PORT_ENFORCE = enum_ib_uverbs_device_cap_flags.define('IB_UVERBS_DEVICE_UD_AV_PORT_ENFORCE', 64)
IB_UVERBS_DEVICE_CURR_QP_STATE_MOD = enum_ib_uverbs_device_cap_flags.define('IB_UVERBS_DEVICE_CURR_QP_STATE_MOD', 128)
IB_UVERBS_DEVICE_SHUTDOWN_PORT = enum_ib_uverbs_device_cap_flags.define('IB_UVERBS_DEVICE_SHUTDOWN_PORT', 256)
IB_UVERBS_DEVICE_PORT_ACTIVE_EVENT = enum_ib_uverbs_device_cap_flags.define('IB_UVERBS_DEVICE_PORT_ACTIVE_EVENT', 1024)
IB_UVERBS_DEVICE_SYS_IMAGE_GUID = enum_ib_uverbs_device_cap_flags.define('IB_UVERBS_DEVICE_SYS_IMAGE_GUID', 2048)
IB_UVERBS_DEVICE_RC_RNR_NAK_GEN = enum_ib_uverbs_device_cap_flags.define('IB_UVERBS_DEVICE_RC_RNR_NAK_GEN', 4096)
IB_UVERBS_DEVICE_SRQ_RESIZE = enum_ib_uverbs_device_cap_flags.define('IB_UVERBS_DEVICE_SRQ_RESIZE', 8192)
IB_UVERBS_DEVICE_N_NOTIFY_CQ = enum_ib_uverbs_device_cap_flags.define('IB_UVERBS_DEVICE_N_NOTIFY_CQ', 16384)
IB_UVERBS_DEVICE_MEM_WINDOW = enum_ib_uverbs_device_cap_flags.define('IB_UVERBS_DEVICE_MEM_WINDOW', 131072)
IB_UVERBS_DEVICE_UD_IP_CSUM = enum_ib_uverbs_device_cap_flags.define('IB_UVERBS_DEVICE_UD_IP_CSUM', 262144)
IB_UVERBS_DEVICE_XRC = enum_ib_uverbs_device_cap_flags.define('IB_UVERBS_DEVICE_XRC', 1048576)
IB_UVERBS_DEVICE_MEM_MGT_EXTENSIONS = enum_ib_uverbs_device_cap_flags.define('IB_UVERBS_DEVICE_MEM_MGT_EXTENSIONS', 2097152)
IB_UVERBS_DEVICE_MEM_WINDOW_TYPE_2A = enum_ib_uverbs_device_cap_flags.define('IB_UVERBS_DEVICE_MEM_WINDOW_TYPE_2A', 8388608)
IB_UVERBS_DEVICE_MEM_WINDOW_TYPE_2B = enum_ib_uverbs_device_cap_flags.define('IB_UVERBS_DEVICE_MEM_WINDOW_TYPE_2B', 16777216)
IB_UVERBS_DEVICE_RC_IP_CSUM = enum_ib_uverbs_device_cap_flags.define('IB_UVERBS_DEVICE_RC_IP_CSUM', 33554432)
IB_UVERBS_DEVICE_RAW_IP_CSUM = enum_ib_uverbs_device_cap_flags.define('IB_UVERBS_DEVICE_RAW_IP_CSUM', 67108864)
IB_UVERBS_DEVICE_MANAGED_FLOW_STEERING = enum_ib_uverbs_device_cap_flags.define('IB_UVERBS_DEVICE_MANAGED_FLOW_STEERING', 536870912)
IB_UVERBS_DEVICE_RAW_SCATTER_FCS = enum_ib_uverbs_device_cap_flags.define('IB_UVERBS_DEVICE_RAW_SCATTER_FCS', 17179869184)
IB_UVERBS_DEVICE_PCI_WRITE_END_PADDING = enum_ib_uverbs_device_cap_flags.define('IB_UVERBS_DEVICE_PCI_WRITE_END_PADDING', 68719476736)
IB_UVERBS_DEVICE_FLUSH_GLOBAL = enum_ib_uverbs_device_cap_flags.define('IB_UVERBS_DEVICE_FLUSH_GLOBAL', 274877906944)
IB_UVERBS_DEVICE_FLUSH_PERSISTENT = enum_ib_uverbs_device_cap_flags.define('IB_UVERBS_DEVICE_FLUSH_PERSISTENT', 549755813888)
IB_UVERBS_DEVICE_ATOMIC_WRITE = enum_ib_uverbs_device_cap_flags.define('IB_UVERBS_DEVICE_ATOMIC_WRITE', 1099511627776)

class enum_ib_uverbs_raw_packet_caps(Annotated[int, ctypes.c_uint32], c.Enum): pass
IB_UVERBS_RAW_PACKET_CAP_CVLAN_STRIPPING = enum_ib_uverbs_raw_packet_caps.define('IB_UVERBS_RAW_PACKET_CAP_CVLAN_STRIPPING', 1)
IB_UVERBS_RAW_PACKET_CAP_SCATTER_FCS = enum_ib_uverbs_raw_packet_caps.define('IB_UVERBS_RAW_PACKET_CAP_SCATTER_FCS', 2)
IB_UVERBS_RAW_PACKET_CAP_IP_CSUM = enum_ib_uverbs_raw_packet_caps.define('IB_UVERBS_RAW_PACKET_CAP_IP_CSUM', 4)
IB_UVERBS_RAW_PACKET_CAP_DELAY_DROP = enum_ib_uverbs_raw_packet_caps.define('IB_UVERBS_RAW_PACKET_CAP_DELAY_DROP', 8)

c.init_records()
vext_field_avail = lambda type,fld,sz: (offsetof(type, fld) < (sz)) # type: ignore
IBV_DEVICE_RAW_SCATTER_FCS = (1 << 34) # type: ignore
IBV_DEVICE_PCI_WRITE_END_PADDING = (1 << 36) # type: ignore
ibv_query_port = lambda context,port_num,port_attr: ___ibv_query_port(context, port_num, port_attr) # type: ignore
ibv_reg_mr = lambda pd,addr,length,access: __ibv_reg_mr(pd, addr, length, access, __builtin_constant_p( ((int)(access) & IBV_ACCESS_OPTIONAL_RANGE) == 0)) # type: ignore
ibv_reg_mr_iova = lambda pd,addr,length,iova,access: __ibv_reg_mr_iova(pd, addr, length, iova, access, __builtin_constant_p( ((access) & IBV_ACCESS_OPTIONAL_RANGE) == 0)) # type: ignore
ETHERNET_LL_SIZE = 6 # type: ignore
IB_ROCE_UDP_ENCAP_VALID_PORT_MIN = (0xC000) # type: ignore
IB_ROCE_UDP_ENCAP_VALID_PORT_MAX = (0xFFFF) # type: ignore
IB_GRH_FLOWLABEL_MASK = (0x000FFFFF) # type: ignore
IBV_FLOW_ACTION_ESP_KEYMAT_AES_GCM = IB_UVERBS_FLOW_ACTION_ESP_KEYMAT_AES_GCM # type: ignore
IBV_FLOW_ACTION_IV_ALGO_SEQ = IB_UVERBS_FLOW_ACTION_IV_ALGO_SEQ # type: ignore
IBV_FLOW_ACTION_ESP_REPLAY_NONE = IB_UVERBS_FLOW_ACTION_ESP_REPLAY_NONE # type: ignore
IBV_FLOW_ACTION_ESP_REPLAY_BMP = IB_UVERBS_FLOW_ACTION_ESP_REPLAY_BMP # type: ignore
IBV_FLOW_ACTION_ESP_FLAGS_INLINE_CRYPTO = IB_UVERBS_FLOW_ACTION_ESP_FLAGS_INLINE_CRYPTO # type: ignore
IBV_FLOW_ACTION_ESP_FLAGS_FULL_OFFLOAD = IB_UVERBS_FLOW_ACTION_ESP_FLAGS_FULL_OFFLOAD # type: ignore
IBV_FLOW_ACTION_ESP_FLAGS_TUNNEL = IB_UVERBS_FLOW_ACTION_ESP_FLAGS_TUNNEL # type: ignore
IBV_FLOW_ACTION_ESP_FLAGS_TRANSPORT = IB_UVERBS_FLOW_ACTION_ESP_FLAGS_TRANSPORT # type: ignore
IBV_FLOW_ACTION_ESP_FLAGS_DECRYPT = IB_UVERBS_FLOW_ACTION_ESP_FLAGS_DECRYPT # type: ignore
IBV_FLOW_ACTION_ESP_FLAGS_ENCRYPT = IB_UVERBS_FLOW_ACTION_ESP_FLAGS_ENCRYPT # type: ignore
IBV_FLOW_ACTION_ESP_FLAGS_ESN_NEW_WINDOW = IB_UVERBS_FLOW_ACTION_ESP_FLAGS_ESN_NEW_WINDOW # type: ignore
IBV_ADVISE_MR_ADVICE_PREFETCH = IB_UVERBS_ADVISE_MR_ADVICE_PREFETCH # type: ignore
IBV_ADVISE_MR_ADVICE_PREFETCH_WRITE = IB_UVERBS_ADVISE_MR_ADVICE_PREFETCH_WRITE # type: ignore
IBV_ADVISE_MR_ADVICE_PREFETCH_NO_FAULT = IB_UVERBS_ADVISE_MR_ADVICE_PREFETCH_NO_FAULT # type: ignore
IBV_ADVISE_MR_FLAG_FLUSH = IB_UVERBS_ADVISE_MR_FLAG_FLUSH # type: ignore
IBV_QPF_GRH_REQUIRED = IB_UVERBS_QPF_GRH_REQUIRED # type: ignore
IBV_ACCESS_OPTIONAL_RANGE = IB_UVERBS_ACCESS_OPTIONAL_RANGE # type: ignore
IB_UVERBS_ACCESS_OPTIONAL_FIRST = (1 << 20) # type: ignore
IB_UVERBS_ACCESS_OPTIONAL_LAST = (1 << 29) # type: ignore
IB_USER_VERBS_ABI_VERSION = 6 # type: ignore
IB_USER_VERBS_CMD_THRESHOLD = 50 # type: ignore
IB_USER_VERBS_CMD_COMMAND_MASK = 0xff # type: ignore
IB_USER_VERBS_CMD_FLAG_EXTENDED = 0x80000000 # type: ignore
IB_USER_VERBS_MAX_LOG_IND_TBL_SIZE = 0x0d # type: ignore
IB_DEVICE_NAME_MAX = 64 # type: ignore