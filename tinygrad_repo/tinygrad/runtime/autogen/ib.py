# mypy: disable-error-code="empty-body"
from __future__ import annotations
import ctypes
from typing import Literal, TypeAlias
from tinygrad.runtime.support.c import _IO, _IOW, _IOR, _IOWR
from tinygrad.runtime.support import c
dll = c.DLL('ib', 'ibverbs', use_errno=True)
@c.record
class union_ibv_gid(c.Struct):
  SIZE = 16
  raw: c.Array[ctypes.c_ubyte, Literal[16]]
  _global: union_ibv_gid_global
uint8_t: TypeAlias = ctypes.c_ubyte
@c.record
class union_ibv_gid_global(c.Struct):
  SIZE = 16
  subnet_prefix: int
  interface_id: int
__be64: TypeAlias = ctypes.c_uint64
union_ibv_gid_global.register_fields([('subnet_prefix', ctypes.c_uint64, 0), ('interface_id', ctypes.c_uint64, 8)])
union_ibv_gid.register_fields([('raw', c.Array[uint8_t, Literal[16]], 0), ('_global', union_ibv_gid_global, 0)])
enum_ibv_gid_type: dict[int, str] = {(IBV_GID_TYPE_IB:=0): 'IBV_GID_TYPE_IB', (IBV_GID_TYPE_ROCE_V1:=1): 'IBV_GID_TYPE_ROCE_V1', (IBV_GID_TYPE_ROCE_V2:=2): 'IBV_GID_TYPE_ROCE_V2'}
@c.record
class struct_ibv_gid_entry(c.Struct):
  SIZE = 32
  gid: union_ibv_gid
  gid_index: int
  port_num: int
  gid_type: int
  ndev_ifindex: int
uint32_t: TypeAlias = ctypes.c_uint32
struct_ibv_gid_entry.register_fields([('gid', union_ibv_gid, 0), ('gid_index', uint32_t, 16), ('port_num', uint32_t, 20), ('gid_type', uint32_t, 24), ('ndev_ifindex', uint32_t, 28)])
enum_ibv_node_type: dict[int, str] = {(IBV_NODE_UNKNOWN:=-1): 'IBV_NODE_UNKNOWN', (IBV_NODE_CA:=1): 'IBV_NODE_CA', (IBV_NODE_SWITCH:=2): 'IBV_NODE_SWITCH', (IBV_NODE_ROUTER:=3): 'IBV_NODE_ROUTER', (IBV_NODE_RNIC:=4): 'IBV_NODE_RNIC', (IBV_NODE_USNIC:=5): 'IBV_NODE_USNIC', (IBV_NODE_USNIC_UDP:=6): 'IBV_NODE_USNIC_UDP', (IBV_NODE_UNSPECIFIED:=7): 'IBV_NODE_UNSPECIFIED'}
enum_ibv_transport_type: dict[int, str] = {(IBV_TRANSPORT_UNKNOWN:=-1): 'IBV_TRANSPORT_UNKNOWN', (IBV_TRANSPORT_IB:=0): 'IBV_TRANSPORT_IB', (IBV_TRANSPORT_IWARP:=1): 'IBV_TRANSPORT_IWARP', (IBV_TRANSPORT_USNIC:=2): 'IBV_TRANSPORT_USNIC', (IBV_TRANSPORT_USNIC_UDP:=3): 'IBV_TRANSPORT_USNIC_UDP', (IBV_TRANSPORT_UNSPECIFIED:=4): 'IBV_TRANSPORT_UNSPECIFIED'}
enum_ibv_device_cap_flags: dict[int, str] = {(IBV_DEVICE_RESIZE_MAX_WR:=1): 'IBV_DEVICE_RESIZE_MAX_WR', (IBV_DEVICE_BAD_PKEY_CNTR:=2): 'IBV_DEVICE_BAD_PKEY_CNTR', (IBV_DEVICE_BAD_QKEY_CNTR:=4): 'IBV_DEVICE_BAD_QKEY_CNTR', (IBV_DEVICE_RAW_MULTI:=8): 'IBV_DEVICE_RAW_MULTI', (IBV_DEVICE_AUTO_PATH_MIG:=16): 'IBV_DEVICE_AUTO_PATH_MIG', (IBV_DEVICE_CHANGE_PHY_PORT:=32): 'IBV_DEVICE_CHANGE_PHY_PORT', (IBV_DEVICE_UD_AV_PORT_ENFORCE:=64): 'IBV_DEVICE_UD_AV_PORT_ENFORCE', (IBV_DEVICE_CURR_QP_STATE_MOD:=128): 'IBV_DEVICE_CURR_QP_STATE_MOD', (IBV_DEVICE_SHUTDOWN_PORT:=256): 'IBV_DEVICE_SHUTDOWN_PORT', (IBV_DEVICE_INIT_TYPE:=512): 'IBV_DEVICE_INIT_TYPE', (IBV_DEVICE_PORT_ACTIVE_EVENT:=1024): 'IBV_DEVICE_PORT_ACTIVE_EVENT', (IBV_DEVICE_SYS_IMAGE_GUID:=2048): 'IBV_DEVICE_SYS_IMAGE_GUID', (IBV_DEVICE_RC_RNR_NAK_GEN:=4096): 'IBV_DEVICE_RC_RNR_NAK_GEN', (IBV_DEVICE_SRQ_RESIZE:=8192): 'IBV_DEVICE_SRQ_RESIZE', (IBV_DEVICE_N_NOTIFY_CQ:=16384): 'IBV_DEVICE_N_NOTIFY_CQ', (IBV_DEVICE_MEM_WINDOW:=131072): 'IBV_DEVICE_MEM_WINDOW', (IBV_DEVICE_UD_IP_CSUM:=262144): 'IBV_DEVICE_UD_IP_CSUM', (IBV_DEVICE_XRC:=1048576): 'IBV_DEVICE_XRC', (IBV_DEVICE_MEM_MGT_EXTENSIONS:=2097152): 'IBV_DEVICE_MEM_MGT_EXTENSIONS', (IBV_DEVICE_MEM_WINDOW_TYPE_2A:=8388608): 'IBV_DEVICE_MEM_WINDOW_TYPE_2A', (IBV_DEVICE_MEM_WINDOW_TYPE_2B:=16777216): 'IBV_DEVICE_MEM_WINDOW_TYPE_2B', (IBV_DEVICE_RC_IP_CSUM:=33554432): 'IBV_DEVICE_RC_IP_CSUM', (IBV_DEVICE_RAW_IP_CSUM:=67108864): 'IBV_DEVICE_RAW_IP_CSUM', (IBV_DEVICE_MANAGED_FLOW_STEERING:=536870912): 'IBV_DEVICE_MANAGED_FLOW_STEERING'}
enum_ibv_fork_status: dict[int, str] = {(IBV_FORK_DISABLED:=0): 'IBV_FORK_DISABLED', (IBV_FORK_ENABLED:=1): 'IBV_FORK_ENABLED', (IBV_FORK_UNNEEDED:=2): 'IBV_FORK_UNNEEDED'}
enum_ibv_atomic_cap: dict[int, str] = {(IBV_ATOMIC_NONE:=0): 'IBV_ATOMIC_NONE', (IBV_ATOMIC_HCA:=1): 'IBV_ATOMIC_HCA', (IBV_ATOMIC_GLOB:=2): 'IBV_ATOMIC_GLOB'}
@c.record
class struct_ibv_alloc_dm_attr(c.Struct):
  SIZE = 16
  length: int
  log_align_req: int
  comp_mask: int
size_t: TypeAlias = ctypes.c_uint64
struct_ibv_alloc_dm_attr.register_fields([('length', size_t, 0), ('log_align_req', uint32_t, 8), ('comp_mask', uint32_t, 12)])
enum_ibv_dm_mask: dict[int, str] = {(IBV_DM_MASK_HANDLE:=1): 'IBV_DM_MASK_HANDLE'}
@c.record
class struct_ibv_dm(c.Struct):
  SIZE = 32
  context: c.POINTER[struct_ibv_context]
  memcpy_to_dm: c.CFUNCTYPE[ctypes.c_int32, [c.POINTER[struct_ibv_dm], ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64]]
  memcpy_from_dm: c.CFUNCTYPE[ctypes.c_int32, [ctypes.c_void_p, c.POINTER[struct_ibv_dm], ctypes.c_uint64, ctypes.c_uint64]]
  comp_mask: int
  handle: int
@c.record
class struct_ibv_context(c.Struct):
  SIZE = 328
  device: c.POINTER[struct_ibv_device]
  ops: struct_ibv_context_ops
  cmd_fd: int
  async_fd: int
  num_comp_vectors: int
  mutex: pthread_mutex_t
  abi_compat: ctypes.c_void_p
@c.record
class struct_ibv_device(c.Struct):
  SIZE = 664
  _ops: struct__ibv_device_ops
  node_type: int
  transport_type: int
  name: c.Array[ctypes.c_char, Literal[64]]
  dev_name: c.Array[ctypes.c_char, Literal[64]]
  dev_path: c.Array[ctypes.c_char, Literal[256]]
  ibdev_path: c.Array[ctypes.c_char, Literal[256]]
@c.record
class struct__ibv_device_ops(c.Struct):
  SIZE = 16
  _dummy1: c.CFUNCTYPE[c.POINTER[struct_ibv_context], [c.POINTER[struct_ibv_device], ctypes.c_int32]]
  _dummy2: c.CFUNCTYPE[None, [c.POINTER[struct_ibv_context]]]
struct__ibv_device_ops.register_fields([('_dummy1', c.CFUNCTYPE[c.POINTER[struct_ibv_context], [c.POINTER[struct_ibv_device], ctypes.c_int32]], 0), ('_dummy2', c.CFUNCTYPE[None, [c.POINTER[struct_ibv_context]]], 8)])
struct_ibv_device.register_fields([('_ops', struct__ibv_device_ops, 0), ('node_type', ctypes.c_int32, 16), ('transport_type', ctypes.c_int32, 20), ('name', c.Array[ctypes.c_char, Literal[64]], 24), ('dev_name', c.Array[ctypes.c_char, Literal[64]], 88), ('dev_path', c.Array[ctypes.c_char, Literal[256]], 152), ('ibdev_path', c.Array[ctypes.c_char, Literal[256]], 408)])
@c.record
class struct_ibv_context_ops(c.Struct):
  SIZE = 256
  _compat_query_device: c.CFUNCTYPE[ctypes.c_int32, [c.POINTER[struct_ibv_context], c.POINTER[struct_ibv_device_attr]]]
  _compat_query_port: c.CFUNCTYPE[ctypes.c_int32, [c.POINTER[struct_ibv_context], ctypes.c_ubyte, c.POINTER[struct__compat_ibv_port_attr]]]
  _compat_alloc_pd: c.CFUNCTYPE[ctypes.c_void_p, []]
  _compat_dealloc_pd: c.CFUNCTYPE[ctypes.c_void_p, []]
  _compat_reg_mr: c.CFUNCTYPE[ctypes.c_void_p, []]
  _compat_rereg_mr: c.CFUNCTYPE[ctypes.c_void_p, []]
  _compat_dereg_mr: c.CFUNCTYPE[ctypes.c_void_p, []]
  alloc_mw: c.CFUNCTYPE[c.POINTER[struct_ibv_mw], [c.POINTER[struct_ibv_pd], ctypes.c_uint32]]
  bind_mw: c.CFUNCTYPE[ctypes.c_int32, [c.POINTER[struct_ibv_qp], c.POINTER[struct_ibv_mw], c.POINTER[struct_ibv_mw_bind]]]
  dealloc_mw: c.CFUNCTYPE[ctypes.c_int32, [c.POINTER[struct_ibv_mw]]]
  _compat_create_cq: c.CFUNCTYPE[ctypes.c_void_p, []]
  poll_cq: c.CFUNCTYPE[ctypes.c_int32, [c.POINTER[struct_ibv_cq], ctypes.c_int32, c.POINTER[struct_ibv_wc]]]
  req_notify_cq: c.CFUNCTYPE[ctypes.c_int32, [c.POINTER[struct_ibv_cq], ctypes.c_int32]]
  _compat_cq_event: c.CFUNCTYPE[ctypes.c_void_p, []]
  _compat_resize_cq: c.CFUNCTYPE[ctypes.c_void_p, []]
  _compat_destroy_cq: c.CFUNCTYPE[ctypes.c_void_p, []]
  _compat_create_srq: c.CFUNCTYPE[ctypes.c_void_p, []]
  _compat_modify_srq: c.CFUNCTYPE[ctypes.c_void_p, []]
  _compat_query_srq: c.CFUNCTYPE[ctypes.c_void_p, []]
  _compat_destroy_srq: c.CFUNCTYPE[ctypes.c_void_p, []]
  post_srq_recv: c.CFUNCTYPE[ctypes.c_int32, [c.POINTER[struct_ibv_srq], c.POINTER[struct_ibv_recv_wr], c.POINTER[c.POINTER[struct_ibv_recv_wr]]]]
  _compat_create_qp: c.CFUNCTYPE[ctypes.c_void_p, []]
  _compat_query_qp: c.CFUNCTYPE[ctypes.c_void_p, []]
  _compat_modify_qp: c.CFUNCTYPE[ctypes.c_void_p, []]
  _compat_destroy_qp: c.CFUNCTYPE[ctypes.c_void_p, []]
  post_send: c.CFUNCTYPE[ctypes.c_int32, [c.POINTER[struct_ibv_qp], c.POINTER[struct_ibv_send_wr], c.POINTER[c.POINTER[struct_ibv_send_wr]]]]
  post_recv: c.CFUNCTYPE[ctypes.c_int32, [c.POINTER[struct_ibv_qp], c.POINTER[struct_ibv_recv_wr], c.POINTER[c.POINTER[struct_ibv_recv_wr]]]]
  _compat_create_ah: c.CFUNCTYPE[ctypes.c_void_p, []]
  _compat_destroy_ah: c.CFUNCTYPE[ctypes.c_void_p, []]
  _compat_attach_mcast: c.CFUNCTYPE[ctypes.c_void_p, []]
  _compat_detach_mcast: c.CFUNCTYPE[ctypes.c_void_p, []]
  _compat_async_event: c.CFUNCTYPE[ctypes.c_void_p, []]
@c.record
class struct_ibv_device_attr(c.Struct):
  SIZE = 232
  fw_ver: c.Array[ctypes.c_char, Literal[64]]
  node_guid: int
  sys_image_guid: int
  max_mr_size: int
  page_size_cap: int
  vendor_id: int
  vendor_part_id: int
  hw_ver: int
  max_qp: int
  max_qp_wr: int
  device_cap_flags: int
  max_sge: int
  max_sge_rd: int
  max_cq: int
  max_cqe: int
  max_mr: int
  max_pd: int
  max_qp_rd_atom: int
  max_ee_rd_atom: int
  max_res_rd_atom: int
  max_qp_init_rd_atom: int
  max_ee_init_rd_atom: int
  atomic_cap: int
  max_ee: int
  max_rdd: int
  max_mw: int
  max_raw_ipv6_qp: int
  max_raw_ethy_qp: int
  max_mcast_grp: int
  max_mcast_qp_attach: int
  max_total_mcast_qp_attach: int
  max_ah: int
  max_fmr: int
  max_map_per_fmr: int
  max_srq: int
  max_srq_wr: int
  max_srq_sge: int
  max_pkeys: int
  local_ca_ack_delay: int
  phys_port_cnt: int
uint64_t: TypeAlias = ctypes.c_uint64
uint16_t: TypeAlias = ctypes.c_uint16
struct_ibv_device_attr.register_fields([('fw_ver', c.Array[ctypes.c_char, Literal[64]], 0), ('node_guid', ctypes.c_uint64, 64), ('sys_image_guid', ctypes.c_uint64, 72), ('max_mr_size', uint64_t, 80), ('page_size_cap', uint64_t, 88), ('vendor_id', uint32_t, 96), ('vendor_part_id', uint32_t, 100), ('hw_ver', uint32_t, 104), ('max_qp', ctypes.c_int32, 108), ('max_qp_wr', ctypes.c_int32, 112), ('device_cap_flags', ctypes.c_uint32, 116), ('max_sge', ctypes.c_int32, 120), ('max_sge_rd', ctypes.c_int32, 124), ('max_cq', ctypes.c_int32, 128), ('max_cqe', ctypes.c_int32, 132), ('max_mr', ctypes.c_int32, 136), ('max_pd', ctypes.c_int32, 140), ('max_qp_rd_atom', ctypes.c_int32, 144), ('max_ee_rd_atom', ctypes.c_int32, 148), ('max_res_rd_atom', ctypes.c_int32, 152), ('max_qp_init_rd_atom', ctypes.c_int32, 156), ('max_ee_init_rd_atom', ctypes.c_int32, 160), ('atomic_cap', ctypes.c_uint32, 164), ('max_ee', ctypes.c_int32, 168), ('max_rdd', ctypes.c_int32, 172), ('max_mw', ctypes.c_int32, 176), ('max_raw_ipv6_qp', ctypes.c_int32, 180), ('max_raw_ethy_qp', ctypes.c_int32, 184), ('max_mcast_grp', ctypes.c_int32, 188), ('max_mcast_qp_attach', ctypes.c_int32, 192), ('max_total_mcast_qp_attach', ctypes.c_int32, 196), ('max_ah', ctypes.c_int32, 200), ('max_fmr', ctypes.c_int32, 204), ('max_map_per_fmr', ctypes.c_int32, 208), ('max_srq', ctypes.c_int32, 212), ('max_srq_wr', ctypes.c_int32, 216), ('max_srq_sge', ctypes.c_int32, 220), ('max_pkeys', uint16_t, 224), ('local_ca_ack_delay', uint8_t, 226), ('phys_port_cnt', uint8_t, 227)])
class struct__compat_ibv_port_attr(c.Struct): pass
@c.record
class struct_ibv_mw(c.Struct):
  SIZE = 32
  context: c.POINTER[struct_ibv_context]
  pd: c.POINTER[struct_ibv_pd]
  rkey: int
  handle: int
  type: int
@c.record
class struct_ibv_pd(c.Struct):
  SIZE = 16
  context: c.POINTER[struct_ibv_context]
  handle: int
struct_ibv_pd.register_fields([('context', c.POINTER[struct_ibv_context], 0), ('handle', uint32_t, 8)])
enum_ibv_mw_type: dict[int, str] = {(IBV_MW_TYPE_1:=1): 'IBV_MW_TYPE_1', (IBV_MW_TYPE_2:=2): 'IBV_MW_TYPE_2'}
struct_ibv_mw.register_fields([('context', c.POINTER[struct_ibv_context], 0), ('pd', c.POINTER[struct_ibv_pd], 8), ('rkey', uint32_t, 16), ('handle', uint32_t, 20), ('type', ctypes.c_uint32, 24)])
@c.record
class struct_ibv_qp(c.Struct):
  SIZE = 160
  context: c.POINTER[struct_ibv_context]
  qp_context: ctypes.c_void_p
  pd: c.POINTER[struct_ibv_pd]
  send_cq: c.POINTER[struct_ibv_cq]
  recv_cq: c.POINTER[struct_ibv_cq]
  srq: c.POINTER[struct_ibv_srq]
  handle: int
  qp_num: int
  state: int
  qp_type: int
  mutex: pthread_mutex_t
  cond: pthread_cond_t
  events_completed: int
@c.record
class struct_ibv_cq(c.Struct):
  SIZE = 128
  context: c.POINTER[struct_ibv_context]
  channel: c.POINTER[struct_ibv_comp_channel]
  cq_context: ctypes.c_void_p
  handle: int
  cqe: int
  mutex: pthread_mutex_t
  cond: pthread_cond_t
  comp_events_completed: int
  async_events_completed: int
@c.record
class struct_ibv_comp_channel(c.Struct):
  SIZE = 16
  context: c.POINTER[struct_ibv_context]
  fd: int
  refcnt: int
struct_ibv_comp_channel.register_fields([('context', c.POINTER[struct_ibv_context], 0), ('fd', ctypes.c_int32, 8), ('refcnt', ctypes.c_int32, 12)])
@c.record
class pthread_mutex_t(c.Struct):
  SIZE = 40
  __data: struct___pthread_mutex_s
  __size: c.Array[ctypes.c_char, Literal[40]]
  __align: int
@c.record
class struct___pthread_mutex_s(c.Struct):
  SIZE = 40
  __lock: int
  __count: int
  __owner: int
  __nusers: int
  __kind: int
  __spins: int
  __elision: int
  __list: struct___pthread_internal_list
@c.record
class struct___pthread_internal_list(c.Struct):
  SIZE = 16
  __prev: c.POINTER[struct___pthread_internal_list]
  __next: c.POINTER[struct___pthread_internal_list]
__pthread_list_t: TypeAlias = struct___pthread_internal_list
struct___pthread_internal_list.register_fields([('__prev', c.POINTER[struct___pthread_internal_list], 0), ('__next', c.POINTER[struct___pthread_internal_list], 8)])
struct___pthread_mutex_s.register_fields([('__lock', ctypes.c_int32, 0), ('__count', ctypes.c_uint32, 4), ('__owner', ctypes.c_int32, 8), ('__nusers', ctypes.c_uint32, 12), ('__kind', ctypes.c_int32, 16), ('__spins', ctypes.c_int16, 20), ('__elision', ctypes.c_int16, 22), ('__list', struct___pthread_internal_list, 24)])
pthread_mutex_t.register_fields([('__data', struct___pthread_mutex_s, 0), ('__size', c.Array[ctypes.c_char, Literal[40]], 0), ('__align', ctypes.c_int64, 0)])
@c.record
class pthread_cond_t(c.Struct):
  SIZE = 48
  __data: struct___pthread_cond_s
  __size: c.Array[ctypes.c_char, Literal[48]]
  __align: int
@c.record
class struct___pthread_cond_s(c.Struct):
  SIZE = 48
  __wseq: __atomic_wide_counter
  __g1_start: __atomic_wide_counter
  __g_refs: c.Array[ctypes.c_uint32, Literal[2]]
  __g_size: c.Array[ctypes.c_uint32, Literal[2]]
  __g1_orig_size: int
  __wrefs: int
  __g_signals: c.Array[ctypes.c_uint32, Literal[2]]
@c.record
class __atomic_wide_counter(c.Struct):
  SIZE = 8
  __value64: int
  __value32: __atomic_wide_counter___value32
@c.record
class __atomic_wide_counter___value32(c.Struct):
  SIZE = 8
  __low: int
  __high: int
__atomic_wide_counter___value32.register_fields([('__low', ctypes.c_uint32, 0), ('__high', ctypes.c_uint32, 4)])
__atomic_wide_counter.register_fields([('__value64', ctypes.c_uint64, 0), ('__value32', __atomic_wide_counter___value32, 0)])
struct___pthread_cond_s.register_fields([('__wseq', __atomic_wide_counter, 0), ('__g1_start', __atomic_wide_counter, 8), ('__g_refs', c.Array[ctypes.c_uint32, Literal[2]], 16), ('__g_size', c.Array[ctypes.c_uint32, Literal[2]], 24), ('__g1_orig_size', ctypes.c_uint32, 32), ('__wrefs', ctypes.c_uint32, 36), ('__g_signals', c.Array[ctypes.c_uint32, Literal[2]], 40)])
pthread_cond_t.register_fields([('__data', struct___pthread_cond_s, 0), ('__size', c.Array[ctypes.c_char, Literal[48]], 0), ('__align', ctypes.c_int64, 0)])
struct_ibv_cq.register_fields([('context', c.POINTER[struct_ibv_context], 0), ('channel', c.POINTER[struct_ibv_comp_channel], 8), ('cq_context', ctypes.c_void_p, 16), ('handle', uint32_t, 24), ('cqe', ctypes.c_int32, 28), ('mutex', pthread_mutex_t, 32), ('cond', pthread_cond_t, 72), ('comp_events_completed', uint32_t, 120), ('async_events_completed', uint32_t, 124)])
@c.record
class struct_ibv_srq(c.Struct):
  SIZE = 128
  context: c.POINTER[struct_ibv_context]
  srq_context: ctypes.c_void_p
  pd: c.POINTER[struct_ibv_pd]
  handle: int
  mutex: pthread_mutex_t
  cond: pthread_cond_t
  events_completed: int
struct_ibv_srq.register_fields([('context', c.POINTER[struct_ibv_context], 0), ('srq_context', ctypes.c_void_p, 8), ('pd', c.POINTER[struct_ibv_pd], 16), ('handle', uint32_t, 24), ('mutex', pthread_mutex_t, 32), ('cond', pthread_cond_t, 72), ('events_completed', uint32_t, 120)])
enum_ibv_qp_state: dict[int, str] = {(IBV_QPS_RESET:=0): 'IBV_QPS_RESET', (IBV_QPS_INIT:=1): 'IBV_QPS_INIT', (IBV_QPS_RTR:=2): 'IBV_QPS_RTR', (IBV_QPS_RTS:=3): 'IBV_QPS_RTS', (IBV_QPS_SQD:=4): 'IBV_QPS_SQD', (IBV_QPS_SQE:=5): 'IBV_QPS_SQE', (IBV_QPS_ERR:=6): 'IBV_QPS_ERR', (IBV_QPS_UNKNOWN:=7): 'IBV_QPS_UNKNOWN'}
enum_ibv_qp_type: dict[int, str] = {(IBV_QPT_RC:=2): 'IBV_QPT_RC', (IBV_QPT_UC:=3): 'IBV_QPT_UC', (IBV_QPT_UD:=4): 'IBV_QPT_UD', (IBV_QPT_RAW_PACKET:=8): 'IBV_QPT_RAW_PACKET', (IBV_QPT_XRC_SEND:=9): 'IBV_QPT_XRC_SEND', (IBV_QPT_XRC_RECV:=10): 'IBV_QPT_XRC_RECV', (IBV_QPT_DRIVER:=255): 'IBV_QPT_DRIVER'}
struct_ibv_qp.register_fields([('context', c.POINTER[struct_ibv_context], 0), ('qp_context', ctypes.c_void_p, 8), ('pd', c.POINTER[struct_ibv_pd], 16), ('send_cq', c.POINTER[struct_ibv_cq], 24), ('recv_cq', c.POINTER[struct_ibv_cq], 32), ('srq', c.POINTER[struct_ibv_srq], 40), ('handle', uint32_t, 48), ('qp_num', uint32_t, 52), ('state', ctypes.c_uint32, 56), ('qp_type', ctypes.c_uint32, 60), ('mutex', pthread_mutex_t, 64), ('cond', pthread_cond_t, 104), ('events_completed', uint32_t, 152)])
@c.record
class struct_ibv_mw_bind(c.Struct):
  SIZE = 48
  wr_id: int
  send_flags: int
  bind_info: struct_ibv_mw_bind_info
@c.record
class struct_ibv_mw_bind_info(c.Struct):
  SIZE = 32
  mr: c.POINTER[struct_ibv_mr]
  addr: int
  length: int
  mw_access_flags: int
@c.record
class struct_ibv_mr(c.Struct):
  SIZE = 48
  context: c.POINTER[struct_ibv_context]
  pd: c.POINTER[struct_ibv_pd]
  addr: ctypes.c_void_p
  length: int
  handle: int
  lkey: int
  rkey: int
struct_ibv_mr.register_fields([('context', c.POINTER[struct_ibv_context], 0), ('pd', c.POINTER[struct_ibv_pd], 8), ('addr', ctypes.c_void_p, 16), ('length', size_t, 24), ('handle', uint32_t, 32), ('lkey', uint32_t, 36), ('rkey', uint32_t, 40)])
struct_ibv_mw_bind_info.register_fields([('mr', c.POINTER[struct_ibv_mr], 0), ('addr', uint64_t, 8), ('length', uint64_t, 16), ('mw_access_flags', ctypes.c_uint32, 24)])
struct_ibv_mw_bind.register_fields([('wr_id', uint64_t, 0), ('send_flags', ctypes.c_uint32, 8), ('bind_info', struct_ibv_mw_bind_info, 16)])
@c.record
class struct_ibv_wc(c.Struct):
  SIZE = 48
  wr_id: int
  status: int
  opcode: int
  vendor_err: int
  byte_len: int
  imm_data: int
  invalidated_rkey: int
  qp_num: int
  src_qp: int
  wc_flags: int
  pkey_index: int
  slid: int
  sl: int
  dlid_path_bits: int
enum_ibv_wc_status: dict[int, str] = {(IBV_WC_SUCCESS:=0): 'IBV_WC_SUCCESS', (IBV_WC_LOC_LEN_ERR:=1): 'IBV_WC_LOC_LEN_ERR', (IBV_WC_LOC_QP_OP_ERR:=2): 'IBV_WC_LOC_QP_OP_ERR', (IBV_WC_LOC_EEC_OP_ERR:=3): 'IBV_WC_LOC_EEC_OP_ERR', (IBV_WC_LOC_PROT_ERR:=4): 'IBV_WC_LOC_PROT_ERR', (IBV_WC_WR_FLUSH_ERR:=5): 'IBV_WC_WR_FLUSH_ERR', (IBV_WC_MW_BIND_ERR:=6): 'IBV_WC_MW_BIND_ERR', (IBV_WC_BAD_RESP_ERR:=7): 'IBV_WC_BAD_RESP_ERR', (IBV_WC_LOC_ACCESS_ERR:=8): 'IBV_WC_LOC_ACCESS_ERR', (IBV_WC_REM_INV_REQ_ERR:=9): 'IBV_WC_REM_INV_REQ_ERR', (IBV_WC_REM_ACCESS_ERR:=10): 'IBV_WC_REM_ACCESS_ERR', (IBV_WC_REM_OP_ERR:=11): 'IBV_WC_REM_OP_ERR', (IBV_WC_RETRY_EXC_ERR:=12): 'IBV_WC_RETRY_EXC_ERR', (IBV_WC_RNR_RETRY_EXC_ERR:=13): 'IBV_WC_RNR_RETRY_EXC_ERR', (IBV_WC_LOC_RDD_VIOL_ERR:=14): 'IBV_WC_LOC_RDD_VIOL_ERR', (IBV_WC_REM_INV_RD_REQ_ERR:=15): 'IBV_WC_REM_INV_RD_REQ_ERR', (IBV_WC_REM_ABORT_ERR:=16): 'IBV_WC_REM_ABORT_ERR', (IBV_WC_INV_EECN_ERR:=17): 'IBV_WC_INV_EECN_ERR', (IBV_WC_INV_EEC_STATE_ERR:=18): 'IBV_WC_INV_EEC_STATE_ERR', (IBV_WC_FATAL_ERR:=19): 'IBV_WC_FATAL_ERR', (IBV_WC_RESP_TIMEOUT_ERR:=20): 'IBV_WC_RESP_TIMEOUT_ERR', (IBV_WC_GENERAL_ERR:=21): 'IBV_WC_GENERAL_ERR', (IBV_WC_TM_ERR:=22): 'IBV_WC_TM_ERR', (IBV_WC_TM_RNDV_INCOMPLETE:=23): 'IBV_WC_TM_RNDV_INCOMPLETE'}
enum_ibv_wc_opcode: dict[int, str] = {(IBV_WC_SEND:=0): 'IBV_WC_SEND', (IBV_WC_RDMA_WRITE:=1): 'IBV_WC_RDMA_WRITE', (IBV_WC_RDMA_READ:=2): 'IBV_WC_RDMA_READ', (IBV_WC_COMP_SWAP:=3): 'IBV_WC_COMP_SWAP', (IBV_WC_FETCH_ADD:=4): 'IBV_WC_FETCH_ADD', (IBV_WC_BIND_MW:=5): 'IBV_WC_BIND_MW', (IBV_WC_LOCAL_INV:=6): 'IBV_WC_LOCAL_INV', (IBV_WC_TSO:=7): 'IBV_WC_TSO', (IBV_WC_FLUSH:=8): 'IBV_WC_FLUSH', (IBV_WC_ATOMIC_WRITE:=9): 'IBV_WC_ATOMIC_WRITE', (IBV_WC_RECV:=128): 'IBV_WC_RECV', (IBV_WC_RECV_RDMA_WITH_IMM:=129): 'IBV_WC_RECV_RDMA_WITH_IMM', (IBV_WC_TM_ADD:=130): 'IBV_WC_TM_ADD', (IBV_WC_TM_DEL:=131): 'IBV_WC_TM_DEL', (IBV_WC_TM_SYNC:=132): 'IBV_WC_TM_SYNC', (IBV_WC_TM_RECV:=133): 'IBV_WC_TM_RECV', (IBV_WC_TM_NO_TAG:=134): 'IBV_WC_TM_NO_TAG', (IBV_WC_DRIVER1:=135): 'IBV_WC_DRIVER1', (IBV_WC_DRIVER2:=136): 'IBV_WC_DRIVER2', (IBV_WC_DRIVER3:=137): 'IBV_WC_DRIVER3'}
__be32: TypeAlias = ctypes.c_uint32
struct_ibv_wc.register_fields([('wr_id', uint64_t, 0), ('status', ctypes.c_uint32, 8), ('opcode', ctypes.c_uint32, 12), ('vendor_err', uint32_t, 16), ('byte_len', uint32_t, 20), ('imm_data', ctypes.c_uint32, 24), ('invalidated_rkey', uint32_t, 24), ('qp_num', uint32_t, 28), ('src_qp', uint32_t, 32), ('wc_flags', ctypes.c_uint32, 36), ('pkey_index', uint16_t, 40), ('slid', uint16_t, 42), ('sl', uint8_t, 44), ('dlid_path_bits', uint8_t, 45)])
@c.record
class struct_ibv_recv_wr(c.Struct):
  SIZE = 32
  wr_id: int
  next: c.POINTER[struct_ibv_recv_wr]
  sg_list: c.POINTER[struct_ibv_sge]
  num_sge: int
@c.record
class struct_ibv_sge(c.Struct):
  SIZE = 16
  addr: int
  length: int
  lkey: int
struct_ibv_sge.register_fields([('addr', uint64_t, 0), ('length', uint32_t, 8), ('lkey', uint32_t, 12)])
struct_ibv_recv_wr.register_fields([('wr_id', uint64_t, 0), ('next', c.POINTER[struct_ibv_recv_wr], 8), ('sg_list', c.POINTER[struct_ibv_sge], 16), ('num_sge', ctypes.c_int32, 24)])
@c.record
class struct_ibv_send_wr(c.Struct):
  SIZE = 128
  wr_id: int
  next: c.POINTER[struct_ibv_send_wr]
  sg_list: c.POINTER[struct_ibv_sge]
  num_sge: int
  opcode: int
  send_flags: int
  imm_data: int
  invalidate_rkey: int
  wr: struct_ibv_send_wr_wr
  qp_type: struct_ibv_send_wr_qp_type
  bind_mw: struct_ibv_send_wr_bind_mw
  tso: struct_ibv_send_wr_tso
enum_ibv_wr_opcode: dict[int, str] = {(IBV_WR_RDMA_WRITE:=0): 'IBV_WR_RDMA_WRITE', (IBV_WR_RDMA_WRITE_WITH_IMM:=1): 'IBV_WR_RDMA_WRITE_WITH_IMM', (IBV_WR_SEND:=2): 'IBV_WR_SEND', (IBV_WR_SEND_WITH_IMM:=3): 'IBV_WR_SEND_WITH_IMM', (IBV_WR_RDMA_READ:=4): 'IBV_WR_RDMA_READ', (IBV_WR_ATOMIC_CMP_AND_SWP:=5): 'IBV_WR_ATOMIC_CMP_AND_SWP', (IBV_WR_ATOMIC_FETCH_AND_ADD:=6): 'IBV_WR_ATOMIC_FETCH_AND_ADD', (IBV_WR_LOCAL_INV:=7): 'IBV_WR_LOCAL_INV', (IBV_WR_BIND_MW:=8): 'IBV_WR_BIND_MW', (IBV_WR_SEND_WITH_INV:=9): 'IBV_WR_SEND_WITH_INV', (IBV_WR_TSO:=10): 'IBV_WR_TSO', (IBV_WR_DRIVER1:=11): 'IBV_WR_DRIVER1', (IBV_WR_FLUSH:=14): 'IBV_WR_FLUSH', (IBV_WR_ATOMIC_WRITE:=15): 'IBV_WR_ATOMIC_WRITE'}
@c.record
class struct_ibv_send_wr_wr(c.Struct):
  SIZE = 32
  rdma: struct_ibv_send_wr_wr_rdma
  atomic: struct_ibv_send_wr_wr_atomic
  ud: struct_ibv_send_wr_wr_ud
@c.record
class struct_ibv_send_wr_wr_rdma(c.Struct):
  SIZE = 16
  remote_addr: int
  rkey: int
struct_ibv_send_wr_wr_rdma.register_fields([('remote_addr', uint64_t, 0), ('rkey', uint32_t, 8)])
@c.record
class struct_ibv_send_wr_wr_atomic(c.Struct):
  SIZE = 32
  remote_addr: int
  compare_add: int
  swap: int
  rkey: int
struct_ibv_send_wr_wr_atomic.register_fields([('remote_addr', uint64_t, 0), ('compare_add', uint64_t, 8), ('swap', uint64_t, 16), ('rkey', uint32_t, 24)])
@c.record
class struct_ibv_send_wr_wr_ud(c.Struct):
  SIZE = 16
  ah: c.POINTER[struct_ibv_ah]
  remote_qpn: int
  remote_qkey: int
@c.record
class struct_ibv_ah(c.Struct):
  SIZE = 24
  context: c.POINTER[struct_ibv_context]
  pd: c.POINTER[struct_ibv_pd]
  handle: int
struct_ibv_ah.register_fields([('context', c.POINTER[struct_ibv_context], 0), ('pd', c.POINTER[struct_ibv_pd], 8), ('handle', uint32_t, 16)])
struct_ibv_send_wr_wr_ud.register_fields([('ah', c.POINTER[struct_ibv_ah], 0), ('remote_qpn', uint32_t, 8), ('remote_qkey', uint32_t, 12)])
struct_ibv_send_wr_wr.register_fields([('rdma', struct_ibv_send_wr_wr_rdma, 0), ('atomic', struct_ibv_send_wr_wr_atomic, 0), ('ud', struct_ibv_send_wr_wr_ud, 0)])
@c.record
class struct_ibv_send_wr_qp_type(c.Struct):
  SIZE = 4
  xrc: struct_ibv_send_wr_qp_type_xrc
@c.record
class struct_ibv_send_wr_qp_type_xrc(c.Struct):
  SIZE = 4
  remote_srqn: int
struct_ibv_send_wr_qp_type_xrc.register_fields([('remote_srqn', uint32_t, 0)])
struct_ibv_send_wr_qp_type.register_fields([('xrc', struct_ibv_send_wr_qp_type_xrc, 0)])
@c.record
class struct_ibv_send_wr_bind_mw(c.Struct):
  SIZE = 48
  mw: c.POINTER[struct_ibv_mw]
  rkey: int
  bind_info: struct_ibv_mw_bind_info
struct_ibv_send_wr_bind_mw.register_fields([('mw', c.POINTER[struct_ibv_mw], 0), ('rkey', uint32_t, 8), ('bind_info', struct_ibv_mw_bind_info, 16)])
@c.record
class struct_ibv_send_wr_tso(c.Struct):
  SIZE = 16
  hdr: ctypes.c_void_p
  hdr_sz: int
  mss: int
struct_ibv_send_wr_tso.register_fields([('hdr', ctypes.c_void_p, 0), ('hdr_sz', uint16_t, 8), ('mss', uint16_t, 10)])
struct_ibv_send_wr.register_fields([('wr_id', uint64_t, 0), ('next', c.POINTER[struct_ibv_send_wr], 8), ('sg_list', c.POINTER[struct_ibv_sge], 16), ('num_sge', ctypes.c_int32, 24), ('opcode', ctypes.c_uint32, 28), ('send_flags', ctypes.c_uint32, 32), ('imm_data', ctypes.c_uint32, 36), ('invalidate_rkey', uint32_t, 36), ('wr', struct_ibv_send_wr_wr, 40), ('qp_type', struct_ibv_send_wr_qp_type, 72), ('bind_mw', struct_ibv_send_wr_bind_mw, 80), ('tso', struct_ibv_send_wr_tso, 80)])
struct_ibv_context_ops.register_fields([('_compat_query_device', c.CFUNCTYPE[ctypes.c_int32, [c.POINTER[struct_ibv_context], c.POINTER[struct_ibv_device_attr]]], 0), ('_compat_query_port', c.CFUNCTYPE[ctypes.c_int32, [c.POINTER[struct_ibv_context], uint8_t, c.POINTER[struct__compat_ibv_port_attr]]], 8), ('_compat_alloc_pd', c.CFUNCTYPE[ctypes.c_void_p, []], 16), ('_compat_dealloc_pd', c.CFUNCTYPE[ctypes.c_void_p, []], 24), ('_compat_reg_mr', c.CFUNCTYPE[ctypes.c_void_p, []], 32), ('_compat_rereg_mr', c.CFUNCTYPE[ctypes.c_void_p, []], 40), ('_compat_dereg_mr', c.CFUNCTYPE[ctypes.c_void_p, []], 48), ('alloc_mw', c.CFUNCTYPE[c.POINTER[struct_ibv_mw], [c.POINTER[struct_ibv_pd], ctypes.c_uint32]], 56), ('bind_mw', c.CFUNCTYPE[ctypes.c_int32, [c.POINTER[struct_ibv_qp], c.POINTER[struct_ibv_mw], c.POINTER[struct_ibv_mw_bind]]], 64), ('dealloc_mw', c.CFUNCTYPE[ctypes.c_int32, [c.POINTER[struct_ibv_mw]]], 72), ('_compat_create_cq', c.CFUNCTYPE[ctypes.c_void_p, []], 80), ('poll_cq', c.CFUNCTYPE[ctypes.c_int32, [c.POINTER[struct_ibv_cq], ctypes.c_int32, c.POINTER[struct_ibv_wc]]], 88), ('req_notify_cq', c.CFUNCTYPE[ctypes.c_int32, [c.POINTER[struct_ibv_cq], ctypes.c_int32]], 96), ('_compat_cq_event', c.CFUNCTYPE[ctypes.c_void_p, []], 104), ('_compat_resize_cq', c.CFUNCTYPE[ctypes.c_void_p, []], 112), ('_compat_destroy_cq', c.CFUNCTYPE[ctypes.c_void_p, []], 120), ('_compat_create_srq', c.CFUNCTYPE[ctypes.c_void_p, []], 128), ('_compat_modify_srq', c.CFUNCTYPE[ctypes.c_void_p, []], 136), ('_compat_query_srq', c.CFUNCTYPE[ctypes.c_void_p, []], 144), ('_compat_destroy_srq', c.CFUNCTYPE[ctypes.c_void_p, []], 152), ('post_srq_recv', c.CFUNCTYPE[ctypes.c_int32, [c.POINTER[struct_ibv_srq], c.POINTER[struct_ibv_recv_wr], c.POINTER[c.POINTER[struct_ibv_recv_wr]]]], 160), ('_compat_create_qp', c.CFUNCTYPE[ctypes.c_void_p, []], 168), ('_compat_query_qp', c.CFUNCTYPE[ctypes.c_void_p, []], 176), ('_compat_modify_qp', c.CFUNCTYPE[ctypes.c_void_p, []], 184), ('_compat_destroy_qp', c.CFUNCTYPE[ctypes.c_void_p, []], 192), ('post_send', c.CFUNCTYPE[ctypes.c_int32, [c.POINTER[struct_ibv_qp], c.POINTER[struct_ibv_send_wr], c.POINTER[c.POINTER[struct_ibv_send_wr]]]], 200), ('post_recv', c.CFUNCTYPE[ctypes.c_int32, [c.POINTER[struct_ibv_qp], c.POINTER[struct_ibv_recv_wr], c.POINTER[c.POINTER[struct_ibv_recv_wr]]]], 208), ('_compat_create_ah', c.CFUNCTYPE[ctypes.c_void_p, []], 216), ('_compat_destroy_ah', c.CFUNCTYPE[ctypes.c_void_p, []], 224), ('_compat_attach_mcast', c.CFUNCTYPE[ctypes.c_void_p, []], 232), ('_compat_detach_mcast', c.CFUNCTYPE[ctypes.c_void_p, []], 240), ('_compat_async_event', c.CFUNCTYPE[ctypes.c_void_p, []], 248)])
struct_ibv_context.register_fields([('device', c.POINTER[struct_ibv_device], 0), ('ops', struct_ibv_context_ops, 8), ('cmd_fd', ctypes.c_int32, 264), ('async_fd', ctypes.c_int32, 268), ('num_comp_vectors', ctypes.c_int32, 272), ('mutex', pthread_mutex_t, 280), ('abi_compat', ctypes.c_void_p, 320)])
struct_ibv_dm.register_fields([('context', c.POINTER[struct_ibv_context], 0), ('memcpy_to_dm', c.CFUNCTYPE[ctypes.c_int32, [c.POINTER[struct_ibv_dm], uint64_t, ctypes.c_void_p, size_t]], 8), ('memcpy_from_dm', c.CFUNCTYPE[ctypes.c_int32, [ctypes.c_void_p, c.POINTER[struct_ibv_dm], uint64_t, size_t]], 16), ('comp_mask', uint32_t, 24), ('handle', uint32_t, 28)])
@c.record
class struct_ibv_query_device_ex_input(c.Struct):
  SIZE = 4
  comp_mask: int
struct_ibv_query_device_ex_input.register_fields([('comp_mask', uint32_t, 0)])
enum_ibv_odp_transport_cap_bits: dict[int, str] = {(IBV_ODP_SUPPORT_SEND:=1): 'IBV_ODP_SUPPORT_SEND', (IBV_ODP_SUPPORT_RECV:=2): 'IBV_ODP_SUPPORT_RECV', (IBV_ODP_SUPPORT_WRITE:=4): 'IBV_ODP_SUPPORT_WRITE', (IBV_ODP_SUPPORT_READ:=8): 'IBV_ODP_SUPPORT_READ', (IBV_ODP_SUPPORT_ATOMIC:=16): 'IBV_ODP_SUPPORT_ATOMIC', (IBV_ODP_SUPPORT_SRQ_RECV:=32): 'IBV_ODP_SUPPORT_SRQ_RECV'}
@c.record
class struct_ibv_odp_caps(c.Struct):
  SIZE = 24
  general_caps: int
  per_transport_caps: struct_ibv_odp_caps_per_transport_caps
@c.record
class struct_ibv_odp_caps_per_transport_caps(c.Struct):
  SIZE = 12
  rc_odp_caps: int
  uc_odp_caps: int
  ud_odp_caps: int
struct_ibv_odp_caps_per_transport_caps.register_fields([('rc_odp_caps', uint32_t, 0), ('uc_odp_caps', uint32_t, 4), ('ud_odp_caps', uint32_t, 8)])
struct_ibv_odp_caps.register_fields([('general_caps', uint64_t, 0), ('per_transport_caps', struct_ibv_odp_caps_per_transport_caps, 8)])
enum_ibv_odp_general_caps: dict[int, str] = {(IBV_ODP_SUPPORT:=1): 'IBV_ODP_SUPPORT', (IBV_ODP_SUPPORT_IMPLICIT:=2): 'IBV_ODP_SUPPORT_IMPLICIT'}
@c.record
class struct_ibv_tso_caps(c.Struct):
  SIZE = 8
  max_tso: int
  supported_qpts: int
struct_ibv_tso_caps.register_fields([('max_tso', uint32_t, 0), ('supported_qpts', uint32_t, 4)])
enum_ibv_rx_hash_function_flags: dict[int, str] = {(IBV_RX_HASH_FUNC_TOEPLITZ:=1): 'IBV_RX_HASH_FUNC_TOEPLITZ'}
enum_ibv_rx_hash_fields: dict[int, str] = {(IBV_RX_HASH_SRC_IPV4:=1): 'IBV_RX_HASH_SRC_IPV4', (IBV_RX_HASH_DST_IPV4:=2): 'IBV_RX_HASH_DST_IPV4', (IBV_RX_HASH_SRC_IPV6:=4): 'IBV_RX_HASH_SRC_IPV6', (IBV_RX_HASH_DST_IPV6:=8): 'IBV_RX_HASH_DST_IPV6', (IBV_RX_HASH_SRC_PORT_TCP:=16): 'IBV_RX_HASH_SRC_PORT_TCP', (IBV_RX_HASH_DST_PORT_TCP:=32): 'IBV_RX_HASH_DST_PORT_TCP', (IBV_RX_HASH_SRC_PORT_UDP:=64): 'IBV_RX_HASH_SRC_PORT_UDP', (IBV_RX_HASH_DST_PORT_UDP:=128): 'IBV_RX_HASH_DST_PORT_UDP', (IBV_RX_HASH_IPSEC_SPI:=256): 'IBV_RX_HASH_IPSEC_SPI', (IBV_RX_HASH_INNER:=2147483648): 'IBV_RX_HASH_INNER'}
@c.record
class struct_ibv_rss_caps(c.Struct):
  SIZE = 32
  supported_qpts: int
  max_rwq_indirection_tables: int
  max_rwq_indirection_table_size: int
  rx_hash_fields_mask: int
  rx_hash_function: int
struct_ibv_rss_caps.register_fields([('supported_qpts', uint32_t, 0), ('max_rwq_indirection_tables', uint32_t, 4), ('max_rwq_indirection_table_size', uint32_t, 8), ('rx_hash_fields_mask', uint64_t, 16), ('rx_hash_function', uint8_t, 24)])
@c.record
class struct_ibv_packet_pacing_caps(c.Struct):
  SIZE = 12
  qp_rate_limit_min: int
  qp_rate_limit_max: int
  supported_qpts: int
struct_ibv_packet_pacing_caps.register_fields([('qp_rate_limit_min', uint32_t, 0), ('qp_rate_limit_max', uint32_t, 4), ('supported_qpts', uint32_t, 8)])
enum_ibv_raw_packet_caps: dict[int, str] = {(IBV_RAW_PACKET_CAP_CVLAN_STRIPPING:=1): 'IBV_RAW_PACKET_CAP_CVLAN_STRIPPING', (IBV_RAW_PACKET_CAP_SCATTER_FCS:=2): 'IBV_RAW_PACKET_CAP_SCATTER_FCS', (IBV_RAW_PACKET_CAP_IP_CSUM:=4): 'IBV_RAW_PACKET_CAP_IP_CSUM', (IBV_RAW_PACKET_CAP_DELAY_DROP:=8): 'IBV_RAW_PACKET_CAP_DELAY_DROP'}
enum_ibv_tm_cap_flags: dict[int, str] = {(IBV_TM_CAP_RC:=1): 'IBV_TM_CAP_RC'}
@c.record
class struct_ibv_tm_caps(c.Struct):
  SIZE = 20
  max_rndv_hdr_size: int
  max_num_tags: int
  flags: int
  max_ops: int
  max_sge: int
struct_ibv_tm_caps.register_fields([('max_rndv_hdr_size', uint32_t, 0), ('max_num_tags', uint32_t, 4), ('flags', uint32_t, 8), ('max_ops', uint32_t, 12), ('max_sge', uint32_t, 16)])
@c.record
class struct_ibv_cq_moderation_caps(c.Struct):
  SIZE = 4
  max_cq_count: int
  max_cq_period: int
struct_ibv_cq_moderation_caps.register_fields([('max_cq_count', uint16_t, 0), ('max_cq_period', uint16_t, 2)])
enum_ibv_pci_atomic_op_size: dict[int, str] = {(IBV_PCI_ATOMIC_OPERATION_4_BYTE_SIZE_SUP:=1): 'IBV_PCI_ATOMIC_OPERATION_4_BYTE_SIZE_SUP', (IBV_PCI_ATOMIC_OPERATION_8_BYTE_SIZE_SUP:=2): 'IBV_PCI_ATOMIC_OPERATION_8_BYTE_SIZE_SUP', (IBV_PCI_ATOMIC_OPERATION_16_BYTE_SIZE_SUP:=4): 'IBV_PCI_ATOMIC_OPERATION_16_BYTE_SIZE_SUP'}
@c.record
class struct_ibv_pci_atomic_caps(c.Struct):
  SIZE = 6
  fetch_add: int
  swap: int
  compare_swap: int
struct_ibv_pci_atomic_caps.register_fields([('fetch_add', uint16_t, 0), ('swap', uint16_t, 2), ('compare_swap', uint16_t, 4)])
@c.record
class struct_ibv_device_attr_ex(c.Struct):
  SIZE = 400
  orig_attr: struct_ibv_device_attr
  comp_mask: int
  odp_caps: struct_ibv_odp_caps
  completion_timestamp_mask: int
  hca_core_clock: int
  device_cap_flags_ex: int
  tso_caps: struct_ibv_tso_caps
  rss_caps: struct_ibv_rss_caps
  max_wq_type_rq: int
  packet_pacing_caps: struct_ibv_packet_pacing_caps
  raw_packet_caps: int
  tm_caps: struct_ibv_tm_caps
  cq_mod_caps: struct_ibv_cq_moderation_caps
  max_dm_size: int
  pci_atomic_caps: struct_ibv_pci_atomic_caps
  xrc_odp_caps: int
  phys_port_cnt_ex: int
struct_ibv_device_attr_ex.register_fields([('orig_attr', struct_ibv_device_attr, 0), ('comp_mask', uint32_t, 232), ('odp_caps', struct_ibv_odp_caps, 240), ('completion_timestamp_mask', uint64_t, 264), ('hca_core_clock', uint64_t, 272), ('device_cap_flags_ex', uint64_t, 280), ('tso_caps', struct_ibv_tso_caps, 288), ('rss_caps', struct_ibv_rss_caps, 296), ('max_wq_type_rq', uint32_t, 328), ('packet_pacing_caps', struct_ibv_packet_pacing_caps, 332), ('raw_packet_caps', uint32_t, 344), ('tm_caps', struct_ibv_tm_caps, 348), ('cq_mod_caps', struct_ibv_cq_moderation_caps, 368), ('max_dm_size', uint64_t, 376), ('pci_atomic_caps', struct_ibv_pci_atomic_caps, 384), ('xrc_odp_caps', uint32_t, 392), ('phys_port_cnt_ex', uint32_t, 396)])
enum_ibv_mtu: dict[int, str] = {(IBV_MTU_256:=1): 'IBV_MTU_256', (IBV_MTU_512:=2): 'IBV_MTU_512', (IBV_MTU_1024:=3): 'IBV_MTU_1024', (IBV_MTU_2048:=4): 'IBV_MTU_2048', (IBV_MTU_4096:=5): 'IBV_MTU_4096'}
enum_ibv_port_state: dict[int, str] = {(IBV_PORT_NOP:=0): 'IBV_PORT_NOP', (IBV_PORT_DOWN:=1): 'IBV_PORT_DOWN', (IBV_PORT_INIT:=2): 'IBV_PORT_INIT', (IBV_PORT_ARMED:=3): 'IBV_PORT_ARMED', (IBV_PORT_ACTIVE:=4): 'IBV_PORT_ACTIVE', (IBV_PORT_ACTIVE_DEFER:=5): 'IBV_PORT_ACTIVE_DEFER'}
_anonenum0: dict[int, str] = {(IBV_LINK_LAYER_UNSPECIFIED:=0): 'IBV_LINK_LAYER_UNSPECIFIED', (IBV_LINK_LAYER_INFINIBAND:=1): 'IBV_LINK_LAYER_INFINIBAND', (IBV_LINK_LAYER_ETHERNET:=2): 'IBV_LINK_LAYER_ETHERNET'}
enum_ibv_port_cap_flags: dict[int, str] = {(IBV_PORT_SM:=2): 'IBV_PORT_SM', (IBV_PORT_NOTICE_SUP:=4): 'IBV_PORT_NOTICE_SUP', (IBV_PORT_TRAP_SUP:=8): 'IBV_PORT_TRAP_SUP', (IBV_PORT_OPT_IPD_SUP:=16): 'IBV_PORT_OPT_IPD_SUP', (IBV_PORT_AUTO_MIGR_SUP:=32): 'IBV_PORT_AUTO_MIGR_SUP', (IBV_PORT_SL_MAP_SUP:=64): 'IBV_PORT_SL_MAP_SUP', (IBV_PORT_MKEY_NVRAM:=128): 'IBV_PORT_MKEY_NVRAM', (IBV_PORT_PKEY_NVRAM:=256): 'IBV_PORT_PKEY_NVRAM', (IBV_PORT_LED_INFO_SUP:=512): 'IBV_PORT_LED_INFO_SUP', (IBV_PORT_SYS_IMAGE_GUID_SUP:=2048): 'IBV_PORT_SYS_IMAGE_GUID_SUP', (IBV_PORT_PKEY_SW_EXT_PORT_TRAP_SUP:=4096): 'IBV_PORT_PKEY_SW_EXT_PORT_TRAP_SUP', (IBV_PORT_EXTENDED_SPEEDS_SUP:=16384): 'IBV_PORT_EXTENDED_SPEEDS_SUP', (IBV_PORT_CAP_MASK2_SUP:=32768): 'IBV_PORT_CAP_MASK2_SUP', (IBV_PORT_CM_SUP:=65536): 'IBV_PORT_CM_SUP', (IBV_PORT_SNMP_TUNNEL_SUP:=131072): 'IBV_PORT_SNMP_TUNNEL_SUP', (IBV_PORT_REINIT_SUP:=262144): 'IBV_PORT_REINIT_SUP', (IBV_PORT_DEVICE_MGMT_SUP:=524288): 'IBV_PORT_DEVICE_MGMT_SUP', (IBV_PORT_VENDOR_CLASS_SUP:=1048576): 'IBV_PORT_VENDOR_CLASS_SUP', (IBV_PORT_DR_NOTICE_SUP:=2097152): 'IBV_PORT_DR_NOTICE_SUP', (IBV_PORT_CAP_MASK_NOTICE_SUP:=4194304): 'IBV_PORT_CAP_MASK_NOTICE_SUP', (IBV_PORT_BOOT_MGMT_SUP:=8388608): 'IBV_PORT_BOOT_MGMT_SUP', (IBV_PORT_LINK_LATENCY_SUP:=16777216): 'IBV_PORT_LINK_LATENCY_SUP', (IBV_PORT_CLIENT_REG_SUP:=33554432): 'IBV_PORT_CLIENT_REG_SUP', (IBV_PORT_IP_BASED_GIDS:=67108864): 'IBV_PORT_IP_BASED_GIDS'}
enum_ibv_port_cap_flags2: dict[int, str] = {(IBV_PORT_SET_NODE_DESC_SUP:=1): 'IBV_PORT_SET_NODE_DESC_SUP', (IBV_PORT_INFO_EXT_SUP:=2): 'IBV_PORT_INFO_EXT_SUP', (IBV_PORT_VIRT_SUP:=4): 'IBV_PORT_VIRT_SUP', (IBV_PORT_SWITCH_PORT_STATE_TABLE_SUP:=8): 'IBV_PORT_SWITCH_PORT_STATE_TABLE_SUP', (IBV_PORT_LINK_WIDTH_2X_SUP:=16): 'IBV_PORT_LINK_WIDTH_2X_SUP', (IBV_PORT_LINK_SPEED_HDR_SUP:=32): 'IBV_PORT_LINK_SPEED_HDR_SUP', (IBV_PORT_LINK_SPEED_NDR_SUP:=1024): 'IBV_PORT_LINK_SPEED_NDR_SUP', (IBV_PORT_LINK_SPEED_XDR_SUP:=4096): 'IBV_PORT_LINK_SPEED_XDR_SUP'}
@c.record
class struct_ibv_port_attr(c.Struct):
  SIZE = 56
  state: int
  max_mtu: int
  active_mtu: int
  gid_tbl_len: int
  port_cap_flags: int
  max_msg_sz: int
  bad_pkey_cntr: int
  qkey_viol_cntr: int
  pkey_tbl_len: int
  lid: int
  sm_lid: int
  lmc: int
  max_vl_num: int
  sm_sl: int
  subnet_timeout: int
  init_type_reply: int
  active_width: int
  active_speed: int
  phys_state: int
  link_layer: int
  flags: int
  port_cap_flags2: int
  active_speed_ex: int
struct_ibv_port_attr.register_fields([('state', ctypes.c_uint32, 0), ('max_mtu', ctypes.c_uint32, 4), ('active_mtu', ctypes.c_uint32, 8), ('gid_tbl_len', ctypes.c_int32, 12), ('port_cap_flags', uint32_t, 16), ('max_msg_sz', uint32_t, 20), ('bad_pkey_cntr', uint32_t, 24), ('qkey_viol_cntr', uint32_t, 28), ('pkey_tbl_len', uint16_t, 32), ('lid', uint16_t, 34), ('sm_lid', uint16_t, 36), ('lmc', uint8_t, 38), ('max_vl_num', uint8_t, 39), ('sm_sl', uint8_t, 40), ('subnet_timeout', uint8_t, 41), ('init_type_reply', uint8_t, 42), ('active_width', uint8_t, 43), ('active_speed', uint8_t, 44), ('phys_state', uint8_t, 45), ('link_layer', uint8_t, 46), ('flags', uint8_t, 47), ('port_cap_flags2', uint16_t, 48), ('active_speed_ex', uint32_t, 52)])
enum_ibv_event_type: dict[int, str] = {(IBV_EVENT_CQ_ERR:=0): 'IBV_EVENT_CQ_ERR', (IBV_EVENT_QP_FATAL:=1): 'IBV_EVENT_QP_FATAL', (IBV_EVENT_QP_REQ_ERR:=2): 'IBV_EVENT_QP_REQ_ERR', (IBV_EVENT_QP_ACCESS_ERR:=3): 'IBV_EVENT_QP_ACCESS_ERR', (IBV_EVENT_COMM_EST:=4): 'IBV_EVENT_COMM_EST', (IBV_EVENT_SQ_DRAINED:=5): 'IBV_EVENT_SQ_DRAINED', (IBV_EVENT_PATH_MIG:=6): 'IBV_EVENT_PATH_MIG', (IBV_EVENT_PATH_MIG_ERR:=7): 'IBV_EVENT_PATH_MIG_ERR', (IBV_EVENT_DEVICE_FATAL:=8): 'IBV_EVENT_DEVICE_FATAL', (IBV_EVENT_PORT_ACTIVE:=9): 'IBV_EVENT_PORT_ACTIVE', (IBV_EVENT_PORT_ERR:=10): 'IBV_EVENT_PORT_ERR', (IBV_EVENT_LID_CHANGE:=11): 'IBV_EVENT_LID_CHANGE', (IBV_EVENT_PKEY_CHANGE:=12): 'IBV_EVENT_PKEY_CHANGE', (IBV_EVENT_SM_CHANGE:=13): 'IBV_EVENT_SM_CHANGE', (IBV_EVENT_SRQ_ERR:=14): 'IBV_EVENT_SRQ_ERR', (IBV_EVENT_SRQ_LIMIT_REACHED:=15): 'IBV_EVENT_SRQ_LIMIT_REACHED', (IBV_EVENT_QP_LAST_WQE_REACHED:=16): 'IBV_EVENT_QP_LAST_WQE_REACHED', (IBV_EVENT_CLIENT_REREGISTER:=17): 'IBV_EVENT_CLIENT_REREGISTER', (IBV_EVENT_GID_CHANGE:=18): 'IBV_EVENT_GID_CHANGE', (IBV_EVENT_WQ_FATAL:=19): 'IBV_EVENT_WQ_FATAL'}
@c.record
class struct_ibv_async_event(c.Struct):
  SIZE = 16
  element: struct_ibv_async_event_element
  event_type: int
@c.record
class struct_ibv_async_event_element(c.Struct):
  SIZE = 8
  cq: c.POINTER[struct_ibv_cq]
  qp: c.POINTER[struct_ibv_qp]
  srq: c.POINTER[struct_ibv_srq]
  wq: c.POINTER[struct_ibv_wq]
  port_num: int
@c.record
class struct_ibv_wq(c.Struct):
  SIZE = 152
  context: c.POINTER[struct_ibv_context]
  wq_context: ctypes.c_void_p
  pd: c.POINTER[struct_ibv_pd]
  cq: c.POINTER[struct_ibv_cq]
  wq_num: int
  handle: int
  state: int
  wq_type: int
  post_recv: c.CFUNCTYPE[ctypes.c_int32, [c.POINTER[struct_ibv_wq], c.POINTER[struct_ibv_recv_wr], c.POINTER[c.POINTER[struct_ibv_recv_wr]]]]
  mutex: pthread_mutex_t
  cond: pthread_cond_t
  events_completed: int
  comp_mask: int
enum_ibv_wq_state: dict[int, str] = {(IBV_WQS_RESET:=0): 'IBV_WQS_RESET', (IBV_WQS_RDY:=1): 'IBV_WQS_RDY', (IBV_WQS_ERR:=2): 'IBV_WQS_ERR', (IBV_WQS_UNKNOWN:=3): 'IBV_WQS_UNKNOWN'}
enum_ibv_wq_type: dict[int, str] = {(IBV_WQT_RQ:=0): 'IBV_WQT_RQ'}
struct_ibv_wq.register_fields([('context', c.POINTER[struct_ibv_context], 0), ('wq_context', ctypes.c_void_p, 8), ('pd', c.POINTER[struct_ibv_pd], 16), ('cq', c.POINTER[struct_ibv_cq], 24), ('wq_num', uint32_t, 32), ('handle', uint32_t, 36), ('state', ctypes.c_uint32, 40), ('wq_type', ctypes.c_uint32, 44), ('post_recv', c.CFUNCTYPE[ctypes.c_int32, [c.POINTER[struct_ibv_wq], c.POINTER[struct_ibv_recv_wr], c.POINTER[c.POINTER[struct_ibv_recv_wr]]]], 48), ('mutex', pthread_mutex_t, 56), ('cond', pthread_cond_t, 96), ('events_completed', uint32_t, 144), ('comp_mask', uint32_t, 148)])
struct_ibv_async_event_element.register_fields([('cq', c.POINTER[struct_ibv_cq], 0), ('qp', c.POINTER[struct_ibv_qp], 0), ('srq', c.POINTER[struct_ibv_srq], 0), ('wq', c.POINTER[struct_ibv_wq], 0), ('port_num', ctypes.c_int32, 0)])
struct_ibv_async_event.register_fields([('element', struct_ibv_async_event_element, 0), ('event_type', ctypes.c_uint32, 8)])
@dll.bind(c.POINTER[ctypes.c_char], ctypes.c_uint32)
def ibv_wc_status_str(status:ctypes.c_uint32) -> c.POINTER[ctypes.c_char]: ...
_anonenum1: dict[int, str] = {(IBV_WC_IP_CSUM_OK_SHIFT:=2): 'IBV_WC_IP_CSUM_OK_SHIFT'}
enum_ibv_create_cq_wc_flags: dict[int, str] = {(IBV_WC_EX_WITH_BYTE_LEN:=1): 'IBV_WC_EX_WITH_BYTE_LEN', (IBV_WC_EX_WITH_IMM:=2): 'IBV_WC_EX_WITH_IMM', (IBV_WC_EX_WITH_QP_NUM:=4): 'IBV_WC_EX_WITH_QP_NUM', (IBV_WC_EX_WITH_SRC_QP:=8): 'IBV_WC_EX_WITH_SRC_QP', (IBV_WC_EX_WITH_SLID:=16): 'IBV_WC_EX_WITH_SLID', (IBV_WC_EX_WITH_SL:=32): 'IBV_WC_EX_WITH_SL', (IBV_WC_EX_WITH_DLID_PATH_BITS:=64): 'IBV_WC_EX_WITH_DLID_PATH_BITS', (IBV_WC_EX_WITH_COMPLETION_TIMESTAMP:=128): 'IBV_WC_EX_WITH_COMPLETION_TIMESTAMP', (IBV_WC_EX_WITH_CVLAN:=256): 'IBV_WC_EX_WITH_CVLAN', (IBV_WC_EX_WITH_FLOW_TAG:=512): 'IBV_WC_EX_WITH_FLOW_TAG', (IBV_WC_EX_WITH_TM_INFO:=1024): 'IBV_WC_EX_WITH_TM_INFO', (IBV_WC_EX_WITH_COMPLETION_TIMESTAMP_WALLCLOCK:=2048): 'IBV_WC_EX_WITH_COMPLETION_TIMESTAMP_WALLCLOCK'}
_anonenum2: dict[int, str] = {(IBV_WC_STANDARD_FLAGS:=127): 'IBV_WC_STANDARD_FLAGS'}
_anonenum3: dict[int, str] = {(IBV_CREATE_CQ_SUP_WC_FLAGS:=4095): 'IBV_CREATE_CQ_SUP_WC_FLAGS'}
enum_ibv_wc_flags: dict[int, str] = {(IBV_WC_GRH:=1): 'IBV_WC_GRH', (IBV_WC_WITH_IMM:=2): 'IBV_WC_WITH_IMM', (IBV_WC_IP_CSUM_OK:=4): 'IBV_WC_IP_CSUM_OK', (IBV_WC_WITH_INV:=8): 'IBV_WC_WITH_INV', (IBV_WC_TM_SYNC_REQ:=16): 'IBV_WC_TM_SYNC_REQ', (IBV_WC_TM_MATCH:=32): 'IBV_WC_TM_MATCH', (IBV_WC_TM_DATA_VALID:=64): 'IBV_WC_TM_DATA_VALID'}
enum_ibv_access_flags: dict[int, str] = {(IBV_ACCESS_LOCAL_WRITE:=1): 'IBV_ACCESS_LOCAL_WRITE', (IBV_ACCESS_REMOTE_WRITE:=2): 'IBV_ACCESS_REMOTE_WRITE', (IBV_ACCESS_REMOTE_READ:=4): 'IBV_ACCESS_REMOTE_READ', (IBV_ACCESS_REMOTE_ATOMIC:=8): 'IBV_ACCESS_REMOTE_ATOMIC', (IBV_ACCESS_MW_BIND:=16): 'IBV_ACCESS_MW_BIND', (IBV_ACCESS_ZERO_BASED:=32): 'IBV_ACCESS_ZERO_BASED', (IBV_ACCESS_ON_DEMAND:=64): 'IBV_ACCESS_ON_DEMAND', (IBV_ACCESS_HUGETLB:=128): 'IBV_ACCESS_HUGETLB', (IBV_ACCESS_FLUSH_GLOBAL:=256): 'IBV_ACCESS_FLUSH_GLOBAL', (IBV_ACCESS_FLUSH_PERSISTENT:=512): 'IBV_ACCESS_FLUSH_PERSISTENT', (IBV_ACCESS_RELAXED_ORDERING:=1048576): 'IBV_ACCESS_RELAXED_ORDERING'}
@c.record
class struct_ibv_td_init_attr(c.Struct):
  SIZE = 4
  comp_mask: int
struct_ibv_td_init_attr.register_fields([('comp_mask', uint32_t, 0)])
@c.record
class struct_ibv_td(c.Struct):
  SIZE = 8
  context: c.POINTER[struct_ibv_context]
struct_ibv_td.register_fields([('context', c.POINTER[struct_ibv_context], 0)])
enum_ibv_xrcd_init_attr_mask: dict[int, str] = {(IBV_XRCD_INIT_ATTR_FD:=1): 'IBV_XRCD_INIT_ATTR_FD', (IBV_XRCD_INIT_ATTR_OFLAGS:=2): 'IBV_XRCD_INIT_ATTR_OFLAGS', (IBV_XRCD_INIT_ATTR_RESERVED:=4): 'IBV_XRCD_INIT_ATTR_RESERVED'}
@c.record
class struct_ibv_xrcd_init_attr(c.Struct):
  SIZE = 12
  comp_mask: int
  fd: int
  oflags: int
struct_ibv_xrcd_init_attr.register_fields([('comp_mask', uint32_t, 0), ('fd', ctypes.c_int32, 4), ('oflags', ctypes.c_int32, 8)])
@c.record
class struct_ibv_xrcd(c.Struct):
  SIZE = 8
  context: c.POINTER[struct_ibv_context]
struct_ibv_xrcd.register_fields([('context', c.POINTER[struct_ibv_context], 0)])
enum_ibv_rereg_mr_flags: dict[int, str] = {(IBV_REREG_MR_CHANGE_TRANSLATION:=1): 'IBV_REREG_MR_CHANGE_TRANSLATION', (IBV_REREG_MR_CHANGE_PD:=2): 'IBV_REREG_MR_CHANGE_PD', (IBV_REREG_MR_CHANGE_ACCESS:=4): 'IBV_REREG_MR_CHANGE_ACCESS', (IBV_REREG_MR_FLAGS_SUPPORTED:=7): 'IBV_REREG_MR_FLAGS_SUPPORTED'}
@c.record
class struct_ibv_global_route(c.Struct):
  SIZE = 24
  dgid: union_ibv_gid
  flow_label: int
  sgid_index: int
  hop_limit: int
  traffic_class: int
struct_ibv_global_route.register_fields([('dgid', union_ibv_gid, 0), ('flow_label', uint32_t, 16), ('sgid_index', uint8_t, 20), ('hop_limit', uint8_t, 21), ('traffic_class', uint8_t, 22)])
@c.record
class struct_ibv_grh(c.Struct):
  SIZE = 40
  version_tclass_flow: int
  paylen: int
  next_hdr: int
  hop_limit: int
  sgid: union_ibv_gid
  dgid: union_ibv_gid
__be16: TypeAlias = ctypes.c_uint16
struct_ibv_grh.register_fields([('version_tclass_flow', ctypes.c_uint32, 0), ('paylen', ctypes.c_uint16, 4), ('next_hdr', uint8_t, 6), ('hop_limit', uint8_t, 7), ('sgid', union_ibv_gid, 8), ('dgid', union_ibv_gid, 24)])
enum_ibv_rate: dict[int, str] = {(IBV_RATE_MAX:=0): 'IBV_RATE_MAX', (IBV_RATE_2_5_GBPS:=2): 'IBV_RATE_2_5_GBPS', (IBV_RATE_5_GBPS:=5): 'IBV_RATE_5_GBPS', (IBV_RATE_10_GBPS:=3): 'IBV_RATE_10_GBPS', (IBV_RATE_20_GBPS:=6): 'IBV_RATE_20_GBPS', (IBV_RATE_30_GBPS:=4): 'IBV_RATE_30_GBPS', (IBV_RATE_40_GBPS:=7): 'IBV_RATE_40_GBPS', (IBV_RATE_60_GBPS:=8): 'IBV_RATE_60_GBPS', (IBV_RATE_80_GBPS:=9): 'IBV_RATE_80_GBPS', (IBV_RATE_120_GBPS:=10): 'IBV_RATE_120_GBPS', (IBV_RATE_14_GBPS:=11): 'IBV_RATE_14_GBPS', (IBV_RATE_56_GBPS:=12): 'IBV_RATE_56_GBPS', (IBV_RATE_112_GBPS:=13): 'IBV_RATE_112_GBPS', (IBV_RATE_168_GBPS:=14): 'IBV_RATE_168_GBPS', (IBV_RATE_25_GBPS:=15): 'IBV_RATE_25_GBPS', (IBV_RATE_100_GBPS:=16): 'IBV_RATE_100_GBPS', (IBV_RATE_200_GBPS:=17): 'IBV_RATE_200_GBPS', (IBV_RATE_300_GBPS:=18): 'IBV_RATE_300_GBPS', (IBV_RATE_28_GBPS:=19): 'IBV_RATE_28_GBPS', (IBV_RATE_50_GBPS:=20): 'IBV_RATE_50_GBPS', (IBV_RATE_400_GBPS:=21): 'IBV_RATE_400_GBPS', (IBV_RATE_600_GBPS:=22): 'IBV_RATE_600_GBPS', (IBV_RATE_800_GBPS:=23): 'IBV_RATE_800_GBPS', (IBV_RATE_1200_GBPS:=24): 'IBV_RATE_1200_GBPS'}
@dll.bind(ctypes.c_int32, ctypes.c_uint32)
def ibv_rate_to_mult(rate:ctypes.c_uint32) -> int: ...
@dll.bind(ctypes.c_uint32, ctypes.c_int32)
def mult_to_ibv_rate(mult:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_int32, ctypes.c_uint32)
def ibv_rate_to_mbps(rate:ctypes.c_uint32) -> int: ...
@dll.bind(ctypes.c_uint32, ctypes.c_int32)
def mbps_to_ibv_rate(mbps:int) -> ctypes.c_uint32: ...
@c.record
class struct_ibv_ah_attr(c.Struct):
  SIZE = 32
  grh: struct_ibv_global_route
  dlid: int
  sl: int
  src_path_bits: int
  static_rate: int
  is_global: int
  port_num: int
struct_ibv_ah_attr.register_fields([('grh', struct_ibv_global_route, 0), ('dlid', uint16_t, 24), ('sl', uint8_t, 26), ('src_path_bits', uint8_t, 27), ('static_rate', uint8_t, 28), ('is_global', uint8_t, 29), ('port_num', uint8_t, 30)])
enum_ibv_srq_attr_mask: dict[int, str] = {(IBV_SRQ_MAX_WR:=1): 'IBV_SRQ_MAX_WR', (IBV_SRQ_LIMIT:=2): 'IBV_SRQ_LIMIT'}
@c.record
class struct_ibv_srq_attr(c.Struct):
  SIZE = 12
  max_wr: int
  max_sge: int
  srq_limit: int
struct_ibv_srq_attr.register_fields([('max_wr', uint32_t, 0), ('max_sge', uint32_t, 4), ('srq_limit', uint32_t, 8)])
@c.record
class struct_ibv_srq_init_attr(c.Struct):
  SIZE = 24
  srq_context: ctypes.c_void_p
  attr: struct_ibv_srq_attr
struct_ibv_srq_init_attr.register_fields([('srq_context', ctypes.c_void_p, 0), ('attr', struct_ibv_srq_attr, 8)])
enum_ibv_srq_type: dict[int, str] = {(IBV_SRQT_BASIC:=0): 'IBV_SRQT_BASIC', (IBV_SRQT_XRC:=1): 'IBV_SRQT_XRC', (IBV_SRQT_TM:=2): 'IBV_SRQT_TM'}
enum_ibv_srq_init_attr_mask: dict[int, str] = {(IBV_SRQ_INIT_ATTR_TYPE:=1): 'IBV_SRQ_INIT_ATTR_TYPE', (IBV_SRQ_INIT_ATTR_PD:=2): 'IBV_SRQ_INIT_ATTR_PD', (IBV_SRQ_INIT_ATTR_XRCD:=4): 'IBV_SRQ_INIT_ATTR_XRCD', (IBV_SRQ_INIT_ATTR_CQ:=8): 'IBV_SRQ_INIT_ATTR_CQ', (IBV_SRQ_INIT_ATTR_TM:=16): 'IBV_SRQ_INIT_ATTR_TM', (IBV_SRQ_INIT_ATTR_RESERVED:=32): 'IBV_SRQ_INIT_ATTR_RESERVED'}
@c.record
class struct_ibv_tm_cap(c.Struct):
  SIZE = 8
  max_num_tags: int
  max_ops: int
struct_ibv_tm_cap.register_fields([('max_num_tags', uint32_t, 0), ('max_ops', uint32_t, 4)])
@c.record
class struct_ibv_srq_init_attr_ex(c.Struct):
  SIZE = 64
  srq_context: ctypes.c_void_p
  attr: struct_ibv_srq_attr
  comp_mask: int
  srq_type: int
  pd: c.POINTER[struct_ibv_pd]
  xrcd: c.POINTER[struct_ibv_xrcd]
  cq: c.POINTER[struct_ibv_cq]
  tm_cap: struct_ibv_tm_cap
struct_ibv_srq_init_attr_ex.register_fields([('srq_context', ctypes.c_void_p, 0), ('attr', struct_ibv_srq_attr, 8), ('comp_mask', uint32_t, 20), ('srq_type', ctypes.c_uint32, 24), ('pd', c.POINTER[struct_ibv_pd], 32), ('xrcd', c.POINTER[struct_ibv_xrcd], 40), ('cq', c.POINTER[struct_ibv_cq], 48), ('tm_cap', struct_ibv_tm_cap, 56)])
enum_ibv_wq_init_attr_mask: dict[int, str] = {(IBV_WQ_INIT_ATTR_FLAGS:=1): 'IBV_WQ_INIT_ATTR_FLAGS', (IBV_WQ_INIT_ATTR_RESERVED:=2): 'IBV_WQ_INIT_ATTR_RESERVED'}
enum_ibv_wq_flags: dict[int, str] = {(IBV_WQ_FLAGS_CVLAN_STRIPPING:=1): 'IBV_WQ_FLAGS_CVLAN_STRIPPING', (IBV_WQ_FLAGS_SCATTER_FCS:=2): 'IBV_WQ_FLAGS_SCATTER_FCS', (IBV_WQ_FLAGS_DELAY_DROP:=4): 'IBV_WQ_FLAGS_DELAY_DROP', (IBV_WQ_FLAGS_PCI_WRITE_END_PADDING:=8): 'IBV_WQ_FLAGS_PCI_WRITE_END_PADDING', (IBV_WQ_FLAGS_RESERVED:=16): 'IBV_WQ_FLAGS_RESERVED'}
@c.record
class struct_ibv_wq_init_attr(c.Struct):
  SIZE = 48
  wq_context: ctypes.c_void_p
  wq_type: int
  max_wr: int
  max_sge: int
  pd: c.POINTER[struct_ibv_pd]
  cq: c.POINTER[struct_ibv_cq]
  comp_mask: int
  create_flags: int
struct_ibv_wq_init_attr.register_fields([('wq_context', ctypes.c_void_p, 0), ('wq_type', ctypes.c_uint32, 8), ('max_wr', uint32_t, 12), ('max_sge', uint32_t, 16), ('pd', c.POINTER[struct_ibv_pd], 24), ('cq', c.POINTER[struct_ibv_cq], 32), ('comp_mask', uint32_t, 40), ('create_flags', uint32_t, 44)])
enum_ibv_wq_attr_mask: dict[int, str] = {(IBV_WQ_ATTR_STATE:=1): 'IBV_WQ_ATTR_STATE', (IBV_WQ_ATTR_CURR_STATE:=2): 'IBV_WQ_ATTR_CURR_STATE', (IBV_WQ_ATTR_FLAGS:=4): 'IBV_WQ_ATTR_FLAGS', (IBV_WQ_ATTR_RESERVED:=8): 'IBV_WQ_ATTR_RESERVED'}
@c.record
class struct_ibv_wq_attr(c.Struct):
  SIZE = 20
  attr_mask: int
  wq_state: int
  curr_wq_state: int
  flags: int
  flags_mask: int
struct_ibv_wq_attr.register_fields([('attr_mask', uint32_t, 0), ('wq_state', ctypes.c_uint32, 4), ('curr_wq_state', ctypes.c_uint32, 8), ('flags', uint32_t, 12), ('flags_mask', uint32_t, 16)])
@c.record
class struct_ibv_rwq_ind_table(c.Struct):
  SIZE = 24
  context: c.POINTER[struct_ibv_context]
  ind_tbl_handle: int
  ind_tbl_num: int
  comp_mask: int
struct_ibv_rwq_ind_table.register_fields([('context', c.POINTER[struct_ibv_context], 0), ('ind_tbl_handle', ctypes.c_int32, 8), ('ind_tbl_num', ctypes.c_int32, 12), ('comp_mask', uint32_t, 16)])
enum_ibv_ind_table_init_attr_mask: dict[int, str] = {(IBV_CREATE_IND_TABLE_RESERVED:=1): 'IBV_CREATE_IND_TABLE_RESERVED'}
@c.record
class struct_ibv_rwq_ind_table_init_attr(c.Struct):
  SIZE = 24
  log_ind_tbl_size: int
  ind_tbl: c.POINTER[c.POINTER[struct_ibv_wq]]
  comp_mask: int
struct_ibv_rwq_ind_table_init_attr.register_fields([('log_ind_tbl_size', uint32_t, 0), ('ind_tbl', c.POINTER[c.POINTER[struct_ibv_wq]], 8), ('comp_mask', uint32_t, 16)])
@c.record
class struct_ibv_qp_cap(c.Struct):
  SIZE = 20
  max_send_wr: int
  max_recv_wr: int
  max_send_sge: int
  max_recv_sge: int
  max_inline_data: int
struct_ibv_qp_cap.register_fields([('max_send_wr', uint32_t, 0), ('max_recv_wr', uint32_t, 4), ('max_send_sge', uint32_t, 8), ('max_recv_sge', uint32_t, 12), ('max_inline_data', uint32_t, 16)])
@c.record
class struct_ibv_qp_init_attr(c.Struct):
  SIZE = 64
  qp_context: ctypes.c_void_p
  send_cq: c.POINTER[struct_ibv_cq]
  recv_cq: c.POINTER[struct_ibv_cq]
  srq: c.POINTER[struct_ibv_srq]
  cap: struct_ibv_qp_cap
  qp_type: int
  sq_sig_all: int
struct_ibv_qp_init_attr.register_fields([('qp_context', ctypes.c_void_p, 0), ('send_cq', c.POINTER[struct_ibv_cq], 8), ('recv_cq', c.POINTER[struct_ibv_cq], 16), ('srq', c.POINTER[struct_ibv_srq], 24), ('cap', struct_ibv_qp_cap, 32), ('qp_type', ctypes.c_uint32, 52), ('sq_sig_all', ctypes.c_int32, 56)])
enum_ibv_qp_init_attr_mask: dict[int, str] = {(IBV_QP_INIT_ATTR_PD:=1): 'IBV_QP_INIT_ATTR_PD', (IBV_QP_INIT_ATTR_XRCD:=2): 'IBV_QP_INIT_ATTR_XRCD', (IBV_QP_INIT_ATTR_CREATE_FLAGS:=4): 'IBV_QP_INIT_ATTR_CREATE_FLAGS', (IBV_QP_INIT_ATTR_MAX_TSO_HEADER:=8): 'IBV_QP_INIT_ATTR_MAX_TSO_HEADER', (IBV_QP_INIT_ATTR_IND_TABLE:=16): 'IBV_QP_INIT_ATTR_IND_TABLE', (IBV_QP_INIT_ATTR_RX_HASH:=32): 'IBV_QP_INIT_ATTR_RX_HASH', (IBV_QP_INIT_ATTR_SEND_OPS_FLAGS:=64): 'IBV_QP_INIT_ATTR_SEND_OPS_FLAGS'}
enum_ibv_qp_create_flags: dict[int, str] = {(IBV_QP_CREATE_BLOCK_SELF_MCAST_LB:=2): 'IBV_QP_CREATE_BLOCK_SELF_MCAST_LB', (IBV_QP_CREATE_SCATTER_FCS:=256): 'IBV_QP_CREATE_SCATTER_FCS', (IBV_QP_CREATE_CVLAN_STRIPPING:=512): 'IBV_QP_CREATE_CVLAN_STRIPPING', (IBV_QP_CREATE_SOURCE_QPN:=1024): 'IBV_QP_CREATE_SOURCE_QPN', (IBV_QP_CREATE_PCI_WRITE_END_PADDING:=2048): 'IBV_QP_CREATE_PCI_WRITE_END_PADDING'}
enum_ibv_qp_create_send_ops_flags: dict[int, str] = {(IBV_QP_EX_WITH_RDMA_WRITE:=1): 'IBV_QP_EX_WITH_RDMA_WRITE', (IBV_QP_EX_WITH_RDMA_WRITE_WITH_IMM:=2): 'IBV_QP_EX_WITH_RDMA_WRITE_WITH_IMM', (IBV_QP_EX_WITH_SEND:=4): 'IBV_QP_EX_WITH_SEND', (IBV_QP_EX_WITH_SEND_WITH_IMM:=8): 'IBV_QP_EX_WITH_SEND_WITH_IMM', (IBV_QP_EX_WITH_RDMA_READ:=16): 'IBV_QP_EX_WITH_RDMA_READ', (IBV_QP_EX_WITH_ATOMIC_CMP_AND_SWP:=32): 'IBV_QP_EX_WITH_ATOMIC_CMP_AND_SWP', (IBV_QP_EX_WITH_ATOMIC_FETCH_AND_ADD:=64): 'IBV_QP_EX_WITH_ATOMIC_FETCH_AND_ADD', (IBV_QP_EX_WITH_LOCAL_INV:=128): 'IBV_QP_EX_WITH_LOCAL_INV', (IBV_QP_EX_WITH_BIND_MW:=256): 'IBV_QP_EX_WITH_BIND_MW', (IBV_QP_EX_WITH_SEND_WITH_INV:=512): 'IBV_QP_EX_WITH_SEND_WITH_INV', (IBV_QP_EX_WITH_TSO:=1024): 'IBV_QP_EX_WITH_TSO', (IBV_QP_EX_WITH_FLUSH:=2048): 'IBV_QP_EX_WITH_FLUSH', (IBV_QP_EX_WITH_ATOMIC_WRITE:=4096): 'IBV_QP_EX_WITH_ATOMIC_WRITE'}
@c.record
class struct_ibv_rx_hash_conf(c.Struct):
  SIZE = 24
  rx_hash_function: int
  rx_hash_key_len: int
  rx_hash_key: c.POINTER[ctypes.c_ubyte]
  rx_hash_fields_mask: int
struct_ibv_rx_hash_conf.register_fields([('rx_hash_function', uint8_t, 0), ('rx_hash_key_len', uint8_t, 1), ('rx_hash_key', c.POINTER[uint8_t], 8), ('rx_hash_fields_mask', uint64_t, 16)])
@c.record
class struct_ibv_qp_init_attr_ex(c.Struct):
  SIZE = 136
  qp_context: ctypes.c_void_p
  send_cq: c.POINTER[struct_ibv_cq]
  recv_cq: c.POINTER[struct_ibv_cq]
  srq: c.POINTER[struct_ibv_srq]
  cap: struct_ibv_qp_cap
  qp_type: int
  sq_sig_all: int
  comp_mask: int
  pd: c.POINTER[struct_ibv_pd]
  xrcd: c.POINTER[struct_ibv_xrcd]
  create_flags: int
  max_tso_header: int
  rwq_ind_tbl: c.POINTER[struct_ibv_rwq_ind_table]
  rx_hash_conf: struct_ibv_rx_hash_conf
  source_qpn: int
  send_ops_flags: int
struct_ibv_qp_init_attr_ex.register_fields([('qp_context', ctypes.c_void_p, 0), ('send_cq', c.POINTER[struct_ibv_cq], 8), ('recv_cq', c.POINTER[struct_ibv_cq], 16), ('srq', c.POINTER[struct_ibv_srq], 24), ('cap', struct_ibv_qp_cap, 32), ('qp_type', ctypes.c_uint32, 52), ('sq_sig_all', ctypes.c_int32, 56), ('comp_mask', uint32_t, 60), ('pd', c.POINTER[struct_ibv_pd], 64), ('xrcd', c.POINTER[struct_ibv_xrcd], 72), ('create_flags', uint32_t, 80), ('max_tso_header', uint16_t, 84), ('rwq_ind_tbl', c.POINTER[struct_ibv_rwq_ind_table], 88), ('rx_hash_conf', struct_ibv_rx_hash_conf, 96), ('source_qpn', uint32_t, 120), ('send_ops_flags', uint64_t, 128)])
enum_ibv_qp_open_attr_mask: dict[int, str] = {(IBV_QP_OPEN_ATTR_NUM:=1): 'IBV_QP_OPEN_ATTR_NUM', (IBV_QP_OPEN_ATTR_XRCD:=2): 'IBV_QP_OPEN_ATTR_XRCD', (IBV_QP_OPEN_ATTR_CONTEXT:=4): 'IBV_QP_OPEN_ATTR_CONTEXT', (IBV_QP_OPEN_ATTR_TYPE:=8): 'IBV_QP_OPEN_ATTR_TYPE', (IBV_QP_OPEN_ATTR_RESERVED:=16): 'IBV_QP_OPEN_ATTR_RESERVED'}
@c.record
class struct_ibv_qp_open_attr(c.Struct):
  SIZE = 32
  comp_mask: int
  qp_num: int
  xrcd: c.POINTER[struct_ibv_xrcd]
  qp_context: ctypes.c_void_p
  qp_type: int
struct_ibv_qp_open_attr.register_fields([('comp_mask', uint32_t, 0), ('qp_num', uint32_t, 4), ('xrcd', c.POINTER[struct_ibv_xrcd], 8), ('qp_context', ctypes.c_void_p, 16), ('qp_type', ctypes.c_uint32, 24)])
enum_ibv_qp_attr_mask: dict[int, str] = {(IBV_QP_STATE:=1): 'IBV_QP_STATE', (IBV_QP_CUR_STATE:=2): 'IBV_QP_CUR_STATE', (IBV_QP_EN_SQD_ASYNC_NOTIFY:=4): 'IBV_QP_EN_SQD_ASYNC_NOTIFY', (IBV_QP_ACCESS_FLAGS:=8): 'IBV_QP_ACCESS_FLAGS', (IBV_QP_PKEY_INDEX:=16): 'IBV_QP_PKEY_INDEX', (IBV_QP_PORT:=32): 'IBV_QP_PORT', (IBV_QP_QKEY:=64): 'IBV_QP_QKEY', (IBV_QP_AV:=128): 'IBV_QP_AV', (IBV_QP_PATH_MTU:=256): 'IBV_QP_PATH_MTU', (IBV_QP_TIMEOUT:=512): 'IBV_QP_TIMEOUT', (IBV_QP_RETRY_CNT:=1024): 'IBV_QP_RETRY_CNT', (IBV_QP_RNR_RETRY:=2048): 'IBV_QP_RNR_RETRY', (IBV_QP_RQ_PSN:=4096): 'IBV_QP_RQ_PSN', (IBV_QP_MAX_QP_RD_ATOMIC:=8192): 'IBV_QP_MAX_QP_RD_ATOMIC', (IBV_QP_ALT_PATH:=16384): 'IBV_QP_ALT_PATH', (IBV_QP_MIN_RNR_TIMER:=32768): 'IBV_QP_MIN_RNR_TIMER', (IBV_QP_SQ_PSN:=65536): 'IBV_QP_SQ_PSN', (IBV_QP_MAX_DEST_RD_ATOMIC:=131072): 'IBV_QP_MAX_DEST_RD_ATOMIC', (IBV_QP_PATH_MIG_STATE:=262144): 'IBV_QP_PATH_MIG_STATE', (IBV_QP_CAP:=524288): 'IBV_QP_CAP', (IBV_QP_DEST_QPN:=1048576): 'IBV_QP_DEST_QPN', (IBV_QP_RATE_LIMIT:=33554432): 'IBV_QP_RATE_LIMIT'}
enum_ibv_query_qp_data_in_order_flags: dict[int, str] = {(IBV_QUERY_QP_DATA_IN_ORDER_RETURN_CAPS:=1): 'IBV_QUERY_QP_DATA_IN_ORDER_RETURN_CAPS'}
enum_ibv_query_qp_data_in_order_caps: dict[int, str] = {(IBV_QUERY_QP_DATA_IN_ORDER_WHOLE_MSG:=1): 'IBV_QUERY_QP_DATA_IN_ORDER_WHOLE_MSG', (IBV_QUERY_QP_DATA_IN_ORDER_ALIGNED_128_BYTES:=2): 'IBV_QUERY_QP_DATA_IN_ORDER_ALIGNED_128_BYTES'}
enum_ibv_mig_state: dict[int, str] = {(IBV_MIG_MIGRATED:=0): 'IBV_MIG_MIGRATED', (IBV_MIG_REARM:=1): 'IBV_MIG_REARM', (IBV_MIG_ARMED:=2): 'IBV_MIG_ARMED'}
@c.record
class struct_ibv_qp_attr(c.Struct):
  SIZE = 144
  qp_state: int
  cur_qp_state: int
  path_mtu: int
  path_mig_state: int
  qkey: int
  rq_psn: int
  sq_psn: int
  dest_qp_num: int
  qp_access_flags: int
  cap: struct_ibv_qp_cap
  ah_attr: struct_ibv_ah_attr
  alt_ah_attr: struct_ibv_ah_attr
  pkey_index: int
  alt_pkey_index: int
  en_sqd_async_notify: int
  sq_draining: int
  max_rd_atomic: int
  max_dest_rd_atomic: int
  min_rnr_timer: int
  port_num: int
  timeout: int
  retry_cnt: int
  rnr_retry: int
  alt_port_num: int
  alt_timeout: int
  rate_limit: int
struct_ibv_qp_attr.register_fields([('qp_state', ctypes.c_uint32, 0), ('cur_qp_state', ctypes.c_uint32, 4), ('path_mtu', ctypes.c_uint32, 8), ('path_mig_state', ctypes.c_uint32, 12), ('qkey', uint32_t, 16), ('rq_psn', uint32_t, 20), ('sq_psn', uint32_t, 24), ('dest_qp_num', uint32_t, 28), ('qp_access_flags', ctypes.c_uint32, 32), ('cap', struct_ibv_qp_cap, 36), ('ah_attr', struct_ibv_ah_attr, 56), ('alt_ah_attr', struct_ibv_ah_attr, 88), ('pkey_index', uint16_t, 120), ('alt_pkey_index', uint16_t, 122), ('en_sqd_async_notify', uint8_t, 124), ('sq_draining', uint8_t, 125), ('max_rd_atomic', uint8_t, 126), ('max_dest_rd_atomic', uint8_t, 127), ('min_rnr_timer', uint8_t, 128), ('port_num', uint8_t, 129), ('timeout', uint8_t, 130), ('retry_cnt', uint8_t, 131), ('rnr_retry', uint8_t, 132), ('alt_port_num', uint8_t, 133), ('alt_timeout', uint8_t, 134), ('rate_limit', uint32_t, 136)])
@c.record
class struct_ibv_qp_rate_limit_attr(c.Struct):
  SIZE = 16
  rate_limit: int
  max_burst_sz: int
  typical_pkt_sz: int
  comp_mask: int
struct_ibv_qp_rate_limit_attr.register_fields([('rate_limit', uint32_t, 0), ('max_burst_sz', uint32_t, 4), ('typical_pkt_sz', uint16_t, 8), ('comp_mask', uint32_t, 12)])
@dll.bind(c.POINTER[ctypes.c_char], ctypes.c_uint32)
def ibv_wr_opcode_str(opcode:ctypes.c_uint32) -> c.POINTER[ctypes.c_char]: ...
enum_ibv_send_flags: dict[int, str] = {(IBV_SEND_FENCE:=1): 'IBV_SEND_FENCE', (IBV_SEND_SIGNALED:=2): 'IBV_SEND_SIGNALED', (IBV_SEND_SOLICITED:=4): 'IBV_SEND_SOLICITED', (IBV_SEND_INLINE:=8): 'IBV_SEND_INLINE', (IBV_SEND_IP_CSUM:=16): 'IBV_SEND_IP_CSUM'}
enum_ibv_placement_type: dict[int, str] = {(IBV_FLUSH_GLOBAL:=1): 'IBV_FLUSH_GLOBAL', (IBV_FLUSH_PERSISTENT:=2): 'IBV_FLUSH_PERSISTENT'}
enum_ibv_selectivity_level: dict[int, str] = {(IBV_FLUSH_RANGE:=0): 'IBV_FLUSH_RANGE', (IBV_FLUSH_MR:=1): 'IBV_FLUSH_MR'}
@c.record
class struct_ibv_data_buf(c.Struct):
  SIZE = 16
  addr: ctypes.c_void_p
  length: int
struct_ibv_data_buf.register_fields([('addr', ctypes.c_void_p, 0), ('length', size_t, 8)])
enum_ibv_ops_wr_opcode: dict[int, str] = {(IBV_WR_TAG_ADD:=0): 'IBV_WR_TAG_ADD', (IBV_WR_TAG_DEL:=1): 'IBV_WR_TAG_DEL', (IBV_WR_TAG_SYNC:=2): 'IBV_WR_TAG_SYNC'}
enum_ibv_ops_flags: dict[int, str] = {(IBV_OPS_SIGNALED:=1): 'IBV_OPS_SIGNALED', (IBV_OPS_TM_SYNC:=2): 'IBV_OPS_TM_SYNC'}
@c.record
class struct_ibv_ops_wr(c.Struct):
  SIZE = 72
  wr_id: int
  next: c.POINTER[struct_ibv_ops_wr]
  opcode: int
  flags: int
  tm: struct_ibv_ops_wr_tm
@c.record
class struct_ibv_ops_wr_tm(c.Struct):
  SIZE = 48
  unexpected_cnt: int
  handle: int
  add: struct_ibv_ops_wr_tm_add
@c.record
class struct_ibv_ops_wr_tm_add(c.Struct):
  SIZE = 40
  recv_wr_id: int
  sg_list: c.POINTER[struct_ibv_sge]
  num_sge: int
  tag: int
  mask: int
struct_ibv_ops_wr_tm_add.register_fields([('recv_wr_id', uint64_t, 0), ('sg_list', c.POINTER[struct_ibv_sge], 8), ('num_sge', ctypes.c_int32, 16), ('tag', uint64_t, 24), ('mask', uint64_t, 32)])
struct_ibv_ops_wr_tm.register_fields([('unexpected_cnt', uint32_t, 0), ('handle', uint32_t, 4), ('add', struct_ibv_ops_wr_tm_add, 8)])
struct_ibv_ops_wr.register_fields([('wr_id', uint64_t, 0), ('next', c.POINTER[struct_ibv_ops_wr], 8), ('opcode', ctypes.c_uint32, 16), ('flags', ctypes.c_int32, 20), ('tm', struct_ibv_ops_wr_tm, 24)])
@c.record
class struct_ibv_qp_ex(c.Struct):
  SIZE = 360
  qp_base: struct_ibv_qp
  comp_mask: int
  wr_id: int
  wr_flags: int
  wr_atomic_cmp_swp: c.CFUNCTYPE[None, [c.POINTER[struct_ibv_qp_ex], ctypes.c_uint32, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]]
  wr_atomic_fetch_add: c.CFUNCTYPE[None, [c.POINTER[struct_ibv_qp_ex], ctypes.c_uint32, ctypes.c_uint64, ctypes.c_uint64]]
  wr_bind_mw: c.CFUNCTYPE[None, [c.POINTER[struct_ibv_qp_ex], c.POINTER[struct_ibv_mw], ctypes.c_uint32, c.POINTER[struct_ibv_mw_bind_info]]]
  wr_local_inv: c.CFUNCTYPE[None, [c.POINTER[struct_ibv_qp_ex], ctypes.c_uint32]]
  wr_rdma_read: c.CFUNCTYPE[None, [c.POINTER[struct_ibv_qp_ex], ctypes.c_uint32, ctypes.c_uint64]]
  wr_rdma_write: c.CFUNCTYPE[None, [c.POINTER[struct_ibv_qp_ex], ctypes.c_uint32, ctypes.c_uint64]]
  wr_rdma_write_imm: c.CFUNCTYPE[None, [c.POINTER[struct_ibv_qp_ex], ctypes.c_uint32, ctypes.c_uint64, ctypes.c_uint32]]
  wr_send: c.CFUNCTYPE[None, [c.POINTER[struct_ibv_qp_ex]]]
  wr_send_imm: c.CFUNCTYPE[None, [c.POINTER[struct_ibv_qp_ex], ctypes.c_uint32]]
  wr_send_inv: c.CFUNCTYPE[None, [c.POINTER[struct_ibv_qp_ex], ctypes.c_uint32]]
  wr_send_tso: c.CFUNCTYPE[None, [c.POINTER[struct_ibv_qp_ex], ctypes.c_void_p, ctypes.c_uint16, ctypes.c_uint16]]
  wr_set_ud_addr: c.CFUNCTYPE[None, [c.POINTER[struct_ibv_qp_ex], c.POINTER[struct_ibv_ah], ctypes.c_uint32, ctypes.c_uint32]]
  wr_set_xrc_srqn: c.CFUNCTYPE[None, [c.POINTER[struct_ibv_qp_ex], ctypes.c_uint32]]
  wr_set_inline_data: c.CFUNCTYPE[None, [c.POINTER[struct_ibv_qp_ex], ctypes.c_void_p, ctypes.c_uint64]]
  wr_set_inline_data_list: c.CFUNCTYPE[None, [c.POINTER[struct_ibv_qp_ex], ctypes.c_uint64, c.POINTER[struct_ibv_data_buf]]]
  wr_set_sge: c.CFUNCTYPE[None, [c.POINTER[struct_ibv_qp_ex], ctypes.c_uint32, ctypes.c_uint64, ctypes.c_uint32]]
  wr_set_sge_list: c.CFUNCTYPE[None, [c.POINTER[struct_ibv_qp_ex], ctypes.c_uint64, c.POINTER[struct_ibv_sge]]]
  wr_start: c.CFUNCTYPE[None, [c.POINTER[struct_ibv_qp_ex]]]
  wr_complete: c.CFUNCTYPE[ctypes.c_int32, [c.POINTER[struct_ibv_qp_ex]]]
  wr_abort: c.CFUNCTYPE[None, [c.POINTER[struct_ibv_qp_ex]]]
  wr_atomic_write: c.CFUNCTYPE[None, [c.POINTER[struct_ibv_qp_ex], ctypes.c_uint32, ctypes.c_uint64, ctypes.c_void_p]]
  wr_flush: c.CFUNCTYPE[None, [c.POINTER[struct_ibv_qp_ex], ctypes.c_uint32, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_ubyte, ctypes.c_ubyte]]
struct_ibv_qp_ex.register_fields([('qp_base', struct_ibv_qp, 0), ('comp_mask', uint64_t, 160), ('wr_id', uint64_t, 168), ('wr_flags', ctypes.c_uint32, 176), ('wr_atomic_cmp_swp', c.CFUNCTYPE[None, [c.POINTER[struct_ibv_qp_ex], uint32_t, uint64_t, uint64_t, uint64_t]], 184), ('wr_atomic_fetch_add', c.CFUNCTYPE[None, [c.POINTER[struct_ibv_qp_ex], uint32_t, uint64_t, uint64_t]], 192), ('wr_bind_mw', c.CFUNCTYPE[None, [c.POINTER[struct_ibv_qp_ex], c.POINTER[struct_ibv_mw], uint32_t, c.POINTER[struct_ibv_mw_bind_info]]], 200), ('wr_local_inv', c.CFUNCTYPE[None, [c.POINTER[struct_ibv_qp_ex], uint32_t]], 208), ('wr_rdma_read', c.CFUNCTYPE[None, [c.POINTER[struct_ibv_qp_ex], uint32_t, uint64_t]], 216), ('wr_rdma_write', c.CFUNCTYPE[None, [c.POINTER[struct_ibv_qp_ex], uint32_t, uint64_t]], 224), ('wr_rdma_write_imm', c.CFUNCTYPE[None, [c.POINTER[struct_ibv_qp_ex], uint32_t, uint64_t, ctypes.c_uint32]], 232), ('wr_send', c.CFUNCTYPE[None, [c.POINTER[struct_ibv_qp_ex]]], 240), ('wr_send_imm', c.CFUNCTYPE[None, [c.POINTER[struct_ibv_qp_ex], ctypes.c_uint32]], 248), ('wr_send_inv', c.CFUNCTYPE[None, [c.POINTER[struct_ibv_qp_ex], uint32_t]], 256), ('wr_send_tso', c.CFUNCTYPE[None, [c.POINTER[struct_ibv_qp_ex], ctypes.c_void_p, uint16_t, uint16_t]], 264), ('wr_set_ud_addr', c.CFUNCTYPE[None, [c.POINTER[struct_ibv_qp_ex], c.POINTER[struct_ibv_ah], uint32_t, uint32_t]], 272), ('wr_set_xrc_srqn', c.CFUNCTYPE[None, [c.POINTER[struct_ibv_qp_ex], uint32_t]], 280), ('wr_set_inline_data', c.CFUNCTYPE[None, [c.POINTER[struct_ibv_qp_ex], ctypes.c_void_p, size_t]], 288), ('wr_set_inline_data_list', c.CFUNCTYPE[None, [c.POINTER[struct_ibv_qp_ex], size_t, c.POINTER[struct_ibv_data_buf]]], 296), ('wr_set_sge', c.CFUNCTYPE[None, [c.POINTER[struct_ibv_qp_ex], uint32_t, uint64_t, uint32_t]], 304), ('wr_set_sge_list', c.CFUNCTYPE[None, [c.POINTER[struct_ibv_qp_ex], size_t, c.POINTER[struct_ibv_sge]]], 312), ('wr_start', c.CFUNCTYPE[None, [c.POINTER[struct_ibv_qp_ex]]], 320), ('wr_complete', c.CFUNCTYPE[ctypes.c_int32, [c.POINTER[struct_ibv_qp_ex]]], 328), ('wr_abort', c.CFUNCTYPE[None, [c.POINTER[struct_ibv_qp_ex]]], 336), ('wr_atomic_write', c.CFUNCTYPE[None, [c.POINTER[struct_ibv_qp_ex], uint32_t, uint64_t, ctypes.c_void_p]], 344), ('wr_flush', c.CFUNCTYPE[None, [c.POINTER[struct_ibv_qp_ex], uint32_t, uint64_t, size_t, uint8_t, uint8_t]], 352)])
@dll.bind(c.POINTER[struct_ibv_qp_ex], c.POINTER[struct_ibv_qp])
def ibv_qp_to_qp_ex(qp:c.POINTER[struct_ibv_qp]) -> c.POINTER[struct_ibv_qp_ex]: ...
@c.record
class struct_ibv_ece(c.Struct):
  SIZE = 12
  vendor_id: int
  options: int
  comp_mask: int
struct_ibv_ece.register_fields([('vendor_id', uint32_t, 0), ('options', uint32_t, 4), ('comp_mask', uint32_t, 8)])
@c.record
class struct_ibv_poll_cq_attr(c.Struct):
  SIZE = 4
  comp_mask: int
struct_ibv_poll_cq_attr.register_fields([('comp_mask', uint32_t, 0)])
@c.record
class struct_ibv_wc_tm_info(c.Struct):
  SIZE = 16
  tag: int
  priv: int
struct_ibv_wc_tm_info.register_fields([('tag', uint64_t, 0), ('priv', uint32_t, 8)])
@c.record
class struct_ibv_cq_ex(c.Struct):
  SIZE = 288
  context: c.POINTER[struct_ibv_context]
  channel: c.POINTER[struct_ibv_comp_channel]
  cq_context: ctypes.c_void_p
  handle: int
  cqe: int
  mutex: pthread_mutex_t
  cond: pthread_cond_t
  comp_events_completed: int
  async_events_completed: int
  comp_mask: int
  status: int
  wr_id: int
  start_poll: c.CFUNCTYPE[ctypes.c_int32, [c.POINTER[struct_ibv_cq_ex], c.POINTER[struct_ibv_poll_cq_attr]]]
  next_poll: c.CFUNCTYPE[ctypes.c_int32, [c.POINTER[struct_ibv_cq_ex]]]
  end_poll: c.CFUNCTYPE[None, [c.POINTER[struct_ibv_cq_ex]]]
  read_opcode: c.CFUNCTYPE[ctypes.c_uint32, [c.POINTER[struct_ibv_cq_ex]]]
  read_vendor_err: c.CFUNCTYPE[ctypes.c_uint32, [c.POINTER[struct_ibv_cq_ex]]]
  read_byte_len: c.CFUNCTYPE[ctypes.c_uint32, [c.POINTER[struct_ibv_cq_ex]]]
  read_imm_data: c.CFUNCTYPE[ctypes.c_uint32, [c.POINTER[struct_ibv_cq_ex]]]
  read_qp_num: c.CFUNCTYPE[ctypes.c_uint32, [c.POINTER[struct_ibv_cq_ex]]]
  read_src_qp: c.CFUNCTYPE[ctypes.c_uint32, [c.POINTER[struct_ibv_cq_ex]]]
  read_wc_flags: c.CFUNCTYPE[ctypes.c_uint32, [c.POINTER[struct_ibv_cq_ex]]]
  read_slid: c.CFUNCTYPE[ctypes.c_uint32, [c.POINTER[struct_ibv_cq_ex]]]
  read_sl: c.CFUNCTYPE[ctypes.c_ubyte, [c.POINTER[struct_ibv_cq_ex]]]
  read_dlid_path_bits: c.CFUNCTYPE[ctypes.c_ubyte, [c.POINTER[struct_ibv_cq_ex]]]
  read_completion_ts: c.CFUNCTYPE[ctypes.c_uint64, [c.POINTER[struct_ibv_cq_ex]]]
  read_cvlan: c.CFUNCTYPE[ctypes.c_uint16, [c.POINTER[struct_ibv_cq_ex]]]
  read_flow_tag: c.CFUNCTYPE[ctypes.c_uint32, [c.POINTER[struct_ibv_cq_ex]]]
  read_tm_info: c.CFUNCTYPE[None, [c.POINTER[struct_ibv_cq_ex], c.POINTER[struct_ibv_wc_tm_info]]]
  read_completion_wallclock_ns: c.CFUNCTYPE[ctypes.c_uint64, [c.POINTER[struct_ibv_cq_ex]]]
struct_ibv_cq_ex.register_fields([('context', c.POINTER[struct_ibv_context], 0), ('channel', c.POINTER[struct_ibv_comp_channel], 8), ('cq_context', ctypes.c_void_p, 16), ('handle', uint32_t, 24), ('cqe', ctypes.c_int32, 28), ('mutex', pthread_mutex_t, 32), ('cond', pthread_cond_t, 72), ('comp_events_completed', uint32_t, 120), ('async_events_completed', uint32_t, 124), ('comp_mask', uint32_t, 128), ('status', ctypes.c_uint32, 132), ('wr_id', uint64_t, 136), ('start_poll', c.CFUNCTYPE[ctypes.c_int32, [c.POINTER[struct_ibv_cq_ex], c.POINTER[struct_ibv_poll_cq_attr]]], 144), ('next_poll', c.CFUNCTYPE[ctypes.c_int32, [c.POINTER[struct_ibv_cq_ex]]], 152), ('end_poll', c.CFUNCTYPE[None, [c.POINTER[struct_ibv_cq_ex]]], 160), ('read_opcode', c.CFUNCTYPE[ctypes.c_uint32, [c.POINTER[struct_ibv_cq_ex]]], 168), ('read_vendor_err', c.CFUNCTYPE[uint32_t, [c.POINTER[struct_ibv_cq_ex]]], 176), ('read_byte_len', c.CFUNCTYPE[uint32_t, [c.POINTER[struct_ibv_cq_ex]]], 184), ('read_imm_data', c.CFUNCTYPE[ctypes.c_uint32, [c.POINTER[struct_ibv_cq_ex]]], 192), ('read_qp_num', c.CFUNCTYPE[uint32_t, [c.POINTER[struct_ibv_cq_ex]]], 200), ('read_src_qp', c.CFUNCTYPE[uint32_t, [c.POINTER[struct_ibv_cq_ex]]], 208), ('read_wc_flags', c.CFUNCTYPE[ctypes.c_uint32, [c.POINTER[struct_ibv_cq_ex]]], 216), ('read_slid', c.CFUNCTYPE[uint32_t, [c.POINTER[struct_ibv_cq_ex]]], 224), ('read_sl', c.CFUNCTYPE[uint8_t, [c.POINTER[struct_ibv_cq_ex]]], 232), ('read_dlid_path_bits', c.CFUNCTYPE[uint8_t, [c.POINTER[struct_ibv_cq_ex]]], 240), ('read_completion_ts', c.CFUNCTYPE[uint64_t, [c.POINTER[struct_ibv_cq_ex]]], 248), ('read_cvlan', c.CFUNCTYPE[uint16_t, [c.POINTER[struct_ibv_cq_ex]]], 256), ('read_flow_tag', c.CFUNCTYPE[uint32_t, [c.POINTER[struct_ibv_cq_ex]]], 264), ('read_tm_info', c.CFUNCTYPE[None, [c.POINTER[struct_ibv_cq_ex], c.POINTER[struct_ibv_wc_tm_info]]], 272), ('read_completion_wallclock_ns', c.CFUNCTYPE[uint64_t, [c.POINTER[struct_ibv_cq_ex]]], 280)])
enum_ibv_cq_attr_mask: dict[int, str] = {(IBV_CQ_ATTR_MODERATE:=1): 'IBV_CQ_ATTR_MODERATE', (IBV_CQ_ATTR_RESERVED:=2): 'IBV_CQ_ATTR_RESERVED'}
@c.record
class struct_ibv_moderate_cq(c.Struct):
  SIZE = 4
  cq_count: int
  cq_period: int
struct_ibv_moderate_cq.register_fields([('cq_count', uint16_t, 0), ('cq_period', uint16_t, 2)])
@c.record
class struct_ibv_modify_cq_attr(c.Struct):
  SIZE = 8
  attr_mask: int
  moderate: struct_ibv_moderate_cq
struct_ibv_modify_cq_attr.register_fields([('attr_mask', uint32_t, 0), ('moderate', struct_ibv_moderate_cq, 4)])
enum_ibv_flow_flags: dict[int, str] = {(IBV_FLOW_ATTR_FLAGS_DONT_TRAP:=2): 'IBV_FLOW_ATTR_FLAGS_DONT_TRAP', (IBV_FLOW_ATTR_FLAGS_EGRESS:=4): 'IBV_FLOW_ATTR_FLAGS_EGRESS'}
enum_ibv_flow_attr_type: dict[int, str] = {(IBV_FLOW_ATTR_NORMAL:=0): 'IBV_FLOW_ATTR_NORMAL', (IBV_FLOW_ATTR_ALL_DEFAULT:=1): 'IBV_FLOW_ATTR_ALL_DEFAULT', (IBV_FLOW_ATTR_MC_DEFAULT:=2): 'IBV_FLOW_ATTR_MC_DEFAULT', (IBV_FLOW_ATTR_SNIFFER:=3): 'IBV_FLOW_ATTR_SNIFFER'}
enum_ibv_flow_spec_type: dict[int, str] = {(IBV_FLOW_SPEC_ETH:=32): 'IBV_FLOW_SPEC_ETH', (IBV_FLOW_SPEC_IPV4:=48): 'IBV_FLOW_SPEC_IPV4', (IBV_FLOW_SPEC_IPV6:=49): 'IBV_FLOW_SPEC_IPV6', (IBV_FLOW_SPEC_IPV4_EXT:=50): 'IBV_FLOW_SPEC_IPV4_EXT', (IBV_FLOW_SPEC_ESP:=52): 'IBV_FLOW_SPEC_ESP', (IBV_FLOW_SPEC_TCP:=64): 'IBV_FLOW_SPEC_TCP', (IBV_FLOW_SPEC_UDP:=65): 'IBV_FLOW_SPEC_UDP', (IBV_FLOW_SPEC_VXLAN_TUNNEL:=80): 'IBV_FLOW_SPEC_VXLAN_TUNNEL', (IBV_FLOW_SPEC_GRE:=81): 'IBV_FLOW_SPEC_GRE', (IBV_FLOW_SPEC_MPLS:=96): 'IBV_FLOW_SPEC_MPLS', (IBV_FLOW_SPEC_INNER:=256): 'IBV_FLOW_SPEC_INNER', (IBV_FLOW_SPEC_ACTION_TAG:=4096): 'IBV_FLOW_SPEC_ACTION_TAG', (IBV_FLOW_SPEC_ACTION_DROP:=4097): 'IBV_FLOW_SPEC_ACTION_DROP', (IBV_FLOW_SPEC_ACTION_HANDLE:=4098): 'IBV_FLOW_SPEC_ACTION_HANDLE', (IBV_FLOW_SPEC_ACTION_COUNT:=4099): 'IBV_FLOW_SPEC_ACTION_COUNT'}
@c.record
class struct_ibv_flow_eth_filter(c.Struct):
  SIZE = 16
  dst_mac: c.Array[ctypes.c_ubyte, Literal[6]]
  src_mac: c.Array[ctypes.c_ubyte, Literal[6]]
  ether_type: int
  vlan_tag: int
struct_ibv_flow_eth_filter.register_fields([('dst_mac', c.Array[uint8_t, Literal[6]], 0), ('src_mac', c.Array[uint8_t, Literal[6]], 6), ('ether_type', uint16_t, 12), ('vlan_tag', uint16_t, 14)])
@c.record
class struct_ibv_flow_spec_eth(c.Struct):
  SIZE = 40
  type: int
  size: int
  val: struct_ibv_flow_eth_filter
  mask: struct_ibv_flow_eth_filter
struct_ibv_flow_spec_eth.register_fields([('type', ctypes.c_uint32, 0), ('size', uint16_t, 4), ('val', struct_ibv_flow_eth_filter, 6), ('mask', struct_ibv_flow_eth_filter, 22)])
@c.record
class struct_ibv_flow_ipv4_filter(c.Struct):
  SIZE = 8
  src_ip: int
  dst_ip: int
struct_ibv_flow_ipv4_filter.register_fields([('src_ip', uint32_t, 0), ('dst_ip', uint32_t, 4)])
@c.record
class struct_ibv_flow_spec_ipv4(c.Struct):
  SIZE = 24
  type: int
  size: int
  val: struct_ibv_flow_ipv4_filter
  mask: struct_ibv_flow_ipv4_filter
struct_ibv_flow_spec_ipv4.register_fields([('type', ctypes.c_uint32, 0), ('size', uint16_t, 4), ('val', struct_ibv_flow_ipv4_filter, 8), ('mask', struct_ibv_flow_ipv4_filter, 16)])
@c.record
class struct_ibv_flow_ipv4_ext_filter(c.Struct):
  SIZE = 12
  src_ip: int
  dst_ip: int
  proto: int
  tos: int
  ttl: int
  flags: int
struct_ibv_flow_ipv4_ext_filter.register_fields([('src_ip', uint32_t, 0), ('dst_ip', uint32_t, 4), ('proto', uint8_t, 8), ('tos', uint8_t, 9), ('ttl', uint8_t, 10), ('flags', uint8_t, 11)])
@c.record
class struct_ibv_flow_spec_ipv4_ext(c.Struct):
  SIZE = 32
  type: int
  size: int
  val: struct_ibv_flow_ipv4_ext_filter
  mask: struct_ibv_flow_ipv4_ext_filter
struct_ibv_flow_spec_ipv4_ext.register_fields([('type', ctypes.c_uint32, 0), ('size', uint16_t, 4), ('val', struct_ibv_flow_ipv4_ext_filter, 8), ('mask', struct_ibv_flow_ipv4_ext_filter, 20)])
@c.record
class struct_ibv_flow_ipv6_filter(c.Struct):
  SIZE = 40
  src_ip: c.Array[ctypes.c_ubyte, Literal[16]]
  dst_ip: c.Array[ctypes.c_ubyte, Literal[16]]
  flow_label: int
  next_hdr: int
  traffic_class: int
  hop_limit: int
struct_ibv_flow_ipv6_filter.register_fields([('src_ip', c.Array[uint8_t, Literal[16]], 0), ('dst_ip', c.Array[uint8_t, Literal[16]], 16), ('flow_label', uint32_t, 32), ('next_hdr', uint8_t, 36), ('traffic_class', uint8_t, 37), ('hop_limit', uint8_t, 38)])
@c.record
class struct_ibv_flow_spec_ipv6(c.Struct):
  SIZE = 88
  type: int
  size: int
  val: struct_ibv_flow_ipv6_filter
  mask: struct_ibv_flow_ipv6_filter
struct_ibv_flow_spec_ipv6.register_fields([('type', ctypes.c_uint32, 0), ('size', uint16_t, 4), ('val', struct_ibv_flow_ipv6_filter, 8), ('mask', struct_ibv_flow_ipv6_filter, 48)])
@c.record
class struct_ibv_flow_esp_filter(c.Struct):
  SIZE = 8
  spi: int
  seq: int
struct_ibv_flow_esp_filter.register_fields([('spi', uint32_t, 0), ('seq', uint32_t, 4)])
@c.record
class struct_ibv_flow_spec_esp(c.Struct):
  SIZE = 24
  type: int
  size: int
  val: struct_ibv_flow_esp_filter
  mask: struct_ibv_flow_esp_filter
struct_ibv_flow_spec_esp.register_fields([('type', ctypes.c_uint32, 0), ('size', uint16_t, 4), ('val', struct_ibv_flow_esp_filter, 8), ('mask', struct_ibv_flow_esp_filter, 16)])
@c.record
class struct_ibv_flow_tcp_udp_filter(c.Struct):
  SIZE = 4
  dst_port: int
  src_port: int
struct_ibv_flow_tcp_udp_filter.register_fields([('dst_port', uint16_t, 0), ('src_port', uint16_t, 2)])
@c.record
class struct_ibv_flow_spec_tcp_udp(c.Struct):
  SIZE = 16
  type: int
  size: int
  val: struct_ibv_flow_tcp_udp_filter
  mask: struct_ibv_flow_tcp_udp_filter
struct_ibv_flow_spec_tcp_udp.register_fields([('type', ctypes.c_uint32, 0), ('size', uint16_t, 4), ('val', struct_ibv_flow_tcp_udp_filter, 6), ('mask', struct_ibv_flow_tcp_udp_filter, 10)])
@c.record
class struct_ibv_flow_gre_filter(c.Struct):
  SIZE = 8
  c_ks_res0_ver: int
  protocol: int
  key: int
struct_ibv_flow_gre_filter.register_fields([('c_ks_res0_ver', uint16_t, 0), ('protocol', uint16_t, 2), ('key', uint32_t, 4)])
@c.record
class struct_ibv_flow_spec_gre(c.Struct):
  SIZE = 24
  type: int
  size: int
  val: struct_ibv_flow_gre_filter
  mask: struct_ibv_flow_gre_filter
struct_ibv_flow_spec_gre.register_fields([('type', ctypes.c_uint32, 0), ('size', uint16_t, 4), ('val', struct_ibv_flow_gre_filter, 8), ('mask', struct_ibv_flow_gre_filter, 16)])
@c.record
class struct_ibv_flow_mpls_filter(c.Struct):
  SIZE = 4
  label: int
struct_ibv_flow_mpls_filter.register_fields([('label', uint32_t, 0)])
@c.record
class struct_ibv_flow_spec_mpls(c.Struct):
  SIZE = 16
  type: int
  size: int
  val: struct_ibv_flow_mpls_filter
  mask: struct_ibv_flow_mpls_filter
struct_ibv_flow_spec_mpls.register_fields([('type', ctypes.c_uint32, 0), ('size', uint16_t, 4), ('val', struct_ibv_flow_mpls_filter, 8), ('mask', struct_ibv_flow_mpls_filter, 12)])
@c.record
class struct_ibv_flow_tunnel_filter(c.Struct):
  SIZE = 4
  tunnel_id: int
struct_ibv_flow_tunnel_filter.register_fields([('tunnel_id', uint32_t, 0)])
@c.record
class struct_ibv_flow_spec_tunnel(c.Struct):
  SIZE = 16
  type: int
  size: int
  val: struct_ibv_flow_tunnel_filter
  mask: struct_ibv_flow_tunnel_filter
struct_ibv_flow_spec_tunnel.register_fields([('type', ctypes.c_uint32, 0), ('size', uint16_t, 4), ('val', struct_ibv_flow_tunnel_filter, 8), ('mask', struct_ibv_flow_tunnel_filter, 12)])
@c.record
class struct_ibv_flow_spec_action_tag(c.Struct):
  SIZE = 12
  type: int
  size: int
  tag_id: int
struct_ibv_flow_spec_action_tag.register_fields([('type', ctypes.c_uint32, 0), ('size', uint16_t, 4), ('tag_id', uint32_t, 8)])
@c.record
class struct_ibv_flow_spec_action_drop(c.Struct):
  SIZE = 8
  type: int
  size: int
struct_ibv_flow_spec_action_drop.register_fields([('type', ctypes.c_uint32, 0), ('size', uint16_t, 4)])
@c.record
class struct_ibv_flow_spec_action_handle(c.Struct):
  SIZE = 16
  type: int
  size: int
  action: c.POINTER[struct_ibv_flow_action]
@c.record
class struct_ibv_flow_action(c.Struct):
  SIZE = 8
  context: c.POINTER[struct_ibv_context]
struct_ibv_flow_action.register_fields([('context', c.POINTER[struct_ibv_context], 0)])
struct_ibv_flow_spec_action_handle.register_fields([('type', ctypes.c_uint32, 0), ('size', uint16_t, 4), ('action', c.POINTER[struct_ibv_flow_action], 8)])
@c.record
class struct_ibv_flow_spec_counter_action(c.Struct):
  SIZE = 16
  type: int
  size: int
  counters: c.POINTER[struct_ibv_counters]
@c.record
class struct_ibv_counters(c.Struct):
  SIZE = 8
  context: c.POINTER[struct_ibv_context]
struct_ibv_counters.register_fields([('context', c.POINTER[struct_ibv_context], 0)])
struct_ibv_flow_spec_counter_action.register_fields([('type', ctypes.c_uint32, 0), ('size', uint16_t, 4), ('counters', c.POINTER[struct_ibv_counters], 8)])
@c.record
class struct_ibv_flow_spec(c.Struct):
  SIZE = 88
  hdr: struct_ibv_flow_spec_hdr
  eth: struct_ibv_flow_spec_eth
  ipv4: struct_ibv_flow_spec_ipv4
  tcp_udp: struct_ibv_flow_spec_tcp_udp
  ipv4_ext: struct_ibv_flow_spec_ipv4_ext
  ipv6: struct_ibv_flow_spec_ipv6
  esp: struct_ibv_flow_spec_esp
  tunnel: struct_ibv_flow_spec_tunnel
  gre: struct_ibv_flow_spec_gre
  mpls: struct_ibv_flow_spec_mpls
  flow_tag: struct_ibv_flow_spec_action_tag
  drop: struct_ibv_flow_spec_action_drop
  handle: struct_ibv_flow_spec_action_handle
  flow_count: struct_ibv_flow_spec_counter_action
@c.record
class struct_ibv_flow_spec_hdr(c.Struct):
  SIZE = 8
  type: int
  size: int
struct_ibv_flow_spec_hdr.register_fields([('type', ctypes.c_uint32, 0), ('size', uint16_t, 4)])
struct_ibv_flow_spec.register_fields([('hdr', struct_ibv_flow_spec_hdr, 0), ('eth', struct_ibv_flow_spec_eth, 0), ('ipv4', struct_ibv_flow_spec_ipv4, 0), ('tcp_udp', struct_ibv_flow_spec_tcp_udp, 0), ('ipv4_ext', struct_ibv_flow_spec_ipv4_ext, 0), ('ipv6', struct_ibv_flow_spec_ipv6, 0), ('esp', struct_ibv_flow_spec_esp, 0), ('tunnel', struct_ibv_flow_spec_tunnel, 0), ('gre', struct_ibv_flow_spec_gre, 0), ('mpls', struct_ibv_flow_spec_mpls, 0), ('flow_tag', struct_ibv_flow_spec_action_tag, 0), ('drop', struct_ibv_flow_spec_action_drop, 0), ('handle', struct_ibv_flow_spec_action_handle, 0), ('flow_count', struct_ibv_flow_spec_counter_action, 0)])
@c.record
class struct_ibv_flow_attr(c.Struct):
  SIZE = 20
  comp_mask: int
  type: int
  size: int
  priority: int
  num_of_specs: int
  port: int
  flags: int
struct_ibv_flow_attr.register_fields([('comp_mask', uint32_t, 0), ('type', ctypes.c_uint32, 4), ('size', uint16_t, 8), ('priority', uint16_t, 10), ('num_of_specs', uint8_t, 12), ('port', uint8_t, 13), ('flags', uint32_t, 16)])
@c.record
class struct_ibv_flow(c.Struct):
  SIZE = 24
  comp_mask: int
  context: c.POINTER[struct_ibv_context]
  handle: int
struct_ibv_flow.register_fields([('comp_mask', uint32_t, 0), ('context', c.POINTER[struct_ibv_context], 8), ('handle', uint32_t, 16)])
enum_ibv_flow_action_esp_mask: dict[int, str] = {(IBV_FLOW_ACTION_ESP_MASK_ESN:=1): 'IBV_FLOW_ACTION_ESP_MASK_ESN'}
@c.record
class struct_ibv_flow_action_esp_attr(c.Struct):
  SIZE = 56
  esp_attr: c.POINTER[struct_ib_uverbs_flow_action_esp]
  keymat_proto: int
  keymat_len: int
  keymat_ptr: ctypes.c_void_p
  replay_proto: int
  replay_len: int
  replay_ptr: ctypes.c_void_p
  esp_encap: c.POINTER[struct_ib_uverbs_flow_action_esp_encap]
  comp_mask: int
  esn: int
@c.record
class struct_ib_uverbs_flow_action_esp(c.Struct):
  SIZE = 24
  spi: int
  seq: int
  tfc_pad: int
  flags: int
  hard_limit_pkts: int
__u32: TypeAlias = ctypes.c_uint32
__u64: TypeAlias = ctypes.c_uint64
struct_ib_uverbs_flow_action_esp.register_fields([('spi', ctypes.c_uint32, 0), ('seq', ctypes.c_uint32, 4), ('tfc_pad', ctypes.c_uint32, 8), ('flags', ctypes.c_uint32, 12), ('hard_limit_pkts', ctypes.c_uint64, 16)])
enum_ib_uverbs_flow_action_esp_keymat: dict[int, str] = {(IB_UVERBS_FLOW_ACTION_ESP_KEYMAT_AES_GCM:=0): 'IB_UVERBS_FLOW_ACTION_ESP_KEYMAT_AES_GCM'}
enum_ib_uverbs_flow_action_esp_replay: dict[int, str] = {(IB_UVERBS_FLOW_ACTION_ESP_REPLAY_NONE:=0): 'IB_UVERBS_FLOW_ACTION_ESP_REPLAY_NONE', (IB_UVERBS_FLOW_ACTION_ESP_REPLAY_BMP:=1): 'IB_UVERBS_FLOW_ACTION_ESP_REPLAY_BMP'}
@c.record
class struct_ib_uverbs_flow_action_esp_encap(c.Struct):
  SIZE = 24
  val_ptr: ctypes.c_void_p
  val_ptr_data_u64: int
  next_ptr: c.POINTER[struct_ib_uverbs_flow_action_esp_encap]
  next_ptr_data_u64: int
  len: int
  type: int
__u16: TypeAlias = ctypes.c_uint16
struct_ib_uverbs_flow_action_esp_encap.register_fields([('val_ptr', ctypes.c_void_p, 0), ('val_ptr_data_u64', ctypes.c_uint64, 0), ('next_ptr', c.POINTER[struct_ib_uverbs_flow_action_esp_encap], 8), ('next_ptr_data_u64', ctypes.c_uint64, 8), ('len', ctypes.c_uint16, 16), ('type', ctypes.c_uint16, 18)])
struct_ibv_flow_action_esp_attr.register_fields([('esp_attr', c.POINTER[struct_ib_uverbs_flow_action_esp], 0), ('keymat_proto', ctypes.c_uint32, 8), ('keymat_len', uint16_t, 12), ('keymat_ptr', ctypes.c_void_p, 16), ('replay_proto', ctypes.c_uint32, 24), ('replay_len', uint16_t, 28), ('replay_ptr', ctypes.c_void_p, 32), ('esp_encap', c.POINTER[struct_ib_uverbs_flow_action_esp_encap], 40), ('comp_mask', uint32_t, 48), ('esn', uint32_t, 52)])
_anonenum4: dict[int, str] = {(IBV_SYSFS_NAME_MAX:=64): 'IBV_SYSFS_NAME_MAX', (IBV_SYSFS_PATH_MAX:=256): 'IBV_SYSFS_PATH_MAX'}
enum_ibv_cq_init_attr_mask: dict[int, str] = {(IBV_CQ_INIT_ATTR_MASK_FLAGS:=1): 'IBV_CQ_INIT_ATTR_MASK_FLAGS', (IBV_CQ_INIT_ATTR_MASK_PD:=2): 'IBV_CQ_INIT_ATTR_MASK_PD'}
enum_ibv_create_cq_attr_flags: dict[int, str] = {(IBV_CREATE_CQ_ATTR_SINGLE_THREADED:=1): 'IBV_CREATE_CQ_ATTR_SINGLE_THREADED', (IBV_CREATE_CQ_ATTR_IGNORE_OVERRUN:=2): 'IBV_CREATE_CQ_ATTR_IGNORE_OVERRUN'}
@c.record
class struct_ibv_cq_init_attr_ex(c.Struct):
  SIZE = 56
  cqe: int
  cq_context: ctypes.c_void_p
  channel: c.POINTER[struct_ibv_comp_channel]
  comp_vector: int
  wc_flags: int
  comp_mask: int
  flags: int
  parent_domain: c.POINTER[struct_ibv_pd]
struct_ibv_cq_init_attr_ex.register_fields([('cqe', uint32_t, 0), ('cq_context', ctypes.c_void_p, 8), ('channel', c.POINTER[struct_ibv_comp_channel], 16), ('comp_vector', uint32_t, 24), ('wc_flags', uint64_t, 32), ('comp_mask', uint32_t, 40), ('flags', uint32_t, 44), ('parent_domain', c.POINTER[struct_ibv_pd], 48)])
enum_ibv_parent_domain_init_attr_mask: dict[int, str] = {(IBV_PARENT_DOMAIN_INIT_ATTR_ALLOCATORS:=1): 'IBV_PARENT_DOMAIN_INIT_ATTR_ALLOCATORS', (IBV_PARENT_DOMAIN_INIT_ATTR_PD_CONTEXT:=2): 'IBV_PARENT_DOMAIN_INIT_ATTR_PD_CONTEXT'}
@c.record
class struct_ibv_parent_domain_init_attr(c.Struct):
  SIZE = 48
  pd: c.POINTER[struct_ibv_pd]
  td: c.POINTER[struct_ibv_td]
  comp_mask: int
  alloc: c.CFUNCTYPE[ctypes.c_void_p, [c.POINTER[struct_ibv_pd], ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]]
  free: c.CFUNCTYPE[None, [c.POINTER[struct_ibv_pd], ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64]]
  pd_context: ctypes.c_void_p
struct_ibv_parent_domain_init_attr.register_fields([('pd', c.POINTER[struct_ibv_pd], 0), ('td', c.POINTER[struct_ibv_td], 8), ('comp_mask', uint32_t, 16), ('alloc', c.CFUNCTYPE[ctypes.c_void_p, [c.POINTER[struct_ibv_pd], ctypes.c_void_p, size_t, size_t, uint64_t]], 24), ('free', c.CFUNCTYPE[None, [c.POINTER[struct_ibv_pd], ctypes.c_void_p, ctypes.c_void_p, uint64_t]], 32), ('pd_context', ctypes.c_void_p, 40)])
@c.record
class struct_ibv_counters_init_attr(c.Struct):
  SIZE = 4
  comp_mask: int
struct_ibv_counters_init_attr.register_fields([('comp_mask', uint32_t, 0)])
enum_ibv_counter_description: dict[int, str] = {(IBV_COUNTER_PACKETS:=0): 'IBV_COUNTER_PACKETS', (IBV_COUNTER_BYTES:=1): 'IBV_COUNTER_BYTES'}
@c.record
class struct_ibv_counter_attach_attr(c.Struct):
  SIZE = 12
  counter_desc: int
  index: int
  comp_mask: int
struct_ibv_counter_attach_attr.register_fields([('counter_desc', ctypes.c_uint32, 0), ('index', uint32_t, 4), ('comp_mask', uint32_t, 8)])
enum_ibv_read_counters_flags: dict[int, str] = {(IBV_READ_COUNTERS_ATTR_PREFER_CACHED:=1): 'IBV_READ_COUNTERS_ATTR_PREFER_CACHED'}
enum_ibv_values_mask: dict[int, str] = {(IBV_VALUES_MASK_RAW_CLOCK:=1): 'IBV_VALUES_MASK_RAW_CLOCK', (IBV_VALUES_MASK_RESERVED:=2): 'IBV_VALUES_MASK_RESERVED'}
@c.record
class struct_ibv_values_ex(c.Struct):
  SIZE = 24
  comp_mask: int
  raw_clock: struct_timespec
@c.record
class struct_timespec(c.Struct):
  SIZE = 16
  tv_sec: int
  tv_nsec: int
__time_t: TypeAlias = ctypes.c_int64
__syscall_slong_t: TypeAlias = ctypes.c_int64
struct_timespec.register_fields([('tv_sec', ctypes.c_int64, 0), ('tv_nsec', ctypes.c_int64, 8)])
struct_ibv_values_ex.register_fields([('comp_mask', uint32_t, 0), ('raw_clock', struct_timespec, 8)])
@c.record
class struct_verbs_context(c.Struct):
  SIZE = 648
  query_port: c.CFUNCTYPE[ctypes.c_int32, [c.POINTER[struct_ibv_context], ctypes.c_ubyte, c.POINTER[struct_ibv_port_attr], ctypes.c_uint64]]
  advise_mr: c.CFUNCTYPE[ctypes.c_int32, [c.POINTER[struct_ibv_pd], ctypes.c_uint32, ctypes.c_uint32, c.POINTER[struct_ibv_sge], ctypes.c_uint32]]
  alloc_null_mr: c.CFUNCTYPE[c.POINTER[struct_ibv_mr], [c.POINTER[struct_ibv_pd]]]
  read_counters: c.CFUNCTYPE[ctypes.c_int32, [c.POINTER[struct_ibv_counters], c.POINTER[ctypes.c_uint64], ctypes.c_uint32, ctypes.c_uint32]]
  attach_counters_point_flow: c.CFUNCTYPE[ctypes.c_int32, [c.POINTER[struct_ibv_counters], c.POINTER[struct_ibv_counter_attach_attr], c.POINTER[struct_ibv_flow]]]
  create_counters: c.CFUNCTYPE[c.POINTER[struct_ibv_counters], [c.POINTER[struct_ibv_context], c.POINTER[struct_ibv_counters_init_attr]]]
  destroy_counters: c.CFUNCTYPE[ctypes.c_int32, [c.POINTER[struct_ibv_counters]]]
  reg_dm_mr: c.CFUNCTYPE[c.POINTER[struct_ibv_mr], [c.POINTER[struct_ibv_pd], c.POINTER[struct_ibv_dm], ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint32]]
  alloc_dm: c.CFUNCTYPE[c.POINTER[struct_ibv_dm], [c.POINTER[struct_ibv_context], c.POINTER[struct_ibv_alloc_dm_attr]]]
  free_dm: c.CFUNCTYPE[ctypes.c_int32, [c.POINTER[struct_ibv_dm]]]
  modify_flow_action_esp: c.CFUNCTYPE[ctypes.c_int32, [c.POINTER[struct_ibv_flow_action], c.POINTER[struct_ibv_flow_action_esp_attr]]]
  destroy_flow_action: c.CFUNCTYPE[ctypes.c_int32, [c.POINTER[struct_ibv_flow_action]]]
  create_flow_action_esp: c.CFUNCTYPE[c.POINTER[struct_ibv_flow_action], [c.POINTER[struct_ibv_context], c.POINTER[struct_ibv_flow_action_esp_attr]]]
  modify_qp_rate_limit: c.CFUNCTYPE[ctypes.c_int32, [c.POINTER[struct_ibv_qp], c.POINTER[struct_ibv_qp_rate_limit_attr]]]
  alloc_parent_domain: c.CFUNCTYPE[c.POINTER[struct_ibv_pd], [c.POINTER[struct_ibv_context], c.POINTER[struct_ibv_parent_domain_init_attr]]]
  dealloc_td: c.CFUNCTYPE[ctypes.c_int32, [c.POINTER[struct_ibv_td]]]
  alloc_td: c.CFUNCTYPE[c.POINTER[struct_ibv_td], [c.POINTER[struct_ibv_context], c.POINTER[struct_ibv_td_init_attr]]]
  modify_cq: c.CFUNCTYPE[ctypes.c_int32, [c.POINTER[struct_ibv_cq], c.POINTER[struct_ibv_modify_cq_attr]]]
  post_srq_ops: c.CFUNCTYPE[ctypes.c_int32, [c.POINTER[struct_ibv_srq], c.POINTER[struct_ibv_ops_wr], c.POINTER[c.POINTER[struct_ibv_ops_wr]]]]
  destroy_rwq_ind_table: c.CFUNCTYPE[ctypes.c_int32, [c.POINTER[struct_ibv_rwq_ind_table]]]
  create_rwq_ind_table: c.CFUNCTYPE[c.POINTER[struct_ibv_rwq_ind_table], [c.POINTER[struct_ibv_context], c.POINTER[struct_ibv_rwq_ind_table_init_attr]]]
  destroy_wq: c.CFUNCTYPE[ctypes.c_int32, [c.POINTER[struct_ibv_wq]]]
  modify_wq: c.CFUNCTYPE[ctypes.c_int32, [c.POINTER[struct_ibv_wq], c.POINTER[struct_ibv_wq_attr]]]
  create_wq: c.CFUNCTYPE[c.POINTER[struct_ibv_wq], [c.POINTER[struct_ibv_context], c.POINTER[struct_ibv_wq_init_attr]]]
  query_rt_values: c.CFUNCTYPE[ctypes.c_int32, [c.POINTER[struct_ibv_context], c.POINTER[struct_ibv_values_ex]]]
  create_cq_ex: c.CFUNCTYPE[c.POINTER[struct_ibv_cq_ex], [c.POINTER[struct_ibv_context], c.POINTER[struct_ibv_cq_init_attr_ex]]]
  priv: c.POINTER[struct_verbs_ex_private]
  query_device_ex: c.CFUNCTYPE[ctypes.c_int32, [c.POINTER[struct_ibv_context], c.POINTER[struct_ibv_query_device_ex_input], c.POINTER[struct_ibv_device_attr_ex], ctypes.c_uint64]]
  ibv_destroy_flow: c.CFUNCTYPE[ctypes.c_int32, [c.POINTER[struct_ibv_flow]]]
  ABI_placeholder2: c.CFUNCTYPE[None, []]
  ibv_create_flow: c.CFUNCTYPE[c.POINTER[struct_ibv_flow], [c.POINTER[struct_ibv_qp], c.POINTER[struct_ibv_flow_attr]]]
  ABI_placeholder1: c.CFUNCTYPE[None, []]
  open_qp: c.CFUNCTYPE[c.POINTER[struct_ibv_qp], [c.POINTER[struct_ibv_context], c.POINTER[struct_ibv_qp_open_attr]]]
  create_qp_ex: c.CFUNCTYPE[c.POINTER[struct_ibv_qp], [c.POINTER[struct_ibv_context], c.POINTER[struct_ibv_qp_init_attr_ex]]]
  get_srq_num: c.CFUNCTYPE[ctypes.c_int32, [c.POINTER[struct_ibv_srq], c.POINTER[ctypes.c_uint32]]]
  create_srq_ex: c.CFUNCTYPE[c.POINTER[struct_ibv_srq], [c.POINTER[struct_ibv_context], c.POINTER[struct_ibv_srq_init_attr_ex]]]
  open_xrcd: c.CFUNCTYPE[c.POINTER[struct_ibv_xrcd], [c.POINTER[struct_ibv_context], c.POINTER[struct_ibv_xrcd_init_attr]]]
  close_xrcd: c.CFUNCTYPE[ctypes.c_int32, [c.POINTER[struct_ibv_xrcd]]]
  _ABI_placeholder3: int
  sz: int
  context: struct_ibv_context
enum_ib_uverbs_advise_mr_advice: dict[int, str] = {(IB_UVERBS_ADVISE_MR_ADVICE_PREFETCH:=0): 'IB_UVERBS_ADVISE_MR_ADVICE_PREFETCH', (IB_UVERBS_ADVISE_MR_ADVICE_PREFETCH_WRITE:=1): 'IB_UVERBS_ADVISE_MR_ADVICE_PREFETCH_WRITE', (IB_UVERBS_ADVISE_MR_ADVICE_PREFETCH_NO_FAULT:=2): 'IB_UVERBS_ADVISE_MR_ADVICE_PREFETCH_NO_FAULT'}
class struct_verbs_ex_private(c.Struct): pass
struct_verbs_context.register_fields([('query_port', c.CFUNCTYPE[ctypes.c_int32, [c.POINTER[struct_ibv_context], uint8_t, c.POINTER[struct_ibv_port_attr], size_t]], 0), ('advise_mr', c.CFUNCTYPE[ctypes.c_int32, [c.POINTER[struct_ibv_pd], ctypes.c_uint32, uint32_t, c.POINTER[struct_ibv_sge], uint32_t]], 8), ('alloc_null_mr', c.CFUNCTYPE[c.POINTER[struct_ibv_mr], [c.POINTER[struct_ibv_pd]]], 16), ('read_counters', c.CFUNCTYPE[ctypes.c_int32, [c.POINTER[struct_ibv_counters], c.POINTER[uint64_t], uint32_t, uint32_t]], 24), ('attach_counters_point_flow', c.CFUNCTYPE[ctypes.c_int32, [c.POINTER[struct_ibv_counters], c.POINTER[struct_ibv_counter_attach_attr], c.POINTER[struct_ibv_flow]]], 32), ('create_counters', c.CFUNCTYPE[c.POINTER[struct_ibv_counters], [c.POINTER[struct_ibv_context], c.POINTER[struct_ibv_counters_init_attr]]], 40), ('destroy_counters', c.CFUNCTYPE[ctypes.c_int32, [c.POINTER[struct_ibv_counters]]], 48), ('reg_dm_mr', c.CFUNCTYPE[c.POINTER[struct_ibv_mr], [c.POINTER[struct_ibv_pd], c.POINTER[struct_ibv_dm], uint64_t, size_t, ctypes.c_uint32]], 56), ('alloc_dm', c.CFUNCTYPE[c.POINTER[struct_ibv_dm], [c.POINTER[struct_ibv_context], c.POINTER[struct_ibv_alloc_dm_attr]]], 64), ('free_dm', c.CFUNCTYPE[ctypes.c_int32, [c.POINTER[struct_ibv_dm]]], 72), ('modify_flow_action_esp', c.CFUNCTYPE[ctypes.c_int32, [c.POINTER[struct_ibv_flow_action], c.POINTER[struct_ibv_flow_action_esp_attr]]], 80), ('destroy_flow_action', c.CFUNCTYPE[ctypes.c_int32, [c.POINTER[struct_ibv_flow_action]]], 88), ('create_flow_action_esp', c.CFUNCTYPE[c.POINTER[struct_ibv_flow_action], [c.POINTER[struct_ibv_context], c.POINTER[struct_ibv_flow_action_esp_attr]]], 96), ('modify_qp_rate_limit', c.CFUNCTYPE[ctypes.c_int32, [c.POINTER[struct_ibv_qp], c.POINTER[struct_ibv_qp_rate_limit_attr]]], 104), ('alloc_parent_domain', c.CFUNCTYPE[c.POINTER[struct_ibv_pd], [c.POINTER[struct_ibv_context], c.POINTER[struct_ibv_parent_domain_init_attr]]], 112), ('dealloc_td', c.CFUNCTYPE[ctypes.c_int32, [c.POINTER[struct_ibv_td]]], 120), ('alloc_td', c.CFUNCTYPE[c.POINTER[struct_ibv_td], [c.POINTER[struct_ibv_context], c.POINTER[struct_ibv_td_init_attr]]], 128), ('modify_cq', c.CFUNCTYPE[ctypes.c_int32, [c.POINTER[struct_ibv_cq], c.POINTER[struct_ibv_modify_cq_attr]]], 136), ('post_srq_ops', c.CFUNCTYPE[ctypes.c_int32, [c.POINTER[struct_ibv_srq], c.POINTER[struct_ibv_ops_wr], c.POINTER[c.POINTER[struct_ibv_ops_wr]]]], 144), ('destroy_rwq_ind_table', c.CFUNCTYPE[ctypes.c_int32, [c.POINTER[struct_ibv_rwq_ind_table]]], 152), ('create_rwq_ind_table', c.CFUNCTYPE[c.POINTER[struct_ibv_rwq_ind_table], [c.POINTER[struct_ibv_context], c.POINTER[struct_ibv_rwq_ind_table_init_attr]]], 160), ('destroy_wq', c.CFUNCTYPE[ctypes.c_int32, [c.POINTER[struct_ibv_wq]]], 168), ('modify_wq', c.CFUNCTYPE[ctypes.c_int32, [c.POINTER[struct_ibv_wq], c.POINTER[struct_ibv_wq_attr]]], 176), ('create_wq', c.CFUNCTYPE[c.POINTER[struct_ibv_wq], [c.POINTER[struct_ibv_context], c.POINTER[struct_ibv_wq_init_attr]]], 184), ('query_rt_values', c.CFUNCTYPE[ctypes.c_int32, [c.POINTER[struct_ibv_context], c.POINTER[struct_ibv_values_ex]]], 192), ('create_cq_ex', c.CFUNCTYPE[c.POINTER[struct_ibv_cq_ex], [c.POINTER[struct_ibv_context], c.POINTER[struct_ibv_cq_init_attr_ex]]], 200), ('priv', c.POINTER[struct_verbs_ex_private], 208), ('query_device_ex', c.CFUNCTYPE[ctypes.c_int32, [c.POINTER[struct_ibv_context], c.POINTER[struct_ibv_query_device_ex_input], c.POINTER[struct_ibv_device_attr_ex], size_t]], 216), ('ibv_destroy_flow', c.CFUNCTYPE[ctypes.c_int32, [c.POINTER[struct_ibv_flow]]], 224), ('ABI_placeholder2', c.CFUNCTYPE[None, []], 232), ('ibv_create_flow', c.CFUNCTYPE[c.POINTER[struct_ibv_flow], [c.POINTER[struct_ibv_qp], c.POINTER[struct_ibv_flow_attr]]], 240), ('ABI_placeholder1', c.CFUNCTYPE[None, []], 248), ('open_qp', c.CFUNCTYPE[c.POINTER[struct_ibv_qp], [c.POINTER[struct_ibv_context], c.POINTER[struct_ibv_qp_open_attr]]], 256), ('create_qp_ex', c.CFUNCTYPE[c.POINTER[struct_ibv_qp], [c.POINTER[struct_ibv_context], c.POINTER[struct_ibv_qp_init_attr_ex]]], 264), ('get_srq_num', c.CFUNCTYPE[ctypes.c_int32, [c.POINTER[struct_ibv_srq], c.POINTER[uint32_t]]], 272), ('create_srq_ex', c.CFUNCTYPE[c.POINTER[struct_ibv_srq], [c.POINTER[struct_ibv_context], c.POINTER[struct_ibv_srq_init_attr_ex]]], 280), ('open_xrcd', c.CFUNCTYPE[c.POINTER[struct_ibv_xrcd], [c.POINTER[struct_ibv_context], c.POINTER[struct_ibv_xrcd_init_attr]]], 288), ('close_xrcd', c.CFUNCTYPE[ctypes.c_int32, [c.POINTER[struct_ibv_xrcd]]], 296), ('_ABI_placeholder3', uint64_t, 304), ('sz', size_t, 312), ('context', struct_ibv_context, 320)])
@dll.bind(c.POINTER[c.POINTER[struct_ibv_device]], c.POINTER[ctypes.c_int32])
def ibv_get_device_list(num_devices:c.POINTER[ctypes.c_int32]) -> c.POINTER[c.POINTER[struct_ibv_device]]: ...
@dll.bind(None, c.POINTER[c.POINTER[struct_ibv_device]])
def ibv_free_device_list(list:c.POINTER[c.POINTER[struct_ibv_device]]) -> None: ...
@dll.bind(c.POINTER[ctypes.c_char], c.POINTER[struct_ibv_device])
def ibv_get_device_name(device:c.POINTER[struct_ibv_device]) -> c.POINTER[ctypes.c_char]: ...
@dll.bind(ctypes.c_int32, c.POINTER[struct_ibv_device])
def ibv_get_device_index(device:c.POINTER[struct_ibv_device]) -> int: ...
@dll.bind(ctypes.c_uint64, c.POINTER[struct_ibv_device])
def ibv_get_device_guid(device:c.POINTER[struct_ibv_device]) -> ctypes.c_uint64: ...
@dll.bind(c.POINTER[struct_ibv_context], c.POINTER[struct_ibv_device])
def ibv_open_device(device:c.POINTER[struct_ibv_device]) -> c.POINTER[struct_ibv_context]: ...
@dll.bind(ctypes.c_int32, c.POINTER[struct_ibv_context])
def ibv_close_device(context:c.POINTER[struct_ibv_context]) -> int: ...
@dll.bind(c.POINTER[struct_ibv_context], ctypes.c_int32)
def ibv_import_device(cmd_fd:int) -> c.POINTER[struct_ibv_context]: ...
@dll.bind(c.POINTER[struct_ibv_pd], c.POINTER[struct_ibv_context], uint32_t)
def ibv_import_pd(context:c.POINTER[struct_ibv_context], pd_handle:uint32_t) -> c.POINTER[struct_ibv_pd]: ...
@dll.bind(None, c.POINTER[struct_ibv_pd])
def ibv_unimport_pd(pd:c.POINTER[struct_ibv_pd]) -> None: ...
@dll.bind(c.POINTER[struct_ibv_mr], c.POINTER[struct_ibv_pd], uint32_t)
def ibv_import_mr(pd:c.POINTER[struct_ibv_pd], mr_handle:uint32_t) -> c.POINTER[struct_ibv_mr]: ...
@dll.bind(None, c.POINTER[struct_ibv_mr])
def ibv_unimport_mr(mr:c.POINTER[struct_ibv_mr]) -> None: ...
@dll.bind(c.POINTER[struct_ibv_dm], c.POINTER[struct_ibv_context], uint32_t)
def ibv_import_dm(context:c.POINTER[struct_ibv_context], dm_handle:uint32_t) -> c.POINTER[struct_ibv_dm]: ...
@dll.bind(None, c.POINTER[struct_ibv_dm])
def ibv_unimport_dm(dm:c.POINTER[struct_ibv_dm]) -> None: ...
@dll.bind(ctypes.c_int32, c.POINTER[struct_ibv_context], c.POINTER[struct_ibv_async_event])
def ibv_get_async_event(context:c.POINTER[struct_ibv_context], event:c.POINTER[struct_ibv_async_event]) -> int: ...
@dll.bind(None, c.POINTER[struct_ibv_async_event])
def ibv_ack_async_event(event:c.POINTER[struct_ibv_async_event]) -> None: ...
@dll.bind(ctypes.c_int32, c.POINTER[struct_ibv_context], c.POINTER[struct_ibv_device_attr])
def ibv_query_device(context:c.POINTER[struct_ibv_context], device_attr:c.POINTER[struct_ibv_device_attr]) -> int: ...
@dll.bind(ctypes.c_int32, c.POINTER[struct_ibv_context], uint8_t, c.POINTER[struct__compat_ibv_port_attr])
def ibv_query_port(context:c.POINTER[struct_ibv_context], port_num:uint8_t, port_attr:c.POINTER[struct__compat_ibv_port_attr]) -> int: ...
@dll.bind(ctypes.c_int32, c.POINTER[struct_ibv_context], uint8_t, ctypes.c_int32, c.POINTER[union_ibv_gid])
def ibv_query_gid(context:c.POINTER[struct_ibv_context], port_num:uint8_t, index:int, gid:c.POINTER[union_ibv_gid]) -> int: ...
@dll.bind(ctypes.c_int32, c.POINTER[struct_ibv_context], uint32_t, uint32_t, c.POINTER[struct_ibv_gid_entry], uint32_t, size_t)
def _ibv_query_gid_ex(context:c.POINTER[struct_ibv_context], port_num:uint32_t, gid_index:uint32_t, entry:c.POINTER[struct_ibv_gid_entry], flags:uint32_t, entry_size:size_t) -> int: ...
ssize_t: TypeAlias = ctypes.c_int64
@dll.bind(ssize_t, c.POINTER[struct_ibv_context], c.POINTER[struct_ibv_gid_entry], size_t, uint32_t, size_t)
def _ibv_query_gid_table(context:c.POINTER[struct_ibv_context], entries:c.POINTER[struct_ibv_gid_entry], max_entries:size_t, flags:uint32_t, entry_size:size_t) -> ssize_t: ...
@dll.bind(ctypes.c_int32, c.POINTER[struct_ibv_context], uint8_t, ctypes.c_int32, c.POINTER[ctypes.c_uint16])
def ibv_query_pkey(context:c.POINTER[struct_ibv_context], port_num:uint8_t, index:int, pkey:c.POINTER[ctypes.c_uint16]) -> int: ...
@dll.bind(ctypes.c_int32, c.POINTER[struct_ibv_context], uint8_t, ctypes.c_uint16)
def ibv_get_pkey_index(context:c.POINTER[struct_ibv_context], port_num:uint8_t, pkey:ctypes.c_uint16) -> int: ...
@dll.bind(c.POINTER[struct_ibv_pd], c.POINTER[struct_ibv_context])
def ibv_alloc_pd(context:c.POINTER[struct_ibv_context]) -> c.POINTER[struct_ibv_pd]: ...
@dll.bind(ctypes.c_int32, c.POINTER[struct_ibv_pd])
def ibv_dealloc_pd(pd:c.POINTER[struct_ibv_pd]) -> int: ...
@dll.bind(c.POINTER[struct_ibv_mr], c.POINTER[struct_ibv_pd], ctypes.c_void_p, size_t, uint64_t, ctypes.c_uint32)
def ibv_reg_mr_iova2(pd:c.POINTER[struct_ibv_pd], addr:ctypes.c_void_p, length:size_t, iova:uint64_t, access:int) -> c.POINTER[struct_ibv_mr]: ...
@dll.bind(c.POINTER[struct_ibv_mr], c.POINTER[struct_ibv_pd], ctypes.c_void_p, size_t, ctypes.c_int32)
def ibv_reg_mr(pd:c.POINTER[struct_ibv_pd], addr:ctypes.c_void_p, length:size_t, access:int) -> c.POINTER[struct_ibv_mr]: ...
@dll.bind(c.POINTER[struct_ibv_mr], c.POINTER[struct_ibv_pd], ctypes.c_void_p, size_t, uint64_t, ctypes.c_int32)
def ibv_reg_mr_iova(pd:c.POINTER[struct_ibv_pd], addr:ctypes.c_void_p, length:size_t, iova:uint64_t, access:int) -> c.POINTER[struct_ibv_mr]: ...
@dll.bind(c.POINTER[struct_ibv_mr], c.POINTER[struct_ibv_pd], uint64_t, size_t, uint64_t, ctypes.c_int32, ctypes.c_int32)
def ibv_reg_dmabuf_mr(pd:c.POINTER[struct_ibv_pd], offset:uint64_t, length:size_t, iova:uint64_t, fd:int, access:int) -> c.POINTER[struct_ibv_mr]: ...
enum_ibv_rereg_mr_err_code: dict[int, str] = {(IBV_REREG_MR_ERR_INPUT:=-1): 'IBV_REREG_MR_ERR_INPUT', (IBV_REREG_MR_ERR_DONT_FORK_NEW:=-2): 'IBV_REREG_MR_ERR_DONT_FORK_NEW', (IBV_REREG_MR_ERR_DO_FORK_OLD:=-3): 'IBV_REREG_MR_ERR_DO_FORK_OLD', (IBV_REREG_MR_ERR_CMD:=-4): 'IBV_REREG_MR_ERR_CMD', (IBV_REREG_MR_ERR_CMD_AND_DO_FORK_NEW:=-5): 'IBV_REREG_MR_ERR_CMD_AND_DO_FORK_NEW'}
@dll.bind(ctypes.c_int32, c.POINTER[struct_ibv_mr], ctypes.c_int32, c.POINTER[struct_ibv_pd], ctypes.c_void_p, size_t, ctypes.c_int32)
def ibv_rereg_mr(mr:c.POINTER[struct_ibv_mr], flags:int, pd:c.POINTER[struct_ibv_pd], addr:ctypes.c_void_p, length:size_t, access:int) -> int: ...
@dll.bind(ctypes.c_int32, c.POINTER[struct_ibv_mr])
def ibv_dereg_mr(mr:c.POINTER[struct_ibv_mr]) -> int: ...
@dll.bind(c.POINTER[struct_ibv_comp_channel], c.POINTER[struct_ibv_context])
def ibv_create_comp_channel(context:c.POINTER[struct_ibv_context]) -> c.POINTER[struct_ibv_comp_channel]: ...
@dll.bind(ctypes.c_int32, c.POINTER[struct_ibv_comp_channel])
def ibv_destroy_comp_channel(channel:c.POINTER[struct_ibv_comp_channel]) -> int: ...
@dll.bind(c.POINTER[struct_ibv_cq], c.POINTER[struct_ibv_context], ctypes.c_int32, ctypes.c_void_p, c.POINTER[struct_ibv_comp_channel], ctypes.c_int32)
def ibv_create_cq(context:c.POINTER[struct_ibv_context], cqe:int, cq_context:ctypes.c_void_p, channel:c.POINTER[struct_ibv_comp_channel], comp_vector:int) -> c.POINTER[struct_ibv_cq]: ...
@dll.bind(ctypes.c_int32, c.POINTER[struct_ibv_cq], ctypes.c_int32)
def ibv_resize_cq(cq:c.POINTER[struct_ibv_cq], cqe:int) -> int: ...
@dll.bind(ctypes.c_int32, c.POINTER[struct_ibv_cq])
def ibv_destroy_cq(cq:c.POINTER[struct_ibv_cq]) -> int: ...
@dll.bind(ctypes.c_int32, c.POINTER[struct_ibv_comp_channel], c.POINTER[c.POINTER[struct_ibv_cq]], c.POINTER[ctypes.c_void_p])
def ibv_get_cq_event(channel:c.POINTER[struct_ibv_comp_channel], cq:c.POINTER[c.POINTER[struct_ibv_cq]], cq_context:c.POINTER[ctypes.c_void_p]) -> int: ...
@dll.bind(None, c.POINTER[struct_ibv_cq], ctypes.c_uint32)
def ibv_ack_cq_events(cq:c.POINTER[struct_ibv_cq], nevents:int) -> None: ...
@dll.bind(c.POINTER[struct_ibv_srq], c.POINTER[struct_ibv_pd], c.POINTER[struct_ibv_srq_init_attr])
def ibv_create_srq(pd:c.POINTER[struct_ibv_pd], srq_init_attr:c.POINTER[struct_ibv_srq_init_attr]) -> c.POINTER[struct_ibv_srq]: ...
@dll.bind(ctypes.c_int32, c.POINTER[struct_ibv_srq], c.POINTER[struct_ibv_srq_attr], ctypes.c_int32)
def ibv_modify_srq(srq:c.POINTER[struct_ibv_srq], srq_attr:c.POINTER[struct_ibv_srq_attr], srq_attr_mask:int) -> int: ...
@dll.bind(ctypes.c_int32, c.POINTER[struct_ibv_srq], c.POINTER[struct_ibv_srq_attr])
def ibv_query_srq(srq:c.POINTER[struct_ibv_srq], srq_attr:c.POINTER[struct_ibv_srq_attr]) -> int: ...
@dll.bind(ctypes.c_int32, c.POINTER[struct_ibv_srq])
def ibv_destroy_srq(srq:c.POINTER[struct_ibv_srq]) -> int: ...
@dll.bind(c.POINTER[struct_ibv_qp], c.POINTER[struct_ibv_pd], c.POINTER[struct_ibv_qp_init_attr])
def ibv_create_qp(pd:c.POINTER[struct_ibv_pd], qp_init_attr:c.POINTER[struct_ibv_qp_init_attr]) -> c.POINTER[struct_ibv_qp]: ...
@dll.bind(ctypes.c_int32, c.POINTER[struct_ibv_qp], c.POINTER[struct_ibv_qp_attr], ctypes.c_int32)
def ibv_modify_qp(qp:c.POINTER[struct_ibv_qp], attr:c.POINTER[struct_ibv_qp_attr], attr_mask:int) -> int: ...
@dll.bind(ctypes.c_int32, c.POINTER[struct_ibv_qp], ctypes.c_uint32, uint32_t)
def ibv_query_qp_data_in_order(qp:c.POINTER[struct_ibv_qp], op:ctypes.c_uint32, flags:uint32_t) -> int: ...
@dll.bind(ctypes.c_int32, c.POINTER[struct_ibv_qp], c.POINTER[struct_ibv_qp_attr], ctypes.c_int32, c.POINTER[struct_ibv_qp_init_attr])
def ibv_query_qp(qp:c.POINTER[struct_ibv_qp], attr:c.POINTER[struct_ibv_qp_attr], attr_mask:int, init_attr:c.POINTER[struct_ibv_qp_init_attr]) -> int: ...
@dll.bind(ctypes.c_int32, c.POINTER[struct_ibv_qp])
def ibv_destroy_qp(qp:c.POINTER[struct_ibv_qp]) -> int: ...
@dll.bind(c.POINTER[struct_ibv_ah], c.POINTER[struct_ibv_pd], c.POINTER[struct_ibv_ah_attr])
def ibv_create_ah(pd:c.POINTER[struct_ibv_pd], attr:c.POINTER[struct_ibv_ah_attr]) -> c.POINTER[struct_ibv_ah]: ...
@dll.bind(ctypes.c_int32, c.POINTER[struct_ibv_context], uint8_t, c.POINTER[struct_ibv_wc], c.POINTER[struct_ibv_grh], c.POINTER[struct_ibv_ah_attr])
def ibv_init_ah_from_wc(context:c.POINTER[struct_ibv_context], port_num:uint8_t, wc:c.POINTER[struct_ibv_wc], grh:c.POINTER[struct_ibv_grh], ah_attr:c.POINTER[struct_ibv_ah_attr]) -> int: ...
@dll.bind(c.POINTER[struct_ibv_ah], c.POINTER[struct_ibv_pd], c.POINTER[struct_ibv_wc], c.POINTER[struct_ibv_grh], uint8_t)
def ibv_create_ah_from_wc(pd:c.POINTER[struct_ibv_pd], wc:c.POINTER[struct_ibv_wc], grh:c.POINTER[struct_ibv_grh], port_num:uint8_t) -> c.POINTER[struct_ibv_ah]: ...
@dll.bind(ctypes.c_int32, c.POINTER[struct_ibv_ah])
def ibv_destroy_ah(ah:c.POINTER[struct_ibv_ah]) -> int: ...
@dll.bind(ctypes.c_int32, c.POINTER[struct_ibv_qp], c.POINTER[union_ibv_gid], uint16_t)
def ibv_attach_mcast(qp:c.POINTER[struct_ibv_qp], gid:c.POINTER[union_ibv_gid], lid:uint16_t) -> int: ...
@dll.bind(ctypes.c_int32, c.POINTER[struct_ibv_qp], c.POINTER[union_ibv_gid], uint16_t)
def ibv_detach_mcast(qp:c.POINTER[struct_ibv_qp], gid:c.POINTER[union_ibv_gid], lid:uint16_t) -> int: ...
@dll.bind(ctypes.c_int32)
def ibv_fork_init() -> int: ...
@dll.bind(ctypes.c_uint32)
def ibv_is_fork_initialized() -> ctypes.c_uint32: ...
@dll.bind(c.POINTER[ctypes.c_char], ctypes.c_int32)
def ibv_node_type_str(node_type:ctypes.c_int32) -> c.POINTER[ctypes.c_char]: ...
@dll.bind(c.POINTER[ctypes.c_char], ctypes.c_uint32)
def ibv_port_state_str(port_state:ctypes.c_uint32) -> c.POINTER[ctypes.c_char]: ...
@dll.bind(c.POINTER[ctypes.c_char], ctypes.c_uint32)
def ibv_event_type_str(event:ctypes.c_uint32) -> c.POINTER[ctypes.c_char]: ...
@dll.bind(ctypes.c_int32, c.POINTER[struct_ibv_context], c.POINTER[struct_ibv_ah_attr], c.Array[uint8_t, Literal[6]], c.POINTER[uint16_t])
def ibv_resolve_eth_l2_from_gid(context:c.POINTER[struct_ibv_context], attr:c.POINTER[struct_ibv_ah_attr], eth_mac:c.Array[uint8_t, Literal[6]], vid:c.POINTER[uint16_t]) -> int: ...
@dll.bind(ctypes.c_int32, c.POINTER[struct_ibv_qp], c.POINTER[struct_ibv_ece])
def ibv_set_ece(qp:c.POINTER[struct_ibv_qp], ece:c.POINTER[struct_ibv_ece]) -> int: ...
@dll.bind(ctypes.c_int32, c.POINTER[struct_ibv_qp], c.POINTER[struct_ibv_ece])
def ibv_query_ece(qp:c.POINTER[struct_ibv_qp], ece:c.POINTER[struct_ibv_ece]) -> int: ...
enum_ib_uverbs_core_support: dict[int, str] = {(IB_UVERBS_CORE_SUPPORT_OPTIONAL_MR_ACCESS:=1): 'IB_UVERBS_CORE_SUPPORT_OPTIONAL_MR_ACCESS'}
enum_ib_uverbs_access_flags: dict[int, str] = {(IB_UVERBS_ACCESS_LOCAL_WRITE:=1): 'IB_UVERBS_ACCESS_LOCAL_WRITE', (IB_UVERBS_ACCESS_REMOTE_WRITE:=2): 'IB_UVERBS_ACCESS_REMOTE_WRITE', (IB_UVERBS_ACCESS_REMOTE_READ:=4): 'IB_UVERBS_ACCESS_REMOTE_READ', (IB_UVERBS_ACCESS_REMOTE_ATOMIC:=8): 'IB_UVERBS_ACCESS_REMOTE_ATOMIC', (IB_UVERBS_ACCESS_MW_BIND:=16): 'IB_UVERBS_ACCESS_MW_BIND', (IB_UVERBS_ACCESS_ZERO_BASED:=32): 'IB_UVERBS_ACCESS_ZERO_BASED', (IB_UVERBS_ACCESS_ON_DEMAND:=64): 'IB_UVERBS_ACCESS_ON_DEMAND', (IB_UVERBS_ACCESS_HUGETLB:=128): 'IB_UVERBS_ACCESS_HUGETLB', (IB_UVERBS_ACCESS_FLUSH_GLOBAL:=256): 'IB_UVERBS_ACCESS_FLUSH_GLOBAL', (IB_UVERBS_ACCESS_FLUSH_PERSISTENT:=512): 'IB_UVERBS_ACCESS_FLUSH_PERSISTENT', (IB_UVERBS_ACCESS_RELAXED_ORDERING:=1048576): 'IB_UVERBS_ACCESS_RELAXED_ORDERING', (IB_UVERBS_ACCESS_OPTIONAL_RANGE:=1072693248): 'IB_UVERBS_ACCESS_OPTIONAL_RANGE'}
enum_ib_uverbs_srq_type: dict[int, str] = {(IB_UVERBS_SRQT_BASIC:=0): 'IB_UVERBS_SRQT_BASIC', (IB_UVERBS_SRQT_XRC:=1): 'IB_UVERBS_SRQT_XRC', (IB_UVERBS_SRQT_TM:=2): 'IB_UVERBS_SRQT_TM'}
enum_ib_uverbs_wq_type: dict[int, str] = {(IB_UVERBS_WQT_RQ:=0): 'IB_UVERBS_WQT_RQ'}
enum_ib_uverbs_wq_flags: dict[int, str] = {(IB_UVERBS_WQ_FLAGS_CVLAN_STRIPPING:=1): 'IB_UVERBS_WQ_FLAGS_CVLAN_STRIPPING', (IB_UVERBS_WQ_FLAGS_SCATTER_FCS:=2): 'IB_UVERBS_WQ_FLAGS_SCATTER_FCS', (IB_UVERBS_WQ_FLAGS_DELAY_DROP:=4): 'IB_UVERBS_WQ_FLAGS_DELAY_DROP', (IB_UVERBS_WQ_FLAGS_PCI_WRITE_END_PADDING:=8): 'IB_UVERBS_WQ_FLAGS_PCI_WRITE_END_PADDING'}
enum_ib_uverbs_qp_type: dict[int, str] = {(IB_UVERBS_QPT_RC:=2): 'IB_UVERBS_QPT_RC', (IB_UVERBS_QPT_UC:=3): 'IB_UVERBS_QPT_UC', (IB_UVERBS_QPT_UD:=4): 'IB_UVERBS_QPT_UD', (IB_UVERBS_QPT_RAW_PACKET:=8): 'IB_UVERBS_QPT_RAW_PACKET', (IB_UVERBS_QPT_XRC_INI:=9): 'IB_UVERBS_QPT_XRC_INI', (IB_UVERBS_QPT_XRC_TGT:=10): 'IB_UVERBS_QPT_XRC_TGT', (IB_UVERBS_QPT_DRIVER:=255): 'IB_UVERBS_QPT_DRIVER'}
enum_ib_uverbs_qp_create_flags: dict[int, str] = {(IB_UVERBS_QP_CREATE_BLOCK_MULTICAST_LOOPBACK:=2): 'IB_UVERBS_QP_CREATE_BLOCK_MULTICAST_LOOPBACK', (IB_UVERBS_QP_CREATE_SCATTER_FCS:=256): 'IB_UVERBS_QP_CREATE_SCATTER_FCS', (IB_UVERBS_QP_CREATE_CVLAN_STRIPPING:=512): 'IB_UVERBS_QP_CREATE_CVLAN_STRIPPING', (IB_UVERBS_QP_CREATE_PCI_WRITE_END_PADDING:=2048): 'IB_UVERBS_QP_CREATE_PCI_WRITE_END_PADDING', (IB_UVERBS_QP_CREATE_SQ_SIG_ALL:=4096): 'IB_UVERBS_QP_CREATE_SQ_SIG_ALL'}
enum_ib_uverbs_query_port_cap_flags: dict[int, str] = {(IB_UVERBS_PCF_SM:=2): 'IB_UVERBS_PCF_SM', (IB_UVERBS_PCF_NOTICE_SUP:=4): 'IB_UVERBS_PCF_NOTICE_SUP', (IB_UVERBS_PCF_TRAP_SUP:=8): 'IB_UVERBS_PCF_TRAP_SUP', (IB_UVERBS_PCF_OPT_IPD_SUP:=16): 'IB_UVERBS_PCF_OPT_IPD_SUP', (IB_UVERBS_PCF_AUTO_MIGR_SUP:=32): 'IB_UVERBS_PCF_AUTO_MIGR_SUP', (IB_UVERBS_PCF_SL_MAP_SUP:=64): 'IB_UVERBS_PCF_SL_MAP_SUP', (IB_UVERBS_PCF_MKEY_NVRAM:=128): 'IB_UVERBS_PCF_MKEY_NVRAM', (IB_UVERBS_PCF_PKEY_NVRAM:=256): 'IB_UVERBS_PCF_PKEY_NVRAM', (IB_UVERBS_PCF_LED_INFO_SUP:=512): 'IB_UVERBS_PCF_LED_INFO_SUP', (IB_UVERBS_PCF_SM_DISABLED:=1024): 'IB_UVERBS_PCF_SM_DISABLED', (IB_UVERBS_PCF_SYS_IMAGE_GUID_SUP:=2048): 'IB_UVERBS_PCF_SYS_IMAGE_GUID_SUP', (IB_UVERBS_PCF_PKEY_SW_EXT_PORT_TRAP_SUP:=4096): 'IB_UVERBS_PCF_PKEY_SW_EXT_PORT_TRAP_SUP', (IB_UVERBS_PCF_EXTENDED_SPEEDS_SUP:=16384): 'IB_UVERBS_PCF_EXTENDED_SPEEDS_SUP', (IB_UVERBS_PCF_CM_SUP:=65536): 'IB_UVERBS_PCF_CM_SUP', (IB_UVERBS_PCF_SNMP_TUNNEL_SUP:=131072): 'IB_UVERBS_PCF_SNMP_TUNNEL_SUP', (IB_UVERBS_PCF_REINIT_SUP:=262144): 'IB_UVERBS_PCF_REINIT_SUP', (IB_UVERBS_PCF_DEVICE_MGMT_SUP:=524288): 'IB_UVERBS_PCF_DEVICE_MGMT_SUP', (IB_UVERBS_PCF_VENDOR_CLASS_SUP:=1048576): 'IB_UVERBS_PCF_VENDOR_CLASS_SUP', (IB_UVERBS_PCF_DR_NOTICE_SUP:=2097152): 'IB_UVERBS_PCF_DR_NOTICE_SUP', (IB_UVERBS_PCF_CAP_MASK_NOTICE_SUP:=4194304): 'IB_UVERBS_PCF_CAP_MASK_NOTICE_SUP', (IB_UVERBS_PCF_BOOT_MGMT_SUP:=8388608): 'IB_UVERBS_PCF_BOOT_MGMT_SUP', (IB_UVERBS_PCF_LINK_LATENCY_SUP:=16777216): 'IB_UVERBS_PCF_LINK_LATENCY_SUP', (IB_UVERBS_PCF_CLIENT_REG_SUP:=33554432): 'IB_UVERBS_PCF_CLIENT_REG_SUP', (IB_UVERBS_PCF_LINK_SPEED_WIDTH_TABLE_SUP:=134217728): 'IB_UVERBS_PCF_LINK_SPEED_WIDTH_TABLE_SUP', (IB_UVERBS_PCF_VENDOR_SPECIFIC_MADS_TABLE_SUP:=268435456): 'IB_UVERBS_PCF_VENDOR_SPECIFIC_MADS_TABLE_SUP', (IB_UVERBS_PCF_MCAST_PKEY_TRAP_SUPPRESSION_SUP:=536870912): 'IB_UVERBS_PCF_MCAST_PKEY_TRAP_SUPPRESSION_SUP', (IB_UVERBS_PCF_MCAST_FDB_TOP_SUP:=1073741824): 'IB_UVERBS_PCF_MCAST_FDB_TOP_SUP', (IB_UVERBS_PCF_HIERARCHY_INFO_SUP:=2147483648): 'IB_UVERBS_PCF_HIERARCHY_INFO_SUP', (IB_UVERBS_PCF_IP_BASED_GIDS:=67108864): 'IB_UVERBS_PCF_IP_BASED_GIDS'}
enum_ib_uverbs_query_port_flags: dict[int, str] = {(IB_UVERBS_QPF_GRH_REQUIRED:=1): 'IB_UVERBS_QPF_GRH_REQUIRED'}
enum_ib_uverbs_flow_action_esp_keymat_aes_gcm_iv_algo: dict[int, str] = {(IB_UVERBS_FLOW_ACTION_IV_ALGO_SEQ:=0): 'IB_UVERBS_FLOW_ACTION_IV_ALGO_SEQ'}
@c.record
class struct_ib_uverbs_flow_action_esp_keymat_aes_gcm(c.Struct):
  SIZE = 56
  iv: int
  iv_algo: int
  salt: int
  icv_len: int
  key_len: int
  aes_key: c.Array[ctypes.c_uint32, Literal[8]]
struct_ib_uverbs_flow_action_esp_keymat_aes_gcm.register_fields([('iv', ctypes.c_uint64, 0), ('iv_algo', ctypes.c_uint32, 8), ('salt', ctypes.c_uint32, 12), ('icv_len', ctypes.c_uint32, 16), ('key_len', ctypes.c_uint32, 20), ('aes_key', c.Array[ctypes.c_uint32, Literal[8]], 24)])
@c.record
class struct_ib_uverbs_flow_action_esp_replay_bmp(c.Struct):
  SIZE = 4
  size: int
struct_ib_uverbs_flow_action_esp_replay_bmp.register_fields([('size', ctypes.c_uint32, 0)])
enum_ib_uverbs_flow_action_esp_flags: dict[int, str] = {(IB_UVERBS_FLOW_ACTION_ESP_FLAGS_INLINE_CRYPTO:=0): 'IB_UVERBS_FLOW_ACTION_ESP_FLAGS_INLINE_CRYPTO', (IB_UVERBS_FLOW_ACTION_ESP_FLAGS_FULL_OFFLOAD:=1): 'IB_UVERBS_FLOW_ACTION_ESP_FLAGS_FULL_OFFLOAD', (IB_UVERBS_FLOW_ACTION_ESP_FLAGS_TUNNEL:=0): 'IB_UVERBS_FLOW_ACTION_ESP_FLAGS_TUNNEL', (IB_UVERBS_FLOW_ACTION_ESP_FLAGS_TRANSPORT:=2): 'IB_UVERBS_FLOW_ACTION_ESP_FLAGS_TRANSPORT', (IB_UVERBS_FLOW_ACTION_ESP_FLAGS_DECRYPT:=0): 'IB_UVERBS_FLOW_ACTION_ESP_FLAGS_DECRYPT', (IB_UVERBS_FLOW_ACTION_ESP_FLAGS_ENCRYPT:=4): 'IB_UVERBS_FLOW_ACTION_ESP_FLAGS_ENCRYPT', (IB_UVERBS_FLOW_ACTION_ESP_FLAGS_ESN_NEW_WINDOW:=8): 'IB_UVERBS_FLOW_ACTION_ESP_FLAGS_ESN_NEW_WINDOW'}
enum_ib_uverbs_read_counters_flags: dict[int, str] = {(IB_UVERBS_READ_COUNTERS_PREFER_CACHED:=1): 'IB_UVERBS_READ_COUNTERS_PREFER_CACHED'}
enum_ib_uverbs_advise_mr_flag: dict[int, str] = {(IB_UVERBS_ADVISE_MR_FLAG_FLUSH:=1): 'IB_UVERBS_ADVISE_MR_FLAG_FLUSH'}
@c.record
class struct_ib_uverbs_query_port_resp_ex(c.Struct):
  SIZE = 48
  legacy_resp: struct_ib_uverbs_query_port_resp
  port_cap_flags2: int
  reserved: c.Array[ctypes.c_ubyte, Literal[2]]
  active_speed_ex: int
@c.record
class struct_ib_uverbs_query_port_resp(c.Struct):
  SIZE = 40
  port_cap_flags: int
  max_msg_sz: int
  bad_pkey_cntr: int
  qkey_viol_cntr: int
  gid_tbl_len: int
  pkey_tbl_len: int
  lid: int
  sm_lid: int
  state: int
  max_mtu: int
  active_mtu: int
  lmc: int
  max_vl_num: int
  sm_sl: int
  subnet_timeout: int
  init_type_reply: int
  active_width: int
  active_speed: int
  phys_state: int
  link_layer: int
  flags: int
  reserved: int
__u8: TypeAlias = ctypes.c_ubyte
struct_ib_uverbs_query_port_resp.register_fields([('port_cap_flags', ctypes.c_uint32, 0), ('max_msg_sz', ctypes.c_uint32, 4), ('bad_pkey_cntr', ctypes.c_uint32, 8), ('qkey_viol_cntr', ctypes.c_uint32, 12), ('gid_tbl_len', ctypes.c_uint32, 16), ('pkey_tbl_len', ctypes.c_uint16, 20), ('lid', ctypes.c_uint16, 22), ('sm_lid', ctypes.c_uint16, 24), ('state', ctypes.c_ubyte, 26), ('max_mtu', ctypes.c_ubyte, 27), ('active_mtu', ctypes.c_ubyte, 28), ('lmc', ctypes.c_ubyte, 29), ('max_vl_num', ctypes.c_ubyte, 30), ('sm_sl', ctypes.c_ubyte, 31), ('subnet_timeout', ctypes.c_ubyte, 32), ('init_type_reply', ctypes.c_ubyte, 33), ('active_width', ctypes.c_ubyte, 34), ('active_speed', ctypes.c_ubyte, 35), ('phys_state', ctypes.c_ubyte, 36), ('link_layer', ctypes.c_ubyte, 37), ('flags', ctypes.c_ubyte, 38), ('reserved', ctypes.c_ubyte, 39)])
struct_ib_uverbs_query_port_resp_ex.register_fields([('legacy_resp', struct_ib_uverbs_query_port_resp, 0), ('port_cap_flags2', ctypes.c_uint16, 40), ('reserved', c.Array[ctypes.c_ubyte, Literal[2]], 42), ('active_speed_ex', ctypes.c_uint32, 44)])
@c.record
class struct_ib_uverbs_qp_cap(c.Struct):
  SIZE = 20
  max_send_wr: int
  max_recv_wr: int
  max_send_sge: int
  max_recv_sge: int
  max_inline_data: int
struct_ib_uverbs_qp_cap.register_fields([('max_send_wr', ctypes.c_uint32, 0), ('max_recv_wr', ctypes.c_uint32, 4), ('max_send_sge', ctypes.c_uint32, 8), ('max_recv_sge', ctypes.c_uint32, 12), ('max_inline_data', ctypes.c_uint32, 16)])
enum_rdma_driver_id: dict[int, str] = {(RDMA_DRIVER_UNKNOWN:=0): 'RDMA_DRIVER_UNKNOWN', (RDMA_DRIVER_MLX5:=1): 'RDMA_DRIVER_MLX5', (RDMA_DRIVER_MLX4:=2): 'RDMA_DRIVER_MLX4', (RDMA_DRIVER_CXGB3:=3): 'RDMA_DRIVER_CXGB3', (RDMA_DRIVER_CXGB4:=4): 'RDMA_DRIVER_CXGB4', (RDMA_DRIVER_MTHCA:=5): 'RDMA_DRIVER_MTHCA', (RDMA_DRIVER_BNXT_RE:=6): 'RDMA_DRIVER_BNXT_RE', (RDMA_DRIVER_OCRDMA:=7): 'RDMA_DRIVER_OCRDMA', (RDMA_DRIVER_NES:=8): 'RDMA_DRIVER_NES', (RDMA_DRIVER_I40IW:=9): 'RDMA_DRIVER_I40IW', (RDMA_DRIVER_IRDMA:=9): 'RDMA_DRIVER_IRDMA', (RDMA_DRIVER_VMW_PVRDMA:=10): 'RDMA_DRIVER_VMW_PVRDMA', (RDMA_DRIVER_QEDR:=11): 'RDMA_DRIVER_QEDR', (RDMA_DRIVER_HNS:=12): 'RDMA_DRIVER_HNS', (RDMA_DRIVER_USNIC:=13): 'RDMA_DRIVER_USNIC', (RDMA_DRIVER_RXE:=14): 'RDMA_DRIVER_RXE', (RDMA_DRIVER_HFI1:=15): 'RDMA_DRIVER_HFI1', (RDMA_DRIVER_QIB:=16): 'RDMA_DRIVER_QIB', (RDMA_DRIVER_EFA:=17): 'RDMA_DRIVER_EFA', (RDMA_DRIVER_SIW:=18): 'RDMA_DRIVER_SIW', (RDMA_DRIVER_ERDMA:=19): 'RDMA_DRIVER_ERDMA', (RDMA_DRIVER_MANA:=20): 'RDMA_DRIVER_MANA'}
enum_ib_uverbs_gid_type: dict[int, str] = {(IB_UVERBS_GID_TYPE_IB:=0): 'IB_UVERBS_GID_TYPE_IB', (IB_UVERBS_GID_TYPE_ROCE_V1:=1): 'IB_UVERBS_GID_TYPE_ROCE_V1', (IB_UVERBS_GID_TYPE_ROCE_V2:=2): 'IB_UVERBS_GID_TYPE_ROCE_V2'}
@c.record
class struct_ib_uverbs_gid_entry(c.Struct):
  SIZE = 32
  gid: c.Array[ctypes.c_uint64, Literal[2]]
  gid_index: int
  port_num: int
  gid_type: int
  netdev_ifindex: int
struct_ib_uverbs_gid_entry.register_fields([('gid', c.Array[ctypes.c_uint64, Literal[2]], 0), ('gid_index', ctypes.c_uint32, 16), ('port_num', ctypes.c_uint32, 20), ('gid_type', ctypes.c_uint32, 24), ('netdev_ifindex', ctypes.c_uint32, 28)])
enum_ib_uverbs_write_cmds: dict[int, str] = {(IB_USER_VERBS_CMD_GET_CONTEXT:=0): 'IB_USER_VERBS_CMD_GET_CONTEXT', (IB_USER_VERBS_CMD_QUERY_DEVICE:=1): 'IB_USER_VERBS_CMD_QUERY_DEVICE', (IB_USER_VERBS_CMD_QUERY_PORT:=2): 'IB_USER_VERBS_CMD_QUERY_PORT', (IB_USER_VERBS_CMD_ALLOC_PD:=3): 'IB_USER_VERBS_CMD_ALLOC_PD', (IB_USER_VERBS_CMD_DEALLOC_PD:=4): 'IB_USER_VERBS_CMD_DEALLOC_PD', (IB_USER_VERBS_CMD_CREATE_AH:=5): 'IB_USER_VERBS_CMD_CREATE_AH', (IB_USER_VERBS_CMD_MODIFY_AH:=6): 'IB_USER_VERBS_CMD_MODIFY_AH', (IB_USER_VERBS_CMD_QUERY_AH:=7): 'IB_USER_VERBS_CMD_QUERY_AH', (IB_USER_VERBS_CMD_DESTROY_AH:=8): 'IB_USER_VERBS_CMD_DESTROY_AH', (IB_USER_VERBS_CMD_REG_MR:=9): 'IB_USER_VERBS_CMD_REG_MR', (IB_USER_VERBS_CMD_REG_SMR:=10): 'IB_USER_VERBS_CMD_REG_SMR', (IB_USER_VERBS_CMD_REREG_MR:=11): 'IB_USER_VERBS_CMD_REREG_MR', (IB_USER_VERBS_CMD_QUERY_MR:=12): 'IB_USER_VERBS_CMD_QUERY_MR', (IB_USER_VERBS_CMD_DEREG_MR:=13): 'IB_USER_VERBS_CMD_DEREG_MR', (IB_USER_VERBS_CMD_ALLOC_MW:=14): 'IB_USER_VERBS_CMD_ALLOC_MW', (IB_USER_VERBS_CMD_BIND_MW:=15): 'IB_USER_VERBS_CMD_BIND_MW', (IB_USER_VERBS_CMD_DEALLOC_MW:=16): 'IB_USER_VERBS_CMD_DEALLOC_MW', (IB_USER_VERBS_CMD_CREATE_COMP_CHANNEL:=17): 'IB_USER_VERBS_CMD_CREATE_COMP_CHANNEL', (IB_USER_VERBS_CMD_CREATE_CQ:=18): 'IB_USER_VERBS_CMD_CREATE_CQ', (IB_USER_VERBS_CMD_RESIZE_CQ:=19): 'IB_USER_VERBS_CMD_RESIZE_CQ', (IB_USER_VERBS_CMD_DESTROY_CQ:=20): 'IB_USER_VERBS_CMD_DESTROY_CQ', (IB_USER_VERBS_CMD_POLL_CQ:=21): 'IB_USER_VERBS_CMD_POLL_CQ', (IB_USER_VERBS_CMD_PEEK_CQ:=22): 'IB_USER_VERBS_CMD_PEEK_CQ', (IB_USER_VERBS_CMD_REQ_NOTIFY_CQ:=23): 'IB_USER_VERBS_CMD_REQ_NOTIFY_CQ', (IB_USER_VERBS_CMD_CREATE_QP:=24): 'IB_USER_VERBS_CMD_CREATE_QP', (IB_USER_VERBS_CMD_QUERY_QP:=25): 'IB_USER_VERBS_CMD_QUERY_QP', (IB_USER_VERBS_CMD_MODIFY_QP:=26): 'IB_USER_VERBS_CMD_MODIFY_QP', (IB_USER_VERBS_CMD_DESTROY_QP:=27): 'IB_USER_VERBS_CMD_DESTROY_QP', (IB_USER_VERBS_CMD_POST_SEND:=28): 'IB_USER_VERBS_CMD_POST_SEND', (IB_USER_VERBS_CMD_POST_RECV:=29): 'IB_USER_VERBS_CMD_POST_RECV', (IB_USER_VERBS_CMD_ATTACH_MCAST:=30): 'IB_USER_VERBS_CMD_ATTACH_MCAST', (IB_USER_VERBS_CMD_DETACH_MCAST:=31): 'IB_USER_VERBS_CMD_DETACH_MCAST', (IB_USER_VERBS_CMD_CREATE_SRQ:=32): 'IB_USER_VERBS_CMD_CREATE_SRQ', (IB_USER_VERBS_CMD_MODIFY_SRQ:=33): 'IB_USER_VERBS_CMD_MODIFY_SRQ', (IB_USER_VERBS_CMD_QUERY_SRQ:=34): 'IB_USER_VERBS_CMD_QUERY_SRQ', (IB_USER_VERBS_CMD_DESTROY_SRQ:=35): 'IB_USER_VERBS_CMD_DESTROY_SRQ', (IB_USER_VERBS_CMD_POST_SRQ_RECV:=36): 'IB_USER_VERBS_CMD_POST_SRQ_RECV', (IB_USER_VERBS_CMD_OPEN_XRCD:=37): 'IB_USER_VERBS_CMD_OPEN_XRCD', (IB_USER_VERBS_CMD_CLOSE_XRCD:=38): 'IB_USER_VERBS_CMD_CLOSE_XRCD', (IB_USER_VERBS_CMD_CREATE_XSRQ:=39): 'IB_USER_VERBS_CMD_CREATE_XSRQ', (IB_USER_VERBS_CMD_OPEN_QP:=40): 'IB_USER_VERBS_CMD_OPEN_QP'}
_anonenum5: dict[int, str] = {(IB_USER_VERBS_EX_CMD_QUERY_DEVICE:=1): 'IB_USER_VERBS_EX_CMD_QUERY_DEVICE', (IB_USER_VERBS_EX_CMD_CREATE_CQ:=18): 'IB_USER_VERBS_EX_CMD_CREATE_CQ', (IB_USER_VERBS_EX_CMD_CREATE_QP:=24): 'IB_USER_VERBS_EX_CMD_CREATE_QP', (IB_USER_VERBS_EX_CMD_MODIFY_QP:=26): 'IB_USER_VERBS_EX_CMD_MODIFY_QP', (IB_USER_VERBS_EX_CMD_CREATE_FLOW:=50): 'IB_USER_VERBS_EX_CMD_CREATE_FLOW', (IB_USER_VERBS_EX_CMD_DESTROY_FLOW:=51): 'IB_USER_VERBS_EX_CMD_DESTROY_FLOW', (IB_USER_VERBS_EX_CMD_CREATE_WQ:=52): 'IB_USER_VERBS_EX_CMD_CREATE_WQ', (IB_USER_VERBS_EX_CMD_MODIFY_WQ:=53): 'IB_USER_VERBS_EX_CMD_MODIFY_WQ', (IB_USER_VERBS_EX_CMD_DESTROY_WQ:=54): 'IB_USER_VERBS_EX_CMD_DESTROY_WQ', (IB_USER_VERBS_EX_CMD_CREATE_RWQ_IND_TBL:=55): 'IB_USER_VERBS_EX_CMD_CREATE_RWQ_IND_TBL', (IB_USER_VERBS_EX_CMD_DESTROY_RWQ_IND_TBL:=56): 'IB_USER_VERBS_EX_CMD_DESTROY_RWQ_IND_TBL', (IB_USER_VERBS_EX_CMD_MODIFY_CQ:=57): 'IB_USER_VERBS_EX_CMD_MODIFY_CQ'}
enum_ib_placement_type: dict[int, str] = {(IB_FLUSH_GLOBAL:=1): 'IB_FLUSH_GLOBAL', (IB_FLUSH_PERSISTENT:=2): 'IB_FLUSH_PERSISTENT'}
enum_ib_selectivity_level: dict[int, str] = {(IB_FLUSH_RANGE:=0): 'IB_FLUSH_RANGE', (IB_FLUSH_MR:=1): 'IB_FLUSH_MR'}
@c.record
class struct_ib_uverbs_async_event_desc(c.Struct):
  SIZE = 16
  element: int
  event_type: int
  reserved: int
struct_ib_uverbs_async_event_desc.register_fields([('element', ctypes.c_uint64, 0), ('event_type', ctypes.c_uint32, 8), ('reserved', ctypes.c_uint32, 12)])
@c.record
class struct_ib_uverbs_comp_event_desc(c.Struct):
  SIZE = 8
  cq_handle: int
struct_ib_uverbs_comp_event_desc.register_fields([('cq_handle', ctypes.c_uint64, 0)])
@c.record
class struct_ib_uverbs_cq_moderation_caps(c.Struct):
  SIZE = 8
  max_cq_moderation_count: int
  max_cq_moderation_period: int
  reserved: int
struct_ib_uverbs_cq_moderation_caps.register_fields([('max_cq_moderation_count', ctypes.c_uint16, 0), ('max_cq_moderation_period', ctypes.c_uint16, 2), ('reserved', ctypes.c_uint32, 4)])
@c.record
class struct_ib_uverbs_cmd_hdr(c.Struct):
  SIZE = 8
  command: int
  in_words: int
  out_words: int
struct_ib_uverbs_cmd_hdr.register_fields([('command', ctypes.c_uint32, 0), ('in_words', ctypes.c_uint16, 4), ('out_words', ctypes.c_uint16, 6)])
@c.record
class struct_ib_uverbs_ex_cmd_hdr(c.Struct):
  SIZE = 16
  response: int
  provider_in_words: int
  provider_out_words: int
  cmd_hdr_reserved: int
struct_ib_uverbs_ex_cmd_hdr.register_fields([('response', ctypes.c_uint64, 0), ('provider_in_words', ctypes.c_uint16, 8), ('provider_out_words', ctypes.c_uint16, 10), ('cmd_hdr_reserved', ctypes.c_uint32, 12)])
@c.record
class struct_ib_uverbs_get_context(c.Struct):
  SIZE = 8
  response: int
  driver_data: c.Array[ctypes.c_uint64, Literal[0]]
struct_ib_uverbs_get_context.register_fields([('response', ctypes.c_uint64, 0), ('driver_data', c.Array[ctypes.c_uint64, Literal[0]], 8)])
@c.record
class struct_ib_uverbs_get_context_resp(c.Struct):
  SIZE = 8
  async_fd: int
  num_comp_vectors: int
  driver_data: c.Array[ctypes.c_uint64, Literal[0]]
struct_ib_uverbs_get_context_resp.register_fields([('async_fd', ctypes.c_uint32, 0), ('num_comp_vectors', ctypes.c_uint32, 4), ('driver_data', c.Array[ctypes.c_uint64, Literal[0]], 8)])
@c.record
class struct_ib_uverbs_query_device(c.Struct):
  SIZE = 8
  response: int
  driver_data: c.Array[ctypes.c_uint64, Literal[0]]
struct_ib_uverbs_query_device.register_fields([('response', ctypes.c_uint64, 0), ('driver_data', c.Array[ctypes.c_uint64, Literal[0]], 8)])
@c.record
class struct_ib_uverbs_query_device_resp(c.Struct):
  SIZE = 176
  fw_ver: int
  node_guid: int
  sys_image_guid: int
  max_mr_size: int
  page_size_cap: int
  vendor_id: int
  vendor_part_id: int
  hw_ver: int
  max_qp: int
  max_qp_wr: int
  device_cap_flags: int
  max_sge: int
  max_sge_rd: int
  max_cq: int
  max_cqe: int
  max_mr: int
  max_pd: int
  max_qp_rd_atom: int
  max_ee_rd_atom: int
  max_res_rd_atom: int
  max_qp_init_rd_atom: int
  max_ee_init_rd_atom: int
  atomic_cap: int
  max_ee: int
  max_rdd: int
  max_mw: int
  max_raw_ipv6_qp: int
  max_raw_ethy_qp: int
  max_mcast_grp: int
  max_mcast_qp_attach: int
  max_total_mcast_qp_attach: int
  max_ah: int
  max_fmr: int
  max_map_per_fmr: int
  max_srq: int
  max_srq_wr: int
  max_srq_sge: int
  max_pkeys: int
  local_ca_ack_delay: int
  phys_port_cnt: int
  reserved: c.Array[ctypes.c_ubyte, Literal[4]]
struct_ib_uverbs_query_device_resp.register_fields([('fw_ver', ctypes.c_uint64, 0), ('node_guid', ctypes.c_uint64, 8), ('sys_image_guid', ctypes.c_uint64, 16), ('max_mr_size', ctypes.c_uint64, 24), ('page_size_cap', ctypes.c_uint64, 32), ('vendor_id', ctypes.c_uint32, 40), ('vendor_part_id', ctypes.c_uint32, 44), ('hw_ver', ctypes.c_uint32, 48), ('max_qp', ctypes.c_uint32, 52), ('max_qp_wr', ctypes.c_uint32, 56), ('device_cap_flags', ctypes.c_uint32, 60), ('max_sge', ctypes.c_uint32, 64), ('max_sge_rd', ctypes.c_uint32, 68), ('max_cq', ctypes.c_uint32, 72), ('max_cqe', ctypes.c_uint32, 76), ('max_mr', ctypes.c_uint32, 80), ('max_pd', ctypes.c_uint32, 84), ('max_qp_rd_atom', ctypes.c_uint32, 88), ('max_ee_rd_atom', ctypes.c_uint32, 92), ('max_res_rd_atom', ctypes.c_uint32, 96), ('max_qp_init_rd_atom', ctypes.c_uint32, 100), ('max_ee_init_rd_atom', ctypes.c_uint32, 104), ('atomic_cap', ctypes.c_uint32, 108), ('max_ee', ctypes.c_uint32, 112), ('max_rdd', ctypes.c_uint32, 116), ('max_mw', ctypes.c_uint32, 120), ('max_raw_ipv6_qp', ctypes.c_uint32, 124), ('max_raw_ethy_qp', ctypes.c_uint32, 128), ('max_mcast_grp', ctypes.c_uint32, 132), ('max_mcast_qp_attach', ctypes.c_uint32, 136), ('max_total_mcast_qp_attach', ctypes.c_uint32, 140), ('max_ah', ctypes.c_uint32, 144), ('max_fmr', ctypes.c_uint32, 148), ('max_map_per_fmr', ctypes.c_uint32, 152), ('max_srq', ctypes.c_uint32, 156), ('max_srq_wr', ctypes.c_uint32, 160), ('max_srq_sge', ctypes.c_uint32, 164), ('max_pkeys', ctypes.c_uint16, 168), ('local_ca_ack_delay', ctypes.c_ubyte, 170), ('phys_port_cnt', ctypes.c_ubyte, 171), ('reserved', c.Array[ctypes.c_ubyte, Literal[4]], 172)])
@c.record
class struct_ib_uverbs_ex_query_device(c.Struct):
  SIZE = 8
  comp_mask: int
  reserved: int
struct_ib_uverbs_ex_query_device.register_fields([('comp_mask', ctypes.c_uint32, 0), ('reserved', ctypes.c_uint32, 4)])
@c.record
class struct_ib_uverbs_odp_caps(c.Struct):
  SIZE = 24
  general_caps: int
  per_transport_caps: struct_ib_uverbs_odp_caps_per_transport_caps
  reserved: int
@c.record
class struct_ib_uverbs_odp_caps_per_transport_caps(c.Struct):
  SIZE = 12
  rc_odp_caps: int
  uc_odp_caps: int
  ud_odp_caps: int
struct_ib_uverbs_odp_caps_per_transport_caps.register_fields([('rc_odp_caps', ctypes.c_uint32, 0), ('uc_odp_caps', ctypes.c_uint32, 4), ('ud_odp_caps', ctypes.c_uint32, 8)])
struct_ib_uverbs_odp_caps.register_fields([('general_caps', ctypes.c_uint64, 0), ('per_transport_caps', struct_ib_uverbs_odp_caps_per_transport_caps, 8), ('reserved', ctypes.c_uint32, 20)])
@c.record
class struct_ib_uverbs_rss_caps(c.Struct):
  SIZE = 16
  supported_qpts: int
  max_rwq_indirection_tables: int
  max_rwq_indirection_table_size: int
  reserved: int
struct_ib_uverbs_rss_caps.register_fields([('supported_qpts', ctypes.c_uint32, 0), ('max_rwq_indirection_tables', ctypes.c_uint32, 4), ('max_rwq_indirection_table_size', ctypes.c_uint32, 8), ('reserved', ctypes.c_uint32, 12)])
@c.record
class struct_ib_uverbs_tm_caps(c.Struct):
  SIZE = 24
  max_rndv_hdr_size: int
  max_num_tags: int
  flags: int
  max_ops: int
  max_sge: int
  reserved: int
struct_ib_uverbs_tm_caps.register_fields([('max_rndv_hdr_size', ctypes.c_uint32, 0), ('max_num_tags', ctypes.c_uint32, 4), ('flags', ctypes.c_uint32, 8), ('max_ops', ctypes.c_uint32, 12), ('max_sge', ctypes.c_uint32, 16), ('reserved', ctypes.c_uint32, 20)])
@c.record
class struct_ib_uverbs_ex_query_device_resp(c.Struct):
  SIZE = 304
  base: struct_ib_uverbs_query_device_resp
  comp_mask: int
  response_length: int
  odp_caps: struct_ib_uverbs_odp_caps
  timestamp_mask: int
  hca_core_clock: int
  device_cap_flags_ex: int
  rss_caps: struct_ib_uverbs_rss_caps
  max_wq_type_rq: int
  raw_packet_caps: int
  tm_caps: struct_ib_uverbs_tm_caps
  cq_moderation_caps: struct_ib_uverbs_cq_moderation_caps
  max_dm_size: int
  xrc_odp_caps: int
  reserved: int
struct_ib_uverbs_ex_query_device_resp.register_fields([('base', struct_ib_uverbs_query_device_resp, 0), ('comp_mask', ctypes.c_uint32, 176), ('response_length', ctypes.c_uint32, 180), ('odp_caps', struct_ib_uverbs_odp_caps, 184), ('timestamp_mask', ctypes.c_uint64, 208), ('hca_core_clock', ctypes.c_uint64, 216), ('device_cap_flags_ex', ctypes.c_uint64, 224), ('rss_caps', struct_ib_uverbs_rss_caps, 232), ('max_wq_type_rq', ctypes.c_uint32, 248), ('raw_packet_caps', ctypes.c_uint32, 252), ('tm_caps', struct_ib_uverbs_tm_caps, 256), ('cq_moderation_caps', struct_ib_uverbs_cq_moderation_caps, 280), ('max_dm_size', ctypes.c_uint64, 288), ('xrc_odp_caps', ctypes.c_uint32, 296), ('reserved', ctypes.c_uint32, 300)])
@c.record
class struct_ib_uverbs_query_port(c.Struct):
  SIZE = 16
  response: int
  port_num: int
  reserved: c.Array[ctypes.c_ubyte, Literal[7]]
  driver_data: c.Array[ctypes.c_uint64, Literal[0]]
struct_ib_uverbs_query_port.register_fields([('response', ctypes.c_uint64, 0), ('port_num', ctypes.c_ubyte, 8), ('reserved', c.Array[ctypes.c_ubyte, Literal[7]], 9), ('driver_data', c.Array[ctypes.c_uint64, Literal[0]], 16)])
@c.record
class struct_ib_uverbs_alloc_pd(c.Struct):
  SIZE = 8
  response: int
  driver_data: c.Array[ctypes.c_uint64, Literal[0]]
struct_ib_uverbs_alloc_pd.register_fields([('response', ctypes.c_uint64, 0), ('driver_data', c.Array[ctypes.c_uint64, Literal[0]], 8)])
@c.record
class struct_ib_uverbs_alloc_pd_resp(c.Struct):
  SIZE = 4
  pd_handle: int
  driver_data: c.Array[ctypes.c_uint32, Literal[0]]
struct_ib_uverbs_alloc_pd_resp.register_fields([('pd_handle', ctypes.c_uint32, 0), ('driver_data', c.Array[ctypes.c_uint32, Literal[0]], 4)])
@c.record
class struct_ib_uverbs_dealloc_pd(c.Struct):
  SIZE = 4
  pd_handle: int
struct_ib_uverbs_dealloc_pd.register_fields([('pd_handle', ctypes.c_uint32, 0)])
@c.record
class struct_ib_uverbs_open_xrcd(c.Struct):
  SIZE = 16
  response: int
  fd: int
  oflags: int
  driver_data: c.Array[ctypes.c_uint64, Literal[0]]
struct_ib_uverbs_open_xrcd.register_fields([('response', ctypes.c_uint64, 0), ('fd', ctypes.c_uint32, 8), ('oflags', ctypes.c_uint32, 12), ('driver_data', c.Array[ctypes.c_uint64, Literal[0]], 16)])
@c.record
class struct_ib_uverbs_open_xrcd_resp(c.Struct):
  SIZE = 4
  xrcd_handle: int
  driver_data: c.Array[ctypes.c_uint32, Literal[0]]
struct_ib_uverbs_open_xrcd_resp.register_fields([('xrcd_handle', ctypes.c_uint32, 0), ('driver_data', c.Array[ctypes.c_uint32, Literal[0]], 4)])
@c.record
class struct_ib_uverbs_close_xrcd(c.Struct):
  SIZE = 4
  xrcd_handle: int
struct_ib_uverbs_close_xrcd.register_fields([('xrcd_handle', ctypes.c_uint32, 0)])
@c.record
class struct_ib_uverbs_reg_mr(c.Struct):
  SIZE = 40
  response: int
  start: int
  length: int
  hca_va: int
  pd_handle: int
  access_flags: int
  driver_data: c.Array[ctypes.c_uint64, Literal[0]]
struct_ib_uverbs_reg_mr.register_fields([('response', ctypes.c_uint64, 0), ('start', ctypes.c_uint64, 8), ('length', ctypes.c_uint64, 16), ('hca_va', ctypes.c_uint64, 24), ('pd_handle', ctypes.c_uint32, 32), ('access_flags', ctypes.c_uint32, 36), ('driver_data', c.Array[ctypes.c_uint64, Literal[0]], 40)])
@c.record
class struct_ib_uverbs_reg_mr_resp(c.Struct):
  SIZE = 12
  mr_handle: int
  lkey: int
  rkey: int
  driver_data: c.Array[ctypes.c_uint32, Literal[0]]
struct_ib_uverbs_reg_mr_resp.register_fields([('mr_handle', ctypes.c_uint32, 0), ('lkey', ctypes.c_uint32, 4), ('rkey', ctypes.c_uint32, 8), ('driver_data', c.Array[ctypes.c_uint32, Literal[0]], 12)])
@c.record
class struct_ib_uverbs_rereg_mr(c.Struct):
  SIZE = 48
  response: int
  mr_handle: int
  flags: int
  start: int
  length: int
  hca_va: int
  pd_handle: int
  access_flags: int
  driver_data: c.Array[ctypes.c_uint64, Literal[0]]
struct_ib_uverbs_rereg_mr.register_fields([('response', ctypes.c_uint64, 0), ('mr_handle', ctypes.c_uint32, 8), ('flags', ctypes.c_uint32, 12), ('start', ctypes.c_uint64, 16), ('length', ctypes.c_uint64, 24), ('hca_va', ctypes.c_uint64, 32), ('pd_handle', ctypes.c_uint32, 40), ('access_flags', ctypes.c_uint32, 44), ('driver_data', c.Array[ctypes.c_uint64, Literal[0]], 48)])
@c.record
class struct_ib_uverbs_rereg_mr_resp(c.Struct):
  SIZE = 8
  lkey: int
  rkey: int
  driver_data: c.Array[ctypes.c_uint64, Literal[0]]
struct_ib_uverbs_rereg_mr_resp.register_fields([('lkey', ctypes.c_uint32, 0), ('rkey', ctypes.c_uint32, 4), ('driver_data', c.Array[ctypes.c_uint64, Literal[0]], 8)])
@c.record
class struct_ib_uverbs_dereg_mr(c.Struct):
  SIZE = 4
  mr_handle: int
struct_ib_uverbs_dereg_mr.register_fields([('mr_handle', ctypes.c_uint32, 0)])
@c.record
class struct_ib_uverbs_alloc_mw(c.Struct):
  SIZE = 16
  response: int
  pd_handle: int
  mw_type: int
  reserved: c.Array[ctypes.c_ubyte, Literal[3]]
  driver_data: c.Array[ctypes.c_uint64, Literal[0]]
struct_ib_uverbs_alloc_mw.register_fields([('response', ctypes.c_uint64, 0), ('pd_handle', ctypes.c_uint32, 8), ('mw_type', ctypes.c_ubyte, 12), ('reserved', c.Array[ctypes.c_ubyte, Literal[3]], 13), ('driver_data', c.Array[ctypes.c_uint64, Literal[0]], 16)])
@c.record
class struct_ib_uverbs_alloc_mw_resp(c.Struct):
  SIZE = 8
  mw_handle: int
  rkey: int
  driver_data: c.Array[ctypes.c_uint64, Literal[0]]
struct_ib_uverbs_alloc_mw_resp.register_fields([('mw_handle', ctypes.c_uint32, 0), ('rkey', ctypes.c_uint32, 4), ('driver_data', c.Array[ctypes.c_uint64, Literal[0]], 8)])
@c.record
class struct_ib_uverbs_dealloc_mw(c.Struct):
  SIZE = 4
  mw_handle: int
struct_ib_uverbs_dealloc_mw.register_fields([('mw_handle', ctypes.c_uint32, 0)])
@c.record
class struct_ib_uverbs_create_comp_channel(c.Struct):
  SIZE = 8
  response: int
struct_ib_uverbs_create_comp_channel.register_fields([('response', ctypes.c_uint64, 0)])
@c.record
class struct_ib_uverbs_create_comp_channel_resp(c.Struct):
  SIZE = 4
  fd: int
struct_ib_uverbs_create_comp_channel_resp.register_fields([('fd', ctypes.c_uint32, 0)])
@c.record
class struct_ib_uverbs_create_cq(c.Struct):
  SIZE = 32
  response: int
  user_handle: int
  cqe: int
  comp_vector: int
  comp_channel: int
  reserved: int
  driver_data: c.Array[ctypes.c_uint64, Literal[0]]
__s32: TypeAlias = ctypes.c_int32
struct_ib_uverbs_create_cq.register_fields([('response', ctypes.c_uint64, 0), ('user_handle', ctypes.c_uint64, 8), ('cqe', ctypes.c_uint32, 16), ('comp_vector', ctypes.c_uint32, 20), ('comp_channel', ctypes.c_int32, 24), ('reserved', ctypes.c_uint32, 28), ('driver_data', c.Array[ctypes.c_uint64, Literal[0]], 32)])
enum_ib_uverbs_ex_create_cq_flags: dict[int, str] = {(IB_UVERBS_CQ_FLAGS_TIMESTAMP_COMPLETION:=1): 'IB_UVERBS_CQ_FLAGS_TIMESTAMP_COMPLETION', (IB_UVERBS_CQ_FLAGS_IGNORE_OVERRUN:=2): 'IB_UVERBS_CQ_FLAGS_IGNORE_OVERRUN'}
@c.record
class struct_ib_uverbs_ex_create_cq(c.Struct):
  SIZE = 32
  user_handle: int
  cqe: int
  comp_vector: int
  comp_channel: int
  comp_mask: int
  flags: int
  reserved: int
struct_ib_uverbs_ex_create_cq.register_fields([('user_handle', ctypes.c_uint64, 0), ('cqe', ctypes.c_uint32, 8), ('comp_vector', ctypes.c_uint32, 12), ('comp_channel', ctypes.c_int32, 16), ('comp_mask', ctypes.c_uint32, 20), ('flags', ctypes.c_uint32, 24), ('reserved', ctypes.c_uint32, 28)])
@c.record
class struct_ib_uverbs_create_cq_resp(c.Struct):
  SIZE = 8
  cq_handle: int
  cqe: int
  driver_data: c.Array[ctypes.c_uint64, Literal[0]]
struct_ib_uverbs_create_cq_resp.register_fields([('cq_handle', ctypes.c_uint32, 0), ('cqe', ctypes.c_uint32, 4), ('driver_data', c.Array[ctypes.c_uint64, Literal[0]], 8)])
@c.record
class struct_ib_uverbs_ex_create_cq_resp(c.Struct):
  SIZE = 16
  base: struct_ib_uverbs_create_cq_resp
  comp_mask: int
  response_length: int
struct_ib_uverbs_ex_create_cq_resp.register_fields([('base', struct_ib_uverbs_create_cq_resp, 0), ('comp_mask', ctypes.c_uint32, 8), ('response_length', ctypes.c_uint32, 12)])
@c.record
class struct_ib_uverbs_resize_cq(c.Struct):
  SIZE = 16
  response: int
  cq_handle: int
  cqe: int
  driver_data: c.Array[ctypes.c_uint64, Literal[0]]
struct_ib_uverbs_resize_cq.register_fields([('response', ctypes.c_uint64, 0), ('cq_handle', ctypes.c_uint32, 8), ('cqe', ctypes.c_uint32, 12), ('driver_data', c.Array[ctypes.c_uint64, Literal[0]], 16)])
@c.record
class struct_ib_uverbs_resize_cq_resp(c.Struct):
  SIZE = 8
  cqe: int
  reserved: int
  driver_data: c.Array[ctypes.c_uint64, Literal[0]]
struct_ib_uverbs_resize_cq_resp.register_fields([('cqe', ctypes.c_uint32, 0), ('reserved', ctypes.c_uint32, 4), ('driver_data', c.Array[ctypes.c_uint64, Literal[0]], 8)])
@c.record
class struct_ib_uverbs_poll_cq(c.Struct):
  SIZE = 16
  response: int
  cq_handle: int
  ne: int
struct_ib_uverbs_poll_cq.register_fields([('response', ctypes.c_uint64, 0), ('cq_handle', ctypes.c_uint32, 8), ('ne', ctypes.c_uint32, 12)])
enum_ib_uverbs_wc_opcode: dict[int, str] = {(IB_UVERBS_WC_SEND:=0): 'IB_UVERBS_WC_SEND', (IB_UVERBS_WC_RDMA_WRITE:=1): 'IB_UVERBS_WC_RDMA_WRITE', (IB_UVERBS_WC_RDMA_READ:=2): 'IB_UVERBS_WC_RDMA_READ', (IB_UVERBS_WC_COMP_SWAP:=3): 'IB_UVERBS_WC_COMP_SWAP', (IB_UVERBS_WC_FETCH_ADD:=4): 'IB_UVERBS_WC_FETCH_ADD', (IB_UVERBS_WC_BIND_MW:=5): 'IB_UVERBS_WC_BIND_MW', (IB_UVERBS_WC_LOCAL_INV:=6): 'IB_UVERBS_WC_LOCAL_INV', (IB_UVERBS_WC_TSO:=7): 'IB_UVERBS_WC_TSO', (IB_UVERBS_WC_FLUSH:=8): 'IB_UVERBS_WC_FLUSH', (IB_UVERBS_WC_ATOMIC_WRITE:=9): 'IB_UVERBS_WC_ATOMIC_WRITE'}
@c.record
class struct_ib_uverbs_wc(c.Struct):
  SIZE = 48
  wr_id: int
  status: int
  opcode: int
  vendor_err: int
  byte_len: int
  ex: struct_ib_uverbs_wc_ex
  qp_num: int
  src_qp: int
  wc_flags: int
  pkey_index: int
  slid: int
  sl: int
  dlid_path_bits: int
  port_num: int
  reserved: int
@c.record
class struct_ib_uverbs_wc_ex(c.Struct):
  SIZE = 4
  imm_data: int
  invalidate_rkey: int
struct_ib_uverbs_wc_ex.register_fields([('imm_data', ctypes.c_uint32, 0), ('invalidate_rkey', ctypes.c_uint32, 0)])
struct_ib_uverbs_wc.register_fields([('wr_id', ctypes.c_uint64, 0), ('status', ctypes.c_uint32, 8), ('opcode', ctypes.c_uint32, 12), ('vendor_err', ctypes.c_uint32, 16), ('byte_len', ctypes.c_uint32, 20), ('ex', struct_ib_uverbs_wc_ex, 24), ('qp_num', ctypes.c_uint32, 28), ('src_qp', ctypes.c_uint32, 32), ('wc_flags', ctypes.c_uint32, 36), ('pkey_index', ctypes.c_uint16, 40), ('slid', ctypes.c_uint16, 42), ('sl', ctypes.c_ubyte, 44), ('dlid_path_bits', ctypes.c_ubyte, 45), ('port_num', ctypes.c_ubyte, 46), ('reserved', ctypes.c_ubyte, 47)])
@c.record
class struct_ib_uverbs_poll_cq_resp(c.Struct):
  SIZE = 8
  count: int
  reserved: int
  wc: c.Array[struct_ib_uverbs_wc, Literal[0]]
struct_ib_uverbs_poll_cq_resp.register_fields([('count', ctypes.c_uint32, 0), ('reserved', ctypes.c_uint32, 4), ('wc', c.Array[struct_ib_uverbs_wc, Literal[0]], 8)])
@c.record
class struct_ib_uverbs_req_notify_cq(c.Struct):
  SIZE = 8
  cq_handle: int
  solicited_only: int
struct_ib_uverbs_req_notify_cq.register_fields([('cq_handle', ctypes.c_uint32, 0), ('solicited_only', ctypes.c_uint32, 4)])
@c.record
class struct_ib_uverbs_destroy_cq(c.Struct):
  SIZE = 16
  response: int
  cq_handle: int
  reserved: int
struct_ib_uverbs_destroy_cq.register_fields([('response', ctypes.c_uint64, 0), ('cq_handle', ctypes.c_uint32, 8), ('reserved', ctypes.c_uint32, 12)])
@c.record
class struct_ib_uverbs_destroy_cq_resp(c.Struct):
  SIZE = 8
  comp_events_reported: int
  async_events_reported: int
struct_ib_uverbs_destroy_cq_resp.register_fields([('comp_events_reported', ctypes.c_uint32, 0), ('async_events_reported', ctypes.c_uint32, 4)])
@c.record
class struct_ib_uverbs_global_route(c.Struct):
  SIZE = 24
  dgid: c.Array[ctypes.c_ubyte, Literal[16]]
  flow_label: int
  sgid_index: int
  hop_limit: int
  traffic_class: int
  reserved: int
struct_ib_uverbs_global_route.register_fields([('dgid', c.Array[ctypes.c_ubyte, Literal[16]], 0), ('flow_label', ctypes.c_uint32, 16), ('sgid_index', ctypes.c_ubyte, 20), ('hop_limit', ctypes.c_ubyte, 21), ('traffic_class', ctypes.c_ubyte, 22), ('reserved', ctypes.c_ubyte, 23)])
@c.record
class struct_ib_uverbs_ah_attr(c.Struct):
  SIZE = 32
  grh: struct_ib_uverbs_global_route
  dlid: int
  sl: int
  src_path_bits: int
  static_rate: int
  is_global: int
  port_num: int
  reserved: int
struct_ib_uverbs_ah_attr.register_fields([('grh', struct_ib_uverbs_global_route, 0), ('dlid', ctypes.c_uint16, 24), ('sl', ctypes.c_ubyte, 26), ('src_path_bits', ctypes.c_ubyte, 27), ('static_rate', ctypes.c_ubyte, 28), ('is_global', ctypes.c_ubyte, 29), ('port_num', ctypes.c_ubyte, 30), ('reserved', ctypes.c_ubyte, 31)])
@c.record
class struct_ib_uverbs_qp_attr(c.Struct):
  SIZE = 144
  qp_attr_mask: int
  qp_state: int
  cur_qp_state: int
  path_mtu: int
  path_mig_state: int
  qkey: int
  rq_psn: int
  sq_psn: int
  dest_qp_num: int
  qp_access_flags: int
  ah_attr: struct_ib_uverbs_ah_attr
  alt_ah_attr: struct_ib_uverbs_ah_attr
  max_send_wr: int
  max_recv_wr: int
  max_send_sge: int
  max_recv_sge: int
  max_inline_data: int
  pkey_index: int
  alt_pkey_index: int
  en_sqd_async_notify: int
  sq_draining: int
  max_rd_atomic: int
  max_dest_rd_atomic: int
  min_rnr_timer: int
  port_num: int
  timeout: int
  retry_cnt: int
  rnr_retry: int
  alt_port_num: int
  alt_timeout: int
  reserved: c.Array[ctypes.c_ubyte, Literal[5]]
struct_ib_uverbs_qp_attr.register_fields([('qp_attr_mask', ctypes.c_uint32, 0), ('qp_state', ctypes.c_uint32, 4), ('cur_qp_state', ctypes.c_uint32, 8), ('path_mtu', ctypes.c_uint32, 12), ('path_mig_state', ctypes.c_uint32, 16), ('qkey', ctypes.c_uint32, 20), ('rq_psn', ctypes.c_uint32, 24), ('sq_psn', ctypes.c_uint32, 28), ('dest_qp_num', ctypes.c_uint32, 32), ('qp_access_flags', ctypes.c_uint32, 36), ('ah_attr', struct_ib_uverbs_ah_attr, 40), ('alt_ah_attr', struct_ib_uverbs_ah_attr, 72), ('max_send_wr', ctypes.c_uint32, 104), ('max_recv_wr', ctypes.c_uint32, 108), ('max_send_sge', ctypes.c_uint32, 112), ('max_recv_sge', ctypes.c_uint32, 116), ('max_inline_data', ctypes.c_uint32, 120), ('pkey_index', ctypes.c_uint16, 124), ('alt_pkey_index', ctypes.c_uint16, 126), ('en_sqd_async_notify', ctypes.c_ubyte, 128), ('sq_draining', ctypes.c_ubyte, 129), ('max_rd_atomic', ctypes.c_ubyte, 130), ('max_dest_rd_atomic', ctypes.c_ubyte, 131), ('min_rnr_timer', ctypes.c_ubyte, 132), ('port_num', ctypes.c_ubyte, 133), ('timeout', ctypes.c_ubyte, 134), ('retry_cnt', ctypes.c_ubyte, 135), ('rnr_retry', ctypes.c_ubyte, 136), ('alt_port_num', ctypes.c_ubyte, 137), ('alt_timeout', ctypes.c_ubyte, 138), ('reserved', c.Array[ctypes.c_ubyte, Literal[5]], 139)])
@c.record
class struct_ib_uverbs_create_qp(c.Struct):
  SIZE = 56
  response: int
  user_handle: int
  pd_handle: int
  send_cq_handle: int
  recv_cq_handle: int
  srq_handle: int
  max_send_wr: int
  max_recv_wr: int
  max_send_sge: int
  max_recv_sge: int
  max_inline_data: int
  sq_sig_all: int
  qp_type: int
  is_srq: int
  reserved: int
  driver_data: c.Array[ctypes.c_uint64, Literal[0]]
struct_ib_uverbs_create_qp.register_fields([('response', ctypes.c_uint64, 0), ('user_handle', ctypes.c_uint64, 8), ('pd_handle', ctypes.c_uint32, 16), ('send_cq_handle', ctypes.c_uint32, 20), ('recv_cq_handle', ctypes.c_uint32, 24), ('srq_handle', ctypes.c_uint32, 28), ('max_send_wr', ctypes.c_uint32, 32), ('max_recv_wr', ctypes.c_uint32, 36), ('max_send_sge', ctypes.c_uint32, 40), ('max_recv_sge', ctypes.c_uint32, 44), ('max_inline_data', ctypes.c_uint32, 48), ('sq_sig_all', ctypes.c_ubyte, 52), ('qp_type', ctypes.c_ubyte, 53), ('is_srq', ctypes.c_ubyte, 54), ('reserved', ctypes.c_ubyte, 55), ('driver_data', c.Array[ctypes.c_uint64, Literal[0]], 56)])
enum_ib_uverbs_create_qp_mask: dict[int, str] = {(IB_UVERBS_CREATE_QP_MASK_IND_TABLE:=1): 'IB_UVERBS_CREATE_QP_MASK_IND_TABLE'}
_anonenum6: dict[int, str] = {(IB_UVERBS_CREATE_QP_SUP_COMP_MASK:=1): 'IB_UVERBS_CREATE_QP_SUP_COMP_MASK'}
@c.record
class struct_ib_uverbs_ex_create_qp(c.Struct):
  SIZE = 64
  user_handle: int
  pd_handle: int
  send_cq_handle: int
  recv_cq_handle: int
  srq_handle: int
  max_send_wr: int
  max_recv_wr: int
  max_send_sge: int
  max_recv_sge: int
  max_inline_data: int
  sq_sig_all: int
  qp_type: int
  is_srq: int
  reserved: int
  comp_mask: int
  create_flags: int
  rwq_ind_tbl_handle: int
  source_qpn: int
struct_ib_uverbs_ex_create_qp.register_fields([('user_handle', ctypes.c_uint64, 0), ('pd_handle', ctypes.c_uint32, 8), ('send_cq_handle', ctypes.c_uint32, 12), ('recv_cq_handle', ctypes.c_uint32, 16), ('srq_handle', ctypes.c_uint32, 20), ('max_send_wr', ctypes.c_uint32, 24), ('max_recv_wr', ctypes.c_uint32, 28), ('max_send_sge', ctypes.c_uint32, 32), ('max_recv_sge', ctypes.c_uint32, 36), ('max_inline_data', ctypes.c_uint32, 40), ('sq_sig_all', ctypes.c_ubyte, 44), ('qp_type', ctypes.c_ubyte, 45), ('is_srq', ctypes.c_ubyte, 46), ('reserved', ctypes.c_ubyte, 47), ('comp_mask', ctypes.c_uint32, 48), ('create_flags', ctypes.c_uint32, 52), ('rwq_ind_tbl_handle', ctypes.c_uint32, 56), ('source_qpn', ctypes.c_uint32, 60)])
@c.record
class struct_ib_uverbs_open_qp(c.Struct):
  SIZE = 32
  response: int
  user_handle: int
  pd_handle: int
  qpn: int
  qp_type: int
  reserved: c.Array[ctypes.c_ubyte, Literal[7]]
  driver_data: c.Array[ctypes.c_uint64, Literal[0]]
struct_ib_uverbs_open_qp.register_fields([('response', ctypes.c_uint64, 0), ('user_handle', ctypes.c_uint64, 8), ('pd_handle', ctypes.c_uint32, 16), ('qpn', ctypes.c_uint32, 20), ('qp_type', ctypes.c_ubyte, 24), ('reserved', c.Array[ctypes.c_ubyte, Literal[7]], 25), ('driver_data', c.Array[ctypes.c_uint64, Literal[0]], 32)])
@c.record
class struct_ib_uverbs_create_qp_resp(c.Struct):
  SIZE = 32
  qp_handle: int
  qpn: int
  max_send_wr: int
  max_recv_wr: int
  max_send_sge: int
  max_recv_sge: int
  max_inline_data: int
  reserved: int
  driver_data: c.Array[ctypes.c_uint32, Literal[0]]
struct_ib_uverbs_create_qp_resp.register_fields([('qp_handle', ctypes.c_uint32, 0), ('qpn', ctypes.c_uint32, 4), ('max_send_wr', ctypes.c_uint32, 8), ('max_recv_wr', ctypes.c_uint32, 12), ('max_send_sge', ctypes.c_uint32, 16), ('max_recv_sge', ctypes.c_uint32, 20), ('max_inline_data', ctypes.c_uint32, 24), ('reserved', ctypes.c_uint32, 28), ('driver_data', c.Array[ctypes.c_uint32, Literal[0]], 32)])
@c.record
class struct_ib_uverbs_ex_create_qp_resp(c.Struct):
  SIZE = 40
  base: struct_ib_uverbs_create_qp_resp
  comp_mask: int
  response_length: int
struct_ib_uverbs_ex_create_qp_resp.register_fields([('base', struct_ib_uverbs_create_qp_resp, 0), ('comp_mask', ctypes.c_uint32, 32), ('response_length', ctypes.c_uint32, 36)])
@c.record
class struct_ib_uverbs_qp_dest(c.Struct):
  SIZE = 32
  dgid: c.Array[ctypes.c_ubyte, Literal[16]]
  flow_label: int
  dlid: int
  reserved: int
  sgid_index: int
  hop_limit: int
  traffic_class: int
  sl: int
  src_path_bits: int
  static_rate: int
  is_global: int
  port_num: int
struct_ib_uverbs_qp_dest.register_fields([('dgid', c.Array[ctypes.c_ubyte, Literal[16]], 0), ('flow_label', ctypes.c_uint32, 16), ('dlid', ctypes.c_uint16, 20), ('reserved', ctypes.c_uint16, 22), ('sgid_index', ctypes.c_ubyte, 24), ('hop_limit', ctypes.c_ubyte, 25), ('traffic_class', ctypes.c_ubyte, 26), ('sl', ctypes.c_ubyte, 27), ('src_path_bits', ctypes.c_ubyte, 28), ('static_rate', ctypes.c_ubyte, 29), ('is_global', ctypes.c_ubyte, 30), ('port_num', ctypes.c_ubyte, 31)])
@c.record
class struct_ib_uverbs_query_qp(c.Struct):
  SIZE = 16
  response: int
  qp_handle: int
  attr_mask: int
  driver_data: c.Array[ctypes.c_uint64, Literal[0]]
struct_ib_uverbs_query_qp.register_fields([('response', ctypes.c_uint64, 0), ('qp_handle', ctypes.c_uint32, 8), ('attr_mask', ctypes.c_uint32, 12), ('driver_data', c.Array[ctypes.c_uint64, Literal[0]], 16)])
@c.record
class struct_ib_uverbs_query_qp_resp(c.Struct):
  SIZE = 128
  dest: struct_ib_uverbs_qp_dest
  alt_dest: struct_ib_uverbs_qp_dest
  max_send_wr: int
  max_recv_wr: int
  max_send_sge: int
  max_recv_sge: int
  max_inline_data: int
  qkey: int
  rq_psn: int
  sq_psn: int
  dest_qp_num: int
  qp_access_flags: int
  pkey_index: int
  alt_pkey_index: int
  qp_state: int
  cur_qp_state: int
  path_mtu: int
  path_mig_state: int
  sq_draining: int
  max_rd_atomic: int
  max_dest_rd_atomic: int
  min_rnr_timer: int
  port_num: int
  timeout: int
  retry_cnt: int
  rnr_retry: int
  alt_port_num: int
  alt_timeout: int
  sq_sig_all: int
  reserved: c.Array[ctypes.c_ubyte, Literal[5]]
  driver_data: c.Array[ctypes.c_uint64, Literal[0]]
struct_ib_uverbs_query_qp_resp.register_fields([('dest', struct_ib_uverbs_qp_dest, 0), ('alt_dest', struct_ib_uverbs_qp_dest, 32), ('max_send_wr', ctypes.c_uint32, 64), ('max_recv_wr', ctypes.c_uint32, 68), ('max_send_sge', ctypes.c_uint32, 72), ('max_recv_sge', ctypes.c_uint32, 76), ('max_inline_data', ctypes.c_uint32, 80), ('qkey', ctypes.c_uint32, 84), ('rq_psn', ctypes.c_uint32, 88), ('sq_psn', ctypes.c_uint32, 92), ('dest_qp_num', ctypes.c_uint32, 96), ('qp_access_flags', ctypes.c_uint32, 100), ('pkey_index', ctypes.c_uint16, 104), ('alt_pkey_index', ctypes.c_uint16, 106), ('qp_state', ctypes.c_ubyte, 108), ('cur_qp_state', ctypes.c_ubyte, 109), ('path_mtu', ctypes.c_ubyte, 110), ('path_mig_state', ctypes.c_ubyte, 111), ('sq_draining', ctypes.c_ubyte, 112), ('max_rd_atomic', ctypes.c_ubyte, 113), ('max_dest_rd_atomic', ctypes.c_ubyte, 114), ('min_rnr_timer', ctypes.c_ubyte, 115), ('port_num', ctypes.c_ubyte, 116), ('timeout', ctypes.c_ubyte, 117), ('retry_cnt', ctypes.c_ubyte, 118), ('rnr_retry', ctypes.c_ubyte, 119), ('alt_port_num', ctypes.c_ubyte, 120), ('alt_timeout', ctypes.c_ubyte, 121), ('sq_sig_all', ctypes.c_ubyte, 122), ('reserved', c.Array[ctypes.c_ubyte, Literal[5]], 123), ('driver_data', c.Array[ctypes.c_uint64, Literal[0]], 128)])
@c.record
class struct_ib_uverbs_modify_qp(c.Struct):
  SIZE = 112
  dest: struct_ib_uverbs_qp_dest
  alt_dest: struct_ib_uverbs_qp_dest
  qp_handle: int
  attr_mask: int
  qkey: int
  rq_psn: int
  sq_psn: int
  dest_qp_num: int
  qp_access_flags: int
  pkey_index: int
  alt_pkey_index: int
  qp_state: int
  cur_qp_state: int
  path_mtu: int
  path_mig_state: int
  en_sqd_async_notify: int
  max_rd_atomic: int
  max_dest_rd_atomic: int
  min_rnr_timer: int
  port_num: int
  timeout: int
  retry_cnt: int
  rnr_retry: int
  alt_port_num: int
  alt_timeout: int
  reserved: c.Array[ctypes.c_ubyte, Literal[2]]
  driver_data: c.Array[ctypes.c_uint64, Literal[0]]
struct_ib_uverbs_modify_qp.register_fields([('dest', struct_ib_uverbs_qp_dest, 0), ('alt_dest', struct_ib_uverbs_qp_dest, 32), ('qp_handle', ctypes.c_uint32, 64), ('attr_mask', ctypes.c_uint32, 68), ('qkey', ctypes.c_uint32, 72), ('rq_psn', ctypes.c_uint32, 76), ('sq_psn', ctypes.c_uint32, 80), ('dest_qp_num', ctypes.c_uint32, 84), ('qp_access_flags', ctypes.c_uint32, 88), ('pkey_index', ctypes.c_uint16, 92), ('alt_pkey_index', ctypes.c_uint16, 94), ('qp_state', ctypes.c_ubyte, 96), ('cur_qp_state', ctypes.c_ubyte, 97), ('path_mtu', ctypes.c_ubyte, 98), ('path_mig_state', ctypes.c_ubyte, 99), ('en_sqd_async_notify', ctypes.c_ubyte, 100), ('max_rd_atomic', ctypes.c_ubyte, 101), ('max_dest_rd_atomic', ctypes.c_ubyte, 102), ('min_rnr_timer', ctypes.c_ubyte, 103), ('port_num', ctypes.c_ubyte, 104), ('timeout', ctypes.c_ubyte, 105), ('retry_cnt', ctypes.c_ubyte, 106), ('rnr_retry', ctypes.c_ubyte, 107), ('alt_port_num', ctypes.c_ubyte, 108), ('alt_timeout', ctypes.c_ubyte, 109), ('reserved', c.Array[ctypes.c_ubyte, Literal[2]], 110), ('driver_data', c.Array[ctypes.c_uint64, Literal[0]], 112)])
@c.record
class struct_ib_uverbs_ex_modify_qp(c.Struct):
  SIZE = 120
  base: struct_ib_uverbs_modify_qp
  rate_limit: int
  reserved: int
struct_ib_uverbs_ex_modify_qp.register_fields([('base', struct_ib_uverbs_modify_qp, 0), ('rate_limit', ctypes.c_uint32, 112), ('reserved', ctypes.c_uint32, 116)])
@c.record
class struct_ib_uverbs_ex_modify_qp_resp(c.Struct):
  SIZE = 8
  comp_mask: int
  response_length: int
struct_ib_uverbs_ex_modify_qp_resp.register_fields([('comp_mask', ctypes.c_uint32, 0), ('response_length', ctypes.c_uint32, 4)])
@c.record
class struct_ib_uverbs_destroy_qp(c.Struct):
  SIZE = 16
  response: int
  qp_handle: int
  reserved: int
struct_ib_uverbs_destroy_qp.register_fields([('response', ctypes.c_uint64, 0), ('qp_handle', ctypes.c_uint32, 8), ('reserved', ctypes.c_uint32, 12)])
@c.record
class struct_ib_uverbs_destroy_qp_resp(c.Struct):
  SIZE = 4
  events_reported: int
struct_ib_uverbs_destroy_qp_resp.register_fields([('events_reported', ctypes.c_uint32, 0)])
@c.record
class struct_ib_uverbs_sge(c.Struct):
  SIZE = 16
  addr: int
  length: int
  lkey: int
struct_ib_uverbs_sge.register_fields([('addr', ctypes.c_uint64, 0), ('length', ctypes.c_uint32, 8), ('lkey', ctypes.c_uint32, 12)])
enum_ib_uverbs_wr_opcode: dict[int, str] = {(IB_UVERBS_WR_RDMA_WRITE:=0): 'IB_UVERBS_WR_RDMA_WRITE', (IB_UVERBS_WR_RDMA_WRITE_WITH_IMM:=1): 'IB_UVERBS_WR_RDMA_WRITE_WITH_IMM', (IB_UVERBS_WR_SEND:=2): 'IB_UVERBS_WR_SEND', (IB_UVERBS_WR_SEND_WITH_IMM:=3): 'IB_UVERBS_WR_SEND_WITH_IMM', (IB_UVERBS_WR_RDMA_READ:=4): 'IB_UVERBS_WR_RDMA_READ', (IB_UVERBS_WR_ATOMIC_CMP_AND_SWP:=5): 'IB_UVERBS_WR_ATOMIC_CMP_AND_SWP', (IB_UVERBS_WR_ATOMIC_FETCH_AND_ADD:=6): 'IB_UVERBS_WR_ATOMIC_FETCH_AND_ADD', (IB_UVERBS_WR_LOCAL_INV:=7): 'IB_UVERBS_WR_LOCAL_INV', (IB_UVERBS_WR_BIND_MW:=8): 'IB_UVERBS_WR_BIND_MW', (IB_UVERBS_WR_SEND_WITH_INV:=9): 'IB_UVERBS_WR_SEND_WITH_INV', (IB_UVERBS_WR_TSO:=10): 'IB_UVERBS_WR_TSO', (IB_UVERBS_WR_RDMA_READ_WITH_INV:=11): 'IB_UVERBS_WR_RDMA_READ_WITH_INV', (IB_UVERBS_WR_MASKED_ATOMIC_CMP_AND_SWP:=12): 'IB_UVERBS_WR_MASKED_ATOMIC_CMP_AND_SWP', (IB_UVERBS_WR_MASKED_ATOMIC_FETCH_AND_ADD:=13): 'IB_UVERBS_WR_MASKED_ATOMIC_FETCH_AND_ADD', (IB_UVERBS_WR_FLUSH:=14): 'IB_UVERBS_WR_FLUSH', (IB_UVERBS_WR_ATOMIC_WRITE:=15): 'IB_UVERBS_WR_ATOMIC_WRITE'}
@c.record
class struct_ib_uverbs_send_wr(c.Struct):
  SIZE = 56
  wr_id: int
  num_sge: int
  opcode: int
  send_flags: int
  ex: struct_ib_uverbs_send_wr_ex
  wr: struct_ib_uverbs_send_wr_wr
@c.record
class struct_ib_uverbs_send_wr_ex(c.Struct):
  SIZE = 4
  imm_data: int
  invalidate_rkey: int
struct_ib_uverbs_send_wr_ex.register_fields([('imm_data', ctypes.c_uint32, 0), ('invalidate_rkey', ctypes.c_uint32, 0)])
@c.record
class struct_ib_uverbs_send_wr_wr(c.Struct):
  SIZE = 32
  rdma: struct_ib_uverbs_send_wr_wr_rdma
  atomic: struct_ib_uverbs_send_wr_wr_atomic
  ud: struct_ib_uverbs_send_wr_wr_ud
@c.record
class struct_ib_uverbs_send_wr_wr_rdma(c.Struct):
  SIZE = 16
  remote_addr: int
  rkey: int
  reserved: int
struct_ib_uverbs_send_wr_wr_rdma.register_fields([('remote_addr', ctypes.c_uint64, 0), ('rkey', ctypes.c_uint32, 8), ('reserved', ctypes.c_uint32, 12)])
@c.record
class struct_ib_uverbs_send_wr_wr_atomic(c.Struct):
  SIZE = 32
  remote_addr: int
  compare_add: int
  swap: int
  rkey: int
  reserved: int
struct_ib_uverbs_send_wr_wr_atomic.register_fields([('remote_addr', ctypes.c_uint64, 0), ('compare_add', ctypes.c_uint64, 8), ('swap', ctypes.c_uint64, 16), ('rkey', ctypes.c_uint32, 24), ('reserved', ctypes.c_uint32, 28)])
@c.record
class struct_ib_uverbs_send_wr_wr_ud(c.Struct):
  SIZE = 16
  ah: int
  remote_qpn: int
  remote_qkey: int
  reserved: int
struct_ib_uverbs_send_wr_wr_ud.register_fields([('ah', ctypes.c_uint32, 0), ('remote_qpn', ctypes.c_uint32, 4), ('remote_qkey', ctypes.c_uint32, 8), ('reserved', ctypes.c_uint32, 12)])
struct_ib_uverbs_send_wr_wr.register_fields([('rdma', struct_ib_uverbs_send_wr_wr_rdma, 0), ('atomic', struct_ib_uverbs_send_wr_wr_atomic, 0), ('ud', struct_ib_uverbs_send_wr_wr_ud, 0)])
struct_ib_uverbs_send_wr.register_fields([('wr_id', ctypes.c_uint64, 0), ('num_sge', ctypes.c_uint32, 8), ('opcode', ctypes.c_uint32, 12), ('send_flags', ctypes.c_uint32, 16), ('ex', struct_ib_uverbs_send_wr_ex, 20), ('wr', struct_ib_uverbs_send_wr_wr, 24)])
@c.record
class struct_ib_uverbs_post_send(c.Struct):
  SIZE = 24
  response: int
  qp_handle: int
  wr_count: int
  sge_count: int
  wqe_size: int
  send_wr: c.Array[struct_ib_uverbs_send_wr, Literal[0]]
struct_ib_uverbs_post_send.register_fields([('response', ctypes.c_uint64, 0), ('qp_handle', ctypes.c_uint32, 8), ('wr_count', ctypes.c_uint32, 12), ('sge_count', ctypes.c_uint32, 16), ('wqe_size', ctypes.c_uint32, 20), ('send_wr', c.Array[struct_ib_uverbs_send_wr, Literal[0]], 24)])
@c.record
class struct_ib_uverbs_post_send_resp(c.Struct):
  SIZE = 4
  bad_wr: int
struct_ib_uverbs_post_send_resp.register_fields([('bad_wr', ctypes.c_uint32, 0)])
@c.record
class struct_ib_uverbs_recv_wr(c.Struct):
  SIZE = 16
  wr_id: int
  num_sge: int
  reserved: int
struct_ib_uverbs_recv_wr.register_fields([('wr_id', ctypes.c_uint64, 0), ('num_sge', ctypes.c_uint32, 8), ('reserved', ctypes.c_uint32, 12)])
@c.record
class struct_ib_uverbs_post_recv(c.Struct):
  SIZE = 24
  response: int
  qp_handle: int
  wr_count: int
  sge_count: int
  wqe_size: int
  recv_wr: c.Array[struct_ib_uverbs_recv_wr, Literal[0]]
struct_ib_uverbs_post_recv.register_fields([('response', ctypes.c_uint64, 0), ('qp_handle', ctypes.c_uint32, 8), ('wr_count', ctypes.c_uint32, 12), ('sge_count', ctypes.c_uint32, 16), ('wqe_size', ctypes.c_uint32, 20), ('recv_wr', c.Array[struct_ib_uverbs_recv_wr, Literal[0]], 24)])
@c.record
class struct_ib_uverbs_post_recv_resp(c.Struct):
  SIZE = 4
  bad_wr: int
struct_ib_uverbs_post_recv_resp.register_fields([('bad_wr', ctypes.c_uint32, 0)])
@c.record
class struct_ib_uverbs_post_srq_recv(c.Struct):
  SIZE = 24
  response: int
  srq_handle: int
  wr_count: int
  sge_count: int
  wqe_size: int
  recv: c.Array[struct_ib_uverbs_recv_wr, Literal[0]]
struct_ib_uverbs_post_srq_recv.register_fields([('response', ctypes.c_uint64, 0), ('srq_handle', ctypes.c_uint32, 8), ('wr_count', ctypes.c_uint32, 12), ('sge_count', ctypes.c_uint32, 16), ('wqe_size', ctypes.c_uint32, 20), ('recv', c.Array[struct_ib_uverbs_recv_wr, Literal[0]], 24)])
@c.record
class struct_ib_uverbs_post_srq_recv_resp(c.Struct):
  SIZE = 4
  bad_wr: int
struct_ib_uverbs_post_srq_recv_resp.register_fields([('bad_wr', ctypes.c_uint32, 0)])
@c.record
class struct_ib_uverbs_create_ah(c.Struct):
  SIZE = 56
  response: int
  user_handle: int
  pd_handle: int
  reserved: int
  attr: struct_ib_uverbs_ah_attr
  driver_data: c.Array[ctypes.c_uint64, Literal[0]]
struct_ib_uverbs_create_ah.register_fields([('response', ctypes.c_uint64, 0), ('user_handle', ctypes.c_uint64, 8), ('pd_handle', ctypes.c_uint32, 16), ('reserved', ctypes.c_uint32, 20), ('attr', struct_ib_uverbs_ah_attr, 24), ('driver_data', c.Array[ctypes.c_uint64, Literal[0]], 56)])
@c.record
class struct_ib_uverbs_create_ah_resp(c.Struct):
  SIZE = 4
  ah_handle: int
  driver_data: c.Array[ctypes.c_uint32, Literal[0]]
struct_ib_uverbs_create_ah_resp.register_fields([('ah_handle', ctypes.c_uint32, 0), ('driver_data', c.Array[ctypes.c_uint32, Literal[0]], 4)])
@c.record
class struct_ib_uverbs_destroy_ah(c.Struct):
  SIZE = 4
  ah_handle: int
struct_ib_uverbs_destroy_ah.register_fields([('ah_handle', ctypes.c_uint32, 0)])
@c.record
class struct_ib_uverbs_attach_mcast(c.Struct):
  SIZE = 24
  gid: c.Array[ctypes.c_ubyte, Literal[16]]
  qp_handle: int
  mlid: int
  reserved: int
  driver_data: c.Array[ctypes.c_uint64, Literal[0]]
struct_ib_uverbs_attach_mcast.register_fields([('gid', c.Array[ctypes.c_ubyte, Literal[16]], 0), ('qp_handle', ctypes.c_uint32, 16), ('mlid', ctypes.c_uint16, 20), ('reserved', ctypes.c_uint16, 22), ('driver_data', c.Array[ctypes.c_uint64, Literal[0]], 24)])
@c.record
class struct_ib_uverbs_detach_mcast(c.Struct):
  SIZE = 24
  gid: c.Array[ctypes.c_ubyte, Literal[16]]
  qp_handle: int
  mlid: int
  reserved: int
  driver_data: c.Array[ctypes.c_uint64, Literal[0]]
struct_ib_uverbs_detach_mcast.register_fields([('gid', c.Array[ctypes.c_ubyte, Literal[16]], 0), ('qp_handle', ctypes.c_uint32, 16), ('mlid', ctypes.c_uint16, 20), ('reserved', ctypes.c_uint16, 22), ('driver_data', c.Array[ctypes.c_uint64, Literal[0]], 24)])
@c.record
class struct_ib_uverbs_flow_spec_hdr(c.Struct):
  SIZE = 8
  type: int
  size: int
  reserved: int
  flow_spec_data: c.Array[ctypes.c_uint64, Literal[0]]
struct_ib_uverbs_flow_spec_hdr.register_fields([('type', ctypes.c_uint32, 0), ('size', ctypes.c_uint16, 4), ('reserved', ctypes.c_uint16, 6), ('flow_spec_data', c.Array[ctypes.c_uint64, Literal[0]], 8)])
@c.record
class struct_ib_uverbs_flow_eth_filter(c.Struct):
  SIZE = 16
  dst_mac: c.Array[ctypes.c_ubyte, Literal[6]]
  src_mac: c.Array[ctypes.c_ubyte, Literal[6]]
  ether_type: int
  vlan_tag: int
struct_ib_uverbs_flow_eth_filter.register_fields([('dst_mac', c.Array[ctypes.c_ubyte, Literal[6]], 0), ('src_mac', c.Array[ctypes.c_ubyte, Literal[6]], 6), ('ether_type', ctypes.c_uint16, 12), ('vlan_tag', ctypes.c_uint16, 14)])
@c.record
class struct_ib_uverbs_flow_spec_eth(c.Struct):
  SIZE = 40
  hdr: struct_ib_uverbs_flow_spec_hdr
  type: int
  size: int
  reserved: int
  val: struct_ib_uverbs_flow_eth_filter
  mask: struct_ib_uverbs_flow_eth_filter
struct_ib_uverbs_flow_spec_eth.register_fields([('hdr', struct_ib_uverbs_flow_spec_hdr, 0), ('type', ctypes.c_uint32, 0), ('size', ctypes.c_uint16, 4), ('reserved', ctypes.c_uint16, 6), ('val', struct_ib_uverbs_flow_eth_filter, 8), ('mask', struct_ib_uverbs_flow_eth_filter, 24)])
@c.record
class struct_ib_uverbs_flow_ipv4_filter(c.Struct):
  SIZE = 12
  src_ip: int
  dst_ip: int
  proto: int
  tos: int
  ttl: int
  flags: int
struct_ib_uverbs_flow_ipv4_filter.register_fields([('src_ip', ctypes.c_uint32, 0), ('dst_ip', ctypes.c_uint32, 4), ('proto', ctypes.c_ubyte, 8), ('tos', ctypes.c_ubyte, 9), ('ttl', ctypes.c_ubyte, 10), ('flags', ctypes.c_ubyte, 11)])
@c.record
class struct_ib_uverbs_flow_spec_ipv4(c.Struct):
  SIZE = 32
  hdr: struct_ib_uverbs_flow_spec_hdr
  type: int
  size: int
  reserved: int
  val: struct_ib_uverbs_flow_ipv4_filter
  mask: struct_ib_uverbs_flow_ipv4_filter
struct_ib_uverbs_flow_spec_ipv4.register_fields([('hdr', struct_ib_uverbs_flow_spec_hdr, 0), ('type', ctypes.c_uint32, 0), ('size', ctypes.c_uint16, 4), ('reserved', ctypes.c_uint16, 6), ('val', struct_ib_uverbs_flow_ipv4_filter, 8), ('mask', struct_ib_uverbs_flow_ipv4_filter, 20)])
@c.record
class struct_ib_uverbs_flow_tcp_udp_filter(c.Struct):
  SIZE = 4
  dst_port: int
  src_port: int
struct_ib_uverbs_flow_tcp_udp_filter.register_fields([('dst_port', ctypes.c_uint16, 0), ('src_port', ctypes.c_uint16, 2)])
@c.record
class struct_ib_uverbs_flow_spec_tcp_udp(c.Struct):
  SIZE = 16
  hdr: struct_ib_uverbs_flow_spec_hdr
  type: int
  size: int
  reserved: int
  val: struct_ib_uverbs_flow_tcp_udp_filter
  mask: struct_ib_uverbs_flow_tcp_udp_filter
struct_ib_uverbs_flow_spec_tcp_udp.register_fields([('hdr', struct_ib_uverbs_flow_spec_hdr, 0), ('type', ctypes.c_uint32, 0), ('size', ctypes.c_uint16, 4), ('reserved', ctypes.c_uint16, 6), ('val', struct_ib_uverbs_flow_tcp_udp_filter, 8), ('mask', struct_ib_uverbs_flow_tcp_udp_filter, 12)])
@c.record
class struct_ib_uverbs_flow_ipv6_filter(c.Struct):
  SIZE = 40
  src_ip: c.Array[ctypes.c_ubyte, Literal[16]]
  dst_ip: c.Array[ctypes.c_ubyte, Literal[16]]
  flow_label: int
  next_hdr: int
  traffic_class: int
  hop_limit: int
  reserved: int
struct_ib_uverbs_flow_ipv6_filter.register_fields([('src_ip', c.Array[ctypes.c_ubyte, Literal[16]], 0), ('dst_ip', c.Array[ctypes.c_ubyte, Literal[16]], 16), ('flow_label', ctypes.c_uint32, 32), ('next_hdr', ctypes.c_ubyte, 36), ('traffic_class', ctypes.c_ubyte, 37), ('hop_limit', ctypes.c_ubyte, 38), ('reserved', ctypes.c_ubyte, 39)])
@c.record
class struct_ib_uverbs_flow_spec_ipv6(c.Struct):
  SIZE = 88
  hdr: struct_ib_uverbs_flow_spec_hdr
  type: int
  size: int
  reserved: int
  val: struct_ib_uverbs_flow_ipv6_filter
  mask: struct_ib_uverbs_flow_ipv6_filter
struct_ib_uverbs_flow_spec_ipv6.register_fields([('hdr', struct_ib_uverbs_flow_spec_hdr, 0), ('type', ctypes.c_uint32, 0), ('size', ctypes.c_uint16, 4), ('reserved', ctypes.c_uint16, 6), ('val', struct_ib_uverbs_flow_ipv6_filter, 8), ('mask', struct_ib_uverbs_flow_ipv6_filter, 48)])
@c.record
class struct_ib_uverbs_flow_spec_action_tag(c.Struct):
  SIZE = 16
  hdr: struct_ib_uverbs_flow_spec_hdr
  type: int
  size: int
  reserved: int
  tag_id: int
  reserved1: int
struct_ib_uverbs_flow_spec_action_tag.register_fields([('hdr', struct_ib_uverbs_flow_spec_hdr, 0), ('type', ctypes.c_uint32, 0), ('size', ctypes.c_uint16, 4), ('reserved', ctypes.c_uint16, 6), ('tag_id', ctypes.c_uint32, 8), ('reserved1', ctypes.c_uint32, 12)])
@c.record
class struct_ib_uverbs_flow_spec_action_drop(c.Struct):
  SIZE = 8
  hdr: struct_ib_uverbs_flow_spec_hdr
  type: int
  size: int
  reserved: int
struct_ib_uverbs_flow_spec_action_drop.register_fields([('hdr', struct_ib_uverbs_flow_spec_hdr, 0), ('type', ctypes.c_uint32, 0), ('size', ctypes.c_uint16, 4), ('reserved', ctypes.c_uint16, 6)])
@c.record
class struct_ib_uverbs_flow_spec_action_handle(c.Struct):
  SIZE = 16
  hdr: struct_ib_uverbs_flow_spec_hdr
  type: int
  size: int
  reserved: int
  handle: int
  reserved1: int
struct_ib_uverbs_flow_spec_action_handle.register_fields([('hdr', struct_ib_uverbs_flow_spec_hdr, 0), ('type', ctypes.c_uint32, 0), ('size', ctypes.c_uint16, 4), ('reserved', ctypes.c_uint16, 6), ('handle', ctypes.c_uint32, 8), ('reserved1', ctypes.c_uint32, 12)])
@c.record
class struct_ib_uverbs_flow_spec_action_count(c.Struct):
  SIZE = 16
  hdr: struct_ib_uverbs_flow_spec_hdr
  type: int
  size: int
  reserved: int
  handle: int
  reserved1: int
struct_ib_uverbs_flow_spec_action_count.register_fields([('hdr', struct_ib_uverbs_flow_spec_hdr, 0), ('type', ctypes.c_uint32, 0), ('size', ctypes.c_uint16, 4), ('reserved', ctypes.c_uint16, 6), ('handle', ctypes.c_uint32, 8), ('reserved1', ctypes.c_uint32, 12)])
@c.record
class struct_ib_uverbs_flow_tunnel_filter(c.Struct):
  SIZE = 4
  tunnel_id: int
struct_ib_uverbs_flow_tunnel_filter.register_fields([('tunnel_id', ctypes.c_uint32, 0)])
@c.record
class struct_ib_uverbs_flow_spec_tunnel(c.Struct):
  SIZE = 16
  hdr: struct_ib_uverbs_flow_spec_hdr
  type: int
  size: int
  reserved: int
  val: struct_ib_uverbs_flow_tunnel_filter
  mask: struct_ib_uverbs_flow_tunnel_filter
struct_ib_uverbs_flow_spec_tunnel.register_fields([('hdr', struct_ib_uverbs_flow_spec_hdr, 0), ('type', ctypes.c_uint32, 0), ('size', ctypes.c_uint16, 4), ('reserved', ctypes.c_uint16, 6), ('val', struct_ib_uverbs_flow_tunnel_filter, 8), ('mask', struct_ib_uverbs_flow_tunnel_filter, 12)])
@c.record
class struct_ib_uverbs_flow_spec_esp_filter(c.Struct):
  SIZE = 8
  spi: int
  seq: int
struct_ib_uverbs_flow_spec_esp_filter.register_fields([('spi', ctypes.c_uint32, 0), ('seq', ctypes.c_uint32, 4)])
@c.record
class struct_ib_uverbs_flow_spec_esp(c.Struct):
  SIZE = 24
  hdr: struct_ib_uverbs_flow_spec_hdr
  type: int
  size: int
  reserved: int
  val: struct_ib_uverbs_flow_spec_esp_filter
  mask: struct_ib_uverbs_flow_spec_esp_filter
struct_ib_uverbs_flow_spec_esp.register_fields([('hdr', struct_ib_uverbs_flow_spec_hdr, 0), ('type', ctypes.c_uint32, 0), ('size', ctypes.c_uint16, 4), ('reserved', ctypes.c_uint16, 6), ('val', struct_ib_uverbs_flow_spec_esp_filter, 8), ('mask', struct_ib_uverbs_flow_spec_esp_filter, 16)])
@c.record
class struct_ib_uverbs_flow_gre_filter(c.Struct):
  SIZE = 8
  c_ks_res0_ver: int
  protocol: int
  key: int
struct_ib_uverbs_flow_gre_filter.register_fields([('c_ks_res0_ver', ctypes.c_uint16, 0), ('protocol', ctypes.c_uint16, 2), ('key', ctypes.c_uint32, 4)])
@c.record
class struct_ib_uverbs_flow_spec_gre(c.Struct):
  SIZE = 24
  hdr: struct_ib_uverbs_flow_spec_hdr
  type: int
  size: int
  reserved: int
  val: struct_ib_uverbs_flow_gre_filter
  mask: struct_ib_uverbs_flow_gre_filter
struct_ib_uverbs_flow_spec_gre.register_fields([('hdr', struct_ib_uverbs_flow_spec_hdr, 0), ('type', ctypes.c_uint32, 0), ('size', ctypes.c_uint16, 4), ('reserved', ctypes.c_uint16, 6), ('val', struct_ib_uverbs_flow_gre_filter, 8), ('mask', struct_ib_uverbs_flow_gre_filter, 16)])
@c.record
class struct_ib_uverbs_flow_mpls_filter(c.Struct):
  SIZE = 4
  label: int
struct_ib_uverbs_flow_mpls_filter.register_fields([('label', ctypes.c_uint32, 0)])
@c.record
class struct_ib_uverbs_flow_spec_mpls(c.Struct):
  SIZE = 16
  hdr: struct_ib_uverbs_flow_spec_hdr
  type: int
  size: int
  reserved: int
  val: struct_ib_uverbs_flow_mpls_filter
  mask: struct_ib_uverbs_flow_mpls_filter
struct_ib_uverbs_flow_spec_mpls.register_fields([('hdr', struct_ib_uverbs_flow_spec_hdr, 0), ('type', ctypes.c_uint32, 0), ('size', ctypes.c_uint16, 4), ('reserved', ctypes.c_uint16, 6), ('val', struct_ib_uverbs_flow_mpls_filter, 8), ('mask', struct_ib_uverbs_flow_mpls_filter, 12)])
@c.record
class struct_ib_uverbs_flow_attr(c.Struct):
  SIZE = 16
  type: int
  size: int
  priority: int
  num_of_specs: int
  reserved: c.Array[ctypes.c_ubyte, Literal[2]]
  port: int
  flags: int
  flow_specs: c.Array[struct_ib_uverbs_flow_spec_hdr, Literal[0]]
struct_ib_uverbs_flow_attr.register_fields([('type', ctypes.c_uint32, 0), ('size', ctypes.c_uint16, 4), ('priority', ctypes.c_uint16, 6), ('num_of_specs', ctypes.c_ubyte, 8), ('reserved', c.Array[ctypes.c_ubyte, Literal[2]], 9), ('port', ctypes.c_ubyte, 11), ('flags', ctypes.c_uint32, 12), ('flow_specs', c.Array[struct_ib_uverbs_flow_spec_hdr, Literal[0]], 16)])
@c.record
class struct_ib_uverbs_create_flow(c.Struct):
  SIZE = 24
  comp_mask: int
  qp_handle: int
  flow_attr: struct_ib_uverbs_flow_attr
struct_ib_uverbs_create_flow.register_fields([('comp_mask', ctypes.c_uint32, 0), ('qp_handle', ctypes.c_uint32, 4), ('flow_attr', struct_ib_uverbs_flow_attr, 8)])
@c.record
class struct_ib_uverbs_create_flow_resp(c.Struct):
  SIZE = 8
  comp_mask: int
  flow_handle: int
struct_ib_uverbs_create_flow_resp.register_fields([('comp_mask', ctypes.c_uint32, 0), ('flow_handle', ctypes.c_uint32, 4)])
@c.record
class struct_ib_uverbs_destroy_flow(c.Struct):
  SIZE = 8
  comp_mask: int
  flow_handle: int
struct_ib_uverbs_destroy_flow.register_fields([('comp_mask', ctypes.c_uint32, 0), ('flow_handle', ctypes.c_uint32, 4)])
@c.record
class struct_ib_uverbs_create_srq(c.Struct):
  SIZE = 32
  response: int
  user_handle: int
  pd_handle: int
  max_wr: int
  max_sge: int
  srq_limit: int
  driver_data: c.Array[ctypes.c_uint64, Literal[0]]
struct_ib_uverbs_create_srq.register_fields([('response', ctypes.c_uint64, 0), ('user_handle', ctypes.c_uint64, 8), ('pd_handle', ctypes.c_uint32, 16), ('max_wr', ctypes.c_uint32, 20), ('max_sge', ctypes.c_uint32, 24), ('srq_limit', ctypes.c_uint32, 28), ('driver_data', c.Array[ctypes.c_uint64, Literal[0]], 32)])
@c.record
class struct_ib_uverbs_create_xsrq(c.Struct):
  SIZE = 48
  response: int
  user_handle: int
  srq_type: int
  pd_handle: int
  max_wr: int
  max_sge: int
  srq_limit: int
  max_num_tags: int
  xrcd_handle: int
  cq_handle: int
  driver_data: c.Array[ctypes.c_uint64, Literal[0]]
struct_ib_uverbs_create_xsrq.register_fields([('response', ctypes.c_uint64, 0), ('user_handle', ctypes.c_uint64, 8), ('srq_type', ctypes.c_uint32, 16), ('pd_handle', ctypes.c_uint32, 20), ('max_wr', ctypes.c_uint32, 24), ('max_sge', ctypes.c_uint32, 28), ('srq_limit', ctypes.c_uint32, 32), ('max_num_tags', ctypes.c_uint32, 36), ('xrcd_handle', ctypes.c_uint32, 40), ('cq_handle', ctypes.c_uint32, 44), ('driver_data', c.Array[ctypes.c_uint64, Literal[0]], 48)])
@c.record
class struct_ib_uverbs_create_srq_resp(c.Struct):
  SIZE = 16
  srq_handle: int
  max_wr: int
  max_sge: int
  srqn: int
  driver_data: c.Array[ctypes.c_uint32, Literal[0]]
struct_ib_uverbs_create_srq_resp.register_fields([('srq_handle', ctypes.c_uint32, 0), ('max_wr', ctypes.c_uint32, 4), ('max_sge', ctypes.c_uint32, 8), ('srqn', ctypes.c_uint32, 12), ('driver_data', c.Array[ctypes.c_uint32, Literal[0]], 16)])
@c.record
class struct_ib_uverbs_modify_srq(c.Struct):
  SIZE = 16
  srq_handle: int
  attr_mask: int
  max_wr: int
  srq_limit: int
  driver_data: c.Array[ctypes.c_uint64, Literal[0]]
struct_ib_uverbs_modify_srq.register_fields([('srq_handle', ctypes.c_uint32, 0), ('attr_mask', ctypes.c_uint32, 4), ('max_wr', ctypes.c_uint32, 8), ('srq_limit', ctypes.c_uint32, 12), ('driver_data', c.Array[ctypes.c_uint64, Literal[0]], 16)])
@c.record
class struct_ib_uverbs_query_srq(c.Struct):
  SIZE = 16
  response: int
  srq_handle: int
  reserved: int
  driver_data: c.Array[ctypes.c_uint64, Literal[0]]
struct_ib_uverbs_query_srq.register_fields([('response', ctypes.c_uint64, 0), ('srq_handle', ctypes.c_uint32, 8), ('reserved', ctypes.c_uint32, 12), ('driver_data', c.Array[ctypes.c_uint64, Literal[0]], 16)])
@c.record
class struct_ib_uverbs_query_srq_resp(c.Struct):
  SIZE = 16
  max_wr: int
  max_sge: int
  srq_limit: int
  reserved: int
struct_ib_uverbs_query_srq_resp.register_fields([('max_wr', ctypes.c_uint32, 0), ('max_sge', ctypes.c_uint32, 4), ('srq_limit', ctypes.c_uint32, 8), ('reserved', ctypes.c_uint32, 12)])
@c.record
class struct_ib_uverbs_destroy_srq(c.Struct):
  SIZE = 16
  response: int
  srq_handle: int
  reserved: int
struct_ib_uverbs_destroy_srq.register_fields([('response', ctypes.c_uint64, 0), ('srq_handle', ctypes.c_uint32, 8), ('reserved', ctypes.c_uint32, 12)])
@c.record
class struct_ib_uverbs_destroy_srq_resp(c.Struct):
  SIZE = 4
  events_reported: int
struct_ib_uverbs_destroy_srq_resp.register_fields([('events_reported', ctypes.c_uint32, 0)])
@c.record
class struct_ib_uverbs_ex_create_wq(c.Struct):
  SIZE = 40
  comp_mask: int
  wq_type: int
  user_handle: int
  pd_handle: int
  cq_handle: int
  max_wr: int
  max_sge: int
  create_flags: int
  reserved: int
struct_ib_uverbs_ex_create_wq.register_fields([('comp_mask', ctypes.c_uint32, 0), ('wq_type', ctypes.c_uint32, 4), ('user_handle', ctypes.c_uint64, 8), ('pd_handle', ctypes.c_uint32, 16), ('cq_handle', ctypes.c_uint32, 20), ('max_wr', ctypes.c_uint32, 24), ('max_sge', ctypes.c_uint32, 28), ('create_flags', ctypes.c_uint32, 32), ('reserved', ctypes.c_uint32, 36)])
@c.record
class struct_ib_uverbs_ex_create_wq_resp(c.Struct):
  SIZE = 24
  comp_mask: int
  response_length: int
  wq_handle: int
  max_wr: int
  max_sge: int
  wqn: int
struct_ib_uverbs_ex_create_wq_resp.register_fields([('comp_mask', ctypes.c_uint32, 0), ('response_length', ctypes.c_uint32, 4), ('wq_handle', ctypes.c_uint32, 8), ('max_wr', ctypes.c_uint32, 12), ('max_sge', ctypes.c_uint32, 16), ('wqn', ctypes.c_uint32, 20)])
@c.record
class struct_ib_uverbs_ex_destroy_wq(c.Struct):
  SIZE = 8
  comp_mask: int
  wq_handle: int
struct_ib_uverbs_ex_destroy_wq.register_fields([('comp_mask', ctypes.c_uint32, 0), ('wq_handle', ctypes.c_uint32, 4)])
@c.record
class struct_ib_uverbs_ex_destroy_wq_resp(c.Struct):
  SIZE = 16
  comp_mask: int
  response_length: int
  events_reported: int
  reserved: int
struct_ib_uverbs_ex_destroy_wq_resp.register_fields([('comp_mask', ctypes.c_uint32, 0), ('response_length', ctypes.c_uint32, 4), ('events_reported', ctypes.c_uint32, 8), ('reserved', ctypes.c_uint32, 12)])
@c.record
class struct_ib_uverbs_ex_modify_wq(c.Struct):
  SIZE = 24
  attr_mask: int
  wq_handle: int
  wq_state: int
  curr_wq_state: int
  flags: int
  flags_mask: int
struct_ib_uverbs_ex_modify_wq.register_fields([('attr_mask', ctypes.c_uint32, 0), ('wq_handle', ctypes.c_uint32, 4), ('wq_state', ctypes.c_uint32, 8), ('curr_wq_state', ctypes.c_uint32, 12), ('flags', ctypes.c_uint32, 16), ('flags_mask', ctypes.c_uint32, 20)])
@c.record
class struct_ib_uverbs_ex_create_rwq_ind_table(c.Struct):
  SIZE = 8
  comp_mask: int
  log_ind_tbl_size: int
  wq_handles: c.Array[ctypes.c_uint32, Literal[0]]
struct_ib_uverbs_ex_create_rwq_ind_table.register_fields([('comp_mask', ctypes.c_uint32, 0), ('log_ind_tbl_size', ctypes.c_uint32, 4), ('wq_handles', c.Array[ctypes.c_uint32, Literal[0]], 8)])
@c.record
class struct_ib_uverbs_ex_create_rwq_ind_table_resp(c.Struct):
  SIZE = 16
  comp_mask: int
  response_length: int
  ind_tbl_handle: int
  ind_tbl_num: int
struct_ib_uverbs_ex_create_rwq_ind_table_resp.register_fields([('comp_mask', ctypes.c_uint32, 0), ('response_length', ctypes.c_uint32, 4), ('ind_tbl_handle', ctypes.c_uint32, 8), ('ind_tbl_num', ctypes.c_uint32, 12)])
@c.record
class struct_ib_uverbs_ex_destroy_rwq_ind_table(c.Struct):
  SIZE = 8
  comp_mask: int
  ind_tbl_handle: int
struct_ib_uverbs_ex_destroy_rwq_ind_table.register_fields([('comp_mask', ctypes.c_uint32, 0), ('ind_tbl_handle', ctypes.c_uint32, 4)])
@c.record
class struct_ib_uverbs_cq_moderation(c.Struct):
  SIZE = 4
  cq_count: int
  cq_period: int
struct_ib_uverbs_cq_moderation.register_fields([('cq_count', ctypes.c_uint16, 0), ('cq_period', ctypes.c_uint16, 2)])
@c.record
class struct_ib_uverbs_ex_modify_cq(c.Struct):
  SIZE = 16
  cq_handle: int
  attr_mask: int
  attr: struct_ib_uverbs_cq_moderation
  reserved: int
struct_ib_uverbs_ex_modify_cq.register_fields([('cq_handle', ctypes.c_uint32, 0), ('attr_mask', ctypes.c_uint32, 4), ('attr', struct_ib_uverbs_cq_moderation, 8), ('reserved', ctypes.c_uint32, 12)])
enum_ib_uverbs_device_cap_flags: dict[int, str] = {(IB_UVERBS_DEVICE_RESIZE_MAX_WR:=1): 'IB_UVERBS_DEVICE_RESIZE_MAX_WR', (IB_UVERBS_DEVICE_BAD_PKEY_CNTR:=2): 'IB_UVERBS_DEVICE_BAD_PKEY_CNTR', (IB_UVERBS_DEVICE_BAD_QKEY_CNTR:=4): 'IB_UVERBS_DEVICE_BAD_QKEY_CNTR', (IB_UVERBS_DEVICE_RAW_MULTI:=8): 'IB_UVERBS_DEVICE_RAW_MULTI', (IB_UVERBS_DEVICE_AUTO_PATH_MIG:=16): 'IB_UVERBS_DEVICE_AUTO_PATH_MIG', (IB_UVERBS_DEVICE_CHANGE_PHY_PORT:=32): 'IB_UVERBS_DEVICE_CHANGE_PHY_PORT', (IB_UVERBS_DEVICE_UD_AV_PORT_ENFORCE:=64): 'IB_UVERBS_DEVICE_UD_AV_PORT_ENFORCE', (IB_UVERBS_DEVICE_CURR_QP_STATE_MOD:=128): 'IB_UVERBS_DEVICE_CURR_QP_STATE_MOD', (IB_UVERBS_DEVICE_SHUTDOWN_PORT:=256): 'IB_UVERBS_DEVICE_SHUTDOWN_PORT', (IB_UVERBS_DEVICE_PORT_ACTIVE_EVENT:=1024): 'IB_UVERBS_DEVICE_PORT_ACTIVE_EVENT', (IB_UVERBS_DEVICE_SYS_IMAGE_GUID:=2048): 'IB_UVERBS_DEVICE_SYS_IMAGE_GUID', (IB_UVERBS_DEVICE_RC_RNR_NAK_GEN:=4096): 'IB_UVERBS_DEVICE_RC_RNR_NAK_GEN', (IB_UVERBS_DEVICE_SRQ_RESIZE:=8192): 'IB_UVERBS_DEVICE_SRQ_RESIZE', (IB_UVERBS_DEVICE_N_NOTIFY_CQ:=16384): 'IB_UVERBS_DEVICE_N_NOTIFY_CQ', (IB_UVERBS_DEVICE_MEM_WINDOW:=131072): 'IB_UVERBS_DEVICE_MEM_WINDOW', (IB_UVERBS_DEVICE_UD_IP_CSUM:=262144): 'IB_UVERBS_DEVICE_UD_IP_CSUM', (IB_UVERBS_DEVICE_XRC:=1048576): 'IB_UVERBS_DEVICE_XRC', (IB_UVERBS_DEVICE_MEM_MGT_EXTENSIONS:=2097152): 'IB_UVERBS_DEVICE_MEM_MGT_EXTENSIONS', (IB_UVERBS_DEVICE_MEM_WINDOW_TYPE_2A:=8388608): 'IB_UVERBS_DEVICE_MEM_WINDOW_TYPE_2A', (IB_UVERBS_DEVICE_MEM_WINDOW_TYPE_2B:=16777216): 'IB_UVERBS_DEVICE_MEM_WINDOW_TYPE_2B', (IB_UVERBS_DEVICE_RC_IP_CSUM:=33554432): 'IB_UVERBS_DEVICE_RC_IP_CSUM', (IB_UVERBS_DEVICE_RAW_IP_CSUM:=67108864): 'IB_UVERBS_DEVICE_RAW_IP_CSUM', (IB_UVERBS_DEVICE_MANAGED_FLOW_STEERING:=536870912): 'IB_UVERBS_DEVICE_MANAGED_FLOW_STEERING', (IB_UVERBS_DEVICE_RAW_SCATTER_FCS:=17179869184): 'IB_UVERBS_DEVICE_RAW_SCATTER_FCS', (IB_UVERBS_DEVICE_PCI_WRITE_END_PADDING:=68719476736): 'IB_UVERBS_DEVICE_PCI_WRITE_END_PADDING', (IB_UVERBS_DEVICE_FLUSH_GLOBAL:=274877906944): 'IB_UVERBS_DEVICE_FLUSH_GLOBAL', (IB_UVERBS_DEVICE_FLUSH_PERSISTENT:=549755813888): 'IB_UVERBS_DEVICE_FLUSH_PERSISTENT', (IB_UVERBS_DEVICE_ATOMIC_WRITE:=1099511627776): 'IB_UVERBS_DEVICE_ATOMIC_WRITE'}
enum_ib_uverbs_raw_packet_caps: dict[int, str] = {(IB_UVERBS_RAW_PACKET_CAP_CVLAN_STRIPPING:=1): 'IB_UVERBS_RAW_PACKET_CAP_CVLAN_STRIPPING', (IB_UVERBS_RAW_PACKET_CAP_SCATTER_FCS:=2): 'IB_UVERBS_RAW_PACKET_CAP_SCATTER_FCS', (IB_UVERBS_RAW_PACKET_CAP_IP_CSUM:=4): 'IB_UVERBS_RAW_PACKET_CAP_IP_CSUM', (IB_UVERBS_RAW_PACKET_CAP_DELAY_DROP:=8): 'IB_UVERBS_RAW_PACKET_CAP_DELAY_DROP'}
vext_field_avail = lambda type,fld,sz: (offsetof(type, fld) < (sz)) # type: ignore
IBV_DEVICE_RAW_SCATTER_FCS = (1 << 34)
IBV_DEVICE_PCI_WRITE_END_PADDING = (1 << 36)
ibv_query_port = lambda context,port_num,port_attr: ___ibv_query_port(context, port_num, port_attr) # type: ignore
ibv_reg_mr = lambda pd,addr,length,access: __ibv_reg_mr(pd, addr, length, access, __builtin_constant_p( ((int)(access) & IBV_ACCESS_OPTIONAL_RANGE) == 0)) # type: ignore
ibv_reg_mr_iova = lambda pd,addr,length,iova,access: __ibv_reg_mr_iova(pd, addr, length, iova, access, __builtin_constant_p( ((access) & IBV_ACCESS_OPTIONAL_RANGE) == 0)) # type: ignore
ETHERNET_LL_SIZE = 6
IB_ROCE_UDP_ENCAP_VALID_PORT_MIN = (0xC000)
IB_ROCE_UDP_ENCAP_VALID_PORT_MAX = (0xFFFF)
IB_GRH_FLOWLABEL_MASK = (0x000FFFFF)
IBV_FLOW_ACTION_ESP_KEYMAT_AES_GCM = IB_UVERBS_FLOW_ACTION_ESP_KEYMAT_AES_GCM
IBV_FLOW_ACTION_IV_ALGO_SEQ = IB_UVERBS_FLOW_ACTION_IV_ALGO_SEQ
IBV_FLOW_ACTION_ESP_REPLAY_NONE = IB_UVERBS_FLOW_ACTION_ESP_REPLAY_NONE
IBV_FLOW_ACTION_ESP_REPLAY_BMP = IB_UVERBS_FLOW_ACTION_ESP_REPLAY_BMP
IBV_FLOW_ACTION_ESP_FLAGS_INLINE_CRYPTO = IB_UVERBS_FLOW_ACTION_ESP_FLAGS_INLINE_CRYPTO
IBV_FLOW_ACTION_ESP_FLAGS_FULL_OFFLOAD = IB_UVERBS_FLOW_ACTION_ESP_FLAGS_FULL_OFFLOAD
IBV_FLOW_ACTION_ESP_FLAGS_TUNNEL = IB_UVERBS_FLOW_ACTION_ESP_FLAGS_TUNNEL
IBV_FLOW_ACTION_ESP_FLAGS_TRANSPORT = IB_UVERBS_FLOW_ACTION_ESP_FLAGS_TRANSPORT
IBV_FLOW_ACTION_ESP_FLAGS_DECRYPT = IB_UVERBS_FLOW_ACTION_ESP_FLAGS_DECRYPT
IBV_FLOW_ACTION_ESP_FLAGS_ENCRYPT = IB_UVERBS_FLOW_ACTION_ESP_FLAGS_ENCRYPT
IBV_FLOW_ACTION_ESP_FLAGS_ESN_NEW_WINDOW = IB_UVERBS_FLOW_ACTION_ESP_FLAGS_ESN_NEW_WINDOW
IBV_ADVISE_MR_ADVICE_PREFETCH = IB_UVERBS_ADVISE_MR_ADVICE_PREFETCH
IBV_ADVISE_MR_ADVICE_PREFETCH_WRITE = IB_UVERBS_ADVISE_MR_ADVICE_PREFETCH_WRITE
IBV_ADVISE_MR_ADVICE_PREFETCH_NO_FAULT = IB_UVERBS_ADVISE_MR_ADVICE_PREFETCH_NO_FAULT
IBV_ADVISE_MR_FLAG_FLUSH = IB_UVERBS_ADVISE_MR_FLAG_FLUSH
IBV_QPF_GRH_REQUIRED = IB_UVERBS_QPF_GRH_REQUIRED
IBV_ACCESS_OPTIONAL_RANGE = IB_UVERBS_ACCESS_OPTIONAL_RANGE
IB_UVERBS_ACCESS_OPTIONAL_FIRST = (1 << 20)
IB_UVERBS_ACCESS_OPTIONAL_LAST = (1 << 29)
IB_USER_VERBS_ABI_VERSION = 6
IB_USER_VERBS_CMD_THRESHOLD = 50
IB_USER_VERBS_CMD_COMMAND_MASK = 0xff
IB_USER_VERBS_CMD_FLAG_EXTENDED = 0x80000000
IB_USER_VERBS_MAX_LOG_IND_TBL_SIZE = 0x0d
IB_DEVICE_NAME_MAX = 64