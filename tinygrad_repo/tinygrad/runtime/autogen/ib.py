# mypy: ignore-errors
import ctypes
from tinygrad.runtime.support.c import DLL, Struct, CEnum, _IO, _IOW, _IOR, _IOWR
dll = DLL('ib', 'ibverbs', use_errno=True)
class union_ibv_gid(ctypes.Union): pass
uint8_t = ctypes.c_ubyte
class union_ibv_gid_global(Struct): pass
__be64 = ctypes.c_uint64
union_ibv_gid_global._fields_ = [
  ('subnet_prefix', ctypes.c_uint64),
  ('interface_id', ctypes.c_uint64),
]
union_ibv_gid._fields_ = [
  ('raw', (uint8_t * 16)),
  ('global', union_ibv_gid_global),
]
enum_ibv_gid_type = CEnum(ctypes.c_uint32)
IBV_GID_TYPE_IB = enum_ibv_gid_type.define('IBV_GID_TYPE_IB', 0)
IBV_GID_TYPE_ROCE_V1 = enum_ibv_gid_type.define('IBV_GID_TYPE_ROCE_V1', 1)
IBV_GID_TYPE_ROCE_V2 = enum_ibv_gid_type.define('IBV_GID_TYPE_ROCE_V2', 2)

class struct_ibv_gid_entry(Struct): pass
uint32_t = ctypes.c_uint32
struct_ibv_gid_entry._fields_ = [
  ('gid', union_ibv_gid),
  ('gid_index', uint32_t),
  ('port_num', uint32_t),
  ('gid_type', uint32_t),
  ('ndev_ifindex', uint32_t),
]
enum_ibv_node_type = CEnum(ctypes.c_int32)
IBV_NODE_UNKNOWN = enum_ibv_node_type.define('IBV_NODE_UNKNOWN', -1)
IBV_NODE_CA = enum_ibv_node_type.define('IBV_NODE_CA', 1)
IBV_NODE_SWITCH = enum_ibv_node_type.define('IBV_NODE_SWITCH', 2)
IBV_NODE_ROUTER = enum_ibv_node_type.define('IBV_NODE_ROUTER', 3)
IBV_NODE_RNIC = enum_ibv_node_type.define('IBV_NODE_RNIC', 4)
IBV_NODE_USNIC = enum_ibv_node_type.define('IBV_NODE_USNIC', 5)
IBV_NODE_USNIC_UDP = enum_ibv_node_type.define('IBV_NODE_USNIC_UDP', 6)
IBV_NODE_UNSPECIFIED = enum_ibv_node_type.define('IBV_NODE_UNSPECIFIED', 7)

enum_ibv_transport_type = CEnum(ctypes.c_int32)
IBV_TRANSPORT_UNKNOWN = enum_ibv_transport_type.define('IBV_TRANSPORT_UNKNOWN', -1)
IBV_TRANSPORT_IB = enum_ibv_transport_type.define('IBV_TRANSPORT_IB', 0)
IBV_TRANSPORT_IWARP = enum_ibv_transport_type.define('IBV_TRANSPORT_IWARP', 1)
IBV_TRANSPORT_USNIC = enum_ibv_transport_type.define('IBV_TRANSPORT_USNIC', 2)
IBV_TRANSPORT_USNIC_UDP = enum_ibv_transport_type.define('IBV_TRANSPORT_USNIC_UDP', 3)
IBV_TRANSPORT_UNSPECIFIED = enum_ibv_transport_type.define('IBV_TRANSPORT_UNSPECIFIED', 4)

enum_ibv_device_cap_flags = CEnum(ctypes.c_uint32)
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

enum_ibv_fork_status = CEnum(ctypes.c_uint32)
IBV_FORK_DISABLED = enum_ibv_fork_status.define('IBV_FORK_DISABLED', 0)
IBV_FORK_ENABLED = enum_ibv_fork_status.define('IBV_FORK_ENABLED', 1)
IBV_FORK_UNNEEDED = enum_ibv_fork_status.define('IBV_FORK_UNNEEDED', 2)

enum_ibv_atomic_cap = CEnum(ctypes.c_uint32)
IBV_ATOMIC_NONE = enum_ibv_atomic_cap.define('IBV_ATOMIC_NONE', 0)
IBV_ATOMIC_HCA = enum_ibv_atomic_cap.define('IBV_ATOMIC_HCA', 1)
IBV_ATOMIC_GLOB = enum_ibv_atomic_cap.define('IBV_ATOMIC_GLOB', 2)

class struct_ibv_alloc_dm_attr(Struct): pass
size_t = ctypes.c_uint64
struct_ibv_alloc_dm_attr._fields_ = [
  ('length', size_t),
  ('log_align_req', uint32_t),
  ('comp_mask', uint32_t),
]
enum_ibv_dm_mask = CEnum(ctypes.c_uint32)
IBV_DM_MASK_HANDLE = enum_ibv_dm_mask.define('IBV_DM_MASK_HANDLE', 1)

class struct_ibv_dm(Struct): pass
class struct_ibv_context(Struct): pass
class struct_ibv_device(Struct): pass
class struct__ibv_device_ops(Struct): pass
struct__ibv_device_ops._fields_ = [
  ('_dummy1', ctypes.CFUNCTYPE(ctypes.POINTER(struct_ibv_context), ctypes.POINTER(struct_ibv_device), ctypes.c_int32)),
  ('_dummy2', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ibv_context))),
]
struct_ibv_device._fields_ = [
  ('_ops', struct__ibv_device_ops),
  ('node_type', enum_ibv_node_type),
  ('transport_type', enum_ibv_transport_type),
  ('name', (ctypes.c_char * 64)),
  ('dev_name', (ctypes.c_char * 64)),
  ('dev_path', (ctypes.c_char * 256)),
  ('ibdev_path', (ctypes.c_char * 256)),
]
class struct_ibv_context_ops(Struct): pass
class struct_ibv_device_attr(Struct): pass
uint64_t = ctypes.c_uint64
uint16_t = ctypes.c_uint16
struct_ibv_device_attr._fields_ = [
  ('fw_ver', (ctypes.c_char * 64)),
  ('node_guid', ctypes.c_uint64),
  ('sys_image_guid', ctypes.c_uint64),
  ('max_mr_size', uint64_t),
  ('page_size_cap', uint64_t),
  ('vendor_id', uint32_t),
  ('vendor_part_id', uint32_t),
  ('hw_ver', uint32_t),
  ('max_qp', ctypes.c_int32),
  ('max_qp_wr', ctypes.c_int32),
  ('device_cap_flags', ctypes.c_uint32),
  ('max_sge', ctypes.c_int32),
  ('max_sge_rd', ctypes.c_int32),
  ('max_cq', ctypes.c_int32),
  ('max_cqe', ctypes.c_int32),
  ('max_mr', ctypes.c_int32),
  ('max_pd', ctypes.c_int32),
  ('max_qp_rd_atom', ctypes.c_int32),
  ('max_ee_rd_atom', ctypes.c_int32),
  ('max_res_rd_atom', ctypes.c_int32),
  ('max_qp_init_rd_atom', ctypes.c_int32),
  ('max_ee_init_rd_atom', ctypes.c_int32),
  ('atomic_cap', enum_ibv_atomic_cap),
  ('max_ee', ctypes.c_int32),
  ('max_rdd', ctypes.c_int32),
  ('max_mw', ctypes.c_int32),
  ('max_raw_ipv6_qp', ctypes.c_int32),
  ('max_raw_ethy_qp', ctypes.c_int32),
  ('max_mcast_grp', ctypes.c_int32),
  ('max_mcast_qp_attach', ctypes.c_int32),
  ('max_total_mcast_qp_attach', ctypes.c_int32),
  ('max_ah', ctypes.c_int32),
  ('max_fmr', ctypes.c_int32),
  ('max_map_per_fmr', ctypes.c_int32),
  ('max_srq', ctypes.c_int32),
  ('max_srq_wr', ctypes.c_int32),
  ('max_srq_sge', ctypes.c_int32),
  ('max_pkeys', uint16_t),
  ('local_ca_ack_delay', uint8_t),
  ('phys_port_cnt', uint8_t),
]
class struct__compat_ibv_port_attr(Struct): pass
class struct_ibv_mw(Struct): pass
class struct_ibv_pd(Struct): pass
struct_ibv_pd._fields_ = [
  ('context', ctypes.POINTER(struct_ibv_context)),
  ('handle', uint32_t),
]
enum_ibv_mw_type = CEnum(ctypes.c_uint32)
IBV_MW_TYPE_1 = enum_ibv_mw_type.define('IBV_MW_TYPE_1', 1)
IBV_MW_TYPE_2 = enum_ibv_mw_type.define('IBV_MW_TYPE_2', 2)

struct_ibv_mw._fields_ = [
  ('context', ctypes.POINTER(struct_ibv_context)),
  ('pd', ctypes.POINTER(struct_ibv_pd)),
  ('rkey', uint32_t),
  ('handle', uint32_t),
  ('type', enum_ibv_mw_type),
]
class struct_ibv_qp(Struct): pass
class struct_ibv_cq(Struct): pass
class struct_ibv_comp_channel(Struct): pass
struct_ibv_comp_channel._fields_ = [
  ('context', ctypes.POINTER(struct_ibv_context)),
  ('fd', ctypes.c_int32),
  ('refcnt', ctypes.c_int32),
]
class pthread_mutex_t(ctypes.Union): pass
class struct___pthread_mutex_s(Struct): pass
class struct___pthread_internal_list(Struct): pass
__pthread_list_t = struct___pthread_internal_list
struct___pthread_internal_list._fields_ = [
  ('__prev', ctypes.POINTER(struct___pthread_internal_list)),
  ('__next', ctypes.POINTER(struct___pthread_internal_list)),
]
struct___pthread_mutex_s._fields_ = [
  ('__lock', ctypes.c_int32),
  ('__count', ctypes.c_uint32),
  ('__owner', ctypes.c_int32),
  ('__nusers', ctypes.c_uint32),
  ('__kind', ctypes.c_int32),
  ('__spins', ctypes.c_int16),
  ('__elision', ctypes.c_int16),
  ('__list', struct___pthread_internal_list),
]
pthread_mutex_t._fields_ = [
  ('__data', struct___pthread_mutex_s),
  ('__size', (ctypes.c_char * 40)),
  ('__align', ctypes.c_int64),
]
class pthread_cond_t(ctypes.Union): pass
class struct___pthread_cond_s(Struct): pass
class __atomic_wide_counter(ctypes.Union): pass
class __atomic_wide_counter___value32(Struct): pass
__atomic_wide_counter___value32._fields_ = [
  ('__low', ctypes.c_uint32),
  ('__high', ctypes.c_uint32),
]
__atomic_wide_counter._fields_ = [
  ('__value64', ctypes.c_uint64),
  ('__value32', __atomic_wide_counter___value32),
]
struct___pthread_cond_s._fields_ = [
  ('__wseq', __atomic_wide_counter),
  ('__g1_start', __atomic_wide_counter),
  ('__g_refs', (ctypes.c_uint32 * 2)),
  ('__g_size', (ctypes.c_uint32 * 2)),
  ('__g1_orig_size', ctypes.c_uint32),
  ('__wrefs', ctypes.c_uint32),
  ('__g_signals', (ctypes.c_uint32 * 2)),
]
pthread_cond_t._fields_ = [
  ('__data', struct___pthread_cond_s),
  ('__size', (ctypes.c_char * 48)),
  ('__align', ctypes.c_int64),
]
struct_ibv_cq._fields_ = [
  ('context', ctypes.POINTER(struct_ibv_context)),
  ('channel', ctypes.POINTER(struct_ibv_comp_channel)),
  ('cq_context', ctypes.c_void_p),
  ('handle', uint32_t),
  ('cqe', ctypes.c_int32),
  ('mutex', pthread_mutex_t),
  ('cond', pthread_cond_t),
  ('comp_events_completed', uint32_t),
  ('async_events_completed', uint32_t),
]
class struct_ibv_srq(Struct): pass
struct_ibv_srq._fields_ = [
  ('context', ctypes.POINTER(struct_ibv_context)),
  ('srq_context', ctypes.c_void_p),
  ('pd', ctypes.POINTER(struct_ibv_pd)),
  ('handle', uint32_t),
  ('mutex', pthread_mutex_t),
  ('cond', pthread_cond_t),
  ('events_completed', uint32_t),
]
enum_ibv_qp_state = CEnum(ctypes.c_uint32)
IBV_QPS_RESET = enum_ibv_qp_state.define('IBV_QPS_RESET', 0)
IBV_QPS_INIT = enum_ibv_qp_state.define('IBV_QPS_INIT', 1)
IBV_QPS_RTR = enum_ibv_qp_state.define('IBV_QPS_RTR', 2)
IBV_QPS_RTS = enum_ibv_qp_state.define('IBV_QPS_RTS', 3)
IBV_QPS_SQD = enum_ibv_qp_state.define('IBV_QPS_SQD', 4)
IBV_QPS_SQE = enum_ibv_qp_state.define('IBV_QPS_SQE', 5)
IBV_QPS_ERR = enum_ibv_qp_state.define('IBV_QPS_ERR', 6)
IBV_QPS_UNKNOWN = enum_ibv_qp_state.define('IBV_QPS_UNKNOWN', 7)

enum_ibv_qp_type = CEnum(ctypes.c_uint32)
IBV_QPT_RC = enum_ibv_qp_type.define('IBV_QPT_RC', 2)
IBV_QPT_UC = enum_ibv_qp_type.define('IBV_QPT_UC', 3)
IBV_QPT_UD = enum_ibv_qp_type.define('IBV_QPT_UD', 4)
IBV_QPT_RAW_PACKET = enum_ibv_qp_type.define('IBV_QPT_RAW_PACKET', 8)
IBV_QPT_XRC_SEND = enum_ibv_qp_type.define('IBV_QPT_XRC_SEND', 9)
IBV_QPT_XRC_RECV = enum_ibv_qp_type.define('IBV_QPT_XRC_RECV', 10)
IBV_QPT_DRIVER = enum_ibv_qp_type.define('IBV_QPT_DRIVER', 255)

struct_ibv_qp._fields_ = [
  ('context', ctypes.POINTER(struct_ibv_context)),
  ('qp_context', ctypes.c_void_p),
  ('pd', ctypes.POINTER(struct_ibv_pd)),
  ('send_cq', ctypes.POINTER(struct_ibv_cq)),
  ('recv_cq', ctypes.POINTER(struct_ibv_cq)),
  ('srq', ctypes.POINTER(struct_ibv_srq)),
  ('handle', uint32_t),
  ('qp_num', uint32_t),
  ('state', enum_ibv_qp_state),
  ('qp_type', enum_ibv_qp_type),
  ('mutex', pthread_mutex_t),
  ('cond', pthread_cond_t),
  ('events_completed', uint32_t),
]
class struct_ibv_mw_bind(Struct): pass
class struct_ibv_mw_bind_info(Struct): pass
class struct_ibv_mr(Struct): pass
struct_ibv_mr._fields_ = [
  ('context', ctypes.POINTER(struct_ibv_context)),
  ('pd', ctypes.POINTER(struct_ibv_pd)),
  ('addr', ctypes.c_void_p),
  ('length', size_t),
  ('handle', uint32_t),
  ('lkey', uint32_t),
  ('rkey', uint32_t),
]
struct_ibv_mw_bind_info._fields_ = [
  ('mr', ctypes.POINTER(struct_ibv_mr)),
  ('addr', uint64_t),
  ('length', uint64_t),
  ('mw_access_flags', ctypes.c_uint32),
]
struct_ibv_mw_bind._fields_ = [
  ('wr_id', uint64_t),
  ('send_flags', ctypes.c_uint32),
  ('bind_info', struct_ibv_mw_bind_info),
]
class struct_ibv_wc(Struct): pass
enum_ibv_wc_status = CEnum(ctypes.c_uint32)
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

enum_ibv_wc_opcode = CEnum(ctypes.c_uint32)
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

class struct_ibv_wc_0(ctypes.Union): pass
__be32 = ctypes.c_uint32
struct_ibv_wc_0._fields_ = [
  ('imm_data', ctypes.c_uint32),
  ('invalidated_rkey', uint32_t),
]
struct_ibv_wc._anonymous_ = ['_0']
struct_ibv_wc._fields_ = [
  ('wr_id', uint64_t),
  ('status', enum_ibv_wc_status),
  ('opcode', enum_ibv_wc_opcode),
  ('vendor_err', uint32_t),
  ('byte_len', uint32_t),
  ('_0', struct_ibv_wc_0),
  ('qp_num', uint32_t),
  ('src_qp', uint32_t),
  ('wc_flags', ctypes.c_uint32),
  ('pkey_index', uint16_t),
  ('slid', uint16_t),
  ('sl', uint8_t),
  ('dlid_path_bits', uint8_t),
]
class struct_ibv_recv_wr(Struct): pass
class struct_ibv_sge(Struct): pass
struct_ibv_sge._fields_ = [
  ('addr', uint64_t),
  ('length', uint32_t),
  ('lkey', uint32_t),
]
struct_ibv_recv_wr._fields_ = [
  ('wr_id', uint64_t),
  ('next', ctypes.POINTER(struct_ibv_recv_wr)),
  ('sg_list', ctypes.POINTER(struct_ibv_sge)),
  ('num_sge', ctypes.c_int32),
]
class struct_ibv_send_wr(Struct): pass
enum_ibv_wr_opcode = CEnum(ctypes.c_uint32)
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

class struct_ibv_send_wr_0(ctypes.Union): pass
struct_ibv_send_wr_0._fields_ = [
  ('imm_data', ctypes.c_uint32),
  ('invalidate_rkey', uint32_t),
]
class struct_ibv_send_wr_wr(ctypes.Union): pass
class struct_ibv_send_wr_wr_rdma(Struct): pass
struct_ibv_send_wr_wr_rdma._fields_ = [
  ('remote_addr', uint64_t),
  ('rkey', uint32_t),
]
class struct_ibv_send_wr_wr_atomic(Struct): pass
struct_ibv_send_wr_wr_atomic._fields_ = [
  ('remote_addr', uint64_t),
  ('compare_add', uint64_t),
  ('swap', uint64_t),
  ('rkey', uint32_t),
]
class struct_ibv_send_wr_wr_ud(Struct): pass
class struct_ibv_ah(Struct): pass
struct_ibv_ah._fields_ = [
  ('context', ctypes.POINTER(struct_ibv_context)),
  ('pd', ctypes.POINTER(struct_ibv_pd)),
  ('handle', uint32_t),
]
struct_ibv_send_wr_wr_ud._fields_ = [
  ('ah', ctypes.POINTER(struct_ibv_ah)),
  ('remote_qpn', uint32_t),
  ('remote_qkey', uint32_t),
]
struct_ibv_send_wr_wr._fields_ = [
  ('rdma', struct_ibv_send_wr_wr_rdma),
  ('atomic', struct_ibv_send_wr_wr_atomic),
  ('ud', struct_ibv_send_wr_wr_ud),
]
class struct_ibv_send_wr_qp_type(ctypes.Union): pass
class struct_ibv_send_wr_qp_type_xrc(Struct): pass
struct_ibv_send_wr_qp_type_xrc._fields_ = [
  ('remote_srqn', uint32_t),
]
struct_ibv_send_wr_qp_type._fields_ = [
  ('xrc', struct_ibv_send_wr_qp_type_xrc),
]
class struct_ibv_send_wr_1(ctypes.Union): pass
class struct_ibv_send_wr_1_bind_mw(Struct): pass
struct_ibv_send_wr_1_bind_mw._fields_ = [
  ('mw', ctypes.POINTER(struct_ibv_mw)),
  ('rkey', uint32_t),
  ('bind_info', struct_ibv_mw_bind_info),
]
class struct_ibv_send_wr_1_tso(Struct): pass
struct_ibv_send_wr_1_tso._fields_ = [
  ('hdr', ctypes.c_void_p),
  ('hdr_sz', uint16_t),
  ('mss', uint16_t),
]
struct_ibv_send_wr_1._fields_ = [
  ('bind_mw', struct_ibv_send_wr_1_bind_mw),
  ('tso', struct_ibv_send_wr_1_tso),
]
struct_ibv_send_wr._anonymous_ = ['_0', '_1']
struct_ibv_send_wr._fields_ = [
  ('wr_id', uint64_t),
  ('next', ctypes.POINTER(struct_ibv_send_wr)),
  ('sg_list', ctypes.POINTER(struct_ibv_sge)),
  ('num_sge', ctypes.c_int32),
  ('opcode', enum_ibv_wr_opcode),
  ('send_flags', ctypes.c_uint32),
  ('_0', struct_ibv_send_wr_0),
  ('wr', struct_ibv_send_wr_wr),
  ('qp_type', struct_ibv_send_wr_qp_type),
  ('_1', struct_ibv_send_wr_1),
]
struct_ibv_context_ops._fields_ = [
  ('_compat_query_device', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_ibv_context), ctypes.POINTER(struct_ibv_device_attr))),
  ('_compat_query_port', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_ibv_context), uint8_t, ctypes.POINTER(struct__compat_ibv_port_attr))),
  ('_compat_alloc_pd', ctypes.CFUNCTYPE(ctypes.c_void_p, )),
  ('_compat_dealloc_pd', ctypes.CFUNCTYPE(ctypes.c_void_p, )),
  ('_compat_reg_mr', ctypes.CFUNCTYPE(ctypes.c_void_p, )),
  ('_compat_rereg_mr', ctypes.CFUNCTYPE(ctypes.c_void_p, )),
  ('_compat_dereg_mr', ctypes.CFUNCTYPE(ctypes.c_void_p, )),
  ('alloc_mw', ctypes.CFUNCTYPE(ctypes.POINTER(struct_ibv_mw), ctypes.POINTER(struct_ibv_pd), enum_ibv_mw_type)),
  ('bind_mw', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_ibv_qp), ctypes.POINTER(struct_ibv_mw), ctypes.POINTER(struct_ibv_mw_bind))),
  ('dealloc_mw', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_ibv_mw))),
  ('_compat_create_cq', ctypes.CFUNCTYPE(ctypes.c_void_p, )),
  ('poll_cq', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_ibv_cq), ctypes.c_int32, ctypes.POINTER(struct_ibv_wc))),
  ('req_notify_cq', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_ibv_cq), ctypes.c_int32)),
  ('_compat_cq_event', ctypes.CFUNCTYPE(ctypes.c_void_p, )),
  ('_compat_resize_cq', ctypes.CFUNCTYPE(ctypes.c_void_p, )),
  ('_compat_destroy_cq', ctypes.CFUNCTYPE(ctypes.c_void_p, )),
  ('_compat_create_srq', ctypes.CFUNCTYPE(ctypes.c_void_p, )),
  ('_compat_modify_srq', ctypes.CFUNCTYPE(ctypes.c_void_p, )),
  ('_compat_query_srq', ctypes.CFUNCTYPE(ctypes.c_void_p, )),
  ('_compat_destroy_srq', ctypes.CFUNCTYPE(ctypes.c_void_p, )),
  ('post_srq_recv', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_ibv_srq), ctypes.POINTER(struct_ibv_recv_wr), ctypes.POINTER(ctypes.POINTER(struct_ibv_recv_wr)))),
  ('_compat_create_qp', ctypes.CFUNCTYPE(ctypes.c_void_p, )),
  ('_compat_query_qp', ctypes.CFUNCTYPE(ctypes.c_void_p, )),
  ('_compat_modify_qp', ctypes.CFUNCTYPE(ctypes.c_void_p, )),
  ('_compat_destroy_qp', ctypes.CFUNCTYPE(ctypes.c_void_p, )),
  ('post_send', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_ibv_qp), ctypes.POINTER(struct_ibv_send_wr), ctypes.POINTER(ctypes.POINTER(struct_ibv_send_wr)))),
  ('post_recv', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_ibv_qp), ctypes.POINTER(struct_ibv_recv_wr), ctypes.POINTER(ctypes.POINTER(struct_ibv_recv_wr)))),
  ('_compat_create_ah', ctypes.CFUNCTYPE(ctypes.c_void_p, )),
  ('_compat_destroy_ah', ctypes.CFUNCTYPE(ctypes.c_void_p, )),
  ('_compat_attach_mcast', ctypes.CFUNCTYPE(ctypes.c_void_p, )),
  ('_compat_detach_mcast', ctypes.CFUNCTYPE(ctypes.c_void_p, )),
  ('_compat_async_event', ctypes.CFUNCTYPE(ctypes.c_void_p, )),
]
struct_ibv_context._fields_ = [
  ('device', ctypes.POINTER(struct_ibv_device)),
  ('ops', struct_ibv_context_ops),
  ('cmd_fd', ctypes.c_int32),
  ('async_fd', ctypes.c_int32),
  ('num_comp_vectors', ctypes.c_int32),
  ('mutex', pthread_mutex_t),
  ('abi_compat', ctypes.c_void_p),
]
struct_ibv_dm._fields_ = [
  ('context', ctypes.POINTER(struct_ibv_context)),
  ('memcpy_to_dm', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_ibv_dm), uint64_t, ctypes.c_void_p, size_t)),
  ('memcpy_from_dm', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p, ctypes.POINTER(struct_ibv_dm), uint64_t, size_t)),
  ('comp_mask', uint32_t),
  ('handle', uint32_t),
]
class struct_ibv_query_device_ex_input(Struct): pass
struct_ibv_query_device_ex_input._fields_ = [
  ('comp_mask', uint32_t),
]
enum_ibv_odp_transport_cap_bits = CEnum(ctypes.c_uint32)
IBV_ODP_SUPPORT_SEND = enum_ibv_odp_transport_cap_bits.define('IBV_ODP_SUPPORT_SEND', 1)
IBV_ODP_SUPPORT_RECV = enum_ibv_odp_transport_cap_bits.define('IBV_ODP_SUPPORT_RECV', 2)
IBV_ODP_SUPPORT_WRITE = enum_ibv_odp_transport_cap_bits.define('IBV_ODP_SUPPORT_WRITE', 4)
IBV_ODP_SUPPORT_READ = enum_ibv_odp_transport_cap_bits.define('IBV_ODP_SUPPORT_READ', 8)
IBV_ODP_SUPPORT_ATOMIC = enum_ibv_odp_transport_cap_bits.define('IBV_ODP_SUPPORT_ATOMIC', 16)
IBV_ODP_SUPPORT_SRQ_RECV = enum_ibv_odp_transport_cap_bits.define('IBV_ODP_SUPPORT_SRQ_RECV', 32)

class struct_ibv_odp_caps(Struct): pass
class struct_ibv_odp_caps_per_transport_caps(Struct): pass
struct_ibv_odp_caps_per_transport_caps._fields_ = [
  ('rc_odp_caps', uint32_t),
  ('uc_odp_caps', uint32_t),
  ('ud_odp_caps', uint32_t),
]
struct_ibv_odp_caps._fields_ = [
  ('general_caps', uint64_t),
  ('per_transport_caps', struct_ibv_odp_caps_per_transport_caps),
]
enum_ibv_odp_general_caps = CEnum(ctypes.c_uint32)
IBV_ODP_SUPPORT = enum_ibv_odp_general_caps.define('IBV_ODP_SUPPORT', 1)
IBV_ODP_SUPPORT_IMPLICIT = enum_ibv_odp_general_caps.define('IBV_ODP_SUPPORT_IMPLICIT', 2)

class struct_ibv_tso_caps(Struct): pass
struct_ibv_tso_caps._fields_ = [
  ('max_tso', uint32_t),
  ('supported_qpts', uint32_t),
]
enum_ibv_rx_hash_function_flags = CEnum(ctypes.c_uint32)
IBV_RX_HASH_FUNC_TOEPLITZ = enum_ibv_rx_hash_function_flags.define('IBV_RX_HASH_FUNC_TOEPLITZ', 1)

enum_ibv_rx_hash_fields = CEnum(ctypes.c_uint32)
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

class struct_ibv_rss_caps(Struct): pass
struct_ibv_rss_caps._fields_ = [
  ('supported_qpts', uint32_t),
  ('max_rwq_indirection_tables', uint32_t),
  ('max_rwq_indirection_table_size', uint32_t),
  ('rx_hash_fields_mask', uint64_t),
  ('rx_hash_function', uint8_t),
]
class struct_ibv_packet_pacing_caps(Struct): pass
struct_ibv_packet_pacing_caps._fields_ = [
  ('qp_rate_limit_min', uint32_t),
  ('qp_rate_limit_max', uint32_t),
  ('supported_qpts', uint32_t),
]
enum_ibv_raw_packet_caps = CEnum(ctypes.c_uint32)
IBV_RAW_PACKET_CAP_CVLAN_STRIPPING = enum_ibv_raw_packet_caps.define('IBV_RAW_PACKET_CAP_CVLAN_STRIPPING', 1)
IBV_RAW_PACKET_CAP_SCATTER_FCS = enum_ibv_raw_packet_caps.define('IBV_RAW_PACKET_CAP_SCATTER_FCS', 2)
IBV_RAW_PACKET_CAP_IP_CSUM = enum_ibv_raw_packet_caps.define('IBV_RAW_PACKET_CAP_IP_CSUM', 4)
IBV_RAW_PACKET_CAP_DELAY_DROP = enum_ibv_raw_packet_caps.define('IBV_RAW_PACKET_CAP_DELAY_DROP', 8)

enum_ibv_tm_cap_flags = CEnum(ctypes.c_uint32)
IBV_TM_CAP_RC = enum_ibv_tm_cap_flags.define('IBV_TM_CAP_RC', 1)

class struct_ibv_tm_caps(Struct): pass
struct_ibv_tm_caps._fields_ = [
  ('max_rndv_hdr_size', uint32_t),
  ('max_num_tags', uint32_t),
  ('flags', uint32_t),
  ('max_ops', uint32_t),
  ('max_sge', uint32_t),
]
class struct_ibv_cq_moderation_caps(Struct): pass
struct_ibv_cq_moderation_caps._fields_ = [
  ('max_cq_count', uint16_t),
  ('max_cq_period', uint16_t),
]
enum_ibv_pci_atomic_op_size = CEnum(ctypes.c_uint32)
IBV_PCI_ATOMIC_OPERATION_4_BYTE_SIZE_SUP = enum_ibv_pci_atomic_op_size.define('IBV_PCI_ATOMIC_OPERATION_4_BYTE_SIZE_SUP', 1)
IBV_PCI_ATOMIC_OPERATION_8_BYTE_SIZE_SUP = enum_ibv_pci_atomic_op_size.define('IBV_PCI_ATOMIC_OPERATION_8_BYTE_SIZE_SUP', 2)
IBV_PCI_ATOMIC_OPERATION_16_BYTE_SIZE_SUP = enum_ibv_pci_atomic_op_size.define('IBV_PCI_ATOMIC_OPERATION_16_BYTE_SIZE_SUP', 4)

class struct_ibv_pci_atomic_caps(Struct): pass
struct_ibv_pci_atomic_caps._fields_ = [
  ('fetch_add', uint16_t),
  ('swap', uint16_t),
  ('compare_swap', uint16_t),
]
class struct_ibv_device_attr_ex(Struct): pass
struct_ibv_device_attr_ex._fields_ = [
  ('orig_attr', struct_ibv_device_attr),
  ('comp_mask', uint32_t),
  ('odp_caps', struct_ibv_odp_caps),
  ('completion_timestamp_mask', uint64_t),
  ('hca_core_clock', uint64_t),
  ('device_cap_flags_ex', uint64_t),
  ('tso_caps', struct_ibv_tso_caps),
  ('rss_caps', struct_ibv_rss_caps),
  ('max_wq_type_rq', uint32_t),
  ('packet_pacing_caps', struct_ibv_packet_pacing_caps),
  ('raw_packet_caps', uint32_t),
  ('tm_caps', struct_ibv_tm_caps),
  ('cq_mod_caps', struct_ibv_cq_moderation_caps),
  ('max_dm_size', uint64_t),
  ('pci_atomic_caps', struct_ibv_pci_atomic_caps),
  ('xrc_odp_caps', uint32_t),
  ('phys_port_cnt_ex', uint32_t),
]
enum_ibv_mtu = CEnum(ctypes.c_uint32)
IBV_MTU_256 = enum_ibv_mtu.define('IBV_MTU_256', 1)
IBV_MTU_512 = enum_ibv_mtu.define('IBV_MTU_512', 2)
IBV_MTU_1024 = enum_ibv_mtu.define('IBV_MTU_1024', 3)
IBV_MTU_2048 = enum_ibv_mtu.define('IBV_MTU_2048', 4)
IBV_MTU_4096 = enum_ibv_mtu.define('IBV_MTU_4096', 5)

enum_ibv_port_state = CEnum(ctypes.c_uint32)
IBV_PORT_NOP = enum_ibv_port_state.define('IBV_PORT_NOP', 0)
IBV_PORT_DOWN = enum_ibv_port_state.define('IBV_PORT_DOWN', 1)
IBV_PORT_INIT = enum_ibv_port_state.define('IBV_PORT_INIT', 2)
IBV_PORT_ARMED = enum_ibv_port_state.define('IBV_PORT_ARMED', 3)
IBV_PORT_ACTIVE = enum_ibv_port_state.define('IBV_PORT_ACTIVE', 4)
IBV_PORT_ACTIVE_DEFER = enum_ibv_port_state.define('IBV_PORT_ACTIVE_DEFER', 5)

_anonenum0 = CEnum(ctypes.c_uint32)
IBV_LINK_LAYER_UNSPECIFIED = _anonenum0.define('IBV_LINK_LAYER_UNSPECIFIED', 0)
IBV_LINK_LAYER_INFINIBAND = _anonenum0.define('IBV_LINK_LAYER_INFINIBAND', 1)
IBV_LINK_LAYER_ETHERNET = _anonenum0.define('IBV_LINK_LAYER_ETHERNET', 2)

enum_ibv_port_cap_flags = CEnum(ctypes.c_uint32)
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

enum_ibv_port_cap_flags2 = CEnum(ctypes.c_uint32)
IBV_PORT_SET_NODE_DESC_SUP = enum_ibv_port_cap_flags2.define('IBV_PORT_SET_NODE_DESC_SUP', 1)
IBV_PORT_INFO_EXT_SUP = enum_ibv_port_cap_flags2.define('IBV_PORT_INFO_EXT_SUP', 2)
IBV_PORT_VIRT_SUP = enum_ibv_port_cap_flags2.define('IBV_PORT_VIRT_SUP', 4)
IBV_PORT_SWITCH_PORT_STATE_TABLE_SUP = enum_ibv_port_cap_flags2.define('IBV_PORT_SWITCH_PORT_STATE_TABLE_SUP', 8)
IBV_PORT_LINK_WIDTH_2X_SUP = enum_ibv_port_cap_flags2.define('IBV_PORT_LINK_WIDTH_2X_SUP', 16)
IBV_PORT_LINK_SPEED_HDR_SUP = enum_ibv_port_cap_flags2.define('IBV_PORT_LINK_SPEED_HDR_SUP', 32)
IBV_PORT_LINK_SPEED_NDR_SUP = enum_ibv_port_cap_flags2.define('IBV_PORT_LINK_SPEED_NDR_SUP', 1024)
IBV_PORT_LINK_SPEED_XDR_SUP = enum_ibv_port_cap_flags2.define('IBV_PORT_LINK_SPEED_XDR_SUP', 4096)

class struct_ibv_port_attr(Struct): pass
struct_ibv_port_attr._fields_ = [
  ('state', enum_ibv_port_state),
  ('max_mtu', enum_ibv_mtu),
  ('active_mtu', enum_ibv_mtu),
  ('gid_tbl_len', ctypes.c_int32),
  ('port_cap_flags', uint32_t),
  ('max_msg_sz', uint32_t),
  ('bad_pkey_cntr', uint32_t),
  ('qkey_viol_cntr', uint32_t),
  ('pkey_tbl_len', uint16_t),
  ('lid', uint16_t),
  ('sm_lid', uint16_t),
  ('lmc', uint8_t),
  ('max_vl_num', uint8_t),
  ('sm_sl', uint8_t),
  ('subnet_timeout', uint8_t),
  ('init_type_reply', uint8_t),
  ('active_width', uint8_t),
  ('active_speed', uint8_t),
  ('phys_state', uint8_t),
  ('link_layer', uint8_t),
  ('flags', uint8_t),
  ('port_cap_flags2', uint16_t),
  ('active_speed_ex', uint32_t),
]
enum_ibv_event_type = CEnum(ctypes.c_uint32)
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

class struct_ibv_async_event(Struct): pass
class struct_ibv_async_event_element(ctypes.Union): pass
class struct_ibv_wq(Struct): pass
enum_ibv_wq_state = CEnum(ctypes.c_uint32)
IBV_WQS_RESET = enum_ibv_wq_state.define('IBV_WQS_RESET', 0)
IBV_WQS_RDY = enum_ibv_wq_state.define('IBV_WQS_RDY', 1)
IBV_WQS_ERR = enum_ibv_wq_state.define('IBV_WQS_ERR', 2)
IBV_WQS_UNKNOWN = enum_ibv_wq_state.define('IBV_WQS_UNKNOWN', 3)

enum_ibv_wq_type = CEnum(ctypes.c_uint32)
IBV_WQT_RQ = enum_ibv_wq_type.define('IBV_WQT_RQ', 0)

struct_ibv_wq._fields_ = [
  ('context', ctypes.POINTER(struct_ibv_context)),
  ('wq_context', ctypes.c_void_p),
  ('pd', ctypes.POINTER(struct_ibv_pd)),
  ('cq', ctypes.POINTER(struct_ibv_cq)),
  ('wq_num', uint32_t),
  ('handle', uint32_t),
  ('state', enum_ibv_wq_state),
  ('wq_type', enum_ibv_wq_type),
  ('post_recv', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_ibv_wq), ctypes.POINTER(struct_ibv_recv_wr), ctypes.POINTER(ctypes.POINTER(struct_ibv_recv_wr)))),
  ('mutex', pthread_mutex_t),
  ('cond', pthread_cond_t),
  ('events_completed', uint32_t),
  ('comp_mask', uint32_t),
]
struct_ibv_async_event_element._fields_ = [
  ('cq', ctypes.POINTER(struct_ibv_cq)),
  ('qp', ctypes.POINTER(struct_ibv_qp)),
  ('srq', ctypes.POINTER(struct_ibv_srq)),
  ('wq', ctypes.POINTER(struct_ibv_wq)),
  ('port_num', ctypes.c_int32),
]
struct_ibv_async_event._fields_ = [
  ('element', struct_ibv_async_event_element),
  ('event_type', enum_ibv_event_type),
]
try: (ibv_wc_status_str:=dll.ibv_wc_status_str).restype, ibv_wc_status_str.argtypes = ctypes.POINTER(ctypes.c_char), [enum_ibv_wc_status]
except AttributeError: pass

_anonenum1 = CEnum(ctypes.c_uint32)
IBV_WC_IP_CSUM_OK_SHIFT = _anonenum1.define('IBV_WC_IP_CSUM_OK_SHIFT', 2)

enum_ibv_create_cq_wc_flags = CEnum(ctypes.c_uint32)
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

_anonenum2 = CEnum(ctypes.c_uint32)
IBV_WC_STANDARD_FLAGS = _anonenum2.define('IBV_WC_STANDARD_FLAGS', 127)

_anonenum3 = CEnum(ctypes.c_uint32)
IBV_CREATE_CQ_SUP_WC_FLAGS = _anonenum3.define('IBV_CREATE_CQ_SUP_WC_FLAGS', 4095)

enum_ibv_wc_flags = CEnum(ctypes.c_uint32)
IBV_WC_GRH = enum_ibv_wc_flags.define('IBV_WC_GRH', 1)
IBV_WC_WITH_IMM = enum_ibv_wc_flags.define('IBV_WC_WITH_IMM', 2)
IBV_WC_IP_CSUM_OK = enum_ibv_wc_flags.define('IBV_WC_IP_CSUM_OK', 4)
IBV_WC_WITH_INV = enum_ibv_wc_flags.define('IBV_WC_WITH_INV', 8)
IBV_WC_TM_SYNC_REQ = enum_ibv_wc_flags.define('IBV_WC_TM_SYNC_REQ', 16)
IBV_WC_TM_MATCH = enum_ibv_wc_flags.define('IBV_WC_TM_MATCH', 32)
IBV_WC_TM_DATA_VALID = enum_ibv_wc_flags.define('IBV_WC_TM_DATA_VALID', 64)

enum_ibv_access_flags = CEnum(ctypes.c_uint32)
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

class struct_ibv_td_init_attr(Struct): pass
struct_ibv_td_init_attr._fields_ = [
  ('comp_mask', uint32_t),
]
class struct_ibv_td(Struct): pass
struct_ibv_td._fields_ = [
  ('context', ctypes.POINTER(struct_ibv_context)),
]
enum_ibv_xrcd_init_attr_mask = CEnum(ctypes.c_uint32)
IBV_XRCD_INIT_ATTR_FD = enum_ibv_xrcd_init_attr_mask.define('IBV_XRCD_INIT_ATTR_FD', 1)
IBV_XRCD_INIT_ATTR_OFLAGS = enum_ibv_xrcd_init_attr_mask.define('IBV_XRCD_INIT_ATTR_OFLAGS', 2)
IBV_XRCD_INIT_ATTR_RESERVED = enum_ibv_xrcd_init_attr_mask.define('IBV_XRCD_INIT_ATTR_RESERVED', 4)

class struct_ibv_xrcd_init_attr(Struct): pass
struct_ibv_xrcd_init_attr._fields_ = [
  ('comp_mask', uint32_t),
  ('fd', ctypes.c_int32),
  ('oflags', ctypes.c_int32),
]
class struct_ibv_xrcd(Struct): pass
struct_ibv_xrcd._fields_ = [
  ('context', ctypes.POINTER(struct_ibv_context)),
]
enum_ibv_rereg_mr_flags = CEnum(ctypes.c_uint32)
IBV_REREG_MR_CHANGE_TRANSLATION = enum_ibv_rereg_mr_flags.define('IBV_REREG_MR_CHANGE_TRANSLATION', 1)
IBV_REREG_MR_CHANGE_PD = enum_ibv_rereg_mr_flags.define('IBV_REREG_MR_CHANGE_PD', 2)
IBV_REREG_MR_CHANGE_ACCESS = enum_ibv_rereg_mr_flags.define('IBV_REREG_MR_CHANGE_ACCESS', 4)
IBV_REREG_MR_FLAGS_SUPPORTED = enum_ibv_rereg_mr_flags.define('IBV_REREG_MR_FLAGS_SUPPORTED', 7)

class struct_ibv_global_route(Struct): pass
struct_ibv_global_route._fields_ = [
  ('dgid', union_ibv_gid),
  ('flow_label', uint32_t),
  ('sgid_index', uint8_t),
  ('hop_limit', uint8_t),
  ('traffic_class', uint8_t),
]
class struct_ibv_grh(Struct): pass
__be16 = ctypes.c_uint16
struct_ibv_grh._fields_ = [
  ('version_tclass_flow', ctypes.c_uint32),
  ('paylen', ctypes.c_uint16),
  ('next_hdr', uint8_t),
  ('hop_limit', uint8_t),
  ('sgid', union_ibv_gid),
  ('dgid', union_ibv_gid),
]
enum_ibv_rate = CEnum(ctypes.c_uint32)
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

try: (ibv_rate_to_mult:=dll.ibv_rate_to_mult).restype, ibv_rate_to_mult.argtypes = ctypes.c_int32, [enum_ibv_rate]
except AttributeError: pass

try: (mult_to_ibv_rate:=dll.mult_to_ibv_rate).restype, mult_to_ibv_rate.argtypes = enum_ibv_rate, [ctypes.c_int32]
except AttributeError: pass

try: (ibv_rate_to_mbps:=dll.ibv_rate_to_mbps).restype, ibv_rate_to_mbps.argtypes = ctypes.c_int32, [enum_ibv_rate]
except AttributeError: pass

try: (mbps_to_ibv_rate:=dll.mbps_to_ibv_rate).restype, mbps_to_ibv_rate.argtypes = enum_ibv_rate, [ctypes.c_int32]
except AttributeError: pass

class struct_ibv_ah_attr(Struct): pass
struct_ibv_ah_attr._fields_ = [
  ('grh', struct_ibv_global_route),
  ('dlid', uint16_t),
  ('sl', uint8_t),
  ('src_path_bits', uint8_t),
  ('static_rate', uint8_t),
  ('is_global', uint8_t),
  ('port_num', uint8_t),
]
enum_ibv_srq_attr_mask = CEnum(ctypes.c_uint32)
IBV_SRQ_MAX_WR = enum_ibv_srq_attr_mask.define('IBV_SRQ_MAX_WR', 1)
IBV_SRQ_LIMIT = enum_ibv_srq_attr_mask.define('IBV_SRQ_LIMIT', 2)

class struct_ibv_srq_attr(Struct): pass
struct_ibv_srq_attr._fields_ = [
  ('max_wr', uint32_t),
  ('max_sge', uint32_t),
  ('srq_limit', uint32_t),
]
class struct_ibv_srq_init_attr(Struct): pass
struct_ibv_srq_init_attr._fields_ = [
  ('srq_context', ctypes.c_void_p),
  ('attr', struct_ibv_srq_attr),
]
enum_ibv_srq_type = CEnum(ctypes.c_uint32)
IBV_SRQT_BASIC = enum_ibv_srq_type.define('IBV_SRQT_BASIC', 0)
IBV_SRQT_XRC = enum_ibv_srq_type.define('IBV_SRQT_XRC', 1)
IBV_SRQT_TM = enum_ibv_srq_type.define('IBV_SRQT_TM', 2)

enum_ibv_srq_init_attr_mask = CEnum(ctypes.c_uint32)
IBV_SRQ_INIT_ATTR_TYPE = enum_ibv_srq_init_attr_mask.define('IBV_SRQ_INIT_ATTR_TYPE', 1)
IBV_SRQ_INIT_ATTR_PD = enum_ibv_srq_init_attr_mask.define('IBV_SRQ_INIT_ATTR_PD', 2)
IBV_SRQ_INIT_ATTR_XRCD = enum_ibv_srq_init_attr_mask.define('IBV_SRQ_INIT_ATTR_XRCD', 4)
IBV_SRQ_INIT_ATTR_CQ = enum_ibv_srq_init_attr_mask.define('IBV_SRQ_INIT_ATTR_CQ', 8)
IBV_SRQ_INIT_ATTR_TM = enum_ibv_srq_init_attr_mask.define('IBV_SRQ_INIT_ATTR_TM', 16)
IBV_SRQ_INIT_ATTR_RESERVED = enum_ibv_srq_init_attr_mask.define('IBV_SRQ_INIT_ATTR_RESERVED', 32)

class struct_ibv_tm_cap(Struct): pass
struct_ibv_tm_cap._fields_ = [
  ('max_num_tags', uint32_t),
  ('max_ops', uint32_t),
]
class struct_ibv_srq_init_attr_ex(Struct): pass
struct_ibv_srq_init_attr_ex._fields_ = [
  ('srq_context', ctypes.c_void_p),
  ('attr', struct_ibv_srq_attr),
  ('comp_mask', uint32_t),
  ('srq_type', enum_ibv_srq_type),
  ('pd', ctypes.POINTER(struct_ibv_pd)),
  ('xrcd', ctypes.POINTER(struct_ibv_xrcd)),
  ('cq', ctypes.POINTER(struct_ibv_cq)),
  ('tm_cap', struct_ibv_tm_cap),
]
enum_ibv_wq_init_attr_mask = CEnum(ctypes.c_uint32)
IBV_WQ_INIT_ATTR_FLAGS = enum_ibv_wq_init_attr_mask.define('IBV_WQ_INIT_ATTR_FLAGS', 1)
IBV_WQ_INIT_ATTR_RESERVED = enum_ibv_wq_init_attr_mask.define('IBV_WQ_INIT_ATTR_RESERVED', 2)

enum_ibv_wq_flags = CEnum(ctypes.c_uint32)
IBV_WQ_FLAGS_CVLAN_STRIPPING = enum_ibv_wq_flags.define('IBV_WQ_FLAGS_CVLAN_STRIPPING', 1)
IBV_WQ_FLAGS_SCATTER_FCS = enum_ibv_wq_flags.define('IBV_WQ_FLAGS_SCATTER_FCS', 2)
IBV_WQ_FLAGS_DELAY_DROP = enum_ibv_wq_flags.define('IBV_WQ_FLAGS_DELAY_DROP', 4)
IBV_WQ_FLAGS_PCI_WRITE_END_PADDING = enum_ibv_wq_flags.define('IBV_WQ_FLAGS_PCI_WRITE_END_PADDING', 8)
IBV_WQ_FLAGS_RESERVED = enum_ibv_wq_flags.define('IBV_WQ_FLAGS_RESERVED', 16)

class struct_ibv_wq_init_attr(Struct): pass
struct_ibv_wq_init_attr._fields_ = [
  ('wq_context', ctypes.c_void_p),
  ('wq_type', enum_ibv_wq_type),
  ('max_wr', uint32_t),
  ('max_sge', uint32_t),
  ('pd', ctypes.POINTER(struct_ibv_pd)),
  ('cq', ctypes.POINTER(struct_ibv_cq)),
  ('comp_mask', uint32_t),
  ('create_flags', uint32_t),
]
enum_ibv_wq_attr_mask = CEnum(ctypes.c_uint32)
IBV_WQ_ATTR_STATE = enum_ibv_wq_attr_mask.define('IBV_WQ_ATTR_STATE', 1)
IBV_WQ_ATTR_CURR_STATE = enum_ibv_wq_attr_mask.define('IBV_WQ_ATTR_CURR_STATE', 2)
IBV_WQ_ATTR_FLAGS = enum_ibv_wq_attr_mask.define('IBV_WQ_ATTR_FLAGS', 4)
IBV_WQ_ATTR_RESERVED = enum_ibv_wq_attr_mask.define('IBV_WQ_ATTR_RESERVED', 8)

class struct_ibv_wq_attr(Struct): pass
struct_ibv_wq_attr._fields_ = [
  ('attr_mask', uint32_t),
  ('wq_state', enum_ibv_wq_state),
  ('curr_wq_state', enum_ibv_wq_state),
  ('flags', uint32_t),
  ('flags_mask', uint32_t),
]
class struct_ibv_rwq_ind_table(Struct): pass
struct_ibv_rwq_ind_table._fields_ = [
  ('context', ctypes.POINTER(struct_ibv_context)),
  ('ind_tbl_handle', ctypes.c_int32),
  ('ind_tbl_num', ctypes.c_int32),
  ('comp_mask', uint32_t),
]
enum_ibv_ind_table_init_attr_mask = CEnum(ctypes.c_uint32)
IBV_CREATE_IND_TABLE_RESERVED = enum_ibv_ind_table_init_attr_mask.define('IBV_CREATE_IND_TABLE_RESERVED', 1)

class struct_ibv_rwq_ind_table_init_attr(Struct): pass
struct_ibv_rwq_ind_table_init_attr._fields_ = [
  ('log_ind_tbl_size', uint32_t),
  ('ind_tbl', ctypes.POINTER(ctypes.POINTER(struct_ibv_wq))),
  ('comp_mask', uint32_t),
]
class struct_ibv_qp_cap(Struct): pass
struct_ibv_qp_cap._fields_ = [
  ('max_send_wr', uint32_t),
  ('max_recv_wr', uint32_t),
  ('max_send_sge', uint32_t),
  ('max_recv_sge', uint32_t),
  ('max_inline_data', uint32_t),
]
class struct_ibv_qp_init_attr(Struct): pass
struct_ibv_qp_init_attr._fields_ = [
  ('qp_context', ctypes.c_void_p),
  ('send_cq', ctypes.POINTER(struct_ibv_cq)),
  ('recv_cq', ctypes.POINTER(struct_ibv_cq)),
  ('srq', ctypes.POINTER(struct_ibv_srq)),
  ('cap', struct_ibv_qp_cap),
  ('qp_type', enum_ibv_qp_type),
  ('sq_sig_all', ctypes.c_int32),
]
enum_ibv_qp_init_attr_mask = CEnum(ctypes.c_uint32)
IBV_QP_INIT_ATTR_PD = enum_ibv_qp_init_attr_mask.define('IBV_QP_INIT_ATTR_PD', 1)
IBV_QP_INIT_ATTR_XRCD = enum_ibv_qp_init_attr_mask.define('IBV_QP_INIT_ATTR_XRCD', 2)
IBV_QP_INIT_ATTR_CREATE_FLAGS = enum_ibv_qp_init_attr_mask.define('IBV_QP_INIT_ATTR_CREATE_FLAGS', 4)
IBV_QP_INIT_ATTR_MAX_TSO_HEADER = enum_ibv_qp_init_attr_mask.define('IBV_QP_INIT_ATTR_MAX_TSO_HEADER', 8)
IBV_QP_INIT_ATTR_IND_TABLE = enum_ibv_qp_init_attr_mask.define('IBV_QP_INIT_ATTR_IND_TABLE', 16)
IBV_QP_INIT_ATTR_RX_HASH = enum_ibv_qp_init_attr_mask.define('IBV_QP_INIT_ATTR_RX_HASH', 32)
IBV_QP_INIT_ATTR_SEND_OPS_FLAGS = enum_ibv_qp_init_attr_mask.define('IBV_QP_INIT_ATTR_SEND_OPS_FLAGS', 64)

enum_ibv_qp_create_flags = CEnum(ctypes.c_uint32)
IBV_QP_CREATE_BLOCK_SELF_MCAST_LB = enum_ibv_qp_create_flags.define('IBV_QP_CREATE_BLOCK_SELF_MCAST_LB', 2)
IBV_QP_CREATE_SCATTER_FCS = enum_ibv_qp_create_flags.define('IBV_QP_CREATE_SCATTER_FCS', 256)
IBV_QP_CREATE_CVLAN_STRIPPING = enum_ibv_qp_create_flags.define('IBV_QP_CREATE_CVLAN_STRIPPING', 512)
IBV_QP_CREATE_SOURCE_QPN = enum_ibv_qp_create_flags.define('IBV_QP_CREATE_SOURCE_QPN', 1024)
IBV_QP_CREATE_PCI_WRITE_END_PADDING = enum_ibv_qp_create_flags.define('IBV_QP_CREATE_PCI_WRITE_END_PADDING', 2048)

enum_ibv_qp_create_send_ops_flags = CEnum(ctypes.c_uint32)
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

class struct_ibv_rx_hash_conf(Struct): pass
struct_ibv_rx_hash_conf._fields_ = [
  ('rx_hash_function', uint8_t),
  ('rx_hash_key_len', uint8_t),
  ('rx_hash_key', ctypes.POINTER(uint8_t)),
  ('rx_hash_fields_mask', uint64_t),
]
class struct_ibv_qp_init_attr_ex(Struct): pass
struct_ibv_qp_init_attr_ex._fields_ = [
  ('qp_context', ctypes.c_void_p),
  ('send_cq', ctypes.POINTER(struct_ibv_cq)),
  ('recv_cq', ctypes.POINTER(struct_ibv_cq)),
  ('srq', ctypes.POINTER(struct_ibv_srq)),
  ('cap', struct_ibv_qp_cap),
  ('qp_type', enum_ibv_qp_type),
  ('sq_sig_all', ctypes.c_int32),
  ('comp_mask', uint32_t),
  ('pd', ctypes.POINTER(struct_ibv_pd)),
  ('xrcd', ctypes.POINTER(struct_ibv_xrcd)),
  ('create_flags', uint32_t),
  ('max_tso_header', uint16_t),
  ('rwq_ind_tbl', ctypes.POINTER(struct_ibv_rwq_ind_table)),
  ('rx_hash_conf', struct_ibv_rx_hash_conf),
  ('source_qpn', uint32_t),
  ('send_ops_flags', uint64_t),
]
enum_ibv_qp_open_attr_mask = CEnum(ctypes.c_uint32)
IBV_QP_OPEN_ATTR_NUM = enum_ibv_qp_open_attr_mask.define('IBV_QP_OPEN_ATTR_NUM', 1)
IBV_QP_OPEN_ATTR_XRCD = enum_ibv_qp_open_attr_mask.define('IBV_QP_OPEN_ATTR_XRCD', 2)
IBV_QP_OPEN_ATTR_CONTEXT = enum_ibv_qp_open_attr_mask.define('IBV_QP_OPEN_ATTR_CONTEXT', 4)
IBV_QP_OPEN_ATTR_TYPE = enum_ibv_qp_open_attr_mask.define('IBV_QP_OPEN_ATTR_TYPE', 8)
IBV_QP_OPEN_ATTR_RESERVED = enum_ibv_qp_open_attr_mask.define('IBV_QP_OPEN_ATTR_RESERVED', 16)

class struct_ibv_qp_open_attr(Struct): pass
struct_ibv_qp_open_attr._fields_ = [
  ('comp_mask', uint32_t),
  ('qp_num', uint32_t),
  ('xrcd', ctypes.POINTER(struct_ibv_xrcd)),
  ('qp_context', ctypes.c_void_p),
  ('qp_type', enum_ibv_qp_type),
]
enum_ibv_qp_attr_mask = CEnum(ctypes.c_uint32)
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

enum_ibv_query_qp_data_in_order_flags = CEnum(ctypes.c_uint32)
IBV_QUERY_QP_DATA_IN_ORDER_RETURN_CAPS = enum_ibv_query_qp_data_in_order_flags.define('IBV_QUERY_QP_DATA_IN_ORDER_RETURN_CAPS', 1)

enum_ibv_query_qp_data_in_order_caps = CEnum(ctypes.c_uint32)
IBV_QUERY_QP_DATA_IN_ORDER_WHOLE_MSG = enum_ibv_query_qp_data_in_order_caps.define('IBV_QUERY_QP_DATA_IN_ORDER_WHOLE_MSG', 1)
IBV_QUERY_QP_DATA_IN_ORDER_ALIGNED_128_BYTES = enum_ibv_query_qp_data_in_order_caps.define('IBV_QUERY_QP_DATA_IN_ORDER_ALIGNED_128_BYTES', 2)

enum_ibv_mig_state = CEnum(ctypes.c_uint32)
IBV_MIG_MIGRATED = enum_ibv_mig_state.define('IBV_MIG_MIGRATED', 0)
IBV_MIG_REARM = enum_ibv_mig_state.define('IBV_MIG_REARM', 1)
IBV_MIG_ARMED = enum_ibv_mig_state.define('IBV_MIG_ARMED', 2)

class struct_ibv_qp_attr(Struct): pass
struct_ibv_qp_attr._fields_ = [
  ('qp_state', enum_ibv_qp_state),
  ('cur_qp_state', enum_ibv_qp_state),
  ('path_mtu', enum_ibv_mtu),
  ('path_mig_state', enum_ibv_mig_state),
  ('qkey', uint32_t),
  ('rq_psn', uint32_t),
  ('sq_psn', uint32_t),
  ('dest_qp_num', uint32_t),
  ('qp_access_flags', ctypes.c_uint32),
  ('cap', struct_ibv_qp_cap),
  ('ah_attr', struct_ibv_ah_attr),
  ('alt_ah_attr', struct_ibv_ah_attr),
  ('pkey_index', uint16_t),
  ('alt_pkey_index', uint16_t),
  ('en_sqd_async_notify', uint8_t),
  ('sq_draining', uint8_t),
  ('max_rd_atomic', uint8_t),
  ('max_dest_rd_atomic', uint8_t),
  ('min_rnr_timer', uint8_t),
  ('port_num', uint8_t),
  ('timeout', uint8_t),
  ('retry_cnt', uint8_t),
  ('rnr_retry', uint8_t),
  ('alt_port_num', uint8_t),
  ('alt_timeout', uint8_t),
  ('rate_limit', uint32_t),
]
class struct_ibv_qp_rate_limit_attr(Struct): pass
struct_ibv_qp_rate_limit_attr._fields_ = [
  ('rate_limit', uint32_t),
  ('max_burst_sz', uint32_t),
  ('typical_pkt_sz', uint16_t),
  ('comp_mask', uint32_t),
]
try: (ibv_wr_opcode_str:=dll.ibv_wr_opcode_str).restype, ibv_wr_opcode_str.argtypes = ctypes.POINTER(ctypes.c_char), [enum_ibv_wr_opcode]
except AttributeError: pass

enum_ibv_send_flags = CEnum(ctypes.c_uint32)
IBV_SEND_FENCE = enum_ibv_send_flags.define('IBV_SEND_FENCE', 1)
IBV_SEND_SIGNALED = enum_ibv_send_flags.define('IBV_SEND_SIGNALED', 2)
IBV_SEND_SOLICITED = enum_ibv_send_flags.define('IBV_SEND_SOLICITED', 4)
IBV_SEND_INLINE = enum_ibv_send_flags.define('IBV_SEND_INLINE', 8)
IBV_SEND_IP_CSUM = enum_ibv_send_flags.define('IBV_SEND_IP_CSUM', 16)

enum_ibv_placement_type = CEnum(ctypes.c_uint32)
IBV_FLUSH_GLOBAL = enum_ibv_placement_type.define('IBV_FLUSH_GLOBAL', 1)
IBV_FLUSH_PERSISTENT = enum_ibv_placement_type.define('IBV_FLUSH_PERSISTENT', 2)

enum_ibv_selectivity_level = CEnum(ctypes.c_uint32)
IBV_FLUSH_RANGE = enum_ibv_selectivity_level.define('IBV_FLUSH_RANGE', 0)
IBV_FLUSH_MR = enum_ibv_selectivity_level.define('IBV_FLUSH_MR', 1)

class struct_ibv_data_buf(Struct): pass
struct_ibv_data_buf._fields_ = [
  ('addr', ctypes.c_void_p),
  ('length', size_t),
]
enum_ibv_ops_wr_opcode = CEnum(ctypes.c_uint32)
IBV_WR_TAG_ADD = enum_ibv_ops_wr_opcode.define('IBV_WR_TAG_ADD', 0)
IBV_WR_TAG_DEL = enum_ibv_ops_wr_opcode.define('IBV_WR_TAG_DEL', 1)
IBV_WR_TAG_SYNC = enum_ibv_ops_wr_opcode.define('IBV_WR_TAG_SYNC', 2)

enum_ibv_ops_flags = CEnum(ctypes.c_uint32)
IBV_OPS_SIGNALED = enum_ibv_ops_flags.define('IBV_OPS_SIGNALED', 1)
IBV_OPS_TM_SYNC = enum_ibv_ops_flags.define('IBV_OPS_TM_SYNC', 2)

class struct_ibv_ops_wr(Struct): pass
class struct_ibv_ops_wr_tm(Struct): pass
class struct_ibv_ops_wr_tm_add(Struct): pass
struct_ibv_ops_wr_tm_add._fields_ = [
  ('recv_wr_id', uint64_t),
  ('sg_list', ctypes.POINTER(struct_ibv_sge)),
  ('num_sge', ctypes.c_int32),
  ('tag', uint64_t),
  ('mask', uint64_t),
]
struct_ibv_ops_wr_tm._fields_ = [
  ('unexpected_cnt', uint32_t),
  ('handle', uint32_t),
  ('add', struct_ibv_ops_wr_tm_add),
]
struct_ibv_ops_wr._fields_ = [
  ('wr_id', uint64_t),
  ('next', ctypes.POINTER(struct_ibv_ops_wr)),
  ('opcode', enum_ibv_ops_wr_opcode),
  ('flags', ctypes.c_int32),
  ('tm', struct_ibv_ops_wr_tm),
]
class struct_ibv_qp_ex(Struct): pass
struct_ibv_qp_ex._fields_ = [
  ('qp_base', struct_ibv_qp),
  ('comp_mask', uint64_t),
  ('wr_id', uint64_t),
  ('wr_flags', ctypes.c_uint32),
  ('wr_atomic_cmp_swp', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ibv_qp_ex), uint32_t, uint64_t, uint64_t, uint64_t)),
  ('wr_atomic_fetch_add', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ibv_qp_ex), uint32_t, uint64_t, uint64_t)),
  ('wr_bind_mw', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ibv_qp_ex), ctypes.POINTER(struct_ibv_mw), uint32_t, ctypes.POINTER(struct_ibv_mw_bind_info))),
  ('wr_local_inv', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ibv_qp_ex), uint32_t)),
  ('wr_rdma_read', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ibv_qp_ex), uint32_t, uint64_t)),
  ('wr_rdma_write', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ibv_qp_ex), uint32_t, uint64_t)),
  ('wr_rdma_write_imm', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ibv_qp_ex), uint32_t, uint64_t, ctypes.c_uint32)),
  ('wr_send', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ibv_qp_ex))),
  ('wr_send_imm', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ibv_qp_ex), ctypes.c_uint32)),
  ('wr_send_inv', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ibv_qp_ex), uint32_t)),
  ('wr_send_tso', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ibv_qp_ex), ctypes.c_void_p, uint16_t, uint16_t)),
  ('wr_set_ud_addr', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ibv_qp_ex), ctypes.POINTER(struct_ibv_ah), uint32_t, uint32_t)),
  ('wr_set_xrc_srqn', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ibv_qp_ex), uint32_t)),
  ('wr_set_inline_data', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ibv_qp_ex), ctypes.c_void_p, size_t)),
  ('wr_set_inline_data_list', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ibv_qp_ex), size_t, ctypes.POINTER(struct_ibv_data_buf))),
  ('wr_set_sge', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ibv_qp_ex), uint32_t, uint64_t, uint32_t)),
  ('wr_set_sge_list', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ibv_qp_ex), size_t, ctypes.POINTER(struct_ibv_sge))),
  ('wr_start', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ibv_qp_ex))),
  ('wr_complete', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_ibv_qp_ex))),
  ('wr_abort', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ibv_qp_ex))),
  ('wr_atomic_write', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ibv_qp_ex), uint32_t, uint64_t, ctypes.c_void_p)),
  ('wr_flush', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ibv_qp_ex), uint32_t, uint64_t, size_t, uint8_t, uint8_t)),
]
try: (ibv_qp_to_qp_ex:=dll.ibv_qp_to_qp_ex).restype, ibv_qp_to_qp_ex.argtypes = ctypes.POINTER(struct_ibv_qp_ex), [ctypes.POINTER(struct_ibv_qp)]
except AttributeError: pass

class struct_ibv_ece(Struct): pass
struct_ibv_ece._fields_ = [
  ('vendor_id', uint32_t),
  ('options', uint32_t),
  ('comp_mask', uint32_t),
]
class struct_ibv_poll_cq_attr(Struct): pass
struct_ibv_poll_cq_attr._fields_ = [
  ('comp_mask', uint32_t),
]
class struct_ibv_wc_tm_info(Struct): pass
struct_ibv_wc_tm_info._fields_ = [
  ('tag', uint64_t),
  ('priv', uint32_t),
]
class struct_ibv_cq_ex(Struct): pass
struct_ibv_cq_ex._fields_ = [
  ('context', ctypes.POINTER(struct_ibv_context)),
  ('channel', ctypes.POINTER(struct_ibv_comp_channel)),
  ('cq_context', ctypes.c_void_p),
  ('handle', uint32_t),
  ('cqe', ctypes.c_int32),
  ('mutex', pthread_mutex_t),
  ('cond', pthread_cond_t),
  ('comp_events_completed', uint32_t),
  ('async_events_completed', uint32_t),
  ('comp_mask', uint32_t),
  ('status', enum_ibv_wc_status),
  ('wr_id', uint64_t),
  ('start_poll', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_ibv_cq_ex), ctypes.POINTER(struct_ibv_poll_cq_attr))),
  ('next_poll', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_ibv_cq_ex))),
  ('end_poll', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ibv_cq_ex))),
  ('read_opcode', ctypes.CFUNCTYPE(enum_ibv_wc_opcode, ctypes.POINTER(struct_ibv_cq_ex))),
  ('read_vendor_err', ctypes.CFUNCTYPE(uint32_t, ctypes.POINTER(struct_ibv_cq_ex))),
  ('read_byte_len', ctypes.CFUNCTYPE(uint32_t, ctypes.POINTER(struct_ibv_cq_ex))),
  ('read_imm_data', ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.POINTER(struct_ibv_cq_ex))),
  ('read_qp_num', ctypes.CFUNCTYPE(uint32_t, ctypes.POINTER(struct_ibv_cq_ex))),
  ('read_src_qp', ctypes.CFUNCTYPE(uint32_t, ctypes.POINTER(struct_ibv_cq_ex))),
  ('read_wc_flags', ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.POINTER(struct_ibv_cq_ex))),
  ('read_slid', ctypes.CFUNCTYPE(uint32_t, ctypes.POINTER(struct_ibv_cq_ex))),
  ('read_sl', ctypes.CFUNCTYPE(uint8_t, ctypes.POINTER(struct_ibv_cq_ex))),
  ('read_dlid_path_bits', ctypes.CFUNCTYPE(uint8_t, ctypes.POINTER(struct_ibv_cq_ex))),
  ('read_completion_ts', ctypes.CFUNCTYPE(uint64_t, ctypes.POINTER(struct_ibv_cq_ex))),
  ('read_cvlan', ctypes.CFUNCTYPE(uint16_t, ctypes.POINTER(struct_ibv_cq_ex))),
  ('read_flow_tag', ctypes.CFUNCTYPE(uint32_t, ctypes.POINTER(struct_ibv_cq_ex))),
  ('read_tm_info', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ibv_cq_ex), ctypes.POINTER(struct_ibv_wc_tm_info))),
  ('read_completion_wallclock_ns', ctypes.CFUNCTYPE(uint64_t, ctypes.POINTER(struct_ibv_cq_ex))),
]
enum_ibv_cq_attr_mask = CEnum(ctypes.c_uint32)
IBV_CQ_ATTR_MODERATE = enum_ibv_cq_attr_mask.define('IBV_CQ_ATTR_MODERATE', 1)
IBV_CQ_ATTR_RESERVED = enum_ibv_cq_attr_mask.define('IBV_CQ_ATTR_RESERVED', 2)

class struct_ibv_moderate_cq(Struct): pass
struct_ibv_moderate_cq._fields_ = [
  ('cq_count', uint16_t),
  ('cq_period', uint16_t),
]
class struct_ibv_modify_cq_attr(Struct): pass
struct_ibv_modify_cq_attr._fields_ = [
  ('attr_mask', uint32_t),
  ('moderate', struct_ibv_moderate_cq),
]
enum_ibv_flow_flags = CEnum(ctypes.c_uint32)
IBV_FLOW_ATTR_FLAGS_DONT_TRAP = enum_ibv_flow_flags.define('IBV_FLOW_ATTR_FLAGS_DONT_TRAP', 2)
IBV_FLOW_ATTR_FLAGS_EGRESS = enum_ibv_flow_flags.define('IBV_FLOW_ATTR_FLAGS_EGRESS', 4)

enum_ibv_flow_attr_type = CEnum(ctypes.c_uint32)
IBV_FLOW_ATTR_NORMAL = enum_ibv_flow_attr_type.define('IBV_FLOW_ATTR_NORMAL', 0)
IBV_FLOW_ATTR_ALL_DEFAULT = enum_ibv_flow_attr_type.define('IBV_FLOW_ATTR_ALL_DEFAULT', 1)
IBV_FLOW_ATTR_MC_DEFAULT = enum_ibv_flow_attr_type.define('IBV_FLOW_ATTR_MC_DEFAULT', 2)
IBV_FLOW_ATTR_SNIFFER = enum_ibv_flow_attr_type.define('IBV_FLOW_ATTR_SNIFFER', 3)

enum_ibv_flow_spec_type = CEnum(ctypes.c_uint32)
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

class struct_ibv_flow_eth_filter(Struct): pass
struct_ibv_flow_eth_filter._fields_ = [
  ('dst_mac', (uint8_t * 6)),
  ('src_mac', (uint8_t * 6)),
  ('ether_type', uint16_t),
  ('vlan_tag', uint16_t),
]
class struct_ibv_flow_spec_eth(Struct): pass
struct_ibv_flow_spec_eth._fields_ = [
  ('type', enum_ibv_flow_spec_type),
  ('size', uint16_t),
  ('val', struct_ibv_flow_eth_filter),
  ('mask', struct_ibv_flow_eth_filter),
]
class struct_ibv_flow_ipv4_filter(Struct): pass
struct_ibv_flow_ipv4_filter._fields_ = [
  ('src_ip', uint32_t),
  ('dst_ip', uint32_t),
]
class struct_ibv_flow_spec_ipv4(Struct): pass
struct_ibv_flow_spec_ipv4._fields_ = [
  ('type', enum_ibv_flow_spec_type),
  ('size', uint16_t),
  ('val', struct_ibv_flow_ipv4_filter),
  ('mask', struct_ibv_flow_ipv4_filter),
]
class struct_ibv_flow_ipv4_ext_filter(Struct): pass
struct_ibv_flow_ipv4_ext_filter._fields_ = [
  ('src_ip', uint32_t),
  ('dst_ip', uint32_t),
  ('proto', uint8_t),
  ('tos', uint8_t),
  ('ttl', uint8_t),
  ('flags', uint8_t),
]
class struct_ibv_flow_spec_ipv4_ext(Struct): pass
struct_ibv_flow_spec_ipv4_ext._fields_ = [
  ('type', enum_ibv_flow_spec_type),
  ('size', uint16_t),
  ('val', struct_ibv_flow_ipv4_ext_filter),
  ('mask', struct_ibv_flow_ipv4_ext_filter),
]
class struct_ibv_flow_ipv6_filter(Struct): pass
struct_ibv_flow_ipv6_filter._fields_ = [
  ('src_ip', (uint8_t * 16)),
  ('dst_ip', (uint8_t * 16)),
  ('flow_label', uint32_t),
  ('next_hdr', uint8_t),
  ('traffic_class', uint8_t),
  ('hop_limit', uint8_t),
]
class struct_ibv_flow_spec_ipv6(Struct): pass
struct_ibv_flow_spec_ipv6._fields_ = [
  ('type', enum_ibv_flow_spec_type),
  ('size', uint16_t),
  ('val', struct_ibv_flow_ipv6_filter),
  ('mask', struct_ibv_flow_ipv6_filter),
]
class struct_ibv_flow_esp_filter(Struct): pass
struct_ibv_flow_esp_filter._fields_ = [
  ('spi', uint32_t),
  ('seq', uint32_t),
]
class struct_ibv_flow_spec_esp(Struct): pass
struct_ibv_flow_spec_esp._fields_ = [
  ('type', enum_ibv_flow_spec_type),
  ('size', uint16_t),
  ('val', struct_ibv_flow_esp_filter),
  ('mask', struct_ibv_flow_esp_filter),
]
class struct_ibv_flow_tcp_udp_filter(Struct): pass
struct_ibv_flow_tcp_udp_filter._fields_ = [
  ('dst_port', uint16_t),
  ('src_port', uint16_t),
]
class struct_ibv_flow_spec_tcp_udp(Struct): pass
struct_ibv_flow_spec_tcp_udp._fields_ = [
  ('type', enum_ibv_flow_spec_type),
  ('size', uint16_t),
  ('val', struct_ibv_flow_tcp_udp_filter),
  ('mask', struct_ibv_flow_tcp_udp_filter),
]
class struct_ibv_flow_gre_filter(Struct): pass
struct_ibv_flow_gre_filter._fields_ = [
  ('c_ks_res0_ver', uint16_t),
  ('protocol', uint16_t),
  ('key', uint32_t),
]
class struct_ibv_flow_spec_gre(Struct): pass
struct_ibv_flow_spec_gre._fields_ = [
  ('type', enum_ibv_flow_spec_type),
  ('size', uint16_t),
  ('val', struct_ibv_flow_gre_filter),
  ('mask', struct_ibv_flow_gre_filter),
]
class struct_ibv_flow_mpls_filter(Struct): pass
struct_ibv_flow_mpls_filter._fields_ = [
  ('label', uint32_t),
]
class struct_ibv_flow_spec_mpls(Struct): pass
struct_ibv_flow_spec_mpls._fields_ = [
  ('type', enum_ibv_flow_spec_type),
  ('size', uint16_t),
  ('val', struct_ibv_flow_mpls_filter),
  ('mask', struct_ibv_flow_mpls_filter),
]
class struct_ibv_flow_tunnel_filter(Struct): pass
struct_ibv_flow_tunnel_filter._fields_ = [
  ('tunnel_id', uint32_t),
]
class struct_ibv_flow_spec_tunnel(Struct): pass
struct_ibv_flow_spec_tunnel._fields_ = [
  ('type', enum_ibv_flow_spec_type),
  ('size', uint16_t),
  ('val', struct_ibv_flow_tunnel_filter),
  ('mask', struct_ibv_flow_tunnel_filter),
]
class struct_ibv_flow_spec_action_tag(Struct): pass
struct_ibv_flow_spec_action_tag._fields_ = [
  ('type', enum_ibv_flow_spec_type),
  ('size', uint16_t),
  ('tag_id', uint32_t),
]
class struct_ibv_flow_spec_action_drop(Struct): pass
struct_ibv_flow_spec_action_drop._fields_ = [
  ('type', enum_ibv_flow_spec_type),
  ('size', uint16_t),
]
class struct_ibv_flow_spec_action_handle(Struct): pass
class struct_ibv_flow_action(Struct): pass
struct_ibv_flow_action._fields_ = [
  ('context', ctypes.POINTER(struct_ibv_context)),
]
struct_ibv_flow_spec_action_handle._fields_ = [
  ('type', enum_ibv_flow_spec_type),
  ('size', uint16_t),
  ('action', ctypes.POINTER(struct_ibv_flow_action)),
]
class struct_ibv_flow_spec_counter_action(Struct): pass
class struct_ibv_counters(Struct): pass
struct_ibv_counters._fields_ = [
  ('context', ctypes.POINTER(struct_ibv_context)),
]
struct_ibv_flow_spec_counter_action._fields_ = [
  ('type', enum_ibv_flow_spec_type),
  ('size', uint16_t),
  ('counters', ctypes.POINTER(struct_ibv_counters)),
]
class struct_ibv_flow_spec(Struct): pass
class struct_ibv_flow_spec_0(ctypes.Union): pass
class struct_ibv_flow_spec_0_hdr(Struct): pass
struct_ibv_flow_spec_0_hdr._fields_ = [
  ('type', enum_ibv_flow_spec_type),
  ('size', uint16_t),
]
struct_ibv_flow_spec_0._fields_ = [
  ('hdr', struct_ibv_flow_spec_0_hdr),
  ('eth', struct_ibv_flow_spec_eth),
  ('ipv4', struct_ibv_flow_spec_ipv4),
  ('tcp_udp', struct_ibv_flow_spec_tcp_udp),
  ('ipv4_ext', struct_ibv_flow_spec_ipv4_ext),
  ('ipv6', struct_ibv_flow_spec_ipv6),
  ('esp', struct_ibv_flow_spec_esp),
  ('tunnel', struct_ibv_flow_spec_tunnel),
  ('gre', struct_ibv_flow_spec_gre),
  ('mpls', struct_ibv_flow_spec_mpls),
  ('flow_tag', struct_ibv_flow_spec_action_tag),
  ('drop', struct_ibv_flow_spec_action_drop),
  ('handle', struct_ibv_flow_spec_action_handle),
  ('flow_count', struct_ibv_flow_spec_counter_action),
]
struct_ibv_flow_spec._anonymous_ = ['_0']
struct_ibv_flow_spec._fields_ = [
  ('_0', struct_ibv_flow_spec_0),
]
class struct_ibv_flow_attr(Struct): pass
struct_ibv_flow_attr._fields_ = [
  ('comp_mask', uint32_t),
  ('type', enum_ibv_flow_attr_type),
  ('size', uint16_t),
  ('priority', uint16_t),
  ('num_of_specs', uint8_t),
  ('port', uint8_t),
  ('flags', uint32_t),
]
class struct_ibv_flow(Struct): pass
struct_ibv_flow._fields_ = [
  ('comp_mask', uint32_t),
  ('context', ctypes.POINTER(struct_ibv_context)),
  ('handle', uint32_t),
]
enum_ibv_flow_action_esp_mask = CEnum(ctypes.c_uint32)
IBV_FLOW_ACTION_ESP_MASK_ESN = enum_ibv_flow_action_esp_mask.define('IBV_FLOW_ACTION_ESP_MASK_ESN', 1)

class struct_ibv_flow_action_esp_attr(Struct): pass
class struct_ib_uverbs_flow_action_esp(Struct): pass
__u32 = ctypes.c_uint32
__u64 = ctypes.c_uint64
struct_ib_uverbs_flow_action_esp._fields_ = [
  ('spi', ctypes.c_uint32),
  ('seq', ctypes.c_uint32),
  ('tfc_pad', ctypes.c_uint32),
  ('flags', ctypes.c_uint32),
  ('hard_limit_pkts', ctypes.c_uint64),
]
enum_ib_uverbs_flow_action_esp_keymat = CEnum(ctypes.c_uint32)
IB_UVERBS_FLOW_ACTION_ESP_KEYMAT_AES_GCM = enum_ib_uverbs_flow_action_esp_keymat.define('IB_UVERBS_FLOW_ACTION_ESP_KEYMAT_AES_GCM', 0)

enum_ib_uverbs_flow_action_esp_replay = CEnum(ctypes.c_uint32)
IB_UVERBS_FLOW_ACTION_ESP_REPLAY_NONE = enum_ib_uverbs_flow_action_esp_replay.define('IB_UVERBS_FLOW_ACTION_ESP_REPLAY_NONE', 0)
IB_UVERBS_FLOW_ACTION_ESP_REPLAY_BMP = enum_ib_uverbs_flow_action_esp_replay.define('IB_UVERBS_FLOW_ACTION_ESP_REPLAY_BMP', 1)

class struct_ib_uverbs_flow_action_esp_encap(Struct): pass
class struct_ib_uverbs_flow_action_esp_encap_0(ctypes.Union): pass
struct_ib_uverbs_flow_action_esp_encap_0._fields_ = [
  ('val_ptr', ctypes.c_void_p),
  ('val_ptr_data_u64', ctypes.c_uint64),
]
class struct_ib_uverbs_flow_action_esp_encap_1(ctypes.Union): pass
struct_ib_uverbs_flow_action_esp_encap_1._fields_ = [
  ('next_ptr', ctypes.POINTER(struct_ib_uverbs_flow_action_esp_encap)),
  ('next_ptr_data_u64', ctypes.c_uint64),
]
__u16 = ctypes.c_uint16
struct_ib_uverbs_flow_action_esp_encap._anonymous_ = ['_0', '_1']
struct_ib_uverbs_flow_action_esp_encap._fields_ = [
  ('_0', struct_ib_uverbs_flow_action_esp_encap_0),
  ('_1', struct_ib_uverbs_flow_action_esp_encap_1),
  ('len', ctypes.c_uint16),
  ('type', ctypes.c_uint16),
]
struct_ibv_flow_action_esp_attr._fields_ = [
  ('esp_attr', ctypes.POINTER(struct_ib_uverbs_flow_action_esp)),
  ('keymat_proto', enum_ib_uverbs_flow_action_esp_keymat),
  ('keymat_len', uint16_t),
  ('keymat_ptr', ctypes.c_void_p),
  ('replay_proto', enum_ib_uverbs_flow_action_esp_replay),
  ('replay_len', uint16_t),
  ('replay_ptr', ctypes.c_void_p),
  ('esp_encap', ctypes.POINTER(struct_ib_uverbs_flow_action_esp_encap)),
  ('comp_mask', uint32_t),
  ('esn', uint32_t),
]
_anonenum4 = CEnum(ctypes.c_uint32)
IBV_SYSFS_NAME_MAX = _anonenum4.define('IBV_SYSFS_NAME_MAX', 64)
IBV_SYSFS_PATH_MAX = _anonenum4.define('IBV_SYSFS_PATH_MAX', 256)

enum_ibv_cq_init_attr_mask = CEnum(ctypes.c_uint32)
IBV_CQ_INIT_ATTR_MASK_FLAGS = enum_ibv_cq_init_attr_mask.define('IBV_CQ_INIT_ATTR_MASK_FLAGS', 1)
IBV_CQ_INIT_ATTR_MASK_PD = enum_ibv_cq_init_attr_mask.define('IBV_CQ_INIT_ATTR_MASK_PD', 2)

enum_ibv_create_cq_attr_flags = CEnum(ctypes.c_uint32)
IBV_CREATE_CQ_ATTR_SINGLE_THREADED = enum_ibv_create_cq_attr_flags.define('IBV_CREATE_CQ_ATTR_SINGLE_THREADED', 1)
IBV_CREATE_CQ_ATTR_IGNORE_OVERRUN = enum_ibv_create_cq_attr_flags.define('IBV_CREATE_CQ_ATTR_IGNORE_OVERRUN', 2)

class struct_ibv_cq_init_attr_ex(Struct): pass
struct_ibv_cq_init_attr_ex._fields_ = [
  ('cqe', uint32_t),
  ('cq_context', ctypes.c_void_p),
  ('channel', ctypes.POINTER(struct_ibv_comp_channel)),
  ('comp_vector', uint32_t),
  ('wc_flags', uint64_t),
  ('comp_mask', uint32_t),
  ('flags', uint32_t),
  ('parent_domain', ctypes.POINTER(struct_ibv_pd)),
]
enum_ibv_parent_domain_init_attr_mask = CEnum(ctypes.c_uint32)
IBV_PARENT_DOMAIN_INIT_ATTR_ALLOCATORS = enum_ibv_parent_domain_init_attr_mask.define('IBV_PARENT_DOMAIN_INIT_ATTR_ALLOCATORS', 1)
IBV_PARENT_DOMAIN_INIT_ATTR_PD_CONTEXT = enum_ibv_parent_domain_init_attr_mask.define('IBV_PARENT_DOMAIN_INIT_ATTR_PD_CONTEXT', 2)

class struct_ibv_parent_domain_init_attr(Struct): pass
struct_ibv_parent_domain_init_attr._fields_ = [
  ('pd', ctypes.POINTER(struct_ibv_pd)),
  ('td', ctypes.POINTER(struct_ibv_td)),
  ('comp_mask', uint32_t),
  ('alloc', ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.POINTER(struct_ibv_pd), ctypes.c_void_p, size_t, size_t, uint64_t)),
  ('free', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ibv_pd), ctypes.c_void_p, ctypes.c_void_p, uint64_t)),
  ('pd_context', ctypes.c_void_p),
]
class struct_ibv_counters_init_attr(Struct): pass
struct_ibv_counters_init_attr._fields_ = [
  ('comp_mask', uint32_t),
]
enum_ibv_counter_description = CEnum(ctypes.c_uint32)
IBV_COUNTER_PACKETS = enum_ibv_counter_description.define('IBV_COUNTER_PACKETS', 0)
IBV_COUNTER_BYTES = enum_ibv_counter_description.define('IBV_COUNTER_BYTES', 1)

class struct_ibv_counter_attach_attr(Struct): pass
struct_ibv_counter_attach_attr._fields_ = [
  ('counter_desc', enum_ibv_counter_description),
  ('index', uint32_t),
  ('comp_mask', uint32_t),
]
enum_ibv_read_counters_flags = CEnum(ctypes.c_uint32)
IBV_READ_COUNTERS_ATTR_PREFER_CACHED = enum_ibv_read_counters_flags.define('IBV_READ_COUNTERS_ATTR_PREFER_CACHED', 1)

enum_ibv_values_mask = CEnum(ctypes.c_uint32)
IBV_VALUES_MASK_RAW_CLOCK = enum_ibv_values_mask.define('IBV_VALUES_MASK_RAW_CLOCK', 1)
IBV_VALUES_MASK_RESERVED = enum_ibv_values_mask.define('IBV_VALUES_MASK_RESERVED', 2)

class struct_ibv_values_ex(Struct): pass
class struct_timespec(Struct): pass
__time_t = ctypes.c_int64
__syscall_slong_t = ctypes.c_int64
struct_timespec._fields_ = [
  ('tv_sec', ctypes.c_int64),
  ('tv_nsec', ctypes.c_int64),
]
struct_ibv_values_ex._fields_ = [
  ('comp_mask', uint32_t),
  ('raw_clock', struct_timespec),
]
class struct_verbs_context(Struct): pass
enum_ib_uverbs_advise_mr_advice = CEnum(ctypes.c_uint32)
IB_UVERBS_ADVISE_MR_ADVICE_PREFETCH = enum_ib_uverbs_advise_mr_advice.define('IB_UVERBS_ADVISE_MR_ADVICE_PREFETCH', 0)
IB_UVERBS_ADVISE_MR_ADVICE_PREFETCH_WRITE = enum_ib_uverbs_advise_mr_advice.define('IB_UVERBS_ADVISE_MR_ADVICE_PREFETCH_WRITE', 1)
IB_UVERBS_ADVISE_MR_ADVICE_PREFETCH_NO_FAULT = enum_ib_uverbs_advise_mr_advice.define('IB_UVERBS_ADVISE_MR_ADVICE_PREFETCH_NO_FAULT', 2)

class struct_verbs_ex_private(Struct): pass
struct_verbs_context._fields_ = [
  ('query_port', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_ibv_context), uint8_t, ctypes.POINTER(struct_ibv_port_attr), size_t)),
  ('advise_mr', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_ibv_pd), enum_ib_uverbs_advise_mr_advice, uint32_t, ctypes.POINTER(struct_ibv_sge), uint32_t)),
  ('alloc_null_mr', ctypes.CFUNCTYPE(ctypes.POINTER(struct_ibv_mr), ctypes.POINTER(struct_ibv_pd))),
  ('read_counters', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_ibv_counters), ctypes.POINTER(uint64_t), uint32_t, uint32_t)),
  ('attach_counters_point_flow', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_ibv_counters), ctypes.POINTER(struct_ibv_counter_attach_attr), ctypes.POINTER(struct_ibv_flow))),
  ('create_counters', ctypes.CFUNCTYPE(ctypes.POINTER(struct_ibv_counters), ctypes.POINTER(struct_ibv_context), ctypes.POINTER(struct_ibv_counters_init_attr))),
  ('destroy_counters', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_ibv_counters))),
  ('reg_dm_mr', ctypes.CFUNCTYPE(ctypes.POINTER(struct_ibv_mr), ctypes.POINTER(struct_ibv_pd), ctypes.POINTER(struct_ibv_dm), uint64_t, size_t, ctypes.c_uint32)),
  ('alloc_dm', ctypes.CFUNCTYPE(ctypes.POINTER(struct_ibv_dm), ctypes.POINTER(struct_ibv_context), ctypes.POINTER(struct_ibv_alloc_dm_attr))),
  ('free_dm', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_ibv_dm))),
  ('modify_flow_action_esp', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_ibv_flow_action), ctypes.POINTER(struct_ibv_flow_action_esp_attr))),
  ('destroy_flow_action', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_ibv_flow_action))),
  ('create_flow_action_esp', ctypes.CFUNCTYPE(ctypes.POINTER(struct_ibv_flow_action), ctypes.POINTER(struct_ibv_context), ctypes.POINTER(struct_ibv_flow_action_esp_attr))),
  ('modify_qp_rate_limit', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_ibv_qp), ctypes.POINTER(struct_ibv_qp_rate_limit_attr))),
  ('alloc_parent_domain', ctypes.CFUNCTYPE(ctypes.POINTER(struct_ibv_pd), ctypes.POINTER(struct_ibv_context), ctypes.POINTER(struct_ibv_parent_domain_init_attr))),
  ('dealloc_td', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_ibv_td))),
  ('alloc_td', ctypes.CFUNCTYPE(ctypes.POINTER(struct_ibv_td), ctypes.POINTER(struct_ibv_context), ctypes.POINTER(struct_ibv_td_init_attr))),
  ('modify_cq', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_ibv_cq), ctypes.POINTER(struct_ibv_modify_cq_attr))),
  ('post_srq_ops', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_ibv_srq), ctypes.POINTER(struct_ibv_ops_wr), ctypes.POINTER(ctypes.POINTER(struct_ibv_ops_wr)))),
  ('destroy_rwq_ind_table', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_ibv_rwq_ind_table))),
  ('create_rwq_ind_table', ctypes.CFUNCTYPE(ctypes.POINTER(struct_ibv_rwq_ind_table), ctypes.POINTER(struct_ibv_context), ctypes.POINTER(struct_ibv_rwq_ind_table_init_attr))),
  ('destroy_wq', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_ibv_wq))),
  ('modify_wq', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_ibv_wq), ctypes.POINTER(struct_ibv_wq_attr))),
  ('create_wq', ctypes.CFUNCTYPE(ctypes.POINTER(struct_ibv_wq), ctypes.POINTER(struct_ibv_context), ctypes.POINTER(struct_ibv_wq_init_attr))),
  ('query_rt_values', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_ibv_context), ctypes.POINTER(struct_ibv_values_ex))),
  ('create_cq_ex', ctypes.CFUNCTYPE(ctypes.POINTER(struct_ibv_cq_ex), ctypes.POINTER(struct_ibv_context), ctypes.POINTER(struct_ibv_cq_init_attr_ex))),
  ('priv', ctypes.POINTER(struct_verbs_ex_private)),
  ('query_device_ex', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_ibv_context), ctypes.POINTER(struct_ibv_query_device_ex_input), ctypes.POINTER(struct_ibv_device_attr_ex), size_t)),
  ('ibv_destroy_flow', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_ibv_flow))),
  ('ABI_placeholder2', ctypes.CFUNCTYPE(None, )),
  ('ibv_create_flow', ctypes.CFUNCTYPE(ctypes.POINTER(struct_ibv_flow), ctypes.POINTER(struct_ibv_qp), ctypes.POINTER(struct_ibv_flow_attr))),
  ('ABI_placeholder1', ctypes.CFUNCTYPE(None, )),
  ('open_qp', ctypes.CFUNCTYPE(ctypes.POINTER(struct_ibv_qp), ctypes.POINTER(struct_ibv_context), ctypes.POINTER(struct_ibv_qp_open_attr))),
  ('create_qp_ex', ctypes.CFUNCTYPE(ctypes.POINTER(struct_ibv_qp), ctypes.POINTER(struct_ibv_context), ctypes.POINTER(struct_ibv_qp_init_attr_ex))),
  ('get_srq_num', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_ibv_srq), ctypes.POINTER(uint32_t))),
  ('create_srq_ex', ctypes.CFUNCTYPE(ctypes.POINTER(struct_ibv_srq), ctypes.POINTER(struct_ibv_context), ctypes.POINTER(struct_ibv_srq_init_attr_ex))),
  ('open_xrcd', ctypes.CFUNCTYPE(ctypes.POINTER(struct_ibv_xrcd), ctypes.POINTER(struct_ibv_context), ctypes.POINTER(struct_ibv_xrcd_init_attr))),
  ('close_xrcd', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(struct_ibv_xrcd))),
  ('_ABI_placeholder3', uint64_t),
  ('sz', size_t),
  ('context', struct_ibv_context),
]
try: (ibv_get_device_list:=dll.ibv_get_device_list).restype, ibv_get_device_list.argtypes = ctypes.POINTER(ctypes.POINTER(struct_ibv_device)), [ctypes.POINTER(ctypes.c_int32)]
except AttributeError: pass

try: (ibv_free_device_list:=dll.ibv_free_device_list).restype, ibv_free_device_list.argtypes = None, [ctypes.POINTER(ctypes.POINTER(struct_ibv_device))]
except AttributeError: pass

try: (ibv_get_device_name:=dll.ibv_get_device_name).restype, ibv_get_device_name.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(struct_ibv_device)]
except AttributeError: pass

try: (ibv_get_device_index:=dll.ibv_get_device_index).restype, ibv_get_device_index.argtypes = ctypes.c_int32, [ctypes.POINTER(struct_ibv_device)]
except AttributeError: pass

try: (ibv_get_device_guid:=dll.ibv_get_device_guid).restype, ibv_get_device_guid.argtypes = ctypes.c_uint64, [ctypes.POINTER(struct_ibv_device)]
except AttributeError: pass

try: (ibv_open_device:=dll.ibv_open_device).restype, ibv_open_device.argtypes = ctypes.POINTER(struct_ibv_context), [ctypes.POINTER(struct_ibv_device)]
except AttributeError: pass

try: (ibv_close_device:=dll.ibv_close_device).restype, ibv_close_device.argtypes = ctypes.c_int32, [ctypes.POINTER(struct_ibv_context)]
except AttributeError: pass

try: (ibv_import_device:=dll.ibv_import_device).restype, ibv_import_device.argtypes = ctypes.POINTER(struct_ibv_context), [ctypes.c_int32]
except AttributeError: pass

try: (ibv_import_pd:=dll.ibv_import_pd).restype, ibv_import_pd.argtypes = ctypes.POINTER(struct_ibv_pd), [ctypes.POINTER(struct_ibv_context), uint32_t]
except AttributeError: pass

try: (ibv_unimport_pd:=dll.ibv_unimport_pd).restype, ibv_unimport_pd.argtypes = None, [ctypes.POINTER(struct_ibv_pd)]
except AttributeError: pass

try: (ibv_import_mr:=dll.ibv_import_mr).restype, ibv_import_mr.argtypes = ctypes.POINTER(struct_ibv_mr), [ctypes.POINTER(struct_ibv_pd), uint32_t]
except AttributeError: pass

try: (ibv_unimport_mr:=dll.ibv_unimport_mr).restype, ibv_unimport_mr.argtypes = None, [ctypes.POINTER(struct_ibv_mr)]
except AttributeError: pass

try: (ibv_import_dm:=dll.ibv_import_dm).restype, ibv_import_dm.argtypes = ctypes.POINTER(struct_ibv_dm), [ctypes.POINTER(struct_ibv_context), uint32_t]
except AttributeError: pass

try: (ibv_unimport_dm:=dll.ibv_unimport_dm).restype, ibv_unimport_dm.argtypes = None, [ctypes.POINTER(struct_ibv_dm)]
except AttributeError: pass

try: (ibv_get_async_event:=dll.ibv_get_async_event).restype, ibv_get_async_event.argtypes = ctypes.c_int32, [ctypes.POINTER(struct_ibv_context), ctypes.POINTER(struct_ibv_async_event)]
except AttributeError: pass

try: (ibv_ack_async_event:=dll.ibv_ack_async_event).restype, ibv_ack_async_event.argtypes = None, [ctypes.POINTER(struct_ibv_async_event)]
except AttributeError: pass

try: (ibv_query_device:=dll.ibv_query_device).restype, ibv_query_device.argtypes = ctypes.c_int32, [ctypes.POINTER(struct_ibv_context), ctypes.POINTER(struct_ibv_device_attr)]
except AttributeError: pass

try: (ibv_query_port:=dll.ibv_query_port).restype, ibv_query_port.argtypes = ctypes.c_int32, [ctypes.POINTER(struct_ibv_context), uint8_t, ctypes.POINTER(struct__compat_ibv_port_attr)]
except AttributeError: pass

try: (ibv_query_gid:=dll.ibv_query_gid).restype, ibv_query_gid.argtypes = ctypes.c_int32, [ctypes.POINTER(struct_ibv_context), uint8_t, ctypes.c_int32, ctypes.POINTER(union_ibv_gid)]
except AttributeError: pass

try: (_ibv_query_gid_ex:=dll._ibv_query_gid_ex).restype, _ibv_query_gid_ex.argtypes = ctypes.c_int32, [ctypes.POINTER(struct_ibv_context), uint32_t, uint32_t, ctypes.POINTER(struct_ibv_gid_entry), uint32_t, size_t]
except AttributeError: pass

ssize_t = ctypes.c_int64
try: (_ibv_query_gid_table:=dll._ibv_query_gid_table).restype, _ibv_query_gid_table.argtypes = ssize_t, [ctypes.POINTER(struct_ibv_context), ctypes.POINTER(struct_ibv_gid_entry), size_t, uint32_t, size_t]
except AttributeError: pass

try: (ibv_query_pkey:=dll.ibv_query_pkey).restype, ibv_query_pkey.argtypes = ctypes.c_int32, [ctypes.POINTER(struct_ibv_context), uint8_t, ctypes.c_int32, ctypes.POINTER(ctypes.c_uint16)]
except AttributeError: pass

try: (ibv_get_pkey_index:=dll.ibv_get_pkey_index).restype, ibv_get_pkey_index.argtypes = ctypes.c_int32, [ctypes.POINTER(struct_ibv_context), uint8_t, ctypes.c_uint16]
except AttributeError: pass

try: (ibv_alloc_pd:=dll.ibv_alloc_pd).restype, ibv_alloc_pd.argtypes = ctypes.POINTER(struct_ibv_pd), [ctypes.POINTER(struct_ibv_context)]
except AttributeError: pass

try: (ibv_dealloc_pd:=dll.ibv_dealloc_pd).restype, ibv_dealloc_pd.argtypes = ctypes.c_int32, [ctypes.POINTER(struct_ibv_pd)]
except AttributeError: pass

try: (ibv_reg_mr_iova2:=dll.ibv_reg_mr_iova2).restype, ibv_reg_mr_iova2.argtypes = ctypes.POINTER(struct_ibv_mr), [ctypes.POINTER(struct_ibv_pd), ctypes.c_void_p, size_t, uint64_t, ctypes.c_uint32]
except AttributeError: pass

try: (ibv_reg_mr:=dll.ibv_reg_mr).restype, ibv_reg_mr.argtypes = ctypes.POINTER(struct_ibv_mr), [ctypes.POINTER(struct_ibv_pd), ctypes.c_void_p, size_t, ctypes.c_int32]
except AttributeError: pass

try: (ibv_reg_mr_iova:=dll.ibv_reg_mr_iova).restype, ibv_reg_mr_iova.argtypes = ctypes.POINTER(struct_ibv_mr), [ctypes.POINTER(struct_ibv_pd), ctypes.c_void_p, size_t, uint64_t, ctypes.c_int32]
except AttributeError: pass

try: (ibv_reg_dmabuf_mr:=dll.ibv_reg_dmabuf_mr).restype, ibv_reg_dmabuf_mr.argtypes = ctypes.POINTER(struct_ibv_mr), [ctypes.POINTER(struct_ibv_pd), uint64_t, size_t, uint64_t, ctypes.c_int32, ctypes.c_int32]
except AttributeError: pass

enum_ibv_rereg_mr_err_code = CEnum(ctypes.c_int32)
IBV_REREG_MR_ERR_INPUT = enum_ibv_rereg_mr_err_code.define('IBV_REREG_MR_ERR_INPUT', -1)
IBV_REREG_MR_ERR_DONT_FORK_NEW = enum_ibv_rereg_mr_err_code.define('IBV_REREG_MR_ERR_DONT_FORK_NEW', -2)
IBV_REREG_MR_ERR_DO_FORK_OLD = enum_ibv_rereg_mr_err_code.define('IBV_REREG_MR_ERR_DO_FORK_OLD', -3)
IBV_REREG_MR_ERR_CMD = enum_ibv_rereg_mr_err_code.define('IBV_REREG_MR_ERR_CMD', -4)
IBV_REREG_MR_ERR_CMD_AND_DO_FORK_NEW = enum_ibv_rereg_mr_err_code.define('IBV_REREG_MR_ERR_CMD_AND_DO_FORK_NEW', -5)

try: (ibv_rereg_mr:=dll.ibv_rereg_mr).restype, ibv_rereg_mr.argtypes = ctypes.c_int32, [ctypes.POINTER(struct_ibv_mr), ctypes.c_int32, ctypes.POINTER(struct_ibv_pd), ctypes.c_void_p, size_t, ctypes.c_int32]
except AttributeError: pass

try: (ibv_dereg_mr:=dll.ibv_dereg_mr).restype, ibv_dereg_mr.argtypes = ctypes.c_int32, [ctypes.POINTER(struct_ibv_mr)]
except AttributeError: pass

try: (ibv_create_comp_channel:=dll.ibv_create_comp_channel).restype, ibv_create_comp_channel.argtypes = ctypes.POINTER(struct_ibv_comp_channel), [ctypes.POINTER(struct_ibv_context)]
except AttributeError: pass

try: (ibv_destroy_comp_channel:=dll.ibv_destroy_comp_channel).restype, ibv_destroy_comp_channel.argtypes = ctypes.c_int32, [ctypes.POINTER(struct_ibv_comp_channel)]
except AttributeError: pass

try: (ibv_create_cq:=dll.ibv_create_cq).restype, ibv_create_cq.argtypes = ctypes.POINTER(struct_ibv_cq), [ctypes.POINTER(struct_ibv_context), ctypes.c_int32, ctypes.c_void_p, ctypes.POINTER(struct_ibv_comp_channel), ctypes.c_int32]
except AttributeError: pass

try: (ibv_resize_cq:=dll.ibv_resize_cq).restype, ibv_resize_cq.argtypes = ctypes.c_int32, [ctypes.POINTER(struct_ibv_cq), ctypes.c_int32]
except AttributeError: pass

try: (ibv_destroy_cq:=dll.ibv_destroy_cq).restype, ibv_destroy_cq.argtypes = ctypes.c_int32, [ctypes.POINTER(struct_ibv_cq)]
except AttributeError: pass

try: (ibv_get_cq_event:=dll.ibv_get_cq_event).restype, ibv_get_cq_event.argtypes = ctypes.c_int32, [ctypes.POINTER(struct_ibv_comp_channel), ctypes.POINTER(ctypes.POINTER(struct_ibv_cq)), ctypes.POINTER(ctypes.c_void_p)]
except AttributeError: pass

try: (ibv_ack_cq_events:=dll.ibv_ack_cq_events).restype, ibv_ack_cq_events.argtypes = None, [ctypes.POINTER(struct_ibv_cq), ctypes.c_uint32]
except AttributeError: pass

try: (ibv_create_srq:=dll.ibv_create_srq).restype, ibv_create_srq.argtypes = ctypes.POINTER(struct_ibv_srq), [ctypes.POINTER(struct_ibv_pd), ctypes.POINTER(struct_ibv_srq_init_attr)]
except AttributeError: pass

try: (ibv_modify_srq:=dll.ibv_modify_srq).restype, ibv_modify_srq.argtypes = ctypes.c_int32, [ctypes.POINTER(struct_ibv_srq), ctypes.POINTER(struct_ibv_srq_attr), ctypes.c_int32]
except AttributeError: pass

try: (ibv_query_srq:=dll.ibv_query_srq).restype, ibv_query_srq.argtypes = ctypes.c_int32, [ctypes.POINTER(struct_ibv_srq), ctypes.POINTER(struct_ibv_srq_attr)]
except AttributeError: pass

try: (ibv_destroy_srq:=dll.ibv_destroy_srq).restype, ibv_destroy_srq.argtypes = ctypes.c_int32, [ctypes.POINTER(struct_ibv_srq)]
except AttributeError: pass

try: (ibv_create_qp:=dll.ibv_create_qp).restype, ibv_create_qp.argtypes = ctypes.POINTER(struct_ibv_qp), [ctypes.POINTER(struct_ibv_pd), ctypes.POINTER(struct_ibv_qp_init_attr)]
except AttributeError: pass

try: (ibv_modify_qp:=dll.ibv_modify_qp).restype, ibv_modify_qp.argtypes = ctypes.c_int32, [ctypes.POINTER(struct_ibv_qp), ctypes.POINTER(struct_ibv_qp_attr), ctypes.c_int32]
except AttributeError: pass

try: (ibv_query_qp_data_in_order:=dll.ibv_query_qp_data_in_order).restype, ibv_query_qp_data_in_order.argtypes = ctypes.c_int32, [ctypes.POINTER(struct_ibv_qp), enum_ibv_wr_opcode, uint32_t]
except AttributeError: pass

try: (ibv_query_qp:=dll.ibv_query_qp).restype, ibv_query_qp.argtypes = ctypes.c_int32, [ctypes.POINTER(struct_ibv_qp), ctypes.POINTER(struct_ibv_qp_attr), ctypes.c_int32, ctypes.POINTER(struct_ibv_qp_init_attr)]
except AttributeError: pass

try: (ibv_destroy_qp:=dll.ibv_destroy_qp).restype, ibv_destroy_qp.argtypes = ctypes.c_int32, [ctypes.POINTER(struct_ibv_qp)]
except AttributeError: pass

try: (ibv_create_ah:=dll.ibv_create_ah).restype, ibv_create_ah.argtypes = ctypes.POINTER(struct_ibv_ah), [ctypes.POINTER(struct_ibv_pd), ctypes.POINTER(struct_ibv_ah_attr)]
except AttributeError: pass

try: (ibv_init_ah_from_wc:=dll.ibv_init_ah_from_wc).restype, ibv_init_ah_from_wc.argtypes = ctypes.c_int32, [ctypes.POINTER(struct_ibv_context), uint8_t, ctypes.POINTER(struct_ibv_wc), ctypes.POINTER(struct_ibv_grh), ctypes.POINTER(struct_ibv_ah_attr)]
except AttributeError: pass

try: (ibv_create_ah_from_wc:=dll.ibv_create_ah_from_wc).restype, ibv_create_ah_from_wc.argtypes = ctypes.POINTER(struct_ibv_ah), [ctypes.POINTER(struct_ibv_pd), ctypes.POINTER(struct_ibv_wc), ctypes.POINTER(struct_ibv_grh), uint8_t]
except AttributeError: pass

try: (ibv_destroy_ah:=dll.ibv_destroy_ah).restype, ibv_destroy_ah.argtypes = ctypes.c_int32, [ctypes.POINTER(struct_ibv_ah)]
except AttributeError: pass

try: (ibv_attach_mcast:=dll.ibv_attach_mcast).restype, ibv_attach_mcast.argtypes = ctypes.c_int32, [ctypes.POINTER(struct_ibv_qp), ctypes.POINTER(union_ibv_gid), uint16_t]
except AttributeError: pass

try: (ibv_detach_mcast:=dll.ibv_detach_mcast).restype, ibv_detach_mcast.argtypes = ctypes.c_int32, [ctypes.POINTER(struct_ibv_qp), ctypes.POINTER(union_ibv_gid), uint16_t]
except AttributeError: pass

try: (ibv_fork_init:=dll.ibv_fork_init).restype, ibv_fork_init.argtypes = ctypes.c_int32, []
except AttributeError: pass

try: (ibv_is_fork_initialized:=dll.ibv_is_fork_initialized).restype, ibv_is_fork_initialized.argtypes = enum_ibv_fork_status, []
except AttributeError: pass

try: (ibv_node_type_str:=dll.ibv_node_type_str).restype, ibv_node_type_str.argtypes = ctypes.POINTER(ctypes.c_char), [enum_ibv_node_type]
except AttributeError: pass

try: (ibv_port_state_str:=dll.ibv_port_state_str).restype, ibv_port_state_str.argtypes = ctypes.POINTER(ctypes.c_char), [enum_ibv_port_state]
except AttributeError: pass

try: (ibv_event_type_str:=dll.ibv_event_type_str).restype, ibv_event_type_str.argtypes = ctypes.POINTER(ctypes.c_char), [enum_ibv_event_type]
except AttributeError: pass

try: (ibv_resolve_eth_l2_from_gid:=dll.ibv_resolve_eth_l2_from_gid).restype, ibv_resolve_eth_l2_from_gid.argtypes = ctypes.c_int32, [ctypes.POINTER(struct_ibv_context), ctypes.POINTER(struct_ibv_ah_attr), (uint8_t * 6), ctypes.POINTER(uint16_t)]
except AttributeError: pass

try: (ibv_set_ece:=dll.ibv_set_ece).restype, ibv_set_ece.argtypes = ctypes.c_int32, [ctypes.POINTER(struct_ibv_qp), ctypes.POINTER(struct_ibv_ece)]
except AttributeError: pass

try: (ibv_query_ece:=dll.ibv_query_ece).restype, ibv_query_ece.argtypes = ctypes.c_int32, [ctypes.POINTER(struct_ibv_qp), ctypes.POINTER(struct_ibv_ece)]
except AttributeError: pass

enum_ib_uverbs_core_support = CEnum(ctypes.c_uint32)
IB_UVERBS_CORE_SUPPORT_OPTIONAL_MR_ACCESS = enum_ib_uverbs_core_support.define('IB_UVERBS_CORE_SUPPORT_OPTIONAL_MR_ACCESS', 1)

enum_ib_uverbs_access_flags = CEnum(ctypes.c_uint32)
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

enum_ib_uverbs_srq_type = CEnum(ctypes.c_uint32)
IB_UVERBS_SRQT_BASIC = enum_ib_uverbs_srq_type.define('IB_UVERBS_SRQT_BASIC', 0)
IB_UVERBS_SRQT_XRC = enum_ib_uverbs_srq_type.define('IB_UVERBS_SRQT_XRC', 1)
IB_UVERBS_SRQT_TM = enum_ib_uverbs_srq_type.define('IB_UVERBS_SRQT_TM', 2)

enum_ib_uverbs_wq_type = CEnum(ctypes.c_uint32)
IB_UVERBS_WQT_RQ = enum_ib_uverbs_wq_type.define('IB_UVERBS_WQT_RQ', 0)

enum_ib_uverbs_wq_flags = CEnum(ctypes.c_uint32)
IB_UVERBS_WQ_FLAGS_CVLAN_STRIPPING = enum_ib_uverbs_wq_flags.define('IB_UVERBS_WQ_FLAGS_CVLAN_STRIPPING', 1)
IB_UVERBS_WQ_FLAGS_SCATTER_FCS = enum_ib_uverbs_wq_flags.define('IB_UVERBS_WQ_FLAGS_SCATTER_FCS', 2)
IB_UVERBS_WQ_FLAGS_DELAY_DROP = enum_ib_uverbs_wq_flags.define('IB_UVERBS_WQ_FLAGS_DELAY_DROP', 4)
IB_UVERBS_WQ_FLAGS_PCI_WRITE_END_PADDING = enum_ib_uverbs_wq_flags.define('IB_UVERBS_WQ_FLAGS_PCI_WRITE_END_PADDING', 8)

enum_ib_uverbs_qp_type = CEnum(ctypes.c_uint32)
IB_UVERBS_QPT_RC = enum_ib_uverbs_qp_type.define('IB_UVERBS_QPT_RC', 2)
IB_UVERBS_QPT_UC = enum_ib_uverbs_qp_type.define('IB_UVERBS_QPT_UC', 3)
IB_UVERBS_QPT_UD = enum_ib_uverbs_qp_type.define('IB_UVERBS_QPT_UD', 4)
IB_UVERBS_QPT_RAW_PACKET = enum_ib_uverbs_qp_type.define('IB_UVERBS_QPT_RAW_PACKET', 8)
IB_UVERBS_QPT_XRC_INI = enum_ib_uverbs_qp_type.define('IB_UVERBS_QPT_XRC_INI', 9)
IB_UVERBS_QPT_XRC_TGT = enum_ib_uverbs_qp_type.define('IB_UVERBS_QPT_XRC_TGT', 10)
IB_UVERBS_QPT_DRIVER = enum_ib_uverbs_qp_type.define('IB_UVERBS_QPT_DRIVER', 255)

enum_ib_uverbs_qp_create_flags = CEnum(ctypes.c_uint32)
IB_UVERBS_QP_CREATE_BLOCK_MULTICAST_LOOPBACK = enum_ib_uverbs_qp_create_flags.define('IB_UVERBS_QP_CREATE_BLOCK_MULTICAST_LOOPBACK', 2)
IB_UVERBS_QP_CREATE_SCATTER_FCS = enum_ib_uverbs_qp_create_flags.define('IB_UVERBS_QP_CREATE_SCATTER_FCS', 256)
IB_UVERBS_QP_CREATE_CVLAN_STRIPPING = enum_ib_uverbs_qp_create_flags.define('IB_UVERBS_QP_CREATE_CVLAN_STRIPPING', 512)
IB_UVERBS_QP_CREATE_PCI_WRITE_END_PADDING = enum_ib_uverbs_qp_create_flags.define('IB_UVERBS_QP_CREATE_PCI_WRITE_END_PADDING', 2048)
IB_UVERBS_QP_CREATE_SQ_SIG_ALL = enum_ib_uverbs_qp_create_flags.define('IB_UVERBS_QP_CREATE_SQ_SIG_ALL', 4096)

enum_ib_uverbs_query_port_cap_flags = CEnum(ctypes.c_uint32)
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

enum_ib_uverbs_query_port_flags = CEnum(ctypes.c_uint32)
IB_UVERBS_QPF_GRH_REQUIRED = enum_ib_uverbs_query_port_flags.define('IB_UVERBS_QPF_GRH_REQUIRED', 1)

enum_ib_uverbs_flow_action_esp_keymat_aes_gcm_iv_algo = CEnum(ctypes.c_uint32)
IB_UVERBS_FLOW_ACTION_IV_ALGO_SEQ = enum_ib_uverbs_flow_action_esp_keymat_aes_gcm_iv_algo.define('IB_UVERBS_FLOW_ACTION_IV_ALGO_SEQ', 0)

class struct_ib_uverbs_flow_action_esp_keymat_aes_gcm(Struct): pass
struct_ib_uverbs_flow_action_esp_keymat_aes_gcm._fields_ = [
  ('iv', ctypes.c_uint64),
  ('iv_algo', ctypes.c_uint32),
  ('salt', ctypes.c_uint32),
  ('icv_len', ctypes.c_uint32),
  ('key_len', ctypes.c_uint32),
  ('aes_key', (ctypes.c_uint32 * 8)),
]
class struct_ib_uverbs_flow_action_esp_replay_bmp(Struct): pass
struct_ib_uverbs_flow_action_esp_replay_bmp._fields_ = [
  ('size', ctypes.c_uint32),
]
enum_ib_uverbs_flow_action_esp_flags = CEnum(ctypes.c_uint32)
IB_UVERBS_FLOW_ACTION_ESP_FLAGS_INLINE_CRYPTO = enum_ib_uverbs_flow_action_esp_flags.define('IB_UVERBS_FLOW_ACTION_ESP_FLAGS_INLINE_CRYPTO', 0)
IB_UVERBS_FLOW_ACTION_ESP_FLAGS_FULL_OFFLOAD = enum_ib_uverbs_flow_action_esp_flags.define('IB_UVERBS_FLOW_ACTION_ESP_FLAGS_FULL_OFFLOAD', 1)
IB_UVERBS_FLOW_ACTION_ESP_FLAGS_TUNNEL = enum_ib_uverbs_flow_action_esp_flags.define('IB_UVERBS_FLOW_ACTION_ESP_FLAGS_TUNNEL', 0)
IB_UVERBS_FLOW_ACTION_ESP_FLAGS_TRANSPORT = enum_ib_uverbs_flow_action_esp_flags.define('IB_UVERBS_FLOW_ACTION_ESP_FLAGS_TRANSPORT', 2)
IB_UVERBS_FLOW_ACTION_ESP_FLAGS_DECRYPT = enum_ib_uverbs_flow_action_esp_flags.define('IB_UVERBS_FLOW_ACTION_ESP_FLAGS_DECRYPT', 0)
IB_UVERBS_FLOW_ACTION_ESP_FLAGS_ENCRYPT = enum_ib_uverbs_flow_action_esp_flags.define('IB_UVERBS_FLOW_ACTION_ESP_FLAGS_ENCRYPT', 4)
IB_UVERBS_FLOW_ACTION_ESP_FLAGS_ESN_NEW_WINDOW = enum_ib_uverbs_flow_action_esp_flags.define('IB_UVERBS_FLOW_ACTION_ESP_FLAGS_ESN_NEW_WINDOW', 8)

enum_ib_uverbs_read_counters_flags = CEnum(ctypes.c_uint32)
IB_UVERBS_READ_COUNTERS_PREFER_CACHED = enum_ib_uverbs_read_counters_flags.define('IB_UVERBS_READ_COUNTERS_PREFER_CACHED', 1)

enum_ib_uverbs_advise_mr_flag = CEnum(ctypes.c_uint32)
IB_UVERBS_ADVISE_MR_FLAG_FLUSH = enum_ib_uverbs_advise_mr_flag.define('IB_UVERBS_ADVISE_MR_FLAG_FLUSH', 1)

class struct_ib_uverbs_query_port_resp_ex(Struct): pass
class struct_ib_uverbs_query_port_resp(Struct): pass
__u8 = ctypes.c_ubyte
struct_ib_uverbs_query_port_resp._fields_ = [
  ('port_cap_flags', ctypes.c_uint32),
  ('max_msg_sz', ctypes.c_uint32),
  ('bad_pkey_cntr', ctypes.c_uint32),
  ('qkey_viol_cntr', ctypes.c_uint32),
  ('gid_tbl_len', ctypes.c_uint32),
  ('pkey_tbl_len', ctypes.c_uint16),
  ('lid', ctypes.c_uint16),
  ('sm_lid', ctypes.c_uint16),
  ('state', ctypes.c_ubyte),
  ('max_mtu', ctypes.c_ubyte),
  ('active_mtu', ctypes.c_ubyte),
  ('lmc', ctypes.c_ubyte),
  ('max_vl_num', ctypes.c_ubyte),
  ('sm_sl', ctypes.c_ubyte),
  ('subnet_timeout', ctypes.c_ubyte),
  ('init_type_reply', ctypes.c_ubyte),
  ('active_width', ctypes.c_ubyte),
  ('active_speed', ctypes.c_ubyte),
  ('phys_state', ctypes.c_ubyte),
  ('link_layer', ctypes.c_ubyte),
  ('flags', ctypes.c_ubyte),
  ('reserved', ctypes.c_ubyte),
]
struct_ib_uverbs_query_port_resp_ex._fields_ = [
  ('legacy_resp', struct_ib_uverbs_query_port_resp),
  ('port_cap_flags2', ctypes.c_uint16),
  ('reserved', (ctypes.c_ubyte * 2)),
  ('active_speed_ex', ctypes.c_uint32),
]
class struct_ib_uverbs_qp_cap(Struct): pass
struct_ib_uverbs_qp_cap._fields_ = [
  ('max_send_wr', ctypes.c_uint32),
  ('max_recv_wr', ctypes.c_uint32),
  ('max_send_sge', ctypes.c_uint32),
  ('max_recv_sge', ctypes.c_uint32),
  ('max_inline_data', ctypes.c_uint32),
]
enum_rdma_driver_id = CEnum(ctypes.c_uint32)
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

enum_ib_uverbs_gid_type = CEnum(ctypes.c_uint32)
IB_UVERBS_GID_TYPE_IB = enum_ib_uverbs_gid_type.define('IB_UVERBS_GID_TYPE_IB', 0)
IB_UVERBS_GID_TYPE_ROCE_V1 = enum_ib_uverbs_gid_type.define('IB_UVERBS_GID_TYPE_ROCE_V1', 1)
IB_UVERBS_GID_TYPE_ROCE_V2 = enum_ib_uverbs_gid_type.define('IB_UVERBS_GID_TYPE_ROCE_V2', 2)

class struct_ib_uverbs_gid_entry(Struct): pass
struct_ib_uverbs_gid_entry._fields_ = [
  ('gid', (ctypes.c_uint64 * 2)),
  ('gid_index', ctypes.c_uint32),
  ('port_num', ctypes.c_uint32),
  ('gid_type', ctypes.c_uint32),
  ('netdev_ifindex', ctypes.c_uint32),
]
enum_ib_uverbs_write_cmds = CEnum(ctypes.c_uint32)
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

_anonenum5 = CEnum(ctypes.c_uint32)
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

enum_ib_placement_type = CEnum(ctypes.c_uint32)
IB_FLUSH_GLOBAL = enum_ib_placement_type.define('IB_FLUSH_GLOBAL', 1)
IB_FLUSH_PERSISTENT = enum_ib_placement_type.define('IB_FLUSH_PERSISTENT', 2)

enum_ib_selectivity_level = CEnum(ctypes.c_uint32)
IB_FLUSH_RANGE = enum_ib_selectivity_level.define('IB_FLUSH_RANGE', 0)
IB_FLUSH_MR = enum_ib_selectivity_level.define('IB_FLUSH_MR', 1)

class struct_ib_uverbs_async_event_desc(Struct): pass
struct_ib_uverbs_async_event_desc._fields_ = [
  ('element', ctypes.c_uint64),
  ('event_type', ctypes.c_uint32),
  ('reserved', ctypes.c_uint32),
]
class struct_ib_uverbs_comp_event_desc(Struct): pass
struct_ib_uverbs_comp_event_desc._fields_ = [
  ('cq_handle', ctypes.c_uint64),
]
class struct_ib_uverbs_cq_moderation_caps(Struct): pass
struct_ib_uverbs_cq_moderation_caps._fields_ = [
  ('max_cq_moderation_count', ctypes.c_uint16),
  ('max_cq_moderation_period', ctypes.c_uint16),
  ('reserved', ctypes.c_uint32),
]
class struct_ib_uverbs_cmd_hdr(Struct): pass
struct_ib_uverbs_cmd_hdr._fields_ = [
  ('command', ctypes.c_uint32),
  ('in_words', ctypes.c_uint16),
  ('out_words', ctypes.c_uint16),
]
class struct_ib_uverbs_ex_cmd_hdr(Struct): pass
struct_ib_uverbs_ex_cmd_hdr._fields_ = [
  ('response', ctypes.c_uint64),
  ('provider_in_words', ctypes.c_uint16),
  ('provider_out_words', ctypes.c_uint16),
  ('cmd_hdr_reserved', ctypes.c_uint32),
]
class struct_ib_uverbs_get_context(Struct): pass
struct_ib_uverbs_get_context._fields_ = [
  ('response', ctypes.c_uint64),
  ('driver_data', (ctypes.c_uint64 * 0)),
]
class struct_ib_uverbs_get_context_resp(Struct): pass
struct_ib_uverbs_get_context_resp._fields_ = [
  ('async_fd', ctypes.c_uint32),
  ('num_comp_vectors', ctypes.c_uint32),
  ('driver_data', (ctypes.c_uint64 * 0)),
]
class struct_ib_uverbs_query_device(Struct): pass
struct_ib_uverbs_query_device._fields_ = [
  ('response', ctypes.c_uint64),
  ('driver_data', (ctypes.c_uint64 * 0)),
]
class struct_ib_uverbs_query_device_resp(Struct): pass
struct_ib_uverbs_query_device_resp._fields_ = [
  ('fw_ver', ctypes.c_uint64),
  ('node_guid', ctypes.c_uint64),
  ('sys_image_guid', ctypes.c_uint64),
  ('max_mr_size', ctypes.c_uint64),
  ('page_size_cap', ctypes.c_uint64),
  ('vendor_id', ctypes.c_uint32),
  ('vendor_part_id', ctypes.c_uint32),
  ('hw_ver', ctypes.c_uint32),
  ('max_qp', ctypes.c_uint32),
  ('max_qp_wr', ctypes.c_uint32),
  ('device_cap_flags', ctypes.c_uint32),
  ('max_sge', ctypes.c_uint32),
  ('max_sge_rd', ctypes.c_uint32),
  ('max_cq', ctypes.c_uint32),
  ('max_cqe', ctypes.c_uint32),
  ('max_mr', ctypes.c_uint32),
  ('max_pd', ctypes.c_uint32),
  ('max_qp_rd_atom', ctypes.c_uint32),
  ('max_ee_rd_atom', ctypes.c_uint32),
  ('max_res_rd_atom', ctypes.c_uint32),
  ('max_qp_init_rd_atom', ctypes.c_uint32),
  ('max_ee_init_rd_atom', ctypes.c_uint32),
  ('atomic_cap', ctypes.c_uint32),
  ('max_ee', ctypes.c_uint32),
  ('max_rdd', ctypes.c_uint32),
  ('max_mw', ctypes.c_uint32),
  ('max_raw_ipv6_qp', ctypes.c_uint32),
  ('max_raw_ethy_qp', ctypes.c_uint32),
  ('max_mcast_grp', ctypes.c_uint32),
  ('max_mcast_qp_attach', ctypes.c_uint32),
  ('max_total_mcast_qp_attach', ctypes.c_uint32),
  ('max_ah', ctypes.c_uint32),
  ('max_fmr', ctypes.c_uint32),
  ('max_map_per_fmr', ctypes.c_uint32),
  ('max_srq', ctypes.c_uint32),
  ('max_srq_wr', ctypes.c_uint32),
  ('max_srq_sge', ctypes.c_uint32),
  ('max_pkeys', ctypes.c_uint16),
  ('local_ca_ack_delay', ctypes.c_ubyte),
  ('phys_port_cnt', ctypes.c_ubyte),
  ('reserved', (ctypes.c_ubyte * 4)),
]
class struct_ib_uverbs_ex_query_device(Struct): pass
struct_ib_uverbs_ex_query_device._fields_ = [
  ('comp_mask', ctypes.c_uint32),
  ('reserved', ctypes.c_uint32),
]
class struct_ib_uverbs_odp_caps(Struct): pass
class struct_ib_uverbs_odp_caps_per_transport_caps(Struct): pass
struct_ib_uverbs_odp_caps_per_transport_caps._fields_ = [
  ('rc_odp_caps', ctypes.c_uint32),
  ('uc_odp_caps', ctypes.c_uint32),
  ('ud_odp_caps', ctypes.c_uint32),
]
struct_ib_uverbs_odp_caps._fields_ = [
  ('general_caps', ctypes.c_uint64),
  ('per_transport_caps', struct_ib_uverbs_odp_caps_per_transport_caps),
  ('reserved', ctypes.c_uint32),
]
class struct_ib_uverbs_rss_caps(Struct): pass
struct_ib_uverbs_rss_caps._fields_ = [
  ('supported_qpts', ctypes.c_uint32),
  ('max_rwq_indirection_tables', ctypes.c_uint32),
  ('max_rwq_indirection_table_size', ctypes.c_uint32),
  ('reserved', ctypes.c_uint32),
]
class struct_ib_uverbs_tm_caps(Struct): pass
struct_ib_uverbs_tm_caps._fields_ = [
  ('max_rndv_hdr_size', ctypes.c_uint32),
  ('max_num_tags', ctypes.c_uint32),
  ('flags', ctypes.c_uint32),
  ('max_ops', ctypes.c_uint32),
  ('max_sge', ctypes.c_uint32),
  ('reserved', ctypes.c_uint32),
]
class struct_ib_uverbs_ex_query_device_resp(Struct): pass
struct_ib_uverbs_ex_query_device_resp._fields_ = [
  ('base', struct_ib_uverbs_query_device_resp),
  ('comp_mask', ctypes.c_uint32),
  ('response_length', ctypes.c_uint32),
  ('odp_caps', struct_ib_uverbs_odp_caps),
  ('timestamp_mask', ctypes.c_uint64),
  ('hca_core_clock', ctypes.c_uint64),
  ('device_cap_flags_ex', ctypes.c_uint64),
  ('rss_caps', struct_ib_uverbs_rss_caps),
  ('max_wq_type_rq', ctypes.c_uint32),
  ('raw_packet_caps', ctypes.c_uint32),
  ('tm_caps', struct_ib_uverbs_tm_caps),
  ('cq_moderation_caps', struct_ib_uverbs_cq_moderation_caps),
  ('max_dm_size', ctypes.c_uint64),
  ('xrc_odp_caps', ctypes.c_uint32),
  ('reserved', ctypes.c_uint32),
]
class struct_ib_uverbs_query_port(Struct): pass
struct_ib_uverbs_query_port._fields_ = [
  ('response', ctypes.c_uint64),
  ('port_num', ctypes.c_ubyte),
  ('reserved', (ctypes.c_ubyte * 7)),
  ('driver_data', (ctypes.c_uint64 * 0)),
]
class struct_ib_uverbs_alloc_pd(Struct): pass
struct_ib_uverbs_alloc_pd._fields_ = [
  ('response', ctypes.c_uint64),
  ('driver_data', (ctypes.c_uint64 * 0)),
]
class struct_ib_uverbs_alloc_pd_resp(Struct): pass
struct_ib_uverbs_alloc_pd_resp._fields_ = [
  ('pd_handle', ctypes.c_uint32),
  ('driver_data', (ctypes.c_uint32 * 0)),
]
class struct_ib_uverbs_dealloc_pd(Struct): pass
struct_ib_uverbs_dealloc_pd._fields_ = [
  ('pd_handle', ctypes.c_uint32),
]
class struct_ib_uverbs_open_xrcd(Struct): pass
struct_ib_uverbs_open_xrcd._fields_ = [
  ('response', ctypes.c_uint64),
  ('fd', ctypes.c_uint32),
  ('oflags', ctypes.c_uint32),
  ('driver_data', (ctypes.c_uint64 * 0)),
]
class struct_ib_uverbs_open_xrcd_resp(Struct): pass
struct_ib_uverbs_open_xrcd_resp._fields_ = [
  ('xrcd_handle', ctypes.c_uint32),
  ('driver_data', (ctypes.c_uint32 * 0)),
]
class struct_ib_uverbs_close_xrcd(Struct): pass
struct_ib_uverbs_close_xrcd._fields_ = [
  ('xrcd_handle', ctypes.c_uint32),
]
class struct_ib_uverbs_reg_mr(Struct): pass
struct_ib_uverbs_reg_mr._fields_ = [
  ('response', ctypes.c_uint64),
  ('start', ctypes.c_uint64),
  ('length', ctypes.c_uint64),
  ('hca_va', ctypes.c_uint64),
  ('pd_handle', ctypes.c_uint32),
  ('access_flags', ctypes.c_uint32),
  ('driver_data', (ctypes.c_uint64 * 0)),
]
class struct_ib_uverbs_reg_mr_resp(Struct): pass
struct_ib_uverbs_reg_mr_resp._fields_ = [
  ('mr_handle', ctypes.c_uint32),
  ('lkey', ctypes.c_uint32),
  ('rkey', ctypes.c_uint32),
  ('driver_data', (ctypes.c_uint32 * 0)),
]
class struct_ib_uverbs_rereg_mr(Struct): pass
struct_ib_uverbs_rereg_mr._fields_ = [
  ('response', ctypes.c_uint64),
  ('mr_handle', ctypes.c_uint32),
  ('flags', ctypes.c_uint32),
  ('start', ctypes.c_uint64),
  ('length', ctypes.c_uint64),
  ('hca_va', ctypes.c_uint64),
  ('pd_handle', ctypes.c_uint32),
  ('access_flags', ctypes.c_uint32),
  ('driver_data', (ctypes.c_uint64 * 0)),
]
class struct_ib_uverbs_rereg_mr_resp(Struct): pass
struct_ib_uverbs_rereg_mr_resp._fields_ = [
  ('lkey', ctypes.c_uint32),
  ('rkey', ctypes.c_uint32),
  ('driver_data', (ctypes.c_uint64 * 0)),
]
class struct_ib_uverbs_dereg_mr(Struct): pass
struct_ib_uverbs_dereg_mr._fields_ = [
  ('mr_handle', ctypes.c_uint32),
]
class struct_ib_uverbs_alloc_mw(Struct): pass
struct_ib_uverbs_alloc_mw._fields_ = [
  ('response', ctypes.c_uint64),
  ('pd_handle', ctypes.c_uint32),
  ('mw_type', ctypes.c_ubyte),
  ('reserved', (ctypes.c_ubyte * 3)),
  ('driver_data', (ctypes.c_uint64 * 0)),
]
class struct_ib_uverbs_alloc_mw_resp(Struct): pass
struct_ib_uverbs_alloc_mw_resp._fields_ = [
  ('mw_handle', ctypes.c_uint32),
  ('rkey', ctypes.c_uint32),
  ('driver_data', (ctypes.c_uint64 * 0)),
]
class struct_ib_uverbs_dealloc_mw(Struct): pass
struct_ib_uverbs_dealloc_mw._fields_ = [
  ('mw_handle', ctypes.c_uint32),
]
class struct_ib_uverbs_create_comp_channel(Struct): pass
struct_ib_uverbs_create_comp_channel._fields_ = [
  ('response', ctypes.c_uint64),
]
class struct_ib_uverbs_create_comp_channel_resp(Struct): pass
struct_ib_uverbs_create_comp_channel_resp._fields_ = [
  ('fd', ctypes.c_uint32),
]
class struct_ib_uverbs_create_cq(Struct): pass
__s32 = ctypes.c_int32
struct_ib_uverbs_create_cq._fields_ = [
  ('response', ctypes.c_uint64),
  ('user_handle', ctypes.c_uint64),
  ('cqe', ctypes.c_uint32),
  ('comp_vector', ctypes.c_uint32),
  ('comp_channel', ctypes.c_int32),
  ('reserved', ctypes.c_uint32),
  ('driver_data', (ctypes.c_uint64 * 0)),
]
enum_ib_uverbs_ex_create_cq_flags = CEnum(ctypes.c_uint32)
IB_UVERBS_CQ_FLAGS_TIMESTAMP_COMPLETION = enum_ib_uverbs_ex_create_cq_flags.define('IB_UVERBS_CQ_FLAGS_TIMESTAMP_COMPLETION', 1)
IB_UVERBS_CQ_FLAGS_IGNORE_OVERRUN = enum_ib_uverbs_ex_create_cq_flags.define('IB_UVERBS_CQ_FLAGS_IGNORE_OVERRUN', 2)

class struct_ib_uverbs_ex_create_cq(Struct): pass
struct_ib_uverbs_ex_create_cq._fields_ = [
  ('user_handle', ctypes.c_uint64),
  ('cqe', ctypes.c_uint32),
  ('comp_vector', ctypes.c_uint32),
  ('comp_channel', ctypes.c_int32),
  ('comp_mask', ctypes.c_uint32),
  ('flags', ctypes.c_uint32),
  ('reserved', ctypes.c_uint32),
]
class struct_ib_uverbs_create_cq_resp(Struct): pass
struct_ib_uverbs_create_cq_resp._fields_ = [
  ('cq_handle', ctypes.c_uint32),
  ('cqe', ctypes.c_uint32),
  ('driver_data', (ctypes.c_uint64 * 0)),
]
class struct_ib_uverbs_ex_create_cq_resp(Struct): pass
struct_ib_uverbs_ex_create_cq_resp._fields_ = [
  ('base', struct_ib_uverbs_create_cq_resp),
  ('comp_mask', ctypes.c_uint32),
  ('response_length', ctypes.c_uint32),
]
class struct_ib_uverbs_resize_cq(Struct): pass
struct_ib_uverbs_resize_cq._fields_ = [
  ('response', ctypes.c_uint64),
  ('cq_handle', ctypes.c_uint32),
  ('cqe', ctypes.c_uint32),
  ('driver_data', (ctypes.c_uint64 * 0)),
]
class struct_ib_uverbs_resize_cq_resp(Struct): pass
struct_ib_uverbs_resize_cq_resp._fields_ = [
  ('cqe', ctypes.c_uint32),
  ('reserved', ctypes.c_uint32),
  ('driver_data', (ctypes.c_uint64 * 0)),
]
class struct_ib_uverbs_poll_cq(Struct): pass
struct_ib_uverbs_poll_cq._fields_ = [
  ('response', ctypes.c_uint64),
  ('cq_handle', ctypes.c_uint32),
  ('ne', ctypes.c_uint32),
]
enum_ib_uverbs_wc_opcode = CEnum(ctypes.c_uint32)
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

class struct_ib_uverbs_wc(Struct): pass
class struct_ib_uverbs_wc_ex(ctypes.Union): pass
struct_ib_uverbs_wc_ex._fields_ = [
  ('imm_data', ctypes.c_uint32),
  ('invalidate_rkey', ctypes.c_uint32),
]
struct_ib_uverbs_wc._fields_ = [
  ('wr_id', ctypes.c_uint64),
  ('status', ctypes.c_uint32),
  ('opcode', ctypes.c_uint32),
  ('vendor_err', ctypes.c_uint32),
  ('byte_len', ctypes.c_uint32),
  ('ex', struct_ib_uverbs_wc_ex),
  ('qp_num', ctypes.c_uint32),
  ('src_qp', ctypes.c_uint32),
  ('wc_flags', ctypes.c_uint32),
  ('pkey_index', ctypes.c_uint16),
  ('slid', ctypes.c_uint16),
  ('sl', ctypes.c_ubyte),
  ('dlid_path_bits', ctypes.c_ubyte),
  ('port_num', ctypes.c_ubyte),
  ('reserved', ctypes.c_ubyte),
]
class struct_ib_uverbs_poll_cq_resp(Struct): pass
struct_ib_uverbs_poll_cq_resp._fields_ = [
  ('count', ctypes.c_uint32),
  ('reserved', ctypes.c_uint32),
  ('wc', (struct_ib_uverbs_wc * 0)),
]
class struct_ib_uverbs_req_notify_cq(Struct): pass
struct_ib_uverbs_req_notify_cq._fields_ = [
  ('cq_handle', ctypes.c_uint32),
  ('solicited_only', ctypes.c_uint32),
]
class struct_ib_uverbs_destroy_cq(Struct): pass
struct_ib_uverbs_destroy_cq._fields_ = [
  ('response', ctypes.c_uint64),
  ('cq_handle', ctypes.c_uint32),
  ('reserved', ctypes.c_uint32),
]
class struct_ib_uverbs_destroy_cq_resp(Struct): pass
struct_ib_uverbs_destroy_cq_resp._fields_ = [
  ('comp_events_reported', ctypes.c_uint32),
  ('async_events_reported', ctypes.c_uint32),
]
class struct_ib_uverbs_global_route(Struct): pass
struct_ib_uverbs_global_route._fields_ = [
  ('dgid', (ctypes.c_ubyte * 16)),
  ('flow_label', ctypes.c_uint32),
  ('sgid_index', ctypes.c_ubyte),
  ('hop_limit', ctypes.c_ubyte),
  ('traffic_class', ctypes.c_ubyte),
  ('reserved', ctypes.c_ubyte),
]
class struct_ib_uverbs_ah_attr(Struct): pass
struct_ib_uverbs_ah_attr._fields_ = [
  ('grh', struct_ib_uverbs_global_route),
  ('dlid', ctypes.c_uint16),
  ('sl', ctypes.c_ubyte),
  ('src_path_bits', ctypes.c_ubyte),
  ('static_rate', ctypes.c_ubyte),
  ('is_global', ctypes.c_ubyte),
  ('port_num', ctypes.c_ubyte),
  ('reserved', ctypes.c_ubyte),
]
class struct_ib_uverbs_qp_attr(Struct): pass
struct_ib_uverbs_qp_attr._fields_ = [
  ('qp_attr_mask', ctypes.c_uint32),
  ('qp_state', ctypes.c_uint32),
  ('cur_qp_state', ctypes.c_uint32),
  ('path_mtu', ctypes.c_uint32),
  ('path_mig_state', ctypes.c_uint32),
  ('qkey', ctypes.c_uint32),
  ('rq_psn', ctypes.c_uint32),
  ('sq_psn', ctypes.c_uint32),
  ('dest_qp_num', ctypes.c_uint32),
  ('qp_access_flags', ctypes.c_uint32),
  ('ah_attr', struct_ib_uverbs_ah_attr),
  ('alt_ah_attr', struct_ib_uverbs_ah_attr),
  ('max_send_wr', ctypes.c_uint32),
  ('max_recv_wr', ctypes.c_uint32),
  ('max_send_sge', ctypes.c_uint32),
  ('max_recv_sge', ctypes.c_uint32),
  ('max_inline_data', ctypes.c_uint32),
  ('pkey_index', ctypes.c_uint16),
  ('alt_pkey_index', ctypes.c_uint16),
  ('en_sqd_async_notify', ctypes.c_ubyte),
  ('sq_draining', ctypes.c_ubyte),
  ('max_rd_atomic', ctypes.c_ubyte),
  ('max_dest_rd_atomic', ctypes.c_ubyte),
  ('min_rnr_timer', ctypes.c_ubyte),
  ('port_num', ctypes.c_ubyte),
  ('timeout', ctypes.c_ubyte),
  ('retry_cnt', ctypes.c_ubyte),
  ('rnr_retry', ctypes.c_ubyte),
  ('alt_port_num', ctypes.c_ubyte),
  ('alt_timeout', ctypes.c_ubyte),
  ('reserved', (ctypes.c_ubyte * 5)),
]
class struct_ib_uverbs_create_qp(Struct): pass
struct_ib_uverbs_create_qp._fields_ = [
  ('response', ctypes.c_uint64),
  ('user_handle', ctypes.c_uint64),
  ('pd_handle', ctypes.c_uint32),
  ('send_cq_handle', ctypes.c_uint32),
  ('recv_cq_handle', ctypes.c_uint32),
  ('srq_handle', ctypes.c_uint32),
  ('max_send_wr', ctypes.c_uint32),
  ('max_recv_wr', ctypes.c_uint32),
  ('max_send_sge', ctypes.c_uint32),
  ('max_recv_sge', ctypes.c_uint32),
  ('max_inline_data', ctypes.c_uint32),
  ('sq_sig_all', ctypes.c_ubyte),
  ('qp_type', ctypes.c_ubyte),
  ('is_srq', ctypes.c_ubyte),
  ('reserved', ctypes.c_ubyte),
  ('driver_data', (ctypes.c_uint64 * 0)),
]
enum_ib_uverbs_create_qp_mask = CEnum(ctypes.c_uint32)
IB_UVERBS_CREATE_QP_MASK_IND_TABLE = enum_ib_uverbs_create_qp_mask.define('IB_UVERBS_CREATE_QP_MASK_IND_TABLE', 1)

_anonenum6 = CEnum(ctypes.c_uint32)
IB_UVERBS_CREATE_QP_SUP_COMP_MASK = _anonenum6.define('IB_UVERBS_CREATE_QP_SUP_COMP_MASK', 1)

class struct_ib_uverbs_ex_create_qp(Struct): pass
struct_ib_uverbs_ex_create_qp._fields_ = [
  ('user_handle', ctypes.c_uint64),
  ('pd_handle', ctypes.c_uint32),
  ('send_cq_handle', ctypes.c_uint32),
  ('recv_cq_handle', ctypes.c_uint32),
  ('srq_handle', ctypes.c_uint32),
  ('max_send_wr', ctypes.c_uint32),
  ('max_recv_wr', ctypes.c_uint32),
  ('max_send_sge', ctypes.c_uint32),
  ('max_recv_sge', ctypes.c_uint32),
  ('max_inline_data', ctypes.c_uint32),
  ('sq_sig_all', ctypes.c_ubyte),
  ('qp_type', ctypes.c_ubyte),
  ('is_srq', ctypes.c_ubyte),
  ('reserved', ctypes.c_ubyte),
  ('comp_mask', ctypes.c_uint32),
  ('create_flags', ctypes.c_uint32),
  ('rwq_ind_tbl_handle', ctypes.c_uint32),
  ('source_qpn', ctypes.c_uint32),
]
class struct_ib_uverbs_open_qp(Struct): pass
struct_ib_uverbs_open_qp._fields_ = [
  ('response', ctypes.c_uint64),
  ('user_handle', ctypes.c_uint64),
  ('pd_handle', ctypes.c_uint32),
  ('qpn', ctypes.c_uint32),
  ('qp_type', ctypes.c_ubyte),
  ('reserved', (ctypes.c_ubyte * 7)),
  ('driver_data', (ctypes.c_uint64 * 0)),
]
class struct_ib_uverbs_create_qp_resp(Struct): pass
struct_ib_uverbs_create_qp_resp._fields_ = [
  ('qp_handle', ctypes.c_uint32),
  ('qpn', ctypes.c_uint32),
  ('max_send_wr', ctypes.c_uint32),
  ('max_recv_wr', ctypes.c_uint32),
  ('max_send_sge', ctypes.c_uint32),
  ('max_recv_sge', ctypes.c_uint32),
  ('max_inline_data', ctypes.c_uint32),
  ('reserved', ctypes.c_uint32),
  ('driver_data', (ctypes.c_uint32 * 0)),
]
class struct_ib_uverbs_ex_create_qp_resp(Struct): pass
struct_ib_uverbs_ex_create_qp_resp._fields_ = [
  ('base', struct_ib_uverbs_create_qp_resp),
  ('comp_mask', ctypes.c_uint32),
  ('response_length', ctypes.c_uint32),
]
class struct_ib_uverbs_qp_dest(Struct): pass
struct_ib_uverbs_qp_dest._fields_ = [
  ('dgid', (ctypes.c_ubyte * 16)),
  ('flow_label', ctypes.c_uint32),
  ('dlid', ctypes.c_uint16),
  ('reserved', ctypes.c_uint16),
  ('sgid_index', ctypes.c_ubyte),
  ('hop_limit', ctypes.c_ubyte),
  ('traffic_class', ctypes.c_ubyte),
  ('sl', ctypes.c_ubyte),
  ('src_path_bits', ctypes.c_ubyte),
  ('static_rate', ctypes.c_ubyte),
  ('is_global', ctypes.c_ubyte),
  ('port_num', ctypes.c_ubyte),
]
class struct_ib_uverbs_query_qp(Struct): pass
struct_ib_uverbs_query_qp._fields_ = [
  ('response', ctypes.c_uint64),
  ('qp_handle', ctypes.c_uint32),
  ('attr_mask', ctypes.c_uint32),
  ('driver_data', (ctypes.c_uint64 * 0)),
]
class struct_ib_uverbs_query_qp_resp(Struct): pass
struct_ib_uverbs_query_qp_resp._fields_ = [
  ('dest', struct_ib_uverbs_qp_dest),
  ('alt_dest', struct_ib_uverbs_qp_dest),
  ('max_send_wr', ctypes.c_uint32),
  ('max_recv_wr', ctypes.c_uint32),
  ('max_send_sge', ctypes.c_uint32),
  ('max_recv_sge', ctypes.c_uint32),
  ('max_inline_data', ctypes.c_uint32),
  ('qkey', ctypes.c_uint32),
  ('rq_psn', ctypes.c_uint32),
  ('sq_psn', ctypes.c_uint32),
  ('dest_qp_num', ctypes.c_uint32),
  ('qp_access_flags', ctypes.c_uint32),
  ('pkey_index', ctypes.c_uint16),
  ('alt_pkey_index', ctypes.c_uint16),
  ('qp_state', ctypes.c_ubyte),
  ('cur_qp_state', ctypes.c_ubyte),
  ('path_mtu', ctypes.c_ubyte),
  ('path_mig_state', ctypes.c_ubyte),
  ('sq_draining', ctypes.c_ubyte),
  ('max_rd_atomic', ctypes.c_ubyte),
  ('max_dest_rd_atomic', ctypes.c_ubyte),
  ('min_rnr_timer', ctypes.c_ubyte),
  ('port_num', ctypes.c_ubyte),
  ('timeout', ctypes.c_ubyte),
  ('retry_cnt', ctypes.c_ubyte),
  ('rnr_retry', ctypes.c_ubyte),
  ('alt_port_num', ctypes.c_ubyte),
  ('alt_timeout', ctypes.c_ubyte),
  ('sq_sig_all', ctypes.c_ubyte),
  ('reserved', (ctypes.c_ubyte * 5)),
  ('driver_data', (ctypes.c_uint64 * 0)),
]
class struct_ib_uverbs_modify_qp(Struct): pass
struct_ib_uverbs_modify_qp._fields_ = [
  ('dest', struct_ib_uverbs_qp_dest),
  ('alt_dest', struct_ib_uverbs_qp_dest),
  ('qp_handle', ctypes.c_uint32),
  ('attr_mask', ctypes.c_uint32),
  ('qkey', ctypes.c_uint32),
  ('rq_psn', ctypes.c_uint32),
  ('sq_psn', ctypes.c_uint32),
  ('dest_qp_num', ctypes.c_uint32),
  ('qp_access_flags', ctypes.c_uint32),
  ('pkey_index', ctypes.c_uint16),
  ('alt_pkey_index', ctypes.c_uint16),
  ('qp_state', ctypes.c_ubyte),
  ('cur_qp_state', ctypes.c_ubyte),
  ('path_mtu', ctypes.c_ubyte),
  ('path_mig_state', ctypes.c_ubyte),
  ('en_sqd_async_notify', ctypes.c_ubyte),
  ('max_rd_atomic', ctypes.c_ubyte),
  ('max_dest_rd_atomic', ctypes.c_ubyte),
  ('min_rnr_timer', ctypes.c_ubyte),
  ('port_num', ctypes.c_ubyte),
  ('timeout', ctypes.c_ubyte),
  ('retry_cnt', ctypes.c_ubyte),
  ('rnr_retry', ctypes.c_ubyte),
  ('alt_port_num', ctypes.c_ubyte),
  ('alt_timeout', ctypes.c_ubyte),
  ('reserved', (ctypes.c_ubyte * 2)),
  ('driver_data', (ctypes.c_uint64 * 0)),
]
class struct_ib_uverbs_ex_modify_qp(Struct): pass
struct_ib_uverbs_ex_modify_qp._fields_ = [
  ('base', struct_ib_uverbs_modify_qp),
  ('rate_limit', ctypes.c_uint32),
  ('reserved', ctypes.c_uint32),
]
class struct_ib_uverbs_ex_modify_qp_resp(Struct): pass
struct_ib_uverbs_ex_modify_qp_resp._fields_ = [
  ('comp_mask', ctypes.c_uint32),
  ('response_length', ctypes.c_uint32),
]
class struct_ib_uverbs_destroy_qp(Struct): pass
struct_ib_uverbs_destroy_qp._fields_ = [
  ('response', ctypes.c_uint64),
  ('qp_handle', ctypes.c_uint32),
  ('reserved', ctypes.c_uint32),
]
class struct_ib_uverbs_destroy_qp_resp(Struct): pass
struct_ib_uverbs_destroy_qp_resp._fields_ = [
  ('events_reported', ctypes.c_uint32),
]
class struct_ib_uverbs_sge(Struct): pass
struct_ib_uverbs_sge._fields_ = [
  ('addr', ctypes.c_uint64),
  ('length', ctypes.c_uint32),
  ('lkey', ctypes.c_uint32),
]
enum_ib_uverbs_wr_opcode = CEnum(ctypes.c_uint32)
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

class struct_ib_uverbs_send_wr(Struct): pass
class struct_ib_uverbs_send_wr_ex(ctypes.Union): pass
struct_ib_uverbs_send_wr_ex._fields_ = [
  ('imm_data', ctypes.c_uint32),
  ('invalidate_rkey', ctypes.c_uint32),
]
class struct_ib_uverbs_send_wr_wr(ctypes.Union): pass
class struct_ib_uverbs_send_wr_wr_rdma(Struct): pass
struct_ib_uverbs_send_wr_wr_rdma._fields_ = [
  ('remote_addr', ctypes.c_uint64),
  ('rkey', ctypes.c_uint32),
  ('reserved', ctypes.c_uint32),
]
class struct_ib_uverbs_send_wr_wr_atomic(Struct): pass
struct_ib_uverbs_send_wr_wr_atomic._fields_ = [
  ('remote_addr', ctypes.c_uint64),
  ('compare_add', ctypes.c_uint64),
  ('swap', ctypes.c_uint64),
  ('rkey', ctypes.c_uint32),
  ('reserved', ctypes.c_uint32),
]
class struct_ib_uverbs_send_wr_wr_ud(Struct): pass
struct_ib_uverbs_send_wr_wr_ud._fields_ = [
  ('ah', ctypes.c_uint32),
  ('remote_qpn', ctypes.c_uint32),
  ('remote_qkey', ctypes.c_uint32),
  ('reserved', ctypes.c_uint32),
]
struct_ib_uverbs_send_wr_wr._fields_ = [
  ('rdma', struct_ib_uverbs_send_wr_wr_rdma),
  ('atomic', struct_ib_uverbs_send_wr_wr_atomic),
  ('ud', struct_ib_uverbs_send_wr_wr_ud),
]
struct_ib_uverbs_send_wr._fields_ = [
  ('wr_id', ctypes.c_uint64),
  ('num_sge', ctypes.c_uint32),
  ('opcode', ctypes.c_uint32),
  ('send_flags', ctypes.c_uint32),
  ('ex', struct_ib_uverbs_send_wr_ex),
  ('wr', struct_ib_uverbs_send_wr_wr),
]
class struct_ib_uverbs_post_send(Struct): pass
struct_ib_uverbs_post_send._fields_ = [
  ('response', ctypes.c_uint64),
  ('qp_handle', ctypes.c_uint32),
  ('wr_count', ctypes.c_uint32),
  ('sge_count', ctypes.c_uint32),
  ('wqe_size', ctypes.c_uint32),
  ('send_wr', (struct_ib_uverbs_send_wr * 0)),
]
class struct_ib_uverbs_post_send_resp(Struct): pass
struct_ib_uverbs_post_send_resp._fields_ = [
  ('bad_wr', ctypes.c_uint32),
]
class struct_ib_uverbs_recv_wr(Struct): pass
struct_ib_uverbs_recv_wr._fields_ = [
  ('wr_id', ctypes.c_uint64),
  ('num_sge', ctypes.c_uint32),
  ('reserved', ctypes.c_uint32),
]
class struct_ib_uverbs_post_recv(Struct): pass
struct_ib_uverbs_post_recv._fields_ = [
  ('response', ctypes.c_uint64),
  ('qp_handle', ctypes.c_uint32),
  ('wr_count', ctypes.c_uint32),
  ('sge_count', ctypes.c_uint32),
  ('wqe_size', ctypes.c_uint32),
  ('recv_wr', (struct_ib_uverbs_recv_wr * 0)),
]
class struct_ib_uverbs_post_recv_resp(Struct): pass
struct_ib_uverbs_post_recv_resp._fields_ = [
  ('bad_wr', ctypes.c_uint32),
]
class struct_ib_uverbs_post_srq_recv(Struct): pass
struct_ib_uverbs_post_srq_recv._fields_ = [
  ('response', ctypes.c_uint64),
  ('srq_handle', ctypes.c_uint32),
  ('wr_count', ctypes.c_uint32),
  ('sge_count', ctypes.c_uint32),
  ('wqe_size', ctypes.c_uint32),
  ('recv', (struct_ib_uverbs_recv_wr * 0)),
]
class struct_ib_uverbs_post_srq_recv_resp(Struct): pass
struct_ib_uverbs_post_srq_recv_resp._fields_ = [
  ('bad_wr', ctypes.c_uint32),
]
class struct_ib_uverbs_create_ah(Struct): pass
struct_ib_uverbs_create_ah._fields_ = [
  ('response', ctypes.c_uint64),
  ('user_handle', ctypes.c_uint64),
  ('pd_handle', ctypes.c_uint32),
  ('reserved', ctypes.c_uint32),
  ('attr', struct_ib_uverbs_ah_attr),
  ('driver_data', (ctypes.c_uint64 * 0)),
]
class struct_ib_uverbs_create_ah_resp(Struct): pass
struct_ib_uverbs_create_ah_resp._fields_ = [
  ('ah_handle', ctypes.c_uint32),
  ('driver_data', (ctypes.c_uint32 * 0)),
]
class struct_ib_uverbs_destroy_ah(Struct): pass
struct_ib_uverbs_destroy_ah._fields_ = [
  ('ah_handle', ctypes.c_uint32),
]
class struct_ib_uverbs_attach_mcast(Struct): pass
struct_ib_uverbs_attach_mcast._fields_ = [
  ('gid', (ctypes.c_ubyte * 16)),
  ('qp_handle', ctypes.c_uint32),
  ('mlid', ctypes.c_uint16),
  ('reserved', ctypes.c_uint16),
  ('driver_data', (ctypes.c_uint64 * 0)),
]
class struct_ib_uverbs_detach_mcast(Struct): pass
struct_ib_uverbs_detach_mcast._fields_ = [
  ('gid', (ctypes.c_ubyte * 16)),
  ('qp_handle', ctypes.c_uint32),
  ('mlid', ctypes.c_uint16),
  ('reserved', ctypes.c_uint16),
  ('driver_data', (ctypes.c_uint64 * 0)),
]
class struct_ib_uverbs_flow_spec_hdr(Struct): pass
struct_ib_uverbs_flow_spec_hdr._fields_ = [
  ('type', ctypes.c_uint32),
  ('size', ctypes.c_uint16),
  ('reserved', ctypes.c_uint16),
  ('flow_spec_data', (ctypes.c_uint64 * 0)),
]
class struct_ib_uverbs_flow_eth_filter(Struct): pass
struct_ib_uverbs_flow_eth_filter._fields_ = [
  ('dst_mac', (ctypes.c_ubyte * 6)),
  ('src_mac', (ctypes.c_ubyte * 6)),
  ('ether_type', ctypes.c_uint16),
  ('vlan_tag', ctypes.c_uint16),
]
class struct_ib_uverbs_flow_spec_eth(Struct): pass
class struct_ib_uverbs_flow_spec_eth_0(ctypes.Union): pass
class struct_ib_uverbs_flow_spec_eth_0_0(Struct): pass
struct_ib_uverbs_flow_spec_eth_0_0._fields_ = [
  ('type', ctypes.c_uint32),
  ('size', ctypes.c_uint16),
  ('reserved', ctypes.c_uint16),
]
struct_ib_uverbs_flow_spec_eth_0._anonymous_ = ['_0']
struct_ib_uverbs_flow_spec_eth_0._fields_ = [
  ('hdr', struct_ib_uverbs_flow_spec_hdr),
  ('_0', struct_ib_uverbs_flow_spec_eth_0_0),
]
struct_ib_uverbs_flow_spec_eth._anonymous_ = ['_0']
struct_ib_uverbs_flow_spec_eth._fields_ = [
  ('_0', struct_ib_uverbs_flow_spec_eth_0),
  ('val', struct_ib_uverbs_flow_eth_filter),
  ('mask', struct_ib_uverbs_flow_eth_filter),
]
class struct_ib_uverbs_flow_ipv4_filter(Struct): pass
struct_ib_uverbs_flow_ipv4_filter._fields_ = [
  ('src_ip', ctypes.c_uint32),
  ('dst_ip', ctypes.c_uint32),
  ('proto', ctypes.c_ubyte),
  ('tos', ctypes.c_ubyte),
  ('ttl', ctypes.c_ubyte),
  ('flags', ctypes.c_ubyte),
]
class struct_ib_uverbs_flow_spec_ipv4(Struct): pass
class struct_ib_uverbs_flow_spec_ipv4_0(ctypes.Union): pass
class struct_ib_uverbs_flow_spec_ipv4_0_0(Struct): pass
struct_ib_uverbs_flow_spec_ipv4_0_0._fields_ = [
  ('type', ctypes.c_uint32),
  ('size', ctypes.c_uint16),
  ('reserved', ctypes.c_uint16),
]
struct_ib_uverbs_flow_spec_ipv4_0._anonymous_ = ['_0']
struct_ib_uverbs_flow_spec_ipv4_0._fields_ = [
  ('hdr', struct_ib_uverbs_flow_spec_hdr),
  ('_0', struct_ib_uverbs_flow_spec_ipv4_0_0),
]
struct_ib_uverbs_flow_spec_ipv4._anonymous_ = ['_0']
struct_ib_uverbs_flow_spec_ipv4._fields_ = [
  ('_0', struct_ib_uverbs_flow_spec_ipv4_0),
  ('val', struct_ib_uverbs_flow_ipv4_filter),
  ('mask', struct_ib_uverbs_flow_ipv4_filter),
]
class struct_ib_uverbs_flow_tcp_udp_filter(Struct): pass
struct_ib_uverbs_flow_tcp_udp_filter._fields_ = [
  ('dst_port', ctypes.c_uint16),
  ('src_port', ctypes.c_uint16),
]
class struct_ib_uverbs_flow_spec_tcp_udp(Struct): pass
class struct_ib_uverbs_flow_spec_tcp_udp_0(ctypes.Union): pass
class struct_ib_uverbs_flow_spec_tcp_udp_0_0(Struct): pass
struct_ib_uverbs_flow_spec_tcp_udp_0_0._fields_ = [
  ('type', ctypes.c_uint32),
  ('size', ctypes.c_uint16),
  ('reserved', ctypes.c_uint16),
]
struct_ib_uverbs_flow_spec_tcp_udp_0._anonymous_ = ['_0']
struct_ib_uverbs_flow_spec_tcp_udp_0._fields_ = [
  ('hdr', struct_ib_uverbs_flow_spec_hdr),
  ('_0', struct_ib_uverbs_flow_spec_tcp_udp_0_0),
]
struct_ib_uverbs_flow_spec_tcp_udp._anonymous_ = ['_0']
struct_ib_uverbs_flow_spec_tcp_udp._fields_ = [
  ('_0', struct_ib_uverbs_flow_spec_tcp_udp_0),
  ('val', struct_ib_uverbs_flow_tcp_udp_filter),
  ('mask', struct_ib_uverbs_flow_tcp_udp_filter),
]
class struct_ib_uverbs_flow_ipv6_filter(Struct): pass
struct_ib_uverbs_flow_ipv6_filter._fields_ = [
  ('src_ip', (ctypes.c_ubyte * 16)),
  ('dst_ip', (ctypes.c_ubyte * 16)),
  ('flow_label', ctypes.c_uint32),
  ('next_hdr', ctypes.c_ubyte),
  ('traffic_class', ctypes.c_ubyte),
  ('hop_limit', ctypes.c_ubyte),
  ('reserved', ctypes.c_ubyte),
]
class struct_ib_uverbs_flow_spec_ipv6(Struct): pass
class struct_ib_uverbs_flow_spec_ipv6_0(ctypes.Union): pass
class struct_ib_uverbs_flow_spec_ipv6_0_0(Struct): pass
struct_ib_uverbs_flow_spec_ipv6_0_0._fields_ = [
  ('type', ctypes.c_uint32),
  ('size', ctypes.c_uint16),
  ('reserved', ctypes.c_uint16),
]
struct_ib_uverbs_flow_spec_ipv6_0._anonymous_ = ['_0']
struct_ib_uverbs_flow_spec_ipv6_0._fields_ = [
  ('hdr', struct_ib_uverbs_flow_spec_hdr),
  ('_0', struct_ib_uverbs_flow_spec_ipv6_0_0),
]
struct_ib_uverbs_flow_spec_ipv6._anonymous_ = ['_0']
struct_ib_uverbs_flow_spec_ipv6._fields_ = [
  ('_0', struct_ib_uverbs_flow_spec_ipv6_0),
  ('val', struct_ib_uverbs_flow_ipv6_filter),
  ('mask', struct_ib_uverbs_flow_ipv6_filter),
]
class struct_ib_uverbs_flow_spec_action_tag(Struct): pass
class struct_ib_uverbs_flow_spec_action_tag_0(ctypes.Union): pass
class struct_ib_uverbs_flow_spec_action_tag_0_0(Struct): pass
struct_ib_uverbs_flow_spec_action_tag_0_0._fields_ = [
  ('type', ctypes.c_uint32),
  ('size', ctypes.c_uint16),
  ('reserved', ctypes.c_uint16),
]
struct_ib_uverbs_flow_spec_action_tag_0._anonymous_ = ['_0']
struct_ib_uverbs_flow_spec_action_tag_0._fields_ = [
  ('hdr', struct_ib_uverbs_flow_spec_hdr),
  ('_0', struct_ib_uverbs_flow_spec_action_tag_0_0),
]
struct_ib_uverbs_flow_spec_action_tag._anonymous_ = ['_0']
struct_ib_uverbs_flow_spec_action_tag._fields_ = [
  ('_0', struct_ib_uverbs_flow_spec_action_tag_0),
  ('tag_id', ctypes.c_uint32),
  ('reserved1', ctypes.c_uint32),
]
class struct_ib_uverbs_flow_spec_action_drop(Struct): pass
class struct_ib_uverbs_flow_spec_action_drop_0(ctypes.Union): pass
class struct_ib_uverbs_flow_spec_action_drop_0_0(Struct): pass
struct_ib_uverbs_flow_spec_action_drop_0_0._fields_ = [
  ('type', ctypes.c_uint32),
  ('size', ctypes.c_uint16),
  ('reserved', ctypes.c_uint16),
]
struct_ib_uverbs_flow_spec_action_drop_0._anonymous_ = ['_0']
struct_ib_uverbs_flow_spec_action_drop_0._fields_ = [
  ('hdr', struct_ib_uverbs_flow_spec_hdr),
  ('_0', struct_ib_uverbs_flow_spec_action_drop_0_0),
]
struct_ib_uverbs_flow_spec_action_drop._anonymous_ = ['_0']
struct_ib_uverbs_flow_spec_action_drop._fields_ = [
  ('_0', struct_ib_uverbs_flow_spec_action_drop_0),
]
class struct_ib_uverbs_flow_spec_action_handle(Struct): pass
class struct_ib_uverbs_flow_spec_action_handle_0(ctypes.Union): pass
class struct_ib_uverbs_flow_spec_action_handle_0_0(Struct): pass
struct_ib_uverbs_flow_spec_action_handle_0_0._fields_ = [
  ('type', ctypes.c_uint32),
  ('size', ctypes.c_uint16),
  ('reserved', ctypes.c_uint16),
]
struct_ib_uverbs_flow_spec_action_handle_0._anonymous_ = ['_0']
struct_ib_uverbs_flow_spec_action_handle_0._fields_ = [
  ('hdr', struct_ib_uverbs_flow_spec_hdr),
  ('_0', struct_ib_uverbs_flow_spec_action_handle_0_0),
]
struct_ib_uverbs_flow_spec_action_handle._anonymous_ = ['_0']
struct_ib_uverbs_flow_spec_action_handle._fields_ = [
  ('_0', struct_ib_uverbs_flow_spec_action_handle_0),
  ('handle', ctypes.c_uint32),
  ('reserved1', ctypes.c_uint32),
]
class struct_ib_uverbs_flow_spec_action_count(Struct): pass
class struct_ib_uverbs_flow_spec_action_count_0(ctypes.Union): pass
class struct_ib_uverbs_flow_spec_action_count_0_0(Struct): pass
struct_ib_uverbs_flow_spec_action_count_0_0._fields_ = [
  ('type', ctypes.c_uint32),
  ('size', ctypes.c_uint16),
  ('reserved', ctypes.c_uint16),
]
struct_ib_uverbs_flow_spec_action_count_0._anonymous_ = ['_0']
struct_ib_uverbs_flow_spec_action_count_0._fields_ = [
  ('hdr', struct_ib_uverbs_flow_spec_hdr),
  ('_0', struct_ib_uverbs_flow_spec_action_count_0_0),
]
struct_ib_uverbs_flow_spec_action_count._anonymous_ = ['_0']
struct_ib_uverbs_flow_spec_action_count._fields_ = [
  ('_0', struct_ib_uverbs_flow_spec_action_count_0),
  ('handle', ctypes.c_uint32),
  ('reserved1', ctypes.c_uint32),
]
class struct_ib_uverbs_flow_tunnel_filter(Struct): pass
struct_ib_uverbs_flow_tunnel_filter._fields_ = [
  ('tunnel_id', ctypes.c_uint32),
]
class struct_ib_uverbs_flow_spec_tunnel(Struct): pass
class struct_ib_uverbs_flow_spec_tunnel_0(ctypes.Union): pass
class struct_ib_uverbs_flow_spec_tunnel_0_0(Struct): pass
struct_ib_uverbs_flow_spec_tunnel_0_0._fields_ = [
  ('type', ctypes.c_uint32),
  ('size', ctypes.c_uint16),
  ('reserved', ctypes.c_uint16),
]
struct_ib_uverbs_flow_spec_tunnel_0._anonymous_ = ['_0']
struct_ib_uverbs_flow_spec_tunnel_0._fields_ = [
  ('hdr', struct_ib_uverbs_flow_spec_hdr),
  ('_0', struct_ib_uverbs_flow_spec_tunnel_0_0),
]
struct_ib_uverbs_flow_spec_tunnel._anonymous_ = ['_0']
struct_ib_uverbs_flow_spec_tunnel._fields_ = [
  ('_0', struct_ib_uverbs_flow_spec_tunnel_0),
  ('val', struct_ib_uverbs_flow_tunnel_filter),
  ('mask', struct_ib_uverbs_flow_tunnel_filter),
]
class struct_ib_uverbs_flow_spec_esp_filter(Struct): pass
struct_ib_uverbs_flow_spec_esp_filter._fields_ = [
  ('spi', ctypes.c_uint32),
  ('seq', ctypes.c_uint32),
]
class struct_ib_uverbs_flow_spec_esp(Struct): pass
class struct_ib_uverbs_flow_spec_esp_0(ctypes.Union): pass
class struct_ib_uverbs_flow_spec_esp_0_0(Struct): pass
struct_ib_uverbs_flow_spec_esp_0_0._fields_ = [
  ('type', ctypes.c_uint32),
  ('size', ctypes.c_uint16),
  ('reserved', ctypes.c_uint16),
]
struct_ib_uverbs_flow_spec_esp_0._anonymous_ = ['_0']
struct_ib_uverbs_flow_spec_esp_0._fields_ = [
  ('hdr', struct_ib_uverbs_flow_spec_hdr),
  ('_0', struct_ib_uverbs_flow_spec_esp_0_0),
]
struct_ib_uverbs_flow_spec_esp._anonymous_ = ['_0']
struct_ib_uverbs_flow_spec_esp._fields_ = [
  ('_0', struct_ib_uverbs_flow_spec_esp_0),
  ('val', struct_ib_uverbs_flow_spec_esp_filter),
  ('mask', struct_ib_uverbs_flow_spec_esp_filter),
]
class struct_ib_uverbs_flow_gre_filter(Struct): pass
struct_ib_uverbs_flow_gre_filter._fields_ = [
  ('c_ks_res0_ver', ctypes.c_uint16),
  ('protocol', ctypes.c_uint16),
  ('key', ctypes.c_uint32),
]
class struct_ib_uverbs_flow_spec_gre(Struct): pass
class struct_ib_uverbs_flow_spec_gre_0(ctypes.Union): pass
class struct_ib_uverbs_flow_spec_gre_0_0(Struct): pass
struct_ib_uverbs_flow_spec_gre_0_0._fields_ = [
  ('type', ctypes.c_uint32),
  ('size', ctypes.c_uint16),
  ('reserved', ctypes.c_uint16),
]
struct_ib_uverbs_flow_spec_gre_0._anonymous_ = ['_0']
struct_ib_uverbs_flow_spec_gre_0._fields_ = [
  ('hdr', struct_ib_uverbs_flow_spec_hdr),
  ('_0', struct_ib_uverbs_flow_spec_gre_0_0),
]
struct_ib_uverbs_flow_spec_gre._anonymous_ = ['_0']
struct_ib_uverbs_flow_spec_gre._fields_ = [
  ('_0', struct_ib_uverbs_flow_spec_gre_0),
  ('val', struct_ib_uverbs_flow_gre_filter),
  ('mask', struct_ib_uverbs_flow_gre_filter),
]
class struct_ib_uverbs_flow_mpls_filter(Struct): pass
struct_ib_uverbs_flow_mpls_filter._fields_ = [
  ('label', ctypes.c_uint32),
]
class struct_ib_uverbs_flow_spec_mpls(Struct): pass
class struct_ib_uverbs_flow_spec_mpls_0(ctypes.Union): pass
class struct_ib_uverbs_flow_spec_mpls_0_0(Struct): pass
struct_ib_uverbs_flow_spec_mpls_0_0._fields_ = [
  ('type', ctypes.c_uint32),
  ('size', ctypes.c_uint16),
  ('reserved', ctypes.c_uint16),
]
struct_ib_uverbs_flow_spec_mpls_0._anonymous_ = ['_0']
struct_ib_uverbs_flow_spec_mpls_0._fields_ = [
  ('hdr', struct_ib_uverbs_flow_spec_hdr),
  ('_0', struct_ib_uverbs_flow_spec_mpls_0_0),
]
struct_ib_uverbs_flow_spec_mpls._anonymous_ = ['_0']
struct_ib_uverbs_flow_spec_mpls._fields_ = [
  ('_0', struct_ib_uverbs_flow_spec_mpls_0),
  ('val', struct_ib_uverbs_flow_mpls_filter),
  ('mask', struct_ib_uverbs_flow_mpls_filter),
]
class struct_ib_uverbs_flow_attr(Struct): pass
struct_ib_uverbs_flow_attr._fields_ = [
  ('type', ctypes.c_uint32),
  ('size', ctypes.c_uint16),
  ('priority', ctypes.c_uint16),
  ('num_of_specs', ctypes.c_ubyte),
  ('reserved', (ctypes.c_ubyte * 2)),
  ('port', ctypes.c_ubyte),
  ('flags', ctypes.c_uint32),
  ('flow_specs', (struct_ib_uverbs_flow_spec_hdr * 0)),
]
class struct_ib_uverbs_create_flow(Struct): pass
struct_ib_uverbs_create_flow._fields_ = [
  ('comp_mask', ctypes.c_uint32),
  ('qp_handle', ctypes.c_uint32),
  ('flow_attr', struct_ib_uverbs_flow_attr),
]
class struct_ib_uverbs_create_flow_resp(Struct): pass
struct_ib_uverbs_create_flow_resp._fields_ = [
  ('comp_mask', ctypes.c_uint32),
  ('flow_handle', ctypes.c_uint32),
]
class struct_ib_uverbs_destroy_flow(Struct): pass
struct_ib_uverbs_destroy_flow._fields_ = [
  ('comp_mask', ctypes.c_uint32),
  ('flow_handle', ctypes.c_uint32),
]
class struct_ib_uverbs_create_srq(Struct): pass
struct_ib_uverbs_create_srq._fields_ = [
  ('response', ctypes.c_uint64),
  ('user_handle', ctypes.c_uint64),
  ('pd_handle', ctypes.c_uint32),
  ('max_wr', ctypes.c_uint32),
  ('max_sge', ctypes.c_uint32),
  ('srq_limit', ctypes.c_uint32),
  ('driver_data', (ctypes.c_uint64 * 0)),
]
class struct_ib_uverbs_create_xsrq(Struct): pass
struct_ib_uverbs_create_xsrq._fields_ = [
  ('response', ctypes.c_uint64),
  ('user_handle', ctypes.c_uint64),
  ('srq_type', ctypes.c_uint32),
  ('pd_handle', ctypes.c_uint32),
  ('max_wr', ctypes.c_uint32),
  ('max_sge', ctypes.c_uint32),
  ('srq_limit', ctypes.c_uint32),
  ('max_num_tags', ctypes.c_uint32),
  ('xrcd_handle', ctypes.c_uint32),
  ('cq_handle', ctypes.c_uint32),
  ('driver_data', (ctypes.c_uint64 * 0)),
]
class struct_ib_uverbs_create_srq_resp(Struct): pass
struct_ib_uverbs_create_srq_resp._fields_ = [
  ('srq_handle', ctypes.c_uint32),
  ('max_wr', ctypes.c_uint32),
  ('max_sge', ctypes.c_uint32),
  ('srqn', ctypes.c_uint32),
  ('driver_data', (ctypes.c_uint32 * 0)),
]
class struct_ib_uverbs_modify_srq(Struct): pass
struct_ib_uverbs_modify_srq._fields_ = [
  ('srq_handle', ctypes.c_uint32),
  ('attr_mask', ctypes.c_uint32),
  ('max_wr', ctypes.c_uint32),
  ('srq_limit', ctypes.c_uint32),
  ('driver_data', (ctypes.c_uint64 * 0)),
]
class struct_ib_uverbs_query_srq(Struct): pass
struct_ib_uverbs_query_srq._fields_ = [
  ('response', ctypes.c_uint64),
  ('srq_handle', ctypes.c_uint32),
  ('reserved', ctypes.c_uint32),
  ('driver_data', (ctypes.c_uint64 * 0)),
]
class struct_ib_uverbs_query_srq_resp(Struct): pass
struct_ib_uverbs_query_srq_resp._fields_ = [
  ('max_wr', ctypes.c_uint32),
  ('max_sge', ctypes.c_uint32),
  ('srq_limit', ctypes.c_uint32),
  ('reserved', ctypes.c_uint32),
]
class struct_ib_uverbs_destroy_srq(Struct): pass
struct_ib_uverbs_destroy_srq._fields_ = [
  ('response', ctypes.c_uint64),
  ('srq_handle', ctypes.c_uint32),
  ('reserved', ctypes.c_uint32),
]
class struct_ib_uverbs_destroy_srq_resp(Struct): pass
struct_ib_uverbs_destroy_srq_resp._fields_ = [
  ('events_reported', ctypes.c_uint32),
]
class struct_ib_uverbs_ex_create_wq(Struct): pass
struct_ib_uverbs_ex_create_wq._fields_ = [
  ('comp_mask', ctypes.c_uint32),
  ('wq_type', ctypes.c_uint32),
  ('user_handle', ctypes.c_uint64),
  ('pd_handle', ctypes.c_uint32),
  ('cq_handle', ctypes.c_uint32),
  ('max_wr', ctypes.c_uint32),
  ('max_sge', ctypes.c_uint32),
  ('create_flags', ctypes.c_uint32),
  ('reserved', ctypes.c_uint32),
]
class struct_ib_uverbs_ex_create_wq_resp(Struct): pass
struct_ib_uverbs_ex_create_wq_resp._fields_ = [
  ('comp_mask', ctypes.c_uint32),
  ('response_length', ctypes.c_uint32),
  ('wq_handle', ctypes.c_uint32),
  ('max_wr', ctypes.c_uint32),
  ('max_sge', ctypes.c_uint32),
  ('wqn', ctypes.c_uint32),
]
class struct_ib_uverbs_ex_destroy_wq(Struct): pass
struct_ib_uverbs_ex_destroy_wq._fields_ = [
  ('comp_mask', ctypes.c_uint32),
  ('wq_handle', ctypes.c_uint32),
]
class struct_ib_uverbs_ex_destroy_wq_resp(Struct): pass
struct_ib_uverbs_ex_destroy_wq_resp._fields_ = [
  ('comp_mask', ctypes.c_uint32),
  ('response_length', ctypes.c_uint32),
  ('events_reported', ctypes.c_uint32),
  ('reserved', ctypes.c_uint32),
]
class struct_ib_uverbs_ex_modify_wq(Struct): pass
struct_ib_uverbs_ex_modify_wq._fields_ = [
  ('attr_mask', ctypes.c_uint32),
  ('wq_handle', ctypes.c_uint32),
  ('wq_state', ctypes.c_uint32),
  ('curr_wq_state', ctypes.c_uint32),
  ('flags', ctypes.c_uint32),
  ('flags_mask', ctypes.c_uint32),
]
class struct_ib_uverbs_ex_create_rwq_ind_table(Struct): pass
struct_ib_uverbs_ex_create_rwq_ind_table._fields_ = [
  ('comp_mask', ctypes.c_uint32),
  ('log_ind_tbl_size', ctypes.c_uint32),
  ('wq_handles', (ctypes.c_uint32 * 0)),
]
class struct_ib_uverbs_ex_create_rwq_ind_table_resp(Struct): pass
struct_ib_uverbs_ex_create_rwq_ind_table_resp._fields_ = [
  ('comp_mask', ctypes.c_uint32),
  ('response_length', ctypes.c_uint32),
  ('ind_tbl_handle', ctypes.c_uint32),
  ('ind_tbl_num', ctypes.c_uint32),
]
class struct_ib_uverbs_ex_destroy_rwq_ind_table(Struct): pass
struct_ib_uverbs_ex_destroy_rwq_ind_table._fields_ = [
  ('comp_mask', ctypes.c_uint32),
  ('ind_tbl_handle', ctypes.c_uint32),
]
class struct_ib_uverbs_cq_moderation(Struct): pass
struct_ib_uverbs_cq_moderation._fields_ = [
  ('cq_count', ctypes.c_uint16),
  ('cq_period', ctypes.c_uint16),
]
class struct_ib_uverbs_ex_modify_cq(Struct): pass
struct_ib_uverbs_ex_modify_cq._fields_ = [
  ('cq_handle', ctypes.c_uint32),
  ('attr_mask', ctypes.c_uint32),
  ('attr', struct_ib_uverbs_cq_moderation),
  ('reserved', ctypes.c_uint32),
]
enum_ib_uverbs_device_cap_flags = CEnum(ctypes.c_uint64)
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

enum_ib_uverbs_raw_packet_caps = CEnum(ctypes.c_uint32)
IB_UVERBS_RAW_PACKET_CAP_CVLAN_STRIPPING = enum_ib_uverbs_raw_packet_caps.define('IB_UVERBS_RAW_PACKET_CAP_CVLAN_STRIPPING', 1)
IB_UVERBS_RAW_PACKET_CAP_SCATTER_FCS = enum_ib_uverbs_raw_packet_caps.define('IB_UVERBS_RAW_PACKET_CAP_SCATTER_FCS', 2)
IB_UVERBS_RAW_PACKET_CAP_IP_CSUM = enum_ib_uverbs_raw_packet_caps.define('IB_UVERBS_RAW_PACKET_CAP_IP_CSUM', 4)
IB_UVERBS_RAW_PACKET_CAP_DELAY_DROP = enum_ib_uverbs_raw_packet_caps.define('IB_UVERBS_RAW_PACKET_CAP_DELAY_DROP', 8)

vext_field_avail = lambda type,fld,sz: (offsetof(type, fld) < (sz))
IBV_DEVICE_RAW_SCATTER_FCS = (1 << 34)
IBV_DEVICE_PCI_WRITE_END_PADDING = (1 << 36)
ibv_query_port = lambda context,port_num,port_attr: ___ibv_query_port(context, port_num, port_attr)
ibv_reg_mr = lambda pd,addr,length,access: __ibv_reg_mr(pd, addr, length, access, __builtin_constant_p( ((int)(access) & IBV_ACCESS_OPTIONAL_RANGE) == 0))
ibv_reg_mr_iova = lambda pd,addr,length,iova,access: __ibv_reg_mr_iova(pd, addr, length, iova, access, __builtin_constant_p( ((access) & IBV_ACCESS_OPTIONAL_RANGE) == 0))
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