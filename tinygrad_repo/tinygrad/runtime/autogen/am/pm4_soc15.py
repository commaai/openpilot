# mypy: ignore-errors
import ctypes
from tinygrad.runtime.support.c import DLL, Struct, CEnum, _IO, _IOW, _IOR, _IOWR
class union_PM4_MES_TYPE_3_HEADER(ctypes.Union): pass
enum_mes_set_resources_queue_type_enum = CEnum(ctypes.c_uint32)
queue_type__mes_set_resources__kernel_interface_queue_kiq = enum_mes_set_resources_queue_type_enum.define('queue_type__mes_set_resources__kernel_interface_queue_kiq', 0)
queue_type__mes_set_resources__hsa_interface_queue_hiq = enum_mes_set_resources_queue_type_enum.define('queue_type__mes_set_resources__hsa_interface_queue_hiq', 1)
queue_type__mes_set_resources__hsa_debug_interface_queue = enum_mes_set_resources_queue_type_enum.define('queue_type__mes_set_resources__hsa_debug_interface_queue', 4)

class struct_pm4_mes_set_resources(Struct): pass
class struct_pm4_mes_runlist(Struct): pass
class struct_pm4_mes_map_process(Struct): pass
class struct_PM4_MES_MAP_PROCESS_VM(Struct): pass
enum_mes_map_queues_queue_sel_enum = CEnum(ctypes.c_uint32)
queue_sel__mes_map_queues__map_to_specified_queue_slots_vi = enum_mes_map_queues_queue_sel_enum.define('queue_sel__mes_map_queues__map_to_specified_queue_slots_vi', 0)
queue_sel__mes_map_queues__map_to_hws_determined_queue_slots_vi = enum_mes_map_queues_queue_sel_enum.define('queue_sel__mes_map_queues__map_to_hws_determined_queue_slots_vi', 1)

enum_mes_map_queues_queue_type_enum = CEnum(ctypes.c_uint32)
queue_type__mes_map_queues__normal_compute_vi = enum_mes_map_queues_queue_type_enum.define('queue_type__mes_map_queues__normal_compute_vi', 0)
queue_type__mes_map_queues__debug_interface_queue_vi = enum_mes_map_queues_queue_type_enum.define('queue_type__mes_map_queues__debug_interface_queue_vi', 1)
queue_type__mes_map_queues__normal_latency_static_queue_vi = enum_mes_map_queues_queue_type_enum.define('queue_type__mes_map_queues__normal_latency_static_queue_vi', 2)
queue_type__mes_map_queues__low_latency_static_queue_vi = enum_mes_map_queues_queue_type_enum.define('queue_type__mes_map_queues__low_latency_static_queue_vi', 3)

enum_mes_map_queues_engine_sel_enum = CEnum(ctypes.c_uint32)
engine_sel__mes_map_queues__compute_vi = enum_mes_map_queues_engine_sel_enum.define('engine_sel__mes_map_queues__compute_vi', 0)
engine_sel__mes_map_queues__sdma0_vi = enum_mes_map_queues_engine_sel_enum.define('engine_sel__mes_map_queues__sdma0_vi', 2)
engine_sel__mes_map_queues__sdma1_vi = enum_mes_map_queues_engine_sel_enum.define('engine_sel__mes_map_queues__sdma1_vi', 3)

enum_mes_map_queues_extended_engine_sel_enum = CEnum(ctypes.c_uint32)
extended_engine_sel__mes_map_queues__legacy_engine_sel = enum_mes_map_queues_extended_engine_sel_enum.define('extended_engine_sel__mes_map_queues__legacy_engine_sel', 0)
extended_engine_sel__mes_map_queues__sdma0_to_7_sel = enum_mes_map_queues_extended_engine_sel_enum.define('extended_engine_sel__mes_map_queues__sdma0_to_7_sel', 1)
extended_engine_sel__mes_map_queues__sdma8_to_15_sel = enum_mes_map_queues_extended_engine_sel_enum.define('extended_engine_sel__mes_map_queues__sdma8_to_15_sel', 2)

class struct_pm4_mes_map_queues(Struct): pass
enum_mes_query_status_interrupt_sel_enum = CEnum(ctypes.c_uint32)
interrupt_sel__mes_query_status__completion_status = enum_mes_query_status_interrupt_sel_enum.define('interrupt_sel__mes_query_status__completion_status', 0)
interrupt_sel__mes_query_status__process_status = enum_mes_query_status_interrupt_sel_enum.define('interrupt_sel__mes_query_status__process_status', 1)
interrupt_sel__mes_query_status__queue_status = enum_mes_query_status_interrupt_sel_enum.define('interrupt_sel__mes_query_status__queue_status', 2)

enum_mes_query_status_command_enum = CEnum(ctypes.c_uint32)
command__mes_query_status__interrupt_only = enum_mes_query_status_command_enum.define('command__mes_query_status__interrupt_only', 0)
command__mes_query_status__fence_only_immediate = enum_mes_query_status_command_enum.define('command__mes_query_status__fence_only_immediate', 1)
command__mes_query_status__fence_only_after_write_ack = enum_mes_query_status_command_enum.define('command__mes_query_status__fence_only_after_write_ack', 2)
command__mes_query_status__fence_wait_for_write_ack_send_interrupt = enum_mes_query_status_command_enum.define('command__mes_query_status__fence_wait_for_write_ack_send_interrupt', 3)

enum_mes_query_status_engine_sel_enum = CEnum(ctypes.c_uint32)
engine_sel__mes_query_status__compute = enum_mes_query_status_engine_sel_enum.define('engine_sel__mes_query_status__compute', 0)
engine_sel__mes_query_status__sdma0_queue = enum_mes_query_status_engine_sel_enum.define('engine_sel__mes_query_status__sdma0_queue', 2)
engine_sel__mes_query_status__sdma1_queue = enum_mes_query_status_engine_sel_enum.define('engine_sel__mes_query_status__sdma1_queue', 3)

class struct_pm4_mes_query_status(Struct): pass
enum_mes_unmap_queues_action_enum = CEnum(ctypes.c_uint32)
action__mes_unmap_queues__preempt_queues = enum_mes_unmap_queues_action_enum.define('action__mes_unmap_queues__preempt_queues', 0)
action__mes_unmap_queues__reset_queues = enum_mes_unmap_queues_action_enum.define('action__mes_unmap_queues__reset_queues', 1)
action__mes_unmap_queues__disable_process_queues = enum_mes_unmap_queues_action_enum.define('action__mes_unmap_queues__disable_process_queues', 2)
action__mes_unmap_queues__reserved = enum_mes_unmap_queues_action_enum.define('action__mes_unmap_queues__reserved', 3)

enum_mes_unmap_queues_queue_sel_enum = CEnum(ctypes.c_uint32)
queue_sel__mes_unmap_queues__perform_request_on_specified_queues = enum_mes_unmap_queues_queue_sel_enum.define('queue_sel__mes_unmap_queues__perform_request_on_specified_queues', 0)
queue_sel__mes_unmap_queues__perform_request_on_pasid_queues = enum_mes_unmap_queues_queue_sel_enum.define('queue_sel__mes_unmap_queues__perform_request_on_pasid_queues', 1)
queue_sel__mes_unmap_queues__unmap_all_queues = enum_mes_unmap_queues_queue_sel_enum.define('queue_sel__mes_unmap_queues__unmap_all_queues', 2)
queue_sel__mes_unmap_queues__unmap_all_non_static_queues = enum_mes_unmap_queues_queue_sel_enum.define('queue_sel__mes_unmap_queues__unmap_all_non_static_queues', 3)

enum_mes_unmap_queues_engine_sel_enum = CEnum(ctypes.c_uint32)
engine_sel__mes_unmap_queues__compute = enum_mes_unmap_queues_engine_sel_enum.define('engine_sel__mes_unmap_queues__compute', 0)
engine_sel__mes_unmap_queues__sdma0 = enum_mes_unmap_queues_engine_sel_enum.define('engine_sel__mes_unmap_queues__sdma0', 2)
engine_sel__mes_unmap_queues__sdmal = enum_mes_unmap_queues_engine_sel_enum.define('engine_sel__mes_unmap_queues__sdmal', 3)

enum_mes_unmap_queues_extended_engine_sel_enum = CEnum(ctypes.c_uint32)
extended_engine_sel__mes_unmap_queues__legacy_engine_sel = enum_mes_unmap_queues_extended_engine_sel_enum.define('extended_engine_sel__mes_unmap_queues__legacy_engine_sel', 0)
extended_engine_sel__mes_unmap_queues__sdma0_to_7_sel = enum_mes_unmap_queues_extended_engine_sel_enum.define('extended_engine_sel__mes_unmap_queues__sdma0_to_7_sel', 1)

class struct_pm4_mes_unmap_queues(Struct): pass
enum_mec_release_mem_event_index_enum = CEnum(ctypes.c_uint32)
event_index__mec_release_mem__end_of_pipe = enum_mec_release_mem_event_index_enum.define('event_index__mec_release_mem__end_of_pipe', 5)
event_index__mec_release_mem__shader_done = enum_mec_release_mem_event_index_enum.define('event_index__mec_release_mem__shader_done', 6)

enum_mec_release_mem_cache_policy_enum = CEnum(ctypes.c_uint32)
cache_policy__mec_release_mem__lru = enum_mec_release_mem_cache_policy_enum.define('cache_policy__mec_release_mem__lru', 0)
cache_policy__mec_release_mem__stream = enum_mec_release_mem_cache_policy_enum.define('cache_policy__mec_release_mem__stream', 1)

enum_mec_release_mem_pq_exe_status_enum = CEnum(ctypes.c_uint32)
pq_exe_status__mec_release_mem__default = enum_mec_release_mem_pq_exe_status_enum.define('pq_exe_status__mec_release_mem__default', 0)
pq_exe_status__mec_release_mem__phase_update = enum_mec_release_mem_pq_exe_status_enum.define('pq_exe_status__mec_release_mem__phase_update', 1)

enum_mec_release_mem_dst_sel_enum = CEnum(ctypes.c_uint32)
dst_sel__mec_release_mem__memory_controller = enum_mec_release_mem_dst_sel_enum.define('dst_sel__mec_release_mem__memory_controller', 0)
dst_sel__mec_release_mem__tc_l2 = enum_mec_release_mem_dst_sel_enum.define('dst_sel__mec_release_mem__tc_l2', 1)
dst_sel__mec_release_mem__queue_write_pointer_register = enum_mec_release_mem_dst_sel_enum.define('dst_sel__mec_release_mem__queue_write_pointer_register', 2)
dst_sel__mec_release_mem__queue_write_pointer_poll_mask_bit = enum_mec_release_mem_dst_sel_enum.define('dst_sel__mec_release_mem__queue_write_pointer_poll_mask_bit', 3)

enum_mec_release_mem_int_sel_enum = CEnum(ctypes.c_uint32)
int_sel__mec_release_mem__none = enum_mec_release_mem_int_sel_enum.define('int_sel__mec_release_mem__none', 0)
int_sel__mec_release_mem__send_interrupt_only = enum_mec_release_mem_int_sel_enum.define('int_sel__mec_release_mem__send_interrupt_only', 1)
int_sel__mec_release_mem__send_interrupt_after_write_confirm = enum_mec_release_mem_int_sel_enum.define('int_sel__mec_release_mem__send_interrupt_after_write_confirm', 2)
int_sel__mec_release_mem__send_data_after_write_confirm = enum_mec_release_mem_int_sel_enum.define('int_sel__mec_release_mem__send_data_after_write_confirm', 3)
int_sel__mec_release_mem__unconditionally_send_int_ctxid = enum_mec_release_mem_int_sel_enum.define('int_sel__mec_release_mem__unconditionally_send_int_ctxid', 4)
int_sel__mec_release_mem__conditionally_send_int_ctxid_based_on_32_bit_compare = enum_mec_release_mem_int_sel_enum.define('int_sel__mec_release_mem__conditionally_send_int_ctxid_based_on_32_bit_compare', 5)
int_sel__mec_release_mem__conditionally_send_int_ctxid_based_on_64_bit_compare = enum_mec_release_mem_int_sel_enum.define('int_sel__mec_release_mem__conditionally_send_int_ctxid_based_on_64_bit_compare', 6)

enum_mec_release_mem_data_sel_enum = CEnum(ctypes.c_uint32)
data_sel__mec_release_mem__none = enum_mec_release_mem_data_sel_enum.define('data_sel__mec_release_mem__none', 0)
data_sel__mec_release_mem__send_32_bit_low = enum_mec_release_mem_data_sel_enum.define('data_sel__mec_release_mem__send_32_bit_low', 1)
data_sel__mec_release_mem__send_64_bit_data = enum_mec_release_mem_data_sel_enum.define('data_sel__mec_release_mem__send_64_bit_data', 2)
data_sel__mec_release_mem__send_gpu_clock_counter = enum_mec_release_mem_data_sel_enum.define('data_sel__mec_release_mem__send_gpu_clock_counter', 3)
data_sel__mec_release_mem__send_cp_perfcounter_hi_lo = enum_mec_release_mem_data_sel_enum.define('data_sel__mec_release_mem__send_cp_perfcounter_hi_lo', 4)
data_sel__mec_release_mem__store_gds_data_to_memory = enum_mec_release_mem_data_sel_enum.define('data_sel__mec_release_mem__store_gds_data_to_memory', 5)

class struct_pm4_mec_release_mem(Struct): pass
enum_WRITE_DATA_dst_sel_enum = CEnum(ctypes.c_uint32)
dst_sel___write_data__mem_mapped_register = enum_WRITE_DATA_dst_sel_enum.define('dst_sel___write_data__mem_mapped_register', 0)
dst_sel___write_data__tc_l2 = enum_WRITE_DATA_dst_sel_enum.define('dst_sel___write_data__tc_l2', 2)
dst_sel___write_data__gds = enum_WRITE_DATA_dst_sel_enum.define('dst_sel___write_data__gds', 3)
dst_sel___write_data__memory = enum_WRITE_DATA_dst_sel_enum.define('dst_sel___write_data__memory', 5)
dst_sel___write_data__memory_mapped_adc_persistent_state = enum_WRITE_DATA_dst_sel_enum.define('dst_sel___write_data__memory_mapped_adc_persistent_state', 6)

enum_WRITE_DATA_addr_incr_enum = CEnum(ctypes.c_uint32)
addr_incr___write_data__increment_address = enum_WRITE_DATA_addr_incr_enum.define('addr_incr___write_data__increment_address', 0)
addr_incr___write_data__do_not_increment_address = enum_WRITE_DATA_addr_incr_enum.define('addr_incr___write_data__do_not_increment_address', 1)

enum_WRITE_DATA_wr_confirm_enum = CEnum(ctypes.c_uint32)
wr_confirm___write_data__do_not_wait_for_write_confirmation = enum_WRITE_DATA_wr_confirm_enum.define('wr_confirm___write_data__do_not_wait_for_write_confirmation', 0)
wr_confirm___write_data__wait_for_write_confirmation = enum_WRITE_DATA_wr_confirm_enum.define('wr_confirm___write_data__wait_for_write_confirmation', 1)

enum_WRITE_DATA_cache_policy_enum = CEnum(ctypes.c_uint32)
cache_policy___write_data__lru = enum_WRITE_DATA_cache_policy_enum.define('cache_policy___write_data__lru', 0)
cache_policy___write_data__stream = enum_WRITE_DATA_cache_policy_enum.define('cache_policy___write_data__stream', 1)

class struct_pm4_mec_write_data_mmio(Struct): pass
_anonenum0 = CEnum(ctypes.c_uint32)
CACHE_FLUSH_AND_INV_TS_EVENT = _anonenum0.define('CACHE_FLUSH_AND_INV_TS_EVENT', 20)

GFX9_NUM_GFX_RINGS = 1
GFX9_NUM_COMPUTE_RINGS = 8
PACKET_TYPE0 = 0
PACKET_TYPE1 = 1
PACKET_TYPE2 = 2
PACKET_TYPE3 = 3
CP_PACKET_GET_TYPE = lambda h: (((h) >> 30) & 3)
CP_PACKET_GET_COUNT = lambda h: (((h) >> 16) & 0x3FFF)
CP_PACKET0_GET_REG = lambda h: ((h) & 0xFFFF)
CP_PACKET3_GET_OPCODE = lambda h: (((h) >> 8) & 0xFF)
PACKET0 = lambda reg,n: ((PACKET_TYPE0 << 30) | ((reg) & 0xFFFF) | ((n) & 0x3FFF) << 16)
CP_PACKET2 = 0x80000000
PACKET2_PAD_SHIFT = 0
PACKET2_PAD_MASK = (0x3fffffff << 0)
PACKET2 = lambda v: (CP_PACKET2 | REG_SET(PACKET2_PAD, (v)))
PACKET3 = lambda op,n: ((PACKET_TYPE3 << 30) | (((op) & 0xFF) << 8) | ((n) & 0x3FFF) << 16)
PACKET3_COMPUTE = lambda op,n: (PACKET3(op, n) | 1 << 1)
PACKETJ_CONDITION_CHECK0 = 0
PACKETJ_CONDITION_CHECK1 = 1
PACKETJ_CONDITION_CHECK2 = 2
PACKETJ_CONDITION_CHECK3 = 3
PACKETJ_CONDITION_CHECK4 = 4
PACKETJ_CONDITION_CHECK5 = 5
PACKETJ_CONDITION_CHECK6 = 6
PACKETJ_CONDITION_CHECK7 = 7
PACKETJ_TYPE0 = 0
PACKETJ_TYPE1 = 1
PACKETJ_TYPE2 = 2
PACKETJ_TYPE3 = 3
PACKETJ_TYPE4 = 4
PACKETJ_TYPE5 = 5
PACKETJ_TYPE6 = 6
PACKETJ_TYPE7 = 7
PACKETJ = lambda reg,r,cond,type: ((reg & 0x3FFFF) | ((r & 0x3F) << 18) | ((cond & 0xF) << 24) | ((type & 0xF) << 28))
CP_PACKETJ_NOP = 0x60000000
CP_PACKETJ_GET_REG = lambda x: ((x) & 0x3FFFF)
CP_PACKETJ_GET_RES = lambda x: (((x) >> 18) & 0x3F)
CP_PACKETJ_GET_COND = lambda x: (((x) >> 24) & 0xF)
CP_PACKETJ_GET_TYPE = lambda x: (((x) >> 28) & 0xF)
PACKET3_NOP = 0x10
PACKET3_SET_BASE = 0x11
PACKET3_BASE_INDEX = lambda x: ((x) << 0)
CE_PARTITION_BASE = 3
PACKET3_CLEAR_STATE = 0x12
PACKET3_INDEX_BUFFER_SIZE = 0x13
PACKET3_DISPATCH_DIRECT = 0x15
PACKET3_DISPATCH_INDIRECT = 0x16
PACKET3_ATOMIC_GDS = 0x1D
PACKET3_ATOMIC_MEM = 0x1E
PACKET3_OCCLUSION_QUERY = 0x1F
PACKET3_SET_PREDICATION = 0x20
PACKET3_REG_RMW = 0x21
PACKET3_COND_EXEC = 0x22
PACKET3_PRED_EXEC = 0x23
PACKET3_DRAW_INDIRECT = 0x24
PACKET3_DRAW_INDEX_INDIRECT = 0x25
PACKET3_INDEX_BASE = 0x26
PACKET3_DRAW_INDEX_2 = 0x27
PACKET3_CONTEXT_CONTROL = 0x28
PACKET3_INDEX_TYPE = 0x2A
PACKET3_DRAW_INDIRECT_MULTI = 0x2C
PACKET3_DRAW_INDEX_AUTO = 0x2D
PACKET3_NUM_INSTANCES = 0x2F
PACKET3_DRAW_INDEX_MULTI_AUTO = 0x30
PACKET3_INDIRECT_BUFFER_CONST = 0x33
PACKET3_STRMOUT_BUFFER_UPDATE = 0x34
PACKET3_DRAW_INDEX_OFFSET_2 = 0x35
PACKET3_DRAW_PREAMBLE = 0x36
PACKET3_WRITE_DATA = 0x37
WRITE_DATA_DST_SEL = lambda x: ((x) << 8)
WR_ONE_ADDR = (1 << 16)
WR_CONFIRM = (1 << 20)
WRITE_DATA_CACHE_POLICY = lambda x: ((x) << 25)
WRITE_DATA_ENGINE_SEL = lambda x: ((x) << 30)
PACKET3_DRAW_INDEX_INDIRECT_MULTI = 0x38
PACKET3_MEM_SEMAPHORE = 0x39
PACKET3_SEM_USE_MAILBOX = (0x1 << 16)
PACKET3_SEM_SEL_SIGNAL_TYPE = (0x1 << 20)
PACKET3_SEM_SEL_SIGNAL = (0x6 << 29)
PACKET3_SEM_SEL_WAIT = (0x7 << 29)
PACKET3_WAIT_REG_MEM = 0x3C
WAIT_REG_MEM_FUNCTION = lambda x: ((x) << 0)
WAIT_REG_MEM_MEM_SPACE = lambda x: ((x) << 4)
WAIT_REG_MEM_OPERATION = lambda x: ((x) << 6)
WAIT_REG_MEM_ENGINE = lambda x: ((x) << 8)
PACKET3_INDIRECT_BUFFER = 0x3F
INDIRECT_BUFFER_VALID = (1 << 23)
INDIRECT_BUFFER_CACHE_POLICY = lambda x: ((x) << 28)
INDIRECT_BUFFER_PRE_ENB = lambda x: ((x) << 21)
INDIRECT_BUFFER_PRE_RESUME = lambda x: ((x) << 30)
PACKET3_COPY_DATA = 0x40
PACKET3_PFP_SYNC_ME = 0x42
PACKET3_COND_WRITE = 0x45
PACKET3_EVENT_WRITE = 0x46
EVENT_TYPE = lambda x: ((x) << 0)
EVENT_INDEX = lambda x: ((x) << 8)
PACKET3_RELEASE_MEM = 0x49
EVENT_TYPE = lambda x: ((x) << 0)
EVENT_INDEX = lambda x: ((x) << 8)
EOP_TCL1_VOL_ACTION_EN = (1 << 12)
EOP_TC_VOL_ACTION_EN = (1 << 13)
EOP_TC_WB_ACTION_EN = (1 << 15)
EOP_TCL1_ACTION_EN = (1 << 16)
EOP_TC_ACTION_EN = (1 << 17)
EOP_TC_NC_ACTION_EN = (1 << 19)
EOP_TC_MD_ACTION_EN = (1 << 21)
EOP_EXEC = (1 << 28)
DATA_SEL = lambda x: ((x) << 29)
INT_SEL = lambda x: ((x) << 24)
DST_SEL = lambda x: ((x) << 16)
PACKET3_PREAMBLE_CNTL = 0x4A
PACKET3_PREAMBLE_BEGIN_CLEAR_STATE = (2 << 28)
PACKET3_PREAMBLE_END_CLEAR_STATE = (3 << 28)
PACKET3_DMA_DATA = 0x50
PACKET3_DMA_DATA_ENGINE = lambda x: ((x) << 0)
PACKET3_DMA_DATA_SRC_CACHE_POLICY = lambda x: ((x) << 13)
PACKET3_DMA_DATA_DST_SEL = lambda x: ((x) << 20)
PACKET3_DMA_DATA_DST_CACHE_POLICY = lambda x: ((x) << 25)
PACKET3_DMA_DATA_SRC_SEL = lambda x: ((x) << 29)
PACKET3_DMA_DATA_CP_SYNC = (1 << 31)
PACKET3_DMA_DATA_CMD_SAS = (1 << 26)
PACKET3_DMA_DATA_CMD_DAS = (1 << 27)
PACKET3_DMA_DATA_CMD_SAIC = (1 << 28)
PACKET3_DMA_DATA_CMD_DAIC = (1 << 29)
PACKET3_DMA_DATA_CMD_RAW_WAIT = (1 << 30)
PACKET3_ACQUIRE_MEM = 0x58
PACKET3_ACQUIRE_MEM_CP_COHER_CNTL_TC_NC_ACTION_ENA = lambda x: ((x) << 3)
PACKET3_ACQUIRE_MEM_CP_COHER_CNTL_TC_WC_ACTION_ENA = lambda x: ((x) << 4)
PACKET3_ACQUIRE_MEM_CP_COHER_CNTL_TC_INV_METADATA_ACTION_ENA = lambda x: ((x) << 5)
PACKET3_ACQUIRE_MEM_CP_COHER_CNTL_TCL1_VOL_ACTION_ENA = lambda x: ((x) << 15)
PACKET3_ACQUIRE_MEM_CP_COHER_CNTL_TC_WB_ACTION_ENA = lambda x: ((x) << 18)
PACKET3_ACQUIRE_MEM_CP_COHER_CNTL_TCL1_ACTION_ENA = lambda x: ((x) << 22)
PACKET3_ACQUIRE_MEM_CP_COHER_CNTL_TC_ACTION_ENA = lambda x: ((x) << 23)
PACKET3_ACQUIRE_MEM_CP_COHER_CNTL_CB_ACTION_ENA = lambda x: ((x) << 25)
PACKET3_ACQUIRE_MEM_CP_COHER_CNTL_DB_ACTION_ENA = lambda x: ((x) << 26)
PACKET3_ACQUIRE_MEM_CP_COHER_CNTL_SH_KCACHE_ACTION_ENA = lambda x: ((x) << 27)
PACKET3_ACQUIRE_MEM_CP_COHER_CNTL_SH_KCACHE_VOL_ACTION_ENA = lambda x: ((x) << 28)
PACKET3_ACQUIRE_MEM_CP_COHER_CNTL_SH_ICACHE_ACTION_ENA = lambda x: ((x) << 29)
PACKET3_ACQUIRE_MEM_CP_COHER_CNTL_SH_KCACHE_WB_ACTION_ENA = lambda x: ((x) << 30)
PACKET3_REWIND = 0x59
PACKET3_LOAD_UCONFIG_REG = 0x5E
PACKET3_LOAD_SH_REG = 0x5F
PACKET3_LOAD_CONFIG_REG = 0x60
PACKET3_LOAD_CONTEXT_REG = 0x61
PACKET3_SET_CONFIG_REG = 0x68
PACKET3_SET_CONFIG_REG_START = 0x00002000
PACKET3_SET_CONFIG_REG_END = 0x00002c00
PACKET3_SET_CONTEXT_REG = 0x69
PACKET3_SET_CONTEXT_REG_START = 0x0000a000
PACKET3_SET_CONTEXT_REG_END = 0x0000a400
PACKET3_SET_CONTEXT_REG_INDIRECT = 0x73
PACKET3_SET_SH_REG = 0x76
PACKET3_SET_SH_REG_START = 0x00002c00
PACKET3_SET_SH_REG_END = 0x00003000
PACKET3_SET_SH_REG_OFFSET = 0x77
PACKET3_SET_QUEUE_REG = 0x78
PACKET3_SET_UCONFIG_REG = 0x79
PACKET3_SET_UCONFIG_REG_START = 0x0000c000
PACKET3_SET_UCONFIG_REG_END = 0x0000c400
PACKET3_SET_UCONFIG_REG_INDEX_TYPE = (2 << 28)
PACKET3_SCRATCH_RAM_WRITE = 0x7D
PACKET3_SCRATCH_RAM_READ = 0x7E
PACKET3_LOAD_CONST_RAM = 0x80
PACKET3_WRITE_CONST_RAM = 0x81
PACKET3_DUMP_CONST_RAM = 0x83
PACKET3_INCREMENT_CE_COUNTER = 0x84
PACKET3_INCREMENT_DE_COUNTER = 0x85
PACKET3_WAIT_ON_CE_COUNTER = 0x86
PACKET3_WAIT_ON_DE_COUNTER_DIFF = 0x88
PACKET3_SWITCH_BUFFER = 0x8B
PACKET3_FRAME_CONTROL = 0x90
FRAME_TMZ = (1 << 0)
FRAME_CMD = lambda x: ((x) << 28)
PACKET3_INVALIDATE_TLBS = 0x98
PACKET3_INVALIDATE_TLBS_DST_SEL = lambda x: ((x) << 0)
PACKET3_INVALIDATE_TLBS_ALL_HUB = lambda x: ((x) << 4)
PACKET3_INVALIDATE_TLBS_PASID = lambda x: ((x) << 5)
PACKET3_INVALIDATE_TLBS_FLUSH_TYPE = lambda x: ((x) << 29)
PACKET3_SET_RESOURCES = 0xA0
PACKET3_SET_RESOURCES_VMID_MASK = lambda x: ((x) << 0)
PACKET3_SET_RESOURCES_UNMAP_LATENTY = lambda x: ((x) << 16)
PACKET3_SET_RESOURCES_QUEUE_TYPE = lambda x: ((x) << 29)
PACKET3_MAP_QUEUES = 0xA2
PACKET3_MAP_QUEUES_QUEUE_SEL = lambda x: ((x) << 4)
PACKET3_MAP_QUEUES_VMID = lambda x: ((x) << 8)
PACKET3_MAP_QUEUES_QUEUE = lambda x: ((x) << 13)
PACKET3_MAP_QUEUES_PIPE = lambda x: ((x) << 16)
PACKET3_MAP_QUEUES_ME = lambda x: ((x) << 18)
PACKET3_MAP_QUEUES_QUEUE_TYPE = lambda x: ((x) << 21)
PACKET3_MAP_QUEUES_ALLOC_FORMAT = lambda x: ((x) << 24)
PACKET3_MAP_QUEUES_ENGINE_SEL = lambda x: ((x) << 26)
PACKET3_MAP_QUEUES_NUM_QUEUES = lambda x: ((x) << 29)
PACKET3_MAP_QUEUES_CHECK_DISABLE = lambda x: ((x) << 1)
PACKET3_MAP_QUEUES_DOORBELL_OFFSET = lambda x: ((x) << 2)
PACKET3_UNMAP_QUEUES = 0xA3
PACKET3_UNMAP_QUEUES_ACTION = lambda x: ((x) << 0)
PACKET3_UNMAP_QUEUES_QUEUE_SEL = lambda x: ((x) << 4)
PACKET3_UNMAP_QUEUES_ENGINE_SEL = lambda x: ((x) << 26)
PACKET3_UNMAP_QUEUES_NUM_QUEUES = lambda x: ((x) << 29)
PACKET3_UNMAP_QUEUES_PASID = lambda x: ((x) << 0)
PACKET3_UNMAP_QUEUES_DOORBELL_OFFSET0 = lambda x: ((x) << 2)
PACKET3_UNMAP_QUEUES_DOORBELL_OFFSET1 = lambda x: ((x) << 2)
PACKET3_UNMAP_QUEUES_RB_WPTR = lambda x: ((x) << 0)
PACKET3_UNMAP_QUEUES_DOORBELL_OFFSET2 = lambda x: ((x) << 2)
PACKET3_UNMAP_QUEUES_DOORBELL_OFFSET3 = lambda x: ((x) << 2)
PACKET3_QUERY_STATUS = 0xA4
PACKET3_QUERY_STATUS_CONTEXT_ID = lambda x: ((x) << 0)
PACKET3_QUERY_STATUS_INTERRUPT_SEL = lambda x: ((x) << 28)
PACKET3_QUERY_STATUS_COMMAND = lambda x: ((x) << 30)
PACKET3_QUERY_STATUS_PASID = lambda x: ((x) << 0)
PACKET3_QUERY_STATUS_DOORBELL_OFFSET = lambda x: ((x) << 2)
PACKET3_QUERY_STATUS_ENG_SEL = lambda x: ((x) << 25)
PACKET3_RUN_CLEANER_SHADER = 0xD2
VCE_CMD_NO_OP = 0x00000000
VCE_CMD_END = 0x00000001
VCE_CMD_IB = 0x00000002
VCE_CMD_FENCE = 0x00000003
VCE_CMD_TRAP = 0x00000004
VCE_CMD_IB_AUTO = 0x00000005
VCE_CMD_SEMAPHORE = 0x00000006
VCE_CMD_IB_VM = 0x00000102
VCE_CMD_WAIT_GE = 0x00000106
VCE_CMD_UPDATE_PTB = 0x00000107
VCE_CMD_FLUSH_TLB = 0x00000108
VCE_CMD_REG_WRITE = 0x00000109
VCE_CMD_REG_WAIT = 0x0000010a
HEVC_ENC_CMD_NO_OP = 0x00000000
HEVC_ENC_CMD_END = 0x00000001
HEVC_ENC_CMD_FENCE = 0x00000003
HEVC_ENC_CMD_TRAP = 0x00000004
HEVC_ENC_CMD_IB_VM = 0x00000102
HEVC_ENC_CMD_REG_WRITE = 0x00000109
HEVC_ENC_CMD_REG_WAIT = 0x0000010a