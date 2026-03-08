# mypy: disable-error-code="empty-body"
from __future__ import annotations
import ctypes
from typing import Annotated, Literal, TypeAlias
from tinygrad.runtime.support.c import _IO, _IOW, _IOR, _IOWR
from tinygrad.runtime.support import c
class union_PM4_MES_TYPE_3_HEADER(ctypes.Union): pass
class enum_mes_set_resources_queue_type_enum(Annotated[int, ctypes.c_uint32], c.Enum): pass
queue_type__mes_set_resources__kernel_interface_queue_kiq = enum_mes_set_resources_queue_type_enum.define('queue_type__mes_set_resources__kernel_interface_queue_kiq', 0)
queue_type__mes_set_resources__hsa_interface_queue_hiq = enum_mes_set_resources_queue_type_enum.define('queue_type__mes_set_resources__hsa_interface_queue_hiq', 1)
queue_type__mes_set_resources__hsa_debug_interface_queue = enum_mes_set_resources_queue_type_enum.define('queue_type__mes_set_resources__hsa_debug_interface_queue', 4)

class struct_pm4_mes_set_resources(ctypes.Structure): pass
class struct_pm4_mes_runlist(ctypes.Structure): pass
class struct_pm4_mes_map_process(ctypes.Structure): pass
class struct_PM4_MES_MAP_PROCESS_VM(ctypes.Structure): pass
class enum_mes_map_queues_queue_sel_enum(Annotated[int, ctypes.c_uint32], c.Enum): pass
queue_sel__mes_map_queues__map_to_specified_queue_slots_vi = enum_mes_map_queues_queue_sel_enum.define('queue_sel__mes_map_queues__map_to_specified_queue_slots_vi', 0)
queue_sel__mes_map_queues__map_to_hws_determined_queue_slots_vi = enum_mes_map_queues_queue_sel_enum.define('queue_sel__mes_map_queues__map_to_hws_determined_queue_slots_vi', 1)

class enum_mes_map_queues_queue_type_enum(Annotated[int, ctypes.c_uint32], c.Enum): pass
queue_type__mes_map_queues__normal_compute_vi = enum_mes_map_queues_queue_type_enum.define('queue_type__mes_map_queues__normal_compute_vi', 0)
queue_type__mes_map_queues__debug_interface_queue_vi = enum_mes_map_queues_queue_type_enum.define('queue_type__mes_map_queues__debug_interface_queue_vi', 1)
queue_type__mes_map_queues__normal_latency_static_queue_vi = enum_mes_map_queues_queue_type_enum.define('queue_type__mes_map_queues__normal_latency_static_queue_vi', 2)
queue_type__mes_map_queues__low_latency_static_queue_vi = enum_mes_map_queues_queue_type_enum.define('queue_type__mes_map_queues__low_latency_static_queue_vi', 3)

class enum_mes_map_queues_engine_sel_enum(Annotated[int, ctypes.c_uint32], c.Enum): pass
engine_sel__mes_map_queues__compute_vi = enum_mes_map_queues_engine_sel_enum.define('engine_sel__mes_map_queues__compute_vi', 0)
engine_sel__mes_map_queues__sdma0_vi = enum_mes_map_queues_engine_sel_enum.define('engine_sel__mes_map_queues__sdma0_vi', 2)
engine_sel__mes_map_queues__sdma1_vi = enum_mes_map_queues_engine_sel_enum.define('engine_sel__mes_map_queues__sdma1_vi', 3)

class enum_mes_map_queues_extended_engine_sel_enum(Annotated[int, ctypes.c_uint32], c.Enum): pass
extended_engine_sel__mes_map_queues__legacy_engine_sel = enum_mes_map_queues_extended_engine_sel_enum.define('extended_engine_sel__mes_map_queues__legacy_engine_sel', 0)
extended_engine_sel__mes_map_queues__sdma0_to_7_sel = enum_mes_map_queues_extended_engine_sel_enum.define('extended_engine_sel__mes_map_queues__sdma0_to_7_sel', 1)
extended_engine_sel__mes_map_queues__sdma8_to_15_sel = enum_mes_map_queues_extended_engine_sel_enum.define('extended_engine_sel__mes_map_queues__sdma8_to_15_sel', 2)

class struct_pm4_mes_map_queues(ctypes.Structure): pass
class enum_mes_query_status_interrupt_sel_enum(Annotated[int, ctypes.c_uint32], c.Enum): pass
interrupt_sel__mes_query_status__completion_status = enum_mes_query_status_interrupt_sel_enum.define('interrupt_sel__mes_query_status__completion_status', 0)
interrupt_sel__mes_query_status__process_status = enum_mes_query_status_interrupt_sel_enum.define('interrupt_sel__mes_query_status__process_status', 1)
interrupt_sel__mes_query_status__queue_status = enum_mes_query_status_interrupt_sel_enum.define('interrupt_sel__mes_query_status__queue_status', 2)

class enum_mes_query_status_command_enum(Annotated[int, ctypes.c_uint32], c.Enum): pass
command__mes_query_status__interrupt_only = enum_mes_query_status_command_enum.define('command__mes_query_status__interrupt_only', 0)
command__mes_query_status__fence_only_immediate = enum_mes_query_status_command_enum.define('command__mes_query_status__fence_only_immediate', 1)
command__mes_query_status__fence_only_after_write_ack = enum_mes_query_status_command_enum.define('command__mes_query_status__fence_only_after_write_ack', 2)
command__mes_query_status__fence_wait_for_write_ack_send_interrupt = enum_mes_query_status_command_enum.define('command__mes_query_status__fence_wait_for_write_ack_send_interrupt', 3)

class enum_mes_query_status_engine_sel_enum(Annotated[int, ctypes.c_uint32], c.Enum): pass
engine_sel__mes_query_status__compute = enum_mes_query_status_engine_sel_enum.define('engine_sel__mes_query_status__compute', 0)
engine_sel__mes_query_status__sdma0_queue = enum_mes_query_status_engine_sel_enum.define('engine_sel__mes_query_status__sdma0_queue', 2)
engine_sel__mes_query_status__sdma1_queue = enum_mes_query_status_engine_sel_enum.define('engine_sel__mes_query_status__sdma1_queue', 3)

class struct_pm4_mes_query_status(ctypes.Structure): pass
class enum_mes_unmap_queues_action_enum(Annotated[int, ctypes.c_uint32], c.Enum): pass
action__mes_unmap_queues__preempt_queues = enum_mes_unmap_queues_action_enum.define('action__mes_unmap_queues__preempt_queues', 0)
action__mes_unmap_queues__reset_queues = enum_mes_unmap_queues_action_enum.define('action__mes_unmap_queues__reset_queues', 1)
action__mes_unmap_queues__disable_process_queues = enum_mes_unmap_queues_action_enum.define('action__mes_unmap_queues__disable_process_queues', 2)
action__mes_unmap_queues__reserved = enum_mes_unmap_queues_action_enum.define('action__mes_unmap_queues__reserved', 3)

class enum_mes_unmap_queues_queue_sel_enum(Annotated[int, ctypes.c_uint32], c.Enum): pass
queue_sel__mes_unmap_queues__perform_request_on_specified_queues = enum_mes_unmap_queues_queue_sel_enum.define('queue_sel__mes_unmap_queues__perform_request_on_specified_queues', 0)
queue_sel__mes_unmap_queues__perform_request_on_pasid_queues = enum_mes_unmap_queues_queue_sel_enum.define('queue_sel__mes_unmap_queues__perform_request_on_pasid_queues', 1)
queue_sel__mes_unmap_queues__unmap_all_queues = enum_mes_unmap_queues_queue_sel_enum.define('queue_sel__mes_unmap_queues__unmap_all_queues', 2)
queue_sel__mes_unmap_queues__unmap_all_non_static_queues = enum_mes_unmap_queues_queue_sel_enum.define('queue_sel__mes_unmap_queues__unmap_all_non_static_queues', 3)

class enum_mes_unmap_queues_engine_sel_enum(Annotated[int, ctypes.c_uint32], c.Enum): pass
engine_sel__mes_unmap_queues__compute = enum_mes_unmap_queues_engine_sel_enum.define('engine_sel__mes_unmap_queues__compute', 0)
engine_sel__mes_unmap_queues__sdma0 = enum_mes_unmap_queues_engine_sel_enum.define('engine_sel__mes_unmap_queues__sdma0', 2)
engine_sel__mes_unmap_queues__sdmal = enum_mes_unmap_queues_engine_sel_enum.define('engine_sel__mes_unmap_queues__sdmal', 3)

class enum_mes_unmap_queues_extended_engine_sel_enum(Annotated[int, ctypes.c_uint32], c.Enum): pass
extended_engine_sel__mes_unmap_queues__legacy_engine_sel = enum_mes_unmap_queues_extended_engine_sel_enum.define('extended_engine_sel__mes_unmap_queues__legacy_engine_sel', 0)
extended_engine_sel__mes_unmap_queues__sdma0_to_7_sel = enum_mes_unmap_queues_extended_engine_sel_enum.define('extended_engine_sel__mes_unmap_queues__sdma0_to_7_sel', 1)

class struct_pm4_mes_unmap_queues(ctypes.Structure): pass
class enum_mec_release_mem_event_index_enum(Annotated[int, ctypes.c_uint32], c.Enum): pass
event_index__mec_release_mem__end_of_pipe = enum_mec_release_mem_event_index_enum.define('event_index__mec_release_mem__end_of_pipe', 5)
event_index__mec_release_mem__shader_done = enum_mec_release_mem_event_index_enum.define('event_index__mec_release_mem__shader_done', 6)

class enum_mec_release_mem_cache_policy_enum(Annotated[int, ctypes.c_uint32], c.Enum): pass
cache_policy__mec_release_mem__lru = enum_mec_release_mem_cache_policy_enum.define('cache_policy__mec_release_mem__lru', 0)
cache_policy__mec_release_mem__stream = enum_mec_release_mem_cache_policy_enum.define('cache_policy__mec_release_mem__stream', 1)

class enum_mec_release_mem_pq_exe_status_enum(Annotated[int, ctypes.c_uint32], c.Enum): pass
pq_exe_status__mec_release_mem__default = enum_mec_release_mem_pq_exe_status_enum.define('pq_exe_status__mec_release_mem__default', 0)
pq_exe_status__mec_release_mem__phase_update = enum_mec_release_mem_pq_exe_status_enum.define('pq_exe_status__mec_release_mem__phase_update', 1)

class enum_mec_release_mem_dst_sel_enum(Annotated[int, ctypes.c_uint32], c.Enum): pass
dst_sel__mec_release_mem__memory_controller = enum_mec_release_mem_dst_sel_enum.define('dst_sel__mec_release_mem__memory_controller', 0)
dst_sel__mec_release_mem__tc_l2 = enum_mec_release_mem_dst_sel_enum.define('dst_sel__mec_release_mem__tc_l2', 1)
dst_sel__mec_release_mem__queue_write_pointer_register = enum_mec_release_mem_dst_sel_enum.define('dst_sel__mec_release_mem__queue_write_pointer_register', 2)
dst_sel__mec_release_mem__queue_write_pointer_poll_mask_bit = enum_mec_release_mem_dst_sel_enum.define('dst_sel__mec_release_mem__queue_write_pointer_poll_mask_bit', 3)

class enum_mec_release_mem_int_sel_enum(Annotated[int, ctypes.c_uint32], c.Enum): pass
int_sel__mec_release_mem__none = enum_mec_release_mem_int_sel_enum.define('int_sel__mec_release_mem__none', 0)
int_sel__mec_release_mem__send_interrupt_only = enum_mec_release_mem_int_sel_enum.define('int_sel__mec_release_mem__send_interrupt_only', 1)
int_sel__mec_release_mem__send_interrupt_after_write_confirm = enum_mec_release_mem_int_sel_enum.define('int_sel__mec_release_mem__send_interrupt_after_write_confirm', 2)
int_sel__mec_release_mem__send_data_after_write_confirm = enum_mec_release_mem_int_sel_enum.define('int_sel__mec_release_mem__send_data_after_write_confirm', 3)
int_sel__mec_release_mem__unconditionally_send_int_ctxid = enum_mec_release_mem_int_sel_enum.define('int_sel__mec_release_mem__unconditionally_send_int_ctxid', 4)
int_sel__mec_release_mem__conditionally_send_int_ctxid_based_on_32_bit_compare = enum_mec_release_mem_int_sel_enum.define('int_sel__mec_release_mem__conditionally_send_int_ctxid_based_on_32_bit_compare', 5)
int_sel__mec_release_mem__conditionally_send_int_ctxid_based_on_64_bit_compare = enum_mec_release_mem_int_sel_enum.define('int_sel__mec_release_mem__conditionally_send_int_ctxid_based_on_64_bit_compare', 6)

class enum_mec_release_mem_data_sel_enum(Annotated[int, ctypes.c_uint32], c.Enum): pass
data_sel__mec_release_mem__none = enum_mec_release_mem_data_sel_enum.define('data_sel__mec_release_mem__none', 0)
data_sel__mec_release_mem__send_32_bit_low = enum_mec_release_mem_data_sel_enum.define('data_sel__mec_release_mem__send_32_bit_low', 1)
data_sel__mec_release_mem__send_64_bit_data = enum_mec_release_mem_data_sel_enum.define('data_sel__mec_release_mem__send_64_bit_data', 2)
data_sel__mec_release_mem__send_gpu_clock_counter = enum_mec_release_mem_data_sel_enum.define('data_sel__mec_release_mem__send_gpu_clock_counter', 3)
data_sel__mec_release_mem__send_cp_perfcounter_hi_lo = enum_mec_release_mem_data_sel_enum.define('data_sel__mec_release_mem__send_cp_perfcounter_hi_lo', 4)
data_sel__mec_release_mem__store_gds_data_to_memory = enum_mec_release_mem_data_sel_enum.define('data_sel__mec_release_mem__store_gds_data_to_memory', 5)

class struct_pm4_mec_release_mem(ctypes.Structure): pass
class enum_WRITE_DATA_dst_sel_enum(Annotated[int, ctypes.c_uint32], c.Enum): pass
dst_sel___write_data__mem_mapped_register = enum_WRITE_DATA_dst_sel_enum.define('dst_sel___write_data__mem_mapped_register', 0)
dst_sel___write_data__tc_l2 = enum_WRITE_DATA_dst_sel_enum.define('dst_sel___write_data__tc_l2', 2)
dst_sel___write_data__gds = enum_WRITE_DATA_dst_sel_enum.define('dst_sel___write_data__gds', 3)
dst_sel___write_data__memory = enum_WRITE_DATA_dst_sel_enum.define('dst_sel___write_data__memory', 5)
dst_sel___write_data__memory_mapped_adc_persistent_state = enum_WRITE_DATA_dst_sel_enum.define('dst_sel___write_data__memory_mapped_adc_persistent_state', 6)

class enum_WRITE_DATA_addr_incr_enum(Annotated[int, ctypes.c_uint32], c.Enum): pass
addr_incr___write_data__increment_address = enum_WRITE_DATA_addr_incr_enum.define('addr_incr___write_data__increment_address', 0)
addr_incr___write_data__do_not_increment_address = enum_WRITE_DATA_addr_incr_enum.define('addr_incr___write_data__do_not_increment_address', 1)

class enum_WRITE_DATA_wr_confirm_enum(Annotated[int, ctypes.c_uint32], c.Enum): pass
wr_confirm___write_data__do_not_wait_for_write_confirmation = enum_WRITE_DATA_wr_confirm_enum.define('wr_confirm___write_data__do_not_wait_for_write_confirmation', 0)
wr_confirm___write_data__wait_for_write_confirmation = enum_WRITE_DATA_wr_confirm_enum.define('wr_confirm___write_data__wait_for_write_confirmation', 1)

class enum_WRITE_DATA_cache_policy_enum(Annotated[int, ctypes.c_uint32], c.Enum): pass
cache_policy___write_data__lru = enum_WRITE_DATA_cache_policy_enum.define('cache_policy___write_data__lru', 0)
cache_policy___write_data__stream = enum_WRITE_DATA_cache_policy_enum.define('cache_policy___write_data__stream', 1)

class struct_pm4_mec_write_data_mmio(ctypes.Structure): pass
class _anonenum0(Annotated[int, ctypes.c_uint32], c.Enum): pass
CACHE_FLUSH_AND_INV_TS_EVENT = _anonenum0.define('CACHE_FLUSH_AND_INV_TS_EVENT', 20)

c.init_records()
PACKET_TYPE0 = 0 # type: ignore
PACKET_TYPE1 = 1 # type: ignore
PACKET_TYPE2 = 2 # type: ignore
PACKET_TYPE3 = 3 # type: ignore
CP_PACKET_GET_TYPE = lambda h: (((h) >> 30) & 3) # type: ignore
CP_PACKET_GET_COUNT = lambda h: (((h) >> 16) & 0x3FFF) # type: ignore
CP_PACKET0_GET_REG = lambda h: ((h) & 0xFFFF) # type: ignore
CP_PACKET3_GET_OPCODE = lambda h: (((h) >> 8) & 0xFF) # type: ignore
PACKET0 = lambda reg,n: ((PACKET_TYPE0 << 30) | ((reg) & 0xFFFF) | ((n) & 0x3FFF) << 16) # type: ignore
CP_PACKET2 = 0x80000000 # type: ignore
PACKET2_PAD_SHIFT = 0 # type: ignore
PACKET2_PAD_MASK = (0x3fffffff << 0) # type: ignore
PACKET2 = lambda v: (CP_PACKET2 | REG_SET(PACKET2_PAD, (v))) # type: ignore
PACKET3 = lambda op,n: ((PACKET_TYPE3 << 30) | (((op) & 0xFF) << 8) | ((n) & 0x3FFF) << 16) # type: ignore
PACKET3_COMPUTE = lambda op,n: (PACKET3(op, n) | 1 << 1) # type: ignore
PACKET3_NOP = 0x10 # type: ignore
PACKET3_SET_BASE = 0x11 # type: ignore
PACKET3_BASE_INDEX = lambda x: ((x) << 0) # type: ignore
CE_PARTITION_BASE = 3 # type: ignore
PACKET3_CLEAR_STATE = 0x12 # type: ignore
PACKET3_INDEX_BUFFER_SIZE = 0x13 # type: ignore
PACKET3_DISPATCH_DIRECT = 0x15 # type: ignore
PACKET3_DISPATCH_INDIRECT = 0x16 # type: ignore
PACKET3_INDIRECT_BUFFER_END = 0x17 # type: ignore
PACKET3_INDIRECT_BUFFER_CNST_END = 0x19 # type: ignore
PACKET3_ATOMIC_GDS = 0x1D # type: ignore
PACKET3_ATOMIC_MEM = 0x1E # type: ignore
PACKET3_ATOMIC_MEM__ATOMIC = lambda x: ((((unsigned)(x)) & 0x7F) << 0) # type: ignore
PACKET3_ATOMIC_MEM__COMMAND = lambda x: ((((unsigned)(x)) & 0xF) << 8) # type: ignore
PACKET3_ATOMIC_MEM__CACHE_POLICY = lambda x: ((((unsigned)(x)) & 0x3) << 25) # type: ignore
PACKET3_ATOMIC_MEM__ADDR_LO = lambda x: (((unsigned)(x))) # type: ignore
PACKET3_ATOMIC_MEM__ADDR_HI = lambda x: (((unsigned)(x))) # type: ignore
PACKET3_ATOMIC_MEM__SRC_DATA_LO = lambda x: (((unsigned)(x))) # type: ignore
PACKET3_ATOMIC_MEM__SRC_DATA_HI = lambda x: (((unsigned)(x))) # type: ignore
PACKET3_ATOMIC_MEM__CMP_DATA_LO = lambda x: (((unsigned)(x))) # type: ignore
PACKET3_ATOMIC_MEM__CMP_DATA_HI = lambda x: (((unsigned)(x))) # type: ignore
PACKET3_ATOMIC_MEM__LOOP_INTERVAL = lambda x: ((((unsigned)(x)) & 0x1FFF) << 0) # type: ignore
PACKET3_ATOMIC_MEM__COMMAND__SINGLE_PASS_ATOMIC = 0 # type: ignore
PACKET3_ATOMIC_MEM__COMMAND__LOOP_UNTIL_COMPARE_SATISFIED = 1 # type: ignore
PACKET3_ATOMIC_MEM__COMMAND__WAIT_FOR_WRITE_CONFIRMATION = 2 # type: ignore
PACKET3_ATOMIC_MEM__COMMAND__SEND_AND_CONTINUE = 3 # type: ignore
PACKET3_ATOMIC_MEM__CACHE_POLICY__LRU = 0 # type: ignore
PACKET3_ATOMIC_MEM__CACHE_POLICY__STREAM = 1 # type: ignore
PACKET3_ATOMIC_MEM__CACHE_POLICY__NOA = 2 # type: ignore
PACKET3_ATOMIC_MEM__CACHE_POLICY__BYPASS = 3 # type: ignore
PACKET3_OCCLUSION_QUERY = 0x1F # type: ignore
PACKET3_SET_PREDICATION = 0x20 # type: ignore
PACKET3_REG_RMW = 0x21 # type: ignore
PACKET3_COND_EXEC = 0x22 # type: ignore
PACKET3_PRED_EXEC = 0x23 # type: ignore
PACKET3_DRAW_INDIRECT = 0x24 # type: ignore
PACKET3_DRAW_INDEX_INDIRECT = 0x25 # type: ignore
PACKET3_INDEX_BASE = 0x26 # type: ignore
PACKET3_DRAW_INDEX_2 = 0x27 # type: ignore
PACKET3_CONTEXT_CONTROL = 0x28 # type: ignore
PACKET3_INDEX_TYPE = 0x2A # type: ignore
PACKET3_DRAW_INDIRECT_MULTI = 0x2C # type: ignore
PACKET3_DRAW_INDEX_AUTO = 0x2D # type: ignore
PACKET3_NUM_INSTANCES = 0x2F # type: ignore
PACKET3_DRAW_INDEX_MULTI_AUTO = 0x30 # type: ignore
PACKET3_INDIRECT_BUFFER_PRIV = 0x32 # type: ignore
PACKET3_INDIRECT_BUFFER_CNST = 0x33 # type: ignore
PACKET3_COND_INDIRECT_BUFFER_CNST = 0x33 # type: ignore
PACKET3_STRMOUT_BUFFER_UPDATE = 0x34 # type: ignore
PACKET3_DRAW_INDEX_OFFSET_2 = 0x35 # type: ignore
PACKET3_DRAW_PREAMBLE = 0x36 # type: ignore
PACKET3_WRITE_DATA = 0x37 # type: ignore
WRITE_DATA_DST_SEL = lambda x: ((x) << 8) # type: ignore
WR_ONE_ADDR = (1 << 16) # type: ignore
WR_CONFIRM = (1 << 20) # type: ignore
WRITE_DATA_CACHE_POLICY = lambda x: ((x) << 25) # type: ignore
WRITE_DATA_ENGINE_SEL = lambda x: ((x) << 30) # type: ignore
PACKET3_WRITE_DATA__DST_SEL = lambda x: ((((unsigned)(x)) & 0xF) << 8) # type: ignore
PACKET3_WRITE_DATA__ADDR_INCR = lambda x: ((((unsigned)(x)) & 0x1) << 16) # type: ignore
PACKET3_WRITE_DATA__WR_CONFIRM = lambda x: ((((unsigned)(x)) & 0x1) << 20) # type: ignore
PACKET3_WRITE_DATA__CACHE_POLICY = lambda x: ((((unsigned)(x)) & 0x3) << 25) # type: ignore
PACKET3_WRITE_DATA__DST_MMREG_ADDR = lambda x: ((((unsigned)(x)) & 0x3FFFF) << 0) # type: ignore
PACKET3_WRITE_DATA__DST_GDS_ADDR = lambda x: ((((unsigned)(x)) & 0xFFFF) << 0) # type: ignore
PACKET3_WRITE_DATA__DST_MEM_ADDR_LO = lambda x: ((((unsigned)(x)) & 0x3FFFFFFF) << 2) # type: ignore
PACKET3_WRITE_DATA__DST_MEM_ADDR_HI = lambda x: ((unsigned)(x)) # type: ignore
PACKET3_WRITE_DATA__MODE = lambda x: ((((unsigned)(x)) & 0x1) << 21) # type: ignore
PACKET3_WRITE_DATA__AID_ID = lambda x: ((((unsigned)(x)) & 0x3) << 22) # type: ignore
PACKET3_WRITE_DATA__TEMPORAL = lambda x: ((((unsigned)(x)) & 0x3) << 24) # type: ignore
PACKET3_WRITE_DATA__DST_MMREG_ADDR_LO = lambda x: ((unsigned)(x)) # type: ignore
PACKET3_WRITE_DATA__DST_MMREG_ADDR_HI = lambda x: ((((unsigned)(x)) & 0xFF) << 0) # type: ignore
PACKET3_WRITE_DATA__DST_SEL__MEM_MAPPED_REGISTER = 0 # type: ignore
PACKET3_WRITE_DATA__DST_SEL__TC_L2 = 2 # type: ignore
PACKET3_WRITE_DATA__DST_SEL__GDS = 3 # type: ignore
PACKET3_WRITE_DATA__DST_SEL__MEMORY = 5 # type: ignore
PACKET3_WRITE_DATA__DST_SEL__MEMORY_MAPPED_ADC_PERSISTENT_STATE = 6 # type: ignore
PACKET3_WRITE_DATA__ADDR_INCR__INCREMENT_ADDRESS = 0 # type: ignore
PACKET3_WRITE_DATA__ADDR_INCR__DO_NOT_INCREMENT_ADDRESS = 1 # type: ignore
PACKET3_WRITE_DATA__WR_CONFIRM__DO_NOT_WAIT_FOR_WRITE_CONFIRMATION = 0 # type: ignore
PACKET3_WRITE_DATA__WR_CONFIRM__WAIT_FOR_WRITE_CONFIRMATION = 1 # type: ignore
PACKET3_WRITE_DATA__MODE__PF_VF_DISABLED = 0 # type: ignore
PACKET3_WRITE_DATA__MODE__PF_VF_ENABLED = 1 # type: ignore
PACKET3_WRITE_DATA__TEMPORAL__RT = 0 # type: ignore
PACKET3_WRITE_DATA__TEMPORAL__NT = 1 # type: ignore
PACKET3_WRITE_DATA__TEMPORAL__HT = 2 # type: ignore
PACKET3_WRITE_DATA__TEMPORAL__LU = 3 # type: ignore
PACKET3_WRITE_DATA__CACHE_POLICY__LRU = 0 # type: ignore
PACKET3_WRITE_DATA__CACHE_POLICY__STREAM = 1 # type: ignore
PACKET3_WRITE_DATA__CACHE_POLICY__NOA = 2 # type: ignore
PACKET3_WRITE_DATA__CACHE_POLICY__BYPASS = 3 # type: ignore
PACKET3_DRAW_INDEX_INDIRECT_MULTI = 0x38 # type: ignore
PACKET3_MEM_SEMAPHORE = 0x39 # type: ignore
PACKET3_SEM_USE_MAILBOX = (0x1 << 16) # type: ignore
PACKET3_SEM_SEL_SIGNAL_TYPE = (0x1 << 20) # type: ignore
PACKET3_SEM_SEL_SIGNAL = (0x6 << 29) # type: ignore
PACKET3_SEM_SEL_WAIT = (0x7 << 29) # type: ignore
PACKET3_DRAW_INDEX_MULTI_INST = 0x3A # type: ignore
PACKET3_COPY_DW = 0x3B # type: ignore
PACKET3_WAIT_REG_MEM = 0x3C # type: ignore
WAIT_REG_MEM_FUNCTION = lambda x: ((x) << 0) # type: ignore
WAIT_REG_MEM_MEM_SPACE = lambda x: ((x) << 4) # type: ignore
WAIT_REG_MEM_OPERATION = lambda x: ((x) << 6) # type: ignore
WAIT_REG_MEM_ENGINE = lambda x: ((x) << 8) # type: ignore
PACKET3_WAIT_REG_MEM__FUNCTION = lambda x: ((((unsigned)(x)) & 0x7) << 0) # type: ignore
PACKET3_WAIT_REG_MEM__MEM_SPACE = lambda x: ((((unsigned)(x)) & 0x3) << 4) # type: ignore
PACKET3_WAIT_REG_MEM__OPERATION = lambda x: ((((unsigned)(x)) & 0x3) << 6) # type: ignore
PACKET3_WAIT_REG_MEM__MES_INTR_PIPE = lambda x: ((((unsigned)(x)) & 0x3) << 22) # type: ignore
PACKET3_WAIT_REG_MEM__MES_ACTION = lambda x: ((((unsigned)(x)) & 0x1) << 24) # type: ignore
PACKET3_WAIT_REG_MEM__CACHE_POLICY = lambda x: ((((unsigned)(x)) & 0x3) << 25) # type: ignore
PACKET3_WAIT_REG_MEM__TEMPORAL = lambda x: ((((unsigned)(x)) & 0x3) << 25) # type: ignore
PACKET3_WAIT_REG_MEM__MEM_POLL_ADDR_LO = lambda x: ((((unsigned)(x)) & 0x3FFFFFFF) << 2) # type: ignore
PACKET3_WAIT_REG_MEM__REG_POLL_ADDR = lambda x: ((((unsigned)(x)) & 0X3FFFF) << 0) # type: ignore
PACKET3_WAIT_REG_MEM__REG_WRITE_ADDR1 = lambda x: ((((unsigned)(x)) & 0X3FFFF) << 0) # type: ignore
PACKET3_WAIT_REG_MEM__MEM_POLL_ADDR_HI = lambda x: ((unsigned)(x)) # type: ignore
PACKET3_WAIT_REG_MEM__REG_WRITE_ADDR2 = lambda x: ((((unsigned)(x)) & 0x3FFFF) << 0) # type: ignore
PACKET3_WAIT_REG_MEM__REFERENCE = lambda x: ((unsigned)(x)) # type: ignore
PACKET3_WAIT_REG_MEM__MASK = lambda x: ((unsigned)(x)) # type: ignore
PACKET3_WAIT_REG_MEM__POLL_INTERVAL = lambda x: ((((unsigned)(x)) & 0xFFFF) << 0) # type: ignore
PACKET3_WAIT_REG_MEM__OPTIMIZE_ACE_OFFLOAD_MODE = lambda x: ((((unsigned)(x)) & 0x1) << 31) # type: ignore
PACKET3_WAIT_REG_MEM__FUNCTION__ALWAYS_PASS = 0 # type: ignore
PACKET3_WAIT_REG_MEM__FUNCTION__LESS_THAN_REF_VALUE = 1 # type: ignore
PACKET3_WAIT_REG_MEM__FUNCTION__LESS_THAN_EQUAL_TO_THE_REF_VALUE = 2 # type: ignore
PACKET3_WAIT_REG_MEM__FUNCTION__EQUAL_TO_THE_REFERENCE_VALUE = 3 # type: ignore
PACKET3_WAIT_REG_MEM__FUNCTION__NOT_EQUAL_REFERENCE_VALUE = 4 # type: ignore
PACKET3_WAIT_REG_MEM__FUNCTION__GREATER_THAN_OR_EQUAL_REFERENCE_VALUE = 5 # type: ignore
PACKET3_WAIT_REG_MEM__FUNCTION__GREATER_THAN_REFERENCE_VALUE = 6 # type: ignore
PACKET3_WAIT_REG_MEM__MEM_SPACE__REGISTER_SPACE = 0 # type: ignore
PACKET3_WAIT_REG_MEM__MEM_SPACE__MEMORY_SPACE = 1 # type: ignore
PACKET3_WAIT_REG_MEM__OPERATION__WAIT_REG_MEM = 0 # type: ignore
PACKET3_WAIT_REG_MEM__OPERATION__WR_WAIT_WR_REG = 1 # type: ignore
PACKET3_WAIT_REG_MEM__OPERATION__WAIT_MEM_PREEMPTABLE = 3 # type: ignore
PACKET3_WAIT_REG_MEM__CACHE_POLICY__LRU = 0 # type: ignore
PACKET3_WAIT_REG_MEM__CACHE_POLICY__STREAM = 1 # type: ignore
PACKET3_WAIT_REG_MEM__CACHE_POLICY__NOA = 2 # type: ignore
PACKET3_WAIT_REG_MEM__CACHE_POLICY__BYPASS = 3 # type: ignore
PACKET3_WAIT_REG_MEM__TEMPORAL__RT = 0 # type: ignore
PACKET3_WAIT_REG_MEM__TEMPORAL__NT = 1 # type: ignore
PACKET3_WAIT_REG_MEM__TEMPORAL__HT = 2 # type: ignore
PACKET3_WAIT_REG_MEM__TEMPORAL__LU = 3 # type: ignore
PACKET3_INDIRECT_BUFFER = 0x3F # type: ignore
INDIRECT_BUFFER_VALID = (1 << 23) # type: ignore
INDIRECT_BUFFER_CACHE_POLICY = lambda x: ((x) << 28) # type: ignore
INDIRECT_BUFFER_PRE_ENB = lambda x: ((x) << 21) # type: ignore
INDIRECT_BUFFER_PRE_RESUME = lambda x: ((x) << 30) # type: ignore
PACKET3_INDIRECT_BUFFER__IB_BASE_LO = lambda x: ((((unsigned)(x)) & 0x3FFFFFFF) << 2) # type: ignore
PACKET3_INDIRECT_BUFFER__IB_BASE_HI = lambda x: ((unsigned)(x)) # type: ignore
PACKET3_INDIRECT_BUFFER__IB_SIZE = lambda x: ((((unsigned)(x)) & 0xFFFFF) << 0) # type: ignore
PACKET3_INDIRECT_BUFFER__CHAIN = lambda x: ((((unsigned)(x)) & 0x1) << 20) # type: ignore
PACKET3_INDIRECT_BUFFER__OFFLOAD_POLLING = lambda x: ((((unsigned)(x)) & 0x1) << 21) # type: ignore
PACKET3_INDIRECT_BUFFER__VALID = lambda x: ((((unsigned)(x)) & 0x1) << 23) # type: ignore
PACKET3_INDIRECT_BUFFER__VMID = lambda x: ((((unsigned)(x)) & 0xF) << 24) # type: ignore
PACKET3_INDIRECT_BUFFER__CACHE_POLICY = lambda x: ((((unsigned)(x)) & 0x3) << 28) # type: ignore
PACKET3_INDIRECT_BUFFER__TEMPORAL = lambda x: ((((unsigned)(x)) & 0x3) << 28) # type: ignore
PACKET3_INDIRECT_BUFFER__PRIV = lambda x: ((((unsigned)(x)) & 0x1) << 31) # type: ignore
PACKET3_INDIRECT_BUFFER__TEMPORAL__RT = 0 # type: ignore
PACKET3_INDIRECT_BUFFER__TEMPORAL__NT = 1 # type: ignore
PACKET3_INDIRECT_BUFFER__TEMPORAL__HT = 2 # type: ignore
PACKET3_INDIRECT_BUFFER__TEMPORAL__LU = 3 # type: ignore
PACKET3_INDIRECT_BUFFER__CACHE_POLICY__LRU = 0 # type: ignore
PACKET3_INDIRECT_BUFFER__CACHE_POLICY__STREAM = 1 # type: ignore
PACKET3_INDIRECT_BUFFER__CACHE_POLICY__NOA = 2 # type: ignore
PACKET3_INDIRECT_BUFFER__CACHE_POLICY__BYPASS = 3 # type: ignore
PACKET3_COND_INDIRECT_BUFFER = 0x3F # type: ignore
PACKET3_COPY_DATA = 0x40 # type: ignore
PACKET3_COPY_DATA__SRC_SEL = lambda x: ((((unsigned)(x)) & 0xF) << 0) # type: ignore
PACKET3_COPY_DATA__DST_SEL = lambda x: ((((unsigned)(x)) & 0xF) << 8) # type: ignore
PACKET3_COPY_DATA__SRC_CACHE_POLICY = lambda x: ((((unsigned)(x)) & 0x3) << 13) # type: ignore
PACKET3_COPY_DATA__SRC_TEMPORAL = lambda x: ((((unsigned)(x)) & 0x3) << 13) # type: ignore
PACKET3_COPY_DATA__COUNT_SEL = lambda x: ((((unsigned)(x)) & 0x1) << 16) # type: ignore
PACKET3_COPY_DATA__WR_CONFIRM = lambda x: ((((unsigned)(x)) & 0x1) << 20) # type: ignore
PACKET3_COPY_DATA__DST_CACHE_POLICY = lambda x: ((((unsigned)(x)) & 0x3) << 25) # type: ignore
PACKET3_COPY_DATA__PQ_EXE_STATUS = lambda x: ((((unsigned)(x)) & 0x1) << 29) # type: ignore
PACKET3_COPY_DATA__SRC_REG_OFFSET = lambda x: ((((unsigned)(x)) & 0x3FFFF) << 0) # type: ignore
PACKET3_COPY_DATA__SRC_32B_ADDR_LO = lambda x: ((((unsigned)(x)) & 0x3FFFFFFF) << 2) # type: ignore
PACKET3_COPY_DATA__SRC_64B_ADDR_LO = lambda x: ((((unsigned)(x)) & 0x1FFFFFFF) << 3) # type: ignore
PACKET3_COPY_DATA__SRC_GDS_ADDR_LO = lambda x: ((((unsigned)(x)) & 0xFFFF) << 0) # type: ignore
PACKET3_COPY_DATA__IMM_DATA = lambda x: ((unsigned)(x)) # type: ignore
PACKET3_COPY_DATA__SRC_MEMTC_ADDR_HI = lambda x: ((unsigned)(x)) # type: ignore
PACKET3_COPY_DATA__SRC_IMM_DATA = lambda x: ((unsigned)(x)) # type: ignore
PACKET3_COPY_DATA__DST_REG_OFFSET = lambda x: ((((unsigned)(x)) & 0x3FFFF) << 0) # type: ignore
PACKET3_COPY_DATA__DST_32B_ADDR_LO = lambda x: ((((unsigned)(x)) & 0x3FFFFFFF) << 2) # type: ignore
PACKET3_COPY_DATA__DST_64B_ADDR_LO = lambda x: ((((unsigned)(x)) & 0x1FFFFFFF) << 3) # type: ignore
PACKET3_COPY_DATA__DST_GDS_ADDR_LO = lambda x: ((((unsigned)(x)) & 0xFFFF) << 0) # type: ignore
PACKET3_COPY_DATA__DST_ADDR_HI = lambda x: ((unsigned)(x)) # type: ignore
PACKET3_COPY_DATA__MODE = lambda x: ((((unsigned)(x)) & 0x1) << 21) # type: ignore
PACKET3_COPY_DATA__AID_ID = lambda x: ((((unsigned)(x)) & 0x3) << 23) # type: ignore
PACKET3_COPY_DATA__DST_TEMPORAL = lambda x: ((((unsigned)(x)) & 0x3) << 25) # type: ignore
PACKET3_COPY_DATA__SRC_REG_OFFSET_LO = lambda x: ((unsigned)(x)) # type: ignore
PACKET3_COPY_DATA__SRC_REG_OFFSET_HI = lambda x: ((((unsigned)(x)) & 0xFF) << 0) # type: ignore
PACKET3_COPY_DATA__DST_REG_OFFSET_LO = lambda x: ((unsigned)(x)) # type: ignore
PACKET3_COPY_DATA__DST_REG_OFFSET_HI = lambda x: ((((unsigned)(x)) & 0xFF) << 0) # type: ignore
PACKET3_COPY_DATA__SRC_SEL__MEM_MAPPED_REGISTER = 0 # type: ignore
PACKET3_COPY_DATA__SRC_SEL__TC_L2_OBSOLETE = 1 # type: ignore
PACKET3_COPY_DATA__SRC_SEL__TC_L2 = 2 # type: ignore
PACKET3_COPY_DATA__SRC_SEL__GDS = 3 # type: ignore
PACKET3_COPY_DATA__SRC_SEL__PERFCOUNTERS = 4 # type: ignore
PACKET3_COPY_DATA__SRC_SEL__IMMEDIATE_DATA = 5 # type: ignore
PACKET3_COPY_DATA__SRC_SEL__ATOMIC_RETURN_DATA = 6 # type: ignore
PACKET3_COPY_DATA__SRC_SEL__GDS_ATOMIC_RETURN_DATA0 = 7 # type: ignore
PACKET3_COPY_DATA__SRC_SEL__GDS_ATOMIC_RETURN_DATA1 = 8 # type: ignore
PACKET3_COPY_DATA__SRC_SEL__GPU_CLOCK_COUNT = 9 # type: ignore
PACKET3_COPY_DATA__SRC_SEL__SYSTEM_CLOCK_COUNT = 10 # type: ignore
PACKET3_COPY_DATA__DST_SEL__MEM_MAPPED_REGISTER = 0 # type: ignore
PACKET3_COPY_DATA__DST_SEL__TC_L2 = 2 # type: ignore
PACKET3_COPY_DATA__DST_SEL__GDS = 3 # type: ignore
PACKET3_COPY_DATA__DST_SEL__PERFCOUNTERS = 4 # type: ignore
PACKET3_COPY_DATA__DST_SEL__TC_L2_OBSOLETE = 5 # type: ignore
PACKET3_COPY_DATA__DST_SEL__MEM_MAPPED_REG_DC = 6 # type: ignore
PACKET3_COPY_DATA__SRC_TEMPORAL__RT = 0 # type: ignore
PACKET3_COPY_DATA__SRC_TEMPORAL__NT = 1 # type: ignore
PACKET3_COPY_DATA__SRC_TEMPORAL__HT = 2 # type: ignore
PACKET3_COPY_DATA__SRC_TEMPORAL__LU = 3 # type: ignore
PACKET3_COPY_DATA__SRC_CACHE_POLICY__LRU = 0 # type: ignore
PACKET3_COPY_DATA__SRC_CACHE_POLICY__STREAM = 1 # type: ignore
PACKET3_COPY_DATA__SRC_CACHE_POLICY__NOA = 2 # type: ignore
PACKET3_COPY_DATA__SRC_CACHE_POLICY__BYPASS = 3 # type: ignore
PACKET3_COPY_DATA__COUNT_SEL__32_BITS_OF_DATA = 0 # type: ignore
PACKET3_COPY_DATA__COUNT_SEL__64_BITS_OF_DATA = 1 # type: ignore
PACKET3_COPY_DATA__WR_CONFIRM__DO_NOT_WAIT_FOR_CONFIRMATION = 0 # type: ignore
PACKET3_COPY_DATA__WR_CONFIRM__WAIT_FOR_CONFIRMATION = 1 # type: ignore
PACKET3_COPY_DATA__MODE__PF_VF_DISABLED = 0 # type: ignore
PACKET3_COPY_DATA__MODE__PF_VF_ENABLED = 1 # type: ignore
PACKET3_COPY_DATA__DST_TEMPORAL__RT = 0 # type: ignore
PACKET3_COPY_DATA__DST_TEMPORAL__NT = 1 # type: ignore
PACKET3_COPY_DATA__DST_TEMPORAL__HT = 2 # type: ignore
PACKET3_COPY_DATA__DST_TEMPORAL__LU = 3 # type: ignore
PACKET3_COPY_DATA__DST_CACHE_POLICY__LRU = 0 # type: ignore
PACKET3_COPY_DATA__DST_CACHE_POLICY__STREAM = 1 # type: ignore
PACKET3_COPY_DATA__DST_CACHE_POLICY__NOA = 2 # type: ignore
PACKET3_COPY_DATA__DST_CACHE_POLICY__BYPASS = 3 # type: ignore
PACKET3_COPY_DATA__PQ_EXE_STATUS__DEFAULT = 0 # type: ignore
PACKET3_COPY_DATA__PQ_EXE_STATUS__PHASE_UPDATE = 1 # type: ignore
PACKET3_CP_DMA = 0x41 # type: ignore
PACKET3_PFP_SYNC_ME = 0x42 # type: ignore
PACKET3_SURFACE_SYNC = 0x43 # type: ignore
PACKET3_ME_INITIALIZE = 0x44 # type: ignore
PACKET3_COND_WRITE = 0x45 # type: ignore
PACKET3_EVENT_WRITE = 0x46 # type: ignore
EVENT_TYPE = lambda x: ((x) << 0) # type: ignore
EVENT_INDEX = lambda x: ((x) << 8) # type: ignore
PACKET3_EVENT_WRITE__EVENT_TYPE = lambda x: ((((unsigned)(x)) & 0x3F) << 0) # type: ignore
PACKET3_EVENT_WRITE__EVENT_INDEX = lambda x: ((((unsigned)(x)) & 0xF) << 8) # type: ignore
PACKET3_EVENT_WRITE__SAMP_PLST_CNTR_MODE = lambda x: ((((unsigned)(x)) & 0x3) << 29) # type: ignore
PACKET3_EVENT_WRITE__OFFLOAD_ENABLE = lambda x: ((((unsigned)(x)) & 0x1) << 0) # type: ignore
PACKET3_EVENT_WRITE__ADDRESS_LO = lambda x: ((((unsigned)(x)) & 0x1FFFFFFF) << 3) # type: ignore
PACKET3_EVENT_WRITE__ADDRESS_HI = lambda x: ((unsigned)(x)) # type: ignore
PACKET3_EVENT_WRITE__EVENT_INDEX__OTHER = 0 # type: ignore
PACKET3_EVENT_WRITE__EVENT_INDEX__SAMPLE_PIPELINESTAT = 2 # type: ignore
PACKET3_EVENT_WRITE__EVENT_INDEX__CS_PARTIAL_FLUSH = 4 # type: ignore
PACKET3_EVENT_WRITE__EVENT_INDEX__SAMPLE_STREAMOUTSTATS = 8 # type: ignore
PACKET3_EVENT_WRITE__EVENT_INDEX__SAMPLE_STREAMOUTSTATS1 = 9 # type: ignore
PACKET3_EVENT_WRITE__EVENT_INDEX__SAMPLE_STREAMOUTSTATS2 = 10 # type: ignore
PACKET3_EVENT_WRITE__EVENT_INDEX__SAMPLE_STREAMOUTSTATS3 = 11 # type: ignore
PACKET3_EVENT_WRITE__SAMP_PLST_CNTR_MODE__LEGACY_MODE = 0 # type: ignore
PACKET3_EVENT_WRITE__SAMP_PLST_CNTR_MODE__MIXED_MODE1 = 1 # type: ignore
PACKET3_EVENT_WRITE__SAMP_PLST_CNTR_MODE__NEW_MODE = 2 # type: ignore
PACKET3_EVENT_WRITE__SAMP_PLST_CNTR_MODE__MIXED_MODE3 = 3 # type: ignore
PACKET3_EVENT_WRITE_EOP = 0x47 # type: ignore
PACKET3_EVENT_WRITE_EOS = 0x48 # type: ignore
PACKET3_RELEASE_MEM = 0x49 # type: ignore
PACKET3_RELEASE_MEM_EVENT_TYPE = lambda x: ((x) << 0) # type: ignore
PACKET3_RELEASE_MEM_EVENT_INDEX = lambda x: ((x) << 8) # type: ignore
PACKET3_RELEASE_MEM_GCR_GLM_WB = (1 << 12) # type: ignore
PACKET3_RELEASE_MEM_GCR_GLM_INV = (1 << 13) # type: ignore
PACKET3_RELEASE_MEM_GCR_GLV_INV = (1 << 14) # type: ignore
PACKET3_RELEASE_MEM_GCR_GL1_INV = (1 << 15) # type: ignore
PACKET3_RELEASE_MEM_GCR_GL2_US = (1 << 16) # type: ignore
PACKET3_RELEASE_MEM_GCR_GL2_RANGE = (1 << 17) # type: ignore
PACKET3_RELEASE_MEM_GCR_GL2_DISCARD = (1 << 19) # type: ignore
PACKET3_RELEASE_MEM_GCR_GL2_INV = (1 << 20) # type: ignore
PACKET3_RELEASE_MEM_GCR_GL2_WB = (1 << 21) # type: ignore
PACKET3_RELEASE_MEM_GCR_SEQ = (1 << 22) # type: ignore
PACKET3_RELEASE_MEM_CACHE_POLICY = lambda x: ((x) << 25) # type: ignore
PACKET3_RELEASE_MEM_EXECUTE = (1 << 28) # type: ignore
PACKET3_RELEASE_MEM_DATA_SEL = lambda x: ((x) << 29) # type: ignore
PACKET3_RELEASE_MEM_INT_SEL = lambda x: ((x) << 24) # type: ignore
PACKET3_RELEASE_MEM_DST_SEL = lambda x: ((x) << 16) # type: ignore
PACKET3_PREAMBLE_CNTL = 0x4A # type: ignore
PACKET3_PREAMBLE_BEGIN_CLEAR_STATE = (2 << 28) # type: ignore
PACKET3_PREAMBLE_END_CLEAR_STATE = (3 << 28) # type: ignore
PACKET3_DMA_DATA = 0x50 # type: ignore
PACKET3_DMA_DATA_ENGINE = lambda x: ((x) << 0) # type: ignore
PACKET3_DMA_DATA_SRC_CACHE_POLICY = lambda x: ((x) << 13) # type: ignore
PACKET3_DMA_DATA_DST_SEL = lambda x: ((x) << 20) # type: ignore
PACKET3_DMA_DATA_DST_CACHE_POLICY = lambda x: ((x) << 25) # type: ignore
PACKET3_DMA_DATA_SRC_SEL = lambda x: ((x) << 29) # type: ignore
PACKET3_DMA_DATA_CP_SYNC = (1 << 31) # type: ignore
PACKET3_DMA_DATA_CMD_SAS = (1 << 26) # type: ignore
PACKET3_DMA_DATA_CMD_DAS = (1 << 27) # type: ignore
PACKET3_DMA_DATA_CMD_SAIC = (1 << 28) # type: ignore
PACKET3_DMA_DATA_CMD_DAIC = (1 << 29) # type: ignore
PACKET3_DMA_DATA_CMD_RAW_WAIT = (1 << 30) # type: ignore
PACKET3_CONTEXT_REG_RMW = 0x51 # type: ignore
PACKET3_GFX_CNTX_UPDATE = 0x52 # type: ignore
PACKET3_BLK_CNTX_UPDATE = 0x53 # type: ignore
PACKET3_INCR_UPDT_STATE = 0x55 # type: ignore
PACKET3_ACQUIRE_MEM = 0x58 # type: ignore
PACKET3_ACQUIRE_MEM_GCR_CNTL_GLI_INV = lambda x: ((x) << 0) # type: ignore
PACKET3_ACQUIRE_MEM_GCR_CNTL_GL1_RANGE = lambda x: ((x) << 2) # type: ignore
PACKET3_ACQUIRE_MEM_GCR_CNTL_GLM_WB = lambda x: ((x) << 4) # type: ignore
PACKET3_ACQUIRE_MEM_GCR_CNTL_GLM_INV = lambda x: ((x) << 5) # type: ignore
PACKET3_ACQUIRE_MEM_GCR_CNTL_GLK_WB = lambda x: ((x) << 6) # type: ignore
PACKET3_ACQUIRE_MEM_GCR_CNTL_GLK_INV = lambda x: ((x) << 7) # type: ignore
PACKET3_ACQUIRE_MEM_GCR_CNTL_GLV_INV = lambda x: ((x) << 8) # type: ignore
PACKET3_ACQUIRE_MEM_GCR_CNTL_GL1_INV = lambda x: ((x) << 9) # type: ignore
PACKET3_ACQUIRE_MEM_GCR_CNTL_GL2_US = lambda x: ((x) << 10) # type: ignore
PACKET3_ACQUIRE_MEM_GCR_CNTL_GL2_RANGE = lambda x: ((x) << 11) # type: ignore
PACKET3_ACQUIRE_MEM_GCR_CNTL_GL2_DISCARD = lambda x: ((x) << 13) # type: ignore
PACKET3_ACQUIRE_MEM_GCR_CNTL_GL2_INV = lambda x: ((x) << 14) # type: ignore
PACKET3_ACQUIRE_MEM_GCR_CNTL_GL2_WB = lambda x: ((x) << 15) # type: ignore
PACKET3_ACQUIRE_MEM_GCR_CNTL_SEQ = lambda x: ((x) << 16) # type: ignore
PACKET3_ACQUIRE_MEM_GCR_RANGE_IS_PA = (1 << 18) # type: ignore
PACKET3_ACQUIRE_MEM__COHER_SIZE = lambda x: ((unsigned)(x)) # type: ignore
PACKET3_ACQUIRE_MEM__COHER_SIZE_HI = lambda x: ((((unsigned)(x)) & 0xFF) << 0) # type: ignore
PACKET3_ACQUIRE_MEM__COHER_BASE_LO = lambda x: ((unsigned)(x)) # type: ignore
PACKET3_ACQUIRE_MEM__COHER_BASE_HI = lambda x: ((((unsigned)(x)) & 0xFFFFFF) << 0) # type: ignore
PACKET3_ACQUIRE_MEM__POLL_INTERVAL = lambda x: ((((unsigned)(x)) & 0xFFFF) << 0) # type: ignore
PACKET3_ACQUIRE_MEM__GCR_CNTL = lambda x: ((((unsigned)(x)) & 0x7FFFF) << 0) # type: ignore
PACKET3_REWIND = 0x59 # type: ignore
PACKET3_INTERRUPT = 0x5A # type: ignore
PACKET3_GEN_PDEPTE = 0x5B # type: ignore
PACKET3_INDIRECT_BUFFER_PASID = 0x5C # type: ignore
PACKET3_PRIME_UTCL2 = 0x5D # type: ignore
PACKET3_LOAD_UCONFIG_REG = 0x5E # type: ignore
PACKET3_LOAD_SH_REG = 0x5F # type: ignore
PACKET3_LOAD_CONFIG_REG = 0x60 # type: ignore
PACKET3_LOAD_CONTEXT_REG = 0x61 # type: ignore
PACKET3_LOAD_COMPUTE_STATE = 0x62 # type: ignore
PACKET3_LOAD_SH_REG_INDEX = 0x63 # type: ignore
PACKET3_SET_CONFIG_REG = 0x68 # type: ignore
PACKET3_SET_CONFIG_REG_START = 0x00002000 # type: ignore
PACKET3_SET_CONFIG_REG_END = 0x00002c00 # type: ignore
PACKET3_SET_CONTEXT_REG = 0x69 # type: ignore
PACKET3_SET_CONTEXT_REG_START = 0x0000a000 # type: ignore
PACKET3_SET_CONTEXT_REG_END = 0x0000a400 # type: ignore
PACKET3_SET_CONTEXT_REG_INDEX = 0x6A # type: ignore
PACKET3_SET_VGPR_REG_DI_MULTI = 0x71 # type: ignore
PACKET3_SET_SH_REG_DI = 0x72 # type: ignore
PACKET3_SET_CONTEXT_REG_INDIRECT = 0x73 # type: ignore
PACKET3_SET_SH_REG_DI_MULTI = 0x74 # type: ignore
PACKET3_GFX_PIPE_LOCK = 0x75 # type: ignore
PACKET3_SET_SH_REG = 0x76 # type: ignore
PACKET3_SET_SH_REG_START = 0x00002c00 # type: ignore
PACKET3_SET_SH_REG_END = 0x00003000 # type: ignore
PACKET3_SET_SH_REG__REG_OFFSET = lambda x: ((((unsigned)(x)) & 0xFFFF) << 0) # type: ignore
PACKET3_SET_SH_REG__VMID_SHIFT = lambda x: ((((unsigned)(x)) & 0x1F) << 23) # type: ignore
PACKET3_SET_SH_REG__INDEX = lambda x: ((((unsigned)(x)) & 0xF) << 28) # type: ignore
PACKET3_SET_SH_REG__INDEX__DEFAULT = 0 # type: ignore
PACKET3_SET_SH_REG__INDEX__INSERT_VMID = 1 # type: ignore
PACKET3_SET_SH_REG_OFFSET = 0x77 # type: ignore
PACKET3_SET_QUEUE_REG = 0x78 # type: ignore
PACKET3_SET_UCONFIG_REG = 0x79 # type: ignore
PACKET3_SET_UCONFIG_REG_START = 0x0000c000 # type: ignore
PACKET3_SET_UCONFIG_REG_END = 0x0000c400 # type: ignore
PACKET3_SET_UCONFIG_REG__REG_OFFSET = lambda x: ((((unsigned)(x)) & 0xFFFF) << 0) # type: ignore
PACKET3_SET_UCONFIG_REG_INDEX = 0x7A # type: ignore
PACKET3_FORWARD_HEADER = 0x7C # type: ignore
PACKET3_SCRATCH_RAM_WRITE = 0x7D # type: ignore
PACKET3_SCRATCH_RAM_READ = 0x7E # type: ignore
PACKET3_LOAD_CONST_RAM = 0x80 # type: ignore
PACKET3_WRITE_CONST_RAM = 0x81 # type: ignore
PACKET3_DUMP_CONST_RAM = 0x83 # type: ignore
PACKET3_INCREMENT_CE_COUNTER = 0x84 # type: ignore
PACKET3_INCREMENT_DE_COUNTER = 0x85 # type: ignore
PACKET3_WAIT_ON_CE_COUNTER = 0x86 # type: ignore
PACKET3_WAIT_ON_DE_COUNTER_DIFF = 0x88 # type: ignore
PACKET3_SWITCH_BUFFER = 0x8B # type: ignore
PACKET3_DISPATCH_DRAW_PREAMBLE = 0x8C # type: ignore
PACKET3_DISPATCH_DRAW_PREAMBLE_ACE = 0x8C # type: ignore
PACKET3_DISPATCH_DRAW = 0x8D # type: ignore
PACKET3_DISPATCH_DRAW_ACE = 0x8D # type: ignore
PACKET3_GET_LOD_STATS = 0x8E # type: ignore
PACKET3_DRAW_MULTI_PREAMBLE = 0x8F # type: ignore
PACKET3_FRAME_CONTROL = 0x90 # type: ignore
FRAME_TMZ = (1 << 0) # type: ignore
FRAME_CMD = lambda x: ((x) << 28) # type: ignore
PACKET3_INDEX_ATTRIBUTES_INDIRECT = 0x91 # type: ignore
PACKET3_WAIT_REG_MEM64 = 0x93 # type: ignore
PACKET3_COND_PREEMPT = 0x94 # type: ignore
PACKET3_HDP_FLUSH = 0x95 # type: ignore
PACKET3_COPY_DATA_RB = 0x96 # type: ignore
PACKET3_INVALIDATE_TLBS = 0x98 # type: ignore
PACKET3_INVALIDATE_TLBS_DST_SEL = lambda x: ((x) << 0) # type: ignore
PACKET3_INVALIDATE_TLBS_ALL_HUB = lambda x: ((x) << 4) # type: ignore
PACKET3_INVALIDATE_TLBS_PASID = lambda x: ((x) << 5) # type: ignore
PACKET3_INVALIDATE_TLBS_FLUSH_TYPE = lambda x: ((x) << 29) # type: ignore
PACKET3_AQL_PACKET = 0x99 # type: ignore
PACKET3_DMA_DATA_FILL_MULTI = 0x9A # type: ignore
PACKET3_SET_SH_REG_INDEX = 0x9B # type: ignore
PACKET3_DRAW_INDIRECT_COUNT_MULTI = 0x9C # type: ignore
PACKET3_DRAW_INDEX_INDIRECT_COUNT_MULTI = 0x9D # type: ignore
PACKET3_DUMP_CONST_RAM_OFFSET = 0x9E # type: ignore
PACKET3_LOAD_CONTEXT_REG_INDEX = 0x9F # type: ignore
PACKET3_SET_RESOURCES = 0xA0 # type: ignore
PACKET3_SET_RESOURCES_VMID_MASK = lambda x: ((x) << 0) # type: ignore
PACKET3_SET_RESOURCES_UNMAP_LATENTY = lambda x: ((x) << 16) # type: ignore
PACKET3_SET_RESOURCES_QUEUE_TYPE = lambda x: ((x) << 29) # type: ignore
PACKET3_MAP_PROCESS = 0xA1 # type: ignore
PACKET3_MAP_QUEUES = 0xA2 # type: ignore
PACKET3_MAP_QUEUES_QUEUE_SEL = lambda x: ((x) << 4) # type: ignore
PACKET3_MAP_QUEUES_VMID = lambda x: ((x) << 8) # type: ignore
PACKET3_MAP_QUEUES_QUEUE = lambda x: ((x) << 13) # type: ignore
PACKET3_MAP_QUEUES_PIPE = lambda x: ((x) << 16) # type: ignore
PACKET3_MAP_QUEUES_ME = lambda x: ((x) << 18) # type: ignore
PACKET3_MAP_QUEUES_QUEUE_TYPE = lambda x: ((x) << 21) # type: ignore
PACKET3_MAP_QUEUES_ALLOC_FORMAT = lambda x: ((x) << 24) # type: ignore
PACKET3_MAP_QUEUES_ENGINE_SEL = lambda x: ((x) << 26) # type: ignore
PACKET3_MAP_QUEUES_NUM_QUEUES = lambda x: ((x) << 29) # type: ignore
PACKET3_MAP_QUEUES_CHECK_DISABLE = lambda x: ((x) << 1) # type: ignore
PACKET3_MAP_QUEUES_DOORBELL_OFFSET = lambda x: ((x) << 2) # type: ignore
PACKET3_UNMAP_QUEUES = 0xA3 # type: ignore
PACKET3_UNMAP_QUEUES_ACTION = lambda x: ((x) << 0) # type: ignore
PACKET3_UNMAP_QUEUES_QUEUE_SEL = lambda x: ((x) << 4) # type: ignore
PACKET3_UNMAP_QUEUES_ENGINE_SEL = lambda x: ((x) << 26) # type: ignore
PACKET3_UNMAP_QUEUES_NUM_QUEUES = lambda x: ((x) << 29) # type: ignore
PACKET3_UNMAP_QUEUES_PASID = lambda x: ((x) << 0) # type: ignore
PACKET3_UNMAP_QUEUES_DOORBELL_OFFSET0 = lambda x: ((x) << 2) # type: ignore
PACKET3_UNMAP_QUEUES_DOORBELL_OFFSET1 = lambda x: ((x) << 2) # type: ignore
PACKET3_UNMAP_QUEUES_RB_WPTR = lambda x: ((x) << 0) # type: ignore
PACKET3_UNMAP_QUEUES_DOORBELL_OFFSET2 = lambda x: ((x) << 2) # type: ignore
PACKET3_UNMAP_QUEUES_DOORBELL_OFFSET3 = lambda x: ((x) << 2) # type: ignore
PACKET3_QUERY_STATUS = 0xA4 # type: ignore
PACKET3_QUERY_STATUS_CONTEXT_ID = lambda x: ((x) << 0) # type: ignore
PACKET3_QUERY_STATUS_INTERRUPT_SEL = lambda x: ((x) << 28) # type: ignore
PACKET3_QUERY_STATUS_COMMAND = lambda x: ((x) << 30) # type: ignore
PACKET3_QUERY_STATUS_PASID = lambda x: ((x) << 0) # type: ignore
PACKET3_QUERY_STATUS_DOORBELL_OFFSET = lambda x: ((x) << 2) # type: ignore
PACKET3_QUERY_STATUS_ENG_SEL = lambda x: ((x) << 25) # type: ignore
PACKET3_RUN_LIST = 0xA5 # type: ignore
PACKET3_MAP_PROCESS_VM = 0xA6 # type: ignore
PACKET3_RUN_CLEANER_SHADER = 0xD2 # type: ignore
PACKET3_SET_Q_PREEMPTION_MODE = 0xF0 # type: ignore
PACKET3_SET_Q_PREEMPTION_MODE_IB_VMID = lambda x: ((x) << 0) # type: ignore
PACKET3_SET_Q_PREEMPTION_MODE_INIT_SHADOW_MEM = (1 << 0) # type: ignore