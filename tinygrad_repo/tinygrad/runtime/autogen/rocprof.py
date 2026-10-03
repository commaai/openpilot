# mypy: disable-error-code="empty-body"
from __future__ import annotations
import ctypes
from typing import Literal, TypeAlias
from tinygrad.runtime.support.c import _IO, _IOW, _IOR, _IOWR
from tinygrad.runtime.support import c
dll = c.DLL('rocprof', ['rocprof-trace-decoder', p:='/usr/local/lib/rocprof-trace-decoder.so', p.replace('so','dylib')])
rocprofiler_thread_trace_decoder_status_t: dict[int, str] = {(ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS:=0): 'ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS', (ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR:=1): 'ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR', (ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_OUT_OF_RESOURCES:=2): 'ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_OUT_OF_RESOURCES', (ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT:=3): 'ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT', (ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_SHADER_DATA:=4): 'ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_SHADER_DATA', (ROCPROFILER_THREAD_TRACE_DECODER_STATUS_LAST:=5): 'ROCPROFILER_THREAD_TRACE_DECODER_STATUS_LAST'}
enum_rocprofiler_thread_trace_decoder_record_type_t: dict[int, str] = {(ROCPROFILER_THREAD_TRACE_DECODER_RECORD_GFXIP:=0): 'ROCPROFILER_THREAD_TRACE_DECODER_RECORD_GFXIP', (ROCPROFILER_THREAD_TRACE_DECODER_RECORD_OCCUPANCY:=1): 'ROCPROFILER_THREAD_TRACE_DECODER_RECORD_OCCUPANCY', (ROCPROFILER_THREAD_TRACE_DECODER_RECORD_PERFEVENT:=2): 'ROCPROFILER_THREAD_TRACE_DECODER_RECORD_PERFEVENT', (ROCPROFILER_THREAD_TRACE_DECODER_RECORD_WAVE:=3): 'ROCPROFILER_THREAD_TRACE_DECODER_RECORD_WAVE', (ROCPROFILER_THREAD_TRACE_DECODER_RECORD_INFO:=4): 'ROCPROFILER_THREAD_TRACE_DECODER_RECORD_INFO', (ROCPROFILER_THREAD_TRACE_DECODER_RECORD_DEBUG:=5): 'ROCPROFILER_THREAD_TRACE_DECODER_RECORD_DEBUG', (ROCPROFILER_THREAD_TRACE_DECODER_RECORD_SHADERDATA:=6): 'ROCPROFILER_THREAD_TRACE_DECODER_RECORD_SHADERDATA', (ROCPROFILER_THREAD_TRACE_DECODER_RECORD_REALTIME:=7): 'ROCPROFILER_THREAD_TRACE_DECODER_RECORD_REALTIME', (ROCPROFILER_THREAD_TRACE_DECODER_RECORD_RT_FREQUENCY:=8): 'ROCPROFILER_THREAD_TRACE_DECODER_RECORD_RT_FREQUENCY', (ROCPROFILER_THREAD_TRACE_DECODER_RECORD_LAST:=9): 'ROCPROFILER_THREAD_TRACE_DECODER_RECORD_LAST'}
rocprof_trace_decoder_trace_callback_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint32, [ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p]]
@c.record
class struct_rocprofiler_thread_trace_decoder_pc_t(c.Struct):
  SIZE = 16
  address: int
  code_object_id: int
uint64_t: TypeAlias = ctypes.c_uint64
struct_rocprofiler_thread_trace_decoder_pc_t.register_fields([('address', uint64_t, 0), ('code_object_id', uint64_t, 8)])
rocprof_trace_decoder_isa_callback_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint32, [c.POINTER[ctypes.c_char], c.POINTER[ctypes.c_uint64], c.POINTER[ctypes.c_uint64], struct_rocprofiler_thread_trace_decoder_pc_t, ctypes.c_void_p]]
rocprof_trace_decoder_se_data_callback_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint64, [c.POINTER[c.POINTER[ctypes.c_ubyte]], c.POINTER[ctypes.c_uint64], ctypes.c_void_p]]
@dll.bind(ctypes.c_uint32, rocprof_trace_decoder_se_data_callback_t, rocprof_trace_decoder_trace_callback_t, rocprof_trace_decoder_isa_callback_t, ctypes.c_void_p)
def rocprof_trace_decoder_parse_data(se_data_callback:rocprof_trace_decoder_se_data_callback_t, trace_callback:rocprof_trace_decoder_trace_callback_t, isa_callback:rocprof_trace_decoder_isa_callback_t, userdata:ctypes.c_void_p) -> ctypes.c_uint32: ...
enum_rocprofiler_thread_trace_decoder_info_t: dict[int, str] = {(ROCPROFILER_THREAD_TRACE_DECODER_INFO_NONE:=0): 'ROCPROFILER_THREAD_TRACE_DECODER_INFO_NONE', (ROCPROFILER_THREAD_TRACE_DECODER_INFO_DATA_LOST:=1): 'ROCPROFILER_THREAD_TRACE_DECODER_INFO_DATA_LOST', (ROCPROFILER_THREAD_TRACE_DECODER_INFO_STITCH_INCOMPLETE:=2): 'ROCPROFILER_THREAD_TRACE_DECODER_INFO_STITCH_INCOMPLETE', (ROCPROFILER_THREAD_TRACE_DECODER_INFO_WAVE_INCOMPLETE:=3): 'ROCPROFILER_THREAD_TRACE_DECODER_INFO_WAVE_INCOMPLETE', (ROCPROFILER_THREAD_TRACE_DECODER_INFO_LAST:=4): 'ROCPROFILER_THREAD_TRACE_DECODER_INFO_LAST'}
rocprofiler_thread_trace_decoder_info_t: TypeAlias = ctypes.c_uint32
@dll.bind(c.POINTER[ctypes.c_char], rocprofiler_thread_trace_decoder_info_t)
def rocprof_trace_decoder_get_info_string(info:rocprofiler_thread_trace_decoder_info_t) -> c.POINTER[ctypes.c_char]: ...
@dll.bind(c.POINTER[ctypes.c_char], ctypes.c_uint32)
def rocprof_trace_decoder_get_status_string(status:ctypes.c_uint32) -> c.POINTER[ctypes.c_char]: ...
rocprofiler_thread_trace_decoder_debug_callback_t: TypeAlias = c.CFUNCTYPE[None, [ctypes.c_int64, c.POINTER[ctypes.c_char], c.POINTER[ctypes.c_char], ctypes.c_void_p]]
@dll.bind(ctypes.c_uint32, c.POINTER[ctypes.c_char], uint64_t, rocprofiler_thread_trace_decoder_debug_callback_t, ctypes.c_void_p)
def rocprof_trace_decoder_dump_data(data:c.POINTER[ctypes.c_char], data_size:uint64_t, cb:rocprofiler_thread_trace_decoder_debug_callback_t, userdata:ctypes.c_void_p) -> ctypes.c_uint32: ...
@c.record
class union_rocprof_trace_decoder_gfx9_header_t(c.Struct):
  SIZE = 8
  legacy_version: int
  gfx9_version2: int
  DSIMDM: int
  DCU: int
  reserved1: int
  SEID: int
  reserved2: int
  raw: int
union_rocprof_trace_decoder_gfx9_header_t.register_fields([('legacy_version', uint64_t, 0, 13, 0), ('gfx9_version2', uint64_t, 1, 3, 5), ('DSIMDM', uint64_t, 2, 4, 0), ('DCU', uint64_t, 2, 5, 4), ('reserved1', uint64_t, 3, 1, 1), ('SEID', uint64_t, 3, 6, 2), ('reserved2', uint64_t, 4, 32, 0), ('raw', uint64_t, 0)])
rocprof_trace_decoder_gfx9_header_t: TypeAlias = union_rocprof_trace_decoder_gfx9_header_t
@c.record
class union_rocprof_trace_decoder_instrument_enable_t(c.Struct):
  SIZE = 4
  char1: int
  char2: int
  char3: int
  char4: int
  u32All: int
union_rocprof_trace_decoder_instrument_enable_t.register_fields([('char1', ctypes.c_uint32, 0, 8, 0), ('char2', ctypes.c_uint32, 1, 8, 0), ('char3', ctypes.c_uint32, 2, 8, 0), ('char4', ctypes.c_uint32, 3, 8, 0), ('u32All', ctypes.c_uint32, 0)])
rocprof_trace_decoder_instrument_enable_t: TypeAlias = union_rocprof_trace_decoder_instrument_enable_t
@c.record
class union_rocprof_trace_decoder_packet_header_t(c.Struct):
  SIZE = 4
  opcode: int
  type: int
  data20: int
  u32All: int
union_rocprof_trace_decoder_packet_header_t.register_fields([('opcode', ctypes.c_uint32, 0, 8, 0), ('type', ctypes.c_uint32, 1, 4, 0), ('data20', ctypes.c_uint32, 1, 20, 4), ('u32All', ctypes.c_uint32, 0)])
rocprof_trace_decoder_packet_header_t: TypeAlias = union_rocprof_trace_decoder_packet_header_t
enum_rocprof_trace_decoder_packet_opcode_t: dict[int, str] = {(ROCPROF_TRACE_DECODER_PACKET_OPCODE_CODEOBJ:=4): 'ROCPROF_TRACE_DECODER_PACKET_OPCODE_CODEOBJ', (ROCPROF_TRACE_DECODER_PACKET_OPCODE_RT_TIMESTAMP:=5): 'ROCPROF_TRACE_DECODER_PACKET_OPCODE_RT_TIMESTAMP', (ROCPROF_TRACE_DECODER_PACKET_OPCODE_AGENT_INFO:=6): 'ROCPROF_TRACE_DECODER_PACKET_OPCODE_AGENT_INFO'}
rocprof_trace_decoder_packet_opcode_t: TypeAlias = ctypes.c_uint32
enum_rocprof_trace_decoder_agent_info_type_t: dict[int, str] = {(ROCPROF_TRACE_DECODER_AGENT_INFO_TYPE_RT_FREQUENCY_KHZ:=0): 'ROCPROF_TRACE_DECODER_AGENT_INFO_TYPE_RT_FREQUENCY_KHZ', (ROCPROF_TRACE_DECODER_AGENT_INFO_TYPE_COUNTER_INTERVAL:=1): 'ROCPROF_TRACE_DECODER_AGENT_INFO_TYPE_COUNTER_INTERVAL', (ROCPROF_TRACE_DECODER_AGENT_INFO_TYPE_LAST:=2): 'ROCPROF_TRACE_DECODER_AGENT_INFO_TYPE_LAST'}
rocprof_trace_decoder_agent_info_type_t: TypeAlias = ctypes.c_uint32
@c.record
class union_rocprof_trace_decoder_codeobj_marker_tail_t(c.Struct):
  SIZE = 4
  isUnload: int
  bFromStart: int
  legacy_id: int
  raw: int
uint32_t: TypeAlias = ctypes.c_uint32
union_rocprof_trace_decoder_codeobj_marker_tail_t.register_fields([('isUnload', uint32_t, 0, 1, 0), ('bFromStart', uint32_t, 0, 1, 1), ('legacy_id', uint32_t, 0, 30, 2), ('raw', uint32_t, 0)])
rocprof_trace_decoder_codeobj_marker_tail_t: TypeAlias = union_rocprof_trace_decoder_codeobj_marker_tail_t
enum_rocprof_trace_decoder_codeobj_marker_type_t: dict[int, str] = {(ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_TAIL:=0): 'ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_TAIL', (ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_SIZE_LO:=1): 'ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_SIZE_LO', (ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ADDR_LO:=2): 'ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ADDR_LO', (ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ADDR_HI:=3): 'ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ADDR_HI', (ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_SIZE_HI:=4): 'ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_SIZE_HI', (ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ID_LO:=5): 'ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ID_LO', (ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ID_HI:=6): 'ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ID_HI', (ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_LAST:=7): 'ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_LAST'}
rocprof_trace_decoder_codeobj_marker_type_t: TypeAlias = ctypes.c_uint32
rocprofiler_thread_trace_decoder_pc_t: TypeAlias = struct_rocprofiler_thread_trace_decoder_pc_t
@c.record
class struct_rocprofiler_thread_trace_decoder_perfevent_t(c.Struct):
  SIZE = 24
  time: int
  events0: int
  events1: int
  events2: int
  events3: int
  CU: int
  bank: int
int64_t: TypeAlias = ctypes.c_int64
uint16_t: TypeAlias = ctypes.c_uint16
uint8_t: TypeAlias = ctypes.c_ubyte
struct_rocprofiler_thread_trace_decoder_perfevent_t.register_fields([('time', int64_t, 0), ('events0', uint16_t, 8), ('events1', uint16_t, 10), ('events2', uint16_t, 12), ('events3', uint16_t, 14), ('CU', uint8_t, 16), ('bank', uint8_t, 17)])
rocprofiler_thread_trace_decoder_perfevent_t: TypeAlias = struct_rocprofiler_thread_trace_decoder_perfevent_t
@c.record
class struct_rocprofiler_thread_trace_decoder_occupancy_t(c.Struct):
  SIZE = 32
  pc: struct_rocprofiler_thread_trace_decoder_pc_t
  time: int
  reserved: int
  cu: int
  simd: int
  wave_id: int
  start: int
  _rsvd: int
struct_rocprofiler_thread_trace_decoder_occupancy_t.register_fields([('pc', rocprofiler_thread_trace_decoder_pc_t, 0), ('time', uint64_t, 16), ('reserved', uint8_t, 24), ('cu', uint8_t, 25), ('simd', uint8_t, 26), ('wave_id', uint8_t, 27), ('start', uint32_t, 28, 1, 0), ('_rsvd', uint32_t, 28, 31, 1)])
rocprofiler_thread_trace_decoder_occupancy_t: TypeAlias = struct_rocprofiler_thread_trace_decoder_occupancy_t
enum_rocprofiler_thread_trace_decoder_wstate_type_t: dict[int, str] = {(ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_EMPTY:=0): 'ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_EMPTY', (ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_IDLE:=1): 'ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_IDLE', (ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_EXEC:=2): 'ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_EXEC', (ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_WAIT:=3): 'ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_WAIT', (ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_STALL:=4): 'ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_STALL', (ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_LAST:=5): 'ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_LAST'}
rocprofiler_thread_trace_decoder_wstate_type_t: TypeAlias = ctypes.c_uint32
@c.record
class struct_rocprofiler_thread_trace_decoder_wave_state_t(c.Struct):
  SIZE = 8
  type: int
  duration: int
int32_t: TypeAlias = ctypes.c_int32
struct_rocprofiler_thread_trace_decoder_wave_state_t.register_fields([('type', int32_t, 0), ('duration', int32_t, 4)])
rocprofiler_thread_trace_decoder_wave_state_t: TypeAlias = struct_rocprofiler_thread_trace_decoder_wave_state_t
enum_rocprofiler_thread_trace_decoder_inst_category_t: dict[int, str] = {(ROCPROFILER_THREAD_TRACE_DECODER_INST_NONE:=0): 'ROCPROFILER_THREAD_TRACE_DECODER_INST_NONE', (ROCPROFILER_THREAD_TRACE_DECODER_INST_SMEM:=1): 'ROCPROFILER_THREAD_TRACE_DECODER_INST_SMEM', (ROCPROFILER_THREAD_TRACE_DECODER_INST_SALU:=2): 'ROCPROFILER_THREAD_TRACE_DECODER_INST_SALU', (ROCPROFILER_THREAD_TRACE_DECODER_INST_VMEM:=3): 'ROCPROFILER_THREAD_TRACE_DECODER_INST_VMEM', (ROCPROFILER_THREAD_TRACE_DECODER_INST_FLAT:=4): 'ROCPROFILER_THREAD_TRACE_DECODER_INST_FLAT', (ROCPROFILER_THREAD_TRACE_DECODER_INST_LDS:=5): 'ROCPROFILER_THREAD_TRACE_DECODER_INST_LDS', (ROCPROFILER_THREAD_TRACE_DECODER_INST_VALU:=6): 'ROCPROFILER_THREAD_TRACE_DECODER_INST_VALU', (ROCPROFILER_THREAD_TRACE_DECODER_INST_JUMP:=7): 'ROCPROFILER_THREAD_TRACE_DECODER_INST_JUMP', (ROCPROFILER_THREAD_TRACE_DECODER_INST_NEXT:=8): 'ROCPROFILER_THREAD_TRACE_DECODER_INST_NEXT', (ROCPROFILER_THREAD_TRACE_DECODER_INST_IMMED:=9): 'ROCPROFILER_THREAD_TRACE_DECODER_INST_IMMED', (ROCPROFILER_THREAD_TRACE_DECODER_INST_CONTEXT:=10): 'ROCPROFILER_THREAD_TRACE_DECODER_INST_CONTEXT', (ROCPROFILER_THREAD_TRACE_DECODER_INST_MESSAGE:=11): 'ROCPROFILER_THREAD_TRACE_DECODER_INST_MESSAGE', (ROCPROFILER_THREAD_TRACE_DECODER_INST_BVH:=12): 'ROCPROFILER_THREAD_TRACE_DECODER_INST_BVH', (ROCPROFILER_THREAD_TRACE_DECODER_INST_LAST:=13): 'ROCPROFILER_THREAD_TRACE_DECODER_INST_LAST'}
rocprofiler_thread_trace_decoder_inst_category_t: TypeAlias = ctypes.c_uint32
@c.record
class struct_rocprofiler_thread_trace_decoder_inst_t(c.Struct):
  SIZE = 32
  category: int
  stall: int
  duration: int
  time: int
  pc: struct_rocprofiler_thread_trace_decoder_pc_t
struct_rocprofiler_thread_trace_decoder_inst_t.register_fields([('category', uint32_t, 0, 8, 0), ('stall', uint32_t, 1, 24, 0), ('duration', int32_t, 4), ('time', int64_t, 8), ('pc', rocprofiler_thread_trace_decoder_pc_t, 16)])
rocprofiler_thread_trace_decoder_inst_t: TypeAlias = struct_rocprofiler_thread_trace_decoder_inst_t
@c.record
class struct_rocprofiler_thread_trace_decoder_wave_t(c.Struct):
  SIZE = 64
  cu: int
  simd: int
  wave_id: int
  contexts: int
  _rsvd1: int
  _rsvd2: int
  _rsvd3: int
  begin_time: int
  end_time: int
  timeline_size: int
  instructions_size: int
  timeline_array: c.POINTER[struct_rocprofiler_thread_trace_decoder_wave_state_t]
  instructions_array: c.POINTER[struct_rocprofiler_thread_trace_decoder_inst_t]
struct_rocprofiler_thread_trace_decoder_wave_t.register_fields([('cu', uint8_t, 0), ('simd', uint8_t, 1), ('wave_id', uint8_t, 2), ('contexts', uint8_t, 3), ('_rsvd1', uint32_t, 4), ('_rsvd2', uint32_t, 8), ('_rsvd3', uint32_t, 12), ('begin_time', int64_t, 16), ('end_time', int64_t, 24), ('timeline_size', uint64_t, 32), ('instructions_size', uint64_t, 40), ('timeline_array', c.POINTER[rocprofiler_thread_trace_decoder_wave_state_t], 48), ('instructions_array', c.POINTER[rocprofiler_thread_trace_decoder_inst_t], 56)])
rocprofiler_thread_trace_decoder_wave_t: TypeAlias = struct_rocprofiler_thread_trace_decoder_wave_t
@c.record
class struct_rocprofiler_thread_trace_decoder_realtime_t(c.Struct):
  SIZE = 24
  shader_clock: int
  realtime_clock: int
  reserved: int
struct_rocprofiler_thread_trace_decoder_realtime_t.register_fields([('shader_clock', int64_t, 0), ('realtime_clock', uint64_t, 8), ('reserved', uint64_t, 16)])
rocprofiler_thread_trace_decoder_realtime_t: TypeAlias = struct_rocprofiler_thread_trace_decoder_realtime_t
enum_rocprofiler_thread_trace_decoder_shaderdata_flags_t: dict[int, str] = {(ROCPROFILER_THREAD_TRACE_DECODER_SHADERDATA_FLAGS_IMM:=0): 'ROCPROFILER_THREAD_TRACE_DECODER_SHADERDATA_FLAGS_IMM', (ROCPROFILER_THREAD_TRACE_DECODER_SHADERDATA_FLAGS_PRIV:=1): 'ROCPROFILER_THREAD_TRACE_DECODER_SHADERDATA_FLAGS_PRIV'}
rocprofiler_thread_trace_decoder_shaderdata_flags_t: TypeAlias = ctypes.c_uint32
@c.record
class struct_rocprofiler_thread_trace_decoder_shaderdata_t(c.Struct):
  SIZE = 24
  time: int
  value: int
  cu: int
  simd: int
  wave_id: int
  flags: int
  reserved: int
struct_rocprofiler_thread_trace_decoder_shaderdata_t.register_fields([('time', int64_t, 0), ('value', uint64_t, 8), ('cu', uint8_t, 16), ('simd', uint8_t, 17), ('wave_id', uint8_t, 18), ('flags', uint8_t, 19), ('reserved', uint32_t, 20)])
rocprofiler_thread_trace_decoder_shaderdata_t: TypeAlias = struct_rocprofiler_thread_trace_decoder_shaderdata_t
rocprofiler_thread_trace_decoder_record_type_t: TypeAlias = ctypes.c_uint32
