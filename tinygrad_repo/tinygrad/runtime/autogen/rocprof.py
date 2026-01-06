# mypy: ignore-errors
import ctypes
from tinygrad.runtime.support.c import DLL, Struct, CEnum, _IO, _IOW, _IOR, _IOWR
dll = DLL('rocprof', ['rocprof-trace-decoder', p:='/usr/local/lib/rocprof-trace-decoder.so', p.replace('so','dylib')])
rocprofiler_thread_trace_decoder_status_t = CEnum(ctypes.c_uint32)
ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS = rocprofiler_thread_trace_decoder_status_t.define('ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS', 0)
ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR = rocprofiler_thread_trace_decoder_status_t.define('ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR', 1)
ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_OUT_OF_RESOURCES = rocprofiler_thread_trace_decoder_status_t.define('ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_OUT_OF_RESOURCES', 2)
ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT = rocprofiler_thread_trace_decoder_status_t.define('ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT', 3)
ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_SHADER_DATA = rocprofiler_thread_trace_decoder_status_t.define('ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_SHADER_DATA', 4)
ROCPROFILER_THREAD_TRACE_DECODER_STATUS_LAST = rocprofiler_thread_trace_decoder_status_t.define('ROCPROFILER_THREAD_TRACE_DECODER_STATUS_LAST', 5)

enum_rocprofiler_thread_trace_decoder_record_type_t = CEnum(ctypes.c_uint32)
ROCPROFILER_THREAD_TRACE_DECODER_RECORD_GFXIP = enum_rocprofiler_thread_trace_decoder_record_type_t.define('ROCPROFILER_THREAD_TRACE_DECODER_RECORD_GFXIP', 0)
ROCPROFILER_THREAD_TRACE_DECODER_RECORD_OCCUPANCY = enum_rocprofiler_thread_trace_decoder_record_type_t.define('ROCPROFILER_THREAD_TRACE_DECODER_RECORD_OCCUPANCY', 1)
ROCPROFILER_THREAD_TRACE_DECODER_RECORD_PERFEVENT = enum_rocprofiler_thread_trace_decoder_record_type_t.define('ROCPROFILER_THREAD_TRACE_DECODER_RECORD_PERFEVENT', 2)
ROCPROFILER_THREAD_TRACE_DECODER_RECORD_WAVE = enum_rocprofiler_thread_trace_decoder_record_type_t.define('ROCPROFILER_THREAD_TRACE_DECODER_RECORD_WAVE', 3)
ROCPROFILER_THREAD_TRACE_DECODER_RECORD_INFO = enum_rocprofiler_thread_trace_decoder_record_type_t.define('ROCPROFILER_THREAD_TRACE_DECODER_RECORD_INFO', 4)
ROCPROFILER_THREAD_TRACE_DECODER_RECORD_DEBUG = enum_rocprofiler_thread_trace_decoder_record_type_t.define('ROCPROFILER_THREAD_TRACE_DECODER_RECORD_DEBUG', 5)
ROCPROFILER_THREAD_TRACE_DECODER_RECORD_SHADERDATA = enum_rocprofiler_thread_trace_decoder_record_type_t.define('ROCPROFILER_THREAD_TRACE_DECODER_RECORD_SHADERDATA', 6)
ROCPROFILER_THREAD_TRACE_DECODER_RECORD_REALTIME = enum_rocprofiler_thread_trace_decoder_record_type_t.define('ROCPROFILER_THREAD_TRACE_DECODER_RECORD_REALTIME', 7)
ROCPROFILER_THREAD_TRACE_DECODER_RECORD_RT_FREQUENCY = enum_rocprofiler_thread_trace_decoder_record_type_t.define('ROCPROFILER_THREAD_TRACE_DECODER_RECORD_RT_FREQUENCY', 8)
ROCPROFILER_THREAD_TRACE_DECODER_RECORD_LAST = enum_rocprofiler_thread_trace_decoder_record_type_t.define('ROCPROFILER_THREAD_TRACE_DECODER_RECORD_LAST', 9)

rocprof_trace_decoder_trace_callback_t = ctypes.CFUNCTYPE(rocprofiler_thread_trace_decoder_status_t, enum_rocprofiler_thread_trace_decoder_record_type_t, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p)
class struct_rocprofiler_thread_trace_decoder_pc_t(Struct): pass
uint64_t = ctypes.c_uint64
struct_rocprofiler_thread_trace_decoder_pc_t._fields_ = [
  ('address', uint64_t),
  ('code_object_id', uint64_t),
]
rocprof_trace_decoder_isa_callback_t = ctypes.CFUNCTYPE(rocprofiler_thread_trace_decoder_status_t, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), struct_rocprofiler_thread_trace_decoder_pc_t, ctypes.c_void_p)
rocprof_trace_decoder_se_data_callback_t = ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)), ctypes.POINTER(ctypes.c_uint64), ctypes.c_void_p)
try: (rocprof_trace_decoder_parse_data:=dll.rocprof_trace_decoder_parse_data).restype, rocprof_trace_decoder_parse_data.argtypes = rocprofiler_thread_trace_decoder_status_t, [rocprof_trace_decoder_se_data_callback_t, rocprof_trace_decoder_trace_callback_t, rocprof_trace_decoder_isa_callback_t, ctypes.c_void_p]
except AttributeError: pass

enum_rocprofiler_thread_trace_decoder_info_t = CEnum(ctypes.c_uint32)
ROCPROFILER_THREAD_TRACE_DECODER_INFO_NONE = enum_rocprofiler_thread_trace_decoder_info_t.define('ROCPROFILER_THREAD_TRACE_DECODER_INFO_NONE', 0)
ROCPROFILER_THREAD_TRACE_DECODER_INFO_DATA_LOST = enum_rocprofiler_thread_trace_decoder_info_t.define('ROCPROFILER_THREAD_TRACE_DECODER_INFO_DATA_LOST', 1)
ROCPROFILER_THREAD_TRACE_DECODER_INFO_STITCH_INCOMPLETE = enum_rocprofiler_thread_trace_decoder_info_t.define('ROCPROFILER_THREAD_TRACE_DECODER_INFO_STITCH_INCOMPLETE', 2)
ROCPROFILER_THREAD_TRACE_DECODER_INFO_WAVE_INCOMPLETE = enum_rocprofiler_thread_trace_decoder_info_t.define('ROCPROFILER_THREAD_TRACE_DECODER_INFO_WAVE_INCOMPLETE', 3)
ROCPROFILER_THREAD_TRACE_DECODER_INFO_LAST = enum_rocprofiler_thread_trace_decoder_info_t.define('ROCPROFILER_THREAD_TRACE_DECODER_INFO_LAST', 4)

rocprofiler_thread_trace_decoder_info_t = enum_rocprofiler_thread_trace_decoder_info_t
try: (rocprof_trace_decoder_get_info_string:=dll.rocprof_trace_decoder_get_info_string).restype, rocprof_trace_decoder_get_info_string.argtypes = ctypes.POINTER(ctypes.c_char), [rocprofiler_thread_trace_decoder_info_t]
except AttributeError: pass

try: (rocprof_trace_decoder_get_status_string:=dll.rocprof_trace_decoder_get_status_string).restype, rocprof_trace_decoder_get_status_string.argtypes = ctypes.POINTER(ctypes.c_char), [rocprofiler_thread_trace_decoder_status_t]
except AttributeError: pass

rocprofiler_thread_trace_decoder_debug_callback_t = ctypes.CFUNCTYPE(None, ctypes.c_int64, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), ctypes.c_void_p)
try: (rocprof_trace_decoder_dump_data:=dll.rocprof_trace_decoder_dump_data).restype, rocprof_trace_decoder_dump_data.argtypes = rocprofiler_thread_trace_decoder_status_t, [ctypes.POINTER(ctypes.c_char), uint64_t, rocprofiler_thread_trace_decoder_debug_callback_t, ctypes.c_void_p]
except AttributeError: pass

class union_rocprof_trace_decoder_gfx9_header_t(ctypes.Union): pass
class union_rocprof_trace_decoder_gfx9_header_t_0(Struct): pass
union_rocprof_trace_decoder_gfx9_header_t_0._fields_ = [
  ('legacy_version', uint64_t,13),
  ('gfx9_version2', uint64_t,3),
  ('DSIMDM', uint64_t,4),
  ('DCU', uint64_t,5),
  ('reserved1', uint64_t,1),
  ('SEID', uint64_t,6),
  ('reserved2', uint64_t,32),
]
union_rocprof_trace_decoder_gfx9_header_t._anonymous_ = ['_0']
union_rocprof_trace_decoder_gfx9_header_t._fields_ = [
  ('_0', union_rocprof_trace_decoder_gfx9_header_t_0),
  ('raw', uint64_t),
]
rocprof_trace_decoder_gfx9_header_t = union_rocprof_trace_decoder_gfx9_header_t
class union_rocprof_trace_decoder_instrument_enable_t(ctypes.Union): pass
class union_rocprof_trace_decoder_instrument_enable_t_0(Struct): pass
union_rocprof_trace_decoder_instrument_enable_t_0._fields_ = [
  ('char1', ctypes.c_uint32,8),
  ('char2', ctypes.c_uint32,8),
  ('char3', ctypes.c_uint32,8),
  ('char4', ctypes.c_uint32,8),
]
union_rocprof_trace_decoder_instrument_enable_t._anonymous_ = ['_0']
union_rocprof_trace_decoder_instrument_enable_t._fields_ = [
  ('_0', union_rocprof_trace_decoder_instrument_enable_t_0),
  ('u32All', ctypes.c_uint32),
]
rocprof_trace_decoder_instrument_enable_t = union_rocprof_trace_decoder_instrument_enable_t
class union_rocprof_trace_decoder_packet_header_t(ctypes.Union): pass
class union_rocprof_trace_decoder_packet_header_t_0(Struct): pass
union_rocprof_trace_decoder_packet_header_t_0._fields_ = [
  ('opcode', ctypes.c_uint32,8),
  ('type', ctypes.c_uint32,4),
  ('data20', ctypes.c_uint32,20),
]
union_rocprof_trace_decoder_packet_header_t._anonymous_ = ['_0']
union_rocprof_trace_decoder_packet_header_t._fields_ = [
  ('_0', union_rocprof_trace_decoder_packet_header_t_0),
  ('u32All', ctypes.c_uint32),
]
rocprof_trace_decoder_packet_header_t = union_rocprof_trace_decoder_packet_header_t
enum_rocprof_trace_decoder_packet_opcode_t = CEnum(ctypes.c_uint32)
ROCPROF_TRACE_DECODER_PACKET_OPCODE_CODEOBJ = enum_rocprof_trace_decoder_packet_opcode_t.define('ROCPROF_TRACE_DECODER_PACKET_OPCODE_CODEOBJ', 4)
ROCPROF_TRACE_DECODER_PACKET_OPCODE_RT_TIMESTAMP = enum_rocprof_trace_decoder_packet_opcode_t.define('ROCPROF_TRACE_DECODER_PACKET_OPCODE_RT_TIMESTAMP', 5)
ROCPROF_TRACE_DECODER_PACKET_OPCODE_AGENT_INFO = enum_rocprof_trace_decoder_packet_opcode_t.define('ROCPROF_TRACE_DECODER_PACKET_OPCODE_AGENT_INFO', 6)

rocprof_trace_decoder_packet_opcode_t = enum_rocprof_trace_decoder_packet_opcode_t
enum_rocprof_trace_decoder_agent_info_type_t = CEnum(ctypes.c_uint32)
ROCPROF_TRACE_DECODER_AGENT_INFO_TYPE_RT_FREQUENCY_KHZ = enum_rocprof_trace_decoder_agent_info_type_t.define('ROCPROF_TRACE_DECODER_AGENT_INFO_TYPE_RT_FREQUENCY_KHZ', 0)
ROCPROF_TRACE_DECODER_AGENT_INFO_TYPE_COUNTER_INTERVAL = enum_rocprof_trace_decoder_agent_info_type_t.define('ROCPROF_TRACE_DECODER_AGENT_INFO_TYPE_COUNTER_INTERVAL', 1)
ROCPROF_TRACE_DECODER_AGENT_INFO_TYPE_LAST = enum_rocprof_trace_decoder_agent_info_type_t.define('ROCPROF_TRACE_DECODER_AGENT_INFO_TYPE_LAST', 2)

rocprof_trace_decoder_agent_info_type_t = enum_rocprof_trace_decoder_agent_info_type_t
class union_rocprof_trace_decoder_codeobj_marker_tail_t(ctypes.Union): pass
class union_rocprof_trace_decoder_codeobj_marker_tail_t_0(Struct): pass
uint32_t = ctypes.c_uint32
union_rocprof_trace_decoder_codeobj_marker_tail_t_0._fields_ = [
  ('isUnload', uint32_t,1),
  ('bFromStart', uint32_t,1),
  ('legacy_id', uint32_t,30),
]
union_rocprof_trace_decoder_codeobj_marker_tail_t._anonymous_ = ['_0']
union_rocprof_trace_decoder_codeobj_marker_tail_t._fields_ = [
  ('_0', union_rocprof_trace_decoder_codeobj_marker_tail_t_0),
  ('raw', uint32_t),
]
rocprof_trace_decoder_codeobj_marker_tail_t = union_rocprof_trace_decoder_codeobj_marker_tail_t
enum_rocprof_trace_decoder_codeobj_marker_type_t = CEnum(ctypes.c_uint32)
ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_TAIL = enum_rocprof_trace_decoder_codeobj_marker_type_t.define('ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_TAIL', 0)
ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_SIZE_LO = enum_rocprof_trace_decoder_codeobj_marker_type_t.define('ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_SIZE_LO', 1)
ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ADDR_LO = enum_rocprof_trace_decoder_codeobj_marker_type_t.define('ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ADDR_LO', 2)
ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ADDR_HI = enum_rocprof_trace_decoder_codeobj_marker_type_t.define('ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ADDR_HI', 3)
ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_SIZE_HI = enum_rocprof_trace_decoder_codeobj_marker_type_t.define('ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_SIZE_HI', 4)
ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ID_LO = enum_rocprof_trace_decoder_codeobj_marker_type_t.define('ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ID_LO', 5)
ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ID_HI = enum_rocprof_trace_decoder_codeobj_marker_type_t.define('ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ID_HI', 6)
ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_LAST = enum_rocprof_trace_decoder_codeobj_marker_type_t.define('ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_LAST', 7)

rocprof_trace_decoder_codeobj_marker_type_t = enum_rocprof_trace_decoder_codeobj_marker_type_t
rocprofiler_thread_trace_decoder_pc_t = struct_rocprofiler_thread_trace_decoder_pc_t
class struct_rocprofiler_thread_trace_decoder_perfevent_t(Struct): pass
int64_t = ctypes.c_int64
uint16_t = ctypes.c_uint16
uint8_t = ctypes.c_ubyte
struct_rocprofiler_thread_trace_decoder_perfevent_t._fields_ = [
  ('time', int64_t),
  ('events0', uint16_t),
  ('events1', uint16_t),
  ('events2', uint16_t),
  ('events3', uint16_t),
  ('CU', uint8_t),
  ('bank', uint8_t),
]
rocprofiler_thread_trace_decoder_perfevent_t = struct_rocprofiler_thread_trace_decoder_perfevent_t
class struct_rocprofiler_thread_trace_decoder_occupancy_t(Struct): pass
struct_rocprofiler_thread_trace_decoder_occupancy_t._fields_ = [
  ('pc', rocprofiler_thread_trace_decoder_pc_t),
  ('time', uint64_t),
  ('reserved', uint8_t),
  ('cu', uint8_t),
  ('simd', uint8_t),
  ('wave_id', uint8_t),
  ('start', uint32_t,1),
  ('_rsvd', uint32_t,31),
]
rocprofiler_thread_trace_decoder_occupancy_t = struct_rocprofiler_thread_trace_decoder_occupancy_t
enum_rocprofiler_thread_trace_decoder_wstate_type_t = CEnum(ctypes.c_uint32)
ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_EMPTY = enum_rocprofiler_thread_trace_decoder_wstate_type_t.define('ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_EMPTY', 0)
ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_IDLE = enum_rocprofiler_thread_trace_decoder_wstate_type_t.define('ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_IDLE', 1)
ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_EXEC = enum_rocprofiler_thread_trace_decoder_wstate_type_t.define('ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_EXEC', 2)
ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_WAIT = enum_rocprofiler_thread_trace_decoder_wstate_type_t.define('ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_WAIT', 3)
ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_STALL = enum_rocprofiler_thread_trace_decoder_wstate_type_t.define('ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_STALL', 4)
ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_LAST = enum_rocprofiler_thread_trace_decoder_wstate_type_t.define('ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_LAST', 5)

rocprofiler_thread_trace_decoder_wstate_type_t = enum_rocprofiler_thread_trace_decoder_wstate_type_t
class struct_rocprofiler_thread_trace_decoder_wave_state_t(Struct): pass
int32_t = ctypes.c_int32
struct_rocprofiler_thread_trace_decoder_wave_state_t._fields_ = [
  ('type', int32_t),
  ('duration', int32_t),
]
rocprofiler_thread_trace_decoder_wave_state_t = struct_rocprofiler_thread_trace_decoder_wave_state_t
enum_rocprofiler_thread_trace_decoder_inst_category_t = CEnum(ctypes.c_uint32)
ROCPROFILER_THREAD_TRACE_DECODER_INST_NONE = enum_rocprofiler_thread_trace_decoder_inst_category_t.define('ROCPROFILER_THREAD_TRACE_DECODER_INST_NONE', 0)
ROCPROFILER_THREAD_TRACE_DECODER_INST_SMEM = enum_rocprofiler_thread_trace_decoder_inst_category_t.define('ROCPROFILER_THREAD_TRACE_DECODER_INST_SMEM', 1)
ROCPROFILER_THREAD_TRACE_DECODER_INST_SALU = enum_rocprofiler_thread_trace_decoder_inst_category_t.define('ROCPROFILER_THREAD_TRACE_DECODER_INST_SALU', 2)
ROCPROFILER_THREAD_TRACE_DECODER_INST_VMEM = enum_rocprofiler_thread_trace_decoder_inst_category_t.define('ROCPROFILER_THREAD_TRACE_DECODER_INST_VMEM', 3)
ROCPROFILER_THREAD_TRACE_DECODER_INST_FLAT = enum_rocprofiler_thread_trace_decoder_inst_category_t.define('ROCPROFILER_THREAD_TRACE_DECODER_INST_FLAT', 4)
ROCPROFILER_THREAD_TRACE_DECODER_INST_LDS = enum_rocprofiler_thread_trace_decoder_inst_category_t.define('ROCPROFILER_THREAD_TRACE_DECODER_INST_LDS', 5)
ROCPROFILER_THREAD_TRACE_DECODER_INST_VALU = enum_rocprofiler_thread_trace_decoder_inst_category_t.define('ROCPROFILER_THREAD_TRACE_DECODER_INST_VALU', 6)
ROCPROFILER_THREAD_TRACE_DECODER_INST_JUMP = enum_rocprofiler_thread_trace_decoder_inst_category_t.define('ROCPROFILER_THREAD_TRACE_DECODER_INST_JUMP', 7)
ROCPROFILER_THREAD_TRACE_DECODER_INST_NEXT = enum_rocprofiler_thread_trace_decoder_inst_category_t.define('ROCPROFILER_THREAD_TRACE_DECODER_INST_NEXT', 8)
ROCPROFILER_THREAD_TRACE_DECODER_INST_IMMED = enum_rocprofiler_thread_trace_decoder_inst_category_t.define('ROCPROFILER_THREAD_TRACE_DECODER_INST_IMMED', 9)
ROCPROFILER_THREAD_TRACE_DECODER_INST_CONTEXT = enum_rocprofiler_thread_trace_decoder_inst_category_t.define('ROCPROFILER_THREAD_TRACE_DECODER_INST_CONTEXT', 10)
ROCPROFILER_THREAD_TRACE_DECODER_INST_MESSAGE = enum_rocprofiler_thread_trace_decoder_inst_category_t.define('ROCPROFILER_THREAD_TRACE_DECODER_INST_MESSAGE', 11)
ROCPROFILER_THREAD_TRACE_DECODER_INST_BVH = enum_rocprofiler_thread_trace_decoder_inst_category_t.define('ROCPROFILER_THREAD_TRACE_DECODER_INST_BVH', 12)
ROCPROFILER_THREAD_TRACE_DECODER_INST_LAST = enum_rocprofiler_thread_trace_decoder_inst_category_t.define('ROCPROFILER_THREAD_TRACE_DECODER_INST_LAST', 13)

rocprofiler_thread_trace_decoder_inst_category_t = enum_rocprofiler_thread_trace_decoder_inst_category_t
class struct_rocprofiler_thread_trace_decoder_inst_t(Struct): pass
struct_rocprofiler_thread_trace_decoder_inst_t._fields_ = [
  ('category', uint32_t,8),
  ('stall', uint32_t,24),
  ('duration', int32_t),
  ('time', int64_t),
  ('pc', rocprofiler_thread_trace_decoder_pc_t),
]
rocprofiler_thread_trace_decoder_inst_t = struct_rocprofiler_thread_trace_decoder_inst_t
class struct_rocprofiler_thread_trace_decoder_wave_t(Struct): pass
struct_rocprofiler_thread_trace_decoder_wave_t._fields_ = [
  ('cu', uint8_t),
  ('simd', uint8_t),
  ('wave_id', uint8_t),
  ('contexts', uint8_t),
  ('_rsvd1', uint32_t),
  ('_rsvd2', uint32_t),
  ('_rsvd3', uint32_t),
  ('begin_time', int64_t),
  ('end_time', int64_t),
  ('timeline_size', uint64_t),
  ('instructions_size', uint64_t),
  ('timeline_array', ctypes.POINTER(rocprofiler_thread_trace_decoder_wave_state_t)),
  ('instructions_array', ctypes.POINTER(rocprofiler_thread_trace_decoder_inst_t)),
]
rocprofiler_thread_trace_decoder_wave_t = struct_rocprofiler_thread_trace_decoder_wave_t
class struct_rocprofiler_thread_trace_decoder_realtime_t(Struct): pass
struct_rocprofiler_thread_trace_decoder_realtime_t._fields_ = [
  ('shader_clock', int64_t),
  ('realtime_clock', uint64_t),
  ('reserved', uint64_t),
]
rocprofiler_thread_trace_decoder_realtime_t = struct_rocprofiler_thread_trace_decoder_realtime_t
enum_rocprofiler_thread_trace_decoder_shaderdata_flags_t = CEnum(ctypes.c_uint32)
ROCPROFILER_THREAD_TRACE_DECODER_SHADERDATA_FLAGS_IMM = enum_rocprofiler_thread_trace_decoder_shaderdata_flags_t.define('ROCPROFILER_THREAD_TRACE_DECODER_SHADERDATA_FLAGS_IMM', 0)
ROCPROFILER_THREAD_TRACE_DECODER_SHADERDATA_FLAGS_PRIV = enum_rocprofiler_thread_trace_decoder_shaderdata_flags_t.define('ROCPROFILER_THREAD_TRACE_DECODER_SHADERDATA_FLAGS_PRIV', 1)

rocprofiler_thread_trace_decoder_shaderdata_flags_t = enum_rocprofiler_thread_trace_decoder_shaderdata_flags_t
class struct_rocprofiler_thread_trace_decoder_shaderdata_t(Struct): pass
struct_rocprofiler_thread_trace_decoder_shaderdata_t._fields_ = [
  ('time', int64_t),
  ('value', uint64_t),
  ('cu', uint8_t),
  ('simd', uint8_t),
  ('wave_id', uint8_t),
  ('flags', uint8_t),
  ('reserved', uint32_t),
]
rocprofiler_thread_trace_decoder_shaderdata_t = struct_rocprofiler_thread_trace_decoder_shaderdata_t
rocprofiler_thread_trace_decoder_record_type_t = enum_rocprofiler_thread_trace_decoder_record_type_t
