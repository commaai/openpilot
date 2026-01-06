# mypy: ignore-errors
import ctypes
from tinygrad.runtime.support.c import DLL, Struct, CEnum, _IO, _IOW, _IOR, _IOWR
class struct_sqtt_data_info(Struct): pass
uint32_t = ctypes.c_uint32
class struct_sqtt_data_info_0(ctypes.Union): pass
struct_sqtt_data_info_0._fields_ = [
  ('gfx9_write_counter', uint32_t),
  ('gfx10_dropped_cntr', uint32_t),
]
struct_sqtt_data_info._anonymous_ = ['_0']
struct_sqtt_data_info._fields_ = [
  ('cur_offset', uint32_t),
  ('trace_status', uint32_t),
  ('_0', struct_sqtt_data_info_0),
]
class struct_sqtt_data_se(Struct): pass
struct_sqtt_data_se._fields_ = [
  ('info', struct_sqtt_data_info),
  ('data_ptr', ctypes.c_void_p),
  ('shader_engine', uint32_t),
  ('compute_unit', uint32_t),
]
enum_sqtt_version = CEnum(ctypes.c_uint32)
SQTT_VERSION_NONE = enum_sqtt_version.define('SQTT_VERSION_NONE', 0)
SQTT_VERSION_2_2 = enum_sqtt_version.define('SQTT_VERSION_2_2', 5)
SQTT_VERSION_2_3 = enum_sqtt_version.define('SQTT_VERSION_2_3', 6)
SQTT_VERSION_2_4 = enum_sqtt_version.define('SQTT_VERSION_2_4', 7)
SQTT_VERSION_3_2 = enum_sqtt_version.define('SQTT_VERSION_3_2', 11)
SQTT_VERSION_3_3 = enum_sqtt_version.define('SQTT_VERSION_3_3', 12)

enum_sqtt_file_chunk_type = CEnum(ctypes.c_uint32)
SQTT_FILE_CHUNK_TYPE_ASIC_INFO = enum_sqtt_file_chunk_type.define('SQTT_FILE_CHUNK_TYPE_ASIC_INFO', 0)
SQTT_FILE_CHUNK_TYPE_SQTT_DESC = enum_sqtt_file_chunk_type.define('SQTT_FILE_CHUNK_TYPE_SQTT_DESC', 1)
SQTT_FILE_CHUNK_TYPE_SQTT_DATA = enum_sqtt_file_chunk_type.define('SQTT_FILE_CHUNK_TYPE_SQTT_DATA', 2)
SQTT_FILE_CHUNK_TYPE_API_INFO = enum_sqtt_file_chunk_type.define('SQTT_FILE_CHUNK_TYPE_API_INFO', 3)
SQTT_FILE_CHUNK_TYPE_RESERVED = enum_sqtt_file_chunk_type.define('SQTT_FILE_CHUNK_TYPE_RESERVED', 4)
SQTT_FILE_CHUNK_TYPE_QUEUE_EVENT_TIMINGS = enum_sqtt_file_chunk_type.define('SQTT_FILE_CHUNK_TYPE_QUEUE_EVENT_TIMINGS', 5)
SQTT_FILE_CHUNK_TYPE_CLOCK_CALIBRATION = enum_sqtt_file_chunk_type.define('SQTT_FILE_CHUNK_TYPE_CLOCK_CALIBRATION', 6)
SQTT_FILE_CHUNK_TYPE_CPU_INFO = enum_sqtt_file_chunk_type.define('SQTT_FILE_CHUNK_TYPE_CPU_INFO', 7)
SQTT_FILE_CHUNK_TYPE_SPM_DB = enum_sqtt_file_chunk_type.define('SQTT_FILE_CHUNK_TYPE_SPM_DB', 8)
SQTT_FILE_CHUNK_TYPE_CODE_OBJECT_DATABASE = enum_sqtt_file_chunk_type.define('SQTT_FILE_CHUNK_TYPE_CODE_OBJECT_DATABASE', 9)
SQTT_FILE_CHUNK_TYPE_CODE_OBJECT_LOADER_EVENTS = enum_sqtt_file_chunk_type.define('SQTT_FILE_CHUNK_TYPE_CODE_OBJECT_LOADER_EVENTS', 10)
SQTT_FILE_CHUNK_TYPE_PSO_CORRELATION = enum_sqtt_file_chunk_type.define('SQTT_FILE_CHUNK_TYPE_PSO_CORRELATION', 11)
SQTT_FILE_CHUNK_TYPE_INSTRUMENTATION_TABLE = enum_sqtt_file_chunk_type.define('SQTT_FILE_CHUNK_TYPE_INSTRUMENTATION_TABLE', 12)
SQTT_FILE_CHUNK_TYPE_COUNT = enum_sqtt_file_chunk_type.define('SQTT_FILE_CHUNK_TYPE_COUNT', 13)

class struct_sqtt_file_chunk_id(Struct): pass
int32_t = ctypes.c_int32
struct_sqtt_file_chunk_id._fields_ = [
  ('type', int32_t,8),
  ('index', int32_t,8),
  ('reserved', int32_t,16),
]
class struct_sqtt_file_chunk_header(Struct): pass
uint16_t = ctypes.c_uint16
struct_sqtt_file_chunk_header._fields_ = [
  ('chunk_id', struct_sqtt_file_chunk_id),
  ('minor_version', uint16_t),
  ('major_version', uint16_t),
  ('size_in_bytes', int32_t),
  ('padding', int32_t),
]
class struct_sqtt_file_header_flags(Struct): pass
class struct_sqtt_file_header_flags_0(ctypes.Union): pass
class struct_sqtt_file_header_flags_0_0(Struct): pass
struct_sqtt_file_header_flags_0_0._fields_ = [
  ('is_semaphore_queue_timing_etw', uint32_t,1),
  ('no_queue_semaphore_timestamps', uint32_t,1),
  ('reserved', uint32_t,30),
]
struct_sqtt_file_header_flags_0._anonymous_ = ['_0']
struct_sqtt_file_header_flags_0._fields_ = [
  ('_0', struct_sqtt_file_header_flags_0_0),
  ('value', uint32_t),
]
struct_sqtt_file_header_flags._anonymous_ = ['_0']
struct_sqtt_file_header_flags._fields_ = [
  ('_0', struct_sqtt_file_header_flags_0),
]
class struct_sqtt_file_header(Struct): pass
struct_sqtt_file_header._fields_ = [
  ('magic_number', uint32_t),
  ('version_major', uint32_t),
  ('version_minor', uint32_t),
  ('flags', struct_sqtt_file_header_flags),
  ('chunk_offset', int32_t),
  ('second', int32_t),
  ('minute', int32_t),
  ('hour', int32_t),
  ('day_in_month', int32_t),
  ('month', int32_t),
  ('year', int32_t),
  ('day_in_week', int32_t),
  ('day_in_year', int32_t),
  ('is_daylight_savings', int32_t),
]
class struct_sqtt_file_chunk_cpu_info(Struct): pass
uint64_t = ctypes.c_uint64
struct_sqtt_file_chunk_cpu_info._fields_ = [
  ('header', struct_sqtt_file_chunk_header),
  ('vendor_id', (uint32_t * 4)),
  ('processor_brand', (uint32_t * 12)),
  ('reserved', (uint32_t * 2)),
  ('cpu_timestamp_freq', uint64_t),
  ('clock_speed', uint32_t),
  ('num_logical_cores', uint32_t),
  ('num_physical_cores', uint32_t),
  ('system_ram_size', uint32_t),
]
enum_sqtt_file_chunk_asic_info_flags = CEnum(ctypes.c_uint32)
SQTT_FILE_CHUNK_ASIC_INFO_FLAG_SC_PACKER_NUMBERING = enum_sqtt_file_chunk_asic_info_flags.define('SQTT_FILE_CHUNK_ASIC_INFO_FLAG_SC_PACKER_NUMBERING', 1)
SQTT_FILE_CHUNK_ASIC_INFO_FLAG_PS1_EVENT_TOKENS_ENABLED = enum_sqtt_file_chunk_asic_info_flags.define('SQTT_FILE_CHUNK_ASIC_INFO_FLAG_PS1_EVENT_TOKENS_ENABLED', 2)

enum_sqtt_gpu_type = CEnum(ctypes.c_uint32)
SQTT_GPU_TYPE_UNKNOWN = enum_sqtt_gpu_type.define('SQTT_GPU_TYPE_UNKNOWN', 0)
SQTT_GPU_TYPE_INTEGRATED = enum_sqtt_gpu_type.define('SQTT_GPU_TYPE_INTEGRATED', 1)
SQTT_GPU_TYPE_DISCRETE = enum_sqtt_gpu_type.define('SQTT_GPU_TYPE_DISCRETE', 2)
SQTT_GPU_TYPE_VIRTUAL = enum_sqtt_gpu_type.define('SQTT_GPU_TYPE_VIRTUAL', 3)

enum_sqtt_gfxip_level = CEnum(ctypes.c_uint32)
SQTT_GFXIP_LEVEL_NONE = enum_sqtt_gfxip_level.define('SQTT_GFXIP_LEVEL_NONE', 0)
SQTT_GFXIP_LEVEL_GFXIP_6 = enum_sqtt_gfxip_level.define('SQTT_GFXIP_LEVEL_GFXIP_6', 1)
SQTT_GFXIP_LEVEL_GFXIP_7 = enum_sqtt_gfxip_level.define('SQTT_GFXIP_LEVEL_GFXIP_7', 2)
SQTT_GFXIP_LEVEL_GFXIP_8 = enum_sqtt_gfxip_level.define('SQTT_GFXIP_LEVEL_GFXIP_8', 3)
SQTT_GFXIP_LEVEL_GFXIP_8_1 = enum_sqtt_gfxip_level.define('SQTT_GFXIP_LEVEL_GFXIP_8_1', 4)
SQTT_GFXIP_LEVEL_GFXIP_9 = enum_sqtt_gfxip_level.define('SQTT_GFXIP_LEVEL_GFXIP_9', 5)
SQTT_GFXIP_LEVEL_GFXIP_10_1 = enum_sqtt_gfxip_level.define('SQTT_GFXIP_LEVEL_GFXIP_10_1', 7)
SQTT_GFXIP_LEVEL_GFXIP_10_3 = enum_sqtt_gfxip_level.define('SQTT_GFXIP_LEVEL_GFXIP_10_3', 9)
SQTT_GFXIP_LEVEL_GFXIP_11_0 = enum_sqtt_gfxip_level.define('SQTT_GFXIP_LEVEL_GFXIP_11_0', 12)
SQTT_GFXIP_LEVEL_GFXIP_11_5 = enum_sqtt_gfxip_level.define('SQTT_GFXIP_LEVEL_GFXIP_11_5', 13)
SQTT_GFXIP_LEVEL_GFXIP_12 = enum_sqtt_gfxip_level.define('SQTT_GFXIP_LEVEL_GFXIP_12', 16)

enum_sqtt_memory_type = CEnum(ctypes.c_uint32)
SQTT_MEMORY_TYPE_UNKNOWN = enum_sqtt_memory_type.define('SQTT_MEMORY_TYPE_UNKNOWN', 0)
SQTT_MEMORY_TYPE_DDR = enum_sqtt_memory_type.define('SQTT_MEMORY_TYPE_DDR', 1)
SQTT_MEMORY_TYPE_DDR2 = enum_sqtt_memory_type.define('SQTT_MEMORY_TYPE_DDR2', 2)
SQTT_MEMORY_TYPE_DDR3 = enum_sqtt_memory_type.define('SQTT_MEMORY_TYPE_DDR3', 3)
SQTT_MEMORY_TYPE_DDR4 = enum_sqtt_memory_type.define('SQTT_MEMORY_TYPE_DDR4', 4)
SQTT_MEMORY_TYPE_DDR5 = enum_sqtt_memory_type.define('SQTT_MEMORY_TYPE_DDR5', 5)
SQTT_MEMORY_TYPE_GDDR3 = enum_sqtt_memory_type.define('SQTT_MEMORY_TYPE_GDDR3', 16)
SQTT_MEMORY_TYPE_GDDR4 = enum_sqtt_memory_type.define('SQTT_MEMORY_TYPE_GDDR4', 17)
SQTT_MEMORY_TYPE_GDDR5 = enum_sqtt_memory_type.define('SQTT_MEMORY_TYPE_GDDR5', 18)
SQTT_MEMORY_TYPE_GDDR6 = enum_sqtt_memory_type.define('SQTT_MEMORY_TYPE_GDDR6', 19)
SQTT_MEMORY_TYPE_HBM = enum_sqtt_memory_type.define('SQTT_MEMORY_TYPE_HBM', 32)
SQTT_MEMORY_TYPE_HBM2 = enum_sqtt_memory_type.define('SQTT_MEMORY_TYPE_HBM2', 33)
SQTT_MEMORY_TYPE_HBM3 = enum_sqtt_memory_type.define('SQTT_MEMORY_TYPE_HBM3', 34)
SQTT_MEMORY_TYPE_LPDDR4 = enum_sqtt_memory_type.define('SQTT_MEMORY_TYPE_LPDDR4', 48)
SQTT_MEMORY_TYPE_LPDDR5 = enum_sqtt_memory_type.define('SQTT_MEMORY_TYPE_LPDDR5', 49)

class struct_sqtt_file_chunk_asic_info(Struct): pass
int64_t = ctypes.c_int64
struct_sqtt_file_chunk_asic_info._fields_ = [
  ('header', struct_sqtt_file_chunk_header),
  ('flags', uint64_t),
  ('trace_shader_core_clock', uint64_t),
  ('trace_memory_clock', uint64_t),
  ('device_id', int32_t),
  ('device_revision_id', int32_t),
  ('vgprs_per_simd', int32_t),
  ('sgprs_per_simd', int32_t),
  ('shader_engines', int32_t),
  ('compute_unit_per_shader_engine', int32_t),
  ('simd_per_compute_unit', int32_t),
  ('wavefronts_per_simd', int32_t),
  ('minimum_vgpr_alloc', int32_t),
  ('vgpr_alloc_granularity', int32_t),
  ('minimum_sgpr_alloc', int32_t),
  ('sgpr_alloc_granularity', int32_t),
  ('hardware_contexts', int32_t),
  ('gpu_type', enum_sqtt_gpu_type),
  ('gfxip_level', enum_sqtt_gfxip_level),
  ('gpu_index', int32_t),
  ('gds_size', int32_t),
  ('gds_per_shader_engine', int32_t),
  ('ce_ram_size', int32_t),
  ('ce_ram_size_graphics', int32_t),
  ('ce_ram_size_compute', int32_t),
  ('max_number_of_dedicated_cus', int32_t),
  ('vram_size', int64_t),
  ('vram_bus_width', int32_t),
  ('l2_cache_size', int32_t),
  ('l1_cache_size', int32_t),
  ('lds_size', int32_t),
  ('gpu_name', (ctypes.c_char * 256)),
  ('alu_per_clock', ctypes.c_float),
  ('texture_per_clock', ctypes.c_float),
  ('prims_per_clock', ctypes.c_float),
  ('pixels_per_clock', ctypes.c_float),
  ('gpu_timestamp_frequency', uint64_t),
  ('max_shader_core_clock', uint64_t),
  ('max_memory_clock', uint64_t),
  ('memory_ops_per_clock', uint32_t),
  ('memory_chip_type', enum_sqtt_memory_type),
  ('lds_granularity', uint32_t),
  ('cu_mask', ((uint16_t * 2) * 32)),
  ('reserved1', (ctypes.c_char * 128)),
  ('active_pixel_packer_mask', (uint32_t * 4)),
  ('reserved2', (ctypes.c_char * 16)),
  ('gl1_cache_size', uint32_t),
  ('instruction_cache_size', uint32_t),
  ('scalar_cache_size', uint32_t),
  ('mall_cache_size', uint32_t),
  ('padding', (ctypes.c_char * 4)),
]
enum_sqtt_api_type = CEnum(ctypes.c_uint32)
SQTT_API_TYPE_DIRECTX_12 = enum_sqtt_api_type.define('SQTT_API_TYPE_DIRECTX_12', 0)
SQTT_API_TYPE_VULKAN = enum_sqtt_api_type.define('SQTT_API_TYPE_VULKAN', 1)
SQTT_API_TYPE_GENERIC = enum_sqtt_api_type.define('SQTT_API_TYPE_GENERIC', 2)
SQTT_API_TYPE_OPENCL = enum_sqtt_api_type.define('SQTT_API_TYPE_OPENCL', 3)

enum_sqtt_instruction_trace_mode = CEnum(ctypes.c_uint32)
SQTT_INSTRUCTION_TRACE_DISABLED = enum_sqtt_instruction_trace_mode.define('SQTT_INSTRUCTION_TRACE_DISABLED', 0)
SQTT_INSTRUCTION_TRACE_FULL_FRAME = enum_sqtt_instruction_trace_mode.define('SQTT_INSTRUCTION_TRACE_FULL_FRAME', 1)
SQTT_INSTRUCTION_TRACE_API_PSO = enum_sqtt_instruction_trace_mode.define('SQTT_INSTRUCTION_TRACE_API_PSO', 2)

enum_sqtt_profiling_mode = CEnum(ctypes.c_uint32)
SQTT_PROFILING_MODE_PRESENT = enum_sqtt_profiling_mode.define('SQTT_PROFILING_MODE_PRESENT', 0)
SQTT_PROFILING_MODE_USER_MARKERS = enum_sqtt_profiling_mode.define('SQTT_PROFILING_MODE_USER_MARKERS', 1)
SQTT_PROFILING_MODE_INDEX = enum_sqtt_profiling_mode.define('SQTT_PROFILING_MODE_INDEX', 2)
SQTT_PROFILING_MODE_TAG = enum_sqtt_profiling_mode.define('SQTT_PROFILING_MODE_TAG', 3)

class union_sqtt_profiling_mode_data(ctypes.Union): pass
class union_sqtt_profiling_mode_data_user_marker_profiling_data(Struct): pass
union_sqtt_profiling_mode_data_user_marker_profiling_data._fields_ = [
  ('start', (ctypes.c_char * 256)),
  ('end', (ctypes.c_char * 256)),
]
class union_sqtt_profiling_mode_data_index_profiling_data(Struct): pass
union_sqtt_profiling_mode_data_index_profiling_data._fields_ = [
  ('start', uint32_t),
  ('end', uint32_t),
]
class union_sqtt_profiling_mode_data_tag_profiling_data(Struct): pass
union_sqtt_profiling_mode_data_tag_profiling_data._fields_ = [
  ('begin_hi', uint32_t),
  ('begin_lo', uint32_t),
  ('end_hi', uint32_t),
  ('end_lo', uint32_t),
]
union_sqtt_profiling_mode_data._fields_ = [
  ('user_marker_profiling_data', union_sqtt_profiling_mode_data_user_marker_profiling_data),
  ('index_profiling_data', union_sqtt_profiling_mode_data_index_profiling_data),
  ('tag_profiling_data', union_sqtt_profiling_mode_data_tag_profiling_data),
]
class union_sqtt_instruction_trace_data(ctypes.Union): pass
class union_sqtt_instruction_trace_data_api_pso_data(Struct): pass
union_sqtt_instruction_trace_data_api_pso_data._fields_ = [
  ('api_pso_filter', uint64_t),
]
class union_sqtt_instruction_trace_data_shader_engine_filter(Struct): pass
union_sqtt_instruction_trace_data_shader_engine_filter._fields_ = [
  ('mask', uint32_t),
]
union_sqtt_instruction_trace_data._fields_ = [
  ('api_pso_data', union_sqtt_instruction_trace_data_api_pso_data),
  ('shader_engine_filter', union_sqtt_instruction_trace_data_shader_engine_filter),
]
class struct_sqtt_file_chunk_api_info(Struct): pass
struct_sqtt_file_chunk_api_info._fields_ = [
  ('header', struct_sqtt_file_chunk_header),
  ('api_type', enum_sqtt_api_type),
  ('major_version', uint16_t),
  ('minor_version', uint16_t),
  ('profiling_mode', enum_sqtt_profiling_mode),
  ('reserved', uint32_t),
  ('profiling_mode_data', union_sqtt_profiling_mode_data),
  ('instruction_trace_mode', enum_sqtt_instruction_trace_mode),
  ('reserved2', uint32_t),
  ('instruction_trace_data', union_sqtt_instruction_trace_data),
]
class struct_sqtt_code_object_database_record(Struct): pass
struct_sqtt_code_object_database_record._fields_ = [
  ('size', uint32_t),
]
class struct_sqtt_file_chunk_code_object_database(Struct): pass
struct_sqtt_file_chunk_code_object_database._fields_ = [
  ('header', struct_sqtt_file_chunk_header),
  ('offset', uint32_t),
  ('flags', uint32_t),
  ('size', uint32_t),
  ('record_count', uint32_t),
]
class struct_sqtt_code_object_loader_events_record(Struct): pass
struct_sqtt_code_object_loader_events_record._fields_ = [
  ('loader_event_type', uint32_t),
  ('reserved', uint32_t),
  ('base_address', uint64_t),
  ('code_object_hash', (uint64_t * 2)),
  ('time_stamp', uint64_t),
]
class struct_sqtt_file_chunk_code_object_loader_events(Struct): pass
struct_sqtt_file_chunk_code_object_loader_events._fields_ = [
  ('header', struct_sqtt_file_chunk_header),
  ('offset', uint32_t),
  ('flags', uint32_t),
  ('record_size', uint32_t),
  ('record_count', uint32_t),
]
class struct_sqtt_pso_correlation_record(Struct): pass
struct_sqtt_pso_correlation_record._fields_ = [
  ('api_pso_hash', uint64_t),
  ('pipeline_hash', (uint64_t * 2)),
  ('api_level_obj_name', (ctypes.c_char * 64)),
]
class struct_sqtt_file_chunk_pso_correlation(Struct): pass
struct_sqtt_file_chunk_pso_correlation._fields_ = [
  ('header', struct_sqtt_file_chunk_header),
  ('offset', uint32_t),
  ('flags', uint32_t),
  ('record_size', uint32_t),
  ('record_count', uint32_t),
]
class struct_sqtt_file_chunk_sqtt_desc(Struct): pass
class struct_sqtt_file_chunk_sqtt_desc_0(ctypes.Union): pass
class struct_sqtt_file_chunk_sqtt_desc_0_v0(Struct): pass
struct_sqtt_file_chunk_sqtt_desc_0_v0._fields_ = [
  ('instrumentation_version', int32_t),
]
class struct_sqtt_file_chunk_sqtt_desc_0_v1(Struct): pass
int16_t = ctypes.c_int16
struct_sqtt_file_chunk_sqtt_desc_0_v1._fields_ = [
  ('instrumentation_spec_version', int16_t),
  ('instrumentation_api_version', int16_t),
  ('compute_unit_index', int32_t),
]
struct_sqtt_file_chunk_sqtt_desc_0._fields_ = [
  ('v0', struct_sqtt_file_chunk_sqtt_desc_0_v0),
  ('v1', struct_sqtt_file_chunk_sqtt_desc_0_v1),
]
struct_sqtt_file_chunk_sqtt_desc._anonymous_ = ['_0']
struct_sqtt_file_chunk_sqtt_desc._fields_ = [
  ('header', struct_sqtt_file_chunk_header),
  ('shader_engine_index', int32_t),
  ('sqtt_version', enum_sqtt_version),
  ('_0', struct_sqtt_file_chunk_sqtt_desc_0),
]
class struct_sqtt_file_chunk_sqtt_data(Struct): pass
struct_sqtt_file_chunk_sqtt_data._fields_ = [
  ('header', struct_sqtt_file_chunk_header),
  ('offset', int32_t),
  ('size', int32_t),
]
class struct_sqtt_file_chunk_queue_event_timings(Struct): pass
struct_sqtt_file_chunk_queue_event_timings._fields_ = [
  ('header', struct_sqtt_file_chunk_header),
  ('queue_info_table_record_count', uint32_t),
  ('queue_info_table_size', uint32_t),
  ('queue_event_table_record_count', uint32_t),
  ('queue_event_table_size', uint32_t),
]
enum_sqtt_queue_type = CEnum(ctypes.c_uint32)
SQTT_QUEUE_TYPE_UNKNOWN = enum_sqtt_queue_type.define('SQTT_QUEUE_TYPE_UNKNOWN', 0)
SQTT_QUEUE_TYPE_UNIVERSAL = enum_sqtt_queue_type.define('SQTT_QUEUE_TYPE_UNIVERSAL', 1)
SQTT_QUEUE_TYPE_COMPUTE = enum_sqtt_queue_type.define('SQTT_QUEUE_TYPE_COMPUTE', 2)
SQTT_QUEUE_TYPE_DMA = enum_sqtt_queue_type.define('SQTT_QUEUE_TYPE_DMA', 3)

enum_sqtt_engine_type = CEnum(ctypes.c_uint32)
SQTT_ENGINE_TYPE_UNKNOWN = enum_sqtt_engine_type.define('SQTT_ENGINE_TYPE_UNKNOWN', 0)
SQTT_ENGINE_TYPE_UNIVERSAL = enum_sqtt_engine_type.define('SQTT_ENGINE_TYPE_UNIVERSAL', 1)
SQTT_ENGINE_TYPE_COMPUTE = enum_sqtt_engine_type.define('SQTT_ENGINE_TYPE_COMPUTE', 2)
SQTT_ENGINE_TYPE_EXCLUSIVE_COMPUTE = enum_sqtt_engine_type.define('SQTT_ENGINE_TYPE_EXCLUSIVE_COMPUTE', 3)
SQTT_ENGINE_TYPE_DMA = enum_sqtt_engine_type.define('SQTT_ENGINE_TYPE_DMA', 4)
SQTT_ENGINE_TYPE_HIGH_PRIORITY_UNIVERSAL = enum_sqtt_engine_type.define('SQTT_ENGINE_TYPE_HIGH_PRIORITY_UNIVERSAL', 7)
SQTT_ENGINE_TYPE_HIGH_PRIORITY_GRAPHICS = enum_sqtt_engine_type.define('SQTT_ENGINE_TYPE_HIGH_PRIORITY_GRAPHICS', 8)

class struct_sqtt_queue_hardware_info(Struct): pass
class struct_sqtt_queue_hardware_info_0(ctypes.Union): pass
class struct_sqtt_queue_hardware_info_0_0(Struct): pass
struct_sqtt_queue_hardware_info_0_0._fields_ = [
  ('queue_type', int32_t,8),
  ('engine_type', int32_t,8),
  ('reserved', uint32_t,16),
]
struct_sqtt_queue_hardware_info_0._anonymous_ = ['_0']
struct_sqtt_queue_hardware_info_0._fields_ = [
  ('_0', struct_sqtt_queue_hardware_info_0_0),
  ('value', uint32_t),
]
struct_sqtt_queue_hardware_info._anonymous_ = ['_0']
struct_sqtt_queue_hardware_info._fields_ = [
  ('_0', struct_sqtt_queue_hardware_info_0),
]
class struct_sqtt_queue_info_record(Struct): pass
struct_sqtt_queue_info_record._fields_ = [
  ('queue_id', uint64_t),
  ('queue_context', uint64_t),
  ('hardware_info', struct_sqtt_queue_hardware_info),
  ('reserved', uint32_t),
]
enum_sqtt_queue_event_type = CEnum(ctypes.c_uint32)
SQTT_QUEUE_TIMING_EVENT_CMDBUF_SUBMIT = enum_sqtt_queue_event_type.define('SQTT_QUEUE_TIMING_EVENT_CMDBUF_SUBMIT', 0)
SQTT_QUEUE_TIMING_EVENT_SIGNAL_SEMAPHORE = enum_sqtt_queue_event_type.define('SQTT_QUEUE_TIMING_EVENT_SIGNAL_SEMAPHORE', 1)
SQTT_QUEUE_TIMING_EVENT_WAIT_SEMAPHORE = enum_sqtt_queue_event_type.define('SQTT_QUEUE_TIMING_EVENT_WAIT_SEMAPHORE', 2)
SQTT_QUEUE_TIMING_EVENT_PRESENT = enum_sqtt_queue_event_type.define('SQTT_QUEUE_TIMING_EVENT_PRESENT', 3)

class struct_sqtt_queue_event_record(Struct): pass
struct_sqtt_queue_event_record._fields_ = [
  ('event_type', enum_sqtt_queue_event_type),
  ('sqtt_cb_id', uint32_t),
  ('frame_index', uint64_t),
  ('queue_info_index', uint32_t),
  ('submit_sub_index', uint32_t),
  ('api_id', uint64_t),
  ('cpu_timestamp', uint64_t),
  ('gpu_timestamps', (uint64_t * 2)),
]
class struct_sqtt_file_chunk_clock_calibration(Struct): pass
struct_sqtt_file_chunk_clock_calibration._fields_ = [
  ('header', struct_sqtt_file_chunk_header),
  ('cpu_timestamp', uint64_t),
  ('gpu_timestamp', uint64_t),
  ('reserved', uint64_t),
]
enum_elf_gfxip_level = CEnum(ctypes.c_uint32)
EF_AMDGPU_MACH_AMDGCN_GFX801 = enum_elf_gfxip_level.define('EF_AMDGPU_MACH_AMDGCN_GFX801', 40)
EF_AMDGPU_MACH_AMDGCN_GFX900 = enum_elf_gfxip_level.define('EF_AMDGPU_MACH_AMDGCN_GFX900', 44)
EF_AMDGPU_MACH_AMDGCN_GFX1010 = enum_elf_gfxip_level.define('EF_AMDGPU_MACH_AMDGCN_GFX1010', 51)
EF_AMDGPU_MACH_AMDGCN_GFX1030 = enum_elf_gfxip_level.define('EF_AMDGPU_MACH_AMDGCN_GFX1030', 54)
EF_AMDGPU_MACH_AMDGCN_GFX1100 = enum_elf_gfxip_level.define('EF_AMDGPU_MACH_AMDGCN_GFX1100', 65)
EF_AMDGPU_MACH_AMDGCN_GFX1150 = enum_elf_gfxip_level.define('EF_AMDGPU_MACH_AMDGCN_GFX1150', 67)
EF_AMDGPU_MACH_AMDGCN_GFX1200 = enum_elf_gfxip_level.define('EF_AMDGPU_MACH_AMDGCN_GFX1200', 78)

class struct_sqtt_file_chunk_spm_db(Struct): pass
struct_sqtt_file_chunk_spm_db._fields_ = [
  ('header', struct_sqtt_file_chunk_header),
  ('flags', uint32_t),
  ('preamble_size', uint32_t),
  ('num_timestamps', uint32_t),
  ('num_spm_counter_info', uint32_t),
  ('spm_counter_info_size', uint32_t),
  ('sample_interval', uint32_t),
]
enum_rgp_sqtt_marker_identifier = CEnum(ctypes.c_uint32)
RGP_SQTT_MARKER_IDENTIFIER_EVENT = enum_rgp_sqtt_marker_identifier.define('RGP_SQTT_MARKER_IDENTIFIER_EVENT', 0)
RGP_SQTT_MARKER_IDENTIFIER_CB_START = enum_rgp_sqtt_marker_identifier.define('RGP_SQTT_MARKER_IDENTIFIER_CB_START', 1)
RGP_SQTT_MARKER_IDENTIFIER_CB_END = enum_rgp_sqtt_marker_identifier.define('RGP_SQTT_MARKER_IDENTIFIER_CB_END', 2)
RGP_SQTT_MARKER_IDENTIFIER_BARRIER_START = enum_rgp_sqtt_marker_identifier.define('RGP_SQTT_MARKER_IDENTIFIER_BARRIER_START', 3)
RGP_SQTT_MARKER_IDENTIFIER_BARRIER_END = enum_rgp_sqtt_marker_identifier.define('RGP_SQTT_MARKER_IDENTIFIER_BARRIER_END', 4)
RGP_SQTT_MARKER_IDENTIFIER_USER_EVENT = enum_rgp_sqtt_marker_identifier.define('RGP_SQTT_MARKER_IDENTIFIER_USER_EVENT', 5)
RGP_SQTT_MARKER_IDENTIFIER_GENERAL_API = enum_rgp_sqtt_marker_identifier.define('RGP_SQTT_MARKER_IDENTIFIER_GENERAL_API', 6)
RGP_SQTT_MARKER_IDENTIFIER_SYNC = enum_rgp_sqtt_marker_identifier.define('RGP_SQTT_MARKER_IDENTIFIER_SYNC', 7)
RGP_SQTT_MARKER_IDENTIFIER_PRESENT = enum_rgp_sqtt_marker_identifier.define('RGP_SQTT_MARKER_IDENTIFIER_PRESENT', 8)
RGP_SQTT_MARKER_IDENTIFIER_LAYOUT_TRANSITION = enum_rgp_sqtt_marker_identifier.define('RGP_SQTT_MARKER_IDENTIFIER_LAYOUT_TRANSITION', 9)
RGP_SQTT_MARKER_IDENTIFIER_RENDER_PASS = enum_rgp_sqtt_marker_identifier.define('RGP_SQTT_MARKER_IDENTIFIER_RENDER_PASS', 10)
RGP_SQTT_MARKER_IDENTIFIER_RESERVED2 = enum_rgp_sqtt_marker_identifier.define('RGP_SQTT_MARKER_IDENTIFIER_RESERVED2', 11)
RGP_SQTT_MARKER_IDENTIFIER_BIND_PIPELINE = enum_rgp_sqtt_marker_identifier.define('RGP_SQTT_MARKER_IDENTIFIER_BIND_PIPELINE', 12)
RGP_SQTT_MARKER_IDENTIFIER_RESERVED4 = enum_rgp_sqtt_marker_identifier.define('RGP_SQTT_MARKER_IDENTIFIER_RESERVED4', 13)
RGP_SQTT_MARKER_IDENTIFIER_RESERVED5 = enum_rgp_sqtt_marker_identifier.define('RGP_SQTT_MARKER_IDENTIFIER_RESERVED5', 14)
RGP_SQTT_MARKER_IDENTIFIER_RESERVED6 = enum_rgp_sqtt_marker_identifier.define('RGP_SQTT_MARKER_IDENTIFIER_RESERVED6', 15)

class union_rgp_sqtt_marker_cb_id(ctypes.Union): pass
class union_rgp_sqtt_marker_cb_id_per_frame_cb_id(Struct): pass
union_rgp_sqtt_marker_cb_id_per_frame_cb_id._fields_ = [
  ('per_frame', uint32_t,1),
  ('frame_index', uint32_t,7),
  ('cb_index', uint32_t,12),
  ('reserved', uint32_t,12),
]
class union_rgp_sqtt_marker_cb_id_global_cb_id(Struct): pass
union_rgp_sqtt_marker_cb_id_global_cb_id._fields_ = [
  ('per_frame', uint32_t,1),
  ('cb_index', uint32_t,19),
  ('reserved', uint32_t,12),
]
union_rgp_sqtt_marker_cb_id._fields_ = [
  ('per_frame_cb_id', union_rgp_sqtt_marker_cb_id_per_frame_cb_id),
  ('global_cb_id', union_rgp_sqtt_marker_cb_id_global_cb_id),
  ('all', uint32_t),
]
class struct_rgp_sqtt_marker_cb_start(Struct): pass
class struct_rgp_sqtt_marker_cb_start_0(ctypes.Union): pass
class struct_rgp_sqtt_marker_cb_start_0_0(Struct): pass
struct_rgp_sqtt_marker_cb_start_0_0._fields_ = [
  ('identifier', uint32_t,4),
  ('ext_dwords', uint32_t,3),
  ('cb_id', uint32_t,20),
  ('queue', uint32_t,5),
]
struct_rgp_sqtt_marker_cb_start_0._anonymous_ = ['_0']
struct_rgp_sqtt_marker_cb_start_0._fields_ = [
  ('_0', struct_rgp_sqtt_marker_cb_start_0_0),
  ('dword01', uint32_t),
]
class struct_rgp_sqtt_marker_cb_start_1(ctypes.Union): pass
struct_rgp_sqtt_marker_cb_start_1._fields_ = [
  ('device_id_low', uint32_t),
  ('dword02', uint32_t),
]
class struct_rgp_sqtt_marker_cb_start_2(ctypes.Union): pass
struct_rgp_sqtt_marker_cb_start_2._fields_ = [
  ('device_id_high', uint32_t),
  ('dword03', uint32_t),
]
class struct_rgp_sqtt_marker_cb_start_3(ctypes.Union): pass
struct_rgp_sqtt_marker_cb_start_3._fields_ = [
  ('queue_flags', uint32_t),
  ('dword04', uint32_t),
]
struct_rgp_sqtt_marker_cb_start._anonymous_ = ['_0', '_1', '_2', '_3']
struct_rgp_sqtt_marker_cb_start._fields_ = [
  ('_0', struct_rgp_sqtt_marker_cb_start_0),
  ('_1', struct_rgp_sqtt_marker_cb_start_1),
  ('_2', struct_rgp_sqtt_marker_cb_start_2),
  ('_3', struct_rgp_sqtt_marker_cb_start_3),
]
class struct_rgp_sqtt_marker_cb_end(Struct): pass
class struct_rgp_sqtt_marker_cb_end_0(ctypes.Union): pass
class struct_rgp_sqtt_marker_cb_end_0_0(Struct): pass
struct_rgp_sqtt_marker_cb_end_0_0._fields_ = [
  ('identifier', uint32_t,4),
  ('ext_dwords', uint32_t,3),
  ('cb_id', uint32_t,20),
  ('reserved', uint32_t,5),
]
struct_rgp_sqtt_marker_cb_end_0._anonymous_ = ['_0']
struct_rgp_sqtt_marker_cb_end_0._fields_ = [
  ('_0', struct_rgp_sqtt_marker_cb_end_0_0),
  ('dword01', uint32_t),
]
class struct_rgp_sqtt_marker_cb_end_1(ctypes.Union): pass
struct_rgp_sqtt_marker_cb_end_1._fields_ = [
  ('device_id_low', uint32_t),
  ('dword02', uint32_t),
]
class struct_rgp_sqtt_marker_cb_end_2(ctypes.Union): pass
struct_rgp_sqtt_marker_cb_end_2._fields_ = [
  ('device_id_high', uint32_t),
  ('dword03', uint32_t),
]
struct_rgp_sqtt_marker_cb_end._anonymous_ = ['_0', '_1', '_2']
struct_rgp_sqtt_marker_cb_end._fields_ = [
  ('_0', struct_rgp_sqtt_marker_cb_end_0),
  ('_1', struct_rgp_sqtt_marker_cb_end_1),
  ('_2', struct_rgp_sqtt_marker_cb_end_2),
]
enum_rgp_sqtt_marker_general_api_type = CEnum(ctypes.c_uint32)
ApiCmdBindPipeline = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdBindPipeline', 0)
ApiCmdBindDescriptorSets = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdBindDescriptorSets', 1)
ApiCmdBindIndexBuffer = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdBindIndexBuffer', 2)
ApiCmdBindVertexBuffers = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdBindVertexBuffers', 3)
ApiCmdDraw = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdDraw', 4)
ApiCmdDrawIndexed = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdDrawIndexed', 5)
ApiCmdDrawIndirect = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdDrawIndirect', 6)
ApiCmdDrawIndexedIndirect = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdDrawIndexedIndirect', 7)
ApiCmdDrawIndirectCountAMD = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdDrawIndirectCountAMD', 8)
ApiCmdDrawIndexedIndirectCountAMD = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdDrawIndexedIndirectCountAMD', 9)
ApiCmdDispatch = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdDispatch', 10)
ApiCmdDispatchIndirect = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdDispatchIndirect', 11)
ApiCmdCopyBuffer = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdCopyBuffer', 12)
ApiCmdCopyImage = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdCopyImage', 13)
ApiCmdBlitImage = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdBlitImage', 14)
ApiCmdCopyBufferToImage = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdCopyBufferToImage', 15)
ApiCmdCopyImageToBuffer = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdCopyImageToBuffer', 16)
ApiCmdUpdateBuffer = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdUpdateBuffer', 17)
ApiCmdFillBuffer = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdFillBuffer', 18)
ApiCmdClearColorImage = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdClearColorImage', 19)
ApiCmdClearDepthStencilImage = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdClearDepthStencilImage', 20)
ApiCmdClearAttachments = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdClearAttachments', 21)
ApiCmdResolveImage = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdResolveImage', 22)
ApiCmdWaitEvents = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdWaitEvents', 23)
ApiCmdPipelineBarrier = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdPipelineBarrier', 24)
ApiCmdBeginQuery = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdBeginQuery', 25)
ApiCmdEndQuery = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdEndQuery', 26)
ApiCmdResetQueryPool = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdResetQueryPool', 27)
ApiCmdWriteTimestamp = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdWriteTimestamp', 28)
ApiCmdCopyQueryPoolResults = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdCopyQueryPoolResults', 29)
ApiCmdPushConstants = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdPushConstants', 30)
ApiCmdBeginRenderPass = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdBeginRenderPass', 31)
ApiCmdNextSubpass = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdNextSubpass', 32)
ApiCmdEndRenderPass = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdEndRenderPass', 33)
ApiCmdExecuteCommands = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdExecuteCommands', 34)
ApiCmdSetViewport = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdSetViewport', 35)
ApiCmdSetScissor = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdSetScissor', 36)
ApiCmdSetLineWidth = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdSetLineWidth', 37)
ApiCmdSetDepthBias = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdSetDepthBias', 38)
ApiCmdSetBlendConstants = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdSetBlendConstants', 39)
ApiCmdSetDepthBounds = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdSetDepthBounds', 40)
ApiCmdSetStencilCompareMask = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdSetStencilCompareMask', 41)
ApiCmdSetStencilWriteMask = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdSetStencilWriteMask', 42)
ApiCmdSetStencilReference = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdSetStencilReference', 43)
ApiCmdDrawIndirectCount = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdDrawIndirectCount', 44)
ApiCmdDrawIndexedIndirectCount = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdDrawIndexedIndirectCount', 45)
ApiCmdDrawMeshTasksEXT = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdDrawMeshTasksEXT', 47)
ApiCmdDrawMeshTasksIndirectCountEXT = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdDrawMeshTasksIndirectCountEXT', 48)
ApiCmdDrawMeshTasksIndirectEXT = enum_rgp_sqtt_marker_general_api_type.define('ApiCmdDrawMeshTasksIndirectEXT', 49)
ApiRayTracingSeparateCompiled = enum_rgp_sqtt_marker_general_api_type.define('ApiRayTracingSeparateCompiled', 8388608)
ApiInvalid = enum_rgp_sqtt_marker_general_api_type.define('ApiInvalid', 4294967295)

class struct_rgp_sqtt_marker_general_api(Struct): pass
class struct_rgp_sqtt_marker_general_api_0(ctypes.Union): pass
class struct_rgp_sqtt_marker_general_api_0_0(Struct): pass
struct_rgp_sqtt_marker_general_api_0_0._fields_ = [
  ('identifier', uint32_t,4),
  ('ext_dwords', uint32_t,3),
  ('api_type', uint32_t,20),
  ('is_end', uint32_t,1),
  ('reserved', uint32_t,4),
]
struct_rgp_sqtt_marker_general_api_0._anonymous_ = ['_0']
struct_rgp_sqtt_marker_general_api_0._fields_ = [
  ('_0', struct_rgp_sqtt_marker_general_api_0_0),
  ('dword01', uint32_t),
]
struct_rgp_sqtt_marker_general_api._anonymous_ = ['_0']
struct_rgp_sqtt_marker_general_api._fields_ = [
  ('_0', struct_rgp_sqtt_marker_general_api_0),
]
enum_rgp_sqtt_marker_event_type = CEnum(ctypes.c_uint32)
EventCmdDraw = enum_rgp_sqtt_marker_event_type.define('EventCmdDraw', 0)
EventCmdDrawIndexed = enum_rgp_sqtt_marker_event_type.define('EventCmdDrawIndexed', 1)
EventCmdDrawIndirect = enum_rgp_sqtt_marker_event_type.define('EventCmdDrawIndirect', 2)
EventCmdDrawIndexedIndirect = enum_rgp_sqtt_marker_event_type.define('EventCmdDrawIndexedIndirect', 3)
EventCmdDrawIndirectCountAMD = enum_rgp_sqtt_marker_event_type.define('EventCmdDrawIndirectCountAMD', 4)
EventCmdDrawIndexedIndirectCountAMD = enum_rgp_sqtt_marker_event_type.define('EventCmdDrawIndexedIndirectCountAMD', 5)
EventCmdDispatch = enum_rgp_sqtt_marker_event_type.define('EventCmdDispatch', 6)
EventCmdDispatchIndirect = enum_rgp_sqtt_marker_event_type.define('EventCmdDispatchIndirect', 7)
EventCmdCopyBuffer = enum_rgp_sqtt_marker_event_type.define('EventCmdCopyBuffer', 8)
EventCmdCopyImage = enum_rgp_sqtt_marker_event_type.define('EventCmdCopyImage', 9)
EventCmdBlitImage = enum_rgp_sqtt_marker_event_type.define('EventCmdBlitImage', 10)
EventCmdCopyBufferToImage = enum_rgp_sqtt_marker_event_type.define('EventCmdCopyBufferToImage', 11)
EventCmdCopyImageToBuffer = enum_rgp_sqtt_marker_event_type.define('EventCmdCopyImageToBuffer', 12)
EventCmdUpdateBuffer = enum_rgp_sqtt_marker_event_type.define('EventCmdUpdateBuffer', 13)
EventCmdFillBuffer = enum_rgp_sqtt_marker_event_type.define('EventCmdFillBuffer', 14)
EventCmdClearColorImage = enum_rgp_sqtt_marker_event_type.define('EventCmdClearColorImage', 15)
EventCmdClearDepthStencilImage = enum_rgp_sqtt_marker_event_type.define('EventCmdClearDepthStencilImage', 16)
EventCmdClearAttachments = enum_rgp_sqtt_marker_event_type.define('EventCmdClearAttachments', 17)
EventCmdResolveImage = enum_rgp_sqtt_marker_event_type.define('EventCmdResolveImage', 18)
EventCmdWaitEvents = enum_rgp_sqtt_marker_event_type.define('EventCmdWaitEvents', 19)
EventCmdPipelineBarrier = enum_rgp_sqtt_marker_event_type.define('EventCmdPipelineBarrier', 20)
EventCmdResetQueryPool = enum_rgp_sqtt_marker_event_type.define('EventCmdResetQueryPool', 21)
EventCmdCopyQueryPoolResults = enum_rgp_sqtt_marker_event_type.define('EventCmdCopyQueryPoolResults', 22)
EventRenderPassColorClear = enum_rgp_sqtt_marker_event_type.define('EventRenderPassColorClear', 23)
EventRenderPassDepthStencilClear = enum_rgp_sqtt_marker_event_type.define('EventRenderPassDepthStencilClear', 24)
EventRenderPassResolve = enum_rgp_sqtt_marker_event_type.define('EventRenderPassResolve', 25)
EventInternalUnknown = enum_rgp_sqtt_marker_event_type.define('EventInternalUnknown', 26)
EventCmdDrawIndirectCount = enum_rgp_sqtt_marker_event_type.define('EventCmdDrawIndirectCount', 27)
EventCmdDrawIndexedIndirectCount = enum_rgp_sqtt_marker_event_type.define('EventCmdDrawIndexedIndirectCount', 28)
EventCmdTraceRaysKHR = enum_rgp_sqtt_marker_event_type.define('EventCmdTraceRaysKHR', 30)
EventCmdTraceRaysIndirectKHR = enum_rgp_sqtt_marker_event_type.define('EventCmdTraceRaysIndirectKHR', 31)
EventCmdBuildAccelerationStructuresKHR = enum_rgp_sqtt_marker_event_type.define('EventCmdBuildAccelerationStructuresKHR', 32)
EventCmdBuildAccelerationStructuresIndirectKHR = enum_rgp_sqtt_marker_event_type.define('EventCmdBuildAccelerationStructuresIndirectKHR', 33)
EventCmdCopyAccelerationStructureKHR = enum_rgp_sqtt_marker_event_type.define('EventCmdCopyAccelerationStructureKHR', 34)
EventCmdCopyAccelerationStructureToMemoryKHR = enum_rgp_sqtt_marker_event_type.define('EventCmdCopyAccelerationStructureToMemoryKHR', 35)
EventCmdCopyMemoryToAccelerationStructureKHR = enum_rgp_sqtt_marker_event_type.define('EventCmdCopyMemoryToAccelerationStructureKHR', 36)
EventCmdDrawMeshTasksEXT = enum_rgp_sqtt_marker_event_type.define('EventCmdDrawMeshTasksEXT', 41)
EventCmdDrawMeshTasksIndirectCountEXT = enum_rgp_sqtt_marker_event_type.define('EventCmdDrawMeshTasksIndirectCountEXT', 42)
EventCmdDrawMeshTasksIndirectEXT = enum_rgp_sqtt_marker_event_type.define('EventCmdDrawMeshTasksIndirectEXT', 43)
EventUnknown = enum_rgp_sqtt_marker_event_type.define('EventUnknown', 32767)
EventInvalid = enum_rgp_sqtt_marker_event_type.define('EventInvalid', 4294967295)

class struct_rgp_sqtt_marker_event(Struct): pass
class struct_rgp_sqtt_marker_event_0(ctypes.Union): pass
class struct_rgp_sqtt_marker_event_0_0(Struct): pass
struct_rgp_sqtt_marker_event_0_0._fields_ = [
  ('identifier', uint32_t,4),
  ('ext_dwords', uint32_t,3),
  ('api_type', uint32_t,24),
  ('has_thread_dims', uint32_t,1),
]
struct_rgp_sqtt_marker_event_0._anonymous_ = ['_0']
struct_rgp_sqtt_marker_event_0._fields_ = [
  ('_0', struct_rgp_sqtt_marker_event_0_0),
  ('dword01', uint32_t),
]
class struct_rgp_sqtt_marker_event_1(ctypes.Union): pass
class struct_rgp_sqtt_marker_event_1_0(Struct): pass
struct_rgp_sqtt_marker_event_1_0._fields_ = [
  ('cb_id', uint32_t,20),
  ('vertex_offset_reg_idx', uint32_t,4),
  ('instance_offset_reg_idx', uint32_t,4),
  ('draw_index_reg_idx', uint32_t,4),
]
struct_rgp_sqtt_marker_event_1._anonymous_ = ['_0']
struct_rgp_sqtt_marker_event_1._fields_ = [
  ('_0', struct_rgp_sqtt_marker_event_1_0),
  ('dword02', uint32_t),
]
class struct_rgp_sqtt_marker_event_2(ctypes.Union): pass
struct_rgp_sqtt_marker_event_2._fields_ = [
  ('cmd_id', uint32_t),
  ('dword03', uint32_t),
]
struct_rgp_sqtt_marker_event._anonymous_ = ['_0', '_1', '_2']
struct_rgp_sqtt_marker_event._fields_ = [
  ('_0', struct_rgp_sqtt_marker_event_0),
  ('_1', struct_rgp_sqtt_marker_event_1),
  ('_2', struct_rgp_sqtt_marker_event_2),
]
class struct_rgp_sqtt_marker_event_with_dims(Struct): pass
struct_rgp_sqtt_marker_event_with_dims._fields_ = [
  ('event', struct_rgp_sqtt_marker_event),
  ('thread_x', uint32_t),
  ('thread_y', uint32_t),
  ('thread_z', uint32_t),
]
class struct_rgp_sqtt_marker_barrier_start(Struct): pass
class struct_rgp_sqtt_marker_barrier_start_0(ctypes.Union): pass
class struct_rgp_sqtt_marker_barrier_start_0_0(Struct): pass
struct_rgp_sqtt_marker_barrier_start_0_0._fields_ = [
  ('identifier', uint32_t,4),
  ('ext_dwords', uint32_t,3),
  ('cb_id', uint32_t,20),
  ('reserved', uint32_t,5),
]
struct_rgp_sqtt_marker_barrier_start_0._anonymous_ = ['_0']
struct_rgp_sqtt_marker_barrier_start_0._fields_ = [
  ('_0', struct_rgp_sqtt_marker_barrier_start_0_0),
  ('dword01', uint32_t),
]
class struct_rgp_sqtt_marker_barrier_start_1(ctypes.Union): pass
class struct_rgp_sqtt_marker_barrier_start_1_0(Struct): pass
struct_rgp_sqtt_marker_barrier_start_1_0._fields_ = [
  ('driver_reason', uint32_t,31),
  ('internal', uint32_t,1),
]
struct_rgp_sqtt_marker_barrier_start_1._anonymous_ = ['_0']
struct_rgp_sqtt_marker_barrier_start_1._fields_ = [
  ('_0', struct_rgp_sqtt_marker_barrier_start_1_0),
  ('dword02', uint32_t),
]
struct_rgp_sqtt_marker_barrier_start._anonymous_ = ['_0', '_1']
struct_rgp_sqtt_marker_barrier_start._fields_ = [
  ('_0', struct_rgp_sqtt_marker_barrier_start_0),
  ('_1', struct_rgp_sqtt_marker_barrier_start_1),
]
class struct_rgp_sqtt_marker_barrier_end(Struct): pass
class struct_rgp_sqtt_marker_barrier_end_0(ctypes.Union): pass
class struct_rgp_sqtt_marker_barrier_end_0_0(Struct): pass
struct_rgp_sqtt_marker_barrier_end_0_0._fields_ = [
  ('identifier', uint32_t,4),
  ('ext_dwords', uint32_t,3),
  ('cb_id', uint32_t,20),
  ('wait_on_eop_ts', uint32_t,1),
  ('vs_partial_flush', uint32_t,1),
  ('ps_partial_flush', uint32_t,1),
  ('cs_partial_flush', uint32_t,1),
  ('pfp_sync_me', uint32_t,1),
]
struct_rgp_sqtt_marker_barrier_end_0._anonymous_ = ['_0']
struct_rgp_sqtt_marker_barrier_end_0._fields_ = [
  ('_0', struct_rgp_sqtt_marker_barrier_end_0_0),
  ('dword01', uint32_t),
]
class struct_rgp_sqtt_marker_barrier_end_1(ctypes.Union): pass
class struct_rgp_sqtt_marker_barrier_end_1_0(Struct): pass
struct_rgp_sqtt_marker_barrier_end_1_0._fields_ = [
  ('sync_cp_dma', uint32_t,1),
  ('inval_tcp', uint32_t,1),
  ('inval_sqI', uint32_t,1),
  ('inval_sqK', uint32_t,1),
  ('flush_tcc', uint32_t,1),
  ('inval_tcc', uint32_t,1),
  ('flush_cb', uint32_t,1),
  ('inval_cb', uint32_t,1),
  ('flush_db', uint32_t,1),
  ('inval_db', uint32_t,1),
  ('num_layout_transitions', uint32_t,16),
  ('inval_gl1', uint32_t,1),
  ('wait_on_ts', uint32_t,1),
  ('eop_ts_bottom_of_pipe', uint32_t,1),
  ('eos_ts_ps_done', uint32_t,1),
  ('eos_ts_cs_done', uint32_t,1),
  ('reserved', uint32_t,1),
]
struct_rgp_sqtt_marker_barrier_end_1._anonymous_ = ['_0']
struct_rgp_sqtt_marker_barrier_end_1._fields_ = [
  ('_0', struct_rgp_sqtt_marker_barrier_end_1_0),
  ('dword02', uint32_t),
]
struct_rgp_sqtt_marker_barrier_end._anonymous_ = ['_0', '_1']
struct_rgp_sqtt_marker_barrier_end._fields_ = [
  ('_0', struct_rgp_sqtt_marker_barrier_end_0),
  ('_1', struct_rgp_sqtt_marker_barrier_end_1),
]
class struct_rgp_sqtt_marker_layout_transition(Struct): pass
class struct_rgp_sqtt_marker_layout_transition_0(ctypes.Union): pass
class struct_rgp_sqtt_marker_layout_transition_0_0(Struct): pass
struct_rgp_sqtt_marker_layout_transition_0_0._fields_ = [
  ('identifier', uint32_t,4),
  ('ext_dwords', uint32_t,3),
  ('depth_stencil_expand', uint32_t,1),
  ('htile_hiz_range_expand', uint32_t,1),
  ('depth_stencil_resummarize', uint32_t,1),
  ('dcc_decompress', uint32_t,1),
  ('fmask_decompress', uint32_t,1),
  ('fast_clear_eliminate', uint32_t,1),
  ('fmask_color_expand', uint32_t,1),
  ('init_mask_ram', uint32_t,1),
  ('reserved1', uint32_t,17),
]
struct_rgp_sqtt_marker_layout_transition_0._anonymous_ = ['_0']
struct_rgp_sqtt_marker_layout_transition_0._fields_ = [
  ('_0', struct_rgp_sqtt_marker_layout_transition_0_0),
  ('dword01', uint32_t),
]
class struct_rgp_sqtt_marker_layout_transition_1(ctypes.Union): pass
class struct_rgp_sqtt_marker_layout_transition_1_0(Struct): pass
struct_rgp_sqtt_marker_layout_transition_1_0._fields_ = [
  ('reserved2', uint32_t,32),
]
struct_rgp_sqtt_marker_layout_transition_1._anonymous_ = ['_0']
struct_rgp_sqtt_marker_layout_transition_1._fields_ = [
  ('_0', struct_rgp_sqtt_marker_layout_transition_1_0),
  ('dword02', uint32_t),
]
struct_rgp_sqtt_marker_layout_transition._anonymous_ = ['_0', '_1']
struct_rgp_sqtt_marker_layout_transition._fields_ = [
  ('_0', struct_rgp_sqtt_marker_layout_transition_0),
  ('_1', struct_rgp_sqtt_marker_layout_transition_1),
]
class struct_rgp_sqtt_marker_user_event(Struct): pass
class struct_rgp_sqtt_marker_user_event_0(ctypes.Union): pass
class struct_rgp_sqtt_marker_user_event_0_0(Struct): pass
struct_rgp_sqtt_marker_user_event_0_0._fields_ = [
  ('identifier', uint32_t,4),
  ('reserved0', uint32_t,8),
  ('data_type', uint32_t,8),
  ('reserved1', uint32_t,12),
]
struct_rgp_sqtt_marker_user_event_0._anonymous_ = ['_0']
struct_rgp_sqtt_marker_user_event_0._fields_ = [
  ('_0', struct_rgp_sqtt_marker_user_event_0_0),
  ('dword01', uint32_t),
]
struct_rgp_sqtt_marker_user_event._anonymous_ = ['_0']
struct_rgp_sqtt_marker_user_event._fields_ = [
  ('_0', struct_rgp_sqtt_marker_user_event_0),
]
class struct_rgp_sqtt_marker_user_event_with_length(Struct): pass
struct_rgp_sqtt_marker_user_event_with_length._fields_ = [
  ('user_event', struct_rgp_sqtt_marker_user_event),
  ('length', uint32_t),
]
enum_rgp_sqtt_marker_user_event_type = CEnum(ctypes.c_uint32)
UserEventTrigger = enum_rgp_sqtt_marker_user_event_type.define('UserEventTrigger', 0)
UserEventPop = enum_rgp_sqtt_marker_user_event_type.define('UserEventPop', 1)
UserEventPush = enum_rgp_sqtt_marker_user_event_type.define('UserEventPush', 2)
UserEventObjectName = enum_rgp_sqtt_marker_user_event_type.define('UserEventObjectName', 3)

class struct_rgp_sqtt_marker_pipeline_bind(Struct): pass
class struct_rgp_sqtt_marker_pipeline_bind_0(ctypes.Union): pass
class struct_rgp_sqtt_marker_pipeline_bind_0_0(Struct): pass
struct_rgp_sqtt_marker_pipeline_bind_0_0._fields_ = [
  ('identifier', uint32_t,4),
  ('ext_dwords', uint32_t,3),
  ('bind_point', uint32_t,1),
  ('cb_id', uint32_t,20),
  ('reserved', uint32_t,4),
]
struct_rgp_sqtt_marker_pipeline_bind_0._anonymous_ = ['_0']
struct_rgp_sqtt_marker_pipeline_bind_0._fields_ = [
  ('_0', struct_rgp_sqtt_marker_pipeline_bind_0_0),
  ('dword01', uint32_t),
]
class struct_rgp_sqtt_marker_pipeline_bind_1(ctypes.Union): pass
class struct_rgp_sqtt_marker_pipeline_bind_1_0(Struct): pass
struct_rgp_sqtt_marker_pipeline_bind_1_0._fields_ = [
  ('dword02', uint32_t),
  ('dword03', uint32_t),
]
struct_rgp_sqtt_marker_pipeline_bind_1._anonymous_ = ['_0']
struct_rgp_sqtt_marker_pipeline_bind_1._fields_ = [
  ('api_pso_hash', (uint32_t * 2)),
  ('_0', struct_rgp_sqtt_marker_pipeline_bind_1_0),
]
struct_rgp_sqtt_marker_pipeline_bind._anonymous_ = ['_0', '_1']
struct_rgp_sqtt_marker_pipeline_bind._fields_ = [
  ('_0', struct_rgp_sqtt_marker_pipeline_bind_0),
  ('_1', struct_rgp_sqtt_marker_pipeline_bind_1),
]
SQTT_FILE_MAGIC_NUMBER = 0x50303042
SQTT_FILE_VERSION_MAJOR = 1
SQTT_FILE_VERSION_MINOR = 5
SQTT_GPU_NAME_MAX_SIZE = 256
SQTT_MAX_NUM_SE = 32
SQTT_SA_PER_SE = 2
SQTT_ACTIVE_PIXEL_PACKER_MASK_DWORDS = 4