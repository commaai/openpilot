# mypy: disable-error-code="empty-body"
from __future__ import annotations
import ctypes
from typing import Literal, TypeAlias
from tinygrad.runtime.support.c import _IO, _IOW, _IOR, _IOWR
from tinygrad.runtime.support import c
@c.record
class struct_sqtt_data_info(c.Struct):
  SIZE = 12
  cur_offset: int
  trace_status: int
  gfx9_write_counter: int
  gfx10_dropped_cntr: int
uint32_t: TypeAlias = ctypes.c_uint32
struct_sqtt_data_info.register_fields([('cur_offset', uint32_t, 0), ('trace_status', uint32_t, 4), ('gfx9_write_counter', uint32_t, 8), ('gfx10_dropped_cntr', uint32_t, 8)])
@c.record
class struct_sqtt_data_se(c.Struct):
  SIZE = 32
  info: struct_sqtt_data_info
  data_ptr: ctypes.c_void_p
  shader_engine: int
  compute_unit: int
struct_sqtt_data_se.register_fields([('info', struct_sqtt_data_info, 0), ('data_ptr', ctypes.c_void_p, 16), ('shader_engine', uint32_t, 24), ('compute_unit', uint32_t, 28)])
enum_sqtt_version: dict[int, str] = {(SQTT_VERSION_NONE:=0): 'SQTT_VERSION_NONE', (SQTT_VERSION_2_2:=5): 'SQTT_VERSION_2_2', (SQTT_VERSION_2_3:=6): 'SQTT_VERSION_2_3', (SQTT_VERSION_2_4:=7): 'SQTT_VERSION_2_4', (SQTT_VERSION_3_2:=11): 'SQTT_VERSION_3_2', (SQTT_VERSION_3_3:=12): 'SQTT_VERSION_3_3'}
enum_sqtt_file_chunk_type: dict[int, str] = {(SQTT_FILE_CHUNK_TYPE_ASIC_INFO:=0): 'SQTT_FILE_CHUNK_TYPE_ASIC_INFO', (SQTT_FILE_CHUNK_TYPE_SQTT_DESC:=1): 'SQTT_FILE_CHUNK_TYPE_SQTT_DESC', (SQTT_FILE_CHUNK_TYPE_SQTT_DATA:=2): 'SQTT_FILE_CHUNK_TYPE_SQTT_DATA', (SQTT_FILE_CHUNK_TYPE_API_INFO:=3): 'SQTT_FILE_CHUNK_TYPE_API_INFO', (SQTT_FILE_CHUNK_TYPE_RESERVED:=4): 'SQTT_FILE_CHUNK_TYPE_RESERVED', (SQTT_FILE_CHUNK_TYPE_QUEUE_EVENT_TIMINGS:=5): 'SQTT_FILE_CHUNK_TYPE_QUEUE_EVENT_TIMINGS', (SQTT_FILE_CHUNK_TYPE_CLOCK_CALIBRATION:=6): 'SQTT_FILE_CHUNK_TYPE_CLOCK_CALIBRATION', (SQTT_FILE_CHUNK_TYPE_CPU_INFO:=7): 'SQTT_FILE_CHUNK_TYPE_CPU_INFO', (SQTT_FILE_CHUNK_TYPE_SPM_DB:=8): 'SQTT_FILE_CHUNK_TYPE_SPM_DB', (SQTT_FILE_CHUNK_TYPE_CODE_OBJECT_DATABASE:=9): 'SQTT_FILE_CHUNK_TYPE_CODE_OBJECT_DATABASE', (SQTT_FILE_CHUNK_TYPE_CODE_OBJECT_LOADER_EVENTS:=10): 'SQTT_FILE_CHUNK_TYPE_CODE_OBJECT_LOADER_EVENTS', (SQTT_FILE_CHUNK_TYPE_PSO_CORRELATION:=11): 'SQTT_FILE_CHUNK_TYPE_PSO_CORRELATION', (SQTT_FILE_CHUNK_TYPE_INSTRUMENTATION_TABLE:=12): 'SQTT_FILE_CHUNK_TYPE_INSTRUMENTATION_TABLE', (SQTT_FILE_CHUNK_TYPE_COUNT:=13): 'SQTT_FILE_CHUNK_TYPE_COUNT'}
@c.record
class struct_sqtt_file_chunk_id(c.Struct):
  SIZE = 4
  type: int
  index: int
  reserved: int
int32_t: TypeAlias = ctypes.c_int32
struct_sqtt_file_chunk_id.register_fields([('type', int32_t, 0, 8, 0), ('index', int32_t, 1, 8, 0), ('reserved', int32_t, 2, 16, 0)])
@c.record
class struct_sqtt_file_chunk_header(c.Struct):
  SIZE = 16
  chunk_id: struct_sqtt_file_chunk_id
  minor_version: int
  major_version: int
  size_in_bytes: int
  padding: int
uint16_t: TypeAlias = ctypes.c_uint16
struct_sqtt_file_chunk_header.register_fields([('chunk_id', struct_sqtt_file_chunk_id, 0), ('minor_version', uint16_t, 4), ('major_version', uint16_t, 6), ('size_in_bytes', int32_t, 8), ('padding', int32_t, 12)])
@c.record
class struct_sqtt_file_header_flags(c.Struct):
  SIZE = 4
  is_semaphore_queue_timing_etw: int
  no_queue_semaphore_timestamps: int
  reserved: int
  value: int
struct_sqtt_file_header_flags.register_fields([('is_semaphore_queue_timing_etw', uint32_t, 0, 1, 0), ('no_queue_semaphore_timestamps', uint32_t, 0, 1, 1), ('reserved', uint32_t, 0, 30, 2), ('value', uint32_t, 0)])
@c.record
class struct_sqtt_file_header(c.Struct):
  SIZE = 56
  magic_number: int
  version_major: int
  version_minor: int
  flags: struct_sqtt_file_header_flags
  chunk_offset: int
  second: int
  minute: int
  hour: int
  day_in_month: int
  month: int
  year: int
  day_in_week: int
  day_in_year: int
  is_daylight_savings: int
struct_sqtt_file_header.register_fields([('magic_number', uint32_t, 0), ('version_major', uint32_t, 4), ('version_minor', uint32_t, 8), ('flags', struct_sqtt_file_header_flags, 12), ('chunk_offset', int32_t, 16), ('second', int32_t, 20), ('minute', int32_t, 24), ('hour', int32_t, 28), ('day_in_month', int32_t, 32), ('month', int32_t, 36), ('year', int32_t, 40), ('day_in_week', int32_t, 44), ('day_in_year', int32_t, 48), ('is_daylight_savings', int32_t, 52)])
@c.record
class struct_sqtt_file_chunk_cpu_info(c.Struct):
  SIZE = 112
  header: struct_sqtt_file_chunk_header
  vendor_id: c.Array[ctypes.c_uint32, Literal[4]]
  processor_brand: c.Array[ctypes.c_uint32, Literal[12]]
  reserved: c.Array[ctypes.c_uint32, Literal[2]]
  cpu_timestamp_freq: int
  clock_speed: int
  num_logical_cores: int
  num_physical_cores: int
  system_ram_size: int
uint64_t: TypeAlias = ctypes.c_uint64
struct_sqtt_file_chunk_cpu_info.register_fields([('header', struct_sqtt_file_chunk_header, 0), ('vendor_id', c.Array[uint32_t, Literal[4]], 16), ('processor_brand', c.Array[uint32_t, Literal[12]], 32), ('reserved', c.Array[uint32_t, Literal[2]], 80), ('cpu_timestamp_freq', uint64_t, 88), ('clock_speed', uint32_t, 96), ('num_logical_cores', uint32_t, 100), ('num_physical_cores', uint32_t, 104), ('system_ram_size', uint32_t, 108)])
enum_sqtt_file_chunk_asic_info_flags: dict[int, str] = {(SQTT_FILE_CHUNK_ASIC_INFO_FLAG_SC_PACKER_NUMBERING:=1): 'SQTT_FILE_CHUNK_ASIC_INFO_FLAG_SC_PACKER_NUMBERING', (SQTT_FILE_CHUNK_ASIC_INFO_FLAG_PS1_EVENT_TOKENS_ENABLED:=2): 'SQTT_FILE_CHUNK_ASIC_INFO_FLAG_PS1_EVENT_TOKENS_ENABLED'}
enum_sqtt_gpu_type: dict[int, str] = {(SQTT_GPU_TYPE_UNKNOWN:=0): 'SQTT_GPU_TYPE_UNKNOWN', (SQTT_GPU_TYPE_INTEGRATED:=1): 'SQTT_GPU_TYPE_INTEGRATED', (SQTT_GPU_TYPE_DISCRETE:=2): 'SQTT_GPU_TYPE_DISCRETE', (SQTT_GPU_TYPE_VIRTUAL:=3): 'SQTT_GPU_TYPE_VIRTUAL'}
enum_sqtt_gfxip_level: dict[int, str] = {(SQTT_GFXIP_LEVEL_NONE:=0): 'SQTT_GFXIP_LEVEL_NONE', (SQTT_GFXIP_LEVEL_GFXIP_6:=1): 'SQTT_GFXIP_LEVEL_GFXIP_6', (SQTT_GFXIP_LEVEL_GFXIP_7:=2): 'SQTT_GFXIP_LEVEL_GFXIP_7', (SQTT_GFXIP_LEVEL_GFXIP_8:=3): 'SQTT_GFXIP_LEVEL_GFXIP_8', (SQTT_GFXIP_LEVEL_GFXIP_8_1:=4): 'SQTT_GFXIP_LEVEL_GFXIP_8_1', (SQTT_GFXIP_LEVEL_GFXIP_9:=5): 'SQTT_GFXIP_LEVEL_GFXIP_9', (SQTT_GFXIP_LEVEL_GFXIP_10_1:=7): 'SQTT_GFXIP_LEVEL_GFXIP_10_1', (SQTT_GFXIP_LEVEL_GFXIP_10_3:=9): 'SQTT_GFXIP_LEVEL_GFXIP_10_3', (SQTT_GFXIP_LEVEL_GFXIP_11_0:=12): 'SQTT_GFXIP_LEVEL_GFXIP_11_0', (SQTT_GFXIP_LEVEL_GFXIP_11_5:=13): 'SQTT_GFXIP_LEVEL_GFXIP_11_5', (SQTT_GFXIP_LEVEL_GFXIP_12:=16): 'SQTT_GFXIP_LEVEL_GFXIP_12'}
enum_sqtt_memory_type: dict[int, str] = {(SQTT_MEMORY_TYPE_UNKNOWN:=0): 'SQTT_MEMORY_TYPE_UNKNOWN', (SQTT_MEMORY_TYPE_DDR:=1): 'SQTT_MEMORY_TYPE_DDR', (SQTT_MEMORY_TYPE_DDR2:=2): 'SQTT_MEMORY_TYPE_DDR2', (SQTT_MEMORY_TYPE_DDR3:=3): 'SQTT_MEMORY_TYPE_DDR3', (SQTT_MEMORY_TYPE_DDR4:=4): 'SQTT_MEMORY_TYPE_DDR4', (SQTT_MEMORY_TYPE_DDR5:=5): 'SQTT_MEMORY_TYPE_DDR5', (SQTT_MEMORY_TYPE_GDDR3:=16): 'SQTT_MEMORY_TYPE_GDDR3', (SQTT_MEMORY_TYPE_GDDR4:=17): 'SQTT_MEMORY_TYPE_GDDR4', (SQTT_MEMORY_TYPE_GDDR5:=18): 'SQTT_MEMORY_TYPE_GDDR5', (SQTT_MEMORY_TYPE_GDDR6:=19): 'SQTT_MEMORY_TYPE_GDDR6', (SQTT_MEMORY_TYPE_HBM:=32): 'SQTT_MEMORY_TYPE_HBM', (SQTT_MEMORY_TYPE_HBM2:=33): 'SQTT_MEMORY_TYPE_HBM2', (SQTT_MEMORY_TYPE_HBM3:=34): 'SQTT_MEMORY_TYPE_HBM3', (SQTT_MEMORY_TYPE_LPDDR4:=48): 'SQTT_MEMORY_TYPE_LPDDR4', (SQTT_MEMORY_TYPE_LPDDR5:=49): 'SQTT_MEMORY_TYPE_LPDDR5'}
@c.record
class struct_sqtt_file_chunk_asic_info(c.Struct):
  SIZE = 768
  header: struct_sqtt_file_chunk_header
  flags: int
  trace_shader_core_clock: int
  trace_memory_clock: int
  device_id: int
  device_revision_id: int
  vgprs_per_simd: int
  sgprs_per_simd: int
  shader_engines: int
  compute_unit_per_shader_engine: int
  simd_per_compute_unit: int
  wavefronts_per_simd: int
  minimum_vgpr_alloc: int
  vgpr_alloc_granularity: int
  minimum_sgpr_alloc: int
  sgpr_alloc_granularity: int
  hardware_contexts: int
  gpu_type: int
  gfxip_level: int
  gpu_index: int
  gds_size: int
  gds_per_shader_engine: int
  ce_ram_size: int
  ce_ram_size_graphics: int
  ce_ram_size_compute: int
  max_number_of_dedicated_cus: int
  vram_size: int
  vram_bus_width: int
  l2_cache_size: int
  l1_cache_size: int
  lds_size: int
  gpu_name: c.Array[ctypes.c_char, Literal[256]]
  alu_per_clock: float
  texture_per_clock: float
  prims_per_clock: float
  pixels_per_clock: float
  gpu_timestamp_frequency: int
  max_shader_core_clock: int
  max_memory_clock: int
  memory_ops_per_clock: int
  memory_chip_type: int
  lds_granularity: int
  cu_mask: c.Array[c.Array[ctypes.c_uint16, Literal[2]], Literal[32]]
  reserved1: c.Array[ctypes.c_char, Literal[128]]
  active_pixel_packer_mask: c.Array[ctypes.c_uint32, Literal[4]]
  reserved2: c.Array[ctypes.c_char, Literal[16]]
  gl1_cache_size: int
  instruction_cache_size: int
  scalar_cache_size: int
  mall_cache_size: int
  padding: c.Array[ctypes.c_char, Literal[4]]
int64_t: TypeAlias = ctypes.c_int64
struct_sqtt_file_chunk_asic_info.register_fields([('header', struct_sqtt_file_chunk_header, 0), ('flags', uint64_t, 16), ('trace_shader_core_clock', uint64_t, 24), ('trace_memory_clock', uint64_t, 32), ('device_id', int32_t, 40), ('device_revision_id', int32_t, 44), ('vgprs_per_simd', int32_t, 48), ('sgprs_per_simd', int32_t, 52), ('shader_engines', int32_t, 56), ('compute_unit_per_shader_engine', int32_t, 60), ('simd_per_compute_unit', int32_t, 64), ('wavefronts_per_simd', int32_t, 68), ('minimum_vgpr_alloc', int32_t, 72), ('vgpr_alloc_granularity', int32_t, 76), ('minimum_sgpr_alloc', int32_t, 80), ('sgpr_alloc_granularity', int32_t, 84), ('hardware_contexts', int32_t, 88), ('gpu_type', ctypes.c_uint32, 92), ('gfxip_level', ctypes.c_uint32, 96), ('gpu_index', int32_t, 100), ('gds_size', int32_t, 104), ('gds_per_shader_engine', int32_t, 108), ('ce_ram_size', int32_t, 112), ('ce_ram_size_graphics', int32_t, 116), ('ce_ram_size_compute', int32_t, 120), ('max_number_of_dedicated_cus', int32_t, 124), ('vram_size', int64_t, 128), ('vram_bus_width', int32_t, 136), ('l2_cache_size', int32_t, 140), ('l1_cache_size', int32_t, 144), ('lds_size', int32_t, 148), ('gpu_name', c.Array[ctypes.c_char, Literal[256]], 152), ('alu_per_clock', ctypes.c_float, 408), ('texture_per_clock', ctypes.c_float, 412), ('prims_per_clock', ctypes.c_float, 416), ('pixels_per_clock', ctypes.c_float, 420), ('gpu_timestamp_frequency', uint64_t, 424), ('max_shader_core_clock', uint64_t, 432), ('max_memory_clock', uint64_t, 440), ('memory_ops_per_clock', uint32_t, 448), ('memory_chip_type', ctypes.c_uint32, 452), ('lds_granularity', uint32_t, 456), ('cu_mask', c.Array[c.Array[uint16_t, Literal[2]], Literal[32]], 460), ('reserved1', c.Array[ctypes.c_char, Literal[128]], 588), ('active_pixel_packer_mask', c.Array[uint32_t, Literal[4]], 716), ('reserved2', c.Array[ctypes.c_char, Literal[16]], 732), ('gl1_cache_size', uint32_t, 748), ('instruction_cache_size', uint32_t, 752), ('scalar_cache_size', uint32_t, 756), ('mall_cache_size', uint32_t, 760), ('padding', c.Array[ctypes.c_char, Literal[4]], 764)])
enum_sqtt_api_type: dict[int, str] = {(SQTT_API_TYPE_DIRECTX_12:=0): 'SQTT_API_TYPE_DIRECTX_12', (SQTT_API_TYPE_VULKAN:=1): 'SQTT_API_TYPE_VULKAN', (SQTT_API_TYPE_GENERIC:=2): 'SQTT_API_TYPE_GENERIC', (SQTT_API_TYPE_OPENCL:=3): 'SQTT_API_TYPE_OPENCL'}
enum_sqtt_instruction_trace_mode: dict[int, str] = {(SQTT_INSTRUCTION_TRACE_DISABLED:=0): 'SQTT_INSTRUCTION_TRACE_DISABLED', (SQTT_INSTRUCTION_TRACE_FULL_FRAME:=1): 'SQTT_INSTRUCTION_TRACE_FULL_FRAME', (SQTT_INSTRUCTION_TRACE_API_PSO:=2): 'SQTT_INSTRUCTION_TRACE_API_PSO'}
enum_sqtt_profiling_mode: dict[int, str] = {(SQTT_PROFILING_MODE_PRESENT:=0): 'SQTT_PROFILING_MODE_PRESENT', (SQTT_PROFILING_MODE_USER_MARKERS:=1): 'SQTT_PROFILING_MODE_USER_MARKERS', (SQTT_PROFILING_MODE_INDEX:=2): 'SQTT_PROFILING_MODE_INDEX', (SQTT_PROFILING_MODE_TAG:=3): 'SQTT_PROFILING_MODE_TAG'}
@c.record
class union_sqtt_profiling_mode_data(c.Struct):
  SIZE = 512
  user_marker_profiling_data: union_sqtt_profiling_mode_data_user_marker_profiling_data
  index_profiling_data: union_sqtt_profiling_mode_data_index_profiling_data
  tag_profiling_data: union_sqtt_profiling_mode_data_tag_profiling_data
@c.record
class union_sqtt_profiling_mode_data_user_marker_profiling_data(c.Struct):
  SIZE = 512
  start: c.Array[ctypes.c_char, Literal[256]]
  end: c.Array[ctypes.c_char, Literal[256]]
union_sqtt_profiling_mode_data_user_marker_profiling_data.register_fields([('start', c.Array[ctypes.c_char, Literal[256]], 0), ('end', c.Array[ctypes.c_char, Literal[256]], 256)])
@c.record
class union_sqtt_profiling_mode_data_index_profiling_data(c.Struct):
  SIZE = 8
  start: int
  end: int
union_sqtt_profiling_mode_data_index_profiling_data.register_fields([('start', uint32_t, 0), ('end', uint32_t, 4)])
@c.record
class union_sqtt_profiling_mode_data_tag_profiling_data(c.Struct):
  SIZE = 16
  begin_hi: int
  begin_lo: int
  end_hi: int
  end_lo: int
union_sqtt_profiling_mode_data_tag_profiling_data.register_fields([('begin_hi', uint32_t, 0), ('begin_lo', uint32_t, 4), ('end_hi', uint32_t, 8), ('end_lo', uint32_t, 12)])
union_sqtt_profiling_mode_data.register_fields([('user_marker_profiling_data', union_sqtt_profiling_mode_data_user_marker_profiling_data, 0), ('index_profiling_data', union_sqtt_profiling_mode_data_index_profiling_data, 0), ('tag_profiling_data', union_sqtt_profiling_mode_data_tag_profiling_data, 0)])
@c.record
class union_sqtt_instruction_trace_data(c.Struct):
  SIZE = 8
  api_pso_data: union_sqtt_instruction_trace_data_api_pso_data
  shader_engine_filter: union_sqtt_instruction_trace_data_shader_engine_filter
@c.record
class union_sqtt_instruction_trace_data_api_pso_data(c.Struct):
  SIZE = 8
  api_pso_filter: int
union_sqtt_instruction_trace_data_api_pso_data.register_fields([('api_pso_filter', uint64_t, 0)])
@c.record
class union_sqtt_instruction_trace_data_shader_engine_filter(c.Struct):
  SIZE = 4
  mask: int
union_sqtt_instruction_trace_data_shader_engine_filter.register_fields([('mask', uint32_t, 0)])
union_sqtt_instruction_trace_data.register_fields([('api_pso_data', union_sqtt_instruction_trace_data_api_pso_data, 0), ('shader_engine_filter', union_sqtt_instruction_trace_data_shader_engine_filter, 0)])
@c.record
class struct_sqtt_file_chunk_api_info(c.Struct):
  SIZE = 560
  header: struct_sqtt_file_chunk_header
  api_type: int
  major_version: int
  minor_version: int
  profiling_mode: int
  reserved: int
  profiling_mode_data: union_sqtt_profiling_mode_data
  instruction_trace_mode: int
  reserved2: int
  instruction_trace_data: union_sqtt_instruction_trace_data
struct_sqtt_file_chunk_api_info.register_fields([('header', struct_sqtt_file_chunk_header, 0), ('api_type', ctypes.c_uint32, 16), ('major_version', uint16_t, 20), ('minor_version', uint16_t, 22), ('profiling_mode', ctypes.c_uint32, 24), ('reserved', uint32_t, 28), ('profiling_mode_data', union_sqtt_profiling_mode_data, 32), ('instruction_trace_mode', ctypes.c_uint32, 544), ('reserved2', uint32_t, 548), ('instruction_trace_data', union_sqtt_instruction_trace_data, 552)])
@c.record
class struct_sqtt_code_object_database_record(c.Struct):
  SIZE = 4
  size: int
struct_sqtt_code_object_database_record.register_fields([('size', uint32_t, 0)])
@c.record
class struct_sqtt_file_chunk_code_object_database(c.Struct):
  SIZE = 32
  header: struct_sqtt_file_chunk_header
  offset: int
  flags: int
  size: int
  record_count: int
struct_sqtt_file_chunk_code_object_database.register_fields([('header', struct_sqtt_file_chunk_header, 0), ('offset', uint32_t, 16), ('flags', uint32_t, 20), ('size', uint32_t, 24), ('record_count', uint32_t, 28)])
@c.record
class struct_sqtt_code_object_loader_events_record(c.Struct):
  SIZE = 40
  loader_event_type: int
  reserved: int
  base_address: int
  code_object_hash: c.Array[ctypes.c_uint64, Literal[2]]
  time_stamp: int
struct_sqtt_code_object_loader_events_record.register_fields([('loader_event_type', uint32_t, 0), ('reserved', uint32_t, 4), ('base_address', uint64_t, 8), ('code_object_hash', c.Array[uint64_t, Literal[2]], 16), ('time_stamp', uint64_t, 32)])
@c.record
class struct_sqtt_file_chunk_code_object_loader_events(c.Struct):
  SIZE = 32
  header: struct_sqtt_file_chunk_header
  offset: int
  flags: int
  record_size: int
  record_count: int
struct_sqtt_file_chunk_code_object_loader_events.register_fields([('header', struct_sqtt_file_chunk_header, 0), ('offset', uint32_t, 16), ('flags', uint32_t, 20), ('record_size', uint32_t, 24), ('record_count', uint32_t, 28)])
@c.record
class struct_sqtt_pso_correlation_record(c.Struct):
  SIZE = 88
  api_pso_hash: int
  pipeline_hash: c.Array[ctypes.c_uint64, Literal[2]]
  api_level_obj_name: c.Array[ctypes.c_char, Literal[64]]
struct_sqtt_pso_correlation_record.register_fields([('api_pso_hash', uint64_t, 0), ('pipeline_hash', c.Array[uint64_t, Literal[2]], 8), ('api_level_obj_name', c.Array[ctypes.c_char, Literal[64]], 24)])
@c.record
class struct_sqtt_file_chunk_pso_correlation(c.Struct):
  SIZE = 32
  header: struct_sqtt_file_chunk_header
  offset: int
  flags: int
  record_size: int
  record_count: int
struct_sqtt_file_chunk_pso_correlation.register_fields([('header', struct_sqtt_file_chunk_header, 0), ('offset', uint32_t, 16), ('flags', uint32_t, 20), ('record_size', uint32_t, 24), ('record_count', uint32_t, 28)])
@c.record
class struct_sqtt_file_chunk_sqtt_desc(c.Struct):
  SIZE = 32
  header: struct_sqtt_file_chunk_header
  shader_engine_index: int
  sqtt_version: int
  v0: struct_sqtt_file_chunk_sqtt_desc_v0
  v1: struct_sqtt_file_chunk_sqtt_desc_v1
@c.record
class struct_sqtt_file_chunk_sqtt_desc_v0(c.Struct):
  SIZE = 4
  instrumentation_version: int
struct_sqtt_file_chunk_sqtt_desc_v0.register_fields([('instrumentation_version', int32_t, 0)])
@c.record
class struct_sqtt_file_chunk_sqtt_desc_v1(c.Struct):
  SIZE = 8
  instrumentation_spec_version: int
  instrumentation_api_version: int
  compute_unit_index: int
int16_t: TypeAlias = ctypes.c_int16
struct_sqtt_file_chunk_sqtt_desc_v1.register_fields([('instrumentation_spec_version', int16_t, 0), ('instrumentation_api_version', int16_t, 2), ('compute_unit_index', int32_t, 4)])
struct_sqtt_file_chunk_sqtt_desc.register_fields([('header', struct_sqtt_file_chunk_header, 0), ('shader_engine_index', int32_t, 16), ('sqtt_version', ctypes.c_uint32, 20), ('v0', struct_sqtt_file_chunk_sqtt_desc_v0, 24), ('v1', struct_sqtt_file_chunk_sqtt_desc_v1, 24)])
@c.record
class struct_sqtt_file_chunk_sqtt_data(c.Struct):
  SIZE = 24
  header: struct_sqtt_file_chunk_header
  offset: int
  size: int
struct_sqtt_file_chunk_sqtt_data.register_fields([('header', struct_sqtt_file_chunk_header, 0), ('offset', int32_t, 16), ('size', int32_t, 20)])
@c.record
class struct_sqtt_file_chunk_queue_event_timings(c.Struct):
  SIZE = 32
  header: struct_sqtt_file_chunk_header
  queue_info_table_record_count: int
  queue_info_table_size: int
  queue_event_table_record_count: int
  queue_event_table_size: int
struct_sqtt_file_chunk_queue_event_timings.register_fields([('header', struct_sqtt_file_chunk_header, 0), ('queue_info_table_record_count', uint32_t, 16), ('queue_info_table_size', uint32_t, 20), ('queue_event_table_record_count', uint32_t, 24), ('queue_event_table_size', uint32_t, 28)])
enum_sqtt_queue_type: dict[int, str] = {(SQTT_QUEUE_TYPE_UNKNOWN:=0): 'SQTT_QUEUE_TYPE_UNKNOWN', (SQTT_QUEUE_TYPE_UNIVERSAL:=1): 'SQTT_QUEUE_TYPE_UNIVERSAL', (SQTT_QUEUE_TYPE_COMPUTE:=2): 'SQTT_QUEUE_TYPE_COMPUTE', (SQTT_QUEUE_TYPE_DMA:=3): 'SQTT_QUEUE_TYPE_DMA'}
enum_sqtt_engine_type: dict[int, str] = {(SQTT_ENGINE_TYPE_UNKNOWN:=0): 'SQTT_ENGINE_TYPE_UNKNOWN', (SQTT_ENGINE_TYPE_UNIVERSAL:=1): 'SQTT_ENGINE_TYPE_UNIVERSAL', (SQTT_ENGINE_TYPE_COMPUTE:=2): 'SQTT_ENGINE_TYPE_COMPUTE', (SQTT_ENGINE_TYPE_EXCLUSIVE_COMPUTE:=3): 'SQTT_ENGINE_TYPE_EXCLUSIVE_COMPUTE', (SQTT_ENGINE_TYPE_DMA:=4): 'SQTT_ENGINE_TYPE_DMA', (SQTT_ENGINE_TYPE_HIGH_PRIORITY_UNIVERSAL:=7): 'SQTT_ENGINE_TYPE_HIGH_PRIORITY_UNIVERSAL', (SQTT_ENGINE_TYPE_HIGH_PRIORITY_GRAPHICS:=8): 'SQTT_ENGINE_TYPE_HIGH_PRIORITY_GRAPHICS'}
@c.record
class struct_sqtt_queue_hardware_info(c.Struct):
  SIZE = 4
  queue_type: int
  engine_type: int
  reserved: int
  value: int
struct_sqtt_queue_hardware_info.register_fields([('queue_type', int32_t, 0, 8, 0), ('engine_type', int32_t, 1, 8, 0), ('reserved', uint32_t, 2, 16, 0), ('value', uint32_t, 0)])
@c.record
class struct_sqtt_queue_info_record(c.Struct):
  SIZE = 24
  queue_id: int
  queue_context: int
  hardware_info: struct_sqtt_queue_hardware_info
  reserved: int
struct_sqtt_queue_info_record.register_fields([('queue_id', uint64_t, 0), ('queue_context', uint64_t, 8), ('hardware_info', struct_sqtt_queue_hardware_info, 16), ('reserved', uint32_t, 20)])
enum_sqtt_queue_event_type: dict[int, str] = {(SQTT_QUEUE_TIMING_EVENT_CMDBUF_SUBMIT:=0): 'SQTT_QUEUE_TIMING_EVENT_CMDBUF_SUBMIT', (SQTT_QUEUE_TIMING_EVENT_SIGNAL_SEMAPHORE:=1): 'SQTT_QUEUE_TIMING_EVENT_SIGNAL_SEMAPHORE', (SQTT_QUEUE_TIMING_EVENT_WAIT_SEMAPHORE:=2): 'SQTT_QUEUE_TIMING_EVENT_WAIT_SEMAPHORE', (SQTT_QUEUE_TIMING_EVENT_PRESENT:=3): 'SQTT_QUEUE_TIMING_EVENT_PRESENT'}
@c.record
class struct_sqtt_queue_event_record(c.Struct):
  SIZE = 56
  event_type: int
  sqtt_cb_id: int
  frame_index: int
  queue_info_index: int
  submit_sub_index: int
  api_id: int
  cpu_timestamp: int
  gpu_timestamps: c.Array[ctypes.c_uint64, Literal[2]]
struct_sqtt_queue_event_record.register_fields([('event_type', ctypes.c_uint32, 0), ('sqtt_cb_id', uint32_t, 4), ('frame_index', uint64_t, 8), ('queue_info_index', uint32_t, 16), ('submit_sub_index', uint32_t, 20), ('api_id', uint64_t, 24), ('cpu_timestamp', uint64_t, 32), ('gpu_timestamps', c.Array[uint64_t, Literal[2]], 40)])
@c.record
class struct_sqtt_file_chunk_clock_calibration(c.Struct):
  SIZE = 40
  header: struct_sqtt_file_chunk_header
  cpu_timestamp: int
  gpu_timestamp: int
  reserved: int
struct_sqtt_file_chunk_clock_calibration.register_fields([('header', struct_sqtt_file_chunk_header, 0), ('cpu_timestamp', uint64_t, 16), ('gpu_timestamp', uint64_t, 24), ('reserved', uint64_t, 32)])
enum_elf_gfxip_level: dict[int, str] = {(EF_AMDGPU_MACH_AMDGCN_GFX801:=40): 'EF_AMDGPU_MACH_AMDGCN_GFX801', (EF_AMDGPU_MACH_AMDGCN_GFX900:=44): 'EF_AMDGPU_MACH_AMDGCN_GFX900', (EF_AMDGPU_MACH_AMDGCN_GFX1010:=51): 'EF_AMDGPU_MACH_AMDGCN_GFX1010', (EF_AMDGPU_MACH_AMDGCN_GFX1030:=54): 'EF_AMDGPU_MACH_AMDGCN_GFX1030', (EF_AMDGPU_MACH_AMDGCN_GFX1100:=65): 'EF_AMDGPU_MACH_AMDGCN_GFX1100', (EF_AMDGPU_MACH_AMDGCN_GFX1150:=67): 'EF_AMDGPU_MACH_AMDGCN_GFX1150', (EF_AMDGPU_MACH_AMDGCN_GFX1200:=78): 'EF_AMDGPU_MACH_AMDGCN_GFX1200'}
@c.record
class struct_sqtt_file_chunk_spm_db(c.Struct):
  SIZE = 40
  header: struct_sqtt_file_chunk_header
  flags: int
  preamble_size: int
  num_timestamps: int
  num_spm_counter_info: int
  spm_counter_info_size: int
  sample_interval: int
struct_sqtt_file_chunk_spm_db.register_fields([('header', struct_sqtt_file_chunk_header, 0), ('flags', uint32_t, 16), ('preamble_size', uint32_t, 20), ('num_timestamps', uint32_t, 24), ('num_spm_counter_info', uint32_t, 28), ('spm_counter_info_size', uint32_t, 32), ('sample_interval', uint32_t, 36)])
enum_rgp_sqtt_marker_identifier: dict[int, str] = {(RGP_SQTT_MARKER_IDENTIFIER_EVENT:=0): 'RGP_SQTT_MARKER_IDENTIFIER_EVENT', (RGP_SQTT_MARKER_IDENTIFIER_CB_START:=1): 'RGP_SQTT_MARKER_IDENTIFIER_CB_START', (RGP_SQTT_MARKER_IDENTIFIER_CB_END:=2): 'RGP_SQTT_MARKER_IDENTIFIER_CB_END', (RGP_SQTT_MARKER_IDENTIFIER_BARRIER_START:=3): 'RGP_SQTT_MARKER_IDENTIFIER_BARRIER_START', (RGP_SQTT_MARKER_IDENTIFIER_BARRIER_END:=4): 'RGP_SQTT_MARKER_IDENTIFIER_BARRIER_END', (RGP_SQTT_MARKER_IDENTIFIER_USER_EVENT:=5): 'RGP_SQTT_MARKER_IDENTIFIER_USER_EVENT', (RGP_SQTT_MARKER_IDENTIFIER_GENERAL_API:=6): 'RGP_SQTT_MARKER_IDENTIFIER_GENERAL_API', (RGP_SQTT_MARKER_IDENTIFIER_SYNC:=7): 'RGP_SQTT_MARKER_IDENTIFIER_SYNC', (RGP_SQTT_MARKER_IDENTIFIER_PRESENT:=8): 'RGP_SQTT_MARKER_IDENTIFIER_PRESENT', (RGP_SQTT_MARKER_IDENTIFIER_LAYOUT_TRANSITION:=9): 'RGP_SQTT_MARKER_IDENTIFIER_LAYOUT_TRANSITION', (RGP_SQTT_MARKER_IDENTIFIER_RENDER_PASS:=10): 'RGP_SQTT_MARKER_IDENTIFIER_RENDER_PASS', (RGP_SQTT_MARKER_IDENTIFIER_RESERVED2:=11): 'RGP_SQTT_MARKER_IDENTIFIER_RESERVED2', (RGP_SQTT_MARKER_IDENTIFIER_BIND_PIPELINE:=12): 'RGP_SQTT_MARKER_IDENTIFIER_BIND_PIPELINE', (RGP_SQTT_MARKER_IDENTIFIER_RESERVED4:=13): 'RGP_SQTT_MARKER_IDENTIFIER_RESERVED4', (RGP_SQTT_MARKER_IDENTIFIER_RESERVED5:=14): 'RGP_SQTT_MARKER_IDENTIFIER_RESERVED5', (RGP_SQTT_MARKER_IDENTIFIER_RESERVED6:=15): 'RGP_SQTT_MARKER_IDENTIFIER_RESERVED6'}
@c.record
class union_rgp_sqtt_marker_cb_id(c.Struct):
  SIZE = 4
  per_frame_cb_id: union_rgp_sqtt_marker_cb_id_per_frame_cb_id
  global_cb_id: union_rgp_sqtt_marker_cb_id_global_cb_id
  all: int
@c.record
class union_rgp_sqtt_marker_cb_id_per_frame_cb_id(c.Struct):
  SIZE = 4
  per_frame: int
  frame_index: int
  cb_index: int
  reserved: int
union_rgp_sqtt_marker_cb_id_per_frame_cb_id.register_fields([('per_frame', uint32_t, 0, 1, 0), ('frame_index', uint32_t, 0, 7, 1), ('cb_index', uint32_t, 1, 12, 0), ('reserved', uint32_t, 2, 12, 4)])
@c.record
class union_rgp_sqtt_marker_cb_id_global_cb_id(c.Struct):
  SIZE = 4
  per_frame: int
  cb_index: int
  reserved: int
union_rgp_sqtt_marker_cb_id_global_cb_id.register_fields([('per_frame', uint32_t, 0, 1, 0), ('cb_index', uint32_t, 0, 19, 1), ('reserved', uint32_t, 2, 12, 4)])
union_rgp_sqtt_marker_cb_id.register_fields([('per_frame_cb_id', union_rgp_sqtt_marker_cb_id_per_frame_cb_id, 0), ('global_cb_id', union_rgp_sqtt_marker_cb_id_global_cb_id, 0), ('all', uint32_t, 0)])
@c.record
class struct_rgp_sqtt_marker_cb_start(c.Struct):
  SIZE = 16
  identifier: int
  ext_dwords: int
  cb_id: int
  queue: int
  dword01: int
  device_id_low: int
  dword02: int
  device_id_high: int
  dword03: int
  queue_flags: int
  dword04: int
struct_rgp_sqtt_marker_cb_start.register_fields([('identifier', uint32_t, 0, 4, 0), ('ext_dwords', uint32_t, 0, 3, 4), ('cb_id', uint32_t, 0, 20, 7), ('queue', uint32_t, 3, 5, 3), ('dword01', uint32_t, 0), ('device_id_low', uint32_t, 4), ('dword02', uint32_t, 4), ('device_id_high', uint32_t, 8), ('dword03', uint32_t, 8), ('queue_flags', uint32_t, 12), ('dword04', uint32_t, 12)])
@c.record
class struct_rgp_sqtt_marker_cb_end(c.Struct):
  SIZE = 12
  identifier: int
  ext_dwords: int
  cb_id: int
  reserved: int
  dword01: int
  device_id_low: int
  dword02: int
  device_id_high: int
  dword03: int
struct_rgp_sqtt_marker_cb_end.register_fields([('identifier', uint32_t, 0, 4, 0), ('ext_dwords', uint32_t, 0, 3, 4), ('cb_id', uint32_t, 0, 20, 7), ('reserved', uint32_t, 3, 5, 3), ('dword01', uint32_t, 0), ('device_id_low', uint32_t, 4), ('dword02', uint32_t, 4), ('device_id_high', uint32_t, 8), ('dword03', uint32_t, 8)])
enum_rgp_sqtt_marker_general_api_type: dict[int, str] = {(ApiCmdBindPipeline:=0): 'ApiCmdBindPipeline', (ApiCmdBindDescriptorSets:=1): 'ApiCmdBindDescriptorSets', (ApiCmdBindIndexBuffer:=2): 'ApiCmdBindIndexBuffer', (ApiCmdBindVertexBuffers:=3): 'ApiCmdBindVertexBuffers', (ApiCmdDraw:=4): 'ApiCmdDraw', (ApiCmdDrawIndexed:=5): 'ApiCmdDrawIndexed', (ApiCmdDrawIndirect:=6): 'ApiCmdDrawIndirect', (ApiCmdDrawIndexedIndirect:=7): 'ApiCmdDrawIndexedIndirect', (ApiCmdDrawIndirectCountAMD:=8): 'ApiCmdDrawIndirectCountAMD', (ApiCmdDrawIndexedIndirectCountAMD:=9): 'ApiCmdDrawIndexedIndirectCountAMD', (ApiCmdDispatch:=10): 'ApiCmdDispatch', (ApiCmdDispatchIndirect:=11): 'ApiCmdDispatchIndirect', (ApiCmdCopyBuffer:=12): 'ApiCmdCopyBuffer', (ApiCmdCopyImage:=13): 'ApiCmdCopyImage', (ApiCmdBlitImage:=14): 'ApiCmdBlitImage', (ApiCmdCopyBufferToImage:=15): 'ApiCmdCopyBufferToImage', (ApiCmdCopyImageToBuffer:=16): 'ApiCmdCopyImageToBuffer', (ApiCmdUpdateBuffer:=17): 'ApiCmdUpdateBuffer', (ApiCmdFillBuffer:=18): 'ApiCmdFillBuffer', (ApiCmdClearColorImage:=19): 'ApiCmdClearColorImage', (ApiCmdClearDepthStencilImage:=20): 'ApiCmdClearDepthStencilImage', (ApiCmdClearAttachments:=21): 'ApiCmdClearAttachments', (ApiCmdResolveImage:=22): 'ApiCmdResolveImage', (ApiCmdWaitEvents:=23): 'ApiCmdWaitEvents', (ApiCmdPipelineBarrier:=24): 'ApiCmdPipelineBarrier', (ApiCmdBeginQuery:=25): 'ApiCmdBeginQuery', (ApiCmdEndQuery:=26): 'ApiCmdEndQuery', (ApiCmdResetQueryPool:=27): 'ApiCmdResetQueryPool', (ApiCmdWriteTimestamp:=28): 'ApiCmdWriteTimestamp', (ApiCmdCopyQueryPoolResults:=29): 'ApiCmdCopyQueryPoolResults', (ApiCmdPushConstants:=30): 'ApiCmdPushConstants', (ApiCmdBeginRenderPass:=31): 'ApiCmdBeginRenderPass', (ApiCmdNextSubpass:=32): 'ApiCmdNextSubpass', (ApiCmdEndRenderPass:=33): 'ApiCmdEndRenderPass', (ApiCmdExecuteCommands:=34): 'ApiCmdExecuteCommands', (ApiCmdSetViewport:=35): 'ApiCmdSetViewport', (ApiCmdSetScissor:=36): 'ApiCmdSetScissor', (ApiCmdSetLineWidth:=37): 'ApiCmdSetLineWidth', (ApiCmdSetDepthBias:=38): 'ApiCmdSetDepthBias', (ApiCmdSetBlendConstants:=39): 'ApiCmdSetBlendConstants', (ApiCmdSetDepthBounds:=40): 'ApiCmdSetDepthBounds', (ApiCmdSetStencilCompareMask:=41): 'ApiCmdSetStencilCompareMask', (ApiCmdSetStencilWriteMask:=42): 'ApiCmdSetStencilWriteMask', (ApiCmdSetStencilReference:=43): 'ApiCmdSetStencilReference', (ApiCmdDrawIndirectCount:=44): 'ApiCmdDrawIndirectCount', (ApiCmdDrawIndexedIndirectCount:=45): 'ApiCmdDrawIndexedIndirectCount', (ApiCmdDrawMeshTasksEXT:=47): 'ApiCmdDrawMeshTasksEXT', (ApiCmdDrawMeshTasksIndirectCountEXT:=48): 'ApiCmdDrawMeshTasksIndirectCountEXT', (ApiCmdDrawMeshTasksIndirectEXT:=49): 'ApiCmdDrawMeshTasksIndirectEXT', (ApiRayTracingSeparateCompiled:=8388608): 'ApiRayTracingSeparateCompiled', (ApiInvalid:=4294967295): 'ApiInvalid'}
@c.record
class struct_rgp_sqtt_marker_general_api(c.Struct):
  SIZE = 4
  identifier: int
  ext_dwords: int
  api_type: int
  is_end: int
  reserved: int
  dword01: int
struct_rgp_sqtt_marker_general_api.register_fields([('identifier', uint32_t, 0, 4, 0), ('ext_dwords', uint32_t, 0, 3, 4), ('api_type', uint32_t, 0, 20, 7), ('is_end', uint32_t, 3, 1, 3), ('reserved', uint32_t, 3, 4, 4), ('dword01', uint32_t, 0)])
enum_rgp_sqtt_marker_event_type: dict[int, str] = {(EventCmdDraw:=0): 'EventCmdDraw', (EventCmdDrawIndexed:=1): 'EventCmdDrawIndexed', (EventCmdDrawIndirect:=2): 'EventCmdDrawIndirect', (EventCmdDrawIndexedIndirect:=3): 'EventCmdDrawIndexedIndirect', (EventCmdDrawIndirectCountAMD:=4): 'EventCmdDrawIndirectCountAMD', (EventCmdDrawIndexedIndirectCountAMD:=5): 'EventCmdDrawIndexedIndirectCountAMD', (EventCmdDispatch:=6): 'EventCmdDispatch', (EventCmdDispatchIndirect:=7): 'EventCmdDispatchIndirect', (EventCmdCopyBuffer:=8): 'EventCmdCopyBuffer', (EventCmdCopyImage:=9): 'EventCmdCopyImage', (EventCmdBlitImage:=10): 'EventCmdBlitImage', (EventCmdCopyBufferToImage:=11): 'EventCmdCopyBufferToImage', (EventCmdCopyImageToBuffer:=12): 'EventCmdCopyImageToBuffer', (EventCmdUpdateBuffer:=13): 'EventCmdUpdateBuffer', (EventCmdFillBuffer:=14): 'EventCmdFillBuffer', (EventCmdClearColorImage:=15): 'EventCmdClearColorImage', (EventCmdClearDepthStencilImage:=16): 'EventCmdClearDepthStencilImage', (EventCmdClearAttachments:=17): 'EventCmdClearAttachments', (EventCmdResolveImage:=18): 'EventCmdResolveImage', (EventCmdWaitEvents:=19): 'EventCmdWaitEvents', (EventCmdPipelineBarrier:=20): 'EventCmdPipelineBarrier', (EventCmdResetQueryPool:=21): 'EventCmdResetQueryPool', (EventCmdCopyQueryPoolResults:=22): 'EventCmdCopyQueryPoolResults', (EventRenderPassColorClear:=23): 'EventRenderPassColorClear', (EventRenderPassDepthStencilClear:=24): 'EventRenderPassDepthStencilClear', (EventRenderPassResolve:=25): 'EventRenderPassResolve', (EventInternalUnknown:=26): 'EventInternalUnknown', (EventCmdDrawIndirectCount:=27): 'EventCmdDrawIndirectCount', (EventCmdDrawIndexedIndirectCount:=28): 'EventCmdDrawIndexedIndirectCount', (EventCmdTraceRaysKHR:=30): 'EventCmdTraceRaysKHR', (EventCmdTraceRaysIndirectKHR:=31): 'EventCmdTraceRaysIndirectKHR', (EventCmdBuildAccelerationStructuresKHR:=32): 'EventCmdBuildAccelerationStructuresKHR', (EventCmdBuildAccelerationStructuresIndirectKHR:=33): 'EventCmdBuildAccelerationStructuresIndirectKHR', (EventCmdCopyAccelerationStructureKHR:=34): 'EventCmdCopyAccelerationStructureKHR', (EventCmdCopyAccelerationStructureToMemoryKHR:=35): 'EventCmdCopyAccelerationStructureToMemoryKHR', (EventCmdCopyMemoryToAccelerationStructureKHR:=36): 'EventCmdCopyMemoryToAccelerationStructureKHR', (EventCmdDrawMeshTasksEXT:=41): 'EventCmdDrawMeshTasksEXT', (EventCmdDrawMeshTasksIndirectCountEXT:=42): 'EventCmdDrawMeshTasksIndirectCountEXT', (EventCmdDrawMeshTasksIndirectEXT:=43): 'EventCmdDrawMeshTasksIndirectEXT', (EventUnknown:=32767): 'EventUnknown', (EventInvalid:=4294967295): 'EventInvalid'}
@c.record
class struct_rgp_sqtt_marker_event(c.Struct):
  SIZE = 12
  identifier: int
  ext_dwords: int
  api_type: int
  has_thread_dims: int
  dword01: int
  cb_id: int
  vertex_offset_reg_idx: int
  instance_offset_reg_idx: int
  draw_index_reg_idx: int
  dword02: int
  cmd_id: int
  dword03: int
struct_rgp_sqtt_marker_event.register_fields([('identifier', uint32_t, 0, 4, 0), ('ext_dwords', uint32_t, 0, 3, 4), ('api_type', uint32_t, 0, 24, 7), ('has_thread_dims', uint32_t, 3, 1, 7), ('dword01', uint32_t, 0), ('cb_id', uint32_t, 4, 20, 0), ('vertex_offset_reg_idx', uint32_t, 6, 4, 4), ('instance_offset_reg_idx', uint32_t, 7, 4, 0), ('draw_index_reg_idx', uint32_t, 7, 4, 4), ('dword02', uint32_t, 4), ('cmd_id', uint32_t, 8), ('dword03', uint32_t, 8)])
@c.record
class struct_rgp_sqtt_marker_event_with_dims(c.Struct):
  SIZE = 24
  event: struct_rgp_sqtt_marker_event
  thread_x: int
  thread_y: int
  thread_z: int
struct_rgp_sqtt_marker_event_with_dims.register_fields([('event', struct_rgp_sqtt_marker_event, 0), ('thread_x', uint32_t, 12), ('thread_y', uint32_t, 16), ('thread_z', uint32_t, 20)])
@c.record
class struct_rgp_sqtt_marker_barrier_start(c.Struct):
  SIZE = 8
  identifier: int
  ext_dwords: int
  cb_id: int
  reserved: int
  dword01: int
  driver_reason: int
  internal: int
  dword02: int
struct_rgp_sqtt_marker_barrier_start.register_fields([('identifier', uint32_t, 0, 4, 0), ('ext_dwords', uint32_t, 0, 3, 4), ('cb_id', uint32_t, 0, 20, 7), ('reserved', uint32_t, 3, 5, 3), ('dword01', uint32_t, 0), ('driver_reason', uint32_t, 4, 31, 0), ('internal', uint32_t, 7, 1, 7), ('dword02', uint32_t, 4)])
@c.record
class struct_rgp_sqtt_marker_barrier_end(c.Struct):
  SIZE = 8
  identifier: int
  ext_dwords: int
  cb_id: int
  wait_on_eop_ts: int
  vs_partial_flush: int
  ps_partial_flush: int
  cs_partial_flush: int
  pfp_sync_me: int
  dword01: int
  sync_cp_dma: int
  inval_tcp: int
  inval_sqI: int
  inval_sqK: int
  flush_tcc: int
  inval_tcc: int
  flush_cb: int
  inval_cb: int
  flush_db: int
  inval_db: int
  num_layout_transitions: int
  inval_gl1: int
  wait_on_ts: int
  eop_ts_bottom_of_pipe: int
  eos_ts_ps_done: int
  eos_ts_cs_done: int
  reserved: int
  dword02: int
struct_rgp_sqtt_marker_barrier_end.register_fields([('identifier', uint32_t, 0, 4, 0), ('ext_dwords', uint32_t, 0, 3, 4), ('cb_id', uint32_t, 0, 20, 7), ('wait_on_eop_ts', uint32_t, 3, 1, 3), ('vs_partial_flush', uint32_t, 3, 1, 4), ('ps_partial_flush', uint32_t, 3, 1, 5), ('cs_partial_flush', uint32_t, 3, 1, 6), ('pfp_sync_me', uint32_t, 3, 1, 7), ('dword01', uint32_t, 0), ('sync_cp_dma', uint32_t, 4, 1, 0), ('inval_tcp', uint32_t, 4, 1, 1), ('inval_sqI', uint32_t, 4, 1, 2), ('inval_sqK', uint32_t, 4, 1, 3), ('flush_tcc', uint32_t, 4, 1, 4), ('inval_tcc', uint32_t, 4, 1, 5), ('flush_cb', uint32_t, 4, 1, 6), ('inval_cb', uint32_t, 4, 1, 7), ('flush_db', uint32_t, 5, 1, 0), ('inval_db', uint32_t, 5, 1, 1), ('num_layout_transitions', uint32_t, 5, 16, 2), ('inval_gl1', uint32_t, 7, 1, 2), ('wait_on_ts', uint32_t, 7, 1, 3), ('eop_ts_bottom_of_pipe', uint32_t, 7, 1, 4), ('eos_ts_ps_done', uint32_t, 7, 1, 5), ('eos_ts_cs_done', uint32_t, 7, 1, 6), ('reserved', uint32_t, 7, 1, 7), ('dword02', uint32_t, 4)])
@c.record
class struct_rgp_sqtt_marker_layout_transition(c.Struct):
  SIZE = 8
  identifier: int
  ext_dwords: int
  depth_stencil_expand: int
  htile_hiz_range_expand: int
  depth_stencil_resummarize: int
  dcc_decompress: int
  fmask_decompress: int
  fast_clear_eliminate: int
  fmask_color_expand: int
  init_mask_ram: int
  reserved1: int
  dword01: int
  reserved2: int
  dword02: int
struct_rgp_sqtt_marker_layout_transition.register_fields([('identifier', uint32_t, 0, 4, 0), ('ext_dwords', uint32_t, 0, 3, 4), ('depth_stencil_expand', uint32_t, 0, 1, 7), ('htile_hiz_range_expand', uint32_t, 1, 1, 0), ('depth_stencil_resummarize', uint32_t, 1, 1, 1), ('dcc_decompress', uint32_t, 1, 1, 2), ('fmask_decompress', uint32_t, 1, 1, 3), ('fast_clear_eliminate', uint32_t, 1, 1, 4), ('fmask_color_expand', uint32_t, 1, 1, 5), ('init_mask_ram', uint32_t, 1, 1, 6), ('reserved1', uint32_t, 1, 17, 7), ('dword01', uint32_t, 0), ('reserved2', uint32_t, 4, 32, 0), ('dword02', uint32_t, 4)])
@c.record
class struct_rgp_sqtt_marker_user_event(c.Struct):
  SIZE = 4
  identifier: int
  reserved0: int
  data_type: int
  reserved1: int
  dword01: int
struct_rgp_sqtt_marker_user_event.register_fields([('identifier', uint32_t, 0, 4, 0), ('reserved0', uint32_t, 0, 8, 4), ('data_type', uint32_t, 1, 8, 4), ('reserved1', uint32_t, 2, 12, 4), ('dword01', uint32_t, 0)])
@c.record
class struct_rgp_sqtt_marker_user_event_with_length(c.Struct):
  SIZE = 8
  user_event: struct_rgp_sqtt_marker_user_event
  length: int
struct_rgp_sqtt_marker_user_event_with_length.register_fields([('user_event', struct_rgp_sqtt_marker_user_event, 0), ('length', uint32_t, 4)])
enum_rgp_sqtt_marker_user_event_type: dict[int, str] = {(UserEventTrigger:=0): 'UserEventTrigger', (UserEventPop:=1): 'UserEventPop', (UserEventPush:=2): 'UserEventPush', (UserEventObjectName:=3): 'UserEventObjectName'}
@c.record
class struct_rgp_sqtt_marker_pipeline_bind(c.Struct):
  SIZE = 12
  identifier: int
  ext_dwords: int
  bind_point: int
  cb_id: int
  reserved: int
  dword01: int
  api_pso_hash: c.Array[ctypes.c_uint32, Literal[2]]
  dword02: int
  dword03: int
struct_rgp_sqtt_marker_pipeline_bind.register_fields([('identifier', uint32_t, 0, 4, 0), ('ext_dwords', uint32_t, 0, 3, 4), ('bind_point', uint32_t, 0, 1, 7), ('cb_id', uint32_t, 1, 20, 0), ('reserved', uint32_t, 3, 4, 4), ('dword01', uint32_t, 0), ('api_pso_hash', c.Array[uint32_t, Literal[2]], 4), ('dword02', uint32_t, 4), ('dword03', uint32_t, 8)])
SQTT_FILE_MAGIC_NUMBER = 0x50303042
SQTT_FILE_VERSION_MAJOR = 1
SQTT_FILE_VERSION_MINOR = 5
SQTT_GPU_NAME_MAX_SIZE = 256
SQTT_MAX_NUM_SE = 32
SQTT_SA_PER_SE = 2
SQTT_ACTIVE_PIXEL_PACKER_MASK_DWORDS = 4