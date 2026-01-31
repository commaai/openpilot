# mypy: disable-error-code="empty-body"
from __future__ import annotations
import ctypes
from typing import Annotated, Literal, TypeAlias
from tinygrad.runtime.support.c import _IO, _IOW, _IOR, _IOWR
from tinygrad.runtime.support import c
@c.record
class struct_sqtt_data_info(c.Struct):
  SIZE = 12
  cur_offset: Annotated[uint32_t, 0]
  trace_status: Annotated[uint32_t, 4]
  gfx9_write_counter: Annotated[uint32_t, 8]
  gfx10_dropped_cntr: Annotated[uint32_t, 8]
uint32_t: TypeAlias = Annotated[int, ctypes.c_uint32]
@c.record
class struct_sqtt_data_se(c.Struct):
  SIZE = 32
  info: Annotated[struct_sqtt_data_info, 0]
  data_ptr: Annotated[ctypes.c_void_p, 16]
  shader_engine: Annotated[uint32_t, 24]
  compute_unit: Annotated[uint32_t, 28]
class enum_sqtt_version(Annotated[int, ctypes.c_uint32], c.Enum): pass
SQTT_VERSION_NONE = enum_sqtt_version.define('SQTT_VERSION_NONE', 0)
SQTT_VERSION_2_2 = enum_sqtt_version.define('SQTT_VERSION_2_2', 5)
SQTT_VERSION_2_3 = enum_sqtt_version.define('SQTT_VERSION_2_3', 6)
SQTT_VERSION_2_4 = enum_sqtt_version.define('SQTT_VERSION_2_4', 7)
SQTT_VERSION_3_2 = enum_sqtt_version.define('SQTT_VERSION_3_2', 11)
SQTT_VERSION_3_3 = enum_sqtt_version.define('SQTT_VERSION_3_3', 12)

class enum_sqtt_file_chunk_type(Annotated[int, ctypes.c_uint32], c.Enum): pass
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

@c.record
class struct_sqtt_file_chunk_id(c.Struct):
  SIZE = 4
  type: Annotated[int32_t, 0, 8, 0]
  index: Annotated[int32_t, 1, 8, 0]
  reserved: Annotated[int32_t, 2, 16, 0]
int32_t: TypeAlias = Annotated[int, ctypes.c_int32]
@c.record
class struct_sqtt_file_chunk_header(c.Struct):
  SIZE = 16
  chunk_id: Annotated[struct_sqtt_file_chunk_id, 0]
  minor_version: Annotated[uint16_t, 4]
  major_version: Annotated[uint16_t, 6]
  size_in_bytes: Annotated[int32_t, 8]
  padding: Annotated[int32_t, 12]
uint16_t: TypeAlias = Annotated[int, ctypes.c_uint16]
@c.record
class struct_sqtt_file_header_flags(c.Struct):
  SIZE = 4
  is_semaphore_queue_timing_etw: Annotated[uint32_t, 0, 1, 0]
  no_queue_semaphore_timestamps: Annotated[uint32_t, 0, 1, 1]
  reserved: Annotated[uint32_t, 0, 30, 2]
  value: Annotated[uint32_t, 0]
@c.record
class struct_sqtt_file_header(c.Struct):
  SIZE = 56
  magic_number: Annotated[uint32_t, 0]
  version_major: Annotated[uint32_t, 4]
  version_minor: Annotated[uint32_t, 8]
  flags: Annotated[struct_sqtt_file_header_flags, 12]
  chunk_offset: Annotated[int32_t, 16]
  second: Annotated[int32_t, 20]
  minute: Annotated[int32_t, 24]
  hour: Annotated[int32_t, 28]
  day_in_month: Annotated[int32_t, 32]
  month: Annotated[int32_t, 36]
  year: Annotated[int32_t, 40]
  day_in_week: Annotated[int32_t, 44]
  day_in_year: Annotated[int32_t, 48]
  is_daylight_savings: Annotated[int32_t, 52]
@c.record
class struct_sqtt_file_chunk_cpu_info(c.Struct):
  SIZE = 112
  header: Annotated[struct_sqtt_file_chunk_header, 0]
  vendor_id: Annotated[c.Array[uint32_t, Literal[4]], 16]
  processor_brand: Annotated[c.Array[uint32_t, Literal[12]], 32]
  reserved: Annotated[c.Array[uint32_t, Literal[2]], 80]
  cpu_timestamp_freq: Annotated[uint64_t, 88]
  clock_speed: Annotated[uint32_t, 96]
  num_logical_cores: Annotated[uint32_t, 100]
  num_physical_cores: Annotated[uint32_t, 104]
  system_ram_size: Annotated[uint32_t, 108]
uint64_t: TypeAlias = Annotated[int, ctypes.c_uint64]
class enum_sqtt_file_chunk_asic_info_flags(Annotated[int, ctypes.c_uint32], c.Enum): pass
SQTT_FILE_CHUNK_ASIC_INFO_FLAG_SC_PACKER_NUMBERING = enum_sqtt_file_chunk_asic_info_flags.define('SQTT_FILE_CHUNK_ASIC_INFO_FLAG_SC_PACKER_NUMBERING', 1)
SQTT_FILE_CHUNK_ASIC_INFO_FLAG_PS1_EVENT_TOKENS_ENABLED = enum_sqtt_file_chunk_asic_info_flags.define('SQTT_FILE_CHUNK_ASIC_INFO_FLAG_PS1_EVENT_TOKENS_ENABLED', 2)

class enum_sqtt_gpu_type(Annotated[int, ctypes.c_uint32], c.Enum): pass
SQTT_GPU_TYPE_UNKNOWN = enum_sqtt_gpu_type.define('SQTT_GPU_TYPE_UNKNOWN', 0)
SQTT_GPU_TYPE_INTEGRATED = enum_sqtt_gpu_type.define('SQTT_GPU_TYPE_INTEGRATED', 1)
SQTT_GPU_TYPE_DISCRETE = enum_sqtt_gpu_type.define('SQTT_GPU_TYPE_DISCRETE', 2)
SQTT_GPU_TYPE_VIRTUAL = enum_sqtt_gpu_type.define('SQTT_GPU_TYPE_VIRTUAL', 3)

class enum_sqtt_gfxip_level(Annotated[int, ctypes.c_uint32], c.Enum): pass
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

class enum_sqtt_memory_type(Annotated[int, ctypes.c_uint32], c.Enum): pass
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

@c.record
class struct_sqtt_file_chunk_asic_info(c.Struct):
  SIZE = 768
  header: Annotated[struct_sqtt_file_chunk_header, 0]
  flags: Annotated[uint64_t, 16]
  trace_shader_core_clock: Annotated[uint64_t, 24]
  trace_memory_clock: Annotated[uint64_t, 32]
  device_id: Annotated[int32_t, 40]
  device_revision_id: Annotated[int32_t, 44]
  vgprs_per_simd: Annotated[int32_t, 48]
  sgprs_per_simd: Annotated[int32_t, 52]
  shader_engines: Annotated[int32_t, 56]
  compute_unit_per_shader_engine: Annotated[int32_t, 60]
  simd_per_compute_unit: Annotated[int32_t, 64]
  wavefronts_per_simd: Annotated[int32_t, 68]
  minimum_vgpr_alloc: Annotated[int32_t, 72]
  vgpr_alloc_granularity: Annotated[int32_t, 76]
  minimum_sgpr_alloc: Annotated[int32_t, 80]
  sgpr_alloc_granularity: Annotated[int32_t, 84]
  hardware_contexts: Annotated[int32_t, 88]
  gpu_type: Annotated[enum_sqtt_gpu_type, 92]
  gfxip_level: Annotated[enum_sqtt_gfxip_level, 96]
  gpu_index: Annotated[int32_t, 100]
  gds_size: Annotated[int32_t, 104]
  gds_per_shader_engine: Annotated[int32_t, 108]
  ce_ram_size: Annotated[int32_t, 112]
  ce_ram_size_graphics: Annotated[int32_t, 116]
  ce_ram_size_compute: Annotated[int32_t, 120]
  max_number_of_dedicated_cus: Annotated[int32_t, 124]
  vram_size: Annotated[int64_t, 128]
  vram_bus_width: Annotated[int32_t, 136]
  l2_cache_size: Annotated[int32_t, 140]
  l1_cache_size: Annotated[int32_t, 144]
  lds_size: Annotated[int32_t, 148]
  gpu_name: Annotated[c.Array[Annotated[bytes, ctypes.c_char], Literal[256]], 152]
  alu_per_clock: Annotated[Annotated[float, ctypes.c_float], 408]
  texture_per_clock: Annotated[Annotated[float, ctypes.c_float], 412]
  prims_per_clock: Annotated[Annotated[float, ctypes.c_float], 416]
  pixels_per_clock: Annotated[Annotated[float, ctypes.c_float], 420]
  gpu_timestamp_frequency: Annotated[uint64_t, 424]
  max_shader_core_clock: Annotated[uint64_t, 432]
  max_memory_clock: Annotated[uint64_t, 440]
  memory_ops_per_clock: Annotated[uint32_t, 448]
  memory_chip_type: Annotated[enum_sqtt_memory_type, 452]
  lds_granularity: Annotated[uint32_t, 456]
  cu_mask: Annotated[c.Array[c.Array[uint16_t, Literal[2]], Literal[32]], 460]
  reserved1: Annotated[c.Array[Annotated[bytes, ctypes.c_char], Literal[128]], 588]
  active_pixel_packer_mask: Annotated[c.Array[uint32_t, Literal[4]], 716]
  reserved2: Annotated[c.Array[Annotated[bytes, ctypes.c_char], Literal[16]], 732]
  gl1_cache_size: Annotated[uint32_t, 748]
  instruction_cache_size: Annotated[uint32_t, 752]
  scalar_cache_size: Annotated[uint32_t, 756]
  mall_cache_size: Annotated[uint32_t, 760]
  padding: Annotated[c.Array[Annotated[bytes, ctypes.c_char], Literal[4]], 764]
int64_t: TypeAlias = Annotated[int, ctypes.c_int64]
class enum_sqtt_api_type(Annotated[int, ctypes.c_uint32], c.Enum): pass
SQTT_API_TYPE_DIRECTX_12 = enum_sqtt_api_type.define('SQTT_API_TYPE_DIRECTX_12', 0)
SQTT_API_TYPE_VULKAN = enum_sqtt_api_type.define('SQTT_API_TYPE_VULKAN', 1)
SQTT_API_TYPE_GENERIC = enum_sqtt_api_type.define('SQTT_API_TYPE_GENERIC', 2)
SQTT_API_TYPE_OPENCL = enum_sqtt_api_type.define('SQTT_API_TYPE_OPENCL', 3)

class enum_sqtt_instruction_trace_mode(Annotated[int, ctypes.c_uint32], c.Enum): pass
SQTT_INSTRUCTION_TRACE_DISABLED = enum_sqtt_instruction_trace_mode.define('SQTT_INSTRUCTION_TRACE_DISABLED', 0)
SQTT_INSTRUCTION_TRACE_FULL_FRAME = enum_sqtt_instruction_trace_mode.define('SQTT_INSTRUCTION_TRACE_FULL_FRAME', 1)
SQTT_INSTRUCTION_TRACE_API_PSO = enum_sqtt_instruction_trace_mode.define('SQTT_INSTRUCTION_TRACE_API_PSO', 2)

class enum_sqtt_profiling_mode(Annotated[int, ctypes.c_uint32], c.Enum): pass
SQTT_PROFILING_MODE_PRESENT = enum_sqtt_profiling_mode.define('SQTT_PROFILING_MODE_PRESENT', 0)
SQTT_PROFILING_MODE_USER_MARKERS = enum_sqtt_profiling_mode.define('SQTT_PROFILING_MODE_USER_MARKERS', 1)
SQTT_PROFILING_MODE_INDEX = enum_sqtt_profiling_mode.define('SQTT_PROFILING_MODE_INDEX', 2)
SQTT_PROFILING_MODE_TAG = enum_sqtt_profiling_mode.define('SQTT_PROFILING_MODE_TAG', 3)

@c.record
class union_sqtt_profiling_mode_data(c.Struct):
  SIZE = 512
  user_marker_profiling_data: Annotated[union_sqtt_profiling_mode_data_user_marker_profiling_data, 0]
  index_profiling_data: Annotated[union_sqtt_profiling_mode_data_index_profiling_data, 0]
  tag_profiling_data: Annotated[union_sqtt_profiling_mode_data_tag_profiling_data, 0]
@c.record
class union_sqtt_profiling_mode_data_user_marker_profiling_data(c.Struct):
  SIZE = 512
  start: Annotated[c.Array[Annotated[bytes, ctypes.c_char], Literal[256]], 0]
  end: Annotated[c.Array[Annotated[bytes, ctypes.c_char], Literal[256]], 256]
@c.record
class union_sqtt_profiling_mode_data_index_profiling_data(c.Struct):
  SIZE = 8
  start: Annotated[uint32_t, 0]
  end: Annotated[uint32_t, 4]
@c.record
class union_sqtt_profiling_mode_data_tag_profiling_data(c.Struct):
  SIZE = 16
  begin_hi: Annotated[uint32_t, 0]
  begin_lo: Annotated[uint32_t, 4]
  end_hi: Annotated[uint32_t, 8]
  end_lo: Annotated[uint32_t, 12]
@c.record
class union_sqtt_instruction_trace_data(c.Struct):
  SIZE = 8
  api_pso_data: Annotated[union_sqtt_instruction_trace_data_api_pso_data, 0]
  shader_engine_filter: Annotated[union_sqtt_instruction_trace_data_shader_engine_filter, 0]
@c.record
class union_sqtt_instruction_trace_data_api_pso_data(c.Struct):
  SIZE = 8
  api_pso_filter: Annotated[uint64_t, 0]
@c.record
class union_sqtt_instruction_trace_data_shader_engine_filter(c.Struct):
  SIZE = 4
  mask: Annotated[uint32_t, 0]
@c.record
class struct_sqtt_file_chunk_api_info(c.Struct):
  SIZE = 560
  header: Annotated[struct_sqtt_file_chunk_header, 0]
  api_type: Annotated[enum_sqtt_api_type, 16]
  major_version: Annotated[uint16_t, 20]
  minor_version: Annotated[uint16_t, 22]
  profiling_mode: Annotated[enum_sqtt_profiling_mode, 24]
  reserved: Annotated[uint32_t, 28]
  profiling_mode_data: Annotated[union_sqtt_profiling_mode_data, 32]
  instruction_trace_mode: Annotated[enum_sqtt_instruction_trace_mode, 544]
  reserved2: Annotated[uint32_t, 548]
  instruction_trace_data: Annotated[union_sqtt_instruction_trace_data, 552]
@c.record
class struct_sqtt_code_object_database_record(c.Struct):
  SIZE = 4
  size: Annotated[uint32_t, 0]
@c.record
class struct_sqtt_file_chunk_code_object_database(c.Struct):
  SIZE = 32
  header: Annotated[struct_sqtt_file_chunk_header, 0]
  offset: Annotated[uint32_t, 16]
  flags: Annotated[uint32_t, 20]
  size: Annotated[uint32_t, 24]
  record_count: Annotated[uint32_t, 28]
@c.record
class struct_sqtt_code_object_loader_events_record(c.Struct):
  SIZE = 40
  loader_event_type: Annotated[uint32_t, 0]
  reserved: Annotated[uint32_t, 4]
  base_address: Annotated[uint64_t, 8]
  code_object_hash: Annotated[c.Array[uint64_t, Literal[2]], 16]
  time_stamp: Annotated[uint64_t, 32]
@c.record
class struct_sqtt_file_chunk_code_object_loader_events(c.Struct):
  SIZE = 32
  header: Annotated[struct_sqtt_file_chunk_header, 0]
  offset: Annotated[uint32_t, 16]
  flags: Annotated[uint32_t, 20]
  record_size: Annotated[uint32_t, 24]
  record_count: Annotated[uint32_t, 28]
@c.record
class struct_sqtt_pso_correlation_record(c.Struct):
  SIZE = 88
  api_pso_hash: Annotated[uint64_t, 0]
  pipeline_hash: Annotated[c.Array[uint64_t, Literal[2]], 8]
  api_level_obj_name: Annotated[c.Array[Annotated[bytes, ctypes.c_char], Literal[64]], 24]
@c.record
class struct_sqtt_file_chunk_pso_correlation(c.Struct):
  SIZE = 32
  header: Annotated[struct_sqtt_file_chunk_header, 0]
  offset: Annotated[uint32_t, 16]
  flags: Annotated[uint32_t, 20]
  record_size: Annotated[uint32_t, 24]
  record_count: Annotated[uint32_t, 28]
@c.record
class struct_sqtt_file_chunk_sqtt_desc(c.Struct):
  SIZE = 32
  header: Annotated[struct_sqtt_file_chunk_header, 0]
  shader_engine_index: Annotated[int32_t, 16]
  sqtt_version: Annotated[enum_sqtt_version, 20]
  v0: Annotated[struct_sqtt_file_chunk_sqtt_desc_v0, 24]
  v1: Annotated[struct_sqtt_file_chunk_sqtt_desc_v1, 24]
@c.record
class struct_sqtt_file_chunk_sqtt_desc_v0(c.Struct):
  SIZE = 4
  instrumentation_version: Annotated[int32_t, 0]
@c.record
class struct_sqtt_file_chunk_sqtt_desc_v1(c.Struct):
  SIZE = 8
  instrumentation_spec_version: Annotated[int16_t, 0]
  instrumentation_api_version: Annotated[int16_t, 2]
  compute_unit_index: Annotated[int32_t, 4]
int16_t: TypeAlias = Annotated[int, ctypes.c_int16]
@c.record
class struct_sqtt_file_chunk_sqtt_data(c.Struct):
  SIZE = 24
  header: Annotated[struct_sqtt_file_chunk_header, 0]
  offset: Annotated[int32_t, 16]
  size: Annotated[int32_t, 20]
@c.record
class struct_sqtt_file_chunk_queue_event_timings(c.Struct):
  SIZE = 32
  header: Annotated[struct_sqtt_file_chunk_header, 0]
  queue_info_table_record_count: Annotated[uint32_t, 16]
  queue_info_table_size: Annotated[uint32_t, 20]
  queue_event_table_record_count: Annotated[uint32_t, 24]
  queue_event_table_size: Annotated[uint32_t, 28]
class enum_sqtt_queue_type(Annotated[int, ctypes.c_uint32], c.Enum): pass
SQTT_QUEUE_TYPE_UNKNOWN = enum_sqtt_queue_type.define('SQTT_QUEUE_TYPE_UNKNOWN', 0)
SQTT_QUEUE_TYPE_UNIVERSAL = enum_sqtt_queue_type.define('SQTT_QUEUE_TYPE_UNIVERSAL', 1)
SQTT_QUEUE_TYPE_COMPUTE = enum_sqtt_queue_type.define('SQTT_QUEUE_TYPE_COMPUTE', 2)
SQTT_QUEUE_TYPE_DMA = enum_sqtt_queue_type.define('SQTT_QUEUE_TYPE_DMA', 3)

class enum_sqtt_engine_type(Annotated[int, ctypes.c_uint32], c.Enum): pass
SQTT_ENGINE_TYPE_UNKNOWN = enum_sqtt_engine_type.define('SQTT_ENGINE_TYPE_UNKNOWN', 0)
SQTT_ENGINE_TYPE_UNIVERSAL = enum_sqtt_engine_type.define('SQTT_ENGINE_TYPE_UNIVERSAL', 1)
SQTT_ENGINE_TYPE_COMPUTE = enum_sqtt_engine_type.define('SQTT_ENGINE_TYPE_COMPUTE', 2)
SQTT_ENGINE_TYPE_EXCLUSIVE_COMPUTE = enum_sqtt_engine_type.define('SQTT_ENGINE_TYPE_EXCLUSIVE_COMPUTE', 3)
SQTT_ENGINE_TYPE_DMA = enum_sqtt_engine_type.define('SQTT_ENGINE_TYPE_DMA', 4)
SQTT_ENGINE_TYPE_HIGH_PRIORITY_UNIVERSAL = enum_sqtt_engine_type.define('SQTT_ENGINE_TYPE_HIGH_PRIORITY_UNIVERSAL', 7)
SQTT_ENGINE_TYPE_HIGH_PRIORITY_GRAPHICS = enum_sqtt_engine_type.define('SQTT_ENGINE_TYPE_HIGH_PRIORITY_GRAPHICS', 8)

@c.record
class struct_sqtt_queue_hardware_info(c.Struct):
  SIZE = 4
  queue_type: Annotated[int32_t, 0, 8, 0]
  engine_type: Annotated[int32_t, 1, 8, 0]
  reserved: Annotated[uint32_t, 2, 16, 0]
  value: Annotated[uint32_t, 0]
@c.record
class struct_sqtt_queue_info_record(c.Struct):
  SIZE = 24
  queue_id: Annotated[uint64_t, 0]
  queue_context: Annotated[uint64_t, 8]
  hardware_info: Annotated[struct_sqtt_queue_hardware_info, 16]
  reserved: Annotated[uint32_t, 20]
class enum_sqtt_queue_event_type(Annotated[int, ctypes.c_uint32], c.Enum): pass
SQTT_QUEUE_TIMING_EVENT_CMDBUF_SUBMIT = enum_sqtt_queue_event_type.define('SQTT_QUEUE_TIMING_EVENT_CMDBUF_SUBMIT', 0)
SQTT_QUEUE_TIMING_EVENT_SIGNAL_SEMAPHORE = enum_sqtt_queue_event_type.define('SQTT_QUEUE_TIMING_EVENT_SIGNAL_SEMAPHORE', 1)
SQTT_QUEUE_TIMING_EVENT_WAIT_SEMAPHORE = enum_sqtt_queue_event_type.define('SQTT_QUEUE_TIMING_EVENT_WAIT_SEMAPHORE', 2)
SQTT_QUEUE_TIMING_EVENT_PRESENT = enum_sqtt_queue_event_type.define('SQTT_QUEUE_TIMING_EVENT_PRESENT', 3)

@c.record
class struct_sqtt_queue_event_record(c.Struct):
  SIZE = 56
  event_type: Annotated[enum_sqtt_queue_event_type, 0]
  sqtt_cb_id: Annotated[uint32_t, 4]
  frame_index: Annotated[uint64_t, 8]
  queue_info_index: Annotated[uint32_t, 16]
  submit_sub_index: Annotated[uint32_t, 20]
  api_id: Annotated[uint64_t, 24]
  cpu_timestamp: Annotated[uint64_t, 32]
  gpu_timestamps: Annotated[c.Array[uint64_t, Literal[2]], 40]
@c.record
class struct_sqtt_file_chunk_clock_calibration(c.Struct):
  SIZE = 40
  header: Annotated[struct_sqtt_file_chunk_header, 0]
  cpu_timestamp: Annotated[uint64_t, 16]
  gpu_timestamp: Annotated[uint64_t, 24]
  reserved: Annotated[uint64_t, 32]
class enum_elf_gfxip_level(Annotated[int, ctypes.c_uint32], c.Enum): pass
EF_AMDGPU_MACH_AMDGCN_GFX801 = enum_elf_gfxip_level.define('EF_AMDGPU_MACH_AMDGCN_GFX801', 40)
EF_AMDGPU_MACH_AMDGCN_GFX900 = enum_elf_gfxip_level.define('EF_AMDGPU_MACH_AMDGCN_GFX900', 44)
EF_AMDGPU_MACH_AMDGCN_GFX1010 = enum_elf_gfxip_level.define('EF_AMDGPU_MACH_AMDGCN_GFX1010', 51)
EF_AMDGPU_MACH_AMDGCN_GFX1030 = enum_elf_gfxip_level.define('EF_AMDGPU_MACH_AMDGCN_GFX1030', 54)
EF_AMDGPU_MACH_AMDGCN_GFX1100 = enum_elf_gfxip_level.define('EF_AMDGPU_MACH_AMDGCN_GFX1100', 65)
EF_AMDGPU_MACH_AMDGCN_GFX1150 = enum_elf_gfxip_level.define('EF_AMDGPU_MACH_AMDGCN_GFX1150', 67)
EF_AMDGPU_MACH_AMDGCN_GFX1200 = enum_elf_gfxip_level.define('EF_AMDGPU_MACH_AMDGCN_GFX1200', 78)

@c.record
class struct_sqtt_file_chunk_spm_db(c.Struct):
  SIZE = 40
  header: Annotated[struct_sqtt_file_chunk_header, 0]
  flags: Annotated[uint32_t, 16]
  preamble_size: Annotated[uint32_t, 20]
  num_timestamps: Annotated[uint32_t, 24]
  num_spm_counter_info: Annotated[uint32_t, 28]
  spm_counter_info_size: Annotated[uint32_t, 32]
  sample_interval: Annotated[uint32_t, 36]
class enum_rgp_sqtt_marker_identifier(Annotated[int, ctypes.c_uint32], c.Enum): pass
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

@c.record
class union_rgp_sqtt_marker_cb_id(c.Struct):
  SIZE = 4
  per_frame_cb_id: Annotated[union_rgp_sqtt_marker_cb_id_per_frame_cb_id, 0]
  global_cb_id: Annotated[union_rgp_sqtt_marker_cb_id_global_cb_id, 0]
  all: Annotated[uint32_t, 0]
@c.record
class union_rgp_sqtt_marker_cb_id_per_frame_cb_id(c.Struct):
  SIZE = 4
  per_frame: Annotated[uint32_t, 0, 1, 0]
  frame_index: Annotated[uint32_t, 0, 7, 1]
  cb_index: Annotated[uint32_t, 1, 12, 0]
  reserved: Annotated[uint32_t, 2, 12, 4]
@c.record
class union_rgp_sqtt_marker_cb_id_global_cb_id(c.Struct):
  SIZE = 4
  per_frame: Annotated[uint32_t, 0, 1, 0]
  cb_index: Annotated[uint32_t, 0, 19, 1]
  reserved: Annotated[uint32_t, 2, 12, 4]
@c.record
class struct_rgp_sqtt_marker_cb_start(c.Struct):
  SIZE = 16
  identifier: Annotated[uint32_t, 0, 4, 0]
  ext_dwords: Annotated[uint32_t, 0, 3, 4]
  cb_id: Annotated[uint32_t, 0, 20, 7]
  queue: Annotated[uint32_t, 3, 5, 3]
  dword01: Annotated[uint32_t, 0]
  device_id_low: Annotated[uint32_t, 4]
  dword02: Annotated[uint32_t, 4]
  device_id_high: Annotated[uint32_t, 8]
  dword03: Annotated[uint32_t, 8]
  queue_flags: Annotated[uint32_t, 12]
  dword04: Annotated[uint32_t, 12]
@c.record
class struct_rgp_sqtt_marker_cb_end(c.Struct):
  SIZE = 12
  identifier: Annotated[uint32_t, 0, 4, 0]
  ext_dwords: Annotated[uint32_t, 0, 3, 4]
  cb_id: Annotated[uint32_t, 0, 20, 7]
  reserved: Annotated[uint32_t, 3, 5, 3]
  dword01: Annotated[uint32_t, 0]
  device_id_low: Annotated[uint32_t, 4]
  dword02: Annotated[uint32_t, 4]
  device_id_high: Annotated[uint32_t, 8]
  dword03: Annotated[uint32_t, 8]
class enum_rgp_sqtt_marker_general_api_type(Annotated[int, ctypes.c_uint32], c.Enum): pass
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

@c.record
class struct_rgp_sqtt_marker_general_api(c.Struct):
  SIZE = 4
  identifier: Annotated[uint32_t, 0, 4, 0]
  ext_dwords: Annotated[uint32_t, 0, 3, 4]
  api_type: Annotated[uint32_t, 0, 20, 7]
  is_end: Annotated[uint32_t, 3, 1, 3]
  reserved: Annotated[uint32_t, 3, 4, 4]
  dword01: Annotated[uint32_t, 0]
class enum_rgp_sqtt_marker_event_type(Annotated[int, ctypes.c_uint32], c.Enum): pass
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

@c.record
class struct_rgp_sqtt_marker_event(c.Struct):
  SIZE = 12
  identifier: Annotated[uint32_t, 0, 4, 0]
  ext_dwords: Annotated[uint32_t, 0, 3, 4]
  api_type: Annotated[uint32_t, 0, 24, 7]
  has_thread_dims: Annotated[uint32_t, 3, 1, 7]
  dword01: Annotated[uint32_t, 0]
  cb_id: Annotated[uint32_t, 4, 20, 0]
  vertex_offset_reg_idx: Annotated[uint32_t, 6, 4, 4]
  instance_offset_reg_idx: Annotated[uint32_t, 7, 4, 0]
  draw_index_reg_idx: Annotated[uint32_t, 7, 4, 4]
  dword02: Annotated[uint32_t, 4]
  cmd_id: Annotated[uint32_t, 8]
  dword03: Annotated[uint32_t, 8]
@c.record
class struct_rgp_sqtt_marker_event_with_dims(c.Struct):
  SIZE = 24
  event: Annotated[struct_rgp_sqtt_marker_event, 0]
  thread_x: Annotated[uint32_t, 12]
  thread_y: Annotated[uint32_t, 16]
  thread_z: Annotated[uint32_t, 20]
@c.record
class struct_rgp_sqtt_marker_barrier_start(c.Struct):
  SIZE = 8
  identifier: Annotated[uint32_t, 0, 4, 0]
  ext_dwords: Annotated[uint32_t, 0, 3, 4]
  cb_id: Annotated[uint32_t, 0, 20, 7]
  reserved: Annotated[uint32_t, 3, 5, 3]
  dword01: Annotated[uint32_t, 0]
  driver_reason: Annotated[uint32_t, 4, 31, 0]
  internal: Annotated[uint32_t, 7, 1, 7]
  dword02: Annotated[uint32_t, 4]
@c.record
class struct_rgp_sqtt_marker_barrier_end(c.Struct):
  SIZE = 8
  identifier: Annotated[uint32_t, 0, 4, 0]
  ext_dwords: Annotated[uint32_t, 0, 3, 4]
  cb_id: Annotated[uint32_t, 0, 20, 7]
  wait_on_eop_ts: Annotated[uint32_t, 3, 1, 3]
  vs_partial_flush: Annotated[uint32_t, 3, 1, 4]
  ps_partial_flush: Annotated[uint32_t, 3, 1, 5]
  cs_partial_flush: Annotated[uint32_t, 3, 1, 6]
  pfp_sync_me: Annotated[uint32_t, 3, 1, 7]
  dword01: Annotated[uint32_t, 0]
  sync_cp_dma: Annotated[uint32_t, 4, 1, 0]
  inval_tcp: Annotated[uint32_t, 4, 1, 1]
  inval_sqI: Annotated[uint32_t, 4, 1, 2]
  inval_sqK: Annotated[uint32_t, 4, 1, 3]
  flush_tcc: Annotated[uint32_t, 4, 1, 4]
  inval_tcc: Annotated[uint32_t, 4, 1, 5]
  flush_cb: Annotated[uint32_t, 4, 1, 6]
  inval_cb: Annotated[uint32_t, 4, 1, 7]
  flush_db: Annotated[uint32_t, 5, 1, 0]
  inval_db: Annotated[uint32_t, 5, 1, 1]
  num_layout_transitions: Annotated[uint32_t, 5, 16, 2]
  inval_gl1: Annotated[uint32_t, 7, 1, 2]
  wait_on_ts: Annotated[uint32_t, 7, 1, 3]
  eop_ts_bottom_of_pipe: Annotated[uint32_t, 7, 1, 4]
  eos_ts_ps_done: Annotated[uint32_t, 7, 1, 5]
  eos_ts_cs_done: Annotated[uint32_t, 7, 1, 6]
  reserved: Annotated[uint32_t, 7, 1, 7]
  dword02: Annotated[uint32_t, 4]
@c.record
class struct_rgp_sqtt_marker_layout_transition(c.Struct):
  SIZE = 8
  identifier: Annotated[uint32_t, 0, 4, 0]
  ext_dwords: Annotated[uint32_t, 0, 3, 4]
  depth_stencil_expand: Annotated[uint32_t, 0, 1, 7]
  htile_hiz_range_expand: Annotated[uint32_t, 1, 1, 0]
  depth_stencil_resummarize: Annotated[uint32_t, 1, 1, 1]
  dcc_decompress: Annotated[uint32_t, 1, 1, 2]
  fmask_decompress: Annotated[uint32_t, 1, 1, 3]
  fast_clear_eliminate: Annotated[uint32_t, 1, 1, 4]
  fmask_color_expand: Annotated[uint32_t, 1, 1, 5]
  init_mask_ram: Annotated[uint32_t, 1, 1, 6]
  reserved1: Annotated[uint32_t, 1, 17, 7]
  dword01: Annotated[uint32_t, 0]
  reserved2: Annotated[uint32_t, 4, 32, 0]
  dword02: Annotated[uint32_t, 4]
@c.record
class struct_rgp_sqtt_marker_user_event(c.Struct):
  SIZE = 4
  identifier: Annotated[uint32_t, 0, 4, 0]
  reserved0: Annotated[uint32_t, 0, 8, 4]
  data_type: Annotated[uint32_t, 1, 8, 4]
  reserved1: Annotated[uint32_t, 2, 12, 4]
  dword01: Annotated[uint32_t, 0]
@c.record
class struct_rgp_sqtt_marker_user_event_with_length(c.Struct):
  SIZE = 8
  user_event: Annotated[struct_rgp_sqtt_marker_user_event, 0]
  length: Annotated[uint32_t, 4]
class enum_rgp_sqtt_marker_user_event_type(Annotated[int, ctypes.c_uint32], c.Enum): pass
UserEventTrigger = enum_rgp_sqtt_marker_user_event_type.define('UserEventTrigger', 0)
UserEventPop = enum_rgp_sqtt_marker_user_event_type.define('UserEventPop', 1)
UserEventPush = enum_rgp_sqtt_marker_user_event_type.define('UserEventPush', 2)
UserEventObjectName = enum_rgp_sqtt_marker_user_event_type.define('UserEventObjectName', 3)

@c.record
class struct_rgp_sqtt_marker_pipeline_bind(c.Struct):
  SIZE = 12
  identifier: Annotated[uint32_t, 0, 4, 0]
  ext_dwords: Annotated[uint32_t, 0, 3, 4]
  bind_point: Annotated[uint32_t, 0, 1, 7]
  cb_id: Annotated[uint32_t, 1, 20, 0]
  reserved: Annotated[uint32_t, 3, 4, 4]
  dword01: Annotated[uint32_t, 0]
  api_pso_hash: Annotated[c.Array[uint32_t, Literal[2]], 4]
  dword02: Annotated[uint32_t, 4]
  dword03: Annotated[uint32_t, 8]
c.init_records()
SQTT_FILE_MAGIC_NUMBER = 0x50303042 # type: ignore
SQTT_FILE_VERSION_MAJOR = 1 # type: ignore
SQTT_FILE_VERSION_MINOR = 5 # type: ignore
SQTT_GPU_NAME_MAX_SIZE = 256 # type: ignore
SQTT_MAX_NUM_SE = 32 # type: ignore
SQTT_SA_PER_SE = 2 # type: ignore
SQTT_ACTIVE_PIXEL_PACKER_MASK_DWORDS = 4 # type: ignore