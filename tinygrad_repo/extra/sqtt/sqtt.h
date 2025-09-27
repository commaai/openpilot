#include <stdint.h>

// Original definition in pal is in c++ and clang2py can't autogen it correctly
// Most of this is copy pasted from mesa/src/amd/common/ac_rgp.{h, c}

/*
 * Copyright 2020 Advanced Micro Devices, Inc.
 * Copyright 2020 Valve Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#define SQTT_FILE_MAGIC_NUMBER  0x50303042
#define SQTT_FILE_VERSION_MAJOR 1
#define SQTT_FILE_VERSION_MINOR 5

#define SQTT_GPU_NAME_MAX_SIZE 256
#define SQTT_MAX_NUM_SE        32
#define SQTT_SA_PER_SE         2
#define SQTT_ACTIVE_PIXEL_PACKER_MASK_DWORDS 4

struct sqtt_data_info {
   uint32_t cur_offset;
   uint32_t trace_status;
   union {
      uint32_t gfx9_write_counter;
      uint32_t gfx10_dropped_cntr;
   };
};

struct sqtt_data_se {
   struct sqtt_data_info info;
   void *data_ptr;
   uint32_t shader_engine;
   uint32_t compute_unit;
};


enum sqtt_version
{
   SQTT_VERSION_NONE = 0x0,
   SQTT_VERSION_2_2 = 0x5, /* GFX8 */
   SQTT_VERSION_2_3 = 0x6, /* GFX9 */
   SQTT_VERSION_2_4 = 0x7, /* GFX10+ */
   SQTT_VERSION_3_2 = 0xb, /* GFX11+ */
};

enum sqtt_file_chunk_type
{
   SQTT_FILE_CHUNK_TYPE_ASIC_INFO,
   SQTT_FILE_CHUNK_TYPE_SQTT_DESC,
   SQTT_FILE_CHUNK_TYPE_SQTT_DATA,
   SQTT_FILE_CHUNK_TYPE_API_INFO,
   SQTT_FILE_CHUNK_TYPE_RESERVED,
   SQTT_FILE_CHUNK_TYPE_QUEUE_EVENT_TIMINGS,
   SQTT_FILE_CHUNK_TYPE_CLOCK_CALIBRATION,
   SQTT_FILE_CHUNK_TYPE_CPU_INFO,
   SQTT_FILE_CHUNK_TYPE_SPM_DB,
   SQTT_FILE_CHUNK_TYPE_CODE_OBJECT_DATABASE,
   SQTT_FILE_CHUNK_TYPE_CODE_OBJECT_LOADER_EVENTS,
   SQTT_FILE_CHUNK_TYPE_PSO_CORRELATION,
   SQTT_FILE_CHUNK_TYPE_INSTRUMENTATION_TABLE,
   SQTT_FILE_CHUNK_TYPE_COUNT
};


struct sqtt_file_chunk_id {
   int32_t type : 8;
   int32_t index : 8;
   int32_t reserved : 16;
};

struct sqtt_file_chunk_header {
   struct sqtt_file_chunk_id chunk_id;
   uint16_t minor_version;
   uint16_t major_version;
   int32_t size_in_bytes;
   int32_t padding;
};

struct sqtt_file_header_flags {
   union {
      struct {
         uint32_t is_semaphore_queue_timing_etw : 1;
         uint32_t no_queue_semaphore_timestamps : 1;
         uint32_t reserved : 30;
      };

      uint32_t value;
   };
};

struct sqtt_file_header {
   uint32_t magic_number;
   uint32_t version_major;
   uint32_t version_minor;
   struct sqtt_file_header_flags flags;
   int32_t chunk_offset;
   int32_t second;
   int32_t minute;
   int32_t hour;
   int32_t day_in_month;
   int32_t month;
   int32_t year;
   int32_t day_in_week;
   int32_t day_in_year;
   int32_t is_daylight_savings;
};

struct sqtt_file_chunk_cpu_info {
   struct sqtt_file_chunk_header header;
   uint32_t vendor_id[4];
   uint32_t processor_brand[12];
   uint32_t reserved[2];
   uint64_t cpu_timestamp_freq;
   uint32_t clock_speed;
   uint32_t num_logical_cores;
   uint32_t num_physical_cores;
   uint32_t system_ram_size;
};

enum sqtt_file_chunk_asic_info_flags
{
   SQTT_FILE_CHUNK_ASIC_INFO_FLAG_SC_PACKER_NUMBERING = (1 << 0),
   SQTT_FILE_CHUNK_ASIC_INFO_FLAG_PS1_EVENT_TOKENS_ENABLED = (1 << 1)
};

enum sqtt_gpu_type
{
   SQTT_GPU_TYPE_UNKNOWN = 0x0,
   SQTT_GPU_TYPE_INTEGRATED = 0x1,
   SQTT_GPU_TYPE_DISCRETE = 0x2,
   SQTT_GPU_TYPE_VIRTUAL = 0x3
};

enum sqtt_gfxip_level
{
   SQTT_GFXIP_LEVEL_NONE = 0x0,
   SQTT_GFXIP_LEVEL_GFXIP_6 = 0x1,
   SQTT_GFXIP_LEVEL_GFXIP_7 = 0x2,
   SQTT_GFXIP_LEVEL_GFXIP_8 = 0x3,
   SQTT_GFXIP_LEVEL_GFXIP_8_1 = 0x4,
   SQTT_GFXIP_LEVEL_GFXIP_9 = 0x5,
   SQTT_GFXIP_LEVEL_GFXIP_10_1 = 0x7,
   SQTT_GFXIP_LEVEL_GFXIP_10_3 = 0x9,
   SQTT_GFXIP_LEVEL_GFXIP_11_0 = 0xc,
};

enum sqtt_memory_type
{
   SQTT_MEMORY_TYPE_UNKNOWN = 0x0,
   SQTT_MEMORY_TYPE_DDR = 0x1,
   SQTT_MEMORY_TYPE_DDR2 = 0x2,
   SQTT_MEMORY_TYPE_DDR3 = 0x3,
   SQTT_MEMORY_TYPE_DDR4 = 0x4,
   SQTT_MEMORY_TYPE_DDR5 = 0x5,
   SQTT_MEMORY_TYPE_GDDR3 = 0x10,
   SQTT_MEMORY_TYPE_GDDR4 = 0x11,
   SQTT_MEMORY_TYPE_GDDR5 = 0x12,
   SQTT_MEMORY_TYPE_GDDR6 = 0x13,
   SQTT_MEMORY_TYPE_HBM = 0x20,
   SQTT_MEMORY_TYPE_HBM2 = 0x21,
   SQTT_MEMORY_TYPE_HBM3 = 0x22,
   SQTT_MEMORY_TYPE_LPDDR4 = 0x30,
   SQTT_MEMORY_TYPE_LPDDR5 = 0x31,
};

struct sqtt_file_chunk_asic_info {
   struct sqtt_file_chunk_header header;
   uint64_t flags;
   uint64_t trace_shader_core_clock;
   uint64_t trace_memory_clock;
   int32_t device_id;
   int32_t device_revision_id;
   int32_t vgprs_per_simd;
   int32_t sgprs_per_simd;
   int32_t shader_engines;
   int32_t compute_unit_per_shader_engine;
   int32_t simd_per_compute_unit;
   int32_t wavefronts_per_simd;
   int32_t minimum_vgpr_alloc;
   int32_t vgpr_alloc_granularity;
   int32_t minimum_sgpr_alloc;
   int32_t sgpr_alloc_granularity;
   int32_t hardware_contexts;
   enum sqtt_gpu_type gpu_type;
   enum sqtt_gfxip_level gfxip_level;
   int32_t gpu_index;
   int32_t gds_size;
   int32_t gds_per_shader_engine;
   int32_t ce_ram_size;
   int32_t ce_ram_size_graphics;
   int32_t ce_ram_size_compute;
   int32_t max_number_of_dedicated_cus;
   int64_t vram_size;
   int32_t vram_bus_width;
   int32_t l2_cache_size;
   int32_t l1_cache_size;
   int32_t lds_size;
   char gpu_name[SQTT_GPU_NAME_MAX_SIZE];
   float alu_per_clock;
   float texture_per_clock;
   float prims_per_clock;
   float pixels_per_clock;
   uint64_t gpu_timestamp_frequency;
   uint64_t max_shader_core_clock;
   uint64_t max_memory_clock;
   uint32_t memory_ops_per_clock;
   enum sqtt_memory_type memory_chip_type;
   uint32_t lds_granularity;
   uint16_t cu_mask[SQTT_MAX_NUM_SE][SQTT_SA_PER_SE];
   char reserved1[128];
   uint32_t active_pixel_packer_mask[SQTT_ACTIVE_PIXEL_PACKER_MASK_DWORDS];
   char reserved2[16];
   uint32_t gl1_cache_size;
   uint32_t instruction_cache_size;
   uint32_t scalar_cache_size;
   uint32_t mall_cache_size;
   char padding[4];
};

enum sqtt_api_type
{
   SQTT_API_TYPE_DIRECTX_12,
   SQTT_API_TYPE_VULKAN,
   SQTT_API_TYPE_GENERIC,
   SQTT_API_TYPE_OPENCL
};

enum sqtt_instruction_trace_mode
{
   SQTT_INSTRUCTION_TRACE_DISABLED = 0x0,
   SQTT_INSTRUCTION_TRACE_FULL_FRAME = 0x1,
   SQTT_INSTRUCTION_TRACE_API_PSO = 0x2,
};

enum sqtt_profiling_mode
{
   SQTT_PROFILING_MODE_PRESENT = 0x0,
   SQTT_PROFILING_MODE_USER_MARKERS = 0x1,
   SQTT_PROFILING_MODE_INDEX = 0x2,
   SQTT_PROFILING_MODE_TAG = 0x3,
};

union sqtt_profiling_mode_data {
   struct {
      char start[256];
      char end[256];
   } user_marker_profiling_data;

   struct {
      uint32_t start;
      uint32_t end;
   } index_profiling_data;

   struct {
      uint32_t begin_hi;
      uint32_t begin_lo;
      uint32_t end_hi;
      uint32_t end_lo;
   } tag_profiling_data;
};

union sqtt_instruction_trace_data {
   struct {
      uint64_t api_pso_filter;
   } api_pso_data;

   struct {
      uint32_t mask;
   } shader_engine_filter;
};

struct sqtt_file_chunk_api_info {
   struct sqtt_file_chunk_header header;
   enum sqtt_api_type api_type;
   uint16_t major_version;
   uint16_t minor_version;
   enum sqtt_profiling_mode profiling_mode;
   uint32_t reserved;
   union sqtt_profiling_mode_data profiling_mode_data;
   enum sqtt_instruction_trace_mode instruction_trace_mode;
   uint32_t reserved2;
   union sqtt_instruction_trace_data instruction_trace_data;
};


struct sqtt_code_object_database_record {
   uint32_t size;
};

struct sqtt_file_chunk_code_object_database {
   struct sqtt_file_chunk_header header;
   uint32_t offset;
   uint32_t flags;
   uint32_t size;
   uint32_t record_count;
};


struct sqtt_code_object_loader_events_record {
   uint32_t loader_event_type;
   uint32_t reserved;
   uint64_t base_address;
   uint64_t code_object_hash[2];
   uint64_t time_stamp;
};

struct sqtt_file_chunk_code_object_loader_events {
   struct sqtt_file_chunk_header header;
   uint32_t offset;
   uint32_t flags;
   uint32_t record_size;
   uint32_t record_count;
};

struct sqtt_pso_correlation_record {
   uint64_t api_pso_hash;
   uint64_t pipeline_hash[2];
   char api_level_obj_name[64];
};

struct sqtt_file_chunk_pso_correlation {
   struct sqtt_file_chunk_header header;
   uint32_t offset;
   uint32_t flags;
   uint32_t record_size;
   uint32_t record_count;
};

struct sqtt_file_chunk_sqtt_desc {
   struct sqtt_file_chunk_header header;
   int32_t shader_engine_index;
   enum sqtt_version sqtt_version;
   union {
      struct {
         int32_t instrumentation_version;
      } v0;
      struct {
         int16_t instrumentation_spec_version;
         int16_t instrumentation_api_version;
         int32_t compute_unit_index;
      } v1;
   };
};

struct sqtt_file_chunk_sqtt_data {
   struct sqtt_file_chunk_header header;
   int32_t offset; /* in bytes */
   int32_t size;   /* in bytes */
};

struct sqtt_file_chunk_queue_event_timings {
   struct sqtt_file_chunk_header header;
   uint32_t queue_info_table_record_count;
   uint32_t queue_info_table_size;
   uint32_t queue_event_table_record_count;
   uint32_t queue_event_table_size;
};


enum sqtt_queue_type {
   SQTT_QUEUE_TYPE_UNKNOWN   = 0x0,
   SQTT_QUEUE_TYPE_UNIVERSAL = 0x1,
   SQTT_QUEUE_TYPE_COMPUTE   = 0x2,
   SQTT_QUEUE_TYPE_DMA       = 0x3,
};

enum sqtt_engine_type {
   SQTT_ENGINE_TYPE_UNKNOWN                 = 0x0,
   SQTT_ENGINE_TYPE_UNIVERSAL               = 0x1,
   SQTT_ENGINE_TYPE_COMPUTE                 = 0x2,
   SQTT_ENGINE_TYPE_EXCLUSIVE_COMPUTE       = 0x3,
   SQTT_ENGINE_TYPE_DMA                     = 0x4,
   SQTT_ENGINE_TYPE_HIGH_PRIORITY_UNIVERSAL = 0x7,
   SQTT_ENGINE_TYPE_HIGH_PRIORITY_GRAPHICS  = 0x8,
};

struct sqtt_queue_hardware_info {
   union {
      struct {
         int32_t queue_type : 8;
         int32_t engine_type : 8;
         uint32_t reserved : 16;
      };
      uint32_t value;
   };
};


struct sqtt_queue_info_record {
   uint64_t queue_id;
   uint64_t queue_context;
   struct sqtt_queue_hardware_info hardware_info;
   uint32_t reserved;
};

enum sqtt_queue_event_type {
   SQTT_QUEUE_TIMING_EVENT_CMDBUF_SUBMIT,
   SQTT_QUEUE_TIMING_EVENT_SIGNAL_SEMAPHORE,
   SQTT_QUEUE_TIMING_EVENT_WAIT_SEMAPHORE,
   SQTT_QUEUE_TIMING_EVENT_PRESENT
};

struct sqtt_queue_event_record {
   enum sqtt_queue_event_type event_type;
   uint32_t sqtt_cb_id;
   uint64_t frame_index;
   uint32_t queue_info_index;
   uint32_t submit_sub_index;
   uint64_t api_id;
   uint64_t cpu_timestamp;
   uint64_t gpu_timestamps[2];
};

struct sqtt_file_chunk_clock_calibration {
   struct sqtt_file_chunk_header header;
   uint64_t cpu_timestamp;
   uint64_t gpu_timestamp;
   uint64_t reserved;
};

enum elf_gfxip_level
{
   EF_AMDGPU_MACH_AMDGCN_GFX801 = 0x028,
   EF_AMDGPU_MACH_AMDGCN_GFX900 = 0x02c,
   EF_AMDGPU_MACH_AMDGCN_GFX1010 = 0x033,
   EF_AMDGPU_MACH_AMDGCN_GFX1030 = 0x036,
   EF_AMDGPU_MACH_AMDGCN_GFX1100 = 0x041,
};

struct sqtt_file_chunk_spm_db {
   struct sqtt_file_chunk_header header;
   uint32_t flags;
   uint32_t preamble_size;
   uint32_t num_timestamps;
   uint32_t num_spm_counter_info;
   uint32_t spm_counter_info_size;
   uint32_t sample_interval;
};

/**
 * Identifiers for RGP SQ thread-tracing markers (Table 1)
 */
enum rgp_sqtt_marker_identifier
{
   RGP_SQTT_MARKER_IDENTIFIER_EVENT = 0x0,
   RGP_SQTT_MARKER_IDENTIFIER_CB_START = 0x1,
   RGP_SQTT_MARKER_IDENTIFIER_CB_END = 0x2,
   RGP_SQTT_MARKER_IDENTIFIER_BARRIER_START = 0x3,
   RGP_SQTT_MARKER_IDENTIFIER_BARRIER_END = 0x4,
   RGP_SQTT_MARKER_IDENTIFIER_USER_EVENT = 0x5,
   RGP_SQTT_MARKER_IDENTIFIER_GENERAL_API = 0x6,
   RGP_SQTT_MARKER_IDENTIFIER_SYNC = 0x7,
   RGP_SQTT_MARKER_IDENTIFIER_PRESENT = 0x8,
   RGP_SQTT_MARKER_IDENTIFIER_LAYOUT_TRANSITION = 0x9,
   RGP_SQTT_MARKER_IDENTIFIER_RENDER_PASS = 0xA,
   RGP_SQTT_MARKER_IDENTIFIER_RESERVED2 = 0xB,
   RGP_SQTT_MARKER_IDENTIFIER_BIND_PIPELINE = 0xC,
   RGP_SQTT_MARKER_IDENTIFIER_RESERVED4 = 0xD,
   RGP_SQTT_MARKER_IDENTIFIER_RESERVED5 = 0xE,
   RGP_SQTT_MARKER_IDENTIFIER_RESERVED6 = 0xF
};

/**
 * Command buffer IDs used in RGP SQ thread-tracing markers (only 20 bits).
 */
union rgp_sqtt_marker_cb_id {
   struct {
      uint32_t per_frame : 1; /* Must be 1, frame-based command buffer ID. */
      uint32_t frame_index : 7;
      uint32_t cb_index : 12; /* Command buffer index within the frame. */
      uint32_t reserved : 12;
   } per_frame_cb_id;

   struct {
      uint32_t per_frame : 1; /* Must be 0, global command buffer ID. */
      uint32_t cb_index : 19; /* Global command buffer index. */
      uint32_t reserved : 12;
   } global_cb_id;

   uint32_t all;
};

/**
 * RGP SQ thread-tracing marker for the start of a command buffer. (Table 2)
 */
struct rgp_sqtt_marker_cb_start {
   union {
      struct {
         uint32_t identifier : 4;
         uint32_t ext_dwords : 3;
         uint32_t cb_id : 20;
         uint32_t queue : 5;
      };
      uint32_t dword01;
   };
   union {
      uint32_t device_id_low;
      uint32_t dword02;
   };
   union {
      uint32_t device_id_high;
      uint32_t dword03;
   };
   union {
      uint32_t queue_flags;
      uint32_t dword04;
   };
};

/**
 *
 * RGP SQ thread-tracing marker for the end of a command buffer. (Table 3)
 */
struct rgp_sqtt_marker_cb_end {
   union {
      struct {
         uint32_t identifier : 4;
         uint32_t ext_dwords : 3;
         uint32_t cb_id : 20;
         uint32_t reserved : 5;
      };
      uint32_t dword01;
   };
   union {
      uint32_t device_id_low;
      uint32_t dword02;
   };
   union {
      uint32_t device_id_high;
      uint32_t dword03;
   };
};

/**
 * API types used in RGP SQ thread-tracing markers for the "General API"
 * packet.
 */
enum rgp_sqtt_marker_general_api_type
{
   ApiCmdBindPipeline = 0,
   ApiCmdBindDescriptorSets = 1,
   ApiCmdBindIndexBuffer = 2,
   ApiCmdBindVertexBuffers = 3,
   ApiCmdDraw = 4,
   ApiCmdDrawIndexed = 5,
   ApiCmdDrawIndirect = 6,
   ApiCmdDrawIndexedIndirect = 7,
   ApiCmdDrawIndirectCountAMD = 8,
   ApiCmdDrawIndexedIndirectCountAMD = 9,
   ApiCmdDispatch = 10,
   ApiCmdDispatchIndirect = 11,
   ApiCmdCopyBuffer = 12,
   ApiCmdCopyImage = 13,
   ApiCmdBlitImage = 14,
   ApiCmdCopyBufferToImage = 15,
   ApiCmdCopyImageToBuffer = 16,
   ApiCmdUpdateBuffer = 17,
   ApiCmdFillBuffer = 18,
   ApiCmdClearColorImage = 19,
   ApiCmdClearDepthStencilImage = 20,
   ApiCmdClearAttachments = 21,
   ApiCmdResolveImage = 22,
   ApiCmdWaitEvents = 23,
   ApiCmdPipelineBarrier = 24,
   ApiCmdBeginQuery = 25,
   ApiCmdEndQuery = 26,
   ApiCmdResetQueryPool = 27,
   ApiCmdWriteTimestamp = 28,
   ApiCmdCopyQueryPoolResults = 29,
   ApiCmdPushConstants = 30,
   ApiCmdBeginRenderPass = 31,
   ApiCmdNextSubpass = 32,
   ApiCmdEndRenderPass = 33,
   ApiCmdExecuteCommands = 34,
   ApiCmdSetViewport = 35,
   ApiCmdSetScissor = 36,
   ApiCmdSetLineWidth = 37,
   ApiCmdSetDepthBias = 38,
   ApiCmdSetBlendConstants = 39,
   ApiCmdSetDepthBounds = 40,
   ApiCmdSetStencilCompareMask = 41,
   ApiCmdSetStencilWriteMask = 42,
   ApiCmdSetStencilReference = 43,
   ApiCmdDrawIndirectCount = 44,
   ApiCmdDrawIndexedIndirectCount = 45,
   /* gap */
   ApiCmdDrawMeshTasksEXT = 47,
   ApiCmdDrawMeshTasksIndirectCountEXT = 48,
   ApiCmdDrawMeshTasksIndirectEXT = 49,

   ApiRayTracingSeparateCompiled = 0x800000,
   ApiInvalid = 0xffffffff
};

/**
 * RGP SQ thread-tracing marker for a "General API" instrumentation packet.
 */
struct rgp_sqtt_marker_general_api {
   union {
      struct {
         uint32_t identifier : 4;
         uint32_t ext_dwords : 3;
         uint32_t api_type : 20;
         uint32_t is_end : 1;
         uint32_t reserved : 4;
      };
      uint32_t dword01;
   };
};

/**
 * API types used in RGP SQ thread-tracing markers (Table 16).
 */
enum rgp_sqtt_marker_event_type
{
   EventCmdDraw = 0,
   EventCmdDrawIndexed = 1,
   EventCmdDrawIndirect = 2,
   EventCmdDrawIndexedIndirect = 3,
   EventCmdDrawIndirectCountAMD = 4,
   EventCmdDrawIndexedIndirectCountAMD = 5,
   EventCmdDispatch = 6,
   EventCmdDispatchIndirect = 7,
   EventCmdCopyBuffer = 8,
   EventCmdCopyImage = 9,
   EventCmdBlitImage = 10,
   EventCmdCopyBufferToImage = 11,
   EventCmdCopyImageToBuffer = 12,
   EventCmdUpdateBuffer = 13,
   EventCmdFillBuffer = 14,
   EventCmdClearColorImage = 15,
   EventCmdClearDepthStencilImage = 16,
   EventCmdClearAttachments = 17,
   EventCmdResolveImage = 18,
   EventCmdWaitEvents = 19,
   EventCmdPipelineBarrier = 20,
   EventCmdResetQueryPool = 21,
   EventCmdCopyQueryPoolResults = 22,
   EventRenderPassColorClear = 23,
   EventRenderPassDepthStencilClear = 24,
   EventRenderPassResolve = 25,
   EventInternalUnknown = 26,
   EventCmdDrawIndirectCount = 27,
   EventCmdDrawIndexedIndirectCount = 28,
   /* gap */
   EventCmdTraceRaysKHR = 30,
   EventCmdTraceRaysIndirectKHR = 31,
   EventCmdBuildAccelerationStructuresKHR = 32,
   EventCmdBuildAccelerationStructuresIndirectKHR = 33,
   EventCmdCopyAccelerationStructureKHR = 34,
   EventCmdCopyAccelerationStructureToMemoryKHR = 35,
   EventCmdCopyMemoryToAccelerationStructureKHR = 36,
   /* gap */
   EventCmdDrawMeshTasksEXT = 41,
   EventCmdDrawMeshTasksIndirectCountEXT = 42,
   EventCmdDrawMeshTasksIndirectEXT = 43,
   EventUnknown = 0x7fff,
   EventInvalid = 0xffffffff
};

/**
 * "Event (Per-draw/dispatch)" RGP SQ thread-tracing marker. (Table 4)
 */
struct rgp_sqtt_marker_event {
   union {
      struct {
         uint32_t identifier : 4;
         uint32_t ext_dwords : 3;
         uint32_t api_type : 24;
         uint32_t has_thread_dims : 1;
      };
      uint32_t dword01;
   };
   union {
      struct {
         uint32_t cb_id : 20;
         uint32_t vertex_offset_reg_idx : 4;
         uint32_t instance_offset_reg_idx : 4;
         uint32_t draw_index_reg_idx : 4;
      };
      uint32_t dword02;
   };
   union {
      uint32_t cmd_id;
      uint32_t dword03;
   };
};

/**
 * Per-dispatch specific marker where workgroup dims are included.
 */
struct rgp_sqtt_marker_event_with_dims {
   struct rgp_sqtt_marker_event event;
   uint32_t thread_x;
   uint32_t thread_y;
   uint32_t thread_z;
};

/**
 * "Barrier Start" RGP SQTT instrumentation marker (Table 5)
 */
struct rgp_sqtt_marker_barrier_start {
   union {
      struct {
         uint32_t identifier : 4;
         uint32_t ext_dwords : 3;
         uint32_t cb_id : 20;
         uint32_t reserved : 5;
      };
      uint32_t dword01;
   };
   union {
      struct {
         uint32_t driver_reason : 31;
         uint32_t internal : 1;
      };
      uint32_t dword02;
   };
};

/**
 * "Barrier End" RGP SQTT instrumentation marker (Table 6)
 */
struct rgp_sqtt_marker_barrier_end {
   union {
      struct {
         uint32_t identifier : 4;
         uint32_t ext_dwords : 3;
         uint32_t cb_id : 20;
         uint32_t wait_on_eop_ts : 1;
         uint32_t vs_partial_flush : 1;
         uint32_t ps_partial_flush : 1;
         uint32_t cs_partial_flush : 1;
         uint32_t pfp_sync_me : 1;
      };
      uint32_t dword01;
   };
   union {
      struct {
         uint32_t sync_cp_dma : 1;
         uint32_t inval_tcp : 1;
         uint32_t inval_sqI : 1;
         uint32_t inval_sqK : 1;
         uint32_t flush_tcc : 1;
         uint32_t inval_tcc : 1;
         uint32_t flush_cb : 1;
         uint32_t inval_cb : 1;
         uint32_t flush_db : 1;
         uint32_t inval_db : 1;
         uint32_t num_layout_transitions : 16;
         uint32_t inval_gl1 : 1;
         uint32_t wait_on_ts : 1;
         uint32_t eop_ts_bottom_of_pipe : 1;
         uint32_t eos_ts_ps_done : 1;
         uint32_t eos_ts_cs_done : 1;
         uint32_t reserved : 1;
      };
      uint32_t dword02;
   };
};

/**
 * "Layout Transition" RGP SQTT instrumentation marker (Table 7)
 */
struct rgp_sqtt_marker_layout_transition {
   union {
      struct {
         uint32_t identifier : 4;
         uint32_t ext_dwords : 3;
         uint32_t depth_stencil_expand : 1;
         uint32_t htile_hiz_range_expand : 1;
         uint32_t depth_stencil_resummarize : 1;
         uint32_t dcc_decompress : 1;
         uint32_t fmask_decompress : 1;
         uint32_t fast_clear_eliminate : 1;
         uint32_t fmask_color_expand : 1;
         uint32_t init_mask_ram : 1;
         uint32_t reserved1 : 17;
      };
      uint32_t dword01;
   };
   union {
      struct {
         uint32_t reserved2 : 32;
      };
      uint32_t dword02;
   };
};

/**
 * "User Event" RGP SQTT instrumentation marker (Table 8)
 */
struct rgp_sqtt_marker_user_event {
   union {
      struct {
         uint32_t identifier : 4;
         uint32_t reserved0 : 8;
         uint32_t data_type : 8;
         uint32_t reserved1 : 12;
      };
      uint32_t dword01;
   };
};
struct rgp_sqtt_marker_user_event_with_length {
   struct rgp_sqtt_marker_user_event user_event;
   uint32_t length;
};

enum rgp_sqtt_marker_user_event_type
{
   UserEventTrigger = 0,
   UserEventPop,
   UserEventPush,
   UserEventObjectName,
};

/**
 * "Pipeline bind" RGP SQTT instrumentation marker (Table 12)
 */
struct rgp_sqtt_marker_pipeline_bind {
   union {
      struct {
         uint32_t identifier : 4;
         uint32_t ext_dwords : 3;
         uint32_t bind_point : 1;
         uint32_t cb_id : 20;
         uint32_t reserved : 4;
      };
      uint32_t dword01;
   };
   union {
      uint32_t api_pso_hash[2];
      struct {
         uint32_t dword02;
         uint32_t dword03;
      };
   };
};
