# mypy: disable-error-code="empty-body"
from __future__ import annotations
import ctypes
from typing import Literal, TypeAlias
from tinygrad.runtime.support.c import _IO, _IOW, _IOR, _IOWR
from tinygrad.runtime.support import c
@c.record
class struct_kfd_ioctl_get_version_args(c.Struct):
  SIZE = 8
  major_version: int
  minor_version: int
__u32: TypeAlias = ctypes.c_uint32
struct_kfd_ioctl_get_version_args.register_fields([('major_version', ctypes.c_uint32, 0), ('minor_version', ctypes.c_uint32, 4)])
@c.record
class struct_kfd_ioctl_create_queue_args(c.Struct):
  SIZE = 96
  ring_base_address: int
  write_pointer_address: int
  read_pointer_address: int
  doorbell_offset: int
  ring_size: int
  gpu_id: int
  queue_type: int
  queue_percentage: int
  queue_priority: int
  queue_id: int
  eop_buffer_address: int
  eop_buffer_size: int
  ctx_save_restore_address: int
  ctx_save_restore_size: int
  ctl_stack_size: int
  sdma_engine_id: int
  pad: int
__u64: TypeAlias = ctypes.c_uint64
struct_kfd_ioctl_create_queue_args.register_fields([('ring_base_address', ctypes.c_uint64, 0), ('write_pointer_address', ctypes.c_uint64, 8), ('read_pointer_address', ctypes.c_uint64, 16), ('doorbell_offset', ctypes.c_uint64, 24), ('ring_size', ctypes.c_uint32, 32), ('gpu_id', ctypes.c_uint32, 36), ('queue_type', ctypes.c_uint32, 40), ('queue_percentage', ctypes.c_uint32, 44), ('queue_priority', ctypes.c_uint32, 48), ('queue_id', ctypes.c_uint32, 52), ('eop_buffer_address', ctypes.c_uint64, 56), ('eop_buffer_size', ctypes.c_uint64, 64), ('ctx_save_restore_address', ctypes.c_uint64, 72), ('ctx_save_restore_size', ctypes.c_uint32, 80), ('ctl_stack_size', ctypes.c_uint32, 84), ('sdma_engine_id', ctypes.c_uint32, 88), ('pad', ctypes.c_uint32, 92)])
@c.record
class struct_kfd_ioctl_destroy_queue_args(c.Struct):
  SIZE = 8
  queue_id: int
  pad: int
struct_kfd_ioctl_destroy_queue_args.register_fields([('queue_id', ctypes.c_uint32, 0), ('pad', ctypes.c_uint32, 4)])
@c.record
class struct_kfd_ioctl_update_queue_args(c.Struct):
  SIZE = 24
  ring_base_address: int
  queue_id: int
  ring_size: int
  queue_percentage: int
  queue_priority: int
struct_kfd_ioctl_update_queue_args.register_fields([('ring_base_address', ctypes.c_uint64, 0), ('queue_id', ctypes.c_uint32, 8), ('ring_size', ctypes.c_uint32, 12), ('queue_percentage', ctypes.c_uint32, 16), ('queue_priority', ctypes.c_uint32, 20)])
@c.record
class struct_kfd_ioctl_set_cu_mask_args(c.Struct):
  SIZE = 16
  queue_id: int
  num_cu_mask: int
  cu_mask_ptr: int
struct_kfd_ioctl_set_cu_mask_args.register_fields([('queue_id', ctypes.c_uint32, 0), ('num_cu_mask', ctypes.c_uint32, 4), ('cu_mask_ptr', ctypes.c_uint64, 8)])
@c.record
class struct_kfd_ioctl_get_queue_wave_state_args(c.Struct):
  SIZE = 24
  ctl_stack_address: int
  ctl_stack_used_size: int
  save_area_used_size: int
  queue_id: int
  pad: int
struct_kfd_ioctl_get_queue_wave_state_args.register_fields([('ctl_stack_address', ctypes.c_uint64, 0), ('ctl_stack_used_size', ctypes.c_uint32, 8), ('save_area_used_size', ctypes.c_uint32, 12), ('queue_id', ctypes.c_uint32, 16), ('pad', ctypes.c_uint32, 20)])
@c.record
class struct_kfd_ioctl_get_available_memory_args(c.Struct):
  SIZE = 16
  available: int
  gpu_id: int
  pad: int
struct_kfd_ioctl_get_available_memory_args.register_fields([('available', ctypes.c_uint64, 0), ('gpu_id', ctypes.c_uint32, 8), ('pad', ctypes.c_uint32, 12)])
@c.record
class struct_kfd_dbg_device_info_entry(c.Struct):
  SIZE = 120
  exception_status: int
  lds_base: int
  lds_limit: int
  scratch_base: int
  scratch_limit: int
  gpuvm_base: int
  gpuvm_limit: int
  gpu_id: int
  location_id: int
  vendor_id: int
  device_id: int
  revision_id: int
  subsystem_vendor_id: int
  subsystem_device_id: int
  fw_version: int
  gfx_target_version: int
  simd_count: int
  max_waves_per_simd: int
  array_count: int
  simd_arrays_per_engine: int
  num_xcc: int
  capability: int
  debug_prop: int
struct_kfd_dbg_device_info_entry.register_fields([('exception_status', ctypes.c_uint64, 0), ('lds_base', ctypes.c_uint64, 8), ('lds_limit', ctypes.c_uint64, 16), ('scratch_base', ctypes.c_uint64, 24), ('scratch_limit', ctypes.c_uint64, 32), ('gpuvm_base', ctypes.c_uint64, 40), ('gpuvm_limit', ctypes.c_uint64, 48), ('gpu_id', ctypes.c_uint32, 56), ('location_id', ctypes.c_uint32, 60), ('vendor_id', ctypes.c_uint32, 64), ('device_id', ctypes.c_uint32, 68), ('revision_id', ctypes.c_uint32, 72), ('subsystem_vendor_id', ctypes.c_uint32, 76), ('subsystem_device_id', ctypes.c_uint32, 80), ('fw_version', ctypes.c_uint32, 84), ('gfx_target_version', ctypes.c_uint32, 88), ('simd_count', ctypes.c_uint32, 92), ('max_waves_per_simd', ctypes.c_uint32, 96), ('array_count', ctypes.c_uint32, 100), ('simd_arrays_per_engine', ctypes.c_uint32, 104), ('num_xcc', ctypes.c_uint32, 108), ('capability', ctypes.c_uint32, 112), ('debug_prop', ctypes.c_uint32, 116)])
@c.record
class struct_kfd_ioctl_set_memory_policy_args(c.Struct):
  SIZE = 32
  alternate_aperture_base: int
  alternate_aperture_size: int
  gpu_id: int
  default_policy: int
  alternate_policy: int
  pad: int
struct_kfd_ioctl_set_memory_policy_args.register_fields([('alternate_aperture_base', ctypes.c_uint64, 0), ('alternate_aperture_size', ctypes.c_uint64, 8), ('gpu_id', ctypes.c_uint32, 16), ('default_policy', ctypes.c_uint32, 20), ('alternate_policy', ctypes.c_uint32, 24), ('pad', ctypes.c_uint32, 28)])
@c.record
class struct_kfd_ioctl_get_clock_counters_args(c.Struct):
  SIZE = 40
  gpu_clock_counter: int
  cpu_clock_counter: int
  system_clock_counter: int
  system_clock_freq: int
  gpu_id: int
  pad: int
struct_kfd_ioctl_get_clock_counters_args.register_fields([('gpu_clock_counter', ctypes.c_uint64, 0), ('cpu_clock_counter', ctypes.c_uint64, 8), ('system_clock_counter', ctypes.c_uint64, 16), ('system_clock_freq', ctypes.c_uint64, 24), ('gpu_id', ctypes.c_uint32, 32), ('pad', ctypes.c_uint32, 36)])
@c.record
class struct_kfd_process_device_apertures(c.Struct):
  SIZE = 56
  lds_base: int
  lds_limit: int
  scratch_base: int
  scratch_limit: int
  gpuvm_base: int
  gpuvm_limit: int
  gpu_id: int
  pad: int
struct_kfd_process_device_apertures.register_fields([('lds_base', ctypes.c_uint64, 0), ('lds_limit', ctypes.c_uint64, 8), ('scratch_base', ctypes.c_uint64, 16), ('scratch_limit', ctypes.c_uint64, 24), ('gpuvm_base', ctypes.c_uint64, 32), ('gpuvm_limit', ctypes.c_uint64, 40), ('gpu_id', ctypes.c_uint32, 48), ('pad', ctypes.c_uint32, 52)])
@c.record
class struct_kfd_ioctl_get_process_apertures_args(c.Struct):
  SIZE = 400
  process_apertures: c.Array[struct_kfd_process_device_apertures, Literal[7]]
  num_of_nodes: int
  pad: int
struct_kfd_ioctl_get_process_apertures_args.register_fields([('process_apertures', c.Array[struct_kfd_process_device_apertures, Literal[7]], 0), ('num_of_nodes', ctypes.c_uint32, 392), ('pad', ctypes.c_uint32, 396)])
@c.record
class struct_kfd_ioctl_get_process_apertures_new_args(c.Struct):
  SIZE = 16
  kfd_process_device_apertures_ptr: int
  num_of_nodes: int
  pad: int
struct_kfd_ioctl_get_process_apertures_new_args.register_fields([('kfd_process_device_apertures_ptr', ctypes.c_uint64, 0), ('num_of_nodes', ctypes.c_uint32, 8), ('pad', ctypes.c_uint32, 12)])
@c.record
class struct_kfd_ioctl_dbg_register_args(c.Struct):
  SIZE = 8
  gpu_id: int
  pad: int
struct_kfd_ioctl_dbg_register_args.register_fields([('gpu_id', ctypes.c_uint32, 0), ('pad', ctypes.c_uint32, 4)])
@c.record
class struct_kfd_ioctl_dbg_unregister_args(c.Struct):
  SIZE = 8
  gpu_id: int
  pad: int
struct_kfd_ioctl_dbg_unregister_args.register_fields([('gpu_id', ctypes.c_uint32, 0), ('pad', ctypes.c_uint32, 4)])
@c.record
class struct_kfd_ioctl_dbg_address_watch_args(c.Struct):
  SIZE = 16
  content_ptr: int
  gpu_id: int
  buf_size_in_bytes: int
struct_kfd_ioctl_dbg_address_watch_args.register_fields([('content_ptr', ctypes.c_uint64, 0), ('gpu_id', ctypes.c_uint32, 8), ('buf_size_in_bytes', ctypes.c_uint32, 12)])
@c.record
class struct_kfd_ioctl_dbg_wave_control_args(c.Struct):
  SIZE = 16
  content_ptr: int
  gpu_id: int
  buf_size_in_bytes: int
struct_kfd_ioctl_dbg_wave_control_args.register_fields([('content_ptr', ctypes.c_uint64, 0), ('gpu_id', ctypes.c_uint32, 8), ('buf_size_in_bytes', ctypes.c_uint32, 12)])
@c.record
class struct_kfd_ioctl_dbg_trap_args_deprecated(c.Struct):
  SIZE = 40
  exception_mask: int
  ptr: int
  pid: int
  op: int
  data1: int
  data2: int
  data3: int
  data4: int
struct_kfd_ioctl_dbg_trap_args_deprecated.register_fields([('exception_mask', ctypes.c_uint64, 0), ('ptr', ctypes.c_uint64, 8), ('pid', ctypes.c_uint32, 16), ('op', ctypes.c_uint32, 20), ('data1', ctypes.c_uint32, 24), ('data2', ctypes.c_uint32, 28), ('data3', ctypes.c_uint32, 32), ('data4', ctypes.c_uint32, 36)])
@c.record
class struct_kfd_ioctl_create_event_args(c.Struct):
  SIZE = 32
  event_page_offset: int
  event_trigger_data: int
  event_type: int
  auto_reset: int
  node_id: int
  event_id: int
  event_slot_index: int
struct_kfd_ioctl_create_event_args.register_fields([('event_page_offset', ctypes.c_uint64, 0), ('event_trigger_data', ctypes.c_uint32, 8), ('event_type', ctypes.c_uint32, 12), ('auto_reset', ctypes.c_uint32, 16), ('node_id', ctypes.c_uint32, 20), ('event_id', ctypes.c_uint32, 24), ('event_slot_index', ctypes.c_uint32, 28)])
@c.record
class struct_kfd_ioctl_destroy_event_args(c.Struct):
  SIZE = 8
  event_id: int
  pad: int
struct_kfd_ioctl_destroy_event_args.register_fields([('event_id', ctypes.c_uint32, 0), ('pad', ctypes.c_uint32, 4)])
@c.record
class struct_kfd_ioctl_set_event_args(c.Struct):
  SIZE = 8
  event_id: int
  pad: int
struct_kfd_ioctl_set_event_args.register_fields([('event_id', ctypes.c_uint32, 0), ('pad', ctypes.c_uint32, 4)])
@c.record
class struct_kfd_ioctl_reset_event_args(c.Struct):
  SIZE = 8
  event_id: int
  pad: int
struct_kfd_ioctl_reset_event_args.register_fields([('event_id', ctypes.c_uint32, 0), ('pad', ctypes.c_uint32, 4)])
@c.record
class struct_kfd_memory_exception_failure(c.Struct):
  SIZE = 16
  NotPresent: int
  ReadOnly: int
  NoExecute: int
  imprecise: int
struct_kfd_memory_exception_failure.register_fields([('NotPresent', ctypes.c_uint32, 0), ('ReadOnly', ctypes.c_uint32, 4), ('NoExecute', ctypes.c_uint32, 8), ('imprecise', ctypes.c_uint32, 12)])
@c.record
class struct_kfd_hsa_memory_exception_data(c.Struct):
  SIZE = 32
  failure: struct_kfd_memory_exception_failure
  va: int
  gpu_id: int
  ErrorType: int
struct_kfd_hsa_memory_exception_data.register_fields([('failure', struct_kfd_memory_exception_failure, 0), ('va', ctypes.c_uint64, 16), ('gpu_id', ctypes.c_uint32, 24), ('ErrorType', ctypes.c_uint32, 28)])
@c.record
class struct_kfd_hsa_hw_exception_data(c.Struct):
  SIZE = 16
  reset_type: int
  reset_cause: int
  memory_lost: int
  gpu_id: int
struct_kfd_hsa_hw_exception_data.register_fields([('reset_type', ctypes.c_uint32, 0), ('reset_cause', ctypes.c_uint32, 4), ('memory_lost', ctypes.c_uint32, 8), ('gpu_id', ctypes.c_uint32, 12)])
@c.record
class struct_kfd_hsa_signal_event_data(c.Struct):
  SIZE = 8
  last_event_age: int
struct_kfd_hsa_signal_event_data.register_fields([('last_event_age', ctypes.c_uint64, 0)])
@c.record
class struct_kfd_event_data(c.Struct):
  SIZE = 48
  memory_exception_data: struct_kfd_hsa_memory_exception_data
  hw_exception_data: struct_kfd_hsa_hw_exception_data
  signal_event_data: struct_kfd_hsa_signal_event_data
  kfd_event_data_ext: int
  event_id: int
  pad: int
struct_kfd_event_data.register_fields([('memory_exception_data', struct_kfd_hsa_memory_exception_data, 0), ('hw_exception_data', struct_kfd_hsa_hw_exception_data, 0), ('signal_event_data', struct_kfd_hsa_signal_event_data, 0), ('kfd_event_data_ext', ctypes.c_uint64, 32), ('event_id', ctypes.c_uint32, 40), ('pad', ctypes.c_uint32, 44)])
@c.record
class struct_kfd_ioctl_wait_events_args(c.Struct):
  SIZE = 24
  events_ptr: int
  num_events: int
  wait_for_all: int
  timeout: int
  wait_result: int
struct_kfd_ioctl_wait_events_args.register_fields([('events_ptr', ctypes.c_uint64, 0), ('num_events', ctypes.c_uint32, 8), ('wait_for_all', ctypes.c_uint32, 12), ('timeout', ctypes.c_uint32, 16), ('wait_result', ctypes.c_uint32, 20)])
@c.record
class struct_kfd_ioctl_set_scratch_backing_va_args(c.Struct):
  SIZE = 16
  va_addr: int
  gpu_id: int
  pad: int
struct_kfd_ioctl_set_scratch_backing_va_args.register_fields([('va_addr', ctypes.c_uint64, 0), ('gpu_id', ctypes.c_uint32, 8), ('pad', ctypes.c_uint32, 12)])
@c.record
class struct_kfd_ioctl_get_tile_config_args(c.Struct):
  SIZE = 40
  tile_config_ptr: int
  macro_tile_config_ptr: int
  num_tile_configs: int
  num_macro_tile_configs: int
  gpu_id: int
  gb_addr_config: int
  num_banks: int
  num_ranks: int
struct_kfd_ioctl_get_tile_config_args.register_fields([('tile_config_ptr', ctypes.c_uint64, 0), ('macro_tile_config_ptr', ctypes.c_uint64, 8), ('num_tile_configs', ctypes.c_uint32, 16), ('num_macro_tile_configs', ctypes.c_uint32, 20), ('gpu_id', ctypes.c_uint32, 24), ('gb_addr_config', ctypes.c_uint32, 28), ('num_banks', ctypes.c_uint32, 32), ('num_ranks', ctypes.c_uint32, 36)])
@c.record
class struct_kfd_ioctl_set_trap_handler_args(c.Struct):
  SIZE = 24
  tba_addr: int
  tma_addr: int
  gpu_id: int
  pad: int
struct_kfd_ioctl_set_trap_handler_args.register_fields([('tba_addr', ctypes.c_uint64, 0), ('tma_addr', ctypes.c_uint64, 8), ('gpu_id', ctypes.c_uint32, 16), ('pad', ctypes.c_uint32, 20)])
@c.record
class struct_kfd_ioctl_acquire_vm_args(c.Struct):
  SIZE = 8
  drm_fd: int
  gpu_id: int
struct_kfd_ioctl_acquire_vm_args.register_fields([('drm_fd', ctypes.c_uint32, 0), ('gpu_id', ctypes.c_uint32, 4)])
@c.record
class struct_kfd_ioctl_alloc_memory_of_gpu_args(c.Struct):
  SIZE = 40
  va_addr: int
  size: int
  handle: int
  mmap_offset: int
  gpu_id: int
  flags: int
struct_kfd_ioctl_alloc_memory_of_gpu_args.register_fields([('va_addr', ctypes.c_uint64, 0), ('size', ctypes.c_uint64, 8), ('handle', ctypes.c_uint64, 16), ('mmap_offset', ctypes.c_uint64, 24), ('gpu_id', ctypes.c_uint32, 32), ('flags', ctypes.c_uint32, 36)])
@c.record
class struct_kfd_ioctl_free_memory_of_gpu_args(c.Struct):
  SIZE = 8
  handle: int
struct_kfd_ioctl_free_memory_of_gpu_args.register_fields([('handle', ctypes.c_uint64, 0)])
@c.record
class struct_kfd_ioctl_map_memory_to_gpu_args(c.Struct):
  SIZE = 24
  handle: int
  device_ids_array_ptr: int
  n_devices: int
  n_success: int
struct_kfd_ioctl_map_memory_to_gpu_args.register_fields([('handle', ctypes.c_uint64, 0), ('device_ids_array_ptr', ctypes.c_uint64, 8), ('n_devices', ctypes.c_uint32, 16), ('n_success', ctypes.c_uint32, 20)])
@c.record
class struct_kfd_ioctl_unmap_memory_from_gpu_args(c.Struct):
  SIZE = 24
  handle: int
  device_ids_array_ptr: int
  n_devices: int
  n_success: int
struct_kfd_ioctl_unmap_memory_from_gpu_args.register_fields([('handle', ctypes.c_uint64, 0), ('device_ids_array_ptr', ctypes.c_uint64, 8), ('n_devices', ctypes.c_uint32, 16), ('n_success', ctypes.c_uint32, 20)])
@c.record
class struct_kfd_ioctl_alloc_queue_gws_args(c.Struct):
  SIZE = 16
  queue_id: int
  num_gws: int
  first_gws: int
  pad: int
struct_kfd_ioctl_alloc_queue_gws_args.register_fields([('queue_id', ctypes.c_uint32, 0), ('num_gws', ctypes.c_uint32, 4), ('first_gws', ctypes.c_uint32, 8), ('pad', ctypes.c_uint32, 12)])
@c.record
class struct_kfd_ioctl_get_dmabuf_info_args(c.Struct):
  SIZE = 32
  size: int
  metadata_ptr: int
  metadata_size: int
  gpu_id: int
  flags: int
  dmabuf_fd: int
struct_kfd_ioctl_get_dmabuf_info_args.register_fields([('size', ctypes.c_uint64, 0), ('metadata_ptr', ctypes.c_uint64, 8), ('metadata_size', ctypes.c_uint32, 16), ('gpu_id', ctypes.c_uint32, 20), ('flags', ctypes.c_uint32, 24), ('dmabuf_fd', ctypes.c_uint32, 28)])
@c.record
class struct_kfd_ioctl_import_dmabuf_args(c.Struct):
  SIZE = 24
  va_addr: int
  handle: int
  gpu_id: int
  dmabuf_fd: int
struct_kfd_ioctl_import_dmabuf_args.register_fields([('va_addr', ctypes.c_uint64, 0), ('handle', ctypes.c_uint64, 8), ('gpu_id', ctypes.c_uint32, 16), ('dmabuf_fd', ctypes.c_uint32, 20)])
@c.record
class struct_kfd_ioctl_export_dmabuf_args(c.Struct):
  SIZE = 16
  handle: int
  flags: int
  dmabuf_fd: int
struct_kfd_ioctl_export_dmabuf_args.register_fields([('handle', ctypes.c_uint64, 0), ('flags', ctypes.c_uint32, 8), ('dmabuf_fd', ctypes.c_uint32, 12)])
enum_kfd_smi_event: dict[int, str] = {(KFD_SMI_EVENT_NONE:=0): 'KFD_SMI_EVENT_NONE', (KFD_SMI_EVENT_VMFAULT:=1): 'KFD_SMI_EVENT_VMFAULT', (KFD_SMI_EVENT_THERMAL_THROTTLE:=2): 'KFD_SMI_EVENT_THERMAL_THROTTLE', (KFD_SMI_EVENT_GPU_PRE_RESET:=3): 'KFD_SMI_EVENT_GPU_PRE_RESET', (KFD_SMI_EVENT_GPU_POST_RESET:=4): 'KFD_SMI_EVENT_GPU_POST_RESET', (KFD_SMI_EVENT_MIGRATE_START:=5): 'KFD_SMI_EVENT_MIGRATE_START', (KFD_SMI_EVENT_MIGRATE_END:=6): 'KFD_SMI_EVENT_MIGRATE_END', (KFD_SMI_EVENT_PAGE_FAULT_START:=7): 'KFD_SMI_EVENT_PAGE_FAULT_START', (KFD_SMI_EVENT_PAGE_FAULT_END:=8): 'KFD_SMI_EVENT_PAGE_FAULT_END', (KFD_SMI_EVENT_QUEUE_EVICTION:=9): 'KFD_SMI_EVENT_QUEUE_EVICTION', (KFD_SMI_EVENT_QUEUE_RESTORE:=10): 'KFD_SMI_EVENT_QUEUE_RESTORE', (KFD_SMI_EVENT_UNMAP_FROM_GPU:=11): 'KFD_SMI_EVENT_UNMAP_FROM_GPU', (KFD_SMI_EVENT_ALL_PROCESS:=64): 'KFD_SMI_EVENT_ALL_PROCESS'}
enum_KFD_MIGRATE_TRIGGERS: dict[int, str] = {(KFD_MIGRATE_TRIGGER_PREFETCH:=0): 'KFD_MIGRATE_TRIGGER_PREFETCH', (KFD_MIGRATE_TRIGGER_PAGEFAULT_GPU:=1): 'KFD_MIGRATE_TRIGGER_PAGEFAULT_GPU', (KFD_MIGRATE_TRIGGER_PAGEFAULT_CPU:=2): 'KFD_MIGRATE_TRIGGER_PAGEFAULT_CPU', (KFD_MIGRATE_TRIGGER_TTM_EVICTION:=3): 'KFD_MIGRATE_TRIGGER_TTM_EVICTION'}
enum_KFD_QUEUE_EVICTION_TRIGGERS: dict[int, str] = {(KFD_QUEUE_EVICTION_TRIGGER_SVM:=0): 'KFD_QUEUE_EVICTION_TRIGGER_SVM', (KFD_QUEUE_EVICTION_TRIGGER_USERPTR:=1): 'KFD_QUEUE_EVICTION_TRIGGER_USERPTR', (KFD_QUEUE_EVICTION_TRIGGER_TTM:=2): 'KFD_QUEUE_EVICTION_TRIGGER_TTM', (KFD_QUEUE_EVICTION_TRIGGER_SUSPEND:=3): 'KFD_QUEUE_EVICTION_TRIGGER_SUSPEND', (KFD_QUEUE_EVICTION_CRIU_CHECKPOINT:=4): 'KFD_QUEUE_EVICTION_CRIU_CHECKPOINT', (KFD_QUEUE_EVICTION_CRIU_RESTORE:=5): 'KFD_QUEUE_EVICTION_CRIU_RESTORE'}
enum_KFD_SVM_UNMAP_TRIGGERS: dict[int, str] = {(KFD_SVM_UNMAP_TRIGGER_MMU_NOTIFY:=0): 'KFD_SVM_UNMAP_TRIGGER_MMU_NOTIFY', (KFD_SVM_UNMAP_TRIGGER_MMU_NOTIFY_MIGRATE:=1): 'KFD_SVM_UNMAP_TRIGGER_MMU_NOTIFY_MIGRATE', (KFD_SVM_UNMAP_TRIGGER_UNMAP_FROM_CPU:=2): 'KFD_SVM_UNMAP_TRIGGER_UNMAP_FROM_CPU'}
@c.record
class struct_kfd_ioctl_smi_events_args(c.Struct):
  SIZE = 8
  gpuid: int
  anon_fd: int
struct_kfd_ioctl_smi_events_args.register_fields([('gpuid', ctypes.c_uint32, 0), ('anon_fd', ctypes.c_uint32, 4)])
enum_kfd_ioctl_spm_op: dict[int, str] = {(KFD_IOCTL_SPM_OP_ACQUIRE:=0): 'KFD_IOCTL_SPM_OP_ACQUIRE', (KFD_IOCTL_SPM_OP_RELEASE:=1): 'KFD_IOCTL_SPM_OP_RELEASE', (KFD_IOCTL_SPM_OP_SET_DEST_BUF:=2): 'KFD_IOCTL_SPM_OP_SET_DEST_BUF'}
@c.record
class struct_kfd_ioctl_spm_args(c.Struct):
  SIZE = 32
  dest_buf: int
  buf_size: int
  op: int
  timeout: int
  gpu_id: int
  bytes_copied: int
  has_data_loss: int
struct_kfd_ioctl_spm_args.register_fields([('dest_buf', ctypes.c_uint64, 0), ('buf_size', ctypes.c_uint32, 8), ('op', ctypes.c_uint32, 12), ('timeout', ctypes.c_uint32, 16), ('gpu_id', ctypes.c_uint32, 20), ('bytes_copied', ctypes.c_uint32, 24), ('has_data_loss', ctypes.c_uint32, 28)])
enum_kfd_criu_op: dict[int, str] = {(KFD_CRIU_OP_PROCESS_INFO:=0): 'KFD_CRIU_OP_PROCESS_INFO', (KFD_CRIU_OP_CHECKPOINT:=1): 'KFD_CRIU_OP_CHECKPOINT', (KFD_CRIU_OP_UNPAUSE:=2): 'KFD_CRIU_OP_UNPAUSE', (KFD_CRIU_OP_RESTORE:=3): 'KFD_CRIU_OP_RESTORE', (KFD_CRIU_OP_RESUME:=4): 'KFD_CRIU_OP_RESUME'}
@c.record
class struct_kfd_ioctl_criu_args(c.Struct):
  SIZE = 56
  devices: int
  bos: int
  priv_data: int
  priv_data_size: int
  num_devices: int
  num_bos: int
  num_objects: int
  pid: int
  op: int
struct_kfd_ioctl_criu_args.register_fields([('devices', ctypes.c_uint64, 0), ('bos', ctypes.c_uint64, 8), ('priv_data', ctypes.c_uint64, 16), ('priv_data_size', ctypes.c_uint64, 24), ('num_devices', ctypes.c_uint32, 32), ('num_bos', ctypes.c_uint32, 36), ('num_objects', ctypes.c_uint32, 40), ('pid', ctypes.c_uint32, 44), ('op', ctypes.c_uint32, 48)])
@c.record
class struct_kfd_criu_device_bucket(c.Struct):
  SIZE = 16
  user_gpu_id: int
  actual_gpu_id: int
  drm_fd: int
  pad: int
struct_kfd_criu_device_bucket.register_fields([('user_gpu_id', ctypes.c_uint32, 0), ('actual_gpu_id', ctypes.c_uint32, 4), ('drm_fd', ctypes.c_uint32, 8), ('pad', ctypes.c_uint32, 12)])
@c.record
class struct_kfd_criu_bo_bucket(c.Struct):
  SIZE = 48
  addr: int
  size: int
  offset: int
  restored_offset: int
  gpu_id: int
  alloc_flags: int
  dmabuf_fd: int
  pad: int
struct_kfd_criu_bo_bucket.register_fields([('addr', ctypes.c_uint64, 0), ('size', ctypes.c_uint64, 8), ('offset', ctypes.c_uint64, 16), ('restored_offset', ctypes.c_uint64, 24), ('gpu_id', ctypes.c_uint32, 32), ('alloc_flags', ctypes.c_uint32, 36), ('dmabuf_fd', ctypes.c_uint32, 40), ('pad', ctypes.c_uint32, 44)])
enum_kfd_mmio_remap: dict[int, str] = {(KFD_MMIO_REMAP_HDP_MEM_FLUSH_CNTL:=0): 'KFD_MMIO_REMAP_HDP_MEM_FLUSH_CNTL', (KFD_MMIO_REMAP_HDP_REG_FLUSH_CNTL:=4): 'KFD_MMIO_REMAP_HDP_REG_FLUSH_CNTL'}
@c.record
class struct_kfd_ioctl_ipc_export_handle_args(c.Struct):
  SIZE = 32
  handle: int
  share_handle: c.Array[ctypes.c_uint32, Literal[4]]
  gpu_id: int
  flags: int
struct_kfd_ioctl_ipc_export_handle_args.register_fields([('handle', ctypes.c_uint64, 0), ('share_handle', c.Array[ctypes.c_uint32, Literal[4]], 8), ('gpu_id', ctypes.c_uint32, 24), ('flags', ctypes.c_uint32, 28)])
@c.record
class struct_kfd_ioctl_ipc_import_handle_args(c.Struct):
  SIZE = 48
  handle: int
  va_addr: int
  mmap_offset: int
  share_handle: c.Array[ctypes.c_uint32, Literal[4]]
  gpu_id: int
  flags: int
struct_kfd_ioctl_ipc_import_handle_args.register_fields([('handle', ctypes.c_uint64, 0), ('va_addr', ctypes.c_uint64, 8), ('mmap_offset', ctypes.c_uint64, 16), ('share_handle', c.Array[ctypes.c_uint32, Literal[4]], 24), ('gpu_id', ctypes.c_uint32, 40), ('flags', ctypes.c_uint32, 44)])
@c.record
class struct_kfd_ioctl_cross_memory_copy_deprecated_args(c.Struct):
  SIZE = 48
  pid: int
  flags: int
  src_mem_range_array: int
  src_mem_array_size: int
  dst_mem_range_array: int
  dst_mem_array_size: int
  bytes_copied: int
struct_kfd_ioctl_cross_memory_copy_deprecated_args.register_fields([('pid', ctypes.c_uint32, 0), ('flags', ctypes.c_uint32, 4), ('src_mem_range_array', ctypes.c_uint64, 8), ('src_mem_array_size', ctypes.c_uint64, 16), ('dst_mem_range_array', ctypes.c_uint64, 24), ('dst_mem_array_size', ctypes.c_uint64, 32), ('bytes_copied', ctypes.c_uint64, 40)])
enum_kfd_ioctl_svm_op: dict[int, str] = {(KFD_IOCTL_SVM_OP_SET_ATTR:=0): 'KFD_IOCTL_SVM_OP_SET_ATTR', (KFD_IOCTL_SVM_OP_GET_ATTR:=1): 'KFD_IOCTL_SVM_OP_GET_ATTR'}
enum_kfd_ioctl_svm_location: dict[int, str] = {(KFD_IOCTL_SVM_LOCATION_SYSMEM:=0): 'KFD_IOCTL_SVM_LOCATION_SYSMEM', (KFD_IOCTL_SVM_LOCATION_UNDEFINED:=4294967295): 'KFD_IOCTL_SVM_LOCATION_UNDEFINED'}
enum_kfd_ioctl_svm_attr_type: dict[int, str] = {(KFD_IOCTL_SVM_ATTR_PREFERRED_LOC:=0): 'KFD_IOCTL_SVM_ATTR_PREFERRED_LOC', (KFD_IOCTL_SVM_ATTR_PREFETCH_LOC:=1): 'KFD_IOCTL_SVM_ATTR_PREFETCH_LOC', (KFD_IOCTL_SVM_ATTR_ACCESS:=2): 'KFD_IOCTL_SVM_ATTR_ACCESS', (KFD_IOCTL_SVM_ATTR_ACCESS_IN_PLACE:=3): 'KFD_IOCTL_SVM_ATTR_ACCESS_IN_PLACE', (KFD_IOCTL_SVM_ATTR_NO_ACCESS:=4): 'KFD_IOCTL_SVM_ATTR_NO_ACCESS', (KFD_IOCTL_SVM_ATTR_SET_FLAGS:=5): 'KFD_IOCTL_SVM_ATTR_SET_FLAGS', (KFD_IOCTL_SVM_ATTR_CLR_FLAGS:=6): 'KFD_IOCTL_SVM_ATTR_CLR_FLAGS', (KFD_IOCTL_SVM_ATTR_GRANULARITY:=7): 'KFD_IOCTL_SVM_ATTR_GRANULARITY'}
@c.record
class struct_kfd_ioctl_svm_attribute(c.Struct):
  SIZE = 8
  type: int
  value: int
struct_kfd_ioctl_svm_attribute.register_fields([('type', ctypes.c_uint32, 0), ('value', ctypes.c_uint32, 4)])
@c.record
class struct_kfd_ioctl_svm_args(c.Struct):
  SIZE = 24
  start_addr: int
  size: int
  op: int
  nattr: int
  attrs: c.Array[struct_kfd_ioctl_svm_attribute, Literal[0]]
struct_kfd_ioctl_svm_args.register_fields([('start_addr', ctypes.c_uint64, 0), ('size', ctypes.c_uint64, 8), ('op', ctypes.c_uint32, 16), ('nattr', ctypes.c_uint32, 20), ('attrs', c.Array[struct_kfd_ioctl_svm_attribute, Literal[0]], 24)])
@c.record
class struct_kfd_ioctl_set_xnack_mode_args(c.Struct):
  SIZE = 4
  xnack_enabled: int
__s32: TypeAlias = ctypes.c_int32
struct_kfd_ioctl_set_xnack_mode_args.register_fields([('xnack_enabled', ctypes.c_int32, 0)])
enum_kfd_dbg_trap_override_mode: dict[int, str] = {(KFD_DBG_TRAP_OVERRIDE_OR:=0): 'KFD_DBG_TRAP_OVERRIDE_OR', (KFD_DBG_TRAP_OVERRIDE_REPLACE:=1): 'KFD_DBG_TRAP_OVERRIDE_REPLACE'}
enum_kfd_dbg_trap_mask: dict[int, str] = {(KFD_DBG_TRAP_MASK_FP_INVALID:=1): 'KFD_DBG_TRAP_MASK_FP_INVALID', (KFD_DBG_TRAP_MASK_FP_INPUT_DENORMAL:=2): 'KFD_DBG_TRAP_MASK_FP_INPUT_DENORMAL', (KFD_DBG_TRAP_MASK_FP_DIVIDE_BY_ZERO:=4): 'KFD_DBG_TRAP_MASK_FP_DIVIDE_BY_ZERO', (KFD_DBG_TRAP_MASK_FP_OVERFLOW:=8): 'KFD_DBG_TRAP_MASK_FP_OVERFLOW', (KFD_DBG_TRAP_MASK_FP_UNDERFLOW:=16): 'KFD_DBG_TRAP_MASK_FP_UNDERFLOW', (KFD_DBG_TRAP_MASK_FP_INEXACT:=32): 'KFD_DBG_TRAP_MASK_FP_INEXACT', (KFD_DBG_TRAP_MASK_INT_DIVIDE_BY_ZERO:=64): 'KFD_DBG_TRAP_MASK_INT_DIVIDE_BY_ZERO', (KFD_DBG_TRAP_MASK_DBG_ADDRESS_WATCH:=128): 'KFD_DBG_TRAP_MASK_DBG_ADDRESS_WATCH', (KFD_DBG_TRAP_MASK_DBG_MEMORY_VIOLATION:=256): 'KFD_DBG_TRAP_MASK_DBG_MEMORY_VIOLATION', (KFD_DBG_TRAP_MASK_TRAP_ON_WAVE_START:=1073741824): 'KFD_DBG_TRAP_MASK_TRAP_ON_WAVE_START', (KFD_DBG_TRAP_MASK_TRAP_ON_WAVE_END:=-2147483648): 'KFD_DBG_TRAP_MASK_TRAP_ON_WAVE_END'}
enum_kfd_dbg_trap_wave_launch_mode: dict[int, str] = {(KFD_DBG_TRAP_WAVE_LAUNCH_MODE_NORMAL:=0): 'KFD_DBG_TRAP_WAVE_LAUNCH_MODE_NORMAL', (KFD_DBG_TRAP_WAVE_LAUNCH_MODE_HALT:=1): 'KFD_DBG_TRAP_WAVE_LAUNCH_MODE_HALT', (KFD_DBG_TRAP_WAVE_LAUNCH_MODE_DEBUG:=3): 'KFD_DBG_TRAP_WAVE_LAUNCH_MODE_DEBUG'}
enum_kfd_dbg_trap_address_watch_mode: dict[int, str] = {(KFD_DBG_TRAP_ADDRESS_WATCH_MODE_READ:=0): 'KFD_DBG_TRAP_ADDRESS_WATCH_MODE_READ', (KFD_DBG_TRAP_ADDRESS_WATCH_MODE_NONREAD:=1): 'KFD_DBG_TRAP_ADDRESS_WATCH_MODE_NONREAD', (KFD_DBG_TRAP_ADDRESS_WATCH_MODE_ATOMIC:=2): 'KFD_DBG_TRAP_ADDRESS_WATCH_MODE_ATOMIC', (KFD_DBG_TRAP_ADDRESS_WATCH_MODE_ALL:=3): 'KFD_DBG_TRAP_ADDRESS_WATCH_MODE_ALL'}
enum_kfd_dbg_trap_flags: dict[int, str] = {(KFD_DBG_TRAP_FLAG_SINGLE_MEM_OP:=1): 'KFD_DBG_TRAP_FLAG_SINGLE_MEM_OP', (KFD_DBG_TRAP_FLAG_SINGLE_ALU_OP:=2): 'KFD_DBG_TRAP_FLAG_SINGLE_ALU_OP'}
enum_kfd_dbg_trap_exception_code: dict[int, str] = {(EC_NONE:=0): 'EC_NONE', (EC_QUEUE_WAVE_ABORT:=1): 'EC_QUEUE_WAVE_ABORT', (EC_QUEUE_WAVE_TRAP:=2): 'EC_QUEUE_WAVE_TRAP', (EC_QUEUE_WAVE_MATH_ERROR:=3): 'EC_QUEUE_WAVE_MATH_ERROR', (EC_QUEUE_WAVE_ILLEGAL_INSTRUCTION:=4): 'EC_QUEUE_WAVE_ILLEGAL_INSTRUCTION', (EC_QUEUE_WAVE_MEMORY_VIOLATION:=5): 'EC_QUEUE_WAVE_MEMORY_VIOLATION', (EC_QUEUE_WAVE_APERTURE_VIOLATION:=6): 'EC_QUEUE_WAVE_APERTURE_VIOLATION', (EC_QUEUE_PACKET_DISPATCH_DIM_INVALID:=16): 'EC_QUEUE_PACKET_DISPATCH_DIM_INVALID', (EC_QUEUE_PACKET_DISPATCH_GROUP_SEGMENT_SIZE_INVALID:=17): 'EC_QUEUE_PACKET_DISPATCH_GROUP_SEGMENT_SIZE_INVALID', (EC_QUEUE_PACKET_DISPATCH_CODE_INVALID:=18): 'EC_QUEUE_PACKET_DISPATCH_CODE_INVALID', (EC_QUEUE_PACKET_RESERVED:=19): 'EC_QUEUE_PACKET_RESERVED', (EC_QUEUE_PACKET_UNSUPPORTED:=20): 'EC_QUEUE_PACKET_UNSUPPORTED', (EC_QUEUE_PACKET_DISPATCH_WORK_GROUP_SIZE_INVALID:=21): 'EC_QUEUE_PACKET_DISPATCH_WORK_GROUP_SIZE_INVALID', (EC_QUEUE_PACKET_DISPATCH_REGISTER_INVALID:=22): 'EC_QUEUE_PACKET_DISPATCH_REGISTER_INVALID', (EC_QUEUE_PACKET_VENDOR_UNSUPPORTED:=23): 'EC_QUEUE_PACKET_VENDOR_UNSUPPORTED', (EC_QUEUE_PREEMPTION_ERROR:=30): 'EC_QUEUE_PREEMPTION_ERROR', (EC_QUEUE_NEW:=31): 'EC_QUEUE_NEW', (EC_DEVICE_QUEUE_DELETE:=32): 'EC_DEVICE_QUEUE_DELETE', (EC_DEVICE_MEMORY_VIOLATION:=33): 'EC_DEVICE_MEMORY_VIOLATION', (EC_DEVICE_RAS_ERROR:=34): 'EC_DEVICE_RAS_ERROR', (EC_DEVICE_FATAL_HALT:=35): 'EC_DEVICE_FATAL_HALT', (EC_DEVICE_NEW:=36): 'EC_DEVICE_NEW', (EC_PROCESS_RUNTIME:=48): 'EC_PROCESS_RUNTIME', (EC_PROCESS_DEVICE_REMOVE:=49): 'EC_PROCESS_DEVICE_REMOVE', (EC_MAX:=50): 'EC_MAX'}
enum_kfd_dbg_runtime_state: dict[int, str] = {(DEBUG_RUNTIME_STATE_DISABLED:=0): 'DEBUG_RUNTIME_STATE_DISABLED', (DEBUG_RUNTIME_STATE_ENABLED:=1): 'DEBUG_RUNTIME_STATE_ENABLED', (DEBUG_RUNTIME_STATE_ENABLED_BUSY:=2): 'DEBUG_RUNTIME_STATE_ENABLED_BUSY', (DEBUG_RUNTIME_STATE_ENABLED_ERROR:=3): 'DEBUG_RUNTIME_STATE_ENABLED_ERROR'}
@c.record
class struct_kfd_runtime_info(c.Struct):
  SIZE = 16
  r_debug: int
  runtime_state: int
  ttmp_setup: int
struct_kfd_runtime_info.register_fields([('r_debug', ctypes.c_uint64, 0), ('runtime_state', ctypes.c_uint32, 8), ('ttmp_setup', ctypes.c_uint32, 12)])
@c.record
class struct_kfd_ioctl_runtime_enable_args(c.Struct):
  SIZE = 16
  r_debug: int
  mode_mask: int
  capabilities_mask: int
struct_kfd_ioctl_runtime_enable_args.register_fields([('r_debug', ctypes.c_uint64, 0), ('mode_mask', ctypes.c_uint32, 8), ('capabilities_mask', ctypes.c_uint32, 12)])
@c.record
class struct_kfd_queue_snapshot_entry(c.Struct):
  SIZE = 64
  exception_status: int
  ring_base_address: int
  write_pointer_address: int
  read_pointer_address: int
  ctx_save_restore_address: int
  queue_id: int
  gpu_id: int
  ring_size: int
  queue_type: int
  ctx_save_restore_area_size: int
  reserved: int
struct_kfd_queue_snapshot_entry.register_fields([('exception_status', ctypes.c_uint64, 0), ('ring_base_address', ctypes.c_uint64, 8), ('write_pointer_address', ctypes.c_uint64, 16), ('read_pointer_address', ctypes.c_uint64, 24), ('ctx_save_restore_address', ctypes.c_uint64, 32), ('queue_id', ctypes.c_uint32, 40), ('gpu_id', ctypes.c_uint32, 44), ('ring_size', ctypes.c_uint32, 48), ('queue_type', ctypes.c_uint32, 52), ('ctx_save_restore_area_size', ctypes.c_uint32, 56), ('reserved', ctypes.c_uint32, 60)])
@c.record
class struct_kfd_context_save_area_header(c.Struct):
  SIZE = 40
  wave_state: struct_kfd_context_save_area_header_wave_state
  debug_offset: int
  debug_size: int
  err_payload_addr: int
  err_event_id: int
  reserved1: int
@c.record
class struct_kfd_context_save_area_header_wave_state(c.Struct):
  SIZE = 16
  control_stack_offset: int
  control_stack_size: int
  wave_state_offset: int
  wave_state_size: int
struct_kfd_context_save_area_header_wave_state.register_fields([('control_stack_offset', ctypes.c_uint32, 0), ('control_stack_size', ctypes.c_uint32, 4), ('wave_state_offset', ctypes.c_uint32, 8), ('wave_state_size', ctypes.c_uint32, 12)])
struct_kfd_context_save_area_header.register_fields([('wave_state', struct_kfd_context_save_area_header_wave_state, 0), ('debug_offset', ctypes.c_uint32, 16), ('debug_size', ctypes.c_uint32, 20), ('err_payload_addr', ctypes.c_uint64, 24), ('err_event_id', ctypes.c_uint32, 32), ('reserved1', ctypes.c_uint32, 36)])
enum_kfd_dbg_trap_operations: dict[int, str] = {(KFD_IOC_DBG_TRAP_ENABLE:=0): 'KFD_IOC_DBG_TRAP_ENABLE', (KFD_IOC_DBG_TRAP_DISABLE:=1): 'KFD_IOC_DBG_TRAP_DISABLE', (KFD_IOC_DBG_TRAP_SEND_RUNTIME_EVENT:=2): 'KFD_IOC_DBG_TRAP_SEND_RUNTIME_EVENT', (KFD_IOC_DBG_TRAP_SET_EXCEPTIONS_ENABLED:=3): 'KFD_IOC_DBG_TRAP_SET_EXCEPTIONS_ENABLED', (KFD_IOC_DBG_TRAP_SET_WAVE_LAUNCH_OVERRIDE:=4): 'KFD_IOC_DBG_TRAP_SET_WAVE_LAUNCH_OVERRIDE', (KFD_IOC_DBG_TRAP_SET_WAVE_LAUNCH_MODE:=5): 'KFD_IOC_DBG_TRAP_SET_WAVE_LAUNCH_MODE', (KFD_IOC_DBG_TRAP_SUSPEND_QUEUES:=6): 'KFD_IOC_DBG_TRAP_SUSPEND_QUEUES', (KFD_IOC_DBG_TRAP_RESUME_QUEUES:=7): 'KFD_IOC_DBG_TRAP_RESUME_QUEUES', (KFD_IOC_DBG_TRAP_SET_NODE_ADDRESS_WATCH:=8): 'KFD_IOC_DBG_TRAP_SET_NODE_ADDRESS_WATCH', (KFD_IOC_DBG_TRAP_CLEAR_NODE_ADDRESS_WATCH:=9): 'KFD_IOC_DBG_TRAP_CLEAR_NODE_ADDRESS_WATCH', (KFD_IOC_DBG_TRAP_SET_FLAGS:=10): 'KFD_IOC_DBG_TRAP_SET_FLAGS', (KFD_IOC_DBG_TRAP_QUERY_DEBUG_EVENT:=11): 'KFD_IOC_DBG_TRAP_QUERY_DEBUG_EVENT', (KFD_IOC_DBG_TRAP_QUERY_EXCEPTION_INFO:=12): 'KFD_IOC_DBG_TRAP_QUERY_EXCEPTION_INFO', (KFD_IOC_DBG_TRAP_GET_QUEUE_SNAPSHOT:=13): 'KFD_IOC_DBG_TRAP_GET_QUEUE_SNAPSHOT', (KFD_IOC_DBG_TRAP_GET_DEVICE_SNAPSHOT:=14): 'KFD_IOC_DBG_TRAP_GET_DEVICE_SNAPSHOT'}
@c.record
class struct_kfd_ioctl_dbg_trap_enable_args(c.Struct):
  SIZE = 24
  exception_mask: int
  rinfo_ptr: int
  rinfo_size: int
  dbg_fd: int
struct_kfd_ioctl_dbg_trap_enable_args.register_fields([('exception_mask', ctypes.c_uint64, 0), ('rinfo_ptr', ctypes.c_uint64, 8), ('rinfo_size', ctypes.c_uint32, 16), ('dbg_fd', ctypes.c_uint32, 20)])
@c.record
class struct_kfd_ioctl_dbg_trap_send_runtime_event_args(c.Struct):
  SIZE = 16
  exception_mask: int
  gpu_id: int
  queue_id: int
struct_kfd_ioctl_dbg_trap_send_runtime_event_args.register_fields([('exception_mask', ctypes.c_uint64, 0), ('gpu_id', ctypes.c_uint32, 8), ('queue_id', ctypes.c_uint32, 12)])
@c.record
class struct_kfd_ioctl_dbg_trap_set_exceptions_enabled_args(c.Struct):
  SIZE = 8
  exception_mask: int
struct_kfd_ioctl_dbg_trap_set_exceptions_enabled_args.register_fields([('exception_mask', ctypes.c_uint64, 0)])
@c.record
class struct_kfd_ioctl_dbg_trap_set_wave_launch_override_args(c.Struct):
  SIZE = 16
  override_mode: int
  enable_mask: int
  support_request_mask: int
  pad: int
struct_kfd_ioctl_dbg_trap_set_wave_launch_override_args.register_fields([('override_mode', ctypes.c_uint32, 0), ('enable_mask', ctypes.c_uint32, 4), ('support_request_mask', ctypes.c_uint32, 8), ('pad', ctypes.c_uint32, 12)])
@c.record
class struct_kfd_ioctl_dbg_trap_set_wave_launch_mode_args(c.Struct):
  SIZE = 8
  launch_mode: int
  pad: int
struct_kfd_ioctl_dbg_trap_set_wave_launch_mode_args.register_fields([('launch_mode', ctypes.c_uint32, 0), ('pad', ctypes.c_uint32, 4)])
@c.record
class struct_kfd_ioctl_dbg_trap_suspend_queues_args(c.Struct):
  SIZE = 24
  exception_mask: int
  queue_array_ptr: int
  num_queues: int
  grace_period: int
struct_kfd_ioctl_dbg_trap_suspend_queues_args.register_fields([('exception_mask', ctypes.c_uint64, 0), ('queue_array_ptr', ctypes.c_uint64, 8), ('num_queues', ctypes.c_uint32, 16), ('grace_period', ctypes.c_uint32, 20)])
@c.record
class struct_kfd_ioctl_dbg_trap_resume_queues_args(c.Struct):
  SIZE = 16
  queue_array_ptr: int
  num_queues: int
  pad: int
struct_kfd_ioctl_dbg_trap_resume_queues_args.register_fields([('queue_array_ptr', ctypes.c_uint64, 0), ('num_queues', ctypes.c_uint32, 8), ('pad', ctypes.c_uint32, 12)])
@c.record
class struct_kfd_ioctl_dbg_trap_set_node_address_watch_args(c.Struct):
  SIZE = 24
  address: int
  mode: int
  mask: int
  gpu_id: int
  id: int
struct_kfd_ioctl_dbg_trap_set_node_address_watch_args.register_fields([('address', ctypes.c_uint64, 0), ('mode', ctypes.c_uint32, 8), ('mask', ctypes.c_uint32, 12), ('gpu_id', ctypes.c_uint32, 16), ('id', ctypes.c_uint32, 20)])
@c.record
class struct_kfd_ioctl_dbg_trap_clear_node_address_watch_args(c.Struct):
  SIZE = 8
  gpu_id: int
  id: int
struct_kfd_ioctl_dbg_trap_clear_node_address_watch_args.register_fields([('gpu_id', ctypes.c_uint32, 0), ('id', ctypes.c_uint32, 4)])
@c.record
class struct_kfd_ioctl_dbg_trap_set_flags_args(c.Struct):
  SIZE = 8
  flags: int
  pad: int
struct_kfd_ioctl_dbg_trap_set_flags_args.register_fields([('flags', ctypes.c_uint32, 0), ('pad', ctypes.c_uint32, 4)])
@c.record
class struct_kfd_ioctl_dbg_trap_query_debug_event_args(c.Struct):
  SIZE = 16
  exception_mask: int
  gpu_id: int
  queue_id: int
struct_kfd_ioctl_dbg_trap_query_debug_event_args.register_fields([('exception_mask', ctypes.c_uint64, 0), ('gpu_id', ctypes.c_uint32, 8), ('queue_id', ctypes.c_uint32, 12)])
@c.record
class struct_kfd_ioctl_dbg_trap_query_exception_info_args(c.Struct):
  SIZE = 24
  info_ptr: int
  info_size: int
  source_id: int
  exception_code: int
  clear_exception: int
struct_kfd_ioctl_dbg_trap_query_exception_info_args.register_fields([('info_ptr', ctypes.c_uint64, 0), ('info_size', ctypes.c_uint32, 8), ('source_id', ctypes.c_uint32, 12), ('exception_code', ctypes.c_uint32, 16), ('clear_exception', ctypes.c_uint32, 20)])
@c.record
class struct_kfd_ioctl_dbg_trap_queue_snapshot_args(c.Struct):
  SIZE = 24
  exception_mask: int
  snapshot_buf_ptr: int
  num_queues: int
  entry_size: int
struct_kfd_ioctl_dbg_trap_queue_snapshot_args.register_fields([('exception_mask', ctypes.c_uint64, 0), ('snapshot_buf_ptr', ctypes.c_uint64, 8), ('num_queues', ctypes.c_uint32, 16), ('entry_size', ctypes.c_uint32, 20)])
@c.record
class struct_kfd_ioctl_dbg_trap_device_snapshot_args(c.Struct):
  SIZE = 24
  exception_mask: int
  snapshot_buf_ptr: int
  num_devices: int
  entry_size: int
struct_kfd_ioctl_dbg_trap_device_snapshot_args.register_fields([('exception_mask', ctypes.c_uint64, 0), ('snapshot_buf_ptr', ctypes.c_uint64, 8), ('num_devices', ctypes.c_uint32, 16), ('entry_size', ctypes.c_uint32, 20)])
@c.record
class struct_kfd_ioctl_dbg_trap_args(c.Struct):
  SIZE = 32
  pid: int
  op: int
  enable: struct_kfd_ioctl_dbg_trap_enable_args
  send_runtime_event: struct_kfd_ioctl_dbg_trap_send_runtime_event_args
  set_exceptions_enabled: struct_kfd_ioctl_dbg_trap_set_exceptions_enabled_args
  launch_override: struct_kfd_ioctl_dbg_trap_set_wave_launch_override_args
  launch_mode: struct_kfd_ioctl_dbg_trap_set_wave_launch_mode_args
  suspend_queues: struct_kfd_ioctl_dbg_trap_suspend_queues_args
  resume_queues: struct_kfd_ioctl_dbg_trap_resume_queues_args
  set_node_address_watch: struct_kfd_ioctl_dbg_trap_set_node_address_watch_args
  clear_node_address_watch: struct_kfd_ioctl_dbg_trap_clear_node_address_watch_args
  set_flags: struct_kfd_ioctl_dbg_trap_set_flags_args
  query_debug_event: struct_kfd_ioctl_dbg_trap_query_debug_event_args
  query_exception_info: struct_kfd_ioctl_dbg_trap_query_exception_info_args
  queue_snapshot: struct_kfd_ioctl_dbg_trap_queue_snapshot_args
  device_snapshot: struct_kfd_ioctl_dbg_trap_device_snapshot_args
struct_kfd_ioctl_dbg_trap_args.register_fields([('pid', ctypes.c_uint32, 0), ('op', ctypes.c_uint32, 4), ('enable', struct_kfd_ioctl_dbg_trap_enable_args, 8), ('send_runtime_event', struct_kfd_ioctl_dbg_trap_send_runtime_event_args, 8), ('set_exceptions_enabled', struct_kfd_ioctl_dbg_trap_set_exceptions_enabled_args, 8), ('launch_override', struct_kfd_ioctl_dbg_trap_set_wave_launch_override_args, 8), ('launch_mode', struct_kfd_ioctl_dbg_trap_set_wave_launch_mode_args, 8), ('suspend_queues', struct_kfd_ioctl_dbg_trap_suspend_queues_args, 8), ('resume_queues', struct_kfd_ioctl_dbg_trap_resume_queues_args, 8), ('set_node_address_watch', struct_kfd_ioctl_dbg_trap_set_node_address_watch_args, 8), ('clear_node_address_watch', struct_kfd_ioctl_dbg_trap_clear_node_address_watch_args, 8), ('set_flags', struct_kfd_ioctl_dbg_trap_set_flags_args, 8), ('query_debug_event', struct_kfd_ioctl_dbg_trap_query_debug_event_args, 8), ('query_exception_info', struct_kfd_ioctl_dbg_trap_query_exception_info_args, 8), ('queue_snapshot', struct_kfd_ioctl_dbg_trap_queue_snapshot_args, 8), ('device_snapshot', struct_kfd_ioctl_dbg_trap_device_snapshot_args, 8)])
enum_kfd_ioctl_pc_sample_op: dict[int, str] = {(KFD_IOCTL_PCS_OP_QUERY_CAPABILITIES:=0): 'KFD_IOCTL_PCS_OP_QUERY_CAPABILITIES', (KFD_IOCTL_PCS_OP_CREATE:=1): 'KFD_IOCTL_PCS_OP_CREATE', (KFD_IOCTL_PCS_OP_DESTROY:=2): 'KFD_IOCTL_PCS_OP_DESTROY', (KFD_IOCTL_PCS_OP_START:=3): 'KFD_IOCTL_PCS_OP_START', (KFD_IOCTL_PCS_OP_STOP:=4): 'KFD_IOCTL_PCS_OP_STOP'}
enum_kfd_ioctl_pc_sample_method: dict[int, str] = {(KFD_IOCTL_PCS_METHOD_HOSTTRAP:=1): 'KFD_IOCTL_PCS_METHOD_HOSTTRAP', (KFD_IOCTL_PCS_METHOD_STOCHASTIC:=2): 'KFD_IOCTL_PCS_METHOD_STOCHASTIC'}
enum_kfd_ioctl_pc_sample_type: dict[int, str] = {(KFD_IOCTL_PCS_TYPE_TIME_US:=0): 'KFD_IOCTL_PCS_TYPE_TIME_US', (KFD_IOCTL_PCS_TYPE_CLOCK_CYCLES:=1): 'KFD_IOCTL_PCS_TYPE_CLOCK_CYCLES', (KFD_IOCTL_PCS_TYPE_INSTRUCTIONS:=2): 'KFD_IOCTL_PCS_TYPE_INSTRUCTIONS'}
@c.record
class struct_kfd_pc_sample_info(c.Struct):
  SIZE = 40
  interval: int
  interval_min: int
  interval_max: int
  flags: int
  method: int
  type: int
struct_kfd_pc_sample_info.register_fields([('interval', ctypes.c_uint64, 0), ('interval_min', ctypes.c_uint64, 8), ('interval_max', ctypes.c_uint64, 16), ('flags', ctypes.c_uint64, 24), ('method', ctypes.c_uint32, 32), ('type', ctypes.c_uint32, 36)])
@c.record
class struct_kfd_ioctl_pc_sample_args(c.Struct):
  SIZE = 32
  sample_info_ptr: int
  num_sample_info: int
  op: int
  gpu_id: int
  trace_id: int
  flags: int
  version: int
struct_kfd_ioctl_pc_sample_args.register_fields([('sample_info_ptr', ctypes.c_uint64, 0), ('num_sample_info', ctypes.c_uint32, 8), ('op', ctypes.c_uint32, 12), ('gpu_id', ctypes.c_uint32, 16), ('trace_id', ctypes.c_uint32, 20), ('flags', ctypes.c_uint32, 24), ('version', ctypes.c_uint32, 28)])
enum_kfd_profiler_ops: dict[int, str] = {(KFD_IOC_PROFILER_PMC:=0): 'KFD_IOC_PROFILER_PMC', (KFD_IOC_PROFILER_PC_SAMPLE:=1): 'KFD_IOC_PROFILER_PC_SAMPLE', (KFD_IOC_PROFILER_VERSION:=2): 'KFD_IOC_PROFILER_VERSION'}
@c.record
class struct_kfd_ioctl_pmc_settings(c.Struct):
  SIZE = 12
  gpu_id: int
  lock: int
  perfcount_enable: int
struct_kfd_ioctl_pmc_settings.register_fields([('gpu_id', ctypes.c_uint32, 0), ('lock', ctypes.c_uint32, 4), ('perfcount_enable', ctypes.c_uint32, 8)])
@c.record
class struct_kfd_ioctl_profiler_args(c.Struct):
  SIZE = 40
  op: int
  pc_sample: struct_kfd_ioctl_pc_sample_args
  pmc: struct_kfd_ioctl_pmc_settings
  version: int
struct_kfd_ioctl_profiler_args.register_fields([('op', ctypes.c_uint32, 0), ('pc_sample', struct_kfd_ioctl_pc_sample_args, 8), ('pmc', struct_kfd_ioctl_pmc_settings, 8), ('version', ctypes.c_uint32, 8)])
KFD_IOCTL_MAJOR_VERSION = 1
KFD_IOCTL_MINOR_VERSION = 17
KFD_IOC_QUEUE_TYPE_COMPUTE = 0x0
KFD_IOC_QUEUE_TYPE_SDMA = 0x1
KFD_IOC_QUEUE_TYPE_COMPUTE_AQL = 0x2
KFD_IOC_QUEUE_TYPE_SDMA_XGMI = 0x3
KFD_IOC_QUEUE_TYPE_SDMA_BY_ENG_ID = 0x4
KFD_MAX_QUEUE_PERCENTAGE = 100
KFD_MAX_QUEUE_PRIORITY = 15
KFD_IOC_CACHE_POLICY_COHERENT = 0
KFD_IOC_CACHE_POLICY_NONCOHERENT = 1
NUM_OF_SUPPORTED_GPUS = 7
MAX_ALLOWED_NUM_POINTS = 100
MAX_ALLOWED_AW_BUFF_SIZE = 4096
MAX_ALLOWED_WAC_BUFF_SIZE = 128
KFD_INVALID_FD = 0xffffffff
KFD_IOC_EVENT_SIGNAL = 0
KFD_IOC_EVENT_NODECHANGE = 1
KFD_IOC_EVENT_DEVICESTATECHANGE = 2
KFD_IOC_EVENT_HW_EXCEPTION = 3
KFD_IOC_EVENT_SYSTEM_EVENT = 4
KFD_IOC_EVENT_DEBUG_EVENT = 5
KFD_IOC_EVENT_PROFILE_EVENT = 6
KFD_IOC_EVENT_QUEUE_EVENT = 7
KFD_IOC_EVENT_MEMORY = 8
KFD_IOC_WAIT_RESULT_COMPLETE = 0
KFD_IOC_WAIT_RESULT_TIMEOUT = 1
KFD_IOC_WAIT_RESULT_FAIL = 2
KFD_SIGNAL_EVENT_LIMIT = 4096
KFD_HW_EXCEPTION_WHOLE_GPU_RESET = 0
KFD_HW_EXCEPTION_PER_ENGINE_RESET = 1
KFD_HW_EXCEPTION_GPU_HANG = 0
KFD_HW_EXCEPTION_ECC = 1
KFD_MEM_ERR_NO_RAS = 0
KFD_MEM_ERR_SRAM_ECC = 1
KFD_MEM_ERR_POISON_CONSUMED = 2
KFD_MEM_ERR_GPU_HANG = 3
KFD_IOC_ALLOC_MEM_FLAGS_VRAM = (1 << 0)
KFD_IOC_ALLOC_MEM_FLAGS_GTT = (1 << 1)
KFD_IOC_ALLOC_MEM_FLAGS_USERPTR = (1 << 2)
KFD_IOC_ALLOC_MEM_FLAGS_DOORBELL = (1 << 3)
KFD_IOC_ALLOC_MEM_FLAGS_MMIO_REMAP = (1 << 4)
KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE = (1 << 31)
KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE = (1 << 30)
KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC = (1 << 29)
KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE = (1 << 28)
KFD_IOC_ALLOC_MEM_FLAGS_AQL_QUEUE_MEM = (1 << 27)
KFD_IOC_ALLOC_MEM_FLAGS_COHERENT = (1 << 26)
KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED = (1 << 25)
KFD_IOC_ALLOC_MEM_FLAGS_EXT_COHERENT = (1 << 24)
KFD_IOC_ALLOC_MEM_FLAGS_CONTIGUOUS = (1 << 23)
KFD_SMI_EVENT_MASK_FROM_INDEX = lambda i: (1 << ((i) - 1)) # type: ignore
KFD_SMI_EVENT_MSG_SIZE = 96
KFD_IOCTL_SVM_FLAG_HOST_ACCESS = 0x00000001
KFD_IOCTL_SVM_FLAG_COHERENT = 0x00000002
KFD_IOCTL_SVM_FLAG_HIVE_LOCAL = 0x00000004
KFD_IOCTL_SVM_FLAG_GPU_RO = 0x00000008
KFD_IOCTL_SVM_FLAG_GPU_EXEC = 0x00000010
KFD_IOCTL_SVM_FLAG_GPU_READ_MOSTLY = 0x00000020
KFD_IOCTL_SVM_FLAG_GPU_ALWAYS_MAPPED = 0x00000040
KFD_IOCTL_SVM_FLAG_EXT_COHERENT = 0x00000080
KFD_EC_MASK = lambda ecode: (1 << (ecode - 1)) # type: ignore
KFD_EC_MASK_QUEUE = (KFD_EC_MASK(EC_QUEUE_WAVE_ABORT) | KFD_EC_MASK(EC_QUEUE_WAVE_TRAP) | KFD_EC_MASK(EC_QUEUE_WAVE_MATH_ERROR) | KFD_EC_MASK(EC_QUEUE_WAVE_ILLEGAL_INSTRUCTION) | KFD_EC_MASK(EC_QUEUE_WAVE_MEMORY_VIOLATION) | KFD_EC_MASK(EC_QUEUE_WAVE_APERTURE_VIOLATION) | KFD_EC_MASK(EC_QUEUE_PACKET_DISPATCH_DIM_INVALID) | KFD_EC_MASK(EC_QUEUE_PACKET_DISPATCH_GROUP_SEGMENT_SIZE_INVALID) | KFD_EC_MASK(EC_QUEUE_PACKET_DISPATCH_CODE_INVALID) | KFD_EC_MASK(EC_QUEUE_PACKET_RESERVED) | KFD_EC_MASK(EC_QUEUE_PACKET_UNSUPPORTED) | KFD_EC_MASK(EC_QUEUE_PACKET_DISPATCH_WORK_GROUP_SIZE_INVALID) | KFD_EC_MASK(EC_QUEUE_PACKET_DISPATCH_REGISTER_INVALID) | KFD_EC_MASK(EC_QUEUE_PACKET_VENDOR_UNSUPPORTED)	| KFD_EC_MASK(EC_QUEUE_PREEMPTION_ERROR)	| KFD_EC_MASK(EC_QUEUE_NEW))
KFD_EC_MASK_DEVICE = (KFD_EC_MASK(EC_DEVICE_QUEUE_DELETE) | KFD_EC_MASK(EC_DEVICE_RAS_ERROR) | KFD_EC_MASK(EC_DEVICE_FATAL_HALT) | KFD_EC_MASK(EC_DEVICE_MEMORY_VIOLATION) | KFD_EC_MASK(EC_DEVICE_NEW))
KFD_EC_MASK_PROCESS = (KFD_EC_MASK(EC_PROCESS_RUNTIME) | KFD_EC_MASK(EC_PROCESS_DEVICE_REMOVE))
KFD_EC_MASK_PACKET = (KFD_EC_MASK(EC_QUEUE_PACKET_DISPATCH_DIM_INVALID) | KFD_EC_MASK(EC_QUEUE_PACKET_DISPATCH_GROUP_SEGMENT_SIZE_INVALID) | KFD_EC_MASK(EC_QUEUE_PACKET_DISPATCH_CODE_INVALID) | KFD_EC_MASK(EC_QUEUE_PACKET_RESERVED) | KFD_EC_MASK(EC_QUEUE_PACKET_UNSUPPORTED) | KFD_EC_MASK(EC_QUEUE_PACKET_DISPATCH_WORK_GROUP_SIZE_INVALID) | KFD_EC_MASK(EC_QUEUE_PACKET_DISPATCH_REGISTER_INVALID) | KFD_EC_MASK(EC_QUEUE_PACKET_VENDOR_UNSUPPORTED))
KFD_DBG_EC_IS_VALID = lambda ecode: (ecode > EC_NONE and ecode < EC_MAX) # type: ignore
KFD_DBG_EC_TYPE_IS_QUEUE = lambda ecode: (KFD_DBG_EC_IS_VALID(ecode) and not  not (KFD_EC_MASK(ecode) & KFD_EC_MASK_QUEUE)) # type: ignore
KFD_DBG_EC_TYPE_IS_DEVICE = lambda ecode: (KFD_DBG_EC_IS_VALID(ecode) and not  not (KFD_EC_MASK(ecode) & KFD_EC_MASK_DEVICE)) # type: ignore
KFD_DBG_EC_TYPE_IS_PROCESS = lambda ecode: (KFD_DBG_EC_IS_VALID(ecode) and not  not (KFD_EC_MASK(ecode) & KFD_EC_MASK_PROCESS)) # type: ignore
KFD_DBG_EC_TYPE_IS_PACKET = lambda ecode: (KFD_DBG_EC_IS_VALID(ecode) and not  not (KFD_EC_MASK(ecode) & KFD_EC_MASK_PACKET)) # type: ignore
KFD_RUNTIME_ENABLE_MODE_ENABLE_MASK = 1
KFD_RUNTIME_ENABLE_MODE_TTMP_SAVE_MASK = 2
KFD_DBG_QUEUE_ERROR_BIT = 30
KFD_DBG_QUEUE_INVALID_BIT = 31
KFD_DBG_QUEUE_ERROR_MASK = (1 << KFD_DBG_QUEUE_ERROR_BIT)
KFD_DBG_QUEUE_INVALID_MASK = (1 << KFD_DBG_QUEUE_INVALID_BIT)
KFD_IOCTL_PCS_FLAG_POWER_OF_2 = 0x00000001
KFD_IOCTL_PCS_QUERY_TYPE_FULL = (1 << 0)
KFD_IOC_PROFILER_VERSION_NUM = 1
AMDKFD_IOCTL_BASE = 'K'
AMDKFD_IO = lambda nr: _IO(AMDKFD_IOCTL_BASE, nr) # type: ignore
AMDKFD_IOR = lambda nr,type: _IOR(AMDKFD_IOCTL_BASE, nr, type) # type: ignore
AMDKFD_IOW = lambda nr,type: _IOW(AMDKFD_IOCTL_BASE, nr, type) # type: ignore
AMDKFD_IOWR = lambda nr,type: _IOWR(AMDKFD_IOCTL_BASE, nr, type) # type: ignore
AMDKFD_IOC_GET_VERSION = AMDKFD_IOR(0x01, struct_kfd_ioctl_get_version_args)
AMDKFD_IOC_CREATE_QUEUE = AMDKFD_IOWR(0x02, struct_kfd_ioctl_create_queue_args)
AMDKFD_IOC_DESTROY_QUEUE = AMDKFD_IOWR(0x03, struct_kfd_ioctl_destroy_queue_args)
AMDKFD_IOC_SET_MEMORY_POLICY = AMDKFD_IOW(0x04, struct_kfd_ioctl_set_memory_policy_args)
AMDKFD_IOC_GET_CLOCK_COUNTERS = AMDKFD_IOWR(0x05, struct_kfd_ioctl_get_clock_counters_args)
AMDKFD_IOC_GET_PROCESS_APERTURES = AMDKFD_IOR(0x06, struct_kfd_ioctl_get_process_apertures_args)
AMDKFD_IOC_UPDATE_QUEUE = AMDKFD_IOW(0x07, struct_kfd_ioctl_update_queue_args)
AMDKFD_IOC_CREATE_EVENT = AMDKFD_IOWR(0x08, struct_kfd_ioctl_create_event_args)
AMDKFD_IOC_DESTROY_EVENT = AMDKFD_IOW(0x09, struct_kfd_ioctl_destroy_event_args)
AMDKFD_IOC_SET_EVENT = AMDKFD_IOW(0x0A, struct_kfd_ioctl_set_event_args)
AMDKFD_IOC_RESET_EVENT = AMDKFD_IOW(0x0B, struct_kfd_ioctl_reset_event_args)
AMDKFD_IOC_WAIT_EVENTS = AMDKFD_IOWR(0x0C, struct_kfd_ioctl_wait_events_args)
AMDKFD_IOC_DBG_REGISTER_DEPRECATED = AMDKFD_IOW(0x0D, struct_kfd_ioctl_dbg_register_args)
AMDKFD_IOC_DBG_UNREGISTER_DEPRECATED = AMDKFD_IOW(0x0E, struct_kfd_ioctl_dbg_unregister_args)
AMDKFD_IOC_DBG_ADDRESS_WATCH_DEPRECATED = AMDKFD_IOW(0x0F, struct_kfd_ioctl_dbg_address_watch_args)
AMDKFD_IOC_DBG_WAVE_CONTROL_DEPRECATED = AMDKFD_IOW(0x10, struct_kfd_ioctl_dbg_wave_control_args)
AMDKFD_IOC_SET_SCRATCH_BACKING_VA = AMDKFD_IOWR(0x11, struct_kfd_ioctl_set_scratch_backing_va_args)
AMDKFD_IOC_GET_TILE_CONFIG = AMDKFD_IOWR(0x12, struct_kfd_ioctl_get_tile_config_args)
AMDKFD_IOC_SET_TRAP_HANDLER = AMDKFD_IOW(0x13, struct_kfd_ioctl_set_trap_handler_args)
AMDKFD_IOC_GET_PROCESS_APERTURES_NEW = AMDKFD_IOWR(0x14, struct_kfd_ioctl_get_process_apertures_new_args)
AMDKFD_IOC_ACQUIRE_VM = AMDKFD_IOW(0x15, struct_kfd_ioctl_acquire_vm_args)
AMDKFD_IOC_ALLOC_MEMORY_OF_GPU = AMDKFD_IOWR(0x16, struct_kfd_ioctl_alloc_memory_of_gpu_args)
AMDKFD_IOC_FREE_MEMORY_OF_GPU = AMDKFD_IOW(0x17, struct_kfd_ioctl_free_memory_of_gpu_args)
AMDKFD_IOC_MAP_MEMORY_TO_GPU = AMDKFD_IOWR(0x18, struct_kfd_ioctl_map_memory_to_gpu_args)
AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU = AMDKFD_IOWR(0x19, struct_kfd_ioctl_unmap_memory_from_gpu_args)
AMDKFD_IOC_SET_CU_MASK = AMDKFD_IOW(0x1A, struct_kfd_ioctl_set_cu_mask_args)
AMDKFD_IOC_GET_QUEUE_WAVE_STATE = AMDKFD_IOWR(0x1B, struct_kfd_ioctl_get_queue_wave_state_args)
AMDKFD_IOC_GET_DMABUF_INFO = AMDKFD_IOWR(0x1C, struct_kfd_ioctl_get_dmabuf_info_args)
AMDKFD_IOC_IMPORT_DMABUF = AMDKFD_IOWR(0x1D, struct_kfd_ioctl_import_dmabuf_args)
AMDKFD_IOC_ALLOC_QUEUE_GWS = AMDKFD_IOWR(0x1E, struct_kfd_ioctl_alloc_queue_gws_args)
AMDKFD_IOC_SMI_EVENTS = AMDKFD_IOWR(0x1F, struct_kfd_ioctl_smi_events_args)
AMDKFD_IOC_SVM = AMDKFD_IOWR(0x20, struct_kfd_ioctl_svm_args)
AMDKFD_IOC_SET_XNACK_MODE = AMDKFD_IOWR(0x21, struct_kfd_ioctl_set_xnack_mode_args)
AMDKFD_IOC_CRIU_OP = AMDKFD_IOWR(0x22, struct_kfd_ioctl_criu_args)
AMDKFD_IOC_AVAILABLE_MEMORY = AMDKFD_IOWR(0x23, struct_kfd_ioctl_get_available_memory_args)
AMDKFD_IOC_EXPORT_DMABUF = AMDKFD_IOWR(0x24, struct_kfd_ioctl_export_dmabuf_args)
AMDKFD_IOC_RUNTIME_ENABLE = AMDKFD_IOWR(0x25, struct_kfd_ioctl_runtime_enable_args)
AMDKFD_IOC_DBG_TRAP = AMDKFD_IOWR(0x26, struct_kfd_ioctl_dbg_trap_args)
AMDKFD_COMMAND_START = 0x01
AMDKFD_COMMAND_END = 0x27
AMDKFD_IOC_IPC_IMPORT_HANDLE = AMDKFD_IOWR(0x80, struct_kfd_ioctl_ipc_import_handle_args)
AMDKFD_IOC_IPC_EXPORT_HANDLE = AMDKFD_IOWR(0x81, struct_kfd_ioctl_ipc_export_handle_args)
AMDKFD_IOC_DBG_TRAP_DEPRECATED = AMDKFD_IOWR(0x82, struct_kfd_ioctl_dbg_trap_args_deprecated)
AMDKFD_IOC_CROSS_MEMORY_COPY_DEPRECATED = AMDKFD_IOWR(0x83, struct_kfd_ioctl_cross_memory_copy_deprecated_args)
AMDKFD_IOC_RLC_SPM = AMDKFD_IOWR(0x84, struct_kfd_ioctl_spm_args)
AMDKFD_IOC_PC_SAMPLE = AMDKFD_IOWR(0x85, struct_kfd_ioctl_pc_sample_args)
AMDKFD_IOC_PROFILER = AMDKFD_IOWR(0x86, struct_kfd_ioctl_profiler_args)
AMDKFD_COMMAND_START_2 = 0x80
AMDKFD_COMMAND_END_2 = 0x87