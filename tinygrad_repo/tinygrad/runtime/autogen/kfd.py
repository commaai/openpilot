# mypy: disable-error-code="empty-body"
from __future__ import annotations
import ctypes
from typing import Annotated, Literal, TypeAlias
from tinygrad.runtime.support.c import _IO, _IOW, _IOR, _IOWR
from tinygrad.runtime.support import c
@c.record
class struct_kfd_ioctl_get_version_args(c.Struct):
  SIZE = 8
  major_version: Annotated[Annotated[int, ctypes.c_uint32], 0]
  minor_version: Annotated[Annotated[int, ctypes.c_uint32], 4]
__u32: TypeAlias = Annotated[int, ctypes.c_uint32]
@c.record
class struct_kfd_ioctl_create_queue_args(c.Struct):
  SIZE = 96
  ring_base_address: Annotated[Annotated[int, ctypes.c_uint64], 0]
  write_pointer_address: Annotated[Annotated[int, ctypes.c_uint64], 8]
  read_pointer_address: Annotated[Annotated[int, ctypes.c_uint64], 16]
  doorbell_offset: Annotated[Annotated[int, ctypes.c_uint64], 24]
  ring_size: Annotated[Annotated[int, ctypes.c_uint32], 32]
  gpu_id: Annotated[Annotated[int, ctypes.c_uint32], 36]
  queue_type: Annotated[Annotated[int, ctypes.c_uint32], 40]
  queue_percentage: Annotated[Annotated[int, ctypes.c_uint32], 44]
  queue_priority: Annotated[Annotated[int, ctypes.c_uint32], 48]
  queue_id: Annotated[Annotated[int, ctypes.c_uint32], 52]
  eop_buffer_address: Annotated[Annotated[int, ctypes.c_uint64], 56]
  eop_buffer_size: Annotated[Annotated[int, ctypes.c_uint64], 64]
  ctx_save_restore_address: Annotated[Annotated[int, ctypes.c_uint64], 72]
  ctx_save_restore_size: Annotated[Annotated[int, ctypes.c_uint32], 80]
  ctl_stack_size: Annotated[Annotated[int, ctypes.c_uint32], 84]
  sdma_engine_id: Annotated[Annotated[int, ctypes.c_uint32], 88]
  pad: Annotated[Annotated[int, ctypes.c_uint32], 92]
__u64: TypeAlias = Annotated[int, ctypes.c_uint64]
@c.record
class struct_kfd_ioctl_destroy_queue_args(c.Struct):
  SIZE = 8
  queue_id: Annotated[Annotated[int, ctypes.c_uint32], 0]
  pad: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_kfd_ioctl_update_queue_args(c.Struct):
  SIZE = 24
  ring_base_address: Annotated[Annotated[int, ctypes.c_uint64], 0]
  queue_id: Annotated[Annotated[int, ctypes.c_uint32], 8]
  ring_size: Annotated[Annotated[int, ctypes.c_uint32], 12]
  queue_percentage: Annotated[Annotated[int, ctypes.c_uint32], 16]
  queue_priority: Annotated[Annotated[int, ctypes.c_uint32], 20]
@c.record
class struct_kfd_ioctl_set_cu_mask_args(c.Struct):
  SIZE = 16
  queue_id: Annotated[Annotated[int, ctypes.c_uint32], 0]
  num_cu_mask: Annotated[Annotated[int, ctypes.c_uint32], 4]
  cu_mask_ptr: Annotated[Annotated[int, ctypes.c_uint64], 8]
@c.record
class struct_kfd_ioctl_get_queue_wave_state_args(c.Struct):
  SIZE = 24
  ctl_stack_address: Annotated[Annotated[int, ctypes.c_uint64], 0]
  ctl_stack_used_size: Annotated[Annotated[int, ctypes.c_uint32], 8]
  save_area_used_size: Annotated[Annotated[int, ctypes.c_uint32], 12]
  queue_id: Annotated[Annotated[int, ctypes.c_uint32], 16]
  pad: Annotated[Annotated[int, ctypes.c_uint32], 20]
@c.record
class struct_kfd_ioctl_get_available_memory_args(c.Struct):
  SIZE = 16
  available: Annotated[Annotated[int, ctypes.c_uint64], 0]
  gpu_id: Annotated[Annotated[int, ctypes.c_uint32], 8]
  pad: Annotated[Annotated[int, ctypes.c_uint32], 12]
@c.record
class struct_kfd_dbg_device_info_entry(c.Struct):
  SIZE = 120
  exception_status: Annotated[Annotated[int, ctypes.c_uint64], 0]
  lds_base: Annotated[Annotated[int, ctypes.c_uint64], 8]
  lds_limit: Annotated[Annotated[int, ctypes.c_uint64], 16]
  scratch_base: Annotated[Annotated[int, ctypes.c_uint64], 24]
  scratch_limit: Annotated[Annotated[int, ctypes.c_uint64], 32]
  gpuvm_base: Annotated[Annotated[int, ctypes.c_uint64], 40]
  gpuvm_limit: Annotated[Annotated[int, ctypes.c_uint64], 48]
  gpu_id: Annotated[Annotated[int, ctypes.c_uint32], 56]
  location_id: Annotated[Annotated[int, ctypes.c_uint32], 60]
  vendor_id: Annotated[Annotated[int, ctypes.c_uint32], 64]
  device_id: Annotated[Annotated[int, ctypes.c_uint32], 68]
  revision_id: Annotated[Annotated[int, ctypes.c_uint32], 72]
  subsystem_vendor_id: Annotated[Annotated[int, ctypes.c_uint32], 76]
  subsystem_device_id: Annotated[Annotated[int, ctypes.c_uint32], 80]
  fw_version: Annotated[Annotated[int, ctypes.c_uint32], 84]
  gfx_target_version: Annotated[Annotated[int, ctypes.c_uint32], 88]
  simd_count: Annotated[Annotated[int, ctypes.c_uint32], 92]
  max_waves_per_simd: Annotated[Annotated[int, ctypes.c_uint32], 96]
  array_count: Annotated[Annotated[int, ctypes.c_uint32], 100]
  simd_arrays_per_engine: Annotated[Annotated[int, ctypes.c_uint32], 104]
  num_xcc: Annotated[Annotated[int, ctypes.c_uint32], 108]
  capability: Annotated[Annotated[int, ctypes.c_uint32], 112]
  debug_prop: Annotated[Annotated[int, ctypes.c_uint32], 116]
@c.record
class struct_kfd_ioctl_set_memory_policy_args(c.Struct):
  SIZE = 32
  alternate_aperture_base: Annotated[Annotated[int, ctypes.c_uint64], 0]
  alternate_aperture_size: Annotated[Annotated[int, ctypes.c_uint64], 8]
  gpu_id: Annotated[Annotated[int, ctypes.c_uint32], 16]
  default_policy: Annotated[Annotated[int, ctypes.c_uint32], 20]
  alternate_policy: Annotated[Annotated[int, ctypes.c_uint32], 24]
  pad: Annotated[Annotated[int, ctypes.c_uint32], 28]
@c.record
class struct_kfd_ioctl_get_clock_counters_args(c.Struct):
  SIZE = 40
  gpu_clock_counter: Annotated[Annotated[int, ctypes.c_uint64], 0]
  cpu_clock_counter: Annotated[Annotated[int, ctypes.c_uint64], 8]
  system_clock_counter: Annotated[Annotated[int, ctypes.c_uint64], 16]
  system_clock_freq: Annotated[Annotated[int, ctypes.c_uint64], 24]
  gpu_id: Annotated[Annotated[int, ctypes.c_uint32], 32]
  pad: Annotated[Annotated[int, ctypes.c_uint32], 36]
@c.record
class struct_kfd_process_device_apertures(c.Struct):
  SIZE = 56
  lds_base: Annotated[Annotated[int, ctypes.c_uint64], 0]
  lds_limit: Annotated[Annotated[int, ctypes.c_uint64], 8]
  scratch_base: Annotated[Annotated[int, ctypes.c_uint64], 16]
  scratch_limit: Annotated[Annotated[int, ctypes.c_uint64], 24]
  gpuvm_base: Annotated[Annotated[int, ctypes.c_uint64], 32]
  gpuvm_limit: Annotated[Annotated[int, ctypes.c_uint64], 40]
  gpu_id: Annotated[Annotated[int, ctypes.c_uint32], 48]
  pad: Annotated[Annotated[int, ctypes.c_uint32], 52]
@c.record
class struct_kfd_ioctl_get_process_apertures_args(c.Struct):
  SIZE = 400
  process_apertures: Annotated[c.Array[struct_kfd_process_device_apertures, Literal[7]], 0]
  num_of_nodes: Annotated[Annotated[int, ctypes.c_uint32], 392]
  pad: Annotated[Annotated[int, ctypes.c_uint32], 396]
@c.record
class struct_kfd_ioctl_get_process_apertures_new_args(c.Struct):
  SIZE = 16
  kfd_process_device_apertures_ptr: Annotated[Annotated[int, ctypes.c_uint64], 0]
  num_of_nodes: Annotated[Annotated[int, ctypes.c_uint32], 8]
  pad: Annotated[Annotated[int, ctypes.c_uint32], 12]
@c.record
class struct_kfd_ioctl_dbg_register_args(c.Struct):
  SIZE = 8
  gpu_id: Annotated[Annotated[int, ctypes.c_uint32], 0]
  pad: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_kfd_ioctl_dbg_unregister_args(c.Struct):
  SIZE = 8
  gpu_id: Annotated[Annotated[int, ctypes.c_uint32], 0]
  pad: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_kfd_ioctl_dbg_address_watch_args(c.Struct):
  SIZE = 16
  content_ptr: Annotated[Annotated[int, ctypes.c_uint64], 0]
  gpu_id: Annotated[Annotated[int, ctypes.c_uint32], 8]
  buf_size_in_bytes: Annotated[Annotated[int, ctypes.c_uint32], 12]
@c.record
class struct_kfd_ioctl_dbg_wave_control_args(c.Struct):
  SIZE = 16
  content_ptr: Annotated[Annotated[int, ctypes.c_uint64], 0]
  gpu_id: Annotated[Annotated[int, ctypes.c_uint32], 8]
  buf_size_in_bytes: Annotated[Annotated[int, ctypes.c_uint32], 12]
@c.record
class struct_kfd_ioctl_dbg_trap_args_deprecated(c.Struct):
  SIZE = 40
  exception_mask: Annotated[Annotated[int, ctypes.c_uint64], 0]
  ptr: Annotated[Annotated[int, ctypes.c_uint64], 8]
  pid: Annotated[Annotated[int, ctypes.c_uint32], 16]
  op: Annotated[Annotated[int, ctypes.c_uint32], 20]
  data1: Annotated[Annotated[int, ctypes.c_uint32], 24]
  data2: Annotated[Annotated[int, ctypes.c_uint32], 28]
  data3: Annotated[Annotated[int, ctypes.c_uint32], 32]
  data4: Annotated[Annotated[int, ctypes.c_uint32], 36]
@c.record
class struct_kfd_ioctl_create_event_args(c.Struct):
  SIZE = 32
  event_page_offset: Annotated[Annotated[int, ctypes.c_uint64], 0]
  event_trigger_data: Annotated[Annotated[int, ctypes.c_uint32], 8]
  event_type: Annotated[Annotated[int, ctypes.c_uint32], 12]
  auto_reset: Annotated[Annotated[int, ctypes.c_uint32], 16]
  node_id: Annotated[Annotated[int, ctypes.c_uint32], 20]
  event_id: Annotated[Annotated[int, ctypes.c_uint32], 24]
  event_slot_index: Annotated[Annotated[int, ctypes.c_uint32], 28]
@c.record
class struct_kfd_ioctl_destroy_event_args(c.Struct):
  SIZE = 8
  event_id: Annotated[Annotated[int, ctypes.c_uint32], 0]
  pad: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_kfd_ioctl_set_event_args(c.Struct):
  SIZE = 8
  event_id: Annotated[Annotated[int, ctypes.c_uint32], 0]
  pad: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_kfd_ioctl_reset_event_args(c.Struct):
  SIZE = 8
  event_id: Annotated[Annotated[int, ctypes.c_uint32], 0]
  pad: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_kfd_memory_exception_failure(c.Struct):
  SIZE = 16
  NotPresent: Annotated[Annotated[int, ctypes.c_uint32], 0]
  ReadOnly: Annotated[Annotated[int, ctypes.c_uint32], 4]
  NoExecute: Annotated[Annotated[int, ctypes.c_uint32], 8]
  imprecise: Annotated[Annotated[int, ctypes.c_uint32], 12]
@c.record
class struct_kfd_hsa_memory_exception_data(c.Struct):
  SIZE = 32
  failure: Annotated[struct_kfd_memory_exception_failure, 0]
  va: Annotated[Annotated[int, ctypes.c_uint64], 16]
  gpu_id: Annotated[Annotated[int, ctypes.c_uint32], 24]
  ErrorType: Annotated[Annotated[int, ctypes.c_uint32], 28]
@c.record
class struct_kfd_hsa_hw_exception_data(c.Struct):
  SIZE = 16
  reset_type: Annotated[Annotated[int, ctypes.c_uint32], 0]
  reset_cause: Annotated[Annotated[int, ctypes.c_uint32], 4]
  memory_lost: Annotated[Annotated[int, ctypes.c_uint32], 8]
  gpu_id: Annotated[Annotated[int, ctypes.c_uint32], 12]
@c.record
class struct_kfd_hsa_signal_event_data(c.Struct):
  SIZE = 8
  last_event_age: Annotated[Annotated[int, ctypes.c_uint64], 0]
@c.record
class struct_kfd_event_data(c.Struct):
  SIZE = 48
  memory_exception_data: Annotated[struct_kfd_hsa_memory_exception_data, 0]
  hw_exception_data: Annotated[struct_kfd_hsa_hw_exception_data, 0]
  signal_event_data: Annotated[struct_kfd_hsa_signal_event_data, 0]
  kfd_event_data_ext: Annotated[Annotated[int, ctypes.c_uint64], 32]
  event_id: Annotated[Annotated[int, ctypes.c_uint32], 40]
  pad: Annotated[Annotated[int, ctypes.c_uint32], 44]
@c.record
class struct_kfd_ioctl_wait_events_args(c.Struct):
  SIZE = 24
  events_ptr: Annotated[Annotated[int, ctypes.c_uint64], 0]
  num_events: Annotated[Annotated[int, ctypes.c_uint32], 8]
  wait_for_all: Annotated[Annotated[int, ctypes.c_uint32], 12]
  timeout: Annotated[Annotated[int, ctypes.c_uint32], 16]
  wait_result: Annotated[Annotated[int, ctypes.c_uint32], 20]
@c.record
class struct_kfd_ioctl_set_scratch_backing_va_args(c.Struct):
  SIZE = 16
  va_addr: Annotated[Annotated[int, ctypes.c_uint64], 0]
  gpu_id: Annotated[Annotated[int, ctypes.c_uint32], 8]
  pad: Annotated[Annotated[int, ctypes.c_uint32], 12]
@c.record
class struct_kfd_ioctl_get_tile_config_args(c.Struct):
  SIZE = 40
  tile_config_ptr: Annotated[Annotated[int, ctypes.c_uint64], 0]
  macro_tile_config_ptr: Annotated[Annotated[int, ctypes.c_uint64], 8]
  num_tile_configs: Annotated[Annotated[int, ctypes.c_uint32], 16]
  num_macro_tile_configs: Annotated[Annotated[int, ctypes.c_uint32], 20]
  gpu_id: Annotated[Annotated[int, ctypes.c_uint32], 24]
  gb_addr_config: Annotated[Annotated[int, ctypes.c_uint32], 28]
  num_banks: Annotated[Annotated[int, ctypes.c_uint32], 32]
  num_ranks: Annotated[Annotated[int, ctypes.c_uint32], 36]
@c.record
class struct_kfd_ioctl_set_trap_handler_args(c.Struct):
  SIZE = 24
  tba_addr: Annotated[Annotated[int, ctypes.c_uint64], 0]
  tma_addr: Annotated[Annotated[int, ctypes.c_uint64], 8]
  gpu_id: Annotated[Annotated[int, ctypes.c_uint32], 16]
  pad: Annotated[Annotated[int, ctypes.c_uint32], 20]
@c.record
class struct_kfd_ioctl_acquire_vm_args(c.Struct):
  SIZE = 8
  drm_fd: Annotated[Annotated[int, ctypes.c_uint32], 0]
  gpu_id: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_kfd_ioctl_alloc_memory_of_gpu_args(c.Struct):
  SIZE = 40
  va_addr: Annotated[Annotated[int, ctypes.c_uint64], 0]
  size: Annotated[Annotated[int, ctypes.c_uint64], 8]
  handle: Annotated[Annotated[int, ctypes.c_uint64], 16]
  mmap_offset: Annotated[Annotated[int, ctypes.c_uint64], 24]
  gpu_id: Annotated[Annotated[int, ctypes.c_uint32], 32]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 36]
@c.record
class struct_kfd_ioctl_free_memory_of_gpu_args(c.Struct):
  SIZE = 8
  handle: Annotated[Annotated[int, ctypes.c_uint64], 0]
@c.record
class struct_kfd_ioctl_map_memory_to_gpu_args(c.Struct):
  SIZE = 24
  handle: Annotated[Annotated[int, ctypes.c_uint64], 0]
  device_ids_array_ptr: Annotated[Annotated[int, ctypes.c_uint64], 8]
  n_devices: Annotated[Annotated[int, ctypes.c_uint32], 16]
  n_success: Annotated[Annotated[int, ctypes.c_uint32], 20]
@c.record
class struct_kfd_ioctl_unmap_memory_from_gpu_args(c.Struct):
  SIZE = 24
  handle: Annotated[Annotated[int, ctypes.c_uint64], 0]
  device_ids_array_ptr: Annotated[Annotated[int, ctypes.c_uint64], 8]
  n_devices: Annotated[Annotated[int, ctypes.c_uint32], 16]
  n_success: Annotated[Annotated[int, ctypes.c_uint32], 20]
@c.record
class struct_kfd_ioctl_alloc_queue_gws_args(c.Struct):
  SIZE = 16
  queue_id: Annotated[Annotated[int, ctypes.c_uint32], 0]
  num_gws: Annotated[Annotated[int, ctypes.c_uint32], 4]
  first_gws: Annotated[Annotated[int, ctypes.c_uint32], 8]
  pad: Annotated[Annotated[int, ctypes.c_uint32], 12]
@c.record
class struct_kfd_ioctl_get_dmabuf_info_args(c.Struct):
  SIZE = 32
  size: Annotated[Annotated[int, ctypes.c_uint64], 0]
  metadata_ptr: Annotated[Annotated[int, ctypes.c_uint64], 8]
  metadata_size: Annotated[Annotated[int, ctypes.c_uint32], 16]
  gpu_id: Annotated[Annotated[int, ctypes.c_uint32], 20]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 24]
  dmabuf_fd: Annotated[Annotated[int, ctypes.c_uint32], 28]
@c.record
class struct_kfd_ioctl_import_dmabuf_args(c.Struct):
  SIZE = 24
  va_addr: Annotated[Annotated[int, ctypes.c_uint64], 0]
  handle: Annotated[Annotated[int, ctypes.c_uint64], 8]
  gpu_id: Annotated[Annotated[int, ctypes.c_uint32], 16]
  dmabuf_fd: Annotated[Annotated[int, ctypes.c_uint32], 20]
@c.record
class struct_kfd_ioctl_export_dmabuf_args(c.Struct):
  SIZE = 16
  handle: Annotated[Annotated[int, ctypes.c_uint64], 0]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 8]
  dmabuf_fd: Annotated[Annotated[int, ctypes.c_uint32], 12]
class enum_kfd_smi_event(Annotated[int, ctypes.c_uint32], c.Enum): pass
KFD_SMI_EVENT_NONE = enum_kfd_smi_event.define('KFD_SMI_EVENT_NONE', 0)
KFD_SMI_EVENT_VMFAULT = enum_kfd_smi_event.define('KFD_SMI_EVENT_VMFAULT', 1)
KFD_SMI_EVENT_THERMAL_THROTTLE = enum_kfd_smi_event.define('KFD_SMI_EVENT_THERMAL_THROTTLE', 2)
KFD_SMI_EVENT_GPU_PRE_RESET = enum_kfd_smi_event.define('KFD_SMI_EVENT_GPU_PRE_RESET', 3)
KFD_SMI_EVENT_GPU_POST_RESET = enum_kfd_smi_event.define('KFD_SMI_EVENT_GPU_POST_RESET', 4)
KFD_SMI_EVENT_MIGRATE_START = enum_kfd_smi_event.define('KFD_SMI_EVENT_MIGRATE_START', 5)
KFD_SMI_EVENT_MIGRATE_END = enum_kfd_smi_event.define('KFD_SMI_EVENT_MIGRATE_END', 6)
KFD_SMI_EVENT_PAGE_FAULT_START = enum_kfd_smi_event.define('KFD_SMI_EVENT_PAGE_FAULT_START', 7)
KFD_SMI_EVENT_PAGE_FAULT_END = enum_kfd_smi_event.define('KFD_SMI_EVENT_PAGE_FAULT_END', 8)
KFD_SMI_EVENT_QUEUE_EVICTION = enum_kfd_smi_event.define('KFD_SMI_EVENT_QUEUE_EVICTION', 9)
KFD_SMI_EVENT_QUEUE_RESTORE = enum_kfd_smi_event.define('KFD_SMI_EVENT_QUEUE_RESTORE', 10)
KFD_SMI_EVENT_UNMAP_FROM_GPU = enum_kfd_smi_event.define('KFD_SMI_EVENT_UNMAP_FROM_GPU', 11)
KFD_SMI_EVENT_ALL_PROCESS = enum_kfd_smi_event.define('KFD_SMI_EVENT_ALL_PROCESS', 64)

class enum_KFD_MIGRATE_TRIGGERS(Annotated[int, ctypes.c_uint32], c.Enum): pass
KFD_MIGRATE_TRIGGER_PREFETCH = enum_KFD_MIGRATE_TRIGGERS.define('KFD_MIGRATE_TRIGGER_PREFETCH', 0)
KFD_MIGRATE_TRIGGER_PAGEFAULT_GPU = enum_KFD_MIGRATE_TRIGGERS.define('KFD_MIGRATE_TRIGGER_PAGEFAULT_GPU', 1)
KFD_MIGRATE_TRIGGER_PAGEFAULT_CPU = enum_KFD_MIGRATE_TRIGGERS.define('KFD_MIGRATE_TRIGGER_PAGEFAULT_CPU', 2)
KFD_MIGRATE_TRIGGER_TTM_EVICTION = enum_KFD_MIGRATE_TRIGGERS.define('KFD_MIGRATE_TRIGGER_TTM_EVICTION', 3)

class enum_KFD_QUEUE_EVICTION_TRIGGERS(Annotated[int, ctypes.c_uint32], c.Enum): pass
KFD_QUEUE_EVICTION_TRIGGER_SVM = enum_KFD_QUEUE_EVICTION_TRIGGERS.define('KFD_QUEUE_EVICTION_TRIGGER_SVM', 0)
KFD_QUEUE_EVICTION_TRIGGER_USERPTR = enum_KFD_QUEUE_EVICTION_TRIGGERS.define('KFD_QUEUE_EVICTION_TRIGGER_USERPTR', 1)
KFD_QUEUE_EVICTION_TRIGGER_TTM = enum_KFD_QUEUE_EVICTION_TRIGGERS.define('KFD_QUEUE_EVICTION_TRIGGER_TTM', 2)
KFD_QUEUE_EVICTION_TRIGGER_SUSPEND = enum_KFD_QUEUE_EVICTION_TRIGGERS.define('KFD_QUEUE_EVICTION_TRIGGER_SUSPEND', 3)
KFD_QUEUE_EVICTION_CRIU_CHECKPOINT = enum_KFD_QUEUE_EVICTION_TRIGGERS.define('KFD_QUEUE_EVICTION_CRIU_CHECKPOINT', 4)
KFD_QUEUE_EVICTION_CRIU_RESTORE = enum_KFD_QUEUE_EVICTION_TRIGGERS.define('KFD_QUEUE_EVICTION_CRIU_RESTORE', 5)

class enum_KFD_SVM_UNMAP_TRIGGERS(Annotated[int, ctypes.c_uint32], c.Enum): pass
KFD_SVM_UNMAP_TRIGGER_MMU_NOTIFY = enum_KFD_SVM_UNMAP_TRIGGERS.define('KFD_SVM_UNMAP_TRIGGER_MMU_NOTIFY', 0)
KFD_SVM_UNMAP_TRIGGER_MMU_NOTIFY_MIGRATE = enum_KFD_SVM_UNMAP_TRIGGERS.define('KFD_SVM_UNMAP_TRIGGER_MMU_NOTIFY_MIGRATE', 1)
KFD_SVM_UNMAP_TRIGGER_UNMAP_FROM_CPU = enum_KFD_SVM_UNMAP_TRIGGERS.define('KFD_SVM_UNMAP_TRIGGER_UNMAP_FROM_CPU', 2)

@c.record
class struct_kfd_ioctl_smi_events_args(c.Struct):
  SIZE = 8
  gpuid: Annotated[Annotated[int, ctypes.c_uint32], 0]
  anon_fd: Annotated[Annotated[int, ctypes.c_uint32], 4]
class enum_kfd_ioctl_spm_op(Annotated[int, ctypes.c_uint32], c.Enum): pass
KFD_IOCTL_SPM_OP_ACQUIRE = enum_kfd_ioctl_spm_op.define('KFD_IOCTL_SPM_OP_ACQUIRE', 0)
KFD_IOCTL_SPM_OP_RELEASE = enum_kfd_ioctl_spm_op.define('KFD_IOCTL_SPM_OP_RELEASE', 1)
KFD_IOCTL_SPM_OP_SET_DEST_BUF = enum_kfd_ioctl_spm_op.define('KFD_IOCTL_SPM_OP_SET_DEST_BUF', 2)

@c.record
class struct_kfd_ioctl_spm_args(c.Struct):
  SIZE = 32
  dest_buf: Annotated[Annotated[int, ctypes.c_uint64], 0]
  buf_size: Annotated[Annotated[int, ctypes.c_uint32], 8]
  op: Annotated[Annotated[int, ctypes.c_uint32], 12]
  timeout: Annotated[Annotated[int, ctypes.c_uint32], 16]
  gpu_id: Annotated[Annotated[int, ctypes.c_uint32], 20]
  bytes_copied: Annotated[Annotated[int, ctypes.c_uint32], 24]
  has_data_loss: Annotated[Annotated[int, ctypes.c_uint32], 28]
class enum_kfd_criu_op(Annotated[int, ctypes.c_uint32], c.Enum): pass
KFD_CRIU_OP_PROCESS_INFO = enum_kfd_criu_op.define('KFD_CRIU_OP_PROCESS_INFO', 0)
KFD_CRIU_OP_CHECKPOINT = enum_kfd_criu_op.define('KFD_CRIU_OP_CHECKPOINT', 1)
KFD_CRIU_OP_UNPAUSE = enum_kfd_criu_op.define('KFD_CRIU_OP_UNPAUSE', 2)
KFD_CRIU_OP_RESTORE = enum_kfd_criu_op.define('KFD_CRIU_OP_RESTORE', 3)
KFD_CRIU_OP_RESUME = enum_kfd_criu_op.define('KFD_CRIU_OP_RESUME', 4)

@c.record
class struct_kfd_ioctl_criu_args(c.Struct):
  SIZE = 56
  devices: Annotated[Annotated[int, ctypes.c_uint64], 0]
  bos: Annotated[Annotated[int, ctypes.c_uint64], 8]
  priv_data: Annotated[Annotated[int, ctypes.c_uint64], 16]
  priv_data_size: Annotated[Annotated[int, ctypes.c_uint64], 24]
  num_devices: Annotated[Annotated[int, ctypes.c_uint32], 32]
  num_bos: Annotated[Annotated[int, ctypes.c_uint32], 36]
  num_objects: Annotated[Annotated[int, ctypes.c_uint32], 40]
  pid: Annotated[Annotated[int, ctypes.c_uint32], 44]
  op: Annotated[Annotated[int, ctypes.c_uint32], 48]
@c.record
class struct_kfd_criu_device_bucket(c.Struct):
  SIZE = 16
  user_gpu_id: Annotated[Annotated[int, ctypes.c_uint32], 0]
  actual_gpu_id: Annotated[Annotated[int, ctypes.c_uint32], 4]
  drm_fd: Annotated[Annotated[int, ctypes.c_uint32], 8]
  pad: Annotated[Annotated[int, ctypes.c_uint32], 12]
@c.record
class struct_kfd_criu_bo_bucket(c.Struct):
  SIZE = 48
  addr: Annotated[Annotated[int, ctypes.c_uint64], 0]
  size: Annotated[Annotated[int, ctypes.c_uint64], 8]
  offset: Annotated[Annotated[int, ctypes.c_uint64], 16]
  restored_offset: Annotated[Annotated[int, ctypes.c_uint64], 24]
  gpu_id: Annotated[Annotated[int, ctypes.c_uint32], 32]
  alloc_flags: Annotated[Annotated[int, ctypes.c_uint32], 36]
  dmabuf_fd: Annotated[Annotated[int, ctypes.c_uint32], 40]
  pad: Annotated[Annotated[int, ctypes.c_uint32], 44]
class enum_kfd_mmio_remap(Annotated[int, ctypes.c_uint32], c.Enum): pass
KFD_MMIO_REMAP_HDP_MEM_FLUSH_CNTL = enum_kfd_mmio_remap.define('KFD_MMIO_REMAP_HDP_MEM_FLUSH_CNTL', 0)
KFD_MMIO_REMAP_HDP_REG_FLUSH_CNTL = enum_kfd_mmio_remap.define('KFD_MMIO_REMAP_HDP_REG_FLUSH_CNTL', 4)

@c.record
class struct_kfd_ioctl_ipc_export_handle_args(c.Struct):
  SIZE = 32
  handle: Annotated[Annotated[int, ctypes.c_uint64], 0]
  share_handle: Annotated[c.Array[Annotated[int, ctypes.c_uint32], Literal[4]], 8]
  gpu_id: Annotated[Annotated[int, ctypes.c_uint32], 24]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 28]
@c.record
class struct_kfd_ioctl_ipc_import_handle_args(c.Struct):
  SIZE = 48
  handle: Annotated[Annotated[int, ctypes.c_uint64], 0]
  va_addr: Annotated[Annotated[int, ctypes.c_uint64], 8]
  mmap_offset: Annotated[Annotated[int, ctypes.c_uint64], 16]
  share_handle: Annotated[c.Array[Annotated[int, ctypes.c_uint32], Literal[4]], 24]
  gpu_id: Annotated[Annotated[int, ctypes.c_uint32], 40]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 44]
@c.record
class struct_kfd_ioctl_cross_memory_copy_deprecated_args(c.Struct):
  SIZE = 48
  pid: Annotated[Annotated[int, ctypes.c_uint32], 0]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 4]
  src_mem_range_array: Annotated[Annotated[int, ctypes.c_uint64], 8]
  src_mem_array_size: Annotated[Annotated[int, ctypes.c_uint64], 16]
  dst_mem_range_array: Annotated[Annotated[int, ctypes.c_uint64], 24]
  dst_mem_array_size: Annotated[Annotated[int, ctypes.c_uint64], 32]
  bytes_copied: Annotated[Annotated[int, ctypes.c_uint64], 40]
class enum_kfd_ioctl_svm_op(Annotated[int, ctypes.c_uint32], c.Enum): pass
KFD_IOCTL_SVM_OP_SET_ATTR = enum_kfd_ioctl_svm_op.define('KFD_IOCTL_SVM_OP_SET_ATTR', 0)
KFD_IOCTL_SVM_OP_GET_ATTR = enum_kfd_ioctl_svm_op.define('KFD_IOCTL_SVM_OP_GET_ATTR', 1)

class enum_kfd_ioctl_svm_location(Annotated[int, ctypes.c_uint32], c.Enum): pass
KFD_IOCTL_SVM_LOCATION_SYSMEM = enum_kfd_ioctl_svm_location.define('KFD_IOCTL_SVM_LOCATION_SYSMEM', 0)
KFD_IOCTL_SVM_LOCATION_UNDEFINED = enum_kfd_ioctl_svm_location.define('KFD_IOCTL_SVM_LOCATION_UNDEFINED', 4294967295)

class enum_kfd_ioctl_svm_attr_type(Annotated[int, ctypes.c_uint32], c.Enum): pass
KFD_IOCTL_SVM_ATTR_PREFERRED_LOC = enum_kfd_ioctl_svm_attr_type.define('KFD_IOCTL_SVM_ATTR_PREFERRED_LOC', 0)
KFD_IOCTL_SVM_ATTR_PREFETCH_LOC = enum_kfd_ioctl_svm_attr_type.define('KFD_IOCTL_SVM_ATTR_PREFETCH_LOC', 1)
KFD_IOCTL_SVM_ATTR_ACCESS = enum_kfd_ioctl_svm_attr_type.define('KFD_IOCTL_SVM_ATTR_ACCESS', 2)
KFD_IOCTL_SVM_ATTR_ACCESS_IN_PLACE = enum_kfd_ioctl_svm_attr_type.define('KFD_IOCTL_SVM_ATTR_ACCESS_IN_PLACE', 3)
KFD_IOCTL_SVM_ATTR_NO_ACCESS = enum_kfd_ioctl_svm_attr_type.define('KFD_IOCTL_SVM_ATTR_NO_ACCESS', 4)
KFD_IOCTL_SVM_ATTR_SET_FLAGS = enum_kfd_ioctl_svm_attr_type.define('KFD_IOCTL_SVM_ATTR_SET_FLAGS', 5)
KFD_IOCTL_SVM_ATTR_CLR_FLAGS = enum_kfd_ioctl_svm_attr_type.define('KFD_IOCTL_SVM_ATTR_CLR_FLAGS', 6)
KFD_IOCTL_SVM_ATTR_GRANULARITY = enum_kfd_ioctl_svm_attr_type.define('KFD_IOCTL_SVM_ATTR_GRANULARITY', 7)

@c.record
class struct_kfd_ioctl_svm_attribute(c.Struct):
  SIZE = 8
  type: Annotated[Annotated[int, ctypes.c_uint32], 0]
  value: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_kfd_ioctl_svm_args(c.Struct):
  SIZE = 24
  start_addr: Annotated[Annotated[int, ctypes.c_uint64], 0]
  size: Annotated[Annotated[int, ctypes.c_uint64], 8]
  op: Annotated[Annotated[int, ctypes.c_uint32], 16]
  nattr: Annotated[Annotated[int, ctypes.c_uint32], 20]
  attrs: Annotated[c.Array[struct_kfd_ioctl_svm_attribute, Literal[0]], 24]
@c.record
class struct_kfd_ioctl_set_xnack_mode_args(c.Struct):
  SIZE = 4
  xnack_enabled: Annotated[Annotated[int, ctypes.c_int32], 0]
__s32: TypeAlias = Annotated[int, ctypes.c_int32]
class enum_kfd_dbg_trap_override_mode(Annotated[int, ctypes.c_uint32], c.Enum): pass
KFD_DBG_TRAP_OVERRIDE_OR = enum_kfd_dbg_trap_override_mode.define('KFD_DBG_TRAP_OVERRIDE_OR', 0)
KFD_DBG_TRAP_OVERRIDE_REPLACE = enum_kfd_dbg_trap_override_mode.define('KFD_DBG_TRAP_OVERRIDE_REPLACE', 1)

class enum_kfd_dbg_trap_mask(Annotated[int, ctypes.c_int32], c.Enum): pass
KFD_DBG_TRAP_MASK_FP_INVALID = enum_kfd_dbg_trap_mask.define('KFD_DBG_TRAP_MASK_FP_INVALID', 1)
KFD_DBG_TRAP_MASK_FP_INPUT_DENORMAL = enum_kfd_dbg_trap_mask.define('KFD_DBG_TRAP_MASK_FP_INPUT_DENORMAL', 2)
KFD_DBG_TRAP_MASK_FP_DIVIDE_BY_ZERO = enum_kfd_dbg_trap_mask.define('KFD_DBG_TRAP_MASK_FP_DIVIDE_BY_ZERO', 4)
KFD_DBG_TRAP_MASK_FP_OVERFLOW = enum_kfd_dbg_trap_mask.define('KFD_DBG_TRAP_MASK_FP_OVERFLOW', 8)
KFD_DBG_TRAP_MASK_FP_UNDERFLOW = enum_kfd_dbg_trap_mask.define('KFD_DBG_TRAP_MASK_FP_UNDERFLOW', 16)
KFD_DBG_TRAP_MASK_FP_INEXACT = enum_kfd_dbg_trap_mask.define('KFD_DBG_TRAP_MASK_FP_INEXACT', 32)
KFD_DBG_TRAP_MASK_INT_DIVIDE_BY_ZERO = enum_kfd_dbg_trap_mask.define('KFD_DBG_TRAP_MASK_INT_DIVIDE_BY_ZERO', 64)
KFD_DBG_TRAP_MASK_DBG_ADDRESS_WATCH = enum_kfd_dbg_trap_mask.define('KFD_DBG_TRAP_MASK_DBG_ADDRESS_WATCH', 128)
KFD_DBG_TRAP_MASK_DBG_MEMORY_VIOLATION = enum_kfd_dbg_trap_mask.define('KFD_DBG_TRAP_MASK_DBG_MEMORY_VIOLATION', 256)
KFD_DBG_TRAP_MASK_TRAP_ON_WAVE_START = enum_kfd_dbg_trap_mask.define('KFD_DBG_TRAP_MASK_TRAP_ON_WAVE_START', 1073741824)
KFD_DBG_TRAP_MASK_TRAP_ON_WAVE_END = enum_kfd_dbg_trap_mask.define('KFD_DBG_TRAP_MASK_TRAP_ON_WAVE_END', -2147483648)

class enum_kfd_dbg_trap_wave_launch_mode(Annotated[int, ctypes.c_uint32], c.Enum): pass
KFD_DBG_TRAP_WAVE_LAUNCH_MODE_NORMAL = enum_kfd_dbg_trap_wave_launch_mode.define('KFD_DBG_TRAP_WAVE_LAUNCH_MODE_NORMAL', 0)
KFD_DBG_TRAP_WAVE_LAUNCH_MODE_HALT = enum_kfd_dbg_trap_wave_launch_mode.define('KFD_DBG_TRAP_WAVE_LAUNCH_MODE_HALT', 1)
KFD_DBG_TRAP_WAVE_LAUNCH_MODE_DEBUG = enum_kfd_dbg_trap_wave_launch_mode.define('KFD_DBG_TRAP_WAVE_LAUNCH_MODE_DEBUG', 3)

class enum_kfd_dbg_trap_address_watch_mode(Annotated[int, ctypes.c_uint32], c.Enum): pass
KFD_DBG_TRAP_ADDRESS_WATCH_MODE_READ = enum_kfd_dbg_trap_address_watch_mode.define('KFD_DBG_TRAP_ADDRESS_WATCH_MODE_READ', 0)
KFD_DBG_TRAP_ADDRESS_WATCH_MODE_NONREAD = enum_kfd_dbg_trap_address_watch_mode.define('KFD_DBG_TRAP_ADDRESS_WATCH_MODE_NONREAD', 1)
KFD_DBG_TRAP_ADDRESS_WATCH_MODE_ATOMIC = enum_kfd_dbg_trap_address_watch_mode.define('KFD_DBG_TRAP_ADDRESS_WATCH_MODE_ATOMIC', 2)
KFD_DBG_TRAP_ADDRESS_WATCH_MODE_ALL = enum_kfd_dbg_trap_address_watch_mode.define('KFD_DBG_TRAP_ADDRESS_WATCH_MODE_ALL', 3)

class enum_kfd_dbg_trap_flags(Annotated[int, ctypes.c_uint32], c.Enum): pass
KFD_DBG_TRAP_FLAG_SINGLE_MEM_OP = enum_kfd_dbg_trap_flags.define('KFD_DBG_TRAP_FLAG_SINGLE_MEM_OP', 1)
KFD_DBG_TRAP_FLAG_SINGLE_ALU_OP = enum_kfd_dbg_trap_flags.define('KFD_DBG_TRAP_FLAG_SINGLE_ALU_OP', 2)

class enum_kfd_dbg_trap_exception_code(Annotated[int, ctypes.c_uint32], c.Enum): pass
EC_NONE = enum_kfd_dbg_trap_exception_code.define('EC_NONE', 0)
EC_QUEUE_WAVE_ABORT = enum_kfd_dbg_trap_exception_code.define('EC_QUEUE_WAVE_ABORT', 1)
EC_QUEUE_WAVE_TRAP = enum_kfd_dbg_trap_exception_code.define('EC_QUEUE_WAVE_TRAP', 2)
EC_QUEUE_WAVE_MATH_ERROR = enum_kfd_dbg_trap_exception_code.define('EC_QUEUE_WAVE_MATH_ERROR', 3)
EC_QUEUE_WAVE_ILLEGAL_INSTRUCTION = enum_kfd_dbg_trap_exception_code.define('EC_QUEUE_WAVE_ILLEGAL_INSTRUCTION', 4)
EC_QUEUE_WAVE_MEMORY_VIOLATION = enum_kfd_dbg_trap_exception_code.define('EC_QUEUE_WAVE_MEMORY_VIOLATION', 5)
EC_QUEUE_WAVE_APERTURE_VIOLATION = enum_kfd_dbg_trap_exception_code.define('EC_QUEUE_WAVE_APERTURE_VIOLATION', 6)
EC_QUEUE_PACKET_DISPATCH_DIM_INVALID = enum_kfd_dbg_trap_exception_code.define('EC_QUEUE_PACKET_DISPATCH_DIM_INVALID', 16)
EC_QUEUE_PACKET_DISPATCH_GROUP_SEGMENT_SIZE_INVALID = enum_kfd_dbg_trap_exception_code.define('EC_QUEUE_PACKET_DISPATCH_GROUP_SEGMENT_SIZE_INVALID', 17)
EC_QUEUE_PACKET_DISPATCH_CODE_INVALID = enum_kfd_dbg_trap_exception_code.define('EC_QUEUE_PACKET_DISPATCH_CODE_INVALID', 18)
EC_QUEUE_PACKET_RESERVED = enum_kfd_dbg_trap_exception_code.define('EC_QUEUE_PACKET_RESERVED', 19)
EC_QUEUE_PACKET_UNSUPPORTED = enum_kfd_dbg_trap_exception_code.define('EC_QUEUE_PACKET_UNSUPPORTED', 20)
EC_QUEUE_PACKET_DISPATCH_WORK_GROUP_SIZE_INVALID = enum_kfd_dbg_trap_exception_code.define('EC_QUEUE_PACKET_DISPATCH_WORK_GROUP_SIZE_INVALID', 21)
EC_QUEUE_PACKET_DISPATCH_REGISTER_INVALID = enum_kfd_dbg_trap_exception_code.define('EC_QUEUE_PACKET_DISPATCH_REGISTER_INVALID', 22)
EC_QUEUE_PACKET_VENDOR_UNSUPPORTED = enum_kfd_dbg_trap_exception_code.define('EC_QUEUE_PACKET_VENDOR_UNSUPPORTED', 23)
EC_QUEUE_PREEMPTION_ERROR = enum_kfd_dbg_trap_exception_code.define('EC_QUEUE_PREEMPTION_ERROR', 30)
EC_QUEUE_NEW = enum_kfd_dbg_trap_exception_code.define('EC_QUEUE_NEW', 31)
EC_DEVICE_QUEUE_DELETE = enum_kfd_dbg_trap_exception_code.define('EC_DEVICE_QUEUE_DELETE', 32)
EC_DEVICE_MEMORY_VIOLATION = enum_kfd_dbg_trap_exception_code.define('EC_DEVICE_MEMORY_VIOLATION', 33)
EC_DEVICE_RAS_ERROR = enum_kfd_dbg_trap_exception_code.define('EC_DEVICE_RAS_ERROR', 34)
EC_DEVICE_FATAL_HALT = enum_kfd_dbg_trap_exception_code.define('EC_DEVICE_FATAL_HALT', 35)
EC_DEVICE_NEW = enum_kfd_dbg_trap_exception_code.define('EC_DEVICE_NEW', 36)
EC_PROCESS_RUNTIME = enum_kfd_dbg_trap_exception_code.define('EC_PROCESS_RUNTIME', 48)
EC_PROCESS_DEVICE_REMOVE = enum_kfd_dbg_trap_exception_code.define('EC_PROCESS_DEVICE_REMOVE', 49)
EC_MAX = enum_kfd_dbg_trap_exception_code.define('EC_MAX', 50)

class enum_kfd_dbg_runtime_state(Annotated[int, ctypes.c_uint32], c.Enum): pass
DEBUG_RUNTIME_STATE_DISABLED = enum_kfd_dbg_runtime_state.define('DEBUG_RUNTIME_STATE_DISABLED', 0)
DEBUG_RUNTIME_STATE_ENABLED = enum_kfd_dbg_runtime_state.define('DEBUG_RUNTIME_STATE_ENABLED', 1)
DEBUG_RUNTIME_STATE_ENABLED_BUSY = enum_kfd_dbg_runtime_state.define('DEBUG_RUNTIME_STATE_ENABLED_BUSY', 2)
DEBUG_RUNTIME_STATE_ENABLED_ERROR = enum_kfd_dbg_runtime_state.define('DEBUG_RUNTIME_STATE_ENABLED_ERROR', 3)

@c.record
class struct_kfd_runtime_info(c.Struct):
  SIZE = 16
  r_debug: Annotated[Annotated[int, ctypes.c_uint64], 0]
  runtime_state: Annotated[Annotated[int, ctypes.c_uint32], 8]
  ttmp_setup: Annotated[Annotated[int, ctypes.c_uint32], 12]
@c.record
class struct_kfd_ioctl_runtime_enable_args(c.Struct):
  SIZE = 16
  r_debug: Annotated[Annotated[int, ctypes.c_uint64], 0]
  mode_mask: Annotated[Annotated[int, ctypes.c_uint32], 8]
  capabilities_mask: Annotated[Annotated[int, ctypes.c_uint32], 12]
@c.record
class struct_kfd_queue_snapshot_entry(c.Struct):
  SIZE = 64
  exception_status: Annotated[Annotated[int, ctypes.c_uint64], 0]
  ring_base_address: Annotated[Annotated[int, ctypes.c_uint64], 8]
  write_pointer_address: Annotated[Annotated[int, ctypes.c_uint64], 16]
  read_pointer_address: Annotated[Annotated[int, ctypes.c_uint64], 24]
  ctx_save_restore_address: Annotated[Annotated[int, ctypes.c_uint64], 32]
  queue_id: Annotated[Annotated[int, ctypes.c_uint32], 40]
  gpu_id: Annotated[Annotated[int, ctypes.c_uint32], 44]
  ring_size: Annotated[Annotated[int, ctypes.c_uint32], 48]
  queue_type: Annotated[Annotated[int, ctypes.c_uint32], 52]
  ctx_save_restore_area_size: Annotated[Annotated[int, ctypes.c_uint32], 56]
  reserved: Annotated[Annotated[int, ctypes.c_uint32], 60]
@c.record
class struct_kfd_context_save_area_header(c.Struct):
  SIZE = 40
  wave_state: Annotated[struct_kfd_context_save_area_header_wave_state, 0]
  debug_offset: Annotated[Annotated[int, ctypes.c_uint32], 16]
  debug_size: Annotated[Annotated[int, ctypes.c_uint32], 20]
  err_payload_addr: Annotated[Annotated[int, ctypes.c_uint64], 24]
  err_event_id: Annotated[Annotated[int, ctypes.c_uint32], 32]
  reserved1: Annotated[Annotated[int, ctypes.c_uint32], 36]
@c.record
class struct_kfd_context_save_area_header_wave_state(c.Struct):
  SIZE = 16
  control_stack_offset: Annotated[Annotated[int, ctypes.c_uint32], 0]
  control_stack_size: Annotated[Annotated[int, ctypes.c_uint32], 4]
  wave_state_offset: Annotated[Annotated[int, ctypes.c_uint32], 8]
  wave_state_size: Annotated[Annotated[int, ctypes.c_uint32], 12]
class enum_kfd_dbg_trap_operations(Annotated[int, ctypes.c_uint32], c.Enum): pass
KFD_IOC_DBG_TRAP_ENABLE = enum_kfd_dbg_trap_operations.define('KFD_IOC_DBG_TRAP_ENABLE', 0)
KFD_IOC_DBG_TRAP_DISABLE = enum_kfd_dbg_trap_operations.define('KFD_IOC_DBG_TRAP_DISABLE', 1)
KFD_IOC_DBG_TRAP_SEND_RUNTIME_EVENT = enum_kfd_dbg_trap_operations.define('KFD_IOC_DBG_TRAP_SEND_RUNTIME_EVENT', 2)
KFD_IOC_DBG_TRAP_SET_EXCEPTIONS_ENABLED = enum_kfd_dbg_trap_operations.define('KFD_IOC_DBG_TRAP_SET_EXCEPTIONS_ENABLED', 3)
KFD_IOC_DBG_TRAP_SET_WAVE_LAUNCH_OVERRIDE = enum_kfd_dbg_trap_operations.define('KFD_IOC_DBG_TRAP_SET_WAVE_LAUNCH_OVERRIDE', 4)
KFD_IOC_DBG_TRAP_SET_WAVE_LAUNCH_MODE = enum_kfd_dbg_trap_operations.define('KFD_IOC_DBG_TRAP_SET_WAVE_LAUNCH_MODE', 5)
KFD_IOC_DBG_TRAP_SUSPEND_QUEUES = enum_kfd_dbg_trap_operations.define('KFD_IOC_DBG_TRAP_SUSPEND_QUEUES', 6)
KFD_IOC_DBG_TRAP_RESUME_QUEUES = enum_kfd_dbg_trap_operations.define('KFD_IOC_DBG_TRAP_RESUME_QUEUES', 7)
KFD_IOC_DBG_TRAP_SET_NODE_ADDRESS_WATCH = enum_kfd_dbg_trap_operations.define('KFD_IOC_DBG_TRAP_SET_NODE_ADDRESS_WATCH', 8)
KFD_IOC_DBG_TRAP_CLEAR_NODE_ADDRESS_WATCH = enum_kfd_dbg_trap_operations.define('KFD_IOC_DBG_TRAP_CLEAR_NODE_ADDRESS_WATCH', 9)
KFD_IOC_DBG_TRAP_SET_FLAGS = enum_kfd_dbg_trap_operations.define('KFD_IOC_DBG_TRAP_SET_FLAGS', 10)
KFD_IOC_DBG_TRAP_QUERY_DEBUG_EVENT = enum_kfd_dbg_trap_operations.define('KFD_IOC_DBG_TRAP_QUERY_DEBUG_EVENT', 11)
KFD_IOC_DBG_TRAP_QUERY_EXCEPTION_INFO = enum_kfd_dbg_trap_operations.define('KFD_IOC_DBG_TRAP_QUERY_EXCEPTION_INFO', 12)
KFD_IOC_DBG_TRAP_GET_QUEUE_SNAPSHOT = enum_kfd_dbg_trap_operations.define('KFD_IOC_DBG_TRAP_GET_QUEUE_SNAPSHOT', 13)
KFD_IOC_DBG_TRAP_GET_DEVICE_SNAPSHOT = enum_kfd_dbg_trap_operations.define('KFD_IOC_DBG_TRAP_GET_DEVICE_SNAPSHOT', 14)

@c.record
class struct_kfd_ioctl_dbg_trap_enable_args(c.Struct):
  SIZE = 24
  exception_mask: Annotated[Annotated[int, ctypes.c_uint64], 0]
  rinfo_ptr: Annotated[Annotated[int, ctypes.c_uint64], 8]
  rinfo_size: Annotated[Annotated[int, ctypes.c_uint32], 16]
  dbg_fd: Annotated[Annotated[int, ctypes.c_uint32], 20]
@c.record
class struct_kfd_ioctl_dbg_trap_send_runtime_event_args(c.Struct):
  SIZE = 16
  exception_mask: Annotated[Annotated[int, ctypes.c_uint64], 0]
  gpu_id: Annotated[Annotated[int, ctypes.c_uint32], 8]
  queue_id: Annotated[Annotated[int, ctypes.c_uint32], 12]
@c.record
class struct_kfd_ioctl_dbg_trap_set_exceptions_enabled_args(c.Struct):
  SIZE = 8
  exception_mask: Annotated[Annotated[int, ctypes.c_uint64], 0]
@c.record
class struct_kfd_ioctl_dbg_trap_set_wave_launch_override_args(c.Struct):
  SIZE = 16
  override_mode: Annotated[Annotated[int, ctypes.c_uint32], 0]
  enable_mask: Annotated[Annotated[int, ctypes.c_uint32], 4]
  support_request_mask: Annotated[Annotated[int, ctypes.c_uint32], 8]
  pad: Annotated[Annotated[int, ctypes.c_uint32], 12]
@c.record
class struct_kfd_ioctl_dbg_trap_set_wave_launch_mode_args(c.Struct):
  SIZE = 8
  launch_mode: Annotated[Annotated[int, ctypes.c_uint32], 0]
  pad: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_kfd_ioctl_dbg_trap_suspend_queues_args(c.Struct):
  SIZE = 24
  exception_mask: Annotated[Annotated[int, ctypes.c_uint64], 0]
  queue_array_ptr: Annotated[Annotated[int, ctypes.c_uint64], 8]
  num_queues: Annotated[Annotated[int, ctypes.c_uint32], 16]
  grace_period: Annotated[Annotated[int, ctypes.c_uint32], 20]
@c.record
class struct_kfd_ioctl_dbg_trap_resume_queues_args(c.Struct):
  SIZE = 16
  queue_array_ptr: Annotated[Annotated[int, ctypes.c_uint64], 0]
  num_queues: Annotated[Annotated[int, ctypes.c_uint32], 8]
  pad: Annotated[Annotated[int, ctypes.c_uint32], 12]
@c.record
class struct_kfd_ioctl_dbg_trap_set_node_address_watch_args(c.Struct):
  SIZE = 24
  address: Annotated[Annotated[int, ctypes.c_uint64], 0]
  mode: Annotated[Annotated[int, ctypes.c_uint32], 8]
  mask: Annotated[Annotated[int, ctypes.c_uint32], 12]
  gpu_id: Annotated[Annotated[int, ctypes.c_uint32], 16]
  id: Annotated[Annotated[int, ctypes.c_uint32], 20]
@c.record
class struct_kfd_ioctl_dbg_trap_clear_node_address_watch_args(c.Struct):
  SIZE = 8
  gpu_id: Annotated[Annotated[int, ctypes.c_uint32], 0]
  id: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_kfd_ioctl_dbg_trap_set_flags_args(c.Struct):
  SIZE = 8
  flags: Annotated[Annotated[int, ctypes.c_uint32], 0]
  pad: Annotated[Annotated[int, ctypes.c_uint32], 4]
@c.record
class struct_kfd_ioctl_dbg_trap_query_debug_event_args(c.Struct):
  SIZE = 16
  exception_mask: Annotated[Annotated[int, ctypes.c_uint64], 0]
  gpu_id: Annotated[Annotated[int, ctypes.c_uint32], 8]
  queue_id: Annotated[Annotated[int, ctypes.c_uint32], 12]
@c.record
class struct_kfd_ioctl_dbg_trap_query_exception_info_args(c.Struct):
  SIZE = 24
  info_ptr: Annotated[Annotated[int, ctypes.c_uint64], 0]
  info_size: Annotated[Annotated[int, ctypes.c_uint32], 8]
  source_id: Annotated[Annotated[int, ctypes.c_uint32], 12]
  exception_code: Annotated[Annotated[int, ctypes.c_uint32], 16]
  clear_exception: Annotated[Annotated[int, ctypes.c_uint32], 20]
@c.record
class struct_kfd_ioctl_dbg_trap_queue_snapshot_args(c.Struct):
  SIZE = 24
  exception_mask: Annotated[Annotated[int, ctypes.c_uint64], 0]
  snapshot_buf_ptr: Annotated[Annotated[int, ctypes.c_uint64], 8]
  num_queues: Annotated[Annotated[int, ctypes.c_uint32], 16]
  entry_size: Annotated[Annotated[int, ctypes.c_uint32], 20]
@c.record
class struct_kfd_ioctl_dbg_trap_device_snapshot_args(c.Struct):
  SIZE = 24
  exception_mask: Annotated[Annotated[int, ctypes.c_uint64], 0]
  snapshot_buf_ptr: Annotated[Annotated[int, ctypes.c_uint64], 8]
  num_devices: Annotated[Annotated[int, ctypes.c_uint32], 16]
  entry_size: Annotated[Annotated[int, ctypes.c_uint32], 20]
@c.record
class struct_kfd_ioctl_dbg_trap_args(c.Struct):
  SIZE = 32
  pid: Annotated[Annotated[int, ctypes.c_uint32], 0]
  op: Annotated[Annotated[int, ctypes.c_uint32], 4]
  enable: Annotated[struct_kfd_ioctl_dbg_trap_enable_args, 8]
  send_runtime_event: Annotated[struct_kfd_ioctl_dbg_trap_send_runtime_event_args, 8]
  set_exceptions_enabled: Annotated[struct_kfd_ioctl_dbg_trap_set_exceptions_enabled_args, 8]
  launch_override: Annotated[struct_kfd_ioctl_dbg_trap_set_wave_launch_override_args, 8]
  launch_mode: Annotated[struct_kfd_ioctl_dbg_trap_set_wave_launch_mode_args, 8]
  suspend_queues: Annotated[struct_kfd_ioctl_dbg_trap_suspend_queues_args, 8]
  resume_queues: Annotated[struct_kfd_ioctl_dbg_trap_resume_queues_args, 8]
  set_node_address_watch: Annotated[struct_kfd_ioctl_dbg_trap_set_node_address_watch_args, 8]
  clear_node_address_watch: Annotated[struct_kfd_ioctl_dbg_trap_clear_node_address_watch_args, 8]
  set_flags: Annotated[struct_kfd_ioctl_dbg_trap_set_flags_args, 8]
  query_debug_event: Annotated[struct_kfd_ioctl_dbg_trap_query_debug_event_args, 8]
  query_exception_info: Annotated[struct_kfd_ioctl_dbg_trap_query_exception_info_args, 8]
  queue_snapshot: Annotated[struct_kfd_ioctl_dbg_trap_queue_snapshot_args, 8]
  device_snapshot: Annotated[struct_kfd_ioctl_dbg_trap_device_snapshot_args, 8]
class enum_kfd_ioctl_pc_sample_op(Annotated[int, ctypes.c_uint32], c.Enum): pass
KFD_IOCTL_PCS_OP_QUERY_CAPABILITIES = enum_kfd_ioctl_pc_sample_op.define('KFD_IOCTL_PCS_OP_QUERY_CAPABILITIES', 0)
KFD_IOCTL_PCS_OP_CREATE = enum_kfd_ioctl_pc_sample_op.define('KFD_IOCTL_PCS_OP_CREATE', 1)
KFD_IOCTL_PCS_OP_DESTROY = enum_kfd_ioctl_pc_sample_op.define('KFD_IOCTL_PCS_OP_DESTROY', 2)
KFD_IOCTL_PCS_OP_START = enum_kfd_ioctl_pc_sample_op.define('KFD_IOCTL_PCS_OP_START', 3)
KFD_IOCTL_PCS_OP_STOP = enum_kfd_ioctl_pc_sample_op.define('KFD_IOCTL_PCS_OP_STOP', 4)

class enum_kfd_ioctl_pc_sample_method(Annotated[int, ctypes.c_uint32], c.Enum): pass
KFD_IOCTL_PCS_METHOD_HOSTTRAP = enum_kfd_ioctl_pc_sample_method.define('KFD_IOCTL_PCS_METHOD_HOSTTRAP', 1)
KFD_IOCTL_PCS_METHOD_STOCHASTIC = enum_kfd_ioctl_pc_sample_method.define('KFD_IOCTL_PCS_METHOD_STOCHASTIC', 2)

class enum_kfd_ioctl_pc_sample_type(Annotated[int, ctypes.c_uint32], c.Enum): pass
KFD_IOCTL_PCS_TYPE_TIME_US = enum_kfd_ioctl_pc_sample_type.define('KFD_IOCTL_PCS_TYPE_TIME_US', 0)
KFD_IOCTL_PCS_TYPE_CLOCK_CYCLES = enum_kfd_ioctl_pc_sample_type.define('KFD_IOCTL_PCS_TYPE_CLOCK_CYCLES', 1)
KFD_IOCTL_PCS_TYPE_INSTRUCTIONS = enum_kfd_ioctl_pc_sample_type.define('KFD_IOCTL_PCS_TYPE_INSTRUCTIONS', 2)

@c.record
class struct_kfd_pc_sample_info(c.Struct):
  SIZE = 40
  interval: Annotated[Annotated[int, ctypes.c_uint64], 0]
  interval_min: Annotated[Annotated[int, ctypes.c_uint64], 8]
  interval_max: Annotated[Annotated[int, ctypes.c_uint64], 16]
  flags: Annotated[Annotated[int, ctypes.c_uint64], 24]
  method: Annotated[Annotated[int, ctypes.c_uint32], 32]
  type: Annotated[Annotated[int, ctypes.c_uint32], 36]
@c.record
class struct_kfd_ioctl_pc_sample_args(c.Struct):
  SIZE = 32
  sample_info_ptr: Annotated[Annotated[int, ctypes.c_uint64], 0]
  num_sample_info: Annotated[Annotated[int, ctypes.c_uint32], 8]
  op: Annotated[Annotated[int, ctypes.c_uint32], 12]
  gpu_id: Annotated[Annotated[int, ctypes.c_uint32], 16]
  trace_id: Annotated[Annotated[int, ctypes.c_uint32], 20]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 24]
  version: Annotated[Annotated[int, ctypes.c_uint32], 28]
class enum_kfd_profiler_ops(Annotated[int, ctypes.c_uint32], c.Enum): pass
KFD_IOC_PROFILER_PMC = enum_kfd_profiler_ops.define('KFD_IOC_PROFILER_PMC', 0)
KFD_IOC_PROFILER_PC_SAMPLE = enum_kfd_profiler_ops.define('KFD_IOC_PROFILER_PC_SAMPLE', 1)
KFD_IOC_PROFILER_VERSION = enum_kfd_profiler_ops.define('KFD_IOC_PROFILER_VERSION', 2)

@c.record
class struct_kfd_ioctl_pmc_settings(c.Struct):
  SIZE = 12
  gpu_id: Annotated[Annotated[int, ctypes.c_uint32], 0]
  lock: Annotated[Annotated[int, ctypes.c_uint32], 4]
  perfcount_enable: Annotated[Annotated[int, ctypes.c_uint32], 8]
@c.record
class struct_kfd_ioctl_profiler_args(c.Struct):
  SIZE = 40
  op: Annotated[Annotated[int, ctypes.c_uint32], 0]
  pc_sample: Annotated[struct_kfd_ioctl_pc_sample_args, 8]
  pmc: Annotated[struct_kfd_ioctl_pmc_settings, 8]
  version: Annotated[Annotated[int, ctypes.c_uint32], 8]
c.init_records()
KFD_IOCTL_MAJOR_VERSION = 1 # type: ignore
KFD_IOCTL_MINOR_VERSION = 17 # type: ignore
KFD_IOC_QUEUE_TYPE_COMPUTE = 0x0 # type: ignore
KFD_IOC_QUEUE_TYPE_SDMA = 0x1 # type: ignore
KFD_IOC_QUEUE_TYPE_COMPUTE_AQL = 0x2 # type: ignore
KFD_IOC_QUEUE_TYPE_SDMA_XGMI = 0x3 # type: ignore
KFD_IOC_QUEUE_TYPE_SDMA_BY_ENG_ID = 0x4 # type: ignore
KFD_MAX_QUEUE_PERCENTAGE = 100 # type: ignore
KFD_MAX_QUEUE_PRIORITY = 15 # type: ignore
KFD_IOC_CACHE_POLICY_COHERENT = 0 # type: ignore
KFD_IOC_CACHE_POLICY_NONCOHERENT = 1 # type: ignore
NUM_OF_SUPPORTED_GPUS = 7 # type: ignore
MAX_ALLOWED_NUM_POINTS = 100 # type: ignore
MAX_ALLOWED_AW_BUFF_SIZE = 4096 # type: ignore
MAX_ALLOWED_WAC_BUFF_SIZE = 128 # type: ignore
KFD_INVALID_FD = 0xffffffff # type: ignore
KFD_IOC_EVENT_SIGNAL = 0 # type: ignore
KFD_IOC_EVENT_NODECHANGE = 1 # type: ignore
KFD_IOC_EVENT_DEVICESTATECHANGE = 2 # type: ignore
KFD_IOC_EVENT_HW_EXCEPTION = 3 # type: ignore
KFD_IOC_EVENT_SYSTEM_EVENT = 4 # type: ignore
KFD_IOC_EVENT_DEBUG_EVENT = 5 # type: ignore
KFD_IOC_EVENT_PROFILE_EVENT = 6 # type: ignore
KFD_IOC_EVENT_QUEUE_EVENT = 7 # type: ignore
KFD_IOC_EVENT_MEMORY = 8 # type: ignore
KFD_IOC_WAIT_RESULT_COMPLETE = 0 # type: ignore
KFD_IOC_WAIT_RESULT_TIMEOUT = 1 # type: ignore
KFD_IOC_WAIT_RESULT_FAIL = 2 # type: ignore
KFD_SIGNAL_EVENT_LIMIT = 4096 # type: ignore
KFD_HW_EXCEPTION_WHOLE_GPU_RESET = 0 # type: ignore
KFD_HW_EXCEPTION_PER_ENGINE_RESET = 1 # type: ignore
KFD_HW_EXCEPTION_GPU_HANG = 0 # type: ignore
KFD_HW_EXCEPTION_ECC = 1 # type: ignore
KFD_MEM_ERR_NO_RAS = 0 # type: ignore
KFD_MEM_ERR_SRAM_ECC = 1 # type: ignore
KFD_MEM_ERR_POISON_CONSUMED = 2 # type: ignore
KFD_MEM_ERR_GPU_HANG = 3 # type: ignore
KFD_IOC_ALLOC_MEM_FLAGS_VRAM = (1 << 0) # type: ignore
KFD_IOC_ALLOC_MEM_FLAGS_GTT = (1 << 1) # type: ignore
KFD_IOC_ALLOC_MEM_FLAGS_USERPTR = (1 << 2) # type: ignore
KFD_IOC_ALLOC_MEM_FLAGS_DOORBELL = (1 << 3) # type: ignore
KFD_IOC_ALLOC_MEM_FLAGS_MMIO_REMAP = (1 << 4) # type: ignore
KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE = (1 << 31) # type: ignore
KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE = (1 << 30) # type: ignore
KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC = (1 << 29) # type: ignore
KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE = (1 << 28) # type: ignore
KFD_IOC_ALLOC_MEM_FLAGS_AQL_QUEUE_MEM = (1 << 27) # type: ignore
KFD_IOC_ALLOC_MEM_FLAGS_COHERENT = (1 << 26) # type: ignore
KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED = (1 << 25) # type: ignore
KFD_IOC_ALLOC_MEM_FLAGS_EXT_COHERENT = (1 << 24) # type: ignore
KFD_IOC_ALLOC_MEM_FLAGS_CONTIGUOUS = (1 << 23) # type: ignore
KFD_SMI_EVENT_MASK_FROM_INDEX = lambda i: (1 << ((i) - 1)) # type: ignore
KFD_SMI_EVENT_MSG_SIZE = 96 # type: ignore
KFD_IOCTL_SVM_FLAG_HOST_ACCESS = 0x00000001 # type: ignore
KFD_IOCTL_SVM_FLAG_COHERENT = 0x00000002 # type: ignore
KFD_IOCTL_SVM_FLAG_HIVE_LOCAL = 0x00000004 # type: ignore
KFD_IOCTL_SVM_FLAG_GPU_RO = 0x00000008 # type: ignore
KFD_IOCTL_SVM_FLAG_GPU_EXEC = 0x00000010 # type: ignore
KFD_IOCTL_SVM_FLAG_GPU_READ_MOSTLY = 0x00000020 # type: ignore
KFD_IOCTL_SVM_FLAG_GPU_ALWAYS_MAPPED = 0x00000040 # type: ignore
KFD_IOCTL_SVM_FLAG_EXT_COHERENT = 0x00000080 # type: ignore
KFD_EC_MASK = lambda ecode: (1 << (ecode - 1)) # type: ignore
KFD_EC_MASK_QUEUE = (KFD_EC_MASK(EC_QUEUE_WAVE_ABORT) | KFD_EC_MASK(EC_QUEUE_WAVE_TRAP) | KFD_EC_MASK(EC_QUEUE_WAVE_MATH_ERROR) | KFD_EC_MASK(EC_QUEUE_WAVE_ILLEGAL_INSTRUCTION) | KFD_EC_MASK(EC_QUEUE_WAVE_MEMORY_VIOLATION) | KFD_EC_MASK(EC_QUEUE_WAVE_APERTURE_VIOLATION) | KFD_EC_MASK(EC_QUEUE_PACKET_DISPATCH_DIM_INVALID) | KFD_EC_MASK(EC_QUEUE_PACKET_DISPATCH_GROUP_SEGMENT_SIZE_INVALID) | KFD_EC_MASK(EC_QUEUE_PACKET_DISPATCH_CODE_INVALID) | KFD_EC_MASK(EC_QUEUE_PACKET_RESERVED) | KFD_EC_MASK(EC_QUEUE_PACKET_UNSUPPORTED) | KFD_EC_MASK(EC_QUEUE_PACKET_DISPATCH_WORK_GROUP_SIZE_INVALID) | KFD_EC_MASK(EC_QUEUE_PACKET_DISPATCH_REGISTER_INVALID) | KFD_EC_MASK(EC_QUEUE_PACKET_VENDOR_UNSUPPORTED)	| KFD_EC_MASK(EC_QUEUE_PREEMPTION_ERROR)	| KFD_EC_MASK(EC_QUEUE_NEW)) # type: ignore
KFD_EC_MASK_DEVICE = (KFD_EC_MASK(EC_DEVICE_QUEUE_DELETE) | KFD_EC_MASK(EC_DEVICE_RAS_ERROR) | KFD_EC_MASK(EC_DEVICE_FATAL_HALT) | KFD_EC_MASK(EC_DEVICE_MEMORY_VIOLATION) | KFD_EC_MASK(EC_DEVICE_NEW)) # type: ignore
KFD_EC_MASK_PROCESS = (KFD_EC_MASK(EC_PROCESS_RUNTIME) | KFD_EC_MASK(EC_PROCESS_DEVICE_REMOVE)) # type: ignore
KFD_EC_MASK_PACKET = (KFD_EC_MASK(EC_QUEUE_PACKET_DISPATCH_DIM_INVALID) | KFD_EC_MASK(EC_QUEUE_PACKET_DISPATCH_GROUP_SEGMENT_SIZE_INVALID) | KFD_EC_MASK(EC_QUEUE_PACKET_DISPATCH_CODE_INVALID) | KFD_EC_MASK(EC_QUEUE_PACKET_RESERVED) | KFD_EC_MASK(EC_QUEUE_PACKET_UNSUPPORTED) | KFD_EC_MASK(EC_QUEUE_PACKET_DISPATCH_WORK_GROUP_SIZE_INVALID) | KFD_EC_MASK(EC_QUEUE_PACKET_DISPATCH_REGISTER_INVALID) | KFD_EC_MASK(EC_QUEUE_PACKET_VENDOR_UNSUPPORTED)) # type: ignore
KFD_DBG_EC_IS_VALID = lambda ecode: (ecode > EC_NONE and ecode < EC_MAX) # type: ignore
KFD_DBG_EC_TYPE_IS_QUEUE = lambda ecode: (KFD_DBG_EC_IS_VALID(ecode) and not  not (KFD_EC_MASK(ecode) & KFD_EC_MASK_QUEUE)) # type: ignore
KFD_DBG_EC_TYPE_IS_DEVICE = lambda ecode: (KFD_DBG_EC_IS_VALID(ecode) and not  not (KFD_EC_MASK(ecode) & KFD_EC_MASK_DEVICE)) # type: ignore
KFD_DBG_EC_TYPE_IS_PROCESS = lambda ecode: (KFD_DBG_EC_IS_VALID(ecode) and not  not (KFD_EC_MASK(ecode) & KFD_EC_MASK_PROCESS)) # type: ignore
KFD_DBG_EC_TYPE_IS_PACKET = lambda ecode: (KFD_DBG_EC_IS_VALID(ecode) and not  not (KFD_EC_MASK(ecode) & KFD_EC_MASK_PACKET)) # type: ignore
KFD_RUNTIME_ENABLE_MODE_ENABLE_MASK = 1 # type: ignore
KFD_RUNTIME_ENABLE_MODE_TTMP_SAVE_MASK = 2 # type: ignore
KFD_DBG_QUEUE_ERROR_BIT = 30 # type: ignore
KFD_DBG_QUEUE_INVALID_BIT = 31 # type: ignore
KFD_DBG_QUEUE_ERROR_MASK = (1 << KFD_DBG_QUEUE_ERROR_BIT) # type: ignore
KFD_DBG_QUEUE_INVALID_MASK = (1 << KFD_DBG_QUEUE_INVALID_BIT) # type: ignore
KFD_IOCTL_PCS_FLAG_POWER_OF_2 = 0x00000001 # type: ignore
KFD_IOCTL_PCS_QUERY_TYPE_FULL = (1 << 0) # type: ignore
KFD_IOC_PROFILER_VERSION_NUM = 1 # type: ignore
AMDKFD_IOCTL_BASE = 'K' # type: ignore
AMDKFD_IO = lambda nr: _IO(AMDKFD_IOCTL_BASE, nr) # type: ignore
AMDKFD_IOR = lambda nr,type: _IOR(AMDKFD_IOCTL_BASE, nr, type) # type: ignore
AMDKFD_IOW = lambda nr,type: _IOW(AMDKFD_IOCTL_BASE, nr, type) # type: ignore
AMDKFD_IOWR = lambda nr,type: _IOWR(AMDKFD_IOCTL_BASE, nr, type) # type: ignore
AMDKFD_IOC_GET_VERSION = AMDKFD_IOR(0x01, struct_kfd_ioctl_get_version_args) # type: ignore
AMDKFD_IOC_CREATE_QUEUE = AMDKFD_IOWR(0x02, struct_kfd_ioctl_create_queue_args) # type: ignore
AMDKFD_IOC_DESTROY_QUEUE = AMDKFD_IOWR(0x03, struct_kfd_ioctl_destroy_queue_args) # type: ignore
AMDKFD_IOC_SET_MEMORY_POLICY = AMDKFD_IOW(0x04, struct_kfd_ioctl_set_memory_policy_args) # type: ignore
AMDKFD_IOC_GET_CLOCK_COUNTERS = AMDKFD_IOWR(0x05, struct_kfd_ioctl_get_clock_counters_args) # type: ignore
AMDKFD_IOC_GET_PROCESS_APERTURES = AMDKFD_IOR(0x06, struct_kfd_ioctl_get_process_apertures_args) # type: ignore
AMDKFD_IOC_UPDATE_QUEUE = AMDKFD_IOW(0x07, struct_kfd_ioctl_update_queue_args) # type: ignore
AMDKFD_IOC_CREATE_EVENT = AMDKFD_IOWR(0x08, struct_kfd_ioctl_create_event_args) # type: ignore
AMDKFD_IOC_DESTROY_EVENT = AMDKFD_IOW(0x09, struct_kfd_ioctl_destroy_event_args) # type: ignore
AMDKFD_IOC_SET_EVENT = AMDKFD_IOW(0x0A, struct_kfd_ioctl_set_event_args) # type: ignore
AMDKFD_IOC_RESET_EVENT = AMDKFD_IOW(0x0B, struct_kfd_ioctl_reset_event_args) # type: ignore
AMDKFD_IOC_WAIT_EVENTS = AMDKFD_IOWR(0x0C, struct_kfd_ioctl_wait_events_args) # type: ignore
AMDKFD_IOC_DBG_REGISTER_DEPRECATED = AMDKFD_IOW(0x0D, struct_kfd_ioctl_dbg_register_args) # type: ignore
AMDKFD_IOC_DBG_UNREGISTER_DEPRECATED = AMDKFD_IOW(0x0E, struct_kfd_ioctl_dbg_unregister_args) # type: ignore
AMDKFD_IOC_DBG_ADDRESS_WATCH_DEPRECATED = AMDKFD_IOW(0x0F, struct_kfd_ioctl_dbg_address_watch_args) # type: ignore
AMDKFD_IOC_DBG_WAVE_CONTROL_DEPRECATED = AMDKFD_IOW(0x10, struct_kfd_ioctl_dbg_wave_control_args) # type: ignore
AMDKFD_IOC_SET_SCRATCH_BACKING_VA = AMDKFD_IOWR(0x11, struct_kfd_ioctl_set_scratch_backing_va_args) # type: ignore
AMDKFD_IOC_GET_TILE_CONFIG = AMDKFD_IOWR(0x12, struct_kfd_ioctl_get_tile_config_args) # type: ignore
AMDKFD_IOC_SET_TRAP_HANDLER = AMDKFD_IOW(0x13, struct_kfd_ioctl_set_trap_handler_args) # type: ignore
AMDKFD_IOC_GET_PROCESS_APERTURES_NEW = AMDKFD_IOWR(0x14, struct_kfd_ioctl_get_process_apertures_new_args) # type: ignore
AMDKFD_IOC_ACQUIRE_VM = AMDKFD_IOW(0x15, struct_kfd_ioctl_acquire_vm_args) # type: ignore
AMDKFD_IOC_ALLOC_MEMORY_OF_GPU = AMDKFD_IOWR(0x16, struct_kfd_ioctl_alloc_memory_of_gpu_args) # type: ignore
AMDKFD_IOC_FREE_MEMORY_OF_GPU = AMDKFD_IOW(0x17, struct_kfd_ioctl_free_memory_of_gpu_args) # type: ignore
AMDKFD_IOC_MAP_MEMORY_TO_GPU = AMDKFD_IOWR(0x18, struct_kfd_ioctl_map_memory_to_gpu_args) # type: ignore
AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU = AMDKFD_IOWR(0x19, struct_kfd_ioctl_unmap_memory_from_gpu_args) # type: ignore
AMDKFD_IOC_SET_CU_MASK = AMDKFD_IOW(0x1A, struct_kfd_ioctl_set_cu_mask_args) # type: ignore
AMDKFD_IOC_GET_QUEUE_WAVE_STATE = AMDKFD_IOWR(0x1B, struct_kfd_ioctl_get_queue_wave_state_args) # type: ignore
AMDKFD_IOC_GET_DMABUF_INFO = AMDKFD_IOWR(0x1C, struct_kfd_ioctl_get_dmabuf_info_args) # type: ignore
AMDKFD_IOC_IMPORT_DMABUF = AMDKFD_IOWR(0x1D, struct_kfd_ioctl_import_dmabuf_args) # type: ignore
AMDKFD_IOC_ALLOC_QUEUE_GWS = AMDKFD_IOWR(0x1E, struct_kfd_ioctl_alloc_queue_gws_args) # type: ignore
AMDKFD_IOC_SMI_EVENTS = AMDKFD_IOWR(0x1F, struct_kfd_ioctl_smi_events_args) # type: ignore
AMDKFD_IOC_SVM = AMDKFD_IOWR(0x20, struct_kfd_ioctl_svm_args) # type: ignore
AMDKFD_IOC_SET_XNACK_MODE = AMDKFD_IOWR(0x21, struct_kfd_ioctl_set_xnack_mode_args) # type: ignore
AMDKFD_IOC_CRIU_OP = AMDKFD_IOWR(0x22, struct_kfd_ioctl_criu_args) # type: ignore
AMDKFD_IOC_AVAILABLE_MEMORY = AMDKFD_IOWR(0x23, struct_kfd_ioctl_get_available_memory_args) # type: ignore
AMDKFD_IOC_EXPORT_DMABUF = AMDKFD_IOWR(0x24, struct_kfd_ioctl_export_dmabuf_args) # type: ignore
AMDKFD_IOC_RUNTIME_ENABLE = AMDKFD_IOWR(0x25, struct_kfd_ioctl_runtime_enable_args) # type: ignore
AMDKFD_IOC_DBG_TRAP = AMDKFD_IOWR(0x26, struct_kfd_ioctl_dbg_trap_args) # type: ignore
AMDKFD_COMMAND_START = 0x01 # type: ignore
AMDKFD_COMMAND_END = 0x27 # type: ignore
AMDKFD_IOC_IPC_IMPORT_HANDLE = AMDKFD_IOWR(0x80, struct_kfd_ioctl_ipc_import_handle_args) # type: ignore
AMDKFD_IOC_IPC_EXPORT_HANDLE = AMDKFD_IOWR(0x81, struct_kfd_ioctl_ipc_export_handle_args) # type: ignore
AMDKFD_IOC_DBG_TRAP_DEPRECATED = AMDKFD_IOWR(0x82, struct_kfd_ioctl_dbg_trap_args_deprecated) # type: ignore
AMDKFD_IOC_CROSS_MEMORY_COPY_DEPRECATED = AMDKFD_IOWR(0x83, struct_kfd_ioctl_cross_memory_copy_deprecated_args) # type: ignore
AMDKFD_IOC_RLC_SPM = AMDKFD_IOWR(0x84, struct_kfd_ioctl_spm_args) # type: ignore
AMDKFD_IOC_PC_SAMPLE = AMDKFD_IOWR(0x85, struct_kfd_ioctl_pc_sample_args) # type: ignore
AMDKFD_IOC_PROFILER = AMDKFD_IOWR(0x86, struct_kfd_ioctl_profiler_args) # type: ignore
AMDKFD_COMMAND_START_2 = 0x80 # type: ignore
AMDKFD_COMMAND_END_2 = 0x87 # type: ignore