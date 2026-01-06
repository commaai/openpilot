# mypy: ignore-errors
import ctypes
from tinygrad.runtime.support.c import DLL, Struct, CEnum, _IO, _IOW, _IOR, _IOWR
class struct_kfd_ioctl_get_version_args(Struct): pass
__u32 = ctypes.c_uint32
struct_kfd_ioctl_get_version_args._fields_ = [
  ('major_version', ctypes.c_uint32),
  ('minor_version', ctypes.c_uint32),
]
class struct_kfd_ioctl_create_queue_args(Struct): pass
__u64 = ctypes.c_uint64
struct_kfd_ioctl_create_queue_args._fields_ = [
  ('ring_base_address', ctypes.c_uint64),
  ('write_pointer_address', ctypes.c_uint64),
  ('read_pointer_address', ctypes.c_uint64),
  ('doorbell_offset', ctypes.c_uint64),
  ('ring_size', ctypes.c_uint32),
  ('gpu_id', ctypes.c_uint32),
  ('queue_type', ctypes.c_uint32),
  ('queue_percentage', ctypes.c_uint32),
  ('queue_priority', ctypes.c_uint32),
  ('queue_id', ctypes.c_uint32),
  ('eop_buffer_address', ctypes.c_uint64),
  ('eop_buffer_size', ctypes.c_uint64),
  ('ctx_save_restore_address', ctypes.c_uint64),
  ('ctx_save_restore_size', ctypes.c_uint32),
  ('ctl_stack_size', ctypes.c_uint32),
]
class struct_kfd_ioctl_destroy_queue_args(Struct): pass
struct_kfd_ioctl_destroy_queue_args._fields_ = [
  ('queue_id', ctypes.c_uint32),
  ('pad', ctypes.c_uint32),
]
class struct_kfd_ioctl_update_queue_args(Struct): pass
struct_kfd_ioctl_update_queue_args._fields_ = [
  ('ring_base_address', ctypes.c_uint64),
  ('queue_id', ctypes.c_uint32),
  ('ring_size', ctypes.c_uint32),
  ('queue_percentage', ctypes.c_uint32),
  ('queue_priority', ctypes.c_uint32),
]
class struct_kfd_ioctl_set_cu_mask_args(Struct): pass
struct_kfd_ioctl_set_cu_mask_args._fields_ = [
  ('queue_id', ctypes.c_uint32),
  ('num_cu_mask', ctypes.c_uint32),
  ('cu_mask_ptr', ctypes.c_uint64),
]
class struct_kfd_ioctl_get_queue_wave_state_args(Struct): pass
struct_kfd_ioctl_get_queue_wave_state_args._fields_ = [
  ('ctl_stack_address', ctypes.c_uint64),
  ('ctl_stack_used_size', ctypes.c_uint32),
  ('save_area_used_size', ctypes.c_uint32),
  ('queue_id', ctypes.c_uint32),
  ('pad', ctypes.c_uint32),
]
class struct_kfd_ioctl_get_available_memory_args(Struct): pass
struct_kfd_ioctl_get_available_memory_args._fields_ = [
  ('available', ctypes.c_uint64),
  ('gpu_id', ctypes.c_uint32),
  ('pad', ctypes.c_uint32),
]
class struct_kfd_dbg_device_info_entry(Struct): pass
struct_kfd_dbg_device_info_entry._fields_ = [
  ('exception_status', ctypes.c_uint64),
  ('lds_base', ctypes.c_uint64),
  ('lds_limit', ctypes.c_uint64),
  ('scratch_base', ctypes.c_uint64),
  ('scratch_limit', ctypes.c_uint64),
  ('gpuvm_base', ctypes.c_uint64),
  ('gpuvm_limit', ctypes.c_uint64),
  ('gpu_id', ctypes.c_uint32),
  ('location_id', ctypes.c_uint32),
  ('vendor_id', ctypes.c_uint32),
  ('device_id', ctypes.c_uint32),
  ('revision_id', ctypes.c_uint32),
  ('subsystem_vendor_id', ctypes.c_uint32),
  ('subsystem_device_id', ctypes.c_uint32),
  ('fw_version', ctypes.c_uint32),
  ('gfx_target_version', ctypes.c_uint32),
  ('simd_count', ctypes.c_uint32),
  ('max_waves_per_simd', ctypes.c_uint32),
  ('array_count', ctypes.c_uint32),
  ('simd_arrays_per_engine', ctypes.c_uint32),
  ('num_xcc', ctypes.c_uint32),
  ('capability', ctypes.c_uint32),
  ('debug_prop', ctypes.c_uint32),
]
class struct_kfd_ioctl_set_memory_policy_args(Struct): pass
struct_kfd_ioctl_set_memory_policy_args._fields_ = [
  ('alternate_aperture_base', ctypes.c_uint64),
  ('alternate_aperture_size', ctypes.c_uint64),
  ('gpu_id', ctypes.c_uint32),
  ('default_policy', ctypes.c_uint32),
  ('alternate_policy', ctypes.c_uint32),
  ('pad', ctypes.c_uint32),
]
class struct_kfd_ioctl_get_clock_counters_args(Struct): pass
struct_kfd_ioctl_get_clock_counters_args._fields_ = [
  ('gpu_clock_counter', ctypes.c_uint64),
  ('cpu_clock_counter', ctypes.c_uint64),
  ('system_clock_counter', ctypes.c_uint64),
  ('system_clock_freq', ctypes.c_uint64),
  ('gpu_id', ctypes.c_uint32),
  ('pad', ctypes.c_uint32),
]
class struct_kfd_process_device_apertures(Struct): pass
struct_kfd_process_device_apertures._fields_ = [
  ('lds_base', ctypes.c_uint64),
  ('lds_limit', ctypes.c_uint64),
  ('scratch_base', ctypes.c_uint64),
  ('scratch_limit', ctypes.c_uint64),
  ('gpuvm_base', ctypes.c_uint64),
  ('gpuvm_limit', ctypes.c_uint64),
  ('gpu_id', ctypes.c_uint32),
  ('pad', ctypes.c_uint32),
]
class struct_kfd_ioctl_get_process_apertures_args(Struct): pass
struct_kfd_ioctl_get_process_apertures_args._fields_ = [
  ('process_apertures', (struct_kfd_process_device_apertures * 7)),
  ('num_of_nodes', ctypes.c_uint32),
  ('pad', ctypes.c_uint32),
]
class struct_kfd_ioctl_get_process_apertures_new_args(Struct): pass
struct_kfd_ioctl_get_process_apertures_new_args._fields_ = [
  ('kfd_process_device_apertures_ptr', ctypes.c_uint64),
  ('num_of_nodes', ctypes.c_uint32),
  ('pad', ctypes.c_uint32),
]
class struct_kfd_ioctl_dbg_register_args(Struct): pass
struct_kfd_ioctl_dbg_register_args._fields_ = [
  ('gpu_id', ctypes.c_uint32),
  ('pad', ctypes.c_uint32),
]
class struct_kfd_ioctl_dbg_unregister_args(Struct): pass
struct_kfd_ioctl_dbg_unregister_args._fields_ = [
  ('gpu_id', ctypes.c_uint32),
  ('pad', ctypes.c_uint32),
]
class struct_kfd_ioctl_dbg_address_watch_args(Struct): pass
struct_kfd_ioctl_dbg_address_watch_args._fields_ = [
  ('content_ptr', ctypes.c_uint64),
  ('gpu_id', ctypes.c_uint32),
  ('buf_size_in_bytes', ctypes.c_uint32),
]
class struct_kfd_ioctl_dbg_wave_control_args(Struct): pass
struct_kfd_ioctl_dbg_wave_control_args._fields_ = [
  ('content_ptr', ctypes.c_uint64),
  ('gpu_id', ctypes.c_uint32),
  ('buf_size_in_bytes', ctypes.c_uint32),
]
class struct_kfd_ioctl_create_event_args(Struct): pass
struct_kfd_ioctl_create_event_args._fields_ = [
  ('event_page_offset', ctypes.c_uint64),
  ('event_trigger_data', ctypes.c_uint32),
  ('event_type', ctypes.c_uint32),
  ('auto_reset', ctypes.c_uint32),
  ('node_id', ctypes.c_uint32),
  ('event_id', ctypes.c_uint32),
  ('event_slot_index', ctypes.c_uint32),
]
class struct_kfd_ioctl_destroy_event_args(Struct): pass
struct_kfd_ioctl_destroy_event_args._fields_ = [
  ('event_id', ctypes.c_uint32),
  ('pad', ctypes.c_uint32),
]
class struct_kfd_ioctl_set_event_args(Struct): pass
struct_kfd_ioctl_set_event_args._fields_ = [
  ('event_id', ctypes.c_uint32),
  ('pad', ctypes.c_uint32),
]
class struct_kfd_ioctl_reset_event_args(Struct): pass
struct_kfd_ioctl_reset_event_args._fields_ = [
  ('event_id', ctypes.c_uint32),
  ('pad', ctypes.c_uint32),
]
class struct_kfd_memory_exception_failure(Struct): pass
struct_kfd_memory_exception_failure._fields_ = [
  ('NotPresent', ctypes.c_uint32),
  ('ReadOnly', ctypes.c_uint32),
  ('NoExecute', ctypes.c_uint32),
  ('imprecise', ctypes.c_uint32),
]
class struct_kfd_hsa_memory_exception_data(Struct): pass
struct_kfd_hsa_memory_exception_data._fields_ = [
  ('failure', struct_kfd_memory_exception_failure),
  ('va', ctypes.c_uint64),
  ('gpu_id', ctypes.c_uint32),
  ('ErrorType', ctypes.c_uint32),
]
class struct_kfd_hsa_hw_exception_data(Struct): pass
struct_kfd_hsa_hw_exception_data._fields_ = [
  ('reset_type', ctypes.c_uint32),
  ('reset_cause', ctypes.c_uint32),
  ('memory_lost', ctypes.c_uint32),
  ('gpu_id', ctypes.c_uint32),
]
class struct_kfd_hsa_signal_event_data(Struct): pass
struct_kfd_hsa_signal_event_data._fields_ = [
  ('last_event_age', ctypes.c_uint64),
]
class struct_kfd_event_data(Struct): pass
class struct_kfd_event_data_0(ctypes.Union): pass
struct_kfd_event_data_0._fields_ = [
  ('memory_exception_data', struct_kfd_hsa_memory_exception_data),
  ('hw_exception_data', struct_kfd_hsa_hw_exception_data),
  ('signal_event_data', struct_kfd_hsa_signal_event_data),
]
struct_kfd_event_data._anonymous_ = ['_0']
struct_kfd_event_data._fields_ = [
  ('_0', struct_kfd_event_data_0),
  ('kfd_event_data_ext', ctypes.c_uint64),
  ('event_id', ctypes.c_uint32),
  ('pad', ctypes.c_uint32),
]
class struct_kfd_ioctl_wait_events_args(Struct): pass
struct_kfd_ioctl_wait_events_args._fields_ = [
  ('events_ptr', ctypes.c_uint64),
  ('num_events', ctypes.c_uint32),
  ('wait_for_all', ctypes.c_uint32),
  ('timeout', ctypes.c_uint32),
  ('wait_result', ctypes.c_uint32),
]
class struct_kfd_ioctl_set_scratch_backing_va_args(Struct): pass
struct_kfd_ioctl_set_scratch_backing_va_args._fields_ = [
  ('va_addr', ctypes.c_uint64),
  ('gpu_id', ctypes.c_uint32),
  ('pad', ctypes.c_uint32),
]
class struct_kfd_ioctl_get_tile_config_args(Struct): pass
struct_kfd_ioctl_get_tile_config_args._fields_ = [
  ('tile_config_ptr', ctypes.c_uint64),
  ('macro_tile_config_ptr', ctypes.c_uint64),
  ('num_tile_configs', ctypes.c_uint32),
  ('num_macro_tile_configs', ctypes.c_uint32),
  ('gpu_id', ctypes.c_uint32),
  ('gb_addr_config', ctypes.c_uint32),
  ('num_banks', ctypes.c_uint32),
  ('num_ranks', ctypes.c_uint32),
]
class struct_kfd_ioctl_set_trap_handler_args(Struct): pass
struct_kfd_ioctl_set_trap_handler_args._fields_ = [
  ('tba_addr', ctypes.c_uint64),
  ('tma_addr', ctypes.c_uint64),
  ('gpu_id', ctypes.c_uint32),
  ('pad', ctypes.c_uint32),
]
class struct_kfd_ioctl_acquire_vm_args(Struct): pass
struct_kfd_ioctl_acquire_vm_args._fields_ = [
  ('drm_fd', ctypes.c_uint32),
  ('gpu_id', ctypes.c_uint32),
]
class struct_kfd_ioctl_alloc_memory_of_gpu_args(Struct): pass
struct_kfd_ioctl_alloc_memory_of_gpu_args._fields_ = [
  ('va_addr', ctypes.c_uint64),
  ('size', ctypes.c_uint64),
  ('handle', ctypes.c_uint64),
  ('mmap_offset', ctypes.c_uint64),
  ('gpu_id', ctypes.c_uint32),
  ('flags', ctypes.c_uint32),
]
class struct_kfd_ioctl_free_memory_of_gpu_args(Struct): pass
struct_kfd_ioctl_free_memory_of_gpu_args._fields_ = [
  ('handle', ctypes.c_uint64),
]
class struct_kfd_ioctl_map_memory_to_gpu_args(Struct): pass
struct_kfd_ioctl_map_memory_to_gpu_args._fields_ = [
  ('handle', ctypes.c_uint64),
  ('device_ids_array_ptr', ctypes.c_uint64),
  ('n_devices', ctypes.c_uint32),
  ('n_success', ctypes.c_uint32),
]
class struct_kfd_ioctl_unmap_memory_from_gpu_args(Struct): pass
struct_kfd_ioctl_unmap_memory_from_gpu_args._fields_ = [
  ('handle', ctypes.c_uint64),
  ('device_ids_array_ptr', ctypes.c_uint64),
  ('n_devices', ctypes.c_uint32),
  ('n_success', ctypes.c_uint32),
]
class struct_kfd_ioctl_alloc_queue_gws_args(Struct): pass
struct_kfd_ioctl_alloc_queue_gws_args._fields_ = [
  ('queue_id', ctypes.c_uint32),
  ('num_gws', ctypes.c_uint32),
  ('first_gws', ctypes.c_uint32),
  ('pad', ctypes.c_uint32),
]
class struct_kfd_ioctl_get_dmabuf_info_args(Struct): pass
struct_kfd_ioctl_get_dmabuf_info_args._fields_ = [
  ('size', ctypes.c_uint64),
  ('metadata_ptr', ctypes.c_uint64),
  ('metadata_size', ctypes.c_uint32),
  ('gpu_id', ctypes.c_uint32),
  ('flags', ctypes.c_uint32),
  ('dmabuf_fd', ctypes.c_uint32),
]
class struct_kfd_ioctl_import_dmabuf_args(Struct): pass
struct_kfd_ioctl_import_dmabuf_args._fields_ = [
  ('va_addr', ctypes.c_uint64),
  ('handle', ctypes.c_uint64),
  ('gpu_id', ctypes.c_uint32),
  ('dmabuf_fd', ctypes.c_uint32),
]
class struct_kfd_ioctl_export_dmabuf_args(Struct): pass
struct_kfd_ioctl_export_dmabuf_args._fields_ = [
  ('handle', ctypes.c_uint64),
  ('flags', ctypes.c_uint32),
  ('dmabuf_fd', ctypes.c_uint32),
]
enum_kfd_smi_event = CEnum(ctypes.c_uint32)
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

enum_KFD_MIGRATE_TRIGGERS = CEnum(ctypes.c_uint32)
KFD_MIGRATE_TRIGGER_PREFETCH = enum_KFD_MIGRATE_TRIGGERS.define('KFD_MIGRATE_TRIGGER_PREFETCH', 0)
KFD_MIGRATE_TRIGGER_PAGEFAULT_GPU = enum_KFD_MIGRATE_TRIGGERS.define('KFD_MIGRATE_TRIGGER_PAGEFAULT_GPU', 1)
KFD_MIGRATE_TRIGGER_PAGEFAULT_CPU = enum_KFD_MIGRATE_TRIGGERS.define('KFD_MIGRATE_TRIGGER_PAGEFAULT_CPU', 2)
KFD_MIGRATE_TRIGGER_TTM_EVICTION = enum_KFD_MIGRATE_TRIGGERS.define('KFD_MIGRATE_TRIGGER_TTM_EVICTION', 3)

enum_KFD_QUEUE_EVICTION_TRIGGERS = CEnum(ctypes.c_uint32)
KFD_QUEUE_EVICTION_TRIGGER_SVM = enum_KFD_QUEUE_EVICTION_TRIGGERS.define('KFD_QUEUE_EVICTION_TRIGGER_SVM', 0)
KFD_QUEUE_EVICTION_TRIGGER_USERPTR = enum_KFD_QUEUE_EVICTION_TRIGGERS.define('KFD_QUEUE_EVICTION_TRIGGER_USERPTR', 1)
KFD_QUEUE_EVICTION_TRIGGER_TTM = enum_KFD_QUEUE_EVICTION_TRIGGERS.define('KFD_QUEUE_EVICTION_TRIGGER_TTM', 2)
KFD_QUEUE_EVICTION_TRIGGER_SUSPEND = enum_KFD_QUEUE_EVICTION_TRIGGERS.define('KFD_QUEUE_EVICTION_TRIGGER_SUSPEND', 3)
KFD_QUEUE_EVICTION_CRIU_CHECKPOINT = enum_KFD_QUEUE_EVICTION_TRIGGERS.define('KFD_QUEUE_EVICTION_CRIU_CHECKPOINT', 4)
KFD_QUEUE_EVICTION_CRIU_RESTORE = enum_KFD_QUEUE_EVICTION_TRIGGERS.define('KFD_QUEUE_EVICTION_CRIU_RESTORE', 5)

enum_KFD_SVM_UNMAP_TRIGGERS = CEnum(ctypes.c_uint32)
KFD_SVM_UNMAP_TRIGGER_MMU_NOTIFY = enum_KFD_SVM_UNMAP_TRIGGERS.define('KFD_SVM_UNMAP_TRIGGER_MMU_NOTIFY', 0)
KFD_SVM_UNMAP_TRIGGER_MMU_NOTIFY_MIGRATE = enum_KFD_SVM_UNMAP_TRIGGERS.define('KFD_SVM_UNMAP_TRIGGER_MMU_NOTIFY_MIGRATE', 1)
KFD_SVM_UNMAP_TRIGGER_UNMAP_FROM_CPU = enum_KFD_SVM_UNMAP_TRIGGERS.define('KFD_SVM_UNMAP_TRIGGER_UNMAP_FROM_CPU', 2)

class struct_kfd_ioctl_smi_events_args(Struct): pass
struct_kfd_ioctl_smi_events_args._fields_ = [
  ('gpuid', ctypes.c_uint32),
  ('anon_fd', ctypes.c_uint32),
]
enum_kfd_criu_op = CEnum(ctypes.c_uint32)
KFD_CRIU_OP_PROCESS_INFO = enum_kfd_criu_op.define('KFD_CRIU_OP_PROCESS_INFO', 0)
KFD_CRIU_OP_CHECKPOINT = enum_kfd_criu_op.define('KFD_CRIU_OP_CHECKPOINT', 1)
KFD_CRIU_OP_UNPAUSE = enum_kfd_criu_op.define('KFD_CRIU_OP_UNPAUSE', 2)
KFD_CRIU_OP_RESTORE = enum_kfd_criu_op.define('KFD_CRIU_OP_RESTORE', 3)
KFD_CRIU_OP_RESUME = enum_kfd_criu_op.define('KFD_CRIU_OP_RESUME', 4)

class struct_kfd_ioctl_criu_args(Struct): pass
struct_kfd_ioctl_criu_args._fields_ = [
  ('devices', ctypes.c_uint64),
  ('bos', ctypes.c_uint64),
  ('priv_data', ctypes.c_uint64),
  ('priv_data_size', ctypes.c_uint64),
  ('num_devices', ctypes.c_uint32),
  ('num_bos', ctypes.c_uint32),
  ('num_objects', ctypes.c_uint32),
  ('pid', ctypes.c_uint32),
  ('op', ctypes.c_uint32),
]
class struct_kfd_criu_device_bucket(Struct): pass
struct_kfd_criu_device_bucket._fields_ = [
  ('user_gpu_id', ctypes.c_uint32),
  ('actual_gpu_id', ctypes.c_uint32),
  ('drm_fd', ctypes.c_uint32),
  ('pad', ctypes.c_uint32),
]
class struct_kfd_criu_bo_bucket(Struct): pass
struct_kfd_criu_bo_bucket._fields_ = [
  ('addr', ctypes.c_uint64),
  ('size', ctypes.c_uint64),
  ('offset', ctypes.c_uint64),
  ('restored_offset', ctypes.c_uint64),
  ('gpu_id', ctypes.c_uint32),
  ('alloc_flags', ctypes.c_uint32),
  ('dmabuf_fd', ctypes.c_uint32),
  ('pad', ctypes.c_uint32),
]
enum_kfd_mmio_remap = CEnum(ctypes.c_uint32)
KFD_MMIO_REMAP_HDP_MEM_FLUSH_CNTL = enum_kfd_mmio_remap.define('KFD_MMIO_REMAP_HDP_MEM_FLUSH_CNTL', 0)
KFD_MMIO_REMAP_HDP_REG_FLUSH_CNTL = enum_kfd_mmio_remap.define('KFD_MMIO_REMAP_HDP_REG_FLUSH_CNTL', 4)

enum_kfd_ioctl_svm_op = CEnum(ctypes.c_uint32)
KFD_IOCTL_SVM_OP_SET_ATTR = enum_kfd_ioctl_svm_op.define('KFD_IOCTL_SVM_OP_SET_ATTR', 0)
KFD_IOCTL_SVM_OP_GET_ATTR = enum_kfd_ioctl_svm_op.define('KFD_IOCTL_SVM_OP_GET_ATTR', 1)

enum_kfd_ioctl_svm_location = CEnum(ctypes.c_uint32)
KFD_IOCTL_SVM_LOCATION_SYSMEM = enum_kfd_ioctl_svm_location.define('KFD_IOCTL_SVM_LOCATION_SYSMEM', 0)
KFD_IOCTL_SVM_LOCATION_UNDEFINED = enum_kfd_ioctl_svm_location.define('KFD_IOCTL_SVM_LOCATION_UNDEFINED', 4294967295)

enum_kfd_ioctl_svm_attr_type = CEnum(ctypes.c_uint32)
KFD_IOCTL_SVM_ATTR_PREFERRED_LOC = enum_kfd_ioctl_svm_attr_type.define('KFD_IOCTL_SVM_ATTR_PREFERRED_LOC', 0)
KFD_IOCTL_SVM_ATTR_PREFETCH_LOC = enum_kfd_ioctl_svm_attr_type.define('KFD_IOCTL_SVM_ATTR_PREFETCH_LOC', 1)
KFD_IOCTL_SVM_ATTR_ACCESS = enum_kfd_ioctl_svm_attr_type.define('KFD_IOCTL_SVM_ATTR_ACCESS', 2)
KFD_IOCTL_SVM_ATTR_ACCESS_IN_PLACE = enum_kfd_ioctl_svm_attr_type.define('KFD_IOCTL_SVM_ATTR_ACCESS_IN_PLACE', 3)
KFD_IOCTL_SVM_ATTR_NO_ACCESS = enum_kfd_ioctl_svm_attr_type.define('KFD_IOCTL_SVM_ATTR_NO_ACCESS', 4)
KFD_IOCTL_SVM_ATTR_SET_FLAGS = enum_kfd_ioctl_svm_attr_type.define('KFD_IOCTL_SVM_ATTR_SET_FLAGS', 5)
KFD_IOCTL_SVM_ATTR_CLR_FLAGS = enum_kfd_ioctl_svm_attr_type.define('KFD_IOCTL_SVM_ATTR_CLR_FLAGS', 6)
KFD_IOCTL_SVM_ATTR_GRANULARITY = enum_kfd_ioctl_svm_attr_type.define('KFD_IOCTL_SVM_ATTR_GRANULARITY', 7)

class struct_kfd_ioctl_svm_attribute(Struct): pass
struct_kfd_ioctl_svm_attribute._fields_ = [
  ('type', ctypes.c_uint32),
  ('value', ctypes.c_uint32),
]
class struct_kfd_ioctl_svm_args(Struct): pass
struct_kfd_ioctl_svm_args._fields_ = [
  ('start_addr', ctypes.c_uint64),
  ('size', ctypes.c_uint64),
  ('op', ctypes.c_uint32),
  ('nattr', ctypes.c_uint32),
  ('attrs', (struct_kfd_ioctl_svm_attribute * 0)),
]
class struct_kfd_ioctl_set_xnack_mode_args(Struct): pass
__s32 = ctypes.c_int32
struct_kfd_ioctl_set_xnack_mode_args._fields_ = [
  ('xnack_enabled', ctypes.c_int32),
]
enum_kfd_dbg_trap_override_mode = CEnum(ctypes.c_uint32)
KFD_DBG_TRAP_OVERRIDE_OR = enum_kfd_dbg_trap_override_mode.define('KFD_DBG_TRAP_OVERRIDE_OR', 0)
KFD_DBG_TRAP_OVERRIDE_REPLACE = enum_kfd_dbg_trap_override_mode.define('KFD_DBG_TRAP_OVERRIDE_REPLACE', 1)

enum_kfd_dbg_trap_mask = CEnum(ctypes.c_int32)
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

enum_kfd_dbg_trap_wave_launch_mode = CEnum(ctypes.c_uint32)
KFD_DBG_TRAP_WAVE_LAUNCH_MODE_NORMAL = enum_kfd_dbg_trap_wave_launch_mode.define('KFD_DBG_TRAP_WAVE_LAUNCH_MODE_NORMAL', 0)
KFD_DBG_TRAP_WAVE_LAUNCH_MODE_HALT = enum_kfd_dbg_trap_wave_launch_mode.define('KFD_DBG_TRAP_WAVE_LAUNCH_MODE_HALT', 1)
KFD_DBG_TRAP_WAVE_LAUNCH_MODE_DEBUG = enum_kfd_dbg_trap_wave_launch_mode.define('KFD_DBG_TRAP_WAVE_LAUNCH_MODE_DEBUG', 3)

enum_kfd_dbg_trap_address_watch_mode = CEnum(ctypes.c_uint32)
KFD_DBG_TRAP_ADDRESS_WATCH_MODE_READ = enum_kfd_dbg_trap_address_watch_mode.define('KFD_DBG_TRAP_ADDRESS_WATCH_MODE_READ', 0)
KFD_DBG_TRAP_ADDRESS_WATCH_MODE_NONREAD = enum_kfd_dbg_trap_address_watch_mode.define('KFD_DBG_TRAP_ADDRESS_WATCH_MODE_NONREAD', 1)
KFD_DBG_TRAP_ADDRESS_WATCH_MODE_ATOMIC = enum_kfd_dbg_trap_address_watch_mode.define('KFD_DBG_TRAP_ADDRESS_WATCH_MODE_ATOMIC', 2)
KFD_DBG_TRAP_ADDRESS_WATCH_MODE_ALL = enum_kfd_dbg_trap_address_watch_mode.define('KFD_DBG_TRAP_ADDRESS_WATCH_MODE_ALL', 3)

enum_kfd_dbg_trap_flags = CEnum(ctypes.c_uint32)
KFD_DBG_TRAP_FLAG_SINGLE_MEM_OP = enum_kfd_dbg_trap_flags.define('KFD_DBG_TRAP_FLAG_SINGLE_MEM_OP', 1)

enum_kfd_dbg_trap_exception_code = CEnum(ctypes.c_uint32)
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

enum_kfd_dbg_runtime_state = CEnum(ctypes.c_uint32)
DEBUG_RUNTIME_STATE_DISABLED = enum_kfd_dbg_runtime_state.define('DEBUG_RUNTIME_STATE_DISABLED', 0)
DEBUG_RUNTIME_STATE_ENABLED = enum_kfd_dbg_runtime_state.define('DEBUG_RUNTIME_STATE_ENABLED', 1)
DEBUG_RUNTIME_STATE_ENABLED_BUSY = enum_kfd_dbg_runtime_state.define('DEBUG_RUNTIME_STATE_ENABLED_BUSY', 2)
DEBUG_RUNTIME_STATE_ENABLED_ERROR = enum_kfd_dbg_runtime_state.define('DEBUG_RUNTIME_STATE_ENABLED_ERROR', 3)

class struct_kfd_runtime_info(Struct): pass
struct_kfd_runtime_info._fields_ = [
  ('r_debug', ctypes.c_uint64),
  ('runtime_state', ctypes.c_uint32),
  ('ttmp_setup', ctypes.c_uint32),
]
class struct_kfd_ioctl_runtime_enable_args(Struct): pass
struct_kfd_ioctl_runtime_enable_args._fields_ = [
  ('r_debug', ctypes.c_uint64),
  ('mode_mask', ctypes.c_uint32),
  ('capabilities_mask', ctypes.c_uint32),
]
class struct_kfd_queue_snapshot_entry(Struct): pass
struct_kfd_queue_snapshot_entry._fields_ = [
  ('exception_status', ctypes.c_uint64),
  ('ring_base_address', ctypes.c_uint64),
  ('write_pointer_address', ctypes.c_uint64),
  ('read_pointer_address', ctypes.c_uint64),
  ('ctx_save_restore_address', ctypes.c_uint64),
  ('queue_id', ctypes.c_uint32),
  ('gpu_id', ctypes.c_uint32),
  ('ring_size', ctypes.c_uint32),
  ('queue_type', ctypes.c_uint32),
  ('ctx_save_restore_area_size', ctypes.c_uint32),
  ('reserved', ctypes.c_uint32),
]
class struct_kfd_context_save_area_header(Struct): pass
class struct_kfd_context_save_area_header_wave_state(Struct): pass
struct_kfd_context_save_area_header_wave_state._fields_ = [
  ('control_stack_offset', ctypes.c_uint32),
  ('control_stack_size', ctypes.c_uint32),
  ('wave_state_offset', ctypes.c_uint32),
  ('wave_state_size', ctypes.c_uint32),
]
struct_kfd_context_save_area_header._fields_ = [
  ('wave_state', struct_kfd_context_save_area_header_wave_state),
  ('debug_offset', ctypes.c_uint32),
  ('debug_size', ctypes.c_uint32),
  ('err_payload_addr', ctypes.c_uint64),
  ('err_event_id', ctypes.c_uint32),
  ('reserved1', ctypes.c_uint32),
]
enum_kfd_dbg_trap_operations = CEnum(ctypes.c_uint32)
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

class struct_kfd_ioctl_dbg_trap_enable_args(Struct): pass
struct_kfd_ioctl_dbg_trap_enable_args._fields_ = [
  ('exception_mask', ctypes.c_uint64),
  ('rinfo_ptr', ctypes.c_uint64),
  ('rinfo_size', ctypes.c_uint32),
  ('dbg_fd', ctypes.c_uint32),
]
class struct_kfd_ioctl_dbg_trap_send_runtime_event_args(Struct): pass
struct_kfd_ioctl_dbg_trap_send_runtime_event_args._fields_ = [
  ('exception_mask', ctypes.c_uint64),
  ('gpu_id', ctypes.c_uint32),
  ('queue_id', ctypes.c_uint32),
]
class struct_kfd_ioctl_dbg_trap_set_exceptions_enabled_args(Struct): pass
struct_kfd_ioctl_dbg_trap_set_exceptions_enabled_args._fields_ = [
  ('exception_mask', ctypes.c_uint64),
]
class struct_kfd_ioctl_dbg_trap_set_wave_launch_override_args(Struct): pass
struct_kfd_ioctl_dbg_trap_set_wave_launch_override_args._fields_ = [
  ('override_mode', ctypes.c_uint32),
  ('enable_mask', ctypes.c_uint32),
  ('support_request_mask', ctypes.c_uint32),
  ('pad', ctypes.c_uint32),
]
class struct_kfd_ioctl_dbg_trap_set_wave_launch_mode_args(Struct): pass
struct_kfd_ioctl_dbg_trap_set_wave_launch_mode_args._fields_ = [
  ('launch_mode', ctypes.c_uint32),
  ('pad', ctypes.c_uint32),
]
class struct_kfd_ioctl_dbg_trap_suspend_queues_args(Struct): pass
struct_kfd_ioctl_dbg_trap_suspend_queues_args._fields_ = [
  ('exception_mask', ctypes.c_uint64),
  ('queue_array_ptr', ctypes.c_uint64),
  ('num_queues', ctypes.c_uint32),
  ('grace_period', ctypes.c_uint32),
]
class struct_kfd_ioctl_dbg_trap_resume_queues_args(Struct): pass
struct_kfd_ioctl_dbg_trap_resume_queues_args._fields_ = [
  ('queue_array_ptr', ctypes.c_uint64),
  ('num_queues', ctypes.c_uint32),
  ('pad', ctypes.c_uint32),
]
class struct_kfd_ioctl_dbg_trap_set_node_address_watch_args(Struct): pass
struct_kfd_ioctl_dbg_trap_set_node_address_watch_args._fields_ = [
  ('address', ctypes.c_uint64),
  ('mode', ctypes.c_uint32),
  ('mask', ctypes.c_uint32),
  ('gpu_id', ctypes.c_uint32),
  ('id', ctypes.c_uint32),
]
class struct_kfd_ioctl_dbg_trap_clear_node_address_watch_args(Struct): pass
struct_kfd_ioctl_dbg_trap_clear_node_address_watch_args._fields_ = [
  ('gpu_id', ctypes.c_uint32),
  ('id', ctypes.c_uint32),
]
class struct_kfd_ioctl_dbg_trap_set_flags_args(Struct): pass
struct_kfd_ioctl_dbg_trap_set_flags_args._fields_ = [
  ('flags', ctypes.c_uint32),
  ('pad', ctypes.c_uint32),
]
class struct_kfd_ioctl_dbg_trap_query_debug_event_args(Struct): pass
struct_kfd_ioctl_dbg_trap_query_debug_event_args._fields_ = [
  ('exception_mask', ctypes.c_uint64),
  ('gpu_id', ctypes.c_uint32),
  ('queue_id', ctypes.c_uint32),
]
class struct_kfd_ioctl_dbg_trap_query_exception_info_args(Struct): pass
struct_kfd_ioctl_dbg_trap_query_exception_info_args._fields_ = [
  ('info_ptr', ctypes.c_uint64),
  ('info_size', ctypes.c_uint32),
  ('source_id', ctypes.c_uint32),
  ('exception_code', ctypes.c_uint32),
  ('clear_exception', ctypes.c_uint32),
]
class struct_kfd_ioctl_dbg_trap_queue_snapshot_args(Struct): pass
struct_kfd_ioctl_dbg_trap_queue_snapshot_args._fields_ = [
  ('exception_mask', ctypes.c_uint64),
  ('snapshot_buf_ptr', ctypes.c_uint64),
  ('num_queues', ctypes.c_uint32),
  ('entry_size', ctypes.c_uint32),
]
class struct_kfd_ioctl_dbg_trap_device_snapshot_args(Struct): pass
struct_kfd_ioctl_dbg_trap_device_snapshot_args._fields_ = [
  ('exception_mask', ctypes.c_uint64),
  ('snapshot_buf_ptr', ctypes.c_uint64),
  ('num_devices', ctypes.c_uint32),
  ('entry_size', ctypes.c_uint32),
]
class struct_kfd_ioctl_dbg_trap_args(Struct): pass
class struct_kfd_ioctl_dbg_trap_args_0(ctypes.Union): pass
struct_kfd_ioctl_dbg_trap_args_0._fields_ = [
  ('enable', struct_kfd_ioctl_dbg_trap_enable_args),
  ('send_runtime_event', struct_kfd_ioctl_dbg_trap_send_runtime_event_args),
  ('set_exceptions_enabled', struct_kfd_ioctl_dbg_trap_set_exceptions_enabled_args),
  ('launch_override', struct_kfd_ioctl_dbg_trap_set_wave_launch_override_args),
  ('launch_mode', struct_kfd_ioctl_dbg_trap_set_wave_launch_mode_args),
  ('suspend_queues', struct_kfd_ioctl_dbg_trap_suspend_queues_args),
  ('resume_queues', struct_kfd_ioctl_dbg_trap_resume_queues_args),
  ('set_node_address_watch', struct_kfd_ioctl_dbg_trap_set_node_address_watch_args),
  ('clear_node_address_watch', struct_kfd_ioctl_dbg_trap_clear_node_address_watch_args),
  ('set_flags', struct_kfd_ioctl_dbg_trap_set_flags_args),
  ('query_debug_event', struct_kfd_ioctl_dbg_trap_query_debug_event_args),
  ('query_exception_info', struct_kfd_ioctl_dbg_trap_query_exception_info_args),
  ('queue_snapshot', struct_kfd_ioctl_dbg_trap_queue_snapshot_args),
  ('device_snapshot', struct_kfd_ioctl_dbg_trap_device_snapshot_args),
]
struct_kfd_ioctl_dbg_trap_args._anonymous_ = ['_0']
struct_kfd_ioctl_dbg_trap_args._fields_ = [
  ('pid', ctypes.c_uint32),
  ('op', ctypes.c_uint32),
  ('_0', struct_kfd_ioctl_dbg_trap_args_0),
]
KFD_IOCTL_MAJOR_VERSION = 1
KFD_IOCTL_MINOR_VERSION = 14
KFD_IOC_QUEUE_TYPE_COMPUTE = 0x0
KFD_IOC_QUEUE_TYPE_SDMA = 0x1
KFD_IOC_QUEUE_TYPE_COMPUTE_AQL = 0x2
KFD_IOC_QUEUE_TYPE_SDMA_XGMI = 0x3
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
KFD_SMI_EVENT_MASK_FROM_INDEX = lambda i: (1 << ((i) - 1))
KFD_SMI_EVENT_MSG_SIZE = 96
KFD_IOCTL_SVM_FLAG_HOST_ACCESS = 0x00000001
KFD_IOCTL_SVM_FLAG_COHERENT = 0x00000002
KFD_IOCTL_SVM_FLAG_HIVE_LOCAL = 0x00000004
KFD_IOCTL_SVM_FLAG_GPU_RO = 0x00000008
KFD_IOCTL_SVM_FLAG_GPU_EXEC = 0x00000010
KFD_IOCTL_SVM_FLAG_GPU_READ_MOSTLY = 0x00000020
KFD_IOCTL_SVM_FLAG_GPU_ALWAYS_MAPPED = 0x00000040
KFD_IOCTL_SVM_FLAG_EXT_COHERENT = 0x00000080
KFD_EC_MASK = lambda ecode: (1 << (ecode - 1))
KFD_EC_MASK_QUEUE = (KFD_EC_MASK(EC_QUEUE_WAVE_ABORT) | KFD_EC_MASK(EC_QUEUE_WAVE_TRAP) | KFD_EC_MASK(EC_QUEUE_WAVE_MATH_ERROR) | KFD_EC_MASK(EC_QUEUE_WAVE_ILLEGAL_INSTRUCTION) | KFD_EC_MASK(EC_QUEUE_WAVE_MEMORY_VIOLATION) | KFD_EC_MASK(EC_QUEUE_WAVE_APERTURE_VIOLATION) | KFD_EC_MASK(EC_QUEUE_PACKET_DISPATCH_DIM_INVALID) | KFD_EC_MASK(EC_QUEUE_PACKET_DISPATCH_GROUP_SEGMENT_SIZE_INVALID) | KFD_EC_MASK(EC_QUEUE_PACKET_DISPATCH_CODE_INVALID) | KFD_EC_MASK(EC_QUEUE_PACKET_RESERVED) | KFD_EC_MASK(EC_QUEUE_PACKET_UNSUPPORTED) | KFD_EC_MASK(EC_QUEUE_PACKET_DISPATCH_WORK_GROUP_SIZE_INVALID) | KFD_EC_MASK(EC_QUEUE_PACKET_DISPATCH_REGISTER_INVALID) | KFD_EC_MASK(EC_QUEUE_PACKET_VENDOR_UNSUPPORTED)	| KFD_EC_MASK(EC_QUEUE_PREEMPTION_ERROR)	| KFD_EC_MASK(EC_QUEUE_NEW))
KFD_EC_MASK_DEVICE = (KFD_EC_MASK(EC_DEVICE_QUEUE_DELETE) | KFD_EC_MASK(EC_DEVICE_RAS_ERROR) | KFD_EC_MASK(EC_DEVICE_FATAL_HALT) | KFD_EC_MASK(EC_DEVICE_MEMORY_VIOLATION) | KFD_EC_MASK(EC_DEVICE_NEW))
KFD_EC_MASK_PROCESS = (KFD_EC_MASK(EC_PROCESS_RUNTIME) | KFD_EC_MASK(EC_PROCESS_DEVICE_REMOVE))
KFD_EC_MASK_PACKET = (KFD_EC_MASK(EC_QUEUE_PACKET_DISPATCH_DIM_INVALID) | KFD_EC_MASK(EC_QUEUE_PACKET_DISPATCH_GROUP_SEGMENT_SIZE_INVALID) | KFD_EC_MASK(EC_QUEUE_PACKET_DISPATCH_CODE_INVALID) | KFD_EC_MASK(EC_QUEUE_PACKET_RESERVED) | KFD_EC_MASK(EC_QUEUE_PACKET_UNSUPPORTED) | KFD_EC_MASK(EC_QUEUE_PACKET_DISPATCH_WORK_GROUP_SIZE_INVALID) | KFD_EC_MASK(EC_QUEUE_PACKET_DISPATCH_REGISTER_INVALID) | KFD_EC_MASK(EC_QUEUE_PACKET_VENDOR_UNSUPPORTED))
KFD_DBG_EC_IS_VALID = lambda ecode: (ecode > EC_NONE and ecode < EC_MAX)
KFD_DBG_EC_TYPE_IS_QUEUE = lambda ecode: (KFD_DBG_EC_IS_VALID(ecode) and not  not (KFD_EC_MASK(ecode) & KFD_EC_MASK_QUEUE))
KFD_DBG_EC_TYPE_IS_DEVICE = lambda ecode: (KFD_DBG_EC_IS_VALID(ecode) and not  not (KFD_EC_MASK(ecode) & KFD_EC_MASK_DEVICE))
KFD_DBG_EC_TYPE_IS_PROCESS = lambda ecode: (KFD_DBG_EC_IS_VALID(ecode) and not  not (KFD_EC_MASK(ecode) & KFD_EC_MASK_PROCESS))
KFD_DBG_EC_TYPE_IS_PACKET = lambda ecode: (KFD_DBG_EC_IS_VALID(ecode) and not  not (KFD_EC_MASK(ecode) & KFD_EC_MASK_PACKET))
KFD_RUNTIME_ENABLE_MODE_ENABLE_MASK = 1
KFD_RUNTIME_ENABLE_MODE_TTMP_SAVE_MASK = 2
KFD_DBG_QUEUE_ERROR_BIT = 30
KFD_DBG_QUEUE_INVALID_BIT = 31
KFD_DBG_QUEUE_ERROR_MASK = (1 << KFD_DBG_QUEUE_ERROR_BIT)
KFD_DBG_QUEUE_INVALID_MASK = (1 << KFD_DBG_QUEUE_INVALID_BIT)
AMDKFD_IOCTL_BASE = 'K'
AMDKFD_IO = lambda nr: _IO(AMDKFD_IOCTL_BASE, nr)
AMDKFD_IOR = lambda nr,type: _IOR(AMDKFD_IOCTL_BASE, nr, type)
AMDKFD_IOW = lambda nr,type: _IOW(AMDKFD_IOCTL_BASE, nr, type)
AMDKFD_IOWR = lambda nr,type: _IOWR(AMDKFD_IOCTL_BASE, nr, type)
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