# mypy: ignore-errors
# -*- coding: utf-8 -*-
#
# TARGET arch is: []
# WORD_SIZE is: 8
# POINTER_SIZE is: 8
# LONGDOUBLE_SIZE is: 16
#
import ctypes, os



import functools
from tinygrad.runtime.support.hcq import FileIOInterface

def _do_ioctl(__idir, __base, __nr, __user_struct, __fd:FileIOInterface, **kwargs):
  ret = __fd.ioctl((__idir<<30) | (ctypes.sizeof(made := __user_struct(**kwargs))<<16) | (__base<<8) | __nr, made)
  if ret != 0: raise RuntimeError(f"ioctl returned {ret}")
  return made

def _IO(base, nr): return functools.partial(_do_ioctl, 0, ord(base) if isinstance(base, str) else base, nr, None)
def _IOW(base, nr, type): return functools.partial(_do_ioctl, 1, ord(base) if isinstance(base, str) else base, nr, type)
def _IOR(base, nr, type): return functools.partial(_do_ioctl, 2, ord(base) if isinstance(base, str) else base, nr, type)
def _IOWR(base, nr, type): return functools.partial(_do_ioctl, 3, ord(base) if isinstance(base, str) else base, nr, type)

class AsDictMixin:
    @classmethod
    def as_dict(cls, self):
        result = {}
        if not isinstance(self, AsDictMixin):
            # not a structure, assume it's already a python object
            return self
        if not hasattr(cls, "_fields_"):
            return result
        # sys.version_info >= (3, 5)
        # for (field, *_) in cls._fields_:  # noqa
        for field_tuple in cls._fields_:  # noqa
            field = field_tuple[0]
            if field.startswith('PADDING_'):
                continue
            value = getattr(self, field)
            type_ = type(value)
            if hasattr(value, "_length_") and hasattr(value, "_type_"):
                # array
                if not hasattr(type_, "as_dict"):
                    value = [v for v in value]
                else:
                    type_ = type_._type_
                    value = [type_.as_dict(v) for v in value]
            elif hasattr(value, "contents") and hasattr(value, "_type_"):
                # pointer
                try:
                    if not hasattr(type_, "as_dict"):
                        value = value.contents
                    else:
                        type_ = type_._type_
                        value = type_.as_dict(value.contents)
                except ValueError:
                    # nullptr
                    value = None
            elif isinstance(value, AsDictMixin):
                # other structure
                value = type_.as_dict(value)
            result[field] = value
        return result


class Structure(ctypes.Structure, AsDictMixin):

    def __init__(self, *args, **kwds):
        # We don't want to use positional arguments fill PADDING_* fields

        args = dict(zip(self.__class__._field_names_(), args))
        args.update(kwds)
        super(Structure, self).__init__(**args)

    @classmethod
    def _field_names_(cls):
        if hasattr(cls, '_fields_'):
            return (f[0] for f in cls._fields_ if not f[0].startswith('PADDING'))
        else:
            return ()

    @classmethod
    def get_type(cls, field):
        for f in cls._fields_:
            if f[0] == field:
                return f[1]
        return None

    @classmethod
    def bind(cls, bound_fields):
        fields = {}
        for name, type_ in cls._fields_:
            if hasattr(type_, "restype"):
                if name in bound_fields:
                    if bound_fields[name] is None:
                        fields[name] = type_()
                    else:
                        # use a closure to capture the callback from the loop scope
                        fields[name] = (
                            type_((lambda callback: lambda *args: callback(*args))(
                                bound_fields[name]))
                        )
                    del bound_fields[name]
                else:
                    # default callback implementation (does nothing)
                    try:
                        default_ = type_(0).restype().value
                    except TypeError:
                        default_ = None
                    fields[name] = type_((
                        lambda default_: lambda *args: default_)(default_))
            else:
                # not a callback function, use default initialization
                if name in bound_fields:
                    fields[name] = bound_fields[name]
                    del bound_fields[name]
                else:
                    fields[name] = type_()
        if len(bound_fields) != 0:
            raise ValueError(
                "Cannot bind the following unknown callback(s) {}.{}".format(
                    cls.__name__, bound_fields.keys()
            ))
        return cls(**fields)


class Union(ctypes.Union, AsDictMixin):
    pass





KFD_IOCTL_H_INCLUDED = True # macro
KFD_IOCTL_MAJOR_VERSION = 1 # macro
KFD_IOCTL_MINOR_VERSION = 14 # macro
KFD_IOC_QUEUE_TYPE_COMPUTE = 0x0 # macro
KFD_IOC_QUEUE_TYPE_SDMA = 0x1 # macro
KFD_IOC_QUEUE_TYPE_COMPUTE_AQL = 0x2 # macro
KFD_IOC_QUEUE_TYPE_SDMA_XGMI = 0x3 # macro
KFD_MAX_QUEUE_PERCENTAGE = 100 # macro
KFD_MAX_QUEUE_PRIORITY = 15 # macro
KFD_IOC_CACHE_POLICY_COHERENT = 0 # macro
KFD_IOC_CACHE_POLICY_NONCOHERENT = 1 # macro
NUM_OF_SUPPORTED_GPUS = 7 # macro
MAX_ALLOWED_NUM_POINTS = 100 # macro
MAX_ALLOWED_AW_BUFF_SIZE = 4096 # macro
MAX_ALLOWED_WAC_BUFF_SIZE = 128 # macro
KFD_INVALID_FD = 0xffffffff # macro
KFD_IOC_EVENT_SIGNAL = 0 # macro
KFD_IOC_EVENT_NODECHANGE = 1 # macro
KFD_IOC_EVENT_DEVICESTATECHANGE = 2 # macro
KFD_IOC_EVENT_HW_EXCEPTION = 3 # macro
KFD_IOC_EVENT_SYSTEM_EVENT = 4 # macro
KFD_IOC_EVENT_DEBUG_EVENT = 5 # macro
KFD_IOC_EVENT_PROFILE_EVENT = 6 # macro
KFD_IOC_EVENT_QUEUE_EVENT = 7 # macro
KFD_IOC_EVENT_MEMORY = 8 # macro
KFD_IOC_WAIT_RESULT_COMPLETE = 0 # macro
KFD_IOC_WAIT_RESULT_TIMEOUT = 1 # macro
KFD_IOC_WAIT_RESULT_FAIL = 2 # macro
KFD_SIGNAL_EVENT_LIMIT = 4096 # macro
KFD_HW_EXCEPTION_WHOLE_GPU_RESET = 0 # macro
KFD_HW_EXCEPTION_PER_ENGINE_RESET = 1 # macro
KFD_HW_EXCEPTION_GPU_HANG = 0 # macro
KFD_HW_EXCEPTION_ECC = 1 # macro
KFD_MEM_ERR_NO_RAS = 0 # macro
KFD_MEM_ERR_SRAM_ECC = 1 # macro
KFD_MEM_ERR_POISON_CONSUMED = 2 # macro
KFD_MEM_ERR_GPU_HANG = 3 # macro
KFD_IOC_ALLOC_MEM_FLAGS_VRAM = (1<<0) # macro
KFD_IOC_ALLOC_MEM_FLAGS_GTT = (1<<1) # macro
KFD_IOC_ALLOC_MEM_FLAGS_USERPTR = (1<<2) # macro
KFD_IOC_ALLOC_MEM_FLAGS_DOORBELL = (1<<3) # macro
KFD_IOC_ALLOC_MEM_FLAGS_MMIO_REMAP = (1<<4) # macro
KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE = (1<<31) # macro
KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE = (1<<30) # macro
KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC = (1<<29) # macro
KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE = (1<<28) # macro
KFD_IOC_ALLOC_MEM_FLAGS_AQL_QUEUE_MEM = (1<<27) # macro
KFD_IOC_ALLOC_MEM_FLAGS_COHERENT = (1<<26) # macro
KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED = (1<<25) # macro
KFD_IOC_ALLOC_MEM_FLAGS_EXT_COHERENT = (1<<24) # macro
def KFD_SMI_EVENT_MASK_FROM_INDEX(i):  # macro
   return (1<<((i)-1))
KFD_SMI_EVENT_MSG_SIZE = 96 # macro
KFD_IOCTL_SVM_FLAG_HOST_ACCESS = 0x00000001 # macro
KFD_IOCTL_SVM_FLAG_COHERENT = 0x00000002 # macro
KFD_IOCTL_SVM_FLAG_HIVE_LOCAL = 0x00000004 # macro
KFD_IOCTL_SVM_FLAG_GPU_RO = 0x00000008 # macro
KFD_IOCTL_SVM_FLAG_GPU_EXEC = 0x00000010 # macro
KFD_IOCTL_SVM_FLAG_GPU_READ_MOSTLY = 0x00000020 # macro
KFD_IOCTL_SVM_FLAG_GPU_ALWAYS_MAPPED = 0x00000040 # macro
KFD_IOCTL_SVM_FLAG_EXT_COHERENT = 0x00000080 # macro
def KFD_EC_MASK(ecode):  # macro
   return (1<<(ecode-1))
KFD_RUNTIME_ENABLE_MODE_ENABLE_MASK = 1 # macro
KFD_RUNTIME_ENABLE_MODE_TTMP_SAVE_MASK = 2 # macro
KFD_DBG_QUEUE_ERROR_BIT = 30 # macro
KFD_DBG_QUEUE_INVALID_BIT = 31 # macro
KFD_DBG_QUEUE_ERROR_MASK = (1<<30) # macro
KFD_DBG_QUEUE_INVALID_MASK = (1<<31) # macro
AMDKFD_IOCTL_BASE = 'K' # macro
def AMDKFD_IO(nr):  # macro
   return _IO('K',nr)
def AMDKFD_IOR(nr, type):  # macro
   return _IOR('K',nr,type)
def AMDKFD_IOW(nr, type):  # macro
   return _IOW('K',nr,type)
def AMDKFD_IOWR(nr, type):  # macro
   return _IOWR('K',nr,type)
AMDKFD_COMMAND_START = 0x01 # macro
AMDKFD_COMMAND_END = 0x27 # macro
class struct_kfd_ioctl_get_version_args(Structure):
    pass

struct_kfd_ioctl_get_version_args._pack_ = 1 # source:False
struct_kfd_ioctl_get_version_args._fields_ = [
    ('major_version', ctypes.c_uint32),
    ('minor_version', ctypes.c_uint32),
]

AMDKFD_IOC_GET_VERSION = AMDKFD_IOR ( 0x01 , struct_kfd_ioctl_get_version_args ) # macro (from list)
class struct_kfd_ioctl_create_queue_args(Structure):
    pass

struct_kfd_ioctl_create_queue_args._pack_ = 1 # source:False
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

AMDKFD_IOC_CREATE_QUEUE = AMDKFD_IOWR ( 0x02 , struct_kfd_ioctl_create_queue_args ) # macro (from list)
class struct_kfd_ioctl_destroy_queue_args(Structure):
    pass

struct_kfd_ioctl_destroy_queue_args._pack_ = 1 # source:False
struct_kfd_ioctl_destroy_queue_args._fields_ = [
    ('queue_id', ctypes.c_uint32),
    ('pad', ctypes.c_uint32),
]

AMDKFD_IOC_DESTROY_QUEUE = AMDKFD_IOWR ( 0x03 , struct_kfd_ioctl_destroy_queue_args ) # macro (from list)
class struct_kfd_ioctl_update_queue_args(Structure):
    pass

struct_kfd_ioctl_update_queue_args._pack_ = 1 # source:False
struct_kfd_ioctl_update_queue_args._fields_ = [
    ('ring_base_address', ctypes.c_uint64),
    ('queue_id', ctypes.c_uint32),
    ('ring_size', ctypes.c_uint32),
    ('queue_percentage', ctypes.c_uint32),
    ('queue_priority', ctypes.c_uint32),
]

AMDKFD_IOC_UPDATE_QUEUE = AMDKFD_IOW ( 0x07 , struct_kfd_ioctl_update_queue_args ) # macro (from list)
class struct_kfd_ioctl_set_cu_mask_args(Structure):
    pass

struct_kfd_ioctl_set_cu_mask_args._pack_ = 1 # source:False
struct_kfd_ioctl_set_cu_mask_args._fields_ = [
    ('queue_id', ctypes.c_uint32),
    ('num_cu_mask', ctypes.c_uint32),
    ('cu_mask_ptr', ctypes.c_uint64),
]

AMDKFD_IOC_SET_CU_MASK = AMDKFD_IOW ( 0x1A , struct_kfd_ioctl_set_cu_mask_args ) # macro (from list)
class struct_kfd_ioctl_get_queue_wave_state_args(Structure):
    pass

struct_kfd_ioctl_get_queue_wave_state_args._pack_ = 1 # source:False
struct_kfd_ioctl_get_queue_wave_state_args._fields_ = [
    ('ctl_stack_address', ctypes.c_uint64),
    ('ctl_stack_used_size', ctypes.c_uint32),
    ('save_area_used_size', ctypes.c_uint32),
    ('queue_id', ctypes.c_uint32),
    ('pad', ctypes.c_uint32),
]

AMDKFD_IOC_GET_QUEUE_WAVE_STATE = AMDKFD_IOWR ( 0x1B , struct_kfd_ioctl_get_queue_wave_state_args ) # macro (from list)
class struct_kfd_ioctl_get_available_memory_args(Structure):
    pass

struct_kfd_ioctl_get_available_memory_args._pack_ = 1 # source:False
struct_kfd_ioctl_get_available_memory_args._fields_ = [
    ('available', ctypes.c_uint64),
    ('gpu_id', ctypes.c_uint32),
    ('pad', ctypes.c_uint32),
]

AMDKFD_IOC_AVAILABLE_MEMORY = AMDKFD_IOWR ( 0x23 , struct_kfd_ioctl_get_available_memory_args ) # macro (from list)
class struct_kfd_dbg_device_info_entry(Structure):
    pass

struct_kfd_dbg_device_info_entry._pack_ = 1 # source:False
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

class struct_kfd_ioctl_set_memory_policy_args(Structure):
    pass

struct_kfd_ioctl_set_memory_policy_args._pack_ = 1 # source:False
struct_kfd_ioctl_set_memory_policy_args._fields_ = [
    ('alternate_aperture_base', ctypes.c_uint64),
    ('alternate_aperture_size', ctypes.c_uint64),
    ('gpu_id', ctypes.c_uint32),
    ('default_policy', ctypes.c_uint32),
    ('alternate_policy', ctypes.c_uint32),
    ('pad', ctypes.c_uint32),
]

AMDKFD_IOC_SET_MEMORY_POLICY = AMDKFD_IOW ( 0x04 , struct_kfd_ioctl_set_memory_policy_args ) # macro (from list)
class struct_kfd_ioctl_get_clock_counters_args(Structure):
    pass

struct_kfd_ioctl_get_clock_counters_args._pack_ = 1 # source:False
struct_kfd_ioctl_get_clock_counters_args._fields_ = [
    ('gpu_clock_counter', ctypes.c_uint64),
    ('cpu_clock_counter', ctypes.c_uint64),
    ('system_clock_counter', ctypes.c_uint64),
    ('system_clock_freq', ctypes.c_uint64),
    ('gpu_id', ctypes.c_uint32),
    ('pad', ctypes.c_uint32),
]

AMDKFD_IOC_GET_CLOCK_COUNTERS = AMDKFD_IOWR ( 0x05 , struct_kfd_ioctl_get_clock_counters_args ) # macro (from list)
class struct_kfd_process_device_apertures(Structure):
    pass

struct_kfd_process_device_apertures._pack_ = 1 # source:False
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

class struct_kfd_ioctl_get_process_apertures_args(Structure):
    pass

struct_kfd_ioctl_get_process_apertures_args._pack_ = 1 # source:False
struct_kfd_ioctl_get_process_apertures_args._fields_ = [
    ('process_apertures', struct_kfd_process_device_apertures * 7),
    ('num_of_nodes', ctypes.c_uint32),
    ('pad', ctypes.c_uint32),
]

AMDKFD_IOC_GET_PROCESS_APERTURES = AMDKFD_IOR ( 0x06 , struct_kfd_ioctl_get_process_apertures_args ) # macro (from list)
class struct_kfd_ioctl_get_process_apertures_new_args(Structure):
    pass

struct_kfd_ioctl_get_process_apertures_new_args._pack_ = 1 # source:False
struct_kfd_ioctl_get_process_apertures_new_args._fields_ = [
    ('kfd_process_device_apertures_ptr', ctypes.c_uint64),
    ('num_of_nodes', ctypes.c_uint32),
    ('pad', ctypes.c_uint32),
]

AMDKFD_IOC_GET_PROCESS_APERTURES_NEW = AMDKFD_IOWR ( 0x14 , struct_kfd_ioctl_get_process_apertures_new_args ) # macro (from list)
class struct_kfd_ioctl_dbg_register_args(Structure):
    pass

struct_kfd_ioctl_dbg_register_args._pack_ = 1 # source:False
struct_kfd_ioctl_dbg_register_args._fields_ = [
    ('gpu_id', ctypes.c_uint32),
    ('pad', ctypes.c_uint32),
]

AMDKFD_IOC_DBG_REGISTER_DEPRECATED = AMDKFD_IOW ( 0x0D , struct_kfd_ioctl_dbg_register_args ) # macro (from list)
class struct_kfd_ioctl_dbg_unregister_args(Structure):
    pass

struct_kfd_ioctl_dbg_unregister_args._pack_ = 1 # source:False
struct_kfd_ioctl_dbg_unregister_args._fields_ = [
    ('gpu_id', ctypes.c_uint32),
    ('pad', ctypes.c_uint32),
]

AMDKFD_IOC_DBG_UNREGISTER_DEPRECATED = AMDKFD_IOW ( 0x0E , struct_kfd_ioctl_dbg_unregister_args ) # macro (from list)
class struct_kfd_ioctl_dbg_address_watch_args(Structure):
    pass

struct_kfd_ioctl_dbg_address_watch_args._pack_ = 1 # source:False
struct_kfd_ioctl_dbg_address_watch_args._fields_ = [
    ('content_ptr', ctypes.c_uint64),
    ('gpu_id', ctypes.c_uint32),
    ('buf_size_in_bytes', ctypes.c_uint32),
]

AMDKFD_IOC_DBG_ADDRESS_WATCH_DEPRECATED = AMDKFD_IOW ( 0x0F , struct_kfd_ioctl_dbg_address_watch_args ) # macro (from list)
class struct_kfd_ioctl_dbg_wave_control_args(Structure):
    pass

struct_kfd_ioctl_dbg_wave_control_args._pack_ = 1 # source:False
struct_kfd_ioctl_dbg_wave_control_args._fields_ = [
    ('content_ptr', ctypes.c_uint64),
    ('gpu_id', ctypes.c_uint32),
    ('buf_size_in_bytes', ctypes.c_uint32),
]

AMDKFD_IOC_DBG_WAVE_CONTROL_DEPRECATED = AMDKFD_IOW ( 0x10 , struct_kfd_ioctl_dbg_wave_control_args ) # macro (from list)
class struct_kfd_ioctl_create_event_args(Structure):
    pass

struct_kfd_ioctl_create_event_args._pack_ = 1 # source:False
struct_kfd_ioctl_create_event_args._fields_ = [
    ('event_page_offset', ctypes.c_uint64),
    ('event_trigger_data', ctypes.c_uint32),
    ('event_type', ctypes.c_uint32),
    ('auto_reset', ctypes.c_uint32),
    ('node_id', ctypes.c_uint32),
    ('event_id', ctypes.c_uint32),
    ('event_slot_index', ctypes.c_uint32),
]

AMDKFD_IOC_CREATE_EVENT = AMDKFD_IOWR ( 0x08 , struct_kfd_ioctl_create_event_args ) # macro (from list)
class struct_kfd_ioctl_destroy_event_args(Structure):
    pass

struct_kfd_ioctl_destroy_event_args._pack_ = 1 # source:False
struct_kfd_ioctl_destroy_event_args._fields_ = [
    ('event_id', ctypes.c_uint32),
    ('pad', ctypes.c_uint32),
]

AMDKFD_IOC_DESTROY_EVENT = AMDKFD_IOW ( 0x09 , struct_kfd_ioctl_destroy_event_args ) # macro (from list)
class struct_kfd_ioctl_set_event_args(Structure):
    pass

struct_kfd_ioctl_set_event_args._pack_ = 1 # source:False
struct_kfd_ioctl_set_event_args._fields_ = [
    ('event_id', ctypes.c_uint32),
    ('pad', ctypes.c_uint32),
]

AMDKFD_IOC_SET_EVENT = AMDKFD_IOW ( 0x0A , struct_kfd_ioctl_set_event_args ) # macro (from list)
class struct_kfd_ioctl_reset_event_args(Structure):
    pass

struct_kfd_ioctl_reset_event_args._pack_ = 1 # source:False
struct_kfd_ioctl_reset_event_args._fields_ = [
    ('event_id', ctypes.c_uint32),
    ('pad', ctypes.c_uint32),
]

AMDKFD_IOC_RESET_EVENT = AMDKFD_IOW ( 0x0B , struct_kfd_ioctl_reset_event_args ) # macro (from list)
class struct_kfd_memory_exception_failure(Structure):
    pass

struct_kfd_memory_exception_failure._pack_ = 1 # source:False
struct_kfd_memory_exception_failure._fields_ = [
    ('NotPresent', ctypes.c_uint32),
    ('ReadOnly', ctypes.c_uint32),
    ('NoExecute', ctypes.c_uint32),
    ('imprecise', ctypes.c_uint32),
]

class struct_kfd_hsa_memory_exception_data(Structure):
    pass

struct_kfd_hsa_memory_exception_data._pack_ = 1 # source:False
struct_kfd_hsa_memory_exception_data._fields_ = [
    ('failure', struct_kfd_memory_exception_failure),
    ('va', ctypes.c_uint64),
    ('gpu_id', ctypes.c_uint32),
    ('ErrorType', ctypes.c_uint32),
]

class struct_kfd_hsa_hw_exception_data(Structure):
    pass

struct_kfd_hsa_hw_exception_data._pack_ = 1 # source:False
struct_kfd_hsa_hw_exception_data._fields_ = [
    ('reset_type', ctypes.c_uint32),
    ('reset_cause', ctypes.c_uint32),
    ('memory_lost', ctypes.c_uint32),
    ('gpu_id', ctypes.c_uint32),
]

class struct_kfd_hsa_signal_event_data(Structure):
    pass

struct_kfd_hsa_signal_event_data._pack_ = 1 # source:False
struct_kfd_hsa_signal_event_data._fields_ = [
    ('last_event_age', ctypes.c_uint64),
]

class struct_kfd_event_data(Structure):
    pass

class union_kfd_event_data_0(Union):
    pass

union_kfd_event_data_0._pack_ = 1 # source:False
union_kfd_event_data_0._fields_ = [
    ('memory_exception_data', struct_kfd_hsa_memory_exception_data),
    ('hw_exception_data', struct_kfd_hsa_hw_exception_data),
    ('signal_event_data', struct_kfd_hsa_signal_event_data),
    ('PADDING_0', ctypes.c_ubyte * 24),
]

struct_kfd_event_data._pack_ = 1 # source:False
struct_kfd_event_data._anonymous_ = ('_0',)
struct_kfd_event_data._fields_ = [
    ('_0', union_kfd_event_data_0),
    ('kfd_event_data_ext', ctypes.c_uint64),
    ('event_id', ctypes.c_uint32),
    ('pad', ctypes.c_uint32),
]

class struct_kfd_ioctl_wait_events_args(Structure):
    pass

struct_kfd_ioctl_wait_events_args._pack_ = 1 # source:False
struct_kfd_ioctl_wait_events_args._fields_ = [
    ('events_ptr', ctypes.c_uint64),
    ('num_events', ctypes.c_uint32),
    ('wait_for_all', ctypes.c_uint32),
    ('timeout', ctypes.c_uint32),
    ('wait_result', ctypes.c_uint32),
]

AMDKFD_IOC_WAIT_EVENTS = AMDKFD_IOWR ( 0x0C , struct_kfd_ioctl_wait_events_args ) # macro (from list)
class struct_kfd_ioctl_set_scratch_backing_va_args(Structure):
    pass

struct_kfd_ioctl_set_scratch_backing_va_args._pack_ = 1 # source:False
struct_kfd_ioctl_set_scratch_backing_va_args._fields_ = [
    ('va_addr', ctypes.c_uint64),
    ('gpu_id', ctypes.c_uint32),
    ('pad', ctypes.c_uint32),
]

AMDKFD_IOC_SET_SCRATCH_BACKING_VA = AMDKFD_IOWR ( 0x11 , struct_kfd_ioctl_set_scratch_backing_va_args ) # macro (from list)
class struct_kfd_ioctl_get_tile_config_args(Structure):
    pass

struct_kfd_ioctl_get_tile_config_args._pack_ = 1 # source:False
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

AMDKFD_IOC_GET_TILE_CONFIG = AMDKFD_IOWR ( 0x12 , struct_kfd_ioctl_get_tile_config_args ) # macro (from list)
class struct_kfd_ioctl_set_trap_handler_args(Structure):
    pass

struct_kfd_ioctl_set_trap_handler_args._pack_ = 1 # source:False
struct_kfd_ioctl_set_trap_handler_args._fields_ = [
    ('tba_addr', ctypes.c_uint64),
    ('tma_addr', ctypes.c_uint64),
    ('gpu_id', ctypes.c_uint32),
    ('pad', ctypes.c_uint32),
]

AMDKFD_IOC_SET_TRAP_HANDLER = AMDKFD_IOW ( 0x13 , struct_kfd_ioctl_set_trap_handler_args ) # macro (from list)
class struct_kfd_ioctl_acquire_vm_args(Structure):
    pass

struct_kfd_ioctl_acquire_vm_args._pack_ = 1 # source:False
struct_kfd_ioctl_acquire_vm_args._fields_ = [
    ('drm_fd', ctypes.c_uint32),
    ('gpu_id', ctypes.c_uint32),
]

AMDKFD_IOC_ACQUIRE_VM = AMDKFD_IOW ( 0x15 , struct_kfd_ioctl_acquire_vm_args ) # macro (from list)
class struct_kfd_ioctl_alloc_memory_of_gpu_args(Structure):
    pass

struct_kfd_ioctl_alloc_memory_of_gpu_args._pack_ = 1 # source:False
struct_kfd_ioctl_alloc_memory_of_gpu_args._fields_ = [
    ('va_addr', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
    ('handle', ctypes.c_uint64),
    ('mmap_offset', ctypes.c_uint64),
    ('gpu_id', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
]

AMDKFD_IOC_ALLOC_MEMORY_OF_GPU = AMDKFD_IOWR ( 0x16 , struct_kfd_ioctl_alloc_memory_of_gpu_args ) # macro (from list)
class struct_kfd_ioctl_free_memory_of_gpu_args(Structure):
    pass

struct_kfd_ioctl_free_memory_of_gpu_args._pack_ = 1 # source:False
struct_kfd_ioctl_free_memory_of_gpu_args._fields_ = [
    ('handle', ctypes.c_uint64),
]

AMDKFD_IOC_FREE_MEMORY_OF_GPU = AMDKFD_IOW ( 0x17 , struct_kfd_ioctl_free_memory_of_gpu_args ) # macro (from list)
class struct_kfd_ioctl_map_memory_to_gpu_args(Structure):
    pass

struct_kfd_ioctl_map_memory_to_gpu_args._pack_ = 1 # source:False
struct_kfd_ioctl_map_memory_to_gpu_args._fields_ = [
    ('handle', ctypes.c_uint64),
    ('device_ids_array_ptr', ctypes.c_uint64),
    ('n_devices', ctypes.c_uint32),
    ('n_success', ctypes.c_uint32),
]

AMDKFD_IOC_MAP_MEMORY_TO_GPU = AMDKFD_IOWR ( 0x18 , struct_kfd_ioctl_map_memory_to_gpu_args ) # macro (from list)
class struct_kfd_ioctl_unmap_memory_from_gpu_args(Structure):
    pass

struct_kfd_ioctl_unmap_memory_from_gpu_args._pack_ = 1 # source:False
struct_kfd_ioctl_unmap_memory_from_gpu_args._fields_ = [
    ('handle', ctypes.c_uint64),
    ('device_ids_array_ptr', ctypes.c_uint64),
    ('n_devices', ctypes.c_uint32),
    ('n_success', ctypes.c_uint32),
]

AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU = AMDKFD_IOWR ( 0x19 , struct_kfd_ioctl_unmap_memory_from_gpu_args ) # macro (from list)
class struct_kfd_ioctl_alloc_queue_gws_args(Structure):
    pass

struct_kfd_ioctl_alloc_queue_gws_args._pack_ = 1 # source:False
struct_kfd_ioctl_alloc_queue_gws_args._fields_ = [
    ('queue_id', ctypes.c_uint32),
    ('num_gws', ctypes.c_uint32),
    ('first_gws', ctypes.c_uint32),
    ('pad', ctypes.c_uint32),
]

AMDKFD_IOC_ALLOC_QUEUE_GWS = AMDKFD_IOWR ( 0x1E , struct_kfd_ioctl_alloc_queue_gws_args ) # macro (from list)
class struct_kfd_ioctl_get_dmabuf_info_args(Structure):
    pass

struct_kfd_ioctl_get_dmabuf_info_args._pack_ = 1 # source:False
struct_kfd_ioctl_get_dmabuf_info_args._fields_ = [
    ('size', ctypes.c_uint64),
    ('metadata_ptr', ctypes.c_uint64),
    ('metadata_size', ctypes.c_uint32),
    ('gpu_id', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('dmabuf_fd', ctypes.c_uint32),
]

AMDKFD_IOC_GET_DMABUF_INFO = AMDKFD_IOWR ( 0x1C , struct_kfd_ioctl_get_dmabuf_info_args ) # macro (from list)
class struct_kfd_ioctl_import_dmabuf_args(Structure):
    pass

struct_kfd_ioctl_import_dmabuf_args._pack_ = 1 # source:False
struct_kfd_ioctl_import_dmabuf_args._fields_ = [
    ('va_addr', ctypes.c_uint64),
    ('handle', ctypes.c_uint64),
    ('gpu_id', ctypes.c_uint32),
    ('dmabuf_fd', ctypes.c_uint32),
]

AMDKFD_IOC_IMPORT_DMABUF = AMDKFD_IOWR ( 0x1D , struct_kfd_ioctl_import_dmabuf_args ) # macro (from list)
class struct_kfd_ioctl_export_dmabuf_args(Structure):
    pass

struct_kfd_ioctl_export_dmabuf_args._pack_ = 1 # source:False
struct_kfd_ioctl_export_dmabuf_args._fields_ = [
    ('handle', ctypes.c_uint64),
    ('flags', ctypes.c_uint32),
    ('dmabuf_fd', ctypes.c_uint32),
]

AMDKFD_IOC_EXPORT_DMABUF = AMDKFD_IOWR ( 0x24 , struct_kfd_ioctl_export_dmabuf_args ) # macro (from list)

# values for enumeration 'kfd_smi_event'
kfd_smi_event__enumvalues = {
    0: 'KFD_SMI_EVENT_NONE',
    1: 'KFD_SMI_EVENT_VMFAULT',
    2: 'KFD_SMI_EVENT_THERMAL_THROTTLE',
    3: 'KFD_SMI_EVENT_GPU_PRE_RESET',
    4: 'KFD_SMI_EVENT_GPU_POST_RESET',
    5: 'KFD_SMI_EVENT_MIGRATE_START',
    6: 'KFD_SMI_EVENT_MIGRATE_END',
    7: 'KFD_SMI_EVENT_PAGE_FAULT_START',
    8: 'KFD_SMI_EVENT_PAGE_FAULT_END',
    9: 'KFD_SMI_EVENT_QUEUE_EVICTION',
    10: 'KFD_SMI_EVENT_QUEUE_RESTORE',
    11: 'KFD_SMI_EVENT_UNMAP_FROM_GPU',
    64: 'KFD_SMI_EVENT_ALL_PROCESS',
}
KFD_SMI_EVENT_NONE = 0
KFD_SMI_EVENT_VMFAULT = 1
KFD_SMI_EVENT_THERMAL_THROTTLE = 2
KFD_SMI_EVENT_GPU_PRE_RESET = 3
KFD_SMI_EVENT_GPU_POST_RESET = 4
KFD_SMI_EVENT_MIGRATE_START = 5
KFD_SMI_EVENT_MIGRATE_END = 6
KFD_SMI_EVENT_PAGE_FAULT_START = 7
KFD_SMI_EVENT_PAGE_FAULT_END = 8
KFD_SMI_EVENT_QUEUE_EVICTION = 9
KFD_SMI_EVENT_QUEUE_RESTORE = 10
KFD_SMI_EVENT_UNMAP_FROM_GPU = 11
KFD_SMI_EVENT_ALL_PROCESS = 64
kfd_smi_event = ctypes.c_uint32 # enum

# values for enumeration 'KFD_MIGRATE_TRIGGERS'
KFD_MIGRATE_TRIGGERS__enumvalues = {
    0: 'KFD_MIGRATE_TRIGGER_PREFETCH',
    1: 'KFD_MIGRATE_TRIGGER_PAGEFAULT_GPU',
    2: 'KFD_MIGRATE_TRIGGER_PAGEFAULT_CPU',
    3: 'KFD_MIGRATE_TRIGGER_TTM_EVICTION',
}
KFD_MIGRATE_TRIGGER_PREFETCH = 0
KFD_MIGRATE_TRIGGER_PAGEFAULT_GPU = 1
KFD_MIGRATE_TRIGGER_PAGEFAULT_CPU = 2
KFD_MIGRATE_TRIGGER_TTM_EVICTION = 3
KFD_MIGRATE_TRIGGERS = ctypes.c_uint32 # enum

# values for enumeration 'KFD_QUEUE_EVICTION_TRIGGERS'
KFD_QUEUE_EVICTION_TRIGGERS__enumvalues = {
    0: 'KFD_QUEUE_EVICTION_TRIGGER_SVM',
    1: 'KFD_QUEUE_EVICTION_TRIGGER_USERPTR',
    2: 'KFD_QUEUE_EVICTION_TRIGGER_TTM',
    3: 'KFD_QUEUE_EVICTION_TRIGGER_SUSPEND',
    4: 'KFD_QUEUE_EVICTION_CRIU_CHECKPOINT',
    5: 'KFD_QUEUE_EVICTION_CRIU_RESTORE',
}
KFD_QUEUE_EVICTION_TRIGGER_SVM = 0
KFD_QUEUE_EVICTION_TRIGGER_USERPTR = 1
KFD_QUEUE_EVICTION_TRIGGER_TTM = 2
KFD_QUEUE_EVICTION_TRIGGER_SUSPEND = 3
KFD_QUEUE_EVICTION_CRIU_CHECKPOINT = 4
KFD_QUEUE_EVICTION_CRIU_RESTORE = 5
KFD_QUEUE_EVICTION_TRIGGERS = ctypes.c_uint32 # enum

# values for enumeration 'KFD_SVM_UNMAP_TRIGGERS'
KFD_SVM_UNMAP_TRIGGERS__enumvalues = {
    0: 'KFD_SVM_UNMAP_TRIGGER_MMU_NOTIFY',
    1: 'KFD_SVM_UNMAP_TRIGGER_MMU_NOTIFY_MIGRATE',
    2: 'KFD_SVM_UNMAP_TRIGGER_UNMAP_FROM_CPU',
}
KFD_SVM_UNMAP_TRIGGER_MMU_NOTIFY = 0
KFD_SVM_UNMAP_TRIGGER_MMU_NOTIFY_MIGRATE = 1
KFD_SVM_UNMAP_TRIGGER_UNMAP_FROM_CPU = 2
KFD_SVM_UNMAP_TRIGGERS = ctypes.c_uint32 # enum
class struct_kfd_ioctl_smi_events_args(Structure):
    pass

struct_kfd_ioctl_smi_events_args._pack_ = 1 # source:False
struct_kfd_ioctl_smi_events_args._fields_ = [
    ('gpuid', ctypes.c_uint32),
    ('anon_fd', ctypes.c_uint32),
]

AMDKFD_IOC_SMI_EVENTS = AMDKFD_IOWR ( 0x1F , struct_kfd_ioctl_smi_events_args ) # macro (from list)

# values for enumeration 'kfd_criu_op'
kfd_criu_op__enumvalues = {
    0: 'KFD_CRIU_OP_PROCESS_INFO',
    1: 'KFD_CRIU_OP_CHECKPOINT',
    2: 'KFD_CRIU_OP_UNPAUSE',
    3: 'KFD_CRIU_OP_RESTORE',
    4: 'KFD_CRIU_OP_RESUME',
}
KFD_CRIU_OP_PROCESS_INFO = 0
KFD_CRIU_OP_CHECKPOINT = 1
KFD_CRIU_OP_UNPAUSE = 2
KFD_CRIU_OP_RESTORE = 3
KFD_CRIU_OP_RESUME = 4
kfd_criu_op = ctypes.c_uint32 # enum
class struct_kfd_ioctl_criu_args(Structure):
    pass

struct_kfd_ioctl_criu_args._pack_ = 1 # source:False
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
    ('PADDING_0', ctypes.c_ubyte * 4),
]

AMDKFD_IOC_CRIU_OP = AMDKFD_IOWR ( 0x22 , struct_kfd_ioctl_criu_args ) # macro (from list)
class struct_kfd_criu_device_bucket(Structure):
    pass

struct_kfd_criu_device_bucket._pack_ = 1 # source:False
struct_kfd_criu_device_bucket._fields_ = [
    ('user_gpu_id', ctypes.c_uint32),
    ('actual_gpu_id', ctypes.c_uint32),
    ('drm_fd', ctypes.c_uint32),
    ('pad', ctypes.c_uint32),
]

class struct_kfd_criu_bo_bucket(Structure):
    pass

struct_kfd_criu_bo_bucket._pack_ = 1 # source:False
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


# values for enumeration 'kfd_mmio_remap'
kfd_mmio_remap__enumvalues = {
    0: 'KFD_MMIO_REMAP_HDP_MEM_FLUSH_CNTL',
    4: 'KFD_MMIO_REMAP_HDP_REG_FLUSH_CNTL',
}
KFD_MMIO_REMAP_HDP_MEM_FLUSH_CNTL = 0
KFD_MMIO_REMAP_HDP_REG_FLUSH_CNTL = 4
kfd_mmio_remap = ctypes.c_uint32 # enum

# values for enumeration 'kfd_ioctl_svm_op'
kfd_ioctl_svm_op__enumvalues = {
    0: 'KFD_IOCTL_SVM_OP_SET_ATTR',
    1: 'KFD_IOCTL_SVM_OP_GET_ATTR',
}
KFD_IOCTL_SVM_OP_SET_ATTR = 0
KFD_IOCTL_SVM_OP_GET_ATTR = 1
kfd_ioctl_svm_op = ctypes.c_uint32 # enum

# values for enumeration 'kfd_ioctl_svm_location'
kfd_ioctl_svm_location__enumvalues = {
    0: 'KFD_IOCTL_SVM_LOCATION_SYSMEM',
    4294967295: 'KFD_IOCTL_SVM_LOCATION_UNDEFINED',
}
KFD_IOCTL_SVM_LOCATION_SYSMEM = 0
KFD_IOCTL_SVM_LOCATION_UNDEFINED = 4294967295
kfd_ioctl_svm_location = ctypes.c_uint32 # enum

# values for enumeration 'kfd_ioctl_svm_attr_type'
kfd_ioctl_svm_attr_type__enumvalues = {
    0: 'KFD_IOCTL_SVM_ATTR_PREFERRED_LOC',
    1: 'KFD_IOCTL_SVM_ATTR_PREFETCH_LOC',
    2: 'KFD_IOCTL_SVM_ATTR_ACCESS',
    3: 'KFD_IOCTL_SVM_ATTR_ACCESS_IN_PLACE',
    4: 'KFD_IOCTL_SVM_ATTR_NO_ACCESS',
    5: 'KFD_IOCTL_SVM_ATTR_SET_FLAGS',
    6: 'KFD_IOCTL_SVM_ATTR_CLR_FLAGS',
    7: 'KFD_IOCTL_SVM_ATTR_GRANULARITY',
}
KFD_IOCTL_SVM_ATTR_PREFERRED_LOC = 0
KFD_IOCTL_SVM_ATTR_PREFETCH_LOC = 1
KFD_IOCTL_SVM_ATTR_ACCESS = 2
KFD_IOCTL_SVM_ATTR_ACCESS_IN_PLACE = 3
KFD_IOCTL_SVM_ATTR_NO_ACCESS = 4
KFD_IOCTL_SVM_ATTR_SET_FLAGS = 5
KFD_IOCTL_SVM_ATTR_CLR_FLAGS = 6
KFD_IOCTL_SVM_ATTR_GRANULARITY = 7
kfd_ioctl_svm_attr_type = ctypes.c_uint32 # enum
class struct_kfd_ioctl_svm_attribute(Structure):
    pass

struct_kfd_ioctl_svm_attribute._pack_ = 1 # source:False
struct_kfd_ioctl_svm_attribute._fields_ = [
    ('type', ctypes.c_uint32),
    ('value', ctypes.c_uint32),
]

class struct_kfd_ioctl_svm_args(Structure):
    pass

struct_kfd_ioctl_svm_args._pack_ = 1 # source:False
struct_kfd_ioctl_svm_args._fields_ = [
    ('start_addr', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
    ('op', ctypes.c_uint32),
    ('nattr', ctypes.c_uint32),
    ('attrs', struct_kfd_ioctl_svm_attribute * 0),
]

AMDKFD_IOC_SVM = AMDKFD_IOWR ( 0x20 , struct_kfd_ioctl_svm_args ) # macro (from list)
class struct_kfd_ioctl_set_xnack_mode_args(Structure):
    pass

struct_kfd_ioctl_set_xnack_mode_args._pack_ = 1 # source:False
struct_kfd_ioctl_set_xnack_mode_args._fields_ = [
    ('xnack_enabled', ctypes.c_int32),
]

AMDKFD_IOC_SET_XNACK_MODE = AMDKFD_IOWR ( 0x21 , struct_kfd_ioctl_set_xnack_mode_args ) # macro (from list)

# values for enumeration 'kfd_dbg_trap_override_mode'
kfd_dbg_trap_override_mode__enumvalues = {
    0: 'KFD_DBG_TRAP_OVERRIDE_OR',
    1: 'KFD_DBG_TRAP_OVERRIDE_REPLACE',
}
KFD_DBG_TRAP_OVERRIDE_OR = 0
KFD_DBG_TRAP_OVERRIDE_REPLACE = 1
kfd_dbg_trap_override_mode = ctypes.c_uint32 # enum

# values for enumeration 'kfd_dbg_trap_mask'
kfd_dbg_trap_mask__enumvalues = {
    1: 'KFD_DBG_TRAP_MASK_FP_INVALID',
    2: 'KFD_DBG_TRAP_MASK_FP_INPUT_DENORMAL',
    4: 'KFD_DBG_TRAP_MASK_FP_DIVIDE_BY_ZERO',
    8: 'KFD_DBG_TRAP_MASK_FP_OVERFLOW',
    16: 'KFD_DBG_TRAP_MASK_FP_UNDERFLOW',
    32: 'KFD_DBG_TRAP_MASK_FP_INEXACT',
    64: 'KFD_DBG_TRAP_MASK_INT_DIVIDE_BY_ZERO',
    128: 'KFD_DBG_TRAP_MASK_DBG_ADDRESS_WATCH',
    256: 'KFD_DBG_TRAP_MASK_DBG_MEMORY_VIOLATION',
    1073741824: 'KFD_DBG_TRAP_MASK_TRAP_ON_WAVE_START',
    -2147483648: 'KFD_DBG_TRAP_MASK_TRAP_ON_WAVE_END',
}
KFD_DBG_TRAP_MASK_FP_INVALID = 1
KFD_DBG_TRAP_MASK_FP_INPUT_DENORMAL = 2
KFD_DBG_TRAP_MASK_FP_DIVIDE_BY_ZERO = 4
KFD_DBG_TRAP_MASK_FP_OVERFLOW = 8
KFD_DBG_TRAP_MASK_FP_UNDERFLOW = 16
KFD_DBG_TRAP_MASK_FP_INEXACT = 32
KFD_DBG_TRAP_MASK_INT_DIVIDE_BY_ZERO = 64
KFD_DBG_TRAP_MASK_DBG_ADDRESS_WATCH = 128
KFD_DBG_TRAP_MASK_DBG_MEMORY_VIOLATION = 256
KFD_DBG_TRAP_MASK_TRAP_ON_WAVE_START = 1073741824
KFD_DBG_TRAP_MASK_TRAP_ON_WAVE_END = -2147483648
kfd_dbg_trap_mask = ctypes.c_int32 # enum

# values for enumeration 'kfd_dbg_trap_wave_launch_mode'
kfd_dbg_trap_wave_launch_mode__enumvalues = {
    0: 'KFD_DBG_TRAP_WAVE_LAUNCH_MODE_NORMAL',
    1: 'KFD_DBG_TRAP_WAVE_LAUNCH_MODE_HALT',
    3: 'KFD_DBG_TRAP_WAVE_LAUNCH_MODE_DEBUG',
}
KFD_DBG_TRAP_WAVE_LAUNCH_MODE_NORMAL = 0
KFD_DBG_TRAP_WAVE_LAUNCH_MODE_HALT = 1
KFD_DBG_TRAP_WAVE_LAUNCH_MODE_DEBUG = 3
kfd_dbg_trap_wave_launch_mode = ctypes.c_uint32 # enum

# values for enumeration 'kfd_dbg_trap_address_watch_mode'
kfd_dbg_trap_address_watch_mode__enumvalues = {
    0: 'KFD_DBG_TRAP_ADDRESS_WATCH_MODE_READ',
    1: 'KFD_DBG_TRAP_ADDRESS_WATCH_MODE_NONREAD',
    2: 'KFD_DBG_TRAP_ADDRESS_WATCH_MODE_ATOMIC',
    3: 'KFD_DBG_TRAP_ADDRESS_WATCH_MODE_ALL',
}
KFD_DBG_TRAP_ADDRESS_WATCH_MODE_READ = 0
KFD_DBG_TRAP_ADDRESS_WATCH_MODE_NONREAD = 1
KFD_DBG_TRAP_ADDRESS_WATCH_MODE_ATOMIC = 2
KFD_DBG_TRAP_ADDRESS_WATCH_MODE_ALL = 3
kfd_dbg_trap_address_watch_mode = ctypes.c_uint32 # enum

# values for enumeration 'kfd_dbg_trap_flags'
kfd_dbg_trap_flags__enumvalues = {
    1: 'KFD_DBG_TRAP_FLAG_SINGLE_MEM_OP',
}
KFD_DBG_TRAP_FLAG_SINGLE_MEM_OP = 1
kfd_dbg_trap_flags = ctypes.c_uint32 # enum

# values for enumeration 'kfd_dbg_trap_exception_code'
kfd_dbg_trap_exception_code__enumvalues = {
    0: 'EC_NONE',
    1: 'EC_QUEUE_WAVE_ABORT',
    2: 'EC_QUEUE_WAVE_TRAP',
    3: 'EC_QUEUE_WAVE_MATH_ERROR',
    4: 'EC_QUEUE_WAVE_ILLEGAL_INSTRUCTION',
    5: 'EC_QUEUE_WAVE_MEMORY_VIOLATION',
    6: 'EC_QUEUE_WAVE_APERTURE_VIOLATION',
    16: 'EC_QUEUE_PACKET_DISPATCH_DIM_INVALID',
    17: 'EC_QUEUE_PACKET_DISPATCH_GROUP_SEGMENT_SIZE_INVALID',
    18: 'EC_QUEUE_PACKET_DISPATCH_CODE_INVALID',
    19: 'EC_QUEUE_PACKET_RESERVED',
    20: 'EC_QUEUE_PACKET_UNSUPPORTED',
    21: 'EC_QUEUE_PACKET_DISPATCH_WORK_GROUP_SIZE_INVALID',
    22: 'EC_QUEUE_PACKET_DISPATCH_REGISTER_INVALID',
    23: 'EC_QUEUE_PACKET_VENDOR_UNSUPPORTED',
    30: 'EC_QUEUE_PREEMPTION_ERROR',
    31: 'EC_QUEUE_NEW',
    32: 'EC_DEVICE_QUEUE_DELETE',
    33: 'EC_DEVICE_MEMORY_VIOLATION',
    34: 'EC_DEVICE_RAS_ERROR',
    35: 'EC_DEVICE_FATAL_HALT',
    36: 'EC_DEVICE_NEW',
    48: 'EC_PROCESS_RUNTIME',
    49: 'EC_PROCESS_DEVICE_REMOVE',
    50: 'EC_MAX',
}
EC_NONE = 0
EC_QUEUE_WAVE_ABORT = 1
EC_QUEUE_WAVE_TRAP = 2
EC_QUEUE_WAVE_MATH_ERROR = 3
EC_QUEUE_WAVE_ILLEGAL_INSTRUCTION = 4
EC_QUEUE_WAVE_MEMORY_VIOLATION = 5
EC_QUEUE_WAVE_APERTURE_VIOLATION = 6
EC_QUEUE_PACKET_DISPATCH_DIM_INVALID = 16
EC_QUEUE_PACKET_DISPATCH_GROUP_SEGMENT_SIZE_INVALID = 17
EC_QUEUE_PACKET_DISPATCH_CODE_INVALID = 18
EC_QUEUE_PACKET_RESERVED = 19
EC_QUEUE_PACKET_UNSUPPORTED = 20
EC_QUEUE_PACKET_DISPATCH_WORK_GROUP_SIZE_INVALID = 21
EC_QUEUE_PACKET_DISPATCH_REGISTER_INVALID = 22
EC_QUEUE_PACKET_VENDOR_UNSUPPORTED = 23
EC_QUEUE_PREEMPTION_ERROR = 30
EC_QUEUE_NEW = 31
EC_DEVICE_QUEUE_DELETE = 32
EC_DEVICE_MEMORY_VIOLATION = 33
EC_DEVICE_RAS_ERROR = 34
EC_DEVICE_FATAL_HALT = 35
EC_DEVICE_NEW = 36
EC_PROCESS_RUNTIME = 48
EC_PROCESS_DEVICE_REMOVE = 49
EC_MAX = 50
kfd_dbg_trap_exception_code = ctypes.c_uint32 # enum
KFD_EC_MASK_QUEUE = (KFD_EC_MASK(EC_QUEUE_WAVE_ABORT)|KFD_EC_MASK(EC_QUEUE_WAVE_TRAP)|KFD_EC_MASK(EC_QUEUE_WAVE_MATH_ERROR)|KFD_EC_MASK(EC_QUEUE_WAVE_ILLEGAL_INSTRUCTION)|KFD_EC_MASK(EC_QUEUE_WAVE_MEMORY_VIOLATION)|KFD_EC_MASK(EC_QUEUE_WAVE_APERTURE_VIOLATION)|KFD_EC_MASK(EC_QUEUE_PACKET_DISPATCH_DIM_INVALID)|KFD_EC_MASK(EC_QUEUE_PACKET_DISPATCH_GROUP_SEGMENT_SIZE_INVALID)|KFD_EC_MASK(EC_QUEUE_PACKET_DISPATCH_CODE_INVALID)|KFD_EC_MASK(EC_QUEUE_PACKET_RESERVED)|KFD_EC_MASK(EC_QUEUE_PACKET_UNSUPPORTED)|KFD_EC_MASK(EC_QUEUE_PACKET_DISPATCH_WORK_GROUP_SIZE_INVALID)|KFD_EC_MASK(EC_QUEUE_PACKET_DISPATCH_REGISTER_INVALID)|KFD_EC_MASK(EC_QUEUE_PACKET_VENDOR_UNSUPPORTED)|KFD_EC_MASK(EC_QUEUE_PREEMPTION_ERROR)|KFD_EC_MASK(EC_QUEUE_NEW)) # macro
KFD_EC_MASK_DEVICE = (KFD_EC_MASK(EC_DEVICE_QUEUE_DELETE)|KFD_EC_MASK(EC_DEVICE_RAS_ERROR)|KFD_EC_MASK(EC_DEVICE_FATAL_HALT)|KFD_EC_MASK(EC_DEVICE_MEMORY_VIOLATION)|KFD_EC_MASK(EC_DEVICE_NEW)) # macro
KFD_EC_MASK_PROCESS = (KFD_EC_MASK(EC_PROCESS_RUNTIME)|KFD_EC_MASK(EC_PROCESS_DEVICE_REMOVE)) # macro
KFD_EC_MASK_PACKET = (KFD_EC_MASK(EC_QUEUE_PACKET_DISPATCH_DIM_INVALID)|KFD_EC_MASK(EC_QUEUE_PACKET_DISPATCH_GROUP_SEGMENT_SIZE_INVALID)|KFD_EC_MASK(EC_QUEUE_PACKET_DISPATCH_CODE_INVALID)|KFD_EC_MASK(EC_QUEUE_PACKET_RESERVED)|KFD_EC_MASK(EC_QUEUE_PACKET_UNSUPPORTED)|KFD_EC_MASK(EC_QUEUE_PACKET_DISPATCH_WORK_GROUP_SIZE_INVALID)|KFD_EC_MASK(EC_QUEUE_PACKET_DISPATCH_REGISTER_INVALID)|KFD_EC_MASK(EC_QUEUE_PACKET_VENDOR_UNSUPPORTED)) # macro
def KFD_DBG_EC_IS_VALID(ecode):  # macro
   return (ecode>EC_NONE and ecode<EC_MAX)
def KFD_DBG_EC_TYPE_IS_QUEUE(ecode):  # macro
   return (KFD_DBG_EC_IS_VALID(ecode) and not not (KFD_EC_MASK(ecode)&(KFD_EC_MASK(EC_QUEUE_WAVE_ABORT)|KFD_EC_MASK(EC_QUEUE_WAVE_TRAP)|KFD_EC_MASK(EC_QUEUE_WAVE_MATH_ERROR)|KFD_EC_MASK(EC_QUEUE_WAVE_ILLEGAL_INSTRUCTION)|KFD_EC_MASK(EC_QUEUE_WAVE_MEMORY_VIOLATION)|KFD_EC_MASK(EC_QUEUE_WAVE_APERTURE_VIOLATION)|KFD_EC_MASK(EC_QUEUE_PACKET_DISPATCH_DIM_INVALID)|KFD_EC_MASK(EC_QUEUE_PACKET_DISPATCH_GROUP_SEGMENT_SIZE_INVALID)|KFD_EC_MASK(EC_QUEUE_PACKET_DISPATCH_CODE_INVALID)|KFD_EC_MASK(EC_QUEUE_PACKET_RESERVED)|KFD_EC_MASK(EC_QUEUE_PACKET_UNSUPPORTED)|KFD_EC_MASK(EC_QUEUE_PACKET_DISPATCH_WORK_GROUP_SIZE_INVALID)|KFD_EC_MASK(EC_QUEUE_PACKET_DISPATCH_REGISTER_INVALID)|KFD_EC_MASK(EC_QUEUE_PACKET_VENDOR_UNSUPPORTED)|KFD_EC_MASK(EC_QUEUE_PREEMPTION_ERROR)|KFD_EC_MASK(EC_QUEUE_NEW))))
def KFD_DBG_EC_TYPE_IS_DEVICE(ecode):  # macro
   return (KFD_DBG_EC_IS_VALID(ecode) and not not (KFD_EC_MASK(ecode)&(KFD_EC_MASK(EC_DEVICE_QUEUE_DELETE)|KFD_EC_MASK(EC_DEVICE_RAS_ERROR)|KFD_EC_MASK(EC_DEVICE_FATAL_HALT)|KFD_EC_MASK(EC_DEVICE_MEMORY_VIOLATION)|KFD_EC_MASK(EC_DEVICE_NEW))))
def KFD_DBG_EC_TYPE_IS_PROCESS(ecode):  # macro
   return (KFD_DBG_EC_IS_VALID(ecode) and not not (KFD_EC_MASK(ecode)&(KFD_EC_MASK(EC_PROCESS_RUNTIME)|KFD_EC_MASK(EC_PROCESS_DEVICE_REMOVE))))
def KFD_DBG_EC_TYPE_IS_PACKET(ecode):  # macro
   return (KFD_DBG_EC_IS_VALID(ecode) and not not (KFD_EC_MASK(ecode)&(KFD_EC_MASK(EC_QUEUE_PACKET_DISPATCH_DIM_INVALID)|KFD_EC_MASK(EC_QUEUE_PACKET_DISPATCH_GROUP_SEGMENT_SIZE_INVALID)|KFD_EC_MASK(EC_QUEUE_PACKET_DISPATCH_CODE_INVALID)|KFD_EC_MASK(EC_QUEUE_PACKET_RESERVED)|KFD_EC_MASK(EC_QUEUE_PACKET_UNSUPPORTED)|KFD_EC_MASK(EC_QUEUE_PACKET_DISPATCH_WORK_GROUP_SIZE_INVALID)|KFD_EC_MASK(EC_QUEUE_PACKET_DISPATCH_REGISTER_INVALID)|KFD_EC_MASK(EC_QUEUE_PACKET_VENDOR_UNSUPPORTED))))

# values for enumeration 'kfd_dbg_runtime_state'
kfd_dbg_runtime_state__enumvalues = {
    0: 'DEBUG_RUNTIME_STATE_DISABLED',
    1: 'DEBUG_RUNTIME_STATE_ENABLED',
    2: 'DEBUG_RUNTIME_STATE_ENABLED_BUSY',
    3: 'DEBUG_RUNTIME_STATE_ENABLED_ERROR',
}
DEBUG_RUNTIME_STATE_DISABLED = 0
DEBUG_RUNTIME_STATE_ENABLED = 1
DEBUG_RUNTIME_STATE_ENABLED_BUSY = 2
DEBUG_RUNTIME_STATE_ENABLED_ERROR = 3
kfd_dbg_runtime_state = ctypes.c_uint32 # enum
class struct_kfd_runtime_info(Structure):
    pass

struct_kfd_runtime_info._pack_ = 1 # source:False
struct_kfd_runtime_info._fields_ = [
    ('r_debug', ctypes.c_uint64),
    ('runtime_state', ctypes.c_uint32),
    ('ttmp_setup', ctypes.c_uint32),
]

class struct_kfd_ioctl_runtime_enable_args(Structure):
    pass

struct_kfd_ioctl_runtime_enable_args._pack_ = 1 # source:False
struct_kfd_ioctl_runtime_enable_args._fields_ = [
    ('r_debug', ctypes.c_uint64),
    ('mode_mask', ctypes.c_uint32),
    ('capabilities_mask', ctypes.c_uint32),
]

AMDKFD_IOC_RUNTIME_ENABLE = AMDKFD_IOWR ( 0x25 , struct_kfd_ioctl_runtime_enable_args ) # macro (from list)
class struct_kfd_queue_snapshot_entry(Structure):
    pass

struct_kfd_queue_snapshot_entry._pack_ = 1 # source:False
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

class struct_kfd_context_save_area_header(Structure):
    pass

class struct_kfd_context_save_area_header_wave_state(Structure):
    pass

struct_kfd_context_save_area_header_wave_state._pack_ = 1 # source:False
struct_kfd_context_save_area_header_wave_state._fields_ = [
    ('control_stack_offset', ctypes.c_uint32),
    ('control_stack_size', ctypes.c_uint32),
    ('wave_state_offset', ctypes.c_uint32),
    ('wave_state_size', ctypes.c_uint32),
]

struct_kfd_context_save_area_header._pack_ = 1 # source:False
struct_kfd_context_save_area_header._fields_ = [
    ('wave_state', struct_kfd_context_save_area_header_wave_state),
    ('debug_offset', ctypes.c_uint32),
    ('debug_size', ctypes.c_uint32),
    ('err_payload_addr', ctypes.c_uint64),
    ('err_event_id', ctypes.c_uint32),
    ('reserved1', ctypes.c_uint32),
]


# values for enumeration 'kfd_dbg_trap_operations'
kfd_dbg_trap_operations__enumvalues = {
    0: 'KFD_IOC_DBG_TRAP_ENABLE',
    1: 'KFD_IOC_DBG_TRAP_DISABLE',
    2: 'KFD_IOC_DBG_TRAP_SEND_RUNTIME_EVENT',
    3: 'KFD_IOC_DBG_TRAP_SET_EXCEPTIONS_ENABLED',
    4: 'KFD_IOC_DBG_TRAP_SET_WAVE_LAUNCH_OVERRIDE',
    5: 'KFD_IOC_DBG_TRAP_SET_WAVE_LAUNCH_MODE',
    6: 'KFD_IOC_DBG_TRAP_SUSPEND_QUEUES',
    7: 'KFD_IOC_DBG_TRAP_RESUME_QUEUES',
    8: 'KFD_IOC_DBG_TRAP_SET_NODE_ADDRESS_WATCH',
    9: 'KFD_IOC_DBG_TRAP_CLEAR_NODE_ADDRESS_WATCH',
    10: 'KFD_IOC_DBG_TRAP_SET_FLAGS',
    11: 'KFD_IOC_DBG_TRAP_QUERY_DEBUG_EVENT',
    12: 'KFD_IOC_DBG_TRAP_QUERY_EXCEPTION_INFO',
    13: 'KFD_IOC_DBG_TRAP_GET_QUEUE_SNAPSHOT',
    14: 'KFD_IOC_DBG_TRAP_GET_DEVICE_SNAPSHOT',
}
KFD_IOC_DBG_TRAP_ENABLE = 0
KFD_IOC_DBG_TRAP_DISABLE = 1
KFD_IOC_DBG_TRAP_SEND_RUNTIME_EVENT = 2
KFD_IOC_DBG_TRAP_SET_EXCEPTIONS_ENABLED = 3
KFD_IOC_DBG_TRAP_SET_WAVE_LAUNCH_OVERRIDE = 4
KFD_IOC_DBG_TRAP_SET_WAVE_LAUNCH_MODE = 5
KFD_IOC_DBG_TRAP_SUSPEND_QUEUES = 6
KFD_IOC_DBG_TRAP_RESUME_QUEUES = 7
KFD_IOC_DBG_TRAP_SET_NODE_ADDRESS_WATCH = 8
KFD_IOC_DBG_TRAP_CLEAR_NODE_ADDRESS_WATCH = 9
KFD_IOC_DBG_TRAP_SET_FLAGS = 10
KFD_IOC_DBG_TRAP_QUERY_DEBUG_EVENT = 11
KFD_IOC_DBG_TRAP_QUERY_EXCEPTION_INFO = 12
KFD_IOC_DBG_TRAP_GET_QUEUE_SNAPSHOT = 13
KFD_IOC_DBG_TRAP_GET_DEVICE_SNAPSHOT = 14
kfd_dbg_trap_operations = ctypes.c_uint32 # enum
class struct_kfd_ioctl_dbg_trap_enable_args(Structure):
    pass

struct_kfd_ioctl_dbg_trap_enable_args._pack_ = 1 # source:False
struct_kfd_ioctl_dbg_trap_enable_args._fields_ = [
    ('exception_mask', ctypes.c_uint64),
    ('rinfo_ptr', ctypes.c_uint64),
    ('rinfo_size', ctypes.c_uint32),
    ('dbg_fd', ctypes.c_uint32),
]

class struct_kfd_ioctl_dbg_trap_send_runtime_event_args(Structure):
    pass

struct_kfd_ioctl_dbg_trap_send_runtime_event_args._pack_ = 1 # source:False
struct_kfd_ioctl_dbg_trap_send_runtime_event_args._fields_ = [
    ('exception_mask', ctypes.c_uint64),
    ('gpu_id', ctypes.c_uint32),
    ('queue_id', ctypes.c_uint32),
]

class struct_kfd_ioctl_dbg_trap_set_exceptions_enabled_args(Structure):
    pass

struct_kfd_ioctl_dbg_trap_set_exceptions_enabled_args._pack_ = 1 # source:False
struct_kfd_ioctl_dbg_trap_set_exceptions_enabled_args._fields_ = [
    ('exception_mask', ctypes.c_uint64),
]

class struct_kfd_ioctl_dbg_trap_set_wave_launch_override_args(Structure):
    pass

struct_kfd_ioctl_dbg_trap_set_wave_launch_override_args._pack_ = 1 # source:False
struct_kfd_ioctl_dbg_trap_set_wave_launch_override_args._fields_ = [
    ('override_mode', ctypes.c_uint32),
    ('enable_mask', ctypes.c_uint32),
    ('support_request_mask', ctypes.c_uint32),
    ('pad', ctypes.c_uint32),
]

class struct_kfd_ioctl_dbg_trap_set_wave_launch_mode_args(Structure):
    pass

struct_kfd_ioctl_dbg_trap_set_wave_launch_mode_args._pack_ = 1 # source:False
struct_kfd_ioctl_dbg_trap_set_wave_launch_mode_args._fields_ = [
    ('launch_mode', ctypes.c_uint32),
    ('pad', ctypes.c_uint32),
]

class struct_kfd_ioctl_dbg_trap_suspend_queues_args(Structure):
    pass

struct_kfd_ioctl_dbg_trap_suspend_queues_args._pack_ = 1 # source:False
struct_kfd_ioctl_dbg_trap_suspend_queues_args._fields_ = [
    ('exception_mask', ctypes.c_uint64),
    ('queue_array_ptr', ctypes.c_uint64),
    ('num_queues', ctypes.c_uint32),
    ('grace_period', ctypes.c_uint32),
]

class struct_kfd_ioctl_dbg_trap_resume_queues_args(Structure):
    pass

struct_kfd_ioctl_dbg_trap_resume_queues_args._pack_ = 1 # source:False
struct_kfd_ioctl_dbg_trap_resume_queues_args._fields_ = [
    ('queue_array_ptr', ctypes.c_uint64),
    ('num_queues', ctypes.c_uint32),
    ('pad', ctypes.c_uint32),
]

class struct_kfd_ioctl_dbg_trap_set_node_address_watch_args(Structure):
    pass

struct_kfd_ioctl_dbg_trap_set_node_address_watch_args._pack_ = 1 # source:False
struct_kfd_ioctl_dbg_trap_set_node_address_watch_args._fields_ = [
    ('address', ctypes.c_uint64),
    ('mode', ctypes.c_uint32),
    ('mask', ctypes.c_uint32),
    ('gpu_id', ctypes.c_uint32),
    ('id', ctypes.c_uint32),
]

class struct_kfd_ioctl_dbg_trap_clear_node_address_watch_args(Structure):
    pass

struct_kfd_ioctl_dbg_trap_clear_node_address_watch_args._pack_ = 1 # source:False
struct_kfd_ioctl_dbg_trap_clear_node_address_watch_args._fields_ = [
    ('gpu_id', ctypes.c_uint32),
    ('id', ctypes.c_uint32),
]

class struct_kfd_ioctl_dbg_trap_set_flags_args(Structure):
    pass

struct_kfd_ioctl_dbg_trap_set_flags_args._pack_ = 1 # source:False
struct_kfd_ioctl_dbg_trap_set_flags_args._fields_ = [
    ('flags', ctypes.c_uint32),
    ('pad', ctypes.c_uint32),
]

class struct_kfd_ioctl_dbg_trap_query_debug_event_args(Structure):
    pass

struct_kfd_ioctl_dbg_trap_query_debug_event_args._pack_ = 1 # source:False
struct_kfd_ioctl_dbg_trap_query_debug_event_args._fields_ = [
    ('exception_mask', ctypes.c_uint64),
    ('gpu_id', ctypes.c_uint32),
    ('queue_id', ctypes.c_uint32),
]

class struct_kfd_ioctl_dbg_trap_query_exception_info_args(Structure):
    pass

struct_kfd_ioctl_dbg_trap_query_exception_info_args._pack_ = 1 # source:False
struct_kfd_ioctl_dbg_trap_query_exception_info_args._fields_ = [
    ('info_ptr', ctypes.c_uint64),
    ('info_size', ctypes.c_uint32),
    ('source_id', ctypes.c_uint32),
    ('exception_code', ctypes.c_uint32),
    ('clear_exception', ctypes.c_uint32),
]

class struct_kfd_ioctl_dbg_trap_queue_snapshot_args(Structure):
    pass

struct_kfd_ioctl_dbg_trap_queue_snapshot_args._pack_ = 1 # source:False
struct_kfd_ioctl_dbg_trap_queue_snapshot_args._fields_ = [
    ('exception_mask', ctypes.c_uint64),
    ('snapshot_buf_ptr', ctypes.c_uint64),
    ('num_queues', ctypes.c_uint32),
    ('entry_size', ctypes.c_uint32),
]

class struct_kfd_ioctl_dbg_trap_device_snapshot_args(Structure):
    pass

struct_kfd_ioctl_dbg_trap_device_snapshot_args._pack_ = 1 # source:False
struct_kfd_ioctl_dbg_trap_device_snapshot_args._fields_ = [
    ('exception_mask', ctypes.c_uint64),
    ('snapshot_buf_ptr', ctypes.c_uint64),
    ('num_devices', ctypes.c_uint32),
    ('entry_size', ctypes.c_uint32),
]

class struct_kfd_ioctl_dbg_trap_args(Structure):
    pass

class union_kfd_ioctl_dbg_trap_args_0(Union):
    _pack_ = 1 # source:False
    _fields_ = [
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

struct_kfd_ioctl_dbg_trap_args._pack_ = 1 # source:False
struct_kfd_ioctl_dbg_trap_args._anonymous_ = ('_0',)
struct_kfd_ioctl_dbg_trap_args._fields_ = [
    ('pid', ctypes.c_uint32),
    ('op', ctypes.c_uint32),
    ('_0', union_kfd_ioctl_dbg_trap_args_0),
]

AMDKFD_IOC_DBG_TRAP = AMDKFD_IOWR ( 0x26 , struct_kfd_ioctl_dbg_trap_args ) # macro (from list)
__all__ = \
    ['AMDKFD_COMMAND_END', 'AMDKFD_COMMAND_START',
    'AMDKFD_IOCTL_BASE', 'DEBUG_RUNTIME_STATE_DISABLED',
    'DEBUG_RUNTIME_STATE_ENABLED', 'DEBUG_RUNTIME_STATE_ENABLED_BUSY',
    'DEBUG_RUNTIME_STATE_ENABLED_ERROR', 'EC_DEVICE_FATAL_HALT',
    'EC_DEVICE_MEMORY_VIOLATION', 'EC_DEVICE_NEW',
    'EC_DEVICE_QUEUE_DELETE', 'EC_DEVICE_RAS_ERROR', 'EC_MAX',
    'EC_NONE', 'EC_PROCESS_DEVICE_REMOVE', 'EC_PROCESS_RUNTIME',
    'EC_QUEUE_NEW', 'EC_QUEUE_PACKET_DISPATCH_CODE_INVALID',
    'EC_QUEUE_PACKET_DISPATCH_DIM_INVALID',
    'EC_QUEUE_PACKET_DISPATCH_GROUP_SEGMENT_SIZE_INVALID',
    'EC_QUEUE_PACKET_DISPATCH_REGISTER_INVALID',
    'EC_QUEUE_PACKET_DISPATCH_WORK_GROUP_SIZE_INVALID',
    'EC_QUEUE_PACKET_RESERVED', 'EC_QUEUE_PACKET_UNSUPPORTED',
    'EC_QUEUE_PACKET_VENDOR_UNSUPPORTED', 'EC_QUEUE_PREEMPTION_ERROR',
    'EC_QUEUE_WAVE_ABORT', 'EC_QUEUE_WAVE_APERTURE_VIOLATION',
    'EC_QUEUE_WAVE_ILLEGAL_INSTRUCTION', 'EC_QUEUE_WAVE_MATH_ERROR',
    'EC_QUEUE_WAVE_MEMORY_VIOLATION', 'EC_QUEUE_WAVE_TRAP',
    'KFD_CRIU_OP_CHECKPOINT', 'KFD_CRIU_OP_PROCESS_INFO',
    'KFD_CRIU_OP_RESTORE', 'KFD_CRIU_OP_RESUME',
    'KFD_CRIU_OP_UNPAUSE', 'KFD_DBG_QUEUE_ERROR_BIT',
    'KFD_DBG_QUEUE_ERROR_MASK', 'KFD_DBG_QUEUE_INVALID_BIT',
    'KFD_DBG_QUEUE_INVALID_MASK',
    'KFD_DBG_TRAP_ADDRESS_WATCH_MODE_ALL',
    'KFD_DBG_TRAP_ADDRESS_WATCH_MODE_ATOMIC',
    'KFD_DBG_TRAP_ADDRESS_WATCH_MODE_NONREAD',
    'KFD_DBG_TRAP_ADDRESS_WATCH_MODE_READ',
    'KFD_DBG_TRAP_FLAG_SINGLE_MEM_OP',
    'KFD_DBG_TRAP_MASK_DBG_ADDRESS_WATCH',
    'KFD_DBG_TRAP_MASK_DBG_MEMORY_VIOLATION',
    'KFD_DBG_TRAP_MASK_FP_DIVIDE_BY_ZERO',
    'KFD_DBG_TRAP_MASK_FP_INEXACT',
    'KFD_DBG_TRAP_MASK_FP_INPUT_DENORMAL',
    'KFD_DBG_TRAP_MASK_FP_INVALID', 'KFD_DBG_TRAP_MASK_FP_OVERFLOW',
    'KFD_DBG_TRAP_MASK_FP_UNDERFLOW',
    'KFD_DBG_TRAP_MASK_INT_DIVIDE_BY_ZERO',
    'KFD_DBG_TRAP_MASK_TRAP_ON_WAVE_END',
    'KFD_DBG_TRAP_MASK_TRAP_ON_WAVE_START',
    'KFD_DBG_TRAP_OVERRIDE_OR', 'KFD_DBG_TRAP_OVERRIDE_REPLACE',
    'KFD_DBG_TRAP_WAVE_LAUNCH_MODE_DEBUG',
    'KFD_DBG_TRAP_WAVE_LAUNCH_MODE_HALT',
    'KFD_DBG_TRAP_WAVE_LAUNCH_MODE_NORMAL', 'KFD_EC_MASK_DEVICE',
    'KFD_EC_MASK_PACKET', 'KFD_EC_MASK_PROCESS', 'KFD_EC_MASK_QUEUE',
    'KFD_HW_EXCEPTION_ECC', 'KFD_HW_EXCEPTION_GPU_HANG',
    'KFD_HW_EXCEPTION_PER_ENGINE_RESET',
    'KFD_HW_EXCEPTION_WHOLE_GPU_RESET', 'KFD_INVALID_FD',
    'KFD_IOCTL_H_INCLUDED', 'KFD_IOCTL_MAJOR_VERSION',
    'KFD_IOCTL_MINOR_VERSION', 'KFD_IOCTL_SVM_ATTR_ACCESS',
    'KFD_IOCTL_SVM_ATTR_ACCESS_IN_PLACE',
    'KFD_IOCTL_SVM_ATTR_CLR_FLAGS', 'KFD_IOCTL_SVM_ATTR_GRANULARITY',
    'KFD_IOCTL_SVM_ATTR_NO_ACCESS',
    'KFD_IOCTL_SVM_ATTR_PREFERRED_LOC',
    'KFD_IOCTL_SVM_ATTR_PREFETCH_LOC', 'KFD_IOCTL_SVM_ATTR_SET_FLAGS',
    'KFD_IOCTL_SVM_FLAG_COHERENT', 'KFD_IOCTL_SVM_FLAG_EXT_COHERENT',
    'KFD_IOCTL_SVM_FLAG_GPU_ALWAYS_MAPPED',
    'KFD_IOCTL_SVM_FLAG_GPU_EXEC',
    'KFD_IOCTL_SVM_FLAG_GPU_READ_MOSTLY', 'KFD_IOCTL_SVM_FLAG_GPU_RO',
    'KFD_IOCTL_SVM_FLAG_HIVE_LOCAL', 'KFD_IOCTL_SVM_FLAG_HOST_ACCESS',
    'KFD_IOCTL_SVM_LOCATION_SYSMEM',
    'KFD_IOCTL_SVM_LOCATION_UNDEFINED', 'KFD_IOCTL_SVM_OP_GET_ATTR',
    'KFD_IOCTL_SVM_OP_SET_ATTR',
    'KFD_IOC_ALLOC_MEM_FLAGS_AQL_QUEUE_MEM',
    'KFD_IOC_ALLOC_MEM_FLAGS_COHERENT',
    'KFD_IOC_ALLOC_MEM_FLAGS_DOORBELL',
    'KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE',
    'KFD_IOC_ALLOC_MEM_FLAGS_EXT_COHERENT',
    'KFD_IOC_ALLOC_MEM_FLAGS_GTT',
    'KFD_IOC_ALLOC_MEM_FLAGS_MMIO_REMAP',
    'KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE',
    'KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC',
    'KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED',
    'KFD_IOC_ALLOC_MEM_FLAGS_USERPTR', 'KFD_IOC_ALLOC_MEM_FLAGS_VRAM',
    'KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE',
    'KFD_IOC_CACHE_POLICY_COHERENT',
    'KFD_IOC_CACHE_POLICY_NONCOHERENT',
    'KFD_IOC_DBG_TRAP_CLEAR_NODE_ADDRESS_WATCH',
    'KFD_IOC_DBG_TRAP_DISABLE', 'KFD_IOC_DBG_TRAP_ENABLE',
    'KFD_IOC_DBG_TRAP_GET_DEVICE_SNAPSHOT',
    'KFD_IOC_DBG_TRAP_GET_QUEUE_SNAPSHOT',
    'KFD_IOC_DBG_TRAP_QUERY_DEBUG_EVENT',
    'KFD_IOC_DBG_TRAP_QUERY_EXCEPTION_INFO',
    'KFD_IOC_DBG_TRAP_RESUME_QUEUES',
    'KFD_IOC_DBG_TRAP_SEND_RUNTIME_EVENT',
    'KFD_IOC_DBG_TRAP_SET_EXCEPTIONS_ENABLED',
    'KFD_IOC_DBG_TRAP_SET_FLAGS',
    'KFD_IOC_DBG_TRAP_SET_NODE_ADDRESS_WATCH',
    'KFD_IOC_DBG_TRAP_SET_WAVE_LAUNCH_MODE',
    'KFD_IOC_DBG_TRAP_SET_WAVE_LAUNCH_OVERRIDE',
    'KFD_IOC_DBG_TRAP_SUSPEND_QUEUES', 'KFD_IOC_EVENT_DEBUG_EVENT',
    'KFD_IOC_EVENT_DEVICESTATECHANGE', 'KFD_IOC_EVENT_HW_EXCEPTION',
    'KFD_IOC_EVENT_MEMORY', 'KFD_IOC_EVENT_NODECHANGE',
    'KFD_IOC_EVENT_PROFILE_EVENT', 'KFD_IOC_EVENT_QUEUE_EVENT',
    'KFD_IOC_EVENT_SIGNAL', 'KFD_IOC_EVENT_SYSTEM_EVENT',
    'KFD_IOC_QUEUE_TYPE_COMPUTE', 'KFD_IOC_QUEUE_TYPE_COMPUTE_AQL',
    'KFD_IOC_QUEUE_TYPE_SDMA', 'KFD_IOC_QUEUE_TYPE_SDMA_XGMI',
    'KFD_IOC_WAIT_RESULT_COMPLETE', 'KFD_IOC_WAIT_RESULT_FAIL',
    'KFD_IOC_WAIT_RESULT_TIMEOUT', 'KFD_MAX_QUEUE_PERCENTAGE',
    'KFD_MAX_QUEUE_PRIORITY', 'KFD_MEM_ERR_GPU_HANG',
    'KFD_MEM_ERR_NO_RAS', 'KFD_MEM_ERR_POISON_CONSUMED',
    'KFD_MEM_ERR_SRAM_ECC', 'KFD_MIGRATE_TRIGGERS',
    'KFD_MIGRATE_TRIGGER_PAGEFAULT_CPU',
    'KFD_MIGRATE_TRIGGER_PAGEFAULT_GPU',
    'KFD_MIGRATE_TRIGGER_PREFETCH',
    'KFD_MIGRATE_TRIGGER_TTM_EVICTION',
    'KFD_MMIO_REMAP_HDP_MEM_FLUSH_CNTL',
    'KFD_MMIO_REMAP_HDP_REG_FLUSH_CNTL',
    'KFD_QUEUE_EVICTION_CRIU_CHECKPOINT',
    'KFD_QUEUE_EVICTION_CRIU_RESTORE', 'KFD_QUEUE_EVICTION_TRIGGERS',
    'KFD_QUEUE_EVICTION_TRIGGER_SUSPEND',
    'KFD_QUEUE_EVICTION_TRIGGER_SVM',
    'KFD_QUEUE_EVICTION_TRIGGER_TTM',
    'KFD_QUEUE_EVICTION_TRIGGER_USERPTR',
    'KFD_RUNTIME_ENABLE_MODE_ENABLE_MASK',
    'KFD_RUNTIME_ENABLE_MODE_TTMP_SAVE_MASK',
    'KFD_SIGNAL_EVENT_LIMIT', 'KFD_SMI_EVENT_ALL_PROCESS',
    'KFD_SMI_EVENT_GPU_POST_RESET', 'KFD_SMI_EVENT_GPU_PRE_RESET',
    'KFD_SMI_EVENT_MIGRATE_END', 'KFD_SMI_EVENT_MIGRATE_START',
    'KFD_SMI_EVENT_MSG_SIZE', 'KFD_SMI_EVENT_NONE',
    'KFD_SMI_EVENT_PAGE_FAULT_END', 'KFD_SMI_EVENT_PAGE_FAULT_START',
    'KFD_SMI_EVENT_QUEUE_EVICTION', 'KFD_SMI_EVENT_QUEUE_RESTORE',
    'KFD_SMI_EVENT_THERMAL_THROTTLE', 'KFD_SMI_EVENT_UNMAP_FROM_GPU',
    'KFD_SMI_EVENT_VMFAULT', 'KFD_SVM_UNMAP_TRIGGERS',
    'KFD_SVM_UNMAP_TRIGGER_MMU_NOTIFY',
    'KFD_SVM_UNMAP_TRIGGER_MMU_NOTIFY_MIGRATE',
    'KFD_SVM_UNMAP_TRIGGER_UNMAP_FROM_CPU',
    'MAX_ALLOWED_AW_BUFF_SIZE', 'MAX_ALLOWED_NUM_POINTS',
    'MAX_ALLOWED_WAC_BUFF_SIZE', 'NUM_OF_SUPPORTED_GPUS', '_IO',
    '_IOR', '_IOW', '_IOWR', 'kfd_criu_op', 'kfd_dbg_runtime_state',
    'kfd_dbg_trap_address_watch_mode', 'kfd_dbg_trap_exception_code',
    'kfd_dbg_trap_flags', 'kfd_dbg_trap_mask',
    'kfd_dbg_trap_operations', 'kfd_dbg_trap_override_mode',
    'kfd_dbg_trap_wave_launch_mode', 'kfd_ioctl_svm_attr_type',
    'kfd_ioctl_svm_location', 'kfd_ioctl_svm_op', 'kfd_mmio_remap',
    'kfd_smi_event', 'struct_kfd_context_save_area_header',
    'struct_kfd_context_save_area_header_wave_state',
    'struct_kfd_criu_bo_bucket', 'struct_kfd_criu_device_bucket',
    'struct_kfd_dbg_device_info_entry', 'struct_kfd_event_data',
    'struct_kfd_hsa_hw_exception_data',
    'struct_kfd_hsa_memory_exception_data',
    'struct_kfd_hsa_signal_event_data',
    'struct_kfd_ioctl_acquire_vm_args',
    'struct_kfd_ioctl_alloc_memory_of_gpu_args',
    'struct_kfd_ioctl_alloc_queue_gws_args',
    'struct_kfd_ioctl_create_event_args',
    'struct_kfd_ioctl_create_queue_args',
    'struct_kfd_ioctl_criu_args',
    'struct_kfd_ioctl_dbg_address_watch_args',
    'struct_kfd_ioctl_dbg_register_args',
    'struct_kfd_ioctl_dbg_trap_args',
    'struct_kfd_ioctl_dbg_trap_clear_node_address_watch_args',
    'struct_kfd_ioctl_dbg_trap_device_snapshot_args',
    'struct_kfd_ioctl_dbg_trap_enable_args',
    'struct_kfd_ioctl_dbg_trap_query_debug_event_args',
    'struct_kfd_ioctl_dbg_trap_query_exception_info_args',
    'struct_kfd_ioctl_dbg_trap_queue_snapshot_args',
    'struct_kfd_ioctl_dbg_trap_resume_queues_args',
    'struct_kfd_ioctl_dbg_trap_send_runtime_event_args',
    'struct_kfd_ioctl_dbg_trap_set_exceptions_enabled_args',
    'struct_kfd_ioctl_dbg_trap_set_flags_args',
    'struct_kfd_ioctl_dbg_trap_set_node_address_watch_args',
    'struct_kfd_ioctl_dbg_trap_set_wave_launch_mode_args',
    'struct_kfd_ioctl_dbg_trap_set_wave_launch_override_args',
    'struct_kfd_ioctl_dbg_trap_suspend_queues_args',
    'struct_kfd_ioctl_dbg_unregister_args',
    'struct_kfd_ioctl_dbg_wave_control_args',
    'struct_kfd_ioctl_destroy_event_args',
    'struct_kfd_ioctl_destroy_queue_args',
    'struct_kfd_ioctl_export_dmabuf_args',
    'struct_kfd_ioctl_free_memory_of_gpu_args',
    'struct_kfd_ioctl_get_available_memory_args',
    'struct_kfd_ioctl_get_clock_counters_args',
    'struct_kfd_ioctl_get_dmabuf_info_args',
    'struct_kfd_ioctl_get_process_apertures_args',
    'struct_kfd_ioctl_get_process_apertures_new_args',
    'struct_kfd_ioctl_get_queue_wave_state_args',
    'struct_kfd_ioctl_get_tile_config_args',
    'struct_kfd_ioctl_get_version_args',
    'struct_kfd_ioctl_import_dmabuf_args',
    'struct_kfd_ioctl_map_memory_to_gpu_args',
    'struct_kfd_ioctl_reset_event_args',
    'struct_kfd_ioctl_runtime_enable_args',
    'struct_kfd_ioctl_set_cu_mask_args',
    'struct_kfd_ioctl_set_event_args',
    'struct_kfd_ioctl_set_memory_policy_args',
    'struct_kfd_ioctl_set_scratch_backing_va_args',
    'struct_kfd_ioctl_set_trap_handler_args',
    'struct_kfd_ioctl_set_xnack_mode_args',
    'struct_kfd_ioctl_smi_events_args', 'struct_kfd_ioctl_svm_args',
    'struct_kfd_ioctl_svm_attribute',
    'struct_kfd_ioctl_unmap_memory_from_gpu_args',
    'struct_kfd_ioctl_update_queue_args',
    'struct_kfd_ioctl_wait_events_args',
    'struct_kfd_memory_exception_failure',
    'struct_kfd_process_device_apertures',
    'struct_kfd_queue_snapshot_entry', 'struct_kfd_runtime_info',
    'union_kfd_event_data_0', 'union_kfd_ioctl_dbg_trap_args_0']
