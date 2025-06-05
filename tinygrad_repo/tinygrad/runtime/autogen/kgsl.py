# mypy: ignore-errors
# -*- coding: utf-8 -*-
#
# TARGET arch is: []
# WORD_SIZE is: 8
# POINTER_SIZE is: 8
# LONGDOUBLE_SIZE is: 16
#
import ctypes, os


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



c_int128 = ctypes.c_ubyte*16
c_uint128 = c_int128
void = None
if ctypes.sizeof(ctypes.c_longdouble) == 16:
    c_long_double_t = ctypes.c_longdouble
else:
    c_long_double_t = ctypes.c_ubyte*16


import fcntl, functools

def _do_ioctl(__idir, __base, __nr, __user_struct, __fd, __payload=None, **kwargs):
  ret = __fd.ioctl((__idir<<30) | (ctypes.sizeof(made := (__payload or __user_struct(**kwargs)))<<16) | (__base<<8) | __nr, made)
  if ret != 0: raise RuntimeError(f"ioctl returned {ret}")
  return made

def _IO(base, nr): return functools.partial(_do_ioctl, 0, ord(base) if isinstance(base, str) else base, nr, None)
def _IOW(base, nr, type): return functools.partial(_do_ioctl, 1, ord(base) if isinstance(base, str) else base, nr, type)
def _IOR(base, nr, type): return functools.partial(_do_ioctl, 2, ord(base) if isinstance(base, str) else base, nr, type)
def _IOWR(base, nr, type): return functools.partial(_do_ioctl, 3, ord(base) if isinstance(base, str) else base, nr, type)



_UAPI_MSM_KGSL_H = True # macro
size_t = True # macro
uint64_t = True # macro
KGSL_VERSION_MAJOR = 3 # macro
KGSL_VERSION_MINOR = 14 # macro
KGSL_CONTEXT_SAVE_GMEM = 0x00000001 # macro
KGSL_CONTEXT_NO_GMEM_ALLOC = 0x00000002 # macro
KGSL_CONTEXT_SUBMIT_IB_LIST = 0x00000004 # macro
KGSL_CONTEXT_CTX_SWITCH = 0x00000008 # macro
KGSL_CONTEXT_PREAMBLE = 0x00000010 # macro
KGSL_CONTEXT_TRASH_STATE = 0x00000020 # macro
KGSL_CONTEXT_PER_CONTEXT_TS = 0x00000040 # macro
KGSL_CONTEXT_USER_GENERATED_TS = 0x00000080 # macro
KGSL_CONTEXT_END_OF_FRAME = 0x00000100 # macro
KGSL_CONTEXT_NO_FAULT_TOLERANCE = 0x00000200 # macro
KGSL_CONTEXT_SYNC = 0x00000400 # macro
KGSL_CONTEXT_PWR_CONSTRAINT = 0x00000800 # macro
KGSL_CONTEXT_PRIORITY_MASK = 0x0000F000 # macro
KGSL_CONTEXT_PRIORITY_SHIFT = 12 # macro
KGSL_CONTEXT_PRIORITY_UNDEF = 0 # macro
KGSL_CONTEXT_IFH_NOP = 0x00010000 # macro
KGSL_CONTEXT_SECURE = 0x00020000 # macro
KGSL_CONTEXT_PREEMPT_STYLE_MASK = 0x0E000000 # macro
KGSL_CONTEXT_PREEMPT_STYLE_SHIFT = 25 # macro
KGSL_CONTEXT_PREEMPT_STYLE_DEFAULT = 0x0 # macro
KGSL_CONTEXT_PREEMPT_STYLE_RINGBUFFER = 0x1 # macro
KGSL_CONTEXT_PREEMPT_STYLE_FINEGRAIN = 0x2 # macro
KGSL_CONTEXT_TYPE_MASK = 0x01F00000 # macro
KGSL_CONTEXT_TYPE_SHIFT = 20 # macro
KGSL_CONTEXT_TYPE_ANY = 0 # macro
KGSL_CONTEXT_TYPE_GL = 1 # macro
KGSL_CONTEXT_TYPE_CL = 2 # macro
KGSL_CONTEXT_TYPE_C2D = 3 # macro
KGSL_CONTEXT_TYPE_RS = 4 # macro
KGSL_CONTEXT_TYPE_UNKNOWN = 0x1E # macro
KGSL_CONTEXT_INVALID = 0xffffffff # macro
KGSL_CMDBATCH_MEMLIST = 0x00000001 # macro
KGSL_CMDBATCH_MARKER = 0x00000002 # macro
KGSL_CMDBATCH_SUBMIT_IB_LIST = 0x00000004 # macro
KGSL_CMDBATCH_CTX_SWITCH = 0x00000008 # macro
KGSL_CMDBATCH_PROFILING = 0x00000010 # macro
KGSL_CMDBATCH_PROFILING_KTIME = 0x00000020 # macro
KGSL_CMDBATCH_END_OF_FRAME = 0x00000100 # macro
KGSL_CMDBATCH_SYNC = 0x00000400 # macro
KGSL_CMDBATCH_PWR_CONSTRAINT = 0x00000800 # macro
KGSL_CMDLIST_IB = 0x00000001 # macro
KGSL_CMDLIST_CTXTSWITCH_PREAMBLE = 0x00000002 # macro
KGSL_CMDLIST_IB_PREAMBLE = 0x00000004 # macro
KGSL_OBJLIST_MEMOBJ = 0x00000008 # macro
KGSL_OBJLIST_PROFILE = 0x00000010 # macro
KGSL_CMD_SYNCPOINT_TYPE_TIMESTAMP = 0 # macro
KGSL_CMD_SYNCPOINT_TYPE_FENCE = 1 # macro
KGSL_MEMFLAGS_SECURE = 0x00000008 # macro
KGSL_MEMFLAGS_GPUREADONLY = 0x01000000 # macro
KGSL_MEMFLAGS_GPUWRITEONLY = 0x02000000 # macro
KGSL_MEMFLAGS_FORCE_32BIT = 0x100000000 # macro
KGSL_CACHEMODE_MASK = 0x0C000000 # macro
KGSL_CACHEMODE_SHIFT = 26 # macro
KGSL_CACHEMODE_WRITECOMBINE = 0 # macro
KGSL_CACHEMODE_UNCACHED = 1 # macro
KGSL_CACHEMODE_WRITETHROUGH = 2 # macro
KGSL_CACHEMODE_WRITEBACK = 3 # macro
KGSL_MEMFLAGS_USE_CPU_MAP = 0x10000000 # macro
KGSL_MEMTYPE_MASK = 0x0000FF00 # macro
KGSL_MEMTYPE_SHIFT = 8 # macro
KGSL_MEMTYPE_OBJECTANY = 0 # macro
KGSL_MEMTYPE_FRAMEBUFFER = 1 # macro
KGSL_MEMTYPE_RENDERBUFFER = 2 # macro
KGSL_MEMTYPE_ARRAYBUFFER = 3 # macro
KGSL_MEMTYPE_ELEMENTARRAYBUFFER = 4 # macro
KGSL_MEMTYPE_VERTEXARRAYBUFFER = 5 # macro
KGSL_MEMTYPE_TEXTURE = 6 # macro
KGSL_MEMTYPE_SURFACE = 7 # macro
KGSL_MEMTYPE_EGL_SURFACE = 8 # macro
KGSL_MEMTYPE_GL = 9 # macro
KGSL_MEMTYPE_CL = 10 # macro
KGSL_MEMTYPE_CL_BUFFER_MAP = 11 # macro
KGSL_MEMTYPE_CL_BUFFER_NOMAP = 12 # macro
KGSL_MEMTYPE_CL_IMAGE_MAP = 13 # macro
KGSL_MEMTYPE_CL_IMAGE_NOMAP = 14 # macro
KGSL_MEMTYPE_CL_KERNEL_STACK = 15 # macro
KGSL_MEMTYPE_COMMAND = 16 # macro
KGSL_MEMTYPE_2D = 17 # macro
KGSL_MEMTYPE_EGL_IMAGE = 18 # macro
KGSL_MEMTYPE_EGL_SHADOW = 19 # macro
KGSL_MEMTYPE_MULTISAMPLE = 20 # macro
KGSL_MEMTYPE_KERNEL = 255 # macro
KGSL_MEMALIGN_MASK = 0x00FF0000 # macro
KGSL_MEMALIGN_SHIFT = 16 # macro
KGSL_MEMFLAGS_USERMEM_MASK = 0x000000e0 # macro
KGSL_MEMFLAGS_USERMEM_SHIFT = 5 # macro
def KGSL_USERMEM_FLAG(x):  # macro
   return (((x)+1)<<5)
KGSL_MEMFLAGS_NOT_USERMEM = 0 # macro
KGSL_FLAGS_NORMALMODE = 0x00000000 # macro
KGSL_FLAGS_SAFEMODE = 0x00000001 # macro
KGSL_FLAGS_INITIALIZED0 = 0x00000002 # macro
KGSL_FLAGS_INITIALIZED = 0x00000004 # macro
KGSL_FLAGS_STARTED = 0x00000008 # macro
KGSL_FLAGS_ACTIVE = 0x00000010 # macro
KGSL_FLAGS_RESERVED0 = 0x00000020 # macro
KGSL_FLAGS_RESERVED1 = 0x00000040 # macro
KGSL_FLAGS_RESERVED2 = 0x00000080 # macro
KGSL_FLAGS_SOFT_RESET = 0x00000100 # macro
KGSL_FLAGS_PER_CONTEXT_TIMESTAMPS = 0x00000200 # macro
KGSL_SYNCOBJ_SERVER_TIMEOUT = 2000 # macro
def KGSL_CONVERT_TO_MBPS(val):  # macro
   return (val*1000*1000)
KGSL_PROP_DEVICE_INFO = 0x1 # macro
KGSL_PROP_DEVICE_SHADOW = 0x2 # macro
KGSL_PROP_DEVICE_POWER = 0x3 # macro
KGSL_PROP_SHMEM = 0x4 # macro
KGSL_PROP_SHMEM_APERTURES = 0x5 # macro
KGSL_PROP_MMU_ENABLE = 0x6 # macro
KGSL_PROP_INTERRUPT_WAITS = 0x7 # macro
KGSL_PROP_VERSION = 0x8 # macro
KGSL_PROP_GPU_RESET_STAT = 0x9 # macro
KGSL_PROP_PWRCTRL = 0xE # macro
KGSL_PROP_PWR_CONSTRAINT = 0x12 # macro
KGSL_PROP_UCHE_GMEM_VADDR = 0x13 # macro
KGSL_PROP_SP_GENERIC_MEM = 0x14 # macro
KGSL_PROP_UCODE_VERSION = 0x15 # macro
KGSL_PROP_GPMU_VERSION = 0x16 # macro
KGSL_PROP_DEVICE_BITNESS = 0x18 # macro
KGSL_PERFCOUNTER_GROUP_CP = 0x0 # macro
KGSL_PERFCOUNTER_GROUP_RBBM = 0x1 # macro
KGSL_PERFCOUNTER_GROUP_PC = 0x2 # macro
KGSL_PERFCOUNTER_GROUP_VFD = 0x3 # macro
KGSL_PERFCOUNTER_GROUP_HLSQ = 0x4 # macro
KGSL_PERFCOUNTER_GROUP_VPC = 0x5 # macro
KGSL_PERFCOUNTER_GROUP_TSE = 0x6 # macro
KGSL_PERFCOUNTER_GROUP_RAS = 0x7 # macro
KGSL_PERFCOUNTER_GROUP_UCHE = 0x8 # macro
KGSL_PERFCOUNTER_GROUP_TP = 0x9 # macro
KGSL_PERFCOUNTER_GROUP_SP = 0xA # macro
KGSL_PERFCOUNTER_GROUP_RB = 0xB # macro
KGSL_PERFCOUNTER_GROUP_PWR = 0xC # macro
KGSL_PERFCOUNTER_GROUP_VBIF = 0xD # macro
KGSL_PERFCOUNTER_GROUP_VBIF_PWR = 0xE # macro
KGSL_PERFCOUNTER_GROUP_MH = 0xF # macro
KGSL_PERFCOUNTER_GROUP_PA_SU = 0x10 # macro
KGSL_PERFCOUNTER_GROUP_SQ = 0x11 # macro
KGSL_PERFCOUNTER_GROUP_SX = 0x12 # macro
KGSL_PERFCOUNTER_GROUP_TCF = 0x13 # macro
KGSL_PERFCOUNTER_GROUP_TCM = 0x14 # macro
KGSL_PERFCOUNTER_GROUP_TCR = 0x15 # macro
KGSL_PERFCOUNTER_GROUP_L2 = 0x16 # macro
KGSL_PERFCOUNTER_GROUP_VSC = 0x17 # macro
KGSL_PERFCOUNTER_GROUP_CCU = 0x18 # macro
KGSL_PERFCOUNTER_GROUP_LRZ = 0x19 # macro
KGSL_PERFCOUNTER_GROUP_CMP = 0x1A # macro
KGSL_PERFCOUNTER_GROUP_ALWAYSON = 0x1B # macro
KGSL_PERFCOUNTER_GROUP_SP_PWR = 0x1C # macro
KGSL_PERFCOUNTER_GROUP_TP_PWR = 0x1D # macro
KGSL_PERFCOUNTER_GROUP_RB_PWR = 0x1E # macro
KGSL_PERFCOUNTER_GROUP_CCU_PWR = 0x1F # macro
KGSL_PERFCOUNTER_GROUP_UCHE_PWR = 0x20 # macro
KGSL_PERFCOUNTER_GROUP_CP_PWR = 0x21 # macro
KGSL_PERFCOUNTER_GROUP_GPMU_PWR = 0x22 # macro
KGSL_PERFCOUNTER_GROUP_ALWAYSON_PWR = 0x23 # macro
KGSL_PERFCOUNTER_GROUP_MAX = 0x24 # macro
KGSL_PERFCOUNTER_NOT_USED = 0xFFFFFFFF # macro
KGSL_PERFCOUNTER_BROKEN = 0xFFFFFFFE # macro
KGSL_IOC_TYPE = 0x09 # macro
KGSL_TIMESTAMP_EVENT_GENLOCK = 1 # macro
KGSL_TIMESTAMP_EVENT_FENCE = 2 # macro
KGSL_GPUMEM_CACHE_CLEAN = (1<<0) # macro
KGSL_GPUMEM_CACHE_TO_GPU = (1<<0) # macro
KGSL_GPUMEM_CACHE_INV = (1<<1) # macro
KGSL_GPUMEM_CACHE_FROM_GPU = (1<<1) # macro
KGSL_GPUMEM_CACHE_FLUSH = ((1<<0)|(1<<1)) # macro
KGSL_GPUMEM_CACHE_RANGE = (1<<31) # macro
KGSL_IBDESC_MEMLIST = 0x1 # macro
KGSL_IBDESC_PROFILING_BUFFER = 0x2 # macro
KGSL_CONSTRAINT_NONE = 0 # macro
KGSL_CONSTRAINT_PWRLEVEL = 1 # macro
KGSL_CONSTRAINT_PWR_MIN = 0 # macro
KGSL_CONSTRAINT_PWR_MAX = 1 # macro
KGSL_GPUOBJ_ALLOC_METADATA_MAX = 64 # macro
KGSL_GPUOBJ_FREE_ON_EVENT = 1 # macro
KGSL_GPU_EVENT_TIMESTAMP = 1 # macro
KGSL_GPU_EVENT_FENCE = 2 # macro
KGSL_GPUOBJ_SET_INFO_METADATA = (1<<0) # macro
KGSL_GPUOBJ_SET_INFO_TYPE = (1<<1) # macro

# values for enumeration 'kgsl_user_mem_type'
kgsl_user_mem_type__enumvalues = {
    0: 'KGSL_USER_MEM_TYPE_PMEM',
    1: 'KGSL_USER_MEM_TYPE_ASHMEM',
    2: 'KGSL_USER_MEM_TYPE_ADDR',
    3: 'KGSL_USER_MEM_TYPE_ION',
    3: 'KGSL_USER_MEM_TYPE_DMABUF',
    7: 'KGSL_USER_MEM_TYPE_MAX',
}
KGSL_USER_MEM_TYPE_PMEM = 0
KGSL_USER_MEM_TYPE_ASHMEM = 1
KGSL_USER_MEM_TYPE_ADDR = 2
KGSL_USER_MEM_TYPE_ION = 3
KGSL_USER_MEM_TYPE_DMABUF = 3
KGSL_USER_MEM_TYPE_MAX = 7
kgsl_user_mem_type = ctypes.c_uint32 # enum
KGSL_MEMFLAGS_USERMEM_PMEM = KGSL_USERMEM_FLAG ( KGSL_USER_MEM_TYPE_PMEM ) # macro (from list)
KGSL_MEMFLAGS_USERMEM_ASHMEM = KGSL_USERMEM_FLAG ( KGSL_USER_MEM_TYPE_ASHMEM ) # macro (from list)
KGSL_MEMFLAGS_USERMEM_ADDR = KGSL_USERMEM_FLAG ( KGSL_USER_MEM_TYPE_ADDR ) # macro (from list)
KGSL_MEMFLAGS_USERMEM_ION = KGSL_USERMEM_FLAG ( KGSL_USER_MEM_TYPE_ION ) # macro (from list)

# values for enumeration 'kgsl_ctx_reset_stat'
kgsl_ctx_reset_stat__enumvalues = {
    0: 'KGSL_CTX_STAT_NO_ERROR',
    1: 'KGSL_CTX_STAT_GUILTY_CONTEXT_RESET_EXT',
    2: 'KGSL_CTX_STAT_INNOCENT_CONTEXT_RESET_EXT',
    3: 'KGSL_CTX_STAT_UNKNOWN_CONTEXT_RESET_EXT',
}
KGSL_CTX_STAT_NO_ERROR = 0
KGSL_CTX_STAT_GUILTY_CONTEXT_RESET_EXT = 1
KGSL_CTX_STAT_INNOCENT_CONTEXT_RESET_EXT = 2
KGSL_CTX_STAT_UNKNOWN_CONTEXT_RESET_EXT = 3
kgsl_ctx_reset_stat = ctypes.c_uint32 # enum

# values for enumeration 'kgsl_deviceid'
kgsl_deviceid__enumvalues = {
    0: 'KGSL_DEVICE_3D0',
    1: 'KGSL_DEVICE_MAX',
}
KGSL_DEVICE_3D0 = 0
KGSL_DEVICE_MAX = 1
kgsl_deviceid = ctypes.c_uint32 # enum
class struct_kgsl_devinfo(Structure):
    pass

struct_kgsl_devinfo._pack_ = 1 # source:False
struct_kgsl_devinfo._fields_ = [
    ('device_id', ctypes.c_uint32),
    ('chip_id', ctypes.c_uint32),
    ('mmu_enabled', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('gmem_gpubaseaddr', ctypes.c_uint64),
    ('gpu_id', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('gmem_sizebytes', ctypes.c_uint64),
]

class struct_kgsl_devmemstore(Structure):
    pass

struct_kgsl_devmemstore._pack_ = 1 # source:False
struct_kgsl_devmemstore._fields_ = [
    ('soptimestamp', ctypes.c_uint32),
    ('sbz', ctypes.c_uint32),
    ('eoptimestamp', ctypes.c_uint32),
    ('sbz2', ctypes.c_uint32),
    ('preempted', ctypes.c_uint32),
    ('sbz3', ctypes.c_uint32),
    ('ref_wait_ts', ctypes.c_uint32),
    ('sbz4', ctypes.c_uint32),
    ('current_context', ctypes.c_uint32),
    ('sbz5', ctypes.c_uint32),
]

# def KGSL_MEMSTORE_OFFSET(ctxt_id, field):  # macro
#    return ((ctxt_id)*ctypes.sizeof(struct_kgsl_devmemstore)+offsetof(struct_kgsl_devmemstore,field))

# values for enumeration 'kgsl_timestamp_type'
kgsl_timestamp_type__enumvalues = {
    1: 'KGSL_TIMESTAMP_CONSUMED',
    2: 'KGSL_TIMESTAMP_RETIRED',
    3: 'KGSL_TIMESTAMP_QUEUED',
}
KGSL_TIMESTAMP_CONSUMED = 1
KGSL_TIMESTAMP_RETIRED = 2
KGSL_TIMESTAMP_QUEUED = 3
kgsl_timestamp_type = ctypes.c_uint32 # enum
class struct_kgsl_shadowprop(Structure):
    pass

struct_kgsl_shadowprop._pack_ = 1 # source:False
struct_kgsl_shadowprop._fields_ = [
    ('gpuaddr', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
    ('flags', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

class struct_kgsl_version(Structure):
    pass

struct_kgsl_version._pack_ = 1 # source:False
struct_kgsl_version._fields_ = [
    ('drv_major', ctypes.c_uint32),
    ('drv_minor', ctypes.c_uint32),
    ('dev_major', ctypes.c_uint32),
    ('dev_minor', ctypes.c_uint32),
]

class struct_kgsl_sp_generic_mem(Structure):
    pass

struct_kgsl_sp_generic_mem._pack_ = 1 # source:False
struct_kgsl_sp_generic_mem._fields_ = [
    ('local', ctypes.c_uint64),
    ('pvt', ctypes.c_uint64),
]

class struct_kgsl_ucode_version(Structure):
    pass

struct_kgsl_ucode_version._pack_ = 1 # source:False
struct_kgsl_ucode_version._fields_ = [
    ('pfp', ctypes.c_uint32),
    ('pm4', ctypes.c_uint32),
]

class struct_kgsl_gpmu_version(Structure):
    pass

struct_kgsl_gpmu_version._pack_ = 1 # source:False
struct_kgsl_gpmu_version._fields_ = [
    ('major', ctypes.c_uint32),
    ('minor', ctypes.c_uint32),
    ('features', ctypes.c_uint32),
]

class struct_kgsl_ibdesc(Structure):
    pass

struct_kgsl_ibdesc._pack_ = 1 # source:False
struct_kgsl_ibdesc._fields_ = [
    ('gpuaddr', ctypes.c_uint64),
    ('__pad', ctypes.c_uint64),
    ('sizedwords', ctypes.c_uint64),
    ('ctrl', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

class struct_kgsl_cmdbatch_profiling_buffer(Structure):
    pass

struct_kgsl_cmdbatch_profiling_buffer._pack_ = 1 # source:False
struct_kgsl_cmdbatch_profiling_buffer._fields_ = [
    ('wall_clock_s', ctypes.c_uint64),
    ('wall_clock_ns', ctypes.c_uint64),
    ('gpu_ticks_queued', ctypes.c_uint64),
    ('gpu_ticks_submitted', ctypes.c_uint64),
    ('gpu_ticks_retired', ctypes.c_uint64),
]

class struct_kgsl_device_getproperty(Structure):
    pass

struct_kgsl_device_getproperty._pack_ = 1 # source:False
struct_kgsl_device_getproperty._fields_ = [
    ('type', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('value', ctypes.POINTER(None)),
    ('sizebytes', ctypes.c_uint64),
]

IOCTL_KGSL_DEVICE_GETPROPERTY = _IOWR ( 0x09 , 0x2 , struct_kgsl_device_getproperty ) # macro (from list)
IOCTL_KGSL_SETPROPERTY = _IOW ( 0x09 , 0x32 , struct_kgsl_device_getproperty ) # macro (from list)
class struct_kgsl_device_waittimestamp(Structure):
    pass

struct_kgsl_device_waittimestamp._pack_ = 1 # source:False
struct_kgsl_device_waittimestamp._fields_ = [
    ('timestamp', ctypes.c_uint32),
    ('timeout', ctypes.c_uint32),
]

IOCTL_KGSL_DEVICE_WAITTIMESTAMP = _IOW ( 0x09 , 0x6 , struct_kgsl_device_waittimestamp ) # macro (from list)
class struct_kgsl_device_waittimestamp_ctxtid(Structure):
    pass

struct_kgsl_device_waittimestamp_ctxtid._pack_ = 1 # source:False
struct_kgsl_device_waittimestamp_ctxtid._fields_ = [
    ('context_id', ctypes.c_uint32),
    ('timestamp', ctypes.c_uint32),
    ('timeout', ctypes.c_uint32),
]

IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID = _IOW ( 0x09 , 0x7 , struct_kgsl_device_waittimestamp_ctxtid ) # macro (from list)
class struct_kgsl_ringbuffer_issueibcmds(Structure):
    pass

struct_kgsl_ringbuffer_issueibcmds._pack_ = 1 # source:False
struct_kgsl_ringbuffer_issueibcmds._fields_ = [
    ('drawctxt_id', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('ibdesc_addr', ctypes.c_uint64),
    ('numibs', ctypes.c_uint32),
    ('timestamp', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

IOCTL_KGSL_RINGBUFFER_ISSUEIBCMDS = _IOWR ( 0x09 , 0x10 , struct_kgsl_ringbuffer_issueibcmds ) # macro (from list)
class struct_kgsl_cmdstream_readtimestamp(Structure):
    pass

struct_kgsl_cmdstream_readtimestamp._pack_ = 1 # source:False
struct_kgsl_cmdstream_readtimestamp._fields_ = [
    ('type', ctypes.c_uint32),
    ('timestamp', ctypes.c_uint32),
]

IOCTL_KGSL_CMDSTREAM_READTIMESTAMP_OLD = _IOR ( 0x09 , 0x11 , struct_kgsl_cmdstream_readtimestamp ) # macro (from list)
IOCTL_KGSL_CMDSTREAM_READTIMESTAMP = _IOWR ( 0x09 , 0x11 , struct_kgsl_cmdstream_readtimestamp ) # macro (from list)
class struct_kgsl_cmdstream_freememontimestamp(Structure):
    pass

struct_kgsl_cmdstream_freememontimestamp._pack_ = 1 # source:False
struct_kgsl_cmdstream_freememontimestamp._fields_ = [
    ('gpuaddr', ctypes.c_uint64),
    ('type', ctypes.c_uint32),
    ('timestamp', ctypes.c_uint32),
]

IOCTL_KGSL_CMDSTREAM_FREEMEMONTIMESTAMP = _IOW ( 0x09 , 0x12 , struct_kgsl_cmdstream_freememontimestamp ) # macro (from list)
IOCTL_KGSL_CMDSTREAM_FREEMEMONTIMESTAMP_OLD = _IOR ( 0x09 , 0x12 , struct_kgsl_cmdstream_freememontimestamp ) # macro (from list)
class struct_kgsl_drawctxt_create(Structure):
    pass

struct_kgsl_drawctxt_create._pack_ = 1 # source:False
struct_kgsl_drawctxt_create._fields_ = [
    ('flags', ctypes.c_uint32),
    ('drawctxt_id', ctypes.c_uint32),
]

IOCTL_KGSL_DRAWCTXT_CREATE = _IOWR ( 0x09 , 0x13 , struct_kgsl_drawctxt_create ) # macro (from list)
class struct_kgsl_drawctxt_destroy(Structure):
    pass

struct_kgsl_drawctxt_destroy._pack_ = 1 # source:False
struct_kgsl_drawctxt_destroy._fields_ = [
    ('drawctxt_id', ctypes.c_uint32),
]

IOCTL_KGSL_DRAWCTXT_DESTROY = _IOW ( 0x09 , 0x14 , struct_kgsl_drawctxt_destroy ) # macro (from list)
class struct_kgsl_map_user_mem(Structure):
    pass

struct_kgsl_map_user_mem._pack_ = 1 # source:False
struct_kgsl_map_user_mem._fields_ = [
    ('fd', ctypes.c_int32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('gpuaddr', ctypes.c_uint64),
    ('len', ctypes.c_uint64),
    ('offset', ctypes.c_uint64),
    ('hostptr', ctypes.c_uint64),
    ('memtype', kgsl_user_mem_type),
    ('flags', ctypes.c_uint32),
]

IOCTL_KGSL_MAP_USER_MEM = _IOWR ( 0x09 , 0x15 , struct_kgsl_map_user_mem ) # macro (from list)
class struct_kgsl_cmdstream_readtimestamp_ctxtid(Structure):
    pass

struct_kgsl_cmdstream_readtimestamp_ctxtid._pack_ = 1 # source:False
struct_kgsl_cmdstream_readtimestamp_ctxtid._fields_ = [
    ('context_id', ctypes.c_uint32),
    ('type', ctypes.c_uint32),
    ('timestamp', ctypes.c_uint32),
]

IOCTL_KGSL_CMDSTREAM_READTIMESTAMP_CTXTID = _IOWR ( 0x09 , 0x16 , struct_kgsl_cmdstream_readtimestamp_ctxtid ) # macro (from list)
class struct_kgsl_cmdstream_freememontimestamp_ctxtid(Structure):
    pass

struct_kgsl_cmdstream_freememontimestamp_ctxtid._pack_ = 1 # source:False
struct_kgsl_cmdstream_freememontimestamp_ctxtid._fields_ = [
    ('context_id', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('gpuaddr', ctypes.c_uint64),
    ('type', ctypes.c_uint32),
    ('timestamp', ctypes.c_uint32),
]

IOCTL_KGSL_CMDSTREAM_FREEMEMONTIMESTAMP_CTXTID = _IOW ( 0x09 , 0x17 , struct_kgsl_cmdstream_freememontimestamp_ctxtid ) # macro (from list)
class struct_kgsl_sharedmem_from_pmem(Structure):
    pass

struct_kgsl_sharedmem_from_pmem._pack_ = 1 # source:False
struct_kgsl_sharedmem_from_pmem._fields_ = [
    ('pmem_fd', ctypes.c_int32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('gpuaddr', ctypes.c_uint64),
    ('len', ctypes.c_uint32),
    ('offset', ctypes.c_uint32),
]

IOCTL_KGSL_SHAREDMEM_FROM_PMEM = _IOWR ( 0x09 , 0x20 , struct_kgsl_sharedmem_from_pmem ) # macro (from list)
class struct_kgsl_sharedmem_free(Structure):
    pass

struct_kgsl_sharedmem_free._pack_ = 1 # source:False
struct_kgsl_sharedmem_free._fields_ = [
    ('gpuaddr', ctypes.c_uint64),
]

IOCTL_KGSL_SHAREDMEM_FREE = _IOW ( 0x09 , 0x21 , struct_kgsl_sharedmem_free ) # macro (from list)
IOCTL_KGSL_SHAREDMEM_FLUSH_CACHE = _IOW ( 0x09 , 0x24 , struct_kgsl_sharedmem_free ) # macro (from list)
class struct_kgsl_cff_user_event(Structure):
    pass

struct_kgsl_cff_user_event._pack_ = 1 # source:False
struct_kgsl_cff_user_event._fields_ = [
    ('cff_opcode', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 3),
    ('op1', ctypes.c_uint32),
    ('op2', ctypes.c_uint32),
    ('op3', ctypes.c_uint32),
    ('op4', ctypes.c_uint32),
    ('op5', ctypes.c_uint32),
    ('__pad', ctypes.c_uint32 * 2),
]

IOCTL_KGSL_CFF_USER_EVENT = _IOW ( 0x09 , 0x31 , struct_kgsl_cff_user_event ) # macro (from list)
class struct_kgsl_gmem_desc(Structure):
    pass

struct_kgsl_gmem_desc._pack_ = 1 # source:False
struct_kgsl_gmem_desc._fields_ = [
    ('x', ctypes.c_uint32),
    ('y', ctypes.c_uint32),
    ('width', ctypes.c_uint32),
    ('height', ctypes.c_uint32),
    ('pitch', ctypes.c_uint32),
]

class struct_kgsl_buffer_desc(Structure):
    pass

struct_kgsl_buffer_desc._pack_ = 1 # source:False
struct_kgsl_buffer_desc._fields_ = [
    ('hostptr', ctypes.POINTER(None)),
    ('gpuaddr', ctypes.c_uint64),
    ('size', ctypes.c_int32),
    ('format', ctypes.c_uint32),
    ('pitch', ctypes.c_uint32),
    ('enabled', ctypes.c_uint32),
]

class struct_kgsl_bind_gmem_shadow(Structure):
    pass

struct_kgsl_bind_gmem_shadow._pack_ = 1 # source:False
struct_kgsl_bind_gmem_shadow._fields_ = [
    ('drawctxt_id', ctypes.c_uint32),
    ('gmem_desc', struct_kgsl_gmem_desc),
    ('shadow_x', ctypes.c_uint32),
    ('shadow_y', ctypes.c_uint32),
    ('shadow_buffer', struct_kgsl_buffer_desc),
    ('buffer_id', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

IOCTL_KGSL_DRAWCTXT_BIND_GMEM_SHADOW = _IOW ( 0x09 , 0x22 , struct_kgsl_bind_gmem_shadow ) # macro (from list)
class struct_kgsl_sharedmem_from_vmalloc(Structure):
    pass

struct_kgsl_sharedmem_from_vmalloc._pack_ = 1 # source:False
struct_kgsl_sharedmem_from_vmalloc._fields_ = [
    ('gpuaddr', ctypes.c_uint64),
    ('hostptr', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
]

IOCTL_KGSL_SHAREDMEM_FROM_VMALLOC = _IOWR ( 0x09 , 0x23 , struct_kgsl_sharedmem_from_vmalloc ) # macro (from list)
class struct_kgsl_drawctxt_set_bin_base_offset(Structure):
    pass

struct_kgsl_drawctxt_set_bin_base_offset._pack_ = 1 # source:False
struct_kgsl_drawctxt_set_bin_base_offset._fields_ = [
    ('drawctxt_id', ctypes.c_uint32),
    ('offset', ctypes.c_uint32),
]

IOCTL_KGSL_DRAWCTXT_SET_BIN_BASE_OFFSET = _IOW ( 0x09 , 0x25 , struct_kgsl_drawctxt_set_bin_base_offset ) # macro (from list)

# values for enumeration 'kgsl_cmdwindow_type'
kgsl_cmdwindow_type__enumvalues = {
    0: 'KGSL_CMDWINDOW_MIN',
    0: 'KGSL_CMDWINDOW_2D',
    1: 'KGSL_CMDWINDOW_3D',
    2: 'KGSL_CMDWINDOW_MMU',
    255: 'KGSL_CMDWINDOW_ARBITER',
    255: 'KGSL_CMDWINDOW_MAX',
}
KGSL_CMDWINDOW_MIN = 0
KGSL_CMDWINDOW_2D = 0
KGSL_CMDWINDOW_3D = 1
KGSL_CMDWINDOW_MMU = 2
KGSL_CMDWINDOW_ARBITER = 255
KGSL_CMDWINDOW_MAX = 255
kgsl_cmdwindow_type = ctypes.c_uint32 # enum
class struct_kgsl_cmdwindow_write(Structure):
    pass

struct_kgsl_cmdwindow_write._pack_ = 1 # source:False
struct_kgsl_cmdwindow_write._fields_ = [
    ('target', kgsl_cmdwindow_type),
    ('addr', ctypes.c_uint32),
    ('data', ctypes.c_uint32),
]

IOCTL_KGSL_CMDWINDOW_WRITE = _IOW ( 0x09 , 0x2e , struct_kgsl_cmdwindow_write ) # macro (from list)
class struct_kgsl_gpumem_alloc(Structure):
    pass

struct_kgsl_gpumem_alloc._pack_ = 1 # source:False
struct_kgsl_gpumem_alloc._fields_ = [
    ('gpuaddr', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
    ('flags', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

IOCTL_KGSL_GPUMEM_ALLOC = _IOWR ( 0x09 , 0x2f , struct_kgsl_gpumem_alloc ) # macro (from list)
class struct_kgsl_cff_syncmem(Structure):
    pass

struct_kgsl_cff_syncmem._pack_ = 1 # source:False
struct_kgsl_cff_syncmem._fields_ = [
    ('gpuaddr', ctypes.c_uint64),
    ('len', ctypes.c_uint64),
    ('__pad', ctypes.c_uint32 * 2),
]

IOCTL_KGSL_CFF_SYNCMEM = _IOW ( 0x09 , 0x30 , struct_kgsl_cff_syncmem ) # macro (from list)
class struct_kgsl_timestamp_event(Structure):
    pass

struct_kgsl_timestamp_event._pack_ = 1 # source:False
struct_kgsl_timestamp_event._fields_ = [
    ('type', ctypes.c_int32),
    ('timestamp', ctypes.c_uint32),
    ('context_id', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('priv', ctypes.POINTER(None)),
    ('len', ctypes.c_uint64),
]

IOCTL_KGSL_TIMESTAMP_EVENT_OLD = _IOW ( 0x09 , 0x31 , struct_kgsl_timestamp_event ) # macro (from list)
IOCTL_KGSL_TIMESTAMP_EVENT = _IOWR ( 0x09 , 0x33 , struct_kgsl_timestamp_event ) # macro (from list)
class struct_kgsl_timestamp_event_genlock(Structure):
    pass

struct_kgsl_timestamp_event_genlock._pack_ = 1 # source:False
struct_kgsl_timestamp_event_genlock._fields_ = [
    ('handle', ctypes.c_int32),
]

class struct_kgsl_timestamp_event_fence(Structure):
    pass

struct_kgsl_timestamp_event_fence._pack_ = 1 # source:False
struct_kgsl_timestamp_event_fence._fields_ = [
    ('fence_fd', ctypes.c_int32),
]

class struct_kgsl_gpumem_alloc_id(Structure):
    pass

struct_kgsl_gpumem_alloc_id._pack_ = 1 # source:False
struct_kgsl_gpumem_alloc_id._fields_ = [
    ('id', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('size', ctypes.c_uint64),
    ('mmapsize', ctypes.c_uint64),
    ('gpuaddr', ctypes.c_uint64),
    ('__pad', ctypes.c_uint64 * 2),
]

IOCTL_KGSL_GPUMEM_ALLOC_ID = _IOWR ( 0x09 , 0x34 , struct_kgsl_gpumem_alloc_id ) # macro (from list)
class struct_kgsl_gpumem_free_id(Structure):
    pass

struct_kgsl_gpumem_free_id._pack_ = 1 # source:False
struct_kgsl_gpumem_free_id._fields_ = [
    ('id', ctypes.c_uint32),
    ('__pad', ctypes.c_uint32),
]

IOCTL_KGSL_GPUMEM_FREE_ID = _IOWR ( 0x09 , 0x35 , struct_kgsl_gpumem_free_id ) # macro (from list)
class struct_kgsl_gpumem_get_info(Structure):
    pass

struct_kgsl_gpumem_get_info._pack_ = 1 # source:False
struct_kgsl_gpumem_get_info._fields_ = [
    ('gpuaddr', ctypes.c_uint64),
    ('id', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('size', ctypes.c_uint64),
    ('mmapsize', ctypes.c_uint64),
    ('useraddr', ctypes.c_uint64),
    ('__pad', ctypes.c_uint64 * 4),
]

IOCTL_KGSL_GPUMEM_GET_INFO = _IOWR ( 0x09 , 0x36 , struct_kgsl_gpumem_get_info ) # macro (from list)
class struct_kgsl_gpumem_sync_cache(Structure):
    pass

struct_kgsl_gpumem_sync_cache._pack_ = 1 # source:False
struct_kgsl_gpumem_sync_cache._fields_ = [
    ('gpuaddr', ctypes.c_uint64),
    ('id', ctypes.c_uint32),
    ('op', ctypes.c_uint32),
    ('offset', ctypes.c_uint64),
    ('length', ctypes.c_uint64),
]

IOCTL_KGSL_GPUMEM_SYNC_CACHE = _IOW ( 0x09 , 0x37 , struct_kgsl_gpumem_sync_cache ) # macro (from list)
class struct_kgsl_perfcounter_get(Structure):
    pass

struct_kgsl_perfcounter_get._pack_ = 1 # source:False
struct_kgsl_perfcounter_get._fields_ = [
    ('groupid', ctypes.c_uint32),
    ('countable', ctypes.c_uint32),
    ('offset', ctypes.c_uint32),
    ('offset_hi', ctypes.c_uint32),
    ('__pad', ctypes.c_uint32),
]

IOCTL_KGSL_PERFCOUNTER_GET = _IOWR ( 0x09 , 0x38 , struct_kgsl_perfcounter_get ) # macro (from list)
class struct_kgsl_perfcounter_put(Structure):
    pass

struct_kgsl_perfcounter_put._pack_ = 1 # source:False
struct_kgsl_perfcounter_put._fields_ = [
    ('groupid', ctypes.c_uint32),
    ('countable', ctypes.c_uint32),
    ('__pad', ctypes.c_uint32 * 2),
]

IOCTL_KGSL_PERFCOUNTER_PUT = _IOW ( 0x09 , 0x39 , struct_kgsl_perfcounter_put ) # macro (from list)
class struct_kgsl_perfcounter_query(Structure):
    pass

struct_kgsl_perfcounter_query._pack_ = 1 # source:False
struct_kgsl_perfcounter_query._fields_ = [
    ('groupid', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('countables', ctypes.POINTER(ctypes.c_uint32)),
    ('count', ctypes.c_uint32),
    ('max_counters', ctypes.c_uint32),
    ('__pad', ctypes.c_uint32 * 2),
]

IOCTL_KGSL_PERFCOUNTER_QUERY = _IOWR ( 0x09 , 0x3A , struct_kgsl_perfcounter_query ) # macro (from list)
class struct_kgsl_perfcounter_read_group(Structure):
    pass

struct_kgsl_perfcounter_read_group._pack_ = 1 # source:False
struct_kgsl_perfcounter_read_group._fields_ = [
    ('groupid', ctypes.c_uint32),
    ('countable', ctypes.c_uint32),
    ('value', ctypes.c_uint64),
]

class struct_kgsl_perfcounter_read(Structure):
    pass

struct_kgsl_perfcounter_read._pack_ = 1 # source:False
struct_kgsl_perfcounter_read._fields_ = [
    ('reads', ctypes.POINTER(struct_kgsl_perfcounter_read_group)),
    ('count', ctypes.c_uint32),
    ('__pad', ctypes.c_uint32 * 2),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

IOCTL_KGSL_PERFCOUNTER_READ = _IOWR ( 0x09 , 0x3B , struct_kgsl_perfcounter_read ) # macro (from list)
class struct_kgsl_gpumem_sync_cache_bulk(Structure):
    pass

struct_kgsl_gpumem_sync_cache_bulk._pack_ = 1 # source:False
struct_kgsl_gpumem_sync_cache_bulk._fields_ = [
    ('id_list', ctypes.POINTER(ctypes.c_uint32)),
    ('count', ctypes.c_uint32),
    ('op', ctypes.c_uint32),
    ('__pad', ctypes.c_uint32 * 2),
]

IOCTL_KGSL_GPUMEM_SYNC_CACHE_BULK = _IOWR ( 0x09 , 0x3C , struct_kgsl_gpumem_sync_cache_bulk ) # macro (from list)
class struct_kgsl_cmd_syncpoint_timestamp(Structure):
    pass

struct_kgsl_cmd_syncpoint_timestamp._pack_ = 1 # source:False
struct_kgsl_cmd_syncpoint_timestamp._fields_ = [
    ('context_id', ctypes.c_uint32),
    ('timestamp', ctypes.c_uint32),
]

class struct_kgsl_cmd_syncpoint_fence(Structure):
    pass

struct_kgsl_cmd_syncpoint_fence._pack_ = 1 # source:False
struct_kgsl_cmd_syncpoint_fence._fields_ = [
    ('fd', ctypes.c_int32),
]

class struct_kgsl_cmd_syncpoint(Structure):
    pass

struct_kgsl_cmd_syncpoint._pack_ = 1 # source:False
struct_kgsl_cmd_syncpoint._fields_ = [
    ('type', ctypes.c_int32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('priv', ctypes.POINTER(None)),
    ('size', ctypes.c_uint64),
]

class struct_kgsl_submit_commands(Structure):
    pass

struct_kgsl_submit_commands._pack_ = 1 # source:False
struct_kgsl_submit_commands._fields_ = [
    ('context_id', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('cmdlist', ctypes.POINTER(struct_kgsl_ibdesc)),
    ('numcmds', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('synclist', ctypes.POINTER(struct_kgsl_cmd_syncpoint)),
    ('numsyncs', ctypes.c_uint32),
    ('timestamp', ctypes.c_uint32),
    ('__pad', ctypes.c_uint32 * 4),
]

IOCTL_KGSL_SUBMIT_COMMANDS = _IOWR ( 0x09 , 0x3D , struct_kgsl_submit_commands ) # macro (from list)
class struct_kgsl_device_constraint(Structure):
    pass

struct_kgsl_device_constraint._pack_ = 1 # source:False
struct_kgsl_device_constraint._fields_ = [
    ('type', ctypes.c_uint32),
    ('context_id', ctypes.c_uint32),
    ('data', ctypes.POINTER(None)),
    ('size', ctypes.c_uint64),
]

class struct_kgsl_device_constraint_pwrlevel(Structure):
    pass

struct_kgsl_device_constraint_pwrlevel._pack_ = 1 # source:False
struct_kgsl_device_constraint_pwrlevel._fields_ = [
    ('level', ctypes.c_uint32),
]

class struct_kgsl_syncsource_create(Structure):
    pass

struct_kgsl_syncsource_create._pack_ = 1 # source:False
struct_kgsl_syncsource_create._fields_ = [
    ('id', ctypes.c_uint32),
    ('__pad', ctypes.c_uint32 * 3),
]

IOCTL_KGSL_SYNCSOURCE_CREATE = _IOWR ( 0x09 , 0x40 , struct_kgsl_syncsource_create ) # macro (from list)
class struct_kgsl_syncsource_destroy(Structure):
    pass

struct_kgsl_syncsource_destroy._pack_ = 1 # source:False
struct_kgsl_syncsource_destroy._fields_ = [
    ('id', ctypes.c_uint32),
    ('__pad', ctypes.c_uint32 * 3),
]

IOCTL_KGSL_SYNCSOURCE_DESTROY = _IOWR ( 0x09 , 0x41 , struct_kgsl_syncsource_destroy ) # macro (from list)
class struct_kgsl_syncsource_create_fence(Structure):
    pass

struct_kgsl_syncsource_create_fence._pack_ = 1 # source:False
struct_kgsl_syncsource_create_fence._fields_ = [
    ('id', ctypes.c_uint32),
    ('fence_fd', ctypes.c_int32),
    ('__pad', ctypes.c_uint32 * 4),
]

IOCTL_KGSL_SYNCSOURCE_CREATE_FENCE = _IOWR ( 0x09 , 0x42 , struct_kgsl_syncsource_create_fence ) # macro (from list)
class struct_kgsl_syncsource_signal_fence(Structure):
    pass

struct_kgsl_syncsource_signal_fence._pack_ = 1 # source:False
struct_kgsl_syncsource_signal_fence._fields_ = [
    ('id', ctypes.c_uint32),
    ('fence_fd', ctypes.c_int32),
    ('__pad', ctypes.c_uint32 * 4),
]

IOCTL_KGSL_SYNCSOURCE_SIGNAL_FENCE = _IOWR ( 0x09 , 0x43 , struct_kgsl_syncsource_signal_fence ) # macro (from list)
class struct_kgsl_cff_sync_gpuobj(Structure):
    pass

struct_kgsl_cff_sync_gpuobj._pack_ = 1 # source:False
struct_kgsl_cff_sync_gpuobj._fields_ = [
    ('offset', ctypes.c_uint64),
    ('length', ctypes.c_uint64),
    ('id', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

IOCTL_KGSL_CFF_SYNC_GPUOBJ = _IOW ( 0x09 , 0x44 , struct_kgsl_cff_sync_gpuobj ) # macro (from list)
class struct_kgsl_gpuobj_alloc(Structure):
    pass

struct_kgsl_gpuobj_alloc._pack_ = 1 # source:False
struct_kgsl_gpuobj_alloc._fields_ = [
    ('size', ctypes.c_uint64),
    ('flags', ctypes.c_uint64),
    ('va_len', ctypes.c_uint64),
    ('mmapsize', ctypes.c_uint64),
    ('id', ctypes.c_uint32),
    ('metadata_len', ctypes.c_uint32),
    ('metadata', ctypes.c_uint64),
]

IOCTL_KGSL_GPUOBJ_ALLOC = _IOWR ( 0x09 , 0x45 , struct_kgsl_gpuobj_alloc ) # macro (from list)
class struct_kgsl_gpuobj_free(Structure):
    pass

struct_kgsl_gpuobj_free._pack_ = 1 # source:False
struct_kgsl_gpuobj_free._fields_ = [
    ('flags', ctypes.c_uint64),
    ('priv', ctypes.c_uint64),
    ('id', ctypes.c_uint32),
    ('type', ctypes.c_uint32),
    ('len', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

IOCTL_KGSL_GPUOBJ_FREE = _IOW ( 0x09 , 0x46 , struct_kgsl_gpuobj_free ) # macro (from list)
class struct_kgsl_gpu_event_timestamp(Structure):
    pass

struct_kgsl_gpu_event_timestamp._pack_ = 1 # source:False
struct_kgsl_gpu_event_timestamp._fields_ = [
    ('context_id', ctypes.c_uint32),
    ('timestamp', ctypes.c_uint32),
]

class struct_kgsl_gpu_event_fence(Structure):
    pass

struct_kgsl_gpu_event_fence._pack_ = 1 # source:False
struct_kgsl_gpu_event_fence._fields_ = [
    ('fd', ctypes.c_int32),
]

class struct_kgsl_gpuobj_info(Structure):
    pass

struct_kgsl_gpuobj_info._pack_ = 1 # source:False
struct_kgsl_gpuobj_info._fields_ = [
    ('gpuaddr', ctypes.c_uint64),
    ('flags', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
    ('va_len', ctypes.c_uint64),
    ('va_addr', ctypes.c_uint64),
    ('id', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

IOCTL_KGSL_GPUOBJ_INFO = _IOWR ( 0x09 , 0x47 , struct_kgsl_gpuobj_info ) # macro (from list)
class struct_kgsl_gpuobj_import(Structure):
    pass

struct_kgsl_gpuobj_import._pack_ = 1 # source:False
struct_kgsl_gpuobj_import._fields_ = [
    ('priv', ctypes.c_uint64),
    ('priv_len', ctypes.c_uint64),
    ('flags', ctypes.c_uint64),
    ('type', ctypes.c_uint32),
    ('id', ctypes.c_uint32),
]

IOCTL_KGSL_GPUOBJ_IMPORT = _IOWR ( 0x09 , 0x48 , struct_kgsl_gpuobj_import ) # macro (from list)
class struct_kgsl_gpuobj_import_dma_buf(Structure):
    pass

struct_kgsl_gpuobj_import_dma_buf._pack_ = 1 # source:False
struct_kgsl_gpuobj_import_dma_buf._fields_ = [
    ('fd', ctypes.c_int32),
]

class struct_kgsl_gpuobj_import_useraddr(Structure):
    pass

struct_kgsl_gpuobj_import_useraddr._pack_ = 1 # source:False
struct_kgsl_gpuobj_import_useraddr._fields_ = [
    ('virtaddr', ctypes.c_uint64),
]

class struct_kgsl_gpuobj_sync_obj(Structure):
    pass

struct_kgsl_gpuobj_sync_obj._pack_ = 1 # source:False
struct_kgsl_gpuobj_sync_obj._fields_ = [
    ('offset', ctypes.c_uint64),
    ('length', ctypes.c_uint64),
    ('id', ctypes.c_uint32),
    ('op', ctypes.c_uint32),
]

class struct_kgsl_gpuobj_sync(Structure):
    pass

struct_kgsl_gpuobj_sync._pack_ = 1 # source:False
struct_kgsl_gpuobj_sync._fields_ = [
    ('objs', ctypes.c_uint64),
    ('obj_len', ctypes.c_uint32),
    ('count', ctypes.c_uint32),
]

IOCTL_KGSL_GPUOBJ_SYNC = _IOW ( 0x09 , 0x49 , struct_kgsl_gpuobj_sync ) # macro (from list)
class struct_kgsl_command_object(Structure):
    pass

struct_kgsl_command_object._pack_ = 1 # source:False
struct_kgsl_command_object._fields_ = [
    ('offset', ctypes.c_uint64),
    ('gpuaddr', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
    ('flags', ctypes.c_uint32),
    ('id', ctypes.c_uint32),
]

class struct_kgsl_command_syncpoint(Structure):
    pass

struct_kgsl_command_syncpoint._pack_ = 1 # source:False
struct_kgsl_command_syncpoint._fields_ = [
    ('priv', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
    ('type', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

class struct_kgsl_gpu_command(Structure):
    pass

struct_kgsl_gpu_command._pack_ = 1 # source:False
struct_kgsl_gpu_command._fields_ = [
    ('flags', ctypes.c_uint64),
    ('cmdlist', ctypes.c_uint64),
    ('cmdsize', ctypes.c_uint32),
    ('numcmds', ctypes.c_uint32),
    ('objlist', ctypes.c_uint64),
    ('objsize', ctypes.c_uint32),
    ('numobjs', ctypes.c_uint32),
    ('synclist', ctypes.c_uint64),
    ('syncsize', ctypes.c_uint32),
    ('numsyncs', ctypes.c_uint32),
    ('context_id', ctypes.c_uint32),
    ('timestamp', ctypes.c_uint32),
]

IOCTL_KGSL_GPU_COMMAND = _IOWR ( 0x09 , 0x4A , struct_kgsl_gpu_command ) # macro (from list)
class struct_kgsl_preemption_counters_query(Structure):
    pass

struct_kgsl_preemption_counters_query._pack_ = 1 # source:False
struct_kgsl_preemption_counters_query._fields_ = [
    ('counters', ctypes.c_uint64),
    ('size_user', ctypes.c_uint32),
    ('size_priority_level', ctypes.c_uint32),
    ('max_priority_level', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

IOCTL_KGSL_PREEMPTIONCOUNTER_QUERY = _IOWR ( 0x09 , 0x4B , struct_kgsl_preemption_counters_query ) # macro (from list)
class struct_kgsl_gpuobj_set_info(Structure):
    pass

struct_kgsl_gpuobj_set_info._pack_ = 1 # source:False
struct_kgsl_gpuobj_set_info._fields_ = [
    ('flags', ctypes.c_uint64),
    ('metadata', ctypes.c_uint64),
    ('id', ctypes.c_uint32),
    ('metadata_len', ctypes.c_uint32),
    ('type', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

IOCTL_KGSL_GPUOBJ_SET_INFO = _IOW ( 0x09 , 0x4C , struct_kgsl_gpuobj_set_info ) # macro (from list)
__all__ = \
    ['KGSL_CACHEMODE_MASK', 'KGSL_CACHEMODE_SHIFT',
    'KGSL_CACHEMODE_UNCACHED', 'KGSL_CACHEMODE_WRITEBACK',
    'KGSL_CACHEMODE_WRITECOMBINE', 'KGSL_CACHEMODE_WRITETHROUGH',
    'KGSL_CMDBATCH_CTX_SWITCH', 'KGSL_CMDBATCH_END_OF_FRAME',
    'KGSL_CMDBATCH_MARKER', 'KGSL_CMDBATCH_MEMLIST',
    'KGSL_CMDBATCH_PROFILING', 'KGSL_CMDBATCH_PROFILING_KTIME',
    'KGSL_CMDBATCH_PWR_CONSTRAINT', 'KGSL_CMDBATCH_SUBMIT_IB_LIST',
    'KGSL_CMDBATCH_SYNC', 'KGSL_CMDLIST_CTXTSWITCH_PREAMBLE',
    'KGSL_CMDLIST_IB', 'KGSL_CMDLIST_IB_PREAMBLE',
    'KGSL_CMDWINDOW_2D', 'KGSL_CMDWINDOW_3D',
    'KGSL_CMDWINDOW_ARBITER', 'KGSL_CMDWINDOW_MAX',
    'KGSL_CMDWINDOW_MIN', 'KGSL_CMDWINDOW_MMU',
    'KGSL_CMD_SYNCPOINT_TYPE_FENCE',
    'KGSL_CMD_SYNCPOINT_TYPE_TIMESTAMP', 'KGSL_CONSTRAINT_NONE',
    'KGSL_CONSTRAINT_PWRLEVEL', 'KGSL_CONSTRAINT_PWR_MAX',
    'KGSL_CONSTRAINT_PWR_MIN', 'KGSL_CONTEXT_CTX_SWITCH',
    'KGSL_CONTEXT_END_OF_FRAME', 'KGSL_CONTEXT_IFH_NOP',
    'KGSL_CONTEXT_INVALID', 'KGSL_CONTEXT_NO_FAULT_TOLERANCE',
    'KGSL_CONTEXT_NO_GMEM_ALLOC', 'KGSL_CONTEXT_PER_CONTEXT_TS',
    'KGSL_CONTEXT_PREAMBLE', 'KGSL_CONTEXT_PREEMPT_STYLE_DEFAULT',
    'KGSL_CONTEXT_PREEMPT_STYLE_FINEGRAIN',
    'KGSL_CONTEXT_PREEMPT_STYLE_MASK',
    'KGSL_CONTEXT_PREEMPT_STYLE_RINGBUFFER',
    'KGSL_CONTEXT_PREEMPT_STYLE_SHIFT', 'KGSL_CONTEXT_PRIORITY_MASK',
    'KGSL_CONTEXT_PRIORITY_SHIFT', 'KGSL_CONTEXT_PRIORITY_UNDEF',
    'KGSL_CONTEXT_PWR_CONSTRAINT', 'KGSL_CONTEXT_SAVE_GMEM',
    'KGSL_CONTEXT_SECURE', 'KGSL_CONTEXT_SUBMIT_IB_LIST',
    'KGSL_CONTEXT_SYNC', 'KGSL_CONTEXT_TRASH_STATE',
    'KGSL_CONTEXT_TYPE_ANY', 'KGSL_CONTEXT_TYPE_C2D',
    'KGSL_CONTEXT_TYPE_CL', 'KGSL_CONTEXT_TYPE_GL',
    'KGSL_CONTEXT_TYPE_MASK', 'KGSL_CONTEXT_TYPE_RS',
    'KGSL_CONTEXT_TYPE_SHIFT', 'KGSL_CONTEXT_TYPE_UNKNOWN',
    'KGSL_CONTEXT_USER_GENERATED_TS',
    'KGSL_CTX_STAT_GUILTY_CONTEXT_RESET_EXT',
    'KGSL_CTX_STAT_INNOCENT_CONTEXT_RESET_EXT',
    'KGSL_CTX_STAT_NO_ERROR',
    'KGSL_CTX_STAT_UNKNOWN_CONTEXT_RESET_EXT', 'KGSL_DEVICE_3D0',
    'KGSL_DEVICE_MAX', 'KGSL_FLAGS_ACTIVE', 'KGSL_FLAGS_INITIALIZED',
    'KGSL_FLAGS_INITIALIZED0', 'KGSL_FLAGS_NORMALMODE',
    'KGSL_FLAGS_PER_CONTEXT_TIMESTAMPS', 'KGSL_FLAGS_RESERVED0',
    'KGSL_FLAGS_RESERVED1', 'KGSL_FLAGS_RESERVED2',
    'KGSL_FLAGS_SAFEMODE', 'KGSL_FLAGS_SOFT_RESET',
    'KGSL_FLAGS_STARTED', 'KGSL_GPUMEM_CACHE_CLEAN',
    'KGSL_GPUMEM_CACHE_FLUSH', 'KGSL_GPUMEM_CACHE_FROM_GPU',
    'KGSL_GPUMEM_CACHE_INV', 'KGSL_GPUMEM_CACHE_RANGE',
    'KGSL_GPUMEM_CACHE_TO_GPU', 'KGSL_GPUOBJ_ALLOC_METADATA_MAX',
    'KGSL_GPUOBJ_FREE_ON_EVENT', 'KGSL_GPUOBJ_SET_INFO_METADATA',
    'KGSL_GPUOBJ_SET_INFO_TYPE', 'KGSL_GPU_EVENT_FENCE',
    'KGSL_GPU_EVENT_TIMESTAMP', 'KGSL_IBDESC_MEMLIST',
    'KGSL_IBDESC_PROFILING_BUFFER', 'KGSL_IOC_TYPE',
    'KGSL_MEMALIGN_MASK', 'KGSL_MEMALIGN_SHIFT',
    'KGSL_MEMFLAGS_FORCE_32BIT', 'KGSL_MEMFLAGS_GPUREADONLY',
    'KGSL_MEMFLAGS_GPUWRITEONLY', 'KGSL_MEMFLAGS_NOT_USERMEM',
    'KGSL_MEMFLAGS_SECURE', 'KGSL_MEMFLAGS_USERMEM_MASK',
    'KGSL_MEMFLAGS_USERMEM_SHIFT', 'KGSL_MEMFLAGS_USE_CPU_MAP',
    'KGSL_MEMTYPE_2D', 'KGSL_MEMTYPE_ARRAYBUFFER', 'KGSL_MEMTYPE_CL',
    'KGSL_MEMTYPE_CL_BUFFER_MAP', 'KGSL_MEMTYPE_CL_BUFFER_NOMAP',
    'KGSL_MEMTYPE_CL_IMAGE_MAP', 'KGSL_MEMTYPE_CL_IMAGE_NOMAP',
    'KGSL_MEMTYPE_CL_KERNEL_STACK', 'KGSL_MEMTYPE_COMMAND',
    'KGSL_MEMTYPE_EGL_IMAGE', 'KGSL_MEMTYPE_EGL_SHADOW',
    'KGSL_MEMTYPE_EGL_SURFACE', 'KGSL_MEMTYPE_ELEMENTARRAYBUFFER',
    'KGSL_MEMTYPE_FRAMEBUFFER', 'KGSL_MEMTYPE_GL',
    'KGSL_MEMTYPE_KERNEL', 'KGSL_MEMTYPE_MASK',
    'KGSL_MEMTYPE_MULTISAMPLE', 'KGSL_MEMTYPE_OBJECTANY',
    'KGSL_MEMTYPE_RENDERBUFFER', 'KGSL_MEMTYPE_SHIFT',
    'KGSL_MEMTYPE_SURFACE', 'KGSL_MEMTYPE_TEXTURE',
    'KGSL_MEMTYPE_VERTEXARRAYBUFFER', 'KGSL_OBJLIST_MEMOBJ',
    'KGSL_OBJLIST_PROFILE', 'KGSL_PERFCOUNTER_BROKEN',
    'KGSL_PERFCOUNTER_GROUP_ALWAYSON',
    'KGSL_PERFCOUNTER_GROUP_ALWAYSON_PWR',
    'KGSL_PERFCOUNTER_GROUP_CCU', 'KGSL_PERFCOUNTER_GROUP_CCU_PWR',
    'KGSL_PERFCOUNTER_GROUP_CMP', 'KGSL_PERFCOUNTER_GROUP_CP',
    'KGSL_PERFCOUNTER_GROUP_CP_PWR',
    'KGSL_PERFCOUNTER_GROUP_GPMU_PWR', 'KGSL_PERFCOUNTER_GROUP_HLSQ',
    'KGSL_PERFCOUNTER_GROUP_L2', 'KGSL_PERFCOUNTER_GROUP_LRZ',
    'KGSL_PERFCOUNTER_GROUP_MAX', 'KGSL_PERFCOUNTER_GROUP_MH',
    'KGSL_PERFCOUNTER_GROUP_PA_SU', 'KGSL_PERFCOUNTER_GROUP_PC',
    'KGSL_PERFCOUNTER_GROUP_PWR', 'KGSL_PERFCOUNTER_GROUP_RAS',
    'KGSL_PERFCOUNTER_GROUP_RB', 'KGSL_PERFCOUNTER_GROUP_RBBM',
    'KGSL_PERFCOUNTER_GROUP_RB_PWR', 'KGSL_PERFCOUNTER_GROUP_SP',
    'KGSL_PERFCOUNTER_GROUP_SP_PWR', 'KGSL_PERFCOUNTER_GROUP_SQ',
    'KGSL_PERFCOUNTER_GROUP_SX', 'KGSL_PERFCOUNTER_GROUP_TCF',
    'KGSL_PERFCOUNTER_GROUP_TCM', 'KGSL_PERFCOUNTER_GROUP_TCR',
    'KGSL_PERFCOUNTER_GROUP_TP', 'KGSL_PERFCOUNTER_GROUP_TP_PWR',
    'KGSL_PERFCOUNTER_GROUP_TSE', 'KGSL_PERFCOUNTER_GROUP_UCHE',
    'KGSL_PERFCOUNTER_GROUP_UCHE_PWR', 'KGSL_PERFCOUNTER_GROUP_VBIF',
    'KGSL_PERFCOUNTER_GROUP_VBIF_PWR', 'KGSL_PERFCOUNTER_GROUP_VFD',
    'KGSL_PERFCOUNTER_GROUP_VPC', 'KGSL_PERFCOUNTER_GROUP_VSC',
    'KGSL_PERFCOUNTER_NOT_USED', 'KGSL_PROP_DEVICE_BITNESS',
    'KGSL_PROP_DEVICE_INFO', 'KGSL_PROP_DEVICE_POWER',
    'KGSL_PROP_DEVICE_SHADOW', 'KGSL_PROP_GPMU_VERSION',
    'KGSL_PROP_GPU_RESET_STAT', 'KGSL_PROP_INTERRUPT_WAITS',
    'KGSL_PROP_MMU_ENABLE', 'KGSL_PROP_PWRCTRL',
    'KGSL_PROP_PWR_CONSTRAINT', 'KGSL_PROP_SHMEM',
    'KGSL_PROP_SHMEM_APERTURES', 'KGSL_PROP_SP_GENERIC_MEM',
    'KGSL_PROP_UCHE_GMEM_VADDR', 'KGSL_PROP_UCODE_VERSION',
    'KGSL_PROP_VERSION', 'KGSL_SYNCOBJ_SERVER_TIMEOUT',
    'KGSL_TIMESTAMP_CONSUMED', 'KGSL_TIMESTAMP_EVENT_FENCE',
    'KGSL_TIMESTAMP_EVENT_GENLOCK', 'KGSL_TIMESTAMP_QUEUED',
    'KGSL_TIMESTAMP_RETIRED', 'KGSL_USER_MEM_TYPE_ADDR',
    'KGSL_USER_MEM_TYPE_ASHMEM', 'KGSL_USER_MEM_TYPE_DMABUF',
    'KGSL_USER_MEM_TYPE_ION', 'KGSL_USER_MEM_TYPE_MAX',
    'KGSL_USER_MEM_TYPE_PMEM', 'KGSL_VERSION_MAJOR',
    'KGSL_VERSION_MINOR', '_IO', '_IOR', '_IOW', '_IOWR',
    '_UAPI_MSM_KGSL_H', 'kgsl_cmdwindow_type', 'kgsl_ctx_reset_stat',
    'kgsl_deviceid', 'kgsl_timestamp_type', 'kgsl_user_mem_type',
    'size_t', 'struct_kgsl_bind_gmem_shadow',
    'struct_kgsl_buffer_desc', 'struct_kgsl_cff_sync_gpuobj',
    'struct_kgsl_cff_syncmem', 'struct_kgsl_cff_user_event',
    'struct_kgsl_cmd_syncpoint', 'struct_kgsl_cmd_syncpoint_fence',
    'struct_kgsl_cmd_syncpoint_timestamp',
    'struct_kgsl_cmdbatch_profiling_buffer',
    'struct_kgsl_cmdstream_freememontimestamp',
    'struct_kgsl_cmdstream_freememontimestamp_ctxtid',
    'struct_kgsl_cmdstream_readtimestamp',
    'struct_kgsl_cmdstream_readtimestamp_ctxtid',
    'struct_kgsl_cmdwindow_write', 'struct_kgsl_command_object',
    'struct_kgsl_command_syncpoint', 'struct_kgsl_device_constraint',
    'struct_kgsl_device_constraint_pwrlevel',
    'struct_kgsl_device_getproperty',
    'struct_kgsl_device_waittimestamp',
    'struct_kgsl_device_waittimestamp_ctxtid', 'struct_kgsl_devinfo',
    'struct_kgsl_devmemstore', 'struct_kgsl_drawctxt_create',
    'struct_kgsl_drawctxt_destroy',
    'struct_kgsl_drawctxt_set_bin_base_offset',
    'struct_kgsl_gmem_desc', 'struct_kgsl_gpmu_version',
    'struct_kgsl_gpu_command', 'struct_kgsl_gpu_event_fence',
    'struct_kgsl_gpu_event_timestamp', 'struct_kgsl_gpumem_alloc',
    'struct_kgsl_gpumem_alloc_id', 'struct_kgsl_gpumem_free_id',
    'struct_kgsl_gpumem_get_info', 'struct_kgsl_gpumem_sync_cache',
    'struct_kgsl_gpumem_sync_cache_bulk', 'struct_kgsl_gpuobj_alloc',
    'struct_kgsl_gpuobj_free', 'struct_kgsl_gpuobj_import',
    'struct_kgsl_gpuobj_import_dma_buf',
    'struct_kgsl_gpuobj_import_useraddr', 'struct_kgsl_gpuobj_info',
    'struct_kgsl_gpuobj_set_info', 'struct_kgsl_gpuobj_sync',
    'struct_kgsl_gpuobj_sync_obj', 'struct_kgsl_ibdesc',
    'struct_kgsl_map_user_mem', 'struct_kgsl_perfcounter_get',
    'struct_kgsl_perfcounter_put', 'struct_kgsl_perfcounter_query',
    'struct_kgsl_perfcounter_read',
    'struct_kgsl_perfcounter_read_group',
    'struct_kgsl_preemption_counters_query',
    'struct_kgsl_ringbuffer_issueibcmds', 'struct_kgsl_shadowprop',
    'struct_kgsl_sharedmem_free', 'struct_kgsl_sharedmem_from_pmem',
    'struct_kgsl_sharedmem_from_vmalloc',
    'struct_kgsl_sp_generic_mem', 'struct_kgsl_submit_commands',
    'struct_kgsl_syncsource_create',
    'struct_kgsl_syncsource_create_fence',
    'struct_kgsl_syncsource_destroy',
    'struct_kgsl_syncsource_signal_fence',
    'struct_kgsl_timestamp_event',
    'struct_kgsl_timestamp_event_fence',
    'struct_kgsl_timestamp_event_genlock',
    'struct_kgsl_ucode_version', 'struct_kgsl_version', 'uint64_t']
def KGSL_CONTEXT_PRIORITY(val): return (val << KGSL_CONTEXT_PRIORITY_SHIFT) & KGSL_CONTEXT_PRIORITY_MASK
def KGSL_CONTEXT_PREEMPT_STYLE(val): return (val << KGSL_CONTEXT_PREEMPT_STYLE_SHIFT) & KGSL_CONTEXT_PREEMPT_STYLE_MASK
def KGSL_CONTEXT_TYPE(val): return (val << KGSL_CONTEXT_TYPE_SHIFT) & KGSL_CONTEXT_TYPE_MASK
def KGSL_CACHEMODE(val): return (val << KGSL_CACHEMODE_SHIFT) & KGSL_CACHEMODE_MASK
def KGSL_MEMTYPE(val): return (val << KGSL_MEMTYPE_SHIFT) & KGSL_MEMTYPE_MASK
def KGSL_MEMALIGN(val): return (val << KGSL_MEMALIGN_SHIFT) & KGSL_MEMALIGN_MASK
def KGSL_MEMFLAGS_USERMEM(val): return (val << KGSL_MEMFLAGS_USERMEM_SHIFT) & KGSL_MEMFLAGS_USERMEM_MASK
