# mypy: ignore-errors
# -*- coding: utf-8 -*-
#
# TARGET arch is: []
# WORD_SIZE is: 8
# POINTER_SIZE is: 8
# LONGDOUBLE_SIZE is: 16
#
import ctypes


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




import fcntl, functools

def _do_ioctl(__idir, __base, __nr, __user_struct, __fd, *args, **kwargs):
  ret = fcntl.ioctl(__fd, (__idir<<30) | (ctypes.sizeof(made := __user_struct(*args, **kwargs))<<16) | (__base<<8) | __nr, made)
  if ret != 0: raise OSError(f"ioctl returned {ret}")
  return made

def _IO(base, nr): return functools.partial(_do_ioctl, 0, ord(base) if isinstance(base, str) else base, nr, None)
def _IOW(base, nr, type): return functools.partial(_do_ioctl, 1, ord(base) if isinstance(base, str) else base, nr, type)
def _IOR(base, nr, type): return functools.partial(_do_ioctl, 2, ord(base) if isinstance(base, str) else base, nr, type)
def _IOWR(base, nr, type): return functools.partial(_do_ioctl, 3, ord(base) if isinstance(base, str) else base, nr, type)

c_int128 = ctypes.c_ubyte*16
c_uint128 = c_int128
void = None
if ctypes.sizeof(ctypes.c_longdouble) == 16:
    c_long_double_t = ctypes.c_longdouble
else:
    c_long_double_t = ctypes.c_ubyte*16

class FunctionFactoryStub:
    def __getattr__(self, _):
      return ctypes.CFUNCTYPE(lambda y:y)

# libraries['FIXME_STUB'] explanation
# As you did not list (-l libraryname.so) a library that exports this function
# This is a non-working stub instead.
# You can either re-run clan2py with -l /path/to/library.so
# Or manually fix this by comment the ctypes.CDLL loading
_libraries = {}
_libraries['FIXME_STUB'] = FunctionFactoryStub() #  ctypes.CDLL('FIXME_STUB')
def string_cast(char_pointer, encoding='utf-8', errors='strict'):
    value = ctypes.cast(char_pointer, ctypes.c_char_p).value
    if value is not None and encoding is not None:
        value = value.decode(encoding, errors=errors)
    return value


def char_pointer_cast(string, encoding='utf-8'):
    if encoding is not None:
        try:
            string = string.encode(encoding)
        except AttributeError:
            # In Python3, bytes has no encode attribute
            pass
    string = ctypes.c_char_p(string)
    return ctypes.cast(string, ctypes.POINTER(ctypes.c_char))





_UAPI_LINUX_ION_H = True # macro
# ION_NUM_HEAP_IDS = (ctypes.sizeof*8) # macro
ION_FLAG_CACHED = 1 # macro
ION_FLAG_CACHED_NEEDS_SYNC = 2 # macro
ION_IOC_MAGIC = 'I' # macro
ion_user_handle_t = ctypes.c_int32

# values for enumeration 'ion_heap_type'
ion_heap_type__enumvalues = {
    0: 'ION_HEAP_TYPE_SYSTEM',
    1: 'ION_HEAP_TYPE_SYSTEM_CONTIG',
    2: 'ION_HEAP_TYPE_CARVEOUT',
    3: 'ION_HEAP_TYPE_CHUNK',
    4: 'ION_HEAP_TYPE_DMA',
    5: 'ION_HEAP_TYPE_CUSTOM',
    16: 'ION_NUM_HEAPS',
}
ION_HEAP_TYPE_SYSTEM = 0
ION_HEAP_TYPE_SYSTEM_CONTIG = 1
ION_HEAP_TYPE_CARVEOUT = 2
ION_HEAP_TYPE_CHUNK = 3
ION_HEAP_TYPE_DMA = 4
ION_HEAP_TYPE_CUSTOM = 5
ION_NUM_HEAPS = 16
ion_heap_type = ctypes.c_uint32 # enum
ION_HEAP_SYSTEM_MASK = ((1<<ION_HEAP_TYPE_SYSTEM)) # macro
ION_HEAP_SYSTEM_CONTIG_MASK = ((1<<ION_HEAP_TYPE_SYSTEM_CONTIG)) # macro
ION_HEAP_CARVEOUT_MASK = ((1<<ION_HEAP_TYPE_CARVEOUT)) # macro
ION_HEAP_TYPE_DMA_MASK = ((1<<ION_HEAP_TYPE_DMA)) # macro
class struct_ion_allocation_data(Structure):
    pass

struct_ion_allocation_data._pack_ = 1 # source:False
struct_ion_allocation_data._fields_ = [
    ('len', ctypes.c_uint64),
    ('align', ctypes.c_uint64),
    ('heap_id_mask', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('handle', ctypes.c_int32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

ION_IOC_ALLOC = _IOWR ( 'I' , 0 , struct_ion_allocation_data ) # macro (from list)
class struct_ion_fd_data(Structure):
    pass

struct_ion_fd_data._pack_ = 1 # source:False
struct_ion_fd_data._fields_ = [
    ('handle', ctypes.c_int32),
    ('fd', ctypes.c_int32),
]

ION_IOC_MAP = _IOWR ( 'I' , 2 , struct_ion_fd_data ) # macro (from list)
ION_IOC_SHARE = _IOWR ( 'I' , 4 , struct_ion_fd_data ) # macro (from list)
ION_IOC_IMPORT = _IOWR ( 'I' , 5 , struct_ion_fd_data ) # macro (from list)
ION_IOC_SYNC = _IOWR ( 'I' , 7 , struct_ion_fd_data ) # macro (from list)
class struct_ion_handle_data(Structure):
    pass

struct_ion_handle_data._pack_ = 1 # source:False
struct_ion_handle_data._fields_ = [
    ('handle', ctypes.c_int32),
]

ION_IOC_FREE = _IOWR ( 'I' , 1 , struct_ion_handle_data ) # macro (from list)
class struct_ion_custom_data(Structure):
    pass

struct_ion_custom_data._pack_ = 1 # source:False
struct_ion_custom_data._fields_ = [
    ('cmd', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('arg', ctypes.c_uint64),
]

ION_IOC_CUSTOM = _IOWR ( 'I' , 6 , struct_ion_custom_data ) # macro (from list)
_UAPI_MSM_ION_H = True # macro
ION_HEAP_TYPE_IOMMU = ION_HEAP_TYPE_SYSTEM # macro
ION_FLAG_CP_TOUCH = (1<<17) # macro
ION_FLAG_CP_BITSTREAM = (1<<18) # macro
ION_FLAG_CP_PIXEL = (1<<19) # macro
ION_FLAG_CP_NON_PIXEL = (1<<20) # macro
ION_FLAG_CP_CAMERA = (1<<21) # macro
ION_FLAG_CP_HLOS = (1<<22) # macro
ION_FLAG_CP_HLOS_FREE = (1<<23) # macro
ION_FLAG_CP_SEC_DISPLAY = (1<<25) # macro
ION_FLAG_CP_APP = (1<<26) # macro
ION_FLAG_ALLOW_NON_CONTIG = (1<<24) # macro
ION_FLAG_FORCE_CONTIGUOUS = (1<<30) # macro
ION_FLAG_POOL_FORCE_ALLOC = (1<<16) # macro
ION_FLAG_POOL_PREFETCH = (1<<27) # macro
ION_FORCE_CONTIGUOUS = (1<<30) # macro
def ION_HEAP(bit):  # macro
   return (1<<(bit))
ION_ADSP_HEAP_NAME = "adsp" # macro
ION_SYSTEM_HEAP_NAME = "system" # macro
ION_VMALLOC_HEAP_NAME = "system" # macro
ION_KMALLOC_HEAP_NAME = "kmalloc" # macro
ION_AUDIO_HEAP_NAME = "audio" # macro
ION_SF_HEAP_NAME = "sf" # macro
ION_MM_HEAP_NAME = "mm" # macro
ION_CAMERA_HEAP_NAME = "camera_preview" # macro
ION_IOMMU_HEAP_NAME = "iommu" # macro
ION_MFC_HEAP_NAME = "mfc" # macro
ION_WB_HEAP_NAME = "wb" # macro
ION_MM_FIRMWARE_HEAP_NAME = "mm_fw" # macro
ION_PIL1_HEAP_NAME = "pil_1" # macro
ION_PIL2_HEAP_NAME = "pil_2" # macro
ION_QSECOM_HEAP_NAME = "qsecom" # macro
ION_SECURE_HEAP_NAME = "secure_heap" # macro
ION_SECURE_DISPLAY_HEAP_NAME = "secure_display" # macro
def ION_SET_CACHED(__cache):  # macro
   return (__cache|1)
def ION_SET_UNCACHED(__cache):  # macro
   return (__cache&~1)
def ION_IS_CACHED(__flags):  # macro
   return ((__flags)&1)
ION_IOC_MSM_MAGIC = 'M' # macro

# values for enumeration 'msm_ion_heap_types'
msm_ion_heap_types__enumvalues = {
    6: 'ION_HEAP_TYPE_MSM_START',
    6: 'ION_HEAP_TYPE_SECURE_DMA',
    7: 'ION_HEAP_TYPE_SYSTEM_SECURE',
    8: 'ION_HEAP_TYPE_HYP_CMA',
}
ION_HEAP_TYPE_MSM_START = 6
ION_HEAP_TYPE_SECURE_DMA = 6
ION_HEAP_TYPE_SYSTEM_SECURE = 7
ION_HEAP_TYPE_HYP_CMA = 8
msm_ion_heap_types = ctypes.c_uint32 # enum

# values for enumeration 'ion_heap_ids'
ion_heap_ids__enumvalues = {
    -1: 'INVALID_HEAP_ID',
    8: 'ION_CP_MM_HEAP_ID',
    9: 'ION_SECURE_HEAP_ID',
    10: 'ION_SECURE_DISPLAY_HEAP_ID',
    12: 'ION_CP_MFC_HEAP_ID',
    16: 'ION_CP_WB_HEAP_ID',
    20: 'ION_CAMERA_HEAP_ID',
    21: 'ION_SYSTEM_CONTIG_HEAP_ID',
    22: 'ION_ADSP_HEAP_ID',
    23: 'ION_PIL1_HEAP_ID',
    24: 'ION_SF_HEAP_ID',
    25: 'ION_SYSTEM_HEAP_ID',
    26: 'ION_PIL2_HEAP_ID',
    27: 'ION_QSECOM_HEAP_ID',
    28: 'ION_AUDIO_HEAP_ID',
    29: 'ION_MM_FIRMWARE_HEAP_ID',
    31: 'ION_HEAP_ID_RESERVED',
}
INVALID_HEAP_ID = -1
ION_CP_MM_HEAP_ID = 8
ION_SECURE_HEAP_ID = 9
ION_SECURE_DISPLAY_HEAP_ID = 10
ION_CP_MFC_HEAP_ID = 12
ION_CP_WB_HEAP_ID = 16
ION_CAMERA_HEAP_ID = 20
ION_SYSTEM_CONTIG_HEAP_ID = 21
ION_ADSP_HEAP_ID = 22
ION_PIL1_HEAP_ID = 23
ION_SF_HEAP_ID = 24
ION_SYSTEM_HEAP_ID = 25
ION_PIL2_HEAP_ID = 26
ION_QSECOM_HEAP_ID = 27
ION_AUDIO_HEAP_ID = 28
ION_MM_FIRMWARE_HEAP_ID = 29
ION_HEAP_ID_RESERVED = 31
ion_heap_ids = ctypes.c_int32 # enum
ION_IOMMU_HEAP_ID = ION_SYSTEM_HEAP_ID # macro
ION_FLAG_SECURE = (1<<ION_HEAP_ID_RESERVED) # macro
ION_SECURE = (1<<ION_HEAP_ID_RESERVED) # macro

# values for enumeration 'ion_fixed_position'
ion_fixed_position__enumvalues = {
    0: 'NOT_FIXED',
    1: 'FIXED_LOW',
    2: 'FIXED_MIDDLE',
    3: 'FIXED_HIGH',
}
NOT_FIXED = 0
FIXED_LOW = 1
FIXED_MIDDLE = 2
FIXED_HIGH = 3
ion_fixed_position = ctypes.c_uint32 # enum

# values for enumeration 'cp_mem_usage'
cp_mem_usage__enumvalues = {
    1: 'VIDEO_BITSTREAM',
    2: 'VIDEO_PIXEL',
    3: 'VIDEO_NONPIXEL',
    4: 'DISPLAY_SECURE_CP_USAGE',
    5: 'CAMERA_SECURE_CP_USAGE',
    6: 'MAX_USAGE',
    2147483647: 'UNKNOWN',
}
VIDEO_BITSTREAM = 1
VIDEO_PIXEL = 2
VIDEO_NONPIXEL = 3
DISPLAY_SECURE_CP_USAGE = 4
CAMERA_SECURE_CP_USAGE = 5
MAX_USAGE = 6
UNKNOWN = 2147483647
cp_mem_usage = ctypes.c_uint32 # enum
class struct_ion_flush_data(Structure):
    pass

struct_ion_flush_data._pack_ = 1 # source:False
struct_ion_flush_data._fields_ = [
    ('handle', ctypes.c_int32),
    ('fd', ctypes.c_int32),
    ('vaddr', ctypes.POINTER(None)),
    ('offset', ctypes.c_uint32),
    ('length', ctypes.c_uint32),
]

ION_IOC_CLEAN_CACHES = _IOWR ( 'M' , 0 , struct_ion_flush_data ) # macro (from list)
ION_IOC_INV_CACHES = _IOWR ( 'M' , 1 , struct_ion_flush_data ) # macro (from list)
ION_IOC_CLEAN_INV_CACHES = _IOWR ( 'M' , 2 , struct_ion_flush_data ) # macro (from list)
class struct_ion_prefetch_regions(Structure):
    pass

struct_ion_prefetch_regions._pack_ = 1 # source:False
struct_ion_prefetch_regions._fields_ = [
    ('vmid', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('sizes', ctypes.POINTER(ctypes.c_uint64)),
    ('nr_sizes', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

class struct_ion_prefetch_data(Structure):
    pass

struct_ion_prefetch_data._pack_ = 1 # source:False
struct_ion_prefetch_data._fields_ = [
    ('heap_id', ctypes.c_int32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('len', ctypes.c_uint64),
    ('regions', ctypes.POINTER(struct_ion_prefetch_regions)),
    ('nr_regions', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

ION_IOC_PREFETCH = _IOWR ( 'M' , 3 , struct_ion_prefetch_data ) # macro (from list)
ION_IOC_DRAIN = _IOWR ( 'M' , 4 , struct_ion_prefetch_data ) # macro (from list)
ADSPRPC_SHARED_H = True # macro
FASTRPC_GLINK_GUID = "fastrpcglink-apps-dsp" # macro
FASTRPC_SMD_GUID = "fastrpcsmd-apps-dsp" # macro
DEVICE_NAME = "adsprpc-smd" # macro
FASTRPC_ATTR_NOVA = 0x1 # macro
FASTRPC_ATTR_NON_COHERENT = 0x2 # macro
FASTRPC_ATTR_COHERENT = 0x4 # macro
FASTRPC_ATTR_KEEP_MAP = 0x8 # macro
FASTRPC_ATTR_NOMAP = (16) # macro
FASTRPC_MODE_PARALLEL = 0 # macro
FASTRPC_MODE_SERIAL = 1 # macro
FASTRPC_MODE_PROFILE = 2 # macro
FASTRPC_MODE_SESSION = 4 # macro
FASTRPC_INIT_ATTACH = 0 # macro
FASTRPC_INIT_CREATE = 1 # macro
FASTRPC_INIT_CREATE_STATIC = 2 # macro
FASTRPC_INIT_ATTACH_SENSORS = 3 # macro
def REMOTE_SCALARS_INBUFS(dwScalars):  # macro
   return (((dwScalars)>>16)&0x0ff)
def REMOTE_SCALARS_OUTBUFS(dwScalars):  # macro
   return (((dwScalars)>>8)&0x0ff)
def REMOTE_SCALARS_INHANDLES(dwScalars):  # macro
   return (((dwScalars)>>4)&0x0f)
def REMOTE_SCALARS_OUTHANDLES(dwScalars):  # macro
   return ((dwScalars)&0x0f)
def REMOTE_SCALARS_LENGTH(sc):  # macro
   return (REMOTE_SCALARS_INBUFS(sc)+REMOTE_SCALARS_OUTBUFS(sc)+REMOTE_SCALARS_INHANDLES(sc)+REMOTE_SCALARS_OUTHANDLES(sc))
def REMOTE_SCALARS_MAKEX(nAttr, nMethod, nIn, nOut, noIn, noOut):  # macro
   return ((((nAttr)&0x7)<<29)|(((nMethod)&0x1f)<<24)|(((nIn)&0xff)<<16)|(((nOut)&0xff)<<8)|(((noIn)&0x0f)<<4)|((noOut)&0x0f))
def REMOTE_SCALARS_MAKE(nMethod, nIn, nOut):  # macro
   return REMOTE_SCALARS_MAKEX(0,nMethod,nIn,nOut,0,0)
# def VERIFY_EPRINTF(format, args):  # macro
#    return (void)0
# def VERIFY_IPRINTF(args):  # macro
#    return (void)0
# def __STR__(x):  # macro
#    return #x":"
def __TOSTR__(x):  # macro
   return __STR__(x)
# __FILE_LINE__ = __FILE__ ":" __TOSTR__ ( __LINE__ ) # macro
# def VERIFY(err, val):  # macro
#    return {VERIFY_IPRINTF(__FILE__":"__TOSTR__(__LINE__)"info: calling: "#val"\n");((val)==0){(err)=(err)==0?-1:(err);VERIFY_EPRINTF(__FILE__":"__TOSTR__(__LINE__)"error: %d: "#val"\n",(err));}{VERIFY_IPRINTF(__FILE__":"__TOSTR__(__LINE__)"info: passed: "#val"\n");}\}(0)
# remote_arg64_t = remote_arg64 # macro
FASTRPC_CONTROL_LATENCY = (1) # macro
FASTRPC_CONTROL_SMMU = (2) # macro
FASTRPC_CONTROL_KALLOC = (3) # macro
class struct_remote_buf64(Structure):
    pass

struct_remote_buf64._pack_ = 1 # source:False
struct_remote_buf64._fields_ = [
    ('pv', ctypes.c_uint64),
    ('len', ctypes.c_uint64),
]

class struct_remote_dma_handle64(Structure):
    pass

struct_remote_dma_handle64._pack_ = 1 # source:False
struct_remote_dma_handle64._fields_ = [
    ('fd', ctypes.c_int32),
    ('offset', ctypes.c_uint32),
    ('len', ctypes.c_uint32),
]

class union_remote_arg64(Union):
    pass

union_remote_arg64._pack_ = 1 # source:False
union_remote_arg64._fields_ = [
    ('buf', struct_remote_buf64),
    ('dma', struct_remote_dma_handle64),
    ('h', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 12),
]

class struct_remote_buf(Structure):
    pass

struct_remote_buf._pack_ = 1 # source:False
struct_remote_buf._fields_ = [
    ('pv', ctypes.POINTER(None)),
    ('len', ctypes.c_uint64),
]

class struct_remote_dma_handle(Structure):
    pass

struct_remote_dma_handle._pack_ = 1 # source:False
struct_remote_dma_handle._fields_ = [
    ('fd', ctypes.c_int32),
    ('offset', ctypes.c_uint32),
]

class union_remote_arg(Union):
    pass

union_remote_arg._pack_ = 1 # source:False
union_remote_arg._fields_ = [
    ('buf', struct_remote_buf),
    ('dma', struct_remote_dma_handle),
    ('h', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 12),
]

class struct_fastrpc_ioctl_invoke(Structure):
    pass

struct_fastrpc_ioctl_invoke._pack_ = 1 # source:False
struct_fastrpc_ioctl_invoke._fields_ = [
    ('handle', ctypes.c_uint32),
    ('sc', ctypes.c_uint32),
    ('pra', ctypes.POINTER(union_remote_arg)),
]

FASTRPC_IOCTL_INVOKE = _IOWR ( 'R' , 1 , struct_fastrpc_ioctl_invoke ) # macro (from list)
class struct_fastrpc_ioctl_invoke_fd(Structure):
    pass

struct_fastrpc_ioctl_invoke_fd._pack_ = 1 # source:False
struct_fastrpc_ioctl_invoke_fd._fields_ = [
    ('inv', struct_fastrpc_ioctl_invoke),
    ('fds', ctypes.POINTER(ctypes.c_int32)),
]

FASTRPC_IOCTL_INVOKE_FD = _IOWR ( 'R' , 4 , struct_fastrpc_ioctl_invoke_fd ) # macro (from list)
class struct_fastrpc_ioctl_invoke_attrs(Structure):
    pass

struct_fastrpc_ioctl_invoke_attrs._pack_ = 1 # source:False
struct_fastrpc_ioctl_invoke_attrs._fields_ = [
    ('inv', struct_fastrpc_ioctl_invoke),
    ('fds', ctypes.POINTER(ctypes.c_int32)),
    ('attrs', ctypes.POINTER(ctypes.c_uint32)),
]

FASTRPC_IOCTL_INVOKE_ATTRS = _IOWR ( 'R' , 7 , struct_fastrpc_ioctl_invoke_attrs ) # macro (from list)
class struct_fastrpc_ioctl_invoke_crc(Structure):
    pass

struct_fastrpc_ioctl_invoke_crc._pack_ = 1 # source:False
struct_fastrpc_ioctl_invoke_crc._fields_ = [
    ('inv', struct_fastrpc_ioctl_invoke),
    ('fds', ctypes.POINTER(ctypes.c_int32)),
    ('attrs', ctypes.POINTER(ctypes.c_uint32)),
    ('crc', ctypes.POINTER(ctypes.c_uint32)),
]

FASTRPC_IOCTL_INVOKE_CRC = _IOWR ( 'R' , 11 , struct_fastrpc_ioctl_invoke_crc ) # macro (from list)
class struct_fastrpc_ioctl_init(Structure):
    pass

struct_fastrpc_ioctl_init._pack_ = 1 # source:False
struct_fastrpc_ioctl_init._fields_ = [
    ('flags', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('file', ctypes.c_uint64),
    ('filelen', ctypes.c_uint32),
    ('filefd', ctypes.c_int32),
    ('mem', ctypes.c_uint64),
    ('memlen', ctypes.c_uint32),
    ('memfd', ctypes.c_int32),
]

FASTRPC_IOCTL_INIT = _IOWR ( 'R' , 6 , struct_fastrpc_ioctl_init ) # macro (from list)
class struct_fastrpc_ioctl_init_attrs(Structure):
    pass

struct_fastrpc_ioctl_init_attrs._pack_ = 1 # source:False
struct_fastrpc_ioctl_init_attrs._fields_ = [
    ('init', struct_fastrpc_ioctl_init),
    ('attrs', ctypes.c_int32),
    ('siglen', ctypes.c_uint32),
]

FASTRPC_IOCTL_INIT_ATTRS = _IOWR ( 'R' , 10 , struct_fastrpc_ioctl_init_attrs ) # macro (from list)
class struct_fastrpc_ioctl_munmap(Structure):
    pass

struct_fastrpc_ioctl_munmap._pack_ = 1 # source:False
struct_fastrpc_ioctl_munmap._fields_ = [
    ('vaddrout', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
]

FASTRPC_IOCTL_MUNMAP = _IOWR ( 'R' , 3 , struct_fastrpc_ioctl_munmap ) # macro (from list)
class struct_fastrpc_ioctl_munmap_64(Structure):
    pass

struct_fastrpc_ioctl_munmap_64._pack_ = 1 # source:False
struct_fastrpc_ioctl_munmap_64._fields_ = [
    ('vaddrout', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
]

FASTRPC_IOCTL_MUNMAP_64 = _IOWR ( 'R' , 15 , struct_fastrpc_ioctl_munmap_64 ) # macro (from list)
class struct_fastrpc_ioctl_mmap(Structure):
    pass

struct_fastrpc_ioctl_mmap._pack_ = 1 # source:False
struct_fastrpc_ioctl_mmap._fields_ = [
    ('fd', ctypes.c_int32),
    ('flags', ctypes.c_uint32),
    ('vaddrin', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
    ('vaddrout', ctypes.c_uint64),
]

FASTRPC_IOCTL_MMAP = _IOWR ( 'R' , 2 , struct_fastrpc_ioctl_mmap ) # macro (from list)
class struct_fastrpc_ioctl_mmap_64(Structure):
    pass

struct_fastrpc_ioctl_mmap_64._pack_ = 1 # source:False
struct_fastrpc_ioctl_mmap_64._fields_ = [
    ('fd', ctypes.c_int32),
    ('flags', ctypes.c_uint32),
    ('vaddrin', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
    ('vaddrout', ctypes.c_uint64),
]

FASTRPC_IOCTL_MMAP_64 = _IOWR ( 'R' , 14 , struct_fastrpc_ioctl_mmap_64 ) # macro (from list)
class struct_fastrpc_ioctl_munmap_fd(Structure):
    pass

struct_fastrpc_ioctl_munmap_fd._pack_ = 1 # source:False
struct_fastrpc_ioctl_munmap_fd._fields_ = [
    ('fd', ctypes.c_int32),
    ('flags', ctypes.c_uint32),
    ('va', ctypes.c_uint64),
    ('len', ctypes.c_int64),
]

FASTRPC_IOCTL_MUNMAP_FD = _IOWR ( 'R' , 13 , struct_fastrpc_ioctl_munmap_fd ) # macro (from list)
class struct_fastrpc_ioctl_perf(Structure):
    pass

struct_fastrpc_ioctl_perf._pack_ = 1 # source:False
struct_fastrpc_ioctl_perf._fields_ = [
    ('data', ctypes.c_uint64),
    ('numkeys', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('keys', ctypes.c_uint64),
]

FASTRPC_IOCTL_GETPERF = _IOWR ( 'R' , 9 , struct_fastrpc_ioctl_perf ) # macro (from list)
class struct_fastrpc_ctrl_latency(Structure):
    pass

struct_fastrpc_ctrl_latency._pack_ = 1 # source:False
struct_fastrpc_ctrl_latency._fields_ = [
    ('enable', ctypes.c_uint32),
    ('level', ctypes.c_uint32),
]

class struct_fastrpc_ctrl_smmu(Structure):
    pass

struct_fastrpc_ctrl_smmu._pack_ = 1 # source:False
struct_fastrpc_ctrl_smmu._fields_ = [
    ('sharedcb', ctypes.c_uint32),
]

class struct_fastrpc_ctrl_kalloc(Structure):
    pass

struct_fastrpc_ctrl_kalloc._pack_ = 1 # source:False
struct_fastrpc_ctrl_kalloc._fields_ = [
    ('kalloc_support', ctypes.c_uint32),
]

class struct_fastrpc_ioctl_control(Structure):
    pass

class union_fastrpc_ioctl_control_0(Union):
    pass

union_fastrpc_ioctl_control_0._pack_ = 1 # source:False
union_fastrpc_ioctl_control_0._fields_ = [
    ('lp', struct_fastrpc_ctrl_latency),
    ('smmu', struct_fastrpc_ctrl_smmu),
    ('kalloc', struct_fastrpc_ctrl_kalloc),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

struct_fastrpc_ioctl_control._pack_ = 1 # source:False
struct_fastrpc_ioctl_control._anonymous_ = ('_0',)
struct_fastrpc_ioctl_control._fields_ = [
    ('req', ctypes.c_uint32),
    ('_0', union_fastrpc_ioctl_control_0),
]

FASTRPC_IOCTL_CONTROL = _IOWR ( 'R' , 12 , struct_fastrpc_ioctl_control ) # macro (from list)
class struct_smq_null_invoke(Structure):
    pass

struct_smq_null_invoke._pack_ = 1 # source:False
struct_smq_null_invoke._fields_ = [
    ('ctx', ctypes.c_uint64),
    ('handle', ctypes.c_uint32),
    ('sc', ctypes.c_uint32),
]

class struct_smq_phy_page(Structure):
    pass

struct_smq_phy_page._pack_ = 1 # source:False
struct_smq_phy_page._fields_ = [
    ('addr', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
]

class struct_smq_invoke_buf(Structure):
    pass

struct_smq_invoke_buf._pack_ = 1 # source:False
struct_smq_invoke_buf._fields_ = [
    ('num', ctypes.c_int32),
    ('pgidx', ctypes.c_int32),
]

class struct_smq_invoke(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('header', struct_smq_null_invoke),
    ('page', struct_smq_phy_page),
     ]

class struct_smq_msg(Structure):
    pass

struct_smq_msg._pack_ = 1 # source:False
struct_smq_msg._fields_ = [
    ('pid', ctypes.c_uint32),
    ('tid', ctypes.c_uint32),
    ('invoke', struct_smq_invoke),
]

class struct_smq_invoke_rsp(Structure):
    pass

struct_smq_invoke_rsp._pack_ = 1 # source:False
struct_smq_invoke_rsp._fields_ = [
    ('ctx', ctypes.c_uint64),
    ('retval', ctypes.c_int32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

uint32_t = ctypes.c_uint32
FASTRPC_IOCTL_SETMODE = _IOWR ( 'R' , 5 , uint32_t ) # macro (from list)
FASTRPC_IOCTL_GETINFO = _IOWR ( 'R' , 8 , uint32_t ) # macro (from list)
try:
    smq_invoke_buf_start = _libraries['FIXME_STUB'].smq_invoke_buf_start
    smq_invoke_buf_start.restype = ctypes.POINTER(struct_smq_invoke_buf)
    smq_invoke_buf_start.argtypes = [ctypes.POINTER(union_remote_arg64), uint32_t]
except AttributeError:
    pass
try:
    smq_phy_page_start = _libraries['FIXME_STUB'].smq_phy_page_start
    smq_phy_page_start.restype = ctypes.POINTER(struct_smq_phy_page)
    smq_phy_page_start.argtypes = [uint32_t, ctypes.POINTER(struct_smq_invoke_buf)]
except AttributeError:
    pass
REMOTE_DEFAULT_H = True # macro
def REMOTE_SCALARS_METHOD_ATTR(dwScalars):  # macro
   return (((dwScalars)>>29)&0x7)
def REMOTE_SCALARS_METHOD(dwScalars):  # macro
   return (((dwScalars)>>24)&0x1f)
def __QAIC_REMOTE(ff):  # macro
   return ff
__QAIC_REMOTE_EXPORT = True # macro
__QAIC_REMOTE_ATTRIBUTE = True # macro
NUM_DOMAINS = 4 # macro
NUM_SESSIONS = 2 # macro
DOMAIN_ID_MASK = 3 # macro
DEFAULT_DOMAIN_ID = 0 # macro
ADSP_DOMAIN_ID = 0 # macro
MDSP_DOMAIN_ID = 1 # macro
SDSP_DOMAIN_ID = 2 # macro
CDSP_DOMAIN_ID = 3 # macro
ADSP_DOMAIN = "&_dom=adsp" # macro
MDSP_DOMAIN = "&_dom=mdsp" # macro
SDSP_DOMAIN = "&_dom=sdsp" # macro
CDSP_DOMAIN = "&_dom=cdsp" # macro
FASTRPC_WAKELOCK_CONTROL_SUPPORTED = 1 # macro
REMOTE_MODE_PARALLEL = 0 # macro
REMOTE_MODE_SERIAL = 1 # macro
# ITRANSPORT_PREFIX = "'\":;./\\" # macro
remote_handle = ctypes.c_uint32
remote_handle64 = ctypes.c_uint64
fastrpc_async_jobid = ctypes.c_uint64
class struct_c__SA_remote_buf(Structure):
    pass

struct_c__SA_remote_buf._pack_ = 1 # source:False
struct_c__SA_remote_buf._fields_ = [
    ('pv', ctypes.POINTER(None)),
    ('nLen', ctypes.c_uint64),
]

remote_buf = struct_c__SA_remote_buf
class struct_c__SA_remote_dma_handle(Structure):
    pass

struct_c__SA_remote_dma_handle._pack_ = 1 # source:False
struct_c__SA_remote_dma_handle._fields_ = [
    ('fd', ctypes.c_int32),
    ('offset', ctypes.c_uint32),
]

remote_dma_handle = struct_c__SA_remote_dma_handle
class union_c__UA_remote_arg(Union):
    pass

union_c__UA_remote_arg._pack_ = 1 # source:False
union_c__UA_remote_arg._fields_ = [
    ('buf', remote_buf),
    ('h', ctypes.c_uint32),
    ('h64', ctypes.c_uint64),
    ('dma', remote_dma_handle),
    ('PADDING_0', ctypes.c_ubyte * 8),
]

remote_arg = union_c__UA_remote_arg
remote_arg_t = remote_arg # macro

# values for enumeration 'fastrpc_async_notify_type'
fastrpc_async_notify_type__enumvalues = {
    0: 'FASTRPC_ASYNC_NO_SYNC',
    1: 'FASTRPC_ASYNC_CALLBACK',
    2: 'FASTRPC_ASYNC_POLL',
    3: 'FASTRPC_ASYNC_TYPE_MAX',
}
FASTRPC_ASYNC_NO_SYNC = 0
FASTRPC_ASYNC_CALLBACK = 1
FASTRPC_ASYNC_POLL = 2
FASTRPC_ASYNC_TYPE_MAX = 3
fastrpc_async_notify_type = ctypes.c_uint32 # enum
class struct_fastrpc_async_callback(Structure):
    pass

struct_fastrpc_async_callback._pack_ = 1 # source:False
struct_fastrpc_async_callback._fields_ = [
    ('fn', ctypes.CFUNCTYPE(None, ctypes.c_uint64, ctypes.POINTER(None), ctypes.c_int32)),
    ('context', ctypes.POINTER(None)),
]

fastrpc_async_callback_t = struct_fastrpc_async_callback
class struct_fastrpc_async_descriptor(Structure):
    pass

class union_fastrpc_async_descriptor_0(Union):
    _pack_ = 1 # source:False
    _fields_ = [
    ('cb', fastrpc_async_callback_t),
     ]

struct_fastrpc_async_descriptor._pack_ = 1 # source:False
struct_fastrpc_async_descriptor._anonymous_ = ('_0',)
struct_fastrpc_async_descriptor._fields_ = [
    ('type', fastrpc_async_notify_type),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('jobid', ctypes.c_uint64),
    ('_0', union_fastrpc_async_descriptor_0),
]

fastrpc_async_descriptor_t = struct_fastrpc_async_descriptor

# values for enumeration 'fastrpc_process_type'
fastrpc_process_type__enumvalues = {
    0: 'PROCESS_TYPE_SIGNED',
    1: 'PROCESS_TYPE_UNSIGNED',
}
PROCESS_TYPE_SIGNED = 0
PROCESS_TYPE_UNSIGNED = 1
fastrpc_process_type = ctypes.c_uint32 # enum
try:
    remote_handle_open = _libraries['FIXME_STUB'].remote_handle_open
    remote_handle_open.restype = ctypes.c_int32
    remote_handle_open.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    remote_handle64_open = _libraries['FIXME_STUB'].remote_handle64_open
    remote_handle64_open.restype = ctypes.c_int32
    remote_handle64_open.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    remote_handle_invoke = _libraries['FIXME_STUB'].remote_handle_invoke
    remote_handle_invoke.restype = ctypes.c_int32
    remote_handle_invoke.argtypes = [remote_handle, uint32_t, ctypes.POINTER(union_c__UA_remote_arg)]
except AttributeError:
    pass
try:
    remote_handle64_invoke = _libraries['FIXME_STUB'].remote_handle64_invoke
    remote_handle64_invoke.restype = ctypes.c_int32
    remote_handle64_invoke.argtypes = [remote_handle64, uint32_t, ctypes.POINTER(union_c__UA_remote_arg)]
except AttributeError:
    pass
try:
    remote_handle_invoke_async = _libraries['FIXME_STUB'].remote_handle_invoke_async
    remote_handle_invoke_async.restype = ctypes.c_int32
    remote_handle_invoke_async.argtypes = [remote_handle, ctypes.POINTER(struct_fastrpc_async_descriptor), uint32_t, ctypes.POINTER(union_c__UA_remote_arg)]
except AttributeError:
    pass
try:
    remote_handle64_invoke_async = _libraries['FIXME_STUB'].remote_handle64_invoke_async
    remote_handle64_invoke_async.restype = ctypes.c_int32
    remote_handle64_invoke_async.argtypes = [remote_handle64, ctypes.POINTER(struct_fastrpc_async_descriptor), uint32_t, ctypes.POINTER(union_c__UA_remote_arg)]
except AttributeError:
    pass
try:
    remote_handle_close = _libraries['FIXME_STUB'].remote_handle_close
    remote_handle_close.restype = ctypes.c_int32
    remote_handle_close.argtypes = [remote_handle]
except AttributeError:
    pass
try:
    remote_handle64_close = _libraries['FIXME_STUB'].remote_handle64_close
    remote_handle64_close.restype = ctypes.c_int32
    remote_handle64_close.argtypes = [remote_handle64]
except AttributeError:
    pass

# values for enumeration 'handle_control_req_id'
handle_control_req_id__enumvalues = {
    1: 'DSPRPC_CONTROL_LATENCY',
    2: 'DSPRPC_GET_DSP_INFO',
    3: 'DSPRPC_CONTROL_WAKELOCK',
    4: 'DSPRPC_GET_DOMAIN',
}
DSPRPC_CONTROL_LATENCY = 1
DSPRPC_GET_DSP_INFO = 2
DSPRPC_CONTROL_WAKELOCK = 3
DSPRPC_GET_DOMAIN = 4
handle_control_req_id = ctypes.c_uint32 # enum

# values for enumeration 'remote_rpc_latency_flags'
remote_rpc_latency_flags__enumvalues = {
    0: 'RPC_DISABLE_QOS',
    1: 'RPC_PM_QOS',
    2: 'RPC_ADAPTIVE_QOS',
    3: 'RPC_POLL_QOS',
}
RPC_DISABLE_QOS = 0
RPC_PM_QOS = 1
RPC_ADAPTIVE_QOS = 2
RPC_POLL_QOS = 3
remote_rpc_latency_flags = ctypes.c_uint32 # enum
remote_rpc_control_latency_t = remote_rpc_latency_flags
remote_rpc_control_latency_t__enumvalues = remote_rpc_latency_flags__enumvalues
class struct_remote_rpc_control_latency(Structure):
    pass

struct_remote_rpc_control_latency._pack_ = 1 # source:False
struct_remote_rpc_control_latency._fields_ = [
    ('enable', remote_rpc_control_latency_t),
    ('latency', ctypes.c_uint32),
]


# values for enumeration 'remote_dsp_attributes'
remote_dsp_attributes__enumvalues = {
    0: 'DOMAIN_SUPPORT',
    1: 'UNSIGNED_PD_SUPPORT',
    2: 'HVX_SUPPORT_64B',
    3: 'HVX_SUPPORT_128B',
    4: 'VTCM_PAGE',
    5: 'VTCM_COUNT',
    6: 'ARCH_VER',
    7: 'HMX_SUPPORT_DEPTH',
    8: 'HMX_SUPPORT_SPATIAL',
    9: 'ASYNC_FASTRPC_SUPPORT',
    10: 'STATUS_NOTIFICATION_SUPPORT',
    11: 'FASTRPC_MAX_DSP_ATTRIBUTES',
}
DOMAIN_SUPPORT = 0
UNSIGNED_PD_SUPPORT = 1
HVX_SUPPORT_64B = 2
HVX_SUPPORT_128B = 3
VTCM_PAGE = 4
VTCM_COUNT = 5
ARCH_VER = 6
HMX_SUPPORT_DEPTH = 7
HMX_SUPPORT_SPATIAL = 8
ASYNC_FASTRPC_SUPPORT = 9
STATUS_NOTIFICATION_SUPPORT = 10
FASTRPC_MAX_DSP_ATTRIBUTES = 11
remote_dsp_attributes = ctypes.c_uint32 # enum
class struct_remote_dsp_capability(Structure):
    pass

struct_remote_dsp_capability._pack_ = 1 # source:False
struct_remote_dsp_capability._fields_ = [
    ('domain', ctypes.c_uint32),
    ('attribute_ID', ctypes.c_uint32),
    ('capability', ctypes.c_uint32),
]

fastrpc_capability = struct_remote_dsp_capability
class struct_remote_rpc_control_wakelock(Structure):
    pass

struct_remote_rpc_control_wakelock._pack_ = 1 # source:False
struct_remote_rpc_control_wakelock._fields_ = [
    ('enable', ctypes.c_uint32),
]

class struct_remote_rpc_get_domain(Structure):
    pass

struct_remote_rpc_get_domain._pack_ = 1 # source:False
struct_remote_rpc_get_domain._fields_ = [
    ('domain', ctypes.c_int32),
]

remote_rpc_get_domain_t = struct_remote_rpc_get_domain
try:
    remote_handle_control = _libraries['FIXME_STUB'].remote_handle_control
    remote_handle_control.restype = ctypes.c_int32
    remote_handle_control.argtypes = [uint32_t, ctypes.POINTER(None), uint32_t]
except AttributeError:
    pass
try:
    remote_handle64_control = _libraries['FIXME_STUB'].remote_handle64_control
    remote_handle64_control.restype = ctypes.c_int32
    remote_handle64_control.argtypes = [remote_handle64, uint32_t, ctypes.POINTER(None), uint32_t]
except AttributeError:
    pass

# values for enumeration 'session_control_req_id'
session_control_req_id__enumvalues = {
    1: 'FASTRPC_THREAD_PARAMS',
    2: 'DSPRPC_CONTROL_UNSIGNED_MODULE',
    4: 'FASTRPC_RELATIVE_THREAD_PRIORITY',
    6: 'FASTRPC_REMOTE_PROCESS_KILL',
    7: 'FASTRPC_SESSION_CLOSE',
    8: 'FASTRPC_CONTROL_PD_DUMP',
    9: 'FASTRPC_REMOTE_PROCESS_EXCEPTION',
    10: 'FASTRPC_REMOTE_PROCESS_TYPE',
    11: 'FASTRPC_REGISTER_STATUS_NOTIFICATIONS',
}
FASTRPC_THREAD_PARAMS = 1
DSPRPC_CONTROL_UNSIGNED_MODULE = 2
FASTRPC_RELATIVE_THREAD_PRIORITY = 4
FASTRPC_REMOTE_PROCESS_KILL = 6
FASTRPC_SESSION_CLOSE = 7
FASTRPC_CONTROL_PD_DUMP = 8
FASTRPC_REMOTE_PROCESS_EXCEPTION = 9
FASTRPC_REMOTE_PROCESS_TYPE = 10
FASTRPC_REGISTER_STATUS_NOTIFICATIONS = 11
session_control_req_id = ctypes.c_uint32 # enum
class struct_remote_rpc_thread_params(Structure):
    pass

struct_remote_rpc_thread_params._pack_ = 1 # source:False
struct_remote_rpc_thread_params._fields_ = [
    ('domain', ctypes.c_int32),
    ('prio', ctypes.c_int32),
    ('stack_size', ctypes.c_int32),
]

class struct_remote_rpc_control_unsigned_module(Structure):
    pass

struct_remote_rpc_control_unsigned_module._pack_ = 1 # source:False
struct_remote_rpc_control_unsigned_module._fields_ = [
    ('domain', ctypes.c_int32),
    ('enable', ctypes.c_int32),
]

class struct_remote_rpc_relative_thread_priority(Structure):
    pass

struct_remote_rpc_relative_thread_priority._pack_ = 1 # source:False
struct_remote_rpc_relative_thread_priority._fields_ = [
    ('domain', ctypes.c_int32),
    ('relative_thread_priority', ctypes.c_int32),
]

class struct_remote_rpc_process_clean_params(Structure):
    pass

struct_remote_rpc_process_clean_params._pack_ = 1 # source:False
struct_remote_rpc_process_clean_params._fields_ = [
    ('domain', ctypes.c_int32),
]

class struct_remote_rpc_session_close(Structure):
    pass

struct_remote_rpc_session_close._pack_ = 1 # source:False
struct_remote_rpc_session_close._fields_ = [
    ('domain', ctypes.c_int32),
]

class struct_remote_rpc_control_pd_dump(Structure):
    pass

struct_remote_rpc_control_pd_dump._pack_ = 1 # source:False
struct_remote_rpc_control_pd_dump._fields_ = [
    ('domain', ctypes.c_int32),
    ('enable', ctypes.c_int32),
]

class struct_remote_process_type(Structure):
    pass

struct_remote_process_type._pack_ = 1 # source:False
struct_remote_process_type._fields_ = [
    ('domain', ctypes.c_int32),
    ('process_type', ctypes.c_int32),
]

remote_rpc_process_exception = struct_remote_rpc_process_clean_params

# values for enumeration 'remote_rpc_status_flags'
remote_rpc_status_flags__enumvalues = {
    0: 'FASTRPC_USER_PD_UP',
    1: 'FASTRPC_USER_PD_EXIT',
    2: 'FASTRPC_USER_PD_FORCE_KILL',
    3: 'FASTRPC_USER_PD_EXCEPTION',
    4: 'FASTRPC_DSP_SSR',
}
FASTRPC_USER_PD_UP = 0
FASTRPC_USER_PD_EXIT = 1
FASTRPC_USER_PD_FORCE_KILL = 2
FASTRPC_USER_PD_EXCEPTION = 3
FASTRPC_DSP_SSR = 4
remote_rpc_status_flags = ctypes.c_uint32 # enum
remote_rpc_status_flags_t = remote_rpc_status_flags
remote_rpc_status_flags_t__enumvalues = remote_rpc_status_flags__enumvalues
fastrpc_notif_fn_t = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(None), ctypes.c_int32, ctypes.c_int32, remote_rpc_status_flags)
class struct_remote_rpc_notif_register(Structure):
    pass

struct_remote_rpc_notif_register._pack_ = 1 # source:False
struct_remote_rpc_notif_register._fields_ = [
    ('context', ctypes.POINTER(None)),
    ('domain', ctypes.c_int32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('notifier_fn', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(None), ctypes.c_int32, ctypes.c_int32, remote_rpc_status_flags)),
]

remote_rpc_notif_register_t = struct_remote_rpc_notif_register
try:
    remote_session_control = _libraries['FIXME_STUB'].remote_session_control
    remote_session_control.restype = ctypes.c_int32
    remote_session_control.argtypes = [uint32_t, ctypes.POINTER(None), uint32_t]
except AttributeError:
    pass
try:
    remote_mmap = _libraries['FIXME_STUB'].remote_mmap
    remote_mmap.restype = ctypes.c_int32
    remote_mmap.argtypes = [ctypes.c_int32, uint32_t, uint32_t, ctypes.c_int32, ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    remote_munmap = _libraries['FIXME_STUB'].remote_munmap
    remote_munmap.restype = ctypes.c_int32
    remote_munmap.argtypes = [uint32_t, ctypes.c_int32]
except AttributeError:
    pass

# values for enumeration 'remote_mem_map_flags'
remote_mem_map_flags__enumvalues = {
    0: 'REMOTE_MAP_MEM_STATIC',
    1: 'REMOTE_MAP_MAX_FLAG',
}
REMOTE_MAP_MEM_STATIC = 0
REMOTE_MAP_MAX_FLAG = 1
remote_mem_map_flags = ctypes.c_uint32 # enum
uint64_t = ctypes.c_uint64
size_t = ctypes.c_uint64
try:
    remote_mem_map = _libraries['FIXME_STUB'].remote_mem_map
    remote_mem_map.restype = ctypes.c_int32
    remote_mem_map.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, uint64_t, size_t, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    remote_mem_unmap = _libraries['FIXME_STUB'].remote_mem_unmap
    remote_mem_unmap.restype = ctypes.c_int32
    remote_mem_unmap.argtypes = [ctypes.c_int32, uint64_t, size_t]
except AttributeError:
    pass

# values for enumeration 'remote_buf_attributes'
remote_buf_attributes__enumvalues = {
    2: 'FASTRPC_ATTR_NON_COHERENT',
    4: 'FASTRPC_ATTR_COHERENT',
    8: 'FASTRPC_ATTR_KEEP_MAP',
    16: 'FASTRPC_ATTR_NOMAP',
    32: 'FASTRPC_ATTR_FORCE_NOFLUSH',
    64: 'FASTRPC_ATTR_FORCE_NOINVALIDATE',
    128: 'FASTRPC_ATTR_TRY_MAP_STATIC',
}
FASTRPC_ATTR_NON_COHERENT = 2
FASTRPC_ATTR_COHERENT = 4
FASTRPC_ATTR_KEEP_MAP = 8
FASTRPC_ATTR_NOMAP = 16
FASTRPC_ATTR_FORCE_NOFLUSH = 32
FASTRPC_ATTR_FORCE_NOINVALIDATE = 64
FASTRPC_ATTR_TRY_MAP_STATIC = 128
remote_buf_attributes = ctypes.c_uint32 # enum
try:
    remote_register_buf_attr = _libraries['FIXME_STUB'].remote_register_buf_attr
    remote_register_buf_attr.restype = None
    remote_register_buf_attr.argtypes = [ctypes.POINTER(None), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
except AttributeError:
    pass
try:
    remote_register_buf = _libraries['FIXME_STUB'].remote_register_buf
    remote_register_buf.restype = None
    remote_register_buf.argtypes = [ctypes.POINTER(None), ctypes.c_int32, ctypes.c_int32]
except AttributeError:
    pass
try:
    remote_register_dma_handle_attr = _libraries['FIXME_STUB'].remote_register_dma_handle_attr
    remote_register_dma_handle_attr.restype = ctypes.c_int32
    remote_register_dma_handle_attr.argtypes = [ctypes.c_int32, uint32_t, uint32_t]
except AttributeError:
    pass
try:
    remote_register_dma_handle = _libraries['FIXME_STUB'].remote_register_dma_handle
    remote_register_dma_handle.restype = ctypes.c_int32
    remote_register_dma_handle.argtypes = [ctypes.c_int32, uint32_t]
except AttributeError:
    pass
try:
    remote_register_fd = _libraries['FIXME_STUB'].remote_register_fd
    remote_register_fd.restype = ctypes.POINTER(None)
    remote_register_fd.argtypes = [ctypes.c_int32, ctypes.c_int32]
except AttributeError:
    pass
try:
    fastrpc_async_get_status = _libraries['FIXME_STUB'].fastrpc_async_get_status
    fastrpc_async_get_status.restype = ctypes.c_int32
    fastrpc_async_get_status.argtypes = [fastrpc_async_jobid, ctypes.c_int32, ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    fastrpc_release_async_job = _libraries['FIXME_STUB'].fastrpc_release_async_job
    fastrpc_release_async_job.restype = ctypes.c_int32
    fastrpc_release_async_job.argtypes = [fastrpc_async_jobid]
except AttributeError:
    pass
try:
    remote_set_mode = _libraries['FIXME_STUB'].remote_set_mode
    remote_set_mode.restype = ctypes.c_int32
    remote_set_mode.argtypes = [uint32_t]
except AttributeError:
    pass

# values for enumeration 'fastrpc_map_flags'
fastrpc_map_flags__enumvalues = {
    0: 'FASTRPC_MAP_STATIC',
    1: 'FASTRPC_MAP_RESERVED',
    2: 'FASTRPC_MAP_FD',
    3: 'FASTRPC_MAP_FD_DELAYED',
    4: 'FASTRPC_MAP_MAX',
}
FASTRPC_MAP_STATIC = 0
FASTRPC_MAP_RESERVED = 1
FASTRPC_MAP_FD = 2
FASTRPC_MAP_FD_DELAYED = 3
FASTRPC_MAP_MAX = 4
fastrpc_map_flags = ctypes.c_uint32 # enum
try:
    fastrpc_mmap = _libraries['FIXME_STUB'].fastrpc_mmap
    fastrpc_mmap.restype = ctypes.c_int32
    fastrpc_mmap.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(None), ctypes.c_int32, size_t, fastrpc_map_flags]
except AttributeError:
    pass
try:
    fastrpc_munmap = _libraries['FIXME_STUB'].fastrpc_munmap
    fastrpc_munmap.restype = ctypes.c_int32
    fastrpc_munmap.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(None), size_t]
except AttributeError:
    pass
_APPS_STD_H = True # macro
def __QAIC_HEADER(ff):  # macro
   return ff
__QAIC_HEADER_EXPORT = True # macro
__QAIC_HEADER_ATTRIBUTE = True # macro
def __QAIC_IMPL(ff):  # macro
   return ff
__QAIC_IMPL_EXPORT = True # macro
__QAIC_IMPL_ATTRIBUTE = True # macro
__QAIC_STRING1_OBJECT_DEFINED__ = True # macro
__STRING1_OBJECT__ = True # macro
class struct__cstring1_s(Structure):
    pass

struct__cstring1_s._pack_ = 1 # source:False
struct__cstring1_s._fields_ = [
    ('data', ctypes.POINTER(ctypes.c_char)),
    ('dataLen', ctypes.c_int32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

_cstring1_t = struct__cstring1_s
apps_std_FILE = ctypes.c_int32

# values for enumeration 'apps_std_SEEK'
apps_std_SEEK__enumvalues = {
    0: 'APPS_STD_SEEK_SET',
    1: 'APPS_STD_SEEK_CUR',
    2: 'APPS_STD_SEEK_END',
    2147483647: '_32BIT_PLACEHOLDER_apps_std_SEEK',
}
APPS_STD_SEEK_SET = 0
APPS_STD_SEEK_CUR = 1
APPS_STD_SEEK_END = 2
_32BIT_PLACEHOLDER_apps_std_SEEK = 2147483647
apps_std_SEEK = ctypes.c_uint32 # enum
class struct_apps_std_DIR(Structure):
    pass

struct_apps_std_DIR._pack_ = 1 # source:False
struct_apps_std_DIR._fields_ = [
    ('handle', ctypes.c_uint64),
]

apps_std_DIR = struct_apps_std_DIR
class struct_apps_std_DIRENT(Structure):
    pass

struct_apps_std_DIRENT._pack_ = 1 # source:False
struct_apps_std_DIRENT._fields_ = [
    ('ino', ctypes.c_int32),
    ('name', ctypes.c_char * 255),
    ('PADDING_0', ctypes.c_ubyte),
]

apps_std_DIRENT = struct_apps_std_DIRENT
class struct_apps_std_STAT(Structure):
    pass

struct_apps_std_STAT._pack_ = 1 # source:False
struct_apps_std_STAT._fields_ = [
    ('tsz', ctypes.c_uint64),
    ('dev', ctypes.c_uint64),
    ('ino', ctypes.c_uint64),
    ('mode', ctypes.c_uint32),
    ('nlink', ctypes.c_uint32),
    ('rdev', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
    ('atime', ctypes.c_int64),
    ('atimensec', ctypes.c_int64),
    ('mtime', ctypes.c_int64),
    ('mtimensec', ctypes.c_int64),
    ('ctime', ctypes.c_int64),
    ('ctimensec', ctypes.c_int64),
]

apps_std_STAT = struct_apps_std_STAT
try:
    apps_std_fopen = _libraries['FIXME_STUB'].apps_std_fopen
    apps_std_fopen.restype = ctypes.c_int32
    apps_std_fopen.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    apps_std_freopen = _libraries['FIXME_STUB'].apps_std_freopen
    apps_std_freopen.restype = ctypes.c_int32
    apps_std_freopen.argtypes = [apps_std_FILE, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    apps_std_fflush = _libraries['FIXME_STUB'].apps_std_fflush
    apps_std_fflush.restype = ctypes.c_int32
    apps_std_fflush.argtypes = [apps_std_FILE]
except AttributeError:
    pass
try:
    apps_std_fclose = _libraries['FIXME_STUB'].apps_std_fclose
    apps_std_fclose.restype = ctypes.c_int32
    apps_std_fclose.argtypes = [apps_std_FILE]
except AttributeError:
    pass
try:
    apps_std_fread = _libraries['FIXME_STUB'].apps_std_fread
    apps_std_fread.restype = ctypes.c_int32
    apps_std_fread.argtypes = [apps_std_FILE, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int32, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    apps_std_fwrite = _libraries['FIXME_STUB'].apps_std_fwrite
    apps_std_fwrite.restype = ctypes.c_int32
    apps_std_fwrite.argtypes = [apps_std_FILE, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int32, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    apps_std_fgetpos = _libraries['FIXME_STUB'].apps_std_fgetpos
    apps_std_fgetpos.restype = ctypes.c_int32
    apps_std_fgetpos.argtypes = [apps_std_FILE, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int32, ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    apps_std_fsetpos = _libraries['FIXME_STUB'].apps_std_fsetpos
    apps_std_fsetpos.restype = ctypes.c_int32
    apps_std_fsetpos.argtypes = [apps_std_FILE, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int32]
except AttributeError:
    pass
try:
    apps_std_ftell = _libraries['FIXME_STUB'].apps_std_ftell
    apps_std_ftell.restype = ctypes.c_int32
    apps_std_ftell.argtypes = [apps_std_FILE, ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    apps_std_fseek = _libraries['FIXME_STUB'].apps_std_fseek
    apps_std_fseek.restype = ctypes.c_int32
    apps_std_fseek.argtypes = [apps_std_FILE, ctypes.c_int32, apps_std_SEEK]
except AttributeError:
    pass
try:
    apps_std_flen = _libraries['FIXME_STUB'].apps_std_flen
    apps_std_flen.restype = ctypes.c_int32
    apps_std_flen.argtypes = [apps_std_FILE, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    apps_std_rewind = _libraries['FIXME_STUB'].apps_std_rewind
    apps_std_rewind.restype = ctypes.c_int32
    apps_std_rewind.argtypes = [apps_std_FILE]
except AttributeError:
    pass
try:
    apps_std_feof = _libraries['FIXME_STUB'].apps_std_feof
    apps_std_feof.restype = ctypes.c_int32
    apps_std_feof.argtypes = [apps_std_FILE, ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    apps_std_ferror = _libraries['FIXME_STUB'].apps_std_ferror
    apps_std_ferror.restype = ctypes.c_int32
    apps_std_ferror.argtypes = [apps_std_FILE, ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    apps_std_clearerr = _libraries['FIXME_STUB'].apps_std_clearerr
    apps_std_clearerr.restype = ctypes.c_int32
    apps_std_clearerr.argtypes = [apps_std_FILE]
except AttributeError:
    pass
try:
    apps_std_print_string = _libraries['FIXME_STUB'].apps_std_print_string
    apps_std_print_string.restype = ctypes.c_int32
    apps_std_print_string.argtypes = [ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    apps_std_getenv = _libraries['FIXME_STUB'].apps_std_getenv
    apps_std_getenv.restype = ctypes.c_int32
    apps_std_getenv.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), ctypes.c_int32, ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    apps_std_setenv = _libraries['FIXME_STUB'].apps_std_setenv
    apps_std_setenv.restype = ctypes.c_int32
    apps_std_setenv.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), ctypes.c_int32]
except AttributeError:
    pass
try:
    apps_std_unsetenv = _libraries['FIXME_STUB'].apps_std_unsetenv
    apps_std_unsetenv.restype = ctypes.c_int32
    apps_std_unsetenv.argtypes = [ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    apps_std_fopen_with_env = _libraries['FIXME_STUB'].apps_std_fopen_with_env
    apps_std_fopen_with_env.restype = ctypes.c_int32
    apps_std_fopen_with_env.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    apps_std_fgets = _libraries['FIXME_STUB'].apps_std_fgets
    apps_std_fgets.restype = ctypes.c_int32
    apps_std_fgets.argtypes = [apps_std_FILE, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int32, ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    apps_std_get_search_paths_with_env = _libraries['FIXME_STUB'].apps_std_get_search_paths_with_env
    apps_std_get_search_paths_with_env.restype = ctypes.c_int32
    apps_std_get_search_paths_with_env.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(struct__cstring1_s), ctypes.c_int32, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint16)]
except AttributeError:
    pass
try:
    apps_std_fileExists = _libraries['FIXME_STUB'].apps_std_fileExists
    apps_std_fileExists.restype = ctypes.c_int32
    apps_std_fileExists.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_ubyte)]
except AttributeError:
    pass
try:
    apps_std_fsync = _libraries['FIXME_STUB'].apps_std_fsync
    apps_std_fsync.restype = ctypes.c_int32
    apps_std_fsync.argtypes = [apps_std_FILE]
except AttributeError:
    pass
try:
    apps_std_fremove = _libraries['FIXME_STUB'].apps_std_fremove
    apps_std_fremove.restype = ctypes.c_int32
    apps_std_fremove.argtypes = [ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    apps_std_fdopen_decrypt = _libraries['FIXME_STUB'].apps_std_fdopen_decrypt
    apps_std_fdopen_decrypt.restype = ctypes.c_int32
    apps_std_fdopen_decrypt.argtypes = [apps_std_FILE, ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    apps_std_opendir = _libraries['FIXME_STUB'].apps_std_opendir
    apps_std_opendir.restype = ctypes.c_int32
    apps_std_opendir.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(struct_apps_std_DIR)]
except AttributeError:
    pass
try:
    apps_std_closedir = _libraries['FIXME_STUB'].apps_std_closedir
    apps_std_closedir.restype = ctypes.c_int32
    apps_std_closedir.argtypes = [ctypes.POINTER(struct_apps_std_DIR)]
except AttributeError:
    pass
try:
    apps_std_readdir = _libraries['FIXME_STUB'].apps_std_readdir
    apps_std_readdir.restype = ctypes.c_int32
    apps_std_readdir.argtypes = [ctypes.POINTER(struct_apps_std_DIR), ctypes.POINTER(struct_apps_std_DIRENT), ctypes.POINTER(ctypes.c_int32)]
except AttributeError:
    pass
try:
    apps_std_mkdir = _libraries['FIXME_STUB'].apps_std_mkdir
    apps_std_mkdir.restype = ctypes.c_int32
    apps_std_mkdir.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.c_int32]
except AttributeError:
    pass
try:
    apps_std_rmdir = _libraries['FIXME_STUB'].apps_std_rmdir
    apps_std_rmdir.restype = ctypes.c_int32
    apps_std_rmdir.argtypes = [ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    apps_std_stat = _libraries['FIXME_STUB'].apps_std_stat
    apps_std_stat.restype = ctypes.c_int32
    apps_std_stat.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(struct_apps_std_STAT)]
except AttributeError:
    pass
try:
    apps_std_ftrunc = _libraries['FIXME_STUB'].apps_std_ftrunc
    apps_std_ftrunc.restype = ctypes.c_int32
    apps_std_ftrunc.argtypes = [apps_std_FILE, ctypes.c_int32]
except AttributeError:
    pass
try:
    apps_std_frename = _libraries['FIXME_STUB'].apps_std_frename
    apps_std_frename.restype = ctypes.c_int32
    apps_std_frename.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
__all__ = \
    ['ADSPRPC_SHARED_H', 'ADSP_DOMAIN', 'ADSP_DOMAIN_ID',
    'APPS_STD_SEEK_CUR', 'APPS_STD_SEEK_END', 'APPS_STD_SEEK_SET',
    'ARCH_VER', 'ASYNC_FASTRPC_SUPPORT', 'CAMERA_SECURE_CP_USAGE',
    'CDSP_DOMAIN', 'CDSP_DOMAIN_ID', 'DEFAULT_DOMAIN_ID',
    'DEVICE_NAME', 'DISPLAY_SECURE_CP_USAGE', 'DOMAIN_ID_MASK',
    'DOMAIN_SUPPORT', 'DSPRPC_CONTROL_LATENCY',
    'DSPRPC_CONTROL_UNSIGNED_MODULE', 'DSPRPC_CONTROL_WAKELOCK',
    'DSPRPC_GET_DOMAIN', 'DSPRPC_GET_DSP_INFO',
    'FASTRPC_ASYNC_CALLBACK', 'FASTRPC_ASYNC_NO_SYNC',
    'FASTRPC_ASYNC_POLL', 'FASTRPC_ASYNC_TYPE_MAX',
    'FASTRPC_ATTR_COHERENT', 'FASTRPC_ATTR_COHERENT',
    'FASTRPC_ATTR_FORCE_NOFLUSH', 'FASTRPC_ATTR_FORCE_NOINVALIDATE',
    'FASTRPC_ATTR_KEEP_MAP', 'FASTRPC_ATTR_KEEP_MAP',
    'FASTRPC_ATTR_NOMAP', 'FASTRPC_ATTR_NOMAP',
    'FASTRPC_ATTR_NON_COHERENT', 'FASTRPC_ATTR_NON_COHERENT',
    'FASTRPC_ATTR_NOVA', 'FASTRPC_ATTR_TRY_MAP_STATIC',
    'FASTRPC_CONTROL_KALLOC', 'FASTRPC_CONTROL_LATENCY',
    'FASTRPC_CONTROL_PD_DUMP', 'FASTRPC_CONTROL_SMMU',
    'FASTRPC_DSP_SSR', 'FASTRPC_GLINK_GUID', 'FASTRPC_INIT_ATTACH',
    'FASTRPC_INIT_ATTACH_SENSORS', 'FASTRPC_INIT_CREATE',
    'FASTRPC_INIT_CREATE_STATIC', 'FASTRPC_MAP_FD',
    'FASTRPC_MAP_FD_DELAYED', 'FASTRPC_MAP_MAX',
    'FASTRPC_MAP_RESERVED', 'FASTRPC_MAP_STATIC',
    'FASTRPC_MAX_DSP_ATTRIBUTES', 'FASTRPC_MODE_PARALLEL',
    'FASTRPC_MODE_PROFILE', 'FASTRPC_MODE_SERIAL',
    'FASTRPC_MODE_SESSION', 'FASTRPC_REGISTER_STATUS_NOTIFICATIONS',
    'FASTRPC_RELATIVE_THREAD_PRIORITY',
    'FASTRPC_REMOTE_PROCESS_EXCEPTION', 'FASTRPC_REMOTE_PROCESS_KILL',
    'FASTRPC_REMOTE_PROCESS_TYPE', 'FASTRPC_SESSION_CLOSE',
    'FASTRPC_SMD_GUID', 'FASTRPC_THREAD_PARAMS',
    'FASTRPC_USER_PD_EXCEPTION', 'FASTRPC_USER_PD_EXIT',
    'FASTRPC_USER_PD_FORCE_KILL', 'FASTRPC_USER_PD_UP',
    'FASTRPC_WAKELOCK_CONTROL_SUPPORTED', 'FIXED_HIGH', 'FIXED_LOW',
    'FIXED_MIDDLE', 'HMX_SUPPORT_DEPTH', 'HMX_SUPPORT_SPATIAL',
    'HVX_SUPPORT_128B', 'HVX_SUPPORT_64B', 'INVALID_HEAP_ID',
    'ION_ADSP_HEAP_ID', 'ION_ADSP_HEAP_NAME', 'ION_AUDIO_HEAP_ID',
    'ION_AUDIO_HEAP_NAME', 'ION_CAMERA_HEAP_ID',
    'ION_CAMERA_HEAP_NAME', 'ION_CP_MFC_HEAP_ID', 'ION_CP_MM_HEAP_ID',
    'ION_CP_WB_HEAP_ID', 'ION_FLAG_ALLOW_NON_CONTIG',
    'ION_FLAG_CACHED', 'ION_FLAG_CACHED_NEEDS_SYNC',
    'ION_FLAG_CP_APP', 'ION_FLAG_CP_BITSTREAM', 'ION_FLAG_CP_CAMERA',
    'ION_FLAG_CP_HLOS', 'ION_FLAG_CP_HLOS_FREE',
    'ION_FLAG_CP_NON_PIXEL', 'ION_FLAG_CP_PIXEL',
    'ION_FLAG_CP_SEC_DISPLAY', 'ION_FLAG_CP_TOUCH',
    'ION_FLAG_FORCE_CONTIGUOUS', 'ION_FLAG_POOL_FORCE_ALLOC',
    'ION_FLAG_POOL_PREFETCH', 'ION_FLAG_SECURE',
    'ION_FORCE_CONTIGUOUS', 'ION_HEAP_CARVEOUT_MASK',
    'ION_HEAP_ID_RESERVED', 'ION_HEAP_SYSTEM_CONTIG_MASK',
    'ION_HEAP_SYSTEM_MASK', 'ION_HEAP_TYPE_CARVEOUT',
    'ION_HEAP_TYPE_CHUNK', 'ION_HEAP_TYPE_CUSTOM',
    'ION_HEAP_TYPE_DMA', 'ION_HEAP_TYPE_DMA_MASK',
    'ION_HEAP_TYPE_HYP_CMA', 'ION_HEAP_TYPE_IOMMU',
    'ION_HEAP_TYPE_MSM_START', 'ION_HEAP_TYPE_SECURE_DMA',
    'ION_HEAP_TYPE_SYSTEM', 'ION_HEAP_TYPE_SYSTEM_CONTIG',
    'ION_HEAP_TYPE_SYSTEM_SECURE', 'ION_IOC_MAGIC',
    'ION_IOC_MSM_MAGIC', 'ION_IOMMU_HEAP_ID', 'ION_IOMMU_HEAP_NAME',
    'ION_KMALLOC_HEAP_NAME', 'ION_MFC_HEAP_NAME',
    'ION_MM_FIRMWARE_HEAP_ID', 'ION_MM_FIRMWARE_HEAP_NAME',
    'ION_MM_HEAP_NAME', 'ION_NUM_HEAPS', 'ION_PIL1_HEAP_ID',
    'ION_PIL1_HEAP_NAME', 'ION_PIL2_HEAP_ID', 'ION_PIL2_HEAP_NAME',
    'ION_QSECOM_HEAP_ID', 'ION_QSECOM_HEAP_NAME', 'ION_SECURE',
    'ION_SECURE_DISPLAY_HEAP_ID', 'ION_SECURE_DISPLAY_HEAP_NAME',
    'ION_SECURE_HEAP_ID', 'ION_SECURE_HEAP_NAME', 'ION_SF_HEAP_ID',
    'ION_SF_HEAP_NAME', 'ION_SYSTEM_CONTIG_HEAP_ID',
    'ION_SYSTEM_HEAP_ID', 'ION_SYSTEM_HEAP_NAME',
    'ION_VMALLOC_HEAP_NAME', 'ION_WB_HEAP_NAME', 'MAX_USAGE',
    'MDSP_DOMAIN', 'MDSP_DOMAIN_ID', 'NOT_FIXED', 'NUM_DOMAINS',
    'NUM_SESSIONS', 'PROCESS_TYPE_SIGNED', 'PROCESS_TYPE_UNSIGNED',
    'REMOTE_DEFAULT_H', 'REMOTE_MAP_MAX_FLAG',
    'REMOTE_MAP_MEM_STATIC', 'REMOTE_MODE_PARALLEL',
    'REMOTE_MODE_SERIAL', 'RPC_ADAPTIVE_QOS', 'RPC_DISABLE_QOS',
    'RPC_PM_QOS', 'RPC_POLL_QOS', 'SDSP_DOMAIN', 'SDSP_DOMAIN_ID',
    'STATUS_NOTIFICATION_SUPPORT', 'UNKNOWN', 'UNSIGNED_PD_SUPPORT',
    'VIDEO_BITSTREAM', 'VIDEO_NONPIXEL', 'VIDEO_PIXEL', 'VTCM_COUNT',
    'VTCM_PAGE', '_32BIT_PLACEHOLDER_apps_std_SEEK', '_APPS_STD_H',
    '_IO', '_IOR', '_IOW', '_IOWR', '_UAPI_LINUX_ION_H',
    '_UAPI_MSM_ION_H', '__QAIC_HEADER_ATTRIBUTE',
    '__QAIC_HEADER_EXPORT', '__QAIC_IMPL_ATTRIBUTE',
    '__QAIC_IMPL_EXPORT', '__QAIC_REMOTE_ATTRIBUTE',
    '__QAIC_REMOTE_EXPORT', '__QAIC_STRING1_OBJECT_DEFINED__',
    '__STRING1_OBJECT__', '_cstring1_t', 'apps_std_DIR',
    'apps_std_DIRENT', 'apps_std_FILE', 'apps_std_SEEK',
    'apps_std_STAT', 'apps_std_clearerr', 'apps_std_closedir',
    'apps_std_fclose', 'apps_std_fdopen_decrypt', 'apps_std_feof',
    'apps_std_ferror', 'apps_std_fflush', 'apps_std_fgetpos',
    'apps_std_fgets', 'apps_std_fileExists', 'apps_std_flen',
    'apps_std_fopen', 'apps_std_fopen_with_env', 'apps_std_fread',
    'apps_std_fremove', 'apps_std_frename', 'apps_std_freopen',
    'apps_std_fseek', 'apps_std_fsetpos', 'apps_std_fsync',
    'apps_std_ftell', 'apps_std_ftrunc', 'apps_std_fwrite',
    'apps_std_get_search_paths_with_env', 'apps_std_getenv',
    'apps_std_mkdir', 'apps_std_opendir', 'apps_std_print_string',
    'apps_std_readdir', 'apps_std_rewind', 'apps_std_rmdir',
    'apps_std_setenv', 'apps_std_stat', 'apps_std_unsetenv',
    'cp_mem_usage', 'fastrpc_async_callback_t',
    'fastrpc_async_descriptor_t', 'fastrpc_async_get_status',
    'fastrpc_async_jobid', 'fastrpc_async_notify_type',
    'fastrpc_capability', 'fastrpc_map_flags', 'fastrpc_mmap',
    'fastrpc_munmap', 'fastrpc_notif_fn_t', 'fastrpc_process_type',
    'fastrpc_release_async_job', 'handle_control_req_id',
    'ion_fixed_position', 'ion_heap_ids', 'ion_heap_type',
    'ion_user_handle_t', 'msm_ion_heap_types', 'remote_arg',
    'remote_arg_t', 'remote_buf', 'remote_buf_attributes',
    'remote_dma_handle', 'remote_dsp_attributes', 'remote_handle',
    'remote_handle64', 'remote_handle64_close',
    'remote_handle64_control', 'remote_handle64_invoke',
    'remote_handle64_invoke_async', 'remote_handle64_open',
    'remote_handle_close', 'remote_handle_control',
    'remote_handle_invoke', 'remote_handle_invoke_async',
    'remote_handle_open', 'remote_mem_map', 'remote_mem_map_flags',
    'remote_mem_unmap', 'remote_mmap', 'remote_munmap',
    'remote_register_buf', 'remote_register_buf_attr',
    'remote_register_dma_handle', 'remote_register_dma_handle_attr',
    'remote_register_fd', 'remote_rpc_control_latency_t',
    'remote_rpc_control_latency_t__enumvalues',
    'remote_rpc_get_domain_t', 'remote_rpc_latency_flags',
    'remote_rpc_notif_register_t', 'remote_rpc_process_exception',
    'remote_rpc_status_flags', 'remote_rpc_status_flags_t',
    'remote_rpc_status_flags_t__enumvalues', 'remote_session_control',
    'remote_set_mode', 'session_control_req_id', 'size_t',
    'smq_invoke_buf_start', 'smq_phy_page_start',
    'struct__cstring1_s', 'struct_apps_std_DIR',
    'struct_apps_std_DIRENT', 'struct_apps_std_STAT',
    'struct_c__SA_remote_buf', 'struct_c__SA_remote_dma_handle',
    'struct_fastrpc_async_callback',
    'struct_fastrpc_async_descriptor', 'struct_fastrpc_ctrl_kalloc',
    'struct_fastrpc_ctrl_latency', 'struct_fastrpc_ctrl_smmu',
    'struct_fastrpc_ioctl_control', 'struct_fastrpc_ioctl_init',
    'struct_fastrpc_ioctl_init_attrs', 'struct_fastrpc_ioctl_invoke',
    'struct_fastrpc_ioctl_invoke_attrs',
    'struct_fastrpc_ioctl_invoke_crc',
    'struct_fastrpc_ioctl_invoke_fd', 'struct_fastrpc_ioctl_mmap',
    'struct_fastrpc_ioctl_mmap_64', 'struct_fastrpc_ioctl_munmap',
    'struct_fastrpc_ioctl_munmap_64',
    'struct_fastrpc_ioctl_munmap_fd', 'struct_fastrpc_ioctl_perf',
    'struct_ion_allocation_data', 'struct_ion_custom_data',
    'struct_ion_fd_data', 'struct_ion_flush_data',
    'struct_ion_handle_data', 'struct_ion_prefetch_data',
    'struct_ion_prefetch_regions', 'struct_remote_buf',
    'struct_remote_buf64', 'struct_remote_dma_handle',
    'struct_remote_dma_handle64', 'struct_remote_dsp_capability',
    'struct_remote_process_type', 'struct_remote_rpc_control_latency',
    'struct_remote_rpc_control_pd_dump',
    'struct_remote_rpc_control_unsigned_module',
    'struct_remote_rpc_control_wakelock',
    'struct_remote_rpc_get_domain',
    'struct_remote_rpc_notif_register',
    'struct_remote_rpc_process_clean_params',
    'struct_remote_rpc_relative_thread_priority',
    'struct_remote_rpc_session_close',
    'struct_remote_rpc_thread_params', 'struct_smq_invoke',
    'struct_smq_invoke_buf', 'struct_smq_invoke_rsp',
    'struct_smq_msg', 'struct_smq_null_invoke', 'struct_smq_phy_page',
    'uint32_t', 'uint64_t', 'union_c__UA_remote_arg',
    'union_fastrpc_async_descriptor_0',
    'union_fastrpc_ioctl_control_0', 'union_remote_arg',
    'union_remote_arg64']
