# mypy: ignore-errors
# -*- coding: utf-8 -*-
#
# TARGET arch is: ['-DRPC_MESSAGE_STRUCTURES', '-DRPC_STRUCTURES', '-include', '/tmp/open-gpu-kernel-modules-81fe4fb417c8ac3b9bdcc1d56827d116743892a5/src/common/sdk/nvidia/inc/nvtypes.h', '-I/tmp/open-gpu-kernel-modules-81fe4fb417c8ac3b9bdcc1d56827d116743892a5/src/nvidia/generated', '-I/tmp/open-gpu-kernel-modules-81fe4fb417c8ac3b9bdcc1d56827d116743892a5/src/common/inc', '-I/tmp/open-gpu-kernel-modules-81fe4fb417c8ac3b9bdcc1d56827d116743892a5/src/nvidia/inc', '-I/tmp/open-gpu-kernel-modules-81fe4fb417c8ac3b9bdcc1d56827d116743892a5/src/nvidia/interface/', '-I/tmp/open-gpu-kernel-modules-81fe4fb417c8ac3b9bdcc1d56827d116743892a5/src/nvidia/inc/kernel', '-I/tmp/open-gpu-kernel-modules-81fe4fb417c8ac3b9bdcc1d56827d116743892a5/src/nvidia/inc/libraries', '-I/tmp/open-gpu-kernel-modules-81fe4fb417c8ac3b9bdcc1d56827d116743892a5/src/nvidia/arch/nvalloc/common/inc', '-I/tmp/open-gpu-kernel-modules-81fe4fb417c8ac3b9bdcc1d56827d116743892a5/kernel-open/nvidia-uvm', '-I/tmp/open-gpu-kernel-modules-81fe4fb417c8ac3b9bdcc1d56827d116743892a5/kernel-open/common/inc', '-I/tmp/open-gpu-kernel-modules-81fe4fb417c8ac3b9bdcc1d56827d116743892a5/src/common/sdk/nvidia/inc', '-I/tmp/open-gpu-kernel-modules-81fe4fb417c8ac3b9bdcc1d56827d116743892a5/src/nvidia/arch/nvalloc/unix/include', '-I/tmp/open-gpu-kernel-modules-81fe4fb417c8ac3b9bdcc1d56827d116743892a5/src/common/sdk/nvidia/inc/ctrl']
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



c_int128 = ctypes.c_ubyte*16
c_uint128 = c_int128
void = None
if ctypes.sizeof(ctypes.c_longdouble) == 16:
    c_long_double_t = ctypes.c_longdouble
else:
    c_long_double_t = ctypes.c_ubyte*16

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





KERN_FSP_COT_PAYLOAD_H = True # macro
class struct_c__SA_MCTP_HEADER(Structure):
    pass

struct_c__SA_MCTP_HEADER._pack_ = 1 # source:False
struct_c__SA_MCTP_HEADER._fields_ = [
    ('constBlob', ctypes.c_uint32),
    ('msgType', ctypes.c_ubyte),
    ('vendorId', ctypes.c_uint16),
]

MCTP_HEADER = struct_c__SA_MCTP_HEADER
class struct_c__SA_NVDM_PAYLOAD_COT(Structure):
    pass

struct_c__SA_NVDM_PAYLOAD_COT._pack_ = 1 # source:False
struct_c__SA_NVDM_PAYLOAD_COT._fields_ = [
    ('version', ctypes.c_uint16),
    ('size', ctypes.c_uint16),
    ('gspFmcSysmemOffset', ctypes.c_uint64),
    ('frtsSysmemOffset', ctypes.c_uint64),
    ('frtsSysmemSize', ctypes.c_uint32),
    ('frtsVidmemOffset', ctypes.c_uint64),
    ('frtsVidmemSize', ctypes.c_uint32),
    ('hash384', ctypes.c_uint32 * 12),
    ('publicKey', ctypes.c_uint32 * 96),
    ('signature', ctypes.c_uint32 * 96),
    ('gspBootArgsSysmemOffset', ctypes.c_uint64),
]

NVDM_PAYLOAD_COT = struct_c__SA_NVDM_PAYLOAD_COT
GSPIFPUB_H = True # macro

# values for enumeration 'c__EA_GSP_DMA_TARGET'
c__EA_GSP_DMA_TARGET__enumvalues = {
    0: 'GSP_DMA_TARGET_LOCAL_FB',
    1: 'GSP_DMA_TARGET_COHERENT_SYSTEM',
    2: 'GSP_DMA_TARGET_NONCOHERENT_SYSTEM',
    3: 'GSP_DMA_TARGET_COUNT',
}
GSP_DMA_TARGET_LOCAL_FB = 0
GSP_DMA_TARGET_COHERENT_SYSTEM = 1
GSP_DMA_TARGET_NONCOHERENT_SYSTEM = 2
GSP_DMA_TARGET_COUNT = 3
c__EA_GSP_DMA_TARGET = ctypes.c_uint32 # enum
GSP_DMA_TARGET = c__EA_GSP_DMA_TARGET
GSP_DMA_TARGET__enumvalues = c__EA_GSP_DMA_TARGET__enumvalues
class struct_GSP_FMC_INIT_PARAMS(Structure):
    pass

struct_GSP_FMC_INIT_PARAMS._pack_ = 1 # source:False
struct_GSP_FMC_INIT_PARAMS._fields_ = [
    ('regkeys', ctypes.c_uint32),
]

GSP_FMC_INIT_PARAMS = struct_GSP_FMC_INIT_PARAMS
class struct_GSP_ACR_BOOT_GSP_RM_PARAMS(Structure):
    pass

struct_GSP_ACR_BOOT_GSP_RM_PARAMS._pack_ = 1 # source:False
struct_GSP_ACR_BOOT_GSP_RM_PARAMS._fields_ = [
    ('target', GSP_DMA_TARGET),
    ('gspRmDescSize', ctypes.c_uint32),
    ('gspRmDescOffset', ctypes.c_uint64),
    ('wprCarveoutOffset', ctypes.c_uint64),
    ('wprCarveoutSize', ctypes.c_uint32),
    ('bIsGspRmBoot', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 3),
]

GSP_ACR_BOOT_GSP_RM_PARAMS = struct_GSP_ACR_BOOT_GSP_RM_PARAMS
class struct_GSP_RM_PARAMS(Structure):
    pass

struct_GSP_RM_PARAMS._pack_ = 1 # source:False
struct_GSP_RM_PARAMS._fields_ = [
    ('target', GSP_DMA_TARGET),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('bootArgsOffset', ctypes.c_uint64),
]

GSP_RM_PARAMS = struct_GSP_RM_PARAMS
class struct_GSP_SPDM_PARAMS(Structure):
    pass

struct_GSP_SPDM_PARAMS._pack_ = 1 # source:False
struct_GSP_SPDM_PARAMS._fields_ = [
    ('target', GSP_DMA_TARGET),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('payloadBufferOffset', ctypes.c_uint64),
    ('payloadBufferSize', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

GSP_SPDM_PARAMS = struct_GSP_SPDM_PARAMS
class struct_GSP_FMC_BOOT_PARAMS(Structure):
    pass

struct_GSP_FMC_BOOT_PARAMS._pack_ = 1 # source:False
struct_GSP_FMC_BOOT_PARAMS._fields_ = [
    ('initParams', GSP_FMC_INIT_PARAMS),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('bootGspRmParams', GSP_ACR_BOOT_GSP_RM_PARAMS),
    ('gspRmParams', GSP_RM_PARAMS),
    ('gspSpdmParams', GSP_SPDM_PARAMS),
]

GSP_FMC_BOOT_PARAMS = struct_GSP_FMC_BOOT_PARAMS
GSP_FW_WPR_META_H_ = True # macro
GSP_FW_WPR_META_VERIFIED = 0xa0a0a0a0a0a0a0a0 # macro
GSP_FW_WPR_META_REVISION = 1 # macro
GSP_FW_WPR_META_MAGIC = 0xdc3aae21371a60b3 # macro
GSP_FW_WPR_HEAP_FREE_REGION_COUNT = 128 # macro
GSP_FW_HEAP_FREE_LIST_MAGIC = 0x4845415046524545 # macro
# GSP_FW_FLAGS = 8 : 0 # macro
# GSP_FW_FLAGS_CLOCK_BOOST = NVBIT ( 0 ) # macro
# GSP_FW_FLAGS_RECOVERY_MARGIN_PRESENT = NVBIT ( 1 ) # macro
# GSP_FW_FLAGS_PPCIE_ENABLED = NVBIT ( 2 ) # macro
class struct_c__SA_GspFwWprMeta(Structure):
    pass

class union_c__SA_GspFwWprMeta_0(Union):
    pass

class struct_c__SA_GspFwWprMeta_0_0(Structure):
    pass

struct_c__SA_GspFwWprMeta_0_0._pack_ = 1 # source:False
struct_c__SA_GspFwWprMeta_0_0._fields_ = [
    ('sysmemAddrOfSignature', ctypes.c_uint64),
    ('sizeOfSignature', ctypes.c_uint64),
]

class struct_c__SA_GspFwWprMeta_0_1(Structure):
    pass

struct_c__SA_GspFwWprMeta_0_1._pack_ = 1 # source:False
struct_c__SA_GspFwWprMeta_0_1._fields_ = [
    ('gspFwHeapFreeListWprOffset', ctypes.c_uint32),
    ('unused0', ctypes.c_uint32),
    ('unused1', ctypes.c_uint64),
]

union_c__SA_GspFwWprMeta_0._pack_ = 1 # source:False
union_c__SA_GspFwWprMeta_0._anonymous_ = ('_0', '_1',)
union_c__SA_GspFwWprMeta_0._fields_ = [
    ('_0', struct_c__SA_GspFwWprMeta_0_0),
    ('_1', struct_c__SA_GspFwWprMeta_0_1),
]

class union_c__SA_GspFwWprMeta_1(Union):
    pass

class struct_c__SA_GspFwWprMeta_1_0(Structure):
    pass

struct_c__SA_GspFwWprMeta_1_0._pack_ = 1 # source:False
struct_c__SA_GspFwWprMeta_1_0._fields_ = [
    ('partitionRpcAddr', ctypes.c_uint64),
    ('partitionRpcRequestOffset', ctypes.c_uint16),
    ('partitionRpcReplyOffset', ctypes.c_uint16),
    ('elfCodeOffset', ctypes.c_uint32),
    ('elfDataOffset', ctypes.c_uint32),
    ('elfCodeSize', ctypes.c_uint32),
    ('elfDataSize', ctypes.c_uint32),
    ('lsUcodeVersion', ctypes.c_uint32),
]

class struct_c__SA_GspFwWprMeta_1_1(Structure):
    pass

struct_c__SA_GspFwWprMeta_1_1._pack_ = 1 # source:False
struct_c__SA_GspFwWprMeta_1_1._fields_ = [
    ('partitionRpcPadding', ctypes.c_uint32 * 4),
    ('sysmemAddrOfCrashReportQueue', ctypes.c_uint64),
    ('sizeOfCrashReportQueue', ctypes.c_uint32),
    ('lsUcodeVersionPadding', ctypes.c_uint32 * 1),
]

union_c__SA_GspFwWprMeta_1._pack_ = 1 # source:False
union_c__SA_GspFwWprMeta_1._anonymous_ = ('_0', '_1',)
union_c__SA_GspFwWprMeta_1._fields_ = [
    ('_0', struct_c__SA_GspFwWprMeta_1_0),
    ('_1', struct_c__SA_GspFwWprMeta_1_1),
]

struct_c__SA_GspFwWprMeta._pack_ = 1 # source:False
struct_c__SA_GspFwWprMeta._anonymous_ = ('_0', '_1',)
struct_c__SA_GspFwWprMeta._fields_ = [
    ('magic', ctypes.c_uint64),
    ('revision', ctypes.c_uint64),
    ('sysmemAddrOfRadix3Elf', ctypes.c_uint64),
    ('sizeOfRadix3Elf', ctypes.c_uint64),
    ('sysmemAddrOfBootloader', ctypes.c_uint64),
    ('sizeOfBootloader', ctypes.c_uint64),
    ('bootloaderCodeOffset', ctypes.c_uint64),
    ('bootloaderDataOffset', ctypes.c_uint64),
    ('bootloaderManifestOffset', ctypes.c_uint64),
    ('_0', union_c__SA_GspFwWprMeta_0),
    ('gspFwRsvdStart', ctypes.c_uint64),
    ('nonWprHeapOffset', ctypes.c_uint64),
    ('nonWprHeapSize', ctypes.c_uint64),
    ('gspFwWprStart', ctypes.c_uint64),
    ('gspFwHeapOffset', ctypes.c_uint64),
    ('gspFwHeapSize', ctypes.c_uint64),
    ('gspFwOffset', ctypes.c_uint64),
    ('bootBinOffset', ctypes.c_uint64),
    ('frtsOffset', ctypes.c_uint64),
    ('frtsSize', ctypes.c_uint64),
    ('gspFwWprEnd', ctypes.c_uint64),
    ('fbSize', ctypes.c_uint64),
    ('vgaWorkspaceOffset', ctypes.c_uint64),
    ('vgaWorkspaceSize', ctypes.c_uint64),
    ('bootCount', ctypes.c_uint64),
    ('_1', union_c__SA_GspFwWprMeta_1),
    ('gspFwHeapVfPartitionCount', ctypes.c_ubyte),
    ('flags', ctypes.c_ubyte),
    ('padding', ctypes.c_ubyte * 2),
    ('pmuReservedSize', ctypes.c_uint32),
    ('verified', ctypes.c_uint64),
]

GspFwWprMeta = struct_c__SA_GspFwWprMeta
class struct_c__SA_GspFwHeapFreeRegion(Structure):
    pass

struct_c__SA_GspFwHeapFreeRegion._pack_ = 1 # source:False
struct_c__SA_GspFwHeapFreeRegion._fields_ = [
    ('offs', ctypes.c_uint32),
    ('length', ctypes.c_uint32),
]

GspFwHeapFreeRegion = struct_c__SA_GspFwHeapFreeRegion
class struct_c__SA_GspFwHeapFreeList(Structure):
    pass

struct_c__SA_GspFwHeapFreeList._pack_ = 1 # source:False
struct_c__SA_GspFwHeapFreeList._fields_ = [
    ('magic', ctypes.c_uint64),
    ('nregions', ctypes.c_uint32),
    ('regions', struct_c__SA_GspFwHeapFreeRegion * 128),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

GspFwHeapFreeList = struct_c__SA_GspFwHeapFreeList
GSP_FW_SR_META_H_ = True # macro
GSP_FW_SR_META_MAGIC = 0x8a3bb9e6c6c39d93 # macro
GSP_FW_SR_META_REVISION = 2 # macro
GSP_FW_SR_META_INTERNAL_SIZE = 128 # macro
class struct_c__SA_GspFwSRMeta(Structure):
    pass

struct_c__SA_GspFwSRMeta._pack_ = 1 # source:False
struct_c__SA_GspFwSRMeta._fields_ = [
    ('magic', ctypes.c_uint64),
    ('revision', ctypes.c_uint64),
    ('sysmemAddrOfSuspendResumeData', ctypes.c_uint64),
    ('sizeOfSuspendResumeData', ctypes.c_uint64),
    ('internal', ctypes.c_uint32 * 32),
    ('flags', ctypes.c_uint32),
    ('subrevision', ctypes.c_uint32),
    ('padding', ctypes.c_uint32 * 22),
]

GspFwSRMeta = struct_c__SA_GspFwSRMeta
GSP_INIT_ARGS_H = True # macro
class struct_c__SA_MESSAGE_QUEUE_INIT_ARGUMENTS(Structure):
    pass

struct_c__SA_MESSAGE_QUEUE_INIT_ARGUMENTS._pack_ = 1 # source:False
struct_c__SA_MESSAGE_QUEUE_INIT_ARGUMENTS._fields_ = [
    ('sharedMemPhysAddr', ctypes.c_uint64),
    ('pageTableEntryCount', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('cmdQueueOffset', ctypes.c_uint64),
    ('statQueueOffset', ctypes.c_uint64),
]

MESSAGE_QUEUE_INIT_ARGUMENTS = struct_c__SA_MESSAGE_QUEUE_INIT_ARGUMENTS
class struct_c__SA_GSP_SR_INIT_ARGUMENTS(Structure):
    pass

struct_c__SA_GSP_SR_INIT_ARGUMENTS._pack_ = 1 # source:False
struct_c__SA_GSP_SR_INIT_ARGUMENTS._fields_ = [
    ('oldLevel', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('bInPMTransition', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 3),
]

GSP_SR_INIT_ARGUMENTS = struct_c__SA_GSP_SR_INIT_ARGUMENTS
class struct_c__SA_GSP_ARGUMENTS_CACHED(Structure):
    pass

class struct_c__SA_GSP_ARGUMENTS_CACHED_profilerArgs(Structure):
    pass

struct_c__SA_GSP_ARGUMENTS_CACHED_profilerArgs._pack_ = 1 # source:False
struct_c__SA_GSP_ARGUMENTS_CACHED_profilerArgs._fields_ = [
    ('pa', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
]

struct_c__SA_GSP_ARGUMENTS_CACHED._pack_ = 1 # source:False
struct_c__SA_GSP_ARGUMENTS_CACHED._fields_ = [
    ('messageQueueInitArguments', MESSAGE_QUEUE_INIT_ARGUMENTS),
    ('srInitArguments', GSP_SR_INIT_ARGUMENTS),
    ('gpuInstance', ctypes.c_uint32),
    ('bDmemStack', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 7),
    ('profilerArgs', struct_c__SA_GSP_ARGUMENTS_CACHED_profilerArgs),
]

GSP_ARGUMENTS_CACHED = struct_c__SA_GSP_ARGUMENTS_CACHED
LIBOS_INIT_H_ = True # macro
LIBOS_MEMORY_REGION_INIT_ARGUMENTS_MAX = 4096 # macro
LIBOS_MEMORY_REGION_RADIX_PAGE_SIZE = 4096 # macro
LIBOS_MEMORY_REGION_RADIX_PAGE_LOG2 = 12 # macro
LibosAddress = ctypes.c_uint64

# values for enumeration 'c__EA_LibosMemoryRegionKind'
c__EA_LibosMemoryRegionKind__enumvalues = {
    0: 'LIBOS_MEMORY_REGION_NONE',
    1: 'LIBOS_MEMORY_REGION_CONTIGUOUS',
    2: 'LIBOS_MEMORY_REGION_RADIX3',
}
LIBOS_MEMORY_REGION_NONE = 0
LIBOS_MEMORY_REGION_CONTIGUOUS = 1
LIBOS_MEMORY_REGION_RADIX3 = 2
c__EA_LibosMemoryRegionKind = ctypes.c_uint32 # enum
LibosMemoryRegionKind = c__EA_LibosMemoryRegionKind
LibosMemoryRegionKind__enumvalues = c__EA_LibosMemoryRegionKind__enumvalues

# values for enumeration 'c__EA_LibosMemoryRegionLoc'
c__EA_LibosMemoryRegionLoc__enumvalues = {
    0: 'LIBOS_MEMORY_REGION_LOC_NONE',
    1: 'LIBOS_MEMORY_REGION_LOC_SYSMEM',
    2: 'LIBOS_MEMORY_REGION_LOC_FB',
}
LIBOS_MEMORY_REGION_LOC_NONE = 0
LIBOS_MEMORY_REGION_LOC_SYSMEM = 1
LIBOS_MEMORY_REGION_LOC_FB = 2
c__EA_LibosMemoryRegionLoc = ctypes.c_uint32 # enum
LibosMemoryRegionLoc = c__EA_LibosMemoryRegionLoc
LibosMemoryRegionLoc__enumvalues = c__EA_LibosMemoryRegionLoc__enumvalues
class struct_c__SA_LibosMemoryRegionInitArgument(Structure):
    pass

struct_c__SA_LibosMemoryRegionInitArgument._pack_ = 1 # source:False
struct_c__SA_LibosMemoryRegionInitArgument._fields_ = [
    ('id8', ctypes.c_uint64),
    ('pa', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
    ('kind', ctypes.c_ubyte),
    ('loc', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 6),
]

LibosMemoryRegionInitArgument = struct_c__SA_LibosMemoryRegionInitArgument
RM_RISCV_UCODE_H = True # macro
class struct_c__SA_RM_RISCV_UCODE_DESC(Structure):
    pass

struct_c__SA_RM_RISCV_UCODE_DESC._pack_ = 1 # source:False
struct_c__SA_RM_RISCV_UCODE_DESC._fields_ = [
    ('version', ctypes.c_uint32),
    ('bootloaderOffset', ctypes.c_uint32),
    ('bootloaderSize', ctypes.c_uint32),
    ('bootloaderParamOffset', ctypes.c_uint32),
    ('bootloaderParamSize', ctypes.c_uint32),
    ('riscvElfOffset', ctypes.c_uint32),
    ('riscvElfSize', ctypes.c_uint32),
    ('appVersion', ctypes.c_uint32),
    ('manifestOffset', ctypes.c_uint32),
    ('manifestSize', ctypes.c_uint32),
    ('monitorDataOffset', ctypes.c_uint32),
    ('monitorDataSize', ctypes.c_uint32),
    ('monitorCodeOffset', ctypes.c_uint32),
    ('monitorCodeSize', ctypes.c_uint32),
    ('bIsMonitorEnabled', ctypes.c_uint32),
    ('swbromCodeOffset', ctypes.c_uint32),
    ('swbromCodeSize', ctypes.c_uint32),
    ('swbromDataOffset', ctypes.c_uint32),
    ('swbromDataSize', ctypes.c_uint32),
    ('fbReservedSize', ctypes.c_uint32),
    ('bSignedAsCode', ctypes.c_uint32),
]

RM_RISCV_UCODE_DESC = struct_c__SA_RM_RISCV_UCODE_DESC
MSGQ_PRIV_H = True # macro
MSGQ_VERSION = 0 # macro
class struct_c__SA_msgqTxHeader(Structure):
    pass

struct_c__SA_msgqTxHeader._pack_ = 1 # source:False
struct_c__SA_msgqTxHeader._fields_ = [
    ('version', ctypes.c_uint32),
    ('size', ctypes.c_uint32),
    ('msgSize', ctypes.c_uint32),
    ('msgCount', ctypes.c_uint32),
    ('writePtr', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('rxHdrOff', ctypes.c_uint32),
    ('entryOff', ctypes.c_uint32),
]

msgqTxHeader = struct_c__SA_msgqTxHeader
class struct_c__SA_msgqRxHeader(Structure):
    pass

struct_c__SA_msgqRxHeader._pack_ = 1 # source:False
struct_c__SA_msgqRxHeader._fields_ = [
    ('readPtr', ctypes.c_uint32),
]

msgqRxHeader = struct_c__SA_msgqRxHeader
class struct_c__SA_msgqMetadata(Structure):
    pass

struct_c__SA_msgqMetadata._pack_ = 1 # source:False
struct_c__SA_msgqMetadata._fields_ = [
    ('pOurTxHdr', ctypes.POINTER(struct_c__SA_msgqTxHeader)),
    ('pTheirTxHdr', ctypes.POINTER(struct_c__SA_msgqTxHeader)),
    ('pOurRxHdr', ctypes.POINTER(struct_c__SA_msgqRxHeader)),
    ('pTheirRxHdr', ctypes.POINTER(struct_c__SA_msgqRxHeader)),
    ('pOurEntries', ctypes.POINTER(ctypes.c_ubyte)),
    ('pTheirEntries', ctypes.POINTER(ctypes.c_ubyte)),
    ('pReadIncoming', ctypes.POINTER(ctypes.c_uint32)),
    ('pWriteIncoming', ctypes.POINTER(ctypes.c_uint32)),
    ('pReadOutgoing', ctypes.POINTER(ctypes.c_uint32)),
    ('pWriteOutgoing', ctypes.POINTER(ctypes.c_uint32)),
    ('tx', msgqTxHeader),
    ('txReadPtr', ctypes.c_uint32),
    ('txFree', ctypes.c_uint32),
    ('txLinked', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 3),
    ('rx', msgqTxHeader),
    ('rxReadPtr', ctypes.c_uint32),
    ('rxAvail', ctypes.c_uint32),
    ('rxLinked', ctypes.c_ubyte),
    ('rxSwapped', ctypes.c_ubyte),
    ('PADDING_1', ctypes.c_ubyte * 2),
    ('fcnNotify', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(None))),
    ('fcnNotifyArg', ctypes.POINTER(None)),
    ('fcnBackendRw', ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.POINTER(None), ctypes.POINTER(None), ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(None))),
    ('fcnBackendRwArg', ctypes.POINTER(None)),
    ('fcnInvalidate', ctypes.CFUNCTYPE(None, ctypes.POINTER(None), ctypes.c_uint32)),
    ('fcnFlush', ctypes.CFUNCTYPE(None, ctypes.POINTER(None), ctypes.c_uint32)),
    ('fcnZero', ctypes.CFUNCTYPE(None, ctypes.POINTER(None), ctypes.c_uint32)),
    ('fcnBarrier', ctypes.CFUNCTYPE(None)),
]

msgqMetadata = struct_c__SA_msgqMetadata
__vgpu_rpc_nv_headers_h__ = True # macro
MAX_GPC_COUNT = 32 # macro
VGPU_MAX_REGOPS_PER_RPC = 100 # macro
VGPU_RESERVED_HANDLE_BASE = 0xCAF3F000 # macro
VGPU_RESERVED_HANDLE_RANGE = 0x1000 # macro
# def VGPU_CALC_PARAM_OFFSET(prev_offset, prev_params):  # macro
#    return (prev_offset+NV_ALIGN_UP(ctypes.sizeof(prev_params),ctypes.sizeof(NvU32)))
# NV_VGPU_MSG_HEADER_VERSION_MAJOR = 31 : 24 # macro
# NV_VGPU_MSG_HEADER_VERSION_MINOR = 23 : 16 # macro
NV_VGPU_MSG_HEADER_VERSION_MAJOR_TOT = 0x00000003 # macro
NV_VGPU_MSG_HEADER_VERSION_MINOR_TOT = 0x00000000 # macro
NV_VGPU_MSG_SIGNATURE_VALID = 0x43505256 # macro
_RPC_GLOBAL_ENUMS_H_ = True # macro
# def X(UNIT, RPC, VAL):  # macro
#    return NV_VGPU_MSG_FUNCTION_##RPC=VAL,
DEFINING_X_IN_RPC_GLOBAL_ENUMS_H = True # macro
# def E(RPC, VAL):  # macro
#    return NV_VGPU_MSG_EVENT_##RPC=VAL,
DEFINING_E_IN_RPC_GLOBAL_ENUMS_H = True # macro
# NV_VGPU_MSG_RESULT__RM = NV_ERR_GENERIC : 0x00000000 # macro
# NV_VGPU_MSG_RESULT_SUCCESS = NV_OK # macro
# NV_VGPU_MSG_RESULT_CARD_NOT_PRESENT = NV_ERR_CARD_NOT_PRESENT # macro
# NV_VGPU_MSG_RESULT_DUAL_LINK_INUSE = NV_ERR_DUAL_LINK_INUSE # macro
# NV_VGPU_MSG_RESULT_GENERIC = NV_ERR_GENERIC # macro
# NV_VGPU_MSG_RESULT_GPU_NOT_FULL_POWER = NV_ERR_GPU_NOT_FULL_POWER # macro
# NV_VGPU_MSG_RESULT_IN_USE = NV_ERR_IN_USE # macro
# NV_VGPU_MSG_RESULT_INSUFFICIENT_RESOURCES = NV_ERR_INSUFFICIENT_RESOURCES # macro
# NV_VGPU_MSG_RESULT_INVALID_ACCESS_TYPE = NV_ERR_INVALID_ACCESS_TYPE # macro
# NV_VGPU_MSG_RESULT_INVALID_ARGUMENT = NV_ERR_INVALID_ARGUMENT # macro
# NV_VGPU_MSG_RESULT_INVALID_BASE = NV_ERR_INVALID_BASE # macro
# NV_VGPU_MSG_RESULT_INVALID_CHANNEL = NV_ERR_INVALID_CHANNEL # macro
# NV_VGPU_MSG_RESULT_INVALID_CLASS = NV_ERR_INVALID_CLASS # macro
# NV_VGPU_MSG_RESULT_INVALID_CLIENT = NV_ERR_INVALID_CLIENT # macro
# NV_VGPU_MSG_RESULT_INVALID_COMMAND = NV_ERR_INVALID_COMMAND # macro
# NV_VGPU_MSG_RESULT_INVALID_DATA = NV_ERR_INVALID_DATA # macro
# NV_VGPU_MSG_RESULT_INVALID_DEVICE = NV_ERR_INVALID_DEVICE # macro
# NV_VGPU_MSG_RESULT_INVALID_DMA_SPECIFIER = NV_ERR_INVALID_DMA_SPECIFIER # macro
# NV_VGPU_MSG_RESULT_INVALID_EVENT = NV_ERR_INVALID_EVENT # macro
# NV_VGPU_MSG_RESULT_INVALID_FLAGS = NV_ERR_INVALID_FLAGS # macro
# NV_VGPU_MSG_RESULT_INVALID_FUNCTION = NV_ERR_INVALID_FUNCTION # macro
# NV_VGPU_MSG_RESULT_INVALID_HEAP = NV_ERR_INVALID_HEAP # macro
# NV_VGPU_MSG_RESULT_INVALID_INDEX = NV_ERR_INVALID_INDEX # macro
# NV_VGPU_MSG_RESULT_INVALID_LIMIT = NV_ERR_INVALID_LIMIT # macro
# NV_VGPU_MSG_RESULT_INVALID_METHOD = NV_ERR_INVALID_METHOD # macro
# NV_VGPU_MSG_RESULT_INVALID_OBJECT_BUFFER = NV_ERR_INVALID_OBJECT_BUFFER # macro
# NV_VGPU_MSG_RESULT_INVALID_OBJECT_ERROR = NV_ERR_INVALID_OBJECT # macro
# NV_VGPU_MSG_RESULT_INVALID_OBJECT_HANDLE = NV_ERR_INVALID_OBJECT_HANDLE # macro
# NV_VGPU_MSG_RESULT_INVALID_OBJECT_NEW = NV_ERR_INVALID_OBJECT_NEW # macro
# NV_VGPU_MSG_RESULT_INVALID_OBJECT_OLD = NV_ERR_INVALID_OBJECT_OLD # macro
# NV_VGPU_MSG_RESULT_INVALID_OBJECT_PARENT = NV_ERR_INVALID_OBJECT_PARENT # macro
# NV_VGPU_MSG_RESULT_INVALID_OFFSET = NV_ERR_INVALID_OFFSET # macro
# NV_VGPU_MSG_RESULT_INVALID_OWNER = NV_ERR_INVALID_OWNER # macro
# NV_VGPU_MSG_RESULT_INVALID_PARAM_STRUCT = NV_ERR_INVALID_PARAM_STRUCT # macro
# NV_VGPU_MSG_RESULT_INVALID_PARAMETER = NV_ERR_INVALID_PARAMETER # macro
# NV_VGPU_MSG_RESULT_INVALID_POINTER = NV_ERR_INVALID_POINTER # macro
# NV_VGPU_MSG_RESULT_INVALID_REGISTRY_KEY = NV_ERR_INVALID_REGISTRY_KEY # macro
# NV_VGPU_MSG_RESULT_INVALID_STATE = NV_ERR_INVALID_STATE # macro
# NV_VGPU_MSG_RESULT_INVALID_STRING_LENGTH = NV_ERR_INVALID_STRING_LENGTH # macro
# NV_VGPU_MSG_RESULT_INVALID_XLATE = NV_ERR_INVALID_XLATE # macro
# NV_VGPU_MSG_RESULT_IRQ_NOT_FIRING = NV_ERR_IRQ_NOT_FIRING # macro
# NV_VGPU_MSG_RESULT_MULTIPLE_MEMORY_TYPES = NV_ERR_MULTIPLE_MEMORY_TYPES # macro
# NV_VGPU_MSG_RESULT_NOT_SUPPORTED = NV_ERR_NOT_SUPPORTED # macro
# NV_VGPU_MSG_RESULT_OPERATING_SYSTEM = NV_ERR_OPERATING_SYSTEM # macro
# NV_VGPU_MSG_RESULT_PROTECTION_FAULT = NV_ERR_PROTECTION_FAULT # macro
# NV_VGPU_MSG_RESULT_TIMEOUT = NV_ERR_TIMEOUT # macro
# NV_VGPU_MSG_RESULT_TOO_MANY_PRIMARIES = NV_ERR_TOO_MANY_PRIMARIES # macro
# NV_VGPU_MSG_RESULT_IRQ_EDGE_TRIGGERED = NV_ERR_IRQ_EDGE_TRIGGERED # macro
# NV_VGPU_MSG_RESULT_GUEST_HOST_DRIVER_MISMATCH = NV_ERR_LIB_RM_VERSION_MISMATCH # macro
# NV_VGPU_MSG_RESULT__VMIOP = 0xFF00000a : 0xFF000000 # macro
NV_VGPU_MSG_RESULT_VMIOP_INVAL = 0xFF000001 # macro
NV_VGPU_MSG_RESULT_VMIOP_RESOURCE = 0xFF000002 # macro
NV_VGPU_MSG_RESULT_VMIOP_RANGE = 0xFF000003 # macro
NV_VGPU_MSG_RESULT_VMIOP_READ_ONLY = 0xFF000004 # macro
NV_VGPU_MSG_RESULT_VMIOP_NOT_FOUND = 0xFF000005 # macro
NV_VGPU_MSG_RESULT_VMIOP_NO_ADDRESS_SPACE = 0xFF000006 # macro
NV_VGPU_MSG_RESULT_VMIOP_TIMEOUT = 0xFF000007 # macro
NV_VGPU_MSG_RESULT_VMIOP_NOT_ALLOWED_IN_CALLBACK = 0xFF000008 # macro
NV_VGPU_MSG_RESULT_VMIOP_ECC_MISMATCH = 0xFF000009 # macro
NV_VGPU_MSG_RESULT_VMIOP_NOT_SUPPORTED = 0xFF00000a # macro
# NV_VGPU_MSG_RESULT__RPC = 0xFF100009 : 0xFF100000 # macro
NV_VGPU_MSG_RESULT_RPC_UNKNOWN_FUNCTION = 0xFF100001 # macro
NV_VGPU_MSG_RESULT_RPC_INVALID_MESSAGE_FORMAT = 0xFF100002 # macro
NV_VGPU_MSG_RESULT_RPC_HANDLE_NOT_FOUND = 0xFF100003 # macro
NV_VGPU_MSG_RESULT_RPC_HANDLE_EXISTS = 0xFF100004 # macro
NV_VGPU_MSG_RESULT_RPC_UNKNOWN_RM_ERROR = 0xFF100005 # macro
NV_VGPU_MSG_RESULT_RPC_UNKNOWN_VMIOP_ERROR = 0xFF100006 # macro
NV_VGPU_MSG_RESULT_RPC_RESERVED_HANDLE = 0xFF100007 # macro
NV_VGPU_MSG_RESULT_RPC_CUDA_PROFILING_DISABLED = 0xFF100008 # macro
NV_VGPU_MSG_RESULT_RPC_API_CONTROL_NOT_SUPPORTED = 0xFF100009 # macro
NV_VGPU_MSG_RESULT_RPC_PENDING = 0xFFFFFFFF # macro
NV_VGPU_MSG_UNION_INIT = 0x00000000 # macro
NV_VGPU_PTEDESC_INIT = 0x00000000 # macro
NV_VGPU_PTEDESC__PROD = 0x00000000 # macro
NV_VGPU_PTEDESC_IDR_NONE = 0x00000000 # macro
NV_VGPU_PTEDESC_IDR_SINGLE = 0x00000001 # macro
NV_VGPU_PTEDESC_IDR_DOUBLE = 0x00000002 # macro
NV_VGPU_PTEDESC_IDR_TRIPLE = 0x00000003 # macro
NV_VGPU_PTE_PAGE_SIZE = 0x1000 # macro
NV_VGPU_PTE_SIZE = 4 # macro
NV_VGPU_PTE_INDEX_SHIFT = 10 # macro
NV_VGPU_PTE_INDEX_MASK = 0x3FF # macro
NV_VGPU_PTE_64_PAGE_SIZE = 0x1000 # macro
NV_VGPU_PTE_64_SIZE = 8 # macro
NV_VGPU_PTE_64_INDEX_SHIFT = 9 # macro
NV_VGPU_PTE_64_INDEX_MASK = 0x1FF # macro
NV_VGPU_LOG_LEVEL_FATAL = 0x00000000 # macro
NV_VGPU_LOG_LEVEL_ERROR = 0x00000001 # macro
NV_VGPU_LOG_LEVEL_NOTICE = 0x00000002 # macro
NV_VGPU_LOG_LEVEL_STATUS = 0x00000003 # macro
NV_VGPU_LOG_LEVEL_DEBUG = 0x00000004 # macro
VGPU_RPC_GET_P2P_CAPS_V2_MAX_GPUS_SQUARED_PER_RPC = 512 # macro
GR_MAX_RPC_CTX_BUFFER_COUNT = 32 # macro
VGPU_RPC_CTRL_DEBUG_READ_ALL_SM_ERROR_STATES_PER_RPC_v21_06 = 80 # macro

# values for enumeration 'c__Ea_NV_VGPU_MSG_FUNCTION_NOP'
c__Ea_NV_VGPU_MSG_FUNCTION_NOP__enumvalues = {
    0: 'NV_VGPU_MSG_FUNCTION_NOP',
    1: 'NV_VGPU_MSG_FUNCTION_SET_GUEST_SYSTEM_INFO',
    2: 'NV_VGPU_MSG_FUNCTION_ALLOC_ROOT',
    3: 'NV_VGPU_MSG_FUNCTION_ALLOC_DEVICE',
    4: 'NV_VGPU_MSG_FUNCTION_ALLOC_MEMORY',
    5: 'NV_VGPU_MSG_FUNCTION_ALLOC_CTX_DMA',
    6: 'NV_VGPU_MSG_FUNCTION_ALLOC_CHANNEL_DMA',
    7: 'NV_VGPU_MSG_FUNCTION_MAP_MEMORY',
    8: 'NV_VGPU_MSG_FUNCTION_BIND_CTX_DMA',
    9: 'NV_VGPU_MSG_FUNCTION_ALLOC_OBJECT',
    10: 'NV_VGPU_MSG_FUNCTION_FREE',
    11: 'NV_VGPU_MSG_FUNCTION_LOG',
    12: 'NV_VGPU_MSG_FUNCTION_ALLOC_VIDMEM',
    13: 'NV_VGPU_MSG_FUNCTION_UNMAP_MEMORY',
    14: 'NV_VGPU_MSG_FUNCTION_MAP_MEMORY_DMA',
    15: 'NV_VGPU_MSG_FUNCTION_UNMAP_MEMORY_DMA',
    16: 'NV_VGPU_MSG_FUNCTION_GET_EDID',
    17: 'NV_VGPU_MSG_FUNCTION_ALLOC_DISP_CHANNEL',
    18: 'NV_VGPU_MSG_FUNCTION_ALLOC_DISP_OBJECT',
    19: 'NV_VGPU_MSG_FUNCTION_ALLOC_SUBDEVICE',
    20: 'NV_VGPU_MSG_FUNCTION_ALLOC_DYNAMIC_MEMORY',
    21: 'NV_VGPU_MSG_FUNCTION_DUP_OBJECT',
    22: 'NV_VGPU_MSG_FUNCTION_IDLE_CHANNELS',
    23: 'NV_VGPU_MSG_FUNCTION_ALLOC_EVENT',
    24: 'NV_VGPU_MSG_FUNCTION_SEND_EVENT',
    25: 'NV_VGPU_MSG_FUNCTION_REMAPPER_CONTROL',
    26: 'NV_VGPU_MSG_FUNCTION_DMA_CONTROL',
    27: 'NV_VGPU_MSG_FUNCTION_DMA_FILL_PTE_MEM',
    28: 'NV_VGPU_MSG_FUNCTION_MANAGE_HW_RESOURCE',
    29: 'NV_VGPU_MSG_FUNCTION_BIND_ARBITRARY_CTX_DMA',
    30: 'NV_VGPU_MSG_FUNCTION_CREATE_FB_SEGMENT',
    31: 'NV_VGPU_MSG_FUNCTION_DESTROY_FB_SEGMENT',
    32: 'NV_VGPU_MSG_FUNCTION_ALLOC_SHARE_DEVICE',
    33: 'NV_VGPU_MSG_FUNCTION_DEFERRED_API_CONTROL',
    34: 'NV_VGPU_MSG_FUNCTION_REMOVE_DEFERRED_API',
    35: 'NV_VGPU_MSG_FUNCTION_SIM_ESCAPE_READ',
    36: 'NV_VGPU_MSG_FUNCTION_SIM_ESCAPE_WRITE',
    37: 'NV_VGPU_MSG_FUNCTION_SIM_MANAGE_DISPLAY_CONTEXT_DMA',
    38: 'NV_VGPU_MSG_FUNCTION_FREE_VIDMEM_VIRT',
    39: 'NV_VGPU_MSG_FUNCTION_PERF_GET_PSTATE_INFO',
    40: 'NV_VGPU_MSG_FUNCTION_PERF_GET_PERFMON_SAMPLE',
    41: 'NV_VGPU_MSG_FUNCTION_PERF_GET_VIRTUAL_PSTATE_INFO',
    42: 'NV_VGPU_MSG_FUNCTION_PERF_GET_LEVEL_INFO',
    43: 'NV_VGPU_MSG_FUNCTION_MAP_SEMA_MEMORY',
    44: 'NV_VGPU_MSG_FUNCTION_UNMAP_SEMA_MEMORY',
    45: 'NV_VGPU_MSG_FUNCTION_SET_SURFACE_PROPERTIES',
    46: 'NV_VGPU_MSG_FUNCTION_CLEANUP_SURFACE',
    47: 'NV_VGPU_MSG_FUNCTION_UNLOADING_GUEST_DRIVER',
    48: 'NV_VGPU_MSG_FUNCTION_TDR_SET_TIMEOUT_STATE',
    49: 'NV_VGPU_MSG_FUNCTION_SWITCH_TO_VGA',
    50: 'NV_VGPU_MSG_FUNCTION_GPU_EXEC_REG_OPS',
    51: 'NV_VGPU_MSG_FUNCTION_GET_STATIC_INFO',
    52: 'NV_VGPU_MSG_FUNCTION_ALLOC_VIRTMEM',
    53: 'NV_VGPU_MSG_FUNCTION_UPDATE_PDE_2',
    54: 'NV_VGPU_MSG_FUNCTION_SET_PAGE_DIRECTORY',
    55: 'NV_VGPU_MSG_FUNCTION_GET_STATIC_PSTATE_INFO',
    56: 'NV_VGPU_MSG_FUNCTION_TRANSLATE_GUEST_GPU_PTES',
    57: 'NV_VGPU_MSG_FUNCTION_RESERVED_57',
    58: 'NV_VGPU_MSG_FUNCTION_RESET_CURRENT_GR_CONTEXT',
    59: 'NV_VGPU_MSG_FUNCTION_SET_SEMA_MEM_VALIDATION_STATE',
    60: 'NV_VGPU_MSG_FUNCTION_GET_ENGINE_UTILIZATION',
    61: 'NV_VGPU_MSG_FUNCTION_UPDATE_GPU_PDES',
    62: 'NV_VGPU_MSG_FUNCTION_GET_ENCODER_CAPACITY',
    63: 'NV_VGPU_MSG_FUNCTION_VGPU_PF_REG_READ32',
    64: 'NV_VGPU_MSG_FUNCTION_SET_GUEST_SYSTEM_INFO_EXT',
    65: 'NV_VGPU_MSG_FUNCTION_GET_GSP_STATIC_INFO',
    66: 'NV_VGPU_MSG_FUNCTION_RMFS_INIT',
    67: 'NV_VGPU_MSG_FUNCTION_RMFS_CLOSE_QUEUE',
    68: 'NV_VGPU_MSG_FUNCTION_RMFS_CLEANUP',
    69: 'NV_VGPU_MSG_FUNCTION_RMFS_TEST',
    70: 'NV_VGPU_MSG_FUNCTION_UPDATE_BAR_PDE',
    71: 'NV_VGPU_MSG_FUNCTION_CONTINUATION_RECORD',
    72: 'NV_VGPU_MSG_FUNCTION_GSP_SET_SYSTEM_INFO',
    73: 'NV_VGPU_MSG_FUNCTION_SET_REGISTRY',
    74: 'NV_VGPU_MSG_FUNCTION_GSP_INIT_POST_OBJGPU',
    75: 'NV_VGPU_MSG_FUNCTION_SUBDEV_EVENT_SET_NOTIFICATION',
    76: 'NV_VGPU_MSG_FUNCTION_GSP_RM_CONTROL',
    77: 'NV_VGPU_MSG_FUNCTION_GET_STATIC_INFO2',
    78: 'NV_VGPU_MSG_FUNCTION_DUMP_PROTOBUF_COMPONENT',
    79: 'NV_VGPU_MSG_FUNCTION_UNSET_PAGE_DIRECTORY',
    80: 'NV_VGPU_MSG_FUNCTION_GET_CONSOLIDATED_STATIC_INFO',
    81: 'NV_VGPU_MSG_FUNCTION_GMMU_REGISTER_FAULT_BUFFER',
    82: 'NV_VGPU_MSG_FUNCTION_GMMU_UNREGISTER_FAULT_BUFFER',
    83: 'NV_VGPU_MSG_FUNCTION_GMMU_REGISTER_CLIENT_SHADOW_FAULT_BUFFER',
    84: 'NV_VGPU_MSG_FUNCTION_GMMU_UNREGISTER_CLIENT_SHADOW_FAULT_BUFFER',
    85: 'NV_VGPU_MSG_FUNCTION_CTRL_SET_VGPU_FB_USAGE',
    86: 'NV_VGPU_MSG_FUNCTION_CTRL_NVFBC_SW_SESSION_UPDATE_INFO',
    87: 'NV_VGPU_MSG_FUNCTION_CTRL_NVENC_SW_SESSION_UPDATE_INFO',
    88: 'NV_VGPU_MSG_FUNCTION_CTRL_RESET_CHANNEL',
    89: 'NV_VGPU_MSG_FUNCTION_CTRL_RESET_ISOLATED_CHANNEL',
    90: 'NV_VGPU_MSG_FUNCTION_CTRL_GPU_HANDLE_VF_PRI_FAULT',
    91: 'NV_VGPU_MSG_FUNCTION_CTRL_CLK_GET_EXTENDED_INFO',
    92: 'NV_VGPU_MSG_FUNCTION_CTRL_PERF_BOOST',
    93: 'NV_VGPU_MSG_FUNCTION_CTRL_PERF_VPSTATES_GET_CONTROL',
    94: 'NV_VGPU_MSG_FUNCTION_CTRL_GET_ZBC_CLEAR_TABLE',
    95: 'NV_VGPU_MSG_FUNCTION_CTRL_SET_ZBC_COLOR_CLEAR',
    96: 'NV_VGPU_MSG_FUNCTION_CTRL_SET_ZBC_DEPTH_CLEAR',
    97: 'NV_VGPU_MSG_FUNCTION_CTRL_GPFIFO_SCHEDULE',
    98: 'NV_VGPU_MSG_FUNCTION_CTRL_SET_TIMESLICE',
    99: 'NV_VGPU_MSG_FUNCTION_CTRL_PREEMPT',
    100: 'NV_VGPU_MSG_FUNCTION_CTRL_FIFO_DISABLE_CHANNELS',
    101: 'NV_VGPU_MSG_FUNCTION_CTRL_SET_TSG_INTERLEAVE_LEVEL',
    102: 'NV_VGPU_MSG_FUNCTION_CTRL_SET_CHANNEL_INTERLEAVE_LEVEL',
    103: 'NV_VGPU_MSG_FUNCTION_GSP_RM_ALLOC',
    104: 'NV_VGPU_MSG_FUNCTION_CTRL_GET_P2P_CAPS_V2',
    105: 'NV_VGPU_MSG_FUNCTION_CTRL_CIPHER_AES_ENCRYPT',
    106: 'NV_VGPU_MSG_FUNCTION_CTRL_CIPHER_SESSION_KEY',
    107: 'NV_VGPU_MSG_FUNCTION_CTRL_CIPHER_SESSION_KEY_STATUS',
    108: 'NV_VGPU_MSG_FUNCTION_CTRL_DBG_CLEAR_ALL_SM_ERROR_STATES',
    109: 'NV_VGPU_MSG_FUNCTION_CTRL_DBG_READ_ALL_SM_ERROR_STATES',
    110: 'NV_VGPU_MSG_FUNCTION_CTRL_DBG_SET_EXCEPTION_MASK',
    111: 'NV_VGPU_MSG_FUNCTION_CTRL_GPU_PROMOTE_CTX',
    112: 'NV_VGPU_MSG_FUNCTION_CTRL_GR_CTXSW_PREEMPTION_BIND',
    113: 'NV_VGPU_MSG_FUNCTION_CTRL_GR_SET_CTXSW_PREEMPTION_MODE',
    114: 'NV_VGPU_MSG_FUNCTION_CTRL_GR_CTXSW_ZCULL_BIND',
    115: 'NV_VGPU_MSG_FUNCTION_CTRL_GPU_INITIALIZE_CTX',
    116: 'NV_VGPU_MSG_FUNCTION_CTRL_VASPACE_COPY_SERVER_RESERVED_PDES',
    117: 'NV_VGPU_MSG_FUNCTION_CTRL_FIFO_CLEAR_FAULTED_BIT',
    118: 'NV_VGPU_MSG_FUNCTION_CTRL_GET_LATEST_ECC_ADDRESSES',
    119: 'NV_VGPU_MSG_FUNCTION_CTRL_MC_SERVICE_INTERRUPTS',
    120: 'NV_VGPU_MSG_FUNCTION_CTRL_DMA_SET_DEFAULT_VASPACE',
    121: 'NV_VGPU_MSG_FUNCTION_CTRL_GET_CE_PCE_MASK',
    122: 'NV_VGPU_MSG_FUNCTION_CTRL_GET_ZBC_CLEAR_TABLE_ENTRY',
    123: 'NV_VGPU_MSG_FUNCTION_CTRL_GET_NVLINK_PEER_ID_MASK',
    124: 'NV_VGPU_MSG_FUNCTION_CTRL_GET_NVLINK_STATUS',
    125: 'NV_VGPU_MSG_FUNCTION_CTRL_GET_P2P_CAPS',
    126: 'NV_VGPU_MSG_FUNCTION_CTRL_GET_P2P_CAPS_MATRIX',
    127: 'NV_VGPU_MSG_FUNCTION_RESERVED_0',
    128: 'NV_VGPU_MSG_FUNCTION_CTRL_RESERVE_PM_AREA_SMPC',
    129: 'NV_VGPU_MSG_FUNCTION_CTRL_RESERVE_HWPM_LEGACY',
    130: 'NV_VGPU_MSG_FUNCTION_CTRL_B0CC_EXEC_REG_OPS',
    131: 'NV_VGPU_MSG_FUNCTION_CTRL_BIND_PM_RESOURCES',
    132: 'NV_VGPU_MSG_FUNCTION_CTRL_DBG_SUSPEND_CONTEXT',
    133: 'NV_VGPU_MSG_FUNCTION_CTRL_DBG_RESUME_CONTEXT',
    134: 'NV_VGPU_MSG_FUNCTION_CTRL_DBG_EXEC_REG_OPS',
    135: 'NV_VGPU_MSG_FUNCTION_CTRL_DBG_SET_MODE_MMU_DEBUG',
    136: 'NV_VGPU_MSG_FUNCTION_CTRL_DBG_READ_SINGLE_SM_ERROR_STATE',
    137: 'NV_VGPU_MSG_FUNCTION_CTRL_DBG_CLEAR_SINGLE_SM_ERROR_STATE',
    138: 'NV_VGPU_MSG_FUNCTION_CTRL_DBG_SET_MODE_ERRBAR_DEBUG',
    139: 'NV_VGPU_MSG_FUNCTION_CTRL_DBG_SET_NEXT_STOP_TRIGGER_TYPE',
    140: 'NV_VGPU_MSG_FUNCTION_CTRL_ALLOC_PMA_STREAM',
    141: 'NV_VGPU_MSG_FUNCTION_CTRL_PMA_STREAM_UPDATE_GET_PUT',
    142: 'NV_VGPU_MSG_FUNCTION_CTRL_FB_GET_INFO_V2',
    143: 'NV_VGPU_MSG_FUNCTION_CTRL_FIFO_SET_CHANNEL_PROPERTIES',
    144: 'NV_VGPU_MSG_FUNCTION_CTRL_GR_GET_CTX_BUFFER_INFO',
    145: 'NV_VGPU_MSG_FUNCTION_CTRL_KGR_GET_CTX_BUFFER_PTES',
    146: 'NV_VGPU_MSG_FUNCTION_CTRL_GPU_EVICT_CTX',
    147: 'NV_VGPU_MSG_FUNCTION_CTRL_FB_GET_FS_INFO',
    148: 'NV_VGPU_MSG_FUNCTION_CTRL_GRMGR_GET_GR_FS_INFO',
    149: 'NV_VGPU_MSG_FUNCTION_CTRL_STOP_CHANNEL',
    150: 'NV_VGPU_MSG_FUNCTION_CTRL_GR_PC_SAMPLING_MODE',
    151: 'NV_VGPU_MSG_FUNCTION_CTRL_PERF_RATED_TDP_GET_STATUS',
    152: 'NV_VGPU_MSG_FUNCTION_CTRL_PERF_RATED_TDP_SET_CONTROL',
    153: 'NV_VGPU_MSG_FUNCTION_CTRL_FREE_PMA_STREAM',
    154: 'NV_VGPU_MSG_FUNCTION_CTRL_TIMER_SET_GR_TICK_FREQ',
    155: 'NV_VGPU_MSG_FUNCTION_CTRL_FIFO_SETUP_VF_ZOMBIE_SUBCTX_PDB',
    156: 'NV_VGPU_MSG_FUNCTION_GET_CONSOLIDATED_GR_STATIC_INFO',
    157: 'NV_VGPU_MSG_FUNCTION_CTRL_DBG_SET_SINGLE_SM_SINGLE_STEP',
    158: 'NV_VGPU_MSG_FUNCTION_CTRL_GR_GET_TPC_PARTITION_MODE',
    159: 'NV_VGPU_MSG_FUNCTION_CTRL_GR_SET_TPC_PARTITION_MODE',
    160: 'NV_VGPU_MSG_FUNCTION_UVM_PAGING_CHANNEL_ALLOCATE',
    161: 'NV_VGPU_MSG_FUNCTION_UVM_PAGING_CHANNEL_DESTROY',
    162: 'NV_VGPU_MSG_FUNCTION_UVM_PAGING_CHANNEL_MAP',
    163: 'NV_VGPU_MSG_FUNCTION_UVM_PAGING_CHANNEL_UNMAP',
    164: 'NV_VGPU_MSG_FUNCTION_UVM_PAGING_CHANNEL_PUSH_STREAM',
    165: 'NV_VGPU_MSG_FUNCTION_UVM_PAGING_CHANNEL_SET_HANDLES',
    166: 'NV_VGPU_MSG_FUNCTION_UVM_METHOD_STREAM_GUEST_PAGES_OPERATION',
    167: 'NV_VGPU_MSG_FUNCTION_CTRL_INTERNAL_QUIESCE_PMA_CHANNEL',
    168: 'NV_VGPU_MSG_FUNCTION_DCE_RM_INIT',
    169: 'NV_VGPU_MSG_FUNCTION_REGISTER_VIRTUAL_EVENT_BUFFER',
    170: 'NV_VGPU_MSG_FUNCTION_CTRL_EVENT_BUFFER_UPDATE_GET',
    171: 'NV_VGPU_MSG_FUNCTION_GET_PLCABLE_ADDRESS_KIND',
    172: 'NV_VGPU_MSG_FUNCTION_CTRL_PERF_LIMITS_SET_STATUS_V2',
    173: 'NV_VGPU_MSG_FUNCTION_CTRL_INTERNAL_SRIOV_PROMOTE_PMA_STREAM',
    174: 'NV_VGPU_MSG_FUNCTION_CTRL_GET_MMU_DEBUG_MODE',
    175: 'NV_VGPU_MSG_FUNCTION_CTRL_INTERNAL_PROMOTE_FAULT_METHOD_BUFFERS',
    176: 'NV_VGPU_MSG_FUNCTION_CTRL_FLCN_GET_CTX_BUFFER_SIZE',
    177: 'NV_VGPU_MSG_FUNCTION_CTRL_FLCN_GET_CTX_BUFFER_INFO',
    178: 'NV_VGPU_MSG_FUNCTION_DISABLE_CHANNELS',
    179: 'NV_VGPU_MSG_FUNCTION_CTRL_FABRIC_MEMORY_DESCRIBE',
    180: 'NV_VGPU_MSG_FUNCTION_CTRL_FABRIC_MEM_STATS',
    181: 'NV_VGPU_MSG_FUNCTION_SAVE_HIBERNATION_DATA',
    182: 'NV_VGPU_MSG_FUNCTION_RESTORE_HIBERNATION_DATA',
    183: 'NV_VGPU_MSG_FUNCTION_CTRL_INTERNAL_MEMSYS_SET_ZBC_REFERENCED',
    184: 'NV_VGPU_MSG_FUNCTION_CTRL_EXEC_PARTITIONS_CREATE',
    185: 'NV_VGPU_MSG_FUNCTION_CTRL_EXEC_PARTITIONS_DELETE',
    186: 'NV_VGPU_MSG_FUNCTION_CTRL_GPFIFO_GET_WORK_SUBMIT_TOKEN',
    187: 'NV_VGPU_MSG_FUNCTION_CTRL_GPFIFO_SET_WORK_SUBMIT_TOKEN_NOTIF_INDEX',
    188: 'NV_VGPU_MSG_FUNCTION_PMA_SCRUBBER_SHARED_BUFFER_GUEST_PAGES_OPERATION',
    189: 'NV_VGPU_MSG_FUNCTION_CTRL_MASTER_GET_VIRTUAL_FUNCTION_ERROR_CONT_INTR_MASK',
    190: 'NV_VGPU_MSG_FUNCTION_SET_SYSMEM_DIRTY_PAGE_TRACKING_BUFFER',
    191: 'NV_VGPU_MSG_FUNCTION_CTRL_SUBDEVICE_GET_P2P_CAPS',
    192: 'NV_VGPU_MSG_FUNCTION_CTRL_BUS_SET_P2P_MAPPING',
    193: 'NV_VGPU_MSG_FUNCTION_CTRL_BUS_UNSET_P2P_MAPPING',
    194: 'NV_VGPU_MSG_FUNCTION_CTRL_FLA_SETUP_INSTANCE_MEM_BLOCK',
    195: 'NV_VGPU_MSG_FUNCTION_CTRL_GPU_MIGRATABLE_OPS',
    196: 'NV_VGPU_MSG_FUNCTION_CTRL_GET_TOTAL_HS_CREDITS',
    197: 'NV_VGPU_MSG_FUNCTION_CTRL_GET_HS_CREDITS',
    198: 'NV_VGPU_MSG_FUNCTION_CTRL_SET_HS_CREDITS',
    199: 'NV_VGPU_MSG_FUNCTION_CTRL_PM_AREA_PC_SAMPLER',
    200: 'NV_VGPU_MSG_FUNCTION_INVALIDATE_TLB',
    201: 'NV_VGPU_MSG_FUNCTION_CTRL_GPU_QUERY_ECC_STATUS',
    202: 'NV_VGPU_MSG_FUNCTION_ECC_NOTIFIER_WRITE_ACK',
    203: 'NV_VGPU_MSG_FUNCTION_CTRL_DBG_GET_MODE_MMU_DEBUG',
    204: 'NV_VGPU_MSG_FUNCTION_RM_API_CONTROL',
    205: 'NV_VGPU_MSG_FUNCTION_CTRL_CMD_INTERNAL_GPU_START_FABRIC_PROBE',
    206: 'NV_VGPU_MSG_FUNCTION_CTRL_NVLINK_GET_INBAND_RECEIVED_DATA',
    207: 'NV_VGPU_MSG_FUNCTION_GET_STATIC_DATA',
    208: 'NV_VGPU_MSG_FUNCTION_RESERVED_208',
    209: 'NV_VGPU_MSG_FUNCTION_CTRL_GPU_GET_INFO_V2',
    210: 'NV_VGPU_MSG_FUNCTION_GET_BRAND_CAPS',
    211: 'NV_VGPU_MSG_FUNCTION_CTRL_CMD_NVLINK_INBAND_SEND_DATA',
    212: 'NV_VGPU_MSG_FUNCTION_UPDATE_GPM_GUEST_BUFFER_INFO',
    213: 'NV_VGPU_MSG_FUNCTION_CTRL_CMD_INTERNAL_CONTROL_GSP_TRACE',
    214: 'NV_VGPU_MSG_FUNCTION_CTRL_SET_ZBC_STENCIL_CLEAR',
    215: 'NV_VGPU_MSG_FUNCTION_CTRL_SUBDEVICE_GET_VGPU_HEAP_STATS',
    216: 'NV_VGPU_MSG_FUNCTION_CTRL_SUBDEVICE_GET_LIBOS_HEAP_STATS',
    217: 'NV_VGPU_MSG_FUNCTION_CTRL_DBG_SET_MODE_MMU_GCC_DEBUG',
    218: 'NV_VGPU_MSG_FUNCTION_CTRL_DBG_GET_MODE_MMU_GCC_DEBUG',
    219: 'NV_VGPU_MSG_FUNCTION_CTRL_RESERVE_HES',
    220: 'NV_VGPU_MSG_FUNCTION_CTRL_RELEASE_HES',
    221: 'NV_VGPU_MSG_FUNCTION_CTRL_RESERVE_CCU_PROF',
    222: 'NV_VGPU_MSG_FUNCTION_CTRL_RELEASE_CCU_PROF',
    223: 'NV_VGPU_MSG_FUNCTION_NUM_FUNCTIONS',
}
NV_VGPU_MSG_FUNCTION_NOP = 0
NV_VGPU_MSG_FUNCTION_SET_GUEST_SYSTEM_INFO = 1
NV_VGPU_MSG_FUNCTION_ALLOC_ROOT = 2
NV_VGPU_MSG_FUNCTION_ALLOC_DEVICE = 3
NV_VGPU_MSG_FUNCTION_ALLOC_MEMORY = 4
NV_VGPU_MSG_FUNCTION_ALLOC_CTX_DMA = 5
NV_VGPU_MSG_FUNCTION_ALLOC_CHANNEL_DMA = 6
NV_VGPU_MSG_FUNCTION_MAP_MEMORY = 7
NV_VGPU_MSG_FUNCTION_BIND_CTX_DMA = 8
NV_VGPU_MSG_FUNCTION_ALLOC_OBJECT = 9
NV_VGPU_MSG_FUNCTION_FREE = 10
NV_VGPU_MSG_FUNCTION_LOG = 11
NV_VGPU_MSG_FUNCTION_ALLOC_VIDMEM = 12
NV_VGPU_MSG_FUNCTION_UNMAP_MEMORY = 13
NV_VGPU_MSG_FUNCTION_MAP_MEMORY_DMA = 14
NV_VGPU_MSG_FUNCTION_UNMAP_MEMORY_DMA = 15
NV_VGPU_MSG_FUNCTION_GET_EDID = 16
NV_VGPU_MSG_FUNCTION_ALLOC_DISP_CHANNEL = 17
NV_VGPU_MSG_FUNCTION_ALLOC_DISP_OBJECT = 18
NV_VGPU_MSG_FUNCTION_ALLOC_SUBDEVICE = 19
NV_VGPU_MSG_FUNCTION_ALLOC_DYNAMIC_MEMORY = 20
NV_VGPU_MSG_FUNCTION_DUP_OBJECT = 21
NV_VGPU_MSG_FUNCTION_IDLE_CHANNELS = 22
NV_VGPU_MSG_FUNCTION_ALLOC_EVENT = 23
NV_VGPU_MSG_FUNCTION_SEND_EVENT = 24
NV_VGPU_MSG_FUNCTION_REMAPPER_CONTROL = 25
NV_VGPU_MSG_FUNCTION_DMA_CONTROL = 26
NV_VGPU_MSG_FUNCTION_DMA_FILL_PTE_MEM = 27
NV_VGPU_MSG_FUNCTION_MANAGE_HW_RESOURCE = 28
NV_VGPU_MSG_FUNCTION_BIND_ARBITRARY_CTX_DMA = 29
NV_VGPU_MSG_FUNCTION_CREATE_FB_SEGMENT = 30
NV_VGPU_MSG_FUNCTION_DESTROY_FB_SEGMENT = 31
NV_VGPU_MSG_FUNCTION_ALLOC_SHARE_DEVICE = 32
NV_VGPU_MSG_FUNCTION_DEFERRED_API_CONTROL = 33
NV_VGPU_MSG_FUNCTION_REMOVE_DEFERRED_API = 34
NV_VGPU_MSG_FUNCTION_SIM_ESCAPE_READ = 35
NV_VGPU_MSG_FUNCTION_SIM_ESCAPE_WRITE = 36
NV_VGPU_MSG_FUNCTION_SIM_MANAGE_DISPLAY_CONTEXT_DMA = 37
NV_VGPU_MSG_FUNCTION_FREE_VIDMEM_VIRT = 38
NV_VGPU_MSG_FUNCTION_PERF_GET_PSTATE_INFO = 39
NV_VGPU_MSG_FUNCTION_PERF_GET_PERFMON_SAMPLE = 40
NV_VGPU_MSG_FUNCTION_PERF_GET_VIRTUAL_PSTATE_INFO = 41
NV_VGPU_MSG_FUNCTION_PERF_GET_LEVEL_INFO = 42
NV_VGPU_MSG_FUNCTION_MAP_SEMA_MEMORY = 43
NV_VGPU_MSG_FUNCTION_UNMAP_SEMA_MEMORY = 44
NV_VGPU_MSG_FUNCTION_SET_SURFACE_PROPERTIES = 45
NV_VGPU_MSG_FUNCTION_CLEANUP_SURFACE = 46
NV_VGPU_MSG_FUNCTION_UNLOADING_GUEST_DRIVER = 47
NV_VGPU_MSG_FUNCTION_TDR_SET_TIMEOUT_STATE = 48
NV_VGPU_MSG_FUNCTION_SWITCH_TO_VGA = 49
NV_VGPU_MSG_FUNCTION_GPU_EXEC_REG_OPS = 50
NV_VGPU_MSG_FUNCTION_GET_STATIC_INFO = 51
NV_VGPU_MSG_FUNCTION_ALLOC_VIRTMEM = 52
NV_VGPU_MSG_FUNCTION_UPDATE_PDE_2 = 53
NV_VGPU_MSG_FUNCTION_SET_PAGE_DIRECTORY = 54
NV_VGPU_MSG_FUNCTION_GET_STATIC_PSTATE_INFO = 55
NV_VGPU_MSG_FUNCTION_TRANSLATE_GUEST_GPU_PTES = 56
NV_VGPU_MSG_FUNCTION_RESERVED_57 = 57
NV_VGPU_MSG_FUNCTION_RESET_CURRENT_GR_CONTEXT = 58
NV_VGPU_MSG_FUNCTION_SET_SEMA_MEM_VALIDATION_STATE = 59
NV_VGPU_MSG_FUNCTION_GET_ENGINE_UTILIZATION = 60
NV_VGPU_MSG_FUNCTION_UPDATE_GPU_PDES = 61
NV_VGPU_MSG_FUNCTION_GET_ENCODER_CAPACITY = 62
NV_VGPU_MSG_FUNCTION_VGPU_PF_REG_READ32 = 63
NV_VGPU_MSG_FUNCTION_SET_GUEST_SYSTEM_INFO_EXT = 64
NV_VGPU_MSG_FUNCTION_GET_GSP_STATIC_INFO = 65
NV_VGPU_MSG_FUNCTION_RMFS_INIT = 66
NV_VGPU_MSG_FUNCTION_RMFS_CLOSE_QUEUE = 67
NV_VGPU_MSG_FUNCTION_RMFS_CLEANUP = 68
NV_VGPU_MSG_FUNCTION_RMFS_TEST = 69
NV_VGPU_MSG_FUNCTION_UPDATE_BAR_PDE = 70
NV_VGPU_MSG_FUNCTION_CONTINUATION_RECORD = 71
NV_VGPU_MSG_FUNCTION_GSP_SET_SYSTEM_INFO = 72
NV_VGPU_MSG_FUNCTION_SET_REGISTRY = 73
NV_VGPU_MSG_FUNCTION_GSP_INIT_POST_OBJGPU = 74
NV_VGPU_MSG_FUNCTION_SUBDEV_EVENT_SET_NOTIFICATION = 75
NV_VGPU_MSG_FUNCTION_GSP_RM_CONTROL = 76
NV_VGPU_MSG_FUNCTION_GET_STATIC_INFO2 = 77
NV_VGPU_MSG_FUNCTION_DUMP_PROTOBUF_COMPONENT = 78
NV_VGPU_MSG_FUNCTION_UNSET_PAGE_DIRECTORY = 79
NV_VGPU_MSG_FUNCTION_GET_CONSOLIDATED_STATIC_INFO = 80
NV_VGPU_MSG_FUNCTION_GMMU_REGISTER_FAULT_BUFFER = 81
NV_VGPU_MSG_FUNCTION_GMMU_UNREGISTER_FAULT_BUFFER = 82
NV_VGPU_MSG_FUNCTION_GMMU_REGISTER_CLIENT_SHADOW_FAULT_BUFFER = 83
NV_VGPU_MSG_FUNCTION_GMMU_UNREGISTER_CLIENT_SHADOW_FAULT_BUFFER = 84
NV_VGPU_MSG_FUNCTION_CTRL_SET_VGPU_FB_USAGE = 85
NV_VGPU_MSG_FUNCTION_CTRL_NVFBC_SW_SESSION_UPDATE_INFO = 86
NV_VGPU_MSG_FUNCTION_CTRL_NVENC_SW_SESSION_UPDATE_INFO = 87
NV_VGPU_MSG_FUNCTION_CTRL_RESET_CHANNEL = 88
NV_VGPU_MSG_FUNCTION_CTRL_RESET_ISOLATED_CHANNEL = 89
NV_VGPU_MSG_FUNCTION_CTRL_GPU_HANDLE_VF_PRI_FAULT = 90
NV_VGPU_MSG_FUNCTION_CTRL_CLK_GET_EXTENDED_INFO = 91
NV_VGPU_MSG_FUNCTION_CTRL_PERF_BOOST = 92
NV_VGPU_MSG_FUNCTION_CTRL_PERF_VPSTATES_GET_CONTROL = 93
NV_VGPU_MSG_FUNCTION_CTRL_GET_ZBC_CLEAR_TABLE = 94
NV_VGPU_MSG_FUNCTION_CTRL_SET_ZBC_COLOR_CLEAR = 95
NV_VGPU_MSG_FUNCTION_CTRL_SET_ZBC_DEPTH_CLEAR = 96
NV_VGPU_MSG_FUNCTION_CTRL_GPFIFO_SCHEDULE = 97
NV_VGPU_MSG_FUNCTION_CTRL_SET_TIMESLICE = 98
NV_VGPU_MSG_FUNCTION_CTRL_PREEMPT = 99
NV_VGPU_MSG_FUNCTION_CTRL_FIFO_DISABLE_CHANNELS = 100
NV_VGPU_MSG_FUNCTION_CTRL_SET_TSG_INTERLEAVE_LEVEL = 101
NV_VGPU_MSG_FUNCTION_CTRL_SET_CHANNEL_INTERLEAVE_LEVEL = 102
NV_VGPU_MSG_FUNCTION_GSP_RM_ALLOC = 103
NV_VGPU_MSG_FUNCTION_CTRL_GET_P2P_CAPS_V2 = 104
NV_VGPU_MSG_FUNCTION_CTRL_CIPHER_AES_ENCRYPT = 105
NV_VGPU_MSG_FUNCTION_CTRL_CIPHER_SESSION_KEY = 106
NV_VGPU_MSG_FUNCTION_CTRL_CIPHER_SESSION_KEY_STATUS = 107
NV_VGPU_MSG_FUNCTION_CTRL_DBG_CLEAR_ALL_SM_ERROR_STATES = 108
NV_VGPU_MSG_FUNCTION_CTRL_DBG_READ_ALL_SM_ERROR_STATES = 109
NV_VGPU_MSG_FUNCTION_CTRL_DBG_SET_EXCEPTION_MASK = 110
NV_VGPU_MSG_FUNCTION_CTRL_GPU_PROMOTE_CTX = 111
NV_VGPU_MSG_FUNCTION_CTRL_GR_CTXSW_PREEMPTION_BIND = 112
NV_VGPU_MSG_FUNCTION_CTRL_GR_SET_CTXSW_PREEMPTION_MODE = 113
NV_VGPU_MSG_FUNCTION_CTRL_GR_CTXSW_ZCULL_BIND = 114
NV_VGPU_MSG_FUNCTION_CTRL_GPU_INITIALIZE_CTX = 115
NV_VGPU_MSG_FUNCTION_CTRL_VASPACE_COPY_SERVER_RESERVED_PDES = 116
NV_VGPU_MSG_FUNCTION_CTRL_FIFO_CLEAR_FAULTED_BIT = 117
NV_VGPU_MSG_FUNCTION_CTRL_GET_LATEST_ECC_ADDRESSES = 118
NV_VGPU_MSG_FUNCTION_CTRL_MC_SERVICE_INTERRUPTS = 119
NV_VGPU_MSG_FUNCTION_CTRL_DMA_SET_DEFAULT_VASPACE = 120
NV_VGPU_MSG_FUNCTION_CTRL_GET_CE_PCE_MASK = 121
NV_VGPU_MSG_FUNCTION_CTRL_GET_ZBC_CLEAR_TABLE_ENTRY = 122
NV_VGPU_MSG_FUNCTION_CTRL_GET_NVLINK_PEER_ID_MASK = 123
NV_VGPU_MSG_FUNCTION_CTRL_GET_NVLINK_STATUS = 124
NV_VGPU_MSG_FUNCTION_CTRL_GET_P2P_CAPS = 125
NV_VGPU_MSG_FUNCTION_CTRL_GET_P2P_CAPS_MATRIX = 126
NV_VGPU_MSG_FUNCTION_RESERVED_0 = 127
NV_VGPU_MSG_FUNCTION_CTRL_RESERVE_PM_AREA_SMPC = 128
NV_VGPU_MSG_FUNCTION_CTRL_RESERVE_HWPM_LEGACY = 129
NV_VGPU_MSG_FUNCTION_CTRL_B0CC_EXEC_REG_OPS = 130
NV_VGPU_MSG_FUNCTION_CTRL_BIND_PM_RESOURCES = 131
NV_VGPU_MSG_FUNCTION_CTRL_DBG_SUSPEND_CONTEXT = 132
NV_VGPU_MSG_FUNCTION_CTRL_DBG_RESUME_CONTEXT = 133
NV_VGPU_MSG_FUNCTION_CTRL_DBG_EXEC_REG_OPS = 134
NV_VGPU_MSG_FUNCTION_CTRL_DBG_SET_MODE_MMU_DEBUG = 135
NV_VGPU_MSG_FUNCTION_CTRL_DBG_READ_SINGLE_SM_ERROR_STATE = 136
NV_VGPU_MSG_FUNCTION_CTRL_DBG_CLEAR_SINGLE_SM_ERROR_STATE = 137
NV_VGPU_MSG_FUNCTION_CTRL_DBG_SET_MODE_ERRBAR_DEBUG = 138
NV_VGPU_MSG_FUNCTION_CTRL_DBG_SET_NEXT_STOP_TRIGGER_TYPE = 139
NV_VGPU_MSG_FUNCTION_CTRL_ALLOC_PMA_STREAM = 140
NV_VGPU_MSG_FUNCTION_CTRL_PMA_STREAM_UPDATE_GET_PUT = 141
NV_VGPU_MSG_FUNCTION_CTRL_FB_GET_INFO_V2 = 142
NV_VGPU_MSG_FUNCTION_CTRL_FIFO_SET_CHANNEL_PROPERTIES = 143
NV_VGPU_MSG_FUNCTION_CTRL_GR_GET_CTX_BUFFER_INFO = 144
NV_VGPU_MSG_FUNCTION_CTRL_KGR_GET_CTX_BUFFER_PTES = 145
NV_VGPU_MSG_FUNCTION_CTRL_GPU_EVICT_CTX = 146
NV_VGPU_MSG_FUNCTION_CTRL_FB_GET_FS_INFO = 147
NV_VGPU_MSG_FUNCTION_CTRL_GRMGR_GET_GR_FS_INFO = 148
NV_VGPU_MSG_FUNCTION_CTRL_STOP_CHANNEL = 149
NV_VGPU_MSG_FUNCTION_CTRL_GR_PC_SAMPLING_MODE = 150
NV_VGPU_MSG_FUNCTION_CTRL_PERF_RATED_TDP_GET_STATUS = 151
NV_VGPU_MSG_FUNCTION_CTRL_PERF_RATED_TDP_SET_CONTROL = 152
NV_VGPU_MSG_FUNCTION_CTRL_FREE_PMA_STREAM = 153
NV_VGPU_MSG_FUNCTION_CTRL_TIMER_SET_GR_TICK_FREQ = 154
NV_VGPU_MSG_FUNCTION_CTRL_FIFO_SETUP_VF_ZOMBIE_SUBCTX_PDB = 155
NV_VGPU_MSG_FUNCTION_GET_CONSOLIDATED_GR_STATIC_INFO = 156
NV_VGPU_MSG_FUNCTION_CTRL_DBG_SET_SINGLE_SM_SINGLE_STEP = 157
NV_VGPU_MSG_FUNCTION_CTRL_GR_GET_TPC_PARTITION_MODE = 158
NV_VGPU_MSG_FUNCTION_CTRL_GR_SET_TPC_PARTITION_MODE = 159
NV_VGPU_MSG_FUNCTION_UVM_PAGING_CHANNEL_ALLOCATE = 160
NV_VGPU_MSG_FUNCTION_UVM_PAGING_CHANNEL_DESTROY = 161
NV_VGPU_MSG_FUNCTION_UVM_PAGING_CHANNEL_MAP = 162
NV_VGPU_MSG_FUNCTION_UVM_PAGING_CHANNEL_UNMAP = 163
NV_VGPU_MSG_FUNCTION_UVM_PAGING_CHANNEL_PUSH_STREAM = 164
NV_VGPU_MSG_FUNCTION_UVM_PAGING_CHANNEL_SET_HANDLES = 165
NV_VGPU_MSG_FUNCTION_UVM_METHOD_STREAM_GUEST_PAGES_OPERATION = 166
NV_VGPU_MSG_FUNCTION_CTRL_INTERNAL_QUIESCE_PMA_CHANNEL = 167
NV_VGPU_MSG_FUNCTION_DCE_RM_INIT = 168
NV_VGPU_MSG_FUNCTION_REGISTER_VIRTUAL_EVENT_BUFFER = 169
NV_VGPU_MSG_FUNCTION_CTRL_EVENT_BUFFER_UPDATE_GET = 170
NV_VGPU_MSG_FUNCTION_GET_PLCABLE_ADDRESS_KIND = 171
NV_VGPU_MSG_FUNCTION_CTRL_PERF_LIMITS_SET_STATUS_V2 = 172
NV_VGPU_MSG_FUNCTION_CTRL_INTERNAL_SRIOV_PROMOTE_PMA_STREAM = 173
NV_VGPU_MSG_FUNCTION_CTRL_GET_MMU_DEBUG_MODE = 174
NV_VGPU_MSG_FUNCTION_CTRL_INTERNAL_PROMOTE_FAULT_METHOD_BUFFERS = 175
NV_VGPU_MSG_FUNCTION_CTRL_FLCN_GET_CTX_BUFFER_SIZE = 176
NV_VGPU_MSG_FUNCTION_CTRL_FLCN_GET_CTX_BUFFER_INFO = 177
NV_VGPU_MSG_FUNCTION_DISABLE_CHANNELS = 178
NV_VGPU_MSG_FUNCTION_CTRL_FABRIC_MEMORY_DESCRIBE = 179
NV_VGPU_MSG_FUNCTION_CTRL_FABRIC_MEM_STATS = 180
NV_VGPU_MSG_FUNCTION_SAVE_HIBERNATION_DATA = 181
NV_VGPU_MSG_FUNCTION_RESTORE_HIBERNATION_DATA = 182
NV_VGPU_MSG_FUNCTION_CTRL_INTERNAL_MEMSYS_SET_ZBC_REFERENCED = 183
NV_VGPU_MSG_FUNCTION_CTRL_EXEC_PARTITIONS_CREATE = 184
NV_VGPU_MSG_FUNCTION_CTRL_EXEC_PARTITIONS_DELETE = 185
NV_VGPU_MSG_FUNCTION_CTRL_GPFIFO_GET_WORK_SUBMIT_TOKEN = 186
NV_VGPU_MSG_FUNCTION_CTRL_GPFIFO_SET_WORK_SUBMIT_TOKEN_NOTIF_INDEX = 187
NV_VGPU_MSG_FUNCTION_PMA_SCRUBBER_SHARED_BUFFER_GUEST_PAGES_OPERATION = 188
NV_VGPU_MSG_FUNCTION_CTRL_MASTER_GET_VIRTUAL_FUNCTION_ERROR_CONT_INTR_MASK = 189
NV_VGPU_MSG_FUNCTION_SET_SYSMEM_DIRTY_PAGE_TRACKING_BUFFER = 190
NV_VGPU_MSG_FUNCTION_CTRL_SUBDEVICE_GET_P2P_CAPS = 191
NV_VGPU_MSG_FUNCTION_CTRL_BUS_SET_P2P_MAPPING = 192
NV_VGPU_MSG_FUNCTION_CTRL_BUS_UNSET_P2P_MAPPING = 193
NV_VGPU_MSG_FUNCTION_CTRL_FLA_SETUP_INSTANCE_MEM_BLOCK = 194
NV_VGPU_MSG_FUNCTION_CTRL_GPU_MIGRATABLE_OPS = 195
NV_VGPU_MSG_FUNCTION_CTRL_GET_TOTAL_HS_CREDITS = 196
NV_VGPU_MSG_FUNCTION_CTRL_GET_HS_CREDITS = 197
NV_VGPU_MSG_FUNCTION_CTRL_SET_HS_CREDITS = 198
NV_VGPU_MSG_FUNCTION_CTRL_PM_AREA_PC_SAMPLER = 199
NV_VGPU_MSG_FUNCTION_INVALIDATE_TLB = 200
NV_VGPU_MSG_FUNCTION_CTRL_GPU_QUERY_ECC_STATUS = 201
NV_VGPU_MSG_FUNCTION_ECC_NOTIFIER_WRITE_ACK = 202
NV_VGPU_MSG_FUNCTION_CTRL_DBG_GET_MODE_MMU_DEBUG = 203
NV_VGPU_MSG_FUNCTION_RM_API_CONTROL = 204
NV_VGPU_MSG_FUNCTION_CTRL_CMD_INTERNAL_GPU_START_FABRIC_PROBE = 205
NV_VGPU_MSG_FUNCTION_CTRL_NVLINK_GET_INBAND_RECEIVED_DATA = 206
NV_VGPU_MSG_FUNCTION_GET_STATIC_DATA = 207
NV_VGPU_MSG_FUNCTION_RESERVED_208 = 208
NV_VGPU_MSG_FUNCTION_CTRL_GPU_GET_INFO_V2 = 209
NV_VGPU_MSG_FUNCTION_GET_BRAND_CAPS = 210
NV_VGPU_MSG_FUNCTION_CTRL_CMD_NVLINK_INBAND_SEND_DATA = 211
NV_VGPU_MSG_FUNCTION_UPDATE_GPM_GUEST_BUFFER_INFO = 212
NV_VGPU_MSG_FUNCTION_CTRL_CMD_INTERNAL_CONTROL_GSP_TRACE = 213
NV_VGPU_MSG_FUNCTION_CTRL_SET_ZBC_STENCIL_CLEAR = 214
NV_VGPU_MSG_FUNCTION_CTRL_SUBDEVICE_GET_VGPU_HEAP_STATS = 215
NV_VGPU_MSG_FUNCTION_CTRL_SUBDEVICE_GET_LIBOS_HEAP_STATS = 216
NV_VGPU_MSG_FUNCTION_CTRL_DBG_SET_MODE_MMU_GCC_DEBUG = 217
NV_VGPU_MSG_FUNCTION_CTRL_DBG_GET_MODE_MMU_GCC_DEBUG = 218
NV_VGPU_MSG_FUNCTION_CTRL_RESERVE_HES = 219
NV_VGPU_MSG_FUNCTION_CTRL_RELEASE_HES = 220
NV_VGPU_MSG_FUNCTION_CTRL_RESERVE_CCU_PROF = 221
NV_VGPU_MSG_FUNCTION_CTRL_RELEASE_CCU_PROF = 222
NV_VGPU_MSG_FUNCTION_NUM_FUNCTIONS = 223
c__Ea_NV_VGPU_MSG_FUNCTION_NOP = ctypes.c_uint32 # enum

# values for enumeration 'c__Ea_NV_VGPU_MSG_EVENT_FIRST_EVENT'
c__Ea_NV_VGPU_MSG_EVENT_FIRST_EVENT__enumvalues = {
    4096: 'NV_VGPU_MSG_EVENT_FIRST_EVENT',
    4097: 'NV_VGPU_MSG_EVENT_GSP_INIT_DONE',
    4098: 'NV_VGPU_MSG_EVENT_GSP_RUN_CPU_SEQUENCER',
    4099: 'NV_VGPU_MSG_EVENT_POST_EVENT',
    4100: 'NV_VGPU_MSG_EVENT_RC_TRIGGERED',
    4101: 'NV_VGPU_MSG_EVENT_MMU_FAULT_QUEUED',
    4102: 'NV_VGPU_MSG_EVENT_OS_ERROR_LOG',
    4103: 'NV_VGPU_MSG_EVENT_RG_LINE_INTR',
    4104: 'NV_VGPU_MSG_EVENT_GPUACCT_PERFMON_UTIL_SAMPLES',
    4105: 'NV_VGPU_MSG_EVENT_SIM_READ',
    4106: 'NV_VGPU_MSG_EVENT_SIM_WRITE',
    4107: 'NV_VGPU_MSG_EVENT_SEMAPHORE_SCHEDULE_CALLBACK',
    4108: 'NV_VGPU_MSG_EVENT_UCODE_LIBOS_PRINT',
    4109: 'NV_VGPU_MSG_EVENT_VGPU_GSP_PLUGIN_TRIGGERED',
    4110: 'NV_VGPU_MSG_EVENT_PERF_GPU_BOOST_SYNC_LIMITS_CALLBACK',
    4111: 'NV_VGPU_MSG_EVENT_PERF_BRIDGELESS_INFO_UPDATE',
    4112: 'NV_VGPU_MSG_EVENT_VGPU_CONFIG',
    4113: 'NV_VGPU_MSG_EVENT_DISPLAY_MODESET',
    4114: 'NV_VGPU_MSG_EVENT_EXTDEV_INTR_SERVICE',
    4115: 'NV_VGPU_MSG_EVENT_NVLINK_INBAND_RECEIVED_DATA_256',
    4116: 'NV_VGPU_MSG_EVENT_NVLINK_INBAND_RECEIVED_DATA_512',
    4117: 'NV_VGPU_MSG_EVENT_NVLINK_INBAND_RECEIVED_DATA_1024',
    4118: 'NV_VGPU_MSG_EVENT_NVLINK_INBAND_RECEIVED_DATA_2048',
    4119: 'NV_VGPU_MSG_EVENT_NVLINK_INBAND_RECEIVED_DATA_4096',
    4120: 'NV_VGPU_MSG_EVENT_TIMED_SEMAPHORE_RELEASE',
    4121: 'NV_VGPU_MSG_EVENT_NVLINK_IS_GPU_DEGRADED',
    4122: 'NV_VGPU_MSG_EVENT_PFM_REQ_HNDLR_STATE_SYNC_CALLBACK',
    4123: 'NV_VGPU_MSG_EVENT_NVLINK_FAULT_UP',
    4124: 'NV_VGPU_MSG_EVENT_GSP_LOCKDOWN_NOTICE',
    4125: 'NV_VGPU_MSG_EVENT_MIG_CI_CONFIG_UPDATE',
    4126: 'NV_VGPU_MSG_EVENT_UPDATE_GSP_TRACE',
    4127: 'NV_VGPU_MSG_EVENT_NVLINK_FATAL_ERROR_RECOVERY',
    4128: 'NV_VGPU_MSG_EVENT_GSP_POST_NOCAT_RECORD',
    4129: 'NV_VGPU_MSG_EVENT_FECS_ERROR',
    4130: 'NV_VGPU_MSG_EVENT_RECOVERY_ACTION',
    4131: 'NV_VGPU_MSG_EVENT_NUM_EVENTS',
}
NV_VGPU_MSG_EVENT_FIRST_EVENT = 4096
NV_VGPU_MSG_EVENT_GSP_INIT_DONE = 4097
NV_VGPU_MSG_EVENT_GSP_RUN_CPU_SEQUENCER = 4098
NV_VGPU_MSG_EVENT_POST_EVENT = 4099
NV_VGPU_MSG_EVENT_RC_TRIGGERED = 4100
NV_VGPU_MSG_EVENT_MMU_FAULT_QUEUED = 4101
NV_VGPU_MSG_EVENT_OS_ERROR_LOG = 4102
NV_VGPU_MSG_EVENT_RG_LINE_INTR = 4103
NV_VGPU_MSG_EVENT_GPUACCT_PERFMON_UTIL_SAMPLES = 4104
NV_VGPU_MSG_EVENT_SIM_READ = 4105
NV_VGPU_MSG_EVENT_SIM_WRITE = 4106
NV_VGPU_MSG_EVENT_SEMAPHORE_SCHEDULE_CALLBACK = 4107
NV_VGPU_MSG_EVENT_UCODE_LIBOS_PRINT = 4108
NV_VGPU_MSG_EVENT_VGPU_GSP_PLUGIN_TRIGGERED = 4109
NV_VGPU_MSG_EVENT_PERF_GPU_BOOST_SYNC_LIMITS_CALLBACK = 4110
NV_VGPU_MSG_EVENT_PERF_BRIDGELESS_INFO_UPDATE = 4111
NV_VGPU_MSG_EVENT_VGPU_CONFIG = 4112
NV_VGPU_MSG_EVENT_DISPLAY_MODESET = 4113
NV_VGPU_MSG_EVENT_EXTDEV_INTR_SERVICE = 4114
NV_VGPU_MSG_EVENT_NVLINK_INBAND_RECEIVED_DATA_256 = 4115
NV_VGPU_MSG_EVENT_NVLINK_INBAND_RECEIVED_DATA_512 = 4116
NV_VGPU_MSG_EVENT_NVLINK_INBAND_RECEIVED_DATA_1024 = 4117
NV_VGPU_MSG_EVENT_NVLINK_INBAND_RECEIVED_DATA_2048 = 4118
NV_VGPU_MSG_EVENT_NVLINK_INBAND_RECEIVED_DATA_4096 = 4119
NV_VGPU_MSG_EVENT_TIMED_SEMAPHORE_RELEASE = 4120
NV_VGPU_MSG_EVENT_NVLINK_IS_GPU_DEGRADED = 4121
NV_VGPU_MSG_EVENT_PFM_REQ_HNDLR_STATE_SYNC_CALLBACK = 4122
NV_VGPU_MSG_EVENT_NVLINK_FAULT_UP = 4123
NV_VGPU_MSG_EVENT_GSP_LOCKDOWN_NOTICE = 4124
NV_VGPU_MSG_EVENT_MIG_CI_CONFIG_UPDATE = 4125
NV_VGPU_MSG_EVENT_UPDATE_GSP_TRACE = 4126
NV_VGPU_MSG_EVENT_NVLINK_FATAL_ERROR_RECOVERY = 4127
NV_VGPU_MSG_EVENT_GSP_POST_NOCAT_RECORD = 4128
NV_VGPU_MSG_EVENT_FECS_ERROR = 4129
NV_VGPU_MSG_EVENT_RECOVERY_ACTION = 4130
NV_VGPU_MSG_EVENT_NUM_EVENTS = 4131
c__Ea_NV_VGPU_MSG_EVENT_FIRST_EVENT = ctypes.c_uint32 # enum

# values for enumeration 'c__EA_RPC_GR_BUFFER_TYPE'
c__EA_RPC_GR_BUFFER_TYPE__enumvalues = {
    0: 'RPC_GR_BUFFER_TYPE_GRAPHICS',
    1: 'RPC_GR_BUFFER_TYPE_GRAPHICS_ZCULL',
    2: 'RPC_GR_BUFFER_TYPE_GRAPHICS_GRAPHICS_PM',
    3: 'RPC_GR_BUFFER_TYPE_COMPUTE_PREEMPT',
    4: 'RPC_GR_BUFFER_TYPE_GRAPHICS_PATCH',
    5: 'RPC_GR_BUFFER_TYPE_GRAPHICS_BUNDLE_CB',
    6: 'RPC_GR_BUFFER_TYPE_GRAPHICS_PAGEPOOL_GLOBAL',
    7: 'RPC_GR_BUFFER_TYPE_GRAPHICS_ATTRIBUTE_CB',
    8: 'RPC_GR_BUFFER_TYPE_GRAPHICS_RTV_CB_GLOBAL',
    9: 'RPC_GR_BUFFER_TYPE_GRAPHICS_GFXP_POOL',
    10: 'RPC_GR_BUFFER_TYPE_GRAPHICS_GFXP_CTRL_BLK',
    11: 'RPC_GR_BUFFER_TYPE_GRAPHICS_FECS_EVENT',
    12: 'RPC_GR_BUFFER_TYPE_GRAPHICS_PRIV_ACCESS_MAP',
    13: 'RPC_GR_BUFFER_TYPE_GRAPHICS_MAX',
}
RPC_GR_BUFFER_TYPE_GRAPHICS = 0
RPC_GR_BUFFER_TYPE_GRAPHICS_ZCULL = 1
RPC_GR_BUFFER_TYPE_GRAPHICS_GRAPHICS_PM = 2
RPC_GR_BUFFER_TYPE_COMPUTE_PREEMPT = 3
RPC_GR_BUFFER_TYPE_GRAPHICS_PATCH = 4
RPC_GR_BUFFER_TYPE_GRAPHICS_BUNDLE_CB = 5
RPC_GR_BUFFER_TYPE_GRAPHICS_PAGEPOOL_GLOBAL = 6
RPC_GR_BUFFER_TYPE_GRAPHICS_ATTRIBUTE_CB = 7
RPC_GR_BUFFER_TYPE_GRAPHICS_RTV_CB_GLOBAL = 8
RPC_GR_BUFFER_TYPE_GRAPHICS_GFXP_POOL = 9
RPC_GR_BUFFER_TYPE_GRAPHICS_GFXP_CTRL_BLK = 10
RPC_GR_BUFFER_TYPE_GRAPHICS_FECS_EVENT = 11
RPC_GR_BUFFER_TYPE_GRAPHICS_PRIV_ACCESS_MAP = 12
RPC_GR_BUFFER_TYPE_GRAPHICS_MAX = 13
c__EA_RPC_GR_BUFFER_TYPE = ctypes.c_uint32 # enum
RPC_GR_BUFFER_TYPE = c__EA_RPC_GR_BUFFER_TYPE
RPC_GR_BUFFER_TYPE__enumvalues = c__EA_RPC_GR_BUFFER_TYPE__enumvalues

# values for enumeration 'c__EA_FECS_ERROR_EVENT_TYPE'
c__EA_FECS_ERROR_EVENT_TYPE__enumvalues = {
    0: 'FECS_ERROR_EVENT_TYPE_NONE',
    1: 'FECS_ERROR_EVENT_TYPE_BUFFER_RESET_REQUIRED',
    2: 'FECS_ERROR_EVENT_TYPE_BUFFER_FULL',
    3: 'FECS_ERROR_EVENT_TYPE_MAX',
}
FECS_ERROR_EVENT_TYPE_NONE = 0
FECS_ERROR_EVENT_TYPE_BUFFER_RESET_REQUIRED = 1
FECS_ERROR_EVENT_TYPE_BUFFER_FULL = 2
FECS_ERROR_EVENT_TYPE_MAX = 3
c__EA_FECS_ERROR_EVENT_TYPE = ctypes.c_uint32 # enum
FECS_ERROR_EVENT_TYPE = c__EA_FECS_ERROR_EVENT_TYPE
FECS_ERROR_EVENT_TYPE__enumvalues = c__EA_FECS_ERROR_EVENT_TYPE__enumvalues

# values for enumeration 'c__EA_NV_RPC_UPDATE_PDE_BAR_TYPE'
c__EA_NV_RPC_UPDATE_PDE_BAR_TYPE__enumvalues = {
    0: 'NV_RPC_UPDATE_PDE_BAR_1',
    1: 'NV_RPC_UPDATE_PDE_BAR_2',
    2: 'NV_RPC_UPDATE_PDE_BAR_INVALID',
}
NV_RPC_UPDATE_PDE_BAR_1 = 0
NV_RPC_UPDATE_PDE_BAR_2 = 1
NV_RPC_UPDATE_PDE_BAR_INVALID = 2
c__EA_NV_RPC_UPDATE_PDE_BAR_TYPE = ctypes.c_uint32 # enum
NV_RPC_UPDATE_PDE_BAR_TYPE = c__EA_NV_RPC_UPDATE_PDE_BAR_TYPE
NV_RPC_UPDATE_PDE_BAR_TYPE__enumvalues = c__EA_NV_RPC_UPDATE_PDE_BAR_TYPE__enumvalues
class struct_VIRTUAL_DISPLAY_GET_MAX_RESOLUTION_PARAMS(Structure):
    pass

struct_VIRTUAL_DISPLAY_GET_MAX_RESOLUTION_PARAMS._pack_ = 1 # source:False
struct_VIRTUAL_DISPLAY_GET_MAX_RESOLUTION_PARAMS._fields_ = [
    ('headIndex', ctypes.c_uint32),
    ('maxHResolution', ctypes.c_uint32),
    ('maxVResolution', ctypes.c_uint32),
]

VIRTUAL_DISPLAY_GET_MAX_RESOLUTION_PARAMS = struct_VIRTUAL_DISPLAY_GET_MAX_RESOLUTION_PARAMS
class struct_VIRTUAL_DISPLAY_GET_NUM_HEADS_PARAMS(Structure):
    pass

struct_VIRTUAL_DISPLAY_GET_NUM_HEADS_PARAMS._pack_ = 1 # source:False
struct_VIRTUAL_DISPLAY_GET_NUM_HEADS_PARAMS._fields_ = [
    ('numHeads', ctypes.c_uint32),
    ('maxNumHeads', ctypes.c_uint32),
]

VIRTUAL_DISPLAY_GET_NUM_HEADS_PARAMS = struct_VIRTUAL_DISPLAY_GET_NUM_HEADS_PARAMS

# values for enumeration 'c__EA_GPU_RECOVERY_EVENT_TYPE'
c__EA_GPU_RECOVERY_EVENT_TYPE__enumvalues = {
    0: 'GPU_RECOVERY_EVENT_TYPE_REFRESH',
    1: 'GPU_RECOVERY_EVENT_TYPE_GPU_DRAIN_P2P',
    2: 'GPU_RECOVERY_EVENT_TYPE_SYS_REBOOT',
}
GPU_RECOVERY_EVENT_TYPE_REFRESH = 0
GPU_RECOVERY_EVENT_TYPE_GPU_DRAIN_P2P = 1
GPU_RECOVERY_EVENT_TYPE_SYS_REBOOT = 2
c__EA_GPU_RECOVERY_EVENT_TYPE = ctypes.c_uint32 # enum
GPU_RECOVERY_EVENT_TYPE = c__EA_GPU_RECOVERY_EVENT_TYPE
GPU_RECOVERY_EVENT_TYPE__enumvalues = c__EA_GPU_RECOVERY_EVENT_TYPE__enumvalues
SDK_STRUCTURES = True # macro
class struct_rpc_set_guest_system_info_v03_00(Structure):
    pass

struct_rpc_set_guest_system_info_v03_00._pack_ = 1 # source:False
struct_rpc_set_guest_system_info_v03_00._fields_ = [
    ('vgxVersionMajorNum', ctypes.c_uint32),
    ('vgxVersionMinorNum', ctypes.c_uint32),
    ('guestDriverVersionBufferLength', ctypes.c_uint32),
    ('guestVersionBufferLength', ctypes.c_uint32),
    ('guestTitleBufferLength', ctypes.c_uint32),
    ('guestClNum', ctypes.c_uint32),
    ('guestDriverVersion', ctypes.c_char * 256),
    ('guestVersion', ctypes.c_char * 256),
    ('guestTitle', ctypes.c_char * 256),
]

rpc_set_guest_system_info_v03_00 = struct_rpc_set_guest_system_info_v03_00
rpc_set_guest_system_info_v = struct_rpc_set_guest_system_info_v03_00
class struct_rpc_set_guest_system_info_ext_v15_02(Structure):
    pass

struct_rpc_set_guest_system_info_ext_v15_02._pack_ = 1 # source:False
struct_rpc_set_guest_system_info_ext_v15_02._fields_ = [
    ('guestDriverBranch', ctypes.c_char * 256),
    ('domain', ctypes.c_uint32),
    ('bus', ctypes.c_uint16),
    ('device', ctypes.c_uint16),
]

rpc_set_guest_system_info_ext_v15_02 = struct_rpc_set_guest_system_info_ext_v15_02
class struct_rpc_set_guest_system_info_ext_v25_1B(Structure):
    pass

struct_rpc_set_guest_system_info_ext_v25_1B._pack_ = 1 # source:False
struct_rpc_set_guest_system_info_ext_v25_1B._fields_ = [
    ('guestDriverBranch', ctypes.c_char * 256),
    ('domain', ctypes.c_uint32),
    ('bus', ctypes.c_uint16),
    ('device', ctypes.c_uint16),
    ('gridBuildCsp', ctypes.c_uint32),
]

rpc_set_guest_system_info_ext_v25_1B = struct_rpc_set_guest_system_info_ext_v25_1B
rpc_set_guest_system_info_ext_v = struct_rpc_set_guest_system_info_ext_v25_1B
class struct_rpc_alloc_root_v07_00(Structure):
    pass

struct_rpc_alloc_root_v07_00._pack_ = 1 # source:False
struct_rpc_alloc_root_v07_00._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('processID', ctypes.c_uint32),
    ('processName', ctypes.c_char * 100),
]

rpc_alloc_root_v07_00 = struct_rpc_alloc_root_v07_00
rpc_alloc_root_v = struct_rpc_alloc_root_v07_00
class struct_rpc_alloc_memory_v13_01(Structure):
    pass

class struct_pte_desc(Structure):
    pass

class union_pte_desc_0(Union):
    pass

union_pte_desc_0._pack_ = 1 # source:False
union_pte_desc_0._fields_ = [
    ('pte', ctypes.c_uint64),
    ('pde', ctypes.c_uint64),
]

struct_pte_desc._pack_ = 1 # source:False
struct_pte_desc._fields_ = [
    ('idr', ctypes.c_uint32, 2),
    ('reserved1', ctypes.c_uint32, 14),
    ('length', ctypes.c_uint32, 16),
    ('PADDING_0', ctypes.c_uint32, 32),
    ('pte_pde', union_pte_desc_0 * 0),
]

struct_rpc_alloc_memory_v13_01._pack_ = 1 # source:False
struct_rpc_alloc_memory_v13_01._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hDevice', ctypes.c_uint32),
    ('hMemory', ctypes.c_uint32),
    ('hClass', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('pteAdjust', ctypes.c_uint32),
    ('format', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('length', ctypes.c_uint64),
    ('pageCount', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('pteDesc', struct_pte_desc),
]

rpc_alloc_memory_v13_01 = struct_rpc_alloc_memory_v13_01
rpc_alloc_memory_v = struct_rpc_alloc_memory_v13_01
class struct_rpc_alloc_channel_dma_v1F_04(Structure):
    pass

class struct_NV_CHANNEL_ALLOC_PARAMS_v1F_04(Structure):
    pass

class struct_NV_MEMORY_DESC_PARAMS_v18_01(Structure):
    pass

struct_NV_MEMORY_DESC_PARAMS_v18_01._pack_ = 1 # source:False
struct_NV_MEMORY_DESC_PARAMS_v18_01._fields_ = [
    ('base', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
    ('addressSpace', ctypes.c_uint32),
    ('cacheAttrib', ctypes.c_uint32),
]

struct_NV_CHANNEL_ALLOC_PARAMS_v1F_04._pack_ = 1 # source:False
struct_NV_CHANNEL_ALLOC_PARAMS_v1F_04._fields_ = [
    ('hObjectError', ctypes.c_uint32),
    ('hObjectBuffer', ctypes.c_uint32),
    ('gpFifoOffset', ctypes.c_uint64),
    ('gpFifoEntries', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('hContextShare', ctypes.c_uint32),
    ('hVASpace', ctypes.c_uint32),
    ('hUserdMemory', ctypes.c_uint32 * 1),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('userdOffset', ctypes.c_uint64 * 1),
    ('engineType', ctypes.c_uint32),
    ('hObjectEccError', ctypes.c_uint32),
    ('instanceMem', struct_NV_MEMORY_DESC_PARAMS_v18_01),
    ('ramfcMem', struct_NV_MEMORY_DESC_PARAMS_v18_01),
    ('userdMem', struct_NV_MEMORY_DESC_PARAMS_v18_01),
    ('mthdbufMem', struct_NV_MEMORY_DESC_PARAMS_v18_01),
    ('hPhysChannelGroup', ctypes.c_uint32),
    ('subDeviceId', ctypes.c_uint32),
    ('internalFlags', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('errorNotifierMem', struct_NV_MEMORY_DESC_PARAMS_v18_01),
    ('eccErrorNotifierMem', struct_NV_MEMORY_DESC_PARAMS_v18_01),
]

struct_rpc_alloc_channel_dma_v1F_04._pack_ = 1 # source:False
struct_rpc_alloc_channel_dma_v1F_04._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hDevice', ctypes.c_uint32),
    ('hChannel', ctypes.c_uint32),
    ('hClass', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('params', struct_NV_CHANNEL_ALLOC_PARAMS_v1F_04),
    ('chid', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

rpc_alloc_channel_dma_v1F_04 = struct_rpc_alloc_channel_dma_v1F_04
rpc_alloc_channel_dma_v = struct_rpc_alloc_channel_dma_v1F_04
class struct_rpc_alloc_object_v25_08(Structure):
    pass

class union_alloc_object_params_v25_08(Union):
    pass

class struct_alloc_object_NV50_TESLA_v03_00(Structure):
    pass

struct_alloc_object_NV50_TESLA_v03_00._pack_ = 1 # source:False
struct_alloc_object_NV50_TESLA_v03_00._fields_ = [
    ('version', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('size', ctypes.c_uint32),
    ('caps', ctypes.c_uint32),
]

class struct_alloc_object_GT212_DMA_COPY_v03_00(Structure):
    pass

struct_alloc_object_GT212_DMA_COPY_v03_00._pack_ = 1 # source:False
struct_alloc_object_GT212_DMA_COPY_v03_00._fields_ = [
    ('version', ctypes.c_uint32),
    ('engineInstance', ctypes.c_uint32),
]

class struct_alloc_object_GF100_DISP_SW_v03_00(Structure):
    pass

struct_alloc_object_GF100_DISP_SW_v03_00._pack_ = 1 # source:False
struct_alloc_object_GF100_DISP_SW_v03_00._fields_ = [
    ('_reserved1', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('_reserved2', ctypes.c_uint64),
    ('logicalHeadId', ctypes.c_uint32),
    ('displayMask', ctypes.c_uint32),
    ('caps', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

class struct_alloc_object_KEPLER_CHANNEL_GROUP_A_v12_08(Structure):
    pass

struct_alloc_object_KEPLER_CHANNEL_GROUP_A_v12_08._pack_ = 1 # source:False
struct_alloc_object_KEPLER_CHANNEL_GROUP_A_v12_08._fields_ = [
    ('hObjectError', ctypes.c_uint32),
    ('hVASpace', ctypes.c_uint32),
    ('engineType', ctypes.c_uint32),
]

class struct_alloc_object_FERMI_CONTEXT_SHARE_A_v04_00(Structure):
    pass

struct_alloc_object_FERMI_CONTEXT_SHARE_A_v04_00._pack_ = 1 # source:False
struct_alloc_object_FERMI_CONTEXT_SHARE_A_v04_00._fields_ = [
    ('hVASpace', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('subctxId', ctypes.c_uint32),
]

class struct_alloc_object_NVD0B7_VIDEO_ENCODER_v03_00(Structure):
    pass

struct_alloc_object_NVD0B7_VIDEO_ENCODER_v03_00._pack_ = 1 # source:False
struct_alloc_object_NVD0B7_VIDEO_ENCODER_v03_00._fields_ = [
    ('size', ctypes.c_uint32),
    ('prohibitMultipleInstances', ctypes.c_uint32),
    ('engineInstance', ctypes.c_uint32),
]

class struct_alloc_object_FERMI_VASPACE_A_v03_00(Structure):
    pass

struct_alloc_object_FERMI_VASPACE_A_v03_00._pack_ = 1 # source:False
struct_alloc_object_FERMI_VASPACE_A_v03_00._fields_ = [
    ('index', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('vaSize', ctypes.c_uint64),
    ('bigPageSize', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('vaBase', ctypes.c_uint64),
]

class struct_alloc_object_NVB0B0_VIDEO_DECODER_v03_00(Structure):
    pass

struct_alloc_object_NVB0B0_VIDEO_DECODER_v03_00._pack_ = 1 # source:False
struct_alloc_object_NVB0B0_VIDEO_DECODER_v03_00._fields_ = [
    ('size', ctypes.c_uint32),
    ('prohibitMultipleInstances', ctypes.c_uint32),
]

class struct_alloc_object_NV83DE_ALLOC_PARAMETERS_v03_00(Structure):
    pass

struct_alloc_object_NV83DE_ALLOC_PARAMETERS_v03_00._pack_ = 1 # source:False
struct_alloc_object_NV83DE_ALLOC_PARAMETERS_v03_00._fields_ = [
    ('hDebuggerClient', ctypes.c_uint32),
    ('hAppClient', ctypes.c_uint32),
    ('hClass3dObject', ctypes.c_uint32),
]

class struct_alloc_object_NVENC_SW_SESSION_v06_01(Structure):
    pass

struct_alloc_object_NVENC_SW_SESSION_v06_01._pack_ = 1 # source:False
struct_alloc_object_NVENC_SW_SESSION_v06_01._fields_ = [
    ('codecType', ctypes.c_uint32),
    ('hResolution', ctypes.c_uint32),
    ('vResolution', ctypes.c_uint32),
]

class struct_alloc_object_NVC4B0_VIDEO_DECODER_v12_02(Structure):
    pass

struct_alloc_object_NVC4B0_VIDEO_DECODER_v12_02._pack_ = 1 # source:False
struct_alloc_object_NVC4B0_VIDEO_DECODER_v12_02._fields_ = [
    ('size', ctypes.c_uint32),
    ('prohibitMultipleInstances', ctypes.c_uint32),
    ('engineInstance', ctypes.c_uint32),
]

class struct_alloc_object_NVFBC_SW_SESSION_v12_04(Structure):
    pass

struct_alloc_object_NVFBC_SW_SESSION_v12_04._pack_ = 1 # source:False
struct_alloc_object_NVFBC_SW_SESSION_v12_04._fields_ = [
    ('displayOrdinal', ctypes.c_uint32),
    ('sessionType', ctypes.c_uint32),
    ('sessionFlags', ctypes.c_uint32),
    ('hMaxResolution', ctypes.c_uint32),
    ('vMaxResolution', ctypes.c_uint32),
]

class struct_alloc_object_NV_NVJPG_ALLOCATION_PARAMETERS_v20_02(Structure):
    pass

struct_alloc_object_NV_NVJPG_ALLOCATION_PARAMETERS_v20_02._pack_ = 1 # source:False
struct_alloc_object_NV_NVJPG_ALLOCATION_PARAMETERS_v20_02._fields_ = [
    ('size', ctypes.c_uint32),
    ('prohibitMultipleInstances', ctypes.c_uint32),
    ('engineInstance', ctypes.c_uint32),
]

class struct_alloc_object_NV503B_ALLOC_PARAMETERS_v1D_02(Structure):
    pass

struct_alloc_object_NV503B_ALLOC_PARAMETERS_v1D_02._pack_ = 1 # source:False
struct_alloc_object_NV503B_ALLOC_PARAMETERS_v1D_02._fields_ = [
    ('hSubDevice', ctypes.c_uint32),
    ('hPeerSubDevice', ctypes.c_uint32),
    ('subDevicePeerIdMask', ctypes.c_uint32),
    ('peerSubDevicePeerIdMask', ctypes.c_uint32),
    ('mailboxBar1Addr', ctypes.c_uint64),
    ('mailboxTotalSize', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
]

class struct_alloc_object_NVC637_ALLOCATION_PARAMETERS_v13_00(Structure):
    pass

struct_alloc_object_NVC637_ALLOCATION_PARAMETERS_v13_00._pack_ = 1 # source:False
struct_alloc_object_NVC637_ALLOCATION_PARAMETERS_v13_00._fields_ = [
    ('swizzId', ctypes.c_uint32),
]

class struct_alloc_object_NV_MEMORY_VIRTUAL_ALLOCATION_PARAMS_v13_03(Structure):
    pass

struct_alloc_object_NV_MEMORY_VIRTUAL_ALLOCATION_PARAMS_v13_03._pack_ = 1 # source:False
struct_alloc_object_NV_MEMORY_VIRTUAL_ALLOCATION_PARAMS_v13_03._fields_ = [
    ('offset', ctypes.c_uint64),
    ('limit', ctypes.c_uint64),
    ('hVASpace', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

class struct_alloc_object_NVC638_ALLOCATION_PARAMETERS_v18_06(Structure):
    pass

struct_alloc_object_NVC638_ALLOCATION_PARAMETERS_v18_06._pack_ = 1 # source:False
struct_alloc_object_NVC638_ALLOCATION_PARAMETERS_v18_06._fields_ = [
    ('execPartitionId', ctypes.c_uint32),
]

class struct_alloc_object_NV503C_ALLOC_PARAMETERS_v18_15(Structure):
    pass

struct_alloc_object_NV503C_ALLOC_PARAMETERS_v18_15._pack_ = 1 # source:False
struct_alloc_object_NV503C_ALLOC_PARAMETERS_v18_15._fields_ = [
    ('flags', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('p2pToken', ctypes.c_uint64),
]

class struct_alloc_object_NVC670_ALLOCATION_PARAMETERS_v1A_01(Structure):
    pass

struct_alloc_object_NVC670_ALLOCATION_PARAMETERS_v1A_01._pack_ = 1 # source:False
struct_alloc_object_NVC670_ALLOCATION_PARAMETERS_v1A_01._fields_ = [
    ('numHeads', ctypes.c_uint32),
    ('numSors', ctypes.c_uint32),
    ('numDsis', ctypes.c_uint32),
]

class struct_alloc_object_NVB1CC_ALLOC_PARAMETERS_v1A_03(Structure):
    pass

struct_alloc_object_NVB1CC_ALLOC_PARAMETERS_v1A_03._pack_ = 1 # source:False
struct_alloc_object_NVB1CC_ALLOC_PARAMETERS_v1A_03._fields_ = [
    ('hSubDevice', ctypes.c_uint32),
]

class struct_alloc_object_NVB2CC_ALLOC_PARAMETERS_v1A_03(Structure):
    pass

struct_alloc_object_NVB2CC_ALLOC_PARAMETERS_v1A_03._pack_ = 1 # source:False
struct_alloc_object_NVB2CC_ALLOC_PARAMETERS_v1A_03._fields_ = [
    ('hClientTarget', ctypes.c_uint32),
    ('hContextTarget', ctypes.c_uint32),
]

class struct_NV_GR_ALLOCATION_PARAMETERS_v1A_17(Structure):
    pass

struct_NV_GR_ALLOCATION_PARAMETERS_v1A_17._pack_ = 1 # source:False
struct_NV_GR_ALLOCATION_PARAMETERS_v1A_17._fields_ = [
    ('version', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('size', ctypes.c_uint32),
    ('caps', ctypes.c_uint32),
]

class struct_alloc_object_NV_UVM_CHANNEL_RETAINER_ALLOC_PARAMS_v1A_1B(Structure):
    pass

struct_alloc_object_NV_UVM_CHANNEL_RETAINER_ALLOC_PARAMS_v1A_1B._pack_ = 1 # source:False
struct_alloc_object_NV_UVM_CHANNEL_RETAINER_ALLOC_PARAMS_v1A_1B._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hChannel', ctypes.c_uint32),
]

class struct_alloc_object_NV00F8_ALLOCATION_PARAMETERS_v1E_0C(Structure):
    pass

class struct_NV00F8_ALLOCATION_PARAMETERS_MAP_STRUCT_v1E_0C(Structure):
    pass

struct_NV00F8_ALLOCATION_PARAMETERS_MAP_STRUCT_v1E_0C._pack_ = 1 # source:False
struct_NV00F8_ALLOCATION_PARAMETERS_MAP_STRUCT_v1E_0C._fields_ = [
    ('offset', ctypes.c_uint64),
    ('hVidMem', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
]

struct_alloc_object_NV00F8_ALLOCATION_PARAMETERS_v1E_0C._pack_ = 1 # source:False
struct_alloc_object_NV00F8_ALLOCATION_PARAMETERS_v1E_0C._fields_ = [
    ('alignment', ctypes.c_uint64),
    ('allocSize', ctypes.c_uint64),
    ('pageSize', ctypes.c_uint32),
    ('allocFlags', ctypes.c_uint32),
    ('map', struct_NV00F8_ALLOCATION_PARAMETERS_MAP_STRUCT_v1E_0C),
]

class struct_alloc_object_NVC9FA_VIDEO_OFA_v1F_00(Structure):
    pass

struct_alloc_object_NVC9FA_VIDEO_OFA_v1F_00._pack_ = 1 # source:False
struct_alloc_object_NVC9FA_VIDEO_OFA_v1F_00._fields_ = [
    ('size', ctypes.c_uint32),
    ('prohibitMultipleInstances', ctypes.c_uint32),
]

class struct_alloc_object_NV2081_ALLOC_PARAMETERS_v25_08(Structure):
    pass

struct_alloc_object_NV2081_ALLOC_PARAMETERS_v25_08._pack_ = 1 # source:False
struct_alloc_object_NV2081_ALLOC_PARAMETERS_v25_08._fields_ = [
    ('reserved', ctypes.c_uint32),
]

union_alloc_object_params_v25_08._pack_ = 1 # source:False
union_alloc_object_params_v25_08._fields_ = [
    ('param_NV50_TESLA', struct_alloc_object_NV50_TESLA_v03_00),
    ('param_GT212_DMA_COPY', struct_alloc_object_GT212_DMA_COPY_v03_00),
    ('param_GF100_DISP_SW', struct_alloc_object_GF100_DISP_SW_v03_00),
    ('param_KEPLER_CHANNEL_GROUP_A', struct_alloc_object_KEPLER_CHANNEL_GROUP_A_v12_08),
    ('param_FERMI_CONTEXT_SHARE_A', struct_alloc_object_FERMI_CONTEXT_SHARE_A_v04_00),
    ('param_NVD0B7_VIDEO_ENCODER', struct_alloc_object_NVD0B7_VIDEO_ENCODER_v03_00),
    ('param_FERMI_VASPACE_A', struct_alloc_object_FERMI_VASPACE_A_v03_00),
    ('param_NVB0B0_VIDEO_DECODER', struct_alloc_object_NVB0B0_VIDEO_DECODER_v03_00),
    ('param_NV83DE_ALLOC_PARAMETERS', struct_alloc_object_NV83DE_ALLOC_PARAMETERS_v03_00),
    ('param_NVENC_SW_SESSION', struct_alloc_object_NVENC_SW_SESSION_v06_01),
    ('param_NVC4B0_VIDEO_DECODER', struct_alloc_object_NVC4B0_VIDEO_DECODER_v12_02),
    ('param_NVFBC_SW_SESSION', struct_alloc_object_NVFBC_SW_SESSION_v12_04),
    ('param_NV_NVJPG_ALLOCATION_PARAMETERS', struct_alloc_object_NV_NVJPG_ALLOCATION_PARAMETERS_v20_02),
    ('param_NV503B_ALLOC_PARAMETERS', struct_alloc_object_NV503B_ALLOC_PARAMETERS_v1D_02),
    ('param_NVC637_ALLOCATION_PARAMETERS', struct_alloc_object_NVC637_ALLOCATION_PARAMETERS_v13_00),
    ('param_NV_MEMORY_VIRTUAL_ALLOCATION_PARAMS', struct_alloc_object_NV_MEMORY_VIRTUAL_ALLOCATION_PARAMS_v13_03),
    ('param_NVC638_ALLOCATION_PARAMETERS', struct_alloc_object_NVC638_ALLOCATION_PARAMETERS_v18_06),
    ('param_NV503C_ALLOC_PARAMETERS', struct_alloc_object_NV503C_ALLOC_PARAMETERS_v18_15),
    ('param_NVC670_ALLOCATION_PARAMETERS', struct_alloc_object_NVC670_ALLOCATION_PARAMETERS_v1A_01),
    ('param_NVB1CC_ALLOC_PARAMETERS', struct_alloc_object_NVB1CC_ALLOC_PARAMETERS_v1A_03),
    ('param_NVB2CC_ALLOC_PARAMETERS', struct_alloc_object_NVB2CC_ALLOC_PARAMETERS_v1A_03),
    ('param_NV_GR_ALLOCATION_PARAMETERS', struct_NV_GR_ALLOCATION_PARAMETERS_v1A_17),
    ('param_NV_UVM_CHANNEL_RETAINER_ALLOC_PARAMS', struct_alloc_object_NV_UVM_CHANNEL_RETAINER_ALLOC_PARAMS_v1A_1B),
    ('param_NV00F8_ALLOCATION_PARAMETERS', struct_alloc_object_NV00F8_ALLOCATION_PARAMETERS_v1E_0C),
    ('param_NVC9FA_VIDEO_OFA', struct_alloc_object_NVC9FA_VIDEO_OFA_v1F_00),
    ('param_NV2081_ALLOC_PARAMETERS', struct_alloc_object_NV2081_ALLOC_PARAMETERS_v25_08),
    ('PADDING_0', ctypes.c_ubyte * 36),
]

struct_rpc_alloc_object_v25_08._pack_ = 1 # source:False
struct_rpc_alloc_object_v25_08._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hParent', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('hClass', ctypes.c_uint32),
    ('param_len', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('params', union_alloc_object_params_v25_08),
]

rpc_alloc_object_v25_08 = struct_rpc_alloc_object_v25_08
class struct_rpc_alloc_object_v26_00(Structure):
    pass

class union_alloc_object_params_v26_00(Union):
    pass

union_alloc_object_params_v26_00._pack_ = 1 # source:False
union_alloc_object_params_v26_00._fields_ = [
    ('param_NV50_TESLA', struct_alloc_object_NV50_TESLA_v03_00),
    ('param_GT212_DMA_COPY', struct_alloc_object_GT212_DMA_COPY_v03_00),
    ('param_GF100_DISP_SW', struct_alloc_object_GF100_DISP_SW_v03_00),
    ('param_KEPLER_CHANNEL_GROUP_A', struct_alloc_object_KEPLER_CHANNEL_GROUP_A_v12_08),
    ('param_FERMI_CONTEXT_SHARE_A', struct_alloc_object_FERMI_CONTEXT_SHARE_A_v04_00),
    ('param_NVD0B7_VIDEO_ENCODER', struct_alloc_object_NVD0B7_VIDEO_ENCODER_v03_00),
    ('param_FERMI_VASPACE_A', struct_alloc_object_FERMI_VASPACE_A_v03_00),
    ('param_NVB0B0_VIDEO_DECODER', struct_alloc_object_NVB0B0_VIDEO_DECODER_v03_00),
    ('param_NV83DE_ALLOC_PARAMETERS', struct_alloc_object_NV83DE_ALLOC_PARAMETERS_v03_00),
    ('param_NVENC_SW_SESSION', struct_alloc_object_NVENC_SW_SESSION_v06_01),
    ('param_NVC4B0_VIDEO_DECODER', struct_alloc_object_NVC4B0_VIDEO_DECODER_v12_02),
    ('param_NVFBC_SW_SESSION', struct_alloc_object_NVFBC_SW_SESSION_v12_04),
    ('param_NV_NVJPG_ALLOCATION_PARAMETERS', struct_alloc_object_NV_NVJPG_ALLOCATION_PARAMETERS_v20_02),
    ('param_NV503B_ALLOC_PARAMETERS', struct_alloc_object_NV503B_ALLOC_PARAMETERS_v1D_02),
    ('param_NVC637_ALLOCATION_PARAMETERS', struct_alloc_object_NVC637_ALLOCATION_PARAMETERS_v13_00),
    ('param_NV_MEMORY_VIRTUAL_ALLOCATION_PARAMS', struct_alloc_object_NV_MEMORY_VIRTUAL_ALLOCATION_PARAMS_v13_03),
    ('param_NVC638_ALLOCATION_PARAMETERS', struct_alloc_object_NVC638_ALLOCATION_PARAMETERS_v18_06),
    ('param_NV503C_ALLOC_PARAMETERS', struct_alloc_object_NV503C_ALLOC_PARAMETERS_v18_15),
    ('param_NVC670_ALLOCATION_PARAMETERS', struct_alloc_object_NVC670_ALLOCATION_PARAMETERS_v1A_01),
    ('param_NVB1CC_ALLOC_PARAMETERS', struct_alloc_object_NVB1CC_ALLOC_PARAMETERS_v1A_03),
    ('param_NVB2CC_ALLOC_PARAMETERS', struct_alloc_object_NVB2CC_ALLOC_PARAMETERS_v1A_03),
    ('param_NV_GR_ALLOCATION_PARAMETERS', struct_NV_GR_ALLOCATION_PARAMETERS_v1A_17),
    ('param_NV_UVM_CHANNEL_RETAINER_ALLOC_PARAMS', struct_alloc_object_NV_UVM_CHANNEL_RETAINER_ALLOC_PARAMS_v1A_1B),
    ('param_NV00F8_ALLOCATION_PARAMETERS', struct_alloc_object_NV00F8_ALLOCATION_PARAMETERS_v1E_0C),
    ('param_NVC9FA_VIDEO_OFA', struct_alloc_object_NVC9FA_VIDEO_OFA_v1F_00),
    ('param_NV2081_ALLOC_PARAMETERS', struct_alloc_object_NV2081_ALLOC_PARAMETERS_v25_08),
    ('param_padding', ctypes.c_ubyte * 56),
]

struct_rpc_alloc_object_v26_00._pack_ = 1 # source:False
struct_rpc_alloc_object_v26_00._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hParent', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('hClass', ctypes.c_uint32),
    ('param_len', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('params', union_alloc_object_params_v26_00),
]

rpc_alloc_object_v26_00 = struct_rpc_alloc_object_v26_00
class struct_rpc_alloc_object_v27_00(Structure):
    pass

class union_alloc_object_params_v27_00(Union):
    pass

union_alloc_object_params_v27_00._pack_ = 1 # source:False
union_alloc_object_params_v27_00._fields_ = [
    ('param_NV50_TESLA', struct_alloc_object_NV50_TESLA_v03_00),
    ('param_GT212_DMA_COPY', struct_alloc_object_GT212_DMA_COPY_v03_00),
    ('param_GF100_DISP_SW', struct_alloc_object_GF100_DISP_SW_v03_00),
    ('param_KEPLER_CHANNEL_GROUP_A', struct_alloc_object_KEPLER_CHANNEL_GROUP_A_v12_08),
    ('param_FERMI_CONTEXT_SHARE_A', struct_alloc_object_FERMI_CONTEXT_SHARE_A_v04_00),
    ('param_NVD0B7_VIDEO_ENCODER', struct_alloc_object_NVD0B7_VIDEO_ENCODER_v03_00),
    ('param_FERMI_VASPACE_A', struct_alloc_object_FERMI_VASPACE_A_v03_00),
    ('param_NVB0B0_VIDEO_DECODER', struct_alloc_object_NVB0B0_VIDEO_DECODER_v03_00),
    ('param_NV83DE_ALLOC_PARAMETERS', struct_alloc_object_NV83DE_ALLOC_PARAMETERS_v03_00),
    ('param_NVENC_SW_SESSION', struct_alloc_object_NVENC_SW_SESSION_v06_01),
    ('param_NVC4B0_VIDEO_DECODER', struct_alloc_object_NVC4B0_VIDEO_DECODER_v12_02),
    ('param_NVFBC_SW_SESSION', struct_alloc_object_NVFBC_SW_SESSION_v12_04),
    ('param_NV_NVJPG_ALLOCATION_PARAMETERS', struct_alloc_object_NV_NVJPG_ALLOCATION_PARAMETERS_v20_02),
    ('param_NV503B_ALLOC_PARAMETERS', struct_alloc_object_NV503B_ALLOC_PARAMETERS_v1D_02),
    ('param_NVC637_ALLOCATION_PARAMETERS', struct_alloc_object_NVC637_ALLOCATION_PARAMETERS_v13_00),
    ('param_NV_MEMORY_VIRTUAL_ALLOCATION_PARAMS', struct_alloc_object_NV_MEMORY_VIRTUAL_ALLOCATION_PARAMS_v13_03),
    ('param_NVC638_ALLOCATION_PARAMETERS', struct_alloc_object_NVC638_ALLOCATION_PARAMETERS_v18_06),
    ('param_NV503C_ALLOC_PARAMETERS', struct_alloc_object_NV503C_ALLOC_PARAMETERS_v18_15),
    ('param_NVC670_ALLOCATION_PARAMETERS', struct_alloc_object_NVC670_ALLOCATION_PARAMETERS_v1A_01),
    ('param_NVB1CC_ALLOC_PARAMETERS', struct_alloc_object_NVB1CC_ALLOC_PARAMETERS_v1A_03),
    ('param_NVB2CC_ALLOC_PARAMETERS', struct_alloc_object_NVB2CC_ALLOC_PARAMETERS_v1A_03),
    ('param_NV_GR_ALLOCATION_PARAMETERS', struct_NV_GR_ALLOCATION_PARAMETERS_v1A_17),
    ('param_NV_UVM_CHANNEL_RETAINER_ALLOC_PARAMS', struct_alloc_object_NV_UVM_CHANNEL_RETAINER_ALLOC_PARAMS_v1A_1B),
    ('param_NV00F8_ALLOCATION_PARAMETERS', struct_alloc_object_NV00F8_ALLOCATION_PARAMETERS_v1E_0C),
    ('param_NVC9FA_VIDEO_OFA', struct_alloc_object_NVC9FA_VIDEO_OFA_v1F_00),
    ('param_NV2081_ALLOC_PARAMETERS', struct_alloc_object_NV2081_ALLOC_PARAMETERS_v25_08),
    ('param_padding', ctypes.c_ubyte * 56),
]

struct_rpc_alloc_object_v27_00._pack_ = 1 # source:False
struct_rpc_alloc_object_v27_00._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hParent', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('hClass', ctypes.c_uint32),
    ('param_len', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('params', union_alloc_object_params_v27_00),
]

rpc_alloc_object_v27_00 = struct_rpc_alloc_object_v27_00
class struct_rpc_alloc_object_v29_06(Structure):
    pass

class union_alloc_object_params_v29_06(Union):
    pass

class struct_alloc_object_NVC9FA_VIDEO_OFA_v29_06(Structure):
    pass

struct_alloc_object_NVC9FA_VIDEO_OFA_v29_06._pack_ = 1 # source:False
struct_alloc_object_NVC9FA_VIDEO_OFA_v29_06._fields_ = [
    ('size', ctypes.c_uint32),
    ('prohibitMultipleInstances', ctypes.c_uint32),
    ('engineInstance', ctypes.c_uint32),
]

union_alloc_object_params_v29_06._pack_ = 1 # source:False
union_alloc_object_params_v29_06._fields_ = [
    ('param_NV50_TESLA', struct_alloc_object_NV50_TESLA_v03_00),
    ('param_GT212_DMA_COPY', struct_alloc_object_GT212_DMA_COPY_v03_00),
    ('param_GF100_DISP_SW', struct_alloc_object_GF100_DISP_SW_v03_00),
    ('param_KEPLER_CHANNEL_GROUP_A', struct_alloc_object_KEPLER_CHANNEL_GROUP_A_v12_08),
    ('param_FERMI_CONTEXT_SHARE_A', struct_alloc_object_FERMI_CONTEXT_SHARE_A_v04_00),
    ('param_NVD0B7_VIDEO_ENCODER', struct_alloc_object_NVD0B7_VIDEO_ENCODER_v03_00),
    ('param_FERMI_VASPACE_A', struct_alloc_object_FERMI_VASPACE_A_v03_00),
    ('param_NVB0B0_VIDEO_DECODER', struct_alloc_object_NVB0B0_VIDEO_DECODER_v03_00),
    ('param_NV83DE_ALLOC_PARAMETERS', struct_alloc_object_NV83DE_ALLOC_PARAMETERS_v03_00),
    ('param_NVENC_SW_SESSION', struct_alloc_object_NVENC_SW_SESSION_v06_01),
    ('param_NVC4B0_VIDEO_DECODER', struct_alloc_object_NVC4B0_VIDEO_DECODER_v12_02),
    ('param_NVFBC_SW_SESSION', struct_alloc_object_NVFBC_SW_SESSION_v12_04),
    ('param_NV_NVJPG_ALLOCATION_PARAMETERS', struct_alloc_object_NV_NVJPG_ALLOCATION_PARAMETERS_v20_02),
    ('param_NV503B_ALLOC_PARAMETERS', struct_alloc_object_NV503B_ALLOC_PARAMETERS_v1D_02),
    ('param_NVC637_ALLOCATION_PARAMETERS', struct_alloc_object_NVC637_ALLOCATION_PARAMETERS_v13_00),
    ('param_NV_MEMORY_VIRTUAL_ALLOCATION_PARAMS', struct_alloc_object_NV_MEMORY_VIRTUAL_ALLOCATION_PARAMS_v13_03),
    ('param_NVC638_ALLOCATION_PARAMETERS', struct_alloc_object_NVC638_ALLOCATION_PARAMETERS_v18_06),
    ('param_NV503C_ALLOC_PARAMETERS', struct_alloc_object_NV503C_ALLOC_PARAMETERS_v18_15),
    ('param_NVC670_ALLOCATION_PARAMETERS', struct_alloc_object_NVC670_ALLOCATION_PARAMETERS_v1A_01),
    ('param_NVB1CC_ALLOC_PARAMETERS', struct_alloc_object_NVB1CC_ALLOC_PARAMETERS_v1A_03),
    ('param_NVB2CC_ALLOC_PARAMETERS', struct_alloc_object_NVB2CC_ALLOC_PARAMETERS_v1A_03),
    ('param_NV_GR_ALLOCATION_PARAMETERS', struct_NV_GR_ALLOCATION_PARAMETERS_v1A_17),
    ('param_NV_UVM_CHANNEL_RETAINER_ALLOC_PARAMS', struct_alloc_object_NV_UVM_CHANNEL_RETAINER_ALLOC_PARAMS_v1A_1B),
    ('param_NV00F8_ALLOCATION_PARAMETERS', struct_alloc_object_NV00F8_ALLOCATION_PARAMETERS_v1E_0C),
    ('param_NVC9FA_VIDEO_OFA', struct_alloc_object_NVC9FA_VIDEO_OFA_v29_06),
    ('param_NV2081_ALLOC_PARAMETERS', struct_alloc_object_NV2081_ALLOC_PARAMETERS_v25_08),
    ('param_padding', ctypes.c_ubyte * 56),
]

struct_rpc_alloc_object_v29_06._pack_ = 1 # source:False
struct_rpc_alloc_object_v29_06._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hParent', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('hClass', ctypes.c_uint32),
    ('param_len', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('params', union_alloc_object_params_v29_06),
]

rpc_alloc_object_v29_06 = struct_rpc_alloc_object_v29_06
rpc_alloc_object_v = struct_rpc_alloc_object_v29_06
class struct_rpc_free_v03_00(Structure):
    pass

class struct_NVOS00_PARAMETERS_v03_00(Structure):
    pass

struct_NVOS00_PARAMETERS_v03_00._pack_ = 1 # source:False
struct_NVOS00_PARAMETERS_v03_00._fields_ = [
    ('hRoot', ctypes.c_uint32),
    ('hObjectParent', ctypes.c_uint32),
    ('hObjectOld', ctypes.c_uint32),
    ('status', ctypes.c_uint32),
]

struct_rpc_free_v03_00._pack_ = 1 # source:False
struct_rpc_free_v03_00._fields_ = [
    ('params', struct_NVOS00_PARAMETERS_v03_00),
]

rpc_free_v03_00 = struct_rpc_free_v03_00
rpc_free_v = struct_rpc_free_v03_00
class struct_rpc_log_v03_00(Structure):
    pass

struct_rpc_log_v03_00._pack_ = 1 # source:False
struct_rpc_log_v03_00._fields_ = [
    ('level', ctypes.c_uint32),
    ('log_len', ctypes.c_uint32),
    ('log_msg', ctypes.c_char * 0),
]

rpc_log_v03_00 = struct_rpc_log_v03_00
rpc_log_v = struct_rpc_log_v03_00
class struct_rpc_map_memory_dma_v03_00(Structure):
    pass

class struct_NVOS46_PARAMETERS_v03_00(Structure):
    pass

struct_NVOS46_PARAMETERS_v03_00._pack_ = 1 # source:False
struct_NVOS46_PARAMETERS_v03_00._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hDevice', ctypes.c_uint32),
    ('hDma', ctypes.c_uint32),
    ('hMemory', ctypes.c_uint32),
    ('offset', ctypes.c_uint64),
    ('length', ctypes.c_uint64),
    ('flags', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('dmaOffset', ctypes.c_uint64),
    ('status', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

struct_rpc_map_memory_dma_v03_00._pack_ = 1 # source:False
struct_rpc_map_memory_dma_v03_00._fields_ = [
    ('params', struct_NVOS46_PARAMETERS_v03_00),
]

rpc_map_memory_dma_v03_00 = struct_rpc_map_memory_dma_v03_00
rpc_map_memory_dma_v = struct_rpc_map_memory_dma_v03_00
class struct_rpc_unmap_memory_dma_v03_00(Structure):
    pass

class struct_NVOS47_PARAMETERS_v03_00(Structure):
    pass

struct_NVOS47_PARAMETERS_v03_00._pack_ = 1 # source:False
struct_NVOS47_PARAMETERS_v03_00._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hDevice', ctypes.c_uint32),
    ('hDma', ctypes.c_uint32),
    ('hMemory', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('dmaOffset', ctypes.c_uint64),
    ('status', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

struct_rpc_unmap_memory_dma_v03_00._pack_ = 1 # source:False
struct_rpc_unmap_memory_dma_v03_00._fields_ = [
    ('params', struct_NVOS47_PARAMETERS_v03_00),
]

rpc_unmap_memory_dma_v03_00 = struct_rpc_unmap_memory_dma_v03_00
rpc_unmap_memory_dma_v = struct_rpc_unmap_memory_dma_v03_00
class struct_rpc_alloc_subdevice_v08_01(Structure):
    pass

class struct_NVOS21_PARAMETERS_v03_00(Structure):
    pass

struct_NVOS21_PARAMETERS_v03_00._pack_ = 1 # source:False
struct_NVOS21_PARAMETERS_v03_00._fields_ = [
    ('hRoot', ctypes.c_uint32),
    ('hObjectParent', ctypes.c_uint32),
    ('hObjectNew', ctypes.c_uint32),
    ('hClass', ctypes.c_uint32),
    ('pAllocParms', ctypes.POINTER(None)),
    ('status', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

struct_rpc_alloc_subdevice_v08_01._pack_ = 1 # source:False
struct_rpc_alloc_subdevice_v08_01._fields_ = [
    ('subDeviceInst', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('params', struct_NVOS21_PARAMETERS_v03_00),
]

rpc_alloc_subdevice_v08_01 = struct_rpc_alloc_subdevice_v08_01
rpc_alloc_subdevice_v = struct_rpc_alloc_subdevice_v08_01
class struct_rpc_dup_object_v03_00(Structure):
    pass

class struct_NVOS55_PARAMETERS_v03_00(Structure):
    pass

struct_NVOS55_PARAMETERS_v03_00._pack_ = 1 # source:False
struct_NVOS55_PARAMETERS_v03_00._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hParent', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('hClientSrc', ctypes.c_uint32),
    ('hObjectSrc', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('status', ctypes.c_uint32),
]

struct_rpc_dup_object_v03_00._pack_ = 1 # source:False
struct_rpc_dup_object_v03_00._fields_ = [
    ('params', struct_NVOS55_PARAMETERS_v03_00),
]

rpc_dup_object_v03_00 = struct_rpc_dup_object_v03_00
rpc_dup_object_v = struct_rpc_dup_object_v03_00
class struct_rpc_idle_channels_v03_00(Structure):
    pass

class struct_idle_channel_list_v03_00(Structure):
    pass

struct_idle_channel_list_v03_00._pack_ = 1 # source:False
struct_idle_channel_list_v03_00._fields_ = [
    ('phClient', ctypes.c_uint32),
    ('phDevice', ctypes.c_uint32),
    ('phChannel', ctypes.c_uint32),
]

struct_rpc_idle_channels_v03_00._pack_ = 1 # source:False
struct_rpc_idle_channels_v03_00._fields_ = [
    ('flags', ctypes.c_uint32),
    ('timeout', ctypes.c_uint32),
    ('nchannels', ctypes.c_uint32),
    ('channel_list', struct_idle_channel_list_v03_00 * 0),
]

rpc_idle_channels_v03_00 = struct_rpc_idle_channels_v03_00
rpc_idle_channels_v = struct_rpc_idle_channels_v03_00
class struct_rpc_alloc_event_v03_00(Structure):
    pass

struct_rpc_alloc_event_v03_00._pack_ = 1 # source:False
struct_rpc_alloc_event_v03_00._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hParentClient', ctypes.c_uint32),
    ('hChannel', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('hEvent', ctypes.c_uint32),
    ('hClass', ctypes.c_uint32),
    ('notifyIndex', ctypes.c_uint32),
]

rpc_alloc_event_v03_00 = struct_rpc_alloc_event_v03_00
rpc_alloc_event_v = struct_rpc_alloc_event_v03_00
class struct_rpc_rm_api_control_v25_0D(Structure):
    pass

class struct_NVOS54_PARAMETERS_v03_00(Structure):
    pass

struct_NVOS54_PARAMETERS_v03_00._pack_ = 1 # source:False
struct_NVOS54_PARAMETERS_v03_00._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('cmd', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('params', ctypes.POINTER(None)),
    ('paramsSize', ctypes.c_uint32),
    ('status', ctypes.c_uint32),
]

struct_rpc_rm_api_control_v25_0D._pack_ = 1 # source:False
struct_rpc_rm_api_control_v25_0D._fields_ = [
    ('params', struct_NVOS54_PARAMETERS_v03_00),
    ('rm_api_params', ctypes.POINTER(None)),
]

rpc_rm_api_control_v25_0D = struct_rpc_rm_api_control_v25_0D
class struct_rpc_rm_api_control_v25_0F(Structure):
    pass

struct_rpc_rm_api_control_v25_0F._pack_ = 1 # source:False
struct_rpc_rm_api_control_v25_0F._fields_ = [
    ('params', struct_NVOS54_PARAMETERS_v03_00),
    ('rm_api_params', ctypes.POINTER(None)),
]

rpc_rm_api_control_v25_0F = struct_rpc_rm_api_control_v25_0F
class struct_rpc_rm_api_control_v25_10(Structure):
    pass

struct_rpc_rm_api_control_v25_10._pack_ = 1 # source:False
struct_rpc_rm_api_control_v25_10._fields_ = [
    ('params', struct_NVOS54_PARAMETERS_v03_00),
    ('rm_api_params', ctypes.POINTER(None)),
]

rpc_rm_api_control_v25_10 = struct_rpc_rm_api_control_v25_10
class struct_rpc_rm_api_control_v25_14(Structure):
    pass

struct_rpc_rm_api_control_v25_14._pack_ = 1 # source:False
struct_rpc_rm_api_control_v25_14._fields_ = [
    ('params', struct_NVOS54_PARAMETERS_v03_00),
    ('rm_api_params', ctypes.POINTER(None)),
]

rpc_rm_api_control_v25_14 = struct_rpc_rm_api_control_v25_14
class struct_rpc_rm_api_control_v25_15(Structure):
    pass

struct_rpc_rm_api_control_v25_15._pack_ = 1 # source:False
struct_rpc_rm_api_control_v25_15._fields_ = [
    ('params', struct_NVOS54_PARAMETERS_v03_00),
    ('rm_api_params', ctypes.POINTER(None)),
]

rpc_rm_api_control_v25_15 = struct_rpc_rm_api_control_v25_15
class struct_rpc_rm_api_control_v25_16(Structure):
    pass

struct_rpc_rm_api_control_v25_16._pack_ = 1 # source:False
struct_rpc_rm_api_control_v25_16._fields_ = [
    ('params', struct_NVOS54_PARAMETERS_v03_00),
    ('rm_api_params', ctypes.POINTER(None)),
]

rpc_rm_api_control_v25_16 = struct_rpc_rm_api_control_v25_16
class struct_rpc_rm_api_control_v25_17(Structure):
    pass

struct_rpc_rm_api_control_v25_17._pack_ = 1 # source:False
struct_rpc_rm_api_control_v25_17._fields_ = [
    ('params', struct_NVOS54_PARAMETERS_v03_00),
    ('rm_api_params', ctypes.POINTER(None)),
]

rpc_rm_api_control_v25_17 = struct_rpc_rm_api_control_v25_17
class struct_rpc_rm_api_control_v25_18(Structure):
    pass

struct_rpc_rm_api_control_v25_18._pack_ = 1 # source:False
struct_rpc_rm_api_control_v25_18._fields_ = [
    ('params', struct_NVOS54_PARAMETERS_v03_00),
    ('rm_api_params', ctypes.POINTER(None)),
]

rpc_rm_api_control_v25_18 = struct_rpc_rm_api_control_v25_18
class struct_rpc_rm_api_control_v25_19(Structure):
    pass

struct_rpc_rm_api_control_v25_19._pack_ = 1 # source:False
struct_rpc_rm_api_control_v25_19._fields_ = [
    ('params', struct_NVOS54_PARAMETERS_v03_00),
    ('rm_api_params', ctypes.POINTER(None)),
]

rpc_rm_api_control_v25_19 = struct_rpc_rm_api_control_v25_19
class struct_rpc_rm_api_control_v25_1A(Structure):
    pass

struct_rpc_rm_api_control_v25_1A._pack_ = 1 # source:False
struct_rpc_rm_api_control_v25_1A._fields_ = [
    ('params', struct_NVOS54_PARAMETERS_v03_00),
    ('rm_api_params', ctypes.POINTER(None)),
]

rpc_rm_api_control_v25_1A = struct_rpc_rm_api_control_v25_1A
class struct_rpc_rm_api_control_v27_03(Structure):
    pass

struct_rpc_rm_api_control_v27_03._pack_ = 1 # source:False
struct_rpc_rm_api_control_v27_03._fields_ = [
    ('params', struct_NVOS54_PARAMETERS_v03_00),
    ('rm_api_params', ctypes.POINTER(None)),
]

rpc_rm_api_control_v27_03 = struct_rpc_rm_api_control_v27_03
class struct_rpc_rm_api_control_v29_04(Structure):
    pass

struct_rpc_rm_api_control_v29_04._pack_ = 1 # source:False
struct_rpc_rm_api_control_v29_04._fields_ = [
    ('params', struct_NVOS54_PARAMETERS_v03_00),
    ('rm_api_params', ctypes.POINTER(None)),
]

rpc_rm_api_control_v29_04 = struct_rpc_rm_api_control_v29_04
class struct_rpc_rm_api_control_v29_09(Structure):
    pass

struct_rpc_rm_api_control_v29_09._pack_ = 1 # source:False
struct_rpc_rm_api_control_v29_09._fields_ = [
    ('params', struct_NVOS54_PARAMETERS_v03_00),
    ('rm_api_params', ctypes.POINTER(None)),
]

rpc_rm_api_control_v29_09 = struct_rpc_rm_api_control_v29_09
rpc_rm_api_control_v = struct_rpc_rm_api_control_v29_09
class struct_rpc_alloc_share_device_v03_00(Structure):
    pass

class struct_NV_DEVICE_ALLOCATION_PARAMETERS_v03_00(Structure):
    pass

struct_NV_DEVICE_ALLOCATION_PARAMETERS_v03_00._pack_ = 1 # source:False
struct_NV_DEVICE_ALLOCATION_PARAMETERS_v03_00._fields_ = [
    ('szName', ctypes.POINTER(None)),
    ('hClientShare', ctypes.c_uint32),
    ('hTargetClient', ctypes.c_uint32),
    ('hTargetDevice', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('vaSpaceSize', ctypes.c_uint64),
    ('vaMode', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('vaBase', ctypes.c_uint64),
]

struct_rpc_alloc_share_device_v03_00._pack_ = 1 # source:False
struct_rpc_alloc_share_device_v03_00._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hDevice', ctypes.c_uint32),
    ('hClass', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('params', struct_NV_DEVICE_ALLOCATION_PARAMETERS_v03_00),
]

rpc_alloc_share_device_v03_00 = struct_rpc_alloc_share_device_v03_00
rpc_alloc_share_device_v = struct_rpc_alloc_share_device_v03_00
class struct_rpc_get_engine_utilization_v1F_0E(Structure):
    pass

class union_vgpuGetEngineUtilization_data_v1F_0E(Union):
    pass

class struct_NV2080_CTRL_PERF_GET_VID_ENG_PERFMON_SAMPLE_PARAMS_v05_00(Structure):
    pass


# values for enumeration 'NV2080_CTRL_CMD_PERF_VID_ENG'
NV2080_CTRL_CMD_PERF_VID_ENG__enumvalues = {
    1: 'NV2080_CTRL_CMD_PERF_VID_ENG_NVENC',
    2: 'NV2080_CTRL_CMD_PERF_VID_ENG_NVDEC',
    3: 'NV2080_CTRL_CMD_PERF_VID_ENG_NVJPG',
    4: 'NV2080_CTRL_CMD_PERF_VID_ENG_NVOFA',
}
NV2080_CTRL_CMD_PERF_VID_ENG_NVENC = 1
NV2080_CTRL_CMD_PERF_VID_ENG_NVDEC = 2
NV2080_CTRL_CMD_PERF_VID_ENG_NVJPG = 3
NV2080_CTRL_CMD_PERF_VID_ENG_NVOFA = 4
NV2080_CTRL_CMD_PERF_VID_ENG = ctypes.c_uint32 # enum
struct_NV2080_CTRL_PERF_GET_VID_ENG_PERFMON_SAMPLE_PARAMS_v05_00._pack_ = 1 # source:False
struct_NV2080_CTRL_PERF_GET_VID_ENG_PERFMON_SAMPLE_PARAMS_v05_00._fields_ = [
    ('engineType', NV2080_CTRL_CMD_PERF_VID_ENG),
    ('clkPercentBusy', ctypes.c_uint32),
    ('samplingPeriodUs', ctypes.c_uint32),
]

class struct_NV0000_CTRL_GPUACCT_GET_ACCOUNTING_STATE_PARAMS_v09_0C(Structure):
    pass

struct_NV0000_CTRL_GPUACCT_GET_ACCOUNTING_STATE_PARAMS_v09_0C._pack_ = 1 # source:False
struct_NV0000_CTRL_GPUACCT_GET_ACCOUNTING_STATE_PARAMS_v09_0C._fields_ = [
    ('gpuId', ctypes.c_uint32),
    ('vmPid', ctypes.c_uint32),
    ('state', ctypes.c_uint32),
]

class struct_NV0000_CTRL_GPUACCT_SET_ACCOUNTING_STATE_PARAMS_v09_0C(Structure):
    pass

struct_NV0000_CTRL_GPUACCT_SET_ACCOUNTING_STATE_PARAMS_v09_0C._pack_ = 1 # source:False
struct_NV0000_CTRL_GPUACCT_SET_ACCOUNTING_STATE_PARAMS_v09_0C._fields_ = [
    ('gpuId', ctypes.c_uint32),
    ('vmPid', ctypes.c_uint32),
    ('newState', ctypes.c_uint32),
]

class struct_NV0000_CTRL_GPUACCT_GET_ACCOUNTING_PIDS_PARAMS_v09_0C(Structure):
    pass

struct_NV0000_CTRL_GPUACCT_GET_ACCOUNTING_PIDS_PARAMS_v09_0C._pack_ = 1 # source:False
struct_NV0000_CTRL_GPUACCT_GET_ACCOUNTING_PIDS_PARAMS_v09_0C._fields_ = [
    ('gpuId', ctypes.c_uint32),
    ('vmPid', ctypes.c_uint32),
    ('passIndex', ctypes.c_uint32),
    ('pidCount', ctypes.c_uint32),
    ('pidTable', ctypes.c_uint32 * 1000),
]

class struct_NV0000_CTRL_GPUACCT_GET_PROC_ACCOUNTING_INFO_PARAMS_v09_0C(Structure):
    pass

struct_NV0000_CTRL_GPUACCT_GET_PROC_ACCOUNTING_INFO_PARAMS_v09_0C._pack_ = 1 # source:False
struct_NV0000_CTRL_GPUACCT_GET_PROC_ACCOUNTING_INFO_PARAMS_v09_0C._fields_ = [
    ('gpuId', ctypes.c_uint32),
    ('pid', ctypes.c_uint32),
    ('subPid', ctypes.c_uint32),
    ('gpuUtil', ctypes.c_uint32),
    ('fbUtil', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('maxFbUsage', ctypes.c_uint64),
    ('startTime', ctypes.c_uint64),
    ('endTime', ctypes.c_uint64),
]

class struct_NV0000_CTRL_GPUACCT_CLEAR_ACCOUNTING_DATA_PARAMS_v09_0C(Structure):
    pass

struct_NV0000_CTRL_GPUACCT_CLEAR_ACCOUNTING_DATA_PARAMS_v09_0C._pack_ = 1 # source:False
struct_NV0000_CTRL_GPUACCT_CLEAR_ACCOUNTING_DATA_PARAMS_v09_0C._fields_ = [
    ('gpuId', ctypes.c_uint32),
    ('vmPid', ctypes.c_uint32),
]

class struct_NV2080_CTRL_PERF_GPUMON_PERFMON_UTIL_SAMPLE_v1F_0E(Structure):
    pass

class struct_NV2080_CTRL_PERF_GPUMON_ENGINE_UTIL_SAMPLE_v17_00(Structure):
    pass

struct_NV2080_CTRL_PERF_GPUMON_ENGINE_UTIL_SAMPLE_v17_00._pack_ = 1 # source:False
struct_NV2080_CTRL_PERF_GPUMON_ENGINE_UTIL_SAMPLE_v17_00._fields_ = [
    ('util', ctypes.c_uint32),
    ('procId', ctypes.c_uint32),
    ('subProcessID', ctypes.c_uint32),
]

struct_NV2080_CTRL_PERF_GPUMON_PERFMON_UTIL_SAMPLE_v1F_0E._pack_ = 1 # source:False
struct_NV2080_CTRL_PERF_GPUMON_PERFMON_UTIL_SAMPLE_v1F_0E._fields_ = [
    ('timeStamp', ctypes.c_uint64),
    ('fb', struct_NV2080_CTRL_PERF_GPUMON_ENGINE_UTIL_SAMPLE_v17_00),
    ('gr', struct_NV2080_CTRL_PERF_GPUMON_ENGINE_UTIL_SAMPLE_v17_00),
    ('nvenc', struct_NV2080_CTRL_PERF_GPUMON_ENGINE_UTIL_SAMPLE_v17_00),
    ('nvdec', struct_NV2080_CTRL_PERF_GPUMON_ENGINE_UTIL_SAMPLE_v17_00),
]

union_vgpuGetEngineUtilization_data_v1F_0E._pack_ = 1 # source:False
union_vgpuGetEngineUtilization_data_v1F_0E._fields_ = [
    ('vidPerfmonSample', struct_NV2080_CTRL_PERF_GET_VID_ENG_PERFMON_SAMPLE_PARAMS_v05_00),
    ('getAccountingState', struct_NV0000_CTRL_GPUACCT_GET_ACCOUNTING_STATE_PARAMS_v09_0C),
    ('setAccountingState', struct_NV0000_CTRL_GPUACCT_SET_ACCOUNTING_STATE_PARAMS_v09_0C),
    ('getAccountingPidList', struct_NV0000_CTRL_GPUACCT_GET_ACCOUNTING_PIDS_PARAMS_v09_0C),
    ('procAccountingInfo', struct_NV0000_CTRL_GPUACCT_GET_PROC_ACCOUNTING_INFO_PARAMS_v09_0C),
    ('clearAccountingInfo', struct_NV0000_CTRL_GPUACCT_CLEAR_ACCOUNTING_DATA_PARAMS_v09_0C),
    ('gpumonPerfmonsampleV2', struct_NV2080_CTRL_PERF_GPUMON_PERFMON_UTIL_SAMPLE_v1F_0E * 72),
]

struct_rpc_get_engine_utilization_v1F_0E._pack_ = 1 # source:False
struct_rpc_get_engine_utilization_v1F_0E._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('cmd', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('params', union_vgpuGetEngineUtilization_data_v1F_0E),
]

rpc_get_engine_utilization_v1F_0E = struct_rpc_get_engine_utilization_v1F_0E
rpc_get_engine_utilization_v = struct_rpc_get_engine_utilization_v1F_0E
class struct_rpc_perf_get_level_info_v03_00(Structure):
    pass

struct_rpc_perf_get_level_info_v03_00._pack_ = 1 # source:False
struct_rpc_perf_get_level_info_v03_00._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('level', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('perfGetClkInfoListSize', ctypes.c_uint32),
    ('param_size', ctypes.c_uint32),
    ('params', ctypes.c_uint32 * 0),
]

rpc_perf_get_level_info_v03_00 = struct_rpc_perf_get_level_info_v03_00
rpc_perf_get_level_info_v = struct_rpc_perf_get_level_info_v03_00
class struct_rpc_set_surface_properties_v07_07(Structure):
    pass

class struct_NVA080_CTRL_VGPU_DISPLAY_SET_SURFACE_PROPERTIES_v07_07(Structure):
    pass

struct_NVA080_CTRL_VGPU_DISPLAY_SET_SURFACE_PROPERTIES_v07_07._pack_ = 1 # source:False
struct_NVA080_CTRL_VGPU_DISPLAY_SET_SURFACE_PROPERTIES_v07_07._fields_ = [
    ('headIndex', ctypes.c_uint32),
    ('isPrimary', ctypes.c_uint32),
    ('offset', ctypes.c_uint32),
    ('surfaceType', ctypes.c_uint32),
    ('surfaceBlockHeight', ctypes.c_uint32),
    ('surfacePitch', ctypes.c_uint32),
    ('surfaceFormat', ctypes.c_uint32),
    ('surfaceWidth', ctypes.c_uint32),
    ('surfaceHeight', ctypes.c_uint32),
    ('rectX', ctypes.c_uint32),
    ('rectY', ctypes.c_uint32),
    ('rectWidth', ctypes.c_uint32),
    ('rectHeight', ctypes.c_uint32),
    ('surfaceSize', ctypes.c_uint32),
    ('surfaceKind', ctypes.c_uint32),
    ('hHwResDevice', ctypes.c_uint32),
    ('hHwResHandle', ctypes.c_uint32),
    ('effectiveFbPageSize', ctypes.c_uint32),
]

struct_rpc_set_surface_properties_v07_07._pack_ = 1 # source:False
struct_rpc_set_surface_properties_v07_07._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('params', struct_NVA080_CTRL_VGPU_DISPLAY_SET_SURFACE_PROPERTIES_v07_07),
]

rpc_set_surface_properties_v07_07 = struct_rpc_set_surface_properties_v07_07
rpc_set_surface_properties_v = struct_rpc_set_surface_properties_v07_07
class struct_rpc_cleanup_surface_v03_00(Structure):
    pass

class struct_NVA080_CTRL_VGPU_DISPLAY_CLEANUP_SURFACE_PARAMS_v03_00(Structure):
    pass

struct_NVA080_CTRL_VGPU_DISPLAY_CLEANUP_SURFACE_PARAMS_v03_00._pack_ = 1 # source:False
struct_NVA080_CTRL_VGPU_DISPLAY_CLEANUP_SURFACE_PARAMS_v03_00._fields_ = [
    ('headIndex', ctypes.c_uint32),
    ('blankingEnabled', ctypes.c_uint32),
]

struct_rpc_cleanup_surface_v03_00._pack_ = 1 # source:False
struct_rpc_cleanup_surface_v03_00._fields_ = [
    ('params', struct_NVA080_CTRL_VGPU_DISPLAY_CLEANUP_SURFACE_PARAMS_v03_00),
]

rpc_cleanup_surface_v03_00 = struct_rpc_cleanup_surface_v03_00
rpc_cleanup_surface_v = struct_rpc_cleanup_surface_v03_00
class struct_rpc_unloading_guest_driver_v1F_07(Structure):
    pass

struct_rpc_unloading_guest_driver_v1F_07._pack_ = 1 # source:False
struct_rpc_unloading_guest_driver_v1F_07._fields_ = [
    ('bInPMTransition', ctypes.c_ubyte),
    ('bGc6Entering', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 2),
    ('newLevel', ctypes.c_uint32),
]

rpc_unloading_guest_driver_v1F_07 = struct_rpc_unloading_guest_driver_v1F_07
rpc_unloading_guest_driver_v = struct_rpc_unloading_guest_driver_v1F_07
class struct_rpc_gpu_exec_reg_ops_v12_01(Structure):
    pass

class struct_gpu_exec_reg_ops_v12_01(Structure):
    pass

class struct_NV2080_CTRL_GPU_EXEC_REG_OPS_PARAMS_v12_01(Structure):
    pass

class struct_NV2080_CTRL_GR_ROUTE_INFO_v12_01(Structure):
    pass

struct_NV2080_CTRL_GR_ROUTE_INFO_v12_01._pack_ = 1 # source:False
struct_NV2080_CTRL_GR_ROUTE_INFO_v12_01._fields_ = [
    ('flags', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('route', ctypes.c_uint64),
]

struct_NV2080_CTRL_GPU_EXEC_REG_OPS_PARAMS_v12_01._pack_ = 1 # source:False
struct_NV2080_CTRL_GPU_EXEC_REG_OPS_PARAMS_v12_01._fields_ = [
    ('hClientTarget', ctypes.c_uint32),
    ('hChannelTarget', ctypes.c_uint32),
    ('reserved00', ctypes.c_uint32 * 3),
    ('regOpCount', ctypes.c_uint32),
    ('grRouteInfo', struct_NV2080_CTRL_GR_ROUTE_INFO_v12_01),
    ('regOps', ctypes.POINTER(None)),
]

class struct_NV2080_CTRL_GPU_REG_OP_v03_00(Structure):
    pass

struct_NV2080_CTRL_GPU_REG_OP_v03_00._pack_ = 1 # source:False
struct_NV2080_CTRL_GPU_REG_OP_v03_00._fields_ = [
    ('regOp', ctypes.c_ubyte),
    ('regType', ctypes.c_ubyte),
    ('regStatus', ctypes.c_ubyte),
    ('regQuad', ctypes.c_ubyte),
    ('regGroupMask', ctypes.c_uint32),
    ('regSubGroupMask', ctypes.c_uint32),
    ('regOffset', ctypes.c_uint32),
    ('regValueHi', ctypes.c_uint32),
    ('regValueLo', ctypes.c_uint32),
    ('regAndNMaskHi', ctypes.c_uint32),
    ('regAndNMaskLo', ctypes.c_uint32),
]

struct_gpu_exec_reg_ops_v12_01._pack_ = 1 # source:False
struct_gpu_exec_reg_ops_v12_01._fields_ = [
    ('reg_op_params', struct_NV2080_CTRL_GPU_EXEC_REG_OPS_PARAMS_v12_01),
    ('operations', struct_NV2080_CTRL_GPU_REG_OP_v03_00 * 0),
]

struct_rpc_gpu_exec_reg_ops_v12_01._pack_ = 1 # source:False
struct_rpc_gpu_exec_reg_ops_v12_01._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('params', struct_gpu_exec_reg_ops_v12_01),
]

rpc_gpu_exec_reg_ops_v12_01 = struct_rpc_gpu_exec_reg_ops_v12_01
rpc_gpu_exec_reg_ops_v = struct_rpc_gpu_exec_reg_ops_v12_01
class struct_rpc_get_static_data_v25_0E(Structure):
    pass

struct_rpc_get_static_data_v25_0E._pack_ = 1 # source:False
struct_rpc_get_static_data_v25_0E._fields_ = [
    ('offset', ctypes.c_uint32),
    ('size', ctypes.c_uint32),
    ('payload', ctypes.c_ubyte * 0),
]

rpc_get_static_data_v25_0E = struct_rpc_get_static_data_v25_0E
class struct_rpc_get_static_data_v27_01(Structure):
    pass

struct_rpc_get_static_data_v27_01._pack_ = 1 # source:False
struct_rpc_get_static_data_v27_01._fields_ = [
    ('offset', ctypes.c_uint32),
    ('size', ctypes.c_uint32),
    ('payload', ctypes.c_ubyte * 0),
]

rpc_get_static_data_v27_01 = struct_rpc_get_static_data_v27_01
rpc_get_static_data_v = struct_rpc_get_static_data_v27_01
class struct_rpc_get_consolidated_gr_static_info_v1B_04(Structure):
    pass

struct_rpc_get_consolidated_gr_static_info_v1B_04._pack_ = 1 # source:False
struct_rpc_get_consolidated_gr_static_info_v1B_04._fields_ = [
    ('offset', ctypes.c_uint32),
    ('size', ctypes.c_uint32),
    ('payload', ctypes.c_ubyte * 0),
]

rpc_get_consolidated_gr_static_info_v1B_04 = struct_rpc_get_consolidated_gr_static_info_v1B_04
rpc_get_consolidated_gr_static_info_v = struct_rpc_get_consolidated_gr_static_info_v1B_04
class struct_rpc_set_page_directory_v1E_05(Structure):
    pass

class struct_NV0080_CTRL_DMA_SET_PAGE_DIRECTORY_PARAMS_v1E_05(Structure):
    pass

struct_NV0080_CTRL_DMA_SET_PAGE_DIRECTORY_PARAMS_v1E_05._pack_ = 1 # source:False
struct_NV0080_CTRL_DMA_SET_PAGE_DIRECTORY_PARAMS_v1E_05._fields_ = [
    ('physAddress', ctypes.c_uint64),
    ('numEntries', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('hVASpace', ctypes.c_uint32),
    ('chId', ctypes.c_uint32),
    ('subDeviceId', ctypes.c_uint32),
    ('pasid', ctypes.c_uint32),
]

struct_rpc_set_page_directory_v1E_05._pack_ = 1 # source:False
struct_rpc_set_page_directory_v1E_05._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hDevice', ctypes.c_uint32),
    ('pasid', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('params', struct_NV0080_CTRL_DMA_SET_PAGE_DIRECTORY_PARAMS_v1E_05),
]

rpc_set_page_directory_v1E_05 = struct_rpc_set_page_directory_v1E_05
rpc_set_page_directory_v = struct_rpc_set_page_directory_v1E_05
class struct_rpc_unset_page_directory_v1E_05(Structure):
    pass

class struct_NV0080_CTRL_DMA_UNSET_PAGE_DIRECTORY_PARAMS_v1E_05(Structure):
    pass

struct_NV0080_CTRL_DMA_UNSET_PAGE_DIRECTORY_PARAMS_v1E_05._pack_ = 1 # source:False
struct_NV0080_CTRL_DMA_UNSET_PAGE_DIRECTORY_PARAMS_v1E_05._fields_ = [
    ('hVASpace', ctypes.c_uint32),
    ('subDeviceId', ctypes.c_uint32),
]

struct_rpc_unset_page_directory_v1E_05._pack_ = 1 # source:False
struct_rpc_unset_page_directory_v1E_05._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hDevice', ctypes.c_uint32),
    ('params', struct_NV0080_CTRL_DMA_UNSET_PAGE_DIRECTORY_PARAMS_v1E_05),
]

rpc_unset_page_directory_v1E_05 = struct_rpc_unset_page_directory_v1E_05
rpc_unset_page_directory_v = struct_rpc_unset_page_directory_v1E_05
class struct_rpc_get_gsp_static_info_v14_00(Structure):
    pass

struct_rpc_get_gsp_static_info_v14_00._pack_ = 1 # source:False
struct_rpc_get_gsp_static_info_v14_00._fields_ = [
    ('data', ctypes.c_uint32),
]

rpc_get_gsp_static_info_v14_00 = struct_rpc_get_gsp_static_info_v14_00
rpc_get_gsp_static_info_v = struct_rpc_get_gsp_static_info_v14_00
class struct_rpc_update_bar_pde_v15_00(Structure):
    pass

class struct_UpdateBarPde_v15_00(Structure):
    pass

struct_UpdateBarPde_v15_00._pack_ = 1 # source:False
struct_UpdateBarPde_v15_00._fields_ = [
    ('barType', NV_RPC_UPDATE_PDE_BAR_TYPE),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('entryValue', ctypes.c_uint64),
    ('entryLevelShift', ctypes.c_uint64),
]

struct_rpc_update_bar_pde_v15_00._pack_ = 1 # source:False
struct_rpc_update_bar_pde_v15_00._fields_ = [
    ('info', struct_UpdateBarPde_v15_00),
]

rpc_update_bar_pde_v15_00 = struct_rpc_update_bar_pde_v15_00
rpc_update_bar_pde_v = struct_rpc_update_bar_pde_v15_00
class struct_rpc_get_encoder_capacity_v07_00(Structure):
    pass

struct_rpc_get_encoder_capacity_v07_00._pack_ = 1 # source:False
struct_rpc_get_encoder_capacity_v07_00._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('encoderCapacity', ctypes.c_uint32),
]

rpc_get_encoder_capacity_v07_00 = struct_rpc_get_encoder_capacity_v07_00
rpc_get_encoder_capacity_v = struct_rpc_get_encoder_capacity_v07_00
class struct_rpc_vgpu_pf_reg_read32_v15_00(Structure):
    pass

struct_rpc_vgpu_pf_reg_read32_v15_00._pack_ = 1 # source:False
struct_rpc_vgpu_pf_reg_read32_v15_00._fields_ = [
    ('address', ctypes.c_uint64),
    ('value', ctypes.c_uint32),
    ('grEngId', ctypes.c_uint32),
]

rpc_vgpu_pf_reg_read32_v15_00 = struct_rpc_vgpu_pf_reg_read32_v15_00
rpc_vgpu_pf_reg_read32_v = struct_rpc_vgpu_pf_reg_read32_v15_00
class struct_rpc_ctrl_set_vgpu_fb_usage_v1A_08(Structure):
    pass

class struct_NVA080_CTRL_SET_FB_USAGE_PARAMS_v07_02(Structure):
    pass

struct_NVA080_CTRL_SET_FB_USAGE_PARAMS_v07_02._pack_ = 1 # source:False
struct_NVA080_CTRL_SET_FB_USAGE_PARAMS_v07_02._fields_ = [
    ('fbUsed', ctypes.c_uint64),
]

struct_rpc_ctrl_set_vgpu_fb_usage_v1A_08._pack_ = 1 # source:False
struct_rpc_ctrl_set_vgpu_fb_usage_v1A_08._fields_ = [
    ('setFbUsage', struct_NVA080_CTRL_SET_FB_USAGE_PARAMS_v07_02),
]

rpc_ctrl_set_vgpu_fb_usage_v1A_08 = struct_rpc_ctrl_set_vgpu_fb_usage_v1A_08
rpc_ctrl_set_vgpu_fb_usage_v = struct_rpc_ctrl_set_vgpu_fb_usage_v1A_08
class struct_rpc_ctrl_nvenc_sw_session_update_info_v1A_09(Structure):
    pass

class struct_NVA0BC_CTRL_NVENC_SW_SESSION_UPDATE_INFO_PARAMS_v06_01(Structure):
    pass

struct_NVA0BC_CTRL_NVENC_SW_SESSION_UPDATE_INFO_PARAMS_v06_01._pack_ = 1 # source:False
struct_NVA0BC_CTRL_NVENC_SW_SESSION_UPDATE_INFO_PARAMS_v06_01._fields_ = [
    ('hResolution', ctypes.c_uint32),
    ('vResolution', ctypes.c_uint32),
    ('averageEncodeLatency', ctypes.c_uint32),
    ('averageEncodeFps', ctypes.c_uint32),
    ('timestampBufferSize', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('timestampBuffer', ctypes.POINTER(None)),
]

struct_rpc_ctrl_nvenc_sw_session_update_info_v1A_09._pack_ = 1 # source:False
struct_rpc_ctrl_nvenc_sw_session_update_info_v1A_09._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('nvencSessionUpdate', struct_NVA0BC_CTRL_NVENC_SW_SESSION_UPDATE_INFO_PARAMS_v06_01),
]

rpc_ctrl_nvenc_sw_session_update_info_v1A_09 = struct_rpc_ctrl_nvenc_sw_session_update_info_v1A_09
rpc_ctrl_nvenc_sw_session_update_info_v = struct_rpc_ctrl_nvenc_sw_session_update_info_v1A_09
class struct_rpc_ctrl_reset_channel_v1A_09(Structure):
    pass

class struct_NV906F_CTRL_CMD_RESET_CHANNEL_PARAMS_v10_01(Structure):
    pass

struct_NV906F_CTRL_CMD_RESET_CHANNEL_PARAMS_v10_01._pack_ = 1 # source:False
struct_NV906F_CTRL_CMD_RESET_CHANNEL_PARAMS_v10_01._fields_ = [
    ('engineID', ctypes.c_uint32),
    ('subdeviceInstance', ctypes.c_uint32),
    ('resetReason', ctypes.c_uint32),
]

struct_rpc_ctrl_reset_channel_v1A_09._pack_ = 1 # source:False
struct_rpc_ctrl_reset_channel_v1A_09._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('resetChannel', struct_NV906F_CTRL_CMD_RESET_CHANNEL_PARAMS_v10_01),
]

rpc_ctrl_reset_channel_v1A_09 = struct_rpc_ctrl_reset_channel_v1A_09
rpc_ctrl_reset_channel_v = struct_rpc_ctrl_reset_channel_v1A_09
class struct_rpc_ctrl_reset_isolated_channel_v1A_09(Structure):
    pass

class struct_NV506F_CTRL_CMD_RESET_ISOLATED_CHANNEL_PARAMS_v03_00(Structure):
    pass

struct_NV506F_CTRL_CMD_RESET_ISOLATED_CHANNEL_PARAMS_v03_00._pack_ = 1 # source:False
struct_NV506F_CTRL_CMD_RESET_ISOLATED_CHANNEL_PARAMS_v03_00._fields_ = [
    ('exceptType', ctypes.c_uint32),
    ('engineID', ctypes.c_uint32),
]

struct_rpc_ctrl_reset_isolated_channel_v1A_09._pack_ = 1 # source:False
struct_rpc_ctrl_reset_isolated_channel_v1A_09._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('resetIsolatedChannel', struct_NV506F_CTRL_CMD_RESET_ISOLATED_CHANNEL_PARAMS_v03_00),
]

rpc_ctrl_reset_isolated_channel_v1A_09 = struct_rpc_ctrl_reset_isolated_channel_v1A_09
rpc_ctrl_reset_isolated_channel_v = struct_rpc_ctrl_reset_isolated_channel_v1A_09
class struct_rpc_ctrl_gpu_handle_vf_pri_fault_v1A_09(Structure):
    pass

class struct_NV2080_CTRL_CMD_GPU_HANDLE_VF_PRI_FAULT_PARAMS_v18_09(Structure):
    pass

struct_NV2080_CTRL_CMD_GPU_HANDLE_VF_PRI_FAULT_PARAMS_v18_09._pack_ = 1 # source:False
struct_NV2080_CTRL_CMD_GPU_HANDLE_VF_PRI_FAULT_PARAMS_v18_09._fields_ = [
    ('faultType', ctypes.c_uint32),
]

struct_rpc_ctrl_gpu_handle_vf_pri_fault_v1A_09._pack_ = 1 # source:False
struct_rpc_ctrl_gpu_handle_vf_pri_fault_v1A_09._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('handleVfPriFault', struct_NV2080_CTRL_CMD_GPU_HANDLE_VF_PRI_FAULT_PARAMS_v18_09),
]

rpc_ctrl_gpu_handle_vf_pri_fault_v1A_09 = struct_rpc_ctrl_gpu_handle_vf_pri_fault_v1A_09
rpc_ctrl_gpu_handle_vf_pri_fault_v = struct_rpc_ctrl_gpu_handle_vf_pri_fault_v1A_09
class struct_rpc_ctrl_perf_boost_v1A_09(Structure):
    pass

class struct_NV2080_CTRL_PERF_BOOST_PARAMS_v03_00(Structure):
    pass

struct_NV2080_CTRL_PERF_BOOST_PARAMS_v03_00._pack_ = 1 # source:False
struct_NV2080_CTRL_PERF_BOOST_PARAMS_v03_00._fields_ = [
    ('flags', ctypes.c_uint32),
    ('duration', ctypes.c_uint32),
]

struct_rpc_ctrl_perf_boost_v1A_09._pack_ = 1 # source:False
struct_rpc_ctrl_perf_boost_v1A_09._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('perfBoost', struct_NV2080_CTRL_PERF_BOOST_PARAMS_v03_00),
]

rpc_ctrl_perf_boost_v1A_09 = struct_rpc_ctrl_perf_boost_v1A_09
rpc_ctrl_perf_boost_v = struct_rpc_ctrl_perf_boost_v1A_09
class struct_rpc_ctrl_get_zbc_clear_table_v1A_09(Structure):
    pass

class struct_NV9096_CTRL_GET_ZBC_CLEAR_TABLE_PARAMS_v04_00(Structure):
    pass

class struct_NV9096_CTRL_GET_ZBC_CLEAR_TABLE_PARAMS_value_v04_00(Structure):
    pass

struct_NV9096_CTRL_GET_ZBC_CLEAR_TABLE_PARAMS_value_v04_00._pack_ = 1 # source:False
struct_NV9096_CTRL_GET_ZBC_CLEAR_TABLE_PARAMS_value_v04_00._fields_ = [
    ('colorFB', ctypes.c_uint32 * 4),
    ('colorDS', ctypes.c_uint32 * 4),
    ('depth', ctypes.c_uint32),
    ('stencil', ctypes.c_uint32),
]

struct_NV9096_CTRL_GET_ZBC_CLEAR_TABLE_PARAMS_v04_00._pack_ = 1 # source:False
struct_NV9096_CTRL_GET_ZBC_CLEAR_TABLE_PARAMS_v04_00._fields_ = [
    ('value', struct_NV9096_CTRL_GET_ZBC_CLEAR_TABLE_PARAMS_value_v04_00),
    ('indexSize', ctypes.c_uint32),
    ('indexUsed', ctypes.c_uint32),
    ('format', ctypes.c_uint32),
    ('valType', ctypes.c_uint32),
]

struct_rpc_ctrl_get_zbc_clear_table_v1A_09._pack_ = 1 # source:False
struct_rpc_ctrl_get_zbc_clear_table_v1A_09._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('getZbcClearTable', struct_NV9096_CTRL_GET_ZBC_CLEAR_TABLE_PARAMS_v04_00),
]

rpc_ctrl_get_zbc_clear_table_v1A_09 = struct_rpc_ctrl_get_zbc_clear_table_v1A_09
rpc_ctrl_get_zbc_clear_table_v = struct_rpc_ctrl_get_zbc_clear_table_v1A_09
class struct_rpc_ctrl_set_zbc_color_clear_v1A_09(Structure):
    pass

class struct_NV9096_CTRL_SET_ZBC_COLOR_CLEAR_PARAMS_v03_00(Structure):
    pass

struct_NV9096_CTRL_SET_ZBC_COLOR_CLEAR_PARAMS_v03_00._pack_ = 1 # source:False
struct_NV9096_CTRL_SET_ZBC_COLOR_CLEAR_PARAMS_v03_00._fields_ = [
    ('colorFB', ctypes.c_uint32 * 4),
    ('colorDS', ctypes.c_uint32 * 4),
    ('format', ctypes.c_uint32),
]

struct_rpc_ctrl_set_zbc_color_clear_v1A_09._pack_ = 1 # source:False
struct_rpc_ctrl_set_zbc_color_clear_v1A_09._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('setZbcColorClr', struct_NV9096_CTRL_SET_ZBC_COLOR_CLEAR_PARAMS_v03_00),
]

rpc_ctrl_set_zbc_color_clear_v1A_09 = struct_rpc_ctrl_set_zbc_color_clear_v1A_09
rpc_ctrl_set_zbc_color_clear_v = struct_rpc_ctrl_set_zbc_color_clear_v1A_09
class struct_rpc_ctrl_set_zbc_depth_clear_v1A_09(Structure):
    pass

class struct_NV9096_CTRL_SET_ZBC_DEPTH_CLEAR_PARAMS_v03_00(Structure):
    pass

struct_NV9096_CTRL_SET_ZBC_DEPTH_CLEAR_PARAMS_v03_00._pack_ = 1 # source:False
struct_NV9096_CTRL_SET_ZBC_DEPTH_CLEAR_PARAMS_v03_00._fields_ = [
    ('depth', ctypes.c_uint32),
    ('format', ctypes.c_uint32),
]

struct_rpc_ctrl_set_zbc_depth_clear_v1A_09._pack_ = 1 # source:False
struct_rpc_ctrl_set_zbc_depth_clear_v1A_09._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('setZbcDepthClr', struct_NV9096_CTRL_SET_ZBC_DEPTH_CLEAR_PARAMS_v03_00),
]

rpc_ctrl_set_zbc_depth_clear_v1A_09 = struct_rpc_ctrl_set_zbc_depth_clear_v1A_09
rpc_ctrl_set_zbc_depth_clear_v = struct_rpc_ctrl_set_zbc_depth_clear_v1A_09
class struct_rpc_ctrl_set_zbc_stencil_clear_v27_06(Structure):
    pass

class struct_NV9096_CTRL_SET_ZBC_STENCIL_CLEAR_PARAMS_v27_06(Structure):
    pass

struct_NV9096_CTRL_SET_ZBC_STENCIL_CLEAR_PARAMS_v27_06._pack_ = 1 # source:False
struct_NV9096_CTRL_SET_ZBC_STENCIL_CLEAR_PARAMS_v27_06._fields_ = [
    ('stencil', ctypes.c_uint32),
    ('format', ctypes.c_uint32),
    ('bSkipL2Table', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 3),
]

struct_rpc_ctrl_set_zbc_stencil_clear_v27_06._pack_ = 1 # source:False
struct_rpc_ctrl_set_zbc_stencil_clear_v27_06._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('setZbcStencilClr', struct_NV9096_CTRL_SET_ZBC_STENCIL_CLEAR_PARAMS_v27_06),
]

rpc_ctrl_set_zbc_stencil_clear_v27_06 = struct_rpc_ctrl_set_zbc_stencil_clear_v27_06
rpc_ctrl_set_zbc_stencil_clear_v = struct_rpc_ctrl_set_zbc_stencil_clear_v27_06
class struct_rpc_ctrl_gpfifo_schedule_v1A_0A(Structure):
    pass

class struct_NVA06F_CTRL_GPFIFO_SCHEDULE_PARAMS_v03_00(Structure):
    pass

struct_NVA06F_CTRL_GPFIFO_SCHEDULE_PARAMS_v03_00._pack_ = 1 # source:False
struct_NVA06F_CTRL_GPFIFO_SCHEDULE_PARAMS_v03_00._fields_ = [
    ('bEnable', ctypes.c_ubyte),
]

struct_rpc_ctrl_gpfifo_schedule_v1A_0A._pack_ = 1 # source:False
struct_rpc_ctrl_gpfifo_schedule_v1A_0A._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('cmd', ctypes.c_uint32),
    ('gpfifoSchedule', struct_NVA06F_CTRL_GPFIFO_SCHEDULE_PARAMS_v03_00),
    ('PADDING_0', ctypes.c_ubyte * 3),
]

rpc_ctrl_gpfifo_schedule_v1A_0A = struct_rpc_ctrl_gpfifo_schedule_v1A_0A
rpc_ctrl_gpfifo_schedule_v = struct_rpc_ctrl_gpfifo_schedule_v1A_0A
class struct_rpc_ctrl_set_timeslice_v1A_0A(Structure):
    pass

class struct_NVA06C_CTRL_TIMESLICE_PARAMS_v06_00(Structure):
    pass

struct_NVA06C_CTRL_TIMESLICE_PARAMS_v06_00._pack_ = 1 # source:False
struct_NVA06C_CTRL_TIMESLICE_PARAMS_v06_00._fields_ = [
    ('timesliceUs', ctypes.c_uint64),
]

struct_rpc_ctrl_set_timeslice_v1A_0A._pack_ = 1 # source:False
struct_rpc_ctrl_set_timeslice_v1A_0A._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('setTimeSlice', struct_NVA06C_CTRL_TIMESLICE_PARAMS_v06_00),
]

rpc_ctrl_set_timeslice_v1A_0A = struct_rpc_ctrl_set_timeslice_v1A_0A
rpc_ctrl_set_timeslice_v = struct_rpc_ctrl_set_timeslice_v1A_0A
class struct_rpc_ctrl_fifo_disable_channels_v1A_0A(Structure):
    pass

class struct_NV2080_CTRL_FIFO_DISABLE_CHANNELS_PARAMS_v06_00(Structure):
    pass

struct_NV2080_CTRL_FIFO_DISABLE_CHANNELS_PARAMS_v06_00._pack_ = 1 # source:False
struct_NV2080_CTRL_FIFO_DISABLE_CHANNELS_PARAMS_v06_00._fields_ = [
    ('bDisable', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 3),
    ('numChannels', ctypes.c_uint32),
    ('bOnlyDisableScheduling', ctypes.c_ubyte),
    ('bRewindGpPut', ctypes.c_ubyte),
    ('PADDING_1', ctypes.c_ubyte * 6),
    ('pRunlistPreemptEvent', ctypes.POINTER(None)),
    ('hClientList', ctypes.c_uint32 * 64),
    ('hChannelList', ctypes.c_uint32 * 64),
]

struct_rpc_ctrl_fifo_disable_channels_v1A_0A._pack_ = 1 # source:False
struct_rpc_ctrl_fifo_disable_channels_v1A_0A._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('fifoDisableChannels', struct_NV2080_CTRL_FIFO_DISABLE_CHANNELS_PARAMS_v06_00),
]

rpc_ctrl_fifo_disable_channels_v1A_0A = struct_rpc_ctrl_fifo_disable_channels_v1A_0A
rpc_ctrl_fifo_disable_channels_v = struct_rpc_ctrl_fifo_disable_channels_v1A_0A
class struct_rpc_ctrl_preempt_v1A_0A(Structure):
    pass

class struct_NVA06C_CTRL_PREEMPT_PARAMS_v09_0A(Structure):
    pass

struct_NVA06C_CTRL_PREEMPT_PARAMS_v09_0A._pack_ = 1 # source:False
struct_NVA06C_CTRL_PREEMPT_PARAMS_v09_0A._fields_ = [
    ('bWait', ctypes.c_ubyte),
    ('bManualTimeout', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 2),
    ('timeoutUs', ctypes.c_uint32),
]

struct_rpc_ctrl_preempt_v1A_0A._pack_ = 1 # source:False
struct_rpc_ctrl_preempt_v1A_0A._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('cmdPreempt', struct_NVA06C_CTRL_PREEMPT_PARAMS_v09_0A),
]

rpc_ctrl_preempt_v1A_0A = struct_rpc_ctrl_preempt_v1A_0A
rpc_ctrl_preempt_v = struct_rpc_ctrl_preempt_v1A_0A
class struct_rpc_ctrl_set_tsg_interleave_level_v1A_0A(Structure):
    pass

class struct_NVA06C_CTRL_INTERLEAVE_LEVEL_PARAMS_v17_02(Structure):
    pass

struct_NVA06C_CTRL_INTERLEAVE_LEVEL_PARAMS_v17_02._pack_ = 1 # source:False
struct_NVA06C_CTRL_INTERLEAVE_LEVEL_PARAMS_v17_02._fields_ = [
    ('tsgInterleaveLevel', ctypes.c_uint32),
]

struct_rpc_ctrl_set_tsg_interleave_level_v1A_0A._pack_ = 1 # source:False
struct_rpc_ctrl_set_tsg_interleave_level_v1A_0A._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('interleaveLevelTSG', struct_NVA06C_CTRL_INTERLEAVE_LEVEL_PARAMS_v17_02),
]

rpc_ctrl_set_tsg_interleave_level_v1A_0A = struct_rpc_ctrl_set_tsg_interleave_level_v1A_0A
rpc_ctrl_set_tsg_interleave_level_v = struct_rpc_ctrl_set_tsg_interleave_level_v1A_0A
class struct_rpc_ctrl_set_channel_interleave_level_v1A_0A(Structure):
    pass

class struct_NVA06F_CTRL_INTERLEAVE_LEVEL_PARAMS_v17_02(Structure):
    pass

struct_NVA06F_CTRL_INTERLEAVE_LEVEL_PARAMS_v17_02._pack_ = 1 # source:False
struct_NVA06F_CTRL_INTERLEAVE_LEVEL_PARAMS_v17_02._fields_ = [
    ('channelInterleaveLevel', ctypes.c_uint32),
]

struct_rpc_ctrl_set_channel_interleave_level_v1A_0A._pack_ = 1 # source:False
struct_rpc_ctrl_set_channel_interleave_level_v1A_0A._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('interleaveLevelChannel', struct_NVA06F_CTRL_INTERLEAVE_LEVEL_PARAMS_v17_02),
]

rpc_ctrl_set_channel_interleave_level_v1A_0A = struct_rpc_ctrl_set_channel_interleave_level_v1A_0A
rpc_ctrl_set_channel_interleave_level_v = struct_rpc_ctrl_set_channel_interleave_level_v1A_0A
class struct_rpc_ctrl_gr_ctxsw_preemption_bind_v1A_0E(Structure):
    pass

class struct_NV2080_CTRL_GR_CTXSW_PREEMPTION_BIND_PARAMS_v12_01(Structure):
    pass

struct_NV2080_CTRL_GR_CTXSW_PREEMPTION_BIND_PARAMS_v12_01._pack_ = 1 # source:False
struct_NV2080_CTRL_GR_CTXSW_PREEMPTION_BIND_PARAMS_v12_01._fields_ = [
    ('flags', ctypes.c_uint32),
    ('hClient', ctypes.c_uint32),
    ('hChannel', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('vMemPtrs', ctypes.c_uint64 * 8),
    ('gfxpPreemptMode', ctypes.c_uint32),
    ('cilpPreemptMode', ctypes.c_uint32),
    ('grRouteInfo', struct_NV2080_CTRL_GR_ROUTE_INFO_v12_01),
]

struct_rpc_ctrl_gr_ctxsw_preemption_bind_v1A_0E._pack_ = 1 # source:False
struct_rpc_ctrl_gr_ctxsw_preemption_bind_v1A_0E._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('ctrlParams', struct_NV2080_CTRL_GR_CTXSW_PREEMPTION_BIND_PARAMS_v12_01),
]

rpc_ctrl_gr_ctxsw_preemption_bind_v1A_0E = struct_rpc_ctrl_gr_ctxsw_preemption_bind_v1A_0E
class struct_rpc_ctrl_gr_ctxsw_preemption_bind_v28_07(Structure):
    pass

class struct_NV2080_CTRL_GR_CTXSW_PREEMPTION_BIND_PARAMS_v28_07(Structure):
    pass

struct_NV2080_CTRL_GR_CTXSW_PREEMPTION_BIND_PARAMS_v28_07._pack_ = 1 # source:False
struct_NV2080_CTRL_GR_CTXSW_PREEMPTION_BIND_PARAMS_v28_07._fields_ = [
    ('flags', ctypes.c_uint32),
    ('hClient', ctypes.c_uint32),
    ('hChannel', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('vMemPtrs', ctypes.c_uint64 * 9),
    ('gfxpPreemptMode', ctypes.c_uint32),
    ('cilpPreemptMode', ctypes.c_uint32),
    ('grRouteInfo', struct_NV2080_CTRL_GR_ROUTE_INFO_v12_01),
]

struct_rpc_ctrl_gr_ctxsw_preemption_bind_v28_07._pack_ = 1 # source:False
struct_rpc_ctrl_gr_ctxsw_preemption_bind_v28_07._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('ctrlParams', struct_NV2080_CTRL_GR_CTXSW_PREEMPTION_BIND_PARAMS_v28_07),
]

rpc_ctrl_gr_ctxsw_preemption_bind_v28_07 = struct_rpc_ctrl_gr_ctxsw_preemption_bind_v28_07
rpc_ctrl_gr_ctxsw_preemption_bind_v = struct_rpc_ctrl_gr_ctxsw_preemption_bind_v28_07
class struct_rpc_ctrl_gr_set_ctxsw_preemption_mode_v1A_0E(Structure):
    pass

class struct_NV2080_CTRL_GR_SET_CTXSW_PREEMPTION_MODE_PARAMS_v12_01(Structure):
    pass

struct_NV2080_CTRL_GR_SET_CTXSW_PREEMPTION_MODE_PARAMS_v12_01._pack_ = 1 # source:False
struct_NV2080_CTRL_GR_SET_CTXSW_PREEMPTION_MODE_PARAMS_v12_01._fields_ = [
    ('flags', ctypes.c_uint32),
    ('hChannel', ctypes.c_uint32),
    ('gfxpPreemptMode', ctypes.c_uint32),
    ('cilpPreemptMode', ctypes.c_uint32),
    ('grRouteInfo', struct_NV2080_CTRL_GR_ROUTE_INFO_v12_01),
]

struct_rpc_ctrl_gr_set_ctxsw_preemption_mode_v1A_0E._pack_ = 1 # source:False
struct_rpc_ctrl_gr_set_ctxsw_preemption_mode_v1A_0E._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('ctrlParams', struct_NV2080_CTRL_GR_SET_CTXSW_PREEMPTION_MODE_PARAMS_v12_01),
]

rpc_ctrl_gr_set_ctxsw_preemption_mode_v1A_0E = struct_rpc_ctrl_gr_set_ctxsw_preemption_mode_v1A_0E
rpc_ctrl_gr_set_ctxsw_preemption_mode_v = struct_rpc_ctrl_gr_set_ctxsw_preemption_mode_v1A_0E
class struct_rpc_ctrl_gr_ctxsw_zcull_bind_v1A_0E(Structure):
    pass

class struct_NV2080_CTRL_GR_CTXSW_ZCULL_BIND_PARAMS_v03_00(Structure):
    pass

struct_NV2080_CTRL_GR_CTXSW_ZCULL_BIND_PARAMS_v03_00._pack_ = 1 # source:False
struct_NV2080_CTRL_GR_CTXSW_ZCULL_BIND_PARAMS_v03_00._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hChannel', ctypes.c_uint32),
    ('vMemPtr', ctypes.c_uint64),
    ('zcullMode', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

struct_rpc_ctrl_gr_ctxsw_zcull_bind_v1A_0E._pack_ = 1 # source:False
struct_rpc_ctrl_gr_ctxsw_zcull_bind_v1A_0E._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('ctrlParams', struct_NV2080_CTRL_GR_CTXSW_ZCULL_BIND_PARAMS_v03_00),
]

rpc_ctrl_gr_ctxsw_zcull_bind_v1A_0E = struct_rpc_ctrl_gr_ctxsw_zcull_bind_v1A_0E
rpc_ctrl_gr_ctxsw_zcull_bind_v = struct_rpc_ctrl_gr_ctxsw_zcull_bind_v1A_0E
class struct_rpc_ctrl_gpu_initialize_ctx_v1A_0E(Structure):
    pass

class struct_NV2080_CTRL_GPU_INITIALIZE_CTX_PARAMS_v03_00(Structure):
    pass

struct_NV2080_CTRL_GPU_INITIALIZE_CTX_PARAMS_v03_00._pack_ = 1 # source:False
struct_NV2080_CTRL_GPU_INITIALIZE_CTX_PARAMS_v03_00._fields_ = [
    ('engineType', ctypes.c_uint32),
    ('hClient', ctypes.c_uint32),
    ('ChID', ctypes.c_uint32),
    ('hChanClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('hVirtMemory', ctypes.c_uint32),
    ('physAddress', ctypes.c_uint64),
    ('physAttr', ctypes.c_uint32),
    ('hDmaHandle', ctypes.c_uint32),
    ('index', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('size', ctypes.c_uint64),
]

struct_rpc_ctrl_gpu_initialize_ctx_v1A_0E._pack_ = 1 # source:False
struct_rpc_ctrl_gpu_initialize_ctx_v1A_0E._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('ctrlParams', struct_NV2080_CTRL_GPU_INITIALIZE_CTX_PARAMS_v03_00),
]

rpc_ctrl_gpu_initialize_ctx_v1A_0E = struct_rpc_ctrl_gpu_initialize_ctx_v1A_0E
rpc_ctrl_gpu_initialize_ctx_v = struct_rpc_ctrl_gpu_initialize_ctx_v1A_0E
class struct_rpc_ctrl_vaspace_copy_server_reserved_pdes_v1E_04(Structure):
    pass

class struct_NV90F1_CTRL_VASPACE_COPY_SERVER_RESERVED_PDES_PARAMS_v1E_04(Structure):
    pass

class struct_NV90F1_CTRL_VASPACE_COPY_SERVER_RESERVED_PDES_PARAMS_levels_v1E_04(Structure):
    pass

struct_NV90F1_CTRL_VASPACE_COPY_SERVER_RESERVED_PDES_PARAMS_levels_v1E_04._pack_ = 1 # source:False
struct_NV90F1_CTRL_VASPACE_COPY_SERVER_RESERVED_PDES_PARAMS_levels_v1E_04._fields_ = [
    ('physAddress', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
    ('aperture', ctypes.c_uint32),
    ('pageShift', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 3),
]

struct_NV90F1_CTRL_VASPACE_COPY_SERVER_RESERVED_PDES_PARAMS_v1E_04._pack_ = 1 # source:False
struct_NV90F1_CTRL_VASPACE_COPY_SERVER_RESERVED_PDES_PARAMS_v1E_04._fields_ = [
    ('hSubDevice', ctypes.c_uint32),
    ('subDeviceId', ctypes.c_uint32),
    ('pageSize', ctypes.c_uint64),
    ('virtAddrLo', ctypes.c_uint64),
    ('virtAddrHi', ctypes.c_uint64),
    ('numLevelsToCopy', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('levels', struct_NV90F1_CTRL_VASPACE_COPY_SERVER_RESERVED_PDES_PARAMS_levels_v1E_04 * 6),
]

struct_rpc_ctrl_vaspace_copy_server_reserved_pdes_v1E_04._pack_ = 1 # source:False
struct_rpc_ctrl_vaspace_copy_server_reserved_pdes_v1E_04._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('ctrlParams', struct_NV90F1_CTRL_VASPACE_COPY_SERVER_RESERVED_PDES_PARAMS_v1E_04),
]

rpc_ctrl_vaspace_copy_server_reserved_pdes_v1E_04 = struct_rpc_ctrl_vaspace_copy_server_reserved_pdes_v1E_04
rpc_ctrl_vaspace_copy_server_reserved_pdes_v = struct_rpc_ctrl_vaspace_copy_server_reserved_pdes_v1E_04
class struct_rpc_ctrl_mc_service_interrupts_v1A_0E(Structure):
    pass

class struct_NV2080_CTRL_MC_SERVICE_INTERRUPTS_PARAMS_v15_01(Structure):
    pass

struct_NV2080_CTRL_MC_SERVICE_INTERRUPTS_PARAMS_v15_01._pack_ = 1 # source:False
struct_NV2080_CTRL_MC_SERVICE_INTERRUPTS_PARAMS_v15_01._fields_ = [
    ('engines', ctypes.c_uint32),
]

struct_rpc_ctrl_mc_service_interrupts_v1A_0E._pack_ = 1 # source:False
struct_rpc_ctrl_mc_service_interrupts_v1A_0E._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('ctrlParams', struct_NV2080_CTRL_MC_SERVICE_INTERRUPTS_PARAMS_v15_01),
]

rpc_ctrl_mc_service_interrupts_v1A_0E = struct_rpc_ctrl_mc_service_interrupts_v1A_0E
rpc_ctrl_mc_service_interrupts_v = struct_rpc_ctrl_mc_service_interrupts_v1A_0E
class struct_rpc_ctrl_get_p2p_caps_v2_v1F_0D(Structure):
    pass

struct_rpc_ctrl_get_p2p_caps_v2_v1F_0D._pack_ = 1 # source:False
struct_rpc_ctrl_get_p2p_caps_v2_v1F_0D._fields_ = [
    ('iter', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 3),
    ('gpuIds', ctypes.c_uint32 * 32),
    ('gpuCount', ctypes.c_uint32),
    ('p2pCaps', ctypes.c_uint32),
    ('p2pOptimalReadCEs', ctypes.c_uint32),
    ('p2pOptimalWriteCEs', ctypes.c_uint32),
    ('p2pCapsStatus', ctypes.c_ubyte * 9),
    ('PADDING_1', ctypes.c_ubyte * 3),
    ('busPeerIds', ctypes.c_uint32 * 512),
]

rpc_ctrl_get_p2p_caps_v2_v1F_0D = struct_rpc_ctrl_get_p2p_caps_v2_v1F_0D
rpc_ctrl_get_p2p_caps_v2_v = struct_rpc_ctrl_get_p2p_caps_v2_v1F_0D
class struct_rpc_ctrl_subdevice_get_p2p_caps_v21_02(Structure):
    pass

class struct_NV2080_CTRL_GET_P2P_CAPS_PARAMS_v21_02(Structure):
    pass

class struct_NV2080_CTRL_GPU_P2P_PEER_CAPS_PEER_INFO_v21_02(Structure):
    pass

struct_NV2080_CTRL_GPU_P2P_PEER_CAPS_PEER_INFO_v21_02._pack_ = 1 # source:False
struct_NV2080_CTRL_GPU_P2P_PEER_CAPS_PEER_INFO_v21_02._fields_ = [
    ('gpuId', ctypes.c_uint32),
    ('gpuUuid', ctypes.c_ubyte * 16),
    ('p2pCaps', ctypes.c_uint32),
    ('p2pOptimalReadCEs', ctypes.c_uint32),
    ('p2pOptimalWriteCEs', ctypes.c_uint32),
    ('p2pCapsStatus', ctypes.c_ubyte * 9),
    ('PADDING_0', ctypes.c_ubyte * 3),
    ('busPeerId', ctypes.c_uint32),
]

struct_NV2080_CTRL_GET_P2P_CAPS_PARAMS_v21_02._pack_ = 1 # source:False
struct_NV2080_CTRL_GET_P2P_CAPS_PARAMS_v21_02._fields_ = [
    ('bAllCaps', ctypes.c_ubyte),
    ('bUseUuid', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 2),
    ('peerGpuCount', ctypes.c_uint32),
    ('peerGpuCaps', struct_NV2080_CTRL_GPU_P2P_PEER_CAPS_PEER_INFO_v21_02 * 32),
]

struct_rpc_ctrl_subdevice_get_p2p_caps_v21_02._pack_ = 1 # source:False
struct_rpc_ctrl_subdevice_get_p2p_caps_v21_02._fields_ = [
    ('ctrlParams', struct_NV2080_CTRL_GET_P2P_CAPS_PARAMS_v21_02),
]

rpc_ctrl_subdevice_get_p2p_caps_v21_02 = struct_rpc_ctrl_subdevice_get_p2p_caps_v21_02
rpc_ctrl_subdevice_get_p2p_caps_v = struct_rpc_ctrl_subdevice_get_p2p_caps_v21_02
class struct_rpc_ctrl_subdevice_get_vgpu_heap_stats_v28_03(Structure):
    pass

class struct_NV2080_CTRL_CMD_GSP_GET_VGPU_HEAP_STATS_PARAMS_v28_03(Structure):
    pass

struct_NV2080_CTRL_CMD_GSP_GET_VGPU_HEAP_STATS_PARAMS_v28_03._pack_ = 1 # source:False
struct_NV2080_CTRL_CMD_GSP_GET_VGPU_HEAP_STATS_PARAMS_v28_03._fields_ = [
    ('allocatedSize', ctypes.c_uint64),
    ('peakAllocatedSize', ctypes.c_uint64),
    ('managedSize', ctypes.c_uint64),
    ('allocationCount', ctypes.c_uint32),
    ('peakAllocationCount', ctypes.c_uint32),
]

struct_rpc_ctrl_subdevice_get_vgpu_heap_stats_v28_03._pack_ = 1 # source:False
struct_rpc_ctrl_subdevice_get_vgpu_heap_stats_v28_03._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('ctrlParams', struct_NV2080_CTRL_CMD_GSP_GET_VGPU_HEAP_STATS_PARAMS_v28_03),
]

rpc_ctrl_subdevice_get_vgpu_heap_stats_v28_03 = struct_rpc_ctrl_subdevice_get_vgpu_heap_stats_v28_03
class struct_rpc_ctrl_subdevice_get_vgpu_heap_stats_v28_06(Structure):
    pass

class struct_NV2080_CTRL_CMD_GSP_GET_VGPU_HEAP_STATS_PARAMS_v28_06(Structure):
    pass

struct_NV2080_CTRL_CMD_GSP_GET_VGPU_HEAP_STATS_PARAMS_v28_06._pack_ = 1 # source:False
struct_NV2080_CTRL_CMD_GSP_GET_VGPU_HEAP_STATS_PARAMS_v28_06._fields_ = [
    ('allocatedSize', ctypes.c_uint64),
    ('peakAllocatedSize', ctypes.c_uint64),
    ('managedSize', ctypes.c_uint64),
    ('allocationCount', ctypes.c_uint32),
    ('peakAllocationCount', ctypes.c_uint32),
    ('largestFreeChunkSize', ctypes.c_uint64),
]

struct_rpc_ctrl_subdevice_get_vgpu_heap_stats_v28_06._pack_ = 1 # source:False
struct_rpc_ctrl_subdevice_get_vgpu_heap_stats_v28_06._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('ctrlParams', struct_NV2080_CTRL_CMD_GSP_GET_VGPU_HEAP_STATS_PARAMS_v28_06),
]

rpc_ctrl_subdevice_get_vgpu_heap_stats_v28_06 = struct_rpc_ctrl_subdevice_get_vgpu_heap_stats_v28_06
rpc_ctrl_subdevice_get_vgpu_heap_stats_v = struct_rpc_ctrl_subdevice_get_vgpu_heap_stats_v28_06
class struct_rpc_ctrl_dbg_clear_all_sm_error_states_v1A_0C(Structure):
    pass

class struct_NV83DE_CTRL_DEBUG_CLEAR_ALL_SM_ERROR_STATES_PARAMS_v03_00(Structure):
    pass

struct_NV83DE_CTRL_DEBUG_CLEAR_ALL_SM_ERROR_STATES_PARAMS_v03_00._pack_ = 1 # source:False
struct_NV83DE_CTRL_DEBUG_CLEAR_ALL_SM_ERROR_STATES_PARAMS_v03_00._fields_ = [
    ('hTargetChannel', ctypes.c_uint32),
    ('numSMsToClear', ctypes.c_uint32),
]

struct_rpc_ctrl_dbg_clear_all_sm_error_states_v1A_0C._pack_ = 1 # source:False
struct_rpc_ctrl_dbg_clear_all_sm_error_states_v1A_0C._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('ctrlParams', struct_NV83DE_CTRL_DEBUG_CLEAR_ALL_SM_ERROR_STATES_PARAMS_v03_00),
]

rpc_ctrl_dbg_clear_all_sm_error_states_v1A_0C = struct_rpc_ctrl_dbg_clear_all_sm_error_states_v1A_0C
rpc_ctrl_dbg_clear_all_sm_error_states_v = struct_rpc_ctrl_dbg_clear_all_sm_error_states_v1A_0C
class struct_rpc_ctrl_dbg_read_all_sm_error_states_v21_06(Structure):
    pass

class struct_NV83DE_CTRL_DEBUG_READ_ALL_SM_ERROR_STATES_PARAMS_v21_06(Structure):
    pass

class struct_NV83DE_SM_ERROR_STATE_REGISTERS_v21_06(Structure):
    pass

struct_NV83DE_SM_ERROR_STATE_REGISTERS_v21_06._pack_ = 1 # source:False
struct_NV83DE_SM_ERROR_STATE_REGISTERS_v21_06._fields_ = [
    ('hwwGlobalEsr', ctypes.c_uint32),
    ('hwwWarpEsr', ctypes.c_uint32),
    ('hwwWarpEsrPc', ctypes.c_uint32),
    ('hwwGlobalEsrReportMask', ctypes.c_uint32),
    ('hwwWarpEsrReportMask', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('hwwEsrAddr', ctypes.c_uint64),
    ('hwwWarpEsrPc64', ctypes.c_uint64),
    ('hwwCgaEsr', ctypes.c_uint32),
    ('hwwCgaEsrReportMask', ctypes.c_uint32),
]

class struct_NV83DE_MMU_FAULT_INFO_v16_03(Structure):
    pass

struct_NV83DE_MMU_FAULT_INFO_v16_03._pack_ = 1 # source:False
struct_NV83DE_MMU_FAULT_INFO_v16_03._fields_ = [
    ('valid', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 3),
    ('faultInfo', ctypes.c_uint32),
]

struct_NV83DE_CTRL_DEBUG_READ_ALL_SM_ERROR_STATES_PARAMS_v21_06._pack_ = 1 # source:False
struct_NV83DE_CTRL_DEBUG_READ_ALL_SM_ERROR_STATES_PARAMS_v21_06._fields_ = [
    ('hTargetChannel', ctypes.c_uint32),
    ('numSMsToRead', ctypes.c_uint32),
    ('smErrorStateArray', struct_NV83DE_SM_ERROR_STATE_REGISTERS_v21_06 * 80),
    ('mmuFaultInfo', ctypes.c_uint32),
    ('mmuFault', struct_NV83DE_MMU_FAULT_INFO_v16_03),
    ('startingSM', ctypes.c_uint32),
]

struct_rpc_ctrl_dbg_read_all_sm_error_states_v21_06._pack_ = 1 # source:False
struct_rpc_ctrl_dbg_read_all_sm_error_states_v21_06._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('ctrlParams', struct_NV83DE_CTRL_DEBUG_READ_ALL_SM_ERROR_STATES_PARAMS_v21_06),
]

rpc_ctrl_dbg_read_all_sm_error_states_v21_06 = struct_rpc_ctrl_dbg_read_all_sm_error_states_v21_06
rpc_ctrl_dbg_read_all_sm_error_states_v = struct_rpc_ctrl_dbg_read_all_sm_error_states_v21_06
class struct_rpc_ctrl_dbg_set_exception_mask_v1A_0C(Structure):
    pass

class struct_NV83DE_CTRL_DEBUG_SET_EXCEPTION_MASK_PARAMS_v03_00(Structure):
    pass

struct_NV83DE_CTRL_DEBUG_SET_EXCEPTION_MASK_PARAMS_v03_00._pack_ = 1 # source:False
struct_NV83DE_CTRL_DEBUG_SET_EXCEPTION_MASK_PARAMS_v03_00._fields_ = [
    ('exceptionMask', ctypes.c_uint32),
]

struct_rpc_ctrl_dbg_set_exception_mask_v1A_0C._pack_ = 1 # source:False
struct_rpc_ctrl_dbg_set_exception_mask_v1A_0C._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('ctrlParams', struct_NV83DE_CTRL_DEBUG_SET_EXCEPTION_MASK_PARAMS_v03_00),
]

rpc_ctrl_dbg_set_exception_mask_v1A_0C = struct_rpc_ctrl_dbg_set_exception_mask_v1A_0C
rpc_ctrl_dbg_set_exception_mask_v = struct_rpc_ctrl_dbg_set_exception_mask_v1A_0C
class struct_rpc_ctrl_gpu_promote_ctx_v1A_20(Structure):
    pass

class struct_NV2080_CTRL_GPU_PROMOTE_CTX_PARAMS_v1A_20(Structure):
    pass

class struct_NV2080_CTRL_GPU_PROMOTE_CTX_BUFFER_ENTRY_v1A_20(Structure):
    pass

struct_NV2080_CTRL_GPU_PROMOTE_CTX_BUFFER_ENTRY_v1A_20._pack_ = 1 # source:False
struct_NV2080_CTRL_GPU_PROMOTE_CTX_BUFFER_ENTRY_v1A_20._fields_ = [
    ('gpuPhysAddr', ctypes.c_uint64),
    ('gpuVirtAddr', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
    ('physAttr', ctypes.c_uint32),
    ('bufferId', ctypes.c_uint16),
    ('bInitialize', ctypes.c_ubyte),
    ('bNonmapped', ctypes.c_ubyte),
]

struct_NV2080_CTRL_GPU_PROMOTE_CTX_PARAMS_v1A_20._pack_ = 1 # source:False
struct_NV2080_CTRL_GPU_PROMOTE_CTX_PARAMS_v1A_20._fields_ = [
    ('engineType', ctypes.c_uint32),
    ('hClient', ctypes.c_uint32),
    ('ChID', ctypes.c_uint32),
    ('hChanClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('hVirtMemory', ctypes.c_uint32),
    ('virtAddress', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
    ('entryCount', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('promoteEntry', struct_NV2080_CTRL_GPU_PROMOTE_CTX_BUFFER_ENTRY_v1A_20 * 16),
]

struct_rpc_ctrl_gpu_promote_ctx_v1A_20._pack_ = 1 # source:False
struct_rpc_ctrl_gpu_promote_ctx_v1A_20._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('promoteCtx', struct_NV2080_CTRL_GPU_PROMOTE_CTX_PARAMS_v1A_20),
]

rpc_ctrl_gpu_promote_ctx_v1A_20 = struct_rpc_ctrl_gpu_promote_ctx_v1A_20
rpc_ctrl_gpu_promote_ctx_v = struct_rpc_ctrl_gpu_promote_ctx_v1A_20
class struct_rpc_ctrl_dbg_suspend_context_v1A_10(Structure):
    pass

class struct_NV83DE_CTRL_CMD_DEBUG_SUSPEND_CONTEXT_PARAMS_v1A_06(Structure):
    pass

struct_NV83DE_CTRL_CMD_DEBUG_SUSPEND_CONTEXT_PARAMS_v1A_06._pack_ = 1 # source:False
struct_NV83DE_CTRL_CMD_DEBUG_SUSPEND_CONTEXT_PARAMS_v1A_06._fields_ = [
    ('waitForEvent', ctypes.c_uint32),
    ('hResidentChannel', ctypes.c_uint32),
]

struct_rpc_ctrl_dbg_suspend_context_v1A_10._pack_ = 1 # source:False
struct_rpc_ctrl_dbg_suspend_context_v1A_10._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('ctrlParams', struct_NV83DE_CTRL_CMD_DEBUG_SUSPEND_CONTEXT_PARAMS_v1A_06),
]

rpc_ctrl_dbg_suspend_context_v1A_10 = struct_rpc_ctrl_dbg_suspend_context_v1A_10
rpc_ctrl_dbg_suspend_context_v = struct_rpc_ctrl_dbg_suspend_context_v1A_10
class struct_rpc_ctrl_dbg_resume_context_v1A_10(Structure):
    pass

struct_rpc_ctrl_dbg_resume_context_v1A_10._pack_ = 1 # source:False
struct_rpc_ctrl_dbg_resume_context_v1A_10._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
]

rpc_ctrl_dbg_resume_context_v1A_10 = struct_rpc_ctrl_dbg_resume_context_v1A_10
rpc_ctrl_dbg_resume_context_v = struct_rpc_ctrl_dbg_resume_context_v1A_10
class struct_rpc_ctrl_dbg_exec_reg_ops_v1A_10(Structure):
    pass

class struct_NV83DE_CTRL_DEBUG_EXEC_REG_OPS_PARAMS_v1A_06(Structure):
    pass

struct_NV83DE_CTRL_DEBUG_EXEC_REG_OPS_PARAMS_v1A_06._pack_ = 1 # source:False
struct_NV83DE_CTRL_DEBUG_EXEC_REG_OPS_PARAMS_v1A_06._fields_ = [
    ('bNonTransactional', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 3),
    ('regOpCount', ctypes.c_uint32),
    ('regOps', struct_NV2080_CTRL_GPU_REG_OP_v03_00 * 100),
]

struct_rpc_ctrl_dbg_exec_reg_ops_v1A_10._pack_ = 1 # source:False
struct_rpc_ctrl_dbg_exec_reg_ops_v1A_10._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('ctrlParams', struct_NV83DE_CTRL_DEBUG_EXEC_REG_OPS_PARAMS_v1A_06),
]

rpc_ctrl_dbg_exec_reg_ops_v1A_10 = struct_rpc_ctrl_dbg_exec_reg_ops_v1A_10
rpc_ctrl_dbg_exec_reg_ops_v = struct_rpc_ctrl_dbg_exec_reg_ops_v1A_10
class struct_rpc_ctrl_dbg_set_mode_mmu_debug_v1A_10(Structure):
    pass

class struct_NV83DE_CTRL_DEBUG_SET_MODE_MMU_DEBUG_PARAMS_v1A_06(Structure):
    pass

struct_NV83DE_CTRL_DEBUG_SET_MODE_MMU_DEBUG_PARAMS_v1A_06._pack_ = 1 # source:False
struct_NV83DE_CTRL_DEBUG_SET_MODE_MMU_DEBUG_PARAMS_v1A_06._fields_ = [
    ('action', ctypes.c_uint32),
]

struct_rpc_ctrl_dbg_set_mode_mmu_debug_v1A_10._pack_ = 1 # source:False
struct_rpc_ctrl_dbg_set_mode_mmu_debug_v1A_10._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('ctrlParams', struct_NV83DE_CTRL_DEBUG_SET_MODE_MMU_DEBUG_PARAMS_v1A_06),
]

rpc_ctrl_dbg_set_mode_mmu_debug_v1A_10 = struct_rpc_ctrl_dbg_set_mode_mmu_debug_v1A_10
rpc_ctrl_dbg_set_mode_mmu_debug_v = struct_rpc_ctrl_dbg_set_mode_mmu_debug_v1A_10
class struct_rpc_ctrl_dbg_set_mode_mmu_gcc_debug_v29_07(Structure):
    pass

class struct_NV83DE_CTRL_DEBUG_SET_MODE_MMU_GCC_DEBUG_PARAMS_v29_07(Structure):
    pass

struct_NV83DE_CTRL_DEBUG_SET_MODE_MMU_GCC_DEBUG_PARAMS_v29_07._pack_ = 1 # source:False
struct_NV83DE_CTRL_DEBUG_SET_MODE_MMU_GCC_DEBUG_PARAMS_v29_07._fields_ = [
    ('action', ctypes.c_uint32),
]

struct_rpc_ctrl_dbg_set_mode_mmu_gcc_debug_v29_07._pack_ = 1 # source:False
struct_rpc_ctrl_dbg_set_mode_mmu_gcc_debug_v29_07._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('ctrlParams', struct_NV83DE_CTRL_DEBUG_SET_MODE_MMU_GCC_DEBUG_PARAMS_v29_07),
]

rpc_ctrl_dbg_set_mode_mmu_gcc_debug_v29_07 = struct_rpc_ctrl_dbg_set_mode_mmu_gcc_debug_v29_07
rpc_ctrl_dbg_set_mode_mmu_gcc_debug_v = struct_rpc_ctrl_dbg_set_mode_mmu_gcc_debug_v29_07
class struct_rpc_ctrl_dbg_read_single_sm_error_state_v21_06(Structure):
    pass

class struct_NV83DE_CTRL_DEBUG_READ_SINGLE_SM_ERROR_STATE_PARAMS_v21_06(Structure):
    pass

struct_NV83DE_CTRL_DEBUG_READ_SINGLE_SM_ERROR_STATE_PARAMS_v21_06._pack_ = 1 # source:False
struct_NV83DE_CTRL_DEBUG_READ_SINGLE_SM_ERROR_STATE_PARAMS_v21_06._fields_ = [
    ('hTargetChannel', ctypes.c_uint32),
    ('smID', ctypes.c_uint32),
    ('smErrorState', struct_NV83DE_SM_ERROR_STATE_REGISTERS_v21_06),
]

struct_rpc_ctrl_dbg_read_single_sm_error_state_v21_06._pack_ = 1 # source:False
struct_rpc_ctrl_dbg_read_single_sm_error_state_v21_06._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('ctrlParams', struct_NV83DE_CTRL_DEBUG_READ_SINGLE_SM_ERROR_STATE_PARAMS_v21_06),
]

rpc_ctrl_dbg_read_single_sm_error_state_v21_06 = struct_rpc_ctrl_dbg_read_single_sm_error_state_v21_06
rpc_ctrl_dbg_read_single_sm_error_state_v = struct_rpc_ctrl_dbg_read_single_sm_error_state_v21_06
class struct_rpc_ctrl_dbg_clear_single_sm_error_state_v1A_10(Structure):
    pass

class struct_NV83DE_CTRL_DEBUG_CLEAR_SINGLE_SM_ERROR_STATE_PARAMS_v1A_06(Structure):
    pass

struct_NV83DE_CTRL_DEBUG_CLEAR_SINGLE_SM_ERROR_STATE_PARAMS_v1A_06._pack_ = 1 # source:False
struct_NV83DE_CTRL_DEBUG_CLEAR_SINGLE_SM_ERROR_STATE_PARAMS_v1A_06._fields_ = [
    ('hTargetChannel', ctypes.c_uint32),
    ('smID', ctypes.c_uint32),
]

struct_rpc_ctrl_dbg_clear_single_sm_error_state_v1A_10._pack_ = 1 # source:False
struct_rpc_ctrl_dbg_clear_single_sm_error_state_v1A_10._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('ctrlParams', struct_NV83DE_CTRL_DEBUG_CLEAR_SINGLE_SM_ERROR_STATE_PARAMS_v1A_06),
]

rpc_ctrl_dbg_clear_single_sm_error_state_v1A_10 = struct_rpc_ctrl_dbg_clear_single_sm_error_state_v1A_10
rpc_ctrl_dbg_clear_single_sm_error_state_v = struct_rpc_ctrl_dbg_clear_single_sm_error_state_v1A_10
class struct_rpc_ctrl_dbg_set_mode_errbar_debug_v1A_10(Structure):
    pass

class struct_NV83DE_CTRL_DEBUG_SET_MODE_ERRBAR_DEBUG_PARAMS_v1A_06(Structure):
    pass

struct_NV83DE_CTRL_DEBUG_SET_MODE_ERRBAR_DEBUG_PARAMS_v1A_06._pack_ = 1 # source:False
struct_NV83DE_CTRL_DEBUG_SET_MODE_ERRBAR_DEBUG_PARAMS_v1A_06._fields_ = [
    ('action', ctypes.c_uint32),
]

struct_rpc_ctrl_dbg_set_mode_errbar_debug_v1A_10._pack_ = 1 # source:False
struct_rpc_ctrl_dbg_set_mode_errbar_debug_v1A_10._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('ctrlParams', struct_NV83DE_CTRL_DEBUG_SET_MODE_ERRBAR_DEBUG_PARAMS_v1A_06),
]

rpc_ctrl_dbg_set_mode_errbar_debug_v1A_10 = struct_rpc_ctrl_dbg_set_mode_errbar_debug_v1A_10
rpc_ctrl_dbg_set_mode_errbar_debug_v = struct_rpc_ctrl_dbg_set_mode_errbar_debug_v1A_10
class struct_rpc_ctrl_dbg_set_next_stop_trigger_type_v1A_10(Structure):
    pass

class struct_NV83DE_CTRL_DEBUG_SET_NEXT_STOP_TRIGGER_TYPE_PARAMS_v1A_06(Structure):
    pass

struct_NV83DE_CTRL_DEBUG_SET_NEXT_STOP_TRIGGER_TYPE_PARAMS_v1A_06._pack_ = 1 # source:False
struct_NV83DE_CTRL_DEBUG_SET_NEXT_STOP_TRIGGER_TYPE_PARAMS_v1A_06._fields_ = [
    ('stopTriggerType', ctypes.c_uint32),
]

struct_rpc_ctrl_dbg_set_next_stop_trigger_type_v1A_10._pack_ = 1 # source:False
struct_rpc_ctrl_dbg_set_next_stop_trigger_type_v1A_10._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('ctrlParams', struct_NV83DE_CTRL_DEBUG_SET_NEXT_STOP_TRIGGER_TYPE_PARAMS_v1A_06),
]

rpc_ctrl_dbg_set_next_stop_trigger_type_v1A_10 = struct_rpc_ctrl_dbg_set_next_stop_trigger_type_v1A_10
rpc_ctrl_dbg_set_next_stop_trigger_type_v = struct_rpc_ctrl_dbg_set_next_stop_trigger_type_v1A_10
class struct_rpc_ctrl_dma_set_default_vaspace_v1A_0E(Structure):
    pass

class struct_NV0080_CTRL_DMA_SET_DEFAULT_VASPACE_PARAMS_v03_00(Structure):
    pass

struct_NV0080_CTRL_DMA_SET_DEFAULT_VASPACE_PARAMS_v03_00._pack_ = 1 # source:False
struct_NV0080_CTRL_DMA_SET_DEFAULT_VASPACE_PARAMS_v03_00._fields_ = [
    ('hVASpace', ctypes.c_uint32),
]

struct_rpc_ctrl_dma_set_default_vaspace_v1A_0E._pack_ = 1 # source:False
struct_rpc_ctrl_dma_set_default_vaspace_v1A_0E._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('ctrlParams', struct_NV0080_CTRL_DMA_SET_DEFAULT_VASPACE_PARAMS_v03_00),
]

rpc_ctrl_dma_set_default_vaspace_v1A_0E = struct_rpc_ctrl_dma_set_default_vaspace_v1A_0E
rpc_ctrl_dma_set_default_vaspace_v = struct_rpc_ctrl_dma_set_default_vaspace_v1A_0E
class struct_rpc_ctrl_get_ce_pce_mask_v1A_0E(Structure):
    pass

class struct_NV2080_CTRL_CE_GET_CE_PCE_MASK_PARAMS_v1A_07(Structure):
    pass

struct_NV2080_CTRL_CE_GET_CE_PCE_MASK_PARAMS_v1A_07._pack_ = 1 # source:False
struct_NV2080_CTRL_CE_GET_CE_PCE_MASK_PARAMS_v1A_07._fields_ = [
    ('ceEngineType', ctypes.c_uint32),
    ('pceMask', ctypes.c_uint32),
]

struct_rpc_ctrl_get_ce_pce_mask_v1A_0E._pack_ = 1 # source:False
struct_rpc_ctrl_get_ce_pce_mask_v1A_0E._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('ctrlParams', struct_NV2080_CTRL_CE_GET_CE_PCE_MASK_PARAMS_v1A_07),
]

rpc_ctrl_get_ce_pce_mask_v1A_0E = struct_rpc_ctrl_get_ce_pce_mask_v1A_0E
rpc_ctrl_get_ce_pce_mask_v = struct_rpc_ctrl_get_ce_pce_mask_v1A_0E
class struct_rpc_ctrl_get_zbc_clear_table_entry_v1A_0E(Structure):
    pass

class struct_NV9096_CTRL_GET_ZBC_CLEAR_TABLE_ENTRY_PARAMS_v1A_07(Structure):
    pass

class struct_NV9096_CTRL_GET_ZBC_CLEAR_TABLE_ENTRY_PARAMS_value_v1A_07(Structure):
    pass

struct_NV9096_CTRL_GET_ZBC_CLEAR_TABLE_ENTRY_PARAMS_value_v1A_07._pack_ = 1 # source:False
struct_NV9096_CTRL_GET_ZBC_CLEAR_TABLE_ENTRY_PARAMS_value_v1A_07._fields_ = [
    ('colorFB', ctypes.c_uint32 * 4),
    ('colorDS', ctypes.c_uint32 * 4),
    ('depth', ctypes.c_uint32),
    ('stencil', ctypes.c_uint32),
]


# values for enumeration 'NV9096_CTRL_ZBC_CLEAR_TABLE_TYPE'
NV9096_CTRL_ZBC_CLEAR_TABLE_TYPE__enumvalues = {
    0: 'NV9096_CTRL_ZBC_CLEAR_TABLE_TYPE_INVALID',
    1: 'NV9096_CTRL_ZBC_CLEAR_TABLE_TYPE_COLOR',
    2: 'NV9096_CTRL_ZBC_CLEAR_TABLE_TYPE_DEPTH',
    3: 'NV9096_CTRL_ZBC_CLEAR_TABLE_TYPE_STENCIL',
    4: 'NV9096_CTRL_ZBC_CLEAR_TABLE_TYPE_COUNT',
}
NV9096_CTRL_ZBC_CLEAR_TABLE_TYPE_INVALID = 0
NV9096_CTRL_ZBC_CLEAR_TABLE_TYPE_COLOR = 1
NV9096_CTRL_ZBC_CLEAR_TABLE_TYPE_DEPTH = 2
NV9096_CTRL_ZBC_CLEAR_TABLE_TYPE_STENCIL = 3
NV9096_CTRL_ZBC_CLEAR_TABLE_TYPE_COUNT = 4
NV9096_CTRL_ZBC_CLEAR_TABLE_TYPE = ctypes.c_uint32 # enum
struct_NV9096_CTRL_GET_ZBC_CLEAR_TABLE_ENTRY_PARAMS_v1A_07._pack_ = 1 # source:False
struct_NV9096_CTRL_GET_ZBC_CLEAR_TABLE_ENTRY_PARAMS_v1A_07._fields_ = [
    ('value', struct_NV9096_CTRL_GET_ZBC_CLEAR_TABLE_ENTRY_PARAMS_value_v1A_07),
    ('format', ctypes.c_uint32),
    ('index', ctypes.c_uint32),
    ('bIndexValid', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 3),
    ('tableType', NV9096_CTRL_ZBC_CLEAR_TABLE_TYPE),
]

struct_rpc_ctrl_get_zbc_clear_table_entry_v1A_0E._pack_ = 1 # source:False
struct_rpc_ctrl_get_zbc_clear_table_entry_v1A_0E._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('ctrlParams', struct_NV9096_CTRL_GET_ZBC_CLEAR_TABLE_ENTRY_PARAMS_v1A_07),
]

rpc_ctrl_get_zbc_clear_table_entry_v1A_0E = struct_rpc_ctrl_get_zbc_clear_table_entry_v1A_0E
rpc_ctrl_get_zbc_clear_table_entry_v = struct_rpc_ctrl_get_zbc_clear_table_entry_v1A_0E
class struct_rpc_ctrl_get_nvlink_status_v23_04(Structure):
    pass

class struct_NV2080_CTRL_CMD_NVLINK_GET_NVLINK_STATUS_PARAMS_v23_04(Structure):
    pass

class struct_NV2080_CTRL_NVLINK_LINK_STATUS_INFO_v18_0D(Structure):
    pass

class struct_NV2080_CTRL_NVLINK_DEVICE_INFO_v15_02(Structure):
    pass

struct_NV2080_CTRL_NVLINK_DEVICE_INFO_v15_02._pack_ = 1 # source:False
struct_NV2080_CTRL_NVLINK_DEVICE_INFO_v15_02._fields_ = [
    ('deviceIdFlags', ctypes.c_uint32),
    ('domain', ctypes.c_uint32),
    ('bus', ctypes.c_uint16),
    ('device', ctypes.c_uint16),
    ('function', ctypes.c_uint16),
    ('PADDING_0', ctypes.c_ubyte * 2),
    ('pciDeviceId', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('deviceType', ctypes.c_uint64),
    ('deviceUUID', ctypes.c_ubyte * 16),
]

struct_NV2080_CTRL_NVLINK_LINK_STATUS_INFO_v18_0D._pack_ = 1 # source:False
struct_NV2080_CTRL_NVLINK_LINK_STATUS_INFO_v18_0D._fields_ = [
    ('capsTbl', ctypes.c_uint32),
    ('phyType', ctypes.c_ubyte),
    ('subLinkWidth', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 2),
    ('linkState', ctypes.c_uint32),
    ('rxSublinkStatus', ctypes.c_ubyte),
    ('txSublinkStatus', ctypes.c_ubyte),
    ('nvlinkVersion', ctypes.c_ubyte),
    ('nciVersion', ctypes.c_ubyte),
    ('phyVersion', ctypes.c_ubyte),
    ('PADDING_1', ctypes.c_ubyte * 3),
    ('nvlinkLinkClockKHz', ctypes.c_uint32),
    ('nvlinkLineRateMbps', ctypes.c_uint32),
    ('connected', ctypes.c_ubyte),
    ('remoteDeviceLinkNumber', ctypes.c_ubyte),
    ('localDeviceLinkNumber', ctypes.c_ubyte),
    ('PADDING_2', ctypes.c_ubyte),
    ('remoteDeviceInfo', struct_NV2080_CTRL_NVLINK_DEVICE_INFO_v15_02),
    ('localDeviceInfo', struct_NV2080_CTRL_NVLINK_DEVICE_INFO_v15_02),
]

struct_NV2080_CTRL_CMD_NVLINK_GET_NVLINK_STATUS_PARAMS_v23_04._pack_ = 1 # source:False
struct_NV2080_CTRL_CMD_NVLINK_GET_NVLINK_STATUS_PARAMS_v23_04._fields_ = [
    ('enabledLinkMask', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('linkInfo', struct_NV2080_CTRL_NVLINK_LINK_STATUS_INFO_v18_0D * 24),
]

struct_rpc_ctrl_get_nvlink_status_v23_04._pack_ = 1 # source:False
struct_rpc_ctrl_get_nvlink_status_v23_04._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('ctrlParams', struct_NV2080_CTRL_CMD_NVLINK_GET_NVLINK_STATUS_PARAMS_v23_04),
]

rpc_ctrl_get_nvlink_status_v23_04 = struct_rpc_ctrl_get_nvlink_status_v23_04
class struct_rpc_ctrl_get_nvlink_status_v28_09(Structure):
    pass

class struct_NV2080_CTRL_CMD_NVLINK_GET_NVLINK_STATUS_PARAMS_v28_09(Structure):
    pass

class struct_NV2080_CTRL_NVLINK_LINK_STATUS_INFO_v28_09(Structure):
    pass

class struct_NV2080_CTRL_NVLINK_DEVICE_INFO_v28_09(Structure):
    pass

struct_NV2080_CTRL_NVLINK_DEVICE_INFO_v28_09._pack_ = 1 # source:False
struct_NV2080_CTRL_NVLINK_DEVICE_INFO_v28_09._fields_ = [
    ('deviceIdFlags', ctypes.c_uint32),
    ('domain', ctypes.c_uint32),
    ('bus', ctypes.c_uint16),
    ('device', ctypes.c_uint16),
    ('function', ctypes.c_uint16),
    ('PADDING_0', ctypes.c_ubyte * 2),
    ('pciDeviceId', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('deviceType', ctypes.c_uint64),
    ('deviceUUID', ctypes.c_ubyte * 16),
    ('fabricRecoveryStatusMask', ctypes.c_uint32),
    ('PADDING_2', ctypes.c_ubyte * 4),
]

struct_NV2080_CTRL_NVLINK_LINK_STATUS_INFO_v28_09._pack_ = 1 # source:False
struct_NV2080_CTRL_NVLINK_LINK_STATUS_INFO_v28_09._fields_ = [
    ('capsTbl', ctypes.c_uint32),
    ('phyType', ctypes.c_ubyte),
    ('subLinkWidth', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 2),
    ('linkState', ctypes.c_uint32),
    ('rxSublinkStatus', ctypes.c_ubyte),
    ('txSublinkStatus', ctypes.c_ubyte),
    ('nvlinkVersion', ctypes.c_ubyte),
    ('nciVersion', ctypes.c_ubyte),
    ('phyVersion', ctypes.c_ubyte),
    ('PADDING_1', ctypes.c_ubyte * 3),
    ('nvlinkLinkClockKHz', ctypes.c_uint32),
    ('nvlinkLineRateMbps', ctypes.c_uint32),
    ('connected', ctypes.c_ubyte),
    ('remoteDeviceLinkNumber', ctypes.c_ubyte),
    ('localDeviceLinkNumber', ctypes.c_ubyte),
    ('PADDING_2', ctypes.c_ubyte),
    ('remoteDeviceInfo', struct_NV2080_CTRL_NVLINK_DEVICE_INFO_v28_09),
    ('localDeviceInfo', struct_NV2080_CTRL_NVLINK_DEVICE_INFO_v28_09),
]

struct_NV2080_CTRL_CMD_NVLINK_GET_NVLINK_STATUS_PARAMS_v28_09._pack_ = 1 # source:False
struct_NV2080_CTRL_CMD_NVLINK_GET_NVLINK_STATUS_PARAMS_v28_09._fields_ = [
    ('enabledLinkMask', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('linkInfo', struct_NV2080_CTRL_NVLINK_LINK_STATUS_INFO_v28_09 * 24),
]

struct_rpc_ctrl_get_nvlink_status_v28_09._pack_ = 1 # source:False
struct_rpc_ctrl_get_nvlink_status_v28_09._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('ctrlParams', struct_NV2080_CTRL_CMD_NVLINK_GET_NVLINK_STATUS_PARAMS_v28_09),
]

rpc_ctrl_get_nvlink_status_v28_09 = struct_rpc_ctrl_get_nvlink_status_v28_09
rpc_ctrl_get_nvlink_status_v = struct_rpc_ctrl_get_nvlink_status_v28_09
class struct_rpc_ctrl_get_p2p_caps_v1F_0D(Structure):
    pass

class struct_NV0000_CTRL_SYSTEM_GET_P2P_CAPS_PARAMS_v1F_0D(Structure):
    pass

struct_NV0000_CTRL_SYSTEM_GET_P2P_CAPS_PARAMS_v1F_0D._pack_ = 1 # source:False
struct_NV0000_CTRL_SYSTEM_GET_P2P_CAPS_PARAMS_v1F_0D._fields_ = [
    ('gpuIds', ctypes.c_uint32 * 32),
    ('gpuCount', ctypes.c_uint32),
    ('p2pCaps', ctypes.c_uint32),
    ('p2pOptimalReadCEs', ctypes.c_uint32),
    ('p2pOptimalWriteCEs', ctypes.c_uint32),
    ('p2pCapsStatus', ctypes.c_ubyte * 9),
    ('PADDING_0', ctypes.c_ubyte * 3),
]

struct_rpc_ctrl_get_p2p_caps_v1F_0D._pack_ = 1 # source:False
struct_rpc_ctrl_get_p2p_caps_v1F_0D._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('ctrlParams', struct_NV0000_CTRL_SYSTEM_GET_P2P_CAPS_PARAMS_v1F_0D),
]

rpc_ctrl_get_p2p_caps_v1F_0D = struct_rpc_ctrl_get_p2p_caps_v1F_0D
rpc_ctrl_get_p2p_caps_v = struct_rpc_ctrl_get_p2p_caps_v1F_0D
class struct_rpc_ctrl_get_p2p_caps_matrix_v1A_0E(Structure):
    pass

class struct_NV0000_CTRL_SYSTEM_GET_P2P_CAPS_MATRIX_PARAMS_v18_0A(Structure):
    pass

class struct_NV0000_CTRL_P2P_CAPS_MATRIX_ROW_v18_0A(Structure):
    pass

struct_NV0000_CTRL_P2P_CAPS_MATRIX_ROW_v18_0A._pack_ = 1 # source:False
struct_NV0000_CTRL_P2P_CAPS_MATRIX_ROW_v18_0A._fields_ = [
    ('array', ctypes.c_uint32 * 8),
]

struct_NV0000_CTRL_SYSTEM_GET_P2P_CAPS_MATRIX_PARAMS_v18_0A._pack_ = 1 # source:False
struct_NV0000_CTRL_SYSTEM_GET_P2P_CAPS_MATRIX_PARAMS_v18_0A._fields_ = [
    ('grpACount', ctypes.c_uint32),
    ('grpBCount', ctypes.c_uint32),
    ('gpuIdGrpA', ctypes.c_uint32 * 8),
    ('gpuIdGrpB', ctypes.c_uint32 * 8),
    ('p2pCaps', struct_NV0000_CTRL_P2P_CAPS_MATRIX_ROW_v18_0A * 8),
    ('a2bOptimalReadCes', struct_NV0000_CTRL_P2P_CAPS_MATRIX_ROW_v18_0A * 8),
    ('a2bOptimalWriteCes', struct_NV0000_CTRL_P2P_CAPS_MATRIX_ROW_v18_0A * 8),
    ('b2aOptimalReadCes', struct_NV0000_CTRL_P2P_CAPS_MATRIX_ROW_v18_0A * 8),
    ('b2aOptimalWriteCes', struct_NV0000_CTRL_P2P_CAPS_MATRIX_ROW_v18_0A * 8),
]

struct_rpc_ctrl_get_p2p_caps_matrix_v1A_0E._pack_ = 1 # source:False
struct_rpc_ctrl_get_p2p_caps_matrix_v1A_0E._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('ctrlParams', struct_NV0000_CTRL_SYSTEM_GET_P2P_CAPS_MATRIX_PARAMS_v18_0A),
]

rpc_ctrl_get_p2p_caps_matrix_v1A_0E = struct_rpc_ctrl_get_p2p_caps_matrix_v1A_0E
rpc_ctrl_get_p2p_caps_matrix_v = struct_rpc_ctrl_get_p2p_caps_matrix_v1A_0E
class struct_rpc_ctrl_reserve_pm_area_smpc_v1A_0F(Structure):
    pass

class struct_NVB0CC_CTRL_RESERVE_PM_AREA_SMPC_PARAMS_v1A_0F(Structure):
    pass

struct_NVB0CC_CTRL_RESERVE_PM_AREA_SMPC_PARAMS_v1A_0F._pack_ = 1 # source:False
struct_NVB0CC_CTRL_RESERVE_PM_AREA_SMPC_PARAMS_v1A_0F._fields_ = [
    ('ctxsw', ctypes.c_ubyte),
]

struct_rpc_ctrl_reserve_pm_area_smpc_v1A_0F._pack_ = 1 # source:False
struct_rpc_ctrl_reserve_pm_area_smpc_v1A_0F._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('params', struct_NVB0CC_CTRL_RESERVE_PM_AREA_SMPC_PARAMS_v1A_0F),
    ('PADDING_0', ctypes.c_ubyte * 3),
]

rpc_ctrl_reserve_pm_area_smpc_v1A_0F = struct_rpc_ctrl_reserve_pm_area_smpc_v1A_0F
rpc_ctrl_reserve_pm_area_smpc_v = struct_rpc_ctrl_reserve_pm_area_smpc_v1A_0F
class struct_rpc_ctrl_reserve_hwpm_legacy_v1A_0F(Structure):
    pass

class struct_NVB0CC_CTRL_RESERVE_HWPM_LEGACY_PARAMS_v1A_0F(Structure):
    pass

struct_NVB0CC_CTRL_RESERVE_HWPM_LEGACY_PARAMS_v1A_0F._pack_ = 1 # source:False
struct_NVB0CC_CTRL_RESERVE_HWPM_LEGACY_PARAMS_v1A_0F._fields_ = [
    ('ctxsw', ctypes.c_ubyte),
]

struct_rpc_ctrl_reserve_hwpm_legacy_v1A_0F._pack_ = 1 # source:False
struct_rpc_ctrl_reserve_hwpm_legacy_v1A_0F._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('params', struct_NVB0CC_CTRL_RESERVE_HWPM_LEGACY_PARAMS_v1A_0F),
    ('PADDING_0', ctypes.c_ubyte * 3),
]

rpc_ctrl_reserve_hwpm_legacy_v1A_0F = struct_rpc_ctrl_reserve_hwpm_legacy_v1A_0F
rpc_ctrl_reserve_hwpm_legacy_v = struct_rpc_ctrl_reserve_hwpm_legacy_v1A_0F
class struct_rpc_ctrl_b0cc_exec_reg_ops_v1A_0F(Structure):
    pass

class struct_NVB0CC_CTRL_EXEC_REG_OPS_PARAMS_v1A_0F(Structure):
    pass


# values for enumeration 'NVB0CC_REGOPS_MODE'
NVB0CC_REGOPS_MODE__enumvalues = {
    0: 'NVB0CC_REGOPS_MODE_ALL_OR_NONE',
    1: 'NVB0CC_REGOPS_MODE_CONTINUE_ON_ERROR',
}
NVB0CC_REGOPS_MODE_ALL_OR_NONE = 0
NVB0CC_REGOPS_MODE_CONTINUE_ON_ERROR = 1
NVB0CC_REGOPS_MODE = ctypes.c_uint32 # enum
struct_NVB0CC_CTRL_EXEC_REG_OPS_PARAMS_v1A_0F._pack_ = 1 # source:False
struct_NVB0CC_CTRL_EXEC_REG_OPS_PARAMS_v1A_0F._fields_ = [
    ('regOpCount', ctypes.c_uint32),
    ('mode', NVB0CC_REGOPS_MODE),
    ('bPassed', ctypes.c_ubyte),
    ('bDirect', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 2),
    ('regOps', struct_NV2080_CTRL_GPU_REG_OP_v03_00 * 124),
]

struct_rpc_ctrl_b0cc_exec_reg_ops_v1A_0F._pack_ = 1 # source:False
struct_rpc_ctrl_b0cc_exec_reg_ops_v1A_0F._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('params', struct_NVB0CC_CTRL_EXEC_REG_OPS_PARAMS_v1A_0F),
]

rpc_ctrl_b0cc_exec_reg_ops_v1A_0F = struct_rpc_ctrl_b0cc_exec_reg_ops_v1A_0F
rpc_ctrl_b0cc_exec_reg_ops_v = struct_rpc_ctrl_b0cc_exec_reg_ops_v1A_0F
class struct_rpc_ctrl_bind_pm_resources_v1A_0F(Structure):
    pass

struct_rpc_ctrl_bind_pm_resources_v1A_0F._pack_ = 1 # source:False
struct_rpc_ctrl_bind_pm_resources_v1A_0F._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
]

rpc_ctrl_bind_pm_resources_v1A_0F = struct_rpc_ctrl_bind_pm_resources_v1A_0F
rpc_ctrl_bind_pm_resources_v = struct_rpc_ctrl_bind_pm_resources_v1A_0F
class struct_rpc_ctrl_alloc_pma_stream_v1A_14(Structure):
    pass

class struct_NVB0CC_CTRL_ALLOC_PMA_STREAM_PARAMS_v1A_14(Structure):
    pass

struct_NVB0CC_CTRL_ALLOC_PMA_STREAM_PARAMS_v1A_14._pack_ = 1 # source:False
struct_NVB0CC_CTRL_ALLOC_PMA_STREAM_PARAMS_v1A_14._fields_ = [
    ('hMemPmaBuffer', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pmaBufferOffset', ctypes.c_uint64),
    ('pmaBufferSize', ctypes.c_uint64),
    ('hMemPmaBytesAvailable', ctypes.c_uint32),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('pmaBytesAvailableOffset', ctypes.c_uint64),
    ('ctxsw', ctypes.c_ubyte),
    ('PADDING_2', ctypes.c_ubyte * 3),
    ('pmaChannelIdx', ctypes.c_uint32),
    ('pmaBufferVA', ctypes.c_uint64),
]

struct_rpc_ctrl_alloc_pma_stream_v1A_14._pack_ = 1 # source:False
struct_rpc_ctrl_alloc_pma_stream_v1A_14._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('params', struct_NVB0CC_CTRL_ALLOC_PMA_STREAM_PARAMS_v1A_14),
]

rpc_ctrl_alloc_pma_stream_v1A_14 = struct_rpc_ctrl_alloc_pma_stream_v1A_14
rpc_ctrl_alloc_pma_stream_v = struct_rpc_ctrl_alloc_pma_stream_v1A_14
class struct_rpc_ctrl_pma_stream_update_get_put_v1A_14(Structure):
    pass

class struct_NVB0CC_CTRL_PMA_STREAM_UPDATE_GET_PUT_PARAMS_v1A_14(Structure):
    pass

struct_NVB0CC_CTRL_PMA_STREAM_UPDATE_GET_PUT_PARAMS_v1A_14._pack_ = 1 # source:False
struct_NVB0CC_CTRL_PMA_STREAM_UPDATE_GET_PUT_PARAMS_v1A_14._fields_ = [
    ('bytesConsumed', ctypes.c_uint64),
    ('bUpdateAvailableBytes', ctypes.c_ubyte),
    ('bWait', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 6),
    ('bytesAvailable', ctypes.c_uint64),
    ('bReturnPut', ctypes.c_ubyte),
    ('PADDING_1', ctypes.c_ubyte * 7),
    ('putPtr', ctypes.c_uint64),
    ('pmaChannelIdx', ctypes.c_uint32),
    ('PADDING_2', ctypes.c_ubyte * 4),
]

struct_rpc_ctrl_pma_stream_update_get_put_v1A_14._pack_ = 1 # source:False
struct_rpc_ctrl_pma_stream_update_get_put_v1A_14._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('params', struct_NVB0CC_CTRL_PMA_STREAM_UPDATE_GET_PUT_PARAMS_v1A_14),
]

rpc_ctrl_pma_stream_update_get_put_v1A_14 = struct_rpc_ctrl_pma_stream_update_get_put_v1A_14
rpc_ctrl_pma_stream_update_get_put_v = struct_rpc_ctrl_pma_stream_update_get_put_v1A_14
class struct_rpc_ctrl_fb_get_info_v2_v25_0A(Structure):
    pass

class struct_NV2080_CTRL_FB_GET_INFO_V2_PARAMS_v25_0A(Structure):
    pass

class struct_NV2080_CTRL_FB_INFO_v1A_15(Structure):
    pass

struct_NV2080_CTRL_FB_INFO_v1A_15._pack_ = 1 # source:False
struct_NV2080_CTRL_FB_INFO_v1A_15._fields_ = [
    ('index', ctypes.c_uint32),
    ('data', ctypes.c_uint32),
]

struct_NV2080_CTRL_FB_GET_INFO_V2_PARAMS_v25_0A._pack_ = 1 # source:False
struct_NV2080_CTRL_FB_GET_INFO_V2_PARAMS_v25_0A._fields_ = [
    ('fbInfoListSize', ctypes.c_uint32),
    ('fbInfoList', struct_NV2080_CTRL_FB_INFO_v1A_15 * 55),
]

struct_rpc_ctrl_fb_get_info_v2_v25_0A._pack_ = 1 # source:False
struct_rpc_ctrl_fb_get_info_v2_v25_0A._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('ctrlParams', struct_NV2080_CTRL_FB_GET_INFO_V2_PARAMS_v25_0A),
]

rpc_ctrl_fb_get_info_v2_v25_0A = struct_rpc_ctrl_fb_get_info_v2_v25_0A
class struct_rpc_ctrl_fb_get_info_v2_v27_00(Structure):
    pass

class struct_NV2080_CTRL_FB_GET_INFO_V2_PARAMS_v27_00(Structure):
    pass

struct_NV2080_CTRL_FB_GET_INFO_V2_PARAMS_v27_00._pack_ = 1 # source:False
struct_NV2080_CTRL_FB_GET_INFO_V2_PARAMS_v27_00._fields_ = [
    ('fbInfoListSize', ctypes.c_uint32),
    ('fbInfoList', struct_NV2080_CTRL_FB_INFO_v1A_15 * 57),
]

struct_rpc_ctrl_fb_get_info_v2_v27_00._pack_ = 1 # source:False
struct_rpc_ctrl_fb_get_info_v2_v27_00._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('ctrlParams', struct_NV2080_CTRL_FB_GET_INFO_V2_PARAMS_v27_00),
]

rpc_ctrl_fb_get_info_v2_v27_00 = struct_rpc_ctrl_fb_get_info_v2_v27_00
rpc_ctrl_fb_get_info_v2_v = struct_rpc_ctrl_fb_get_info_v2_v27_00
class struct_rpc_ctrl_fifo_set_channel_properties_v1A_16(Structure):
    pass

class struct_NV0080_CTRL_FIFO_SET_CHANNEL_PROPERTIES_PARAMS_v03_00(Structure):
    pass

struct_NV0080_CTRL_FIFO_SET_CHANNEL_PROPERTIES_PARAMS_v03_00._pack_ = 1 # source:False
struct_NV0080_CTRL_FIFO_SET_CHANNEL_PROPERTIES_PARAMS_v03_00._fields_ = [
    ('hChannel', ctypes.c_uint32),
    ('property', ctypes.c_uint32),
    ('value', ctypes.c_uint64),
]

struct_rpc_ctrl_fifo_set_channel_properties_v1A_16._pack_ = 1 # source:False
struct_rpc_ctrl_fifo_set_channel_properties_v1A_16._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('ctrlParams', struct_NV0080_CTRL_FIFO_SET_CHANNEL_PROPERTIES_PARAMS_v03_00),
]

rpc_ctrl_fifo_set_channel_properties_v1A_16 = struct_rpc_ctrl_fifo_set_channel_properties_v1A_16
rpc_ctrl_fifo_set_channel_properties_v = struct_rpc_ctrl_fifo_set_channel_properties_v1A_16
class struct_rpc_ctrl_gpu_evict_ctx_v1A_1C(Structure):
    pass

class struct_NV2080_CTRL_GPU_EVICT_CTX_PARAMS_v03_00(Structure):
    pass

struct_NV2080_CTRL_GPU_EVICT_CTX_PARAMS_v03_00._pack_ = 1 # source:False
struct_NV2080_CTRL_GPU_EVICT_CTX_PARAMS_v03_00._fields_ = [
    ('engineType', ctypes.c_uint32),
    ('hClient', ctypes.c_uint32),
    ('ChID', ctypes.c_uint32),
    ('hChanClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
]

struct_rpc_ctrl_gpu_evict_ctx_v1A_1C._pack_ = 1 # source:False
struct_rpc_ctrl_gpu_evict_ctx_v1A_1C._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('params', struct_NV2080_CTRL_GPU_EVICT_CTX_PARAMS_v03_00),
]

rpc_ctrl_gpu_evict_ctx_v1A_1C = struct_rpc_ctrl_gpu_evict_ctx_v1A_1C
rpc_ctrl_gpu_evict_ctx_v = struct_rpc_ctrl_gpu_evict_ctx_v1A_1C
class struct_rpc_ctrl_fb_get_fs_info_v24_00(Structure):
    pass

class struct_NV2080_CTRL_FB_GET_FS_INFO_PARAMS_v24_00(Structure):
    pass

class struct_NV2080_CTRL_FB_FS_INFO_QUERY_v1A_1D(Structure):
    pass

class union_NV2080_CTRL_FB_FS_INFO_QUERY_DATA_v1A_1D(Union):
    pass

class struct_NV2080_CTRL_FB_FS_INFO_INVALID_QUERY_PARAMS_v1A_1D(Structure):
    pass

struct_NV2080_CTRL_FB_FS_INFO_INVALID_QUERY_PARAMS_v1A_1D._pack_ = 1 # source:False
struct_NV2080_CTRL_FB_FS_INFO_INVALID_QUERY_PARAMS_v1A_1D._fields_ = [
    ('data', ctypes.c_ubyte * 24),
]

class struct_NV2080_CTRL_FB_FS_INFO_FBP_MASK_PARAMS_v1A_1D(Structure):
    pass

struct_NV2080_CTRL_FB_FS_INFO_FBP_MASK_PARAMS_v1A_1D._pack_ = 1 # source:False
struct_NV2080_CTRL_FB_FS_INFO_FBP_MASK_PARAMS_v1A_1D._fields_ = [
    ('swizzId', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('fbpEnMask', ctypes.c_uint64),
]

class struct_NV2080_CTRL_FB_FS_INFO_LTC_MASK_PARAMS_v1A_1D(Structure):
    pass

struct_NV2080_CTRL_FB_FS_INFO_LTC_MASK_PARAMS_v1A_1D._pack_ = 1 # source:False
struct_NV2080_CTRL_FB_FS_INFO_LTC_MASK_PARAMS_v1A_1D._fields_ = [
    ('fbpIndex', ctypes.c_uint32),
    ('ltcEnMask', ctypes.c_uint32),
]

class struct_NV2080_CTRL_FB_FS_INFO_LTS_MASK_PARAMS_v1A_1D(Structure):
    pass

struct_NV2080_CTRL_FB_FS_INFO_LTS_MASK_PARAMS_v1A_1D._pack_ = 1 # source:False
struct_NV2080_CTRL_FB_FS_INFO_LTS_MASK_PARAMS_v1A_1D._fields_ = [
    ('fbpIndex', ctypes.c_uint32),
    ('ltsEnMask', ctypes.c_uint32),
]

class struct_NV2080_CTRL_FB_FS_INFO_FBPA_MASK_PARAMS_v1A_1D(Structure):
    pass

struct_NV2080_CTRL_FB_FS_INFO_FBPA_MASK_PARAMS_v1A_1D._pack_ = 1 # source:False
struct_NV2080_CTRL_FB_FS_INFO_FBPA_MASK_PARAMS_v1A_1D._fields_ = [
    ('fbpIndex', ctypes.c_uint32),
    ('fbpaEnMask', ctypes.c_uint32),
]

class struct_NV2080_CTRL_FB_FS_INFO_ROP_MASK_PARAMS_v1A_1D(Structure):
    pass

struct_NV2080_CTRL_FB_FS_INFO_ROP_MASK_PARAMS_v1A_1D._pack_ = 1 # source:False
struct_NV2080_CTRL_FB_FS_INFO_ROP_MASK_PARAMS_v1A_1D._fields_ = [
    ('fbpIndex', ctypes.c_uint32),
    ('ropEnMask', ctypes.c_uint32),
]

class struct_NV2080_CTRL_FB_FS_INFO_PROFILER_MON_LTC_MASK_PARAMS_v1A_1D(Structure):
    pass

struct_NV2080_CTRL_FB_FS_INFO_PROFILER_MON_LTC_MASK_PARAMS_v1A_1D._pack_ = 1 # source:False
struct_NV2080_CTRL_FB_FS_INFO_PROFILER_MON_LTC_MASK_PARAMS_v1A_1D._fields_ = [
    ('fbpIndex', ctypes.c_uint32),
    ('swizzId', ctypes.c_uint32),
    ('ltcEnMask', ctypes.c_uint32),
]

class struct_NV2080_CTRL_FB_FS_INFO_PROFILER_MON_LTS_MASK_PARAMS_v1A_1D(Structure):
    pass

struct_NV2080_CTRL_FB_FS_INFO_PROFILER_MON_LTS_MASK_PARAMS_v1A_1D._pack_ = 1 # source:False
struct_NV2080_CTRL_FB_FS_INFO_PROFILER_MON_LTS_MASK_PARAMS_v1A_1D._fields_ = [
    ('fbpIndex', ctypes.c_uint32),
    ('swizzId', ctypes.c_uint32),
    ('ltsEnMask', ctypes.c_uint32),
]

class struct_NV2080_CTRL_FB_FS_INFO_PROFILER_MON_FBPA_MASK_PARAMS_v1A_1D(Structure):
    pass

struct_NV2080_CTRL_FB_FS_INFO_PROFILER_MON_FBPA_MASK_PARAMS_v1A_1D._pack_ = 1 # source:False
struct_NV2080_CTRL_FB_FS_INFO_PROFILER_MON_FBPA_MASK_PARAMS_v1A_1D._fields_ = [
    ('fbpIndex', ctypes.c_uint32),
    ('swizzId', ctypes.c_uint32),
    ('fbpaEnMask', ctypes.c_uint32),
]

class struct_NV2080_CTRL_FB_FS_INFO_PROFILER_MON_ROP_MASK_PARAMS_v1A_1D(Structure):
    pass

struct_NV2080_CTRL_FB_FS_INFO_PROFILER_MON_ROP_MASK_PARAMS_v1A_1D._pack_ = 1 # source:False
struct_NV2080_CTRL_FB_FS_INFO_PROFILER_MON_ROP_MASK_PARAMS_v1A_1D._fields_ = [
    ('fbpIndex', ctypes.c_uint32),
    ('swizzId', ctypes.c_uint32),
    ('ropEnMask', ctypes.c_uint32),
]

class struct_NV2080_CTRL_FB_FS_INFO_PROFILER_MON_FBPA_SUBP_MASK_PARAMS_v1A_1D(Structure):
    pass

struct_NV2080_CTRL_FB_FS_INFO_PROFILER_MON_FBPA_SUBP_MASK_PARAMS_v1A_1D._pack_ = 1 # source:False
struct_NV2080_CTRL_FB_FS_INFO_PROFILER_MON_FBPA_SUBP_MASK_PARAMS_v1A_1D._fields_ = [
    ('fbpIndex', ctypes.c_uint32),
    ('swizzId', ctypes.c_uint32),
    ('fbpaSubpEnMask', ctypes.c_uint64),
]

class struct_NV2080_CTRL_FB_FS_INFO_FBPA_SUBP_MASK_PARAMS_v1A_1D(Structure):
    pass

struct_NV2080_CTRL_FB_FS_INFO_FBPA_SUBP_MASK_PARAMS_v1A_1D._pack_ = 1 # source:False
struct_NV2080_CTRL_FB_FS_INFO_FBPA_SUBP_MASK_PARAMS_v1A_1D._fields_ = [
    ('fbpIndex', ctypes.c_uint32),
    ('fbpaSubpEnMask', ctypes.c_uint32),
]

class struct_NV2080_CTRL_FB_FS_INFO_FBP_LOGICAL_MAP_PARAMS_v1A_1D(Structure):
    pass

struct_NV2080_CTRL_FB_FS_INFO_FBP_LOGICAL_MAP_PARAMS_v1A_1D._pack_ = 1 # source:False
struct_NV2080_CTRL_FB_FS_INFO_FBP_LOGICAL_MAP_PARAMS_v1A_1D._fields_ = [
    ('fbpIndex', ctypes.c_uint32),
    ('fbpLogicalIndex', ctypes.c_uint32),
]

union_NV2080_CTRL_FB_FS_INFO_QUERY_DATA_v1A_1D._pack_ = 1 # source:False
union_NV2080_CTRL_FB_FS_INFO_QUERY_DATA_v1A_1D._fields_ = [
    ('inv', struct_NV2080_CTRL_FB_FS_INFO_INVALID_QUERY_PARAMS_v1A_1D),
    ('fbp', struct_NV2080_CTRL_FB_FS_INFO_FBP_MASK_PARAMS_v1A_1D),
    ('ltc', struct_NV2080_CTRL_FB_FS_INFO_LTC_MASK_PARAMS_v1A_1D),
    ('lts', struct_NV2080_CTRL_FB_FS_INFO_LTS_MASK_PARAMS_v1A_1D),
    ('fbpa', struct_NV2080_CTRL_FB_FS_INFO_FBPA_MASK_PARAMS_v1A_1D),
    ('rop', struct_NV2080_CTRL_FB_FS_INFO_ROP_MASK_PARAMS_v1A_1D),
    ('dmLtc', struct_NV2080_CTRL_FB_FS_INFO_PROFILER_MON_LTC_MASK_PARAMS_v1A_1D),
    ('dmLts', struct_NV2080_CTRL_FB_FS_INFO_PROFILER_MON_LTS_MASK_PARAMS_v1A_1D),
    ('dmFbpa', struct_NV2080_CTRL_FB_FS_INFO_PROFILER_MON_FBPA_MASK_PARAMS_v1A_1D),
    ('dmRop', struct_NV2080_CTRL_FB_FS_INFO_PROFILER_MON_ROP_MASK_PARAMS_v1A_1D),
    ('dmFbpaSubp', struct_NV2080_CTRL_FB_FS_INFO_PROFILER_MON_FBPA_SUBP_MASK_PARAMS_v1A_1D),
    ('fbpaSubp', struct_NV2080_CTRL_FB_FS_INFO_FBPA_SUBP_MASK_PARAMS_v1A_1D),
    ('fbpLogicalMap', struct_NV2080_CTRL_FB_FS_INFO_FBP_LOGICAL_MAP_PARAMS_v1A_1D),
    ('PADDING_0', ctypes.c_ubyte * 16),
]

struct_NV2080_CTRL_FB_FS_INFO_QUERY_v1A_1D._pack_ = 1 # source:False
struct_NV2080_CTRL_FB_FS_INFO_QUERY_v1A_1D._fields_ = [
    ('queryType', ctypes.c_uint16),
    ('reserved', ctypes.c_ubyte * 2),
    ('status', ctypes.c_uint32),
    ('queryParams', union_NV2080_CTRL_FB_FS_INFO_QUERY_DATA_v1A_1D),
]

struct_NV2080_CTRL_FB_GET_FS_INFO_PARAMS_v24_00._pack_ = 1 # source:False
struct_NV2080_CTRL_FB_GET_FS_INFO_PARAMS_v24_00._fields_ = [
    ('numQueries', ctypes.c_uint16),
    ('reserved', ctypes.c_ubyte * 6),
    ('queries', struct_NV2080_CTRL_FB_FS_INFO_QUERY_v1A_1D * 120),
]

struct_rpc_ctrl_fb_get_fs_info_v24_00._pack_ = 1 # source:False
struct_rpc_ctrl_fb_get_fs_info_v24_00._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('params', struct_NV2080_CTRL_FB_GET_FS_INFO_PARAMS_v24_00),
]

rpc_ctrl_fb_get_fs_info_v24_00 = struct_rpc_ctrl_fb_get_fs_info_v24_00
class struct_rpc_ctrl_fb_get_fs_info_v26_04(Structure):
    pass

class struct_NV2080_CTRL_FB_GET_FS_INFO_PARAMS_v26_04(Structure):
    pass

class struct_NV2080_CTRL_FB_FS_INFO_QUERY_v26_04(Structure):
    pass

class union_NV2080_CTRL_FB_FS_INFO_QUERY_DATA_v26_04(Union):
    pass

class struct_NV2080_CTRL_SYSL2_FS_INFO_SYSLTC_MASK_PARAMS_v26_04(Structure):
    pass

struct_NV2080_CTRL_SYSL2_FS_INFO_SYSLTC_MASK_PARAMS_v26_04._pack_ = 1 # source:False
struct_NV2080_CTRL_SYSL2_FS_INFO_SYSLTC_MASK_PARAMS_v26_04._fields_ = [
    ('sysIdx', ctypes.c_uint32),
    ('sysl2LtcEnMask', ctypes.c_uint32),
]

class struct_NV2080_CTRL_FB_FS_INFO_PAC_MASK_PARAMS_v26_04(Structure):
    pass

struct_NV2080_CTRL_FB_FS_INFO_PAC_MASK_PARAMS_v26_04._pack_ = 1 # source:False
struct_NV2080_CTRL_FB_FS_INFO_PAC_MASK_PARAMS_v26_04._fields_ = [
    ('fbpIndex', ctypes.c_uint32),
    ('pacEnMask', ctypes.c_uint32),
]

class struct_NV2080_CTRL_FB_FS_INFO_LOGICAL_LTC_MASK_PARAMS_v26_04(Structure):
    pass

struct_NV2080_CTRL_FB_FS_INFO_LOGICAL_LTC_MASK_PARAMS_v26_04._pack_ = 1 # source:False
struct_NV2080_CTRL_FB_FS_INFO_LOGICAL_LTC_MASK_PARAMS_v26_04._fields_ = [
    ('fbpIndex', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('logicalLtcEnMask', ctypes.c_uint64),
]

class struct_NV2080_CTRL_FB_FS_INFO_PROFILER_MON_LOGICAL_LTC_MASK_PARAMS_v26_04(Structure):
    pass

struct_NV2080_CTRL_FB_FS_INFO_PROFILER_MON_LOGICAL_LTC_MASK_PARAMS_v26_04._pack_ = 1 # source:False
struct_NV2080_CTRL_FB_FS_INFO_PROFILER_MON_LOGICAL_LTC_MASK_PARAMS_v26_04._fields_ = [
    ('fbpIndex', ctypes.c_uint32),
    ('swizzId', ctypes.c_uint32),
    ('logicalLtcEnMask', ctypes.c_uint64),
]

union_NV2080_CTRL_FB_FS_INFO_QUERY_DATA_v26_04._pack_ = 1 # source:False
union_NV2080_CTRL_FB_FS_INFO_QUERY_DATA_v26_04._fields_ = [
    ('inv', struct_NV2080_CTRL_FB_FS_INFO_INVALID_QUERY_PARAMS_v1A_1D),
    ('fbp', struct_NV2080_CTRL_FB_FS_INFO_FBP_MASK_PARAMS_v1A_1D),
    ('ltc', struct_NV2080_CTRL_FB_FS_INFO_LTC_MASK_PARAMS_v1A_1D),
    ('lts', struct_NV2080_CTRL_FB_FS_INFO_LTS_MASK_PARAMS_v1A_1D),
    ('fbpa', struct_NV2080_CTRL_FB_FS_INFO_FBPA_MASK_PARAMS_v1A_1D),
    ('rop', struct_NV2080_CTRL_FB_FS_INFO_ROP_MASK_PARAMS_v1A_1D),
    ('dmLtc', struct_NV2080_CTRL_FB_FS_INFO_PROFILER_MON_LTC_MASK_PARAMS_v1A_1D),
    ('dmLts', struct_NV2080_CTRL_FB_FS_INFO_PROFILER_MON_LTS_MASK_PARAMS_v1A_1D),
    ('dmFbpa', struct_NV2080_CTRL_FB_FS_INFO_PROFILER_MON_FBPA_MASK_PARAMS_v1A_1D),
    ('dmRop', struct_NV2080_CTRL_FB_FS_INFO_PROFILER_MON_ROP_MASK_PARAMS_v1A_1D),
    ('dmFbpaSubp', struct_NV2080_CTRL_FB_FS_INFO_PROFILER_MON_FBPA_SUBP_MASK_PARAMS_v1A_1D),
    ('fbpaSubp', struct_NV2080_CTRL_FB_FS_INFO_FBPA_SUBP_MASK_PARAMS_v1A_1D),
    ('fbpLogicalMap', struct_NV2080_CTRL_FB_FS_INFO_FBP_LOGICAL_MAP_PARAMS_v1A_1D),
    ('sysl2Ltc', struct_NV2080_CTRL_SYSL2_FS_INFO_SYSLTC_MASK_PARAMS_v26_04),
    ('pac', struct_NV2080_CTRL_FB_FS_INFO_PAC_MASK_PARAMS_v26_04),
    ('logicalLtc', struct_NV2080_CTRL_FB_FS_INFO_LOGICAL_LTC_MASK_PARAMS_v26_04),
    ('dmLogicalLtc', struct_NV2080_CTRL_FB_FS_INFO_PROFILER_MON_LOGICAL_LTC_MASK_PARAMS_v26_04),
    ('PADDING_0', ctypes.c_ubyte * 8),
]

struct_NV2080_CTRL_FB_FS_INFO_QUERY_v26_04._pack_ = 1 # source:False
struct_NV2080_CTRL_FB_FS_INFO_QUERY_v26_04._fields_ = [
    ('queryType', ctypes.c_uint16),
    ('reserved', ctypes.c_ubyte * 2),
    ('status', ctypes.c_uint32),
    ('queryParams', union_NV2080_CTRL_FB_FS_INFO_QUERY_DATA_v26_04),
]

struct_NV2080_CTRL_FB_GET_FS_INFO_PARAMS_v26_04._pack_ = 1 # source:False
struct_NV2080_CTRL_FB_GET_FS_INFO_PARAMS_v26_04._fields_ = [
    ('numQueries', ctypes.c_uint16),
    ('reserved', ctypes.c_ubyte * 6),
    ('queries', struct_NV2080_CTRL_FB_FS_INFO_QUERY_v26_04 * 120),
]

struct_rpc_ctrl_fb_get_fs_info_v26_04._pack_ = 1 # source:False
struct_rpc_ctrl_fb_get_fs_info_v26_04._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('params', struct_NV2080_CTRL_FB_GET_FS_INFO_PARAMS_v26_04),
]

rpc_ctrl_fb_get_fs_info_v26_04 = struct_rpc_ctrl_fb_get_fs_info_v26_04
rpc_ctrl_fb_get_fs_info_v = struct_rpc_ctrl_fb_get_fs_info_v26_04
class struct_rpc_ctrl_grmgr_get_gr_fs_info_v1A_1D(Structure):
    pass

class struct_NV2080_CTRL_GRMGR_GET_GR_FS_INFO_PARAMS_v1A_1D(Structure):
    pass

class struct_NV2080_CTRL_GRMGR_GR_FS_INFO_QUERY_PARAMS_v1A_1D(Structure):
    pass

class union_NV2080_CTRL_GRMGR_GR_FS_INFO_QUERY_DATA_v1A_1D(Union):
    pass

class struct_NV2080_CTRL_GRMGR_GR_FS_INFO_GPC_COUNT_PARAMS_v1A_1D(Structure):
    pass

struct_NV2080_CTRL_GRMGR_GR_FS_INFO_GPC_COUNT_PARAMS_v1A_1D._pack_ = 1 # source:False
struct_NV2080_CTRL_GRMGR_GR_FS_INFO_GPC_COUNT_PARAMS_v1A_1D._fields_ = [
    ('gpcCount', ctypes.c_uint32),
]

class struct_NV2080_CTRL_GRMGR_GR_FS_INFO_CHIPLET_GPC_MAP_PARAMS_v1A_1D(Structure):
    pass

struct_NV2080_CTRL_GRMGR_GR_FS_INFO_CHIPLET_GPC_MAP_PARAMS_v1A_1D._pack_ = 1 # source:False
struct_NV2080_CTRL_GRMGR_GR_FS_INFO_CHIPLET_GPC_MAP_PARAMS_v1A_1D._fields_ = [
    ('gpcId', ctypes.c_uint32),
    ('chipletGpcMap', ctypes.c_uint32),
]

class struct_NV2080_CTRL_GRMGR_GR_FS_INFO_TPC_MASK_PARAMS_v1A_1D(Structure):
    pass

struct_NV2080_CTRL_GRMGR_GR_FS_INFO_TPC_MASK_PARAMS_v1A_1D._pack_ = 1 # source:False
struct_NV2080_CTRL_GRMGR_GR_FS_INFO_TPC_MASK_PARAMS_v1A_1D._fields_ = [
    ('gpcId', ctypes.c_uint32),
    ('tpcMask', ctypes.c_uint32),
]

class struct_NV2080_CTRL_GRMGR_GR_FS_INFO_PPC_MASK_PARAMS_v1A_1D(Structure):
    pass

struct_NV2080_CTRL_GRMGR_GR_FS_INFO_PPC_MASK_PARAMS_v1A_1D._pack_ = 1 # source:False
struct_NV2080_CTRL_GRMGR_GR_FS_INFO_PPC_MASK_PARAMS_v1A_1D._fields_ = [
    ('gpcId', ctypes.c_uint32),
    ('ppcMask', ctypes.c_uint32),
]

class struct_NV2080_CTRL_GRMGR_GR_FS_INFO_PARTITION_CHIPLET_GPC_MAP_PARAMS_v1A_1D(Structure):
    pass

struct_NV2080_CTRL_GRMGR_GR_FS_INFO_PARTITION_CHIPLET_GPC_MAP_PARAMS_v1A_1D._pack_ = 1 # source:False
struct_NV2080_CTRL_GRMGR_GR_FS_INFO_PARTITION_CHIPLET_GPC_MAP_PARAMS_v1A_1D._fields_ = [
    ('swizzId', ctypes.c_uint32),
    ('gpcId', ctypes.c_uint32),
    ('chipletGpcMap', ctypes.c_uint32),
]

class struct_NV2080_CTRL_GRMGR_GR_FS_INFO_CHIPLET_SYSPIPE_MASK_PARAMS_v1A_1D(Structure):
    pass

struct_NV2080_CTRL_GRMGR_GR_FS_INFO_CHIPLET_SYSPIPE_MASK_PARAMS_v1A_1D._pack_ = 1 # source:False
struct_NV2080_CTRL_GRMGR_GR_FS_INFO_CHIPLET_SYSPIPE_MASK_PARAMS_v1A_1D._fields_ = [
    ('chipletSyspipeMask', ctypes.c_uint32),
]

class struct_NV2080_CTRL_GRMGR_GR_FS_INFO_PARTITION_CHIPLET_SYSPIPE_IDS_PARAMS_v1A_1D(Structure):
    pass

struct_NV2080_CTRL_GRMGR_GR_FS_INFO_PARTITION_CHIPLET_SYSPIPE_IDS_PARAMS_v1A_1D._pack_ = 1 # source:False
struct_NV2080_CTRL_GRMGR_GR_FS_INFO_PARTITION_CHIPLET_SYSPIPE_IDS_PARAMS_v1A_1D._fields_ = [
    ('swizzId', ctypes.c_uint16),
    ('physSyspipeIdCount', ctypes.c_uint16),
    ('physSyspipeId', ctypes.c_ubyte * 8),
]

class struct_NV2080_CTRL_GRMGR_GR_FS_INFO_PROFILER_MON_GPC_MASK_PARAMS_v1A_1D(Structure):
    pass

struct_NV2080_CTRL_GRMGR_GR_FS_INFO_PROFILER_MON_GPC_MASK_PARAMS_v1A_1D._pack_ = 1 # source:False
struct_NV2080_CTRL_GRMGR_GR_FS_INFO_PROFILER_MON_GPC_MASK_PARAMS_v1A_1D._fields_ = [
    ('swizzId', ctypes.c_uint32),
    ('grIdx', ctypes.c_uint32),
    ('gpcEnMask', ctypes.c_uint32),
]

class struct_NV2080_CTRL_GRMGR_GR_FS_INFO_PARTITION_SYSPIPE_ID_PARAMS_v1A_1D(Structure):
    pass

struct_NV2080_CTRL_GRMGR_GR_FS_INFO_PARTITION_SYSPIPE_ID_PARAMS_v1A_1D._pack_ = 1 # source:False
struct_NV2080_CTRL_GRMGR_GR_FS_INFO_PARTITION_SYSPIPE_ID_PARAMS_v1A_1D._fields_ = [
    ('syspipeId', ctypes.c_uint32),
]

class struct_NV2080_CTRL_GRMGR_GR_FS_INFO_ROP_MASK_PARAMS_v1A_1D(Structure):
    pass

struct_NV2080_CTRL_GRMGR_GR_FS_INFO_ROP_MASK_PARAMS_v1A_1D._pack_ = 1 # source:False
struct_NV2080_CTRL_GRMGR_GR_FS_INFO_ROP_MASK_PARAMS_v1A_1D._fields_ = [
    ('gpcId', ctypes.c_uint32),
    ('ropMask', ctypes.c_uint32),
]

union_NV2080_CTRL_GRMGR_GR_FS_INFO_QUERY_DATA_v1A_1D._pack_ = 1 # source:False
union_NV2080_CTRL_GRMGR_GR_FS_INFO_QUERY_DATA_v1A_1D._fields_ = [
    ('gpcCountData', struct_NV2080_CTRL_GRMGR_GR_FS_INFO_GPC_COUNT_PARAMS_v1A_1D),
    ('chipletGpcMapData', struct_NV2080_CTRL_GRMGR_GR_FS_INFO_CHIPLET_GPC_MAP_PARAMS_v1A_1D),
    ('tpcMaskData', struct_NV2080_CTRL_GRMGR_GR_FS_INFO_TPC_MASK_PARAMS_v1A_1D),
    ('ppcMaskData', struct_NV2080_CTRL_GRMGR_GR_FS_INFO_PPC_MASK_PARAMS_v1A_1D),
    ('partitionGpcMapData', struct_NV2080_CTRL_GRMGR_GR_FS_INFO_PARTITION_CHIPLET_GPC_MAP_PARAMS_v1A_1D),
    ('syspipeMaskData', struct_NV2080_CTRL_GRMGR_GR_FS_INFO_CHIPLET_SYSPIPE_MASK_PARAMS_v1A_1D),
    ('partitionChipletSyspipeData', struct_NV2080_CTRL_GRMGR_GR_FS_INFO_PARTITION_CHIPLET_SYSPIPE_IDS_PARAMS_v1A_1D),
    ('dmGpcMaskData', struct_NV2080_CTRL_GRMGR_GR_FS_INFO_PROFILER_MON_GPC_MASK_PARAMS_v1A_1D),
    ('partitionSyspipeIdData', struct_NV2080_CTRL_GRMGR_GR_FS_INFO_PARTITION_SYSPIPE_ID_PARAMS_v1A_1D),
    ('ropMaskData', struct_NV2080_CTRL_GRMGR_GR_FS_INFO_ROP_MASK_PARAMS_v1A_1D),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

struct_NV2080_CTRL_GRMGR_GR_FS_INFO_QUERY_PARAMS_v1A_1D._pack_ = 1 # source:False
struct_NV2080_CTRL_GRMGR_GR_FS_INFO_QUERY_PARAMS_v1A_1D._fields_ = [
    ('queryType', ctypes.c_uint16),
    ('reserved', ctypes.c_ubyte * 2),
    ('status', ctypes.c_uint32),
    ('queryData', union_NV2080_CTRL_GRMGR_GR_FS_INFO_QUERY_DATA_v1A_1D),
]

struct_NV2080_CTRL_GRMGR_GET_GR_FS_INFO_PARAMS_v1A_1D._pack_ = 1 # source:False
struct_NV2080_CTRL_GRMGR_GET_GR_FS_INFO_PARAMS_v1A_1D._fields_ = [
    ('numQueries', ctypes.c_uint16),
    ('reserved', ctypes.c_ubyte * 6),
    ('queries', struct_NV2080_CTRL_GRMGR_GR_FS_INFO_QUERY_PARAMS_v1A_1D * 96),
]

struct_rpc_ctrl_grmgr_get_gr_fs_info_v1A_1D._pack_ = 1 # source:False
struct_rpc_ctrl_grmgr_get_gr_fs_info_v1A_1D._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('params', struct_NV2080_CTRL_GRMGR_GET_GR_FS_INFO_PARAMS_v1A_1D),
]

rpc_ctrl_grmgr_get_gr_fs_info_v1A_1D = struct_rpc_ctrl_grmgr_get_gr_fs_info_v1A_1D
rpc_ctrl_grmgr_get_gr_fs_info_v = struct_rpc_ctrl_grmgr_get_gr_fs_info_v1A_1D
class struct_rpc_ctrl_stop_channel_v1A_1E(Structure):
    pass

class struct_NVA06F_CTRL_STOP_CHANNEL_PARAMS_v1A_1E(Structure):
    pass

struct_NVA06F_CTRL_STOP_CHANNEL_PARAMS_v1A_1E._pack_ = 1 # source:False
struct_NVA06F_CTRL_STOP_CHANNEL_PARAMS_v1A_1E._fields_ = [
    ('bImmediate', ctypes.c_ubyte),
]

struct_rpc_ctrl_stop_channel_v1A_1E._pack_ = 1 # source:False
struct_rpc_ctrl_stop_channel_v1A_1E._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('params', struct_NVA06F_CTRL_STOP_CHANNEL_PARAMS_v1A_1E),
    ('PADDING_0', ctypes.c_ubyte * 3),
]

rpc_ctrl_stop_channel_v1A_1E = struct_rpc_ctrl_stop_channel_v1A_1E
rpc_ctrl_stop_channel_v = struct_rpc_ctrl_stop_channel_v1A_1E
class struct_rpc_ctrl_gr_pc_sampling_mode_v1A_1F(Structure):
    pass

class struct_NV2080_CTRL_GR_PC_SAMPLING_MODE_PARAMS_v1A_1F(Structure):
    pass

struct_NV2080_CTRL_GR_PC_SAMPLING_MODE_PARAMS_v1A_1F._pack_ = 1 # source:False
struct_NV2080_CTRL_GR_PC_SAMPLING_MODE_PARAMS_v1A_1F._fields_ = [
    ('hChannel', ctypes.c_uint32),
    ('samplingMode', ctypes.c_uint32),
    ('grRouteInfo', struct_NV2080_CTRL_GR_ROUTE_INFO_v12_01),
]

struct_rpc_ctrl_gr_pc_sampling_mode_v1A_1F._pack_ = 1 # source:False
struct_rpc_ctrl_gr_pc_sampling_mode_v1A_1F._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('params', struct_NV2080_CTRL_GR_PC_SAMPLING_MODE_PARAMS_v1A_1F),
]

rpc_ctrl_gr_pc_sampling_mode_v1A_1F = struct_rpc_ctrl_gr_pc_sampling_mode_v1A_1F
rpc_ctrl_gr_pc_sampling_mode_v = struct_rpc_ctrl_gr_pc_sampling_mode_v1A_1F
class struct_rpc_ctrl_perf_rated_tdp_get_status_v1A_1F(Structure):
    pass

class struct_NV2080_CTRL_PERF_RATED_TDP_STATUS_PARAMS_v1A_1F(Structure):
    pass

class struct_PERF_RATED_TDP_RM_INTERNAL_STATE_STRUCT_v1A_1F(Structure):
    pass

struct_PERF_RATED_TDP_RM_INTERNAL_STATE_STRUCT_v1A_1F._pack_ = 1 # source:False
struct_PERF_RATED_TDP_RM_INTERNAL_STATE_STRUCT_v1A_1F._fields_ = [
    ('clientActiveMask', ctypes.c_uint32),
    ('bRegkeyLimitRatedTdp', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 3),
]


# values for enumeration 'NV2080_CTRL_PERF_RATED_TDP_ACTION'
NV2080_CTRL_PERF_RATED_TDP_ACTION__enumvalues = {
    0: 'NV2080_CTRL_PERF_RATED_TDP_ACTION_DEFAULT',
    1: 'NV2080_CTRL_PERF_RATED_TDP_ACTION_FORCE_EXCEED',
    2: 'NV2080_CTRL_PERF_RATED_TDP_ACTION_FORCE_LIMIT',
    3: 'NV2080_CTRL_PERF_RATED_TDP_ACTION_FORCE_LOCK',
    4: 'NV2080_CTRL_PERF_RATED_TDP_ACTION_FORCE_FLOOR',
}
NV2080_CTRL_PERF_RATED_TDP_ACTION_DEFAULT = 0
NV2080_CTRL_PERF_RATED_TDP_ACTION_FORCE_EXCEED = 1
NV2080_CTRL_PERF_RATED_TDP_ACTION_FORCE_LIMIT = 2
NV2080_CTRL_PERF_RATED_TDP_ACTION_FORCE_LOCK = 3
NV2080_CTRL_PERF_RATED_TDP_ACTION_FORCE_FLOOR = 4
NV2080_CTRL_PERF_RATED_TDP_ACTION = ctypes.c_uint32 # enum
struct_NV2080_CTRL_PERF_RATED_TDP_STATUS_PARAMS_v1A_1F._pack_ = 1 # source:False
struct_NV2080_CTRL_PERF_RATED_TDP_STATUS_PARAMS_v1A_1F._fields_ = [
    ('rm', struct_PERF_RATED_TDP_RM_INTERNAL_STATE_STRUCT_v1A_1F),
    ('output', NV2080_CTRL_PERF_RATED_TDP_ACTION),
    ('inputs', NV2080_CTRL_PERF_RATED_TDP_ACTION * 5),
]

struct_rpc_ctrl_perf_rated_tdp_get_status_v1A_1F._pack_ = 1 # source:False
struct_rpc_ctrl_perf_rated_tdp_get_status_v1A_1F._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('params', struct_NV2080_CTRL_PERF_RATED_TDP_STATUS_PARAMS_v1A_1F),
]

rpc_ctrl_perf_rated_tdp_get_status_v1A_1F = struct_rpc_ctrl_perf_rated_tdp_get_status_v1A_1F
rpc_ctrl_perf_rated_tdp_get_status_v = struct_rpc_ctrl_perf_rated_tdp_get_status_v1A_1F
class struct_rpc_ctrl_perf_rated_tdp_set_control_v1A_1F(Structure):
    pass

class struct_NV2080_CTRL_PERF_RATED_TDP_CONTROL_PARAMS_v1A_1F(Structure):
    pass


# values for enumeration 'NV2080_CTRL_PERF_RATED_TDP_CLIENT'
NV2080_CTRL_PERF_RATED_TDP_CLIENT__enumvalues = {
    0: 'NV2080_CTRL_PERF_RATED_TDP_CLIENT_RM',
    1: 'NV2080_CTRL_PERF_RATED_TDP_CLIENT_WAR_BUG_1785342',
    2: 'NV2080_CTRL_PERF_RATED_TDP_CLIENT_GLOBAL',
    3: 'NV2080_CTRL_PERF_RATED_TDP_CLIENT_OS',
    4: 'NV2080_CTRL_PERF_RATED_TDP_CLIENT_PROFILE',
    5: 'NV2080_CTRL_PERF_RATED_TDP_CLIENT_NUM_CLIENTS',
}
NV2080_CTRL_PERF_RATED_TDP_CLIENT_RM = 0
NV2080_CTRL_PERF_RATED_TDP_CLIENT_WAR_BUG_1785342 = 1
NV2080_CTRL_PERF_RATED_TDP_CLIENT_GLOBAL = 2
NV2080_CTRL_PERF_RATED_TDP_CLIENT_OS = 3
NV2080_CTRL_PERF_RATED_TDP_CLIENT_PROFILE = 4
NV2080_CTRL_PERF_RATED_TDP_CLIENT_NUM_CLIENTS = 5
NV2080_CTRL_PERF_RATED_TDP_CLIENT = ctypes.c_uint32 # enum
struct_NV2080_CTRL_PERF_RATED_TDP_CONTROL_PARAMS_v1A_1F._pack_ = 1 # source:False
struct_NV2080_CTRL_PERF_RATED_TDP_CONTROL_PARAMS_v1A_1F._fields_ = [
    ('client', NV2080_CTRL_PERF_RATED_TDP_CLIENT),
    ('input', NV2080_CTRL_PERF_RATED_TDP_ACTION),
]

struct_rpc_ctrl_perf_rated_tdp_set_control_v1A_1F._pack_ = 1 # source:False
struct_rpc_ctrl_perf_rated_tdp_set_control_v1A_1F._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('params', struct_NV2080_CTRL_PERF_RATED_TDP_CONTROL_PARAMS_v1A_1F),
]

rpc_ctrl_perf_rated_tdp_set_control_v1A_1F = struct_rpc_ctrl_perf_rated_tdp_set_control_v1A_1F
rpc_ctrl_perf_rated_tdp_set_control_v = struct_rpc_ctrl_perf_rated_tdp_set_control_v1A_1F
class struct_rpc_ctrl_timer_set_gr_tick_freq_v1A_1F(Structure):
    pass

class struct_NV2080_CTRL_CMD_TIMER_SET_GR_TICK_FREQ_PARAMS_v1A_1F(Structure):
    pass

struct_NV2080_CTRL_CMD_TIMER_SET_GR_TICK_FREQ_PARAMS_v1A_1F._pack_ = 1 # source:False
struct_NV2080_CTRL_CMD_TIMER_SET_GR_TICK_FREQ_PARAMS_v1A_1F._fields_ = [
    ('bSetMaxFreq', ctypes.c_ubyte),
]

struct_rpc_ctrl_timer_set_gr_tick_freq_v1A_1F._pack_ = 1 # source:False
struct_rpc_ctrl_timer_set_gr_tick_freq_v1A_1F._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('params', struct_NV2080_CTRL_CMD_TIMER_SET_GR_TICK_FREQ_PARAMS_v1A_1F),
    ('PADDING_0', ctypes.c_ubyte * 3),
]

rpc_ctrl_timer_set_gr_tick_freq_v1A_1F = struct_rpc_ctrl_timer_set_gr_tick_freq_v1A_1F
rpc_ctrl_timer_set_gr_tick_freq_v = struct_rpc_ctrl_timer_set_gr_tick_freq_v1A_1F
class struct_rpc_ctrl_free_pma_stream_v1A_1F(Structure):
    pass

class struct_NVB0CC_CTRL_FREE_PMA_STREAM_PARAMS_v1A_1F(Structure):
    pass

struct_NVB0CC_CTRL_FREE_PMA_STREAM_PARAMS_v1A_1F._pack_ = 1 # source:False
struct_NVB0CC_CTRL_FREE_PMA_STREAM_PARAMS_v1A_1F._fields_ = [
    ('pmaChannelIdx', ctypes.c_uint32),
]

struct_rpc_ctrl_free_pma_stream_v1A_1F._pack_ = 1 # source:False
struct_rpc_ctrl_free_pma_stream_v1A_1F._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('params', struct_NVB0CC_CTRL_FREE_PMA_STREAM_PARAMS_v1A_1F),
]

rpc_ctrl_free_pma_stream_v1A_1F = struct_rpc_ctrl_free_pma_stream_v1A_1F
rpc_ctrl_free_pma_stream_v = struct_rpc_ctrl_free_pma_stream_v1A_1F
class struct_rpc_ctrl_fifo_setup_vf_zombie_subctx_pdb_v1A_23(Structure):
    pass

class struct_NV2080_CTRL_FIFO_SETUP_VF_ZOMBIE_SUBCTX_PDB_PARAMS_v1A_23(Structure):
    pass

struct_NV2080_CTRL_FIFO_SETUP_VF_ZOMBIE_SUBCTX_PDB_PARAMS_v1A_23._pack_ = 1 # source:False
struct_NV2080_CTRL_FIFO_SETUP_VF_ZOMBIE_SUBCTX_PDB_PARAMS_v1A_23._fields_ = [
    ('base', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
    ('addressSpace', ctypes.c_uint32),
    ('cacheAttrib', ctypes.c_uint32),
]

struct_rpc_ctrl_fifo_setup_vf_zombie_subctx_pdb_v1A_23._pack_ = 1 # source:False
struct_rpc_ctrl_fifo_setup_vf_zombie_subctx_pdb_v1A_23._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('params', struct_NV2080_CTRL_FIFO_SETUP_VF_ZOMBIE_SUBCTX_PDB_PARAMS_v1A_23),
]

rpc_ctrl_fifo_setup_vf_zombie_subctx_pdb_v1A_23 = struct_rpc_ctrl_fifo_setup_vf_zombie_subctx_pdb_v1A_23
rpc_ctrl_fifo_setup_vf_zombie_subctx_pdb_v = struct_rpc_ctrl_fifo_setup_vf_zombie_subctx_pdb_v1A_23
class struct_rpc_ctrl_dbg_set_single_sm_single_step_v1C_02(Structure):
    pass

class struct_NV83DE_CTRL_DEBUG_SET_SINGLE_SM_SINGLE_STEP_PARAMS_v1C_02(Structure):
    pass

struct_NV83DE_CTRL_DEBUG_SET_SINGLE_SM_SINGLE_STEP_PARAMS_v1C_02._pack_ = 1 # source:False
struct_NV83DE_CTRL_DEBUG_SET_SINGLE_SM_SINGLE_STEP_PARAMS_v1C_02._fields_ = [
    ('smID', ctypes.c_uint32),
    ('bSingleStep', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 3),
]

struct_rpc_ctrl_dbg_set_single_sm_single_step_v1C_02._pack_ = 1 # source:False
struct_rpc_ctrl_dbg_set_single_sm_single_step_v1C_02._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('params', struct_NV83DE_CTRL_DEBUG_SET_SINGLE_SM_SINGLE_STEP_PARAMS_v1C_02),
]

rpc_ctrl_dbg_set_single_sm_single_step_v1C_02 = struct_rpc_ctrl_dbg_set_single_sm_single_step_v1C_02
rpc_ctrl_dbg_set_single_sm_single_step_v = struct_rpc_ctrl_dbg_set_single_sm_single_step_v1C_02
class struct_rpc_ctrl_gr_get_tpc_partition_mode_v1C_04(Structure):
    pass

class struct_NV0080_CTRL_GR_TPC_PARTITION_MODE_PARAMS_v1C_04(Structure):
    pass


# values for enumeration 'NV0080_CTRL_GR_TPC_PARTITION_MODE'
NV0080_CTRL_GR_TPC_PARTITION_MODE__enumvalues = {
    0: 'NV0080_CTRL_GR_TPC_PARTITION_MODE_NONE',
    1: 'NV0080_CTRL_GR_TPC_PARTITION_MODE_STATIC',
    2: 'NV0080_CTRL_GR_TPC_PARTITION_MODE_DYNAMIC',
}
NV0080_CTRL_GR_TPC_PARTITION_MODE_NONE = 0
NV0080_CTRL_GR_TPC_PARTITION_MODE_STATIC = 1
NV0080_CTRL_GR_TPC_PARTITION_MODE_DYNAMIC = 2
NV0080_CTRL_GR_TPC_PARTITION_MODE = ctypes.c_uint32 # enum
struct_NV0080_CTRL_GR_TPC_PARTITION_MODE_PARAMS_v1C_04._pack_ = 1 # source:False
struct_NV0080_CTRL_GR_TPC_PARTITION_MODE_PARAMS_v1C_04._fields_ = [
    ('hChannelGroup', ctypes.c_uint32),
    ('mode', NV0080_CTRL_GR_TPC_PARTITION_MODE),
    ('bEnableAllTpcs', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 7),
    ('grRouteInfo', struct_NV2080_CTRL_GR_ROUTE_INFO_v12_01),
]

struct_rpc_ctrl_gr_get_tpc_partition_mode_v1C_04._pack_ = 1 # source:False
struct_rpc_ctrl_gr_get_tpc_partition_mode_v1C_04._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('params', struct_NV0080_CTRL_GR_TPC_PARTITION_MODE_PARAMS_v1C_04),
]

rpc_ctrl_gr_get_tpc_partition_mode_v1C_04 = struct_rpc_ctrl_gr_get_tpc_partition_mode_v1C_04
rpc_ctrl_gr_get_tpc_partition_mode_v = struct_rpc_ctrl_gr_get_tpc_partition_mode_v1C_04
class struct_rpc_ctrl_gr_set_tpc_partition_mode_v1C_04(Structure):
    pass

struct_rpc_ctrl_gr_set_tpc_partition_mode_v1C_04._pack_ = 1 # source:False
struct_rpc_ctrl_gr_set_tpc_partition_mode_v1C_04._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('params', struct_NV0080_CTRL_GR_TPC_PARTITION_MODE_PARAMS_v1C_04),
]

rpc_ctrl_gr_set_tpc_partition_mode_v1C_04 = struct_rpc_ctrl_gr_set_tpc_partition_mode_v1C_04
rpc_ctrl_gr_set_tpc_partition_mode_v = struct_rpc_ctrl_gr_set_tpc_partition_mode_v1C_04
class struct_rpc_ctrl_internal_promote_fault_method_buffers_v1E_07(Structure):
    pass

class struct_NVA06C_CTRL_INTERNAL_PROMOTE_FAULT_METHOD_BUFFERS_PARAMS_v1E_07(Structure):
    pass

class struct_NV2080_CTRL_INTERNAL_MEMDESC_INFO_v1E_07(Structure):
    pass

struct_NV2080_CTRL_INTERNAL_MEMDESC_INFO_v1E_07._pack_ = 1 # source:False
struct_NV2080_CTRL_INTERNAL_MEMDESC_INFO_v1E_07._fields_ = [
    ('base', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
    ('alignment', ctypes.c_uint64),
    ('addressSpace', ctypes.c_uint32),
    ('cpuCacheAttrib', ctypes.c_uint32),
]

struct_NVA06C_CTRL_INTERNAL_PROMOTE_FAULT_METHOD_BUFFERS_PARAMS_v1E_07._pack_ = 1 # source:False
struct_NVA06C_CTRL_INTERNAL_PROMOTE_FAULT_METHOD_BUFFERS_PARAMS_v1E_07._fields_ = [
    ('methodBufferMemdesc', struct_NV2080_CTRL_INTERNAL_MEMDESC_INFO_v1E_07 * 2),
    ('bar2Addr', ctypes.c_uint64 * 2),
    ('numValidEntries', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

struct_rpc_ctrl_internal_promote_fault_method_buffers_v1E_07._pack_ = 1 # source:False
struct_rpc_ctrl_internal_promote_fault_method_buffers_v1E_07._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('params', struct_NVA06C_CTRL_INTERNAL_PROMOTE_FAULT_METHOD_BUFFERS_PARAMS_v1E_07),
]

rpc_ctrl_internal_promote_fault_method_buffers_v1E_07 = struct_rpc_ctrl_internal_promote_fault_method_buffers_v1E_07
rpc_ctrl_internal_promote_fault_method_buffers_v = struct_rpc_ctrl_internal_promote_fault_method_buffers_v1E_07
class struct_rpc_ctrl_internal_memsys_set_zbc_referenced_v1F_05(Structure):
    pass

class struct_NV2080_CTRL_CMD_INTERNAL_MEMSYS_SET_ZBC_REFERENCED_v1F_05(Structure):
    pass

struct_NV2080_CTRL_CMD_INTERNAL_MEMSYS_SET_ZBC_REFERENCED_v1F_05._pack_ = 1 # source:False
struct_NV2080_CTRL_CMD_INTERNAL_MEMSYS_SET_ZBC_REFERENCED_v1F_05._fields_ = [
    ('bZbcSurfacesExist', ctypes.c_ubyte),
]

struct_rpc_ctrl_internal_memsys_set_zbc_referenced_v1F_05._pack_ = 1 # source:False
struct_rpc_ctrl_internal_memsys_set_zbc_referenced_v1F_05._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('params', struct_NV2080_CTRL_CMD_INTERNAL_MEMSYS_SET_ZBC_REFERENCED_v1F_05),
    ('PADDING_0', ctypes.c_ubyte * 3),
]

rpc_ctrl_internal_memsys_set_zbc_referenced_v1F_05 = struct_rpc_ctrl_internal_memsys_set_zbc_referenced_v1F_05
rpc_ctrl_internal_memsys_set_zbc_referenced_v = struct_rpc_ctrl_internal_memsys_set_zbc_referenced_v1F_05
class struct_rpc_ctrl_fabric_memory_describe_v1E_0C(Structure):
    pass

class struct_NV00F8_CTRL_DESCRIBE_PARAMS_v1E_0C(Structure):
    pass

struct_NV00F8_CTRL_DESCRIBE_PARAMS_v1E_0C._pack_ = 1 # source:False
struct_NV00F8_CTRL_DESCRIBE_PARAMS_v1E_0C._fields_ = [
    ('offset', ctypes.c_uint64),
    ('totalPfns', ctypes.c_uint64),
    ('pfnArray', ctypes.c_uint32 * 512),
    ('numPfns', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

struct_rpc_ctrl_fabric_memory_describe_v1E_0C._pack_ = 1 # source:False
struct_rpc_ctrl_fabric_memory_describe_v1E_0C._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('params', struct_NV00F8_CTRL_DESCRIBE_PARAMS_v1E_0C),
]

rpc_ctrl_fabric_memory_describe_v1E_0C = struct_rpc_ctrl_fabric_memory_describe_v1E_0C
rpc_ctrl_fabric_memory_describe_v = struct_rpc_ctrl_fabric_memory_describe_v1E_0C
class struct_rpc_ctrl_fabric_mem_stats_v1E_0C(Structure):
    pass

class struct_NV2080_CTRL_FLA_GET_FABRIC_MEM_STATS_PARAMS_v1E_0C(Structure):
    pass

struct_NV2080_CTRL_FLA_GET_FABRIC_MEM_STATS_PARAMS_v1E_0C._pack_ = 1 # source:False
struct_NV2080_CTRL_FLA_GET_FABRIC_MEM_STATS_PARAMS_v1E_0C._fields_ = [
    ('totalSize', ctypes.c_uint64),
    ('freeSize', ctypes.c_uint64),
]

struct_rpc_ctrl_fabric_mem_stats_v1E_0C._pack_ = 1 # source:False
struct_rpc_ctrl_fabric_mem_stats_v1E_0C._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('params', struct_NV2080_CTRL_FLA_GET_FABRIC_MEM_STATS_PARAMS_v1E_0C),
]

rpc_ctrl_fabric_mem_stats_v1E_0C = struct_rpc_ctrl_fabric_mem_stats_v1E_0C
rpc_ctrl_fabric_mem_stats_v = struct_rpc_ctrl_fabric_mem_stats_v1E_0C
class struct_rpc_ctrl_bus_set_p2p_mapping_v21_03(Structure):
    pass

class struct_NV2080_CTRL_BUS_SET_P2P_MAPPING_PARAMS_v21_03(Structure):
    pass

struct_NV2080_CTRL_BUS_SET_P2P_MAPPING_PARAMS_v21_03._pack_ = 1 # source:False
struct_NV2080_CTRL_BUS_SET_P2P_MAPPING_PARAMS_v21_03._fields_ = [
    ('connectionType', ctypes.c_uint32),
    ('peerId', ctypes.c_uint32),
    ('bSpaAccessOnly', ctypes.c_uint32),
    ('bUseUuid', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 3),
    ('remoteGpuId', ctypes.c_uint32),
    ('remoteGpuUuid', ctypes.c_ubyte * 16),
]

struct_rpc_ctrl_bus_set_p2p_mapping_v21_03._pack_ = 1 # source:False
struct_rpc_ctrl_bus_set_p2p_mapping_v21_03._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('params', struct_NV2080_CTRL_BUS_SET_P2P_MAPPING_PARAMS_v21_03),
]

rpc_ctrl_bus_set_p2p_mapping_v21_03 = struct_rpc_ctrl_bus_set_p2p_mapping_v21_03
class struct_rpc_ctrl_bus_set_p2p_mapping_v29_08(Structure):
    pass

class struct_NV2080_CTRL_BUS_SET_P2P_MAPPING_PARAMS_v29_08(Structure):
    pass

struct_NV2080_CTRL_BUS_SET_P2P_MAPPING_PARAMS_v29_08._pack_ = 1 # source:False
struct_NV2080_CTRL_BUS_SET_P2P_MAPPING_PARAMS_v29_08._fields_ = [
    ('connectionType', ctypes.c_uint32),
    ('peerId', ctypes.c_uint32),
    ('bEgmPeer', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 3),
    ('bSpaAccessOnly', ctypes.c_uint32),
    ('bUseUuid', ctypes.c_ubyte),
    ('PADDING_1', ctypes.c_ubyte * 3),
    ('remoteGpuId', ctypes.c_uint32),
    ('remoteGpuUuid', ctypes.c_ubyte * 16),
]

struct_rpc_ctrl_bus_set_p2p_mapping_v29_08._pack_ = 1 # source:False
struct_rpc_ctrl_bus_set_p2p_mapping_v29_08._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('params', struct_NV2080_CTRL_BUS_SET_P2P_MAPPING_PARAMS_v29_08),
]

rpc_ctrl_bus_set_p2p_mapping_v29_08 = struct_rpc_ctrl_bus_set_p2p_mapping_v29_08
rpc_ctrl_bus_set_p2p_mapping_v = struct_rpc_ctrl_bus_set_p2p_mapping_v29_08
class struct_rpc_ctrl_bus_unset_p2p_mapping_v21_03(Structure):
    pass

class struct_NV2080_CTRL_BUS_UNSET_P2P_MAPPING_PARAMS_v21_03(Structure):
    pass

struct_NV2080_CTRL_BUS_UNSET_P2P_MAPPING_PARAMS_v21_03._pack_ = 1 # source:False
struct_NV2080_CTRL_BUS_UNSET_P2P_MAPPING_PARAMS_v21_03._fields_ = [
    ('connectionType', ctypes.c_uint32),
    ('peerId', ctypes.c_uint32),
    ('bUseUuid', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 3),
    ('remoteGpuId', ctypes.c_uint32),
    ('remoteGpuUuid', ctypes.c_ubyte * 16),
]

struct_rpc_ctrl_bus_unset_p2p_mapping_v21_03._pack_ = 1 # source:False
struct_rpc_ctrl_bus_unset_p2p_mapping_v21_03._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('params', struct_NV2080_CTRL_BUS_UNSET_P2P_MAPPING_PARAMS_v21_03),
]

rpc_ctrl_bus_unset_p2p_mapping_v21_03 = struct_rpc_ctrl_bus_unset_p2p_mapping_v21_03
rpc_ctrl_bus_unset_p2p_mapping_v = struct_rpc_ctrl_bus_unset_p2p_mapping_v21_03
class struct_rpc_ctrl_gpu_get_info_v2_v25_11(Structure):
    pass

class struct_NV2080_CTRL_GPU_GET_INFO_V2_PARAMS_v25_11(Structure):
    pass

class struct_NV2080_CTRL_GPU_INFO_v25_11(Structure):
    pass

struct_NV2080_CTRL_GPU_INFO_v25_11._pack_ = 1 # source:False
struct_NV2080_CTRL_GPU_INFO_v25_11._fields_ = [
    ('index', ctypes.c_uint32),
    ('data', ctypes.c_uint32),
]

struct_NV2080_CTRL_GPU_GET_INFO_V2_PARAMS_v25_11._pack_ = 1 # source:False
struct_NV2080_CTRL_GPU_GET_INFO_V2_PARAMS_v25_11._fields_ = [
    ('gpuInfoListSize', ctypes.c_uint32),
    ('gpuInfoList', struct_NV2080_CTRL_GPU_INFO_v25_11 * 65),
]

struct_rpc_ctrl_gpu_get_info_v2_v25_11._pack_ = 1 # source:False
struct_rpc_ctrl_gpu_get_info_v2_v25_11._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('params', struct_NV2080_CTRL_GPU_GET_INFO_V2_PARAMS_v25_11),
]

rpc_ctrl_gpu_get_info_v2_v25_11 = struct_rpc_ctrl_gpu_get_info_v2_v25_11
rpc_ctrl_gpu_get_info_v2_v = struct_rpc_ctrl_gpu_get_info_v2_v25_11
class struct_rpc_update_gpm_guest_buffer_info_v27_01(Structure):
    pass

struct_rpc_update_gpm_guest_buffer_info_v27_01._pack_ = 1 # source:False
struct_rpc_update_gpm_guest_buffer_info_v27_01._fields_ = [
    ('gpfn', ctypes.c_uint64),
    ('swizzId', ctypes.c_uint32),
    ('computeId', ctypes.c_uint32),
    ('bufSize', ctypes.c_uint32),
    ('bMap', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 3),
]

rpc_update_gpm_guest_buffer_info_v27_01 = struct_rpc_update_gpm_guest_buffer_info_v27_01
rpc_update_gpm_guest_buffer_info_v = struct_rpc_update_gpm_guest_buffer_info_v27_01
class struct_rpc_ctrl_internal_quiesce_pma_channel_v1C_08(Structure):
    pass

class struct_NVB0CC_CTRL_INTERNAL_QUIESCE_PMA_CHANNEL_PARAMS_v1C_08(Structure):
    pass

struct_NVB0CC_CTRL_INTERNAL_QUIESCE_PMA_CHANNEL_PARAMS_v1C_08._pack_ = 1 # source:False
struct_NVB0CC_CTRL_INTERNAL_QUIESCE_PMA_CHANNEL_PARAMS_v1C_08._fields_ = [
    ('pmaChannelIdx', ctypes.c_uint32),
    ('bMembytesPollingRequired', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 3),
]

struct_rpc_ctrl_internal_quiesce_pma_channel_v1C_08._pack_ = 1 # source:False
struct_rpc_ctrl_internal_quiesce_pma_channel_v1C_08._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('params', struct_NVB0CC_CTRL_INTERNAL_QUIESCE_PMA_CHANNEL_PARAMS_v1C_08),
]

rpc_ctrl_internal_quiesce_pma_channel_v1C_08 = struct_rpc_ctrl_internal_quiesce_pma_channel_v1C_08
rpc_ctrl_internal_quiesce_pma_channel_v = struct_rpc_ctrl_internal_quiesce_pma_channel_v1C_08
class struct_rpc_ctrl_internal_sriov_promote_pma_stream_v1C_0C(Structure):
    pass

class struct_NVB0CC_CTRL_INTERNAL_SRIOV_PROMOTE_PMA_STREAM_PARAMS_v1C_0C(Structure):
    pass

struct_NVB0CC_CTRL_INTERNAL_SRIOV_PROMOTE_PMA_STREAM_PARAMS_v1C_0C._pack_ = 1 # source:False
struct_NVB0CC_CTRL_INTERNAL_SRIOV_PROMOTE_PMA_STREAM_PARAMS_v1C_0C._fields_ = [
    ('pmaChannelIdx', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('pmaBufferVA', ctypes.c_uint64),
    ('pmaBufferSize', ctypes.c_uint64),
    ('membytesVA', ctypes.c_uint64),
    ('hwpmIBPA', ctypes.c_uint64),
    ('hwpmIBAperture', ctypes.c_ubyte),
    ('PADDING_1', ctypes.c_ubyte * 7),
]

struct_rpc_ctrl_internal_sriov_promote_pma_stream_v1C_0C._pack_ = 1 # source:False
struct_rpc_ctrl_internal_sriov_promote_pma_stream_v1C_0C._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('params', struct_NVB0CC_CTRL_INTERNAL_SRIOV_PROMOTE_PMA_STREAM_PARAMS_v1C_0C),
]

rpc_ctrl_internal_sriov_promote_pma_stream_v1C_0C = struct_rpc_ctrl_internal_sriov_promote_pma_stream_v1C_0C
rpc_ctrl_internal_sriov_promote_pma_stream_v = struct_rpc_ctrl_internal_sriov_promote_pma_stream_v1C_0C
class struct_rpc_ctrl_exec_partitions_create_v24_05(Structure):
    pass

class struct_NVC637_CTRL_EXEC_PARTITIONS_CREATE_PARAMS_v24_05(Structure):
    pass

class struct_NVC637_CTRL_EXEC_PARTITIONS_INFO_v24_05(Structure):
    pass

struct_NVC637_CTRL_EXEC_PARTITIONS_INFO_v24_05._pack_ = 1 # source:False
struct_NVC637_CTRL_EXEC_PARTITIONS_INFO_v24_05._fields_ = [
    ('gpcCount', ctypes.c_uint32),
    ('gfxGpcCount', ctypes.c_uint32),
    ('veidCount', ctypes.c_uint32),
    ('ceCount', ctypes.c_uint32),
    ('nvEncCount', ctypes.c_uint32),
    ('nvDecCount', ctypes.c_uint32),
    ('nvJpgCount', ctypes.c_uint32),
    ('ofaCount', ctypes.c_uint32),
    ('sharedEngFlag', ctypes.c_uint32),
    ('smCount', ctypes.c_uint32),
    ('spanStart', ctypes.c_uint32),
    ('computeSize', ctypes.c_uint32),
]

struct_NVC637_CTRL_EXEC_PARTITIONS_CREATE_PARAMS_v24_05._pack_ = 1 # source:False
struct_NVC637_CTRL_EXEC_PARTITIONS_CREATE_PARAMS_v24_05._fields_ = [
    ('bQuery', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 3),
    ('execPartCount', ctypes.c_uint32),
    ('execPartInfo', struct_NVC637_CTRL_EXEC_PARTITIONS_INFO_v24_05 * 8),
    ('execPartId', ctypes.c_uint32 * 8),
]

struct_rpc_ctrl_exec_partitions_create_v24_05._pack_ = 1 # source:False
struct_rpc_ctrl_exec_partitions_create_v24_05._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('status', ctypes.c_uint32),
    ('execPartitionsCreate', struct_NVC637_CTRL_EXEC_PARTITIONS_CREATE_PARAMS_v24_05),
]

rpc_ctrl_exec_partitions_create_v24_05 = struct_rpc_ctrl_exec_partitions_create_v24_05
rpc_ctrl_exec_partitions_create_v = struct_rpc_ctrl_exec_partitions_create_v24_05
class struct_rpc_ctrl_fla_setup_instance_mem_block_v21_05(Structure):
    pass

class struct_NV2080_CTRL_FLA_SETUP_INSTANCE_MEM_BLOCK_PARAMS_v13_04(Structure):
    pass

struct_NV2080_CTRL_FLA_SETUP_INSTANCE_MEM_BLOCK_PARAMS_v13_04._pack_ = 1 # source:False
struct_NV2080_CTRL_FLA_SETUP_INSTANCE_MEM_BLOCK_PARAMS_v13_04._fields_ = [
    ('imbPhysAddr', ctypes.c_uint64),
    ('addrSpace', ctypes.c_uint32),
    ('flaAction', ctypes.c_uint32),
]

struct_rpc_ctrl_fla_setup_instance_mem_block_v21_05._pack_ = 1 # source:False
struct_rpc_ctrl_fla_setup_instance_mem_block_v21_05._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('params', struct_NV2080_CTRL_FLA_SETUP_INSTANCE_MEM_BLOCK_PARAMS_v13_04),
]

rpc_ctrl_fla_setup_instance_mem_block_v21_05 = struct_rpc_ctrl_fla_setup_instance_mem_block_v21_05
rpc_ctrl_fla_setup_instance_mem_block_v = struct_rpc_ctrl_fla_setup_instance_mem_block_v21_05
class struct_rpc_ctrl_get_total_hs_credits_v21_08(Structure):
    pass

class struct_NVB0CC_CTRL_GET_TOTAL_HS_CREDITS_PARAMS_v21_08(Structure):
    pass

struct_NVB0CC_CTRL_GET_TOTAL_HS_CREDITS_PARAMS_v21_08._pack_ = 1 # source:False
struct_NVB0CC_CTRL_GET_TOTAL_HS_CREDITS_PARAMS_v21_08._fields_ = [
    ('numCredits', ctypes.c_uint32),
]

struct_rpc_ctrl_get_total_hs_credits_v21_08._pack_ = 1 # source:False
struct_rpc_ctrl_get_total_hs_credits_v21_08._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('params', struct_NVB0CC_CTRL_GET_TOTAL_HS_CREDITS_PARAMS_v21_08),
]

rpc_ctrl_get_total_hs_credits_v21_08 = struct_rpc_ctrl_get_total_hs_credits_v21_08
rpc_ctrl_get_total_hs_credits_v = struct_rpc_ctrl_get_total_hs_credits_v21_08
class struct_rpc_ctrl_get_hs_credits_v21_08(Structure):
    pass

class struct_NVB0CC_CTRL_GET_HS_CREDITS_PARAMS_v21_08(Structure):
    pass

class struct_NVB0CC_CTRL_PMA_STREAM_HS_CREDITS_STATUS_v21_08(Structure):
    pass

struct_NVB0CC_CTRL_PMA_STREAM_HS_CREDITS_STATUS_v21_08._pack_ = 1 # source:False
struct_NVB0CC_CTRL_PMA_STREAM_HS_CREDITS_STATUS_v21_08._fields_ = [
    ('status', ctypes.c_ubyte),
    ('entryIndex', ctypes.c_ubyte),
]

class struct_NVB0CC_CTRL_PMA_STREAM_HS_CREDITS_INFO_v21_08(Structure):
    pass

struct_NVB0CC_CTRL_PMA_STREAM_HS_CREDITS_INFO_v21_08._pack_ = 1 # source:False
struct_NVB0CC_CTRL_PMA_STREAM_HS_CREDITS_INFO_v21_08._fields_ = [
    ('chipletType', ctypes.c_ubyte),
    ('chipletIndex', ctypes.c_ubyte),
    ('numCredits', ctypes.c_uint16),
]

struct_NVB0CC_CTRL_GET_HS_CREDITS_PARAMS_v21_08._pack_ = 1 # source:False
struct_NVB0CC_CTRL_GET_HS_CREDITS_PARAMS_v21_08._fields_ = [
    ('pmaChannelIdx', ctypes.c_ubyte),
    ('numEntries', ctypes.c_ubyte),
    ('statusInfo', struct_NVB0CC_CTRL_PMA_STREAM_HS_CREDITS_STATUS_v21_08),
    ('creditInfo', struct_NVB0CC_CTRL_PMA_STREAM_HS_CREDITS_INFO_v21_08 * 63),
]

struct_rpc_ctrl_get_hs_credits_v21_08._pack_ = 1 # source:False
struct_rpc_ctrl_get_hs_credits_v21_08._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('params', struct_NVB0CC_CTRL_GET_HS_CREDITS_PARAMS_v21_08),
]

rpc_ctrl_get_hs_credits_v21_08 = struct_rpc_ctrl_get_hs_credits_v21_08
rpc_ctrl_get_hs_credits_v = struct_rpc_ctrl_get_hs_credits_v21_08
class struct_rpc_ctrl_reserve_hes_v29_07(Structure):
    pass

class struct_NVB0CC_CTRL_RESERVE_HES_PARAMS_v29_07(Structure):
    pass

class struct_NVB0CC_CTRL_HES_RESERVATION_UNION_v29_07(Structure):
    pass

class struct_NVB0CC_CTRL_RESERVE_HES_CWD_PARAMS_v29_07(Structure):
    pass

struct_NVB0CC_CTRL_RESERVE_HES_CWD_PARAMS_v29_07._pack_ = 1 # source:False
struct_NVB0CC_CTRL_RESERVE_HES_CWD_PARAMS_v29_07._fields_ = [
    ('ctxsw', ctypes.c_ubyte),
]

struct_NVB0CC_CTRL_HES_RESERVATION_UNION_v29_07._pack_ = 1 # source:False
struct_NVB0CC_CTRL_HES_RESERVATION_UNION_v29_07._fields_ = [
    ('cwd', struct_NVB0CC_CTRL_RESERVE_HES_CWD_PARAMS_v29_07),
]

struct_NVB0CC_CTRL_RESERVE_HES_PARAMS_v29_07._pack_ = 1 # source:False
struct_NVB0CC_CTRL_RESERVE_HES_PARAMS_v29_07._fields_ = [
    ('type', ctypes.c_uint32),
    ('reserveParams', struct_NVB0CC_CTRL_HES_RESERVATION_UNION_v29_07),
    ('PADDING_0', ctypes.c_ubyte * 3),
]

struct_rpc_ctrl_reserve_hes_v29_07._pack_ = 1 # source:False
struct_rpc_ctrl_reserve_hes_v29_07._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('params', struct_NVB0CC_CTRL_RESERVE_HES_PARAMS_v29_07),
]

rpc_ctrl_reserve_hes_v29_07 = struct_rpc_ctrl_reserve_hes_v29_07
rpc_ctrl_reserve_hes_v = struct_rpc_ctrl_reserve_hes_v29_07
class struct_rpc_ctrl_release_hes_v29_07(Structure):
    pass

class struct_NVB0CC_CTRL_RELEASE_HES_PARAMS_v29_07(Structure):
    pass

struct_NVB0CC_CTRL_RELEASE_HES_PARAMS_v29_07._pack_ = 1 # source:False
struct_NVB0CC_CTRL_RELEASE_HES_PARAMS_v29_07._fields_ = [
    ('type', ctypes.c_uint32),
]

struct_rpc_ctrl_release_hes_v29_07._pack_ = 1 # source:False
struct_rpc_ctrl_release_hes_v29_07._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('params', struct_NVB0CC_CTRL_RELEASE_HES_PARAMS_v29_07),
]

rpc_ctrl_release_hes_v29_07 = struct_rpc_ctrl_release_hes_v29_07
rpc_ctrl_release_hes_v = struct_rpc_ctrl_release_hes_v29_07
class struct_rpc_ctrl_reserve_ccu_prof_v29_07(Structure):
    pass

class struct_NVB0CC_CTRL_RESERVE_CCUPROF_PARAMS_v29_07(Structure):
    pass

struct_NVB0CC_CTRL_RESERVE_CCUPROF_PARAMS_v29_07._pack_ = 1 # source:False
struct_NVB0CC_CTRL_RESERVE_CCUPROF_PARAMS_v29_07._fields_ = [
    ('ctxsw', ctypes.c_ubyte),
]

struct_rpc_ctrl_reserve_ccu_prof_v29_07._pack_ = 1 # source:False
struct_rpc_ctrl_reserve_ccu_prof_v29_07._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('params', struct_NVB0CC_CTRL_RESERVE_CCUPROF_PARAMS_v29_07),
    ('PADDING_0', ctypes.c_ubyte * 3),
]

rpc_ctrl_reserve_ccu_prof_v29_07 = struct_rpc_ctrl_reserve_ccu_prof_v29_07
rpc_ctrl_reserve_ccu_prof_v = struct_rpc_ctrl_reserve_ccu_prof_v29_07
class struct_rpc_ctrl_release_ccu_prof_v29_07(Structure):
    pass

struct_rpc_ctrl_release_ccu_prof_v29_07._pack_ = 1 # source:False
struct_rpc_ctrl_release_ccu_prof_v29_07._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
]

rpc_ctrl_release_ccu_prof_v29_07 = struct_rpc_ctrl_release_ccu_prof_v29_07
rpc_ctrl_release_ccu_prof_v = struct_rpc_ctrl_release_ccu_prof_v29_07
class struct_rpc_ctrl_set_hs_credits_v21_08(Structure):
    pass

class struct_NVB0CC_CTRL_SET_HS_CREDITS_PARAMS_v21_08(Structure):
    pass

struct_NVB0CC_CTRL_SET_HS_CREDITS_PARAMS_v21_08._pack_ = 1 # source:False
struct_NVB0CC_CTRL_SET_HS_CREDITS_PARAMS_v21_08._fields_ = [
    ('pmaChannelIdx', ctypes.c_ubyte),
    ('numEntries', ctypes.c_ubyte),
    ('statusInfo', struct_NVB0CC_CTRL_PMA_STREAM_HS_CREDITS_STATUS_v21_08),
    ('creditInfo', struct_NVB0CC_CTRL_PMA_STREAM_HS_CREDITS_INFO_v21_08 * 63),
]

struct_rpc_ctrl_set_hs_credits_v21_08._pack_ = 1 # source:False
struct_rpc_ctrl_set_hs_credits_v21_08._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('params', struct_NVB0CC_CTRL_SET_HS_CREDITS_PARAMS_v21_08),
]

rpc_ctrl_set_hs_credits_v21_08 = struct_rpc_ctrl_set_hs_credits_v21_08
rpc_ctrl_set_hs_credits_v = struct_rpc_ctrl_set_hs_credits_v21_08
class struct_rpc_ctrl_pm_area_pc_sampler_v21_0B(Structure):
    pass

struct_rpc_ctrl_pm_area_pc_sampler_v21_0B._pack_ = 1 # source:False
struct_rpc_ctrl_pm_area_pc_sampler_v21_0B._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('cmd', ctypes.c_uint32),
]

rpc_ctrl_pm_area_pc_sampler_v21_0B = struct_rpc_ctrl_pm_area_pc_sampler_v21_0B
rpc_ctrl_pm_area_pc_sampler_v = struct_rpc_ctrl_pm_area_pc_sampler_v21_0B
class struct_rpc_ctrl_exec_partitions_delete_v1F_0A(Structure):
    pass

class struct_NVC637_CTRL_EXEC_PARTITIONS_DELETE_PARAMS_v18_05(Structure):
    pass

struct_NVC637_CTRL_EXEC_PARTITIONS_DELETE_PARAMS_v18_05._pack_ = 1 # source:False
struct_NVC637_CTRL_EXEC_PARTITIONS_DELETE_PARAMS_v18_05._fields_ = [
    ('execPartCount', ctypes.c_uint32),
    ('execPartId', ctypes.c_uint32 * 8),
]

struct_rpc_ctrl_exec_partitions_delete_v1F_0A._pack_ = 1 # source:False
struct_rpc_ctrl_exec_partitions_delete_v1F_0A._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('execPartitionsDelete', struct_NVC637_CTRL_EXEC_PARTITIONS_DELETE_PARAMS_v18_05),
]

rpc_ctrl_exec_partitions_delete_v1F_0A = struct_rpc_ctrl_exec_partitions_delete_v1F_0A
rpc_ctrl_exec_partitions_delete_v = struct_rpc_ctrl_exec_partitions_delete_v1F_0A
class struct_rpc_ctrl_gpfifo_get_work_submit_token_v1F_0A(Structure):
    pass

class struct_NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS_v08_00(Structure):
    pass

struct_NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS_v08_00._pack_ = 1 # source:False
struct_NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS_v08_00._fields_ = [
    ('workSubmitToken', ctypes.c_uint32),
]

struct_rpc_ctrl_gpfifo_get_work_submit_token_v1F_0A._pack_ = 1 # source:False
struct_rpc_ctrl_gpfifo_get_work_submit_token_v1F_0A._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('workSubmitToken', struct_NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS_v08_00),
]

rpc_ctrl_gpfifo_get_work_submit_token_v1F_0A = struct_rpc_ctrl_gpfifo_get_work_submit_token_v1F_0A
rpc_ctrl_gpfifo_get_work_submit_token_v = struct_rpc_ctrl_gpfifo_get_work_submit_token_v1F_0A
class struct_rpc_ctrl_gpfifo_set_work_submit_token_notif_index_v1F_0A(Structure):
    pass

class struct_NVC36F_CTRL_GPFIFO_SET_WORK_SUBMIT_TOKEN_NOTIF_INDEX_PARAMS_v16_04(Structure):
    pass

struct_NVC36F_CTRL_GPFIFO_SET_WORK_SUBMIT_TOKEN_NOTIF_INDEX_PARAMS_v16_04._pack_ = 1 # source:False
struct_NVC36F_CTRL_GPFIFO_SET_WORK_SUBMIT_TOKEN_NOTIF_INDEX_PARAMS_v16_04._fields_ = [
    ('index', ctypes.c_uint32),
]

struct_rpc_ctrl_gpfifo_set_work_submit_token_notif_index_v1F_0A._pack_ = 1 # source:False
struct_rpc_ctrl_gpfifo_set_work_submit_token_notif_index_v1F_0A._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('setWorkSubmitTokenIndex', struct_NVC36F_CTRL_GPFIFO_SET_WORK_SUBMIT_TOKEN_NOTIF_INDEX_PARAMS_v16_04),
]

rpc_ctrl_gpfifo_set_work_submit_token_notif_index_v1F_0A = struct_rpc_ctrl_gpfifo_set_work_submit_token_notif_index_v1F_0A
rpc_ctrl_gpfifo_set_work_submit_token_notif_index_v = struct_rpc_ctrl_gpfifo_set_work_submit_token_notif_index_v1F_0A
class struct_rpc_ctrl_master_get_virtual_function_error_cont_intr_mask_v1F_0D(Structure):
    pass

class struct_NV90E6_CTRL_MASTER_GET_VIRTUAL_FUNCTION_ERROR_CONT_INTR_MASK_PARAMS_v18_0B(Structure):
    pass

struct_NV90E6_CTRL_MASTER_GET_VIRTUAL_FUNCTION_ERROR_CONT_INTR_MASK_PARAMS_v18_0B._pack_ = 1 # source:False
struct_NV90E6_CTRL_MASTER_GET_VIRTUAL_FUNCTION_ERROR_CONT_INTR_MASK_PARAMS_v18_0B._fields_ = [
    ('eccMask', ctypes.c_uint32),
    ('nvlinkMask', ctypes.c_uint32),
]

struct_rpc_ctrl_master_get_virtual_function_error_cont_intr_mask_v1F_0D._pack_ = 1 # source:False
struct_rpc_ctrl_master_get_virtual_function_error_cont_intr_mask_v1F_0D._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('vfErrContIntrMask', struct_NV90E6_CTRL_MASTER_GET_VIRTUAL_FUNCTION_ERROR_CONT_INTR_MASK_PARAMS_v18_0B),
]

rpc_ctrl_master_get_virtual_function_error_cont_intr_mask_v1F_0D = struct_rpc_ctrl_master_get_virtual_function_error_cont_intr_mask_v1F_0D
rpc_ctrl_master_get_virtual_function_error_cont_intr_mask_v = struct_rpc_ctrl_master_get_virtual_function_error_cont_intr_mask_v1F_0D
class struct_rpc_save_hibernation_data_v1E_0E(Structure):
    pass

struct_rpc_save_hibernation_data_v1E_0E._pack_ = 1 # source:False
struct_rpc_save_hibernation_data_v1E_0E._fields_ = [
    ('remainedBytes', ctypes.c_uint32),
    ('payload', ctypes.c_ubyte * 0),
]

rpc_save_hibernation_data_v1E_0E = struct_rpc_save_hibernation_data_v1E_0E
rpc_save_hibernation_data_v = struct_rpc_save_hibernation_data_v1E_0E
class struct_rpc_restore_hibernation_data_v1E_0E(Structure):
    pass

struct_rpc_restore_hibernation_data_v1E_0E._pack_ = 1 # source:False
struct_rpc_restore_hibernation_data_v1E_0E._fields_ = [
    ('remainedBytes', ctypes.c_uint32),
    ('payload', ctypes.c_ubyte * 0),
]

rpc_restore_hibernation_data_v1E_0E = struct_rpc_restore_hibernation_data_v1E_0E
rpc_restore_hibernation_data_v = struct_rpc_restore_hibernation_data_v1E_0E
class struct_rpc_ctrl_get_mmu_debug_mode_v1E_06(Structure):
    pass

class struct_NV0090_CTRL_GET_MMU_DEBUG_MODE_PARAMS_v1E_06(Structure):
    pass

struct_NV0090_CTRL_GET_MMU_DEBUG_MODE_PARAMS_v1E_06._pack_ = 1 # source:False
struct_NV0090_CTRL_GET_MMU_DEBUG_MODE_PARAMS_v1E_06._fields_ = [
    ('bMode', ctypes.c_ubyte),
]

struct_rpc_ctrl_get_mmu_debug_mode_v1E_06._pack_ = 1 # source:False
struct_rpc_ctrl_get_mmu_debug_mode_v1E_06._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('params', struct_NV0090_CTRL_GET_MMU_DEBUG_MODE_PARAMS_v1E_06),
    ('PADDING_0', ctypes.c_ubyte * 3),
]

rpc_ctrl_get_mmu_debug_mode_v1E_06 = struct_rpc_ctrl_get_mmu_debug_mode_v1E_06
rpc_ctrl_get_mmu_debug_mode_v = struct_rpc_ctrl_get_mmu_debug_mode_v1E_06
class struct_rpc_disable_channels_v1E_0B(Structure):
    pass

struct_rpc_disable_channels_v1E_0B._pack_ = 1 # source:False
struct_rpc_disable_channels_v1E_0B._fields_ = [
    ('bDisable', ctypes.c_uint32),
]

rpc_disable_channels_v1E_0B = struct_rpc_disable_channels_v1E_0B
rpc_disable_channels_v = struct_rpc_disable_channels_v1E_0B
class struct_rpc_ctrl_gpu_migratable_ops_v21_07(Structure):
    pass

class struct_NV2080_CTRL_GPU_MIGRATABLE_OPS_PARAMS_v21_07(Structure):
    pass

struct_NV2080_CTRL_GPU_MIGRATABLE_OPS_PARAMS_v21_07._pack_ = 1 # source:False
struct_NV2080_CTRL_GPU_MIGRATABLE_OPS_PARAMS_v21_07._fields_ = [
    ('hClientTarget', ctypes.c_uint32),
    ('hChannelTarget', ctypes.c_uint32),
    ('bNonTransactional', ctypes.c_uint32),
    ('regOpCount', ctypes.c_uint32),
    ('smIds', ctypes.c_uint32 * 50),
    ('regOps', struct_NV2080_CTRL_GPU_REG_OP_v03_00 * 50),
    ('grRouteInfo', struct_NV2080_CTRL_GR_ROUTE_INFO_v12_01),
]

struct_rpc_ctrl_gpu_migratable_ops_v21_07._pack_ = 1 # source:False
struct_rpc_ctrl_gpu_migratable_ops_v21_07._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('ctrlParams', struct_NV2080_CTRL_GPU_MIGRATABLE_OPS_PARAMS_v21_07),
]

rpc_ctrl_gpu_migratable_ops_v21_07 = struct_rpc_ctrl_gpu_migratable_ops_v21_07
rpc_ctrl_gpu_migratable_ops_v = struct_rpc_ctrl_gpu_migratable_ops_v21_07
class struct_rpc_invalidate_tlb_v23_03(Structure):
    pass

struct_rpc_invalidate_tlb_v23_03._pack_ = 1 # source:False
struct_rpc_invalidate_tlb_v23_03._fields_ = [
    ('pdbAddress', ctypes.c_uint64),
    ('regVal', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

rpc_invalidate_tlb_v23_03 = struct_rpc_invalidate_tlb_v23_03
rpc_invalidate_tlb_v = struct_rpc_invalidate_tlb_v23_03
class struct_rpc_get_brand_caps_v25_12(Structure):
    pass

struct_rpc_get_brand_caps_v25_12._pack_ = 1 # source:False
struct_rpc_get_brand_caps_v25_12._fields_ = [
    ('brands', ctypes.c_uint32),
]

rpc_get_brand_caps_v25_12 = struct_rpc_get_brand_caps_v25_12
rpc_get_brand_caps_v = struct_rpc_get_brand_caps_v25_12
class struct_rpc_gsp_set_system_info_v17_00(Structure):
    pass

struct_rpc_gsp_set_system_info_v17_00._pack_ = 1 # source:False
struct_rpc_gsp_set_system_info_v17_00._fields_ = [
    ('data', ctypes.c_uint32),
]

rpc_gsp_set_system_info_v17_00 = struct_rpc_gsp_set_system_info_v17_00
rpc_gsp_set_system_info_v = struct_rpc_gsp_set_system_info_v17_00
class struct_rpc_gsp_rm_alloc_v03_00(Structure):
    pass

struct_rpc_gsp_rm_alloc_v03_00._pack_ = 1 # source:False
struct_rpc_gsp_rm_alloc_v03_00._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hParent', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('hClass', ctypes.c_uint32),
    ('status', ctypes.c_uint32),
    ('paramsSize', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('reserved', ctypes.c_ubyte * 4),
    ('params', ctypes.c_ubyte * 0),
]

rpc_gsp_rm_alloc_v03_00 = struct_rpc_gsp_rm_alloc_v03_00
rpc_gsp_rm_alloc_v = struct_rpc_gsp_rm_alloc_v03_00
class struct_rpc_gsp_rm_control_v03_00(Structure):
    pass

struct_rpc_gsp_rm_control_v03_00._pack_ = 1 # source:False
struct_rpc_gsp_rm_control_v03_00._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('cmd', ctypes.c_uint32),
    ('status', ctypes.c_uint32),
    ('paramsSize', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('params', ctypes.c_ubyte * 0),
]

rpc_gsp_rm_control_v03_00 = struct_rpc_gsp_rm_control_v03_00
rpc_gsp_rm_control_v = struct_rpc_gsp_rm_control_v03_00
class struct_rpc_dump_protobuf_component_v18_12(Structure):
    pass

struct_rpc_dump_protobuf_component_v18_12._pack_ = 1 # source:False
struct_rpc_dump_protobuf_component_v18_12._fields_ = [
    ('component', ctypes.c_uint16),
    ('nvDumpType', ctypes.c_ubyte),
    ('countOnly', ctypes.c_ubyte),
    ('bugCheckCode', ctypes.c_uint32),
    ('internalCode', ctypes.c_uint32),
    ('bufferSize', ctypes.c_uint32),
    ('blob', ctypes.c_ubyte * 0),
]

rpc_dump_protobuf_component_v18_12 = struct_rpc_dump_protobuf_component_v18_12
rpc_dump_protobuf_component_v = struct_rpc_dump_protobuf_component_v18_12
class struct_rpc_run_cpu_sequencer_v17_00(Structure):
    pass

struct_rpc_run_cpu_sequencer_v17_00._pack_ = 1 # source:False
struct_rpc_run_cpu_sequencer_v17_00._fields_ = [
    ('bufferSizeDWord', ctypes.c_uint32),
    ('cmdIndex', ctypes.c_uint32),
    ('regSaveArea', ctypes.c_uint32 * 8),
    ('commandBuffer', ctypes.c_uint32 * 0),
]

rpc_run_cpu_sequencer_v17_00 = struct_rpc_run_cpu_sequencer_v17_00
rpc_run_cpu_sequencer_v = struct_rpc_run_cpu_sequencer_v17_00
class struct_rpc_post_event_v17_00(Structure):
    pass

struct_rpc_post_event_v17_00._pack_ = 1 # source:False
struct_rpc_post_event_v17_00._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hEvent', ctypes.c_uint32),
    ('notifyIndex', ctypes.c_uint32),
    ('data', ctypes.c_uint32),
    ('info16', ctypes.c_uint16),
    ('PADDING_0', ctypes.c_ubyte * 2),
    ('status', ctypes.c_uint32),
    ('eventDataSize', ctypes.c_uint32),
    ('bNotifyList', ctypes.c_ubyte),
    ('eventData', ctypes.c_ubyte * 0),
    ('PADDING_1', ctypes.c_ubyte * 3),
]

rpc_post_event_v17_00 = struct_rpc_post_event_v17_00
rpc_post_event_v = struct_rpc_post_event_v17_00
class struct_rpc_rc_triggered_v17_02(Structure):
    pass

struct_rpc_rc_triggered_v17_02._pack_ = 1 # source:False
struct_rpc_rc_triggered_v17_02._fields_ = [
    ('nv2080EngineType', ctypes.c_uint32),
    ('chid', ctypes.c_uint32),
    ('gfid', ctypes.c_uint32),
    ('exceptLevel', ctypes.c_uint32),
    ('exceptType', ctypes.c_uint32),
    ('scope', ctypes.c_uint32),
    ('partitionAttributionId', ctypes.c_uint16),
    ('PADDING_0', ctypes.c_ubyte * 2),
    ('mmuFaultAddrLo', ctypes.c_uint32),
    ('mmuFaultAddrHi', ctypes.c_uint32),
    ('mmuFaultType', ctypes.c_uint32),
    ('bCallbackNeeded', ctypes.c_ubyte),
    ('PADDING_1', ctypes.c_ubyte * 3),
    ('rcJournalBufferSize', ctypes.c_uint32),
    ('rcJournalBuffer', ctypes.c_ubyte * 0),
]

rpc_rc_triggered_v17_02 = struct_rpc_rc_triggered_v17_02
rpc_rc_triggered_v = struct_rpc_rc_triggered_v17_02
class struct_rpc_os_error_log_v17_00(Structure):
    pass

struct_rpc_os_error_log_v17_00._pack_ = 1 # source:False
struct_rpc_os_error_log_v17_00._fields_ = [
    ('exceptType', ctypes.c_uint32),
    ('runlistId', ctypes.c_uint32),
    ('chid', ctypes.c_uint32),
    ('errString', ctypes.c_char * 256),
]

rpc_os_error_log_v17_00 = struct_rpc_os_error_log_v17_00
rpc_os_error_log_v = struct_rpc_os_error_log_v17_00
class struct_rpc_rg_line_intr_v17_00(Structure):
    pass

struct_rpc_rg_line_intr_v17_00._pack_ = 1 # source:False
struct_rpc_rg_line_intr_v17_00._fields_ = [
    ('head', ctypes.c_uint32),
    ('rgIntr', ctypes.c_uint32),
]

rpc_rg_line_intr_v17_00 = struct_rpc_rg_line_intr_v17_00
rpc_rg_line_intr_v = struct_rpc_rg_line_intr_v17_00
class struct_rpc_display_modeset_v01_00(Structure):
    pass

struct_rpc_display_modeset_v01_00._pack_ = 1 # source:False
struct_rpc_display_modeset_v01_00._fields_ = [
    ('bModesetStart', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 3),
    ('minRequiredIsoBandwidthKBPS', ctypes.c_uint32),
    ('minRequiredFloorBandwidthKBPS', ctypes.c_uint32),
]

rpc_display_modeset_v01_00 = struct_rpc_display_modeset_v01_00
rpc_display_modeset_v = struct_rpc_display_modeset_v01_00
class struct_rpc_gpuacct_perfmon_util_samples_v1F_0E(Structure):
    pass

class struct_NV2080_CTRL_PERF_GET_GPUMON_PERFMON_UTIL_SAMPLES_V2_PARAMS_v1F_0E(Structure):
    pass

struct_NV2080_CTRL_PERF_GET_GPUMON_PERFMON_UTIL_SAMPLES_V2_PARAMS_v1F_0E._pack_ = 1 # source:False
struct_NV2080_CTRL_PERF_GET_GPUMON_PERFMON_UTIL_SAMPLES_V2_PARAMS_v1F_0E._fields_ = [
    ('type', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 3),
    ('bufSize', ctypes.c_uint32),
    ('count', ctypes.c_uint32),
    ('tracker', ctypes.c_uint32),
    ('samples', struct_NV2080_CTRL_PERF_GPUMON_PERFMON_UTIL_SAMPLE_v1F_0E * 72),
]

struct_rpc_gpuacct_perfmon_util_samples_v1F_0E._pack_ = 1 # source:False
struct_rpc_gpuacct_perfmon_util_samples_v1F_0E._fields_ = [
    ('params', struct_NV2080_CTRL_PERF_GET_GPUMON_PERFMON_UTIL_SAMPLES_V2_PARAMS_v1F_0E),
]

rpc_gpuacct_perfmon_util_samples_v1F_0E = struct_rpc_gpuacct_perfmon_util_samples_v1F_0E
rpc_gpuacct_perfmon_util_samples_v = struct_rpc_gpuacct_perfmon_util_samples_v1F_0E
class struct_rpc_vgpu_gsp_plugin_triggered_v17_00(Structure):
    pass

struct_rpc_vgpu_gsp_plugin_triggered_v17_00._pack_ = 1 # source:False
struct_rpc_vgpu_gsp_plugin_triggered_v17_00._fields_ = [
    ('gfid', ctypes.c_uint32),
    ('notifyIndex', ctypes.c_uint32),
]

rpc_vgpu_gsp_plugin_triggered_v17_00 = struct_rpc_vgpu_gsp_plugin_triggered_v17_00
rpc_vgpu_gsp_plugin_triggered_v = struct_rpc_vgpu_gsp_plugin_triggered_v17_00
class struct_rpc_vgpu_config_event_v17_00(Structure):
    pass

struct_rpc_vgpu_config_event_v17_00._pack_ = 1 # source:False
struct_rpc_vgpu_config_event_v17_00._fields_ = [
    ('notifyIndex', ctypes.c_uint32),
]

rpc_vgpu_config_event_v17_00 = struct_rpc_vgpu_config_event_v17_00
rpc_vgpu_config_event_v = struct_rpc_vgpu_config_event_v17_00
class struct_rpc_dce_rm_init_v01_00(Structure):
    pass

struct_rpc_dce_rm_init_v01_00._pack_ = 1 # source:False
struct_rpc_dce_rm_init_v01_00._fields_ = [
    ('bInit', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 3),
    ('hInternalClient', ctypes.c_uint32),
]

rpc_dce_rm_init_v01_00 = struct_rpc_dce_rm_init_v01_00
rpc_dce_rm_init_v = struct_rpc_dce_rm_init_v01_00
class struct_rpc_sim_read_v1E_01(Structure):
    pass

struct_rpc_sim_read_v1E_01._pack_ = 1 # source:False
struct_rpc_sim_read_v1E_01._fields_ = [
    ('path', ctypes.c_char * 256),
    ('index', ctypes.c_uint32),
    ('count', ctypes.c_uint32),
]

rpc_sim_read_v1E_01 = struct_rpc_sim_read_v1E_01
rpc_sim_read_v = struct_rpc_sim_read_v1E_01
class struct_rpc_sim_write_v1E_01(Structure):
    pass

struct_rpc_sim_write_v1E_01._pack_ = 1 # source:False
struct_rpc_sim_write_v1E_01._fields_ = [
    ('path', ctypes.c_char * 256),
    ('index', ctypes.c_uint32),
    ('count', ctypes.c_uint32),
    ('data', ctypes.c_uint32),
]

rpc_sim_write_v1E_01 = struct_rpc_sim_write_v1E_01
rpc_sim_write_v = struct_rpc_sim_write_v1E_01
class struct_rpc_ucode_libos_print_v1E_08(Structure):
    pass

struct_rpc_ucode_libos_print_v1E_08._pack_ = 1 # source:False
struct_rpc_ucode_libos_print_v1E_08._fields_ = [
    ('ucodeEngDesc', ctypes.c_uint32),
    ('libosPrintBufSize', ctypes.c_uint32),
    ('libosPrintBuf', ctypes.c_ubyte * 0),
]

rpc_ucode_libos_print_v1E_08 = struct_rpc_ucode_libos_print_v1E_08
rpc_ucode_libos_print_v = struct_rpc_ucode_libos_print_v1E_08
class struct_rpc_init_done_v17_00(Structure):
    pass

struct_rpc_init_done_v17_00._pack_ = 1 # source:False
struct_rpc_init_done_v17_00._fields_ = [
    ('not_used', ctypes.c_uint32),
]

rpc_init_done_v17_00 = struct_rpc_init_done_v17_00
rpc_init_done_v = struct_rpc_init_done_v17_00
class struct_rpc_semaphore_schedule_callback_v17_00(Structure):
    pass

struct_rpc_semaphore_schedule_callback_v17_00._pack_ = 1 # source:False
struct_rpc_semaphore_schedule_callback_v17_00._fields_ = [
    ('GPUVA', ctypes.c_uint64),
    ('hVASpace', ctypes.c_uint32),
    ('ReleaseValue', ctypes.c_uint32),
    ('Flags', ctypes.c_uint32),
    ('completionStatus', ctypes.c_uint32),
    ('hClient', ctypes.c_uint32),
    ('hEvent', ctypes.c_uint32),
]

rpc_semaphore_schedule_callback_v17_00 = struct_rpc_semaphore_schedule_callback_v17_00
rpc_semaphore_schedule_callback_v = struct_rpc_semaphore_schedule_callback_v17_00
class struct_rpc_timed_semaphore_release_v01_00(Structure):
    pass

struct_rpc_timed_semaphore_release_v01_00._pack_ = 1 # source:False
struct_rpc_timed_semaphore_release_v01_00._fields_ = [
    ('semaphoreVA', ctypes.c_uint64),
    ('notifierVA', ctypes.c_uint64),
    ('hVASpace', ctypes.c_uint32),
    ('releaseValue', ctypes.c_uint32),
    ('completionStatus', ctypes.c_uint32),
    ('hClient', ctypes.c_uint32),
    ('hDevice', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

rpc_timed_semaphore_release_v01_00 = struct_rpc_timed_semaphore_release_v01_00
rpc_timed_semaphore_release_v = struct_rpc_timed_semaphore_release_v01_00
class struct_rpc_perf_gpu_boost_sync_limits_callback_v17_00(Structure):
    pass

class struct_NV2080_CTRL_INTERNAL_PERF_GPU_BOOST_SYNC_SET_LIMITS_PARAMS_v17_00(Structure):
    pass

struct_NV2080_CTRL_INTERNAL_PERF_GPU_BOOST_SYNC_SET_LIMITS_PARAMS_v17_00._pack_ = 1 # source:False
struct_NV2080_CTRL_INTERNAL_PERF_GPU_BOOST_SYNC_SET_LIMITS_PARAMS_v17_00._fields_ = [
    ('flags', ctypes.c_uint32),
    ('bBridgeless', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 3),
    ('currLimits', ctypes.c_uint32 * 2),
]

struct_rpc_perf_gpu_boost_sync_limits_callback_v17_00._pack_ = 1 # source:False
struct_rpc_perf_gpu_boost_sync_limits_callback_v17_00._fields_ = [
    ('params', struct_NV2080_CTRL_INTERNAL_PERF_GPU_BOOST_SYNC_SET_LIMITS_PARAMS_v17_00),
]

rpc_perf_gpu_boost_sync_limits_callback_v17_00 = struct_rpc_perf_gpu_boost_sync_limits_callback_v17_00
rpc_perf_gpu_boost_sync_limits_callback_v = struct_rpc_perf_gpu_boost_sync_limits_callback_v17_00
class struct_rpc_perf_bridgeless_info_update_v17_00(Structure):
    pass

struct_rpc_perf_bridgeless_info_update_v17_00._pack_ = 1 # source:False
struct_rpc_perf_bridgeless_info_update_v17_00._fields_ = [
    ('bBridgeless', ctypes.c_uint64),
]

rpc_perf_bridgeless_info_update_v17_00 = struct_rpc_perf_bridgeless_info_update_v17_00
rpc_perf_bridgeless_info_update_v = struct_rpc_perf_bridgeless_info_update_v17_00
class struct_rpc_nvlink_fault_up_v17_00(Structure):
    pass

struct_rpc_nvlink_fault_up_v17_00._pack_ = 1 # source:False
struct_rpc_nvlink_fault_up_v17_00._fields_ = [
    ('linkId', ctypes.c_uint32),
]

rpc_nvlink_fault_up_v17_00 = struct_rpc_nvlink_fault_up_v17_00
rpc_nvlink_fault_up_v = struct_rpc_nvlink_fault_up_v17_00
class struct_rpc_nvlink_inband_received_data_256_v17_00(Structure):
    pass

class struct_NV2080_CTRL_NVLINK_INBAND_RECEIVED_DATA_256_PARAMS_v17_00(Structure):
    pass

struct_NV2080_CTRL_NVLINK_INBAND_RECEIVED_DATA_256_PARAMS_v17_00._pack_ = 1 # source:False
struct_NV2080_CTRL_NVLINK_INBAND_RECEIVED_DATA_256_PARAMS_v17_00._fields_ = [
    ('dataSize', ctypes.c_uint32),
    ('data', ctypes.c_ubyte * 256),
]

struct_rpc_nvlink_inband_received_data_256_v17_00._pack_ = 1 # source:False
struct_rpc_nvlink_inband_received_data_256_v17_00._fields_ = [
    ('params', struct_NV2080_CTRL_NVLINK_INBAND_RECEIVED_DATA_256_PARAMS_v17_00),
]

rpc_nvlink_inband_received_data_256_v17_00 = struct_rpc_nvlink_inband_received_data_256_v17_00
rpc_nvlink_inband_received_data_256_v = struct_rpc_nvlink_inband_received_data_256_v17_00
class struct_rpc_nvlink_inband_received_data_512_v17_00(Structure):
    pass

class struct_NV2080_CTRL_NVLINK_INBAND_RECEIVED_DATA_512_PARAMS_v17_00(Structure):
    pass

struct_NV2080_CTRL_NVLINK_INBAND_RECEIVED_DATA_512_PARAMS_v17_00._pack_ = 1 # source:False
struct_NV2080_CTRL_NVLINK_INBAND_RECEIVED_DATA_512_PARAMS_v17_00._fields_ = [
    ('dataSize', ctypes.c_uint32),
    ('data', ctypes.c_ubyte * 512),
]

struct_rpc_nvlink_inband_received_data_512_v17_00._pack_ = 1 # source:False
struct_rpc_nvlink_inband_received_data_512_v17_00._fields_ = [
    ('params', struct_NV2080_CTRL_NVLINK_INBAND_RECEIVED_DATA_512_PARAMS_v17_00),
]

rpc_nvlink_inband_received_data_512_v17_00 = struct_rpc_nvlink_inband_received_data_512_v17_00
rpc_nvlink_inband_received_data_512_v = struct_rpc_nvlink_inband_received_data_512_v17_00
class struct_rpc_nvlink_inband_received_data_1024_v17_00(Structure):
    pass

class struct_NV2080_CTRL_NVLINK_INBAND_RECEIVED_DATA_1024_PARAMS_v17_00(Structure):
    pass

struct_NV2080_CTRL_NVLINK_INBAND_RECEIVED_DATA_1024_PARAMS_v17_00._pack_ = 1 # source:False
struct_NV2080_CTRL_NVLINK_INBAND_RECEIVED_DATA_1024_PARAMS_v17_00._fields_ = [
    ('dataSize', ctypes.c_uint32),
    ('data', ctypes.c_ubyte * 1024),
]

struct_rpc_nvlink_inband_received_data_1024_v17_00._pack_ = 1 # source:False
struct_rpc_nvlink_inband_received_data_1024_v17_00._fields_ = [
    ('params', struct_NV2080_CTRL_NVLINK_INBAND_RECEIVED_DATA_1024_PARAMS_v17_00),
]

rpc_nvlink_inband_received_data_1024_v17_00 = struct_rpc_nvlink_inband_received_data_1024_v17_00
rpc_nvlink_inband_received_data_1024_v = struct_rpc_nvlink_inband_received_data_1024_v17_00
class struct_rpc_nvlink_inband_received_data_2048_v17_00(Structure):
    pass

class struct_NV2080_CTRL_NVLINK_INBAND_RECEIVED_DATA_2048_PARAMS_v17_00(Structure):
    pass

struct_NV2080_CTRL_NVLINK_INBAND_RECEIVED_DATA_2048_PARAMS_v17_00._pack_ = 1 # source:False
struct_NV2080_CTRL_NVLINK_INBAND_RECEIVED_DATA_2048_PARAMS_v17_00._fields_ = [
    ('dataSize', ctypes.c_uint32),
    ('data', ctypes.c_ubyte * 2048),
]

struct_rpc_nvlink_inband_received_data_2048_v17_00._pack_ = 1 # source:False
struct_rpc_nvlink_inband_received_data_2048_v17_00._fields_ = [
    ('params', struct_NV2080_CTRL_NVLINK_INBAND_RECEIVED_DATA_2048_PARAMS_v17_00),
]

rpc_nvlink_inband_received_data_2048_v17_00 = struct_rpc_nvlink_inband_received_data_2048_v17_00
rpc_nvlink_inband_received_data_2048_v = struct_rpc_nvlink_inband_received_data_2048_v17_00
class struct_rpc_nvlink_inband_received_data_4096_v17_00(Structure):
    pass

class struct_NV2080_CTRL_NVLINK_INBAND_RECEIVED_DATA_4096_PARAMS_v17_00(Structure):
    pass

struct_NV2080_CTRL_NVLINK_INBAND_RECEIVED_DATA_4096_PARAMS_v17_00._pack_ = 1 # source:False
struct_NV2080_CTRL_NVLINK_INBAND_RECEIVED_DATA_4096_PARAMS_v17_00._fields_ = [
    ('dataSize', ctypes.c_uint32),
    ('data', ctypes.c_ubyte * 4096),
]

struct_rpc_nvlink_inband_received_data_4096_v17_00._pack_ = 1 # source:False
struct_rpc_nvlink_inband_received_data_4096_v17_00._fields_ = [
    ('params', struct_NV2080_CTRL_NVLINK_INBAND_RECEIVED_DATA_4096_PARAMS_v17_00),
]

rpc_nvlink_inband_received_data_4096_v17_00 = struct_rpc_nvlink_inband_received_data_4096_v17_00
rpc_nvlink_inband_received_data_4096_v = struct_rpc_nvlink_inband_received_data_4096_v17_00
class struct_rpc_nvlink_is_gpu_degraded_v17_00(Structure):
    pass

class struct_NV2080_CTRL_NVLINK_IS_GPU_DEGRADED_PARAMS_v17_00(Structure):
    pass

struct_NV2080_CTRL_NVLINK_IS_GPU_DEGRADED_PARAMS_v17_00._pack_ = 1 # source:False
struct_NV2080_CTRL_NVLINK_IS_GPU_DEGRADED_PARAMS_v17_00._fields_ = [
    ('linkId', ctypes.c_uint32),
    ('bIsGpuDegraded', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 3),
]

struct_rpc_nvlink_is_gpu_degraded_v17_00._pack_ = 1 # source:False
struct_rpc_nvlink_is_gpu_degraded_v17_00._fields_ = [
    ('params', struct_NV2080_CTRL_NVLINK_IS_GPU_DEGRADED_PARAMS_v17_00),
]

rpc_nvlink_is_gpu_degraded_v17_00 = struct_rpc_nvlink_is_gpu_degraded_v17_00
rpc_nvlink_is_gpu_degraded_v = struct_rpc_nvlink_is_gpu_degraded_v17_00
class struct_rpc_nvlink_fatal_error_recovery_v17_00(Structure):
    pass

class struct_NV2080_CTRL_NVLINK_FATAL_ERROR_RECOVERY_PARAMS_v17_00(Structure):
    pass

struct_NV2080_CTRL_NVLINK_FATAL_ERROR_RECOVERY_PARAMS_v17_00._pack_ = 1 # source:False
struct_NV2080_CTRL_NVLINK_FATAL_ERROR_RECOVERY_PARAMS_v17_00._fields_ = [
    ('bRecoverable', ctypes.c_ubyte),
    ('bLazy', ctypes.c_ubyte),
]

struct_rpc_nvlink_fatal_error_recovery_v17_00._pack_ = 1 # source:False
struct_rpc_nvlink_fatal_error_recovery_v17_00._fields_ = [
    ('params', struct_NV2080_CTRL_NVLINK_FATAL_ERROR_RECOVERY_PARAMS_v17_00),
]

rpc_nvlink_fatal_error_recovery_v17_00 = struct_rpc_nvlink_fatal_error_recovery_v17_00
rpc_nvlink_fatal_error_recovery_v = struct_rpc_nvlink_fatal_error_recovery_v17_00
class struct_rpc_update_gsp_trace_v01_00(Structure):
    pass

struct_rpc_update_gsp_trace_v01_00._pack_ = 1 # source:False
struct_rpc_update_gsp_trace_v01_00._fields_ = [
    ('records', ctypes.c_uint32),
    ('data', ctypes.c_uint32),
]

rpc_update_gsp_trace_v01_00 = struct_rpc_update_gsp_trace_v01_00
rpc_update_gsp_trace_v = struct_rpc_update_gsp_trace_v01_00
class struct_rpc_gsp_post_nocat_record_v01_00(Structure):
    pass

struct_rpc_gsp_post_nocat_record_v01_00._pack_ = 1 # source:False
struct_rpc_gsp_post_nocat_record_v01_00._fields_ = [
    ('data', ctypes.c_uint32),
]

rpc_gsp_post_nocat_record_v01_00 = struct_rpc_gsp_post_nocat_record_v01_00
rpc_gsp_post_nocat_record_v = struct_rpc_gsp_post_nocat_record_v01_00
class struct_rpc_extdev_intr_service_v17_00(Structure):
    pass

struct_rpc_extdev_intr_service_v17_00._pack_ = 1 # source:False
struct_rpc_extdev_intr_service_v17_00._fields_ = [
    ('lossRegStatus', ctypes.c_ubyte),
    ('gainRegStatus', ctypes.c_ubyte),
    ('miscRegStatus', ctypes.c_ubyte),
    ('rmStatus', ctypes.c_ubyte),
]

rpc_extdev_intr_service_v17_00 = struct_rpc_extdev_intr_service_v17_00
rpc_extdev_intr_service_v = struct_rpc_extdev_intr_service_v17_00
class struct_rpc_pfm_req_hndlr_state_sync_callback_v21_04(Structure):
    pass

class struct_NV2080_CTRL_INTERNAL_PFM_REQ_HNDLR_STATE_SYNC_PARAMS_v21_04(Structure):
    pass

class struct_NV2080_CTRL_INTERNAL_PFM_REQ_HNDLR_STATE_SYNC_data_v21_04(Structure):
    pass

class union_NV2080_CTRL_INTERNAL_PFM_REQ_HNDLR_STATE_SYNC_DATA_type_v21_04(Union):
    pass

class struct_NV2080_CTRL_INTERNAL_PFM_REQ_HNDLR_STATE_SYNC_DATA_SMBPBI_v21_04(Structure):
    pass

struct_NV2080_CTRL_INTERNAL_PFM_REQ_HNDLR_STATE_SYNC_DATA_SMBPBI_v21_04._pack_ = 1 # source:False
struct_NV2080_CTRL_INTERNAL_PFM_REQ_HNDLR_STATE_SYNC_DATA_SMBPBI_v21_04._fields_ = [
    ('sensorId', ctypes.c_uint32),
    ('limit', ctypes.c_uint32),
]

union_NV2080_CTRL_INTERNAL_PFM_REQ_HNDLR_STATE_SYNC_DATA_type_v21_04._pack_ = 1 # source:False
union_NV2080_CTRL_INTERNAL_PFM_REQ_HNDLR_STATE_SYNC_DATA_type_v21_04._fields_ = [
    ('smbpbi', struct_NV2080_CTRL_INTERNAL_PFM_REQ_HNDLR_STATE_SYNC_DATA_SMBPBI_v21_04),
]

struct_NV2080_CTRL_INTERNAL_PFM_REQ_HNDLR_STATE_SYNC_data_v21_04._pack_ = 1 # source:False
struct_NV2080_CTRL_INTERNAL_PFM_REQ_HNDLR_STATE_SYNC_data_v21_04._fields_ = [
    ('type', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 3),
    ('data', union_NV2080_CTRL_INTERNAL_PFM_REQ_HNDLR_STATE_SYNC_DATA_type_v21_04),
]

struct_NV2080_CTRL_INTERNAL_PFM_REQ_HNDLR_STATE_SYNC_PARAMS_v21_04._pack_ = 1 # source:False
struct_NV2080_CTRL_INTERNAL_PFM_REQ_HNDLR_STATE_SYNC_PARAMS_v21_04._fields_ = [
    ('flags', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 3),
    ('syncData', struct_NV2080_CTRL_INTERNAL_PFM_REQ_HNDLR_STATE_SYNC_data_v21_04),
]

struct_rpc_pfm_req_hndlr_state_sync_callback_v21_04._pack_ = 1 # source:False
struct_rpc_pfm_req_hndlr_state_sync_callback_v21_04._fields_ = [
    ('params', struct_NV2080_CTRL_INTERNAL_PFM_REQ_HNDLR_STATE_SYNC_PARAMS_v21_04),
]

rpc_pfm_req_hndlr_state_sync_callback_v21_04 = struct_rpc_pfm_req_hndlr_state_sync_callback_v21_04
rpc_pfm_req_hndlr_state_sync_callback_v = struct_rpc_pfm_req_hndlr_state_sync_callback_v21_04
class struct_rpc_vgpu_gsp_mig_ci_config_v21_03(Structure):
    pass

struct_rpc_vgpu_gsp_mig_ci_config_v21_03._pack_ = 1 # source:False
struct_rpc_vgpu_gsp_mig_ci_config_v21_03._fields_ = [
    ('execPartCount', ctypes.c_uint32),
    ('execPartId', ctypes.c_uint32 * 8),
    ('gfid', ctypes.c_uint32),
    ('bDelete', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 3),
]

rpc_vgpu_gsp_mig_ci_config_v21_03 = struct_rpc_vgpu_gsp_mig_ci_config_v21_03
rpc_vgpu_gsp_mig_ci_config_v = struct_rpc_vgpu_gsp_mig_ci_config_v21_03
class struct_rpc_gsp_lockdown_notice_v17_00(Structure):
    pass

struct_rpc_gsp_lockdown_notice_v17_00._pack_ = 1 # source:False
struct_rpc_gsp_lockdown_notice_v17_00._fields_ = [
    ('bLockdownEngaging', ctypes.c_ubyte),
]

rpc_gsp_lockdown_notice_v17_00 = struct_rpc_gsp_lockdown_notice_v17_00
rpc_gsp_lockdown_notice_v = struct_rpc_gsp_lockdown_notice_v17_00
class struct_rpc_ctrl_gpu_query_ecc_status_v24_06(Structure):
    pass

class struct_NV2080_CTRL_GPU_QUERY_ECC_STATUS_DEPRECATED_RPC_PARAMS_v24_06(Structure):
    pass

class struct_NV2080_CTRL_GPU_QUERY_ECC_UNIT_STATUS_v15_01(Structure):
    pass

class struct_NV2080_CTRL_GPU_QUERY_ECC_EXCEPTION_STATUS_v15_01(Structure):
    pass

struct_NV2080_CTRL_GPU_QUERY_ECC_EXCEPTION_STATUS_v15_01._pack_ = 1 # source:False
struct_NV2080_CTRL_GPU_QUERY_ECC_EXCEPTION_STATUS_v15_01._fields_ = [
    ('count', ctypes.c_uint64),
]

struct_NV2080_CTRL_GPU_QUERY_ECC_UNIT_STATUS_v15_01._pack_ = 1 # source:False
struct_NV2080_CTRL_GPU_QUERY_ECC_UNIT_STATUS_v15_01._fields_ = [
    ('enabled', ctypes.c_ubyte),
    ('scrubComplete', ctypes.c_ubyte),
    ('supported', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 5),
    ('dbe', struct_NV2080_CTRL_GPU_QUERY_ECC_EXCEPTION_STATUS_v15_01),
    ('dbeNonResettable', struct_NV2080_CTRL_GPU_QUERY_ECC_EXCEPTION_STATUS_v15_01),
    ('sbe', struct_NV2080_CTRL_GPU_QUERY_ECC_EXCEPTION_STATUS_v15_01),
    ('sbeNonResettable', struct_NV2080_CTRL_GPU_QUERY_ECC_EXCEPTION_STATUS_v15_01),
]

struct_NV2080_CTRL_GPU_QUERY_ECC_STATUS_DEPRECATED_RPC_PARAMS_v24_06._pack_ = 1 # source:False
struct_NV2080_CTRL_GPU_QUERY_ECC_STATUS_DEPRECATED_RPC_PARAMS_v24_06._fields_ = [
    ('units', struct_NV2080_CTRL_GPU_QUERY_ECC_UNIT_STATUS_v15_01 * 25),
    ('bFatalPoisonError', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 3),
    ('flags', ctypes.c_uint32),
]

struct_rpc_ctrl_gpu_query_ecc_status_v24_06._pack_ = 1 # source:False
struct_rpc_ctrl_gpu_query_ecc_status_v24_06._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('params', struct_NV2080_CTRL_GPU_QUERY_ECC_STATUS_DEPRECATED_RPC_PARAMS_v24_06),
]

rpc_ctrl_gpu_query_ecc_status_v24_06 = struct_rpc_ctrl_gpu_query_ecc_status_v24_06
class struct_rpc_ctrl_gpu_query_ecc_status_v26_02(Structure):
    pass

class struct_NV2080_CTRL_GPU_QUERY_ECC_STATUS_PARAMS_v26_02(Structure):
    pass

struct_NV2080_CTRL_GPU_QUERY_ECC_STATUS_PARAMS_v26_02._pack_ = 1 # source:False
struct_NV2080_CTRL_GPU_QUERY_ECC_STATUS_PARAMS_v26_02._fields_ = [
    ('units', struct_NV2080_CTRL_GPU_QUERY_ECC_UNIT_STATUS_v15_01 * 30),
    ('bFatalPoisonError', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 3),
    ('flags', ctypes.c_uint32),
]

struct_rpc_ctrl_gpu_query_ecc_status_v26_02._pack_ = 1 # source:False
struct_rpc_ctrl_gpu_query_ecc_status_v26_02._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('params', struct_NV2080_CTRL_GPU_QUERY_ECC_STATUS_PARAMS_v26_02),
]

rpc_ctrl_gpu_query_ecc_status_v26_02 = struct_rpc_ctrl_gpu_query_ecc_status_v26_02
rpc_ctrl_gpu_query_ecc_status_v = struct_rpc_ctrl_gpu_query_ecc_status_v26_02
class struct_rpc_ctrl_dbg_get_mode_mmu_debug_v25_04(Structure):
    pass

class struct_NV83DE_CTRL_DEBUG_GET_MODE_MMU_DEBUG_PARAMS_v25_04(Structure):
    pass

struct_NV83DE_CTRL_DEBUG_GET_MODE_MMU_DEBUG_PARAMS_v25_04._pack_ = 1 # source:False
struct_NV83DE_CTRL_DEBUG_GET_MODE_MMU_DEBUG_PARAMS_v25_04._fields_ = [
    ('value', ctypes.c_uint32),
]

struct_rpc_ctrl_dbg_get_mode_mmu_debug_v25_04._pack_ = 1 # source:False
struct_rpc_ctrl_dbg_get_mode_mmu_debug_v25_04._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('ctrlParams', struct_NV83DE_CTRL_DEBUG_GET_MODE_MMU_DEBUG_PARAMS_v25_04),
]

rpc_ctrl_dbg_get_mode_mmu_debug_v25_04 = struct_rpc_ctrl_dbg_get_mode_mmu_debug_v25_04
rpc_ctrl_dbg_get_mode_mmu_debug_v = struct_rpc_ctrl_dbg_get_mode_mmu_debug_v25_04
class struct_rpc_ctrl_dbg_get_mode_mmu_gcc_debug_v29_07(Structure):
    pass

class struct_NV83DE_CTRL_DEBUG_GET_MODE_MMU_GCC_DEBUG_PARAMS_v29_07(Structure):
    pass

struct_NV83DE_CTRL_DEBUG_GET_MODE_MMU_GCC_DEBUG_PARAMS_v29_07._pack_ = 1 # source:False
struct_NV83DE_CTRL_DEBUG_GET_MODE_MMU_GCC_DEBUG_PARAMS_v29_07._fields_ = [
    ('value', ctypes.c_uint32),
]

struct_rpc_ctrl_dbg_get_mode_mmu_gcc_debug_v29_07._pack_ = 1 # source:False
struct_rpc_ctrl_dbg_get_mode_mmu_gcc_debug_v29_07._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('ctrlParams', struct_NV83DE_CTRL_DEBUG_GET_MODE_MMU_GCC_DEBUG_PARAMS_v29_07),
]

rpc_ctrl_dbg_get_mode_mmu_gcc_debug_v29_07 = struct_rpc_ctrl_dbg_get_mode_mmu_gcc_debug_v29_07
rpc_ctrl_dbg_get_mode_mmu_gcc_debug_v = struct_rpc_ctrl_dbg_get_mode_mmu_gcc_debug_v29_07
class struct_rpc_ctrl_cmd_internal_gpu_start_fabric_probe_v25_09(Structure):
    pass

struct_rpc_ctrl_cmd_internal_gpu_start_fabric_probe_v25_09._pack_ = 1 # source:False
struct_rpc_ctrl_cmd_internal_gpu_start_fabric_probe_v25_09._fields_ = [
    ('bwMode', ctypes.c_ubyte),
]

rpc_ctrl_cmd_internal_gpu_start_fabric_probe_v25_09 = struct_rpc_ctrl_cmd_internal_gpu_start_fabric_probe_v25_09
rpc_ctrl_cmd_internal_gpu_start_fabric_probe_v = struct_rpc_ctrl_cmd_internal_gpu_start_fabric_probe_v25_09
class struct_rpc_ctrl_nvlink_get_inband_received_data_v25_0C(Structure):
    pass

class struct_NV2080_CTRL_NVLINK_INBAND_RECEIVED_DATA_PARAMS_v25_0C(Structure):
    pass

struct_NV2080_CTRL_NVLINK_INBAND_RECEIVED_DATA_PARAMS_v25_0C._pack_ = 1 # source:False
struct_NV2080_CTRL_NVLINK_INBAND_RECEIVED_DATA_PARAMS_v25_0C._fields_ = [
    ('dataSize', ctypes.c_uint32),
    ('data', ctypes.c_ubyte * 512),
]

struct_rpc_ctrl_nvlink_get_inband_received_data_v25_0C._pack_ = 1 # source:False
struct_rpc_ctrl_nvlink_get_inband_received_data_v25_0C._fields_ = [
    ('message_type', ctypes.c_uint16),
    ('more', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte),
    ('payload', struct_NV2080_CTRL_NVLINK_INBAND_RECEIVED_DATA_PARAMS_v25_0C),
]

rpc_ctrl_nvlink_get_inband_received_data_v25_0C = struct_rpc_ctrl_nvlink_get_inband_received_data_v25_0C
rpc_ctrl_nvlink_get_inband_received_data_v = struct_rpc_ctrl_nvlink_get_inband_received_data_v25_0C
class struct_rpc_fecs_error_v26_02(Structure):
    pass

struct_rpc_fecs_error_v26_02._pack_ = 1 # source:False
struct_rpc_fecs_error_v26_02._fields_ = [
    ('grIdx', ctypes.c_uint32),
    ('error_type', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 3),
]

rpc_fecs_error_v26_02 = struct_rpc_fecs_error_v26_02
rpc_fecs_error_v = struct_rpc_fecs_error_v26_02
class struct_rpc_ctrl_cmd_nvlink_inband_send_data_v26_05(Structure):
    pass

struct_rpc_ctrl_cmd_nvlink_inband_send_data_v26_05._pack_ = 1 # source:False
struct_rpc_ctrl_cmd_nvlink_inband_send_data_v26_05._fields_ = [
    ('buffer', ctypes.c_ubyte * 1024),
    ('dataSize', ctypes.c_uint32),
]

rpc_ctrl_cmd_nvlink_inband_send_data_v26_05 = struct_rpc_ctrl_cmd_nvlink_inband_send_data_v26_05
rpc_ctrl_cmd_nvlink_inband_send_data_v = struct_rpc_ctrl_cmd_nvlink_inband_send_data_v26_05
class struct_rpc_ctrl_cmd_internal_control_gsp_trace_v28_00(Structure):
    pass

struct_rpc_ctrl_cmd_internal_control_gsp_trace_v28_00._pack_ = 1 # source:False
struct_rpc_ctrl_cmd_internal_control_gsp_trace_v28_00._fields_ = [
    ('bufferSize', ctypes.c_uint32),
    ('tracepointMask', ctypes.c_uint32),
    ('bufferWatermark', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('bufferAddr', ctypes.c_uint64),
    ('flag', ctypes.c_ubyte),
    ('PADDING_1', ctypes.c_ubyte * 7),
]

rpc_ctrl_cmd_internal_control_gsp_trace_v28_00 = struct_rpc_ctrl_cmd_internal_control_gsp_trace_v28_00
rpc_ctrl_cmd_internal_control_gsp_trace_v = struct_rpc_ctrl_cmd_internal_control_gsp_trace_v28_00
class struct_rpc_recovery_action_v28_01(Structure):
    pass

struct_rpc_recovery_action_v28_01._pack_ = 1 # source:False
struct_rpc_recovery_action_v28_01._fields_ = [
    ('type', ctypes.c_uint32),
    ('value', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 3),
]

rpc_recovery_action_v28_01 = struct_rpc_recovery_action_v28_01
rpc_recovery_action_v = struct_rpc_recovery_action_v28_01
class struct_rpc_ctrl_subdevice_get_libos_heap_stats_v29_02(Structure):
    pass

class struct_NV2080_CTRL_CMD_GSP_GET_LIBOS_HEAP_STATS_PARAMS_v29_02(Structure):
    pass

class struct_NV2080_CTRL_GSP_LIBOS_POOL_STATS_v29_02(Structure):
    pass

struct_NV2080_CTRL_GSP_LIBOS_POOL_STATS_v29_02._pack_ = 1 # source:False
struct_NV2080_CTRL_GSP_LIBOS_POOL_STATS_v29_02._fields_ = [
    ('allocations', ctypes.c_uint32),
    ('peakAllocations', ctypes.c_uint32),
    ('objectSize', ctypes.c_uint64),
]

struct_NV2080_CTRL_CMD_GSP_GET_LIBOS_HEAP_STATS_PARAMS_v29_02._pack_ = 1 # source:False
struct_NV2080_CTRL_CMD_GSP_GET_LIBOS_HEAP_STATS_PARAMS_v29_02._fields_ = [
    ('poolStats', struct_NV2080_CTRL_GSP_LIBOS_POOL_STATS_v29_02 * 64),
    ('totalHeapSize', ctypes.c_uint64),
    ('poolCount', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 7),
]

struct_rpc_ctrl_subdevice_get_libos_heap_stats_v29_02._pack_ = 1 # source:False
struct_rpc_ctrl_subdevice_get_libos_heap_stats_v29_02._fields_ = [
    ('hClient', ctypes.c_uint32),
    ('hObject', ctypes.c_uint32),
    ('ctrlParams', struct_NV2080_CTRL_CMD_GSP_GET_LIBOS_HEAP_STATS_PARAMS_v29_02),
]

rpc_ctrl_subdevice_get_libos_heap_stats_v29_02 = struct_rpc_ctrl_subdevice_get_libos_heap_stats_v29_02
rpc_ctrl_subdevice_get_libos_heap_stats_v = struct_rpc_ctrl_subdevice_get_libos_heap_stats_v29_02
class struct_GSP_MSG_QUEUE_ELEMENT(Structure):
    pass

struct_GSP_MSG_QUEUE_ELEMENT._pack_ = 1 # source:False
struct_GSP_MSG_QUEUE_ELEMENT._fields_ = [
    ('authTagBuffer', ctypes.c_ubyte * 16),
    ('aadBuffer', ctypes.c_ubyte * 16),
    ('checkSum', ctypes.c_uint32),
    ('seqNum', ctypes.c_uint32),
    ('elemCount', ctypes.c_uint32),
    ('padding', ctypes.c_uint32),
]

GSP_MSG_QUEUE_ELEMENT = struct_GSP_MSG_QUEUE_ELEMENT
class union_rpc_message_rpc_union_field_v03_00(Union):
    pass

union_rpc_message_rpc_union_field_v03_00._pack_ = 1 # source:False
union_rpc_message_rpc_union_field_v03_00._fields_ = [
    ('spare', ctypes.c_uint32),
    ('cpuRmGfid', ctypes.c_uint32),
]

rpc_message_rpc_union_field_v03_00 = union_rpc_message_rpc_union_field_v03_00
rpc_message_rpc_union_field_v = union_rpc_message_rpc_union_field_v03_00
class struct_rpc_message_header_v03_00(Structure):
    pass

struct_rpc_message_header_v03_00._pack_ = 1 # source:False
struct_rpc_message_header_v03_00._fields_ = [
    ('header_version', ctypes.c_uint32),
    ('signature', ctypes.c_uint32),
    ('length', ctypes.c_uint32),
    ('function', ctypes.c_uint32),
    ('rpc_result', ctypes.c_uint32),
    ('rpc_result_private', ctypes.c_uint32),
    ('sequence', ctypes.c_uint32),
    ('u', rpc_message_rpc_union_field_v),
]

rpc_message_header_v03_00 = struct_rpc_message_header_v03_00
rpc_message_header_v = struct_rpc_message_header_v03_00
GSP_STATIC_CONFIG_H = True # macro
MAX_DSM_SUPPORTED_FUNCS_RTN_LEN = 8 # macro
NV_ACPI_GENERIC_FUNC_COUNT = 8 # macro
REGISTRY_TABLE_ENTRY_TYPE_UNKNOWN = 0 # macro
REGISTRY_TABLE_ENTRY_TYPE_DWORD = 1 # macro
REGISTRY_TABLE_ENTRY_TYPE_BINARY = 2 # macro
REGISTRY_TABLE_ENTRY_TYPE_STRING = 3 # macro
MAX_GROUP_COUNT = 2 # macro
RM_ENGINE_TYPE_COPY_SIZE = 20 # macro
RM_ENGINE_TYPE_NVENC_SIZE = 4 # macro
RM_ENGINE_TYPE_NVJPEG_SIZE = 8 # macro
RM_ENGINE_TYPE_NVDEC_SIZE = 8 # macro
RM_ENGINE_TYPE_OFA_SIZE = 2 # macro
RM_ENGINE_TYPE_GR_SIZE = 8 # macro
NVGPU_ENGINE_CAPS_MASK_BITS = 32 # macro
# def NVGPU_GET_ENGINE_CAPS_MASK(caps, id):  # macro
#    return (caps[(id)/32]&NVBIT((id)%32))
# def NVGPU_SET_ENGINE_CAPS_MASK(caps, id):  # macro
#    return (caps[(id)/32]|=NVBIT((id)%32))
class struct_PACKED_REGISTRY_ENTRY(Structure):
    pass

struct_PACKED_REGISTRY_ENTRY._pack_ = 1 # source:False
struct_PACKED_REGISTRY_ENTRY._fields_ = [
    ('nameOffset', ctypes.c_uint32),
    ('type', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 3),
    ('data', ctypes.c_uint32),
    ('length', ctypes.c_uint32),
]

PACKED_REGISTRY_ENTRY = struct_PACKED_REGISTRY_ENTRY
class struct_PACKED_REGISTRY_TABLE(Structure):
    pass

struct_PACKED_REGISTRY_TABLE._pack_ = 1 # source:False
struct_PACKED_REGISTRY_TABLE._fields_ = [
    ('size', ctypes.c_uint32),
    ('numEntries', ctypes.c_uint32),
]

PACKED_REGISTRY_TABLE = struct_PACKED_REGISTRY_TABLE

# values for enumeration 'c__EA_DISPMUXSTATE'
c__EA_DISPMUXSTATE__enumvalues = {
    0: 'dispMuxState_None',
    1: 'dispMuxState_IntegratedGPU',
    2: 'dispMuxState_DiscreteGPU',
}
dispMuxState_None = 0
dispMuxState_IntegratedGPU = 1
dispMuxState_DiscreteGPU = 2
c__EA_DISPMUXSTATE = ctypes.c_uint32 # enum
DISPMUXSTATE = c__EA_DISPMUXSTATE
DISPMUXSTATE__enumvalues = c__EA_DISPMUXSTATE__enumvalues
class struct_c__SA_ACPI_DSM_CACHE(Structure):
    pass

struct_c__SA_ACPI_DSM_CACHE._pack_ = 1 # source:False
struct_c__SA_ACPI_DSM_CACHE._fields_ = [
    ('suppFuncStatus', ctypes.c_uint32),
    ('suppFuncs', ctypes.c_ubyte * 8),
    ('suppFuncsLen', ctypes.c_uint32),
    ('bArg3isInteger', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 3),
    ('callbackStatus', ctypes.c_uint32),
    ('callback', ctypes.c_uint32),
]

ACPI_DSM_CACHE = struct_c__SA_ACPI_DSM_CACHE
class struct_c__SA_ACPI_DATA(Structure):
    pass


# values for enumeration '_ACPI_DSM_FUNCTION'
_ACPI_DSM_FUNCTION__enumvalues = {
    0: 'ACPI_DSM_FUNCTION_NBSI',
    1: 'ACPI_DSM_FUNCTION_NVHG',
    2: 'ACPI_DSM_FUNCTION_MXM',
    3: 'ACPI_DSM_FUNCTION_NBCI',
    4: 'ACPI_DSM_FUNCTION_NVOP',
    5: 'ACPI_DSM_FUNCTION_PCFG',
    6: 'ACPI_DSM_FUNCTION_GPS_2X',
    7: 'ACPI_DSM_FUNCTION_JT',
    8: 'ACPI_DSM_FUNCTION_PEX',
    9: 'ACPI_DSM_FUNCTION_NVPCF_2X',
    10: 'ACPI_DSM_FUNCTION_GPS',
    11: 'ACPI_DSM_FUNCTION_NVPCF',
    12: 'ACPI_DSM_FUNCTION_COUNT',
    13: 'ACPI_DSM_FUNCTION_CURRENT',
    255: 'ACPI_DSM_FUNCTION_INVALID',
}
ACPI_DSM_FUNCTION_NBSI = 0
ACPI_DSM_FUNCTION_NVHG = 1
ACPI_DSM_FUNCTION_MXM = 2
ACPI_DSM_FUNCTION_NBCI = 3
ACPI_DSM_FUNCTION_NVOP = 4
ACPI_DSM_FUNCTION_PCFG = 5
ACPI_DSM_FUNCTION_GPS_2X = 6
ACPI_DSM_FUNCTION_JT = 7
ACPI_DSM_FUNCTION_PEX = 8
ACPI_DSM_FUNCTION_NVPCF_2X = 9
ACPI_DSM_FUNCTION_GPS = 10
ACPI_DSM_FUNCTION_NVPCF = 11
ACPI_DSM_FUNCTION_COUNT = 12
ACPI_DSM_FUNCTION_CURRENT = 13
ACPI_DSM_FUNCTION_INVALID = 255
_ACPI_DSM_FUNCTION = ctypes.c_uint32 # enum
struct_c__SA_ACPI_DATA._pack_ = 1 # source:False
struct_c__SA_ACPI_DATA._fields_ = [
    ('dsm', struct_c__SA_ACPI_DSM_CACHE * 12),
    ('dispStatusHotplugFunc', _ACPI_DSM_FUNCTION),
    ('dispStatusConfigFunc', _ACPI_DSM_FUNCTION),
    ('perfPostPowerStateFunc', _ACPI_DSM_FUNCTION),
    ('stereo3dStateActiveFunc', _ACPI_DSM_FUNCTION),
    ('dsmPlatCapsCache', ctypes.c_uint32 * 12),
    ('MDTLFeatureSupport', ctypes.c_uint32),
    ('dsmCurrentFunc', _ACPI_DSM_FUNCTION * 8),
    ('dsmCurrentSubFunc', ctypes.c_uint32 * 8),
    ('dsmCurrentFuncSupport', ctypes.c_uint32),
]

ACPI_DATA = struct_c__SA_ACPI_DATA
class struct_DOD_METHOD_DATA(Structure):
    pass

struct_DOD_METHOD_DATA._pack_ = 1 # source:False
struct_DOD_METHOD_DATA._fields_ = [
    ('status', ctypes.c_uint32),
    ('acpiIdListLen', ctypes.c_uint32),
    ('acpiIdList', ctypes.c_uint32 * 16),
]

DOD_METHOD_DATA = struct_DOD_METHOD_DATA
class struct_JT_METHOD_DATA(Structure):
    pass

struct_JT_METHOD_DATA._pack_ = 1 # source:False
struct_JT_METHOD_DATA._fields_ = [
    ('status', ctypes.c_uint32),
    ('jtCaps', ctypes.c_uint32),
    ('jtRevId', ctypes.c_uint16),
    ('bSBIOSCaps', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte),
]

JT_METHOD_DATA = struct_JT_METHOD_DATA
class struct_MUX_METHOD_DATA_ELEMENT(Structure):
    pass

struct_MUX_METHOD_DATA_ELEMENT._pack_ = 1 # source:False
struct_MUX_METHOD_DATA_ELEMENT._fields_ = [
    ('acpiId', ctypes.c_uint32),
    ('mode', ctypes.c_uint32),
    ('status', ctypes.c_uint32),
]

MUX_METHOD_DATA_ELEMENT = struct_MUX_METHOD_DATA_ELEMENT
class struct_MUX_METHOD_DATA(Structure):
    pass

struct_MUX_METHOD_DATA._pack_ = 1 # source:False
struct_MUX_METHOD_DATA._fields_ = [
    ('tableLen', ctypes.c_uint32),
    ('acpiIdMuxModeTable', struct_MUX_METHOD_DATA_ELEMENT * 16),
    ('acpiIdMuxPartTable', struct_MUX_METHOD_DATA_ELEMENT * 16),
    ('acpiIdMuxStateTable', struct_MUX_METHOD_DATA_ELEMENT * 16),
]

MUX_METHOD_DATA = struct_MUX_METHOD_DATA
class struct_CAPS_METHOD_DATA(Structure):
    pass

struct_CAPS_METHOD_DATA._pack_ = 1 # source:False
struct_CAPS_METHOD_DATA._fields_ = [
    ('status', ctypes.c_uint32),
    ('optimusCaps', ctypes.c_uint32),
]

CAPS_METHOD_DATA = struct_CAPS_METHOD_DATA
class struct_ACPI_METHOD_DATA(Structure):
    pass

struct_ACPI_METHOD_DATA._pack_ = 1 # source:False
struct_ACPI_METHOD_DATA._fields_ = [
    ('bValid', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 3),
    ('dodMethodData', DOD_METHOD_DATA),
    ('jtMethodData', JT_METHOD_DATA),
    ('muxMethodData', MUX_METHOD_DATA),
    ('capsMethodData', CAPS_METHOD_DATA),
]

ACPI_METHOD_DATA = struct_ACPI_METHOD_DATA

# values for enumeration 'c__EA_RM_ENGINE_TYPE'
c__EA_RM_ENGINE_TYPE__enumvalues = {
    0: 'RM_ENGINE_TYPE_NULL',
    1: 'RM_ENGINE_TYPE_GR0',
    2: 'RM_ENGINE_TYPE_GR1',
    3: 'RM_ENGINE_TYPE_GR2',
    4: 'RM_ENGINE_TYPE_GR3',
    5: 'RM_ENGINE_TYPE_GR4',
    6: 'RM_ENGINE_TYPE_GR5',
    7: 'RM_ENGINE_TYPE_GR6',
    8: 'RM_ENGINE_TYPE_GR7',
    9: 'RM_ENGINE_TYPE_COPY0',
    10: 'RM_ENGINE_TYPE_COPY1',
    11: 'RM_ENGINE_TYPE_COPY2',
    12: 'RM_ENGINE_TYPE_COPY3',
    13: 'RM_ENGINE_TYPE_COPY4',
    14: 'RM_ENGINE_TYPE_COPY5',
    15: 'RM_ENGINE_TYPE_COPY6',
    16: 'RM_ENGINE_TYPE_COPY7',
    17: 'RM_ENGINE_TYPE_COPY8',
    18: 'RM_ENGINE_TYPE_COPY9',
    19: 'RM_ENGINE_TYPE_COPY10',
    20: 'RM_ENGINE_TYPE_COPY11',
    21: 'RM_ENGINE_TYPE_COPY12',
    22: 'RM_ENGINE_TYPE_COPY13',
    23: 'RM_ENGINE_TYPE_COPY14',
    24: 'RM_ENGINE_TYPE_COPY15',
    25: 'RM_ENGINE_TYPE_COPY16',
    26: 'RM_ENGINE_TYPE_COPY17',
    27: 'RM_ENGINE_TYPE_COPY18',
    28: 'RM_ENGINE_TYPE_COPY19',
    29: 'RM_ENGINE_TYPE_NVDEC0',
    30: 'RM_ENGINE_TYPE_NVDEC1',
    31: 'RM_ENGINE_TYPE_NVDEC2',
    32: 'RM_ENGINE_TYPE_NVDEC3',
    33: 'RM_ENGINE_TYPE_NVDEC4',
    34: 'RM_ENGINE_TYPE_NVDEC5',
    35: 'RM_ENGINE_TYPE_NVDEC6',
    36: 'RM_ENGINE_TYPE_NVDEC7',
    37: 'RM_ENGINE_TYPE_NVENC0',
    38: 'RM_ENGINE_TYPE_NVENC1',
    39: 'RM_ENGINE_TYPE_NVENC2',
    40: 'RM_ENGINE_TYPE_NVENC3',
    41: 'RM_ENGINE_TYPE_VP',
    42: 'RM_ENGINE_TYPE_ME',
    43: 'RM_ENGINE_TYPE_PPP',
    44: 'RM_ENGINE_TYPE_MPEG',
    45: 'RM_ENGINE_TYPE_SW',
    46: 'RM_ENGINE_TYPE_TSEC',
    47: 'RM_ENGINE_TYPE_VIC',
    48: 'RM_ENGINE_TYPE_MP',
    49: 'RM_ENGINE_TYPE_SEC2',
    50: 'RM_ENGINE_TYPE_HOST',
    51: 'RM_ENGINE_TYPE_DPU',
    52: 'RM_ENGINE_TYPE_PMU',
    53: 'RM_ENGINE_TYPE_FBFLCN',
    54: 'RM_ENGINE_TYPE_NVJPEG0',
    55: 'RM_ENGINE_TYPE_NVJPEG1',
    56: 'RM_ENGINE_TYPE_NVJPEG2',
    57: 'RM_ENGINE_TYPE_NVJPEG3',
    58: 'RM_ENGINE_TYPE_NVJPEG4',
    59: 'RM_ENGINE_TYPE_NVJPEG5',
    60: 'RM_ENGINE_TYPE_NVJPEG6',
    61: 'RM_ENGINE_TYPE_NVJPEG7',
    62: 'RM_ENGINE_TYPE_OFA0',
    63: 'RM_ENGINE_TYPE_OFA1',
    64: 'RM_ENGINE_TYPE_RESERVED40',
    65: 'RM_ENGINE_TYPE_RESERVED41',
    66: 'RM_ENGINE_TYPE_RESERVED42',
    67: 'RM_ENGINE_TYPE_RESERVED43',
    68: 'RM_ENGINE_TYPE_RESERVED44',
    69: 'RM_ENGINE_TYPE_RESERVED45',
    70: 'RM_ENGINE_TYPE_RESERVED46',
    71: 'RM_ENGINE_TYPE_RESERVED47',
    72: 'RM_ENGINE_TYPE_RESERVED48',
    73: 'RM_ENGINE_TYPE_RESERVED49',
    74: 'RM_ENGINE_TYPE_RESERVED4a',
    75: 'RM_ENGINE_TYPE_RESERVED4b',
    76: 'RM_ENGINE_TYPE_RESERVED4c',
    77: 'RM_ENGINE_TYPE_RESERVED4d',
    78: 'RM_ENGINE_TYPE_RESERVED4e',
    79: 'RM_ENGINE_TYPE_RESERVED4f',
    80: 'RM_ENGINE_TYPE_RESERVED50',
    81: 'RM_ENGINE_TYPE_RESERVED51',
    82: 'RM_ENGINE_TYPE_RESERVED52',
    83: 'RM_ENGINE_TYPE_RESERVED53',
    84: 'RM_ENGINE_TYPE_LAST',
}
RM_ENGINE_TYPE_NULL = 0
RM_ENGINE_TYPE_GR0 = 1
RM_ENGINE_TYPE_GR1 = 2
RM_ENGINE_TYPE_GR2 = 3
RM_ENGINE_TYPE_GR3 = 4
RM_ENGINE_TYPE_GR4 = 5
RM_ENGINE_TYPE_GR5 = 6
RM_ENGINE_TYPE_GR6 = 7
RM_ENGINE_TYPE_GR7 = 8
RM_ENGINE_TYPE_COPY0 = 9
RM_ENGINE_TYPE_COPY1 = 10
RM_ENGINE_TYPE_COPY2 = 11
RM_ENGINE_TYPE_COPY3 = 12
RM_ENGINE_TYPE_COPY4 = 13
RM_ENGINE_TYPE_COPY5 = 14
RM_ENGINE_TYPE_COPY6 = 15
RM_ENGINE_TYPE_COPY7 = 16
RM_ENGINE_TYPE_COPY8 = 17
RM_ENGINE_TYPE_COPY9 = 18
RM_ENGINE_TYPE_COPY10 = 19
RM_ENGINE_TYPE_COPY11 = 20
RM_ENGINE_TYPE_COPY12 = 21
RM_ENGINE_TYPE_COPY13 = 22
RM_ENGINE_TYPE_COPY14 = 23
RM_ENGINE_TYPE_COPY15 = 24
RM_ENGINE_TYPE_COPY16 = 25
RM_ENGINE_TYPE_COPY17 = 26
RM_ENGINE_TYPE_COPY18 = 27
RM_ENGINE_TYPE_COPY19 = 28
RM_ENGINE_TYPE_NVDEC0 = 29
RM_ENGINE_TYPE_NVDEC1 = 30
RM_ENGINE_TYPE_NVDEC2 = 31
RM_ENGINE_TYPE_NVDEC3 = 32
RM_ENGINE_TYPE_NVDEC4 = 33
RM_ENGINE_TYPE_NVDEC5 = 34
RM_ENGINE_TYPE_NVDEC6 = 35
RM_ENGINE_TYPE_NVDEC7 = 36
RM_ENGINE_TYPE_NVENC0 = 37
RM_ENGINE_TYPE_NVENC1 = 38
RM_ENGINE_TYPE_NVENC2 = 39
RM_ENGINE_TYPE_NVENC3 = 40
RM_ENGINE_TYPE_VP = 41
RM_ENGINE_TYPE_ME = 42
RM_ENGINE_TYPE_PPP = 43
RM_ENGINE_TYPE_MPEG = 44
RM_ENGINE_TYPE_SW = 45
RM_ENGINE_TYPE_TSEC = 46
RM_ENGINE_TYPE_VIC = 47
RM_ENGINE_TYPE_MP = 48
RM_ENGINE_TYPE_SEC2 = 49
RM_ENGINE_TYPE_HOST = 50
RM_ENGINE_TYPE_DPU = 51
RM_ENGINE_TYPE_PMU = 52
RM_ENGINE_TYPE_FBFLCN = 53
RM_ENGINE_TYPE_NVJPEG0 = 54
RM_ENGINE_TYPE_NVJPEG1 = 55
RM_ENGINE_TYPE_NVJPEG2 = 56
RM_ENGINE_TYPE_NVJPEG3 = 57
RM_ENGINE_TYPE_NVJPEG4 = 58
RM_ENGINE_TYPE_NVJPEG5 = 59
RM_ENGINE_TYPE_NVJPEG6 = 60
RM_ENGINE_TYPE_NVJPEG7 = 61
RM_ENGINE_TYPE_OFA0 = 62
RM_ENGINE_TYPE_OFA1 = 63
RM_ENGINE_TYPE_RESERVED40 = 64
RM_ENGINE_TYPE_RESERVED41 = 65
RM_ENGINE_TYPE_RESERVED42 = 66
RM_ENGINE_TYPE_RESERVED43 = 67
RM_ENGINE_TYPE_RESERVED44 = 68
RM_ENGINE_TYPE_RESERVED45 = 69
RM_ENGINE_TYPE_RESERVED46 = 70
RM_ENGINE_TYPE_RESERVED47 = 71
RM_ENGINE_TYPE_RESERVED48 = 72
RM_ENGINE_TYPE_RESERVED49 = 73
RM_ENGINE_TYPE_RESERVED4a = 74
RM_ENGINE_TYPE_RESERVED4b = 75
RM_ENGINE_TYPE_RESERVED4c = 76
RM_ENGINE_TYPE_RESERVED4d = 77
RM_ENGINE_TYPE_RESERVED4e = 78
RM_ENGINE_TYPE_RESERVED4f = 79
RM_ENGINE_TYPE_RESERVED50 = 80
RM_ENGINE_TYPE_RESERVED51 = 81
RM_ENGINE_TYPE_RESERVED52 = 82
RM_ENGINE_TYPE_RESERVED53 = 83
RM_ENGINE_TYPE_LAST = 84
c__EA_RM_ENGINE_TYPE = ctypes.c_uint32 # enum
RM_ENGINE_TYPE_GRAPHICS = RM_ENGINE_TYPE_GR0 # macro
RM_ENGINE_TYPE_BSP = RM_ENGINE_TYPE_NVDEC0 # macro
RM_ENGINE_TYPE_MSENC = RM_ENGINE_TYPE_NVENC0 # macro
RM_ENGINE_TYPE_CIPHER = RM_ENGINE_TYPE_TSEC # macro
RM_ENGINE_TYPE_NVJPG = RM_ENGINE_TYPE_NVJPEG0 # macro
NVGPU_ENGINE_CAPS_MASK_ARRAY_MAX = ((RM_ENGINE_TYPE_LAST-1)/32+1) # macro
RM_ENGINE_TYPE = c__EA_RM_ENGINE_TYPE
RM_ENGINE_TYPE__enumvalues = c__EA_RM_ENGINE_TYPE__enumvalues
class struct_c__SA_BUSINFO(Structure):
    pass

struct_c__SA_BUSINFO._pack_ = 1 # source:False
struct_c__SA_BUSINFO._fields_ = [
    ('deviceID', ctypes.c_uint16),
    ('vendorID', ctypes.c_uint16),
    ('subdeviceID', ctypes.c_uint16),
    ('subvendorID', ctypes.c_uint16),
    ('revisionID', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte),
]

BUSINFO = struct_c__SA_BUSINFO
class struct_GSP_VF_INFO(Structure):
    pass

struct_GSP_VF_INFO._pack_ = 1 # source:False
struct_GSP_VF_INFO._fields_ = [
    ('totalVFs', ctypes.c_uint32),
    ('firstVFOffset', ctypes.c_uint32),
    ('FirstVFBar0Address', ctypes.c_uint64),
    ('FirstVFBar1Address', ctypes.c_uint64),
    ('FirstVFBar2Address', ctypes.c_uint64),
    ('b64bitBar0', ctypes.c_ubyte),
    ('b64bitBar1', ctypes.c_ubyte),
    ('b64bitBar2', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 5),
]

GSP_VF_INFO = struct_GSP_VF_INFO
class struct_c__SA_GSP_PCIE_CONFIG_REG(Structure):
    pass

struct_c__SA_GSP_PCIE_CONFIG_REG._pack_ = 1 # source:False
struct_c__SA_GSP_PCIE_CONFIG_REG._fields_ = [
    ('linkCap', ctypes.c_uint32),
]

GSP_PCIE_CONFIG_REG = struct_c__SA_GSP_PCIE_CONFIG_REG
class struct_c__SA_EcidManufacturingInfo(Structure):
    pass

struct_c__SA_EcidManufacturingInfo._pack_ = 1 # source:False
struct_c__SA_EcidManufacturingInfo._fields_ = [
    ('ecidLow', ctypes.c_uint32),
    ('ecidHigh', ctypes.c_uint32),
    ('ecidExtended', ctypes.c_uint32),
]

EcidManufacturingInfo = struct_c__SA_EcidManufacturingInfo
class struct_c__SA_FW_WPR_LAYOUT_OFFSET(Structure):
    pass

struct_c__SA_FW_WPR_LAYOUT_OFFSET._pack_ = 1 # source:False
struct_c__SA_FW_WPR_LAYOUT_OFFSET._fields_ = [
    ('nonWprHeapOffset', ctypes.c_uint64),
    ('frtsOffset', ctypes.c_uint64),
]

FW_WPR_LAYOUT_OFFSET = struct_c__SA_FW_WPR_LAYOUT_OFFSET
class struct_GspStaticConfigInfo_t(Structure):
    pass

class struct_NV2080_CTRL_GPU_GET_GID_INFO_PARAMS(Structure):
    pass

struct_NV2080_CTRL_GPU_GET_GID_INFO_PARAMS._pack_ = 1 # source:False
struct_NV2080_CTRL_GPU_GET_GID_INFO_PARAMS._fields_ = [
    ('index', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
    ('length', ctypes.c_uint32),
    ('data', ctypes.c_ubyte * 256),
]

class struct_NV2080_CTRL_BIOS_GET_SKU_INFO_PARAMS(Structure):
    pass

struct_NV2080_CTRL_BIOS_GET_SKU_INFO_PARAMS._pack_ = 1 # source:False
struct_NV2080_CTRL_BIOS_GET_SKU_INFO_PARAMS._fields_ = [
    ('BoardID', ctypes.c_uint32),
    ('chipSKU', ctypes.c_char * 9),
    ('chipSKUMod', ctypes.c_char * 5),
    ('PADDING_0', ctypes.c_ubyte * 2),
    ('skuConfigVersion', ctypes.c_uint32),
    ('project', ctypes.c_char * 5),
    ('projectSKU', ctypes.c_char * 5),
    ('CDP', ctypes.c_char * 6),
    ('projectSKUMod', ctypes.c_char * 2),
    ('PADDING_1', ctypes.c_ubyte * 2),
    ('businessCycle', ctypes.c_uint32),
]

class struct_NV2080_CTRL_CMD_FB_GET_FB_REGION_INFO_PARAMS(Structure):
    pass

class struct_NV2080_CTRL_CMD_FB_GET_FB_REGION_FB_REGION_INFO(Structure):
    pass

struct_NV2080_CTRL_CMD_FB_GET_FB_REGION_FB_REGION_INFO._pack_ = 1 # source:False
struct_NV2080_CTRL_CMD_FB_GET_FB_REGION_FB_REGION_INFO._fields_ = [
    ('base', ctypes.c_uint64),
    ('limit', ctypes.c_uint64),
    ('reserved', ctypes.c_uint64),
    ('performance', ctypes.c_uint32),
    ('supportCompressed', ctypes.c_ubyte),
    ('supportISO', ctypes.c_ubyte),
    ('bProtected', ctypes.c_ubyte),
    ('blackList', ctypes.c_ubyte * 17),
]

struct_NV2080_CTRL_CMD_FB_GET_FB_REGION_INFO_PARAMS._pack_ = 1 # source:False
struct_NV2080_CTRL_CMD_FB_GET_FB_REGION_INFO_PARAMS._fields_ = [
    ('numFBRegions', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('fbRegion', struct_NV2080_CTRL_CMD_FB_GET_FB_REGION_FB_REGION_INFO * 16),
]

class struct_NV0080_CTRL_GPU_GET_SRIOV_CAPS_PARAMS(Structure):
    pass

struct_NV0080_CTRL_GPU_GET_SRIOV_CAPS_PARAMS._pack_ = 1 # source:False
struct_NV0080_CTRL_GPU_GET_SRIOV_CAPS_PARAMS._fields_ = [
    ('totalVFs', ctypes.c_uint32),
    ('firstVfOffset', ctypes.c_uint32),
    ('vfFeatureMask', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('FirstVFBar0Address', ctypes.c_uint64),
    ('FirstVFBar1Address', ctypes.c_uint64),
    ('FirstVFBar2Address', ctypes.c_uint64),
    ('bar0Size', ctypes.c_uint64),
    ('bar1Size', ctypes.c_uint64),
    ('bar2Size', ctypes.c_uint64),
    ('b64bitBar0', ctypes.c_ubyte),
    ('b64bitBar1', ctypes.c_ubyte),
    ('b64bitBar2', ctypes.c_ubyte),
    ('bSriovEnabled', ctypes.c_ubyte),
    ('bSriovHeavyEnabled', ctypes.c_ubyte),
    ('bEmulateVFBar0TlbInvalidationRegister', ctypes.c_ubyte),
    ('bClientRmAllocatedCtxBuffer', ctypes.c_ubyte),
    ('bNonPowerOf2ChannelCountSupported', ctypes.c_ubyte),
    ('bVfResizableBAR1Supported', ctypes.c_ubyte),
    ('PADDING_1', ctypes.c_ubyte * 7),
]

struct_GspStaticConfigInfo_t._pack_ = 1 # source:False
struct_GspStaticConfigInfo_t._fields_ = [
    ('grCapsBits', ctypes.c_ubyte * 23),
    ('PADDING_0', ctypes.c_ubyte),
    ('gidInfo', struct_NV2080_CTRL_GPU_GET_GID_INFO_PARAMS),
    ('SKUInfo', struct_NV2080_CTRL_BIOS_GET_SKU_INFO_PARAMS),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('fbRegionInfoParams', struct_NV2080_CTRL_CMD_FB_GET_FB_REGION_INFO_PARAMS),
    ('sriovCaps', struct_NV0080_CTRL_GPU_GET_SRIOV_CAPS_PARAMS),
    ('sriovMaxGfid', ctypes.c_uint32),
    ('engineCaps', ctypes.c_uint32 * 3),
    ('poisonFuseEnabled', ctypes.c_ubyte),
    ('PADDING_2', ctypes.c_ubyte * 7),
    ('fb_length', ctypes.c_uint64),
    ('fbio_mask', ctypes.c_uint64),
    ('fb_bus_width', ctypes.c_uint32),
    ('fb_ram_type', ctypes.c_uint32),
    ('fbp_mask', ctypes.c_uint64),
    ('l2_cache_size', ctypes.c_uint32),
    ('gpuNameString', ctypes.c_ubyte * 64),
    ('gpuShortNameString', ctypes.c_ubyte * 64),
    ('gpuNameString_Unicode', ctypes.c_uint16 * 64),
    ('bGpuInternalSku', ctypes.c_ubyte),
    ('bIsQuadroGeneric', ctypes.c_ubyte),
    ('bIsQuadroAd', ctypes.c_ubyte),
    ('bIsNvidiaNvs', ctypes.c_ubyte),
    ('bIsVgx', ctypes.c_ubyte),
    ('bGeforceSmb', ctypes.c_ubyte),
    ('bIsTitan', ctypes.c_ubyte),
    ('bIsTesla', ctypes.c_ubyte),
    ('bIsMobile', ctypes.c_ubyte),
    ('bIsGc6Rtd3Allowed', ctypes.c_ubyte),
    ('bIsGc8Rtd3Allowed', ctypes.c_ubyte),
    ('bIsGcOffRtd3Allowed', ctypes.c_ubyte),
    ('bIsGcoffLegacyAllowed', ctypes.c_ubyte),
    ('bIsMigSupported', ctypes.c_ubyte),
    ('RTD3GC6TotalBoardPower', ctypes.c_uint16),
    ('RTD3GC6PerstDelay', ctypes.c_uint16),
    ('PADDING_3', ctypes.c_ubyte * 2),
    ('bar1PdeBase', ctypes.c_uint64),
    ('bar2PdeBase', ctypes.c_uint64),
    ('bVbiosValid', ctypes.c_ubyte),
    ('PADDING_4', ctypes.c_ubyte * 3),
    ('vbiosSubVendor', ctypes.c_uint32),
    ('vbiosSubDevice', ctypes.c_uint32),
    ('bPageRetirementSupported', ctypes.c_ubyte),
    ('bSplitVasBetweenServerClientRm', ctypes.c_ubyte),
    ('bClRootportNeedsNosnoopWAR', ctypes.c_ubyte),
    ('PADDING_5', ctypes.c_ubyte),
    ('displaylessMaxHeads', VIRTUAL_DISPLAY_GET_NUM_HEADS_PARAMS),
    ('displaylessMaxResolution', VIRTUAL_DISPLAY_GET_MAX_RESOLUTION_PARAMS),
    ('PADDING_6', ctypes.c_ubyte * 4),
    ('displaylessMaxPixels', ctypes.c_uint64),
    ('hInternalClient', ctypes.c_uint32),
    ('hInternalDevice', ctypes.c_uint32),
    ('hInternalSubdevice', ctypes.c_uint32),
    ('bSelfHostedMode', ctypes.c_ubyte),
    ('bAtsSupported', ctypes.c_ubyte),
    ('bIsGpuUefi', ctypes.c_ubyte),
    ('bIsEfiInit', ctypes.c_ubyte),
    ('ecidInfo', struct_c__SA_EcidManufacturingInfo * 2),
    ('fwWprLayoutOffset', FW_WPR_LAYOUT_OFFSET),
]

GspStaticConfigInfo = struct_GspStaticConfigInfo_t
class struct_GspSystemInfo(Structure):
    pass

struct_GspSystemInfo._pack_ = 1 # source:False
struct_GspSystemInfo._fields_ = [
    ('gpuPhysAddr', ctypes.c_uint64),
    ('gpuPhysFbAddr', ctypes.c_uint64),
    ('gpuPhysInstAddr', ctypes.c_uint64),
    ('gpuPhysIoAddr', ctypes.c_uint64),
    ('nvDomainBusDeviceFunc', ctypes.c_uint64),
    ('simAccessBufPhysAddr', ctypes.c_uint64),
    ('notifyOpSharedSurfacePhysAddr', ctypes.c_uint64),
    ('pcieAtomicsOpMask', ctypes.c_uint64),
    ('consoleMemSize', ctypes.c_uint64),
    ('maxUserVa', ctypes.c_uint64),
    ('pciConfigMirrorBase', ctypes.c_uint32),
    ('pciConfigMirrorSize', ctypes.c_uint32),
    ('PCIDeviceID', ctypes.c_uint32),
    ('PCISubDeviceID', ctypes.c_uint32),
    ('PCIRevisionID', ctypes.c_uint32),
    ('pcieAtomicsCplDeviceCapMask', ctypes.c_uint32),
    ('oorArch', ctypes.c_ubyte),
    ('PADDING_0', ctypes.c_ubyte * 7),
    ('clPdbProperties', ctypes.c_uint64),
    ('Chipset', ctypes.c_uint32),
    ('bGpuBehindBridge', ctypes.c_ubyte),
    ('bFlrSupported', ctypes.c_ubyte),
    ('b64bBar0Supported', ctypes.c_ubyte),
    ('bMnocAvailable', ctypes.c_ubyte),
    ('chipsetL1ssEnable', ctypes.c_uint32),
    ('bUpstreamL0sUnsupported', ctypes.c_ubyte),
    ('bUpstreamL1Unsupported', ctypes.c_ubyte),
    ('bUpstreamL1PorSupported', ctypes.c_ubyte),
    ('bUpstreamL1PorMobileOnly', ctypes.c_ubyte),
    ('bSystemHasMux', ctypes.c_ubyte),
    ('upstreamAddressValid', ctypes.c_ubyte),
    ('FHBBusInfo', BUSINFO),
    ('chipsetIDInfo', BUSINFO),
    ('PADDING_1', ctypes.c_ubyte * 2),
    ('acpiMethodData', ACPI_METHOD_DATA),
    ('hypervisorType', ctypes.c_uint32),
    ('bIsPassthru', ctypes.c_ubyte),
    ('PADDING_2', ctypes.c_ubyte * 7),
    ('sysTimerOffsetNs', ctypes.c_uint64),
    ('gspVFInfo', GSP_VF_INFO),
    ('bIsPrimary', ctypes.c_ubyte),
    ('isGridBuild', ctypes.c_ubyte),
    ('PADDING_3', ctypes.c_ubyte * 2),
    ('pcieConfigReg', GSP_PCIE_CONFIG_REG),
    ('gridBuildCsp', ctypes.c_uint32),
    ('bPreserveVideoMemoryAllocations', ctypes.c_ubyte),
    ('bTdrEventSupported', ctypes.c_ubyte),
    ('bFeatureStretchVblankCapable', ctypes.c_ubyte),
    ('bEnableDynamicGranularityPageArrays', ctypes.c_ubyte),
    ('bClockBoostSupported', ctypes.c_ubyte),
    ('bRouteDispIntrsToCPU', ctypes.c_ubyte),
    ('PADDING_4', ctypes.c_ubyte * 6),
    ('hostPageSize', ctypes.c_uint64),
]

GspSystemInfo = struct_GspSystemInfo
VBIOS_H = True # macro
FALCON_APPLICATION_INTERFACE_ENTRY_ID_DMEMMAPPER = (0x4) # macro
FALCON_APPLICATION_INTERFACE_DMEM_MAPPER_V3_CMD_FRTS = (0x15) # macro
FALCON_APPLICATION_INTERFACE_DMEM_MAPPER_V3_CMD_SB = (0x19) # macro
BIT_HEADER_ID = 0xB8FF # macro
BIT_HEADER_SIGNATURE = 0x00544942 # macro
BIT_HEADER_SIZE_OFFSET = 8 # macro
BIT_HEADER_V1_00_FMT = "1w1d1w4b" # macro
BIT_TOKEN_V1_00_SIZE_6 = 6 # macro
BIT_TOKEN_V1_00_SIZE_8 = 8 # macro
BIT_TOKEN_V1_00_FMT_SIZE_6 = "2b2w" # macro
BIT_TOKEN_V1_00_FMT_SIZE_8 = "2b1w1d" # macro
BIT_TOKEN_BIOSDATA = 0x42 # macro
BIT_DATA_BIOSDATA_VERSION_1 = 0x1 # macro
BIT_DATA_BIOSDATA_VERSION_2 = 0x2 # macro
BIT_DATA_BIOSDATA_BINVER_FMT = "1d1b" # macro
BIT_DATA_BIOSDATA_BINVER_SIZE_5 = 5 # macro
BIT_TOKEN_FALCON_DATA = 0x70 # macro
BIT_DATA_FALCON_DATA_V2_4_FMT = "1d" # macro
BIT_DATA_FALCON_DATA_V2_SIZE_4 = 4 # macro
FALCON_UCODE_TABLE_HDR_V1_VERSION = 1 # macro
FALCON_UCODE_TABLE_HDR_V1_SIZE_6 = 6 # macro
FALCON_UCODE_TABLE_HDR_V1_6_FMT = "6b" # macro
FALCON_UCODE_TABLE_ENTRY_V1_VERSION = 1 # macro
FALCON_UCODE_TABLE_ENTRY_V1_SIZE_6 = 6 # macro
FALCON_UCODE_TABLE_ENTRY_V1_6_FMT = "2b1d" # macro
FALCON_UCODE_ENTRY_APPID_FIRMWARE_SEC_LIC = 0x05 # macro
FALCON_UCODE_ENTRY_APPID_FWSEC_DBG = 0x45 # macro
FALCON_UCODE_ENTRY_APPID_FWSEC_PROD = 0x85 # macro
# NV_BIT_FALCON_UCODE_DESC_HEADER_VDESC_FLAGS_VERSION = 0 : 0 # macro
NV_BIT_FALCON_UCODE_DESC_HEADER_VDESC_FLAGS_VERSION_UNAVAILABLE = 0x00 # macro
NV_BIT_FALCON_UCODE_DESC_HEADER_VDESC_FLAGS_VERSION_AVAILABLE = 0x01 # macro
# NV_BIT_FALCON_UCODE_DESC_HEADER_VDESC_FLAGS_RESERVED = 1 : 1 # macro
# NV_BIT_FALCON_UCODE_DESC_HEADER_VDESC_FLAGS_ENCRYPTED = 2 : 2 # macro
# NV_BIT_FALCON_UCODE_DESC_HEADER_VDESC_RESERVED = 7 : 3 # macro
# NV_BIT_FALCON_UCODE_DESC_HEADER_VDESC_VERSION = 15 : 8 # macro
NV_BIT_FALCON_UCODE_DESC_HEADER_VDESC_VERSION_V1 = 0x01 # macro
NV_BIT_FALCON_UCODE_DESC_HEADER_VDESC_VERSION_V2 = 0x02 # macro
NV_BIT_FALCON_UCODE_DESC_HEADER_VDESC_VERSION_V3 = 0x03 # macro
NV_BIT_FALCON_UCODE_DESC_HEADER_VDESC_VERSION_V4 = 0x04 # macro
# NV_BIT_FALCON_UCODE_DESC_HEADER_VDESC_SIZE = 31 : 16 # macro
FALCON_UCODE_DESC_HEADER_FORMAT = "1d" # macro
FALCON_UCODE_DESC_V3_SIZE_44 = 44 # macro
FALCON_UCODE_DESC_V3_44_FMT = "9d1w2b2w" # macro
BCRT30_RSA3K_SIG_SIZE = 384 # macro
FWSECLIC_READ_VBIOS_STRUCT_FLAGS = (2) # macro
FWSECLIC_FRTS_REGION_MEDIA_FB = (2) # macro
FWSECLIC_FRTS_REGION_SIZE_1MB_IN_4K = (0x100) # macro
class struct_c__SA_FALCON_APPLICATION_INTERFACE_DMEM_MAPPER_V3(Structure):
    pass

struct_c__SA_FALCON_APPLICATION_INTERFACE_DMEM_MAPPER_V3._pack_ = 1 # source:True
struct_c__SA_FALCON_APPLICATION_INTERFACE_DMEM_MAPPER_V3._fields_ = [
    ('signature', ctypes.c_uint32),
    ('version', ctypes.c_uint16),
    ('size', ctypes.c_uint16),
    ('cmd_in_buffer_offset', ctypes.c_uint32),
    ('cmd_in_buffer_size', ctypes.c_uint32),
    ('cmd_out_buffer_offset', ctypes.c_uint32),
    ('cmd_out_buffer_size', ctypes.c_uint32),
    ('nvf_img_data_buffer_offset', ctypes.c_uint32),
    ('nvf_img_data_buffer_size', ctypes.c_uint32),
    ('printfBufferHdr', ctypes.c_uint32),
    ('ucode_build_time_stamp', ctypes.c_uint32),
    ('ucode_signature', ctypes.c_uint32),
    ('init_cmd', ctypes.c_uint32),
    ('ucode_feature', ctypes.c_uint32),
    ('ucode_cmd_mask0', ctypes.c_uint32),
    ('ucode_cmd_mask1', ctypes.c_uint32),
    ('multiTgtTbl', ctypes.c_uint32),
]

FALCON_APPLICATION_INTERFACE_DMEM_MAPPER_V3 = struct_c__SA_FALCON_APPLICATION_INTERFACE_DMEM_MAPPER_V3
class struct_BIT_HEADER_V1_00(Structure):
    pass

struct_BIT_HEADER_V1_00._pack_ = 1 # source:True
struct_BIT_HEADER_V1_00._fields_ = [
    ('Id', ctypes.c_uint16),
    ('Signature', ctypes.c_uint32),
    ('BCD_Version', ctypes.c_uint16),
    ('HeaderSize', ctypes.c_ubyte),
    ('TokenSize', ctypes.c_ubyte),
    ('TokenEntries', ctypes.c_ubyte),
    ('HeaderChksum', ctypes.c_ubyte),
]

BIT_HEADER_V1_00 = struct_BIT_HEADER_V1_00
class struct_BIT_TOKEN_V1_00(Structure):
    pass

struct_BIT_TOKEN_V1_00._pack_ = 1 # source:True
struct_BIT_TOKEN_V1_00._fields_ = [
    ('TokenId', ctypes.c_ubyte),
    ('DataVersion', ctypes.c_ubyte),
    ('DataSize', ctypes.c_uint16),
    ('DataPtr', ctypes.c_uint32),
]

BIT_TOKEN_V1_00 = struct_BIT_TOKEN_V1_00
class struct_c__SA_BIT_DATA_BIOSDATA_BINVER(Structure):
    pass

struct_c__SA_BIT_DATA_BIOSDATA_BINVER._pack_ = 1 # source:True
struct_c__SA_BIT_DATA_BIOSDATA_BINVER._fields_ = [
    ('Version', ctypes.c_uint32),
    ('OemVersion', ctypes.c_ubyte),
]

BIT_DATA_BIOSDATA_BINVER = struct_c__SA_BIT_DATA_BIOSDATA_BINVER
class struct_c__SA_BIT_DATA_FALCON_DATA_V2(Structure):
    pass

struct_c__SA_BIT_DATA_FALCON_DATA_V2._pack_ = 1 # source:True
struct_c__SA_BIT_DATA_FALCON_DATA_V2._fields_ = [
    ('FalconUcodeTablePtr', ctypes.c_uint32),
]

BIT_DATA_FALCON_DATA_V2 = struct_c__SA_BIT_DATA_FALCON_DATA_V2
class struct_c__SA_FALCON_UCODE_TABLE_HDR_V1(Structure):
    pass

struct_c__SA_FALCON_UCODE_TABLE_HDR_V1._pack_ = 1 # source:True
struct_c__SA_FALCON_UCODE_TABLE_HDR_V1._fields_ = [
    ('Version', ctypes.c_ubyte),
    ('HeaderSize', ctypes.c_ubyte),
    ('EntrySize', ctypes.c_ubyte),
    ('EntryCount', ctypes.c_ubyte),
    ('DescVersion', ctypes.c_ubyte),
    ('DescSize', ctypes.c_ubyte),
]

FALCON_UCODE_TABLE_HDR_V1 = struct_c__SA_FALCON_UCODE_TABLE_HDR_V1
class struct_c__SA_FALCON_UCODE_TABLE_ENTRY_V1(Structure):
    pass

struct_c__SA_FALCON_UCODE_TABLE_ENTRY_V1._pack_ = 1 # source:True
struct_c__SA_FALCON_UCODE_TABLE_ENTRY_V1._fields_ = [
    ('ApplicationID', ctypes.c_ubyte),
    ('TargetID', ctypes.c_ubyte),
    ('DescPtr', ctypes.c_uint32),
]

FALCON_UCODE_TABLE_ENTRY_V1 = struct_c__SA_FALCON_UCODE_TABLE_ENTRY_V1
class struct_c__SA_FALCON_UCODE_DESC_HEADER(Structure):
    pass

struct_c__SA_FALCON_UCODE_DESC_HEADER._pack_ = 1 # source:True
struct_c__SA_FALCON_UCODE_DESC_HEADER._fields_ = [
    ('vDesc', ctypes.c_uint32),
]

FALCON_UCODE_DESC_HEADER = struct_c__SA_FALCON_UCODE_DESC_HEADER
class struct_c__SA_FALCON_UCODE_DESC_V3(Structure):
    pass

struct_c__SA_FALCON_UCODE_DESC_V3._pack_ = 1 # source:False
struct_c__SA_FALCON_UCODE_DESC_V3._fields_ = [
    ('Hdr', FALCON_UCODE_DESC_HEADER),
    ('StoredSize', ctypes.c_uint32),
    ('PKCDataOffset', ctypes.c_uint32),
    ('InterfaceOffset', ctypes.c_uint32),
    ('IMEMPhysBase', ctypes.c_uint32),
    ('IMEMLoadSize', ctypes.c_uint32),
    ('IMEMVirtBase', ctypes.c_uint32),
    ('DMEMPhysBase', ctypes.c_uint32),
    ('DMEMLoadSize', ctypes.c_uint32),
    ('EngineIdMask', ctypes.c_uint16),
    ('UcodeId', ctypes.c_ubyte),
    ('SignatureCount', ctypes.c_ubyte),
    ('SignatureVersions', ctypes.c_uint16),
    ('Reserved', ctypes.c_uint16),
]

FALCON_UCODE_DESC_V3 = struct_c__SA_FALCON_UCODE_DESC_V3
class struct_c__SA_FWSECLIC_READ_VBIOS_DESC(Structure):
    pass

struct_c__SA_FWSECLIC_READ_VBIOS_DESC._pack_ = 1 # source:True
struct_c__SA_FWSECLIC_READ_VBIOS_DESC._fields_ = [
    ('version', ctypes.c_uint32),
    ('size', ctypes.c_uint32),
    ('gfwImageOffset', ctypes.c_uint64),
    ('gfwImageSize', ctypes.c_uint32),
    ('flags', ctypes.c_uint32),
]

FWSECLIC_READ_VBIOS_DESC = struct_c__SA_FWSECLIC_READ_VBIOS_DESC
class struct_c__SA_FWSECLIC_FRTS_REGION_DESC(Structure):
    pass

struct_c__SA_FWSECLIC_FRTS_REGION_DESC._pack_ = 1 # source:True
struct_c__SA_FWSECLIC_FRTS_REGION_DESC._fields_ = [
    ('version', ctypes.c_uint32),
    ('size', ctypes.c_uint32),
    ('frtsRegionOffset4K', ctypes.c_uint32),
    ('frtsRegionSize', ctypes.c_uint32),
    ('frtsRegionMediaType', ctypes.c_uint32),
]

FWSECLIC_FRTS_REGION_DESC = struct_c__SA_FWSECLIC_FRTS_REGION_DESC
class struct_c__SA_FWSECLIC_FRTS_CMD(Structure):
    _pack_ = 1 # source:True
    _fields_ = [
    ('readVbiosDesc', FWSECLIC_READ_VBIOS_DESC),
    ('frtsRegionDesc', FWSECLIC_FRTS_REGION_DESC),
     ]

FWSECLIC_FRTS_CMD = struct_c__SA_FWSECLIC_FRTS_CMD
__all__ = \
    ['ACPI_DATA', 'ACPI_DSM_CACHE', 'ACPI_DSM_FUNCTION_COUNT',
    'ACPI_DSM_FUNCTION_CURRENT', 'ACPI_DSM_FUNCTION_GPS',
    'ACPI_DSM_FUNCTION_GPS_2X', 'ACPI_DSM_FUNCTION_INVALID',
    'ACPI_DSM_FUNCTION_JT', 'ACPI_DSM_FUNCTION_MXM',
    'ACPI_DSM_FUNCTION_NBCI', 'ACPI_DSM_FUNCTION_NBSI',
    'ACPI_DSM_FUNCTION_NVHG', 'ACPI_DSM_FUNCTION_NVOP',
    'ACPI_DSM_FUNCTION_NVPCF', 'ACPI_DSM_FUNCTION_NVPCF_2X',
    'ACPI_DSM_FUNCTION_PCFG', 'ACPI_DSM_FUNCTION_PEX',
    'ACPI_METHOD_DATA', 'BCRT30_RSA3K_SIG_SIZE',
    'BIT_DATA_BIOSDATA_BINVER', 'BIT_DATA_BIOSDATA_BINVER_FMT',
    'BIT_DATA_BIOSDATA_BINVER_SIZE_5', 'BIT_DATA_BIOSDATA_VERSION_1',
    'BIT_DATA_BIOSDATA_VERSION_2', 'BIT_DATA_FALCON_DATA_V2',
    'BIT_DATA_FALCON_DATA_V2_4_FMT', 'BIT_DATA_FALCON_DATA_V2_SIZE_4',
    'BIT_HEADER_ID', 'BIT_HEADER_SIGNATURE', 'BIT_HEADER_SIZE_OFFSET',
    'BIT_HEADER_V1_00', 'BIT_HEADER_V1_00_FMT', 'BIT_TOKEN_BIOSDATA',
    'BIT_TOKEN_FALCON_DATA', 'BIT_TOKEN_V1_00',
    'BIT_TOKEN_V1_00_FMT_SIZE_6', 'BIT_TOKEN_V1_00_FMT_SIZE_8',
    'BIT_TOKEN_V1_00_SIZE_6', 'BIT_TOKEN_V1_00_SIZE_8', 'BUSINFO',
    'CAPS_METHOD_DATA', 'DEFINING_E_IN_RPC_GLOBAL_ENUMS_H',
    'DEFINING_X_IN_RPC_GLOBAL_ENUMS_H', 'DISPMUXSTATE',
    'DISPMUXSTATE__enumvalues', 'DOD_METHOD_DATA',
    'EcidManufacturingInfo',
    'FALCON_APPLICATION_INTERFACE_DMEM_MAPPER_V3',
    'FALCON_APPLICATION_INTERFACE_DMEM_MAPPER_V3_CMD_FRTS',
    'FALCON_APPLICATION_INTERFACE_DMEM_MAPPER_V3_CMD_SB',
    'FALCON_APPLICATION_INTERFACE_ENTRY_ID_DMEMMAPPER',
    'FALCON_UCODE_DESC_HEADER', 'FALCON_UCODE_DESC_HEADER_FORMAT',
    'FALCON_UCODE_DESC_V3', 'FALCON_UCODE_DESC_V3_44_FMT',
    'FALCON_UCODE_DESC_V3_SIZE_44',
    'FALCON_UCODE_ENTRY_APPID_FIRMWARE_SEC_LIC',
    'FALCON_UCODE_ENTRY_APPID_FWSEC_DBG',
    'FALCON_UCODE_ENTRY_APPID_FWSEC_PROD',
    'FALCON_UCODE_TABLE_ENTRY_V1',
    'FALCON_UCODE_TABLE_ENTRY_V1_6_FMT',
    'FALCON_UCODE_TABLE_ENTRY_V1_SIZE_6',
    'FALCON_UCODE_TABLE_ENTRY_V1_VERSION',
    'FALCON_UCODE_TABLE_HDR_V1', 'FALCON_UCODE_TABLE_HDR_V1_6_FMT',
    'FALCON_UCODE_TABLE_HDR_V1_SIZE_6',
    'FALCON_UCODE_TABLE_HDR_V1_VERSION', 'FECS_ERROR_EVENT_TYPE',
    'FECS_ERROR_EVENT_TYPE_BUFFER_FULL',
    'FECS_ERROR_EVENT_TYPE_BUFFER_RESET_REQUIRED',
    'FECS_ERROR_EVENT_TYPE_MAX', 'FECS_ERROR_EVENT_TYPE_NONE',
    'FECS_ERROR_EVENT_TYPE__enumvalues', 'FWSECLIC_FRTS_CMD',
    'FWSECLIC_FRTS_REGION_DESC', 'FWSECLIC_FRTS_REGION_MEDIA_FB',
    'FWSECLIC_FRTS_REGION_SIZE_1MB_IN_4K', 'FWSECLIC_READ_VBIOS_DESC',
    'FWSECLIC_READ_VBIOS_STRUCT_FLAGS', 'FW_WPR_LAYOUT_OFFSET',
    'GPU_RECOVERY_EVENT_TYPE',
    'GPU_RECOVERY_EVENT_TYPE_GPU_DRAIN_P2P',
    'GPU_RECOVERY_EVENT_TYPE_REFRESH',
    'GPU_RECOVERY_EVENT_TYPE_SYS_REBOOT',
    'GPU_RECOVERY_EVENT_TYPE__enumvalues',
    'GR_MAX_RPC_CTX_BUFFER_COUNT', 'GSPIFPUB_H',
    'GSP_ACR_BOOT_GSP_RM_PARAMS', 'GSP_ARGUMENTS_CACHED',
    'GSP_DMA_TARGET', 'GSP_DMA_TARGET_COHERENT_SYSTEM',
    'GSP_DMA_TARGET_COUNT', 'GSP_DMA_TARGET_LOCAL_FB',
    'GSP_DMA_TARGET_NONCOHERENT_SYSTEM', 'GSP_DMA_TARGET__enumvalues',
    'GSP_FMC_BOOT_PARAMS', 'GSP_FMC_INIT_PARAMS',
    'GSP_FW_HEAP_FREE_LIST_MAGIC', 'GSP_FW_SR_META_H_',
    'GSP_FW_SR_META_INTERNAL_SIZE', 'GSP_FW_SR_META_MAGIC',
    'GSP_FW_SR_META_REVISION', 'GSP_FW_WPR_HEAP_FREE_REGION_COUNT',
    'GSP_FW_WPR_META_H_', 'GSP_FW_WPR_META_MAGIC',
    'GSP_FW_WPR_META_REVISION', 'GSP_FW_WPR_META_VERIFIED',
    'GSP_INIT_ARGS_H', 'GSP_MSG_QUEUE_ELEMENT', 'GSP_PCIE_CONFIG_REG',
    'GSP_RM_PARAMS', 'GSP_SPDM_PARAMS', 'GSP_SR_INIT_ARGUMENTS',
    'GSP_STATIC_CONFIG_H', 'GSP_VF_INFO', 'GspFwHeapFreeList',
    'GspFwHeapFreeRegion', 'GspFwSRMeta', 'GspFwWprMeta',
    'GspStaticConfigInfo', 'GspSystemInfo', 'JT_METHOD_DATA',
    'KERN_FSP_COT_PAYLOAD_H', 'LIBOS_INIT_H_',
    'LIBOS_MEMORY_REGION_CONTIGUOUS',
    'LIBOS_MEMORY_REGION_INIT_ARGUMENTS_MAX',
    'LIBOS_MEMORY_REGION_LOC_FB', 'LIBOS_MEMORY_REGION_LOC_NONE',
    'LIBOS_MEMORY_REGION_LOC_SYSMEM', 'LIBOS_MEMORY_REGION_NONE',
    'LIBOS_MEMORY_REGION_RADIX3',
    'LIBOS_MEMORY_REGION_RADIX_PAGE_LOG2',
    'LIBOS_MEMORY_REGION_RADIX_PAGE_SIZE', 'LibosAddress',
    'LibosMemoryRegionInitArgument', 'LibosMemoryRegionKind',
    'LibosMemoryRegionKind__enumvalues', 'LibosMemoryRegionLoc',
    'LibosMemoryRegionLoc__enumvalues',
    'MAX_DSM_SUPPORTED_FUNCS_RTN_LEN', 'MAX_GPC_COUNT',
    'MAX_GROUP_COUNT', 'MCTP_HEADER', 'MESSAGE_QUEUE_INIT_ARGUMENTS',
    'MSGQ_PRIV_H', 'MSGQ_VERSION', 'MUX_METHOD_DATA',
    'MUX_METHOD_DATA_ELEMENT', 'NV0080_CTRL_GR_TPC_PARTITION_MODE',
    'NV0080_CTRL_GR_TPC_PARTITION_MODE_DYNAMIC',
    'NV0080_CTRL_GR_TPC_PARTITION_MODE_NONE',
    'NV0080_CTRL_GR_TPC_PARTITION_MODE_STATIC',
    'NV2080_CTRL_CMD_PERF_VID_ENG',
    'NV2080_CTRL_CMD_PERF_VID_ENG_NVDEC',
    'NV2080_CTRL_CMD_PERF_VID_ENG_NVENC',
    'NV2080_CTRL_CMD_PERF_VID_ENG_NVJPG',
    'NV2080_CTRL_CMD_PERF_VID_ENG_NVOFA',
    'NV2080_CTRL_PERF_RATED_TDP_ACTION',
    'NV2080_CTRL_PERF_RATED_TDP_ACTION_DEFAULT',
    'NV2080_CTRL_PERF_RATED_TDP_ACTION_FORCE_EXCEED',
    'NV2080_CTRL_PERF_RATED_TDP_ACTION_FORCE_FLOOR',
    'NV2080_CTRL_PERF_RATED_TDP_ACTION_FORCE_LIMIT',
    'NV2080_CTRL_PERF_RATED_TDP_ACTION_FORCE_LOCK',
    'NV2080_CTRL_PERF_RATED_TDP_CLIENT',
    'NV2080_CTRL_PERF_RATED_TDP_CLIENT_GLOBAL',
    'NV2080_CTRL_PERF_RATED_TDP_CLIENT_NUM_CLIENTS',
    'NV2080_CTRL_PERF_RATED_TDP_CLIENT_OS',
    'NV2080_CTRL_PERF_RATED_TDP_CLIENT_PROFILE',
    'NV2080_CTRL_PERF_RATED_TDP_CLIENT_RM',
    'NV2080_CTRL_PERF_RATED_TDP_CLIENT_WAR_BUG_1785342',
    'NV9096_CTRL_ZBC_CLEAR_TABLE_TYPE',
    'NV9096_CTRL_ZBC_CLEAR_TABLE_TYPE_COLOR',
    'NV9096_CTRL_ZBC_CLEAR_TABLE_TYPE_COUNT',
    'NV9096_CTRL_ZBC_CLEAR_TABLE_TYPE_DEPTH',
    'NV9096_CTRL_ZBC_CLEAR_TABLE_TYPE_INVALID',
    'NV9096_CTRL_ZBC_CLEAR_TABLE_TYPE_STENCIL', 'NVB0CC_REGOPS_MODE',
    'NVB0CC_REGOPS_MODE_ALL_OR_NONE',
    'NVB0CC_REGOPS_MODE_CONTINUE_ON_ERROR', 'NVDM_PAYLOAD_COT',
    'NVGPU_ENGINE_CAPS_MASK_ARRAY_MAX', 'NVGPU_ENGINE_CAPS_MASK_BITS',
    'NV_ACPI_GENERIC_FUNC_COUNT',
    'NV_BIT_FALCON_UCODE_DESC_HEADER_VDESC_FLAGS_VERSION_AVAILABLE',
    'NV_BIT_FALCON_UCODE_DESC_HEADER_VDESC_FLAGS_VERSION_UNAVAILABLE',
    'NV_BIT_FALCON_UCODE_DESC_HEADER_VDESC_VERSION_V1',
    'NV_BIT_FALCON_UCODE_DESC_HEADER_VDESC_VERSION_V2',
    'NV_BIT_FALCON_UCODE_DESC_HEADER_VDESC_VERSION_V3',
    'NV_BIT_FALCON_UCODE_DESC_HEADER_VDESC_VERSION_V4',
    'NV_RPC_UPDATE_PDE_BAR_1', 'NV_RPC_UPDATE_PDE_BAR_2',
    'NV_RPC_UPDATE_PDE_BAR_INVALID', 'NV_RPC_UPDATE_PDE_BAR_TYPE',
    'NV_RPC_UPDATE_PDE_BAR_TYPE__enumvalues',
    'NV_VGPU_LOG_LEVEL_DEBUG', 'NV_VGPU_LOG_LEVEL_ERROR',
    'NV_VGPU_LOG_LEVEL_FATAL', 'NV_VGPU_LOG_LEVEL_NOTICE',
    'NV_VGPU_LOG_LEVEL_STATUS', 'NV_VGPU_MSG_EVENT_DISPLAY_MODESET',
    'NV_VGPU_MSG_EVENT_EXTDEV_INTR_SERVICE',
    'NV_VGPU_MSG_EVENT_FECS_ERROR', 'NV_VGPU_MSG_EVENT_FIRST_EVENT',
    'NV_VGPU_MSG_EVENT_GPUACCT_PERFMON_UTIL_SAMPLES',
    'NV_VGPU_MSG_EVENT_GSP_INIT_DONE',
    'NV_VGPU_MSG_EVENT_GSP_LOCKDOWN_NOTICE',
    'NV_VGPU_MSG_EVENT_GSP_POST_NOCAT_RECORD',
    'NV_VGPU_MSG_EVENT_GSP_RUN_CPU_SEQUENCER',
    'NV_VGPU_MSG_EVENT_MIG_CI_CONFIG_UPDATE',
    'NV_VGPU_MSG_EVENT_MMU_FAULT_QUEUED',
    'NV_VGPU_MSG_EVENT_NUM_EVENTS',
    'NV_VGPU_MSG_EVENT_NVLINK_FATAL_ERROR_RECOVERY',
    'NV_VGPU_MSG_EVENT_NVLINK_FAULT_UP',
    'NV_VGPU_MSG_EVENT_NVLINK_INBAND_RECEIVED_DATA_1024',
    'NV_VGPU_MSG_EVENT_NVLINK_INBAND_RECEIVED_DATA_2048',
    'NV_VGPU_MSG_EVENT_NVLINK_INBAND_RECEIVED_DATA_256',
    'NV_VGPU_MSG_EVENT_NVLINK_INBAND_RECEIVED_DATA_4096',
    'NV_VGPU_MSG_EVENT_NVLINK_INBAND_RECEIVED_DATA_512',
    'NV_VGPU_MSG_EVENT_NVLINK_IS_GPU_DEGRADED',
    'NV_VGPU_MSG_EVENT_OS_ERROR_LOG',
    'NV_VGPU_MSG_EVENT_PERF_BRIDGELESS_INFO_UPDATE',
    'NV_VGPU_MSG_EVENT_PERF_GPU_BOOST_SYNC_LIMITS_CALLBACK',
    'NV_VGPU_MSG_EVENT_PFM_REQ_HNDLR_STATE_SYNC_CALLBACK',
    'NV_VGPU_MSG_EVENT_POST_EVENT', 'NV_VGPU_MSG_EVENT_RC_TRIGGERED',
    'NV_VGPU_MSG_EVENT_RECOVERY_ACTION',
    'NV_VGPU_MSG_EVENT_RG_LINE_INTR',
    'NV_VGPU_MSG_EVENT_SEMAPHORE_SCHEDULE_CALLBACK',
    'NV_VGPU_MSG_EVENT_SIM_READ', 'NV_VGPU_MSG_EVENT_SIM_WRITE',
    'NV_VGPU_MSG_EVENT_TIMED_SEMAPHORE_RELEASE',
    'NV_VGPU_MSG_EVENT_UCODE_LIBOS_PRINT',
    'NV_VGPU_MSG_EVENT_UPDATE_GSP_TRACE',
    'NV_VGPU_MSG_EVENT_VGPU_CONFIG',
    'NV_VGPU_MSG_EVENT_VGPU_GSP_PLUGIN_TRIGGERED',
    'NV_VGPU_MSG_FUNCTION_ALLOC_CHANNEL_DMA',
    'NV_VGPU_MSG_FUNCTION_ALLOC_CTX_DMA',
    'NV_VGPU_MSG_FUNCTION_ALLOC_DEVICE',
    'NV_VGPU_MSG_FUNCTION_ALLOC_DISP_CHANNEL',
    'NV_VGPU_MSG_FUNCTION_ALLOC_DISP_OBJECT',
    'NV_VGPU_MSG_FUNCTION_ALLOC_DYNAMIC_MEMORY',
    'NV_VGPU_MSG_FUNCTION_ALLOC_EVENT',
    'NV_VGPU_MSG_FUNCTION_ALLOC_MEMORY',
    'NV_VGPU_MSG_FUNCTION_ALLOC_OBJECT',
    'NV_VGPU_MSG_FUNCTION_ALLOC_ROOT',
    'NV_VGPU_MSG_FUNCTION_ALLOC_SHARE_DEVICE',
    'NV_VGPU_MSG_FUNCTION_ALLOC_SUBDEVICE',
    'NV_VGPU_MSG_FUNCTION_ALLOC_VIDMEM',
    'NV_VGPU_MSG_FUNCTION_ALLOC_VIRTMEM',
    'NV_VGPU_MSG_FUNCTION_BIND_ARBITRARY_CTX_DMA',
    'NV_VGPU_MSG_FUNCTION_BIND_CTX_DMA',
    'NV_VGPU_MSG_FUNCTION_CLEANUP_SURFACE',
    'NV_VGPU_MSG_FUNCTION_CONTINUATION_RECORD',
    'NV_VGPU_MSG_FUNCTION_CREATE_FB_SEGMENT',
    'NV_VGPU_MSG_FUNCTION_CTRL_ALLOC_PMA_STREAM',
    'NV_VGPU_MSG_FUNCTION_CTRL_B0CC_EXEC_REG_OPS',
    'NV_VGPU_MSG_FUNCTION_CTRL_BIND_PM_RESOURCES',
    'NV_VGPU_MSG_FUNCTION_CTRL_BUS_SET_P2P_MAPPING',
    'NV_VGPU_MSG_FUNCTION_CTRL_BUS_UNSET_P2P_MAPPING',
    'NV_VGPU_MSG_FUNCTION_CTRL_CIPHER_AES_ENCRYPT',
    'NV_VGPU_MSG_FUNCTION_CTRL_CIPHER_SESSION_KEY',
    'NV_VGPU_MSG_FUNCTION_CTRL_CIPHER_SESSION_KEY_STATUS',
    'NV_VGPU_MSG_FUNCTION_CTRL_CLK_GET_EXTENDED_INFO',
    'NV_VGPU_MSG_FUNCTION_CTRL_CMD_INTERNAL_CONTROL_GSP_TRACE',
    'NV_VGPU_MSG_FUNCTION_CTRL_CMD_INTERNAL_GPU_START_FABRIC_PROBE',
    'NV_VGPU_MSG_FUNCTION_CTRL_CMD_NVLINK_INBAND_SEND_DATA',
    'NV_VGPU_MSG_FUNCTION_CTRL_DBG_CLEAR_ALL_SM_ERROR_STATES',
    'NV_VGPU_MSG_FUNCTION_CTRL_DBG_CLEAR_SINGLE_SM_ERROR_STATE',
    'NV_VGPU_MSG_FUNCTION_CTRL_DBG_EXEC_REG_OPS',
    'NV_VGPU_MSG_FUNCTION_CTRL_DBG_GET_MODE_MMU_DEBUG',
    'NV_VGPU_MSG_FUNCTION_CTRL_DBG_GET_MODE_MMU_GCC_DEBUG',
    'NV_VGPU_MSG_FUNCTION_CTRL_DBG_READ_ALL_SM_ERROR_STATES',
    'NV_VGPU_MSG_FUNCTION_CTRL_DBG_READ_SINGLE_SM_ERROR_STATE',
    'NV_VGPU_MSG_FUNCTION_CTRL_DBG_RESUME_CONTEXT',
    'NV_VGPU_MSG_FUNCTION_CTRL_DBG_SET_EXCEPTION_MASK',
    'NV_VGPU_MSG_FUNCTION_CTRL_DBG_SET_MODE_ERRBAR_DEBUG',
    'NV_VGPU_MSG_FUNCTION_CTRL_DBG_SET_MODE_MMU_DEBUG',
    'NV_VGPU_MSG_FUNCTION_CTRL_DBG_SET_MODE_MMU_GCC_DEBUG',
    'NV_VGPU_MSG_FUNCTION_CTRL_DBG_SET_NEXT_STOP_TRIGGER_TYPE',
    'NV_VGPU_MSG_FUNCTION_CTRL_DBG_SET_SINGLE_SM_SINGLE_STEP',
    'NV_VGPU_MSG_FUNCTION_CTRL_DBG_SUSPEND_CONTEXT',
    'NV_VGPU_MSG_FUNCTION_CTRL_DMA_SET_DEFAULT_VASPACE',
    'NV_VGPU_MSG_FUNCTION_CTRL_EVENT_BUFFER_UPDATE_GET',
    'NV_VGPU_MSG_FUNCTION_CTRL_EXEC_PARTITIONS_CREATE',
    'NV_VGPU_MSG_FUNCTION_CTRL_EXEC_PARTITIONS_DELETE',
    'NV_VGPU_MSG_FUNCTION_CTRL_FABRIC_MEMORY_DESCRIBE',
    'NV_VGPU_MSG_FUNCTION_CTRL_FABRIC_MEM_STATS',
    'NV_VGPU_MSG_FUNCTION_CTRL_FB_GET_FS_INFO',
    'NV_VGPU_MSG_FUNCTION_CTRL_FB_GET_INFO_V2',
    'NV_VGPU_MSG_FUNCTION_CTRL_FIFO_CLEAR_FAULTED_BIT',
    'NV_VGPU_MSG_FUNCTION_CTRL_FIFO_DISABLE_CHANNELS',
    'NV_VGPU_MSG_FUNCTION_CTRL_FIFO_SETUP_VF_ZOMBIE_SUBCTX_PDB',
    'NV_VGPU_MSG_FUNCTION_CTRL_FIFO_SET_CHANNEL_PROPERTIES',
    'NV_VGPU_MSG_FUNCTION_CTRL_FLA_SETUP_INSTANCE_MEM_BLOCK',
    'NV_VGPU_MSG_FUNCTION_CTRL_FLCN_GET_CTX_BUFFER_INFO',
    'NV_VGPU_MSG_FUNCTION_CTRL_FLCN_GET_CTX_BUFFER_SIZE',
    'NV_VGPU_MSG_FUNCTION_CTRL_FREE_PMA_STREAM',
    'NV_VGPU_MSG_FUNCTION_CTRL_GET_CE_PCE_MASK',
    'NV_VGPU_MSG_FUNCTION_CTRL_GET_HS_CREDITS',
    'NV_VGPU_MSG_FUNCTION_CTRL_GET_LATEST_ECC_ADDRESSES',
    'NV_VGPU_MSG_FUNCTION_CTRL_GET_MMU_DEBUG_MODE',
    'NV_VGPU_MSG_FUNCTION_CTRL_GET_NVLINK_PEER_ID_MASK',
    'NV_VGPU_MSG_FUNCTION_CTRL_GET_NVLINK_STATUS',
    'NV_VGPU_MSG_FUNCTION_CTRL_GET_P2P_CAPS',
    'NV_VGPU_MSG_FUNCTION_CTRL_GET_P2P_CAPS_MATRIX',
    'NV_VGPU_MSG_FUNCTION_CTRL_GET_P2P_CAPS_V2',
    'NV_VGPU_MSG_FUNCTION_CTRL_GET_TOTAL_HS_CREDITS',
    'NV_VGPU_MSG_FUNCTION_CTRL_GET_ZBC_CLEAR_TABLE',
    'NV_VGPU_MSG_FUNCTION_CTRL_GET_ZBC_CLEAR_TABLE_ENTRY',
    'NV_VGPU_MSG_FUNCTION_CTRL_GPFIFO_GET_WORK_SUBMIT_TOKEN',
    'NV_VGPU_MSG_FUNCTION_CTRL_GPFIFO_SCHEDULE',
    'NV_VGPU_MSG_FUNCTION_CTRL_GPFIFO_SET_WORK_SUBMIT_TOKEN_NOTIF_INDEX',
    'NV_VGPU_MSG_FUNCTION_CTRL_GPU_EVICT_CTX',
    'NV_VGPU_MSG_FUNCTION_CTRL_GPU_GET_INFO_V2',
    'NV_VGPU_MSG_FUNCTION_CTRL_GPU_HANDLE_VF_PRI_FAULT',
    'NV_VGPU_MSG_FUNCTION_CTRL_GPU_INITIALIZE_CTX',
    'NV_VGPU_MSG_FUNCTION_CTRL_GPU_MIGRATABLE_OPS',
    'NV_VGPU_MSG_FUNCTION_CTRL_GPU_PROMOTE_CTX',
    'NV_VGPU_MSG_FUNCTION_CTRL_GPU_QUERY_ECC_STATUS',
    'NV_VGPU_MSG_FUNCTION_CTRL_GRMGR_GET_GR_FS_INFO',
    'NV_VGPU_MSG_FUNCTION_CTRL_GR_CTXSW_PREEMPTION_BIND',
    'NV_VGPU_MSG_FUNCTION_CTRL_GR_CTXSW_ZCULL_BIND',
    'NV_VGPU_MSG_FUNCTION_CTRL_GR_GET_CTX_BUFFER_INFO',
    'NV_VGPU_MSG_FUNCTION_CTRL_GR_GET_TPC_PARTITION_MODE',
    'NV_VGPU_MSG_FUNCTION_CTRL_GR_PC_SAMPLING_MODE',
    'NV_VGPU_MSG_FUNCTION_CTRL_GR_SET_CTXSW_PREEMPTION_MODE',
    'NV_VGPU_MSG_FUNCTION_CTRL_GR_SET_TPC_PARTITION_MODE',
    'NV_VGPU_MSG_FUNCTION_CTRL_INTERNAL_MEMSYS_SET_ZBC_REFERENCED',
    'NV_VGPU_MSG_FUNCTION_CTRL_INTERNAL_PROMOTE_FAULT_METHOD_BUFFERS',
    'NV_VGPU_MSG_FUNCTION_CTRL_INTERNAL_QUIESCE_PMA_CHANNEL',
    'NV_VGPU_MSG_FUNCTION_CTRL_INTERNAL_SRIOV_PROMOTE_PMA_STREAM',
    'NV_VGPU_MSG_FUNCTION_CTRL_KGR_GET_CTX_BUFFER_PTES',
    'NV_VGPU_MSG_FUNCTION_CTRL_MASTER_GET_VIRTUAL_FUNCTION_ERROR_CONT_INTR_MASK',
    'NV_VGPU_MSG_FUNCTION_CTRL_MC_SERVICE_INTERRUPTS',
    'NV_VGPU_MSG_FUNCTION_CTRL_NVENC_SW_SESSION_UPDATE_INFO',
    'NV_VGPU_MSG_FUNCTION_CTRL_NVFBC_SW_SESSION_UPDATE_INFO',
    'NV_VGPU_MSG_FUNCTION_CTRL_NVLINK_GET_INBAND_RECEIVED_DATA',
    'NV_VGPU_MSG_FUNCTION_CTRL_PERF_BOOST',
    'NV_VGPU_MSG_FUNCTION_CTRL_PERF_LIMITS_SET_STATUS_V2',
    'NV_VGPU_MSG_FUNCTION_CTRL_PERF_RATED_TDP_GET_STATUS',
    'NV_VGPU_MSG_FUNCTION_CTRL_PERF_RATED_TDP_SET_CONTROL',
    'NV_VGPU_MSG_FUNCTION_CTRL_PERF_VPSTATES_GET_CONTROL',
    'NV_VGPU_MSG_FUNCTION_CTRL_PMA_STREAM_UPDATE_GET_PUT',
    'NV_VGPU_MSG_FUNCTION_CTRL_PM_AREA_PC_SAMPLER',
    'NV_VGPU_MSG_FUNCTION_CTRL_PREEMPT',
    'NV_VGPU_MSG_FUNCTION_CTRL_RELEASE_CCU_PROF',
    'NV_VGPU_MSG_FUNCTION_CTRL_RELEASE_HES',
    'NV_VGPU_MSG_FUNCTION_CTRL_RESERVE_CCU_PROF',
    'NV_VGPU_MSG_FUNCTION_CTRL_RESERVE_HES',
    'NV_VGPU_MSG_FUNCTION_CTRL_RESERVE_HWPM_LEGACY',
    'NV_VGPU_MSG_FUNCTION_CTRL_RESERVE_PM_AREA_SMPC',
    'NV_VGPU_MSG_FUNCTION_CTRL_RESET_CHANNEL',
    'NV_VGPU_MSG_FUNCTION_CTRL_RESET_ISOLATED_CHANNEL',
    'NV_VGPU_MSG_FUNCTION_CTRL_SET_CHANNEL_INTERLEAVE_LEVEL',
    'NV_VGPU_MSG_FUNCTION_CTRL_SET_HS_CREDITS',
    'NV_VGPU_MSG_FUNCTION_CTRL_SET_TIMESLICE',
    'NV_VGPU_MSG_FUNCTION_CTRL_SET_TSG_INTERLEAVE_LEVEL',
    'NV_VGPU_MSG_FUNCTION_CTRL_SET_VGPU_FB_USAGE',
    'NV_VGPU_MSG_FUNCTION_CTRL_SET_ZBC_COLOR_CLEAR',
    'NV_VGPU_MSG_FUNCTION_CTRL_SET_ZBC_DEPTH_CLEAR',
    'NV_VGPU_MSG_FUNCTION_CTRL_SET_ZBC_STENCIL_CLEAR',
    'NV_VGPU_MSG_FUNCTION_CTRL_STOP_CHANNEL',
    'NV_VGPU_MSG_FUNCTION_CTRL_SUBDEVICE_GET_LIBOS_HEAP_STATS',
    'NV_VGPU_MSG_FUNCTION_CTRL_SUBDEVICE_GET_P2P_CAPS',
    'NV_VGPU_MSG_FUNCTION_CTRL_SUBDEVICE_GET_VGPU_HEAP_STATS',
    'NV_VGPU_MSG_FUNCTION_CTRL_TIMER_SET_GR_TICK_FREQ',
    'NV_VGPU_MSG_FUNCTION_CTRL_VASPACE_COPY_SERVER_RESERVED_PDES',
    'NV_VGPU_MSG_FUNCTION_DCE_RM_INIT',
    'NV_VGPU_MSG_FUNCTION_DEFERRED_API_CONTROL',
    'NV_VGPU_MSG_FUNCTION_DESTROY_FB_SEGMENT',
    'NV_VGPU_MSG_FUNCTION_DISABLE_CHANNELS',
    'NV_VGPU_MSG_FUNCTION_DMA_CONTROL',
    'NV_VGPU_MSG_FUNCTION_DMA_FILL_PTE_MEM',
    'NV_VGPU_MSG_FUNCTION_DUMP_PROTOBUF_COMPONENT',
    'NV_VGPU_MSG_FUNCTION_DUP_OBJECT',
    'NV_VGPU_MSG_FUNCTION_ECC_NOTIFIER_WRITE_ACK',
    'NV_VGPU_MSG_FUNCTION_FREE',
    'NV_VGPU_MSG_FUNCTION_FREE_VIDMEM_VIRT',
    'NV_VGPU_MSG_FUNCTION_GET_BRAND_CAPS',
    'NV_VGPU_MSG_FUNCTION_GET_CONSOLIDATED_GR_STATIC_INFO',
    'NV_VGPU_MSG_FUNCTION_GET_CONSOLIDATED_STATIC_INFO',
    'NV_VGPU_MSG_FUNCTION_GET_EDID',
    'NV_VGPU_MSG_FUNCTION_GET_ENCODER_CAPACITY',
    'NV_VGPU_MSG_FUNCTION_GET_ENGINE_UTILIZATION',
    'NV_VGPU_MSG_FUNCTION_GET_GSP_STATIC_INFO',
    'NV_VGPU_MSG_FUNCTION_GET_PLCABLE_ADDRESS_KIND',
    'NV_VGPU_MSG_FUNCTION_GET_STATIC_DATA',
    'NV_VGPU_MSG_FUNCTION_GET_STATIC_INFO',
    'NV_VGPU_MSG_FUNCTION_GET_STATIC_INFO2',
    'NV_VGPU_MSG_FUNCTION_GET_STATIC_PSTATE_INFO',
    'NV_VGPU_MSG_FUNCTION_GMMU_REGISTER_CLIENT_SHADOW_FAULT_BUFFER',
    'NV_VGPU_MSG_FUNCTION_GMMU_REGISTER_FAULT_BUFFER',
    'NV_VGPU_MSG_FUNCTION_GMMU_UNREGISTER_CLIENT_SHADOW_FAULT_BUFFER',
    'NV_VGPU_MSG_FUNCTION_GMMU_UNREGISTER_FAULT_BUFFER',
    'NV_VGPU_MSG_FUNCTION_GPU_EXEC_REG_OPS',
    'NV_VGPU_MSG_FUNCTION_GSP_INIT_POST_OBJGPU',
    'NV_VGPU_MSG_FUNCTION_GSP_RM_ALLOC',
    'NV_VGPU_MSG_FUNCTION_GSP_RM_CONTROL',
    'NV_VGPU_MSG_FUNCTION_GSP_SET_SYSTEM_INFO',
    'NV_VGPU_MSG_FUNCTION_IDLE_CHANNELS',
    'NV_VGPU_MSG_FUNCTION_INVALIDATE_TLB', 'NV_VGPU_MSG_FUNCTION_LOG',
    'NV_VGPU_MSG_FUNCTION_MANAGE_HW_RESOURCE',
    'NV_VGPU_MSG_FUNCTION_MAP_MEMORY',
    'NV_VGPU_MSG_FUNCTION_MAP_MEMORY_DMA',
    'NV_VGPU_MSG_FUNCTION_MAP_SEMA_MEMORY',
    'NV_VGPU_MSG_FUNCTION_NOP', 'NV_VGPU_MSG_FUNCTION_NUM_FUNCTIONS',
    'NV_VGPU_MSG_FUNCTION_PERF_GET_LEVEL_INFO',
    'NV_VGPU_MSG_FUNCTION_PERF_GET_PERFMON_SAMPLE',
    'NV_VGPU_MSG_FUNCTION_PERF_GET_PSTATE_INFO',
    'NV_VGPU_MSG_FUNCTION_PERF_GET_VIRTUAL_PSTATE_INFO',
    'NV_VGPU_MSG_FUNCTION_PMA_SCRUBBER_SHARED_BUFFER_GUEST_PAGES_OPERATION',
    'NV_VGPU_MSG_FUNCTION_REGISTER_VIRTUAL_EVENT_BUFFER',
    'NV_VGPU_MSG_FUNCTION_REMAPPER_CONTROL',
    'NV_VGPU_MSG_FUNCTION_REMOVE_DEFERRED_API',
    'NV_VGPU_MSG_FUNCTION_RESERVED_0',
    'NV_VGPU_MSG_FUNCTION_RESERVED_208',
    'NV_VGPU_MSG_FUNCTION_RESERVED_57',
    'NV_VGPU_MSG_FUNCTION_RESET_CURRENT_GR_CONTEXT',
    'NV_VGPU_MSG_FUNCTION_RESTORE_HIBERNATION_DATA',
    'NV_VGPU_MSG_FUNCTION_RMFS_CLEANUP',
    'NV_VGPU_MSG_FUNCTION_RMFS_CLOSE_QUEUE',
    'NV_VGPU_MSG_FUNCTION_RMFS_INIT',
    'NV_VGPU_MSG_FUNCTION_RMFS_TEST',
    'NV_VGPU_MSG_FUNCTION_RM_API_CONTROL',
    'NV_VGPU_MSG_FUNCTION_SAVE_HIBERNATION_DATA',
    'NV_VGPU_MSG_FUNCTION_SEND_EVENT',
    'NV_VGPU_MSG_FUNCTION_SET_GUEST_SYSTEM_INFO',
    'NV_VGPU_MSG_FUNCTION_SET_GUEST_SYSTEM_INFO_EXT',
    'NV_VGPU_MSG_FUNCTION_SET_PAGE_DIRECTORY',
    'NV_VGPU_MSG_FUNCTION_SET_REGISTRY',
    'NV_VGPU_MSG_FUNCTION_SET_SEMA_MEM_VALIDATION_STATE',
    'NV_VGPU_MSG_FUNCTION_SET_SURFACE_PROPERTIES',
    'NV_VGPU_MSG_FUNCTION_SET_SYSMEM_DIRTY_PAGE_TRACKING_BUFFER',
    'NV_VGPU_MSG_FUNCTION_SIM_ESCAPE_READ',
    'NV_VGPU_MSG_FUNCTION_SIM_ESCAPE_WRITE',
    'NV_VGPU_MSG_FUNCTION_SIM_MANAGE_DISPLAY_CONTEXT_DMA',
    'NV_VGPU_MSG_FUNCTION_SUBDEV_EVENT_SET_NOTIFICATION',
    'NV_VGPU_MSG_FUNCTION_SWITCH_TO_VGA',
    'NV_VGPU_MSG_FUNCTION_TDR_SET_TIMEOUT_STATE',
    'NV_VGPU_MSG_FUNCTION_TRANSLATE_GUEST_GPU_PTES',
    'NV_VGPU_MSG_FUNCTION_UNLOADING_GUEST_DRIVER',
    'NV_VGPU_MSG_FUNCTION_UNMAP_MEMORY',
    'NV_VGPU_MSG_FUNCTION_UNMAP_MEMORY_DMA',
    'NV_VGPU_MSG_FUNCTION_UNMAP_SEMA_MEMORY',
    'NV_VGPU_MSG_FUNCTION_UNSET_PAGE_DIRECTORY',
    'NV_VGPU_MSG_FUNCTION_UPDATE_BAR_PDE',
    'NV_VGPU_MSG_FUNCTION_UPDATE_GPM_GUEST_BUFFER_INFO',
    'NV_VGPU_MSG_FUNCTION_UPDATE_GPU_PDES',
    'NV_VGPU_MSG_FUNCTION_UPDATE_PDE_2',
    'NV_VGPU_MSG_FUNCTION_UVM_METHOD_STREAM_GUEST_PAGES_OPERATION',
    'NV_VGPU_MSG_FUNCTION_UVM_PAGING_CHANNEL_ALLOCATE',
    'NV_VGPU_MSG_FUNCTION_UVM_PAGING_CHANNEL_DESTROY',
    'NV_VGPU_MSG_FUNCTION_UVM_PAGING_CHANNEL_MAP',
    'NV_VGPU_MSG_FUNCTION_UVM_PAGING_CHANNEL_PUSH_STREAM',
    'NV_VGPU_MSG_FUNCTION_UVM_PAGING_CHANNEL_SET_HANDLES',
    'NV_VGPU_MSG_FUNCTION_UVM_PAGING_CHANNEL_UNMAP',
    'NV_VGPU_MSG_FUNCTION_VGPU_PF_REG_READ32',
    'NV_VGPU_MSG_HEADER_VERSION_MAJOR_TOT',
    'NV_VGPU_MSG_HEADER_VERSION_MINOR_TOT',
    'NV_VGPU_MSG_RESULT_RPC_API_CONTROL_NOT_SUPPORTED',
    'NV_VGPU_MSG_RESULT_RPC_CUDA_PROFILING_DISABLED',
    'NV_VGPU_MSG_RESULT_RPC_HANDLE_EXISTS',
    'NV_VGPU_MSG_RESULT_RPC_HANDLE_NOT_FOUND',
    'NV_VGPU_MSG_RESULT_RPC_INVALID_MESSAGE_FORMAT',
    'NV_VGPU_MSG_RESULT_RPC_PENDING',
    'NV_VGPU_MSG_RESULT_RPC_RESERVED_HANDLE',
    'NV_VGPU_MSG_RESULT_RPC_UNKNOWN_FUNCTION',
    'NV_VGPU_MSG_RESULT_RPC_UNKNOWN_RM_ERROR',
    'NV_VGPU_MSG_RESULT_RPC_UNKNOWN_VMIOP_ERROR',
    'NV_VGPU_MSG_RESULT_VMIOP_ECC_MISMATCH',
    'NV_VGPU_MSG_RESULT_VMIOP_INVAL',
    'NV_VGPU_MSG_RESULT_VMIOP_NOT_ALLOWED_IN_CALLBACK',
    'NV_VGPU_MSG_RESULT_VMIOP_NOT_FOUND',
    'NV_VGPU_MSG_RESULT_VMIOP_NOT_SUPPORTED',
    'NV_VGPU_MSG_RESULT_VMIOP_NO_ADDRESS_SPACE',
    'NV_VGPU_MSG_RESULT_VMIOP_RANGE',
    'NV_VGPU_MSG_RESULT_VMIOP_READ_ONLY',
    'NV_VGPU_MSG_RESULT_VMIOP_RESOURCE',
    'NV_VGPU_MSG_RESULT_VMIOP_TIMEOUT', 'NV_VGPU_MSG_SIGNATURE_VALID',
    'NV_VGPU_MSG_UNION_INIT', 'NV_VGPU_PTEDESC_IDR_DOUBLE',
    'NV_VGPU_PTEDESC_IDR_NONE', 'NV_VGPU_PTEDESC_IDR_SINGLE',
    'NV_VGPU_PTEDESC_IDR_TRIPLE', 'NV_VGPU_PTEDESC_INIT',
    'NV_VGPU_PTEDESC__PROD', 'NV_VGPU_PTE_64_INDEX_MASK',
    'NV_VGPU_PTE_64_INDEX_SHIFT', 'NV_VGPU_PTE_64_PAGE_SIZE',
    'NV_VGPU_PTE_64_SIZE', 'NV_VGPU_PTE_INDEX_MASK',
    'NV_VGPU_PTE_INDEX_SHIFT', 'NV_VGPU_PTE_PAGE_SIZE',
    'NV_VGPU_PTE_SIZE', 'PACKED_REGISTRY_ENTRY',
    'PACKED_REGISTRY_TABLE', 'REGISTRY_TABLE_ENTRY_TYPE_BINARY',
    'REGISTRY_TABLE_ENTRY_TYPE_DWORD',
    'REGISTRY_TABLE_ENTRY_TYPE_STRING',
    'REGISTRY_TABLE_ENTRY_TYPE_UNKNOWN', 'RM_ENGINE_TYPE',
    'RM_ENGINE_TYPE_BSP', 'RM_ENGINE_TYPE_CIPHER',
    'RM_ENGINE_TYPE_COPY0', 'RM_ENGINE_TYPE_COPY1',
    'RM_ENGINE_TYPE_COPY10', 'RM_ENGINE_TYPE_COPY11',
    'RM_ENGINE_TYPE_COPY12', 'RM_ENGINE_TYPE_COPY13',
    'RM_ENGINE_TYPE_COPY14', 'RM_ENGINE_TYPE_COPY15',
    'RM_ENGINE_TYPE_COPY16', 'RM_ENGINE_TYPE_COPY17',
    'RM_ENGINE_TYPE_COPY18', 'RM_ENGINE_TYPE_COPY19',
    'RM_ENGINE_TYPE_COPY2', 'RM_ENGINE_TYPE_COPY3',
    'RM_ENGINE_TYPE_COPY4', 'RM_ENGINE_TYPE_COPY5',
    'RM_ENGINE_TYPE_COPY6', 'RM_ENGINE_TYPE_COPY7',
    'RM_ENGINE_TYPE_COPY8', 'RM_ENGINE_TYPE_COPY9',
    'RM_ENGINE_TYPE_COPY_SIZE', 'RM_ENGINE_TYPE_DPU',
    'RM_ENGINE_TYPE_FBFLCN', 'RM_ENGINE_TYPE_GR0',
    'RM_ENGINE_TYPE_GR1', 'RM_ENGINE_TYPE_GR2', 'RM_ENGINE_TYPE_GR3',
    'RM_ENGINE_TYPE_GR4', 'RM_ENGINE_TYPE_GR5', 'RM_ENGINE_TYPE_GR6',
    'RM_ENGINE_TYPE_GR7', 'RM_ENGINE_TYPE_GRAPHICS',
    'RM_ENGINE_TYPE_GR_SIZE', 'RM_ENGINE_TYPE_HOST',
    'RM_ENGINE_TYPE_LAST', 'RM_ENGINE_TYPE_ME', 'RM_ENGINE_TYPE_MP',
    'RM_ENGINE_TYPE_MPEG', 'RM_ENGINE_TYPE_MSENC',
    'RM_ENGINE_TYPE_NULL', 'RM_ENGINE_TYPE_NVDEC0',
    'RM_ENGINE_TYPE_NVDEC1', 'RM_ENGINE_TYPE_NVDEC2',
    'RM_ENGINE_TYPE_NVDEC3', 'RM_ENGINE_TYPE_NVDEC4',
    'RM_ENGINE_TYPE_NVDEC5', 'RM_ENGINE_TYPE_NVDEC6',
    'RM_ENGINE_TYPE_NVDEC7', 'RM_ENGINE_TYPE_NVDEC_SIZE',
    'RM_ENGINE_TYPE_NVENC0', 'RM_ENGINE_TYPE_NVENC1',
    'RM_ENGINE_TYPE_NVENC2', 'RM_ENGINE_TYPE_NVENC3',
    'RM_ENGINE_TYPE_NVENC_SIZE', 'RM_ENGINE_TYPE_NVJPEG0',
    'RM_ENGINE_TYPE_NVJPEG1', 'RM_ENGINE_TYPE_NVJPEG2',
    'RM_ENGINE_TYPE_NVJPEG3', 'RM_ENGINE_TYPE_NVJPEG4',
    'RM_ENGINE_TYPE_NVJPEG5', 'RM_ENGINE_TYPE_NVJPEG6',
    'RM_ENGINE_TYPE_NVJPEG7', 'RM_ENGINE_TYPE_NVJPEG_SIZE',
    'RM_ENGINE_TYPE_NVJPG', 'RM_ENGINE_TYPE_OFA0',
    'RM_ENGINE_TYPE_OFA1', 'RM_ENGINE_TYPE_OFA_SIZE',
    'RM_ENGINE_TYPE_PMU', 'RM_ENGINE_TYPE_PPP',
    'RM_ENGINE_TYPE_RESERVED40', 'RM_ENGINE_TYPE_RESERVED41',
    'RM_ENGINE_TYPE_RESERVED42', 'RM_ENGINE_TYPE_RESERVED43',
    'RM_ENGINE_TYPE_RESERVED44', 'RM_ENGINE_TYPE_RESERVED45',
    'RM_ENGINE_TYPE_RESERVED46', 'RM_ENGINE_TYPE_RESERVED47',
    'RM_ENGINE_TYPE_RESERVED48', 'RM_ENGINE_TYPE_RESERVED49',
    'RM_ENGINE_TYPE_RESERVED4a', 'RM_ENGINE_TYPE_RESERVED4b',
    'RM_ENGINE_TYPE_RESERVED4c', 'RM_ENGINE_TYPE_RESERVED4d',
    'RM_ENGINE_TYPE_RESERVED4e', 'RM_ENGINE_TYPE_RESERVED4f',
    'RM_ENGINE_TYPE_RESERVED50', 'RM_ENGINE_TYPE_RESERVED51',
    'RM_ENGINE_TYPE_RESERVED52', 'RM_ENGINE_TYPE_RESERVED53',
    'RM_ENGINE_TYPE_SEC2', 'RM_ENGINE_TYPE_SW', 'RM_ENGINE_TYPE_TSEC',
    'RM_ENGINE_TYPE_VIC', 'RM_ENGINE_TYPE_VP',
    'RM_ENGINE_TYPE__enumvalues', 'RM_RISCV_UCODE_DESC',
    'RM_RISCV_UCODE_H', 'RPC_GR_BUFFER_TYPE',
    'RPC_GR_BUFFER_TYPE_COMPUTE_PREEMPT',
    'RPC_GR_BUFFER_TYPE_GRAPHICS',
    'RPC_GR_BUFFER_TYPE_GRAPHICS_ATTRIBUTE_CB',
    'RPC_GR_BUFFER_TYPE_GRAPHICS_BUNDLE_CB',
    'RPC_GR_BUFFER_TYPE_GRAPHICS_FECS_EVENT',
    'RPC_GR_BUFFER_TYPE_GRAPHICS_GFXP_CTRL_BLK',
    'RPC_GR_BUFFER_TYPE_GRAPHICS_GFXP_POOL',
    'RPC_GR_BUFFER_TYPE_GRAPHICS_GRAPHICS_PM',
    'RPC_GR_BUFFER_TYPE_GRAPHICS_MAX',
    'RPC_GR_BUFFER_TYPE_GRAPHICS_PAGEPOOL_GLOBAL',
    'RPC_GR_BUFFER_TYPE_GRAPHICS_PATCH',
    'RPC_GR_BUFFER_TYPE_GRAPHICS_PRIV_ACCESS_MAP',
    'RPC_GR_BUFFER_TYPE_GRAPHICS_RTV_CB_GLOBAL',
    'RPC_GR_BUFFER_TYPE_GRAPHICS_ZCULL',
    'RPC_GR_BUFFER_TYPE__enumvalues', 'SDK_STRUCTURES', 'VBIOS_H',
    'VGPU_MAX_REGOPS_PER_RPC', 'VGPU_RESERVED_HANDLE_BASE',
    'VGPU_RESERVED_HANDLE_RANGE',
    'VGPU_RPC_CTRL_DEBUG_READ_ALL_SM_ERROR_STATES_PER_RPC_v21_06',
    'VGPU_RPC_GET_P2P_CAPS_V2_MAX_GPUS_SQUARED_PER_RPC',
    'VIRTUAL_DISPLAY_GET_MAX_RESOLUTION_PARAMS',
    'VIRTUAL_DISPLAY_GET_NUM_HEADS_PARAMS', '_ACPI_DSM_FUNCTION',
    '_RPC_GLOBAL_ENUMS_H_', '__vgpu_rpc_nv_headers_h__',
    'c__EA_DISPMUXSTATE', 'c__EA_FECS_ERROR_EVENT_TYPE',
    'c__EA_GPU_RECOVERY_EVENT_TYPE', 'c__EA_GSP_DMA_TARGET',
    'c__EA_LibosMemoryRegionKind', 'c__EA_LibosMemoryRegionLoc',
    'c__EA_NV_RPC_UPDATE_PDE_BAR_TYPE', 'c__EA_RM_ENGINE_TYPE',
    'c__EA_RPC_GR_BUFFER_TYPE', 'c__Ea_NV_VGPU_MSG_EVENT_FIRST_EVENT',
    'c__Ea_NV_VGPU_MSG_FUNCTION_NOP', 'dispMuxState_DiscreteGPU',
    'dispMuxState_IntegratedGPU', 'dispMuxState_None', 'msgqMetadata',
    'msgqRxHeader', 'msgqTxHeader', 'rpc_alloc_channel_dma_v',
    'rpc_alloc_channel_dma_v1F_04', 'rpc_alloc_event_v',
    'rpc_alloc_event_v03_00', 'rpc_alloc_memory_v',
    'rpc_alloc_memory_v13_01', 'rpc_alloc_object_v',
    'rpc_alloc_object_v25_08', 'rpc_alloc_object_v26_00',
    'rpc_alloc_object_v27_00', 'rpc_alloc_object_v29_06',
    'rpc_alloc_root_v', 'rpc_alloc_root_v07_00',
    'rpc_alloc_share_device_v', 'rpc_alloc_share_device_v03_00',
    'rpc_alloc_subdevice_v', 'rpc_alloc_subdevice_v08_01',
    'rpc_cleanup_surface_v', 'rpc_cleanup_surface_v03_00',
    'rpc_ctrl_alloc_pma_stream_v', 'rpc_ctrl_alloc_pma_stream_v1A_14',
    'rpc_ctrl_b0cc_exec_reg_ops_v',
    'rpc_ctrl_b0cc_exec_reg_ops_v1A_0F',
    'rpc_ctrl_bind_pm_resources_v',
    'rpc_ctrl_bind_pm_resources_v1A_0F',
    'rpc_ctrl_bus_set_p2p_mapping_v',
    'rpc_ctrl_bus_set_p2p_mapping_v21_03',
    'rpc_ctrl_bus_set_p2p_mapping_v29_08',
    'rpc_ctrl_bus_unset_p2p_mapping_v',
    'rpc_ctrl_bus_unset_p2p_mapping_v21_03',
    'rpc_ctrl_cmd_internal_control_gsp_trace_v',
    'rpc_ctrl_cmd_internal_control_gsp_trace_v28_00',
    'rpc_ctrl_cmd_internal_gpu_start_fabric_probe_v',
    'rpc_ctrl_cmd_internal_gpu_start_fabric_probe_v25_09',
    'rpc_ctrl_cmd_nvlink_inband_send_data_v',
    'rpc_ctrl_cmd_nvlink_inband_send_data_v26_05',
    'rpc_ctrl_dbg_clear_all_sm_error_states_v',
    'rpc_ctrl_dbg_clear_all_sm_error_states_v1A_0C',
    'rpc_ctrl_dbg_clear_single_sm_error_state_v',
    'rpc_ctrl_dbg_clear_single_sm_error_state_v1A_10',
    'rpc_ctrl_dbg_exec_reg_ops_v', 'rpc_ctrl_dbg_exec_reg_ops_v1A_10',
    'rpc_ctrl_dbg_get_mode_mmu_debug_v',
    'rpc_ctrl_dbg_get_mode_mmu_debug_v25_04',
    'rpc_ctrl_dbg_get_mode_mmu_gcc_debug_v',
    'rpc_ctrl_dbg_get_mode_mmu_gcc_debug_v29_07',
    'rpc_ctrl_dbg_read_all_sm_error_states_v',
    'rpc_ctrl_dbg_read_all_sm_error_states_v21_06',
    'rpc_ctrl_dbg_read_single_sm_error_state_v',
    'rpc_ctrl_dbg_read_single_sm_error_state_v21_06',
    'rpc_ctrl_dbg_resume_context_v',
    'rpc_ctrl_dbg_resume_context_v1A_10',
    'rpc_ctrl_dbg_set_exception_mask_v',
    'rpc_ctrl_dbg_set_exception_mask_v1A_0C',
    'rpc_ctrl_dbg_set_mode_errbar_debug_v',
    'rpc_ctrl_dbg_set_mode_errbar_debug_v1A_10',
    'rpc_ctrl_dbg_set_mode_mmu_debug_v',
    'rpc_ctrl_dbg_set_mode_mmu_debug_v1A_10',
    'rpc_ctrl_dbg_set_mode_mmu_gcc_debug_v',
    'rpc_ctrl_dbg_set_mode_mmu_gcc_debug_v29_07',
    'rpc_ctrl_dbg_set_next_stop_trigger_type_v',
    'rpc_ctrl_dbg_set_next_stop_trigger_type_v1A_10',
    'rpc_ctrl_dbg_set_single_sm_single_step_v',
    'rpc_ctrl_dbg_set_single_sm_single_step_v1C_02',
    'rpc_ctrl_dbg_suspend_context_v',
    'rpc_ctrl_dbg_suspend_context_v1A_10',
    'rpc_ctrl_dma_set_default_vaspace_v',
    'rpc_ctrl_dma_set_default_vaspace_v1A_0E',
    'rpc_ctrl_exec_partitions_create_v',
    'rpc_ctrl_exec_partitions_create_v24_05',
    'rpc_ctrl_exec_partitions_delete_v',
    'rpc_ctrl_exec_partitions_delete_v1F_0A',
    'rpc_ctrl_fabric_mem_stats_v', 'rpc_ctrl_fabric_mem_stats_v1E_0C',
    'rpc_ctrl_fabric_memory_describe_v',
    'rpc_ctrl_fabric_memory_describe_v1E_0C',
    'rpc_ctrl_fb_get_fs_info_v', 'rpc_ctrl_fb_get_fs_info_v24_00',
    'rpc_ctrl_fb_get_fs_info_v26_04', 'rpc_ctrl_fb_get_info_v2_v',
    'rpc_ctrl_fb_get_info_v2_v25_0A',
    'rpc_ctrl_fb_get_info_v2_v27_00',
    'rpc_ctrl_fifo_disable_channels_v',
    'rpc_ctrl_fifo_disable_channels_v1A_0A',
    'rpc_ctrl_fifo_set_channel_properties_v',
    'rpc_ctrl_fifo_set_channel_properties_v1A_16',
    'rpc_ctrl_fifo_setup_vf_zombie_subctx_pdb_v',
    'rpc_ctrl_fifo_setup_vf_zombie_subctx_pdb_v1A_23',
    'rpc_ctrl_fla_setup_instance_mem_block_v',
    'rpc_ctrl_fla_setup_instance_mem_block_v21_05',
    'rpc_ctrl_free_pma_stream_v', 'rpc_ctrl_free_pma_stream_v1A_1F',
    'rpc_ctrl_get_ce_pce_mask_v', 'rpc_ctrl_get_ce_pce_mask_v1A_0E',
    'rpc_ctrl_get_hs_credits_v', 'rpc_ctrl_get_hs_credits_v21_08',
    'rpc_ctrl_get_mmu_debug_mode_v',
    'rpc_ctrl_get_mmu_debug_mode_v1E_06',
    'rpc_ctrl_get_nvlink_status_v',
    'rpc_ctrl_get_nvlink_status_v23_04',
    'rpc_ctrl_get_nvlink_status_v28_09',
    'rpc_ctrl_get_p2p_caps_matrix_v',
    'rpc_ctrl_get_p2p_caps_matrix_v1A_0E', 'rpc_ctrl_get_p2p_caps_v',
    'rpc_ctrl_get_p2p_caps_v1F_0D', 'rpc_ctrl_get_p2p_caps_v2_v',
    'rpc_ctrl_get_p2p_caps_v2_v1F_0D',
    'rpc_ctrl_get_total_hs_credits_v',
    'rpc_ctrl_get_total_hs_credits_v21_08',
    'rpc_ctrl_get_zbc_clear_table_entry_v',
    'rpc_ctrl_get_zbc_clear_table_entry_v1A_0E',
    'rpc_ctrl_get_zbc_clear_table_v',
    'rpc_ctrl_get_zbc_clear_table_v1A_09',
    'rpc_ctrl_gpfifo_get_work_submit_token_v',
    'rpc_ctrl_gpfifo_get_work_submit_token_v1F_0A',
    'rpc_ctrl_gpfifo_schedule_v', 'rpc_ctrl_gpfifo_schedule_v1A_0A',
    'rpc_ctrl_gpfifo_set_work_submit_token_notif_index_v',
    'rpc_ctrl_gpfifo_set_work_submit_token_notif_index_v1F_0A',
    'rpc_ctrl_gpu_evict_ctx_v', 'rpc_ctrl_gpu_evict_ctx_v1A_1C',
    'rpc_ctrl_gpu_get_info_v2_v', 'rpc_ctrl_gpu_get_info_v2_v25_11',
    'rpc_ctrl_gpu_handle_vf_pri_fault_v',
    'rpc_ctrl_gpu_handle_vf_pri_fault_v1A_09',
    'rpc_ctrl_gpu_initialize_ctx_v',
    'rpc_ctrl_gpu_initialize_ctx_v1A_0E',
    'rpc_ctrl_gpu_migratable_ops_v',
    'rpc_ctrl_gpu_migratable_ops_v21_07',
    'rpc_ctrl_gpu_promote_ctx_v', 'rpc_ctrl_gpu_promote_ctx_v1A_20',
    'rpc_ctrl_gpu_query_ecc_status_v',
    'rpc_ctrl_gpu_query_ecc_status_v24_06',
    'rpc_ctrl_gpu_query_ecc_status_v26_02',
    'rpc_ctrl_gr_ctxsw_preemption_bind_v',
    'rpc_ctrl_gr_ctxsw_preemption_bind_v1A_0E',
    'rpc_ctrl_gr_ctxsw_preemption_bind_v28_07',
    'rpc_ctrl_gr_ctxsw_zcull_bind_v',
    'rpc_ctrl_gr_ctxsw_zcull_bind_v1A_0E',
    'rpc_ctrl_gr_get_tpc_partition_mode_v',
    'rpc_ctrl_gr_get_tpc_partition_mode_v1C_04',
    'rpc_ctrl_gr_pc_sampling_mode_v',
    'rpc_ctrl_gr_pc_sampling_mode_v1A_1F',
    'rpc_ctrl_gr_set_ctxsw_preemption_mode_v',
    'rpc_ctrl_gr_set_ctxsw_preemption_mode_v1A_0E',
    'rpc_ctrl_gr_set_tpc_partition_mode_v',
    'rpc_ctrl_gr_set_tpc_partition_mode_v1C_04',
    'rpc_ctrl_grmgr_get_gr_fs_info_v',
    'rpc_ctrl_grmgr_get_gr_fs_info_v1A_1D',
    'rpc_ctrl_internal_memsys_set_zbc_referenced_v',
    'rpc_ctrl_internal_memsys_set_zbc_referenced_v1F_05',
    'rpc_ctrl_internal_promote_fault_method_buffers_v',
    'rpc_ctrl_internal_promote_fault_method_buffers_v1E_07',
    'rpc_ctrl_internal_quiesce_pma_channel_v',
    'rpc_ctrl_internal_quiesce_pma_channel_v1C_08',
    'rpc_ctrl_internal_sriov_promote_pma_stream_v',
    'rpc_ctrl_internal_sriov_promote_pma_stream_v1C_0C',
    'rpc_ctrl_master_get_virtual_function_error_cont_intr_mask_v',
    'rpc_ctrl_master_get_virtual_function_error_cont_intr_mask_v1F_0D',
    'rpc_ctrl_mc_service_interrupts_v',
    'rpc_ctrl_mc_service_interrupts_v1A_0E',
    'rpc_ctrl_nvenc_sw_session_update_info_v',
    'rpc_ctrl_nvenc_sw_session_update_info_v1A_09',
    'rpc_ctrl_nvlink_get_inband_received_data_v',
    'rpc_ctrl_nvlink_get_inband_received_data_v25_0C',
    'rpc_ctrl_perf_boost_v', 'rpc_ctrl_perf_boost_v1A_09',
    'rpc_ctrl_perf_rated_tdp_get_status_v',
    'rpc_ctrl_perf_rated_tdp_get_status_v1A_1F',
    'rpc_ctrl_perf_rated_tdp_set_control_v',
    'rpc_ctrl_perf_rated_tdp_set_control_v1A_1F',
    'rpc_ctrl_pm_area_pc_sampler_v',
    'rpc_ctrl_pm_area_pc_sampler_v21_0B',
    'rpc_ctrl_pma_stream_update_get_put_v',
    'rpc_ctrl_pma_stream_update_get_put_v1A_14', 'rpc_ctrl_preempt_v',
    'rpc_ctrl_preempt_v1A_0A', 'rpc_ctrl_release_ccu_prof_v',
    'rpc_ctrl_release_ccu_prof_v29_07', 'rpc_ctrl_release_hes_v',
    'rpc_ctrl_release_hes_v29_07', 'rpc_ctrl_reserve_ccu_prof_v',
    'rpc_ctrl_reserve_ccu_prof_v29_07', 'rpc_ctrl_reserve_hes_v',
    'rpc_ctrl_reserve_hes_v29_07', 'rpc_ctrl_reserve_hwpm_legacy_v',
    'rpc_ctrl_reserve_hwpm_legacy_v1A_0F',
    'rpc_ctrl_reserve_pm_area_smpc_v',
    'rpc_ctrl_reserve_pm_area_smpc_v1A_0F',
    'rpc_ctrl_reset_channel_v', 'rpc_ctrl_reset_channel_v1A_09',
    'rpc_ctrl_reset_isolated_channel_v',
    'rpc_ctrl_reset_isolated_channel_v1A_09',
    'rpc_ctrl_set_channel_interleave_level_v',
    'rpc_ctrl_set_channel_interleave_level_v1A_0A',
    'rpc_ctrl_set_hs_credits_v', 'rpc_ctrl_set_hs_credits_v21_08',
    'rpc_ctrl_set_timeslice_v', 'rpc_ctrl_set_timeslice_v1A_0A',
    'rpc_ctrl_set_tsg_interleave_level_v',
    'rpc_ctrl_set_tsg_interleave_level_v1A_0A',
    'rpc_ctrl_set_vgpu_fb_usage_v',
    'rpc_ctrl_set_vgpu_fb_usage_v1A_08',
    'rpc_ctrl_set_zbc_color_clear_v',
    'rpc_ctrl_set_zbc_color_clear_v1A_09',
    'rpc_ctrl_set_zbc_depth_clear_v',
    'rpc_ctrl_set_zbc_depth_clear_v1A_09',
    'rpc_ctrl_set_zbc_stencil_clear_v',
    'rpc_ctrl_set_zbc_stencil_clear_v27_06',
    'rpc_ctrl_stop_channel_v', 'rpc_ctrl_stop_channel_v1A_1E',
    'rpc_ctrl_subdevice_get_libos_heap_stats_v',
    'rpc_ctrl_subdevice_get_libos_heap_stats_v29_02',
    'rpc_ctrl_subdevice_get_p2p_caps_v',
    'rpc_ctrl_subdevice_get_p2p_caps_v21_02',
    'rpc_ctrl_subdevice_get_vgpu_heap_stats_v',
    'rpc_ctrl_subdevice_get_vgpu_heap_stats_v28_03',
    'rpc_ctrl_subdevice_get_vgpu_heap_stats_v28_06',
    'rpc_ctrl_timer_set_gr_tick_freq_v',
    'rpc_ctrl_timer_set_gr_tick_freq_v1A_1F',
    'rpc_ctrl_vaspace_copy_server_reserved_pdes_v',
    'rpc_ctrl_vaspace_copy_server_reserved_pdes_v1E_04',
    'rpc_dce_rm_init_v', 'rpc_dce_rm_init_v01_00',
    'rpc_disable_channels_v', 'rpc_disable_channels_v1E_0B',
    'rpc_display_modeset_v', 'rpc_display_modeset_v01_00',
    'rpc_dump_protobuf_component_v',
    'rpc_dump_protobuf_component_v18_12', 'rpc_dup_object_v',
    'rpc_dup_object_v03_00', 'rpc_extdev_intr_service_v',
    'rpc_extdev_intr_service_v17_00', 'rpc_fecs_error_v',
    'rpc_fecs_error_v26_02', 'rpc_free_v', 'rpc_free_v03_00',
    'rpc_get_brand_caps_v', 'rpc_get_brand_caps_v25_12',
    'rpc_get_consolidated_gr_static_info_v',
    'rpc_get_consolidated_gr_static_info_v1B_04',
    'rpc_get_encoder_capacity_v', 'rpc_get_encoder_capacity_v07_00',
    'rpc_get_engine_utilization_v',
    'rpc_get_engine_utilization_v1F_0E', 'rpc_get_gsp_static_info_v',
    'rpc_get_gsp_static_info_v14_00', 'rpc_get_static_data_v',
    'rpc_get_static_data_v25_0E', 'rpc_get_static_data_v27_01',
    'rpc_gpu_exec_reg_ops_v', 'rpc_gpu_exec_reg_ops_v12_01',
    'rpc_gpuacct_perfmon_util_samples_v',
    'rpc_gpuacct_perfmon_util_samples_v1F_0E',
    'rpc_gsp_lockdown_notice_v', 'rpc_gsp_lockdown_notice_v17_00',
    'rpc_gsp_post_nocat_record_v', 'rpc_gsp_post_nocat_record_v01_00',
    'rpc_gsp_rm_alloc_v', 'rpc_gsp_rm_alloc_v03_00',
    'rpc_gsp_rm_control_v', 'rpc_gsp_rm_control_v03_00',
    'rpc_gsp_set_system_info_v', 'rpc_gsp_set_system_info_v17_00',
    'rpc_idle_channels_v', 'rpc_idle_channels_v03_00',
    'rpc_init_done_v', 'rpc_init_done_v17_00', 'rpc_invalidate_tlb_v',
    'rpc_invalidate_tlb_v23_03', 'rpc_log_v', 'rpc_log_v03_00',
    'rpc_map_memory_dma_v', 'rpc_map_memory_dma_v03_00',
    'rpc_message_header_v', 'rpc_message_header_v03_00',
    'rpc_message_rpc_union_field_v',
    'rpc_message_rpc_union_field_v03_00',
    'rpc_nvlink_fatal_error_recovery_v',
    'rpc_nvlink_fatal_error_recovery_v17_00', 'rpc_nvlink_fault_up_v',
    'rpc_nvlink_fault_up_v17_00',
    'rpc_nvlink_inband_received_data_1024_v',
    'rpc_nvlink_inband_received_data_1024_v17_00',
    'rpc_nvlink_inband_received_data_2048_v',
    'rpc_nvlink_inband_received_data_2048_v17_00',
    'rpc_nvlink_inband_received_data_256_v',
    'rpc_nvlink_inband_received_data_256_v17_00',
    'rpc_nvlink_inband_received_data_4096_v',
    'rpc_nvlink_inband_received_data_4096_v17_00',
    'rpc_nvlink_inband_received_data_512_v',
    'rpc_nvlink_inband_received_data_512_v17_00',
    'rpc_nvlink_is_gpu_degraded_v',
    'rpc_nvlink_is_gpu_degraded_v17_00', 'rpc_os_error_log_v',
    'rpc_os_error_log_v17_00', 'rpc_perf_bridgeless_info_update_v',
    'rpc_perf_bridgeless_info_update_v17_00',
    'rpc_perf_get_level_info_v', 'rpc_perf_get_level_info_v03_00',
    'rpc_perf_gpu_boost_sync_limits_callback_v',
    'rpc_perf_gpu_boost_sync_limits_callback_v17_00',
    'rpc_pfm_req_hndlr_state_sync_callback_v',
    'rpc_pfm_req_hndlr_state_sync_callback_v21_04',
    'rpc_post_event_v', 'rpc_post_event_v17_00', 'rpc_rc_triggered_v',
    'rpc_rc_triggered_v17_02', 'rpc_recovery_action_v',
    'rpc_recovery_action_v28_01', 'rpc_restore_hibernation_data_v',
    'rpc_restore_hibernation_data_v1E_0E', 'rpc_rg_line_intr_v',
    'rpc_rg_line_intr_v17_00', 'rpc_rm_api_control_v',
    'rpc_rm_api_control_v25_0D', 'rpc_rm_api_control_v25_0F',
    'rpc_rm_api_control_v25_10', 'rpc_rm_api_control_v25_14',
    'rpc_rm_api_control_v25_15', 'rpc_rm_api_control_v25_16',
    'rpc_rm_api_control_v25_17', 'rpc_rm_api_control_v25_18',
    'rpc_rm_api_control_v25_19', 'rpc_rm_api_control_v25_1A',
    'rpc_rm_api_control_v27_03', 'rpc_rm_api_control_v29_04',
    'rpc_rm_api_control_v29_09', 'rpc_run_cpu_sequencer_v',
    'rpc_run_cpu_sequencer_v17_00', 'rpc_save_hibernation_data_v',
    'rpc_save_hibernation_data_v1E_0E',
    'rpc_semaphore_schedule_callback_v',
    'rpc_semaphore_schedule_callback_v17_00',
    'rpc_set_guest_system_info_ext_v',
    'rpc_set_guest_system_info_ext_v15_02',
    'rpc_set_guest_system_info_ext_v25_1B',
    'rpc_set_guest_system_info_v', 'rpc_set_guest_system_info_v03_00',
    'rpc_set_page_directory_v', 'rpc_set_page_directory_v1E_05',
    'rpc_set_surface_properties_v',
    'rpc_set_surface_properties_v07_07', 'rpc_sim_read_v',
    'rpc_sim_read_v1E_01', 'rpc_sim_write_v', 'rpc_sim_write_v1E_01',
    'rpc_timed_semaphore_release_v',
    'rpc_timed_semaphore_release_v01_00', 'rpc_ucode_libos_print_v',
    'rpc_ucode_libos_print_v1E_08', 'rpc_unloading_guest_driver_v',
    'rpc_unloading_guest_driver_v1F_07', 'rpc_unmap_memory_dma_v',
    'rpc_unmap_memory_dma_v03_00', 'rpc_unset_page_directory_v',
    'rpc_unset_page_directory_v1E_05', 'rpc_update_bar_pde_v',
    'rpc_update_bar_pde_v15_00', 'rpc_update_gpm_guest_buffer_info_v',
    'rpc_update_gpm_guest_buffer_info_v27_01',
    'rpc_update_gsp_trace_v', 'rpc_update_gsp_trace_v01_00',
    'rpc_vgpu_config_event_v', 'rpc_vgpu_config_event_v17_00',
    'rpc_vgpu_gsp_mig_ci_config_v',
    'rpc_vgpu_gsp_mig_ci_config_v21_03',
    'rpc_vgpu_gsp_plugin_triggered_v',
    'rpc_vgpu_gsp_plugin_triggered_v17_00',
    'rpc_vgpu_pf_reg_read32_v', 'rpc_vgpu_pf_reg_read32_v15_00',
    'struct_ACPI_METHOD_DATA', 'struct_BIT_HEADER_V1_00',
    'struct_BIT_TOKEN_V1_00', 'struct_CAPS_METHOD_DATA',
    'struct_DOD_METHOD_DATA', 'struct_GSP_ACR_BOOT_GSP_RM_PARAMS',
    'struct_GSP_FMC_BOOT_PARAMS', 'struct_GSP_FMC_INIT_PARAMS',
    'struct_GSP_MSG_QUEUE_ELEMENT', 'struct_GSP_RM_PARAMS',
    'struct_GSP_SPDM_PARAMS', 'struct_GSP_VF_INFO',
    'struct_GspStaticConfigInfo_t', 'struct_GspSystemInfo',
    'struct_JT_METHOD_DATA', 'struct_MUX_METHOD_DATA',
    'struct_MUX_METHOD_DATA_ELEMENT',
    'struct_NV0000_CTRL_GPUACCT_CLEAR_ACCOUNTING_DATA_PARAMS_v09_0C',
    'struct_NV0000_CTRL_GPUACCT_GET_ACCOUNTING_PIDS_PARAMS_v09_0C',
    'struct_NV0000_CTRL_GPUACCT_GET_ACCOUNTING_STATE_PARAMS_v09_0C',
    'struct_NV0000_CTRL_GPUACCT_GET_PROC_ACCOUNTING_INFO_PARAMS_v09_0C',
    'struct_NV0000_CTRL_GPUACCT_SET_ACCOUNTING_STATE_PARAMS_v09_0C',
    'struct_NV0000_CTRL_P2P_CAPS_MATRIX_ROW_v18_0A',
    'struct_NV0000_CTRL_SYSTEM_GET_P2P_CAPS_MATRIX_PARAMS_v18_0A',
    'struct_NV0000_CTRL_SYSTEM_GET_P2P_CAPS_PARAMS_v1F_0D',
    'struct_NV0080_CTRL_DMA_SET_DEFAULT_VASPACE_PARAMS_v03_00',
    'struct_NV0080_CTRL_DMA_SET_PAGE_DIRECTORY_PARAMS_v1E_05',
    'struct_NV0080_CTRL_DMA_UNSET_PAGE_DIRECTORY_PARAMS_v1E_05',
    'struct_NV0080_CTRL_FIFO_SET_CHANNEL_PROPERTIES_PARAMS_v03_00',
    'struct_NV0080_CTRL_GPU_GET_SRIOV_CAPS_PARAMS',
    'struct_NV0080_CTRL_GR_TPC_PARTITION_MODE_PARAMS_v1C_04',
    'struct_NV0090_CTRL_GET_MMU_DEBUG_MODE_PARAMS_v1E_06',
    'struct_NV00F8_ALLOCATION_PARAMETERS_MAP_STRUCT_v1E_0C',
    'struct_NV00F8_CTRL_DESCRIBE_PARAMS_v1E_0C',
    'struct_NV2080_CTRL_BIOS_GET_SKU_INFO_PARAMS',
    'struct_NV2080_CTRL_BUS_SET_P2P_MAPPING_PARAMS_v21_03',
    'struct_NV2080_CTRL_BUS_SET_P2P_MAPPING_PARAMS_v29_08',
    'struct_NV2080_CTRL_BUS_UNSET_P2P_MAPPING_PARAMS_v21_03',
    'struct_NV2080_CTRL_CE_GET_CE_PCE_MASK_PARAMS_v1A_07',
    'struct_NV2080_CTRL_CMD_FB_GET_FB_REGION_FB_REGION_INFO',
    'struct_NV2080_CTRL_CMD_FB_GET_FB_REGION_INFO_PARAMS',
    'struct_NV2080_CTRL_CMD_GPU_HANDLE_VF_PRI_FAULT_PARAMS_v18_09',
    'struct_NV2080_CTRL_CMD_GSP_GET_LIBOS_HEAP_STATS_PARAMS_v29_02',
    'struct_NV2080_CTRL_CMD_GSP_GET_VGPU_HEAP_STATS_PARAMS_v28_03',
    'struct_NV2080_CTRL_CMD_GSP_GET_VGPU_HEAP_STATS_PARAMS_v28_06',
    'struct_NV2080_CTRL_CMD_INTERNAL_MEMSYS_SET_ZBC_REFERENCED_v1F_05',
    'struct_NV2080_CTRL_CMD_NVLINK_GET_NVLINK_STATUS_PARAMS_v23_04',
    'struct_NV2080_CTRL_CMD_NVLINK_GET_NVLINK_STATUS_PARAMS_v28_09',
    'struct_NV2080_CTRL_CMD_TIMER_SET_GR_TICK_FREQ_PARAMS_v1A_1F',
    'struct_NV2080_CTRL_FB_FS_INFO_FBPA_MASK_PARAMS_v1A_1D',
    'struct_NV2080_CTRL_FB_FS_INFO_FBPA_SUBP_MASK_PARAMS_v1A_1D',
    'struct_NV2080_CTRL_FB_FS_INFO_FBP_LOGICAL_MAP_PARAMS_v1A_1D',
    'struct_NV2080_CTRL_FB_FS_INFO_FBP_MASK_PARAMS_v1A_1D',
    'struct_NV2080_CTRL_FB_FS_INFO_INVALID_QUERY_PARAMS_v1A_1D',
    'struct_NV2080_CTRL_FB_FS_INFO_LOGICAL_LTC_MASK_PARAMS_v26_04',
    'struct_NV2080_CTRL_FB_FS_INFO_LTC_MASK_PARAMS_v1A_1D',
    'struct_NV2080_CTRL_FB_FS_INFO_LTS_MASK_PARAMS_v1A_1D',
    'struct_NV2080_CTRL_FB_FS_INFO_PAC_MASK_PARAMS_v26_04',
    'struct_NV2080_CTRL_FB_FS_INFO_PROFILER_MON_FBPA_MASK_PARAMS_v1A_1D',
    'struct_NV2080_CTRL_FB_FS_INFO_PROFILER_MON_FBPA_SUBP_MASK_PARAMS_v1A_1D',
    'struct_NV2080_CTRL_FB_FS_INFO_PROFILER_MON_LOGICAL_LTC_MASK_PARAMS_v26_04',
    'struct_NV2080_CTRL_FB_FS_INFO_PROFILER_MON_LTC_MASK_PARAMS_v1A_1D',
    'struct_NV2080_CTRL_FB_FS_INFO_PROFILER_MON_LTS_MASK_PARAMS_v1A_1D',
    'struct_NV2080_CTRL_FB_FS_INFO_PROFILER_MON_ROP_MASK_PARAMS_v1A_1D',
    'struct_NV2080_CTRL_FB_FS_INFO_QUERY_v1A_1D',
    'struct_NV2080_CTRL_FB_FS_INFO_QUERY_v26_04',
    'struct_NV2080_CTRL_FB_FS_INFO_ROP_MASK_PARAMS_v1A_1D',
    'struct_NV2080_CTRL_FB_GET_FS_INFO_PARAMS_v24_00',
    'struct_NV2080_CTRL_FB_GET_FS_INFO_PARAMS_v26_04',
    'struct_NV2080_CTRL_FB_GET_INFO_V2_PARAMS_v25_0A',
    'struct_NV2080_CTRL_FB_GET_INFO_V2_PARAMS_v27_00',
    'struct_NV2080_CTRL_FB_INFO_v1A_15',
    'struct_NV2080_CTRL_FIFO_DISABLE_CHANNELS_PARAMS_v06_00',
    'struct_NV2080_CTRL_FIFO_SETUP_VF_ZOMBIE_SUBCTX_PDB_PARAMS_v1A_23',
    'struct_NV2080_CTRL_FLA_GET_FABRIC_MEM_STATS_PARAMS_v1E_0C',
    'struct_NV2080_CTRL_FLA_SETUP_INSTANCE_MEM_BLOCK_PARAMS_v13_04',
    'struct_NV2080_CTRL_GET_P2P_CAPS_PARAMS_v21_02',
    'struct_NV2080_CTRL_GPU_EVICT_CTX_PARAMS_v03_00',
    'struct_NV2080_CTRL_GPU_EXEC_REG_OPS_PARAMS_v12_01',
    'struct_NV2080_CTRL_GPU_GET_GID_INFO_PARAMS',
    'struct_NV2080_CTRL_GPU_GET_INFO_V2_PARAMS_v25_11',
    'struct_NV2080_CTRL_GPU_INFO_v25_11',
    'struct_NV2080_CTRL_GPU_INITIALIZE_CTX_PARAMS_v03_00',
    'struct_NV2080_CTRL_GPU_MIGRATABLE_OPS_PARAMS_v21_07',
    'struct_NV2080_CTRL_GPU_P2P_PEER_CAPS_PEER_INFO_v21_02',
    'struct_NV2080_CTRL_GPU_PROMOTE_CTX_BUFFER_ENTRY_v1A_20',
    'struct_NV2080_CTRL_GPU_PROMOTE_CTX_PARAMS_v1A_20',
    'struct_NV2080_CTRL_GPU_QUERY_ECC_EXCEPTION_STATUS_v15_01',
    'struct_NV2080_CTRL_GPU_QUERY_ECC_STATUS_DEPRECATED_RPC_PARAMS_v24_06',
    'struct_NV2080_CTRL_GPU_QUERY_ECC_STATUS_PARAMS_v26_02',
    'struct_NV2080_CTRL_GPU_QUERY_ECC_UNIT_STATUS_v15_01',
    'struct_NV2080_CTRL_GPU_REG_OP_v03_00',
    'struct_NV2080_CTRL_GRMGR_GET_GR_FS_INFO_PARAMS_v1A_1D',
    'struct_NV2080_CTRL_GRMGR_GR_FS_INFO_CHIPLET_GPC_MAP_PARAMS_v1A_1D',
    'struct_NV2080_CTRL_GRMGR_GR_FS_INFO_CHIPLET_SYSPIPE_MASK_PARAMS_v1A_1D',
    'struct_NV2080_CTRL_GRMGR_GR_FS_INFO_GPC_COUNT_PARAMS_v1A_1D',
    'struct_NV2080_CTRL_GRMGR_GR_FS_INFO_PARTITION_CHIPLET_GPC_MAP_PARAMS_v1A_1D',
    'struct_NV2080_CTRL_GRMGR_GR_FS_INFO_PARTITION_CHIPLET_SYSPIPE_IDS_PARAMS_v1A_1D',
    'struct_NV2080_CTRL_GRMGR_GR_FS_INFO_PARTITION_SYSPIPE_ID_PARAMS_v1A_1D',
    'struct_NV2080_CTRL_GRMGR_GR_FS_INFO_PPC_MASK_PARAMS_v1A_1D',
    'struct_NV2080_CTRL_GRMGR_GR_FS_INFO_PROFILER_MON_GPC_MASK_PARAMS_v1A_1D',
    'struct_NV2080_CTRL_GRMGR_GR_FS_INFO_QUERY_PARAMS_v1A_1D',
    'struct_NV2080_CTRL_GRMGR_GR_FS_INFO_ROP_MASK_PARAMS_v1A_1D',
    'struct_NV2080_CTRL_GRMGR_GR_FS_INFO_TPC_MASK_PARAMS_v1A_1D',
    'struct_NV2080_CTRL_GR_CTXSW_PREEMPTION_BIND_PARAMS_v12_01',
    'struct_NV2080_CTRL_GR_CTXSW_PREEMPTION_BIND_PARAMS_v28_07',
    'struct_NV2080_CTRL_GR_CTXSW_ZCULL_BIND_PARAMS_v03_00',
    'struct_NV2080_CTRL_GR_PC_SAMPLING_MODE_PARAMS_v1A_1F',
    'struct_NV2080_CTRL_GR_ROUTE_INFO_v12_01',
    'struct_NV2080_CTRL_GR_SET_CTXSW_PREEMPTION_MODE_PARAMS_v12_01',
    'struct_NV2080_CTRL_GSP_LIBOS_POOL_STATS_v29_02',
    'struct_NV2080_CTRL_INTERNAL_MEMDESC_INFO_v1E_07',
    'struct_NV2080_CTRL_INTERNAL_PERF_GPU_BOOST_SYNC_SET_LIMITS_PARAMS_v17_00',
    'struct_NV2080_CTRL_INTERNAL_PFM_REQ_HNDLR_STATE_SYNC_DATA_SMBPBI_v21_04',
    'struct_NV2080_CTRL_INTERNAL_PFM_REQ_HNDLR_STATE_SYNC_PARAMS_v21_04',
    'struct_NV2080_CTRL_INTERNAL_PFM_REQ_HNDLR_STATE_SYNC_data_v21_04',
    'struct_NV2080_CTRL_MC_SERVICE_INTERRUPTS_PARAMS_v15_01',
    'struct_NV2080_CTRL_NVLINK_DEVICE_INFO_v15_02',
    'struct_NV2080_CTRL_NVLINK_DEVICE_INFO_v28_09',
    'struct_NV2080_CTRL_NVLINK_FATAL_ERROR_RECOVERY_PARAMS_v17_00',
    'struct_NV2080_CTRL_NVLINK_INBAND_RECEIVED_DATA_1024_PARAMS_v17_00',
    'struct_NV2080_CTRL_NVLINK_INBAND_RECEIVED_DATA_2048_PARAMS_v17_00',
    'struct_NV2080_CTRL_NVLINK_INBAND_RECEIVED_DATA_256_PARAMS_v17_00',
    'struct_NV2080_CTRL_NVLINK_INBAND_RECEIVED_DATA_4096_PARAMS_v17_00',
    'struct_NV2080_CTRL_NVLINK_INBAND_RECEIVED_DATA_512_PARAMS_v17_00',
    'struct_NV2080_CTRL_NVLINK_INBAND_RECEIVED_DATA_PARAMS_v25_0C',
    'struct_NV2080_CTRL_NVLINK_IS_GPU_DEGRADED_PARAMS_v17_00',
    'struct_NV2080_CTRL_NVLINK_LINK_STATUS_INFO_v18_0D',
    'struct_NV2080_CTRL_NVLINK_LINK_STATUS_INFO_v28_09',
    'struct_NV2080_CTRL_PERF_BOOST_PARAMS_v03_00',
    'struct_NV2080_CTRL_PERF_GET_GPUMON_PERFMON_UTIL_SAMPLES_V2_PARAMS_v1F_0E',
    'struct_NV2080_CTRL_PERF_GET_VID_ENG_PERFMON_SAMPLE_PARAMS_v05_00',
    'struct_NV2080_CTRL_PERF_GPUMON_ENGINE_UTIL_SAMPLE_v17_00',
    'struct_NV2080_CTRL_PERF_GPUMON_PERFMON_UTIL_SAMPLE_v1F_0E',
    'struct_NV2080_CTRL_PERF_RATED_TDP_CONTROL_PARAMS_v1A_1F',
    'struct_NV2080_CTRL_PERF_RATED_TDP_STATUS_PARAMS_v1A_1F',
    'struct_NV2080_CTRL_SYSL2_FS_INFO_SYSLTC_MASK_PARAMS_v26_04',
    'struct_NV506F_CTRL_CMD_RESET_ISOLATED_CHANNEL_PARAMS_v03_00',
    'struct_NV83DE_CTRL_CMD_DEBUG_SUSPEND_CONTEXT_PARAMS_v1A_06',
    'struct_NV83DE_CTRL_DEBUG_CLEAR_ALL_SM_ERROR_STATES_PARAMS_v03_00',
    'struct_NV83DE_CTRL_DEBUG_CLEAR_SINGLE_SM_ERROR_STATE_PARAMS_v1A_06',
    'struct_NV83DE_CTRL_DEBUG_EXEC_REG_OPS_PARAMS_v1A_06',
    'struct_NV83DE_CTRL_DEBUG_GET_MODE_MMU_DEBUG_PARAMS_v25_04',
    'struct_NV83DE_CTRL_DEBUG_GET_MODE_MMU_GCC_DEBUG_PARAMS_v29_07',
    'struct_NV83DE_CTRL_DEBUG_READ_ALL_SM_ERROR_STATES_PARAMS_v21_06',
    'struct_NV83DE_CTRL_DEBUG_READ_SINGLE_SM_ERROR_STATE_PARAMS_v21_06',
    'struct_NV83DE_CTRL_DEBUG_SET_EXCEPTION_MASK_PARAMS_v03_00',
    'struct_NV83DE_CTRL_DEBUG_SET_MODE_ERRBAR_DEBUG_PARAMS_v1A_06',
    'struct_NV83DE_CTRL_DEBUG_SET_MODE_MMU_DEBUG_PARAMS_v1A_06',
    'struct_NV83DE_CTRL_DEBUG_SET_MODE_MMU_GCC_DEBUG_PARAMS_v29_07',
    'struct_NV83DE_CTRL_DEBUG_SET_NEXT_STOP_TRIGGER_TYPE_PARAMS_v1A_06',
    'struct_NV83DE_CTRL_DEBUG_SET_SINGLE_SM_SINGLE_STEP_PARAMS_v1C_02',
    'struct_NV83DE_MMU_FAULT_INFO_v16_03',
    'struct_NV83DE_SM_ERROR_STATE_REGISTERS_v21_06',
    'struct_NV906F_CTRL_CMD_RESET_CHANNEL_PARAMS_v10_01',
    'struct_NV9096_CTRL_GET_ZBC_CLEAR_TABLE_ENTRY_PARAMS_v1A_07',
    'struct_NV9096_CTRL_GET_ZBC_CLEAR_TABLE_ENTRY_PARAMS_value_v1A_07',
    'struct_NV9096_CTRL_GET_ZBC_CLEAR_TABLE_PARAMS_v04_00',
    'struct_NV9096_CTRL_GET_ZBC_CLEAR_TABLE_PARAMS_value_v04_00',
    'struct_NV9096_CTRL_SET_ZBC_COLOR_CLEAR_PARAMS_v03_00',
    'struct_NV9096_CTRL_SET_ZBC_DEPTH_CLEAR_PARAMS_v03_00',
    'struct_NV9096_CTRL_SET_ZBC_STENCIL_CLEAR_PARAMS_v27_06',
    'struct_NV90E6_CTRL_MASTER_GET_VIRTUAL_FUNCTION_ERROR_CONT_INTR_MASK_PARAMS_v18_0B',
    'struct_NV90F1_CTRL_VASPACE_COPY_SERVER_RESERVED_PDES_PARAMS_levels_v1E_04',
    'struct_NV90F1_CTRL_VASPACE_COPY_SERVER_RESERVED_PDES_PARAMS_v1E_04',
    'struct_NVA06C_CTRL_INTERLEAVE_LEVEL_PARAMS_v17_02',
    'struct_NVA06C_CTRL_INTERNAL_PROMOTE_FAULT_METHOD_BUFFERS_PARAMS_v1E_07',
    'struct_NVA06C_CTRL_PREEMPT_PARAMS_v09_0A',
    'struct_NVA06C_CTRL_TIMESLICE_PARAMS_v06_00',
    'struct_NVA06F_CTRL_GPFIFO_SCHEDULE_PARAMS_v03_00',
    'struct_NVA06F_CTRL_INTERLEAVE_LEVEL_PARAMS_v17_02',
    'struct_NVA06F_CTRL_STOP_CHANNEL_PARAMS_v1A_1E',
    'struct_NVA080_CTRL_SET_FB_USAGE_PARAMS_v07_02',
    'struct_NVA080_CTRL_VGPU_DISPLAY_CLEANUP_SURFACE_PARAMS_v03_00',
    'struct_NVA080_CTRL_VGPU_DISPLAY_SET_SURFACE_PROPERTIES_v07_07',
    'struct_NVA0BC_CTRL_NVENC_SW_SESSION_UPDATE_INFO_PARAMS_v06_01',
    'struct_NVB0CC_CTRL_ALLOC_PMA_STREAM_PARAMS_v1A_14',
    'struct_NVB0CC_CTRL_EXEC_REG_OPS_PARAMS_v1A_0F',
    'struct_NVB0CC_CTRL_FREE_PMA_STREAM_PARAMS_v1A_1F',
    'struct_NVB0CC_CTRL_GET_HS_CREDITS_PARAMS_v21_08',
    'struct_NVB0CC_CTRL_GET_TOTAL_HS_CREDITS_PARAMS_v21_08',
    'struct_NVB0CC_CTRL_HES_RESERVATION_UNION_v29_07',
    'struct_NVB0CC_CTRL_INTERNAL_QUIESCE_PMA_CHANNEL_PARAMS_v1C_08',
    'struct_NVB0CC_CTRL_INTERNAL_SRIOV_PROMOTE_PMA_STREAM_PARAMS_v1C_0C',
    'struct_NVB0CC_CTRL_PMA_STREAM_HS_CREDITS_INFO_v21_08',
    'struct_NVB0CC_CTRL_PMA_STREAM_HS_CREDITS_STATUS_v21_08',
    'struct_NVB0CC_CTRL_PMA_STREAM_UPDATE_GET_PUT_PARAMS_v1A_14',
    'struct_NVB0CC_CTRL_RELEASE_HES_PARAMS_v29_07',
    'struct_NVB0CC_CTRL_RESERVE_CCUPROF_PARAMS_v29_07',
    'struct_NVB0CC_CTRL_RESERVE_HES_CWD_PARAMS_v29_07',
    'struct_NVB0CC_CTRL_RESERVE_HES_PARAMS_v29_07',
    'struct_NVB0CC_CTRL_RESERVE_HWPM_LEGACY_PARAMS_v1A_0F',
    'struct_NVB0CC_CTRL_RESERVE_PM_AREA_SMPC_PARAMS_v1A_0F',
    'struct_NVB0CC_CTRL_SET_HS_CREDITS_PARAMS_v21_08',
    'struct_NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS_v08_00',
    'struct_NVC36F_CTRL_GPFIFO_SET_WORK_SUBMIT_TOKEN_NOTIF_INDEX_PARAMS_v16_04',
    'struct_NVC637_CTRL_EXEC_PARTITIONS_CREATE_PARAMS_v24_05',
    'struct_NVC637_CTRL_EXEC_PARTITIONS_DELETE_PARAMS_v18_05',
    'struct_NVC637_CTRL_EXEC_PARTITIONS_INFO_v24_05',
    'struct_NVOS00_PARAMETERS_v03_00',
    'struct_NVOS21_PARAMETERS_v03_00',
    'struct_NVOS46_PARAMETERS_v03_00',
    'struct_NVOS47_PARAMETERS_v03_00',
    'struct_NVOS54_PARAMETERS_v03_00',
    'struct_NVOS55_PARAMETERS_v03_00',
    'struct_NV_CHANNEL_ALLOC_PARAMS_v1F_04',
    'struct_NV_DEVICE_ALLOCATION_PARAMETERS_v03_00',
    'struct_NV_GR_ALLOCATION_PARAMETERS_v1A_17',
    'struct_NV_MEMORY_DESC_PARAMS_v18_01',
    'struct_PACKED_REGISTRY_ENTRY', 'struct_PACKED_REGISTRY_TABLE',
    'struct_PERF_RATED_TDP_RM_INTERNAL_STATE_STRUCT_v1A_1F',
    'struct_UpdateBarPde_v15_00',
    'struct_VIRTUAL_DISPLAY_GET_MAX_RESOLUTION_PARAMS',
    'struct_VIRTUAL_DISPLAY_GET_NUM_HEADS_PARAMS',
    'struct_alloc_object_FERMI_CONTEXT_SHARE_A_v04_00',
    'struct_alloc_object_FERMI_VASPACE_A_v03_00',
    'struct_alloc_object_GF100_DISP_SW_v03_00',
    'struct_alloc_object_GT212_DMA_COPY_v03_00',
    'struct_alloc_object_KEPLER_CHANNEL_GROUP_A_v12_08',
    'struct_alloc_object_NV00F8_ALLOCATION_PARAMETERS_v1E_0C',
    'struct_alloc_object_NV2081_ALLOC_PARAMETERS_v25_08',
    'struct_alloc_object_NV503B_ALLOC_PARAMETERS_v1D_02',
    'struct_alloc_object_NV503C_ALLOC_PARAMETERS_v18_15',
    'struct_alloc_object_NV50_TESLA_v03_00',
    'struct_alloc_object_NV83DE_ALLOC_PARAMETERS_v03_00',
    'struct_alloc_object_NVB0B0_VIDEO_DECODER_v03_00',
    'struct_alloc_object_NVB1CC_ALLOC_PARAMETERS_v1A_03',
    'struct_alloc_object_NVB2CC_ALLOC_PARAMETERS_v1A_03',
    'struct_alloc_object_NVC4B0_VIDEO_DECODER_v12_02',
    'struct_alloc_object_NVC637_ALLOCATION_PARAMETERS_v13_00',
    'struct_alloc_object_NVC638_ALLOCATION_PARAMETERS_v18_06',
    'struct_alloc_object_NVC670_ALLOCATION_PARAMETERS_v1A_01',
    'struct_alloc_object_NVC9FA_VIDEO_OFA_v1F_00',
    'struct_alloc_object_NVC9FA_VIDEO_OFA_v29_06',
    'struct_alloc_object_NVD0B7_VIDEO_ENCODER_v03_00',
    'struct_alloc_object_NVENC_SW_SESSION_v06_01',
    'struct_alloc_object_NVFBC_SW_SESSION_v12_04',
    'struct_alloc_object_NV_MEMORY_VIRTUAL_ALLOCATION_PARAMS_v13_03',
    'struct_alloc_object_NV_NVJPG_ALLOCATION_PARAMETERS_v20_02',
    'struct_alloc_object_NV_UVM_CHANNEL_RETAINER_ALLOC_PARAMS_v1A_1B',
    'struct_c__SA_ACPI_DATA', 'struct_c__SA_ACPI_DSM_CACHE',
    'struct_c__SA_BIT_DATA_BIOSDATA_BINVER',
    'struct_c__SA_BIT_DATA_FALCON_DATA_V2', 'struct_c__SA_BUSINFO',
    'struct_c__SA_EcidManufacturingInfo',
    'struct_c__SA_FALCON_APPLICATION_INTERFACE_DMEM_MAPPER_V3',
    'struct_c__SA_FALCON_UCODE_DESC_HEADER',
    'struct_c__SA_FALCON_UCODE_DESC_V3',
    'struct_c__SA_FALCON_UCODE_TABLE_ENTRY_V1',
    'struct_c__SA_FALCON_UCODE_TABLE_HDR_V1',
    'struct_c__SA_FWSECLIC_FRTS_CMD',
    'struct_c__SA_FWSECLIC_FRTS_REGION_DESC',
    'struct_c__SA_FWSECLIC_READ_VBIOS_DESC',
    'struct_c__SA_FW_WPR_LAYOUT_OFFSET',
    'struct_c__SA_GSP_ARGUMENTS_CACHED',
    'struct_c__SA_GSP_ARGUMENTS_CACHED_profilerArgs',
    'struct_c__SA_GSP_PCIE_CONFIG_REG',
    'struct_c__SA_GSP_SR_INIT_ARGUMENTS',
    'struct_c__SA_GspFwHeapFreeList',
    'struct_c__SA_GspFwHeapFreeRegion', 'struct_c__SA_GspFwSRMeta',
    'struct_c__SA_GspFwWprMeta', 'struct_c__SA_GspFwWprMeta_0_0',
    'struct_c__SA_GspFwWprMeta_0_1', 'struct_c__SA_GspFwWprMeta_1_0',
    'struct_c__SA_GspFwWprMeta_1_1',
    'struct_c__SA_LibosMemoryRegionInitArgument',
    'struct_c__SA_MCTP_HEADER',
    'struct_c__SA_MESSAGE_QUEUE_INIT_ARGUMENTS',
    'struct_c__SA_NVDM_PAYLOAD_COT',
    'struct_c__SA_RM_RISCV_UCODE_DESC', 'struct_c__SA_msgqMetadata',
    'struct_c__SA_msgqRxHeader', 'struct_c__SA_msgqTxHeader',
    'struct_gpu_exec_reg_ops_v12_01',
    'struct_idle_channel_list_v03_00', 'struct_pte_desc',
    'struct_rpc_alloc_channel_dma_v1F_04',
    'struct_rpc_alloc_event_v03_00', 'struct_rpc_alloc_memory_v13_01',
    'struct_rpc_alloc_object_v25_08',
    'struct_rpc_alloc_object_v26_00',
    'struct_rpc_alloc_object_v27_00',
    'struct_rpc_alloc_object_v29_06', 'struct_rpc_alloc_root_v07_00',
    'struct_rpc_alloc_share_device_v03_00',
    'struct_rpc_alloc_subdevice_v08_01',
    'struct_rpc_cleanup_surface_v03_00',
    'struct_rpc_ctrl_alloc_pma_stream_v1A_14',
    'struct_rpc_ctrl_b0cc_exec_reg_ops_v1A_0F',
    'struct_rpc_ctrl_bind_pm_resources_v1A_0F',
    'struct_rpc_ctrl_bus_set_p2p_mapping_v21_03',
    'struct_rpc_ctrl_bus_set_p2p_mapping_v29_08',
    'struct_rpc_ctrl_bus_unset_p2p_mapping_v21_03',
    'struct_rpc_ctrl_cmd_internal_control_gsp_trace_v28_00',
    'struct_rpc_ctrl_cmd_internal_gpu_start_fabric_probe_v25_09',
    'struct_rpc_ctrl_cmd_nvlink_inband_send_data_v26_05',
    'struct_rpc_ctrl_dbg_clear_all_sm_error_states_v1A_0C',
    'struct_rpc_ctrl_dbg_clear_single_sm_error_state_v1A_10',
    'struct_rpc_ctrl_dbg_exec_reg_ops_v1A_10',
    'struct_rpc_ctrl_dbg_get_mode_mmu_debug_v25_04',
    'struct_rpc_ctrl_dbg_get_mode_mmu_gcc_debug_v29_07',
    'struct_rpc_ctrl_dbg_read_all_sm_error_states_v21_06',
    'struct_rpc_ctrl_dbg_read_single_sm_error_state_v21_06',
    'struct_rpc_ctrl_dbg_resume_context_v1A_10',
    'struct_rpc_ctrl_dbg_set_exception_mask_v1A_0C',
    'struct_rpc_ctrl_dbg_set_mode_errbar_debug_v1A_10',
    'struct_rpc_ctrl_dbg_set_mode_mmu_debug_v1A_10',
    'struct_rpc_ctrl_dbg_set_mode_mmu_gcc_debug_v29_07',
    'struct_rpc_ctrl_dbg_set_next_stop_trigger_type_v1A_10',
    'struct_rpc_ctrl_dbg_set_single_sm_single_step_v1C_02',
    'struct_rpc_ctrl_dbg_suspend_context_v1A_10',
    'struct_rpc_ctrl_dma_set_default_vaspace_v1A_0E',
    'struct_rpc_ctrl_exec_partitions_create_v24_05',
    'struct_rpc_ctrl_exec_partitions_delete_v1F_0A',
    'struct_rpc_ctrl_fabric_mem_stats_v1E_0C',
    'struct_rpc_ctrl_fabric_memory_describe_v1E_0C',
    'struct_rpc_ctrl_fb_get_fs_info_v24_00',
    'struct_rpc_ctrl_fb_get_fs_info_v26_04',
    'struct_rpc_ctrl_fb_get_info_v2_v25_0A',
    'struct_rpc_ctrl_fb_get_info_v2_v27_00',
    'struct_rpc_ctrl_fifo_disable_channels_v1A_0A',
    'struct_rpc_ctrl_fifo_set_channel_properties_v1A_16',
    'struct_rpc_ctrl_fifo_setup_vf_zombie_subctx_pdb_v1A_23',
    'struct_rpc_ctrl_fla_setup_instance_mem_block_v21_05',
    'struct_rpc_ctrl_free_pma_stream_v1A_1F',
    'struct_rpc_ctrl_get_ce_pce_mask_v1A_0E',
    'struct_rpc_ctrl_get_hs_credits_v21_08',
    'struct_rpc_ctrl_get_mmu_debug_mode_v1E_06',
    'struct_rpc_ctrl_get_nvlink_status_v23_04',
    'struct_rpc_ctrl_get_nvlink_status_v28_09',
    'struct_rpc_ctrl_get_p2p_caps_matrix_v1A_0E',
    'struct_rpc_ctrl_get_p2p_caps_v1F_0D',
    'struct_rpc_ctrl_get_p2p_caps_v2_v1F_0D',
    'struct_rpc_ctrl_get_total_hs_credits_v21_08',
    'struct_rpc_ctrl_get_zbc_clear_table_entry_v1A_0E',
    'struct_rpc_ctrl_get_zbc_clear_table_v1A_09',
    'struct_rpc_ctrl_gpfifo_get_work_submit_token_v1F_0A',
    'struct_rpc_ctrl_gpfifo_schedule_v1A_0A',
    'struct_rpc_ctrl_gpfifo_set_work_submit_token_notif_index_v1F_0A',
    'struct_rpc_ctrl_gpu_evict_ctx_v1A_1C',
    'struct_rpc_ctrl_gpu_get_info_v2_v25_11',
    'struct_rpc_ctrl_gpu_handle_vf_pri_fault_v1A_09',
    'struct_rpc_ctrl_gpu_initialize_ctx_v1A_0E',
    'struct_rpc_ctrl_gpu_migratable_ops_v21_07',
    'struct_rpc_ctrl_gpu_promote_ctx_v1A_20',
    'struct_rpc_ctrl_gpu_query_ecc_status_v24_06',
    'struct_rpc_ctrl_gpu_query_ecc_status_v26_02',
    'struct_rpc_ctrl_gr_ctxsw_preemption_bind_v1A_0E',
    'struct_rpc_ctrl_gr_ctxsw_preemption_bind_v28_07',
    'struct_rpc_ctrl_gr_ctxsw_zcull_bind_v1A_0E',
    'struct_rpc_ctrl_gr_get_tpc_partition_mode_v1C_04',
    'struct_rpc_ctrl_gr_pc_sampling_mode_v1A_1F',
    'struct_rpc_ctrl_gr_set_ctxsw_preemption_mode_v1A_0E',
    'struct_rpc_ctrl_gr_set_tpc_partition_mode_v1C_04',
    'struct_rpc_ctrl_grmgr_get_gr_fs_info_v1A_1D',
    'struct_rpc_ctrl_internal_memsys_set_zbc_referenced_v1F_05',
    'struct_rpc_ctrl_internal_promote_fault_method_buffers_v1E_07',
    'struct_rpc_ctrl_internal_quiesce_pma_channel_v1C_08',
    'struct_rpc_ctrl_internal_sriov_promote_pma_stream_v1C_0C',
    'struct_rpc_ctrl_master_get_virtual_function_error_cont_intr_mask_v1F_0D',
    'struct_rpc_ctrl_mc_service_interrupts_v1A_0E',
    'struct_rpc_ctrl_nvenc_sw_session_update_info_v1A_09',
    'struct_rpc_ctrl_nvlink_get_inband_received_data_v25_0C',
    'struct_rpc_ctrl_perf_boost_v1A_09',
    'struct_rpc_ctrl_perf_rated_tdp_get_status_v1A_1F',
    'struct_rpc_ctrl_perf_rated_tdp_set_control_v1A_1F',
    'struct_rpc_ctrl_pm_area_pc_sampler_v21_0B',
    'struct_rpc_ctrl_pma_stream_update_get_put_v1A_14',
    'struct_rpc_ctrl_preempt_v1A_0A',
    'struct_rpc_ctrl_release_ccu_prof_v29_07',
    'struct_rpc_ctrl_release_hes_v29_07',
    'struct_rpc_ctrl_reserve_ccu_prof_v29_07',
    'struct_rpc_ctrl_reserve_hes_v29_07',
    'struct_rpc_ctrl_reserve_hwpm_legacy_v1A_0F',
    'struct_rpc_ctrl_reserve_pm_area_smpc_v1A_0F',
    'struct_rpc_ctrl_reset_channel_v1A_09',
    'struct_rpc_ctrl_reset_isolated_channel_v1A_09',
    'struct_rpc_ctrl_set_channel_interleave_level_v1A_0A',
    'struct_rpc_ctrl_set_hs_credits_v21_08',
    'struct_rpc_ctrl_set_timeslice_v1A_0A',
    'struct_rpc_ctrl_set_tsg_interleave_level_v1A_0A',
    'struct_rpc_ctrl_set_vgpu_fb_usage_v1A_08',
    'struct_rpc_ctrl_set_zbc_color_clear_v1A_09',
    'struct_rpc_ctrl_set_zbc_depth_clear_v1A_09',
    'struct_rpc_ctrl_set_zbc_stencil_clear_v27_06',
    'struct_rpc_ctrl_stop_channel_v1A_1E',
    'struct_rpc_ctrl_subdevice_get_libos_heap_stats_v29_02',
    'struct_rpc_ctrl_subdevice_get_p2p_caps_v21_02',
    'struct_rpc_ctrl_subdevice_get_vgpu_heap_stats_v28_03',
    'struct_rpc_ctrl_subdevice_get_vgpu_heap_stats_v28_06',
    'struct_rpc_ctrl_timer_set_gr_tick_freq_v1A_1F',
    'struct_rpc_ctrl_vaspace_copy_server_reserved_pdes_v1E_04',
    'struct_rpc_dce_rm_init_v01_00',
    'struct_rpc_disable_channels_v1E_0B',
    'struct_rpc_display_modeset_v01_00',
    'struct_rpc_dump_protobuf_component_v18_12',
    'struct_rpc_dup_object_v03_00',
    'struct_rpc_extdev_intr_service_v17_00',
    'struct_rpc_fecs_error_v26_02', 'struct_rpc_free_v03_00',
    'struct_rpc_get_brand_caps_v25_12',
    'struct_rpc_get_consolidated_gr_static_info_v1B_04',
    'struct_rpc_get_encoder_capacity_v07_00',
    'struct_rpc_get_engine_utilization_v1F_0E',
    'struct_rpc_get_gsp_static_info_v14_00',
    'struct_rpc_get_static_data_v25_0E',
    'struct_rpc_get_static_data_v27_01',
    'struct_rpc_gpu_exec_reg_ops_v12_01',
    'struct_rpc_gpuacct_perfmon_util_samples_v1F_0E',
    'struct_rpc_gsp_lockdown_notice_v17_00',
    'struct_rpc_gsp_post_nocat_record_v01_00',
    'struct_rpc_gsp_rm_alloc_v03_00',
    'struct_rpc_gsp_rm_control_v03_00',
    'struct_rpc_gsp_set_system_info_v17_00',
    'struct_rpc_idle_channels_v03_00', 'struct_rpc_init_done_v17_00',
    'struct_rpc_invalidate_tlb_v23_03', 'struct_rpc_log_v03_00',
    'struct_rpc_map_memory_dma_v03_00',
    'struct_rpc_message_header_v03_00',
    'struct_rpc_nvlink_fatal_error_recovery_v17_00',
    'struct_rpc_nvlink_fault_up_v17_00',
    'struct_rpc_nvlink_inband_received_data_1024_v17_00',
    'struct_rpc_nvlink_inband_received_data_2048_v17_00',
    'struct_rpc_nvlink_inband_received_data_256_v17_00',
    'struct_rpc_nvlink_inband_received_data_4096_v17_00',
    'struct_rpc_nvlink_inband_received_data_512_v17_00',
    'struct_rpc_nvlink_is_gpu_degraded_v17_00',
    'struct_rpc_os_error_log_v17_00',
    'struct_rpc_perf_bridgeless_info_update_v17_00',
    'struct_rpc_perf_get_level_info_v03_00',
    'struct_rpc_perf_gpu_boost_sync_limits_callback_v17_00',
    'struct_rpc_pfm_req_hndlr_state_sync_callback_v21_04',
    'struct_rpc_post_event_v17_00', 'struct_rpc_rc_triggered_v17_02',
    'struct_rpc_recovery_action_v28_01',
    'struct_rpc_restore_hibernation_data_v1E_0E',
    'struct_rpc_rg_line_intr_v17_00',
    'struct_rpc_rm_api_control_v25_0D',
    'struct_rpc_rm_api_control_v25_0F',
    'struct_rpc_rm_api_control_v25_10',
    'struct_rpc_rm_api_control_v25_14',
    'struct_rpc_rm_api_control_v25_15',
    'struct_rpc_rm_api_control_v25_16',
    'struct_rpc_rm_api_control_v25_17',
    'struct_rpc_rm_api_control_v25_18',
    'struct_rpc_rm_api_control_v25_19',
    'struct_rpc_rm_api_control_v25_1A',
    'struct_rpc_rm_api_control_v27_03',
    'struct_rpc_rm_api_control_v29_04',
    'struct_rpc_rm_api_control_v29_09',
    'struct_rpc_run_cpu_sequencer_v17_00',
    'struct_rpc_save_hibernation_data_v1E_0E',
    'struct_rpc_semaphore_schedule_callback_v17_00',
    'struct_rpc_set_guest_system_info_ext_v15_02',
    'struct_rpc_set_guest_system_info_ext_v25_1B',
    'struct_rpc_set_guest_system_info_v03_00',
    'struct_rpc_set_page_directory_v1E_05',
    'struct_rpc_set_surface_properties_v07_07',
    'struct_rpc_sim_read_v1E_01', 'struct_rpc_sim_write_v1E_01',
    'struct_rpc_timed_semaphore_release_v01_00',
    'struct_rpc_ucode_libos_print_v1E_08',
    'struct_rpc_unloading_guest_driver_v1F_07',
    'struct_rpc_unmap_memory_dma_v03_00',
    'struct_rpc_unset_page_directory_v1E_05',
    'struct_rpc_update_bar_pde_v15_00',
    'struct_rpc_update_gpm_guest_buffer_info_v27_01',
    'struct_rpc_update_gsp_trace_v01_00',
    'struct_rpc_vgpu_config_event_v17_00',
    'struct_rpc_vgpu_gsp_mig_ci_config_v21_03',
    'struct_rpc_vgpu_gsp_plugin_triggered_v17_00',
    'struct_rpc_vgpu_pf_reg_read32_v15_00',
    'union_NV2080_CTRL_FB_FS_INFO_QUERY_DATA_v1A_1D',
    'union_NV2080_CTRL_FB_FS_INFO_QUERY_DATA_v26_04',
    'union_NV2080_CTRL_GRMGR_GR_FS_INFO_QUERY_DATA_v1A_1D',
    'union_NV2080_CTRL_INTERNAL_PFM_REQ_HNDLR_STATE_SYNC_DATA_type_v21_04',
    'union_alloc_object_params_v25_08',
    'union_alloc_object_params_v26_00',
    'union_alloc_object_params_v27_00',
    'union_alloc_object_params_v29_06', 'union_c__SA_GspFwWprMeta_0',
    'union_c__SA_GspFwWprMeta_1', 'union_pte_desc_0',
    'union_rpc_message_rpc_union_field_v03_00',
    'union_vgpuGetEngineUtilization_data_v1F_0E']
