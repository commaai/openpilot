# mypy: ignore-errors
import ctypes
from tinygrad.runtime.support.c import DLL, Struct, CEnum, _IO, _IOW, _IOR, _IOWR
import os
dll = DLL('hsa', [os.getenv('ROCM_PATH', '/opt/rocm')+'/lib/libhsa-runtime64.so', 'hsa-runtime64'])
enum_SQ_RSRC_BUF_TYPE = CEnum(ctypes.c_uint32)
SQ_RSRC_BUF = enum_SQ_RSRC_BUF_TYPE.define('SQ_RSRC_BUF', 0)
SQ_RSRC_BUF_RSVD_1 = enum_SQ_RSRC_BUF_TYPE.define('SQ_RSRC_BUF_RSVD_1', 1)
SQ_RSRC_BUF_RSVD_2 = enum_SQ_RSRC_BUF_TYPE.define('SQ_RSRC_BUF_RSVD_2', 2)
SQ_RSRC_BUF_RSVD_3 = enum_SQ_RSRC_BUF_TYPE.define('SQ_RSRC_BUF_RSVD_3', 3)

SQ_RSRC_BUF_TYPE = enum_SQ_RSRC_BUF_TYPE
enum_BUF_DATA_FORMAT = CEnum(ctypes.c_uint32)
BUF_DATA_FORMAT_INVALID = enum_BUF_DATA_FORMAT.define('BUF_DATA_FORMAT_INVALID', 0)
BUF_DATA_FORMAT_8 = enum_BUF_DATA_FORMAT.define('BUF_DATA_FORMAT_8', 1)
BUF_DATA_FORMAT_16 = enum_BUF_DATA_FORMAT.define('BUF_DATA_FORMAT_16', 2)
BUF_DATA_FORMAT_8_8 = enum_BUF_DATA_FORMAT.define('BUF_DATA_FORMAT_8_8', 3)
BUF_DATA_FORMAT_32 = enum_BUF_DATA_FORMAT.define('BUF_DATA_FORMAT_32', 4)
BUF_DATA_FORMAT_16_16 = enum_BUF_DATA_FORMAT.define('BUF_DATA_FORMAT_16_16', 5)
BUF_DATA_FORMAT_10_11_11 = enum_BUF_DATA_FORMAT.define('BUF_DATA_FORMAT_10_11_11', 6)
BUF_DATA_FORMAT_11_11_10 = enum_BUF_DATA_FORMAT.define('BUF_DATA_FORMAT_11_11_10', 7)
BUF_DATA_FORMAT_10_10_10_2 = enum_BUF_DATA_FORMAT.define('BUF_DATA_FORMAT_10_10_10_2', 8)
BUF_DATA_FORMAT_2_10_10_10 = enum_BUF_DATA_FORMAT.define('BUF_DATA_FORMAT_2_10_10_10', 9)
BUF_DATA_FORMAT_8_8_8_8 = enum_BUF_DATA_FORMAT.define('BUF_DATA_FORMAT_8_8_8_8', 10)
BUF_DATA_FORMAT_32_32 = enum_BUF_DATA_FORMAT.define('BUF_DATA_FORMAT_32_32', 11)
BUF_DATA_FORMAT_16_16_16_16 = enum_BUF_DATA_FORMAT.define('BUF_DATA_FORMAT_16_16_16_16', 12)
BUF_DATA_FORMAT_32_32_32 = enum_BUF_DATA_FORMAT.define('BUF_DATA_FORMAT_32_32_32', 13)
BUF_DATA_FORMAT_32_32_32_32 = enum_BUF_DATA_FORMAT.define('BUF_DATA_FORMAT_32_32_32_32', 14)
BUF_DATA_FORMAT_RESERVED_15 = enum_BUF_DATA_FORMAT.define('BUF_DATA_FORMAT_RESERVED_15', 15)

BUF_DATA_FORMAT = enum_BUF_DATA_FORMAT
enum_BUF_NUM_FORMAT = CEnum(ctypes.c_uint32)
BUF_NUM_FORMAT_UNORM = enum_BUF_NUM_FORMAT.define('BUF_NUM_FORMAT_UNORM', 0)
BUF_NUM_FORMAT_SNORM = enum_BUF_NUM_FORMAT.define('BUF_NUM_FORMAT_SNORM', 1)
BUF_NUM_FORMAT_USCALED = enum_BUF_NUM_FORMAT.define('BUF_NUM_FORMAT_USCALED', 2)
BUF_NUM_FORMAT_SSCALED = enum_BUF_NUM_FORMAT.define('BUF_NUM_FORMAT_SSCALED', 3)
BUF_NUM_FORMAT_UINT = enum_BUF_NUM_FORMAT.define('BUF_NUM_FORMAT_UINT', 4)
BUF_NUM_FORMAT_SINT = enum_BUF_NUM_FORMAT.define('BUF_NUM_FORMAT_SINT', 5)
BUF_NUM_FORMAT_SNORM_OGL__SI__CI = enum_BUF_NUM_FORMAT.define('BUF_NUM_FORMAT_SNORM_OGL__SI__CI', 6)
BUF_NUM_FORMAT_RESERVED_6__VI = enum_BUF_NUM_FORMAT.define('BUF_NUM_FORMAT_RESERVED_6__VI', 6)
BUF_NUM_FORMAT_FLOAT = enum_BUF_NUM_FORMAT.define('BUF_NUM_FORMAT_FLOAT', 7)

BUF_NUM_FORMAT = enum_BUF_NUM_FORMAT
enum_BUF_FORMAT = CEnum(ctypes.c_uint32)
BUF_FORMAT_32_UINT = enum_BUF_FORMAT.define('BUF_FORMAT_32_UINT', 20)

BUF_FORMAT = enum_BUF_FORMAT
enum_SQ_SEL_XYZW01 = CEnum(ctypes.c_uint32)
SQ_SEL_0 = enum_SQ_SEL_XYZW01.define('SQ_SEL_0', 0)
SQ_SEL_1 = enum_SQ_SEL_XYZW01.define('SQ_SEL_1', 1)
SQ_SEL_RESERVED_0 = enum_SQ_SEL_XYZW01.define('SQ_SEL_RESERVED_0', 2)
SQ_SEL_RESERVED_1 = enum_SQ_SEL_XYZW01.define('SQ_SEL_RESERVED_1', 3)
SQ_SEL_X = enum_SQ_SEL_XYZW01.define('SQ_SEL_X', 4)
SQ_SEL_Y = enum_SQ_SEL_XYZW01.define('SQ_SEL_Y', 5)
SQ_SEL_Z = enum_SQ_SEL_XYZW01.define('SQ_SEL_Z', 6)
SQ_SEL_W = enum_SQ_SEL_XYZW01.define('SQ_SEL_W', 7)

SQ_SEL_XYZW01 = enum_SQ_SEL_XYZW01
class union_COMPUTE_TMPRING_SIZE(ctypes.Union): pass
class union_COMPUTE_TMPRING_SIZE_bitfields(Struct): pass
union_COMPUTE_TMPRING_SIZE_bitfields._fields_ = [
  ('WAVES', ctypes.c_uint32,12),
  ('WAVESIZE', ctypes.c_uint32,13),
  ('', ctypes.c_uint32,7),
]
union_COMPUTE_TMPRING_SIZE._fields_ = [
  ('bitfields', union_COMPUTE_TMPRING_SIZE_bitfields),
  ('bits', union_COMPUTE_TMPRING_SIZE_bitfields),
  ('u32All', ctypes.c_uint32),
  ('i32All', ctypes.c_int32),
  ('f32All', ctypes.c_float),
]
class union_COMPUTE_TMPRING_SIZE_GFX11(ctypes.Union): pass
class union_COMPUTE_TMPRING_SIZE_GFX11_bitfields(Struct): pass
union_COMPUTE_TMPRING_SIZE_GFX11_bitfields._fields_ = [
  ('WAVES', ctypes.c_uint32,12),
  ('WAVESIZE', ctypes.c_uint32,15),
  ('', ctypes.c_uint32,5),
]
union_COMPUTE_TMPRING_SIZE_GFX11._fields_ = [
  ('bitfields', union_COMPUTE_TMPRING_SIZE_GFX11_bitfields),
  ('bits', union_COMPUTE_TMPRING_SIZE_GFX11_bitfields),
  ('u32All', ctypes.c_uint32),
  ('i32All', ctypes.c_int32),
  ('f32All', ctypes.c_float),
]
class union_COMPUTE_TMPRING_SIZE_GFX12(ctypes.Union): pass
class union_COMPUTE_TMPRING_SIZE_GFX12_bitfields(Struct): pass
union_COMPUTE_TMPRING_SIZE_GFX12_bitfields._fields_ = [
  ('WAVES', ctypes.c_uint32,12),
  ('WAVESIZE', ctypes.c_uint32,18),
  ('', ctypes.c_uint32,2),
]
union_COMPUTE_TMPRING_SIZE_GFX12._fields_ = [
  ('bitfields', union_COMPUTE_TMPRING_SIZE_GFX12_bitfields),
  ('bits', union_COMPUTE_TMPRING_SIZE_GFX12_bitfields),
  ('u32All', ctypes.c_uint32),
  ('i32All', ctypes.c_int32),
  ('f32All', ctypes.c_float),
]
class union_SQ_BUF_RSRC_WORD0(ctypes.Union): pass
class union_SQ_BUF_RSRC_WORD0_bitfields(Struct): pass
union_SQ_BUF_RSRC_WORD0_bitfields._fields_ = [
  ('BASE_ADDRESS', ctypes.c_uint32,32),
]
union_SQ_BUF_RSRC_WORD0._fields_ = [
  ('bitfields', union_SQ_BUF_RSRC_WORD0_bitfields),
  ('bits', union_SQ_BUF_RSRC_WORD0_bitfields),
  ('u32All', ctypes.c_uint32),
  ('i32All', ctypes.c_int32),
  ('f32All', ctypes.c_float),
]
class union_SQ_BUF_RSRC_WORD1(ctypes.Union): pass
class union_SQ_BUF_RSRC_WORD1_bitfields(Struct): pass
union_SQ_BUF_RSRC_WORD1_bitfields._fields_ = [
  ('BASE_ADDRESS_HI', ctypes.c_uint32,16),
  ('STRIDE', ctypes.c_uint32,14),
  ('CACHE_SWIZZLE', ctypes.c_uint32,1),
  ('SWIZZLE_ENABLE', ctypes.c_uint32,1),
]
union_SQ_BUF_RSRC_WORD1._fields_ = [
  ('bitfields', union_SQ_BUF_RSRC_WORD1_bitfields),
  ('bits', union_SQ_BUF_RSRC_WORD1_bitfields),
  ('u32All', ctypes.c_uint32),
  ('i32All', ctypes.c_int32),
  ('f32All', ctypes.c_float),
]
class union_SQ_BUF_RSRC_WORD1_GFX11(ctypes.Union): pass
class union_SQ_BUF_RSRC_WORD1_GFX11_bitfields(Struct): pass
union_SQ_BUF_RSRC_WORD1_GFX11_bitfields._fields_ = [
  ('BASE_ADDRESS_HI', ctypes.c_uint32,16),
  ('STRIDE', ctypes.c_uint32,14),
  ('SWIZZLE_ENABLE', ctypes.c_uint32,2),
]
union_SQ_BUF_RSRC_WORD1_GFX11._fields_ = [
  ('bitfields', union_SQ_BUF_RSRC_WORD1_GFX11_bitfields),
  ('bits', union_SQ_BUF_RSRC_WORD1_GFX11_bitfields),
  ('u32All', ctypes.c_uint32),
  ('i32All', ctypes.c_int32),
  ('f32All', ctypes.c_float),
]
class union_SQ_BUF_RSRC_WORD2(ctypes.Union): pass
class union_SQ_BUF_RSRC_WORD2_bitfields(Struct): pass
union_SQ_BUF_RSRC_WORD2_bitfields._fields_ = [
  ('NUM_RECORDS', ctypes.c_uint32,32),
]
union_SQ_BUF_RSRC_WORD2._fields_ = [
  ('bitfields', union_SQ_BUF_RSRC_WORD2_bitfields),
  ('bits', union_SQ_BUF_RSRC_WORD2_bitfields),
  ('u32All', ctypes.c_uint32),
  ('i32All', ctypes.c_int32),
  ('f32All', ctypes.c_float),
]
class union_SQ_BUF_RSRC_WORD3(ctypes.Union): pass
class union_SQ_BUF_RSRC_WORD3_bitfields(Struct): pass
union_SQ_BUF_RSRC_WORD3_bitfields._fields_ = [
  ('DST_SEL_X', ctypes.c_uint32,3),
  ('DST_SEL_Y', ctypes.c_uint32,3),
  ('DST_SEL_Z', ctypes.c_uint32,3),
  ('DST_SEL_W', ctypes.c_uint32,3),
  ('NUM_FORMAT', ctypes.c_uint32,3),
  ('DATA_FORMAT', ctypes.c_uint32,4),
  ('ELEMENT_SIZE', ctypes.c_uint32,2),
  ('INDEX_STRIDE', ctypes.c_uint32,2),
  ('ADD_TID_ENABLE', ctypes.c_uint32,1),
  ('ATC__CI__VI', ctypes.c_uint32,1),
  ('HASH_ENABLE', ctypes.c_uint32,1),
  ('HEAP', ctypes.c_uint32,1),
  ('MTYPE__CI__VI', ctypes.c_uint32,3),
  ('TYPE', ctypes.c_uint32,2),
]
union_SQ_BUF_RSRC_WORD3._fields_ = [
  ('bitfields', union_SQ_BUF_RSRC_WORD3_bitfields),
  ('bits', union_SQ_BUF_RSRC_WORD3_bitfields),
  ('u32All', ctypes.c_uint32),
  ('i32All', ctypes.c_int32),
  ('f32All', ctypes.c_float),
]
class union_SQ_BUF_RSRC_WORD3_GFX10(ctypes.Union): pass
class union_SQ_BUF_RSRC_WORD3_GFX10_bitfields(Struct): pass
union_SQ_BUF_RSRC_WORD3_GFX10_bitfields._fields_ = [
  ('DST_SEL_X', ctypes.c_uint32,3),
  ('DST_SEL_Y', ctypes.c_uint32,3),
  ('DST_SEL_Z', ctypes.c_uint32,3),
  ('DST_SEL_W', ctypes.c_uint32,3),
  ('FORMAT', ctypes.c_uint32,7),
  ('RESERVED1', ctypes.c_uint32,2),
  ('INDEX_STRIDE', ctypes.c_uint32,2),
  ('ADD_TID_ENABLE', ctypes.c_uint32,1),
  ('RESOURCE_LEVEL', ctypes.c_uint32,1),
  ('RESERVED2', ctypes.c_uint32,3),
  ('OOB_SELECT', ctypes.c_uint32,2),
  ('TYPE', ctypes.c_uint32,2),
]
union_SQ_BUF_RSRC_WORD3_GFX10._fields_ = [
  ('bitfields', union_SQ_BUF_RSRC_WORD3_GFX10_bitfields),
  ('bits', union_SQ_BUF_RSRC_WORD3_GFX10_bitfields),
  ('u32All', ctypes.c_uint32),
  ('i32All', ctypes.c_int32),
  ('f32All', ctypes.c_float),
]
class union_SQ_BUF_RSRC_WORD3_GFX11(ctypes.Union): pass
class union_SQ_BUF_RSRC_WORD3_GFX11_bitfields(Struct): pass
union_SQ_BUF_RSRC_WORD3_GFX11_bitfields._fields_ = [
  ('DST_SEL_X', ctypes.c_uint32,3),
  ('DST_SEL_Y', ctypes.c_uint32,3),
  ('DST_SEL_Z', ctypes.c_uint32,3),
  ('DST_SEL_W', ctypes.c_uint32,3),
  ('FORMAT', ctypes.c_uint32,6),
  ('RESERVED1', ctypes.c_uint32,3),
  ('INDEX_STRIDE', ctypes.c_uint32,2),
  ('ADD_TID_ENABLE', ctypes.c_uint32,1),
  ('RESERVED2', ctypes.c_uint32,4),
  ('OOB_SELECT', ctypes.c_uint32,2),
  ('TYPE', ctypes.c_uint32,2),
]
union_SQ_BUF_RSRC_WORD3_GFX11._fields_ = [
  ('bitfields', union_SQ_BUF_RSRC_WORD3_GFX11_bitfields),
  ('bits', union_SQ_BUF_RSRC_WORD3_GFX11_bitfields),
  ('u32All', ctypes.c_uint32),
  ('i32All', ctypes.c_int32),
  ('f32All', ctypes.c_float),
]
class union_SQ_BUF_RSRC_WORD3_GFX12(ctypes.Union): pass
class union_SQ_BUF_RSRC_WORD3_GFX12_bitfields(Struct): pass
union_SQ_BUF_RSRC_WORD3_GFX12_bitfields._fields_ = [
  ('DST_SEL_X', ctypes.c_uint32,3),
  ('DST_SEL_Y', ctypes.c_uint32,3),
  ('DST_SEL_Z', ctypes.c_uint32,3),
  ('DST_SEL_W', ctypes.c_uint32,3),
  ('FORMAT', ctypes.c_uint32,6),
  ('RESERVED1', ctypes.c_uint32,3),
  ('INDEX_STRIDE', ctypes.c_uint32,2),
  ('ADD_TID_ENABLE', ctypes.c_uint32,1),
  ('WRITE_COMPRESS_ENABLE', ctypes.c_uint32,1),
  ('COMPRESSION_EN', ctypes.c_uint32,1),
  ('COMPRESSION_ACCESS_MODE', ctypes.c_uint32,2),
  ('OOB_SELECT', ctypes.c_uint32,2),
  ('TYPE', ctypes.c_uint32,2),
]
union_SQ_BUF_RSRC_WORD3_GFX12._fields_ = [
  ('bitfields', union_SQ_BUF_RSRC_WORD3_GFX12_bitfields),
  ('bits', union_SQ_BUF_RSRC_WORD3_GFX12_bitfields),
  ('u32All', ctypes.c_uint32),
  ('i32All', ctypes.c_int32),
  ('f32All', ctypes.c_float),
]
hsa_status_t = CEnum(ctypes.c_uint32)
HSA_STATUS_SUCCESS = hsa_status_t.define('HSA_STATUS_SUCCESS', 0)
HSA_STATUS_INFO_BREAK = hsa_status_t.define('HSA_STATUS_INFO_BREAK', 1)
HSA_STATUS_ERROR = hsa_status_t.define('HSA_STATUS_ERROR', 4096)
HSA_STATUS_ERROR_INVALID_ARGUMENT = hsa_status_t.define('HSA_STATUS_ERROR_INVALID_ARGUMENT', 4097)
HSA_STATUS_ERROR_INVALID_QUEUE_CREATION = hsa_status_t.define('HSA_STATUS_ERROR_INVALID_QUEUE_CREATION', 4098)
HSA_STATUS_ERROR_INVALID_ALLOCATION = hsa_status_t.define('HSA_STATUS_ERROR_INVALID_ALLOCATION', 4099)
HSA_STATUS_ERROR_INVALID_AGENT = hsa_status_t.define('HSA_STATUS_ERROR_INVALID_AGENT', 4100)
HSA_STATUS_ERROR_INVALID_REGION = hsa_status_t.define('HSA_STATUS_ERROR_INVALID_REGION', 4101)
HSA_STATUS_ERROR_INVALID_SIGNAL = hsa_status_t.define('HSA_STATUS_ERROR_INVALID_SIGNAL', 4102)
HSA_STATUS_ERROR_INVALID_QUEUE = hsa_status_t.define('HSA_STATUS_ERROR_INVALID_QUEUE', 4103)
HSA_STATUS_ERROR_OUT_OF_RESOURCES = hsa_status_t.define('HSA_STATUS_ERROR_OUT_OF_RESOURCES', 4104)
HSA_STATUS_ERROR_INVALID_PACKET_FORMAT = hsa_status_t.define('HSA_STATUS_ERROR_INVALID_PACKET_FORMAT', 4105)
HSA_STATUS_ERROR_RESOURCE_FREE = hsa_status_t.define('HSA_STATUS_ERROR_RESOURCE_FREE', 4106)
HSA_STATUS_ERROR_NOT_INITIALIZED = hsa_status_t.define('HSA_STATUS_ERROR_NOT_INITIALIZED', 4107)
HSA_STATUS_ERROR_REFCOUNT_OVERFLOW = hsa_status_t.define('HSA_STATUS_ERROR_REFCOUNT_OVERFLOW', 4108)
HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS = hsa_status_t.define('HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS', 4109)
HSA_STATUS_ERROR_INVALID_INDEX = hsa_status_t.define('HSA_STATUS_ERROR_INVALID_INDEX', 4110)
HSA_STATUS_ERROR_INVALID_ISA = hsa_status_t.define('HSA_STATUS_ERROR_INVALID_ISA', 4111)
HSA_STATUS_ERROR_INVALID_ISA_NAME = hsa_status_t.define('HSA_STATUS_ERROR_INVALID_ISA_NAME', 4119)
HSA_STATUS_ERROR_INVALID_CODE_OBJECT = hsa_status_t.define('HSA_STATUS_ERROR_INVALID_CODE_OBJECT', 4112)
HSA_STATUS_ERROR_INVALID_EXECUTABLE = hsa_status_t.define('HSA_STATUS_ERROR_INVALID_EXECUTABLE', 4113)
HSA_STATUS_ERROR_FROZEN_EXECUTABLE = hsa_status_t.define('HSA_STATUS_ERROR_FROZEN_EXECUTABLE', 4114)
HSA_STATUS_ERROR_INVALID_SYMBOL_NAME = hsa_status_t.define('HSA_STATUS_ERROR_INVALID_SYMBOL_NAME', 4115)
HSA_STATUS_ERROR_VARIABLE_ALREADY_DEFINED = hsa_status_t.define('HSA_STATUS_ERROR_VARIABLE_ALREADY_DEFINED', 4116)
HSA_STATUS_ERROR_VARIABLE_UNDEFINED = hsa_status_t.define('HSA_STATUS_ERROR_VARIABLE_UNDEFINED', 4117)
HSA_STATUS_ERROR_EXCEPTION = hsa_status_t.define('HSA_STATUS_ERROR_EXCEPTION', 4118)
HSA_STATUS_ERROR_INVALID_CODE_SYMBOL = hsa_status_t.define('HSA_STATUS_ERROR_INVALID_CODE_SYMBOL', 4120)
HSA_STATUS_ERROR_INVALID_EXECUTABLE_SYMBOL = hsa_status_t.define('HSA_STATUS_ERROR_INVALID_EXECUTABLE_SYMBOL', 4121)
HSA_STATUS_ERROR_INVALID_FILE = hsa_status_t.define('HSA_STATUS_ERROR_INVALID_FILE', 4128)
HSA_STATUS_ERROR_INVALID_CODE_OBJECT_READER = hsa_status_t.define('HSA_STATUS_ERROR_INVALID_CODE_OBJECT_READER', 4129)
HSA_STATUS_ERROR_INVALID_CACHE = hsa_status_t.define('HSA_STATUS_ERROR_INVALID_CACHE', 4130)
HSA_STATUS_ERROR_INVALID_WAVEFRONT = hsa_status_t.define('HSA_STATUS_ERROR_INVALID_WAVEFRONT', 4131)
HSA_STATUS_ERROR_INVALID_SIGNAL_GROUP = hsa_status_t.define('HSA_STATUS_ERROR_INVALID_SIGNAL_GROUP', 4132)
HSA_STATUS_ERROR_INVALID_RUNTIME_STATE = hsa_status_t.define('HSA_STATUS_ERROR_INVALID_RUNTIME_STATE', 4133)
HSA_STATUS_ERROR_FATAL = hsa_status_t.define('HSA_STATUS_ERROR_FATAL', 4134)

try: (hsa_status_string:=dll.hsa_status_string).restype, hsa_status_string.argtypes = hsa_status_t, [hsa_status_t, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

class struct_hsa_dim3_s(Struct): pass
uint32_t = ctypes.c_uint32
struct_hsa_dim3_s._fields_ = [
  ('x', uint32_t),
  ('y', uint32_t),
  ('z', uint32_t),
]
hsa_dim3_t = struct_hsa_dim3_s
hsa_access_permission_t = CEnum(ctypes.c_uint32)
HSA_ACCESS_PERMISSION_NONE = hsa_access_permission_t.define('HSA_ACCESS_PERMISSION_NONE', 0)
HSA_ACCESS_PERMISSION_RO = hsa_access_permission_t.define('HSA_ACCESS_PERMISSION_RO', 1)
HSA_ACCESS_PERMISSION_WO = hsa_access_permission_t.define('HSA_ACCESS_PERMISSION_WO', 2)
HSA_ACCESS_PERMISSION_RW = hsa_access_permission_t.define('HSA_ACCESS_PERMISSION_RW', 3)

hsa_file_t = ctypes.c_int32
try: (hsa_init:=dll.hsa_init).restype, hsa_init.argtypes = hsa_status_t, []
except AttributeError: pass

try: (hsa_shut_down:=dll.hsa_shut_down).restype, hsa_shut_down.argtypes = hsa_status_t, []
except AttributeError: pass

hsa_endianness_t = CEnum(ctypes.c_uint32)
HSA_ENDIANNESS_LITTLE = hsa_endianness_t.define('HSA_ENDIANNESS_LITTLE', 0)
HSA_ENDIANNESS_BIG = hsa_endianness_t.define('HSA_ENDIANNESS_BIG', 1)

hsa_machine_model_t = CEnum(ctypes.c_uint32)
HSA_MACHINE_MODEL_SMALL = hsa_machine_model_t.define('HSA_MACHINE_MODEL_SMALL', 0)
HSA_MACHINE_MODEL_LARGE = hsa_machine_model_t.define('HSA_MACHINE_MODEL_LARGE', 1)

hsa_profile_t = CEnum(ctypes.c_uint32)
HSA_PROFILE_BASE = hsa_profile_t.define('HSA_PROFILE_BASE', 0)
HSA_PROFILE_FULL = hsa_profile_t.define('HSA_PROFILE_FULL', 1)

hsa_system_info_t = CEnum(ctypes.c_uint32)
HSA_SYSTEM_INFO_VERSION_MAJOR = hsa_system_info_t.define('HSA_SYSTEM_INFO_VERSION_MAJOR', 0)
HSA_SYSTEM_INFO_VERSION_MINOR = hsa_system_info_t.define('HSA_SYSTEM_INFO_VERSION_MINOR', 1)
HSA_SYSTEM_INFO_TIMESTAMP = hsa_system_info_t.define('HSA_SYSTEM_INFO_TIMESTAMP', 2)
HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY = hsa_system_info_t.define('HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY', 3)
HSA_SYSTEM_INFO_SIGNAL_MAX_WAIT = hsa_system_info_t.define('HSA_SYSTEM_INFO_SIGNAL_MAX_WAIT', 4)
HSA_SYSTEM_INFO_ENDIANNESS = hsa_system_info_t.define('HSA_SYSTEM_INFO_ENDIANNESS', 5)
HSA_SYSTEM_INFO_MACHINE_MODEL = hsa_system_info_t.define('HSA_SYSTEM_INFO_MACHINE_MODEL', 6)
HSA_SYSTEM_INFO_EXTENSIONS = hsa_system_info_t.define('HSA_SYSTEM_INFO_EXTENSIONS', 7)
HSA_AMD_SYSTEM_INFO_BUILD_VERSION = hsa_system_info_t.define('HSA_AMD_SYSTEM_INFO_BUILD_VERSION', 512)
HSA_AMD_SYSTEM_INFO_SVM_SUPPORTED = hsa_system_info_t.define('HSA_AMD_SYSTEM_INFO_SVM_SUPPORTED', 513)
HSA_AMD_SYSTEM_INFO_SVM_ACCESSIBLE_BY_DEFAULT = hsa_system_info_t.define('HSA_AMD_SYSTEM_INFO_SVM_ACCESSIBLE_BY_DEFAULT', 514)
HSA_AMD_SYSTEM_INFO_MWAITX_ENABLED = hsa_system_info_t.define('HSA_AMD_SYSTEM_INFO_MWAITX_ENABLED', 515)
HSA_AMD_SYSTEM_INFO_DMABUF_SUPPORTED = hsa_system_info_t.define('HSA_AMD_SYSTEM_INFO_DMABUF_SUPPORTED', 516)
HSA_AMD_SYSTEM_INFO_VIRTUAL_MEM_API_SUPPORTED = hsa_system_info_t.define('HSA_AMD_SYSTEM_INFO_VIRTUAL_MEM_API_SUPPORTED', 517)
HSA_AMD_SYSTEM_INFO_XNACK_ENABLED = hsa_system_info_t.define('HSA_AMD_SYSTEM_INFO_XNACK_ENABLED', 518)
HSA_AMD_SYSTEM_INFO_EXT_VERSION_MAJOR = hsa_system_info_t.define('HSA_AMD_SYSTEM_INFO_EXT_VERSION_MAJOR', 519)
HSA_AMD_SYSTEM_INFO_EXT_VERSION_MINOR = hsa_system_info_t.define('HSA_AMD_SYSTEM_INFO_EXT_VERSION_MINOR', 520)

try: (hsa_system_get_info:=dll.hsa_system_get_info).restype, hsa_system_get_info.argtypes = hsa_status_t, [hsa_system_info_t, ctypes.c_void_p]
except AttributeError: pass

hsa_extension_t = CEnum(ctypes.c_uint32)
HSA_EXTENSION_FINALIZER = hsa_extension_t.define('HSA_EXTENSION_FINALIZER', 0)
HSA_EXTENSION_IMAGES = hsa_extension_t.define('HSA_EXTENSION_IMAGES', 1)
HSA_EXTENSION_PERFORMANCE_COUNTERS = hsa_extension_t.define('HSA_EXTENSION_PERFORMANCE_COUNTERS', 2)
HSA_EXTENSION_PROFILING_EVENTS = hsa_extension_t.define('HSA_EXTENSION_PROFILING_EVENTS', 3)
HSA_EXTENSION_STD_LAST = hsa_extension_t.define('HSA_EXTENSION_STD_LAST', 3)
HSA_AMD_FIRST_EXTENSION = hsa_extension_t.define('HSA_AMD_FIRST_EXTENSION', 512)
HSA_EXTENSION_AMD_PROFILER = hsa_extension_t.define('HSA_EXTENSION_AMD_PROFILER', 512)
HSA_EXTENSION_AMD_LOADER = hsa_extension_t.define('HSA_EXTENSION_AMD_LOADER', 513)
HSA_EXTENSION_AMD_AQLPROFILE = hsa_extension_t.define('HSA_EXTENSION_AMD_AQLPROFILE', 514)
HSA_EXTENSION_AMD_PC_SAMPLING = hsa_extension_t.define('HSA_EXTENSION_AMD_PC_SAMPLING', 515)
HSA_AMD_LAST_EXTENSION = hsa_extension_t.define('HSA_AMD_LAST_EXTENSION', 515)

uint16_t = ctypes.c_uint16
try: (hsa_extension_get_name:=dll.hsa_extension_get_name).restype, hsa_extension_get_name.argtypes = hsa_status_t, [uint16_t, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

try: (hsa_system_extension_supported:=dll.hsa_system_extension_supported).restype, hsa_system_extension_supported.argtypes = hsa_status_t, [uint16_t, uint16_t, uint16_t, ctypes.POINTER(ctypes.c_bool)]
except AttributeError: pass

try: (hsa_system_major_extension_supported:=dll.hsa_system_major_extension_supported).restype, hsa_system_major_extension_supported.argtypes = hsa_status_t, [uint16_t, uint16_t, ctypes.POINTER(uint16_t), ctypes.POINTER(ctypes.c_bool)]
except AttributeError: pass

try: (hsa_system_get_extension_table:=dll.hsa_system_get_extension_table).restype, hsa_system_get_extension_table.argtypes = hsa_status_t, [uint16_t, uint16_t, uint16_t, ctypes.c_void_p]
except AttributeError: pass

size_t = ctypes.c_uint64
try: (hsa_system_get_major_extension_table:=dll.hsa_system_get_major_extension_table).restype, hsa_system_get_major_extension_table.argtypes = hsa_status_t, [uint16_t, uint16_t, size_t, ctypes.c_void_p]
except AttributeError: pass

class struct_hsa_agent_s(Struct): pass
uint64_t = ctypes.c_uint64
struct_hsa_agent_s._fields_ = [
  ('handle', uint64_t),
]
hsa_agent_t = struct_hsa_agent_s
hsa_agent_feature_t = CEnum(ctypes.c_uint32)
HSA_AGENT_FEATURE_KERNEL_DISPATCH = hsa_agent_feature_t.define('HSA_AGENT_FEATURE_KERNEL_DISPATCH', 1)
HSA_AGENT_FEATURE_AGENT_DISPATCH = hsa_agent_feature_t.define('HSA_AGENT_FEATURE_AGENT_DISPATCH', 2)

hsa_device_type_t = CEnum(ctypes.c_uint32)
HSA_DEVICE_TYPE_CPU = hsa_device_type_t.define('HSA_DEVICE_TYPE_CPU', 0)
HSA_DEVICE_TYPE_GPU = hsa_device_type_t.define('HSA_DEVICE_TYPE_GPU', 1)
HSA_DEVICE_TYPE_DSP = hsa_device_type_t.define('HSA_DEVICE_TYPE_DSP', 2)
HSA_DEVICE_TYPE_AIE = hsa_device_type_t.define('HSA_DEVICE_TYPE_AIE', 3)

hsa_default_float_rounding_mode_t = CEnum(ctypes.c_uint32)
HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT = hsa_default_float_rounding_mode_t.define('HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT', 0)
HSA_DEFAULT_FLOAT_ROUNDING_MODE_ZERO = hsa_default_float_rounding_mode_t.define('HSA_DEFAULT_FLOAT_ROUNDING_MODE_ZERO', 1)
HSA_DEFAULT_FLOAT_ROUNDING_MODE_NEAR = hsa_default_float_rounding_mode_t.define('HSA_DEFAULT_FLOAT_ROUNDING_MODE_NEAR', 2)

hsa_agent_info_t = CEnum(ctypes.c_uint32)
HSA_AGENT_INFO_NAME = hsa_agent_info_t.define('HSA_AGENT_INFO_NAME', 0)
HSA_AGENT_INFO_VENDOR_NAME = hsa_agent_info_t.define('HSA_AGENT_INFO_VENDOR_NAME', 1)
HSA_AGENT_INFO_FEATURE = hsa_agent_info_t.define('HSA_AGENT_INFO_FEATURE', 2)
HSA_AGENT_INFO_MACHINE_MODEL = hsa_agent_info_t.define('HSA_AGENT_INFO_MACHINE_MODEL', 3)
HSA_AGENT_INFO_PROFILE = hsa_agent_info_t.define('HSA_AGENT_INFO_PROFILE', 4)
HSA_AGENT_INFO_DEFAULT_FLOAT_ROUNDING_MODE = hsa_agent_info_t.define('HSA_AGENT_INFO_DEFAULT_FLOAT_ROUNDING_MODE', 5)
HSA_AGENT_INFO_BASE_PROFILE_DEFAULT_FLOAT_ROUNDING_MODES = hsa_agent_info_t.define('HSA_AGENT_INFO_BASE_PROFILE_DEFAULT_FLOAT_ROUNDING_MODES', 23)
HSA_AGENT_INFO_FAST_F16_OPERATION = hsa_agent_info_t.define('HSA_AGENT_INFO_FAST_F16_OPERATION', 24)
HSA_AGENT_INFO_WAVEFRONT_SIZE = hsa_agent_info_t.define('HSA_AGENT_INFO_WAVEFRONT_SIZE', 6)
HSA_AGENT_INFO_WORKGROUP_MAX_DIM = hsa_agent_info_t.define('HSA_AGENT_INFO_WORKGROUP_MAX_DIM', 7)
HSA_AGENT_INFO_WORKGROUP_MAX_SIZE = hsa_agent_info_t.define('HSA_AGENT_INFO_WORKGROUP_MAX_SIZE', 8)
HSA_AGENT_INFO_GRID_MAX_DIM = hsa_agent_info_t.define('HSA_AGENT_INFO_GRID_MAX_DIM', 9)
HSA_AGENT_INFO_GRID_MAX_SIZE = hsa_agent_info_t.define('HSA_AGENT_INFO_GRID_MAX_SIZE', 10)
HSA_AGENT_INFO_FBARRIER_MAX_SIZE = hsa_agent_info_t.define('HSA_AGENT_INFO_FBARRIER_MAX_SIZE', 11)
HSA_AGENT_INFO_QUEUES_MAX = hsa_agent_info_t.define('HSA_AGENT_INFO_QUEUES_MAX', 12)
HSA_AGENT_INFO_QUEUE_MIN_SIZE = hsa_agent_info_t.define('HSA_AGENT_INFO_QUEUE_MIN_SIZE', 13)
HSA_AGENT_INFO_QUEUE_MAX_SIZE = hsa_agent_info_t.define('HSA_AGENT_INFO_QUEUE_MAX_SIZE', 14)
HSA_AGENT_INFO_QUEUE_TYPE = hsa_agent_info_t.define('HSA_AGENT_INFO_QUEUE_TYPE', 15)
HSA_AGENT_INFO_NODE = hsa_agent_info_t.define('HSA_AGENT_INFO_NODE', 16)
HSA_AGENT_INFO_DEVICE = hsa_agent_info_t.define('HSA_AGENT_INFO_DEVICE', 17)
HSA_AGENT_INFO_CACHE_SIZE = hsa_agent_info_t.define('HSA_AGENT_INFO_CACHE_SIZE', 18)
HSA_AGENT_INFO_ISA = hsa_agent_info_t.define('HSA_AGENT_INFO_ISA', 19)
HSA_AGENT_INFO_EXTENSIONS = hsa_agent_info_t.define('HSA_AGENT_INFO_EXTENSIONS', 20)
HSA_AGENT_INFO_VERSION_MAJOR = hsa_agent_info_t.define('HSA_AGENT_INFO_VERSION_MAJOR', 21)
HSA_AGENT_INFO_VERSION_MINOR = hsa_agent_info_t.define('HSA_AGENT_INFO_VERSION_MINOR', 22)
HSA_AGENT_INFO_LAST = hsa_agent_info_t.define('HSA_AGENT_INFO_LAST', 2147483647)

try: (hsa_agent_get_info:=dll.hsa_agent_get_info).restype, hsa_agent_get_info.argtypes = hsa_status_t, [hsa_agent_t, hsa_agent_info_t, ctypes.c_void_p]
except AttributeError: pass

try: (hsa_iterate_agents:=dll.hsa_iterate_agents).restype, hsa_iterate_agents.argtypes = hsa_status_t, [ctypes.CFUNCTYPE(hsa_status_t, hsa_agent_t, ctypes.c_void_p), ctypes.c_void_p]
except AttributeError: pass

hsa_exception_policy_t = CEnum(ctypes.c_uint32)
HSA_EXCEPTION_POLICY_BREAK = hsa_exception_policy_t.define('HSA_EXCEPTION_POLICY_BREAK', 1)
HSA_EXCEPTION_POLICY_DETECT = hsa_exception_policy_t.define('HSA_EXCEPTION_POLICY_DETECT', 2)

try: (hsa_agent_get_exception_policies:=dll.hsa_agent_get_exception_policies).restype, hsa_agent_get_exception_policies.argtypes = hsa_status_t, [hsa_agent_t, hsa_profile_t, ctypes.POINTER(uint16_t)]
except AttributeError: pass

class struct_hsa_cache_s(Struct): pass
struct_hsa_cache_s._fields_ = [
  ('handle', uint64_t),
]
hsa_cache_t = struct_hsa_cache_s
hsa_cache_info_t = CEnum(ctypes.c_uint32)
HSA_CACHE_INFO_NAME_LENGTH = hsa_cache_info_t.define('HSA_CACHE_INFO_NAME_LENGTH', 0)
HSA_CACHE_INFO_NAME = hsa_cache_info_t.define('HSA_CACHE_INFO_NAME', 1)
HSA_CACHE_INFO_LEVEL = hsa_cache_info_t.define('HSA_CACHE_INFO_LEVEL', 2)
HSA_CACHE_INFO_SIZE = hsa_cache_info_t.define('HSA_CACHE_INFO_SIZE', 3)

try: (hsa_cache_get_info:=dll.hsa_cache_get_info).restype, hsa_cache_get_info.argtypes = hsa_status_t, [hsa_cache_t, hsa_cache_info_t, ctypes.c_void_p]
except AttributeError: pass

try: (hsa_agent_iterate_caches:=dll.hsa_agent_iterate_caches).restype, hsa_agent_iterate_caches.argtypes = hsa_status_t, [hsa_agent_t, ctypes.CFUNCTYPE(hsa_status_t, hsa_cache_t, ctypes.c_void_p), ctypes.c_void_p]
except AttributeError: pass

try: (hsa_agent_extension_supported:=dll.hsa_agent_extension_supported).restype, hsa_agent_extension_supported.argtypes = hsa_status_t, [uint16_t, hsa_agent_t, uint16_t, uint16_t, ctypes.POINTER(ctypes.c_bool)]
except AttributeError: pass

try: (hsa_agent_major_extension_supported:=dll.hsa_agent_major_extension_supported).restype, hsa_agent_major_extension_supported.argtypes = hsa_status_t, [uint16_t, hsa_agent_t, uint16_t, ctypes.POINTER(uint16_t), ctypes.POINTER(ctypes.c_bool)]
except AttributeError: pass

class struct_hsa_signal_s(Struct): pass
struct_hsa_signal_s._fields_ = [
  ('handle', uint64_t),
]
hsa_signal_t = struct_hsa_signal_s
hsa_signal_value_t = ctypes.c_int64
try: (hsa_signal_create:=dll.hsa_signal_create).restype, hsa_signal_create.argtypes = hsa_status_t, [hsa_signal_value_t, uint32_t, ctypes.POINTER(hsa_agent_t), ctypes.POINTER(hsa_signal_t)]
except AttributeError: pass

try: (hsa_signal_destroy:=dll.hsa_signal_destroy).restype, hsa_signal_destroy.argtypes = hsa_status_t, [hsa_signal_t]
except AttributeError: pass

try: (hsa_signal_load_scacquire:=dll.hsa_signal_load_scacquire).restype, hsa_signal_load_scacquire.argtypes = hsa_signal_value_t, [hsa_signal_t]
except AttributeError: pass

try: (hsa_signal_load_relaxed:=dll.hsa_signal_load_relaxed).restype, hsa_signal_load_relaxed.argtypes = hsa_signal_value_t, [hsa_signal_t]
except AttributeError: pass

try: (hsa_signal_load_acquire:=dll.hsa_signal_load_acquire).restype, hsa_signal_load_acquire.argtypes = hsa_signal_value_t, [hsa_signal_t]
except AttributeError: pass

try: (hsa_signal_store_relaxed:=dll.hsa_signal_store_relaxed).restype, hsa_signal_store_relaxed.argtypes = None, [hsa_signal_t, hsa_signal_value_t]
except AttributeError: pass

try: (hsa_signal_store_screlease:=dll.hsa_signal_store_screlease).restype, hsa_signal_store_screlease.argtypes = None, [hsa_signal_t, hsa_signal_value_t]
except AttributeError: pass

try: (hsa_signal_store_release:=dll.hsa_signal_store_release).restype, hsa_signal_store_release.argtypes = None, [hsa_signal_t, hsa_signal_value_t]
except AttributeError: pass

try: (hsa_signal_silent_store_relaxed:=dll.hsa_signal_silent_store_relaxed).restype, hsa_signal_silent_store_relaxed.argtypes = None, [hsa_signal_t, hsa_signal_value_t]
except AttributeError: pass

try: (hsa_signal_silent_store_screlease:=dll.hsa_signal_silent_store_screlease).restype, hsa_signal_silent_store_screlease.argtypes = None, [hsa_signal_t, hsa_signal_value_t]
except AttributeError: pass

try: (hsa_signal_exchange_scacq_screl:=dll.hsa_signal_exchange_scacq_screl).restype, hsa_signal_exchange_scacq_screl.argtypes = hsa_signal_value_t, [hsa_signal_t, hsa_signal_value_t]
except AttributeError: pass

try: (hsa_signal_exchange_acq_rel:=dll.hsa_signal_exchange_acq_rel).restype, hsa_signal_exchange_acq_rel.argtypes = hsa_signal_value_t, [hsa_signal_t, hsa_signal_value_t]
except AttributeError: pass

try: (hsa_signal_exchange_scacquire:=dll.hsa_signal_exchange_scacquire).restype, hsa_signal_exchange_scacquire.argtypes = hsa_signal_value_t, [hsa_signal_t, hsa_signal_value_t]
except AttributeError: pass

try: (hsa_signal_exchange_acquire:=dll.hsa_signal_exchange_acquire).restype, hsa_signal_exchange_acquire.argtypes = hsa_signal_value_t, [hsa_signal_t, hsa_signal_value_t]
except AttributeError: pass

try: (hsa_signal_exchange_relaxed:=dll.hsa_signal_exchange_relaxed).restype, hsa_signal_exchange_relaxed.argtypes = hsa_signal_value_t, [hsa_signal_t, hsa_signal_value_t]
except AttributeError: pass

try: (hsa_signal_exchange_screlease:=dll.hsa_signal_exchange_screlease).restype, hsa_signal_exchange_screlease.argtypes = hsa_signal_value_t, [hsa_signal_t, hsa_signal_value_t]
except AttributeError: pass

try: (hsa_signal_exchange_release:=dll.hsa_signal_exchange_release).restype, hsa_signal_exchange_release.argtypes = hsa_signal_value_t, [hsa_signal_t, hsa_signal_value_t]
except AttributeError: pass

try: (hsa_signal_cas_scacq_screl:=dll.hsa_signal_cas_scacq_screl).restype, hsa_signal_cas_scacq_screl.argtypes = hsa_signal_value_t, [hsa_signal_t, hsa_signal_value_t, hsa_signal_value_t]
except AttributeError: pass

try: (hsa_signal_cas_acq_rel:=dll.hsa_signal_cas_acq_rel).restype, hsa_signal_cas_acq_rel.argtypes = hsa_signal_value_t, [hsa_signal_t, hsa_signal_value_t, hsa_signal_value_t]
except AttributeError: pass

try: (hsa_signal_cas_scacquire:=dll.hsa_signal_cas_scacquire).restype, hsa_signal_cas_scacquire.argtypes = hsa_signal_value_t, [hsa_signal_t, hsa_signal_value_t, hsa_signal_value_t]
except AttributeError: pass

try: (hsa_signal_cas_acquire:=dll.hsa_signal_cas_acquire).restype, hsa_signal_cas_acquire.argtypes = hsa_signal_value_t, [hsa_signal_t, hsa_signal_value_t, hsa_signal_value_t]
except AttributeError: pass

try: (hsa_signal_cas_relaxed:=dll.hsa_signal_cas_relaxed).restype, hsa_signal_cas_relaxed.argtypes = hsa_signal_value_t, [hsa_signal_t, hsa_signal_value_t, hsa_signal_value_t]
except AttributeError: pass

try: (hsa_signal_cas_screlease:=dll.hsa_signal_cas_screlease).restype, hsa_signal_cas_screlease.argtypes = hsa_signal_value_t, [hsa_signal_t, hsa_signal_value_t, hsa_signal_value_t]
except AttributeError: pass

try: (hsa_signal_cas_release:=dll.hsa_signal_cas_release).restype, hsa_signal_cas_release.argtypes = hsa_signal_value_t, [hsa_signal_t, hsa_signal_value_t, hsa_signal_value_t]
except AttributeError: pass

try: (hsa_signal_add_scacq_screl:=dll.hsa_signal_add_scacq_screl).restype, hsa_signal_add_scacq_screl.argtypes = None, [hsa_signal_t, hsa_signal_value_t]
except AttributeError: pass

try: (hsa_signal_add_acq_rel:=dll.hsa_signal_add_acq_rel).restype, hsa_signal_add_acq_rel.argtypes = None, [hsa_signal_t, hsa_signal_value_t]
except AttributeError: pass

try: (hsa_signal_add_scacquire:=dll.hsa_signal_add_scacquire).restype, hsa_signal_add_scacquire.argtypes = None, [hsa_signal_t, hsa_signal_value_t]
except AttributeError: pass

try: (hsa_signal_add_acquire:=dll.hsa_signal_add_acquire).restype, hsa_signal_add_acquire.argtypes = None, [hsa_signal_t, hsa_signal_value_t]
except AttributeError: pass

try: (hsa_signal_add_relaxed:=dll.hsa_signal_add_relaxed).restype, hsa_signal_add_relaxed.argtypes = None, [hsa_signal_t, hsa_signal_value_t]
except AttributeError: pass

try: (hsa_signal_add_screlease:=dll.hsa_signal_add_screlease).restype, hsa_signal_add_screlease.argtypes = None, [hsa_signal_t, hsa_signal_value_t]
except AttributeError: pass

try: (hsa_signal_add_release:=dll.hsa_signal_add_release).restype, hsa_signal_add_release.argtypes = None, [hsa_signal_t, hsa_signal_value_t]
except AttributeError: pass

try: (hsa_signal_subtract_scacq_screl:=dll.hsa_signal_subtract_scacq_screl).restype, hsa_signal_subtract_scacq_screl.argtypes = None, [hsa_signal_t, hsa_signal_value_t]
except AttributeError: pass

try: (hsa_signal_subtract_acq_rel:=dll.hsa_signal_subtract_acq_rel).restype, hsa_signal_subtract_acq_rel.argtypes = None, [hsa_signal_t, hsa_signal_value_t]
except AttributeError: pass

try: (hsa_signal_subtract_scacquire:=dll.hsa_signal_subtract_scacquire).restype, hsa_signal_subtract_scacquire.argtypes = None, [hsa_signal_t, hsa_signal_value_t]
except AttributeError: pass

try: (hsa_signal_subtract_acquire:=dll.hsa_signal_subtract_acquire).restype, hsa_signal_subtract_acquire.argtypes = None, [hsa_signal_t, hsa_signal_value_t]
except AttributeError: pass

try: (hsa_signal_subtract_relaxed:=dll.hsa_signal_subtract_relaxed).restype, hsa_signal_subtract_relaxed.argtypes = None, [hsa_signal_t, hsa_signal_value_t]
except AttributeError: pass

try: (hsa_signal_subtract_screlease:=dll.hsa_signal_subtract_screlease).restype, hsa_signal_subtract_screlease.argtypes = None, [hsa_signal_t, hsa_signal_value_t]
except AttributeError: pass

try: (hsa_signal_subtract_release:=dll.hsa_signal_subtract_release).restype, hsa_signal_subtract_release.argtypes = None, [hsa_signal_t, hsa_signal_value_t]
except AttributeError: pass

try: (hsa_signal_and_scacq_screl:=dll.hsa_signal_and_scacq_screl).restype, hsa_signal_and_scacq_screl.argtypes = None, [hsa_signal_t, hsa_signal_value_t]
except AttributeError: pass

try: (hsa_signal_and_acq_rel:=dll.hsa_signal_and_acq_rel).restype, hsa_signal_and_acq_rel.argtypes = None, [hsa_signal_t, hsa_signal_value_t]
except AttributeError: pass

try: (hsa_signal_and_scacquire:=dll.hsa_signal_and_scacquire).restype, hsa_signal_and_scacquire.argtypes = None, [hsa_signal_t, hsa_signal_value_t]
except AttributeError: pass

try: (hsa_signal_and_acquire:=dll.hsa_signal_and_acquire).restype, hsa_signal_and_acquire.argtypes = None, [hsa_signal_t, hsa_signal_value_t]
except AttributeError: pass

try: (hsa_signal_and_relaxed:=dll.hsa_signal_and_relaxed).restype, hsa_signal_and_relaxed.argtypes = None, [hsa_signal_t, hsa_signal_value_t]
except AttributeError: pass

try: (hsa_signal_and_screlease:=dll.hsa_signal_and_screlease).restype, hsa_signal_and_screlease.argtypes = None, [hsa_signal_t, hsa_signal_value_t]
except AttributeError: pass

try: (hsa_signal_and_release:=dll.hsa_signal_and_release).restype, hsa_signal_and_release.argtypes = None, [hsa_signal_t, hsa_signal_value_t]
except AttributeError: pass

try: (hsa_signal_or_scacq_screl:=dll.hsa_signal_or_scacq_screl).restype, hsa_signal_or_scacq_screl.argtypes = None, [hsa_signal_t, hsa_signal_value_t]
except AttributeError: pass

try: (hsa_signal_or_acq_rel:=dll.hsa_signal_or_acq_rel).restype, hsa_signal_or_acq_rel.argtypes = None, [hsa_signal_t, hsa_signal_value_t]
except AttributeError: pass

try: (hsa_signal_or_scacquire:=dll.hsa_signal_or_scacquire).restype, hsa_signal_or_scacquire.argtypes = None, [hsa_signal_t, hsa_signal_value_t]
except AttributeError: pass

try: (hsa_signal_or_acquire:=dll.hsa_signal_or_acquire).restype, hsa_signal_or_acquire.argtypes = None, [hsa_signal_t, hsa_signal_value_t]
except AttributeError: pass

try: (hsa_signal_or_relaxed:=dll.hsa_signal_or_relaxed).restype, hsa_signal_or_relaxed.argtypes = None, [hsa_signal_t, hsa_signal_value_t]
except AttributeError: pass

try: (hsa_signal_or_screlease:=dll.hsa_signal_or_screlease).restype, hsa_signal_or_screlease.argtypes = None, [hsa_signal_t, hsa_signal_value_t]
except AttributeError: pass

try: (hsa_signal_or_release:=dll.hsa_signal_or_release).restype, hsa_signal_or_release.argtypes = None, [hsa_signal_t, hsa_signal_value_t]
except AttributeError: pass

try: (hsa_signal_xor_scacq_screl:=dll.hsa_signal_xor_scacq_screl).restype, hsa_signal_xor_scacq_screl.argtypes = None, [hsa_signal_t, hsa_signal_value_t]
except AttributeError: pass

try: (hsa_signal_xor_acq_rel:=dll.hsa_signal_xor_acq_rel).restype, hsa_signal_xor_acq_rel.argtypes = None, [hsa_signal_t, hsa_signal_value_t]
except AttributeError: pass

try: (hsa_signal_xor_scacquire:=dll.hsa_signal_xor_scacquire).restype, hsa_signal_xor_scacquire.argtypes = None, [hsa_signal_t, hsa_signal_value_t]
except AttributeError: pass

try: (hsa_signal_xor_acquire:=dll.hsa_signal_xor_acquire).restype, hsa_signal_xor_acquire.argtypes = None, [hsa_signal_t, hsa_signal_value_t]
except AttributeError: pass

try: (hsa_signal_xor_relaxed:=dll.hsa_signal_xor_relaxed).restype, hsa_signal_xor_relaxed.argtypes = None, [hsa_signal_t, hsa_signal_value_t]
except AttributeError: pass

try: (hsa_signal_xor_screlease:=dll.hsa_signal_xor_screlease).restype, hsa_signal_xor_screlease.argtypes = None, [hsa_signal_t, hsa_signal_value_t]
except AttributeError: pass

try: (hsa_signal_xor_release:=dll.hsa_signal_xor_release).restype, hsa_signal_xor_release.argtypes = None, [hsa_signal_t, hsa_signal_value_t]
except AttributeError: pass

hsa_signal_condition_t = CEnum(ctypes.c_uint32)
HSA_SIGNAL_CONDITION_EQ = hsa_signal_condition_t.define('HSA_SIGNAL_CONDITION_EQ', 0)
HSA_SIGNAL_CONDITION_NE = hsa_signal_condition_t.define('HSA_SIGNAL_CONDITION_NE', 1)
HSA_SIGNAL_CONDITION_LT = hsa_signal_condition_t.define('HSA_SIGNAL_CONDITION_LT', 2)
HSA_SIGNAL_CONDITION_GTE = hsa_signal_condition_t.define('HSA_SIGNAL_CONDITION_GTE', 3)

hsa_wait_state_t = CEnum(ctypes.c_uint32)
HSA_WAIT_STATE_BLOCKED = hsa_wait_state_t.define('HSA_WAIT_STATE_BLOCKED', 0)
HSA_WAIT_STATE_ACTIVE = hsa_wait_state_t.define('HSA_WAIT_STATE_ACTIVE', 1)

try: (hsa_signal_wait_scacquire:=dll.hsa_signal_wait_scacquire).restype, hsa_signal_wait_scacquire.argtypes = hsa_signal_value_t, [hsa_signal_t, hsa_signal_condition_t, hsa_signal_value_t, uint64_t, hsa_wait_state_t]
except AttributeError: pass

try: (hsa_signal_wait_relaxed:=dll.hsa_signal_wait_relaxed).restype, hsa_signal_wait_relaxed.argtypes = hsa_signal_value_t, [hsa_signal_t, hsa_signal_condition_t, hsa_signal_value_t, uint64_t, hsa_wait_state_t]
except AttributeError: pass

try: (hsa_signal_wait_acquire:=dll.hsa_signal_wait_acquire).restype, hsa_signal_wait_acquire.argtypes = hsa_signal_value_t, [hsa_signal_t, hsa_signal_condition_t, hsa_signal_value_t, uint64_t, hsa_wait_state_t]
except AttributeError: pass

class struct_hsa_signal_group_s(Struct): pass
struct_hsa_signal_group_s._fields_ = [
  ('handle', uint64_t),
]
hsa_signal_group_t = struct_hsa_signal_group_s
try: (hsa_signal_group_create:=dll.hsa_signal_group_create).restype, hsa_signal_group_create.argtypes = hsa_status_t, [uint32_t, ctypes.POINTER(hsa_signal_t), uint32_t, ctypes.POINTER(hsa_agent_t), ctypes.POINTER(hsa_signal_group_t)]
except AttributeError: pass

try: (hsa_signal_group_destroy:=dll.hsa_signal_group_destroy).restype, hsa_signal_group_destroy.argtypes = hsa_status_t, [hsa_signal_group_t]
except AttributeError: pass

try: (hsa_signal_group_wait_any_scacquire:=dll.hsa_signal_group_wait_any_scacquire).restype, hsa_signal_group_wait_any_scacquire.argtypes = hsa_status_t, [hsa_signal_group_t, ctypes.POINTER(hsa_signal_condition_t), ctypes.POINTER(hsa_signal_value_t), hsa_wait_state_t, ctypes.POINTER(hsa_signal_t), ctypes.POINTER(hsa_signal_value_t)]
except AttributeError: pass

try: (hsa_signal_group_wait_any_relaxed:=dll.hsa_signal_group_wait_any_relaxed).restype, hsa_signal_group_wait_any_relaxed.argtypes = hsa_status_t, [hsa_signal_group_t, ctypes.POINTER(hsa_signal_condition_t), ctypes.POINTER(hsa_signal_value_t), hsa_wait_state_t, ctypes.POINTER(hsa_signal_t), ctypes.POINTER(hsa_signal_value_t)]
except AttributeError: pass

class struct_hsa_region_s(Struct): pass
struct_hsa_region_s._fields_ = [
  ('handle', uint64_t),
]
hsa_region_t = struct_hsa_region_s
hsa_queue_type_t = CEnum(ctypes.c_uint32)
HSA_QUEUE_TYPE_MULTI = hsa_queue_type_t.define('HSA_QUEUE_TYPE_MULTI', 0)
HSA_QUEUE_TYPE_SINGLE = hsa_queue_type_t.define('HSA_QUEUE_TYPE_SINGLE', 1)
HSA_QUEUE_TYPE_COOPERATIVE = hsa_queue_type_t.define('HSA_QUEUE_TYPE_COOPERATIVE', 2)

hsa_queue_type32_t = ctypes.c_uint32
hsa_queue_feature_t = CEnum(ctypes.c_uint32)
HSA_QUEUE_FEATURE_KERNEL_DISPATCH = hsa_queue_feature_t.define('HSA_QUEUE_FEATURE_KERNEL_DISPATCH', 1)
HSA_QUEUE_FEATURE_AGENT_DISPATCH = hsa_queue_feature_t.define('HSA_QUEUE_FEATURE_AGENT_DISPATCH', 2)

class struct_hsa_queue_s(Struct): pass
struct_hsa_queue_s._fields_ = [
  ('type', hsa_queue_type32_t),
  ('features', uint32_t),
  ('base_address', ctypes.c_void_p),
  ('doorbell_signal', hsa_signal_t),
  ('size', uint32_t),
  ('reserved1', uint32_t),
  ('id', uint64_t),
]
hsa_queue_t = struct_hsa_queue_s
try: (hsa_queue_create:=dll.hsa_queue_create).restype, hsa_queue_create.argtypes = hsa_status_t, [hsa_agent_t, uint32_t, hsa_queue_type32_t, ctypes.CFUNCTYPE(None, hsa_status_t, ctypes.POINTER(hsa_queue_t), ctypes.c_void_p), ctypes.c_void_p, uint32_t, uint32_t, ctypes.POINTER(ctypes.POINTER(hsa_queue_t))]
except AttributeError: pass

try: (hsa_soft_queue_create:=dll.hsa_soft_queue_create).restype, hsa_soft_queue_create.argtypes = hsa_status_t, [hsa_region_t, uint32_t, hsa_queue_type32_t, uint32_t, hsa_signal_t, ctypes.POINTER(ctypes.POINTER(hsa_queue_t))]
except AttributeError: pass

try: (hsa_queue_destroy:=dll.hsa_queue_destroy).restype, hsa_queue_destroy.argtypes = hsa_status_t, [ctypes.POINTER(hsa_queue_t)]
except AttributeError: pass

try: (hsa_queue_inactivate:=dll.hsa_queue_inactivate).restype, hsa_queue_inactivate.argtypes = hsa_status_t, [ctypes.POINTER(hsa_queue_t)]
except AttributeError: pass

try: (hsa_queue_load_read_index_acquire:=dll.hsa_queue_load_read_index_acquire).restype, hsa_queue_load_read_index_acquire.argtypes = uint64_t, [ctypes.POINTER(hsa_queue_t)]
except AttributeError: pass

try: (hsa_queue_load_read_index_scacquire:=dll.hsa_queue_load_read_index_scacquire).restype, hsa_queue_load_read_index_scacquire.argtypes = uint64_t, [ctypes.POINTER(hsa_queue_t)]
except AttributeError: pass

try: (hsa_queue_load_read_index_relaxed:=dll.hsa_queue_load_read_index_relaxed).restype, hsa_queue_load_read_index_relaxed.argtypes = uint64_t, [ctypes.POINTER(hsa_queue_t)]
except AttributeError: pass

try: (hsa_queue_load_write_index_acquire:=dll.hsa_queue_load_write_index_acquire).restype, hsa_queue_load_write_index_acquire.argtypes = uint64_t, [ctypes.POINTER(hsa_queue_t)]
except AttributeError: pass

try: (hsa_queue_load_write_index_scacquire:=dll.hsa_queue_load_write_index_scacquire).restype, hsa_queue_load_write_index_scacquire.argtypes = uint64_t, [ctypes.POINTER(hsa_queue_t)]
except AttributeError: pass

try: (hsa_queue_load_write_index_relaxed:=dll.hsa_queue_load_write_index_relaxed).restype, hsa_queue_load_write_index_relaxed.argtypes = uint64_t, [ctypes.POINTER(hsa_queue_t)]
except AttributeError: pass

try: (hsa_queue_store_write_index_relaxed:=dll.hsa_queue_store_write_index_relaxed).restype, hsa_queue_store_write_index_relaxed.argtypes = None, [ctypes.POINTER(hsa_queue_t), uint64_t]
except AttributeError: pass

try: (hsa_queue_store_write_index_release:=dll.hsa_queue_store_write_index_release).restype, hsa_queue_store_write_index_release.argtypes = None, [ctypes.POINTER(hsa_queue_t), uint64_t]
except AttributeError: pass

try: (hsa_queue_store_write_index_screlease:=dll.hsa_queue_store_write_index_screlease).restype, hsa_queue_store_write_index_screlease.argtypes = None, [ctypes.POINTER(hsa_queue_t), uint64_t]
except AttributeError: pass

try: (hsa_queue_cas_write_index_acq_rel:=dll.hsa_queue_cas_write_index_acq_rel).restype, hsa_queue_cas_write_index_acq_rel.argtypes = uint64_t, [ctypes.POINTER(hsa_queue_t), uint64_t, uint64_t]
except AttributeError: pass

try: (hsa_queue_cas_write_index_scacq_screl:=dll.hsa_queue_cas_write_index_scacq_screl).restype, hsa_queue_cas_write_index_scacq_screl.argtypes = uint64_t, [ctypes.POINTER(hsa_queue_t), uint64_t, uint64_t]
except AttributeError: pass

try: (hsa_queue_cas_write_index_acquire:=dll.hsa_queue_cas_write_index_acquire).restype, hsa_queue_cas_write_index_acquire.argtypes = uint64_t, [ctypes.POINTER(hsa_queue_t), uint64_t, uint64_t]
except AttributeError: pass

try: (hsa_queue_cas_write_index_scacquire:=dll.hsa_queue_cas_write_index_scacquire).restype, hsa_queue_cas_write_index_scacquire.argtypes = uint64_t, [ctypes.POINTER(hsa_queue_t), uint64_t, uint64_t]
except AttributeError: pass

try: (hsa_queue_cas_write_index_relaxed:=dll.hsa_queue_cas_write_index_relaxed).restype, hsa_queue_cas_write_index_relaxed.argtypes = uint64_t, [ctypes.POINTER(hsa_queue_t), uint64_t, uint64_t]
except AttributeError: pass

try: (hsa_queue_cas_write_index_release:=dll.hsa_queue_cas_write_index_release).restype, hsa_queue_cas_write_index_release.argtypes = uint64_t, [ctypes.POINTER(hsa_queue_t), uint64_t, uint64_t]
except AttributeError: pass

try: (hsa_queue_cas_write_index_screlease:=dll.hsa_queue_cas_write_index_screlease).restype, hsa_queue_cas_write_index_screlease.argtypes = uint64_t, [ctypes.POINTER(hsa_queue_t), uint64_t, uint64_t]
except AttributeError: pass

try: (hsa_queue_add_write_index_acq_rel:=dll.hsa_queue_add_write_index_acq_rel).restype, hsa_queue_add_write_index_acq_rel.argtypes = uint64_t, [ctypes.POINTER(hsa_queue_t), uint64_t]
except AttributeError: pass

try: (hsa_queue_add_write_index_scacq_screl:=dll.hsa_queue_add_write_index_scacq_screl).restype, hsa_queue_add_write_index_scacq_screl.argtypes = uint64_t, [ctypes.POINTER(hsa_queue_t), uint64_t]
except AttributeError: pass

try: (hsa_queue_add_write_index_acquire:=dll.hsa_queue_add_write_index_acquire).restype, hsa_queue_add_write_index_acquire.argtypes = uint64_t, [ctypes.POINTER(hsa_queue_t), uint64_t]
except AttributeError: pass

try: (hsa_queue_add_write_index_scacquire:=dll.hsa_queue_add_write_index_scacquire).restype, hsa_queue_add_write_index_scacquire.argtypes = uint64_t, [ctypes.POINTER(hsa_queue_t), uint64_t]
except AttributeError: pass

try: (hsa_queue_add_write_index_relaxed:=dll.hsa_queue_add_write_index_relaxed).restype, hsa_queue_add_write_index_relaxed.argtypes = uint64_t, [ctypes.POINTER(hsa_queue_t), uint64_t]
except AttributeError: pass

try: (hsa_queue_add_write_index_release:=dll.hsa_queue_add_write_index_release).restype, hsa_queue_add_write_index_release.argtypes = uint64_t, [ctypes.POINTER(hsa_queue_t), uint64_t]
except AttributeError: pass

try: (hsa_queue_add_write_index_screlease:=dll.hsa_queue_add_write_index_screlease).restype, hsa_queue_add_write_index_screlease.argtypes = uint64_t, [ctypes.POINTER(hsa_queue_t), uint64_t]
except AttributeError: pass

try: (hsa_queue_store_read_index_relaxed:=dll.hsa_queue_store_read_index_relaxed).restype, hsa_queue_store_read_index_relaxed.argtypes = None, [ctypes.POINTER(hsa_queue_t), uint64_t]
except AttributeError: pass

try: (hsa_queue_store_read_index_release:=dll.hsa_queue_store_read_index_release).restype, hsa_queue_store_read_index_release.argtypes = None, [ctypes.POINTER(hsa_queue_t), uint64_t]
except AttributeError: pass

try: (hsa_queue_store_read_index_screlease:=dll.hsa_queue_store_read_index_screlease).restype, hsa_queue_store_read_index_screlease.argtypes = None, [ctypes.POINTER(hsa_queue_t), uint64_t]
except AttributeError: pass

hsa_packet_type_t = CEnum(ctypes.c_uint32)
HSA_PACKET_TYPE_VENDOR_SPECIFIC = hsa_packet_type_t.define('HSA_PACKET_TYPE_VENDOR_SPECIFIC', 0)
HSA_PACKET_TYPE_INVALID = hsa_packet_type_t.define('HSA_PACKET_TYPE_INVALID', 1)
HSA_PACKET_TYPE_KERNEL_DISPATCH = hsa_packet_type_t.define('HSA_PACKET_TYPE_KERNEL_DISPATCH', 2)
HSA_PACKET_TYPE_BARRIER_AND = hsa_packet_type_t.define('HSA_PACKET_TYPE_BARRIER_AND', 3)
HSA_PACKET_TYPE_AGENT_DISPATCH = hsa_packet_type_t.define('HSA_PACKET_TYPE_AGENT_DISPATCH', 4)
HSA_PACKET_TYPE_BARRIER_OR = hsa_packet_type_t.define('HSA_PACKET_TYPE_BARRIER_OR', 5)

hsa_fence_scope_t = CEnum(ctypes.c_uint32)
HSA_FENCE_SCOPE_NONE = hsa_fence_scope_t.define('HSA_FENCE_SCOPE_NONE', 0)
HSA_FENCE_SCOPE_AGENT = hsa_fence_scope_t.define('HSA_FENCE_SCOPE_AGENT', 1)
HSA_FENCE_SCOPE_SYSTEM = hsa_fence_scope_t.define('HSA_FENCE_SCOPE_SYSTEM', 2)

hsa_packet_header_t = CEnum(ctypes.c_uint32)
HSA_PACKET_HEADER_TYPE = hsa_packet_header_t.define('HSA_PACKET_HEADER_TYPE', 0)
HSA_PACKET_HEADER_BARRIER = hsa_packet_header_t.define('HSA_PACKET_HEADER_BARRIER', 8)
HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE = hsa_packet_header_t.define('HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE', 9)
HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE = hsa_packet_header_t.define('HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE', 9)
HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE = hsa_packet_header_t.define('HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE', 11)
HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE = hsa_packet_header_t.define('HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE', 11)

hsa_packet_header_width_t = CEnum(ctypes.c_uint32)
HSA_PACKET_HEADER_WIDTH_TYPE = hsa_packet_header_width_t.define('HSA_PACKET_HEADER_WIDTH_TYPE', 8)
HSA_PACKET_HEADER_WIDTH_BARRIER = hsa_packet_header_width_t.define('HSA_PACKET_HEADER_WIDTH_BARRIER', 1)
HSA_PACKET_HEADER_WIDTH_SCACQUIRE_FENCE_SCOPE = hsa_packet_header_width_t.define('HSA_PACKET_HEADER_WIDTH_SCACQUIRE_FENCE_SCOPE', 2)
HSA_PACKET_HEADER_WIDTH_ACQUIRE_FENCE_SCOPE = hsa_packet_header_width_t.define('HSA_PACKET_HEADER_WIDTH_ACQUIRE_FENCE_SCOPE', 2)
HSA_PACKET_HEADER_WIDTH_SCRELEASE_FENCE_SCOPE = hsa_packet_header_width_t.define('HSA_PACKET_HEADER_WIDTH_SCRELEASE_FENCE_SCOPE', 2)
HSA_PACKET_HEADER_WIDTH_RELEASE_FENCE_SCOPE = hsa_packet_header_width_t.define('HSA_PACKET_HEADER_WIDTH_RELEASE_FENCE_SCOPE', 2)

hsa_kernel_dispatch_packet_setup_t = CEnum(ctypes.c_uint32)
HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS = hsa_kernel_dispatch_packet_setup_t.define('HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS', 0)

hsa_kernel_dispatch_packet_setup_width_t = CEnum(ctypes.c_uint32)
HSA_KERNEL_DISPATCH_PACKET_SETUP_WIDTH_DIMENSIONS = hsa_kernel_dispatch_packet_setup_width_t.define('HSA_KERNEL_DISPATCH_PACKET_SETUP_WIDTH_DIMENSIONS', 2)

class struct_hsa_kernel_dispatch_packet_s(Struct): pass
class struct_hsa_kernel_dispatch_packet_s_0(ctypes.Union): pass
class struct_hsa_kernel_dispatch_packet_s_0_0(Struct): pass
struct_hsa_kernel_dispatch_packet_s_0_0._fields_ = [
  ('header', uint16_t),
  ('setup', uint16_t),
]
struct_hsa_kernel_dispatch_packet_s_0._anonymous_ = ['_0']
struct_hsa_kernel_dispatch_packet_s_0._fields_ = [
  ('_0', struct_hsa_kernel_dispatch_packet_s_0_0),
  ('full_header', uint32_t),
]
struct_hsa_kernel_dispatch_packet_s._anonymous_ = ['_0']
struct_hsa_kernel_dispatch_packet_s._fields_ = [
  ('_0', struct_hsa_kernel_dispatch_packet_s_0),
  ('workgroup_size_x', uint16_t),
  ('workgroup_size_y', uint16_t),
  ('workgroup_size_z', uint16_t),
  ('reserved0', uint16_t),
  ('grid_size_x', uint32_t),
  ('grid_size_y', uint32_t),
  ('grid_size_z', uint32_t),
  ('private_segment_size', uint32_t),
  ('group_segment_size', uint32_t),
  ('kernel_object', uint64_t),
  ('kernarg_address', ctypes.c_void_p),
  ('reserved2', uint64_t),
  ('completion_signal', hsa_signal_t),
]
hsa_kernel_dispatch_packet_t = struct_hsa_kernel_dispatch_packet_s
class struct_hsa_agent_dispatch_packet_s(Struct): pass
struct_hsa_agent_dispatch_packet_s._fields_ = [
  ('header', uint16_t),
  ('type', uint16_t),
  ('reserved0', uint32_t),
  ('return_address', ctypes.c_void_p),
  ('arg', (uint64_t * 4)),
  ('reserved2', uint64_t),
  ('completion_signal', hsa_signal_t),
]
hsa_agent_dispatch_packet_t = struct_hsa_agent_dispatch_packet_s
class struct_hsa_barrier_and_packet_s(Struct): pass
struct_hsa_barrier_and_packet_s._fields_ = [
  ('header', uint16_t),
  ('reserved0', uint16_t),
  ('reserved1', uint32_t),
  ('dep_signal', (hsa_signal_t * 5)),
  ('reserved2', uint64_t),
  ('completion_signal', hsa_signal_t),
]
hsa_barrier_and_packet_t = struct_hsa_barrier_and_packet_s
class struct_hsa_barrier_or_packet_s(Struct): pass
struct_hsa_barrier_or_packet_s._fields_ = [
  ('header', uint16_t),
  ('reserved0', uint16_t),
  ('reserved1', uint32_t),
  ('dep_signal', (hsa_signal_t * 5)),
  ('reserved2', uint64_t),
  ('completion_signal', hsa_signal_t),
]
hsa_barrier_or_packet_t = struct_hsa_barrier_or_packet_s
hsa_region_segment_t = CEnum(ctypes.c_uint32)
HSA_REGION_SEGMENT_GLOBAL = hsa_region_segment_t.define('HSA_REGION_SEGMENT_GLOBAL', 0)
HSA_REGION_SEGMENT_READONLY = hsa_region_segment_t.define('HSA_REGION_SEGMENT_READONLY', 1)
HSA_REGION_SEGMENT_PRIVATE = hsa_region_segment_t.define('HSA_REGION_SEGMENT_PRIVATE', 2)
HSA_REGION_SEGMENT_GROUP = hsa_region_segment_t.define('HSA_REGION_SEGMENT_GROUP', 3)
HSA_REGION_SEGMENT_KERNARG = hsa_region_segment_t.define('HSA_REGION_SEGMENT_KERNARG', 4)

hsa_region_global_flag_t = CEnum(ctypes.c_uint32)
HSA_REGION_GLOBAL_FLAG_KERNARG = hsa_region_global_flag_t.define('HSA_REGION_GLOBAL_FLAG_KERNARG', 1)
HSA_REGION_GLOBAL_FLAG_FINE_GRAINED = hsa_region_global_flag_t.define('HSA_REGION_GLOBAL_FLAG_FINE_GRAINED', 2)
HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED = hsa_region_global_flag_t.define('HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED', 4)
HSA_REGION_GLOBAL_FLAG_EXTENDED_SCOPE_FINE_GRAINED = hsa_region_global_flag_t.define('HSA_REGION_GLOBAL_FLAG_EXTENDED_SCOPE_FINE_GRAINED', 8)

hsa_region_info_t = CEnum(ctypes.c_uint32)
HSA_REGION_INFO_SEGMENT = hsa_region_info_t.define('HSA_REGION_INFO_SEGMENT', 0)
HSA_REGION_INFO_GLOBAL_FLAGS = hsa_region_info_t.define('HSA_REGION_INFO_GLOBAL_FLAGS', 1)
HSA_REGION_INFO_SIZE = hsa_region_info_t.define('HSA_REGION_INFO_SIZE', 2)
HSA_REGION_INFO_ALLOC_MAX_SIZE = hsa_region_info_t.define('HSA_REGION_INFO_ALLOC_MAX_SIZE', 4)
HSA_REGION_INFO_ALLOC_MAX_PRIVATE_WORKGROUP_SIZE = hsa_region_info_t.define('HSA_REGION_INFO_ALLOC_MAX_PRIVATE_WORKGROUP_SIZE', 8)
HSA_REGION_INFO_RUNTIME_ALLOC_ALLOWED = hsa_region_info_t.define('HSA_REGION_INFO_RUNTIME_ALLOC_ALLOWED', 5)
HSA_REGION_INFO_RUNTIME_ALLOC_GRANULE = hsa_region_info_t.define('HSA_REGION_INFO_RUNTIME_ALLOC_GRANULE', 6)
HSA_REGION_INFO_RUNTIME_ALLOC_ALIGNMENT = hsa_region_info_t.define('HSA_REGION_INFO_RUNTIME_ALLOC_ALIGNMENT', 7)

try: (hsa_region_get_info:=dll.hsa_region_get_info).restype, hsa_region_get_info.argtypes = hsa_status_t, [hsa_region_t, hsa_region_info_t, ctypes.c_void_p]
except AttributeError: pass

try: (hsa_agent_iterate_regions:=dll.hsa_agent_iterate_regions).restype, hsa_agent_iterate_regions.argtypes = hsa_status_t, [hsa_agent_t, ctypes.CFUNCTYPE(hsa_status_t, hsa_region_t, ctypes.c_void_p), ctypes.c_void_p]
except AttributeError: pass

try: (hsa_memory_allocate:=dll.hsa_memory_allocate).restype, hsa_memory_allocate.argtypes = hsa_status_t, [hsa_region_t, size_t, ctypes.POINTER(ctypes.c_void_p)]
except AttributeError: pass

try: (hsa_memory_free:=dll.hsa_memory_free).restype, hsa_memory_free.argtypes = hsa_status_t, [ctypes.c_void_p]
except AttributeError: pass

try: (hsa_memory_copy:=dll.hsa_memory_copy).restype, hsa_memory_copy.argtypes = hsa_status_t, [ctypes.c_void_p, ctypes.c_void_p, size_t]
except AttributeError: pass

try: (hsa_memory_assign_agent:=dll.hsa_memory_assign_agent).restype, hsa_memory_assign_agent.argtypes = hsa_status_t, [ctypes.c_void_p, hsa_agent_t, hsa_access_permission_t]
except AttributeError: pass

try: (hsa_memory_register:=dll.hsa_memory_register).restype, hsa_memory_register.argtypes = hsa_status_t, [ctypes.c_void_p, size_t]
except AttributeError: pass

try: (hsa_memory_deregister:=dll.hsa_memory_deregister).restype, hsa_memory_deregister.argtypes = hsa_status_t, [ctypes.c_void_p, size_t]
except AttributeError: pass

class struct_hsa_isa_s(Struct): pass
struct_hsa_isa_s._fields_ = [
  ('handle', uint64_t),
]
hsa_isa_t = struct_hsa_isa_s
try: (hsa_isa_from_name:=dll.hsa_isa_from_name).restype, hsa_isa_from_name.argtypes = hsa_status_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(hsa_isa_t)]
except AttributeError: pass

try: (hsa_agent_iterate_isas:=dll.hsa_agent_iterate_isas).restype, hsa_agent_iterate_isas.argtypes = hsa_status_t, [hsa_agent_t, ctypes.CFUNCTYPE(hsa_status_t, hsa_isa_t, ctypes.c_void_p), ctypes.c_void_p]
except AttributeError: pass

hsa_isa_info_t = CEnum(ctypes.c_uint32)
HSA_ISA_INFO_NAME_LENGTH = hsa_isa_info_t.define('HSA_ISA_INFO_NAME_LENGTH', 0)
HSA_ISA_INFO_NAME = hsa_isa_info_t.define('HSA_ISA_INFO_NAME', 1)
HSA_ISA_INFO_CALL_CONVENTION_COUNT = hsa_isa_info_t.define('HSA_ISA_INFO_CALL_CONVENTION_COUNT', 2)
HSA_ISA_INFO_CALL_CONVENTION_INFO_WAVEFRONT_SIZE = hsa_isa_info_t.define('HSA_ISA_INFO_CALL_CONVENTION_INFO_WAVEFRONT_SIZE', 3)
HSA_ISA_INFO_CALL_CONVENTION_INFO_WAVEFRONTS_PER_COMPUTE_UNIT = hsa_isa_info_t.define('HSA_ISA_INFO_CALL_CONVENTION_INFO_WAVEFRONTS_PER_COMPUTE_UNIT', 4)
HSA_ISA_INFO_MACHINE_MODELS = hsa_isa_info_t.define('HSA_ISA_INFO_MACHINE_MODELS', 5)
HSA_ISA_INFO_PROFILES = hsa_isa_info_t.define('HSA_ISA_INFO_PROFILES', 6)
HSA_ISA_INFO_DEFAULT_FLOAT_ROUNDING_MODES = hsa_isa_info_t.define('HSA_ISA_INFO_DEFAULT_FLOAT_ROUNDING_MODES', 7)
HSA_ISA_INFO_BASE_PROFILE_DEFAULT_FLOAT_ROUNDING_MODES = hsa_isa_info_t.define('HSA_ISA_INFO_BASE_PROFILE_DEFAULT_FLOAT_ROUNDING_MODES', 8)
HSA_ISA_INFO_FAST_F16_OPERATION = hsa_isa_info_t.define('HSA_ISA_INFO_FAST_F16_OPERATION', 9)
HSA_ISA_INFO_WORKGROUP_MAX_DIM = hsa_isa_info_t.define('HSA_ISA_INFO_WORKGROUP_MAX_DIM', 12)
HSA_ISA_INFO_WORKGROUP_MAX_SIZE = hsa_isa_info_t.define('HSA_ISA_INFO_WORKGROUP_MAX_SIZE', 13)
HSA_ISA_INFO_GRID_MAX_DIM = hsa_isa_info_t.define('HSA_ISA_INFO_GRID_MAX_DIM', 14)
HSA_ISA_INFO_GRID_MAX_SIZE = hsa_isa_info_t.define('HSA_ISA_INFO_GRID_MAX_SIZE', 16)
HSA_ISA_INFO_FBARRIER_MAX_SIZE = hsa_isa_info_t.define('HSA_ISA_INFO_FBARRIER_MAX_SIZE', 17)

try: (hsa_isa_get_info:=dll.hsa_isa_get_info).restype, hsa_isa_get_info.argtypes = hsa_status_t, [hsa_isa_t, hsa_isa_info_t, uint32_t, ctypes.c_void_p]
except AttributeError: pass

try: (hsa_isa_get_info_alt:=dll.hsa_isa_get_info_alt).restype, hsa_isa_get_info_alt.argtypes = hsa_status_t, [hsa_isa_t, hsa_isa_info_t, ctypes.c_void_p]
except AttributeError: pass

try: (hsa_isa_get_exception_policies:=dll.hsa_isa_get_exception_policies).restype, hsa_isa_get_exception_policies.argtypes = hsa_status_t, [hsa_isa_t, hsa_profile_t, ctypes.POINTER(uint16_t)]
except AttributeError: pass

hsa_fp_type_t = CEnum(ctypes.c_uint32)
HSA_FP_TYPE_16 = hsa_fp_type_t.define('HSA_FP_TYPE_16', 1)
HSA_FP_TYPE_32 = hsa_fp_type_t.define('HSA_FP_TYPE_32', 2)
HSA_FP_TYPE_64 = hsa_fp_type_t.define('HSA_FP_TYPE_64', 4)

hsa_flush_mode_t = CEnum(ctypes.c_uint32)
HSA_FLUSH_MODE_FTZ = hsa_flush_mode_t.define('HSA_FLUSH_MODE_FTZ', 1)
HSA_FLUSH_MODE_NON_FTZ = hsa_flush_mode_t.define('HSA_FLUSH_MODE_NON_FTZ', 2)

hsa_round_method_t = CEnum(ctypes.c_uint32)
HSA_ROUND_METHOD_SINGLE = hsa_round_method_t.define('HSA_ROUND_METHOD_SINGLE', 1)
HSA_ROUND_METHOD_DOUBLE = hsa_round_method_t.define('HSA_ROUND_METHOD_DOUBLE', 2)

try: (hsa_isa_get_round_method:=dll.hsa_isa_get_round_method).restype, hsa_isa_get_round_method.argtypes = hsa_status_t, [hsa_isa_t, hsa_fp_type_t, hsa_flush_mode_t, ctypes.POINTER(hsa_round_method_t)]
except AttributeError: pass

class struct_hsa_wavefront_s(Struct): pass
struct_hsa_wavefront_s._fields_ = [
  ('handle', uint64_t),
]
hsa_wavefront_t = struct_hsa_wavefront_s
hsa_wavefront_info_t = CEnum(ctypes.c_uint32)
HSA_WAVEFRONT_INFO_SIZE = hsa_wavefront_info_t.define('HSA_WAVEFRONT_INFO_SIZE', 0)

try: (hsa_wavefront_get_info:=dll.hsa_wavefront_get_info).restype, hsa_wavefront_get_info.argtypes = hsa_status_t, [hsa_wavefront_t, hsa_wavefront_info_t, ctypes.c_void_p]
except AttributeError: pass

try: (hsa_isa_iterate_wavefronts:=dll.hsa_isa_iterate_wavefronts).restype, hsa_isa_iterate_wavefronts.argtypes = hsa_status_t, [hsa_isa_t, ctypes.CFUNCTYPE(hsa_status_t, hsa_wavefront_t, ctypes.c_void_p), ctypes.c_void_p]
except AttributeError: pass

try: (hsa_isa_compatible:=dll.hsa_isa_compatible).restype, hsa_isa_compatible.argtypes = hsa_status_t, [hsa_isa_t, hsa_isa_t, ctypes.POINTER(ctypes.c_bool)]
except AttributeError: pass

class struct_hsa_code_object_reader_s(Struct): pass
struct_hsa_code_object_reader_s._fields_ = [
  ('handle', uint64_t),
]
hsa_code_object_reader_t = struct_hsa_code_object_reader_s
try: (hsa_code_object_reader_create_from_file:=dll.hsa_code_object_reader_create_from_file).restype, hsa_code_object_reader_create_from_file.argtypes = hsa_status_t, [hsa_file_t, ctypes.POINTER(hsa_code_object_reader_t)]
except AttributeError: pass

try: (hsa_code_object_reader_create_from_memory:=dll.hsa_code_object_reader_create_from_memory).restype, hsa_code_object_reader_create_from_memory.argtypes = hsa_status_t, [ctypes.c_void_p, size_t, ctypes.POINTER(hsa_code_object_reader_t)]
except AttributeError: pass

try: (hsa_code_object_reader_destroy:=dll.hsa_code_object_reader_destroy).restype, hsa_code_object_reader_destroy.argtypes = hsa_status_t, [hsa_code_object_reader_t]
except AttributeError: pass

class struct_hsa_executable_s(Struct): pass
struct_hsa_executable_s._fields_ = [
  ('handle', uint64_t),
]
hsa_executable_t = struct_hsa_executable_s
hsa_executable_state_t = CEnum(ctypes.c_uint32)
HSA_EXECUTABLE_STATE_UNFROZEN = hsa_executable_state_t.define('HSA_EXECUTABLE_STATE_UNFROZEN', 0)
HSA_EXECUTABLE_STATE_FROZEN = hsa_executable_state_t.define('HSA_EXECUTABLE_STATE_FROZEN', 1)

try: (hsa_executable_create:=dll.hsa_executable_create).restype, hsa_executable_create.argtypes = hsa_status_t, [hsa_profile_t, hsa_executable_state_t, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(hsa_executable_t)]
except AttributeError: pass

try: (hsa_executable_create_alt:=dll.hsa_executable_create_alt).restype, hsa_executable_create_alt.argtypes = hsa_status_t, [hsa_profile_t, hsa_default_float_rounding_mode_t, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(hsa_executable_t)]
except AttributeError: pass

try: (hsa_executable_destroy:=dll.hsa_executable_destroy).restype, hsa_executable_destroy.argtypes = hsa_status_t, [hsa_executable_t]
except AttributeError: pass

class struct_hsa_loaded_code_object_s(Struct): pass
struct_hsa_loaded_code_object_s._fields_ = [
  ('handle', uint64_t),
]
hsa_loaded_code_object_t = struct_hsa_loaded_code_object_s
try: (hsa_executable_load_program_code_object:=dll.hsa_executable_load_program_code_object).restype, hsa_executable_load_program_code_object.argtypes = hsa_status_t, [hsa_executable_t, hsa_code_object_reader_t, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(hsa_loaded_code_object_t)]
except AttributeError: pass

try: (hsa_executable_load_agent_code_object:=dll.hsa_executable_load_agent_code_object).restype, hsa_executable_load_agent_code_object.argtypes = hsa_status_t, [hsa_executable_t, hsa_agent_t, hsa_code_object_reader_t, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(hsa_loaded_code_object_t)]
except AttributeError: pass

try: (hsa_executable_freeze:=dll.hsa_executable_freeze).restype, hsa_executable_freeze.argtypes = hsa_status_t, [hsa_executable_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

hsa_executable_info_t = CEnum(ctypes.c_uint32)
HSA_EXECUTABLE_INFO_PROFILE = hsa_executable_info_t.define('HSA_EXECUTABLE_INFO_PROFILE', 1)
HSA_EXECUTABLE_INFO_STATE = hsa_executable_info_t.define('HSA_EXECUTABLE_INFO_STATE', 2)
HSA_EXECUTABLE_INFO_DEFAULT_FLOAT_ROUNDING_MODE = hsa_executable_info_t.define('HSA_EXECUTABLE_INFO_DEFAULT_FLOAT_ROUNDING_MODE', 3)

try: (hsa_executable_get_info:=dll.hsa_executable_get_info).restype, hsa_executable_get_info.argtypes = hsa_status_t, [hsa_executable_t, hsa_executable_info_t, ctypes.c_void_p]
except AttributeError: pass

try: (hsa_executable_global_variable_define:=dll.hsa_executable_global_variable_define).restype, hsa_executable_global_variable_define.argtypes = hsa_status_t, [hsa_executable_t, ctypes.POINTER(ctypes.c_char), ctypes.c_void_p]
except AttributeError: pass

try: (hsa_executable_agent_global_variable_define:=dll.hsa_executable_agent_global_variable_define).restype, hsa_executable_agent_global_variable_define.argtypes = hsa_status_t, [hsa_executable_t, hsa_agent_t, ctypes.POINTER(ctypes.c_char), ctypes.c_void_p]
except AttributeError: pass

try: (hsa_executable_readonly_variable_define:=dll.hsa_executable_readonly_variable_define).restype, hsa_executable_readonly_variable_define.argtypes = hsa_status_t, [hsa_executable_t, hsa_agent_t, ctypes.POINTER(ctypes.c_char), ctypes.c_void_p]
except AttributeError: pass

try: (hsa_executable_validate:=dll.hsa_executable_validate).restype, hsa_executable_validate.argtypes = hsa_status_t, [hsa_executable_t, ctypes.POINTER(uint32_t)]
except AttributeError: pass

try: (hsa_executable_validate_alt:=dll.hsa_executable_validate_alt).restype, hsa_executable_validate_alt.argtypes = hsa_status_t, [hsa_executable_t, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(uint32_t)]
except AttributeError: pass

class struct_hsa_executable_symbol_s(Struct): pass
struct_hsa_executable_symbol_s._fields_ = [
  ('handle', uint64_t),
]
hsa_executable_symbol_t = struct_hsa_executable_symbol_s
int32_t = ctypes.c_int32
try: (hsa_executable_get_symbol:=dll.hsa_executable_get_symbol).restype, hsa_executable_get_symbol.argtypes = hsa_status_t, [hsa_executable_t, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), hsa_agent_t, int32_t, ctypes.POINTER(hsa_executable_symbol_t)]
except AttributeError: pass

try: (hsa_executable_get_symbol_by_name:=dll.hsa_executable_get_symbol_by_name).restype, hsa_executable_get_symbol_by_name.argtypes = hsa_status_t, [hsa_executable_t, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(hsa_agent_t), ctypes.POINTER(hsa_executable_symbol_t)]
except AttributeError: pass

hsa_symbol_kind_t = CEnum(ctypes.c_uint32)
HSA_SYMBOL_KIND_VARIABLE = hsa_symbol_kind_t.define('HSA_SYMBOL_KIND_VARIABLE', 0)
HSA_SYMBOL_KIND_KERNEL = hsa_symbol_kind_t.define('HSA_SYMBOL_KIND_KERNEL', 1)
HSA_SYMBOL_KIND_INDIRECT_FUNCTION = hsa_symbol_kind_t.define('HSA_SYMBOL_KIND_INDIRECT_FUNCTION', 2)

hsa_symbol_linkage_t = CEnum(ctypes.c_uint32)
HSA_SYMBOL_LINKAGE_MODULE = hsa_symbol_linkage_t.define('HSA_SYMBOL_LINKAGE_MODULE', 0)
HSA_SYMBOL_LINKAGE_PROGRAM = hsa_symbol_linkage_t.define('HSA_SYMBOL_LINKAGE_PROGRAM', 1)

hsa_variable_allocation_t = CEnum(ctypes.c_uint32)
HSA_VARIABLE_ALLOCATION_AGENT = hsa_variable_allocation_t.define('HSA_VARIABLE_ALLOCATION_AGENT', 0)
HSA_VARIABLE_ALLOCATION_PROGRAM = hsa_variable_allocation_t.define('HSA_VARIABLE_ALLOCATION_PROGRAM', 1)

hsa_variable_segment_t = CEnum(ctypes.c_uint32)
HSA_VARIABLE_SEGMENT_GLOBAL = hsa_variable_segment_t.define('HSA_VARIABLE_SEGMENT_GLOBAL', 0)
HSA_VARIABLE_SEGMENT_READONLY = hsa_variable_segment_t.define('HSA_VARIABLE_SEGMENT_READONLY', 1)

hsa_executable_symbol_info_t = CEnum(ctypes.c_uint32)
HSA_EXECUTABLE_SYMBOL_INFO_TYPE = hsa_executable_symbol_info_t.define('HSA_EXECUTABLE_SYMBOL_INFO_TYPE', 0)
HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH = hsa_executable_symbol_info_t.define('HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH', 1)
HSA_EXECUTABLE_SYMBOL_INFO_NAME = hsa_executable_symbol_info_t.define('HSA_EXECUTABLE_SYMBOL_INFO_NAME', 2)
HSA_EXECUTABLE_SYMBOL_INFO_MODULE_NAME_LENGTH = hsa_executable_symbol_info_t.define('HSA_EXECUTABLE_SYMBOL_INFO_MODULE_NAME_LENGTH', 3)
HSA_EXECUTABLE_SYMBOL_INFO_MODULE_NAME = hsa_executable_symbol_info_t.define('HSA_EXECUTABLE_SYMBOL_INFO_MODULE_NAME', 4)
HSA_EXECUTABLE_SYMBOL_INFO_AGENT = hsa_executable_symbol_info_t.define('HSA_EXECUTABLE_SYMBOL_INFO_AGENT', 20)
HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS = hsa_executable_symbol_info_t.define('HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS', 21)
HSA_EXECUTABLE_SYMBOL_INFO_LINKAGE = hsa_executable_symbol_info_t.define('HSA_EXECUTABLE_SYMBOL_INFO_LINKAGE', 5)
HSA_EXECUTABLE_SYMBOL_INFO_IS_DEFINITION = hsa_executable_symbol_info_t.define('HSA_EXECUTABLE_SYMBOL_INFO_IS_DEFINITION', 17)
HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ALLOCATION = hsa_executable_symbol_info_t.define('HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ALLOCATION', 6)
HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_SEGMENT = hsa_executable_symbol_info_t.define('HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_SEGMENT', 7)
HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ALIGNMENT = hsa_executable_symbol_info_t.define('HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ALIGNMENT', 8)
HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_SIZE = hsa_executable_symbol_info_t.define('HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_SIZE', 9)
HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_IS_CONST = hsa_executable_symbol_info_t.define('HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_IS_CONST', 10)
HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT = hsa_executable_symbol_info_t.define('HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT', 22)
HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE = hsa_executable_symbol_info_t.define('HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE', 11)
HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT = hsa_executable_symbol_info_t.define('HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT', 12)
HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE = hsa_executable_symbol_info_t.define('HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE', 13)
HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE = hsa_executable_symbol_info_t.define('HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE', 14)
HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_DYNAMIC_CALLSTACK = hsa_executable_symbol_info_t.define('HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_DYNAMIC_CALLSTACK', 15)
HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_CALL_CONVENTION = hsa_executable_symbol_info_t.define('HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_CALL_CONVENTION', 18)
HSA_EXECUTABLE_SYMBOL_INFO_INDIRECT_FUNCTION_OBJECT = hsa_executable_symbol_info_t.define('HSA_EXECUTABLE_SYMBOL_INFO_INDIRECT_FUNCTION_OBJECT', 23)
HSA_EXECUTABLE_SYMBOL_INFO_INDIRECT_FUNCTION_CALL_CONVENTION = hsa_executable_symbol_info_t.define('HSA_EXECUTABLE_SYMBOL_INFO_INDIRECT_FUNCTION_CALL_CONVENTION', 16)

try: (hsa_executable_symbol_get_info:=dll.hsa_executable_symbol_get_info).restype, hsa_executable_symbol_get_info.argtypes = hsa_status_t, [hsa_executable_symbol_t, hsa_executable_symbol_info_t, ctypes.c_void_p]
except AttributeError: pass

try: (hsa_executable_iterate_symbols:=dll.hsa_executable_iterate_symbols).restype, hsa_executable_iterate_symbols.argtypes = hsa_status_t, [hsa_executable_t, ctypes.CFUNCTYPE(hsa_status_t, hsa_executable_t, hsa_executable_symbol_t, ctypes.c_void_p), ctypes.c_void_p]
except AttributeError: pass

try: (hsa_executable_iterate_agent_symbols:=dll.hsa_executable_iterate_agent_symbols).restype, hsa_executable_iterate_agent_symbols.argtypes = hsa_status_t, [hsa_executable_t, hsa_agent_t, ctypes.CFUNCTYPE(hsa_status_t, hsa_executable_t, hsa_agent_t, hsa_executable_symbol_t, ctypes.c_void_p), ctypes.c_void_p]
except AttributeError: pass

try: (hsa_executable_iterate_program_symbols:=dll.hsa_executable_iterate_program_symbols).restype, hsa_executable_iterate_program_symbols.argtypes = hsa_status_t, [hsa_executable_t, ctypes.CFUNCTYPE(hsa_status_t, hsa_executable_t, hsa_executable_symbol_t, ctypes.c_void_p), ctypes.c_void_p]
except AttributeError: pass

class struct_hsa_code_object_s(Struct): pass
struct_hsa_code_object_s._fields_ = [
  ('handle', uint64_t),
]
hsa_code_object_t = struct_hsa_code_object_s
class struct_hsa_callback_data_s(Struct): pass
struct_hsa_callback_data_s._fields_ = [
  ('handle', uint64_t),
]
hsa_callback_data_t = struct_hsa_callback_data_s
try: (hsa_code_object_serialize:=dll.hsa_code_object_serialize).restype, hsa_code_object_serialize.argtypes = hsa_status_t, [hsa_code_object_t, ctypes.CFUNCTYPE(hsa_status_t, size_t, hsa_callback_data_t, ctypes.POINTER(ctypes.c_void_p)), hsa_callback_data_t, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(size_t)]
except AttributeError: pass

try: (hsa_code_object_deserialize:=dll.hsa_code_object_deserialize).restype, hsa_code_object_deserialize.argtypes = hsa_status_t, [ctypes.c_void_p, size_t, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(hsa_code_object_t)]
except AttributeError: pass

try: (hsa_code_object_destroy:=dll.hsa_code_object_destroy).restype, hsa_code_object_destroy.argtypes = hsa_status_t, [hsa_code_object_t]
except AttributeError: pass

hsa_code_object_type_t = CEnum(ctypes.c_uint32)
HSA_CODE_OBJECT_TYPE_PROGRAM = hsa_code_object_type_t.define('HSA_CODE_OBJECT_TYPE_PROGRAM', 0)

hsa_code_object_info_t = CEnum(ctypes.c_uint32)
HSA_CODE_OBJECT_INFO_VERSION = hsa_code_object_info_t.define('HSA_CODE_OBJECT_INFO_VERSION', 0)
HSA_CODE_OBJECT_INFO_TYPE = hsa_code_object_info_t.define('HSA_CODE_OBJECT_INFO_TYPE', 1)
HSA_CODE_OBJECT_INFO_ISA = hsa_code_object_info_t.define('HSA_CODE_OBJECT_INFO_ISA', 2)
HSA_CODE_OBJECT_INFO_MACHINE_MODEL = hsa_code_object_info_t.define('HSA_CODE_OBJECT_INFO_MACHINE_MODEL', 3)
HSA_CODE_OBJECT_INFO_PROFILE = hsa_code_object_info_t.define('HSA_CODE_OBJECT_INFO_PROFILE', 4)
HSA_CODE_OBJECT_INFO_DEFAULT_FLOAT_ROUNDING_MODE = hsa_code_object_info_t.define('HSA_CODE_OBJECT_INFO_DEFAULT_FLOAT_ROUNDING_MODE', 5)

try: (hsa_code_object_get_info:=dll.hsa_code_object_get_info).restype, hsa_code_object_get_info.argtypes = hsa_status_t, [hsa_code_object_t, hsa_code_object_info_t, ctypes.c_void_p]
except AttributeError: pass

try: (hsa_executable_load_code_object:=dll.hsa_executable_load_code_object).restype, hsa_executable_load_code_object.argtypes = hsa_status_t, [hsa_executable_t, hsa_agent_t, hsa_code_object_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

class struct_hsa_code_symbol_s(Struct): pass
struct_hsa_code_symbol_s._fields_ = [
  ('handle', uint64_t),
]
hsa_code_symbol_t = struct_hsa_code_symbol_s
try: (hsa_code_object_get_symbol:=dll.hsa_code_object_get_symbol).restype, hsa_code_object_get_symbol.argtypes = hsa_status_t, [hsa_code_object_t, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(hsa_code_symbol_t)]
except AttributeError: pass

try: (hsa_code_object_get_symbol_from_name:=dll.hsa_code_object_get_symbol_from_name).restype, hsa_code_object_get_symbol_from_name.argtypes = hsa_status_t, [hsa_code_object_t, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(hsa_code_symbol_t)]
except AttributeError: pass

hsa_code_symbol_info_t = CEnum(ctypes.c_uint32)
HSA_CODE_SYMBOL_INFO_TYPE = hsa_code_symbol_info_t.define('HSA_CODE_SYMBOL_INFO_TYPE', 0)
HSA_CODE_SYMBOL_INFO_NAME_LENGTH = hsa_code_symbol_info_t.define('HSA_CODE_SYMBOL_INFO_NAME_LENGTH', 1)
HSA_CODE_SYMBOL_INFO_NAME = hsa_code_symbol_info_t.define('HSA_CODE_SYMBOL_INFO_NAME', 2)
HSA_CODE_SYMBOL_INFO_MODULE_NAME_LENGTH = hsa_code_symbol_info_t.define('HSA_CODE_SYMBOL_INFO_MODULE_NAME_LENGTH', 3)
HSA_CODE_SYMBOL_INFO_MODULE_NAME = hsa_code_symbol_info_t.define('HSA_CODE_SYMBOL_INFO_MODULE_NAME', 4)
HSA_CODE_SYMBOL_INFO_LINKAGE = hsa_code_symbol_info_t.define('HSA_CODE_SYMBOL_INFO_LINKAGE', 5)
HSA_CODE_SYMBOL_INFO_IS_DEFINITION = hsa_code_symbol_info_t.define('HSA_CODE_SYMBOL_INFO_IS_DEFINITION', 17)
HSA_CODE_SYMBOL_INFO_VARIABLE_ALLOCATION = hsa_code_symbol_info_t.define('HSA_CODE_SYMBOL_INFO_VARIABLE_ALLOCATION', 6)
HSA_CODE_SYMBOL_INFO_VARIABLE_SEGMENT = hsa_code_symbol_info_t.define('HSA_CODE_SYMBOL_INFO_VARIABLE_SEGMENT', 7)
HSA_CODE_SYMBOL_INFO_VARIABLE_ALIGNMENT = hsa_code_symbol_info_t.define('HSA_CODE_SYMBOL_INFO_VARIABLE_ALIGNMENT', 8)
HSA_CODE_SYMBOL_INFO_VARIABLE_SIZE = hsa_code_symbol_info_t.define('HSA_CODE_SYMBOL_INFO_VARIABLE_SIZE', 9)
HSA_CODE_SYMBOL_INFO_VARIABLE_IS_CONST = hsa_code_symbol_info_t.define('HSA_CODE_SYMBOL_INFO_VARIABLE_IS_CONST', 10)
HSA_CODE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE = hsa_code_symbol_info_t.define('HSA_CODE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE', 11)
HSA_CODE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT = hsa_code_symbol_info_t.define('HSA_CODE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT', 12)
HSA_CODE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE = hsa_code_symbol_info_t.define('HSA_CODE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE', 13)
HSA_CODE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE = hsa_code_symbol_info_t.define('HSA_CODE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE', 14)
HSA_CODE_SYMBOL_INFO_KERNEL_DYNAMIC_CALLSTACK = hsa_code_symbol_info_t.define('HSA_CODE_SYMBOL_INFO_KERNEL_DYNAMIC_CALLSTACK', 15)
HSA_CODE_SYMBOL_INFO_KERNEL_CALL_CONVENTION = hsa_code_symbol_info_t.define('HSA_CODE_SYMBOL_INFO_KERNEL_CALL_CONVENTION', 18)
HSA_CODE_SYMBOL_INFO_INDIRECT_FUNCTION_CALL_CONVENTION = hsa_code_symbol_info_t.define('HSA_CODE_SYMBOL_INFO_INDIRECT_FUNCTION_CALL_CONVENTION', 16)
HSA_CODE_SYMBOL_INFO_KERNEL_WAVEFRONT_SIZE = hsa_code_symbol_info_t.define('HSA_CODE_SYMBOL_INFO_KERNEL_WAVEFRONT_SIZE', 19)

try: (hsa_code_symbol_get_info:=dll.hsa_code_symbol_get_info).restype, hsa_code_symbol_get_info.argtypes = hsa_status_t, [hsa_code_symbol_t, hsa_code_symbol_info_t, ctypes.c_void_p]
except AttributeError: pass

try: (hsa_code_object_iterate_symbols:=dll.hsa_code_object_iterate_symbols).restype, hsa_code_object_iterate_symbols.argtypes = hsa_status_t, [hsa_code_object_t, ctypes.CFUNCTYPE(hsa_status_t, hsa_code_object_t, hsa_code_symbol_t, ctypes.c_void_p), ctypes.c_void_p]
except AttributeError: pass

hsa_signal_condition32_t = ctypes.c_uint32
hsa_amd_packet_type_t = CEnum(ctypes.c_uint32)
HSA_AMD_PACKET_TYPE_BARRIER_VALUE = hsa_amd_packet_type_t.define('HSA_AMD_PACKET_TYPE_BARRIER_VALUE', 2)
HSA_AMD_PACKET_TYPE_AIE_ERT = hsa_amd_packet_type_t.define('HSA_AMD_PACKET_TYPE_AIE_ERT', 3)

hsa_amd_packet_type8_t = ctypes.c_ubyte
class struct_hsa_amd_packet_header_s(Struct): pass
uint8_t = ctypes.c_ubyte
struct_hsa_amd_packet_header_s._fields_ = [
  ('header', uint16_t),
  ('AmdFormat', hsa_amd_packet_type8_t),
  ('reserved', uint8_t),
]
hsa_amd_vendor_packet_header_t = struct_hsa_amd_packet_header_s
class struct_hsa_amd_barrier_value_packet_s(Struct): pass
struct_hsa_amd_barrier_value_packet_s._fields_ = [
  ('header', hsa_amd_vendor_packet_header_t),
  ('reserved0', uint32_t),
  ('signal', hsa_signal_t),
  ('value', hsa_signal_value_t),
  ('mask', hsa_signal_value_t),
  ('cond', hsa_signal_condition32_t),
  ('reserved1', uint32_t),
  ('reserved2', uint64_t),
  ('reserved3', uint64_t),
  ('completion_signal', hsa_signal_t),
]
hsa_amd_barrier_value_packet_t = struct_hsa_amd_barrier_value_packet_s
hsa_amd_aie_ert_state = CEnum(ctypes.c_uint32)
HSA_AMD_AIE_ERT_STATE_NEW = hsa_amd_aie_ert_state.define('HSA_AMD_AIE_ERT_STATE_NEW', 1)
HSA_AMD_AIE_ERT_STATE_QUEUED = hsa_amd_aie_ert_state.define('HSA_AMD_AIE_ERT_STATE_QUEUED', 2)
HSA_AMD_AIE_ERT_STATE_RUNNING = hsa_amd_aie_ert_state.define('HSA_AMD_AIE_ERT_STATE_RUNNING', 3)
HSA_AMD_AIE_ERT_STATE_COMPLETED = hsa_amd_aie_ert_state.define('HSA_AMD_AIE_ERT_STATE_COMPLETED', 4)
HSA_AMD_AIE_ERT_STATE_ERROR = hsa_amd_aie_ert_state.define('HSA_AMD_AIE_ERT_STATE_ERROR', 5)
HSA_AMD_AIE_ERT_STATE_ABORT = hsa_amd_aie_ert_state.define('HSA_AMD_AIE_ERT_STATE_ABORT', 6)
HSA_AMD_AIE_ERT_STATE_SUBMITTED = hsa_amd_aie_ert_state.define('HSA_AMD_AIE_ERT_STATE_SUBMITTED', 7)
HSA_AMD_AIE_ERT_STATE_TIMEOUT = hsa_amd_aie_ert_state.define('HSA_AMD_AIE_ERT_STATE_TIMEOUT', 8)
HSA_AMD_AIE_ERT_STATE_NORESPONSE = hsa_amd_aie_ert_state.define('HSA_AMD_AIE_ERT_STATE_NORESPONSE', 9)
HSA_AMD_AIE_ERT_STATE_SKERROR = hsa_amd_aie_ert_state.define('HSA_AMD_AIE_ERT_STATE_SKERROR', 10)
HSA_AMD_AIE_ERT_STATE_SKCRASHED = hsa_amd_aie_ert_state.define('HSA_AMD_AIE_ERT_STATE_SKCRASHED', 11)
HSA_AMD_AIE_ERT_STATE_MAX = hsa_amd_aie_ert_state.define('HSA_AMD_AIE_ERT_STATE_MAX', 12)

hsa_amd_aie_ert_cmd_opcode_t = CEnum(ctypes.c_uint32)
HSA_AMD_AIE_ERT_START_CU = hsa_amd_aie_ert_cmd_opcode_t.define('HSA_AMD_AIE_ERT_START_CU', 0)
HSA_AMD_AIE_ERT_START_KERNEL = hsa_amd_aie_ert_cmd_opcode_t.define('HSA_AMD_AIE_ERT_START_KERNEL', 0)
HSA_AMD_AIE_ERT_CONFIGURE = hsa_amd_aie_ert_cmd_opcode_t.define('HSA_AMD_AIE_ERT_CONFIGURE', 2)
HSA_AMD_AIE_ERT_EXIT = hsa_amd_aie_ert_cmd_opcode_t.define('HSA_AMD_AIE_ERT_EXIT', 3)
HSA_AMD_AIE_ERT_ABORT = hsa_amd_aie_ert_cmd_opcode_t.define('HSA_AMD_AIE_ERT_ABORT', 4)
HSA_AMD_AIE_ERT_EXEC_WRITE = hsa_amd_aie_ert_cmd_opcode_t.define('HSA_AMD_AIE_ERT_EXEC_WRITE', 5)
HSA_AMD_AIE_ERT_CU_STAT = hsa_amd_aie_ert_cmd_opcode_t.define('HSA_AMD_AIE_ERT_CU_STAT', 6)
HSA_AMD_AIE_ERT_START_COPYBO = hsa_amd_aie_ert_cmd_opcode_t.define('HSA_AMD_AIE_ERT_START_COPYBO', 7)
HSA_AMD_AIE_ERT_SK_CONFIG = hsa_amd_aie_ert_cmd_opcode_t.define('HSA_AMD_AIE_ERT_SK_CONFIG', 8)
HSA_AMD_AIE_ERT_SK_START = hsa_amd_aie_ert_cmd_opcode_t.define('HSA_AMD_AIE_ERT_SK_START', 9)
HSA_AMD_AIE_ERT_SK_UNCONFIG = hsa_amd_aie_ert_cmd_opcode_t.define('HSA_AMD_AIE_ERT_SK_UNCONFIG', 10)
HSA_AMD_AIE_ERT_INIT_CU = hsa_amd_aie_ert_cmd_opcode_t.define('HSA_AMD_AIE_ERT_INIT_CU', 11)
HSA_AMD_AIE_ERT_START_FA = hsa_amd_aie_ert_cmd_opcode_t.define('HSA_AMD_AIE_ERT_START_FA', 12)
HSA_AMD_AIE_ERT_CLK_CALIB = hsa_amd_aie_ert_cmd_opcode_t.define('HSA_AMD_AIE_ERT_CLK_CALIB', 13)
HSA_AMD_AIE_ERT_MB_VALIDATE = hsa_amd_aie_ert_cmd_opcode_t.define('HSA_AMD_AIE_ERT_MB_VALIDATE', 14)
HSA_AMD_AIE_ERT_START_KEY_VAL = hsa_amd_aie_ert_cmd_opcode_t.define('HSA_AMD_AIE_ERT_START_KEY_VAL', 15)
HSA_AMD_AIE_ERT_ACCESS_TEST_C = hsa_amd_aie_ert_cmd_opcode_t.define('HSA_AMD_AIE_ERT_ACCESS_TEST_C', 16)
HSA_AMD_AIE_ERT_ACCESS_TEST = hsa_amd_aie_ert_cmd_opcode_t.define('HSA_AMD_AIE_ERT_ACCESS_TEST', 17)
HSA_AMD_AIE_ERT_START_DPU = hsa_amd_aie_ert_cmd_opcode_t.define('HSA_AMD_AIE_ERT_START_DPU', 18)
HSA_AMD_AIE_ERT_CMD_CHAIN = hsa_amd_aie_ert_cmd_opcode_t.define('HSA_AMD_AIE_ERT_CMD_CHAIN', 19)
HSA_AMD_AIE_ERT_START_NPU = hsa_amd_aie_ert_cmd_opcode_t.define('HSA_AMD_AIE_ERT_START_NPU', 20)
HSA_AMD_AIE_ERT_START_NPU_PREEMPT = hsa_amd_aie_ert_cmd_opcode_t.define('HSA_AMD_AIE_ERT_START_NPU_PREEMPT', 21)

class struct_hsa_amd_aie_ert_start_kernel_data_s(Struct): pass
struct_hsa_amd_aie_ert_start_kernel_data_s._fields_ = [
  ('pdi_addr', ctypes.c_void_p),
  ('data', (uint32_t * 0)),
]
hsa_amd_aie_ert_start_kernel_data_t = struct_hsa_amd_aie_ert_start_kernel_data_s
class struct_hsa_amd_aie_ert_packet_s(Struct): pass
struct_hsa_amd_aie_ert_packet_s._fields_ = [
  ('header', hsa_amd_vendor_packet_header_t),
  ('state', uint32_t,4),
  ('custom', uint32_t,8),
  ('count', uint32_t,11),
  ('opcode', uint32_t,5),
  ('type', uint32_t,4),
  ('reserved0', uint64_t),
  ('reserved1', uint64_t),
  ('reserved2', uint64_t),
  ('reserved3', uint64_t),
  ('reserved4', uint64_t),
  ('reserved5', uint64_t),
  ('payload_data', uint64_t),
]
hsa_amd_aie_ert_packet_t = struct_hsa_amd_aie_ert_packet_s
_anonenum0 = CEnum(ctypes.c_uint32)
HSA_STATUS_ERROR_INVALID_MEMORY_POOL = _anonenum0.define('HSA_STATUS_ERROR_INVALID_MEMORY_POOL', 40)
HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION = _anonenum0.define('HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION', 41)
HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION = _anonenum0.define('HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION', 42)
HSA_STATUS_ERROR_MEMORY_FAULT = _anonenum0.define('HSA_STATUS_ERROR_MEMORY_FAULT', 43)
HSA_STATUS_CU_MASK_REDUCED = _anonenum0.define('HSA_STATUS_CU_MASK_REDUCED', 44)
HSA_STATUS_ERROR_OUT_OF_REGISTERS = _anonenum0.define('HSA_STATUS_ERROR_OUT_OF_REGISTERS', 45)
HSA_STATUS_ERROR_RESOURCE_BUSY = _anonenum0.define('HSA_STATUS_ERROR_RESOURCE_BUSY', 46)
HSA_STATUS_ERROR_NOT_SUPPORTED = _anonenum0.define('HSA_STATUS_ERROR_NOT_SUPPORTED', 47)

hsa_amd_iommu_version_t = CEnum(ctypes.c_uint32)
HSA_IOMMU_SUPPORT_NONE = hsa_amd_iommu_version_t.define('HSA_IOMMU_SUPPORT_NONE', 0)
HSA_IOMMU_SUPPORT_V2 = hsa_amd_iommu_version_t.define('HSA_IOMMU_SUPPORT_V2', 1)

class struct_hsa_amd_clock_counters_s(Struct): pass
struct_hsa_amd_clock_counters_s._fields_ = [
  ('gpu_clock_counter', uint64_t),
  ('cpu_clock_counter', uint64_t),
  ('system_clock_counter', uint64_t),
  ('system_clock_frequency', uint64_t),
]
hsa_amd_clock_counters_t = struct_hsa_amd_clock_counters_s
enum_hsa_amd_agent_info_s = CEnum(ctypes.c_uint32)
HSA_AMD_AGENT_INFO_CHIP_ID = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_CHIP_ID', 40960)
HSA_AMD_AGENT_INFO_CACHELINE_SIZE = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_CACHELINE_SIZE', 40961)
HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT', 40962)
HSA_AMD_AGENT_INFO_MAX_CLOCK_FREQUENCY = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_MAX_CLOCK_FREQUENCY', 40963)
HSA_AMD_AGENT_INFO_DRIVER_NODE_ID = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_DRIVER_NODE_ID', 40964)
HSA_AMD_AGENT_INFO_MAX_ADDRESS_WATCH_POINTS = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_MAX_ADDRESS_WATCH_POINTS', 40965)
HSA_AMD_AGENT_INFO_BDFID = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_BDFID', 40966)
HSA_AMD_AGENT_INFO_MEMORY_WIDTH = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_MEMORY_WIDTH', 40967)
HSA_AMD_AGENT_INFO_MEMORY_MAX_FREQUENCY = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_MEMORY_MAX_FREQUENCY', 40968)
HSA_AMD_AGENT_INFO_PRODUCT_NAME = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_PRODUCT_NAME', 40969)
HSA_AMD_AGENT_INFO_MAX_WAVES_PER_CU = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_MAX_WAVES_PER_CU', 40970)
HSA_AMD_AGENT_INFO_NUM_SIMDS_PER_CU = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_NUM_SIMDS_PER_CU', 40971)
HSA_AMD_AGENT_INFO_NUM_SHADER_ENGINES = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_NUM_SHADER_ENGINES', 40972)
HSA_AMD_AGENT_INFO_NUM_SHADER_ARRAYS_PER_SE = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_NUM_SHADER_ARRAYS_PER_SE', 40973)
HSA_AMD_AGENT_INFO_HDP_FLUSH = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_HDP_FLUSH', 40974)
HSA_AMD_AGENT_INFO_DOMAIN = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_DOMAIN', 40975)
HSA_AMD_AGENT_INFO_COOPERATIVE_QUEUES = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_COOPERATIVE_QUEUES', 40976)
HSA_AMD_AGENT_INFO_UUID = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_UUID', 40977)
HSA_AMD_AGENT_INFO_ASIC_REVISION = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_ASIC_REVISION', 40978)
HSA_AMD_AGENT_INFO_SVM_DIRECT_HOST_ACCESS = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_SVM_DIRECT_HOST_ACCESS', 40979)
HSA_AMD_AGENT_INFO_COOPERATIVE_COMPUTE_UNIT_COUNT = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_COOPERATIVE_COMPUTE_UNIT_COUNT', 40980)
HSA_AMD_AGENT_INFO_MEMORY_AVAIL = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_MEMORY_AVAIL', 40981)
HSA_AMD_AGENT_INFO_TIMESTAMP_FREQUENCY = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_TIMESTAMP_FREQUENCY', 40982)
HSA_AMD_AGENT_INFO_ASIC_FAMILY_ID = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_ASIC_FAMILY_ID', 41223)
HSA_AMD_AGENT_INFO_UCODE_VERSION = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_UCODE_VERSION', 41224)
HSA_AMD_AGENT_INFO_SDMA_UCODE_VERSION = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_SDMA_UCODE_VERSION', 41225)
HSA_AMD_AGENT_INFO_NUM_SDMA_ENG = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_NUM_SDMA_ENG', 41226)
HSA_AMD_AGENT_INFO_NUM_SDMA_XGMI_ENG = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_NUM_SDMA_XGMI_ENG', 41227)
HSA_AMD_AGENT_INFO_IOMMU_SUPPORT = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_IOMMU_SUPPORT', 41232)
HSA_AMD_AGENT_INFO_NUM_XCC = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_NUM_XCC', 41233)
HSA_AMD_AGENT_INFO_DRIVER_UID = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_DRIVER_UID', 41234)
HSA_AMD_AGENT_INFO_NEAREST_CPU = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_NEAREST_CPU', 41235)
HSA_AMD_AGENT_INFO_MEMORY_PROPERTIES = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_MEMORY_PROPERTIES', 41236)
HSA_AMD_AGENT_INFO_AQL_EXTENSIONS = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_AQL_EXTENSIONS', 41237)
HSA_AMD_AGENT_INFO_SCRATCH_LIMIT_MAX = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_SCRATCH_LIMIT_MAX', 41238)
HSA_AMD_AGENT_INFO_SCRATCH_LIMIT_CURRENT = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_SCRATCH_LIMIT_CURRENT', 41239)
HSA_AMD_AGENT_INFO_CLOCK_COUNTERS = enum_hsa_amd_agent_info_s.define('HSA_AMD_AGENT_INFO_CLOCK_COUNTERS', 41240)

hsa_amd_agent_info_t = enum_hsa_amd_agent_info_s
enum_hsa_amd_agent_memory_properties_s = CEnum(ctypes.c_uint32)
HSA_AMD_MEMORY_PROPERTY_AGENT_IS_APU = enum_hsa_amd_agent_memory_properties_s.define('HSA_AMD_MEMORY_PROPERTY_AGENT_IS_APU', 1)

hsa_amd_agent_memory_properties_t = enum_hsa_amd_agent_memory_properties_s
enum_hsa_amd_sdma_engine_id = CEnum(ctypes.c_uint32)
HSA_AMD_SDMA_ENGINE_0 = enum_hsa_amd_sdma_engine_id.define('HSA_AMD_SDMA_ENGINE_0', 1)
HSA_AMD_SDMA_ENGINE_1 = enum_hsa_amd_sdma_engine_id.define('HSA_AMD_SDMA_ENGINE_1', 2)
HSA_AMD_SDMA_ENGINE_2 = enum_hsa_amd_sdma_engine_id.define('HSA_AMD_SDMA_ENGINE_2', 4)
HSA_AMD_SDMA_ENGINE_3 = enum_hsa_amd_sdma_engine_id.define('HSA_AMD_SDMA_ENGINE_3', 8)
HSA_AMD_SDMA_ENGINE_4 = enum_hsa_amd_sdma_engine_id.define('HSA_AMD_SDMA_ENGINE_4', 16)
HSA_AMD_SDMA_ENGINE_5 = enum_hsa_amd_sdma_engine_id.define('HSA_AMD_SDMA_ENGINE_5', 32)
HSA_AMD_SDMA_ENGINE_6 = enum_hsa_amd_sdma_engine_id.define('HSA_AMD_SDMA_ENGINE_6', 64)
HSA_AMD_SDMA_ENGINE_7 = enum_hsa_amd_sdma_engine_id.define('HSA_AMD_SDMA_ENGINE_7', 128)
HSA_AMD_SDMA_ENGINE_8 = enum_hsa_amd_sdma_engine_id.define('HSA_AMD_SDMA_ENGINE_8', 256)
HSA_AMD_SDMA_ENGINE_9 = enum_hsa_amd_sdma_engine_id.define('HSA_AMD_SDMA_ENGINE_9', 512)
HSA_AMD_SDMA_ENGINE_10 = enum_hsa_amd_sdma_engine_id.define('HSA_AMD_SDMA_ENGINE_10', 1024)
HSA_AMD_SDMA_ENGINE_11 = enum_hsa_amd_sdma_engine_id.define('HSA_AMD_SDMA_ENGINE_11', 2048)
HSA_AMD_SDMA_ENGINE_12 = enum_hsa_amd_sdma_engine_id.define('HSA_AMD_SDMA_ENGINE_12', 4096)
HSA_AMD_SDMA_ENGINE_13 = enum_hsa_amd_sdma_engine_id.define('HSA_AMD_SDMA_ENGINE_13', 8192)
HSA_AMD_SDMA_ENGINE_14 = enum_hsa_amd_sdma_engine_id.define('HSA_AMD_SDMA_ENGINE_14', 16384)
HSA_AMD_SDMA_ENGINE_15 = enum_hsa_amd_sdma_engine_id.define('HSA_AMD_SDMA_ENGINE_15', 32768)

hsa_amd_sdma_engine_id_t = enum_hsa_amd_sdma_engine_id
class struct_hsa_amd_hdp_flush_s(Struct): pass
struct_hsa_amd_hdp_flush_s._fields_ = [
  ('HDP_MEM_FLUSH_CNTL', ctypes.POINTER(uint32_t)),
  ('HDP_REG_FLUSH_CNTL', ctypes.POINTER(uint32_t)),
]
hsa_amd_hdp_flush_t = struct_hsa_amd_hdp_flush_s
enum_hsa_amd_region_info_s = CEnum(ctypes.c_uint32)
HSA_AMD_REGION_INFO_HOST_ACCESSIBLE = enum_hsa_amd_region_info_s.define('HSA_AMD_REGION_INFO_HOST_ACCESSIBLE', 40960)
HSA_AMD_REGION_INFO_BASE = enum_hsa_amd_region_info_s.define('HSA_AMD_REGION_INFO_BASE', 40961)
HSA_AMD_REGION_INFO_BUS_WIDTH = enum_hsa_amd_region_info_s.define('HSA_AMD_REGION_INFO_BUS_WIDTH', 40962)
HSA_AMD_REGION_INFO_MAX_CLOCK_FREQUENCY = enum_hsa_amd_region_info_s.define('HSA_AMD_REGION_INFO_MAX_CLOCK_FREQUENCY', 40963)

hsa_amd_region_info_t = enum_hsa_amd_region_info_s
enum_hsa_amd_coherency_type_s = CEnum(ctypes.c_uint32)
HSA_AMD_COHERENCY_TYPE_COHERENT = enum_hsa_amd_coherency_type_s.define('HSA_AMD_COHERENCY_TYPE_COHERENT', 0)
HSA_AMD_COHERENCY_TYPE_NONCOHERENT = enum_hsa_amd_coherency_type_s.define('HSA_AMD_COHERENCY_TYPE_NONCOHERENT', 1)

hsa_amd_coherency_type_t = enum_hsa_amd_coherency_type_s
enum_hsa_amd_dma_buf_mapping_type_s = CEnum(ctypes.c_uint32)
HSA_AMD_DMABUF_MAPPING_TYPE_NONE = enum_hsa_amd_dma_buf_mapping_type_s.define('HSA_AMD_DMABUF_MAPPING_TYPE_NONE', 0)
HSA_AMD_DMABUF_MAPPING_TYPE_PCIE = enum_hsa_amd_dma_buf_mapping_type_s.define('HSA_AMD_DMABUF_MAPPING_TYPE_PCIE', 1)

hsa_amd_dma_buf_mapping_type_t = enum_hsa_amd_dma_buf_mapping_type_s
try: (hsa_amd_coherency_get_type:=dll.hsa_amd_coherency_get_type).restype, hsa_amd_coherency_get_type.argtypes = hsa_status_t, [hsa_agent_t, ctypes.POINTER(hsa_amd_coherency_type_t)]
except AttributeError: pass

try: (hsa_amd_coherency_set_type:=dll.hsa_amd_coherency_set_type).restype, hsa_amd_coherency_set_type.argtypes = hsa_status_t, [hsa_agent_t, hsa_amd_coherency_type_t]
except AttributeError: pass

class struct_hsa_amd_profiling_dispatch_time_s(Struct): pass
struct_hsa_amd_profiling_dispatch_time_s._fields_ = [
  ('start', uint64_t),
  ('end', uint64_t),
]
hsa_amd_profiling_dispatch_time_t = struct_hsa_amd_profiling_dispatch_time_s
class struct_hsa_amd_profiling_async_copy_time_s(Struct): pass
struct_hsa_amd_profiling_async_copy_time_s._fields_ = [
  ('start', uint64_t),
  ('end', uint64_t),
]
hsa_amd_profiling_async_copy_time_t = struct_hsa_amd_profiling_async_copy_time_s
try: (hsa_amd_profiling_set_profiler_enabled:=dll.hsa_amd_profiling_set_profiler_enabled).restype, hsa_amd_profiling_set_profiler_enabled.argtypes = hsa_status_t, [ctypes.POINTER(hsa_queue_t), ctypes.c_int32]
except AttributeError: pass

try: (hsa_amd_profiling_async_copy_enable:=dll.hsa_amd_profiling_async_copy_enable).restype, hsa_amd_profiling_async_copy_enable.argtypes = hsa_status_t, [ctypes.c_bool]
except AttributeError: pass

try: (hsa_amd_profiling_get_dispatch_time:=dll.hsa_amd_profiling_get_dispatch_time).restype, hsa_amd_profiling_get_dispatch_time.argtypes = hsa_status_t, [hsa_agent_t, hsa_signal_t, ctypes.POINTER(hsa_amd_profiling_dispatch_time_t)]
except AttributeError: pass

try: (hsa_amd_profiling_get_async_copy_time:=dll.hsa_amd_profiling_get_async_copy_time).restype, hsa_amd_profiling_get_async_copy_time.argtypes = hsa_status_t, [hsa_signal_t, ctypes.POINTER(hsa_amd_profiling_async_copy_time_t)]
except AttributeError: pass

try: (hsa_amd_profiling_convert_tick_to_system_domain:=dll.hsa_amd_profiling_convert_tick_to_system_domain).restype, hsa_amd_profiling_convert_tick_to_system_domain.argtypes = hsa_status_t, [hsa_agent_t, uint64_t, ctypes.POINTER(uint64_t)]
except AttributeError: pass

hsa_amd_signal_attribute_t = CEnum(ctypes.c_uint32)
HSA_AMD_SIGNAL_AMD_GPU_ONLY = hsa_amd_signal_attribute_t.define('HSA_AMD_SIGNAL_AMD_GPU_ONLY', 1)
HSA_AMD_SIGNAL_IPC = hsa_amd_signal_attribute_t.define('HSA_AMD_SIGNAL_IPC', 2)

try: (hsa_amd_signal_create:=dll.hsa_amd_signal_create).restype, hsa_amd_signal_create.argtypes = hsa_status_t, [hsa_signal_value_t, uint32_t, ctypes.POINTER(hsa_agent_t), uint64_t, ctypes.POINTER(hsa_signal_t)]
except AttributeError: pass

try: (hsa_amd_signal_value_pointer:=dll.hsa_amd_signal_value_pointer).restype, hsa_amd_signal_value_pointer.argtypes = hsa_status_t, [hsa_signal_t, ctypes.POINTER(ctypes.POINTER(hsa_signal_value_t))]
except AttributeError: pass

hsa_amd_signal_handler = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_int64, ctypes.c_void_p)
try: (hsa_amd_signal_async_handler:=dll.hsa_amd_signal_async_handler).restype, hsa_amd_signal_async_handler.argtypes = hsa_status_t, [hsa_signal_t, hsa_signal_condition_t, hsa_signal_value_t, hsa_amd_signal_handler, ctypes.c_void_p]
except AttributeError: pass

try: (hsa_amd_signal_wait_all:=dll.hsa_amd_signal_wait_all).restype, hsa_amd_signal_wait_all.argtypes = uint32_t, [uint32_t, ctypes.POINTER(hsa_signal_t), ctypes.POINTER(hsa_signal_condition_t), ctypes.POINTER(hsa_signal_value_t), uint64_t, hsa_wait_state_t, ctypes.POINTER(hsa_signal_value_t)]
except AttributeError: pass

try: (hsa_amd_signal_wait_any:=dll.hsa_amd_signal_wait_any).restype, hsa_amd_signal_wait_any.argtypes = uint32_t, [uint32_t, ctypes.POINTER(hsa_signal_t), ctypes.POINTER(hsa_signal_condition_t), ctypes.POINTER(hsa_signal_value_t), uint64_t, hsa_wait_state_t, ctypes.POINTER(hsa_signal_value_t)]
except AttributeError: pass

try: (hsa_amd_async_function:=dll.hsa_amd_async_function).restype, hsa_amd_async_function.argtypes = hsa_status_t, [ctypes.CFUNCTYPE(None, ctypes.c_void_p), ctypes.c_void_p]
except AttributeError: pass

class struct_hsa_amd_image_descriptor_s(Struct): pass
struct_hsa_amd_image_descriptor_s._fields_ = [
  ('version', uint32_t),
  ('deviceID', uint32_t),
  ('data', (uint32_t * 1)),
]
hsa_amd_image_descriptor_t = struct_hsa_amd_image_descriptor_s
class struct_hsa_ext_image_descriptor_s(Struct): pass
hsa_ext_image_descriptor_t = struct_hsa_ext_image_descriptor_s
hsa_ext_image_geometry_t = CEnum(ctypes.c_uint32)
HSA_EXT_IMAGE_GEOMETRY_1D = hsa_ext_image_geometry_t.define('HSA_EXT_IMAGE_GEOMETRY_1D', 0)
HSA_EXT_IMAGE_GEOMETRY_2D = hsa_ext_image_geometry_t.define('HSA_EXT_IMAGE_GEOMETRY_2D', 1)
HSA_EXT_IMAGE_GEOMETRY_3D = hsa_ext_image_geometry_t.define('HSA_EXT_IMAGE_GEOMETRY_3D', 2)
HSA_EXT_IMAGE_GEOMETRY_1DA = hsa_ext_image_geometry_t.define('HSA_EXT_IMAGE_GEOMETRY_1DA', 3)
HSA_EXT_IMAGE_GEOMETRY_2DA = hsa_ext_image_geometry_t.define('HSA_EXT_IMAGE_GEOMETRY_2DA', 4)
HSA_EXT_IMAGE_GEOMETRY_1DB = hsa_ext_image_geometry_t.define('HSA_EXT_IMAGE_GEOMETRY_1DB', 5)
HSA_EXT_IMAGE_GEOMETRY_2DDEPTH = hsa_ext_image_geometry_t.define('HSA_EXT_IMAGE_GEOMETRY_2DDEPTH', 6)
HSA_EXT_IMAGE_GEOMETRY_2DADEPTH = hsa_ext_image_geometry_t.define('HSA_EXT_IMAGE_GEOMETRY_2DADEPTH', 7)

class struct_hsa_ext_image_format_s(Struct): pass
hsa_ext_image_format_t = struct_hsa_ext_image_format_s
hsa_ext_image_channel_type32_t = ctypes.c_uint32
hsa_ext_image_channel_order32_t = ctypes.c_uint32
struct_hsa_ext_image_format_s._fields_ = [
  ('channel_type', hsa_ext_image_channel_type32_t),
  ('channel_order', hsa_ext_image_channel_order32_t),
]
struct_hsa_ext_image_descriptor_s._fields_ = [
  ('geometry', hsa_ext_image_geometry_t),
  ('width', size_t),
  ('height', size_t),
  ('depth', size_t),
  ('array_size', size_t),
  ('format', hsa_ext_image_format_t),
]
class struct_hsa_ext_image_s(Struct): pass
hsa_ext_image_t = struct_hsa_ext_image_s
struct_hsa_ext_image_s._fields_ = [
  ('handle', uint64_t),
]
try: (hsa_amd_image_create:=dll.hsa_amd_image_create).restype, hsa_amd_image_create.argtypes = hsa_status_t, [hsa_agent_t, ctypes.POINTER(hsa_ext_image_descriptor_t), ctypes.POINTER(hsa_amd_image_descriptor_t), ctypes.c_void_p, hsa_access_permission_t, ctypes.POINTER(hsa_ext_image_t)]
except AttributeError: pass

try: (hsa_amd_image_get_info_max_dim:=dll.hsa_amd_image_get_info_max_dim).restype, hsa_amd_image_get_info_max_dim.argtypes = hsa_status_t, [hsa_agent_t, hsa_agent_info_t, ctypes.c_void_p]
except AttributeError: pass

try: (hsa_amd_queue_cu_set_mask:=dll.hsa_amd_queue_cu_set_mask).restype, hsa_amd_queue_cu_set_mask.argtypes = hsa_status_t, [ctypes.POINTER(hsa_queue_t), uint32_t, ctypes.POINTER(uint32_t)]
except AttributeError: pass

try: (hsa_amd_queue_cu_get_mask:=dll.hsa_amd_queue_cu_get_mask).restype, hsa_amd_queue_cu_get_mask.argtypes = hsa_status_t, [ctypes.POINTER(hsa_queue_t), uint32_t, ctypes.POINTER(uint32_t)]
except AttributeError: pass

hsa_amd_segment_t = CEnum(ctypes.c_uint32)
HSA_AMD_SEGMENT_GLOBAL = hsa_amd_segment_t.define('HSA_AMD_SEGMENT_GLOBAL', 0)
HSA_AMD_SEGMENT_READONLY = hsa_amd_segment_t.define('HSA_AMD_SEGMENT_READONLY', 1)
HSA_AMD_SEGMENT_PRIVATE = hsa_amd_segment_t.define('HSA_AMD_SEGMENT_PRIVATE', 2)
HSA_AMD_SEGMENT_GROUP = hsa_amd_segment_t.define('HSA_AMD_SEGMENT_GROUP', 3)

class struct_hsa_amd_memory_pool_s(Struct): pass
struct_hsa_amd_memory_pool_s._fields_ = [
  ('handle', uint64_t),
]
hsa_amd_memory_pool_t = struct_hsa_amd_memory_pool_s
enum_hsa_amd_memory_pool_global_flag_s = CEnum(ctypes.c_uint32)
HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT = enum_hsa_amd_memory_pool_global_flag_s.define('HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT', 1)
HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED = enum_hsa_amd_memory_pool_global_flag_s.define('HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED', 2)
HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED = enum_hsa_amd_memory_pool_global_flag_s.define('HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED', 4)
HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_EXTENDED_SCOPE_FINE_GRAINED = enum_hsa_amd_memory_pool_global_flag_s.define('HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_EXTENDED_SCOPE_FINE_GRAINED', 8)

hsa_amd_memory_pool_global_flag_t = enum_hsa_amd_memory_pool_global_flag_s
enum_hsa_amd_memory_pool_location_s = CEnum(ctypes.c_uint32)
HSA_AMD_MEMORY_POOL_LOCATION_CPU = enum_hsa_amd_memory_pool_location_s.define('HSA_AMD_MEMORY_POOL_LOCATION_CPU', 0)
HSA_AMD_MEMORY_POOL_LOCATION_GPU = enum_hsa_amd_memory_pool_location_s.define('HSA_AMD_MEMORY_POOL_LOCATION_GPU', 1)

hsa_amd_memory_pool_location_t = enum_hsa_amd_memory_pool_location_s
hsa_amd_memory_pool_info_t = CEnum(ctypes.c_uint32)
HSA_AMD_MEMORY_POOL_INFO_SEGMENT = hsa_amd_memory_pool_info_t.define('HSA_AMD_MEMORY_POOL_INFO_SEGMENT', 0)
HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS = hsa_amd_memory_pool_info_t.define('HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS', 1)
HSA_AMD_MEMORY_POOL_INFO_SIZE = hsa_amd_memory_pool_info_t.define('HSA_AMD_MEMORY_POOL_INFO_SIZE', 2)
HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED = hsa_amd_memory_pool_info_t.define('HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED', 5)
HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE = hsa_amd_memory_pool_info_t.define('HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE', 6)
HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALIGNMENT = hsa_amd_memory_pool_info_t.define('HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALIGNMENT', 7)
HSA_AMD_MEMORY_POOL_INFO_ACCESSIBLE_BY_ALL = hsa_amd_memory_pool_info_t.define('HSA_AMD_MEMORY_POOL_INFO_ACCESSIBLE_BY_ALL', 15)
HSA_AMD_MEMORY_POOL_INFO_ALLOC_MAX_SIZE = hsa_amd_memory_pool_info_t.define('HSA_AMD_MEMORY_POOL_INFO_ALLOC_MAX_SIZE', 16)
HSA_AMD_MEMORY_POOL_INFO_LOCATION = hsa_amd_memory_pool_info_t.define('HSA_AMD_MEMORY_POOL_INFO_LOCATION', 17)
HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_REC_GRANULE = hsa_amd_memory_pool_info_t.define('HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_REC_GRANULE', 18)

enum_hsa_amd_memory_pool_flag_s = CEnum(ctypes.c_uint32)
HSA_AMD_MEMORY_POOL_STANDARD_FLAG = enum_hsa_amd_memory_pool_flag_s.define('HSA_AMD_MEMORY_POOL_STANDARD_FLAG', 0)
HSA_AMD_MEMORY_POOL_PCIE_FLAG = enum_hsa_amd_memory_pool_flag_s.define('HSA_AMD_MEMORY_POOL_PCIE_FLAG', 1)
HSA_AMD_MEMORY_POOL_CONTIGUOUS_FLAG = enum_hsa_amd_memory_pool_flag_s.define('HSA_AMD_MEMORY_POOL_CONTIGUOUS_FLAG', 2)
HSA_AMD_MEMORY_POOL_EXECUTABLE_FLAG = enum_hsa_amd_memory_pool_flag_s.define('HSA_AMD_MEMORY_POOL_EXECUTABLE_FLAG', 4)
HSA_AMD_MEMORY_POOL_UNCACHED_FLAG = enum_hsa_amd_memory_pool_flag_s.define('HSA_AMD_MEMORY_POOL_UNCACHED_FLAG', 8)

hsa_amd_memory_pool_flag_t = enum_hsa_amd_memory_pool_flag_s
try: (hsa_amd_memory_pool_get_info:=dll.hsa_amd_memory_pool_get_info).restype, hsa_amd_memory_pool_get_info.argtypes = hsa_status_t, [hsa_amd_memory_pool_t, hsa_amd_memory_pool_info_t, ctypes.c_void_p]
except AttributeError: pass

try: (hsa_amd_agent_iterate_memory_pools:=dll.hsa_amd_agent_iterate_memory_pools).restype, hsa_amd_agent_iterate_memory_pools.argtypes = hsa_status_t, [hsa_agent_t, ctypes.CFUNCTYPE(hsa_status_t, hsa_amd_memory_pool_t, ctypes.c_void_p), ctypes.c_void_p]
except AttributeError: pass

try: (hsa_amd_memory_pool_allocate:=dll.hsa_amd_memory_pool_allocate).restype, hsa_amd_memory_pool_allocate.argtypes = hsa_status_t, [hsa_amd_memory_pool_t, size_t, uint32_t, ctypes.POINTER(ctypes.c_void_p)]
except AttributeError: pass

try: (hsa_amd_memory_pool_free:=dll.hsa_amd_memory_pool_free).restype, hsa_amd_memory_pool_free.argtypes = hsa_status_t, [ctypes.c_void_p]
except AttributeError: pass

try: (hsa_amd_memory_async_copy:=dll.hsa_amd_memory_async_copy).restype, hsa_amd_memory_async_copy.argtypes = hsa_status_t, [ctypes.c_void_p, hsa_agent_t, ctypes.c_void_p, hsa_agent_t, size_t, uint32_t, ctypes.POINTER(hsa_signal_t), hsa_signal_t]
except AttributeError: pass

try: (hsa_amd_memory_async_copy_on_engine:=dll.hsa_amd_memory_async_copy_on_engine).restype, hsa_amd_memory_async_copy_on_engine.argtypes = hsa_status_t, [ctypes.c_void_p, hsa_agent_t, ctypes.c_void_p, hsa_agent_t, size_t, uint32_t, ctypes.POINTER(hsa_signal_t), hsa_signal_t, hsa_amd_sdma_engine_id_t, ctypes.c_bool]
except AttributeError: pass

try: (hsa_amd_memory_copy_engine_status:=dll.hsa_amd_memory_copy_engine_status).restype, hsa_amd_memory_copy_engine_status.argtypes = hsa_status_t, [hsa_agent_t, hsa_agent_t, ctypes.POINTER(uint32_t)]
except AttributeError: pass

try: (hsa_amd_memory_get_preferred_copy_engine:=dll.hsa_amd_memory_get_preferred_copy_engine).restype, hsa_amd_memory_get_preferred_copy_engine.argtypes = hsa_status_t, [hsa_agent_t, hsa_agent_t, ctypes.POINTER(uint32_t)]
except AttributeError: pass

class struct_hsa_pitched_ptr_s(Struct): pass
struct_hsa_pitched_ptr_s._fields_ = [
  ('base', ctypes.c_void_p),
  ('pitch', size_t),
  ('slice', size_t),
]
hsa_pitched_ptr_t = struct_hsa_pitched_ptr_s
hsa_amd_copy_direction_t = CEnum(ctypes.c_uint32)
hsaHostToHost = hsa_amd_copy_direction_t.define('hsaHostToHost', 0)
hsaHostToDevice = hsa_amd_copy_direction_t.define('hsaHostToDevice', 1)
hsaDeviceToHost = hsa_amd_copy_direction_t.define('hsaDeviceToHost', 2)
hsaDeviceToDevice = hsa_amd_copy_direction_t.define('hsaDeviceToDevice', 3)

try: (hsa_amd_memory_async_copy_rect:=dll.hsa_amd_memory_async_copy_rect).restype, hsa_amd_memory_async_copy_rect.argtypes = hsa_status_t, [ctypes.POINTER(hsa_pitched_ptr_t), ctypes.POINTER(hsa_dim3_t), ctypes.POINTER(hsa_pitched_ptr_t), ctypes.POINTER(hsa_dim3_t), ctypes.POINTER(hsa_dim3_t), hsa_agent_t, hsa_amd_copy_direction_t, uint32_t, ctypes.POINTER(hsa_signal_t), hsa_signal_t]
except AttributeError: pass

hsa_amd_memory_pool_access_t = CEnum(ctypes.c_uint32)
HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED = hsa_amd_memory_pool_access_t.define('HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED', 0)
HSA_AMD_MEMORY_POOL_ACCESS_ALLOWED_BY_DEFAULT = hsa_amd_memory_pool_access_t.define('HSA_AMD_MEMORY_POOL_ACCESS_ALLOWED_BY_DEFAULT', 1)
HSA_AMD_MEMORY_POOL_ACCESS_DISALLOWED_BY_DEFAULT = hsa_amd_memory_pool_access_t.define('HSA_AMD_MEMORY_POOL_ACCESS_DISALLOWED_BY_DEFAULT', 2)

hsa_amd_link_info_type_t = CEnum(ctypes.c_uint32)
HSA_AMD_LINK_INFO_TYPE_HYPERTRANSPORT = hsa_amd_link_info_type_t.define('HSA_AMD_LINK_INFO_TYPE_HYPERTRANSPORT', 0)
HSA_AMD_LINK_INFO_TYPE_QPI = hsa_amd_link_info_type_t.define('HSA_AMD_LINK_INFO_TYPE_QPI', 1)
HSA_AMD_LINK_INFO_TYPE_PCIE = hsa_amd_link_info_type_t.define('HSA_AMD_LINK_INFO_TYPE_PCIE', 2)
HSA_AMD_LINK_INFO_TYPE_INFINBAND = hsa_amd_link_info_type_t.define('HSA_AMD_LINK_INFO_TYPE_INFINBAND', 3)
HSA_AMD_LINK_INFO_TYPE_XGMI = hsa_amd_link_info_type_t.define('HSA_AMD_LINK_INFO_TYPE_XGMI', 4)

class struct_hsa_amd_memory_pool_link_info_s(Struct): pass
struct_hsa_amd_memory_pool_link_info_s._fields_ = [
  ('min_latency', uint32_t),
  ('max_latency', uint32_t),
  ('min_bandwidth', uint32_t),
  ('max_bandwidth', uint32_t),
  ('atomic_support_32bit', ctypes.c_bool),
  ('atomic_support_64bit', ctypes.c_bool),
  ('coherent_support', ctypes.c_bool),
  ('link_type', hsa_amd_link_info_type_t),
  ('numa_distance', uint32_t),
]
hsa_amd_memory_pool_link_info_t = struct_hsa_amd_memory_pool_link_info_s
hsa_amd_agent_memory_pool_info_t = CEnum(ctypes.c_uint32)
HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS = hsa_amd_agent_memory_pool_info_t.define('HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS', 0)
HSA_AMD_AGENT_MEMORY_POOL_INFO_NUM_LINK_HOPS = hsa_amd_agent_memory_pool_info_t.define('HSA_AMD_AGENT_MEMORY_POOL_INFO_NUM_LINK_HOPS', 1)
HSA_AMD_AGENT_MEMORY_POOL_INFO_LINK_INFO = hsa_amd_agent_memory_pool_info_t.define('HSA_AMD_AGENT_MEMORY_POOL_INFO_LINK_INFO', 2)

try: (hsa_amd_agent_memory_pool_get_info:=dll.hsa_amd_agent_memory_pool_get_info).restype, hsa_amd_agent_memory_pool_get_info.argtypes = hsa_status_t, [hsa_agent_t, hsa_amd_memory_pool_t, hsa_amd_agent_memory_pool_info_t, ctypes.c_void_p]
except AttributeError: pass

try: (hsa_amd_agents_allow_access:=dll.hsa_amd_agents_allow_access).restype, hsa_amd_agents_allow_access.argtypes = hsa_status_t, [uint32_t, ctypes.POINTER(hsa_agent_t), ctypes.POINTER(uint32_t), ctypes.c_void_p]
except AttributeError: pass

try: (hsa_amd_memory_pool_can_migrate:=dll.hsa_amd_memory_pool_can_migrate).restype, hsa_amd_memory_pool_can_migrate.argtypes = hsa_status_t, [hsa_amd_memory_pool_t, hsa_amd_memory_pool_t, ctypes.POINTER(ctypes.c_bool)]
except AttributeError: pass

try: (hsa_amd_memory_migrate:=dll.hsa_amd_memory_migrate).restype, hsa_amd_memory_migrate.argtypes = hsa_status_t, [ctypes.c_void_p, hsa_amd_memory_pool_t, uint32_t]
except AttributeError: pass

try: (hsa_amd_memory_lock:=dll.hsa_amd_memory_lock).restype, hsa_amd_memory_lock.argtypes = hsa_status_t, [ctypes.c_void_p, size_t, ctypes.POINTER(hsa_agent_t), ctypes.c_int32, ctypes.POINTER(ctypes.c_void_p)]
except AttributeError: pass

try: (hsa_amd_memory_lock_to_pool:=dll.hsa_amd_memory_lock_to_pool).restype, hsa_amd_memory_lock_to_pool.argtypes = hsa_status_t, [ctypes.c_void_p, size_t, ctypes.POINTER(hsa_agent_t), ctypes.c_int32, hsa_amd_memory_pool_t, uint32_t, ctypes.POINTER(ctypes.c_void_p)]
except AttributeError: pass

try: (hsa_amd_memory_unlock:=dll.hsa_amd_memory_unlock).restype, hsa_amd_memory_unlock.argtypes = hsa_status_t, [ctypes.c_void_p]
except AttributeError: pass

try: (hsa_amd_memory_fill:=dll.hsa_amd_memory_fill).restype, hsa_amd_memory_fill.argtypes = hsa_status_t, [ctypes.c_void_p, uint32_t, size_t]
except AttributeError: pass

try: (hsa_amd_interop_map_buffer:=dll.hsa_amd_interop_map_buffer).restype, hsa_amd_interop_map_buffer.argtypes = hsa_status_t, [uint32_t, ctypes.POINTER(hsa_agent_t), ctypes.c_int32, uint32_t, ctypes.POINTER(size_t), ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(size_t), ctypes.POINTER(ctypes.c_void_p)]
except AttributeError: pass

try: (hsa_amd_interop_unmap_buffer:=dll.hsa_amd_interop_unmap_buffer).restype, hsa_amd_interop_unmap_buffer.argtypes = hsa_status_t, [ctypes.c_void_p]
except AttributeError: pass

hsa_amd_pointer_type_t = CEnum(ctypes.c_uint32)
HSA_EXT_POINTER_TYPE_UNKNOWN = hsa_amd_pointer_type_t.define('HSA_EXT_POINTER_TYPE_UNKNOWN', 0)
HSA_EXT_POINTER_TYPE_HSA = hsa_amd_pointer_type_t.define('HSA_EXT_POINTER_TYPE_HSA', 1)
HSA_EXT_POINTER_TYPE_LOCKED = hsa_amd_pointer_type_t.define('HSA_EXT_POINTER_TYPE_LOCKED', 2)
HSA_EXT_POINTER_TYPE_GRAPHICS = hsa_amd_pointer_type_t.define('HSA_EXT_POINTER_TYPE_GRAPHICS', 3)
HSA_EXT_POINTER_TYPE_IPC = hsa_amd_pointer_type_t.define('HSA_EXT_POINTER_TYPE_IPC', 4)
HSA_EXT_POINTER_TYPE_RESERVED_ADDR = hsa_amd_pointer_type_t.define('HSA_EXT_POINTER_TYPE_RESERVED_ADDR', 5)
HSA_EXT_POINTER_TYPE_HSA_VMEM = hsa_amd_pointer_type_t.define('HSA_EXT_POINTER_TYPE_HSA_VMEM', 6)

class struct_hsa_amd_pointer_info_s(Struct): pass
struct_hsa_amd_pointer_info_s._fields_ = [
  ('size', uint32_t),
  ('type', hsa_amd_pointer_type_t),
  ('agentBaseAddress', ctypes.c_void_p),
  ('hostBaseAddress', ctypes.c_void_p),
  ('sizeInBytes', size_t),
  ('userData', ctypes.c_void_p),
  ('agentOwner', hsa_agent_t),
  ('global_flags', uint32_t),
  ('registered', ctypes.c_bool),
]
hsa_amd_pointer_info_t = struct_hsa_amd_pointer_info_s
try: (hsa_amd_pointer_info:=dll.hsa_amd_pointer_info).restype, hsa_amd_pointer_info.argtypes = hsa_status_t, [ctypes.c_void_p, ctypes.POINTER(hsa_amd_pointer_info_t), ctypes.CFUNCTYPE(ctypes.c_void_p, size_t), ctypes.POINTER(uint32_t), ctypes.POINTER(ctypes.POINTER(hsa_agent_t))]
except AttributeError: pass

try: (hsa_amd_pointer_info_set_userdata:=dll.hsa_amd_pointer_info_set_userdata).restype, hsa_amd_pointer_info_set_userdata.argtypes = hsa_status_t, [ctypes.c_void_p, ctypes.c_void_p]
except AttributeError: pass

class struct_hsa_amd_ipc_memory_s(Struct): pass
struct_hsa_amd_ipc_memory_s._fields_ = [
  ('handle', (uint32_t * 8)),
]
hsa_amd_ipc_memory_t = struct_hsa_amd_ipc_memory_s
try: (hsa_amd_ipc_memory_create:=dll.hsa_amd_ipc_memory_create).restype, hsa_amd_ipc_memory_create.argtypes = hsa_status_t, [ctypes.c_void_p, size_t, ctypes.POINTER(hsa_amd_ipc_memory_t)]
except AttributeError: pass

try: (hsa_amd_ipc_memory_attach:=dll.hsa_amd_ipc_memory_attach).restype, hsa_amd_ipc_memory_attach.argtypes = hsa_status_t, [ctypes.POINTER(hsa_amd_ipc_memory_t), size_t, uint32_t, ctypes.POINTER(hsa_agent_t), ctypes.POINTER(ctypes.c_void_p)]
except AttributeError: pass

try: (hsa_amd_ipc_memory_detach:=dll.hsa_amd_ipc_memory_detach).restype, hsa_amd_ipc_memory_detach.argtypes = hsa_status_t, [ctypes.c_void_p]
except AttributeError: pass

hsa_amd_ipc_signal_t = struct_hsa_amd_ipc_memory_s
try: (hsa_amd_ipc_signal_create:=dll.hsa_amd_ipc_signal_create).restype, hsa_amd_ipc_signal_create.argtypes = hsa_status_t, [hsa_signal_t, ctypes.POINTER(hsa_amd_ipc_signal_t)]
except AttributeError: pass

try: (hsa_amd_ipc_signal_attach:=dll.hsa_amd_ipc_signal_attach).restype, hsa_amd_ipc_signal_attach.argtypes = hsa_status_t, [ctypes.POINTER(hsa_amd_ipc_signal_t), ctypes.POINTER(hsa_signal_t)]
except AttributeError: pass

enum_hsa_amd_event_type_s = CEnum(ctypes.c_uint32)
HSA_AMD_GPU_MEMORY_FAULT_EVENT = enum_hsa_amd_event_type_s.define('HSA_AMD_GPU_MEMORY_FAULT_EVENT', 0)
HSA_AMD_GPU_HW_EXCEPTION_EVENT = enum_hsa_amd_event_type_s.define('HSA_AMD_GPU_HW_EXCEPTION_EVENT', 1)
HSA_AMD_GPU_MEMORY_ERROR_EVENT = enum_hsa_amd_event_type_s.define('HSA_AMD_GPU_MEMORY_ERROR_EVENT', 2)

hsa_amd_event_type_t = enum_hsa_amd_event_type_s
hsa_amd_memory_fault_reason_t = CEnum(ctypes.c_uint32)
HSA_AMD_MEMORY_FAULT_PAGE_NOT_PRESENT = hsa_amd_memory_fault_reason_t.define('HSA_AMD_MEMORY_FAULT_PAGE_NOT_PRESENT', 1)
HSA_AMD_MEMORY_FAULT_READ_ONLY = hsa_amd_memory_fault_reason_t.define('HSA_AMD_MEMORY_FAULT_READ_ONLY', 2)
HSA_AMD_MEMORY_FAULT_NX = hsa_amd_memory_fault_reason_t.define('HSA_AMD_MEMORY_FAULT_NX', 4)
HSA_AMD_MEMORY_FAULT_HOST_ONLY = hsa_amd_memory_fault_reason_t.define('HSA_AMD_MEMORY_FAULT_HOST_ONLY', 8)
HSA_AMD_MEMORY_FAULT_DRAMECC = hsa_amd_memory_fault_reason_t.define('HSA_AMD_MEMORY_FAULT_DRAMECC', 16)
HSA_AMD_MEMORY_FAULT_IMPRECISE = hsa_amd_memory_fault_reason_t.define('HSA_AMD_MEMORY_FAULT_IMPRECISE', 32)
HSA_AMD_MEMORY_FAULT_SRAMECC = hsa_amd_memory_fault_reason_t.define('HSA_AMD_MEMORY_FAULT_SRAMECC', 64)
HSA_AMD_MEMORY_FAULT_HANG = hsa_amd_memory_fault_reason_t.define('HSA_AMD_MEMORY_FAULT_HANG', 2147483648)

class struct_hsa_amd_gpu_memory_fault_info_s(Struct): pass
struct_hsa_amd_gpu_memory_fault_info_s._fields_ = [
  ('agent', hsa_agent_t),
  ('virtual_address', uint64_t),
  ('fault_reason_mask', uint32_t),
]
hsa_amd_gpu_memory_fault_info_t = struct_hsa_amd_gpu_memory_fault_info_s
hsa_amd_memory_error_reason_t = CEnum(ctypes.c_uint32)
HSA_AMD_MEMORY_ERROR_MEMORY_IN_USE = hsa_amd_memory_error_reason_t.define('HSA_AMD_MEMORY_ERROR_MEMORY_IN_USE', 1)

class struct_hsa_amd_gpu_memory_error_info_s(Struct): pass
struct_hsa_amd_gpu_memory_error_info_s._fields_ = [
  ('agent', hsa_agent_t),
  ('virtual_address', uint64_t),
  ('error_reason_mask', uint32_t),
]
hsa_amd_gpu_memory_error_info_t = struct_hsa_amd_gpu_memory_error_info_s
hsa_amd_hw_exception_reset_type_t = CEnum(ctypes.c_uint32)
HSA_AMD_HW_EXCEPTION_RESET_TYPE_OTHER = hsa_amd_hw_exception_reset_type_t.define('HSA_AMD_HW_EXCEPTION_RESET_TYPE_OTHER', 1)

hsa_amd_hw_exception_reset_cause_t = CEnum(ctypes.c_uint32)
HSA_AMD_HW_EXCEPTION_CAUSE_GPU_HANG = hsa_amd_hw_exception_reset_cause_t.define('HSA_AMD_HW_EXCEPTION_CAUSE_GPU_HANG', 1)
HSA_AMD_HW_EXCEPTION_CAUSE_ECC = hsa_amd_hw_exception_reset_cause_t.define('HSA_AMD_HW_EXCEPTION_CAUSE_ECC', 2)

class struct_hsa_amd_gpu_hw_exception_info_s(Struct): pass
struct_hsa_amd_gpu_hw_exception_info_s._fields_ = [
  ('agent', hsa_agent_t),
  ('reset_type', hsa_amd_hw_exception_reset_type_t),
  ('reset_cause', hsa_amd_hw_exception_reset_cause_t),
]
hsa_amd_gpu_hw_exception_info_t = struct_hsa_amd_gpu_hw_exception_info_s
class struct_hsa_amd_event_s(Struct): pass
class struct_hsa_amd_event_s_0(ctypes.Union): pass
struct_hsa_amd_event_s_0._fields_ = [
  ('memory_fault', hsa_amd_gpu_memory_fault_info_t),
  ('hw_exception', hsa_amd_gpu_hw_exception_info_t),
  ('memory_error', hsa_amd_gpu_memory_error_info_t),
]
struct_hsa_amd_event_s._anonymous_ = ['_0']
struct_hsa_amd_event_s._fields_ = [
  ('event_type', hsa_amd_event_type_t),
  ('_0', struct_hsa_amd_event_s_0),
]
hsa_amd_event_t = struct_hsa_amd_event_s
hsa_amd_system_event_callback_t = ctypes.CFUNCTYPE(hsa_status_t, ctypes.POINTER(struct_hsa_amd_event_s), ctypes.c_void_p)
try: (hsa_amd_register_system_event_handler:=dll.hsa_amd_register_system_event_handler).restype, hsa_amd_register_system_event_handler.argtypes = hsa_status_t, [hsa_amd_system_event_callback_t, ctypes.c_void_p]
except AttributeError: pass

enum_hsa_amd_queue_priority_s = CEnum(ctypes.c_uint32)
HSA_AMD_QUEUE_PRIORITY_LOW = enum_hsa_amd_queue_priority_s.define('HSA_AMD_QUEUE_PRIORITY_LOW', 0)
HSA_AMD_QUEUE_PRIORITY_NORMAL = enum_hsa_amd_queue_priority_s.define('HSA_AMD_QUEUE_PRIORITY_NORMAL', 1)
HSA_AMD_QUEUE_PRIORITY_HIGH = enum_hsa_amd_queue_priority_s.define('HSA_AMD_QUEUE_PRIORITY_HIGH', 2)

hsa_amd_queue_priority_t = enum_hsa_amd_queue_priority_s
try: (hsa_amd_queue_set_priority:=dll.hsa_amd_queue_set_priority).restype, hsa_amd_queue_set_priority.argtypes = hsa_status_t, [ctypes.POINTER(hsa_queue_t), hsa_amd_queue_priority_t]
except AttributeError: pass

hsa_amd_queue_create_flag_t = CEnum(ctypes.c_uint32)
HSA_AMD_QUEUE_CREATE_SYSTEM_MEM = hsa_amd_queue_create_flag_t.define('HSA_AMD_QUEUE_CREATE_SYSTEM_MEM', 0)
HSA_AMD_QUEUE_CREATE_DEVICE_MEM_RING_BUF = hsa_amd_queue_create_flag_t.define('HSA_AMD_QUEUE_CREATE_DEVICE_MEM_RING_BUF', 1)
HSA_AMD_QUEUE_CREATE_DEVICE_MEM_QUEUE_DESCRIPTOR = hsa_amd_queue_create_flag_t.define('HSA_AMD_QUEUE_CREATE_DEVICE_MEM_QUEUE_DESCRIPTOR', 2)

hsa_amd_deallocation_callback_t = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p)
try: (hsa_amd_register_deallocation_callback:=dll.hsa_amd_register_deallocation_callback).restype, hsa_amd_register_deallocation_callback.argtypes = hsa_status_t, [ctypes.c_void_p, hsa_amd_deallocation_callback_t, ctypes.c_void_p]
except AttributeError: pass

try: (hsa_amd_deregister_deallocation_callback:=dll.hsa_amd_deregister_deallocation_callback).restype, hsa_amd_deregister_deallocation_callback.argtypes = hsa_status_t, [ctypes.c_void_p, hsa_amd_deallocation_callback_t]
except AttributeError: pass

enum_hsa_amd_svm_model_s = CEnum(ctypes.c_uint32)
HSA_AMD_SVM_GLOBAL_FLAG_FINE_GRAINED = enum_hsa_amd_svm_model_s.define('HSA_AMD_SVM_GLOBAL_FLAG_FINE_GRAINED', 0)
HSA_AMD_SVM_GLOBAL_FLAG_COARSE_GRAINED = enum_hsa_amd_svm_model_s.define('HSA_AMD_SVM_GLOBAL_FLAG_COARSE_GRAINED', 1)
HSA_AMD_SVM_GLOBAL_FLAG_INDETERMINATE = enum_hsa_amd_svm_model_s.define('HSA_AMD_SVM_GLOBAL_FLAG_INDETERMINATE', 2)

hsa_amd_svm_model_t = enum_hsa_amd_svm_model_s
enum_hsa_amd_svm_attribute_s = CEnum(ctypes.c_uint32)
HSA_AMD_SVM_ATTRIB_GLOBAL_FLAG = enum_hsa_amd_svm_attribute_s.define('HSA_AMD_SVM_ATTRIB_GLOBAL_FLAG', 0)
HSA_AMD_SVM_ATTRIB_READ_ONLY = enum_hsa_amd_svm_attribute_s.define('HSA_AMD_SVM_ATTRIB_READ_ONLY', 1)
HSA_AMD_SVM_ATTRIB_HIVE_LOCAL = enum_hsa_amd_svm_attribute_s.define('HSA_AMD_SVM_ATTRIB_HIVE_LOCAL', 2)
HSA_AMD_SVM_ATTRIB_MIGRATION_GRANULARITY = enum_hsa_amd_svm_attribute_s.define('HSA_AMD_SVM_ATTRIB_MIGRATION_GRANULARITY', 3)
HSA_AMD_SVM_ATTRIB_PREFERRED_LOCATION = enum_hsa_amd_svm_attribute_s.define('HSA_AMD_SVM_ATTRIB_PREFERRED_LOCATION', 4)
HSA_AMD_SVM_ATTRIB_PREFETCH_LOCATION = enum_hsa_amd_svm_attribute_s.define('HSA_AMD_SVM_ATTRIB_PREFETCH_LOCATION', 5)
HSA_AMD_SVM_ATTRIB_READ_MOSTLY = enum_hsa_amd_svm_attribute_s.define('HSA_AMD_SVM_ATTRIB_READ_MOSTLY', 6)
HSA_AMD_SVM_ATTRIB_GPU_EXEC = enum_hsa_amd_svm_attribute_s.define('HSA_AMD_SVM_ATTRIB_GPU_EXEC', 7)
HSA_AMD_SVM_ATTRIB_AGENT_ACCESSIBLE = enum_hsa_amd_svm_attribute_s.define('HSA_AMD_SVM_ATTRIB_AGENT_ACCESSIBLE', 512)
HSA_AMD_SVM_ATTRIB_AGENT_ACCESSIBLE_IN_PLACE = enum_hsa_amd_svm_attribute_s.define('HSA_AMD_SVM_ATTRIB_AGENT_ACCESSIBLE_IN_PLACE', 513)
HSA_AMD_SVM_ATTRIB_AGENT_NO_ACCESS = enum_hsa_amd_svm_attribute_s.define('HSA_AMD_SVM_ATTRIB_AGENT_NO_ACCESS', 514)
HSA_AMD_SVM_ATTRIB_ACCESS_QUERY = enum_hsa_amd_svm_attribute_s.define('HSA_AMD_SVM_ATTRIB_ACCESS_QUERY', 515)

hsa_amd_svm_attribute_t = enum_hsa_amd_svm_attribute_s
class struct_hsa_amd_svm_attribute_pair_s(Struct): pass
struct_hsa_amd_svm_attribute_pair_s._fields_ = [
  ('attribute', uint64_t),
  ('value', uint64_t),
]
hsa_amd_svm_attribute_pair_t = struct_hsa_amd_svm_attribute_pair_s
try: (hsa_amd_svm_attributes_set:=dll.hsa_amd_svm_attributes_set).restype, hsa_amd_svm_attributes_set.argtypes = hsa_status_t, [ctypes.c_void_p, size_t, ctypes.POINTER(hsa_amd_svm_attribute_pair_t), size_t]
except AttributeError: pass

try: (hsa_amd_svm_attributes_get:=dll.hsa_amd_svm_attributes_get).restype, hsa_amd_svm_attributes_get.argtypes = hsa_status_t, [ctypes.c_void_p, size_t, ctypes.POINTER(hsa_amd_svm_attribute_pair_t), size_t]
except AttributeError: pass

try: (hsa_amd_svm_prefetch_async:=dll.hsa_amd_svm_prefetch_async).restype, hsa_amd_svm_prefetch_async.argtypes = hsa_status_t, [ctypes.c_void_p, size_t, hsa_agent_t, uint32_t, ctypes.POINTER(hsa_signal_t), hsa_signal_t]
except AttributeError: pass

try: (hsa_amd_spm_acquire:=dll.hsa_amd_spm_acquire).restype, hsa_amd_spm_acquire.argtypes = hsa_status_t, [hsa_agent_t]
except AttributeError: pass

try: (hsa_amd_spm_release:=dll.hsa_amd_spm_release).restype, hsa_amd_spm_release.argtypes = hsa_status_t, [hsa_agent_t]
except AttributeError: pass

try: (hsa_amd_spm_set_dest_buffer:=dll.hsa_amd_spm_set_dest_buffer).restype, hsa_amd_spm_set_dest_buffer.argtypes = hsa_status_t, [hsa_agent_t, size_t, ctypes.POINTER(uint32_t), ctypes.POINTER(uint32_t), ctypes.c_void_p, ctypes.POINTER(ctypes.c_bool)]
except AttributeError: pass

try: (hsa_amd_portable_export_dmabuf:=dll.hsa_amd_portable_export_dmabuf).restype, hsa_amd_portable_export_dmabuf.argtypes = hsa_status_t, [ctypes.c_void_p, size_t, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(uint64_t)]
except AttributeError: pass

try: (hsa_amd_portable_export_dmabuf_v2:=dll.hsa_amd_portable_export_dmabuf_v2).restype, hsa_amd_portable_export_dmabuf_v2.argtypes = hsa_status_t, [ctypes.c_void_p, size_t, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(uint64_t), uint64_t]
except AttributeError: pass

try: (hsa_amd_portable_close_dmabuf:=dll.hsa_amd_portable_close_dmabuf).restype, hsa_amd_portable_close_dmabuf.argtypes = hsa_status_t, [ctypes.c_int32]
except AttributeError: pass

enum_hsa_amd_vmem_address_reserve_flag_s = CEnum(ctypes.c_uint32)
HSA_AMD_VMEM_ADDRESS_NO_REGISTER = enum_hsa_amd_vmem_address_reserve_flag_s.define('HSA_AMD_VMEM_ADDRESS_NO_REGISTER', 1)

hsa_amd_vmem_address_reserve_flag_t = enum_hsa_amd_vmem_address_reserve_flag_s
try: (hsa_amd_vmem_address_reserve:=dll.hsa_amd_vmem_address_reserve).restype, hsa_amd_vmem_address_reserve.argtypes = hsa_status_t, [ctypes.POINTER(ctypes.c_void_p), size_t, uint64_t, uint64_t]
except AttributeError: pass

try: (hsa_amd_vmem_address_reserve_align:=dll.hsa_amd_vmem_address_reserve_align).restype, hsa_amd_vmem_address_reserve_align.argtypes = hsa_status_t, [ctypes.POINTER(ctypes.c_void_p), size_t, uint64_t, uint64_t, uint64_t]
except AttributeError: pass

try: (hsa_amd_vmem_address_free:=dll.hsa_amd_vmem_address_free).restype, hsa_amd_vmem_address_free.argtypes = hsa_status_t, [ctypes.c_void_p, size_t]
except AttributeError: pass

class struct_hsa_amd_vmem_alloc_handle_s(Struct): pass
struct_hsa_amd_vmem_alloc_handle_s._fields_ = [
  ('handle', uint64_t),
]
hsa_amd_vmem_alloc_handle_t = struct_hsa_amd_vmem_alloc_handle_s
hsa_amd_memory_type_t = CEnum(ctypes.c_uint32)
MEMORY_TYPE_NONE = hsa_amd_memory_type_t.define('MEMORY_TYPE_NONE', 0)
MEMORY_TYPE_PINNED = hsa_amd_memory_type_t.define('MEMORY_TYPE_PINNED', 1)

try: (hsa_amd_vmem_handle_create:=dll.hsa_amd_vmem_handle_create).restype, hsa_amd_vmem_handle_create.argtypes = hsa_status_t, [hsa_amd_memory_pool_t, size_t, hsa_amd_memory_type_t, uint64_t, ctypes.POINTER(hsa_amd_vmem_alloc_handle_t)]
except AttributeError: pass

try: (hsa_amd_vmem_handle_release:=dll.hsa_amd_vmem_handle_release).restype, hsa_amd_vmem_handle_release.argtypes = hsa_status_t, [hsa_amd_vmem_alloc_handle_t]
except AttributeError: pass

try: (hsa_amd_vmem_map:=dll.hsa_amd_vmem_map).restype, hsa_amd_vmem_map.argtypes = hsa_status_t, [ctypes.c_void_p, size_t, size_t, hsa_amd_vmem_alloc_handle_t, uint64_t]
except AttributeError: pass

try: (hsa_amd_vmem_unmap:=dll.hsa_amd_vmem_unmap).restype, hsa_amd_vmem_unmap.argtypes = hsa_status_t, [ctypes.c_void_p, size_t]
except AttributeError: pass

class struct_hsa_amd_memory_access_desc_s(Struct): pass
struct_hsa_amd_memory_access_desc_s._fields_ = [
  ('permissions', hsa_access_permission_t),
  ('agent_handle', hsa_agent_t),
]
hsa_amd_memory_access_desc_t = struct_hsa_amd_memory_access_desc_s
try: (hsa_amd_vmem_set_access:=dll.hsa_amd_vmem_set_access).restype, hsa_amd_vmem_set_access.argtypes = hsa_status_t, [ctypes.c_void_p, size_t, ctypes.POINTER(hsa_amd_memory_access_desc_t), size_t]
except AttributeError: pass

try: (hsa_amd_vmem_get_access:=dll.hsa_amd_vmem_get_access).restype, hsa_amd_vmem_get_access.argtypes = hsa_status_t, [ctypes.c_void_p, ctypes.POINTER(hsa_access_permission_t), hsa_agent_t]
except AttributeError: pass

try: (hsa_amd_vmem_export_shareable_handle:=dll.hsa_amd_vmem_export_shareable_handle).restype, hsa_amd_vmem_export_shareable_handle.argtypes = hsa_status_t, [ctypes.POINTER(ctypes.c_int32), hsa_amd_vmem_alloc_handle_t, uint64_t]
except AttributeError: pass

try: (hsa_amd_vmem_import_shareable_handle:=dll.hsa_amd_vmem_import_shareable_handle).restype, hsa_amd_vmem_import_shareable_handle.argtypes = hsa_status_t, [ctypes.c_int32, ctypes.POINTER(hsa_amd_vmem_alloc_handle_t)]
except AttributeError: pass

try: (hsa_amd_vmem_retain_alloc_handle:=dll.hsa_amd_vmem_retain_alloc_handle).restype, hsa_amd_vmem_retain_alloc_handle.argtypes = hsa_status_t, [ctypes.POINTER(hsa_amd_vmem_alloc_handle_t), ctypes.c_void_p]
except AttributeError: pass

try: (hsa_amd_vmem_get_alloc_properties_from_handle:=dll.hsa_amd_vmem_get_alloc_properties_from_handle).restype, hsa_amd_vmem_get_alloc_properties_from_handle.argtypes = hsa_status_t, [hsa_amd_vmem_alloc_handle_t, ctypes.POINTER(hsa_amd_memory_pool_t), ctypes.POINTER(hsa_amd_memory_type_t)]
except AttributeError: pass

try: (hsa_amd_agent_set_async_scratch_limit:=dll.hsa_amd_agent_set_async_scratch_limit).restype, hsa_amd_agent_set_async_scratch_limit.argtypes = hsa_status_t, [hsa_agent_t, size_t]
except AttributeError: pass

hsa_queue_info_attribute_t = CEnum(ctypes.c_uint32)
HSA_AMD_QUEUE_INFO_AGENT = hsa_queue_info_attribute_t.define('HSA_AMD_QUEUE_INFO_AGENT', 0)
HSA_AMD_QUEUE_INFO_DOORBELL_ID = hsa_queue_info_attribute_t.define('HSA_AMD_QUEUE_INFO_DOORBELL_ID', 1)

try: (hsa_amd_queue_get_info:=dll.hsa_amd_queue_get_info).restype, hsa_amd_queue_get_info.argtypes = hsa_status_t, [ctypes.POINTER(hsa_queue_t), hsa_queue_info_attribute_t, ctypes.c_void_p]
except AttributeError: pass

class struct_hsa_amd_ais_file_handle_s(Struct): pass
class struct_hsa_amd_ais_file_handle_s_0(ctypes.Union): pass
struct_hsa_amd_ais_file_handle_s_0._fields_ = [
  ('handle', ctypes.c_void_p),
  ('fd', ctypes.c_int32),
  ('pad', (uint8_t * 8)),
]
struct_hsa_amd_ais_file_handle_s._anonymous_ = ['_0']
struct_hsa_amd_ais_file_handle_s._fields_ = [
  ('_0', struct_hsa_amd_ais_file_handle_s_0),
]
hsa_amd_ais_file_handle_t = struct_hsa_amd_ais_file_handle_s
int64_t = ctypes.c_int64
try: (hsa_amd_ais_file_write:=dll.hsa_amd_ais_file_write).restype, hsa_amd_ais_file_write.argtypes = hsa_status_t, [hsa_amd_ais_file_handle_t, ctypes.c_void_p, uint64_t, int64_t, ctypes.POINTER(uint64_t), ctypes.POINTER(int32_t)]
except AttributeError: pass

try: (hsa_amd_ais_file_read:=dll.hsa_amd_ais_file_read).restype, hsa_amd_ais_file_read.argtypes = hsa_status_t, [hsa_amd_ais_file_handle_t, ctypes.c_void_p, uint64_t, int64_t, ctypes.POINTER(uint64_t), ctypes.POINTER(int32_t)]
except AttributeError: pass

enum_hsa_amd_log_flag_s = CEnum(ctypes.c_uint32)
HSA_AMD_LOG_FLAG_BLIT_KERNEL_PKTS = enum_hsa_amd_log_flag_s.define('HSA_AMD_LOG_FLAG_BLIT_KERNEL_PKTS', 0)
HSA_AMD_LOG_FLAG_AQL = enum_hsa_amd_log_flag_s.define('HSA_AMD_LOG_FLAG_AQL', 0)
HSA_AMD_LOG_FLAG_SDMA = enum_hsa_amd_log_flag_s.define('HSA_AMD_LOG_FLAG_SDMA', 1)
HSA_AMD_LOG_FLAG_INFO = enum_hsa_amd_log_flag_s.define('HSA_AMD_LOG_FLAG_INFO', 2)

hsa_amd_log_flag_t = enum_hsa_amd_log_flag_s
try: (hsa_amd_enable_logging:=dll.hsa_amd_enable_logging).restype, hsa_amd_enable_logging.argtypes = hsa_status_t, [ctypes.POINTER(uint8_t), ctypes.c_void_p]
except AttributeError: pass

amd_signal_kind64_t = ctypes.c_int64
enum_amd_signal_kind_t = CEnum(ctypes.c_int32)
AMD_SIGNAL_KIND_INVALID = enum_amd_signal_kind_t.define('AMD_SIGNAL_KIND_INVALID', 0)
AMD_SIGNAL_KIND_USER = enum_amd_signal_kind_t.define('AMD_SIGNAL_KIND_USER', 1)
AMD_SIGNAL_KIND_DOORBELL = enum_amd_signal_kind_t.define('AMD_SIGNAL_KIND_DOORBELL', -1)
AMD_SIGNAL_KIND_LEGACY_DOORBELL = enum_amd_signal_kind_t.define('AMD_SIGNAL_KIND_LEGACY_DOORBELL', -2)

class struct_amd_signal_s(Struct): pass
class struct_amd_signal_s_0(ctypes.Union): pass
struct_amd_signal_s_0._fields_ = [
  ('value', int64_t),
  ('hardware_doorbell_ptr', ctypes.POINTER(uint64_t)),
]
class struct_amd_signal_s_1(ctypes.Union): pass
class struct_amd_queue_v2_s(Struct): pass
amd_queue_v2_t = struct_amd_queue_v2_s
amd_queue_properties32_t = ctypes.c_uint32
class struct_scratch_last_used_index_xcc_s(Struct): pass
scratch_last_used_index_xcc_t = struct_scratch_last_used_index_xcc_s
struct_scratch_last_used_index_xcc_s._fields_ = [
  ('main', uint64_t),
  ('alt', uint64_t),
]
struct_amd_queue_v2_s._fields_ = [
  ('hsa_queue', hsa_queue_t),
  ('caps', uint32_t),
  ('reserved1', (uint32_t * 3)),
  ('write_dispatch_id', uint64_t),
  ('group_segment_aperture_base_hi', uint32_t),
  ('private_segment_aperture_base_hi', uint32_t),
  ('max_cu_id', uint32_t),
  ('max_wave_id', uint32_t),
  ('max_legacy_doorbell_dispatch_id_plus_1', uint64_t),
  ('legacy_doorbell_lock', uint32_t),
  ('reserved2', (uint32_t * 9)),
  ('read_dispatch_id', uint64_t),
  ('read_dispatch_id_field_base_byte_offset', uint32_t),
  ('compute_tmpring_size', uint32_t),
  ('scratch_resource_descriptor', (uint32_t * 4)),
  ('scratch_backing_memory_location', uint64_t),
  ('scratch_backing_memory_byte_size', uint64_t),
  ('scratch_wave64_lane_byte_size', uint32_t),
  ('queue_properties', amd_queue_properties32_t),
  ('scratch_max_use_index', uint64_t),
  ('queue_inactive_signal', hsa_signal_t),
  ('alt_scratch_max_use_index', uint64_t),
  ('alt_scratch_resource_descriptor', (uint32_t * 4)),
  ('alt_scratch_backing_memory_location', uint64_t),
  ('alt_scratch_dispatch_limit_x', uint32_t),
  ('alt_scratch_dispatch_limit_y', uint32_t),
  ('alt_scratch_dispatch_limit_z', uint32_t),
  ('alt_scratch_wave64_lane_byte_size', uint32_t),
  ('alt_compute_tmpring_size', uint32_t),
  ('reserved5', uint32_t),
  ('scratch_last_used_index', (scratch_last_used_index_xcc_t * 128)),
]
struct_amd_signal_s_1._fields_ = [
  ('queue_ptr', ctypes.POINTER(amd_queue_v2_t)),
  ('reserved2', uint64_t),
]
struct_amd_signal_s._anonymous_ = ['_0', '_1']
struct_amd_signal_s._fields_ = [
  ('kind', amd_signal_kind64_t),
  ('_0', struct_amd_signal_s_0),
  ('event_mailbox_ptr', uint64_t),
  ('event_id', uint32_t),
  ('reserved1', uint32_t),
  ('start_ts', uint64_t),
  ('end_ts', uint64_t),
  ('_1', struct_amd_signal_s_1),
  ('reserved3', (uint32_t * 2)),
]
amd_signal_t = struct_amd_signal_s
enum_amd_queue_properties_t = CEnum(ctypes.c_int32)
AMD_QUEUE_PROPERTIES_ENABLE_TRAP_HANDLER_SHIFT = enum_amd_queue_properties_t.define('AMD_QUEUE_PROPERTIES_ENABLE_TRAP_HANDLER_SHIFT', 0)
AMD_QUEUE_PROPERTIES_ENABLE_TRAP_HANDLER_WIDTH = enum_amd_queue_properties_t.define('AMD_QUEUE_PROPERTIES_ENABLE_TRAP_HANDLER_WIDTH', 1)
AMD_QUEUE_PROPERTIES_ENABLE_TRAP_HANDLER = enum_amd_queue_properties_t.define('AMD_QUEUE_PROPERTIES_ENABLE_TRAP_HANDLER', 1)
AMD_QUEUE_PROPERTIES_IS_PTR64_SHIFT = enum_amd_queue_properties_t.define('AMD_QUEUE_PROPERTIES_IS_PTR64_SHIFT', 1)
AMD_QUEUE_PROPERTIES_IS_PTR64_WIDTH = enum_amd_queue_properties_t.define('AMD_QUEUE_PROPERTIES_IS_PTR64_WIDTH', 1)
AMD_QUEUE_PROPERTIES_IS_PTR64 = enum_amd_queue_properties_t.define('AMD_QUEUE_PROPERTIES_IS_PTR64', 2)
AMD_QUEUE_PROPERTIES_ENABLE_TRAP_HANDLER_DEBUG_SGPRS_SHIFT = enum_amd_queue_properties_t.define('AMD_QUEUE_PROPERTIES_ENABLE_TRAP_HANDLER_DEBUG_SGPRS_SHIFT', 2)
AMD_QUEUE_PROPERTIES_ENABLE_TRAP_HANDLER_DEBUG_SGPRS_WIDTH = enum_amd_queue_properties_t.define('AMD_QUEUE_PROPERTIES_ENABLE_TRAP_HANDLER_DEBUG_SGPRS_WIDTH', 1)
AMD_QUEUE_PROPERTIES_ENABLE_TRAP_HANDLER_DEBUG_SGPRS = enum_amd_queue_properties_t.define('AMD_QUEUE_PROPERTIES_ENABLE_TRAP_HANDLER_DEBUG_SGPRS', 4)
AMD_QUEUE_PROPERTIES_ENABLE_PROFILING_SHIFT = enum_amd_queue_properties_t.define('AMD_QUEUE_PROPERTIES_ENABLE_PROFILING_SHIFT', 3)
AMD_QUEUE_PROPERTIES_ENABLE_PROFILING_WIDTH = enum_amd_queue_properties_t.define('AMD_QUEUE_PROPERTIES_ENABLE_PROFILING_WIDTH', 1)
AMD_QUEUE_PROPERTIES_ENABLE_PROFILING = enum_amd_queue_properties_t.define('AMD_QUEUE_PROPERTIES_ENABLE_PROFILING', 8)
AMD_QUEUE_PROPERTIES_USE_SCRATCH_ONCE_SHIFT = enum_amd_queue_properties_t.define('AMD_QUEUE_PROPERTIES_USE_SCRATCH_ONCE_SHIFT', 4)
AMD_QUEUE_PROPERTIES_USE_SCRATCH_ONCE_WIDTH = enum_amd_queue_properties_t.define('AMD_QUEUE_PROPERTIES_USE_SCRATCH_ONCE_WIDTH', 1)
AMD_QUEUE_PROPERTIES_USE_SCRATCH_ONCE = enum_amd_queue_properties_t.define('AMD_QUEUE_PROPERTIES_USE_SCRATCH_ONCE', 16)
AMD_QUEUE_PROPERTIES_RESERVED1_SHIFT = enum_amd_queue_properties_t.define('AMD_QUEUE_PROPERTIES_RESERVED1_SHIFT', 5)
AMD_QUEUE_PROPERTIES_RESERVED1_WIDTH = enum_amd_queue_properties_t.define('AMD_QUEUE_PROPERTIES_RESERVED1_WIDTH', 27)
AMD_QUEUE_PROPERTIES_RESERVED1 = enum_amd_queue_properties_t.define('AMD_QUEUE_PROPERTIES_RESERVED1', -32)

amd_queue_capabilities32_t = ctypes.c_uint32
enum_amd_queue_capabilities_t = CEnum(ctypes.c_uint32)
AMD_QUEUE_CAPS_CP_ASYNC_RECLAIM_SHIFT = enum_amd_queue_capabilities_t.define('AMD_QUEUE_CAPS_CP_ASYNC_RECLAIM_SHIFT', 0)
AMD_QUEUE_CAPS_CP_ASYNC_RECLAIM_WIDTH = enum_amd_queue_capabilities_t.define('AMD_QUEUE_CAPS_CP_ASYNC_RECLAIM_WIDTH', 1)
AMD_QUEUE_CAPS_CP_ASYNC_RECLAIM = enum_amd_queue_capabilities_t.define('AMD_QUEUE_CAPS_CP_ASYNC_RECLAIM', 1)
AMD_QUEUE_CAPS_SW_ASYNC_RECLAIM_SHIFT = enum_amd_queue_capabilities_t.define('AMD_QUEUE_CAPS_SW_ASYNC_RECLAIM_SHIFT', 1)
AMD_QUEUE_CAPS_SW_ASYNC_RECLAIM_WIDTH = enum_amd_queue_capabilities_t.define('AMD_QUEUE_CAPS_SW_ASYNC_RECLAIM_WIDTH', 1)
AMD_QUEUE_CAPS_SW_ASYNC_RECLAIM = enum_amd_queue_capabilities_t.define('AMD_QUEUE_CAPS_SW_ASYNC_RECLAIM', 2)

class struct_amd_queue_s(Struct): pass
struct_amd_queue_s._fields_ = [
  ('hsa_queue', hsa_queue_t),
  ('caps', uint32_t),
  ('reserved1', (uint32_t * 3)),
  ('write_dispatch_id', uint64_t),
  ('group_segment_aperture_base_hi', uint32_t),
  ('private_segment_aperture_base_hi', uint32_t),
  ('max_cu_id', uint32_t),
  ('max_wave_id', uint32_t),
  ('max_legacy_doorbell_dispatch_id_plus_1', uint64_t),
  ('legacy_doorbell_lock', uint32_t),
  ('reserved2', (uint32_t * 9)),
  ('read_dispatch_id', uint64_t),
  ('read_dispatch_id_field_base_byte_offset', uint32_t),
  ('compute_tmpring_size', uint32_t),
  ('scratch_resource_descriptor', (uint32_t * 4)),
  ('scratch_backing_memory_location', uint64_t),
  ('reserved3', (uint32_t * 2)),
  ('scratch_wave64_lane_byte_size', uint32_t),
  ('queue_properties', amd_queue_properties32_t),
  ('reserved4', (uint32_t * 2)),
  ('queue_inactive_signal', hsa_signal_t),
  ('reserved5', (uint32_t * 14)),
]
amd_queue_t = struct_amd_queue_s
amd_kernel_code_version32_t = ctypes.c_uint32
enum_amd_kernel_code_version_t = CEnum(ctypes.c_uint32)
AMD_KERNEL_CODE_VERSION_MAJOR = enum_amd_kernel_code_version_t.define('AMD_KERNEL_CODE_VERSION_MAJOR', 1)
AMD_KERNEL_CODE_VERSION_MINOR = enum_amd_kernel_code_version_t.define('AMD_KERNEL_CODE_VERSION_MINOR', 1)

amd_machine_kind16_t = ctypes.c_uint16
enum_amd_machine_kind_t = CEnum(ctypes.c_uint32)
AMD_MACHINE_KIND_UNDEFINED = enum_amd_machine_kind_t.define('AMD_MACHINE_KIND_UNDEFINED', 0)
AMD_MACHINE_KIND_AMDGPU = enum_amd_machine_kind_t.define('AMD_MACHINE_KIND_AMDGPU', 1)

amd_machine_version16_t = ctypes.c_uint16
enum_amd_float_round_mode_t = CEnum(ctypes.c_uint32)
AMD_FLOAT_ROUND_MODE_NEAREST_EVEN = enum_amd_float_round_mode_t.define('AMD_FLOAT_ROUND_MODE_NEAREST_EVEN', 0)
AMD_FLOAT_ROUND_MODE_PLUS_INFINITY = enum_amd_float_round_mode_t.define('AMD_FLOAT_ROUND_MODE_PLUS_INFINITY', 1)
AMD_FLOAT_ROUND_MODE_MINUS_INFINITY = enum_amd_float_round_mode_t.define('AMD_FLOAT_ROUND_MODE_MINUS_INFINITY', 2)
AMD_FLOAT_ROUND_MODE_ZERO = enum_amd_float_round_mode_t.define('AMD_FLOAT_ROUND_MODE_ZERO', 3)

enum_amd_float_denorm_mode_t = CEnum(ctypes.c_uint32)
AMD_FLOAT_DENORM_MODE_FLUSH_SOURCE_OUTPUT = enum_amd_float_denorm_mode_t.define('AMD_FLOAT_DENORM_MODE_FLUSH_SOURCE_OUTPUT', 0)
AMD_FLOAT_DENORM_MODE_FLUSH_OUTPUT = enum_amd_float_denorm_mode_t.define('AMD_FLOAT_DENORM_MODE_FLUSH_OUTPUT', 1)
AMD_FLOAT_DENORM_MODE_FLUSH_SOURCE = enum_amd_float_denorm_mode_t.define('AMD_FLOAT_DENORM_MODE_FLUSH_SOURCE', 2)
AMD_FLOAT_DENORM_MODE_NO_FLUSH = enum_amd_float_denorm_mode_t.define('AMD_FLOAT_DENORM_MODE_NO_FLUSH', 3)

amd_compute_pgm_rsrc_one32_t = ctypes.c_uint32
enum_amd_compute_pgm_rsrc_one_t = CEnum(ctypes.c_int32)
AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WORKITEM_VGPR_COUNT_SHIFT = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WORKITEM_VGPR_COUNT_SHIFT', 0)
AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WORKITEM_VGPR_COUNT_WIDTH = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WORKITEM_VGPR_COUNT_WIDTH', 6)
AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WORKITEM_VGPR_COUNT = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WORKITEM_VGPR_COUNT', 63)
AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WAVEFRONT_SGPR_COUNT_SHIFT = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WAVEFRONT_SGPR_COUNT_SHIFT', 6)
AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WAVEFRONT_SGPR_COUNT_WIDTH = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WAVEFRONT_SGPR_COUNT_WIDTH', 4)
AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WAVEFRONT_SGPR_COUNT = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WAVEFRONT_SGPR_COUNT', 960)
AMD_COMPUTE_PGM_RSRC_ONE_PRIORITY_SHIFT = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_PRIORITY_SHIFT', 10)
AMD_COMPUTE_PGM_RSRC_ONE_PRIORITY_WIDTH = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_PRIORITY_WIDTH', 2)
AMD_COMPUTE_PGM_RSRC_ONE_PRIORITY = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_PRIORITY', 3072)
AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_ROUND_MODE_32_SHIFT = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_ROUND_MODE_32_SHIFT', 12)
AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_ROUND_MODE_32_WIDTH = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_ROUND_MODE_32_WIDTH', 2)
AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_ROUND_MODE_32 = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_ROUND_MODE_32', 12288)
AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_ROUND_MODE_16_64_SHIFT = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_ROUND_MODE_16_64_SHIFT', 14)
AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_ROUND_MODE_16_64_WIDTH = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_ROUND_MODE_16_64_WIDTH', 2)
AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_ROUND_MODE_16_64 = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_ROUND_MODE_16_64', 49152)
AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_32_SHIFT = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_32_SHIFT', 16)
AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_32_WIDTH = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_32_WIDTH', 2)
AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_32 = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_32', 196608)
AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_16_64_SHIFT = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_16_64_SHIFT', 18)
AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_16_64_WIDTH = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_16_64_WIDTH', 2)
AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_16_64 = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_16_64', 786432)
AMD_COMPUTE_PGM_RSRC_ONE_PRIV_SHIFT = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_PRIV_SHIFT', 20)
AMD_COMPUTE_PGM_RSRC_ONE_PRIV_WIDTH = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_PRIV_WIDTH', 1)
AMD_COMPUTE_PGM_RSRC_ONE_PRIV = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_PRIV', 1048576)
AMD_COMPUTE_PGM_RSRC_ONE_ENABLE_DX10_CLAMP_SHIFT = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_ENABLE_DX10_CLAMP_SHIFT', 21)
AMD_COMPUTE_PGM_RSRC_ONE_ENABLE_DX10_CLAMP_WIDTH = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_ENABLE_DX10_CLAMP_WIDTH', 1)
AMD_COMPUTE_PGM_RSRC_ONE_ENABLE_DX10_CLAMP = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_ENABLE_DX10_CLAMP', 2097152)
AMD_COMPUTE_PGM_RSRC_ONE_DEBUG_MODE_SHIFT = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_DEBUG_MODE_SHIFT', 22)
AMD_COMPUTE_PGM_RSRC_ONE_DEBUG_MODE_WIDTH = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_DEBUG_MODE_WIDTH', 1)
AMD_COMPUTE_PGM_RSRC_ONE_DEBUG_MODE = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_DEBUG_MODE', 4194304)
AMD_COMPUTE_PGM_RSRC_ONE_ENABLE_IEEE_MODE_SHIFT = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_ENABLE_IEEE_MODE_SHIFT', 23)
AMD_COMPUTE_PGM_RSRC_ONE_ENABLE_IEEE_MODE_WIDTH = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_ENABLE_IEEE_MODE_WIDTH', 1)
AMD_COMPUTE_PGM_RSRC_ONE_ENABLE_IEEE_MODE = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_ENABLE_IEEE_MODE', 8388608)
AMD_COMPUTE_PGM_RSRC_ONE_BULKY_SHIFT = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_BULKY_SHIFT', 24)
AMD_COMPUTE_PGM_RSRC_ONE_BULKY_WIDTH = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_BULKY_WIDTH', 1)
AMD_COMPUTE_PGM_RSRC_ONE_BULKY = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_BULKY', 16777216)
AMD_COMPUTE_PGM_RSRC_ONE_CDBG_USER_SHIFT = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_CDBG_USER_SHIFT', 25)
AMD_COMPUTE_PGM_RSRC_ONE_CDBG_USER_WIDTH = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_CDBG_USER_WIDTH', 1)
AMD_COMPUTE_PGM_RSRC_ONE_CDBG_USER = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_CDBG_USER', 33554432)
AMD_COMPUTE_PGM_RSRC_ONE_RESERVED1_SHIFT = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_RESERVED1_SHIFT', 26)
AMD_COMPUTE_PGM_RSRC_ONE_RESERVED1_WIDTH = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_RESERVED1_WIDTH', 6)
AMD_COMPUTE_PGM_RSRC_ONE_RESERVED1 = enum_amd_compute_pgm_rsrc_one_t.define('AMD_COMPUTE_PGM_RSRC_ONE_RESERVED1', -67108864)

enum_amd_system_vgpr_workitem_id_t = CEnum(ctypes.c_uint32)
AMD_SYSTEM_VGPR_WORKITEM_ID_X = enum_amd_system_vgpr_workitem_id_t.define('AMD_SYSTEM_VGPR_WORKITEM_ID_X', 0)
AMD_SYSTEM_VGPR_WORKITEM_ID_X_Y = enum_amd_system_vgpr_workitem_id_t.define('AMD_SYSTEM_VGPR_WORKITEM_ID_X_Y', 1)
AMD_SYSTEM_VGPR_WORKITEM_ID_X_Y_Z = enum_amd_system_vgpr_workitem_id_t.define('AMD_SYSTEM_VGPR_WORKITEM_ID_X_Y_Z', 2)
AMD_SYSTEM_VGPR_WORKITEM_ID_UNDEFINED = enum_amd_system_vgpr_workitem_id_t.define('AMD_SYSTEM_VGPR_WORKITEM_ID_UNDEFINED', 3)

amd_compute_pgm_rsrc_two32_t = ctypes.c_uint32
enum_amd_compute_pgm_rsrc_two_t = CEnum(ctypes.c_int32)
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_PRIVATE_SEGMENT_WAVE_BYTE_OFFSET_SHIFT = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_PRIVATE_SEGMENT_WAVE_BYTE_OFFSET_SHIFT', 0)
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_PRIVATE_SEGMENT_WAVE_BYTE_OFFSET_WIDTH = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_PRIVATE_SEGMENT_WAVE_BYTE_OFFSET_WIDTH', 1)
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_PRIVATE_SEGMENT_WAVE_BYTE_OFFSET = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_PRIVATE_SEGMENT_WAVE_BYTE_OFFSET', 1)
AMD_COMPUTE_PGM_RSRC_TWO_USER_SGPR_COUNT_SHIFT = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_USER_SGPR_COUNT_SHIFT', 1)
AMD_COMPUTE_PGM_RSRC_TWO_USER_SGPR_COUNT_WIDTH = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_USER_SGPR_COUNT_WIDTH', 5)
AMD_COMPUTE_PGM_RSRC_TWO_USER_SGPR_COUNT = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_USER_SGPR_COUNT', 62)
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_TRAP_HANDLER_SHIFT = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_TRAP_HANDLER_SHIFT', 6)
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_TRAP_HANDLER_WIDTH = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_TRAP_HANDLER_WIDTH', 1)
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_TRAP_HANDLER = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_TRAP_HANDLER', 64)
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_X_SHIFT = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_X_SHIFT', 7)
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_X_WIDTH = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_X_WIDTH', 1)
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_X = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_X', 128)
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Y_SHIFT = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Y_SHIFT', 8)
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Y_WIDTH = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Y_WIDTH', 1)
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Y = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Y', 256)
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Z_SHIFT = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Z_SHIFT', 9)
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Z_WIDTH = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Z_WIDTH', 1)
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Z = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Z', 512)
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_INFO_SHIFT = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_INFO_SHIFT', 10)
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_INFO_WIDTH = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_INFO_WIDTH', 1)
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_INFO = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_INFO', 1024)
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_VGPR_WORKITEM_ID_SHIFT = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_VGPR_WORKITEM_ID_SHIFT', 11)
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_VGPR_WORKITEM_ID_WIDTH = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_VGPR_WORKITEM_ID_WIDTH', 2)
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_VGPR_WORKITEM_ID = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_VGPR_WORKITEM_ID', 6144)
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_ADDRESS_WATCH_SHIFT = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_ADDRESS_WATCH_SHIFT', 13)
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_ADDRESS_WATCH_WIDTH = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_ADDRESS_WATCH_WIDTH', 1)
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_ADDRESS_WATCH = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_ADDRESS_WATCH', 8192)
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_MEMORY_VIOLATION_SHIFT = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_MEMORY_VIOLATION_SHIFT', 14)
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_MEMORY_VIOLATION_WIDTH = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_MEMORY_VIOLATION_WIDTH', 1)
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_MEMORY_VIOLATION = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_MEMORY_VIOLATION', 16384)
AMD_COMPUTE_PGM_RSRC_TWO_GRANULATED_LDS_SIZE_SHIFT = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_GRANULATED_LDS_SIZE_SHIFT', 15)
AMD_COMPUTE_PGM_RSRC_TWO_GRANULATED_LDS_SIZE_WIDTH = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_GRANULATED_LDS_SIZE_WIDTH', 9)
AMD_COMPUTE_PGM_RSRC_TWO_GRANULATED_LDS_SIZE = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_GRANULATED_LDS_SIZE', 16744448)
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_INVALID_OPERATION_SHIFT = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_INVALID_OPERATION_SHIFT', 24)
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_INVALID_OPERATION_WIDTH = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_INVALID_OPERATION_WIDTH', 1)
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_INVALID_OPERATION = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_INVALID_OPERATION', 16777216)
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_FP_DENORMAL_SOURCE_SHIFT = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_FP_DENORMAL_SOURCE_SHIFT', 25)
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_FP_DENORMAL_SOURCE_WIDTH = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_FP_DENORMAL_SOURCE_WIDTH', 1)
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_FP_DENORMAL_SOURCE = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_FP_DENORMAL_SOURCE', 33554432)
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_DIVISION_BY_ZERO_SHIFT = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_DIVISION_BY_ZERO_SHIFT', 26)
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_DIVISION_BY_ZERO_WIDTH = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_DIVISION_BY_ZERO_WIDTH', 1)
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_DIVISION_BY_ZERO = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_DIVISION_BY_ZERO', 67108864)
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_OVERFLOW_SHIFT = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_OVERFLOW_SHIFT', 27)
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_OVERFLOW_WIDTH = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_OVERFLOW_WIDTH', 1)
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_OVERFLOW = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_OVERFLOW', 134217728)
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_UNDERFLOW_SHIFT = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_UNDERFLOW_SHIFT', 28)
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_UNDERFLOW_WIDTH = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_UNDERFLOW_WIDTH', 1)
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_UNDERFLOW = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_UNDERFLOW', 268435456)
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_INEXACT_SHIFT = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_INEXACT_SHIFT', 29)
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_INEXACT_WIDTH = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_INEXACT_WIDTH', 1)
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_INEXACT = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_INEXACT', 536870912)
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_INT_DIVISION_BY_ZERO_SHIFT = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_INT_DIVISION_BY_ZERO_SHIFT', 30)
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_INT_DIVISION_BY_ZERO_WIDTH = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_INT_DIVISION_BY_ZERO_WIDTH', 1)
AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_INT_DIVISION_BY_ZERO = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_INT_DIVISION_BY_ZERO', 1073741824)
AMD_COMPUTE_PGM_RSRC_TWO_RESERVED1_SHIFT = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_RESERVED1_SHIFT', 31)
AMD_COMPUTE_PGM_RSRC_TWO_RESERVED1_WIDTH = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_RESERVED1_WIDTH', 1)
AMD_COMPUTE_PGM_RSRC_TWO_RESERVED1 = enum_amd_compute_pgm_rsrc_two_t.define('AMD_COMPUTE_PGM_RSRC_TWO_RESERVED1', -2147483648)

enum_amd_element_byte_size_t = CEnum(ctypes.c_uint32)
AMD_ELEMENT_BYTE_SIZE_2 = enum_amd_element_byte_size_t.define('AMD_ELEMENT_BYTE_SIZE_2', 0)
AMD_ELEMENT_BYTE_SIZE_4 = enum_amd_element_byte_size_t.define('AMD_ELEMENT_BYTE_SIZE_4', 1)
AMD_ELEMENT_BYTE_SIZE_8 = enum_amd_element_byte_size_t.define('AMD_ELEMENT_BYTE_SIZE_8', 2)
AMD_ELEMENT_BYTE_SIZE_16 = enum_amd_element_byte_size_t.define('AMD_ELEMENT_BYTE_SIZE_16', 3)

amd_kernel_code_properties32_t = ctypes.c_uint32
enum_amd_kernel_code_properties_t = CEnum(ctypes.c_int32)
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER_SHIFT = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER_SHIFT', 0)
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER_WIDTH = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER_WIDTH', 1)
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER', 1)
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_PTR_SHIFT = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_PTR_SHIFT', 1)
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_PTR_WIDTH = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_PTR_WIDTH', 1)
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_PTR = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_PTR', 2)
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_QUEUE_PTR_SHIFT = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_QUEUE_PTR_SHIFT', 2)
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_QUEUE_PTR_WIDTH = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_QUEUE_PTR_WIDTH', 1)
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_QUEUE_PTR = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_QUEUE_PTR', 4)
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_KERNARG_SEGMENT_PTR_SHIFT = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_KERNARG_SEGMENT_PTR_SHIFT', 3)
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_KERNARG_SEGMENT_PTR_WIDTH = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_KERNARG_SEGMENT_PTR_WIDTH', 1)
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_KERNARG_SEGMENT_PTR = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_KERNARG_SEGMENT_PTR', 8)
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_ID_SHIFT = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_ID_SHIFT', 4)
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_ID_WIDTH = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_ID_WIDTH', 1)
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_ID = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_ID', 16)
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_FLAT_SCRATCH_INIT_SHIFT = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_FLAT_SCRATCH_INIT_SHIFT', 5)
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_FLAT_SCRATCH_INIT_WIDTH = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_FLAT_SCRATCH_INIT_WIDTH', 1)
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_FLAT_SCRATCH_INIT = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_FLAT_SCRATCH_INIT', 32)
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE_SHIFT = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE_SHIFT', 6)
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE_WIDTH = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE_WIDTH', 1)
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE', 64)
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_X_SHIFT = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_X_SHIFT', 7)
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_X_WIDTH = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_X_WIDTH', 1)
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_X = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_X', 128)
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Y_SHIFT = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Y_SHIFT', 8)
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Y_WIDTH = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Y_WIDTH', 1)
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Y = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Y', 256)
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Z_SHIFT = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Z_SHIFT', 9)
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Z_WIDTH = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Z_WIDTH', 1)
AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Z = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Z', 512)
AMD_KERNEL_CODE_PROPERTIES_ENABLE_WAVEFRONT_SIZE32_SHIFT = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_WAVEFRONT_SIZE32_SHIFT', 10)
AMD_KERNEL_CODE_PROPERTIES_ENABLE_WAVEFRONT_SIZE32_WIDTH = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_WAVEFRONT_SIZE32_WIDTH', 1)
AMD_KERNEL_CODE_PROPERTIES_ENABLE_WAVEFRONT_SIZE32 = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_WAVEFRONT_SIZE32', 1024)
AMD_KERNEL_CODE_PROPERTIES_RESERVED1_SHIFT = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_RESERVED1_SHIFT', 11)
AMD_KERNEL_CODE_PROPERTIES_RESERVED1_WIDTH = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_RESERVED1_WIDTH', 5)
AMD_KERNEL_CODE_PROPERTIES_RESERVED1 = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_RESERVED1', 63488)
AMD_KERNEL_CODE_PROPERTIES_ENABLE_ORDERED_APPEND_GDS_SHIFT = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_ORDERED_APPEND_GDS_SHIFT', 16)
AMD_KERNEL_CODE_PROPERTIES_ENABLE_ORDERED_APPEND_GDS_WIDTH = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_ORDERED_APPEND_GDS_WIDTH', 1)
AMD_KERNEL_CODE_PROPERTIES_ENABLE_ORDERED_APPEND_GDS = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_ENABLE_ORDERED_APPEND_GDS', 65536)
AMD_KERNEL_CODE_PROPERTIES_PRIVATE_ELEMENT_SIZE_SHIFT = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_PRIVATE_ELEMENT_SIZE_SHIFT', 17)
AMD_KERNEL_CODE_PROPERTIES_PRIVATE_ELEMENT_SIZE_WIDTH = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_PRIVATE_ELEMENT_SIZE_WIDTH', 2)
AMD_KERNEL_CODE_PROPERTIES_PRIVATE_ELEMENT_SIZE = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_PRIVATE_ELEMENT_SIZE', 393216)
AMD_KERNEL_CODE_PROPERTIES_IS_PTR64_SHIFT = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_IS_PTR64_SHIFT', 19)
AMD_KERNEL_CODE_PROPERTIES_IS_PTR64_WIDTH = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_IS_PTR64_WIDTH', 1)
AMD_KERNEL_CODE_PROPERTIES_IS_PTR64 = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_IS_PTR64', 524288)
AMD_KERNEL_CODE_PROPERTIES_IS_DYNAMIC_CALLSTACK_SHIFT = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_IS_DYNAMIC_CALLSTACK_SHIFT', 20)
AMD_KERNEL_CODE_PROPERTIES_IS_DYNAMIC_CALLSTACK_WIDTH = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_IS_DYNAMIC_CALLSTACK_WIDTH', 1)
AMD_KERNEL_CODE_PROPERTIES_IS_DYNAMIC_CALLSTACK = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_IS_DYNAMIC_CALLSTACK', 1048576)
AMD_KERNEL_CODE_PROPERTIES_IS_DEBUG_ENABLED_SHIFT = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_IS_DEBUG_ENABLED_SHIFT', 21)
AMD_KERNEL_CODE_PROPERTIES_IS_DEBUG_ENABLED_WIDTH = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_IS_DEBUG_ENABLED_WIDTH', 1)
AMD_KERNEL_CODE_PROPERTIES_IS_DEBUG_ENABLED = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_IS_DEBUG_ENABLED', 2097152)
AMD_KERNEL_CODE_PROPERTIES_IS_XNACK_ENABLED_SHIFT = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_IS_XNACK_ENABLED_SHIFT', 22)
AMD_KERNEL_CODE_PROPERTIES_IS_XNACK_ENABLED_WIDTH = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_IS_XNACK_ENABLED_WIDTH', 1)
AMD_KERNEL_CODE_PROPERTIES_IS_XNACK_ENABLED = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_IS_XNACK_ENABLED', 4194304)
AMD_KERNEL_CODE_PROPERTIES_RESERVED2_SHIFT = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_RESERVED2_SHIFT', 23)
AMD_KERNEL_CODE_PROPERTIES_RESERVED2_WIDTH = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_RESERVED2_WIDTH', 9)
AMD_KERNEL_CODE_PROPERTIES_RESERVED2 = enum_amd_kernel_code_properties_t.define('AMD_KERNEL_CODE_PROPERTIES_RESERVED2', -8388608)

amd_powertwo8_t = ctypes.c_ubyte
enum_amd_powertwo_t = CEnum(ctypes.c_uint32)
AMD_POWERTWO_1 = enum_amd_powertwo_t.define('AMD_POWERTWO_1', 0)
AMD_POWERTWO_2 = enum_amd_powertwo_t.define('AMD_POWERTWO_2', 1)
AMD_POWERTWO_4 = enum_amd_powertwo_t.define('AMD_POWERTWO_4', 2)
AMD_POWERTWO_8 = enum_amd_powertwo_t.define('AMD_POWERTWO_8', 3)
AMD_POWERTWO_16 = enum_amd_powertwo_t.define('AMD_POWERTWO_16', 4)
AMD_POWERTWO_32 = enum_amd_powertwo_t.define('AMD_POWERTWO_32', 5)
AMD_POWERTWO_64 = enum_amd_powertwo_t.define('AMD_POWERTWO_64', 6)
AMD_POWERTWO_128 = enum_amd_powertwo_t.define('AMD_POWERTWO_128', 7)
AMD_POWERTWO_256 = enum_amd_powertwo_t.define('AMD_POWERTWO_256', 8)

amd_enabled_control_directive64_t = ctypes.c_uint64
enum_amd_enabled_control_directive_t = CEnum(ctypes.c_uint32)
AMD_ENABLED_CONTROL_DIRECTIVE_ENABLE_BREAK_EXCEPTIONS = enum_amd_enabled_control_directive_t.define('AMD_ENABLED_CONTROL_DIRECTIVE_ENABLE_BREAK_EXCEPTIONS', 1)
AMD_ENABLED_CONTROL_DIRECTIVE_ENABLE_DETECT_EXCEPTIONS = enum_amd_enabled_control_directive_t.define('AMD_ENABLED_CONTROL_DIRECTIVE_ENABLE_DETECT_EXCEPTIONS', 2)
AMD_ENABLED_CONTROL_DIRECTIVE_MAX_DYNAMIC_GROUP_SIZE = enum_amd_enabled_control_directive_t.define('AMD_ENABLED_CONTROL_DIRECTIVE_MAX_DYNAMIC_GROUP_SIZE', 4)
AMD_ENABLED_CONTROL_DIRECTIVE_MAX_FLAT_GRID_SIZE = enum_amd_enabled_control_directive_t.define('AMD_ENABLED_CONTROL_DIRECTIVE_MAX_FLAT_GRID_SIZE', 8)
AMD_ENABLED_CONTROL_DIRECTIVE_MAX_FLAT_WORKGROUP_SIZE = enum_amd_enabled_control_directive_t.define('AMD_ENABLED_CONTROL_DIRECTIVE_MAX_FLAT_WORKGROUP_SIZE', 16)
AMD_ENABLED_CONTROL_DIRECTIVE_REQUIRED_DIM = enum_amd_enabled_control_directive_t.define('AMD_ENABLED_CONTROL_DIRECTIVE_REQUIRED_DIM', 32)
AMD_ENABLED_CONTROL_DIRECTIVE_REQUIRED_GRID_SIZE = enum_amd_enabled_control_directive_t.define('AMD_ENABLED_CONTROL_DIRECTIVE_REQUIRED_GRID_SIZE', 64)
AMD_ENABLED_CONTROL_DIRECTIVE_REQUIRED_WORKGROUP_SIZE = enum_amd_enabled_control_directive_t.define('AMD_ENABLED_CONTROL_DIRECTIVE_REQUIRED_WORKGROUP_SIZE', 128)
AMD_ENABLED_CONTROL_DIRECTIVE_REQUIRE_NO_PARTIAL_WORKGROUPS = enum_amd_enabled_control_directive_t.define('AMD_ENABLED_CONTROL_DIRECTIVE_REQUIRE_NO_PARTIAL_WORKGROUPS', 256)

amd_exception_kind16_t = ctypes.c_uint16
enum_amd_exception_kind_t = CEnum(ctypes.c_uint32)
AMD_EXCEPTION_KIND_INVALID_OPERATION = enum_amd_exception_kind_t.define('AMD_EXCEPTION_KIND_INVALID_OPERATION', 1)
AMD_EXCEPTION_KIND_DIVISION_BY_ZERO = enum_amd_exception_kind_t.define('AMD_EXCEPTION_KIND_DIVISION_BY_ZERO', 2)
AMD_EXCEPTION_KIND_OVERFLOW = enum_amd_exception_kind_t.define('AMD_EXCEPTION_KIND_OVERFLOW', 4)
AMD_EXCEPTION_KIND_UNDERFLOW = enum_amd_exception_kind_t.define('AMD_EXCEPTION_KIND_UNDERFLOW', 8)
AMD_EXCEPTION_KIND_INEXACT = enum_amd_exception_kind_t.define('AMD_EXCEPTION_KIND_INEXACT', 16)

class struct_amd_control_directives_s(Struct): pass
struct_amd_control_directives_s._fields_ = [
  ('enabled_control_directives', amd_enabled_control_directive64_t),
  ('enable_break_exceptions', uint16_t),
  ('enable_detect_exceptions', uint16_t),
  ('max_dynamic_group_size', uint32_t),
  ('max_flat_grid_size', uint64_t),
  ('max_flat_workgroup_size', uint32_t),
  ('required_dim', uint8_t),
  ('reserved1', (uint8_t * 3)),
  ('required_grid_size', (uint64_t * 3)),
  ('required_workgroup_size', (uint32_t * 3)),
  ('reserved2', (uint8_t * 60)),
]
amd_control_directives_t = struct_amd_control_directives_s
class struct_amd_kernel_code_s(Struct): pass
struct_amd_kernel_code_s._fields_ = [
  ('amd_kernel_code_version_major', amd_kernel_code_version32_t),
  ('amd_kernel_code_version_minor', amd_kernel_code_version32_t),
  ('amd_machine_kind', amd_machine_kind16_t),
  ('amd_machine_version_major', amd_machine_version16_t),
  ('amd_machine_version_minor', amd_machine_version16_t),
  ('amd_machine_version_stepping', amd_machine_version16_t),
  ('kernel_code_entry_byte_offset', int64_t),
  ('kernel_code_prefetch_byte_offset', int64_t),
  ('kernel_code_prefetch_byte_size', uint64_t),
  ('max_scratch_backing_memory_byte_size', uint64_t),
  ('compute_pgm_rsrc1', amd_compute_pgm_rsrc_one32_t),
  ('compute_pgm_rsrc2', amd_compute_pgm_rsrc_two32_t),
  ('kernel_code_properties', amd_kernel_code_properties32_t),
  ('workitem_private_segment_byte_size', uint32_t),
  ('workgroup_group_segment_byte_size', uint32_t),
  ('gds_segment_byte_size', uint32_t),
  ('kernarg_segment_byte_size', uint64_t),
  ('workgroup_fbarrier_count', uint32_t),
  ('wavefront_sgpr_count', uint16_t),
  ('workitem_vgpr_count', uint16_t),
  ('reserved_vgpr_first', uint16_t),
  ('reserved_vgpr_count', uint16_t),
  ('reserved_sgpr_first', uint16_t),
  ('reserved_sgpr_count', uint16_t),
  ('debug_wavefront_private_segment_offset_sgpr', uint16_t),
  ('debug_private_segment_buffer_sgpr', uint16_t),
  ('kernarg_segment_alignment', amd_powertwo8_t),
  ('group_segment_alignment', amd_powertwo8_t),
  ('private_segment_alignment', amd_powertwo8_t),
  ('wavefront_size', amd_powertwo8_t),
  ('call_convention', int32_t),
  ('reserved1', (uint8_t * 12)),
  ('runtime_loader_kernel_symbol', uint64_t),
  ('control_directives', amd_control_directives_t),
]
amd_kernel_code_t = struct_amd_kernel_code_s
class struct_amd_runtime_loader_debug_info_s(Struct): pass
struct_amd_runtime_loader_debug_info_s._fields_ = [
  ('elf_raw', ctypes.c_void_p),
  ('elf_size', size_t),
  ('kernel_name', ctypes.POINTER(ctypes.c_char)),
  ('owning_segment', ctypes.c_void_p),
]
amd_runtime_loader_debug_info_t = struct_amd_runtime_loader_debug_info_s
class struct_BrigModuleHeader(Struct): pass
BrigModule_t = ctypes.POINTER(struct_BrigModuleHeader)
_anonenum1 = CEnum(ctypes.c_uint32)
HSA_EXT_STATUS_ERROR_INVALID_PROGRAM = _anonenum1.define('HSA_EXT_STATUS_ERROR_INVALID_PROGRAM', 8192)
HSA_EXT_STATUS_ERROR_INVALID_MODULE = _anonenum1.define('HSA_EXT_STATUS_ERROR_INVALID_MODULE', 8193)
HSA_EXT_STATUS_ERROR_INCOMPATIBLE_MODULE = _anonenum1.define('HSA_EXT_STATUS_ERROR_INCOMPATIBLE_MODULE', 8194)
HSA_EXT_STATUS_ERROR_MODULE_ALREADY_INCLUDED = _anonenum1.define('HSA_EXT_STATUS_ERROR_MODULE_ALREADY_INCLUDED', 8195)
HSA_EXT_STATUS_ERROR_SYMBOL_MISMATCH = _anonenum1.define('HSA_EXT_STATUS_ERROR_SYMBOL_MISMATCH', 8196)
HSA_EXT_STATUS_ERROR_FINALIZATION_FAILED = _anonenum1.define('HSA_EXT_STATUS_ERROR_FINALIZATION_FAILED', 8197)
HSA_EXT_STATUS_ERROR_DIRECTIVE_MISMATCH = _anonenum1.define('HSA_EXT_STATUS_ERROR_DIRECTIVE_MISMATCH', 8198)

hsa_ext_module_t = ctypes.POINTER(struct_BrigModuleHeader)
class struct_hsa_ext_program_s(Struct): pass
struct_hsa_ext_program_s._fields_ = [
  ('handle', uint64_t),
]
hsa_ext_program_t = struct_hsa_ext_program_s
try: (hsa_ext_program_create:=dll.hsa_ext_program_create).restype, hsa_ext_program_create.argtypes = hsa_status_t, [hsa_machine_model_t, hsa_profile_t, hsa_default_float_rounding_mode_t, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(hsa_ext_program_t)]
except AttributeError: pass

try: (hsa_ext_program_destroy:=dll.hsa_ext_program_destroy).restype, hsa_ext_program_destroy.argtypes = hsa_status_t, [hsa_ext_program_t]
except AttributeError: pass

try: (hsa_ext_program_add_module:=dll.hsa_ext_program_add_module).restype, hsa_ext_program_add_module.argtypes = hsa_status_t, [hsa_ext_program_t, hsa_ext_module_t]
except AttributeError: pass

try: (hsa_ext_program_iterate_modules:=dll.hsa_ext_program_iterate_modules).restype, hsa_ext_program_iterate_modules.argtypes = hsa_status_t, [hsa_ext_program_t, ctypes.CFUNCTYPE(hsa_status_t, hsa_ext_program_t, hsa_ext_module_t, ctypes.c_void_p), ctypes.c_void_p]
except AttributeError: pass

hsa_ext_program_info_t = CEnum(ctypes.c_uint32)
HSA_EXT_PROGRAM_INFO_MACHINE_MODEL = hsa_ext_program_info_t.define('HSA_EXT_PROGRAM_INFO_MACHINE_MODEL', 0)
HSA_EXT_PROGRAM_INFO_PROFILE = hsa_ext_program_info_t.define('HSA_EXT_PROGRAM_INFO_PROFILE', 1)
HSA_EXT_PROGRAM_INFO_DEFAULT_FLOAT_ROUNDING_MODE = hsa_ext_program_info_t.define('HSA_EXT_PROGRAM_INFO_DEFAULT_FLOAT_ROUNDING_MODE', 2)

try: (hsa_ext_program_get_info:=dll.hsa_ext_program_get_info).restype, hsa_ext_program_get_info.argtypes = hsa_status_t, [hsa_ext_program_t, hsa_ext_program_info_t, ctypes.c_void_p]
except AttributeError: pass

hsa_ext_finalizer_call_convention_t = CEnum(ctypes.c_int32)
HSA_EXT_FINALIZER_CALL_CONVENTION_AUTO = hsa_ext_finalizer_call_convention_t.define('HSA_EXT_FINALIZER_CALL_CONVENTION_AUTO', -1)

class struct_hsa_ext_control_directives_s(Struct): pass
struct_hsa_ext_control_directives_s._fields_ = [
  ('control_directives_mask', uint64_t),
  ('break_exceptions_mask', uint16_t),
  ('detect_exceptions_mask', uint16_t),
  ('max_dynamic_group_size', uint32_t),
  ('max_flat_grid_size', uint64_t),
  ('max_flat_workgroup_size', uint32_t),
  ('reserved1', uint32_t),
  ('required_grid_size', (uint64_t * 3)),
  ('required_workgroup_size', hsa_dim3_t),
  ('required_dim', uint8_t),
  ('reserved2', (uint8_t * 75)),
]
hsa_ext_control_directives_t = struct_hsa_ext_control_directives_s
try: (hsa_ext_program_finalize:=dll.hsa_ext_program_finalize).restype, hsa_ext_program_finalize.argtypes = hsa_status_t, [hsa_ext_program_t, hsa_isa_t, int32_t, hsa_ext_control_directives_t, ctypes.POINTER(ctypes.c_char), hsa_code_object_type_t, ctypes.POINTER(hsa_code_object_t)]
except AttributeError: pass

class struct_hsa_ext_finalizer_1_00_pfn_s(Struct): pass
struct_hsa_ext_finalizer_1_00_pfn_s._fields_ = [
  ('hsa_ext_program_create', ctypes.CFUNCTYPE(hsa_status_t, hsa_machine_model_t, hsa_profile_t, hsa_default_float_rounding_mode_t, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(hsa_ext_program_t))),
  ('hsa_ext_program_destroy', ctypes.CFUNCTYPE(hsa_status_t, hsa_ext_program_t)),
  ('hsa_ext_program_add_module', ctypes.CFUNCTYPE(hsa_status_t, hsa_ext_program_t, hsa_ext_module_t)),
  ('hsa_ext_program_iterate_modules', ctypes.CFUNCTYPE(hsa_status_t, hsa_ext_program_t, ctypes.CFUNCTYPE(hsa_status_t, hsa_ext_program_t, hsa_ext_module_t, ctypes.c_void_p), ctypes.c_void_p)),
  ('hsa_ext_program_get_info', ctypes.CFUNCTYPE(hsa_status_t, hsa_ext_program_t, hsa_ext_program_info_t, ctypes.c_void_p)),
  ('hsa_ext_program_finalize', ctypes.CFUNCTYPE(hsa_status_t, hsa_ext_program_t, hsa_isa_t, int32_t, hsa_ext_control_directives_t, ctypes.POINTER(ctypes.c_char), hsa_code_object_type_t, ctypes.POINTER(hsa_code_object_t))),
]
hsa_ext_finalizer_1_00_pfn_t = struct_hsa_ext_finalizer_1_00_pfn_s
_anonenum2 = CEnum(ctypes.c_uint32)
HSA_EXT_STATUS_ERROR_IMAGE_FORMAT_UNSUPPORTED = _anonenum2.define('HSA_EXT_STATUS_ERROR_IMAGE_FORMAT_UNSUPPORTED', 12288)
HSA_EXT_STATUS_ERROR_IMAGE_SIZE_UNSUPPORTED = _anonenum2.define('HSA_EXT_STATUS_ERROR_IMAGE_SIZE_UNSUPPORTED', 12289)
HSA_EXT_STATUS_ERROR_IMAGE_PITCH_UNSUPPORTED = _anonenum2.define('HSA_EXT_STATUS_ERROR_IMAGE_PITCH_UNSUPPORTED', 12290)
HSA_EXT_STATUS_ERROR_SAMPLER_DESCRIPTOR_UNSUPPORTED = _anonenum2.define('HSA_EXT_STATUS_ERROR_SAMPLER_DESCRIPTOR_UNSUPPORTED', 12291)

_anonenum3 = CEnum(ctypes.c_uint32)
HSA_EXT_AGENT_INFO_IMAGE_1D_MAX_ELEMENTS = _anonenum3.define('HSA_EXT_AGENT_INFO_IMAGE_1D_MAX_ELEMENTS', 12288)
HSA_EXT_AGENT_INFO_IMAGE_1DA_MAX_ELEMENTS = _anonenum3.define('HSA_EXT_AGENT_INFO_IMAGE_1DA_MAX_ELEMENTS', 12289)
HSA_EXT_AGENT_INFO_IMAGE_1DB_MAX_ELEMENTS = _anonenum3.define('HSA_EXT_AGENT_INFO_IMAGE_1DB_MAX_ELEMENTS', 12290)
HSA_EXT_AGENT_INFO_IMAGE_2D_MAX_ELEMENTS = _anonenum3.define('HSA_EXT_AGENT_INFO_IMAGE_2D_MAX_ELEMENTS', 12291)
HSA_EXT_AGENT_INFO_IMAGE_2DA_MAX_ELEMENTS = _anonenum3.define('HSA_EXT_AGENT_INFO_IMAGE_2DA_MAX_ELEMENTS', 12292)
HSA_EXT_AGENT_INFO_IMAGE_2DDEPTH_MAX_ELEMENTS = _anonenum3.define('HSA_EXT_AGENT_INFO_IMAGE_2DDEPTH_MAX_ELEMENTS', 12293)
HSA_EXT_AGENT_INFO_IMAGE_2DADEPTH_MAX_ELEMENTS = _anonenum3.define('HSA_EXT_AGENT_INFO_IMAGE_2DADEPTH_MAX_ELEMENTS', 12294)
HSA_EXT_AGENT_INFO_IMAGE_3D_MAX_ELEMENTS = _anonenum3.define('HSA_EXT_AGENT_INFO_IMAGE_3D_MAX_ELEMENTS', 12295)
HSA_EXT_AGENT_INFO_IMAGE_ARRAY_MAX_LAYERS = _anonenum3.define('HSA_EXT_AGENT_INFO_IMAGE_ARRAY_MAX_LAYERS', 12296)
HSA_EXT_AGENT_INFO_MAX_IMAGE_RD_HANDLES = _anonenum3.define('HSA_EXT_AGENT_INFO_MAX_IMAGE_RD_HANDLES', 12297)
HSA_EXT_AGENT_INFO_MAX_IMAGE_RORW_HANDLES = _anonenum3.define('HSA_EXT_AGENT_INFO_MAX_IMAGE_RORW_HANDLES', 12298)
HSA_EXT_AGENT_INFO_MAX_SAMPLER_HANDLERS = _anonenum3.define('HSA_EXT_AGENT_INFO_MAX_SAMPLER_HANDLERS', 12299)
HSA_EXT_AGENT_INFO_IMAGE_LINEAR_ROW_PITCH_ALIGNMENT = _anonenum3.define('HSA_EXT_AGENT_INFO_IMAGE_LINEAR_ROW_PITCH_ALIGNMENT', 12300)

hsa_ext_image_channel_type_t = CEnum(ctypes.c_uint32)
HSA_EXT_IMAGE_CHANNEL_TYPE_SNORM_INT8 = hsa_ext_image_channel_type_t.define('HSA_EXT_IMAGE_CHANNEL_TYPE_SNORM_INT8', 0)
HSA_EXT_IMAGE_CHANNEL_TYPE_SNORM_INT16 = hsa_ext_image_channel_type_t.define('HSA_EXT_IMAGE_CHANNEL_TYPE_SNORM_INT16', 1)
HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT8 = hsa_ext_image_channel_type_t.define('HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT8', 2)
HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT16 = hsa_ext_image_channel_type_t.define('HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT16', 3)
HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT24 = hsa_ext_image_channel_type_t.define('HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT24', 4)
HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_555 = hsa_ext_image_channel_type_t.define('HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_555', 5)
HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_565 = hsa_ext_image_channel_type_t.define('HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_565', 6)
HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_101010 = hsa_ext_image_channel_type_t.define('HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_101010', 7)
HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT8 = hsa_ext_image_channel_type_t.define('HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT8', 8)
HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT16 = hsa_ext_image_channel_type_t.define('HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT16', 9)
HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT32 = hsa_ext_image_channel_type_t.define('HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT32', 10)
HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8 = hsa_ext_image_channel_type_t.define('HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8', 11)
HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16 = hsa_ext_image_channel_type_t.define('HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16', 12)
HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32 = hsa_ext_image_channel_type_t.define('HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32', 13)
HSA_EXT_IMAGE_CHANNEL_TYPE_HALF_FLOAT = hsa_ext_image_channel_type_t.define('HSA_EXT_IMAGE_CHANNEL_TYPE_HALF_FLOAT', 14)
HSA_EXT_IMAGE_CHANNEL_TYPE_FLOAT = hsa_ext_image_channel_type_t.define('HSA_EXT_IMAGE_CHANNEL_TYPE_FLOAT', 15)

hsa_ext_image_channel_order_t = CEnum(ctypes.c_uint32)
HSA_EXT_IMAGE_CHANNEL_ORDER_A = hsa_ext_image_channel_order_t.define('HSA_EXT_IMAGE_CHANNEL_ORDER_A', 0)
HSA_EXT_IMAGE_CHANNEL_ORDER_R = hsa_ext_image_channel_order_t.define('HSA_EXT_IMAGE_CHANNEL_ORDER_R', 1)
HSA_EXT_IMAGE_CHANNEL_ORDER_RX = hsa_ext_image_channel_order_t.define('HSA_EXT_IMAGE_CHANNEL_ORDER_RX', 2)
HSA_EXT_IMAGE_CHANNEL_ORDER_RG = hsa_ext_image_channel_order_t.define('HSA_EXT_IMAGE_CHANNEL_ORDER_RG', 3)
HSA_EXT_IMAGE_CHANNEL_ORDER_RGX = hsa_ext_image_channel_order_t.define('HSA_EXT_IMAGE_CHANNEL_ORDER_RGX', 4)
HSA_EXT_IMAGE_CHANNEL_ORDER_RA = hsa_ext_image_channel_order_t.define('HSA_EXT_IMAGE_CHANNEL_ORDER_RA', 5)
HSA_EXT_IMAGE_CHANNEL_ORDER_RGB = hsa_ext_image_channel_order_t.define('HSA_EXT_IMAGE_CHANNEL_ORDER_RGB', 6)
HSA_EXT_IMAGE_CHANNEL_ORDER_RGBX = hsa_ext_image_channel_order_t.define('HSA_EXT_IMAGE_CHANNEL_ORDER_RGBX', 7)
HSA_EXT_IMAGE_CHANNEL_ORDER_RGBA = hsa_ext_image_channel_order_t.define('HSA_EXT_IMAGE_CHANNEL_ORDER_RGBA', 8)
HSA_EXT_IMAGE_CHANNEL_ORDER_BGRA = hsa_ext_image_channel_order_t.define('HSA_EXT_IMAGE_CHANNEL_ORDER_BGRA', 9)
HSA_EXT_IMAGE_CHANNEL_ORDER_ARGB = hsa_ext_image_channel_order_t.define('HSA_EXT_IMAGE_CHANNEL_ORDER_ARGB', 10)
HSA_EXT_IMAGE_CHANNEL_ORDER_ABGR = hsa_ext_image_channel_order_t.define('HSA_EXT_IMAGE_CHANNEL_ORDER_ABGR', 11)
HSA_EXT_IMAGE_CHANNEL_ORDER_SRGB = hsa_ext_image_channel_order_t.define('HSA_EXT_IMAGE_CHANNEL_ORDER_SRGB', 12)
HSA_EXT_IMAGE_CHANNEL_ORDER_SRGBX = hsa_ext_image_channel_order_t.define('HSA_EXT_IMAGE_CHANNEL_ORDER_SRGBX', 13)
HSA_EXT_IMAGE_CHANNEL_ORDER_SRGBA = hsa_ext_image_channel_order_t.define('HSA_EXT_IMAGE_CHANNEL_ORDER_SRGBA', 14)
HSA_EXT_IMAGE_CHANNEL_ORDER_SBGRA = hsa_ext_image_channel_order_t.define('HSA_EXT_IMAGE_CHANNEL_ORDER_SBGRA', 15)
HSA_EXT_IMAGE_CHANNEL_ORDER_INTENSITY = hsa_ext_image_channel_order_t.define('HSA_EXT_IMAGE_CHANNEL_ORDER_INTENSITY', 16)
HSA_EXT_IMAGE_CHANNEL_ORDER_LUMINANCE = hsa_ext_image_channel_order_t.define('HSA_EXT_IMAGE_CHANNEL_ORDER_LUMINANCE', 17)
HSA_EXT_IMAGE_CHANNEL_ORDER_DEPTH = hsa_ext_image_channel_order_t.define('HSA_EXT_IMAGE_CHANNEL_ORDER_DEPTH', 18)
HSA_EXT_IMAGE_CHANNEL_ORDER_DEPTH_STENCIL = hsa_ext_image_channel_order_t.define('HSA_EXT_IMAGE_CHANNEL_ORDER_DEPTH_STENCIL', 19)

hsa_ext_image_capability_t = CEnum(ctypes.c_uint32)
HSA_EXT_IMAGE_CAPABILITY_NOT_SUPPORTED = hsa_ext_image_capability_t.define('HSA_EXT_IMAGE_CAPABILITY_NOT_SUPPORTED', 0)
HSA_EXT_IMAGE_CAPABILITY_READ_ONLY = hsa_ext_image_capability_t.define('HSA_EXT_IMAGE_CAPABILITY_READ_ONLY', 1)
HSA_EXT_IMAGE_CAPABILITY_WRITE_ONLY = hsa_ext_image_capability_t.define('HSA_EXT_IMAGE_CAPABILITY_WRITE_ONLY', 2)
HSA_EXT_IMAGE_CAPABILITY_READ_WRITE = hsa_ext_image_capability_t.define('HSA_EXT_IMAGE_CAPABILITY_READ_WRITE', 4)
HSA_EXT_IMAGE_CAPABILITY_READ_MODIFY_WRITE = hsa_ext_image_capability_t.define('HSA_EXT_IMAGE_CAPABILITY_READ_MODIFY_WRITE', 8)
HSA_EXT_IMAGE_CAPABILITY_ACCESS_INVARIANT_DATA_LAYOUT = hsa_ext_image_capability_t.define('HSA_EXT_IMAGE_CAPABILITY_ACCESS_INVARIANT_DATA_LAYOUT', 16)

hsa_ext_image_data_layout_t = CEnum(ctypes.c_uint32)
HSA_EXT_IMAGE_DATA_LAYOUT_OPAQUE = hsa_ext_image_data_layout_t.define('HSA_EXT_IMAGE_DATA_LAYOUT_OPAQUE', 0)
HSA_EXT_IMAGE_DATA_LAYOUT_LINEAR = hsa_ext_image_data_layout_t.define('HSA_EXT_IMAGE_DATA_LAYOUT_LINEAR', 1)

try: (hsa_ext_image_get_capability:=dll.hsa_ext_image_get_capability).restype, hsa_ext_image_get_capability.argtypes = hsa_status_t, [hsa_agent_t, hsa_ext_image_geometry_t, ctypes.POINTER(hsa_ext_image_format_t), ctypes.POINTER(uint32_t)]
except AttributeError: pass

try: (hsa_ext_image_get_capability_with_layout:=dll.hsa_ext_image_get_capability_with_layout).restype, hsa_ext_image_get_capability_with_layout.argtypes = hsa_status_t, [hsa_agent_t, hsa_ext_image_geometry_t, ctypes.POINTER(hsa_ext_image_format_t), hsa_ext_image_data_layout_t, ctypes.POINTER(uint32_t)]
except AttributeError: pass

class struct_hsa_ext_image_data_info_s(Struct): pass
struct_hsa_ext_image_data_info_s._fields_ = [
  ('size', size_t),
  ('alignment', size_t),
]
hsa_ext_image_data_info_t = struct_hsa_ext_image_data_info_s
try: (hsa_ext_image_data_get_info:=dll.hsa_ext_image_data_get_info).restype, hsa_ext_image_data_get_info.argtypes = hsa_status_t, [hsa_agent_t, ctypes.POINTER(hsa_ext_image_descriptor_t), hsa_access_permission_t, ctypes.POINTER(hsa_ext_image_data_info_t)]
except AttributeError: pass

try: (hsa_ext_image_data_get_info_with_layout:=dll.hsa_ext_image_data_get_info_with_layout).restype, hsa_ext_image_data_get_info_with_layout.argtypes = hsa_status_t, [hsa_agent_t, ctypes.POINTER(hsa_ext_image_descriptor_t), hsa_access_permission_t, hsa_ext_image_data_layout_t, size_t, size_t, ctypes.POINTER(hsa_ext_image_data_info_t)]
except AttributeError: pass

try: (hsa_ext_image_create:=dll.hsa_ext_image_create).restype, hsa_ext_image_create.argtypes = hsa_status_t, [hsa_agent_t, ctypes.POINTER(hsa_ext_image_descriptor_t), ctypes.c_void_p, hsa_access_permission_t, ctypes.POINTER(hsa_ext_image_t)]
except AttributeError: pass

try: (hsa_ext_image_create_with_layout:=dll.hsa_ext_image_create_with_layout).restype, hsa_ext_image_create_with_layout.argtypes = hsa_status_t, [hsa_agent_t, ctypes.POINTER(hsa_ext_image_descriptor_t), ctypes.c_void_p, hsa_access_permission_t, hsa_ext_image_data_layout_t, size_t, size_t, ctypes.POINTER(hsa_ext_image_t)]
except AttributeError: pass

try: (hsa_ext_image_destroy:=dll.hsa_ext_image_destroy).restype, hsa_ext_image_destroy.argtypes = hsa_status_t, [hsa_agent_t, hsa_ext_image_t]
except AttributeError: pass

try: (hsa_ext_image_copy:=dll.hsa_ext_image_copy).restype, hsa_ext_image_copy.argtypes = hsa_status_t, [hsa_agent_t, hsa_ext_image_t, ctypes.POINTER(hsa_dim3_t), hsa_ext_image_t, ctypes.POINTER(hsa_dim3_t), ctypes.POINTER(hsa_dim3_t)]
except AttributeError: pass

class struct_hsa_ext_image_region_s(Struct): pass
struct_hsa_ext_image_region_s._fields_ = [
  ('offset', hsa_dim3_t),
  ('range', hsa_dim3_t),
]
hsa_ext_image_region_t = struct_hsa_ext_image_region_s
try: (hsa_ext_image_import:=dll.hsa_ext_image_import).restype, hsa_ext_image_import.argtypes = hsa_status_t, [hsa_agent_t, ctypes.c_void_p, size_t, size_t, hsa_ext_image_t, ctypes.POINTER(hsa_ext_image_region_t)]
except AttributeError: pass

try: (hsa_ext_image_export:=dll.hsa_ext_image_export).restype, hsa_ext_image_export.argtypes = hsa_status_t, [hsa_agent_t, hsa_ext_image_t, ctypes.c_void_p, size_t, size_t, ctypes.POINTER(hsa_ext_image_region_t)]
except AttributeError: pass

try: (hsa_ext_image_clear:=dll.hsa_ext_image_clear).restype, hsa_ext_image_clear.argtypes = hsa_status_t, [hsa_agent_t, hsa_ext_image_t, ctypes.c_void_p, ctypes.POINTER(hsa_ext_image_region_t)]
except AttributeError: pass

class struct_hsa_ext_sampler_s(Struct): pass
struct_hsa_ext_sampler_s._fields_ = [
  ('handle', uint64_t),
]
hsa_ext_sampler_t = struct_hsa_ext_sampler_s
hsa_ext_sampler_addressing_mode_t = CEnum(ctypes.c_uint32)
HSA_EXT_SAMPLER_ADDRESSING_MODE_UNDEFINED = hsa_ext_sampler_addressing_mode_t.define('HSA_EXT_SAMPLER_ADDRESSING_MODE_UNDEFINED', 0)
HSA_EXT_SAMPLER_ADDRESSING_MODE_CLAMP_TO_EDGE = hsa_ext_sampler_addressing_mode_t.define('HSA_EXT_SAMPLER_ADDRESSING_MODE_CLAMP_TO_EDGE', 1)
HSA_EXT_SAMPLER_ADDRESSING_MODE_CLAMP_TO_BORDER = hsa_ext_sampler_addressing_mode_t.define('HSA_EXT_SAMPLER_ADDRESSING_MODE_CLAMP_TO_BORDER', 2)
HSA_EXT_SAMPLER_ADDRESSING_MODE_REPEAT = hsa_ext_sampler_addressing_mode_t.define('HSA_EXT_SAMPLER_ADDRESSING_MODE_REPEAT', 3)
HSA_EXT_SAMPLER_ADDRESSING_MODE_MIRRORED_REPEAT = hsa_ext_sampler_addressing_mode_t.define('HSA_EXT_SAMPLER_ADDRESSING_MODE_MIRRORED_REPEAT', 4)

hsa_ext_sampler_addressing_mode32_t = ctypes.c_uint32
hsa_ext_sampler_coordinate_mode_t = CEnum(ctypes.c_uint32)
HSA_EXT_SAMPLER_COORDINATE_MODE_UNNORMALIZED = hsa_ext_sampler_coordinate_mode_t.define('HSA_EXT_SAMPLER_COORDINATE_MODE_UNNORMALIZED', 0)
HSA_EXT_SAMPLER_COORDINATE_MODE_NORMALIZED = hsa_ext_sampler_coordinate_mode_t.define('HSA_EXT_SAMPLER_COORDINATE_MODE_NORMALIZED', 1)

hsa_ext_sampler_coordinate_mode32_t = ctypes.c_uint32
hsa_ext_sampler_filter_mode_t = CEnum(ctypes.c_uint32)
HSA_EXT_SAMPLER_FILTER_MODE_NEAREST = hsa_ext_sampler_filter_mode_t.define('HSA_EXT_SAMPLER_FILTER_MODE_NEAREST', 0)
HSA_EXT_SAMPLER_FILTER_MODE_LINEAR = hsa_ext_sampler_filter_mode_t.define('HSA_EXT_SAMPLER_FILTER_MODE_LINEAR', 1)

hsa_ext_sampler_filter_mode32_t = ctypes.c_uint32
class struct_hsa_ext_sampler_descriptor_s(Struct): pass
struct_hsa_ext_sampler_descriptor_s._fields_ = [
  ('coordinate_mode', hsa_ext_sampler_coordinate_mode32_t),
  ('filter_mode', hsa_ext_sampler_filter_mode32_t),
  ('address_mode', hsa_ext_sampler_addressing_mode32_t),
]
hsa_ext_sampler_descriptor_t = struct_hsa_ext_sampler_descriptor_s
class struct_hsa_ext_sampler_descriptor_v2_s(Struct): pass
struct_hsa_ext_sampler_descriptor_v2_s._fields_ = [
  ('coordinate_mode', hsa_ext_sampler_coordinate_mode32_t),
  ('filter_mode', hsa_ext_sampler_filter_mode32_t),
  ('address_modes', (hsa_ext_sampler_addressing_mode32_t * 3)),
]
hsa_ext_sampler_descriptor_v2_t = struct_hsa_ext_sampler_descriptor_v2_s
try: (hsa_ext_sampler_create:=dll.hsa_ext_sampler_create).restype, hsa_ext_sampler_create.argtypes = hsa_status_t, [hsa_agent_t, ctypes.POINTER(hsa_ext_sampler_descriptor_t), ctypes.POINTER(hsa_ext_sampler_t)]
except AttributeError: pass

try: (hsa_ext_sampler_create_v2:=dll.hsa_ext_sampler_create_v2).restype, hsa_ext_sampler_create_v2.argtypes = hsa_status_t, [hsa_agent_t, ctypes.POINTER(hsa_ext_sampler_descriptor_v2_t), ctypes.POINTER(hsa_ext_sampler_t)]
except AttributeError: pass

try: (hsa_ext_sampler_destroy:=dll.hsa_ext_sampler_destroy).restype, hsa_ext_sampler_destroy.argtypes = hsa_status_t, [hsa_agent_t, hsa_ext_sampler_t]
except AttributeError: pass

class struct_hsa_ext_images_1_00_pfn_s(Struct): pass
struct_hsa_ext_images_1_00_pfn_s._fields_ = [
  ('hsa_ext_image_get_capability', ctypes.CFUNCTYPE(hsa_status_t, hsa_agent_t, hsa_ext_image_geometry_t, ctypes.POINTER(hsa_ext_image_format_t), ctypes.POINTER(uint32_t))),
  ('hsa_ext_image_data_get_info', ctypes.CFUNCTYPE(hsa_status_t, hsa_agent_t, ctypes.POINTER(hsa_ext_image_descriptor_t), hsa_access_permission_t, ctypes.POINTER(hsa_ext_image_data_info_t))),
  ('hsa_ext_image_create', ctypes.CFUNCTYPE(hsa_status_t, hsa_agent_t, ctypes.POINTER(hsa_ext_image_descriptor_t), ctypes.c_void_p, hsa_access_permission_t, ctypes.POINTER(hsa_ext_image_t))),
  ('hsa_ext_image_destroy', ctypes.CFUNCTYPE(hsa_status_t, hsa_agent_t, hsa_ext_image_t)),
  ('hsa_ext_image_copy', ctypes.CFUNCTYPE(hsa_status_t, hsa_agent_t, hsa_ext_image_t, ctypes.POINTER(hsa_dim3_t), hsa_ext_image_t, ctypes.POINTER(hsa_dim3_t), ctypes.POINTER(hsa_dim3_t))),
  ('hsa_ext_image_import', ctypes.CFUNCTYPE(hsa_status_t, hsa_agent_t, ctypes.c_void_p, size_t, size_t, hsa_ext_image_t, ctypes.POINTER(hsa_ext_image_region_t))),
  ('hsa_ext_image_export', ctypes.CFUNCTYPE(hsa_status_t, hsa_agent_t, hsa_ext_image_t, ctypes.c_void_p, size_t, size_t, ctypes.POINTER(hsa_ext_image_region_t))),
  ('hsa_ext_image_clear', ctypes.CFUNCTYPE(hsa_status_t, hsa_agent_t, hsa_ext_image_t, ctypes.c_void_p, ctypes.POINTER(hsa_ext_image_region_t))),
  ('hsa_ext_sampler_create', ctypes.CFUNCTYPE(hsa_status_t, hsa_agent_t, ctypes.POINTER(hsa_ext_sampler_descriptor_t), ctypes.POINTER(hsa_ext_sampler_t))),
  ('hsa_ext_sampler_destroy', ctypes.CFUNCTYPE(hsa_status_t, hsa_agent_t, hsa_ext_sampler_t)),
]
hsa_ext_images_1_00_pfn_t = struct_hsa_ext_images_1_00_pfn_s
class struct_hsa_ext_images_1_pfn_s(Struct): pass
struct_hsa_ext_images_1_pfn_s._fields_ = [
  ('hsa_ext_image_get_capability', ctypes.CFUNCTYPE(hsa_status_t, hsa_agent_t, hsa_ext_image_geometry_t, ctypes.POINTER(hsa_ext_image_format_t), ctypes.POINTER(uint32_t))),
  ('hsa_ext_image_data_get_info', ctypes.CFUNCTYPE(hsa_status_t, hsa_agent_t, ctypes.POINTER(hsa_ext_image_descriptor_t), hsa_access_permission_t, ctypes.POINTER(hsa_ext_image_data_info_t))),
  ('hsa_ext_image_create', ctypes.CFUNCTYPE(hsa_status_t, hsa_agent_t, ctypes.POINTER(hsa_ext_image_descriptor_t), ctypes.c_void_p, hsa_access_permission_t, ctypes.POINTER(hsa_ext_image_t))),
  ('hsa_ext_image_destroy', ctypes.CFUNCTYPE(hsa_status_t, hsa_agent_t, hsa_ext_image_t)),
  ('hsa_ext_image_copy', ctypes.CFUNCTYPE(hsa_status_t, hsa_agent_t, hsa_ext_image_t, ctypes.POINTER(hsa_dim3_t), hsa_ext_image_t, ctypes.POINTER(hsa_dim3_t), ctypes.POINTER(hsa_dim3_t))),
  ('hsa_ext_image_import', ctypes.CFUNCTYPE(hsa_status_t, hsa_agent_t, ctypes.c_void_p, size_t, size_t, hsa_ext_image_t, ctypes.POINTER(hsa_ext_image_region_t))),
  ('hsa_ext_image_export', ctypes.CFUNCTYPE(hsa_status_t, hsa_agent_t, hsa_ext_image_t, ctypes.c_void_p, size_t, size_t, ctypes.POINTER(hsa_ext_image_region_t))),
  ('hsa_ext_image_clear', ctypes.CFUNCTYPE(hsa_status_t, hsa_agent_t, hsa_ext_image_t, ctypes.c_void_p, ctypes.POINTER(hsa_ext_image_region_t))),
  ('hsa_ext_sampler_create', ctypes.CFUNCTYPE(hsa_status_t, hsa_agent_t, ctypes.POINTER(hsa_ext_sampler_descriptor_t), ctypes.POINTER(hsa_ext_sampler_t))),
  ('hsa_ext_sampler_destroy', ctypes.CFUNCTYPE(hsa_status_t, hsa_agent_t, hsa_ext_sampler_t)),
  ('hsa_ext_image_get_capability_with_layout', ctypes.CFUNCTYPE(hsa_status_t, hsa_agent_t, hsa_ext_image_geometry_t, ctypes.POINTER(hsa_ext_image_format_t), hsa_ext_image_data_layout_t, ctypes.POINTER(uint32_t))),
  ('hsa_ext_image_data_get_info_with_layout', ctypes.CFUNCTYPE(hsa_status_t, hsa_agent_t, ctypes.POINTER(hsa_ext_image_descriptor_t), hsa_access_permission_t, hsa_ext_image_data_layout_t, size_t, size_t, ctypes.POINTER(hsa_ext_image_data_info_t))),
  ('hsa_ext_image_create_with_layout', ctypes.CFUNCTYPE(hsa_status_t, hsa_agent_t, ctypes.POINTER(hsa_ext_image_descriptor_t), ctypes.c_void_p, hsa_access_permission_t, hsa_ext_image_data_layout_t, size_t, size_t, ctypes.POINTER(hsa_ext_image_t))),
  ('hsa_ext_sampler_create_v2', ctypes.CFUNCTYPE(hsa_status_t, hsa_agent_t, ctypes.POINTER(hsa_ext_sampler_descriptor_v2_t), ctypes.POINTER(hsa_ext_sampler_t))),
]
hsa_ext_images_1_pfn_t = struct_hsa_ext_images_1_pfn_s
try: (hsa_ven_amd_aqlprofile_version_major:=dll.hsa_ven_amd_aqlprofile_version_major).restype, hsa_ven_amd_aqlprofile_version_major.argtypes = uint32_t, []
except AttributeError: pass

try: (hsa_ven_amd_aqlprofile_version_minor:=dll.hsa_ven_amd_aqlprofile_version_minor).restype, hsa_ven_amd_aqlprofile_version_minor.argtypes = uint32_t, []
except AttributeError: pass

hsa_ven_amd_aqlprofile_event_type_t = CEnum(ctypes.c_uint32)
HSA_VEN_AMD_AQLPROFILE_EVENT_TYPE_PMC = hsa_ven_amd_aqlprofile_event_type_t.define('HSA_VEN_AMD_AQLPROFILE_EVENT_TYPE_PMC', 0)
HSA_VEN_AMD_AQLPROFILE_EVENT_TYPE_TRACE = hsa_ven_amd_aqlprofile_event_type_t.define('HSA_VEN_AMD_AQLPROFILE_EVENT_TYPE_TRACE', 1)

hsa_ven_amd_aqlprofile_block_name_t = CEnum(ctypes.c_uint32)
HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_CPC = hsa_ven_amd_aqlprofile_block_name_t.define('HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_CPC', 0)
HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_CPF = hsa_ven_amd_aqlprofile_block_name_t.define('HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_CPF', 1)
HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GDS = hsa_ven_amd_aqlprofile_block_name_t.define('HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GDS', 2)
HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GRBM = hsa_ven_amd_aqlprofile_block_name_t.define('HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GRBM', 3)
HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GRBMSE = hsa_ven_amd_aqlprofile_block_name_t.define('HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GRBMSE', 4)
HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SPI = hsa_ven_amd_aqlprofile_block_name_t.define('HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SPI', 5)
HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ = hsa_ven_amd_aqlprofile_block_name_t.define('HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ', 6)
HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQCS = hsa_ven_amd_aqlprofile_block_name_t.define('HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQCS', 7)
HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SRBM = hsa_ven_amd_aqlprofile_block_name_t.define('HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SRBM', 8)
HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SX = hsa_ven_amd_aqlprofile_block_name_t.define('HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SX', 9)
HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA = hsa_ven_amd_aqlprofile_block_name_t.define('HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA', 10)
HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCA = hsa_ven_amd_aqlprofile_block_name_t.define('HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCA', 11)
HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCC = hsa_ven_amd_aqlprofile_block_name_t.define('HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCC', 12)
HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP = hsa_ven_amd_aqlprofile_block_name_t.define('HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP', 13)
HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TD = hsa_ven_amd_aqlprofile_block_name_t.define('HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TD', 14)
HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCARB = hsa_ven_amd_aqlprofile_block_name_t.define('HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCARB', 15)
HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCHUB = hsa_ven_amd_aqlprofile_block_name_t.define('HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCHUB', 16)
HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCMCBVM = hsa_ven_amd_aqlprofile_block_name_t.define('HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCMCBVM', 17)
HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCSEQ = hsa_ven_amd_aqlprofile_block_name_t.define('HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCSEQ', 18)
HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCVML2 = hsa_ven_amd_aqlprofile_block_name_t.define('HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCVML2', 19)
HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCXBAR = hsa_ven_amd_aqlprofile_block_name_t.define('HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCXBAR', 20)
HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_ATC = hsa_ven_amd_aqlprofile_block_name_t.define('HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_ATC', 21)
HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_ATCL2 = hsa_ven_amd_aqlprofile_block_name_t.define('HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_ATCL2', 22)
HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GCEA = hsa_ven_amd_aqlprofile_block_name_t.define('HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GCEA', 23)
HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_RPB = hsa_ven_amd_aqlprofile_block_name_t.define('HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_RPB', 24)
HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SDMA = hsa_ven_amd_aqlprofile_block_name_t.define('HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SDMA', 25)
HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GL1A = hsa_ven_amd_aqlprofile_block_name_t.define('HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GL1A', 26)
HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GL1C = hsa_ven_amd_aqlprofile_block_name_t.define('HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GL1C', 27)
HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GL2A = hsa_ven_amd_aqlprofile_block_name_t.define('HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GL2A', 28)
HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GL2C = hsa_ven_amd_aqlprofile_block_name_t.define('HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GL2C', 29)
HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GCR = hsa_ven_amd_aqlprofile_block_name_t.define('HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GCR', 30)
HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GUS = hsa_ven_amd_aqlprofile_block_name_t.define('HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GUS', 31)
HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_UMC = hsa_ven_amd_aqlprofile_block_name_t.define('HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_UMC', 32)
HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MMEA = hsa_ven_amd_aqlprofile_block_name_t.define('HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MMEA', 33)
HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER = hsa_ven_amd_aqlprofile_block_name_t.define('HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER', 34)

class hsa_ven_amd_aqlprofile_event_t(Struct): pass
hsa_ven_amd_aqlprofile_event_t._fields_ = [
  ('block_name', hsa_ven_amd_aqlprofile_block_name_t),
  ('block_index', uint32_t),
  ('counter_id', uint32_t),
]
try: (hsa_ven_amd_aqlprofile_validate_event:=dll.hsa_ven_amd_aqlprofile_validate_event).restype, hsa_ven_amd_aqlprofile_validate_event.argtypes = hsa_status_t, [hsa_agent_t, ctypes.POINTER(hsa_ven_amd_aqlprofile_event_t), ctypes.POINTER(ctypes.c_bool)]
except AttributeError: pass

hsa_ven_amd_aqlprofile_parameter_name_t = CEnum(ctypes.c_uint32)
HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_COMPUTE_UNIT_TARGET = hsa_ven_amd_aqlprofile_parameter_name_t.define('HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_COMPUTE_UNIT_TARGET', 0)
HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_VM_ID_MASK = hsa_ven_amd_aqlprofile_parameter_name_t.define('HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_VM_ID_MASK', 1)
HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_MASK = hsa_ven_amd_aqlprofile_parameter_name_t.define('HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_MASK', 2)
HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_TOKEN_MASK = hsa_ven_amd_aqlprofile_parameter_name_t.define('HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_TOKEN_MASK', 3)
HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_TOKEN_MASK2 = hsa_ven_amd_aqlprofile_parameter_name_t.define('HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_TOKEN_MASK2', 4)
HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_SE_MASK = hsa_ven_amd_aqlprofile_parameter_name_t.define('HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_SE_MASK', 5)
HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_SAMPLE_RATE = hsa_ven_amd_aqlprofile_parameter_name_t.define('HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_SAMPLE_RATE', 6)
HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_K_CONCURRENT = hsa_ven_amd_aqlprofile_parameter_name_t.define('HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_K_CONCURRENT', 7)
HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_SIMD_SELECTION = hsa_ven_amd_aqlprofile_parameter_name_t.define('HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_SIMD_SELECTION', 8)
HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_OCCUPANCY_MODE = hsa_ven_amd_aqlprofile_parameter_name_t.define('HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_OCCUPANCY_MODE', 9)
HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_ATT_BUFFER_SIZE = hsa_ven_amd_aqlprofile_parameter_name_t.define('HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_ATT_BUFFER_SIZE', 10)
HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_PERFCOUNTER_MASK = hsa_ven_amd_aqlprofile_parameter_name_t.define('HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_PERFCOUNTER_MASK', 240)
HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_PERFCOUNTER_CTRL = hsa_ven_amd_aqlprofile_parameter_name_t.define('HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_PERFCOUNTER_CTRL', 241)
HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_PERFCOUNTER_NAME = hsa_ven_amd_aqlprofile_parameter_name_t.define('HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_PERFCOUNTER_NAME', 242)

class hsa_ven_amd_aqlprofile_parameter_t(Struct): pass
hsa_ven_amd_aqlprofile_parameter_t._fields_ = [
  ('parameter_name', hsa_ven_amd_aqlprofile_parameter_name_t),
  ('value', uint32_t),
]
hsa_ven_amd_aqlprofile_att_marker_channel_t = CEnum(ctypes.c_uint32)
HSA_VEN_AMD_AQLPROFILE_ATT_CHANNEL_0 = hsa_ven_amd_aqlprofile_att_marker_channel_t.define('HSA_VEN_AMD_AQLPROFILE_ATT_CHANNEL_0', 0)
HSA_VEN_AMD_AQLPROFILE_ATT_CHANNEL_1 = hsa_ven_amd_aqlprofile_att_marker_channel_t.define('HSA_VEN_AMD_AQLPROFILE_ATT_CHANNEL_1', 1)
HSA_VEN_AMD_AQLPROFILE_ATT_CHANNEL_2 = hsa_ven_amd_aqlprofile_att_marker_channel_t.define('HSA_VEN_AMD_AQLPROFILE_ATT_CHANNEL_2', 2)
HSA_VEN_AMD_AQLPROFILE_ATT_CHANNEL_3 = hsa_ven_amd_aqlprofile_att_marker_channel_t.define('HSA_VEN_AMD_AQLPROFILE_ATT_CHANNEL_3', 3)

class hsa_ven_amd_aqlprofile_descriptor_t(Struct): pass
hsa_ven_amd_aqlprofile_descriptor_t._fields_ = [
  ('ptr', ctypes.c_void_p),
  ('size', uint32_t),
]
class hsa_ven_amd_aqlprofile_profile_t(Struct): pass
hsa_ven_amd_aqlprofile_profile_t._fields_ = [
  ('agent', hsa_agent_t),
  ('type', hsa_ven_amd_aqlprofile_event_type_t),
  ('events', ctypes.POINTER(hsa_ven_amd_aqlprofile_event_t)),
  ('event_count', uint32_t),
  ('parameters', ctypes.POINTER(hsa_ven_amd_aqlprofile_parameter_t)),
  ('parameter_count', uint32_t),
  ('output_buffer', hsa_ven_amd_aqlprofile_descriptor_t),
  ('command_buffer', hsa_ven_amd_aqlprofile_descriptor_t),
]
class hsa_ext_amd_aql_pm4_packet_t(Struct): pass
hsa_ext_amd_aql_pm4_packet_t._fields_ = [
  ('header', uint16_t),
  ('pm4_command', (uint16_t * 27)),
  ('completion_signal', hsa_signal_t),
]
try: (hsa_ven_amd_aqlprofile_start:=dll.hsa_ven_amd_aqlprofile_start).restype, hsa_ven_amd_aqlprofile_start.argtypes = hsa_status_t, [ctypes.POINTER(hsa_ven_amd_aqlprofile_profile_t), ctypes.POINTER(hsa_ext_amd_aql_pm4_packet_t)]
except AttributeError: pass

try: (hsa_ven_amd_aqlprofile_stop:=dll.hsa_ven_amd_aqlprofile_stop).restype, hsa_ven_amd_aqlprofile_stop.argtypes = hsa_status_t, [ctypes.POINTER(hsa_ven_amd_aqlprofile_profile_t), ctypes.POINTER(hsa_ext_amd_aql_pm4_packet_t)]
except AttributeError: pass

try: (hsa_ven_amd_aqlprofile_read:=dll.hsa_ven_amd_aqlprofile_read).restype, hsa_ven_amd_aqlprofile_read.argtypes = hsa_status_t, [ctypes.POINTER(hsa_ven_amd_aqlprofile_profile_t), ctypes.POINTER(hsa_ext_amd_aql_pm4_packet_t)]
except AttributeError: pass

try: HSA_VEN_AMD_AQLPROFILE_LEGACY_PM4_PACKET_SIZE = ctypes.c_uint32.in_dll(dll, 'HSA_VEN_AMD_AQLPROFILE_LEGACY_PM4_PACKET_SIZE')
except (ValueError,AttributeError): pass
try: (hsa_ven_amd_aqlprofile_legacy_get_pm4:=dll.hsa_ven_amd_aqlprofile_legacy_get_pm4).restype, hsa_ven_amd_aqlprofile_legacy_get_pm4.argtypes = hsa_status_t, [ctypes.POINTER(hsa_ext_amd_aql_pm4_packet_t), ctypes.c_void_p]
except AttributeError: pass

try: (hsa_ven_amd_aqlprofile_att_marker:=dll.hsa_ven_amd_aqlprofile_att_marker).restype, hsa_ven_amd_aqlprofile_att_marker.argtypes = hsa_status_t, [ctypes.POINTER(hsa_ven_amd_aqlprofile_profile_t), ctypes.POINTER(hsa_ext_amd_aql_pm4_packet_t), uint32_t, hsa_ven_amd_aqlprofile_att_marker_channel_t]
except AttributeError: pass

class hsa_ven_amd_aqlprofile_info_data_t(Struct): pass
class hsa_ven_amd_aqlprofile_info_data_t_0(ctypes.Union): pass
class hsa_ven_amd_aqlprofile_info_data_t_0_pmc_data(Struct): pass
hsa_ven_amd_aqlprofile_info_data_t_0_pmc_data._fields_ = [
  ('event', hsa_ven_amd_aqlprofile_event_t),
  ('result', uint64_t),
]
hsa_ven_amd_aqlprofile_info_data_t_0._fields_ = [
  ('pmc_data', hsa_ven_amd_aqlprofile_info_data_t_0_pmc_data),
  ('trace_data', hsa_ven_amd_aqlprofile_descriptor_t),
]
hsa_ven_amd_aqlprofile_info_data_t._anonymous_ = ['_0']
hsa_ven_amd_aqlprofile_info_data_t._fields_ = [
  ('sample_id', uint32_t),
  ('_0', hsa_ven_amd_aqlprofile_info_data_t_0),
]
class hsa_ven_amd_aqlprofile_id_query_t(Struct): pass
hsa_ven_amd_aqlprofile_id_query_t._fields_ = [
  ('name', ctypes.POINTER(ctypes.c_char)),
  ('id', uint32_t),
  ('instance_count', uint32_t),
]
hsa_ven_amd_aqlprofile_info_type_t = CEnum(ctypes.c_uint32)
HSA_VEN_AMD_AQLPROFILE_INFO_COMMAND_BUFFER_SIZE = hsa_ven_amd_aqlprofile_info_type_t.define('HSA_VEN_AMD_AQLPROFILE_INFO_COMMAND_BUFFER_SIZE', 0)
HSA_VEN_AMD_AQLPROFILE_INFO_PMC_DATA_SIZE = hsa_ven_amd_aqlprofile_info_type_t.define('HSA_VEN_AMD_AQLPROFILE_INFO_PMC_DATA_SIZE', 1)
HSA_VEN_AMD_AQLPROFILE_INFO_PMC_DATA = hsa_ven_amd_aqlprofile_info_type_t.define('HSA_VEN_AMD_AQLPROFILE_INFO_PMC_DATA', 2)
HSA_VEN_AMD_AQLPROFILE_INFO_TRACE_DATA = hsa_ven_amd_aqlprofile_info_type_t.define('HSA_VEN_AMD_AQLPROFILE_INFO_TRACE_DATA', 3)
HSA_VEN_AMD_AQLPROFILE_INFO_BLOCK_COUNTERS = hsa_ven_amd_aqlprofile_info_type_t.define('HSA_VEN_AMD_AQLPROFILE_INFO_BLOCK_COUNTERS', 4)
HSA_VEN_AMD_AQLPROFILE_INFO_BLOCK_ID = hsa_ven_amd_aqlprofile_info_type_t.define('HSA_VEN_AMD_AQLPROFILE_INFO_BLOCK_ID', 5)
HSA_VEN_AMD_AQLPROFILE_INFO_ENABLE_CMD = hsa_ven_amd_aqlprofile_info_type_t.define('HSA_VEN_AMD_AQLPROFILE_INFO_ENABLE_CMD', 6)
HSA_VEN_AMD_AQLPROFILE_INFO_DISABLE_CMD = hsa_ven_amd_aqlprofile_info_type_t.define('HSA_VEN_AMD_AQLPROFILE_INFO_DISABLE_CMD', 7)

hsa_ven_amd_aqlprofile_data_callback_t = ctypes.CFUNCTYPE(hsa_status_t, hsa_ven_amd_aqlprofile_info_type_t, ctypes.POINTER(hsa_ven_amd_aqlprofile_info_data_t), ctypes.c_void_p)
try: (hsa_ven_amd_aqlprofile_get_info:=dll.hsa_ven_amd_aqlprofile_get_info).restype, hsa_ven_amd_aqlprofile_get_info.argtypes = hsa_status_t, [ctypes.POINTER(hsa_ven_amd_aqlprofile_profile_t), hsa_ven_amd_aqlprofile_info_type_t, ctypes.c_void_p]
except AttributeError: pass

try: (hsa_ven_amd_aqlprofile_iterate_data:=dll.hsa_ven_amd_aqlprofile_iterate_data).restype, hsa_ven_amd_aqlprofile_iterate_data.argtypes = hsa_status_t, [ctypes.POINTER(hsa_ven_amd_aqlprofile_profile_t), hsa_ven_amd_aqlprofile_data_callback_t, ctypes.c_void_p]
except AttributeError: pass

try: (hsa_ven_amd_aqlprofile_error_string:=dll.hsa_ven_amd_aqlprofile_error_string).restype, hsa_ven_amd_aqlprofile_error_string.argtypes = hsa_status_t, [ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

hsa_ven_amd_aqlprofile_eventname_callback_t = ctypes.CFUNCTYPE(hsa_status_t, ctypes.c_int32, ctypes.POINTER(ctypes.c_char))
try: (hsa_ven_amd_aqlprofile_iterate_event_ids:=dll.hsa_ven_amd_aqlprofile_iterate_event_ids).restype, hsa_ven_amd_aqlprofile_iterate_event_ids.argtypes = hsa_status_t, [hsa_ven_amd_aqlprofile_eventname_callback_t]
except AttributeError: pass

hsa_ven_amd_aqlprofile_coordinate_callback_t = ctypes.CFUNCTYPE(hsa_status_t, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(ctypes.c_char), ctypes.c_void_p)
try: (hsa_ven_amd_aqlprofile_iterate_event_coord:=dll.hsa_ven_amd_aqlprofile_iterate_event_coord).restype, hsa_ven_amd_aqlprofile_iterate_event_coord.argtypes = hsa_status_t, [hsa_agent_t, hsa_ven_amd_aqlprofile_event_t, uint32_t, hsa_ven_amd_aqlprofile_coordinate_callback_t, ctypes.c_void_p]
except AttributeError: pass

class struct_hsa_ven_amd_aqlprofile_1_00_pfn_s(Struct): pass
struct_hsa_ven_amd_aqlprofile_1_00_pfn_s._fields_ = [
  ('hsa_ven_amd_aqlprofile_version_major', ctypes.CFUNCTYPE(uint32_t)),
  ('hsa_ven_amd_aqlprofile_version_minor', ctypes.CFUNCTYPE(uint32_t)),
  ('hsa_ven_amd_aqlprofile_error_string', ctypes.CFUNCTYPE(hsa_status_t, ctypes.POINTER(ctypes.POINTER(ctypes.c_char)))),
  ('hsa_ven_amd_aqlprofile_validate_event', ctypes.CFUNCTYPE(hsa_status_t, hsa_agent_t, ctypes.POINTER(hsa_ven_amd_aqlprofile_event_t), ctypes.POINTER(ctypes.c_bool))),
  ('hsa_ven_amd_aqlprofile_start', ctypes.CFUNCTYPE(hsa_status_t, ctypes.POINTER(hsa_ven_amd_aqlprofile_profile_t), ctypes.POINTER(hsa_ext_amd_aql_pm4_packet_t))),
  ('hsa_ven_amd_aqlprofile_stop', ctypes.CFUNCTYPE(hsa_status_t, ctypes.POINTER(hsa_ven_amd_aqlprofile_profile_t), ctypes.POINTER(hsa_ext_amd_aql_pm4_packet_t))),
  ('hsa_ven_amd_aqlprofile_read', ctypes.CFUNCTYPE(hsa_status_t, ctypes.POINTER(hsa_ven_amd_aqlprofile_profile_t), ctypes.POINTER(hsa_ext_amd_aql_pm4_packet_t))),
  ('hsa_ven_amd_aqlprofile_legacy_get_pm4', ctypes.CFUNCTYPE(hsa_status_t, ctypes.POINTER(hsa_ext_amd_aql_pm4_packet_t), ctypes.c_void_p)),
  ('hsa_ven_amd_aqlprofile_get_info', ctypes.CFUNCTYPE(hsa_status_t, ctypes.POINTER(hsa_ven_amd_aqlprofile_profile_t), hsa_ven_amd_aqlprofile_info_type_t, ctypes.c_void_p)),
  ('hsa_ven_amd_aqlprofile_iterate_data', ctypes.CFUNCTYPE(hsa_status_t, ctypes.POINTER(hsa_ven_amd_aqlprofile_profile_t), hsa_ven_amd_aqlprofile_data_callback_t, ctypes.c_void_p)),
  ('hsa_ven_amd_aqlprofile_iterate_event_ids', ctypes.CFUNCTYPE(hsa_status_t, hsa_ven_amd_aqlprofile_eventname_callback_t)),
  ('hsa_ven_amd_aqlprofile_iterate_event_coord', ctypes.CFUNCTYPE(hsa_status_t, hsa_agent_t, hsa_ven_amd_aqlprofile_event_t, uint32_t, hsa_ven_amd_aqlprofile_coordinate_callback_t, ctypes.c_void_p)),
  ('hsa_ven_amd_aqlprofile_att_marker', ctypes.CFUNCTYPE(hsa_status_t, ctypes.POINTER(hsa_ven_amd_aqlprofile_profile_t), ctypes.POINTER(hsa_ext_amd_aql_pm4_packet_t), uint32_t, hsa_ven_amd_aqlprofile_att_marker_channel_t)),
]
hsa_ven_amd_aqlprofile_1_00_pfn_t = struct_hsa_ven_amd_aqlprofile_1_00_pfn_s
hsa_ven_amd_aqlprofile_pfn_t = struct_hsa_ven_amd_aqlprofile_1_00_pfn_s
HSA_VERSION_1_0 = 1
HSA_AMD_INTERFACE_VERSION_MAJOR = 1
HSA_AMD_INTERFACE_VERSION_MINOR = 14
AMD_SIGNAL_ALIGN_BYTES = 64
AMD_QUEUE_ALIGN_BYTES = 64
MAX_NUM_XCC = 128
AMD_CONTROL_DIRECTIVES_ALIGN_BYTES = 64
AMD_ISA_ALIGN_BYTES = 256
AMD_KERNEL_CODE_ALIGN_BYTES = 64
HSA_AQLPROFILE_VERSION_MAJOR = 2
HSA_AQLPROFILE_VERSION_MINOR = 0
hsa_ven_amd_aqlprofile_VERSION_MAJOR = 1