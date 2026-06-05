# mypy: disable-error-code="empty-body"
from __future__ import annotations
import ctypes
from typing import Literal, TypeAlias
from tinygrad.runtime.support.c import _IO, _IOW, _IOR, _IOWR
from tinygrad.runtime.support import c
import os
dll = c.DLL('hsa', [os.getenv('ROCM_PATH', '/opt/rocm')+'/lib/libhsa-runtime64.so', 'hsa-runtime64'])
enum_SQ_RSRC_BUF_TYPE: dict[int, str] = {(SQ_RSRC_BUF:=0): 'SQ_RSRC_BUF', (SQ_RSRC_BUF_RSVD_1:=1): 'SQ_RSRC_BUF_RSVD_1', (SQ_RSRC_BUF_RSVD_2:=2): 'SQ_RSRC_BUF_RSVD_2', (SQ_RSRC_BUF_RSVD_3:=3): 'SQ_RSRC_BUF_RSVD_3'}
SQ_RSRC_BUF_TYPE: TypeAlias = ctypes.c_uint32
enum_BUF_DATA_FORMAT: dict[int, str] = {(BUF_DATA_FORMAT_INVALID:=0): 'BUF_DATA_FORMAT_INVALID', (BUF_DATA_FORMAT_8:=1): 'BUF_DATA_FORMAT_8', (BUF_DATA_FORMAT_16:=2): 'BUF_DATA_FORMAT_16', (BUF_DATA_FORMAT_8_8:=3): 'BUF_DATA_FORMAT_8_8', (BUF_DATA_FORMAT_32:=4): 'BUF_DATA_FORMAT_32', (BUF_DATA_FORMAT_16_16:=5): 'BUF_DATA_FORMAT_16_16', (BUF_DATA_FORMAT_10_11_11:=6): 'BUF_DATA_FORMAT_10_11_11', (BUF_DATA_FORMAT_11_11_10:=7): 'BUF_DATA_FORMAT_11_11_10', (BUF_DATA_FORMAT_10_10_10_2:=8): 'BUF_DATA_FORMAT_10_10_10_2', (BUF_DATA_FORMAT_2_10_10_10:=9): 'BUF_DATA_FORMAT_2_10_10_10', (BUF_DATA_FORMAT_8_8_8_8:=10): 'BUF_DATA_FORMAT_8_8_8_8', (BUF_DATA_FORMAT_32_32:=11): 'BUF_DATA_FORMAT_32_32', (BUF_DATA_FORMAT_16_16_16_16:=12): 'BUF_DATA_FORMAT_16_16_16_16', (BUF_DATA_FORMAT_32_32_32:=13): 'BUF_DATA_FORMAT_32_32_32', (BUF_DATA_FORMAT_32_32_32_32:=14): 'BUF_DATA_FORMAT_32_32_32_32', (BUF_DATA_FORMAT_RESERVED_15:=15): 'BUF_DATA_FORMAT_RESERVED_15'}
BUF_DATA_FORMAT: TypeAlias = ctypes.c_uint32
enum_BUF_NUM_FORMAT: dict[int, str] = {(BUF_NUM_FORMAT_UNORM:=0): 'BUF_NUM_FORMAT_UNORM', (BUF_NUM_FORMAT_SNORM:=1): 'BUF_NUM_FORMAT_SNORM', (BUF_NUM_FORMAT_USCALED:=2): 'BUF_NUM_FORMAT_USCALED', (BUF_NUM_FORMAT_SSCALED:=3): 'BUF_NUM_FORMAT_SSCALED', (BUF_NUM_FORMAT_UINT:=4): 'BUF_NUM_FORMAT_UINT', (BUF_NUM_FORMAT_SINT:=5): 'BUF_NUM_FORMAT_SINT', (BUF_NUM_FORMAT_SNORM_OGL__SI__CI:=6): 'BUF_NUM_FORMAT_SNORM_OGL__SI__CI', (BUF_NUM_FORMAT_RESERVED_6__VI:=6): 'BUF_NUM_FORMAT_RESERVED_6__VI', (BUF_NUM_FORMAT_FLOAT:=7): 'BUF_NUM_FORMAT_FLOAT'}
BUF_NUM_FORMAT: TypeAlias = ctypes.c_uint32
enum_BUF_FORMAT: dict[int, str] = {(BUF_FORMAT_32_UINT:=20): 'BUF_FORMAT_32_UINT'}
BUF_FORMAT: TypeAlias = ctypes.c_uint32
enum_SQ_SEL_XYZW01: dict[int, str] = {(SQ_SEL_0:=0): 'SQ_SEL_0', (SQ_SEL_1:=1): 'SQ_SEL_1', (SQ_SEL_RESERVED_0:=2): 'SQ_SEL_RESERVED_0', (SQ_SEL_RESERVED_1:=3): 'SQ_SEL_RESERVED_1', (SQ_SEL_X:=4): 'SQ_SEL_X', (SQ_SEL_Y:=5): 'SQ_SEL_Y', (SQ_SEL_Z:=6): 'SQ_SEL_Z', (SQ_SEL_W:=7): 'SQ_SEL_W'}
SQ_SEL_XYZW01: TypeAlias = ctypes.c_uint32
@c.record
class union_COMPUTE_TMPRING_SIZE(c.Struct):
  SIZE = 4
  bitfields: union_COMPUTE_TMPRING_SIZE_bitfields
  bits: union_COMPUTE_TMPRING_SIZE_bitfields
  u32All: int
  i32All: int
  f32All: float
@c.record
class union_COMPUTE_TMPRING_SIZE_bitfields(c.Struct):
  SIZE = 4
  WAVES: int
  WAVESIZE: int
union_COMPUTE_TMPRING_SIZE_bitfields.register_fields([('WAVES', ctypes.c_uint32, 0, 12, 0), ('WAVESIZE', ctypes.c_uint32, 1, 13, 4)])
union_COMPUTE_TMPRING_SIZE.register_fields([('bitfields', union_COMPUTE_TMPRING_SIZE_bitfields, 0), ('bits', union_COMPUTE_TMPRING_SIZE_bitfields, 0), ('u32All', ctypes.c_uint32, 0), ('i32All', ctypes.c_int32, 0), ('f32All', ctypes.c_float, 0)])
@c.record
class union_COMPUTE_TMPRING_SIZE_GFX11(c.Struct):
  SIZE = 4
  bitfields: union_COMPUTE_TMPRING_SIZE_GFX11_bitfields
  bits: union_COMPUTE_TMPRING_SIZE_GFX11_bitfields
  u32All: int
  i32All: int
  f32All: float
@c.record
class union_COMPUTE_TMPRING_SIZE_GFX11_bitfields(c.Struct):
  SIZE = 4
  WAVES: int
  WAVESIZE: int
union_COMPUTE_TMPRING_SIZE_GFX11_bitfields.register_fields([('WAVES', ctypes.c_uint32, 0, 12, 0), ('WAVESIZE', ctypes.c_uint32, 1, 15, 4)])
union_COMPUTE_TMPRING_SIZE_GFX11.register_fields([('bitfields', union_COMPUTE_TMPRING_SIZE_GFX11_bitfields, 0), ('bits', union_COMPUTE_TMPRING_SIZE_GFX11_bitfields, 0), ('u32All', ctypes.c_uint32, 0), ('i32All', ctypes.c_int32, 0), ('f32All', ctypes.c_float, 0)])
@c.record
class union_COMPUTE_TMPRING_SIZE_GFX12(c.Struct):
  SIZE = 4
  bitfields: union_COMPUTE_TMPRING_SIZE_GFX12_bitfields
  bits: union_COMPUTE_TMPRING_SIZE_GFX12_bitfields
  u32All: int
  i32All: int
  f32All: float
@c.record
class union_COMPUTE_TMPRING_SIZE_GFX12_bitfields(c.Struct):
  SIZE = 4
  WAVES: int
  WAVESIZE: int
union_COMPUTE_TMPRING_SIZE_GFX12_bitfields.register_fields([('WAVES', ctypes.c_uint32, 0, 12, 0), ('WAVESIZE', ctypes.c_uint32, 1, 18, 4)])
union_COMPUTE_TMPRING_SIZE_GFX12.register_fields([('bitfields', union_COMPUTE_TMPRING_SIZE_GFX12_bitfields, 0), ('bits', union_COMPUTE_TMPRING_SIZE_GFX12_bitfields, 0), ('u32All', ctypes.c_uint32, 0), ('i32All', ctypes.c_int32, 0), ('f32All', ctypes.c_float, 0)])
@c.record
class union_SQ_BUF_RSRC_WORD0(c.Struct):
  SIZE = 4
  bitfields: union_SQ_BUF_RSRC_WORD0_bitfields
  bits: union_SQ_BUF_RSRC_WORD0_bitfields
  u32All: int
  i32All: int
  f32All: float
@c.record
class union_SQ_BUF_RSRC_WORD0_bitfields(c.Struct):
  SIZE = 4
  BASE_ADDRESS: int
union_SQ_BUF_RSRC_WORD0_bitfields.register_fields([('BASE_ADDRESS', ctypes.c_uint32, 0, 32, 0)])
union_SQ_BUF_RSRC_WORD0.register_fields([('bitfields', union_SQ_BUF_RSRC_WORD0_bitfields, 0), ('bits', union_SQ_BUF_RSRC_WORD0_bitfields, 0), ('u32All', ctypes.c_uint32, 0), ('i32All', ctypes.c_int32, 0), ('f32All', ctypes.c_float, 0)])
@c.record
class union_SQ_BUF_RSRC_WORD1(c.Struct):
  SIZE = 4
  bitfields: union_SQ_BUF_RSRC_WORD1_bitfields
  bits: union_SQ_BUF_RSRC_WORD1_bitfields
  u32All: int
  i32All: int
  f32All: float
@c.record
class union_SQ_BUF_RSRC_WORD1_bitfields(c.Struct):
  SIZE = 4
  BASE_ADDRESS_HI: int
  STRIDE: int
  CACHE_SWIZZLE: int
  SWIZZLE_ENABLE: int
union_SQ_BUF_RSRC_WORD1_bitfields.register_fields([('BASE_ADDRESS_HI', ctypes.c_uint32, 0, 16, 0), ('STRIDE', ctypes.c_uint32, 2, 14, 0), ('CACHE_SWIZZLE', ctypes.c_uint32, 3, 1, 6), ('SWIZZLE_ENABLE', ctypes.c_uint32, 3, 1, 7)])
union_SQ_BUF_RSRC_WORD1.register_fields([('bitfields', union_SQ_BUF_RSRC_WORD1_bitfields, 0), ('bits', union_SQ_BUF_RSRC_WORD1_bitfields, 0), ('u32All', ctypes.c_uint32, 0), ('i32All', ctypes.c_int32, 0), ('f32All', ctypes.c_float, 0)])
@c.record
class union_SQ_BUF_RSRC_WORD1_GFX11(c.Struct):
  SIZE = 4
  bitfields: union_SQ_BUF_RSRC_WORD1_GFX11_bitfields
  bits: union_SQ_BUF_RSRC_WORD1_GFX11_bitfields
  u32All: int
  i32All: int
  f32All: float
@c.record
class union_SQ_BUF_RSRC_WORD1_GFX11_bitfields(c.Struct):
  SIZE = 4
  BASE_ADDRESS_HI: int
  STRIDE: int
  SWIZZLE_ENABLE: int
union_SQ_BUF_RSRC_WORD1_GFX11_bitfields.register_fields([('BASE_ADDRESS_HI', ctypes.c_uint32, 0, 16, 0), ('STRIDE', ctypes.c_uint32, 2, 14, 0), ('SWIZZLE_ENABLE', ctypes.c_uint32, 3, 2, 6)])
union_SQ_BUF_RSRC_WORD1_GFX11.register_fields([('bitfields', union_SQ_BUF_RSRC_WORD1_GFX11_bitfields, 0), ('bits', union_SQ_BUF_RSRC_WORD1_GFX11_bitfields, 0), ('u32All', ctypes.c_uint32, 0), ('i32All', ctypes.c_int32, 0), ('f32All', ctypes.c_float, 0)])
@c.record
class union_SQ_BUF_RSRC_WORD2(c.Struct):
  SIZE = 4
  bitfields: union_SQ_BUF_RSRC_WORD2_bitfields
  bits: union_SQ_BUF_RSRC_WORD2_bitfields
  u32All: int
  i32All: int
  f32All: float
@c.record
class union_SQ_BUF_RSRC_WORD2_bitfields(c.Struct):
  SIZE = 4
  NUM_RECORDS: int
union_SQ_BUF_RSRC_WORD2_bitfields.register_fields([('NUM_RECORDS', ctypes.c_uint32, 0, 32, 0)])
union_SQ_BUF_RSRC_WORD2.register_fields([('bitfields', union_SQ_BUF_RSRC_WORD2_bitfields, 0), ('bits', union_SQ_BUF_RSRC_WORD2_bitfields, 0), ('u32All', ctypes.c_uint32, 0), ('i32All', ctypes.c_int32, 0), ('f32All', ctypes.c_float, 0)])
@c.record
class union_SQ_BUF_RSRC_WORD3(c.Struct):
  SIZE = 4
  bitfields: union_SQ_BUF_RSRC_WORD3_bitfields
  bits: union_SQ_BUF_RSRC_WORD3_bitfields
  u32All: int
  i32All: int
  f32All: float
@c.record
class union_SQ_BUF_RSRC_WORD3_bitfields(c.Struct):
  SIZE = 4
  DST_SEL_X: int
  DST_SEL_Y: int
  DST_SEL_Z: int
  DST_SEL_W: int
  NUM_FORMAT: int
  DATA_FORMAT: int
  ELEMENT_SIZE: int
  INDEX_STRIDE: int
  ADD_TID_ENABLE: int
  ATC__CI__VI: int
  HASH_ENABLE: int
  HEAP: int
  MTYPE__CI__VI: int
  TYPE: int
union_SQ_BUF_RSRC_WORD3_bitfields.register_fields([('DST_SEL_X', ctypes.c_uint32, 0, 3, 0), ('DST_SEL_Y', ctypes.c_uint32, 0, 3, 3), ('DST_SEL_Z', ctypes.c_uint32, 0, 3, 6), ('DST_SEL_W', ctypes.c_uint32, 1, 3, 1), ('NUM_FORMAT', ctypes.c_uint32, 1, 3, 4), ('DATA_FORMAT', ctypes.c_uint32, 1, 4, 7), ('ELEMENT_SIZE', ctypes.c_uint32, 2, 2, 3), ('INDEX_STRIDE', ctypes.c_uint32, 2, 2, 5), ('ADD_TID_ENABLE', ctypes.c_uint32, 2, 1, 7), ('ATC__CI__VI', ctypes.c_uint32, 3, 1, 0), ('HASH_ENABLE', ctypes.c_uint32, 3, 1, 1), ('HEAP', ctypes.c_uint32, 3, 1, 2), ('MTYPE__CI__VI', ctypes.c_uint32, 3, 3, 3), ('TYPE', ctypes.c_uint32, 3, 2, 6)])
union_SQ_BUF_RSRC_WORD3.register_fields([('bitfields', union_SQ_BUF_RSRC_WORD3_bitfields, 0), ('bits', union_SQ_BUF_RSRC_WORD3_bitfields, 0), ('u32All', ctypes.c_uint32, 0), ('i32All', ctypes.c_int32, 0), ('f32All', ctypes.c_float, 0)])
@c.record
class union_SQ_BUF_RSRC_WORD3_GFX10(c.Struct):
  SIZE = 4
  bitfields: union_SQ_BUF_RSRC_WORD3_GFX10_bitfields
  bits: union_SQ_BUF_RSRC_WORD3_GFX10_bitfields
  u32All: int
  i32All: int
  f32All: float
@c.record
class union_SQ_BUF_RSRC_WORD3_GFX10_bitfields(c.Struct):
  SIZE = 4
  DST_SEL_X: int
  DST_SEL_Y: int
  DST_SEL_Z: int
  DST_SEL_W: int
  FORMAT: int
  RESERVED1: int
  INDEX_STRIDE: int
  ADD_TID_ENABLE: int
  RESOURCE_LEVEL: int
  RESERVED2: int
  OOB_SELECT: int
  TYPE: int
union_SQ_BUF_RSRC_WORD3_GFX10_bitfields.register_fields([('DST_SEL_X', ctypes.c_uint32, 0, 3, 0), ('DST_SEL_Y', ctypes.c_uint32, 0, 3, 3), ('DST_SEL_Z', ctypes.c_uint32, 0, 3, 6), ('DST_SEL_W', ctypes.c_uint32, 1, 3, 1), ('FORMAT', ctypes.c_uint32, 1, 7, 4), ('RESERVED1', ctypes.c_uint32, 2, 2, 3), ('INDEX_STRIDE', ctypes.c_uint32, 2, 2, 5), ('ADD_TID_ENABLE', ctypes.c_uint32, 2, 1, 7), ('RESOURCE_LEVEL', ctypes.c_uint32, 3, 1, 0), ('RESERVED2', ctypes.c_uint32, 3, 3, 1), ('OOB_SELECT', ctypes.c_uint32, 3, 2, 4), ('TYPE', ctypes.c_uint32, 3, 2, 6)])
union_SQ_BUF_RSRC_WORD3_GFX10.register_fields([('bitfields', union_SQ_BUF_RSRC_WORD3_GFX10_bitfields, 0), ('bits', union_SQ_BUF_RSRC_WORD3_GFX10_bitfields, 0), ('u32All', ctypes.c_uint32, 0), ('i32All', ctypes.c_int32, 0), ('f32All', ctypes.c_float, 0)])
@c.record
class union_SQ_BUF_RSRC_WORD3_GFX11(c.Struct):
  SIZE = 4
  bitfields: union_SQ_BUF_RSRC_WORD3_GFX11_bitfields
  bits: union_SQ_BUF_RSRC_WORD3_GFX11_bitfields
  u32All: int
  i32All: int
  f32All: float
@c.record
class union_SQ_BUF_RSRC_WORD3_GFX11_bitfields(c.Struct):
  SIZE = 4
  DST_SEL_X: int
  DST_SEL_Y: int
  DST_SEL_Z: int
  DST_SEL_W: int
  FORMAT: int
  RESERVED1: int
  INDEX_STRIDE: int
  ADD_TID_ENABLE: int
  RESERVED2: int
  OOB_SELECT: int
  TYPE: int
union_SQ_BUF_RSRC_WORD3_GFX11_bitfields.register_fields([('DST_SEL_X', ctypes.c_uint32, 0, 3, 0), ('DST_SEL_Y', ctypes.c_uint32, 0, 3, 3), ('DST_SEL_Z', ctypes.c_uint32, 0, 3, 6), ('DST_SEL_W', ctypes.c_uint32, 1, 3, 1), ('FORMAT', ctypes.c_uint32, 1, 6, 4), ('RESERVED1', ctypes.c_uint32, 2, 3, 2), ('INDEX_STRIDE', ctypes.c_uint32, 2, 2, 5), ('ADD_TID_ENABLE', ctypes.c_uint32, 2, 1, 7), ('RESERVED2', ctypes.c_uint32, 3, 4, 0), ('OOB_SELECT', ctypes.c_uint32, 3, 2, 4), ('TYPE', ctypes.c_uint32, 3, 2, 6)])
union_SQ_BUF_RSRC_WORD3_GFX11.register_fields([('bitfields', union_SQ_BUF_RSRC_WORD3_GFX11_bitfields, 0), ('bits', union_SQ_BUF_RSRC_WORD3_GFX11_bitfields, 0), ('u32All', ctypes.c_uint32, 0), ('i32All', ctypes.c_int32, 0), ('f32All', ctypes.c_float, 0)])
@c.record
class union_SQ_BUF_RSRC_WORD3_GFX12(c.Struct):
  SIZE = 4
  bitfields: union_SQ_BUF_RSRC_WORD3_GFX12_bitfields
  bits: union_SQ_BUF_RSRC_WORD3_GFX12_bitfields
  u32All: int
  i32All: int
  f32All: float
@c.record
class union_SQ_BUF_RSRC_WORD3_GFX12_bitfields(c.Struct):
  SIZE = 4
  DST_SEL_X: int
  DST_SEL_Y: int
  DST_SEL_Z: int
  DST_SEL_W: int
  FORMAT: int
  RESERVED1: int
  INDEX_STRIDE: int
  ADD_TID_ENABLE: int
  WRITE_COMPRESS_ENABLE: int
  COMPRESSION_EN: int
  COMPRESSION_ACCESS_MODE: int
  OOB_SELECT: int
  TYPE: int
union_SQ_BUF_RSRC_WORD3_GFX12_bitfields.register_fields([('DST_SEL_X', ctypes.c_uint32, 0, 3, 0), ('DST_SEL_Y', ctypes.c_uint32, 0, 3, 3), ('DST_SEL_Z', ctypes.c_uint32, 0, 3, 6), ('DST_SEL_W', ctypes.c_uint32, 1, 3, 1), ('FORMAT', ctypes.c_uint32, 1, 6, 4), ('RESERVED1', ctypes.c_uint32, 2, 3, 2), ('INDEX_STRIDE', ctypes.c_uint32, 2, 2, 5), ('ADD_TID_ENABLE', ctypes.c_uint32, 2, 1, 7), ('WRITE_COMPRESS_ENABLE', ctypes.c_uint32, 3, 1, 0), ('COMPRESSION_EN', ctypes.c_uint32, 3, 1, 1), ('COMPRESSION_ACCESS_MODE', ctypes.c_uint32, 3, 2, 2), ('OOB_SELECT', ctypes.c_uint32, 3, 2, 4), ('TYPE', ctypes.c_uint32, 3, 2, 6)])
union_SQ_BUF_RSRC_WORD3_GFX12.register_fields([('bitfields', union_SQ_BUF_RSRC_WORD3_GFX12_bitfields, 0), ('bits', union_SQ_BUF_RSRC_WORD3_GFX12_bitfields, 0), ('u32All', ctypes.c_uint32, 0), ('i32All', ctypes.c_int32, 0), ('f32All', ctypes.c_float, 0)])
hsa_status_t: dict[int, str] = {(HSA_STATUS_SUCCESS:=0): 'HSA_STATUS_SUCCESS', (HSA_STATUS_INFO_BREAK:=1): 'HSA_STATUS_INFO_BREAK', (HSA_STATUS_ERROR:=4096): 'HSA_STATUS_ERROR', (HSA_STATUS_ERROR_INVALID_ARGUMENT:=4097): 'HSA_STATUS_ERROR_INVALID_ARGUMENT', (HSA_STATUS_ERROR_INVALID_QUEUE_CREATION:=4098): 'HSA_STATUS_ERROR_INVALID_QUEUE_CREATION', (HSA_STATUS_ERROR_INVALID_ALLOCATION:=4099): 'HSA_STATUS_ERROR_INVALID_ALLOCATION', (HSA_STATUS_ERROR_INVALID_AGENT:=4100): 'HSA_STATUS_ERROR_INVALID_AGENT', (HSA_STATUS_ERROR_INVALID_REGION:=4101): 'HSA_STATUS_ERROR_INVALID_REGION', (HSA_STATUS_ERROR_INVALID_SIGNAL:=4102): 'HSA_STATUS_ERROR_INVALID_SIGNAL', (HSA_STATUS_ERROR_INVALID_QUEUE:=4103): 'HSA_STATUS_ERROR_INVALID_QUEUE', (HSA_STATUS_ERROR_OUT_OF_RESOURCES:=4104): 'HSA_STATUS_ERROR_OUT_OF_RESOURCES', (HSA_STATUS_ERROR_INVALID_PACKET_FORMAT:=4105): 'HSA_STATUS_ERROR_INVALID_PACKET_FORMAT', (HSA_STATUS_ERROR_RESOURCE_FREE:=4106): 'HSA_STATUS_ERROR_RESOURCE_FREE', (HSA_STATUS_ERROR_NOT_INITIALIZED:=4107): 'HSA_STATUS_ERROR_NOT_INITIALIZED', (HSA_STATUS_ERROR_REFCOUNT_OVERFLOW:=4108): 'HSA_STATUS_ERROR_REFCOUNT_OVERFLOW', (HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS:=4109): 'HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS', (HSA_STATUS_ERROR_INVALID_INDEX:=4110): 'HSA_STATUS_ERROR_INVALID_INDEX', (HSA_STATUS_ERROR_INVALID_ISA:=4111): 'HSA_STATUS_ERROR_INVALID_ISA', (HSA_STATUS_ERROR_INVALID_ISA_NAME:=4119): 'HSA_STATUS_ERROR_INVALID_ISA_NAME', (HSA_STATUS_ERROR_INVALID_CODE_OBJECT:=4112): 'HSA_STATUS_ERROR_INVALID_CODE_OBJECT', (HSA_STATUS_ERROR_INVALID_EXECUTABLE:=4113): 'HSA_STATUS_ERROR_INVALID_EXECUTABLE', (HSA_STATUS_ERROR_FROZEN_EXECUTABLE:=4114): 'HSA_STATUS_ERROR_FROZEN_EXECUTABLE', (HSA_STATUS_ERROR_INVALID_SYMBOL_NAME:=4115): 'HSA_STATUS_ERROR_INVALID_SYMBOL_NAME', (HSA_STATUS_ERROR_VARIABLE_ALREADY_DEFINED:=4116): 'HSA_STATUS_ERROR_VARIABLE_ALREADY_DEFINED', (HSA_STATUS_ERROR_VARIABLE_UNDEFINED:=4117): 'HSA_STATUS_ERROR_VARIABLE_UNDEFINED', (HSA_STATUS_ERROR_EXCEPTION:=4118): 'HSA_STATUS_ERROR_EXCEPTION', (HSA_STATUS_ERROR_INVALID_CODE_SYMBOL:=4120): 'HSA_STATUS_ERROR_INVALID_CODE_SYMBOL', (HSA_STATUS_ERROR_INVALID_EXECUTABLE_SYMBOL:=4121): 'HSA_STATUS_ERROR_INVALID_EXECUTABLE_SYMBOL', (HSA_STATUS_ERROR_INVALID_FILE:=4128): 'HSA_STATUS_ERROR_INVALID_FILE', (HSA_STATUS_ERROR_INVALID_CODE_OBJECT_READER:=4129): 'HSA_STATUS_ERROR_INVALID_CODE_OBJECT_READER', (HSA_STATUS_ERROR_INVALID_CACHE:=4130): 'HSA_STATUS_ERROR_INVALID_CACHE', (HSA_STATUS_ERROR_INVALID_WAVEFRONT:=4131): 'HSA_STATUS_ERROR_INVALID_WAVEFRONT', (HSA_STATUS_ERROR_INVALID_SIGNAL_GROUP:=4132): 'HSA_STATUS_ERROR_INVALID_SIGNAL_GROUP', (HSA_STATUS_ERROR_INVALID_RUNTIME_STATE:=4133): 'HSA_STATUS_ERROR_INVALID_RUNTIME_STATE', (HSA_STATUS_ERROR_FATAL:=4134): 'HSA_STATUS_ERROR_FATAL'}
@dll.bind(ctypes.c_uint32, ctypes.c_uint32, c.POINTER[c.POINTER[ctypes.c_char]])
def hsa_status_string(status:ctypes.c_uint32, status_string:c.POINTER[c.POINTER[ctypes.c_char]]) -> ctypes.c_uint32: ...
@c.record
class struct_hsa_dim3_s(c.Struct):
  SIZE = 12
  x: int
  y: int
  z: int
uint32_t: TypeAlias = ctypes.c_uint32
struct_hsa_dim3_s.register_fields([('x', uint32_t, 0), ('y', uint32_t, 4), ('z', uint32_t, 8)])
hsa_dim3_t: TypeAlias = struct_hsa_dim3_s
hsa_access_permission_t: dict[int, str] = {(HSA_ACCESS_PERMISSION_NONE:=0): 'HSA_ACCESS_PERMISSION_NONE', (HSA_ACCESS_PERMISSION_RO:=1): 'HSA_ACCESS_PERMISSION_RO', (HSA_ACCESS_PERMISSION_WO:=2): 'HSA_ACCESS_PERMISSION_WO', (HSA_ACCESS_PERMISSION_RW:=3): 'HSA_ACCESS_PERMISSION_RW'}
hsa_file_t: TypeAlias = ctypes.c_int32
@dll.bind(ctypes.c_uint32)
def hsa_init() -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32)
def hsa_shut_down() -> ctypes.c_uint32: ...
hsa_endianness_t: dict[int, str] = {(HSA_ENDIANNESS_LITTLE:=0): 'HSA_ENDIANNESS_LITTLE', (HSA_ENDIANNESS_BIG:=1): 'HSA_ENDIANNESS_BIG'}
hsa_machine_model_t: dict[int, str] = {(HSA_MACHINE_MODEL_SMALL:=0): 'HSA_MACHINE_MODEL_SMALL', (HSA_MACHINE_MODEL_LARGE:=1): 'HSA_MACHINE_MODEL_LARGE'}
hsa_profile_t: dict[int, str] = {(HSA_PROFILE_BASE:=0): 'HSA_PROFILE_BASE', (HSA_PROFILE_FULL:=1): 'HSA_PROFILE_FULL'}
hsa_system_info_t: dict[int, str] = {(HSA_SYSTEM_INFO_VERSION_MAJOR:=0): 'HSA_SYSTEM_INFO_VERSION_MAJOR', (HSA_SYSTEM_INFO_VERSION_MINOR:=1): 'HSA_SYSTEM_INFO_VERSION_MINOR', (HSA_SYSTEM_INFO_TIMESTAMP:=2): 'HSA_SYSTEM_INFO_TIMESTAMP', (HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY:=3): 'HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY', (HSA_SYSTEM_INFO_SIGNAL_MAX_WAIT:=4): 'HSA_SYSTEM_INFO_SIGNAL_MAX_WAIT', (HSA_SYSTEM_INFO_ENDIANNESS:=5): 'HSA_SYSTEM_INFO_ENDIANNESS', (HSA_SYSTEM_INFO_MACHINE_MODEL:=6): 'HSA_SYSTEM_INFO_MACHINE_MODEL', (HSA_SYSTEM_INFO_EXTENSIONS:=7): 'HSA_SYSTEM_INFO_EXTENSIONS', (HSA_AMD_SYSTEM_INFO_BUILD_VERSION:=512): 'HSA_AMD_SYSTEM_INFO_BUILD_VERSION', (HSA_AMD_SYSTEM_INFO_SVM_SUPPORTED:=513): 'HSA_AMD_SYSTEM_INFO_SVM_SUPPORTED', (HSA_AMD_SYSTEM_INFO_SVM_ACCESSIBLE_BY_DEFAULT:=514): 'HSA_AMD_SYSTEM_INFO_SVM_ACCESSIBLE_BY_DEFAULT', (HSA_AMD_SYSTEM_INFO_MWAITX_ENABLED:=515): 'HSA_AMD_SYSTEM_INFO_MWAITX_ENABLED', (HSA_AMD_SYSTEM_INFO_DMABUF_SUPPORTED:=516): 'HSA_AMD_SYSTEM_INFO_DMABUF_SUPPORTED', (HSA_AMD_SYSTEM_INFO_VIRTUAL_MEM_API_SUPPORTED:=517): 'HSA_AMD_SYSTEM_INFO_VIRTUAL_MEM_API_SUPPORTED', (HSA_AMD_SYSTEM_INFO_XNACK_ENABLED:=518): 'HSA_AMD_SYSTEM_INFO_XNACK_ENABLED', (HSA_AMD_SYSTEM_INFO_EXT_VERSION_MAJOR:=519): 'HSA_AMD_SYSTEM_INFO_EXT_VERSION_MAJOR', (HSA_AMD_SYSTEM_INFO_EXT_VERSION_MINOR:=520): 'HSA_AMD_SYSTEM_INFO_EXT_VERSION_MINOR'}
@dll.bind(ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p)
def hsa_system_get_info(attribute:ctypes.c_uint32, value:ctypes.c_void_p) -> ctypes.c_uint32: ...
hsa_extension_t: dict[int, str] = {(HSA_EXTENSION_FINALIZER:=0): 'HSA_EXTENSION_FINALIZER', (HSA_EXTENSION_IMAGES:=1): 'HSA_EXTENSION_IMAGES', (HSA_EXTENSION_PERFORMANCE_COUNTERS:=2): 'HSA_EXTENSION_PERFORMANCE_COUNTERS', (HSA_EXTENSION_PROFILING_EVENTS:=3): 'HSA_EXTENSION_PROFILING_EVENTS', (HSA_EXTENSION_STD_LAST:=3): 'HSA_EXTENSION_STD_LAST', (HSA_AMD_FIRST_EXTENSION:=512): 'HSA_AMD_FIRST_EXTENSION', (HSA_EXTENSION_AMD_PROFILER:=512): 'HSA_EXTENSION_AMD_PROFILER', (HSA_EXTENSION_AMD_LOADER:=513): 'HSA_EXTENSION_AMD_LOADER', (HSA_EXTENSION_AMD_AQLPROFILE:=514): 'HSA_EXTENSION_AMD_AQLPROFILE', (HSA_EXTENSION_AMD_PC_SAMPLING:=515): 'HSA_EXTENSION_AMD_PC_SAMPLING', (HSA_AMD_LAST_EXTENSION:=515): 'HSA_AMD_LAST_EXTENSION'}
uint16_t: TypeAlias = ctypes.c_uint16
@dll.bind(ctypes.c_uint32, uint16_t, c.POINTER[c.POINTER[ctypes.c_char]])
def hsa_extension_get_name(extension:uint16_t, name:c.POINTER[c.POINTER[ctypes.c_char]]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, uint16_t, uint16_t, uint16_t, c.POINTER[ctypes.c_bool])
def hsa_system_extension_supported(extension:uint16_t, version_major:uint16_t, version_minor:uint16_t, result:c.POINTER[ctypes.c_bool]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, uint16_t, uint16_t, c.POINTER[uint16_t], c.POINTER[ctypes.c_bool])
def hsa_system_major_extension_supported(extension:uint16_t, version_major:uint16_t, version_minor:c.POINTER[uint16_t], result:c.POINTER[ctypes.c_bool]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, uint16_t, uint16_t, uint16_t, ctypes.c_void_p)
def hsa_system_get_extension_table(extension:uint16_t, version_major:uint16_t, version_minor:uint16_t, table:ctypes.c_void_p) -> ctypes.c_uint32: ...
size_t: TypeAlias = ctypes.c_uint64
@dll.bind(ctypes.c_uint32, uint16_t, uint16_t, size_t, ctypes.c_void_p)
def hsa_system_get_major_extension_table(extension:uint16_t, version_major:uint16_t, table_length:size_t, table:ctypes.c_void_p) -> ctypes.c_uint32: ...
@c.record
class struct_hsa_agent_s(c.Struct):
  SIZE = 8
  handle: int
uint64_t: TypeAlias = ctypes.c_uint64
struct_hsa_agent_s.register_fields([('handle', uint64_t, 0)])
hsa_agent_t: TypeAlias = struct_hsa_agent_s
hsa_agent_feature_t: dict[int, str] = {(HSA_AGENT_FEATURE_KERNEL_DISPATCH:=1): 'HSA_AGENT_FEATURE_KERNEL_DISPATCH', (HSA_AGENT_FEATURE_AGENT_DISPATCH:=2): 'HSA_AGENT_FEATURE_AGENT_DISPATCH'}
hsa_device_type_t: dict[int, str] = {(HSA_DEVICE_TYPE_CPU:=0): 'HSA_DEVICE_TYPE_CPU', (HSA_DEVICE_TYPE_GPU:=1): 'HSA_DEVICE_TYPE_GPU', (HSA_DEVICE_TYPE_DSP:=2): 'HSA_DEVICE_TYPE_DSP', (HSA_DEVICE_TYPE_AIE:=3): 'HSA_DEVICE_TYPE_AIE'}
hsa_default_float_rounding_mode_t: dict[int, str] = {(HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT:=0): 'HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT', (HSA_DEFAULT_FLOAT_ROUNDING_MODE_ZERO:=1): 'HSA_DEFAULT_FLOAT_ROUNDING_MODE_ZERO', (HSA_DEFAULT_FLOAT_ROUNDING_MODE_NEAR:=2): 'HSA_DEFAULT_FLOAT_ROUNDING_MODE_NEAR'}
hsa_agent_info_t: dict[int, str] = {(HSA_AGENT_INFO_NAME:=0): 'HSA_AGENT_INFO_NAME', (HSA_AGENT_INFO_VENDOR_NAME:=1): 'HSA_AGENT_INFO_VENDOR_NAME', (HSA_AGENT_INFO_FEATURE:=2): 'HSA_AGENT_INFO_FEATURE', (HSA_AGENT_INFO_MACHINE_MODEL:=3): 'HSA_AGENT_INFO_MACHINE_MODEL', (HSA_AGENT_INFO_PROFILE:=4): 'HSA_AGENT_INFO_PROFILE', (HSA_AGENT_INFO_DEFAULT_FLOAT_ROUNDING_MODE:=5): 'HSA_AGENT_INFO_DEFAULT_FLOAT_ROUNDING_MODE', (HSA_AGENT_INFO_BASE_PROFILE_DEFAULT_FLOAT_ROUNDING_MODES:=23): 'HSA_AGENT_INFO_BASE_PROFILE_DEFAULT_FLOAT_ROUNDING_MODES', (HSA_AGENT_INFO_FAST_F16_OPERATION:=24): 'HSA_AGENT_INFO_FAST_F16_OPERATION', (HSA_AGENT_INFO_WAVEFRONT_SIZE:=6): 'HSA_AGENT_INFO_WAVEFRONT_SIZE', (HSA_AGENT_INFO_WORKGROUP_MAX_DIM:=7): 'HSA_AGENT_INFO_WORKGROUP_MAX_DIM', (HSA_AGENT_INFO_WORKGROUP_MAX_SIZE:=8): 'HSA_AGENT_INFO_WORKGROUP_MAX_SIZE', (HSA_AGENT_INFO_GRID_MAX_DIM:=9): 'HSA_AGENT_INFO_GRID_MAX_DIM', (HSA_AGENT_INFO_GRID_MAX_SIZE:=10): 'HSA_AGENT_INFO_GRID_MAX_SIZE', (HSA_AGENT_INFO_FBARRIER_MAX_SIZE:=11): 'HSA_AGENT_INFO_FBARRIER_MAX_SIZE', (HSA_AGENT_INFO_QUEUES_MAX:=12): 'HSA_AGENT_INFO_QUEUES_MAX', (HSA_AGENT_INFO_QUEUE_MIN_SIZE:=13): 'HSA_AGENT_INFO_QUEUE_MIN_SIZE', (HSA_AGENT_INFO_QUEUE_MAX_SIZE:=14): 'HSA_AGENT_INFO_QUEUE_MAX_SIZE', (HSA_AGENT_INFO_QUEUE_TYPE:=15): 'HSA_AGENT_INFO_QUEUE_TYPE', (HSA_AGENT_INFO_NODE:=16): 'HSA_AGENT_INFO_NODE', (HSA_AGENT_INFO_DEVICE:=17): 'HSA_AGENT_INFO_DEVICE', (HSA_AGENT_INFO_CACHE_SIZE:=18): 'HSA_AGENT_INFO_CACHE_SIZE', (HSA_AGENT_INFO_ISA:=19): 'HSA_AGENT_INFO_ISA', (HSA_AGENT_INFO_EXTENSIONS:=20): 'HSA_AGENT_INFO_EXTENSIONS', (HSA_AGENT_INFO_VERSION_MAJOR:=21): 'HSA_AGENT_INFO_VERSION_MAJOR', (HSA_AGENT_INFO_VERSION_MINOR:=22): 'HSA_AGENT_INFO_VERSION_MINOR', (HSA_AGENT_INFO_LAST:=2147483647): 'HSA_AGENT_INFO_LAST'}
@dll.bind(ctypes.c_uint32, hsa_agent_t, ctypes.c_uint32, ctypes.c_void_p)
def hsa_agent_get_info(agent:hsa_agent_t, attribute:ctypes.c_uint32, value:ctypes.c_void_p) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.CFUNCTYPE[ctypes.c_uint32, [hsa_agent_t, ctypes.c_void_p]], ctypes.c_void_p)
def hsa_iterate_agents(callback:c.CFUNCTYPE[ctypes.c_uint32, [hsa_agent_t, ctypes.c_void_p]], data:ctypes.c_void_p) -> ctypes.c_uint32: ...
hsa_exception_policy_t: dict[int, str] = {(HSA_EXCEPTION_POLICY_BREAK:=1): 'HSA_EXCEPTION_POLICY_BREAK', (HSA_EXCEPTION_POLICY_DETECT:=2): 'HSA_EXCEPTION_POLICY_DETECT'}
@dll.bind(ctypes.c_uint32, hsa_agent_t, ctypes.c_uint32, c.POINTER[uint16_t])
def hsa_agent_get_exception_policies(agent:hsa_agent_t, profile:ctypes.c_uint32, mask:c.POINTER[uint16_t]) -> ctypes.c_uint32: ...
@c.record
class struct_hsa_cache_s(c.Struct):
  SIZE = 8
  handle: int
struct_hsa_cache_s.register_fields([('handle', uint64_t, 0)])
hsa_cache_t: TypeAlias = struct_hsa_cache_s
hsa_cache_info_t: dict[int, str] = {(HSA_CACHE_INFO_NAME_LENGTH:=0): 'HSA_CACHE_INFO_NAME_LENGTH', (HSA_CACHE_INFO_NAME:=1): 'HSA_CACHE_INFO_NAME', (HSA_CACHE_INFO_LEVEL:=2): 'HSA_CACHE_INFO_LEVEL', (HSA_CACHE_INFO_SIZE:=3): 'HSA_CACHE_INFO_SIZE'}
@dll.bind(ctypes.c_uint32, hsa_cache_t, ctypes.c_uint32, ctypes.c_void_p)
def hsa_cache_get_info(cache:hsa_cache_t, attribute:ctypes.c_uint32, value:ctypes.c_void_p) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hsa_agent_t, c.CFUNCTYPE[ctypes.c_uint32, [hsa_cache_t, ctypes.c_void_p]], ctypes.c_void_p)
def hsa_agent_iterate_caches(agent:hsa_agent_t, callback:c.CFUNCTYPE[ctypes.c_uint32, [hsa_cache_t, ctypes.c_void_p]], data:ctypes.c_void_p) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, uint16_t, hsa_agent_t, uint16_t, uint16_t, c.POINTER[ctypes.c_bool])
def hsa_agent_extension_supported(extension:uint16_t, agent:hsa_agent_t, version_major:uint16_t, version_minor:uint16_t, result:c.POINTER[ctypes.c_bool]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, uint16_t, hsa_agent_t, uint16_t, c.POINTER[uint16_t], c.POINTER[ctypes.c_bool])
def hsa_agent_major_extension_supported(extension:uint16_t, agent:hsa_agent_t, version_major:uint16_t, version_minor:c.POINTER[uint16_t], result:c.POINTER[ctypes.c_bool]) -> ctypes.c_uint32: ...
@c.record
class struct_hsa_signal_s(c.Struct):
  SIZE = 8
  handle: int
struct_hsa_signal_s.register_fields([('handle', uint64_t, 0)])
hsa_signal_t: TypeAlias = struct_hsa_signal_s
hsa_signal_value_t: TypeAlias = ctypes.c_int64
@dll.bind(ctypes.c_uint32, hsa_signal_value_t, uint32_t, c.POINTER[hsa_agent_t], c.POINTER[hsa_signal_t])
def hsa_signal_create(initial_value:hsa_signal_value_t, num_consumers:uint32_t, consumers:c.POINTER[hsa_agent_t], signal:c.POINTER[hsa_signal_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hsa_signal_t)
def hsa_signal_destroy(signal:hsa_signal_t) -> ctypes.c_uint32: ...
@dll.bind(hsa_signal_value_t, hsa_signal_t)
def hsa_signal_load_scacquire(signal:hsa_signal_t) -> hsa_signal_value_t: ...
@dll.bind(hsa_signal_value_t, hsa_signal_t)
def hsa_signal_load_relaxed(signal:hsa_signal_t) -> hsa_signal_value_t: ...
@dll.bind(hsa_signal_value_t, hsa_signal_t)
def hsa_signal_load_acquire(signal:hsa_signal_t) -> hsa_signal_value_t: ...
@dll.bind(None, hsa_signal_t, hsa_signal_value_t)
def hsa_signal_store_relaxed(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind(None, hsa_signal_t, hsa_signal_value_t)
def hsa_signal_store_screlease(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind(None, hsa_signal_t, hsa_signal_value_t)
def hsa_signal_store_release(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind(None, hsa_signal_t, hsa_signal_value_t)
def hsa_signal_silent_store_relaxed(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind(None, hsa_signal_t, hsa_signal_value_t)
def hsa_signal_silent_store_screlease(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind(hsa_signal_value_t, hsa_signal_t, hsa_signal_value_t)
def hsa_signal_exchange_scacq_screl(signal:hsa_signal_t, value:hsa_signal_value_t) -> hsa_signal_value_t: ...
@dll.bind(hsa_signal_value_t, hsa_signal_t, hsa_signal_value_t)
def hsa_signal_exchange_acq_rel(signal:hsa_signal_t, value:hsa_signal_value_t) -> hsa_signal_value_t: ...
@dll.bind(hsa_signal_value_t, hsa_signal_t, hsa_signal_value_t)
def hsa_signal_exchange_scacquire(signal:hsa_signal_t, value:hsa_signal_value_t) -> hsa_signal_value_t: ...
@dll.bind(hsa_signal_value_t, hsa_signal_t, hsa_signal_value_t)
def hsa_signal_exchange_acquire(signal:hsa_signal_t, value:hsa_signal_value_t) -> hsa_signal_value_t: ...
@dll.bind(hsa_signal_value_t, hsa_signal_t, hsa_signal_value_t)
def hsa_signal_exchange_relaxed(signal:hsa_signal_t, value:hsa_signal_value_t) -> hsa_signal_value_t: ...
@dll.bind(hsa_signal_value_t, hsa_signal_t, hsa_signal_value_t)
def hsa_signal_exchange_screlease(signal:hsa_signal_t, value:hsa_signal_value_t) -> hsa_signal_value_t: ...
@dll.bind(hsa_signal_value_t, hsa_signal_t, hsa_signal_value_t)
def hsa_signal_exchange_release(signal:hsa_signal_t, value:hsa_signal_value_t) -> hsa_signal_value_t: ...
@dll.bind(hsa_signal_value_t, hsa_signal_t, hsa_signal_value_t, hsa_signal_value_t)
def hsa_signal_cas_scacq_screl(signal:hsa_signal_t, expected:hsa_signal_value_t, value:hsa_signal_value_t) -> hsa_signal_value_t: ...
@dll.bind(hsa_signal_value_t, hsa_signal_t, hsa_signal_value_t, hsa_signal_value_t)
def hsa_signal_cas_acq_rel(signal:hsa_signal_t, expected:hsa_signal_value_t, value:hsa_signal_value_t) -> hsa_signal_value_t: ...
@dll.bind(hsa_signal_value_t, hsa_signal_t, hsa_signal_value_t, hsa_signal_value_t)
def hsa_signal_cas_scacquire(signal:hsa_signal_t, expected:hsa_signal_value_t, value:hsa_signal_value_t) -> hsa_signal_value_t: ...
@dll.bind(hsa_signal_value_t, hsa_signal_t, hsa_signal_value_t, hsa_signal_value_t)
def hsa_signal_cas_acquire(signal:hsa_signal_t, expected:hsa_signal_value_t, value:hsa_signal_value_t) -> hsa_signal_value_t: ...
@dll.bind(hsa_signal_value_t, hsa_signal_t, hsa_signal_value_t, hsa_signal_value_t)
def hsa_signal_cas_relaxed(signal:hsa_signal_t, expected:hsa_signal_value_t, value:hsa_signal_value_t) -> hsa_signal_value_t: ...
@dll.bind(hsa_signal_value_t, hsa_signal_t, hsa_signal_value_t, hsa_signal_value_t)
def hsa_signal_cas_screlease(signal:hsa_signal_t, expected:hsa_signal_value_t, value:hsa_signal_value_t) -> hsa_signal_value_t: ...
@dll.bind(hsa_signal_value_t, hsa_signal_t, hsa_signal_value_t, hsa_signal_value_t)
def hsa_signal_cas_release(signal:hsa_signal_t, expected:hsa_signal_value_t, value:hsa_signal_value_t) -> hsa_signal_value_t: ...
@dll.bind(None, hsa_signal_t, hsa_signal_value_t)
def hsa_signal_add_scacq_screl(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind(None, hsa_signal_t, hsa_signal_value_t)
def hsa_signal_add_acq_rel(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind(None, hsa_signal_t, hsa_signal_value_t)
def hsa_signal_add_scacquire(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind(None, hsa_signal_t, hsa_signal_value_t)
def hsa_signal_add_acquire(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind(None, hsa_signal_t, hsa_signal_value_t)
def hsa_signal_add_relaxed(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind(None, hsa_signal_t, hsa_signal_value_t)
def hsa_signal_add_screlease(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind(None, hsa_signal_t, hsa_signal_value_t)
def hsa_signal_add_release(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind(None, hsa_signal_t, hsa_signal_value_t)
def hsa_signal_subtract_scacq_screl(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind(None, hsa_signal_t, hsa_signal_value_t)
def hsa_signal_subtract_acq_rel(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind(None, hsa_signal_t, hsa_signal_value_t)
def hsa_signal_subtract_scacquire(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind(None, hsa_signal_t, hsa_signal_value_t)
def hsa_signal_subtract_acquire(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind(None, hsa_signal_t, hsa_signal_value_t)
def hsa_signal_subtract_relaxed(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind(None, hsa_signal_t, hsa_signal_value_t)
def hsa_signal_subtract_screlease(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind(None, hsa_signal_t, hsa_signal_value_t)
def hsa_signal_subtract_release(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind(None, hsa_signal_t, hsa_signal_value_t)
def hsa_signal_and_scacq_screl(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind(None, hsa_signal_t, hsa_signal_value_t)
def hsa_signal_and_acq_rel(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind(None, hsa_signal_t, hsa_signal_value_t)
def hsa_signal_and_scacquire(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind(None, hsa_signal_t, hsa_signal_value_t)
def hsa_signal_and_acquire(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind(None, hsa_signal_t, hsa_signal_value_t)
def hsa_signal_and_relaxed(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind(None, hsa_signal_t, hsa_signal_value_t)
def hsa_signal_and_screlease(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind(None, hsa_signal_t, hsa_signal_value_t)
def hsa_signal_and_release(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind(None, hsa_signal_t, hsa_signal_value_t)
def hsa_signal_or_scacq_screl(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind(None, hsa_signal_t, hsa_signal_value_t)
def hsa_signal_or_acq_rel(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind(None, hsa_signal_t, hsa_signal_value_t)
def hsa_signal_or_scacquire(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind(None, hsa_signal_t, hsa_signal_value_t)
def hsa_signal_or_acquire(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind(None, hsa_signal_t, hsa_signal_value_t)
def hsa_signal_or_relaxed(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind(None, hsa_signal_t, hsa_signal_value_t)
def hsa_signal_or_screlease(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind(None, hsa_signal_t, hsa_signal_value_t)
def hsa_signal_or_release(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind(None, hsa_signal_t, hsa_signal_value_t)
def hsa_signal_xor_scacq_screl(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind(None, hsa_signal_t, hsa_signal_value_t)
def hsa_signal_xor_acq_rel(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind(None, hsa_signal_t, hsa_signal_value_t)
def hsa_signal_xor_scacquire(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind(None, hsa_signal_t, hsa_signal_value_t)
def hsa_signal_xor_acquire(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind(None, hsa_signal_t, hsa_signal_value_t)
def hsa_signal_xor_relaxed(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind(None, hsa_signal_t, hsa_signal_value_t)
def hsa_signal_xor_screlease(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
@dll.bind(None, hsa_signal_t, hsa_signal_value_t)
def hsa_signal_xor_release(signal:hsa_signal_t, value:hsa_signal_value_t) -> None: ...
hsa_signal_condition_t: dict[int, str] = {(HSA_SIGNAL_CONDITION_EQ:=0): 'HSA_SIGNAL_CONDITION_EQ', (HSA_SIGNAL_CONDITION_NE:=1): 'HSA_SIGNAL_CONDITION_NE', (HSA_SIGNAL_CONDITION_LT:=2): 'HSA_SIGNAL_CONDITION_LT', (HSA_SIGNAL_CONDITION_GTE:=3): 'HSA_SIGNAL_CONDITION_GTE'}
hsa_wait_state_t: dict[int, str] = {(HSA_WAIT_STATE_BLOCKED:=0): 'HSA_WAIT_STATE_BLOCKED', (HSA_WAIT_STATE_ACTIVE:=1): 'HSA_WAIT_STATE_ACTIVE'}
@dll.bind(hsa_signal_value_t, hsa_signal_t, ctypes.c_uint32, hsa_signal_value_t, uint64_t, ctypes.c_uint32)
def hsa_signal_wait_scacquire(signal:hsa_signal_t, condition:ctypes.c_uint32, compare_value:hsa_signal_value_t, timeout_hint:uint64_t, wait_state_hint:ctypes.c_uint32) -> hsa_signal_value_t: ...
@dll.bind(hsa_signal_value_t, hsa_signal_t, ctypes.c_uint32, hsa_signal_value_t, uint64_t, ctypes.c_uint32)
def hsa_signal_wait_relaxed(signal:hsa_signal_t, condition:ctypes.c_uint32, compare_value:hsa_signal_value_t, timeout_hint:uint64_t, wait_state_hint:ctypes.c_uint32) -> hsa_signal_value_t: ...
@dll.bind(hsa_signal_value_t, hsa_signal_t, ctypes.c_uint32, hsa_signal_value_t, uint64_t, ctypes.c_uint32)
def hsa_signal_wait_acquire(signal:hsa_signal_t, condition:ctypes.c_uint32, compare_value:hsa_signal_value_t, timeout_hint:uint64_t, wait_state_hint:ctypes.c_uint32) -> hsa_signal_value_t: ...
@c.record
class struct_hsa_signal_group_s(c.Struct):
  SIZE = 8
  handle: int
struct_hsa_signal_group_s.register_fields([('handle', uint64_t, 0)])
hsa_signal_group_t: TypeAlias = struct_hsa_signal_group_s
@dll.bind(ctypes.c_uint32, uint32_t, c.POINTER[hsa_signal_t], uint32_t, c.POINTER[hsa_agent_t], c.POINTER[hsa_signal_group_t])
def hsa_signal_group_create(num_signals:uint32_t, signals:c.POINTER[hsa_signal_t], num_consumers:uint32_t, consumers:c.POINTER[hsa_agent_t], signal_group:c.POINTER[hsa_signal_group_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hsa_signal_group_t)
def hsa_signal_group_destroy(signal_group:hsa_signal_group_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hsa_signal_group_t, c.POINTER[ctypes.c_uint32], c.POINTER[hsa_signal_value_t], ctypes.c_uint32, c.POINTER[hsa_signal_t], c.POINTER[hsa_signal_value_t])
def hsa_signal_group_wait_any_scacquire(signal_group:hsa_signal_group_t, conditions:c.POINTER[ctypes.c_uint32], compare_values:c.POINTER[hsa_signal_value_t], wait_state_hint:ctypes.c_uint32, signal:c.POINTER[hsa_signal_t], value:c.POINTER[hsa_signal_value_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hsa_signal_group_t, c.POINTER[ctypes.c_uint32], c.POINTER[hsa_signal_value_t], ctypes.c_uint32, c.POINTER[hsa_signal_t], c.POINTER[hsa_signal_value_t])
def hsa_signal_group_wait_any_relaxed(signal_group:hsa_signal_group_t, conditions:c.POINTER[ctypes.c_uint32], compare_values:c.POINTER[hsa_signal_value_t], wait_state_hint:ctypes.c_uint32, signal:c.POINTER[hsa_signal_t], value:c.POINTER[hsa_signal_value_t]) -> ctypes.c_uint32: ...
@c.record
class struct_hsa_region_s(c.Struct):
  SIZE = 8
  handle: int
struct_hsa_region_s.register_fields([('handle', uint64_t, 0)])
hsa_region_t: TypeAlias = struct_hsa_region_s
hsa_queue_type_t: dict[int, str] = {(HSA_QUEUE_TYPE_MULTI:=0): 'HSA_QUEUE_TYPE_MULTI', (HSA_QUEUE_TYPE_SINGLE:=1): 'HSA_QUEUE_TYPE_SINGLE', (HSA_QUEUE_TYPE_COOPERATIVE:=2): 'HSA_QUEUE_TYPE_COOPERATIVE'}
hsa_queue_type32_t: TypeAlias = ctypes.c_uint32
hsa_queue_feature_t: dict[int, str] = {(HSA_QUEUE_FEATURE_KERNEL_DISPATCH:=1): 'HSA_QUEUE_FEATURE_KERNEL_DISPATCH', (HSA_QUEUE_FEATURE_AGENT_DISPATCH:=2): 'HSA_QUEUE_FEATURE_AGENT_DISPATCH'}
@c.record
class struct_hsa_queue_s(c.Struct):
  SIZE = 40
  type: int
  features: int
  base_address: ctypes.c_void_p
  doorbell_signal: struct_hsa_signal_s
  size: int
  reserved1: int
  id: int
struct_hsa_queue_s.register_fields([('type', hsa_queue_type32_t, 0), ('features', uint32_t, 4), ('base_address', ctypes.c_void_p, 8), ('doorbell_signal', hsa_signal_t, 16), ('size', uint32_t, 24), ('reserved1', uint32_t, 28), ('id', uint64_t, 32)])
hsa_queue_t: TypeAlias = struct_hsa_queue_s
@dll.bind(ctypes.c_uint32, hsa_agent_t, uint32_t, hsa_queue_type32_t, c.CFUNCTYPE[None, [ctypes.c_uint32, c.POINTER[hsa_queue_t], ctypes.c_void_p]], ctypes.c_void_p, uint32_t, uint32_t, c.POINTER[c.POINTER[hsa_queue_t]])
def hsa_queue_create(agent:hsa_agent_t, size:uint32_t, type:hsa_queue_type32_t, callback:c.CFUNCTYPE[None, [ctypes.c_uint32, c.POINTER[hsa_queue_t], ctypes.c_void_p]], data:ctypes.c_void_p, private_segment_size:uint32_t, group_segment_size:uint32_t, queue:c.POINTER[c.POINTER[hsa_queue_t]]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hsa_region_t, uint32_t, hsa_queue_type32_t, uint32_t, hsa_signal_t, c.POINTER[c.POINTER[hsa_queue_t]])
def hsa_soft_queue_create(region:hsa_region_t, size:uint32_t, type:hsa_queue_type32_t, features:uint32_t, doorbell_signal:hsa_signal_t, queue:c.POINTER[c.POINTER[hsa_queue_t]]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hsa_queue_t])
def hsa_queue_destroy(queue:c.POINTER[hsa_queue_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hsa_queue_t])
def hsa_queue_inactivate(queue:c.POINTER[hsa_queue_t]) -> ctypes.c_uint32: ...
@dll.bind(uint64_t, c.POINTER[hsa_queue_t])
def hsa_queue_load_read_index_acquire(queue:c.POINTER[hsa_queue_t]) -> uint64_t: ...
@dll.bind(uint64_t, c.POINTER[hsa_queue_t])
def hsa_queue_load_read_index_scacquire(queue:c.POINTER[hsa_queue_t]) -> uint64_t: ...
@dll.bind(uint64_t, c.POINTER[hsa_queue_t])
def hsa_queue_load_read_index_relaxed(queue:c.POINTER[hsa_queue_t]) -> uint64_t: ...
@dll.bind(uint64_t, c.POINTER[hsa_queue_t])
def hsa_queue_load_write_index_acquire(queue:c.POINTER[hsa_queue_t]) -> uint64_t: ...
@dll.bind(uint64_t, c.POINTER[hsa_queue_t])
def hsa_queue_load_write_index_scacquire(queue:c.POINTER[hsa_queue_t]) -> uint64_t: ...
@dll.bind(uint64_t, c.POINTER[hsa_queue_t])
def hsa_queue_load_write_index_relaxed(queue:c.POINTER[hsa_queue_t]) -> uint64_t: ...
@dll.bind(None, c.POINTER[hsa_queue_t], uint64_t)
def hsa_queue_store_write_index_relaxed(queue:c.POINTER[hsa_queue_t], value:uint64_t) -> None: ...
@dll.bind(None, c.POINTER[hsa_queue_t], uint64_t)
def hsa_queue_store_write_index_release(queue:c.POINTER[hsa_queue_t], value:uint64_t) -> None: ...
@dll.bind(None, c.POINTER[hsa_queue_t], uint64_t)
def hsa_queue_store_write_index_screlease(queue:c.POINTER[hsa_queue_t], value:uint64_t) -> None: ...
@dll.bind(uint64_t, c.POINTER[hsa_queue_t], uint64_t, uint64_t)
def hsa_queue_cas_write_index_acq_rel(queue:c.POINTER[hsa_queue_t], expected:uint64_t, value:uint64_t) -> uint64_t: ...
@dll.bind(uint64_t, c.POINTER[hsa_queue_t], uint64_t, uint64_t)
def hsa_queue_cas_write_index_scacq_screl(queue:c.POINTER[hsa_queue_t], expected:uint64_t, value:uint64_t) -> uint64_t: ...
@dll.bind(uint64_t, c.POINTER[hsa_queue_t], uint64_t, uint64_t)
def hsa_queue_cas_write_index_acquire(queue:c.POINTER[hsa_queue_t], expected:uint64_t, value:uint64_t) -> uint64_t: ...
@dll.bind(uint64_t, c.POINTER[hsa_queue_t], uint64_t, uint64_t)
def hsa_queue_cas_write_index_scacquire(queue:c.POINTER[hsa_queue_t], expected:uint64_t, value:uint64_t) -> uint64_t: ...
@dll.bind(uint64_t, c.POINTER[hsa_queue_t], uint64_t, uint64_t)
def hsa_queue_cas_write_index_relaxed(queue:c.POINTER[hsa_queue_t], expected:uint64_t, value:uint64_t) -> uint64_t: ...
@dll.bind(uint64_t, c.POINTER[hsa_queue_t], uint64_t, uint64_t)
def hsa_queue_cas_write_index_release(queue:c.POINTER[hsa_queue_t], expected:uint64_t, value:uint64_t) -> uint64_t: ...
@dll.bind(uint64_t, c.POINTER[hsa_queue_t], uint64_t, uint64_t)
def hsa_queue_cas_write_index_screlease(queue:c.POINTER[hsa_queue_t], expected:uint64_t, value:uint64_t) -> uint64_t: ...
@dll.bind(uint64_t, c.POINTER[hsa_queue_t], uint64_t)
def hsa_queue_add_write_index_acq_rel(queue:c.POINTER[hsa_queue_t], value:uint64_t) -> uint64_t: ...
@dll.bind(uint64_t, c.POINTER[hsa_queue_t], uint64_t)
def hsa_queue_add_write_index_scacq_screl(queue:c.POINTER[hsa_queue_t], value:uint64_t) -> uint64_t: ...
@dll.bind(uint64_t, c.POINTER[hsa_queue_t], uint64_t)
def hsa_queue_add_write_index_acquire(queue:c.POINTER[hsa_queue_t], value:uint64_t) -> uint64_t: ...
@dll.bind(uint64_t, c.POINTER[hsa_queue_t], uint64_t)
def hsa_queue_add_write_index_scacquire(queue:c.POINTER[hsa_queue_t], value:uint64_t) -> uint64_t: ...
@dll.bind(uint64_t, c.POINTER[hsa_queue_t], uint64_t)
def hsa_queue_add_write_index_relaxed(queue:c.POINTER[hsa_queue_t], value:uint64_t) -> uint64_t: ...
@dll.bind(uint64_t, c.POINTER[hsa_queue_t], uint64_t)
def hsa_queue_add_write_index_release(queue:c.POINTER[hsa_queue_t], value:uint64_t) -> uint64_t: ...
@dll.bind(uint64_t, c.POINTER[hsa_queue_t], uint64_t)
def hsa_queue_add_write_index_screlease(queue:c.POINTER[hsa_queue_t], value:uint64_t) -> uint64_t: ...
@dll.bind(None, c.POINTER[hsa_queue_t], uint64_t)
def hsa_queue_store_read_index_relaxed(queue:c.POINTER[hsa_queue_t], value:uint64_t) -> None: ...
@dll.bind(None, c.POINTER[hsa_queue_t], uint64_t)
def hsa_queue_store_read_index_release(queue:c.POINTER[hsa_queue_t], value:uint64_t) -> None: ...
@dll.bind(None, c.POINTER[hsa_queue_t], uint64_t)
def hsa_queue_store_read_index_screlease(queue:c.POINTER[hsa_queue_t], value:uint64_t) -> None: ...
hsa_packet_type_t: dict[int, str] = {(HSA_PACKET_TYPE_VENDOR_SPECIFIC:=0): 'HSA_PACKET_TYPE_VENDOR_SPECIFIC', (HSA_PACKET_TYPE_INVALID:=1): 'HSA_PACKET_TYPE_INVALID', (HSA_PACKET_TYPE_KERNEL_DISPATCH:=2): 'HSA_PACKET_TYPE_KERNEL_DISPATCH', (HSA_PACKET_TYPE_BARRIER_AND:=3): 'HSA_PACKET_TYPE_BARRIER_AND', (HSA_PACKET_TYPE_AGENT_DISPATCH:=4): 'HSA_PACKET_TYPE_AGENT_DISPATCH', (HSA_PACKET_TYPE_BARRIER_OR:=5): 'HSA_PACKET_TYPE_BARRIER_OR'}
hsa_fence_scope_t: dict[int, str] = {(HSA_FENCE_SCOPE_NONE:=0): 'HSA_FENCE_SCOPE_NONE', (HSA_FENCE_SCOPE_AGENT:=1): 'HSA_FENCE_SCOPE_AGENT', (HSA_FENCE_SCOPE_SYSTEM:=2): 'HSA_FENCE_SCOPE_SYSTEM'}
hsa_packet_header_t: dict[int, str] = {(HSA_PACKET_HEADER_TYPE:=0): 'HSA_PACKET_HEADER_TYPE', (HSA_PACKET_HEADER_BARRIER:=8): 'HSA_PACKET_HEADER_BARRIER', (HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE:=9): 'HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE', (HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE:=9): 'HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE', (HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE:=11): 'HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE', (HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE:=11): 'HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE'}
hsa_packet_header_width_t: dict[int, str] = {(HSA_PACKET_HEADER_WIDTH_TYPE:=8): 'HSA_PACKET_HEADER_WIDTH_TYPE', (HSA_PACKET_HEADER_WIDTH_BARRIER:=1): 'HSA_PACKET_HEADER_WIDTH_BARRIER', (HSA_PACKET_HEADER_WIDTH_SCACQUIRE_FENCE_SCOPE:=2): 'HSA_PACKET_HEADER_WIDTH_SCACQUIRE_FENCE_SCOPE', (HSA_PACKET_HEADER_WIDTH_ACQUIRE_FENCE_SCOPE:=2): 'HSA_PACKET_HEADER_WIDTH_ACQUIRE_FENCE_SCOPE', (HSA_PACKET_HEADER_WIDTH_SCRELEASE_FENCE_SCOPE:=2): 'HSA_PACKET_HEADER_WIDTH_SCRELEASE_FENCE_SCOPE', (HSA_PACKET_HEADER_WIDTH_RELEASE_FENCE_SCOPE:=2): 'HSA_PACKET_HEADER_WIDTH_RELEASE_FENCE_SCOPE'}
hsa_kernel_dispatch_packet_setup_t: dict[int, str] = {(HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS:=0): 'HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS'}
hsa_kernel_dispatch_packet_setup_width_t: dict[int, str] = {(HSA_KERNEL_DISPATCH_PACKET_SETUP_WIDTH_DIMENSIONS:=2): 'HSA_KERNEL_DISPATCH_PACKET_SETUP_WIDTH_DIMENSIONS'}
@c.record
class struct_hsa_kernel_dispatch_packet_s(c.Struct):
  SIZE = 64
  header: int
  setup: int
  full_header: int
  workgroup_size_x: int
  workgroup_size_y: int
  workgroup_size_z: int
  reserved0: int
  grid_size_x: int
  grid_size_y: int
  grid_size_z: int
  private_segment_size: int
  group_segment_size: int
  kernel_object: int
  kernarg_address: ctypes.c_void_p
  reserved2: int
  completion_signal: struct_hsa_signal_s
struct_hsa_kernel_dispatch_packet_s.register_fields([('header', uint16_t, 0), ('setup', uint16_t, 2), ('full_header', uint32_t, 0), ('workgroup_size_x', uint16_t, 4), ('workgroup_size_y', uint16_t, 6), ('workgroup_size_z', uint16_t, 8), ('reserved0', uint16_t, 10), ('grid_size_x', uint32_t, 12), ('grid_size_y', uint32_t, 16), ('grid_size_z', uint32_t, 20), ('private_segment_size', uint32_t, 24), ('group_segment_size', uint32_t, 28), ('kernel_object', uint64_t, 32), ('kernarg_address', ctypes.c_void_p, 40), ('reserved2', uint64_t, 48), ('completion_signal', hsa_signal_t, 56)])
hsa_kernel_dispatch_packet_t: TypeAlias = struct_hsa_kernel_dispatch_packet_s
@c.record
class struct_hsa_agent_dispatch_packet_s(c.Struct):
  SIZE = 64
  header: int
  type: int
  reserved0: int
  return_address: ctypes.c_void_p
  arg: c.Array[ctypes.c_uint64, Literal[4]]
  reserved2: int
  completion_signal: struct_hsa_signal_s
struct_hsa_agent_dispatch_packet_s.register_fields([('header', uint16_t, 0), ('type', uint16_t, 2), ('reserved0', uint32_t, 4), ('return_address', ctypes.c_void_p, 8), ('arg', c.Array[uint64_t, Literal[4]], 16), ('reserved2', uint64_t, 48), ('completion_signal', hsa_signal_t, 56)])
hsa_agent_dispatch_packet_t: TypeAlias = struct_hsa_agent_dispatch_packet_s
@c.record
class struct_hsa_barrier_and_packet_s(c.Struct):
  SIZE = 64
  header: int
  reserved0: int
  reserved1: int
  dep_signal: c.Array[struct_hsa_signal_s, Literal[5]]
  reserved2: int
  completion_signal: struct_hsa_signal_s
struct_hsa_barrier_and_packet_s.register_fields([('header', uint16_t, 0), ('reserved0', uint16_t, 2), ('reserved1', uint32_t, 4), ('dep_signal', c.Array[hsa_signal_t, Literal[5]], 8), ('reserved2', uint64_t, 48), ('completion_signal', hsa_signal_t, 56)])
hsa_barrier_and_packet_t: TypeAlias = struct_hsa_barrier_and_packet_s
@c.record
class struct_hsa_barrier_or_packet_s(c.Struct):
  SIZE = 64
  header: int
  reserved0: int
  reserved1: int
  dep_signal: c.Array[struct_hsa_signal_s, Literal[5]]
  reserved2: int
  completion_signal: struct_hsa_signal_s
struct_hsa_barrier_or_packet_s.register_fields([('header', uint16_t, 0), ('reserved0', uint16_t, 2), ('reserved1', uint32_t, 4), ('dep_signal', c.Array[hsa_signal_t, Literal[5]], 8), ('reserved2', uint64_t, 48), ('completion_signal', hsa_signal_t, 56)])
hsa_barrier_or_packet_t: TypeAlias = struct_hsa_barrier_or_packet_s
hsa_region_segment_t: dict[int, str] = {(HSA_REGION_SEGMENT_GLOBAL:=0): 'HSA_REGION_SEGMENT_GLOBAL', (HSA_REGION_SEGMENT_READONLY:=1): 'HSA_REGION_SEGMENT_READONLY', (HSA_REGION_SEGMENT_PRIVATE:=2): 'HSA_REGION_SEGMENT_PRIVATE', (HSA_REGION_SEGMENT_GROUP:=3): 'HSA_REGION_SEGMENT_GROUP', (HSA_REGION_SEGMENT_KERNARG:=4): 'HSA_REGION_SEGMENT_KERNARG'}
hsa_region_global_flag_t: dict[int, str] = {(HSA_REGION_GLOBAL_FLAG_KERNARG:=1): 'HSA_REGION_GLOBAL_FLAG_KERNARG', (HSA_REGION_GLOBAL_FLAG_FINE_GRAINED:=2): 'HSA_REGION_GLOBAL_FLAG_FINE_GRAINED', (HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED:=4): 'HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED', (HSA_REGION_GLOBAL_FLAG_EXTENDED_SCOPE_FINE_GRAINED:=8): 'HSA_REGION_GLOBAL_FLAG_EXTENDED_SCOPE_FINE_GRAINED'}
hsa_region_info_t: dict[int, str] = {(HSA_REGION_INFO_SEGMENT:=0): 'HSA_REGION_INFO_SEGMENT', (HSA_REGION_INFO_GLOBAL_FLAGS:=1): 'HSA_REGION_INFO_GLOBAL_FLAGS', (HSA_REGION_INFO_SIZE:=2): 'HSA_REGION_INFO_SIZE', (HSA_REGION_INFO_ALLOC_MAX_SIZE:=4): 'HSA_REGION_INFO_ALLOC_MAX_SIZE', (HSA_REGION_INFO_ALLOC_MAX_PRIVATE_WORKGROUP_SIZE:=8): 'HSA_REGION_INFO_ALLOC_MAX_PRIVATE_WORKGROUP_SIZE', (HSA_REGION_INFO_RUNTIME_ALLOC_ALLOWED:=5): 'HSA_REGION_INFO_RUNTIME_ALLOC_ALLOWED', (HSA_REGION_INFO_RUNTIME_ALLOC_GRANULE:=6): 'HSA_REGION_INFO_RUNTIME_ALLOC_GRANULE', (HSA_REGION_INFO_RUNTIME_ALLOC_ALIGNMENT:=7): 'HSA_REGION_INFO_RUNTIME_ALLOC_ALIGNMENT'}
@dll.bind(ctypes.c_uint32, hsa_region_t, ctypes.c_uint32, ctypes.c_void_p)
def hsa_region_get_info(region:hsa_region_t, attribute:ctypes.c_uint32, value:ctypes.c_void_p) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hsa_agent_t, c.CFUNCTYPE[ctypes.c_uint32, [hsa_region_t, ctypes.c_void_p]], ctypes.c_void_p)
def hsa_agent_iterate_regions(agent:hsa_agent_t, callback:c.CFUNCTYPE[ctypes.c_uint32, [hsa_region_t, ctypes.c_void_p]], data:ctypes.c_void_p) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hsa_region_t, size_t, c.POINTER[ctypes.c_void_p])
def hsa_memory_allocate(region:hsa_region_t, size:size_t, ptr:c.POINTER[ctypes.c_void_p]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p)
def hsa_memory_free(ptr:ctypes.c_void_p) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, size_t)
def hsa_memory_copy(dst:ctypes.c_void_p, src:ctypes.c_void_p, size:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, hsa_agent_t, ctypes.c_uint32)
def hsa_memory_assign_agent(ptr:ctypes.c_void_p, agent:hsa_agent_t, access:ctypes.c_uint32) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, size_t)
def hsa_memory_register(ptr:ctypes.c_void_p, size:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, size_t)
def hsa_memory_deregister(ptr:ctypes.c_void_p, size:size_t) -> ctypes.c_uint32: ...
@c.record
class struct_hsa_isa_s(c.Struct):
  SIZE = 8
  handle: int
struct_hsa_isa_s.register_fields([('handle', uint64_t, 0)])
hsa_isa_t: TypeAlias = struct_hsa_isa_s
@dll.bind(ctypes.c_uint32, c.POINTER[ctypes.c_char], c.POINTER[hsa_isa_t])
def hsa_isa_from_name(name:c.POINTER[ctypes.c_char], isa:c.POINTER[hsa_isa_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hsa_agent_t, c.CFUNCTYPE[ctypes.c_uint32, [hsa_isa_t, ctypes.c_void_p]], ctypes.c_void_p)
def hsa_agent_iterate_isas(agent:hsa_agent_t, callback:c.CFUNCTYPE[ctypes.c_uint32, [hsa_isa_t, ctypes.c_void_p]], data:ctypes.c_void_p) -> ctypes.c_uint32: ...
hsa_isa_info_t: dict[int, str] = {(HSA_ISA_INFO_NAME_LENGTH:=0): 'HSA_ISA_INFO_NAME_LENGTH', (HSA_ISA_INFO_NAME:=1): 'HSA_ISA_INFO_NAME', (HSA_ISA_INFO_CALL_CONVENTION_COUNT:=2): 'HSA_ISA_INFO_CALL_CONVENTION_COUNT', (HSA_ISA_INFO_CALL_CONVENTION_INFO_WAVEFRONT_SIZE:=3): 'HSA_ISA_INFO_CALL_CONVENTION_INFO_WAVEFRONT_SIZE', (HSA_ISA_INFO_CALL_CONVENTION_INFO_WAVEFRONTS_PER_COMPUTE_UNIT:=4): 'HSA_ISA_INFO_CALL_CONVENTION_INFO_WAVEFRONTS_PER_COMPUTE_UNIT', (HSA_ISA_INFO_MACHINE_MODELS:=5): 'HSA_ISA_INFO_MACHINE_MODELS', (HSA_ISA_INFO_PROFILES:=6): 'HSA_ISA_INFO_PROFILES', (HSA_ISA_INFO_DEFAULT_FLOAT_ROUNDING_MODES:=7): 'HSA_ISA_INFO_DEFAULT_FLOAT_ROUNDING_MODES', (HSA_ISA_INFO_BASE_PROFILE_DEFAULT_FLOAT_ROUNDING_MODES:=8): 'HSA_ISA_INFO_BASE_PROFILE_DEFAULT_FLOAT_ROUNDING_MODES', (HSA_ISA_INFO_FAST_F16_OPERATION:=9): 'HSA_ISA_INFO_FAST_F16_OPERATION', (HSA_ISA_INFO_WORKGROUP_MAX_DIM:=12): 'HSA_ISA_INFO_WORKGROUP_MAX_DIM', (HSA_ISA_INFO_WORKGROUP_MAX_SIZE:=13): 'HSA_ISA_INFO_WORKGROUP_MAX_SIZE', (HSA_ISA_INFO_GRID_MAX_DIM:=14): 'HSA_ISA_INFO_GRID_MAX_DIM', (HSA_ISA_INFO_GRID_MAX_SIZE:=16): 'HSA_ISA_INFO_GRID_MAX_SIZE', (HSA_ISA_INFO_FBARRIER_MAX_SIZE:=17): 'HSA_ISA_INFO_FBARRIER_MAX_SIZE'}
@dll.bind(ctypes.c_uint32, hsa_isa_t, ctypes.c_uint32, uint32_t, ctypes.c_void_p)
def hsa_isa_get_info(isa:hsa_isa_t, attribute:ctypes.c_uint32, index:uint32_t, value:ctypes.c_void_p) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hsa_isa_t, ctypes.c_uint32, ctypes.c_void_p)
def hsa_isa_get_info_alt(isa:hsa_isa_t, attribute:ctypes.c_uint32, value:ctypes.c_void_p) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hsa_isa_t, ctypes.c_uint32, c.POINTER[uint16_t])
def hsa_isa_get_exception_policies(isa:hsa_isa_t, profile:ctypes.c_uint32, mask:c.POINTER[uint16_t]) -> ctypes.c_uint32: ...
hsa_fp_type_t: dict[int, str] = {(HSA_FP_TYPE_16:=1): 'HSA_FP_TYPE_16', (HSA_FP_TYPE_32:=2): 'HSA_FP_TYPE_32', (HSA_FP_TYPE_64:=4): 'HSA_FP_TYPE_64'}
hsa_flush_mode_t: dict[int, str] = {(HSA_FLUSH_MODE_FTZ:=1): 'HSA_FLUSH_MODE_FTZ', (HSA_FLUSH_MODE_NON_FTZ:=2): 'HSA_FLUSH_MODE_NON_FTZ'}
hsa_round_method_t: dict[int, str] = {(HSA_ROUND_METHOD_SINGLE:=1): 'HSA_ROUND_METHOD_SINGLE', (HSA_ROUND_METHOD_DOUBLE:=2): 'HSA_ROUND_METHOD_DOUBLE'}
@dll.bind(ctypes.c_uint32, hsa_isa_t, ctypes.c_uint32, ctypes.c_uint32, c.POINTER[ctypes.c_uint32])
def hsa_isa_get_round_method(isa:hsa_isa_t, fp_type:ctypes.c_uint32, flush_mode:ctypes.c_uint32, round_method:c.POINTER[ctypes.c_uint32]) -> ctypes.c_uint32: ...
@c.record
class struct_hsa_wavefront_s(c.Struct):
  SIZE = 8
  handle: int
struct_hsa_wavefront_s.register_fields([('handle', uint64_t, 0)])
hsa_wavefront_t: TypeAlias = struct_hsa_wavefront_s
hsa_wavefront_info_t: dict[int, str] = {(HSA_WAVEFRONT_INFO_SIZE:=0): 'HSA_WAVEFRONT_INFO_SIZE'}
@dll.bind(ctypes.c_uint32, hsa_wavefront_t, ctypes.c_uint32, ctypes.c_void_p)
def hsa_wavefront_get_info(wavefront:hsa_wavefront_t, attribute:ctypes.c_uint32, value:ctypes.c_void_p) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hsa_isa_t, c.CFUNCTYPE[ctypes.c_uint32, [hsa_wavefront_t, ctypes.c_void_p]], ctypes.c_void_p)
def hsa_isa_iterate_wavefronts(isa:hsa_isa_t, callback:c.CFUNCTYPE[ctypes.c_uint32, [hsa_wavefront_t, ctypes.c_void_p]], data:ctypes.c_void_p) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hsa_isa_t, hsa_isa_t, c.POINTER[ctypes.c_bool])
def hsa_isa_compatible(code_object_isa:hsa_isa_t, agent_isa:hsa_isa_t, result:c.POINTER[ctypes.c_bool]) -> ctypes.c_uint32: ...
@c.record
class struct_hsa_code_object_reader_s(c.Struct):
  SIZE = 8
  handle: int
struct_hsa_code_object_reader_s.register_fields([('handle', uint64_t, 0)])
hsa_code_object_reader_t: TypeAlias = struct_hsa_code_object_reader_s
@dll.bind(ctypes.c_uint32, hsa_file_t, c.POINTER[hsa_code_object_reader_t])
def hsa_code_object_reader_create_from_file(file:hsa_file_t, code_object_reader:c.POINTER[hsa_code_object_reader_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, size_t, c.POINTER[hsa_code_object_reader_t])
def hsa_code_object_reader_create_from_memory(code_object:ctypes.c_void_p, size:size_t, code_object_reader:c.POINTER[hsa_code_object_reader_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hsa_code_object_reader_t)
def hsa_code_object_reader_destroy(code_object_reader:hsa_code_object_reader_t) -> ctypes.c_uint32: ...
@c.record
class struct_hsa_executable_s(c.Struct):
  SIZE = 8
  handle: int
struct_hsa_executable_s.register_fields([('handle', uint64_t, 0)])
hsa_executable_t: TypeAlias = struct_hsa_executable_s
hsa_executable_state_t: dict[int, str] = {(HSA_EXECUTABLE_STATE_UNFROZEN:=0): 'HSA_EXECUTABLE_STATE_UNFROZEN', (HSA_EXECUTABLE_STATE_FROZEN:=1): 'HSA_EXECUTABLE_STATE_FROZEN'}
@dll.bind(ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, c.POINTER[ctypes.c_char], c.POINTER[hsa_executable_t])
def hsa_executable_create(profile:ctypes.c_uint32, executable_state:ctypes.c_uint32, options:c.POINTER[ctypes.c_char], executable:c.POINTER[hsa_executable_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, c.POINTER[ctypes.c_char], c.POINTER[hsa_executable_t])
def hsa_executable_create_alt(profile:ctypes.c_uint32, default_float_rounding_mode:ctypes.c_uint32, options:c.POINTER[ctypes.c_char], executable:c.POINTER[hsa_executable_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hsa_executable_t)
def hsa_executable_destroy(executable:hsa_executable_t) -> ctypes.c_uint32: ...
@c.record
class struct_hsa_loaded_code_object_s(c.Struct):
  SIZE = 8
  handle: int
struct_hsa_loaded_code_object_s.register_fields([('handle', uint64_t, 0)])
hsa_loaded_code_object_t: TypeAlias = struct_hsa_loaded_code_object_s
@dll.bind(ctypes.c_uint32, hsa_executable_t, hsa_code_object_reader_t, c.POINTER[ctypes.c_char], c.POINTER[hsa_loaded_code_object_t])
def hsa_executable_load_program_code_object(executable:hsa_executable_t, code_object_reader:hsa_code_object_reader_t, options:c.POINTER[ctypes.c_char], loaded_code_object:c.POINTER[hsa_loaded_code_object_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hsa_executable_t, hsa_agent_t, hsa_code_object_reader_t, c.POINTER[ctypes.c_char], c.POINTER[hsa_loaded_code_object_t])
def hsa_executable_load_agent_code_object(executable:hsa_executable_t, agent:hsa_agent_t, code_object_reader:hsa_code_object_reader_t, options:c.POINTER[ctypes.c_char], loaded_code_object:c.POINTER[hsa_loaded_code_object_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hsa_executable_t, c.POINTER[ctypes.c_char])
def hsa_executable_freeze(executable:hsa_executable_t, options:c.POINTER[ctypes.c_char]) -> ctypes.c_uint32: ...
hsa_executable_info_t: dict[int, str] = {(HSA_EXECUTABLE_INFO_PROFILE:=1): 'HSA_EXECUTABLE_INFO_PROFILE', (HSA_EXECUTABLE_INFO_STATE:=2): 'HSA_EXECUTABLE_INFO_STATE', (HSA_EXECUTABLE_INFO_DEFAULT_FLOAT_ROUNDING_MODE:=3): 'HSA_EXECUTABLE_INFO_DEFAULT_FLOAT_ROUNDING_MODE'}
@dll.bind(ctypes.c_uint32, hsa_executable_t, ctypes.c_uint32, ctypes.c_void_p)
def hsa_executable_get_info(executable:hsa_executable_t, attribute:ctypes.c_uint32, value:ctypes.c_void_p) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hsa_executable_t, c.POINTER[ctypes.c_char], ctypes.c_void_p)
def hsa_executable_global_variable_define(executable:hsa_executable_t, variable_name:c.POINTER[ctypes.c_char], address:ctypes.c_void_p) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hsa_executable_t, hsa_agent_t, c.POINTER[ctypes.c_char], ctypes.c_void_p)
def hsa_executable_agent_global_variable_define(executable:hsa_executable_t, agent:hsa_agent_t, variable_name:c.POINTER[ctypes.c_char], address:ctypes.c_void_p) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hsa_executable_t, hsa_agent_t, c.POINTER[ctypes.c_char], ctypes.c_void_p)
def hsa_executable_readonly_variable_define(executable:hsa_executable_t, agent:hsa_agent_t, variable_name:c.POINTER[ctypes.c_char], address:ctypes.c_void_p) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hsa_executable_t, c.POINTER[uint32_t])
def hsa_executable_validate(executable:hsa_executable_t, result:c.POINTER[uint32_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hsa_executable_t, c.POINTER[ctypes.c_char], c.POINTER[uint32_t])
def hsa_executable_validate_alt(executable:hsa_executable_t, options:c.POINTER[ctypes.c_char], result:c.POINTER[uint32_t]) -> ctypes.c_uint32: ...
@c.record
class struct_hsa_executable_symbol_s(c.Struct):
  SIZE = 8
  handle: int
struct_hsa_executable_symbol_s.register_fields([('handle', uint64_t, 0)])
hsa_executable_symbol_t: TypeAlias = struct_hsa_executable_symbol_s
int32_t: TypeAlias = ctypes.c_int32
@dll.bind(ctypes.c_uint32, hsa_executable_t, c.POINTER[ctypes.c_char], c.POINTER[ctypes.c_char], hsa_agent_t, int32_t, c.POINTER[hsa_executable_symbol_t])
def hsa_executable_get_symbol(executable:hsa_executable_t, module_name:c.POINTER[ctypes.c_char], symbol_name:c.POINTER[ctypes.c_char], agent:hsa_agent_t, call_convention:int32_t, symbol:c.POINTER[hsa_executable_symbol_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hsa_executable_t, c.POINTER[ctypes.c_char], c.POINTER[hsa_agent_t], c.POINTER[hsa_executable_symbol_t])
def hsa_executable_get_symbol_by_name(executable:hsa_executable_t, symbol_name:c.POINTER[ctypes.c_char], agent:c.POINTER[hsa_agent_t], symbol:c.POINTER[hsa_executable_symbol_t]) -> ctypes.c_uint32: ...
hsa_symbol_kind_t: dict[int, str] = {(HSA_SYMBOL_KIND_VARIABLE:=0): 'HSA_SYMBOL_KIND_VARIABLE', (HSA_SYMBOL_KIND_KERNEL:=1): 'HSA_SYMBOL_KIND_KERNEL', (HSA_SYMBOL_KIND_INDIRECT_FUNCTION:=2): 'HSA_SYMBOL_KIND_INDIRECT_FUNCTION'}
hsa_symbol_linkage_t: dict[int, str] = {(HSA_SYMBOL_LINKAGE_MODULE:=0): 'HSA_SYMBOL_LINKAGE_MODULE', (HSA_SYMBOL_LINKAGE_PROGRAM:=1): 'HSA_SYMBOL_LINKAGE_PROGRAM'}
hsa_variable_allocation_t: dict[int, str] = {(HSA_VARIABLE_ALLOCATION_AGENT:=0): 'HSA_VARIABLE_ALLOCATION_AGENT', (HSA_VARIABLE_ALLOCATION_PROGRAM:=1): 'HSA_VARIABLE_ALLOCATION_PROGRAM'}
hsa_variable_segment_t: dict[int, str] = {(HSA_VARIABLE_SEGMENT_GLOBAL:=0): 'HSA_VARIABLE_SEGMENT_GLOBAL', (HSA_VARIABLE_SEGMENT_READONLY:=1): 'HSA_VARIABLE_SEGMENT_READONLY'}
hsa_executable_symbol_info_t: dict[int, str] = {(HSA_EXECUTABLE_SYMBOL_INFO_TYPE:=0): 'HSA_EXECUTABLE_SYMBOL_INFO_TYPE', (HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH:=1): 'HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH', (HSA_EXECUTABLE_SYMBOL_INFO_NAME:=2): 'HSA_EXECUTABLE_SYMBOL_INFO_NAME', (HSA_EXECUTABLE_SYMBOL_INFO_MODULE_NAME_LENGTH:=3): 'HSA_EXECUTABLE_SYMBOL_INFO_MODULE_NAME_LENGTH', (HSA_EXECUTABLE_SYMBOL_INFO_MODULE_NAME:=4): 'HSA_EXECUTABLE_SYMBOL_INFO_MODULE_NAME', (HSA_EXECUTABLE_SYMBOL_INFO_AGENT:=20): 'HSA_EXECUTABLE_SYMBOL_INFO_AGENT', (HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS:=21): 'HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS', (HSA_EXECUTABLE_SYMBOL_INFO_LINKAGE:=5): 'HSA_EXECUTABLE_SYMBOL_INFO_LINKAGE', (HSA_EXECUTABLE_SYMBOL_INFO_IS_DEFINITION:=17): 'HSA_EXECUTABLE_SYMBOL_INFO_IS_DEFINITION', (HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ALLOCATION:=6): 'HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ALLOCATION', (HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_SEGMENT:=7): 'HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_SEGMENT', (HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ALIGNMENT:=8): 'HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ALIGNMENT', (HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_SIZE:=9): 'HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_SIZE', (HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_IS_CONST:=10): 'HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_IS_CONST', (HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT:=22): 'HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT', (HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE:=11): 'HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE', (HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT:=12): 'HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT', (HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE:=13): 'HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE', (HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE:=14): 'HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE', (HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_DYNAMIC_CALLSTACK:=15): 'HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_DYNAMIC_CALLSTACK', (HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_CALL_CONVENTION:=18): 'HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_CALL_CONVENTION', (HSA_EXECUTABLE_SYMBOL_INFO_INDIRECT_FUNCTION_OBJECT:=23): 'HSA_EXECUTABLE_SYMBOL_INFO_INDIRECT_FUNCTION_OBJECT', (HSA_EXECUTABLE_SYMBOL_INFO_INDIRECT_FUNCTION_CALL_CONVENTION:=16): 'HSA_EXECUTABLE_SYMBOL_INFO_INDIRECT_FUNCTION_CALL_CONVENTION'}
@dll.bind(ctypes.c_uint32, hsa_executable_symbol_t, ctypes.c_uint32, ctypes.c_void_p)
def hsa_executable_symbol_get_info(executable_symbol:hsa_executable_symbol_t, attribute:ctypes.c_uint32, value:ctypes.c_void_p) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hsa_executable_t, c.CFUNCTYPE[ctypes.c_uint32, [hsa_executable_t, hsa_executable_symbol_t, ctypes.c_void_p]], ctypes.c_void_p)
def hsa_executable_iterate_symbols(executable:hsa_executable_t, callback:c.CFUNCTYPE[ctypes.c_uint32, [hsa_executable_t, hsa_executable_symbol_t, ctypes.c_void_p]], data:ctypes.c_void_p) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hsa_executable_t, hsa_agent_t, c.CFUNCTYPE[ctypes.c_uint32, [hsa_executable_t, hsa_agent_t, hsa_executable_symbol_t, ctypes.c_void_p]], ctypes.c_void_p)
def hsa_executable_iterate_agent_symbols(executable:hsa_executable_t, agent:hsa_agent_t, callback:c.CFUNCTYPE[ctypes.c_uint32, [hsa_executable_t, hsa_agent_t, hsa_executable_symbol_t, ctypes.c_void_p]], data:ctypes.c_void_p) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hsa_executable_t, c.CFUNCTYPE[ctypes.c_uint32, [hsa_executable_t, hsa_executable_symbol_t, ctypes.c_void_p]], ctypes.c_void_p)
def hsa_executable_iterate_program_symbols(executable:hsa_executable_t, callback:c.CFUNCTYPE[ctypes.c_uint32, [hsa_executable_t, hsa_executable_symbol_t, ctypes.c_void_p]], data:ctypes.c_void_p) -> ctypes.c_uint32: ...
@c.record
class struct_hsa_code_object_s(c.Struct):
  SIZE = 8
  handle: int
struct_hsa_code_object_s.register_fields([('handle', uint64_t, 0)])
hsa_code_object_t: TypeAlias = struct_hsa_code_object_s
@c.record
class struct_hsa_callback_data_s(c.Struct):
  SIZE = 8
  handle: int
struct_hsa_callback_data_s.register_fields([('handle', uint64_t, 0)])
hsa_callback_data_t: TypeAlias = struct_hsa_callback_data_s
@dll.bind(ctypes.c_uint32, hsa_code_object_t, c.CFUNCTYPE[ctypes.c_uint32, [size_t, hsa_callback_data_t, c.POINTER[ctypes.c_void_p]]], hsa_callback_data_t, c.POINTER[ctypes.c_char], c.POINTER[ctypes.c_void_p], c.POINTER[size_t])
def hsa_code_object_serialize(code_object:hsa_code_object_t, alloc_callback:c.CFUNCTYPE[ctypes.c_uint32, [size_t, hsa_callback_data_t, c.POINTER[ctypes.c_void_p]]], callback_data:hsa_callback_data_t, options:c.POINTER[ctypes.c_char], serialized_code_object:c.POINTER[ctypes.c_void_p], serialized_code_object_size:c.POINTER[size_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, size_t, c.POINTER[ctypes.c_char], c.POINTER[hsa_code_object_t])
def hsa_code_object_deserialize(serialized_code_object:ctypes.c_void_p, serialized_code_object_size:size_t, options:c.POINTER[ctypes.c_char], code_object:c.POINTER[hsa_code_object_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hsa_code_object_t)
def hsa_code_object_destroy(code_object:hsa_code_object_t) -> ctypes.c_uint32: ...
hsa_code_object_type_t: dict[int, str] = {(HSA_CODE_OBJECT_TYPE_PROGRAM:=0): 'HSA_CODE_OBJECT_TYPE_PROGRAM'}
hsa_code_object_info_t: dict[int, str] = {(HSA_CODE_OBJECT_INFO_VERSION:=0): 'HSA_CODE_OBJECT_INFO_VERSION', (HSA_CODE_OBJECT_INFO_TYPE:=1): 'HSA_CODE_OBJECT_INFO_TYPE', (HSA_CODE_OBJECT_INFO_ISA:=2): 'HSA_CODE_OBJECT_INFO_ISA', (HSA_CODE_OBJECT_INFO_MACHINE_MODEL:=3): 'HSA_CODE_OBJECT_INFO_MACHINE_MODEL', (HSA_CODE_OBJECT_INFO_PROFILE:=4): 'HSA_CODE_OBJECT_INFO_PROFILE', (HSA_CODE_OBJECT_INFO_DEFAULT_FLOAT_ROUNDING_MODE:=5): 'HSA_CODE_OBJECT_INFO_DEFAULT_FLOAT_ROUNDING_MODE'}
@dll.bind(ctypes.c_uint32, hsa_code_object_t, ctypes.c_uint32, ctypes.c_void_p)
def hsa_code_object_get_info(code_object:hsa_code_object_t, attribute:ctypes.c_uint32, value:ctypes.c_void_p) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hsa_executable_t, hsa_agent_t, hsa_code_object_t, c.POINTER[ctypes.c_char])
def hsa_executable_load_code_object(executable:hsa_executable_t, agent:hsa_agent_t, code_object:hsa_code_object_t, options:c.POINTER[ctypes.c_char]) -> ctypes.c_uint32: ...
@c.record
class struct_hsa_code_symbol_s(c.Struct):
  SIZE = 8
  handle: int
struct_hsa_code_symbol_s.register_fields([('handle', uint64_t, 0)])
hsa_code_symbol_t: TypeAlias = struct_hsa_code_symbol_s
@dll.bind(ctypes.c_uint32, hsa_code_object_t, c.POINTER[ctypes.c_char], c.POINTER[hsa_code_symbol_t])
def hsa_code_object_get_symbol(code_object:hsa_code_object_t, symbol_name:c.POINTER[ctypes.c_char], symbol:c.POINTER[hsa_code_symbol_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hsa_code_object_t, c.POINTER[ctypes.c_char], c.POINTER[ctypes.c_char], c.POINTER[hsa_code_symbol_t])
def hsa_code_object_get_symbol_from_name(code_object:hsa_code_object_t, module_name:c.POINTER[ctypes.c_char], symbol_name:c.POINTER[ctypes.c_char], symbol:c.POINTER[hsa_code_symbol_t]) -> ctypes.c_uint32: ...
hsa_code_symbol_info_t: dict[int, str] = {(HSA_CODE_SYMBOL_INFO_TYPE:=0): 'HSA_CODE_SYMBOL_INFO_TYPE', (HSA_CODE_SYMBOL_INFO_NAME_LENGTH:=1): 'HSA_CODE_SYMBOL_INFO_NAME_LENGTH', (HSA_CODE_SYMBOL_INFO_NAME:=2): 'HSA_CODE_SYMBOL_INFO_NAME', (HSA_CODE_SYMBOL_INFO_MODULE_NAME_LENGTH:=3): 'HSA_CODE_SYMBOL_INFO_MODULE_NAME_LENGTH', (HSA_CODE_SYMBOL_INFO_MODULE_NAME:=4): 'HSA_CODE_SYMBOL_INFO_MODULE_NAME', (HSA_CODE_SYMBOL_INFO_LINKAGE:=5): 'HSA_CODE_SYMBOL_INFO_LINKAGE', (HSA_CODE_SYMBOL_INFO_IS_DEFINITION:=17): 'HSA_CODE_SYMBOL_INFO_IS_DEFINITION', (HSA_CODE_SYMBOL_INFO_VARIABLE_ALLOCATION:=6): 'HSA_CODE_SYMBOL_INFO_VARIABLE_ALLOCATION', (HSA_CODE_SYMBOL_INFO_VARIABLE_SEGMENT:=7): 'HSA_CODE_SYMBOL_INFO_VARIABLE_SEGMENT', (HSA_CODE_SYMBOL_INFO_VARIABLE_ALIGNMENT:=8): 'HSA_CODE_SYMBOL_INFO_VARIABLE_ALIGNMENT', (HSA_CODE_SYMBOL_INFO_VARIABLE_SIZE:=9): 'HSA_CODE_SYMBOL_INFO_VARIABLE_SIZE', (HSA_CODE_SYMBOL_INFO_VARIABLE_IS_CONST:=10): 'HSA_CODE_SYMBOL_INFO_VARIABLE_IS_CONST', (HSA_CODE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE:=11): 'HSA_CODE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE', (HSA_CODE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT:=12): 'HSA_CODE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT', (HSA_CODE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE:=13): 'HSA_CODE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE', (HSA_CODE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE:=14): 'HSA_CODE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE', (HSA_CODE_SYMBOL_INFO_KERNEL_DYNAMIC_CALLSTACK:=15): 'HSA_CODE_SYMBOL_INFO_KERNEL_DYNAMIC_CALLSTACK', (HSA_CODE_SYMBOL_INFO_KERNEL_CALL_CONVENTION:=18): 'HSA_CODE_SYMBOL_INFO_KERNEL_CALL_CONVENTION', (HSA_CODE_SYMBOL_INFO_INDIRECT_FUNCTION_CALL_CONVENTION:=16): 'HSA_CODE_SYMBOL_INFO_INDIRECT_FUNCTION_CALL_CONVENTION', (HSA_CODE_SYMBOL_INFO_KERNEL_WAVEFRONT_SIZE:=19): 'HSA_CODE_SYMBOL_INFO_KERNEL_WAVEFRONT_SIZE'}
@dll.bind(ctypes.c_uint32, hsa_code_symbol_t, ctypes.c_uint32, ctypes.c_void_p)
def hsa_code_symbol_get_info(code_symbol:hsa_code_symbol_t, attribute:ctypes.c_uint32, value:ctypes.c_void_p) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hsa_code_object_t, c.CFUNCTYPE[ctypes.c_uint32, [hsa_code_object_t, hsa_code_symbol_t, ctypes.c_void_p]], ctypes.c_void_p)
def hsa_code_object_iterate_symbols(code_object:hsa_code_object_t, callback:c.CFUNCTYPE[ctypes.c_uint32, [hsa_code_object_t, hsa_code_symbol_t, ctypes.c_void_p]], data:ctypes.c_void_p) -> ctypes.c_uint32: ...
hsa_signal_condition32_t: TypeAlias = ctypes.c_uint32
hsa_amd_packet_type_t: dict[int, str] = {(HSA_AMD_PACKET_TYPE_BARRIER_VALUE:=2): 'HSA_AMD_PACKET_TYPE_BARRIER_VALUE', (HSA_AMD_PACKET_TYPE_AIE_ERT:=3): 'HSA_AMD_PACKET_TYPE_AIE_ERT'}
hsa_amd_packet_type8_t: TypeAlias = ctypes.c_ubyte
@c.record
class struct_hsa_amd_packet_header_s(c.Struct):
  SIZE = 4
  header: int
  AmdFormat: int
  reserved: int
uint8_t: TypeAlias = ctypes.c_ubyte
struct_hsa_amd_packet_header_s.register_fields([('header', uint16_t, 0), ('AmdFormat', hsa_amd_packet_type8_t, 2), ('reserved', uint8_t, 3)])
hsa_amd_vendor_packet_header_t: TypeAlias = struct_hsa_amd_packet_header_s
@c.record
class struct_hsa_amd_barrier_value_packet_s(c.Struct):
  SIZE = 64
  header: struct_hsa_amd_packet_header_s
  reserved0: int
  signal: struct_hsa_signal_s
  value: int
  mask: int
  cond: int
  reserved1: int
  reserved2: int
  reserved3: int
  completion_signal: struct_hsa_signal_s
struct_hsa_amd_barrier_value_packet_s.register_fields([('header', hsa_amd_vendor_packet_header_t, 0), ('reserved0', uint32_t, 4), ('signal', hsa_signal_t, 8), ('value', hsa_signal_value_t, 16), ('mask', hsa_signal_value_t, 24), ('cond', hsa_signal_condition32_t, 32), ('reserved1', uint32_t, 36), ('reserved2', uint64_t, 40), ('reserved3', uint64_t, 48), ('completion_signal', hsa_signal_t, 56)])
hsa_amd_barrier_value_packet_t: TypeAlias = struct_hsa_amd_barrier_value_packet_s
hsa_amd_aie_ert_state: dict[int, str] = {(HSA_AMD_AIE_ERT_STATE_NEW:=1): 'HSA_AMD_AIE_ERT_STATE_NEW', (HSA_AMD_AIE_ERT_STATE_QUEUED:=2): 'HSA_AMD_AIE_ERT_STATE_QUEUED', (HSA_AMD_AIE_ERT_STATE_RUNNING:=3): 'HSA_AMD_AIE_ERT_STATE_RUNNING', (HSA_AMD_AIE_ERT_STATE_COMPLETED:=4): 'HSA_AMD_AIE_ERT_STATE_COMPLETED', (HSA_AMD_AIE_ERT_STATE_ERROR:=5): 'HSA_AMD_AIE_ERT_STATE_ERROR', (HSA_AMD_AIE_ERT_STATE_ABORT:=6): 'HSA_AMD_AIE_ERT_STATE_ABORT', (HSA_AMD_AIE_ERT_STATE_SUBMITTED:=7): 'HSA_AMD_AIE_ERT_STATE_SUBMITTED', (HSA_AMD_AIE_ERT_STATE_TIMEOUT:=8): 'HSA_AMD_AIE_ERT_STATE_TIMEOUT', (HSA_AMD_AIE_ERT_STATE_NORESPONSE:=9): 'HSA_AMD_AIE_ERT_STATE_NORESPONSE', (HSA_AMD_AIE_ERT_STATE_SKERROR:=10): 'HSA_AMD_AIE_ERT_STATE_SKERROR', (HSA_AMD_AIE_ERT_STATE_SKCRASHED:=11): 'HSA_AMD_AIE_ERT_STATE_SKCRASHED', (HSA_AMD_AIE_ERT_STATE_MAX:=12): 'HSA_AMD_AIE_ERT_STATE_MAX'}
hsa_amd_aie_ert_cmd_opcode_t: dict[int, str] = {(HSA_AMD_AIE_ERT_START_CU:=0): 'HSA_AMD_AIE_ERT_START_CU', (HSA_AMD_AIE_ERT_START_KERNEL:=0): 'HSA_AMD_AIE_ERT_START_KERNEL', (HSA_AMD_AIE_ERT_CONFIGURE:=2): 'HSA_AMD_AIE_ERT_CONFIGURE', (HSA_AMD_AIE_ERT_EXIT:=3): 'HSA_AMD_AIE_ERT_EXIT', (HSA_AMD_AIE_ERT_ABORT:=4): 'HSA_AMD_AIE_ERT_ABORT', (HSA_AMD_AIE_ERT_EXEC_WRITE:=5): 'HSA_AMD_AIE_ERT_EXEC_WRITE', (HSA_AMD_AIE_ERT_CU_STAT:=6): 'HSA_AMD_AIE_ERT_CU_STAT', (HSA_AMD_AIE_ERT_START_COPYBO:=7): 'HSA_AMD_AIE_ERT_START_COPYBO', (HSA_AMD_AIE_ERT_SK_CONFIG:=8): 'HSA_AMD_AIE_ERT_SK_CONFIG', (HSA_AMD_AIE_ERT_SK_START:=9): 'HSA_AMD_AIE_ERT_SK_START', (HSA_AMD_AIE_ERT_SK_UNCONFIG:=10): 'HSA_AMD_AIE_ERT_SK_UNCONFIG', (HSA_AMD_AIE_ERT_INIT_CU:=11): 'HSA_AMD_AIE_ERT_INIT_CU', (HSA_AMD_AIE_ERT_START_FA:=12): 'HSA_AMD_AIE_ERT_START_FA', (HSA_AMD_AIE_ERT_CLK_CALIB:=13): 'HSA_AMD_AIE_ERT_CLK_CALIB', (HSA_AMD_AIE_ERT_MB_VALIDATE:=14): 'HSA_AMD_AIE_ERT_MB_VALIDATE', (HSA_AMD_AIE_ERT_START_KEY_VAL:=15): 'HSA_AMD_AIE_ERT_START_KEY_VAL', (HSA_AMD_AIE_ERT_ACCESS_TEST_C:=16): 'HSA_AMD_AIE_ERT_ACCESS_TEST_C', (HSA_AMD_AIE_ERT_ACCESS_TEST:=17): 'HSA_AMD_AIE_ERT_ACCESS_TEST', (HSA_AMD_AIE_ERT_START_DPU:=18): 'HSA_AMD_AIE_ERT_START_DPU', (HSA_AMD_AIE_ERT_CMD_CHAIN:=19): 'HSA_AMD_AIE_ERT_CMD_CHAIN', (HSA_AMD_AIE_ERT_START_NPU:=20): 'HSA_AMD_AIE_ERT_START_NPU', (HSA_AMD_AIE_ERT_START_NPU_PREEMPT:=21): 'HSA_AMD_AIE_ERT_START_NPU_PREEMPT'}
@c.record
class struct_hsa_amd_aie_ert_start_kernel_data_s(c.Struct):
  SIZE = 8
  pdi_addr: ctypes.c_void_p
  data: c.Array[ctypes.c_uint32, Literal[0]]
struct_hsa_amd_aie_ert_start_kernel_data_s.register_fields([('pdi_addr', ctypes.c_void_p, 0), ('data', c.Array[uint32_t, Literal[0]], 8)])
hsa_amd_aie_ert_start_kernel_data_t: TypeAlias = struct_hsa_amd_aie_ert_start_kernel_data_s
@c.record
class struct_hsa_amd_aie_ert_packet_s(c.Struct):
  SIZE = 64
  header: struct_hsa_amd_packet_header_s
  state: int
  custom: int
  count: int
  opcode: int
  type: int
  reserved0: int
  reserved1: int
  reserved2: int
  reserved3: int
  reserved4: int
  reserved5: int
  payload_data: int
struct_hsa_amd_aie_ert_packet_s.register_fields([('header', hsa_amd_vendor_packet_header_t, 0), ('state', uint32_t, 4, 4, 0), ('custom', uint32_t, 4, 8, 4), ('count', uint32_t, 5, 11, 4), ('opcode', uint32_t, 6, 5, 7), ('type', uint32_t, 7, 4, 4), ('reserved0', uint64_t, 8), ('reserved1', uint64_t, 16), ('reserved2', uint64_t, 24), ('reserved3', uint64_t, 32), ('reserved4', uint64_t, 40), ('reserved5', uint64_t, 48), ('payload_data', uint64_t, 56)])
hsa_amd_aie_ert_packet_t: TypeAlias = struct_hsa_amd_aie_ert_packet_s
_anonenum0: dict[int, str] = {(HSA_STATUS_ERROR_INVALID_MEMORY_POOL:=40): 'HSA_STATUS_ERROR_INVALID_MEMORY_POOL', (HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION:=41): 'HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION', (HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION:=42): 'HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION', (HSA_STATUS_ERROR_MEMORY_FAULT:=43): 'HSA_STATUS_ERROR_MEMORY_FAULT', (HSA_STATUS_CU_MASK_REDUCED:=44): 'HSA_STATUS_CU_MASK_REDUCED', (HSA_STATUS_ERROR_OUT_OF_REGISTERS:=45): 'HSA_STATUS_ERROR_OUT_OF_REGISTERS', (HSA_STATUS_ERROR_RESOURCE_BUSY:=46): 'HSA_STATUS_ERROR_RESOURCE_BUSY', (HSA_STATUS_ERROR_NOT_SUPPORTED:=47): 'HSA_STATUS_ERROR_NOT_SUPPORTED'}
hsa_amd_iommu_version_t: dict[int, str] = {(HSA_IOMMU_SUPPORT_NONE:=0): 'HSA_IOMMU_SUPPORT_NONE', (HSA_IOMMU_SUPPORT_V2:=1): 'HSA_IOMMU_SUPPORT_V2'}
@c.record
class struct_hsa_amd_clock_counters_s(c.Struct):
  SIZE = 32
  gpu_clock_counter: int
  cpu_clock_counter: int
  system_clock_counter: int
  system_clock_frequency: int
struct_hsa_amd_clock_counters_s.register_fields([('gpu_clock_counter', uint64_t, 0), ('cpu_clock_counter', uint64_t, 8), ('system_clock_counter', uint64_t, 16), ('system_clock_frequency', uint64_t, 24)])
hsa_amd_clock_counters_t: TypeAlias = struct_hsa_amd_clock_counters_s
enum_hsa_amd_agent_info_s: dict[int, str] = {(HSA_AMD_AGENT_INFO_CHIP_ID:=40960): 'HSA_AMD_AGENT_INFO_CHIP_ID', (HSA_AMD_AGENT_INFO_CACHELINE_SIZE:=40961): 'HSA_AMD_AGENT_INFO_CACHELINE_SIZE', (HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT:=40962): 'HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT', (HSA_AMD_AGENT_INFO_MAX_CLOCK_FREQUENCY:=40963): 'HSA_AMD_AGENT_INFO_MAX_CLOCK_FREQUENCY', (HSA_AMD_AGENT_INFO_DRIVER_NODE_ID:=40964): 'HSA_AMD_AGENT_INFO_DRIVER_NODE_ID', (HSA_AMD_AGENT_INFO_MAX_ADDRESS_WATCH_POINTS:=40965): 'HSA_AMD_AGENT_INFO_MAX_ADDRESS_WATCH_POINTS', (HSA_AMD_AGENT_INFO_BDFID:=40966): 'HSA_AMD_AGENT_INFO_BDFID', (HSA_AMD_AGENT_INFO_MEMORY_WIDTH:=40967): 'HSA_AMD_AGENT_INFO_MEMORY_WIDTH', (HSA_AMD_AGENT_INFO_MEMORY_MAX_FREQUENCY:=40968): 'HSA_AMD_AGENT_INFO_MEMORY_MAX_FREQUENCY', (HSA_AMD_AGENT_INFO_PRODUCT_NAME:=40969): 'HSA_AMD_AGENT_INFO_PRODUCT_NAME', (HSA_AMD_AGENT_INFO_MAX_WAVES_PER_CU:=40970): 'HSA_AMD_AGENT_INFO_MAX_WAVES_PER_CU', (HSA_AMD_AGENT_INFO_NUM_SIMDS_PER_CU:=40971): 'HSA_AMD_AGENT_INFO_NUM_SIMDS_PER_CU', (HSA_AMD_AGENT_INFO_NUM_SHADER_ENGINES:=40972): 'HSA_AMD_AGENT_INFO_NUM_SHADER_ENGINES', (HSA_AMD_AGENT_INFO_NUM_SHADER_ARRAYS_PER_SE:=40973): 'HSA_AMD_AGENT_INFO_NUM_SHADER_ARRAYS_PER_SE', (HSA_AMD_AGENT_INFO_HDP_FLUSH:=40974): 'HSA_AMD_AGENT_INFO_HDP_FLUSH', (HSA_AMD_AGENT_INFO_DOMAIN:=40975): 'HSA_AMD_AGENT_INFO_DOMAIN', (HSA_AMD_AGENT_INFO_COOPERATIVE_QUEUES:=40976): 'HSA_AMD_AGENT_INFO_COOPERATIVE_QUEUES', (HSA_AMD_AGENT_INFO_UUID:=40977): 'HSA_AMD_AGENT_INFO_UUID', (HSA_AMD_AGENT_INFO_ASIC_REVISION:=40978): 'HSA_AMD_AGENT_INFO_ASIC_REVISION', (HSA_AMD_AGENT_INFO_SVM_DIRECT_HOST_ACCESS:=40979): 'HSA_AMD_AGENT_INFO_SVM_DIRECT_HOST_ACCESS', (HSA_AMD_AGENT_INFO_COOPERATIVE_COMPUTE_UNIT_COUNT:=40980): 'HSA_AMD_AGENT_INFO_COOPERATIVE_COMPUTE_UNIT_COUNT', (HSA_AMD_AGENT_INFO_MEMORY_AVAIL:=40981): 'HSA_AMD_AGENT_INFO_MEMORY_AVAIL', (HSA_AMD_AGENT_INFO_TIMESTAMP_FREQUENCY:=40982): 'HSA_AMD_AGENT_INFO_TIMESTAMP_FREQUENCY', (HSA_AMD_AGENT_INFO_ASIC_FAMILY_ID:=41223): 'HSA_AMD_AGENT_INFO_ASIC_FAMILY_ID', (HSA_AMD_AGENT_INFO_UCODE_VERSION:=41224): 'HSA_AMD_AGENT_INFO_UCODE_VERSION', (HSA_AMD_AGENT_INFO_SDMA_UCODE_VERSION:=41225): 'HSA_AMD_AGENT_INFO_SDMA_UCODE_VERSION', (HSA_AMD_AGENT_INFO_NUM_SDMA_ENG:=41226): 'HSA_AMD_AGENT_INFO_NUM_SDMA_ENG', (HSA_AMD_AGENT_INFO_NUM_SDMA_XGMI_ENG:=41227): 'HSA_AMD_AGENT_INFO_NUM_SDMA_XGMI_ENG', (HSA_AMD_AGENT_INFO_IOMMU_SUPPORT:=41232): 'HSA_AMD_AGENT_INFO_IOMMU_SUPPORT', (HSA_AMD_AGENT_INFO_NUM_XCC:=41233): 'HSA_AMD_AGENT_INFO_NUM_XCC', (HSA_AMD_AGENT_INFO_DRIVER_UID:=41234): 'HSA_AMD_AGENT_INFO_DRIVER_UID', (HSA_AMD_AGENT_INFO_NEAREST_CPU:=41235): 'HSA_AMD_AGENT_INFO_NEAREST_CPU', (HSA_AMD_AGENT_INFO_MEMORY_PROPERTIES:=41236): 'HSA_AMD_AGENT_INFO_MEMORY_PROPERTIES', (HSA_AMD_AGENT_INFO_AQL_EXTENSIONS:=41237): 'HSA_AMD_AGENT_INFO_AQL_EXTENSIONS', (HSA_AMD_AGENT_INFO_SCRATCH_LIMIT_MAX:=41238): 'HSA_AMD_AGENT_INFO_SCRATCH_LIMIT_MAX', (HSA_AMD_AGENT_INFO_SCRATCH_LIMIT_CURRENT:=41239): 'HSA_AMD_AGENT_INFO_SCRATCH_LIMIT_CURRENT', (HSA_AMD_AGENT_INFO_CLOCK_COUNTERS:=41240): 'HSA_AMD_AGENT_INFO_CLOCK_COUNTERS'}
hsa_amd_agent_info_t: TypeAlias = ctypes.c_uint32
enum_hsa_amd_agent_memory_properties_s: dict[int, str] = {(HSA_AMD_MEMORY_PROPERTY_AGENT_IS_APU:=1): 'HSA_AMD_MEMORY_PROPERTY_AGENT_IS_APU'}
hsa_amd_agent_memory_properties_t: TypeAlias = ctypes.c_uint32
enum_hsa_amd_sdma_engine_id: dict[int, str] = {(HSA_AMD_SDMA_ENGINE_0:=1): 'HSA_AMD_SDMA_ENGINE_0', (HSA_AMD_SDMA_ENGINE_1:=2): 'HSA_AMD_SDMA_ENGINE_1', (HSA_AMD_SDMA_ENGINE_2:=4): 'HSA_AMD_SDMA_ENGINE_2', (HSA_AMD_SDMA_ENGINE_3:=8): 'HSA_AMD_SDMA_ENGINE_3', (HSA_AMD_SDMA_ENGINE_4:=16): 'HSA_AMD_SDMA_ENGINE_4', (HSA_AMD_SDMA_ENGINE_5:=32): 'HSA_AMD_SDMA_ENGINE_5', (HSA_AMD_SDMA_ENGINE_6:=64): 'HSA_AMD_SDMA_ENGINE_6', (HSA_AMD_SDMA_ENGINE_7:=128): 'HSA_AMD_SDMA_ENGINE_7', (HSA_AMD_SDMA_ENGINE_8:=256): 'HSA_AMD_SDMA_ENGINE_8', (HSA_AMD_SDMA_ENGINE_9:=512): 'HSA_AMD_SDMA_ENGINE_9', (HSA_AMD_SDMA_ENGINE_10:=1024): 'HSA_AMD_SDMA_ENGINE_10', (HSA_AMD_SDMA_ENGINE_11:=2048): 'HSA_AMD_SDMA_ENGINE_11', (HSA_AMD_SDMA_ENGINE_12:=4096): 'HSA_AMD_SDMA_ENGINE_12', (HSA_AMD_SDMA_ENGINE_13:=8192): 'HSA_AMD_SDMA_ENGINE_13', (HSA_AMD_SDMA_ENGINE_14:=16384): 'HSA_AMD_SDMA_ENGINE_14', (HSA_AMD_SDMA_ENGINE_15:=32768): 'HSA_AMD_SDMA_ENGINE_15'}
hsa_amd_sdma_engine_id_t: TypeAlias = ctypes.c_uint32
@c.record
class struct_hsa_amd_hdp_flush_s(c.Struct):
  SIZE = 16
  HDP_MEM_FLUSH_CNTL: c.POINTER[ctypes.c_uint32]
  HDP_REG_FLUSH_CNTL: c.POINTER[ctypes.c_uint32]
struct_hsa_amd_hdp_flush_s.register_fields([('HDP_MEM_FLUSH_CNTL', c.POINTER[uint32_t], 0), ('HDP_REG_FLUSH_CNTL', c.POINTER[uint32_t], 8)])
hsa_amd_hdp_flush_t: TypeAlias = struct_hsa_amd_hdp_flush_s
enum_hsa_amd_region_info_s: dict[int, str] = {(HSA_AMD_REGION_INFO_HOST_ACCESSIBLE:=40960): 'HSA_AMD_REGION_INFO_HOST_ACCESSIBLE', (HSA_AMD_REGION_INFO_BASE:=40961): 'HSA_AMD_REGION_INFO_BASE', (HSA_AMD_REGION_INFO_BUS_WIDTH:=40962): 'HSA_AMD_REGION_INFO_BUS_WIDTH', (HSA_AMD_REGION_INFO_MAX_CLOCK_FREQUENCY:=40963): 'HSA_AMD_REGION_INFO_MAX_CLOCK_FREQUENCY'}
hsa_amd_region_info_t: TypeAlias = ctypes.c_uint32
enum_hsa_amd_coherency_type_s: dict[int, str] = {(HSA_AMD_COHERENCY_TYPE_COHERENT:=0): 'HSA_AMD_COHERENCY_TYPE_COHERENT', (HSA_AMD_COHERENCY_TYPE_NONCOHERENT:=1): 'HSA_AMD_COHERENCY_TYPE_NONCOHERENT'}
hsa_amd_coherency_type_t: TypeAlias = ctypes.c_uint32
enum_hsa_amd_dma_buf_mapping_type_s: dict[int, str] = {(HSA_AMD_DMABUF_MAPPING_TYPE_NONE:=0): 'HSA_AMD_DMABUF_MAPPING_TYPE_NONE', (HSA_AMD_DMABUF_MAPPING_TYPE_PCIE:=1): 'HSA_AMD_DMABUF_MAPPING_TYPE_PCIE'}
hsa_amd_dma_buf_mapping_type_t: TypeAlias = ctypes.c_uint32
@dll.bind(ctypes.c_uint32, hsa_agent_t, c.POINTER[hsa_amd_coherency_type_t])
def hsa_amd_coherency_get_type(agent:hsa_agent_t, type:c.POINTER[hsa_amd_coherency_type_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hsa_agent_t, hsa_amd_coherency_type_t)
def hsa_amd_coherency_set_type(agent:hsa_agent_t, type:hsa_amd_coherency_type_t) -> ctypes.c_uint32: ...
@c.record
class struct_hsa_amd_profiling_dispatch_time_s(c.Struct):
  SIZE = 16
  start: int
  end: int
struct_hsa_amd_profiling_dispatch_time_s.register_fields([('start', uint64_t, 0), ('end', uint64_t, 8)])
hsa_amd_profiling_dispatch_time_t: TypeAlias = struct_hsa_amd_profiling_dispatch_time_s
@c.record
class struct_hsa_amd_profiling_async_copy_time_s(c.Struct):
  SIZE = 16
  start: int
  end: int
struct_hsa_amd_profiling_async_copy_time_s.register_fields([('start', uint64_t, 0), ('end', uint64_t, 8)])
hsa_amd_profiling_async_copy_time_t: TypeAlias = struct_hsa_amd_profiling_async_copy_time_s
@dll.bind(ctypes.c_uint32, c.POINTER[hsa_queue_t], ctypes.c_int32)
def hsa_amd_profiling_set_profiler_enabled(queue:c.POINTER[hsa_queue_t], enable:int) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_bool)
def hsa_amd_profiling_async_copy_enable(enable:bool) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hsa_agent_t, hsa_signal_t, c.POINTER[hsa_amd_profiling_dispatch_time_t])
def hsa_amd_profiling_get_dispatch_time(agent:hsa_agent_t, signal:hsa_signal_t, time:c.POINTER[hsa_amd_profiling_dispatch_time_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hsa_signal_t, c.POINTER[hsa_amd_profiling_async_copy_time_t])
def hsa_amd_profiling_get_async_copy_time(signal:hsa_signal_t, time:c.POINTER[hsa_amd_profiling_async_copy_time_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hsa_agent_t, uint64_t, c.POINTER[uint64_t])
def hsa_amd_profiling_convert_tick_to_system_domain(agent:hsa_agent_t, agent_tick:uint64_t, system_tick:c.POINTER[uint64_t]) -> ctypes.c_uint32: ...
hsa_amd_signal_attribute_t: dict[int, str] = {(HSA_AMD_SIGNAL_AMD_GPU_ONLY:=1): 'HSA_AMD_SIGNAL_AMD_GPU_ONLY', (HSA_AMD_SIGNAL_IPC:=2): 'HSA_AMD_SIGNAL_IPC'}
@dll.bind(ctypes.c_uint32, hsa_signal_value_t, uint32_t, c.POINTER[hsa_agent_t], uint64_t, c.POINTER[hsa_signal_t])
def hsa_amd_signal_create(initial_value:hsa_signal_value_t, num_consumers:uint32_t, consumers:c.POINTER[hsa_agent_t], attributes:uint64_t, signal:c.POINTER[hsa_signal_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hsa_signal_t, c.POINTER[c.POINTER[hsa_signal_value_t]])
def hsa_amd_signal_value_pointer(signal:hsa_signal_t, value_ptr:c.POINTER[c.POINTER[hsa_signal_value_t]]) -> ctypes.c_uint32: ...
hsa_amd_signal_handler: TypeAlias = c.CFUNCTYPE[ctypes.c_bool, [ctypes.c_int64, ctypes.c_void_p]]
@dll.bind(ctypes.c_uint32, hsa_signal_t, ctypes.c_uint32, hsa_signal_value_t, hsa_amd_signal_handler, ctypes.c_void_p)
def hsa_amd_signal_async_handler(signal:hsa_signal_t, cond:ctypes.c_uint32, value:hsa_signal_value_t, handler:hsa_amd_signal_handler, arg:ctypes.c_void_p) -> ctypes.c_uint32: ...
@dll.bind(uint32_t, uint32_t, c.POINTER[hsa_signal_t], c.POINTER[ctypes.c_uint32], c.POINTER[hsa_signal_value_t], uint64_t, ctypes.c_uint32, c.POINTER[hsa_signal_value_t])
def hsa_amd_signal_wait_all(signal_count:uint32_t, signals:c.POINTER[hsa_signal_t], conds:c.POINTER[ctypes.c_uint32], values:c.POINTER[hsa_signal_value_t], timeout_hint:uint64_t, wait_hint:ctypes.c_uint32, satisfying_values:c.POINTER[hsa_signal_value_t]) -> uint32_t: ...
@dll.bind(uint32_t, uint32_t, c.POINTER[hsa_signal_t], c.POINTER[ctypes.c_uint32], c.POINTER[hsa_signal_value_t], uint64_t, ctypes.c_uint32, c.POINTER[hsa_signal_value_t])
def hsa_amd_signal_wait_any(signal_count:uint32_t, signals:c.POINTER[hsa_signal_t], conds:c.POINTER[ctypes.c_uint32], values:c.POINTER[hsa_signal_value_t], timeout_hint:uint64_t, wait_hint:ctypes.c_uint32, satisfying_value:c.POINTER[hsa_signal_value_t]) -> uint32_t: ...
@dll.bind(ctypes.c_uint32, c.CFUNCTYPE[None, [ctypes.c_void_p]], ctypes.c_void_p)
def hsa_amd_async_function(callback:c.CFUNCTYPE[None, [ctypes.c_void_p]], arg:ctypes.c_void_p) -> ctypes.c_uint32: ...
@c.record
class struct_hsa_amd_image_descriptor_s(c.Struct):
  SIZE = 12
  version: int
  deviceID: int
  data: c.Array[ctypes.c_uint32, Literal[1]]
struct_hsa_amd_image_descriptor_s.register_fields([('version', uint32_t, 0), ('deviceID', uint32_t, 4), ('data', c.Array[uint32_t, Literal[1]], 8)])
hsa_amd_image_descriptor_t: TypeAlias = struct_hsa_amd_image_descriptor_s
@c.record
class struct_hsa_ext_image_descriptor_s(c.Struct):
  SIZE = 48
  geometry: int
  width: int
  height: int
  depth: int
  array_size: int
  format: struct_hsa_ext_image_format_s
hsa_ext_image_descriptor_t: TypeAlias = struct_hsa_ext_image_descriptor_s
hsa_ext_image_geometry_t: dict[int, str] = {(HSA_EXT_IMAGE_GEOMETRY_1D:=0): 'HSA_EXT_IMAGE_GEOMETRY_1D', (HSA_EXT_IMAGE_GEOMETRY_2D:=1): 'HSA_EXT_IMAGE_GEOMETRY_2D', (HSA_EXT_IMAGE_GEOMETRY_3D:=2): 'HSA_EXT_IMAGE_GEOMETRY_3D', (HSA_EXT_IMAGE_GEOMETRY_1DA:=3): 'HSA_EXT_IMAGE_GEOMETRY_1DA', (HSA_EXT_IMAGE_GEOMETRY_2DA:=4): 'HSA_EXT_IMAGE_GEOMETRY_2DA', (HSA_EXT_IMAGE_GEOMETRY_1DB:=5): 'HSA_EXT_IMAGE_GEOMETRY_1DB', (HSA_EXT_IMAGE_GEOMETRY_2DDEPTH:=6): 'HSA_EXT_IMAGE_GEOMETRY_2DDEPTH', (HSA_EXT_IMAGE_GEOMETRY_2DADEPTH:=7): 'HSA_EXT_IMAGE_GEOMETRY_2DADEPTH'}
@c.record
class struct_hsa_ext_image_format_s(c.Struct):
  SIZE = 8
  channel_type: int
  channel_order: int
hsa_ext_image_format_t: TypeAlias = struct_hsa_ext_image_format_s
hsa_ext_image_channel_type32_t: TypeAlias = ctypes.c_uint32
hsa_ext_image_channel_order32_t: TypeAlias = ctypes.c_uint32
struct_hsa_ext_image_format_s.register_fields([('channel_type', hsa_ext_image_channel_type32_t, 0), ('channel_order', hsa_ext_image_channel_order32_t, 4)])
struct_hsa_ext_image_descriptor_s.register_fields([('geometry', ctypes.c_uint32, 0), ('width', size_t, 8), ('height', size_t, 16), ('depth', size_t, 24), ('array_size', size_t, 32), ('format', hsa_ext_image_format_t, 40)])
@c.record
class struct_hsa_ext_image_s(c.Struct):
  SIZE = 8
  handle: int
hsa_ext_image_t: TypeAlias = struct_hsa_ext_image_s
struct_hsa_ext_image_s.register_fields([('handle', uint64_t, 0)])
@dll.bind(ctypes.c_uint32, hsa_agent_t, c.POINTER[hsa_ext_image_descriptor_t], c.POINTER[hsa_amd_image_descriptor_t], ctypes.c_void_p, ctypes.c_uint32, c.POINTER[hsa_ext_image_t])
def hsa_amd_image_create(agent:hsa_agent_t, image_descriptor:c.POINTER[hsa_ext_image_descriptor_t], image_layout:c.POINTER[hsa_amd_image_descriptor_t], image_data:ctypes.c_void_p, access_permission:ctypes.c_uint32, image:c.POINTER[hsa_ext_image_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hsa_agent_t, ctypes.c_uint32, ctypes.c_void_p)
def hsa_amd_image_get_info_max_dim(agent:hsa_agent_t, attribute:ctypes.c_uint32, value:ctypes.c_void_p) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hsa_queue_t], uint32_t, c.POINTER[uint32_t])
def hsa_amd_queue_cu_set_mask(queue:c.POINTER[hsa_queue_t], num_cu_mask_count:uint32_t, cu_mask:c.POINTER[uint32_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hsa_queue_t], uint32_t, c.POINTER[uint32_t])
def hsa_amd_queue_cu_get_mask(queue:c.POINTER[hsa_queue_t], num_cu_mask_count:uint32_t, cu_mask:c.POINTER[uint32_t]) -> ctypes.c_uint32: ...
hsa_amd_segment_t: dict[int, str] = {(HSA_AMD_SEGMENT_GLOBAL:=0): 'HSA_AMD_SEGMENT_GLOBAL', (HSA_AMD_SEGMENT_READONLY:=1): 'HSA_AMD_SEGMENT_READONLY', (HSA_AMD_SEGMENT_PRIVATE:=2): 'HSA_AMD_SEGMENT_PRIVATE', (HSA_AMD_SEGMENT_GROUP:=3): 'HSA_AMD_SEGMENT_GROUP'}
@c.record
class struct_hsa_amd_memory_pool_s(c.Struct):
  SIZE = 8
  handle: int
struct_hsa_amd_memory_pool_s.register_fields([('handle', uint64_t, 0)])
hsa_amd_memory_pool_t: TypeAlias = struct_hsa_amd_memory_pool_s
enum_hsa_amd_memory_pool_global_flag_s: dict[int, str] = {(HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT:=1): 'HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT', (HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED:=2): 'HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED', (HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED:=4): 'HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED', (HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_EXTENDED_SCOPE_FINE_GRAINED:=8): 'HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_EXTENDED_SCOPE_FINE_GRAINED'}
hsa_amd_memory_pool_global_flag_t: TypeAlias = ctypes.c_uint32
enum_hsa_amd_memory_pool_location_s: dict[int, str] = {(HSA_AMD_MEMORY_POOL_LOCATION_CPU:=0): 'HSA_AMD_MEMORY_POOL_LOCATION_CPU', (HSA_AMD_MEMORY_POOL_LOCATION_GPU:=1): 'HSA_AMD_MEMORY_POOL_LOCATION_GPU'}
hsa_amd_memory_pool_location_t: TypeAlias = ctypes.c_uint32
hsa_amd_memory_pool_info_t: dict[int, str] = {(HSA_AMD_MEMORY_POOL_INFO_SEGMENT:=0): 'HSA_AMD_MEMORY_POOL_INFO_SEGMENT', (HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS:=1): 'HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS', (HSA_AMD_MEMORY_POOL_INFO_SIZE:=2): 'HSA_AMD_MEMORY_POOL_INFO_SIZE', (HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED:=5): 'HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED', (HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE:=6): 'HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE', (HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALIGNMENT:=7): 'HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALIGNMENT', (HSA_AMD_MEMORY_POOL_INFO_ACCESSIBLE_BY_ALL:=15): 'HSA_AMD_MEMORY_POOL_INFO_ACCESSIBLE_BY_ALL', (HSA_AMD_MEMORY_POOL_INFO_ALLOC_MAX_SIZE:=16): 'HSA_AMD_MEMORY_POOL_INFO_ALLOC_MAX_SIZE', (HSA_AMD_MEMORY_POOL_INFO_LOCATION:=17): 'HSA_AMD_MEMORY_POOL_INFO_LOCATION', (HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_REC_GRANULE:=18): 'HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_REC_GRANULE'}
enum_hsa_amd_memory_pool_flag_s: dict[int, str] = {(HSA_AMD_MEMORY_POOL_STANDARD_FLAG:=0): 'HSA_AMD_MEMORY_POOL_STANDARD_FLAG', (HSA_AMD_MEMORY_POOL_PCIE_FLAG:=1): 'HSA_AMD_MEMORY_POOL_PCIE_FLAG', (HSA_AMD_MEMORY_POOL_CONTIGUOUS_FLAG:=2): 'HSA_AMD_MEMORY_POOL_CONTIGUOUS_FLAG', (HSA_AMD_MEMORY_POOL_EXECUTABLE_FLAG:=4): 'HSA_AMD_MEMORY_POOL_EXECUTABLE_FLAG', (HSA_AMD_MEMORY_POOL_UNCACHED_FLAG:=8): 'HSA_AMD_MEMORY_POOL_UNCACHED_FLAG'}
hsa_amd_memory_pool_flag_t: TypeAlias = ctypes.c_uint32
@dll.bind(ctypes.c_uint32, hsa_amd_memory_pool_t, ctypes.c_uint32, ctypes.c_void_p)
def hsa_amd_memory_pool_get_info(memory_pool:hsa_amd_memory_pool_t, attribute:ctypes.c_uint32, value:ctypes.c_void_p) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hsa_agent_t, c.CFUNCTYPE[ctypes.c_uint32, [hsa_amd_memory_pool_t, ctypes.c_void_p]], ctypes.c_void_p)
def hsa_amd_agent_iterate_memory_pools(agent:hsa_agent_t, callback:c.CFUNCTYPE[ctypes.c_uint32, [hsa_amd_memory_pool_t, ctypes.c_void_p]], data:ctypes.c_void_p) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hsa_amd_memory_pool_t, size_t, uint32_t, c.POINTER[ctypes.c_void_p])
def hsa_amd_memory_pool_allocate(memory_pool:hsa_amd_memory_pool_t, size:size_t, flags:uint32_t, ptr:c.POINTER[ctypes.c_void_p]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p)
def hsa_amd_memory_pool_free(ptr:ctypes.c_void_p) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, hsa_agent_t, ctypes.c_void_p, hsa_agent_t, size_t, uint32_t, c.POINTER[hsa_signal_t], hsa_signal_t)
def hsa_amd_memory_async_copy(dst:ctypes.c_void_p, dst_agent:hsa_agent_t, src:ctypes.c_void_p, src_agent:hsa_agent_t, size:size_t, num_dep_signals:uint32_t, dep_signals:c.POINTER[hsa_signal_t], completion_signal:hsa_signal_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, hsa_agent_t, ctypes.c_void_p, hsa_agent_t, size_t, uint32_t, c.POINTER[hsa_signal_t], hsa_signal_t, hsa_amd_sdma_engine_id_t, ctypes.c_bool)
def hsa_amd_memory_async_copy_on_engine(dst:ctypes.c_void_p, dst_agent:hsa_agent_t, src:ctypes.c_void_p, src_agent:hsa_agent_t, size:size_t, num_dep_signals:uint32_t, dep_signals:c.POINTER[hsa_signal_t], completion_signal:hsa_signal_t, engine_id:hsa_amd_sdma_engine_id_t, force_copy_on_sdma:bool) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hsa_agent_t, hsa_agent_t, c.POINTER[uint32_t])
def hsa_amd_memory_copy_engine_status(dst_agent:hsa_agent_t, src_agent:hsa_agent_t, engine_ids_mask:c.POINTER[uint32_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hsa_agent_t, hsa_agent_t, c.POINTER[uint32_t])
def hsa_amd_memory_get_preferred_copy_engine(dst_agent:hsa_agent_t, src_agent:hsa_agent_t, recommended_ids_mask:c.POINTER[uint32_t]) -> ctypes.c_uint32: ...
@c.record
class struct_hsa_pitched_ptr_s(c.Struct):
  SIZE = 24
  base: ctypes.c_void_p
  pitch: int
  slice: int
struct_hsa_pitched_ptr_s.register_fields([('base', ctypes.c_void_p, 0), ('pitch', size_t, 8), ('slice', size_t, 16)])
hsa_pitched_ptr_t: TypeAlias = struct_hsa_pitched_ptr_s
hsa_amd_copy_direction_t: dict[int, str] = {(hsaHostToHost:=0): 'hsaHostToHost', (hsaHostToDevice:=1): 'hsaHostToDevice', (hsaDeviceToHost:=2): 'hsaDeviceToHost', (hsaDeviceToDevice:=3): 'hsaDeviceToDevice'}
@dll.bind(ctypes.c_uint32, c.POINTER[hsa_pitched_ptr_t], c.POINTER[hsa_dim3_t], c.POINTER[hsa_pitched_ptr_t], c.POINTER[hsa_dim3_t], c.POINTER[hsa_dim3_t], hsa_agent_t, ctypes.c_uint32, uint32_t, c.POINTER[hsa_signal_t], hsa_signal_t)
def hsa_amd_memory_async_copy_rect(dst:c.POINTER[hsa_pitched_ptr_t], dst_offset:c.POINTER[hsa_dim3_t], src:c.POINTER[hsa_pitched_ptr_t], src_offset:c.POINTER[hsa_dim3_t], range:c.POINTER[hsa_dim3_t], copy_agent:hsa_agent_t, dir:ctypes.c_uint32, num_dep_signals:uint32_t, dep_signals:c.POINTER[hsa_signal_t], completion_signal:hsa_signal_t) -> ctypes.c_uint32: ...
hsa_amd_memory_pool_access_t: dict[int, str] = {(HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED:=0): 'HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED', (HSA_AMD_MEMORY_POOL_ACCESS_ALLOWED_BY_DEFAULT:=1): 'HSA_AMD_MEMORY_POOL_ACCESS_ALLOWED_BY_DEFAULT', (HSA_AMD_MEMORY_POOL_ACCESS_DISALLOWED_BY_DEFAULT:=2): 'HSA_AMD_MEMORY_POOL_ACCESS_DISALLOWED_BY_DEFAULT'}
hsa_amd_link_info_type_t: dict[int, str] = {(HSA_AMD_LINK_INFO_TYPE_HYPERTRANSPORT:=0): 'HSA_AMD_LINK_INFO_TYPE_HYPERTRANSPORT', (HSA_AMD_LINK_INFO_TYPE_QPI:=1): 'HSA_AMD_LINK_INFO_TYPE_QPI', (HSA_AMD_LINK_INFO_TYPE_PCIE:=2): 'HSA_AMD_LINK_INFO_TYPE_PCIE', (HSA_AMD_LINK_INFO_TYPE_INFINBAND:=3): 'HSA_AMD_LINK_INFO_TYPE_INFINBAND', (HSA_AMD_LINK_INFO_TYPE_XGMI:=4): 'HSA_AMD_LINK_INFO_TYPE_XGMI'}
@c.record
class struct_hsa_amd_memory_pool_link_info_s(c.Struct):
  SIZE = 28
  min_latency: int
  max_latency: int
  min_bandwidth: int
  max_bandwidth: int
  atomic_support_32bit: bool
  atomic_support_64bit: bool
  coherent_support: bool
  link_type: int
  numa_distance: int
struct_hsa_amd_memory_pool_link_info_s.register_fields([('min_latency', uint32_t, 0), ('max_latency', uint32_t, 4), ('min_bandwidth', uint32_t, 8), ('max_bandwidth', uint32_t, 12), ('atomic_support_32bit', ctypes.c_bool, 16), ('atomic_support_64bit', ctypes.c_bool, 17), ('coherent_support', ctypes.c_bool, 18), ('link_type', ctypes.c_uint32, 20), ('numa_distance', uint32_t, 24)])
hsa_amd_memory_pool_link_info_t: TypeAlias = struct_hsa_amd_memory_pool_link_info_s
hsa_amd_agent_memory_pool_info_t: dict[int, str] = {(HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS:=0): 'HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS', (HSA_AMD_AGENT_MEMORY_POOL_INFO_NUM_LINK_HOPS:=1): 'HSA_AMD_AGENT_MEMORY_POOL_INFO_NUM_LINK_HOPS', (HSA_AMD_AGENT_MEMORY_POOL_INFO_LINK_INFO:=2): 'HSA_AMD_AGENT_MEMORY_POOL_INFO_LINK_INFO'}
@dll.bind(ctypes.c_uint32, hsa_agent_t, hsa_amd_memory_pool_t, ctypes.c_uint32, ctypes.c_void_p)
def hsa_amd_agent_memory_pool_get_info(agent:hsa_agent_t, memory_pool:hsa_amd_memory_pool_t, attribute:ctypes.c_uint32, value:ctypes.c_void_p) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, uint32_t, c.POINTER[hsa_agent_t], c.POINTER[uint32_t], ctypes.c_void_p)
def hsa_amd_agents_allow_access(num_agents:uint32_t, agents:c.POINTER[hsa_agent_t], flags:c.POINTER[uint32_t], ptr:ctypes.c_void_p) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hsa_amd_memory_pool_t, hsa_amd_memory_pool_t, c.POINTER[ctypes.c_bool])
def hsa_amd_memory_pool_can_migrate(src_memory_pool:hsa_amd_memory_pool_t, dst_memory_pool:hsa_amd_memory_pool_t, result:c.POINTER[ctypes.c_bool]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, hsa_amd_memory_pool_t, uint32_t)
def hsa_amd_memory_migrate(ptr:ctypes.c_void_p, memory_pool:hsa_amd_memory_pool_t, flags:uint32_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, size_t, c.POINTER[hsa_agent_t], ctypes.c_int32, c.POINTER[ctypes.c_void_p])
def hsa_amd_memory_lock(host_ptr:ctypes.c_void_p, size:size_t, agents:c.POINTER[hsa_agent_t], num_agent:int, agent_ptr:c.POINTER[ctypes.c_void_p]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, size_t, c.POINTER[hsa_agent_t], ctypes.c_int32, hsa_amd_memory_pool_t, uint32_t, c.POINTER[ctypes.c_void_p])
def hsa_amd_memory_lock_to_pool(host_ptr:ctypes.c_void_p, size:size_t, agents:c.POINTER[hsa_agent_t], num_agent:int, pool:hsa_amd_memory_pool_t, flags:uint32_t, agent_ptr:c.POINTER[ctypes.c_void_p]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p)
def hsa_amd_memory_unlock(host_ptr:ctypes.c_void_p) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, uint32_t, size_t)
def hsa_amd_memory_fill(ptr:ctypes.c_void_p, value:uint32_t, count:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, uint32_t, c.POINTER[hsa_agent_t], ctypes.c_int32, uint32_t, c.POINTER[size_t], c.POINTER[ctypes.c_void_p], c.POINTER[size_t], c.POINTER[ctypes.c_void_p])
def hsa_amd_interop_map_buffer(num_agents:uint32_t, agents:c.POINTER[hsa_agent_t], interop_handle:int, flags:uint32_t, size:c.POINTER[size_t], ptr:c.POINTER[ctypes.c_void_p], metadata_size:c.POINTER[size_t], metadata:c.POINTER[ctypes.c_void_p]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p)
def hsa_amd_interop_unmap_buffer(ptr:ctypes.c_void_p) -> ctypes.c_uint32: ...
hsa_amd_pointer_type_t: dict[int, str] = {(HSA_EXT_POINTER_TYPE_UNKNOWN:=0): 'HSA_EXT_POINTER_TYPE_UNKNOWN', (HSA_EXT_POINTER_TYPE_HSA:=1): 'HSA_EXT_POINTER_TYPE_HSA', (HSA_EXT_POINTER_TYPE_LOCKED:=2): 'HSA_EXT_POINTER_TYPE_LOCKED', (HSA_EXT_POINTER_TYPE_GRAPHICS:=3): 'HSA_EXT_POINTER_TYPE_GRAPHICS', (HSA_EXT_POINTER_TYPE_IPC:=4): 'HSA_EXT_POINTER_TYPE_IPC', (HSA_EXT_POINTER_TYPE_RESERVED_ADDR:=5): 'HSA_EXT_POINTER_TYPE_RESERVED_ADDR', (HSA_EXT_POINTER_TYPE_HSA_VMEM:=6): 'HSA_EXT_POINTER_TYPE_HSA_VMEM'}
@c.record
class struct_hsa_amd_pointer_info_s(c.Struct):
  SIZE = 56
  size: int
  type: int
  agentBaseAddress: ctypes.c_void_p
  hostBaseAddress: ctypes.c_void_p
  sizeInBytes: int
  userData: ctypes.c_void_p
  agentOwner: struct_hsa_agent_s
  global_flags: int
  registered: bool
struct_hsa_amd_pointer_info_s.register_fields([('size', uint32_t, 0), ('type', ctypes.c_uint32, 4), ('agentBaseAddress', ctypes.c_void_p, 8), ('hostBaseAddress', ctypes.c_void_p, 16), ('sizeInBytes', size_t, 24), ('userData', ctypes.c_void_p, 32), ('agentOwner', hsa_agent_t, 40), ('global_flags', uint32_t, 48), ('registered', ctypes.c_bool, 52)])
hsa_amd_pointer_info_t: TypeAlias = struct_hsa_amd_pointer_info_s
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, c.POINTER[hsa_amd_pointer_info_t], c.CFUNCTYPE[ctypes.c_void_p, [size_t]], c.POINTER[uint32_t], c.POINTER[c.POINTER[hsa_agent_t]])
def hsa_amd_pointer_info(ptr:ctypes.c_void_p, info:c.POINTER[hsa_amd_pointer_info_t], alloc:c.CFUNCTYPE[ctypes.c_void_p, [size_t]], num_agents_accessible:c.POINTER[uint32_t], accessible:c.POINTER[c.POINTER[hsa_agent_t]]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p)
def hsa_amd_pointer_info_set_userdata(ptr:ctypes.c_void_p, userdata:ctypes.c_void_p) -> ctypes.c_uint32: ...
@c.record
class struct_hsa_amd_ipc_memory_s(c.Struct):
  SIZE = 32
  handle: c.Array[ctypes.c_uint32, Literal[8]]
struct_hsa_amd_ipc_memory_s.register_fields([('handle', c.Array[uint32_t, Literal[8]], 0)])
hsa_amd_ipc_memory_t: TypeAlias = struct_hsa_amd_ipc_memory_s
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, size_t, c.POINTER[hsa_amd_ipc_memory_t])
def hsa_amd_ipc_memory_create(ptr:ctypes.c_void_p, len:size_t, handle:c.POINTER[hsa_amd_ipc_memory_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hsa_amd_ipc_memory_t], size_t, uint32_t, c.POINTER[hsa_agent_t], c.POINTER[ctypes.c_void_p])
def hsa_amd_ipc_memory_attach(handle:c.POINTER[hsa_amd_ipc_memory_t], len:size_t, num_agents:uint32_t, mapping_agents:c.POINTER[hsa_agent_t], mapped_ptr:c.POINTER[ctypes.c_void_p]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p)
def hsa_amd_ipc_memory_detach(mapped_ptr:ctypes.c_void_p) -> ctypes.c_uint32: ...
hsa_amd_ipc_signal_t: TypeAlias = struct_hsa_amd_ipc_memory_s
@dll.bind(ctypes.c_uint32, hsa_signal_t, c.POINTER[hsa_amd_ipc_signal_t])
def hsa_amd_ipc_signal_create(signal:hsa_signal_t, handle:c.POINTER[hsa_amd_ipc_signal_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hsa_amd_ipc_signal_t], c.POINTER[hsa_signal_t])
def hsa_amd_ipc_signal_attach(handle:c.POINTER[hsa_amd_ipc_signal_t], signal:c.POINTER[hsa_signal_t]) -> ctypes.c_uint32: ...
enum_hsa_amd_event_type_s: dict[int, str] = {(HSA_AMD_GPU_MEMORY_FAULT_EVENT:=0): 'HSA_AMD_GPU_MEMORY_FAULT_EVENT', (HSA_AMD_GPU_HW_EXCEPTION_EVENT:=1): 'HSA_AMD_GPU_HW_EXCEPTION_EVENT', (HSA_AMD_GPU_MEMORY_ERROR_EVENT:=2): 'HSA_AMD_GPU_MEMORY_ERROR_EVENT'}
hsa_amd_event_type_t: TypeAlias = ctypes.c_uint32
hsa_amd_memory_fault_reason_t: dict[int, str] = {(HSA_AMD_MEMORY_FAULT_PAGE_NOT_PRESENT:=1): 'HSA_AMD_MEMORY_FAULT_PAGE_NOT_PRESENT', (HSA_AMD_MEMORY_FAULT_READ_ONLY:=2): 'HSA_AMD_MEMORY_FAULT_READ_ONLY', (HSA_AMD_MEMORY_FAULT_NX:=4): 'HSA_AMD_MEMORY_FAULT_NX', (HSA_AMD_MEMORY_FAULT_HOST_ONLY:=8): 'HSA_AMD_MEMORY_FAULT_HOST_ONLY', (HSA_AMD_MEMORY_FAULT_DRAMECC:=16): 'HSA_AMD_MEMORY_FAULT_DRAMECC', (HSA_AMD_MEMORY_FAULT_IMPRECISE:=32): 'HSA_AMD_MEMORY_FAULT_IMPRECISE', (HSA_AMD_MEMORY_FAULT_SRAMECC:=64): 'HSA_AMD_MEMORY_FAULT_SRAMECC', (HSA_AMD_MEMORY_FAULT_HANG:=2147483648): 'HSA_AMD_MEMORY_FAULT_HANG'}
@c.record
class struct_hsa_amd_gpu_memory_fault_info_s(c.Struct):
  SIZE = 24
  agent: struct_hsa_agent_s
  virtual_address: int
  fault_reason_mask: int
struct_hsa_amd_gpu_memory_fault_info_s.register_fields([('agent', hsa_agent_t, 0), ('virtual_address', uint64_t, 8), ('fault_reason_mask', uint32_t, 16)])
hsa_amd_gpu_memory_fault_info_t: TypeAlias = struct_hsa_amd_gpu_memory_fault_info_s
hsa_amd_memory_error_reason_t: dict[int, str] = {(HSA_AMD_MEMORY_ERROR_MEMORY_IN_USE:=1): 'HSA_AMD_MEMORY_ERROR_MEMORY_IN_USE'}
@c.record
class struct_hsa_amd_gpu_memory_error_info_s(c.Struct):
  SIZE = 24
  agent: struct_hsa_agent_s
  virtual_address: int
  error_reason_mask: int
struct_hsa_amd_gpu_memory_error_info_s.register_fields([('agent', hsa_agent_t, 0), ('virtual_address', uint64_t, 8), ('error_reason_mask', uint32_t, 16)])
hsa_amd_gpu_memory_error_info_t: TypeAlias = struct_hsa_amd_gpu_memory_error_info_s
hsa_amd_hw_exception_reset_type_t: dict[int, str] = {(HSA_AMD_HW_EXCEPTION_RESET_TYPE_OTHER:=1): 'HSA_AMD_HW_EXCEPTION_RESET_TYPE_OTHER'}
hsa_amd_hw_exception_reset_cause_t: dict[int, str] = {(HSA_AMD_HW_EXCEPTION_CAUSE_GPU_HANG:=1): 'HSA_AMD_HW_EXCEPTION_CAUSE_GPU_HANG', (HSA_AMD_HW_EXCEPTION_CAUSE_ECC:=2): 'HSA_AMD_HW_EXCEPTION_CAUSE_ECC'}
@c.record
class struct_hsa_amd_gpu_hw_exception_info_s(c.Struct):
  SIZE = 16
  agent: struct_hsa_agent_s
  reset_type: int
  reset_cause: int
struct_hsa_amd_gpu_hw_exception_info_s.register_fields([('agent', hsa_agent_t, 0), ('reset_type', ctypes.c_uint32, 8), ('reset_cause', ctypes.c_uint32, 12)])
hsa_amd_gpu_hw_exception_info_t: TypeAlias = struct_hsa_amd_gpu_hw_exception_info_s
@c.record
class struct_hsa_amd_event_s(c.Struct):
  SIZE = 32
  event_type: int
  memory_fault: struct_hsa_amd_gpu_memory_fault_info_s
  hw_exception: struct_hsa_amd_gpu_hw_exception_info_s
  memory_error: struct_hsa_amd_gpu_memory_error_info_s
struct_hsa_amd_event_s.register_fields([('event_type', hsa_amd_event_type_t, 0), ('memory_fault', hsa_amd_gpu_memory_fault_info_t, 8), ('hw_exception', hsa_amd_gpu_hw_exception_info_t, 8), ('memory_error', hsa_amd_gpu_memory_error_info_t, 8)])
hsa_amd_event_t: TypeAlias = struct_hsa_amd_event_s
hsa_amd_system_event_callback_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint32, [c.POINTER[struct_hsa_amd_event_s], ctypes.c_void_p]]
@dll.bind(ctypes.c_uint32, hsa_amd_system_event_callback_t, ctypes.c_void_p)
def hsa_amd_register_system_event_handler(callback:hsa_amd_system_event_callback_t, data:ctypes.c_void_p) -> ctypes.c_uint32: ...
enum_hsa_amd_queue_priority_s: dict[int, str] = {(HSA_AMD_QUEUE_PRIORITY_LOW:=0): 'HSA_AMD_QUEUE_PRIORITY_LOW', (HSA_AMD_QUEUE_PRIORITY_NORMAL:=1): 'HSA_AMD_QUEUE_PRIORITY_NORMAL', (HSA_AMD_QUEUE_PRIORITY_HIGH:=2): 'HSA_AMD_QUEUE_PRIORITY_HIGH'}
hsa_amd_queue_priority_t: TypeAlias = ctypes.c_uint32
@dll.bind(ctypes.c_uint32, c.POINTER[hsa_queue_t], hsa_amd_queue_priority_t)
def hsa_amd_queue_set_priority(queue:c.POINTER[hsa_queue_t], priority:hsa_amd_queue_priority_t) -> ctypes.c_uint32: ...
hsa_amd_queue_create_flag_t: dict[int, str] = {(HSA_AMD_QUEUE_CREATE_SYSTEM_MEM:=0): 'HSA_AMD_QUEUE_CREATE_SYSTEM_MEM', (HSA_AMD_QUEUE_CREATE_DEVICE_MEM_RING_BUF:=1): 'HSA_AMD_QUEUE_CREATE_DEVICE_MEM_RING_BUF', (HSA_AMD_QUEUE_CREATE_DEVICE_MEM_QUEUE_DESCRIPTOR:=2): 'HSA_AMD_QUEUE_CREATE_DEVICE_MEM_QUEUE_DESCRIPTOR'}
hsa_amd_deallocation_callback_t: TypeAlias = c.CFUNCTYPE[None, [ctypes.c_void_p, ctypes.c_void_p]]
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, hsa_amd_deallocation_callback_t, ctypes.c_void_p)
def hsa_amd_register_deallocation_callback(ptr:ctypes.c_void_p, callback:hsa_amd_deallocation_callback_t, user_data:ctypes.c_void_p) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, hsa_amd_deallocation_callback_t)
def hsa_amd_deregister_deallocation_callback(ptr:ctypes.c_void_p, callback:hsa_amd_deallocation_callback_t) -> ctypes.c_uint32: ...
enum_hsa_amd_svm_model_s: dict[int, str] = {(HSA_AMD_SVM_GLOBAL_FLAG_FINE_GRAINED:=0): 'HSA_AMD_SVM_GLOBAL_FLAG_FINE_GRAINED', (HSA_AMD_SVM_GLOBAL_FLAG_COARSE_GRAINED:=1): 'HSA_AMD_SVM_GLOBAL_FLAG_COARSE_GRAINED', (HSA_AMD_SVM_GLOBAL_FLAG_INDETERMINATE:=2): 'HSA_AMD_SVM_GLOBAL_FLAG_INDETERMINATE'}
hsa_amd_svm_model_t: TypeAlias = ctypes.c_uint32
enum_hsa_amd_svm_attribute_s: dict[int, str] = {(HSA_AMD_SVM_ATTRIB_GLOBAL_FLAG:=0): 'HSA_AMD_SVM_ATTRIB_GLOBAL_FLAG', (HSA_AMD_SVM_ATTRIB_READ_ONLY:=1): 'HSA_AMD_SVM_ATTRIB_READ_ONLY', (HSA_AMD_SVM_ATTRIB_HIVE_LOCAL:=2): 'HSA_AMD_SVM_ATTRIB_HIVE_LOCAL', (HSA_AMD_SVM_ATTRIB_MIGRATION_GRANULARITY:=3): 'HSA_AMD_SVM_ATTRIB_MIGRATION_GRANULARITY', (HSA_AMD_SVM_ATTRIB_PREFERRED_LOCATION:=4): 'HSA_AMD_SVM_ATTRIB_PREFERRED_LOCATION', (HSA_AMD_SVM_ATTRIB_PREFETCH_LOCATION:=5): 'HSA_AMD_SVM_ATTRIB_PREFETCH_LOCATION', (HSA_AMD_SVM_ATTRIB_READ_MOSTLY:=6): 'HSA_AMD_SVM_ATTRIB_READ_MOSTLY', (HSA_AMD_SVM_ATTRIB_GPU_EXEC:=7): 'HSA_AMD_SVM_ATTRIB_GPU_EXEC', (HSA_AMD_SVM_ATTRIB_AGENT_ACCESSIBLE:=512): 'HSA_AMD_SVM_ATTRIB_AGENT_ACCESSIBLE', (HSA_AMD_SVM_ATTRIB_AGENT_ACCESSIBLE_IN_PLACE:=513): 'HSA_AMD_SVM_ATTRIB_AGENT_ACCESSIBLE_IN_PLACE', (HSA_AMD_SVM_ATTRIB_AGENT_NO_ACCESS:=514): 'HSA_AMD_SVM_ATTRIB_AGENT_NO_ACCESS', (HSA_AMD_SVM_ATTRIB_ACCESS_QUERY:=515): 'HSA_AMD_SVM_ATTRIB_ACCESS_QUERY'}
hsa_amd_svm_attribute_t: TypeAlias = ctypes.c_uint32
@c.record
class struct_hsa_amd_svm_attribute_pair_s(c.Struct):
  SIZE = 16
  attribute: int
  value: int
struct_hsa_amd_svm_attribute_pair_s.register_fields([('attribute', uint64_t, 0), ('value', uint64_t, 8)])
hsa_amd_svm_attribute_pair_t: TypeAlias = struct_hsa_amd_svm_attribute_pair_s
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, size_t, c.POINTER[hsa_amd_svm_attribute_pair_t], size_t)
def hsa_amd_svm_attributes_set(ptr:ctypes.c_void_p, size:size_t, attribute_list:c.POINTER[hsa_amd_svm_attribute_pair_t], attribute_count:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, size_t, c.POINTER[hsa_amd_svm_attribute_pair_t], size_t)
def hsa_amd_svm_attributes_get(ptr:ctypes.c_void_p, size:size_t, attribute_list:c.POINTER[hsa_amd_svm_attribute_pair_t], attribute_count:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, size_t, hsa_agent_t, uint32_t, c.POINTER[hsa_signal_t], hsa_signal_t)
def hsa_amd_svm_prefetch_async(ptr:ctypes.c_void_p, size:size_t, agent:hsa_agent_t, num_dep_signals:uint32_t, dep_signals:c.POINTER[hsa_signal_t], completion_signal:hsa_signal_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hsa_agent_t)
def hsa_amd_spm_acquire(preferred_agent:hsa_agent_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hsa_agent_t)
def hsa_amd_spm_release(preferred_agent:hsa_agent_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hsa_agent_t, size_t, c.POINTER[uint32_t], c.POINTER[uint32_t], ctypes.c_void_p, c.POINTER[ctypes.c_bool])
def hsa_amd_spm_set_dest_buffer(preferred_agent:hsa_agent_t, size_in_bytes:size_t, timeout:c.POINTER[uint32_t], size_copied:c.POINTER[uint32_t], dest:ctypes.c_void_p, is_data_loss:c.POINTER[ctypes.c_bool]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, size_t, c.POINTER[ctypes.c_int32], c.POINTER[uint64_t])
def hsa_amd_portable_export_dmabuf(ptr:ctypes.c_void_p, size:size_t, dmabuf:c.POINTER[ctypes.c_int32], offset:c.POINTER[uint64_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, size_t, c.POINTER[ctypes.c_int32], c.POINTER[uint64_t], uint64_t)
def hsa_amd_portable_export_dmabuf_v2(ptr:ctypes.c_void_p, size:size_t, dmabuf:c.POINTER[ctypes.c_int32], offset:c.POINTER[uint64_t], flags:uint64_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_int32)
def hsa_amd_portable_close_dmabuf(dmabuf:int) -> ctypes.c_uint32: ...
enum_hsa_amd_vmem_address_reserve_flag_s: dict[int, str] = {(HSA_AMD_VMEM_ADDRESS_NO_REGISTER:=1): 'HSA_AMD_VMEM_ADDRESS_NO_REGISTER'}
hsa_amd_vmem_address_reserve_flag_t: TypeAlias = ctypes.c_uint32
@dll.bind(ctypes.c_uint32, c.POINTER[ctypes.c_void_p], size_t, uint64_t, uint64_t)
def hsa_amd_vmem_address_reserve(va:c.POINTER[ctypes.c_void_p], size:size_t, address:uint64_t, flags:uint64_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[ctypes.c_void_p], size_t, uint64_t, uint64_t, uint64_t)
def hsa_amd_vmem_address_reserve_align(va:c.POINTER[ctypes.c_void_p], size:size_t, address:uint64_t, alignment:uint64_t, flags:uint64_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, size_t)
def hsa_amd_vmem_address_free(va:ctypes.c_void_p, size:size_t) -> ctypes.c_uint32: ...
@c.record
class struct_hsa_amd_vmem_alloc_handle_s(c.Struct):
  SIZE = 8
  handle: int
struct_hsa_amd_vmem_alloc_handle_s.register_fields([('handle', uint64_t, 0)])
hsa_amd_vmem_alloc_handle_t: TypeAlias = struct_hsa_amd_vmem_alloc_handle_s
hsa_amd_memory_type_t: dict[int, str] = {(MEMORY_TYPE_NONE:=0): 'MEMORY_TYPE_NONE', (MEMORY_TYPE_PINNED:=1): 'MEMORY_TYPE_PINNED'}
@dll.bind(ctypes.c_uint32, hsa_amd_memory_pool_t, size_t, ctypes.c_uint32, uint64_t, c.POINTER[hsa_amd_vmem_alloc_handle_t])
def hsa_amd_vmem_handle_create(pool:hsa_amd_memory_pool_t, size:size_t, type:ctypes.c_uint32, flags:uint64_t, memory_handle:c.POINTER[hsa_amd_vmem_alloc_handle_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hsa_amd_vmem_alloc_handle_t)
def hsa_amd_vmem_handle_release(memory_handle:hsa_amd_vmem_alloc_handle_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, size_t, size_t, hsa_amd_vmem_alloc_handle_t, uint64_t)
def hsa_amd_vmem_map(va:ctypes.c_void_p, size:size_t, in_offset:size_t, memory_handle:hsa_amd_vmem_alloc_handle_t, flags:uint64_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, size_t)
def hsa_amd_vmem_unmap(va:ctypes.c_void_p, size:size_t) -> ctypes.c_uint32: ...
@c.record
class struct_hsa_amd_memory_access_desc_s(c.Struct):
  SIZE = 16
  permissions: int
  agent_handle: struct_hsa_agent_s
struct_hsa_amd_memory_access_desc_s.register_fields([('permissions', ctypes.c_uint32, 0), ('agent_handle', hsa_agent_t, 8)])
hsa_amd_memory_access_desc_t: TypeAlias = struct_hsa_amd_memory_access_desc_s
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, size_t, c.POINTER[hsa_amd_memory_access_desc_t], size_t)
def hsa_amd_vmem_set_access(va:ctypes.c_void_p, size:size_t, desc:c.POINTER[hsa_amd_memory_access_desc_t], desc_cnt:size_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_void_p, c.POINTER[ctypes.c_uint32], hsa_agent_t)
def hsa_amd_vmem_get_access(va:ctypes.c_void_p, perms:c.POINTER[ctypes.c_uint32], agent_handle:hsa_agent_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[ctypes.c_int32], hsa_amd_vmem_alloc_handle_t, uint64_t)
def hsa_amd_vmem_export_shareable_handle(dmabuf_fd:c.POINTER[ctypes.c_int32], handle:hsa_amd_vmem_alloc_handle_t, flags:uint64_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, ctypes.c_int32, c.POINTER[hsa_amd_vmem_alloc_handle_t])
def hsa_amd_vmem_import_shareable_handle(dmabuf_fd:int, handle:c.POINTER[hsa_amd_vmem_alloc_handle_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hsa_amd_vmem_alloc_handle_t], ctypes.c_void_p)
def hsa_amd_vmem_retain_alloc_handle(memory_handle:c.POINTER[hsa_amd_vmem_alloc_handle_t], addr:ctypes.c_void_p) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hsa_amd_vmem_alloc_handle_t, c.POINTER[hsa_amd_memory_pool_t], c.POINTER[ctypes.c_uint32])
def hsa_amd_vmem_get_alloc_properties_from_handle(memory_handle:hsa_amd_vmem_alloc_handle_t, pool:c.POINTER[hsa_amd_memory_pool_t], type:c.POINTER[ctypes.c_uint32]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hsa_agent_t, size_t)
def hsa_amd_agent_set_async_scratch_limit(agent:hsa_agent_t, threshold:size_t) -> ctypes.c_uint32: ...
hsa_queue_info_attribute_t: dict[int, str] = {(HSA_AMD_QUEUE_INFO_AGENT:=0): 'HSA_AMD_QUEUE_INFO_AGENT', (HSA_AMD_QUEUE_INFO_DOORBELL_ID:=1): 'HSA_AMD_QUEUE_INFO_DOORBELL_ID'}
@dll.bind(ctypes.c_uint32, c.POINTER[hsa_queue_t], ctypes.c_uint32, ctypes.c_void_p)
def hsa_amd_queue_get_info(queue:c.POINTER[hsa_queue_t], attribute:ctypes.c_uint32, value:ctypes.c_void_p) -> ctypes.c_uint32: ...
@c.record
class struct_hsa_amd_ais_file_handle_s(c.Struct):
  SIZE = 8
  handle: ctypes.c_void_p
  fd: int
  pad: c.Array[ctypes.c_ubyte, Literal[8]]
struct_hsa_amd_ais_file_handle_s.register_fields([('handle', ctypes.c_void_p, 0), ('fd', ctypes.c_int32, 0), ('pad', c.Array[uint8_t, Literal[8]], 0)])
hsa_amd_ais_file_handle_t: TypeAlias = struct_hsa_amd_ais_file_handle_s
int64_t: TypeAlias = ctypes.c_int64
@dll.bind(ctypes.c_uint32, hsa_amd_ais_file_handle_t, ctypes.c_void_p, uint64_t, int64_t, c.POINTER[uint64_t], c.POINTER[int32_t])
def hsa_amd_ais_file_write(handle:hsa_amd_ais_file_handle_t, devicePtr:ctypes.c_void_p, size:uint64_t, file_offset:int64_t, size_copied:c.POINTER[uint64_t], status:c.POINTER[int32_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hsa_amd_ais_file_handle_t, ctypes.c_void_p, uint64_t, int64_t, c.POINTER[uint64_t], c.POINTER[int32_t])
def hsa_amd_ais_file_read(handle:hsa_amd_ais_file_handle_t, devicePtr:ctypes.c_void_p, size:uint64_t, file_offset:int64_t, size_copied:c.POINTER[uint64_t], status:c.POINTER[int32_t]) -> ctypes.c_uint32: ...
enum_hsa_amd_log_flag_s: dict[int, str] = {(HSA_AMD_LOG_FLAG_BLIT_KERNEL_PKTS:=0): 'HSA_AMD_LOG_FLAG_BLIT_KERNEL_PKTS', (HSA_AMD_LOG_FLAG_AQL:=0): 'HSA_AMD_LOG_FLAG_AQL', (HSA_AMD_LOG_FLAG_SDMA:=1): 'HSA_AMD_LOG_FLAG_SDMA', (HSA_AMD_LOG_FLAG_INFO:=2): 'HSA_AMD_LOG_FLAG_INFO'}
hsa_amd_log_flag_t: TypeAlias = ctypes.c_uint32
@dll.bind(ctypes.c_uint32, c.POINTER[uint8_t], ctypes.c_void_p)
def hsa_amd_enable_logging(flags:c.POINTER[uint8_t], file:ctypes.c_void_p) -> ctypes.c_uint32: ...
amd_signal_kind64_t: TypeAlias = ctypes.c_int64
enum_amd_signal_kind_t: dict[int, str] = {(AMD_SIGNAL_KIND_INVALID:=0): 'AMD_SIGNAL_KIND_INVALID', (AMD_SIGNAL_KIND_USER:=1): 'AMD_SIGNAL_KIND_USER', (AMD_SIGNAL_KIND_DOORBELL:=-1): 'AMD_SIGNAL_KIND_DOORBELL', (AMD_SIGNAL_KIND_LEGACY_DOORBELL:=-2): 'AMD_SIGNAL_KIND_LEGACY_DOORBELL'}
@c.record
class struct_amd_signal_s(c.Struct):
  SIZE = 64
  kind: int
  value: int
  hardware_doorbell_ptr: c.POINTER[ctypes.c_uint64]
  event_mailbox_ptr: int
  event_id: int
  reserved1: int
  start_ts: int
  end_ts: int
  queue_ptr: c.POINTER[struct_amd_queue_v2_s]
  reserved2: int
  reserved3: c.Array[ctypes.c_uint32, Literal[2]]
@c.record
class struct_amd_queue_v2_s(c.Struct):
  SIZE = 2304
  hsa_queue: struct_hsa_queue_s
  caps: int
  reserved1: c.Array[ctypes.c_uint32, Literal[3]]
  write_dispatch_id: int
  group_segment_aperture_base_hi: int
  private_segment_aperture_base_hi: int
  max_cu_id: int
  max_wave_id: int
  max_legacy_doorbell_dispatch_id_plus_1: int
  legacy_doorbell_lock: int
  reserved2: c.Array[ctypes.c_uint32, Literal[9]]
  read_dispatch_id: int
  read_dispatch_id_field_base_byte_offset: int
  compute_tmpring_size: int
  scratch_resource_descriptor: c.Array[ctypes.c_uint32, Literal[4]]
  scratch_backing_memory_location: int
  scratch_backing_memory_byte_size: int
  scratch_wave64_lane_byte_size: int
  queue_properties: int
  scratch_max_use_index: int
  queue_inactive_signal: struct_hsa_signal_s
  alt_scratch_max_use_index: int
  alt_scratch_resource_descriptor: c.Array[ctypes.c_uint32, Literal[4]]
  alt_scratch_backing_memory_location: int
  alt_scratch_dispatch_limit_x: int
  alt_scratch_dispatch_limit_y: int
  alt_scratch_dispatch_limit_z: int
  alt_scratch_wave64_lane_byte_size: int
  alt_compute_tmpring_size: int
  reserved5: int
  scratch_last_used_index: c.Array[struct_scratch_last_used_index_xcc_s, Literal[128]]
amd_queue_v2_t: TypeAlias = struct_amd_queue_v2_s
amd_queue_properties32_t: TypeAlias = ctypes.c_uint32
@c.record
class struct_scratch_last_used_index_xcc_s(c.Struct):
  SIZE = 16
  main: int
  alt: int
scratch_last_used_index_xcc_t: TypeAlias = struct_scratch_last_used_index_xcc_s
struct_scratch_last_used_index_xcc_s.register_fields([('main', uint64_t, 0), ('alt', uint64_t, 8)])
struct_amd_queue_v2_s.register_fields([('hsa_queue', hsa_queue_t, 0), ('caps', uint32_t, 40), ('reserved1', c.Array[uint32_t, Literal[3]], 44), ('write_dispatch_id', uint64_t, 56), ('group_segment_aperture_base_hi', uint32_t, 64), ('private_segment_aperture_base_hi', uint32_t, 68), ('max_cu_id', uint32_t, 72), ('max_wave_id', uint32_t, 76), ('max_legacy_doorbell_dispatch_id_plus_1', uint64_t, 80), ('legacy_doorbell_lock', uint32_t, 88), ('reserved2', c.Array[uint32_t, Literal[9]], 92), ('read_dispatch_id', uint64_t, 128), ('read_dispatch_id_field_base_byte_offset', uint32_t, 136), ('compute_tmpring_size', uint32_t, 140), ('scratch_resource_descriptor', c.Array[uint32_t, Literal[4]], 144), ('scratch_backing_memory_location', uint64_t, 160), ('scratch_backing_memory_byte_size', uint64_t, 168), ('scratch_wave64_lane_byte_size', uint32_t, 176), ('queue_properties', amd_queue_properties32_t, 180), ('scratch_max_use_index', uint64_t, 184), ('queue_inactive_signal', hsa_signal_t, 192), ('alt_scratch_max_use_index', uint64_t, 200), ('alt_scratch_resource_descriptor', c.Array[uint32_t, Literal[4]], 208), ('alt_scratch_backing_memory_location', uint64_t, 224), ('alt_scratch_dispatch_limit_x', uint32_t, 232), ('alt_scratch_dispatch_limit_y', uint32_t, 236), ('alt_scratch_dispatch_limit_z', uint32_t, 240), ('alt_scratch_wave64_lane_byte_size', uint32_t, 244), ('alt_compute_tmpring_size', uint32_t, 248), ('reserved5', uint32_t, 252), ('scratch_last_used_index', c.Array[scratch_last_used_index_xcc_t, Literal[128]], 256)])
struct_amd_signal_s.register_fields([('kind', amd_signal_kind64_t, 0), ('value', int64_t, 8), ('hardware_doorbell_ptr', c.POINTER[uint64_t], 8), ('event_mailbox_ptr', uint64_t, 16), ('event_id', uint32_t, 24), ('reserved1', uint32_t, 28), ('start_ts', uint64_t, 32), ('end_ts', uint64_t, 40), ('queue_ptr', c.POINTER[amd_queue_v2_t], 48), ('reserved2', uint64_t, 48), ('reserved3', c.Array[uint32_t, Literal[2]], 56)])
amd_signal_t: TypeAlias = struct_amd_signal_s
enum_amd_queue_properties_t: dict[int, str] = {(AMD_QUEUE_PROPERTIES_ENABLE_TRAP_HANDLER_SHIFT:=0): 'AMD_QUEUE_PROPERTIES_ENABLE_TRAP_HANDLER_SHIFT', (AMD_QUEUE_PROPERTIES_ENABLE_TRAP_HANDLER_WIDTH:=1): 'AMD_QUEUE_PROPERTIES_ENABLE_TRAP_HANDLER_WIDTH', (AMD_QUEUE_PROPERTIES_ENABLE_TRAP_HANDLER:=1): 'AMD_QUEUE_PROPERTIES_ENABLE_TRAP_HANDLER', (AMD_QUEUE_PROPERTIES_IS_PTR64_SHIFT:=1): 'AMD_QUEUE_PROPERTIES_IS_PTR64_SHIFT', (AMD_QUEUE_PROPERTIES_IS_PTR64_WIDTH:=1): 'AMD_QUEUE_PROPERTIES_IS_PTR64_WIDTH', (AMD_QUEUE_PROPERTIES_IS_PTR64:=2): 'AMD_QUEUE_PROPERTIES_IS_PTR64', (AMD_QUEUE_PROPERTIES_ENABLE_TRAP_HANDLER_DEBUG_SGPRS_SHIFT:=2): 'AMD_QUEUE_PROPERTIES_ENABLE_TRAP_HANDLER_DEBUG_SGPRS_SHIFT', (AMD_QUEUE_PROPERTIES_ENABLE_TRAP_HANDLER_DEBUG_SGPRS_WIDTH:=1): 'AMD_QUEUE_PROPERTIES_ENABLE_TRAP_HANDLER_DEBUG_SGPRS_WIDTH', (AMD_QUEUE_PROPERTIES_ENABLE_TRAP_HANDLER_DEBUG_SGPRS:=4): 'AMD_QUEUE_PROPERTIES_ENABLE_TRAP_HANDLER_DEBUG_SGPRS', (AMD_QUEUE_PROPERTIES_ENABLE_PROFILING_SHIFT:=3): 'AMD_QUEUE_PROPERTIES_ENABLE_PROFILING_SHIFT', (AMD_QUEUE_PROPERTIES_ENABLE_PROFILING_WIDTH:=1): 'AMD_QUEUE_PROPERTIES_ENABLE_PROFILING_WIDTH', (AMD_QUEUE_PROPERTIES_ENABLE_PROFILING:=8): 'AMD_QUEUE_PROPERTIES_ENABLE_PROFILING', (AMD_QUEUE_PROPERTIES_USE_SCRATCH_ONCE_SHIFT:=4): 'AMD_QUEUE_PROPERTIES_USE_SCRATCH_ONCE_SHIFT', (AMD_QUEUE_PROPERTIES_USE_SCRATCH_ONCE_WIDTH:=1): 'AMD_QUEUE_PROPERTIES_USE_SCRATCH_ONCE_WIDTH', (AMD_QUEUE_PROPERTIES_USE_SCRATCH_ONCE:=16): 'AMD_QUEUE_PROPERTIES_USE_SCRATCH_ONCE', (AMD_QUEUE_PROPERTIES_RESERVED1_SHIFT:=5): 'AMD_QUEUE_PROPERTIES_RESERVED1_SHIFT', (AMD_QUEUE_PROPERTIES_RESERVED1_WIDTH:=27): 'AMD_QUEUE_PROPERTIES_RESERVED1_WIDTH', (AMD_QUEUE_PROPERTIES_RESERVED1:=-32): 'AMD_QUEUE_PROPERTIES_RESERVED1'}
amd_queue_capabilities32_t: TypeAlias = ctypes.c_uint32
enum_amd_queue_capabilities_t: dict[int, str] = {(AMD_QUEUE_CAPS_CP_ASYNC_RECLAIM_SHIFT:=0): 'AMD_QUEUE_CAPS_CP_ASYNC_RECLAIM_SHIFT', (AMD_QUEUE_CAPS_CP_ASYNC_RECLAIM_WIDTH:=1): 'AMD_QUEUE_CAPS_CP_ASYNC_RECLAIM_WIDTH', (AMD_QUEUE_CAPS_CP_ASYNC_RECLAIM:=1): 'AMD_QUEUE_CAPS_CP_ASYNC_RECLAIM', (AMD_QUEUE_CAPS_SW_ASYNC_RECLAIM_SHIFT:=1): 'AMD_QUEUE_CAPS_SW_ASYNC_RECLAIM_SHIFT', (AMD_QUEUE_CAPS_SW_ASYNC_RECLAIM_WIDTH:=1): 'AMD_QUEUE_CAPS_SW_ASYNC_RECLAIM_WIDTH', (AMD_QUEUE_CAPS_SW_ASYNC_RECLAIM:=2): 'AMD_QUEUE_CAPS_SW_ASYNC_RECLAIM'}
@c.record
class struct_amd_queue_s(c.Struct):
  SIZE = 256
  hsa_queue: struct_hsa_queue_s
  caps: int
  reserved1: c.Array[ctypes.c_uint32, Literal[3]]
  write_dispatch_id: int
  group_segment_aperture_base_hi: int
  private_segment_aperture_base_hi: int
  max_cu_id: int
  max_wave_id: int
  max_legacy_doorbell_dispatch_id_plus_1: int
  legacy_doorbell_lock: int
  reserved2: c.Array[ctypes.c_uint32, Literal[9]]
  read_dispatch_id: int
  read_dispatch_id_field_base_byte_offset: int
  compute_tmpring_size: int
  scratch_resource_descriptor: c.Array[ctypes.c_uint32, Literal[4]]
  scratch_backing_memory_location: int
  reserved3: c.Array[ctypes.c_uint32, Literal[2]]
  scratch_wave64_lane_byte_size: int
  queue_properties: int
  reserved4: c.Array[ctypes.c_uint32, Literal[2]]
  queue_inactive_signal: struct_hsa_signal_s
  reserved5: c.Array[ctypes.c_uint32, Literal[14]]
struct_amd_queue_s.register_fields([('hsa_queue', hsa_queue_t, 0), ('caps', uint32_t, 40), ('reserved1', c.Array[uint32_t, Literal[3]], 44), ('write_dispatch_id', uint64_t, 56), ('group_segment_aperture_base_hi', uint32_t, 64), ('private_segment_aperture_base_hi', uint32_t, 68), ('max_cu_id', uint32_t, 72), ('max_wave_id', uint32_t, 76), ('max_legacy_doorbell_dispatch_id_plus_1', uint64_t, 80), ('legacy_doorbell_lock', uint32_t, 88), ('reserved2', c.Array[uint32_t, Literal[9]], 92), ('read_dispatch_id', uint64_t, 128), ('read_dispatch_id_field_base_byte_offset', uint32_t, 136), ('compute_tmpring_size', uint32_t, 140), ('scratch_resource_descriptor', c.Array[uint32_t, Literal[4]], 144), ('scratch_backing_memory_location', uint64_t, 160), ('reserved3', c.Array[uint32_t, Literal[2]], 168), ('scratch_wave64_lane_byte_size', uint32_t, 176), ('queue_properties', amd_queue_properties32_t, 180), ('reserved4', c.Array[uint32_t, Literal[2]], 184), ('queue_inactive_signal', hsa_signal_t, 192), ('reserved5', c.Array[uint32_t, Literal[14]], 200)])
amd_queue_t: TypeAlias = struct_amd_queue_s
amd_kernel_code_version32_t: TypeAlias = ctypes.c_uint32
enum_amd_kernel_code_version_t: dict[int, str] = {(AMD_KERNEL_CODE_VERSION_MAJOR:=1): 'AMD_KERNEL_CODE_VERSION_MAJOR', (AMD_KERNEL_CODE_VERSION_MINOR:=1): 'AMD_KERNEL_CODE_VERSION_MINOR'}
amd_machine_kind16_t: TypeAlias = ctypes.c_uint16
enum_amd_machine_kind_t: dict[int, str] = {(AMD_MACHINE_KIND_UNDEFINED:=0): 'AMD_MACHINE_KIND_UNDEFINED', (AMD_MACHINE_KIND_AMDGPU:=1): 'AMD_MACHINE_KIND_AMDGPU'}
amd_machine_version16_t: TypeAlias = ctypes.c_uint16
enum_amd_float_round_mode_t: dict[int, str] = {(AMD_FLOAT_ROUND_MODE_NEAREST_EVEN:=0): 'AMD_FLOAT_ROUND_MODE_NEAREST_EVEN', (AMD_FLOAT_ROUND_MODE_PLUS_INFINITY:=1): 'AMD_FLOAT_ROUND_MODE_PLUS_INFINITY', (AMD_FLOAT_ROUND_MODE_MINUS_INFINITY:=2): 'AMD_FLOAT_ROUND_MODE_MINUS_INFINITY', (AMD_FLOAT_ROUND_MODE_ZERO:=3): 'AMD_FLOAT_ROUND_MODE_ZERO'}
enum_amd_float_denorm_mode_t: dict[int, str] = {(AMD_FLOAT_DENORM_MODE_FLUSH_SOURCE_OUTPUT:=0): 'AMD_FLOAT_DENORM_MODE_FLUSH_SOURCE_OUTPUT', (AMD_FLOAT_DENORM_MODE_FLUSH_OUTPUT:=1): 'AMD_FLOAT_DENORM_MODE_FLUSH_OUTPUT', (AMD_FLOAT_DENORM_MODE_FLUSH_SOURCE:=2): 'AMD_FLOAT_DENORM_MODE_FLUSH_SOURCE', (AMD_FLOAT_DENORM_MODE_NO_FLUSH:=3): 'AMD_FLOAT_DENORM_MODE_NO_FLUSH'}
amd_compute_pgm_rsrc_one32_t: TypeAlias = ctypes.c_uint32
enum_amd_compute_pgm_rsrc_one_t: dict[int, str] = {(AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WORKITEM_VGPR_COUNT_SHIFT:=0): 'AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WORKITEM_VGPR_COUNT_SHIFT', (AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WORKITEM_VGPR_COUNT_WIDTH:=6): 'AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WORKITEM_VGPR_COUNT_WIDTH', (AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WORKITEM_VGPR_COUNT:=63): 'AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WORKITEM_VGPR_COUNT', (AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WAVEFRONT_SGPR_COUNT_SHIFT:=6): 'AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WAVEFRONT_SGPR_COUNT_SHIFT', (AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WAVEFRONT_SGPR_COUNT_WIDTH:=4): 'AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WAVEFRONT_SGPR_COUNT_WIDTH', (AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WAVEFRONT_SGPR_COUNT:=960): 'AMD_COMPUTE_PGM_RSRC_ONE_GRANULATED_WAVEFRONT_SGPR_COUNT', (AMD_COMPUTE_PGM_RSRC_ONE_PRIORITY_SHIFT:=10): 'AMD_COMPUTE_PGM_RSRC_ONE_PRIORITY_SHIFT', (AMD_COMPUTE_PGM_RSRC_ONE_PRIORITY_WIDTH:=2): 'AMD_COMPUTE_PGM_RSRC_ONE_PRIORITY_WIDTH', (AMD_COMPUTE_PGM_RSRC_ONE_PRIORITY:=3072): 'AMD_COMPUTE_PGM_RSRC_ONE_PRIORITY', (AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_ROUND_MODE_32_SHIFT:=12): 'AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_ROUND_MODE_32_SHIFT', (AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_ROUND_MODE_32_WIDTH:=2): 'AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_ROUND_MODE_32_WIDTH', (AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_ROUND_MODE_32:=12288): 'AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_ROUND_MODE_32', (AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_ROUND_MODE_16_64_SHIFT:=14): 'AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_ROUND_MODE_16_64_SHIFT', (AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_ROUND_MODE_16_64_WIDTH:=2): 'AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_ROUND_MODE_16_64_WIDTH', (AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_ROUND_MODE_16_64:=49152): 'AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_ROUND_MODE_16_64', (AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_32_SHIFT:=16): 'AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_32_SHIFT', (AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_32_WIDTH:=2): 'AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_32_WIDTH', (AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_32:=196608): 'AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_32', (AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_16_64_SHIFT:=18): 'AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_16_64_SHIFT', (AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_16_64_WIDTH:=2): 'AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_16_64_WIDTH', (AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_16_64:=786432): 'AMD_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_16_64', (AMD_COMPUTE_PGM_RSRC_ONE_PRIV_SHIFT:=20): 'AMD_COMPUTE_PGM_RSRC_ONE_PRIV_SHIFT', (AMD_COMPUTE_PGM_RSRC_ONE_PRIV_WIDTH:=1): 'AMD_COMPUTE_PGM_RSRC_ONE_PRIV_WIDTH', (AMD_COMPUTE_PGM_RSRC_ONE_PRIV:=1048576): 'AMD_COMPUTE_PGM_RSRC_ONE_PRIV', (AMD_COMPUTE_PGM_RSRC_ONE_ENABLE_DX10_CLAMP_SHIFT:=21): 'AMD_COMPUTE_PGM_RSRC_ONE_ENABLE_DX10_CLAMP_SHIFT', (AMD_COMPUTE_PGM_RSRC_ONE_ENABLE_DX10_CLAMP_WIDTH:=1): 'AMD_COMPUTE_PGM_RSRC_ONE_ENABLE_DX10_CLAMP_WIDTH', (AMD_COMPUTE_PGM_RSRC_ONE_ENABLE_DX10_CLAMP:=2097152): 'AMD_COMPUTE_PGM_RSRC_ONE_ENABLE_DX10_CLAMP', (AMD_COMPUTE_PGM_RSRC_ONE_DEBUG_MODE_SHIFT:=22): 'AMD_COMPUTE_PGM_RSRC_ONE_DEBUG_MODE_SHIFT', (AMD_COMPUTE_PGM_RSRC_ONE_DEBUG_MODE_WIDTH:=1): 'AMD_COMPUTE_PGM_RSRC_ONE_DEBUG_MODE_WIDTH', (AMD_COMPUTE_PGM_RSRC_ONE_DEBUG_MODE:=4194304): 'AMD_COMPUTE_PGM_RSRC_ONE_DEBUG_MODE', (AMD_COMPUTE_PGM_RSRC_ONE_ENABLE_IEEE_MODE_SHIFT:=23): 'AMD_COMPUTE_PGM_RSRC_ONE_ENABLE_IEEE_MODE_SHIFT', (AMD_COMPUTE_PGM_RSRC_ONE_ENABLE_IEEE_MODE_WIDTH:=1): 'AMD_COMPUTE_PGM_RSRC_ONE_ENABLE_IEEE_MODE_WIDTH', (AMD_COMPUTE_PGM_RSRC_ONE_ENABLE_IEEE_MODE:=8388608): 'AMD_COMPUTE_PGM_RSRC_ONE_ENABLE_IEEE_MODE', (AMD_COMPUTE_PGM_RSRC_ONE_BULKY_SHIFT:=24): 'AMD_COMPUTE_PGM_RSRC_ONE_BULKY_SHIFT', (AMD_COMPUTE_PGM_RSRC_ONE_BULKY_WIDTH:=1): 'AMD_COMPUTE_PGM_RSRC_ONE_BULKY_WIDTH', (AMD_COMPUTE_PGM_RSRC_ONE_BULKY:=16777216): 'AMD_COMPUTE_PGM_RSRC_ONE_BULKY', (AMD_COMPUTE_PGM_RSRC_ONE_CDBG_USER_SHIFT:=25): 'AMD_COMPUTE_PGM_RSRC_ONE_CDBG_USER_SHIFT', (AMD_COMPUTE_PGM_RSRC_ONE_CDBG_USER_WIDTH:=1): 'AMD_COMPUTE_PGM_RSRC_ONE_CDBG_USER_WIDTH', (AMD_COMPUTE_PGM_RSRC_ONE_CDBG_USER:=33554432): 'AMD_COMPUTE_PGM_RSRC_ONE_CDBG_USER', (AMD_COMPUTE_PGM_RSRC_ONE_RESERVED1_SHIFT:=26): 'AMD_COMPUTE_PGM_RSRC_ONE_RESERVED1_SHIFT', (AMD_COMPUTE_PGM_RSRC_ONE_RESERVED1_WIDTH:=6): 'AMD_COMPUTE_PGM_RSRC_ONE_RESERVED1_WIDTH', (AMD_COMPUTE_PGM_RSRC_ONE_RESERVED1:=-67108864): 'AMD_COMPUTE_PGM_RSRC_ONE_RESERVED1'}
enum_amd_system_vgpr_workitem_id_t: dict[int, str] = {(AMD_SYSTEM_VGPR_WORKITEM_ID_X:=0): 'AMD_SYSTEM_VGPR_WORKITEM_ID_X', (AMD_SYSTEM_VGPR_WORKITEM_ID_X_Y:=1): 'AMD_SYSTEM_VGPR_WORKITEM_ID_X_Y', (AMD_SYSTEM_VGPR_WORKITEM_ID_X_Y_Z:=2): 'AMD_SYSTEM_VGPR_WORKITEM_ID_X_Y_Z', (AMD_SYSTEM_VGPR_WORKITEM_ID_UNDEFINED:=3): 'AMD_SYSTEM_VGPR_WORKITEM_ID_UNDEFINED'}
amd_compute_pgm_rsrc_two32_t: TypeAlias = ctypes.c_uint32
enum_amd_compute_pgm_rsrc_two_t: dict[int, str] = {(AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_PRIVATE_SEGMENT_WAVE_BYTE_OFFSET_SHIFT:=0): 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_PRIVATE_SEGMENT_WAVE_BYTE_OFFSET_SHIFT', (AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_PRIVATE_SEGMENT_WAVE_BYTE_OFFSET_WIDTH:=1): 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_PRIVATE_SEGMENT_WAVE_BYTE_OFFSET_WIDTH', (AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_PRIVATE_SEGMENT_WAVE_BYTE_OFFSET:=1): 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_PRIVATE_SEGMENT_WAVE_BYTE_OFFSET', (AMD_COMPUTE_PGM_RSRC_TWO_USER_SGPR_COUNT_SHIFT:=1): 'AMD_COMPUTE_PGM_RSRC_TWO_USER_SGPR_COUNT_SHIFT', (AMD_COMPUTE_PGM_RSRC_TWO_USER_SGPR_COUNT_WIDTH:=5): 'AMD_COMPUTE_PGM_RSRC_TWO_USER_SGPR_COUNT_WIDTH', (AMD_COMPUTE_PGM_RSRC_TWO_USER_SGPR_COUNT:=62): 'AMD_COMPUTE_PGM_RSRC_TWO_USER_SGPR_COUNT', (AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_TRAP_HANDLER_SHIFT:=6): 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_TRAP_HANDLER_SHIFT', (AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_TRAP_HANDLER_WIDTH:=1): 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_TRAP_HANDLER_WIDTH', (AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_TRAP_HANDLER:=64): 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_TRAP_HANDLER', (AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_X_SHIFT:=7): 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_X_SHIFT', (AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_X_WIDTH:=1): 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_X_WIDTH', (AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_X:=128): 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_X', (AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Y_SHIFT:=8): 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Y_SHIFT', (AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Y_WIDTH:=1): 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Y_WIDTH', (AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Y:=256): 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Y', (AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Z_SHIFT:=9): 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Z_SHIFT', (AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Z_WIDTH:=1): 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Z_WIDTH', (AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Z:=512): 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Z', (AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_INFO_SHIFT:=10): 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_INFO_SHIFT', (AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_INFO_WIDTH:=1): 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_INFO_WIDTH', (AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_INFO:=1024): 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_INFO', (AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_VGPR_WORKITEM_ID_SHIFT:=11): 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_VGPR_WORKITEM_ID_SHIFT', (AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_VGPR_WORKITEM_ID_WIDTH:=2): 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_VGPR_WORKITEM_ID_WIDTH', (AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_VGPR_WORKITEM_ID:=6144): 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_VGPR_WORKITEM_ID', (AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_ADDRESS_WATCH_SHIFT:=13): 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_ADDRESS_WATCH_SHIFT', (AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_ADDRESS_WATCH_WIDTH:=1): 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_ADDRESS_WATCH_WIDTH', (AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_ADDRESS_WATCH:=8192): 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_ADDRESS_WATCH', (AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_MEMORY_VIOLATION_SHIFT:=14): 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_MEMORY_VIOLATION_SHIFT', (AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_MEMORY_VIOLATION_WIDTH:=1): 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_MEMORY_VIOLATION_WIDTH', (AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_MEMORY_VIOLATION:=16384): 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_MEMORY_VIOLATION', (AMD_COMPUTE_PGM_RSRC_TWO_GRANULATED_LDS_SIZE_SHIFT:=15): 'AMD_COMPUTE_PGM_RSRC_TWO_GRANULATED_LDS_SIZE_SHIFT', (AMD_COMPUTE_PGM_RSRC_TWO_GRANULATED_LDS_SIZE_WIDTH:=9): 'AMD_COMPUTE_PGM_RSRC_TWO_GRANULATED_LDS_SIZE_WIDTH', (AMD_COMPUTE_PGM_RSRC_TWO_GRANULATED_LDS_SIZE:=16744448): 'AMD_COMPUTE_PGM_RSRC_TWO_GRANULATED_LDS_SIZE', (AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_INVALID_OPERATION_SHIFT:=24): 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_INVALID_OPERATION_SHIFT', (AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_INVALID_OPERATION_WIDTH:=1): 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_INVALID_OPERATION_WIDTH', (AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_INVALID_OPERATION:=16777216): 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_INVALID_OPERATION', (AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_FP_DENORMAL_SOURCE_SHIFT:=25): 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_FP_DENORMAL_SOURCE_SHIFT', (AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_FP_DENORMAL_SOURCE_WIDTH:=1): 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_FP_DENORMAL_SOURCE_WIDTH', (AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_FP_DENORMAL_SOURCE:=33554432): 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_FP_DENORMAL_SOURCE', (AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_DIVISION_BY_ZERO_SHIFT:=26): 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_DIVISION_BY_ZERO_SHIFT', (AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_DIVISION_BY_ZERO_WIDTH:=1): 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_DIVISION_BY_ZERO_WIDTH', (AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_DIVISION_BY_ZERO:=67108864): 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_DIVISION_BY_ZERO', (AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_OVERFLOW_SHIFT:=27): 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_OVERFLOW_SHIFT', (AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_OVERFLOW_WIDTH:=1): 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_OVERFLOW_WIDTH', (AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_OVERFLOW:=134217728): 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_OVERFLOW', (AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_UNDERFLOW_SHIFT:=28): 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_UNDERFLOW_SHIFT', (AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_UNDERFLOW_WIDTH:=1): 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_UNDERFLOW_WIDTH', (AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_UNDERFLOW:=268435456): 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_UNDERFLOW', (AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_INEXACT_SHIFT:=29): 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_INEXACT_SHIFT', (AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_INEXACT_WIDTH:=1): 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_INEXACT_WIDTH', (AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_INEXACT:=536870912): 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_INEXACT', (AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_INT_DIVISION_BY_ZERO_SHIFT:=30): 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_INT_DIVISION_BY_ZERO_SHIFT', (AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_INT_DIVISION_BY_ZERO_WIDTH:=1): 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_INT_DIVISION_BY_ZERO_WIDTH', (AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_INT_DIVISION_BY_ZERO:=1073741824): 'AMD_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_INT_DIVISION_BY_ZERO', (AMD_COMPUTE_PGM_RSRC_TWO_RESERVED1_SHIFT:=31): 'AMD_COMPUTE_PGM_RSRC_TWO_RESERVED1_SHIFT', (AMD_COMPUTE_PGM_RSRC_TWO_RESERVED1_WIDTH:=1): 'AMD_COMPUTE_PGM_RSRC_TWO_RESERVED1_WIDTH', (AMD_COMPUTE_PGM_RSRC_TWO_RESERVED1:=-2147483648): 'AMD_COMPUTE_PGM_RSRC_TWO_RESERVED1'}
enum_amd_element_byte_size_t: dict[int, str] = {(AMD_ELEMENT_BYTE_SIZE_2:=0): 'AMD_ELEMENT_BYTE_SIZE_2', (AMD_ELEMENT_BYTE_SIZE_4:=1): 'AMD_ELEMENT_BYTE_SIZE_4', (AMD_ELEMENT_BYTE_SIZE_8:=2): 'AMD_ELEMENT_BYTE_SIZE_8', (AMD_ELEMENT_BYTE_SIZE_16:=3): 'AMD_ELEMENT_BYTE_SIZE_16'}
amd_kernel_code_properties32_t: TypeAlias = ctypes.c_uint32
enum_amd_kernel_code_properties_t: dict[int, str] = {(AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER_SHIFT:=0): 'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER_SHIFT', (AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER_WIDTH:=1): 'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER_WIDTH', (AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER:=1): 'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER', (AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_PTR_SHIFT:=1): 'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_PTR_SHIFT', (AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_PTR_WIDTH:=1): 'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_PTR_WIDTH', (AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_PTR:=2): 'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_PTR', (AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_QUEUE_PTR_SHIFT:=2): 'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_QUEUE_PTR_SHIFT', (AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_QUEUE_PTR_WIDTH:=1): 'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_QUEUE_PTR_WIDTH', (AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_QUEUE_PTR:=4): 'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_QUEUE_PTR', (AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_KERNARG_SEGMENT_PTR_SHIFT:=3): 'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_KERNARG_SEGMENT_PTR_SHIFT', (AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_KERNARG_SEGMENT_PTR_WIDTH:=1): 'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_KERNARG_SEGMENT_PTR_WIDTH', (AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_KERNARG_SEGMENT_PTR:=8): 'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_KERNARG_SEGMENT_PTR', (AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_ID_SHIFT:=4): 'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_ID_SHIFT', (AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_ID_WIDTH:=1): 'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_ID_WIDTH', (AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_ID:=16): 'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_ID', (AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_FLAT_SCRATCH_INIT_SHIFT:=5): 'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_FLAT_SCRATCH_INIT_SHIFT', (AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_FLAT_SCRATCH_INIT_WIDTH:=1): 'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_FLAT_SCRATCH_INIT_WIDTH', (AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_FLAT_SCRATCH_INIT:=32): 'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_FLAT_SCRATCH_INIT', (AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE_SHIFT:=6): 'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE_SHIFT', (AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE_WIDTH:=1): 'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE_WIDTH', (AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE:=64): 'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE', (AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_X_SHIFT:=7): 'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_X_SHIFT', (AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_X_WIDTH:=1): 'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_X_WIDTH', (AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_X:=128): 'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_X', (AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Y_SHIFT:=8): 'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Y_SHIFT', (AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Y_WIDTH:=1): 'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Y_WIDTH', (AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Y:=256): 'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Y', (AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Z_SHIFT:=9): 'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Z_SHIFT', (AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Z_WIDTH:=1): 'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Z_WIDTH', (AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Z:=512): 'AMD_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Z', (AMD_KERNEL_CODE_PROPERTIES_ENABLE_WAVEFRONT_SIZE32_SHIFT:=10): 'AMD_KERNEL_CODE_PROPERTIES_ENABLE_WAVEFRONT_SIZE32_SHIFT', (AMD_KERNEL_CODE_PROPERTIES_ENABLE_WAVEFRONT_SIZE32_WIDTH:=1): 'AMD_KERNEL_CODE_PROPERTIES_ENABLE_WAVEFRONT_SIZE32_WIDTH', (AMD_KERNEL_CODE_PROPERTIES_ENABLE_WAVEFRONT_SIZE32:=1024): 'AMD_KERNEL_CODE_PROPERTIES_ENABLE_WAVEFRONT_SIZE32', (AMD_KERNEL_CODE_PROPERTIES_RESERVED1_SHIFT:=11): 'AMD_KERNEL_CODE_PROPERTIES_RESERVED1_SHIFT', (AMD_KERNEL_CODE_PROPERTIES_RESERVED1_WIDTH:=5): 'AMD_KERNEL_CODE_PROPERTIES_RESERVED1_WIDTH', (AMD_KERNEL_CODE_PROPERTIES_RESERVED1:=63488): 'AMD_KERNEL_CODE_PROPERTIES_RESERVED1', (AMD_KERNEL_CODE_PROPERTIES_ENABLE_ORDERED_APPEND_GDS_SHIFT:=16): 'AMD_KERNEL_CODE_PROPERTIES_ENABLE_ORDERED_APPEND_GDS_SHIFT', (AMD_KERNEL_CODE_PROPERTIES_ENABLE_ORDERED_APPEND_GDS_WIDTH:=1): 'AMD_KERNEL_CODE_PROPERTIES_ENABLE_ORDERED_APPEND_GDS_WIDTH', (AMD_KERNEL_CODE_PROPERTIES_ENABLE_ORDERED_APPEND_GDS:=65536): 'AMD_KERNEL_CODE_PROPERTIES_ENABLE_ORDERED_APPEND_GDS', (AMD_KERNEL_CODE_PROPERTIES_PRIVATE_ELEMENT_SIZE_SHIFT:=17): 'AMD_KERNEL_CODE_PROPERTIES_PRIVATE_ELEMENT_SIZE_SHIFT', (AMD_KERNEL_CODE_PROPERTIES_PRIVATE_ELEMENT_SIZE_WIDTH:=2): 'AMD_KERNEL_CODE_PROPERTIES_PRIVATE_ELEMENT_SIZE_WIDTH', (AMD_KERNEL_CODE_PROPERTIES_PRIVATE_ELEMENT_SIZE:=393216): 'AMD_KERNEL_CODE_PROPERTIES_PRIVATE_ELEMENT_SIZE', (AMD_KERNEL_CODE_PROPERTIES_IS_PTR64_SHIFT:=19): 'AMD_KERNEL_CODE_PROPERTIES_IS_PTR64_SHIFT', (AMD_KERNEL_CODE_PROPERTIES_IS_PTR64_WIDTH:=1): 'AMD_KERNEL_CODE_PROPERTIES_IS_PTR64_WIDTH', (AMD_KERNEL_CODE_PROPERTIES_IS_PTR64:=524288): 'AMD_KERNEL_CODE_PROPERTIES_IS_PTR64', (AMD_KERNEL_CODE_PROPERTIES_IS_DYNAMIC_CALLSTACK_SHIFT:=20): 'AMD_KERNEL_CODE_PROPERTIES_IS_DYNAMIC_CALLSTACK_SHIFT', (AMD_KERNEL_CODE_PROPERTIES_IS_DYNAMIC_CALLSTACK_WIDTH:=1): 'AMD_KERNEL_CODE_PROPERTIES_IS_DYNAMIC_CALLSTACK_WIDTH', (AMD_KERNEL_CODE_PROPERTIES_IS_DYNAMIC_CALLSTACK:=1048576): 'AMD_KERNEL_CODE_PROPERTIES_IS_DYNAMIC_CALLSTACK', (AMD_KERNEL_CODE_PROPERTIES_IS_DEBUG_ENABLED_SHIFT:=21): 'AMD_KERNEL_CODE_PROPERTIES_IS_DEBUG_ENABLED_SHIFT', (AMD_KERNEL_CODE_PROPERTIES_IS_DEBUG_ENABLED_WIDTH:=1): 'AMD_KERNEL_CODE_PROPERTIES_IS_DEBUG_ENABLED_WIDTH', (AMD_KERNEL_CODE_PROPERTIES_IS_DEBUG_ENABLED:=2097152): 'AMD_KERNEL_CODE_PROPERTIES_IS_DEBUG_ENABLED', (AMD_KERNEL_CODE_PROPERTIES_IS_XNACK_ENABLED_SHIFT:=22): 'AMD_KERNEL_CODE_PROPERTIES_IS_XNACK_ENABLED_SHIFT', (AMD_KERNEL_CODE_PROPERTIES_IS_XNACK_ENABLED_WIDTH:=1): 'AMD_KERNEL_CODE_PROPERTIES_IS_XNACK_ENABLED_WIDTH', (AMD_KERNEL_CODE_PROPERTIES_IS_XNACK_ENABLED:=4194304): 'AMD_KERNEL_CODE_PROPERTIES_IS_XNACK_ENABLED', (AMD_KERNEL_CODE_PROPERTIES_RESERVED2_SHIFT:=23): 'AMD_KERNEL_CODE_PROPERTIES_RESERVED2_SHIFT', (AMD_KERNEL_CODE_PROPERTIES_RESERVED2_WIDTH:=9): 'AMD_KERNEL_CODE_PROPERTIES_RESERVED2_WIDTH', (AMD_KERNEL_CODE_PROPERTIES_RESERVED2:=-8388608): 'AMD_KERNEL_CODE_PROPERTIES_RESERVED2'}
amd_powertwo8_t: TypeAlias = ctypes.c_ubyte
enum_amd_powertwo_t: dict[int, str] = {(AMD_POWERTWO_1:=0): 'AMD_POWERTWO_1', (AMD_POWERTWO_2:=1): 'AMD_POWERTWO_2', (AMD_POWERTWO_4:=2): 'AMD_POWERTWO_4', (AMD_POWERTWO_8:=3): 'AMD_POWERTWO_8', (AMD_POWERTWO_16:=4): 'AMD_POWERTWO_16', (AMD_POWERTWO_32:=5): 'AMD_POWERTWO_32', (AMD_POWERTWO_64:=6): 'AMD_POWERTWO_64', (AMD_POWERTWO_128:=7): 'AMD_POWERTWO_128', (AMD_POWERTWO_256:=8): 'AMD_POWERTWO_256'}
amd_enabled_control_directive64_t: TypeAlias = ctypes.c_uint64
enum_amd_enabled_control_directive_t: dict[int, str] = {(AMD_ENABLED_CONTROL_DIRECTIVE_ENABLE_BREAK_EXCEPTIONS:=1): 'AMD_ENABLED_CONTROL_DIRECTIVE_ENABLE_BREAK_EXCEPTIONS', (AMD_ENABLED_CONTROL_DIRECTIVE_ENABLE_DETECT_EXCEPTIONS:=2): 'AMD_ENABLED_CONTROL_DIRECTIVE_ENABLE_DETECT_EXCEPTIONS', (AMD_ENABLED_CONTROL_DIRECTIVE_MAX_DYNAMIC_GROUP_SIZE:=4): 'AMD_ENABLED_CONTROL_DIRECTIVE_MAX_DYNAMIC_GROUP_SIZE', (AMD_ENABLED_CONTROL_DIRECTIVE_MAX_FLAT_GRID_SIZE:=8): 'AMD_ENABLED_CONTROL_DIRECTIVE_MAX_FLAT_GRID_SIZE', (AMD_ENABLED_CONTROL_DIRECTIVE_MAX_FLAT_WORKGROUP_SIZE:=16): 'AMD_ENABLED_CONTROL_DIRECTIVE_MAX_FLAT_WORKGROUP_SIZE', (AMD_ENABLED_CONTROL_DIRECTIVE_REQUIRED_DIM:=32): 'AMD_ENABLED_CONTROL_DIRECTIVE_REQUIRED_DIM', (AMD_ENABLED_CONTROL_DIRECTIVE_REQUIRED_GRID_SIZE:=64): 'AMD_ENABLED_CONTROL_DIRECTIVE_REQUIRED_GRID_SIZE', (AMD_ENABLED_CONTROL_DIRECTIVE_REQUIRED_WORKGROUP_SIZE:=128): 'AMD_ENABLED_CONTROL_DIRECTIVE_REQUIRED_WORKGROUP_SIZE', (AMD_ENABLED_CONTROL_DIRECTIVE_REQUIRE_NO_PARTIAL_WORKGROUPS:=256): 'AMD_ENABLED_CONTROL_DIRECTIVE_REQUIRE_NO_PARTIAL_WORKGROUPS'}
amd_exception_kind16_t: TypeAlias = ctypes.c_uint16
enum_amd_exception_kind_t: dict[int, str] = {(AMD_EXCEPTION_KIND_INVALID_OPERATION:=1): 'AMD_EXCEPTION_KIND_INVALID_OPERATION', (AMD_EXCEPTION_KIND_DIVISION_BY_ZERO:=2): 'AMD_EXCEPTION_KIND_DIVISION_BY_ZERO', (AMD_EXCEPTION_KIND_OVERFLOW:=4): 'AMD_EXCEPTION_KIND_OVERFLOW', (AMD_EXCEPTION_KIND_UNDERFLOW:=8): 'AMD_EXCEPTION_KIND_UNDERFLOW', (AMD_EXCEPTION_KIND_INEXACT:=16): 'AMD_EXCEPTION_KIND_INEXACT'}
@c.record
class struct_amd_control_directives_s(c.Struct):
  SIZE = 128
  enabled_control_directives: int
  enable_break_exceptions: int
  enable_detect_exceptions: int
  max_dynamic_group_size: int
  max_flat_grid_size: int
  max_flat_workgroup_size: int
  required_dim: int
  reserved1: c.Array[ctypes.c_ubyte, Literal[3]]
  required_grid_size: c.Array[ctypes.c_uint64, Literal[3]]
  required_workgroup_size: c.Array[ctypes.c_uint32, Literal[3]]
  reserved2: c.Array[ctypes.c_ubyte, Literal[60]]
struct_amd_control_directives_s.register_fields([('enabled_control_directives', amd_enabled_control_directive64_t, 0), ('enable_break_exceptions', uint16_t, 8), ('enable_detect_exceptions', uint16_t, 10), ('max_dynamic_group_size', uint32_t, 12), ('max_flat_grid_size', uint64_t, 16), ('max_flat_workgroup_size', uint32_t, 24), ('required_dim', uint8_t, 28), ('reserved1', c.Array[uint8_t, Literal[3]], 29), ('required_grid_size', c.Array[uint64_t, Literal[3]], 32), ('required_workgroup_size', c.Array[uint32_t, Literal[3]], 56), ('reserved2', c.Array[uint8_t, Literal[60]], 68)])
amd_control_directives_t: TypeAlias = struct_amd_control_directives_s
@c.record
class struct_amd_kernel_code_s(c.Struct):
  SIZE = 256
  amd_kernel_code_version_major: int
  amd_kernel_code_version_minor: int
  amd_machine_kind: int
  amd_machine_version_major: int
  amd_machine_version_minor: int
  amd_machine_version_stepping: int
  kernel_code_entry_byte_offset: int
  kernel_code_prefetch_byte_offset: int
  kernel_code_prefetch_byte_size: int
  max_scratch_backing_memory_byte_size: int
  compute_pgm_rsrc1: int
  compute_pgm_rsrc2: int
  kernel_code_properties: int
  workitem_private_segment_byte_size: int
  workgroup_group_segment_byte_size: int
  gds_segment_byte_size: int
  kernarg_segment_byte_size: int
  workgroup_fbarrier_count: int
  wavefront_sgpr_count: int
  workitem_vgpr_count: int
  reserved_vgpr_first: int
  reserved_vgpr_count: int
  reserved_sgpr_first: int
  reserved_sgpr_count: int
  debug_wavefront_private_segment_offset_sgpr: int
  debug_private_segment_buffer_sgpr: int
  kernarg_segment_alignment: int
  group_segment_alignment: int
  private_segment_alignment: int
  wavefront_size: int
  call_convention: int
  reserved1: c.Array[ctypes.c_ubyte, Literal[12]]
  runtime_loader_kernel_symbol: int
  control_directives: struct_amd_control_directives_s
struct_amd_kernel_code_s.register_fields([('amd_kernel_code_version_major', amd_kernel_code_version32_t, 0), ('amd_kernel_code_version_minor', amd_kernel_code_version32_t, 4), ('amd_machine_kind', amd_machine_kind16_t, 8), ('amd_machine_version_major', amd_machine_version16_t, 10), ('amd_machine_version_minor', amd_machine_version16_t, 12), ('amd_machine_version_stepping', amd_machine_version16_t, 14), ('kernel_code_entry_byte_offset', int64_t, 16), ('kernel_code_prefetch_byte_offset', int64_t, 24), ('kernel_code_prefetch_byte_size', uint64_t, 32), ('max_scratch_backing_memory_byte_size', uint64_t, 40), ('compute_pgm_rsrc1', amd_compute_pgm_rsrc_one32_t, 48), ('compute_pgm_rsrc2', amd_compute_pgm_rsrc_two32_t, 52), ('kernel_code_properties', amd_kernel_code_properties32_t, 56), ('workitem_private_segment_byte_size', uint32_t, 60), ('workgroup_group_segment_byte_size', uint32_t, 64), ('gds_segment_byte_size', uint32_t, 68), ('kernarg_segment_byte_size', uint64_t, 72), ('workgroup_fbarrier_count', uint32_t, 80), ('wavefront_sgpr_count', uint16_t, 84), ('workitem_vgpr_count', uint16_t, 86), ('reserved_vgpr_first', uint16_t, 88), ('reserved_vgpr_count', uint16_t, 90), ('reserved_sgpr_first', uint16_t, 92), ('reserved_sgpr_count', uint16_t, 94), ('debug_wavefront_private_segment_offset_sgpr', uint16_t, 96), ('debug_private_segment_buffer_sgpr', uint16_t, 98), ('kernarg_segment_alignment', amd_powertwo8_t, 100), ('group_segment_alignment', amd_powertwo8_t, 101), ('private_segment_alignment', amd_powertwo8_t, 102), ('wavefront_size', amd_powertwo8_t, 103), ('call_convention', int32_t, 104), ('reserved1', c.Array[uint8_t, Literal[12]], 108), ('runtime_loader_kernel_symbol', uint64_t, 120), ('control_directives', amd_control_directives_t, 128)])
amd_kernel_code_t: TypeAlias = struct_amd_kernel_code_s
@c.record
class struct_amd_runtime_loader_debug_info_s(c.Struct):
  SIZE = 32
  elf_raw: ctypes.c_void_p
  elf_size: int
  kernel_name: c.POINTER[ctypes.c_char]
  owning_segment: ctypes.c_void_p
struct_amd_runtime_loader_debug_info_s.register_fields([('elf_raw', ctypes.c_void_p, 0), ('elf_size', size_t, 8), ('kernel_name', c.POINTER[ctypes.c_char], 16), ('owning_segment', ctypes.c_void_p, 24)])
amd_runtime_loader_debug_info_t: TypeAlias = struct_amd_runtime_loader_debug_info_s
class struct_BrigModuleHeader(c.Struct): pass
BrigModule_t: TypeAlias = c.POINTER[struct_BrigModuleHeader]
_anonenum1: dict[int, str] = {(HSA_EXT_STATUS_ERROR_INVALID_PROGRAM:=8192): 'HSA_EXT_STATUS_ERROR_INVALID_PROGRAM', (HSA_EXT_STATUS_ERROR_INVALID_MODULE:=8193): 'HSA_EXT_STATUS_ERROR_INVALID_MODULE', (HSA_EXT_STATUS_ERROR_INCOMPATIBLE_MODULE:=8194): 'HSA_EXT_STATUS_ERROR_INCOMPATIBLE_MODULE', (HSA_EXT_STATUS_ERROR_MODULE_ALREADY_INCLUDED:=8195): 'HSA_EXT_STATUS_ERROR_MODULE_ALREADY_INCLUDED', (HSA_EXT_STATUS_ERROR_SYMBOL_MISMATCH:=8196): 'HSA_EXT_STATUS_ERROR_SYMBOL_MISMATCH', (HSA_EXT_STATUS_ERROR_FINALIZATION_FAILED:=8197): 'HSA_EXT_STATUS_ERROR_FINALIZATION_FAILED', (HSA_EXT_STATUS_ERROR_DIRECTIVE_MISMATCH:=8198): 'HSA_EXT_STATUS_ERROR_DIRECTIVE_MISMATCH'}
hsa_ext_module_t: TypeAlias = c.POINTER[struct_BrigModuleHeader]
@c.record
class struct_hsa_ext_program_s(c.Struct):
  SIZE = 8
  handle: int
struct_hsa_ext_program_s.register_fields([('handle', uint64_t, 0)])
hsa_ext_program_t: TypeAlias = struct_hsa_ext_program_s
@dll.bind(ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, c.POINTER[ctypes.c_char], c.POINTER[hsa_ext_program_t])
def hsa_ext_program_create(machine_model:ctypes.c_uint32, profile:ctypes.c_uint32, default_float_rounding_mode:ctypes.c_uint32, options:c.POINTER[ctypes.c_char], program:c.POINTER[hsa_ext_program_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hsa_ext_program_t)
def hsa_ext_program_destroy(program:hsa_ext_program_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hsa_ext_program_t, hsa_ext_module_t)
def hsa_ext_program_add_module(program:hsa_ext_program_t, module:hsa_ext_module_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hsa_ext_program_t, c.CFUNCTYPE[ctypes.c_uint32, [hsa_ext_program_t, hsa_ext_module_t, ctypes.c_void_p]], ctypes.c_void_p)
def hsa_ext_program_iterate_modules(program:hsa_ext_program_t, callback:c.CFUNCTYPE[ctypes.c_uint32, [hsa_ext_program_t, hsa_ext_module_t, ctypes.c_void_p]], data:ctypes.c_void_p) -> ctypes.c_uint32: ...
hsa_ext_program_info_t: dict[int, str] = {(HSA_EXT_PROGRAM_INFO_MACHINE_MODEL:=0): 'HSA_EXT_PROGRAM_INFO_MACHINE_MODEL', (HSA_EXT_PROGRAM_INFO_PROFILE:=1): 'HSA_EXT_PROGRAM_INFO_PROFILE', (HSA_EXT_PROGRAM_INFO_DEFAULT_FLOAT_ROUNDING_MODE:=2): 'HSA_EXT_PROGRAM_INFO_DEFAULT_FLOAT_ROUNDING_MODE'}
@dll.bind(ctypes.c_uint32, hsa_ext_program_t, ctypes.c_uint32, ctypes.c_void_p)
def hsa_ext_program_get_info(program:hsa_ext_program_t, attribute:ctypes.c_uint32, value:ctypes.c_void_p) -> ctypes.c_uint32: ...
hsa_ext_finalizer_call_convention_t: dict[int, str] = {(HSA_EXT_FINALIZER_CALL_CONVENTION_AUTO:=-1): 'HSA_EXT_FINALIZER_CALL_CONVENTION_AUTO'}
@c.record
class struct_hsa_ext_control_directives_s(c.Struct):
  SIZE = 144
  control_directives_mask: int
  break_exceptions_mask: int
  detect_exceptions_mask: int
  max_dynamic_group_size: int
  max_flat_grid_size: int
  max_flat_workgroup_size: int
  reserved1: int
  required_grid_size: c.Array[ctypes.c_uint64, Literal[3]]
  required_workgroup_size: struct_hsa_dim3_s
  required_dim: int
  reserved2: c.Array[ctypes.c_ubyte, Literal[75]]
struct_hsa_ext_control_directives_s.register_fields([('control_directives_mask', uint64_t, 0), ('break_exceptions_mask', uint16_t, 8), ('detect_exceptions_mask', uint16_t, 10), ('max_dynamic_group_size', uint32_t, 12), ('max_flat_grid_size', uint64_t, 16), ('max_flat_workgroup_size', uint32_t, 24), ('reserved1', uint32_t, 28), ('required_grid_size', c.Array[uint64_t, Literal[3]], 32), ('required_workgroup_size', hsa_dim3_t, 56), ('required_dim', uint8_t, 68), ('reserved2', c.Array[uint8_t, Literal[75]], 69)])
hsa_ext_control_directives_t: TypeAlias = struct_hsa_ext_control_directives_s
@dll.bind(ctypes.c_uint32, hsa_ext_program_t, hsa_isa_t, int32_t, hsa_ext_control_directives_t, c.POINTER[ctypes.c_char], ctypes.c_uint32, c.POINTER[hsa_code_object_t])
def hsa_ext_program_finalize(program:hsa_ext_program_t, isa:hsa_isa_t, call_convention:int32_t, control_directives:hsa_ext_control_directives_t, options:c.POINTER[ctypes.c_char], code_object_type:ctypes.c_uint32, code_object:c.POINTER[hsa_code_object_t]) -> ctypes.c_uint32: ...
@c.record
class struct_hsa_ext_finalizer_1_00_pfn_s(c.Struct):
  SIZE = 48
  hsa_ext_program_create: c.CFUNCTYPE[ctypes.c_uint32, [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, c.POINTER[ctypes.c_char], c.POINTER[struct_hsa_ext_program_s]]]
  hsa_ext_program_destroy: c.CFUNCTYPE[ctypes.c_uint32, [struct_hsa_ext_program_s]]
  hsa_ext_program_add_module: c.CFUNCTYPE[ctypes.c_uint32, [struct_hsa_ext_program_s, c.POINTER[struct_BrigModuleHeader]]]
  hsa_ext_program_iterate_modules: c.CFUNCTYPE[ctypes.c_uint32, [struct_hsa_ext_program_s, c.CFUNCTYPE[ctypes.c_uint32, [struct_hsa_ext_program_s, c.POINTER[struct_BrigModuleHeader], ctypes.c_void_p]], ctypes.c_void_p]]
  hsa_ext_program_get_info: c.CFUNCTYPE[ctypes.c_uint32, [struct_hsa_ext_program_s, ctypes.c_uint32, ctypes.c_void_p]]
  hsa_ext_program_finalize: c.CFUNCTYPE[ctypes.c_uint32, [struct_hsa_ext_program_s, struct_hsa_isa_s, ctypes.c_int32, struct_hsa_ext_control_directives_s, c.POINTER[ctypes.c_char], ctypes.c_uint32, c.POINTER[struct_hsa_code_object_s]]]
struct_hsa_ext_finalizer_1_00_pfn_s.register_fields([('hsa_ext_program_create', c.CFUNCTYPE[ctypes.c_uint32, [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, c.POINTER[ctypes.c_char], c.POINTER[hsa_ext_program_t]]], 0), ('hsa_ext_program_destroy', c.CFUNCTYPE[ctypes.c_uint32, [hsa_ext_program_t]], 8), ('hsa_ext_program_add_module', c.CFUNCTYPE[ctypes.c_uint32, [hsa_ext_program_t, hsa_ext_module_t]], 16), ('hsa_ext_program_iterate_modules', c.CFUNCTYPE[ctypes.c_uint32, [hsa_ext_program_t, c.CFUNCTYPE[ctypes.c_uint32, [hsa_ext_program_t, hsa_ext_module_t, ctypes.c_void_p]], ctypes.c_void_p]], 24), ('hsa_ext_program_get_info', c.CFUNCTYPE[ctypes.c_uint32, [hsa_ext_program_t, ctypes.c_uint32, ctypes.c_void_p]], 32), ('hsa_ext_program_finalize', c.CFUNCTYPE[ctypes.c_uint32, [hsa_ext_program_t, hsa_isa_t, int32_t, hsa_ext_control_directives_t, c.POINTER[ctypes.c_char], ctypes.c_uint32, c.POINTER[hsa_code_object_t]]], 40)])
hsa_ext_finalizer_1_00_pfn_t: TypeAlias = struct_hsa_ext_finalizer_1_00_pfn_s
_anonenum2: dict[int, str] = {(HSA_EXT_STATUS_ERROR_IMAGE_FORMAT_UNSUPPORTED:=12288): 'HSA_EXT_STATUS_ERROR_IMAGE_FORMAT_UNSUPPORTED', (HSA_EXT_STATUS_ERROR_IMAGE_SIZE_UNSUPPORTED:=12289): 'HSA_EXT_STATUS_ERROR_IMAGE_SIZE_UNSUPPORTED', (HSA_EXT_STATUS_ERROR_IMAGE_PITCH_UNSUPPORTED:=12290): 'HSA_EXT_STATUS_ERROR_IMAGE_PITCH_UNSUPPORTED', (HSA_EXT_STATUS_ERROR_SAMPLER_DESCRIPTOR_UNSUPPORTED:=12291): 'HSA_EXT_STATUS_ERROR_SAMPLER_DESCRIPTOR_UNSUPPORTED'}
_anonenum3: dict[int, str] = {(HSA_EXT_AGENT_INFO_IMAGE_1D_MAX_ELEMENTS:=12288): 'HSA_EXT_AGENT_INFO_IMAGE_1D_MAX_ELEMENTS', (HSA_EXT_AGENT_INFO_IMAGE_1DA_MAX_ELEMENTS:=12289): 'HSA_EXT_AGENT_INFO_IMAGE_1DA_MAX_ELEMENTS', (HSA_EXT_AGENT_INFO_IMAGE_1DB_MAX_ELEMENTS:=12290): 'HSA_EXT_AGENT_INFO_IMAGE_1DB_MAX_ELEMENTS', (HSA_EXT_AGENT_INFO_IMAGE_2D_MAX_ELEMENTS:=12291): 'HSA_EXT_AGENT_INFO_IMAGE_2D_MAX_ELEMENTS', (HSA_EXT_AGENT_INFO_IMAGE_2DA_MAX_ELEMENTS:=12292): 'HSA_EXT_AGENT_INFO_IMAGE_2DA_MAX_ELEMENTS', (HSA_EXT_AGENT_INFO_IMAGE_2DDEPTH_MAX_ELEMENTS:=12293): 'HSA_EXT_AGENT_INFO_IMAGE_2DDEPTH_MAX_ELEMENTS', (HSA_EXT_AGENT_INFO_IMAGE_2DADEPTH_MAX_ELEMENTS:=12294): 'HSA_EXT_AGENT_INFO_IMAGE_2DADEPTH_MAX_ELEMENTS', (HSA_EXT_AGENT_INFO_IMAGE_3D_MAX_ELEMENTS:=12295): 'HSA_EXT_AGENT_INFO_IMAGE_3D_MAX_ELEMENTS', (HSA_EXT_AGENT_INFO_IMAGE_ARRAY_MAX_LAYERS:=12296): 'HSA_EXT_AGENT_INFO_IMAGE_ARRAY_MAX_LAYERS', (HSA_EXT_AGENT_INFO_MAX_IMAGE_RD_HANDLES:=12297): 'HSA_EXT_AGENT_INFO_MAX_IMAGE_RD_HANDLES', (HSA_EXT_AGENT_INFO_MAX_IMAGE_RORW_HANDLES:=12298): 'HSA_EXT_AGENT_INFO_MAX_IMAGE_RORW_HANDLES', (HSA_EXT_AGENT_INFO_MAX_SAMPLER_HANDLERS:=12299): 'HSA_EXT_AGENT_INFO_MAX_SAMPLER_HANDLERS', (HSA_EXT_AGENT_INFO_IMAGE_LINEAR_ROW_PITCH_ALIGNMENT:=12300): 'HSA_EXT_AGENT_INFO_IMAGE_LINEAR_ROW_PITCH_ALIGNMENT'}
hsa_ext_image_channel_type_t: dict[int, str] = {(HSA_EXT_IMAGE_CHANNEL_TYPE_SNORM_INT8:=0): 'HSA_EXT_IMAGE_CHANNEL_TYPE_SNORM_INT8', (HSA_EXT_IMAGE_CHANNEL_TYPE_SNORM_INT16:=1): 'HSA_EXT_IMAGE_CHANNEL_TYPE_SNORM_INT16', (HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT8:=2): 'HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT8', (HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT16:=3): 'HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT16', (HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT24:=4): 'HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT24', (HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_555:=5): 'HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_555', (HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_565:=6): 'HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_565', (HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_101010:=7): 'HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_101010', (HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT8:=8): 'HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT8', (HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT16:=9): 'HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT16', (HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT32:=10): 'HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT32', (HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8:=11): 'HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8', (HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16:=12): 'HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16', (HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32:=13): 'HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32', (HSA_EXT_IMAGE_CHANNEL_TYPE_HALF_FLOAT:=14): 'HSA_EXT_IMAGE_CHANNEL_TYPE_HALF_FLOAT', (HSA_EXT_IMAGE_CHANNEL_TYPE_FLOAT:=15): 'HSA_EXT_IMAGE_CHANNEL_TYPE_FLOAT'}
hsa_ext_image_channel_order_t: dict[int, str] = {(HSA_EXT_IMAGE_CHANNEL_ORDER_A:=0): 'HSA_EXT_IMAGE_CHANNEL_ORDER_A', (HSA_EXT_IMAGE_CHANNEL_ORDER_R:=1): 'HSA_EXT_IMAGE_CHANNEL_ORDER_R', (HSA_EXT_IMAGE_CHANNEL_ORDER_RX:=2): 'HSA_EXT_IMAGE_CHANNEL_ORDER_RX', (HSA_EXT_IMAGE_CHANNEL_ORDER_RG:=3): 'HSA_EXT_IMAGE_CHANNEL_ORDER_RG', (HSA_EXT_IMAGE_CHANNEL_ORDER_RGX:=4): 'HSA_EXT_IMAGE_CHANNEL_ORDER_RGX', (HSA_EXT_IMAGE_CHANNEL_ORDER_RA:=5): 'HSA_EXT_IMAGE_CHANNEL_ORDER_RA', (HSA_EXT_IMAGE_CHANNEL_ORDER_RGB:=6): 'HSA_EXT_IMAGE_CHANNEL_ORDER_RGB', (HSA_EXT_IMAGE_CHANNEL_ORDER_RGBX:=7): 'HSA_EXT_IMAGE_CHANNEL_ORDER_RGBX', (HSA_EXT_IMAGE_CHANNEL_ORDER_RGBA:=8): 'HSA_EXT_IMAGE_CHANNEL_ORDER_RGBA', (HSA_EXT_IMAGE_CHANNEL_ORDER_BGRA:=9): 'HSA_EXT_IMAGE_CHANNEL_ORDER_BGRA', (HSA_EXT_IMAGE_CHANNEL_ORDER_ARGB:=10): 'HSA_EXT_IMAGE_CHANNEL_ORDER_ARGB', (HSA_EXT_IMAGE_CHANNEL_ORDER_ABGR:=11): 'HSA_EXT_IMAGE_CHANNEL_ORDER_ABGR', (HSA_EXT_IMAGE_CHANNEL_ORDER_SRGB:=12): 'HSA_EXT_IMAGE_CHANNEL_ORDER_SRGB', (HSA_EXT_IMAGE_CHANNEL_ORDER_SRGBX:=13): 'HSA_EXT_IMAGE_CHANNEL_ORDER_SRGBX', (HSA_EXT_IMAGE_CHANNEL_ORDER_SRGBA:=14): 'HSA_EXT_IMAGE_CHANNEL_ORDER_SRGBA', (HSA_EXT_IMAGE_CHANNEL_ORDER_SBGRA:=15): 'HSA_EXT_IMAGE_CHANNEL_ORDER_SBGRA', (HSA_EXT_IMAGE_CHANNEL_ORDER_INTENSITY:=16): 'HSA_EXT_IMAGE_CHANNEL_ORDER_INTENSITY', (HSA_EXT_IMAGE_CHANNEL_ORDER_LUMINANCE:=17): 'HSA_EXT_IMAGE_CHANNEL_ORDER_LUMINANCE', (HSA_EXT_IMAGE_CHANNEL_ORDER_DEPTH:=18): 'HSA_EXT_IMAGE_CHANNEL_ORDER_DEPTH', (HSA_EXT_IMAGE_CHANNEL_ORDER_DEPTH_STENCIL:=19): 'HSA_EXT_IMAGE_CHANNEL_ORDER_DEPTH_STENCIL'}
hsa_ext_image_capability_t: dict[int, str] = {(HSA_EXT_IMAGE_CAPABILITY_NOT_SUPPORTED:=0): 'HSA_EXT_IMAGE_CAPABILITY_NOT_SUPPORTED', (HSA_EXT_IMAGE_CAPABILITY_READ_ONLY:=1): 'HSA_EXT_IMAGE_CAPABILITY_READ_ONLY', (HSA_EXT_IMAGE_CAPABILITY_WRITE_ONLY:=2): 'HSA_EXT_IMAGE_CAPABILITY_WRITE_ONLY', (HSA_EXT_IMAGE_CAPABILITY_READ_WRITE:=4): 'HSA_EXT_IMAGE_CAPABILITY_READ_WRITE', (HSA_EXT_IMAGE_CAPABILITY_READ_MODIFY_WRITE:=8): 'HSA_EXT_IMAGE_CAPABILITY_READ_MODIFY_WRITE', (HSA_EXT_IMAGE_CAPABILITY_ACCESS_INVARIANT_DATA_LAYOUT:=16): 'HSA_EXT_IMAGE_CAPABILITY_ACCESS_INVARIANT_DATA_LAYOUT'}
hsa_ext_image_data_layout_t: dict[int, str] = {(HSA_EXT_IMAGE_DATA_LAYOUT_OPAQUE:=0): 'HSA_EXT_IMAGE_DATA_LAYOUT_OPAQUE', (HSA_EXT_IMAGE_DATA_LAYOUT_LINEAR:=1): 'HSA_EXT_IMAGE_DATA_LAYOUT_LINEAR'}
@dll.bind(ctypes.c_uint32, hsa_agent_t, ctypes.c_uint32, c.POINTER[hsa_ext_image_format_t], c.POINTER[uint32_t])
def hsa_ext_image_get_capability(agent:hsa_agent_t, geometry:ctypes.c_uint32, image_format:c.POINTER[hsa_ext_image_format_t], capability_mask:c.POINTER[uint32_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hsa_agent_t, ctypes.c_uint32, c.POINTER[hsa_ext_image_format_t], ctypes.c_uint32, c.POINTER[uint32_t])
def hsa_ext_image_get_capability_with_layout(agent:hsa_agent_t, geometry:ctypes.c_uint32, image_format:c.POINTER[hsa_ext_image_format_t], image_data_layout:ctypes.c_uint32, capability_mask:c.POINTER[uint32_t]) -> ctypes.c_uint32: ...
@c.record
class struct_hsa_ext_image_data_info_s(c.Struct):
  SIZE = 16
  size: int
  alignment: int
struct_hsa_ext_image_data_info_s.register_fields([('size', size_t, 0), ('alignment', size_t, 8)])
hsa_ext_image_data_info_t: TypeAlias = struct_hsa_ext_image_data_info_s
@dll.bind(ctypes.c_uint32, hsa_agent_t, c.POINTER[hsa_ext_image_descriptor_t], ctypes.c_uint32, c.POINTER[hsa_ext_image_data_info_t])
def hsa_ext_image_data_get_info(agent:hsa_agent_t, image_descriptor:c.POINTER[hsa_ext_image_descriptor_t], access_permission:ctypes.c_uint32, image_data_info:c.POINTER[hsa_ext_image_data_info_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hsa_agent_t, c.POINTER[hsa_ext_image_descriptor_t], ctypes.c_uint32, ctypes.c_uint32, size_t, size_t, c.POINTER[hsa_ext_image_data_info_t])
def hsa_ext_image_data_get_info_with_layout(agent:hsa_agent_t, image_descriptor:c.POINTER[hsa_ext_image_descriptor_t], access_permission:ctypes.c_uint32, image_data_layout:ctypes.c_uint32, image_data_row_pitch:size_t, image_data_slice_pitch:size_t, image_data_info:c.POINTER[hsa_ext_image_data_info_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hsa_agent_t, c.POINTER[hsa_ext_image_descriptor_t], ctypes.c_void_p, ctypes.c_uint32, c.POINTER[hsa_ext_image_t])
def hsa_ext_image_create(agent:hsa_agent_t, image_descriptor:c.POINTER[hsa_ext_image_descriptor_t], image_data:ctypes.c_void_p, access_permission:ctypes.c_uint32, image:c.POINTER[hsa_ext_image_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hsa_agent_t, c.POINTER[hsa_ext_image_descriptor_t], ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, size_t, size_t, c.POINTER[hsa_ext_image_t])
def hsa_ext_image_create_with_layout(agent:hsa_agent_t, image_descriptor:c.POINTER[hsa_ext_image_descriptor_t], image_data:ctypes.c_void_p, access_permission:ctypes.c_uint32, image_data_layout:ctypes.c_uint32, image_data_row_pitch:size_t, image_data_slice_pitch:size_t, image:c.POINTER[hsa_ext_image_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hsa_agent_t, hsa_ext_image_t)
def hsa_ext_image_destroy(agent:hsa_agent_t, image:hsa_ext_image_t) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hsa_agent_t, hsa_ext_image_t, c.POINTER[hsa_dim3_t], hsa_ext_image_t, c.POINTER[hsa_dim3_t], c.POINTER[hsa_dim3_t])
def hsa_ext_image_copy(agent:hsa_agent_t, src_image:hsa_ext_image_t, src_offset:c.POINTER[hsa_dim3_t], dst_image:hsa_ext_image_t, dst_offset:c.POINTER[hsa_dim3_t], range:c.POINTER[hsa_dim3_t]) -> ctypes.c_uint32: ...
@c.record
class struct_hsa_ext_image_region_s(c.Struct):
  SIZE = 24
  offset: struct_hsa_dim3_s
  range: struct_hsa_dim3_s
struct_hsa_ext_image_region_s.register_fields([('offset', hsa_dim3_t, 0), ('range', hsa_dim3_t, 12)])
hsa_ext_image_region_t: TypeAlias = struct_hsa_ext_image_region_s
@dll.bind(ctypes.c_uint32, hsa_agent_t, ctypes.c_void_p, size_t, size_t, hsa_ext_image_t, c.POINTER[hsa_ext_image_region_t])
def hsa_ext_image_import(agent:hsa_agent_t, src_memory:ctypes.c_void_p, src_row_pitch:size_t, src_slice_pitch:size_t, dst_image:hsa_ext_image_t, image_region:c.POINTER[hsa_ext_image_region_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hsa_agent_t, hsa_ext_image_t, ctypes.c_void_p, size_t, size_t, c.POINTER[hsa_ext_image_region_t])
def hsa_ext_image_export(agent:hsa_agent_t, src_image:hsa_ext_image_t, dst_memory:ctypes.c_void_p, dst_row_pitch:size_t, dst_slice_pitch:size_t, image_region:c.POINTER[hsa_ext_image_region_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hsa_agent_t, hsa_ext_image_t, ctypes.c_void_p, c.POINTER[hsa_ext_image_region_t])
def hsa_ext_image_clear(agent:hsa_agent_t, image:hsa_ext_image_t, data:ctypes.c_void_p, image_region:c.POINTER[hsa_ext_image_region_t]) -> ctypes.c_uint32: ...
@c.record
class struct_hsa_ext_sampler_s(c.Struct):
  SIZE = 8
  handle: int
struct_hsa_ext_sampler_s.register_fields([('handle', uint64_t, 0)])
hsa_ext_sampler_t: TypeAlias = struct_hsa_ext_sampler_s
hsa_ext_sampler_addressing_mode_t: dict[int, str] = {(HSA_EXT_SAMPLER_ADDRESSING_MODE_UNDEFINED:=0): 'HSA_EXT_SAMPLER_ADDRESSING_MODE_UNDEFINED', (HSA_EXT_SAMPLER_ADDRESSING_MODE_CLAMP_TO_EDGE:=1): 'HSA_EXT_SAMPLER_ADDRESSING_MODE_CLAMP_TO_EDGE', (HSA_EXT_SAMPLER_ADDRESSING_MODE_CLAMP_TO_BORDER:=2): 'HSA_EXT_SAMPLER_ADDRESSING_MODE_CLAMP_TO_BORDER', (HSA_EXT_SAMPLER_ADDRESSING_MODE_REPEAT:=3): 'HSA_EXT_SAMPLER_ADDRESSING_MODE_REPEAT', (HSA_EXT_SAMPLER_ADDRESSING_MODE_MIRRORED_REPEAT:=4): 'HSA_EXT_SAMPLER_ADDRESSING_MODE_MIRRORED_REPEAT'}
hsa_ext_sampler_addressing_mode32_t: TypeAlias = ctypes.c_uint32
hsa_ext_sampler_coordinate_mode_t: dict[int, str] = {(HSA_EXT_SAMPLER_COORDINATE_MODE_UNNORMALIZED:=0): 'HSA_EXT_SAMPLER_COORDINATE_MODE_UNNORMALIZED', (HSA_EXT_SAMPLER_COORDINATE_MODE_NORMALIZED:=1): 'HSA_EXT_SAMPLER_COORDINATE_MODE_NORMALIZED'}
hsa_ext_sampler_coordinate_mode32_t: TypeAlias = ctypes.c_uint32
hsa_ext_sampler_filter_mode_t: dict[int, str] = {(HSA_EXT_SAMPLER_FILTER_MODE_NEAREST:=0): 'HSA_EXT_SAMPLER_FILTER_MODE_NEAREST', (HSA_EXT_SAMPLER_FILTER_MODE_LINEAR:=1): 'HSA_EXT_SAMPLER_FILTER_MODE_LINEAR'}
hsa_ext_sampler_filter_mode32_t: TypeAlias = ctypes.c_uint32
@c.record
class struct_hsa_ext_sampler_descriptor_s(c.Struct):
  SIZE = 12
  coordinate_mode: int
  filter_mode: int
  address_mode: int
struct_hsa_ext_sampler_descriptor_s.register_fields([('coordinate_mode', hsa_ext_sampler_coordinate_mode32_t, 0), ('filter_mode', hsa_ext_sampler_filter_mode32_t, 4), ('address_mode', hsa_ext_sampler_addressing_mode32_t, 8)])
hsa_ext_sampler_descriptor_t: TypeAlias = struct_hsa_ext_sampler_descriptor_s
@c.record
class struct_hsa_ext_sampler_descriptor_v2_s(c.Struct):
  SIZE = 20
  coordinate_mode: int
  filter_mode: int
  address_modes: c.Array[ctypes.c_uint32, Literal[3]]
struct_hsa_ext_sampler_descriptor_v2_s.register_fields([('coordinate_mode', hsa_ext_sampler_coordinate_mode32_t, 0), ('filter_mode', hsa_ext_sampler_filter_mode32_t, 4), ('address_modes', c.Array[hsa_ext_sampler_addressing_mode32_t, Literal[3]], 8)])
hsa_ext_sampler_descriptor_v2_t: TypeAlias = struct_hsa_ext_sampler_descriptor_v2_s
@dll.bind(ctypes.c_uint32, hsa_agent_t, c.POINTER[hsa_ext_sampler_descriptor_t], c.POINTER[hsa_ext_sampler_t])
def hsa_ext_sampler_create(agent:hsa_agent_t, sampler_descriptor:c.POINTER[hsa_ext_sampler_descriptor_t], sampler:c.POINTER[hsa_ext_sampler_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hsa_agent_t, c.POINTER[hsa_ext_sampler_descriptor_v2_t], c.POINTER[hsa_ext_sampler_t])
def hsa_ext_sampler_create_v2(agent:hsa_agent_t, sampler_descriptor:c.POINTER[hsa_ext_sampler_descriptor_v2_t], sampler:c.POINTER[hsa_ext_sampler_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, hsa_agent_t, hsa_ext_sampler_t)
def hsa_ext_sampler_destroy(agent:hsa_agent_t, sampler:hsa_ext_sampler_t) -> ctypes.c_uint32: ...
@c.record
class struct_hsa_ext_images_1_00_pfn_s(c.Struct):
  SIZE = 80
  hsa_ext_image_get_capability: c.CFUNCTYPE[ctypes.c_uint32, [struct_hsa_agent_s, ctypes.c_uint32, c.POINTER[struct_hsa_ext_image_format_s], c.POINTER[ctypes.c_uint32]]]
  hsa_ext_image_data_get_info: c.CFUNCTYPE[ctypes.c_uint32, [struct_hsa_agent_s, c.POINTER[struct_hsa_ext_image_descriptor_s], ctypes.c_uint32, c.POINTER[struct_hsa_ext_image_data_info_s]]]
  hsa_ext_image_create: c.CFUNCTYPE[ctypes.c_uint32, [struct_hsa_agent_s, c.POINTER[struct_hsa_ext_image_descriptor_s], ctypes.c_void_p, ctypes.c_uint32, c.POINTER[struct_hsa_ext_image_s]]]
  hsa_ext_image_destroy: c.CFUNCTYPE[ctypes.c_uint32, [struct_hsa_agent_s, struct_hsa_ext_image_s]]
  hsa_ext_image_copy: c.CFUNCTYPE[ctypes.c_uint32, [struct_hsa_agent_s, struct_hsa_ext_image_s, c.POINTER[struct_hsa_dim3_s], struct_hsa_ext_image_s, c.POINTER[struct_hsa_dim3_s], c.POINTER[struct_hsa_dim3_s]]]
  hsa_ext_image_import: c.CFUNCTYPE[ctypes.c_uint32, [struct_hsa_agent_s, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64, struct_hsa_ext_image_s, c.POINTER[struct_hsa_ext_image_region_s]]]
  hsa_ext_image_export: c.CFUNCTYPE[ctypes.c_uint32, [struct_hsa_agent_s, struct_hsa_ext_image_s, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64, c.POINTER[struct_hsa_ext_image_region_s]]]
  hsa_ext_image_clear: c.CFUNCTYPE[ctypes.c_uint32, [struct_hsa_agent_s, struct_hsa_ext_image_s, ctypes.c_void_p, c.POINTER[struct_hsa_ext_image_region_s]]]
  hsa_ext_sampler_create: c.CFUNCTYPE[ctypes.c_uint32, [struct_hsa_agent_s, c.POINTER[struct_hsa_ext_sampler_descriptor_s], c.POINTER[struct_hsa_ext_sampler_s]]]
  hsa_ext_sampler_destroy: c.CFUNCTYPE[ctypes.c_uint32, [struct_hsa_agent_s, struct_hsa_ext_sampler_s]]
struct_hsa_ext_images_1_00_pfn_s.register_fields([('hsa_ext_image_get_capability', c.CFUNCTYPE[ctypes.c_uint32, [hsa_agent_t, ctypes.c_uint32, c.POINTER[hsa_ext_image_format_t], c.POINTER[uint32_t]]], 0), ('hsa_ext_image_data_get_info', c.CFUNCTYPE[ctypes.c_uint32, [hsa_agent_t, c.POINTER[hsa_ext_image_descriptor_t], ctypes.c_uint32, c.POINTER[hsa_ext_image_data_info_t]]], 8), ('hsa_ext_image_create', c.CFUNCTYPE[ctypes.c_uint32, [hsa_agent_t, c.POINTER[hsa_ext_image_descriptor_t], ctypes.c_void_p, ctypes.c_uint32, c.POINTER[hsa_ext_image_t]]], 16), ('hsa_ext_image_destroy', c.CFUNCTYPE[ctypes.c_uint32, [hsa_agent_t, hsa_ext_image_t]], 24), ('hsa_ext_image_copy', c.CFUNCTYPE[ctypes.c_uint32, [hsa_agent_t, hsa_ext_image_t, c.POINTER[hsa_dim3_t], hsa_ext_image_t, c.POINTER[hsa_dim3_t], c.POINTER[hsa_dim3_t]]], 32), ('hsa_ext_image_import', c.CFUNCTYPE[ctypes.c_uint32, [hsa_agent_t, ctypes.c_void_p, size_t, size_t, hsa_ext_image_t, c.POINTER[hsa_ext_image_region_t]]], 40), ('hsa_ext_image_export', c.CFUNCTYPE[ctypes.c_uint32, [hsa_agent_t, hsa_ext_image_t, ctypes.c_void_p, size_t, size_t, c.POINTER[hsa_ext_image_region_t]]], 48), ('hsa_ext_image_clear', c.CFUNCTYPE[ctypes.c_uint32, [hsa_agent_t, hsa_ext_image_t, ctypes.c_void_p, c.POINTER[hsa_ext_image_region_t]]], 56), ('hsa_ext_sampler_create', c.CFUNCTYPE[ctypes.c_uint32, [hsa_agent_t, c.POINTER[hsa_ext_sampler_descriptor_t], c.POINTER[hsa_ext_sampler_t]]], 64), ('hsa_ext_sampler_destroy', c.CFUNCTYPE[ctypes.c_uint32, [hsa_agent_t, hsa_ext_sampler_t]], 72)])
hsa_ext_images_1_00_pfn_t: TypeAlias = struct_hsa_ext_images_1_00_pfn_s
@c.record
class struct_hsa_ext_images_1_pfn_s(c.Struct):
  SIZE = 112
  hsa_ext_image_get_capability: c.CFUNCTYPE[ctypes.c_uint32, [struct_hsa_agent_s, ctypes.c_uint32, c.POINTER[struct_hsa_ext_image_format_s], c.POINTER[ctypes.c_uint32]]]
  hsa_ext_image_data_get_info: c.CFUNCTYPE[ctypes.c_uint32, [struct_hsa_agent_s, c.POINTER[struct_hsa_ext_image_descriptor_s], ctypes.c_uint32, c.POINTER[struct_hsa_ext_image_data_info_s]]]
  hsa_ext_image_create: c.CFUNCTYPE[ctypes.c_uint32, [struct_hsa_agent_s, c.POINTER[struct_hsa_ext_image_descriptor_s], ctypes.c_void_p, ctypes.c_uint32, c.POINTER[struct_hsa_ext_image_s]]]
  hsa_ext_image_destroy: c.CFUNCTYPE[ctypes.c_uint32, [struct_hsa_agent_s, struct_hsa_ext_image_s]]
  hsa_ext_image_copy: c.CFUNCTYPE[ctypes.c_uint32, [struct_hsa_agent_s, struct_hsa_ext_image_s, c.POINTER[struct_hsa_dim3_s], struct_hsa_ext_image_s, c.POINTER[struct_hsa_dim3_s], c.POINTER[struct_hsa_dim3_s]]]
  hsa_ext_image_import: c.CFUNCTYPE[ctypes.c_uint32, [struct_hsa_agent_s, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64, struct_hsa_ext_image_s, c.POINTER[struct_hsa_ext_image_region_s]]]
  hsa_ext_image_export: c.CFUNCTYPE[ctypes.c_uint32, [struct_hsa_agent_s, struct_hsa_ext_image_s, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64, c.POINTER[struct_hsa_ext_image_region_s]]]
  hsa_ext_image_clear: c.CFUNCTYPE[ctypes.c_uint32, [struct_hsa_agent_s, struct_hsa_ext_image_s, ctypes.c_void_p, c.POINTER[struct_hsa_ext_image_region_s]]]
  hsa_ext_sampler_create: c.CFUNCTYPE[ctypes.c_uint32, [struct_hsa_agent_s, c.POINTER[struct_hsa_ext_sampler_descriptor_s], c.POINTER[struct_hsa_ext_sampler_s]]]
  hsa_ext_sampler_destroy: c.CFUNCTYPE[ctypes.c_uint32, [struct_hsa_agent_s, struct_hsa_ext_sampler_s]]
  hsa_ext_image_get_capability_with_layout: c.CFUNCTYPE[ctypes.c_uint32, [struct_hsa_agent_s, ctypes.c_uint32, c.POINTER[struct_hsa_ext_image_format_s], ctypes.c_uint32, c.POINTER[ctypes.c_uint32]]]
  hsa_ext_image_data_get_info_with_layout: c.CFUNCTYPE[ctypes.c_uint32, [struct_hsa_agent_s, c.POINTER[struct_hsa_ext_image_descriptor_s], ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint64, ctypes.c_uint64, c.POINTER[struct_hsa_ext_image_data_info_s]]]
  hsa_ext_image_create_with_layout: c.CFUNCTYPE[ctypes.c_uint32, [struct_hsa_agent_s, c.POINTER[struct_hsa_ext_image_descriptor_s], ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint64, ctypes.c_uint64, c.POINTER[struct_hsa_ext_image_s]]]
  hsa_ext_sampler_create_v2: c.CFUNCTYPE[ctypes.c_uint32, [struct_hsa_agent_s, c.POINTER[struct_hsa_ext_sampler_descriptor_v2_s], c.POINTER[struct_hsa_ext_sampler_s]]]
struct_hsa_ext_images_1_pfn_s.register_fields([('hsa_ext_image_get_capability', c.CFUNCTYPE[ctypes.c_uint32, [hsa_agent_t, ctypes.c_uint32, c.POINTER[hsa_ext_image_format_t], c.POINTER[uint32_t]]], 0), ('hsa_ext_image_data_get_info', c.CFUNCTYPE[ctypes.c_uint32, [hsa_agent_t, c.POINTER[hsa_ext_image_descriptor_t], ctypes.c_uint32, c.POINTER[hsa_ext_image_data_info_t]]], 8), ('hsa_ext_image_create', c.CFUNCTYPE[ctypes.c_uint32, [hsa_agent_t, c.POINTER[hsa_ext_image_descriptor_t], ctypes.c_void_p, ctypes.c_uint32, c.POINTER[hsa_ext_image_t]]], 16), ('hsa_ext_image_destroy', c.CFUNCTYPE[ctypes.c_uint32, [hsa_agent_t, hsa_ext_image_t]], 24), ('hsa_ext_image_copy', c.CFUNCTYPE[ctypes.c_uint32, [hsa_agent_t, hsa_ext_image_t, c.POINTER[hsa_dim3_t], hsa_ext_image_t, c.POINTER[hsa_dim3_t], c.POINTER[hsa_dim3_t]]], 32), ('hsa_ext_image_import', c.CFUNCTYPE[ctypes.c_uint32, [hsa_agent_t, ctypes.c_void_p, size_t, size_t, hsa_ext_image_t, c.POINTER[hsa_ext_image_region_t]]], 40), ('hsa_ext_image_export', c.CFUNCTYPE[ctypes.c_uint32, [hsa_agent_t, hsa_ext_image_t, ctypes.c_void_p, size_t, size_t, c.POINTER[hsa_ext_image_region_t]]], 48), ('hsa_ext_image_clear', c.CFUNCTYPE[ctypes.c_uint32, [hsa_agent_t, hsa_ext_image_t, ctypes.c_void_p, c.POINTER[hsa_ext_image_region_t]]], 56), ('hsa_ext_sampler_create', c.CFUNCTYPE[ctypes.c_uint32, [hsa_agent_t, c.POINTER[hsa_ext_sampler_descriptor_t], c.POINTER[hsa_ext_sampler_t]]], 64), ('hsa_ext_sampler_destroy', c.CFUNCTYPE[ctypes.c_uint32, [hsa_agent_t, hsa_ext_sampler_t]], 72), ('hsa_ext_image_get_capability_with_layout', c.CFUNCTYPE[ctypes.c_uint32, [hsa_agent_t, ctypes.c_uint32, c.POINTER[hsa_ext_image_format_t], ctypes.c_uint32, c.POINTER[uint32_t]]], 80), ('hsa_ext_image_data_get_info_with_layout', c.CFUNCTYPE[ctypes.c_uint32, [hsa_agent_t, c.POINTER[hsa_ext_image_descriptor_t], ctypes.c_uint32, ctypes.c_uint32, size_t, size_t, c.POINTER[hsa_ext_image_data_info_t]]], 88), ('hsa_ext_image_create_with_layout', c.CFUNCTYPE[ctypes.c_uint32, [hsa_agent_t, c.POINTER[hsa_ext_image_descriptor_t], ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, size_t, size_t, c.POINTER[hsa_ext_image_t]]], 96), ('hsa_ext_sampler_create_v2', c.CFUNCTYPE[ctypes.c_uint32, [hsa_agent_t, c.POINTER[hsa_ext_sampler_descriptor_v2_t], c.POINTER[hsa_ext_sampler_t]]], 104)])
hsa_ext_images_1_pfn_t: TypeAlias = struct_hsa_ext_images_1_pfn_s
@dll.bind(uint32_t)
def hsa_ven_amd_aqlprofile_version_major() -> uint32_t: ...
@dll.bind(uint32_t)
def hsa_ven_amd_aqlprofile_version_minor() -> uint32_t: ...
hsa_ven_amd_aqlprofile_event_type_t: dict[int, str] = {(HSA_VEN_AMD_AQLPROFILE_EVENT_TYPE_PMC:=0): 'HSA_VEN_AMD_AQLPROFILE_EVENT_TYPE_PMC', (HSA_VEN_AMD_AQLPROFILE_EVENT_TYPE_TRACE:=1): 'HSA_VEN_AMD_AQLPROFILE_EVENT_TYPE_TRACE'}
hsa_ven_amd_aqlprofile_block_name_t: dict[int, str] = {(HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_CPC:=0): 'HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_CPC', (HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_CPF:=1): 'HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_CPF', (HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GDS:=2): 'HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GDS', (HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GRBM:=3): 'HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GRBM', (HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GRBMSE:=4): 'HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GRBMSE', (HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SPI:=5): 'HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SPI', (HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ:=6): 'HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ', (HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQCS:=7): 'HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQCS', (HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SRBM:=8): 'HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SRBM', (HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SX:=9): 'HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SX', (HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA:=10): 'HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA', (HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCA:=11): 'HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCA', (HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCC:=12): 'HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCC', (HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP:=13): 'HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP', (HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TD:=14): 'HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TD', (HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCARB:=15): 'HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCARB', (HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCHUB:=16): 'HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCHUB', (HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCMCBVM:=17): 'HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCMCBVM', (HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCSEQ:=18): 'HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCSEQ', (HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCVML2:=19): 'HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCVML2', (HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCXBAR:=20): 'HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCXBAR', (HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_ATC:=21): 'HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_ATC', (HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_ATCL2:=22): 'HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_ATCL2', (HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GCEA:=23): 'HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GCEA', (HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_RPB:=24): 'HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_RPB', (HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SDMA:=25): 'HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SDMA', (HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GL1A:=26): 'HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GL1A', (HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GL1C:=27): 'HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GL1C', (HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GL2A:=28): 'HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GL2A', (HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GL2C:=29): 'HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GL2C', (HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GCR:=30): 'HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GCR', (HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GUS:=31): 'HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GUS', (HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_UMC:=32): 'HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_UMC', (HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MMEA:=33): 'HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MMEA', (HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER:=34): 'HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER'}
@c.record
class hsa_ven_amd_aqlprofile_event_t(c.Struct):
  SIZE = 12
  block_name: int
  block_index: int
  counter_id: int
hsa_ven_amd_aqlprofile_event_t.register_fields([('block_name', ctypes.c_uint32, 0), ('block_index', uint32_t, 4), ('counter_id', uint32_t, 8)])
@dll.bind(ctypes.c_uint32, hsa_agent_t, c.POINTER[hsa_ven_amd_aqlprofile_event_t], c.POINTER[ctypes.c_bool])
def hsa_ven_amd_aqlprofile_validate_event(agent:hsa_agent_t, event:c.POINTER[hsa_ven_amd_aqlprofile_event_t], result:c.POINTER[ctypes.c_bool]) -> ctypes.c_uint32: ...
hsa_ven_amd_aqlprofile_parameter_name_t: dict[int, str] = {(HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_COMPUTE_UNIT_TARGET:=0): 'HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_COMPUTE_UNIT_TARGET', (HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_VM_ID_MASK:=1): 'HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_VM_ID_MASK', (HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_MASK:=2): 'HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_MASK', (HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_TOKEN_MASK:=3): 'HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_TOKEN_MASK', (HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_TOKEN_MASK2:=4): 'HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_TOKEN_MASK2', (HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_SE_MASK:=5): 'HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_SE_MASK', (HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_SAMPLE_RATE:=6): 'HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_SAMPLE_RATE', (HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_K_CONCURRENT:=7): 'HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_K_CONCURRENT', (HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_SIMD_SELECTION:=8): 'HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_SIMD_SELECTION', (HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_OCCUPANCY_MODE:=9): 'HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_OCCUPANCY_MODE', (HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_ATT_BUFFER_SIZE:=10): 'HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_ATT_BUFFER_SIZE', (HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_PERFCOUNTER_MASK:=240): 'HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_PERFCOUNTER_MASK', (HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_PERFCOUNTER_CTRL:=241): 'HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_PERFCOUNTER_CTRL', (HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_PERFCOUNTER_NAME:=242): 'HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_PERFCOUNTER_NAME'}
@c.record
class hsa_ven_amd_aqlprofile_parameter_t(c.Struct):
  SIZE = 8
  parameter_name: int
  value: int
hsa_ven_amd_aqlprofile_parameter_t.register_fields([('parameter_name', ctypes.c_uint32, 0), ('value', uint32_t, 4)])
hsa_ven_amd_aqlprofile_att_marker_channel_t: dict[int, str] = {(HSA_VEN_AMD_AQLPROFILE_ATT_CHANNEL_0:=0): 'HSA_VEN_AMD_AQLPROFILE_ATT_CHANNEL_0', (HSA_VEN_AMD_AQLPROFILE_ATT_CHANNEL_1:=1): 'HSA_VEN_AMD_AQLPROFILE_ATT_CHANNEL_1', (HSA_VEN_AMD_AQLPROFILE_ATT_CHANNEL_2:=2): 'HSA_VEN_AMD_AQLPROFILE_ATT_CHANNEL_2', (HSA_VEN_AMD_AQLPROFILE_ATT_CHANNEL_3:=3): 'HSA_VEN_AMD_AQLPROFILE_ATT_CHANNEL_3'}
@c.record
class hsa_ven_amd_aqlprofile_descriptor_t(c.Struct):
  SIZE = 16
  ptr: ctypes.c_void_p
  size: int
hsa_ven_amd_aqlprofile_descriptor_t.register_fields([('ptr', ctypes.c_void_p, 0), ('size', uint32_t, 8)])
@c.record
class hsa_ven_amd_aqlprofile_profile_t(c.Struct):
  SIZE = 80
  agent: struct_hsa_agent_s
  type: int
  events: c.POINTER[hsa_ven_amd_aqlprofile_event_t]
  event_count: int
  parameters: c.POINTER[hsa_ven_amd_aqlprofile_parameter_t]
  parameter_count: int
  output_buffer: hsa_ven_amd_aqlprofile_descriptor_t
  command_buffer: hsa_ven_amd_aqlprofile_descriptor_t
hsa_ven_amd_aqlprofile_profile_t.register_fields([('agent', hsa_agent_t, 0), ('type', ctypes.c_uint32, 8), ('events', c.POINTER[hsa_ven_amd_aqlprofile_event_t], 16), ('event_count', uint32_t, 24), ('parameters', c.POINTER[hsa_ven_amd_aqlprofile_parameter_t], 32), ('parameter_count', uint32_t, 40), ('output_buffer', hsa_ven_amd_aqlprofile_descriptor_t, 48), ('command_buffer', hsa_ven_amd_aqlprofile_descriptor_t, 64)])
@c.record
class hsa_ext_amd_aql_pm4_packet_t(c.Struct):
  SIZE = 64
  header: int
  pm4_command: c.Array[ctypes.c_uint16, Literal[27]]
  completion_signal: struct_hsa_signal_s
hsa_ext_amd_aql_pm4_packet_t.register_fields([('header', uint16_t, 0), ('pm4_command', c.Array[uint16_t, Literal[27]], 2), ('completion_signal', hsa_signal_t, 56)])
@dll.bind(ctypes.c_uint32, c.POINTER[hsa_ven_amd_aqlprofile_profile_t], c.POINTER[hsa_ext_amd_aql_pm4_packet_t])
def hsa_ven_amd_aqlprofile_start(profile:c.POINTER[hsa_ven_amd_aqlprofile_profile_t], aql_start_packet:c.POINTER[hsa_ext_amd_aql_pm4_packet_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hsa_ven_amd_aqlprofile_profile_t], c.POINTER[hsa_ext_amd_aql_pm4_packet_t])
def hsa_ven_amd_aqlprofile_stop(profile:c.POINTER[hsa_ven_amd_aqlprofile_profile_t], aql_stop_packet:c.POINTER[hsa_ext_amd_aql_pm4_packet_t]) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hsa_ven_amd_aqlprofile_profile_t], c.POINTER[hsa_ext_amd_aql_pm4_packet_t])
def hsa_ven_amd_aqlprofile_read(profile:c.POINTER[hsa_ven_amd_aqlprofile_profile_t], aql_read_packet:c.POINTER[hsa_ext_amd_aql_pm4_packet_t]) -> ctypes.c_uint32: ...
try: HSA_VEN_AMD_AQLPROFILE_LEGACY_PM4_PACKET_SIZE = ctypes.c_uint32.in_dll(dll, 'HSA_VEN_AMD_AQLPROFILE_LEGACY_PM4_PACKET_SIZE') # type: ignore
except (ValueError,AttributeError): pass
@dll.bind(ctypes.c_uint32, c.POINTER[hsa_ext_amd_aql_pm4_packet_t], ctypes.c_void_p)
def hsa_ven_amd_aqlprofile_legacy_get_pm4(aql_packet:c.POINTER[hsa_ext_amd_aql_pm4_packet_t], data:ctypes.c_void_p) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hsa_ven_amd_aqlprofile_profile_t], c.POINTER[hsa_ext_amd_aql_pm4_packet_t], uint32_t, ctypes.c_uint32)
def hsa_ven_amd_aqlprofile_att_marker(profile:c.POINTER[hsa_ven_amd_aqlprofile_profile_t], aql_marker_packet:c.POINTER[hsa_ext_amd_aql_pm4_packet_t], data:uint32_t, channel:ctypes.c_uint32) -> ctypes.c_uint32: ...
@c.record
class hsa_ven_amd_aqlprofile_info_data_t(c.Struct):
  SIZE = 32
  sample_id: int
  pmc_data: hsa_ven_amd_aqlprofile_info_data_t_pmc_data
  trace_data: hsa_ven_amd_aqlprofile_descriptor_t
@c.record
class hsa_ven_amd_aqlprofile_info_data_t_pmc_data(c.Struct):
  SIZE = 24
  event: hsa_ven_amd_aqlprofile_event_t
  result: int
hsa_ven_amd_aqlprofile_info_data_t_pmc_data.register_fields([('event', hsa_ven_amd_aqlprofile_event_t, 0), ('result', uint64_t, 16)])
hsa_ven_amd_aqlprofile_info_data_t.register_fields([('sample_id', uint32_t, 0), ('pmc_data', hsa_ven_amd_aqlprofile_info_data_t_pmc_data, 8), ('trace_data', hsa_ven_amd_aqlprofile_descriptor_t, 8)])
@c.record
class hsa_ven_amd_aqlprofile_id_query_t(c.Struct):
  SIZE = 16
  name: c.POINTER[ctypes.c_char]
  id: int
  instance_count: int
hsa_ven_amd_aqlprofile_id_query_t.register_fields([('name', c.POINTER[ctypes.c_char], 0), ('id', uint32_t, 8), ('instance_count', uint32_t, 12)])
hsa_ven_amd_aqlprofile_info_type_t: dict[int, str] = {(HSA_VEN_AMD_AQLPROFILE_INFO_COMMAND_BUFFER_SIZE:=0): 'HSA_VEN_AMD_AQLPROFILE_INFO_COMMAND_BUFFER_SIZE', (HSA_VEN_AMD_AQLPROFILE_INFO_PMC_DATA_SIZE:=1): 'HSA_VEN_AMD_AQLPROFILE_INFO_PMC_DATA_SIZE', (HSA_VEN_AMD_AQLPROFILE_INFO_PMC_DATA:=2): 'HSA_VEN_AMD_AQLPROFILE_INFO_PMC_DATA', (HSA_VEN_AMD_AQLPROFILE_INFO_TRACE_DATA:=3): 'HSA_VEN_AMD_AQLPROFILE_INFO_TRACE_DATA', (HSA_VEN_AMD_AQLPROFILE_INFO_BLOCK_COUNTERS:=4): 'HSA_VEN_AMD_AQLPROFILE_INFO_BLOCK_COUNTERS', (HSA_VEN_AMD_AQLPROFILE_INFO_BLOCK_ID:=5): 'HSA_VEN_AMD_AQLPROFILE_INFO_BLOCK_ID', (HSA_VEN_AMD_AQLPROFILE_INFO_ENABLE_CMD:=6): 'HSA_VEN_AMD_AQLPROFILE_INFO_ENABLE_CMD', (HSA_VEN_AMD_AQLPROFILE_INFO_DISABLE_CMD:=7): 'HSA_VEN_AMD_AQLPROFILE_INFO_DISABLE_CMD'}
hsa_ven_amd_aqlprofile_data_callback_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint32, [ctypes.c_uint32, c.POINTER[hsa_ven_amd_aqlprofile_info_data_t], ctypes.c_void_p]]
@dll.bind(ctypes.c_uint32, c.POINTER[hsa_ven_amd_aqlprofile_profile_t], ctypes.c_uint32, ctypes.c_void_p)
def hsa_ven_amd_aqlprofile_get_info(profile:c.POINTER[hsa_ven_amd_aqlprofile_profile_t], attribute:ctypes.c_uint32, value:ctypes.c_void_p) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[hsa_ven_amd_aqlprofile_profile_t], hsa_ven_amd_aqlprofile_data_callback_t, ctypes.c_void_p)
def hsa_ven_amd_aqlprofile_iterate_data(profile:c.POINTER[hsa_ven_amd_aqlprofile_profile_t], callback:hsa_ven_amd_aqlprofile_data_callback_t, data:ctypes.c_void_p) -> ctypes.c_uint32: ...
@dll.bind(ctypes.c_uint32, c.POINTER[c.POINTER[ctypes.c_char]])
def hsa_ven_amd_aqlprofile_error_string(str:c.POINTER[c.POINTER[ctypes.c_char]]) -> ctypes.c_uint32: ...
hsa_ven_amd_aqlprofile_eventname_callback_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint32, [ctypes.c_int32, c.POINTER[ctypes.c_char]]]
@dll.bind(ctypes.c_uint32, hsa_ven_amd_aqlprofile_eventname_callback_t)
def hsa_ven_amd_aqlprofile_iterate_event_ids(_0:hsa_ven_amd_aqlprofile_eventname_callback_t) -> ctypes.c_uint32: ...
hsa_ven_amd_aqlprofile_coordinate_callback_t: TypeAlias = c.CFUNCTYPE[ctypes.c_uint32, [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, c.POINTER[ctypes.c_char], ctypes.c_void_p]]
@dll.bind(ctypes.c_uint32, hsa_agent_t, hsa_ven_amd_aqlprofile_event_t, uint32_t, hsa_ven_amd_aqlprofile_coordinate_callback_t, ctypes.c_void_p)
def hsa_ven_amd_aqlprofile_iterate_event_coord(agent:hsa_agent_t, event:hsa_ven_amd_aqlprofile_event_t, sample_id:uint32_t, callback:hsa_ven_amd_aqlprofile_coordinate_callback_t, userdata:ctypes.c_void_p) -> ctypes.c_uint32: ...
@c.record
class struct_hsa_ven_amd_aqlprofile_1_00_pfn_s(c.Struct):
  SIZE = 104
  hsa_ven_amd_aqlprofile_version_major: c.CFUNCTYPE[ctypes.c_uint32, []]
  hsa_ven_amd_aqlprofile_version_minor: c.CFUNCTYPE[ctypes.c_uint32, []]
  hsa_ven_amd_aqlprofile_error_string: c.CFUNCTYPE[ctypes.c_uint32, [c.POINTER[c.POINTER[ctypes.c_char]]]]
  hsa_ven_amd_aqlprofile_validate_event: c.CFUNCTYPE[ctypes.c_uint32, [struct_hsa_agent_s, c.POINTER[hsa_ven_amd_aqlprofile_event_t], c.POINTER[ctypes.c_bool]]]
  hsa_ven_amd_aqlprofile_start: c.CFUNCTYPE[ctypes.c_uint32, [c.POINTER[hsa_ven_amd_aqlprofile_profile_t], c.POINTER[hsa_ext_amd_aql_pm4_packet_t]]]
  hsa_ven_amd_aqlprofile_stop: c.CFUNCTYPE[ctypes.c_uint32, [c.POINTER[hsa_ven_amd_aqlprofile_profile_t], c.POINTER[hsa_ext_amd_aql_pm4_packet_t]]]
  hsa_ven_amd_aqlprofile_read: c.CFUNCTYPE[ctypes.c_uint32, [c.POINTER[hsa_ven_amd_aqlprofile_profile_t], c.POINTER[hsa_ext_amd_aql_pm4_packet_t]]]
  hsa_ven_amd_aqlprofile_legacy_get_pm4: c.CFUNCTYPE[ctypes.c_uint32, [c.POINTER[hsa_ext_amd_aql_pm4_packet_t], ctypes.c_void_p]]
  hsa_ven_amd_aqlprofile_get_info: c.CFUNCTYPE[ctypes.c_uint32, [c.POINTER[hsa_ven_amd_aqlprofile_profile_t], ctypes.c_uint32, ctypes.c_void_p]]
  hsa_ven_amd_aqlprofile_iterate_data: c.CFUNCTYPE[ctypes.c_uint32, [c.POINTER[hsa_ven_amd_aqlprofile_profile_t], c.CFUNCTYPE[ctypes.c_uint32, [ctypes.c_uint32, c.POINTER[hsa_ven_amd_aqlprofile_info_data_t], ctypes.c_void_p]], ctypes.c_void_p]]
  hsa_ven_amd_aqlprofile_iterate_event_ids: c.CFUNCTYPE[ctypes.c_uint32, [c.CFUNCTYPE[ctypes.c_uint32, [ctypes.c_int32, c.POINTER[ctypes.c_char]]]]]
  hsa_ven_amd_aqlprofile_iterate_event_coord: c.CFUNCTYPE[ctypes.c_uint32, [struct_hsa_agent_s, hsa_ven_amd_aqlprofile_event_t, ctypes.c_uint32, c.CFUNCTYPE[ctypes.c_uint32, [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, c.POINTER[ctypes.c_char], ctypes.c_void_p]], ctypes.c_void_p]]
  hsa_ven_amd_aqlprofile_att_marker: c.CFUNCTYPE[ctypes.c_uint32, [c.POINTER[hsa_ven_amd_aqlprofile_profile_t], c.POINTER[hsa_ext_amd_aql_pm4_packet_t], ctypes.c_uint32, ctypes.c_uint32]]
struct_hsa_ven_amd_aqlprofile_1_00_pfn_s.register_fields([('hsa_ven_amd_aqlprofile_version_major', c.CFUNCTYPE[uint32_t, []], 0), ('hsa_ven_amd_aqlprofile_version_minor', c.CFUNCTYPE[uint32_t, []], 8), ('hsa_ven_amd_aqlprofile_error_string', c.CFUNCTYPE[ctypes.c_uint32, [c.POINTER[c.POINTER[ctypes.c_char]]]], 16), ('hsa_ven_amd_aqlprofile_validate_event', c.CFUNCTYPE[ctypes.c_uint32, [hsa_agent_t, c.POINTER[hsa_ven_amd_aqlprofile_event_t], c.POINTER[ctypes.c_bool]]], 24), ('hsa_ven_amd_aqlprofile_start', c.CFUNCTYPE[ctypes.c_uint32, [c.POINTER[hsa_ven_amd_aqlprofile_profile_t], c.POINTER[hsa_ext_amd_aql_pm4_packet_t]]], 32), ('hsa_ven_amd_aqlprofile_stop', c.CFUNCTYPE[ctypes.c_uint32, [c.POINTER[hsa_ven_amd_aqlprofile_profile_t], c.POINTER[hsa_ext_amd_aql_pm4_packet_t]]], 40), ('hsa_ven_amd_aqlprofile_read', c.CFUNCTYPE[ctypes.c_uint32, [c.POINTER[hsa_ven_amd_aqlprofile_profile_t], c.POINTER[hsa_ext_amd_aql_pm4_packet_t]]], 48), ('hsa_ven_amd_aqlprofile_legacy_get_pm4', c.CFUNCTYPE[ctypes.c_uint32, [c.POINTER[hsa_ext_amd_aql_pm4_packet_t], ctypes.c_void_p]], 56), ('hsa_ven_amd_aqlprofile_get_info', c.CFUNCTYPE[ctypes.c_uint32, [c.POINTER[hsa_ven_amd_aqlprofile_profile_t], ctypes.c_uint32, ctypes.c_void_p]], 64), ('hsa_ven_amd_aqlprofile_iterate_data', c.CFUNCTYPE[ctypes.c_uint32, [c.POINTER[hsa_ven_amd_aqlprofile_profile_t], hsa_ven_amd_aqlprofile_data_callback_t, ctypes.c_void_p]], 72), ('hsa_ven_amd_aqlprofile_iterate_event_ids', c.CFUNCTYPE[ctypes.c_uint32, [hsa_ven_amd_aqlprofile_eventname_callback_t]], 80), ('hsa_ven_amd_aqlprofile_iterate_event_coord', c.CFUNCTYPE[ctypes.c_uint32, [hsa_agent_t, hsa_ven_amd_aqlprofile_event_t, uint32_t, hsa_ven_amd_aqlprofile_coordinate_callback_t, ctypes.c_void_p]], 88), ('hsa_ven_amd_aqlprofile_att_marker', c.CFUNCTYPE[ctypes.c_uint32, [c.POINTER[hsa_ven_amd_aqlprofile_profile_t], c.POINTER[hsa_ext_amd_aql_pm4_packet_t], uint32_t, ctypes.c_uint32]], 96)])
hsa_ven_amd_aqlprofile_1_00_pfn_t: TypeAlias = struct_hsa_ven_amd_aqlprofile_1_00_pfn_s
hsa_ven_amd_aqlprofile_pfn_t: TypeAlias = struct_hsa_ven_amd_aqlprofile_1_00_pfn_s
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