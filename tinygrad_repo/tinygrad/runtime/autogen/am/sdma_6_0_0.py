# mypy: ignore-errors
import ctypes
from tinygrad.runtime.support.c import DLL, Struct, CEnum, _IO, _IOW, _IOR, _IOWR
class rocr_AMD_SDMA_PKT_COPY_LINEAR_TAG(Struct): pass
class rocr_AMD_SDMA_PKT_COPY_LINEAR_TAG_HEADER_UNION(ctypes.Union): pass
class rocr_AMD_SDMA_PKT_COPY_LINEAR_TAG_HEADER_UNION_0(Struct): pass
rocr_AMD_SDMA_PKT_COPY_LINEAR_TAG_HEADER_UNION_0._fields_ = [
  ('op', ctypes.c_uint32,8),
  ('sub_op', ctypes.c_uint32,8),
  ('extra_info', ctypes.c_uint32,16),
]
rocr_AMD_SDMA_PKT_COPY_LINEAR_TAG_HEADER_UNION._anonymous_ = ['_0']
rocr_AMD_SDMA_PKT_COPY_LINEAR_TAG_HEADER_UNION._fields_ = [
  ('_0', rocr_AMD_SDMA_PKT_COPY_LINEAR_TAG_HEADER_UNION_0),
  ('DW_0_DATA', ctypes.c_uint32),
]
class rocr_AMD_SDMA_PKT_COPY_LINEAR_TAG_COUNT_UNION(ctypes.Union): pass
class rocr_AMD_SDMA_PKT_COPY_LINEAR_TAG_COUNT_UNION_0(Struct): pass
rocr_AMD_SDMA_PKT_COPY_LINEAR_TAG_COUNT_UNION_0._fields_ = [
  ('count', ctypes.c_uint32,22),
  ('reserved_0', ctypes.c_uint32,10),
]
rocr_AMD_SDMA_PKT_COPY_LINEAR_TAG_COUNT_UNION._anonymous_ = ['_0']
rocr_AMD_SDMA_PKT_COPY_LINEAR_TAG_COUNT_UNION._fields_ = [
  ('_0', rocr_AMD_SDMA_PKT_COPY_LINEAR_TAG_COUNT_UNION_0),
  ('DW_1_DATA', ctypes.c_uint32),
]
class rocr_AMD_SDMA_PKT_COPY_LINEAR_TAG_PARAMETER_UNION(ctypes.Union): pass
class rocr_AMD_SDMA_PKT_COPY_LINEAR_TAG_PARAMETER_UNION_0(Struct): pass
rocr_AMD_SDMA_PKT_COPY_LINEAR_TAG_PARAMETER_UNION_0._fields_ = [
  ('reserved_0', ctypes.c_uint32,16),
  ('dst_swap', ctypes.c_uint32,2),
  ('reserved_1', ctypes.c_uint32,6),
  ('src_swap', ctypes.c_uint32,2),
  ('reserved_2', ctypes.c_uint32,6),
]
rocr_AMD_SDMA_PKT_COPY_LINEAR_TAG_PARAMETER_UNION._anonymous_ = ['_0']
rocr_AMD_SDMA_PKT_COPY_LINEAR_TAG_PARAMETER_UNION._fields_ = [
  ('_0', rocr_AMD_SDMA_PKT_COPY_LINEAR_TAG_PARAMETER_UNION_0),
  ('DW_2_DATA', ctypes.c_uint32),
]
class rocr_AMD_SDMA_PKT_COPY_LINEAR_TAG_SRC_ADDR_LO_UNION(ctypes.Union): pass
class rocr_AMD_SDMA_PKT_COPY_LINEAR_TAG_SRC_ADDR_LO_UNION_0(Struct): pass
rocr_AMD_SDMA_PKT_COPY_LINEAR_TAG_SRC_ADDR_LO_UNION_0._fields_ = [
  ('src_addr_31_0', ctypes.c_uint32,32),
]
rocr_AMD_SDMA_PKT_COPY_LINEAR_TAG_SRC_ADDR_LO_UNION._anonymous_ = ['_0']
rocr_AMD_SDMA_PKT_COPY_LINEAR_TAG_SRC_ADDR_LO_UNION._fields_ = [
  ('_0', rocr_AMD_SDMA_PKT_COPY_LINEAR_TAG_SRC_ADDR_LO_UNION_0),
  ('DW_3_DATA', ctypes.c_uint32),
]
class rocr_AMD_SDMA_PKT_COPY_LINEAR_TAG_SRC_ADDR_HI_UNION(ctypes.Union): pass
class rocr_AMD_SDMA_PKT_COPY_LINEAR_TAG_SRC_ADDR_HI_UNION_0(Struct): pass
rocr_AMD_SDMA_PKT_COPY_LINEAR_TAG_SRC_ADDR_HI_UNION_0._fields_ = [
  ('src_addr_63_32', ctypes.c_uint32,32),
]
rocr_AMD_SDMA_PKT_COPY_LINEAR_TAG_SRC_ADDR_HI_UNION._anonymous_ = ['_0']
rocr_AMD_SDMA_PKT_COPY_LINEAR_TAG_SRC_ADDR_HI_UNION._fields_ = [
  ('_0', rocr_AMD_SDMA_PKT_COPY_LINEAR_TAG_SRC_ADDR_HI_UNION_0),
  ('DW_4_DATA', ctypes.c_uint32),
]
class rocr_AMD_SDMA_PKT_COPY_LINEAR_TAG_DST_ADDR_LO_UNION(ctypes.Union): pass
class rocr_AMD_SDMA_PKT_COPY_LINEAR_TAG_DST_ADDR_LO_UNION_0(Struct): pass
rocr_AMD_SDMA_PKT_COPY_LINEAR_TAG_DST_ADDR_LO_UNION_0._fields_ = [
  ('dst_addr_31_0', ctypes.c_uint32,32),
]
rocr_AMD_SDMA_PKT_COPY_LINEAR_TAG_DST_ADDR_LO_UNION._anonymous_ = ['_0']
rocr_AMD_SDMA_PKT_COPY_LINEAR_TAG_DST_ADDR_LO_UNION._fields_ = [
  ('_0', rocr_AMD_SDMA_PKT_COPY_LINEAR_TAG_DST_ADDR_LO_UNION_0),
  ('DW_5_DATA', ctypes.c_uint32),
]
class rocr_AMD_SDMA_PKT_COPY_LINEAR_TAG_DST_ADDR_HI_UNION(ctypes.Union): pass
class rocr_AMD_SDMA_PKT_COPY_LINEAR_TAG_DST_ADDR_HI_UNION_0(Struct): pass
rocr_AMD_SDMA_PKT_COPY_LINEAR_TAG_DST_ADDR_HI_UNION_0._fields_ = [
  ('dst_addr_63_32', ctypes.c_uint32,32),
]
rocr_AMD_SDMA_PKT_COPY_LINEAR_TAG_DST_ADDR_HI_UNION._anonymous_ = ['_0']
rocr_AMD_SDMA_PKT_COPY_LINEAR_TAG_DST_ADDR_HI_UNION._fields_ = [
  ('_0', rocr_AMD_SDMA_PKT_COPY_LINEAR_TAG_DST_ADDR_HI_UNION_0),
  ('DW_6_DATA', ctypes.c_uint32),
]
rocr_AMD_SDMA_PKT_COPY_LINEAR_TAG._fields_ = [
  ('HEADER_UNION', rocr_AMD_SDMA_PKT_COPY_LINEAR_TAG_HEADER_UNION),
  ('COUNT_UNION', rocr_AMD_SDMA_PKT_COPY_LINEAR_TAG_COUNT_UNION),
  ('PARAMETER_UNION', rocr_AMD_SDMA_PKT_COPY_LINEAR_TAG_PARAMETER_UNION),
  ('SRC_ADDR_LO_UNION', rocr_AMD_SDMA_PKT_COPY_LINEAR_TAG_SRC_ADDR_LO_UNION),
  ('SRC_ADDR_HI_UNION', rocr_AMD_SDMA_PKT_COPY_LINEAR_TAG_SRC_ADDR_HI_UNION),
  ('DST_ADDR_LO_UNION', rocr_AMD_SDMA_PKT_COPY_LINEAR_TAG_DST_ADDR_LO_UNION),
  ('DST_ADDR_HI_UNION', rocr_AMD_SDMA_PKT_COPY_LINEAR_TAG_DST_ADDR_HI_UNION),
]
rocr_AMD_SDMA_PKT_COPY_LINEAR = rocr_AMD_SDMA_PKT_COPY_LINEAR_TAG
class rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG(Struct): pass
class rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_HEADER_UNION(ctypes.Union): pass
class rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_HEADER_UNION_0(Struct): pass
rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_HEADER_UNION_0._fields_ = [
  ('op', ctypes.c_uint32,8),
  ('sub_op', ctypes.c_uint32,8),
  ('reserved', ctypes.c_uint32,13),
  ('element', ctypes.c_uint32,3),
]
rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_HEADER_UNION._anonymous_ = ['_0']
rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_HEADER_UNION._fields_ = [
  ('_0', rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_HEADER_UNION_0),
  ('DW_0_DATA', ctypes.c_uint32),
]
class rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_ADDR_LO_UNION(ctypes.Union): pass
class rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_ADDR_LO_UNION_0(Struct): pass
rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_ADDR_LO_UNION_0._fields_ = [
  ('src_addr_31_0', ctypes.c_uint32,32),
]
rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_ADDR_LO_UNION._anonymous_ = ['_0']
rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_ADDR_LO_UNION._fields_ = [
  ('_0', rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_ADDR_LO_UNION_0),
  ('DW_1_DATA', ctypes.c_uint32),
]
class rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_ADDR_HI_UNION(ctypes.Union): pass
class rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_ADDR_HI_UNION_0(Struct): pass
rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_ADDR_HI_UNION_0._fields_ = [
  ('src_addr_63_32', ctypes.c_uint32,32),
]
rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_ADDR_HI_UNION._anonymous_ = ['_0']
rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_ADDR_HI_UNION._fields_ = [
  ('_0', rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_ADDR_HI_UNION_0),
  ('DW_2_DATA', ctypes.c_uint32),
]
class rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_1_UNION(ctypes.Union): pass
class rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_1_UNION_0(Struct): pass
rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_1_UNION_0._fields_ = [
  ('src_offset_x', ctypes.c_uint32,14),
  ('reserved_1', ctypes.c_uint32,2),
  ('src_offset_y', ctypes.c_uint32,14),
  ('reserved_2', ctypes.c_uint32,2),
]
rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_1_UNION._anonymous_ = ['_0']
rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_1_UNION._fields_ = [
  ('_0', rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_1_UNION_0),
  ('DW_3_DATA', ctypes.c_uint32),
]
class rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_2_UNION(ctypes.Union): pass
class rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_2_UNION_0(Struct): pass
rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_2_UNION_0._fields_ = [
  ('src_offset_z', ctypes.c_uint32,11),
  ('reserved_1', ctypes.c_uint32,2),
  ('src_pitch', ctypes.c_uint32,19),
]
rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_2_UNION._anonymous_ = ['_0']
rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_2_UNION._fields_ = [
  ('_0', rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_2_UNION_0),
  ('DW_4_DATA', ctypes.c_uint32),
]
class rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_3_UNION(ctypes.Union): pass
class rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_3_UNION_0(Struct): pass
rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_3_UNION_0._fields_ = [
  ('src_slice_pitch', ctypes.c_uint32,28),
  ('reserved_1', ctypes.c_uint32,4),
]
rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_3_UNION._anonymous_ = ['_0']
rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_3_UNION._fields_ = [
  ('_0', rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_3_UNION_0),
  ('DW_5_DATA', ctypes.c_uint32),
]
class rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_ADDR_LO_UNION(ctypes.Union): pass
class rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_ADDR_LO_UNION_0(Struct): pass
rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_ADDR_LO_UNION_0._fields_ = [
  ('dst_addr_31_0', ctypes.c_uint32,32),
]
rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_ADDR_LO_UNION._anonymous_ = ['_0']
rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_ADDR_LO_UNION._fields_ = [
  ('_0', rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_ADDR_LO_UNION_0),
  ('DW_6_DATA', ctypes.c_uint32),
]
class rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_ADDR_HI_UNION(ctypes.Union): pass
class rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_ADDR_HI_UNION_0(Struct): pass
rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_ADDR_HI_UNION_0._fields_ = [
  ('dst_addr_63_32', ctypes.c_uint32,32),
]
rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_ADDR_HI_UNION._anonymous_ = ['_0']
rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_ADDR_HI_UNION._fields_ = [
  ('_0', rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_ADDR_HI_UNION_0),
  ('DW_7_DATA', ctypes.c_uint32),
]
class rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_1_UNION(ctypes.Union): pass
class rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_1_UNION_0(Struct): pass
rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_1_UNION_0._fields_ = [
  ('dst_offset_x', ctypes.c_uint32,14),
  ('reserved_1', ctypes.c_uint32,2),
  ('dst_offset_y', ctypes.c_uint32,14),
  ('reserved_2', ctypes.c_uint32,2),
]
rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_1_UNION._anonymous_ = ['_0']
rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_1_UNION._fields_ = [
  ('_0', rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_1_UNION_0),
  ('DW_8_DATA', ctypes.c_uint32),
]
class rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_2_UNION(ctypes.Union): pass
class rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_2_UNION_0(Struct): pass
rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_2_UNION_0._fields_ = [
  ('dst_offset_z', ctypes.c_uint32,11),
  ('reserved_1', ctypes.c_uint32,2),
  ('dst_pitch', ctypes.c_uint32,19),
]
rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_2_UNION._anonymous_ = ['_0']
rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_2_UNION._fields_ = [
  ('_0', rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_2_UNION_0),
  ('DW_9_DATA', ctypes.c_uint32),
]
class rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_3_UNION(ctypes.Union): pass
class rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_3_UNION_0(Struct): pass
rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_3_UNION_0._fields_ = [
  ('dst_slice_pitch', ctypes.c_uint32,28),
  ('reserved_1', ctypes.c_uint32,4),
]
rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_3_UNION._anonymous_ = ['_0']
rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_3_UNION._fields_ = [
  ('_0', rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_3_UNION_0),
  ('DW_10_DATA', ctypes.c_uint32),
]
class rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_RECT_PARAMETER_1_UNION(ctypes.Union): pass
class rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_RECT_PARAMETER_1_UNION_0(Struct): pass
rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_RECT_PARAMETER_1_UNION_0._fields_ = [
  ('rect_x', ctypes.c_uint32,14),
  ('reserved_1', ctypes.c_uint32,2),
  ('rect_y', ctypes.c_uint32,14),
  ('reserved_2', ctypes.c_uint32,2),
]
rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_RECT_PARAMETER_1_UNION._anonymous_ = ['_0']
rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_RECT_PARAMETER_1_UNION._fields_ = [
  ('_0', rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_RECT_PARAMETER_1_UNION_0),
  ('DW_11_DATA', ctypes.c_uint32),
]
class rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_RECT_PARAMETER_2_UNION(ctypes.Union): pass
class rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_RECT_PARAMETER_2_UNION_0(Struct): pass
rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_RECT_PARAMETER_2_UNION_0._fields_ = [
  ('rect_z', ctypes.c_uint32,11),
  ('reserved_1', ctypes.c_uint32,5),
  ('dst_swap', ctypes.c_uint32,2),
  ('reserved_2', ctypes.c_uint32,6),
  ('src_swap', ctypes.c_uint32,2),
  ('reserved_3', ctypes.c_uint32,6),
]
rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_RECT_PARAMETER_2_UNION._anonymous_ = ['_0']
rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_RECT_PARAMETER_2_UNION._fields_ = [
  ('_0', rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_RECT_PARAMETER_2_UNION_0),
  ('DW_12_DATA', ctypes.c_uint32),
]
rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG._fields_ = [
  ('HEADER_UNION', rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_HEADER_UNION),
  ('SRC_ADDR_LO_UNION', rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_ADDR_LO_UNION),
  ('SRC_ADDR_HI_UNION', rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_ADDR_HI_UNION),
  ('SRC_PARAMETER_1_UNION', rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_1_UNION),
  ('SRC_PARAMETER_2_UNION', rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_2_UNION),
  ('SRC_PARAMETER_3_UNION', rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_3_UNION),
  ('DST_ADDR_LO_UNION', rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_ADDR_LO_UNION),
  ('DST_ADDR_HI_UNION', rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_ADDR_HI_UNION),
  ('DST_PARAMETER_1_UNION', rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_1_UNION),
  ('DST_PARAMETER_2_UNION', rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_2_UNION),
  ('DST_PARAMETER_3_UNION', rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_3_UNION),
  ('RECT_PARAMETER_1_UNION', rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_RECT_PARAMETER_1_UNION),
  ('RECT_PARAMETER_2_UNION', rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG_RECT_PARAMETER_2_UNION),
]
rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT = rocr_AMD_SDMA_PKT_COPY_LINEAR_RECT_TAG
class rocr_AMD_SDMA_PKT_CONSTANT_FILL_TAG(Struct): pass
class rocr_AMD_SDMA_PKT_CONSTANT_FILL_TAG_HEADER_UNION(ctypes.Union): pass
class rocr_AMD_SDMA_PKT_CONSTANT_FILL_TAG_HEADER_UNION_0(Struct): pass
rocr_AMD_SDMA_PKT_CONSTANT_FILL_TAG_HEADER_UNION_0._fields_ = [
  ('op', ctypes.c_uint32,8),
  ('sub_op', ctypes.c_uint32,8),
  ('sw', ctypes.c_uint32,2),
  ('reserved_0', ctypes.c_uint32,12),
  ('fillsize', ctypes.c_uint32,2),
]
rocr_AMD_SDMA_PKT_CONSTANT_FILL_TAG_HEADER_UNION._anonymous_ = ['_0']
rocr_AMD_SDMA_PKT_CONSTANT_FILL_TAG_HEADER_UNION._fields_ = [
  ('_0', rocr_AMD_SDMA_PKT_CONSTANT_FILL_TAG_HEADER_UNION_0),
  ('DW_0_DATA', ctypes.c_uint32),
]
class rocr_AMD_SDMA_PKT_CONSTANT_FILL_TAG_DST_ADDR_LO_UNION(ctypes.Union): pass
class rocr_AMD_SDMA_PKT_CONSTANT_FILL_TAG_DST_ADDR_LO_UNION_0(Struct): pass
rocr_AMD_SDMA_PKT_CONSTANT_FILL_TAG_DST_ADDR_LO_UNION_0._fields_ = [
  ('dst_addr_31_0', ctypes.c_uint32,32),
]
rocr_AMD_SDMA_PKT_CONSTANT_FILL_TAG_DST_ADDR_LO_UNION._anonymous_ = ['_0']
rocr_AMD_SDMA_PKT_CONSTANT_FILL_TAG_DST_ADDR_LO_UNION._fields_ = [
  ('_0', rocr_AMD_SDMA_PKT_CONSTANT_FILL_TAG_DST_ADDR_LO_UNION_0),
  ('DW_1_DATA', ctypes.c_uint32),
]
class rocr_AMD_SDMA_PKT_CONSTANT_FILL_TAG_DST_ADDR_HI_UNION(ctypes.Union): pass
class rocr_AMD_SDMA_PKT_CONSTANT_FILL_TAG_DST_ADDR_HI_UNION_0(Struct): pass
rocr_AMD_SDMA_PKT_CONSTANT_FILL_TAG_DST_ADDR_HI_UNION_0._fields_ = [
  ('dst_addr_63_32', ctypes.c_uint32,32),
]
rocr_AMD_SDMA_PKT_CONSTANT_FILL_TAG_DST_ADDR_HI_UNION._anonymous_ = ['_0']
rocr_AMD_SDMA_PKT_CONSTANT_FILL_TAG_DST_ADDR_HI_UNION._fields_ = [
  ('_0', rocr_AMD_SDMA_PKT_CONSTANT_FILL_TAG_DST_ADDR_HI_UNION_0),
  ('DW_2_DATA', ctypes.c_uint32),
]
class rocr_AMD_SDMA_PKT_CONSTANT_FILL_TAG_DATA_UNION(ctypes.Union): pass
class rocr_AMD_SDMA_PKT_CONSTANT_FILL_TAG_DATA_UNION_0(Struct): pass
rocr_AMD_SDMA_PKT_CONSTANT_FILL_TAG_DATA_UNION_0._fields_ = [
  ('src_data_31_0', ctypes.c_uint32,32),
]
rocr_AMD_SDMA_PKT_CONSTANT_FILL_TAG_DATA_UNION._anonymous_ = ['_0']
rocr_AMD_SDMA_PKT_CONSTANT_FILL_TAG_DATA_UNION._fields_ = [
  ('_0', rocr_AMD_SDMA_PKT_CONSTANT_FILL_TAG_DATA_UNION_0),
  ('DW_3_DATA', ctypes.c_uint32),
]
class rocr_AMD_SDMA_PKT_CONSTANT_FILL_TAG_COUNT_UNION(ctypes.Union): pass
class rocr_AMD_SDMA_PKT_CONSTANT_FILL_TAG_COUNT_UNION_0(Struct): pass
rocr_AMD_SDMA_PKT_CONSTANT_FILL_TAG_COUNT_UNION_0._fields_ = [
  ('count', ctypes.c_uint32,22),
  ('reserved_0', ctypes.c_uint32,10),
]
rocr_AMD_SDMA_PKT_CONSTANT_FILL_TAG_COUNT_UNION._anonymous_ = ['_0']
rocr_AMD_SDMA_PKT_CONSTANT_FILL_TAG_COUNT_UNION._fields_ = [
  ('_0', rocr_AMD_SDMA_PKT_CONSTANT_FILL_TAG_COUNT_UNION_0),
  ('DW_4_DATA', ctypes.c_uint32),
]
rocr_AMD_SDMA_PKT_CONSTANT_FILL_TAG._fields_ = [
  ('HEADER_UNION', rocr_AMD_SDMA_PKT_CONSTANT_FILL_TAG_HEADER_UNION),
  ('DST_ADDR_LO_UNION', rocr_AMD_SDMA_PKT_CONSTANT_FILL_TAG_DST_ADDR_LO_UNION),
  ('DST_ADDR_HI_UNION', rocr_AMD_SDMA_PKT_CONSTANT_FILL_TAG_DST_ADDR_HI_UNION),
  ('DATA_UNION', rocr_AMD_SDMA_PKT_CONSTANT_FILL_TAG_DATA_UNION),
  ('COUNT_UNION', rocr_AMD_SDMA_PKT_CONSTANT_FILL_TAG_COUNT_UNION),
]
rocr_AMD_SDMA_PKT_CONSTANT_FILL = rocr_AMD_SDMA_PKT_CONSTANT_FILL_TAG
class rocr_AMD_SDMA_PKT_FENCE_TAG(Struct): pass
class rocr_AMD_SDMA_PKT_FENCE_TAG_HEADER_UNION(ctypes.Union): pass
class rocr_AMD_SDMA_PKT_FENCE_TAG_HEADER_UNION_0(Struct): pass
rocr_AMD_SDMA_PKT_FENCE_TAG_HEADER_UNION_0._fields_ = [
  ('op', ctypes.c_uint32,8),
  ('sub_op', ctypes.c_uint32,8),
  ('mtype', ctypes.c_uint32,3),
  ('gcc', ctypes.c_uint32,1),
  ('sys', ctypes.c_uint32,1),
  ('pad1', ctypes.c_uint32,1),
  ('snp', ctypes.c_uint32,1),
  ('gpa', ctypes.c_uint32,1),
  ('l2_policy', ctypes.c_uint32,2),
  ('reserved_0', ctypes.c_uint32,6),
]
rocr_AMD_SDMA_PKT_FENCE_TAG_HEADER_UNION._anonymous_ = ['_0']
rocr_AMD_SDMA_PKT_FENCE_TAG_HEADER_UNION._fields_ = [
  ('_0', rocr_AMD_SDMA_PKT_FENCE_TAG_HEADER_UNION_0),
  ('DW_0_DATA', ctypes.c_uint32),
]
class rocr_AMD_SDMA_PKT_FENCE_TAG_ADDR_LO_UNION(ctypes.Union): pass
class rocr_AMD_SDMA_PKT_FENCE_TAG_ADDR_LO_UNION_0(Struct): pass
rocr_AMD_SDMA_PKT_FENCE_TAG_ADDR_LO_UNION_0._fields_ = [
  ('addr_31_0', ctypes.c_uint32,32),
]
rocr_AMD_SDMA_PKT_FENCE_TAG_ADDR_LO_UNION._anonymous_ = ['_0']
rocr_AMD_SDMA_PKT_FENCE_TAG_ADDR_LO_UNION._fields_ = [
  ('_0', rocr_AMD_SDMA_PKT_FENCE_TAG_ADDR_LO_UNION_0),
  ('DW_1_DATA', ctypes.c_uint32),
]
class rocr_AMD_SDMA_PKT_FENCE_TAG_ADDR_HI_UNION(ctypes.Union): pass
class rocr_AMD_SDMA_PKT_FENCE_TAG_ADDR_HI_UNION_0(Struct): pass
rocr_AMD_SDMA_PKT_FENCE_TAG_ADDR_HI_UNION_0._fields_ = [
  ('addr_63_32', ctypes.c_uint32,32),
]
rocr_AMD_SDMA_PKT_FENCE_TAG_ADDR_HI_UNION._anonymous_ = ['_0']
rocr_AMD_SDMA_PKT_FENCE_TAG_ADDR_HI_UNION._fields_ = [
  ('_0', rocr_AMD_SDMA_PKT_FENCE_TAG_ADDR_HI_UNION_0),
  ('DW_2_DATA', ctypes.c_uint32),
]
class rocr_AMD_SDMA_PKT_FENCE_TAG_DATA_UNION(ctypes.Union): pass
class rocr_AMD_SDMA_PKT_FENCE_TAG_DATA_UNION_0(Struct): pass
rocr_AMD_SDMA_PKT_FENCE_TAG_DATA_UNION_0._fields_ = [
  ('data', ctypes.c_uint32,32),
]
rocr_AMD_SDMA_PKT_FENCE_TAG_DATA_UNION._anonymous_ = ['_0']
rocr_AMD_SDMA_PKT_FENCE_TAG_DATA_UNION._fields_ = [
  ('_0', rocr_AMD_SDMA_PKT_FENCE_TAG_DATA_UNION_0),
  ('DW_3_DATA', ctypes.c_uint32),
]
rocr_AMD_SDMA_PKT_FENCE_TAG._fields_ = [
  ('HEADER_UNION', rocr_AMD_SDMA_PKT_FENCE_TAG_HEADER_UNION),
  ('ADDR_LO_UNION', rocr_AMD_SDMA_PKT_FENCE_TAG_ADDR_LO_UNION),
  ('ADDR_HI_UNION', rocr_AMD_SDMA_PKT_FENCE_TAG_ADDR_HI_UNION),
  ('DATA_UNION', rocr_AMD_SDMA_PKT_FENCE_TAG_DATA_UNION),
]
rocr_AMD_SDMA_PKT_FENCE = rocr_AMD_SDMA_PKT_FENCE_TAG
class rocr_AMD_SDMA_PKT_POLL_REGMEM_TAG(Struct): pass
class rocr_AMD_SDMA_PKT_POLL_REGMEM_TAG_HEADER_UNION(ctypes.Union): pass
class rocr_AMD_SDMA_PKT_POLL_REGMEM_TAG_HEADER_UNION_0(Struct): pass
rocr_AMD_SDMA_PKT_POLL_REGMEM_TAG_HEADER_UNION_0._fields_ = [
  ('op', ctypes.c_uint32,8),
  ('sub_op', ctypes.c_uint32,8),
  ('reserved_0', ctypes.c_uint32,10),
  ('hdp_flush', ctypes.c_uint32,1),
  ('reserved_1', ctypes.c_uint32,1),
  ('func', ctypes.c_uint32,3),
  ('mem_poll', ctypes.c_uint32,1),
]
rocr_AMD_SDMA_PKT_POLL_REGMEM_TAG_HEADER_UNION._anonymous_ = ['_0']
rocr_AMD_SDMA_PKT_POLL_REGMEM_TAG_HEADER_UNION._fields_ = [
  ('_0', rocr_AMD_SDMA_PKT_POLL_REGMEM_TAG_HEADER_UNION_0),
  ('DW_0_DATA', ctypes.c_uint32),
]
class rocr_AMD_SDMA_PKT_POLL_REGMEM_TAG_ADDR_LO_UNION(ctypes.Union): pass
class rocr_AMD_SDMA_PKT_POLL_REGMEM_TAG_ADDR_LO_UNION_0(Struct): pass
rocr_AMD_SDMA_PKT_POLL_REGMEM_TAG_ADDR_LO_UNION_0._fields_ = [
  ('addr_31_0', ctypes.c_uint32,32),
]
rocr_AMD_SDMA_PKT_POLL_REGMEM_TAG_ADDR_LO_UNION._anonymous_ = ['_0']
rocr_AMD_SDMA_PKT_POLL_REGMEM_TAG_ADDR_LO_UNION._fields_ = [
  ('_0', rocr_AMD_SDMA_PKT_POLL_REGMEM_TAG_ADDR_LO_UNION_0),
  ('DW_1_DATA', ctypes.c_uint32),
]
class rocr_AMD_SDMA_PKT_POLL_REGMEM_TAG_ADDR_HI_UNION(ctypes.Union): pass
class rocr_AMD_SDMA_PKT_POLL_REGMEM_TAG_ADDR_HI_UNION_0(Struct): pass
rocr_AMD_SDMA_PKT_POLL_REGMEM_TAG_ADDR_HI_UNION_0._fields_ = [
  ('addr_63_32', ctypes.c_uint32,32),
]
rocr_AMD_SDMA_PKT_POLL_REGMEM_TAG_ADDR_HI_UNION._anonymous_ = ['_0']
rocr_AMD_SDMA_PKT_POLL_REGMEM_TAG_ADDR_HI_UNION._fields_ = [
  ('_0', rocr_AMD_SDMA_PKT_POLL_REGMEM_TAG_ADDR_HI_UNION_0),
  ('DW_2_DATA', ctypes.c_uint32),
]
class rocr_AMD_SDMA_PKT_POLL_REGMEM_TAG_VALUE_UNION(ctypes.Union): pass
class rocr_AMD_SDMA_PKT_POLL_REGMEM_TAG_VALUE_UNION_0(Struct): pass
rocr_AMD_SDMA_PKT_POLL_REGMEM_TAG_VALUE_UNION_0._fields_ = [
  ('value', ctypes.c_uint32,32),
]
rocr_AMD_SDMA_PKT_POLL_REGMEM_TAG_VALUE_UNION._anonymous_ = ['_0']
rocr_AMD_SDMA_PKT_POLL_REGMEM_TAG_VALUE_UNION._fields_ = [
  ('_0', rocr_AMD_SDMA_PKT_POLL_REGMEM_TAG_VALUE_UNION_0),
  ('DW_3_DATA', ctypes.c_uint32),
]
class rocr_AMD_SDMA_PKT_POLL_REGMEM_TAG_MASK_UNION(ctypes.Union): pass
class rocr_AMD_SDMA_PKT_POLL_REGMEM_TAG_MASK_UNION_0(Struct): pass
rocr_AMD_SDMA_PKT_POLL_REGMEM_TAG_MASK_UNION_0._fields_ = [
  ('mask', ctypes.c_uint32,32),
]
rocr_AMD_SDMA_PKT_POLL_REGMEM_TAG_MASK_UNION._anonymous_ = ['_0']
rocr_AMD_SDMA_PKT_POLL_REGMEM_TAG_MASK_UNION._fields_ = [
  ('_0', rocr_AMD_SDMA_PKT_POLL_REGMEM_TAG_MASK_UNION_0),
  ('DW_4_DATA', ctypes.c_uint32),
]
class rocr_AMD_SDMA_PKT_POLL_REGMEM_TAG_DW5_UNION(ctypes.Union): pass
class rocr_AMD_SDMA_PKT_POLL_REGMEM_TAG_DW5_UNION_0(Struct): pass
rocr_AMD_SDMA_PKT_POLL_REGMEM_TAG_DW5_UNION_0._fields_ = [
  ('interval', ctypes.c_uint32,16),
  ('retry_count', ctypes.c_uint32,12),
  ('reserved_0', ctypes.c_uint32,4),
]
rocr_AMD_SDMA_PKT_POLL_REGMEM_TAG_DW5_UNION._anonymous_ = ['_0']
rocr_AMD_SDMA_PKT_POLL_REGMEM_TAG_DW5_UNION._fields_ = [
  ('_0', rocr_AMD_SDMA_PKT_POLL_REGMEM_TAG_DW5_UNION_0),
  ('DW_5_DATA', ctypes.c_uint32),
]
rocr_AMD_SDMA_PKT_POLL_REGMEM_TAG._fields_ = [
  ('HEADER_UNION', rocr_AMD_SDMA_PKT_POLL_REGMEM_TAG_HEADER_UNION),
  ('ADDR_LO_UNION', rocr_AMD_SDMA_PKT_POLL_REGMEM_TAG_ADDR_LO_UNION),
  ('ADDR_HI_UNION', rocr_AMD_SDMA_PKT_POLL_REGMEM_TAG_ADDR_HI_UNION),
  ('VALUE_UNION', rocr_AMD_SDMA_PKT_POLL_REGMEM_TAG_VALUE_UNION),
  ('MASK_UNION', rocr_AMD_SDMA_PKT_POLL_REGMEM_TAG_MASK_UNION),
  ('DW5_UNION', rocr_AMD_SDMA_PKT_POLL_REGMEM_TAG_DW5_UNION),
]
rocr_AMD_SDMA_PKT_POLL_REGMEM = rocr_AMD_SDMA_PKT_POLL_REGMEM_TAG
class rocr_AMD_SDMA_PKT_ATOMIC_TAG(Struct): pass
class rocr_AMD_SDMA_PKT_ATOMIC_TAG_HEADER_UNION(ctypes.Union): pass
class rocr_AMD_SDMA_PKT_ATOMIC_TAG_HEADER_UNION_0(Struct): pass
rocr_AMD_SDMA_PKT_ATOMIC_TAG_HEADER_UNION_0._fields_ = [
  ('op', ctypes.c_uint32,8),
  ('sub_op', ctypes.c_uint32,8),
  ('l', ctypes.c_uint32,1),
  ('reserved_0', ctypes.c_uint32,8),
  ('operation', ctypes.c_uint32,7),
]
rocr_AMD_SDMA_PKT_ATOMIC_TAG_HEADER_UNION._anonymous_ = ['_0']
rocr_AMD_SDMA_PKT_ATOMIC_TAG_HEADER_UNION._fields_ = [
  ('_0', rocr_AMD_SDMA_PKT_ATOMIC_TAG_HEADER_UNION_0),
  ('DW_0_DATA', ctypes.c_uint32),
]
class rocr_AMD_SDMA_PKT_ATOMIC_TAG_ADDR_LO_UNION(ctypes.Union): pass
class rocr_AMD_SDMA_PKT_ATOMIC_TAG_ADDR_LO_UNION_0(Struct): pass
rocr_AMD_SDMA_PKT_ATOMIC_TAG_ADDR_LO_UNION_0._fields_ = [
  ('addr_31_0', ctypes.c_uint32,32),
]
rocr_AMD_SDMA_PKT_ATOMIC_TAG_ADDR_LO_UNION._anonymous_ = ['_0']
rocr_AMD_SDMA_PKT_ATOMIC_TAG_ADDR_LO_UNION._fields_ = [
  ('_0', rocr_AMD_SDMA_PKT_ATOMIC_TAG_ADDR_LO_UNION_0),
  ('DW_1_DATA', ctypes.c_uint32),
]
class rocr_AMD_SDMA_PKT_ATOMIC_TAG_ADDR_HI_UNION(ctypes.Union): pass
class rocr_AMD_SDMA_PKT_ATOMIC_TAG_ADDR_HI_UNION_0(Struct): pass
rocr_AMD_SDMA_PKT_ATOMIC_TAG_ADDR_HI_UNION_0._fields_ = [
  ('addr_63_32', ctypes.c_uint32,32),
]
rocr_AMD_SDMA_PKT_ATOMIC_TAG_ADDR_HI_UNION._anonymous_ = ['_0']
rocr_AMD_SDMA_PKT_ATOMIC_TAG_ADDR_HI_UNION._fields_ = [
  ('_0', rocr_AMD_SDMA_PKT_ATOMIC_TAG_ADDR_HI_UNION_0),
  ('DW_2_DATA', ctypes.c_uint32),
]
class rocr_AMD_SDMA_PKT_ATOMIC_TAG_SRC_DATA_LO_UNION(ctypes.Union): pass
class rocr_AMD_SDMA_PKT_ATOMIC_TAG_SRC_DATA_LO_UNION_0(Struct): pass
rocr_AMD_SDMA_PKT_ATOMIC_TAG_SRC_DATA_LO_UNION_0._fields_ = [
  ('src_data_31_0', ctypes.c_uint32,32),
]
rocr_AMD_SDMA_PKT_ATOMIC_TAG_SRC_DATA_LO_UNION._anonymous_ = ['_0']
rocr_AMD_SDMA_PKT_ATOMIC_TAG_SRC_DATA_LO_UNION._fields_ = [
  ('_0', rocr_AMD_SDMA_PKT_ATOMIC_TAG_SRC_DATA_LO_UNION_0),
  ('DW_3_DATA', ctypes.c_uint32),
]
class rocr_AMD_SDMA_PKT_ATOMIC_TAG_SRC_DATA_HI_UNION(ctypes.Union): pass
class rocr_AMD_SDMA_PKT_ATOMIC_TAG_SRC_DATA_HI_UNION_0(Struct): pass
rocr_AMD_SDMA_PKT_ATOMIC_TAG_SRC_DATA_HI_UNION_0._fields_ = [
  ('src_data_63_32', ctypes.c_uint32,32),
]
rocr_AMD_SDMA_PKT_ATOMIC_TAG_SRC_DATA_HI_UNION._anonymous_ = ['_0']
rocr_AMD_SDMA_PKT_ATOMIC_TAG_SRC_DATA_HI_UNION._fields_ = [
  ('_0', rocr_AMD_SDMA_PKT_ATOMIC_TAG_SRC_DATA_HI_UNION_0),
  ('DW_4_DATA', ctypes.c_uint32),
]
class rocr_AMD_SDMA_PKT_ATOMIC_TAG_CMP_DATA_LO_UNION(ctypes.Union): pass
class rocr_AMD_SDMA_PKT_ATOMIC_TAG_CMP_DATA_LO_UNION_0(Struct): pass
rocr_AMD_SDMA_PKT_ATOMIC_TAG_CMP_DATA_LO_UNION_0._fields_ = [
  ('cmp_data_31_0', ctypes.c_uint32,32),
]
rocr_AMD_SDMA_PKT_ATOMIC_TAG_CMP_DATA_LO_UNION._anonymous_ = ['_0']
rocr_AMD_SDMA_PKT_ATOMIC_TAG_CMP_DATA_LO_UNION._fields_ = [
  ('_0', rocr_AMD_SDMA_PKT_ATOMIC_TAG_CMP_DATA_LO_UNION_0),
  ('DW_5_DATA', ctypes.c_uint32),
]
class rocr_AMD_SDMA_PKT_ATOMIC_TAG_CMP_DATA_HI_UNION(ctypes.Union): pass
class rocr_AMD_SDMA_PKT_ATOMIC_TAG_CMP_DATA_HI_UNION_0(Struct): pass
rocr_AMD_SDMA_PKT_ATOMIC_TAG_CMP_DATA_HI_UNION_0._fields_ = [
  ('cmp_data_63_32', ctypes.c_uint32,32),
]
rocr_AMD_SDMA_PKT_ATOMIC_TAG_CMP_DATA_HI_UNION._anonymous_ = ['_0']
rocr_AMD_SDMA_PKT_ATOMIC_TAG_CMP_DATA_HI_UNION._fields_ = [
  ('_0', rocr_AMD_SDMA_PKT_ATOMIC_TAG_CMP_DATA_HI_UNION_0),
  ('DW_6_DATA', ctypes.c_uint32),
]
class rocr_AMD_SDMA_PKT_ATOMIC_TAG_LOOP_UNION(ctypes.Union): pass
class rocr_AMD_SDMA_PKT_ATOMIC_TAG_LOOP_UNION_0(Struct): pass
rocr_AMD_SDMA_PKT_ATOMIC_TAG_LOOP_UNION_0._fields_ = [
  ('loop_interval', ctypes.c_uint32,13),
  ('reserved_0', ctypes.c_uint32,19),
]
rocr_AMD_SDMA_PKT_ATOMIC_TAG_LOOP_UNION._anonymous_ = ['_0']
rocr_AMD_SDMA_PKT_ATOMIC_TAG_LOOP_UNION._fields_ = [
  ('_0', rocr_AMD_SDMA_PKT_ATOMIC_TAG_LOOP_UNION_0),
  ('DW_7_DATA', ctypes.c_uint32),
]
rocr_AMD_SDMA_PKT_ATOMIC_TAG._fields_ = [
  ('HEADER_UNION', rocr_AMD_SDMA_PKT_ATOMIC_TAG_HEADER_UNION),
  ('ADDR_LO_UNION', rocr_AMD_SDMA_PKT_ATOMIC_TAG_ADDR_LO_UNION),
  ('ADDR_HI_UNION', rocr_AMD_SDMA_PKT_ATOMIC_TAG_ADDR_HI_UNION),
  ('SRC_DATA_LO_UNION', rocr_AMD_SDMA_PKT_ATOMIC_TAG_SRC_DATA_LO_UNION),
  ('SRC_DATA_HI_UNION', rocr_AMD_SDMA_PKT_ATOMIC_TAG_SRC_DATA_HI_UNION),
  ('CMP_DATA_LO_UNION', rocr_AMD_SDMA_PKT_ATOMIC_TAG_CMP_DATA_LO_UNION),
  ('CMP_DATA_HI_UNION', rocr_AMD_SDMA_PKT_ATOMIC_TAG_CMP_DATA_HI_UNION),
  ('LOOP_UNION', rocr_AMD_SDMA_PKT_ATOMIC_TAG_LOOP_UNION),
]
rocr_AMD_SDMA_PKT_ATOMIC = rocr_AMD_SDMA_PKT_ATOMIC_TAG
class rocr_AMD_SDMA_PKT_TIMESTAMP_TAG(Struct): pass
class rocr_AMD_SDMA_PKT_TIMESTAMP_TAG_HEADER_UNION(ctypes.Union): pass
class rocr_AMD_SDMA_PKT_TIMESTAMP_TAG_HEADER_UNION_0(Struct): pass
rocr_AMD_SDMA_PKT_TIMESTAMP_TAG_HEADER_UNION_0._fields_ = [
  ('op', ctypes.c_uint32,8),
  ('sub_op', ctypes.c_uint32,8),
  ('reserved_0', ctypes.c_uint32,16),
]
rocr_AMD_SDMA_PKT_TIMESTAMP_TAG_HEADER_UNION._anonymous_ = ['_0']
rocr_AMD_SDMA_PKT_TIMESTAMP_TAG_HEADER_UNION._fields_ = [
  ('_0', rocr_AMD_SDMA_PKT_TIMESTAMP_TAG_HEADER_UNION_0),
  ('DW_0_DATA', ctypes.c_uint32),
]
class rocr_AMD_SDMA_PKT_TIMESTAMP_TAG_ADDR_LO_UNION(ctypes.Union): pass
class rocr_AMD_SDMA_PKT_TIMESTAMP_TAG_ADDR_LO_UNION_0(Struct): pass
rocr_AMD_SDMA_PKT_TIMESTAMP_TAG_ADDR_LO_UNION_0._fields_ = [
  ('addr_31_0', ctypes.c_uint32,32),
]
rocr_AMD_SDMA_PKT_TIMESTAMP_TAG_ADDR_LO_UNION._anonymous_ = ['_0']
rocr_AMD_SDMA_PKT_TIMESTAMP_TAG_ADDR_LO_UNION._fields_ = [
  ('_0', rocr_AMD_SDMA_PKT_TIMESTAMP_TAG_ADDR_LO_UNION_0),
  ('DW_1_DATA', ctypes.c_uint32),
]
class rocr_AMD_SDMA_PKT_TIMESTAMP_TAG_ADDR_HI_UNION(ctypes.Union): pass
class rocr_AMD_SDMA_PKT_TIMESTAMP_TAG_ADDR_HI_UNION_0(Struct): pass
rocr_AMD_SDMA_PKT_TIMESTAMP_TAG_ADDR_HI_UNION_0._fields_ = [
  ('addr_63_32', ctypes.c_uint32,32),
]
rocr_AMD_SDMA_PKT_TIMESTAMP_TAG_ADDR_HI_UNION._anonymous_ = ['_0']
rocr_AMD_SDMA_PKT_TIMESTAMP_TAG_ADDR_HI_UNION._fields_ = [
  ('_0', rocr_AMD_SDMA_PKT_TIMESTAMP_TAG_ADDR_HI_UNION_0),
  ('DW_2_DATA', ctypes.c_uint32),
]
rocr_AMD_SDMA_PKT_TIMESTAMP_TAG._fields_ = [
  ('HEADER_UNION', rocr_AMD_SDMA_PKT_TIMESTAMP_TAG_HEADER_UNION),
  ('ADDR_LO_UNION', rocr_AMD_SDMA_PKT_TIMESTAMP_TAG_ADDR_LO_UNION),
  ('ADDR_HI_UNION', rocr_AMD_SDMA_PKT_TIMESTAMP_TAG_ADDR_HI_UNION),
]
rocr_AMD_SDMA_PKT_TIMESTAMP = rocr_AMD_SDMA_PKT_TIMESTAMP_TAG
class rocr_AMD_SDMA_PKT_TRAP_TAG(Struct): pass
class rocr_AMD_SDMA_PKT_TRAP_TAG_HEADER_UNION(ctypes.Union): pass
class rocr_AMD_SDMA_PKT_TRAP_TAG_HEADER_UNION_0(Struct): pass
rocr_AMD_SDMA_PKT_TRAP_TAG_HEADER_UNION_0._fields_ = [
  ('op', ctypes.c_uint32,8),
  ('sub_op', ctypes.c_uint32,8),
  ('reserved_0', ctypes.c_uint32,16),
]
rocr_AMD_SDMA_PKT_TRAP_TAG_HEADER_UNION._anonymous_ = ['_0']
rocr_AMD_SDMA_PKT_TRAP_TAG_HEADER_UNION._fields_ = [
  ('_0', rocr_AMD_SDMA_PKT_TRAP_TAG_HEADER_UNION_0),
  ('DW_0_DATA', ctypes.c_uint32),
]
class rocr_AMD_SDMA_PKT_TRAP_TAG_INT_CONTEXT_UNION(ctypes.Union): pass
class rocr_AMD_SDMA_PKT_TRAP_TAG_INT_CONTEXT_UNION_0(Struct): pass
rocr_AMD_SDMA_PKT_TRAP_TAG_INT_CONTEXT_UNION_0._fields_ = [
  ('int_ctx', ctypes.c_uint32,28),
  ('reserved_1', ctypes.c_uint32,4),
]
rocr_AMD_SDMA_PKT_TRAP_TAG_INT_CONTEXT_UNION._anonymous_ = ['_0']
rocr_AMD_SDMA_PKT_TRAP_TAG_INT_CONTEXT_UNION._fields_ = [
  ('_0', rocr_AMD_SDMA_PKT_TRAP_TAG_INT_CONTEXT_UNION_0),
  ('DW_1_DATA', ctypes.c_uint32),
]
rocr_AMD_SDMA_PKT_TRAP_TAG._fields_ = [
  ('HEADER_UNION', rocr_AMD_SDMA_PKT_TRAP_TAG_HEADER_UNION),
  ('INT_CONTEXT_UNION', rocr_AMD_SDMA_PKT_TRAP_TAG_INT_CONTEXT_UNION),
]
rocr_AMD_SDMA_PKT_TRAP = rocr_AMD_SDMA_PKT_TRAP_TAG
class rocr_AMD_SDMA_PKT_HDP_FLUSH_TAG(Struct): pass
rocr_AMD_SDMA_PKT_HDP_FLUSH_TAG._fields_ = [
  ('DW_0_DATA', ctypes.c_uint32),
  ('DW_1_DATA', ctypes.c_uint32),
  ('DW_2_DATA', ctypes.c_uint32),
  ('DW_3_DATA', ctypes.c_uint32),
  ('DW_4_DATA', ctypes.c_uint32),
  ('DW_5_DATA', ctypes.c_uint32),
]
rocr_AMD_SDMA_PKT_HDP_FLUSH = rocr_AMD_SDMA_PKT_HDP_FLUSH_TAG
class rocr_AMD_SDMA_PKT_GCR_TAG(Struct): pass
class rocr_AMD_SDMA_PKT_GCR_TAG_HEADER_UNION(ctypes.Union): pass
class rocr_AMD_SDMA_PKT_GCR_TAG_HEADER_UNION_0(Struct): pass
rocr_AMD_SDMA_PKT_GCR_TAG_HEADER_UNION_0._fields_ = [
  ('op', ctypes.c_uint32,8),
  ('sub_op', ctypes.c_uint32,8),
  ('', ctypes.c_uint32,16),
]
rocr_AMD_SDMA_PKT_GCR_TAG_HEADER_UNION._anonymous_ = ['_0']
rocr_AMD_SDMA_PKT_GCR_TAG_HEADER_UNION._fields_ = [
  ('_0', rocr_AMD_SDMA_PKT_GCR_TAG_HEADER_UNION_0),
  ('DW_0_DATA', ctypes.c_uint32),
]
class rocr_AMD_SDMA_PKT_GCR_TAG_WORD1_UNION(ctypes.Union): pass
class rocr_AMD_SDMA_PKT_GCR_TAG_WORD1_UNION_0(Struct): pass
rocr_AMD_SDMA_PKT_GCR_TAG_WORD1_UNION_0._fields_ = [
  ('', ctypes.c_uint32,7),
  ('BaseVA_LO', ctypes.c_uint32,25),
]
rocr_AMD_SDMA_PKT_GCR_TAG_WORD1_UNION._anonymous_ = ['_0']
rocr_AMD_SDMA_PKT_GCR_TAG_WORD1_UNION._fields_ = [
  ('_0', rocr_AMD_SDMA_PKT_GCR_TAG_WORD1_UNION_0),
  ('DW_1_DATA', ctypes.c_uint32),
]
class rocr_AMD_SDMA_PKT_GCR_TAG_WORD2_UNION(ctypes.Union): pass
class rocr_AMD_SDMA_PKT_GCR_TAG_WORD2_UNION_0(Struct): pass
rocr_AMD_SDMA_PKT_GCR_TAG_WORD2_UNION_0._fields_ = [
  ('BaseVA_HI', ctypes.c_uint32,16),
  ('GCR_CONTROL_GLI_INV', ctypes.c_uint32,2),
  ('GCR_CONTROL_GL1_RANGE', ctypes.c_uint32,2),
  ('GCR_CONTROL_GLM_WB', ctypes.c_uint32,1),
  ('GCR_CONTROL_GLM_INV', ctypes.c_uint32,1),
  ('GCR_CONTROL_GLK_WB', ctypes.c_uint32,1),
  ('GCR_CONTROL_GLK_INV', ctypes.c_uint32,1),
  ('GCR_CONTROL_GLV_INV', ctypes.c_uint32,1),
  ('GCR_CONTROL_GL1_INV', ctypes.c_uint32,1),
  ('GCR_CONTROL_GL2_US', ctypes.c_uint32,1),
  ('GCR_CONTROL_GL2_RANGE', ctypes.c_uint32,2),
  ('GCR_CONTROL_GL2_DISCARD', ctypes.c_uint32,1),
  ('GCR_CONTROL_GL2_INV', ctypes.c_uint32,1),
  ('GCR_CONTROL_GL2_WB', ctypes.c_uint32,1),
]
rocr_AMD_SDMA_PKT_GCR_TAG_WORD2_UNION._anonymous_ = ['_0']
rocr_AMD_SDMA_PKT_GCR_TAG_WORD2_UNION._fields_ = [
  ('_0', rocr_AMD_SDMA_PKT_GCR_TAG_WORD2_UNION_0),
  ('DW_2_DATA', ctypes.c_uint32),
]
class rocr_AMD_SDMA_PKT_GCR_TAG_WORD3_UNION(ctypes.Union): pass
class rocr_AMD_SDMA_PKT_GCR_TAG_WORD3_UNION_0(Struct): pass
rocr_AMD_SDMA_PKT_GCR_TAG_WORD3_UNION_0._fields_ = [
  ('GCR_CONTROL_RANGE_IS_PA', ctypes.c_uint32,1),
  ('GCR_CONTROL_SEQ', ctypes.c_uint32,2),
  ('', ctypes.c_uint32,4),
  ('LimitVA_LO', ctypes.c_uint32,25),
]
rocr_AMD_SDMA_PKT_GCR_TAG_WORD3_UNION._anonymous_ = ['_0']
rocr_AMD_SDMA_PKT_GCR_TAG_WORD3_UNION._fields_ = [
  ('_0', rocr_AMD_SDMA_PKT_GCR_TAG_WORD3_UNION_0),
  ('DW_3_DATA', ctypes.c_uint32),
]
class rocr_AMD_SDMA_PKT_GCR_TAG_WORD4_UNION(ctypes.Union): pass
class rocr_AMD_SDMA_PKT_GCR_TAG_WORD4_UNION_0(Struct): pass
rocr_AMD_SDMA_PKT_GCR_TAG_WORD4_UNION_0._fields_ = [
  ('LimitVA_HI', ctypes.c_uint32,16),
  ('', ctypes.c_uint32,8),
  ('VMID', ctypes.c_uint32,4),
  ('', ctypes.c_uint32,4),
]
rocr_AMD_SDMA_PKT_GCR_TAG_WORD4_UNION._anonymous_ = ['_0']
rocr_AMD_SDMA_PKT_GCR_TAG_WORD4_UNION._fields_ = [
  ('_0', rocr_AMD_SDMA_PKT_GCR_TAG_WORD4_UNION_0),
  ('DW_4_DATA', ctypes.c_uint32),
]
rocr_AMD_SDMA_PKT_GCR_TAG._fields_ = [
  ('HEADER_UNION', rocr_AMD_SDMA_PKT_GCR_TAG_HEADER_UNION),
  ('WORD1_UNION', rocr_AMD_SDMA_PKT_GCR_TAG_WORD1_UNION),
  ('WORD2_UNION', rocr_AMD_SDMA_PKT_GCR_TAG_WORD2_UNION),
  ('WORD3_UNION', rocr_AMD_SDMA_PKT_GCR_TAG_WORD3_UNION),
  ('WORD4_UNION', rocr_AMD_SDMA_PKT_GCR_TAG_WORD4_UNION),
]
rocr_AMD_SDMA_PKT_GCR = rocr_AMD_SDMA_PKT_GCR_TAG
SDMA_OP_COPY = 1
SDMA_OP_FENCE = 5
SDMA_OP_TRAP = 6
SDMA_OP_POLL_REGMEM = 8
SDMA_OP_ATOMIC = 10
SDMA_OP_CONST_FILL = 11
SDMA_OP_TIMESTAMP = 13
SDMA_OP_GCR = 17
SDMA_SUBOP_COPY_LINEAR = 0
SDMA_SUBOP_COPY_LINEAR_RECT = 4
SDMA_SUBOP_TIMESTAMP_GET_GLOBAL = 2
SDMA_SUBOP_USER_GCR = 1
SDMA_ATOMIC_ADD64 = 47
SDMA_OP_NOP = 0
SDMA_OP_COPY = 1
SDMA_OP_WRITE = 2
SDMA_OP_INDIRECT = 4
SDMA_OP_FENCE = 5
SDMA_OP_TRAP = 6
SDMA_OP_SEM = 7
SDMA_OP_POLL_REGMEM = 8
SDMA_OP_COND_EXE = 9
SDMA_OP_ATOMIC = 10
SDMA_OP_CONST_FILL = 11
SDMA_OP_PTEPDE = 12
SDMA_OP_TIMESTAMP = 13
SDMA_OP_SRBM_WRITE = 14
SDMA_OP_PRE_EXE = 15
SDMA_OP_GPUVM_INV = 16
SDMA_OP_GCR_REQ = 17
SDMA_OP_DUMMY_TRAP = 32
SDMA_SUBOP_TIMESTAMP_SET = 0
SDMA_SUBOP_TIMESTAMP_GET = 1
SDMA_SUBOP_TIMESTAMP_GET_GLOBAL = 2
SDMA_SUBOP_COPY_LINEAR = 0
SDMA_SUBOP_COPY_LINEAR_SUB_WIND = 4
SDMA_SUBOP_COPY_TILED = 1
SDMA_SUBOP_COPY_TILED_SUB_WIND = 5
SDMA_SUBOP_COPY_T2T_SUB_WIND = 6
SDMA_SUBOP_COPY_SOA = 3
SDMA_SUBOP_COPY_DIRTY_PAGE = 7
SDMA_SUBOP_COPY_LINEAR_PHY = 8
SDMA_SUBOP_COPY_LINEAR_SUB_WIND_LARGE = 36
SDMA_SUBOP_COPY_LINEAR_BC = 16
SDMA_SUBOP_COPY_TILED_BC = 17
SDMA_SUBOP_COPY_LINEAR_SUB_WIND_BC = 20
SDMA_SUBOP_COPY_TILED_SUB_WIND_BC = 21
SDMA_SUBOP_COPY_T2T_SUB_WIND_BC = 22
SDMA_SUBOP_WRITE_LINEAR = 0
SDMA_SUBOP_WRITE_TILED = 1
SDMA_SUBOP_WRITE_TILED_BC = 17
SDMA_SUBOP_PTEPDE_GEN = 0
SDMA_SUBOP_PTEPDE_COPY = 1
SDMA_SUBOP_PTEPDE_RMW = 2
SDMA_SUBOP_PTEPDE_COPY_BACKWARDS = 3
SDMA_SUBOP_MEM_INCR = 1
SDMA_SUBOP_DATA_FILL_MULTI = 1
SDMA_SUBOP_POLL_REG_WRITE_MEM = 1
SDMA_SUBOP_POLL_DBIT_WRITE_MEM = 2
SDMA_SUBOP_POLL_MEM_VERIFY = 3
SDMA_SUBOP_VM_INVALIDATION = 4
HEADER_AGENT_DISPATCH = 4
HEADER_BARRIER = 5
SDMA_OP_AQL_COPY = 0
SDMA_OP_AQL_BARRIER_OR = 0
SDMA_GCR_RANGE_IS_PA = (1 << 18)
SDMA_GCR_SEQ = lambda x: (((x) & 0x3) << 16)
SDMA_GCR_GL2_WB = (1 << 15)
SDMA_GCR_GL2_INV = (1 << 14)
SDMA_GCR_GL2_DISCARD = (1 << 13)
SDMA_GCR_GL2_RANGE = lambda x: (((x) & 0x3) << 11)
SDMA_GCR_GL2_US = (1 << 10)
SDMA_GCR_GL1_INV = (1 << 9)
SDMA_GCR_GLV_INV = (1 << 8)
SDMA_GCR_GLK_INV = (1 << 7)
SDMA_GCR_GLK_WB = (1 << 6)
SDMA_GCR_GLM_INV = (1 << 5)
SDMA_GCR_GLM_WB = (1 << 4)
SDMA_GCR_GL1_RANGE = lambda x: (((x) & 0x3) << 2)
SDMA_GCR_GLI_INV = lambda x: (((x) & 0x3) << 0)
SDMA_DCC_DATA_FORMAT = lambda x: ((x) & 0x3f)
SDMA_DCC_NUM_TYPE = lambda x: (((x) & 0x7) << 9)
SDMA_DCC_READ_CM = lambda x: (((x) & 0x3) << 16)
SDMA_DCC_WRITE_CM = lambda x: (((x) & 0x3) << 18)
SDMA_DCC_MAX_COM = lambda x: (((x) & 0x3) << 24)
SDMA_DCC_MAX_UCOM = lambda x: (((x) & 0x1) << 26)
SDMA_PKT_COPY_LINEAR_HEADER_op_offset = 0
SDMA_PKT_COPY_LINEAR_HEADER_op_mask = 0x000000FF
SDMA_PKT_COPY_LINEAR_HEADER_op_shift = 0
SDMA_PKT_COPY_LINEAR_HEADER_OP = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_HEADER_op_mask) << SDMA_PKT_COPY_LINEAR_HEADER_op_shift)
SDMA_PKT_COPY_LINEAR_HEADER_sub_op_offset = 0
SDMA_PKT_COPY_LINEAR_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_COPY_LINEAR_HEADER_sub_op_shift = 8
SDMA_PKT_COPY_LINEAR_HEADER_SUB_OP = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_HEADER_sub_op_mask) << SDMA_PKT_COPY_LINEAR_HEADER_sub_op_shift)
SDMA_PKT_COPY_LINEAR_HEADER_encrypt_offset = 0
SDMA_PKT_COPY_LINEAR_HEADER_encrypt_mask = 0x00000001
SDMA_PKT_COPY_LINEAR_HEADER_encrypt_shift = 16
SDMA_PKT_COPY_LINEAR_HEADER_ENCRYPT = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_HEADER_encrypt_mask) << SDMA_PKT_COPY_LINEAR_HEADER_encrypt_shift)
SDMA_PKT_COPY_LINEAR_HEADER_tmz_offset = 0
SDMA_PKT_COPY_LINEAR_HEADER_tmz_mask = 0x00000001
SDMA_PKT_COPY_LINEAR_HEADER_tmz_shift = 18
SDMA_PKT_COPY_LINEAR_HEADER_TMZ = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_HEADER_tmz_mask) << SDMA_PKT_COPY_LINEAR_HEADER_tmz_shift)
SDMA_PKT_COPY_LINEAR_HEADER_cpv_offset = 0
SDMA_PKT_COPY_LINEAR_HEADER_cpv_mask = 0x00000001
SDMA_PKT_COPY_LINEAR_HEADER_cpv_shift = 19
SDMA_PKT_COPY_LINEAR_HEADER_CPV = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_HEADER_cpv_mask) << SDMA_PKT_COPY_LINEAR_HEADER_cpv_shift)
SDMA_PKT_COPY_LINEAR_HEADER_backwards_offset = 0
SDMA_PKT_COPY_LINEAR_HEADER_backwards_mask = 0x00000001
SDMA_PKT_COPY_LINEAR_HEADER_backwards_shift = 25
SDMA_PKT_COPY_LINEAR_HEADER_BACKWARDS = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_HEADER_backwards_mask) << SDMA_PKT_COPY_LINEAR_HEADER_backwards_shift)
SDMA_PKT_COPY_LINEAR_HEADER_broadcast_offset = 0
SDMA_PKT_COPY_LINEAR_HEADER_broadcast_mask = 0x00000001
SDMA_PKT_COPY_LINEAR_HEADER_broadcast_shift = 27
SDMA_PKT_COPY_LINEAR_HEADER_BROADCAST = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_HEADER_broadcast_mask) << SDMA_PKT_COPY_LINEAR_HEADER_broadcast_shift)
SDMA_PKT_COPY_LINEAR_COUNT_count_offset = 1
SDMA_PKT_COPY_LINEAR_COUNT_count_mask = 0x3FFFFFFF
SDMA_PKT_COPY_LINEAR_COUNT_count_shift = 0
SDMA_PKT_COPY_LINEAR_COUNT_COUNT = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_COUNT_count_mask) << SDMA_PKT_COPY_LINEAR_COUNT_count_shift)
SDMA_PKT_COPY_LINEAR_PARAMETER_dst_sw_offset = 2
SDMA_PKT_COPY_LINEAR_PARAMETER_dst_sw_mask = 0x00000003
SDMA_PKT_COPY_LINEAR_PARAMETER_dst_sw_shift = 16
SDMA_PKT_COPY_LINEAR_PARAMETER_DST_SW = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_PARAMETER_dst_sw_mask) << SDMA_PKT_COPY_LINEAR_PARAMETER_dst_sw_shift)
SDMA_PKT_COPY_LINEAR_PARAMETER_dst_cache_policy_offset = 2
SDMA_PKT_COPY_LINEAR_PARAMETER_dst_cache_policy_mask = 0x00000007
SDMA_PKT_COPY_LINEAR_PARAMETER_dst_cache_policy_shift = 18
SDMA_PKT_COPY_LINEAR_PARAMETER_DST_CACHE_POLICY = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_PARAMETER_dst_cache_policy_mask) << SDMA_PKT_COPY_LINEAR_PARAMETER_dst_cache_policy_shift)
SDMA_PKT_COPY_LINEAR_PARAMETER_src_sw_offset = 2
SDMA_PKT_COPY_LINEAR_PARAMETER_src_sw_mask = 0x00000003
SDMA_PKT_COPY_LINEAR_PARAMETER_src_sw_shift = 24
SDMA_PKT_COPY_LINEAR_PARAMETER_SRC_SW = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_PARAMETER_src_sw_mask) << SDMA_PKT_COPY_LINEAR_PARAMETER_src_sw_shift)
SDMA_PKT_COPY_LINEAR_PARAMETER_src_cache_policy_offset = 2
SDMA_PKT_COPY_LINEAR_PARAMETER_src_cache_policy_mask = 0x00000007
SDMA_PKT_COPY_LINEAR_PARAMETER_src_cache_policy_shift = 26
SDMA_PKT_COPY_LINEAR_PARAMETER_SRC_CACHE_POLICY = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_PARAMETER_src_cache_policy_mask) << SDMA_PKT_COPY_LINEAR_PARAMETER_src_cache_policy_shift)
SDMA_PKT_COPY_LINEAR_SRC_ADDR_LO_src_addr_31_0_offset = 3
SDMA_PKT_COPY_LINEAR_SRC_ADDR_LO_src_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_LINEAR_SRC_ADDR_LO_src_addr_31_0_shift = 0
SDMA_PKT_COPY_LINEAR_SRC_ADDR_LO_SRC_ADDR_31_0 = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SRC_ADDR_LO_src_addr_31_0_mask) << SDMA_PKT_COPY_LINEAR_SRC_ADDR_LO_src_addr_31_0_shift)
SDMA_PKT_COPY_LINEAR_SRC_ADDR_HI_src_addr_63_32_offset = 4
SDMA_PKT_COPY_LINEAR_SRC_ADDR_HI_src_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_LINEAR_SRC_ADDR_HI_src_addr_63_32_shift = 0
SDMA_PKT_COPY_LINEAR_SRC_ADDR_HI_SRC_ADDR_63_32 = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SRC_ADDR_HI_src_addr_63_32_mask) << SDMA_PKT_COPY_LINEAR_SRC_ADDR_HI_src_addr_63_32_shift)
SDMA_PKT_COPY_LINEAR_DST_ADDR_LO_dst_addr_31_0_offset = 5
SDMA_PKT_COPY_LINEAR_DST_ADDR_LO_dst_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_LINEAR_DST_ADDR_LO_dst_addr_31_0_shift = 0
SDMA_PKT_COPY_LINEAR_DST_ADDR_LO_DST_ADDR_31_0 = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_DST_ADDR_LO_dst_addr_31_0_mask) << SDMA_PKT_COPY_LINEAR_DST_ADDR_LO_dst_addr_31_0_shift)
SDMA_PKT_COPY_LINEAR_DST_ADDR_HI_dst_addr_63_32_offset = 6
SDMA_PKT_COPY_LINEAR_DST_ADDR_HI_dst_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_LINEAR_DST_ADDR_HI_dst_addr_63_32_shift = 0
SDMA_PKT_COPY_LINEAR_DST_ADDR_HI_DST_ADDR_63_32 = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_DST_ADDR_HI_dst_addr_63_32_mask) << SDMA_PKT_COPY_LINEAR_DST_ADDR_HI_dst_addr_63_32_shift)
SDMA_PKT_COPY_LINEAR_BC_HEADER_op_offset = 0
SDMA_PKT_COPY_LINEAR_BC_HEADER_op_mask = 0x000000FF
SDMA_PKT_COPY_LINEAR_BC_HEADER_op_shift = 0
SDMA_PKT_COPY_LINEAR_BC_HEADER_OP = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_BC_HEADER_op_mask) << SDMA_PKT_COPY_LINEAR_BC_HEADER_op_shift)
SDMA_PKT_COPY_LINEAR_BC_HEADER_sub_op_offset = 0
SDMA_PKT_COPY_LINEAR_BC_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_COPY_LINEAR_BC_HEADER_sub_op_shift = 8
SDMA_PKT_COPY_LINEAR_BC_HEADER_SUB_OP = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_BC_HEADER_sub_op_mask) << SDMA_PKT_COPY_LINEAR_BC_HEADER_sub_op_shift)
SDMA_PKT_COPY_LINEAR_BC_COUNT_count_offset = 1
SDMA_PKT_COPY_LINEAR_BC_COUNT_count_mask = 0x003FFFFF
SDMA_PKT_COPY_LINEAR_BC_COUNT_count_shift = 0
SDMA_PKT_COPY_LINEAR_BC_COUNT_COUNT = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_BC_COUNT_count_mask) << SDMA_PKT_COPY_LINEAR_BC_COUNT_count_shift)
SDMA_PKT_COPY_LINEAR_BC_PARAMETER_dst_sw_offset = 2
SDMA_PKT_COPY_LINEAR_BC_PARAMETER_dst_sw_mask = 0x00000003
SDMA_PKT_COPY_LINEAR_BC_PARAMETER_dst_sw_shift = 16
SDMA_PKT_COPY_LINEAR_BC_PARAMETER_DST_SW = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_BC_PARAMETER_dst_sw_mask) << SDMA_PKT_COPY_LINEAR_BC_PARAMETER_dst_sw_shift)
SDMA_PKT_COPY_LINEAR_BC_PARAMETER_dst_ha_offset = 2
SDMA_PKT_COPY_LINEAR_BC_PARAMETER_dst_ha_mask = 0x00000001
SDMA_PKT_COPY_LINEAR_BC_PARAMETER_dst_ha_shift = 19
SDMA_PKT_COPY_LINEAR_BC_PARAMETER_DST_HA = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_BC_PARAMETER_dst_ha_mask) << SDMA_PKT_COPY_LINEAR_BC_PARAMETER_dst_ha_shift)
SDMA_PKT_COPY_LINEAR_BC_PARAMETER_src_sw_offset = 2
SDMA_PKT_COPY_LINEAR_BC_PARAMETER_src_sw_mask = 0x00000003
SDMA_PKT_COPY_LINEAR_BC_PARAMETER_src_sw_shift = 24
SDMA_PKT_COPY_LINEAR_BC_PARAMETER_SRC_SW = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_BC_PARAMETER_src_sw_mask) << SDMA_PKT_COPY_LINEAR_BC_PARAMETER_src_sw_shift)
SDMA_PKT_COPY_LINEAR_BC_PARAMETER_src_ha_offset = 2
SDMA_PKT_COPY_LINEAR_BC_PARAMETER_src_ha_mask = 0x00000001
SDMA_PKT_COPY_LINEAR_BC_PARAMETER_src_ha_shift = 27
SDMA_PKT_COPY_LINEAR_BC_PARAMETER_SRC_HA = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_BC_PARAMETER_src_ha_mask) << SDMA_PKT_COPY_LINEAR_BC_PARAMETER_src_ha_shift)
SDMA_PKT_COPY_LINEAR_BC_SRC_ADDR_LO_src_addr_31_0_offset = 3
SDMA_PKT_COPY_LINEAR_BC_SRC_ADDR_LO_src_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_LINEAR_BC_SRC_ADDR_LO_src_addr_31_0_shift = 0
SDMA_PKT_COPY_LINEAR_BC_SRC_ADDR_LO_SRC_ADDR_31_0 = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_BC_SRC_ADDR_LO_src_addr_31_0_mask) << SDMA_PKT_COPY_LINEAR_BC_SRC_ADDR_LO_src_addr_31_0_shift)
SDMA_PKT_COPY_LINEAR_BC_SRC_ADDR_HI_src_addr_63_32_offset = 4
SDMA_PKT_COPY_LINEAR_BC_SRC_ADDR_HI_src_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_LINEAR_BC_SRC_ADDR_HI_src_addr_63_32_shift = 0
SDMA_PKT_COPY_LINEAR_BC_SRC_ADDR_HI_SRC_ADDR_63_32 = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_BC_SRC_ADDR_HI_src_addr_63_32_mask) << SDMA_PKT_COPY_LINEAR_BC_SRC_ADDR_HI_src_addr_63_32_shift)
SDMA_PKT_COPY_LINEAR_BC_DST_ADDR_LO_dst_addr_31_0_offset = 5
SDMA_PKT_COPY_LINEAR_BC_DST_ADDR_LO_dst_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_LINEAR_BC_DST_ADDR_LO_dst_addr_31_0_shift = 0
SDMA_PKT_COPY_LINEAR_BC_DST_ADDR_LO_DST_ADDR_31_0 = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_BC_DST_ADDR_LO_dst_addr_31_0_mask) << SDMA_PKT_COPY_LINEAR_BC_DST_ADDR_LO_dst_addr_31_0_shift)
SDMA_PKT_COPY_LINEAR_BC_DST_ADDR_HI_dst_addr_63_32_offset = 6
SDMA_PKT_COPY_LINEAR_BC_DST_ADDR_HI_dst_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_LINEAR_BC_DST_ADDR_HI_dst_addr_63_32_shift = 0
SDMA_PKT_COPY_LINEAR_BC_DST_ADDR_HI_DST_ADDR_63_32 = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_BC_DST_ADDR_HI_dst_addr_63_32_mask) << SDMA_PKT_COPY_LINEAR_BC_DST_ADDR_HI_dst_addr_63_32_shift)
SDMA_PKT_COPY_DIRTY_PAGE_HEADER_op_offset = 0
SDMA_PKT_COPY_DIRTY_PAGE_HEADER_op_mask = 0x000000FF
SDMA_PKT_COPY_DIRTY_PAGE_HEADER_op_shift = 0
SDMA_PKT_COPY_DIRTY_PAGE_HEADER_OP = lambda x: (((x) & SDMA_PKT_COPY_DIRTY_PAGE_HEADER_op_mask) << SDMA_PKT_COPY_DIRTY_PAGE_HEADER_op_shift)
SDMA_PKT_COPY_DIRTY_PAGE_HEADER_sub_op_offset = 0
SDMA_PKT_COPY_DIRTY_PAGE_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_COPY_DIRTY_PAGE_HEADER_sub_op_shift = 8
SDMA_PKT_COPY_DIRTY_PAGE_HEADER_SUB_OP = lambda x: (((x) & SDMA_PKT_COPY_DIRTY_PAGE_HEADER_sub_op_mask) << SDMA_PKT_COPY_DIRTY_PAGE_HEADER_sub_op_shift)
SDMA_PKT_COPY_DIRTY_PAGE_HEADER_tmz_offset = 0
SDMA_PKT_COPY_DIRTY_PAGE_HEADER_tmz_mask = 0x00000001
SDMA_PKT_COPY_DIRTY_PAGE_HEADER_tmz_shift = 18
SDMA_PKT_COPY_DIRTY_PAGE_HEADER_TMZ = lambda x: (((x) & SDMA_PKT_COPY_DIRTY_PAGE_HEADER_tmz_mask) << SDMA_PKT_COPY_DIRTY_PAGE_HEADER_tmz_shift)
SDMA_PKT_COPY_DIRTY_PAGE_HEADER_cpv_offset = 0
SDMA_PKT_COPY_DIRTY_PAGE_HEADER_cpv_mask = 0x00000001
SDMA_PKT_COPY_DIRTY_PAGE_HEADER_cpv_shift = 19
SDMA_PKT_COPY_DIRTY_PAGE_HEADER_CPV = lambda x: (((x) & SDMA_PKT_COPY_DIRTY_PAGE_HEADER_cpv_mask) << SDMA_PKT_COPY_DIRTY_PAGE_HEADER_cpv_shift)
SDMA_PKT_COPY_DIRTY_PAGE_HEADER_all_offset = 0
SDMA_PKT_COPY_DIRTY_PAGE_HEADER_all_mask = 0x00000001
SDMA_PKT_COPY_DIRTY_PAGE_HEADER_all_shift = 31
SDMA_PKT_COPY_DIRTY_PAGE_HEADER_ALL = lambda x: (((x) & SDMA_PKT_COPY_DIRTY_PAGE_HEADER_all_mask) << SDMA_PKT_COPY_DIRTY_PAGE_HEADER_all_shift)
SDMA_PKT_COPY_DIRTY_PAGE_COUNT_count_offset = 1
SDMA_PKT_COPY_DIRTY_PAGE_COUNT_count_mask = 0x003FFFFF
SDMA_PKT_COPY_DIRTY_PAGE_COUNT_count_shift = 0
SDMA_PKT_COPY_DIRTY_PAGE_COUNT_COUNT = lambda x: (((x) & SDMA_PKT_COPY_DIRTY_PAGE_COUNT_count_mask) << SDMA_PKT_COPY_DIRTY_PAGE_COUNT_count_shift)
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_mtype_offset = 2
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_mtype_mask = 0x00000007
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_mtype_shift = 3
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_DST_MTYPE = lambda x: (((x) & SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_mtype_mask) << SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_mtype_shift)
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_l2_policy_offset = 2
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_l2_policy_mask = 0x00000003
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_l2_policy_shift = 6
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_DST_L2_POLICY = lambda x: (((x) & SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_l2_policy_mask) << SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_l2_policy_shift)
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_llc_offset = 2
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_llc_mask = 0x00000001
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_llc_shift = 8
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_DST_LLC = lambda x: (((x) & SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_llc_mask) << SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_llc_shift)
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_mtype_offset = 2
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_mtype_mask = 0x00000007
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_mtype_shift = 11
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_SRC_MTYPE = lambda x: (((x) & SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_mtype_mask) << SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_mtype_shift)
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_l2_policy_offset = 2
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_l2_policy_mask = 0x00000003
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_l2_policy_shift = 14
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_SRC_L2_POLICY = lambda x: (((x) & SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_l2_policy_mask) << SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_l2_policy_shift)
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_llc_offset = 2
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_llc_mask = 0x00000001
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_llc_shift = 16
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_SRC_LLC = lambda x: (((x) & SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_llc_mask) << SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_llc_shift)
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_sw_offset = 2
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_sw_mask = 0x00000003
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_sw_shift = 17
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_DST_SW = lambda x: (((x) & SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_sw_mask) << SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_sw_shift)
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_gcc_offset = 2
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_gcc_mask = 0x00000001
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_gcc_shift = 19
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_DST_GCC = lambda x: (((x) & SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_gcc_mask) << SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_gcc_shift)
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_sys_offset = 2
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_sys_mask = 0x00000001
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_sys_shift = 20
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_DST_SYS = lambda x: (((x) & SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_sys_mask) << SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_sys_shift)
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_snoop_offset = 2
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_snoop_mask = 0x00000001
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_snoop_shift = 22
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_DST_SNOOP = lambda x: (((x) & SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_snoop_mask) << SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_snoop_shift)
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_gpa_offset = 2
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_gpa_mask = 0x00000001
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_gpa_shift = 23
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_DST_GPA = lambda x: (((x) & SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_gpa_mask) << SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_gpa_shift)
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_sw_offset = 2
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_sw_mask = 0x00000003
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_sw_shift = 24
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_SRC_SW = lambda x: (((x) & SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_sw_mask) << SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_sw_shift)
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_sys_offset = 2
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_sys_mask = 0x00000001
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_sys_shift = 28
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_SRC_SYS = lambda x: (((x) & SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_sys_mask) << SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_sys_shift)
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_snoop_offset = 2
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_snoop_mask = 0x00000001
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_snoop_shift = 30
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_SRC_SNOOP = lambda x: (((x) & SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_snoop_mask) << SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_snoop_shift)
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_gpa_offset = 2
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_gpa_mask = 0x00000001
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_gpa_shift = 31
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_SRC_GPA = lambda x: (((x) & SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_gpa_mask) << SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_gpa_shift)
SDMA_PKT_COPY_DIRTY_PAGE_SRC_ADDR_LO_src_addr_31_0_offset = 3
SDMA_PKT_COPY_DIRTY_PAGE_SRC_ADDR_LO_src_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_DIRTY_PAGE_SRC_ADDR_LO_src_addr_31_0_shift = 0
SDMA_PKT_COPY_DIRTY_PAGE_SRC_ADDR_LO_SRC_ADDR_31_0 = lambda x: (((x) & SDMA_PKT_COPY_DIRTY_PAGE_SRC_ADDR_LO_src_addr_31_0_mask) << SDMA_PKT_COPY_DIRTY_PAGE_SRC_ADDR_LO_src_addr_31_0_shift)
SDMA_PKT_COPY_DIRTY_PAGE_SRC_ADDR_HI_src_addr_63_32_offset = 4
SDMA_PKT_COPY_DIRTY_PAGE_SRC_ADDR_HI_src_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_DIRTY_PAGE_SRC_ADDR_HI_src_addr_63_32_shift = 0
SDMA_PKT_COPY_DIRTY_PAGE_SRC_ADDR_HI_SRC_ADDR_63_32 = lambda x: (((x) & SDMA_PKT_COPY_DIRTY_PAGE_SRC_ADDR_HI_src_addr_63_32_mask) << SDMA_PKT_COPY_DIRTY_PAGE_SRC_ADDR_HI_src_addr_63_32_shift)
SDMA_PKT_COPY_DIRTY_PAGE_DST_ADDR_LO_dst_addr_31_0_offset = 5
SDMA_PKT_COPY_DIRTY_PAGE_DST_ADDR_LO_dst_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_DIRTY_PAGE_DST_ADDR_LO_dst_addr_31_0_shift = 0
SDMA_PKT_COPY_DIRTY_PAGE_DST_ADDR_LO_DST_ADDR_31_0 = lambda x: (((x) & SDMA_PKT_COPY_DIRTY_PAGE_DST_ADDR_LO_dst_addr_31_0_mask) << SDMA_PKT_COPY_DIRTY_PAGE_DST_ADDR_LO_dst_addr_31_0_shift)
SDMA_PKT_COPY_DIRTY_PAGE_DST_ADDR_HI_dst_addr_63_32_offset = 6
SDMA_PKT_COPY_DIRTY_PAGE_DST_ADDR_HI_dst_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_DIRTY_PAGE_DST_ADDR_HI_dst_addr_63_32_shift = 0
SDMA_PKT_COPY_DIRTY_PAGE_DST_ADDR_HI_DST_ADDR_63_32 = lambda x: (((x) & SDMA_PKT_COPY_DIRTY_PAGE_DST_ADDR_HI_dst_addr_63_32_mask) << SDMA_PKT_COPY_DIRTY_PAGE_DST_ADDR_HI_dst_addr_63_32_shift)
SDMA_PKT_COPY_PHYSICAL_LINEAR_HEADER_op_offset = 0
SDMA_PKT_COPY_PHYSICAL_LINEAR_HEADER_op_mask = 0x000000FF
SDMA_PKT_COPY_PHYSICAL_LINEAR_HEADER_op_shift = 0
SDMA_PKT_COPY_PHYSICAL_LINEAR_HEADER_OP = lambda x: (((x) & SDMA_PKT_COPY_PHYSICAL_LINEAR_HEADER_op_mask) << SDMA_PKT_COPY_PHYSICAL_LINEAR_HEADER_op_shift)
SDMA_PKT_COPY_PHYSICAL_LINEAR_HEADER_sub_op_offset = 0
SDMA_PKT_COPY_PHYSICAL_LINEAR_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_COPY_PHYSICAL_LINEAR_HEADER_sub_op_shift = 8
SDMA_PKT_COPY_PHYSICAL_LINEAR_HEADER_SUB_OP = lambda x: (((x) & SDMA_PKT_COPY_PHYSICAL_LINEAR_HEADER_sub_op_mask) << SDMA_PKT_COPY_PHYSICAL_LINEAR_HEADER_sub_op_shift)
SDMA_PKT_COPY_PHYSICAL_LINEAR_HEADER_tmz_offset = 0
SDMA_PKT_COPY_PHYSICAL_LINEAR_HEADER_tmz_mask = 0x00000001
SDMA_PKT_COPY_PHYSICAL_LINEAR_HEADER_tmz_shift = 18
SDMA_PKT_COPY_PHYSICAL_LINEAR_HEADER_TMZ = lambda x: (((x) & SDMA_PKT_COPY_PHYSICAL_LINEAR_HEADER_tmz_mask) << SDMA_PKT_COPY_PHYSICAL_LINEAR_HEADER_tmz_shift)
SDMA_PKT_COPY_PHYSICAL_LINEAR_HEADER_cpv_offset = 0
SDMA_PKT_COPY_PHYSICAL_LINEAR_HEADER_cpv_mask = 0x00000001
SDMA_PKT_COPY_PHYSICAL_LINEAR_HEADER_cpv_shift = 19
SDMA_PKT_COPY_PHYSICAL_LINEAR_HEADER_CPV = lambda x: (((x) & SDMA_PKT_COPY_PHYSICAL_LINEAR_HEADER_cpv_mask) << SDMA_PKT_COPY_PHYSICAL_LINEAR_HEADER_cpv_shift)
SDMA_PKT_COPY_PHYSICAL_LINEAR_COUNT_count_offset = 1
SDMA_PKT_COPY_PHYSICAL_LINEAR_COUNT_count_mask = 0x003FFFFF
SDMA_PKT_COPY_PHYSICAL_LINEAR_COUNT_count_shift = 0
SDMA_PKT_COPY_PHYSICAL_LINEAR_COUNT_COUNT = lambda x: (((x) & SDMA_PKT_COPY_PHYSICAL_LINEAR_COUNT_count_mask) << SDMA_PKT_COPY_PHYSICAL_LINEAR_COUNT_count_shift)
SDMA_PKT_COPY_PHYSICAL_LINEAR_COUNT_addr_pair_num_offset = 1
SDMA_PKT_COPY_PHYSICAL_LINEAR_COUNT_addr_pair_num_mask = 0x000000FF
SDMA_PKT_COPY_PHYSICAL_LINEAR_COUNT_addr_pair_num_shift = 24
SDMA_PKT_COPY_PHYSICAL_LINEAR_COUNT_ADDR_PAIR_NUM = lambda x: (((x) & SDMA_PKT_COPY_PHYSICAL_LINEAR_COUNT_addr_pair_num_mask) << SDMA_PKT_COPY_PHYSICAL_LINEAR_COUNT_addr_pair_num_shift)
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_mtype_offset = 2
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_mtype_mask = 0x00000007
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_mtype_shift = 3
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_DST_MTYPE = lambda x: (((x) & SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_mtype_mask) << SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_mtype_shift)
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_l2_policy_offset = 2
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_l2_policy_mask = 0x00000003
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_l2_policy_shift = 6
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_DST_L2_POLICY = lambda x: (((x) & SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_l2_policy_mask) << SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_l2_policy_shift)
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_llc_offset = 2
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_llc_mask = 0x00000001
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_llc_shift = 8
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_DST_LLC = lambda x: (((x) & SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_llc_mask) << SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_llc_shift)
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_mtype_offset = 2
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_mtype_mask = 0x00000007
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_mtype_shift = 11
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_SRC_MTYPE = lambda x: (((x) & SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_mtype_mask) << SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_mtype_shift)
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_l2_policy_offset = 2
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_l2_policy_mask = 0x00000003
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_l2_policy_shift = 14
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_SRC_L2_POLICY = lambda x: (((x) & SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_l2_policy_mask) << SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_l2_policy_shift)
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_llc_offset = 2
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_llc_mask = 0x00000001
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_llc_shift = 16
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_SRC_LLC = lambda x: (((x) & SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_llc_mask) << SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_llc_shift)
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_sw_offset = 2
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_sw_mask = 0x00000003
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_sw_shift = 17
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_DST_SW = lambda x: (((x) & SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_sw_mask) << SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_sw_shift)
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_gcc_offset = 2
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_gcc_mask = 0x00000001
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_gcc_shift = 19
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_DST_GCC = lambda x: (((x) & SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_gcc_mask) << SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_gcc_shift)
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_sys_offset = 2
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_sys_mask = 0x00000001
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_sys_shift = 20
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_DST_SYS = lambda x: (((x) & SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_sys_mask) << SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_sys_shift)
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_log_offset = 2
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_log_mask = 0x00000001
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_log_shift = 21
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_DST_LOG = lambda x: (((x) & SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_log_mask) << SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_log_shift)
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_snoop_offset = 2
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_snoop_mask = 0x00000001
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_snoop_shift = 22
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_DST_SNOOP = lambda x: (((x) & SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_snoop_mask) << SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_snoop_shift)
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_gpa_offset = 2
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_gpa_mask = 0x00000001
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_gpa_shift = 23
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_DST_GPA = lambda x: (((x) & SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_gpa_mask) << SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_gpa_shift)
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_sw_offset = 2
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_sw_mask = 0x00000003
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_sw_shift = 24
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_SRC_SW = lambda x: (((x) & SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_sw_mask) << SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_sw_shift)
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_gcc_offset = 2
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_gcc_mask = 0x00000001
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_gcc_shift = 27
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_SRC_GCC = lambda x: (((x) & SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_gcc_mask) << SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_gcc_shift)
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_sys_offset = 2
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_sys_mask = 0x00000001
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_sys_shift = 28
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_SRC_SYS = lambda x: (((x) & SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_sys_mask) << SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_sys_shift)
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_snoop_offset = 2
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_snoop_mask = 0x00000001
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_snoop_shift = 30
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_SRC_SNOOP = lambda x: (((x) & SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_snoop_mask) << SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_snoop_shift)
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_gpa_offset = 2
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_gpa_mask = 0x00000001
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_gpa_shift = 31
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_SRC_GPA = lambda x: (((x) & SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_gpa_mask) << SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_gpa_shift)
SDMA_PKT_COPY_PHYSICAL_LINEAR_SRC_ADDR_LO_src_addr_31_0_offset = 3
SDMA_PKT_COPY_PHYSICAL_LINEAR_SRC_ADDR_LO_src_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_PHYSICAL_LINEAR_SRC_ADDR_LO_src_addr_31_0_shift = 0
SDMA_PKT_COPY_PHYSICAL_LINEAR_SRC_ADDR_LO_SRC_ADDR_31_0 = lambda x: (((x) & SDMA_PKT_COPY_PHYSICAL_LINEAR_SRC_ADDR_LO_src_addr_31_0_mask) << SDMA_PKT_COPY_PHYSICAL_LINEAR_SRC_ADDR_LO_src_addr_31_0_shift)
SDMA_PKT_COPY_PHYSICAL_LINEAR_SRC_ADDR_HI_src_addr_63_32_offset = 4
SDMA_PKT_COPY_PHYSICAL_LINEAR_SRC_ADDR_HI_src_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_PHYSICAL_LINEAR_SRC_ADDR_HI_src_addr_63_32_shift = 0
SDMA_PKT_COPY_PHYSICAL_LINEAR_SRC_ADDR_HI_SRC_ADDR_63_32 = lambda x: (((x) & SDMA_PKT_COPY_PHYSICAL_LINEAR_SRC_ADDR_HI_src_addr_63_32_mask) << SDMA_PKT_COPY_PHYSICAL_LINEAR_SRC_ADDR_HI_src_addr_63_32_shift)
SDMA_PKT_COPY_PHYSICAL_LINEAR_DST_ADDR_LO_dst_addr_31_0_offset = 5
SDMA_PKT_COPY_PHYSICAL_LINEAR_DST_ADDR_LO_dst_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_PHYSICAL_LINEAR_DST_ADDR_LO_dst_addr_31_0_shift = 0
SDMA_PKT_COPY_PHYSICAL_LINEAR_DST_ADDR_LO_DST_ADDR_31_0 = lambda x: (((x) & SDMA_PKT_COPY_PHYSICAL_LINEAR_DST_ADDR_LO_dst_addr_31_0_mask) << SDMA_PKT_COPY_PHYSICAL_LINEAR_DST_ADDR_LO_dst_addr_31_0_shift)
SDMA_PKT_COPY_PHYSICAL_LINEAR_DST_ADDR_HI_dst_addr_63_32_offset = 6
SDMA_PKT_COPY_PHYSICAL_LINEAR_DST_ADDR_HI_dst_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_PHYSICAL_LINEAR_DST_ADDR_HI_dst_addr_63_32_shift = 0
SDMA_PKT_COPY_PHYSICAL_LINEAR_DST_ADDR_HI_DST_ADDR_63_32 = lambda x: (((x) & SDMA_PKT_COPY_PHYSICAL_LINEAR_DST_ADDR_HI_dst_addr_63_32_mask) << SDMA_PKT_COPY_PHYSICAL_LINEAR_DST_ADDR_HI_dst_addr_63_32_shift)
SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_op_offset = 0
SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_op_mask = 0x000000FF
SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_op_shift = 0
SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_OP = lambda x: (((x) & SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_op_mask) << SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_op_shift)
SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_sub_op_offset = 0
SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_sub_op_shift = 8
SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_SUB_OP = lambda x: (((x) & SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_sub_op_mask) << SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_sub_op_shift)
SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_encrypt_offset = 0
SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_encrypt_mask = 0x00000001
SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_encrypt_shift = 16
SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_ENCRYPT = lambda x: (((x) & SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_encrypt_mask) << SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_encrypt_shift)
SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_tmz_offset = 0
SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_tmz_mask = 0x00000001
SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_tmz_shift = 18
SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_TMZ = lambda x: (((x) & SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_tmz_mask) << SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_tmz_shift)
SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_cpv_offset = 0
SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_cpv_mask = 0x00000001
SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_cpv_shift = 19
SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_CPV = lambda x: (((x) & SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_cpv_mask) << SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_cpv_shift)
SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_broadcast_offset = 0
SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_broadcast_mask = 0x00000001
SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_broadcast_shift = 27
SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_BROADCAST = lambda x: (((x) & SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_broadcast_mask) << SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_broadcast_shift)
SDMA_PKT_COPY_BROADCAST_LINEAR_COUNT_count_offset = 1
SDMA_PKT_COPY_BROADCAST_LINEAR_COUNT_count_mask = 0x3FFFFFFF
SDMA_PKT_COPY_BROADCAST_LINEAR_COUNT_count_shift = 0
SDMA_PKT_COPY_BROADCAST_LINEAR_COUNT_COUNT = lambda x: (((x) & SDMA_PKT_COPY_BROADCAST_LINEAR_COUNT_count_mask) << SDMA_PKT_COPY_BROADCAST_LINEAR_COUNT_count_shift)
SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_dst2_sw_offset = 2
SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_dst2_sw_mask = 0x00000003
SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_dst2_sw_shift = 8
SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_DST2_SW = lambda x: (((x) & SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_dst2_sw_mask) << SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_dst2_sw_shift)
SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_dst2_cache_policy_offset = 2
SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_dst2_cache_policy_mask = 0x00000007
SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_dst2_cache_policy_shift = 10
SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_DST2_CACHE_POLICY = lambda x: (((x) & SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_dst2_cache_policy_mask) << SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_dst2_cache_policy_shift)
SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_dst1_sw_offset = 2
SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_dst1_sw_mask = 0x00000003
SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_dst1_sw_shift = 16
SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_DST1_SW = lambda x: (((x) & SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_dst1_sw_mask) << SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_dst1_sw_shift)
SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_dst1_cache_policy_offset = 2
SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_dst1_cache_policy_mask = 0x00000007
SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_dst1_cache_policy_shift = 18
SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_DST1_CACHE_POLICY = lambda x: (((x) & SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_dst1_cache_policy_mask) << SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_dst1_cache_policy_shift)
SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_src_sw_offset = 2
SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_src_sw_mask = 0x00000003
SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_src_sw_shift = 24
SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_SRC_SW = lambda x: (((x) & SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_src_sw_mask) << SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_src_sw_shift)
SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_src_cache_policy_offset = 2
SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_src_cache_policy_mask = 0x00000007
SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_src_cache_policy_shift = 26
SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_SRC_CACHE_POLICY = lambda x: (((x) & SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_src_cache_policy_mask) << SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_src_cache_policy_shift)
SDMA_PKT_COPY_BROADCAST_LINEAR_SRC_ADDR_LO_src_addr_31_0_offset = 3
SDMA_PKT_COPY_BROADCAST_LINEAR_SRC_ADDR_LO_src_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_BROADCAST_LINEAR_SRC_ADDR_LO_src_addr_31_0_shift = 0
SDMA_PKT_COPY_BROADCAST_LINEAR_SRC_ADDR_LO_SRC_ADDR_31_0 = lambda x: (((x) & SDMA_PKT_COPY_BROADCAST_LINEAR_SRC_ADDR_LO_src_addr_31_0_mask) << SDMA_PKT_COPY_BROADCAST_LINEAR_SRC_ADDR_LO_src_addr_31_0_shift)
SDMA_PKT_COPY_BROADCAST_LINEAR_SRC_ADDR_HI_src_addr_63_32_offset = 4
SDMA_PKT_COPY_BROADCAST_LINEAR_SRC_ADDR_HI_src_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_BROADCAST_LINEAR_SRC_ADDR_HI_src_addr_63_32_shift = 0
SDMA_PKT_COPY_BROADCAST_LINEAR_SRC_ADDR_HI_SRC_ADDR_63_32 = lambda x: (((x) & SDMA_PKT_COPY_BROADCAST_LINEAR_SRC_ADDR_HI_src_addr_63_32_mask) << SDMA_PKT_COPY_BROADCAST_LINEAR_SRC_ADDR_HI_src_addr_63_32_shift)
SDMA_PKT_COPY_BROADCAST_LINEAR_DST1_ADDR_LO_dst1_addr_31_0_offset = 5
SDMA_PKT_COPY_BROADCAST_LINEAR_DST1_ADDR_LO_dst1_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_BROADCAST_LINEAR_DST1_ADDR_LO_dst1_addr_31_0_shift = 0
SDMA_PKT_COPY_BROADCAST_LINEAR_DST1_ADDR_LO_DST1_ADDR_31_0 = lambda x: (((x) & SDMA_PKT_COPY_BROADCAST_LINEAR_DST1_ADDR_LO_dst1_addr_31_0_mask) << SDMA_PKT_COPY_BROADCAST_LINEAR_DST1_ADDR_LO_dst1_addr_31_0_shift)
SDMA_PKT_COPY_BROADCAST_LINEAR_DST1_ADDR_HI_dst1_addr_63_32_offset = 6
SDMA_PKT_COPY_BROADCAST_LINEAR_DST1_ADDR_HI_dst1_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_BROADCAST_LINEAR_DST1_ADDR_HI_dst1_addr_63_32_shift = 0
SDMA_PKT_COPY_BROADCAST_LINEAR_DST1_ADDR_HI_DST1_ADDR_63_32 = lambda x: (((x) & SDMA_PKT_COPY_BROADCAST_LINEAR_DST1_ADDR_HI_dst1_addr_63_32_mask) << SDMA_PKT_COPY_BROADCAST_LINEAR_DST1_ADDR_HI_dst1_addr_63_32_shift)
SDMA_PKT_COPY_BROADCAST_LINEAR_DST2_ADDR_LO_dst2_addr_31_0_offset = 7
SDMA_PKT_COPY_BROADCAST_LINEAR_DST2_ADDR_LO_dst2_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_BROADCAST_LINEAR_DST2_ADDR_LO_dst2_addr_31_0_shift = 0
SDMA_PKT_COPY_BROADCAST_LINEAR_DST2_ADDR_LO_DST2_ADDR_31_0 = lambda x: (((x) & SDMA_PKT_COPY_BROADCAST_LINEAR_DST2_ADDR_LO_dst2_addr_31_0_mask) << SDMA_PKT_COPY_BROADCAST_LINEAR_DST2_ADDR_LO_dst2_addr_31_0_shift)
SDMA_PKT_COPY_BROADCAST_LINEAR_DST2_ADDR_HI_dst2_addr_63_32_offset = 8
SDMA_PKT_COPY_BROADCAST_LINEAR_DST2_ADDR_HI_dst2_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_BROADCAST_LINEAR_DST2_ADDR_HI_dst2_addr_63_32_shift = 0
SDMA_PKT_COPY_BROADCAST_LINEAR_DST2_ADDR_HI_DST2_ADDR_63_32 = lambda x: (((x) & SDMA_PKT_COPY_BROADCAST_LINEAR_DST2_ADDR_HI_dst2_addr_63_32_mask) << SDMA_PKT_COPY_BROADCAST_LINEAR_DST2_ADDR_HI_dst2_addr_63_32_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_op_offset = 0
SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_op_mask = 0x000000FF
SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_op_shift = 0
SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_OP = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_op_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_op_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_sub_op_offset = 0
SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_sub_op_shift = 8
SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_SUB_OP = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_sub_op_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_sub_op_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_tmz_offset = 0
SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_tmz_mask = 0x00000001
SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_tmz_shift = 18
SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_TMZ = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_tmz_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_tmz_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_cpv_offset = 0
SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_cpv_mask = 0x00000001
SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_cpv_shift = 19
SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_CPV = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_cpv_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_cpv_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_elementsize_offset = 0
SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_elementsize_mask = 0x00000007
SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_elementsize_shift = 29
SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_ELEMENTSIZE = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_elementsize_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_elementsize_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_SRC_ADDR_LO_src_addr_31_0_offset = 1
SDMA_PKT_COPY_LINEAR_SUBWIN_SRC_ADDR_LO_src_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_LINEAR_SUBWIN_SRC_ADDR_LO_src_addr_31_0_shift = 0
SDMA_PKT_COPY_LINEAR_SUBWIN_SRC_ADDR_LO_SRC_ADDR_31_0 = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_SRC_ADDR_LO_src_addr_31_0_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_SRC_ADDR_LO_src_addr_31_0_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_SRC_ADDR_HI_src_addr_63_32_offset = 2
SDMA_PKT_COPY_LINEAR_SUBWIN_SRC_ADDR_HI_src_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_LINEAR_SUBWIN_SRC_ADDR_HI_src_addr_63_32_shift = 0
SDMA_PKT_COPY_LINEAR_SUBWIN_SRC_ADDR_HI_SRC_ADDR_63_32 = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_SRC_ADDR_HI_src_addr_63_32_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_SRC_ADDR_HI_src_addr_63_32_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_3_src_x_offset = 3
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_3_src_x_mask = 0x00003FFF
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_3_src_x_shift = 0
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_3_SRC_X = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_DW_3_src_x_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_DW_3_src_x_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_3_src_y_offset = 3
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_3_src_y_mask = 0x00003FFF
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_3_src_y_shift = 16
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_3_SRC_Y = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_DW_3_src_y_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_DW_3_src_y_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_4_src_z_offset = 4
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_4_src_z_mask = 0x00001FFF
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_4_src_z_shift = 0
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_4_SRC_Z = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_DW_4_src_z_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_DW_4_src_z_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_4_src_pitch_offset = 4
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_4_src_pitch_mask = 0x0007FFFF
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_4_src_pitch_shift = 13
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_4_SRC_PITCH = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_DW_4_src_pitch_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_DW_4_src_pitch_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_5_src_slice_pitch_offset = 5
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_5_src_slice_pitch_mask = 0x0FFFFFFF
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_5_src_slice_pitch_shift = 0
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_5_SRC_SLICE_PITCH = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_DW_5_src_slice_pitch_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_DW_5_src_slice_pitch_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_DST_ADDR_LO_dst_addr_31_0_offset = 6
SDMA_PKT_COPY_LINEAR_SUBWIN_DST_ADDR_LO_dst_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_LINEAR_SUBWIN_DST_ADDR_LO_dst_addr_31_0_shift = 0
SDMA_PKT_COPY_LINEAR_SUBWIN_DST_ADDR_LO_DST_ADDR_31_0 = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_DST_ADDR_LO_dst_addr_31_0_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_DST_ADDR_LO_dst_addr_31_0_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_DST_ADDR_HI_dst_addr_63_32_offset = 7
SDMA_PKT_COPY_LINEAR_SUBWIN_DST_ADDR_HI_dst_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_LINEAR_SUBWIN_DST_ADDR_HI_dst_addr_63_32_shift = 0
SDMA_PKT_COPY_LINEAR_SUBWIN_DST_ADDR_HI_DST_ADDR_63_32 = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_DST_ADDR_HI_dst_addr_63_32_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_DST_ADDR_HI_dst_addr_63_32_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_8_dst_x_offset = 8
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_8_dst_x_mask = 0x00003FFF
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_8_dst_x_shift = 0
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_8_DST_X = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_DW_8_dst_x_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_DW_8_dst_x_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_8_dst_y_offset = 8
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_8_dst_y_mask = 0x00003FFF
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_8_dst_y_shift = 16
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_8_DST_Y = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_DW_8_dst_y_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_DW_8_dst_y_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_9_dst_z_offset = 9
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_9_dst_z_mask = 0x00001FFF
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_9_dst_z_shift = 0
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_9_DST_Z = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_DW_9_dst_z_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_DW_9_dst_z_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_9_dst_pitch_offset = 9
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_9_dst_pitch_mask = 0x0007FFFF
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_9_dst_pitch_shift = 13
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_9_DST_PITCH = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_DW_9_dst_pitch_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_DW_9_dst_pitch_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_10_dst_slice_pitch_offset = 10
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_10_dst_slice_pitch_mask = 0x0FFFFFFF
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_10_dst_slice_pitch_shift = 0
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_10_DST_SLICE_PITCH = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_DW_10_dst_slice_pitch_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_DW_10_dst_slice_pitch_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_11_rect_x_offset = 11
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_11_rect_x_mask = 0x00003FFF
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_11_rect_x_shift = 0
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_11_RECT_X = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_DW_11_rect_x_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_DW_11_rect_x_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_11_rect_y_offset = 11
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_11_rect_y_mask = 0x00003FFF
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_11_rect_y_shift = 16
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_11_RECT_Y = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_DW_11_rect_y_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_DW_11_rect_y_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_rect_z_offset = 12
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_rect_z_mask = 0x00001FFF
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_rect_z_shift = 0
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_RECT_Z = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_rect_z_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_rect_z_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_dst_sw_offset = 12
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_dst_sw_mask = 0x00000003
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_dst_sw_shift = 16
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_DST_SW = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_dst_sw_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_dst_sw_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_dst_cache_policy_offset = 12
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_dst_cache_policy_mask = 0x00000007
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_dst_cache_policy_shift = 18
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_DST_CACHE_POLICY = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_dst_cache_policy_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_dst_cache_policy_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_src_sw_offset = 12
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_src_sw_mask = 0x00000003
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_src_sw_shift = 24
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_SRC_SW = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_src_sw_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_src_sw_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_src_cache_policy_offset = 12
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_src_cache_policy_mask = 0x00000007
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_src_cache_policy_shift = 26
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_SRC_CACHE_POLICY = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_src_cache_policy_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_src_cache_policy_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_HEADER_op_offset = 0
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_HEADER_op_mask = 0x000000FF
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_HEADER_op_shift = 0
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_HEADER_OP = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_HEADER_op_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_HEADER_op_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_HEADER_sub_op_offset = 0
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_HEADER_sub_op_shift = 8
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_HEADER_SUB_OP = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_HEADER_sub_op_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_HEADER_sub_op_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_HEADER_tmz_offset = 0
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_HEADER_tmz_mask = 0x00000001
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_HEADER_tmz_shift = 18
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_HEADER_TMZ = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_HEADER_tmz_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_HEADER_tmz_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_HEADER_cpv_offset = 0
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_HEADER_cpv_mask = 0x00000001
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_HEADER_cpv_shift = 19
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_HEADER_CPV = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_HEADER_cpv_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_HEADER_cpv_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_SRC_ADDR_LO_src_addr_31_0_offset = 1
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_SRC_ADDR_LO_src_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_SRC_ADDR_LO_src_addr_31_0_shift = 0
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_SRC_ADDR_LO_SRC_ADDR_31_0 = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_SRC_ADDR_LO_src_addr_31_0_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_SRC_ADDR_LO_src_addr_31_0_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_SRC_ADDR_HI_src_addr_63_32_offset = 2
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_SRC_ADDR_HI_src_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_SRC_ADDR_HI_src_addr_63_32_shift = 0
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_SRC_ADDR_HI_SRC_ADDR_63_32 = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_SRC_ADDR_HI_src_addr_63_32_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_SRC_ADDR_HI_src_addr_63_32_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_3_src_x_offset = 3
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_3_src_x_mask = 0xFFFFFFFF
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_3_src_x_shift = 0
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_3_SRC_X = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_3_src_x_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_3_src_x_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_4_src_y_offset = 4
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_4_src_y_mask = 0xFFFFFFFF
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_4_src_y_shift = 0
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_4_SRC_Y = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_4_src_y_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_4_src_y_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_5_src_z_offset = 5
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_5_src_z_mask = 0xFFFFFFFF
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_5_src_z_shift = 0
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_5_SRC_Z = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_5_src_z_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_5_src_z_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_6_src_pitch_offset = 6
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_6_src_pitch_mask = 0xFFFFFFFF
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_6_src_pitch_shift = 0
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_6_SRC_PITCH = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_6_src_pitch_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_6_src_pitch_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_7_src_slice_pitch_31_0_offset = 7
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_7_src_slice_pitch_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_7_src_slice_pitch_31_0_shift = 0
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_7_SRC_SLICE_PITCH_31_0 = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_7_src_slice_pitch_31_0_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_7_src_slice_pitch_31_0_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_8_src_slice_pitch_47_32_offset = 8
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_8_src_slice_pitch_47_32_mask = 0x0000FFFF
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_8_src_slice_pitch_47_32_shift = 0
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_8_SRC_SLICE_PITCH_47_32 = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_8_src_slice_pitch_47_32_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_8_src_slice_pitch_47_32_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DST_ADDR_LO_dst_addr_31_0_offset = 9
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DST_ADDR_LO_dst_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DST_ADDR_LO_dst_addr_31_0_shift = 0
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DST_ADDR_LO_DST_ADDR_31_0 = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DST_ADDR_LO_dst_addr_31_0_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DST_ADDR_LO_dst_addr_31_0_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DST_ADDR_HI_dst_addr_63_32_offset = 10
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DST_ADDR_HI_dst_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DST_ADDR_HI_dst_addr_63_32_shift = 0
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DST_ADDR_HI_DST_ADDR_63_32 = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DST_ADDR_HI_dst_addr_63_32_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DST_ADDR_HI_dst_addr_63_32_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_11_dst_x_offset = 11
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_11_dst_x_mask = 0xFFFFFFFF
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_11_dst_x_shift = 0
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_11_DST_X = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_11_dst_x_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_11_dst_x_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_12_dst_y_offset = 12
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_12_dst_y_mask = 0xFFFFFFFF
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_12_dst_y_shift = 0
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_12_DST_Y = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_12_dst_y_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_12_dst_y_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_13_dst_z_offset = 13
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_13_dst_z_mask = 0xFFFFFFFF
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_13_dst_z_shift = 0
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_13_DST_Z = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_13_dst_z_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_13_dst_z_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_14_dst_pitch_offset = 14
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_14_dst_pitch_mask = 0xFFFFFFFF
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_14_dst_pitch_shift = 0
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_14_DST_PITCH = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_14_dst_pitch_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_14_dst_pitch_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_15_dst_slice_pitch_31_0_offset = 15
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_15_dst_slice_pitch_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_15_dst_slice_pitch_31_0_shift = 0
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_15_DST_SLICE_PITCH_31_0 = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_15_dst_slice_pitch_31_0_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_15_dst_slice_pitch_31_0_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_16_dst_slice_pitch_47_32_offset = 16
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_16_dst_slice_pitch_47_32_mask = 0x0000FFFF
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_16_dst_slice_pitch_47_32_shift = 0
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_16_DST_SLICE_PITCH_47_32 = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_16_dst_slice_pitch_47_32_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_16_dst_slice_pitch_47_32_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_16_dst_sw_offset = 16
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_16_dst_sw_mask = 0x00000003
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_16_dst_sw_shift = 16
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_16_DST_SW = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_16_dst_sw_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_16_dst_sw_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_16_dst_policy_offset = 16
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_16_dst_policy_mask = 0x00000007
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_16_dst_policy_shift = 18
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_16_DST_POLICY = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_16_dst_policy_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_16_dst_policy_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_16_src_sw_offset = 16
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_16_src_sw_mask = 0x00000003
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_16_src_sw_shift = 24
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_16_SRC_SW = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_16_src_sw_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_16_src_sw_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_16_src_policy_offset = 16
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_16_src_policy_mask = 0x00000007
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_16_src_policy_shift = 26
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_16_SRC_POLICY = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_16_src_policy_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_16_src_policy_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_17_rect_x_offset = 17
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_17_rect_x_mask = 0xFFFFFFFF
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_17_rect_x_shift = 0
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_17_RECT_X = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_17_rect_x_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_17_rect_x_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_18_rect_y_offset = 18
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_18_rect_y_mask = 0xFFFFFFFF
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_18_rect_y_shift = 0
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_18_RECT_Y = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_18_rect_y_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_18_rect_y_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_19_rect_z_offset = 19
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_19_rect_z_mask = 0xFFFFFFFF
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_19_rect_z_shift = 0
SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_19_RECT_Z = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_19_rect_z_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_LARGE_DW_19_rect_z_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_HEADER_op_offset = 0
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_HEADER_op_mask = 0x000000FF
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_HEADER_op_shift = 0
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_HEADER_OP = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_BC_HEADER_op_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_BC_HEADER_op_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_HEADER_sub_op_offset = 0
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_HEADER_sub_op_shift = 8
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_HEADER_SUB_OP = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_BC_HEADER_sub_op_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_BC_HEADER_sub_op_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_HEADER_elementsize_offset = 0
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_HEADER_elementsize_mask = 0x00000007
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_HEADER_elementsize_shift = 29
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_HEADER_ELEMENTSIZE = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_BC_HEADER_elementsize_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_BC_HEADER_elementsize_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_SRC_ADDR_LO_src_addr_31_0_offset = 1
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_SRC_ADDR_LO_src_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_SRC_ADDR_LO_src_addr_31_0_shift = 0
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_SRC_ADDR_LO_SRC_ADDR_31_0 = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_BC_SRC_ADDR_LO_src_addr_31_0_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_BC_SRC_ADDR_LO_src_addr_31_0_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_SRC_ADDR_HI_src_addr_63_32_offset = 2
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_SRC_ADDR_HI_src_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_SRC_ADDR_HI_src_addr_63_32_shift = 0
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_SRC_ADDR_HI_SRC_ADDR_63_32 = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_BC_SRC_ADDR_HI_src_addr_63_32_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_BC_SRC_ADDR_HI_src_addr_63_32_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_3_src_x_offset = 3
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_3_src_x_mask = 0x00003FFF
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_3_src_x_shift = 0
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_3_SRC_X = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_3_src_x_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_3_src_x_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_3_src_y_offset = 3
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_3_src_y_mask = 0x00003FFF
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_3_src_y_shift = 16
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_3_SRC_Y = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_3_src_y_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_3_src_y_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_4_src_z_offset = 4
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_4_src_z_mask = 0x000007FF
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_4_src_z_shift = 0
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_4_SRC_Z = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_4_src_z_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_4_src_z_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_4_src_pitch_offset = 4
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_4_src_pitch_mask = 0x00003FFF
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_4_src_pitch_shift = 13
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_4_SRC_PITCH = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_4_src_pitch_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_4_src_pitch_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_5_src_slice_pitch_offset = 5
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_5_src_slice_pitch_mask = 0x0FFFFFFF
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_5_src_slice_pitch_shift = 0
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_5_SRC_SLICE_PITCH = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_5_src_slice_pitch_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_5_src_slice_pitch_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DST_ADDR_LO_dst_addr_31_0_offset = 6
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DST_ADDR_LO_dst_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DST_ADDR_LO_dst_addr_31_0_shift = 0
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DST_ADDR_LO_DST_ADDR_31_0 = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DST_ADDR_LO_dst_addr_31_0_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DST_ADDR_LO_dst_addr_31_0_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DST_ADDR_HI_dst_addr_63_32_offset = 7
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DST_ADDR_HI_dst_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DST_ADDR_HI_dst_addr_63_32_shift = 0
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DST_ADDR_HI_DST_ADDR_63_32 = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DST_ADDR_HI_dst_addr_63_32_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DST_ADDR_HI_dst_addr_63_32_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_8_dst_x_offset = 8
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_8_dst_x_mask = 0x00003FFF
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_8_dst_x_shift = 0
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_8_DST_X = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_8_dst_x_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_8_dst_x_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_8_dst_y_offset = 8
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_8_dst_y_mask = 0x00003FFF
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_8_dst_y_shift = 16
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_8_DST_Y = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_8_dst_y_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_8_dst_y_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_9_dst_z_offset = 9
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_9_dst_z_mask = 0x000007FF
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_9_dst_z_shift = 0
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_9_DST_Z = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_9_dst_z_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_9_dst_z_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_9_dst_pitch_offset = 9
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_9_dst_pitch_mask = 0x00003FFF
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_9_dst_pitch_shift = 13
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_9_DST_PITCH = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_9_dst_pitch_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_9_dst_pitch_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_10_dst_slice_pitch_offset = 10
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_10_dst_slice_pitch_mask = 0x0FFFFFFF
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_10_dst_slice_pitch_shift = 0
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_10_DST_SLICE_PITCH = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_10_dst_slice_pitch_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_10_dst_slice_pitch_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_11_rect_x_offset = 11
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_11_rect_x_mask = 0x00003FFF
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_11_rect_x_shift = 0
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_11_RECT_X = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_11_rect_x_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_11_rect_x_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_11_rect_y_offset = 11
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_11_rect_y_mask = 0x00003FFF
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_11_rect_y_shift = 16
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_11_RECT_Y = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_11_rect_y_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_11_rect_y_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_rect_z_offset = 12
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_rect_z_mask = 0x000007FF
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_rect_z_shift = 0
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_RECT_Z = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_rect_z_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_rect_z_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_dst_sw_offset = 12
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_dst_sw_mask = 0x00000003
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_dst_sw_shift = 16
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_DST_SW = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_dst_sw_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_dst_sw_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_dst_ha_offset = 12
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_dst_ha_mask = 0x00000001
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_dst_ha_shift = 19
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_DST_HA = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_dst_ha_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_dst_ha_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_src_sw_offset = 12
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_src_sw_mask = 0x00000003
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_src_sw_shift = 24
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_SRC_SW = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_src_sw_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_src_sw_shift)
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_src_ha_offset = 12
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_src_ha_mask = 0x00000001
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_src_ha_shift = 27
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_SRC_HA = lambda x: (((x) & SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_src_ha_mask) << SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_src_ha_shift)
SDMA_PKT_COPY_TILED_HEADER_op_offset = 0
SDMA_PKT_COPY_TILED_HEADER_op_mask = 0x000000FF
SDMA_PKT_COPY_TILED_HEADER_op_shift = 0
SDMA_PKT_COPY_TILED_HEADER_OP = lambda x: (((x) & SDMA_PKT_COPY_TILED_HEADER_op_mask) << SDMA_PKT_COPY_TILED_HEADER_op_shift)
SDMA_PKT_COPY_TILED_HEADER_sub_op_offset = 0
SDMA_PKT_COPY_TILED_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_COPY_TILED_HEADER_sub_op_shift = 8
SDMA_PKT_COPY_TILED_HEADER_SUB_OP = lambda x: (((x) & SDMA_PKT_COPY_TILED_HEADER_sub_op_mask) << SDMA_PKT_COPY_TILED_HEADER_sub_op_shift)
SDMA_PKT_COPY_TILED_HEADER_encrypt_offset = 0
SDMA_PKT_COPY_TILED_HEADER_encrypt_mask = 0x00000001
SDMA_PKT_COPY_TILED_HEADER_encrypt_shift = 16
SDMA_PKT_COPY_TILED_HEADER_ENCRYPT = lambda x: (((x) & SDMA_PKT_COPY_TILED_HEADER_encrypt_mask) << SDMA_PKT_COPY_TILED_HEADER_encrypt_shift)
SDMA_PKT_COPY_TILED_HEADER_tmz_offset = 0
SDMA_PKT_COPY_TILED_HEADER_tmz_mask = 0x00000001
SDMA_PKT_COPY_TILED_HEADER_tmz_shift = 18
SDMA_PKT_COPY_TILED_HEADER_TMZ = lambda x: (((x) & SDMA_PKT_COPY_TILED_HEADER_tmz_mask) << SDMA_PKT_COPY_TILED_HEADER_tmz_shift)
SDMA_PKT_COPY_TILED_HEADER_cpv_offset = 0
SDMA_PKT_COPY_TILED_HEADER_cpv_mask = 0x00000001
SDMA_PKT_COPY_TILED_HEADER_cpv_shift = 19
SDMA_PKT_COPY_TILED_HEADER_CPV = lambda x: (((x) & SDMA_PKT_COPY_TILED_HEADER_cpv_mask) << SDMA_PKT_COPY_TILED_HEADER_cpv_shift)
SDMA_PKT_COPY_TILED_HEADER_detile_offset = 0
SDMA_PKT_COPY_TILED_HEADER_detile_mask = 0x00000001
SDMA_PKT_COPY_TILED_HEADER_detile_shift = 31
SDMA_PKT_COPY_TILED_HEADER_DETILE = lambda x: (((x) & SDMA_PKT_COPY_TILED_HEADER_detile_mask) << SDMA_PKT_COPY_TILED_HEADER_detile_shift)
SDMA_PKT_COPY_TILED_TILED_ADDR_LO_tiled_addr_31_0_offset = 1
SDMA_PKT_COPY_TILED_TILED_ADDR_LO_tiled_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_TILED_TILED_ADDR_LO_tiled_addr_31_0_shift = 0
SDMA_PKT_COPY_TILED_TILED_ADDR_LO_TILED_ADDR_31_0 = lambda x: (((x) & SDMA_PKT_COPY_TILED_TILED_ADDR_LO_tiled_addr_31_0_mask) << SDMA_PKT_COPY_TILED_TILED_ADDR_LO_tiled_addr_31_0_shift)
SDMA_PKT_COPY_TILED_TILED_ADDR_HI_tiled_addr_63_32_offset = 2
SDMA_PKT_COPY_TILED_TILED_ADDR_HI_tiled_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_TILED_TILED_ADDR_HI_tiled_addr_63_32_shift = 0
SDMA_PKT_COPY_TILED_TILED_ADDR_HI_TILED_ADDR_63_32 = lambda x: (((x) & SDMA_PKT_COPY_TILED_TILED_ADDR_HI_tiled_addr_63_32_mask) << SDMA_PKT_COPY_TILED_TILED_ADDR_HI_tiled_addr_63_32_shift)
SDMA_PKT_COPY_TILED_DW_3_width_offset = 3
SDMA_PKT_COPY_TILED_DW_3_width_mask = 0x00003FFF
SDMA_PKT_COPY_TILED_DW_3_width_shift = 0
SDMA_PKT_COPY_TILED_DW_3_WIDTH = lambda x: (((x) & SDMA_PKT_COPY_TILED_DW_3_width_mask) << SDMA_PKT_COPY_TILED_DW_3_width_shift)
SDMA_PKT_COPY_TILED_DW_4_height_offset = 4
SDMA_PKT_COPY_TILED_DW_4_height_mask = 0x00003FFF
SDMA_PKT_COPY_TILED_DW_4_height_shift = 0
SDMA_PKT_COPY_TILED_DW_4_HEIGHT = lambda x: (((x) & SDMA_PKT_COPY_TILED_DW_4_height_mask) << SDMA_PKT_COPY_TILED_DW_4_height_shift)
SDMA_PKT_COPY_TILED_DW_4_depth_offset = 4
SDMA_PKT_COPY_TILED_DW_4_depth_mask = 0x00001FFF
SDMA_PKT_COPY_TILED_DW_4_depth_shift = 16
SDMA_PKT_COPY_TILED_DW_4_DEPTH = lambda x: (((x) & SDMA_PKT_COPY_TILED_DW_4_depth_mask) << SDMA_PKT_COPY_TILED_DW_4_depth_shift)
SDMA_PKT_COPY_TILED_DW_5_element_size_offset = 5
SDMA_PKT_COPY_TILED_DW_5_element_size_mask = 0x00000007
SDMA_PKT_COPY_TILED_DW_5_element_size_shift = 0
SDMA_PKT_COPY_TILED_DW_5_ELEMENT_SIZE = lambda x: (((x) & SDMA_PKT_COPY_TILED_DW_5_element_size_mask) << SDMA_PKT_COPY_TILED_DW_5_element_size_shift)
SDMA_PKT_COPY_TILED_DW_5_swizzle_mode_offset = 5
SDMA_PKT_COPY_TILED_DW_5_swizzle_mode_mask = 0x0000001F
SDMA_PKT_COPY_TILED_DW_5_swizzle_mode_shift = 3
SDMA_PKT_COPY_TILED_DW_5_SWIZZLE_MODE = lambda x: (((x) & SDMA_PKT_COPY_TILED_DW_5_swizzle_mode_mask) << SDMA_PKT_COPY_TILED_DW_5_swizzle_mode_shift)
SDMA_PKT_COPY_TILED_DW_5_dimension_offset = 5
SDMA_PKT_COPY_TILED_DW_5_dimension_mask = 0x00000003
SDMA_PKT_COPY_TILED_DW_5_dimension_shift = 9
SDMA_PKT_COPY_TILED_DW_5_DIMENSION = lambda x: (((x) & SDMA_PKT_COPY_TILED_DW_5_dimension_mask) << SDMA_PKT_COPY_TILED_DW_5_dimension_shift)
SDMA_PKT_COPY_TILED_DW_5_mip_max_offset = 5
SDMA_PKT_COPY_TILED_DW_5_mip_max_mask = 0x0000000F
SDMA_PKT_COPY_TILED_DW_5_mip_max_shift = 16
SDMA_PKT_COPY_TILED_DW_5_MIP_MAX = lambda x: (((x) & SDMA_PKT_COPY_TILED_DW_5_mip_max_mask) << SDMA_PKT_COPY_TILED_DW_5_mip_max_shift)
SDMA_PKT_COPY_TILED_DW_6_x_offset = 6
SDMA_PKT_COPY_TILED_DW_6_x_mask = 0x00003FFF
SDMA_PKT_COPY_TILED_DW_6_x_shift = 0
SDMA_PKT_COPY_TILED_DW_6_X = lambda x: (((x) & SDMA_PKT_COPY_TILED_DW_6_x_mask) << SDMA_PKT_COPY_TILED_DW_6_x_shift)
SDMA_PKT_COPY_TILED_DW_6_y_offset = 6
SDMA_PKT_COPY_TILED_DW_6_y_mask = 0x00003FFF
SDMA_PKT_COPY_TILED_DW_6_y_shift = 16
SDMA_PKT_COPY_TILED_DW_6_Y = lambda x: (((x) & SDMA_PKT_COPY_TILED_DW_6_y_mask) << SDMA_PKT_COPY_TILED_DW_6_y_shift)
SDMA_PKT_COPY_TILED_DW_7_z_offset = 7
SDMA_PKT_COPY_TILED_DW_7_z_mask = 0x00001FFF
SDMA_PKT_COPY_TILED_DW_7_z_shift = 0
SDMA_PKT_COPY_TILED_DW_7_Z = lambda x: (((x) & SDMA_PKT_COPY_TILED_DW_7_z_mask) << SDMA_PKT_COPY_TILED_DW_7_z_shift)
SDMA_PKT_COPY_TILED_DW_7_linear_sw_offset = 7
SDMA_PKT_COPY_TILED_DW_7_linear_sw_mask = 0x00000003
SDMA_PKT_COPY_TILED_DW_7_linear_sw_shift = 16
SDMA_PKT_COPY_TILED_DW_7_LINEAR_SW = lambda x: (((x) & SDMA_PKT_COPY_TILED_DW_7_linear_sw_mask) << SDMA_PKT_COPY_TILED_DW_7_linear_sw_shift)
SDMA_PKT_COPY_TILED_DW_7_linear_cache_policy_offset = 7
SDMA_PKT_COPY_TILED_DW_7_linear_cache_policy_mask = 0x00000007
SDMA_PKT_COPY_TILED_DW_7_linear_cache_policy_shift = 18
SDMA_PKT_COPY_TILED_DW_7_LINEAR_CACHE_POLICY = lambda x: (((x) & SDMA_PKT_COPY_TILED_DW_7_linear_cache_policy_mask) << SDMA_PKT_COPY_TILED_DW_7_linear_cache_policy_shift)
SDMA_PKT_COPY_TILED_DW_7_tile_sw_offset = 7
SDMA_PKT_COPY_TILED_DW_7_tile_sw_mask = 0x00000003
SDMA_PKT_COPY_TILED_DW_7_tile_sw_shift = 24
SDMA_PKT_COPY_TILED_DW_7_TILE_SW = lambda x: (((x) & SDMA_PKT_COPY_TILED_DW_7_tile_sw_mask) << SDMA_PKT_COPY_TILED_DW_7_tile_sw_shift)
SDMA_PKT_COPY_TILED_DW_7_tile_cache_policy_offset = 7
SDMA_PKT_COPY_TILED_DW_7_tile_cache_policy_mask = 0x00000007
SDMA_PKT_COPY_TILED_DW_7_tile_cache_policy_shift = 26
SDMA_PKT_COPY_TILED_DW_7_TILE_CACHE_POLICY = lambda x: (((x) & SDMA_PKT_COPY_TILED_DW_7_tile_cache_policy_mask) << SDMA_PKT_COPY_TILED_DW_7_tile_cache_policy_shift)
SDMA_PKT_COPY_TILED_LINEAR_ADDR_LO_linear_addr_31_0_offset = 8
SDMA_PKT_COPY_TILED_LINEAR_ADDR_LO_linear_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_TILED_LINEAR_ADDR_LO_linear_addr_31_0_shift = 0
SDMA_PKT_COPY_TILED_LINEAR_ADDR_LO_LINEAR_ADDR_31_0 = lambda x: (((x) & SDMA_PKT_COPY_TILED_LINEAR_ADDR_LO_linear_addr_31_0_mask) << SDMA_PKT_COPY_TILED_LINEAR_ADDR_LO_linear_addr_31_0_shift)
SDMA_PKT_COPY_TILED_LINEAR_ADDR_HI_linear_addr_63_32_offset = 9
SDMA_PKT_COPY_TILED_LINEAR_ADDR_HI_linear_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_TILED_LINEAR_ADDR_HI_linear_addr_63_32_shift = 0
SDMA_PKT_COPY_TILED_LINEAR_ADDR_HI_LINEAR_ADDR_63_32 = lambda x: (((x) & SDMA_PKT_COPY_TILED_LINEAR_ADDR_HI_linear_addr_63_32_mask) << SDMA_PKT_COPY_TILED_LINEAR_ADDR_HI_linear_addr_63_32_shift)
SDMA_PKT_COPY_TILED_LINEAR_PITCH_linear_pitch_offset = 10
SDMA_PKT_COPY_TILED_LINEAR_PITCH_linear_pitch_mask = 0x0007FFFF
SDMA_PKT_COPY_TILED_LINEAR_PITCH_linear_pitch_shift = 0
SDMA_PKT_COPY_TILED_LINEAR_PITCH_LINEAR_PITCH = lambda x: (((x) & SDMA_PKT_COPY_TILED_LINEAR_PITCH_linear_pitch_mask) << SDMA_PKT_COPY_TILED_LINEAR_PITCH_linear_pitch_shift)
SDMA_PKT_COPY_TILED_LINEAR_SLICE_PITCH_linear_slice_pitch_offset = 11
SDMA_PKT_COPY_TILED_LINEAR_SLICE_PITCH_linear_slice_pitch_mask = 0xFFFFFFFF
SDMA_PKT_COPY_TILED_LINEAR_SLICE_PITCH_linear_slice_pitch_shift = 0
SDMA_PKT_COPY_TILED_LINEAR_SLICE_PITCH_LINEAR_SLICE_PITCH = lambda x: (((x) & SDMA_PKT_COPY_TILED_LINEAR_SLICE_PITCH_linear_slice_pitch_mask) << SDMA_PKT_COPY_TILED_LINEAR_SLICE_PITCH_linear_slice_pitch_shift)
SDMA_PKT_COPY_TILED_COUNT_count_offset = 12
SDMA_PKT_COPY_TILED_COUNT_count_mask = 0x3FFFFFFF
SDMA_PKT_COPY_TILED_COUNT_count_shift = 0
SDMA_PKT_COPY_TILED_COUNT_COUNT = lambda x: (((x) & SDMA_PKT_COPY_TILED_COUNT_count_mask) << SDMA_PKT_COPY_TILED_COUNT_count_shift)
SDMA_PKT_COPY_TILED_BC_HEADER_op_offset = 0
SDMA_PKT_COPY_TILED_BC_HEADER_op_mask = 0x000000FF
SDMA_PKT_COPY_TILED_BC_HEADER_op_shift = 0
SDMA_PKT_COPY_TILED_BC_HEADER_OP = lambda x: (((x) & SDMA_PKT_COPY_TILED_BC_HEADER_op_mask) << SDMA_PKT_COPY_TILED_BC_HEADER_op_shift)
SDMA_PKT_COPY_TILED_BC_HEADER_sub_op_offset = 0
SDMA_PKT_COPY_TILED_BC_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_COPY_TILED_BC_HEADER_sub_op_shift = 8
SDMA_PKT_COPY_TILED_BC_HEADER_SUB_OP = lambda x: (((x) & SDMA_PKT_COPY_TILED_BC_HEADER_sub_op_mask) << SDMA_PKT_COPY_TILED_BC_HEADER_sub_op_shift)
SDMA_PKT_COPY_TILED_BC_HEADER_detile_offset = 0
SDMA_PKT_COPY_TILED_BC_HEADER_detile_mask = 0x00000001
SDMA_PKT_COPY_TILED_BC_HEADER_detile_shift = 31
SDMA_PKT_COPY_TILED_BC_HEADER_DETILE = lambda x: (((x) & SDMA_PKT_COPY_TILED_BC_HEADER_detile_mask) << SDMA_PKT_COPY_TILED_BC_HEADER_detile_shift)
SDMA_PKT_COPY_TILED_BC_TILED_ADDR_LO_tiled_addr_31_0_offset = 1
SDMA_PKT_COPY_TILED_BC_TILED_ADDR_LO_tiled_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_TILED_BC_TILED_ADDR_LO_tiled_addr_31_0_shift = 0
SDMA_PKT_COPY_TILED_BC_TILED_ADDR_LO_TILED_ADDR_31_0 = lambda x: (((x) & SDMA_PKT_COPY_TILED_BC_TILED_ADDR_LO_tiled_addr_31_0_mask) << SDMA_PKT_COPY_TILED_BC_TILED_ADDR_LO_tiled_addr_31_0_shift)
SDMA_PKT_COPY_TILED_BC_TILED_ADDR_HI_tiled_addr_63_32_offset = 2
SDMA_PKT_COPY_TILED_BC_TILED_ADDR_HI_tiled_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_TILED_BC_TILED_ADDR_HI_tiled_addr_63_32_shift = 0
SDMA_PKT_COPY_TILED_BC_TILED_ADDR_HI_TILED_ADDR_63_32 = lambda x: (((x) & SDMA_PKT_COPY_TILED_BC_TILED_ADDR_HI_tiled_addr_63_32_mask) << SDMA_PKT_COPY_TILED_BC_TILED_ADDR_HI_tiled_addr_63_32_shift)
SDMA_PKT_COPY_TILED_BC_DW_3_width_offset = 3
SDMA_PKT_COPY_TILED_BC_DW_3_width_mask = 0x00003FFF
SDMA_PKT_COPY_TILED_BC_DW_3_width_shift = 0
SDMA_PKT_COPY_TILED_BC_DW_3_WIDTH = lambda x: (((x) & SDMA_PKT_COPY_TILED_BC_DW_3_width_mask) << SDMA_PKT_COPY_TILED_BC_DW_3_width_shift)
SDMA_PKT_COPY_TILED_BC_DW_4_height_offset = 4
SDMA_PKT_COPY_TILED_BC_DW_4_height_mask = 0x00003FFF
SDMA_PKT_COPY_TILED_BC_DW_4_height_shift = 0
SDMA_PKT_COPY_TILED_BC_DW_4_HEIGHT = lambda x: (((x) & SDMA_PKT_COPY_TILED_BC_DW_4_height_mask) << SDMA_PKT_COPY_TILED_BC_DW_4_height_shift)
SDMA_PKT_COPY_TILED_BC_DW_4_depth_offset = 4
SDMA_PKT_COPY_TILED_BC_DW_4_depth_mask = 0x000007FF
SDMA_PKT_COPY_TILED_BC_DW_4_depth_shift = 16
SDMA_PKT_COPY_TILED_BC_DW_4_DEPTH = lambda x: (((x) & SDMA_PKT_COPY_TILED_BC_DW_4_depth_mask) << SDMA_PKT_COPY_TILED_BC_DW_4_depth_shift)
SDMA_PKT_COPY_TILED_BC_DW_5_element_size_offset = 5
SDMA_PKT_COPY_TILED_BC_DW_5_element_size_mask = 0x00000007
SDMA_PKT_COPY_TILED_BC_DW_5_element_size_shift = 0
SDMA_PKT_COPY_TILED_BC_DW_5_ELEMENT_SIZE = lambda x: (((x) & SDMA_PKT_COPY_TILED_BC_DW_5_element_size_mask) << SDMA_PKT_COPY_TILED_BC_DW_5_element_size_shift)
SDMA_PKT_COPY_TILED_BC_DW_5_array_mode_offset = 5
SDMA_PKT_COPY_TILED_BC_DW_5_array_mode_mask = 0x0000000F
SDMA_PKT_COPY_TILED_BC_DW_5_array_mode_shift = 3
SDMA_PKT_COPY_TILED_BC_DW_5_ARRAY_MODE = lambda x: (((x) & SDMA_PKT_COPY_TILED_BC_DW_5_array_mode_mask) << SDMA_PKT_COPY_TILED_BC_DW_5_array_mode_shift)
SDMA_PKT_COPY_TILED_BC_DW_5_mit_mode_offset = 5
SDMA_PKT_COPY_TILED_BC_DW_5_mit_mode_mask = 0x00000007
SDMA_PKT_COPY_TILED_BC_DW_5_mit_mode_shift = 8
SDMA_PKT_COPY_TILED_BC_DW_5_MIT_MODE = lambda x: (((x) & SDMA_PKT_COPY_TILED_BC_DW_5_mit_mode_mask) << SDMA_PKT_COPY_TILED_BC_DW_5_mit_mode_shift)
SDMA_PKT_COPY_TILED_BC_DW_5_tilesplit_size_offset = 5
SDMA_PKT_COPY_TILED_BC_DW_5_tilesplit_size_mask = 0x00000007
SDMA_PKT_COPY_TILED_BC_DW_5_tilesplit_size_shift = 11
SDMA_PKT_COPY_TILED_BC_DW_5_TILESPLIT_SIZE = lambda x: (((x) & SDMA_PKT_COPY_TILED_BC_DW_5_tilesplit_size_mask) << SDMA_PKT_COPY_TILED_BC_DW_5_tilesplit_size_shift)
SDMA_PKT_COPY_TILED_BC_DW_5_bank_w_offset = 5
SDMA_PKT_COPY_TILED_BC_DW_5_bank_w_mask = 0x00000003
SDMA_PKT_COPY_TILED_BC_DW_5_bank_w_shift = 15
SDMA_PKT_COPY_TILED_BC_DW_5_BANK_W = lambda x: (((x) & SDMA_PKT_COPY_TILED_BC_DW_5_bank_w_mask) << SDMA_PKT_COPY_TILED_BC_DW_5_bank_w_shift)
SDMA_PKT_COPY_TILED_BC_DW_5_bank_h_offset = 5
SDMA_PKT_COPY_TILED_BC_DW_5_bank_h_mask = 0x00000003
SDMA_PKT_COPY_TILED_BC_DW_5_bank_h_shift = 18
SDMA_PKT_COPY_TILED_BC_DW_5_BANK_H = lambda x: (((x) & SDMA_PKT_COPY_TILED_BC_DW_5_bank_h_mask) << SDMA_PKT_COPY_TILED_BC_DW_5_bank_h_shift)
SDMA_PKT_COPY_TILED_BC_DW_5_num_bank_offset = 5
SDMA_PKT_COPY_TILED_BC_DW_5_num_bank_mask = 0x00000003
SDMA_PKT_COPY_TILED_BC_DW_5_num_bank_shift = 21
SDMA_PKT_COPY_TILED_BC_DW_5_NUM_BANK = lambda x: (((x) & SDMA_PKT_COPY_TILED_BC_DW_5_num_bank_mask) << SDMA_PKT_COPY_TILED_BC_DW_5_num_bank_shift)
SDMA_PKT_COPY_TILED_BC_DW_5_mat_aspt_offset = 5
SDMA_PKT_COPY_TILED_BC_DW_5_mat_aspt_mask = 0x00000003
SDMA_PKT_COPY_TILED_BC_DW_5_mat_aspt_shift = 24
SDMA_PKT_COPY_TILED_BC_DW_5_MAT_ASPT = lambda x: (((x) & SDMA_PKT_COPY_TILED_BC_DW_5_mat_aspt_mask) << SDMA_PKT_COPY_TILED_BC_DW_5_mat_aspt_shift)
SDMA_PKT_COPY_TILED_BC_DW_5_pipe_config_offset = 5
SDMA_PKT_COPY_TILED_BC_DW_5_pipe_config_mask = 0x0000001F
SDMA_PKT_COPY_TILED_BC_DW_5_pipe_config_shift = 26
SDMA_PKT_COPY_TILED_BC_DW_5_PIPE_CONFIG = lambda x: (((x) & SDMA_PKT_COPY_TILED_BC_DW_5_pipe_config_mask) << SDMA_PKT_COPY_TILED_BC_DW_5_pipe_config_shift)
SDMA_PKT_COPY_TILED_BC_DW_6_x_offset = 6
SDMA_PKT_COPY_TILED_BC_DW_6_x_mask = 0x00003FFF
SDMA_PKT_COPY_TILED_BC_DW_6_x_shift = 0
SDMA_PKT_COPY_TILED_BC_DW_6_X = lambda x: (((x) & SDMA_PKT_COPY_TILED_BC_DW_6_x_mask) << SDMA_PKT_COPY_TILED_BC_DW_6_x_shift)
SDMA_PKT_COPY_TILED_BC_DW_6_y_offset = 6
SDMA_PKT_COPY_TILED_BC_DW_6_y_mask = 0x00003FFF
SDMA_PKT_COPY_TILED_BC_DW_6_y_shift = 16
SDMA_PKT_COPY_TILED_BC_DW_6_Y = lambda x: (((x) & SDMA_PKT_COPY_TILED_BC_DW_6_y_mask) << SDMA_PKT_COPY_TILED_BC_DW_6_y_shift)
SDMA_PKT_COPY_TILED_BC_DW_7_z_offset = 7
SDMA_PKT_COPY_TILED_BC_DW_7_z_mask = 0x000007FF
SDMA_PKT_COPY_TILED_BC_DW_7_z_shift = 0
SDMA_PKT_COPY_TILED_BC_DW_7_Z = lambda x: (((x) & SDMA_PKT_COPY_TILED_BC_DW_7_z_mask) << SDMA_PKT_COPY_TILED_BC_DW_7_z_shift)
SDMA_PKT_COPY_TILED_BC_DW_7_linear_sw_offset = 7
SDMA_PKT_COPY_TILED_BC_DW_7_linear_sw_mask = 0x00000003
SDMA_PKT_COPY_TILED_BC_DW_7_linear_sw_shift = 16
SDMA_PKT_COPY_TILED_BC_DW_7_LINEAR_SW = lambda x: (((x) & SDMA_PKT_COPY_TILED_BC_DW_7_linear_sw_mask) << SDMA_PKT_COPY_TILED_BC_DW_7_linear_sw_shift)
SDMA_PKT_COPY_TILED_BC_DW_7_tile_sw_offset = 7
SDMA_PKT_COPY_TILED_BC_DW_7_tile_sw_mask = 0x00000003
SDMA_PKT_COPY_TILED_BC_DW_7_tile_sw_shift = 24
SDMA_PKT_COPY_TILED_BC_DW_7_TILE_SW = lambda x: (((x) & SDMA_PKT_COPY_TILED_BC_DW_7_tile_sw_mask) << SDMA_PKT_COPY_TILED_BC_DW_7_tile_sw_shift)
SDMA_PKT_COPY_TILED_BC_LINEAR_ADDR_LO_linear_addr_31_0_offset = 8
SDMA_PKT_COPY_TILED_BC_LINEAR_ADDR_LO_linear_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_TILED_BC_LINEAR_ADDR_LO_linear_addr_31_0_shift = 0
SDMA_PKT_COPY_TILED_BC_LINEAR_ADDR_LO_LINEAR_ADDR_31_0 = lambda x: (((x) & SDMA_PKT_COPY_TILED_BC_LINEAR_ADDR_LO_linear_addr_31_0_mask) << SDMA_PKT_COPY_TILED_BC_LINEAR_ADDR_LO_linear_addr_31_0_shift)
SDMA_PKT_COPY_TILED_BC_LINEAR_ADDR_HI_linear_addr_63_32_offset = 9
SDMA_PKT_COPY_TILED_BC_LINEAR_ADDR_HI_linear_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_TILED_BC_LINEAR_ADDR_HI_linear_addr_63_32_shift = 0
SDMA_PKT_COPY_TILED_BC_LINEAR_ADDR_HI_LINEAR_ADDR_63_32 = lambda x: (((x) & SDMA_PKT_COPY_TILED_BC_LINEAR_ADDR_HI_linear_addr_63_32_mask) << SDMA_PKT_COPY_TILED_BC_LINEAR_ADDR_HI_linear_addr_63_32_shift)
SDMA_PKT_COPY_TILED_BC_LINEAR_PITCH_linear_pitch_offset = 10
SDMA_PKT_COPY_TILED_BC_LINEAR_PITCH_linear_pitch_mask = 0x0007FFFF
SDMA_PKT_COPY_TILED_BC_LINEAR_PITCH_linear_pitch_shift = 0
SDMA_PKT_COPY_TILED_BC_LINEAR_PITCH_LINEAR_PITCH = lambda x: (((x) & SDMA_PKT_COPY_TILED_BC_LINEAR_PITCH_linear_pitch_mask) << SDMA_PKT_COPY_TILED_BC_LINEAR_PITCH_linear_pitch_shift)
SDMA_PKT_COPY_TILED_BC_LINEAR_SLICE_PITCH_linear_slice_pitch_offset = 11
SDMA_PKT_COPY_TILED_BC_LINEAR_SLICE_PITCH_linear_slice_pitch_mask = 0xFFFFFFFF
SDMA_PKT_COPY_TILED_BC_LINEAR_SLICE_PITCH_linear_slice_pitch_shift = 0
SDMA_PKT_COPY_TILED_BC_LINEAR_SLICE_PITCH_LINEAR_SLICE_PITCH = lambda x: (((x) & SDMA_PKT_COPY_TILED_BC_LINEAR_SLICE_PITCH_linear_slice_pitch_mask) << SDMA_PKT_COPY_TILED_BC_LINEAR_SLICE_PITCH_linear_slice_pitch_shift)
SDMA_PKT_COPY_TILED_BC_COUNT_count_offset = 12
SDMA_PKT_COPY_TILED_BC_COUNT_count_mask = 0x000FFFFF
SDMA_PKT_COPY_TILED_BC_COUNT_count_shift = 2
SDMA_PKT_COPY_TILED_BC_COUNT_COUNT = lambda x: (((x) & SDMA_PKT_COPY_TILED_BC_COUNT_count_mask) << SDMA_PKT_COPY_TILED_BC_COUNT_count_shift)
SDMA_PKT_COPY_L2T_BROADCAST_HEADER_op_offset = 0
SDMA_PKT_COPY_L2T_BROADCAST_HEADER_op_mask = 0x000000FF
SDMA_PKT_COPY_L2T_BROADCAST_HEADER_op_shift = 0
SDMA_PKT_COPY_L2T_BROADCAST_HEADER_OP = lambda x: (((x) & SDMA_PKT_COPY_L2T_BROADCAST_HEADER_op_mask) << SDMA_PKT_COPY_L2T_BROADCAST_HEADER_op_shift)
SDMA_PKT_COPY_L2T_BROADCAST_HEADER_sub_op_offset = 0
SDMA_PKT_COPY_L2T_BROADCAST_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_COPY_L2T_BROADCAST_HEADER_sub_op_shift = 8
SDMA_PKT_COPY_L2T_BROADCAST_HEADER_SUB_OP = lambda x: (((x) & SDMA_PKT_COPY_L2T_BROADCAST_HEADER_sub_op_mask) << SDMA_PKT_COPY_L2T_BROADCAST_HEADER_sub_op_shift)
SDMA_PKT_COPY_L2T_BROADCAST_HEADER_encrypt_offset = 0
SDMA_PKT_COPY_L2T_BROADCAST_HEADER_encrypt_mask = 0x00000001
SDMA_PKT_COPY_L2T_BROADCAST_HEADER_encrypt_shift = 16
SDMA_PKT_COPY_L2T_BROADCAST_HEADER_ENCRYPT = lambda x: (((x) & SDMA_PKT_COPY_L2T_BROADCAST_HEADER_encrypt_mask) << SDMA_PKT_COPY_L2T_BROADCAST_HEADER_encrypt_shift)
SDMA_PKT_COPY_L2T_BROADCAST_HEADER_tmz_offset = 0
SDMA_PKT_COPY_L2T_BROADCAST_HEADER_tmz_mask = 0x00000001
SDMA_PKT_COPY_L2T_BROADCAST_HEADER_tmz_shift = 18
SDMA_PKT_COPY_L2T_BROADCAST_HEADER_TMZ = lambda x: (((x) & SDMA_PKT_COPY_L2T_BROADCAST_HEADER_tmz_mask) << SDMA_PKT_COPY_L2T_BROADCAST_HEADER_tmz_shift)
SDMA_PKT_COPY_L2T_BROADCAST_HEADER_cpv_offset = 0
SDMA_PKT_COPY_L2T_BROADCAST_HEADER_cpv_mask = 0x00000001
SDMA_PKT_COPY_L2T_BROADCAST_HEADER_cpv_shift = 19
SDMA_PKT_COPY_L2T_BROADCAST_HEADER_CPV = lambda x: (((x) & SDMA_PKT_COPY_L2T_BROADCAST_HEADER_cpv_mask) << SDMA_PKT_COPY_L2T_BROADCAST_HEADER_cpv_shift)
SDMA_PKT_COPY_L2T_BROADCAST_HEADER_videocopy_offset = 0
SDMA_PKT_COPY_L2T_BROADCAST_HEADER_videocopy_mask = 0x00000001
SDMA_PKT_COPY_L2T_BROADCAST_HEADER_videocopy_shift = 26
SDMA_PKT_COPY_L2T_BROADCAST_HEADER_VIDEOCOPY = lambda x: (((x) & SDMA_PKT_COPY_L2T_BROADCAST_HEADER_videocopy_mask) << SDMA_PKT_COPY_L2T_BROADCAST_HEADER_videocopy_shift)
SDMA_PKT_COPY_L2T_BROADCAST_HEADER_broadcast_offset = 0
SDMA_PKT_COPY_L2T_BROADCAST_HEADER_broadcast_mask = 0x00000001
SDMA_PKT_COPY_L2T_BROADCAST_HEADER_broadcast_shift = 27
SDMA_PKT_COPY_L2T_BROADCAST_HEADER_BROADCAST = lambda x: (((x) & SDMA_PKT_COPY_L2T_BROADCAST_HEADER_broadcast_mask) << SDMA_PKT_COPY_L2T_BROADCAST_HEADER_broadcast_shift)
SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_LO_0_tiled_addr0_31_0_offset = 1
SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_LO_0_tiled_addr0_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_LO_0_tiled_addr0_31_0_shift = 0
SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_LO_0_TILED_ADDR0_31_0 = lambda x: (((x) & SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_LO_0_tiled_addr0_31_0_mask) << SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_LO_0_tiled_addr0_31_0_shift)
SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_HI_0_tiled_addr0_63_32_offset = 2
SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_HI_0_tiled_addr0_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_HI_0_tiled_addr0_63_32_shift = 0
SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_HI_0_TILED_ADDR0_63_32 = lambda x: (((x) & SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_HI_0_tiled_addr0_63_32_mask) << SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_HI_0_tiled_addr0_63_32_shift)
SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_LO_1_tiled_addr1_31_0_offset = 3
SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_LO_1_tiled_addr1_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_LO_1_tiled_addr1_31_0_shift = 0
SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_LO_1_TILED_ADDR1_31_0 = lambda x: (((x) & SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_LO_1_tiled_addr1_31_0_mask) << SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_LO_1_tiled_addr1_31_0_shift)
SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_HI_1_tiled_addr1_63_32_offset = 4
SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_HI_1_tiled_addr1_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_HI_1_tiled_addr1_63_32_shift = 0
SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_HI_1_TILED_ADDR1_63_32 = lambda x: (((x) & SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_HI_1_tiled_addr1_63_32_mask) << SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_HI_1_tiled_addr1_63_32_shift)
SDMA_PKT_COPY_L2T_BROADCAST_DW_5_width_offset = 5
SDMA_PKT_COPY_L2T_BROADCAST_DW_5_width_mask = 0x00003FFF
SDMA_PKT_COPY_L2T_BROADCAST_DW_5_width_shift = 0
SDMA_PKT_COPY_L2T_BROADCAST_DW_5_WIDTH = lambda x: (((x) & SDMA_PKT_COPY_L2T_BROADCAST_DW_5_width_mask) << SDMA_PKT_COPY_L2T_BROADCAST_DW_5_width_shift)
SDMA_PKT_COPY_L2T_BROADCAST_DW_6_height_offset = 6
SDMA_PKT_COPY_L2T_BROADCAST_DW_6_height_mask = 0x00003FFF
SDMA_PKT_COPY_L2T_BROADCAST_DW_6_height_shift = 0
SDMA_PKT_COPY_L2T_BROADCAST_DW_6_HEIGHT = lambda x: (((x) & SDMA_PKT_COPY_L2T_BROADCAST_DW_6_height_mask) << SDMA_PKT_COPY_L2T_BROADCAST_DW_6_height_shift)
SDMA_PKT_COPY_L2T_BROADCAST_DW_6_depth_offset = 6
SDMA_PKT_COPY_L2T_BROADCAST_DW_6_depth_mask = 0x00001FFF
SDMA_PKT_COPY_L2T_BROADCAST_DW_6_depth_shift = 16
SDMA_PKT_COPY_L2T_BROADCAST_DW_6_DEPTH = lambda x: (((x) & SDMA_PKT_COPY_L2T_BROADCAST_DW_6_depth_mask) << SDMA_PKT_COPY_L2T_BROADCAST_DW_6_depth_shift)
SDMA_PKT_COPY_L2T_BROADCAST_DW_7_element_size_offset = 7
SDMA_PKT_COPY_L2T_BROADCAST_DW_7_element_size_mask = 0x00000007
SDMA_PKT_COPY_L2T_BROADCAST_DW_7_element_size_shift = 0
SDMA_PKT_COPY_L2T_BROADCAST_DW_7_ELEMENT_SIZE = lambda x: (((x) & SDMA_PKT_COPY_L2T_BROADCAST_DW_7_element_size_mask) << SDMA_PKT_COPY_L2T_BROADCAST_DW_7_element_size_shift)
SDMA_PKT_COPY_L2T_BROADCAST_DW_7_swizzle_mode_offset = 7
SDMA_PKT_COPY_L2T_BROADCAST_DW_7_swizzle_mode_mask = 0x0000001F
SDMA_PKT_COPY_L2T_BROADCAST_DW_7_swizzle_mode_shift = 3
SDMA_PKT_COPY_L2T_BROADCAST_DW_7_SWIZZLE_MODE = lambda x: (((x) & SDMA_PKT_COPY_L2T_BROADCAST_DW_7_swizzle_mode_mask) << SDMA_PKT_COPY_L2T_BROADCAST_DW_7_swizzle_mode_shift)
SDMA_PKT_COPY_L2T_BROADCAST_DW_7_dimension_offset = 7
SDMA_PKT_COPY_L2T_BROADCAST_DW_7_dimension_mask = 0x00000003
SDMA_PKT_COPY_L2T_BROADCAST_DW_7_dimension_shift = 9
SDMA_PKT_COPY_L2T_BROADCAST_DW_7_DIMENSION = lambda x: (((x) & SDMA_PKT_COPY_L2T_BROADCAST_DW_7_dimension_mask) << SDMA_PKT_COPY_L2T_BROADCAST_DW_7_dimension_shift)
SDMA_PKT_COPY_L2T_BROADCAST_DW_7_mip_max_offset = 7
SDMA_PKT_COPY_L2T_BROADCAST_DW_7_mip_max_mask = 0x0000000F
SDMA_PKT_COPY_L2T_BROADCAST_DW_7_mip_max_shift = 16
SDMA_PKT_COPY_L2T_BROADCAST_DW_7_MIP_MAX = lambda x: (((x) & SDMA_PKT_COPY_L2T_BROADCAST_DW_7_mip_max_mask) << SDMA_PKT_COPY_L2T_BROADCAST_DW_7_mip_max_shift)
SDMA_PKT_COPY_L2T_BROADCAST_DW_8_x_offset = 8
SDMA_PKT_COPY_L2T_BROADCAST_DW_8_x_mask = 0x00003FFF
SDMA_PKT_COPY_L2T_BROADCAST_DW_8_x_shift = 0
SDMA_PKT_COPY_L2T_BROADCAST_DW_8_X = lambda x: (((x) & SDMA_PKT_COPY_L2T_BROADCAST_DW_8_x_mask) << SDMA_PKT_COPY_L2T_BROADCAST_DW_8_x_shift)
SDMA_PKT_COPY_L2T_BROADCAST_DW_8_y_offset = 8
SDMA_PKT_COPY_L2T_BROADCAST_DW_8_y_mask = 0x00003FFF
SDMA_PKT_COPY_L2T_BROADCAST_DW_8_y_shift = 16
SDMA_PKT_COPY_L2T_BROADCAST_DW_8_Y = lambda x: (((x) & SDMA_PKT_COPY_L2T_BROADCAST_DW_8_y_mask) << SDMA_PKT_COPY_L2T_BROADCAST_DW_8_y_shift)
SDMA_PKT_COPY_L2T_BROADCAST_DW_9_z_offset = 9
SDMA_PKT_COPY_L2T_BROADCAST_DW_9_z_mask = 0x00001FFF
SDMA_PKT_COPY_L2T_BROADCAST_DW_9_z_shift = 0
SDMA_PKT_COPY_L2T_BROADCAST_DW_9_Z = lambda x: (((x) & SDMA_PKT_COPY_L2T_BROADCAST_DW_9_z_mask) << SDMA_PKT_COPY_L2T_BROADCAST_DW_9_z_shift)
SDMA_PKT_COPY_L2T_BROADCAST_DW_10_dst2_sw_offset = 10
SDMA_PKT_COPY_L2T_BROADCAST_DW_10_dst2_sw_mask = 0x00000003
SDMA_PKT_COPY_L2T_BROADCAST_DW_10_dst2_sw_shift = 8
SDMA_PKT_COPY_L2T_BROADCAST_DW_10_DST2_SW = lambda x: (((x) & SDMA_PKT_COPY_L2T_BROADCAST_DW_10_dst2_sw_mask) << SDMA_PKT_COPY_L2T_BROADCAST_DW_10_dst2_sw_shift)
SDMA_PKT_COPY_L2T_BROADCAST_DW_10_dst2_cache_policy_offset = 10
SDMA_PKT_COPY_L2T_BROADCAST_DW_10_dst2_cache_policy_mask = 0x00000007
SDMA_PKT_COPY_L2T_BROADCAST_DW_10_dst2_cache_policy_shift = 10
SDMA_PKT_COPY_L2T_BROADCAST_DW_10_DST2_CACHE_POLICY = lambda x: (((x) & SDMA_PKT_COPY_L2T_BROADCAST_DW_10_dst2_cache_policy_mask) << SDMA_PKT_COPY_L2T_BROADCAST_DW_10_dst2_cache_policy_shift)
SDMA_PKT_COPY_L2T_BROADCAST_DW_10_linear_sw_offset = 10
SDMA_PKT_COPY_L2T_BROADCAST_DW_10_linear_sw_mask = 0x00000003
SDMA_PKT_COPY_L2T_BROADCAST_DW_10_linear_sw_shift = 16
SDMA_PKT_COPY_L2T_BROADCAST_DW_10_LINEAR_SW = lambda x: (((x) & SDMA_PKT_COPY_L2T_BROADCAST_DW_10_linear_sw_mask) << SDMA_PKT_COPY_L2T_BROADCAST_DW_10_linear_sw_shift)
SDMA_PKT_COPY_L2T_BROADCAST_DW_10_linear_cache_policy_offset = 10
SDMA_PKT_COPY_L2T_BROADCAST_DW_10_linear_cache_policy_mask = 0x00000007
SDMA_PKT_COPY_L2T_BROADCAST_DW_10_linear_cache_policy_shift = 18
SDMA_PKT_COPY_L2T_BROADCAST_DW_10_LINEAR_CACHE_POLICY = lambda x: (((x) & SDMA_PKT_COPY_L2T_BROADCAST_DW_10_linear_cache_policy_mask) << SDMA_PKT_COPY_L2T_BROADCAST_DW_10_linear_cache_policy_shift)
SDMA_PKT_COPY_L2T_BROADCAST_DW_10_tile_sw_offset = 10
SDMA_PKT_COPY_L2T_BROADCAST_DW_10_tile_sw_mask = 0x00000003
SDMA_PKT_COPY_L2T_BROADCAST_DW_10_tile_sw_shift = 24
SDMA_PKT_COPY_L2T_BROADCAST_DW_10_TILE_SW = lambda x: (((x) & SDMA_PKT_COPY_L2T_BROADCAST_DW_10_tile_sw_mask) << SDMA_PKT_COPY_L2T_BROADCAST_DW_10_tile_sw_shift)
SDMA_PKT_COPY_L2T_BROADCAST_DW_10_tile_cache_policy_offset = 10
SDMA_PKT_COPY_L2T_BROADCAST_DW_10_tile_cache_policy_mask = 0x00000007
SDMA_PKT_COPY_L2T_BROADCAST_DW_10_tile_cache_policy_shift = 26
SDMA_PKT_COPY_L2T_BROADCAST_DW_10_TILE_CACHE_POLICY = lambda x: (((x) & SDMA_PKT_COPY_L2T_BROADCAST_DW_10_tile_cache_policy_mask) << SDMA_PKT_COPY_L2T_BROADCAST_DW_10_tile_cache_policy_shift)
SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_ADDR_LO_linear_addr_31_0_offset = 11
SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_ADDR_LO_linear_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_ADDR_LO_linear_addr_31_0_shift = 0
SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_ADDR_LO_LINEAR_ADDR_31_0 = lambda x: (((x) & SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_ADDR_LO_linear_addr_31_0_mask) << SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_ADDR_LO_linear_addr_31_0_shift)
SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_ADDR_HI_linear_addr_63_32_offset = 12
SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_ADDR_HI_linear_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_ADDR_HI_linear_addr_63_32_shift = 0
SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_ADDR_HI_LINEAR_ADDR_63_32 = lambda x: (((x) & SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_ADDR_HI_linear_addr_63_32_mask) << SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_ADDR_HI_linear_addr_63_32_shift)
SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_PITCH_linear_pitch_offset = 13
SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_PITCH_linear_pitch_mask = 0x0007FFFF
SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_PITCH_linear_pitch_shift = 0
SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_PITCH_LINEAR_PITCH = lambda x: (((x) & SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_PITCH_linear_pitch_mask) << SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_PITCH_linear_pitch_shift)
SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_SLICE_PITCH_linear_slice_pitch_offset = 14
SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_SLICE_PITCH_linear_slice_pitch_mask = 0xFFFFFFFF
SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_SLICE_PITCH_linear_slice_pitch_shift = 0
SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_SLICE_PITCH_LINEAR_SLICE_PITCH = lambda x: (((x) & SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_SLICE_PITCH_linear_slice_pitch_mask) << SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_SLICE_PITCH_linear_slice_pitch_shift)
SDMA_PKT_COPY_L2T_BROADCAST_COUNT_count_offset = 15
SDMA_PKT_COPY_L2T_BROADCAST_COUNT_count_mask = 0x3FFFFFFF
SDMA_PKT_COPY_L2T_BROADCAST_COUNT_count_shift = 0
SDMA_PKT_COPY_L2T_BROADCAST_COUNT_COUNT = lambda x: (((x) & SDMA_PKT_COPY_L2T_BROADCAST_COUNT_count_mask) << SDMA_PKT_COPY_L2T_BROADCAST_COUNT_count_shift)
SDMA_PKT_COPY_T2T_HEADER_op_offset = 0
SDMA_PKT_COPY_T2T_HEADER_op_mask = 0x000000FF
SDMA_PKT_COPY_T2T_HEADER_op_shift = 0
SDMA_PKT_COPY_T2T_HEADER_OP = lambda x: (((x) & SDMA_PKT_COPY_T2T_HEADER_op_mask) << SDMA_PKT_COPY_T2T_HEADER_op_shift)
SDMA_PKT_COPY_T2T_HEADER_sub_op_offset = 0
SDMA_PKT_COPY_T2T_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_COPY_T2T_HEADER_sub_op_shift = 8
SDMA_PKT_COPY_T2T_HEADER_SUB_OP = lambda x: (((x) & SDMA_PKT_COPY_T2T_HEADER_sub_op_mask) << SDMA_PKT_COPY_T2T_HEADER_sub_op_shift)
SDMA_PKT_COPY_T2T_HEADER_tmz_offset = 0
SDMA_PKT_COPY_T2T_HEADER_tmz_mask = 0x00000001
SDMA_PKT_COPY_T2T_HEADER_tmz_shift = 18
SDMA_PKT_COPY_T2T_HEADER_TMZ = lambda x: (((x) & SDMA_PKT_COPY_T2T_HEADER_tmz_mask) << SDMA_PKT_COPY_T2T_HEADER_tmz_shift)
SDMA_PKT_COPY_T2T_HEADER_dcc_offset = 0
SDMA_PKT_COPY_T2T_HEADER_dcc_mask = 0x00000001
SDMA_PKT_COPY_T2T_HEADER_dcc_shift = 19
SDMA_PKT_COPY_T2T_HEADER_DCC = lambda x: (((x) & SDMA_PKT_COPY_T2T_HEADER_dcc_mask) << SDMA_PKT_COPY_T2T_HEADER_dcc_shift)
SDMA_PKT_COPY_T2T_HEADER_cpv_offset = 0
SDMA_PKT_COPY_T2T_HEADER_cpv_mask = 0x00000001
SDMA_PKT_COPY_T2T_HEADER_cpv_shift = 28
SDMA_PKT_COPY_T2T_HEADER_CPV = lambda x: (((x) & SDMA_PKT_COPY_T2T_HEADER_cpv_mask) << SDMA_PKT_COPY_T2T_HEADER_cpv_shift)
SDMA_PKT_COPY_T2T_HEADER_dcc_dir_offset = 0
SDMA_PKT_COPY_T2T_HEADER_dcc_dir_mask = 0x00000001
SDMA_PKT_COPY_T2T_HEADER_dcc_dir_shift = 31
SDMA_PKT_COPY_T2T_HEADER_DCC_DIR = lambda x: (((x) & SDMA_PKT_COPY_T2T_HEADER_dcc_dir_mask) << SDMA_PKT_COPY_T2T_HEADER_dcc_dir_shift)
SDMA_PKT_COPY_T2T_SRC_ADDR_LO_src_addr_31_0_offset = 1
SDMA_PKT_COPY_T2T_SRC_ADDR_LO_src_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_T2T_SRC_ADDR_LO_src_addr_31_0_shift = 0
SDMA_PKT_COPY_T2T_SRC_ADDR_LO_SRC_ADDR_31_0 = lambda x: (((x) & SDMA_PKT_COPY_T2T_SRC_ADDR_LO_src_addr_31_0_mask) << SDMA_PKT_COPY_T2T_SRC_ADDR_LO_src_addr_31_0_shift)
SDMA_PKT_COPY_T2T_SRC_ADDR_HI_src_addr_63_32_offset = 2
SDMA_PKT_COPY_T2T_SRC_ADDR_HI_src_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_T2T_SRC_ADDR_HI_src_addr_63_32_shift = 0
SDMA_PKT_COPY_T2T_SRC_ADDR_HI_SRC_ADDR_63_32 = lambda x: (((x) & SDMA_PKT_COPY_T2T_SRC_ADDR_HI_src_addr_63_32_mask) << SDMA_PKT_COPY_T2T_SRC_ADDR_HI_src_addr_63_32_shift)
SDMA_PKT_COPY_T2T_DW_3_src_x_offset = 3
SDMA_PKT_COPY_T2T_DW_3_src_x_mask = 0x00003FFF
SDMA_PKT_COPY_T2T_DW_3_src_x_shift = 0
SDMA_PKT_COPY_T2T_DW_3_SRC_X = lambda x: (((x) & SDMA_PKT_COPY_T2T_DW_3_src_x_mask) << SDMA_PKT_COPY_T2T_DW_3_src_x_shift)
SDMA_PKT_COPY_T2T_DW_3_src_y_offset = 3
SDMA_PKT_COPY_T2T_DW_3_src_y_mask = 0x00003FFF
SDMA_PKT_COPY_T2T_DW_3_src_y_shift = 16
SDMA_PKT_COPY_T2T_DW_3_SRC_Y = lambda x: (((x) & SDMA_PKT_COPY_T2T_DW_3_src_y_mask) << SDMA_PKT_COPY_T2T_DW_3_src_y_shift)
SDMA_PKT_COPY_T2T_DW_4_src_z_offset = 4
SDMA_PKT_COPY_T2T_DW_4_src_z_mask = 0x00001FFF
SDMA_PKT_COPY_T2T_DW_4_src_z_shift = 0
SDMA_PKT_COPY_T2T_DW_4_SRC_Z = lambda x: (((x) & SDMA_PKT_COPY_T2T_DW_4_src_z_mask) << SDMA_PKT_COPY_T2T_DW_4_src_z_shift)
SDMA_PKT_COPY_T2T_DW_4_src_width_offset = 4
SDMA_PKT_COPY_T2T_DW_4_src_width_mask = 0x00003FFF
SDMA_PKT_COPY_T2T_DW_4_src_width_shift = 16
SDMA_PKT_COPY_T2T_DW_4_SRC_WIDTH = lambda x: (((x) & SDMA_PKT_COPY_T2T_DW_4_src_width_mask) << SDMA_PKT_COPY_T2T_DW_4_src_width_shift)
SDMA_PKT_COPY_T2T_DW_5_src_height_offset = 5
SDMA_PKT_COPY_T2T_DW_5_src_height_mask = 0x00003FFF
SDMA_PKT_COPY_T2T_DW_5_src_height_shift = 0
SDMA_PKT_COPY_T2T_DW_5_SRC_HEIGHT = lambda x: (((x) & SDMA_PKT_COPY_T2T_DW_5_src_height_mask) << SDMA_PKT_COPY_T2T_DW_5_src_height_shift)
SDMA_PKT_COPY_T2T_DW_5_src_depth_offset = 5
SDMA_PKT_COPY_T2T_DW_5_src_depth_mask = 0x00001FFF
SDMA_PKT_COPY_T2T_DW_5_src_depth_shift = 16
SDMA_PKT_COPY_T2T_DW_5_SRC_DEPTH = lambda x: (((x) & SDMA_PKT_COPY_T2T_DW_5_src_depth_mask) << SDMA_PKT_COPY_T2T_DW_5_src_depth_shift)
SDMA_PKT_COPY_T2T_DW_6_src_element_size_offset = 6
SDMA_PKT_COPY_T2T_DW_6_src_element_size_mask = 0x00000007
SDMA_PKT_COPY_T2T_DW_6_src_element_size_shift = 0
SDMA_PKT_COPY_T2T_DW_6_SRC_ELEMENT_SIZE = lambda x: (((x) & SDMA_PKT_COPY_T2T_DW_6_src_element_size_mask) << SDMA_PKT_COPY_T2T_DW_6_src_element_size_shift)
SDMA_PKT_COPY_T2T_DW_6_src_swizzle_mode_offset = 6
SDMA_PKT_COPY_T2T_DW_6_src_swizzle_mode_mask = 0x0000001F
SDMA_PKT_COPY_T2T_DW_6_src_swizzle_mode_shift = 3
SDMA_PKT_COPY_T2T_DW_6_SRC_SWIZZLE_MODE = lambda x: (((x) & SDMA_PKT_COPY_T2T_DW_6_src_swizzle_mode_mask) << SDMA_PKT_COPY_T2T_DW_6_src_swizzle_mode_shift)
SDMA_PKT_COPY_T2T_DW_6_src_dimension_offset = 6
SDMA_PKT_COPY_T2T_DW_6_src_dimension_mask = 0x00000003
SDMA_PKT_COPY_T2T_DW_6_src_dimension_shift = 9
SDMA_PKT_COPY_T2T_DW_6_SRC_DIMENSION = lambda x: (((x) & SDMA_PKT_COPY_T2T_DW_6_src_dimension_mask) << SDMA_PKT_COPY_T2T_DW_6_src_dimension_shift)
SDMA_PKT_COPY_T2T_DW_6_src_mip_max_offset = 6
SDMA_PKT_COPY_T2T_DW_6_src_mip_max_mask = 0x0000000F
SDMA_PKT_COPY_T2T_DW_6_src_mip_max_shift = 16
SDMA_PKT_COPY_T2T_DW_6_SRC_MIP_MAX = lambda x: (((x) & SDMA_PKT_COPY_T2T_DW_6_src_mip_max_mask) << SDMA_PKT_COPY_T2T_DW_6_src_mip_max_shift)
SDMA_PKT_COPY_T2T_DW_6_src_mip_id_offset = 6
SDMA_PKT_COPY_T2T_DW_6_src_mip_id_mask = 0x0000000F
SDMA_PKT_COPY_T2T_DW_6_src_mip_id_shift = 20
SDMA_PKT_COPY_T2T_DW_6_SRC_MIP_ID = lambda x: (((x) & SDMA_PKT_COPY_T2T_DW_6_src_mip_id_mask) << SDMA_PKT_COPY_T2T_DW_6_src_mip_id_shift)
SDMA_PKT_COPY_T2T_DST_ADDR_LO_dst_addr_31_0_offset = 7
SDMA_PKT_COPY_T2T_DST_ADDR_LO_dst_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_T2T_DST_ADDR_LO_dst_addr_31_0_shift = 0
SDMA_PKT_COPY_T2T_DST_ADDR_LO_DST_ADDR_31_0 = lambda x: (((x) & SDMA_PKT_COPY_T2T_DST_ADDR_LO_dst_addr_31_0_mask) << SDMA_PKT_COPY_T2T_DST_ADDR_LO_dst_addr_31_0_shift)
SDMA_PKT_COPY_T2T_DST_ADDR_HI_dst_addr_63_32_offset = 8
SDMA_PKT_COPY_T2T_DST_ADDR_HI_dst_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_T2T_DST_ADDR_HI_dst_addr_63_32_shift = 0
SDMA_PKT_COPY_T2T_DST_ADDR_HI_DST_ADDR_63_32 = lambda x: (((x) & SDMA_PKT_COPY_T2T_DST_ADDR_HI_dst_addr_63_32_mask) << SDMA_PKT_COPY_T2T_DST_ADDR_HI_dst_addr_63_32_shift)
SDMA_PKT_COPY_T2T_DW_9_dst_x_offset = 9
SDMA_PKT_COPY_T2T_DW_9_dst_x_mask = 0x00003FFF
SDMA_PKT_COPY_T2T_DW_9_dst_x_shift = 0
SDMA_PKT_COPY_T2T_DW_9_DST_X = lambda x: (((x) & SDMA_PKT_COPY_T2T_DW_9_dst_x_mask) << SDMA_PKT_COPY_T2T_DW_9_dst_x_shift)
SDMA_PKT_COPY_T2T_DW_9_dst_y_offset = 9
SDMA_PKT_COPY_T2T_DW_9_dst_y_mask = 0x00003FFF
SDMA_PKT_COPY_T2T_DW_9_dst_y_shift = 16
SDMA_PKT_COPY_T2T_DW_9_DST_Y = lambda x: (((x) & SDMA_PKT_COPY_T2T_DW_9_dst_y_mask) << SDMA_PKT_COPY_T2T_DW_9_dst_y_shift)
SDMA_PKT_COPY_T2T_DW_10_dst_z_offset = 10
SDMA_PKT_COPY_T2T_DW_10_dst_z_mask = 0x00001FFF
SDMA_PKT_COPY_T2T_DW_10_dst_z_shift = 0
SDMA_PKT_COPY_T2T_DW_10_DST_Z = lambda x: (((x) & SDMA_PKT_COPY_T2T_DW_10_dst_z_mask) << SDMA_PKT_COPY_T2T_DW_10_dst_z_shift)
SDMA_PKT_COPY_T2T_DW_10_dst_width_offset = 10
SDMA_PKT_COPY_T2T_DW_10_dst_width_mask = 0x00003FFF
SDMA_PKT_COPY_T2T_DW_10_dst_width_shift = 16
SDMA_PKT_COPY_T2T_DW_10_DST_WIDTH = lambda x: (((x) & SDMA_PKT_COPY_T2T_DW_10_dst_width_mask) << SDMA_PKT_COPY_T2T_DW_10_dst_width_shift)
SDMA_PKT_COPY_T2T_DW_11_dst_height_offset = 11
SDMA_PKT_COPY_T2T_DW_11_dst_height_mask = 0x00003FFF
SDMA_PKT_COPY_T2T_DW_11_dst_height_shift = 0
SDMA_PKT_COPY_T2T_DW_11_DST_HEIGHT = lambda x: (((x) & SDMA_PKT_COPY_T2T_DW_11_dst_height_mask) << SDMA_PKT_COPY_T2T_DW_11_dst_height_shift)
SDMA_PKT_COPY_T2T_DW_11_dst_depth_offset = 11
SDMA_PKT_COPY_T2T_DW_11_dst_depth_mask = 0x00001FFF
SDMA_PKT_COPY_T2T_DW_11_dst_depth_shift = 16
SDMA_PKT_COPY_T2T_DW_11_DST_DEPTH = lambda x: (((x) & SDMA_PKT_COPY_T2T_DW_11_dst_depth_mask) << SDMA_PKT_COPY_T2T_DW_11_dst_depth_shift)
SDMA_PKT_COPY_T2T_DW_12_dst_element_size_offset = 12
SDMA_PKT_COPY_T2T_DW_12_dst_element_size_mask = 0x00000007
SDMA_PKT_COPY_T2T_DW_12_dst_element_size_shift = 0
SDMA_PKT_COPY_T2T_DW_12_DST_ELEMENT_SIZE = lambda x: (((x) & SDMA_PKT_COPY_T2T_DW_12_dst_element_size_mask) << SDMA_PKT_COPY_T2T_DW_12_dst_element_size_shift)
SDMA_PKT_COPY_T2T_DW_12_dst_swizzle_mode_offset = 12
SDMA_PKT_COPY_T2T_DW_12_dst_swizzle_mode_mask = 0x0000001F
SDMA_PKT_COPY_T2T_DW_12_dst_swizzle_mode_shift = 3
SDMA_PKT_COPY_T2T_DW_12_DST_SWIZZLE_MODE = lambda x: (((x) & SDMA_PKT_COPY_T2T_DW_12_dst_swizzle_mode_mask) << SDMA_PKT_COPY_T2T_DW_12_dst_swizzle_mode_shift)
SDMA_PKT_COPY_T2T_DW_12_dst_dimension_offset = 12
SDMA_PKT_COPY_T2T_DW_12_dst_dimension_mask = 0x00000003
SDMA_PKT_COPY_T2T_DW_12_dst_dimension_shift = 9
SDMA_PKT_COPY_T2T_DW_12_DST_DIMENSION = lambda x: (((x) & SDMA_PKT_COPY_T2T_DW_12_dst_dimension_mask) << SDMA_PKT_COPY_T2T_DW_12_dst_dimension_shift)
SDMA_PKT_COPY_T2T_DW_12_dst_mip_max_offset = 12
SDMA_PKT_COPY_T2T_DW_12_dst_mip_max_mask = 0x0000000F
SDMA_PKT_COPY_T2T_DW_12_dst_mip_max_shift = 16
SDMA_PKT_COPY_T2T_DW_12_DST_MIP_MAX = lambda x: (((x) & SDMA_PKT_COPY_T2T_DW_12_dst_mip_max_mask) << SDMA_PKT_COPY_T2T_DW_12_dst_mip_max_shift)
SDMA_PKT_COPY_T2T_DW_12_dst_mip_id_offset = 12
SDMA_PKT_COPY_T2T_DW_12_dst_mip_id_mask = 0x0000000F
SDMA_PKT_COPY_T2T_DW_12_dst_mip_id_shift = 20
SDMA_PKT_COPY_T2T_DW_12_DST_MIP_ID = lambda x: (((x) & SDMA_PKT_COPY_T2T_DW_12_dst_mip_id_mask) << SDMA_PKT_COPY_T2T_DW_12_dst_mip_id_shift)
SDMA_PKT_COPY_T2T_DW_13_rect_x_offset = 13
SDMA_PKT_COPY_T2T_DW_13_rect_x_mask = 0x00003FFF
SDMA_PKT_COPY_T2T_DW_13_rect_x_shift = 0
SDMA_PKT_COPY_T2T_DW_13_RECT_X = lambda x: (((x) & SDMA_PKT_COPY_T2T_DW_13_rect_x_mask) << SDMA_PKT_COPY_T2T_DW_13_rect_x_shift)
SDMA_PKT_COPY_T2T_DW_13_rect_y_offset = 13
SDMA_PKT_COPY_T2T_DW_13_rect_y_mask = 0x00003FFF
SDMA_PKT_COPY_T2T_DW_13_rect_y_shift = 16
SDMA_PKT_COPY_T2T_DW_13_RECT_Y = lambda x: (((x) & SDMA_PKT_COPY_T2T_DW_13_rect_y_mask) << SDMA_PKT_COPY_T2T_DW_13_rect_y_shift)
SDMA_PKT_COPY_T2T_DW_14_rect_z_offset = 14
SDMA_PKT_COPY_T2T_DW_14_rect_z_mask = 0x00001FFF
SDMA_PKT_COPY_T2T_DW_14_rect_z_shift = 0
SDMA_PKT_COPY_T2T_DW_14_RECT_Z = lambda x: (((x) & SDMA_PKT_COPY_T2T_DW_14_rect_z_mask) << SDMA_PKT_COPY_T2T_DW_14_rect_z_shift)
SDMA_PKT_COPY_T2T_DW_14_dst_sw_offset = 14
SDMA_PKT_COPY_T2T_DW_14_dst_sw_mask = 0x00000003
SDMA_PKT_COPY_T2T_DW_14_dst_sw_shift = 16
SDMA_PKT_COPY_T2T_DW_14_DST_SW = lambda x: (((x) & SDMA_PKT_COPY_T2T_DW_14_dst_sw_mask) << SDMA_PKT_COPY_T2T_DW_14_dst_sw_shift)
SDMA_PKT_COPY_T2T_DW_14_dst_cache_policy_offset = 14
SDMA_PKT_COPY_T2T_DW_14_dst_cache_policy_mask = 0x00000007
SDMA_PKT_COPY_T2T_DW_14_dst_cache_policy_shift = 18
SDMA_PKT_COPY_T2T_DW_14_DST_CACHE_POLICY = lambda x: (((x) & SDMA_PKT_COPY_T2T_DW_14_dst_cache_policy_mask) << SDMA_PKT_COPY_T2T_DW_14_dst_cache_policy_shift)
SDMA_PKT_COPY_T2T_DW_14_src_sw_offset = 14
SDMA_PKT_COPY_T2T_DW_14_src_sw_mask = 0x00000003
SDMA_PKT_COPY_T2T_DW_14_src_sw_shift = 24
SDMA_PKT_COPY_T2T_DW_14_SRC_SW = lambda x: (((x) & SDMA_PKT_COPY_T2T_DW_14_src_sw_mask) << SDMA_PKT_COPY_T2T_DW_14_src_sw_shift)
SDMA_PKT_COPY_T2T_DW_14_src_cache_policy_offset = 14
SDMA_PKT_COPY_T2T_DW_14_src_cache_policy_mask = 0x00000007
SDMA_PKT_COPY_T2T_DW_14_src_cache_policy_shift = 26
SDMA_PKT_COPY_T2T_DW_14_SRC_CACHE_POLICY = lambda x: (((x) & SDMA_PKT_COPY_T2T_DW_14_src_cache_policy_mask) << SDMA_PKT_COPY_T2T_DW_14_src_cache_policy_shift)
SDMA_PKT_COPY_T2T_META_ADDR_LO_meta_addr_31_0_offset = 15
SDMA_PKT_COPY_T2T_META_ADDR_LO_meta_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_T2T_META_ADDR_LO_meta_addr_31_0_shift = 0
SDMA_PKT_COPY_T2T_META_ADDR_LO_META_ADDR_31_0 = lambda x: (((x) & SDMA_PKT_COPY_T2T_META_ADDR_LO_meta_addr_31_0_mask) << SDMA_PKT_COPY_T2T_META_ADDR_LO_meta_addr_31_0_shift)
SDMA_PKT_COPY_T2T_META_ADDR_HI_meta_addr_63_32_offset = 16
SDMA_PKT_COPY_T2T_META_ADDR_HI_meta_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_T2T_META_ADDR_HI_meta_addr_63_32_shift = 0
SDMA_PKT_COPY_T2T_META_ADDR_HI_META_ADDR_63_32 = lambda x: (((x) & SDMA_PKT_COPY_T2T_META_ADDR_HI_meta_addr_63_32_mask) << SDMA_PKT_COPY_T2T_META_ADDR_HI_meta_addr_63_32_shift)
SDMA_PKT_COPY_T2T_META_CONFIG_data_format_offset = 17
SDMA_PKT_COPY_T2T_META_CONFIG_data_format_mask = 0x0000007F
SDMA_PKT_COPY_T2T_META_CONFIG_data_format_shift = 0
SDMA_PKT_COPY_T2T_META_CONFIG_DATA_FORMAT = lambda x: (((x) & SDMA_PKT_COPY_T2T_META_CONFIG_data_format_mask) << SDMA_PKT_COPY_T2T_META_CONFIG_data_format_shift)
SDMA_PKT_COPY_T2T_META_CONFIG_color_transform_disable_offset = 17
SDMA_PKT_COPY_T2T_META_CONFIG_color_transform_disable_mask = 0x00000001
SDMA_PKT_COPY_T2T_META_CONFIG_color_transform_disable_shift = 7
SDMA_PKT_COPY_T2T_META_CONFIG_COLOR_TRANSFORM_DISABLE = lambda x: (((x) & SDMA_PKT_COPY_T2T_META_CONFIG_color_transform_disable_mask) << SDMA_PKT_COPY_T2T_META_CONFIG_color_transform_disable_shift)
SDMA_PKT_COPY_T2T_META_CONFIG_alpha_is_on_msb_offset = 17
SDMA_PKT_COPY_T2T_META_CONFIG_alpha_is_on_msb_mask = 0x00000001
SDMA_PKT_COPY_T2T_META_CONFIG_alpha_is_on_msb_shift = 8
SDMA_PKT_COPY_T2T_META_CONFIG_ALPHA_IS_ON_MSB = lambda x: (((x) & SDMA_PKT_COPY_T2T_META_CONFIG_alpha_is_on_msb_mask) << SDMA_PKT_COPY_T2T_META_CONFIG_alpha_is_on_msb_shift)
SDMA_PKT_COPY_T2T_META_CONFIG_number_type_offset = 17
SDMA_PKT_COPY_T2T_META_CONFIG_number_type_mask = 0x00000007
SDMA_PKT_COPY_T2T_META_CONFIG_number_type_shift = 9
SDMA_PKT_COPY_T2T_META_CONFIG_NUMBER_TYPE = lambda x: (((x) & SDMA_PKT_COPY_T2T_META_CONFIG_number_type_mask) << SDMA_PKT_COPY_T2T_META_CONFIG_number_type_shift)
SDMA_PKT_COPY_T2T_META_CONFIG_surface_type_offset = 17
SDMA_PKT_COPY_T2T_META_CONFIG_surface_type_mask = 0x00000003
SDMA_PKT_COPY_T2T_META_CONFIG_surface_type_shift = 12
SDMA_PKT_COPY_T2T_META_CONFIG_SURFACE_TYPE = lambda x: (((x) & SDMA_PKT_COPY_T2T_META_CONFIG_surface_type_mask) << SDMA_PKT_COPY_T2T_META_CONFIG_surface_type_shift)
SDMA_PKT_COPY_T2T_META_CONFIG_meta_llc_offset = 17
SDMA_PKT_COPY_T2T_META_CONFIG_meta_llc_mask = 0x00000001
SDMA_PKT_COPY_T2T_META_CONFIG_meta_llc_shift = 14
SDMA_PKT_COPY_T2T_META_CONFIG_META_LLC = lambda x: (((x) & SDMA_PKT_COPY_T2T_META_CONFIG_meta_llc_mask) << SDMA_PKT_COPY_T2T_META_CONFIG_meta_llc_shift)
SDMA_PKT_COPY_T2T_META_CONFIG_max_comp_block_size_offset = 17
SDMA_PKT_COPY_T2T_META_CONFIG_max_comp_block_size_mask = 0x00000003
SDMA_PKT_COPY_T2T_META_CONFIG_max_comp_block_size_shift = 24
SDMA_PKT_COPY_T2T_META_CONFIG_MAX_COMP_BLOCK_SIZE = lambda x: (((x) & SDMA_PKT_COPY_T2T_META_CONFIG_max_comp_block_size_mask) << SDMA_PKT_COPY_T2T_META_CONFIG_max_comp_block_size_shift)
SDMA_PKT_COPY_T2T_META_CONFIG_max_uncomp_block_size_offset = 17
SDMA_PKT_COPY_T2T_META_CONFIG_max_uncomp_block_size_mask = 0x00000003
SDMA_PKT_COPY_T2T_META_CONFIG_max_uncomp_block_size_shift = 26
SDMA_PKT_COPY_T2T_META_CONFIG_MAX_UNCOMP_BLOCK_SIZE = lambda x: (((x) & SDMA_PKT_COPY_T2T_META_CONFIG_max_uncomp_block_size_mask) << SDMA_PKT_COPY_T2T_META_CONFIG_max_uncomp_block_size_shift)
SDMA_PKT_COPY_T2T_META_CONFIG_write_compress_enable_offset = 17
SDMA_PKT_COPY_T2T_META_CONFIG_write_compress_enable_mask = 0x00000001
SDMA_PKT_COPY_T2T_META_CONFIG_write_compress_enable_shift = 28
SDMA_PKT_COPY_T2T_META_CONFIG_WRITE_COMPRESS_ENABLE = lambda x: (((x) & SDMA_PKT_COPY_T2T_META_CONFIG_write_compress_enable_mask) << SDMA_PKT_COPY_T2T_META_CONFIG_write_compress_enable_shift)
SDMA_PKT_COPY_T2T_META_CONFIG_meta_tmz_offset = 17
SDMA_PKT_COPY_T2T_META_CONFIG_meta_tmz_mask = 0x00000001
SDMA_PKT_COPY_T2T_META_CONFIG_meta_tmz_shift = 29
SDMA_PKT_COPY_T2T_META_CONFIG_META_TMZ = lambda x: (((x) & SDMA_PKT_COPY_T2T_META_CONFIG_meta_tmz_mask) << SDMA_PKT_COPY_T2T_META_CONFIG_meta_tmz_shift)
SDMA_PKT_COPY_T2T_META_CONFIG_pipe_aligned_offset = 17
SDMA_PKT_COPY_T2T_META_CONFIG_pipe_aligned_mask = 0x00000001
SDMA_PKT_COPY_T2T_META_CONFIG_pipe_aligned_shift = 31
SDMA_PKT_COPY_T2T_META_CONFIG_PIPE_ALIGNED = lambda x: (((x) & SDMA_PKT_COPY_T2T_META_CONFIG_pipe_aligned_mask) << SDMA_PKT_COPY_T2T_META_CONFIG_pipe_aligned_shift)
SDMA_PKT_COPY_T2T_BC_HEADER_op_offset = 0
SDMA_PKT_COPY_T2T_BC_HEADER_op_mask = 0x000000FF
SDMA_PKT_COPY_T2T_BC_HEADER_op_shift = 0
SDMA_PKT_COPY_T2T_BC_HEADER_OP = lambda x: (((x) & SDMA_PKT_COPY_T2T_BC_HEADER_op_mask) << SDMA_PKT_COPY_T2T_BC_HEADER_op_shift)
SDMA_PKT_COPY_T2T_BC_HEADER_sub_op_offset = 0
SDMA_PKT_COPY_T2T_BC_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_COPY_T2T_BC_HEADER_sub_op_shift = 8
SDMA_PKT_COPY_T2T_BC_HEADER_SUB_OP = lambda x: (((x) & SDMA_PKT_COPY_T2T_BC_HEADER_sub_op_mask) << SDMA_PKT_COPY_T2T_BC_HEADER_sub_op_shift)
SDMA_PKT_COPY_T2T_BC_SRC_ADDR_LO_src_addr_31_0_offset = 1
SDMA_PKT_COPY_T2T_BC_SRC_ADDR_LO_src_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_T2T_BC_SRC_ADDR_LO_src_addr_31_0_shift = 0
SDMA_PKT_COPY_T2T_BC_SRC_ADDR_LO_SRC_ADDR_31_0 = lambda x: (((x) & SDMA_PKT_COPY_T2T_BC_SRC_ADDR_LO_src_addr_31_0_mask) << SDMA_PKT_COPY_T2T_BC_SRC_ADDR_LO_src_addr_31_0_shift)
SDMA_PKT_COPY_T2T_BC_SRC_ADDR_HI_src_addr_63_32_offset = 2
SDMA_PKT_COPY_T2T_BC_SRC_ADDR_HI_src_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_T2T_BC_SRC_ADDR_HI_src_addr_63_32_shift = 0
SDMA_PKT_COPY_T2T_BC_SRC_ADDR_HI_SRC_ADDR_63_32 = lambda x: (((x) & SDMA_PKT_COPY_T2T_BC_SRC_ADDR_HI_src_addr_63_32_mask) << SDMA_PKT_COPY_T2T_BC_SRC_ADDR_HI_src_addr_63_32_shift)
SDMA_PKT_COPY_T2T_BC_DW_3_src_x_offset = 3
SDMA_PKT_COPY_T2T_BC_DW_3_src_x_mask = 0x00003FFF
SDMA_PKT_COPY_T2T_BC_DW_3_src_x_shift = 0
SDMA_PKT_COPY_T2T_BC_DW_3_SRC_X = lambda x: (((x) & SDMA_PKT_COPY_T2T_BC_DW_3_src_x_mask) << SDMA_PKT_COPY_T2T_BC_DW_3_src_x_shift)
SDMA_PKT_COPY_T2T_BC_DW_3_src_y_offset = 3
SDMA_PKT_COPY_T2T_BC_DW_3_src_y_mask = 0x00003FFF
SDMA_PKT_COPY_T2T_BC_DW_3_src_y_shift = 16
SDMA_PKT_COPY_T2T_BC_DW_3_SRC_Y = lambda x: (((x) & SDMA_PKT_COPY_T2T_BC_DW_3_src_y_mask) << SDMA_PKT_COPY_T2T_BC_DW_3_src_y_shift)
SDMA_PKT_COPY_T2T_BC_DW_4_src_z_offset = 4
SDMA_PKT_COPY_T2T_BC_DW_4_src_z_mask = 0x000007FF
SDMA_PKT_COPY_T2T_BC_DW_4_src_z_shift = 0
SDMA_PKT_COPY_T2T_BC_DW_4_SRC_Z = lambda x: (((x) & SDMA_PKT_COPY_T2T_BC_DW_4_src_z_mask) << SDMA_PKT_COPY_T2T_BC_DW_4_src_z_shift)
SDMA_PKT_COPY_T2T_BC_DW_4_src_width_offset = 4
SDMA_PKT_COPY_T2T_BC_DW_4_src_width_mask = 0x00003FFF
SDMA_PKT_COPY_T2T_BC_DW_4_src_width_shift = 16
SDMA_PKT_COPY_T2T_BC_DW_4_SRC_WIDTH = lambda x: (((x) & SDMA_PKT_COPY_T2T_BC_DW_4_src_width_mask) << SDMA_PKT_COPY_T2T_BC_DW_4_src_width_shift)
SDMA_PKT_COPY_T2T_BC_DW_5_src_height_offset = 5
SDMA_PKT_COPY_T2T_BC_DW_5_src_height_mask = 0x00003FFF
SDMA_PKT_COPY_T2T_BC_DW_5_src_height_shift = 0
SDMA_PKT_COPY_T2T_BC_DW_5_SRC_HEIGHT = lambda x: (((x) & SDMA_PKT_COPY_T2T_BC_DW_5_src_height_mask) << SDMA_PKT_COPY_T2T_BC_DW_5_src_height_shift)
SDMA_PKT_COPY_T2T_BC_DW_5_src_depth_offset = 5
SDMA_PKT_COPY_T2T_BC_DW_5_src_depth_mask = 0x000007FF
SDMA_PKT_COPY_T2T_BC_DW_5_src_depth_shift = 16
SDMA_PKT_COPY_T2T_BC_DW_5_SRC_DEPTH = lambda x: (((x) & SDMA_PKT_COPY_T2T_BC_DW_5_src_depth_mask) << SDMA_PKT_COPY_T2T_BC_DW_5_src_depth_shift)
SDMA_PKT_COPY_T2T_BC_DW_6_src_element_size_offset = 6
SDMA_PKT_COPY_T2T_BC_DW_6_src_element_size_mask = 0x00000007
SDMA_PKT_COPY_T2T_BC_DW_6_src_element_size_shift = 0
SDMA_PKT_COPY_T2T_BC_DW_6_SRC_ELEMENT_SIZE = lambda x: (((x) & SDMA_PKT_COPY_T2T_BC_DW_6_src_element_size_mask) << SDMA_PKT_COPY_T2T_BC_DW_6_src_element_size_shift)
SDMA_PKT_COPY_T2T_BC_DW_6_src_array_mode_offset = 6
SDMA_PKT_COPY_T2T_BC_DW_6_src_array_mode_mask = 0x0000000F
SDMA_PKT_COPY_T2T_BC_DW_6_src_array_mode_shift = 3
SDMA_PKT_COPY_T2T_BC_DW_6_SRC_ARRAY_MODE = lambda x: (((x) & SDMA_PKT_COPY_T2T_BC_DW_6_src_array_mode_mask) << SDMA_PKT_COPY_T2T_BC_DW_6_src_array_mode_shift)
SDMA_PKT_COPY_T2T_BC_DW_6_src_mit_mode_offset = 6
SDMA_PKT_COPY_T2T_BC_DW_6_src_mit_mode_mask = 0x00000007
SDMA_PKT_COPY_T2T_BC_DW_6_src_mit_mode_shift = 8
SDMA_PKT_COPY_T2T_BC_DW_6_SRC_MIT_MODE = lambda x: (((x) & SDMA_PKT_COPY_T2T_BC_DW_6_src_mit_mode_mask) << SDMA_PKT_COPY_T2T_BC_DW_6_src_mit_mode_shift)
SDMA_PKT_COPY_T2T_BC_DW_6_src_tilesplit_size_offset = 6
SDMA_PKT_COPY_T2T_BC_DW_6_src_tilesplit_size_mask = 0x00000007
SDMA_PKT_COPY_T2T_BC_DW_6_src_tilesplit_size_shift = 11
SDMA_PKT_COPY_T2T_BC_DW_6_SRC_TILESPLIT_SIZE = lambda x: (((x) & SDMA_PKT_COPY_T2T_BC_DW_6_src_tilesplit_size_mask) << SDMA_PKT_COPY_T2T_BC_DW_6_src_tilesplit_size_shift)
SDMA_PKT_COPY_T2T_BC_DW_6_src_bank_w_offset = 6
SDMA_PKT_COPY_T2T_BC_DW_6_src_bank_w_mask = 0x00000003
SDMA_PKT_COPY_T2T_BC_DW_6_src_bank_w_shift = 15
SDMA_PKT_COPY_T2T_BC_DW_6_SRC_BANK_W = lambda x: (((x) & SDMA_PKT_COPY_T2T_BC_DW_6_src_bank_w_mask) << SDMA_PKT_COPY_T2T_BC_DW_6_src_bank_w_shift)
SDMA_PKT_COPY_T2T_BC_DW_6_src_bank_h_offset = 6
SDMA_PKT_COPY_T2T_BC_DW_6_src_bank_h_mask = 0x00000003
SDMA_PKT_COPY_T2T_BC_DW_6_src_bank_h_shift = 18
SDMA_PKT_COPY_T2T_BC_DW_6_SRC_BANK_H = lambda x: (((x) & SDMA_PKT_COPY_T2T_BC_DW_6_src_bank_h_mask) << SDMA_PKT_COPY_T2T_BC_DW_6_src_bank_h_shift)
SDMA_PKT_COPY_T2T_BC_DW_6_src_num_bank_offset = 6
SDMA_PKT_COPY_T2T_BC_DW_6_src_num_bank_mask = 0x00000003
SDMA_PKT_COPY_T2T_BC_DW_6_src_num_bank_shift = 21
SDMA_PKT_COPY_T2T_BC_DW_6_SRC_NUM_BANK = lambda x: (((x) & SDMA_PKT_COPY_T2T_BC_DW_6_src_num_bank_mask) << SDMA_PKT_COPY_T2T_BC_DW_6_src_num_bank_shift)
SDMA_PKT_COPY_T2T_BC_DW_6_src_mat_aspt_offset = 6
SDMA_PKT_COPY_T2T_BC_DW_6_src_mat_aspt_mask = 0x00000003
SDMA_PKT_COPY_T2T_BC_DW_6_src_mat_aspt_shift = 24
SDMA_PKT_COPY_T2T_BC_DW_6_SRC_MAT_ASPT = lambda x: (((x) & SDMA_PKT_COPY_T2T_BC_DW_6_src_mat_aspt_mask) << SDMA_PKT_COPY_T2T_BC_DW_6_src_mat_aspt_shift)
SDMA_PKT_COPY_T2T_BC_DW_6_src_pipe_config_offset = 6
SDMA_PKT_COPY_T2T_BC_DW_6_src_pipe_config_mask = 0x0000001F
SDMA_PKT_COPY_T2T_BC_DW_6_src_pipe_config_shift = 26
SDMA_PKT_COPY_T2T_BC_DW_6_SRC_PIPE_CONFIG = lambda x: (((x) & SDMA_PKT_COPY_T2T_BC_DW_6_src_pipe_config_mask) << SDMA_PKT_COPY_T2T_BC_DW_6_src_pipe_config_shift)
SDMA_PKT_COPY_T2T_BC_DST_ADDR_LO_dst_addr_31_0_offset = 7
SDMA_PKT_COPY_T2T_BC_DST_ADDR_LO_dst_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_T2T_BC_DST_ADDR_LO_dst_addr_31_0_shift = 0
SDMA_PKT_COPY_T2T_BC_DST_ADDR_LO_DST_ADDR_31_0 = lambda x: (((x) & SDMA_PKT_COPY_T2T_BC_DST_ADDR_LO_dst_addr_31_0_mask) << SDMA_PKT_COPY_T2T_BC_DST_ADDR_LO_dst_addr_31_0_shift)
SDMA_PKT_COPY_T2T_BC_DST_ADDR_HI_dst_addr_63_32_offset = 8
SDMA_PKT_COPY_T2T_BC_DST_ADDR_HI_dst_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_T2T_BC_DST_ADDR_HI_dst_addr_63_32_shift = 0
SDMA_PKT_COPY_T2T_BC_DST_ADDR_HI_DST_ADDR_63_32 = lambda x: (((x) & SDMA_PKT_COPY_T2T_BC_DST_ADDR_HI_dst_addr_63_32_mask) << SDMA_PKT_COPY_T2T_BC_DST_ADDR_HI_dst_addr_63_32_shift)
SDMA_PKT_COPY_T2T_BC_DW_9_dst_x_offset = 9
SDMA_PKT_COPY_T2T_BC_DW_9_dst_x_mask = 0x00003FFF
SDMA_PKT_COPY_T2T_BC_DW_9_dst_x_shift = 0
SDMA_PKT_COPY_T2T_BC_DW_9_DST_X = lambda x: (((x) & SDMA_PKT_COPY_T2T_BC_DW_9_dst_x_mask) << SDMA_PKT_COPY_T2T_BC_DW_9_dst_x_shift)
SDMA_PKT_COPY_T2T_BC_DW_9_dst_y_offset = 9
SDMA_PKT_COPY_T2T_BC_DW_9_dst_y_mask = 0x00003FFF
SDMA_PKT_COPY_T2T_BC_DW_9_dst_y_shift = 16
SDMA_PKT_COPY_T2T_BC_DW_9_DST_Y = lambda x: (((x) & SDMA_PKT_COPY_T2T_BC_DW_9_dst_y_mask) << SDMA_PKT_COPY_T2T_BC_DW_9_dst_y_shift)
SDMA_PKT_COPY_T2T_BC_DW_10_dst_z_offset = 10
SDMA_PKT_COPY_T2T_BC_DW_10_dst_z_mask = 0x000007FF
SDMA_PKT_COPY_T2T_BC_DW_10_dst_z_shift = 0
SDMA_PKT_COPY_T2T_BC_DW_10_DST_Z = lambda x: (((x) & SDMA_PKT_COPY_T2T_BC_DW_10_dst_z_mask) << SDMA_PKT_COPY_T2T_BC_DW_10_dst_z_shift)
SDMA_PKT_COPY_T2T_BC_DW_10_dst_width_offset = 10
SDMA_PKT_COPY_T2T_BC_DW_10_dst_width_mask = 0x00003FFF
SDMA_PKT_COPY_T2T_BC_DW_10_dst_width_shift = 16
SDMA_PKT_COPY_T2T_BC_DW_10_DST_WIDTH = lambda x: (((x) & SDMA_PKT_COPY_T2T_BC_DW_10_dst_width_mask) << SDMA_PKT_COPY_T2T_BC_DW_10_dst_width_shift)
SDMA_PKT_COPY_T2T_BC_DW_11_dst_height_offset = 11
SDMA_PKT_COPY_T2T_BC_DW_11_dst_height_mask = 0x00003FFF
SDMA_PKT_COPY_T2T_BC_DW_11_dst_height_shift = 0
SDMA_PKT_COPY_T2T_BC_DW_11_DST_HEIGHT = lambda x: (((x) & SDMA_PKT_COPY_T2T_BC_DW_11_dst_height_mask) << SDMA_PKT_COPY_T2T_BC_DW_11_dst_height_shift)
SDMA_PKT_COPY_T2T_BC_DW_11_dst_depth_offset = 11
SDMA_PKT_COPY_T2T_BC_DW_11_dst_depth_mask = 0x00000FFF
SDMA_PKT_COPY_T2T_BC_DW_11_dst_depth_shift = 16
SDMA_PKT_COPY_T2T_BC_DW_11_DST_DEPTH = lambda x: (((x) & SDMA_PKT_COPY_T2T_BC_DW_11_dst_depth_mask) << SDMA_PKT_COPY_T2T_BC_DW_11_dst_depth_shift)
SDMA_PKT_COPY_T2T_BC_DW_12_dst_element_size_offset = 12
SDMA_PKT_COPY_T2T_BC_DW_12_dst_element_size_mask = 0x00000007
SDMA_PKT_COPY_T2T_BC_DW_12_dst_element_size_shift = 0
SDMA_PKT_COPY_T2T_BC_DW_12_DST_ELEMENT_SIZE = lambda x: (((x) & SDMA_PKT_COPY_T2T_BC_DW_12_dst_element_size_mask) << SDMA_PKT_COPY_T2T_BC_DW_12_dst_element_size_shift)
SDMA_PKT_COPY_T2T_BC_DW_12_dst_array_mode_offset = 12
SDMA_PKT_COPY_T2T_BC_DW_12_dst_array_mode_mask = 0x0000000F
SDMA_PKT_COPY_T2T_BC_DW_12_dst_array_mode_shift = 3
SDMA_PKT_COPY_T2T_BC_DW_12_DST_ARRAY_MODE = lambda x: (((x) & SDMA_PKT_COPY_T2T_BC_DW_12_dst_array_mode_mask) << SDMA_PKT_COPY_T2T_BC_DW_12_dst_array_mode_shift)
SDMA_PKT_COPY_T2T_BC_DW_12_dst_mit_mode_offset = 12
SDMA_PKT_COPY_T2T_BC_DW_12_dst_mit_mode_mask = 0x00000007
SDMA_PKT_COPY_T2T_BC_DW_12_dst_mit_mode_shift = 8
SDMA_PKT_COPY_T2T_BC_DW_12_DST_MIT_MODE = lambda x: (((x) & SDMA_PKT_COPY_T2T_BC_DW_12_dst_mit_mode_mask) << SDMA_PKT_COPY_T2T_BC_DW_12_dst_mit_mode_shift)
SDMA_PKT_COPY_T2T_BC_DW_12_dst_tilesplit_size_offset = 12
SDMA_PKT_COPY_T2T_BC_DW_12_dst_tilesplit_size_mask = 0x00000007
SDMA_PKT_COPY_T2T_BC_DW_12_dst_tilesplit_size_shift = 11
SDMA_PKT_COPY_T2T_BC_DW_12_DST_TILESPLIT_SIZE = lambda x: (((x) & SDMA_PKT_COPY_T2T_BC_DW_12_dst_tilesplit_size_mask) << SDMA_PKT_COPY_T2T_BC_DW_12_dst_tilesplit_size_shift)
SDMA_PKT_COPY_T2T_BC_DW_12_dst_bank_w_offset = 12
SDMA_PKT_COPY_T2T_BC_DW_12_dst_bank_w_mask = 0x00000003
SDMA_PKT_COPY_T2T_BC_DW_12_dst_bank_w_shift = 15
SDMA_PKT_COPY_T2T_BC_DW_12_DST_BANK_W = lambda x: (((x) & SDMA_PKT_COPY_T2T_BC_DW_12_dst_bank_w_mask) << SDMA_PKT_COPY_T2T_BC_DW_12_dst_bank_w_shift)
SDMA_PKT_COPY_T2T_BC_DW_12_dst_bank_h_offset = 12
SDMA_PKT_COPY_T2T_BC_DW_12_dst_bank_h_mask = 0x00000003
SDMA_PKT_COPY_T2T_BC_DW_12_dst_bank_h_shift = 18
SDMA_PKT_COPY_T2T_BC_DW_12_DST_BANK_H = lambda x: (((x) & SDMA_PKT_COPY_T2T_BC_DW_12_dst_bank_h_mask) << SDMA_PKT_COPY_T2T_BC_DW_12_dst_bank_h_shift)
SDMA_PKT_COPY_T2T_BC_DW_12_dst_num_bank_offset = 12
SDMA_PKT_COPY_T2T_BC_DW_12_dst_num_bank_mask = 0x00000003
SDMA_PKT_COPY_T2T_BC_DW_12_dst_num_bank_shift = 21
SDMA_PKT_COPY_T2T_BC_DW_12_DST_NUM_BANK = lambda x: (((x) & SDMA_PKT_COPY_T2T_BC_DW_12_dst_num_bank_mask) << SDMA_PKT_COPY_T2T_BC_DW_12_dst_num_bank_shift)
SDMA_PKT_COPY_T2T_BC_DW_12_dst_mat_aspt_offset = 12
SDMA_PKT_COPY_T2T_BC_DW_12_dst_mat_aspt_mask = 0x00000003
SDMA_PKT_COPY_T2T_BC_DW_12_dst_mat_aspt_shift = 24
SDMA_PKT_COPY_T2T_BC_DW_12_DST_MAT_ASPT = lambda x: (((x) & SDMA_PKT_COPY_T2T_BC_DW_12_dst_mat_aspt_mask) << SDMA_PKT_COPY_T2T_BC_DW_12_dst_mat_aspt_shift)
SDMA_PKT_COPY_T2T_BC_DW_12_dst_pipe_config_offset = 12
SDMA_PKT_COPY_T2T_BC_DW_12_dst_pipe_config_mask = 0x0000001F
SDMA_PKT_COPY_T2T_BC_DW_12_dst_pipe_config_shift = 26
SDMA_PKT_COPY_T2T_BC_DW_12_DST_PIPE_CONFIG = lambda x: (((x) & SDMA_PKT_COPY_T2T_BC_DW_12_dst_pipe_config_mask) << SDMA_PKT_COPY_T2T_BC_DW_12_dst_pipe_config_shift)
SDMA_PKT_COPY_T2T_BC_DW_13_rect_x_offset = 13
SDMA_PKT_COPY_T2T_BC_DW_13_rect_x_mask = 0x00003FFF
SDMA_PKT_COPY_T2T_BC_DW_13_rect_x_shift = 0
SDMA_PKT_COPY_T2T_BC_DW_13_RECT_X = lambda x: (((x) & SDMA_PKT_COPY_T2T_BC_DW_13_rect_x_mask) << SDMA_PKT_COPY_T2T_BC_DW_13_rect_x_shift)
SDMA_PKT_COPY_T2T_BC_DW_13_rect_y_offset = 13
SDMA_PKT_COPY_T2T_BC_DW_13_rect_y_mask = 0x00003FFF
SDMA_PKT_COPY_T2T_BC_DW_13_rect_y_shift = 16
SDMA_PKT_COPY_T2T_BC_DW_13_RECT_Y = lambda x: (((x) & SDMA_PKT_COPY_T2T_BC_DW_13_rect_y_mask) << SDMA_PKT_COPY_T2T_BC_DW_13_rect_y_shift)
SDMA_PKT_COPY_T2T_BC_DW_14_rect_z_offset = 14
SDMA_PKT_COPY_T2T_BC_DW_14_rect_z_mask = 0x000007FF
SDMA_PKT_COPY_T2T_BC_DW_14_rect_z_shift = 0
SDMA_PKT_COPY_T2T_BC_DW_14_RECT_Z = lambda x: (((x) & SDMA_PKT_COPY_T2T_BC_DW_14_rect_z_mask) << SDMA_PKT_COPY_T2T_BC_DW_14_rect_z_shift)
SDMA_PKT_COPY_T2T_BC_DW_14_dst_sw_offset = 14
SDMA_PKT_COPY_T2T_BC_DW_14_dst_sw_mask = 0x00000003
SDMA_PKT_COPY_T2T_BC_DW_14_dst_sw_shift = 16
SDMA_PKT_COPY_T2T_BC_DW_14_DST_SW = lambda x: (((x) & SDMA_PKT_COPY_T2T_BC_DW_14_dst_sw_mask) << SDMA_PKT_COPY_T2T_BC_DW_14_dst_sw_shift)
SDMA_PKT_COPY_T2T_BC_DW_14_src_sw_offset = 14
SDMA_PKT_COPY_T2T_BC_DW_14_src_sw_mask = 0x00000003
SDMA_PKT_COPY_T2T_BC_DW_14_src_sw_shift = 24
SDMA_PKT_COPY_T2T_BC_DW_14_SRC_SW = lambda x: (((x) & SDMA_PKT_COPY_T2T_BC_DW_14_src_sw_mask) << SDMA_PKT_COPY_T2T_BC_DW_14_src_sw_shift)
SDMA_PKT_COPY_TILED_SUBWIN_HEADER_op_offset = 0
SDMA_PKT_COPY_TILED_SUBWIN_HEADER_op_mask = 0x000000FF
SDMA_PKT_COPY_TILED_SUBWIN_HEADER_op_shift = 0
SDMA_PKT_COPY_TILED_SUBWIN_HEADER_OP = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_HEADER_op_mask) << SDMA_PKT_COPY_TILED_SUBWIN_HEADER_op_shift)
SDMA_PKT_COPY_TILED_SUBWIN_HEADER_sub_op_offset = 0
SDMA_PKT_COPY_TILED_SUBWIN_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_COPY_TILED_SUBWIN_HEADER_sub_op_shift = 8
SDMA_PKT_COPY_TILED_SUBWIN_HEADER_SUB_OP = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_HEADER_sub_op_mask) << SDMA_PKT_COPY_TILED_SUBWIN_HEADER_sub_op_shift)
SDMA_PKT_COPY_TILED_SUBWIN_HEADER_tmz_offset = 0
SDMA_PKT_COPY_TILED_SUBWIN_HEADER_tmz_mask = 0x00000001
SDMA_PKT_COPY_TILED_SUBWIN_HEADER_tmz_shift = 18
SDMA_PKT_COPY_TILED_SUBWIN_HEADER_TMZ = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_HEADER_tmz_mask) << SDMA_PKT_COPY_TILED_SUBWIN_HEADER_tmz_shift)
SDMA_PKT_COPY_TILED_SUBWIN_HEADER_dcc_offset = 0
SDMA_PKT_COPY_TILED_SUBWIN_HEADER_dcc_mask = 0x00000001
SDMA_PKT_COPY_TILED_SUBWIN_HEADER_dcc_shift = 19
SDMA_PKT_COPY_TILED_SUBWIN_HEADER_DCC = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_HEADER_dcc_mask) << SDMA_PKT_COPY_TILED_SUBWIN_HEADER_dcc_shift)
SDMA_PKT_COPY_TILED_SUBWIN_HEADER_cpv_offset = 0
SDMA_PKT_COPY_TILED_SUBWIN_HEADER_cpv_mask = 0x00000001
SDMA_PKT_COPY_TILED_SUBWIN_HEADER_cpv_shift = 28
SDMA_PKT_COPY_TILED_SUBWIN_HEADER_CPV = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_HEADER_cpv_mask) << SDMA_PKT_COPY_TILED_SUBWIN_HEADER_cpv_shift)
SDMA_PKT_COPY_TILED_SUBWIN_HEADER_detile_offset = 0
SDMA_PKT_COPY_TILED_SUBWIN_HEADER_detile_mask = 0x00000001
SDMA_PKT_COPY_TILED_SUBWIN_HEADER_detile_shift = 31
SDMA_PKT_COPY_TILED_SUBWIN_HEADER_DETILE = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_HEADER_detile_mask) << SDMA_PKT_COPY_TILED_SUBWIN_HEADER_detile_shift)
SDMA_PKT_COPY_TILED_SUBWIN_TILED_ADDR_LO_tiled_addr_31_0_offset = 1
SDMA_PKT_COPY_TILED_SUBWIN_TILED_ADDR_LO_tiled_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_TILED_SUBWIN_TILED_ADDR_LO_tiled_addr_31_0_shift = 0
SDMA_PKT_COPY_TILED_SUBWIN_TILED_ADDR_LO_TILED_ADDR_31_0 = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_TILED_ADDR_LO_tiled_addr_31_0_mask) << SDMA_PKT_COPY_TILED_SUBWIN_TILED_ADDR_LO_tiled_addr_31_0_shift)
SDMA_PKT_COPY_TILED_SUBWIN_TILED_ADDR_HI_tiled_addr_63_32_offset = 2
SDMA_PKT_COPY_TILED_SUBWIN_TILED_ADDR_HI_tiled_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_TILED_SUBWIN_TILED_ADDR_HI_tiled_addr_63_32_shift = 0
SDMA_PKT_COPY_TILED_SUBWIN_TILED_ADDR_HI_TILED_ADDR_63_32 = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_TILED_ADDR_HI_tiled_addr_63_32_mask) << SDMA_PKT_COPY_TILED_SUBWIN_TILED_ADDR_HI_tiled_addr_63_32_shift)
SDMA_PKT_COPY_TILED_SUBWIN_DW_3_tiled_x_offset = 3
SDMA_PKT_COPY_TILED_SUBWIN_DW_3_tiled_x_mask = 0x00003FFF
SDMA_PKT_COPY_TILED_SUBWIN_DW_3_tiled_x_shift = 0
SDMA_PKT_COPY_TILED_SUBWIN_DW_3_TILED_X = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_DW_3_tiled_x_mask) << SDMA_PKT_COPY_TILED_SUBWIN_DW_3_tiled_x_shift)
SDMA_PKT_COPY_TILED_SUBWIN_DW_3_tiled_y_offset = 3
SDMA_PKT_COPY_TILED_SUBWIN_DW_3_tiled_y_mask = 0x00003FFF
SDMA_PKT_COPY_TILED_SUBWIN_DW_3_tiled_y_shift = 16
SDMA_PKT_COPY_TILED_SUBWIN_DW_3_TILED_Y = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_DW_3_tiled_y_mask) << SDMA_PKT_COPY_TILED_SUBWIN_DW_3_tiled_y_shift)
SDMA_PKT_COPY_TILED_SUBWIN_DW_4_tiled_z_offset = 4
SDMA_PKT_COPY_TILED_SUBWIN_DW_4_tiled_z_mask = 0x00001FFF
SDMA_PKT_COPY_TILED_SUBWIN_DW_4_tiled_z_shift = 0
SDMA_PKT_COPY_TILED_SUBWIN_DW_4_TILED_Z = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_DW_4_tiled_z_mask) << SDMA_PKT_COPY_TILED_SUBWIN_DW_4_tiled_z_shift)
SDMA_PKT_COPY_TILED_SUBWIN_DW_4_width_offset = 4
SDMA_PKT_COPY_TILED_SUBWIN_DW_4_width_mask = 0x00003FFF
SDMA_PKT_COPY_TILED_SUBWIN_DW_4_width_shift = 16
SDMA_PKT_COPY_TILED_SUBWIN_DW_4_WIDTH = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_DW_4_width_mask) << SDMA_PKT_COPY_TILED_SUBWIN_DW_4_width_shift)
SDMA_PKT_COPY_TILED_SUBWIN_DW_5_height_offset = 5
SDMA_PKT_COPY_TILED_SUBWIN_DW_5_height_mask = 0x00003FFF
SDMA_PKT_COPY_TILED_SUBWIN_DW_5_height_shift = 0
SDMA_PKT_COPY_TILED_SUBWIN_DW_5_HEIGHT = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_DW_5_height_mask) << SDMA_PKT_COPY_TILED_SUBWIN_DW_5_height_shift)
SDMA_PKT_COPY_TILED_SUBWIN_DW_5_depth_offset = 5
SDMA_PKT_COPY_TILED_SUBWIN_DW_5_depth_mask = 0x00001FFF
SDMA_PKT_COPY_TILED_SUBWIN_DW_5_depth_shift = 16
SDMA_PKT_COPY_TILED_SUBWIN_DW_5_DEPTH = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_DW_5_depth_mask) << SDMA_PKT_COPY_TILED_SUBWIN_DW_5_depth_shift)
SDMA_PKT_COPY_TILED_SUBWIN_DW_6_element_size_offset = 6
SDMA_PKT_COPY_TILED_SUBWIN_DW_6_element_size_mask = 0x00000007
SDMA_PKT_COPY_TILED_SUBWIN_DW_6_element_size_shift = 0
SDMA_PKT_COPY_TILED_SUBWIN_DW_6_ELEMENT_SIZE = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_DW_6_element_size_mask) << SDMA_PKT_COPY_TILED_SUBWIN_DW_6_element_size_shift)
SDMA_PKT_COPY_TILED_SUBWIN_DW_6_swizzle_mode_offset = 6
SDMA_PKT_COPY_TILED_SUBWIN_DW_6_swizzle_mode_mask = 0x0000001F
SDMA_PKT_COPY_TILED_SUBWIN_DW_6_swizzle_mode_shift = 3
SDMA_PKT_COPY_TILED_SUBWIN_DW_6_SWIZZLE_MODE = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_DW_6_swizzle_mode_mask) << SDMA_PKT_COPY_TILED_SUBWIN_DW_6_swizzle_mode_shift)
SDMA_PKT_COPY_TILED_SUBWIN_DW_6_dimension_offset = 6
SDMA_PKT_COPY_TILED_SUBWIN_DW_6_dimension_mask = 0x00000003
SDMA_PKT_COPY_TILED_SUBWIN_DW_6_dimension_shift = 9
SDMA_PKT_COPY_TILED_SUBWIN_DW_6_DIMENSION = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_DW_6_dimension_mask) << SDMA_PKT_COPY_TILED_SUBWIN_DW_6_dimension_shift)
SDMA_PKT_COPY_TILED_SUBWIN_DW_6_mip_max_offset = 6
SDMA_PKT_COPY_TILED_SUBWIN_DW_6_mip_max_mask = 0x0000000F
SDMA_PKT_COPY_TILED_SUBWIN_DW_6_mip_max_shift = 16
SDMA_PKT_COPY_TILED_SUBWIN_DW_6_MIP_MAX = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_DW_6_mip_max_mask) << SDMA_PKT_COPY_TILED_SUBWIN_DW_6_mip_max_shift)
SDMA_PKT_COPY_TILED_SUBWIN_DW_6_mip_id_offset = 6
SDMA_PKT_COPY_TILED_SUBWIN_DW_6_mip_id_mask = 0x0000000F
SDMA_PKT_COPY_TILED_SUBWIN_DW_6_mip_id_shift = 20
SDMA_PKT_COPY_TILED_SUBWIN_DW_6_MIP_ID = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_DW_6_mip_id_mask) << SDMA_PKT_COPY_TILED_SUBWIN_DW_6_mip_id_shift)
SDMA_PKT_COPY_TILED_SUBWIN_LINEAR_ADDR_LO_linear_addr_31_0_offset = 7
SDMA_PKT_COPY_TILED_SUBWIN_LINEAR_ADDR_LO_linear_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_TILED_SUBWIN_LINEAR_ADDR_LO_linear_addr_31_0_shift = 0
SDMA_PKT_COPY_TILED_SUBWIN_LINEAR_ADDR_LO_LINEAR_ADDR_31_0 = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_LINEAR_ADDR_LO_linear_addr_31_0_mask) << SDMA_PKT_COPY_TILED_SUBWIN_LINEAR_ADDR_LO_linear_addr_31_0_shift)
SDMA_PKT_COPY_TILED_SUBWIN_LINEAR_ADDR_HI_linear_addr_63_32_offset = 8
SDMA_PKT_COPY_TILED_SUBWIN_LINEAR_ADDR_HI_linear_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_TILED_SUBWIN_LINEAR_ADDR_HI_linear_addr_63_32_shift = 0
SDMA_PKT_COPY_TILED_SUBWIN_LINEAR_ADDR_HI_LINEAR_ADDR_63_32 = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_LINEAR_ADDR_HI_linear_addr_63_32_mask) << SDMA_PKT_COPY_TILED_SUBWIN_LINEAR_ADDR_HI_linear_addr_63_32_shift)
SDMA_PKT_COPY_TILED_SUBWIN_DW_9_linear_x_offset = 9
SDMA_PKT_COPY_TILED_SUBWIN_DW_9_linear_x_mask = 0x00003FFF
SDMA_PKT_COPY_TILED_SUBWIN_DW_9_linear_x_shift = 0
SDMA_PKT_COPY_TILED_SUBWIN_DW_9_LINEAR_X = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_DW_9_linear_x_mask) << SDMA_PKT_COPY_TILED_SUBWIN_DW_9_linear_x_shift)
SDMA_PKT_COPY_TILED_SUBWIN_DW_9_linear_y_offset = 9
SDMA_PKT_COPY_TILED_SUBWIN_DW_9_linear_y_mask = 0x00003FFF
SDMA_PKT_COPY_TILED_SUBWIN_DW_9_linear_y_shift = 16
SDMA_PKT_COPY_TILED_SUBWIN_DW_9_LINEAR_Y = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_DW_9_linear_y_mask) << SDMA_PKT_COPY_TILED_SUBWIN_DW_9_linear_y_shift)
SDMA_PKT_COPY_TILED_SUBWIN_DW_10_linear_z_offset = 10
SDMA_PKT_COPY_TILED_SUBWIN_DW_10_linear_z_mask = 0x00001FFF
SDMA_PKT_COPY_TILED_SUBWIN_DW_10_linear_z_shift = 0
SDMA_PKT_COPY_TILED_SUBWIN_DW_10_LINEAR_Z = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_DW_10_linear_z_mask) << SDMA_PKT_COPY_TILED_SUBWIN_DW_10_linear_z_shift)
SDMA_PKT_COPY_TILED_SUBWIN_DW_10_linear_pitch_offset = 10
SDMA_PKT_COPY_TILED_SUBWIN_DW_10_linear_pitch_mask = 0x00003FFF
SDMA_PKT_COPY_TILED_SUBWIN_DW_10_linear_pitch_shift = 16
SDMA_PKT_COPY_TILED_SUBWIN_DW_10_LINEAR_PITCH = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_DW_10_linear_pitch_mask) << SDMA_PKT_COPY_TILED_SUBWIN_DW_10_linear_pitch_shift)
SDMA_PKT_COPY_TILED_SUBWIN_DW_11_linear_slice_pitch_offset = 11
SDMA_PKT_COPY_TILED_SUBWIN_DW_11_linear_slice_pitch_mask = 0x0FFFFFFF
SDMA_PKT_COPY_TILED_SUBWIN_DW_11_linear_slice_pitch_shift = 0
SDMA_PKT_COPY_TILED_SUBWIN_DW_11_LINEAR_SLICE_PITCH = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_DW_11_linear_slice_pitch_mask) << SDMA_PKT_COPY_TILED_SUBWIN_DW_11_linear_slice_pitch_shift)
SDMA_PKT_COPY_TILED_SUBWIN_DW_12_rect_x_offset = 12
SDMA_PKT_COPY_TILED_SUBWIN_DW_12_rect_x_mask = 0x00003FFF
SDMA_PKT_COPY_TILED_SUBWIN_DW_12_rect_x_shift = 0
SDMA_PKT_COPY_TILED_SUBWIN_DW_12_RECT_X = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_DW_12_rect_x_mask) << SDMA_PKT_COPY_TILED_SUBWIN_DW_12_rect_x_shift)
SDMA_PKT_COPY_TILED_SUBWIN_DW_12_rect_y_offset = 12
SDMA_PKT_COPY_TILED_SUBWIN_DW_12_rect_y_mask = 0x00003FFF
SDMA_PKT_COPY_TILED_SUBWIN_DW_12_rect_y_shift = 16
SDMA_PKT_COPY_TILED_SUBWIN_DW_12_RECT_Y = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_DW_12_rect_y_mask) << SDMA_PKT_COPY_TILED_SUBWIN_DW_12_rect_y_shift)
SDMA_PKT_COPY_TILED_SUBWIN_DW_13_rect_z_offset = 13
SDMA_PKT_COPY_TILED_SUBWIN_DW_13_rect_z_mask = 0x00001FFF
SDMA_PKT_COPY_TILED_SUBWIN_DW_13_rect_z_shift = 0
SDMA_PKT_COPY_TILED_SUBWIN_DW_13_RECT_Z = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_DW_13_rect_z_mask) << SDMA_PKT_COPY_TILED_SUBWIN_DW_13_rect_z_shift)
SDMA_PKT_COPY_TILED_SUBWIN_DW_13_linear_sw_offset = 13
SDMA_PKT_COPY_TILED_SUBWIN_DW_13_linear_sw_mask = 0x00000003
SDMA_PKT_COPY_TILED_SUBWIN_DW_13_linear_sw_shift = 16
SDMA_PKT_COPY_TILED_SUBWIN_DW_13_LINEAR_SW = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_DW_13_linear_sw_mask) << SDMA_PKT_COPY_TILED_SUBWIN_DW_13_linear_sw_shift)
SDMA_PKT_COPY_TILED_SUBWIN_DW_13_linear_cache_policy_offset = 13
SDMA_PKT_COPY_TILED_SUBWIN_DW_13_linear_cache_policy_mask = 0x00000007
SDMA_PKT_COPY_TILED_SUBWIN_DW_13_linear_cache_policy_shift = 18
SDMA_PKT_COPY_TILED_SUBWIN_DW_13_LINEAR_CACHE_POLICY = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_DW_13_linear_cache_policy_mask) << SDMA_PKT_COPY_TILED_SUBWIN_DW_13_linear_cache_policy_shift)
SDMA_PKT_COPY_TILED_SUBWIN_DW_13_tile_sw_offset = 13
SDMA_PKT_COPY_TILED_SUBWIN_DW_13_tile_sw_mask = 0x00000003
SDMA_PKT_COPY_TILED_SUBWIN_DW_13_tile_sw_shift = 24
SDMA_PKT_COPY_TILED_SUBWIN_DW_13_TILE_SW = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_DW_13_tile_sw_mask) << SDMA_PKT_COPY_TILED_SUBWIN_DW_13_tile_sw_shift)
SDMA_PKT_COPY_TILED_SUBWIN_DW_13_tile_cache_policy_offset = 13
SDMA_PKT_COPY_TILED_SUBWIN_DW_13_tile_cache_policy_mask = 0x00000007
SDMA_PKT_COPY_TILED_SUBWIN_DW_13_tile_cache_policy_shift = 26
SDMA_PKT_COPY_TILED_SUBWIN_DW_13_TILE_CACHE_POLICY = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_DW_13_tile_cache_policy_mask) << SDMA_PKT_COPY_TILED_SUBWIN_DW_13_tile_cache_policy_shift)
SDMA_PKT_COPY_TILED_SUBWIN_META_ADDR_LO_meta_addr_31_0_offset = 14
SDMA_PKT_COPY_TILED_SUBWIN_META_ADDR_LO_meta_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_TILED_SUBWIN_META_ADDR_LO_meta_addr_31_0_shift = 0
SDMA_PKT_COPY_TILED_SUBWIN_META_ADDR_LO_META_ADDR_31_0 = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_META_ADDR_LO_meta_addr_31_0_mask) << SDMA_PKT_COPY_TILED_SUBWIN_META_ADDR_LO_meta_addr_31_0_shift)
SDMA_PKT_COPY_TILED_SUBWIN_META_ADDR_HI_meta_addr_63_32_offset = 15
SDMA_PKT_COPY_TILED_SUBWIN_META_ADDR_HI_meta_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_TILED_SUBWIN_META_ADDR_HI_meta_addr_63_32_shift = 0
SDMA_PKT_COPY_TILED_SUBWIN_META_ADDR_HI_META_ADDR_63_32 = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_META_ADDR_HI_meta_addr_63_32_mask) << SDMA_PKT_COPY_TILED_SUBWIN_META_ADDR_HI_meta_addr_63_32_shift)
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_data_format_offset = 16
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_data_format_mask = 0x0000007F
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_data_format_shift = 0
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_DATA_FORMAT = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_data_format_mask) << SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_data_format_shift)
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_color_transform_disable_offset = 16
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_color_transform_disable_mask = 0x00000001
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_color_transform_disable_shift = 7
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_COLOR_TRANSFORM_DISABLE = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_color_transform_disable_mask) << SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_color_transform_disable_shift)
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_alpha_is_on_msb_offset = 16
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_alpha_is_on_msb_mask = 0x00000001
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_alpha_is_on_msb_shift = 8
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_ALPHA_IS_ON_MSB = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_alpha_is_on_msb_mask) << SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_alpha_is_on_msb_shift)
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_number_type_offset = 16
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_number_type_mask = 0x00000007
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_number_type_shift = 9
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_NUMBER_TYPE = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_number_type_mask) << SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_number_type_shift)
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_surface_type_offset = 16
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_surface_type_mask = 0x00000003
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_surface_type_shift = 12
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_SURFACE_TYPE = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_surface_type_mask) << SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_surface_type_shift)
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_meta_llc_offset = 16
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_meta_llc_mask = 0x00000001
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_meta_llc_shift = 14
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_META_LLC = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_meta_llc_mask) << SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_meta_llc_shift)
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_max_comp_block_size_offset = 16
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_max_comp_block_size_mask = 0x00000003
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_max_comp_block_size_shift = 24
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_MAX_COMP_BLOCK_SIZE = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_max_comp_block_size_mask) << SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_max_comp_block_size_shift)
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_max_uncomp_block_size_offset = 16
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_max_uncomp_block_size_mask = 0x00000003
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_max_uncomp_block_size_shift = 26
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_MAX_UNCOMP_BLOCK_SIZE = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_max_uncomp_block_size_mask) << SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_max_uncomp_block_size_shift)
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_write_compress_enable_offset = 16
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_write_compress_enable_mask = 0x00000001
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_write_compress_enable_shift = 28
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_WRITE_COMPRESS_ENABLE = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_write_compress_enable_mask) << SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_write_compress_enable_shift)
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_meta_tmz_offset = 16
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_meta_tmz_mask = 0x00000001
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_meta_tmz_shift = 29
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_META_TMZ = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_meta_tmz_mask) << SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_meta_tmz_shift)
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_pipe_aligned_offset = 16
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_pipe_aligned_mask = 0x00000001
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_pipe_aligned_shift = 31
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_PIPE_ALIGNED = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_pipe_aligned_mask) << SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_pipe_aligned_shift)
SDMA_PKT_COPY_TILED_SUBWIN_BC_HEADER_op_offset = 0
SDMA_PKT_COPY_TILED_SUBWIN_BC_HEADER_op_mask = 0x000000FF
SDMA_PKT_COPY_TILED_SUBWIN_BC_HEADER_op_shift = 0
SDMA_PKT_COPY_TILED_SUBWIN_BC_HEADER_OP = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_BC_HEADER_op_mask) << SDMA_PKT_COPY_TILED_SUBWIN_BC_HEADER_op_shift)
SDMA_PKT_COPY_TILED_SUBWIN_BC_HEADER_sub_op_offset = 0
SDMA_PKT_COPY_TILED_SUBWIN_BC_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_COPY_TILED_SUBWIN_BC_HEADER_sub_op_shift = 8
SDMA_PKT_COPY_TILED_SUBWIN_BC_HEADER_SUB_OP = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_BC_HEADER_sub_op_mask) << SDMA_PKT_COPY_TILED_SUBWIN_BC_HEADER_sub_op_shift)
SDMA_PKT_COPY_TILED_SUBWIN_BC_HEADER_detile_offset = 0
SDMA_PKT_COPY_TILED_SUBWIN_BC_HEADER_detile_mask = 0x00000001
SDMA_PKT_COPY_TILED_SUBWIN_BC_HEADER_detile_shift = 31
SDMA_PKT_COPY_TILED_SUBWIN_BC_HEADER_DETILE = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_BC_HEADER_detile_mask) << SDMA_PKT_COPY_TILED_SUBWIN_BC_HEADER_detile_shift)
SDMA_PKT_COPY_TILED_SUBWIN_BC_TILED_ADDR_LO_tiled_addr_31_0_offset = 1
SDMA_PKT_COPY_TILED_SUBWIN_BC_TILED_ADDR_LO_tiled_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_TILED_SUBWIN_BC_TILED_ADDR_LO_tiled_addr_31_0_shift = 0
SDMA_PKT_COPY_TILED_SUBWIN_BC_TILED_ADDR_LO_TILED_ADDR_31_0 = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_BC_TILED_ADDR_LO_tiled_addr_31_0_mask) << SDMA_PKT_COPY_TILED_SUBWIN_BC_TILED_ADDR_LO_tiled_addr_31_0_shift)
SDMA_PKT_COPY_TILED_SUBWIN_BC_TILED_ADDR_HI_tiled_addr_63_32_offset = 2
SDMA_PKT_COPY_TILED_SUBWIN_BC_TILED_ADDR_HI_tiled_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_TILED_SUBWIN_BC_TILED_ADDR_HI_tiled_addr_63_32_shift = 0
SDMA_PKT_COPY_TILED_SUBWIN_BC_TILED_ADDR_HI_TILED_ADDR_63_32 = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_BC_TILED_ADDR_HI_tiled_addr_63_32_mask) << SDMA_PKT_COPY_TILED_SUBWIN_BC_TILED_ADDR_HI_tiled_addr_63_32_shift)
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_3_tiled_x_offset = 3
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_3_tiled_x_mask = 0x00003FFF
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_3_tiled_x_shift = 0
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_3_TILED_X = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_3_tiled_x_mask) << SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_3_tiled_x_shift)
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_3_tiled_y_offset = 3
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_3_tiled_y_mask = 0x00003FFF
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_3_tiled_y_shift = 16
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_3_TILED_Y = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_3_tiled_y_mask) << SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_3_tiled_y_shift)
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_4_tiled_z_offset = 4
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_4_tiled_z_mask = 0x000007FF
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_4_tiled_z_shift = 0
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_4_TILED_Z = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_4_tiled_z_mask) << SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_4_tiled_z_shift)
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_4_width_offset = 4
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_4_width_mask = 0x00003FFF
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_4_width_shift = 16
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_4_WIDTH = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_4_width_mask) << SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_4_width_shift)
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_5_height_offset = 5
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_5_height_mask = 0x00003FFF
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_5_height_shift = 0
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_5_HEIGHT = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_5_height_mask) << SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_5_height_shift)
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_5_depth_offset = 5
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_5_depth_mask = 0x000007FF
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_5_depth_shift = 16
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_5_DEPTH = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_5_depth_mask) << SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_5_depth_shift)
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_element_size_offset = 6
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_element_size_mask = 0x00000007
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_element_size_shift = 0
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_ELEMENT_SIZE = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_element_size_mask) << SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_element_size_shift)
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_array_mode_offset = 6
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_array_mode_mask = 0x0000000F
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_array_mode_shift = 3
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_ARRAY_MODE = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_array_mode_mask) << SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_array_mode_shift)
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_mit_mode_offset = 6
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_mit_mode_mask = 0x00000007
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_mit_mode_shift = 8
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_MIT_MODE = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_mit_mode_mask) << SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_mit_mode_shift)
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_tilesplit_size_offset = 6
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_tilesplit_size_mask = 0x00000007
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_tilesplit_size_shift = 11
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_TILESPLIT_SIZE = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_tilesplit_size_mask) << SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_tilesplit_size_shift)
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_bank_w_offset = 6
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_bank_w_mask = 0x00000003
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_bank_w_shift = 15
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_BANK_W = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_bank_w_mask) << SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_bank_w_shift)
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_bank_h_offset = 6
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_bank_h_mask = 0x00000003
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_bank_h_shift = 18
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_BANK_H = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_bank_h_mask) << SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_bank_h_shift)
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_num_bank_offset = 6
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_num_bank_mask = 0x00000003
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_num_bank_shift = 21
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_NUM_BANK = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_num_bank_mask) << SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_num_bank_shift)
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_mat_aspt_offset = 6
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_mat_aspt_mask = 0x00000003
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_mat_aspt_shift = 24
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_MAT_ASPT = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_mat_aspt_mask) << SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_mat_aspt_shift)
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_pipe_config_offset = 6
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_pipe_config_mask = 0x0000001F
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_pipe_config_shift = 26
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_PIPE_CONFIG = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_pipe_config_mask) << SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_pipe_config_shift)
SDMA_PKT_COPY_TILED_SUBWIN_BC_LINEAR_ADDR_LO_linear_addr_31_0_offset = 7
SDMA_PKT_COPY_TILED_SUBWIN_BC_LINEAR_ADDR_LO_linear_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_TILED_SUBWIN_BC_LINEAR_ADDR_LO_linear_addr_31_0_shift = 0
SDMA_PKT_COPY_TILED_SUBWIN_BC_LINEAR_ADDR_LO_LINEAR_ADDR_31_0 = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_BC_LINEAR_ADDR_LO_linear_addr_31_0_mask) << SDMA_PKT_COPY_TILED_SUBWIN_BC_LINEAR_ADDR_LO_linear_addr_31_0_shift)
SDMA_PKT_COPY_TILED_SUBWIN_BC_LINEAR_ADDR_HI_linear_addr_63_32_offset = 8
SDMA_PKT_COPY_TILED_SUBWIN_BC_LINEAR_ADDR_HI_linear_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_TILED_SUBWIN_BC_LINEAR_ADDR_HI_linear_addr_63_32_shift = 0
SDMA_PKT_COPY_TILED_SUBWIN_BC_LINEAR_ADDR_HI_LINEAR_ADDR_63_32 = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_BC_LINEAR_ADDR_HI_linear_addr_63_32_mask) << SDMA_PKT_COPY_TILED_SUBWIN_BC_LINEAR_ADDR_HI_linear_addr_63_32_shift)
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_9_linear_x_offset = 9
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_9_linear_x_mask = 0x00003FFF
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_9_linear_x_shift = 0
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_9_LINEAR_X = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_9_linear_x_mask) << SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_9_linear_x_shift)
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_9_linear_y_offset = 9
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_9_linear_y_mask = 0x00003FFF
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_9_linear_y_shift = 16
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_9_LINEAR_Y = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_9_linear_y_mask) << SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_9_linear_y_shift)
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_10_linear_z_offset = 10
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_10_linear_z_mask = 0x000007FF
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_10_linear_z_shift = 0
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_10_LINEAR_Z = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_10_linear_z_mask) << SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_10_linear_z_shift)
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_10_linear_pitch_offset = 10
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_10_linear_pitch_mask = 0x00003FFF
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_10_linear_pitch_shift = 16
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_10_LINEAR_PITCH = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_10_linear_pitch_mask) << SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_10_linear_pitch_shift)
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_11_linear_slice_pitch_offset = 11
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_11_linear_slice_pitch_mask = 0x0FFFFFFF
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_11_linear_slice_pitch_shift = 0
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_11_LINEAR_SLICE_PITCH = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_11_linear_slice_pitch_mask) << SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_11_linear_slice_pitch_shift)
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_12_rect_x_offset = 12
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_12_rect_x_mask = 0x00003FFF
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_12_rect_x_shift = 0
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_12_RECT_X = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_12_rect_x_mask) << SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_12_rect_x_shift)
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_12_rect_y_offset = 12
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_12_rect_y_mask = 0x00003FFF
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_12_rect_y_shift = 16
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_12_RECT_Y = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_12_rect_y_mask) << SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_12_rect_y_shift)
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_13_rect_z_offset = 13
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_13_rect_z_mask = 0x000007FF
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_13_rect_z_shift = 0
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_13_RECT_Z = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_13_rect_z_mask) << SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_13_rect_z_shift)
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_13_linear_sw_offset = 13
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_13_linear_sw_mask = 0x00000003
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_13_linear_sw_shift = 16
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_13_LINEAR_SW = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_13_linear_sw_mask) << SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_13_linear_sw_shift)
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_13_tile_sw_offset = 13
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_13_tile_sw_mask = 0x00000003
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_13_tile_sw_shift = 24
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_13_TILE_SW = lambda x: (((x) & SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_13_tile_sw_mask) << SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_13_tile_sw_shift)
SDMA_PKT_COPY_STRUCT_HEADER_op_offset = 0
SDMA_PKT_COPY_STRUCT_HEADER_op_mask = 0x000000FF
SDMA_PKT_COPY_STRUCT_HEADER_op_shift = 0
SDMA_PKT_COPY_STRUCT_HEADER_OP = lambda x: (((x) & SDMA_PKT_COPY_STRUCT_HEADER_op_mask) << SDMA_PKT_COPY_STRUCT_HEADER_op_shift)
SDMA_PKT_COPY_STRUCT_HEADER_sub_op_offset = 0
SDMA_PKT_COPY_STRUCT_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_COPY_STRUCT_HEADER_sub_op_shift = 8
SDMA_PKT_COPY_STRUCT_HEADER_SUB_OP = lambda x: (((x) & SDMA_PKT_COPY_STRUCT_HEADER_sub_op_mask) << SDMA_PKT_COPY_STRUCT_HEADER_sub_op_shift)
SDMA_PKT_COPY_STRUCT_HEADER_tmz_offset = 0
SDMA_PKT_COPY_STRUCT_HEADER_tmz_mask = 0x00000001
SDMA_PKT_COPY_STRUCT_HEADER_tmz_shift = 18
SDMA_PKT_COPY_STRUCT_HEADER_TMZ = lambda x: (((x) & SDMA_PKT_COPY_STRUCT_HEADER_tmz_mask) << SDMA_PKT_COPY_STRUCT_HEADER_tmz_shift)
SDMA_PKT_COPY_STRUCT_HEADER_cpv_offset = 0
SDMA_PKT_COPY_STRUCT_HEADER_cpv_mask = 0x00000001
SDMA_PKT_COPY_STRUCT_HEADER_cpv_shift = 28
SDMA_PKT_COPY_STRUCT_HEADER_CPV = lambda x: (((x) & SDMA_PKT_COPY_STRUCT_HEADER_cpv_mask) << SDMA_PKT_COPY_STRUCT_HEADER_cpv_shift)
SDMA_PKT_COPY_STRUCT_HEADER_detile_offset = 0
SDMA_PKT_COPY_STRUCT_HEADER_detile_mask = 0x00000001
SDMA_PKT_COPY_STRUCT_HEADER_detile_shift = 31
SDMA_PKT_COPY_STRUCT_HEADER_DETILE = lambda x: (((x) & SDMA_PKT_COPY_STRUCT_HEADER_detile_mask) << SDMA_PKT_COPY_STRUCT_HEADER_detile_shift)
SDMA_PKT_COPY_STRUCT_SB_ADDR_LO_sb_addr_31_0_offset = 1
SDMA_PKT_COPY_STRUCT_SB_ADDR_LO_sb_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_STRUCT_SB_ADDR_LO_sb_addr_31_0_shift = 0
SDMA_PKT_COPY_STRUCT_SB_ADDR_LO_SB_ADDR_31_0 = lambda x: (((x) & SDMA_PKT_COPY_STRUCT_SB_ADDR_LO_sb_addr_31_0_mask) << SDMA_PKT_COPY_STRUCT_SB_ADDR_LO_sb_addr_31_0_shift)
SDMA_PKT_COPY_STRUCT_SB_ADDR_HI_sb_addr_63_32_offset = 2
SDMA_PKT_COPY_STRUCT_SB_ADDR_HI_sb_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_STRUCT_SB_ADDR_HI_sb_addr_63_32_shift = 0
SDMA_PKT_COPY_STRUCT_SB_ADDR_HI_SB_ADDR_63_32 = lambda x: (((x) & SDMA_PKT_COPY_STRUCT_SB_ADDR_HI_sb_addr_63_32_mask) << SDMA_PKT_COPY_STRUCT_SB_ADDR_HI_sb_addr_63_32_shift)
SDMA_PKT_COPY_STRUCT_START_INDEX_start_index_offset = 3
SDMA_PKT_COPY_STRUCT_START_INDEX_start_index_mask = 0xFFFFFFFF
SDMA_PKT_COPY_STRUCT_START_INDEX_start_index_shift = 0
SDMA_PKT_COPY_STRUCT_START_INDEX_START_INDEX = lambda x: (((x) & SDMA_PKT_COPY_STRUCT_START_INDEX_start_index_mask) << SDMA_PKT_COPY_STRUCT_START_INDEX_start_index_shift)
SDMA_PKT_COPY_STRUCT_COUNT_count_offset = 4
SDMA_PKT_COPY_STRUCT_COUNT_count_mask = 0xFFFFFFFF
SDMA_PKT_COPY_STRUCT_COUNT_count_shift = 0
SDMA_PKT_COPY_STRUCT_COUNT_COUNT = lambda x: (((x) & SDMA_PKT_COPY_STRUCT_COUNT_count_mask) << SDMA_PKT_COPY_STRUCT_COUNT_count_shift)
SDMA_PKT_COPY_STRUCT_DW_5_stride_offset = 5
SDMA_PKT_COPY_STRUCT_DW_5_stride_mask = 0x000007FF
SDMA_PKT_COPY_STRUCT_DW_5_stride_shift = 0
SDMA_PKT_COPY_STRUCT_DW_5_STRIDE = lambda x: (((x) & SDMA_PKT_COPY_STRUCT_DW_5_stride_mask) << SDMA_PKT_COPY_STRUCT_DW_5_stride_shift)
SDMA_PKT_COPY_STRUCT_DW_5_linear_sw_offset = 5
SDMA_PKT_COPY_STRUCT_DW_5_linear_sw_mask = 0x00000003
SDMA_PKT_COPY_STRUCT_DW_5_linear_sw_shift = 16
SDMA_PKT_COPY_STRUCT_DW_5_LINEAR_SW = lambda x: (((x) & SDMA_PKT_COPY_STRUCT_DW_5_linear_sw_mask) << SDMA_PKT_COPY_STRUCT_DW_5_linear_sw_shift)
SDMA_PKT_COPY_STRUCT_DW_5_linear_cache_policy_offset = 5
SDMA_PKT_COPY_STRUCT_DW_5_linear_cache_policy_mask = 0x00000007
SDMA_PKT_COPY_STRUCT_DW_5_linear_cache_policy_shift = 18
SDMA_PKT_COPY_STRUCT_DW_5_LINEAR_CACHE_POLICY = lambda x: (((x) & SDMA_PKT_COPY_STRUCT_DW_5_linear_cache_policy_mask) << SDMA_PKT_COPY_STRUCT_DW_5_linear_cache_policy_shift)
SDMA_PKT_COPY_STRUCT_DW_5_struct__sw_offset = 5
SDMA_PKT_COPY_STRUCT_DW_5_struct__sw_mask = 0x00000003
SDMA_PKT_COPY_STRUCT_DW_5_struct__sw_shift = 24
SDMA_PKT_COPY_STRUCT_DW_5_STRUCT_SW = lambda x: (((x) & SDMA_PKT_COPY_STRUCT_DW_5_struct__sw_mask) << SDMA_PKT_COPY_STRUCT_DW_5_struct__sw_shift)
SDMA_PKT_COPY_STRUCT_DW_5_struct__cache_policy_offset = 5
SDMA_PKT_COPY_STRUCT_DW_5_struct__cache_policy_mask = 0x00000007
SDMA_PKT_COPY_STRUCT_DW_5_struct__cache_policy_shift = 26
SDMA_PKT_COPY_STRUCT_DW_5_STRUCT_CACHE_POLICY = lambda x: (((x) & SDMA_PKT_COPY_STRUCT_DW_5_struct__cache_policy_mask) << SDMA_PKT_COPY_STRUCT_DW_5_struct__cache_policy_shift)
SDMA_PKT_COPY_STRUCT_LINEAR_ADDR_LO_linear_addr_31_0_offset = 6
SDMA_PKT_COPY_STRUCT_LINEAR_ADDR_LO_linear_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COPY_STRUCT_LINEAR_ADDR_LO_linear_addr_31_0_shift = 0
SDMA_PKT_COPY_STRUCT_LINEAR_ADDR_LO_LINEAR_ADDR_31_0 = lambda x: (((x) & SDMA_PKT_COPY_STRUCT_LINEAR_ADDR_LO_linear_addr_31_0_mask) << SDMA_PKT_COPY_STRUCT_LINEAR_ADDR_LO_linear_addr_31_0_shift)
SDMA_PKT_COPY_STRUCT_LINEAR_ADDR_HI_linear_addr_63_32_offset = 7
SDMA_PKT_COPY_STRUCT_LINEAR_ADDR_HI_linear_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COPY_STRUCT_LINEAR_ADDR_HI_linear_addr_63_32_shift = 0
SDMA_PKT_COPY_STRUCT_LINEAR_ADDR_HI_LINEAR_ADDR_63_32 = lambda x: (((x) & SDMA_PKT_COPY_STRUCT_LINEAR_ADDR_HI_linear_addr_63_32_mask) << SDMA_PKT_COPY_STRUCT_LINEAR_ADDR_HI_linear_addr_63_32_shift)
SDMA_PKT_WRITE_UNTILED_HEADER_op_offset = 0
SDMA_PKT_WRITE_UNTILED_HEADER_op_mask = 0x000000FF
SDMA_PKT_WRITE_UNTILED_HEADER_op_shift = 0
SDMA_PKT_WRITE_UNTILED_HEADER_OP = lambda x: (((x) & SDMA_PKT_WRITE_UNTILED_HEADER_op_mask) << SDMA_PKT_WRITE_UNTILED_HEADER_op_shift)
SDMA_PKT_WRITE_UNTILED_HEADER_sub_op_offset = 0
SDMA_PKT_WRITE_UNTILED_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_WRITE_UNTILED_HEADER_sub_op_shift = 8
SDMA_PKT_WRITE_UNTILED_HEADER_SUB_OP = lambda x: (((x) & SDMA_PKT_WRITE_UNTILED_HEADER_sub_op_mask) << SDMA_PKT_WRITE_UNTILED_HEADER_sub_op_shift)
SDMA_PKT_WRITE_UNTILED_HEADER_encrypt_offset = 0
SDMA_PKT_WRITE_UNTILED_HEADER_encrypt_mask = 0x00000001
SDMA_PKT_WRITE_UNTILED_HEADER_encrypt_shift = 16
SDMA_PKT_WRITE_UNTILED_HEADER_ENCRYPT = lambda x: (((x) & SDMA_PKT_WRITE_UNTILED_HEADER_encrypt_mask) << SDMA_PKT_WRITE_UNTILED_HEADER_encrypt_shift)
SDMA_PKT_WRITE_UNTILED_HEADER_tmz_offset = 0
SDMA_PKT_WRITE_UNTILED_HEADER_tmz_mask = 0x00000001
SDMA_PKT_WRITE_UNTILED_HEADER_tmz_shift = 18
SDMA_PKT_WRITE_UNTILED_HEADER_TMZ = lambda x: (((x) & SDMA_PKT_WRITE_UNTILED_HEADER_tmz_mask) << SDMA_PKT_WRITE_UNTILED_HEADER_tmz_shift)
SDMA_PKT_WRITE_UNTILED_HEADER_cpv_offset = 0
SDMA_PKT_WRITE_UNTILED_HEADER_cpv_mask = 0x00000001
SDMA_PKT_WRITE_UNTILED_HEADER_cpv_shift = 28
SDMA_PKT_WRITE_UNTILED_HEADER_CPV = lambda x: (((x) & SDMA_PKT_WRITE_UNTILED_HEADER_cpv_mask) << SDMA_PKT_WRITE_UNTILED_HEADER_cpv_shift)
SDMA_PKT_WRITE_UNTILED_DST_ADDR_LO_dst_addr_31_0_offset = 1
SDMA_PKT_WRITE_UNTILED_DST_ADDR_LO_dst_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_WRITE_UNTILED_DST_ADDR_LO_dst_addr_31_0_shift = 0
SDMA_PKT_WRITE_UNTILED_DST_ADDR_LO_DST_ADDR_31_0 = lambda x: (((x) & SDMA_PKT_WRITE_UNTILED_DST_ADDR_LO_dst_addr_31_0_mask) << SDMA_PKT_WRITE_UNTILED_DST_ADDR_LO_dst_addr_31_0_shift)
SDMA_PKT_WRITE_UNTILED_DST_ADDR_HI_dst_addr_63_32_offset = 2
SDMA_PKT_WRITE_UNTILED_DST_ADDR_HI_dst_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_WRITE_UNTILED_DST_ADDR_HI_dst_addr_63_32_shift = 0
SDMA_PKT_WRITE_UNTILED_DST_ADDR_HI_DST_ADDR_63_32 = lambda x: (((x) & SDMA_PKT_WRITE_UNTILED_DST_ADDR_HI_dst_addr_63_32_mask) << SDMA_PKT_WRITE_UNTILED_DST_ADDR_HI_dst_addr_63_32_shift)
SDMA_PKT_WRITE_UNTILED_DW_3_count_offset = 3
SDMA_PKT_WRITE_UNTILED_DW_3_count_mask = 0x000FFFFF
SDMA_PKT_WRITE_UNTILED_DW_3_count_shift = 0
SDMA_PKT_WRITE_UNTILED_DW_3_COUNT = lambda x: (((x) & SDMA_PKT_WRITE_UNTILED_DW_3_count_mask) << SDMA_PKT_WRITE_UNTILED_DW_3_count_shift)
SDMA_PKT_WRITE_UNTILED_DW_3_sw_offset = 3
SDMA_PKT_WRITE_UNTILED_DW_3_sw_mask = 0x00000003
SDMA_PKT_WRITE_UNTILED_DW_3_sw_shift = 24
SDMA_PKT_WRITE_UNTILED_DW_3_SW = lambda x: (((x) & SDMA_PKT_WRITE_UNTILED_DW_3_sw_mask) << SDMA_PKT_WRITE_UNTILED_DW_3_sw_shift)
SDMA_PKT_WRITE_UNTILED_DW_3_cache_policy_offset = 3
SDMA_PKT_WRITE_UNTILED_DW_3_cache_policy_mask = 0x00000007
SDMA_PKT_WRITE_UNTILED_DW_3_cache_policy_shift = 26
SDMA_PKT_WRITE_UNTILED_DW_3_CACHE_POLICY = lambda x: (((x) & SDMA_PKT_WRITE_UNTILED_DW_3_cache_policy_mask) << SDMA_PKT_WRITE_UNTILED_DW_3_cache_policy_shift)
SDMA_PKT_WRITE_UNTILED_DATA0_data0_offset = 4
SDMA_PKT_WRITE_UNTILED_DATA0_data0_mask = 0xFFFFFFFF
SDMA_PKT_WRITE_UNTILED_DATA0_data0_shift = 0
SDMA_PKT_WRITE_UNTILED_DATA0_DATA0 = lambda x: (((x) & SDMA_PKT_WRITE_UNTILED_DATA0_data0_mask) << SDMA_PKT_WRITE_UNTILED_DATA0_data0_shift)
SDMA_PKT_WRITE_TILED_HEADER_op_offset = 0
SDMA_PKT_WRITE_TILED_HEADER_op_mask = 0x000000FF
SDMA_PKT_WRITE_TILED_HEADER_op_shift = 0
SDMA_PKT_WRITE_TILED_HEADER_OP = lambda x: (((x) & SDMA_PKT_WRITE_TILED_HEADER_op_mask) << SDMA_PKT_WRITE_TILED_HEADER_op_shift)
SDMA_PKT_WRITE_TILED_HEADER_sub_op_offset = 0
SDMA_PKT_WRITE_TILED_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_WRITE_TILED_HEADER_sub_op_shift = 8
SDMA_PKT_WRITE_TILED_HEADER_SUB_OP = lambda x: (((x) & SDMA_PKT_WRITE_TILED_HEADER_sub_op_mask) << SDMA_PKT_WRITE_TILED_HEADER_sub_op_shift)
SDMA_PKT_WRITE_TILED_HEADER_encrypt_offset = 0
SDMA_PKT_WRITE_TILED_HEADER_encrypt_mask = 0x00000001
SDMA_PKT_WRITE_TILED_HEADER_encrypt_shift = 16
SDMA_PKT_WRITE_TILED_HEADER_ENCRYPT = lambda x: (((x) & SDMA_PKT_WRITE_TILED_HEADER_encrypt_mask) << SDMA_PKT_WRITE_TILED_HEADER_encrypt_shift)
SDMA_PKT_WRITE_TILED_HEADER_tmz_offset = 0
SDMA_PKT_WRITE_TILED_HEADER_tmz_mask = 0x00000001
SDMA_PKT_WRITE_TILED_HEADER_tmz_shift = 18
SDMA_PKT_WRITE_TILED_HEADER_TMZ = lambda x: (((x) & SDMA_PKT_WRITE_TILED_HEADER_tmz_mask) << SDMA_PKT_WRITE_TILED_HEADER_tmz_shift)
SDMA_PKT_WRITE_TILED_HEADER_cpv_offset = 0
SDMA_PKT_WRITE_TILED_HEADER_cpv_mask = 0x00000001
SDMA_PKT_WRITE_TILED_HEADER_cpv_shift = 28
SDMA_PKT_WRITE_TILED_HEADER_CPV = lambda x: (((x) & SDMA_PKT_WRITE_TILED_HEADER_cpv_mask) << SDMA_PKT_WRITE_TILED_HEADER_cpv_shift)
SDMA_PKT_WRITE_TILED_DST_ADDR_LO_dst_addr_31_0_offset = 1
SDMA_PKT_WRITE_TILED_DST_ADDR_LO_dst_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_WRITE_TILED_DST_ADDR_LO_dst_addr_31_0_shift = 0
SDMA_PKT_WRITE_TILED_DST_ADDR_LO_DST_ADDR_31_0 = lambda x: (((x) & SDMA_PKT_WRITE_TILED_DST_ADDR_LO_dst_addr_31_0_mask) << SDMA_PKT_WRITE_TILED_DST_ADDR_LO_dst_addr_31_0_shift)
SDMA_PKT_WRITE_TILED_DST_ADDR_HI_dst_addr_63_32_offset = 2
SDMA_PKT_WRITE_TILED_DST_ADDR_HI_dst_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_WRITE_TILED_DST_ADDR_HI_dst_addr_63_32_shift = 0
SDMA_PKT_WRITE_TILED_DST_ADDR_HI_DST_ADDR_63_32 = lambda x: (((x) & SDMA_PKT_WRITE_TILED_DST_ADDR_HI_dst_addr_63_32_mask) << SDMA_PKT_WRITE_TILED_DST_ADDR_HI_dst_addr_63_32_shift)
SDMA_PKT_WRITE_TILED_DW_3_width_offset = 3
SDMA_PKT_WRITE_TILED_DW_3_width_mask = 0x00003FFF
SDMA_PKT_WRITE_TILED_DW_3_width_shift = 0
SDMA_PKT_WRITE_TILED_DW_3_WIDTH = lambda x: (((x) & SDMA_PKT_WRITE_TILED_DW_3_width_mask) << SDMA_PKT_WRITE_TILED_DW_3_width_shift)
SDMA_PKT_WRITE_TILED_DW_4_height_offset = 4
SDMA_PKT_WRITE_TILED_DW_4_height_mask = 0x00003FFF
SDMA_PKT_WRITE_TILED_DW_4_height_shift = 0
SDMA_PKT_WRITE_TILED_DW_4_HEIGHT = lambda x: (((x) & SDMA_PKT_WRITE_TILED_DW_4_height_mask) << SDMA_PKT_WRITE_TILED_DW_4_height_shift)
SDMA_PKT_WRITE_TILED_DW_4_depth_offset = 4
SDMA_PKT_WRITE_TILED_DW_4_depth_mask = 0x00001FFF
SDMA_PKT_WRITE_TILED_DW_4_depth_shift = 16
SDMA_PKT_WRITE_TILED_DW_4_DEPTH = lambda x: (((x) & SDMA_PKT_WRITE_TILED_DW_4_depth_mask) << SDMA_PKT_WRITE_TILED_DW_4_depth_shift)
SDMA_PKT_WRITE_TILED_DW_5_element_size_offset = 5
SDMA_PKT_WRITE_TILED_DW_5_element_size_mask = 0x00000007
SDMA_PKT_WRITE_TILED_DW_5_element_size_shift = 0
SDMA_PKT_WRITE_TILED_DW_5_ELEMENT_SIZE = lambda x: (((x) & SDMA_PKT_WRITE_TILED_DW_5_element_size_mask) << SDMA_PKT_WRITE_TILED_DW_5_element_size_shift)
SDMA_PKT_WRITE_TILED_DW_5_swizzle_mode_offset = 5
SDMA_PKT_WRITE_TILED_DW_5_swizzle_mode_mask = 0x0000001F
SDMA_PKT_WRITE_TILED_DW_5_swizzle_mode_shift = 3
SDMA_PKT_WRITE_TILED_DW_5_SWIZZLE_MODE = lambda x: (((x) & SDMA_PKT_WRITE_TILED_DW_5_swizzle_mode_mask) << SDMA_PKT_WRITE_TILED_DW_5_swizzle_mode_shift)
SDMA_PKT_WRITE_TILED_DW_5_dimension_offset = 5
SDMA_PKT_WRITE_TILED_DW_5_dimension_mask = 0x00000003
SDMA_PKT_WRITE_TILED_DW_5_dimension_shift = 9
SDMA_PKT_WRITE_TILED_DW_5_DIMENSION = lambda x: (((x) & SDMA_PKT_WRITE_TILED_DW_5_dimension_mask) << SDMA_PKT_WRITE_TILED_DW_5_dimension_shift)
SDMA_PKT_WRITE_TILED_DW_5_mip_max_offset = 5
SDMA_PKT_WRITE_TILED_DW_5_mip_max_mask = 0x0000000F
SDMA_PKT_WRITE_TILED_DW_5_mip_max_shift = 16
SDMA_PKT_WRITE_TILED_DW_5_MIP_MAX = lambda x: (((x) & SDMA_PKT_WRITE_TILED_DW_5_mip_max_mask) << SDMA_PKT_WRITE_TILED_DW_5_mip_max_shift)
SDMA_PKT_WRITE_TILED_DW_6_x_offset = 6
SDMA_PKT_WRITE_TILED_DW_6_x_mask = 0x00003FFF
SDMA_PKT_WRITE_TILED_DW_6_x_shift = 0
SDMA_PKT_WRITE_TILED_DW_6_X = lambda x: (((x) & SDMA_PKT_WRITE_TILED_DW_6_x_mask) << SDMA_PKT_WRITE_TILED_DW_6_x_shift)
SDMA_PKT_WRITE_TILED_DW_6_y_offset = 6
SDMA_PKT_WRITE_TILED_DW_6_y_mask = 0x00003FFF
SDMA_PKT_WRITE_TILED_DW_6_y_shift = 16
SDMA_PKT_WRITE_TILED_DW_6_Y = lambda x: (((x) & SDMA_PKT_WRITE_TILED_DW_6_y_mask) << SDMA_PKT_WRITE_TILED_DW_6_y_shift)
SDMA_PKT_WRITE_TILED_DW_7_z_offset = 7
SDMA_PKT_WRITE_TILED_DW_7_z_mask = 0x00001FFF
SDMA_PKT_WRITE_TILED_DW_7_z_shift = 0
SDMA_PKT_WRITE_TILED_DW_7_Z = lambda x: (((x) & SDMA_PKT_WRITE_TILED_DW_7_z_mask) << SDMA_PKT_WRITE_TILED_DW_7_z_shift)
SDMA_PKT_WRITE_TILED_DW_7_sw_offset = 7
SDMA_PKT_WRITE_TILED_DW_7_sw_mask = 0x00000003
SDMA_PKT_WRITE_TILED_DW_7_sw_shift = 24
SDMA_PKT_WRITE_TILED_DW_7_SW = lambda x: (((x) & SDMA_PKT_WRITE_TILED_DW_7_sw_mask) << SDMA_PKT_WRITE_TILED_DW_7_sw_shift)
SDMA_PKT_WRITE_TILED_DW_7_cache_policy_offset = 7
SDMA_PKT_WRITE_TILED_DW_7_cache_policy_mask = 0x00000007
SDMA_PKT_WRITE_TILED_DW_7_cache_policy_shift = 26
SDMA_PKT_WRITE_TILED_DW_7_CACHE_POLICY = lambda x: (((x) & SDMA_PKT_WRITE_TILED_DW_7_cache_policy_mask) << SDMA_PKT_WRITE_TILED_DW_7_cache_policy_shift)
SDMA_PKT_WRITE_TILED_COUNT_count_offset = 8
SDMA_PKT_WRITE_TILED_COUNT_count_mask = 0x000FFFFF
SDMA_PKT_WRITE_TILED_COUNT_count_shift = 0
SDMA_PKT_WRITE_TILED_COUNT_COUNT = lambda x: (((x) & SDMA_PKT_WRITE_TILED_COUNT_count_mask) << SDMA_PKT_WRITE_TILED_COUNT_count_shift)
SDMA_PKT_WRITE_TILED_DATA0_data0_offset = 9
SDMA_PKT_WRITE_TILED_DATA0_data0_mask = 0xFFFFFFFF
SDMA_PKT_WRITE_TILED_DATA0_data0_shift = 0
SDMA_PKT_WRITE_TILED_DATA0_DATA0 = lambda x: (((x) & SDMA_PKT_WRITE_TILED_DATA0_data0_mask) << SDMA_PKT_WRITE_TILED_DATA0_data0_shift)
SDMA_PKT_WRITE_TILED_BC_HEADER_op_offset = 0
SDMA_PKT_WRITE_TILED_BC_HEADER_op_mask = 0x000000FF
SDMA_PKT_WRITE_TILED_BC_HEADER_op_shift = 0
SDMA_PKT_WRITE_TILED_BC_HEADER_OP = lambda x: (((x) & SDMA_PKT_WRITE_TILED_BC_HEADER_op_mask) << SDMA_PKT_WRITE_TILED_BC_HEADER_op_shift)
SDMA_PKT_WRITE_TILED_BC_HEADER_sub_op_offset = 0
SDMA_PKT_WRITE_TILED_BC_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_WRITE_TILED_BC_HEADER_sub_op_shift = 8
SDMA_PKT_WRITE_TILED_BC_HEADER_SUB_OP = lambda x: (((x) & SDMA_PKT_WRITE_TILED_BC_HEADER_sub_op_mask) << SDMA_PKT_WRITE_TILED_BC_HEADER_sub_op_shift)
SDMA_PKT_WRITE_TILED_BC_DST_ADDR_LO_dst_addr_31_0_offset = 1
SDMA_PKT_WRITE_TILED_BC_DST_ADDR_LO_dst_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_WRITE_TILED_BC_DST_ADDR_LO_dst_addr_31_0_shift = 0
SDMA_PKT_WRITE_TILED_BC_DST_ADDR_LO_DST_ADDR_31_0 = lambda x: (((x) & SDMA_PKT_WRITE_TILED_BC_DST_ADDR_LO_dst_addr_31_0_mask) << SDMA_PKT_WRITE_TILED_BC_DST_ADDR_LO_dst_addr_31_0_shift)
SDMA_PKT_WRITE_TILED_BC_DST_ADDR_HI_dst_addr_63_32_offset = 2
SDMA_PKT_WRITE_TILED_BC_DST_ADDR_HI_dst_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_WRITE_TILED_BC_DST_ADDR_HI_dst_addr_63_32_shift = 0
SDMA_PKT_WRITE_TILED_BC_DST_ADDR_HI_DST_ADDR_63_32 = lambda x: (((x) & SDMA_PKT_WRITE_TILED_BC_DST_ADDR_HI_dst_addr_63_32_mask) << SDMA_PKT_WRITE_TILED_BC_DST_ADDR_HI_dst_addr_63_32_shift)
SDMA_PKT_WRITE_TILED_BC_DW_3_width_offset = 3
SDMA_PKT_WRITE_TILED_BC_DW_3_width_mask = 0x00003FFF
SDMA_PKT_WRITE_TILED_BC_DW_3_width_shift = 0
SDMA_PKT_WRITE_TILED_BC_DW_3_WIDTH = lambda x: (((x) & SDMA_PKT_WRITE_TILED_BC_DW_3_width_mask) << SDMA_PKT_WRITE_TILED_BC_DW_3_width_shift)
SDMA_PKT_WRITE_TILED_BC_DW_4_height_offset = 4
SDMA_PKT_WRITE_TILED_BC_DW_4_height_mask = 0x00003FFF
SDMA_PKT_WRITE_TILED_BC_DW_4_height_shift = 0
SDMA_PKT_WRITE_TILED_BC_DW_4_HEIGHT = lambda x: (((x) & SDMA_PKT_WRITE_TILED_BC_DW_4_height_mask) << SDMA_PKT_WRITE_TILED_BC_DW_4_height_shift)
SDMA_PKT_WRITE_TILED_BC_DW_4_depth_offset = 4
SDMA_PKT_WRITE_TILED_BC_DW_4_depth_mask = 0x000007FF
SDMA_PKT_WRITE_TILED_BC_DW_4_depth_shift = 16
SDMA_PKT_WRITE_TILED_BC_DW_4_DEPTH = lambda x: (((x) & SDMA_PKT_WRITE_TILED_BC_DW_4_depth_mask) << SDMA_PKT_WRITE_TILED_BC_DW_4_depth_shift)
SDMA_PKT_WRITE_TILED_BC_DW_5_element_size_offset = 5
SDMA_PKT_WRITE_TILED_BC_DW_5_element_size_mask = 0x00000007
SDMA_PKT_WRITE_TILED_BC_DW_5_element_size_shift = 0
SDMA_PKT_WRITE_TILED_BC_DW_5_ELEMENT_SIZE = lambda x: (((x) & SDMA_PKT_WRITE_TILED_BC_DW_5_element_size_mask) << SDMA_PKT_WRITE_TILED_BC_DW_5_element_size_shift)
SDMA_PKT_WRITE_TILED_BC_DW_5_array_mode_offset = 5
SDMA_PKT_WRITE_TILED_BC_DW_5_array_mode_mask = 0x0000000F
SDMA_PKT_WRITE_TILED_BC_DW_5_array_mode_shift = 3
SDMA_PKT_WRITE_TILED_BC_DW_5_ARRAY_MODE = lambda x: (((x) & SDMA_PKT_WRITE_TILED_BC_DW_5_array_mode_mask) << SDMA_PKT_WRITE_TILED_BC_DW_5_array_mode_shift)
SDMA_PKT_WRITE_TILED_BC_DW_5_mit_mode_offset = 5
SDMA_PKT_WRITE_TILED_BC_DW_5_mit_mode_mask = 0x00000007
SDMA_PKT_WRITE_TILED_BC_DW_5_mit_mode_shift = 8
SDMA_PKT_WRITE_TILED_BC_DW_5_MIT_MODE = lambda x: (((x) & SDMA_PKT_WRITE_TILED_BC_DW_5_mit_mode_mask) << SDMA_PKT_WRITE_TILED_BC_DW_5_mit_mode_shift)
SDMA_PKT_WRITE_TILED_BC_DW_5_tilesplit_size_offset = 5
SDMA_PKT_WRITE_TILED_BC_DW_5_tilesplit_size_mask = 0x00000007
SDMA_PKT_WRITE_TILED_BC_DW_5_tilesplit_size_shift = 11
SDMA_PKT_WRITE_TILED_BC_DW_5_TILESPLIT_SIZE = lambda x: (((x) & SDMA_PKT_WRITE_TILED_BC_DW_5_tilesplit_size_mask) << SDMA_PKT_WRITE_TILED_BC_DW_5_tilesplit_size_shift)
SDMA_PKT_WRITE_TILED_BC_DW_5_bank_w_offset = 5
SDMA_PKT_WRITE_TILED_BC_DW_5_bank_w_mask = 0x00000003
SDMA_PKT_WRITE_TILED_BC_DW_5_bank_w_shift = 15
SDMA_PKT_WRITE_TILED_BC_DW_5_BANK_W = lambda x: (((x) & SDMA_PKT_WRITE_TILED_BC_DW_5_bank_w_mask) << SDMA_PKT_WRITE_TILED_BC_DW_5_bank_w_shift)
SDMA_PKT_WRITE_TILED_BC_DW_5_bank_h_offset = 5
SDMA_PKT_WRITE_TILED_BC_DW_5_bank_h_mask = 0x00000003
SDMA_PKT_WRITE_TILED_BC_DW_5_bank_h_shift = 18
SDMA_PKT_WRITE_TILED_BC_DW_5_BANK_H = lambda x: (((x) & SDMA_PKT_WRITE_TILED_BC_DW_5_bank_h_mask) << SDMA_PKT_WRITE_TILED_BC_DW_5_bank_h_shift)
SDMA_PKT_WRITE_TILED_BC_DW_5_num_bank_offset = 5
SDMA_PKT_WRITE_TILED_BC_DW_5_num_bank_mask = 0x00000003
SDMA_PKT_WRITE_TILED_BC_DW_5_num_bank_shift = 21
SDMA_PKT_WRITE_TILED_BC_DW_5_NUM_BANK = lambda x: (((x) & SDMA_PKT_WRITE_TILED_BC_DW_5_num_bank_mask) << SDMA_PKT_WRITE_TILED_BC_DW_5_num_bank_shift)
SDMA_PKT_WRITE_TILED_BC_DW_5_mat_aspt_offset = 5
SDMA_PKT_WRITE_TILED_BC_DW_5_mat_aspt_mask = 0x00000003
SDMA_PKT_WRITE_TILED_BC_DW_5_mat_aspt_shift = 24
SDMA_PKT_WRITE_TILED_BC_DW_5_MAT_ASPT = lambda x: (((x) & SDMA_PKT_WRITE_TILED_BC_DW_5_mat_aspt_mask) << SDMA_PKT_WRITE_TILED_BC_DW_5_mat_aspt_shift)
SDMA_PKT_WRITE_TILED_BC_DW_5_pipe_config_offset = 5
SDMA_PKT_WRITE_TILED_BC_DW_5_pipe_config_mask = 0x0000001F
SDMA_PKT_WRITE_TILED_BC_DW_5_pipe_config_shift = 26
SDMA_PKT_WRITE_TILED_BC_DW_5_PIPE_CONFIG = lambda x: (((x) & SDMA_PKT_WRITE_TILED_BC_DW_5_pipe_config_mask) << SDMA_PKT_WRITE_TILED_BC_DW_5_pipe_config_shift)
SDMA_PKT_WRITE_TILED_BC_DW_6_x_offset = 6
SDMA_PKT_WRITE_TILED_BC_DW_6_x_mask = 0x00003FFF
SDMA_PKT_WRITE_TILED_BC_DW_6_x_shift = 0
SDMA_PKT_WRITE_TILED_BC_DW_6_X = lambda x: (((x) & SDMA_PKT_WRITE_TILED_BC_DW_6_x_mask) << SDMA_PKT_WRITE_TILED_BC_DW_6_x_shift)
SDMA_PKT_WRITE_TILED_BC_DW_6_y_offset = 6
SDMA_PKT_WRITE_TILED_BC_DW_6_y_mask = 0x00003FFF
SDMA_PKT_WRITE_TILED_BC_DW_6_y_shift = 16
SDMA_PKT_WRITE_TILED_BC_DW_6_Y = lambda x: (((x) & SDMA_PKT_WRITE_TILED_BC_DW_6_y_mask) << SDMA_PKT_WRITE_TILED_BC_DW_6_y_shift)
SDMA_PKT_WRITE_TILED_BC_DW_7_z_offset = 7
SDMA_PKT_WRITE_TILED_BC_DW_7_z_mask = 0x000007FF
SDMA_PKT_WRITE_TILED_BC_DW_7_z_shift = 0
SDMA_PKT_WRITE_TILED_BC_DW_7_Z = lambda x: (((x) & SDMA_PKT_WRITE_TILED_BC_DW_7_z_mask) << SDMA_PKT_WRITE_TILED_BC_DW_7_z_shift)
SDMA_PKT_WRITE_TILED_BC_DW_7_sw_offset = 7
SDMA_PKT_WRITE_TILED_BC_DW_7_sw_mask = 0x00000003
SDMA_PKT_WRITE_TILED_BC_DW_7_sw_shift = 24
SDMA_PKT_WRITE_TILED_BC_DW_7_SW = lambda x: (((x) & SDMA_PKT_WRITE_TILED_BC_DW_7_sw_mask) << SDMA_PKT_WRITE_TILED_BC_DW_7_sw_shift)
SDMA_PKT_WRITE_TILED_BC_COUNT_count_offset = 8
SDMA_PKT_WRITE_TILED_BC_COUNT_count_mask = 0x000FFFFF
SDMA_PKT_WRITE_TILED_BC_COUNT_count_shift = 2
SDMA_PKT_WRITE_TILED_BC_COUNT_COUNT = lambda x: (((x) & SDMA_PKT_WRITE_TILED_BC_COUNT_count_mask) << SDMA_PKT_WRITE_TILED_BC_COUNT_count_shift)
SDMA_PKT_WRITE_TILED_BC_DATA0_data0_offset = 9
SDMA_PKT_WRITE_TILED_BC_DATA0_data0_mask = 0xFFFFFFFF
SDMA_PKT_WRITE_TILED_BC_DATA0_data0_shift = 0
SDMA_PKT_WRITE_TILED_BC_DATA0_DATA0 = lambda x: (((x) & SDMA_PKT_WRITE_TILED_BC_DATA0_data0_mask) << SDMA_PKT_WRITE_TILED_BC_DATA0_data0_shift)
SDMA_PKT_PTEPDE_COPY_HEADER_op_offset = 0
SDMA_PKT_PTEPDE_COPY_HEADER_op_mask = 0x000000FF
SDMA_PKT_PTEPDE_COPY_HEADER_op_shift = 0
SDMA_PKT_PTEPDE_COPY_HEADER_OP = lambda x: (((x) & SDMA_PKT_PTEPDE_COPY_HEADER_op_mask) << SDMA_PKT_PTEPDE_COPY_HEADER_op_shift)
SDMA_PKT_PTEPDE_COPY_HEADER_sub_op_offset = 0
SDMA_PKT_PTEPDE_COPY_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_PTEPDE_COPY_HEADER_sub_op_shift = 8
SDMA_PKT_PTEPDE_COPY_HEADER_SUB_OP = lambda x: (((x) & SDMA_PKT_PTEPDE_COPY_HEADER_sub_op_mask) << SDMA_PKT_PTEPDE_COPY_HEADER_sub_op_shift)
SDMA_PKT_PTEPDE_COPY_HEADER_tmz_offset = 0
SDMA_PKT_PTEPDE_COPY_HEADER_tmz_mask = 0x00000001
SDMA_PKT_PTEPDE_COPY_HEADER_tmz_shift = 18
SDMA_PKT_PTEPDE_COPY_HEADER_TMZ = lambda x: (((x) & SDMA_PKT_PTEPDE_COPY_HEADER_tmz_mask) << SDMA_PKT_PTEPDE_COPY_HEADER_tmz_shift)
SDMA_PKT_PTEPDE_COPY_HEADER_cpv_offset = 0
SDMA_PKT_PTEPDE_COPY_HEADER_cpv_mask = 0x00000001
SDMA_PKT_PTEPDE_COPY_HEADER_cpv_shift = 28
SDMA_PKT_PTEPDE_COPY_HEADER_CPV = lambda x: (((x) & SDMA_PKT_PTEPDE_COPY_HEADER_cpv_mask) << SDMA_PKT_PTEPDE_COPY_HEADER_cpv_shift)
SDMA_PKT_PTEPDE_COPY_HEADER_ptepde_op_offset = 0
SDMA_PKT_PTEPDE_COPY_HEADER_ptepde_op_mask = 0x00000001
SDMA_PKT_PTEPDE_COPY_HEADER_ptepde_op_shift = 31
SDMA_PKT_PTEPDE_COPY_HEADER_PTEPDE_OP = lambda x: (((x) & SDMA_PKT_PTEPDE_COPY_HEADER_ptepde_op_mask) << SDMA_PKT_PTEPDE_COPY_HEADER_ptepde_op_shift)
SDMA_PKT_PTEPDE_COPY_SRC_ADDR_LO_src_addr_31_0_offset = 1
SDMA_PKT_PTEPDE_COPY_SRC_ADDR_LO_src_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_PTEPDE_COPY_SRC_ADDR_LO_src_addr_31_0_shift = 0
SDMA_PKT_PTEPDE_COPY_SRC_ADDR_LO_SRC_ADDR_31_0 = lambda x: (((x) & SDMA_PKT_PTEPDE_COPY_SRC_ADDR_LO_src_addr_31_0_mask) << SDMA_PKT_PTEPDE_COPY_SRC_ADDR_LO_src_addr_31_0_shift)
SDMA_PKT_PTEPDE_COPY_SRC_ADDR_HI_src_addr_63_32_offset = 2
SDMA_PKT_PTEPDE_COPY_SRC_ADDR_HI_src_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_PTEPDE_COPY_SRC_ADDR_HI_src_addr_63_32_shift = 0
SDMA_PKT_PTEPDE_COPY_SRC_ADDR_HI_SRC_ADDR_63_32 = lambda x: (((x) & SDMA_PKT_PTEPDE_COPY_SRC_ADDR_HI_src_addr_63_32_mask) << SDMA_PKT_PTEPDE_COPY_SRC_ADDR_HI_src_addr_63_32_shift)
SDMA_PKT_PTEPDE_COPY_DST_ADDR_LO_dst_addr_31_0_offset = 3
SDMA_PKT_PTEPDE_COPY_DST_ADDR_LO_dst_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_PTEPDE_COPY_DST_ADDR_LO_dst_addr_31_0_shift = 0
SDMA_PKT_PTEPDE_COPY_DST_ADDR_LO_DST_ADDR_31_0 = lambda x: (((x) & SDMA_PKT_PTEPDE_COPY_DST_ADDR_LO_dst_addr_31_0_mask) << SDMA_PKT_PTEPDE_COPY_DST_ADDR_LO_dst_addr_31_0_shift)
SDMA_PKT_PTEPDE_COPY_DST_ADDR_HI_dst_addr_63_32_offset = 4
SDMA_PKT_PTEPDE_COPY_DST_ADDR_HI_dst_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_PTEPDE_COPY_DST_ADDR_HI_dst_addr_63_32_shift = 0
SDMA_PKT_PTEPDE_COPY_DST_ADDR_HI_DST_ADDR_63_32 = lambda x: (((x) & SDMA_PKT_PTEPDE_COPY_DST_ADDR_HI_dst_addr_63_32_mask) << SDMA_PKT_PTEPDE_COPY_DST_ADDR_HI_dst_addr_63_32_shift)
SDMA_PKT_PTEPDE_COPY_MASK_DW0_mask_dw0_offset = 5
SDMA_PKT_PTEPDE_COPY_MASK_DW0_mask_dw0_mask = 0xFFFFFFFF
SDMA_PKT_PTEPDE_COPY_MASK_DW0_mask_dw0_shift = 0
SDMA_PKT_PTEPDE_COPY_MASK_DW0_MASK_DW0 = lambda x: (((x) & SDMA_PKT_PTEPDE_COPY_MASK_DW0_mask_dw0_mask) << SDMA_PKT_PTEPDE_COPY_MASK_DW0_mask_dw0_shift)
SDMA_PKT_PTEPDE_COPY_MASK_DW1_mask_dw1_offset = 6
SDMA_PKT_PTEPDE_COPY_MASK_DW1_mask_dw1_mask = 0xFFFFFFFF
SDMA_PKT_PTEPDE_COPY_MASK_DW1_mask_dw1_shift = 0
SDMA_PKT_PTEPDE_COPY_MASK_DW1_MASK_DW1 = lambda x: (((x) & SDMA_PKT_PTEPDE_COPY_MASK_DW1_mask_dw1_mask) << SDMA_PKT_PTEPDE_COPY_MASK_DW1_mask_dw1_shift)
SDMA_PKT_PTEPDE_COPY_COUNT_count_offset = 7
SDMA_PKT_PTEPDE_COPY_COUNT_count_mask = 0x0007FFFF
SDMA_PKT_PTEPDE_COPY_COUNT_count_shift = 0
SDMA_PKT_PTEPDE_COPY_COUNT_COUNT = lambda x: (((x) & SDMA_PKT_PTEPDE_COPY_COUNT_count_mask) << SDMA_PKT_PTEPDE_COPY_COUNT_count_shift)
SDMA_PKT_PTEPDE_COPY_COUNT_dst_cache_policy_offset = 7
SDMA_PKT_PTEPDE_COPY_COUNT_dst_cache_policy_mask = 0x00000007
SDMA_PKT_PTEPDE_COPY_COUNT_dst_cache_policy_shift = 22
SDMA_PKT_PTEPDE_COPY_COUNT_DST_CACHE_POLICY = lambda x: (((x) & SDMA_PKT_PTEPDE_COPY_COUNT_dst_cache_policy_mask) << SDMA_PKT_PTEPDE_COPY_COUNT_dst_cache_policy_shift)
SDMA_PKT_PTEPDE_COPY_COUNT_src_cache_policy_offset = 7
SDMA_PKT_PTEPDE_COPY_COUNT_src_cache_policy_mask = 0x00000007
SDMA_PKT_PTEPDE_COPY_COUNT_src_cache_policy_shift = 29
SDMA_PKT_PTEPDE_COPY_COUNT_SRC_CACHE_POLICY = lambda x: (((x) & SDMA_PKT_PTEPDE_COPY_COUNT_src_cache_policy_mask) << SDMA_PKT_PTEPDE_COPY_COUNT_src_cache_policy_shift)
SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_op_offset = 0
SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_op_mask = 0x000000FF
SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_op_shift = 0
SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_OP = lambda x: (((x) & SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_op_mask) << SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_op_shift)
SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_sub_op_offset = 0
SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_sub_op_shift = 8
SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_SUB_OP = lambda x: (((x) & SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_sub_op_mask) << SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_sub_op_shift)
SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_pte_size_offset = 0
SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_pte_size_mask = 0x00000003
SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_pte_size_shift = 28
SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_PTE_SIZE = lambda x: (((x) & SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_pte_size_mask) << SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_pte_size_shift)
SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_direction_offset = 0
SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_direction_mask = 0x00000001
SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_direction_shift = 30
SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_DIRECTION = lambda x: (((x) & SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_direction_mask) << SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_direction_shift)
SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_ptepde_op_offset = 0
SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_ptepde_op_mask = 0x00000001
SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_ptepde_op_shift = 31
SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_PTEPDE_OP = lambda x: (((x) & SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_ptepde_op_mask) << SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_ptepde_op_shift)
SDMA_PKT_PTEPDE_COPY_BACKWARDS_SRC_ADDR_LO_src_addr_31_0_offset = 1
SDMA_PKT_PTEPDE_COPY_BACKWARDS_SRC_ADDR_LO_src_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_PTEPDE_COPY_BACKWARDS_SRC_ADDR_LO_src_addr_31_0_shift = 0
SDMA_PKT_PTEPDE_COPY_BACKWARDS_SRC_ADDR_LO_SRC_ADDR_31_0 = lambda x: (((x) & SDMA_PKT_PTEPDE_COPY_BACKWARDS_SRC_ADDR_LO_src_addr_31_0_mask) << SDMA_PKT_PTEPDE_COPY_BACKWARDS_SRC_ADDR_LO_src_addr_31_0_shift)
SDMA_PKT_PTEPDE_COPY_BACKWARDS_SRC_ADDR_HI_src_addr_63_32_offset = 2
SDMA_PKT_PTEPDE_COPY_BACKWARDS_SRC_ADDR_HI_src_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_PTEPDE_COPY_BACKWARDS_SRC_ADDR_HI_src_addr_63_32_shift = 0
SDMA_PKT_PTEPDE_COPY_BACKWARDS_SRC_ADDR_HI_SRC_ADDR_63_32 = lambda x: (((x) & SDMA_PKT_PTEPDE_COPY_BACKWARDS_SRC_ADDR_HI_src_addr_63_32_mask) << SDMA_PKT_PTEPDE_COPY_BACKWARDS_SRC_ADDR_HI_src_addr_63_32_shift)
SDMA_PKT_PTEPDE_COPY_BACKWARDS_DST_ADDR_LO_dst_addr_31_0_offset = 3
SDMA_PKT_PTEPDE_COPY_BACKWARDS_DST_ADDR_LO_dst_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_PTEPDE_COPY_BACKWARDS_DST_ADDR_LO_dst_addr_31_0_shift = 0
SDMA_PKT_PTEPDE_COPY_BACKWARDS_DST_ADDR_LO_DST_ADDR_31_0 = lambda x: (((x) & SDMA_PKT_PTEPDE_COPY_BACKWARDS_DST_ADDR_LO_dst_addr_31_0_mask) << SDMA_PKT_PTEPDE_COPY_BACKWARDS_DST_ADDR_LO_dst_addr_31_0_shift)
SDMA_PKT_PTEPDE_COPY_BACKWARDS_DST_ADDR_HI_dst_addr_63_32_offset = 4
SDMA_PKT_PTEPDE_COPY_BACKWARDS_DST_ADDR_HI_dst_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_PTEPDE_COPY_BACKWARDS_DST_ADDR_HI_dst_addr_63_32_shift = 0
SDMA_PKT_PTEPDE_COPY_BACKWARDS_DST_ADDR_HI_DST_ADDR_63_32 = lambda x: (((x) & SDMA_PKT_PTEPDE_COPY_BACKWARDS_DST_ADDR_HI_dst_addr_63_32_mask) << SDMA_PKT_PTEPDE_COPY_BACKWARDS_DST_ADDR_HI_dst_addr_63_32_shift)
SDMA_PKT_PTEPDE_COPY_BACKWARDS_MASK_BIT_FOR_DW_mask_first_xfer_offset = 5
SDMA_PKT_PTEPDE_COPY_BACKWARDS_MASK_BIT_FOR_DW_mask_first_xfer_mask = 0x000000FF
SDMA_PKT_PTEPDE_COPY_BACKWARDS_MASK_BIT_FOR_DW_mask_first_xfer_shift = 0
SDMA_PKT_PTEPDE_COPY_BACKWARDS_MASK_BIT_FOR_DW_MASK_FIRST_XFER = lambda x: (((x) & SDMA_PKT_PTEPDE_COPY_BACKWARDS_MASK_BIT_FOR_DW_mask_first_xfer_mask) << SDMA_PKT_PTEPDE_COPY_BACKWARDS_MASK_BIT_FOR_DW_mask_first_xfer_shift)
SDMA_PKT_PTEPDE_COPY_BACKWARDS_MASK_BIT_FOR_DW_mask_last_xfer_offset = 5
SDMA_PKT_PTEPDE_COPY_BACKWARDS_MASK_BIT_FOR_DW_mask_last_xfer_mask = 0x000000FF
SDMA_PKT_PTEPDE_COPY_BACKWARDS_MASK_BIT_FOR_DW_mask_last_xfer_shift = 8
SDMA_PKT_PTEPDE_COPY_BACKWARDS_MASK_BIT_FOR_DW_MASK_LAST_XFER = lambda x: (((x) & SDMA_PKT_PTEPDE_COPY_BACKWARDS_MASK_BIT_FOR_DW_mask_last_xfer_mask) << SDMA_PKT_PTEPDE_COPY_BACKWARDS_MASK_BIT_FOR_DW_mask_last_xfer_shift)
SDMA_PKT_PTEPDE_COPY_BACKWARDS_COUNT_IN_32B_XFER_count_offset = 6
SDMA_PKT_PTEPDE_COPY_BACKWARDS_COUNT_IN_32B_XFER_count_mask = 0x0001FFFF
SDMA_PKT_PTEPDE_COPY_BACKWARDS_COUNT_IN_32B_XFER_count_shift = 0
SDMA_PKT_PTEPDE_COPY_BACKWARDS_COUNT_IN_32B_XFER_COUNT = lambda x: (((x) & SDMA_PKT_PTEPDE_COPY_BACKWARDS_COUNT_IN_32B_XFER_count_mask) << SDMA_PKT_PTEPDE_COPY_BACKWARDS_COUNT_IN_32B_XFER_count_shift)
SDMA_PKT_PTEPDE_RMW_HEADER_op_offset = 0
SDMA_PKT_PTEPDE_RMW_HEADER_op_mask = 0x000000FF
SDMA_PKT_PTEPDE_RMW_HEADER_op_shift = 0
SDMA_PKT_PTEPDE_RMW_HEADER_OP = lambda x: (((x) & SDMA_PKT_PTEPDE_RMW_HEADER_op_mask) << SDMA_PKT_PTEPDE_RMW_HEADER_op_shift)
SDMA_PKT_PTEPDE_RMW_HEADER_sub_op_offset = 0
SDMA_PKT_PTEPDE_RMW_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_PTEPDE_RMW_HEADER_sub_op_shift = 8
SDMA_PKT_PTEPDE_RMW_HEADER_SUB_OP = lambda x: (((x) & SDMA_PKT_PTEPDE_RMW_HEADER_sub_op_mask) << SDMA_PKT_PTEPDE_RMW_HEADER_sub_op_shift)
SDMA_PKT_PTEPDE_RMW_HEADER_mtype_offset = 0
SDMA_PKT_PTEPDE_RMW_HEADER_mtype_mask = 0x00000007
SDMA_PKT_PTEPDE_RMW_HEADER_mtype_shift = 16
SDMA_PKT_PTEPDE_RMW_HEADER_MTYPE = lambda x: (((x) & SDMA_PKT_PTEPDE_RMW_HEADER_mtype_mask) << SDMA_PKT_PTEPDE_RMW_HEADER_mtype_shift)
SDMA_PKT_PTEPDE_RMW_HEADER_gcc_offset = 0
SDMA_PKT_PTEPDE_RMW_HEADER_gcc_mask = 0x00000001
SDMA_PKT_PTEPDE_RMW_HEADER_gcc_shift = 19
SDMA_PKT_PTEPDE_RMW_HEADER_GCC = lambda x: (((x) & SDMA_PKT_PTEPDE_RMW_HEADER_gcc_mask) << SDMA_PKT_PTEPDE_RMW_HEADER_gcc_shift)
SDMA_PKT_PTEPDE_RMW_HEADER_sys_offset = 0
SDMA_PKT_PTEPDE_RMW_HEADER_sys_mask = 0x00000001
SDMA_PKT_PTEPDE_RMW_HEADER_sys_shift = 20
SDMA_PKT_PTEPDE_RMW_HEADER_SYS = lambda x: (((x) & SDMA_PKT_PTEPDE_RMW_HEADER_sys_mask) << SDMA_PKT_PTEPDE_RMW_HEADER_sys_shift)
SDMA_PKT_PTEPDE_RMW_HEADER_snp_offset = 0
SDMA_PKT_PTEPDE_RMW_HEADER_snp_mask = 0x00000001
SDMA_PKT_PTEPDE_RMW_HEADER_snp_shift = 22
SDMA_PKT_PTEPDE_RMW_HEADER_SNP = lambda x: (((x) & SDMA_PKT_PTEPDE_RMW_HEADER_snp_mask) << SDMA_PKT_PTEPDE_RMW_HEADER_snp_shift)
SDMA_PKT_PTEPDE_RMW_HEADER_gpa_offset = 0
SDMA_PKT_PTEPDE_RMW_HEADER_gpa_mask = 0x00000001
SDMA_PKT_PTEPDE_RMW_HEADER_gpa_shift = 23
SDMA_PKT_PTEPDE_RMW_HEADER_GPA = lambda x: (((x) & SDMA_PKT_PTEPDE_RMW_HEADER_gpa_mask) << SDMA_PKT_PTEPDE_RMW_HEADER_gpa_shift)
SDMA_PKT_PTEPDE_RMW_HEADER_l2_policy_offset = 0
SDMA_PKT_PTEPDE_RMW_HEADER_l2_policy_mask = 0x00000003
SDMA_PKT_PTEPDE_RMW_HEADER_l2_policy_shift = 24
SDMA_PKT_PTEPDE_RMW_HEADER_L2_POLICY = lambda x: (((x) & SDMA_PKT_PTEPDE_RMW_HEADER_l2_policy_mask) << SDMA_PKT_PTEPDE_RMW_HEADER_l2_policy_shift)
SDMA_PKT_PTEPDE_RMW_HEADER_llc_policy_offset = 0
SDMA_PKT_PTEPDE_RMW_HEADER_llc_policy_mask = 0x00000001
SDMA_PKT_PTEPDE_RMW_HEADER_llc_policy_shift = 26
SDMA_PKT_PTEPDE_RMW_HEADER_LLC_POLICY = lambda x: (((x) & SDMA_PKT_PTEPDE_RMW_HEADER_llc_policy_mask) << SDMA_PKT_PTEPDE_RMW_HEADER_llc_policy_shift)
SDMA_PKT_PTEPDE_RMW_HEADER_cpv_offset = 0
SDMA_PKT_PTEPDE_RMW_HEADER_cpv_mask = 0x00000001
SDMA_PKT_PTEPDE_RMW_HEADER_cpv_shift = 28
SDMA_PKT_PTEPDE_RMW_HEADER_CPV = lambda x: (((x) & SDMA_PKT_PTEPDE_RMW_HEADER_cpv_mask) << SDMA_PKT_PTEPDE_RMW_HEADER_cpv_shift)
SDMA_PKT_PTEPDE_RMW_ADDR_LO_addr_31_0_offset = 1
SDMA_PKT_PTEPDE_RMW_ADDR_LO_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_PTEPDE_RMW_ADDR_LO_addr_31_0_shift = 0
SDMA_PKT_PTEPDE_RMW_ADDR_LO_ADDR_31_0 = lambda x: (((x) & SDMA_PKT_PTEPDE_RMW_ADDR_LO_addr_31_0_mask) << SDMA_PKT_PTEPDE_RMW_ADDR_LO_addr_31_0_shift)
SDMA_PKT_PTEPDE_RMW_ADDR_HI_addr_63_32_offset = 2
SDMA_PKT_PTEPDE_RMW_ADDR_HI_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_PTEPDE_RMW_ADDR_HI_addr_63_32_shift = 0
SDMA_PKT_PTEPDE_RMW_ADDR_HI_ADDR_63_32 = lambda x: (((x) & SDMA_PKT_PTEPDE_RMW_ADDR_HI_addr_63_32_mask) << SDMA_PKT_PTEPDE_RMW_ADDR_HI_addr_63_32_shift)
SDMA_PKT_PTEPDE_RMW_MASK_LO_mask_31_0_offset = 3
SDMA_PKT_PTEPDE_RMW_MASK_LO_mask_31_0_mask = 0xFFFFFFFF
SDMA_PKT_PTEPDE_RMW_MASK_LO_mask_31_0_shift = 0
SDMA_PKT_PTEPDE_RMW_MASK_LO_MASK_31_0 = lambda x: (((x) & SDMA_PKT_PTEPDE_RMW_MASK_LO_mask_31_0_mask) << SDMA_PKT_PTEPDE_RMW_MASK_LO_mask_31_0_shift)
SDMA_PKT_PTEPDE_RMW_MASK_HI_mask_63_32_offset = 4
SDMA_PKT_PTEPDE_RMW_MASK_HI_mask_63_32_mask = 0xFFFFFFFF
SDMA_PKT_PTEPDE_RMW_MASK_HI_mask_63_32_shift = 0
SDMA_PKT_PTEPDE_RMW_MASK_HI_MASK_63_32 = lambda x: (((x) & SDMA_PKT_PTEPDE_RMW_MASK_HI_mask_63_32_mask) << SDMA_PKT_PTEPDE_RMW_MASK_HI_mask_63_32_shift)
SDMA_PKT_PTEPDE_RMW_VALUE_LO_value_31_0_offset = 5
SDMA_PKT_PTEPDE_RMW_VALUE_LO_value_31_0_mask = 0xFFFFFFFF
SDMA_PKT_PTEPDE_RMW_VALUE_LO_value_31_0_shift = 0
SDMA_PKT_PTEPDE_RMW_VALUE_LO_VALUE_31_0 = lambda x: (((x) & SDMA_PKT_PTEPDE_RMW_VALUE_LO_value_31_0_mask) << SDMA_PKT_PTEPDE_RMW_VALUE_LO_value_31_0_shift)
SDMA_PKT_PTEPDE_RMW_VALUE_HI_value_63_32_offset = 6
SDMA_PKT_PTEPDE_RMW_VALUE_HI_value_63_32_mask = 0xFFFFFFFF
SDMA_PKT_PTEPDE_RMW_VALUE_HI_value_63_32_shift = 0
SDMA_PKT_PTEPDE_RMW_VALUE_HI_VALUE_63_32 = lambda x: (((x) & SDMA_PKT_PTEPDE_RMW_VALUE_HI_value_63_32_mask) << SDMA_PKT_PTEPDE_RMW_VALUE_HI_value_63_32_shift)
SDMA_PKT_PTEPDE_RMW_COUNT_num_of_pte_offset = 7
SDMA_PKT_PTEPDE_RMW_COUNT_num_of_pte_mask = 0xFFFFFFFF
SDMA_PKT_PTEPDE_RMW_COUNT_num_of_pte_shift = 0
SDMA_PKT_PTEPDE_RMW_COUNT_NUM_OF_PTE = lambda x: (((x) & SDMA_PKT_PTEPDE_RMW_COUNT_num_of_pte_mask) << SDMA_PKT_PTEPDE_RMW_COUNT_num_of_pte_shift)
SDMA_PKT_REGISTER_RMW_HEADER_op_offset = 0
SDMA_PKT_REGISTER_RMW_HEADER_op_mask = 0x000000FF
SDMA_PKT_REGISTER_RMW_HEADER_op_shift = 0
SDMA_PKT_REGISTER_RMW_HEADER_OP = lambda x: (((x) & SDMA_PKT_REGISTER_RMW_HEADER_op_mask) << SDMA_PKT_REGISTER_RMW_HEADER_op_shift)
SDMA_PKT_REGISTER_RMW_HEADER_sub_op_offset = 0
SDMA_PKT_REGISTER_RMW_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_REGISTER_RMW_HEADER_sub_op_shift = 8
SDMA_PKT_REGISTER_RMW_HEADER_SUB_OP = lambda x: (((x) & SDMA_PKT_REGISTER_RMW_HEADER_sub_op_mask) << SDMA_PKT_REGISTER_RMW_HEADER_sub_op_shift)
SDMA_PKT_REGISTER_RMW_ADDR_addr_offset = 1
SDMA_PKT_REGISTER_RMW_ADDR_addr_mask = 0x000FFFFF
SDMA_PKT_REGISTER_RMW_ADDR_addr_shift = 0
SDMA_PKT_REGISTER_RMW_ADDR_ADDR = lambda x: (((x) & SDMA_PKT_REGISTER_RMW_ADDR_addr_mask) << SDMA_PKT_REGISTER_RMW_ADDR_addr_shift)
SDMA_PKT_REGISTER_RMW_ADDR_aperture_id_offset = 1
SDMA_PKT_REGISTER_RMW_ADDR_aperture_id_mask = 0x00000FFF
SDMA_PKT_REGISTER_RMW_ADDR_aperture_id_shift = 20
SDMA_PKT_REGISTER_RMW_ADDR_APERTURE_ID = lambda x: (((x) & SDMA_PKT_REGISTER_RMW_ADDR_aperture_id_mask) << SDMA_PKT_REGISTER_RMW_ADDR_aperture_id_shift)
SDMA_PKT_REGISTER_RMW_MASK_mask_offset = 2
SDMA_PKT_REGISTER_RMW_MASK_mask_mask = 0xFFFFFFFF
SDMA_PKT_REGISTER_RMW_MASK_mask_shift = 0
SDMA_PKT_REGISTER_RMW_MASK_MASK = lambda x: (((x) & SDMA_PKT_REGISTER_RMW_MASK_mask_mask) << SDMA_PKT_REGISTER_RMW_MASK_mask_shift)
SDMA_PKT_REGISTER_RMW_VALUE_value_offset = 3
SDMA_PKT_REGISTER_RMW_VALUE_value_mask = 0xFFFFFFFF
SDMA_PKT_REGISTER_RMW_VALUE_value_shift = 0
SDMA_PKT_REGISTER_RMW_VALUE_VALUE = lambda x: (((x) & SDMA_PKT_REGISTER_RMW_VALUE_value_mask) << SDMA_PKT_REGISTER_RMW_VALUE_value_shift)
SDMA_PKT_REGISTER_RMW_MISC_stride_offset = 4
SDMA_PKT_REGISTER_RMW_MISC_stride_mask = 0x000FFFFF
SDMA_PKT_REGISTER_RMW_MISC_stride_shift = 0
SDMA_PKT_REGISTER_RMW_MISC_STRIDE = lambda x: (((x) & SDMA_PKT_REGISTER_RMW_MISC_stride_mask) << SDMA_PKT_REGISTER_RMW_MISC_stride_shift)
SDMA_PKT_REGISTER_RMW_MISC_num_of_reg_offset = 4
SDMA_PKT_REGISTER_RMW_MISC_num_of_reg_mask = 0x00000FFF
SDMA_PKT_REGISTER_RMW_MISC_num_of_reg_shift = 20
SDMA_PKT_REGISTER_RMW_MISC_NUM_OF_REG = lambda x: (((x) & SDMA_PKT_REGISTER_RMW_MISC_num_of_reg_mask) << SDMA_PKT_REGISTER_RMW_MISC_num_of_reg_shift)
SDMA_PKT_WRITE_INCR_HEADER_op_offset = 0
SDMA_PKT_WRITE_INCR_HEADER_op_mask = 0x000000FF
SDMA_PKT_WRITE_INCR_HEADER_op_shift = 0
SDMA_PKT_WRITE_INCR_HEADER_OP = lambda x: (((x) & SDMA_PKT_WRITE_INCR_HEADER_op_mask) << SDMA_PKT_WRITE_INCR_HEADER_op_shift)
SDMA_PKT_WRITE_INCR_HEADER_sub_op_offset = 0
SDMA_PKT_WRITE_INCR_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_WRITE_INCR_HEADER_sub_op_shift = 8
SDMA_PKT_WRITE_INCR_HEADER_SUB_OP = lambda x: (((x) & SDMA_PKT_WRITE_INCR_HEADER_sub_op_mask) << SDMA_PKT_WRITE_INCR_HEADER_sub_op_shift)
SDMA_PKT_WRITE_INCR_HEADER_cache_policy_offset = 0
SDMA_PKT_WRITE_INCR_HEADER_cache_policy_mask = 0x00000007
SDMA_PKT_WRITE_INCR_HEADER_cache_policy_shift = 24
SDMA_PKT_WRITE_INCR_HEADER_CACHE_POLICY = lambda x: (((x) & SDMA_PKT_WRITE_INCR_HEADER_cache_policy_mask) << SDMA_PKT_WRITE_INCR_HEADER_cache_policy_shift)
SDMA_PKT_WRITE_INCR_HEADER_cpv_offset = 0
SDMA_PKT_WRITE_INCR_HEADER_cpv_mask = 0x00000001
SDMA_PKT_WRITE_INCR_HEADER_cpv_shift = 28
SDMA_PKT_WRITE_INCR_HEADER_CPV = lambda x: (((x) & SDMA_PKT_WRITE_INCR_HEADER_cpv_mask) << SDMA_PKT_WRITE_INCR_HEADER_cpv_shift)
SDMA_PKT_WRITE_INCR_DST_ADDR_LO_dst_addr_31_0_offset = 1
SDMA_PKT_WRITE_INCR_DST_ADDR_LO_dst_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_WRITE_INCR_DST_ADDR_LO_dst_addr_31_0_shift = 0
SDMA_PKT_WRITE_INCR_DST_ADDR_LO_DST_ADDR_31_0 = lambda x: (((x) & SDMA_PKT_WRITE_INCR_DST_ADDR_LO_dst_addr_31_0_mask) << SDMA_PKT_WRITE_INCR_DST_ADDR_LO_dst_addr_31_0_shift)
SDMA_PKT_WRITE_INCR_DST_ADDR_HI_dst_addr_63_32_offset = 2
SDMA_PKT_WRITE_INCR_DST_ADDR_HI_dst_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_WRITE_INCR_DST_ADDR_HI_dst_addr_63_32_shift = 0
SDMA_PKT_WRITE_INCR_DST_ADDR_HI_DST_ADDR_63_32 = lambda x: (((x) & SDMA_PKT_WRITE_INCR_DST_ADDR_HI_dst_addr_63_32_mask) << SDMA_PKT_WRITE_INCR_DST_ADDR_HI_dst_addr_63_32_shift)
SDMA_PKT_WRITE_INCR_MASK_DW0_mask_dw0_offset = 3
SDMA_PKT_WRITE_INCR_MASK_DW0_mask_dw0_mask = 0xFFFFFFFF
SDMA_PKT_WRITE_INCR_MASK_DW0_mask_dw0_shift = 0
SDMA_PKT_WRITE_INCR_MASK_DW0_MASK_DW0 = lambda x: (((x) & SDMA_PKT_WRITE_INCR_MASK_DW0_mask_dw0_mask) << SDMA_PKT_WRITE_INCR_MASK_DW0_mask_dw0_shift)
SDMA_PKT_WRITE_INCR_MASK_DW1_mask_dw1_offset = 4
SDMA_PKT_WRITE_INCR_MASK_DW1_mask_dw1_mask = 0xFFFFFFFF
SDMA_PKT_WRITE_INCR_MASK_DW1_mask_dw1_shift = 0
SDMA_PKT_WRITE_INCR_MASK_DW1_MASK_DW1 = lambda x: (((x) & SDMA_PKT_WRITE_INCR_MASK_DW1_mask_dw1_mask) << SDMA_PKT_WRITE_INCR_MASK_DW1_mask_dw1_shift)
SDMA_PKT_WRITE_INCR_INIT_DW0_init_dw0_offset = 5
SDMA_PKT_WRITE_INCR_INIT_DW0_init_dw0_mask = 0xFFFFFFFF
SDMA_PKT_WRITE_INCR_INIT_DW0_init_dw0_shift = 0
SDMA_PKT_WRITE_INCR_INIT_DW0_INIT_DW0 = lambda x: (((x) & SDMA_PKT_WRITE_INCR_INIT_DW0_init_dw0_mask) << SDMA_PKT_WRITE_INCR_INIT_DW0_init_dw0_shift)
SDMA_PKT_WRITE_INCR_INIT_DW1_init_dw1_offset = 6
SDMA_PKT_WRITE_INCR_INIT_DW1_init_dw1_mask = 0xFFFFFFFF
SDMA_PKT_WRITE_INCR_INIT_DW1_init_dw1_shift = 0
SDMA_PKT_WRITE_INCR_INIT_DW1_INIT_DW1 = lambda x: (((x) & SDMA_PKT_WRITE_INCR_INIT_DW1_init_dw1_mask) << SDMA_PKT_WRITE_INCR_INIT_DW1_init_dw1_shift)
SDMA_PKT_WRITE_INCR_INCR_DW0_incr_dw0_offset = 7
SDMA_PKT_WRITE_INCR_INCR_DW0_incr_dw0_mask = 0xFFFFFFFF
SDMA_PKT_WRITE_INCR_INCR_DW0_incr_dw0_shift = 0
SDMA_PKT_WRITE_INCR_INCR_DW0_INCR_DW0 = lambda x: (((x) & SDMA_PKT_WRITE_INCR_INCR_DW0_incr_dw0_mask) << SDMA_PKT_WRITE_INCR_INCR_DW0_incr_dw0_shift)
SDMA_PKT_WRITE_INCR_INCR_DW1_incr_dw1_offset = 8
SDMA_PKT_WRITE_INCR_INCR_DW1_incr_dw1_mask = 0xFFFFFFFF
SDMA_PKT_WRITE_INCR_INCR_DW1_incr_dw1_shift = 0
SDMA_PKT_WRITE_INCR_INCR_DW1_INCR_DW1 = lambda x: (((x) & SDMA_PKT_WRITE_INCR_INCR_DW1_incr_dw1_mask) << SDMA_PKT_WRITE_INCR_INCR_DW1_incr_dw1_shift)
SDMA_PKT_WRITE_INCR_COUNT_count_offset = 9
SDMA_PKT_WRITE_INCR_COUNT_count_mask = 0x0007FFFF
SDMA_PKT_WRITE_INCR_COUNT_count_shift = 0
SDMA_PKT_WRITE_INCR_COUNT_COUNT = lambda x: (((x) & SDMA_PKT_WRITE_INCR_COUNT_count_mask) << SDMA_PKT_WRITE_INCR_COUNT_count_shift)
SDMA_PKT_INDIRECT_HEADER_op_offset = 0
SDMA_PKT_INDIRECT_HEADER_op_mask = 0x000000FF
SDMA_PKT_INDIRECT_HEADER_op_shift = 0
SDMA_PKT_INDIRECT_HEADER_OP = lambda x: (((x) & SDMA_PKT_INDIRECT_HEADER_op_mask) << SDMA_PKT_INDIRECT_HEADER_op_shift)
SDMA_PKT_INDIRECT_HEADER_sub_op_offset = 0
SDMA_PKT_INDIRECT_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_INDIRECT_HEADER_sub_op_shift = 8
SDMA_PKT_INDIRECT_HEADER_SUB_OP = lambda x: (((x) & SDMA_PKT_INDIRECT_HEADER_sub_op_mask) << SDMA_PKT_INDIRECT_HEADER_sub_op_shift)
SDMA_PKT_INDIRECT_HEADER_vmid_offset = 0
SDMA_PKT_INDIRECT_HEADER_vmid_mask = 0x0000000F
SDMA_PKT_INDIRECT_HEADER_vmid_shift = 16
SDMA_PKT_INDIRECT_HEADER_VMID = lambda x: (((x) & SDMA_PKT_INDIRECT_HEADER_vmid_mask) << SDMA_PKT_INDIRECT_HEADER_vmid_shift)
SDMA_PKT_INDIRECT_HEADER_priv_offset = 0
SDMA_PKT_INDIRECT_HEADER_priv_mask = 0x00000001
SDMA_PKT_INDIRECT_HEADER_priv_shift = 31
SDMA_PKT_INDIRECT_HEADER_PRIV = lambda x: (((x) & SDMA_PKT_INDIRECT_HEADER_priv_mask) << SDMA_PKT_INDIRECT_HEADER_priv_shift)
SDMA_PKT_INDIRECT_BASE_LO_ib_base_31_0_offset = 1
SDMA_PKT_INDIRECT_BASE_LO_ib_base_31_0_mask = 0xFFFFFFFF
SDMA_PKT_INDIRECT_BASE_LO_ib_base_31_0_shift = 0
SDMA_PKT_INDIRECT_BASE_LO_IB_BASE_31_0 = lambda x: (((x) & SDMA_PKT_INDIRECT_BASE_LO_ib_base_31_0_mask) << SDMA_PKT_INDIRECT_BASE_LO_ib_base_31_0_shift)
SDMA_PKT_INDIRECT_BASE_HI_ib_base_63_32_offset = 2
SDMA_PKT_INDIRECT_BASE_HI_ib_base_63_32_mask = 0xFFFFFFFF
SDMA_PKT_INDIRECT_BASE_HI_ib_base_63_32_shift = 0
SDMA_PKT_INDIRECT_BASE_HI_IB_BASE_63_32 = lambda x: (((x) & SDMA_PKT_INDIRECT_BASE_HI_ib_base_63_32_mask) << SDMA_PKT_INDIRECT_BASE_HI_ib_base_63_32_shift)
SDMA_PKT_INDIRECT_IB_SIZE_ib_size_offset = 3
SDMA_PKT_INDIRECT_IB_SIZE_ib_size_mask = 0x000FFFFF
SDMA_PKT_INDIRECT_IB_SIZE_ib_size_shift = 0
SDMA_PKT_INDIRECT_IB_SIZE_IB_SIZE = lambda x: (((x) & SDMA_PKT_INDIRECT_IB_SIZE_ib_size_mask) << SDMA_PKT_INDIRECT_IB_SIZE_ib_size_shift)
SDMA_PKT_INDIRECT_CSA_ADDR_LO_csa_addr_31_0_offset = 4
SDMA_PKT_INDIRECT_CSA_ADDR_LO_csa_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_INDIRECT_CSA_ADDR_LO_csa_addr_31_0_shift = 0
SDMA_PKT_INDIRECT_CSA_ADDR_LO_CSA_ADDR_31_0 = lambda x: (((x) & SDMA_PKT_INDIRECT_CSA_ADDR_LO_csa_addr_31_0_mask) << SDMA_PKT_INDIRECT_CSA_ADDR_LO_csa_addr_31_0_shift)
SDMA_PKT_INDIRECT_CSA_ADDR_HI_csa_addr_63_32_offset = 5
SDMA_PKT_INDIRECT_CSA_ADDR_HI_csa_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_INDIRECT_CSA_ADDR_HI_csa_addr_63_32_shift = 0
SDMA_PKT_INDIRECT_CSA_ADDR_HI_CSA_ADDR_63_32 = lambda x: (((x) & SDMA_PKT_INDIRECT_CSA_ADDR_HI_csa_addr_63_32_mask) << SDMA_PKT_INDIRECT_CSA_ADDR_HI_csa_addr_63_32_shift)
SDMA_PKT_SEMAPHORE_HEADER_op_offset = 0
SDMA_PKT_SEMAPHORE_HEADER_op_mask = 0x000000FF
SDMA_PKT_SEMAPHORE_HEADER_op_shift = 0
SDMA_PKT_SEMAPHORE_HEADER_OP = lambda x: (((x) & SDMA_PKT_SEMAPHORE_HEADER_op_mask) << SDMA_PKT_SEMAPHORE_HEADER_op_shift)
SDMA_PKT_SEMAPHORE_HEADER_sub_op_offset = 0
SDMA_PKT_SEMAPHORE_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_SEMAPHORE_HEADER_sub_op_shift = 8
SDMA_PKT_SEMAPHORE_HEADER_SUB_OP = lambda x: (((x) & SDMA_PKT_SEMAPHORE_HEADER_sub_op_mask) << SDMA_PKT_SEMAPHORE_HEADER_sub_op_shift)
SDMA_PKT_SEMAPHORE_HEADER_write_one_offset = 0
SDMA_PKT_SEMAPHORE_HEADER_write_one_mask = 0x00000001
SDMA_PKT_SEMAPHORE_HEADER_write_one_shift = 29
SDMA_PKT_SEMAPHORE_HEADER_WRITE_ONE = lambda x: (((x) & SDMA_PKT_SEMAPHORE_HEADER_write_one_mask) << SDMA_PKT_SEMAPHORE_HEADER_write_one_shift)
SDMA_PKT_SEMAPHORE_HEADER_signal_offset = 0
SDMA_PKT_SEMAPHORE_HEADER_signal_mask = 0x00000001
SDMA_PKT_SEMAPHORE_HEADER_signal_shift = 30
SDMA_PKT_SEMAPHORE_HEADER_SIGNAL = lambda x: (((x) & SDMA_PKT_SEMAPHORE_HEADER_signal_mask) << SDMA_PKT_SEMAPHORE_HEADER_signal_shift)
SDMA_PKT_SEMAPHORE_HEADER_mailbox_offset = 0
SDMA_PKT_SEMAPHORE_HEADER_mailbox_mask = 0x00000001
SDMA_PKT_SEMAPHORE_HEADER_mailbox_shift = 31
SDMA_PKT_SEMAPHORE_HEADER_MAILBOX = lambda x: (((x) & SDMA_PKT_SEMAPHORE_HEADER_mailbox_mask) << SDMA_PKT_SEMAPHORE_HEADER_mailbox_shift)
SDMA_PKT_SEMAPHORE_ADDR_LO_addr_31_0_offset = 1
SDMA_PKT_SEMAPHORE_ADDR_LO_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_SEMAPHORE_ADDR_LO_addr_31_0_shift = 0
SDMA_PKT_SEMAPHORE_ADDR_LO_ADDR_31_0 = lambda x: (((x) & SDMA_PKT_SEMAPHORE_ADDR_LO_addr_31_0_mask) << SDMA_PKT_SEMAPHORE_ADDR_LO_addr_31_0_shift)
SDMA_PKT_SEMAPHORE_ADDR_HI_addr_63_32_offset = 2
SDMA_PKT_SEMAPHORE_ADDR_HI_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_SEMAPHORE_ADDR_HI_addr_63_32_shift = 0
SDMA_PKT_SEMAPHORE_ADDR_HI_ADDR_63_32 = lambda x: (((x) & SDMA_PKT_SEMAPHORE_ADDR_HI_addr_63_32_mask) << SDMA_PKT_SEMAPHORE_ADDR_HI_addr_63_32_shift)
SDMA_PKT_MEM_INCR_HEADER_op_offset = 0
SDMA_PKT_MEM_INCR_HEADER_op_mask = 0x000000FF
SDMA_PKT_MEM_INCR_HEADER_op_shift = 0
SDMA_PKT_MEM_INCR_HEADER_OP = lambda x: (((x) & SDMA_PKT_MEM_INCR_HEADER_op_mask) << SDMA_PKT_MEM_INCR_HEADER_op_shift)
SDMA_PKT_MEM_INCR_HEADER_sub_op_offset = 0
SDMA_PKT_MEM_INCR_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_MEM_INCR_HEADER_sub_op_shift = 8
SDMA_PKT_MEM_INCR_HEADER_SUB_OP = lambda x: (((x) & SDMA_PKT_MEM_INCR_HEADER_sub_op_mask) << SDMA_PKT_MEM_INCR_HEADER_sub_op_shift)
SDMA_PKT_MEM_INCR_HEADER_l2_policy_offset = 0
SDMA_PKT_MEM_INCR_HEADER_l2_policy_mask = 0x00000003
SDMA_PKT_MEM_INCR_HEADER_l2_policy_shift = 24
SDMA_PKT_MEM_INCR_HEADER_L2_POLICY = lambda x: (((x) & SDMA_PKT_MEM_INCR_HEADER_l2_policy_mask) << SDMA_PKT_MEM_INCR_HEADER_l2_policy_shift)
SDMA_PKT_MEM_INCR_HEADER_llc_policy_offset = 0
SDMA_PKT_MEM_INCR_HEADER_llc_policy_mask = 0x00000001
SDMA_PKT_MEM_INCR_HEADER_llc_policy_shift = 26
SDMA_PKT_MEM_INCR_HEADER_LLC_POLICY = lambda x: (((x) & SDMA_PKT_MEM_INCR_HEADER_llc_policy_mask) << SDMA_PKT_MEM_INCR_HEADER_llc_policy_shift)
SDMA_PKT_MEM_INCR_HEADER_cpv_offset = 0
SDMA_PKT_MEM_INCR_HEADER_cpv_mask = 0x00000001
SDMA_PKT_MEM_INCR_HEADER_cpv_shift = 28
SDMA_PKT_MEM_INCR_HEADER_CPV = lambda x: (((x) & SDMA_PKT_MEM_INCR_HEADER_cpv_mask) << SDMA_PKT_MEM_INCR_HEADER_cpv_shift)
SDMA_PKT_MEM_INCR_ADDR_LO_addr_31_0_offset = 1
SDMA_PKT_MEM_INCR_ADDR_LO_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_MEM_INCR_ADDR_LO_addr_31_0_shift = 0
SDMA_PKT_MEM_INCR_ADDR_LO_ADDR_31_0 = lambda x: (((x) & SDMA_PKT_MEM_INCR_ADDR_LO_addr_31_0_mask) << SDMA_PKT_MEM_INCR_ADDR_LO_addr_31_0_shift)
SDMA_PKT_MEM_INCR_ADDR_HI_addr_63_32_offset = 2
SDMA_PKT_MEM_INCR_ADDR_HI_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_MEM_INCR_ADDR_HI_addr_63_32_shift = 0
SDMA_PKT_MEM_INCR_ADDR_HI_ADDR_63_32 = lambda x: (((x) & SDMA_PKT_MEM_INCR_ADDR_HI_addr_63_32_mask) << SDMA_PKT_MEM_INCR_ADDR_HI_addr_63_32_shift)
SDMA_PKT_VM_INVALIDATION_HEADER_op_offset = 0
SDMA_PKT_VM_INVALIDATION_HEADER_op_mask = 0x000000FF
SDMA_PKT_VM_INVALIDATION_HEADER_op_shift = 0
SDMA_PKT_VM_INVALIDATION_HEADER_OP = lambda x: (((x) & SDMA_PKT_VM_INVALIDATION_HEADER_op_mask) << SDMA_PKT_VM_INVALIDATION_HEADER_op_shift)
SDMA_PKT_VM_INVALIDATION_HEADER_sub_op_offset = 0
SDMA_PKT_VM_INVALIDATION_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_VM_INVALIDATION_HEADER_sub_op_shift = 8
SDMA_PKT_VM_INVALIDATION_HEADER_SUB_OP = lambda x: (((x) & SDMA_PKT_VM_INVALIDATION_HEADER_sub_op_mask) << SDMA_PKT_VM_INVALIDATION_HEADER_sub_op_shift)
SDMA_PKT_VM_INVALIDATION_HEADER_gfx_eng_id_offset = 0
SDMA_PKT_VM_INVALIDATION_HEADER_gfx_eng_id_mask = 0x0000001F
SDMA_PKT_VM_INVALIDATION_HEADER_gfx_eng_id_shift = 16
SDMA_PKT_VM_INVALIDATION_HEADER_GFX_ENG_ID = lambda x: (((x) & SDMA_PKT_VM_INVALIDATION_HEADER_gfx_eng_id_mask) << SDMA_PKT_VM_INVALIDATION_HEADER_gfx_eng_id_shift)
SDMA_PKT_VM_INVALIDATION_HEADER_mm_eng_id_offset = 0
SDMA_PKT_VM_INVALIDATION_HEADER_mm_eng_id_mask = 0x0000001F
SDMA_PKT_VM_INVALIDATION_HEADER_mm_eng_id_shift = 24
SDMA_PKT_VM_INVALIDATION_HEADER_MM_ENG_ID = lambda x: (((x) & SDMA_PKT_VM_INVALIDATION_HEADER_mm_eng_id_mask) << SDMA_PKT_VM_INVALIDATION_HEADER_mm_eng_id_shift)
SDMA_PKT_VM_INVALIDATION_INVALIDATEREQ_invalidatereq_offset = 1
SDMA_PKT_VM_INVALIDATION_INVALIDATEREQ_invalidatereq_mask = 0xFFFFFFFF
SDMA_PKT_VM_INVALIDATION_INVALIDATEREQ_invalidatereq_shift = 0
SDMA_PKT_VM_INVALIDATION_INVALIDATEREQ_INVALIDATEREQ = lambda x: (((x) & SDMA_PKT_VM_INVALIDATION_INVALIDATEREQ_invalidatereq_mask) << SDMA_PKT_VM_INVALIDATION_INVALIDATEREQ_invalidatereq_shift)
SDMA_PKT_VM_INVALIDATION_ADDRESSRANGELO_addressrangelo_offset = 2
SDMA_PKT_VM_INVALIDATION_ADDRESSRANGELO_addressrangelo_mask = 0xFFFFFFFF
SDMA_PKT_VM_INVALIDATION_ADDRESSRANGELO_addressrangelo_shift = 0
SDMA_PKT_VM_INVALIDATION_ADDRESSRANGELO_ADDRESSRANGELO = lambda x: (((x) & SDMA_PKT_VM_INVALIDATION_ADDRESSRANGELO_addressrangelo_mask) << SDMA_PKT_VM_INVALIDATION_ADDRESSRANGELO_addressrangelo_shift)
SDMA_PKT_VM_INVALIDATION_ADDRESSRANGEHI_invalidateack_offset = 3
SDMA_PKT_VM_INVALIDATION_ADDRESSRANGEHI_invalidateack_mask = 0x0000FFFF
SDMA_PKT_VM_INVALIDATION_ADDRESSRANGEHI_invalidateack_shift = 0
SDMA_PKT_VM_INVALIDATION_ADDRESSRANGEHI_INVALIDATEACK = lambda x: (((x) & SDMA_PKT_VM_INVALIDATION_ADDRESSRANGEHI_invalidateack_mask) << SDMA_PKT_VM_INVALIDATION_ADDRESSRANGEHI_invalidateack_shift)
SDMA_PKT_VM_INVALIDATION_ADDRESSRANGEHI_addressrangehi_offset = 3
SDMA_PKT_VM_INVALIDATION_ADDRESSRANGEHI_addressrangehi_mask = 0x0000001F
SDMA_PKT_VM_INVALIDATION_ADDRESSRANGEHI_addressrangehi_shift = 16
SDMA_PKT_VM_INVALIDATION_ADDRESSRANGEHI_ADDRESSRANGEHI = lambda x: (((x) & SDMA_PKT_VM_INVALIDATION_ADDRESSRANGEHI_addressrangehi_mask) << SDMA_PKT_VM_INVALIDATION_ADDRESSRANGEHI_addressrangehi_shift)
SDMA_PKT_VM_INVALIDATION_ADDRESSRANGEHI_reserved_offset = 3
SDMA_PKT_VM_INVALIDATION_ADDRESSRANGEHI_reserved_mask = 0x000001FF
SDMA_PKT_VM_INVALIDATION_ADDRESSRANGEHI_reserved_shift = 23
SDMA_PKT_VM_INVALIDATION_ADDRESSRANGEHI_RESERVED = lambda x: (((x) & SDMA_PKT_VM_INVALIDATION_ADDRESSRANGEHI_reserved_mask) << SDMA_PKT_VM_INVALIDATION_ADDRESSRANGEHI_reserved_shift)
SDMA_PKT_FENCE_HEADER_op_offset = 0
SDMA_PKT_FENCE_HEADER_op_mask = 0x000000FF
SDMA_PKT_FENCE_HEADER_op_shift = 0
SDMA_PKT_FENCE_HEADER_OP = lambda x: (((x) & SDMA_PKT_FENCE_HEADER_op_mask) << SDMA_PKT_FENCE_HEADER_op_shift)
SDMA_PKT_FENCE_HEADER_sub_op_offset = 0
SDMA_PKT_FENCE_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_FENCE_HEADER_sub_op_shift = 8
SDMA_PKT_FENCE_HEADER_SUB_OP = lambda x: (((x) & SDMA_PKT_FENCE_HEADER_sub_op_mask) << SDMA_PKT_FENCE_HEADER_sub_op_shift)
SDMA_PKT_FENCE_HEADER_mtype_offset = 0
SDMA_PKT_FENCE_HEADER_mtype_mask = 0x00000007
SDMA_PKT_FENCE_HEADER_mtype_shift = 16
SDMA_PKT_FENCE_HEADER_MTYPE = lambda x: (((x) & SDMA_PKT_FENCE_HEADER_mtype_mask) << SDMA_PKT_FENCE_HEADER_mtype_shift)
SDMA_PKT_FENCE_HEADER_gcc_offset = 0
SDMA_PKT_FENCE_HEADER_gcc_mask = 0x00000001
SDMA_PKT_FENCE_HEADER_gcc_shift = 19
SDMA_PKT_FENCE_HEADER_GCC = lambda x: (((x) & SDMA_PKT_FENCE_HEADER_gcc_mask) << SDMA_PKT_FENCE_HEADER_gcc_shift)
SDMA_PKT_FENCE_HEADER_sys_offset = 0
SDMA_PKT_FENCE_HEADER_sys_mask = 0x00000001
SDMA_PKT_FENCE_HEADER_sys_shift = 20
SDMA_PKT_FENCE_HEADER_SYS = lambda x: (((x) & SDMA_PKT_FENCE_HEADER_sys_mask) << SDMA_PKT_FENCE_HEADER_sys_shift)
SDMA_PKT_FENCE_HEADER_snp_offset = 0
SDMA_PKT_FENCE_HEADER_snp_mask = 0x00000001
SDMA_PKT_FENCE_HEADER_snp_shift = 22
SDMA_PKT_FENCE_HEADER_SNP = lambda x: (((x) & SDMA_PKT_FENCE_HEADER_snp_mask) << SDMA_PKT_FENCE_HEADER_snp_shift)
SDMA_PKT_FENCE_HEADER_gpa_offset = 0
SDMA_PKT_FENCE_HEADER_gpa_mask = 0x00000001
SDMA_PKT_FENCE_HEADER_gpa_shift = 23
SDMA_PKT_FENCE_HEADER_GPA = lambda x: (((x) & SDMA_PKT_FENCE_HEADER_gpa_mask) << SDMA_PKT_FENCE_HEADER_gpa_shift)
SDMA_PKT_FENCE_HEADER_l2_policy_offset = 0
SDMA_PKT_FENCE_HEADER_l2_policy_mask = 0x00000003
SDMA_PKT_FENCE_HEADER_l2_policy_shift = 24
SDMA_PKT_FENCE_HEADER_L2_POLICY = lambda x: (((x) & SDMA_PKT_FENCE_HEADER_l2_policy_mask) << SDMA_PKT_FENCE_HEADER_l2_policy_shift)
SDMA_PKT_FENCE_HEADER_llc_policy_offset = 0
SDMA_PKT_FENCE_HEADER_llc_policy_mask = 0x00000001
SDMA_PKT_FENCE_HEADER_llc_policy_shift = 26
SDMA_PKT_FENCE_HEADER_LLC_POLICY = lambda x: (((x) & SDMA_PKT_FENCE_HEADER_llc_policy_mask) << SDMA_PKT_FENCE_HEADER_llc_policy_shift)
SDMA_PKT_FENCE_HEADER_cpv_offset = 0
SDMA_PKT_FENCE_HEADER_cpv_mask = 0x00000001
SDMA_PKT_FENCE_HEADER_cpv_shift = 28
SDMA_PKT_FENCE_HEADER_CPV = lambda x: (((x) & SDMA_PKT_FENCE_HEADER_cpv_mask) << SDMA_PKT_FENCE_HEADER_cpv_shift)
SDMA_PKT_FENCE_ADDR_LO_addr_31_0_offset = 1
SDMA_PKT_FENCE_ADDR_LO_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_FENCE_ADDR_LO_addr_31_0_shift = 0
SDMA_PKT_FENCE_ADDR_LO_ADDR_31_0 = lambda x: (((x) & SDMA_PKT_FENCE_ADDR_LO_addr_31_0_mask) << SDMA_PKT_FENCE_ADDR_LO_addr_31_0_shift)
SDMA_PKT_FENCE_ADDR_HI_addr_63_32_offset = 2
SDMA_PKT_FENCE_ADDR_HI_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_FENCE_ADDR_HI_addr_63_32_shift = 0
SDMA_PKT_FENCE_ADDR_HI_ADDR_63_32 = lambda x: (((x) & SDMA_PKT_FENCE_ADDR_HI_addr_63_32_mask) << SDMA_PKT_FENCE_ADDR_HI_addr_63_32_shift)
SDMA_PKT_FENCE_DATA_data_offset = 3
SDMA_PKT_FENCE_DATA_data_mask = 0xFFFFFFFF
SDMA_PKT_FENCE_DATA_data_shift = 0
SDMA_PKT_FENCE_DATA_DATA = lambda x: (((x) & SDMA_PKT_FENCE_DATA_data_mask) << SDMA_PKT_FENCE_DATA_data_shift)
SDMA_PKT_SRBM_WRITE_HEADER_op_offset = 0
SDMA_PKT_SRBM_WRITE_HEADER_op_mask = 0x000000FF
SDMA_PKT_SRBM_WRITE_HEADER_op_shift = 0
SDMA_PKT_SRBM_WRITE_HEADER_OP = lambda x: (((x) & SDMA_PKT_SRBM_WRITE_HEADER_op_mask) << SDMA_PKT_SRBM_WRITE_HEADER_op_shift)
SDMA_PKT_SRBM_WRITE_HEADER_sub_op_offset = 0
SDMA_PKT_SRBM_WRITE_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_SRBM_WRITE_HEADER_sub_op_shift = 8
SDMA_PKT_SRBM_WRITE_HEADER_SUB_OP = lambda x: (((x) & SDMA_PKT_SRBM_WRITE_HEADER_sub_op_mask) << SDMA_PKT_SRBM_WRITE_HEADER_sub_op_shift)
SDMA_PKT_SRBM_WRITE_HEADER_byte_en_offset = 0
SDMA_PKT_SRBM_WRITE_HEADER_byte_en_mask = 0x0000000F
SDMA_PKT_SRBM_WRITE_HEADER_byte_en_shift = 28
SDMA_PKT_SRBM_WRITE_HEADER_BYTE_EN = lambda x: (((x) & SDMA_PKT_SRBM_WRITE_HEADER_byte_en_mask) << SDMA_PKT_SRBM_WRITE_HEADER_byte_en_shift)
SDMA_PKT_SRBM_WRITE_ADDR_addr_offset = 1
SDMA_PKT_SRBM_WRITE_ADDR_addr_mask = 0x0003FFFF
SDMA_PKT_SRBM_WRITE_ADDR_addr_shift = 0
SDMA_PKT_SRBM_WRITE_ADDR_ADDR = lambda x: (((x) & SDMA_PKT_SRBM_WRITE_ADDR_addr_mask) << SDMA_PKT_SRBM_WRITE_ADDR_addr_shift)
SDMA_PKT_SRBM_WRITE_ADDR_apertureid_offset = 1
SDMA_PKT_SRBM_WRITE_ADDR_apertureid_mask = 0x00000FFF
SDMA_PKT_SRBM_WRITE_ADDR_apertureid_shift = 20
SDMA_PKT_SRBM_WRITE_ADDR_APERTUREID = lambda x: (((x) & SDMA_PKT_SRBM_WRITE_ADDR_apertureid_mask) << SDMA_PKT_SRBM_WRITE_ADDR_apertureid_shift)
SDMA_PKT_SRBM_WRITE_DATA_data_offset = 2
SDMA_PKT_SRBM_WRITE_DATA_data_mask = 0xFFFFFFFF
SDMA_PKT_SRBM_WRITE_DATA_data_shift = 0
SDMA_PKT_SRBM_WRITE_DATA_DATA = lambda x: (((x) & SDMA_PKT_SRBM_WRITE_DATA_data_mask) << SDMA_PKT_SRBM_WRITE_DATA_data_shift)
SDMA_PKT_PRE_EXE_HEADER_op_offset = 0
SDMA_PKT_PRE_EXE_HEADER_op_mask = 0x000000FF
SDMA_PKT_PRE_EXE_HEADER_op_shift = 0
SDMA_PKT_PRE_EXE_HEADER_OP = lambda x: (((x) & SDMA_PKT_PRE_EXE_HEADER_op_mask) << SDMA_PKT_PRE_EXE_HEADER_op_shift)
SDMA_PKT_PRE_EXE_HEADER_sub_op_offset = 0
SDMA_PKT_PRE_EXE_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_PRE_EXE_HEADER_sub_op_shift = 8
SDMA_PKT_PRE_EXE_HEADER_SUB_OP = lambda x: (((x) & SDMA_PKT_PRE_EXE_HEADER_sub_op_mask) << SDMA_PKT_PRE_EXE_HEADER_sub_op_shift)
SDMA_PKT_PRE_EXE_HEADER_dev_sel_offset = 0
SDMA_PKT_PRE_EXE_HEADER_dev_sel_mask = 0x000000FF
SDMA_PKT_PRE_EXE_HEADER_dev_sel_shift = 16
SDMA_PKT_PRE_EXE_HEADER_DEV_SEL = lambda x: (((x) & SDMA_PKT_PRE_EXE_HEADER_dev_sel_mask) << SDMA_PKT_PRE_EXE_HEADER_dev_sel_shift)
SDMA_PKT_PRE_EXE_EXEC_COUNT_exec_count_offset = 1
SDMA_PKT_PRE_EXE_EXEC_COUNT_exec_count_mask = 0x00003FFF
SDMA_PKT_PRE_EXE_EXEC_COUNT_exec_count_shift = 0
SDMA_PKT_PRE_EXE_EXEC_COUNT_EXEC_COUNT = lambda x: (((x) & SDMA_PKT_PRE_EXE_EXEC_COUNT_exec_count_mask) << SDMA_PKT_PRE_EXE_EXEC_COUNT_exec_count_shift)
SDMA_PKT_COND_EXE_HEADER_op_offset = 0
SDMA_PKT_COND_EXE_HEADER_op_mask = 0x000000FF
SDMA_PKT_COND_EXE_HEADER_op_shift = 0
SDMA_PKT_COND_EXE_HEADER_OP = lambda x: (((x) & SDMA_PKT_COND_EXE_HEADER_op_mask) << SDMA_PKT_COND_EXE_HEADER_op_shift)
SDMA_PKT_COND_EXE_HEADER_sub_op_offset = 0
SDMA_PKT_COND_EXE_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_COND_EXE_HEADER_sub_op_shift = 8
SDMA_PKT_COND_EXE_HEADER_SUB_OP = lambda x: (((x) & SDMA_PKT_COND_EXE_HEADER_sub_op_mask) << SDMA_PKT_COND_EXE_HEADER_sub_op_shift)
SDMA_PKT_COND_EXE_HEADER_cache_policy_offset = 0
SDMA_PKT_COND_EXE_HEADER_cache_policy_mask = 0x00000007
SDMA_PKT_COND_EXE_HEADER_cache_policy_shift = 24
SDMA_PKT_COND_EXE_HEADER_CACHE_POLICY = lambda x: (((x) & SDMA_PKT_COND_EXE_HEADER_cache_policy_mask) << SDMA_PKT_COND_EXE_HEADER_cache_policy_shift)
SDMA_PKT_COND_EXE_HEADER_cpv_offset = 0
SDMA_PKT_COND_EXE_HEADER_cpv_mask = 0x00000001
SDMA_PKT_COND_EXE_HEADER_cpv_shift = 28
SDMA_PKT_COND_EXE_HEADER_CPV = lambda x: (((x) & SDMA_PKT_COND_EXE_HEADER_cpv_mask) << SDMA_PKT_COND_EXE_HEADER_cpv_shift)
SDMA_PKT_COND_EXE_ADDR_LO_addr_31_0_offset = 1
SDMA_PKT_COND_EXE_ADDR_LO_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_COND_EXE_ADDR_LO_addr_31_0_shift = 0
SDMA_PKT_COND_EXE_ADDR_LO_ADDR_31_0 = lambda x: (((x) & SDMA_PKT_COND_EXE_ADDR_LO_addr_31_0_mask) << SDMA_PKT_COND_EXE_ADDR_LO_addr_31_0_shift)
SDMA_PKT_COND_EXE_ADDR_HI_addr_63_32_offset = 2
SDMA_PKT_COND_EXE_ADDR_HI_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_COND_EXE_ADDR_HI_addr_63_32_shift = 0
SDMA_PKT_COND_EXE_ADDR_HI_ADDR_63_32 = lambda x: (((x) & SDMA_PKT_COND_EXE_ADDR_HI_addr_63_32_mask) << SDMA_PKT_COND_EXE_ADDR_HI_addr_63_32_shift)
SDMA_PKT_COND_EXE_REFERENCE_reference_offset = 3
SDMA_PKT_COND_EXE_REFERENCE_reference_mask = 0xFFFFFFFF
SDMA_PKT_COND_EXE_REFERENCE_reference_shift = 0
SDMA_PKT_COND_EXE_REFERENCE_REFERENCE = lambda x: (((x) & SDMA_PKT_COND_EXE_REFERENCE_reference_mask) << SDMA_PKT_COND_EXE_REFERENCE_reference_shift)
SDMA_PKT_COND_EXE_EXEC_COUNT_exec_count_offset = 4
SDMA_PKT_COND_EXE_EXEC_COUNT_exec_count_mask = 0x00003FFF
SDMA_PKT_COND_EXE_EXEC_COUNT_exec_count_shift = 0
SDMA_PKT_COND_EXE_EXEC_COUNT_EXEC_COUNT = lambda x: (((x) & SDMA_PKT_COND_EXE_EXEC_COUNT_exec_count_mask) << SDMA_PKT_COND_EXE_EXEC_COUNT_exec_count_shift)
SDMA_PKT_CONSTANT_FILL_HEADER_op_offset = 0
SDMA_PKT_CONSTANT_FILL_HEADER_op_mask = 0x000000FF
SDMA_PKT_CONSTANT_FILL_HEADER_op_shift = 0
SDMA_PKT_CONSTANT_FILL_HEADER_OP = lambda x: (((x) & SDMA_PKT_CONSTANT_FILL_HEADER_op_mask) << SDMA_PKT_CONSTANT_FILL_HEADER_op_shift)
SDMA_PKT_CONSTANT_FILL_HEADER_sub_op_offset = 0
SDMA_PKT_CONSTANT_FILL_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_CONSTANT_FILL_HEADER_sub_op_shift = 8
SDMA_PKT_CONSTANT_FILL_HEADER_SUB_OP = lambda x: (((x) & SDMA_PKT_CONSTANT_FILL_HEADER_sub_op_mask) << SDMA_PKT_CONSTANT_FILL_HEADER_sub_op_shift)
SDMA_PKT_CONSTANT_FILL_HEADER_sw_offset = 0
SDMA_PKT_CONSTANT_FILL_HEADER_sw_mask = 0x00000003
SDMA_PKT_CONSTANT_FILL_HEADER_sw_shift = 16
SDMA_PKT_CONSTANT_FILL_HEADER_SW = lambda x: (((x) & SDMA_PKT_CONSTANT_FILL_HEADER_sw_mask) << SDMA_PKT_CONSTANT_FILL_HEADER_sw_shift)
SDMA_PKT_CONSTANT_FILL_HEADER_cache_policy_offset = 0
SDMA_PKT_CONSTANT_FILL_HEADER_cache_policy_mask = 0x00000007
SDMA_PKT_CONSTANT_FILL_HEADER_cache_policy_shift = 24
SDMA_PKT_CONSTANT_FILL_HEADER_CACHE_POLICY = lambda x: (((x) & SDMA_PKT_CONSTANT_FILL_HEADER_cache_policy_mask) << SDMA_PKT_CONSTANT_FILL_HEADER_cache_policy_shift)
SDMA_PKT_CONSTANT_FILL_HEADER_cpv_offset = 0
SDMA_PKT_CONSTANT_FILL_HEADER_cpv_mask = 0x00000001
SDMA_PKT_CONSTANT_FILL_HEADER_cpv_shift = 28
SDMA_PKT_CONSTANT_FILL_HEADER_CPV = lambda x: (((x) & SDMA_PKT_CONSTANT_FILL_HEADER_cpv_mask) << SDMA_PKT_CONSTANT_FILL_HEADER_cpv_shift)
SDMA_PKT_CONSTANT_FILL_HEADER_fillsize_offset = 0
SDMA_PKT_CONSTANT_FILL_HEADER_fillsize_mask = 0x00000003
SDMA_PKT_CONSTANT_FILL_HEADER_fillsize_shift = 30
SDMA_PKT_CONSTANT_FILL_HEADER_FILLSIZE = lambda x: (((x) & SDMA_PKT_CONSTANT_FILL_HEADER_fillsize_mask) << SDMA_PKT_CONSTANT_FILL_HEADER_fillsize_shift)
SDMA_PKT_CONSTANT_FILL_DST_ADDR_LO_dst_addr_31_0_offset = 1
SDMA_PKT_CONSTANT_FILL_DST_ADDR_LO_dst_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_CONSTANT_FILL_DST_ADDR_LO_dst_addr_31_0_shift = 0
SDMA_PKT_CONSTANT_FILL_DST_ADDR_LO_DST_ADDR_31_0 = lambda x: (((x) & SDMA_PKT_CONSTANT_FILL_DST_ADDR_LO_dst_addr_31_0_mask) << SDMA_PKT_CONSTANT_FILL_DST_ADDR_LO_dst_addr_31_0_shift)
SDMA_PKT_CONSTANT_FILL_DST_ADDR_HI_dst_addr_63_32_offset = 2
SDMA_PKT_CONSTANT_FILL_DST_ADDR_HI_dst_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_CONSTANT_FILL_DST_ADDR_HI_dst_addr_63_32_shift = 0
SDMA_PKT_CONSTANT_FILL_DST_ADDR_HI_DST_ADDR_63_32 = lambda x: (((x) & SDMA_PKT_CONSTANT_FILL_DST_ADDR_HI_dst_addr_63_32_mask) << SDMA_PKT_CONSTANT_FILL_DST_ADDR_HI_dst_addr_63_32_shift)
SDMA_PKT_CONSTANT_FILL_DATA_src_data_31_0_offset = 3
SDMA_PKT_CONSTANT_FILL_DATA_src_data_31_0_mask = 0xFFFFFFFF
SDMA_PKT_CONSTANT_FILL_DATA_src_data_31_0_shift = 0
SDMA_PKT_CONSTANT_FILL_DATA_SRC_DATA_31_0 = lambda x: (((x) & SDMA_PKT_CONSTANT_FILL_DATA_src_data_31_0_mask) << SDMA_PKT_CONSTANT_FILL_DATA_src_data_31_0_shift)
SDMA_PKT_CONSTANT_FILL_COUNT_count_offset = 4
SDMA_PKT_CONSTANT_FILL_COUNT_count_mask = 0x3FFFFFFF
SDMA_PKT_CONSTANT_FILL_COUNT_count_shift = 0
SDMA_PKT_CONSTANT_FILL_COUNT_COUNT = lambda x: (((x) & SDMA_PKT_CONSTANT_FILL_COUNT_count_mask) << SDMA_PKT_CONSTANT_FILL_COUNT_count_shift)
SDMA_PKT_DATA_FILL_MULTI_HEADER_op_offset = 0
SDMA_PKT_DATA_FILL_MULTI_HEADER_op_mask = 0x000000FF
SDMA_PKT_DATA_FILL_MULTI_HEADER_op_shift = 0
SDMA_PKT_DATA_FILL_MULTI_HEADER_OP = lambda x: (((x) & SDMA_PKT_DATA_FILL_MULTI_HEADER_op_mask) << SDMA_PKT_DATA_FILL_MULTI_HEADER_op_shift)
SDMA_PKT_DATA_FILL_MULTI_HEADER_sub_op_offset = 0
SDMA_PKT_DATA_FILL_MULTI_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_DATA_FILL_MULTI_HEADER_sub_op_shift = 8
SDMA_PKT_DATA_FILL_MULTI_HEADER_SUB_OP = lambda x: (((x) & SDMA_PKT_DATA_FILL_MULTI_HEADER_sub_op_mask) << SDMA_PKT_DATA_FILL_MULTI_HEADER_sub_op_shift)
SDMA_PKT_DATA_FILL_MULTI_HEADER_cache_policy_offset = 0
SDMA_PKT_DATA_FILL_MULTI_HEADER_cache_policy_mask = 0x00000007
SDMA_PKT_DATA_FILL_MULTI_HEADER_cache_policy_shift = 24
SDMA_PKT_DATA_FILL_MULTI_HEADER_CACHE_POLICY = lambda x: (((x) & SDMA_PKT_DATA_FILL_MULTI_HEADER_cache_policy_mask) << SDMA_PKT_DATA_FILL_MULTI_HEADER_cache_policy_shift)
SDMA_PKT_DATA_FILL_MULTI_HEADER_cpv_offset = 0
SDMA_PKT_DATA_FILL_MULTI_HEADER_cpv_mask = 0x00000001
SDMA_PKT_DATA_FILL_MULTI_HEADER_cpv_shift = 28
SDMA_PKT_DATA_FILL_MULTI_HEADER_CPV = lambda x: (((x) & SDMA_PKT_DATA_FILL_MULTI_HEADER_cpv_mask) << SDMA_PKT_DATA_FILL_MULTI_HEADER_cpv_shift)
SDMA_PKT_DATA_FILL_MULTI_HEADER_memlog_clr_offset = 0
SDMA_PKT_DATA_FILL_MULTI_HEADER_memlog_clr_mask = 0x00000001
SDMA_PKT_DATA_FILL_MULTI_HEADER_memlog_clr_shift = 31
SDMA_PKT_DATA_FILL_MULTI_HEADER_MEMLOG_CLR = lambda x: (((x) & SDMA_PKT_DATA_FILL_MULTI_HEADER_memlog_clr_mask) << SDMA_PKT_DATA_FILL_MULTI_HEADER_memlog_clr_shift)
SDMA_PKT_DATA_FILL_MULTI_BYTE_STRIDE_byte_stride_offset = 1
SDMA_PKT_DATA_FILL_MULTI_BYTE_STRIDE_byte_stride_mask = 0xFFFFFFFF
SDMA_PKT_DATA_FILL_MULTI_BYTE_STRIDE_byte_stride_shift = 0
SDMA_PKT_DATA_FILL_MULTI_BYTE_STRIDE_BYTE_STRIDE = lambda x: (((x) & SDMA_PKT_DATA_FILL_MULTI_BYTE_STRIDE_byte_stride_mask) << SDMA_PKT_DATA_FILL_MULTI_BYTE_STRIDE_byte_stride_shift)
SDMA_PKT_DATA_FILL_MULTI_DMA_COUNT_dma_count_offset = 2
SDMA_PKT_DATA_FILL_MULTI_DMA_COUNT_dma_count_mask = 0xFFFFFFFF
SDMA_PKT_DATA_FILL_MULTI_DMA_COUNT_dma_count_shift = 0
SDMA_PKT_DATA_FILL_MULTI_DMA_COUNT_DMA_COUNT = lambda x: (((x) & SDMA_PKT_DATA_FILL_MULTI_DMA_COUNT_dma_count_mask) << SDMA_PKT_DATA_FILL_MULTI_DMA_COUNT_dma_count_shift)
SDMA_PKT_DATA_FILL_MULTI_DST_ADDR_LO_dst_addr_31_0_offset = 3
SDMA_PKT_DATA_FILL_MULTI_DST_ADDR_LO_dst_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_DATA_FILL_MULTI_DST_ADDR_LO_dst_addr_31_0_shift = 0
SDMA_PKT_DATA_FILL_MULTI_DST_ADDR_LO_DST_ADDR_31_0 = lambda x: (((x) & SDMA_PKT_DATA_FILL_MULTI_DST_ADDR_LO_dst_addr_31_0_mask) << SDMA_PKT_DATA_FILL_MULTI_DST_ADDR_LO_dst_addr_31_0_shift)
SDMA_PKT_DATA_FILL_MULTI_DST_ADDR_HI_dst_addr_63_32_offset = 4
SDMA_PKT_DATA_FILL_MULTI_DST_ADDR_HI_dst_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_DATA_FILL_MULTI_DST_ADDR_HI_dst_addr_63_32_shift = 0
SDMA_PKT_DATA_FILL_MULTI_DST_ADDR_HI_DST_ADDR_63_32 = lambda x: (((x) & SDMA_PKT_DATA_FILL_MULTI_DST_ADDR_HI_dst_addr_63_32_mask) << SDMA_PKT_DATA_FILL_MULTI_DST_ADDR_HI_dst_addr_63_32_shift)
SDMA_PKT_DATA_FILL_MULTI_BYTE_COUNT_count_offset = 5
SDMA_PKT_DATA_FILL_MULTI_BYTE_COUNT_count_mask = 0x03FFFFFF
SDMA_PKT_DATA_FILL_MULTI_BYTE_COUNT_count_shift = 0
SDMA_PKT_DATA_FILL_MULTI_BYTE_COUNT_COUNT = lambda x: (((x) & SDMA_PKT_DATA_FILL_MULTI_BYTE_COUNT_count_mask) << SDMA_PKT_DATA_FILL_MULTI_BYTE_COUNT_count_shift)
SDMA_PKT_POLL_REGMEM_HEADER_op_offset = 0
SDMA_PKT_POLL_REGMEM_HEADER_op_mask = 0x000000FF
SDMA_PKT_POLL_REGMEM_HEADER_op_shift = 0
SDMA_PKT_POLL_REGMEM_HEADER_OP = lambda x: (((x) & SDMA_PKT_POLL_REGMEM_HEADER_op_mask) << SDMA_PKT_POLL_REGMEM_HEADER_op_shift)
SDMA_PKT_POLL_REGMEM_HEADER_sub_op_offset = 0
SDMA_PKT_POLL_REGMEM_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_POLL_REGMEM_HEADER_sub_op_shift = 8
SDMA_PKT_POLL_REGMEM_HEADER_SUB_OP = lambda x: (((x) & SDMA_PKT_POLL_REGMEM_HEADER_sub_op_mask) << SDMA_PKT_POLL_REGMEM_HEADER_sub_op_shift)
SDMA_PKT_POLL_REGMEM_HEADER_cache_policy_offset = 0
SDMA_PKT_POLL_REGMEM_HEADER_cache_policy_mask = 0x00000007
SDMA_PKT_POLL_REGMEM_HEADER_cache_policy_shift = 20
SDMA_PKT_POLL_REGMEM_HEADER_CACHE_POLICY = lambda x: (((x) & SDMA_PKT_POLL_REGMEM_HEADER_cache_policy_mask) << SDMA_PKT_POLL_REGMEM_HEADER_cache_policy_shift)
SDMA_PKT_POLL_REGMEM_HEADER_cpv_offset = 0
SDMA_PKT_POLL_REGMEM_HEADER_cpv_mask = 0x00000001
SDMA_PKT_POLL_REGMEM_HEADER_cpv_shift = 24
SDMA_PKT_POLL_REGMEM_HEADER_CPV = lambda x: (((x) & SDMA_PKT_POLL_REGMEM_HEADER_cpv_mask) << SDMA_PKT_POLL_REGMEM_HEADER_cpv_shift)
SDMA_PKT_POLL_REGMEM_HEADER_hdp_flush_offset = 0
SDMA_PKT_POLL_REGMEM_HEADER_hdp_flush_mask = 0x00000001
SDMA_PKT_POLL_REGMEM_HEADER_hdp_flush_shift = 26
SDMA_PKT_POLL_REGMEM_HEADER_HDP_FLUSH = lambda x: (((x) & SDMA_PKT_POLL_REGMEM_HEADER_hdp_flush_mask) << SDMA_PKT_POLL_REGMEM_HEADER_hdp_flush_shift)
SDMA_PKT_POLL_REGMEM_HEADER_func_offset = 0
SDMA_PKT_POLL_REGMEM_HEADER_func_mask = 0x00000007
SDMA_PKT_POLL_REGMEM_HEADER_func_shift = 28
SDMA_PKT_POLL_REGMEM_HEADER_FUNC = lambda x: (((x) & SDMA_PKT_POLL_REGMEM_HEADER_func_mask) << SDMA_PKT_POLL_REGMEM_HEADER_func_shift)
SDMA_PKT_POLL_REGMEM_HEADER_mem_poll_offset = 0
SDMA_PKT_POLL_REGMEM_HEADER_mem_poll_mask = 0x00000001
SDMA_PKT_POLL_REGMEM_HEADER_mem_poll_shift = 31
SDMA_PKT_POLL_REGMEM_HEADER_MEM_POLL = lambda x: (((x) & SDMA_PKT_POLL_REGMEM_HEADER_mem_poll_mask) << SDMA_PKT_POLL_REGMEM_HEADER_mem_poll_shift)
SDMA_PKT_POLL_REGMEM_ADDR_LO_addr_31_0_offset = 1
SDMA_PKT_POLL_REGMEM_ADDR_LO_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_POLL_REGMEM_ADDR_LO_addr_31_0_shift = 0
SDMA_PKT_POLL_REGMEM_ADDR_LO_ADDR_31_0 = lambda x: (((x) & SDMA_PKT_POLL_REGMEM_ADDR_LO_addr_31_0_mask) << SDMA_PKT_POLL_REGMEM_ADDR_LO_addr_31_0_shift)
SDMA_PKT_POLL_REGMEM_ADDR_HI_addr_63_32_offset = 2
SDMA_PKT_POLL_REGMEM_ADDR_HI_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_POLL_REGMEM_ADDR_HI_addr_63_32_shift = 0
SDMA_PKT_POLL_REGMEM_ADDR_HI_ADDR_63_32 = lambda x: (((x) & SDMA_PKT_POLL_REGMEM_ADDR_HI_addr_63_32_mask) << SDMA_PKT_POLL_REGMEM_ADDR_HI_addr_63_32_shift)
SDMA_PKT_POLL_REGMEM_VALUE_value_offset = 3
SDMA_PKT_POLL_REGMEM_VALUE_value_mask = 0xFFFFFFFF
SDMA_PKT_POLL_REGMEM_VALUE_value_shift = 0
SDMA_PKT_POLL_REGMEM_VALUE_VALUE = lambda x: (((x) & SDMA_PKT_POLL_REGMEM_VALUE_value_mask) << SDMA_PKT_POLL_REGMEM_VALUE_value_shift)
SDMA_PKT_POLL_REGMEM_MASK_mask_offset = 4
SDMA_PKT_POLL_REGMEM_MASK_mask_mask = 0xFFFFFFFF
SDMA_PKT_POLL_REGMEM_MASK_mask_shift = 0
SDMA_PKT_POLL_REGMEM_MASK_MASK = lambda x: (((x) & SDMA_PKT_POLL_REGMEM_MASK_mask_mask) << SDMA_PKT_POLL_REGMEM_MASK_mask_shift)
SDMA_PKT_POLL_REGMEM_DW5_interval_offset = 5
SDMA_PKT_POLL_REGMEM_DW5_interval_mask = 0x0000FFFF
SDMA_PKT_POLL_REGMEM_DW5_interval_shift = 0
SDMA_PKT_POLL_REGMEM_DW5_INTERVAL = lambda x: (((x) & SDMA_PKT_POLL_REGMEM_DW5_interval_mask) << SDMA_PKT_POLL_REGMEM_DW5_interval_shift)
SDMA_PKT_POLL_REGMEM_DW5_retry_count_offset = 5
SDMA_PKT_POLL_REGMEM_DW5_retry_count_mask = 0x00000FFF
SDMA_PKT_POLL_REGMEM_DW5_retry_count_shift = 16
SDMA_PKT_POLL_REGMEM_DW5_RETRY_COUNT = lambda x: (((x) & SDMA_PKT_POLL_REGMEM_DW5_retry_count_mask) << SDMA_PKT_POLL_REGMEM_DW5_retry_count_shift)
SDMA_PKT_POLL_REG_WRITE_MEM_HEADER_op_offset = 0
SDMA_PKT_POLL_REG_WRITE_MEM_HEADER_op_mask = 0x000000FF
SDMA_PKT_POLL_REG_WRITE_MEM_HEADER_op_shift = 0
SDMA_PKT_POLL_REG_WRITE_MEM_HEADER_OP = lambda x: (((x) & SDMA_PKT_POLL_REG_WRITE_MEM_HEADER_op_mask) << SDMA_PKT_POLL_REG_WRITE_MEM_HEADER_op_shift)
SDMA_PKT_POLL_REG_WRITE_MEM_HEADER_sub_op_offset = 0
SDMA_PKT_POLL_REG_WRITE_MEM_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_POLL_REG_WRITE_MEM_HEADER_sub_op_shift = 8
SDMA_PKT_POLL_REG_WRITE_MEM_HEADER_SUB_OP = lambda x: (((x) & SDMA_PKT_POLL_REG_WRITE_MEM_HEADER_sub_op_mask) << SDMA_PKT_POLL_REG_WRITE_MEM_HEADER_sub_op_shift)
SDMA_PKT_POLL_REG_WRITE_MEM_HEADER_cache_policy_offset = 0
SDMA_PKT_POLL_REG_WRITE_MEM_HEADER_cache_policy_mask = 0x00000007
SDMA_PKT_POLL_REG_WRITE_MEM_HEADER_cache_policy_shift = 24
SDMA_PKT_POLL_REG_WRITE_MEM_HEADER_CACHE_POLICY = lambda x: (((x) & SDMA_PKT_POLL_REG_WRITE_MEM_HEADER_cache_policy_mask) << SDMA_PKT_POLL_REG_WRITE_MEM_HEADER_cache_policy_shift)
SDMA_PKT_POLL_REG_WRITE_MEM_HEADER_cpv_offset = 0
SDMA_PKT_POLL_REG_WRITE_MEM_HEADER_cpv_mask = 0x00000001
SDMA_PKT_POLL_REG_WRITE_MEM_HEADER_cpv_shift = 28
SDMA_PKT_POLL_REG_WRITE_MEM_HEADER_CPV = lambda x: (((x) & SDMA_PKT_POLL_REG_WRITE_MEM_HEADER_cpv_mask) << SDMA_PKT_POLL_REG_WRITE_MEM_HEADER_cpv_shift)
SDMA_PKT_POLL_REG_WRITE_MEM_SRC_ADDR_addr_31_2_offset = 1
SDMA_PKT_POLL_REG_WRITE_MEM_SRC_ADDR_addr_31_2_mask = 0x3FFFFFFF
SDMA_PKT_POLL_REG_WRITE_MEM_SRC_ADDR_addr_31_2_shift = 2
SDMA_PKT_POLL_REG_WRITE_MEM_SRC_ADDR_ADDR_31_2 = lambda x: (((x) & SDMA_PKT_POLL_REG_WRITE_MEM_SRC_ADDR_addr_31_2_mask) << SDMA_PKT_POLL_REG_WRITE_MEM_SRC_ADDR_addr_31_2_shift)
SDMA_PKT_POLL_REG_WRITE_MEM_DST_ADDR_LO_addr_31_0_offset = 2
SDMA_PKT_POLL_REG_WRITE_MEM_DST_ADDR_LO_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_POLL_REG_WRITE_MEM_DST_ADDR_LO_addr_31_0_shift = 0
SDMA_PKT_POLL_REG_WRITE_MEM_DST_ADDR_LO_ADDR_31_0 = lambda x: (((x) & SDMA_PKT_POLL_REG_WRITE_MEM_DST_ADDR_LO_addr_31_0_mask) << SDMA_PKT_POLL_REG_WRITE_MEM_DST_ADDR_LO_addr_31_0_shift)
SDMA_PKT_POLL_REG_WRITE_MEM_DST_ADDR_HI_addr_63_32_offset = 3
SDMA_PKT_POLL_REG_WRITE_MEM_DST_ADDR_HI_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_POLL_REG_WRITE_MEM_DST_ADDR_HI_addr_63_32_shift = 0
SDMA_PKT_POLL_REG_WRITE_MEM_DST_ADDR_HI_ADDR_63_32 = lambda x: (((x) & SDMA_PKT_POLL_REG_WRITE_MEM_DST_ADDR_HI_addr_63_32_mask) << SDMA_PKT_POLL_REG_WRITE_MEM_DST_ADDR_HI_addr_63_32_shift)
SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_op_offset = 0
SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_op_mask = 0x000000FF
SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_op_shift = 0
SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_OP = lambda x: (((x) & SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_op_mask) << SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_op_shift)
SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_sub_op_offset = 0
SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_sub_op_shift = 8
SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_SUB_OP = lambda x: (((x) & SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_sub_op_mask) << SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_sub_op_shift)
SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_ea_offset = 0
SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_ea_mask = 0x00000003
SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_ea_shift = 16
SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_EA = lambda x: (((x) & SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_ea_mask) << SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_ea_shift)
SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_cache_policy_offset = 0
SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_cache_policy_mask = 0x00000007
SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_cache_policy_shift = 24
SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_CACHE_POLICY = lambda x: (((x) & SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_cache_policy_mask) << SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_cache_policy_shift)
SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_cpv_offset = 0
SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_cpv_mask = 0x00000001
SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_cpv_shift = 28
SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_CPV = lambda x: (((x) & SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_cpv_mask) << SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_cpv_shift)
SDMA_PKT_POLL_DBIT_WRITE_MEM_DST_ADDR_LO_addr_31_0_offset = 1
SDMA_PKT_POLL_DBIT_WRITE_MEM_DST_ADDR_LO_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_POLL_DBIT_WRITE_MEM_DST_ADDR_LO_addr_31_0_shift = 0
SDMA_PKT_POLL_DBIT_WRITE_MEM_DST_ADDR_LO_ADDR_31_0 = lambda x: (((x) & SDMA_PKT_POLL_DBIT_WRITE_MEM_DST_ADDR_LO_addr_31_0_mask) << SDMA_PKT_POLL_DBIT_WRITE_MEM_DST_ADDR_LO_addr_31_0_shift)
SDMA_PKT_POLL_DBIT_WRITE_MEM_DST_ADDR_HI_addr_63_32_offset = 2
SDMA_PKT_POLL_DBIT_WRITE_MEM_DST_ADDR_HI_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_POLL_DBIT_WRITE_MEM_DST_ADDR_HI_addr_63_32_shift = 0
SDMA_PKT_POLL_DBIT_WRITE_MEM_DST_ADDR_HI_ADDR_63_32 = lambda x: (((x) & SDMA_PKT_POLL_DBIT_WRITE_MEM_DST_ADDR_HI_addr_63_32_mask) << SDMA_PKT_POLL_DBIT_WRITE_MEM_DST_ADDR_HI_addr_63_32_shift)
SDMA_PKT_POLL_DBIT_WRITE_MEM_START_PAGE_addr_31_4_offset = 3
SDMA_PKT_POLL_DBIT_WRITE_MEM_START_PAGE_addr_31_4_mask = 0x0FFFFFFF
SDMA_PKT_POLL_DBIT_WRITE_MEM_START_PAGE_addr_31_4_shift = 4
SDMA_PKT_POLL_DBIT_WRITE_MEM_START_PAGE_ADDR_31_4 = lambda x: (((x) & SDMA_PKT_POLL_DBIT_WRITE_MEM_START_PAGE_addr_31_4_mask) << SDMA_PKT_POLL_DBIT_WRITE_MEM_START_PAGE_addr_31_4_shift)
SDMA_PKT_POLL_DBIT_WRITE_MEM_PAGE_NUM_page_num_31_0_offset = 4
SDMA_PKT_POLL_DBIT_WRITE_MEM_PAGE_NUM_page_num_31_0_mask = 0xFFFFFFFF
SDMA_PKT_POLL_DBIT_WRITE_MEM_PAGE_NUM_page_num_31_0_shift = 0
SDMA_PKT_POLL_DBIT_WRITE_MEM_PAGE_NUM_PAGE_NUM_31_0 = lambda x: (((x) & SDMA_PKT_POLL_DBIT_WRITE_MEM_PAGE_NUM_page_num_31_0_mask) << SDMA_PKT_POLL_DBIT_WRITE_MEM_PAGE_NUM_page_num_31_0_shift)
SDMA_PKT_POLL_MEM_VERIFY_HEADER_op_offset = 0
SDMA_PKT_POLL_MEM_VERIFY_HEADER_op_mask = 0x000000FF
SDMA_PKT_POLL_MEM_VERIFY_HEADER_op_shift = 0
SDMA_PKT_POLL_MEM_VERIFY_HEADER_OP = lambda x: (((x) & SDMA_PKT_POLL_MEM_VERIFY_HEADER_op_mask) << SDMA_PKT_POLL_MEM_VERIFY_HEADER_op_shift)
SDMA_PKT_POLL_MEM_VERIFY_HEADER_sub_op_offset = 0
SDMA_PKT_POLL_MEM_VERIFY_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_POLL_MEM_VERIFY_HEADER_sub_op_shift = 8
SDMA_PKT_POLL_MEM_VERIFY_HEADER_SUB_OP = lambda x: (((x) & SDMA_PKT_POLL_MEM_VERIFY_HEADER_sub_op_mask) << SDMA_PKT_POLL_MEM_VERIFY_HEADER_sub_op_shift)
SDMA_PKT_POLL_MEM_VERIFY_HEADER_cache_policy_offset = 0
SDMA_PKT_POLL_MEM_VERIFY_HEADER_cache_policy_mask = 0x00000007
SDMA_PKT_POLL_MEM_VERIFY_HEADER_cache_policy_shift = 24
SDMA_PKT_POLL_MEM_VERIFY_HEADER_CACHE_POLICY = lambda x: (((x) & SDMA_PKT_POLL_MEM_VERIFY_HEADER_cache_policy_mask) << SDMA_PKT_POLL_MEM_VERIFY_HEADER_cache_policy_shift)
SDMA_PKT_POLL_MEM_VERIFY_HEADER_cpv_offset = 0
SDMA_PKT_POLL_MEM_VERIFY_HEADER_cpv_mask = 0x00000001
SDMA_PKT_POLL_MEM_VERIFY_HEADER_cpv_shift = 28
SDMA_PKT_POLL_MEM_VERIFY_HEADER_CPV = lambda x: (((x) & SDMA_PKT_POLL_MEM_VERIFY_HEADER_cpv_mask) << SDMA_PKT_POLL_MEM_VERIFY_HEADER_cpv_shift)
SDMA_PKT_POLL_MEM_VERIFY_HEADER_mode_offset = 0
SDMA_PKT_POLL_MEM_VERIFY_HEADER_mode_mask = 0x00000001
SDMA_PKT_POLL_MEM_VERIFY_HEADER_mode_shift = 31
SDMA_PKT_POLL_MEM_VERIFY_HEADER_MODE = lambda x: (((x) & SDMA_PKT_POLL_MEM_VERIFY_HEADER_mode_mask) << SDMA_PKT_POLL_MEM_VERIFY_HEADER_mode_shift)
SDMA_PKT_POLL_MEM_VERIFY_PATTERN_pattern_offset = 1
SDMA_PKT_POLL_MEM_VERIFY_PATTERN_pattern_mask = 0xFFFFFFFF
SDMA_PKT_POLL_MEM_VERIFY_PATTERN_pattern_shift = 0
SDMA_PKT_POLL_MEM_VERIFY_PATTERN_PATTERN = lambda x: (((x) & SDMA_PKT_POLL_MEM_VERIFY_PATTERN_pattern_mask) << SDMA_PKT_POLL_MEM_VERIFY_PATTERN_pattern_shift)
SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_START_LO_cmp0_start_31_0_offset = 2
SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_START_LO_cmp0_start_31_0_mask = 0xFFFFFFFF
SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_START_LO_cmp0_start_31_0_shift = 0
SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_START_LO_CMP0_START_31_0 = lambda x: (((x) & SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_START_LO_cmp0_start_31_0_mask) << SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_START_LO_cmp0_start_31_0_shift)
SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_START_HI_cmp0_start_63_32_offset = 3
SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_START_HI_cmp0_start_63_32_mask = 0xFFFFFFFF
SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_START_HI_cmp0_start_63_32_shift = 0
SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_START_HI_CMP0_START_63_32 = lambda x: (((x) & SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_START_HI_cmp0_start_63_32_mask) << SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_START_HI_cmp0_start_63_32_shift)
SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_END_LO_cmp0_end_31_0_offset = 4
SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_END_LO_cmp0_end_31_0_mask = 0xFFFFFFFF
SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_END_LO_cmp0_end_31_0_shift = 0
SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_END_LO_CMP0_END_31_0 = lambda x: (((x) & SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_END_LO_cmp0_end_31_0_mask) << SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_END_LO_cmp0_end_31_0_shift)
SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_END_HI_cmp0_end_63_32_offset = 5
SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_END_HI_cmp0_end_63_32_mask = 0xFFFFFFFF
SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_END_HI_cmp0_end_63_32_shift = 0
SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_END_HI_CMP0_END_63_32 = lambda x: (((x) & SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_END_HI_cmp0_end_63_32_mask) << SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_END_HI_cmp0_end_63_32_shift)
SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_START_LO_cmp1_start_31_0_offset = 6
SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_START_LO_cmp1_start_31_0_mask = 0xFFFFFFFF
SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_START_LO_cmp1_start_31_0_shift = 0
SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_START_LO_CMP1_START_31_0 = lambda x: (((x) & SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_START_LO_cmp1_start_31_0_mask) << SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_START_LO_cmp1_start_31_0_shift)
SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_START_HI_cmp1_start_63_32_offset = 7
SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_START_HI_cmp1_start_63_32_mask = 0xFFFFFFFF
SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_START_HI_cmp1_start_63_32_shift = 0
SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_START_HI_CMP1_START_63_32 = lambda x: (((x) & SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_START_HI_cmp1_start_63_32_mask) << SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_START_HI_cmp1_start_63_32_shift)
SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_END_LO_cmp1_end_31_0_offset = 8
SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_END_LO_cmp1_end_31_0_mask = 0xFFFFFFFF
SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_END_LO_cmp1_end_31_0_shift = 0
SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_END_LO_CMP1_END_31_0 = lambda x: (((x) & SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_END_LO_cmp1_end_31_0_mask) << SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_END_LO_cmp1_end_31_0_shift)
SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_END_HI_cmp1_end_63_32_offset = 9
SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_END_HI_cmp1_end_63_32_mask = 0xFFFFFFFF
SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_END_HI_cmp1_end_63_32_shift = 0
SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_END_HI_CMP1_END_63_32 = lambda x: (((x) & SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_END_HI_cmp1_end_63_32_mask) << SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_END_HI_cmp1_end_63_32_shift)
SDMA_PKT_POLL_MEM_VERIFY_REC_ADDR_LO_rec_31_0_offset = 10
SDMA_PKT_POLL_MEM_VERIFY_REC_ADDR_LO_rec_31_0_mask = 0xFFFFFFFF
SDMA_PKT_POLL_MEM_VERIFY_REC_ADDR_LO_rec_31_0_shift = 0
SDMA_PKT_POLL_MEM_VERIFY_REC_ADDR_LO_REC_31_0 = lambda x: (((x) & SDMA_PKT_POLL_MEM_VERIFY_REC_ADDR_LO_rec_31_0_mask) << SDMA_PKT_POLL_MEM_VERIFY_REC_ADDR_LO_rec_31_0_shift)
SDMA_PKT_POLL_MEM_VERIFY_REC_ADDR_HI_rec_63_32_offset = 11
SDMA_PKT_POLL_MEM_VERIFY_REC_ADDR_HI_rec_63_32_mask = 0xFFFFFFFF
SDMA_PKT_POLL_MEM_VERIFY_REC_ADDR_HI_rec_63_32_shift = 0
SDMA_PKT_POLL_MEM_VERIFY_REC_ADDR_HI_REC_63_32 = lambda x: (((x) & SDMA_PKT_POLL_MEM_VERIFY_REC_ADDR_HI_rec_63_32_mask) << SDMA_PKT_POLL_MEM_VERIFY_REC_ADDR_HI_rec_63_32_shift)
SDMA_PKT_POLL_MEM_VERIFY_RESERVED_reserved_offset = 12
SDMA_PKT_POLL_MEM_VERIFY_RESERVED_reserved_mask = 0xFFFFFFFF
SDMA_PKT_POLL_MEM_VERIFY_RESERVED_reserved_shift = 0
SDMA_PKT_POLL_MEM_VERIFY_RESERVED_RESERVED = lambda x: (((x) & SDMA_PKT_POLL_MEM_VERIFY_RESERVED_reserved_mask) << SDMA_PKT_POLL_MEM_VERIFY_RESERVED_reserved_shift)
SDMA_PKT_ATOMIC_HEADER_op_offset = 0
SDMA_PKT_ATOMIC_HEADER_op_mask = 0x000000FF
SDMA_PKT_ATOMIC_HEADER_op_shift = 0
SDMA_PKT_ATOMIC_HEADER_OP = lambda x: (((x) & SDMA_PKT_ATOMIC_HEADER_op_mask) << SDMA_PKT_ATOMIC_HEADER_op_shift)
SDMA_PKT_ATOMIC_HEADER_loop_offset = 0
SDMA_PKT_ATOMIC_HEADER_loop_mask = 0x00000001
SDMA_PKT_ATOMIC_HEADER_loop_shift = 16
SDMA_PKT_ATOMIC_HEADER_LOOP = lambda x: (((x) & SDMA_PKT_ATOMIC_HEADER_loop_mask) << SDMA_PKT_ATOMIC_HEADER_loop_shift)
SDMA_PKT_ATOMIC_HEADER_tmz_offset = 0
SDMA_PKT_ATOMIC_HEADER_tmz_mask = 0x00000001
SDMA_PKT_ATOMIC_HEADER_tmz_shift = 18
SDMA_PKT_ATOMIC_HEADER_TMZ = lambda x: (((x) & SDMA_PKT_ATOMIC_HEADER_tmz_mask) << SDMA_PKT_ATOMIC_HEADER_tmz_shift)
SDMA_PKT_ATOMIC_HEADER_cache_policy_offset = 0
SDMA_PKT_ATOMIC_HEADER_cache_policy_mask = 0x00000007
SDMA_PKT_ATOMIC_HEADER_cache_policy_shift = 20
SDMA_PKT_ATOMIC_HEADER_CACHE_POLICY = lambda x: (((x) & SDMA_PKT_ATOMIC_HEADER_cache_policy_mask) << SDMA_PKT_ATOMIC_HEADER_cache_policy_shift)
SDMA_PKT_ATOMIC_HEADER_cpv_offset = 0
SDMA_PKT_ATOMIC_HEADER_cpv_mask = 0x00000001
SDMA_PKT_ATOMIC_HEADER_cpv_shift = 24
SDMA_PKT_ATOMIC_HEADER_CPV = lambda x: (((x) & SDMA_PKT_ATOMIC_HEADER_cpv_mask) << SDMA_PKT_ATOMIC_HEADER_cpv_shift)
SDMA_PKT_ATOMIC_HEADER_atomic_op_offset = 0
SDMA_PKT_ATOMIC_HEADER_atomic_op_mask = 0x0000007F
SDMA_PKT_ATOMIC_HEADER_atomic_op_shift = 25
SDMA_PKT_ATOMIC_HEADER_ATOMIC_OP = lambda x: (((x) & SDMA_PKT_ATOMIC_HEADER_atomic_op_mask) << SDMA_PKT_ATOMIC_HEADER_atomic_op_shift)
SDMA_PKT_ATOMIC_ADDR_LO_addr_31_0_offset = 1
SDMA_PKT_ATOMIC_ADDR_LO_addr_31_0_mask = 0xFFFFFFFF
SDMA_PKT_ATOMIC_ADDR_LO_addr_31_0_shift = 0
SDMA_PKT_ATOMIC_ADDR_LO_ADDR_31_0 = lambda x: (((x) & SDMA_PKT_ATOMIC_ADDR_LO_addr_31_0_mask) << SDMA_PKT_ATOMIC_ADDR_LO_addr_31_0_shift)
SDMA_PKT_ATOMIC_ADDR_HI_addr_63_32_offset = 2
SDMA_PKT_ATOMIC_ADDR_HI_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_ATOMIC_ADDR_HI_addr_63_32_shift = 0
SDMA_PKT_ATOMIC_ADDR_HI_ADDR_63_32 = lambda x: (((x) & SDMA_PKT_ATOMIC_ADDR_HI_addr_63_32_mask) << SDMA_PKT_ATOMIC_ADDR_HI_addr_63_32_shift)
SDMA_PKT_ATOMIC_SRC_DATA_LO_src_data_31_0_offset = 3
SDMA_PKT_ATOMIC_SRC_DATA_LO_src_data_31_0_mask = 0xFFFFFFFF
SDMA_PKT_ATOMIC_SRC_DATA_LO_src_data_31_0_shift = 0
SDMA_PKT_ATOMIC_SRC_DATA_LO_SRC_DATA_31_0 = lambda x: (((x) & SDMA_PKT_ATOMIC_SRC_DATA_LO_src_data_31_0_mask) << SDMA_PKT_ATOMIC_SRC_DATA_LO_src_data_31_0_shift)
SDMA_PKT_ATOMIC_SRC_DATA_HI_src_data_63_32_offset = 4
SDMA_PKT_ATOMIC_SRC_DATA_HI_src_data_63_32_mask = 0xFFFFFFFF
SDMA_PKT_ATOMIC_SRC_DATA_HI_src_data_63_32_shift = 0
SDMA_PKT_ATOMIC_SRC_DATA_HI_SRC_DATA_63_32 = lambda x: (((x) & SDMA_PKT_ATOMIC_SRC_DATA_HI_src_data_63_32_mask) << SDMA_PKT_ATOMIC_SRC_DATA_HI_src_data_63_32_shift)
SDMA_PKT_ATOMIC_CMP_DATA_LO_cmp_data_31_0_offset = 5
SDMA_PKT_ATOMIC_CMP_DATA_LO_cmp_data_31_0_mask = 0xFFFFFFFF
SDMA_PKT_ATOMIC_CMP_DATA_LO_cmp_data_31_0_shift = 0
SDMA_PKT_ATOMIC_CMP_DATA_LO_CMP_DATA_31_0 = lambda x: (((x) & SDMA_PKT_ATOMIC_CMP_DATA_LO_cmp_data_31_0_mask) << SDMA_PKT_ATOMIC_CMP_DATA_LO_cmp_data_31_0_shift)
SDMA_PKT_ATOMIC_CMP_DATA_HI_cmp_data_63_32_offset = 6
SDMA_PKT_ATOMIC_CMP_DATA_HI_cmp_data_63_32_mask = 0xFFFFFFFF
SDMA_PKT_ATOMIC_CMP_DATA_HI_cmp_data_63_32_shift = 0
SDMA_PKT_ATOMIC_CMP_DATA_HI_CMP_DATA_63_32 = lambda x: (((x) & SDMA_PKT_ATOMIC_CMP_DATA_HI_cmp_data_63_32_mask) << SDMA_PKT_ATOMIC_CMP_DATA_HI_cmp_data_63_32_shift)
SDMA_PKT_ATOMIC_LOOP_INTERVAL_loop_interval_offset = 7
SDMA_PKT_ATOMIC_LOOP_INTERVAL_loop_interval_mask = 0x00001FFF
SDMA_PKT_ATOMIC_LOOP_INTERVAL_loop_interval_shift = 0
SDMA_PKT_ATOMIC_LOOP_INTERVAL_LOOP_INTERVAL = lambda x: (((x) & SDMA_PKT_ATOMIC_LOOP_INTERVAL_loop_interval_mask) << SDMA_PKT_ATOMIC_LOOP_INTERVAL_loop_interval_shift)
SDMA_PKT_TIMESTAMP_SET_HEADER_op_offset = 0
SDMA_PKT_TIMESTAMP_SET_HEADER_op_mask = 0x000000FF
SDMA_PKT_TIMESTAMP_SET_HEADER_op_shift = 0
SDMA_PKT_TIMESTAMP_SET_HEADER_OP = lambda x: (((x) & SDMA_PKT_TIMESTAMP_SET_HEADER_op_mask) << SDMA_PKT_TIMESTAMP_SET_HEADER_op_shift)
SDMA_PKT_TIMESTAMP_SET_HEADER_sub_op_offset = 0
SDMA_PKT_TIMESTAMP_SET_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_TIMESTAMP_SET_HEADER_sub_op_shift = 8
SDMA_PKT_TIMESTAMP_SET_HEADER_SUB_OP = lambda x: (((x) & SDMA_PKT_TIMESTAMP_SET_HEADER_sub_op_mask) << SDMA_PKT_TIMESTAMP_SET_HEADER_sub_op_shift)
SDMA_PKT_TIMESTAMP_SET_INIT_DATA_LO_init_data_31_0_offset = 1
SDMA_PKT_TIMESTAMP_SET_INIT_DATA_LO_init_data_31_0_mask = 0xFFFFFFFF
SDMA_PKT_TIMESTAMP_SET_INIT_DATA_LO_init_data_31_0_shift = 0
SDMA_PKT_TIMESTAMP_SET_INIT_DATA_LO_INIT_DATA_31_0 = lambda x: (((x) & SDMA_PKT_TIMESTAMP_SET_INIT_DATA_LO_init_data_31_0_mask) << SDMA_PKT_TIMESTAMP_SET_INIT_DATA_LO_init_data_31_0_shift)
SDMA_PKT_TIMESTAMP_SET_INIT_DATA_HI_init_data_63_32_offset = 2
SDMA_PKT_TIMESTAMP_SET_INIT_DATA_HI_init_data_63_32_mask = 0xFFFFFFFF
SDMA_PKT_TIMESTAMP_SET_INIT_DATA_HI_init_data_63_32_shift = 0
SDMA_PKT_TIMESTAMP_SET_INIT_DATA_HI_INIT_DATA_63_32 = lambda x: (((x) & SDMA_PKT_TIMESTAMP_SET_INIT_DATA_HI_init_data_63_32_mask) << SDMA_PKT_TIMESTAMP_SET_INIT_DATA_HI_init_data_63_32_shift)
SDMA_PKT_TIMESTAMP_GET_HEADER_op_offset = 0
SDMA_PKT_TIMESTAMP_GET_HEADER_op_mask = 0x000000FF
SDMA_PKT_TIMESTAMP_GET_HEADER_op_shift = 0
SDMA_PKT_TIMESTAMP_GET_HEADER_OP = lambda x: (((x) & SDMA_PKT_TIMESTAMP_GET_HEADER_op_mask) << SDMA_PKT_TIMESTAMP_GET_HEADER_op_shift)
SDMA_PKT_TIMESTAMP_GET_HEADER_sub_op_offset = 0
SDMA_PKT_TIMESTAMP_GET_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_TIMESTAMP_GET_HEADER_sub_op_shift = 8
SDMA_PKT_TIMESTAMP_GET_HEADER_SUB_OP = lambda x: (((x) & SDMA_PKT_TIMESTAMP_GET_HEADER_sub_op_mask) << SDMA_PKT_TIMESTAMP_GET_HEADER_sub_op_shift)
SDMA_PKT_TIMESTAMP_GET_HEADER_l2_policy_offset = 0
SDMA_PKT_TIMESTAMP_GET_HEADER_l2_policy_mask = 0x00000003
SDMA_PKT_TIMESTAMP_GET_HEADER_l2_policy_shift = 24
SDMA_PKT_TIMESTAMP_GET_HEADER_L2_POLICY = lambda x: (((x) & SDMA_PKT_TIMESTAMP_GET_HEADER_l2_policy_mask) << SDMA_PKT_TIMESTAMP_GET_HEADER_l2_policy_shift)
SDMA_PKT_TIMESTAMP_GET_HEADER_llc_policy_offset = 0
SDMA_PKT_TIMESTAMP_GET_HEADER_llc_policy_mask = 0x00000001
SDMA_PKT_TIMESTAMP_GET_HEADER_llc_policy_shift = 26
SDMA_PKT_TIMESTAMP_GET_HEADER_LLC_POLICY = lambda x: (((x) & SDMA_PKT_TIMESTAMP_GET_HEADER_llc_policy_mask) << SDMA_PKT_TIMESTAMP_GET_HEADER_llc_policy_shift)
SDMA_PKT_TIMESTAMP_GET_HEADER_cpv_offset = 0
SDMA_PKT_TIMESTAMP_GET_HEADER_cpv_mask = 0x00000001
SDMA_PKT_TIMESTAMP_GET_HEADER_cpv_shift = 28
SDMA_PKT_TIMESTAMP_GET_HEADER_CPV = lambda x: (((x) & SDMA_PKT_TIMESTAMP_GET_HEADER_cpv_mask) << SDMA_PKT_TIMESTAMP_GET_HEADER_cpv_shift)
SDMA_PKT_TIMESTAMP_GET_WRITE_ADDR_LO_write_addr_31_3_offset = 1
SDMA_PKT_TIMESTAMP_GET_WRITE_ADDR_LO_write_addr_31_3_mask = 0x1FFFFFFF
SDMA_PKT_TIMESTAMP_GET_WRITE_ADDR_LO_write_addr_31_3_shift = 3
SDMA_PKT_TIMESTAMP_GET_WRITE_ADDR_LO_WRITE_ADDR_31_3 = lambda x: (((x) & SDMA_PKT_TIMESTAMP_GET_WRITE_ADDR_LO_write_addr_31_3_mask) << SDMA_PKT_TIMESTAMP_GET_WRITE_ADDR_LO_write_addr_31_3_shift)
SDMA_PKT_TIMESTAMP_GET_WRITE_ADDR_HI_write_addr_63_32_offset = 2
SDMA_PKT_TIMESTAMP_GET_WRITE_ADDR_HI_write_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_TIMESTAMP_GET_WRITE_ADDR_HI_write_addr_63_32_shift = 0
SDMA_PKT_TIMESTAMP_GET_WRITE_ADDR_HI_WRITE_ADDR_63_32 = lambda x: (((x) & SDMA_PKT_TIMESTAMP_GET_WRITE_ADDR_HI_write_addr_63_32_mask) << SDMA_PKT_TIMESTAMP_GET_WRITE_ADDR_HI_write_addr_63_32_shift)
SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_op_offset = 0
SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_op_mask = 0x000000FF
SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_op_shift = 0
SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_OP = lambda x: (((x) & SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_op_mask) << SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_op_shift)
SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_sub_op_offset = 0
SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_sub_op_shift = 8
SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_SUB_OP = lambda x: (((x) & SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_sub_op_mask) << SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_sub_op_shift)
SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_l2_policy_offset = 0
SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_l2_policy_mask = 0x00000003
SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_l2_policy_shift = 24
SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_L2_POLICY = lambda x: (((x) & SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_l2_policy_mask) << SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_l2_policy_shift)
SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_llc_policy_offset = 0
SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_llc_policy_mask = 0x00000001
SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_llc_policy_shift = 26
SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_LLC_POLICY = lambda x: (((x) & SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_llc_policy_mask) << SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_llc_policy_shift)
SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_cpv_offset = 0
SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_cpv_mask = 0x00000001
SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_cpv_shift = 28
SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_CPV = lambda x: (((x) & SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_cpv_mask) << SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_cpv_shift)
SDMA_PKT_TIMESTAMP_GET_GLOBAL_WRITE_ADDR_LO_write_addr_31_3_offset = 1
SDMA_PKT_TIMESTAMP_GET_GLOBAL_WRITE_ADDR_LO_write_addr_31_3_mask = 0x1FFFFFFF
SDMA_PKT_TIMESTAMP_GET_GLOBAL_WRITE_ADDR_LO_write_addr_31_3_shift = 3
SDMA_PKT_TIMESTAMP_GET_GLOBAL_WRITE_ADDR_LO_WRITE_ADDR_31_3 = lambda x: (((x) & SDMA_PKT_TIMESTAMP_GET_GLOBAL_WRITE_ADDR_LO_write_addr_31_3_mask) << SDMA_PKT_TIMESTAMP_GET_GLOBAL_WRITE_ADDR_LO_write_addr_31_3_shift)
SDMA_PKT_TIMESTAMP_GET_GLOBAL_WRITE_ADDR_HI_write_addr_63_32_offset = 2
SDMA_PKT_TIMESTAMP_GET_GLOBAL_WRITE_ADDR_HI_write_addr_63_32_mask = 0xFFFFFFFF
SDMA_PKT_TIMESTAMP_GET_GLOBAL_WRITE_ADDR_HI_write_addr_63_32_shift = 0
SDMA_PKT_TIMESTAMP_GET_GLOBAL_WRITE_ADDR_HI_WRITE_ADDR_63_32 = lambda x: (((x) & SDMA_PKT_TIMESTAMP_GET_GLOBAL_WRITE_ADDR_HI_write_addr_63_32_mask) << SDMA_PKT_TIMESTAMP_GET_GLOBAL_WRITE_ADDR_HI_write_addr_63_32_shift)
SDMA_PKT_TRAP_HEADER_op_offset = 0
SDMA_PKT_TRAP_HEADER_op_mask = 0x000000FF
SDMA_PKT_TRAP_HEADER_op_shift = 0
SDMA_PKT_TRAP_HEADER_OP = lambda x: (((x) & SDMA_PKT_TRAP_HEADER_op_mask) << SDMA_PKT_TRAP_HEADER_op_shift)
SDMA_PKT_TRAP_HEADER_sub_op_offset = 0
SDMA_PKT_TRAP_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_TRAP_HEADER_sub_op_shift = 8
SDMA_PKT_TRAP_HEADER_SUB_OP = lambda x: (((x) & SDMA_PKT_TRAP_HEADER_sub_op_mask) << SDMA_PKT_TRAP_HEADER_sub_op_shift)
SDMA_PKT_TRAP_INT_CONTEXT_int_context_offset = 1
SDMA_PKT_TRAP_INT_CONTEXT_int_context_mask = 0x0FFFFFFF
SDMA_PKT_TRAP_INT_CONTEXT_int_context_shift = 0
SDMA_PKT_TRAP_INT_CONTEXT_INT_CONTEXT = lambda x: (((x) & SDMA_PKT_TRAP_INT_CONTEXT_int_context_mask) << SDMA_PKT_TRAP_INT_CONTEXT_int_context_shift)
SDMA_PKT_DUMMY_TRAP_HEADER_op_offset = 0
SDMA_PKT_DUMMY_TRAP_HEADER_op_mask = 0x000000FF
SDMA_PKT_DUMMY_TRAP_HEADER_op_shift = 0
SDMA_PKT_DUMMY_TRAP_HEADER_OP = lambda x: (((x) & SDMA_PKT_DUMMY_TRAP_HEADER_op_mask) << SDMA_PKT_DUMMY_TRAP_HEADER_op_shift)
SDMA_PKT_DUMMY_TRAP_HEADER_sub_op_offset = 0
SDMA_PKT_DUMMY_TRAP_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_DUMMY_TRAP_HEADER_sub_op_shift = 8
SDMA_PKT_DUMMY_TRAP_HEADER_SUB_OP = lambda x: (((x) & SDMA_PKT_DUMMY_TRAP_HEADER_sub_op_mask) << SDMA_PKT_DUMMY_TRAP_HEADER_sub_op_shift)
SDMA_PKT_DUMMY_TRAP_INT_CONTEXT_int_context_offset = 1
SDMA_PKT_DUMMY_TRAP_INT_CONTEXT_int_context_mask = 0x0FFFFFFF
SDMA_PKT_DUMMY_TRAP_INT_CONTEXT_int_context_shift = 0
SDMA_PKT_DUMMY_TRAP_INT_CONTEXT_INT_CONTEXT = lambda x: (((x) & SDMA_PKT_DUMMY_TRAP_INT_CONTEXT_int_context_mask) << SDMA_PKT_DUMMY_TRAP_INT_CONTEXT_int_context_shift)
SDMA_PKT_GPUVM_INV_HEADER_op_offset = 0
SDMA_PKT_GPUVM_INV_HEADER_op_mask = 0x000000FF
SDMA_PKT_GPUVM_INV_HEADER_op_shift = 0
SDMA_PKT_GPUVM_INV_HEADER_OP = lambda x: (((x) & SDMA_PKT_GPUVM_INV_HEADER_op_mask) << SDMA_PKT_GPUVM_INV_HEADER_op_shift)
SDMA_PKT_GPUVM_INV_HEADER_sub_op_offset = 0
SDMA_PKT_GPUVM_INV_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_GPUVM_INV_HEADER_sub_op_shift = 8
SDMA_PKT_GPUVM_INV_HEADER_SUB_OP = lambda x: (((x) & SDMA_PKT_GPUVM_INV_HEADER_sub_op_mask) << SDMA_PKT_GPUVM_INV_HEADER_sub_op_shift)
SDMA_PKT_GPUVM_INV_PAYLOAD1_per_vmid_inv_req_offset = 1
SDMA_PKT_GPUVM_INV_PAYLOAD1_per_vmid_inv_req_mask = 0x0000FFFF
SDMA_PKT_GPUVM_INV_PAYLOAD1_per_vmid_inv_req_shift = 0
SDMA_PKT_GPUVM_INV_PAYLOAD1_PER_VMID_INV_REQ = lambda x: (((x) & SDMA_PKT_GPUVM_INV_PAYLOAD1_per_vmid_inv_req_mask) << SDMA_PKT_GPUVM_INV_PAYLOAD1_per_vmid_inv_req_shift)
SDMA_PKT_GPUVM_INV_PAYLOAD1_flush_type_offset = 1
SDMA_PKT_GPUVM_INV_PAYLOAD1_flush_type_mask = 0x00000007
SDMA_PKT_GPUVM_INV_PAYLOAD1_flush_type_shift = 16
SDMA_PKT_GPUVM_INV_PAYLOAD1_FLUSH_TYPE = lambda x: (((x) & SDMA_PKT_GPUVM_INV_PAYLOAD1_flush_type_mask) << SDMA_PKT_GPUVM_INV_PAYLOAD1_flush_type_shift)
SDMA_PKT_GPUVM_INV_PAYLOAD1_l2_ptes_offset = 1
SDMA_PKT_GPUVM_INV_PAYLOAD1_l2_ptes_mask = 0x00000001
SDMA_PKT_GPUVM_INV_PAYLOAD1_l2_ptes_shift = 19
SDMA_PKT_GPUVM_INV_PAYLOAD1_L2_PTES = lambda x: (((x) & SDMA_PKT_GPUVM_INV_PAYLOAD1_l2_ptes_mask) << SDMA_PKT_GPUVM_INV_PAYLOAD1_l2_ptes_shift)
SDMA_PKT_GPUVM_INV_PAYLOAD1_l2_pde0_offset = 1
SDMA_PKT_GPUVM_INV_PAYLOAD1_l2_pde0_mask = 0x00000001
SDMA_PKT_GPUVM_INV_PAYLOAD1_l2_pde0_shift = 20
SDMA_PKT_GPUVM_INV_PAYLOAD1_L2_PDE0 = lambda x: (((x) & SDMA_PKT_GPUVM_INV_PAYLOAD1_l2_pde0_mask) << SDMA_PKT_GPUVM_INV_PAYLOAD1_l2_pde0_shift)
SDMA_PKT_GPUVM_INV_PAYLOAD1_l2_pde1_offset = 1
SDMA_PKT_GPUVM_INV_PAYLOAD1_l2_pde1_mask = 0x00000001
SDMA_PKT_GPUVM_INV_PAYLOAD1_l2_pde1_shift = 21
SDMA_PKT_GPUVM_INV_PAYLOAD1_L2_PDE1 = lambda x: (((x) & SDMA_PKT_GPUVM_INV_PAYLOAD1_l2_pde1_mask) << SDMA_PKT_GPUVM_INV_PAYLOAD1_l2_pde1_shift)
SDMA_PKT_GPUVM_INV_PAYLOAD1_l2_pde2_offset = 1
SDMA_PKT_GPUVM_INV_PAYLOAD1_l2_pde2_mask = 0x00000001
SDMA_PKT_GPUVM_INV_PAYLOAD1_l2_pde2_shift = 22
SDMA_PKT_GPUVM_INV_PAYLOAD1_L2_PDE2 = lambda x: (((x) & SDMA_PKT_GPUVM_INV_PAYLOAD1_l2_pde2_mask) << SDMA_PKT_GPUVM_INV_PAYLOAD1_l2_pde2_shift)
SDMA_PKT_GPUVM_INV_PAYLOAD1_l1_ptes_offset = 1
SDMA_PKT_GPUVM_INV_PAYLOAD1_l1_ptes_mask = 0x00000001
SDMA_PKT_GPUVM_INV_PAYLOAD1_l1_ptes_shift = 23
SDMA_PKT_GPUVM_INV_PAYLOAD1_L1_PTES = lambda x: (((x) & SDMA_PKT_GPUVM_INV_PAYLOAD1_l1_ptes_mask) << SDMA_PKT_GPUVM_INV_PAYLOAD1_l1_ptes_shift)
SDMA_PKT_GPUVM_INV_PAYLOAD1_clr_protection_fault_status_addr_offset = 1
SDMA_PKT_GPUVM_INV_PAYLOAD1_clr_protection_fault_status_addr_mask = 0x00000001
SDMA_PKT_GPUVM_INV_PAYLOAD1_clr_protection_fault_status_addr_shift = 24
SDMA_PKT_GPUVM_INV_PAYLOAD1_CLR_PROTECTION_FAULT_STATUS_ADDR = lambda x: (((x) & SDMA_PKT_GPUVM_INV_PAYLOAD1_clr_protection_fault_status_addr_mask) << SDMA_PKT_GPUVM_INV_PAYLOAD1_clr_protection_fault_status_addr_shift)
SDMA_PKT_GPUVM_INV_PAYLOAD1_log_request_offset = 1
SDMA_PKT_GPUVM_INV_PAYLOAD1_log_request_mask = 0x00000001
SDMA_PKT_GPUVM_INV_PAYLOAD1_log_request_shift = 25
SDMA_PKT_GPUVM_INV_PAYLOAD1_LOG_REQUEST = lambda x: (((x) & SDMA_PKT_GPUVM_INV_PAYLOAD1_log_request_mask) << SDMA_PKT_GPUVM_INV_PAYLOAD1_log_request_shift)
SDMA_PKT_GPUVM_INV_PAYLOAD1_four_kilobytes_offset = 1
SDMA_PKT_GPUVM_INV_PAYLOAD1_four_kilobytes_mask = 0x00000001
SDMA_PKT_GPUVM_INV_PAYLOAD1_four_kilobytes_shift = 26
SDMA_PKT_GPUVM_INV_PAYLOAD1_FOUR_KILOBYTES = lambda x: (((x) & SDMA_PKT_GPUVM_INV_PAYLOAD1_four_kilobytes_mask) << SDMA_PKT_GPUVM_INV_PAYLOAD1_four_kilobytes_shift)
SDMA_PKT_GPUVM_INV_PAYLOAD2_s_offset = 2
SDMA_PKT_GPUVM_INV_PAYLOAD2_s_mask = 0x00000001
SDMA_PKT_GPUVM_INV_PAYLOAD2_s_shift = 0
SDMA_PKT_GPUVM_INV_PAYLOAD2_S = lambda x: (((x) & SDMA_PKT_GPUVM_INV_PAYLOAD2_s_mask) << SDMA_PKT_GPUVM_INV_PAYLOAD2_s_shift)
SDMA_PKT_GPUVM_INV_PAYLOAD2_page_va_42_12_offset = 2
SDMA_PKT_GPUVM_INV_PAYLOAD2_page_va_42_12_mask = 0x7FFFFFFF
SDMA_PKT_GPUVM_INV_PAYLOAD2_page_va_42_12_shift = 1
SDMA_PKT_GPUVM_INV_PAYLOAD2_PAGE_VA_42_12 = lambda x: (((x) & SDMA_PKT_GPUVM_INV_PAYLOAD2_page_va_42_12_mask) << SDMA_PKT_GPUVM_INV_PAYLOAD2_page_va_42_12_shift)
SDMA_PKT_GPUVM_INV_PAYLOAD3_page_va_47_43_offset = 3
SDMA_PKT_GPUVM_INV_PAYLOAD3_page_va_47_43_mask = 0x0000003F
SDMA_PKT_GPUVM_INV_PAYLOAD3_page_va_47_43_shift = 0
SDMA_PKT_GPUVM_INV_PAYLOAD3_PAGE_VA_47_43 = lambda x: (((x) & SDMA_PKT_GPUVM_INV_PAYLOAD3_page_va_47_43_mask) << SDMA_PKT_GPUVM_INV_PAYLOAD3_page_va_47_43_shift)
SDMA_PKT_GCR_REQ_HEADER_op_offset = 0
SDMA_PKT_GCR_REQ_HEADER_op_mask = 0x000000FF
SDMA_PKT_GCR_REQ_HEADER_op_shift = 0
SDMA_PKT_GCR_REQ_HEADER_OP = lambda x: (((x) & SDMA_PKT_GCR_REQ_HEADER_op_mask) << SDMA_PKT_GCR_REQ_HEADER_op_shift)
SDMA_PKT_GCR_REQ_HEADER_sub_op_offset = 0
SDMA_PKT_GCR_REQ_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_GCR_REQ_HEADER_sub_op_shift = 8
SDMA_PKT_GCR_REQ_HEADER_SUB_OP = lambda x: (((x) & SDMA_PKT_GCR_REQ_HEADER_sub_op_mask) << SDMA_PKT_GCR_REQ_HEADER_sub_op_shift)
SDMA_PKT_GCR_REQ_PAYLOAD1_base_va_31_7_offset = 1
SDMA_PKT_GCR_REQ_PAYLOAD1_base_va_31_7_mask = 0x01FFFFFF
SDMA_PKT_GCR_REQ_PAYLOAD1_base_va_31_7_shift = 7
SDMA_PKT_GCR_REQ_PAYLOAD1_BASE_VA_31_7 = lambda x: (((x) & SDMA_PKT_GCR_REQ_PAYLOAD1_base_va_31_7_mask) << SDMA_PKT_GCR_REQ_PAYLOAD1_base_va_31_7_shift)
SDMA_PKT_GCR_REQ_PAYLOAD2_base_va_47_32_offset = 2
SDMA_PKT_GCR_REQ_PAYLOAD2_base_va_47_32_mask = 0x0000FFFF
SDMA_PKT_GCR_REQ_PAYLOAD2_base_va_47_32_shift = 0
SDMA_PKT_GCR_REQ_PAYLOAD2_BASE_VA_47_32 = lambda x: (((x) & SDMA_PKT_GCR_REQ_PAYLOAD2_base_va_47_32_mask) << SDMA_PKT_GCR_REQ_PAYLOAD2_base_va_47_32_shift)
SDMA_PKT_GCR_REQ_PAYLOAD2_gcr_control_15_0_offset = 2
SDMA_PKT_GCR_REQ_PAYLOAD2_gcr_control_15_0_mask = 0x0000FFFF
SDMA_PKT_GCR_REQ_PAYLOAD2_gcr_control_15_0_shift = 16
SDMA_PKT_GCR_REQ_PAYLOAD2_GCR_CONTROL_15_0 = lambda x: (((x) & SDMA_PKT_GCR_REQ_PAYLOAD2_gcr_control_15_0_mask) << SDMA_PKT_GCR_REQ_PAYLOAD2_gcr_control_15_0_shift)
SDMA_PKT_GCR_REQ_PAYLOAD3_gcr_control_18_16_offset = 3
SDMA_PKT_GCR_REQ_PAYLOAD3_gcr_control_18_16_mask = 0x00000007
SDMA_PKT_GCR_REQ_PAYLOAD3_gcr_control_18_16_shift = 0
SDMA_PKT_GCR_REQ_PAYLOAD3_GCR_CONTROL_18_16 = lambda x: (((x) & SDMA_PKT_GCR_REQ_PAYLOAD3_gcr_control_18_16_mask) << SDMA_PKT_GCR_REQ_PAYLOAD3_gcr_control_18_16_shift)
SDMA_PKT_GCR_REQ_PAYLOAD3_limit_va_31_7_offset = 3
SDMA_PKT_GCR_REQ_PAYLOAD3_limit_va_31_7_mask = 0x01FFFFFF
SDMA_PKT_GCR_REQ_PAYLOAD3_limit_va_31_7_shift = 7
SDMA_PKT_GCR_REQ_PAYLOAD3_LIMIT_VA_31_7 = lambda x: (((x) & SDMA_PKT_GCR_REQ_PAYLOAD3_limit_va_31_7_mask) << SDMA_PKT_GCR_REQ_PAYLOAD3_limit_va_31_7_shift)
SDMA_PKT_GCR_REQ_PAYLOAD4_limit_va_47_32_offset = 4
SDMA_PKT_GCR_REQ_PAYLOAD4_limit_va_47_32_mask = 0x0000FFFF
SDMA_PKT_GCR_REQ_PAYLOAD4_limit_va_47_32_shift = 0
SDMA_PKT_GCR_REQ_PAYLOAD4_LIMIT_VA_47_32 = lambda x: (((x) & SDMA_PKT_GCR_REQ_PAYLOAD4_limit_va_47_32_mask) << SDMA_PKT_GCR_REQ_PAYLOAD4_limit_va_47_32_shift)
SDMA_PKT_GCR_REQ_PAYLOAD4_vmid_offset = 4
SDMA_PKT_GCR_REQ_PAYLOAD4_vmid_mask = 0x0000000F
SDMA_PKT_GCR_REQ_PAYLOAD4_vmid_shift = 24
SDMA_PKT_GCR_REQ_PAYLOAD4_VMID = lambda x: (((x) & SDMA_PKT_GCR_REQ_PAYLOAD4_vmid_mask) << SDMA_PKT_GCR_REQ_PAYLOAD4_vmid_shift)
SDMA_PKT_NOP_HEADER_op_offset = 0
SDMA_PKT_NOP_HEADER_op_mask = 0x000000FF
SDMA_PKT_NOP_HEADER_op_shift = 0
SDMA_PKT_NOP_HEADER_OP = lambda x: (((x) & SDMA_PKT_NOP_HEADER_op_mask) << SDMA_PKT_NOP_HEADER_op_shift)
SDMA_PKT_NOP_HEADER_sub_op_offset = 0
SDMA_PKT_NOP_HEADER_sub_op_mask = 0x000000FF
SDMA_PKT_NOP_HEADER_sub_op_shift = 8
SDMA_PKT_NOP_HEADER_SUB_OP = lambda x: (((x) & SDMA_PKT_NOP_HEADER_sub_op_mask) << SDMA_PKT_NOP_HEADER_sub_op_shift)
SDMA_PKT_NOP_HEADER_count_offset = 0
SDMA_PKT_NOP_HEADER_count_mask = 0x00003FFF
SDMA_PKT_NOP_HEADER_count_shift = 16
SDMA_PKT_NOP_HEADER_COUNT = lambda x: (((x) & SDMA_PKT_NOP_HEADER_count_mask) << SDMA_PKT_NOP_HEADER_count_shift)
SDMA_PKT_NOP_DATA0_data0_offset = 1
SDMA_PKT_NOP_DATA0_data0_mask = 0xFFFFFFFF
SDMA_PKT_NOP_DATA0_data0_shift = 0
SDMA_PKT_NOP_DATA0_DATA0 = lambda x: (((x) & SDMA_PKT_NOP_DATA0_data0_mask) << SDMA_PKT_NOP_DATA0_data0_shift)
SDMA_AQL_PKT_HEADER_HEADER_format_offset = 0
SDMA_AQL_PKT_HEADER_HEADER_format_mask = 0x000000FF
SDMA_AQL_PKT_HEADER_HEADER_format_shift = 0
SDMA_AQL_PKT_HEADER_HEADER_FORMAT = lambda x: (((x) & SDMA_AQL_PKT_HEADER_HEADER_format_mask) << SDMA_AQL_PKT_HEADER_HEADER_format_shift)
SDMA_AQL_PKT_HEADER_HEADER_barrier_offset = 0
SDMA_AQL_PKT_HEADER_HEADER_barrier_mask = 0x00000001
SDMA_AQL_PKT_HEADER_HEADER_barrier_shift = 8
SDMA_AQL_PKT_HEADER_HEADER_BARRIER = lambda x: (((x) & SDMA_AQL_PKT_HEADER_HEADER_barrier_mask) << SDMA_AQL_PKT_HEADER_HEADER_barrier_shift)
SDMA_AQL_PKT_HEADER_HEADER_acquire_fence_scope_offset = 0
SDMA_AQL_PKT_HEADER_HEADER_acquire_fence_scope_mask = 0x00000003
SDMA_AQL_PKT_HEADER_HEADER_acquire_fence_scope_shift = 9
SDMA_AQL_PKT_HEADER_HEADER_ACQUIRE_FENCE_SCOPE = lambda x: (((x) & SDMA_AQL_PKT_HEADER_HEADER_acquire_fence_scope_mask) << SDMA_AQL_PKT_HEADER_HEADER_acquire_fence_scope_shift)
SDMA_AQL_PKT_HEADER_HEADER_release_fence_scope_offset = 0
SDMA_AQL_PKT_HEADER_HEADER_release_fence_scope_mask = 0x00000003
SDMA_AQL_PKT_HEADER_HEADER_release_fence_scope_shift = 11
SDMA_AQL_PKT_HEADER_HEADER_RELEASE_FENCE_SCOPE = lambda x: (((x) & SDMA_AQL_PKT_HEADER_HEADER_release_fence_scope_mask) << SDMA_AQL_PKT_HEADER_HEADER_release_fence_scope_shift)
SDMA_AQL_PKT_HEADER_HEADER_reserved_offset = 0
SDMA_AQL_PKT_HEADER_HEADER_reserved_mask = 0x00000007
SDMA_AQL_PKT_HEADER_HEADER_reserved_shift = 13
SDMA_AQL_PKT_HEADER_HEADER_RESERVED = lambda x: (((x) & SDMA_AQL_PKT_HEADER_HEADER_reserved_mask) << SDMA_AQL_PKT_HEADER_HEADER_reserved_shift)
SDMA_AQL_PKT_HEADER_HEADER_op_offset = 0
SDMA_AQL_PKT_HEADER_HEADER_op_mask = 0x0000000F
SDMA_AQL_PKT_HEADER_HEADER_op_shift = 16
SDMA_AQL_PKT_HEADER_HEADER_OP = lambda x: (((x) & SDMA_AQL_PKT_HEADER_HEADER_op_mask) << SDMA_AQL_PKT_HEADER_HEADER_op_shift)
SDMA_AQL_PKT_HEADER_HEADER_subop_offset = 0
SDMA_AQL_PKT_HEADER_HEADER_subop_mask = 0x00000007
SDMA_AQL_PKT_HEADER_HEADER_subop_shift = 20
SDMA_AQL_PKT_HEADER_HEADER_SUBOP = lambda x: (((x) & SDMA_AQL_PKT_HEADER_HEADER_subop_mask) << SDMA_AQL_PKT_HEADER_HEADER_subop_shift)
SDMA_AQL_PKT_HEADER_HEADER_cpv_offset = 0
SDMA_AQL_PKT_HEADER_HEADER_cpv_mask = 0x00000001
SDMA_AQL_PKT_HEADER_HEADER_cpv_shift = 28
SDMA_AQL_PKT_HEADER_HEADER_CPV = lambda x: (((x) & SDMA_AQL_PKT_HEADER_HEADER_cpv_mask) << SDMA_AQL_PKT_HEADER_HEADER_cpv_shift)
SDMA_AQL_PKT_COPY_LINEAR_HEADER_format_offset = 0
SDMA_AQL_PKT_COPY_LINEAR_HEADER_format_mask = 0x000000FF
SDMA_AQL_PKT_COPY_LINEAR_HEADER_format_shift = 0
SDMA_AQL_PKT_COPY_LINEAR_HEADER_FORMAT = lambda x: (((x) & SDMA_AQL_PKT_COPY_LINEAR_HEADER_format_mask) << SDMA_AQL_PKT_COPY_LINEAR_HEADER_format_shift)
SDMA_AQL_PKT_COPY_LINEAR_HEADER_barrier_offset = 0
SDMA_AQL_PKT_COPY_LINEAR_HEADER_barrier_mask = 0x00000001
SDMA_AQL_PKT_COPY_LINEAR_HEADER_barrier_shift = 8
SDMA_AQL_PKT_COPY_LINEAR_HEADER_BARRIER = lambda x: (((x) & SDMA_AQL_PKT_COPY_LINEAR_HEADER_barrier_mask) << SDMA_AQL_PKT_COPY_LINEAR_HEADER_barrier_shift)
SDMA_AQL_PKT_COPY_LINEAR_HEADER_acquire_fence_scope_offset = 0
SDMA_AQL_PKT_COPY_LINEAR_HEADER_acquire_fence_scope_mask = 0x00000003
SDMA_AQL_PKT_COPY_LINEAR_HEADER_acquire_fence_scope_shift = 9
SDMA_AQL_PKT_COPY_LINEAR_HEADER_ACQUIRE_FENCE_SCOPE = lambda x: (((x) & SDMA_AQL_PKT_COPY_LINEAR_HEADER_acquire_fence_scope_mask) << SDMA_AQL_PKT_COPY_LINEAR_HEADER_acquire_fence_scope_shift)
SDMA_AQL_PKT_COPY_LINEAR_HEADER_release_fence_scope_offset = 0
SDMA_AQL_PKT_COPY_LINEAR_HEADER_release_fence_scope_mask = 0x00000003
SDMA_AQL_PKT_COPY_LINEAR_HEADER_release_fence_scope_shift = 11
SDMA_AQL_PKT_COPY_LINEAR_HEADER_RELEASE_FENCE_SCOPE = lambda x: (((x) & SDMA_AQL_PKT_COPY_LINEAR_HEADER_release_fence_scope_mask) << SDMA_AQL_PKT_COPY_LINEAR_HEADER_release_fence_scope_shift)
SDMA_AQL_PKT_COPY_LINEAR_HEADER_reserved_offset = 0
SDMA_AQL_PKT_COPY_LINEAR_HEADER_reserved_mask = 0x00000007
SDMA_AQL_PKT_COPY_LINEAR_HEADER_reserved_shift = 13
SDMA_AQL_PKT_COPY_LINEAR_HEADER_RESERVED = lambda x: (((x) & SDMA_AQL_PKT_COPY_LINEAR_HEADER_reserved_mask) << SDMA_AQL_PKT_COPY_LINEAR_HEADER_reserved_shift)
SDMA_AQL_PKT_COPY_LINEAR_HEADER_op_offset = 0
SDMA_AQL_PKT_COPY_LINEAR_HEADER_op_mask = 0x0000000F
SDMA_AQL_PKT_COPY_LINEAR_HEADER_op_shift = 16
SDMA_AQL_PKT_COPY_LINEAR_HEADER_OP = lambda x: (((x) & SDMA_AQL_PKT_COPY_LINEAR_HEADER_op_mask) << SDMA_AQL_PKT_COPY_LINEAR_HEADER_op_shift)
SDMA_AQL_PKT_COPY_LINEAR_HEADER_subop_offset = 0
SDMA_AQL_PKT_COPY_LINEAR_HEADER_subop_mask = 0x00000007
SDMA_AQL_PKT_COPY_LINEAR_HEADER_subop_shift = 20
SDMA_AQL_PKT_COPY_LINEAR_HEADER_SUBOP = lambda x: (((x) & SDMA_AQL_PKT_COPY_LINEAR_HEADER_subop_mask) << SDMA_AQL_PKT_COPY_LINEAR_HEADER_subop_shift)
SDMA_AQL_PKT_COPY_LINEAR_HEADER_cpv_offset = 0
SDMA_AQL_PKT_COPY_LINEAR_HEADER_cpv_mask = 0x00000001
SDMA_AQL_PKT_COPY_LINEAR_HEADER_cpv_shift = 28
SDMA_AQL_PKT_COPY_LINEAR_HEADER_CPV = lambda x: (((x) & SDMA_AQL_PKT_COPY_LINEAR_HEADER_cpv_mask) << SDMA_AQL_PKT_COPY_LINEAR_HEADER_cpv_shift)
SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW1_reserved_dw1_offset = 1
SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW1_reserved_dw1_mask = 0xFFFFFFFF
SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW1_reserved_dw1_shift = 0
SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW1_RESERVED_DW1 = lambda x: (((x) & SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW1_reserved_dw1_mask) << SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW1_reserved_dw1_shift)
SDMA_AQL_PKT_COPY_LINEAR_RETURN_ADDR_LO_return_addr_31_0_offset = 2
SDMA_AQL_PKT_COPY_LINEAR_RETURN_ADDR_LO_return_addr_31_0_mask = 0xFFFFFFFF
SDMA_AQL_PKT_COPY_LINEAR_RETURN_ADDR_LO_return_addr_31_0_shift = 0
SDMA_AQL_PKT_COPY_LINEAR_RETURN_ADDR_LO_RETURN_ADDR_31_0 = lambda x: (((x) & SDMA_AQL_PKT_COPY_LINEAR_RETURN_ADDR_LO_return_addr_31_0_mask) << SDMA_AQL_PKT_COPY_LINEAR_RETURN_ADDR_LO_return_addr_31_0_shift)
SDMA_AQL_PKT_COPY_LINEAR_RETURN_ADDR_HI_return_addr_63_32_offset = 3
SDMA_AQL_PKT_COPY_LINEAR_RETURN_ADDR_HI_return_addr_63_32_mask = 0xFFFFFFFF
SDMA_AQL_PKT_COPY_LINEAR_RETURN_ADDR_HI_return_addr_63_32_shift = 0
SDMA_AQL_PKT_COPY_LINEAR_RETURN_ADDR_HI_RETURN_ADDR_63_32 = lambda x: (((x) & SDMA_AQL_PKT_COPY_LINEAR_RETURN_ADDR_HI_return_addr_63_32_mask) << SDMA_AQL_PKT_COPY_LINEAR_RETURN_ADDR_HI_return_addr_63_32_shift)
SDMA_AQL_PKT_COPY_LINEAR_COUNT_count_offset = 4
SDMA_AQL_PKT_COPY_LINEAR_COUNT_count_mask = 0x003FFFFF
SDMA_AQL_PKT_COPY_LINEAR_COUNT_count_shift = 0
SDMA_AQL_PKT_COPY_LINEAR_COUNT_COUNT = lambda x: (((x) & SDMA_AQL_PKT_COPY_LINEAR_COUNT_count_mask) << SDMA_AQL_PKT_COPY_LINEAR_COUNT_count_shift)
SDMA_AQL_PKT_COPY_LINEAR_PARAMETER_dst_sw_offset = 5
SDMA_AQL_PKT_COPY_LINEAR_PARAMETER_dst_sw_mask = 0x00000003
SDMA_AQL_PKT_COPY_LINEAR_PARAMETER_dst_sw_shift = 16
SDMA_AQL_PKT_COPY_LINEAR_PARAMETER_DST_SW = lambda x: (((x) & SDMA_AQL_PKT_COPY_LINEAR_PARAMETER_dst_sw_mask) << SDMA_AQL_PKT_COPY_LINEAR_PARAMETER_dst_sw_shift)
SDMA_AQL_PKT_COPY_LINEAR_PARAMETER_dst_cache_policy_offset = 5
SDMA_AQL_PKT_COPY_LINEAR_PARAMETER_dst_cache_policy_mask = 0x00000007
SDMA_AQL_PKT_COPY_LINEAR_PARAMETER_dst_cache_policy_shift = 18
SDMA_AQL_PKT_COPY_LINEAR_PARAMETER_DST_CACHE_POLICY = lambda x: (((x) & SDMA_AQL_PKT_COPY_LINEAR_PARAMETER_dst_cache_policy_mask) << SDMA_AQL_PKT_COPY_LINEAR_PARAMETER_dst_cache_policy_shift)
SDMA_AQL_PKT_COPY_LINEAR_PARAMETER_src_sw_offset = 5
SDMA_AQL_PKT_COPY_LINEAR_PARAMETER_src_sw_mask = 0x00000003
SDMA_AQL_PKT_COPY_LINEAR_PARAMETER_src_sw_shift = 24
SDMA_AQL_PKT_COPY_LINEAR_PARAMETER_SRC_SW = lambda x: (((x) & SDMA_AQL_PKT_COPY_LINEAR_PARAMETER_src_sw_mask) << SDMA_AQL_PKT_COPY_LINEAR_PARAMETER_src_sw_shift)
SDMA_AQL_PKT_COPY_LINEAR_PARAMETER_src_cache_policy_offset = 5
SDMA_AQL_PKT_COPY_LINEAR_PARAMETER_src_cache_policy_mask = 0x00000007
SDMA_AQL_PKT_COPY_LINEAR_PARAMETER_src_cache_policy_shift = 26
SDMA_AQL_PKT_COPY_LINEAR_PARAMETER_SRC_CACHE_POLICY = lambda x: (((x) & SDMA_AQL_PKT_COPY_LINEAR_PARAMETER_src_cache_policy_mask) << SDMA_AQL_PKT_COPY_LINEAR_PARAMETER_src_cache_policy_shift)
SDMA_AQL_PKT_COPY_LINEAR_SRC_ADDR_LO_src_addr_31_0_offset = 6
SDMA_AQL_PKT_COPY_LINEAR_SRC_ADDR_LO_src_addr_31_0_mask = 0xFFFFFFFF
SDMA_AQL_PKT_COPY_LINEAR_SRC_ADDR_LO_src_addr_31_0_shift = 0
SDMA_AQL_PKT_COPY_LINEAR_SRC_ADDR_LO_SRC_ADDR_31_0 = lambda x: (((x) & SDMA_AQL_PKT_COPY_LINEAR_SRC_ADDR_LO_src_addr_31_0_mask) << SDMA_AQL_PKT_COPY_LINEAR_SRC_ADDR_LO_src_addr_31_0_shift)
SDMA_AQL_PKT_COPY_LINEAR_SRC_ADDR_HI_src_addr_63_32_offset = 7
SDMA_AQL_PKT_COPY_LINEAR_SRC_ADDR_HI_src_addr_63_32_mask = 0xFFFFFFFF
SDMA_AQL_PKT_COPY_LINEAR_SRC_ADDR_HI_src_addr_63_32_shift = 0
SDMA_AQL_PKT_COPY_LINEAR_SRC_ADDR_HI_SRC_ADDR_63_32 = lambda x: (((x) & SDMA_AQL_PKT_COPY_LINEAR_SRC_ADDR_HI_src_addr_63_32_mask) << SDMA_AQL_PKT_COPY_LINEAR_SRC_ADDR_HI_src_addr_63_32_shift)
SDMA_AQL_PKT_COPY_LINEAR_DST_ADDR_LO_dst_addr_31_0_offset = 8
SDMA_AQL_PKT_COPY_LINEAR_DST_ADDR_LO_dst_addr_31_0_mask = 0xFFFFFFFF
SDMA_AQL_PKT_COPY_LINEAR_DST_ADDR_LO_dst_addr_31_0_shift = 0
SDMA_AQL_PKT_COPY_LINEAR_DST_ADDR_LO_DST_ADDR_31_0 = lambda x: (((x) & SDMA_AQL_PKT_COPY_LINEAR_DST_ADDR_LO_dst_addr_31_0_mask) << SDMA_AQL_PKT_COPY_LINEAR_DST_ADDR_LO_dst_addr_31_0_shift)
SDMA_AQL_PKT_COPY_LINEAR_DST_ADDR_HI_dst_addr_63_32_offset = 9
SDMA_AQL_PKT_COPY_LINEAR_DST_ADDR_HI_dst_addr_63_32_mask = 0xFFFFFFFF
SDMA_AQL_PKT_COPY_LINEAR_DST_ADDR_HI_dst_addr_63_32_shift = 0
SDMA_AQL_PKT_COPY_LINEAR_DST_ADDR_HI_DST_ADDR_63_32 = lambda x: (((x) & SDMA_AQL_PKT_COPY_LINEAR_DST_ADDR_HI_dst_addr_63_32_mask) << SDMA_AQL_PKT_COPY_LINEAR_DST_ADDR_HI_dst_addr_63_32_shift)
SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW10_reserved_dw10_offset = 10
SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW10_reserved_dw10_mask = 0xFFFFFFFF
SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW10_reserved_dw10_shift = 0
SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW10_RESERVED_DW10 = lambda x: (((x) & SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW10_reserved_dw10_mask) << SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW10_reserved_dw10_shift)
SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW11_reserved_dw11_offset = 11
SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW11_reserved_dw11_mask = 0xFFFFFFFF
SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW11_reserved_dw11_shift = 0
SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW11_RESERVED_DW11 = lambda x: (((x) & SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW11_reserved_dw11_mask) << SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW11_reserved_dw11_shift)
SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW12_reserved_dw12_offset = 12
SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW12_reserved_dw12_mask = 0xFFFFFFFF
SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW12_reserved_dw12_shift = 0
SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW12_RESERVED_DW12 = lambda x: (((x) & SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW12_reserved_dw12_mask) << SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW12_reserved_dw12_shift)
SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW13_reserved_dw13_offset = 13
SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW13_reserved_dw13_mask = 0xFFFFFFFF
SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW13_reserved_dw13_shift = 0
SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW13_RESERVED_DW13 = lambda x: (((x) & SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW13_reserved_dw13_mask) << SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW13_reserved_dw13_shift)
SDMA_AQL_PKT_COPY_LINEAR_COMPLETION_SIGNAL_LO_completion_signal_31_0_offset = 14
SDMA_AQL_PKT_COPY_LINEAR_COMPLETION_SIGNAL_LO_completion_signal_31_0_mask = 0xFFFFFFFF
SDMA_AQL_PKT_COPY_LINEAR_COMPLETION_SIGNAL_LO_completion_signal_31_0_shift = 0
SDMA_AQL_PKT_COPY_LINEAR_COMPLETION_SIGNAL_LO_COMPLETION_SIGNAL_31_0 = lambda x: (((x) & SDMA_AQL_PKT_COPY_LINEAR_COMPLETION_SIGNAL_LO_completion_signal_31_0_mask) << SDMA_AQL_PKT_COPY_LINEAR_COMPLETION_SIGNAL_LO_completion_signal_31_0_shift)
SDMA_AQL_PKT_COPY_LINEAR_COMPLETION_SIGNAL_HI_completion_signal_63_32_offset = 15
SDMA_AQL_PKT_COPY_LINEAR_COMPLETION_SIGNAL_HI_completion_signal_63_32_mask = 0xFFFFFFFF
SDMA_AQL_PKT_COPY_LINEAR_COMPLETION_SIGNAL_HI_completion_signal_63_32_shift = 0
SDMA_AQL_PKT_COPY_LINEAR_COMPLETION_SIGNAL_HI_COMPLETION_SIGNAL_63_32 = lambda x: (((x) & SDMA_AQL_PKT_COPY_LINEAR_COMPLETION_SIGNAL_HI_completion_signal_63_32_mask) << SDMA_AQL_PKT_COPY_LINEAR_COMPLETION_SIGNAL_HI_completion_signal_63_32_shift)
SDMA_AQL_PKT_BARRIER_OR_HEADER_format_offset = 0
SDMA_AQL_PKT_BARRIER_OR_HEADER_format_mask = 0x000000FF
SDMA_AQL_PKT_BARRIER_OR_HEADER_format_shift = 0
SDMA_AQL_PKT_BARRIER_OR_HEADER_FORMAT = lambda x: (((x) & SDMA_AQL_PKT_BARRIER_OR_HEADER_format_mask) << SDMA_AQL_PKT_BARRIER_OR_HEADER_format_shift)
SDMA_AQL_PKT_BARRIER_OR_HEADER_barrier_offset = 0
SDMA_AQL_PKT_BARRIER_OR_HEADER_barrier_mask = 0x00000001
SDMA_AQL_PKT_BARRIER_OR_HEADER_barrier_shift = 8
SDMA_AQL_PKT_BARRIER_OR_HEADER_BARRIER = lambda x: (((x) & SDMA_AQL_PKT_BARRIER_OR_HEADER_barrier_mask) << SDMA_AQL_PKT_BARRIER_OR_HEADER_barrier_shift)
SDMA_AQL_PKT_BARRIER_OR_HEADER_acquire_fence_scope_offset = 0
SDMA_AQL_PKT_BARRIER_OR_HEADER_acquire_fence_scope_mask = 0x00000003
SDMA_AQL_PKT_BARRIER_OR_HEADER_acquire_fence_scope_shift = 9
SDMA_AQL_PKT_BARRIER_OR_HEADER_ACQUIRE_FENCE_SCOPE = lambda x: (((x) & SDMA_AQL_PKT_BARRIER_OR_HEADER_acquire_fence_scope_mask) << SDMA_AQL_PKT_BARRIER_OR_HEADER_acquire_fence_scope_shift)
SDMA_AQL_PKT_BARRIER_OR_HEADER_release_fence_scope_offset = 0
SDMA_AQL_PKT_BARRIER_OR_HEADER_release_fence_scope_mask = 0x00000003
SDMA_AQL_PKT_BARRIER_OR_HEADER_release_fence_scope_shift = 11
SDMA_AQL_PKT_BARRIER_OR_HEADER_RELEASE_FENCE_SCOPE = lambda x: (((x) & SDMA_AQL_PKT_BARRIER_OR_HEADER_release_fence_scope_mask) << SDMA_AQL_PKT_BARRIER_OR_HEADER_release_fence_scope_shift)
SDMA_AQL_PKT_BARRIER_OR_HEADER_reserved_offset = 0
SDMA_AQL_PKT_BARRIER_OR_HEADER_reserved_mask = 0x00000007
SDMA_AQL_PKT_BARRIER_OR_HEADER_reserved_shift = 13
SDMA_AQL_PKT_BARRIER_OR_HEADER_RESERVED = lambda x: (((x) & SDMA_AQL_PKT_BARRIER_OR_HEADER_reserved_mask) << SDMA_AQL_PKT_BARRIER_OR_HEADER_reserved_shift)
SDMA_AQL_PKT_BARRIER_OR_HEADER_op_offset = 0
SDMA_AQL_PKT_BARRIER_OR_HEADER_op_mask = 0x0000000F
SDMA_AQL_PKT_BARRIER_OR_HEADER_op_shift = 16
SDMA_AQL_PKT_BARRIER_OR_HEADER_OP = lambda x: (((x) & SDMA_AQL_PKT_BARRIER_OR_HEADER_op_mask) << SDMA_AQL_PKT_BARRIER_OR_HEADER_op_shift)
SDMA_AQL_PKT_BARRIER_OR_HEADER_subop_offset = 0
SDMA_AQL_PKT_BARRIER_OR_HEADER_subop_mask = 0x00000007
SDMA_AQL_PKT_BARRIER_OR_HEADER_subop_shift = 20
SDMA_AQL_PKT_BARRIER_OR_HEADER_SUBOP = lambda x: (((x) & SDMA_AQL_PKT_BARRIER_OR_HEADER_subop_mask) << SDMA_AQL_PKT_BARRIER_OR_HEADER_subop_shift)
SDMA_AQL_PKT_BARRIER_OR_HEADER_cpv_offset = 0
SDMA_AQL_PKT_BARRIER_OR_HEADER_cpv_mask = 0x00000001
SDMA_AQL_PKT_BARRIER_OR_HEADER_cpv_shift = 28
SDMA_AQL_PKT_BARRIER_OR_HEADER_CPV = lambda x: (((x) & SDMA_AQL_PKT_BARRIER_OR_HEADER_cpv_mask) << SDMA_AQL_PKT_BARRIER_OR_HEADER_cpv_shift)
SDMA_AQL_PKT_BARRIER_OR_RESERVED_DW1_reserved_dw1_offset = 1
SDMA_AQL_PKT_BARRIER_OR_RESERVED_DW1_reserved_dw1_mask = 0xFFFFFFFF
SDMA_AQL_PKT_BARRIER_OR_RESERVED_DW1_reserved_dw1_shift = 0
SDMA_AQL_PKT_BARRIER_OR_RESERVED_DW1_RESERVED_DW1 = lambda x: (((x) & SDMA_AQL_PKT_BARRIER_OR_RESERVED_DW1_reserved_dw1_mask) << SDMA_AQL_PKT_BARRIER_OR_RESERVED_DW1_reserved_dw1_shift)
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_0_LO_dependent_addr_0_31_0_offset = 2
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_0_LO_dependent_addr_0_31_0_mask = 0xFFFFFFFF
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_0_LO_dependent_addr_0_31_0_shift = 0
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_0_LO_DEPENDENT_ADDR_0_31_0 = lambda x: (((x) & SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_0_LO_dependent_addr_0_31_0_mask) << SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_0_LO_dependent_addr_0_31_0_shift)
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_0_HI_dependent_addr_0_63_32_offset = 3
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_0_HI_dependent_addr_0_63_32_mask = 0xFFFFFFFF
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_0_HI_dependent_addr_0_63_32_shift = 0
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_0_HI_DEPENDENT_ADDR_0_63_32 = lambda x: (((x) & SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_0_HI_dependent_addr_0_63_32_mask) << SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_0_HI_dependent_addr_0_63_32_shift)
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_1_LO_dependent_addr_1_31_0_offset = 4
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_1_LO_dependent_addr_1_31_0_mask = 0xFFFFFFFF
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_1_LO_dependent_addr_1_31_0_shift = 0
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_1_LO_DEPENDENT_ADDR_1_31_0 = lambda x: (((x) & SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_1_LO_dependent_addr_1_31_0_mask) << SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_1_LO_dependent_addr_1_31_0_shift)
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_1_HI_dependent_addr_1_63_32_offset = 5
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_1_HI_dependent_addr_1_63_32_mask = 0xFFFFFFFF
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_1_HI_dependent_addr_1_63_32_shift = 0
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_1_HI_DEPENDENT_ADDR_1_63_32 = lambda x: (((x) & SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_1_HI_dependent_addr_1_63_32_mask) << SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_1_HI_dependent_addr_1_63_32_shift)
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_2_LO_dependent_addr_2_31_0_offset = 6
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_2_LO_dependent_addr_2_31_0_mask = 0xFFFFFFFF
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_2_LO_dependent_addr_2_31_0_shift = 0
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_2_LO_DEPENDENT_ADDR_2_31_0 = lambda x: (((x) & SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_2_LO_dependent_addr_2_31_0_mask) << SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_2_LO_dependent_addr_2_31_0_shift)
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_2_HI_dependent_addr_2_63_32_offset = 7
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_2_HI_dependent_addr_2_63_32_mask = 0xFFFFFFFF
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_2_HI_dependent_addr_2_63_32_shift = 0
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_2_HI_DEPENDENT_ADDR_2_63_32 = lambda x: (((x) & SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_2_HI_dependent_addr_2_63_32_mask) << SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_2_HI_dependent_addr_2_63_32_shift)
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_3_LO_dependent_addr_3_31_0_offset = 8
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_3_LO_dependent_addr_3_31_0_mask = 0xFFFFFFFF
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_3_LO_dependent_addr_3_31_0_shift = 0
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_3_LO_DEPENDENT_ADDR_3_31_0 = lambda x: (((x) & SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_3_LO_dependent_addr_3_31_0_mask) << SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_3_LO_dependent_addr_3_31_0_shift)
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_3_HI_dependent_addr_3_63_32_offset = 9
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_3_HI_dependent_addr_3_63_32_mask = 0xFFFFFFFF
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_3_HI_dependent_addr_3_63_32_shift = 0
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_3_HI_DEPENDENT_ADDR_3_63_32 = lambda x: (((x) & SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_3_HI_dependent_addr_3_63_32_mask) << SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_3_HI_dependent_addr_3_63_32_shift)
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_4_LO_dependent_addr_4_31_0_offset = 10
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_4_LO_dependent_addr_4_31_0_mask = 0xFFFFFFFF
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_4_LO_dependent_addr_4_31_0_shift = 0
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_4_LO_DEPENDENT_ADDR_4_31_0 = lambda x: (((x) & SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_4_LO_dependent_addr_4_31_0_mask) << SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_4_LO_dependent_addr_4_31_0_shift)
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_4_HI_dependent_addr_4_63_32_offset = 11
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_4_HI_dependent_addr_4_63_32_mask = 0xFFFFFFFF
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_4_HI_dependent_addr_4_63_32_shift = 0
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_4_HI_DEPENDENT_ADDR_4_63_32 = lambda x: (((x) & SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_4_HI_dependent_addr_4_63_32_mask) << SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_4_HI_dependent_addr_4_63_32_shift)
SDMA_AQL_PKT_BARRIER_OR_CACHE_POLICY_cache_policy0_offset = 12
SDMA_AQL_PKT_BARRIER_OR_CACHE_POLICY_cache_policy0_mask = 0x00000007
SDMA_AQL_PKT_BARRIER_OR_CACHE_POLICY_cache_policy0_shift = 0
SDMA_AQL_PKT_BARRIER_OR_CACHE_POLICY_CACHE_POLICY0 = lambda x: (((x) & SDMA_AQL_PKT_BARRIER_OR_CACHE_POLICY_cache_policy0_mask) << SDMA_AQL_PKT_BARRIER_OR_CACHE_POLICY_cache_policy0_shift)
SDMA_AQL_PKT_BARRIER_OR_CACHE_POLICY_cache_policy1_offset = 12
SDMA_AQL_PKT_BARRIER_OR_CACHE_POLICY_cache_policy1_mask = 0x00000007
SDMA_AQL_PKT_BARRIER_OR_CACHE_POLICY_cache_policy1_shift = 5
SDMA_AQL_PKT_BARRIER_OR_CACHE_POLICY_CACHE_POLICY1 = lambda x: (((x) & SDMA_AQL_PKT_BARRIER_OR_CACHE_POLICY_cache_policy1_mask) << SDMA_AQL_PKT_BARRIER_OR_CACHE_POLICY_cache_policy1_shift)
SDMA_AQL_PKT_BARRIER_OR_CACHE_POLICY_cache_policy2_offset = 12
SDMA_AQL_PKT_BARRIER_OR_CACHE_POLICY_cache_policy2_mask = 0x00000007
SDMA_AQL_PKT_BARRIER_OR_CACHE_POLICY_cache_policy2_shift = 10
SDMA_AQL_PKT_BARRIER_OR_CACHE_POLICY_CACHE_POLICY2 = lambda x: (((x) & SDMA_AQL_PKT_BARRIER_OR_CACHE_POLICY_cache_policy2_mask) << SDMA_AQL_PKT_BARRIER_OR_CACHE_POLICY_cache_policy2_shift)
SDMA_AQL_PKT_BARRIER_OR_CACHE_POLICY_cache_policy3_offset = 12
SDMA_AQL_PKT_BARRIER_OR_CACHE_POLICY_cache_policy3_mask = 0x00000007
SDMA_AQL_PKT_BARRIER_OR_CACHE_POLICY_cache_policy3_shift = 15
SDMA_AQL_PKT_BARRIER_OR_CACHE_POLICY_CACHE_POLICY3 = lambda x: (((x) & SDMA_AQL_PKT_BARRIER_OR_CACHE_POLICY_cache_policy3_mask) << SDMA_AQL_PKT_BARRIER_OR_CACHE_POLICY_cache_policy3_shift)
SDMA_AQL_PKT_BARRIER_OR_CACHE_POLICY_cache_policy4_offset = 12
SDMA_AQL_PKT_BARRIER_OR_CACHE_POLICY_cache_policy4_mask = 0x00000007
SDMA_AQL_PKT_BARRIER_OR_CACHE_POLICY_cache_policy4_shift = 20
SDMA_AQL_PKT_BARRIER_OR_CACHE_POLICY_CACHE_POLICY4 = lambda x: (((x) & SDMA_AQL_PKT_BARRIER_OR_CACHE_POLICY_cache_policy4_mask) << SDMA_AQL_PKT_BARRIER_OR_CACHE_POLICY_cache_policy4_shift)
SDMA_AQL_PKT_BARRIER_OR_RESERVED_DW13_reserved_dw13_offset = 13
SDMA_AQL_PKT_BARRIER_OR_RESERVED_DW13_reserved_dw13_mask = 0xFFFFFFFF
SDMA_AQL_PKT_BARRIER_OR_RESERVED_DW13_reserved_dw13_shift = 0
SDMA_AQL_PKT_BARRIER_OR_RESERVED_DW13_RESERVED_DW13 = lambda x: (((x) & SDMA_AQL_PKT_BARRIER_OR_RESERVED_DW13_reserved_dw13_mask) << SDMA_AQL_PKT_BARRIER_OR_RESERVED_DW13_reserved_dw13_shift)
SDMA_AQL_PKT_BARRIER_OR_COMPLETION_SIGNAL_LO_completion_signal_31_0_offset = 14
SDMA_AQL_PKT_BARRIER_OR_COMPLETION_SIGNAL_LO_completion_signal_31_0_mask = 0xFFFFFFFF
SDMA_AQL_PKT_BARRIER_OR_COMPLETION_SIGNAL_LO_completion_signal_31_0_shift = 0
SDMA_AQL_PKT_BARRIER_OR_COMPLETION_SIGNAL_LO_COMPLETION_SIGNAL_31_0 = lambda x: (((x) & SDMA_AQL_PKT_BARRIER_OR_COMPLETION_SIGNAL_LO_completion_signal_31_0_mask) << SDMA_AQL_PKT_BARRIER_OR_COMPLETION_SIGNAL_LO_completion_signal_31_0_shift)
SDMA_AQL_PKT_BARRIER_OR_COMPLETION_SIGNAL_HI_completion_signal_63_32_offset = 15
SDMA_AQL_PKT_BARRIER_OR_COMPLETION_SIGNAL_HI_completion_signal_63_32_mask = 0xFFFFFFFF
SDMA_AQL_PKT_BARRIER_OR_COMPLETION_SIGNAL_HI_completion_signal_63_32_shift = 0
SDMA_AQL_PKT_BARRIER_OR_COMPLETION_SIGNAL_HI_COMPLETION_SIGNAL_63_32 = lambda x: (((x) & SDMA_AQL_PKT_BARRIER_OR_COMPLETION_SIGNAL_HI_completion_signal_63_32_mask) << SDMA_AQL_PKT_BARRIER_OR_COMPLETION_SIGNAL_HI_completion_signal_63_32_shift)