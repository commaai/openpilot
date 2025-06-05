# mypy: ignore-errors
# -*- coding: utf-8 -*-
#
# TARGET arch is: ['-I/opt/rocm/include', '-x', 'c++']
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





HSA_RUNTIME_CORE_INC_SDMA_REGISTERS_H_ = True # macro
SDMA_OP_COPY = 1 # macro
SDMA_OP_FENCE = 5 # macro
SDMA_OP_TRAP = 6 # macro
SDMA_OP_POLL_REGMEM = 8 # macro
SDMA_OP_ATOMIC = 10 # macro
SDMA_OP_CONST_FILL = 11 # macro
SDMA_OP_TIMESTAMP = 13 # macro
SDMA_OP_GCR = 17 # Variable ctypes.c_uint32
SDMA_SUBOP_COPY_LINEAR = 0 # macro
SDMA_SUBOP_COPY_LINEAR_RECT = 4 # Variable ctypes.c_uint32
SDMA_SUBOP_TIMESTAMP_GET_GLOBAL = 2 # macro
SDMA_SUBOP_USER_GCR = 1 # Variable ctypes.c_uint32
SDMA_ATOMIC_ADD64 = 47 # Variable ctypes.c_uint32
class struct_SDMA_PKT_COPY_LINEAR_TAG(Structure):
    pass

class union_SDMA_PKT_COPY_LINEAR_TAG_HEADER_UNION(Union):
    pass

class struct_SDMA_PKT_COPY_LINEAR_TAG_0_0(Structure):
    pass

struct_SDMA_PKT_COPY_LINEAR_TAG_0_0._pack_ = 1 # source:False
struct_SDMA_PKT_COPY_LINEAR_TAG_0_0._fields_ = [
    ('op', ctypes.c_uint32, 8),
    ('sub_op', ctypes.c_uint32, 8),
    ('extra_info', ctypes.c_uint32, 16),
]

union_SDMA_PKT_COPY_LINEAR_TAG_HEADER_UNION._pack_ = 1 # source:False
union_SDMA_PKT_COPY_LINEAR_TAG_HEADER_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_COPY_LINEAR_TAG_HEADER_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_COPY_LINEAR_TAG_0_0),
    ('DW_0_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_COPY_LINEAR_TAG_COUNT_UNION(Union):
    pass

class struct_SDMA_PKT_COPY_LINEAR_TAG_1_0(Structure):
    pass

struct_SDMA_PKT_COPY_LINEAR_TAG_1_0._pack_ = 1 # source:False
struct_SDMA_PKT_COPY_LINEAR_TAG_1_0._fields_ = [
    ('count', ctypes.c_uint32, 22),
    ('reserved_0', ctypes.c_uint32, 10),
]

union_SDMA_PKT_COPY_LINEAR_TAG_COUNT_UNION._pack_ = 1 # source:False
union_SDMA_PKT_COPY_LINEAR_TAG_COUNT_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_COPY_LINEAR_TAG_COUNT_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_COPY_LINEAR_TAG_1_0),
    ('DW_1_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_COPY_LINEAR_TAG_PARAMETER_UNION(Union):
    pass

class struct_SDMA_PKT_COPY_LINEAR_TAG_2_0(Structure):
    pass

struct_SDMA_PKT_COPY_LINEAR_TAG_2_0._pack_ = 1 # source:False
struct_SDMA_PKT_COPY_LINEAR_TAG_2_0._fields_ = [
    ('reserved_0', ctypes.c_uint32, 16),
    ('dst_swap', ctypes.c_uint32, 2),
    ('reserved_1', ctypes.c_uint32, 6),
    ('src_swap', ctypes.c_uint32, 2),
    ('reserved_2', ctypes.c_uint32, 6),
]

union_SDMA_PKT_COPY_LINEAR_TAG_PARAMETER_UNION._pack_ = 1 # source:False
union_SDMA_PKT_COPY_LINEAR_TAG_PARAMETER_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_COPY_LINEAR_TAG_PARAMETER_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_COPY_LINEAR_TAG_2_0),
    ('DW_2_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_COPY_LINEAR_TAG_SRC_ADDR_LO_UNION(Union):
    pass

class struct_SDMA_PKT_COPY_LINEAR_TAG_3_0(Structure):
    pass

struct_SDMA_PKT_COPY_LINEAR_TAG_3_0._pack_ = 1 # source:False
struct_SDMA_PKT_COPY_LINEAR_TAG_3_0._fields_ = [
    ('src_addr_31_0', ctypes.c_uint32, 32),
]

union_SDMA_PKT_COPY_LINEAR_TAG_SRC_ADDR_LO_UNION._pack_ = 1 # source:False
union_SDMA_PKT_COPY_LINEAR_TAG_SRC_ADDR_LO_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_COPY_LINEAR_TAG_SRC_ADDR_LO_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_COPY_LINEAR_TAG_3_0),
    ('DW_3_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_COPY_LINEAR_TAG_SRC_ADDR_HI_UNION(Union):
    pass

class struct_SDMA_PKT_COPY_LINEAR_TAG_4_0(Structure):
    pass

struct_SDMA_PKT_COPY_LINEAR_TAG_4_0._pack_ = 1 # source:False
struct_SDMA_PKT_COPY_LINEAR_TAG_4_0._fields_ = [
    ('src_addr_63_32', ctypes.c_uint32, 32),
]

union_SDMA_PKT_COPY_LINEAR_TAG_SRC_ADDR_HI_UNION._pack_ = 1 # source:False
union_SDMA_PKT_COPY_LINEAR_TAG_SRC_ADDR_HI_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_COPY_LINEAR_TAG_SRC_ADDR_HI_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_COPY_LINEAR_TAG_4_0),
    ('DW_4_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_COPY_LINEAR_TAG_DST_ADDR_LO_UNION(Union):
    pass

class struct_SDMA_PKT_COPY_LINEAR_TAG_5_0(Structure):
    pass

struct_SDMA_PKT_COPY_LINEAR_TAG_5_0._pack_ = 1 # source:False
struct_SDMA_PKT_COPY_LINEAR_TAG_5_0._fields_ = [
    ('dst_addr_31_0', ctypes.c_uint32, 32),
]

union_SDMA_PKT_COPY_LINEAR_TAG_DST_ADDR_LO_UNION._pack_ = 1 # source:False
union_SDMA_PKT_COPY_LINEAR_TAG_DST_ADDR_LO_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_COPY_LINEAR_TAG_DST_ADDR_LO_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_COPY_LINEAR_TAG_5_0),
    ('DW_5_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_COPY_LINEAR_TAG_DST_ADDR_HI_UNION(Union):
    pass

class struct_SDMA_PKT_COPY_LINEAR_TAG_6_0(Structure):
    pass

struct_SDMA_PKT_COPY_LINEAR_TAG_6_0._pack_ = 1 # source:False
struct_SDMA_PKT_COPY_LINEAR_TAG_6_0._fields_ = [
    ('dst_addr_63_32', ctypes.c_uint32, 32),
]

union_SDMA_PKT_COPY_LINEAR_TAG_DST_ADDR_HI_UNION._pack_ = 1 # source:False
union_SDMA_PKT_COPY_LINEAR_TAG_DST_ADDR_HI_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_COPY_LINEAR_TAG_DST_ADDR_HI_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_COPY_LINEAR_TAG_6_0),
    ('DW_6_DATA', ctypes.c_uint32),
]

struct_SDMA_PKT_COPY_LINEAR_TAG._pack_ = 1 # source:False
struct_SDMA_PKT_COPY_LINEAR_TAG._fields_ = [
    ('HEADER_UNION', union_SDMA_PKT_COPY_LINEAR_TAG_HEADER_UNION),
    ('COUNT_UNION', union_SDMA_PKT_COPY_LINEAR_TAG_COUNT_UNION),
    ('PARAMETER_UNION', union_SDMA_PKT_COPY_LINEAR_TAG_PARAMETER_UNION),
    ('SRC_ADDR_LO_UNION', union_SDMA_PKT_COPY_LINEAR_TAG_SRC_ADDR_LO_UNION),
    ('SRC_ADDR_HI_UNION', union_SDMA_PKT_COPY_LINEAR_TAG_SRC_ADDR_HI_UNION),
    ('DST_ADDR_LO_UNION', union_SDMA_PKT_COPY_LINEAR_TAG_DST_ADDR_LO_UNION),
    ('DST_ADDR_HI_UNION', union_SDMA_PKT_COPY_LINEAR_TAG_DST_ADDR_HI_UNION),
]

SDMA_PKT_COPY_LINEAR = struct_SDMA_PKT_COPY_LINEAR_TAG
class struct_SDMA_PKT_COPY_LINEAR_RECT_TAG(Structure):
    pass

class union_SDMA_PKT_COPY_LINEAR_RECT_TAG_HEADER_UNION(Union):
    pass

class struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_0_0(Structure):
    pass

struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_0_0._pack_ = 1 # source:False
struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_0_0._fields_ = [
    ('op', ctypes.c_uint32, 8),
    ('sub_op', ctypes.c_uint32, 8),
    ('reserved', ctypes.c_uint32, 13),
    ('element', ctypes.c_uint32, 3),
]

union_SDMA_PKT_COPY_LINEAR_RECT_TAG_HEADER_UNION._pack_ = 1 # source:False
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_HEADER_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_HEADER_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_0_0),
    ('DW_0_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_ADDR_LO_UNION(Union):
    pass

class struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_1_0(Structure):
    pass

struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_1_0._pack_ = 1 # source:False
struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_1_0._fields_ = [
    ('src_addr_31_0', ctypes.c_uint32, 32),
]

union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_ADDR_LO_UNION._pack_ = 1 # source:False
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_ADDR_LO_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_ADDR_LO_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_1_0),
    ('DW_1_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_ADDR_HI_UNION(Union):
    pass

class struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_2_0(Structure):
    pass

struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_2_0._pack_ = 1 # source:False
struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_2_0._fields_ = [
    ('src_addr_63_32', ctypes.c_uint32, 32),
]

union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_ADDR_HI_UNION._pack_ = 1 # source:False
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_ADDR_HI_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_ADDR_HI_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_2_0),
    ('DW_2_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_1_UNION(Union):
    pass

class struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_3_0(Structure):
    pass

struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_3_0._pack_ = 1 # source:False
struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_3_0._fields_ = [
    ('src_offset_x', ctypes.c_uint32, 14),
    ('reserved_1', ctypes.c_uint32, 2),
    ('src_offset_y', ctypes.c_uint32, 14),
    ('reserved_2', ctypes.c_uint32, 2),
]

union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_1_UNION._pack_ = 1 # source:False
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_1_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_1_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_3_0),
    ('DW_3_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_2_UNION(Union):
    pass

class struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_4_0(Structure):
    pass

struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_4_0._pack_ = 1 # source:False
struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_4_0._fields_ = [
    ('src_offset_z', ctypes.c_uint32, 11),
    ('reserved_1', ctypes.c_uint32, 2),
    ('src_pitch', ctypes.c_uint32, 19),
]

union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_2_UNION._pack_ = 1 # source:False
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_2_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_2_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_4_0),
    ('DW_4_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_3_UNION(Union):
    pass

class struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_5_0(Structure):
    pass

struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_5_0._pack_ = 1 # source:False
struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_5_0._fields_ = [
    ('src_slice_pitch', ctypes.c_uint32, 28),
    ('reserved_1', ctypes.c_uint32, 4),
]

union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_3_UNION._pack_ = 1 # source:False
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_3_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_3_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_5_0),
    ('DW_5_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_ADDR_LO_UNION(Union):
    pass

class struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_6_0(Structure):
    pass

struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_6_0._pack_ = 1 # source:False
struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_6_0._fields_ = [
    ('dst_addr_31_0', ctypes.c_uint32, 32),
]

union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_ADDR_LO_UNION._pack_ = 1 # source:False
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_ADDR_LO_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_ADDR_LO_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_6_0),
    ('DW_6_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_ADDR_HI_UNION(Union):
    pass

class struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_7_0(Structure):
    pass

struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_7_0._pack_ = 1 # source:False
struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_7_0._fields_ = [
    ('dst_addr_63_32', ctypes.c_uint32, 32),
]

union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_ADDR_HI_UNION._pack_ = 1 # source:False
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_ADDR_HI_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_ADDR_HI_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_7_0),
    ('DW_7_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_1_UNION(Union):
    pass

class struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_8_0(Structure):
    pass

struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_8_0._pack_ = 1 # source:False
struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_8_0._fields_ = [
    ('dst_offset_x', ctypes.c_uint32, 14),
    ('reserved_1', ctypes.c_uint32, 2),
    ('dst_offset_y', ctypes.c_uint32, 14),
    ('reserved_2', ctypes.c_uint32, 2),
]

union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_1_UNION._pack_ = 1 # source:False
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_1_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_1_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_8_0),
    ('DW_8_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_2_UNION(Union):
    pass

class struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_9_0(Structure):
    pass

struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_9_0._pack_ = 1 # source:False
struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_9_0._fields_ = [
    ('dst_offset_z', ctypes.c_uint32, 11),
    ('reserved_1', ctypes.c_uint32, 2),
    ('dst_pitch', ctypes.c_uint32, 19),
]

union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_2_UNION._pack_ = 1 # source:False
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_2_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_2_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_9_0),
    ('DW_9_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_3_UNION(Union):
    pass

class struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_10_0(Structure):
    pass

struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_10_0._pack_ = 1 # source:False
struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_10_0._fields_ = [
    ('dst_slice_pitch', ctypes.c_uint32, 28),
    ('reserved_1', ctypes.c_uint32, 4),
]

union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_3_UNION._pack_ = 1 # source:False
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_3_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_3_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_10_0),
    ('DW_10_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_COPY_LINEAR_RECT_TAG_RECT_PARAMETER_1_UNION(Union):
    pass

class struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_11_0(Structure):
    pass

struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_11_0._pack_ = 1 # source:False
struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_11_0._fields_ = [
    ('rect_x', ctypes.c_uint32, 14),
    ('reserved_1', ctypes.c_uint32, 2),
    ('rect_y', ctypes.c_uint32, 14),
    ('reserved_2', ctypes.c_uint32, 2),
]

union_SDMA_PKT_COPY_LINEAR_RECT_TAG_RECT_PARAMETER_1_UNION._pack_ = 1 # source:False
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_RECT_PARAMETER_1_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_RECT_PARAMETER_1_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_11_0),
    ('DW_11_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_COPY_LINEAR_RECT_TAG_RECT_PARAMETER_2_UNION(Union):
    pass

class struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_12_0(Structure):
    pass

struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_12_0._pack_ = 1 # source:False
struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_12_0._fields_ = [
    ('rect_z', ctypes.c_uint32, 11),
    ('reserved_1', ctypes.c_uint32, 5),
    ('dst_swap', ctypes.c_uint32, 2),
    ('reserved_2', ctypes.c_uint32, 6),
    ('src_swap', ctypes.c_uint32, 2),
    ('reserved_3', ctypes.c_uint32, 6),
]

union_SDMA_PKT_COPY_LINEAR_RECT_TAG_RECT_PARAMETER_2_UNION._pack_ = 1 # source:False
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_RECT_PARAMETER_2_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_COPY_LINEAR_RECT_TAG_RECT_PARAMETER_2_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_12_0),
    ('DW_12_DATA', ctypes.c_uint32),
]

struct_SDMA_PKT_COPY_LINEAR_RECT_TAG._pack_ = 1 # source:False
struct_SDMA_PKT_COPY_LINEAR_RECT_TAG._fields_ = [
    ('HEADER_UNION', union_SDMA_PKT_COPY_LINEAR_RECT_TAG_HEADER_UNION),
    ('SRC_ADDR_LO_UNION', union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_ADDR_LO_UNION),
    ('SRC_ADDR_HI_UNION', union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_ADDR_HI_UNION),
    ('SRC_PARAMETER_1_UNION', union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_1_UNION),
    ('SRC_PARAMETER_2_UNION', union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_2_UNION),
    ('SRC_PARAMETER_3_UNION', union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_3_UNION),
    ('DST_ADDR_LO_UNION', union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_ADDR_LO_UNION),
    ('DST_ADDR_HI_UNION', union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_ADDR_HI_UNION),
    ('DST_PARAMETER_1_UNION', union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_1_UNION),
    ('DST_PARAMETER_2_UNION', union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_2_UNION),
    ('DST_PARAMETER_3_UNION', union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_3_UNION),
    ('RECT_PARAMETER_1_UNION', union_SDMA_PKT_COPY_LINEAR_RECT_TAG_RECT_PARAMETER_1_UNION),
    ('RECT_PARAMETER_2_UNION', union_SDMA_PKT_COPY_LINEAR_RECT_TAG_RECT_PARAMETER_2_UNION),
]

SDMA_PKT_COPY_LINEAR_RECT = struct_SDMA_PKT_COPY_LINEAR_RECT_TAG
class struct_SDMA_PKT_CONSTANT_FILL_TAG(Structure):
    pass

class union_SDMA_PKT_CONSTANT_FILL_TAG_HEADER_UNION(Union):
    pass

class struct_SDMA_PKT_CONSTANT_FILL_TAG_0_0(Structure):
    pass

struct_SDMA_PKT_CONSTANT_FILL_TAG_0_0._pack_ = 1 # source:False
struct_SDMA_PKT_CONSTANT_FILL_TAG_0_0._fields_ = [
    ('op', ctypes.c_uint32, 8),
    ('sub_op', ctypes.c_uint32, 8),
    ('sw', ctypes.c_uint32, 2),
    ('reserved_0', ctypes.c_uint32, 12),
    ('fillsize', ctypes.c_uint32, 2),
]

union_SDMA_PKT_CONSTANT_FILL_TAG_HEADER_UNION._pack_ = 1 # source:False
union_SDMA_PKT_CONSTANT_FILL_TAG_HEADER_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_CONSTANT_FILL_TAG_HEADER_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_CONSTANT_FILL_TAG_0_0),
    ('DW_0_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_CONSTANT_FILL_TAG_DST_ADDR_LO_UNION(Union):
    pass

class struct_SDMA_PKT_CONSTANT_FILL_TAG_1_0(Structure):
    pass

struct_SDMA_PKT_CONSTANT_FILL_TAG_1_0._pack_ = 1 # source:False
struct_SDMA_PKT_CONSTANT_FILL_TAG_1_0._fields_ = [
    ('dst_addr_31_0', ctypes.c_uint32, 32),
]

union_SDMA_PKT_CONSTANT_FILL_TAG_DST_ADDR_LO_UNION._pack_ = 1 # source:False
union_SDMA_PKT_CONSTANT_FILL_TAG_DST_ADDR_LO_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_CONSTANT_FILL_TAG_DST_ADDR_LO_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_CONSTANT_FILL_TAG_1_0),
    ('DW_1_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_CONSTANT_FILL_TAG_DST_ADDR_HI_UNION(Union):
    pass

class struct_SDMA_PKT_CONSTANT_FILL_TAG_2_0(Structure):
    pass

struct_SDMA_PKT_CONSTANT_FILL_TAG_2_0._pack_ = 1 # source:False
struct_SDMA_PKT_CONSTANT_FILL_TAG_2_0._fields_ = [
    ('dst_addr_63_32', ctypes.c_uint32, 32),
]

union_SDMA_PKT_CONSTANT_FILL_TAG_DST_ADDR_HI_UNION._pack_ = 1 # source:False
union_SDMA_PKT_CONSTANT_FILL_TAG_DST_ADDR_HI_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_CONSTANT_FILL_TAG_DST_ADDR_HI_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_CONSTANT_FILL_TAG_2_0),
    ('DW_2_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_CONSTANT_FILL_TAG_DATA_UNION(Union):
    pass

class struct_SDMA_PKT_CONSTANT_FILL_TAG_3_0(Structure):
    pass

struct_SDMA_PKT_CONSTANT_FILL_TAG_3_0._pack_ = 1 # source:False
struct_SDMA_PKT_CONSTANT_FILL_TAG_3_0._fields_ = [
    ('src_data_31_0', ctypes.c_uint32, 32),
]

union_SDMA_PKT_CONSTANT_FILL_TAG_DATA_UNION._pack_ = 1 # source:False
union_SDMA_PKT_CONSTANT_FILL_TAG_DATA_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_CONSTANT_FILL_TAG_DATA_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_CONSTANT_FILL_TAG_3_0),
    ('DW_3_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_CONSTANT_FILL_TAG_COUNT_UNION(Union):
    pass

class struct_SDMA_PKT_CONSTANT_FILL_TAG_4_0(Structure):
    pass

struct_SDMA_PKT_CONSTANT_FILL_TAG_4_0._pack_ = 1 # source:False
struct_SDMA_PKT_CONSTANT_FILL_TAG_4_0._fields_ = [
    ('count', ctypes.c_uint32, 22),
    ('reserved_0', ctypes.c_uint32, 10),
]

union_SDMA_PKT_CONSTANT_FILL_TAG_COUNT_UNION._pack_ = 1 # source:False
union_SDMA_PKT_CONSTANT_FILL_TAG_COUNT_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_CONSTANT_FILL_TAG_COUNT_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_CONSTANT_FILL_TAG_4_0),
    ('DW_4_DATA', ctypes.c_uint32),
]

struct_SDMA_PKT_CONSTANT_FILL_TAG._pack_ = 1 # source:False
struct_SDMA_PKT_CONSTANT_FILL_TAG._fields_ = [
    ('HEADER_UNION', union_SDMA_PKT_CONSTANT_FILL_TAG_HEADER_UNION),
    ('DST_ADDR_LO_UNION', union_SDMA_PKT_CONSTANT_FILL_TAG_DST_ADDR_LO_UNION),
    ('DST_ADDR_HI_UNION', union_SDMA_PKT_CONSTANT_FILL_TAG_DST_ADDR_HI_UNION),
    ('DATA_UNION', union_SDMA_PKT_CONSTANT_FILL_TAG_DATA_UNION),
    ('COUNT_UNION', union_SDMA_PKT_CONSTANT_FILL_TAG_COUNT_UNION),
]

SDMA_PKT_CONSTANT_FILL = struct_SDMA_PKT_CONSTANT_FILL_TAG
class struct_SDMA_PKT_FENCE_TAG(Structure):
    pass

class union_SDMA_PKT_FENCE_TAG_HEADER_UNION(Union):
    pass

class struct_SDMA_PKT_FENCE_TAG_0_0(Structure):
    pass

struct_SDMA_PKT_FENCE_TAG_0_0._pack_ = 1 # source:False
struct_SDMA_PKT_FENCE_TAG_0_0._fields_ = [
    ('op', ctypes.c_uint32, 8),
    ('sub_op', ctypes.c_uint32, 8),
    ('mtype', ctypes.c_uint32, 3),
    ('gcc', ctypes.c_uint32, 1),
    ('sys', ctypes.c_uint32, 1),
    ('pad1', ctypes.c_uint32, 1),
    ('snp', ctypes.c_uint32, 1),
    ('gpa', ctypes.c_uint32, 1),
    ('l2_policy', ctypes.c_uint32, 2),
    ('reserved_0', ctypes.c_uint32, 6),
]

union_SDMA_PKT_FENCE_TAG_HEADER_UNION._pack_ = 1 # source:False
union_SDMA_PKT_FENCE_TAG_HEADER_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_FENCE_TAG_HEADER_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_FENCE_TAG_0_0),
    ('DW_0_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_FENCE_TAG_ADDR_LO_UNION(Union):
    pass

class struct_SDMA_PKT_FENCE_TAG_1_0(Structure):
    pass

struct_SDMA_PKT_FENCE_TAG_1_0._pack_ = 1 # source:False
struct_SDMA_PKT_FENCE_TAG_1_0._fields_ = [
    ('addr_31_0', ctypes.c_uint32, 32),
]

union_SDMA_PKT_FENCE_TAG_ADDR_LO_UNION._pack_ = 1 # source:False
union_SDMA_PKT_FENCE_TAG_ADDR_LO_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_FENCE_TAG_ADDR_LO_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_FENCE_TAG_1_0),
    ('DW_1_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_FENCE_TAG_ADDR_HI_UNION(Union):
    pass

class struct_SDMA_PKT_FENCE_TAG_2_0(Structure):
    pass

struct_SDMA_PKT_FENCE_TAG_2_0._pack_ = 1 # source:False
struct_SDMA_PKT_FENCE_TAG_2_0._fields_ = [
    ('addr_63_32', ctypes.c_uint32, 32),
]

union_SDMA_PKT_FENCE_TAG_ADDR_HI_UNION._pack_ = 1 # source:False
union_SDMA_PKT_FENCE_TAG_ADDR_HI_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_FENCE_TAG_ADDR_HI_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_FENCE_TAG_2_0),
    ('DW_2_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_FENCE_TAG_DATA_UNION(Union):
    pass

class struct_SDMA_PKT_FENCE_TAG_3_0(Structure):
    pass

struct_SDMA_PKT_FENCE_TAG_3_0._pack_ = 1 # source:False
struct_SDMA_PKT_FENCE_TAG_3_0._fields_ = [
    ('data', ctypes.c_uint32, 32),
]

union_SDMA_PKT_FENCE_TAG_DATA_UNION._pack_ = 1 # source:False
union_SDMA_PKT_FENCE_TAG_DATA_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_FENCE_TAG_DATA_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_FENCE_TAG_3_0),
    ('DW_3_DATA', ctypes.c_uint32),
]

struct_SDMA_PKT_FENCE_TAG._pack_ = 1 # source:False
struct_SDMA_PKT_FENCE_TAG._fields_ = [
    ('HEADER_UNION', union_SDMA_PKT_FENCE_TAG_HEADER_UNION),
    ('ADDR_LO_UNION', union_SDMA_PKT_FENCE_TAG_ADDR_LO_UNION),
    ('ADDR_HI_UNION', union_SDMA_PKT_FENCE_TAG_ADDR_HI_UNION),
    ('DATA_UNION', union_SDMA_PKT_FENCE_TAG_DATA_UNION),
]

SDMA_PKT_FENCE = struct_SDMA_PKT_FENCE_TAG
class struct_SDMA_PKT_POLL_REGMEM_TAG(Structure):
    pass

class union_SDMA_PKT_POLL_REGMEM_TAG_HEADER_UNION(Union):
    pass

class struct_SDMA_PKT_POLL_REGMEM_TAG_0_0(Structure):
    pass

struct_SDMA_PKT_POLL_REGMEM_TAG_0_0._pack_ = 1 # source:False
struct_SDMA_PKT_POLL_REGMEM_TAG_0_0._fields_ = [
    ('op', ctypes.c_uint32, 8),
    ('sub_op', ctypes.c_uint32, 8),
    ('reserved_0', ctypes.c_uint32, 10),
    ('hdp_flush', ctypes.c_uint32, 1),
    ('reserved_1', ctypes.c_uint32, 1),
    ('func', ctypes.c_uint32, 3),
    ('mem_poll', ctypes.c_uint32, 1),
]

union_SDMA_PKT_POLL_REGMEM_TAG_HEADER_UNION._pack_ = 1 # source:False
union_SDMA_PKT_POLL_REGMEM_TAG_HEADER_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_POLL_REGMEM_TAG_HEADER_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_POLL_REGMEM_TAG_0_0),
    ('DW_0_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_POLL_REGMEM_TAG_ADDR_LO_UNION(Union):
    pass

class struct_SDMA_PKT_POLL_REGMEM_TAG_1_0(Structure):
    pass

struct_SDMA_PKT_POLL_REGMEM_TAG_1_0._pack_ = 1 # source:False
struct_SDMA_PKT_POLL_REGMEM_TAG_1_0._fields_ = [
    ('addr_31_0', ctypes.c_uint32, 32),
]

union_SDMA_PKT_POLL_REGMEM_TAG_ADDR_LO_UNION._pack_ = 1 # source:False
union_SDMA_PKT_POLL_REGMEM_TAG_ADDR_LO_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_POLL_REGMEM_TAG_ADDR_LO_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_POLL_REGMEM_TAG_1_0),
    ('DW_1_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_POLL_REGMEM_TAG_ADDR_HI_UNION(Union):
    pass

class struct_SDMA_PKT_POLL_REGMEM_TAG_2_0(Structure):
    pass

struct_SDMA_PKT_POLL_REGMEM_TAG_2_0._pack_ = 1 # source:False
struct_SDMA_PKT_POLL_REGMEM_TAG_2_0._fields_ = [
    ('addr_63_32', ctypes.c_uint32, 32),
]

union_SDMA_PKT_POLL_REGMEM_TAG_ADDR_HI_UNION._pack_ = 1 # source:False
union_SDMA_PKT_POLL_REGMEM_TAG_ADDR_HI_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_POLL_REGMEM_TAG_ADDR_HI_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_POLL_REGMEM_TAG_2_0),
    ('DW_2_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_POLL_REGMEM_TAG_VALUE_UNION(Union):
    pass

class struct_SDMA_PKT_POLL_REGMEM_TAG_3_0(Structure):
    pass

struct_SDMA_PKT_POLL_REGMEM_TAG_3_0._pack_ = 1 # source:False
struct_SDMA_PKT_POLL_REGMEM_TAG_3_0._fields_ = [
    ('value', ctypes.c_uint32, 32),
]

union_SDMA_PKT_POLL_REGMEM_TAG_VALUE_UNION._pack_ = 1 # source:False
union_SDMA_PKT_POLL_REGMEM_TAG_VALUE_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_POLL_REGMEM_TAG_VALUE_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_POLL_REGMEM_TAG_3_0),
    ('DW_3_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_POLL_REGMEM_TAG_MASK_UNION(Union):
    pass

class struct_SDMA_PKT_POLL_REGMEM_TAG_4_0(Structure):
    pass

struct_SDMA_PKT_POLL_REGMEM_TAG_4_0._pack_ = 1 # source:False
struct_SDMA_PKT_POLL_REGMEM_TAG_4_0._fields_ = [
    ('mask', ctypes.c_uint32, 32),
]

union_SDMA_PKT_POLL_REGMEM_TAG_MASK_UNION._pack_ = 1 # source:False
union_SDMA_PKT_POLL_REGMEM_TAG_MASK_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_POLL_REGMEM_TAG_MASK_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_POLL_REGMEM_TAG_4_0),
    ('DW_4_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_POLL_REGMEM_TAG_DW5_UNION(Union):
    pass

class struct_SDMA_PKT_POLL_REGMEM_TAG_5_0(Structure):
    pass

struct_SDMA_PKT_POLL_REGMEM_TAG_5_0._pack_ = 1 # source:False
struct_SDMA_PKT_POLL_REGMEM_TAG_5_0._fields_ = [
    ('interval', ctypes.c_uint32, 16),
    ('retry_count', ctypes.c_uint32, 12),
    ('reserved_0', ctypes.c_uint32, 4),
]

union_SDMA_PKT_POLL_REGMEM_TAG_DW5_UNION._pack_ = 1 # source:False
union_SDMA_PKT_POLL_REGMEM_TAG_DW5_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_POLL_REGMEM_TAG_DW5_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_POLL_REGMEM_TAG_5_0),
    ('DW_5_DATA', ctypes.c_uint32),
]

struct_SDMA_PKT_POLL_REGMEM_TAG._pack_ = 1 # source:False
struct_SDMA_PKT_POLL_REGMEM_TAG._fields_ = [
    ('HEADER_UNION', union_SDMA_PKT_POLL_REGMEM_TAG_HEADER_UNION),
    ('ADDR_LO_UNION', union_SDMA_PKT_POLL_REGMEM_TAG_ADDR_LO_UNION),
    ('ADDR_HI_UNION', union_SDMA_PKT_POLL_REGMEM_TAG_ADDR_HI_UNION),
    ('VALUE_UNION', union_SDMA_PKT_POLL_REGMEM_TAG_VALUE_UNION),
    ('MASK_UNION', union_SDMA_PKT_POLL_REGMEM_TAG_MASK_UNION),
    ('DW5_UNION', union_SDMA_PKT_POLL_REGMEM_TAG_DW5_UNION),
]

SDMA_PKT_POLL_REGMEM = struct_SDMA_PKT_POLL_REGMEM_TAG
class struct_SDMA_PKT_ATOMIC_TAG(Structure):
    pass

class union_SDMA_PKT_ATOMIC_TAG_HEADER_UNION(Union):
    pass

class struct_SDMA_PKT_ATOMIC_TAG_0_0(Structure):
    pass

struct_SDMA_PKT_ATOMIC_TAG_0_0._pack_ = 1 # source:False
struct_SDMA_PKT_ATOMIC_TAG_0_0._fields_ = [
    ('op', ctypes.c_uint32, 8),
    ('sub_op', ctypes.c_uint32, 8),
    ('l', ctypes.c_uint32, 1),
    ('reserved_0', ctypes.c_uint32, 8),
    ('operation', ctypes.c_uint32, 7),
]

union_SDMA_PKT_ATOMIC_TAG_HEADER_UNION._pack_ = 1 # source:False
union_SDMA_PKT_ATOMIC_TAG_HEADER_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_ATOMIC_TAG_HEADER_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_ATOMIC_TAG_0_0),
    ('DW_0_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_ATOMIC_TAG_ADDR_LO_UNION(Union):
    pass

class struct_SDMA_PKT_ATOMIC_TAG_1_0(Structure):
    pass

struct_SDMA_PKT_ATOMIC_TAG_1_0._pack_ = 1 # source:False
struct_SDMA_PKT_ATOMIC_TAG_1_0._fields_ = [
    ('addr_31_0', ctypes.c_uint32, 32),
]

union_SDMA_PKT_ATOMIC_TAG_ADDR_LO_UNION._pack_ = 1 # source:False
union_SDMA_PKT_ATOMIC_TAG_ADDR_LO_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_ATOMIC_TAG_ADDR_LO_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_ATOMIC_TAG_1_0),
    ('DW_1_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_ATOMIC_TAG_ADDR_HI_UNION(Union):
    pass

class struct_SDMA_PKT_ATOMIC_TAG_2_0(Structure):
    pass

struct_SDMA_PKT_ATOMIC_TAG_2_0._pack_ = 1 # source:False
struct_SDMA_PKT_ATOMIC_TAG_2_0._fields_ = [
    ('addr_63_32', ctypes.c_uint32, 32),
]

union_SDMA_PKT_ATOMIC_TAG_ADDR_HI_UNION._pack_ = 1 # source:False
union_SDMA_PKT_ATOMIC_TAG_ADDR_HI_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_ATOMIC_TAG_ADDR_HI_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_ATOMIC_TAG_2_0),
    ('DW_2_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_ATOMIC_TAG_SRC_DATA_LO_UNION(Union):
    pass

class struct_SDMA_PKT_ATOMIC_TAG_3_0(Structure):
    pass

struct_SDMA_PKT_ATOMIC_TAG_3_0._pack_ = 1 # source:False
struct_SDMA_PKT_ATOMIC_TAG_3_0._fields_ = [
    ('src_data_31_0', ctypes.c_uint32, 32),
]

union_SDMA_PKT_ATOMIC_TAG_SRC_DATA_LO_UNION._pack_ = 1 # source:False
union_SDMA_PKT_ATOMIC_TAG_SRC_DATA_LO_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_ATOMIC_TAG_SRC_DATA_LO_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_ATOMIC_TAG_3_0),
    ('DW_3_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_ATOMIC_TAG_SRC_DATA_HI_UNION(Union):
    pass

class struct_SDMA_PKT_ATOMIC_TAG_4_0(Structure):
    pass

struct_SDMA_PKT_ATOMIC_TAG_4_0._pack_ = 1 # source:False
struct_SDMA_PKT_ATOMIC_TAG_4_0._fields_ = [
    ('src_data_63_32', ctypes.c_uint32, 32),
]

union_SDMA_PKT_ATOMIC_TAG_SRC_DATA_HI_UNION._pack_ = 1 # source:False
union_SDMA_PKT_ATOMIC_TAG_SRC_DATA_HI_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_ATOMIC_TAG_SRC_DATA_HI_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_ATOMIC_TAG_4_0),
    ('DW_4_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_ATOMIC_TAG_CMP_DATA_LO_UNION(Union):
    pass

class struct_SDMA_PKT_ATOMIC_TAG_5_0(Structure):
    pass

struct_SDMA_PKT_ATOMIC_TAG_5_0._pack_ = 1 # source:False
struct_SDMA_PKT_ATOMIC_TAG_5_0._fields_ = [
    ('cmp_data_31_0', ctypes.c_uint32, 32),
]

union_SDMA_PKT_ATOMIC_TAG_CMP_DATA_LO_UNION._pack_ = 1 # source:False
union_SDMA_PKT_ATOMIC_TAG_CMP_DATA_LO_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_ATOMIC_TAG_CMP_DATA_LO_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_ATOMIC_TAG_5_0),
    ('DW_5_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_ATOMIC_TAG_CMP_DATA_HI_UNION(Union):
    pass

class struct_SDMA_PKT_ATOMIC_TAG_6_0(Structure):
    pass

struct_SDMA_PKT_ATOMIC_TAG_6_0._pack_ = 1 # source:False
struct_SDMA_PKT_ATOMIC_TAG_6_0._fields_ = [
    ('cmp_data_63_32', ctypes.c_uint32, 32),
]

union_SDMA_PKT_ATOMIC_TAG_CMP_DATA_HI_UNION._pack_ = 1 # source:False
union_SDMA_PKT_ATOMIC_TAG_CMP_DATA_HI_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_ATOMIC_TAG_CMP_DATA_HI_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_ATOMIC_TAG_6_0),
    ('DW_6_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_ATOMIC_TAG_LOOP_UNION(Union):
    pass

class struct_SDMA_PKT_ATOMIC_TAG_7_0(Structure):
    pass

struct_SDMA_PKT_ATOMIC_TAG_7_0._pack_ = 1 # source:False
struct_SDMA_PKT_ATOMIC_TAG_7_0._fields_ = [
    ('loop_interval', ctypes.c_uint32, 13),
    ('reserved_0', ctypes.c_uint32, 19),
]

union_SDMA_PKT_ATOMIC_TAG_LOOP_UNION._pack_ = 1 # source:False
union_SDMA_PKT_ATOMIC_TAG_LOOP_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_ATOMIC_TAG_LOOP_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_ATOMIC_TAG_7_0),
    ('DW_7_DATA', ctypes.c_uint32),
]

struct_SDMA_PKT_ATOMIC_TAG._pack_ = 1 # source:False
struct_SDMA_PKT_ATOMIC_TAG._fields_ = [
    ('HEADER_UNION', union_SDMA_PKT_ATOMIC_TAG_HEADER_UNION),
    ('ADDR_LO_UNION', union_SDMA_PKT_ATOMIC_TAG_ADDR_LO_UNION),
    ('ADDR_HI_UNION', union_SDMA_PKT_ATOMIC_TAG_ADDR_HI_UNION),
    ('SRC_DATA_LO_UNION', union_SDMA_PKT_ATOMIC_TAG_SRC_DATA_LO_UNION),
    ('SRC_DATA_HI_UNION', union_SDMA_PKT_ATOMIC_TAG_SRC_DATA_HI_UNION),
    ('CMP_DATA_LO_UNION', union_SDMA_PKT_ATOMIC_TAG_CMP_DATA_LO_UNION),
    ('CMP_DATA_HI_UNION', union_SDMA_PKT_ATOMIC_TAG_CMP_DATA_HI_UNION),
    ('LOOP_UNION', union_SDMA_PKT_ATOMIC_TAG_LOOP_UNION),
]

SDMA_PKT_ATOMIC = struct_SDMA_PKT_ATOMIC_TAG
class struct_SDMA_PKT_TIMESTAMP_TAG(Structure):
    pass

class union_SDMA_PKT_TIMESTAMP_TAG_HEADER_UNION(Union):
    pass

class struct_SDMA_PKT_TIMESTAMP_TAG_0_0(Structure):
    pass

struct_SDMA_PKT_TIMESTAMP_TAG_0_0._pack_ = 1 # source:False
struct_SDMA_PKT_TIMESTAMP_TAG_0_0._fields_ = [
    ('op', ctypes.c_uint32, 8),
    ('sub_op', ctypes.c_uint32, 8),
    ('reserved_0', ctypes.c_uint32, 16),
]

union_SDMA_PKT_TIMESTAMP_TAG_HEADER_UNION._pack_ = 1 # source:False
union_SDMA_PKT_TIMESTAMP_TAG_HEADER_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_TIMESTAMP_TAG_HEADER_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_TIMESTAMP_TAG_0_0),
    ('DW_0_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_TIMESTAMP_TAG_ADDR_LO_UNION(Union):
    pass

class struct_SDMA_PKT_TIMESTAMP_TAG_1_0(Structure):
    pass

struct_SDMA_PKT_TIMESTAMP_TAG_1_0._pack_ = 1 # source:False
struct_SDMA_PKT_TIMESTAMP_TAG_1_0._fields_ = [
    ('addr_31_0', ctypes.c_uint32, 32),
]

union_SDMA_PKT_TIMESTAMP_TAG_ADDR_LO_UNION._pack_ = 1 # source:False
union_SDMA_PKT_TIMESTAMP_TAG_ADDR_LO_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_TIMESTAMP_TAG_ADDR_LO_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_TIMESTAMP_TAG_1_0),
    ('DW_1_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_TIMESTAMP_TAG_ADDR_HI_UNION(Union):
    pass

class struct_SDMA_PKT_TIMESTAMP_TAG_2_0(Structure):
    pass

struct_SDMA_PKT_TIMESTAMP_TAG_2_0._pack_ = 1 # source:False
struct_SDMA_PKT_TIMESTAMP_TAG_2_0._fields_ = [
    ('addr_63_32', ctypes.c_uint32, 32),
]

union_SDMA_PKT_TIMESTAMP_TAG_ADDR_HI_UNION._pack_ = 1 # source:False
union_SDMA_PKT_TIMESTAMP_TAG_ADDR_HI_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_TIMESTAMP_TAG_ADDR_HI_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_TIMESTAMP_TAG_2_0),
    ('DW_2_DATA', ctypes.c_uint32),
]

struct_SDMA_PKT_TIMESTAMP_TAG._pack_ = 1 # source:False
struct_SDMA_PKT_TIMESTAMP_TAG._fields_ = [
    ('HEADER_UNION', union_SDMA_PKT_TIMESTAMP_TAG_HEADER_UNION),
    ('ADDR_LO_UNION', union_SDMA_PKT_TIMESTAMP_TAG_ADDR_LO_UNION),
    ('ADDR_HI_UNION', union_SDMA_PKT_TIMESTAMP_TAG_ADDR_HI_UNION),
]

SDMA_PKT_TIMESTAMP = struct_SDMA_PKT_TIMESTAMP_TAG
class struct_SDMA_PKT_TRAP_TAG(Structure):
    pass

class union_SDMA_PKT_TRAP_TAG_HEADER_UNION(Union):
    pass

class struct_SDMA_PKT_TRAP_TAG_0_0(Structure):
    pass

struct_SDMA_PKT_TRAP_TAG_0_0._pack_ = 1 # source:False
struct_SDMA_PKT_TRAP_TAG_0_0._fields_ = [
    ('op', ctypes.c_uint32, 8),
    ('sub_op', ctypes.c_uint32, 8),
    ('reserved_0', ctypes.c_uint32, 16),
]

union_SDMA_PKT_TRAP_TAG_HEADER_UNION._pack_ = 1 # source:False
union_SDMA_PKT_TRAP_TAG_HEADER_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_TRAP_TAG_HEADER_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_TRAP_TAG_0_0),
    ('DW_0_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_TRAP_TAG_INT_CONTEXT_UNION(Union):
    pass

class struct_SDMA_PKT_TRAP_TAG_1_0(Structure):
    pass

struct_SDMA_PKT_TRAP_TAG_1_0._pack_ = 1 # source:False
struct_SDMA_PKT_TRAP_TAG_1_0._fields_ = [
    ('int_ctx', ctypes.c_uint32, 28),
    ('reserved_1', ctypes.c_uint32, 4),
]

union_SDMA_PKT_TRAP_TAG_INT_CONTEXT_UNION._pack_ = 1 # source:False
union_SDMA_PKT_TRAP_TAG_INT_CONTEXT_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_TRAP_TAG_INT_CONTEXT_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_TRAP_TAG_1_0),
    ('DW_1_DATA', ctypes.c_uint32),
]

struct_SDMA_PKT_TRAP_TAG._pack_ = 1 # source:False
struct_SDMA_PKT_TRAP_TAG._fields_ = [
    ('HEADER_UNION', union_SDMA_PKT_TRAP_TAG_HEADER_UNION),
    ('INT_CONTEXT_UNION', union_SDMA_PKT_TRAP_TAG_INT_CONTEXT_UNION),
]

SDMA_PKT_TRAP = struct_SDMA_PKT_TRAP_TAG
class struct_SDMA_PKT_HDP_FLUSH_TAG(Structure):
    pass

struct_SDMA_PKT_HDP_FLUSH_TAG._pack_ = 1 # source:False
struct_SDMA_PKT_HDP_FLUSH_TAG._fields_ = [
    ('DW_0_DATA', ctypes.c_uint32),
    ('DW_1_DATA', ctypes.c_uint32),
    ('DW_2_DATA', ctypes.c_uint32),
    ('DW_3_DATA', ctypes.c_uint32),
    ('DW_4_DATA', ctypes.c_uint32),
    ('DW_5_DATA', ctypes.c_uint32),
]

SDMA_PKT_HDP_FLUSH = struct_SDMA_PKT_HDP_FLUSH_TAG
hdp_flush_cmd = struct_SDMA_PKT_HDP_FLUSH_TAG # Variable struct_SDMA_PKT_HDP_FLUSH_TAG
class struct_SDMA_PKT_GCR_TAG(Structure):
    pass

class union_SDMA_PKT_GCR_TAG_HEADER_UNION(Union):
    pass

class struct_SDMA_PKT_GCR_TAG_0_0(Structure):
    pass

struct_SDMA_PKT_GCR_TAG_0_0._pack_ = 1 # source:False
struct_SDMA_PKT_GCR_TAG_0_0._fields_ = [
    ('op', ctypes.c_uint32, 8),
    ('sub_op', ctypes.c_uint32, 8),
    ('_2', ctypes.c_uint32, 16),
]

union_SDMA_PKT_GCR_TAG_HEADER_UNION._pack_ = 1 # source:False
union_SDMA_PKT_GCR_TAG_HEADER_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_GCR_TAG_HEADER_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_GCR_TAG_0_0),
    ('DW_0_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_GCR_TAG_WORD1_UNION(Union):
    pass

class struct_SDMA_PKT_GCR_TAG_1_0(Structure):
    pass

struct_SDMA_PKT_GCR_TAG_1_0._pack_ = 1 # source:False
struct_SDMA_PKT_GCR_TAG_1_0._fields_ = [
    ('_0', ctypes.c_uint32, 7),
    ('BaseVA_LO', ctypes.c_uint32, 25),
]

union_SDMA_PKT_GCR_TAG_WORD1_UNION._pack_ = 1 # source:False
union_SDMA_PKT_GCR_TAG_WORD1_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_GCR_TAG_WORD1_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_GCR_TAG_1_0),
    ('DW_1_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_GCR_TAG_WORD2_UNION(Union):
    pass

class struct_SDMA_PKT_GCR_TAG_2_0(Structure):
    pass

struct_SDMA_PKT_GCR_TAG_2_0._pack_ = 1 # source:False
struct_SDMA_PKT_GCR_TAG_2_0._fields_ = [
    ('BaseVA_HI', ctypes.c_uint32, 16),
    ('GCR_CONTROL_GLI_INV', ctypes.c_uint32, 2),
    ('GCR_CONTROL_GL1_RANGE', ctypes.c_uint32, 2),
    ('GCR_CONTROL_GLM_WB', ctypes.c_uint32, 1),
    ('GCR_CONTROL_GLM_INV', ctypes.c_uint32, 1),
    ('GCR_CONTROL_GLK_WB', ctypes.c_uint32, 1),
    ('GCR_CONTROL_GLK_INV', ctypes.c_uint32, 1),
    ('GCR_CONTROL_GLV_INV', ctypes.c_uint32, 1),
    ('GCR_CONTROL_GL1_INV', ctypes.c_uint32, 1),
    ('GCR_CONTROL_GL2_US', ctypes.c_uint32, 1),
    ('GCR_CONTROL_GL2_RANGE', ctypes.c_uint32, 2),
    ('GCR_CONTROL_GL2_DISCARD', ctypes.c_uint32, 1),
    ('GCR_CONTROL_GL2_INV', ctypes.c_uint32, 1),
    ('GCR_CONTROL_GL2_WB', ctypes.c_uint32, 1),
]

union_SDMA_PKT_GCR_TAG_WORD2_UNION._pack_ = 1 # source:False
union_SDMA_PKT_GCR_TAG_WORD2_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_GCR_TAG_WORD2_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_GCR_TAG_2_0),
    ('DW_2_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_GCR_TAG_WORD3_UNION(Union):
    pass

class struct_SDMA_PKT_GCR_TAG_3_0(Structure):
    pass

struct_SDMA_PKT_GCR_TAG_3_0._pack_ = 1 # source:False
struct_SDMA_PKT_GCR_TAG_3_0._fields_ = [
    ('GCR_CONTROL_RANGE_IS_PA', ctypes.c_uint32, 1),
    ('GCR_CONTROL_SEQ', ctypes.c_uint32, 2),
    ('_2', ctypes.c_uint32, 4),
    ('LimitVA_LO', ctypes.c_uint32, 25),
]

union_SDMA_PKT_GCR_TAG_WORD3_UNION._pack_ = 1 # source:False
union_SDMA_PKT_GCR_TAG_WORD3_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_GCR_TAG_WORD3_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_GCR_TAG_3_0),
    ('DW_3_DATA', ctypes.c_uint32),
]

class union_SDMA_PKT_GCR_TAG_WORD4_UNION(Union):
    pass

class struct_SDMA_PKT_GCR_TAG_4_0(Structure):
    pass

struct_SDMA_PKT_GCR_TAG_4_0._pack_ = 1 # source:False
struct_SDMA_PKT_GCR_TAG_4_0._fields_ = [
    ('LimitVA_HI', ctypes.c_uint32, 16),
    ('_1', ctypes.c_uint32, 8),
    ('VMID', ctypes.c_uint32, 4),
    ('_3', ctypes.c_uint32, 4),
]

union_SDMA_PKT_GCR_TAG_WORD4_UNION._pack_ = 1 # source:False
union_SDMA_PKT_GCR_TAG_WORD4_UNION._anonymous_ = ('_0',)
union_SDMA_PKT_GCR_TAG_WORD4_UNION._fields_ = [
    ('_0', struct_SDMA_PKT_GCR_TAG_4_0),
    ('DW_4_DATA', ctypes.c_uint32),
]

struct_SDMA_PKT_GCR_TAG._pack_ = 1 # source:False
struct_SDMA_PKT_GCR_TAG._fields_ = [
    ('HEADER_UNION', union_SDMA_PKT_GCR_TAG_HEADER_UNION),
    ('WORD1_UNION', union_SDMA_PKT_GCR_TAG_WORD1_UNION),
    ('WORD2_UNION', union_SDMA_PKT_GCR_TAG_WORD2_UNION),
    ('WORD3_UNION', union_SDMA_PKT_GCR_TAG_WORD3_UNION),
    ('WORD4_UNION', union_SDMA_PKT_GCR_TAG_WORD4_UNION),
]

SDMA_PKT_GCR = struct_SDMA_PKT_GCR_TAG
__NAVI10_SDMA_PKT_OPEN_H_ = True # macro
SDMA_OP_NOP = 0 # macro
SDMA_OP_WRITE = 2 # macro
SDMA_OP_INDIRECT = 4 # macro
SDMA_OP_SEM = 7 # macro
SDMA_OP_COND_EXE = 9 # macro
SDMA_OP_PTEPDE = 12 # macro
SDMA_OP_SRBM_WRITE = 14 # macro
SDMA_OP_PRE_EXE = 15 # macro
SDMA_OP_GPUVM_INV = 16 # macro
SDMA_OP_GCR_REQ = 17 # macro
SDMA_OP_DUMMY_TRAP = 32 # macro
SDMA_SUBOP_TIMESTAMP_SET = 0 # macro
SDMA_SUBOP_TIMESTAMP_GET = 1 # macro
SDMA_SUBOP_COPY_LINEAR_SUB_WIND = 4 # macro
SDMA_SUBOP_COPY_TILED = 1 # macro
SDMA_SUBOP_COPY_TILED_SUB_WIND = 5 # macro
SDMA_SUBOP_COPY_T2T_SUB_WIND = 6 # macro
SDMA_SUBOP_COPY_SOA = 3 # macro
SDMA_SUBOP_COPY_DIRTY_PAGE = 7 # macro
SDMA_SUBOP_COPY_LINEAR_PHY = 8 # macro
SDMA_SUBOP_COPY_LINEAR_BC = 16 # macro
SDMA_SUBOP_COPY_TILED_BC = 17 # macro
SDMA_SUBOP_COPY_LINEAR_SUB_WIND_BC = 20 # macro
SDMA_SUBOP_COPY_TILED_SUB_WIND_BC = 21 # macro
SDMA_SUBOP_COPY_T2T_SUB_WIND_BC = 22 # macro
SDMA_SUBOP_WRITE_LINEAR = 0 # macro
SDMA_SUBOP_WRITE_TILED = 1 # macro
SDMA_SUBOP_WRITE_TILED_BC = 17 # macro
SDMA_SUBOP_PTEPDE_GEN = 0 # macro
SDMA_SUBOP_PTEPDE_COPY = 1 # macro
SDMA_SUBOP_PTEPDE_RMW = 2 # macro
SDMA_SUBOP_PTEPDE_COPY_BACKWARDS = 3 # macro
SDMA_SUBOP_DATA_FILL_MULTI = 1 # macro
SDMA_SUBOP_POLL_REG_WRITE_MEM = 1 # macro
SDMA_SUBOP_POLL_DBIT_WRITE_MEM = 2 # macro
SDMA_SUBOP_POLL_MEM_VERIFY = 3 # macro
SDMA_SUBOP_VM_INVALIDATION = 4 # macro
HEADER_AGENT_DISPATCH = 4 # macro
HEADER_BARRIER = 5 # macro
SDMA_OP_AQL_COPY = 0 # macro
SDMA_OP_AQL_BARRIER_OR = 0 # macro
SDMA_GCR_RANGE_IS_PA = (1<<18) # macro
def SDMA_GCR_SEQ(x):  # macro
   return (((x)&0x3)<<16)
SDMA_GCR_GL2_WB = (1<<15) # macro
SDMA_GCR_GL2_INV = (1<<14) # macro
SDMA_GCR_GL2_DISCARD = (1<<13) # macro
def SDMA_GCR_GL2_RANGE(x):  # macro
   return (((x)&0x3)<<11)
SDMA_GCR_GL2_US = (1<<10) # macro
SDMA_GCR_GL1_INV = (1<<9) # macro
SDMA_GCR_GLV_INV = (1<<8) # macro
SDMA_GCR_GLK_INV = (1<<7) # macro
SDMA_GCR_GLK_WB = (1<<6) # macro
SDMA_GCR_GLM_INV = (1<<5) # macro
SDMA_GCR_GLM_WB = (1<<4) # macro
def SDMA_GCR_GL1_RANGE(x):  # macro
   return (((x)&0x3)<<2)
def SDMA_GCR_GLI_INV(x):  # macro
   return (((x)&0x3)<<0)
SDMA_PKT_HEADER_op_offset = 0 # macro
SDMA_PKT_HEADER_op_mask = 0x000000FF # macro
SDMA_PKT_HEADER_op_shift = 0 # macro
def SDMA_PKT_HEADER_OP(x):  # macro
   return (((x)&0x000000FF)<<0)
SDMA_PKT_HEADER_sub_op_offset = 0 # macro
SDMA_PKT_HEADER_sub_op_mask = 0x000000FF # macro
SDMA_PKT_HEADER_sub_op_shift = 8 # macro
def SDMA_PKT_HEADER_SUB_OP(x):  # macro
   return (((x)&0x000000FF)<<8)
SDMA_PKT_COPY_LINEAR_HEADER_op_offset = 0 # macro
SDMA_PKT_COPY_LINEAR_HEADER_op_mask = 0x000000FF # macro
SDMA_PKT_COPY_LINEAR_HEADER_op_shift = 0 # macro
def SDMA_PKT_COPY_LINEAR_HEADER_OP(x):  # macro
   return (((x)&0x000000FF)<<0)
SDMA_PKT_COPY_LINEAR_HEADER_sub_op_offset = 0 # macro
SDMA_PKT_COPY_LINEAR_HEADER_sub_op_mask = 0x000000FF # macro
SDMA_PKT_COPY_LINEAR_HEADER_sub_op_shift = 8 # macro
def SDMA_PKT_COPY_LINEAR_HEADER_SUB_OP(x):  # macro
   return (((x)&0x000000FF)<<8)
SDMA_PKT_COPY_LINEAR_HEADER_encrypt_offset = 0 # macro
SDMA_PKT_COPY_LINEAR_HEADER_encrypt_mask = 0x00000001 # macro
SDMA_PKT_COPY_LINEAR_HEADER_encrypt_shift = 16 # macro
def SDMA_PKT_COPY_LINEAR_HEADER_ENCRYPT(x):  # macro
   return (((x)&0x00000001)<<16)
SDMA_PKT_COPY_LINEAR_HEADER_tmz_offset = 0 # macro
SDMA_PKT_COPY_LINEAR_HEADER_tmz_mask = 0x00000001 # macro
SDMA_PKT_COPY_LINEAR_HEADER_tmz_shift = 18 # macro
def SDMA_PKT_COPY_LINEAR_HEADER_TMZ(x):  # macro
   return (((x)&0x00000001)<<18)
SDMA_PKT_COPY_LINEAR_HEADER_backwards_offset = 0 # macro
SDMA_PKT_COPY_LINEAR_HEADER_backwards_mask = 0x00000001 # macro
SDMA_PKT_COPY_LINEAR_HEADER_backwards_shift = 25 # macro
def SDMA_PKT_COPY_LINEAR_HEADER_BACKWARDS(x):  # macro
   return (((x)&0x00000001)<<25)
SDMA_PKT_COPY_LINEAR_HEADER_broadcast_offset = 0 # macro
SDMA_PKT_COPY_LINEAR_HEADER_broadcast_mask = 0x00000001 # macro
SDMA_PKT_COPY_LINEAR_HEADER_broadcast_shift = 27 # macro
def SDMA_PKT_COPY_LINEAR_HEADER_BROADCAST(x):  # macro
   return (((x)&0x00000001)<<27)
SDMA_PKT_COPY_LINEAR_COUNT_count_offset = 1 # macro
SDMA_PKT_COPY_LINEAR_COUNT_count_mask = 0x003FFFFF # macro
SDMA_PKT_COPY_LINEAR_COUNT_count_shift = 0 # macro
def SDMA_PKT_COPY_LINEAR_COUNT_COUNT(x):  # macro
   return (((x)&0x003FFFFF)<<0)
SDMA_PKT_COPY_LINEAR_PARAMETER_dst_sw_offset = 2 # macro
SDMA_PKT_COPY_LINEAR_PARAMETER_dst_sw_mask = 0x00000003 # macro
SDMA_PKT_COPY_LINEAR_PARAMETER_dst_sw_shift = 16 # macro
def SDMA_PKT_COPY_LINEAR_PARAMETER_DST_SW(x):  # macro
   return (((x)&0x00000003)<<16)
SDMA_PKT_COPY_LINEAR_PARAMETER_src_sw_offset = 2 # macro
SDMA_PKT_COPY_LINEAR_PARAMETER_src_sw_mask = 0x00000003 # macro
SDMA_PKT_COPY_LINEAR_PARAMETER_src_sw_shift = 24 # macro
def SDMA_PKT_COPY_LINEAR_PARAMETER_SRC_SW(x):  # macro
   return (((x)&0x00000003)<<24)
SDMA_PKT_COPY_LINEAR_SRC_ADDR_LO_src_addr_31_0_offset = 3 # macro
SDMA_PKT_COPY_LINEAR_SRC_ADDR_LO_src_addr_31_0_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_LINEAR_SRC_ADDR_LO_src_addr_31_0_shift = 0 # macro
def SDMA_PKT_COPY_LINEAR_SRC_ADDR_LO_SRC_ADDR_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COPY_LINEAR_SRC_ADDR_HI_src_addr_63_32_offset = 4 # macro
SDMA_PKT_COPY_LINEAR_SRC_ADDR_HI_src_addr_63_32_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_LINEAR_SRC_ADDR_HI_src_addr_63_32_shift = 0 # macro
def SDMA_PKT_COPY_LINEAR_SRC_ADDR_HI_SRC_ADDR_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COPY_LINEAR_DST_ADDR_LO_dst_addr_31_0_offset = 5 # macro
SDMA_PKT_COPY_LINEAR_DST_ADDR_LO_dst_addr_31_0_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_LINEAR_DST_ADDR_LO_dst_addr_31_0_shift = 0 # macro
def SDMA_PKT_COPY_LINEAR_DST_ADDR_LO_DST_ADDR_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COPY_LINEAR_DST_ADDR_HI_dst_addr_63_32_offset = 6 # macro
SDMA_PKT_COPY_LINEAR_DST_ADDR_HI_dst_addr_63_32_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_LINEAR_DST_ADDR_HI_dst_addr_63_32_shift = 0 # macro
def SDMA_PKT_COPY_LINEAR_DST_ADDR_HI_DST_ADDR_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COPY_LINEAR_BC_HEADER_op_offset = 0 # macro
SDMA_PKT_COPY_LINEAR_BC_HEADER_op_mask = 0x000000FF # macro
SDMA_PKT_COPY_LINEAR_BC_HEADER_op_shift = 0 # macro
def SDMA_PKT_COPY_LINEAR_BC_HEADER_OP(x):  # macro
   return (((x)&0x000000FF)<<0)
SDMA_PKT_COPY_LINEAR_BC_HEADER_sub_op_offset = 0 # macro
SDMA_PKT_COPY_LINEAR_BC_HEADER_sub_op_mask = 0x000000FF # macro
SDMA_PKT_COPY_LINEAR_BC_HEADER_sub_op_shift = 8 # macro
def SDMA_PKT_COPY_LINEAR_BC_HEADER_SUB_OP(x):  # macro
   return (((x)&0x000000FF)<<8)
SDMA_PKT_COPY_LINEAR_BC_COUNT_count_offset = 1 # macro
SDMA_PKT_COPY_LINEAR_BC_COUNT_count_mask = 0x003FFFFF # macro
SDMA_PKT_COPY_LINEAR_BC_COUNT_count_shift = 0 # macro
def SDMA_PKT_COPY_LINEAR_BC_COUNT_COUNT(x):  # macro
   return (((x)&0x003FFFFF)<<0)
SDMA_PKT_COPY_LINEAR_BC_PARAMETER_dst_sw_offset = 2 # macro
SDMA_PKT_COPY_LINEAR_BC_PARAMETER_dst_sw_mask = 0x00000003 # macro
SDMA_PKT_COPY_LINEAR_BC_PARAMETER_dst_sw_shift = 16 # macro
def SDMA_PKT_COPY_LINEAR_BC_PARAMETER_DST_SW(x):  # macro
   return (((x)&0x00000003)<<16)
SDMA_PKT_COPY_LINEAR_BC_PARAMETER_dst_ha_offset = 2 # macro
SDMA_PKT_COPY_LINEAR_BC_PARAMETER_dst_ha_mask = 0x00000001 # macro
SDMA_PKT_COPY_LINEAR_BC_PARAMETER_dst_ha_shift = 22 # macro
def SDMA_PKT_COPY_LINEAR_BC_PARAMETER_DST_HA(x):  # macro
   return (((x)&0x00000001)<<22)
SDMA_PKT_COPY_LINEAR_BC_PARAMETER_src_sw_offset = 2 # macro
SDMA_PKT_COPY_LINEAR_BC_PARAMETER_src_sw_mask = 0x00000003 # macro
SDMA_PKT_COPY_LINEAR_BC_PARAMETER_src_sw_shift = 24 # macro
def SDMA_PKT_COPY_LINEAR_BC_PARAMETER_SRC_SW(x):  # macro
   return (((x)&0x00000003)<<24)
SDMA_PKT_COPY_LINEAR_BC_PARAMETER_src_ha_offset = 2 # macro
SDMA_PKT_COPY_LINEAR_BC_PARAMETER_src_ha_mask = 0x00000001 # macro
SDMA_PKT_COPY_LINEAR_BC_PARAMETER_src_ha_shift = 30 # macro
def SDMA_PKT_COPY_LINEAR_BC_PARAMETER_SRC_HA(x):  # macro
   return (((x)&0x00000001)<<30)
SDMA_PKT_COPY_LINEAR_BC_SRC_ADDR_LO_src_addr_31_0_offset = 3 # macro
SDMA_PKT_COPY_LINEAR_BC_SRC_ADDR_LO_src_addr_31_0_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_LINEAR_BC_SRC_ADDR_LO_src_addr_31_0_shift = 0 # macro
def SDMA_PKT_COPY_LINEAR_BC_SRC_ADDR_LO_SRC_ADDR_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COPY_LINEAR_BC_SRC_ADDR_HI_src_addr_63_32_offset = 4 # macro
SDMA_PKT_COPY_LINEAR_BC_SRC_ADDR_HI_src_addr_63_32_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_LINEAR_BC_SRC_ADDR_HI_src_addr_63_32_shift = 0 # macro
def SDMA_PKT_COPY_LINEAR_BC_SRC_ADDR_HI_SRC_ADDR_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COPY_LINEAR_BC_DST_ADDR_LO_dst_addr_31_0_offset = 5 # macro
SDMA_PKT_COPY_LINEAR_BC_DST_ADDR_LO_dst_addr_31_0_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_LINEAR_BC_DST_ADDR_LO_dst_addr_31_0_shift = 0 # macro
def SDMA_PKT_COPY_LINEAR_BC_DST_ADDR_LO_DST_ADDR_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COPY_LINEAR_BC_DST_ADDR_HI_dst_addr_63_32_offset = 6 # macro
SDMA_PKT_COPY_LINEAR_BC_DST_ADDR_HI_dst_addr_63_32_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_LINEAR_BC_DST_ADDR_HI_dst_addr_63_32_shift = 0 # macro
def SDMA_PKT_COPY_LINEAR_BC_DST_ADDR_HI_DST_ADDR_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COPY_DIRTY_PAGE_HEADER_op_offset = 0 # macro
SDMA_PKT_COPY_DIRTY_PAGE_HEADER_op_mask = 0x000000FF # macro
SDMA_PKT_COPY_DIRTY_PAGE_HEADER_op_shift = 0 # macro
def SDMA_PKT_COPY_DIRTY_PAGE_HEADER_OP(x):  # macro
   return (((x)&0x000000FF)<<0)
SDMA_PKT_COPY_DIRTY_PAGE_HEADER_sub_op_offset = 0 # macro
SDMA_PKT_COPY_DIRTY_PAGE_HEADER_sub_op_mask = 0x000000FF # macro
SDMA_PKT_COPY_DIRTY_PAGE_HEADER_sub_op_shift = 8 # macro
def SDMA_PKT_COPY_DIRTY_PAGE_HEADER_SUB_OP(x):  # macro
   return (((x)&0x000000FF)<<8)
SDMA_PKT_COPY_DIRTY_PAGE_HEADER_tmz_offset = 0 # macro
SDMA_PKT_COPY_DIRTY_PAGE_HEADER_tmz_mask = 0x00000001 # macro
SDMA_PKT_COPY_DIRTY_PAGE_HEADER_tmz_shift = 18 # macro
def SDMA_PKT_COPY_DIRTY_PAGE_HEADER_TMZ(x):  # macro
   return (((x)&0x00000001)<<18)
SDMA_PKT_COPY_DIRTY_PAGE_HEADER_all_offset = 0 # macro
SDMA_PKT_COPY_DIRTY_PAGE_HEADER_all_mask = 0x00000001 # macro
SDMA_PKT_COPY_DIRTY_PAGE_HEADER_all_shift = 31 # macro
def SDMA_PKT_COPY_DIRTY_PAGE_HEADER_ALL(x):  # macro
   return (((x)&0x00000001)<<31)
SDMA_PKT_COPY_DIRTY_PAGE_COUNT_count_offset = 1 # macro
SDMA_PKT_COPY_DIRTY_PAGE_COUNT_count_mask = 0x003FFFFF # macro
SDMA_PKT_COPY_DIRTY_PAGE_COUNT_count_shift = 0 # macro
def SDMA_PKT_COPY_DIRTY_PAGE_COUNT_COUNT(x):  # macro
   return (((x)&0x003FFFFF)<<0)
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_mtype_offset = 2 # macro
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_mtype_mask = 0x00000007 # macro
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_mtype_shift = 3 # macro
def SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_DST_MTYPE(x):  # macro
   return (((x)&0x00000007)<<3)
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_l2_policy_offset = 2 # macro
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_l2_policy_mask = 0x00000003 # macro
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_l2_policy_shift = 6 # macro
def SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_DST_L2_POLICY(x):  # macro
   return (((x)&0x00000003)<<6)
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_mtype_offset = 2 # macro
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_mtype_mask = 0x00000007 # macro
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_mtype_shift = 11 # macro
def SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_SRC_MTYPE(x):  # macro
   return (((x)&0x00000007)<<11)
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_l2_policy_offset = 2 # macro
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_l2_policy_mask = 0x00000003 # macro
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_l2_policy_shift = 14 # macro
def SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_SRC_L2_POLICY(x):  # macro
   return (((x)&0x00000003)<<14)
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_sw_offset = 2 # macro
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_sw_mask = 0x00000003 # macro
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_sw_shift = 16 # macro
def SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_DST_SW(x):  # macro
   return (((x)&0x00000003)<<16)
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_gcc_offset = 2 # macro
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_gcc_mask = 0x00000001 # macro
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_gcc_shift = 19 # macro
def SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_DST_GCC(x):  # macro
   return (((x)&0x00000001)<<19)
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_sys_offset = 2 # macro
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_sys_mask = 0x00000001 # macro
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_sys_shift = 20 # macro
def SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_DST_SYS(x):  # macro
   return (((x)&0x00000001)<<20)
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_snoop_offset = 2 # macro
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_snoop_mask = 0x00000001 # macro
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_snoop_shift = 22 # macro
def SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_DST_SNOOP(x):  # macro
   return (((x)&0x00000001)<<22)
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_gpa_offset = 2 # macro
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_gpa_mask = 0x00000001 # macro
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_gpa_shift = 23 # macro
def SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_DST_GPA(x):  # macro
   return (((x)&0x00000001)<<23)
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_sw_offset = 2 # macro
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_sw_mask = 0x00000003 # macro
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_sw_shift = 24 # macro
def SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_SRC_SW(x):  # macro
   return (((x)&0x00000003)<<24)
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_sys_offset = 2 # macro
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_sys_mask = 0x00000001 # macro
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_sys_shift = 28 # macro
def SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_SRC_SYS(x):  # macro
   return (((x)&0x00000001)<<28)
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_snoop_offset = 2 # macro
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_snoop_mask = 0x00000001 # macro
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_snoop_shift = 30 # macro
def SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_SRC_SNOOP(x):  # macro
   return (((x)&0x00000001)<<30)
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_gpa_offset = 2 # macro
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_gpa_mask = 0x00000001 # macro
SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_gpa_shift = 31 # macro
def SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_SRC_GPA(x):  # macro
   return (((x)&0x00000001)<<31)
SDMA_PKT_COPY_DIRTY_PAGE_SRC_ADDR_LO_src_addr_31_0_offset = 3 # macro
SDMA_PKT_COPY_DIRTY_PAGE_SRC_ADDR_LO_src_addr_31_0_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_DIRTY_PAGE_SRC_ADDR_LO_src_addr_31_0_shift = 0 # macro
def SDMA_PKT_COPY_DIRTY_PAGE_SRC_ADDR_LO_SRC_ADDR_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COPY_DIRTY_PAGE_SRC_ADDR_HI_src_addr_63_32_offset = 4 # macro
SDMA_PKT_COPY_DIRTY_PAGE_SRC_ADDR_HI_src_addr_63_32_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_DIRTY_PAGE_SRC_ADDR_HI_src_addr_63_32_shift = 0 # macro
def SDMA_PKT_COPY_DIRTY_PAGE_SRC_ADDR_HI_SRC_ADDR_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COPY_DIRTY_PAGE_DST_ADDR_LO_dst_addr_31_0_offset = 5 # macro
SDMA_PKT_COPY_DIRTY_PAGE_DST_ADDR_LO_dst_addr_31_0_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_DIRTY_PAGE_DST_ADDR_LO_dst_addr_31_0_shift = 0 # macro
def SDMA_PKT_COPY_DIRTY_PAGE_DST_ADDR_LO_DST_ADDR_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COPY_DIRTY_PAGE_DST_ADDR_HI_dst_addr_63_32_offset = 6 # macro
SDMA_PKT_COPY_DIRTY_PAGE_DST_ADDR_HI_dst_addr_63_32_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_DIRTY_PAGE_DST_ADDR_HI_dst_addr_63_32_shift = 0 # macro
def SDMA_PKT_COPY_DIRTY_PAGE_DST_ADDR_HI_DST_ADDR_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COPY_PHYSICAL_LINEAR_HEADER_op_offset = 0 # macro
SDMA_PKT_COPY_PHYSICAL_LINEAR_HEADER_op_mask = 0x000000FF # macro
SDMA_PKT_COPY_PHYSICAL_LINEAR_HEADER_op_shift = 0 # macro
def SDMA_PKT_COPY_PHYSICAL_LINEAR_HEADER_OP(x):  # macro
   return (((x)&0x000000FF)<<0)
SDMA_PKT_COPY_PHYSICAL_LINEAR_HEADER_sub_op_offset = 0 # macro
SDMA_PKT_COPY_PHYSICAL_LINEAR_HEADER_sub_op_mask = 0x000000FF # macro
SDMA_PKT_COPY_PHYSICAL_LINEAR_HEADER_sub_op_shift = 8 # macro
def SDMA_PKT_COPY_PHYSICAL_LINEAR_HEADER_SUB_OP(x):  # macro
   return (((x)&0x000000FF)<<8)
SDMA_PKT_COPY_PHYSICAL_LINEAR_HEADER_tmz_offset = 0 # macro
SDMA_PKT_COPY_PHYSICAL_LINEAR_HEADER_tmz_mask = 0x00000001 # macro
SDMA_PKT_COPY_PHYSICAL_LINEAR_HEADER_tmz_shift = 18 # macro
def SDMA_PKT_COPY_PHYSICAL_LINEAR_HEADER_TMZ(x):  # macro
   return (((x)&0x00000001)<<18)
SDMA_PKT_COPY_PHYSICAL_LINEAR_COUNT_count_offset = 1 # macro
SDMA_PKT_COPY_PHYSICAL_LINEAR_COUNT_count_mask = 0x003FFFFF # macro
SDMA_PKT_COPY_PHYSICAL_LINEAR_COUNT_count_shift = 0 # macro
def SDMA_PKT_COPY_PHYSICAL_LINEAR_COUNT_COUNT(x):  # macro
   return (((x)&0x003FFFFF)<<0)
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_mtype_offset = 2 # macro
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_mtype_mask = 0x00000007 # macro
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_mtype_shift = 3 # macro
def SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_DST_MTYPE(x):  # macro
   return (((x)&0x00000007)<<3)
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_l2_policy_offset = 2 # macro
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_l2_policy_mask = 0x00000003 # macro
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_l2_policy_shift = 6 # macro
def SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_DST_L2_POLICY(x):  # macro
   return (((x)&0x00000003)<<6)
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_mtype_offset = 2 # macro
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_mtype_mask = 0x00000007 # macro
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_mtype_shift = 11 # macro
def SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_SRC_MTYPE(x):  # macro
   return (((x)&0x00000007)<<11)
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_l2_policy_offset = 2 # macro
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_l2_policy_mask = 0x00000003 # macro
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_l2_policy_shift = 14 # macro
def SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_SRC_L2_POLICY(x):  # macro
   return (((x)&0x00000003)<<14)
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_sw_offset = 2 # macro
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_sw_mask = 0x00000003 # macro
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_sw_shift = 16 # macro
def SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_DST_SW(x):  # macro
   return (((x)&0x00000003)<<16)
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_gcc_offset = 2 # macro
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_gcc_mask = 0x00000001 # macro
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_gcc_shift = 19 # macro
def SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_DST_GCC(x):  # macro
   return (((x)&0x00000001)<<19)
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_sys_offset = 2 # macro
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_sys_mask = 0x00000001 # macro
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_sys_shift = 20 # macro
def SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_DST_SYS(x):  # macro
   return (((x)&0x00000001)<<20)
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_log_offset = 2 # macro
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_log_mask = 0x00000001 # macro
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_log_shift = 21 # macro
def SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_DST_LOG(x):  # macro
   return (((x)&0x00000001)<<21)
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_snoop_offset = 2 # macro
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_snoop_mask = 0x00000001 # macro
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_snoop_shift = 22 # macro
def SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_DST_SNOOP(x):  # macro
   return (((x)&0x00000001)<<22)
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_gpa_offset = 2 # macro
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_gpa_mask = 0x00000001 # macro
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_gpa_shift = 23 # macro
def SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_DST_GPA(x):  # macro
   return (((x)&0x00000001)<<23)
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_sw_offset = 2 # macro
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_sw_mask = 0x00000003 # macro
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_sw_shift = 24 # macro
def SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_SRC_SW(x):  # macro
   return (((x)&0x00000003)<<24)
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_gcc_offset = 2 # macro
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_gcc_mask = 0x00000001 # macro
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_gcc_shift = 27 # macro
def SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_SRC_GCC(x):  # macro
   return (((x)&0x00000001)<<27)
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_sys_offset = 2 # macro
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_sys_mask = 0x00000001 # macro
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_sys_shift = 28 # macro
def SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_SRC_SYS(x):  # macro
   return (((x)&0x00000001)<<28)
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_snoop_offset = 2 # macro
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_snoop_mask = 0x00000001 # macro
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_snoop_shift = 30 # macro
def SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_SRC_SNOOP(x):  # macro
   return (((x)&0x00000001)<<30)
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_gpa_offset = 2 # macro
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_gpa_mask = 0x00000001 # macro
SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_gpa_shift = 31 # macro
def SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_SRC_GPA(x):  # macro
   return (((x)&0x00000001)<<31)
SDMA_PKT_COPY_PHYSICAL_LINEAR_SRC_ADDR_LO_src_addr_31_0_offset = 3 # macro
SDMA_PKT_COPY_PHYSICAL_LINEAR_SRC_ADDR_LO_src_addr_31_0_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_PHYSICAL_LINEAR_SRC_ADDR_LO_src_addr_31_0_shift = 0 # macro
def SDMA_PKT_COPY_PHYSICAL_LINEAR_SRC_ADDR_LO_SRC_ADDR_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COPY_PHYSICAL_LINEAR_SRC_ADDR_HI_src_addr_63_32_offset = 4 # macro
SDMA_PKT_COPY_PHYSICAL_LINEAR_SRC_ADDR_HI_src_addr_63_32_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_PHYSICAL_LINEAR_SRC_ADDR_HI_src_addr_63_32_shift = 0 # macro
def SDMA_PKT_COPY_PHYSICAL_LINEAR_SRC_ADDR_HI_SRC_ADDR_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COPY_PHYSICAL_LINEAR_DST_ADDR_LO_dst_addr_31_0_offset = 5 # macro
SDMA_PKT_COPY_PHYSICAL_LINEAR_DST_ADDR_LO_dst_addr_31_0_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_PHYSICAL_LINEAR_DST_ADDR_LO_dst_addr_31_0_shift = 0 # macro
def SDMA_PKT_COPY_PHYSICAL_LINEAR_DST_ADDR_LO_DST_ADDR_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COPY_PHYSICAL_LINEAR_DST_ADDR_HI_dst_addr_63_32_offset = 6 # macro
SDMA_PKT_COPY_PHYSICAL_LINEAR_DST_ADDR_HI_dst_addr_63_32_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_PHYSICAL_LINEAR_DST_ADDR_HI_dst_addr_63_32_shift = 0 # macro
def SDMA_PKT_COPY_PHYSICAL_LINEAR_DST_ADDR_HI_DST_ADDR_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_op_offset = 0 # macro
SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_op_mask = 0x000000FF # macro
SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_op_shift = 0 # macro
def SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_OP(x):  # macro
   return (((x)&0x000000FF)<<0)
SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_sub_op_offset = 0 # macro
SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_sub_op_mask = 0x000000FF # macro
SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_sub_op_shift = 8 # macro
def SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_SUB_OP(x):  # macro
   return (((x)&0x000000FF)<<8)
SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_encrypt_offset = 0 # macro
SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_encrypt_mask = 0x00000001 # macro
SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_encrypt_shift = 16 # macro
def SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_ENCRYPT(x):  # macro
   return (((x)&0x00000001)<<16)
SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_tmz_offset = 0 # macro
SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_tmz_mask = 0x00000001 # macro
SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_tmz_shift = 18 # macro
def SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_TMZ(x):  # macro
   return (((x)&0x00000001)<<18)
SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_broadcast_offset = 0 # macro
SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_broadcast_mask = 0x00000001 # macro
SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_broadcast_shift = 27 # macro
def SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_BROADCAST(x):  # macro
   return (((x)&0x00000001)<<27)
SDMA_PKT_COPY_BROADCAST_LINEAR_COUNT_count_offset = 1 # macro
SDMA_PKT_COPY_BROADCAST_LINEAR_COUNT_count_mask = 0x003FFFFF # macro
SDMA_PKT_COPY_BROADCAST_LINEAR_COUNT_count_shift = 0 # macro
def SDMA_PKT_COPY_BROADCAST_LINEAR_COUNT_COUNT(x):  # macro
   return (((x)&0x003FFFFF)<<0)
SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_dst2_sw_offset = 2 # macro
SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_dst2_sw_mask = 0x00000003 # macro
SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_dst2_sw_shift = 8 # macro
def SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_DST2_SW(x):  # macro
   return (((x)&0x00000003)<<8)
SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_dst1_sw_offset = 2 # macro
SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_dst1_sw_mask = 0x00000003 # macro
SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_dst1_sw_shift = 16 # macro
def SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_DST1_SW(x):  # macro
   return (((x)&0x00000003)<<16)
SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_src_sw_offset = 2 # macro
SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_src_sw_mask = 0x00000003 # macro
SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_src_sw_shift = 24 # macro
def SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_SRC_SW(x):  # macro
   return (((x)&0x00000003)<<24)
SDMA_PKT_COPY_BROADCAST_LINEAR_SRC_ADDR_LO_src_addr_31_0_offset = 3 # macro
SDMA_PKT_COPY_BROADCAST_LINEAR_SRC_ADDR_LO_src_addr_31_0_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_BROADCAST_LINEAR_SRC_ADDR_LO_src_addr_31_0_shift = 0 # macro
def SDMA_PKT_COPY_BROADCAST_LINEAR_SRC_ADDR_LO_SRC_ADDR_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COPY_BROADCAST_LINEAR_SRC_ADDR_HI_src_addr_63_32_offset = 4 # macro
SDMA_PKT_COPY_BROADCAST_LINEAR_SRC_ADDR_HI_src_addr_63_32_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_BROADCAST_LINEAR_SRC_ADDR_HI_src_addr_63_32_shift = 0 # macro
def SDMA_PKT_COPY_BROADCAST_LINEAR_SRC_ADDR_HI_SRC_ADDR_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COPY_BROADCAST_LINEAR_DST1_ADDR_LO_dst1_addr_31_0_offset = 5 # macro
SDMA_PKT_COPY_BROADCAST_LINEAR_DST1_ADDR_LO_dst1_addr_31_0_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_BROADCAST_LINEAR_DST1_ADDR_LO_dst1_addr_31_0_shift = 0 # macro
def SDMA_PKT_COPY_BROADCAST_LINEAR_DST1_ADDR_LO_DST1_ADDR_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COPY_BROADCAST_LINEAR_DST1_ADDR_HI_dst1_addr_63_32_offset = 6 # macro
SDMA_PKT_COPY_BROADCAST_LINEAR_DST1_ADDR_HI_dst1_addr_63_32_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_BROADCAST_LINEAR_DST1_ADDR_HI_dst1_addr_63_32_shift = 0 # macro
def SDMA_PKT_COPY_BROADCAST_LINEAR_DST1_ADDR_HI_DST1_ADDR_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COPY_BROADCAST_LINEAR_DST2_ADDR_LO_dst2_addr_31_0_offset = 7 # macro
SDMA_PKT_COPY_BROADCAST_LINEAR_DST2_ADDR_LO_dst2_addr_31_0_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_BROADCAST_LINEAR_DST2_ADDR_LO_dst2_addr_31_0_shift = 0 # macro
def SDMA_PKT_COPY_BROADCAST_LINEAR_DST2_ADDR_LO_DST2_ADDR_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COPY_BROADCAST_LINEAR_DST2_ADDR_HI_dst2_addr_63_32_offset = 8 # macro
SDMA_PKT_COPY_BROADCAST_LINEAR_DST2_ADDR_HI_dst2_addr_63_32_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_BROADCAST_LINEAR_DST2_ADDR_HI_dst2_addr_63_32_shift = 0 # macro
def SDMA_PKT_COPY_BROADCAST_LINEAR_DST2_ADDR_HI_DST2_ADDR_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_op_offset = 0 # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_op_mask = 0x000000FF # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_op_shift = 0 # macro
def SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_OP(x):  # macro
   return (((x)&0x000000FF)<<0)
SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_sub_op_offset = 0 # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_sub_op_mask = 0x000000FF # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_sub_op_shift = 8 # macro
def SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_SUB_OP(x):  # macro
   return (((x)&0x000000FF)<<8)
SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_tmz_offset = 0 # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_tmz_mask = 0x00000001 # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_tmz_shift = 18 # macro
def SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_TMZ(x):  # macro
   return (((x)&0x00000001)<<18)
SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_elementsize_offset = 0 # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_elementsize_mask = 0x00000007 # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_elementsize_shift = 29 # macro
def SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_ELEMENTSIZE(x):  # macro
   return (((x)&0x00000007)<<29)
SDMA_PKT_COPY_LINEAR_SUBWIN_SRC_ADDR_LO_src_addr_31_0_offset = 1 # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_SRC_ADDR_LO_src_addr_31_0_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_SRC_ADDR_LO_src_addr_31_0_shift = 0 # macro
def SDMA_PKT_COPY_LINEAR_SUBWIN_SRC_ADDR_LO_SRC_ADDR_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COPY_LINEAR_SUBWIN_SRC_ADDR_HI_src_addr_63_32_offset = 2 # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_SRC_ADDR_HI_src_addr_63_32_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_SRC_ADDR_HI_src_addr_63_32_shift = 0 # macro
def SDMA_PKT_COPY_LINEAR_SUBWIN_SRC_ADDR_HI_SRC_ADDR_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_3_src_x_offset = 3 # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_3_src_x_mask = 0x00003FFF # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_3_src_x_shift = 0 # macro
def SDMA_PKT_COPY_LINEAR_SUBWIN_DW_3_SRC_X(x):  # macro
   return (((x)&0x00003FFF)<<0)
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_3_src_y_offset = 3 # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_3_src_y_mask = 0x00003FFF # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_3_src_y_shift = 16 # macro
def SDMA_PKT_COPY_LINEAR_SUBWIN_DW_3_SRC_Y(x):  # macro
   return (((x)&0x00003FFF)<<16)
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_4_src_z_offset = 4 # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_4_src_z_mask = 0x00001FFF # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_4_src_z_shift = 0 # macro
def SDMA_PKT_COPY_LINEAR_SUBWIN_DW_4_SRC_Z(x):  # macro
   return (((x)&0x00001FFF)<<0)
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_4_src_pitch_offset = 4 # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_4_src_pitch_mask = 0x0007FFFF # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_4_src_pitch_shift = 13 # macro
def SDMA_PKT_COPY_LINEAR_SUBWIN_DW_4_SRC_PITCH(x):  # macro
   return (((x)&0x0007FFFF)<<13)
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_5_src_slice_pitch_offset = 5 # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_5_src_slice_pitch_mask = 0x0FFFFFFF # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_5_src_slice_pitch_shift = 0 # macro
def SDMA_PKT_COPY_LINEAR_SUBWIN_DW_5_SRC_SLICE_PITCH(x):  # macro
   return (((x)&0x0FFFFFFF)<<0)
SDMA_PKT_COPY_LINEAR_SUBWIN_DST_ADDR_LO_dst_addr_31_0_offset = 6 # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_DST_ADDR_LO_dst_addr_31_0_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_DST_ADDR_LO_dst_addr_31_0_shift = 0 # macro
def SDMA_PKT_COPY_LINEAR_SUBWIN_DST_ADDR_LO_DST_ADDR_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COPY_LINEAR_SUBWIN_DST_ADDR_HI_dst_addr_63_32_offset = 7 # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_DST_ADDR_HI_dst_addr_63_32_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_DST_ADDR_HI_dst_addr_63_32_shift = 0 # macro
def SDMA_PKT_COPY_LINEAR_SUBWIN_DST_ADDR_HI_DST_ADDR_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_8_dst_x_offset = 8 # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_8_dst_x_mask = 0x00003FFF # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_8_dst_x_shift = 0 # macro
def SDMA_PKT_COPY_LINEAR_SUBWIN_DW_8_DST_X(x):  # macro
   return (((x)&0x00003FFF)<<0)
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_8_dst_y_offset = 8 # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_8_dst_y_mask = 0x00003FFF # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_8_dst_y_shift = 16 # macro
def SDMA_PKT_COPY_LINEAR_SUBWIN_DW_8_DST_Y(x):  # macro
   return (((x)&0x00003FFF)<<16)
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_9_dst_z_offset = 9 # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_9_dst_z_mask = 0x00001FFF # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_9_dst_z_shift = 0 # macro
def SDMA_PKT_COPY_LINEAR_SUBWIN_DW_9_DST_Z(x):  # macro
   return (((x)&0x00001FFF)<<0)
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_9_dst_pitch_offset = 9 # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_9_dst_pitch_mask = 0x0007FFFF # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_9_dst_pitch_shift = 13 # macro
def SDMA_PKT_COPY_LINEAR_SUBWIN_DW_9_DST_PITCH(x):  # macro
   return (((x)&0x0007FFFF)<<13)
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_10_dst_slice_pitch_offset = 10 # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_10_dst_slice_pitch_mask = 0x0FFFFFFF # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_10_dst_slice_pitch_shift = 0 # macro
def SDMA_PKT_COPY_LINEAR_SUBWIN_DW_10_DST_SLICE_PITCH(x):  # macro
   return (((x)&0x0FFFFFFF)<<0)
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_11_rect_x_offset = 11 # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_11_rect_x_mask = 0x00003FFF # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_11_rect_x_shift = 0 # macro
def SDMA_PKT_COPY_LINEAR_SUBWIN_DW_11_RECT_X(x):  # macro
   return (((x)&0x00003FFF)<<0)
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_11_rect_y_offset = 11 # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_11_rect_y_mask = 0x00003FFF # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_11_rect_y_shift = 16 # macro
def SDMA_PKT_COPY_LINEAR_SUBWIN_DW_11_RECT_Y(x):  # macro
   return (((x)&0x00003FFF)<<16)
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_rect_z_offset = 12 # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_rect_z_mask = 0x00001FFF # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_rect_z_shift = 0 # macro
def SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_RECT_Z(x):  # macro
   return (((x)&0x00001FFF)<<0)
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_dst_sw_offset = 12 # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_dst_sw_mask = 0x00000003 # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_dst_sw_shift = 16 # macro
def SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_DST_SW(x):  # macro
   return (((x)&0x00000003)<<16)
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_src_sw_offset = 12 # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_src_sw_mask = 0x00000003 # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_src_sw_shift = 24 # macro
def SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_SRC_SW(x):  # macro
   return (((x)&0x00000003)<<24)
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_HEADER_op_offset = 0 # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_HEADER_op_mask = 0x000000FF # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_HEADER_op_shift = 0 # macro
def SDMA_PKT_COPY_LINEAR_SUBWIN_BC_HEADER_OP(x):  # macro
   return (((x)&0x000000FF)<<0)
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_HEADER_sub_op_offset = 0 # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_HEADER_sub_op_mask = 0x000000FF # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_HEADER_sub_op_shift = 8 # macro
def SDMA_PKT_COPY_LINEAR_SUBWIN_BC_HEADER_SUB_OP(x):  # macro
   return (((x)&0x000000FF)<<8)
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_HEADER_elementsize_offset = 0 # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_HEADER_elementsize_mask = 0x00000007 # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_HEADER_elementsize_shift = 29 # macro
def SDMA_PKT_COPY_LINEAR_SUBWIN_BC_HEADER_ELEMENTSIZE(x):  # macro
   return (((x)&0x00000007)<<29)
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_SRC_ADDR_LO_src_addr_31_0_offset = 1 # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_SRC_ADDR_LO_src_addr_31_0_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_SRC_ADDR_LO_src_addr_31_0_shift = 0 # macro
def SDMA_PKT_COPY_LINEAR_SUBWIN_BC_SRC_ADDR_LO_SRC_ADDR_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_SRC_ADDR_HI_src_addr_63_32_offset = 2 # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_SRC_ADDR_HI_src_addr_63_32_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_SRC_ADDR_HI_src_addr_63_32_shift = 0 # macro
def SDMA_PKT_COPY_LINEAR_SUBWIN_BC_SRC_ADDR_HI_SRC_ADDR_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_3_src_x_offset = 3 # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_3_src_x_mask = 0x00003FFF # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_3_src_x_shift = 0 # macro
def SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_3_SRC_X(x):  # macro
   return (((x)&0x00003FFF)<<0)
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_3_src_y_offset = 3 # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_3_src_y_mask = 0x00003FFF # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_3_src_y_shift = 16 # macro
def SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_3_SRC_Y(x):  # macro
   return (((x)&0x00003FFF)<<16)
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_4_src_z_offset = 4 # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_4_src_z_mask = 0x000007FF # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_4_src_z_shift = 0 # macro
def SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_4_SRC_Z(x):  # macro
   return (((x)&0x000007FF)<<0)
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_4_src_pitch_offset = 4 # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_4_src_pitch_mask = 0x00003FFF # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_4_src_pitch_shift = 13 # macro
def SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_4_SRC_PITCH(x):  # macro
   return (((x)&0x00003FFF)<<13)
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_5_src_slice_pitch_offset = 5 # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_5_src_slice_pitch_mask = 0x0FFFFFFF # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_5_src_slice_pitch_shift = 0 # macro
def SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_5_SRC_SLICE_PITCH(x):  # macro
   return (((x)&0x0FFFFFFF)<<0)
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DST_ADDR_LO_dst_addr_31_0_offset = 6 # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DST_ADDR_LO_dst_addr_31_0_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DST_ADDR_LO_dst_addr_31_0_shift = 0 # macro
def SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DST_ADDR_LO_DST_ADDR_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DST_ADDR_HI_dst_addr_63_32_offset = 7 # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DST_ADDR_HI_dst_addr_63_32_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DST_ADDR_HI_dst_addr_63_32_shift = 0 # macro
def SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DST_ADDR_HI_DST_ADDR_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_8_dst_x_offset = 8 # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_8_dst_x_mask = 0x00003FFF # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_8_dst_x_shift = 0 # macro
def SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_8_DST_X(x):  # macro
   return (((x)&0x00003FFF)<<0)
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_8_dst_y_offset = 8 # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_8_dst_y_mask = 0x00003FFF # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_8_dst_y_shift = 16 # macro
def SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_8_DST_Y(x):  # macro
   return (((x)&0x00003FFF)<<16)
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_9_dst_z_offset = 9 # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_9_dst_z_mask = 0x000007FF # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_9_dst_z_shift = 0 # macro
def SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_9_DST_Z(x):  # macro
   return (((x)&0x000007FF)<<0)
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_9_dst_pitch_offset = 9 # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_9_dst_pitch_mask = 0x00003FFF # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_9_dst_pitch_shift = 13 # macro
def SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_9_DST_PITCH(x):  # macro
   return (((x)&0x00003FFF)<<13)
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_10_dst_slice_pitch_offset = 10 # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_10_dst_slice_pitch_mask = 0x0FFFFFFF # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_10_dst_slice_pitch_shift = 0 # macro
def SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_10_DST_SLICE_PITCH(x):  # macro
   return (((x)&0x0FFFFFFF)<<0)
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_11_rect_x_offset = 11 # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_11_rect_x_mask = 0x00003FFF # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_11_rect_x_shift = 0 # macro
def SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_11_RECT_X(x):  # macro
   return (((x)&0x00003FFF)<<0)
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_11_rect_y_offset = 11 # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_11_rect_y_mask = 0x00003FFF # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_11_rect_y_shift = 16 # macro
def SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_11_RECT_Y(x):  # macro
   return (((x)&0x00003FFF)<<16)
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_rect_z_offset = 12 # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_rect_z_mask = 0x000007FF # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_rect_z_shift = 0 # macro
def SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_RECT_Z(x):  # macro
   return (((x)&0x000007FF)<<0)
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_dst_sw_offset = 12 # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_dst_sw_mask = 0x00000003 # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_dst_sw_shift = 16 # macro
def SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_DST_SW(x):  # macro
   return (((x)&0x00000003)<<16)
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_dst_ha_offset = 12 # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_dst_ha_mask = 0x00000001 # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_dst_ha_shift = 22 # macro
def SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_DST_HA(x):  # macro
   return (((x)&0x00000001)<<22)
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_src_sw_offset = 12 # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_src_sw_mask = 0x00000003 # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_src_sw_shift = 24 # macro
def SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_SRC_SW(x):  # macro
   return (((x)&0x00000003)<<24)
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_src_ha_offset = 12 # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_src_ha_mask = 0x00000001 # macro
SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_src_ha_shift = 30 # macro
def SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_SRC_HA(x):  # macro
   return (((x)&0x00000001)<<30)
SDMA_PKT_COPY_TILED_HEADER_op_offset = 0 # macro
SDMA_PKT_COPY_TILED_HEADER_op_mask = 0x000000FF # macro
SDMA_PKT_COPY_TILED_HEADER_op_shift = 0 # macro
def SDMA_PKT_COPY_TILED_HEADER_OP(x):  # macro
   return (((x)&0x000000FF)<<0)
SDMA_PKT_COPY_TILED_HEADER_sub_op_offset = 0 # macro
SDMA_PKT_COPY_TILED_HEADER_sub_op_mask = 0x000000FF # macro
SDMA_PKT_COPY_TILED_HEADER_sub_op_shift = 8 # macro
def SDMA_PKT_COPY_TILED_HEADER_SUB_OP(x):  # macro
   return (((x)&0x000000FF)<<8)
SDMA_PKT_COPY_TILED_HEADER_encrypt_offset = 0 # macro
SDMA_PKT_COPY_TILED_HEADER_encrypt_mask = 0x00000001 # macro
SDMA_PKT_COPY_TILED_HEADER_encrypt_shift = 16 # macro
def SDMA_PKT_COPY_TILED_HEADER_ENCRYPT(x):  # macro
   return (((x)&0x00000001)<<16)
SDMA_PKT_COPY_TILED_HEADER_tmz_offset = 0 # macro
SDMA_PKT_COPY_TILED_HEADER_tmz_mask = 0x00000001 # macro
SDMA_PKT_COPY_TILED_HEADER_tmz_shift = 18 # macro
def SDMA_PKT_COPY_TILED_HEADER_TMZ(x):  # macro
   return (((x)&0x00000001)<<18)
SDMA_PKT_COPY_TILED_HEADER_detile_offset = 0 # macro
SDMA_PKT_COPY_TILED_HEADER_detile_mask = 0x00000001 # macro
SDMA_PKT_COPY_TILED_HEADER_detile_shift = 31 # macro
def SDMA_PKT_COPY_TILED_HEADER_DETILE(x):  # macro
   return (((x)&0x00000001)<<31)
SDMA_PKT_COPY_TILED_TILED_ADDR_LO_tiled_addr_31_0_offset = 1 # macro
SDMA_PKT_COPY_TILED_TILED_ADDR_LO_tiled_addr_31_0_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_TILED_TILED_ADDR_LO_tiled_addr_31_0_shift = 0 # macro
def SDMA_PKT_COPY_TILED_TILED_ADDR_LO_TILED_ADDR_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COPY_TILED_TILED_ADDR_HI_tiled_addr_63_32_offset = 2 # macro
SDMA_PKT_COPY_TILED_TILED_ADDR_HI_tiled_addr_63_32_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_TILED_TILED_ADDR_HI_tiled_addr_63_32_shift = 0 # macro
def SDMA_PKT_COPY_TILED_TILED_ADDR_HI_TILED_ADDR_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COPY_TILED_DW_3_width_offset = 3 # macro
SDMA_PKT_COPY_TILED_DW_3_width_mask = 0x00003FFF # macro
SDMA_PKT_COPY_TILED_DW_3_width_shift = 0 # macro
def SDMA_PKT_COPY_TILED_DW_3_WIDTH(x):  # macro
   return (((x)&0x00003FFF)<<0)
SDMA_PKT_COPY_TILED_DW_4_height_offset = 4 # macro
SDMA_PKT_COPY_TILED_DW_4_height_mask = 0x00003FFF # macro
SDMA_PKT_COPY_TILED_DW_4_height_shift = 0 # macro
def SDMA_PKT_COPY_TILED_DW_4_HEIGHT(x):  # macro
   return (((x)&0x00003FFF)<<0)
SDMA_PKT_COPY_TILED_DW_4_depth_offset = 4 # macro
SDMA_PKT_COPY_TILED_DW_4_depth_mask = 0x00001FFF # macro
SDMA_PKT_COPY_TILED_DW_4_depth_shift = 16 # macro
def SDMA_PKT_COPY_TILED_DW_4_DEPTH(x):  # macro
   return (((x)&0x00001FFF)<<16)
SDMA_PKT_COPY_TILED_DW_5_element_size_offset = 5 # macro
SDMA_PKT_COPY_TILED_DW_5_element_size_mask = 0x00000007 # macro
SDMA_PKT_COPY_TILED_DW_5_element_size_shift = 0 # macro
def SDMA_PKT_COPY_TILED_DW_5_ELEMENT_SIZE(x):  # macro
   return (((x)&0x00000007)<<0)
SDMA_PKT_COPY_TILED_DW_5_swizzle_mode_offset = 5 # macro
SDMA_PKT_COPY_TILED_DW_5_swizzle_mode_mask = 0x0000001F # macro
SDMA_PKT_COPY_TILED_DW_5_swizzle_mode_shift = 3 # macro
def SDMA_PKT_COPY_TILED_DW_5_SWIZZLE_MODE(x):  # macro
   return (((x)&0x0000001F)<<3)
SDMA_PKT_COPY_TILED_DW_5_dimension_offset = 5 # macro
SDMA_PKT_COPY_TILED_DW_5_dimension_mask = 0x00000003 # macro
SDMA_PKT_COPY_TILED_DW_5_dimension_shift = 9 # macro
def SDMA_PKT_COPY_TILED_DW_5_DIMENSION(x):  # macro
   return (((x)&0x00000003)<<9)
SDMA_PKT_COPY_TILED_DW_5_mip_max_offset = 5 # macro
SDMA_PKT_COPY_TILED_DW_5_mip_max_mask = 0x0000000F # macro
SDMA_PKT_COPY_TILED_DW_5_mip_max_shift = 16 # macro
def SDMA_PKT_COPY_TILED_DW_5_MIP_MAX(x):  # macro
   return (((x)&0x0000000F)<<16)
SDMA_PKT_COPY_TILED_DW_6_x_offset = 6 # macro
SDMA_PKT_COPY_TILED_DW_6_x_mask = 0x00003FFF # macro
SDMA_PKT_COPY_TILED_DW_6_x_shift = 0 # macro
def SDMA_PKT_COPY_TILED_DW_6_X(x):  # macro
   return (((x)&0x00003FFF)<<0)
SDMA_PKT_COPY_TILED_DW_6_y_offset = 6 # macro
SDMA_PKT_COPY_TILED_DW_6_y_mask = 0x00003FFF # macro
SDMA_PKT_COPY_TILED_DW_6_y_shift = 16 # macro
def SDMA_PKT_COPY_TILED_DW_6_Y(x):  # macro
   return (((x)&0x00003FFF)<<16)
SDMA_PKT_COPY_TILED_DW_7_z_offset = 7 # macro
SDMA_PKT_COPY_TILED_DW_7_z_mask = 0x00001FFF # macro
SDMA_PKT_COPY_TILED_DW_7_z_shift = 0 # macro
def SDMA_PKT_COPY_TILED_DW_7_Z(x):  # macro
   return (((x)&0x00001FFF)<<0)
SDMA_PKT_COPY_TILED_DW_7_linear_sw_offset = 7 # macro
SDMA_PKT_COPY_TILED_DW_7_linear_sw_mask = 0x00000003 # macro
SDMA_PKT_COPY_TILED_DW_7_linear_sw_shift = 16 # macro
def SDMA_PKT_COPY_TILED_DW_7_LINEAR_SW(x):  # macro
   return (((x)&0x00000003)<<16)
SDMA_PKT_COPY_TILED_DW_7_linear_cc_offset = 7 # macro
SDMA_PKT_COPY_TILED_DW_7_linear_cc_mask = 0x00000001 # macro
SDMA_PKT_COPY_TILED_DW_7_linear_cc_shift = 20 # macro
def SDMA_PKT_COPY_TILED_DW_7_LINEAR_CC(x):  # macro
   return (((x)&0x00000001)<<20)
SDMA_PKT_COPY_TILED_DW_7_tile_sw_offset = 7 # macro
SDMA_PKT_COPY_TILED_DW_7_tile_sw_mask = 0x00000003 # macro
SDMA_PKT_COPY_TILED_DW_7_tile_sw_shift = 24 # macro
def SDMA_PKT_COPY_TILED_DW_7_TILE_SW(x):  # macro
   return (((x)&0x00000003)<<24)
SDMA_PKT_COPY_TILED_LINEAR_ADDR_LO_linear_addr_31_0_offset = 8 # macro
SDMA_PKT_COPY_TILED_LINEAR_ADDR_LO_linear_addr_31_0_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_TILED_LINEAR_ADDR_LO_linear_addr_31_0_shift = 0 # macro
def SDMA_PKT_COPY_TILED_LINEAR_ADDR_LO_LINEAR_ADDR_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COPY_TILED_LINEAR_ADDR_HI_linear_addr_63_32_offset = 9 # macro
SDMA_PKT_COPY_TILED_LINEAR_ADDR_HI_linear_addr_63_32_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_TILED_LINEAR_ADDR_HI_linear_addr_63_32_shift = 0 # macro
def SDMA_PKT_COPY_TILED_LINEAR_ADDR_HI_LINEAR_ADDR_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COPY_TILED_LINEAR_PITCH_linear_pitch_offset = 10 # macro
SDMA_PKT_COPY_TILED_LINEAR_PITCH_linear_pitch_mask = 0x0007FFFF # macro
SDMA_PKT_COPY_TILED_LINEAR_PITCH_linear_pitch_shift = 0 # macro
def SDMA_PKT_COPY_TILED_LINEAR_PITCH_LINEAR_PITCH(x):  # macro
   return (((x)&0x0007FFFF)<<0)
SDMA_PKT_COPY_TILED_LINEAR_SLICE_PITCH_linear_slice_pitch_offset = 11 # macro
SDMA_PKT_COPY_TILED_LINEAR_SLICE_PITCH_linear_slice_pitch_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_TILED_LINEAR_SLICE_PITCH_linear_slice_pitch_shift = 0 # macro
def SDMA_PKT_COPY_TILED_LINEAR_SLICE_PITCH_LINEAR_SLICE_PITCH(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COPY_TILED_COUNT_count_offset = 12 # macro
SDMA_PKT_COPY_TILED_COUNT_count_mask = 0x003FFFFF # macro
SDMA_PKT_COPY_TILED_COUNT_count_shift = 0 # macro
def SDMA_PKT_COPY_TILED_COUNT_COUNT(x):  # macro
   return (((x)&0x003FFFFF)<<0)
SDMA_PKT_COPY_TILED_BC_HEADER_op_offset = 0 # macro
SDMA_PKT_COPY_TILED_BC_HEADER_op_mask = 0x000000FF # macro
SDMA_PKT_COPY_TILED_BC_HEADER_op_shift = 0 # macro
def SDMA_PKT_COPY_TILED_BC_HEADER_OP(x):  # macro
   return (((x)&0x000000FF)<<0)
SDMA_PKT_COPY_TILED_BC_HEADER_sub_op_offset = 0 # macro
SDMA_PKT_COPY_TILED_BC_HEADER_sub_op_mask = 0x000000FF # macro
SDMA_PKT_COPY_TILED_BC_HEADER_sub_op_shift = 8 # macro
def SDMA_PKT_COPY_TILED_BC_HEADER_SUB_OP(x):  # macro
   return (((x)&0x000000FF)<<8)
SDMA_PKT_COPY_TILED_BC_HEADER_detile_offset = 0 # macro
SDMA_PKT_COPY_TILED_BC_HEADER_detile_mask = 0x00000001 # macro
SDMA_PKT_COPY_TILED_BC_HEADER_detile_shift = 31 # macro
def SDMA_PKT_COPY_TILED_BC_HEADER_DETILE(x):  # macro
   return (((x)&0x00000001)<<31)
SDMA_PKT_COPY_TILED_BC_TILED_ADDR_LO_tiled_addr_31_0_offset = 1 # macro
SDMA_PKT_COPY_TILED_BC_TILED_ADDR_LO_tiled_addr_31_0_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_TILED_BC_TILED_ADDR_LO_tiled_addr_31_0_shift = 0 # macro
def SDMA_PKT_COPY_TILED_BC_TILED_ADDR_LO_TILED_ADDR_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COPY_TILED_BC_TILED_ADDR_HI_tiled_addr_63_32_offset = 2 # macro
SDMA_PKT_COPY_TILED_BC_TILED_ADDR_HI_tiled_addr_63_32_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_TILED_BC_TILED_ADDR_HI_tiled_addr_63_32_shift = 0 # macro
def SDMA_PKT_COPY_TILED_BC_TILED_ADDR_HI_TILED_ADDR_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COPY_TILED_BC_DW_3_width_offset = 3 # macro
SDMA_PKT_COPY_TILED_BC_DW_3_width_mask = 0x00003FFF # macro
SDMA_PKT_COPY_TILED_BC_DW_3_width_shift = 0 # macro
def SDMA_PKT_COPY_TILED_BC_DW_3_WIDTH(x):  # macro
   return (((x)&0x00003FFF)<<0)
SDMA_PKT_COPY_TILED_BC_DW_4_height_offset = 4 # macro
SDMA_PKT_COPY_TILED_BC_DW_4_height_mask = 0x00003FFF # macro
SDMA_PKT_COPY_TILED_BC_DW_4_height_shift = 0 # macro
def SDMA_PKT_COPY_TILED_BC_DW_4_HEIGHT(x):  # macro
   return (((x)&0x00003FFF)<<0)
SDMA_PKT_COPY_TILED_BC_DW_4_depth_offset = 4 # macro
SDMA_PKT_COPY_TILED_BC_DW_4_depth_mask = 0x000007FF # macro
SDMA_PKT_COPY_TILED_BC_DW_4_depth_shift = 16 # macro
def SDMA_PKT_COPY_TILED_BC_DW_4_DEPTH(x):  # macro
   return (((x)&0x000007FF)<<16)
SDMA_PKT_COPY_TILED_BC_DW_5_element_size_offset = 5 # macro
SDMA_PKT_COPY_TILED_BC_DW_5_element_size_mask = 0x00000007 # macro
SDMA_PKT_COPY_TILED_BC_DW_5_element_size_shift = 0 # macro
def SDMA_PKT_COPY_TILED_BC_DW_5_ELEMENT_SIZE(x):  # macro
   return (((x)&0x00000007)<<0)
SDMA_PKT_COPY_TILED_BC_DW_5_array_mode_offset = 5 # macro
SDMA_PKT_COPY_TILED_BC_DW_5_array_mode_mask = 0x0000000F # macro
SDMA_PKT_COPY_TILED_BC_DW_5_array_mode_shift = 3 # macro
def SDMA_PKT_COPY_TILED_BC_DW_5_ARRAY_MODE(x):  # macro
   return (((x)&0x0000000F)<<3)
SDMA_PKT_COPY_TILED_BC_DW_5_mit_mode_offset = 5 # macro
SDMA_PKT_COPY_TILED_BC_DW_5_mit_mode_mask = 0x00000007 # macro
SDMA_PKT_COPY_TILED_BC_DW_5_mit_mode_shift = 8 # macro
def SDMA_PKT_COPY_TILED_BC_DW_5_MIT_MODE(x):  # macro
   return (((x)&0x00000007)<<8)
SDMA_PKT_COPY_TILED_BC_DW_5_tilesplit_size_offset = 5 # macro
SDMA_PKT_COPY_TILED_BC_DW_5_tilesplit_size_mask = 0x00000007 # macro
SDMA_PKT_COPY_TILED_BC_DW_5_tilesplit_size_shift = 11 # macro
def SDMA_PKT_COPY_TILED_BC_DW_5_TILESPLIT_SIZE(x):  # macro
   return (((x)&0x00000007)<<11)
SDMA_PKT_COPY_TILED_BC_DW_5_bank_w_offset = 5 # macro
SDMA_PKT_COPY_TILED_BC_DW_5_bank_w_mask = 0x00000003 # macro
SDMA_PKT_COPY_TILED_BC_DW_5_bank_w_shift = 15 # macro
def SDMA_PKT_COPY_TILED_BC_DW_5_BANK_W(x):  # macro
   return (((x)&0x00000003)<<15)
SDMA_PKT_COPY_TILED_BC_DW_5_bank_h_offset = 5 # macro
SDMA_PKT_COPY_TILED_BC_DW_5_bank_h_mask = 0x00000003 # macro
SDMA_PKT_COPY_TILED_BC_DW_5_bank_h_shift = 18 # macro
def SDMA_PKT_COPY_TILED_BC_DW_5_BANK_H(x):  # macro
   return (((x)&0x00000003)<<18)
SDMA_PKT_COPY_TILED_BC_DW_5_num_bank_offset = 5 # macro
SDMA_PKT_COPY_TILED_BC_DW_5_num_bank_mask = 0x00000003 # macro
SDMA_PKT_COPY_TILED_BC_DW_5_num_bank_shift = 21 # macro
def SDMA_PKT_COPY_TILED_BC_DW_5_NUM_BANK(x):  # macro
   return (((x)&0x00000003)<<21)
SDMA_PKT_COPY_TILED_BC_DW_5_mat_aspt_offset = 5 # macro
SDMA_PKT_COPY_TILED_BC_DW_5_mat_aspt_mask = 0x00000003 # macro
SDMA_PKT_COPY_TILED_BC_DW_5_mat_aspt_shift = 24 # macro
def SDMA_PKT_COPY_TILED_BC_DW_5_MAT_ASPT(x):  # macro
   return (((x)&0x00000003)<<24)
SDMA_PKT_COPY_TILED_BC_DW_5_pipe_config_offset = 5 # macro
SDMA_PKT_COPY_TILED_BC_DW_5_pipe_config_mask = 0x0000001F # macro
SDMA_PKT_COPY_TILED_BC_DW_5_pipe_config_shift = 26 # macro
def SDMA_PKT_COPY_TILED_BC_DW_5_PIPE_CONFIG(x):  # macro
   return (((x)&0x0000001F)<<26)
SDMA_PKT_COPY_TILED_BC_DW_6_x_offset = 6 # macro
SDMA_PKT_COPY_TILED_BC_DW_6_x_mask = 0x00003FFF # macro
SDMA_PKT_COPY_TILED_BC_DW_6_x_shift = 0 # macro
def SDMA_PKT_COPY_TILED_BC_DW_6_X(x):  # macro
   return (((x)&0x00003FFF)<<0)
SDMA_PKT_COPY_TILED_BC_DW_6_y_offset = 6 # macro
SDMA_PKT_COPY_TILED_BC_DW_6_y_mask = 0x00003FFF # macro
SDMA_PKT_COPY_TILED_BC_DW_6_y_shift = 16 # macro
def SDMA_PKT_COPY_TILED_BC_DW_6_Y(x):  # macro
   return (((x)&0x00003FFF)<<16)
SDMA_PKT_COPY_TILED_BC_DW_7_z_offset = 7 # macro
SDMA_PKT_COPY_TILED_BC_DW_7_z_mask = 0x000007FF # macro
SDMA_PKT_COPY_TILED_BC_DW_7_z_shift = 0 # macro
def SDMA_PKT_COPY_TILED_BC_DW_7_Z(x):  # macro
   return (((x)&0x000007FF)<<0)
SDMA_PKT_COPY_TILED_BC_DW_7_linear_sw_offset = 7 # macro
SDMA_PKT_COPY_TILED_BC_DW_7_linear_sw_mask = 0x00000003 # macro
SDMA_PKT_COPY_TILED_BC_DW_7_linear_sw_shift = 16 # macro
def SDMA_PKT_COPY_TILED_BC_DW_7_LINEAR_SW(x):  # macro
   return (((x)&0x00000003)<<16)
SDMA_PKT_COPY_TILED_BC_DW_7_tile_sw_offset = 7 # macro
SDMA_PKT_COPY_TILED_BC_DW_7_tile_sw_mask = 0x00000003 # macro
SDMA_PKT_COPY_TILED_BC_DW_7_tile_sw_shift = 24 # macro
def SDMA_PKT_COPY_TILED_BC_DW_7_TILE_SW(x):  # macro
   return (((x)&0x00000003)<<24)
SDMA_PKT_COPY_TILED_BC_LINEAR_ADDR_LO_linear_addr_31_0_offset = 8 # macro
SDMA_PKT_COPY_TILED_BC_LINEAR_ADDR_LO_linear_addr_31_0_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_TILED_BC_LINEAR_ADDR_LO_linear_addr_31_0_shift = 0 # macro
def SDMA_PKT_COPY_TILED_BC_LINEAR_ADDR_LO_LINEAR_ADDR_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COPY_TILED_BC_LINEAR_ADDR_HI_linear_addr_63_32_offset = 9 # macro
SDMA_PKT_COPY_TILED_BC_LINEAR_ADDR_HI_linear_addr_63_32_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_TILED_BC_LINEAR_ADDR_HI_linear_addr_63_32_shift = 0 # macro
def SDMA_PKT_COPY_TILED_BC_LINEAR_ADDR_HI_LINEAR_ADDR_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COPY_TILED_BC_LINEAR_PITCH_linear_pitch_offset = 10 # macro
SDMA_PKT_COPY_TILED_BC_LINEAR_PITCH_linear_pitch_mask = 0x0007FFFF # macro
SDMA_PKT_COPY_TILED_BC_LINEAR_PITCH_linear_pitch_shift = 0 # macro
def SDMA_PKT_COPY_TILED_BC_LINEAR_PITCH_LINEAR_PITCH(x):  # macro
   return (((x)&0x0007FFFF)<<0)
SDMA_PKT_COPY_TILED_BC_COUNT_count_offset = 11 # macro
SDMA_PKT_COPY_TILED_BC_COUNT_count_mask = 0x000FFFFF # macro
SDMA_PKT_COPY_TILED_BC_COUNT_count_shift = 2 # macro
def SDMA_PKT_COPY_TILED_BC_COUNT_COUNT(x):  # macro
   return (((x)&0x000FFFFF)<<2)
SDMA_PKT_COPY_L2T_BROADCAST_HEADER_op_offset = 0 # macro
SDMA_PKT_COPY_L2T_BROADCAST_HEADER_op_mask = 0x000000FF # macro
SDMA_PKT_COPY_L2T_BROADCAST_HEADER_op_shift = 0 # macro
def SDMA_PKT_COPY_L2T_BROADCAST_HEADER_OP(x):  # macro
   return (((x)&0x000000FF)<<0)
SDMA_PKT_COPY_L2T_BROADCAST_HEADER_sub_op_offset = 0 # macro
SDMA_PKT_COPY_L2T_BROADCAST_HEADER_sub_op_mask = 0x000000FF # macro
SDMA_PKT_COPY_L2T_BROADCAST_HEADER_sub_op_shift = 8 # macro
def SDMA_PKT_COPY_L2T_BROADCAST_HEADER_SUB_OP(x):  # macro
   return (((x)&0x000000FF)<<8)
SDMA_PKT_COPY_L2T_BROADCAST_HEADER_encrypt_offset = 0 # macro
SDMA_PKT_COPY_L2T_BROADCAST_HEADER_encrypt_mask = 0x00000001 # macro
SDMA_PKT_COPY_L2T_BROADCAST_HEADER_encrypt_shift = 16 # macro
def SDMA_PKT_COPY_L2T_BROADCAST_HEADER_ENCRYPT(x):  # macro
   return (((x)&0x00000001)<<16)
SDMA_PKT_COPY_L2T_BROADCAST_HEADER_tmz_offset = 0 # macro
SDMA_PKT_COPY_L2T_BROADCAST_HEADER_tmz_mask = 0x00000001 # macro
SDMA_PKT_COPY_L2T_BROADCAST_HEADER_tmz_shift = 18 # macro
def SDMA_PKT_COPY_L2T_BROADCAST_HEADER_TMZ(x):  # macro
   return (((x)&0x00000001)<<18)
SDMA_PKT_COPY_L2T_BROADCAST_HEADER_videocopy_offset = 0 # macro
SDMA_PKT_COPY_L2T_BROADCAST_HEADER_videocopy_mask = 0x00000001 # macro
SDMA_PKT_COPY_L2T_BROADCAST_HEADER_videocopy_shift = 26 # macro
def SDMA_PKT_COPY_L2T_BROADCAST_HEADER_VIDEOCOPY(x):  # macro
   return (((x)&0x00000001)<<26)
SDMA_PKT_COPY_L2T_BROADCAST_HEADER_broadcast_offset = 0 # macro
SDMA_PKT_COPY_L2T_BROADCAST_HEADER_broadcast_mask = 0x00000001 # macro
SDMA_PKT_COPY_L2T_BROADCAST_HEADER_broadcast_shift = 27 # macro
def SDMA_PKT_COPY_L2T_BROADCAST_HEADER_BROADCAST(x):  # macro
   return (((x)&0x00000001)<<27)
SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_LO_0_tiled_addr0_31_0_offset = 1 # macro
SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_LO_0_tiled_addr0_31_0_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_LO_0_tiled_addr0_31_0_shift = 0 # macro
def SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_LO_0_TILED_ADDR0_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_HI_0_tiled_addr0_63_32_offset = 2 # macro
SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_HI_0_tiled_addr0_63_32_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_HI_0_tiled_addr0_63_32_shift = 0 # macro
def SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_HI_0_TILED_ADDR0_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_LO_1_tiled_addr1_31_0_offset = 3 # macro
SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_LO_1_tiled_addr1_31_0_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_LO_1_tiled_addr1_31_0_shift = 0 # macro
def SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_LO_1_TILED_ADDR1_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_HI_1_tiled_addr1_63_32_offset = 4 # macro
SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_HI_1_tiled_addr1_63_32_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_HI_1_tiled_addr1_63_32_shift = 0 # macro
def SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_HI_1_TILED_ADDR1_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COPY_L2T_BROADCAST_DW_5_width_offset = 5 # macro
SDMA_PKT_COPY_L2T_BROADCAST_DW_5_width_mask = 0x00003FFF # macro
SDMA_PKT_COPY_L2T_BROADCAST_DW_5_width_shift = 0 # macro
def SDMA_PKT_COPY_L2T_BROADCAST_DW_5_WIDTH(x):  # macro
   return (((x)&0x00003FFF)<<0)
SDMA_PKT_COPY_L2T_BROADCAST_DW_6_height_offset = 6 # macro
SDMA_PKT_COPY_L2T_BROADCAST_DW_6_height_mask = 0x00003FFF # macro
SDMA_PKT_COPY_L2T_BROADCAST_DW_6_height_shift = 0 # macro
def SDMA_PKT_COPY_L2T_BROADCAST_DW_6_HEIGHT(x):  # macro
   return (((x)&0x00003FFF)<<0)
SDMA_PKT_COPY_L2T_BROADCAST_DW_6_depth_offset = 6 # macro
SDMA_PKT_COPY_L2T_BROADCAST_DW_6_depth_mask = 0x00001FFF # macro
SDMA_PKT_COPY_L2T_BROADCAST_DW_6_depth_shift = 16 # macro
def SDMA_PKT_COPY_L2T_BROADCAST_DW_6_DEPTH(x):  # macro
   return (((x)&0x00001FFF)<<16)
SDMA_PKT_COPY_L2T_BROADCAST_DW_7_element_size_offset = 7 # macro
SDMA_PKT_COPY_L2T_BROADCAST_DW_7_element_size_mask = 0x00000007 # macro
SDMA_PKT_COPY_L2T_BROADCAST_DW_7_element_size_shift = 0 # macro
def SDMA_PKT_COPY_L2T_BROADCAST_DW_7_ELEMENT_SIZE(x):  # macro
   return (((x)&0x00000007)<<0)
SDMA_PKT_COPY_L2T_BROADCAST_DW_7_swizzle_mode_offset = 7 # macro
SDMA_PKT_COPY_L2T_BROADCAST_DW_7_swizzle_mode_mask = 0x0000001F # macro
SDMA_PKT_COPY_L2T_BROADCAST_DW_7_swizzle_mode_shift = 3 # macro
def SDMA_PKT_COPY_L2T_BROADCAST_DW_7_SWIZZLE_MODE(x):  # macro
   return (((x)&0x0000001F)<<3)
SDMA_PKT_COPY_L2T_BROADCAST_DW_7_dimension_offset = 7 # macro
SDMA_PKT_COPY_L2T_BROADCAST_DW_7_dimension_mask = 0x00000003 # macro
SDMA_PKT_COPY_L2T_BROADCAST_DW_7_dimension_shift = 9 # macro
def SDMA_PKT_COPY_L2T_BROADCAST_DW_7_DIMENSION(x):  # macro
   return (((x)&0x00000003)<<9)
SDMA_PKT_COPY_L2T_BROADCAST_DW_7_mip_max_offset = 7 # macro
SDMA_PKT_COPY_L2T_BROADCAST_DW_7_mip_max_mask = 0x0000000F # macro
SDMA_PKT_COPY_L2T_BROADCAST_DW_7_mip_max_shift = 16 # macro
def SDMA_PKT_COPY_L2T_BROADCAST_DW_7_MIP_MAX(x):  # macro
   return (((x)&0x0000000F)<<16)
SDMA_PKT_COPY_L2T_BROADCAST_DW_8_x_offset = 8 # macro
SDMA_PKT_COPY_L2T_BROADCAST_DW_8_x_mask = 0x00003FFF # macro
SDMA_PKT_COPY_L2T_BROADCAST_DW_8_x_shift = 0 # macro
def SDMA_PKT_COPY_L2T_BROADCAST_DW_8_X(x):  # macro
   return (((x)&0x00003FFF)<<0)
SDMA_PKT_COPY_L2T_BROADCAST_DW_8_y_offset = 8 # macro
SDMA_PKT_COPY_L2T_BROADCAST_DW_8_y_mask = 0x00003FFF # macro
SDMA_PKT_COPY_L2T_BROADCAST_DW_8_y_shift = 16 # macro
def SDMA_PKT_COPY_L2T_BROADCAST_DW_8_Y(x):  # macro
   return (((x)&0x00003FFF)<<16)
SDMA_PKT_COPY_L2T_BROADCAST_DW_9_z_offset = 9 # macro
SDMA_PKT_COPY_L2T_BROADCAST_DW_9_z_mask = 0x00001FFF # macro
SDMA_PKT_COPY_L2T_BROADCAST_DW_9_z_shift = 0 # macro
def SDMA_PKT_COPY_L2T_BROADCAST_DW_9_Z(x):  # macro
   return (((x)&0x00001FFF)<<0)
SDMA_PKT_COPY_L2T_BROADCAST_DW_10_dst2_sw_offset = 10 # macro
SDMA_PKT_COPY_L2T_BROADCAST_DW_10_dst2_sw_mask = 0x00000003 # macro
SDMA_PKT_COPY_L2T_BROADCAST_DW_10_dst2_sw_shift = 8 # macro
def SDMA_PKT_COPY_L2T_BROADCAST_DW_10_DST2_SW(x):  # macro
   return (((x)&0x00000003)<<8)
SDMA_PKT_COPY_L2T_BROADCAST_DW_10_linear_sw_offset = 10 # macro
SDMA_PKT_COPY_L2T_BROADCAST_DW_10_linear_sw_mask = 0x00000003 # macro
SDMA_PKT_COPY_L2T_BROADCAST_DW_10_linear_sw_shift = 16 # macro
def SDMA_PKT_COPY_L2T_BROADCAST_DW_10_LINEAR_SW(x):  # macro
   return (((x)&0x00000003)<<16)
SDMA_PKT_COPY_L2T_BROADCAST_DW_10_tile_sw_offset = 10 # macro
SDMA_PKT_COPY_L2T_BROADCAST_DW_10_tile_sw_mask = 0x00000003 # macro
SDMA_PKT_COPY_L2T_BROADCAST_DW_10_tile_sw_shift = 24 # macro
def SDMA_PKT_COPY_L2T_BROADCAST_DW_10_TILE_SW(x):  # macro
   return (((x)&0x00000003)<<24)
SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_ADDR_LO_linear_addr_31_0_offset = 11 # macro
SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_ADDR_LO_linear_addr_31_0_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_ADDR_LO_linear_addr_31_0_shift = 0 # macro
def SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_ADDR_LO_LINEAR_ADDR_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_ADDR_HI_linear_addr_63_32_offset = 12 # macro
SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_ADDR_HI_linear_addr_63_32_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_ADDR_HI_linear_addr_63_32_shift = 0 # macro
def SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_ADDR_HI_LINEAR_ADDR_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_PITCH_linear_pitch_offset = 13 # macro
SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_PITCH_linear_pitch_mask = 0x0007FFFF # macro
SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_PITCH_linear_pitch_shift = 0 # macro
def SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_PITCH_LINEAR_PITCH(x):  # macro
   return (((x)&0x0007FFFF)<<0)
SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_SLICE_PITCH_linear_slice_pitch_offset = 14 # macro
SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_SLICE_PITCH_linear_slice_pitch_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_SLICE_PITCH_linear_slice_pitch_shift = 0 # macro
def SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_SLICE_PITCH_LINEAR_SLICE_PITCH(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COPY_L2T_BROADCAST_COUNT_count_offset = 15 # macro
SDMA_PKT_COPY_L2T_BROADCAST_COUNT_count_mask = 0x003FFFFF # macro
SDMA_PKT_COPY_L2T_BROADCAST_COUNT_count_shift = 0 # macro
def SDMA_PKT_COPY_L2T_BROADCAST_COUNT_COUNT(x):  # macro
   return (((x)&0x003FFFFF)<<0)
SDMA_PKT_COPY_T2T_HEADER_op_offset = 0 # macro
SDMA_PKT_COPY_T2T_HEADER_op_mask = 0x000000FF # macro
SDMA_PKT_COPY_T2T_HEADER_op_shift = 0 # macro
def SDMA_PKT_COPY_T2T_HEADER_OP(x):  # macro
   return (((x)&0x000000FF)<<0)
SDMA_PKT_COPY_T2T_HEADER_sub_op_offset = 0 # macro
SDMA_PKT_COPY_T2T_HEADER_sub_op_mask = 0x000000FF # macro
SDMA_PKT_COPY_T2T_HEADER_sub_op_shift = 8 # macro
def SDMA_PKT_COPY_T2T_HEADER_SUB_OP(x):  # macro
   return (((x)&0x000000FF)<<8)
SDMA_PKT_COPY_T2T_HEADER_tmz_offset = 0 # macro
SDMA_PKT_COPY_T2T_HEADER_tmz_mask = 0x00000001 # macro
SDMA_PKT_COPY_T2T_HEADER_tmz_shift = 18 # macro
def SDMA_PKT_COPY_T2T_HEADER_TMZ(x):  # macro
   return (((x)&0x00000001)<<18)
SDMA_PKT_COPY_T2T_HEADER_dcc_offset = 0 # macro
SDMA_PKT_COPY_T2T_HEADER_dcc_mask = 0x00000001 # macro
SDMA_PKT_COPY_T2T_HEADER_dcc_shift = 19 # macro
def SDMA_PKT_COPY_T2T_HEADER_DCC(x):  # macro
   return (((x)&0x00000001)<<19)
SDMA_PKT_COPY_T2T_HEADER_dcc_dir_offset = 0 # macro
SDMA_PKT_COPY_T2T_HEADER_dcc_dir_mask = 0x00000001 # macro
SDMA_PKT_COPY_T2T_HEADER_dcc_dir_shift = 31 # macro
def SDMA_PKT_COPY_T2T_HEADER_DCC_DIR(x):  # macro
   return (((x)&0x00000001)<<31)
SDMA_PKT_COPY_T2T_SRC_ADDR_LO_src_addr_31_0_offset = 1 # macro
SDMA_PKT_COPY_T2T_SRC_ADDR_LO_src_addr_31_0_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_T2T_SRC_ADDR_LO_src_addr_31_0_shift = 0 # macro
def SDMA_PKT_COPY_T2T_SRC_ADDR_LO_SRC_ADDR_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COPY_T2T_SRC_ADDR_HI_src_addr_63_32_offset = 2 # macro
SDMA_PKT_COPY_T2T_SRC_ADDR_HI_src_addr_63_32_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_T2T_SRC_ADDR_HI_src_addr_63_32_shift = 0 # macro
def SDMA_PKT_COPY_T2T_SRC_ADDR_HI_SRC_ADDR_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COPY_T2T_DW_3_src_x_offset = 3 # macro
SDMA_PKT_COPY_T2T_DW_3_src_x_mask = 0x00003FFF # macro
SDMA_PKT_COPY_T2T_DW_3_src_x_shift = 0 # macro
def SDMA_PKT_COPY_T2T_DW_3_SRC_X(x):  # macro
   return (((x)&0x00003FFF)<<0)
SDMA_PKT_COPY_T2T_DW_3_src_y_offset = 3 # macro
SDMA_PKT_COPY_T2T_DW_3_src_y_mask = 0x00003FFF # macro
SDMA_PKT_COPY_T2T_DW_3_src_y_shift = 16 # macro
def SDMA_PKT_COPY_T2T_DW_3_SRC_Y(x):  # macro
   return (((x)&0x00003FFF)<<16)
SDMA_PKT_COPY_T2T_DW_4_src_z_offset = 4 # macro
SDMA_PKT_COPY_T2T_DW_4_src_z_mask = 0x00001FFF # macro
SDMA_PKT_COPY_T2T_DW_4_src_z_shift = 0 # macro
def SDMA_PKT_COPY_T2T_DW_4_SRC_Z(x):  # macro
   return (((x)&0x00001FFF)<<0)
SDMA_PKT_COPY_T2T_DW_4_src_width_offset = 4 # macro
SDMA_PKT_COPY_T2T_DW_4_src_width_mask = 0x00003FFF # macro
SDMA_PKT_COPY_T2T_DW_4_src_width_shift = 16 # macro
def SDMA_PKT_COPY_T2T_DW_4_SRC_WIDTH(x):  # macro
   return (((x)&0x00003FFF)<<16)
SDMA_PKT_COPY_T2T_DW_5_src_height_offset = 5 # macro
SDMA_PKT_COPY_T2T_DW_5_src_height_mask = 0x00003FFF # macro
SDMA_PKT_COPY_T2T_DW_5_src_height_shift = 0 # macro
def SDMA_PKT_COPY_T2T_DW_5_SRC_HEIGHT(x):  # macro
   return (((x)&0x00003FFF)<<0)
SDMA_PKT_COPY_T2T_DW_5_src_depth_offset = 5 # macro
SDMA_PKT_COPY_T2T_DW_5_src_depth_mask = 0x00001FFF # macro
SDMA_PKT_COPY_T2T_DW_5_src_depth_shift = 16 # macro
def SDMA_PKT_COPY_T2T_DW_5_SRC_DEPTH(x):  # macro
   return (((x)&0x00001FFF)<<16)
SDMA_PKT_COPY_T2T_DW_6_src_element_size_offset = 6 # macro
SDMA_PKT_COPY_T2T_DW_6_src_element_size_mask = 0x00000007 # macro
SDMA_PKT_COPY_T2T_DW_6_src_element_size_shift = 0 # macro
def SDMA_PKT_COPY_T2T_DW_6_SRC_ELEMENT_SIZE(x):  # macro
   return (((x)&0x00000007)<<0)
SDMA_PKT_COPY_T2T_DW_6_src_swizzle_mode_offset = 6 # macro
SDMA_PKT_COPY_T2T_DW_6_src_swizzle_mode_mask = 0x0000001F # macro
SDMA_PKT_COPY_T2T_DW_6_src_swizzle_mode_shift = 3 # macro
def SDMA_PKT_COPY_T2T_DW_6_SRC_SWIZZLE_MODE(x):  # macro
   return (((x)&0x0000001F)<<3)
SDMA_PKT_COPY_T2T_DW_6_src_dimension_offset = 6 # macro
SDMA_PKT_COPY_T2T_DW_6_src_dimension_mask = 0x00000003 # macro
SDMA_PKT_COPY_T2T_DW_6_src_dimension_shift = 9 # macro
def SDMA_PKT_COPY_T2T_DW_6_SRC_DIMENSION(x):  # macro
   return (((x)&0x00000003)<<9)
SDMA_PKT_COPY_T2T_DW_6_src_mip_max_offset = 6 # macro
SDMA_PKT_COPY_T2T_DW_6_src_mip_max_mask = 0x0000000F # macro
SDMA_PKT_COPY_T2T_DW_6_src_mip_max_shift = 16 # macro
def SDMA_PKT_COPY_T2T_DW_6_SRC_MIP_MAX(x):  # macro
   return (((x)&0x0000000F)<<16)
SDMA_PKT_COPY_T2T_DW_6_src_mip_id_offset = 6 # macro
SDMA_PKT_COPY_T2T_DW_6_src_mip_id_mask = 0x0000000F # macro
SDMA_PKT_COPY_T2T_DW_6_src_mip_id_shift = 20 # macro
def SDMA_PKT_COPY_T2T_DW_6_SRC_MIP_ID(x):  # macro
   return (((x)&0x0000000F)<<20)
SDMA_PKT_COPY_T2T_DST_ADDR_LO_dst_addr_31_0_offset = 7 # macro
SDMA_PKT_COPY_T2T_DST_ADDR_LO_dst_addr_31_0_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_T2T_DST_ADDR_LO_dst_addr_31_0_shift = 0 # macro
def SDMA_PKT_COPY_T2T_DST_ADDR_LO_DST_ADDR_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COPY_T2T_DST_ADDR_HI_dst_addr_63_32_offset = 8 # macro
SDMA_PKT_COPY_T2T_DST_ADDR_HI_dst_addr_63_32_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_T2T_DST_ADDR_HI_dst_addr_63_32_shift = 0 # macro
def SDMA_PKT_COPY_T2T_DST_ADDR_HI_DST_ADDR_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COPY_T2T_DW_9_dst_x_offset = 9 # macro
SDMA_PKT_COPY_T2T_DW_9_dst_x_mask = 0x00003FFF # macro
SDMA_PKT_COPY_T2T_DW_9_dst_x_shift = 0 # macro
def SDMA_PKT_COPY_T2T_DW_9_DST_X(x):  # macro
   return (((x)&0x00003FFF)<<0)
SDMA_PKT_COPY_T2T_DW_9_dst_y_offset = 9 # macro
SDMA_PKT_COPY_T2T_DW_9_dst_y_mask = 0x00003FFF # macro
SDMA_PKT_COPY_T2T_DW_9_dst_y_shift = 16 # macro
def SDMA_PKT_COPY_T2T_DW_9_DST_Y(x):  # macro
   return (((x)&0x00003FFF)<<16)
SDMA_PKT_COPY_T2T_DW_10_dst_z_offset = 10 # macro
SDMA_PKT_COPY_T2T_DW_10_dst_z_mask = 0x00001FFF # macro
SDMA_PKT_COPY_T2T_DW_10_dst_z_shift = 0 # macro
def SDMA_PKT_COPY_T2T_DW_10_DST_Z(x):  # macro
   return (((x)&0x00001FFF)<<0)
SDMA_PKT_COPY_T2T_DW_10_dst_width_offset = 10 # macro
SDMA_PKT_COPY_T2T_DW_10_dst_width_mask = 0x00003FFF # macro
SDMA_PKT_COPY_T2T_DW_10_dst_width_shift = 16 # macro
def SDMA_PKT_COPY_T2T_DW_10_DST_WIDTH(x):  # macro
   return (((x)&0x00003FFF)<<16)
SDMA_PKT_COPY_T2T_DW_11_dst_height_offset = 11 # macro
SDMA_PKT_COPY_T2T_DW_11_dst_height_mask = 0x00003FFF # macro
SDMA_PKT_COPY_T2T_DW_11_dst_height_shift = 0 # macro
def SDMA_PKT_COPY_T2T_DW_11_DST_HEIGHT(x):  # macro
   return (((x)&0x00003FFF)<<0)
SDMA_PKT_COPY_T2T_DW_11_dst_depth_offset = 11 # macro
SDMA_PKT_COPY_T2T_DW_11_dst_depth_mask = 0x00001FFF # macro
SDMA_PKT_COPY_T2T_DW_11_dst_depth_shift = 16 # macro
def SDMA_PKT_COPY_T2T_DW_11_DST_DEPTH(x):  # macro
   return (((x)&0x00001FFF)<<16)
SDMA_PKT_COPY_T2T_DW_12_dst_element_size_offset = 12 # macro
SDMA_PKT_COPY_T2T_DW_12_dst_element_size_mask = 0x00000007 # macro
SDMA_PKT_COPY_T2T_DW_12_dst_element_size_shift = 0 # macro
def SDMA_PKT_COPY_T2T_DW_12_DST_ELEMENT_SIZE(x):  # macro
   return (((x)&0x00000007)<<0)
SDMA_PKT_COPY_T2T_DW_12_dst_swizzle_mode_offset = 12 # macro
SDMA_PKT_COPY_T2T_DW_12_dst_swizzle_mode_mask = 0x0000001F # macro
SDMA_PKT_COPY_T2T_DW_12_dst_swizzle_mode_shift = 3 # macro
def SDMA_PKT_COPY_T2T_DW_12_DST_SWIZZLE_MODE(x):  # macro
   return (((x)&0x0000001F)<<3)
SDMA_PKT_COPY_T2T_DW_12_dst_dimension_offset = 12 # macro
SDMA_PKT_COPY_T2T_DW_12_dst_dimension_mask = 0x00000003 # macro
SDMA_PKT_COPY_T2T_DW_12_dst_dimension_shift = 9 # macro
def SDMA_PKT_COPY_T2T_DW_12_DST_DIMENSION(x):  # macro
   return (((x)&0x00000003)<<9)
SDMA_PKT_COPY_T2T_DW_12_dst_mip_max_offset = 12 # macro
SDMA_PKT_COPY_T2T_DW_12_dst_mip_max_mask = 0x0000000F # macro
SDMA_PKT_COPY_T2T_DW_12_dst_mip_max_shift = 16 # macro
def SDMA_PKT_COPY_T2T_DW_12_DST_MIP_MAX(x):  # macro
   return (((x)&0x0000000F)<<16)
SDMA_PKT_COPY_T2T_DW_12_dst_mip_id_offset = 12 # macro
SDMA_PKT_COPY_T2T_DW_12_dst_mip_id_mask = 0x0000000F # macro
SDMA_PKT_COPY_T2T_DW_12_dst_mip_id_shift = 20 # macro
def SDMA_PKT_COPY_T2T_DW_12_DST_MIP_ID(x):  # macro
   return (((x)&0x0000000F)<<20)
SDMA_PKT_COPY_T2T_DW_13_rect_x_offset = 13 # macro
SDMA_PKT_COPY_T2T_DW_13_rect_x_mask = 0x00003FFF # macro
SDMA_PKT_COPY_T2T_DW_13_rect_x_shift = 0 # macro
def SDMA_PKT_COPY_T2T_DW_13_RECT_X(x):  # macro
   return (((x)&0x00003FFF)<<0)
SDMA_PKT_COPY_T2T_DW_13_rect_y_offset = 13 # macro
SDMA_PKT_COPY_T2T_DW_13_rect_y_mask = 0x00003FFF # macro
SDMA_PKT_COPY_T2T_DW_13_rect_y_shift = 16 # macro
def SDMA_PKT_COPY_T2T_DW_13_RECT_Y(x):  # macro
   return (((x)&0x00003FFF)<<16)
SDMA_PKT_COPY_T2T_DW_14_rect_z_offset = 14 # macro
SDMA_PKT_COPY_T2T_DW_14_rect_z_mask = 0x00001FFF # macro
SDMA_PKT_COPY_T2T_DW_14_rect_z_shift = 0 # macro
def SDMA_PKT_COPY_T2T_DW_14_RECT_Z(x):  # macro
   return (((x)&0x00001FFF)<<0)
SDMA_PKT_COPY_T2T_DW_14_dst_sw_offset = 14 # macro
SDMA_PKT_COPY_T2T_DW_14_dst_sw_mask = 0x00000003 # macro
SDMA_PKT_COPY_T2T_DW_14_dst_sw_shift = 16 # macro
def SDMA_PKT_COPY_T2T_DW_14_DST_SW(x):  # macro
   return (((x)&0x00000003)<<16)
SDMA_PKT_COPY_T2T_DW_14_src_sw_offset = 14 # macro
SDMA_PKT_COPY_T2T_DW_14_src_sw_mask = 0x00000003 # macro
SDMA_PKT_COPY_T2T_DW_14_src_sw_shift = 24 # macro
def SDMA_PKT_COPY_T2T_DW_14_SRC_SW(x):  # macro
   return (((x)&0x00000003)<<24)
SDMA_PKT_COPY_T2T_META_ADDR_LO_meta_addr_31_0_offset = 15 # macro
SDMA_PKT_COPY_T2T_META_ADDR_LO_meta_addr_31_0_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_T2T_META_ADDR_LO_meta_addr_31_0_shift = 0 # macro
def SDMA_PKT_COPY_T2T_META_ADDR_LO_META_ADDR_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COPY_T2T_META_ADDR_HI_meta_addr_63_32_offset = 16 # macro
SDMA_PKT_COPY_T2T_META_ADDR_HI_meta_addr_63_32_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_T2T_META_ADDR_HI_meta_addr_63_32_shift = 0 # macro
def SDMA_PKT_COPY_T2T_META_ADDR_HI_META_ADDR_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COPY_T2T_META_CONFIG_data_format_offset = 17 # macro
SDMA_PKT_COPY_T2T_META_CONFIG_data_format_mask = 0x0000007F # macro
SDMA_PKT_COPY_T2T_META_CONFIG_data_format_shift = 0 # macro
def SDMA_PKT_COPY_T2T_META_CONFIG_DATA_FORMAT(x):  # macro
   return (((x)&0x0000007F)<<0)
SDMA_PKT_COPY_T2T_META_CONFIG_color_transform_disable_offset = 17 # macro
SDMA_PKT_COPY_T2T_META_CONFIG_color_transform_disable_mask = 0x00000001 # macro
SDMA_PKT_COPY_T2T_META_CONFIG_color_transform_disable_shift = 7 # macro
def SDMA_PKT_COPY_T2T_META_CONFIG_COLOR_TRANSFORM_DISABLE(x):  # macro
   return (((x)&0x00000001)<<7)
SDMA_PKT_COPY_T2T_META_CONFIG_alpha_is_on_msb_offset = 17 # macro
SDMA_PKT_COPY_T2T_META_CONFIG_alpha_is_on_msb_mask = 0x00000001 # macro
SDMA_PKT_COPY_T2T_META_CONFIG_alpha_is_on_msb_shift = 8 # macro
def SDMA_PKT_COPY_T2T_META_CONFIG_ALPHA_IS_ON_MSB(x):  # macro
   return (((x)&0x00000001)<<8)
SDMA_PKT_COPY_T2T_META_CONFIG_number_type_offset = 17 # macro
SDMA_PKT_COPY_T2T_META_CONFIG_number_type_mask = 0x00000007 # macro
SDMA_PKT_COPY_T2T_META_CONFIG_number_type_shift = 9 # macro
def SDMA_PKT_COPY_T2T_META_CONFIG_NUMBER_TYPE(x):  # macro
   return (((x)&0x00000007)<<9)
SDMA_PKT_COPY_T2T_META_CONFIG_surface_type_offset = 17 # macro
SDMA_PKT_COPY_T2T_META_CONFIG_surface_type_mask = 0x00000003 # macro
SDMA_PKT_COPY_T2T_META_CONFIG_surface_type_shift = 12 # macro
def SDMA_PKT_COPY_T2T_META_CONFIG_SURFACE_TYPE(x):  # macro
   return (((x)&0x00000003)<<12)
SDMA_PKT_COPY_T2T_META_CONFIG_max_comp_block_size_offset = 17 # macro
SDMA_PKT_COPY_T2T_META_CONFIG_max_comp_block_size_mask = 0x00000003 # macro
SDMA_PKT_COPY_T2T_META_CONFIG_max_comp_block_size_shift = 24 # macro
def SDMA_PKT_COPY_T2T_META_CONFIG_MAX_COMP_BLOCK_SIZE(x):  # macro
   return (((x)&0x00000003)<<24)
SDMA_PKT_COPY_T2T_META_CONFIG_max_uncomp_block_size_offset = 17 # macro
SDMA_PKT_COPY_T2T_META_CONFIG_max_uncomp_block_size_mask = 0x00000003 # macro
SDMA_PKT_COPY_T2T_META_CONFIG_max_uncomp_block_size_shift = 26 # macro
def SDMA_PKT_COPY_T2T_META_CONFIG_MAX_UNCOMP_BLOCK_SIZE(x):  # macro
   return (((x)&0x00000003)<<26)
SDMA_PKT_COPY_T2T_META_CONFIG_write_compress_enable_offset = 17 # macro
SDMA_PKT_COPY_T2T_META_CONFIG_write_compress_enable_mask = 0x00000001 # macro
SDMA_PKT_COPY_T2T_META_CONFIG_write_compress_enable_shift = 28 # macro
def SDMA_PKT_COPY_T2T_META_CONFIG_WRITE_COMPRESS_ENABLE(x):  # macro
   return (((x)&0x00000001)<<28)
SDMA_PKT_COPY_T2T_META_CONFIG_meta_tmz_offset = 17 # macro
SDMA_PKT_COPY_T2T_META_CONFIG_meta_tmz_mask = 0x00000001 # macro
SDMA_PKT_COPY_T2T_META_CONFIG_meta_tmz_shift = 29 # macro
def SDMA_PKT_COPY_T2T_META_CONFIG_META_TMZ(x):  # macro
   return (((x)&0x00000001)<<29)
SDMA_PKT_COPY_T2T_BC_HEADER_op_offset = 0 # macro
SDMA_PKT_COPY_T2T_BC_HEADER_op_mask = 0x000000FF # macro
SDMA_PKT_COPY_T2T_BC_HEADER_op_shift = 0 # macro
def SDMA_PKT_COPY_T2T_BC_HEADER_OP(x):  # macro
   return (((x)&0x000000FF)<<0)
SDMA_PKT_COPY_T2T_BC_HEADER_sub_op_offset = 0 # macro
SDMA_PKT_COPY_T2T_BC_HEADER_sub_op_mask = 0x000000FF # macro
SDMA_PKT_COPY_T2T_BC_HEADER_sub_op_shift = 8 # macro
def SDMA_PKT_COPY_T2T_BC_HEADER_SUB_OP(x):  # macro
   return (((x)&0x000000FF)<<8)
SDMA_PKT_COPY_T2T_BC_SRC_ADDR_LO_src_addr_31_0_offset = 1 # macro
SDMA_PKT_COPY_T2T_BC_SRC_ADDR_LO_src_addr_31_0_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_T2T_BC_SRC_ADDR_LO_src_addr_31_0_shift = 0 # macro
def SDMA_PKT_COPY_T2T_BC_SRC_ADDR_LO_SRC_ADDR_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COPY_T2T_BC_SRC_ADDR_HI_src_addr_63_32_offset = 2 # macro
SDMA_PKT_COPY_T2T_BC_SRC_ADDR_HI_src_addr_63_32_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_T2T_BC_SRC_ADDR_HI_src_addr_63_32_shift = 0 # macro
def SDMA_PKT_COPY_T2T_BC_SRC_ADDR_HI_SRC_ADDR_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COPY_T2T_BC_DW_3_src_x_offset = 3 # macro
SDMA_PKT_COPY_T2T_BC_DW_3_src_x_mask = 0x00003FFF # macro
SDMA_PKT_COPY_T2T_BC_DW_3_src_x_shift = 0 # macro
def SDMA_PKT_COPY_T2T_BC_DW_3_SRC_X(x):  # macro
   return (((x)&0x00003FFF)<<0)
SDMA_PKT_COPY_T2T_BC_DW_3_src_y_offset = 3 # macro
SDMA_PKT_COPY_T2T_BC_DW_3_src_y_mask = 0x00003FFF # macro
SDMA_PKT_COPY_T2T_BC_DW_3_src_y_shift = 16 # macro
def SDMA_PKT_COPY_T2T_BC_DW_3_SRC_Y(x):  # macro
   return (((x)&0x00003FFF)<<16)
SDMA_PKT_COPY_T2T_BC_DW_4_src_z_offset = 4 # macro
SDMA_PKT_COPY_T2T_BC_DW_4_src_z_mask = 0x000007FF # macro
SDMA_PKT_COPY_T2T_BC_DW_4_src_z_shift = 0 # macro
def SDMA_PKT_COPY_T2T_BC_DW_4_SRC_Z(x):  # macro
   return (((x)&0x000007FF)<<0)
SDMA_PKT_COPY_T2T_BC_DW_4_src_width_offset = 4 # macro
SDMA_PKT_COPY_T2T_BC_DW_4_src_width_mask = 0x00003FFF # macro
SDMA_PKT_COPY_T2T_BC_DW_4_src_width_shift = 16 # macro
def SDMA_PKT_COPY_T2T_BC_DW_4_SRC_WIDTH(x):  # macro
   return (((x)&0x00003FFF)<<16)
SDMA_PKT_COPY_T2T_BC_DW_5_src_height_offset = 5 # macro
SDMA_PKT_COPY_T2T_BC_DW_5_src_height_mask = 0x00003FFF # macro
SDMA_PKT_COPY_T2T_BC_DW_5_src_height_shift = 0 # macro
def SDMA_PKT_COPY_T2T_BC_DW_5_SRC_HEIGHT(x):  # macro
   return (((x)&0x00003FFF)<<0)
SDMA_PKT_COPY_T2T_BC_DW_5_src_depth_offset = 5 # macro
SDMA_PKT_COPY_T2T_BC_DW_5_src_depth_mask = 0x000007FF # macro
SDMA_PKT_COPY_T2T_BC_DW_5_src_depth_shift = 16 # macro
def SDMA_PKT_COPY_T2T_BC_DW_5_SRC_DEPTH(x):  # macro
   return (((x)&0x000007FF)<<16)
SDMA_PKT_COPY_T2T_BC_DW_6_src_element_size_offset = 6 # macro
SDMA_PKT_COPY_T2T_BC_DW_6_src_element_size_mask = 0x00000007 # macro
SDMA_PKT_COPY_T2T_BC_DW_6_src_element_size_shift = 0 # macro
def SDMA_PKT_COPY_T2T_BC_DW_6_SRC_ELEMENT_SIZE(x):  # macro
   return (((x)&0x00000007)<<0)
SDMA_PKT_COPY_T2T_BC_DW_6_src_array_mode_offset = 6 # macro
SDMA_PKT_COPY_T2T_BC_DW_6_src_array_mode_mask = 0x0000000F # macro
SDMA_PKT_COPY_T2T_BC_DW_6_src_array_mode_shift = 3 # macro
def SDMA_PKT_COPY_T2T_BC_DW_6_SRC_ARRAY_MODE(x):  # macro
   return (((x)&0x0000000F)<<3)
SDMA_PKT_COPY_T2T_BC_DW_6_src_mit_mode_offset = 6 # macro
SDMA_PKT_COPY_T2T_BC_DW_6_src_mit_mode_mask = 0x00000007 # macro
SDMA_PKT_COPY_T2T_BC_DW_6_src_mit_mode_shift = 8 # macro
def SDMA_PKT_COPY_T2T_BC_DW_6_SRC_MIT_MODE(x):  # macro
   return (((x)&0x00000007)<<8)
SDMA_PKT_COPY_T2T_BC_DW_6_src_tilesplit_size_offset = 6 # macro
SDMA_PKT_COPY_T2T_BC_DW_6_src_tilesplit_size_mask = 0x00000007 # macro
SDMA_PKT_COPY_T2T_BC_DW_6_src_tilesplit_size_shift = 11 # macro
def SDMA_PKT_COPY_T2T_BC_DW_6_SRC_TILESPLIT_SIZE(x):  # macro
   return (((x)&0x00000007)<<11)
SDMA_PKT_COPY_T2T_BC_DW_6_src_bank_w_offset = 6 # macro
SDMA_PKT_COPY_T2T_BC_DW_6_src_bank_w_mask = 0x00000003 # macro
SDMA_PKT_COPY_T2T_BC_DW_6_src_bank_w_shift = 15 # macro
def SDMA_PKT_COPY_T2T_BC_DW_6_SRC_BANK_W(x):  # macro
   return (((x)&0x00000003)<<15)
SDMA_PKT_COPY_T2T_BC_DW_6_src_bank_h_offset = 6 # macro
SDMA_PKT_COPY_T2T_BC_DW_6_src_bank_h_mask = 0x00000003 # macro
SDMA_PKT_COPY_T2T_BC_DW_6_src_bank_h_shift = 18 # macro
def SDMA_PKT_COPY_T2T_BC_DW_6_SRC_BANK_H(x):  # macro
   return (((x)&0x00000003)<<18)
SDMA_PKT_COPY_T2T_BC_DW_6_src_num_bank_offset = 6 # macro
SDMA_PKT_COPY_T2T_BC_DW_6_src_num_bank_mask = 0x00000003 # macro
SDMA_PKT_COPY_T2T_BC_DW_6_src_num_bank_shift = 21 # macro
def SDMA_PKT_COPY_T2T_BC_DW_6_SRC_NUM_BANK(x):  # macro
   return (((x)&0x00000003)<<21)
SDMA_PKT_COPY_T2T_BC_DW_6_src_mat_aspt_offset = 6 # macro
SDMA_PKT_COPY_T2T_BC_DW_6_src_mat_aspt_mask = 0x00000003 # macro
SDMA_PKT_COPY_T2T_BC_DW_6_src_mat_aspt_shift = 24 # macro
def SDMA_PKT_COPY_T2T_BC_DW_6_SRC_MAT_ASPT(x):  # macro
   return (((x)&0x00000003)<<24)
SDMA_PKT_COPY_T2T_BC_DW_6_src_pipe_config_offset = 6 # macro
SDMA_PKT_COPY_T2T_BC_DW_6_src_pipe_config_mask = 0x0000001F # macro
SDMA_PKT_COPY_T2T_BC_DW_6_src_pipe_config_shift = 26 # macro
def SDMA_PKT_COPY_T2T_BC_DW_6_SRC_PIPE_CONFIG(x):  # macro
   return (((x)&0x0000001F)<<26)
SDMA_PKT_COPY_T2T_BC_DST_ADDR_LO_dst_addr_31_0_offset = 7 # macro
SDMA_PKT_COPY_T2T_BC_DST_ADDR_LO_dst_addr_31_0_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_T2T_BC_DST_ADDR_LO_dst_addr_31_0_shift = 0 # macro
def SDMA_PKT_COPY_T2T_BC_DST_ADDR_LO_DST_ADDR_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COPY_T2T_BC_DST_ADDR_HI_dst_addr_63_32_offset = 8 # macro
SDMA_PKT_COPY_T2T_BC_DST_ADDR_HI_dst_addr_63_32_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_T2T_BC_DST_ADDR_HI_dst_addr_63_32_shift = 0 # macro
def SDMA_PKT_COPY_T2T_BC_DST_ADDR_HI_DST_ADDR_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COPY_T2T_BC_DW_9_dst_x_offset = 9 # macro
SDMA_PKT_COPY_T2T_BC_DW_9_dst_x_mask = 0x00003FFF # macro
SDMA_PKT_COPY_T2T_BC_DW_9_dst_x_shift = 0 # macro
def SDMA_PKT_COPY_T2T_BC_DW_9_DST_X(x):  # macro
   return (((x)&0x00003FFF)<<0)
SDMA_PKT_COPY_T2T_BC_DW_9_dst_y_offset = 9 # macro
SDMA_PKT_COPY_T2T_BC_DW_9_dst_y_mask = 0x00003FFF # macro
SDMA_PKT_COPY_T2T_BC_DW_9_dst_y_shift = 16 # macro
def SDMA_PKT_COPY_T2T_BC_DW_9_DST_Y(x):  # macro
   return (((x)&0x00003FFF)<<16)
SDMA_PKT_COPY_T2T_BC_DW_10_dst_z_offset = 10 # macro
SDMA_PKT_COPY_T2T_BC_DW_10_dst_z_mask = 0x000007FF # macro
SDMA_PKT_COPY_T2T_BC_DW_10_dst_z_shift = 0 # macro
def SDMA_PKT_COPY_T2T_BC_DW_10_DST_Z(x):  # macro
   return (((x)&0x000007FF)<<0)
SDMA_PKT_COPY_T2T_BC_DW_10_dst_width_offset = 10 # macro
SDMA_PKT_COPY_T2T_BC_DW_10_dst_width_mask = 0x00003FFF # macro
SDMA_PKT_COPY_T2T_BC_DW_10_dst_width_shift = 16 # macro
def SDMA_PKT_COPY_T2T_BC_DW_10_DST_WIDTH(x):  # macro
   return (((x)&0x00003FFF)<<16)
SDMA_PKT_COPY_T2T_BC_DW_11_dst_height_offset = 11 # macro
SDMA_PKT_COPY_T2T_BC_DW_11_dst_height_mask = 0x00003FFF # macro
SDMA_PKT_COPY_T2T_BC_DW_11_dst_height_shift = 0 # macro
def SDMA_PKT_COPY_T2T_BC_DW_11_DST_HEIGHT(x):  # macro
   return (((x)&0x00003FFF)<<0)
SDMA_PKT_COPY_T2T_BC_DW_11_dst_depth_offset = 11 # macro
SDMA_PKT_COPY_T2T_BC_DW_11_dst_depth_mask = 0x00000FFF # macro
SDMA_PKT_COPY_T2T_BC_DW_11_dst_depth_shift = 16 # macro
def SDMA_PKT_COPY_T2T_BC_DW_11_DST_DEPTH(x):  # macro
   return (((x)&0x00000FFF)<<16)
SDMA_PKT_COPY_T2T_BC_DW_12_dst_element_size_offset = 12 # macro
SDMA_PKT_COPY_T2T_BC_DW_12_dst_element_size_mask = 0x00000007 # macro
SDMA_PKT_COPY_T2T_BC_DW_12_dst_element_size_shift = 0 # macro
def SDMA_PKT_COPY_T2T_BC_DW_12_DST_ELEMENT_SIZE(x):  # macro
   return (((x)&0x00000007)<<0)
SDMA_PKT_COPY_T2T_BC_DW_12_dst_array_mode_offset = 12 # macro
SDMA_PKT_COPY_T2T_BC_DW_12_dst_array_mode_mask = 0x0000000F # macro
SDMA_PKT_COPY_T2T_BC_DW_12_dst_array_mode_shift = 3 # macro
def SDMA_PKT_COPY_T2T_BC_DW_12_DST_ARRAY_MODE(x):  # macro
   return (((x)&0x0000000F)<<3)
SDMA_PKT_COPY_T2T_BC_DW_12_dst_mit_mode_offset = 12 # macro
SDMA_PKT_COPY_T2T_BC_DW_12_dst_mit_mode_mask = 0x00000007 # macro
SDMA_PKT_COPY_T2T_BC_DW_12_dst_mit_mode_shift = 8 # macro
def SDMA_PKT_COPY_T2T_BC_DW_12_DST_MIT_MODE(x):  # macro
   return (((x)&0x00000007)<<8)
SDMA_PKT_COPY_T2T_BC_DW_12_dst_tilesplit_size_offset = 12 # macro
SDMA_PKT_COPY_T2T_BC_DW_12_dst_tilesplit_size_mask = 0x00000007 # macro
SDMA_PKT_COPY_T2T_BC_DW_12_dst_tilesplit_size_shift = 11 # macro
def SDMA_PKT_COPY_T2T_BC_DW_12_DST_TILESPLIT_SIZE(x):  # macro
   return (((x)&0x00000007)<<11)
SDMA_PKT_COPY_T2T_BC_DW_12_dst_bank_w_offset = 12 # macro
SDMA_PKT_COPY_T2T_BC_DW_12_dst_bank_w_mask = 0x00000003 # macro
SDMA_PKT_COPY_T2T_BC_DW_12_dst_bank_w_shift = 15 # macro
def SDMA_PKT_COPY_T2T_BC_DW_12_DST_BANK_W(x):  # macro
   return (((x)&0x00000003)<<15)
SDMA_PKT_COPY_T2T_BC_DW_12_dst_bank_h_offset = 12 # macro
SDMA_PKT_COPY_T2T_BC_DW_12_dst_bank_h_mask = 0x00000003 # macro
SDMA_PKT_COPY_T2T_BC_DW_12_dst_bank_h_shift = 18 # macro
def SDMA_PKT_COPY_T2T_BC_DW_12_DST_BANK_H(x):  # macro
   return (((x)&0x00000003)<<18)
SDMA_PKT_COPY_T2T_BC_DW_12_dst_num_bank_offset = 12 # macro
SDMA_PKT_COPY_T2T_BC_DW_12_dst_num_bank_mask = 0x00000003 # macro
SDMA_PKT_COPY_T2T_BC_DW_12_dst_num_bank_shift = 21 # macro
def SDMA_PKT_COPY_T2T_BC_DW_12_DST_NUM_BANK(x):  # macro
   return (((x)&0x00000003)<<21)
SDMA_PKT_COPY_T2T_BC_DW_12_dst_mat_aspt_offset = 12 # macro
SDMA_PKT_COPY_T2T_BC_DW_12_dst_mat_aspt_mask = 0x00000003 # macro
SDMA_PKT_COPY_T2T_BC_DW_12_dst_mat_aspt_shift = 24 # macro
def SDMA_PKT_COPY_T2T_BC_DW_12_DST_MAT_ASPT(x):  # macro
   return (((x)&0x00000003)<<24)
SDMA_PKT_COPY_T2T_BC_DW_12_dst_pipe_config_offset = 12 # macro
SDMA_PKT_COPY_T2T_BC_DW_12_dst_pipe_config_mask = 0x0000001F # macro
SDMA_PKT_COPY_T2T_BC_DW_12_dst_pipe_config_shift = 26 # macro
def SDMA_PKT_COPY_T2T_BC_DW_12_DST_PIPE_CONFIG(x):  # macro
   return (((x)&0x0000001F)<<26)
SDMA_PKT_COPY_T2T_BC_DW_13_rect_x_offset = 13 # macro
SDMA_PKT_COPY_T2T_BC_DW_13_rect_x_mask = 0x00003FFF # macro
SDMA_PKT_COPY_T2T_BC_DW_13_rect_x_shift = 0 # macro
def SDMA_PKT_COPY_T2T_BC_DW_13_RECT_X(x):  # macro
   return (((x)&0x00003FFF)<<0)
SDMA_PKT_COPY_T2T_BC_DW_13_rect_y_offset = 13 # macro
SDMA_PKT_COPY_T2T_BC_DW_13_rect_y_mask = 0x00003FFF # macro
SDMA_PKT_COPY_T2T_BC_DW_13_rect_y_shift = 16 # macro
def SDMA_PKT_COPY_T2T_BC_DW_13_RECT_Y(x):  # macro
   return (((x)&0x00003FFF)<<16)
SDMA_PKT_COPY_T2T_BC_DW_14_rect_z_offset = 14 # macro
SDMA_PKT_COPY_T2T_BC_DW_14_rect_z_mask = 0x000007FF # macro
SDMA_PKT_COPY_T2T_BC_DW_14_rect_z_shift = 0 # macro
def SDMA_PKT_COPY_T2T_BC_DW_14_RECT_Z(x):  # macro
   return (((x)&0x000007FF)<<0)
SDMA_PKT_COPY_T2T_BC_DW_14_dst_sw_offset = 14 # macro
SDMA_PKT_COPY_T2T_BC_DW_14_dst_sw_mask = 0x00000003 # macro
SDMA_PKT_COPY_T2T_BC_DW_14_dst_sw_shift = 16 # macro
def SDMA_PKT_COPY_T2T_BC_DW_14_DST_SW(x):  # macro
   return (((x)&0x00000003)<<16)
SDMA_PKT_COPY_T2T_BC_DW_14_src_sw_offset = 14 # macro
SDMA_PKT_COPY_T2T_BC_DW_14_src_sw_mask = 0x00000003 # macro
SDMA_PKT_COPY_T2T_BC_DW_14_src_sw_shift = 24 # macro
def SDMA_PKT_COPY_T2T_BC_DW_14_SRC_SW(x):  # macro
   return (((x)&0x00000003)<<24)
SDMA_PKT_COPY_TILED_SUBWIN_HEADER_op_offset = 0 # macro
SDMA_PKT_COPY_TILED_SUBWIN_HEADER_op_mask = 0x000000FF # macro
SDMA_PKT_COPY_TILED_SUBWIN_HEADER_op_shift = 0 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_HEADER_OP(x):  # macro
   return (((x)&0x000000FF)<<0)
SDMA_PKT_COPY_TILED_SUBWIN_HEADER_sub_op_offset = 0 # macro
SDMA_PKT_COPY_TILED_SUBWIN_HEADER_sub_op_mask = 0x000000FF # macro
SDMA_PKT_COPY_TILED_SUBWIN_HEADER_sub_op_shift = 8 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_HEADER_SUB_OP(x):  # macro
   return (((x)&0x000000FF)<<8)
SDMA_PKT_COPY_TILED_SUBWIN_HEADER_tmz_offset = 0 # macro
SDMA_PKT_COPY_TILED_SUBWIN_HEADER_tmz_mask = 0x00000001 # macro
SDMA_PKT_COPY_TILED_SUBWIN_HEADER_tmz_shift = 18 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_HEADER_TMZ(x):  # macro
   return (((x)&0x00000001)<<18)
SDMA_PKT_COPY_TILED_SUBWIN_HEADER_dcc_offset = 0 # macro
SDMA_PKT_COPY_TILED_SUBWIN_HEADER_dcc_mask = 0x00000001 # macro
SDMA_PKT_COPY_TILED_SUBWIN_HEADER_dcc_shift = 19 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_HEADER_DCC(x):  # macro
   return (((x)&0x00000001)<<19)
SDMA_PKT_COPY_TILED_SUBWIN_HEADER_detile_offset = 0 # macro
SDMA_PKT_COPY_TILED_SUBWIN_HEADER_detile_mask = 0x00000001 # macro
SDMA_PKT_COPY_TILED_SUBWIN_HEADER_detile_shift = 31 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_HEADER_DETILE(x):  # macro
   return (((x)&0x00000001)<<31)
SDMA_PKT_COPY_TILED_SUBWIN_TILED_ADDR_LO_tiled_addr_31_0_offset = 1 # macro
SDMA_PKT_COPY_TILED_SUBWIN_TILED_ADDR_LO_tiled_addr_31_0_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_TILED_SUBWIN_TILED_ADDR_LO_tiled_addr_31_0_shift = 0 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_TILED_ADDR_LO_TILED_ADDR_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COPY_TILED_SUBWIN_TILED_ADDR_HI_tiled_addr_63_32_offset = 2 # macro
SDMA_PKT_COPY_TILED_SUBWIN_TILED_ADDR_HI_tiled_addr_63_32_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_TILED_SUBWIN_TILED_ADDR_HI_tiled_addr_63_32_shift = 0 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_TILED_ADDR_HI_TILED_ADDR_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COPY_TILED_SUBWIN_DW_3_tiled_x_offset = 3 # macro
SDMA_PKT_COPY_TILED_SUBWIN_DW_3_tiled_x_mask = 0x00003FFF # macro
SDMA_PKT_COPY_TILED_SUBWIN_DW_3_tiled_x_shift = 0 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_DW_3_TILED_X(x):  # macro
   return (((x)&0x00003FFF)<<0)
SDMA_PKT_COPY_TILED_SUBWIN_DW_3_tiled_y_offset = 3 # macro
SDMA_PKT_COPY_TILED_SUBWIN_DW_3_tiled_y_mask = 0x00003FFF # macro
SDMA_PKT_COPY_TILED_SUBWIN_DW_3_tiled_y_shift = 16 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_DW_3_TILED_Y(x):  # macro
   return (((x)&0x00003FFF)<<16)
SDMA_PKT_COPY_TILED_SUBWIN_DW_4_tiled_z_offset = 4 # macro
SDMA_PKT_COPY_TILED_SUBWIN_DW_4_tiled_z_mask = 0x00001FFF # macro
SDMA_PKT_COPY_TILED_SUBWIN_DW_4_tiled_z_shift = 0 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_DW_4_TILED_Z(x):  # macro
   return (((x)&0x00001FFF)<<0)
SDMA_PKT_COPY_TILED_SUBWIN_DW_4_width_offset = 4 # macro
SDMA_PKT_COPY_TILED_SUBWIN_DW_4_width_mask = 0x00003FFF # macro
SDMA_PKT_COPY_TILED_SUBWIN_DW_4_width_shift = 16 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_DW_4_WIDTH(x):  # macro
   return (((x)&0x00003FFF)<<16)
SDMA_PKT_COPY_TILED_SUBWIN_DW_5_height_offset = 5 # macro
SDMA_PKT_COPY_TILED_SUBWIN_DW_5_height_mask = 0x00003FFF # macro
SDMA_PKT_COPY_TILED_SUBWIN_DW_5_height_shift = 0 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_DW_5_HEIGHT(x):  # macro
   return (((x)&0x00003FFF)<<0)
SDMA_PKT_COPY_TILED_SUBWIN_DW_5_depth_offset = 5 # macro
SDMA_PKT_COPY_TILED_SUBWIN_DW_5_depth_mask = 0x00001FFF # macro
SDMA_PKT_COPY_TILED_SUBWIN_DW_5_depth_shift = 16 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_DW_5_DEPTH(x):  # macro
   return (((x)&0x00001FFF)<<16)
SDMA_PKT_COPY_TILED_SUBWIN_DW_6_element_size_offset = 6 # macro
SDMA_PKT_COPY_TILED_SUBWIN_DW_6_element_size_mask = 0x00000007 # macro
SDMA_PKT_COPY_TILED_SUBWIN_DW_6_element_size_shift = 0 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_DW_6_ELEMENT_SIZE(x):  # macro
   return (((x)&0x00000007)<<0)
SDMA_PKT_COPY_TILED_SUBWIN_DW_6_swizzle_mode_offset = 6 # macro
SDMA_PKT_COPY_TILED_SUBWIN_DW_6_swizzle_mode_mask = 0x0000001F # macro
SDMA_PKT_COPY_TILED_SUBWIN_DW_6_swizzle_mode_shift = 3 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_DW_6_SWIZZLE_MODE(x):  # macro
   return (((x)&0x0000001F)<<3)
SDMA_PKT_COPY_TILED_SUBWIN_DW_6_dimension_offset = 6 # macro
SDMA_PKT_COPY_TILED_SUBWIN_DW_6_dimension_mask = 0x00000003 # macro
SDMA_PKT_COPY_TILED_SUBWIN_DW_6_dimension_shift = 9 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_DW_6_DIMENSION(x):  # macro
   return (((x)&0x00000003)<<9)
SDMA_PKT_COPY_TILED_SUBWIN_DW_6_mip_max_offset = 6 # macro
SDMA_PKT_COPY_TILED_SUBWIN_DW_6_mip_max_mask = 0x0000000F # macro
SDMA_PKT_COPY_TILED_SUBWIN_DW_6_mip_max_shift = 16 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_DW_6_MIP_MAX(x):  # macro
   return (((x)&0x0000000F)<<16)
SDMA_PKT_COPY_TILED_SUBWIN_DW_6_mip_id_offset = 6 # macro
SDMA_PKT_COPY_TILED_SUBWIN_DW_6_mip_id_mask = 0x0000000F # macro
SDMA_PKT_COPY_TILED_SUBWIN_DW_6_mip_id_shift = 20 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_DW_6_MIP_ID(x):  # macro
   return (((x)&0x0000000F)<<20)
SDMA_PKT_COPY_TILED_SUBWIN_LINEAR_ADDR_LO_linear_addr_31_0_offset = 7 # macro
SDMA_PKT_COPY_TILED_SUBWIN_LINEAR_ADDR_LO_linear_addr_31_0_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_TILED_SUBWIN_LINEAR_ADDR_LO_linear_addr_31_0_shift = 0 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_LINEAR_ADDR_LO_LINEAR_ADDR_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COPY_TILED_SUBWIN_LINEAR_ADDR_HI_linear_addr_63_32_offset = 8 # macro
SDMA_PKT_COPY_TILED_SUBWIN_LINEAR_ADDR_HI_linear_addr_63_32_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_TILED_SUBWIN_LINEAR_ADDR_HI_linear_addr_63_32_shift = 0 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_LINEAR_ADDR_HI_LINEAR_ADDR_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COPY_TILED_SUBWIN_DW_9_linear_x_offset = 9 # macro
SDMA_PKT_COPY_TILED_SUBWIN_DW_9_linear_x_mask = 0x00003FFF # macro
SDMA_PKT_COPY_TILED_SUBWIN_DW_9_linear_x_shift = 0 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_DW_9_LINEAR_X(x):  # macro
   return (((x)&0x00003FFF)<<0)
SDMA_PKT_COPY_TILED_SUBWIN_DW_9_linear_y_offset = 9 # macro
SDMA_PKT_COPY_TILED_SUBWIN_DW_9_linear_y_mask = 0x00003FFF # macro
SDMA_PKT_COPY_TILED_SUBWIN_DW_9_linear_y_shift = 16 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_DW_9_LINEAR_Y(x):  # macro
   return (((x)&0x00003FFF)<<16)
SDMA_PKT_COPY_TILED_SUBWIN_DW_10_linear_z_offset = 10 # macro
SDMA_PKT_COPY_TILED_SUBWIN_DW_10_linear_z_mask = 0x00001FFF # macro
SDMA_PKT_COPY_TILED_SUBWIN_DW_10_linear_z_shift = 0 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_DW_10_LINEAR_Z(x):  # macro
   return (((x)&0x00001FFF)<<0)
SDMA_PKT_COPY_TILED_SUBWIN_DW_10_linear_pitch_offset = 10 # macro
SDMA_PKT_COPY_TILED_SUBWIN_DW_10_linear_pitch_mask = 0x00003FFF # macro
SDMA_PKT_COPY_TILED_SUBWIN_DW_10_linear_pitch_shift = 16 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_DW_10_LINEAR_PITCH(x):  # macro
   return (((x)&0x00003FFF)<<16)
SDMA_PKT_COPY_TILED_SUBWIN_DW_11_linear_slice_pitch_offset = 11 # macro
SDMA_PKT_COPY_TILED_SUBWIN_DW_11_linear_slice_pitch_mask = 0x0FFFFFFF # macro
SDMA_PKT_COPY_TILED_SUBWIN_DW_11_linear_slice_pitch_shift = 0 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_DW_11_LINEAR_SLICE_PITCH(x):  # macro
   return (((x)&0x0FFFFFFF)<<0)
SDMA_PKT_COPY_TILED_SUBWIN_DW_12_rect_x_offset = 12 # macro
SDMA_PKT_COPY_TILED_SUBWIN_DW_12_rect_x_mask = 0x00003FFF # macro
SDMA_PKT_COPY_TILED_SUBWIN_DW_12_rect_x_shift = 0 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_DW_12_RECT_X(x):  # macro
   return (((x)&0x00003FFF)<<0)
SDMA_PKT_COPY_TILED_SUBWIN_DW_12_rect_y_offset = 12 # macro
SDMA_PKT_COPY_TILED_SUBWIN_DW_12_rect_y_mask = 0x00003FFF # macro
SDMA_PKT_COPY_TILED_SUBWIN_DW_12_rect_y_shift = 16 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_DW_12_RECT_Y(x):  # macro
   return (((x)&0x00003FFF)<<16)
SDMA_PKT_COPY_TILED_SUBWIN_DW_13_rect_z_offset = 13 # macro
SDMA_PKT_COPY_TILED_SUBWIN_DW_13_rect_z_mask = 0x00001FFF # macro
SDMA_PKT_COPY_TILED_SUBWIN_DW_13_rect_z_shift = 0 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_DW_13_RECT_Z(x):  # macro
   return (((x)&0x00001FFF)<<0)
SDMA_PKT_COPY_TILED_SUBWIN_DW_13_linear_sw_offset = 13 # macro
SDMA_PKT_COPY_TILED_SUBWIN_DW_13_linear_sw_mask = 0x00000003 # macro
SDMA_PKT_COPY_TILED_SUBWIN_DW_13_linear_sw_shift = 16 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_DW_13_LINEAR_SW(x):  # macro
   return (((x)&0x00000003)<<16)
SDMA_PKT_COPY_TILED_SUBWIN_DW_13_tile_sw_offset = 13 # macro
SDMA_PKT_COPY_TILED_SUBWIN_DW_13_tile_sw_mask = 0x00000003 # macro
SDMA_PKT_COPY_TILED_SUBWIN_DW_13_tile_sw_shift = 24 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_DW_13_TILE_SW(x):  # macro
   return (((x)&0x00000003)<<24)
SDMA_PKT_COPY_TILED_SUBWIN_META_ADDR_LO_meta_addr_31_0_offset = 14 # macro
SDMA_PKT_COPY_TILED_SUBWIN_META_ADDR_LO_meta_addr_31_0_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_TILED_SUBWIN_META_ADDR_LO_meta_addr_31_0_shift = 0 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_META_ADDR_LO_META_ADDR_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COPY_TILED_SUBWIN_META_ADDR_HI_meta_addr_63_32_offset = 15 # macro
SDMA_PKT_COPY_TILED_SUBWIN_META_ADDR_HI_meta_addr_63_32_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_TILED_SUBWIN_META_ADDR_HI_meta_addr_63_32_shift = 0 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_META_ADDR_HI_META_ADDR_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_data_format_offset = 16 # macro
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_data_format_mask = 0x0000007F # macro
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_data_format_shift = 0 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_DATA_FORMAT(x):  # macro
   return (((x)&0x0000007F)<<0)
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_color_transform_disable_offset = 16 # macro
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_color_transform_disable_mask = 0x00000001 # macro
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_color_transform_disable_shift = 7 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_COLOR_TRANSFORM_DISABLE(x):  # macro
   return (((x)&0x00000001)<<7)
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_alpha_is_on_msb_offset = 16 # macro
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_alpha_is_on_msb_mask = 0x00000001 # macro
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_alpha_is_on_msb_shift = 8 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_ALPHA_IS_ON_MSB(x):  # macro
   return (((x)&0x00000001)<<8)
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_number_type_offset = 16 # macro
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_number_type_mask = 0x00000007 # macro
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_number_type_shift = 9 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_NUMBER_TYPE(x):  # macro
   return (((x)&0x00000007)<<9)
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_surface_type_offset = 16 # macro
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_surface_type_mask = 0x00000003 # macro
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_surface_type_shift = 12 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_SURFACE_TYPE(x):  # macro
   return (((x)&0x00000003)<<12)
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_max_comp_block_size_offset = 16 # macro
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_max_comp_block_size_mask = 0x00000003 # macro
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_max_comp_block_size_shift = 24 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_MAX_COMP_BLOCK_SIZE(x):  # macro
   return (((x)&0x00000003)<<24)
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_max_uncomp_block_size_offset = 16 # macro
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_max_uncomp_block_size_mask = 0x00000003 # macro
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_max_uncomp_block_size_shift = 26 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_MAX_UNCOMP_BLOCK_SIZE(x):  # macro
   return (((x)&0x00000003)<<26)
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_write_compress_enable_offset = 16 # macro
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_write_compress_enable_mask = 0x00000001 # macro
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_write_compress_enable_shift = 28 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_WRITE_COMPRESS_ENABLE(x):  # macro
   return (((x)&0x00000001)<<28)
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_meta_tmz_offset = 16 # macro
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_meta_tmz_mask = 0x00000001 # macro
SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_meta_tmz_shift = 29 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_META_TMZ(x):  # macro
   return (((x)&0x00000001)<<29)
SDMA_PKT_COPY_TILED_SUBWIN_BC_HEADER_op_offset = 0 # macro
SDMA_PKT_COPY_TILED_SUBWIN_BC_HEADER_op_mask = 0x000000FF # macro
SDMA_PKT_COPY_TILED_SUBWIN_BC_HEADER_op_shift = 0 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_BC_HEADER_OP(x):  # macro
   return (((x)&0x000000FF)<<0)
SDMA_PKT_COPY_TILED_SUBWIN_BC_HEADER_sub_op_offset = 0 # macro
SDMA_PKT_COPY_TILED_SUBWIN_BC_HEADER_sub_op_mask = 0x000000FF # macro
SDMA_PKT_COPY_TILED_SUBWIN_BC_HEADER_sub_op_shift = 8 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_BC_HEADER_SUB_OP(x):  # macro
   return (((x)&0x000000FF)<<8)
SDMA_PKT_COPY_TILED_SUBWIN_BC_HEADER_detile_offset = 0 # macro
SDMA_PKT_COPY_TILED_SUBWIN_BC_HEADER_detile_mask = 0x00000001 # macro
SDMA_PKT_COPY_TILED_SUBWIN_BC_HEADER_detile_shift = 31 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_BC_HEADER_DETILE(x):  # macro
   return (((x)&0x00000001)<<31)
SDMA_PKT_COPY_TILED_SUBWIN_BC_TILED_ADDR_LO_tiled_addr_31_0_offset = 1 # macro
SDMA_PKT_COPY_TILED_SUBWIN_BC_TILED_ADDR_LO_tiled_addr_31_0_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_TILED_SUBWIN_BC_TILED_ADDR_LO_tiled_addr_31_0_shift = 0 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_BC_TILED_ADDR_LO_TILED_ADDR_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COPY_TILED_SUBWIN_BC_TILED_ADDR_HI_tiled_addr_63_32_offset = 2 # macro
SDMA_PKT_COPY_TILED_SUBWIN_BC_TILED_ADDR_HI_tiled_addr_63_32_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_TILED_SUBWIN_BC_TILED_ADDR_HI_tiled_addr_63_32_shift = 0 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_BC_TILED_ADDR_HI_TILED_ADDR_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_3_tiled_x_offset = 3 # macro
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_3_tiled_x_mask = 0x00003FFF # macro
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_3_tiled_x_shift = 0 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_3_TILED_X(x):  # macro
   return (((x)&0x00003FFF)<<0)
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_3_tiled_y_offset = 3 # macro
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_3_tiled_y_mask = 0x00003FFF # macro
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_3_tiled_y_shift = 16 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_3_TILED_Y(x):  # macro
   return (((x)&0x00003FFF)<<16)
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_4_tiled_z_offset = 4 # macro
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_4_tiled_z_mask = 0x000007FF # macro
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_4_tiled_z_shift = 0 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_4_TILED_Z(x):  # macro
   return (((x)&0x000007FF)<<0)
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_4_width_offset = 4 # macro
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_4_width_mask = 0x00003FFF # macro
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_4_width_shift = 16 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_4_WIDTH(x):  # macro
   return (((x)&0x00003FFF)<<16)
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_5_height_offset = 5 # macro
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_5_height_mask = 0x00003FFF # macro
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_5_height_shift = 0 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_5_HEIGHT(x):  # macro
   return (((x)&0x00003FFF)<<0)
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_5_depth_offset = 5 # macro
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_5_depth_mask = 0x000007FF # macro
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_5_depth_shift = 16 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_5_DEPTH(x):  # macro
   return (((x)&0x000007FF)<<16)
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_element_size_offset = 6 # macro
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_element_size_mask = 0x00000007 # macro
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_element_size_shift = 0 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_ELEMENT_SIZE(x):  # macro
   return (((x)&0x00000007)<<0)
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_array_mode_offset = 6 # macro
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_array_mode_mask = 0x0000000F # macro
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_array_mode_shift = 3 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_ARRAY_MODE(x):  # macro
   return (((x)&0x0000000F)<<3)
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_mit_mode_offset = 6 # macro
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_mit_mode_mask = 0x00000007 # macro
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_mit_mode_shift = 8 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_MIT_MODE(x):  # macro
   return (((x)&0x00000007)<<8)
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_tilesplit_size_offset = 6 # macro
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_tilesplit_size_mask = 0x00000007 # macro
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_tilesplit_size_shift = 11 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_TILESPLIT_SIZE(x):  # macro
   return (((x)&0x00000007)<<11)
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_bank_w_offset = 6 # macro
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_bank_w_mask = 0x00000003 # macro
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_bank_w_shift = 15 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_BANK_W(x):  # macro
   return (((x)&0x00000003)<<15)
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_bank_h_offset = 6 # macro
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_bank_h_mask = 0x00000003 # macro
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_bank_h_shift = 18 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_BANK_H(x):  # macro
   return (((x)&0x00000003)<<18)
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_num_bank_offset = 6 # macro
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_num_bank_mask = 0x00000003 # macro
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_num_bank_shift = 21 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_NUM_BANK(x):  # macro
   return (((x)&0x00000003)<<21)
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_mat_aspt_offset = 6 # macro
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_mat_aspt_mask = 0x00000003 # macro
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_mat_aspt_shift = 24 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_MAT_ASPT(x):  # macro
   return ((x&0x00000003)<<24)
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_pipe_config_offset = 6 # macro
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_pipe_config_mask = 0x0000001F # macro
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_pipe_config_shift = 26 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_PIPE_CONFIG(x):  # macro
   return (((x)&0x0000001F)<<26)
SDMA_PKT_COPY_TILED_SUBWIN_BC_LINEAR_ADDR_LO_linear_addr_31_0_offset = 7 # macro
SDMA_PKT_COPY_TILED_SUBWIN_BC_LINEAR_ADDR_LO_linear_addr_31_0_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_TILED_SUBWIN_BC_LINEAR_ADDR_LO_linear_addr_31_0_shift = 0 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_BC_LINEAR_ADDR_LO_LINEAR_ADDR_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COPY_TILED_SUBWIN_BC_LINEAR_ADDR_HI_linear_addr_63_32_offset = 8 # macro
SDMA_PKT_COPY_TILED_SUBWIN_BC_LINEAR_ADDR_HI_linear_addr_63_32_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_TILED_SUBWIN_BC_LINEAR_ADDR_HI_linear_addr_63_32_shift = 0 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_BC_LINEAR_ADDR_HI_LINEAR_ADDR_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_9_linear_x_offset = 9 # macro
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_9_linear_x_mask = 0x00003FFF # macro
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_9_linear_x_shift = 0 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_9_LINEAR_X(x):  # macro
   return (((x)&0x00003FFF)<<0)
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_9_linear_y_offset = 9 # macro
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_9_linear_y_mask = 0x00003FFF # macro
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_9_linear_y_shift = 16 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_9_LINEAR_Y(x):  # macro
   return (((x)&0x00003FFF)<<16)
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_10_linear_z_offset = 10 # macro
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_10_linear_z_mask = 0x000007FF # macro
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_10_linear_z_shift = 0 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_10_LINEAR_Z(x):  # macro
   return (((x)&0x000007FF)<<0)
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_10_linear_pitch_offset = 10 # macro
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_10_linear_pitch_mask = 0x00003FFF # macro
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_10_linear_pitch_shift = 16 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_10_LINEAR_PITCH(x):  # macro
   return (((x)&0x00003FFF)<<16)
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_11_linear_slice_pitch_offset = 11 # macro
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_11_linear_slice_pitch_mask = 0x0FFFFFFF # macro
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_11_linear_slice_pitch_shift = 0 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_11_LINEAR_SLICE_PITCH(x):  # macro
   return (((x)&0x0FFFFFFF)<<0)
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_12_rect_x_offset = 12 # macro
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_12_rect_x_mask = 0x00003FFF # macro
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_12_rect_x_shift = 0 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_12_RECT_X(x):  # macro
   return (((x)&0x00003FFF)<<0)
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_12_rect_y_offset = 12 # macro
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_12_rect_y_mask = 0x00003FFF # macro
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_12_rect_y_shift = 16 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_12_RECT_Y(x):  # macro
   return (((x)&0x00003FFF)<<16)
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_13_rect_z_offset = 13 # macro
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_13_rect_z_mask = 0x000007FF # macro
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_13_rect_z_shift = 0 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_13_RECT_Z(x):  # macro
   return (((x)&0x000007FF)<<0)
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_13_linear_sw_offset = 13 # macro
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_13_linear_sw_mask = 0x00000003 # macro
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_13_linear_sw_shift = 16 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_13_LINEAR_SW(x):  # macro
   return (((x)&0x00000003)<<16)
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_13_tile_sw_offset = 13 # macro
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_13_tile_sw_mask = 0x00000003 # macro
SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_13_tile_sw_shift = 24 # macro
def SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_13_TILE_SW(x):  # macro
   return (((x)&0x00000003)<<24)
SDMA_PKT_COPY_STRUCT_HEADER_op_offset = 0 # macro
SDMA_PKT_COPY_STRUCT_HEADER_op_mask = 0x000000FF # macro
SDMA_PKT_COPY_STRUCT_HEADER_op_shift = 0 # macro
def SDMA_PKT_COPY_STRUCT_HEADER_OP(x):  # macro
   return (((x)&0x000000FF)<<0)
SDMA_PKT_COPY_STRUCT_HEADER_sub_op_offset = 0 # macro
SDMA_PKT_COPY_STRUCT_HEADER_sub_op_mask = 0x000000FF # macro
SDMA_PKT_COPY_STRUCT_HEADER_sub_op_shift = 8 # macro
def SDMA_PKT_COPY_STRUCT_HEADER_SUB_OP(x):  # macro
   return (((x)&0x000000FF)<<8)
SDMA_PKT_COPY_STRUCT_HEADER_tmz_offset = 0 # macro
SDMA_PKT_COPY_STRUCT_HEADER_tmz_mask = 0x00000001 # macro
SDMA_PKT_COPY_STRUCT_HEADER_tmz_shift = 18 # macro
def SDMA_PKT_COPY_STRUCT_HEADER_TMZ(x):  # macro
   return (((x)&0x00000001)<<18)
SDMA_PKT_COPY_STRUCT_HEADER_detile_offset = 0 # macro
SDMA_PKT_COPY_STRUCT_HEADER_detile_mask = 0x00000001 # macro
SDMA_PKT_COPY_STRUCT_HEADER_detile_shift = 31 # macro
def SDMA_PKT_COPY_STRUCT_HEADER_DETILE(x):  # macro
   return (((x)&0x00000001)<<31)
SDMA_PKT_COPY_STRUCT_SB_ADDR_LO_sb_addr_31_0_offset = 1 # macro
SDMA_PKT_COPY_STRUCT_SB_ADDR_LO_sb_addr_31_0_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_STRUCT_SB_ADDR_LO_sb_addr_31_0_shift = 0 # macro
def SDMA_PKT_COPY_STRUCT_SB_ADDR_LO_SB_ADDR_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COPY_STRUCT_SB_ADDR_HI_sb_addr_63_32_offset = 2 # macro
SDMA_PKT_COPY_STRUCT_SB_ADDR_HI_sb_addr_63_32_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_STRUCT_SB_ADDR_HI_sb_addr_63_32_shift = 0 # macro
def SDMA_PKT_COPY_STRUCT_SB_ADDR_HI_SB_ADDR_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COPY_STRUCT_START_INDEX_start_index_offset = 3 # macro
SDMA_PKT_COPY_STRUCT_START_INDEX_start_index_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_STRUCT_START_INDEX_start_index_shift = 0 # macro
def SDMA_PKT_COPY_STRUCT_START_INDEX_START_INDEX(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COPY_STRUCT_COUNT_count_offset = 4 # macro
SDMA_PKT_COPY_STRUCT_COUNT_count_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_STRUCT_COUNT_count_shift = 0 # macro
def SDMA_PKT_COPY_STRUCT_COUNT_COUNT(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COPY_STRUCT_DW_5_stride_offset = 5 # macro
SDMA_PKT_COPY_STRUCT_DW_5_stride_mask = 0x000007FF # macro
SDMA_PKT_COPY_STRUCT_DW_5_stride_shift = 0 # macro
def SDMA_PKT_COPY_STRUCT_DW_5_STRIDE(x):  # macro
   return (((x)&0x000007FF)<<0)
SDMA_PKT_COPY_STRUCT_DW_5_linear_sw_offset = 5 # macro
SDMA_PKT_COPY_STRUCT_DW_5_linear_sw_mask = 0x00000003 # macro
SDMA_PKT_COPY_STRUCT_DW_5_linear_sw_shift = 16 # macro
def SDMA_PKT_COPY_STRUCT_DW_5_LINEAR_SW(x):  # macro
   return (((x)&0x00000003)<<16)
SDMA_PKT_COPY_STRUCT_DW_5_struct_sw_offset = 5 # macro
SDMA_PKT_COPY_STRUCT_DW_5_struct_sw_mask = 0x00000003 # macro
SDMA_PKT_COPY_STRUCT_DW_5_struct_sw_shift = 24 # macro
def SDMA_PKT_COPY_STRUCT_DW_5_STRUCT_SW(x):  # macro
   return (((x)&0x00000003)<<24)
SDMA_PKT_COPY_STRUCT_LINEAR_ADDR_LO_linear_addr_31_0_offset = 6 # macro
SDMA_PKT_COPY_STRUCT_LINEAR_ADDR_LO_linear_addr_31_0_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_STRUCT_LINEAR_ADDR_LO_linear_addr_31_0_shift = 0 # macro
def SDMA_PKT_COPY_STRUCT_LINEAR_ADDR_LO_LINEAR_ADDR_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COPY_STRUCT_LINEAR_ADDR_HI_linear_addr_63_32_offset = 7 # macro
SDMA_PKT_COPY_STRUCT_LINEAR_ADDR_HI_linear_addr_63_32_mask = 0xFFFFFFFF # macro
SDMA_PKT_COPY_STRUCT_LINEAR_ADDR_HI_linear_addr_63_32_shift = 0 # macro
def SDMA_PKT_COPY_STRUCT_LINEAR_ADDR_HI_LINEAR_ADDR_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_WRITE_UNTILED_HEADER_op_offset = 0 # macro
SDMA_PKT_WRITE_UNTILED_HEADER_op_mask = 0x000000FF # macro
SDMA_PKT_WRITE_UNTILED_HEADER_op_shift = 0 # macro
def SDMA_PKT_WRITE_UNTILED_HEADER_OP(x):  # macro
   return (((x)&0x000000FF)<<0)
SDMA_PKT_WRITE_UNTILED_HEADER_sub_op_offset = 0 # macro
SDMA_PKT_WRITE_UNTILED_HEADER_sub_op_mask = 0x000000FF # macro
SDMA_PKT_WRITE_UNTILED_HEADER_sub_op_shift = 8 # macro
def SDMA_PKT_WRITE_UNTILED_HEADER_SUB_OP(x):  # macro
   return (((x)&0x000000FF)<<8)
SDMA_PKT_WRITE_UNTILED_HEADER_encrypt_offset = 0 # macro
SDMA_PKT_WRITE_UNTILED_HEADER_encrypt_mask = 0x00000001 # macro
SDMA_PKT_WRITE_UNTILED_HEADER_encrypt_shift = 16 # macro
def SDMA_PKT_WRITE_UNTILED_HEADER_ENCRYPT(x):  # macro
   return (((x)&0x00000001)<<16)
SDMA_PKT_WRITE_UNTILED_HEADER_tmz_offset = 0 # macro
SDMA_PKT_WRITE_UNTILED_HEADER_tmz_mask = 0x00000001 # macro
SDMA_PKT_WRITE_UNTILED_HEADER_tmz_shift = 18 # macro
def SDMA_PKT_WRITE_UNTILED_HEADER_TMZ(x):  # macro
   return (((x)&0x00000001)<<18)
SDMA_PKT_WRITE_UNTILED_DST_ADDR_LO_dst_addr_31_0_offset = 1 # macro
SDMA_PKT_WRITE_UNTILED_DST_ADDR_LO_dst_addr_31_0_mask = 0xFFFFFFFF # macro
SDMA_PKT_WRITE_UNTILED_DST_ADDR_LO_dst_addr_31_0_shift = 0 # macro
def SDMA_PKT_WRITE_UNTILED_DST_ADDR_LO_DST_ADDR_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_WRITE_UNTILED_DST_ADDR_HI_dst_addr_63_32_offset = 2 # macro
SDMA_PKT_WRITE_UNTILED_DST_ADDR_HI_dst_addr_63_32_mask = 0xFFFFFFFF # macro
SDMA_PKT_WRITE_UNTILED_DST_ADDR_HI_dst_addr_63_32_shift = 0 # macro
def SDMA_PKT_WRITE_UNTILED_DST_ADDR_HI_DST_ADDR_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_WRITE_UNTILED_DW_3_count_offset = 3 # macro
SDMA_PKT_WRITE_UNTILED_DW_3_count_mask = 0x000FFFFF # macro
SDMA_PKT_WRITE_UNTILED_DW_3_count_shift = 0 # macro
def SDMA_PKT_WRITE_UNTILED_DW_3_COUNT(x):  # macro
   return (((x)&0x000FFFFF)<<0)
SDMA_PKT_WRITE_UNTILED_DW_3_sw_offset = 3 # macro
SDMA_PKT_WRITE_UNTILED_DW_3_sw_mask = 0x00000003 # macro
SDMA_PKT_WRITE_UNTILED_DW_3_sw_shift = 24 # macro
def SDMA_PKT_WRITE_UNTILED_DW_3_SW(x):  # macro
   return (((x)&0x00000003)<<24)
SDMA_PKT_WRITE_UNTILED_DATA0_data0_offset = 4 # macro
SDMA_PKT_WRITE_UNTILED_DATA0_data0_mask = 0xFFFFFFFF # macro
SDMA_PKT_WRITE_UNTILED_DATA0_data0_shift = 0 # macro
def SDMA_PKT_WRITE_UNTILED_DATA0_DATA0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_WRITE_TILED_HEADER_op_offset = 0 # macro
SDMA_PKT_WRITE_TILED_HEADER_op_mask = 0x000000FF # macro
SDMA_PKT_WRITE_TILED_HEADER_op_shift = 0 # macro
def SDMA_PKT_WRITE_TILED_HEADER_OP(x):  # macro
   return (((x)&0x000000FF)<<0)
SDMA_PKT_WRITE_TILED_HEADER_sub_op_offset = 0 # macro
SDMA_PKT_WRITE_TILED_HEADER_sub_op_mask = 0x000000FF # macro
SDMA_PKT_WRITE_TILED_HEADER_sub_op_shift = 8 # macro
def SDMA_PKT_WRITE_TILED_HEADER_SUB_OP(x):  # macro
   return (((x)&0x000000FF)<<8)
SDMA_PKT_WRITE_TILED_HEADER_encrypt_offset = 0 # macro
SDMA_PKT_WRITE_TILED_HEADER_encrypt_mask = 0x00000001 # macro
SDMA_PKT_WRITE_TILED_HEADER_encrypt_shift = 16 # macro
def SDMA_PKT_WRITE_TILED_HEADER_ENCRYPT(x):  # macro
   return (((x)&0x00000001)<<16)
SDMA_PKT_WRITE_TILED_HEADER_tmz_offset = 0 # macro
SDMA_PKT_WRITE_TILED_HEADER_tmz_mask = 0x00000001 # macro
SDMA_PKT_WRITE_TILED_HEADER_tmz_shift = 18 # macro
def SDMA_PKT_WRITE_TILED_HEADER_TMZ(x):  # macro
   return (((x)&0x00000001)<<18)
SDMA_PKT_WRITE_TILED_DST_ADDR_LO_dst_addr_31_0_offset = 1 # macro
SDMA_PKT_WRITE_TILED_DST_ADDR_LO_dst_addr_31_0_mask = 0xFFFFFFFF # macro
SDMA_PKT_WRITE_TILED_DST_ADDR_LO_dst_addr_31_0_shift = 0 # macro
def SDMA_PKT_WRITE_TILED_DST_ADDR_LO_DST_ADDR_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_WRITE_TILED_DST_ADDR_HI_dst_addr_63_32_offset = 2 # macro
SDMA_PKT_WRITE_TILED_DST_ADDR_HI_dst_addr_63_32_mask = 0xFFFFFFFF # macro
SDMA_PKT_WRITE_TILED_DST_ADDR_HI_dst_addr_63_32_shift = 0 # macro
def SDMA_PKT_WRITE_TILED_DST_ADDR_HI_DST_ADDR_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_WRITE_TILED_DW_3_width_offset = 3 # macro
SDMA_PKT_WRITE_TILED_DW_3_width_mask = 0x00003FFF # macro
SDMA_PKT_WRITE_TILED_DW_3_width_shift = 0 # macro
def SDMA_PKT_WRITE_TILED_DW_3_WIDTH(x):  # macro
   return (((x)&0x00003FFF)<<0)
SDMA_PKT_WRITE_TILED_DW_4_height_offset = 4 # macro
SDMA_PKT_WRITE_TILED_DW_4_height_mask = 0x00003FFF # macro
SDMA_PKT_WRITE_TILED_DW_4_height_shift = 0 # macro
def SDMA_PKT_WRITE_TILED_DW_4_HEIGHT(x):  # macro
   return (((x)&0x00003FFF)<<0)
SDMA_PKT_WRITE_TILED_DW_4_depth_offset = 4 # macro
SDMA_PKT_WRITE_TILED_DW_4_depth_mask = 0x00001FFF # macro
SDMA_PKT_WRITE_TILED_DW_4_depth_shift = 16 # macro
def SDMA_PKT_WRITE_TILED_DW_4_DEPTH(x):  # macro
   return (((x)&0x00001FFF)<<16)
SDMA_PKT_WRITE_TILED_DW_5_element_size_offset = 5 # macro
SDMA_PKT_WRITE_TILED_DW_5_element_size_mask = 0x00000007 # macro
SDMA_PKT_WRITE_TILED_DW_5_element_size_shift = 0 # macro
def SDMA_PKT_WRITE_TILED_DW_5_ELEMENT_SIZE(x):  # macro
   return (((x)&0x00000007)<<0)
SDMA_PKT_WRITE_TILED_DW_5_swizzle_mode_offset = 5 # macro
SDMA_PKT_WRITE_TILED_DW_5_swizzle_mode_mask = 0x0000001F # macro
SDMA_PKT_WRITE_TILED_DW_5_swizzle_mode_shift = 3 # macro
def SDMA_PKT_WRITE_TILED_DW_5_SWIZZLE_MODE(x):  # macro
   return (((x)&0x0000001F)<<3)
SDMA_PKT_WRITE_TILED_DW_5_dimension_offset = 5 # macro
SDMA_PKT_WRITE_TILED_DW_5_dimension_mask = 0x00000003 # macro
SDMA_PKT_WRITE_TILED_DW_5_dimension_shift = 9 # macro
def SDMA_PKT_WRITE_TILED_DW_5_DIMENSION(x):  # macro
   return (((x)&0x00000003)<<9)
SDMA_PKT_WRITE_TILED_DW_5_mip_max_offset = 5 # macro
SDMA_PKT_WRITE_TILED_DW_5_mip_max_mask = 0x0000000F # macro
SDMA_PKT_WRITE_TILED_DW_5_mip_max_shift = 16 # macro
def SDMA_PKT_WRITE_TILED_DW_5_MIP_MAX(x):  # macro
   return (((x)&0x0000000F)<<16)
SDMA_PKT_WRITE_TILED_DW_6_x_offset = 6 # macro
SDMA_PKT_WRITE_TILED_DW_6_x_mask = 0x00003FFF # macro
SDMA_PKT_WRITE_TILED_DW_6_x_shift = 0 # macro
def SDMA_PKT_WRITE_TILED_DW_6_X(x):  # macro
   return (((x)&0x00003FFF)<<0)
SDMA_PKT_WRITE_TILED_DW_6_y_offset = 6 # macro
SDMA_PKT_WRITE_TILED_DW_6_y_mask = 0x00003FFF # macro
SDMA_PKT_WRITE_TILED_DW_6_y_shift = 16 # macro
def SDMA_PKT_WRITE_TILED_DW_6_Y(x):  # macro
   return (((x)&0x00003FFF)<<16)
SDMA_PKT_WRITE_TILED_DW_7_z_offset = 7 # macro
SDMA_PKT_WRITE_TILED_DW_7_z_mask = 0x00001FFF # macro
SDMA_PKT_WRITE_TILED_DW_7_z_shift = 0 # macro
def SDMA_PKT_WRITE_TILED_DW_7_Z(x):  # macro
   return (((x)&0x00001FFF)<<0)
SDMA_PKT_WRITE_TILED_DW_7_sw_offset = 7 # macro
SDMA_PKT_WRITE_TILED_DW_7_sw_mask = 0x00000003 # macro
SDMA_PKT_WRITE_TILED_DW_7_sw_shift = 24 # macro
def SDMA_PKT_WRITE_TILED_DW_7_SW(x):  # macro
   return (((x)&0x00000003)<<24)
SDMA_PKT_WRITE_TILED_COUNT_count_offset = 8 # macro
SDMA_PKT_WRITE_TILED_COUNT_count_mask = 0x000FFFFF # macro
SDMA_PKT_WRITE_TILED_COUNT_count_shift = 0 # macro
def SDMA_PKT_WRITE_TILED_COUNT_COUNT(x):  # macro
   return (((x)&0x000FFFFF)<<0)
SDMA_PKT_WRITE_TILED_DATA0_data0_offset = 9 # macro
SDMA_PKT_WRITE_TILED_DATA0_data0_mask = 0xFFFFFFFF # macro
SDMA_PKT_WRITE_TILED_DATA0_data0_shift = 0 # macro
def SDMA_PKT_WRITE_TILED_DATA0_DATA0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_WRITE_TILED_BC_HEADER_op_offset = 0 # macro
SDMA_PKT_WRITE_TILED_BC_HEADER_op_mask = 0x000000FF # macro
SDMA_PKT_WRITE_TILED_BC_HEADER_op_shift = 0 # macro
def SDMA_PKT_WRITE_TILED_BC_HEADER_OP(x):  # macro
   return (((x)&0x000000FF)<<0)
SDMA_PKT_WRITE_TILED_BC_HEADER_sub_op_offset = 0 # macro
SDMA_PKT_WRITE_TILED_BC_HEADER_sub_op_mask = 0x000000FF # macro
SDMA_PKT_WRITE_TILED_BC_HEADER_sub_op_shift = 8 # macro
def SDMA_PKT_WRITE_TILED_BC_HEADER_SUB_OP(x):  # macro
   return (((x)&0x000000FF)<<8)
SDMA_PKT_WRITE_TILED_BC_DST_ADDR_LO_dst_addr_31_0_offset = 1 # macro
SDMA_PKT_WRITE_TILED_BC_DST_ADDR_LO_dst_addr_31_0_mask = 0xFFFFFFFF # macro
SDMA_PKT_WRITE_TILED_BC_DST_ADDR_LO_dst_addr_31_0_shift = 0 # macro
def SDMA_PKT_WRITE_TILED_BC_DST_ADDR_LO_DST_ADDR_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_WRITE_TILED_BC_DST_ADDR_HI_dst_addr_63_32_offset = 2 # macro
SDMA_PKT_WRITE_TILED_BC_DST_ADDR_HI_dst_addr_63_32_mask = 0xFFFFFFFF # macro
SDMA_PKT_WRITE_TILED_BC_DST_ADDR_HI_dst_addr_63_32_shift = 0 # macro
def SDMA_PKT_WRITE_TILED_BC_DST_ADDR_HI_DST_ADDR_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_WRITE_TILED_BC_DW_3_width_offset = 3 # macro
SDMA_PKT_WRITE_TILED_BC_DW_3_width_mask = 0x00003FFF # macro
SDMA_PKT_WRITE_TILED_BC_DW_3_width_shift = 0 # macro
def SDMA_PKT_WRITE_TILED_BC_DW_3_WIDTH(x):  # macro
   return (((x)&0x00003FFF)<<0)
SDMA_PKT_WRITE_TILED_BC_DW_4_height_offset = 4 # macro
SDMA_PKT_WRITE_TILED_BC_DW_4_height_mask = 0x00003FFF # macro
SDMA_PKT_WRITE_TILED_BC_DW_4_height_shift = 0 # macro
def SDMA_PKT_WRITE_TILED_BC_DW_4_HEIGHT(x):  # macro
   return (((x)&0x00003FFF)<<0)
SDMA_PKT_WRITE_TILED_BC_DW_4_depth_offset = 4 # macro
SDMA_PKT_WRITE_TILED_BC_DW_4_depth_mask = 0x000007FF # macro
SDMA_PKT_WRITE_TILED_BC_DW_4_depth_shift = 16 # macro
def SDMA_PKT_WRITE_TILED_BC_DW_4_DEPTH(x):  # macro
   return (((x)&0x000007FF)<<16)
SDMA_PKT_WRITE_TILED_BC_DW_5_element_size_offset = 5 # macro
SDMA_PKT_WRITE_TILED_BC_DW_5_element_size_mask = 0x00000007 # macro
SDMA_PKT_WRITE_TILED_BC_DW_5_element_size_shift = 0 # macro
def SDMA_PKT_WRITE_TILED_BC_DW_5_ELEMENT_SIZE(x):  # macro
   return (((x)&0x00000007)<<0)
SDMA_PKT_WRITE_TILED_BC_DW_5_array_mode_offset = 5 # macro
SDMA_PKT_WRITE_TILED_BC_DW_5_array_mode_mask = 0x0000000F # macro
SDMA_PKT_WRITE_TILED_BC_DW_5_array_mode_shift = 3 # macro
def SDMA_PKT_WRITE_TILED_BC_DW_5_ARRAY_MODE(x):  # macro
   return (((x)&0x0000000F)<<3)
SDMA_PKT_WRITE_TILED_BC_DW_5_mit_mode_offset = 5 # macro
SDMA_PKT_WRITE_TILED_BC_DW_5_mit_mode_mask = 0x00000007 # macro
SDMA_PKT_WRITE_TILED_BC_DW_5_mit_mode_shift = 8 # macro
def SDMA_PKT_WRITE_TILED_BC_DW_5_MIT_MODE(x):  # macro
   return (((x)&0x00000007)<<8)
SDMA_PKT_WRITE_TILED_BC_DW_5_tilesplit_size_offset = 5 # macro
SDMA_PKT_WRITE_TILED_BC_DW_5_tilesplit_size_mask = 0x00000007 # macro
SDMA_PKT_WRITE_TILED_BC_DW_5_tilesplit_size_shift = 11 # macro
def SDMA_PKT_WRITE_TILED_BC_DW_5_TILESPLIT_SIZE(x):  # macro
   return (((x)&0x00000007)<<11)
SDMA_PKT_WRITE_TILED_BC_DW_5_bank_w_offset = 5 # macro
SDMA_PKT_WRITE_TILED_BC_DW_5_bank_w_mask = 0x00000003 # macro
SDMA_PKT_WRITE_TILED_BC_DW_5_bank_w_shift = 15 # macro
def SDMA_PKT_WRITE_TILED_BC_DW_5_BANK_W(x):  # macro
   return (((x)&0x00000003)<<15)
SDMA_PKT_WRITE_TILED_BC_DW_5_bank_h_offset = 5 # macro
SDMA_PKT_WRITE_TILED_BC_DW_5_bank_h_mask = 0x00000003 # macro
SDMA_PKT_WRITE_TILED_BC_DW_5_bank_h_shift = 18 # macro
def SDMA_PKT_WRITE_TILED_BC_DW_5_BANK_H(x):  # macro
   return (((x)&0x00000003)<<18)
SDMA_PKT_WRITE_TILED_BC_DW_5_num_bank_offset = 5 # macro
SDMA_PKT_WRITE_TILED_BC_DW_5_num_bank_mask = 0x00000003 # macro
SDMA_PKT_WRITE_TILED_BC_DW_5_num_bank_shift = 21 # macro
def SDMA_PKT_WRITE_TILED_BC_DW_5_NUM_BANK(x):  # macro
   return (((x)&0x00000003)<<21)
SDMA_PKT_WRITE_TILED_BC_DW_5_mat_aspt_offset = 5 # macro
SDMA_PKT_WRITE_TILED_BC_DW_5_mat_aspt_mask = 0x00000003 # macro
SDMA_PKT_WRITE_TILED_BC_DW_5_mat_aspt_shift = 24 # macro
def SDMA_PKT_WRITE_TILED_BC_DW_5_MAT_ASPT(x):  # macro
   return (((x)&0x00000003)<<24)
SDMA_PKT_WRITE_TILED_BC_DW_5_pipe_config_offset = 5 # macro
SDMA_PKT_WRITE_TILED_BC_DW_5_pipe_config_mask = 0x0000001F # macro
SDMA_PKT_WRITE_TILED_BC_DW_5_pipe_config_shift = 26 # macro
def SDMA_PKT_WRITE_TILED_BC_DW_5_PIPE_CONFIG(x):  # macro
   return (((x)&0x0000001F)<<26)
SDMA_PKT_WRITE_TILED_BC_DW_6_x_offset = 6 # macro
SDMA_PKT_WRITE_TILED_BC_DW_6_x_mask = 0x00003FFF # macro
SDMA_PKT_WRITE_TILED_BC_DW_6_x_shift = 0 # macro
def SDMA_PKT_WRITE_TILED_BC_DW_6_X(x):  # macro
   return (((x)&0x00003FFF)<<0)
SDMA_PKT_WRITE_TILED_BC_DW_6_y_offset = 6 # macro
SDMA_PKT_WRITE_TILED_BC_DW_6_y_mask = 0x00003FFF # macro
SDMA_PKT_WRITE_TILED_BC_DW_6_y_shift = 16 # macro
def SDMA_PKT_WRITE_TILED_BC_DW_6_Y(x):  # macro
   return (((x)&0x00003FFF)<<16)
SDMA_PKT_WRITE_TILED_BC_DW_7_z_offset = 7 # macro
SDMA_PKT_WRITE_TILED_BC_DW_7_z_mask = 0x000007FF # macro
SDMA_PKT_WRITE_TILED_BC_DW_7_z_shift = 0 # macro
def SDMA_PKT_WRITE_TILED_BC_DW_7_Z(x):  # macro
   return (((x)&0x000007FF)<<0)
SDMA_PKT_WRITE_TILED_BC_DW_7_sw_offset = 7 # macro
SDMA_PKT_WRITE_TILED_BC_DW_7_sw_mask = 0x00000003 # macro
SDMA_PKT_WRITE_TILED_BC_DW_7_sw_shift = 24 # macro
def SDMA_PKT_WRITE_TILED_BC_DW_7_SW(x):  # macro
   return (((x)&0x00000003)<<24)
SDMA_PKT_WRITE_TILED_BC_COUNT_count_offset = 8 # macro
SDMA_PKT_WRITE_TILED_BC_COUNT_count_mask = 0x000FFFFF # macro
SDMA_PKT_WRITE_TILED_BC_COUNT_count_shift = 2 # macro
def SDMA_PKT_WRITE_TILED_BC_COUNT_COUNT(x):  # macro
   return (((x)&0x000FFFFF)<<2)
SDMA_PKT_WRITE_TILED_BC_DATA0_data0_offset = 9 # macro
SDMA_PKT_WRITE_TILED_BC_DATA0_data0_mask = 0xFFFFFFFF # macro
SDMA_PKT_WRITE_TILED_BC_DATA0_data0_shift = 0 # macro
def SDMA_PKT_WRITE_TILED_BC_DATA0_DATA0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_PTEPDE_COPY_HEADER_op_offset = 0 # macro
SDMA_PKT_PTEPDE_COPY_HEADER_op_mask = 0x000000FF # macro
SDMA_PKT_PTEPDE_COPY_HEADER_op_shift = 0 # macro
def SDMA_PKT_PTEPDE_COPY_HEADER_OP(x):  # macro
   return (((x)&0x000000FF)<<0)
SDMA_PKT_PTEPDE_COPY_HEADER_sub_op_offset = 0 # macro
SDMA_PKT_PTEPDE_COPY_HEADER_sub_op_mask = 0x000000FF # macro
SDMA_PKT_PTEPDE_COPY_HEADER_sub_op_shift = 8 # macro
def SDMA_PKT_PTEPDE_COPY_HEADER_SUB_OP(x):  # macro
   return (((x)&0x000000FF)<<8)
SDMA_PKT_PTEPDE_COPY_HEADER_tmz_offset = 0 # macro
SDMA_PKT_PTEPDE_COPY_HEADER_tmz_mask = 0x00000001 # macro
SDMA_PKT_PTEPDE_COPY_HEADER_tmz_shift = 18 # macro
def SDMA_PKT_PTEPDE_COPY_HEADER_TMZ(x):  # macro
   return (((x)&0x00000001)<<18)
SDMA_PKT_PTEPDE_COPY_HEADER_ptepde_op_offset = 0 # macro
SDMA_PKT_PTEPDE_COPY_HEADER_ptepde_op_mask = 0x00000001 # macro
SDMA_PKT_PTEPDE_COPY_HEADER_ptepde_op_shift = 31 # macro
def SDMA_PKT_PTEPDE_COPY_HEADER_PTEPDE_OP(x):  # macro
   return (((x)&0x00000001)<<31)
SDMA_PKT_PTEPDE_COPY_SRC_ADDR_LO_src_addr_31_0_offset = 1 # macro
SDMA_PKT_PTEPDE_COPY_SRC_ADDR_LO_src_addr_31_0_mask = 0xFFFFFFFF # macro
SDMA_PKT_PTEPDE_COPY_SRC_ADDR_LO_src_addr_31_0_shift = 0 # macro
def SDMA_PKT_PTEPDE_COPY_SRC_ADDR_LO_SRC_ADDR_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_PTEPDE_COPY_SRC_ADDR_HI_src_addr_63_32_offset = 2 # macro
SDMA_PKT_PTEPDE_COPY_SRC_ADDR_HI_src_addr_63_32_mask = 0xFFFFFFFF # macro
SDMA_PKT_PTEPDE_COPY_SRC_ADDR_HI_src_addr_63_32_shift = 0 # macro
def SDMA_PKT_PTEPDE_COPY_SRC_ADDR_HI_SRC_ADDR_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_PTEPDE_COPY_DST_ADDR_LO_dst_addr_31_0_offset = 3 # macro
SDMA_PKT_PTEPDE_COPY_DST_ADDR_LO_dst_addr_31_0_mask = 0xFFFFFFFF # macro
SDMA_PKT_PTEPDE_COPY_DST_ADDR_LO_dst_addr_31_0_shift = 0 # macro
def SDMA_PKT_PTEPDE_COPY_DST_ADDR_LO_DST_ADDR_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_PTEPDE_COPY_DST_ADDR_HI_dst_addr_63_32_offset = 4 # macro
SDMA_PKT_PTEPDE_COPY_DST_ADDR_HI_dst_addr_63_32_mask = 0xFFFFFFFF # macro
SDMA_PKT_PTEPDE_COPY_DST_ADDR_HI_dst_addr_63_32_shift = 0 # macro
def SDMA_PKT_PTEPDE_COPY_DST_ADDR_HI_DST_ADDR_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_PTEPDE_COPY_MASK_DW0_mask_dw0_offset = 5 # macro
SDMA_PKT_PTEPDE_COPY_MASK_DW0_mask_dw0_mask = 0xFFFFFFFF # macro
SDMA_PKT_PTEPDE_COPY_MASK_DW0_mask_dw0_shift = 0 # macro
def SDMA_PKT_PTEPDE_COPY_MASK_DW0_MASK_DW0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_PTEPDE_COPY_MASK_DW1_mask_dw1_offset = 6 # macro
SDMA_PKT_PTEPDE_COPY_MASK_DW1_mask_dw1_mask = 0xFFFFFFFF # macro
SDMA_PKT_PTEPDE_COPY_MASK_DW1_mask_dw1_shift = 0 # macro
def SDMA_PKT_PTEPDE_COPY_MASK_DW1_MASK_DW1(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_PTEPDE_COPY_COUNT_count_offset = 7 # macro
SDMA_PKT_PTEPDE_COPY_COUNT_count_mask = 0x0007FFFF # macro
SDMA_PKT_PTEPDE_COPY_COUNT_count_shift = 0 # macro
def SDMA_PKT_PTEPDE_COPY_COUNT_COUNT(x):  # macro
   return (((x)&0x0007FFFF)<<0)
SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_op_offset = 0 # macro
SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_op_mask = 0x000000FF # macro
SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_op_shift = 0 # macro
def SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_OP(x):  # macro
   return (((x)&0x000000FF)<<0)
SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_sub_op_offset = 0 # macro
SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_sub_op_mask = 0x000000FF # macro
SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_sub_op_shift = 8 # macro
def SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_SUB_OP(x):  # macro
   return (((x)&0x000000FF)<<8)
SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_pte_size_offset = 0 # macro
SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_pte_size_mask = 0x00000003 # macro
SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_pte_size_shift = 28 # macro
def SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_PTE_SIZE(x):  # macro
   return (((x)&0x00000003)<<28)
SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_direction_offset = 0 # macro
SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_direction_mask = 0x00000001 # macro
SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_direction_shift = 30 # macro
def SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_DIRECTION(x):  # macro
   return (((x)&0x00000001)<<30)
SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_ptepde_op_offset = 0 # macro
SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_ptepde_op_mask = 0x00000001 # macro
SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_ptepde_op_shift = 31 # macro
def SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_PTEPDE_OP(x):  # macro
   return (((x)&0x00000001)<<31)
SDMA_PKT_PTEPDE_COPY_BACKWARDS_SRC_ADDR_LO_src_addr_31_0_offset = 1 # macro
SDMA_PKT_PTEPDE_COPY_BACKWARDS_SRC_ADDR_LO_src_addr_31_0_mask = 0xFFFFFFFF # macro
SDMA_PKT_PTEPDE_COPY_BACKWARDS_SRC_ADDR_LO_src_addr_31_0_shift = 0 # macro
def SDMA_PKT_PTEPDE_COPY_BACKWARDS_SRC_ADDR_LO_SRC_ADDR_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_PTEPDE_COPY_BACKWARDS_SRC_ADDR_HI_src_addr_63_32_offset = 2 # macro
SDMA_PKT_PTEPDE_COPY_BACKWARDS_SRC_ADDR_HI_src_addr_63_32_mask = 0xFFFFFFFF # macro
SDMA_PKT_PTEPDE_COPY_BACKWARDS_SRC_ADDR_HI_src_addr_63_32_shift = 0 # macro
def SDMA_PKT_PTEPDE_COPY_BACKWARDS_SRC_ADDR_HI_SRC_ADDR_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_PTEPDE_COPY_BACKWARDS_DST_ADDR_LO_dst_addr_31_0_offset = 3 # macro
SDMA_PKT_PTEPDE_COPY_BACKWARDS_DST_ADDR_LO_dst_addr_31_0_mask = 0xFFFFFFFF # macro
SDMA_PKT_PTEPDE_COPY_BACKWARDS_DST_ADDR_LO_dst_addr_31_0_shift = 0 # macro
def SDMA_PKT_PTEPDE_COPY_BACKWARDS_DST_ADDR_LO_DST_ADDR_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_PTEPDE_COPY_BACKWARDS_DST_ADDR_HI_dst_addr_63_32_offset = 4 # macro
SDMA_PKT_PTEPDE_COPY_BACKWARDS_DST_ADDR_HI_dst_addr_63_32_mask = 0xFFFFFFFF # macro
SDMA_PKT_PTEPDE_COPY_BACKWARDS_DST_ADDR_HI_dst_addr_63_32_shift = 0 # macro
def SDMA_PKT_PTEPDE_COPY_BACKWARDS_DST_ADDR_HI_DST_ADDR_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_PTEPDE_COPY_BACKWARDS_MASK_BIT_FOR_DW_mask_first_xfer_offset = 5 # macro
SDMA_PKT_PTEPDE_COPY_BACKWARDS_MASK_BIT_FOR_DW_mask_first_xfer_mask = 0x000000FF # macro
SDMA_PKT_PTEPDE_COPY_BACKWARDS_MASK_BIT_FOR_DW_mask_first_xfer_shift = 0 # macro
def SDMA_PKT_PTEPDE_COPY_BACKWARDS_MASK_BIT_FOR_DW_MASK_FIRST_XFER(x):  # macro
   return (((x)&0x000000FF)<<0)
SDMA_PKT_PTEPDE_COPY_BACKWARDS_MASK_BIT_FOR_DW_mask_last_xfer_offset = 5 # macro
SDMA_PKT_PTEPDE_COPY_BACKWARDS_MASK_BIT_FOR_DW_mask_last_xfer_mask = 0x000000FF # macro
SDMA_PKT_PTEPDE_COPY_BACKWARDS_MASK_BIT_FOR_DW_mask_last_xfer_shift = 8 # macro
def SDMA_PKT_PTEPDE_COPY_BACKWARDS_MASK_BIT_FOR_DW_MASK_LAST_XFER(x):  # macro
   return (((x)&0x000000FF)<<8)
SDMA_PKT_PTEPDE_COPY_BACKWARDS_COUNT_IN_32B_XFER_count_offset = 6 # macro
SDMA_PKT_PTEPDE_COPY_BACKWARDS_COUNT_IN_32B_XFER_count_mask = 0x0001FFFF # macro
SDMA_PKT_PTEPDE_COPY_BACKWARDS_COUNT_IN_32B_XFER_count_shift = 0 # macro
def SDMA_PKT_PTEPDE_COPY_BACKWARDS_COUNT_IN_32B_XFER_COUNT(x):  # macro
   return (((x)&0x0001FFFF)<<0)
SDMA_PKT_PTEPDE_RMW_HEADER_op_offset = 0 # macro
SDMA_PKT_PTEPDE_RMW_HEADER_op_mask = 0x000000FF # macro
SDMA_PKT_PTEPDE_RMW_HEADER_op_shift = 0 # macro
def SDMA_PKT_PTEPDE_RMW_HEADER_OP(x):  # macro
   return (((x)&0x000000FF)<<0)
SDMA_PKT_PTEPDE_RMW_HEADER_sub_op_offset = 0 # macro
SDMA_PKT_PTEPDE_RMW_HEADER_sub_op_mask = 0x000000FF # macro
SDMA_PKT_PTEPDE_RMW_HEADER_sub_op_shift = 8 # macro
def SDMA_PKT_PTEPDE_RMW_HEADER_SUB_OP(x):  # macro
   return (((x)&0x000000FF)<<8)
SDMA_PKT_PTEPDE_RMW_HEADER_mtype_offset = 0 # macro
SDMA_PKT_PTEPDE_RMW_HEADER_mtype_mask = 0x00000007 # macro
SDMA_PKT_PTEPDE_RMW_HEADER_mtype_shift = 16 # macro
def SDMA_PKT_PTEPDE_RMW_HEADER_MTYPE(x):  # macro
   return (((x)&0x00000007)<<16)
SDMA_PKT_PTEPDE_RMW_HEADER_gcc_offset = 0 # macro
SDMA_PKT_PTEPDE_RMW_HEADER_gcc_mask = 0x00000001 # macro
SDMA_PKT_PTEPDE_RMW_HEADER_gcc_shift = 19 # macro
def SDMA_PKT_PTEPDE_RMW_HEADER_GCC(x):  # macro
   return (((x)&0x00000001)<<19)
SDMA_PKT_PTEPDE_RMW_HEADER_sys_offset = 0 # macro
SDMA_PKT_PTEPDE_RMW_HEADER_sys_mask = 0x00000001 # macro
SDMA_PKT_PTEPDE_RMW_HEADER_sys_shift = 20 # macro
def SDMA_PKT_PTEPDE_RMW_HEADER_SYS(x):  # macro
   return (((x)&0x00000001)<<20)
SDMA_PKT_PTEPDE_RMW_HEADER_snp_offset = 0 # macro
SDMA_PKT_PTEPDE_RMW_HEADER_snp_mask = 0x00000001 # macro
SDMA_PKT_PTEPDE_RMW_HEADER_snp_shift = 22 # macro
def SDMA_PKT_PTEPDE_RMW_HEADER_SNP(x):  # macro
   return (((x)&0x00000001)<<22)
SDMA_PKT_PTEPDE_RMW_HEADER_gpa_offset = 0 # macro
SDMA_PKT_PTEPDE_RMW_HEADER_gpa_mask = 0x00000001 # macro
SDMA_PKT_PTEPDE_RMW_HEADER_gpa_shift = 23 # macro
def SDMA_PKT_PTEPDE_RMW_HEADER_GPA(x):  # macro
   return (((x)&0x00000001)<<23)
SDMA_PKT_PTEPDE_RMW_HEADER_l2_policy_offset = 0 # macro
SDMA_PKT_PTEPDE_RMW_HEADER_l2_policy_mask = 0x00000003 # macro
SDMA_PKT_PTEPDE_RMW_HEADER_l2_policy_shift = 24 # macro
def SDMA_PKT_PTEPDE_RMW_HEADER_L2_POLICY(x):  # macro
   return (((x)&0x00000003)<<24)
SDMA_PKT_PTEPDE_RMW_ADDR_LO_addr_31_0_offset = 1 # macro
SDMA_PKT_PTEPDE_RMW_ADDR_LO_addr_31_0_mask = 0xFFFFFFFF # macro
SDMA_PKT_PTEPDE_RMW_ADDR_LO_addr_31_0_shift = 0 # macro
def SDMA_PKT_PTEPDE_RMW_ADDR_LO_ADDR_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_PTEPDE_RMW_ADDR_HI_addr_63_32_offset = 2 # macro
SDMA_PKT_PTEPDE_RMW_ADDR_HI_addr_63_32_mask = 0xFFFFFFFF # macro
SDMA_PKT_PTEPDE_RMW_ADDR_HI_addr_63_32_shift = 0 # macro
def SDMA_PKT_PTEPDE_RMW_ADDR_HI_ADDR_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_PTEPDE_RMW_MASK_LO_mask_31_0_offset = 3 # macro
SDMA_PKT_PTEPDE_RMW_MASK_LO_mask_31_0_mask = 0xFFFFFFFF # macro
SDMA_PKT_PTEPDE_RMW_MASK_LO_mask_31_0_shift = 0 # macro
def SDMA_PKT_PTEPDE_RMW_MASK_LO_MASK_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_PTEPDE_RMW_MASK_HI_mask_63_32_offset = 4 # macro
SDMA_PKT_PTEPDE_RMW_MASK_HI_mask_63_32_mask = 0xFFFFFFFF # macro
SDMA_PKT_PTEPDE_RMW_MASK_HI_mask_63_32_shift = 0 # macro
def SDMA_PKT_PTEPDE_RMW_MASK_HI_MASK_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_PTEPDE_RMW_VALUE_LO_value_31_0_offset = 5 # macro
SDMA_PKT_PTEPDE_RMW_VALUE_LO_value_31_0_mask = 0xFFFFFFFF # macro
SDMA_PKT_PTEPDE_RMW_VALUE_LO_value_31_0_shift = 0 # macro
def SDMA_PKT_PTEPDE_RMW_VALUE_LO_VALUE_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_PTEPDE_RMW_VALUE_HI_value_63_32_offset = 6 # macro
SDMA_PKT_PTEPDE_RMW_VALUE_HI_value_63_32_mask = 0xFFFFFFFF # macro
SDMA_PKT_PTEPDE_RMW_VALUE_HI_value_63_32_shift = 0 # macro
def SDMA_PKT_PTEPDE_RMW_VALUE_HI_VALUE_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_WRITE_INCR_HEADER_op_offset = 0 # macro
SDMA_PKT_WRITE_INCR_HEADER_op_mask = 0x000000FF # macro
SDMA_PKT_WRITE_INCR_HEADER_op_shift = 0 # macro
def SDMA_PKT_WRITE_INCR_HEADER_OP(x):  # macro
   return (((x)&0x000000FF)<<0)
SDMA_PKT_WRITE_INCR_HEADER_sub_op_offset = 0 # macro
SDMA_PKT_WRITE_INCR_HEADER_sub_op_mask = 0x000000FF # macro
SDMA_PKT_WRITE_INCR_HEADER_sub_op_shift = 8 # macro
def SDMA_PKT_WRITE_INCR_HEADER_SUB_OP(x):  # macro
   return (((x)&0x000000FF)<<8)
SDMA_PKT_WRITE_INCR_DST_ADDR_LO_dst_addr_31_0_offset = 1 # macro
SDMA_PKT_WRITE_INCR_DST_ADDR_LO_dst_addr_31_0_mask = 0xFFFFFFFF # macro
SDMA_PKT_WRITE_INCR_DST_ADDR_LO_dst_addr_31_0_shift = 0 # macro
def SDMA_PKT_WRITE_INCR_DST_ADDR_LO_DST_ADDR_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_WRITE_INCR_DST_ADDR_HI_dst_addr_63_32_offset = 2 # macro
SDMA_PKT_WRITE_INCR_DST_ADDR_HI_dst_addr_63_32_mask = 0xFFFFFFFF # macro
SDMA_PKT_WRITE_INCR_DST_ADDR_HI_dst_addr_63_32_shift = 0 # macro
def SDMA_PKT_WRITE_INCR_DST_ADDR_HI_DST_ADDR_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_WRITE_INCR_MASK_DW0_mask_dw0_offset = 3 # macro
SDMA_PKT_WRITE_INCR_MASK_DW0_mask_dw0_mask = 0xFFFFFFFF # macro
SDMA_PKT_WRITE_INCR_MASK_DW0_mask_dw0_shift = 0 # macro
def SDMA_PKT_WRITE_INCR_MASK_DW0_MASK_DW0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_WRITE_INCR_MASK_DW1_mask_dw1_offset = 4 # macro
SDMA_PKT_WRITE_INCR_MASK_DW1_mask_dw1_mask = 0xFFFFFFFF # macro
SDMA_PKT_WRITE_INCR_MASK_DW1_mask_dw1_shift = 0 # macro
def SDMA_PKT_WRITE_INCR_MASK_DW1_MASK_DW1(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_WRITE_INCR_INIT_DW0_init_dw0_offset = 5 # macro
SDMA_PKT_WRITE_INCR_INIT_DW0_init_dw0_mask = 0xFFFFFFFF # macro
SDMA_PKT_WRITE_INCR_INIT_DW0_init_dw0_shift = 0 # macro
def SDMA_PKT_WRITE_INCR_INIT_DW0_INIT_DW0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_WRITE_INCR_INIT_DW1_init_dw1_offset = 6 # macro
SDMA_PKT_WRITE_INCR_INIT_DW1_init_dw1_mask = 0xFFFFFFFF # macro
SDMA_PKT_WRITE_INCR_INIT_DW1_init_dw1_shift = 0 # macro
def SDMA_PKT_WRITE_INCR_INIT_DW1_INIT_DW1(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_WRITE_INCR_INCR_DW0_incr_dw0_offset = 7 # macro
SDMA_PKT_WRITE_INCR_INCR_DW0_incr_dw0_mask = 0xFFFFFFFF # macro
SDMA_PKT_WRITE_INCR_INCR_DW0_incr_dw0_shift = 0 # macro
def SDMA_PKT_WRITE_INCR_INCR_DW0_INCR_DW0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_WRITE_INCR_INCR_DW1_incr_dw1_offset = 8 # macro
SDMA_PKT_WRITE_INCR_INCR_DW1_incr_dw1_mask = 0xFFFFFFFF # macro
SDMA_PKT_WRITE_INCR_INCR_DW1_incr_dw1_shift = 0 # macro
def SDMA_PKT_WRITE_INCR_INCR_DW1_INCR_DW1(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_WRITE_INCR_COUNT_count_offset = 9 # macro
SDMA_PKT_WRITE_INCR_COUNT_count_mask = 0x0007FFFF # macro
SDMA_PKT_WRITE_INCR_COUNT_count_shift = 0 # macro
def SDMA_PKT_WRITE_INCR_COUNT_COUNT(x):  # macro
   return (((x)&0x0007FFFF)<<0)
SDMA_PKT_INDIRECT_HEADER_op_offset = 0 # macro
SDMA_PKT_INDIRECT_HEADER_op_mask = 0x000000FF # macro
SDMA_PKT_INDIRECT_HEADER_op_shift = 0 # macro
def SDMA_PKT_INDIRECT_HEADER_OP(x):  # macro
   return (((x)&0x000000FF)<<0)
SDMA_PKT_INDIRECT_HEADER_sub_op_offset = 0 # macro
SDMA_PKT_INDIRECT_HEADER_sub_op_mask = 0x000000FF # macro
SDMA_PKT_INDIRECT_HEADER_sub_op_shift = 8 # macro
def SDMA_PKT_INDIRECT_HEADER_SUB_OP(x):  # macro
   return (((x)&0x000000FF)<<8)
SDMA_PKT_INDIRECT_HEADER_vmid_offset = 0 # macro
SDMA_PKT_INDIRECT_HEADER_vmid_mask = 0x0000000F # macro
SDMA_PKT_INDIRECT_HEADER_vmid_shift = 16 # macro
def SDMA_PKT_INDIRECT_HEADER_VMID(x):  # macro
   return (((x)&0x0000000F)<<16)
SDMA_PKT_INDIRECT_HEADER_priv_offset = 0 # macro
SDMA_PKT_INDIRECT_HEADER_priv_mask = 0x00000001 # macro
SDMA_PKT_INDIRECT_HEADER_priv_shift = 31 # macro
def SDMA_PKT_INDIRECT_HEADER_PRIV(x):  # macro
   return (((x)&0x00000001)<<31)
SDMA_PKT_INDIRECT_BASE_LO_ib_base_31_0_offset = 1 # macro
SDMA_PKT_INDIRECT_BASE_LO_ib_base_31_0_mask = 0xFFFFFFFF # macro
SDMA_PKT_INDIRECT_BASE_LO_ib_base_31_0_shift = 0 # macro
def SDMA_PKT_INDIRECT_BASE_LO_IB_BASE_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_INDIRECT_BASE_HI_ib_base_63_32_offset = 2 # macro
SDMA_PKT_INDIRECT_BASE_HI_ib_base_63_32_mask = 0xFFFFFFFF # macro
SDMA_PKT_INDIRECT_BASE_HI_ib_base_63_32_shift = 0 # macro
def SDMA_PKT_INDIRECT_BASE_HI_IB_BASE_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_INDIRECT_IB_SIZE_ib_size_offset = 3 # macro
SDMA_PKT_INDIRECT_IB_SIZE_ib_size_mask = 0x000FFFFF # macro
SDMA_PKT_INDIRECT_IB_SIZE_ib_size_shift = 0 # macro
def SDMA_PKT_INDIRECT_IB_SIZE_IB_SIZE(x):  # macro
   return (((x)&0x000FFFFF)<<0)
SDMA_PKT_INDIRECT_CSA_ADDR_LO_csa_addr_31_0_offset = 4 # macro
SDMA_PKT_INDIRECT_CSA_ADDR_LO_csa_addr_31_0_mask = 0xFFFFFFFF # macro
SDMA_PKT_INDIRECT_CSA_ADDR_LO_csa_addr_31_0_shift = 0 # macro
def SDMA_PKT_INDIRECT_CSA_ADDR_LO_CSA_ADDR_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_INDIRECT_CSA_ADDR_HI_csa_addr_63_32_offset = 5 # macro
SDMA_PKT_INDIRECT_CSA_ADDR_HI_csa_addr_63_32_mask = 0xFFFFFFFF # macro
SDMA_PKT_INDIRECT_CSA_ADDR_HI_csa_addr_63_32_shift = 0 # macro
def SDMA_PKT_INDIRECT_CSA_ADDR_HI_CSA_ADDR_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_SEMAPHORE_HEADER_op_offset = 0 # macro
SDMA_PKT_SEMAPHORE_HEADER_op_mask = 0x000000FF # macro
SDMA_PKT_SEMAPHORE_HEADER_op_shift = 0 # macro
def SDMA_PKT_SEMAPHORE_HEADER_OP(x):  # macro
   return (((x)&0x000000FF)<<0)
SDMA_PKT_SEMAPHORE_HEADER_sub_op_offset = 0 # macro
SDMA_PKT_SEMAPHORE_HEADER_sub_op_mask = 0x000000FF # macro
SDMA_PKT_SEMAPHORE_HEADER_sub_op_shift = 8 # macro
def SDMA_PKT_SEMAPHORE_HEADER_SUB_OP(x):  # macro
   return (((x)&0x000000FF)<<8)
SDMA_PKT_SEMAPHORE_HEADER_write_one_offset = 0 # macro
SDMA_PKT_SEMAPHORE_HEADER_write_one_mask = 0x00000001 # macro
SDMA_PKT_SEMAPHORE_HEADER_write_one_shift = 29 # macro
def SDMA_PKT_SEMAPHORE_HEADER_WRITE_ONE(x):  # macro
   return (((x)&0x00000001)<<29)
SDMA_PKT_SEMAPHORE_HEADER_signal_offset = 0 # macro
SDMA_PKT_SEMAPHORE_HEADER_signal_mask = 0x00000001 # macro
SDMA_PKT_SEMAPHORE_HEADER_signal_shift = 30 # macro
def SDMA_PKT_SEMAPHORE_HEADER_SIGNAL(x):  # macro
   return (((x)&0x00000001)<<30)
SDMA_PKT_SEMAPHORE_HEADER_mailbox_offset = 0 # macro
SDMA_PKT_SEMAPHORE_HEADER_mailbox_mask = 0x00000001 # macro
SDMA_PKT_SEMAPHORE_HEADER_mailbox_shift = 31 # macro
def SDMA_PKT_SEMAPHORE_HEADER_MAILBOX(x):  # macro
   return (((x)&0x00000001)<<31)
SDMA_PKT_SEMAPHORE_ADDR_LO_addr_31_0_offset = 1 # macro
SDMA_PKT_SEMAPHORE_ADDR_LO_addr_31_0_mask = 0xFFFFFFFF # macro
SDMA_PKT_SEMAPHORE_ADDR_LO_addr_31_0_shift = 0 # macro
def SDMA_PKT_SEMAPHORE_ADDR_LO_ADDR_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_SEMAPHORE_ADDR_HI_addr_63_32_offset = 2 # macro
SDMA_PKT_SEMAPHORE_ADDR_HI_addr_63_32_mask = 0xFFFFFFFF # macro
SDMA_PKT_SEMAPHORE_ADDR_HI_addr_63_32_shift = 0 # macro
def SDMA_PKT_SEMAPHORE_ADDR_HI_ADDR_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_FENCE_HEADER_op_offset = 0 # macro
SDMA_PKT_FENCE_HEADER_op_mask = 0x000000FF # macro
SDMA_PKT_FENCE_HEADER_op_shift = 0 # macro
def SDMA_PKT_FENCE_HEADER_OP(x):  # macro
   return (((x)&0x000000FF)<<0)
SDMA_PKT_FENCE_HEADER_sub_op_offset = 0 # macro
SDMA_PKT_FENCE_HEADER_sub_op_mask = 0x000000FF # macro
SDMA_PKT_FENCE_HEADER_sub_op_shift = 8 # macro
def SDMA_PKT_FENCE_HEADER_SUB_OP(x):  # macro
   return (((x)&0x000000FF)<<8)
SDMA_PKT_FENCE_HEADER_mtype_offset = 0 # macro
SDMA_PKT_FENCE_HEADER_mtype_mask = 0x00000007 # macro
SDMA_PKT_FENCE_HEADER_mtype_shift = 16 # macro
def SDMA_PKT_FENCE_HEADER_MTYPE(x):  # macro
   return (((x)&0x00000007)<<16)
SDMA_PKT_FENCE_HEADER_gcc_offset = 0 # macro
SDMA_PKT_FENCE_HEADER_gcc_mask = 0x00000001 # macro
SDMA_PKT_FENCE_HEADER_gcc_shift = 19 # macro
def SDMA_PKT_FENCE_HEADER_GCC(x):  # macro
   return (((x)&0x00000001)<<19)
SDMA_PKT_FENCE_HEADER_sys_offset = 0 # macro
SDMA_PKT_FENCE_HEADER_sys_mask = 0x00000001 # macro
SDMA_PKT_FENCE_HEADER_sys_shift = 20 # macro
def SDMA_PKT_FENCE_HEADER_SYS(x):  # macro
   return (((x)&0x00000001)<<20)
SDMA_PKT_FENCE_HEADER_snp_offset = 0 # macro
SDMA_PKT_FENCE_HEADER_snp_mask = 0x00000001 # macro
SDMA_PKT_FENCE_HEADER_snp_shift = 22 # macro
def SDMA_PKT_FENCE_HEADER_SNP(x):  # macro
   return (((x)&0x00000001)<<22)
SDMA_PKT_FENCE_HEADER_gpa_offset = 0 # macro
SDMA_PKT_FENCE_HEADER_gpa_mask = 0x00000001 # macro
SDMA_PKT_FENCE_HEADER_gpa_shift = 23 # macro
def SDMA_PKT_FENCE_HEADER_GPA(x):  # macro
   return (((x)&0x00000001)<<23)
SDMA_PKT_FENCE_HEADER_l2_policy_offset = 0 # macro
SDMA_PKT_FENCE_HEADER_l2_policy_mask = 0x00000003 # macro
SDMA_PKT_FENCE_HEADER_l2_policy_shift = 24 # macro
def SDMA_PKT_FENCE_HEADER_L2_POLICY(x):  # macro
   return (((x)&0x00000003)<<24)
SDMA_PKT_FENCE_ADDR_LO_addr_31_0_offset = 1 # macro
SDMA_PKT_FENCE_ADDR_LO_addr_31_0_mask = 0xFFFFFFFF # macro
SDMA_PKT_FENCE_ADDR_LO_addr_31_0_shift = 0 # macro
def SDMA_PKT_FENCE_ADDR_LO_ADDR_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_FENCE_ADDR_HI_addr_63_32_offset = 2 # macro
SDMA_PKT_FENCE_ADDR_HI_addr_63_32_mask = 0xFFFFFFFF # macro
SDMA_PKT_FENCE_ADDR_HI_addr_63_32_shift = 0 # macro
def SDMA_PKT_FENCE_ADDR_HI_ADDR_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_FENCE_DATA_data_offset = 3 # macro
SDMA_PKT_FENCE_DATA_data_mask = 0xFFFFFFFF # macro
SDMA_PKT_FENCE_DATA_data_shift = 0 # macro
def SDMA_PKT_FENCE_DATA_DATA(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_SRBM_WRITE_HEADER_op_offset = 0 # macro
SDMA_PKT_SRBM_WRITE_HEADER_op_mask = 0x000000FF # macro
SDMA_PKT_SRBM_WRITE_HEADER_op_shift = 0 # macro
def SDMA_PKT_SRBM_WRITE_HEADER_OP(x):  # macro
   return (((x)&0x000000FF)<<0)
SDMA_PKT_SRBM_WRITE_HEADER_sub_op_offset = 0 # macro
SDMA_PKT_SRBM_WRITE_HEADER_sub_op_mask = 0x000000FF # macro
SDMA_PKT_SRBM_WRITE_HEADER_sub_op_shift = 8 # macro
def SDMA_PKT_SRBM_WRITE_HEADER_SUB_OP(x):  # macro
   return (((x)&0x000000FF)<<8)
SDMA_PKT_SRBM_WRITE_HEADER_byte_en_offset = 0 # macro
SDMA_PKT_SRBM_WRITE_HEADER_byte_en_mask = 0x0000000F # macro
SDMA_PKT_SRBM_WRITE_HEADER_byte_en_shift = 28 # macro
def SDMA_PKT_SRBM_WRITE_HEADER_BYTE_EN(x):  # macro
   return (((x)&0x0000000F)<<28)
SDMA_PKT_SRBM_WRITE_ADDR_addr_offset = 1 # macro
SDMA_PKT_SRBM_WRITE_ADDR_addr_mask = 0x0003FFFF # macro
SDMA_PKT_SRBM_WRITE_ADDR_addr_shift = 0 # macro
def SDMA_PKT_SRBM_WRITE_ADDR_ADDR(x):  # macro
   return (((x)&0x0003FFFF)<<0)
SDMA_PKT_SRBM_WRITE_ADDR_apertureid_offset = 1 # macro
SDMA_PKT_SRBM_WRITE_ADDR_apertureid_mask = 0x00000FFF # macro
SDMA_PKT_SRBM_WRITE_ADDR_apertureid_shift = 20 # macro
def SDMA_PKT_SRBM_WRITE_ADDR_APERTUREID(x):  # macro
   return (((x)&0x00000FFF)<<20)
SDMA_PKT_SRBM_WRITE_DATA_data_offset = 2 # macro
SDMA_PKT_SRBM_WRITE_DATA_data_mask = 0xFFFFFFFF # macro
SDMA_PKT_SRBM_WRITE_DATA_data_shift = 0 # macro
def SDMA_PKT_SRBM_WRITE_DATA_DATA(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_PRE_EXE_HEADER_op_offset = 0 # macro
SDMA_PKT_PRE_EXE_HEADER_op_mask = 0x000000FF # macro
SDMA_PKT_PRE_EXE_HEADER_op_shift = 0 # macro
def SDMA_PKT_PRE_EXE_HEADER_OP(x):  # macro
   return (((x)&0x000000FF)<<0)
SDMA_PKT_PRE_EXE_HEADER_sub_op_offset = 0 # macro
SDMA_PKT_PRE_EXE_HEADER_sub_op_mask = 0x000000FF # macro
SDMA_PKT_PRE_EXE_HEADER_sub_op_shift = 8 # macro
def SDMA_PKT_PRE_EXE_HEADER_SUB_OP(x):  # macro
   return (((x)&0x000000FF)<<8)
SDMA_PKT_PRE_EXE_HEADER_dev_sel_offset = 0 # macro
SDMA_PKT_PRE_EXE_HEADER_dev_sel_mask = 0x000000FF # macro
SDMA_PKT_PRE_EXE_HEADER_dev_sel_shift = 16 # macro
def SDMA_PKT_PRE_EXE_HEADER_DEV_SEL(x):  # macro
   return (((x)&0x000000FF)<<16)
SDMA_PKT_PRE_EXE_EXEC_COUNT_exec_count_offset = 1 # macro
SDMA_PKT_PRE_EXE_EXEC_COUNT_exec_count_mask = 0x00003FFF # macro
SDMA_PKT_PRE_EXE_EXEC_COUNT_exec_count_shift = 0 # macro
def SDMA_PKT_PRE_EXE_EXEC_COUNT_EXEC_COUNT(x):  # macro
   return (((x)&0x00003FFF)<<0)
SDMA_PKT_COND_EXE_HEADER_op_offset = 0 # macro
SDMA_PKT_COND_EXE_HEADER_op_mask = 0x000000FF # macro
SDMA_PKT_COND_EXE_HEADER_op_shift = 0 # macro
def SDMA_PKT_COND_EXE_HEADER_OP(x):  # macro
   return (((x)&0x000000FF)<<0)
SDMA_PKT_COND_EXE_HEADER_sub_op_offset = 0 # macro
SDMA_PKT_COND_EXE_HEADER_sub_op_mask = 0x000000FF # macro
SDMA_PKT_COND_EXE_HEADER_sub_op_shift = 8 # macro
def SDMA_PKT_COND_EXE_HEADER_SUB_OP(x):  # macro
   return (((x)&0x000000FF)<<8)
SDMA_PKT_COND_EXE_ADDR_LO_addr_31_0_offset = 1 # macro
SDMA_PKT_COND_EXE_ADDR_LO_addr_31_0_mask = 0xFFFFFFFF # macro
SDMA_PKT_COND_EXE_ADDR_LO_addr_31_0_shift = 0 # macro
def SDMA_PKT_COND_EXE_ADDR_LO_ADDR_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COND_EXE_ADDR_HI_addr_63_32_offset = 2 # macro
SDMA_PKT_COND_EXE_ADDR_HI_addr_63_32_mask = 0xFFFFFFFF # macro
SDMA_PKT_COND_EXE_ADDR_HI_addr_63_32_shift = 0 # macro
def SDMA_PKT_COND_EXE_ADDR_HI_ADDR_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COND_EXE_REFERENCE_reference_offset = 3 # macro
SDMA_PKT_COND_EXE_REFERENCE_reference_mask = 0xFFFFFFFF # macro
SDMA_PKT_COND_EXE_REFERENCE_reference_shift = 0 # macro
def SDMA_PKT_COND_EXE_REFERENCE_REFERENCE(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_COND_EXE_EXEC_COUNT_exec_count_offset = 4 # macro
SDMA_PKT_COND_EXE_EXEC_COUNT_exec_count_mask = 0x00003FFF # macro
SDMA_PKT_COND_EXE_EXEC_COUNT_exec_count_shift = 0 # macro
def SDMA_PKT_COND_EXE_EXEC_COUNT_EXEC_COUNT(x):  # macro
   return (((x)&0x00003FFF)<<0)
SDMA_PKT_CONSTANT_FILL_HEADER_op_offset = 0 # macro
SDMA_PKT_CONSTANT_FILL_HEADER_op_mask = 0x000000FF # macro
SDMA_PKT_CONSTANT_FILL_HEADER_op_shift = 0 # macro
def SDMA_PKT_CONSTANT_FILL_HEADER_OP(x):  # macro
   return (((x)&0x000000FF)<<0)
SDMA_PKT_CONSTANT_FILL_HEADER_sub_op_offset = 0 # macro
SDMA_PKT_CONSTANT_FILL_HEADER_sub_op_mask = 0x000000FF # macro
SDMA_PKT_CONSTANT_FILL_HEADER_sub_op_shift = 8 # macro
def SDMA_PKT_CONSTANT_FILL_HEADER_SUB_OP(x):  # macro
   return (((x)&0x000000FF)<<8)
SDMA_PKT_CONSTANT_FILL_HEADER_sw_offset = 0 # macro
SDMA_PKT_CONSTANT_FILL_HEADER_sw_mask = 0x00000003 # macro
SDMA_PKT_CONSTANT_FILL_HEADER_sw_shift = 16 # macro
def SDMA_PKT_CONSTANT_FILL_HEADER_SW(x):  # macro
   return (((x)&0x00000003)<<16)
SDMA_PKT_CONSTANT_FILL_HEADER_fillsize_offset = 0 # macro
SDMA_PKT_CONSTANT_FILL_HEADER_fillsize_mask = 0x00000003 # macro
SDMA_PKT_CONSTANT_FILL_HEADER_fillsize_shift = 30 # macro
def SDMA_PKT_CONSTANT_FILL_HEADER_FILLSIZE(x):  # macro
   return (((x)&0x00000003)<<30)
SDMA_PKT_CONSTANT_FILL_DST_ADDR_LO_dst_addr_31_0_offset = 1 # macro
SDMA_PKT_CONSTANT_FILL_DST_ADDR_LO_dst_addr_31_0_mask = 0xFFFFFFFF # macro
SDMA_PKT_CONSTANT_FILL_DST_ADDR_LO_dst_addr_31_0_shift = 0 # macro
def SDMA_PKT_CONSTANT_FILL_DST_ADDR_LO_DST_ADDR_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_CONSTANT_FILL_DST_ADDR_HI_dst_addr_63_32_offset = 2 # macro
SDMA_PKT_CONSTANT_FILL_DST_ADDR_HI_dst_addr_63_32_mask = 0xFFFFFFFF # macro
SDMA_PKT_CONSTANT_FILL_DST_ADDR_HI_dst_addr_63_32_shift = 0 # macro
def SDMA_PKT_CONSTANT_FILL_DST_ADDR_HI_DST_ADDR_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_CONSTANT_FILL_DATA_src_data_31_0_offset = 3 # macro
SDMA_PKT_CONSTANT_FILL_DATA_src_data_31_0_mask = 0xFFFFFFFF # macro
SDMA_PKT_CONSTANT_FILL_DATA_src_data_31_0_shift = 0 # macro
def SDMA_PKT_CONSTANT_FILL_DATA_SRC_DATA_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_CONSTANT_FILL_COUNT_count_offset = 4 # macro
SDMA_PKT_CONSTANT_FILL_COUNT_count_mask = 0x003FFFFF # macro
SDMA_PKT_CONSTANT_FILL_COUNT_count_shift = 0 # macro
def SDMA_PKT_CONSTANT_FILL_COUNT_COUNT(x):  # macro
   return (((x)&0x003FFFFF)<<0)
SDMA_PKT_DATA_FILL_MULTI_HEADER_op_offset = 0 # macro
SDMA_PKT_DATA_FILL_MULTI_HEADER_op_mask = 0x000000FF # macro
SDMA_PKT_DATA_FILL_MULTI_HEADER_op_shift = 0 # macro
def SDMA_PKT_DATA_FILL_MULTI_HEADER_OP(x):  # macro
   return (((x)&0x000000FF)<<0)
SDMA_PKT_DATA_FILL_MULTI_HEADER_sub_op_offset = 0 # macro
SDMA_PKT_DATA_FILL_MULTI_HEADER_sub_op_mask = 0x000000FF # macro
SDMA_PKT_DATA_FILL_MULTI_HEADER_sub_op_shift = 8 # macro
def SDMA_PKT_DATA_FILL_MULTI_HEADER_SUB_OP(x):  # macro
   return (((x)&0x000000FF)<<8)
SDMA_PKT_DATA_FILL_MULTI_HEADER_memlog_clr_offset = 0 # macro
SDMA_PKT_DATA_FILL_MULTI_HEADER_memlog_clr_mask = 0x00000001 # macro
SDMA_PKT_DATA_FILL_MULTI_HEADER_memlog_clr_shift = 31 # macro
def SDMA_PKT_DATA_FILL_MULTI_HEADER_MEMLOG_CLR(x):  # macro
   return (((x)&0x00000001)<<31)
SDMA_PKT_DATA_FILL_MULTI_BYTE_STRIDE_byte_stride_offset = 1 # macro
SDMA_PKT_DATA_FILL_MULTI_BYTE_STRIDE_byte_stride_mask = 0xFFFFFFFF # macro
SDMA_PKT_DATA_FILL_MULTI_BYTE_STRIDE_byte_stride_shift = 0 # macro
def SDMA_PKT_DATA_FILL_MULTI_BYTE_STRIDE_BYTE_STRIDE(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_DATA_FILL_MULTI_DMA_COUNT_dma_count_offset = 2 # macro
SDMA_PKT_DATA_FILL_MULTI_DMA_COUNT_dma_count_mask = 0xFFFFFFFF # macro
SDMA_PKT_DATA_FILL_MULTI_DMA_COUNT_dma_count_shift = 0 # macro
def SDMA_PKT_DATA_FILL_MULTI_DMA_COUNT_DMA_COUNT(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_DATA_FILL_MULTI_DST_ADDR_LO_dst_addr_31_0_offset = 3 # macro
SDMA_PKT_DATA_FILL_MULTI_DST_ADDR_LO_dst_addr_31_0_mask = 0xFFFFFFFF # macro
SDMA_PKT_DATA_FILL_MULTI_DST_ADDR_LO_dst_addr_31_0_shift = 0 # macro
def SDMA_PKT_DATA_FILL_MULTI_DST_ADDR_LO_DST_ADDR_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_DATA_FILL_MULTI_DST_ADDR_HI_dst_addr_63_32_offset = 4 # macro
SDMA_PKT_DATA_FILL_MULTI_DST_ADDR_HI_dst_addr_63_32_mask = 0xFFFFFFFF # macro
SDMA_PKT_DATA_FILL_MULTI_DST_ADDR_HI_dst_addr_63_32_shift = 0 # macro
def SDMA_PKT_DATA_FILL_MULTI_DST_ADDR_HI_DST_ADDR_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_DATA_FILL_MULTI_BYTE_COUNT_count_offset = 5 # macro
SDMA_PKT_DATA_FILL_MULTI_BYTE_COUNT_count_mask = 0x03FFFFFF # macro
SDMA_PKT_DATA_FILL_MULTI_BYTE_COUNT_count_shift = 0 # macro
def SDMA_PKT_DATA_FILL_MULTI_BYTE_COUNT_COUNT(x):  # macro
   return (((x)&0x03FFFFFF)<<0)
SDMA_PKT_POLL_REGMEM_HEADER_op_offset = 0 # macro
SDMA_PKT_POLL_REGMEM_HEADER_op_mask = 0x000000FF # macro
SDMA_PKT_POLL_REGMEM_HEADER_op_shift = 0 # macro
def SDMA_PKT_POLL_REGMEM_HEADER_OP(x):  # macro
   return (((x)&0x000000FF)<<0)
SDMA_PKT_POLL_REGMEM_HEADER_sub_op_offset = 0 # macro
SDMA_PKT_POLL_REGMEM_HEADER_sub_op_mask = 0x000000FF # macro
SDMA_PKT_POLL_REGMEM_HEADER_sub_op_shift = 8 # macro
def SDMA_PKT_POLL_REGMEM_HEADER_SUB_OP(x):  # macro
   return (((x)&0x000000FF)<<8)
SDMA_PKT_POLL_REGMEM_HEADER_hdp_flush_offset = 0 # macro
SDMA_PKT_POLL_REGMEM_HEADER_hdp_flush_mask = 0x00000001 # macro
SDMA_PKT_POLL_REGMEM_HEADER_hdp_flush_shift = 26 # macro
def SDMA_PKT_POLL_REGMEM_HEADER_HDP_FLUSH(x):  # macro
   return (((x)&0x00000001)<<26)
SDMA_PKT_POLL_REGMEM_HEADER_func_offset = 0 # macro
SDMA_PKT_POLL_REGMEM_HEADER_func_mask = 0x00000007 # macro
SDMA_PKT_POLL_REGMEM_HEADER_func_shift = 28 # macro
def SDMA_PKT_POLL_REGMEM_HEADER_FUNC(x):  # macro
   return (((x)&0x00000007)<<28)
SDMA_PKT_POLL_REGMEM_HEADER_mem_poll_offset = 0 # macro
SDMA_PKT_POLL_REGMEM_HEADER_mem_poll_mask = 0x00000001 # macro
SDMA_PKT_POLL_REGMEM_HEADER_mem_poll_shift = 31 # macro
def SDMA_PKT_POLL_REGMEM_HEADER_MEM_POLL(x):  # macro
   return (((x)&0x00000001)<<31)
SDMA_PKT_POLL_REGMEM_ADDR_LO_addr_31_0_offset = 1 # macro
SDMA_PKT_POLL_REGMEM_ADDR_LO_addr_31_0_mask = 0xFFFFFFFF # macro
SDMA_PKT_POLL_REGMEM_ADDR_LO_addr_31_0_shift = 0 # macro
def SDMA_PKT_POLL_REGMEM_ADDR_LO_ADDR_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_POLL_REGMEM_ADDR_HI_addr_63_32_offset = 2 # macro
SDMA_PKT_POLL_REGMEM_ADDR_HI_addr_63_32_mask = 0xFFFFFFFF # macro
SDMA_PKT_POLL_REGMEM_ADDR_HI_addr_63_32_shift = 0 # macro
def SDMA_PKT_POLL_REGMEM_ADDR_HI_ADDR_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_POLL_REGMEM_VALUE_value_offset = 3 # macro
SDMA_PKT_POLL_REGMEM_VALUE_value_mask = 0xFFFFFFFF # macro
SDMA_PKT_POLL_REGMEM_VALUE_value_shift = 0 # macro
def SDMA_PKT_POLL_REGMEM_VALUE_VALUE(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_POLL_REGMEM_MASK_mask_offset = 4 # macro
SDMA_PKT_POLL_REGMEM_MASK_mask_mask = 0xFFFFFFFF # macro
SDMA_PKT_POLL_REGMEM_MASK_mask_shift = 0 # macro
def SDMA_PKT_POLL_REGMEM_MASK_MASK(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_POLL_REGMEM_DW5_interval_offset = 5 # macro
SDMA_PKT_POLL_REGMEM_DW5_interval_mask = 0x0000FFFF # macro
SDMA_PKT_POLL_REGMEM_DW5_interval_shift = 0 # macro
def SDMA_PKT_POLL_REGMEM_DW5_INTERVAL(x):  # macro
   return (((x)&0x0000FFFF)<<0)
SDMA_PKT_POLL_REGMEM_DW5_retry_count_offset = 5 # macro
SDMA_PKT_POLL_REGMEM_DW5_retry_count_mask = 0x00000FFF # macro
SDMA_PKT_POLL_REGMEM_DW5_retry_count_shift = 16 # macro
def SDMA_PKT_POLL_REGMEM_DW5_RETRY_COUNT(x):  # macro
   return (((x)&0x00000FFF)<<16)
SDMA_PKT_POLL_REG_WRITE_MEM_HEADER_op_offset = 0 # macro
SDMA_PKT_POLL_REG_WRITE_MEM_HEADER_op_mask = 0x000000FF # macro
SDMA_PKT_POLL_REG_WRITE_MEM_HEADER_op_shift = 0 # macro
def SDMA_PKT_POLL_REG_WRITE_MEM_HEADER_OP(x):  # macro
   return (((x)&0x000000FF)<<0)
SDMA_PKT_POLL_REG_WRITE_MEM_HEADER_sub_op_offset = 0 # macro
SDMA_PKT_POLL_REG_WRITE_MEM_HEADER_sub_op_mask = 0x000000FF # macro
SDMA_PKT_POLL_REG_WRITE_MEM_HEADER_sub_op_shift = 8 # macro
def SDMA_PKT_POLL_REG_WRITE_MEM_HEADER_SUB_OP(x):  # macro
   return (((x)&0x000000FF)<<8)
SDMA_PKT_POLL_REG_WRITE_MEM_SRC_ADDR_addr_31_2_offset = 1 # macro
SDMA_PKT_POLL_REG_WRITE_MEM_SRC_ADDR_addr_31_2_mask = 0x3FFFFFFF # macro
SDMA_PKT_POLL_REG_WRITE_MEM_SRC_ADDR_addr_31_2_shift = 2 # macro
def SDMA_PKT_POLL_REG_WRITE_MEM_SRC_ADDR_ADDR_31_2(x):  # macro
   return (((x)&0x3FFFFFFF)<<2)
SDMA_PKT_POLL_REG_WRITE_MEM_DST_ADDR_LO_addr_31_0_offset = 2 # macro
SDMA_PKT_POLL_REG_WRITE_MEM_DST_ADDR_LO_addr_31_0_mask = 0xFFFFFFFF # macro
SDMA_PKT_POLL_REG_WRITE_MEM_DST_ADDR_LO_addr_31_0_shift = 0 # macro
def SDMA_PKT_POLL_REG_WRITE_MEM_DST_ADDR_LO_ADDR_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_POLL_REG_WRITE_MEM_DST_ADDR_HI_addr_63_32_offset = 3 # macro
SDMA_PKT_POLL_REG_WRITE_MEM_DST_ADDR_HI_addr_63_32_mask = 0xFFFFFFFF # macro
SDMA_PKT_POLL_REG_WRITE_MEM_DST_ADDR_HI_addr_63_32_shift = 0 # macro
def SDMA_PKT_POLL_REG_WRITE_MEM_DST_ADDR_HI_ADDR_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_op_offset = 0 # macro
SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_op_mask = 0x000000FF # macro
SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_op_shift = 0 # macro
def SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_OP(x):  # macro
   return (((x)&0x000000FF)<<0)
SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_sub_op_offset = 0 # macro
SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_sub_op_mask = 0x000000FF # macro
SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_sub_op_shift = 8 # macro
def SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_SUB_OP(x):  # macro
   return (((x)&0x000000FF)<<8)
SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_ea_offset = 0 # macro
SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_ea_mask = 0x00000003 # macro
SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_ea_shift = 16 # macro
def SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_EA(x):  # macro
   return (((x)&0x00000003)<<16)
SDMA_PKT_POLL_DBIT_WRITE_MEM_DST_ADDR_LO_addr_31_0_offset = 1 # macro
SDMA_PKT_POLL_DBIT_WRITE_MEM_DST_ADDR_LO_addr_31_0_mask = 0xFFFFFFFF # macro
SDMA_PKT_POLL_DBIT_WRITE_MEM_DST_ADDR_LO_addr_31_0_shift = 0 # macro
def SDMA_PKT_POLL_DBIT_WRITE_MEM_DST_ADDR_LO_ADDR_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_POLL_DBIT_WRITE_MEM_DST_ADDR_HI_addr_63_32_offset = 2 # macro
SDMA_PKT_POLL_DBIT_WRITE_MEM_DST_ADDR_HI_addr_63_32_mask = 0xFFFFFFFF # macro
SDMA_PKT_POLL_DBIT_WRITE_MEM_DST_ADDR_HI_addr_63_32_shift = 0 # macro
def SDMA_PKT_POLL_DBIT_WRITE_MEM_DST_ADDR_HI_ADDR_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_POLL_DBIT_WRITE_MEM_START_PAGE_addr_31_4_offset = 3 # macro
SDMA_PKT_POLL_DBIT_WRITE_MEM_START_PAGE_addr_31_4_mask = 0x0FFFFFFF # macro
SDMA_PKT_POLL_DBIT_WRITE_MEM_START_PAGE_addr_31_4_shift = 4 # macro
def SDMA_PKT_POLL_DBIT_WRITE_MEM_START_PAGE_ADDR_31_4(x):  # macro
   return (((x)&0x0FFFFFFF)<<4)
SDMA_PKT_POLL_DBIT_WRITE_MEM_PAGE_NUM_page_num_31_0_offset = 4 # macro
SDMA_PKT_POLL_DBIT_WRITE_MEM_PAGE_NUM_page_num_31_0_mask = 0xFFFFFFFF # macro
SDMA_PKT_POLL_DBIT_WRITE_MEM_PAGE_NUM_page_num_31_0_shift = 0 # macro
def SDMA_PKT_POLL_DBIT_WRITE_MEM_PAGE_NUM_PAGE_NUM_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_POLL_MEM_VERIFY_HEADER_op_offset = 0 # macro
SDMA_PKT_POLL_MEM_VERIFY_HEADER_op_mask = 0x000000FF # macro
SDMA_PKT_POLL_MEM_VERIFY_HEADER_op_shift = 0 # macro
def SDMA_PKT_POLL_MEM_VERIFY_HEADER_OP(x):  # macro
   return (((x)&0x000000FF)<<0)
SDMA_PKT_POLL_MEM_VERIFY_HEADER_sub_op_offset = 0 # macro
SDMA_PKT_POLL_MEM_VERIFY_HEADER_sub_op_mask = 0x000000FF # macro
SDMA_PKT_POLL_MEM_VERIFY_HEADER_sub_op_shift = 8 # macro
def SDMA_PKT_POLL_MEM_VERIFY_HEADER_SUB_OP(x):  # macro
   return (((x)&0x000000FF)<<8)
SDMA_PKT_POLL_MEM_VERIFY_HEADER_mode_offset = 0 # macro
SDMA_PKT_POLL_MEM_VERIFY_HEADER_mode_mask = 0x00000001 # macro
SDMA_PKT_POLL_MEM_VERIFY_HEADER_mode_shift = 31 # macro
def SDMA_PKT_POLL_MEM_VERIFY_HEADER_MODE(x):  # macro
   return (((x)&0x00000001)<<31)
SDMA_PKT_POLL_MEM_VERIFY_PATTERN_pattern_offset = 1 # macro
SDMA_PKT_POLL_MEM_VERIFY_PATTERN_pattern_mask = 0xFFFFFFFF # macro
SDMA_PKT_POLL_MEM_VERIFY_PATTERN_pattern_shift = 0 # macro
def SDMA_PKT_POLL_MEM_VERIFY_PATTERN_PATTERN(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_START_LO_cmp0_start_31_0_offset = 2 # macro
SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_START_LO_cmp0_start_31_0_mask = 0xFFFFFFFF # macro
SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_START_LO_cmp0_start_31_0_shift = 0 # macro
def SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_START_LO_CMP0_START_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_START_HI_cmp0_start_63_32_offset = 3 # macro
SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_START_HI_cmp0_start_63_32_mask = 0xFFFFFFFF # macro
SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_START_HI_cmp0_start_63_32_shift = 0 # macro
def SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_START_HI_CMP0_START_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_END_LO_cmp1_end_31_0_offset = 4 # macro
SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_END_LO_cmp1_end_31_0_mask = 0xFFFFFFFF # macro
SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_END_LO_cmp1_end_31_0_shift = 0 # macro
def SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_END_LO_CMP1_END_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_END_HI_cmp1_end_63_32_offset = 5 # macro
SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_END_HI_cmp1_end_63_32_mask = 0xFFFFFFFF # macro
SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_END_HI_cmp1_end_63_32_shift = 0 # macro
def SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_END_HI_CMP1_END_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_START_LO_cmp1_start_31_0_offset = 6 # macro
SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_START_LO_cmp1_start_31_0_mask = 0xFFFFFFFF # macro
SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_START_LO_cmp1_start_31_0_shift = 0 # macro
def SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_START_LO_CMP1_START_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_START_HI_cmp1_start_63_32_offset = 7 # macro
SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_START_HI_cmp1_start_63_32_mask = 0xFFFFFFFF # macro
SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_START_HI_cmp1_start_63_32_shift = 0 # macro
def SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_START_HI_CMP1_START_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_END_LO_cmp1_end_31_0_offset = 8 # macro
SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_END_LO_cmp1_end_31_0_mask = 0xFFFFFFFF # macro
SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_END_LO_cmp1_end_31_0_shift = 0 # macro
def SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_END_LO_CMP1_END_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_END_HI_cmp1_end_63_32_offset = 9 # macro
SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_END_HI_cmp1_end_63_32_mask = 0xFFFFFFFF # macro
SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_END_HI_cmp1_end_63_32_shift = 0 # macro
def SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_END_HI_CMP1_END_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_POLL_MEM_VERIFY_REC_ADDR_LO_rec_31_0_offset = 10 # macro
SDMA_PKT_POLL_MEM_VERIFY_REC_ADDR_LO_rec_31_0_mask = 0xFFFFFFFF # macro
SDMA_PKT_POLL_MEM_VERIFY_REC_ADDR_LO_rec_31_0_shift = 0 # macro
def SDMA_PKT_POLL_MEM_VERIFY_REC_ADDR_LO_REC_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_POLL_MEM_VERIFY_REC_ADDR_HI_rec_63_32_offset = 11 # macro
SDMA_PKT_POLL_MEM_VERIFY_REC_ADDR_HI_rec_63_32_mask = 0xFFFFFFFF # macro
SDMA_PKT_POLL_MEM_VERIFY_REC_ADDR_HI_rec_63_32_shift = 0 # macro
def SDMA_PKT_POLL_MEM_VERIFY_REC_ADDR_HI_REC_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_POLL_MEM_VERIFY_RESERVED_reserved_offset = 12 # macro
SDMA_PKT_POLL_MEM_VERIFY_RESERVED_reserved_mask = 0xFFFFFFFF # macro
SDMA_PKT_POLL_MEM_VERIFY_RESERVED_reserved_shift = 0 # macro
def SDMA_PKT_POLL_MEM_VERIFY_RESERVED_RESERVED(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_VM_INVALIDATION_HEADER_op_offset = 0 # macro
SDMA_PKT_VM_INVALIDATION_HEADER_op_mask = 0x000000FF # macro
SDMA_PKT_VM_INVALIDATION_HEADER_op_shift = 0 # macro
def SDMA_PKT_VM_INVALIDATION_HEADER_OP(x):  # macro
   return (((x)&0x000000FF)<<0)
SDMA_PKT_VM_INVALIDATION_HEADER_sub_op_offset = 0 # macro
SDMA_PKT_VM_INVALIDATION_HEADER_sub_op_mask = 0x000000FF # macro
SDMA_PKT_VM_INVALIDATION_HEADER_sub_op_shift = 8 # macro
def SDMA_PKT_VM_INVALIDATION_HEADER_SUB_OP(x):  # macro
   return (((x)&0x000000FF)<<8)
SDMA_PKT_VM_INVALIDATION_HEADER_gfx_eng_id_offset = 0 # macro
SDMA_PKT_VM_INVALIDATION_HEADER_gfx_eng_id_mask = 0x0000001F # macro
SDMA_PKT_VM_INVALIDATION_HEADER_gfx_eng_id_shift = 16 # macro
def SDMA_PKT_VM_INVALIDATION_HEADER_GFX_ENG_ID(x):  # macro
   return (((x)&0x0000001F)<<16)
SDMA_PKT_VM_INVALIDATION_HEADER_mm_eng_id_offset = 0 # macro
SDMA_PKT_VM_INVALIDATION_HEADER_mm_eng_id_mask = 0x0000001F # macro
SDMA_PKT_VM_INVALIDATION_HEADER_mm_eng_id_shift = 24 # macro
def SDMA_PKT_VM_INVALIDATION_HEADER_MM_ENG_ID(x):  # macro
   return (((x)&0x0000001F)<<24)
SDMA_PKT_VM_INVALIDATION_INVALIDATEREQ_invalidatereq_offset = 1 # macro
SDMA_PKT_VM_INVALIDATION_INVALIDATEREQ_invalidatereq_mask = 0xFFFFFFFF # macro
SDMA_PKT_VM_INVALIDATION_INVALIDATEREQ_invalidatereq_shift = 0 # macro
def SDMA_PKT_VM_INVALIDATION_INVALIDATEREQ_INVALIDATEREQ(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_VM_INVALIDATION_ADDRESSRANGELO_addressrangelo_offset = 2 # macro
SDMA_PKT_VM_INVALIDATION_ADDRESSRANGELO_addressrangelo_mask = 0xFFFFFFFF # macro
SDMA_PKT_VM_INVALIDATION_ADDRESSRANGELO_addressrangelo_shift = 0 # macro
def SDMA_PKT_VM_INVALIDATION_ADDRESSRANGELO_ADDRESSRANGELO(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_VM_INVALIDATION_ADDRESSRANGEHI_invalidateack_offset = 3 # macro
SDMA_PKT_VM_INVALIDATION_ADDRESSRANGEHI_invalidateack_mask = 0x0000FFFF # macro
SDMA_PKT_VM_INVALIDATION_ADDRESSRANGEHI_invalidateack_shift = 0 # macro
def SDMA_PKT_VM_INVALIDATION_ADDRESSRANGEHI_INVALIDATEACK(x):  # macro
   return (((x)&0x0000FFFF)<<0)
SDMA_PKT_VM_INVALIDATION_ADDRESSRANGEHI_addressrangehi_offset = 3 # macro
SDMA_PKT_VM_INVALIDATION_ADDRESSRANGEHI_addressrangehi_mask = 0x0000001F # macro
SDMA_PKT_VM_INVALIDATION_ADDRESSRANGEHI_addressrangehi_shift = 16 # macro
def SDMA_PKT_VM_INVALIDATION_ADDRESSRANGEHI_ADDRESSRANGEHI(x):  # macro
   return (((x)&0x0000001F)<<16)
SDMA_PKT_VM_INVALIDATION_ADDRESSRANGEHI_reserved_offset = 3 # macro
SDMA_PKT_VM_INVALIDATION_ADDRESSRANGEHI_reserved_mask = 0x000001FF # macro
SDMA_PKT_VM_INVALIDATION_ADDRESSRANGEHI_reserved_shift = 23 # macro
def SDMA_PKT_VM_INVALIDATION_ADDRESSRANGEHI_RESERVED(x):  # macro
   return (((x)&0x000001FF)<<23)
SDMA_PKT_ATOMIC_HEADER_op_offset = 0 # macro
SDMA_PKT_ATOMIC_HEADER_op_mask = 0x000000FF # macro
SDMA_PKT_ATOMIC_HEADER_op_shift = 0 # macro
def SDMA_PKT_ATOMIC_HEADER_OP(x):  # macro
   return (((x)&0x000000FF)<<0)
SDMA_PKT_ATOMIC_HEADER_loop_offset = 0 # macro
SDMA_PKT_ATOMIC_HEADER_loop_mask = 0x00000001 # macro
SDMA_PKT_ATOMIC_HEADER_loop_shift = 16 # macro
def SDMA_PKT_ATOMIC_HEADER_LOOP(x):  # macro
   return (((x)&0x00000001)<<16)
SDMA_PKT_ATOMIC_HEADER_tmz_offset = 0 # macro
SDMA_PKT_ATOMIC_HEADER_tmz_mask = 0x00000001 # macro
SDMA_PKT_ATOMIC_HEADER_tmz_shift = 18 # macro
def SDMA_PKT_ATOMIC_HEADER_TMZ(x):  # macro
   return (((x)&0x00000001)<<18)
SDMA_PKT_ATOMIC_HEADER_atomic_op_offset = 0 # macro
SDMA_PKT_ATOMIC_HEADER_atomic_op_mask = 0x0000007F # macro
SDMA_PKT_ATOMIC_HEADER_atomic_op_shift = 25 # macro
def SDMA_PKT_ATOMIC_HEADER_ATOMIC_OP(x):  # macro
   return (((x)&0x0000007F)<<25)
SDMA_PKT_ATOMIC_ADDR_LO_addr_31_0_offset = 1 # macro
SDMA_PKT_ATOMIC_ADDR_LO_addr_31_0_mask = 0xFFFFFFFF # macro
SDMA_PKT_ATOMIC_ADDR_LO_addr_31_0_shift = 0 # macro
def SDMA_PKT_ATOMIC_ADDR_LO_ADDR_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_ATOMIC_ADDR_HI_addr_63_32_offset = 2 # macro
SDMA_PKT_ATOMIC_ADDR_HI_addr_63_32_mask = 0xFFFFFFFF # macro
SDMA_PKT_ATOMIC_ADDR_HI_addr_63_32_shift = 0 # macro
def SDMA_PKT_ATOMIC_ADDR_HI_ADDR_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_ATOMIC_SRC_DATA_LO_src_data_31_0_offset = 3 # macro
SDMA_PKT_ATOMIC_SRC_DATA_LO_src_data_31_0_mask = 0xFFFFFFFF # macro
SDMA_PKT_ATOMIC_SRC_DATA_LO_src_data_31_0_shift = 0 # macro
def SDMA_PKT_ATOMIC_SRC_DATA_LO_SRC_DATA_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_ATOMIC_SRC_DATA_HI_src_data_63_32_offset = 4 # macro
SDMA_PKT_ATOMIC_SRC_DATA_HI_src_data_63_32_mask = 0xFFFFFFFF # macro
SDMA_PKT_ATOMIC_SRC_DATA_HI_src_data_63_32_shift = 0 # macro
def SDMA_PKT_ATOMIC_SRC_DATA_HI_SRC_DATA_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_ATOMIC_CMP_DATA_LO_cmp_data_31_0_offset = 5 # macro
SDMA_PKT_ATOMIC_CMP_DATA_LO_cmp_data_31_0_mask = 0xFFFFFFFF # macro
SDMA_PKT_ATOMIC_CMP_DATA_LO_cmp_data_31_0_shift = 0 # macro
def SDMA_PKT_ATOMIC_CMP_DATA_LO_CMP_DATA_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_ATOMIC_CMP_DATA_HI_cmp_data_63_32_offset = 6 # macro
SDMA_PKT_ATOMIC_CMP_DATA_HI_cmp_data_63_32_mask = 0xFFFFFFFF # macro
SDMA_PKT_ATOMIC_CMP_DATA_HI_cmp_data_63_32_shift = 0 # macro
def SDMA_PKT_ATOMIC_CMP_DATA_HI_CMP_DATA_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_ATOMIC_LOOP_INTERVAL_loop_interval_offset = 7 # macro
SDMA_PKT_ATOMIC_LOOP_INTERVAL_loop_interval_mask = 0x00001FFF # macro
SDMA_PKT_ATOMIC_LOOP_INTERVAL_loop_interval_shift = 0 # macro
def SDMA_PKT_ATOMIC_LOOP_INTERVAL_LOOP_INTERVAL(x):  # macro
   return (((x)&0x00001FFF)<<0)
SDMA_PKT_TIMESTAMP_SET_HEADER_op_offset = 0 # macro
SDMA_PKT_TIMESTAMP_SET_HEADER_op_mask = 0x000000FF # macro
SDMA_PKT_TIMESTAMP_SET_HEADER_op_shift = 0 # macro
def SDMA_PKT_TIMESTAMP_SET_HEADER_OP(x):  # macro
   return (((x)&0x000000FF)<<0)
SDMA_PKT_TIMESTAMP_SET_HEADER_sub_op_offset = 0 # macro
SDMA_PKT_TIMESTAMP_SET_HEADER_sub_op_mask = 0x000000FF # macro
SDMA_PKT_TIMESTAMP_SET_HEADER_sub_op_shift = 8 # macro
def SDMA_PKT_TIMESTAMP_SET_HEADER_SUB_OP(x):  # macro
   return (((x)&0x000000FF)<<8)
SDMA_PKT_TIMESTAMP_SET_INIT_DATA_LO_init_data_31_0_offset = 1 # macro
SDMA_PKT_TIMESTAMP_SET_INIT_DATA_LO_init_data_31_0_mask = 0xFFFFFFFF # macro
SDMA_PKT_TIMESTAMP_SET_INIT_DATA_LO_init_data_31_0_shift = 0 # macro
def SDMA_PKT_TIMESTAMP_SET_INIT_DATA_LO_INIT_DATA_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_TIMESTAMP_SET_INIT_DATA_HI_init_data_63_32_offset = 2 # macro
SDMA_PKT_TIMESTAMP_SET_INIT_DATA_HI_init_data_63_32_mask = 0xFFFFFFFF # macro
SDMA_PKT_TIMESTAMP_SET_INIT_DATA_HI_init_data_63_32_shift = 0 # macro
def SDMA_PKT_TIMESTAMP_SET_INIT_DATA_HI_INIT_DATA_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_TIMESTAMP_GET_HEADER_op_offset = 0 # macro
SDMA_PKT_TIMESTAMP_GET_HEADER_op_mask = 0x000000FF # macro
SDMA_PKT_TIMESTAMP_GET_HEADER_op_shift = 0 # macro
def SDMA_PKT_TIMESTAMP_GET_HEADER_OP(x):  # macro
   return (((x)&0x000000FF)<<0)
SDMA_PKT_TIMESTAMP_GET_HEADER_sub_op_offset = 0 # macro
SDMA_PKT_TIMESTAMP_GET_HEADER_sub_op_mask = 0x000000FF # macro
SDMA_PKT_TIMESTAMP_GET_HEADER_sub_op_shift = 8 # macro
def SDMA_PKT_TIMESTAMP_GET_HEADER_SUB_OP(x):  # macro
   return (((x)&0x000000FF)<<8)
SDMA_PKT_TIMESTAMP_GET_WRITE_ADDR_LO_write_addr_31_3_offset = 1 # macro
SDMA_PKT_TIMESTAMP_GET_WRITE_ADDR_LO_write_addr_31_3_mask = 0x1FFFFFFF # macro
SDMA_PKT_TIMESTAMP_GET_WRITE_ADDR_LO_write_addr_31_3_shift = 3 # macro
def SDMA_PKT_TIMESTAMP_GET_WRITE_ADDR_LO_WRITE_ADDR_31_3(x):  # macro
   return (((x)&0x1FFFFFFF)<<3)
SDMA_PKT_TIMESTAMP_GET_WRITE_ADDR_HI_write_addr_63_32_offset = 2 # macro
SDMA_PKT_TIMESTAMP_GET_WRITE_ADDR_HI_write_addr_63_32_mask = 0xFFFFFFFF # macro
SDMA_PKT_TIMESTAMP_GET_WRITE_ADDR_HI_write_addr_63_32_shift = 0 # macro
def SDMA_PKT_TIMESTAMP_GET_WRITE_ADDR_HI_WRITE_ADDR_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_op_offset = 0 # macro
SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_op_mask = 0x000000FF # macro
SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_op_shift = 0 # macro
def SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_OP(x):  # macro
   return (((x)&0x000000FF)<<0)
SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_sub_op_offset = 0 # macro
SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_sub_op_mask = 0x000000FF # macro
SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_sub_op_shift = 8 # macro
def SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_SUB_OP(x):  # macro
   return (((x)&0x000000FF)<<8)
SDMA_PKT_TIMESTAMP_GET_GLOBAL_WRITE_ADDR_LO_write_addr_31_3_offset = 1 # macro
SDMA_PKT_TIMESTAMP_GET_GLOBAL_WRITE_ADDR_LO_write_addr_31_3_mask = 0x1FFFFFFF # macro
SDMA_PKT_TIMESTAMP_GET_GLOBAL_WRITE_ADDR_LO_write_addr_31_3_shift = 3 # macro
def SDMA_PKT_TIMESTAMP_GET_GLOBAL_WRITE_ADDR_LO_WRITE_ADDR_31_3(x):  # macro
   return (((x)&0x1FFFFFFF)<<3)
SDMA_PKT_TIMESTAMP_GET_GLOBAL_WRITE_ADDR_HI_write_addr_63_32_offset = 2 # macro
SDMA_PKT_TIMESTAMP_GET_GLOBAL_WRITE_ADDR_HI_write_addr_63_32_mask = 0xFFFFFFFF # macro
SDMA_PKT_TIMESTAMP_GET_GLOBAL_WRITE_ADDR_HI_write_addr_63_32_shift = 0 # macro
def SDMA_PKT_TIMESTAMP_GET_GLOBAL_WRITE_ADDR_HI_WRITE_ADDR_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_PKT_TRAP_HEADER_op_offset = 0 # macro
SDMA_PKT_TRAP_HEADER_op_mask = 0x000000FF # macro
SDMA_PKT_TRAP_HEADER_op_shift = 0 # macro
def SDMA_PKT_TRAP_HEADER_OP(x):  # macro
   return (((x)&0x000000FF)<<0)
SDMA_PKT_TRAP_HEADER_sub_op_offset = 0 # macro
SDMA_PKT_TRAP_HEADER_sub_op_mask = 0x000000FF # macro
SDMA_PKT_TRAP_HEADER_sub_op_shift = 8 # macro
def SDMA_PKT_TRAP_HEADER_SUB_OP(x):  # macro
   return (((x)&0x000000FF)<<8)
SDMA_PKT_TRAP_INT_CONTEXT_int_context_offset = 1 # macro
SDMA_PKT_TRAP_INT_CONTEXT_int_context_mask = 0x0FFFFFFF # macro
SDMA_PKT_TRAP_INT_CONTEXT_int_context_shift = 0 # macro
def SDMA_PKT_TRAP_INT_CONTEXT_INT_CONTEXT(x):  # macro
   return (((x)&0x0FFFFFFF)<<0)
SDMA_PKT_DUMMY_TRAP_HEADER_op_offset = 0 # macro
SDMA_PKT_DUMMY_TRAP_HEADER_op_mask = 0x000000FF # macro
SDMA_PKT_DUMMY_TRAP_HEADER_op_shift = 0 # macro
def SDMA_PKT_DUMMY_TRAP_HEADER_OP(x):  # macro
   return (((x)&0x000000FF)<<0)
SDMA_PKT_DUMMY_TRAP_HEADER_sub_op_offset = 0 # macro
SDMA_PKT_DUMMY_TRAP_HEADER_sub_op_mask = 0x000000FF # macro
SDMA_PKT_DUMMY_TRAP_HEADER_sub_op_shift = 8 # macro
def SDMA_PKT_DUMMY_TRAP_HEADER_SUB_OP(x):  # macro
   return (((x)&0x000000FF)<<8)
SDMA_PKT_DUMMY_TRAP_INT_CONTEXT_int_context_offset = 1 # macro
SDMA_PKT_DUMMY_TRAP_INT_CONTEXT_int_context_mask = 0x0FFFFFFF # macro
SDMA_PKT_DUMMY_TRAP_INT_CONTEXT_int_context_shift = 0 # macro
def SDMA_PKT_DUMMY_TRAP_INT_CONTEXT_INT_CONTEXT(x):  # macro
   return (((x)&0x0FFFFFFF)<<0)
SDMA_PKT_GPUVM_INV_HEADER_op_offset = 0 # macro
SDMA_PKT_GPUVM_INV_HEADER_op_mask = 0x000000FF # macro
SDMA_PKT_GPUVM_INV_HEADER_op_shift = 0 # macro
def SDMA_PKT_GPUVM_INV_HEADER_OP(x):  # macro
   return (((x)&0x000000FF)<<0)
SDMA_PKT_GPUVM_INV_HEADER_sub_op_offset = 0 # macro
SDMA_PKT_GPUVM_INV_HEADER_sub_op_mask = 0x000000FF # macro
SDMA_PKT_GPUVM_INV_HEADER_sub_op_shift = 8 # macro
def SDMA_PKT_GPUVM_INV_HEADER_SUB_OP(x):  # macro
   return (((x)&0x000000FF)<<8)
SDMA_PKT_GPUVM_INV_PAYLOAD1_per_vmid_inv_req_offset = 1 # macro
SDMA_PKT_GPUVM_INV_PAYLOAD1_per_vmid_inv_req_mask = 0x0000FFFF # macro
SDMA_PKT_GPUVM_INV_PAYLOAD1_per_vmid_inv_req_shift = 0 # macro
def SDMA_PKT_GPUVM_INV_PAYLOAD1_PER_VMID_INV_REQ(x):  # macro
   return (((x)&0x0000FFFF)<<0)
SDMA_PKT_GPUVM_INV_PAYLOAD1_flush_type_offset = 1 # macro
SDMA_PKT_GPUVM_INV_PAYLOAD1_flush_type_mask = 0x00000007 # macro
SDMA_PKT_GPUVM_INV_PAYLOAD1_flush_type_shift = 16 # macro
def SDMA_PKT_GPUVM_INV_PAYLOAD1_FLUSH_TYPE(x):  # macro
   return (((x)&0x00000007)<<16)
SDMA_PKT_GPUVM_INV_PAYLOAD1_l2_ptes_offset = 1 # macro
SDMA_PKT_GPUVM_INV_PAYLOAD1_l2_ptes_mask = 0x00000001 # macro
SDMA_PKT_GPUVM_INV_PAYLOAD1_l2_ptes_shift = 19 # macro
def SDMA_PKT_GPUVM_INV_PAYLOAD1_L2_PTES(x):  # macro
   return (((x)&0x00000001)<<19)
SDMA_PKT_GPUVM_INV_PAYLOAD1_l2_pde0_offset = 1 # macro
SDMA_PKT_GPUVM_INV_PAYLOAD1_l2_pde0_mask = 0x00000001 # macro
SDMA_PKT_GPUVM_INV_PAYLOAD1_l2_pde0_shift = 20 # macro
def SDMA_PKT_GPUVM_INV_PAYLOAD1_L2_PDE0(x):  # macro
   return (((x)&0x00000001)<<20)
SDMA_PKT_GPUVM_INV_PAYLOAD1_l2_pde1_offset = 1 # macro
SDMA_PKT_GPUVM_INV_PAYLOAD1_l2_pde1_mask = 0x00000001 # macro
SDMA_PKT_GPUVM_INV_PAYLOAD1_l2_pde1_shift = 21 # macro
def SDMA_PKT_GPUVM_INV_PAYLOAD1_L2_PDE1(x):  # macro
   return (((x)&0x00000001)<<21)
SDMA_PKT_GPUVM_INV_PAYLOAD1_l2_pde2_offset = 1 # macro
SDMA_PKT_GPUVM_INV_PAYLOAD1_l2_pde2_mask = 0x00000001 # macro
SDMA_PKT_GPUVM_INV_PAYLOAD1_l2_pde2_shift = 22 # macro
def SDMA_PKT_GPUVM_INV_PAYLOAD1_L2_PDE2(x):  # macro
   return (((x)&0x00000001)<<22)
SDMA_PKT_GPUVM_INV_PAYLOAD1_l1_ptes_offset = 1 # macro
SDMA_PKT_GPUVM_INV_PAYLOAD1_l1_ptes_mask = 0x00000001 # macro
SDMA_PKT_GPUVM_INV_PAYLOAD1_l1_ptes_shift = 23 # macro
def SDMA_PKT_GPUVM_INV_PAYLOAD1_L1_PTES(x):  # macro
   return (((x)&0x00000001)<<23)
SDMA_PKT_GPUVM_INV_PAYLOAD1_clr_protection_fault_status_addr_offset = 1 # macro
SDMA_PKT_GPUVM_INV_PAYLOAD1_clr_protection_fault_status_addr_mask = 0x00000001 # macro
SDMA_PKT_GPUVM_INV_PAYLOAD1_clr_protection_fault_status_addr_shift = 24 # macro
def SDMA_PKT_GPUVM_INV_PAYLOAD1_CLR_PROTECTION_FAULT_STATUS_ADDR(x):  # macro
   return (((x)&0x00000001)<<24)
SDMA_PKT_GPUVM_INV_PAYLOAD1_log_request_offset = 1 # macro
SDMA_PKT_GPUVM_INV_PAYLOAD1_log_request_mask = 0x00000001 # macro
SDMA_PKT_GPUVM_INV_PAYLOAD1_log_request_shift = 25 # macro
def SDMA_PKT_GPUVM_INV_PAYLOAD1_LOG_REQUEST(x):  # macro
   return (((x)&0x00000001)<<25)
SDMA_PKT_GPUVM_INV_PAYLOAD1_four_kilobytes_offset = 1 # macro
SDMA_PKT_GPUVM_INV_PAYLOAD1_four_kilobytes_mask = 0x00000001 # macro
SDMA_PKT_GPUVM_INV_PAYLOAD1_four_kilobytes_shift = 26 # macro
def SDMA_PKT_GPUVM_INV_PAYLOAD1_FOUR_KILOBYTES(x):  # macro
   return (((x)&0x00000001)<<26)
SDMA_PKT_GPUVM_INV_PAYLOAD2_s_offset = 2 # macro
SDMA_PKT_GPUVM_INV_PAYLOAD2_s_mask = 0x00000001 # macro
SDMA_PKT_GPUVM_INV_PAYLOAD2_s_shift = 0 # macro
def SDMA_PKT_GPUVM_INV_PAYLOAD2_S(x):  # macro
   return (((x)&0x00000001)<<0)
SDMA_PKT_GPUVM_INV_PAYLOAD2_page_va_42_12_offset = 2 # macro
SDMA_PKT_GPUVM_INV_PAYLOAD2_page_va_42_12_mask = 0x7FFFFFFF # macro
SDMA_PKT_GPUVM_INV_PAYLOAD2_page_va_42_12_shift = 1 # macro
def SDMA_PKT_GPUVM_INV_PAYLOAD2_PAGE_VA_42_12(x):  # macro
   return (((x)&0x7FFFFFFF)<<1)
SDMA_PKT_GPUVM_INV_PAYLOAD3_page_va_47_43_offset = 3 # macro
SDMA_PKT_GPUVM_INV_PAYLOAD3_page_va_47_43_mask = 0x0000003F # macro
SDMA_PKT_GPUVM_INV_PAYLOAD3_page_va_47_43_shift = 0 # macro
def SDMA_PKT_GPUVM_INV_PAYLOAD3_PAGE_VA_47_43(x):  # macro
   return (((x)&0x0000003F)<<0)
SDMA_PKT_GCR_REQ_HEADER_op_offset = 0 # macro
SDMA_PKT_GCR_REQ_HEADER_op_mask = 0x000000FF # macro
SDMA_PKT_GCR_REQ_HEADER_op_shift = 0 # macro
def SDMA_PKT_GCR_REQ_HEADER_OP(x):  # macro
   return (((x)&0x000000FF)<<0)
SDMA_PKT_GCR_REQ_HEADER_sub_op_offset = 0 # macro
SDMA_PKT_GCR_REQ_HEADER_sub_op_mask = 0x000000FF # macro
SDMA_PKT_GCR_REQ_HEADER_sub_op_shift = 8 # macro
def SDMA_PKT_GCR_REQ_HEADER_SUB_OP(x):  # macro
   return (((x)&0x000000FF)<<8)
SDMA_PKT_GCR_REQ_PAYLOAD1_base_va_31_7_offset = 1 # macro
SDMA_PKT_GCR_REQ_PAYLOAD1_base_va_31_7_mask = 0x01FFFFFF # macro
SDMA_PKT_GCR_REQ_PAYLOAD1_base_va_31_7_shift = 7 # macro
def SDMA_PKT_GCR_REQ_PAYLOAD1_BASE_VA_31_7(x):  # macro
   return (((x)&0x01FFFFFF)<<7)
SDMA_PKT_GCR_REQ_PAYLOAD2_base_va_47_32_offset = 2 # macro
SDMA_PKT_GCR_REQ_PAYLOAD2_base_va_47_32_mask = 0x0000FFFF # macro
SDMA_PKT_GCR_REQ_PAYLOAD2_base_va_47_32_shift = 0 # macro
def SDMA_PKT_GCR_REQ_PAYLOAD2_BASE_VA_47_32(x):  # macro
   return (((x)&0x0000FFFF)<<0)
SDMA_PKT_GCR_REQ_PAYLOAD2_gcr_control_15_0_offset = 2 # macro
SDMA_PKT_GCR_REQ_PAYLOAD2_gcr_control_15_0_mask = 0x0000FFFF # macro
SDMA_PKT_GCR_REQ_PAYLOAD2_gcr_control_15_0_shift = 16 # macro
def SDMA_PKT_GCR_REQ_PAYLOAD2_GCR_CONTROL_15_0(x):  # macro
   return (((x)&0x0000FFFF)<<16)
SDMA_PKT_GCR_REQ_PAYLOAD3_gcr_control_18_16_offset = 3 # macro
SDMA_PKT_GCR_REQ_PAYLOAD3_gcr_control_18_16_mask = 0x00000007 # macro
SDMA_PKT_GCR_REQ_PAYLOAD3_gcr_control_18_16_shift = 0 # macro
def SDMA_PKT_GCR_REQ_PAYLOAD3_GCR_CONTROL_18_16(x):  # macro
   return (((x)&0x00000007)<<0)
SDMA_PKT_GCR_REQ_PAYLOAD3_limit_va_31_7_offset = 3 # macro
SDMA_PKT_GCR_REQ_PAYLOAD3_limit_va_31_7_mask = 0x01FFFFFF # macro
SDMA_PKT_GCR_REQ_PAYLOAD3_limit_va_31_7_shift = 7 # macro
def SDMA_PKT_GCR_REQ_PAYLOAD3_LIMIT_VA_31_7(x):  # macro
   return (((x)&0x01FFFFFF)<<7)
SDMA_PKT_GCR_REQ_PAYLOAD4_limit_va_47_32_offset = 4 # macro
SDMA_PKT_GCR_REQ_PAYLOAD4_limit_va_47_32_mask = 0x0000FFFF # macro
SDMA_PKT_GCR_REQ_PAYLOAD4_limit_va_47_32_shift = 0 # macro
def SDMA_PKT_GCR_REQ_PAYLOAD4_LIMIT_VA_47_32(x):  # macro
   return (((x)&0x0000FFFF)<<0)
SDMA_PKT_GCR_REQ_PAYLOAD4_vmid_offset = 4 # macro
SDMA_PKT_GCR_REQ_PAYLOAD4_vmid_mask = 0x0000000F # macro
SDMA_PKT_GCR_REQ_PAYLOAD4_vmid_shift = 24 # macro
def SDMA_PKT_GCR_REQ_PAYLOAD4_VMID(x):  # macro
   return (((x)&0x0000000F)<<24)
SDMA_PKT_NOP_HEADER_op_offset = 0 # macro
SDMA_PKT_NOP_HEADER_op_mask = 0x000000FF # macro
SDMA_PKT_NOP_HEADER_op_shift = 0 # macro
def SDMA_PKT_NOP_HEADER_OP(x):  # macro
   return (((x)&0x000000FF)<<0)
SDMA_PKT_NOP_HEADER_sub_op_offset = 0 # macro
SDMA_PKT_NOP_HEADER_sub_op_mask = 0x000000FF # macro
SDMA_PKT_NOP_HEADER_sub_op_shift = 8 # macro
def SDMA_PKT_NOP_HEADER_SUB_OP(x):  # macro
   return (((x)&0x000000FF)<<8)
SDMA_PKT_NOP_HEADER_count_offset = 0 # macro
SDMA_PKT_NOP_HEADER_count_mask = 0x00003FFF # macro
SDMA_PKT_NOP_HEADER_count_shift = 16 # macro
def SDMA_PKT_NOP_HEADER_COUNT(x):  # macro
   return (((x)&0x00003FFF)<<16)
SDMA_PKT_NOP_DATA0_data0_offset = 1 # macro
SDMA_PKT_NOP_DATA0_data0_mask = 0xFFFFFFFF # macro
SDMA_PKT_NOP_DATA0_data0_shift = 0 # macro
def SDMA_PKT_NOP_DATA0_DATA0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_AQL_PKT_HEADER_HEADER_format_offset = 0 # macro
SDMA_AQL_PKT_HEADER_HEADER_format_mask = 0x000000FF # macro
SDMA_AQL_PKT_HEADER_HEADER_format_shift = 0 # macro
def SDMA_AQL_PKT_HEADER_HEADER_FORMAT(x):  # macro
   return (((x)&0x000000FF)<<0)
SDMA_AQL_PKT_HEADER_HEADER_barrier_offset = 0 # macro
SDMA_AQL_PKT_HEADER_HEADER_barrier_mask = 0x00000001 # macro
SDMA_AQL_PKT_HEADER_HEADER_barrier_shift = 8 # macro
def SDMA_AQL_PKT_HEADER_HEADER_BARRIER(x):  # macro
   return (((x)&0x00000001)<<8)
SDMA_AQL_PKT_HEADER_HEADER_acquire_fence_scope_offset = 0 # macro
SDMA_AQL_PKT_HEADER_HEADER_acquire_fence_scope_mask = 0x00000003 # macro
SDMA_AQL_PKT_HEADER_HEADER_acquire_fence_scope_shift = 9 # macro
def SDMA_AQL_PKT_HEADER_HEADER_ACQUIRE_FENCE_SCOPE(x):  # macro
   return (((x)&0x00000003)<<9)
SDMA_AQL_PKT_HEADER_HEADER_release_fence_scope_offset = 0 # macro
SDMA_AQL_PKT_HEADER_HEADER_release_fence_scope_mask = 0x00000003 # macro
SDMA_AQL_PKT_HEADER_HEADER_release_fence_scope_shift = 11 # macro
def SDMA_AQL_PKT_HEADER_HEADER_RELEASE_FENCE_SCOPE(x):  # macro
   return (((x)&0x00000003)<<11)
SDMA_AQL_PKT_HEADER_HEADER_reserved_offset = 0 # macro
SDMA_AQL_PKT_HEADER_HEADER_reserved_mask = 0x00000007 # macro
SDMA_AQL_PKT_HEADER_HEADER_reserved_shift = 13 # macro
def SDMA_AQL_PKT_HEADER_HEADER_RESERVED(x):  # macro
   return (((x)&0x00000007)<<13)
SDMA_AQL_PKT_HEADER_HEADER_op_offset = 0 # macro
SDMA_AQL_PKT_HEADER_HEADER_op_mask = 0x0000000F # macro
SDMA_AQL_PKT_HEADER_HEADER_op_shift = 16 # macro
def SDMA_AQL_PKT_HEADER_HEADER_OP(x):  # macro
   return (((x)&0x0000000F)<<16)
SDMA_AQL_PKT_HEADER_HEADER_subop_offset = 0 # macro
SDMA_AQL_PKT_HEADER_HEADER_subop_mask = 0x00000007 # macro
SDMA_AQL_PKT_HEADER_HEADER_subop_shift = 20 # macro
def SDMA_AQL_PKT_HEADER_HEADER_SUBOP(x):  # macro
   return (((x)&0x00000007)<<20)
SDMA_AQL_PKT_COPY_LINEAR_HEADER_format_offset = 0 # macro
SDMA_AQL_PKT_COPY_LINEAR_HEADER_format_mask = 0x000000FF # macro
SDMA_AQL_PKT_COPY_LINEAR_HEADER_format_shift = 0 # macro
def SDMA_AQL_PKT_COPY_LINEAR_HEADER_FORMAT(x):  # macro
   return (((x)&0x000000FF)<<0)
SDMA_AQL_PKT_COPY_LINEAR_HEADER_barrier_offset = 0 # macro
SDMA_AQL_PKT_COPY_LINEAR_HEADER_barrier_mask = 0x00000001 # macro
SDMA_AQL_PKT_COPY_LINEAR_HEADER_barrier_shift = 8 # macro
def SDMA_AQL_PKT_COPY_LINEAR_HEADER_BARRIER(x):  # macro
   return (((x)&0x00000001)<<8)
SDMA_AQL_PKT_COPY_LINEAR_HEADER_acquire_fence_scope_offset = 0 # macro
SDMA_AQL_PKT_COPY_LINEAR_HEADER_acquire_fence_scope_mask = 0x00000003 # macro
SDMA_AQL_PKT_COPY_LINEAR_HEADER_acquire_fence_scope_shift = 9 # macro
def SDMA_AQL_PKT_COPY_LINEAR_HEADER_ACQUIRE_FENCE_SCOPE(x):  # macro
   return (((x)&0x00000003)<<9)
SDMA_AQL_PKT_COPY_LINEAR_HEADER_release_fence_scope_offset = 0 # macro
SDMA_AQL_PKT_COPY_LINEAR_HEADER_release_fence_scope_mask = 0x00000003 # macro
SDMA_AQL_PKT_COPY_LINEAR_HEADER_release_fence_scope_shift = 11 # macro
def SDMA_AQL_PKT_COPY_LINEAR_HEADER_RELEASE_FENCE_SCOPE(x):  # macro
   return (((x)&0x00000003)<<11)
SDMA_AQL_PKT_COPY_LINEAR_HEADER_reserved_offset = 0 # macro
SDMA_AQL_PKT_COPY_LINEAR_HEADER_reserved_mask = 0x00000007 # macro
SDMA_AQL_PKT_COPY_LINEAR_HEADER_reserved_shift = 13 # macro
def SDMA_AQL_PKT_COPY_LINEAR_HEADER_RESERVED(x):  # macro
   return (((x)&0x00000007)<<13)
SDMA_AQL_PKT_COPY_LINEAR_HEADER_op_offset = 0 # macro
SDMA_AQL_PKT_COPY_LINEAR_HEADER_op_mask = 0x0000000F # macro
SDMA_AQL_PKT_COPY_LINEAR_HEADER_op_shift = 16 # macro
def SDMA_AQL_PKT_COPY_LINEAR_HEADER_OP(x):  # macro
   return (((x)&0x0000000F)<<16)
SDMA_AQL_PKT_COPY_LINEAR_HEADER_subop_offset = 0 # macro
SDMA_AQL_PKT_COPY_LINEAR_HEADER_subop_mask = 0x00000007 # macro
SDMA_AQL_PKT_COPY_LINEAR_HEADER_subop_shift = 20 # macro
def SDMA_AQL_PKT_COPY_LINEAR_HEADER_SUBOP(x):  # macro
   return (((x)&0x00000007)<<20)
SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW1_reserved_dw1_offset = 1 # macro
SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW1_reserved_dw1_mask = 0xFFFFFFFF # macro
SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW1_reserved_dw1_shift = 0 # macro
def SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW1_RESERVED_DW1(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_AQL_PKT_COPY_LINEAR_RETURN_ADDR_LO_return_addr_31_0_offset = 2 # macro
SDMA_AQL_PKT_COPY_LINEAR_RETURN_ADDR_LO_return_addr_31_0_mask = 0xFFFFFFFF # macro
SDMA_AQL_PKT_COPY_LINEAR_RETURN_ADDR_LO_return_addr_31_0_shift = 0 # macro
def SDMA_AQL_PKT_COPY_LINEAR_RETURN_ADDR_LO_RETURN_ADDR_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_AQL_PKT_COPY_LINEAR_RETURN_ADDR_HI_return_addr_63_32_offset = 3 # macro
SDMA_AQL_PKT_COPY_LINEAR_RETURN_ADDR_HI_return_addr_63_32_mask = 0xFFFFFFFF # macro
SDMA_AQL_PKT_COPY_LINEAR_RETURN_ADDR_HI_return_addr_63_32_shift = 0 # macro
def SDMA_AQL_PKT_COPY_LINEAR_RETURN_ADDR_HI_RETURN_ADDR_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_AQL_PKT_COPY_LINEAR_COUNT_count_offset = 4 # macro
SDMA_AQL_PKT_COPY_LINEAR_COUNT_count_mask = 0x003FFFFF # macro
SDMA_AQL_PKT_COPY_LINEAR_COUNT_count_shift = 0 # macro
def SDMA_AQL_PKT_COPY_LINEAR_COUNT_COUNT(x):  # macro
   return (((x)&0x003FFFFF)<<0)
SDMA_AQL_PKT_COPY_LINEAR_PARAMETER_dst_sw_offset = 5 # macro
SDMA_AQL_PKT_COPY_LINEAR_PARAMETER_dst_sw_mask = 0x00000003 # macro
SDMA_AQL_PKT_COPY_LINEAR_PARAMETER_dst_sw_shift = 16 # macro
def SDMA_AQL_PKT_COPY_LINEAR_PARAMETER_DST_SW(x):  # macro
   return (((x)&0x00000003)<<16)
SDMA_AQL_PKT_COPY_LINEAR_PARAMETER_src_sw_offset = 5 # macro
SDMA_AQL_PKT_COPY_LINEAR_PARAMETER_src_sw_mask = 0x00000003 # macro
SDMA_AQL_PKT_COPY_LINEAR_PARAMETER_src_sw_shift = 24 # macro
def SDMA_AQL_PKT_COPY_LINEAR_PARAMETER_SRC_SW(x):  # macro
   return (((x)&0x00000003)<<24)
SDMA_AQL_PKT_COPY_LINEAR_SRC_ADDR_LO_src_addr_31_0_offset = 6 # macro
SDMA_AQL_PKT_COPY_LINEAR_SRC_ADDR_LO_src_addr_31_0_mask = 0xFFFFFFFF # macro
SDMA_AQL_PKT_COPY_LINEAR_SRC_ADDR_LO_src_addr_31_0_shift = 0 # macro
def SDMA_AQL_PKT_COPY_LINEAR_SRC_ADDR_LO_SRC_ADDR_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_AQL_PKT_COPY_LINEAR_SRC_ADDR_HI_src_addr_63_32_offset = 7 # macro
SDMA_AQL_PKT_COPY_LINEAR_SRC_ADDR_HI_src_addr_63_32_mask = 0xFFFFFFFF # macro
SDMA_AQL_PKT_COPY_LINEAR_SRC_ADDR_HI_src_addr_63_32_shift = 0 # macro
def SDMA_AQL_PKT_COPY_LINEAR_SRC_ADDR_HI_SRC_ADDR_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_AQL_PKT_COPY_LINEAR_DST_ADDR_LO_dst_addr_31_0_offset = 8 # macro
SDMA_AQL_PKT_COPY_LINEAR_DST_ADDR_LO_dst_addr_31_0_mask = 0xFFFFFFFF # macro
SDMA_AQL_PKT_COPY_LINEAR_DST_ADDR_LO_dst_addr_31_0_shift = 0 # macro
def SDMA_AQL_PKT_COPY_LINEAR_DST_ADDR_LO_DST_ADDR_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_AQL_PKT_COPY_LINEAR_DST_ADDR_HI_dst_addr_63_32_offset = 9 # macro
SDMA_AQL_PKT_COPY_LINEAR_DST_ADDR_HI_dst_addr_63_32_mask = 0xFFFFFFFF # macro
SDMA_AQL_PKT_COPY_LINEAR_DST_ADDR_HI_dst_addr_63_32_shift = 0 # macro
def SDMA_AQL_PKT_COPY_LINEAR_DST_ADDR_HI_DST_ADDR_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW10_reserved_dw10_offset = 10 # macro
SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW10_reserved_dw10_mask = 0xFFFFFFFF # macro
SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW10_reserved_dw10_shift = 0 # macro
def SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW10_RESERVED_DW10(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW11_reserved_dw11_offset = 11 # macro
SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW11_reserved_dw11_mask = 0xFFFFFFFF # macro
SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW11_reserved_dw11_shift = 0 # macro
def SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW11_RESERVED_DW11(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW12_reserved_dw12_offset = 12 # macro
SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW12_reserved_dw12_mask = 0xFFFFFFFF # macro
SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW12_reserved_dw12_shift = 0 # macro
def SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW12_RESERVED_DW12(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW13_reserved_dw13_offset = 13 # macro
SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW13_reserved_dw13_mask = 0xFFFFFFFF # macro
SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW13_reserved_dw13_shift = 0 # macro
def SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW13_RESERVED_DW13(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_AQL_PKT_COPY_LINEAR_COMPLETION_SIGNAL_LO_completion_signal_31_0_offset = 14 # macro
SDMA_AQL_PKT_COPY_LINEAR_COMPLETION_SIGNAL_LO_completion_signal_31_0_mask = 0xFFFFFFFF # macro
SDMA_AQL_PKT_COPY_LINEAR_COMPLETION_SIGNAL_LO_completion_signal_31_0_shift = 0 # macro
def SDMA_AQL_PKT_COPY_LINEAR_COMPLETION_SIGNAL_LO_COMPLETION_SIGNAL_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_AQL_PKT_COPY_LINEAR_COMPLETION_SIGNAL_HI_completion_signal_63_32_offset = 15 # macro
SDMA_AQL_PKT_COPY_LINEAR_COMPLETION_SIGNAL_HI_completion_signal_63_32_mask = 0xFFFFFFFF # macro
SDMA_AQL_PKT_COPY_LINEAR_COMPLETION_SIGNAL_HI_completion_signal_63_32_shift = 0 # macro
def SDMA_AQL_PKT_COPY_LINEAR_COMPLETION_SIGNAL_HI_COMPLETION_SIGNAL_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_AQL_PKT_BARRIER_OR_HEADER_format_offset = 0 # macro
SDMA_AQL_PKT_BARRIER_OR_HEADER_format_mask = 0x000000FF # macro
SDMA_AQL_PKT_BARRIER_OR_HEADER_format_shift = 0 # macro
def SDMA_AQL_PKT_BARRIER_OR_HEADER_FORMAT(x):  # macro
   return (((x)&0x000000FF)<<0)
SDMA_AQL_PKT_BARRIER_OR_HEADER_barrier_offset = 0 # macro
SDMA_AQL_PKT_BARRIER_OR_HEADER_barrier_mask = 0x00000001 # macro
SDMA_AQL_PKT_BARRIER_OR_HEADER_barrier_shift = 8 # macro
def SDMA_AQL_PKT_BARRIER_OR_HEADER_BARRIER(x):  # macro
   return (((x)&0x00000001)<<8)
SDMA_AQL_PKT_BARRIER_OR_HEADER_acquire_fence_scope_offset = 0 # macro
SDMA_AQL_PKT_BARRIER_OR_HEADER_acquire_fence_scope_mask = 0x00000003 # macro
SDMA_AQL_PKT_BARRIER_OR_HEADER_acquire_fence_scope_shift = 9 # macro
def SDMA_AQL_PKT_BARRIER_OR_HEADER_ACQUIRE_FENCE_SCOPE(x):  # macro
   return (((x)&0x00000003)<<9)
SDMA_AQL_PKT_BARRIER_OR_HEADER_release_fence_scope_offset = 0 # macro
SDMA_AQL_PKT_BARRIER_OR_HEADER_release_fence_scope_mask = 0x00000003 # macro
SDMA_AQL_PKT_BARRIER_OR_HEADER_release_fence_scope_shift = 11 # macro
def SDMA_AQL_PKT_BARRIER_OR_HEADER_RELEASE_FENCE_SCOPE(x):  # macro
   return (((x)&0x00000003)<<11)
SDMA_AQL_PKT_BARRIER_OR_HEADER_reserved_offset = 0 # macro
SDMA_AQL_PKT_BARRIER_OR_HEADER_reserved_mask = 0x00000007 # macro
SDMA_AQL_PKT_BARRIER_OR_HEADER_reserved_shift = 13 # macro
def SDMA_AQL_PKT_BARRIER_OR_HEADER_RESERVED(x):  # macro
   return (((x)&0x00000007)<<13)
SDMA_AQL_PKT_BARRIER_OR_HEADER_op_offset = 0 # macro
SDMA_AQL_PKT_BARRIER_OR_HEADER_op_mask = 0x0000000F # macro
SDMA_AQL_PKT_BARRIER_OR_HEADER_op_shift = 16 # macro
def SDMA_AQL_PKT_BARRIER_OR_HEADER_OP(x):  # macro
   return (((x)&0x0000000F)<<16)
SDMA_AQL_PKT_BARRIER_OR_HEADER_subop_offset = 0 # macro
SDMA_AQL_PKT_BARRIER_OR_HEADER_subop_mask = 0x00000007 # macro
SDMA_AQL_PKT_BARRIER_OR_HEADER_subop_shift = 20 # macro
def SDMA_AQL_PKT_BARRIER_OR_HEADER_SUBOP(x):  # macro
   return (((x)&0x00000007)<<20)
SDMA_AQL_PKT_BARRIER_OR_RESERVED_DW1_reserved_dw1_offset = 1 # macro
SDMA_AQL_PKT_BARRIER_OR_RESERVED_DW1_reserved_dw1_mask = 0xFFFFFFFF # macro
SDMA_AQL_PKT_BARRIER_OR_RESERVED_DW1_reserved_dw1_shift = 0 # macro
def SDMA_AQL_PKT_BARRIER_OR_RESERVED_DW1_RESERVED_DW1(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_0_LO_dependent_addr_0_31_0_offset = 2 # macro
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_0_LO_dependent_addr_0_31_0_mask = 0xFFFFFFFF # macro
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_0_LO_dependent_addr_0_31_0_shift = 0 # macro
def SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_0_LO_DEPENDENT_ADDR_0_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_0_HI_dependent_addr_0_63_32_offset = 3 # macro
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_0_HI_dependent_addr_0_63_32_mask = 0xFFFFFFFF # macro
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_0_HI_dependent_addr_0_63_32_shift = 0 # macro
def SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_0_HI_DEPENDENT_ADDR_0_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_1_LO_dependent_addr_1_31_0_offset = 4 # macro
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_1_LO_dependent_addr_1_31_0_mask = 0xFFFFFFFF # macro
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_1_LO_dependent_addr_1_31_0_shift = 0 # macro
def SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_1_LO_DEPENDENT_ADDR_1_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_1_HI_dependent_addr_1_63_32_offset = 5 # macro
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_1_HI_dependent_addr_1_63_32_mask = 0xFFFFFFFF # macro
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_1_HI_dependent_addr_1_63_32_shift = 0 # macro
def SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_1_HI_DEPENDENT_ADDR_1_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_2_LO_dependent_addr_2_31_0_offset = 6 # macro
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_2_LO_dependent_addr_2_31_0_mask = 0xFFFFFFFF # macro
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_2_LO_dependent_addr_2_31_0_shift = 0 # macro
def SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_2_LO_DEPENDENT_ADDR_2_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_2_HI_dependent_addr_2_63_32_offset = 7 # macro
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_2_HI_dependent_addr_2_63_32_mask = 0xFFFFFFFF # macro
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_2_HI_dependent_addr_2_63_32_shift = 0 # macro
def SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_2_HI_DEPENDENT_ADDR_2_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_3_LO_dependent_addr_3_31_0_offset = 8 # macro
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_3_LO_dependent_addr_3_31_0_mask = 0xFFFFFFFF # macro
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_3_LO_dependent_addr_3_31_0_shift = 0 # macro
def SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_3_LO_DEPENDENT_ADDR_3_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_3_HI_dependent_addr_3_63_32_offset = 9 # macro
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_3_HI_dependent_addr_3_63_32_mask = 0xFFFFFFFF # macro
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_3_HI_dependent_addr_3_63_32_shift = 0 # macro
def SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_3_HI_DEPENDENT_ADDR_3_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_4_LO_dependent_addr_4_31_0_offset = 10 # macro
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_4_LO_dependent_addr_4_31_0_mask = 0xFFFFFFFF # macro
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_4_LO_dependent_addr_4_31_0_shift = 0 # macro
def SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_4_LO_DEPENDENT_ADDR_4_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_4_HI_dependent_addr_4_63_32_offset = 11 # macro
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_4_HI_dependent_addr_4_63_32_mask = 0xFFFFFFFF # macro
SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_4_HI_dependent_addr_4_63_32_shift = 0 # macro
def SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_4_HI_DEPENDENT_ADDR_4_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_AQL_PKT_BARRIER_OR_RESERVED_DW12_reserved_dw12_offset = 12 # macro
SDMA_AQL_PKT_BARRIER_OR_RESERVED_DW12_reserved_dw12_mask = 0xFFFFFFFF # macro
SDMA_AQL_PKT_BARRIER_OR_RESERVED_DW12_reserved_dw12_shift = 0 # macro
def SDMA_AQL_PKT_BARRIER_OR_RESERVED_DW12_RESERVED_DW12(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_AQL_PKT_BARRIER_OR_RESERVED_DW13_reserved_dw13_offset = 13 # macro
SDMA_AQL_PKT_BARRIER_OR_RESERVED_DW13_reserved_dw13_mask = 0xFFFFFFFF # macro
SDMA_AQL_PKT_BARRIER_OR_RESERVED_DW13_reserved_dw13_shift = 0 # macro
def SDMA_AQL_PKT_BARRIER_OR_RESERVED_DW13_RESERVED_DW13(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_AQL_PKT_BARRIER_OR_COMPLETION_SIGNAL_LO_completion_signal_31_0_offset = 14 # macro
SDMA_AQL_PKT_BARRIER_OR_COMPLETION_SIGNAL_LO_completion_signal_31_0_mask = 0xFFFFFFFF # macro
SDMA_AQL_PKT_BARRIER_OR_COMPLETION_SIGNAL_LO_completion_signal_31_0_shift = 0 # macro
def SDMA_AQL_PKT_BARRIER_OR_COMPLETION_SIGNAL_LO_COMPLETION_SIGNAL_31_0(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
SDMA_AQL_PKT_BARRIER_OR_COMPLETION_SIGNAL_HI_completion_signal_63_32_offset = 15 # macro
SDMA_AQL_PKT_BARRIER_OR_COMPLETION_SIGNAL_HI_completion_signal_63_32_mask = 0xFFFFFFFF # macro
SDMA_AQL_PKT_BARRIER_OR_COMPLETION_SIGNAL_HI_completion_signal_63_32_shift = 0 # macro
def SDMA_AQL_PKT_BARRIER_OR_COMPLETION_SIGNAL_HI_COMPLETION_SIGNAL_63_32(x):  # macro
   return (((x)&0xFFFFFFFF)<<0)
__all__ = \
    ['HEADER_AGENT_DISPATCH', 'HEADER_BARRIER',
    'HSA_RUNTIME_CORE_INC_SDMA_REGISTERS_H_',
    'SDMA_AQL_PKT_BARRIER_OR_COMPLETION_SIGNAL_HI_completion_signal_63_32_mask',
    'SDMA_AQL_PKT_BARRIER_OR_COMPLETION_SIGNAL_HI_completion_signal_63_32_offset',
    'SDMA_AQL_PKT_BARRIER_OR_COMPLETION_SIGNAL_HI_completion_signal_63_32_shift',
    'SDMA_AQL_PKT_BARRIER_OR_COMPLETION_SIGNAL_LO_completion_signal_31_0_mask',
    'SDMA_AQL_PKT_BARRIER_OR_COMPLETION_SIGNAL_LO_completion_signal_31_0_offset',
    'SDMA_AQL_PKT_BARRIER_OR_COMPLETION_SIGNAL_LO_completion_signal_31_0_shift',
    'SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_0_HI_dependent_addr_0_63_32_mask',
    'SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_0_HI_dependent_addr_0_63_32_offset',
    'SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_0_HI_dependent_addr_0_63_32_shift',
    'SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_0_LO_dependent_addr_0_31_0_mask',
    'SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_0_LO_dependent_addr_0_31_0_offset',
    'SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_0_LO_dependent_addr_0_31_0_shift',
    'SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_1_HI_dependent_addr_1_63_32_mask',
    'SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_1_HI_dependent_addr_1_63_32_offset',
    'SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_1_HI_dependent_addr_1_63_32_shift',
    'SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_1_LO_dependent_addr_1_31_0_mask',
    'SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_1_LO_dependent_addr_1_31_0_offset',
    'SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_1_LO_dependent_addr_1_31_0_shift',
    'SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_2_HI_dependent_addr_2_63_32_mask',
    'SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_2_HI_dependent_addr_2_63_32_offset',
    'SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_2_HI_dependent_addr_2_63_32_shift',
    'SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_2_LO_dependent_addr_2_31_0_mask',
    'SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_2_LO_dependent_addr_2_31_0_offset',
    'SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_2_LO_dependent_addr_2_31_0_shift',
    'SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_3_HI_dependent_addr_3_63_32_mask',
    'SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_3_HI_dependent_addr_3_63_32_offset',
    'SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_3_HI_dependent_addr_3_63_32_shift',
    'SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_3_LO_dependent_addr_3_31_0_mask',
    'SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_3_LO_dependent_addr_3_31_0_offset',
    'SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_3_LO_dependent_addr_3_31_0_shift',
    'SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_4_HI_dependent_addr_4_63_32_mask',
    'SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_4_HI_dependent_addr_4_63_32_offset',
    'SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_4_HI_dependent_addr_4_63_32_shift',
    'SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_4_LO_dependent_addr_4_31_0_mask',
    'SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_4_LO_dependent_addr_4_31_0_offset',
    'SDMA_AQL_PKT_BARRIER_OR_DEPENDENT_ADDR_4_LO_dependent_addr_4_31_0_shift',
    'SDMA_AQL_PKT_BARRIER_OR_HEADER_acquire_fence_scope_mask',
    'SDMA_AQL_PKT_BARRIER_OR_HEADER_acquire_fence_scope_offset',
    'SDMA_AQL_PKT_BARRIER_OR_HEADER_acquire_fence_scope_shift',
    'SDMA_AQL_PKT_BARRIER_OR_HEADER_barrier_mask',
    'SDMA_AQL_PKT_BARRIER_OR_HEADER_barrier_offset',
    'SDMA_AQL_PKT_BARRIER_OR_HEADER_barrier_shift',
    'SDMA_AQL_PKT_BARRIER_OR_HEADER_format_mask',
    'SDMA_AQL_PKT_BARRIER_OR_HEADER_format_offset',
    'SDMA_AQL_PKT_BARRIER_OR_HEADER_format_shift',
    'SDMA_AQL_PKT_BARRIER_OR_HEADER_op_mask',
    'SDMA_AQL_PKT_BARRIER_OR_HEADER_op_offset',
    'SDMA_AQL_PKT_BARRIER_OR_HEADER_op_shift',
    'SDMA_AQL_PKT_BARRIER_OR_HEADER_release_fence_scope_mask',
    'SDMA_AQL_PKT_BARRIER_OR_HEADER_release_fence_scope_offset',
    'SDMA_AQL_PKT_BARRIER_OR_HEADER_release_fence_scope_shift',
    'SDMA_AQL_PKT_BARRIER_OR_HEADER_reserved_mask',
    'SDMA_AQL_PKT_BARRIER_OR_HEADER_reserved_offset',
    'SDMA_AQL_PKT_BARRIER_OR_HEADER_reserved_shift',
    'SDMA_AQL_PKT_BARRIER_OR_HEADER_subop_mask',
    'SDMA_AQL_PKT_BARRIER_OR_HEADER_subop_offset',
    'SDMA_AQL_PKT_BARRIER_OR_HEADER_subop_shift',
    'SDMA_AQL_PKT_BARRIER_OR_RESERVED_DW12_reserved_dw12_mask',
    'SDMA_AQL_PKT_BARRIER_OR_RESERVED_DW12_reserved_dw12_offset',
    'SDMA_AQL_PKT_BARRIER_OR_RESERVED_DW12_reserved_dw12_shift',
    'SDMA_AQL_PKT_BARRIER_OR_RESERVED_DW13_reserved_dw13_mask',
    'SDMA_AQL_PKT_BARRIER_OR_RESERVED_DW13_reserved_dw13_offset',
    'SDMA_AQL_PKT_BARRIER_OR_RESERVED_DW13_reserved_dw13_shift',
    'SDMA_AQL_PKT_BARRIER_OR_RESERVED_DW1_reserved_dw1_mask',
    'SDMA_AQL_PKT_BARRIER_OR_RESERVED_DW1_reserved_dw1_offset',
    'SDMA_AQL_PKT_BARRIER_OR_RESERVED_DW1_reserved_dw1_shift',
    'SDMA_AQL_PKT_COPY_LINEAR_COMPLETION_SIGNAL_HI_completion_signal_63_32_mask',
    'SDMA_AQL_PKT_COPY_LINEAR_COMPLETION_SIGNAL_HI_completion_signal_63_32_offset',
    'SDMA_AQL_PKT_COPY_LINEAR_COMPLETION_SIGNAL_HI_completion_signal_63_32_shift',
    'SDMA_AQL_PKT_COPY_LINEAR_COMPLETION_SIGNAL_LO_completion_signal_31_0_mask',
    'SDMA_AQL_PKT_COPY_LINEAR_COMPLETION_SIGNAL_LO_completion_signal_31_0_offset',
    'SDMA_AQL_PKT_COPY_LINEAR_COMPLETION_SIGNAL_LO_completion_signal_31_0_shift',
    'SDMA_AQL_PKT_COPY_LINEAR_COUNT_count_mask',
    'SDMA_AQL_PKT_COPY_LINEAR_COUNT_count_offset',
    'SDMA_AQL_PKT_COPY_LINEAR_COUNT_count_shift',
    'SDMA_AQL_PKT_COPY_LINEAR_DST_ADDR_HI_dst_addr_63_32_mask',
    'SDMA_AQL_PKT_COPY_LINEAR_DST_ADDR_HI_dst_addr_63_32_offset',
    'SDMA_AQL_PKT_COPY_LINEAR_DST_ADDR_HI_dst_addr_63_32_shift',
    'SDMA_AQL_PKT_COPY_LINEAR_DST_ADDR_LO_dst_addr_31_0_mask',
    'SDMA_AQL_PKT_COPY_LINEAR_DST_ADDR_LO_dst_addr_31_0_offset',
    'SDMA_AQL_PKT_COPY_LINEAR_DST_ADDR_LO_dst_addr_31_0_shift',
    'SDMA_AQL_PKT_COPY_LINEAR_HEADER_acquire_fence_scope_mask',
    'SDMA_AQL_PKT_COPY_LINEAR_HEADER_acquire_fence_scope_offset',
    'SDMA_AQL_PKT_COPY_LINEAR_HEADER_acquire_fence_scope_shift',
    'SDMA_AQL_PKT_COPY_LINEAR_HEADER_barrier_mask',
    'SDMA_AQL_PKT_COPY_LINEAR_HEADER_barrier_offset',
    'SDMA_AQL_PKT_COPY_LINEAR_HEADER_barrier_shift',
    'SDMA_AQL_PKT_COPY_LINEAR_HEADER_format_mask',
    'SDMA_AQL_PKT_COPY_LINEAR_HEADER_format_offset',
    'SDMA_AQL_PKT_COPY_LINEAR_HEADER_format_shift',
    'SDMA_AQL_PKT_COPY_LINEAR_HEADER_op_mask',
    'SDMA_AQL_PKT_COPY_LINEAR_HEADER_op_offset',
    'SDMA_AQL_PKT_COPY_LINEAR_HEADER_op_shift',
    'SDMA_AQL_PKT_COPY_LINEAR_HEADER_release_fence_scope_mask',
    'SDMA_AQL_PKT_COPY_LINEAR_HEADER_release_fence_scope_offset',
    'SDMA_AQL_PKT_COPY_LINEAR_HEADER_release_fence_scope_shift',
    'SDMA_AQL_PKT_COPY_LINEAR_HEADER_reserved_mask',
    'SDMA_AQL_PKT_COPY_LINEAR_HEADER_reserved_offset',
    'SDMA_AQL_PKT_COPY_LINEAR_HEADER_reserved_shift',
    'SDMA_AQL_PKT_COPY_LINEAR_HEADER_subop_mask',
    'SDMA_AQL_PKT_COPY_LINEAR_HEADER_subop_offset',
    'SDMA_AQL_PKT_COPY_LINEAR_HEADER_subop_shift',
    'SDMA_AQL_PKT_COPY_LINEAR_PARAMETER_dst_sw_mask',
    'SDMA_AQL_PKT_COPY_LINEAR_PARAMETER_dst_sw_offset',
    'SDMA_AQL_PKT_COPY_LINEAR_PARAMETER_dst_sw_shift',
    'SDMA_AQL_PKT_COPY_LINEAR_PARAMETER_src_sw_mask',
    'SDMA_AQL_PKT_COPY_LINEAR_PARAMETER_src_sw_offset',
    'SDMA_AQL_PKT_COPY_LINEAR_PARAMETER_src_sw_shift',
    'SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW10_reserved_dw10_mask',
    'SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW10_reserved_dw10_offset',
    'SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW10_reserved_dw10_shift',
    'SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW11_reserved_dw11_mask',
    'SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW11_reserved_dw11_offset',
    'SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW11_reserved_dw11_shift',
    'SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW12_reserved_dw12_mask',
    'SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW12_reserved_dw12_offset',
    'SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW12_reserved_dw12_shift',
    'SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW13_reserved_dw13_mask',
    'SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW13_reserved_dw13_offset',
    'SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW13_reserved_dw13_shift',
    'SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW1_reserved_dw1_mask',
    'SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW1_reserved_dw1_offset',
    'SDMA_AQL_PKT_COPY_LINEAR_RESERVED_DW1_reserved_dw1_shift',
    'SDMA_AQL_PKT_COPY_LINEAR_RETURN_ADDR_HI_return_addr_63_32_mask',
    'SDMA_AQL_PKT_COPY_LINEAR_RETURN_ADDR_HI_return_addr_63_32_offset',
    'SDMA_AQL_PKT_COPY_LINEAR_RETURN_ADDR_HI_return_addr_63_32_shift',
    'SDMA_AQL_PKT_COPY_LINEAR_RETURN_ADDR_LO_return_addr_31_0_mask',
    'SDMA_AQL_PKT_COPY_LINEAR_RETURN_ADDR_LO_return_addr_31_0_offset',
    'SDMA_AQL_PKT_COPY_LINEAR_RETURN_ADDR_LO_return_addr_31_0_shift',
    'SDMA_AQL_PKT_COPY_LINEAR_SRC_ADDR_HI_src_addr_63_32_mask',
    'SDMA_AQL_PKT_COPY_LINEAR_SRC_ADDR_HI_src_addr_63_32_offset',
    'SDMA_AQL_PKT_COPY_LINEAR_SRC_ADDR_HI_src_addr_63_32_shift',
    'SDMA_AQL_PKT_COPY_LINEAR_SRC_ADDR_LO_src_addr_31_0_mask',
    'SDMA_AQL_PKT_COPY_LINEAR_SRC_ADDR_LO_src_addr_31_0_offset',
    'SDMA_AQL_PKT_COPY_LINEAR_SRC_ADDR_LO_src_addr_31_0_shift',
    'SDMA_AQL_PKT_HEADER_HEADER_acquire_fence_scope_mask',
    'SDMA_AQL_PKT_HEADER_HEADER_acquire_fence_scope_offset',
    'SDMA_AQL_PKT_HEADER_HEADER_acquire_fence_scope_shift',
    'SDMA_AQL_PKT_HEADER_HEADER_barrier_mask',
    'SDMA_AQL_PKT_HEADER_HEADER_barrier_offset',
    'SDMA_AQL_PKT_HEADER_HEADER_barrier_shift',
    'SDMA_AQL_PKT_HEADER_HEADER_format_mask',
    'SDMA_AQL_PKT_HEADER_HEADER_format_offset',
    'SDMA_AQL_PKT_HEADER_HEADER_format_shift',
    'SDMA_AQL_PKT_HEADER_HEADER_op_mask',
    'SDMA_AQL_PKT_HEADER_HEADER_op_offset',
    'SDMA_AQL_PKT_HEADER_HEADER_op_shift',
    'SDMA_AQL_PKT_HEADER_HEADER_release_fence_scope_mask',
    'SDMA_AQL_PKT_HEADER_HEADER_release_fence_scope_offset',
    'SDMA_AQL_PKT_HEADER_HEADER_release_fence_scope_shift',
    'SDMA_AQL_PKT_HEADER_HEADER_reserved_mask',
    'SDMA_AQL_PKT_HEADER_HEADER_reserved_offset',
    'SDMA_AQL_PKT_HEADER_HEADER_reserved_shift',
    'SDMA_AQL_PKT_HEADER_HEADER_subop_mask',
    'SDMA_AQL_PKT_HEADER_HEADER_subop_offset',
    'SDMA_AQL_PKT_HEADER_HEADER_subop_shift', 'SDMA_ATOMIC_ADD64',
    'SDMA_GCR_GL1_INV', 'SDMA_GCR_GL2_DISCARD', 'SDMA_GCR_GL2_INV',
    'SDMA_GCR_GL2_US', 'SDMA_GCR_GL2_WB', 'SDMA_GCR_GLK_INV',
    'SDMA_GCR_GLK_WB', 'SDMA_GCR_GLM_INV', 'SDMA_GCR_GLM_WB',
    'SDMA_GCR_GLV_INV', 'SDMA_GCR_RANGE_IS_PA',
    'SDMA_OP_AQL_BARRIER_OR', 'SDMA_OP_AQL_COPY', 'SDMA_OP_ATOMIC',
    'SDMA_OP_COND_EXE', 'SDMA_OP_CONST_FILL', 'SDMA_OP_COPY',
    'SDMA_OP_DUMMY_TRAP', 'SDMA_OP_FENCE', 'SDMA_OP_GCR',
    'SDMA_OP_GCR_REQ', 'SDMA_OP_GPUVM_INV', 'SDMA_OP_INDIRECT',
    'SDMA_OP_NOP', 'SDMA_OP_POLL_REGMEM', 'SDMA_OP_PRE_EXE',
    'SDMA_OP_PTEPDE', 'SDMA_OP_SEM', 'SDMA_OP_SRBM_WRITE',
    'SDMA_OP_TIMESTAMP', 'SDMA_OP_TRAP', 'SDMA_OP_WRITE',
    'SDMA_PKT_ATOMIC', 'SDMA_PKT_ATOMIC_ADDR_HI_addr_63_32_mask',
    'SDMA_PKT_ATOMIC_ADDR_HI_addr_63_32_offset',
    'SDMA_PKT_ATOMIC_ADDR_HI_addr_63_32_shift',
    'SDMA_PKT_ATOMIC_ADDR_LO_addr_31_0_mask',
    'SDMA_PKT_ATOMIC_ADDR_LO_addr_31_0_offset',
    'SDMA_PKT_ATOMIC_ADDR_LO_addr_31_0_shift',
    'SDMA_PKT_ATOMIC_CMP_DATA_HI_cmp_data_63_32_mask',
    'SDMA_PKT_ATOMIC_CMP_DATA_HI_cmp_data_63_32_offset',
    'SDMA_PKT_ATOMIC_CMP_DATA_HI_cmp_data_63_32_shift',
    'SDMA_PKT_ATOMIC_CMP_DATA_LO_cmp_data_31_0_mask',
    'SDMA_PKT_ATOMIC_CMP_DATA_LO_cmp_data_31_0_offset',
    'SDMA_PKT_ATOMIC_CMP_DATA_LO_cmp_data_31_0_shift',
    'SDMA_PKT_ATOMIC_HEADER_atomic_op_mask',
    'SDMA_PKT_ATOMIC_HEADER_atomic_op_offset',
    'SDMA_PKT_ATOMIC_HEADER_atomic_op_shift',
    'SDMA_PKT_ATOMIC_HEADER_loop_mask',
    'SDMA_PKT_ATOMIC_HEADER_loop_offset',
    'SDMA_PKT_ATOMIC_HEADER_loop_shift',
    'SDMA_PKT_ATOMIC_HEADER_op_mask',
    'SDMA_PKT_ATOMIC_HEADER_op_offset',
    'SDMA_PKT_ATOMIC_HEADER_op_shift',
    'SDMA_PKT_ATOMIC_HEADER_tmz_mask',
    'SDMA_PKT_ATOMIC_HEADER_tmz_offset',
    'SDMA_PKT_ATOMIC_HEADER_tmz_shift',
    'SDMA_PKT_ATOMIC_LOOP_INTERVAL_loop_interval_mask',
    'SDMA_PKT_ATOMIC_LOOP_INTERVAL_loop_interval_offset',
    'SDMA_PKT_ATOMIC_LOOP_INTERVAL_loop_interval_shift',
    'SDMA_PKT_ATOMIC_SRC_DATA_HI_src_data_63_32_mask',
    'SDMA_PKT_ATOMIC_SRC_DATA_HI_src_data_63_32_offset',
    'SDMA_PKT_ATOMIC_SRC_DATA_HI_src_data_63_32_shift',
    'SDMA_PKT_ATOMIC_SRC_DATA_LO_src_data_31_0_mask',
    'SDMA_PKT_ATOMIC_SRC_DATA_LO_src_data_31_0_offset',
    'SDMA_PKT_ATOMIC_SRC_DATA_LO_src_data_31_0_shift',
    'SDMA_PKT_COND_EXE_ADDR_HI_addr_63_32_mask',
    'SDMA_PKT_COND_EXE_ADDR_HI_addr_63_32_offset',
    'SDMA_PKT_COND_EXE_ADDR_HI_addr_63_32_shift',
    'SDMA_PKT_COND_EXE_ADDR_LO_addr_31_0_mask',
    'SDMA_PKT_COND_EXE_ADDR_LO_addr_31_0_offset',
    'SDMA_PKT_COND_EXE_ADDR_LO_addr_31_0_shift',
    'SDMA_PKT_COND_EXE_EXEC_COUNT_exec_count_mask',
    'SDMA_PKT_COND_EXE_EXEC_COUNT_exec_count_offset',
    'SDMA_PKT_COND_EXE_EXEC_COUNT_exec_count_shift',
    'SDMA_PKT_COND_EXE_HEADER_op_mask',
    'SDMA_PKT_COND_EXE_HEADER_op_offset',
    'SDMA_PKT_COND_EXE_HEADER_op_shift',
    'SDMA_PKT_COND_EXE_HEADER_sub_op_mask',
    'SDMA_PKT_COND_EXE_HEADER_sub_op_offset',
    'SDMA_PKT_COND_EXE_HEADER_sub_op_shift',
    'SDMA_PKT_COND_EXE_REFERENCE_reference_mask',
    'SDMA_PKT_COND_EXE_REFERENCE_reference_offset',
    'SDMA_PKT_COND_EXE_REFERENCE_reference_shift',
    'SDMA_PKT_CONSTANT_FILL',
    'SDMA_PKT_CONSTANT_FILL_COUNT_count_mask',
    'SDMA_PKT_CONSTANT_FILL_COUNT_count_offset',
    'SDMA_PKT_CONSTANT_FILL_COUNT_count_shift',
    'SDMA_PKT_CONSTANT_FILL_DATA_src_data_31_0_mask',
    'SDMA_PKT_CONSTANT_FILL_DATA_src_data_31_0_offset',
    'SDMA_PKT_CONSTANT_FILL_DATA_src_data_31_0_shift',
    'SDMA_PKT_CONSTANT_FILL_DST_ADDR_HI_dst_addr_63_32_mask',
    'SDMA_PKT_CONSTANT_FILL_DST_ADDR_HI_dst_addr_63_32_offset',
    'SDMA_PKT_CONSTANT_FILL_DST_ADDR_HI_dst_addr_63_32_shift',
    'SDMA_PKT_CONSTANT_FILL_DST_ADDR_LO_dst_addr_31_0_mask',
    'SDMA_PKT_CONSTANT_FILL_DST_ADDR_LO_dst_addr_31_0_offset',
    'SDMA_PKT_CONSTANT_FILL_DST_ADDR_LO_dst_addr_31_0_shift',
    'SDMA_PKT_CONSTANT_FILL_HEADER_fillsize_mask',
    'SDMA_PKT_CONSTANT_FILL_HEADER_fillsize_offset',
    'SDMA_PKT_CONSTANT_FILL_HEADER_fillsize_shift',
    'SDMA_PKT_CONSTANT_FILL_HEADER_op_mask',
    'SDMA_PKT_CONSTANT_FILL_HEADER_op_offset',
    'SDMA_PKT_CONSTANT_FILL_HEADER_op_shift',
    'SDMA_PKT_CONSTANT_FILL_HEADER_sub_op_mask',
    'SDMA_PKT_CONSTANT_FILL_HEADER_sub_op_offset',
    'SDMA_PKT_CONSTANT_FILL_HEADER_sub_op_shift',
    'SDMA_PKT_CONSTANT_FILL_HEADER_sw_mask',
    'SDMA_PKT_CONSTANT_FILL_HEADER_sw_offset',
    'SDMA_PKT_CONSTANT_FILL_HEADER_sw_shift',
    'SDMA_PKT_COPY_BROADCAST_LINEAR_COUNT_count_mask',
    'SDMA_PKT_COPY_BROADCAST_LINEAR_COUNT_count_offset',
    'SDMA_PKT_COPY_BROADCAST_LINEAR_COUNT_count_shift',
    'SDMA_PKT_COPY_BROADCAST_LINEAR_DST1_ADDR_HI_dst1_addr_63_32_mask',
    'SDMA_PKT_COPY_BROADCAST_LINEAR_DST1_ADDR_HI_dst1_addr_63_32_offset',
    'SDMA_PKT_COPY_BROADCAST_LINEAR_DST1_ADDR_HI_dst1_addr_63_32_shift',
    'SDMA_PKT_COPY_BROADCAST_LINEAR_DST1_ADDR_LO_dst1_addr_31_0_mask',
    'SDMA_PKT_COPY_BROADCAST_LINEAR_DST1_ADDR_LO_dst1_addr_31_0_offset',
    'SDMA_PKT_COPY_BROADCAST_LINEAR_DST1_ADDR_LO_dst1_addr_31_0_shift',
    'SDMA_PKT_COPY_BROADCAST_LINEAR_DST2_ADDR_HI_dst2_addr_63_32_mask',
    'SDMA_PKT_COPY_BROADCAST_LINEAR_DST2_ADDR_HI_dst2_addr_63_32_offset',
    'SDMA_PKT_COPY_BROADCAST_LINEAR_DST2_ADDR_HI_dst2_addr_63_32_shift',
    'SDMA_PKT_COPY_BROADCAST_LINEAR_DST2_ADDR_LO_dst2_addr_31_0_mask',
    'SDMA_PKT_COPY_BROADCAST_LINEAR_DST2_ADDR_LO_dst2_addr_31_0_offset',
    'SDMA_PKT_COPY_BROADCAST_LINEAR_DST2_ADDR_LO_dst2_addr_31_0_shift',
    'SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_broadcast_mask',
    'SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_broadcast_offset',
    'SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_broadcast_shift',
    'SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_encrypt_mask',
    'SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_encrypt_offset',
    'SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_encrypt_shift',
    'SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_op_mask',
    'SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_op_offset',
    'SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_op_shift',
    'SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_sub_op_mask',
    'SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_sub_op_offset',
    'SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_sub_op_shift',
    'SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_tmz_mask',
    'SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_tmz_offset',
    'SDMA_PKT_COPY_BROADCAST_LINEAR_HEADER_tmz_shift',
    'SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_dst1_sw_mask',
    'SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_dst1_sw_offset',
    'SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_dst1_sw_shift',
    'SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_dst2_sw_mask',
    'SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_dst2_sw_offset',
    'SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_dst2_sw_shift',
    'SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_src_sw_mask',
    'SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_src_sw_offset',
    'SDMA_PKT_COPY_BROADCAST_LINEAR_PARAMETER_src_sw_shift',
    'SDMA_PKT_COPY_BROADCAST_LINEAR_SRC_ADDR_HI_src_addr_63_32_mask',
    'SDMA_PKT_COPY_BROADCAST_LINEAR_SRC_ADDR_HI_src_addr_63_32_offset',
    'SDMA_PKT_COPY_BROADCAST_LINEAR_SRC_ADDR_HI_src_addr_63_32_shift',
    'SDMA_PKT_COPY_BROADCAST_LINEAR_SRC_ADDR_LO_src_addr_31_0_mask',
    'SDMA_PKT_COPY_BROADCAST_LINEAR_SRC_ADDR_LO_src_addr_31_0_offset',
    'SDMA_PKT_COPY_BROADCAST_LINEAR_SRC_ADDR_LO_src_addr_31_0_shift',
    'SDMA_PKT_COPY_DIRTY_PAGE_COUNT_count_mask',
    'SDMA_PKT_COPY_DIRTY_PAGE_COUNT_count_offset',
    'SDMA_PKT_COPY_DIRTY_PAGE_COUNT_count_shift',
    'SDMA_PKT_COPY_DIRTY_PAGE_DST_ADDR_HI_dst_addr_63_32_mask',
    'SDMA_PKT_COPY_DIRTY_PAGE_DST_ADDR_HI_dst_addr_63_32_offset',
    'SDMA_PKT_COPY_DIRTY_PAGE_DST_ADDR_HI_dst_addr_63_32_shift',
    'SDMA_PKT_COPY_DIRTY_PAGE_DST_ADDR_LO_dst_addr_31_0_mask',
    'SDMA_PKT_COPY_DIRTY_PAGE_DST_ADDR_LO_dst_addr_31_0_offset',
    'SDMA_PKT_COPY_DIRTY_PAGE_DST_ADDR_LO_dst_addr_31_0_shift',
    'SDMA_PKT_COPY_DIRTY_PAGE_HEADER_all_mask',
    'SDMA_PKT_COPY_DIRTY_PAGE_HEADER_all_offset',
    'SDMA_PKT_COPY_DIRTY_PAGE_HEADER_all_shift',
    'SDMA_PKT_COPY_DIRTY_PAGE_HEADER_op_mask',
    'SDMA_PKT_COPY_DIRTY_PAGE_HEADER_op_offset',
    'SDMA_PKT_COPY_DIRTY_PAGE_HEADER_op_shift',
    'SDMA_PKT_COPY_DIRTY_PAGE_HEADER_sub_op_mask',
    'SDMA_PKT_COPY_DIRTY_PAGE_HEADER_sub_op_offset',
    'SDMA_PKT_COPY_DIRTY_PAGE_HEADER_sub_op_shift',
    'SDMA_PKT_COPY_DIRTY_PAGE_HEADER_tmz_mask',
    'SDMA_PKT_COPY_DIRTY_PAGE_HEADER_tmz_offset',
    'SDMA_PKT_COPY_DIRTY_PAGE_HEADER_tmz_shift',
    'SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_gcc_mask',
    'SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_gcc_offset',
    'SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_gcc_shift',
    'SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_gpa_mask',
    'SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_gpa_offset',
    'SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_gpa_shift',
    'SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_l2_policy_mask',
    'SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_l2_policy_offset',
    'SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_l2_policy_shift',
    'SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_mtype_mask',
    'SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_mtype_offset',
    'SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_mtype_shift',
    'SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_snoop_mask',
    'SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_snoop_offset',
    'SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_snoop_shift',
    'SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_sw_mask',
    'SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_sw_offset',
    'SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_sw_shift',
    'SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_sys_mask',
    'SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_sys_offset',
    'SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_dst_sys_shift',
    'SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_gpa_mask',
    'SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_gpa_offset',
    'SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_gpa_shift',
    'SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_l2_policy_mask',
    'SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_l2_policy_offset',
    'SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_l2_policy_shift',
    'SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_mtype_mask',
    'SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_mtype_offset',
    'SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_mtype_shift',
    'SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_snoop_mask',
    'SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_snoop_offset',
    'SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_snoop_shift',
    'SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_sw_mask',
    'SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_sw_offset',
    'SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_sw_shift',
    'SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_sys_mask',
    'SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_sys_offset',
    'SDMA_PKT_COPY_DIRTY_PAGE_PARAMETER_src_sys_shift',
    'SDMA_PKT_COPY_DIRTY_PAGE_SRC_ADDR_HI_src_addr_63_32_mask',
    'SDMA_PKT_COPY_DIRTY_PAGE_SRC_ADDR_HI_src_addr_63_32_offset',
    'SDMA_PKT_COPY_DIRTY_PAGE_SRC_ADDR_HI_src_addr_63_32_shift',
    'SDMA_PKT_COPY_DIRTY_PAGE_SRC_ADDR_LO_src_addr_31_0_mask',
    'SDMA_PKT_COPY_DIRTY_PAGE_SRC_ADDR_LO_src_addr_31_0_offset',
    'SDMA_PKT_COPY_DIRTY_PAGE_SRC_ADDR_LO_src_addr_31_0_shift',
    'SDMA_PKT_COPY_L2T_BROADCAST_COUNT_count_mask',
    'SDMA_PKT_COPY_L2T_BROADCAST_COUNT_count_offset',
    'SDMA_PKT_COPY_L2T_BROADCAST_COUNT_count_shift',
    'SDMA_PKT_COPY_L2T_BROADCAST_DW_10_dst2_sw_mask',
    'SDMA_PKT_COPY_L2T_BROADCAST_DW_10_dst2_sw_offset',
    'SDMA_PKT_COPY_L2T_BROADCAST_DW_10_dst2_sw_shift',
    'SDMA_PKT_COPY_L2T_BROADCAST_DW_10_linear_sw_mask',
    'SDMA_PKT_COPY_L2T_BROADCAST_DW_10_linear_sw_offset',
    'SDMA_PKT_COPY_L2T_BROADCAST_DW_10_linear_sw_shift',
    'SDMA_PKT_COPY_L2T_BROADCAST_DW_10_tile_sw_mask',
    'SDMA_PKT_COPY_L2T_BROADCAST_DW_10_tile_sw_offset',
    'SDMA_PKT_COPY_L2T_BROADCAST_DW_10_tile_sw_shift',
    'SDMA_PKT_COPY_L2T_BROADCAST_DW_5_width_mask',
    'SDMA_PKT_COPY_L2T_BROADCAST_DW_5_width_offset',
    'SDMA_PKT_COPY_L2T_BROADCAST_DW_5_width_shift',
    'SDMA_PKT_COPY_L2T_BROADCAST_DW_6_depth_mask',
    'SDMA_PKT_COPY_L2T_BROADCAST_DW_6_depth_offset',
    'SDMA_PKT_COPY_L2T_BROADCAST_DW_6_depth_shift',
    'SDMA_PKT_COPY_L2T_BROADCAST_DW_6_height_mask',
    'SDMA_PKT_COPY_L2T_BROADCAST_DW_6_height_offset',
    'SDMA_PKT_COPY_L2T_BROADCAST_DW_6_height_shift',
    'SDMA_PKT_COPY_L2T_BROADCAST_DW_7_dimension_mask',
    'SDMA_PKT_COPY_L2T_BROADCAST_DW_7_dimension_offset',
    'SDMA_PKT_COPY_L2T_BROADCAST_DW_7_dimension_shift',
    'SDMA_PKT_COPY_L2T_BROADCAST_DW_7_element_size_mask',
    'SDMA_PKT_COPY_L2T_BROADCAST_DW_7_element_size_offset',
    'SDMA_PKT_COPY_L2T_BROADCAST_DW_7_element_size_shift',
    'SDMA_PKT_COPY_L2T_BROADCAST_DW_7_mip_max_mask',
    'SDMA_PKT_COPY_L2T_BROADCAST_DW_7_mip_max_offset',
    'SDMA_PKT_COPY_L2T_BROADCAST_DW_7_mip_max_shift',
    'SDMA_PKT_COPY_L2T_BROADCAST_DW_7_swizzle_mode_mask',
    'SDMA_PKT_COPY_L2T_BROADCAST_DW_7_swizzle_mode_offset',
    'SDMA_PKT_COPY_L2T_BROADCAST_DW_7_swizzle_mode_shift',
    'SDMA_PKT_COPY_L2T_BROADCAST_DW_8_x_mask',
    'SDMA_PKT_COPY_L2T_BROADCAST_DW_8_x_offset',
    'SDMA_PKT_COPY_L2T_BROADCAST_DW_8_x_shift',
    'SDMA_PKT_COPY_L2T_BROADCAST_DW_8_y_mask',
    'SDMA_PKT_COPY_L2T_BROADCAST_DW_8_y_offset',
    'SDMA_PKT_COPY_L2T_BROADCAST_DW_8_y_shift',
    'SDMA_PKT_COPY_L2T_BROADCAST_DW_9_z_mask',
    'SDMA_PKT_COPY_L2T_BROADCAST_DW_9_z_offset',
    'SDMA_PKT_COPY_L2T_BROADCAST_DW_9_z_shift',
    'SDMA_PKT_COPY_L2T_BROADCAST_HEADER_broadcast_mask',
    'SDMA_PKT_COPY_L2T_BROADCAST_HEADER_broadcast_offset',
    'SDMA_PKT_COPY_L2T_BROADCAST_HEADER_broadcast_shift',
    'SDMA_PKT_COPY_L2T_BROADCAST_HEADER_encrypt_mask',
    'SDMA_PKT_COPY_L2T_BROADCAST_HEADER_encrypt_offset',
    'SDMA_PKT_COPY_L2T_BROADCAST_HEADER_encrypt_shift',
    'SDMA_PKT_COPY_L2T_BROADCAST_HEADER_op_mask',
    'SDMA_PKT_COPY_L2T_BROADCAST_HEADER_op_offset',
    'SDMA_PKT_COPY_L2T_BROADCAST_HEADER_op_shift',
    'SDMA_PKT_COPY_L2T_BROADCAST_HEADER_sub_op_mask',
    'SDMA_PKT_COPY_L2T_BROADCAST_HEADER_sub_op_offset',
    'SDMA_PKT_COPY_L2T_BROADCAST_HEADER_sub_op_shift',
    'SDMA_PKT_COPY_L2T_BROADCAST_HEADER_tmz_mask',
    'SDMA_PKT_COPY_L2T_BROADCAST_HEADER_tmz_offset',
    'SDMA_PKT_COPY_L2T_BROADCAST_HEADER_tmz_shift',
    'SDMA_PKT_COPY_L2T_BROADCAST_HEADER_videocopy_mask',
    'SDMA_PKT_COPY_L2T_BROADCAST_HEADER_videocopy_offset',
    'SDMA_PKT_COPY_L2T_BROADCAST_HEADER_videocopy_shift',
    'SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_ADDR_HI_linear_addr_63_32_mask',
    'SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_ADDR_HI_linear_addr_63_32_offset',
    'SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_ADDR_HI_linear_addr_63_32_shift',
    'SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_ADDR_LO_linear_addr_31_0_mask',
    'SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_ADDR_LO_linear_addr_31_0_offset',
    'SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_ADDR_LO_linear_addr_31_0_shift',
    'SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_PITCH_linear_pitch_mask',
    'SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_PITCH_linear_pitch_offset',
    'SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_PITCH_linear_pitch_shift',
    'SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_SLICE_PITCH_linear_slice_pitch_mask',
    'SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_SLICE_PITCH_linear_slice_pitch_offset',
    'SDMA_PKT_COPY_L2T_BROADCAST_LINEAR_SLICE_PITCH_linear_slice_pitch_shift',
    'SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_HI_0_tiled_addr0_63_32_mask',
    'SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_HI_0_tiled_addr0_63_32_offset',
    'SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_HI_0_tiled_addr0_63_32_shift',
    'SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_HI_1_tiled_addr1_63_32_mask',
    'SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_HI_1_tiled_addr1_63_32_offset',
    'SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_HI_1_tiled_addr1_63_32_shift',
    'SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_LO_0_tiled_addr0_31_0_mask',
    'SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_LO_0_tiled_addr0_31_0_offset',
    'SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_LO_0_tiled_addr0_31_0_shift',
    'SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_LO_1_tiled_addr1_31_0_mask',
    'SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_LO_1_tiled_addr1_31_0_offset',
    'SDMA_PKT_COPY_L2T_BROADCAST_TILED_ADDR_LO_1_tiled_addr1_31_0_shift',
    'SDMA_PKT_COPY_LINEAR',
    'SDMA_PKT_COPY_LINEAR_BC_COUNT_count_mask',
    'SDMA_PKT_COPY_LINEAR_BC_COUNT_count_offset',
    'SDMA_PKT_COPY_LINEAR_BC_COUNT_count_shift',
    'SDMA_PKT_COPY_LINEAR_BC_DST_ADDR_HI_dst_addr_63_32_mask',
    'SDMA_PKT_COPY_LINEAR_BC_DST_ADDR_HI_dst_addr_63_32_offset',
    'SDMA_PKT_COPY_LINEAR_BC_DST_ADDR_HI_dst_addr_63_32_shift',
    'SDMA_PKT_COPY_LINEAR_BC_DST_ADDR_LO_dst_addr_31_0_mask',
    'SDMA_PKT_COPY_LINEAR_BC_DST_ADDR_LO_dst_addr_31_0_offset',
    'SDMA_PKT_COPY_LINEAR_BC_DST_ADDR_LO_dst_addr_31_0_shift',
    'SDMA_PKT_COPY_LINEAR_BC_HEADER_op_mask',
    'SDMA_PKT_COPY_LINEAR_BC_HEADER_op_offset',
    'SDMA_PKT_COPY_LINEAR_BC_HEADER_op_shift',
    'SDMA_PKT_COPY_LINEAR_BC_HEADER_sub_op_mask',
    'SDMA_PKT_COPY_LINEAR_BC_HEADER_sub_op_offset',
    'SDMA_PKT_COPY_LINEAR_BC_HEADER_sub_op_shift',
    'SDMA_PKT_COPY_LINEAR_BC_PARAMETER_dst_ha_mask',
    'SDMA_PKT_COPY_LINEAR_BC_PARAMETER_dst_ha_offset',
    'SDMA_PKT_COPY_LINEAR_BC_PARAMETER_dst_ha_shift',
    'SDMA_PKT_COPY_LINEAR_BC_PARAMETER_dst_sw_mask',
    'SDMA_PKT_COPY_LINEAR_BC_PARAMETER_dst_sw_offset',
    'SDMA_PKT_COPY_LINEAR_BC_PARAMETER_dst_sw_shift',
    'SDMA_PKT_COPY_LINEAR_BC_PARAMETER_src_ha_mask',
    'SDMA_PKT_COPY_LINEAR_BC_PARAMETER_src_ha_offset',
    'SDMA_PKT_COPY_LINEAR_BC_PARAMETER_src_ha_shift',
    'SDMA_PKT_COPY_LINEAR_BC_PARAMETER_src_sw_mask',
    'SDMA_PKT_COPY_LINEAR_BC_PARAMETER_src_sw_offset',
    'SDMA_PKT_COPY_LINEAR_BC_PARAMETER_src_sw_shift',
    'SDMA_PKT_COPY_LINEAR_BC_SRC_ADDR_HI_src_addr_63_32_mask',
    'SDMA_PKT_COPY_LINEAR_BC_SRC_ADDR_HI_src_addr_63_32_offset',
    'SDMA_PKT_COPY_LINEAR_BC_SRC_ADDR_HI_src_addr_63_32_shift',
    'SDMA_PKT_COPY_LINEAR_BC_SRC_ADDR_LO_src_addr_31_0_mask',
    'SDMA_PKT_COPY_LINEAR_BC_SRC_ADDR_LO_src_addr_31_0_offset',
    'SDMA_PKT_COPY_LINEAR_BC_SRC_ADDR_LO_src_addr_31_0_shift',
    'SDMA_PKT_COPY_LINEAR_COUNT_count_mask',
    'SDMA_PKT_COPY_LINEAR_COUNT_count_offset',
    'SDMA_PKT_COPY_LINEAR_COUNT_count_shift',
    'SDMA_PKT_COPY_LINEAR_DST_ADDR_HI_dst_addr_63_32_mask',
    'SDMA_PKT_COPY_LINEAR_DST_ADDR_HI_dst_addr_63_32_offset',
    'SDMA_PKT_COPY_LINEAR_DST_ADDR_HI_dst_addr_63_32_shift',
    'SDMA_PKT_COPY_LINEAR_DST_ADDR_LO_dst_addr_31_0_mask',
    'SDMA_PKT_COPY_LINEAR_DST_ADDR_LO_dst_addr_31_0_offset',
    'SDMA_PKT_COPY_LINEAR_DST_ADDR_LO_dst_addr_31_0_shift',
    'SDMA_PKT_COPY_LINEAR_HEADER_backwards_mask',
    'SDMA_PKT_COPY_LINEAR_HEADER_backwards_offset',
    'SDMA_PKT_COPY_LINEAR_HEADER_backwards_shift',
    'SDMA_PKT_COPY_LINEAR_HEADER_broadcast_mask',
    'SDMA_PKT_COPY_LINEAR_HEADER_broadcast_offset',
    'SDMA_PKT_COPY_LINEAR_HEADER_broadcast_shift',
    'SDMA_PKT_COPY_LINEAR_HEADER_encrypt_mask',
    'SDMA_PKT_COPY_LINEAR_HEADER_encrypt_offset',
    'SDMA_PKT_COPY_LINEAR_HEADER_encrypt_shift',
    'SDMA_PKT_COPY_LINEAR_HEADER_op_mask',
    'SDMA_PKT_COPY_LINEAR_HEADER_op_offset',
    'SDMA_PKT_COPY_LINEAR_HEADER_op_shift',
    'SDMA_PKT_COPY_LINEAR_HEADER_sub_op_mask',
    'SDMA_PKT_COPY_LINEAR_HEADER_sub_op_offset',
    'SDMA_PKT_COPY_LINEAR_HEADER_sub_op_shift',
    'SDMA_PKT_COPY_LINEAR_HEADER_tmz_mask',
    'SDMA_PKT_COPY_LINEAR_HEADER_tmz_offset',
    'SDMA_PKT_COPY_LINEAR_HEADER_tmz_shift',
    'SDMA_PKT_COPY_LINEAR_PARAMETER_dst_sw_mask',
    'SDMA_PKT_COPY_LINEAR_PARAMETER_dst_sw_offset',
    'SDMA_PKT_COPY_LINEAR_PARAMETER_dst_sw_shift',
    'SDMA_PKT_COPY_LINEAR_PARAMETER_src_sw_mask',
    'SDMA_PKT_COPY_LINEAR_PARAMETER_src_sw_offset',
    'SDMA_PKT_COPY_LINEAR_PARAMETER_src_sw_shift',
    'SDMA_PKT_COPY_LINEAR_RECT',
    'SDMA_PKT_COPY_LINEAR_SRC_ADDR_HI_src_addr_63_32_mask',
    'SDMA_PKT_COPY_LINEAR_SRC_ADDR_HI_src_addr_63_32_offset',
    'SDMA_PKT_COPY_LINEAR_SRC_ADDR_HI_src_addr_63_32_shift',
    'SDMA_PKT_COPY_LINEAR_SRC_ADDR_LO_src_addr_31_0_mask',
    'SDMA_PKT_COPY_LINEAR_SRC_ADDR_LO_src_addr_31_0_offset',
    'SDMA_PKT_COPY_LINEAR_SRC_ADDR_LO_src_addr_31_0_shift',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DST_ADDR_HI_dst_addr_63_32_mask',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DST_ADDR_HI_dst_addr_63_32_offset',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DST_ADDR_HI_dst_addr_63_32_shift',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DST_ADDR_LO_dst_addr_31_0_mask',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DST_ADDR_LO_dst_addr_31_0_offset',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DST_ADDR_LO_dst_addr_31_0_shift',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_10_dst_slice_pitch_mask',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_10_dst_slice_pitch_offset',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_10_dst_slice_pitch_shift',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_11_rect_x_mask',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_11_rect_x_offset',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_11_rect_x_shift',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_11_rect_y_mask',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_11_rect_y_offset',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_11_rect_y_shift',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_dst_ha_mask',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_dst_ha_offset',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_dst_ha_shift',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_dst_sw_mask',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_dst_sw_offset',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_dst_sw_shift',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_rect_z_mask',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_rect_z_offset',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_rect_z_shift',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_src_ha_mask',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_src_ha_offset',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_src_ha_shift',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_src_sw_mask',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_src_sw_offset',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_12_src_sw_shift',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_3_src_x_mask',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_3_src_x_offset',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_3_src_x_shift',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_3_src_y_mask',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_3_src_y_offset',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_3_src_y_shift',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_4_src_pitch_mask',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_4_src_pitch_offset',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_4_src_pitch_shift',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_4_src_z_mask',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_4_src_z_offset',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_4_src_z_shift',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_5_src_slice_pitch_mask',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_5_src_slice_pitch_offset',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_5_src_slice_pitch_shift',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_8_dst_x_mask',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_8_dst_x_offset',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_8_dst_x_shift',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_8_dst_y_mask',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_8_dst_y_offset',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_8_dst_y_shift',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_9_dst_pitch_mask',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_9_dst_pitch_offset',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_9_dst_pitch_shift',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_9_dst_z_mask',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_9_dst_z_offset',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_DW_9_dst_z_shift',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_HEADER_elementsize_mask',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_HEADER_elementsize_offset',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_HEADER_elementsize_shift',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_HEADER_op_mask',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_HEADER_op_offset',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_HEADER_op_shift',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_HEADER_sub_op_mask',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_HEADER_sub_op_offset',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_HEADER_sub_op_shift',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_SRC_ADDR_HI_src_addr_63_32_mask',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_SRC_ADDR_HI_src_addr_63_32_offset',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_SRC_ADDR_HI_src_addr_63_32_shift',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_SRC_ADDR_LO_src_addr_31_0_mask',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_SRC_ADDR_LO_src_addr_31_0_offset',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_BC_SRC_ADDR_LO_src_addr_31_0_shift',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_DST_ADDR_HI_dst_addr_63_32_mask',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_DST_ADDR_HI_dst_addr_63_32_offset',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_DST_ADDR_HI_dst_addr_63_32_shift',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_DST_ADDR_LO_dst_addr_31_0_mask',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_DST_ADDR_LO_dst_addr_31_0_offset',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_DST_ADDR_LO_dst_addr_31_0_shift',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_DW_10_dst_slice_pitch_mask',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_DW_10_dst_slice_pitch_offset',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_DW_10_dst_slice_pitch_shift',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_DW_11_rect_x_mask',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_DW_11_rect_x_offset',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_DW_11_rect_x_shift',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_DW_11_rect_y_mask',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_DW_11_rect_y_offset',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_DW_11_rect_y_shift',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_dst_sw_mask',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_dst_sw_offset',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_dst_sw_shift',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_rect_z_mask',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_rect_z_offset',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_rect_z_shift',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_src_sw_mask',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_src_sw_offset',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_DW_12_src_sw_shift',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_DW_3_src_x_mask',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_DW_3_src_x_offset',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_DW_3_src_x_shift',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_DW_3_src_y_mask',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_DW_3_src_y_offset',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_DW_3_src_y_shift',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_DW_4_src_pitch_mask',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_DW_4_src_pitch_offset',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_DW_4_src_pitch_shift',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_DW_4_src_z_mask',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_DW_4_src_z_offset',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_DW_4_src_z_shift',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_DW_5_src_slice_pitch_mask',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_DW_5_src_slice_pitch_offset',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_DW_5_src_slice_pitch_shift',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_DW_8_dst_x_mask',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_DW_8_dst_x_offset',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_DW_8_dst_x_shift',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_DW_8_dst_y_mask',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_DW_8_dst_y_offset',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_DW_8_dst_y_shift',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_DW_9_dst_pitch_mask',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_DW_9_dst_pitch_offset',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_DW_9_dst_pitch_shift',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_DW_9_dst_z_mask',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_DW_9_dst_z_offset',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_DW_9_dst_z_shift',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_elementsize_mask',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_elementsize_offset',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_elementsize_shift',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_op_mask',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_op_offset',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_op_shift',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_sub_op_mask',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_sub_op_offset',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_sub_op_shift',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_tmz_mask',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_tmz_offset',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_HEADER_tmz_shift',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_SRC_ADDR_HI_src_addr_63_32_mask',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_SRC_ADDR_HI_src_addr_63_32_offset',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_SRC_ADDR_HI_src_addr_63_32_shift',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_SRC_ADDR_LO_src_addr_31_0_mask',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_SRC_ADDR_LO_src_addr_31_0_offset',
    'SDMA_PKT_COPY_LINEAR_SUBWIN_SRC_ADDR_LO_src_addr_31_0_shift',
    'SDMA_PKT_COPY_PHYSICAL_LINEAR_COUNT_count_mask',
    'SDMA_PKT_COPY_PHYSICAL_LINEAR_COUNT_count_offset',
    'SDMA_PKT_COPY_PHYSICAL_LINEAR_COUNT_count_shift',
    'SDMA_PKT_COPY_PHYSICAL_LINEAR_DST_ADDR_HI_dst_addr_63_32_mask',
    'SDMA_PKT_COPY_PHYSICAL_LINEAR_DST_ADDR_HI_dst_addr_63_32_offset',
    'SDMA_PKT_COPY_PHYSICAL_LINEAR_DST_ADDR_HI_dst_addr_63_32_shift',
    'SDMA_PKT_COPY_PHYSICAL_LINEAR_DST_ADDR_LO_dst_addr_31_0_mask',
    'SDMA_PKT_COPY_PHYSICAL_LINEAR_DST_ADDR_LO_dst_addr_31_0_offset',
    'SDMA_PKT_COPY_PHYSICAL_LINEAR_DST_ADDR_LO_dst_addr_31_0_shift',
    'SDMA_PKT_COPY_PHYSICAL_LINEAR_HEADER_op_mask',
    'SDMA_PKT_COPY_PHYSICAL_LINEAR_HEADER_op_offset',
    'SDMA_PKT_COPY_PHYSICAL_LINEAR_HEADER_op_shift',
    'SDMA_PKT_COPY_PHYSICAL_LINEAR_HEADER_sub_op_mask',
    'SDMA_PKT_COPY_PHYSICAL_LINEAR_HEADER_sub_op_offset',
    'SDMA_PKT_COPY_PHYSICAL_LINEAR_HEADER_sub_op_shift',
    'SDMA_PKT_COPY_PHYSICAL_LINEAR_HEADER_tmz_mask',
    'SDMA_PKT_COPY_PHYSICAL_LINEAR_HEADER_tmz_offset',
    'SDMA_PKT_COPY_PHYSICAL_LINEAR_HEADER_tmz_shift',
    'SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_gcc_mask',
    'SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_gcc_offset',
    'SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_gcc_shift',
    'SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_gpa_mask',
    'SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_gpa_offset',
    'SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_gpa_shift',
    'SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_l2_policy_mask',
    'SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_l2_policy_offset',
    'SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_l2_policy_shift',
    'SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_log_mask',
    'SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_log_offset',
    'SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_log_shift',
    'SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_mtype_mask',
    'SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_mtype_offset',
    'SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_mtype_shift',
    'SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_snoop_mask',
    'SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_snoop_offset',
    'SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_snoop_shift',
    'SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_sw_mask',
    'SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_sw_offset',
    'SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_sw_shift',
    'SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_sys_mask',
    'SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_sys_offset',
    'SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_dst_sys_shift',
    'SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_gcc_mask',
    'SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_gcc_offset',
    'SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_gcc_shift',
    'SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_gpa_mask',
    'SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_gpa_offset',
    'SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_gpa_shift',
    'SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_l2_policy_mask',
    'SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_l2_policy_offset',
    'SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_l2_policy_shift',
    'SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_mtype_mask',
    'SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_mtype_offset',
    'SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_mtype_shift',
    'SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_snoop_mask',
    'SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_snoop_offset',
    'SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_snoop_shift',
    'SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_sw_mask',
    'SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_sw_offset',
    'SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_sw_shift',
    'SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_sys_mask',
    'SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_sys_offset',
    'SDMA_PKT_COPY_PHYSICAL_LINEAR_PARAMETER_src_sys_shift',
    'SDMA_PKT_COPY_PHYSICAL_LINEAR_SRC_ADDR_HI_src_addr_63_32_mask',
    'SDMA_PKT_COPY_PHYSICAL_LINEAR_SRC_ADDR_HI_src_addr_63_32_offset',
    'SDMA_PKT_COPY_PHYSICAL_LINEAR_SRC_ADDR_HI_src_addr_63_32_shift',
    'SDMA_PKT_COPY_PHYSICAL_LINEAR_SRC_ADDR_LO_src_addr_31_0_mask',
    'SDMA_PKT_COPY_PHYSICAL_LINEAR_SRC_ADDR_LO_src_addr_31_0_offset',
    'SDMA_PKT_COPY_PHYSICAL_LINEAR_SRC_ADDR_LO_src_addr_31_0_shift',
    'SDMA_PKT_COPY_STRUCT_COUNT_count_mask',
    'SDMA_PKT_COPY_STRUCT_COUNT_count_offset',
    'SDMA_PKT_COPY_STRUCT_COUNT_count_shift',
    'SDMA_PKT_COPY_STRUCT_DW_5_linear_sw_mask',
    'SDMA_PKT_COPY_STRUCT_DW_5_linear_sw_offset',
    'SDMA_PKT_COPY_STRUCT_DW_5_linear_sw_shift',
    'SDMA_PKT_COPY_STRUCT_DW_5_stride_mask',
    'SDMA_PKT_COPY_STRUCT_DW_5_stride_offset',
    'SDMA_PKT_COPY_STRUCT_DW_5_stride_shift',
    'SDMA_PKT_COPY_STRUCT_DW_5_struct_sw_mask',
    'SDMA_PKT_COPY_STRUCT_DW_5_struct_sw_offset',
    'SDMA_PKT_COPY_STRUCT_DW_5_struct_sw_shift',
    'SDMA_PKT_COPY_STRUCT_HEADER_detile_mask',
    'SDMA_PKT_COPY_STRUCT_HEADER_detile_offset',
    'SDMA_PKT_COPY_STRUCT_HEADER_detile_shift',
    'SDMA_PKT_COPY_STRUCT_HEADER_op_mask',
    'SDMA_PKT_COPY_STRUCT_HEADER_op_offset',
    'SDMA_PKT_COPY_STRUCT_HEADER_op_shift',
    'SDMA_PKT_COPY_STRUCT_HEADER_sub_op_mask',
    'SDMA_PKT_COPY_STRUCT_HEADER_sub_op_offset',
    'SDMA_PKT_COPY_STRUCT_HEADER_sub_op_shift',
    'SDMA_PKT_COPY_STRUCT_HEADER_tmz_mask',
    'SDMA_PKT_COPY_STRUCT_HEADER_tmz_offset',
    'SDMA_PKT_COPY_STRUCT_HEADER_tmz_shift',
    'SDMA_PKT_COPY_STRUCT_LINEAR_ADDR_HI_linear_addr_63_32_mask',
    'SDMA_PKT_COPY_STRUCT_LINEAR_ADDR_HI_linear_addr_63_32_offset',
    'SDMA_PKT_COPY_STRUCT_LINEAR_ADDR_HI_linear_addr_63_32_shift',
    'SDMA_PKT_COPY_STRUCT_LINEAR_ADDR_LO_linear_addr_31_0_mask',
    'SDMA_PKT_COPY_STRUCT_LINEAR_ADDR_LO_linear_addr_31_0_offset',
    'SDMA_PKT_COPY_STRUCT_LINEAR_ADDR_LO_linear_addr_31_0_shift',
    'SDMA_PKT_COPY_STRUCT_SB_ADDR_HI_sb_addr_63_32_mask',
    'SDMA_PKT_COPY_STRUCT_SB_ADDR_HI_sb_addr_63_32_offset',
    'SDMA_PKT_COPY_STRUCT_SB_ADDR_HI_sb_addr_63_32_shift',
    'SDMA_PKT_COPY_STRUCT_SB_ADDR_LO_sb_addr_31_0_mask',
    'SDMA_PKT_COPY_STRUCT_SB_ADDR_LO_sb_addr_31_0_offset',
    'SDMA_PKT_COPY_STRUCT_SB_ADDR_LO_sb_addr_31_0_shift',
    'SDMA_PKT_COPY_STRUCT_START_INDEX_start_index_mask',
    'SDMA_PKT_COPY_STRUCT_START_INDEX_start_index_offset',
    'SDMA_PKT_COPY_STRUCT_START_INDEX_start_index_shift',
    'SDMA_PKT_COPY_T2T_BC_DST_ADDR_HI_dst_addr_63_32_mask',
    'SDMA_PKT_COPY_T2T_BC_DST_ADDR_HI_dst_addr_63_32_offset',
    'SDMA_PKT_COPY_T2T_BC_DST_ADDR_HI_dst_addr_63_32_shift',
    'SDMA_PKT_COPY_T2T_BC_DST_ADDR_LO_dst_addr_31_0_mask',
    'SDMA_PKT_COPY_T2T_BC_DST_ADDR_LO_dst_addr_31_0_offset',
    'SDMA_PKT_COPY_T2T_BC_DST_ADDR_LO_dst_addr_31_0_shift',
    'SDMA_PKT_COPY_T2T_BC_DW_10_dst_width_mask',
    'SDMA_PKT_COPY_T2T_BC_DW_10_dst_width_offset',
    'SDMA_PKT_COPY_T2T_BC_DW_10_dst_width_shift',
    'SDMA_PKT_COPY_T2T_BC_DW_10_dst_z_mask',
    'SDMA_PKT_COPY_T2T_BC_DW_10_dst_z_offset',
    'SDMA_PKT_COPY_T2T_BC_DW_10_dst_z_shift',
    'SDMA_PKT_COPY_T2T_BC_DW_11_dst_depth_mask',
    'SDMA_PKT_COPY_T2T_BC_DW_11_dst_depth_offset',
    'SDMA_PKT_COPY_T2T_BC_DW_11_dst_depth_shift',
    'SDMA_PKT_COPY_T2T_BC_DW_11_dst_height_mask',
    'SDMA_PKT_COPY_T2T_BC_DW_11_dst_height_offset',
    'SDMA_PKT_COPY_T2T_BC_DW_11_dst_height_shift',
    'SDMA_PKT_COPY_T2T_BC_DW_12_dst_array_mode_mask',
    'SDMA_PKT_COPY_T2T_BC_DW_12_dst_array_mode_offset',
    'SDMA_PKT_COPY_T2T_BC_DW_12_dst_array_mode_shift',
    'SDMA_PKT_COPY_T2T_BC_DW_12_dst_bank_h_mask',
    'SDMA_PKT_COPY_T2T_BC_DW_12_dst_bank_h_offset',
    'SDMA_PKT_COPY_T2T_BC_DW_12_dst_bank_h_shift',
    'SDMA_PKT_COPY_T2T_BC_DW_12_dst_bank_w_mask',
    'SDMA_PKT_COPY_T2T_BC_DW_12_dst_bank_w_offset',
    'SDMA_PKT_COPY_T2T_BC_DW_12_dst_bank_w_shift',
    'SDMA_PKT_COPY_T2T_BC_DW_12_dst_element_size_mask',
    'SDMA_PKT_COPY_T2T_BC_DW_12_dst_element_size_offset',
    'SDMA_PKT_COPY_T2T_BC_DW_12_dst_element_size_shift',
    'SDMA_PKT_COPY_T2T_BC_DW_12_dst_mat_aspt_mask',
    'SDMA_PKT_COPY_T2T_BC_DW_12_dst_mat_aspt_offset',
    'SDMA_PKT_COPY_T2T_BC_DW_12_dst_mat_aspt_shift',
    'SDMA_PKT_COPY_T2T_BC_DW_12_dst_mit_mode_mask',
    'SDMA_PKT_COPY_T2T_BC_DW_12_dst_mit_mode_offset',
    'SDMA_PKT_COPY_T2T_BC_DW_12_dst_mit_mode_shift',
    'SDMA_PKT_COPY_T2T_BC_DW_12_dst_num_bank_mask',
    'SDMA_PKT_COPY_T2T_BC_DW_12_dst_num_bank_offset',
    'SDMA_PKT_COPY_T2T_BC_DW_12_dst_num_bank_shift',
    'SDMA_PKT_COPY_T2T_BC_DW_12_dst_pipe_config_mask',
    'SDMA_PKT_COPY_T2T_BC_DW_12_dst_pipe_config_offset',
    'SDMA_PKT_COPY_T2T_BC_DW_12_dst_pipe_config_shift',
    'SDMA_PKT_COPY_T2T_BC_DW_12_dst_tilesplit_size_mask',
    'SDMA_PKT_COPY_T2T_BC_DW_12_dst_tilesplit_size_offset',
    'SDMA_PKT_COPY_T2T_BC_DW_12_dst_tilesplit_size_shift',
    'SDMA_PKT_COPY_T2T_BC_DW_13_rect_x_mask',
    'SDMA_PKT_COPY_T2T_BC_DW_13_rect_x_offset',
    'SDMA_PKT_COPY_T2T_BC_DW_13_rect_x_shift',
    'SDMA_PKT_COPY_T2T_BC_DW_13_rect_y_mask',
    'SDMA_PKT_COPY_T2T_BC_DW_13_rect_y_offset',
    'SDMA_PKT_COPY_T2T_BC_DW_13_rect_y_shift',
    'SDMA_PKT_COPY_T2T_BC_DW_14_dst_sw_mask',
    'SDMA_PKT_COPY_T2T_BC_DW_14_dst_sw_offset',
    'SDMA_PKT_COPY_T2T_BC_DW_14_dst_sw_shift',
    'SDMA_PKT_COPY_T2T_BC_DW_14_rect_z_mask',
    'SDMA_PKT_COPY_T2T_BC_DW_14_rect_z_offset',
    'SDMA_PKT_COPY_T2T_BC_DW_14_rect_z_shift',
    'SDMA_PKT_COPY_T2T_BC_DW_14_src_sw_mask',
    'SDMA_PKT_COPY_T2T_BC_DW_14_src_sw_offset',
    'SDMA_PKT_COPY_T2T_BC_DW_14_src_sw_shift',
    'SDMA_PKT_COPY_T2T_BC_DW_3_src_x_mask',
    'SDMA_PKT_COPY_T2T_BC_DW_3_src_x_offset',
    'SDMA_PKT_COPY_T2T_BC_DW_3_src_x_shift',
    'SDMA_PKT_COPY_T2T_BC_DW_3_src_y_mask',
    'SDMA_PKT_COPY_T2T_BC_DW_3_src_y_offset',
    'SDMA_PKT_COPY_T2T_BC_DW_3_src_y_shift',
    'SDMA_PKT_COPY_T2T_BC_DW_4_src_width_mask',
    'SDMA_PKT_COPY_T2T_BC_DW_4_src_width_offset',
    'SDMA_PKT_COPY_T2T_BC_DW_4_src_width_shift',
    'SDMA_PKT_COPY_T2T_BC_DW_4_src_z_mask',
    'SDMA_PKT_COPY_T2T_BC_DW_4_src_z_offset',
    'SDMA_PKT_COPY_T2T_BC_DW_4_src_z_shift',
    'SDMA_PKT_COPY_T2T_BC_DW_5_src_depth_mask',
    'SDMA_PKT_COPY_T2T_BC_DW_5_src_depth_offset',
    'SDMA_PKT_COPY_T2T_BC_DW_5_src_depth_shift',
    'SDMA_PKT_COPY_T2T_BC_DW_5_src_height_mask',
    'SDMA_PKT_COPY_T2T_BC_DW_5_src_height_offset',
    'SDMA_PKT_COPY_T2T_BC_DW_5_src_height_shift',
    'SDMA_PKT_COPY_T2T_BC_DW_6_src_array_mode_mask',
    'SDMA_PKT_COPY_T2T_BC_DW_6_src_array_mode_offset',
    'SDMA_PKT_COPY_T2T_BC_DW_6_src_array_mode_shift',
    'SDMA_PKT_COPY_T2T_BC_DW_6_src_bank_h_mask',
    'SDMA_PKT_COPY_T2T_BC_DW_6_src_bank_h_offset',
    'SDMA_PKT_COPY_T2T_BC_DW_6_src_bank_h_shift',
    'SDMA_PKT_COPY_T2T_BC_DW_6_src_bank_w_mask',
    'SDMA_PKT_COPY_T2T_BC_DW_6_src_bank_w_offset',
    'SDMA_PKT_COPY_T2T_BC_DW_6_src_bank_w_shift',
    'SDMA_PKT_COPY_T2T_BC_DW_6_src_element_size_mask',
    'SDMA_PKT_COPY_T2T_BC_DW_6_src_element_size_offset',
    'SDMA_PKT_COPY_T2T_BC_DW_6_src_element_size_shift',
    'SDMA_PKT_COPY_T2T_BC_DW_6_src_mat_aspt_mask',
    'SDMA_PKT_COPY_T2T_BC_DW_6_src_mat_aspt_offset',
    'SDMA_PKT_COPY_T2T_BC_DW_6_src_mat_aspt_shift',
    'SDMA_PKT_COPY_T2T_BC_DW_6_src_mit_mode_mask',
    'SDMA_PKT_COPY_T2T_BC_DW_6_src_mit_mode_offset',
    'SDMA_PKT_COPY_T2T_BC_DW_6_src_mit_mode_shift',
    'SDMA_PKT_COPY_T2T_BC_DW_6_src_num_bank_mask',
    'SDMA_PKT_COPY_T2T_BC_DW_6_src_num_bank_offset',
    'SDMA_PKT_COPY_T2T_BC_DW_6_src_num_bank_shift',
    'SDMA_PKT_COPY_T2T_BC_DW_6_src_pipe_config_mask',
    'SDMA_PKT_COPY_T2T_BC_DW_6_src_pipe_config_offset',
    'SDMA_PKT_COPY_T2T_BC_DW_6_src_pipe_config_shift',
    'SDMA_PKT_COPY_T2T_BC_DW_6_src_tilesplit_size_mask',
    'SDMA_PKT_COPY_T2T_BC_DW_6_src_tilesplit_size_offset',
    'SDMA_PKT_COPY_T2T_BC_DW_6_src_tilesplit_size_shift',
    'SDMA_PKT_COPY_T2T_BC_DW_9_dst_x_mask',
    'SDMA_PKT_COPY_T2T_BC_DW_9_dst_x_offset',
    'SDMA_PKT_COPY_T2T_BC_DW_9_dst_x_shift',
    'SDMA_PKT_COPY_T2T_BC_DW_9_dst_y_mask',
    'SDMA_PKT_COPY_T2T_BC_DW_9_dst_y_offset',
    'SDMA_PKT_COPY_T2T_BC_DW_9_dst_y_shift',
    'SDMA_PKT_COPY_T2T_BC_HEADER_op_mask',
    'SDMA_PKT_COPY_T2T_BC_HEADER_op_offset',
    'SDMA_PKT_COPY_T2T_BC_HEADER_op_shift',
    'SDMA_PKT_COPY_T2T_BC_HEADER_sub_op_mask',
    'SDMA_PKT_COPY_T2T_BC_HEADER_sub_op_offset',
    'SDMA_PKT_COPY_T2T_BC_HEADER_sub_op_shift',
    'SDMA_PKT_COPY_T2T_BC_SRC_ADDR_HI_src_addr_63_32_mask',
    'SDMA_PKT_COPY_T2T_BC_SRC_ADDR_HI_src_addr_63_32_offset',
    'SDMA_PKT_COPY_T2T_BC_SRC_ADDR_HI_src_addr_63_32_shift',
    'SDMA_PKT_COPY_T2T_BC_SRC_ADDR_LO_src_addr_31_0_mask',
    'SDMA_PKT_COPY_T2T_BC_SRC_ADDR_LO_src_addr_31_0_offset',
    'SDMA_PKT_COPY_T2T_BC_SRC_ADDR_LO_src_addr_31_0_shift',
    'SDMA_PKT_COPY_T2T_DST_ADDR_HI_dst_addr_63_32_mask',
    'SDMA_PKT_COPY_T2T_DST_ADDR_HI_dst_addr_63_32_offset',
    'SDMA_PKT_COPY_T2T_DST_ADDR_HI_dst_addr_63_32_shift',
    'SDMA_PKT_COPY_T2T_DST_ADDR_LO_dst_addr_31_0_mask',
    'SDMA_PKT_COPY_T2T_DST_ADDR_LO_dst_addr_31_0_offset',
    'SDMA_PKT_COPY_T2T_DST_ADDR_LO_dst_addr_31_0_shift',
    'SDMA_PKT_COPY_T2T_DW_10_dst_width_mask',
    'SDMA_PKT_COPY_T2T_DW_10_dst_width_offset',
    'SDMA_PKT_COPY_T2T_DW_10_dst_width_shift',
    'SDMA_PKT_COPY_T2T_DW_10_dst_z_mask',
    'SDMA_PKT_COPY_T2T_DW_10_dst_z_offset',
    'SDMA_PKT_COPY_T2T_DW_10_dst_z_shift',
    'SDMA_PKT_COPY_T2T_DW_11_dst_depth_mask',
    'SDMA_PKT_COPY_T2T_DW_11_dst_depth_offset',
    'SDMA_PKT_COPY_T2T_DW_11_dst_depth_shift',
    'SDMA_PKT_COPY_T2T_DW_11_dst_height_mask',
    'SDMA_PKT_COPY_T2T_DW_11_dst_height_offset',
    'SDMA_PKT_COPY_T2T_DW_11_dst_height_shift',
    'SDMA_PKT_COPY_T2T_DW_12_dst_dimension_mask',
    'SDMA_PKT_COPY_T2T_DW_12_dst_dimension_offset',
    'SDMA_PKT_COPY_T2T_DW_12_dst_dimension_shift',
    'SDMA_PKT_COPY_T2T_DW_12_dst_element_size_mask',
    'SDMA_PKT_COPY_T2T_DW_12_dst_element_size_offset',
    'SDMA_PKT_COPY_T2T_DW_12_dst_element_size_shift',
    'SDMA_PKT_COPY_T2T_DW_12_dst_mip_id_mask',
    'SDMA_PKT_COPY_T2T_DW_12_dst_mip_id_offset',
    'SDMA_PKT_COPY_T2T_DW_12_dst_mip_id_shift',
    'SDMA_PKT_COPY_T2T_DW_12_dst_mip_max_mask',
    'SDMA_PKT_COPY_T2T_DW_12_dst_mip_max_offset',
    'SDMA_PKT_COPY_T2T_DW_12_dst_mip_max_shift',
    'SDMA_PKT_COPY_T2T_DW_12_dst_swizzle_mode_mask',
    'SDMA_PKT_COPY_T2T_DW_12_dst_swizzle_mode_offset',
    'SDMA_PKT_COPY_T2T_DW_12_dst_swizzle_mode_shift',
    'SDMA_PKT_COPY_T2T_DW_13_rect_x_mask',
    'SDMA_PKT_COPY_T2T_DW_13_rect_x_offset',
    'SDMA_PKT_COPY_T2T_DW_13_rect_x_shift',
    'SDMA_PKT_COPY_T2T_DW_13_rect_y_mask',
    'SDMA_PKT_COPY_T2T_DW_13_rect_y_offset',
    'SDMA_PKT_COPY_T2T_DW_13_rect_y_shift',
    'SDMA_PKT_COPY_T2T_DW_14_dst_sw_mask',
    'SDMA_PKT_COPY_T2T_DW_14_dst_sw_offset',
    'SDMA_PKT_COPY_T2T_DW_14_dst_sw_shift',
    'SDMA_PKT_COPY_T2T_DW_14_rect_z_mask',
    'SDMA_PKT_COPY_T2T_DW_14_rect_z_offset',
    'SDMA_PKT_COPY_T2T_DW_14_rect_z_shift',
    'SDMA_PKT_COPY_T2T_DW_14_src_sw_mask',
    'SDMA_PKT_COPY_T2T_DW_14_src_sw_offset',
    'SDMA_PKT_COPY_T2T_DW_14_src_sw_shift',
    'SDMA_PKT_COPY_T2T_DW_3_src_x_mask',
    'SDMA_PKT_COPY_T2T_DW_3_src_x_offset',
    'SDMA_PKT_COPY_T2T_DW_3_src_x_shift',
    'SDMA_PKT_COPY_T2T_DW_3_src_y_mask',
    'SDMA_PKT_COPY_T2T_DW_3_src_y_offset',
    'SDMA_PKT_COPY_T2T_DW_3_src_y_shift',
    'SDMA_PKT_COPY_T2T_DW_4_src_width_mask',
    'SDMA_PKT_COPY_T2T_DW_4_src_width_offset',
    'SDMA_PKT_COPY_T2T_DW_4_src_width_shift',
    'SDMA_PKT_COPY_T2T_DW_4_src_z_mask',
    'SDMA_PKT_COPY_T2T_DW_4_src_z_offset',
    'SDMA_PKT_COPY_T2T_DW_4_src_z_shift',
    'SDMA_PKT_COPY_T2T_DW_5_src_depth_mask',
    'SDMA_PKT_COPY_T2T_DW_5_src_depth_offset',
    'SDMA_PKT_COPY_T2T_DW_5_src_depth_shift',
    'SDMA_PKT_COPY_T2T_DW_5_src_height_mask',
    'SDMA_PKT_COPY_T2T_DW_5_src_height_offset',
    'SDMA_PKT_COPY_T2T_DW_5_src_height_shift',
    'SDMA_PKT_COPY_T2T_DW_6_src_dimension_mask',
    'SDMA_PKT_COPY_T2T_DW_6_src_dimension_offset',
    'SDMA_PKT_COPY_T2T_DW_6_src_dimension_shift',
    'SDMA_PKT_COPY_T2T_DW_6_src_element_size_mask',
    'SDMA_PKT_COPY_T2T_DW_6_src_element_size_offset',
    'SDMA_PKT_COPY_T2T_DW_6_src_element_size_shift',
    'SDMA_PKT_COPY_T2T_DW_6_src_mip_id_mask',
    'SDMA_PKT_COPY_T2T_DW_6_src_mip_id_offset',
    'SDMA_PKT_COPY_T2T_DW_6_src_mip_id_shift',
    'SDMA_PKT_COPY_T2T_DW_6_src_mip_max_mask',
    'SDMA_PKT_COPY_T2T_DW_6_src_mip_max_offset',
    'SDMA_PKT_COPY_T2T_DW_6_src_mip_max_shift',
    'SDMA_PKT_COPY_T2T_DW_6_src_swizzle_mode_mask',
    'SDMA_PKT_COPY_T2T_DW_6_src_swizzle_mode_offset',
    'SDMA_PKT_COPY_T2T_DW_6_src_swizzle_mode_shift',
    'SDMA_PKT_COPY_T2T_DW_9_dst_x_mask',
    'SDMA_PKT_COPY_T2T_DW_9_dst_x_offset',
    'SDMA_PKT_COPY_T2T_DW_9_dst_x_shift',
    'SDMA_PKT_COPY_T2T_DW_9_dst_y_mask',
    'SDMA_PKT_COPY_T2T_DW_9_dst_y_offset',
    'SDMA_PKT_COPY_T2T_DW_9_dst_y_shift',
    'SDMA_PKT_COPY_T2T_HEADER_dcc_dir_mask',
    'SDMA_PKT_COPY_T2T_HEADER_dcc_dir_offset',
    'SDMA_PKT_COPY_T2T_HEADER_dcc_dir_shift',
    'SDMA_PKT_COPY_T2T_HEADER_dcc_mask',
    'SDMA_PKT_COPY_T2T_HEADER_dcc_offset',
    'SDMA_PKT_COPY_T2T_HEADER_dcc_shift',
    'SDMA_PKT_COPY_T2T_HEADER_op_mask',
    'SDMA_PKT_COPY_T2T_HEADER_op_offset',
    'SDMA_PKT_COPY_T2T_HEADER_op_shift',
    'SDMA_PKT_COPY_T2T_HEADER_sub_op_mask',
    'SDMA_PKT_COPY_T2T_HEADER_sub_op_offset',
    'SDMA_PKT_COPY_T2T_HEADER_sub_op_shift',
    'SDMA_PKT_COPY_T2T_HEADER_tmz_mask',
    'SDMA_PKT_COPY_T2T_HEADER_tmz_offset',
    'SDMA_PKT_COPY_T2T_HEADER_tmz_shift',
    'SDMA_PKT_COPY_T2T_META_ADDR_HI_meta_addr_63_32_mask',
    'SDMA_PKT_COPY_T2T_META_ADDR_HI_meta_addr_63_32_offset',
    'SDMA_PKT_COPY_T2T_META_ADDR_HI_meta_addr_63_32_shift',
    'SDMA_PKT_COPY_T2T_META_ADDR_LO_meta_addr_31_0_mask',
    'SDMA_PKT_COPY_T2T_META_ADDR_LO_meta_addr_31_0_offset',
    'SDMA_PKT_COPY_T2T_META_ADDR_LO_meta_addr_31_0_shift',
    'SDMA_PKT_COPY_T2T_META_CONFIG_alpha_is_on_msb_mask',
    'SDMA_PKT_COPY_T2T_META_CONFIG_alpha_is_on_msb_offset',
    'SDMA_PKT_COPY_T2T_META_CONFIG_alpha_is_on_msb_shift',
    'SDMA_PKT_COPY_T2T_META_CONFIG_color_transform_disable_mask',
    'SDMA_PKT_COPY_T2T_META_CONFIG_color_transform_disable_offset',
    'SDMA_PKT_COPY_T2T_META_CONFIG_color_transform_disable_shift',
    'SDMA_PKT_COPY_T2T_META_CONFIG_data_format_mask',
    'SDMA_PKT_COPY_T2T_META_CONFIG_data_format_offset',
    'SDMA_PKT_COPY_T2T_META_CONFIG_data_format_shift',
    'SDMA_PKT_COPY_T2T_META_CONFIG_max_comp_block_size_mask',
    'SDMA_PKT_COPY_T2T_META_CONFIG_max_comp_block_size_offset',
    'SDMA_PKT_COPY_T2T_META_CONFIG_max_comp_block_size_shift',
    'SDMA_PKT_COPY_T2T_META_CONFIG_max_uncomp_block_size_mask',
    'SDMA_PKT_COPY_T2T_META_CONFIG_max_uncomp_block_size_offset',
    'SDMA_PKT_COPY_T2T_META_CONFIG_max_uncomp_block_size_shift',
    'SDMA_PKT_COPY_T2T_META_CONFIG_meta_tmz_mask',
    'SDMA_PKT_COPY_T2T_META_CONFIG_meta_tmz_offset',
    'SDMA_PKT_COPY_T2T_META_CONFIG_meta_tmz_shift',
    'SDMA_PKT_COPY_T2T_META_CONFIG_number_type_mask',
    'SDMA_PKT_COPY_T2T_META_CONFIG_number_type_offset',
    'SDMA_PKT_COPY_T2T_META_CONFIG_number_type_shift',
    'SDMA_PKT_COPY_T2T_META_CONFIG_surface_type_mask',
    'SDMA_PKT_COPY_T2T_META_CONFIG_surface_type_offset',
    'SDMA_PKT_COPY_T2T_META_CONFIG_surface_type_shift',
    'SDMA_PKT_COPY_T2T_META_CONFIG_write_compress_enable_mask',
    'SDMA_PKT_COPY_T2T_META_CONFIG_write_compress_enable_offset',
    'SDMA_PKT_COPY_T2T_META_CONFIG_write_compress_enable_shift',
    'SDMA_PKT_COPY_T2T_SRC_ADDR_HI_src_addr_63_32_mask',
    'SDMA_PKT_COPY_T2T_SRC_ADDR_HI_src_addr_63_32_offset',
    'SDMA_PKT_COPY_T2T_SRC_ADDR_HI_src_addr_63_32_shift',
    'SDMA_PKT_COPY_T2T_SRC_ADDR_LO_src_addr_31_0_mask',
    'SDMA_PKT_COPY_T2T_SRC_ADDR_LO_src_addr_31_0_offset',
    'SDMA_PKT_COPY_T2T_SRC_ADDR_LO_src_addr_31_0_shift',
    'SDMA_PKT_COPY_TILED_BC_COUNT_count_mask',
    'SDMA_PKT_COPY_TILED_BC_COUNT_count_offset',
    'SDMA_PKT_COPY_TILED_BC_COUNT_count_shift',
    'SDMA_PKT_COPY_TILED_BC_DW_3_width_mask',
    'SDMA_PKT_COPY_TILED_BC_DW_3_width_offset',
    'SDMA_PKT_COPY_TILED_BC_DW_3_width_shift',
    'SDMA_PKT_COPY_TILED_BC_DW_4_depth_mask',
    'SDMA_PKT_COPY_TILED_BC_DW_4_depth_offset',
    'SDMA_PKT_COPY_TILED_BC_DW_4_depth_shift',
    'SDMA_PKT_COPY_TILED_BC_DW_4_height_mask',
    'SDMA_PKT_COPY_TILED_BC_DW_4_height_offset',
    'SDMA_PKT_COPY_TILED_BC_DW_4_height_shift',
    'SDMA_PKT_COPY_TILED_BC_DW_5_array_mode_mask',
    'SDMA_PKT_COPY_TILED_BC_DW_5_array_mode_offset',
    'SDMA_PKT_COPY_TILED_BC_DW_5_array_mode_shift',
    'SDMA_PKT_COPY_TILED_BC_DW_5_bank_h_mask',
    'SDMA_PKT_COPY_TILED_BC_DW_5_bank_h_offset',
    'SDMA_PKT_COPY_TILED_BC_DW_5_bank_h_shift',
    'SDMA_PKT_COPY_TILED_BC_DW_5_bank_w_mask',
    'SDMA_PKT_COPY_TILED_BC_DW_5_bank_w_offset',
    'SDMA_PKT_COPY_TILED_BC_DW_5_bank_w_shift',
    'SDMA_PKT_COPY_TILED_BC_DW_5_element_size_mask',
    'SDMA_PKT_COPY_TILED_BC_DW_5_element_size_offset',
    'SDMA_PKT_COPY_TILED_BC_DW_5_element_size_shift',
    'SDMA_PKT_COPY_TILED_BC_DW_5_mat_aspt_mask',
    'SDMA_PKT_COPY_TILED_BC_DW_5_mat_aspt_offset',
    'SDMA_PKT_COPY_TILED_BC_DW_5_mat_aspt_shift',
    'SDMA_PKT_COPY_TILED_BC_DW_5_mit_mode_mask',
    'SDMA_PKT_COPY_TILED_BC_DW_5_mit_mode_offset',
    'SDMA_PKT_COPY_TILED_BC_DW_5_mit_mode_shift',
    'SDMA_PKT_COPY_TILED_BC_DW_5_num_bank_mask',
    'SDMA_PKT_COPY_TILED_BC_DW_5_num_bank_offset',
    'SDMA_PKT_COPY_TILED_BC_DW_5_num_bank_shift',
    'SDMA_PKT_COPY_TILED_BC_DW_5_pipe_config_mask',
    'SDMA_PKT_COPY_TILED_BC_DW_5_pipe_config_offset',
    'SDMA_PKT_COPY_TILED_BC_DW_5_pipe_config_shift',
    'SDMA_PKT_COPY_TILED_BC_DW_5_tilesplit_size_mask',
    'SDMA_PKT_COPY_TILED_BC_DW_5_tilesplit_size_offset',
    'SDMA_PKT_COPY_TILED_BC_DW_5_tilesplit_size_shift',
    'SDMA_PKT_COPY_TILED_BC_DW_6_x_mask',
    'SDMA_PKT_COPY_TILED_BC_DW_6_x_offset',
    'SDMA_PKT_COPY_TILED_BC_DW_6_x_shift',
    'SDMA_PKT_COPY_TILED_BC_DW_6_y_mask',
    'SDMA_PKT_COPY_TILED_BC_DW_6_y_offset',
    'SDMA_PKT_COPY_TILED_BC_DW_6_y_shift',
    'SDMA_PKT_COPY_TILED_BC_DW_7_linear_sw_mask',
    'SDMA_PKT_COPY_TILED_BC_DW_7_linear_sw_offset',
    'SDMA_PKT_COPY_TILED_BC_DW_7_linear_sw_shift',
    'SDMA_PKT_COPY_TILED_BC_DW_7_tile_sw_mask',
    'SDMA_PKT_COPY_TILED_BC_DW_7_tile_sw_offset',
    'SDMA_PKT_COPY_TILED_BC_DW_7_tile_sw_shift',
    'SDMA_PKT_COPY_TILED_BC_DW_7_z_mask',
    'SDMA_PKT_COPY_TILED_BC_DW_7_z_offset',
    'SDMA_PKT_COPY_TILED_BC_DW_7_z_shift',
    'SDMA_PKT_COPY_TILED_BC_HEADER_detile_mask',
    'SDMA_PKT_COPY_TILED_BC_HEADER_detile_offset',
    'SDMA_PKT_COPY_TILED_BC_HEADER_detile_shift',
    'SDMA_PKT_COPY_TILED_BC_HEADER_op_mask',
    'SDMA_PKT_COPY_TILED_BC_HEADER_op_offset',
    'SDMA_PKT_COPY_TILED_BC_HEADER_op_shift',
    'SDMA_PKT_COPY_TILED_BC_HEADER_sub_op_mask',
    'SDMA_PKT_COPY_TILED_BC_HEADER_sub_op_offset',
    'SDMA_PKT_COPY_TILED_BC_HEADER_sub_op_shift',
    'SDMA_PKT_COPY_TILED_BC_LINEAR_ADDR_HI_linear_addr_63_32_mask',
    'SDMA_PKT_COPY_TILED_BC_LINEAR_ADDR_HI_linear_addr_63_32_offset',
    'SDMA_PKT_COPY_TILED_BC_LINEAR_ADDR_HI_linear_addr_63_32_shift',
    'SDMA_PKT_COPY_TILED_BC_LINEAR_ADDR_LO_linear_addr_31_0_mask',
    'SDMA_PKT_COPY_TILED_BC_LINEAR_ADDR_LO_linear_addr_31_0_offset',
    'SDMA_PKT_COPY_TILED_BC_LINEAR_ADDR_LO_linear_addr_31_0_shift',
    'SDMA_PKT_COPY_TILED_BC_LINEAR_PITCH_linear_pitch_mask',
    'SDMA_PKT_COPY_TILED_BC_LINEAR_PITCH_linear_pitch_offset',
    'SDMA_PKT_COPY_TILED_BC_LINEAR_PITCH_linear_pitch_shift',
    'SDMA_PKT_COPY_TILED_BC_TILED_ADDR_HI_tiled_addr_63_32_mask',
    'SDMA_PKT_COPY_TILED_BC_TILED_ADDR_HI_tiled_addr_63_32_offset',
    'SDMA_PKT_COPY_TILED_BC_TILED_ADDR_HI_tiled_addr_63_32_shift',
    'SDMA_PKT_COPY_TILED_BC_TILED_ADDR_LO_tiled_addr_31_0_mask',
    'SDMA_PKT_COPY_TILED_BC_TILED_ADDR_LO_tiled_addr_31_0_offset',
    'SDMA_PKT_COPY_TILED_BC_TILED_ADDR_LO_tiled_addr_31_0_shift',
    'SDMA_PKT_COPY_TILED_COUNT_count_mask',
    'SDMA_PKT_COPY_TILED_COUNT_count_offset',
    'SDMA_PKT_COPY_TILED_COUNT_count_shift',
    'SDMA_PKT_COPY_TILED_DW_3_width_mask',
    'SDMA_PKT_COPY_TILED_DW_3_width_offset',
    'SDMA_PKT_COPY_TILED_DW_3_width_shift',
    'SDMA_PKT_COPY_TILED_DW_4_depth_mask',
    'SDMA_PKT_COPY_TILED_DW_4_depth_offset',
    'SDMA_PKT_COPY_TILED_DW_4_depth_shift',
    'SDMA_PKT_COPY_TILED_DW_4_height_mask',
    'SDMA_PKT_COPY_TILED_DW_4_height_offset',
    'SDMA_PKT_COPY_TILED_DW_4_height_shift',
    'SDMA_PKT_COPY_TILED_DW_5_dimension_mask',
    'SDMA_PKT_COPY_TILED_DW_5_dimension_offset',
    'SDMA_PKT_COPY_TILED_DW_5_dimension_shift',
    'SDMA_PKT_COPY_TILED_DW_5_element_size_mask',
    'SDMA_PKT_COPY_TILED_DW_5_element_size_offset',
    'SDMA_PKT_COPY_TILED_DW_5_element_size_shift',
    'SDMA_PKT_COPY_TILED_DW_5_mip_max_mask',
    'SDMA_PKT_COPY_TILED_DW_5_mip_max_offset',
    'SDMA_PKT_COPY_TILED_DW_5_mip_max_shift',
    'SDMA_PKT_COPY_TILED_DW_5_swizzle_mode_mask',
    'SDMA_PKT_COPY_TILED_DW_5_swizzle_mode_offset',
    'SDMA_PKT_COPY_TILED_DW_5_swizzle_mode_shift',
    'SDMA_PKT_COPY_TILED_DW_6_x_mask',
    'SDMA_PKT_COPY_TILED_DW_6_x_offset',
    'SDMA_PKT_COPY_TILED_DW_6_x_shift',
    'SDMA_PKT_COPY_TILED_DW_6_y_mask',
    'SDMA_PKT_COPY_TILED_DW_6_y_offset',
    'SDMA_PKT_COPY_TILED_DW_6_y_shift',
    'SDMA_PKT_COPY_TILED_DW_7_linear_cc_mask',
    'SDMA_PKT_COPY_TILED_DW_7_linear_cc_offset',
    'SDMA_PKT_COPY_TILED_DW_7_linear_cc_shift',
    'SDMA_PKT_COPY_TILED_DW_7_linear_sw_mask',
    'SDMA_PKT_COPY_TILED_DW_7_linear_sw_offset',
    'SDMA_PKT_COPY_TILED_DW_7_linear_sw_shift',
    'SDMA_PKT_COPY_TILED_DW_7_tile_sw_mask',
    'SDMA_PKT_COPY_TILED_DW_7_tile_sw_offset',
    'SDMA_PKT_COPY_TILED_DW_7_tile_sw_shift',
    'SDMA_PKT_COPY_TILED_DW_7_z_mask',
    'SDMA_PKT_COPY_TILED_DW_7_z_offset',
    'SDMA_PKT_COPY_TILED_DW_7_z_shift',
    'SDMA_PKT_COPY_TILED_HEADER_detile_mask',
    'SDMA_PKT_COPY_TILED_HEADER_detile_offset',
    'SDMA_PKT_COPY_TILED_HEADER_detile_shift',
    'SDMA_PKT_COPY_TILED_HEADER_encrypt_mask',
    'SDMA_PKT_COPY_TILED_HEADER_encrypt_offset',
    'SDMA_PKT_COPY_TILED_HEADER_encrypt_shift',
    'SDMA_PKT_COPY_TILED_HEADER_op_mask',
    'SDMA_PKT_COPY_TILED_HEADER_op_offset',
    'SDMA_PKT_COPY_TILED_HEADER_op_shift',
    'SDMA_PKT_COPY_TILED_HEADER_sub_op_mask',
    'SDMA_PKT_COPY_TILED_HEADER_sub_op_offset',
    'SDMA_PKT_COPY_TILED_HEADER_sub_op_shift',
    'SDMA_PKT_COPY_TILED_HEADER_tmz_mask',
    'SDMA_PKT_COPY_TILED_HEADER_tmz_offset',
    'SDMA_PKT_COPY_TILED_HEADER_tmz_shift',
    'SDMA_PKT_COPY_TILED_LINEAR_ADDR_HI_linear_addr_63_32_mask',
    'SDMA_PKT_COPY_TILED_LINEAR_ADDR_HI_linear_addr_63_32_offset',
    'SDMA_PKT_COPY_TILED_LINEAR_ADDR_HI_linear_addr_63_32_shift',
    'SDMA_PKT_COPY_TILED_LINEAR_ADDR_LO_linear_addr_31_0_mask',
    'SDMA_PKT_COPY_TILED_LINEAR_ADDR_LO_linear_addr_31_0_offset',
    'SDMA_PKT_COPY_TILED_LINEAR_ADDR_LO_linear_addr_31_0_shift',
    'SDMA_PKT_COPY_TILED_LINEAR_PITCH_linear_pitch_mask',
    'SDMA_PKT_COPY_TILED_LINEAR_PITCH_linear_pitch_offset',
    'SDMA_PKT_COPY_TILED_LINEAR_PITCH_linear_pitch_shift',
    'SDMA_PKT_COPY_TILED_LINEAR_SLICE_PITCH_linear_slice_pitch_mask',
    'SDMA_PKT_COPY_TILED_LINEAR_SLICE_PITCH_linear_slice_pitch_offset',
    'SDMA_PKT_COPY_TILED_LINEAR_SLICE_PITCH_linear_slice_pitch_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_10_linear_pitch_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_10_linear_pitch_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_10_linear_pitch_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_10_linear_z_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_10_linear_z_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_10_linear_z_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_11_linear_slice_pitch_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_11_linear_slice_pitch_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_11_linear_slice_pitch_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_12_rect_x_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_12_rect_x_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_12_rect_x_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_12_rect_y_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_12_rect_y_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_12_rect_y_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_13_linear_sw_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_13_linear_sw_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_13_linear_sw_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_13_rect_z_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_13_rect_z_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_13_rect_z_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_13_tile_sw_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_13_tile_sw_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_13_tile_sw_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_3_tiled_x_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_3_tiled_x_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_3_tiled_x_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_3_tiled_y_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_3_tiled_y_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_3_tiled_y_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_4_tiled_z_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_4_tiled_z_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_4_tiled_z_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_4_width_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_4_width_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_4_width_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_5_depth_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_5_depth_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_5_depth_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_5_height_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_5_height_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_5_height_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_array_mode_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_array_mode_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_array_mode_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_bank_h_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_bank_h_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_bank_h_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_bank_w_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_bank_w_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_bank_w_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_element_size_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_element_size_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_element_size_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_mat_aspt_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_mat_aspt_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_mat_aspt_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_mit_mode_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_mit_mode_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_mit_mode_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_num_bank_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_num_bank_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_num_bank_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_pipe_config_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_pipe_config_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_pipe_config_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_tilesplit_size_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_tilesplit_size_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_6_tilesplit_size_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_9_linear_x_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_9_linear_x_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_9_linear_x_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_9_linear_y_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_9_linear_y_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_DW_9_linear_y_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_HEADER_detile_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_HEADER_detile_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_HEADER_detile_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_HEADER_op_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_HEADER_op_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_HEADER_op_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_HEADER_sub_op_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_HEADER_sub_op_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_HEADER_sub_op_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_LINEAR_ADDR_HI_linear_addr_63_32_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_LINEAR_ADDR_HI_linear_addr_63_32_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_LINEAR_ADDR_HI_linear_addr_63_32_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_LINEAR_ADDR_LO_linear_addr_31_0_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_LINEAR_ADDR_LO_linear_addr_31_0_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_LINEAR_ADDR_LO_linear_addr_31_0_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_TILED_ADDR_HI_tiled_addr_63_32_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_TILED_ADDR_HI_tiled_addr_63_32_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_TILED_ADDR_HI_tiled_addr_63_32_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_TILED_ADDR_LO_tiled_addr_31_0_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_TILED_ADDR_LO_tiled_addr_31_0_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_BC_TILED_ADDR_LO_tiled_addr_31_0_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_DW_10_linear_pitch_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_DW_10_linear_pitch_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_DW_10_linear_pitch_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_DW_10_linear_z_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_DW_10_linear_z_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_DW_10_linear_z_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_DW_11_linear_slice_pitch_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_DW_11_linear_slice_pitch_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_DW_11_linear_slice_pitch_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_DW_12_rect_x_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_DW_12_rect_x_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_DW_12_rect_x_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_DW_12_rect_y_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_DW_12_rect_y_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_DW_12_rect_y_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_DW_13_linear_sw_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_DW_13_linear_sw_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_DW_13_linear_sw_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_DW_13_rect_z_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_DW_13_rect_z_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_DW_13_rect_z_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_DW_13_tile_sw_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_DW_13_tile_sw_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_DW_13_tile_sw_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_DW_3_tiled_x_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_DW_3_tiled_x_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_DW_3_tiled_x_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_DW_3_tiled_y_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_DW_3_tiled_y_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_DW_3_tiled_y_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_DW_4_tiled_z_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_DW_4_tiled_z_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_DW_4_tiled_z_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_DW_4_width_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_DW_4_width_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_DW_4_width_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_DW_5_depth_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_DW_5_depth_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_DW_5_depth_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_DW_5_height_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_DW_5_height_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_DW_5_height_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_DW_6_dimension_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_DW_6_dimension_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_DW_6_dimension_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_DW_6_element_size_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_DW_6_element_size_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_DW_6_element_size_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_DW_6_mip_id_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_DW_6_mip_id_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_DW_6_mip_id_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_DW_6_mip_max_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_DW_6_mip_max_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_DW_6_mip_max_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_DW_6_swizzle_mode_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_DW_6_swizzle_mode_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_DW_6_swizzle_mode_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_DW_9_linear_x_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_DW_9_linear_x_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_DW_9_linear_x_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_DW_9_linear_y_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_DW_9_linear_y_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_DW_9_linear_y_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_HEADER_dcc_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_HEADER_dcc_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_HEADER_dcc_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_HEADER_detile_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_HEADER_detile_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_HEADER_detile_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_HEADER_op_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_HEADER_op_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_HEADER_op_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_HEADER_sub_op_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_HEADER_sub_op_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_HEADER_sub_op_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_HEADER_tmz_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_HEADER_tmz_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_HEADER_tmz_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_LINEAR_ADDR_HI_linear_addr_63_32_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_LINEAR_ADDR_HI_linear_addr_63_32_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_LINEAR_ADDR_HI_linear_addr_63_32_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_LINEAR_ADDR_LO_linear_addr_31_0_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_LINEAR_ADDR_LO_linear_addr_31_0_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_LINEAR_ADDR_LO_linear_addr_31_0_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_META_ADDR_HI_meta_addr_63_32_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_META_ADDR_HI_meta_addr_63_32_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_META_ADDR_HI_meta_addr_63_32_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_META_ADDR_LO_meta_addr_31_0_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_META_ADDR_LO_meta_addr_31_0_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_META_ADDR_LO_meta_addr_31_0_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_alpha_is_on_msb_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_alpha_is_on_msb_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_alpha_is_on_msb_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_color_transform_disable_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_color_transform_disable_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_color_transform_disable_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_data_format_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_data_format_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_data_format_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_max_comp_block_size_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_max_comp_block_size_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_max_comp_block_size_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_max_uncomp_block_size_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_max_uncomp_block_size_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_max_uncomp_block_size_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_meta_tmz_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_meta_tmz_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_meta_tmz_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_number_type_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_number_type_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_number_type_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_surface_type_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_surface_type_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_surface_type_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_write_compress_enable_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_write_compress_enable_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_META_CONFIG_write_compress_enable_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_TILED_ADDR_HI_tiled_addr_63_32_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_TILED_ADDR_HI_tiled_addr_63_32_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_TILED_ADDR_HI_tiled_addr_63_32_shift',
    'SDMA_PKT_COPY_TILED_SUBWIN_TILED_ADDR_LO_tiled_addr_31_0_mask',
    'SDMA_PKT_COPY_TILED_SUBWIN_TILED_ADDR_LO_tiled_addr_31_0_offset',
    'SDMA_PKT_COPY_TILED_SUBWIN_TILED_ADDR_LO_tiled_addr_31_0_shift',
    'SDMA_PKT_COPY_TILED_TILED_ADDR_HI_tiled_addr_63_32_mask',
    'SDMA_PKT_COPY_TILED_TILED_ADDR_HI_tiled_addr_63_32_offset',
    'SDMA_PKT_COPY_TILED_TILED_ADDR_HI_tiled_addr_63_32_shift',
    'SDMA_PKT_COPY_TILED_TILED_ADDR_LO_tiled_addr_31_0_mask',
    'SDMA_PKT_COPY_TILED_TILED_ADDR_LO_tiled_addr_31_0_offset',
    'SDMA_PKT_COPY_TILED_TILED_ADDR_LO_tiled_addr_31_0_shift',
    'SDMA_PKT_DATA_FILL_MULTI_BYTE_COUNT_count_mask',
    'SDMA_PKT_DATA_FILL_MULTI_BYTE_COUNT_count_offset',
    'SDMA_PKT_DATA_FILL_MULTI_BYTE_COUNT_count_shift',
    'SDMA_PKT_DATA_FILL_MULTI_BYTE_STRIDE_byte_stride_mask',
    'SDMA_PKT_DATA_FILL_MULTI_BYTE_STRIDE_byte_stride_offset',
    'SDMA_PKT_DATA_FILL_MULTI_BYTE_STRIDE_byte_stride_shift',
    'SDMA_PKT_DATA_FILL_MULTI_DMA_COUNT_dma_count_mask',
    'SDMA_PKT_DATA_FILL_MULTI_DMA_COUNT_dma_count_offset',
    'SDMA_PKT_DATA_FILL_MULTI_DMA_COUNT_dma_count_shift',
    'SDMA_PKT_DATA_FILL_MULTI_DST_ADDR_HI_dst_addr_63_32_mask',
    'SDMA_PKT_DATA_FILL_MULTI_DST_ADDR_HI_dst_addr_63_32_offset',
    'SDMA_PKT_DATA_FILL_MULTI_DST_ADDR_HI_dst_addr_63_32_shift',
    'SDMA_PKT_DATA_FILL_MULTI_DST_ADDR_LO_dst_addr_31_0_mask',
    'SDMA_PKT_DATA_FILL_MULTI_DST_ADDR_LO_dst_addr_31_0_offset',
    'SDMA_PKT_DATA_FILL_MULTI_DST_ADDR_LO_dst_addr_31_0_shift',
    'SDMA_PKT_DATA_FILL_MULTI_HEADER_memlog_clr_mask',
    'SDMA_PKT_DATA_FILL_MULTI_HEADER_memlog_clr_offset',
    'SDMA_PKT_DATA_FILL_MULTI_HEADER_memlog_clr_shift',
    'SDMA_PKT_DATA_FILL_MULTI_HEADER_op_mask',
    'SDMA_PKT_DATA_FILL_MULTI_HEADER_op_offset',
    'SDMA_PKT_DATA_FILL_MULTI_HEADER_op_shift',
    'SDMA_PKT_DATA_FILL_MULTI_HEADER_sub_op_mask',
    'SDMA_PKT_DATA_FILL_MULTI_HEADER_sub_op_offset',
    'SDMA_PKT_DATA_FILL_MULTI_HEADER_sub_op_shift',
    'SDMA_PKT_DUMMY_TRAP_HEADER_op_mask',
    'SDMA_PKT_DUMMY_TRAP_HEADER_op_offset',
    'SDMA_PKT_DUMMY_TRAP_HEADER_op_shift',
    'SDMA_PKT_DUMMY_TRAP_HEADER_sub_op_mask',
    'SDMA_PKT_DUMMY_TRAP_HEADER_sub_op_offset',
    'SDMA_PKT_DUMMY_TRAP_HEADER_sub_op_shift',
    'SDMA_PKT_DUMMY_TRAP_INT_CONTEXT_int_context_mask',
    'SDMA_PKT_DUMMY_TRAP_INT_CONTEXT_int_context_offset',
    'SDMA_PKT_DUMMY_TRAP_INT_CONTEXT_int_context_shift',
    'SDMA_PKT_FENCE', 'SDMA_PKT_FENCE_ADDR_HI_addr_63_32_mask',
    'SDMA_PKT_FENCE_ADDR_HI_addr_63_32_offset',
    'SDMA_PKT_FENCE_ADDR_HI_addr_63_32_shift',
    'SDMA_PKT_FENCE_ADDR_LO_addr_31_0_mask',
    'SDMA_PKT_FENCE_ADDR_LO_addr_31_0_offset',
    'SDMA_PKT_FENCE_ADDR_LO_addr_31_0_shift',
    'SDMA_PKT_FENCE_DATA_data_mask',
    'SDMA_PKT_FENCE_DATA_data_offset',
    'SDMA_PKT_FENCE_DATA_data_shift',
    'SDMA_PKT_FENCE_HEADER_gcc_mask',
    'SDMA_PKT_FENCE_HEADER_gcc_offset',
    'SDMA_PKT_FENCE_HEADER_gcc_shift',
    'SDMA_PKT_FENCE_HEADER_gpa_mask',
    'SDMA_PKT_FENCE_HEADER_gpa_offset',
    'SDMA_PKT_FENCE_HEADER_gpa_shift',
    'SDMA_PKT_FENCE_HEADER_l2_policy_mask',
    'SDMA_PKT_FENCE_HEADER_l2_policy_offset',
    'SDMA_PKT_FENCE_HEADER_l2_policy_shift',
    'SDMA_PKT_FENCE_HEADER_mtype_mask',
    'SDMA_PKT_FENCE_HEADER_mtype_offset',
    'SDMA_PKT_FENCE_HEADER_mtype_shift',
    'SDMA_PKT_FENCE_HEADER_op_mask',
    'SDMA_PKT_FENCE_HEADER_op_offset',
    'SDMA_PKT_FENCE_HEADER_op_shift',
    'SDMA_PKT_FENCE_HEADER_snp_mask',
    'SDMA_PKT_FENCE_HEADER_snp_offset',
    'SDMA_PKT_FENCE_HEADER_snp_shift',
    'SDMA_PKT_FENCE_HEADER_sub_op_mask',
    'SDMA_PKT_FENCE_HEADER_sub_op_offset',
    'SDMA_PKT_FENCE_HEADER_sub_op_shift',
    'SDMA_PKT_FENCE_HEADER_sys_mask',
    'SDMA_PKT_FENCE_HEADER_sys_offset',
    'SDMA_PKT_FENCE_HEADER_sys_shift', 'SDMA_PKT_GCR',
    'SDMA_PKT_GCR_REQ_HEADER_op_mask',
    'SDMA_PKT_GCR_REQ_HEADER_op_offset',
    'SDMA_PKT_GCR_REQ_HEADER_op_shift',
    'SDMA_PKT_GCR_REQ_HEADER_sub_op_mask',
    'SDMA_PKT_GCR_REQ_HEADER_sub_op_offset',
    'SDMA_PKT_GCR_REQ_HEADER_sub_op_shift',
    'SDMA_PKT_GCR_REQ_PAYLOAD1_base_va_31_7_mask',
    'SDMA_PKT_GCR_REQ_PAYLOAD1_base_va_31_7_offset',
    'SDMA_PKT_GCR_REQ_PAYLOAD1_base_va_31_7_shift',
    'SDMA_PKT_GCR_REQ_PAYLOAD2_base_va_47_32_mask',
    'SDMA_PKT_GCR_REQ_PAYLOAD2_base_va_47_32_offset',
    'SDMA_PKT_GCR_REQ_PAYLOAD2_base_va_47_32_shift',
    'SDMA_PKT_GCR_REQ_PAYLOAD2_gcr_control_15_0_mask',
    'SDMA_PKT_GCR_REQ_PAYLOAD2_gcr_control_15_0_offset',
    'SDMA_PKT_GCR_REQ_PAYLOAD2_gcr_control_15_0_shift',
    'SDMA_PKT_GCR_REQ_PAYLOAD3_gcr_control_18_16_mask',
    'SDMA_PKT_GCR_REQ_PAYLOAD3_gcr_control_18_16_offset',
    'SDMA_PKT_GCR_REQ_PAYLOAD3_gcr_control_18_16_shift',
    'SDMA_PKT_GCR_REQ_PAYLOAD3_limit_va_31_7_mask',
    'SDMA_PKT_GCR_REQ_PAYLOAD3_limit_va_31_7_offset',
    'SDMA_PKT_GCR_REQ_PAYLOAD3_limit_va_31_7_shift',
    'SDMA_PKT_GCR_REQ_PAYLOAD4_limit_va_47_32_mask',
    'SDMA_PKT_GCR_REQ_PAYLOAD4_limit_va_47_32_offset',
    'SDMA_PKT_GCR_REQ_PAYLOAD4_limit_va_47_32_shift',
    'SDMA_PKT_GCR_REQ_PAYLOAD4_vmid_mask',
    'SDMA_PKT_GCR_REQ_PAYLOAD4_vmid_offset',
    'SDMA_PKT_GCR_REQ_PAYLOAD4_vmid_shift',
    'SDMA_PKT_GPUVM_INV_HEADER_op_mask',
    'SDMA_PKT_GPUVM_INV_HEADER_op_offset',
    'SDMA_PKT_GPUVM_INV_HEADER_op_shift',
    'SDMA_PKT_GPUVM_INV_HEADER_sub_op_mask',
    'SDMA_PKT_GPUVM_INV_HEADER_sub_op_offset',
    'SDMA_PKT_GPUVM_INV_HEADER_sub_op_shift',
    'SDMA_PKT_GPUVM_INV_PAYLOAD1_clr_protection_fault_status_addr_mask',
    'SDMA_PKT_GPUVM_INV_PAYLOAD1_clr_protection_fault_status_addr_offset',
    'SDMA_PKT_GPUVM_INV_PAYLOAD1_clr_protection_fault_status_addr_shift',
    'SDMA_PKT_GPUVM_INV_PAYLOAD1_flush_type_mask',
    'SDMA_PKT_GPUVM_INV_PAYLOAD1_flush_type_offset',
    'SDMA_PKT_GPUVM_INV_PAYLOAD1_flush_type_shift',
    'SDMA_PKT_GPUVM_INV_PAYLOAD1_four_kilobytes_mask',
    'SDMA_PKT_GPUVM_INV_PAYLOAD1_four_kilobytes_offset',
    'SDMA_PKT_GPUVM_INV_PAYLOAD1_four_kilobytes_shift',
    'SDMA_PKT_GPUVM_INV_PAYLOAD1_l1_ptes_mask',
    'SDMA_PKT_GPUVM_INV_PAYLOAD1_l1_ptes_offset',
    'SDMA_PKT_GPUVM_INV_PAYLOAD1_l1_ptes_shift',
    'SDMA_PKT_GPUVM_INV_PAYLOAD1_l2_pde0_mask',
    'SDMA_PKT_GPUVM_INV_PAYLOAD1_l2_pde0_offset',
    'SDMA_PKT_GPUVM_INV_PAYLOAD1_l2_pde0_shift',
    'SDMA_PKT_GPUVM_INV_PAYLOAD1_l2_pde1_mask',
    'SDMA_PKT_GPUVM_INV_PAYLOAD1_l2_pde1_offset',
    'SDMA_PKT_GPUVM_INV_PAYLOAD1_l2_pde1_shift',
    'SDMA_PKT_GPUVM_INV_PAYLOAD1_l2_pde2_mask',
    'SDMA_PKT_GPUVM_INV_PAYLOAD1_l2_pde2_offset',
    'SDMA_PKT_GPUVM_INV_PAYLOAD1_l2_pde2_shift',
    'SDMA_PKT_GPUVM_INV_PAYLOAD1_l2_ptes_mask',
    'SDMA_PKT_GPUVM_INV_PAYLOAD1_l2_ptes_offset',
    'SDMA_PKT_GPUVM_INV_PAYLOAD1_l2_ptes_shift',
    'SDMA_PKT_GPUVM_INV_PAYLOAD1_log_request_mask',
    'SDMA_PKT_GPUVM_INV_PAYLOAD1_log_request_offset',
    'SDMA_PKT_GPUVM_INV_PAYLOAD1_log_request_shift',
    'SDMA_PKT_GPUVM_INV_PAYLOAD1_per_vmid_inv_req_mask',
    'SDMA_PKT_GPUVM_INV_PAYLOAD1_per_vmid_inv_req_offset',
    'SDMA_PKT_GPUVM_INV_PAYLOAD1_per_vmid_inv_req_shift',
    'SDMA_PKT_GPUVM_INV_PAYLOAD2_page_va_42_12_mask',
    'SDMA_PKT_GPUVM_INV_PAYLOAD2_page_va_42_12_offset',
    'SDMA_PKT_GPUVM_INV_PAYLOAD2_page_va_42_12_shift',
    'SDMA_PKT_GPUVM_INV_PAYLOAD2_s_mask',
    'SDMA_PKT_GPUVM_INV_PAYLOAD2_s_offset',
    'SDMA_PKT_GPUVM_INV_PAYLOAD2_s_shift',
    'SDMA_PKT_GPUVM_INV_PAYLOAD3_page_va_47_43_mask',
    'SDMA_PKT_GPUVM_INV_PAYLOAD3_page_va_47_43_offset',
    'SDMA_PKT_GPUVM_INV_PAYLOAD3_page_va_47_43_shift',
    'SDMA_PKT_HDP_FLUSH', 'SDMA_PKT_HEADER_op_mask',
    'SDMA_PKT_HEADER_op_offset', 'SDMA_PKT_HEADER_op_shift',
    'SDMA_PKT_HEADER_sub_op_mask', 'SDMA_PKT_HEADER_sub_op_offset',
    'SDMA_PKT_HEADER_sub_op_shift',
    'SDMA_PKT_INDIRECT_BASE_HI_ib_base_63_32_mask',
    'SDMA_PKT_INDIRECT_BASE_HI_ib_base_63_32_offset',
    'SDMA_PKT_INDIRECT_BASE_HI_ib_base_63_32_shift',
    'SDMA_PKT_INDIRECT_BASE_LO_ib_base_31_0_mask',
    'SDMA_PKT_INDIRECT_BASE_LO_ib_base_31_0_offset',
    'SDMA_PKT_INDIRECT_BASE_LO_ib_base_31_0_shift',
    'SDMA_PKT_INDIRECT_CSA_ADDR_HI_csa_addr_63_32_mask',
    'SDMA_PKT_INDIRECT_CSA_ADDR_HI_csa_addr_63_32_offset',
    'SDMA_PKT_INDIRECT_CSA_ADDR_HI_csa_addr_63_32_shift',
    'SDMA_PKT_INDIRECT_CSA_ADDR_LO_csa_addr_31_0_mask',
    'SDMA_PKT_INDIRECT_CSA_ADDR_LO_csa_addr_31_0_offset',
    'SDMA_PKT_INDIRECT_CSA_ADDR_LO_csa_addr_31_0_shift',
    'SDMA_PKT_INDIRECT_HEADER_op_mask',
    'SDMA_PKT_INDIRECT_HEADER_op_offset',
    'SDMA_PKT_INDIRECT_HEADER_op_shift',
    'SDMA_PKT_INDIRECT_HEADER_priv_mask',
    'SDMA_PKT_INDIRECT_HEADER_priv_offset',
    'SDMA_PKT_INDIRECT_HEADER_priv_shift',
    'SDMA_PKT_INDIRECT_HEADER_sub_op_mask',
    'SDMA_PKT_INDIRECT_HEADER_sub_op_offset',
    'SDMA_PKT_INDIRECT_HEADER_sub_op_shift',
    'SDMA_PKT_INDIRECT_HEADER_vmid_mask',
    'SDMA_PKT_INDIRECT_HEADER_vmid_offset',
    'SDMA_PKT_INDIRECT_HEADER_vmid_shift',
    'SDMA_PKT_INDIRECT_IB_SIZE_ib_size_mask',
    'SDMA_PKT_INDIRECT_IB_SIZE_ib_size_offset',
    'SDMA_PKT_INDIRECT_IB_SIZE_ib_size_shift',
    'SDMA_PKT_NOP_DATA0_data0_mask',
    'SDMA_PKT_NOP_DATA0_data0_offset',
    'SDMA_PKT_NOP_DATA0_data0_shift',
    'SDMA_PKT_NOP_HEADER_count_mask',
    'SDMA_PKT_NOP_HEADER_count_offset',
    'SDMA_PKT_NOP_HEADER_count_shift', 'SDMA_PKT_NOP_HEADER_op_mask',
    'SDMA_PKT_NOP_HEADER_op_offset', 'SDMA_PKT_NOP_HEADER_op_shift',
    'SDMA_PKT_NOP_HEADER_sub_op_mask',
    'SDMA_PKT_NOP_HEADER_sub_op_offset',
    'SDMA_PKT_NOP_HEADER_sub_op_shift',
    'SDMA_PKT_POLL_DBIT_WRITE_MEM_DST_ADDR_HI_addr_63_32_mask',
    'SDMA_PKT_POLL_DBIT_WRITE_MEM_DST_ADDR_HI_addr_63_32_offset',
    'SDMA_PKT_POLL_DBIT_WRITE_MEM_DST_ADDR_HI_addr_63_32_shift',
    'SDMA_PKT_POLL_DBIT_WRITE_MEM_DST_ADDR_LO_addr_31_0_mask',
    'SDMA_PKT_POLL_DBIT_WRITE_MEM_DST_ADDR_LO_addr_31_0_offset',
    'SDMA_PKT_POLL_DBIT_WRITE_MEM_DST_ADDR_LO_addr_31_0_shift',
    'SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_ea_mask',
    'SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_ea_offset',
    'SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_ea_shift',
    'SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_op_mask',
    'SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_op_offset',
    'SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_op_shift',
    'SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_sub_op_mask',
    'SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_sub_op_offset',
    'SDMA_PKT_POLL_DBIT_WRITE_MEM_HEADER_sub_op_shift',
    'SDMA_PKT_POLL_DBIT_WRITE_MEM_PAGE_NUM_page_num_31_0_mask',
    'SDMA_PKT_POLL_DBIT_WRITE_MEM_PAGE_NUM_page_num_31_0_offset',
    'SDMA_PKT_POLL_DBIT_WRITE_MEM_PAGE_NUM_page_num_31_0_shift',
    'SDMA_PKT_POLL_DBIT_WRITE_MEM_START_PAGE_addr_31_4_mask',
    'SDMA_PKT_POLL_DBIT_WRITE_MEM_START_PAGE_addr_31_4_offset',
    'SDMA_PKT_POLL_DBIT_WRITE_MEM_START_PAGE_addr_31_4_shift',
    'SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_END_HI_cmp1_end_63_32_mask',
    'SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_END_HI_cmp1_end_63_32_offset',
    'SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_END_HI_cmp1_end_63_32_shift',
    'SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_END_LO_cmp1_end_31_0_mask',
    'SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_END_LO_cmp1_end_31_0_offset',
    'SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_END_LO_cmp1_end_31_0_shift',
    'SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_START_HI_cmp0_start_63_32_mask',
    'SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_START_HI_cmp0_start_63_32_offset',
    'SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_START_HI_cmp0_start_63_32_shift',
    'SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_START_LO_cmp0_start_31_0_mask',
    'SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_START_LO_cmp0_start_31_0_offset',
    'SDMA_PKT_POLL_MEM_VERIFY_CMP0_ADDR_START_LO_cmp0_start_31_0_shift',
    'SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_END_HI_cmp1_end_63_32_mask',
    'SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_END_HI_cmp1_end_63_32_offset',
    'SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_END_HI_cmp1_end_63_32_shift',
    'SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_END_LO_cmp1_end_31_0_mask',
    'SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_END_LO_cmp1_end_31_0_offset',
    'SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_END_LO_cmp1_end_31_0_shift',
    'SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_START_HI_cmp1_start_63_32_mask',
    'SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_START_HI_cmp1_start_63_32_offset',
    'SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_START_HI_cmp1_start_63_32_shift',
    'SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_START_LO_cmp1_start_31_0_mask',
    'SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_START_LO_cmp1_start_31_0_offset',
    'SDMA_PKT_POLL_MEM_VERIFY_CMP1_ADDR_START_LO_cmp1_start_31_0_shift',
    'SDMA_PKT_POLL_MEM_VERIFY_HEADER_mode_mask',
    'SDMA_PKT_POLL_MEM_VERIFY_HEADER_mode_offset',
    'SDMA_PKT_POLL_MEM_VERIFY_HEADER_mode_shift',
    'SDMA_PKT_POLL_MEM_VERIFY_HEADER_op_mask',
    'SDMA_PKT_POLL_MEM_VERIFY_HEADER_op_offset',
    'SDMA_PKT_POLL_MEM_VERIFY_HEADER_op_shift',
    'SDMA_PKT_POLL_MEM_VERIFY_HEADER_sub_op_mask',
    'SDMA_PKT_POLL_MEM_VERIFY_HEADER_sub_op_offset',
    'SDMA_PKT_POLL_MEM_VERIFY_HEADER_sub_op_shift',
    'SDMA_PKT_POLL_MEM_VERIFY_PATTERN_pattern_mask',
    'SDMA_PKT_POLL_MEM_VERIFY_PATTERN_pattern_offset',
    'SDMA_PKT_POLL_MEM_VERIFY_PATTERN_pattern_shift',
    'SDMA_PKT_POLL_MEM_VERIFY_REC_ADDR_HI_rec_63_32_mask',
    'SDMA_PKT_POLL_MEM_VERIFY_REC_ADDR_HI_rec_63_32_offset',
    'SDMA_PKT_POLL_MEM_VERIFY_REC_ADDR_HI_rec_63_32_shift',
    'SDMA_PKT_POLL_MEM_VERIFY_REC_ADDR_LO_rec_31_0_mask',
    'SDMA_PKT_POLL_MEM_VERIFY_REC_ADDR_LO_rec_31_0_offset',
    'SDMA_PKT_POLL_MEM_VERIFY_REC_ADDR_LO_rec_31_0_shift',
    'SDMA_PKT_POLL_MEM_VERIFY_RESERVED_reserved_mask',
    'SDMA_PKT_POLL_MEM_VERIFY_RESERVED_reserved_offset',
    'SDMA_PKT_POLL_MEM_VERIFY_RESERVED_reserved_shift',
    'SDMA_PKT_POLL_REGMEM',
    'SDMA_PKT_POLL_REGMEM_ADDR_HI_addr_63_32_mask',
    'SDMA_PKT_POLL_REGMEM_ADDR_HI_addr_63_32_offset',
    'SDMA_PKT_POLL_REGMEM_ADDR_HI_addr_63_32_shift',
    'SDMA_PKT_POLL_REGMEM_ADDR_LO_addr_31_0_mask',
    'SDMA_PKT_POLL_REGMEM_ADDR_LO_addr_31_0_offset',
    'SDMA_PKT_POLL_REGMEM_ADDR_LO_addr_31_0_shift',
    'SDMA_PKT_POLL_REGMEM_DW5_interval_mask',
    'SDMA_PKT_POLL_REGMEM_DW5_interval_offset',
    'SDMA_PKT_POLL_REGMEM_DW5_interval_shift',
    'SDMA_PKT_POLL_REGMEM_DW5_retry_count_mask',
    'SDMA_PKT_POLL_REGMEM_DW5_retry_count_offset',
    'SDMA_PKT_POLL_REGMEM_DW5_retry_count_shift',
    'SDMA_PKT_POLL_REGMEM_HEADER_func_mask',
    'SDMA_PKT_POLL_REGMEM_HEADER_func_offset',
    'SDMA_PKT_POLL_REGMEM_HEADER_func_shift',
    'SDMA_PKT_POLL_REGMEM_HEADER_hdp_flush_mask',
    'SDMA_PKT_POLL_REGMEM_HEADER_hdp_flush_offset',
    'SDMA_PKT_POLL_REGMEM_HEADER_hdp_flush_shift',
    'SDMA_PKT_POLL_REGMEM_HEADER_mem_poll_mask',
    'SDMA_PKT_POLL_REGMEM_HEADER_mem_poll_offset',
    'SDMA_PKT_POLL_REGMEM_HEADER_mem_poll_shift',
    'SDMA_PKT_POLL_REGMEM_HEADER_op_mask',
    'SDMA_PKT_POLL_REGMEM_HEADER_op_offset',
    'SDMA_PKT_POLL_REGMEM_HEADER_op_shift',
    'SDMA_PKT_POLL_REGMEM_HEADER_sub_op_mask',
    'SDMA_PKT_POLL_REGMEM_HEADER_sub_op_offset',
    'SDMA_PKT_POLL_REGMEM_HEADER_sub_op_shift',
    'SDMA_PKT_POLL_REGMEM_MASK_mask_mask',
    'SDMA_PKT_POLL_REGMEM_MASK_mask_offset',
    'SDMA_PKT_POLL_REGMEM_MASK_mask_shift',
    'SDMA_PKT_POLL_REGMEM_VALUE_value_mask',
    'SDMA_PKT_POLL_REGMEM_VALUE_value_offset',
    'SDMA_PKT_POLL_REGMEM_VALUE_value_shift',
    'SDMA_PKT_POLL_REG_WRITE_MEM_DST_ADDR_HI_addr_63_32_mask',
    'SDMA_PKT_POLL_REG_WRITE_MEM_DST_ADDR_HI_addr_63_32_offset',
    'SDMA_PKT_POLL_REG_WRITE_MEM_DST_ADDR_HI_addr_63_32_shift',
    'SDMA_PKT_POLL_REG_WRITE_MEM_DST_ADDR_LO_addr_31_0_mask',
    'SDMA_PKT_POLL_REG_WRITE_MEM_DST_ADDR_LO_addr_31_0_offset',
    'SDMA_PKT_POLL_REG_WRITE_MEM_DST_ADDR_LO_addr_31_0_shift',
    'SDMA_PKT_POLL_REG_WRITE_MEM_HEADER_op_mask',
    'SDMA_PKT_POLL_REG_WRITE_MEM_HEADER_op_offset',
    'SDMA_PKT_POLL_REG_WRITE_MEM_HEADER_op_shift',
    'SDMA_PKT_POLL_REG_WRITE_MEM_HEADER_sub_op_mask',
    'SDMA_PKT_POLL_REG_WRITE_MEM_HEADER_sub_op_offset',
    'SDMA_PKT_POLL_REG_WRITE_MEM_HEADER_sub_op_shift',
    'SDMA_PKT_POLL_REG_WRITE_MEM_SRC_ADDR_addr_31_2_mask',
    'SDMA_PKT_POLL_REG_WRITE_MEM_SRC_ADDR_addr_31_2_offset',
    'SDMA_PKT_POLL_REG_WRITE_MEM_SRC_ADDR_addr_31_2_shift',
    'SDMA_PKT_PRE_EXE_EXEC_COUNT_exec_count_mask',
    'SDMA_PKT_PRE_EXE_EXEC_COUNT_exec_count_offset',
    'SDMA_PKT_PRE_EXE_EXEC_COUNT_exec_count_shift',
    'SDMA_PKT_PRE_EXE_HEADER_dev_sel_mask',
    'SDMA_PKT_PRE_EXE_HEADER_dev_sel_offset',
    'SDMA_PKT_PRE_EXE_HEADER_dev_sel_shift',
    'SDMA_PKT_PRE_EXE_HEADER_op_mask',
    'SDMA_PKT_PRE_EXE_HEADER_op_offset',
    'SDMA_PKT_PRE_EXE_HEADER_op_shift',
    'SDMA_PKT_PRE_EXE_HEADER_sub_op_mask',
    'SDMA_PKT_PRE_EXE_HEADER_sub_op_offset',
    'SDMA_PKT_PRE_EXE_HEADER_sub_op_shift',
    'SDMA_PKT_PTEPDE_COPY_BACKWARDS_COUNT_IN_32B_XFER_count_mask',
    'SDMA_PKT_PTEPDE_COPY_BACKWARDS_COUNT_IN_32B_XFER_count_offset',
    'SDMA_PKT_PTEPDE_COPY_BACKWARDS_COUNT_IN_32B_XFER_count_shift',
    'SDMA_PKT_PTEPDE_COPY_BACKWARDS_DST_ADDR_HI_dst_addr_63_32_mask',
    'SDMA_PKT_PTEPDE_COPY_BACKWARDS_DST_ADDR_HI_dst_addr_63_32_offset',
    'SDMA_PKT_PTEPDE_COPY_BACKWARDS_DST_ADDR_HI_dst_addr_63_32_shift',
    'SDMA_PKT_PTEPDE_COPY_BACKWARDS_DST_ADDR_LO_dst_addr_31_0_mask',
    'SDMA_PKT_PTEPDE_COPY_BACKWARDS_DST_ADDR_LO_dst_addr_31_0_offset',
    'SDMA_PKT_PTEPDE_COPY_BACKWARDS_DST_ADDR_LO_dst_addr_31_0_shift',
    'SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_direction_mask',
    'SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_direction_offset',
    'SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_direction_shift',
    'SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_op_mask',
    'SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_op_offset',
    'SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_op_shift',
    'SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_pte_size_mask',
    'SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_pte_size_offset',
    'SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_pte_size_shift',
    'SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_ptepde_op_mask',
    'SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_ptepde_op_offset',
    'SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_ptepde_op_shift',
    'SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_sub_op_mask',
    'SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_sub_op_offset',
    'SDMA_PKT_PTEPDE_COPY_BACKWARDS_HEADER_sub_op_shift',
    'SDMA_PKT_PTEPDE_COPY_BACKWARDS_MASK_BIT_FOR_DW_mask_first_xfer_mask',
    'SDMA_PKT_PTEPDE_COPY_BACKWARDS_MASK_BIT_FOR_DW_mask_first_xfer_offset',
    'SDMA_PKT_PTEPDE_COPY_BACKWARDS_MASK_BIT_FOR_DW_mask_first_xfer_shift',
    'SDMA_PKT_PTEPDE_COPY_BACKWARDS_MASK_BIT_FOR_DW_mask_last_xfer_mask',
    'SDMA_PKT_PTEPDE_COPY_BACKWARDS_MASK_BIT_FOR_DW_mask_last_xfer_offset',
    'SDMA_PKT_PTEPDE_COPY_BACKWARDS_MASK_BIT_FOR_DW_mask_last_xfer_shift',
    'SDMA_PKT_PTEPDE_COPY_BACKWARDS_SRC_ADDR_HI_src_addr_63_32_mask',
    'SDMA_PKT_PTEPDE_COPY_BACKWARDS_SRC_ADDR_HI_src_addr_63_32_offset',
    'SDMA_PKT_PTEPDE_COPY_BACKWARDS_SRC_ADDR_HI_src_addr_63_32_shift',
    'SDMA_PKT_PTEPDE_COPY_BACKWARDS_SRC_ADDR_LO_src_addr_31_0_mask',
    'SDMA_PKT_PTEPDE_COPY_BACKWARDS_SRC_ADDR_LO_src_addr_31_0_offset',
    'SDMA_PKT_PTEPDE_COPY_BACKWARDS_SRC_ADDR_LO_src_addr_31_0_shift',
    'SDMA_PKT_PTEPDE_COPY_COUNT_count_mask',
    'SDMA_PKT_PTEPDE_COPY_COUNT_count_offset',
    'SDMA_PKT_PTEPDE_COPY_COUNT_count_shift',
    'SDMA_PKT_PTEPDE_COPY_DST_ADDR_HI_dst_addr_63_32_mask',
    'SDMA_PKT_PTEPDE_COPY_DST_ADDR_HI_dst_addr_63_32_offset',
    'SDMA_PKT_PTEPDE_COPY_DST_ADDR_HI_dst_addr_63_32_shift',
    'SDMA_PKT_PTEPDE_COPY_DST_ADDR_LO_dst_addr_31_0_mask',
    'SDMA_PKT_PTEPDE_COPY_DST_ADDR_LO_dst_addr_31_0_offset',
    'SDMA_PKT_PTEPDE_COPY_DST_ADDR_LO_dst_addr_31_0_shift',
    'SDMA_PKT_PTEPDE_COPY_HEADER_op_mask',
    'SDMA_PKT_PTEPDE_COPY_HEADER_op_offset',
    'SDMA_PKT_PTEPDE_COPY_HEADER_op_shift',
    'SDMA_PKT_PTEPDE_COPY_HEADER_ptepde_op_mask',
    'SDMA_PKT_PTEPDE_COPY_HEADER_ptepde_op_offset',
    'SDMA_PKT_PTEPDE_COPY_HEADER_ptepde_op_shift',
    'SDMA_PKT_PTEPDE_COPY_HEADER_sub_op_mask',
    'SDMA_PKT_PTEPDE_COPY_HEADER_sub_op_offset',
    'SDMA_PKT_PTEPDE_COPY_HEADER_sub_op_shift',
    'SDMA_PKT_PTEPDE_COPY_HEADER_tmz_mask',
    'SDMA_PKT_PTEPDE_COPY_HEADER_tmz_offset',
    'SDMA_PKT_PTEPDE_COPY_HEADER_tmz_shift',
    'SDMA_PKT_PTEPDE_COPY_MASK_DW0_mask_dw0_mask',
    'SDMA_PKT_PTEPDE_COPY_MASK_DW0_mask_dw0_offset',
    'SDMA_PKT_PTEPDE_COPY_MASK_DW0_mask_dw0_shift',
    'SDMA_PKT_PTEPDE_COPY_MASK_DW1_mask_dw1_mask',
    'SDMA_PKT_PTEPDE_COPY_MASK_DW1_mask_dw1_offset',
    'SDMA_PKT_PTEPDE_COPY_MASK_DW1_mask_dw1_shift',
    'SDMA_PKT_PTEPDE_COPY_SRC_ADDR_HI_src_addr_63_32_mask',
    'SDMA_PKT_PTEPDE_COPY_SRC_ADDR_HI_src_addr_63_32_offset',
    'SDMA_PKT_PTEPDE_COPY_SRC_ADDR_HI_src_addr_63_32_shift',
    'SDMA_PKT_PTEPDE_COPY_SRC_ADDR_LO_src_addr_31_0_mask',
    'SDMA_PKT_PTEPDE_COPY_SRC_ADDR_LO_src_addr_31_0_offset',
    'SDMA_PKT_PTEPDE_COPY_SRC_ADDR_LO_src_addr_31_0_shift',
    'SDMA_PKT_PTEPDE_RMW_ADDR_HI_addr_63_32_mask',
    'SDMA_PKT_PTEPDE_RMW_ADDR_HI_addr_63_32_offset',
    'SDMA_PKT_PTEPDE_RMW_ADDR_HI_addr_63_32_shift',
    'SDMA_PKT_PTEPDE_RMW_ADDR_LO_addr_31_0_mask',
    'SDMA_PKT_PTEPDE_RMW_ADDR_LO_addr_31_0_offset',
    'SDMA_PKT_PTEPDE_RMW_ADDR_LO_addr_31_0_shift',
    'SDMA_PKT_PTEPDE_RMW_HEADER_gcc_mask',
    'SDMA_PKT_PTEPDE_RMW_HEADER_gcc_offset',
    'SDMA_PKT_PTEPDE_RMW_HEADER_gcc_shift',
    'SDMA_PKT_PTEPDE_RMW_HEADER_gpa_mask',
    'SDMA_PKT_PTEPDE_RMW_HEADER_gpa_offset',
    'SDMA_PKT_PTEPDE_RMW_HEADER_gpa_shift',
    'SDMA_PKT_PTEPDE_RMW_HEADER_l2_policy_mask',
    'SDMA_PKT_PTEPDE_RMW_HEADER_l2_policy_offset',
    'SDMA_PKT_PTEPDE_RMW_HEADER_l2_policy_shift',
    'SDMA_PKT_PTEPDE_RMW_HEADER_mtype_mask',
    'SDMA_PKT_PTEPDE_RMW_HEADER_mtype_offset',
    'SDMA_PKT_PTEPDE_RMW_HEADER_mtype_shift',
    'SDMA_PKT_PTEPDE_RMW_HEADER_op_mask',
    'SDMA_PKT_PTEPDE_RMW_HEADER_op_offset',
    'SDMA_PKT_PTEPDE_RMW_HEADER_op_shift',
    'SDMA_PKT_PTEPDE_RMW_HEADER_snp_mask',
    'SDMA_PKT_PTEPDE_RMW_HEADER_snp_offset',
    'SDMA_PKT_PTEPDE_RMW_HEADER_snp_shift',
    'SDMA_PKT_PTEPDE_RMW_HEADER_sub_op_mask',
    'SDMA_PKT_PTEPDE_RMW_HEADER_sub_op_offset',
    'SDMA_PKT_PTEPDE_RMW_HEADER_sub_op_shift',
    'SDMA_PKT_PTEPDE_RMW_HEADER_sys_mask',
    'SDMA_PKT_PTEPDE_RMW_HEADER_sys_offset',
    'SDMA_PKT_PTEPDE_RMW_HEADER_sys_shift',
    'SDMA_PKT_PTEPDE_RMW_MASK_HI_mask_63_32_mask',
    'SDMA_PKT_PTEPDE_RMW_MASK_HI_mask_63_32_offset',
    'SDMA_PKT_PTEPDE_RMW_MASK_HI_mask_63_32_shift',
    'SDMA_PKT_PTEPDE_RMW_MASK_LO_mask_31_0_mask',
    'SDMA_PKT_PTEPDE_RMW_MASK_LO_mask_31_0_offset',
    'SDMA_PKT_PTEPDE_RMW_MASK_LO_mask_31_0_shift',
    'SDMA_PKT_PTEPDE_RMW_VALUE_HI_value_63_32_mask',
    'SDMA_PKT_PTEPDE_RMW_VALUE_HI_value_63_32_offset',
    'SDMA_PKT_PTEPDE_RMW_VALUE_HI_value_63_32_shift',
    'SDMA_PKT_PTEPDE_RMW_VALUE_LO_value_31_0_mask',
    'SDMA_PKT_PTEPDE_RMW_VALUE_LO_value_31_0_offset',
    'SDMA_PKT_PTEPDE_RMW_VALUE_LO_value_31_0_shift',
    'SDMA_PKT_SEMAPHORE_ADDR_HI_addr_63_32_mask',
    'SDMA_PKT_SEMAPHORE_ADDR_HI_addr_63_32_offset',
    'SDMA_PKT_SEMAPHORE_ADDR_HI_addr_63_32_shift',
    'SDMA_PKT_SEMAPHORE_ADDR_LO_addr_31_0_mask',
    'SDMA_PKT_SEMAPHORE_ADDR_LO_addr_31_0_offset',
    'SDMA_PKT_SEMAPHORE_ADDR_LO_addr_31_0_shift',
    'SDMA_PKT_SEMAPHORE_HEADER_mailbox_mask',
    'SDMA_PKT_SEMAPHORE_HEADER_mailbox_offset',
    'SDMA_PKT_SEMAPHORE_HEADER_mailbox_shift',
    'SDMA_PKT_SEMAPHORE_HEADER_op_mask',
    'SDMA_PKT_SEMAPHORE_HEADER_op_offset',
    'SDMA_PKT_SEMAPHORE_HEADER_op_shift',
    'SDMA_PKT_SEMAPHORE_HEADER_signal_mask',
    'SDMA_PKT_SEMAPHORE_HEADER_signal_offset',
    'SDMA_PKT_SEMAPHORE_HEADER_signal_shift',
    'SDMA_PKT_SEMAPHORE_HEADER_sub_op_mask',
    'SDMA_PKT_SEMAPHORE_HEADER_sub_op_offset',
    'SDMA_PKT_SEMAPHORE_HEADER_sub_op_shift',
    'SDMA_PKT_SEMAPHORE_HEADER_write_one_mask',
    'SDMA_PKT_SEMAPHORE_HEADER_write_one_offset',
    'SDMA_PKT_SEMAPHORE_HEADER_write_one_shift',
    'SDMA_PKT_SRBM_WRITE_ADDR_addr_mask',
    'SDMA_PKT_SRBM_WRITE_ADDR_addr_offset',
    'SDMA_PKT_SRBM_WRITE_ADDR_addr_shift',
    'SDMA_PKT_SRBM_WRITE_ADDR_apertureid_mask',
    'SDMA_PKT_SRBM_WRITE_ADDR_apertureid_offset',
    'SDMA_PKT_SRBM_WRITE_ADDR_apertureid_shift',
    'SDMA_PKT_SRBM_WRITE_DATA_data_mask',
    'SDMA_PKT_SRBM_WRITE_DATA_data_offset',
    'SDMA_PKT_SRBM_WRITE_DATA_data_shift',
    'SDMA_PKT_SRBM_WRITE_HEADER_byte_en_mask',
    'SDMA_PKT_SRBM_WRITE_HEADER_byte_en_offset',
    'SDMA_PKT_SRBM_WRITE_HEADER_byte_en_shift',
    'SDMA_PKT_SRBM_WRITE_HEADER_op_mask',
    'SDMA_PKT_SRBM_WRITE_HEADER_op_offset',
    'SDMA_PKT_SRBM_WRITE_HEADER_op_shift',
    'SDMA_PKT_SRBM_WRITE_HEADER_sub_op_mask',
    'SDMA_PKT_SRBM_WRITE_HEADER_sub_op_offset',
    'SDMA_PKT_SRBM_WRITE_HEADER_sub_op_shift', 'SDMA_PKT_TIMESTAMP',
    'SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_op_mask',
    'SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_op_offset',
    'SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_op_shift',
    'SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_sub_op_mask',
    'SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_sub_op_offset',
    'SDMA_PKT_TIMESTAMP_GET_GLOBAL_HEADER_sub_op_shift',
    'SDMA_PKT_TIMESTAMP_GET_GLOBAL_WRITE_ADDR_HI_write_addr_63_32_mask',
    'SDMA_PKT_TIMESTAMP_GET_GLOBAL_WRITE_ADDR_HI_write_addr_63_32_offset',
    'SDMA_PKT_TIMESTAMP_GET_GLOBAL_WRITE_ADDR_HI_write_addr_63_32_shift',
    'SDMA_PKT_TIMESTAMP_GET_GLOBAL_WRITE_ADDR_LO_write_addr_31_3_mask',
    'SDMA_PKT_TIMESTAMP_GET_GLOBAL_WRITE_ADDR_LO_write_addr_31_3_offset',
    'SDMA_PKT_TIMESTAMP_GET_GLOBAL_WRITE_ADDR_LO_write_addr_31_3_shift',
    'SDMA_PKT_TIMESTAMP_GET_HEADER_op_mask',
    'SDMA_PKT_TIMESTAMP_GET_HEADER_op_offset',
    'SDMA_PKT_TIMESTAMP_GET_HEADER_op_shift',
    'SDMA_PKT_TIMESTAMP_GET_HEADER_sub_op_mask',
    'SDMA_PKT_TIMESTAMP_GET_HEADER_sub_op_offset',
    'SDMA_PKT_TIMESTAMP_GET_HEADER_sub_op_shift',
    'SDMA_PKT_TIMESTAMP_GET_WRITE_ADDR_HI_write_addr_63_32_mask',
    'SDMA_PKT_TIMESTAMP_GET_WRITE_ADDR_HI_write_addr_63_32_offset',
    'SDMA_PKT_TIMESTAMP_GET_WRITE_ADDR_HI_write_addr_63_32_shift',
    'SDMA_PKT_TIMESTAMP_GET_WRITE_ADDR_LO_write_addr_31_3_mask',
    'SDMA_PKT_TIMESTAMP_GET_WRITE_ADDR_LO_write_addr_31_3_offset',
    'SDMA_PKT_TIMESTAMP_GET_WRITE_ADDR_LO_write_addr_31_3_shift',
    'SDMA_PKT_TIMESTAMP_SET_HEADER_op_mask',
    'SDMA_PKT_TIMESTAMP_SET_HEADER_op_offset',
    'SDMA_PKT_TIMESTAMP_SET_HEADER_op_shift',
    'SDMA_PKT_TIMESTAMP_SET_HEADER_sub_op_mask',
    'SDMA_PKT_TIMESTAMP_SET_HEADER_sub_op_offset',
    'SDMA_PKT_TIMESTAMP_SET_HEADER_sub_op_shift',
    'SDMA_PKT_TIMESTAMP_SET_INIT_DATA_HI_init_data_63_32_mask',
    'SDMA_PKT_TIMESTAMP_SET_INIT_DATA_HI_init_data_63_32_offset',
    'SDMA_PKT_TIMESTAMP_SET_INIT_DATA_HI_init_data_63_32_shift',
    'SDMA_PKT_TIMESTAMP_SET_INIT_DATA_LO_init_data_31_0_mask',
    'SDMA_PKT_TIMESTAMP_SET_INIT_DATA_LO_init_data_31_0_offset',
    'SDMA_PKT_TIMESTAMP_SET_INIT_DATA_LO_init_data_31_0_shift',
    'SDMA_PKT_TRAP', 'SDMA_PKT_TRAP_HEADER_op_mask',
    'SDMA_PKT_TRAP_HEADER_op_offset', 'SDMA_PKT_TRAP_HEADER_op_shift',
    'SDMA_PKT_TRAP_HEADER_sub_op_mask',
    'SDMA_PKT_TRAP_HEADER_sub_op_offset',
    'SDMA_PKT_TRAP_HEADER_sub_op_shift',
    'SDMA_PKT_TRAP_INT_CONTEXT_int_context_mask',
    'SDMA_PKT_TRAP_INT_CONTEXT_int_context_offset',
    'SDMA_PKT_TRAP_INT_CONTEXT_int_context_shift',
    'SDMA_PKT_VM_INVALIDATION_ADDRESSRANGEHI_addressrangehi_mask',
    'SDMA_PKT_VM_INVALIDATION_ADDRESSRANGEHI_addressrangehi_offset',
    'SDMA_PKT_VM_INVALIDATION_ADDRESSRANGEHI_addressrangehi_shift',
    'SDMA_PKT_VM_INVALIDATION_ADDRESSRANGEHI_invalidateack_mask',
    'SDMA_PKT_VM_INVALIDATION_ADDRESSRANGEHI_invalidateack_offset',
    'SDMA_PKT_VM_INVALIDATION_ADDRESSRANGEHI_invalidateack_shift',
    'SDMA_PKT_VM_INVALIDATION_ADDRESSRANGEHI_reserved_mask',
    'SDMA_PKT_VM_INVALIDATION_ADDRESSRANGEHI_reserved_offset',
    'SDMA_PKT_VM_INVALIDATION_ADDRESSRANGEHI_reserved_shift',
    'SDMA_PKT_VM_INVALIDATION_ADDRESSRANGELO_addressrangelo_mask',
    'SDMA_PKT_VM_INVALIDATION_ADDRESSRANGELO_addressrangelo_offset',
    'SDMA_PKT_VM_INVALIDATION_ADDRESSRANGELO_addressrangelo_shift',
    'SDMA_PKT_VM_INVALIDATION_HEADER_gfx_eng_id_mask',
    'SDMA_PKT_VM_INVALIDATION_HEADER_gfx_eng_id_offset',
    'SDMA_PKT_VM_INVALIDATION_HEADER_gfx_eng_id_shift',
    'SDMA_PKT_VM_INVALIDATION_HEADER_mm_eng_id_mask',
    'SDMA_PKT_VM_INVALIDATION_HEADER_mm_eng_id_offset',
    'SDMA_PKT_VM_INVALIDATION_HEADER_mm_eng_id_shift',
    'SDMA_PKT_VM_INVALIDATION_HEADER_op_mask',
    'SDMA_PKT_VM_INVALIDATION_HEADER_op_offset',
    'SDMA_PKT_VM_INVALIDATION_HEADER_op_shift',
    'SDMA_PKT_VM_INVALIDATION_HEADER_sub_op_mask',
    'SDMA_PKT_VM_INVALIDATION_HEADER_sub_op_offset',
    'SDMA_PKT_VM_INVALIDATION_HEADER_sub_op_shift',
    'SDMA_PKT_VM_INVALIDATION_INVALIDATEREQ_invalidatereq_mask',
    'SDMA_PKT_VM_INVALIDATION_INVALIDATEREQ_invalidatereq_offset',
    'SDMA_PKT_VM_INVALIDATION_INVALIDATEREQ_invalidatereq_shift',
    'SDMA_PKT_WRITE_INCR_COUNT_count_mask',
    'SDMA_PKT_WRITE_INCR_COUNT_count_offset',
    'SDMA_PKT_WRITE_INCR_COUNT_count_shift',
    'SDMA_PKT_WRITE_INCR_DST_ADDR_HI_dst_addr_63_32_mask',
    'SDMA_PKT_WRITE_INCR_DST_ADDR_HI_dst_addr_63_32_offset',
    'SDMA_PKT_WRITE_INCR_DST_ADDR_HI_dst_addr_63_32_shift',
    'SDMA_PKT_WRITE_INCR_DST_ADDR_LO_dst_addr_31_0_mask',
    'SDMA_PKT_WRITE_INCR_DST_ADDR_LO_dst_addr_31_0_offset',
    'SDMA_PKT_WRITE_INCR_DST_ADDR_LO_dst_addr_31_0_shift',
    'SDMA_PKT_WRITE_INCR_HEADER_op_mask',
    'SDMA_PKT_WRITE_INCR_HEADER_op_offset',
    'SDMA_PKT_WRITE_INCR_HEADER_op_shift',
    'SDMA_PKT_WRITE_INCR_HEADER_sub_op_mask',
    'SDMA_PKT_WRITE_INCR_HEADER_sub_op_offset',
    'SDMA_PKT_WRITE_INCR_HEADER_sub_op_shift',
    'SDMA_PKT_WRITE_INCR_INCR_DW0_incr_dw0_mask',
    'SDMA_PKT_WRITE_INCR_INCR_DW0_incr_dw0_offset',
    'SDMA_PKT_WRITE_INCR_INCR_DW0_incr_dw0_shift',
    'SDMA_PKT_WRITE_INCR_INCR_DW1_incr_dw1_mask',
    'SDMA_PKT_WRITE_INCR_INCR_DW1_incr_dw1_offset',
    'SDMA_PKT_WRITE_INCR_INCR_DW1_incr_dw1_shift',
    'SDMA_PKT_WRITE_INCR_INIT_DW0_init_dw0_mask',
    'SDMA_PKT_WRITE_INCR_INIT_DW0_init_dw0_offset',
    'SDMA_PKT_WRITE_INCR_INIT_DW0_init_dw0_shift',
    'SDMA_PKT_WRITE_INCR_INIT_DW1_init_dw1_mask',
    'SDMA_PKT_WRITE_INCR_INIT_DW1_init_dw1_offset',
    'SDMA_PKT_WRITE_INCR_INIT_DW1_init_dw1_shift',
    'SDMA_PKT_WRITE_INCR_MASK_DW0_mask_dw0_mask',
    'SDMA_PKT_WRITE_INCR_MASK_DW0_mask_dw0_offset',
    'SDMA_PKT_WRITE_INCR_MASK_DW0_mask_dw0_shift',
    'SDMA_PKT_WRITE_INCR_MASK_DW1_mask_dw1_mask',
    'SDMA_PKT_WRITE_INCR_MASK_DW1_mask_dw1_offset',
    'SDMA_PKT_WRITE_INCR_MASK_DW1_mask_dw1_shift',
    'SDMA_PKT_WRITE_TILED_BC_COUNT_count_mask',
    'SDMA_PKT_WRITE_TILED_BC_COUNT_count_offset',
    'SDMA_PKT_WRITE_TILED_BC_COUNT_count_shift',
    'SDMA_PKT_WRITE_TILED_BC_DATA0_data0_mask',
    'SDMA_PKT_WRITE_TILED_BC_DATA0_data0_offset',
    'SDMA_PKT_WRITE_TILED_BC_DATA0_data0_shift',
    'SDMA_PKT_WRITE_TILED_BC_DST_ADDR_HI_dst_addr_63_32_mask',
    'SDMA_PKT_WRITE_TILED_BC_DST_ADDR_HI_dst_addr_63_32_offset',
    'SDMA_PKT_WRITE_TILED_BC_DST_ADDR_HI_dst_addr_63_32_shift',
    'SDMA_PKT_WRITE_TILED_BC_DST_ADDR_LO_dst_addr_31_0_mask',
    'SDMA_PKT_WRITE_TILED_BC_DST_ADDR_LO_dst_addr_31_0_offset',
    'SDMA_PKT_WRITE_TILED_BC_DST_ADDR_LO_dst_addr_31_0_shift',
    'SDMA_PKT_WRITE_TILED_BC_DW_3_width_mask',
    'SDMA_PKT_WRITE_TILED_BC_DW_3_width_offset',
    'SDMA_PKT_WRITE_TILED_BC_DW_3_width_shift',
    'SDMA_PKT_WRITE_TILED_BC_DW_4_depth_mask',
    'SDMA_PKT_WRITE_TILED_BC_DW_4_depth_offset',
    'SDMA_PKT_WRITE_TILED_BC_DW_4_depth_shift',
    'SDMA_PKT_WRITE_TILED_BC_DW_4_height_mask',
    'SDMA_PKT_WRITE_TILED_BC_DW_4_height_offset',
    'SDMA_PKT_WRITE_TILED_BC_DW_4_height_shift',
    'SDMA_PKT_WRITE_TILED_BC_DW_5_array_mode_mask',
    'SDMA_PKT_WRITE_TILED_BC_DW_5_array_mode_offset',
    'SDMA_PKT_WRITE_TILED_BC_DW_5_array_mode_shift',
    'SDMA_PKT_WRITE_TILED_BC_DW_5_bank_h_mask',
    'SDMA_PKT_WRITE_TILED_BC_DW_5_bank_h_offset',
    'SDMA_PKT_WRITE_TILED_BC_DW_5_bank_h_shift',
    'SDMA_PKT_WRITE_TILED_BC_DW_5_bank_w_mask',
    'SDMA_PKT_WRITE_TILED_BC_DW_5_bank_w_offset',
    'SDMA_PKT_WRITE_TILED_BC_DW_5_bank_w_shift',
    'SDMA_PKT_WRITE_TILED_BC_DW_5_element_size_mask',
    'SDMA_PKT_WRITE_TILED_BC_DW_5_element_size_offset',
    'SDMA_PKT_WRITE_TILED_BC_DW_5_element_size_shift',
    'SDMA_PKT_WRITE_TILED_BC_DW_5_mat_aspt_mask',
    'SDMA_PKT_WRITE_TILED_BC_DW_5_mat_aspt_offset',
    'SDMA_PKT_WRITE_TILED_BC_DW_5_mat_aspt_shift',
    'SDMA_PKT_WRITE_TILED_BC_DW_5_mit_mode_mask',
    'SDMA_PKT_WRITE_TILED_BC_DW_5_mit_mode_offset',
    'SDMA_PKT_WRITE_TILED_BC_DW_5_mit_mode_shift',
    'SDMA_PKT_WRITE_TILED_BC_DW_5_num_bank_mask',
    'SDMA_PKT_WRITE_TILED_BC_DW_5_num_bank_offset',
    'SDMA_PKT_WRITE_TILED_BC_DW_5_num_bank_shift',
    'SDMA_PKT_WRITE_TILED_BC_DW_5_pipe_config_mask',
    'SDMA_PKT_WRITE_TILED_BC_DW_5_pipe_config_offset',
    'SDMA_PKT_WRITE_TILED_BC_DW_5_pipe_config_shift',
    'SDMA_PKT_WRITE_TILED_BC_DW_5_tilesplit_size_mask',
    'SDMA_PKT_WRITE_TILED_BC_DW_5_tilesplit_size_offset',
    'SDMA_PKT_WRITE_TILED_BC_DW_5_tilesplit_size_shift',
    'SDMA_PKT_WRITE_TILED_BC_DW_6_x_mask',
    'SDMA_PKT_WRITE_TILED_BC_DW_6_x_offset',
    'SDMA_PKT_WRITE_TILED_BC_DW_6_x_shift',
    'SDMA_PKT_WRITE_TILED_BC_DW_6_y_mask',
    'SDMA_PKT_WRITE_TILED_BC_DW_6_y_offset',
    'SDMA_PKT_WRITE_TILED_BC_DW_6_y_shift',
    'SDMA_PKT_WRITE_TILED_BC_DW_7_sw_mask',
    'SDMA_PKT_WRITE_TILED_BC_DW_7_sw_offset',
    'SDMA_PKT_WRITE_TILED_BC_DW_7_sw_shift',
    'SDMA_PKT_WRITE_TILED_BC_DW_7_z_mask',
    'SDMA_PKT_WRITE_TILED_BC_DW_7_z_offset',
    'SDMA_PKT_WRITE_TILED_BC_DW_7_z_shift',
    'SDMA_PKT_WRITE_TILED_BC_HEADER_op_mask',
    'SDMA_PKT_WRITE_TILED_BC_HEADER_op_offset',
    'SDMA_PKT_WRITE_TILED_BC_HEADER_op_shift',
    'SDMA_PKT_WRITE_TILED_BC_HEADER_sub_op_mask',
    'SDMA_PKT_WRITE_TILED_BC_HEADER_sub_op_offset',
    'SDMA_PKT_WRITE_TILED_BC_HEADER_sub_op_shift',
    'SDMA_PKT_WRITE_TILED_COUNT_count_mask',
    'SDMA_PKT_WRITE_TILED_COUNT_count_offset',
    'SDMA_PKT_WRITE_TILED_COUNT_count_shift',
    'SDMA_PKT_WRITE_TILED_DATA0_data0_mask',
    'SDMA_PKT_WRITE_TILED_DATA0_data0_offset',
    'SDMA_PKT_WRITE_TILED_DATA0_data0_shift',
    'SDMA_PKT_WRITE_TILED_DST_ADDR_HI_dst_addr_63_32_mask',
    'SDMA_PKT_WRITE_TILED_DST_ADDR_HI_dst_addr_63_32_offset',
    'SDMA_PKT_WRITE_TILED_DST_ADDR_HI_dst_addr_63_32_shift',
    'SDMA_PKT_WRITE_TILED_DST_ADDR_LO_dst_addr_31_0_mask',
    'SDMA_PKT_WRITE_TILED_DST_ADDR_LO_dst_addr_31_0_offset',
    'SDMA_PKT_WRITE_TILED_DST_ADDR_LO_dst_addr_31_0_shift',
    'SDMA_PKT_WRITE_TILED_DW_3_width_mask',
    'SDMA_PKT_WRITE_TILED_DW_3_width_offset',
    'SDMA_PKT_WRITE_TILED_DW_3_width_shift',
    'SDMA_PKT_WRITE_TILED_DW_4_depth_mask',
    'SDMA_PKT_WRITE_TILED_DW_4_depth_offset',
    'SDMA_PKT_WRITE_TILED_DW_4_depth_shift',
    'SDMA_PKT_WRITE_TILED_DW_4_height_mask',
    'SDMA_PKT_WRITE_TILED_DW_4_height_offset',
    'SDMA_PKT_WRITE_TILED_DW_4_height_shift',
    'SDMA_PKT_WRITE_TILED_DW_5_dimension_mask',
    'SDMA_PKT_WRITE_TILED_DW_5_dimension_offset',
    'SDMA_PKT_WRITE_TILED_DW_5_dimension_shift',
    'SDMA_PKT_WRITE_TILED_DW_5_element_size_mask',
    'SDMA_PKT_WRITE_TILED_DW_5_element_size_offset',
    'SDMA_PKT_WRITE_TILED_DW_5_element_size_shift',
    'SDMA_PKT_WRITE_TILED_DW_5_mip_max_mask',
    'SDMA_PKT_WRITE_TILED_DW_5_mip_max_offset',
    'SDMA_PKT_WRITE_TILED_DW_5_mip_max_shift',
    'SDMA_PKT_WRITE_TILED_DW_5_swizzle_mode_mask',
    'SDMA_PKT_WRITE_TILED_DW_5_swizzle_mode_offset',
    'SDMA_PKT_WRITE_TILED_DW_5_swizzle_mode_shift',
    'SDMA_PKT_WRITE_TILED_DW_6_x_mask',
    'SDMA_PKT_WRITE_TILED_DW_6_x_offset',
    'SDMA_PKT_WRITE_TILED_DW_6_x_shift',
    'SDMA_PKT_WRITE_TILED_DW_6_y_mask',
    'SDMA_PKT_WRITE_TILED_DW_6_y_offset',
    'SDMA_PKT_WRITE_TILED_DW_6_y_shift',
    'SDMA_PKT_WRITE_TILED_DW_7_sw_mask',
    'SDMA_PKT_WRITE_TILED_DW_7_sw_offset',
    'SDMA_PKT_WRITE_TILED_DW_7_sw_shift',
    'SDMA_PKT_WRITE_TILED_DW_7_z_mask',
    'SDMA_PKT_WRITE_TILED_DW_7_z_offset',
    'SDMA_PKT_WRITE_TILED_DW_7_z_shift',
    'SDMA_PKT_WRITE_TILED_HEADER_encrypt_mask',
    'SDMA_PKT_WRITE_TILED_HEADER_encrypt_offset',
    'SDMA_PKT_WRITE_TILED_HEADER_encrypt_shift',
    'SDMA_PKT_WRITE_TILED_HEADER_op_mask',
    'SDMA_PKT_WRITE_TILED_HEADER_op_offset',
    'SDMA_PKT_WRITE_TILED_HEADER_op_shift',
    'SDMA_PKT_WRITE_TILED_HEADER_sub_op_mask',
    'SDMA_PKT_WRITE_TILED_HEADER_sub_op_offset',
    'SDMA_PKT_WRITE_TILED_HEADER_sub_op_shift',
    'SDMA_PKT_WRITE_TILED_HEADER_tmz_mask',
    'SDMA_PKT_WRITE_TILED_HEADER_tmz_offset',
    'SDMA_PKT_WRITE_TILED_HEADER_tmz_shift',
    'SDMA_PKT_WRITE_UNTILED_DATA0_data0_mask',
    'SDMA_PKT_WRITE_UNTILED_DATA0_data0_offset',
    'SDMA_PKT_WRITE_UNTILED_DATA0_data0_shift',
    'SDMA_PKT_WRITE_UNTILED_DST_ADDR_HI_dst_addr_63_32_mask',
    'SDMA_PKT_WRITE_UNTILED_DST_ADDR_HI_dst_addr_63_32_offset',
    'SDMA_PKT_WRITE_UNTILED_DST_ADDR_HI_dst_addr_63_32_shift',
    'SDMA_PKT_WRITE_UNTILED_DST_ADDR_LO_dst_addr_31_0_mask',
    'SDMA_PKT_WRITE_UNTILED_DST_ADDR_LO_dst_addr_31_0_offset',
    'SDMA_PKT_WRITE_UNTILED_DST_ADDR_LO_dst_addr_31_0_shift',
    'SDMA_PKT_WRITE_UNTILED_DW_3_count_mask',
    'SDMA_PKT_WRITE_UNTILED_DW_3_count_offset',
    'SDMA_PKT_WRITE_UNTILED_DW_3_count_shift',
    'SDMA_PKT_WRITE_UNTILED_DW_3_sw_mask',
    'SDMA_PKT_WRITE_UNTILED_DW_3_sw_offset',
    'SDMA_PKT_WRITE_UNTILED_DW_3_sw_shift',
    'SDMA_PKT_WRITE_UNTILED_HEADER_encrypt_mask',
    'SDMA_PKT_WRITE_UNTILED_HEADER_encrypt_offset',
    'SDMA_PKT_WRITE_UNTILED_HEADER_encrypt_shift',
    'SDMA_PKT_WRITE_UNTILED_HEADER_op_mask',
    'SDMA_PKT_WRITE_UNTILED_HEADER_op_offset',
    'SDMA_PKT_WRITE_UNTILED_HEADER_op_shift',
    'SDMA_PKT_WRITE_UNTILED_HEADER_sub_op_mask',
    'SDMA_PKT_WRITE_UNTILED_HEADER_sub_op_offset',
    'SDMA_PKT_WRITE_UNTILED_HEADER_sub_op_shift',
    'SDMA_PKT_WRITE_UNTILED_HEADER_tmz_mask',
    'SDMA_PKT_WRITE_UNTILED_HEADER_tmz_offset',
    'SDMA_PKT_WRITE_UNTILED_HEADER_tmz_shift',
    'SDMA_SUBOP_COPY_DIRTY_PAGE', 'SDMA_SUBOP_COPY_LINEAR',
    'SDMA_SUBOP_COPY_LINEAR_BC', 'SDMA_SUBOP_COPY_LINEAR_PHY',
    'SDMA_SUBOP_COPY_LINEAR_RECT', 'SDMA_SUBOP_COPY_LINEAR_SUB_WIND',
    'SDMA_SUBOP_COPY_LINEAR_SUB_WIND_BC', 'SDMA_SUBOP_COPY_SOA',
    'SDMA_SUBOP_COPY_T2T_SUB_WIND', 'SDMA_SUBOP_COPY_T2T_SUB_WIND_BC',
    'SDMA_SUBOP_COPY_TILED', 'SDMA_SUBOP_COPY_TILED_BC',
    'SDMA_SUBOP_COPY_TILED_SUB_WIND',
    'SDMA_SUBOP_COPY_TILED_SUB_WIND_BC', 'SDMA_SUBOP_DATA_FILL_MULTI',
    'SDMA_SUBOP_POLL_DBIT_WRITE_MEM', 'SDMA_SUBOP_POLL_MEM_VERIFY',
    'SDMA_SUBOP_POLL_REG_WRITE_MEM', 'SDMA_SUBOP_PTEPDE_COPY',
    'SDMA_SUBOP_PTEPDE_COPY_BACKWARDS', 'SDMA_SUBOP_PTEPDE_GEN',
    'SDMA_SUBOP_PTEPDE_RMW', 'SDMA_SUBOP_TIMESTAMP_GET',
    'SDMA_SUBOP_TIMESTAMP_GET_GLOBAL', 'SDMA_SUBOP_TIMESTAMP_SET',
    'SDMA_SUBOP_USER_GCR', 'SDMA_SUBOP_VM_INVALIDATION',
    'SDMA_SUBOP_WRITE_LINEAR', 'SDMA_SUBOP_WRITE_TILED',
    'SDMA_SUBOP_WRITE_TILED_BC', '__NAVI10_SDMA_PKT_OPEN_H_',
    'hdp_flush_cmd', 'struct_SDMA_PKT_ATOMIC_TAG',
    'struct_SDMA_PKT_ATOMIC_TAG_0_0',
    'struct_SDMA_PKT_ATOMIC_TAG_1_0',
    'struct_SDMA_PKT_ATOMIC_TAG_2_0',
    'struct_SDMA_PKT_ATOMIC_TAG_3_0',
    'struct_SDMA_PKT_ATOMIC_TAG_4_0',
    'struct_SDMA_PKT_ATOMIC_TAG_5_0',
    'struct_SDMA_PKT_ATOMIC_TAG_6_0',
    'struct_SDMA_PKT_ATOMIC_TAG_7_0',
    'struct_SDMA_PKT_CONSTANT_FILL_TAG',
    'struct_SDMA_PKT_CONSTANT_FILL_TAG_0_0',
    'struct_SDMA_PKT_CONSTANT_FILL_TAG_1_0',
    'struct_SDMA_PKT_CONSTANT_FILL_TAG_2_0',
    'struct_SDMA_PKT_CONSTANT_FILL_TAG_3_0',
    'struct_SDMA_PKT_CONSTANT_FILL_TAG_4_0',
    'struct_SDMA_PKT_COPY_LINEAR_RECT_TAG',
    'struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_0_0',
    'struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_10_0',
    'struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_11_0',
    'struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_12_0',
    'struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_1_0',
    'struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_2_0',
    'struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_3_0',
    'struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_4_0',
    'struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_5_0',
    'struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_6_0',
    'struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_7_0',
    'struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_8_0',
    'struct_SDMA_PKT_COPY_LINEAR_RECT_TAG_9_0',
    'struct_SDMA_PKT_COPY_LINEAR_TAG',
    'struct_SDMA_PKT_COPY_LINEAR_TAG_0_0',
    'struct_SDMA_PKT_COPY_LINEAR_TAG_1_0',
    'struct_SDMA_PKT_COPY_LINEAR_TAG_2_0',
    'struct_SDMA_PKT_COPY_LINEAR_TAG_3_0',
    'struct_SDMA_PKT_COPY_LINEAR_TAG_4_0',
    'struct_SDMA_PKT_COPY_LINEAR_TAG_5_0',
    'struct_SDMA_PKT_COPY_LINEAR_TAG_6_0',
    'struct_SDMA_PKT_FENCE_TAG', 'struct_SDMA_PKT_FENCE_TAG_0_0',
    'struct_SDMA_PKT_FENCE_TAG_1_0', 'struct_SDMA_PKT_FENCE_TAG_2_0',
    'struct_SDMA_PKT_FENCE_TAG_3_0', 'struct_SDMA_PKT_GCR_TAG',
    'struct_SDMA_PKT_GCR_TAG_0_0', 'struct_SDMA_PKT_GCR_TAG_1_0',
    'struct_SDMA_PKT_GCR_TAG_2_0', 'struct_SDMA_PKT_GCR_TAG_3_0',
    'struct_SDMA_PKT_GCR_TAG_4_0', 'struct_SDMA_PKT_HDP_FLUSH_TAG',
    'struct_SDMA_PKT_POLL_REGMEM_TAG',
    'struct_SDMA_PKT_POLL_REGMEM_TAG_0_0',
    'struct_SDMA_PKT_POLL_REGMEM_TAG_1_0',
    'struct_SDMA_PKT_POLL_REGMEM_TAG_2_0',
    'struct_SDMA_PKT_POLL_REGMEM_TAG_3_0',
    'struct_SDMA_PKT_POLL_REGMEM_TAG_4_0',
    'struct_SDMA_PKT_POLL_REGMEM_TAG_5_0',
    'struct_SDMA_PKT_TIMESTAMP_TAG',
    'struct_SDMA_PKT_TIMESTAMP_TAG_0_0',
    'struct_SDMA_PKT_TIMESTAMP_TAG_1_0',
    'struct_SDMA_PKT_TIMESTAMP_TAG_2_0', 'struct_SDMA_PKT_TRAP_TAG',
    'struct_SDMA_PKT_TRAP_TAG_0_0', 'struct_SDMA_PKT_TRAP_TAG_1_0',
    'union_SDMA_PKT_ATOMIC_TAG_ADDR_HI_UNION',
    'union_SDMA_PKT_ATOMIC_TAG_ADDR_LO_UNION',
    'union_SDMA_PKT_ATOMIC_TAG_CMP_DATA_HI_UNION',
    'union_SDMA_PKT_ATOMIC_TAG_CMP_DATA_LO_UNION',
    'union_SDMA_PKT_ATOMIC_TAG_HEADER_UNION',
    'union_SDMA_PKT_ATOMIC_TAG_LOOP_UNION',
    'union_SDMA_PKT_ATOMIC_TAG_SRC_DATA_HI_UNION',
    'union_SDMA_PKT_ATOMIC_TAG_SRC_DATA_LO_UNION',
    'union_SDMA_PKT_CONSTANT_FILL_TAG_COUNT_UNION',
    'union_SDMA_PKT_CONSTANT_FILL_TAG_DATA_UNION',
    'union_SDMA_PKT_CONSTANT_FILL_TAG_DST_ADDR_HI_UNION',
    'union_SDMA_PKT_CONSTANT_FILL_TAG_DST_ADDR_LO_UNION',
    'union_SDMA_PKT_CONSTANT_FILL_TAG_HEADER_UNION',
    'union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_ADDR_HI_UNION',
    'union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_ADDR_LO_UNION',
    'union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_1_UNION',
    'union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_2_UNION',
    'union_SDMA_PKT_COPY_LINEAR_RECT_TAG_DST_PARAMETER_3_UNION',
    'union_SDMA_PKT_COPY_LINEAR_RECT_TAG_HEADER_UNION',
    'union_SDMA_PKT_COPY_LINEAR_RECT_TAG_RECT_PARAMETER_1_UNION',
    'union_SDMA_PKT_COPY_LINEAR_RECT_TAG_RECT_PARAMETER_2_UNION',
    'union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_ADDR_HI_UNION',
    'union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_ADDR_LO_UNION',
    'union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_1_UNION',
    'union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_2_UNION',
    'union_SDMA_PKT_COPY_LINEAR_RECT_TAG_SRC_PARAMETER_3_UNION',
    'union_SDMA_PKT_COPY_LINEAR_TAG_COUNT_UNION',
    'union_SDMA_PKT_COPY_LINEAR_TAG_DST_ADDR_HI_UNION',
    'union_SDMA_PKT_COPY_LINEAR_TAG_DST_ADDR_LO_UNION',
    'union_SDMA_PKT_COPY_LINEAR_TAG_HEADER_UNION',
    'union_SDMA_PKT_COPY_LINEAR_TAG_PARAMETER_UNION',
    'union_SDMA_PKT_COPY_LINEAR_TAG_SRC_ADDR_HI_UNION',
    'union_SDMA_PKT_COPY_LINEAR_TAG_SRC_ADDR_LO_UNION',
    'union_SDMA_PKT_FENCE_TAG_ADDR_HI_UNION',
    'union_SDMA_PKT_FENCE_TAG_ADDR_LO_UNION',
    'union_SDMA_PKT_FENCE_TAG_DATA_UNION',
    'union_SDMA_PKT_FENCE_TAG_HEADER_UNION',
    'union_SDMA_PKT_GCR_TAG_HEADER_UNION',
    'union_SDMA_PKT_GCR_TAG_WORD1_UNION',
    'union_SDMA_PKT_GCR_TAG_WORD2_UNION',
    'union_SDMA_PKT_GCR_TAG_WORD3_UNION',
    'union_SDMA_PKT_GCR_TAG_WORD4_UNION',
    'union_SDMA_PKT_POLL_REGMEM_TAG_ADDR_HI_UNION',
    'union_SDMA_PKT_POLL_REGMEM_TAG_ADDR_LO_UNION',
    'union_SDMA_PKT_POLL_REGMEM_TAG_DW5_UNION',
    'union_SDMA_PKT_POLL_REGMEM_TAG_HEADER_UNION',
    'union_SDMA_PKT_POLL_REGMEM_TAG_MASK_UNION',
    'union_SDMA_PKT_POLL_REGMEM_TAG_VALUE_UNION',
    'union_SDMA_PKT_TIMESTAMP_TAG_ADDR_HI_UNION',
    'union_SDMA_PKT_TIMESTAMP_TAG_ADDR_LO_UNION',
    'union_SDMA_PKT_TIMESTAMP_TAG_HEADER_UNION',
    'union_SDMA_PKT_TRAP_TAG_HEADER_UNION',
    'union_SDMA_PKT_TRAP_TAG_INT_CONTEXT_UNION']
